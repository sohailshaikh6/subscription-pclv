import pickle
import mlflow
import numpy as np
import pandas as pd

HORIZON_MONTHS = 24
HORIZON_DAYS = HORIZON_MONTHS * 30

PLAN_PRICE_CHF = {
    "free": 0.0,
    "digital": 29.0,
    "premium": 49.0,
    "print": 65.0,
}

mlflow.set_experiment("/Shared/pclv/05_subscriber_value")

pv = spark.table("pclv.silver.page_views").toPandas()
pv["date"] = pd.to_datetime(pv["date"])
REFERENCE_DATE = pv["date"].max()

client = mlflow.tracking.MlflowClient()
churn_exp = client.get_experiment_by_name("/Shared/pclv/04_churn_survival")
runs = client.search_runs(
    [churn_exp.experiment_id],
    filter_string="tags.mlflow.runName = 'cox_ph'",
    order_by=["start_time DESC"],
    max_results=1,
)
cox_run = runs[0]
local_path = client.download_artifacts(cox_run.info.run_id, "cox_model.pkl")
with open(local_path, "rb") as f:
    cph = pickle.load(f)

state = spark.table("pclv.silver.subscriber_current_state").toPandas()

arpu = state[state["is_active"]][[
    "user_id", "current_plan", "acquisition_channel", "country",
    "signup_cohort", "tenure_days",
]].copy()
arpu["arpu_monthly"] = arpu["current_plan"].map(PLAN_PRICE_CHF).fillna(0.0)
arpu = arpu.reset_index(drop=True)

feat = pd.get_dummies(
    arpu.drop(columns=["user_id", "arpu_monthly"]),
    columns=["current_plan", "acquisition_channel", "country", "signup_cohort"],
    drop_first=True,
).select_dtypes(include=[np.number]).fillna(0)

cox_cov = list(cph.params_.index)
feat = feat.reindex(columns=cox_cov, fill_value=0)

time_grid = np.linspace(1, HORIZON_DAYS, 50)
surv = cph.predict_survival_function(feat, times=time_grid)


def rmst_from_curve(sf_col, tenure, horizon_days):
    idx = sf_col.index.values
    vals = sf_col.values
    s_at_tenure = float(np.interp(tenure, idx, vals))
    if s_at_tenure <= 1e-9:
        return 0.0
    future_days = np.clip(idx - tenure, 0, horizon_days)
    mask = future_days > 0
    cond_s = np.clip(vals[mask] / s_at_tenure, 0, 1)
    return float(np.trapz(cond_s, future_days[mask]))


rmst_days = np.array([
    rmst_from_curve(surv.iloc[:, i], arpu.iloc[i]["tenure_days"], HORIZON_DAYS)
    for i in range(len(arpu))
])
rmst_months = rmst_days / 30.0

with mlflow.start_run(run_name="subscriber_value_rmst"):
    mlflow.log_params({
        "horizon_months": HORIZON_MONTHS,
        "arpu_source": "current_plan_price",
        "cox_run_id": cox_run.info.run_id,
        "integration": "trapezoidal_50pt",
    })

    arpu["rmst_months"] = rmst_months
    arpu["expected_value_chf"] = arpu["arpu_monthly"].values * rmst_months

    mlflow.log_metric("mean_rmst_months", float(np.mean(rmst_months)))
    mlflow.log_metric("mean_expected_value_chf", float(np.mean(arpu["expected_value_chf"])))
    mlflow.log_metric("p50_expected_value_chf", float(np.median(arpu["expected_value_chf"])))
    mlflow.log_metric("p90_expected_value_chf", float(np.quantile(arpu["expected_value_chf"], 0.9)))

    out = arpu[["user_id", "current_plan", "arpu_monthly", "rmst_months", "expected_value_chf"]].copy()
    out["horizon_months"] = HORIZON_MONTHS
    out["scored_at"] = REFERENCE_DATE.date()

    (
        spark.createDataFrame(out)
        .write.mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable("pclv.gold.subscriber_value")
    )
