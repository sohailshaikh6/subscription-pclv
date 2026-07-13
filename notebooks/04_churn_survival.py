import pickle
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import xgboost as xgb

HORIZON_DAYS = 90
RANDOM_STATE = 42

mlflow.set_experiment("/Shared/pclv/04_churn_survival")

state = spark.table("pclv.silver.subscriber_current_state").toPandas()
pv = spark.table("pclv.silver.page_views").toPandas()

pv["date"] = pd.to_datetime(pv["date"])
state["cancel_date"] = pd.to_datetime(state["cancel_date"])
REFERENCE_DATE = pv["date"].max()

pv = pv[~pv["is_post_cancel_view"]]

obs_end = state[["user_id", "cancel_date"]].copy()
obs_end["obs_end"] = obs_end["cancel_date"].fillna(REFERENCE_DATE)

pv_w = pv.merge(obs_end[["user_id", "obs_end"]], on="user_id", how="inner")
pv_w = pv_w[(pv_w["date"] <= pv_w["obs_end"]) &
            (pv_w["date"] >= pv_w["obs_end"] - pd.Timedelta(days=60))]

eng = pv_w.groupby(["user_id", "obs_end"]).agg(
    views_60d=("date", "count"),
    active_days_60d=("date", "nunique"),
    mean_read_time_60d=("read_time_seconds", "mean"),
    section_breadth_60d=("section", "nunique"),
    last_view=("date", "max"),
).reset_index()
eng["recency_at_obs_end"] = (eng["obs_end"] - eng["last_view"]).dt.days
eng = eng.drop(columns=["last_view", "obs_end"])

df = state[[
    "user_id", "tenure_days", "is_censored", "current_plan",
    "acquisition_channel", "country", "signup_cohort",
]].copy()
df = df.rename(columns={"tenure_days": "duration"})
df["event"] = (~df["is_censored"]).astype(int)
df = df.drop(columns=["is_censored"])

df = df.merge(eng, on="user_id", how="left")

df = df.fillna({
    "views_60d": 0, "active_days_60d": 0, "mean_read_time_60d": 0.0,
    "section_breadth_60d": 0, "recency_at_obs_end": 60,
})

with mlflow.start_run(run_name="kaplan_meier_baseline"):
    fig, ax = plt.subplots(figsize=(9, 5))
    for plan in sorted(df["current_plan"].dropna().unique()):
        mask = df["current_plan"] == plan
        km = KaplanMeierFitter().fit(df.loc[mask, "duration"], df.loc[mask, "event"], label=plan)
        km.plot_survival_function(ax=ax, ci_show=False)
    ax.set_title("Subscriber survival by plan (Kaplan-Meier)")
    ax.set_xlabel("Tenure (days)")
    ax.set_ylabel("S(t)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig("/tmp/km_by_plan.png", dpi=120)
    mlflow.log_artifact("/tmp/km_by_plan.png")
    plt.close(fig)

    km_all = KaplanMeierFitter().fit(df["duration"], df["event"])
    mlflow.log_metric("median_survival_days", float(km_all.median_survival_time_))

cox_df = pd.get_dummies(
    df.drop(columns=["user_id"]),
    columns=["current_plan", "acquisition_channel", "country", "signup_cohort"],
    drop_first=True,
).astype({"event": "int"})
cox_df = cox_df.select_dtypes(include=[np.number]).fillna(0)

train_cox, test_cox = train_test_split(
    cox_df, test_size=0.2, random_state=RANDOM_STATE, stratify=cox_df["event"]
)

with mlflow.start_run(run_name="cox_ph") as cox_run:
    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(train_cox, duration_col="duration", event_col="event", show_progress=False)

    train_ci = concordance_index(
        train_cox["duration"], -cph.predict_partial_hazard(train_cox), train_cox["event"]
    )
    test_ci = concordance_index(
        test_cox["duration"], -cph.predict_partial_hazard(test_cox), test_cox["event"]
    )

    mlflow.log_params({"penalizer": 0.01, "n_features": cox_df.shape[1] - 2})
    mlflow.log_metric("c_index_train", float(train_ci))
    mlflow.log_metric("c_index_test", float(test_ci))

    summary = cph.summary[["coef", "exp(coef)", "p"]].sort_values("exp(coef)", ascending=False)
    summary.to_csv("/tmp/cox_summary.csv")
    mlflow.log_artifact("/tmp/cox_summary.csv")

    with open("/tmp/cox_model.pkl", "wb") as f:
        pickle.dump(cph, f)
    mlflow.log_artifact("/tmp/cox_model.pkl")

label_df = df.copy()
label_df["y_90d_churn"] = ((label_df["event"] == 1) & (label_df["duration"] <= 730)).astype(int)
label_df = label_df[label_df["duration"] >= 30].copy()

feat_cols = [
    "views_60d", "active_days_60d", "mean_read_time_60d",
    "section_breadth_60d", "recency_at_obs_end", "duration",
]
X = pd.get_dummies(
    label_df[feat_cols + ["current_plan", "acquisition_channel", "country"]],
    drop_first=True,
).fillna(0)
y = label_df["y_90d_churn"].values

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

with mlflow.start_run(run_name="xgb_90d_churn"):
    params = {
        "objective": "binary:logistic",
        "eval_metric": ["auc", "logloss"],
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "n_estimators": 400,
        "random_state": RANDOM_STATE,
        "scale_pos_weight": float((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1),
    }
    mlflow.log_params(params)

    clf = xgb.XGBClassifier(**params, early_stopping_rounds=30)
    clf.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)

    proba_te = clf.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, proba_te)
    ap = average_precision_score(y_te, proba_te)
    brier = brier_score_loss(y_te, proba_te)

    mlflow.log_metrics({
        "auc_test": float(auc),
        "pr_auc_test": float(ap),
        "brier_test": float(brier),
    })
    mlflow.xgboost.log_model(clf, name="xgb_churn_90d")

    imp = pd.DataFrame({"feature": X_tr.columns, "gain": clf.feature_importances_}) \
        .sort_values("gain", ascending=False)
    imp.to_csv("/tmp/xgb_importance.csv", index=False)
    mlflow.log_artifact("/tmp/xgb_importance.csv")

active_df = df[df["event"] == 0].copy()

cox_features_full = pd.get_dummies(
    df.drop(columns=["user_id"]),
    columns=["current_plan", "acquisition_channel", "country", "signup_cohort"],
    drop_first=True,
).select_dtypes(include=[np.number]).fillna(0)
cox_features_active = cox_features_full.loc[active_df.index]

cox_partial_hazard = cph.predict_partial_hazard(cox_features_active).values.ravel()

future_t = (active_df["duration"].values + HORIZON_DAYS)
survival_90 = np.array([
    float(cph.predict_survival_function(
        cox_features_active.iloc[[i]], times=[future_t[i]]
    ).values[0][0])
    for i in range(len(active_df))
])

X_all = pd.get_dummies(
    df.loc[active_df.index, feat_cols + ["current_plan", "acquisition_channel", "country"]],
    drop_first=True,
).reindex(columns=X.columns, fill_value=0)
xgb_prob = clf.predict_proba(X_all)[:, 1]

out_pdf = pd.DataFrame({
    "user_id": active_df["user_id"].values,
    "cox_partial_hazard": cox_partial_hazard,
    "cox_survival_90d": survival_90,
    "xgb_churn_prob_90d": xgb_prob,
})
out_pdf["scored_at"] = REFERENCE_DATE.date()

(
    spark.createDataFrame(out_pdf)
    .write.mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("pclv.gold.churn_predictions")
)
