import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import implicit
from implicit.evaluation import precision_at_k, train_test_split as als_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
import xgboost as xgb

RANDOM_STATE = 42
ALS_FACTORS = 32
ALS_REGULARIZATION = 0.05
ALS_ITERATIONS = 20
ALS_ALPHA = 15.0
TOP_K = 5

PLAN_LADDER = {"free": 0, "digital": 1, "premium": 2, "print": 3}

mlflow.set_experiment("/Shared/pclv/06_next_best_offer")

pv = spark.table("pclv.silver.page_views").toPandas()
state = spark.table("pclv.silver.subscriber_current_state").toPandas()
sub_events = spark.table("pclv.silver.subscriptions").toPandas()

pv["date"] = pd.to_datetime(pv["date"])
REFERENCE_DATE = pv["date"].max()

pv = pv[~pv["is_post_cancel_view"]]
active = state[state["is_active"]].copy()

interactions = (
    pv[pv["user_id"].isin(active["user_id"])]
    .groupby(["user_id", "section"])["read_time_seconds"].sum()
    .reset_index(name="total_read_time")
)
interactions["weight"] = np.log1p(interactions["total_read_time"]).astype(np.float32)

user_ids = interactions["user_id"].unique()
sections = interactions["section"].unique()
u2idx = {u: i for i, u in enumerate(user_ids)}
s2idx = {s: i for i, s in enumerate(sections)}
idx2s = {i: s for s, i in s2idx.items()}

rows = interactions["user_id"].map(u2idx).values
cols = interactions["section"].map(s2idx).values
data = interactions["weight"].values
mat = csr_matrix((data, (rows, cols)), shape=(len(user_ids), len(sections)))

with mlflow.start_run(run_name="als_section_recommender"):
    mlflow.log_params({
        "factors": ALS_FACTORS,
        "regularization": ALS_REGULARIZATION,
        "iterations": ALS_ITERATIONS,
        "alpha": ALS_ALPHA,
        "n_users": len(user_ids),
        "n_items": len(sections),
        "nnz": mat.nnz,
    })

    train_mat, test_mat = als_split(mat, train_percentage=0.8, random_state=RANDOM_STATE)

    model = implicit.als.AlternatingLeastSquares(
        factors=ALS_FACTORS,
        regularization=ALS_REGULARIZATION,
        iterations=ALS_ITERATIONS,
        alpha=ALS_ALPHA,
        random_state=RANDOM_STATE,
    )
    model.fit(train_mat, show_progress=False)

    try:
        pk = precision_at_k(
            model, train_mat, test_mat,
            K=min(TOP_K, len(sections) - 1), show_progress=False,
        )
        mlflow.log_metric("precision_at_k", float(pk))
    except Exception:
        pass

    model_full = implicit.als.AlternatingLeastSquares(
        factors=ALS_FACTORS,
        regularization=ALS_REGULARIZATION,
        iterations=ALS_ITERATIONS,
        alpha=ALS_ALPHA,
        random_state=RANDOM_STATE,
    )
    model_full.fit(mat, show_progress=False)

recs = []
for uid, idx in u2idx.items():
    ids, scores = model_full.recommend(idx, mat[idx], N=1, filter_already_liked_items=True)
    if len(ids) > 0:
        recs.append({
            "user_id": uid,
            "recommended_content_section": idx2s[ids[0]],
            "als_score": float(scores[0]),
        })
als_recs = pd.DataFrame(recs)

upgrade_flag = (
    sub_events[sub_events["event_type"] == "upgrade"]
    .groupby("user_id").size().reset_index(name="did_upgrade")
)
upgrade_flag["did_upgrade"] = 1

view_agg = (
    pv.groupby("user_id").agg(
        total_views=("date", "count"),
        total_active_days=("date", "nunique"),
        mean_read_time=("read_time_seconds", "mean"),
        section_breadth=("section", "nunique"),
    ).reset_index()
)

feat_df = (
    active[[
        "user_id", "current_plan", "tenure_days",
        "acquisition_channel", "country", "signup_cohort",
    ]]
    .merge(view_agg, on="user_id", how="left")
    .merge(upgrade_flag, on="user_id", how="left")
    .fillna({
        "did_upgrade": 0, "total_views": 0, "total_active_days": 0,
        "mean_read_time": 0.0, "section_breadth": 0,
    })
)

X = pd.get_dummies(
    feat_df.drop(columns=["user_id", "did_upgrade"]),
    columns=["current_plan", "acquisition_channel", "country", "signup_cohort"],
    drop_first=True,
).fillna(0)
y = feat_df["did_upgrade"].astype(int).values

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

with mlflow.start_run(run_name="xgb_upgrade_propensity"):
    params = {
        "objective": "binary:logistic",
        "eval_metric": ["auc", "logloss"],
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "n_estimators": 400,
        "random_state": RANDOM_STATE,
        "scale_pos_weight": float((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1),
    }
    mlflow.log_params(params)

    clf = xgb.XGBClassifier(**params, early_stopping_rounds=30)
    clf.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)

    proba_te = clf.predict_proba(X_te)[:, 1]
    mlflow.log_metric("upgrade_auc", float(roc_auc_score(y_te, proba_te)))
    mlflow.log_metric("upgrade_pr_auc", float(average_precision_score(y_te, proba_te)))
    mlflow.xgboost.log_model(clf, name="xgb_upgrade")

upgrade_prob = clf.predict_proba(X)[:, 1]
prop_df = pd.DataFrame({
    "user_id": feat_df["user_id"].values,
    "upgrade_propensity": upgrade_prob,
    "current_plan": feat_df["current_plan"].values,
})


def decide_action(row):
    plan = row["current_plan"]
    upg_p = row["upgrade_propensity"]
    section = row["recommended_content_section"]

    if plan == "free":
        return "convert_to_digital", section
    if upg_p >= 0.4 and PLAN_LADDER.get(plan, 0) < 3:
        next_plan = {v: k for k, v in PLAN_LADDER.items()}[PLAN_LADDER[plan] + 1]
        return f"upgrade_to_{next_plan}", section
    if upg_p < 0.1:
        return "reengagement_newsletter", section
    return "cross_sell_folio_or_podcast", section


combined = prop_df.merge(als_recs, on="user_id", how="left")
combined["recommended_content_section"] = combined["recommended_content_section"].fillna("politik")
combined[["recommended_action", "recommended_content_section"]] = combined.apply(
    lambda r: pd.Series(decide_action(r)), axis=1
)

als_max = combined["als_score"].fillna(0).max()
combined["confidence_score"] = (
    combined["upgrade_propensity"] * 0.5
    + combined["als_score"].fillna(0) / max(als_max, 1e-9) * 0.5
)

out = combined[[
    "user_id", "current_plan", "recommended_action", "recommended_content_section",
    "upgrade_propensity", "als_score", "confidence_score",
]].copy()
out["scored_at"] = REFERENCE_DATE.date()

(
    spark.createDataFrame(out)
    .write.mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("pclv.gold.next_best_offer")
)
