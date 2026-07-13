import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

TRAILING_DAYS = 90
N_CLUSTERS = 5
RANDOM_STATE = 42

mlflow.set_experiment("/Shared/pclv/03_rfe_segmentation")

state = spark.table("pclv.silver.subscriber_current_state").toPandas()
pv = spark.table("pclv.silver.page_views").toPandas()

pv["date"] = pd.to_datetime(pv["date"])
state["cancel_date"] = pd.to_datetime(state["cancel_date"])

REFERENCE_DATE = pv["date"].max()

active = state[state["is_active"]].copy()
pv = pv[pv["user_id"].isin(active["user_id"])]
pv = pv[pv["date"] >= REFERENCE_DATE - pd.Timedelta(days=TRAILING_DAYS)]

agg = pv.groupby("user_id").agg(
    last_view=("date", "max"),
    active_days_90d=("date", "nunique"),
    views_90d=("date", "count"),
    mean_read_time=("read_time_seconds", "mean"),
    section_breadth=("section", "nunique"),
).reset_index()

agg["recency_days"] = (REFERENCE_DATE - agg["last_view"]).dt.days
agg["engagement_score"] = (
    np.log1p(agg["views_90d"] / (TRAILING_DAYS / 7.0))
    * agg["section_breadth"]
    * np.log1p(agg["mean_read_time"])
)

rfe = active[["user_id"]].merge(agg, on="user_id", how="left")
rfe = rfe.fillna({
    "recency_days": TRAILING_DAYS + 1,
    "active_days_90d": 0,
    "views_90d": 0,
    "mean_read_time": 0.0,
    "section_breadth": 0,
    "engagement_score": 0.0,
}).drop(columns=["last_view"])


def safe_qcut(series, n_bins=5, reverse=False):
    ranks = series.rank(method="first")
    binned = pd.qcut(ranks, n_bins, labels=False, duplicates="drop") + 1
    return (n_bins + 1 - binned).astype(int) if reverse else binned.astype(int)


rfe["r_score"] = safe_qcut(rfe["recency_days"], reverse=True)
rfe["f_score"] = safe_qcut(rfe["active_days_90d"])
rfe["e_score"] = safe_qcut(rfe["engagement_score"])
rfe["rfe_score"] = rfe["r_score"] + rfe["f_score"] + rfe["e_score"]


def label_segment(row):
    r, f, e = row["r_score"], row["f_score"], row["e_score"]
    if r >= 4 and f >= 4 and e >= 4:
        return "Champions"
    if r >= 4 and f >= 3:
        return "Loyal"
    if r >= 4 and f <= 2:
        return "New / Reactivated"
    if r <= 2 and f >= 4 and e >= 3:
        return "At Risk"
    if r <= 2 and f >= 3:
        return "Cooling Off"
    if r <= 2 and f <= 2:
        return "Hibernating"
    return "Regular"


rfe["segment_label"] = rfe.apply(label_segment, axis=1)

with mlflow.start_run(run_name="rfe_segmentation"):
    mlflow.log_params({
        "reference_date": str(REFERENCE_DATE.date()),
        "trailing_days": TRAILING_DAYS,
        "n_clusters": N_CLUSTERS,
    })

    X = rfe[["recency_days", "active_days_90d", "engagement_score"]].values
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    km = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10).fit(X_scaled)
    rfe["kmeans_cluster"] = km.labels_.astype(int)

    sample_idx = np.random.default_rng(RANDOM_STATE).choice(
        len(X_scaled), size=min(10_000, len(X_scaled)), replace=False
    )
    sil = silhouette_score(X_scaled[sample_idx], km.labels_[sample_idx])

    mlflow.log_metric("silhouette_sampled", float(sil))
    mlflow.log_metric("inertia", float(km.inertia_))
    for label, count in rfe["segment_label"].value_counts().items():
        mlflow.log_metric(f"seg_{label.replace(' ', '_').replace('/', '_')}", int(count))

    mlflow.sklearn.log_model(km, name="kmeans")
    mlflow.sklearn.log_model(scaler, name="scaler")

out = rfe[[
    "user_id", "recency_days", "active_days_90d", "views_90d", "mean_read_time",
    "section_breadth", "engagement_score", "r_score", "f_score", "e_score",
    "rfe_score", "segment_label", "kmeans_cluster",
]].copy()
out["scored_at"] = REFERENCE_DATE.date()

(
    spark.createDataFrame(out)
    .write.mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("pclv.gold.rfe_segments")
)
