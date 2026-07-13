import numpy as np
import pandas as pd

pv = spark.table("pclv.silver.page_views").toPandas()
pv["date"] = pd.to_datetime(pv["date"])
REFERENCE_DATE = pv["date"].max().date()

state = spark.table("pclv.silver.subscriber_current_state").toPandas()
state = state[state["is_active"]][[
    "user_id", "current_plan", "tenure_days",
    "acquisition_channel", "country", "signup_cohort",
]]

rfe = spark.table("pclv.gold.rfe_segments").toPandas()[[
    "user_id", "recency_days", "active_days_90d", "views_90d",
    "engagement_score", "r_score", "f_score", "e_score",
    "rfe_score", "segment_label", "kmeans_cluster",
]]

churn = spark.table("pclv.gold.churn_predictions").toPandas()[[
    "user_id", "cox_partial_hazard", "cox_survival_90d", "xgb_churn_prob_90d",
]]

value = spark.table("pclv.gold.subscriber_value").toPandas()[[
    "user_id", "arpu_monthly", "rmst_months", "expected_value_chf", "horizon_months",
]]

nbo = spark.table("pclv.gold.next_best_offer").toPandas()[[
    "user_id", "recommended_action", "recommended_content_section",
    "upgrade_propensity", "als_score", "confidence_score",
]].rename(columns={"confidence_score": "nbo_confidence"})

scores = (
    state
    .merge(rfe, on="user_id", how="left")
    .merge(churn, on="user_id", how="left")
    .merge(value, on="user_id", how="left")
    .merge(nbo, on="user_id", how="left")
)

scores["risk_adjusted_value_chf"] = (
    scores["expected_value_chf"] * scores["cox_survival_90d"].fillna(0.0)
)

scores["churn_risk_tier"] = np.where(
    scores["xgb_churn_prob_90d"] >= 0.5, "high",
    np.where(scores["xgb_churn_prob_90d"] >= 0.25, "medium", "low"),
)

scores["value_tier"] = np.where(
    scores["expected_value_chf"] >= 500, "high",
    np.where(scores["expected_value_chf"] >= 150, "medium", "low"),
)


def priority_tier(row):
    if row["value_tier"] == "high" and row["churn_risk_tier"] == "high":
        return "P1_save"
    if row["value_tier"] == "high":
        return "P2_grow"
    if row["value_tier"] == "medium" and row["churn_risk_tier"] == "high":
        return "P3_retain"
    if row["segment_label"] == "New / Reactivated":
        return "P4_onboard"
    return "P5_nurture"


scores["priority_tier"] = scores.apply(priority_tier, axis=1)
scores["scored_at"] = REFERENCE_DATE

(
    spark.createDataFrame(scores)
    .write.mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("pclv.gold.pclv_customer_scores")
)

n_rows = len(scores)
n_missing_value = scores["expected_value_chf"].isna().sum()
n_missing_churn = scores["xgb_churn_prob_90d"].isna().sum()
n_missing_nbo = scores["recommended_action"].isna().sum()

print(f"pclv_customer_scores rows            : {n_rows}")
print(f"  missing subscriber_value join      : {n_missing_value}")
print(f"  missing churn_predictions join     : {n_missing_churn}")
print(f"  missing next_best_offer join       : {n_missing_nbo}")

print("\nPriority tier distribution:")
print(scores["priority_tier"].value_counts().sort_index().to_string())

print("\nSegment x churn-risk crosstab:")
print(pd.crosstab(scores["segment_label"], scores["churn_risk_tier"]).to_string())
