import numpy as np
import pandas as pd

scores = spark.table("pclv.gold.pclv_customer_scores").toPandas()

subs = spark.table("pclv.silver.subscribers").toPandas()[["user_id", "email_hash"]]

mkt = scores.merge(subs, on="user_id", how="left")


def choose_channel(row):
    tier = row["priority_tier"]
    declining = row["is_declining"] == 1
    recency = row["recency_days"] if not pd.isna(row["recency_days"]) else 999

    if declining or recency > 30:
        if tier in ("P1_save", "P3_retain"):
            return "newsletter_and_social"
        if tier == "P2_grow":
            return "newsletter_and_social"
        return "social_ad"
    if tier in ("P1_save", "P2_grow", "P3_retain", "P4_onboard"):
        return "newsletter"
    return "social_ad"


def campaign_segment(row):
    tier = row["priority_tier"]
    plan = row["current_plan"]
    action = row["recommended_action"] if not pd.isna(row["recommended_action"]) else "nurture"

    if tier == "P1_save":
        return f"save_{plan}"
    if tier == "P2_grow":
        return f"grow_{plan}_{action}"
    if tier == "P3_retain":
        return f"retain_{plan}"
    if tier == "P4_onboard":
        return f"onboard_{plan}"
    return f"nurture_{plan}"


def creative_angle(row):
    section = row["recommended_content_section"]
    tier = row["priority_tier"]
    if tier == "P1_save":
        return f"we_miss_you_{section}"
    if tier == "P2_grow":
        return f"deep_dive_{section}"
    if tier == "P3_retain":
        return f"loyalty_reward_{section}"
    if tier == "P4_onboard":
        return f"welcome_{section}"
    return f"weekly_pick_{section}"


def suppress(row):
    if pd.isna(row["email_hash"]) or row["email_hash"] == "":
        return True, "no_email_hash"
    if pd.isna(row["recommended_action"]):
        return True, "no_recommendation"
    if row["current_plan"] == "free" and row["expected_value_chf"] < 5:
        return True, "low_expected_value_free"
    if row["recency_days"] > 180:
        return True, "dormant_over_180d"
    return False, "none"


mkt["channel"] = mkt.apply(choose_channel, axis=1)
mkt["campaign_segment"] = mkt.apply(campaign_segment, axis=1)
mkt["creative_angle"] = mkt.apply(creative_angle, axis=1)

suppression = mkt.apply(suppress, axis=1, result_type="expand")
mkt["suppressed"] = suppression[0]
mkt["suppression_reason"] = suppression[1]

rav_max = mkt.loc[~mkt["suppressed"], "risk_adjusted_value_chf"].max()
mkt["budget_weight"] = np.where(
    mkt["suppressed"],
    0.0,
    np.clip(mkt["risk_adjusted_value_chf"] / max(rav_max, 1e-9), 0, 1),
)

send_priority_order = {
    "P1_save": 1, "P2_grow": 2, "P3_retain": 3, "P4_onboard": 4, "P5_nurture": 5,
}
mkt["send_priority"] = mkt["priority_tier"].map(send_priority_order).fillna(9).astype(int)

out_cols = [
    "user_id", "email_hash", "country", "current_plan",
    "priority_tier", "send_priority", "campaign_segment", "channel",
    "recommended_action", "recommended_content_section", "creative_angle",
    "value_tier", "churn_risk_tier", "is_declining",
    "expected_value_chf", "risk_adjusted_value_chf", "budget_weight",
    "upgrade_propensity", "nbo_confidence",
    "suppressed", "suppression_reason", "scored_at",
]

target = mkt[out_cols].copy()

(
    spark.createDataFrame(target)
    .write.mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("pclv.gold.marketing_activation")
)

n_total = len(target)
n_suppressed = int(target["suppressed"].sum())
n_targetable = n_total - n_suppressed

print(f"marketing_activation rows          : {n_total}")
print(f"  targetable                       : {n_targetable}")
print(f"  suppressed                       : {n_suppressed}")

print("\nSuppression reason breakdown:")
print(target[target["suppressed"]]["suppression_reason"].value_counts().to_string())

print("\nTargetable users by priority tier x channel:")
print(pd.crosstab(
    target[~target["suppressed"]]["priority_tier"],
    target[~target["suppressed"]]["channel"],
).to_string())

print("\nTargetable users by campaign_segment (top 15):")
print(
    target[~target["suppressed"]]["campaign_segment"]
    .value_counts().head(15).to_string()
)

print("\nExpected value in book (targetable only, CHF):")
print(f"  sum   : {target.loc[~target['suppressed'], 'expected_value_chf'].sum():,.0f}")
print(f"  mean  : {target.loc[~target['suppressed'], 'expected_value_chf'].mean():,.2f}")
print(f"  p90   : {target.loc[~target['suppressed'], 'expected_value_chf'].quantile(0.9):,.2f}")
