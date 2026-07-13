# Subscription pCLV Demo — Dashboard Reference

**Platform:** Databricks AI/BI Dashboard
**Dataset:** `pclv.gold.pclv_customer_scores` (primary), plus `churn_predictions`, `next_best_offer`, `rfe_segments`, `subscriber_value` attached as secondary local datasets
**Status:** Design finalised; four sections × four tiles = 16 tiles + counter strip.

This document is the single-source-of-truth for the dashboard half of the demo.
It sits alongside `data_ingestion.md` (bronze + silver) and
`modelling.md` (gold modelling notebooks).

---

## 1. Where this fits in the project

```
Gold (pclv.gold.pclv_customer_scores)              [covered in modelling.md]
        │
        ▼
AI/BI Dashboard: Subscriber pCLV Activation Dashboard    ◄── this document
        │
        ▼
Marketing team activation + interview walkthrough surface
```

The dashboard is the "so-what" surface of the demo. It's the artefact a Growth
Tribe stakeholder or a marketing manager would actually open, and it's the tile
the reviewers are most likely to click into during the technical deep-dive.

---

## 2. Naming and framing

- **Dashboard name:** *Subscriber pCLV Activation Dashboard*
- **Alternative considered:** *Subscriber Value & Action Cockpit* — more operational tone; kept as a fallback if the "cockpit" framing lands better in the room.
- **Subtitle to display under the title:** *Who to save, who to grow, who to convert — updated {scored_at}*

The dashboard is framed as an **operational surface, not a report**. It reads
left-to-right and top-to-bottom as:

> present state → recent behaviour → segments → future action

which matches how a campaign manager thinks (what's my book? → what's happening
in it? → who are my targets? → what do I do this week?).

---

## 3. Section structure

Four sections, four tiles each. The counter strip in section 1 is the "20-second
read"; sections 2-3 are diagnostic; section 4 is the payoff surface.

### 3.1 Section 1 — Executive Summary (counter strip)

Title: *Executive Summary — The Book Today*

| Tile | What it shows | Field |
|---|---|---|
| 1.1 Active Subscribers | Book size | `COUNT(DISTINCT user_id)` |
| 1.2 Monthly Recurring Revenue | Revenue right now | `SUM(arpu_monthly)` |
| 1.3 Paid Subscriber Share | Conversion state of the book | `% where current_plan != 'free'` |
| 1.4 Average Tenure (months) | Portfolio maturity | `AVG(tenure_days) / 30` |

### 3.2 Section 2 — Current Subscriber Behaviour

Title: *Current Subscriber Behaviour*

| Tile | Chart type | Fields |
|---|---|---|
| 2.1 Plan Mix and Revenue Contribution | Combo (bars + line) | Count by `current_plan`, line = `SUM(arpu_monthly)` |
| 2.2 Acquisition Channel Quality | Combo (bars + line) | `AVG(expected_value_chf)` bars, `AVG(xgb_churn_prob_90d)` line, by `acquisition_channel` |
| 2.3 Engagement vs Recency | Scatter | `recency_days` × `engagement_score`, coloured by `current_plan` |
| 2.4 Cohort Health | Heatmap | `signup_cohort` × `churn_risk_tier`, coloured by count |

### 3.3 Section 3 — Segments & Targetable Population

Title: *Segments & Targetable Population*

| Tile | Chart type | Fields |
|---|---|---|
| 3.1 Subscribers by RFE Segment | Horizontal bar | Count by `segment_label` (from `rfe_segments`) |
| 3.2 Segment Composition by Churn Risk | Stacked bar | `segment_label` × count, stacked by `churn_risk_tier` |
| 3.3 RFE Score Distribution | Bar | Count by `rfe_score` (3–15) |
| 3.4 Targetable Population — Segment × Value Tier | Heatmap | `segment_label` × `value_tier`, coloured by count |

### 3.4 Section 4 — Predicted Value & Next Best Action

Title: *Predicted Value & Next Best Action*

| Tile | Chart type | Fields |
|---|---|---|
| 4.1 Total Expected Value (24m) | Counter | `SUM(expected_value_chf)` with `SUM(risk_adjusted_value_chf)` subtitle |
| 4.2 Subscribers by Priority Tier | Horizontal bar (colour-coded) | Count by `priority_tier`, red → grey |
| 4.3 Value at Stake by Action and Priority | Heatmap | `recommended_action` × `priority_tier`, coloured by `SUM(expected_value_chf)` |
| 4.4 Top 200 Activation Targets | Table | Top 200 by `risk_adjusted_value_chf` desc |

---

## 4. Global filters

Six dashboard-level, multi-select filters, in this order on the filter bar:

1. `current_plan`
2. `priority_tier`
3. `segment_label`
4. `churn_risk_tier`
5. `acquisition_channel`
6. `country`

`scored_at` is deliberately *not* a filter — it's constant across a single
refresh of the gold table. It would become useful once historical scoring runs
are retained, which is a productionisation step, not an MVP concern.

---

## 5. Genie prompts (copy-paste ready)

All prompts assume the corresponding dataset is attached to the dashboard as a
Local dataset.

### Section 1 — Executive Summary

**1.1 Active Subscribers**
> Using the `pclv_customer_scores` dataset, create a counter showing the count of distinct `user_id`. Label it "Active Subscribers".

**1.2 Monthly Recurring Revenue**
> Using the `pclv_customer_scores` dataset, create a counter showing the sum of `arpu_monthly`, formatted as CHF with thousands separator and no decimals. Label it "Monthly Recurring Revenue".

**1.3 Paid Subscriber Share**
> Using the `pclv_customer_scores` dataset, create a counter showing the percentage of subscribers where `current_plan` is not equal to `free`, formatted as a percentage with one decimal. Label it "Paid Subscriber Share".

**1.4 Average Tenure (months)**
> Using the `pclv_customer_scores` dataset, create a counter showing the average of `tenure_days` divided by 30, formatted with one decimal. Label it "Average Tenure (months)".

### Section 2 — Current Subscriber Behaviour

**2.1 Plan Mix and Revenue Contribution**
> Using the `pclv_customer_scores` dataset, create a combo chart grouped by `current_plan`. Show a bar for count of subscribers on the primary y-axis and a line for sum of `arpu_monthly` on the secondary y-axis. Order plans as free, digital, premium, print. Title: "Plan Mix and Monthly Revenue Contribution".

**2.2 Acquisition Channel Quality**
> Using the `pclv_customer_scores` dataset, create a combo chart grouped by `acquisition_channel`. Show a bar for average of `expected_value_chf` on the primary y-axis and a line for average of `xgb_churn_prob_90d` on the secondary y-axis. Sort bars descending by average expected value. Title: "Acquisition Channel Quality — Value vs Churn Risk".

**2.3 Engagement vs Recency**
> Using the `pclv_customer_scores` dataset, create a scatter plot with `recency_days` on the x-axis and `engagement_score` on the y-axis, coloured by `current_plan`. Set point opacity to 0.5. Title: "Engagement vs Recency by Plan".

**2.4 Cohort Health**
> Using the `pclv_customer_scores` dataset, create a heatmap with `signup_cohort` on the y-axis ordered chronologically and `churn_risk_tier` on the x-axis ordered low, medium, high. Colour by count of subscribers. Show the count as a label in each cell. Title: "Cohort Health by Churn Risk".

### Section 3 — Segments & Targetable Population

**3.1 Subscribers by RFE Segment**
> Using the `rfe_segments` dataset, create a horizontal bar chart showing the count of subscribers grouped by `segment_label`, sorted descending. Add a data label with the count on each bar. Title: "Subscribers by RFE Segment".

**3.2 Segment Composition by Churn Risk**
> Using the `pclv_customer_scores` dataset, create a stacked bar chart with `segment_label` on the x-axis and count of subscribers on the y-axis, stacked by `churn_risk_tier` with colours low = green, medium = amber, high = red. Sort segments by total count descending. Title: "Segment Composition by Churn Risk".

**3.3 RFE Score Distribution**
> Using the `rfe_segments` dataset, create a bar chart with `rfe_score` on the x-axis (values 3 to 15) and count of subscribers on the y-axis. Title: "RFE Score Distribution".

**3.4 Targetable Population — Segment × Value Tier**
> Using the `pclv_customer_scores` dataset, create a heatmap with `segment_label` on the y-axis and `value_tier` on the x-axis ordered high, medium, low. Colour by count of subscribers. Show count as a label in each cell. Title: "Targetable Population — Segment × Value Tier".

### Section 4 — Predicted Value & Next Best Action

**4.1 Total Expected Value (24m)**
> Using the `pclv_customer_scores` dataset, create a counter showing the sum of `expected_value_chf`, formatted as CHF with thousands separator and no decimals. Label it "Total Expected Value (24m)". Add a subtitle showing the sum of `risk_adjusted_value_chf` formatted the same way.

**4.2 Subscribers by Priority Tier**
> Using the `pclv_customer_scores` dataset, create a horizontal bar chart showing the count of subscribers grouped by `priority_tier`, sorted by tier name ascending from P1_save to P5_nurture. Colour P1_save red, P2_grow orange, P3_retain amber, P4_onboard blue, P5_nurture grey. Add a data label with the count on each bar. Title: "Subscribers by Priority Tier".

**4.3 Value at Stake by Action and Priority**
> Using the `pclv_customer_scores` dataset, create a heatmap with `recommended_action` on the y-axis and `priority_tier` on the x-axis ordered P1_save to P5_nurture. Colour by sum of `expected_value_chf`. Show the value in each cell formatted as CHF with thousands separator, no decimals. Title: "Value at Stake by Action and Priority".

**4.4 Top 200 Activation Targets**
> Using the `pclv_customer_scores` dataset, create a table showing the top 200 rows ordered by `risk_adjusted_value_chf` descending. Include columns: `user_id`, `current_plan`, `segment_label`, `priority_tier`, `xgb_churn_prob_90d` formatted as percentage with one decimal, `expected_value_chf` formatted as CHF with no decimals, `recommended_action`, `recommended_content_section`. Title: "Top 200 Activation Targets".

### Global filters

> Add dashboard-level filters on `current_plan`, `priority_tier`, `segment_label`, `churn_risk_tier`, `acquisition_channel`, and `country` from the `pclv_customer_scores` dataset. All filters should be multi-select.

---

## 6. Layout

```
┌────────────────────────────────────────────────────────────────┐
│  Subscriber pCLV Activation Dashboard                                 │
│  Filters: plan | priority | segment | risk | channel | country │
├────────────────────────────────────────────────────────────────┤
│  Section 1 — Executive Summary                                 │
│  [1.1] [1.2] [1.3] [1.4]                                       │  ← counter strip
├────────────────────────────────────────────────────────────────┤
│  Section 2 — Current Subscriber Behaviour                     │
│  [2.1 Plan mix combo]     [2.2 Channel quality combo]          │
│  [2.3 Eng vs recency]     [2.4 Cohort heatmap]                 │
├────────────────────────────────────────────────────────────────┤
│  Section 3 — Segments & Targetable Population                  │
│  [3.1 Segment bars]       [3.2 Segment × risk stack]           │
│  [3.3 RFE score dist]     [3.4 Segment × value heatmap]        │
├────────────────────────────────────────────────────────────────┤
│  Section 4 — Predicted Value & Next Best Action                │
│  [4.1 Total value counter][4.2 Priority tier bars]             │
│  [4.3 Value at stake]     [4.4 Top 200 table — full width]     │
└────────────────────────────────────────────────────────────────┘
```

---

## 7. Design decisions worth defending in the interview

### 7.1 Empty P1_save queue and the quantile fix

The initial version of notebook 07 used hardcoded thresholds
(`expected_value_chf >= 500`, `xgb_churn_prob_90d >= 0.5`) for `value_tier` and
`churn_risk_tier`. On the synthetic data this produced an **empty P1_save
queue** — no user simultaneously cleared both bars.

Root cause: the two models are anti-correlated by construction. High
`expected_value_chf` comes from long Cox-derived RMST × high ARPU, which
requires low predicted hazard; the XGBoost 90-day churn model, trained on the
same tenure and engagement features, will also assign low churn probability to
exactly those users. The intersection of "top by value" and "top by churn risk"
is thin or empty.

Fix applied: replaced the absolute cutoffs with **quantile-based cutoffs**
(80th and 50th percentile) so tiers are defined relative to the current book of
business. This has two benefits:

1. **Guaranteed non-empty queues** — top-20% × top-20% ≈ 4% of the book in
   P1_save regardless of the model's absolute score ranges.
2. **Stability under drift** — as the underlying models recalibrate over time,
   the marketing team always receives a top-20% list rather than "whatever the
   model happens to score above 0.5 this week".

Interview framing: *"The empty P1 queue was itself a useful signal — it told me
the two models rank users consistently. The fix isn't to lower thresholds until
a bar appears; it's to define tiers relative to the book, which also happens to
be more robust under drift."* This is exactly the "defensibility as a design
criterion" story worth telling.

### 7.2 Why the four-section structure

The dashboard mirrors how a stakeholder actually consumes information:

- **Section 1 — present**: 20-second read, no drill-down.
- **Section 2 — recent past**: diagnostic behaviour, no predictions.
- **Section 3 — segments**: reachable population by segment × value.
- **Section 4 — future**: prescriptive priorities and the activation queue.

The temporal arc (present → past → structure → future) is a design choice, not
an accident. Worth stating explicitly in the walkthrough — it shows the surface
was designed for the user, not just for the data.

### 7.3 Section 4 as the payoff

If the interview time-boxes the dashboard walkthrough, land on section 4 and
skip 2-3. The heatmap in 4.3 and the top-200 table in 4.4 are the two tiles
that most directly demonstrate the "predictive + prescriptive ML" line from the
JD earning its keep.

### 7.4 Genie complement

The Genie room over `pclv.silver.*` and `pclv.gold.*` handles the long tail of
ad-hoc questions the dashboard can't pre-empt ("show me churn risk for print
subscribers acquired via referral in Q1 2025"). The dashboard is the
opinionated view; Genie is the self-serve escape hatch.

---

## 8. Reproduction

1. Confirm notebook 07 has been rerun with quantile-based tier cutoffs (§7.1).
2. In Databricks: **Dashboards → Create dashboard → Subscriber pCLV Activation Dashboard**.
3. Attach the five gold tables as Local datasets:
   `pclv_customer_scores`, `churn_predictions`, `next_best_offer`,
   `rfe_segments`, `subscriber_value`.
4. Add global filters (§4).
5. Build tiles in order using the Genie prompts (§5). Recommended build order:
   section 4 first (the payoff), then 1, then 2, then 3.
6. Publish and pin to the analytics team workspace.

---

## 9. Known simplifications

- **Static reference date** — `scored_at` is a single value from the current
  batch run. Historical trends would require multiple scoring runs retained
  over time.
- **No uplift measurement** — dashboard shows the *predicted* value at stake;
  measured incremental uplift vs holdout control would require campaign
  execution and post-hoc analysis.
- **Hardcoded tier colour palette** — Databricks AI/BI theming rather than a
  brand palette; a production dashboard would use brand palette.
- **Top 200 hardcoded in tile 4.4** — a production version would parameterise
  the limit and expose an export button for direct CSV download to Marketing.
