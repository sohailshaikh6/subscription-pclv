# Subscription Methodology vs Google "Predicting LTV" Reference

**Purpose:** Map Publisher demo methodology to Google's "Solve with Google — Predicting LTV" reference notebook, showing what was preserved, what was adapted, and what was extended.
**Sits alongside:** `modelling.md`, `the project context`

---

## 1. TL;DR

Google's notebook is a canonical implementation of the **Fader-Hardie BTYD framework** for a **non-contractual, continuous-purchase** business (their example: Google Analytics ecommerce transactions). It composes two probabilistic models — **MBG/NBD** for purchase behaviour and **Gamma-Gamma** for order value — into a 90-day pLTV per customer, then applies descriptive segmentation for activation.

Publisher demo preserves the four-question decomposition (who / when / how much / what) and the compose-two-models-into-value pattern, but swaps every model family because Publisher is **contractual** (subscription): the events Google has to *infer statistically* are events Publisher can *directly observe* in the subscription log. On top of that, Publisher demo adds two components Google doesn't include: a survival-based long-horizon value view and a next-best-offer recommender.

---

## 2. The four questions, side-by-side

| Question | Google notebook | Publisher demo | Reason for change |
|---|---|---|---|
| **Who** (descriptive) | RFM (Recency, Frequency, **Monetary**) — computed by `summary_data_from_transaction_data` | RFE (Recency, Frequency, **Engagement**) — computed from `page_views` in silver | In a subscription business the plan pins Monetary. Engagement (reads/week × section breadth × read-time) is the actual behavioural signal. |
| **When** (probability alive + expected transactions) | **MBG/NBD** (`ModifiedBetaGeoFitter`) — infers latent `probability_alive` from purchase gaps | **Kaplan-Meier + Cox PH + XGBoost 90-day churn** — `is_censored` observed directly from the contract event log | MBG/NBD is *designed* for non-contractual data. Publisher has hard cancel events, so survival analysis is the correct model family. |
| **How much** (expected value) | **Gamma-Gamma** on `monetary_value` → expected AoV; `pLTV = expected_purchases × expected_AoV` | **ARPU × RMST** — plan-price lookup × Cox-derived Restricted Mean Survival Time over 24 months | AoV varies per order in retail; in subscriptions the plan pins revenue. Predicted *tenure* (not price) is the uncertain quantity. |
| **What** (activation) | Descriptive segmentation via `qcut` on 7 dimensions (frequency, idleness, monetary, p(alive), expected purchases, expected AoV, pLTV); export CSV for GMP | **Implicit ALS on user × section + XGBoost upgrade propensity + decision rule** producing one action per user | Google leaves activation to external tools. Publisher demo makes the recommendation explicit and mechanical, using the subscription-ladder structure Publisher actually has. |

---

## 3. Model-family mapping

```
                    Google (non-contractual)              Publisher (contractual)
                    ─────────────────────────             ─────────────────────────
Descriptive     ──► RFM (qcut on M, T, freq)              RFE (quintile on R, F, E)
                                                          + KMeans validation

Alive/churn     ──► MBG/NBD                               Kaplan-Meier (baseline)
                    P(alive) inferred from                Cox PH (survival function)
                    (recency, frequency, T)               XGBoost (90-day discriminator)
                                                          → is_censored OBSERVED

Value           ──► Gamma-Gamma                           ARPU × RMST
                    expected AoV × expected purchases     fixed plan price × Cox-derived tenure

Recommendation  ──► (not in scope; qcut segments         Implicit ALS on sections
                    exported for external activation)     + XGBoost upgrade propensity
                                                          + rule engine → single action
```

---

## 4. The single most important conceptual difference

Both frameworks need to answer the question *"is this customer still with us?"*

- **Google (non-contractual):** liveness is a **latent variable**. Fader-Hardie's BTYD papers spend chapters developing a probabilistic model that infers a hidden `P(alive)` from the pattern of purchase gaps. The MBG/NBD posterior is the answer to a statistical inference problem.
- **Publisher (contractual):** liveness is **directly observed**. The `subscribers` event log contains a `cancel` event with a timestamp. `subscriber_current_state.is_active` and `is_censored` are lookups from silver, not inferences.

The same relationship holds for right-censoring:

| Concept | Google's approach | Publisher's approach |
|---|---|---|
| P(alive) | `bgf.conditional_probability_alive(frequency, recency, T)` — a posterior | `is_active` — a boolean in the silver table |
| Right-censoring | **Excluded**: customers active before `training_period_start` are dropped from training to avoid contamination | **Modelled**: `is_censored` fed into Cox, which handles it natively; XGBoost trained on labelled subset with observation-window rule |
| Cold-start (one-timers, new users) | Gamma-Gamma trained on repeaters only; one-timers scored by applying the fitted distribution | Cox handles all subscribers; RFE has explicit "New / Reactivated" segment; NBO decision rule has explicit `free → convert_to_digital` branch |

This is the mapping the Data Lead is most likely to probe. Being fluent in *why* the model families differ — not just that they do — demonstrates that the framework was understood.

---

## 5. What was preserved from Google's blueprint

- **Four-question decomposition** — who, when, how much, what — as the top-level organising structure.
- **Compose-two-models-into-value pattern** — Google composes MBG/NBD × Gamma-Gamma; Publisher composes Cox → RMST × ARPU. The information flow is identical: model 1 estimates lifetime/behaviour, model 2 estimates value, they multiply into pLTV.
- **90-day horizon for churn** — Google's fixed prediction window is preserved for the XGBoost churn classifier. Value uses a longer 24-month horizon (RMST) because subscription tenure justifies a longer view than transactional cadence does.
- **Calibration/holdout for validation** — Google uses `calibration_and_holdout_data` for RMSE on cumulative purchases; Publisher uses train/test splits with C-index (Cox), AUC / PR-AUC / Brier (XGB), and precision@k (ALS). Same principle, appropriate metrics per model family.
- **Exclusion of extreme outliers** — Google excludes `frequency > 23`; Publisher silver's `@dp.expect` rules and materialised-view canonicalisation handle equivalent hygiene upstream.
- **Segmentation as an activation-facing layer** — Google's `segment_definitions` dictionary and Publisher's `segment_label` + `priority_tier` in `pclv_customer_scores` play the same role: an interpretable label that marketing can filter and target on.

---

## 6. What was extended beyond Google's blueprint

Google's notebook stops at `pLTV × segments → CSV export → GMP`. Publisher demo adds three components:

1. **Explicit next-best-offer recommender** (notebook 06). Google leaves the "what" question implicit — segments are exported and activation is external. Publisher makes the recommendation a mechanical output: implicit ALS on the user × section matrix (HKV confidence formulation) plus an XGBoost upgrade-propensity classifier, combined by a decision rule that respects the plan ladder. This is a direct port of the pattern from Retailer's project (Wide & Deep on CTNs), scaled down because the item catalogue is smaller.

2. **Restricted Mean Survival Time as the value primitive**. Google multiplies expected purchases by expected AoV. Publisher can't do that because subscription revenue isn't a per-transaction random variable. RMST — integrating the Cox survival function over a bounded horizon — gives an expected-remaining-months quantity that behaves the same way (multiplies cleanly with a rate to yield expected value) without requiring a parametric lifetime assumption. This substitution is the interesting piece of Publisher modelling.

3. **A composite priority-tier for marketing routing**. `pclv_customer_scores` combines segment × churn-risk × expected-value into a five-level `priority_tier` (P1_save through P5_nurture) that a campaign manager can filter on directly. Google's segmentation output stops at descriptive tags; Publisher's produces a prescriptive routing label. Quantile-based cutoffs (80th and 50th percentiles on value; three-tier on churn) so tiers scale with the book of business rather than with model calibration drift.

---

## 7. Metric comparison

| Metric | Google | Publisher | Comment |
|---|---|---|---|
| Fit of purchase-count model | RMSE on holdout `frequency_holdout` | C-index (Cox), AUC / PR-AUC / Brier (XGB) | Different model families need different metrics; both are appropriate to their family. |
| Fit of value model | RMSE on `monetary_value_cal` | Distributional summary (mean, p50, p90) + Cox C-index carry-through | Publisher's value model is deterministic given the survival curve, so its accuracy inherits from Cox's C-index. |
| Fit of recommender | *(not applicable)* | Precision@k on 80/20 interaction split (ALS); AUC / PR-AUC (XGB upgrade) | Publisher adds this component so it needed its own validation. |
| Calibration | Visual check only (`plot_period_transactions`) | Brier score logged in MLflow for XGB | Explicit calibration metric worth mentioning if the Data Lead asks. |
| Population-level fit | `plot_cumulative_transactions` over time | Kaplan-Meier curves by plan (logged as MLflow artifact) | Same idea; different visualisation appropriate to model family. |

---

## 8. What Google does that Publisher doesn't (yet)

Three things in the Google notebook that would be worth adding to Publisher before productionisation:

1. **Time-trend accuracy check**. Google's `plot_cumulative_transactions` compares predicted vs actual cumulative volumes over a rolling calibration window. Publisher's KM plot is population-level; a rolling-window equivalent for the Cox model (predicted survivor counts vs observed) would strengthen the monitoring story.

2. **Knee-point analysis for onboarding**. Google's `KneeLocator` on `expected_number_of_purchases_up_to_time` identifies the acquisition-age point beyond which expected purchases fall off sharply — a marketing-actionable finding. The subscription analogue would be a hazard-vs-tenure inflection point, which the Cox model already contains. Not surfaced in the current NBO logic; worth adding as an "onboarding critical window" insight.

3. **Sensitivity heatmaps** (`conditional_probability_alive_matrix`, expected-purchases matrix). Google's 2D heatmaps over (recency, frequency) are excellent storytelling artefacts for stakeholders. The Cox partial-hazard equivalent — a 2D plot over (engagement percentile, tenure) — would translate directly. Cheap to add if there's time before Tuesday.

---

## 9. Interview talking points

Structured statements for the presentation and Q&A:

**Opening framing.** "Publisher demo applies the same framework as the Google 'Predicting LTV' reference — four questions, two models composed into value, activation-facing segmentation — but every model family had to change because Publisher is contractual and Google's example was non-contractual. The change is not stylistic. In Google's world, `P(alive)` is a latent variable that MBG/NBD infers from purchase gaps; in Publisher's world, `is_active` is a column in the silver table."

**On the value model.** "Google's Gamma-Gamma computes expected AoV as a random variable per transaction. In a subscription business AoV is a lookup — plan price. The uncertain quantity is *tenure*, not price. That's why we use RMST on the Cox survival function: it gives expected-remaining-months conditional on current tenure, without requiring a parametric assumption on lifetime distribution."

**On right-censoring.** "Google's notebook handles right-censoring by exclusion — customers active before the training window are dropped. Cox handles it as a first-class part of the likelihood. Since the silver layer already carries `is_censored` and `tenure_days` per subscriber, the survival model is a plug-and-play fit, not a re-derivation."

**On the recommender.** "Google's notebook stops at segments-plus-export. I added an explicit next-best-offer step because Retailer's pCLV project I'm presenting for the main slot had a Wide & Deep recommender doing the same job on ~thousands of SKUs. On Publisher's ~15 sections and 4 plans, ALS with Hu-Koren-Volinsky confidence is sufficient and interprets more cleanly, and the primary recommendation object is a plan action, not a SKU."

**On monitoring.** "Google's monitoring story is a visual calibration curve. In production I'd log Brier score over time on the XGB churn classifier, KS-drift on the engagement features, and segment-level uplift against a holdout — that's the observability layer we didn't have on Retailer's project and I'd add it here from day one."

---

## 10. Summary in one paragraph

Google's notebook is a canonical Fader-Hardie BTYD implementation for retail: MBG/NBD infers `P(alive)`, Gamma-Gamma predicts AoV, their product is 90-day pLTV, and descriptive segmentation exports to activation. Publisher demo preserves the four-question framework, the compose-two-models pattern, and the 90-day horizon, but swaps MBG/NBD for Cox PH + XGBoost (because cancels are observed events, not latent inferences), swaps Gamma-Gamma for ARPU × RMST (because plan price pins revenue), and adds an explicit next-best-offer recommender and a priority-tier routing label because subscription businesses can act on both content-affinity and plan-ladder decisions in ways that Google's retail example cannot.
