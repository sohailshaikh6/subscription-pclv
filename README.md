# Subscription pCLV on Databricks

An end-to-end predictive Customer Lifetime Value (pCLV) demo for a subscription
business, built on Databricks with Unity Catalog, Delta Lake, MLflow, and
Lakeflow SDP. The project adapts Google's *"Predicting LTV"* framework
(originally designed for transactional retail — MBG/NBD + Gamma-Gamma + a
Wide & Deep recommender) to a **contractual / subscription** setting where
churn is directly observed rather than latent.

## What's in here

```
.
├── docs/
│   ├── data_ingestion.md          # Bronze / Silver pipeline (Lakeflow SDP)
│   ├── modelling.md               # Four gold modelling notebooks
│   ├── dashboard.md               # AI/BI dashboard + Genie Room
│   └── methodology_comparison.md  # Contractual vs non-contractual side-by-side
├── notebooks/
│   ├── 03_rfe_segmentation.py     # RFE+T KMeans segmentation
│   ├── 04_churn_survival.py       # XGBoost churn + Cox PH survival
│   ├── 05_subscriber_value.py     # ARPU × RMST subscriber value
│   ├── 06_next_best_offer.py      # ALS + XGBoost recommender
│   ├── 07_pclv_customer_scores.py # Gold consolidation + priority tiering
│   └── 08_marketing_activation.py # CRM/paid-social activation table
└── presentation/
    └── index.html                 # Self-contained HTML slide deck
```

## The four modelling questions

Following the Google "Predicting LTV" decomposition:

| Question | Notebook | Gold table |
|---|---|---|
| **Who** to target | `03_rfe_segmentation.py` | `pclv.gold.rfe_segments` |
| **When** they'll churn | `04_churn_survival.py` | `pclv.gold.churn_predictions` |
| **How much** they'll be worth | `05_subscriber_value.py` | `pclv.gold.subscriber_value` |
| **What** to offer them next | `06_next_best_offer.py` | `pclv.gold.next_best_offer` |

Run order: `03 → 04 → 05 → 06 → 07 → 08`. Notebook 04 must run before 05 —
the Cox model artifact needs to exist in MLflow before subscriber value can
be computed.

## Key methodological point

For a subscription business, **cancellation is directly observed**. This
means the classic non-contractual BTYD stack is not the right fit:

| Question | Non-contractual (retail) | Contractual (subscription) |
|---|---|---|
| Who is still alive? | MBG/NBD latent P(alive) | Directly observed `is_active` |
| When will they churn? | (implicit in MBG/NBD) | Cox PH + XGBoost |
| How much will they be worth? | Gamma-Gamma × E[transactions] | ARPU × RMST |
| What next? | Wide & Deep recommender | ALS + XGBoost upgrade propensity |

See `docs/methodology_comparison.md` for the full side-by-side.

## Platform

Built on Databricks. All notebooks follow a consistent pattern:
`spark.table().toPandas()` → pandas processing → `spark.createDataFrame().saveAsTable()`.
Unity Catalog namespace is `pclv.{bronze,silver,gold}.*`.

## Data

The demo runs on a synthetic subscription dataset generated from public
assumptions (subscriber counts, ARPU bands, plan tiers, engagement
distributions). No real customer data is used.

## License

MIT — see [LICENSE](LICENSE).
