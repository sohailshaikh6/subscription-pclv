# Subscription pCLV Demo — Data Ingestion Reference

**Platform:** Databricks (Lakeflow Spark Declarative Pipelines)
**Status:** Bronze + Silver complete; Gold (feature tables) next.

This document is the single-source-of-truth for the data-ingestion half of the demo.
Read this before touching the modelling notebooks.

---

## 1. Databricks project layout

```
nzz_data_ingestion/                            ← Lakeflow pipeline project
└── transformations/
    ├── bronze/
    │   └── 01_bronze_ingestion_sdp.py         ← Auto Loader from Volume → 5 bronze tables
    └── silver/
        └── 02_silver_pipeline_sdp.py          ← Cleaning, RI, derived state → 6 silver tables
```

**Two separate pipelines**, not one:
- **Bronze pipeline** — target `pclv.bronze`, sources `transformations/bronze/`
- **Silver pipeline** — target `pclv.silver`, sources `transformations/silver/`

Silver reads bronze via fully-qualified name (`pclv.bronze.subscribers`). SDP records the dependency and orders execution automatically.

---

## 2. Unity Catalog surface

**Catalog:** `pclv`

| Schema | Purpose | Managed by |
|---|---|---|
| `raw_data` | Volume `raw_data` — CSV landing zone; `_autoloader_state/` subfolder for AL checkpoints | Manual upload |
| `bronze` | Raw-faithful Delta tables with ingestion metadata | Bronze pipeline |
| `silver` | Conformed, deduped, RI-enforced tables + derived state | Silver pipeline |
| `gold` | (empty — feature tables + model outputs go here) | Modelling notebooks |

### Bronze tables (`pclv.bronze.*`)

| Table | Rows | Grain | Notes |
|---|---|---|---|
| `subscribers` | ~5,000 | one per user | signup + demographics + email_hash |
| `subscriptions` | ~7,400 | one per plan event | start / upgrade / downgrade / cancel |
| `articles` | ~2,000 | one per article | section, topic, paywall_type |
| `page_views` | ~394,000 | one per view event | user × article × timestamp × read_time |
| `campaigns` | ~4,500 | one per (campaign, user) | treatment / control assignment |

Every bronze row carries: `_ingested_at`, `_source_file` (from Auto Loader `_metadata`).

### Silver tables (`pclv.silver.*`)

Same five tables, plus one derived:

| Table | What's different vs Bronze |
|---|---|
| `subscribers` | Canonicalised categoricals, deduped, `signup_cohort` + `signup_year_month` added |
| `articles` | Canonicalised, deduped, `is_premium_content` flag added |
| `subscriptions` | RI-enforced against subscribers; event-type null contracts enforced; `event_seq`, `days_since_signup` added |
| `page_views` | RI against subscribers + articles; causal timestamp check; `read_time_bucket`, `article_age_days`, `hour_of_day`, `day_of_week`, `is_post_cancel_view` added |
| `campaigns` | RI-enforced; `campaign_length_days`, `assignment_hash` added |
| **`subscriber_current_state`** | **NEW** — one row per user with `current_plan`, `is_active`, `cancel_date`, `tenure_days`, `is_censored`. Point-in-time contract state derived from the event log. |

---

## 3. Dependency graph

```
Volume /Volumes/pclv/raw_data/raw_data/*.csv
    │
    │ Auto Loader (streaming, cloudFiles)
    ▼
bronze.subscribers ──►  silver.subscribers    ─────────────┐
bronze.articles    ──►  silver.articles       ─────────┐   │
bronze.subscriptions──► silver.subscriptions   ◄───────┼───┤
                              │                        │   │
                              ▼                        │   │
                    silver.subscriber_current_state ◄──┼───┤
                              │                        │   │
                              ▼                        │   │
bronze.page_views ─────► silver.page_views  ◄──────────┴───┘
bronze.campaigns  ─────► silver.campaigns   ◄──────────────┘
```

Downstream Gold / modelling notebooks read from Silver only.

---

## 4. Design choices worth defending in the interview

### Bronze
- **Streaming tables + Auto Loader** — idempotent by construction; re-runs process only new files. Same code will work when CSVs are replaced by a real event stream.
- **Explicit schemas** — no `inferSchema`; catches upstream drift as a load-time error.
- **Lenient expectations** (`@dp.expect`, warn + keep) — Bronze's job is to land faithfully; enforcement belongs to Silver.
- **CDF enabled** (`delta.enableChangeDataFeed = true`) — makes future incremental Silver builds trivial.
- **Ingestion metadata** via Auto Loader's `_metadata` column — every row is traceable to its source file.

### Silver
- **Two-tier expectations:**
  - `@dp.expect_or_fail` for structural violations (null PKs, unknown event types) — halt the pipeline.
  - `@dp.expect_or_drop` for referential-integrity failures — orphan rows dropped and counted in the event log.
  - `@dp.expect` for soft flags (e.g. `event_date < signup_date`) — kept for investigation.
- **`dp.read("<name>")` for intra-pipeline refs** — SDP records the edge and renders the DAG in the UI.
- **`spark.read.table("pclv.bronze.<name>")` for cross-pipeline refs** — standard Unity Catalog read.
- **Materialised views everywhere at Silver** — the joins and aggregations (especially `subscriber_current_state`) are batch-natural.
- **`subscriber_current_state` as first-class table** — the subscription-business analogue of MBG/NBD's `P(alive)` from Retailer's project. Instead of *inferring* liveness from purchase gaps, we compute it directly from the contract event log.
- **`is_censored` column** — right-censoring pre-computed for the survival model. Direct answer to the "how do you handle right-censoring?" question the Data Lead will ask.

---

## 5. Data-quality monitoring

All expectation results are written to the pipeline event log — no bespoke DQ table needed. To query:

```sql
SELECT
  timestamp,
  details:flow_progress.data_quality.expectations
FROM event_log(TABLE(<pipeline_id>))
WHERE event_type = 'flow_progress'
ORDER BY timestamp DESC;
```

Dropped-row counts per expectation are surfaced automatically in the Lakeflow pipeline UI.

---

## 6. Reproduction steps

### First-time setup
1. Create Unity Catalog objects (one-time, via SQL):
   ```sql
   CREATE CATALOG IF NOT EXISTS pclv;
   CREATE SCHEMA IF NOT EXISTS pclv.raw_data;
   CREATE SCHEMA IF NOT EXISTS pclv.bronze;
   CREATE SCHEMA IF NOT EXISTS pclv.silver;
   CREATE SCHEMA IF NOT EXISTS pclv.gold;
   CREATE VOLUME IF NOT EXISTS pclv.raw_data.raw_data;
   ```
2. Upload the 5 CSVs to `/Volumes/pclv/raw_data/raw_data/`
3. Import `nzz_data_ingestion.dbc` into the workspace

### Create the two pipelines
For each of Bronze and Silver:
- **Workflows → Lakeflow Pipelines → Create pipeline**
- Product edition: **Advanced** (required for `@dp.expect`)
- Serverless: **on**
- Target catalog: `pclv`
- Target schema: `bronze` (or `silver`)
- Source code: point at the corresponding folder

### Run
1. Run Bronze pipeline first — populates `pclv.bronze.*`
2. Run Silver pipeline — reads from bronze, populates `pclv.silver.*`

### Regenerate synthetic data
The generator lives in `generate_data.py`. Run locally, then re-upload CSVs to the Volume. Seed is fixed (`42`) so runs are reproducible.

---

## 7. Known simplifications (things a real project would add)

- Controlled vocabularies (plans, countries, sections) are hardcoded in the Silver code. Production would keep these in reference tables in Unity Catalog and join.
- No PII beyond the sha256 `email_hash` — no name, address, payment info. In a real Publisher pipeline, PII would live in a separate governed schema with column-level access controls.
- No SCD-2 history on `subscribers` — the current implementation is last-write-wins. If the marketing team needs demographic-change history, we'd swap to `dp.create_auto_cdc_flow` or an SCD-2 pattern.
- Bronze is a single-shot ingestion of five CSVs. In production this would be Auto Loader against a continuous file drop or Kafka streams; the Silver code above does not change.

---

## 8. How the JD maps to what got built

| JD skill line | Where it's demonstrated in the pipeline |
|---|---|
| "Predictive and prescriptive ML" | (Modelling layer — to come) |
| "Full model lifecycle: ideation → experimentation → implementation → monitoring" | MLflow tracking on model notebooks; event-log monitoring on pipeline |
| "Code quality, reproducibility" | Two-tier pipeline separation, seeded synthetic data, expectation-driven DQ |
| "Cross-functional work with Marketing" | `subscriber_current_state` designed as the read-model for downstream targeting exports |
| "Paywall & subscription models" | Contract-based `subscriber_current_state`, `is_censored` for survival, `is_post_cancel_view` for winback |
| "GCP + Vertex AI" (JD-preferred) | Talking point ready: Volume → GCS, Delta → BigQuery, Auto Loader → Dataflow, Lakeflow → Vertex AI Pipelines. Architecture translates 1:1. |

---

## 9. Files in this project

| File | Purpose |
|---|---|
| `the project context` | Overall interview strategy and framing |
| `data_ingestion.md` | This document |
| `generate_data.py` | Synthetic data generator (5 CSVs, seed=42) |
| `01_bronze_ingestion_sdp.py` | Bronze pipeline source (in `transformations/bronze/`) |
| `02_silver_pipeline_sdp.py` | Silver pipeline source (in `transformations/silver/`) |
| `silver_cleaning_plan.md` | Original cleaning-rule design doc (kept for interview walkthrough) |
| `nzz_data_ingestion.dbc` | Databricks workspace export of the pipeline project |

---

## 10. Next steps

- [ ] Gold layer — feature tables for RFE segmentation, churn/survival, subscriber value, next-best-offer
- [ ] Four modelling notebooks with MLflow tracking
- [ ] `gold.pclv_customer_scores` joining all model outputs
- [ ] AI/BI Dashboard on `pclv_customer_scores` for the marketing team
- [ ] Genie room over the silver + gold surface for stakeholder self-serve
- [ ] Architecture diagram for slides
