# Continuous Monitoring & Prediction System for Canadian Commercial Real Estate

## System Health

| Metric | Value |
| --- | --- |
| **Last Ingestion Date** | — |
| **Last Forecast Date** | — |
| **Model Accuracy (3-month MAPE)** | — |
| **Data Source** | StatCan Table 18-10-0255-01 (Commercial Rent Services Price Index) |

*(This table is updated automatically: **monthly** ingestion dates; **quarterly** forecast dates and MAPE in Jan, Apr, Jul, Oct.)*

---

## Repository structure

| Path | Purpose |
| --- | --- |
| **data/** | Raw and processed CSVs (`crspi_history.csv`, wide table, zip). |
| **models/** | Inference metadata and last-forecast state (Chronos-T5). |
| **analytics/** | Dashboard image and performance metrics (MAPE/backtesting). |

## Branch policy

- **main** — Stable production. The production pipeline runs from this branch (monthly ingestion; quarterly AI forecast).
- **ai-forecast** — Experimental model tuning and feature development.

---

*Updated automatically by GitHub Actions. See `.github/workflows/production_pipeline.yml` for the dual-cadence schedule.*
