"""
Production-grade macro-intelligence pipeline for StatCan CRSPI (Table 18-10-0255-01).
Dual-cadence: monthly data ingestion + validation; quarterly AI forecasting (Jan, Apr, Jul, Oct).
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Iterable, Tuple
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import torch
from chronos import ChronosPipeline


TABLE_ID = "18100255"
LANG = "eng"
BASE_URL = "https://www150.statcan.gc.ca/n1/tbl/csv"
ZIP_FILENAME = f"{TABLE_ID}-{LANG}.zip"
DATA_FILENAME = f"{TABLE_ID}.csv"
METADATA_FILENAME = f"{TABLE_ID}_MetaData.csv"
ZIP_URL = f"{BASE_URL}/{ZIP_FILENAME}"

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
ANALYTICS_DIR = ROOT_DIR / "analytics"

RAW_ZIP_PATH = DATA_DIR / ZIP_FILENAME
RAW_DATA_CSV_PATH = DATA_DIR / DATA_FILENAME
RAW_METADATA_CSV_PATH = DATA_DIR / METADATA_FILENAME
WIDE_CSV_PATH = DATA_DIR / f"{TABLE_ID}_wide.csv"
CRSPI_HISTORY_CSV = DATA_DIR / "crspi_history.csv"

PERFORMANCE_METRICS_JSON = ANALYTICS_DIR / "performance_metrics.json"
LAST_FORECAST_JSON = MODELS_DIR / "last_forecast.json"
MODEL_METADATA_JSON = MODELS_DIR / "model_metadata.json"

FORECAST_HORIZON_MONTHS = 6
NUM_FORECAST_SAMPLES = 64
BACKTEST_MONTHS = 3
FORECAST_MONTHS = (1, 4, 7, 10)  # Jan, Apr, Jul, Oct — quarterly macro outlook

TARGET_SERIES_COLUMN = "Canada | Total, building type"
DATA_SOURCE_LABEL = "StatCan Table 18-10-0255-01 (Commercial Rent Services Price Index)"

# Validation
OUTLIER_STD_THRESHOLD = 4.0  # flag if value beyond mean ± 4*std
MAX_MOM_PCT_CHANGE = 25.0   # flag single-month change > 25%


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def download_zip(url: str, dest_path: Path) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    logging.info("Downloading %s to %s", url, dest_path)
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    with dest_path.open("wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    logging.info("Download complete (%.2f MB)", dest_path.stat().st_size / 1_000_000)


def extract_zip(zip_path: Path, expected_files: Iterable[str], dest_dir: Path) -> None:
    logging.info("Extracting %s into %s", zip_path, dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
        names = set(zf.namelist())
    missing = [n for n in expected_files if n not in names]
    if missing:
        raise RuntimeError(
            f"Expected files {missing} not found in archive. Archive contained: {sorted(names)}"
        )
    logging.info("Extraction complete; found files: %s", ", ".join(sorted(names)))


def build_wide_table(long_csv: Path, wide_csv: Path) -> pd.DataFrame:
    logging.info("Loading long-format data from %s", long_csv)
    df = pd.read_csv(long_csv)
    if "REF_DATE" not in df.columns or "VALUE" not in df.columns:
        raise ValueError("Unexpected file structure: missing REF_DATE or VALUE.")
    technical_cols = {
        "REF_DATE", "VALUE", "STATUS", "SYMBOL", "TERMINATED", "DECIMALS",
        "COORDINATE", "VECTOR", "SCALAR_ID", "SCALAR_FACTOR", "UOM_ID", "UOM", "DGUID",
    }
    dimension_cols = [c for c in df.columns if c not in technical_cols]
    if dimension_cols:
        df["series_name"] = df[dimension_cols].astype(str).agg(" | ".join, axis=1)
    else:
        df["series_name"] = df.get("VECTOR", df["COORDINATE"]).astype(str)
    wide = df.pivot_table(
        index="REF_DATE", columns="series_name", values="VALUE", aggfunc="first"
    )
    wide.sort_index(inplace=True)
    wide.sort_index(axis=1, inplace=True)
    wide.to_csv(wide_csv)
    logging.info("Wrote wide table to %s", wide_csv)
    return wide


def validate_crspi_data(wide_df: pd.DataFrame) -> None:
    """Check for missing values and extreme outliers in the Canada total series."""
    if TARGET_SERIES_COLUMN not in wide_df.columns:
        candidates = [c for c in wide_df.columns if "Canada" in c and "Total, building type" in c]
        col = candidates[0] if candidates else None
    else:
        col = TARGET_SERIES_COLUMN
    if col is None:
        raise ValueError("Target series column for validation not found.")
    series = wide_df[col].astype(float)
    missing = series.isna().sum()
    if missing > 0:
        logging.warning("Validation: %d missing value(s) in target series.", missing)
    vals = series.dropna()
    if len(vals) < 2:
        return
    mean, std = vals.mean(), vals.std()
    if std == 0:
        return
    outliers = np.abs(vals - mean) > OUTLIER_STD_THRESHOLD * std
    if outliers.any():
        logging.warning(
            "Validation: %d potential outlier(s) (|x - mean| > %.1f * std).",
            outliers.sum(), OUTLIER_STD_THRESHOLD,
        )
    pct = vals.pct_change().dropna() * 100
    extreme = np.abs(pct) > MAX_MOM_PCT_CHANGE
    if extreme.any():
        logging.warning(
            "Validation: %d month(s) with MoM change > %.0f%%.",
            extreme.sum(), MAX_MOM_PCT_CHANGE,
        )
    logging.info("Validation checks completed.")


def update_crspi_history(wide_csv: Path) -> pd.Series:
    """Build or refresh crspi_history.csv (date, value) from the wide table."""
    df = pd.read_csv(wide_csv)
    df["REF_DATE"] = pd.to_datetime(df["REF_DATE"])
    df.set_index("REF_DATE", inplace=True)
    col = TARGET_SERIES_COLUMN
    if col not in df.columns:
        candidates = [c for c in df.columns if "Canada" in c and "Total, building type" in c]
        col = candidates[0] if candidates else None
    if col is None:
        raise KeyError("Canada total column not found in wide table.")
    series = df[col].astype(float).dropna().sort_index()
    history = pd.DataFrame({"date": series.index.strftime("%Y-%m-%d"), "value": series.values})
    CRSPI_HISTORY_CSV.parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(CRSPI_HISTORY_CSV, index=False)
    logging.info("Updated %s with %d observations.", CRSPI_HISTORY_CSV, len(series))
    series.index = pd.to_datetime(series.index)
    return series


def load_macro_series_from_history() -> pd.Series:
    """Load Canada total series from crspi_history.csv for forecasting."""
    if not CRSPI_HISTORY_CSV.exists():
        raise FileNotFoundError(
            "crspi_history.csv not found. Run monthly ingestion first."
        )
    df = pd.read_csv(CRSPI_HISTORY_CSV)
    df["date"] = pd.to_datetime(df["date"])
    series = df.set_index("date")["value"].sort_index()
    if len(series) < 24:
        raise ValueError("Insufficient history for forecasting (need at least 24 months).")
    return series


def load_chronos_pipeline():
    logging.info("Loading Chronos-T5 Tiny (CPU, no gradients).")
    return ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-tiny",
        device_map="cpu",
        torch_dtype=torch.float32,
    )


def run_chronos_forecast(series: pd.Series, pipeline=None) -> pd.DataFrame:
    if pipeline is None:
        pipeline = load_chronos_pipeline()
    context = torch.tensor(series.values.astype(np.float32))
    with torch.no_grad():
        forecast = pipeline.predict(
            context, FORECAST_HORIZON_MONTHS, num_samples=NUM_FORECAST_SAMPLES
        )
    samples = forecast[0].cpu().numpy()
    q10, q50, q90 = np.quantile(samples, [0.1, 0.5, 0.9], axis=0)
    last_period = series.index[-1].to_period("M")
    future_index = pd.period_range(
        last_period + 1, periods=FORECAST_HORIZON_MONTHS, freq="M"
    ).to_timestamp("M")
    return pd.DataFrame(
        {"mean": q50, "p10": q10, "p90": q90},
        index=future_index,
    )


def backtest_mape(series: pd.Series, pipeline, num_months: int = BACKTEST_MONTHS) -> float:
    if len(series) < num_months + 24:
        return float("nan")
    context = torch.tensor(series.iloc[:-num_months].values.astype(np.float32))
    with torch.no_grad():
        forecast = pipeline.predict(context, num_months, num_samples=NUM_FORECAST_SAMPLES)
    median_fc = np.median(forecast[0].cpu().numpy(), axis=0)
    actuals = series.iloc[-num_months:].values.astype(np.float64)
    mape = np.mean(np.abs(median_fc - actuals) / (np.abs(actuals) + 1e-10)) * 100.0
    return float(mape)


def analyze_trend(series: pd.Series) -> Tuple[float, str]:
    if len(series) < 2:
        raise ValueError("Need at least two observations for MoM growth.")
    last_val = float(series.iloc[-1])
    prev_val = float(series.iloc[-2])
    if prev_val == 0:
        raise ValueError("Previous value is zero.")
    mom_growth = last_val / prev_val - 1.0
    if mom_growth > 0.01:
        tag = "High Inflationary Pressure"
    elif mom_growth < 0:
        tag = "Market Cooling"
    else:
        tag = "Neutral"
    return mom_growth, tag


def build_dashboard(
    history: pd.Series,
    forecast_df: pd.DataFrame,
    out_path: Path,
    mape_pct: float | None = None,
) -> None:
    ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)
    history = history.sort_index().tail(120)
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
    ax.plot(history.index, history.values, label="Historical Observations", color="C0")
    ax.plot(
        forecast_df.index, forecast_df["mean"].values,
        label="AI Forecast (mean)", color="C1", linestyle="--",
    )
    ax.fill_between(
        forecast_df.index, forecast_df["p10"].values, forecast_df["p90"].values,
        color="C1", alpha=0.25, label="80% prediction interval",
    )
    ax.set_title(
        "Commercial Rent Services Price Index (Canada Total)\n"
        "Historical vs Chronos-T5 Tiny Zero-shot Forecast"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Index level (2019=100)")
    ax.grid(True, linestyle=":", linewidth=0.5)
    ax.legend(loc="upper left")
    if mape_pct is not None and not np.isnan(mape_pct):
        ax.text(
            0.99, 0.02, f"Model reliability (3-month MAPE): {mape_pct:.2f}%",
            transform=ax.transAxes, fontsize=9, va="bottom", ha="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logging.info("Dashboard written to %s", out_path)


def generate_market_intelligence_summary(
    last_obs_value: float,
    mom_growth: float,
    forecast_df: pd.DataFrame,
    trend_tag: str,
) -> str:
    mom_pct = mom_growth * 100.0
    first_fc = float(forecast_df["mean"].iloc[0])
    last_fc = float(forecast_df["mean"].iloc[-1])
    six_month_change_pct = (last_fc / first_fc - 1.0) * 100.0 if first_fc else 0.0
    sentence1 = (
        f"The Commercial Rent Services Price Index (Canada, total building type) "
        f"stands at {last_obs_value:.2f} (2019=100), with a month-over-month change of {mom_pct:+.2f}%."
    )
    if six_month_change_pct > 0.5:
        trend_desc = f"upward trend of about {six_month_change_pct:.1f}% over the next 6 months"
    elif six_month_change_pct < -0.5:
        trend_desc = f"moderate decline of about {abs(six_month_change_pct):.1f}% over the next 6 months"
    else:
        trend_desc = "broadly flat trajectory over the next 6 months"
    sentence2 = f"The Chronos-T5 zero-shot forecast projects a {trend_desc}."
    if six_month_change_pct > 1.0:
        risk = "Accelerating trend detected; monitor for sustained rent pressure."
    elif six_month_change_pct < -0.5:
        risk = "Forecast suggests cooling momentum in commercial rents."
    else:
        risk = "Forecast indicates stabilizing rents with limited near-term volatility."
    return f"{sentence1} {sentence2} {risk}"


def _readme_technical_methodology() -> str:
    return """
---
## Technical Methodology

This pipeline uses **Amazon Chronos-T5 (Tiny)** as a **zero-shot foundation model** for time series forecasting. Chronos treats the series as a sequence of tokens and leverages a pretrained language-model-style architecture to generate probabilistic forecasts without task-specific training.

Compared with traditional methods such as **ARIMA**, Chronos is better suited to **non-linear economic cycles** and regime shifts: it has been pretrained on large corpora of diverse time series, so it can capture complex patterns (e.g., post-COVID adjustments, supply shocks) that fixed-parameter ARIMA models often miss.

- **Model**: `amazon/chronos-t5-tiny` (8M parameters)  
- **Inference**: CPU, no gradient computation (`torch.no_grad()`).  
- **Backtesting**: 3-month MAPE is computed by comparing out-of-sample forecasts to realized data.
"""


def write_executive_readme(
    readme_path: Path,
    last_ingestion_date: str,
    last_forecast_date: str,
    mape_pct: float,
    last_obs_value: float,
    mom_growth: float,
    forecast_df: pd.DataFrame,
    executive_summary: str,
) -> None:
    """Overwrite README with Executive view: System Health + dashboard + Quick Stats + methodology."""
    mom_pct = mom_growth * 100.0
    six_month_fc = float(forecast_df["mean"].iloc[-1]) if len(forecast_df) else 0.0
    mape_str = f"{mape_pct:.2f}%" if not np.isnan(mape_pct) else "N/A"

    lines = [
        "# Continuous Monitoring & Prediction System for Canadian Commercial Real Estate\n\n",
        "## System Health\n\n",
        "| Metric | Value |\n",
        "| --- | --- |\n",
        f"| **Last Ingestion Date** | {last_ingestion_date} |\n",
        f"| **Last Forecast Date** | {last_forecast_date} |\n",
        f"| **Model Accuracy (3-month MAPE)** | {mape_str} |\n",
        f"| **Data Source** | {DATA_SOURCE_LABEL} |\n\n",
        "---\n\n",
        "![CRSPI Forecast Dashboard](analytics/dashboard.png)\n\n",
        "---\n\n",
        "## Quick Stats\n\n",
        "| Metric | Value |\n",
        "| --- | --- |\n",
        f"| **Current Index** (Canada, total building type) | {last_obs_value:.2f} (2019=100) |\n",
        f"| **MoM Change** | {mom_pct:+.2f}% |\n",
        f"| **6-Month Forecast** (mean) | {six_month_fc:.2f} |\n",
        f"| **Model Accuracy (MAPE)** | {mape_str} |\n\n",
        "---\n\n",
        "## Market Intelligence\n\n",
        executive_summary + "\n\n",
        "---\n\n",
        "## 6-Month AI Forecast (Chronos-T5 Tiny)\n\n",
        "| Month | Mean | P10 | P90 |\n",
        "| --- | --- | --- | --- |\n",
    ]
    for ts, row in forecast_df.iterrows():
        lines.append(f"| {ts.strftime('%Y-%m')} | {row['mean']:.2f} | {row['p10']:.2f} | {row['p90']:.2f} |\n")
    lines.append(_readme_technical_methodology())
    lines.append("\n---\n\n")
    lines.append("*Updated automatically by GitHub Actions. **main** = stable production; **ai-forecast** = experimental model tuning.*\n")
    readme_path.write_text("".join(lines), encoding="utf-8")
    logging.info("README.md overwritten (Executive view).")


def run_quarterly_forecast() -> None:
    """Run Chronos inference, backtest, dashboard, metrics, and README (Jan/Apr/Jul/Oct)."""
    macro_series = load_macro_series_from_history()
    mom_growth, trend_tag = analyze_trend(macro_series)

    pipeline = load_chronos_pipeline()
    forecast_df = run_chronos_forecast(macro_series, pipeline=pipeline)
    mape_pct = backtest_mape(macro_series, pipeline, num_months=BACKTEST_MONTHS)

    ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    build_dashboard(
        macro_series, forecast_df,
        ANALYTICS_DIR / "dashboard.png",
        mape_pct=mape_pct,
    )

    performance = {
        "mape_3m_pct": mape_pct,
        "last_backtest_date": datetime.utcnow().strftime("%Y-%m-%d"),
        "forecast_horizon_months": FORECAST_HORIZON_MONTHS,
    }
    PERFORMANCE_METRICS_JSON.write_text(json.dumps(performance, indent=2), encoding="utf-8")

    forecast_date = datetime.utcnow().strftime("%Y-%m-%d")
    LAST_FORECAST_JSON.write_text(
        json.dumps({"last_forecast_date": forecast_date}, indent=2),
        encoding="utf-8",
    )
    MODEL_METADATA_JSON.write_text(
        json.dumps({
            "model_id": "amazon/chronos-t5-tiny",
            "last_forecast_date": forecast_date,
            "forecast_cadence": "quarterly",
        }, indent=2),
        encoding="utf-8",
    )

    last_obs_value = float(macro_series.iloc[-1])
    executive_summary = generate_market_intelligence_summary(
        last_obs_value, mom_growth, forecast_df, trend_tag
    )

    # Last ingestion = latest date in history
    last_ingestion_date = macro_series.index[-1].strftime("%Y-%m-%d")
    last_forecast_date = forecast_date

    write_executive_readme(
        ROOT_DIR / "README.md",
        last_ingestion_date=last_ingestion_date,
        last_forecast_date=last_forecast_date,
        mape_pct=mape_pct,
        last_obs_value=last_obs_value,
        mom_growth=mom_growth,
        forecast_df=forecast_df,
        executive_summary=executive_summary,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="CRSPI dual-cadence pipeline.")
    parser.add_argument(
        "--forecast-only",
        action="store_true",
        help="Skip ingestion; run quarterly forecast from existing crspi_history.csv",
    )
    args = parser.parse_args()

    configure_logging()
    logging.info("Starting production macro-intelligence pipeline for table %s", TABLE_ID)

    if args.forecast_only:
        logging.info("Forecast-only mode: using existing crspi_history.csv")
        run_quarterly_forecast()
        logging.info("Quarterly forecast run completed.")
        return

    # --- Monthly: Data Engineering Cadence ---
    download_zip(ZIP_URL, RAW_ZIP_PATH)
    extract_zip(
        RAW_ZIP_PATH,
        expected_files=[DATA_FILENAME, METADATA_FILENAME],
        dest_dir=DATA_DIR,
    )
    wide_df = build_wide_table(RAW_DATA_CSV_PATH, WIDE_CSV_PATH)
    validate_crspi_data(wide_df)
    update_crspi_history(WIDE_CSV_PATH)

    # --- Quarterly: Analytical Cadence (Jan, Apr, Jul, Oct) ---
    current_month = datetime.utcnow().month
    if current_month in FORECAST_MONTHS:
        logging.info("Quarterly cadence: running AI forecasting (month=%s).", current_month)
        run_quarterly_forecast()
        logging.info("Monthly ingestion + quarterly forecast completed.")
    else:
        logging.info(
            "Monthly ingestion only (forecast runs in Jan/Apr/Jul/Oct). Current month=%s.",
            current_month,
        )
        logging.info("Data engineering cadence completed.")


if __name__ == "__main__":
    main()
