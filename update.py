from __future__ import annotations

import logging
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
RAW_ZIP_PATH = DATA_DIR / ZIP_FILENAME
RAW_DATA_CSV_PATH = DATA_DIR / DATA_FILENAME
RAW_METADATA_CSV_PATH = DATA_DIR / METADATA_FILENAME
WIDE_CSV_PATH = DATA_DIR / f"{TABLE_ID}_wide.csv"

FORECAST_HORIZON_MONTHS = 6
NUM_FORECAST_SAMPLES = 64
BACKTEST_MONTHS = 3  # MAPE over last N months

# Column used for macro forecasting (Canada-wide total index)
TARGET_SERIES_COLUMN = "Canada | Total, building type"


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

    missing = [name for name in expected_files if name not in names]
    if missing:
        raise RuntimeError(
            f"Expected files {missing} not found in archive. "
            f"Archive contained: {sorted(names)}"
        )

    logging.info("Extraction complete; found files: %s", ", ".join(sorted(names)))


def build_wide_table(long_csv: Path, wide_csv: Path) -> None:
    logging.info("Loading long-format data from %s", long_csv)
    df = pd.read_csv(long_csv)
    logging.info("Loaded %d rows and %d columns", len(df), df.shape[1])

    if "REF_DATE" not in df.columns or "VALUE" not in df.columns:
        raise ValueError(
            "Unexpected file structure: missing required columns 'REF_DATE' or 'VALUE'."
        )

    technical_cols = {
        "REF_DATE",
        "VALUE",
        "STATUS",
        "SYMBOL",
        "TERMINATED",
        "DECIMALS",
        "COORDINATE",
        "VECTOR",
        "SCALAR_ID",
        "SCALAR_FACTOR",
        "UOM_ID",
        "UOM",
        "DGUID",
    }

    dimension_cols = [c for c in df.columns if c not in technical_cols]

    if dimension_cols:
        logging.info("Using dimension columns for series labels: %s", dimension_cols)
        df["series_name"] = df[dimension_cols].astype(str).agg(" | ".join, axis=1)
    else:
        logging.warning(
            "No dimension columns detected; falling back to VECTOR/COORDINATE labels."
        )
        if "VECTOR" in df.columns:
            df["series_name"] = df["VECTOR"].astype(str)
        elif "COORDINATE" in df.columns:
            df["series_name"] = df["COORDINATE"].astype(str)
        else:
            raise ValueError(
                "Could not determine a suitable column for series labels."
            )

    logging.info("Pivoting data to wide format")
    wide = df.pivot_table(
        index="REF_DATE",
        columns="series_name",
        values="VALUE",
        aggfunc="first",
    )

    wide.sort_index(inplace=True)
    wide = wide.sort_index(axis=1)

    wide.to_csv(wide_csv)
    logging.info("Wrote wide-format data to %s", wide_csv)


def load_macro_series(wide_csv: Path) -> pd.Series:
    logging.info("Loading wide-format table from %s", wide_csv)
    df = pd.read_csv(wide_csv)
    df["REF_DATE"] = pd.to_datetime(df["REF_DATE"])
    df.set_index("REF_DATE", inplace=True)

    target_col = TARGET_SERIES_COLUMN
    if target_col not in df.columns:
        # Fallback: try to find a close match for the Canada total column
        candidates = [
            c for c in df.columns if "Canada" in c and "Total, building type" in c
        ]
        if not candidates:
            raise KeyError(
                f"Could not find target series column '{TARGET_SERIES_COLUMN}' "
                f"in wide table. Available columns include: {list(df.columns)[:10]}..."
            )
        target_col = candidates[0]
        logging.warning("Using fallback target column: %s", target_col)

    series = df[target_col].astype(float).dropna()
    if len(series) < 24:
        raise ValueError(
            "Not enough historical observations for forecasting (need at least 24)."
        )

    logging.info(
        "Loaded macro series '%s' with %d monthly observations", target_col, len(series)
    )
    return series


def load_chronos_pipeline():
    """Load Chronos-T5 Tiny once for reuse in forecast and backtest."""
    logging.info(
        "Loading Chronos-T5 Tiny model for zero-shot forecasting (CPU, no gradients)."
    )
    return ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-tiny",
        device_map="cpu",
        torch_dtype=torch.float32,
    )


def run_chronos_forecast(series: pd.Series, pipeline=None) -> pd.DataFrame:
    """
    Run zero-shot forecasting with Chronos-T5 Tiny and return
    a DataFrame with columns ['mean', 'p10', 'p90'] indexed by future timestamps.
    """
    if pipeline is None:
        pipeline = load_chronos_pipeline()

    context = torch.tensor(series.values.astype(np.float32))

    with torch.no_grad():
        forecast = pipeline.predict(
            context,
            FORECAST_HORIZON_MONTHS,
            num_samples=NUM_FORECAST_SAMPLES,
        )

    # forecast: [num_series=1, num_samples, prediction_length]
    samples = forecast[0].cpu().numpy()
    q10, q50, q90 = np.quantile(samples, [0.1, 0.5, 0.9], axis=0)

    last_timestamp = series.index[-1]
    last_period = last_timestamp.to_period("M")
    future_periods = pd.period_range(
        last_period + 1, periods=FORECAST_HORIZON_MONTHS, freq="M"
    )
    future_index = future_periods.to_timestamp("M")

    forecast_df = pd.DataFrame(
        {
            "mean": q50,
            "p10": q10,
            "p90": q90,
        },
        index=future_index,
    )

    logging.info("Generated Chronos forecast for %d future months.", len(forecast_df))
    return forecast_df


def backtest_mape(series: pd.Series, pipeline, num_months: int = BACKTEST_MONTHS) -> float:
    """
    Compare model forecast vs actual for the last `num_months`.
    Returns MAPE (Mean Absolute Percentage Error) in percentage points.
    """
    if len(series) < num_months + 24:
        logging.warning(
            "Insufficient history for %d-month backtest; returning NaN.", num_months
        )
        return float("nan")

    context = torch.tensor(
        series.iloc[:-num_months].values.astype(np.float32)
    )

    with torch.no_grad():
        forecast = pipeline.predict(
            context,
            num_months,
            num_samples=NUM_FORECAST_SAMPLES,
        )

    # [1, num_samples, num_months] -> median over samples
    median_forecast = np.median(forecast[0].cpu().numpy(), axis=0)
    actuals = series.iloc[-num_months:].values.astype(np.float64)
    # MAPE = mean(|forecast - actual| / |actual|) * 100, avoid div by zero
    mape = np.mean(
        np.abs(median_forecast - actuals) / (np.abs(actuals) + 1e-10)
    ) * 100.0

    logging.info(
        "Backtest MAPE (last %d months): %.2f%%", num_months, mape
    )
    return float(mape)


def analyze_trend(series: pd.Series) -> Tuple[float, str]:
    """
    Compute month-over-month growth and map to a qualitative tag.
    """
    if len(series) < 2:
        raise ValueError("Need at least two observations to compute MoM growth.")

    last_val = float(series.iloc[-1])
    prev_val = float(series.iloc[-2])
    if prev_val == 0:
        raise ValueError("Previous value is zero; cannot compute MoM growth.")

    mom_growth = last_val / prev_val - 1.0

    if mom_growth > 0.01:
        tag = "High Inflationary Pressure"
    elif mom_growth < 0:
        tag = "Market Cooling"
    else:
        tag = "Neutral"

    logging.info(
        "Latest MoM growth: %.3f%% (%s)", mom_growth * 100.0, tag
    )
    return mom_growth, tag


def build_dashboard(
    history: pd.Series,
    forecast_df: pd.DataFrame,
    out_path: Path,
    mape_pct: float | None = None,
) -> None:
    logging.info("Building dashboard chart at %s", out_path)

    history = history.sort_index()
    history_tail = history.tail(120)

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)

    ax.plot(
        history_tail.index,
        history_tail.values,
        label="Historical Observations",
        color="C0",
    )

    ax.plot(
        forecast_df.index,
        forecast_df["mean"].values,
        label="AI Forecast (mean)",
        color="C1",
        linestyle="--",
    )

    ax.fill_between(
        forecast_df.index,
        forecast_df["p10"].values,
        forecast_df["p90"].values,
        color="C1",
        alpha=0.25,
        label="80% prediction interval",
    )

    title = (
        "Commercial Rent Services Price Index (Canada Total)\n"
        "Historical vs Chronos-T5 Tiny Zero-shot Forecast"
    )
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Index level (2019=100)")
    ax.grid(True, linestyle=":", linewidth=0.5)
    ax.legend(loc="upper left")

    # Model reliability: MAPE for last 3 months
    if mape_pct is not None and not np.isnan(mape_pct):
        reliability_text = f"Model reliability (3-month MAPE): {mape_pct:.2f}%"
        ax.text(
            0.99,
            0.02,
            reliability_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

    fig.autofmt_xdate()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    logging.info("Dashboard chart written to %s", out_path)


def generate_market_intelligence_summary(
    last_obs_value: float,
    mom_growth: float,
    forecast_df: pd.DataFrame,
    trend_tag: str,
) -> str:
    """
    Generate a 3-sentence Market Intelligence summary:
    1) Current CRSPI value and MoM change
    2) 6-month projected trend
    3) Risk assessment
    """
    mom_pct = mom_growth * 100.0
    first_fc = float(forecast_df["mean"].iloc[0])
    last_fc = float(forecast_df["mean"].iloc[-1])
    six_month_change_pct = (last_fc / first_fc - 1.0) * 100.0 if first_fc else 0.0

    sentence1 = (
        f"The Commercial Rent Services Price Index (Canada, total building type) "
        f"stands at {last_obs_value:.2f} (2019=100), with a month-over-month change "
        f"of {mom_pct:+.2f}%."
    )

    if six_month_change_pct > 0.5:
        trend_desc = f"upward trend of about {six_month_change_pct:.1f}% over the next 6 months"
    elif six_month_change_pct < -0.5:
        trend_desc = f"moderate decline of about {abs(six_month_change_pct):.1f}% over the next 6 months"
    else:
        trend_desc = "broadly flat trajectory over the next 6 months"

    sentence2 = (
        f"The Chronos-T5 zero-shot forecast projects a {trend_desc}."
    )

    if six_month_change_pct > 1.0:
        risk = "Accelerating trend detected; monitor for sustained rent pressure."
    elif six_month_change_pct < -0.5:
        risk = "Forecast suggests cooling momentum in commercial rents."
    else:
        risk = "Forecast indicates stabilizing rents with limited near-term volatility."

    sentence3 = risk

    return f"{sentence1} {sentence2} {sentence3}"


def _readme_technical_methodology() -> str:
    """Static Technical Methodology section for the README."""
    return """
---

## Technical Methodology

This pipeline uses **Amazon Chronos-T5 (Tiny)** as a **zero-shot foundation model** for time series forecasting. Chronos treats the series as a sequence of tokens and leverages a pretrained language-model-style architecture to generate probabilistic forecasts without task-specific training.

Compared with traditional methods such as **ARIMA**, Chronos is better suited to **non-linear economic cycles** and regime shifts: it has been pretrained on large corpora of diverse time series, so it can capture complex patterns (e.g., post-COVID adjustments, supply shocks) that fixed-parameter ARIMA models often miss. The 6-month forecasts and 80% prediction intervals are produced in a single forward pass with no hyperparameter tuning on this series.

- **Model**: `amazon/chronos-t5-tiny` (8M parameters)  
- **Inference**: CPU, no gradient computation (`torch.no_grad()`).  
- **Backtesting**: 3-month MAPE is computed by comparing out-of-sample forecasts to realized data to report model reliability.
"""


def write_full_readme(
    readme_path: Path,
    last_obs_date: pd.Timestamp,
    last_obs_value: float,
    mom_growth: float,
    trend_tag: str,
    forecast_df: pd.DataFrame,
    mape_pct: float,
    executive_summary: str,
) -> None:
    """
    Overwrite README.md with the full dashboard layout:
    H1 title, dashboard image, Quick Stats, Market Intelligence, forecast table, methodology.
    """
    mom_pct = mom_growth * 100.0
    six_month_fc = float(forecast_df["mean"].iloc[-1]) if len(forecast_df) else 0.0
    mape_str = f"{mape_pct:.2f}%" if not np.isnan(mape_pct) else "N/A"

    lines = [
        "# 📈 Real-Time Commercial Rent Intelligence\n\n",
        "![CRSPI Forecast Dashboard](dashboard.png)\n\n",
        "---\n\n",
        "## Quick Stats\n\n",
        "| Metric | Value |\n",
        "| --- | --- |\n",
        f"| **Current Index** (Canada, total building type) | {last_obs_value:.2f} (2019=100) |\n",
        f"| **MoM Change** | {mom_pct:+.2f}% |\n",
        f"| **6-Month Forecast** (mean) | {six_month_fc:.2f} |\n",
        f"| **Model Accuracy (3-month MAPE)** | {mape_str} |\n\n",
        "---\n\n",
        "## Market Intelligence\n\n",
        executive_summary + "\n\n",
        "---\n\n",
        "## 6-Month AI Forecast (Chronos-T5 Tiny)\n\n",
        "| Month | Mean | P10 | P90 |\n",
        "| --- | --- | --- | --- |\n",
    ]

    for ts, row in forecast_df.iterrows():
        lines.append(
            f"| {ts.strftime('%Y-%m')} | {row['mean']:.2f} | "
            f"{row['p10']:.2f} | {row['p90']:.2f} |\n"
        )

    lines.append(_readme_technical_methodology())
    lines.append("\n")
    lines.append("---\n\n")
    lines.append("*Data: Statistics Canada Table 18-10-0255-01 (Commercial Rent Services Price Index). Updated automatically by GitHub Actions.*\n")

    readme_path.write_text("".join(lines), encoding="utf-8")
    logging.info("README.md overwritten with full dashboard content.")


def main() -> None:
    configure_logging()
    logging.info("Starting autonomous macro-forecasting update for table %s", TABLE_ID)

    download_zip(ZIP_URL, RAW_ZIP_PATH)

    extract_zip(
        RAW_ZIP_PATH,
        expected_files=[DATA_FILENAME, METADATA_FILENAME],
        dest_dir=DATA_DIR,
    )

    build_wide_table(RAW_DATA_CSV_PATH, WIDE_CSV_PATH)

    macro_series = load_macro_series(WIDE_CSV_PATH)
    mom_growth, trend_tag = analyze_trend(macro_series)

    pipeline = load_chronos_pipeline()
    forecast_df = run_chronos_forecast(macro_series, pipeline=pipeline)
    mape_pct = backtest_mape(macro_series, pipeline, num_months=BACKTEST_MONTHS)

    dashboard_path = ROOT_DIR / "dashboard.png"
    build_dashboard(
        macro_series,
        forecast_df,
        dashboard_path,
        mape_pct=mape_pct,
    )

    last_obs_date = macro_series.index[-1]
    last_obs_value = float(macro_series.iloc[-1])
    executive_summary = generate_market_intelligence_summary(
        last_obs_value, mom_growth, forecast_df, trend_tag
    )

    readme_path = ROOT_DIR / "README.md"
    write_full_readme(
        readme_path,
        last_obs_date=last_obs_date,
        last_obs_value=last_obs_value,
        mom_growth=mom_growth,
        trend_tag=trend_tag,
        forecast_df=forecast_df,
        mape_pct=mape_pct,
        executive_summary=executive_summary,
    )

    logging.info("Update, forecasting, and reporting completed successfully.")


if __name__ == "__main__":
    main()

