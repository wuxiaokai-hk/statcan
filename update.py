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

# Column used for macro forecasting (Canada-wide total index)
TARGET_SERIES_COLUMN = "Canada | Total, building type"

# Markers for auto-updated section in README
README_FORECAST_START = "<!-- FORECAST_SUMMARY_START -->"
README_FORECAST_END = "<!-- FORECAST_SUMMARY_END -->"


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


def run_chronos_forecast(series: pd.Series) -> pd.DataFrame:
    """
    Run zero-shot forecasting with Chronos-T5 Tiny and return
    a DataFrame with columns ['mean', 'p10', 'p90'] indexed by future timestamps.
    """
    logging.info(
        "Loading Chronos-T5 Tiny model for zero-shot forecasting (CPU, no gradients)."
    )
    pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-tiny",
        device_map="cpu",
        torch_dtype=torch.float32,
    )

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
) -> None:
    logging.info("Building dashboard chart at %s", out_path)

    history = history.sort_index()
    history_tail = history.tail(120)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)

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

    ax.set_title(
        "Commercial Rent Services Price Index (Canada Total)\n"
        "Historical vs Chronos-T5 Tiny Zero-shot Forecast"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Index level (2019=100)")
    ax.grid(True, linestyle=":", linewidth=0.5)
    ax.legend()
    fig.autofmt_xdate()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    logging.info("Dashboard chart written to %s", out_path)


def ensure_readme(path: Path) -> None:
    if path.exists():
        return

    logging.info("Creating README.md at %s", path)
    initial_content = """## Autonomous Macro-Forecasting Agent for StatCan CRSPI

This repository maintains an automated data and forecasting pipeline for
Statistics Canada Table 18-10-0255-01 (Commercial Rent Services Price Index).

The GitHub Actions workflow fetches the latest data, runs an AI forecasting
model (Chronos-T5 Tiny), and publishes updated forecasts and a dashboard image.

"""
    with path.open("w", encoding="utf-8") as f:
        f.write(initial_content)


def update_readme(
    readme_path: Path,
    last_obs_date: pd.Timestamp,
    last_obs_value: float,
    mom_growth: float,
    trend_tag: str,
    forecast_df: pd.DataFrame,
) -> None:
    ensure_readme(readme_path)

    if readme_path.exists():
        text = readme_path.read_text(encoding="utf-8")
    else:
        text = ""

    if README_FORECAST_START not in text or README_FORECAST_END not in text:
        if not text.endswith("\n"):
            text += "\n"
        text += f"\n{README_FORECAST_START}\n{README_FORECAST_END}\n"

    start_idx = text.index(README_FORECAST_START) + len(README_FORECAST_START)
    end_idx = text.index(README_FORECAST_END)

    lines = []
    lines.append("\n")
    lines.append("## Latest CRSPI Macro Forecast\n\n")
    lines.append(
        f"- **Last Observation**: {last_obs_date.strftime('%Y-%m')} = "
        f"{last_obs_value:.2f}\n"
    )
    lines.append(
        f"- **MoM Change**: {mom_growth * 100.0:.2f}%\n"
    )
    lines.append(f"- **Trend Tag**: {trend_tag}\n\n")

    lines.append("### 6-Month AI Forecast (Chronos-T5 Tiny)\n\n")
    lines.append("| Month | Mean | P10 | P90 |\n")
    lines.append("| --- | --- | --- | --- |\n")

    for ts, row in forecast_df.iterrows():
        lines.append(
            f"| {ts.strftime('%Y-%m')} | {row['mean']:.2f} | "
            f"{row['p10']:.2f} | {row['p90']:.2f} |\n"
        )

    lines.append("\n")
    lines.append("![CRSPI Forecast Dashboard](dashboard.png)\n")
    lines.append("\n")

    new_block = "".join(lines)
    new_text = text[:start_idx] + new_block + text[end_idx:]

    readme_path.write_text(new_text, encoding="utf-8")
    logging.info("README.md updated with latest forecast summary.")


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

    forecast_df = run_chronos_forecast(macro_series)

    dashboard_path = ROOT_DIR / "dashboard.png"
    build_dashboard(macro_series, forecast_df, dashboard_path)

    last_obs_date = macro_series.index[-1]
    last_obs_value = float(macro_series.iloc[-1])
    readme_path = ROOT_DIR / "README.md"
    update_readme(
        readme_path,
        last_obs_date=last_obs_date,
        last_obs_value=last_obs_value,
        mom_growth=mom_growth,
        trend_tag=trend_tag,
        forecast_df=forecast_df,
    )

    logging.info("Update, forecasting, and reporting completed successfully.")


if __name__ == "__main__":
    main()

