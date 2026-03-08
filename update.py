from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable
import zipfile

import pandas as pd
import requests


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


def main() -> None:
    configure_logging()
    logging.info("Starting update for Statistics Canada table %s", TABLE_ID)

    download_zip(ZIP_URL, RAW_ZIP_PATH)

    extract_zip(
        RAW_ZIP_PATH,
        expected_files=[DATA_FILENAME, METADATA_FILENAME],
        dest_dir=DATA_DIR,
    )

    build_wide_table(RAW_DATA_CSV_PATH, WIDE_CSV_PATH)
    logging.info("Update completed successfully.")


if __name__ == "__main__":
    main()

