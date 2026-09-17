"""
Step 1: Fetch the Kaggle food-ordering-behavior dataset and do a first pass
of exploration so we know the real schema before designing the DB / agents.

Dataset: rhythmghai/food-ordering-behavior-india-50k-orders

SETUP (one-time, on your Mac):
1. pip install kagglehub pandas pyarrow
2. Get a Kaggle API token:
   - Go to kaggle.com -> Account -> "Create New Token"
   - This downloads kaggle.json
   - Move it to ~/.kaggle/kaggle.json  (mkdir -p ~/.kaggle first)
   - chmod 600 ~/.kaggle/kaggle.json
   (kagglehub will also prompt an interactive browser login if it
   can't find credentials, so this step is optional but recommended
   for repeatable/non-interactive runs.)

RUN:
   python scripts/01_fetch_kaggle_orders.py
"""

import os
import sys
from pathlib import Path

import pandas as pd

DATASET_SLUG = "rhythmghai/food-ordering-behavior-india-50k-orders"
RAW_DIR = Path("data/raw/kaggle_orders")
PROCESSED_DIR = Path("data/processed")


def fetch_dataset() -> Path:
    """Download the dataset via kagglehub and return the local folder path."""
    try:
        import kagglehub
    except ImportError:
        sys.exit(
            "kagglehub not installed. Run: pip install kagglehub"
        )

    print(f"Downloading dataset: {DATASET_SLUG} ...")
    path = kagglehub.dataset_download(DATASET_SLUG)
    print(f"Downloaded to cache: {path}")
    return Path(path)


def load_first_csv(folder: Path) -> pd.DataFrame:
    """Find and load the first CSV in the downloaded folder."""
    csv_files = list(folder.glob("*.csv"))
    if not csv_files:
        sys.exit(f"No CSV files found in {folder}")
    print(f"Found {len(csv_files)} CSV file(s): {[f.name for f in csv_files]}")
    df = pd.read_csv(csv_files[0])
    print(f"Loaded '{csv_files[0].name}' -> {df.shape[0]} rows, {df.shape[1]} cols")
    return df


def explore(df: pd.DataFrame) -> None:
    """Print a quick profile so we can design the schema with real facts."""
    print("\n--- Columns & dtypes ---")
    print(df.dtypes)

    print("\n--- Missing values (top 10) ---")
    print(df.isna().sum().sort_values(ascending=False).head(10))

    print("\n--- Sample rows ---")
    print(df.head(5).to_string())

    print("\n--- Numeric summary ---")
    print(df.describe(include="number").T)

    print("\n--- Categorical cardinality ---")
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        print(f"  {col}: {df[col].nunique()} unique values")


def save_processed(df: pd.DataFrame) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "kaggle_orders_raw.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\nSaved raw copy to: {out_path}")


if __name__ == "__main__":
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dataset_folder = fetch_dataset()
    df = load_first_csv(dataset_folder)
    explore(df)
    save_processed(df)
    print("\nDone. Review the printed profile above, then we'll design "
          "the cleaned schema for the Data Analyst / Segmentation agents.")