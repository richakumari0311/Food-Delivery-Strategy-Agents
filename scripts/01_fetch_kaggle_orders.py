"""Fetch and explore the Kaggle food-ordering behavior dataset.

Downloads the dataset, inspects its schema and data quality, and saves
a raw Parquet copy for downstream data-processing tasks.
"""

import sys
from pathlib import Path

import pandas as pd


DATASET_SLUG = "rhythmghai/food-ordering-behavior-india-50k-orders"
PROCESSED_DIR = Path("data/processed")


def fetch_dataset() -> Path:
    """Download the dataset with kagglehub and return its local path."""
    try:
        import kagglehub
    except ImportError:
        sys.exit("kagglehub is not installed. Run: pip install kagglehub")

    print(f"Downloading dataset: {DATASET_SLUG}...")
    dataset_path = kagglehub.dataset_download(DATASET_SLUG)
    print(f"Downloaded to cache: {dataset_path}")

    return Path(dataset_path)


def load_first_csv(folder: Path) -> pd.DataFrame:
    """Load the first CSV file found in the downloaded dataset folder."""
    csv_files = list(folder.glob("*.csv"))

    if not csv_files:
        sys.exit(f"No CSV files found in {folder}")

    print(
        f"Found {len(csv_files)} CSV file(s): "
        f"{[file.name for file in csv_files]}"
    )

    csv_path = csv_files[0]
    dataframe = pd.read_csv(csv_path)

    print(
        f"Loaded '{csv_path.name}' -> "
        f"{dataframe.shape[0]} rows, {dataframe.shape[1]} columns"
    )

    return dataframe


def explore(dataframe: pd.DataFrame) -> None:
    """Print a basic profile of the dataset."""
    print("\n--- Columns & dtypes ---")
    print(dataframe.dtypes)

    print("\n--- Missing values (top 10) ---")
    print(
        dataframe.isna()
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    print("\n--- Sample rows ---")
    print(dataframe.head().to_string())

    print("\n--- Numeric summary ---")
    print(dataframe.describe(include="number").T)

    print("\n--- Categorical cardinality ---")
    categorical_columns = dataframe.select_dtypes(
        include=["object", "category"]
    ).columns

    for column in categorical_columns:
        print(
            f"  {column}: "
            f"{dataframe[column].nunique()} unique values"
        )


def save_processed(dataframe: pd.DataFrame) -> None:
    """Save the raw dataset as a Parquet file."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    output_path = PROCESSED_DIR / "kaggle_orders_raw.parquet"
    dataframe.to_parquet(output_path, index=False)

    print(f"\nSaved raw copy to: {output_path}")


def main() -> None:
    """Fetch, explore, and save the dataset."""
    dataset_folder = fetch_dataset()
    dataframe = load_first_csv(dataset_folder)

    explore(dataframe)
    save_processed(dataframe)

    print("\nDone. Review the dataset profile before designing the schema.")


if __name__ == "__main__":
    main()