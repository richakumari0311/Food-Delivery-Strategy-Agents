"""Clean the Kaggle orders dataset and build user-level features.

Creates a typed row-level dataset and an aggregated user feature table
for the Segmentation Agent.
"""

from pathlib import Path

import pandas as pd


RAW_PATH = Path("data/processed/kaggle_orders_raw.parquet")
CLEAN_PATH = Path("data/processed/kaggle_orders_clean.parquet")
USER_FEATURES_PATH = Path("data/processed/kaggle_user_features.parquet")

BOOL_YES_NO_COLS = [
    "discount_applied",
    "is_repeat_order",
    "rainy_weather",
]

CATEGORY_COLS = [
    "city",
    "order_time",
    "day_type",
    "cuisine",
    "meal_type",
    "restaurant_type",
    "mood",
    "hunger_level",
    "company",
]

AGE_BINS = [0, 24, 34, 44, 200]
AGE_LABELS = ["18-24", "25-34", "35-44", "45+"]

RESTAURANT_TYPES = ["Budget", "Mid-range", "Premium"]
HUNGER_LEVELS = ["Low", "Medium", "High"]


def clean(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Clean columns, apply types, and create derived features."""
    dataframe = dataframe.copy()

    for column in BOOL_YES_NO_COLS:
        dataframe[column] = (
            dataframe[column]
            .map({"Yes": True, "No": False})
            .astype("boolean")
        )

    for column in CATEGORY_COLS:
        dataframe[column] = dataframe[column].astype("category")

    dataframe["restaurant_type"] = pd.Categorical(
        dataframe["restaurant_type"],
        categories=RESTAURANT_TYPES,
        ordered=True,
    )

    dataframe["hunger_level"] = pd.Categorical(
        dataframe["hunger_level"],
        categories=HUNGER_LEVELS,
        ordered=True,
    )

    dataframe["age_group"] = pd.cut(
        dataframe["age"],
        bins=AGE_BINS,
        labels=AGE_LABELS,
    )

    dataframe["is_weekend"] = dataframe["day_type"].eq("Weekend")
    dataframe["total_paid"] = (
        dataframe["order_value"] + dataframe["delivery_fee"]
    )

    return dataframe


def build_user_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Build one aggregated feature row for each user."""
    user_features = (
        dataframe.groupby("user_id")
        .agg(
            total_orders=("order_id", "count"),
            avg_order_value=("order_value", "mean"),
            avg_total_paid=("total_paid", "mean"),
            avg_rating_given=("rating_given", "mean"),
            repeat_order_rate=("is_repeat_order", "mean"),
            discount_usage_rate=("discount_applied", "mean"),
            weekend_order_rate=("is_weekend", "mean"),
            favorite_cuisine=("cuisine", lambda series: series.mode().iat[0]),
            favorite_meal_type=(
                "meal_type",
                lambda series: series.mode().iat[0],
            ),
            favorite_city=("city", lambda series: series.mode().iat[0]),
        )
        .reset_index()
    )

    static_features = (
        dataframe.groupby("user_id")[["age", "age_group"]]
        .first()
        .reset_index()
    )

    return user_features.merge(
        static_features,
        on="user_id",
        how="left",
    )


def main() -> None:
    """Clean the order data and generate user-level features."""
    if not RAW_PATH.exists():
        raise SystemExit(
            f"{RAW_PATH} not found. "
            "Run 01_fetch_kaggle_orders.py first."
        )

    raw_data = pd.read_parquet(RAW_PATH)

    clean_data = clean(raw_data)
    clean_data.to_parquet(CLEAN_PATH, index=False)

    print(
        f"Cleaned row-level data saved: {CLEAN_PATH} "
        f"({clean_data.shape[0]} rows)"
    )

    user_features = build_user_features(clean_data)
    user_features.to_parquet(USER_FEATURES_PATH, index=False)

    print(
        f"Per-user feature table saved: {USER_FEATURES_PATH} "
        f"({user_features.shape[0]} users)"
    )

    print("\n--- User feature sample ---")
    print(user_features.head().to_string())


if __name__ == "__main__":
    main()