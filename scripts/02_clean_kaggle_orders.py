"""
Step 2: Clean & type the Kaggle orders dataset, derive a few useful
features, and build a per-user aggregate table for the Segmentation Agent.

Input:  data/processed/kaggle_orders_raw.parquet   (from step 1)
Output: data/processed/kaggle_orders_clean.parquet  (row-level, typed)
        data/processed/kaggle_user_features.parquet (one row per user)

RUN:
   python scripts/02_clean_kaggle_orders.py
"""

from pathlib import Path

import pandas as pd

RAW_PATH = Path("data/processed/kaggle_orders_raw.parquet")
CLEAN_PATH = Path("data/processed/kaggle_orders_clean.parquet")
USER_FEATURES_PATH = Path("data/processed/kaggle_user_features.parquet")

BOOL_YES_NO_COLS = ["discount_applied", "is_repeat_order", "rainy_weather"]
CATEGORY_COLS = [
    "city", "order_time", "day_type", "cuisine", "meal_type",
    "restaurant_type", "mood", "hunger_level", "company",
]

AGE_BINS = [0, 24, 34, 44, 200]
AGE_LABELS = ["18-24", "25-34", "35-44", "45+"]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Yes/No -> proper booleans
    for col in BOOL_YES_NO_COLS:
        df[col] = df[col].map({"Yes": True, "No": False}).astype("boolean")

    # Categorical typing (saves memory, makes groupby/plots faster)
    for col in CATEGORY_COLS:
        df[col] = df[col].astype("category")

    # Ordinal encode fields that have a natural order
    df["restaurant_type"] = pd.Categorical(
        df["restaurant_type"], categories=["Budget", "Mid-range", "Premium"], ordered=True
    )
    df["hunger_level"] = pd.Categorical(
        df["hunger_level"], categories=["Low", "Medium", "High"], ordered=True
    )

    # Derived features
    df["age_group"] = pd.cut(df["age"], bins=AGE_BINS, labels=AGE_LABELS)
    df["is_weekend"] = df["day_type"].eq("Weekend")
    df["total_paid"] = df["order_value"] + df["delivery_fee"]

    return df


def build_user_features(df: pd.DataFrame) -> pd.DataFrame:
    """One row per user - the input table for the Segmentation Agent."""
    agg = df.groupby("user_id").agg(
        total_orders=("order_id", "count"),
        avg_order_value=("order_value", "mean"),
        avg_total_paid=("total_paid", "mean"),
        avg_rating_given=("rating_given", "mean"),
        repeat_order_rate=("is_repeat_order", "mean"),
        discount_usage_rate=("discount_applied", "mean"),
        weekend_order_rate=("is_weekend", "mean"),
        favorite_cuisine=("cuisine", lambda s: s.mode().iat[0]),
        favorite_meal_type=("meal_type", lambda s: s.mode().iat[0]),
        favorite_city=("city", lambda s: s.mode().iat[0]),
    ).reset_index()

    # attach static per-user attributes (age, age_group) - take first occurrence
    static = df.groupby("user_id")[["age", "age_group"]].first().reset_index()
    agg = agg.merge(static, on="user_id", how="left")

    return agg


if __name__ == "__main__":
    if not RAW_PATH.exists():
        raise SystemExit(f"{RAW_PATH} not found - run 01_fetch_kaggle_orders.py first")

    raw = pd.read_parquet(RAW_PATH)
    clean_df = clean(raw)
    clean_df.to_parquet(CLEAN_PATH, index=False)
    print(f"Cleaned row-level data saved: {CLEAN_PATH}  ({clean_df.shape[0]} rows)")

    user_features = build_user_features(clean_df)
    user_features.to_parquet(USER_FEATURES_PATH, index=False)
    print(f"Per-user feature table saved: {USER_FEATURES_PATH}  ({user_features.shape[0]} users)")

    print("\n--- User feature sample ---")
    print(user_features.head(5).to_string())