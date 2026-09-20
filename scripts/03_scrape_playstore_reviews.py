"""Fetch Swiggy and Zomato Play Store reviews for the Review Analysis agent."""

from pathlib import Path

import pandas as pd
from google_play_scraper import Sort, reviews


APPS = {
    "swiggy": "in.swiggy.android",
    "zomato": "com.application.zomato",
}

COUNT_PER_APP = 2000
LANGUAGE = "en"
COUNTRY = "in"

OUT_DIR = Path("data/raw/playstore_reviews")

REVIEW_COLUMNS = [
    "app",
    "reviewId",
    "userName",
    "score",
    "content",
    "thumbsUpCount",
    "reviewCreatedVersion",
    "at",
    "appVersion",
]

COLUMN_RENAME_MAP = {
    "score": "rating",
    "content": "review_text",
    "at": "review_date",
    "thumbsUpCount": "helpful_votes",
}


def fetch_reviews(
    app_name: str,
    package_id: str,
    count: int,
) -> pd.DataFrame:
    """Fetch Play Store reviews for an app."""
    print(
        f"Fetching {count} reviews for "
        f"{app_name} ({package_id})..."
    )

    result, _ = reviews(
        package_id,
        lang=LANGUAGE,
        country=COUNTRY,
        sort=Sort.NEWEST,
        count=count,
    )

    dataframe = pd.DataFrame(result)
    dataframe["app"] = app_name

    print(f"  -> got {len(dataframe)} reviews")

    return dataframe


def trim_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Keep required review fields and standardize their names."""
    available_columns = [
        column
        for column in REVIEW_COLUMNS
        if column in dataframe.columns
    ]

    return dataframe[available_columns].rename(
        columns=COLUMN_RENAME_MAP
    )


def save_reviews(
    dataframe: pd.DataFrame,
    app_name: str,
) -> Path:
    """Save reviews for an individual app as a Parquet file."""
    output_path = OUT_DIR / f"{app_name}_reviews.parquet"
    dataframe.to_parquet(output_path, index=False)

    print(f"  saved -> {output_path}")

    return output_path


def print_rating_distribution(dataframe: pd.DataFrame) -> None:
    """Print the rating distribution for each app."""
    print("\n--- Rating distribution by app ---")
    print(
        dataframe
        .groupby("app")["rating"]
        .value_counts()
        .unstack(fill_value=0)
    )


def main() -> None:
    """Fetch, clean, and save Play Store reviews."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_reviews: list[pd.DataFrame] = []
    failed_apps: list[str] = []

    for app_name, package_id in APPS.items():
        try:
            dataframe = fetch_reviews(
                app_name,
                package_id,
                COUNT_PER_APP,
            )
            dataframe = trim_columns(dataframe)

            save_reviews(dataframe, app_name)
            all_reviews.append(dataframe)

        except Exception as exc:
            failed_apps.append(app_name)
            print(f"  FAILED for {app_name}: {exc}")

    if not all_reviews:
        raise RuntimeError("Failed to fetch reviews for all apps.")

    combined = pd.concat(
        all_reviews,
        ignore_index=True,
    )

    combined_path = OUT_DIR / "all_reviews.parquet"
    combined.to_parquet(combined_path, index=False)

    print(
        f"\nCombined file saved: {combined_path} "
        f"({combined.shape[0]} rows total)"
    )

    print_rating_distribution(combined)

    if failed_apps:
        print(
            f"\nWarning: failed to fetch reviews for: "
            f"{', '.join(failed_apps)}"
        )


if __name__ == "__main__":
    main()