"""
Step 3: Pull Swiggy & Zomato Play Store reviews for the Review Analysis Agent.

SETUP:
   pip install google-play-scraper pandas pyarrow tqdm

RUN:
   python scripts/03_scrape_playstore_reviews.py

Notes:
- No API key needed - google-play-scraper reads the public Play Store review pages.
- Default pulls ~2000 "most relevant" reviews per app. Bump COUNT_PER_APP if you
  want more, but going very high increases the chance of rate limiting.
- Zomato's app changed its package/brand to "Eternal" on the business side, but
  the Play Store package name (com.application.zomato) hasn't changed - the
  scraper uses the package id, not the brand name.
"""

from pathlib import Path

import pandas as pd
from google_play_scraper import Sort, reviews

APPS = {
    "swiggy": "in.swiggy.android",
    "zomato": "com.application.zomato",
}

COUNT_PER_APP = 2000
LANG = "en"
COUNTRY = "in"

OUT_DIR = Path("data/raw/playstore_reviews")


def fetch_reviews(app_name: str, package_id: str, count: int) -> pd.DataFrame:
    print(f"Fetching {count} reviews for {app_name} ({package_id}) ...")
    result, _ = reviews(
        package_id,
        lang=LANG,
        country=COUNTRY,
        sort=Sort.NEWEST,
        count=count,
    )
    df = pd.DataFrame(result)
    df["app"] = app_name
    print(f"  -> got {len(df)} reviews")
    return df


def trim_columns(df: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "app", "reviewId", "userName", "score", "content",
        "thumbsUpCount", "reviewCreatedVersion", "at", "appVersion",
    ]
    keep = [c for c in keep if c in df.columns]
    return df[keep].rename(columns={
        "score": "rating",
        "content": "review_text",
        "at": "review_date",
        "thumbsUpCount": "helpful_votes",
    })


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_reviews = []
    for app_name, package_id in APPS.items():
        try:
            df = fetch_reviews(app_name, package_id, COUNT_PER_APP)
            df = trim_columns(df)
            out_path = OUT_DIR / f"{app_name}_reviews.parquet"
            df.to_parquet(out_path, index=False)
            print(f"  saved -> {out_path}")
            all_reviews.append(df)
        except Exception as e:
            print(f"  FAILED for {app_name}: {e}")

    if all_reviews:
        combined = pd.concat(all_reviews, ignore_index=True)
        combined_path = OUT_DIR / "all_reviews.parquet"
        combined.to_parquet(combined_path, index=False)
        print(f"\nCombined file saved: {combined_path}  ({combined.shape[0]} rows total)")

        print("\n--- Rating distribution by app ---")
        print(combined.groupby("app")["rating"].value_counts().unstack(fill_value=0))