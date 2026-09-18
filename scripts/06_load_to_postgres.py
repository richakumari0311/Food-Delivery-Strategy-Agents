"""
Step 6: Apply the schema and load all three processed sources into Postgres.

SETUP:
   1. cp .env.example .env   (adjust credentials if you want)
   2. docker compose up -d
   3. pip install sqlalchemy psycopg2-binary python-dotenv pandas

RUN:
   python 06_load_to_postgres.py
"""

import os
from pathlib import Path

from dotenv import load_dotenv
import pandas as pd
from sqlalchemy import create_engine, text

load_dotenv()

DB_USER = os.getenv("POSTGRES_USER", "app")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "app_password")
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = os.getenv("POSTGRES_PORT", "5433")
DB_NAME = os.getenv("POSTGRES_DB", "food_delivery")

DB_SSLMODE = os.getenv("POSTGRES_SSLMODE", "prefer")

DB_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}?sslmode={DB_SSLMODE}"

SCHEMA_FILE = Path("schema.sql")

ORDERS_PATH = Path("data/processed/kaggle_orders_clean.parquet")
USER_FEATURES_PATH = Path("data/processed/kaggle_user_features.parquet")
REVIEWS_PATH = Path("data/raw/playstore_reviews/all_reviews.parquet")
PDF_CHUNKS_PATH = Path("data/processed/investor_pdf_chunks.parquet")


def apply_schema(engine):
    print(f"Applying schema from {SCHEMA_FILE} ...")
    sql = SCHEMA_FILE.read_text()
    with engine.begin() as conn:
        conn.execute(text(sql))
    print("Schema applied.")


def load_table(engine, path: Path, table_name: str, prep_fn=None):
    if not path.exists():
        print(f"  SKIP {table_name}: {path} not found")
        return
    df = pd.read_parquet(path)
    if prep_fn:
        df = prep_fn(df)
    df.to_sql(table_name, engine, if_exists="append", index=False, method="multi", chunksize=1000)
    print(f"  loaded {table_name}: {len(df)} rows from {path.name}")


def prep_reviews(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={
        "reviewId": "review_id",
        "userName": "user_name",
        "reviewCreatedVersion": "review_created_version",
        "appVersion": "app_version",
    })
    if "review_date" in df.columns:
        df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")
    return df


if __name__ == "__main__":
    engine = create_engine(DB_URL)

    # Quick connectivity check with a clear error if Docker isn't up yet
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as e:
        raise SystemExit(
            f"Could not connect to Postgres at {DB_HOST}:{DB_PORT}.\n"
            f"Is Docker running? Try: docker compose up -d\n\nOriginal error: {e}"
        )

    apply_schema(engine)

    print("\nLoading tables ...")
    load_table(engine, ORDERS_PATH, "orders")
    load_table(engine, USER_FEATURES_PATH, "user_features")
    load_table(engine, REVIEWS_PATH, "reviews", prep_fn=prep_reviews)
    load_table(engine, PDF_CHUNKS_PATH, "investor_pdf_chunks")

    print("\nRow counts in DB:")
    with engine.connect() as conn:
        for table in ["orders", "user_features", "reviews", "investor_pdf_chunks"]:
            count = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
            print(f"  {table}: {count}")