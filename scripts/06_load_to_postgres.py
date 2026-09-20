"""Apply the database schema and load processed data into Postgres."""

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError


load_dotenv()


DB_USER = os.getenv("POSTGRES_USER", "app")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "app_password")
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = os.getenv("POSTGRES_PORT", "5433")
DB_NAME = os.getenv("POSTGRES_DB", "food_delivery")
DB_SSLMODE = os.getenv("POSTGRES_SSLMODE", "prefer")

DB_URL = (
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}"
    f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    f"?sslmode={DB_SSLMODE}"
)

SCHEMA_FILE = Path("schema.sql")

ORDERS_PATH = Path("data/processed/kaggle_orders_clean.parquet")
USER_FEATURES_PATH = Path(
    "data/processed/kaggle_user_features.parquet"
)
REVIEWS_PATH = Path(
    "data/raw/playstore_reviews/all_reviews.parquet"
)
PDF_CHUNKS_PATH = Path(
    "data/processed/investor_pdf_chunks.parquet"
)

TABLES = (
    "orders",
    "user_features",
    "reviews",
    "investor_pdf_chunks",
)


def apply_schema(engine: Engine) -> None:
    """Apply the SQL schema to the Postgres database."""
    print(f"Applying schema from {SCHEMA_FILE} ...")

    sql = SCHEMA_FILE.read_text()

    with engine.begin() as connection:
        connection.execute(text(sql))

    print("Schema applied.")


def load_table(
    engine: Engine,
    path: Path,
    table_name: str,
    prep_fn=None,
) -> None:
    """Load a Parquet file into a Postgres table."""
    if not path.exists():
        print(f"  SKIP {table_name}: {path} not found")
        return

    dataframe = pd.read_parquet(path)

    if prep_fn is not None:
        dataframe = prep_fn(dataframe)

    dataframe.to_sql(
        table_name,
        engine,
        if_exists="append",
        index=False,
        method="multi",
        chunksize=1000,
    )

    print(
        f"  loaded {table_name}: "
        f"{len(dataframe)} rows from {path.name}"
    )


def prep_reviews(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Normalize review column names and date values."""
    dataframe = dataframe.rename(
        columns={
            "reviewId": "review_id",
            "userName": "user_name",
            "reviewCreatedVersion": "review_created_version",
            "appVersion": "app_version",
        }
    )

    if "review_date" in dataframe.columns:
        dataframe["review_date"] = pd.to_datetime(
            dataframe["review_date"],
            errors="coerce",
        )

    return dataframe


def check_connection(engine: Engine) -> None:
    """Verify that Postgres is reachable."""
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except SQLAlchemyError as exc:
        raise SystemExit(
            f"Could not connect to Postgres at "
            f"{DB_HOST}:{DB_PORT}.\n"
            "Is Docker running? Try: docker compose up -d\n\n"
            f"Original error: {exc}"
        ) from exc


def print_row_counts(engine: Engine) -> None:
    """Print row counts for all loaded tables."""
    print("\nRow counts in DB:")

    with engine.connect() as connection:
        for table_name in TABLES:
            count = connection.execute(
                text(f"SELECT COUNT(*) FROM {table_name}")
            ).scalar()

            print(f"  {table_name}: {count}")


def main() -> None:
    """Apply the schema and load all available processed datasets."""
    engine = create_engine(DB_URL)

    check_connection(engine)
    apply_schema(engine)

    print("\nLoading tables ...")

    load_table(
        engine,
        ORDERS_PATH,
        "orders",
    )
    load_table(
        engine,
        USER_FEATURES_PATH,
        "user_features",
    )
    load_table(
        engine,
        REVIEWS_PATH,
        "reviews",
        prep_fn=prep_reviews,
    )
    load_table(
        engine,
        PDF_CHUNKS_PATH,
        "investor_pdf_chunks",
    )

    print_row_counts(engine)


if __name__ == "__main__":
    main()