"""Run sanity checks against the loaded Postgres data."""

import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


load_dotenv()


DB_URL = (
    f"postgresql+psycopg2://{os.getenv('POSTGRES_USER', 'app')}:"
    f"{os.getenv('POSTGRES_PASSWORD', 'app_password')}@"
    f"{os.getenv('POSTGRES_HOST', 'localhost')}:"
    f"{os.getenv('POSTGRES_PORT', '5432')}/"
    f"{os.getenv('POSTGRES_DB', 'food_delivery')}"
    f"?sslmode={os.getenv('POSTGRES_SSLMODE', 'prefer')}"
)


CHECKS = [
    (
        "Orders <-> user_features consistency",
        """
        SELECT
            uf.user_id,
            uf.total_orders AS stated_orders,
            COUNT(o.order_id) AS actual_orders
        FROM user_features uf
        JOIN orders o ON o.user_id = uf.user_id
        GROUP BY uf.user_id, uf.total_orders
        HAVING uf.total_orders != COUNT(o.order_id)
        LIMIT 5;
        """,
    ),
    (
        "Top 5 cities by order volume",
        """
        SELECT
            city,
            COUNT(*) AS order_count,
            ROUND(AVG(order_value), 2) AS avg_order_value
        FROM orders
        GROUP BY city
        ORDER BY order_count DESC
        LIMIT 5;
        """,
    ),
    (
        "Top 5 cuisines by average rating",
        """
        SELECT
            cuisine,
            COUNT(*) AS order_count,
            ROUND(AVG(rating_given), 2) AS avg_rating
        FROM orders
        GROUP BY cuisine
        ORDER BY avg_rating DESC
        LIMIT 5;
        """,
    ),
    (
        "Review rating distribution by app",
        """
        SELECT
            app,
            rating,
            COUNT(*) AS review_count
        FROM reviews
        GROUP BY app, rating
        ORDER BY app, rating;
        """,
    ),
    (
        "Review date range",
        """
        SELECT
            app,
            MIN(review_date) AS earliest,
            MAX(review_date) AS latest,
            COUNT(*) AS review_count
        FROM reviews
        GROUP BY app;
        """,
    ),
    (
        "PDF chunks per company/source",
        """
        SELECT
            company,
            source_file,
            COUNT(*) AS chunk_count,
            AVG(word_count) AS avg_words
        FROM investor_pdf_chunks
        GROUP BY company, source_file
        ORDER BY company, source_file;
        """,
    ),
    (
        "NULL or empty PDF chunks",
        """
        SELECT
            COUNT(*) AS bad_chunks
        FROM investor_pdf_chunks
        WHERE chunk_text IS NULL
            OR TRIM(chunk_text) = '';
        """,
    ),
]


def run_checks(engine: Engine) -> None:
    """Run all configured sanity-check queries."""
    with engine.connect() as connection:
        for title, query in CHECKS:
            print(f"\n=== {title} ===")

            dataframe = pd.read_sql(
                text(query),
                connection,
            )

            if dataframe.empty:
                print("  (no rows returned)")
            else:
                print(dataframe.to_string(index=False))


def main() -> None:
    """Run Postgres sanity checks."""
    engine = create_engine(DB_URL)
    run_checks(engine)


if __name__ == "__main__":
    main()