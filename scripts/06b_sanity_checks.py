"""
Step 6b: Quick sanity checks on the loaded Postgres data - joins, aggregates,
and spot checks across all 4 tables before we build embeddings/agents on top.

RUN:
   python scripts/06b_sanity_checks.py
"""

import os

from dotenv import load_dotenv
import pandas as pd
from sqlalchemy import create_engine, text

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
        "Orders <-> user_features consistency: total_orders should match actual order count",
        """
        SELECT uf.user_id, uf.total_orders AS stated_orders, COUNT(o.order_id) AS actual_orders
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
        SELECT city, COUNT(*) AS order_count, ROUND(AVG(order_value), 2) AS avg_order_value
        FROM orders
        GROUP BY city
        ORDER BY order_count DESC
        LIMIT 5;
        """,
    ),
    (
        "Top 5 cuisines by average rating given",
        """
        SELECT cuisine, COUNT(*) AS n, ROUND(AVG(rating_given), 2) AS avg_rating
        FROM orders
        GROUP BY cuisine
        ORDER BY avg_rating DESC
        LIMIT 5;
        """,
    ),
    (
        "Review rating distribution by app",
        """
        SELECT app, rating, COUNT(*) AS n
        FROM reviews
        GROUP BY app, rating
        ORDER BY app, rating;
        """,
    ),
    (
        "Review date range (sanity check on scrape recency)",
        """
        SELECT app, MIN(review_date) AS earliest, MAX(review_date) AS latest, COUNT(*) AS n
        FROM reviews
        GROUP BY app;
        """,
    ),
    (
        "PDF chunks per company/source (spot check against parse step output)",
        """
        SELECT company, source_file, COUNT(*) AS n_chunks, AVG(word_count) AS avg_words
        FROM investor_pdf_chunks
        GROUP BY company, source_file
        ORDER BY company, source_file;
        """,
    ),
    (
        "Any NULL/empty chunk_text in PDF chunks? (should be 0)",
        """
        SELECT COUNT(*) AS bad_chunks
        FROM investor_pdf_chunks
        WHERE chunk_text IS NULL OR TRIM(chunk_text) = '';
        """,
    ),
]


if __name__ == "__main__":
    engine = create_engine(DB_URL)

    with engine.connect() as conn:
        for title, query in CHECKS:
            print(f"\n=== {title} ===")
            df = pd.read_sql(text(query), conn)
            if df.empty:
                print("  (no rows returned)")
            else:
                print(df.to_string(index=False))