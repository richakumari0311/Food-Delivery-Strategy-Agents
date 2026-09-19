"""
Daily usage limiter, backed by Postgres (not an in-memory counter).

Why DB-backed: Streamlit Cloud can restart or redeploy your app, which
would silently reset an in-memory counter to zero, undoing the whole
point of the limit. A counter row in the database survives restarts and
works correctly even if Streamlit ever runs multiple instances.

Used to cap expensive operations (especially the full 5-agent strategy
run, ~8-9 Gemini calls) so one public app doesn't burn a whole day's
quota from a handful of visitors.
"""

from datetime import date

import psycopg2

from src.utils.db import get_db_config

_TABLE_READY = False


def _ensure_table(conn):
    global _TABLE_READY
    if _TABLE_READY:
        return
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS app_usage_counters (
                usage_date DATE NOT NULL,
                counter_key TEXT NOT NULL,
                count INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (usage_date, counter_key)
            )
        """)
    conn.commit()
    _TABLE_READY = True


def check_and_increment(counter_key: str, daily_limit: int) -> tuple[bool, int]:
    """Returns (allowed, current_count_after_this_attempt).
    Atomically increments only if under the limit, so concurrent requests
    can't both slip through right at the boundary."""
    conn = psycopg2.connect(**get_db_config())
    try:
        _ensure_table(conn)
        today = date.today()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT count FROM app_usage_counters WHERE usage_date = %s AND counter_key = %s",
                (today, counter_key),
            )
            row = cur.fetchone()
            current = row[0] if row else 0

            if current >= daily_limit:
                return False, current

            cur.execute("""
                INSERT INTO app_usage_counters (usage_date, counter_key, count)
                VALUES (%s, %s, 1)
                ON CONFLICT (usage_date, counter_key)
                DO UPDATE SET count = app_usage_counters.count + 1
            """, (today, counter_key))
        conn.commit()
        return True, current + 1
    finally:
        conn.close()


def get_current_count(counter_key: str) -> int:
    conn = psycopg2.connect(**get_db_config())
    try:
        _ensure_table(conn)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT count FROM app_usage_counters WHERE usage_date = %s AND counter_key = %s",
                (date.today(), counter_key),
            )
            row = cur.fetchone()
        return row[0] if row else 0
    finally:
        conn.close()