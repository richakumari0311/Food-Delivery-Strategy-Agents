"""Postgres-backed daily usage limiter for expensive application operations."""

from datetime import date

import psycopg2

from src.utils.db import get_db_config

_TABLE_READY = False


def _ensure_table(connection) -> None:
    """Create the usage-counter table if it does not already exist."""
    global _TABLE_READY

    if _TABLE_READY:
        return

    with connection.cursor() as cursor:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS app_usage_counters (
                usage_date DATE NOT NULL,
                counter_key TEXT NOT NULL,
                count INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (usage_date, counter_key)
            )
            """
        )

    connection.commit()
    _TABLE_READY = True


def check_and_increment(
    counter_key: str,
    daily_limit: int,
) -> tuple[bool, int]:
    """Atomically increment a counter when it is below the daily limit."""
    connection = psycopg2.connect(**get_db_config())

    try:
        _ensure_table(connection)
        today = date.today()

        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO app_usage_counters (
                    usage_date,
                    counter_key,
                    count
                )
                VALUES (%s, %s, 1)
                ON CONFLICT (usage_date, counter_key)
                DO UPDATE SET count = app_usage_counters.count + 1
                WHERE app_usage_counters.count < %s
                RETURNING count
                """,
                (today, counter_key, daily_limit),
            )

            row = cursor.fetchone()

            if row is not None:
                connection.commit()
                return True, row[0]

            cursor.execute(
                """
                SELECT count
                FROM app_usage_counters
                WHERE usage_date = %s
                  AND counter_key = %s
                """,
                (today, counter_key),
            )
            row = cursor.fetchone()

        connection.commit()
        return False, row[0] if row else 0

    finally:
        connection.close()


def get_current_count(counter_key: str) -> int:
    """Return today's usage count for a counter."""
    connection = psycopg2.connect(**get_db_config())

    try:
        _ensure_table(connection)

        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT count
                FROM app_usage_counters
                WHERE usage_date = %s
                  AND counter_key = %s
                """,
                (date.today(), counter_key),
            )
            row = cursor.fetchone()

        return row[0] if row else 0

    finally:
        connection.close()