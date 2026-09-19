"""
Shared Postgres connection config. Centralized here so switching the whole
project from local Docker Postgres to a hosted provider (e.g. Supabase) is
just an .env change, not edits scattered across every agent/script.

sslmode defaults to 'prefer' (works fine against local Docker Postgres,
which doesn't enforce SSL). Supabase and most hosted Postgres providers
REQUIRE SSL, so set POSTGRES_SSLMODE=require in .env when pointing at one.
"""

import os

from dotenv import load_dotenv

load_dotenv()


def get_db_config() -> dict:
    """For psycopg2.connect(**config)."""
    return dict(
        dbname=os.getenv("POSTGRES_DB", "food_delivery"),
        user=os.getenv("POSTGRES_USER", "app"),
        password=os.getenv("POSTGRES_PASSWORD", "app_password"),
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        sslmode=os.getenv("POSTGRES_SSLMODE", "prefer"),
    )


def get_db_url() -> str:
    """For SQLAlchemy (pandas.to_sql/read_sql, create_engine)."""
    cfg = get_db_config()
    return (
        f"postgresql+psycopg2://{cfg['user']}:{cfg['password']}@"
        f"{cfg['host']}:{cfg['port']}/{cfg['dbname']}?sslmode={cfg['sslmode']}"
    )