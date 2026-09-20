"""Shared Postgres connection configuration for the project."""

import os

from dotenv import load_dotenv

load_dotenv()


def get_db_config() -> dict[str, str]:
    """Return connection settings for psycopg2."""
    return {
        "dbname": os.getenv("POSTGRES_DB", "food_delivery"),
        "user": os.getenv("POSTGRES_USER", "app"),
        "password": os.getenv("POSTGRES_PASSWORD", "app_password"),
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": os.getenv("POSTGRES_PORT", "5432"),
        "sslmode": os.getenv("POSTGRES_SSLMODE", "prefer"),
    }


def get_db_url() -> str:
    """Return the SQLAlchemy connection URL."""
    config = get_db_config()

    return (
        f"postgresql+psycopg2://{config['user']}:{config['password']}@"
        f"{config['host']}:{config['port']}/{config['dbname']}"
        f"?sslmode={config['sslmode']}"
    )