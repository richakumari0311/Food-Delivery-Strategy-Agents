"""Retry decorators for transient LLM and Postgres failures."""

import logging
import time
from collections.abc import Callable
from functools import wraps
from typing import TypeVar

import psycopg2

logger = logging.getLogger(__name__)

T = TypeVar("T")

RETRYABLE_LLM_ERROR_SUBSTRINGS = (
    "RESOURCE_EXHAUSTED",
    "UNAVAILABLE",
    "DEADLINE_EXCEEDED",
    "INTERNAL",
    "503",
    "429",
)


def with_llm_retry(
    max_attempts: int = 3,
    initial_wait: float = 5.0,
    backoff_factor: float = 3.0,
):
    """Retry transient LLM failures with exponential backoff."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            wait = initial_wait
            last_exception: Exception | None = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    last_exception = exc
                    error_text = str(exc)

                    is_retryable = any(
                        substring in error_text
                        for substring in RETRYABLE_LLM_ERROR_SUBSTRINGS
                    )

                    if not is_retryable or attempt == max_attempts:
                        break

                    logger.warning(
                        "[%s] attempt %d/%d failed (%s). Retrying in %.0fs.",
                        func.__name__,
                        attempt,
                        max_attempts,
                        error_text[:100],
                        wait,
                    )

                    time.sleep(wait)
                    wait *= backoff_factor

            error_text = str(last_exception)
            is_quota_error = (
                "RESOURCE_EXHAUSTED" in error_text
                or "429" in error_text
            )

            quota_hint = ""
            if is_quota_error:
                quota_hint = (
                    "\nThis looks like a quota/rate-limit error. If retries "
                    "didn't help, the daily quota may be exhausted. Check "
                    "the Gemini rate-limit dashboard or use a higher-quota "
                    "model via GEMINI_MODEL."
                )

            raise RuntimeError(
                f"{func.__name__} failed after {max_attempts} attempt(s). "
                f"Last error: {last_exception}{quota_hint}"
            ) from last_exception

        return wrapper

    return decorator


def with_db_retry(
    max_attempts: int = 3,
    wait_seconds: float = 2.0,
):
    """Retry transient Postgres connection failures."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception: Exception | None = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except psycopg2.OperationalError as exc:
                    last_exception = exc

                    if attempt == max_attempts:
                        break

                    logger.warning(
                        "[%s] DB connection attempt %d/%d failed. "
                        "Retrying in %.0fs.",
                        func.__name__,
                        attempt,
                        max_attempts,
                        wait_seconds,
                    )

                    time.sleep(wait_seconds)

            raise RuntimeError(
                f"{func.__name__} could not connect to Postgres after "
                f"{max_attempts} attempt(s). Check the database and "
                f"your .env credentials.\n"
                f"Last error: {last_exception}"
            ) from last_exception

        return wrapper

    return decorator