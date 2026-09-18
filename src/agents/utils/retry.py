"""
Shared retry decorator for LLM and DB calls, used across all 5 agents.

Two decorators, deliberately different in behavior:

- with_llm_retry: for Gemini/LangChain calls. Retries transient errors
  (RESOURCE_EXHAUSTED, UNAVAILABLE, DEADLINE_EXCEEDED, INTERNAL) with
  exponential backoff, but caps at a SMALL number of attempts with a
  clear final error. This is intentional: a 429 can be either a short
  per-minute burst limit (worth retrying) or a fully exhausted DAILY
  quota (retrying is pointless and just wastes time before failing
  anyway). We can't tell which from the error alone, so we retry a
  little, then fail fast with a message pointing at the quota dashboard
  rather than hanging or retrying indefinitely.

- with_db_retry: for Postgres connection calls. Retries OperationalError
  (e.g. Docker container still starting up) a few times with short,
  fixed delays - this is a genuinely transient, quickly-resolved
  situation, unlike LLM quota exhaustion, so it can retry more eagerly.
"""

import logging
import time
from functools import wraps
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)
T = TypeVar("T")

# Substrings from Gemini/LangChain error messages worth retrying.
# RESOURCE_EXHAUSTED is included but capped low - see module docstring.
RETRYABLE_LLM_ERROR_SUBSTRINGS = (
    "RESOURCE_EXHAUSTED",
    "UNAVAILABLE",
    "DEADLINE_EXCEEDED",
    "INTERNAL",
    "503",
    "429",
)


def with_llm_retry(max_attempts: int = 3, initial_wait: float = 5.0, backoff_factor: float = 3.0):
    """Retry an LLM call a few times with exponential backoff, then fail
    with a clear message. Does NOT retry indefinitely on quota errors -
    if you're hitting a genuinely exhausted daily quota, more retries
    won't help; this fails fast and tells you where to check."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            wait = initial_wait
            last_exception: Exception | None = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    error_str = str(e)
                    is_retryable = any(s in error_str for s in RETRYABLE_LLM_ERROR_SUBSTRINGS)

                    if not is_retryable or attempt == max_attempts:
                        break

                    print(
                        f"  [{func.__name__}] attempt {attempt}/{max_attempts} failed "
                        f"({error_str[:100]}...). Retrying in {wait:.0f}s..."
                    )
                    logger.warning(f"{func.__name__} retry {attempt}/{max_attempts}: {error_str[:200]}")
                    time.sleep(wait)
                    wait *= backoff_factor

            is_quota = last_exception is not None and (
                "RESOURCE_EXHAUSTED" in str(last_exception) or "429" in str(last_exception)
            )
            quota_hint = (
                "\nThis looks like a quota/rate-limit error. If retries didn't help, "
                "this is likely a fully exhausted DAILY quota, not a transient burst - "
                "check https://ai.dev/rate-limit rather than retrying more. Consider "
                "setting GEMINI_MODEL to a higher-quota model in .env while iterating."
                if is_quota else ""
            )
            raise RuntimeError(
                f"{func.__name__} failed after {max_attempts} attempt(s). "
                f"Last error: {last_exception}{quota_hint}"
            ) from last_exception

        return wrapper

    return decorator


def with_db_retry(max_attempts: int = 3, wait_seconds: float = 2.0):
    """Retry a Postgres connection a few times with a short fixed delay -
    for transient situations like Docker still starting up, not quota
    issues, so eager fixed-interval retries make sense here."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            import psycopg2

            last_exception: Exception | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except psycopg2.OperationalError as e:
                    last_exception = e
                    if attempt == max_attempts:
                        break
                    print(
                        f"  [{func.__name__}] DB connection attempt {attempt}/{max_attempts} "
                        f"failed. Retrying in {wait_seconds:.0f}s... "
                        f"(Is Docker running? `docker compose up -d`)"
                    )
                    time.sleep(wait_seconds)

            raise RuntimeError(
                f"{func.__name__} could not connect to Postgres after {max_attempts} attempt(s). "
                f"Check `docker compose up -d` and your .env credentials.\n"
                f"Last error: {last_exception}"
            ) from last_exception

        return wrapper

    return decorator