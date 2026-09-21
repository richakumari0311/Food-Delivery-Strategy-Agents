"""Data analysis agent using Gemini and PostgreSQL."""

import os
import re
from typing import Any, Optional

import pandas as pd
import psycopg2
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from src.utils.db import get_db_config
from src.utils.retry import with_db_retry, with_llm_retry


load_dotenv()

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
DB_CONFIG = get_db_config()
MAX_ROWS = 200

SCHEMA_DESCRIPTION = """
Table: orders (50,000 rows, one row per order)
  order_id, user_id, age, age_group, city, order_time (Morning/Evening/Night/etc daypart),
  day_type (Weekday/Weekend), is_weekend (bool), cuisine, meal_type, restaurant_type
  (Budget/Mid-range/Premium, ordinal), order_value, discount_applied (bool), delivery_fee,
  total_paid, time_taken_to_order, rating_given (1-5), is_repeat_order (bool), mood,
  hunger_level (Low/Medium/High, ordinal), company (who they ordered with), rainy_weather (bool)

Table: user_features (4,000 rows, one row per user - pre-aggregated)
  user_id, total_orders, avg_order_value, avg_total_paid, avg_rating_given,
  repeat_order_rate, discount_usage_rate, weekend_order_rate, favorite_cuisine,
  favorite_meal_type, favorite_city, age, age_group

IMPORTANT DATA CAVEAT: this is a synthetic/generated dataset for pipeline testing.
`rating_given` in particular has been verified to carry NO real signal (near-identical
~3.0 average across every cuisine/city). Do not draw conclusions from rating_given
patterns - if a question is specifically about ratings driving some other variable,
note this limitation rather than reporting spurious correlation as insight.

CRITICAL: if a question asks about real business/financial metrics (revenue, profit,
quarterly earnings, actual company performance), you MUST explicitly state in your
summary and caveats that any number computed from this table (e.g. SUM(order_value))
is a sum over synthetic pipeline-test data, NOT the company's actual real-world
revenue or financial performance. Never present such a number as if it answers a real
business question about the company's true finances.
"""

FORBIDDEN_PATTERN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|GRANT|REVOKE|CREATE|COPY)\b",
    re.IGNORECASE,
)


class SQLGeneration(BaseModel):
    """Structured output generated from the SQL generation step."""

    sql_query: str = Field(
        description="A single read-only PostgreSQL SELECT query."
    )
    explanation: str = Field(
        description="One sentence explaining what the query computes."
    )


class DataAnalysisResult(BaseModel):
    """Structured output generated from query results."""

    summary: str = Field(
        description=(
            "A 2-3 sentence answer to the question, "
            "grounded in the actual query results."
        )
    )
    key_findings: list[str] = Field(
        description="3-5 specific numeric findings from the data."
    )
    caveats: Optional[str] = Field(
        default=None,
        description="Relevant data quality or interpretation caveats.",
    )


def validate_sql(sql: str) -> str:
    """Validate a SQL query and enforce a maximum result size."""

    sql = sql.strip().rstrip(";")

    if not sql.upper().startswith("SELECT"):
        raise ValueError(f"Rejected non-SELECT query: {sql[:80]}")

    if ";" in sql:
        raise ValueError("Rejected multi-statement query")

    if FORBIDDEN_PATTERN.search(sql):
        raise ValueError(
            f"Rejected query containing forbidden keyword: {sql[:80]}"
        )

    if "LIMIT" not in sql.upper():
        sql = f"{sql} LIMIT {MAX_ROWS}"

    return sql


@with_db_retry()
def run_query(sql: str) -> pd.DataFrame:
    """Execute a read-only SQL query and return the results as a DataFrame."""

    conn = psycopg2.connect(**DB_CONFIG)

    try:
        with conn.cursor() as cursor:
            cursor.execute("SET TRANSACTION READ ONLY")
            cursor.execute(sql)

            columns = [description[0] for description in cursor.description]
            rows = cursor.fetchall()

        return pd.DataFrame(rows, columns=columns)
    finally:
        conn.close()


@with_llm_retry()
def _generate_sql(
    llm: ChatGoogleGenerativeAI,
    prompt: str,
) -> SQLGeneration:
    """Generate a structured SQL query using Gemini."""

    return llm.with_structured_output(SQLGeneration).invoke(prompt)


@with_llm_retry()
def _generate_analysis(
    llm: ChatGoogleGenerativeAI,
    prompt: str,
) -> DataAnalysisResult:
    """Generate a structured analysis from query results."""

    return llm.with_structured_output(DataAnalysisResult).invoke(prompt)


def _create_llm() -> ChatGoogleGenerativeAI:
    """Create the configured Gemini client."""

    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=0,
        timeout=60,
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )


def _build_sql_prompt(question: str) -> str:
    """Build the initial SQL generation prompt."""

    return (
        "You write PostgreSQL SELECT queries against this schema:\n"
        f"{SCHEMA_DESCRIPTION}\n\n"
        f"QUESTION: {question}\n\n"
        "Write ONE read-only SELECT query that answers the question. "
        "Prefer aggregates such as GROUP BY, AVG, and COUNT over raw "
        "row dumps."
    )


def _build_retry_prompt(
    question: str,
    sql: str,
    db_error: str,
) -> str:
    """Build a SQL correction prompt using the database error."""

    return (
        "You write PostgreSQL SELECT queries against this schema:\n"
        f"{SCHEMA_DESCRIPTION}\n\n"
        f"QUESTION: {question}\n\n"
        f"PREVIOUS QUERY:\n{sql}\n\n"
        f"DATABASE ERROR:\n{db_error}\n\n"
        "Write a corrected read-only SELECT query. Pay attention to the "
        "actual column types described in the schema. For example, "
        "order_time is a text daypart bucket such as 'Morning' or "
        "'Evening', not a timestamp."
    )


def _build_analysis_prompt(
    question: str,
    sql: str,
    df: pd.DataFrame,
) -> str:
    """Build the prompt used to interpret query results."""

    results_block = df.head(50).to_string(index=False)

    return (
        f"QUESTION: {question}\n\n"
        f"SQL USED: {sql}\n\n"
        f"QUERY RESULTS ({len(df)} rows, showing up to 50):\n"
        f"{results_block}\n\n"
        f"{SCHEMA_DESCRIPTION}\n"
        "Answer the question using ONLY these results. "
        "Cite actual numbers and do not infer unsupported conclusions."
    )


def analyze(
    question: str,
    max_sql_attempts: int = 2,
) -> tuple[DataAnalysisResult, pd.DataFrame, Optional[str]]:
    """Generate, validate, execute, and analyze a SQL query."""

    llm = _create_llm()
    sql_prompt = _build_sql_prompt(question)

    df: Optional[pd.DataFrame] = None
    safe_sql: Optional[str] = None
    last_db_error: Optional[str] = None

    for attempt in range(1, max_sql_attempts + 1):
        sql_generation = _generate_sql(llm, sql_prompt)
        safe_sql = validate_sql(sql_generation.sql_query)

        print(f"Generated SQL (attempt {attempt}):\n  {safe_sql}\n")

        try:
            df = run_query(safe_sql)
            print(f"Returned {len(df)} rows\n")
            break
        except psycopg2.Error as error:
            last_db_error = str(error).strip()
            print(f"SQL execution failed: {last_db_error[:200]}")

            if attempt < max_sql_attempts:
                sql_prompt = _build_retry_prompt(
                    question=question,
                    sql=safe_sql,
                    db_error=last_db_error,
                )

    if df is None:
        return (
            DataAnalysisResult(
                summary=(
                    "Could not answer this question because the "
                    "generated SQL query failed to execute."
                ),
                key_findings=[],
                caveats=(
                    f"Database error after {max_sql_attempts} attempt(s): "
                    f"{last_db_error}"
                ),
            ),
            pd.DataFrame(),
            safe_sql,
        )

    if df.empty:
        return (
            DataAnalysisResult(
                summary="The query returned no rows.",
                key_findings=[],
                caveats=(
                    "The query may be too restrictive, or no matching "
                    "data exists."
                ),
            ),
            df,
            safe_sql,
        )

    analysis_prompt = _build_analysis_prompt(
        question=question,
        sql=safe_sql,
        df=df,
    )

    result = _generate_analysis(llm, analysis_prompt)

    return result, df, safe_sql


if __name__ == "__main__":
    question = (
        "Which cities have the highest average order value, and how does "
        "repeat order rate vary by city?"
    )

    print(f"Question: {question}\n")

    result, df, sql = analyze(question)

    print("--- Structured analysis ---")
    print(result.model_dump_json(indent=2))
