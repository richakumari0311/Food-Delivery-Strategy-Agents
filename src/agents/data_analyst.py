"""
Data Analyst Agent (standalone, pre-LangGraph).

Given a question about order/user behavior, this:
  1. Asks Gemini to write a SQL query against the orders/user_features schema
     (structured output - not free text - so we get a clean query string)
  2. Validates the query is read-only and safe before running it
  3. Executes it and gets real rows back
  4. Asks Gemini again to turn those actual rows into a structured
     narrative finding - grounded in the real numbers, not guessed

SETUP:
   pip install langchain-google-genai pydantic psycopg2-binary python-dotenv pandas

RUN:
   python agents/data_analyst_agent.py
"""

import os
import re
from typing import Optional

from dotenv import load_dotenv
import pandas as pd
import psycopg2
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

from src.utils.retry import with_llm_retry, with_db_retry

load_dotenv()

# Override in .env, e.g. GEMINI_MODEL=gemini-3.5-flash-lite for a much higher
# free-tier daily quota (~500/day vs ~20/day) while iterating.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

DB_CONFIG = dict(
    dbname=os.getenv("POSTGRES_DB", "food_delivery"),
    user=os.getenv("POSTGRES_USER", "app"),
    password=os.getenv("POSTGRES_PASSWORD", "app_password"),
    host=os.getenv("POSTGRES_HOST", "localhost"),
    port=os.getenv("POSTGRES_PORT", "5432"),
)

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
"""

# Guardrails: block anything that isn't a single read-only SELECT
FORBIDDEN_PATTERN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|GRANT|REVOKE|CREATE|COPY)\b",
    re.IGNORECASE,
)


class SQLGeneration(BaseModel):
    sql_query: str = Field(description="A single read-only PostgreSQL SELECT query answering the question")
    explanation: str = Field(description="One sentence on what the query computes")


class DataAnalysisResult(BaseModel):
    summary: str = Field(description="2-3 sentence answer to the question, grounded in the actual query results")
    key_findings: list[str] = Field(description="3-5 specific numeric findings from the data")
    caveats: Optional[str] = Field(default=None, description="Any data quality caveats relevant to this answer")


def validate_sql(sql: str) -> str:
    sql = sql.strip().rstrip(";")
    if not sql.upper().startswith("SELECT"):
        raise ValueError(f"Rejected non-SELECT query: {sql[:80]}")
    if ";" in sql:
        raise ValueError("Rejected multi-statement query")
    if FORBIDDEN_PATTERN.search(sql):
        raise ValueError(f"Rejected query containing forbidden keyword: {sql[:80]}")
    if "LIMIT" not in sql.upper():
        sql = f"{sql} LIMIT {MAX_ROWS}"
    return sql


@with_db_retry()
def run_query(sql: str) -> pd.DataFrame:
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute("SET TRANSACTION READ ONLY")  # defense in depth beyond the regex check
            cur.execute(sql)
            cols = [desc[0] for desc in cur.description]
            rows = cur.fetchall()
        return pd.DataFrame(rows, columns=cols)
    finally:
        conn.close()


@with_llm_retry()
def _generate_sql(llm, prompt: str) -> SQLGeneration:
    return llm.with_structured_output(SQLGeneration).invoke(prompt)


@with_llm_retry()
def _generate_analysis(llm, prompt: str) -> DataAnalysisResult:
    return llm.with_structured_output(DataAnalysisResult).invoke(prompt)


def analyze(question: str):
    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0)

    sql_prompt = (
        "You write PostgreSQL SELECT queries against this schema:\n"
        f"{SCHEMA_DESCRIPTION}\n\n"
        f"QUESTION: {question}\n\n"
        "Write ONE read-only SELECT query that answers it. Prefer aggregates "
        "(GROUP BY, AVG, COUNT) over raw row dumps."
    )
    sql_gen = _generate_sql(llm, sql_prompt)

    safe_sql = validate_sql(sql_gen.sql_query)
    print(f"Generated SQL:\n  {safe_sql}\n")

    df = run_query(safe_sql)
    print(f"Returned {len(df)} rows\n")

    if df.empty:
        return DataAnalysisResult(
            summary="The query returned no rows.",
            key_findings=[],
            caveats="Query may be too restrictive, or no matching data exists.",
        ), df, safe_sql

    results_block = df.head(50).to_string(index=False)
    analysis_prompt = (
        f"QUESTION: {question}\n\n"
        f"SQL USED: {safe_sql}\n\n"
        f"QUERY RESULTS ({len(df)} rows, showing up to 50):\n{results_block}\n\n"
        f"{SCHEMA_DESCRIPTION}\n"
        "Answer the question using ONLY these results. Cite actual numbers."
    )
    result = _generate_analysis(llm, analysis_prompt)

    return result, df, safe_sql


if __name__ == "__main__":
    question = "Which cities have the highest average order value, and how does repeat order rate vary by city?"
    print(f"Question: {question}\n")

    result, df, sql = analyze(question)

    print("--- Structured analysis ---")
    print(result.model_dump_json(indent=2))