"""
Consumer Segmentation Agent (standalone, pre-LangGraph).

Unlike the Review/Data Analyst agents, this one is primarily a clustering
task, not an LLM task:
  1. Pull user_features, select behavioral columns, scale them
  2. Auto-select k via silhouette score (no hardcoded cluster count)
  3. Fit KMeans, compute per-cluster statistical profiles
  4. Use the LLM ONCE at the end to turn those profiles into named,
     human-readable personas - it never sees raw user data, only
     aggregated cluster stats, so it can't hallucinate individual behavior
  5. Write cluster assignments back to Postgres as `user_segments`

NOTE: avg_rating_given is deliberately excluded from clustering features -
prior agents confirmed it carries no real signal in this dataset (near-
uniform ~3.0 average everywhere). Including it would just add noise.

SETUP:
   pip install scikit-learn langchain-google-genai pydantic psycopg2-binary python-dotenv pandas

RUN:
   python -m src.agents.segmentation
"""

import os
from typing import Optional

from dotenv import load_dotenv
import pandas as pd
import psycopg2
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

from src.utils.retry import with_llm_retry, with_db_retry
from src.utils.db import get_db_config, get_db_url

load_dotenv()

# Override in .env, e.g. GEMINI_MODEL=gemini-3.5-flash-lite for a much higher
# free-tier daily quota (~500/day vs ~20/day) while iterating.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

DB_CONFIG = get_db_config()

# Deliberately excludes avg_rating_given - see module docstring.
CLUSTER_FEATURES = [
    "total_orders",
    "avg_order_value",
    "repeat_order_rate",
    "discount_usage_rate",
    "weekend_order_rate",
    "age",
]

K_RANGE = range(3, 7)  # try 3-6 clusters, pick the best by silhouette score


class ClusterPersona(BaseModel):
    cluster_id: int
    persona_name: str = Field(description="Short memorable name, e.g. 'Weekend Splurgers'")
    description: str = Field(description="2-3 sentences on who this segment is and how they order")
    key_traits: list[str] = Field(description="3-4 specific numeric/behavioral traits from the profile")


class PersonaSet(BaseModel):
    personas: list[ClusterPersona]


@with_db_retry()
def load_user_features() -> pd.DataFrame:
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        df = pd.read_sql("SELECT * FROM user_features", conn)
    finally:
        conn.close()
    return df


def pick_best_k(X_scaled) -> tuple[int, float]:
    best_k, best_score = None, -1
    for k in K_RANGE:
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        print(f"  k={k}: silhouette={score:.4f}")
        if score > best_score:
            best_k, best_score = k, score
    return best_k, best_score


def build_cluster_profiles(df: pd.DataFrame) -> pd.DataFrame:
    profile = df.groupby("cluster_id").agg(
        n_users=("user_id", "count"),
        avg_total_orders=("total_orders", "mean"),
        avg_order_value=("avg_order_value", "mean"),
        avg_repeat_rate=("repeat_order_rate", "mean"),
        avg_discount_usage=("discount_usage_rate", "mean"),
        avg_weekend_rate=("weekend_order_rate", "mean"),
        avg_age=("age", "mean"),
        top_cuisine=("favorite_cuisine", lambda s: s.mode().iat[0]),
        top_city=("favorite_city", lambda s: s.mode().iat[0]),
    ).round(2).reset_index()
    return profile


@with_llm_retry()
def generate_personas(profile: pd.DataFrame) -> PersonaSet:
    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0.3)
    prompt = (
        "Here are statistical profiles of customer segments from KMeans clustering "
        "on a food delivery app's user base. For EACH cluster_id, give it a short "
        "persona name and description based ONLY on these aggregated stats - do not "
        "invent details not implied by the numbers.\n\n"
        "Naming style: these will appear in a business strategy report for executives, "
        "not a consumer marketing deck. Use formal, analytical language (e.g. 'High-Value "
        "Repeat Customers', 'Price-Sensitive Occasional Users') rather than casual or "
        "playful phrasing (avoid terms like 'Hunters', 'Splurgers', or slang). Keep each "
        "name to 2-4 words.\n\n"
        f"{profile.to_string(index=False)}"
    )
    return llm.with_structured_output(PersonaSet).invoke(prompt)


@with_db_retry()
def save_segments(df: pd.DataFrame, persona_map: dict[int, str]):
    df = df[["user_id", "cluster_id"]].copy()
    df["persona_name"] = df["cluster_id"].map(persona_map)

    conn = psycopg2.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS user_segments")
            cur.execute("""
                CREATE TABLE user_segments (
                    user_id INTEGER PRIMARY KEY,
                    cluster_id INTEGER NOT NULL,
                    persona_name TEXT NOT NULL
                )
            """)
        conn.commit()
    finally:
        conn.close()

    df.to_sql(
        "user_segments",
        get_db_url(),
        if_exists="append",
        index=False,
    )


def run_segmentation():
    df = load_user_features()
    print(f"Loaded {len(df)} users\n")

    X = df[CLUSTER_FEATURES].fillna(0)
    X_scaled = StandardScaler().fit_transform(X)

    print("Selecting k via silhouette score:")
    best_k, best_score = pick_best_k(X_scaled)
    print(f"\nBest k = {best_k} (silhouette = {best_score:.4f})\n")

    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df["cluster_id"] = kmeans.fit_predict(X_scaled)

    profile = build_cluster_profiles(df)
    print("--- Cluster profiles ---")
    print(profile.to_string(index=False))

    print("\nGenerating personas ...")
    persona_set = generate_personas(profile)
    persona_map = {p.cluster_id: p.persona_name for p in persona_set.personas}

    save_segments(df, persona_map)
    print(f"\nSaved cluster assignments for {len(df)} users to `user_segments` table")

    return persona_set, profile


if __name__ == "__main__":
    persona_set, profile = run_segmentation()

    print("\n--- Personas ---")
    for p in persona_set.personas:
        print(f"\n[{p.cluster_id}] {p.persona_name}")
        print(f"  {p.description}")
        print(f"  Traits: {p.key_traits}")