"""Customer segmentation using KMeans clustering and Gemini-generated personas."""

import os

import pandas as pd
import psycopg2
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.utils.db import get_db_config, get_db_url
from src.utils.retry import with_db_retry, with_llm_retry

load_dotenv()

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
DB_CONFIG = get_db_config()

# avg_rating_given is excluded because it has little variation in this dataset.
CLUSTER_FEATURES = [
    "total_orders",
    "avg_order_value",
    "repeat_order_rate",
    "discount_usage_rate",
    "weekend_order_rate",
    "age",
]

K_RANGE = range(3, 7)


class ClusterPersona(BaseModel):
    """Structured persona generated from an aggregated cluster profile."""

    cluster_id: int
    persona_name: str = Field(
        description="Short, memorable name for the customer segment."
    )
    description: str = Field(
        description="2-3 sentences describing who this segment is and how they order."
    )
    key_traits: list[str] = Field(
        description="3-4 specific numeric or behavioral traits from the profile."
    )


class PersonaSet(BaseModel):
    """Collection of generated customer personas."""

    personas: list[ClusterPersona]


@with_db_retry()
def load_user_features() -> pd.DataFrame:
    """Load user-level behavioral features from Postgres."""
    connection = psycopg2.connect(**DB_CONFIG)
    try:
        return pd.read_sql("SELECT * FROM user_features", connection)
    finally:
        connection.close()


def pick_best_k(X_scaled) -> tuple[int, float]:
    """Select the cluster count with the highest silhouette score."""
    best_k = 0
    best_score = -1.0

    for k in K_RANGE:
        labels = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10,
        ).fit_predict(X_scaled)

        score = silhouette_score(X_scaled, labels)
        print(f"  k={k}: silhouette={score:.4f}")

        if score > best_score:
            best_k = k
            best_score = score

    return best_k, best_score


def build_cluster_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """Build aggregated statistical profiles for each cluster."""
    profile = (
        df.groupby("cluster_id")
        .agg(
            n_users=("user_id", "count"),
            avg_total_orders=("total_orders", "mean"),
            avg_order_value=("avg_order_value", "mean"),
            avg_repeat_rate=("repeat_order_rate", "mean"),
            avg_discount_usage=("discount_usage_rate", "mean"),
            avg_weekend_rate=("weekend_order_rate", "mean"),
            avg_age=("age", "mean"),
            top_cuisine=("favorite_cuisine", lambda series: series.mode().iat[0]),
            top_city=("favorite_city", lambda series: series.mode().iat[0]),
        )
        .round(2)
        .reset_index()
    )

    return profile


@with_llm_retry()
def generate_personas(profile: pd.DataFrame) -> PersonaSet:
    """Generate executive-friendly personas from aggregated cluster statistics."""
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=0.3,
    )

    prompt = (
        "Here are statistical profiles of customer segments from KMeans clustering "
        "on a food delivery app's user base. For EACH cluster_id, give it a short "
        "persona name and description based ONLY on these aggregated stats. Do not "
        "invent details not implied by the numbers.\n\n"
        "Naming style: these will appear in a business strategy report for executives, "
        "not a consumer marketing deck. Use formal, analytical language such as "
        "'High-Value Repeat Customers' or 'Price-Sensitive Occasional Users'. "
        "Avoid casual or playful phrasing. Keep each name to 2-4 words.\n\n"
        f"{profile.to_string(index=False)}"
    )

    structured_llm = llm.with_structured_output(PersonaSet)
    return structured_llm.invoke(prompt)


@with_db_retry()
def save_segments(df: pd.DataFrame, persona_map: dict[int, str]) -> None:
    """Replace the user_segments table with the latest cluster assignments."""
    segments = df[["user_id", "cluster_id"]].copy()
    segments["persona_name"] = segments["cluster_id"].map(persona_map)

    connection = psycopg2.connect(**DB_CONFIG)
    try:
        with connection.cursor() as cursor:
            cursor.execute("DROP TABLE IF EXISTS user_segments")
            cursor.execute(
                """
                CREATE TABLE user_segments (
                    user_id INTEGER PRIMARY KEY,
                    cluster_id INTEGER NOT NULL,
                    persona_name TEXT NOT NULL
                )
                """
            )
        connection.commit()
    finally:
        connection.close()

    segments.to_sql(
        "user_segments",
        get_db_url(),
        if_exists="append",
        index=False,
    )


def run_segmentation() -> tuple[PersonaSet, pd.DataFrame]:
    """Run clustering, generate personas, and persist segment assignments."""
    df = load_user_features()
    print(f"Loaded {len(df)} users\n")

    features = df[CLUSTER_FEATURES].fillna(0)
    scaled_features = StandardScaler().fit_transform(features)

    print("Selecting k via silhouette score:")
    best_k, best_score = pick_best_k(scaled_features)
    print(f"\nBest k = {best_k} (silhouette = {best_score:.4f})\n")

    kmeans = KMeans(
        n_clusters=best_k,
        random_state=42,
        n_init=10,
    )
    df["cluster_id"] = kmeans.fit_predict(scaled_features)

    profile = build_cluster_profiles(df)

    print("--- Cluster profiles ---")
    print(profile.to_string(index=False))

    print("\nGenerating personas ...")
    persona_set = generate_personas(profile)
    persona_map = {
        persona.cluster_id: persona.persona_name
        for persona in persona_set.personas
    }

    save_segments(df, persona_map)
    print(
        f"\nSaved cluster assignments for {len(df)} users "
        "to `user_segments` table"
    )

    return persona_set, profile


def main() -> None:
    """Run customer segmentation and print generated personas."""
    persona_set, _ = run_segmentation()

    print("\n--- Personas ---")
    for persona in persona_set.personas:
        print(f"\n[{persona.cluster_id}] {persona.persona_name}")
        print(f"  {persona.description}")
        print(f"  Traits: {persona.key_traits}")


if __name__ == "__main__":
    main()