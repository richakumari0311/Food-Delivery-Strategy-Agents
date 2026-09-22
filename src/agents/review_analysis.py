"""Review analysis agent using pgvector retrieval and Gemini."""

import os
from typing import Optional, Any

import psycopg2
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pgvector.psycopg2 import register_vector
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from src.utils.db import get_db_config
from src.utils.retry import with_db_retry, with_llm_retry


load_dotenv()

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
DB_CONFIG = get_db_config()

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 15


class ReviewAnalysis(BaseModel):
    """Structured analysis generated from retrieved app reviews."""

    summary: str = Field(
        description=(
            "A 2-3 sentence overview of what the retrieved reviews say."
        )
    )
    top_complaints: list[str] = Field(
        description="3-5 distinct, specific complaint themes."
    )
    top_praises: list[str] = Field(
        description=(
            "2-3 distinct, specific things users like, "
            "if any appear in the reviews."
        )
    )
    overall_sentiment: str = Field(
        description=(
            "One of: very negative, negative, mixed, "
            "positive, very positive."
        )
    )
    confidence_note: Optional[str] = Field(
        default=None,
        description=(
            "Note if the retrieved reviews are too few or off-topic "
            "to answer confidently."
        ),
    )


@with_db_retry()
def _connect_db() -> psycopg2.extensions.connection:
    """Create a database connection with the required search path."""

    conn = psycopg2.connect(**DB_CONFIG)

    with conn.cursor() as cursor:
        cursor.execute("SET search_path TO public, extensions")

    conn.commit()

    return conn


@with_llm_retry()
def _invoke_llm(
    structured_llm: Any,
    prompt: str,
) -> ReviewAnalysis:
    """Generate structured review analysis."""

    return structured_llm.invoke(prompt)


def retrieve_reviews(
    conn: psycopg2.extensions.connection,
    embed_model: SentenceTransformer,
    question: str,
    app_filter: Optional[str] = None,
    top_k: int = TOP_K,
) -> list[dict]:
    """Retrieve the most similar reviews for a question."""

    query_vector = embed_model.encode(
        question,
        normalize_embeddings=True,
    )

    sql = """
        SELECT
            app,
            rating,
            review_text,
            (embedding <=> %s) AS distance
        FROM reviews
        WHERE embedding IS NOT NULL
    """
    params = [query_vector]

    if app_filter:
        sql += " AND app = %s"
        params.append(app_filter)

    sql += " ORDER BY embedding <=> %s LIMIT %s"
    params.extend([query_vector, top_k])

    with conn.cursor() as cursor:
        cursor.execute(sql, params)
        rows = cursor.fetchall()

    return [
        {
            "app": row[0],
            "rating": row[1],
            "text": row[2],
            "distance": float(row[3]),
        }
        for row in rows
    ]


def _create_llm() -> ChatGoogleGenerativeAI:
    """Create the configured Gemini client."""

    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=0,
        timeout=60,
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )


def _build_prompt(
    question: str,
    reviews: list[dict],
) -> str:
    """Build the review analysis prompt."""

    reviews_block = "\n".join(
        f"[{review['app']} | {review['rating']}★] {review['text']}"
        for review in reviews
    )

    return (
        "You are analyzing real user reviews of food delivery apps.\n"
        "Base your analysis ONLY on the reviews below. Do not invent "
        "complaints or praises that are not reflected in them. If the "
        "reviews do not clearly address the question, say so in "
        "confidence_note.\n\n"
        f"QUESTION: {question}\n\n"
        f"RETRIEVED REVIEWS ({len(reviews)}):\n"
        f"{reviews_block}"
    )


def analyze(
    question: str,
    app_filter: Optional[str] = None,
) -> tuple[ReviewAnalysis, list[dict]]:
    """Retrieve relevant reviews and generate a structured analysis."""

    conn = _connect_db()

    try:
        register_vector(conn)
        embed_model = SentenceTransformer(EMBED_MODEL_NAME)

        reviews = retrieve_reviews(
            conn=conn,
            embed_model=embed_model,
            question=question,
            app_filter=app_filter,
        )
    finally:
        conn.close()

    if not reviews:
        return (
            ReviewAnalysis(
                summary="No reviews retrieved.",
                top_complaints=[],
                top_praises=[],
                overall_sentiment="mixed",
                confidence_note=(
                    "No matching reviews were found in the database "
                    "for this query."
                ),
            ),
            [],
        )

    llm = _create_llm()
    structured_llm = llm.with_structured_output(ReviewAnalysis)
    prompt = _build_prompt(question, reviews)

    result = _invoke_llm(structured_llm, prompt)

    return result, reviews


if __name__ == "__main__":
    question = (
        "What are the most common complaints about delivery time "
        "and order accuracy?"
    )

    print(f"Question: {question}\n")

    result, retrieved = analyze(
        question,
        app_filter="swiggy",
    )

    print("--- Retrieved reviews (first 3) ---")

    for review in retrieved[:3]:
        print(
            f"  [{review['rating']}★, "
            f"dist={review['distance']:.3f}] "
            f"{review['text'][:120]}"
        )

    print("\n--- Structured analysis ---")
    print(result.model_dump_json(indent=2))
