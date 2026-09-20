"""Review analysis agent using pgvector retrieval and Gemini."""

import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pgvector.psycopg2 import register_vector
from pydantic import BaseModel, Field
import psycopg2
from sentence_transformers import SentenceTransformer

from src.utils.db import get_db_config
from src.utils.retry import with_db_retry, with_llm_retry


load_dotenv()


GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

DB_CONFIG = get_db_config()

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 15


class ReviewAnalysis(BaseModel):
    """Structured analysis generated from retrieved user reviews."""

    summary: str = Field(
        description="A 2-3 sentence overview of the retrieved reviews."
    )
    top_complaints: list[str] = Field(
        description="3-5 distinct and specific complaint themes."
    )
    top_praises: list[str] = Field(
        description=(
            "2-3 distinct and specific things users like, "
            "if any appear."
        )
    )
    overall_sentiment: str = Field(
        description=(
            "Overall sentiment: very negative, negative, mixed, "
            "positive, or very positive."
        )
    )
    confidence_note: str | None = Field(
        default=None,
        description=(
            "Note when the retrieved reviews are too few or off-topic "
            "to answer confidently."
        ),
    )


@with_db_retry()
def _connect_db():
    """Create a Postgres connection configured for pgvector."""
    connection = psycopg2.connect(**DB_CONFIG)

    # Supabase stores pgvector in the extensions schema. Set the search
    # path on every connection because pooled connections may not reliably
    # inherit database-level search_path settings.
    with connection.cursor() as cursor:
        cursor.execute(
            "SET search_path TO public, extensions;"
        )

    connection.commit()

    return connection


@with_llm_retry()
def _invoke_llm(
    structured_llm,
    prompt: str,
) -> ReviewAnalysis:
    """Generate a structured review analysis."""
    return structured_llm.invoke(prompt)


def retrieve_reviews(
    connection,
    embedding_model,
    question: str,
    app_filter: str | None = None,
    top_k: int = TOP_K,
) -> list[dict]:
    """Retrieve the most relevant reviews using pgvector similarity."""
    query_vector = embedding_model.encode(
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

    sql += """
        ORDER BY embedding <=> %s
        LIMIT %s
    """
    params.extend([query_vector, top_k])

    with connection.cursor() as cursor:
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


def analyze(
    question: str,
    app_filter: str | None = None,
) -> tuple[ReviewAnalysis, list[dict]]:
    """Retrieve relevant reviews and generate a structured analysis."""
    connection = _connect_db()

    try:
        register_vector(connection)
        embedding_model = SentenceTransformer(EMBED_MODEL_NAME)

        reviews = retrieve_reviews(
            connection,
            embedding_model,
            question,
            app_filter,
        )
    finally:
        connection.close()

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

    reviews_block = "\n".join(
        f"[{review['app']} | {review['rating']}★] "
        f"{review['text']}"
        for review in reviews
    )

    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=0,
    )
    structured_llm = llm.with_structured_output(ReviewAnalysis)

    prompt = (
        "You are analyzing real user reviews of food delivery apps.\n"
        "Base your analysis ONLY on the reviews below. Do not invent "
        "complaints or praises that are not reflected in them. If the "
        "reviews do not clearly address the question, say so in "
        "confidence_note.\n\n"
        f"QUESTION: {question}\n\n"
        f"RETRIEVED REVIEWS ({len(reviews)}):\n"
        f"{reviews_block}"
    )

    result = _invoke_llm(structured_llm, prompt)

    return result, reviews


def main() -> None:
    """Run a sample review analysis query."""
    question = (
        "What are the most common complaints about delivery time "
        "and order accuracy?"
    )

    print(f"Question: {question}\n")

    result, retrieved = analyze(
        question,
        app_filter="swiggy",
    )

    print(
        "--- Retrieved reviews "
        "(first 3, for eyeballing relevance) ---"
    )

    for review in retrieved[:3]:
        print(
            f"  [{review['rating']}★, "
            f"dist={review['distance']:.3f}] "
            f"{review['text'][:120]}"
        )

    print("\n--- Structured analysis ---")
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()