"""
Review Analysis Agent (standalone, pre-LangGraph).

Given a question about app reviews (e.g. "what are the top delivery
complaints for Swiggy?"), this:
  1. Embeds the question with the same model used for the reviews table
  2. Retrieves the most similar reviews via pgvector cosine similarity
  3. Sends those reviews + the question to Gemini, asking for a
     STRUCTURED (Pydantic-validated) analysis - not free text

This is intentionally a plain script, not a LangGraph node yet - get this
working and trustworthy in isolation first, then wrap it.

SETUP:
   pip install langchain-google-genai pydantic sentence-transformers pgvector psycopg2-binary python-dotenv
   Add GEMINI_API_KEY=... to your .env

RUN:
   python agents/review_analysis_agent.py
"""

import os
from typing import Optional

from dotenv import load_dotenv
import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

DB_CONFIG = dict(
    dbname=os.getenv("POSTGRES_DB", "food_delivery"),
    user=os.getenv("POSTGRES_USER", "app"),
    password=os.getenv("POSTGRES_PASSWORD", "app_password"),
    host=os.getenv("POSTGRES_HOST", "localhost"),
    port=os.getenv("POSTGRES_PORT", "5432"),
)

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 15


# ---------- Structured output schema ----------

class ReviewAnalysis(BaseModel):
    summary: str = Field(description="2-3 sentence overview of what the retrieved reviews say")
    top_complaints: list[str] = Field(description="3-5 distinct, specific complaint themes")
    top_praises: list[str] = Field(description="2-3 distinct, specific things users like, if any appear")
    overall_sentiment: str = Field(description="One of: very negative, negative, mixed, positive, very positive")
    confidence_note: Optional[str] = Field(
        default=None,
        description="Note if the retrieved reviews are too few/off-topic to answer confidently"
    )


# ---------- Retrieval ----------

def retrieve_reviews(conn, embed_model, question: str, app_filter: Optional[str] = None, top_k: int = TOP_K):
    query_vec = embed_model.encode(question, normalize_embeddings=True)

    sql = """
        SELECT app, rating, review_text, (embedding <=> %s) AS distance
        FROM reviews
        WHERE embedding IS NOT NULL
    """
    params = [query_vec]
    if app_filter:
        sql += " AND app = %s"
        params.append(app_filter)
    sql += " ORDER BY embedding <=> %s LIMIT %s"
    params.extend([query_vec, top_k])

    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    return [{"app": r[0], "rating": r[1], "text": r[2], "distance": float(r[3])} for r in rows]


# ---------- Agent ----------

def analyze(question: str, app_filter: Optional[str] = None) -> ReviewAnalysis:
    conn = psycopg2.connect(**DB_CONFIG)
    register_vector(conn)
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    reviews = retrieve_reviews(conn, embed_model, question, app_filter)
    conn.close()

    if not reviews:
        return ReviewAnalysis(
            summary="No reviews retrieved.",
            top_complaints=[],
            top_praises=[],
            overall_sentiment="mixed",
            confidence_note="No matching reviews found in the database for this query.",
        ), []

    reviews_block = "\n".join(
        f"[{r['app']} | {r['rating']}★] {r['text']}" for r in reviews
    )

    llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0)
    structured_llm = llm.with_structured_output(ReviewAnalysis)

    prompt = (
        "You are analyzing real user reviews of food delivery apps.\n"
        "Base your analysis ONLY on the reviews below - do not invent complaints "
        "or praises that aren't reflected in them. If the reviews don't clearly "
        "address the question, say so in confidence_note.\n\n"
        f"QUESTION: {question}\n\n"
        f"RETRIEVED REVIEWS ({len(reviews)}):\n{reviews_block}"
    )

    result = structured_llm.invoke(prompt)
    return result, reviews


if __name__ == "__main__":
    question = "What are the most common complaints about delivery time and order accuracy?"
    print(f"Question: {question}\n")

    result, retrieved = analyze(question, app_filter="swiggy")

    print("--- Retrieved reviews (first 3, for eyeballing relevance) ---")
    for r in retrieved[:3]:
        print(f"  [{r['rating']}★, dist={r['distance']:.3f}] {r['text'][:120]}")

    print("\n--- Structured analysis ---")
    print(result.model_dump_json(indent=2))