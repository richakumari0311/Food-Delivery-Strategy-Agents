"""
Step 7: Generate embeddings for reviews.review_text and
investor_pdf_chunks.chunk_text using Sentence Transformers, write them
back into the pgvector columns, then build the HNSW similarity indexes
we deferred in schema.sql.

SETUP:
   pip install sentence-transformers pgvector psycopg2-binary python-dotenv

RUN:
   python scripts/07_generate_embeddings.py

NOTE: uses all-MiniLM-L6-v2 (384 dims, matches schema.sql). If you switch
models later, you'll need to change the `vector(384)` column width in
schema.sql to match the new model's output size and re-embed.
"""

import os

from dotenv import load_dotenv
import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer

load_dotenv()

DB_CONFIG = dict(
    dbname=os.getenv("POSTGRES_DB", "food_delivery"),
    user=os.getenv("POSTGRES_USER", "app"),
    password=os.getenv("POSTGRES_PASSWORD", "app_password"),
    host=os.getenv("POSTGRES_HOST", "localhost"),
    port=os.getenv("POSTGRES_PORT", "5432"),
    sslmode=os.getenv("POSTGRES_SSLMODE", "prefer"),
)

MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 64

TABLES = [
    {"table": "reviews", "id_col": "review_id", "text_col": "review_text"},
    {"table": "investor_pdf_chunks", "id_col": "id", "text_col": "chunk_text"},
]


def embed_table(conn, model, table: str, id_col: str, text_col: str):
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT {id_col}, {text_col} FROM {table} "
            f"WHERE embedding IS NULL AND {text_col} IS NOT NULL AND TRIM({text_col}) != ''"
        )
        rows = cur.fetchall()

    if not rows:
        print(f"  {table}: nothing to embed (already done or no rows)")
        return

    ids = [r[0] for r in rows]
    texts = [r[1] for r in rows]
    print(f"  {table}: embedding {len(texts)} rows ...")

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,  # so cosine similarity == dot product later
    )

    with conn.cursor() as cur:
        for row_id, emb in zip(ids, embeddings):
            cur.execute(
                f"UPDATE {table} SET embedding = %s WHERE {id_col} = %s",
                (emb, row_id),
            )
    conn.commit()
    print(f"  {table}: wrote {len(ids)} embeddings")


def build_indexes(conn):
    print("\nBuilding HNSW indexes (skipped if already present) ...")
    with conn.cursor() as cur:
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_reviews_embedding "
            "ON reviews USING hnsw (embedding vector_cosine_ops);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_pdf_chunks_embedding "
            "ON investor_pdf_chunks USING hnsw (embedding vector_cosine_ops);"
        )
    conn.commit()
    print("Indexes built.")


if __name__ == "__main__":
    print(f"Loading model: {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)
    print(f"Model loaded. Output dimension: {model.get_embedding_dimension()}")

    conn = psycopg2.connect(**DB_CONFIG)
    register_vector(conn)

    print("\nGenerating embeddings ...")
    for t in TABLES:
        embed_table(conn, model, **t)

    build_indexes(conn)

    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM reviews WHERE embedding IS NOT NULL")
        r_count = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM investor_pdf_chunks WHERE embedding IS NOT NULL")
        p_count = cur.fetchone()[0]

    print(f"\nreviews with embeddings: {r_count}")
    print(f"investor_pdf_chunks with embeddings: {p_count}")

    conn.close()