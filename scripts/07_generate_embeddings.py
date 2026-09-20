"""Generate and store embeddings for reviews and investor PDF chunks."""

import os

import psycopg2
from dotenv import load_dotenv
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer


load_dotenv()


DB_CONFIG = {
    "dbname": os.getenv("POSTGRES_DB", "food_delivery"),
    "user": os.getenv("POSTGRES_USER", "app"),
    "password": os.getenv("POSTGRES_PASSWORD", "app_password"),
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": os.getenv("POSTGRES_PORT", "5432"),
    "sslmode": os.getenv("POSTGRES_SSLMODE", "prefer"),
}

MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 64

TABLES = [
    {
        "table": "reviews",
        "id_col": "review_id",
        "text_col": "review_text",
    },
    {
        "table": "investor_pdf_chunks",
        "id_col": "id",
        "text_col": "chunk_text",
    },
]


def embed_table(
    conn,
    model: SentenceTransformer,
    table: str,
    id_col: str,
    text_col: str,
) -> None:
    """Generate and store embeddings for rows missing embeddings."""
    with conn.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT {id_col}, {text_col}
            FROM {table}
            WHERE embedding IS NULL
                AND {text_col} IS NOT NULL
                AND TRIM({text_col}) != ''
            """
        )
        rows = cursor.fetchall()

    if not rows:
        print(
            f"  {table}: nothing to embed "
            "(already done or no valid rows)"
        )
        return

    ids = [row[0] for row in rows]
    texts = [row[1] for row in rows]

    print(f"  {table}: embedding {len(texts)} rows ...")

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    with conn.cursor() as cursor:
        for row_id, embedding in zip(ids, embeddings):
            cursor.execute(
                f"""
                UPDATE {table}
                SET embedding = %s
                WHERE {id_col} = %s
                """,
                (embedding, row_id),
            )

    conn.commit()
    print(f"  {table}: wrote {len(ids)} embeddings")


def build_indexes(conn) -> None:
    """Create HNSW indexes for vector similarity search."""
    print("\nBuilding HNSW indexes ...")

    with conn.cursor() as cursor:
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_reviews_embedding
            ON reviews USING hnsw (embedding vector_cosine_ops);
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_pdf_chunks_embedding
            ON investor_pdf_chunks
            USING hnsw (embedding vector_cosine_ops);
            """
        )

    conn.commit()
    print("Indexes built.")


def print_embedding_counts(conn) -> None:
    """Print the number of rows with generated embeddings."""
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT COUNT(*)
            FROM reviews
            WHERE embedding IS NOT NULL
            """
        )
        review_count = cursor.fetchone()[0]

        cursor.execute(
            """
            SELECT COUNT(*)
            FROM investor_pdf_chunks
            WHERE embedding IS NOT NULL
            """
        )
        pdf_count = cursor.fetchone()[0]

    print(f"\nreviews with embeddings: {review_count}")
    print(
        "investor_pdf_chunks with embeddings: "
        f"{pdf_count}"
    )


def main() -> None:
    """Generate embeddings and build vector indexes."""
    print(f"Loading model: {MODEL_NAME} ...")

    model = SentenceTransformer(MODEL_NAME)

    print(
        "Model loaded. "
        f"Output dimension: {model.get_embedding_dimension()}"
    )

    with psycopg2.connect(**DB_CONFIG) as connection:
        register_vector(connection)

        print("\nGenerating embeddings ...")

        for table_config in TABLES:
            embed_table(
                connection,
                model,
                **table_config,
            )

        build_indexes(connection)
        print_embedding_counts(connection)


if __name__ == "__main__":
    main()