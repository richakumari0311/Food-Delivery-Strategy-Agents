-- Schema for the food delivery multi-agent project.

CREATE EXTENSION IF NOT EXISTS vector;


-- Source 1: Kaggle orders and per-user behavioral features.

CREATE TABLE IF NOT EXISTS orders (
    order_id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL,
    age SMALLINT,
    age_group TEXT,
    city TEXT,
    order_time TEXT,
    day_type TEXT,
    is_weekend BOOLEAN,
    cuisine TEXT,
    meal_type TEXT,
    restaurant_type TEXT,
    order_value INTEGER,
    discount_applied BOOLEAN,
    delivery_fee INTEGER,
    total_paid INTEGER,
    time_taken_to_order SMALLINT,
    rating_given SMALLINT,
    is_repeat_order BOOLEAN,
    mood TEXT,
    hunger_level TEXT,
    company TEXT,
    rainy_weather BOOLEAN
);

CREATE INDEX IF NOT EXISTS idx_orders_user_id
    ON orders (user_id);

CREATE INDEX IF NOT EXISTS idx_orders_city
    ON orders (city);

CREATE INDEX IF NOT EXISTS idx_orders_cuisine
    ON orders (cuisine);


CREATE TABLE IF NOT EXISTS user_features (
    user_id INTEGER PRIMARY KEY,
    total_orders INTEGER,
    avg_order_value NUMERIC(10, 2),
    avg_total_paid NUMERIC(10, 2),
    avg_rating_given NUMERIC(4, 2),
    repeat_order_rate NUMERIC(5, 4),
    discount_usage_rate NUMERIC(5, 4),
    weekend_order_rate NUMERIC(5, 4),
    favorite_cuisine TEXT,
    favorite_meal_type TEXT,
    favorite_city TEXT,
    age SMALLINT,
    age_group TEXT
);


-- Source 2: Play Store reviews for the Review Analysis Agent.

CREATE TABLE IF NOT EXISTS reviews (
    review_id TEXT PRIMARY KEY,
    app TEXT NOT NULL,
    user_name TEXT,
    rating SMALLINT,
    review_text TEXT,
    helpful_votes INTEGER,
    review_created_version TEXT,
    review_date TIMESTAMP,
    app_version TEXT,
    embedding vector(384)
);

CREATE INDEX IF NOT EXISTS idx_reviews_app
    ON reviews (app);

CREATE INDEX IF NOT EXISTS idx_reviews_rating
    ON reviews (rating);


-- Source 3: Investor PDF chunks used by RAG/research workflows.

CREATE TABLE IF NOT EXISTS investor_pdf_chunks (
    id SERIAL PRIMARY KEY,
    source_file TEXT NOT NULL,
    company TEXT NOT NULL,
    page_number INTEGER,
    chunk_index INTEGER,
    chunk_text TEXT NOT NULL,
    word_count INTEGER,
    embedding vector(384)
);

CREATE INDEX IF NOT EXISTS idx_pdf_chunks_company
    ON investor_pdf_chunks (company);

CREATE INDEX IF NOT EXISTS idx_pdf_chunks_source_file
    ON investor_pdf_chunks (source_file);


-- Vector indexes are created after embeddings are populated.
-- HNSW works with cosine distance through vector_cosine_ops.

-- CREATE INDEX IF NOT EXISTS idx_reviews_embedding_hnsw
--     ON reviews USING hnsw (embedding vector_cosine_ops);

-- CREATE INDEX IF NOT EXISTS idx_pdf_chunks_embedding_hnsw
--     ON investor_pdf_chunks USING hnsw (embedding vector_cosine_ops);