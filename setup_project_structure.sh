#!/usr/bin/env bash
# Scaffolds the food-delivery-agents project structure.
# Run from your project root: bash setup_project_structure.sh

set -e

echo "Creating directory structure ..."

mkdir -p data/raw/kaggle_orders
mkdir -p data/raw/playstore_reviews
mkdir -p data/raw/investor_pdfs
mkdir -p data/processed
mkdir -p notebooks
mkdir -p scripts
mkdir -p src/agents
mkdir -p api
mkdir -p app
mkdir -p tests

# .gitkeep so empty dirs survive git even with nothing in them yet
for dir in notebooks tests app api data/raw/kaggle_orders data/raw/playstore_reviews data/raw/investor_pdfs data/processed; do
    touch "$dir/.gitkeep"
done

echo "Moving existing pipeline scripts into scripts/ (if present) ..."
for f in 01_fetch_kaggle_orders.py 02_clean_kaggle_orders.py 03_scrape_playstore_reviews.py \
         04_download_investor_pdfs.py 05_parse_investor_pdfs.py 06_load_to_postgres.py; do
    if [ -f "$f" ]; then
        mv "$f" "scripts/$f"
        echo "  moved $f -> scripts/"
    fi
done

echo "Creating starter src/ files (empty placeholders) ..."
touch src/__init__.py
touch src/agents/__init__.py
touch src/agents/data_analyst.py
touch src/agents/review_analysis.py
touch src/agents/segmentation.py
touch src/agents/competitor_research.py
touch src/agents/strategy.py
touch src/orchestrator.py
touch src/db.py
touch src/embeddings.py
touch src/config.py

echo "Creating .gitignore ..."
cat > .gitignore << 'EOF'
# Environments
.env
venv/
.venv/
__pycache__/
*.pyc

# Data (raw/processed data shouldn't live in git - too large, often regenerable)
data/raw/*
data/processed/*
!data/raw/.gitkeep
!data/processed/.gitkeep
!data/raw/*/.gitkeep

# Jupyter
.ipynb_checkpoints/

# OS
.DS_Store

# IDE
.vscode/
.idea/

# Postgres data volume (if ever bind-mounted locally)
pgdata/
EOF

echo "Creating requirements.txt ..."
cat > requirements.txt << 'EOF'
# --- data pipeline ---
pandas
pyarrow
kagglehub
google-play-scraper
requests
pymupdf
tqdm

# --- database ---
sqlalchemy
psycopg2-binary
python-dotenv
pgvector

# --- embeddings / RAG ---
sentence-transformers
ragas

# --- agents / orchestration ---
langgraph
langchain
langchain-google-genai
pydantic

# --- serving / UI ---
fastapi
uvicorn
streamlit

# --- dev ---
pytest
EOF

echo "Creating README.md skeleton (only if it doesn't already exist) ..."
if [ ! -f README.md ]; then
cat > README.md << 'EOF'
# Food Delivery Multi-Agent Insights Project

Multi-agent system analyzing Swiggy/Zomato-style order data, app reviews,
and investor reports to generate consumer segmentation and strategy insights.

## Structure
- `data/` - raw and processed data (gitignored, regenerate via `scripts/`)
- `notebooks/` - exploratory analysis only
- `scripts/` - numbered, one-shot data pipeline steps (run in order)
- `src/` - application code: agents, orchestrator, DB, embeddings
- `api/` - FastAPI serving layer
- `app/` - Streamlit UI
- `tests/` - test suite

## Setup
1. `cp .env.example .env` and adjust credentials
2. `docker compose up -d` (Postgres + pgvector)
3. `pip install -r requirements.txt`
4. Run `scripts/01` through `scripts/06` in order to populate the database
EOF
fi

echo ""
echo "Done. New structure:"
find . -path ./data/raw -prune -o -path ./.git -prune -o -print | sort | head -60