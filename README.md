# Food Delivery Multi-Agent Insights Project

Multi-agent system analyzing Swiggy/Zomato-style order data, app reviews,
and investor reports to generate consumer segmentation and strategy insights.

## Architecture

Agents run independently where possible and are combined through a LangGraph orchestrator:

```
                 START
                /      \
   review_analysis   data_analyst
                \      /
                 merge
                   |
        [ segmentation, competitor_research ]
                   |
              strategy_agent   (synthesizes all outputs)
                   |
                  END
```

- **Review Analysis Agent**: RAG over Play Store reviews (Swiggy, Zomato). Embeds a question, retrieves similar reviews via pgvector cosine similarity, and asks an LLM for a structured sentiment/complaint summary grounded only in retrieved text.
- **Data Analyst Agent**: Text-to-SQL over the `orders` and `user_features` tables, with guardrails (read-only queries only, keyword blocklist, row limits). Generates SQL, executes it, then summarizes the actual returned rows.
- **Consumer Segmentation Agent**: KMeans clustering on behavioral features from `user_features` (order frequency, spend, repeat rate, discount usage, weekend behavior, age). Cluster count is chosen automatically via silhouette score. An LLM names and describes each cluster using only aggregated cluster statistics, not individual user data.
- **Competitor/Research Agent**: Web search based agent for external market/competitor information not present in the internal data.
- **Strategy Agent**: Synthesizes outputs from all other agents into recommendations.

## Data Sources

1. **Kaggle order dataset** (`rhythmghai/food-ordering-behavior-india-50k-orders`): 50,000 synthetic orders across 4,000 users. Note: this dataset has limited real signal in several fields (`rating_given` is close to uniform across all categories, city-level and cuisine-level metrics show minimal variation). Order value and repeat-rate variation across users is more meaningful and is what the Segmentation Agent uses.
2. **Play Store reviews**: Scraped via `google-play-scraper` for both the Swiggy and Zomato apps, no API key required. This is real user-generated text and is the primary source of genuine sentiment/complaint signal in the project.
3. **Investor relations PDFs**: Annual reports and shareholder letters from Swiggy Corporate and Eternal Investor Relations, downloaded directly by URL, parsed with PyMuPDF, and chunked for RAG.

## Structure
- `data/` - raw and processed data (gitignored, regenerate via `scripts/`)
- `notebooks/` - exploratory analysis only
- `scripts/` - numbered, one-shot data pipeline steps (run in order)
- `src/` - application code: agents, orchestrator, DB, embeddings
- `api/` - FastAPI serving layer
- `app/` - Streamlit UI
- `tests/` - test suite

## Tech Stack

Python, Pandas, SQL, LangGraph, LangChain, Gemini, pgvector, Sentence Transformers, scikit-learn, FastAPI, PostgreSQL, Docker, Pydantic, Ragas.

## Setup

1. `cp .env.example .env` and adjust credentials
2. `docker compose up -d` (Postgres + pgvector)
3. `pip install -r requirements.txt`
4. Run `scripts/01` through `scripts/06` in order to populate the database, then `scripts/07_generate_embeddings.py`

### Run an agent standalone

```bash
python src/agents/review_analysis_agent.py
python src/agents/data_analyst_agent.py
python src/agents/segmentation_agent.py
```

### Run the orchestrated graph

```bash
python -m src.orchestrator
```

## Known Limitations

- The Kaggle orders dataset is synthetic. Ratings and several categorical breakdowns (city, cuisine) show little to no real variation. Order-level spend and repeat-behavior metrics show more genuine variance and are the more trustworthy signal from this source.
- Segmentation clusters have modest silhouette scores (around 0.12), meaning the clusters are real but overlap substantially rather than being cleanly separated. This is consistent with the underlying dataset's limited signal, not a flaw in the clustering approach itself.
- Play Store reviews were pulled as a recent snapshot (a few days of "newest" reviews per app), not a long historical range.
- Investor PDF parsing extracts prose text only. Financial tables inside the PDFs are not structured or extracted as numeric data; they are only present as flattened text within chunks.