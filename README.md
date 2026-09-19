# Food Delivery Multi-Agent Insights

A multi-agent AI system analyzing Swiggy and Zomato/Eternal's position in the Indian food delivery and quick commerce market. Five specialized agents combine order data, app reviews, and live competitor research into evidence-cited strategic recommendations, orchestrated with LangGraph and served through a Streamlit interface.

## Architecture

```
    START
   /  |  |  \
  RA  DA SEG CR      (parallel, no dependency)
   \  |  |  /
    strategy          (fan-in, synthesizes all four)
       |
      END
```

| Agent | File | Role |
|---|---|---|
| Review Analysis | `review_analysis.py` | RAG over Play Store reviews, pgvector similarity search |
| Data Analyst | `data_analyst.py` | Text-to-SQL over order data, read-only guardrails |
| Consumer Segmentation | `segmentation.py` | KMeans clustering, LLM-named personas |
| Competitor Research | `competitor_research.py` | Live web search via Tavily |
| Strategy | `strategy.py` | Synthesizes the other four, weighted by data quality |

Strategy explicitly trusts Review Analysis and Competitor Research more than Data Analyst and Segmentation, since the latter two run on a synthetic dataset. All LLM and database calls retry transient failures with backoff (`src/utils/retry.py`) and fail fast on exhausted quota.

## Data Sources

- **Kaggle order dataset**: 50,000 synthetic orders, 4,000 users. Ratings and city/cuisine breakdowns carry little real signal; spend and repeat-behavior metrics are more reliable.
- **Play Store reviews**: scraped for both apps, real user text, the project's strongest sentiment signal.
- **Investor PDFs**: Swiggy and Eternal reports, parsed and chunked for RAG.
- **Live web search** (Tavily): current market and competitor information.

## Tech Stack

Python, Pandas, SQL, LangGraph, LangChain, Gemini, pgvector, Sentence Transformers, scikit-learn, Tavily, PostgreSQL, Supabase, Docker, Pydantic, Streamlit, Plotly.

## Project Structure

```
food-delivery-agents/
├── data/                  # gitignored, regenerate via scripts/
├── scripts/               # numbered pipeline steps, run in order
├── src/
│   ├── agents/             # the 5 agents
│   ├── utils/               # db.py, retry.py, rate_limit.py
│   └── orchestrator.py       # LangGraph wiring
├── app/streamlit_app.py    # public UI
├── .streamlit/               # theme config and secrets template
├── schema.sql
├── docker-compose.yml
├── MIGRATE_TO_SUPABASE.md
└── requirements.txt
```

## Setup

```bash
cp .env.example .env          # add GEMINI_API_KEY, TAVILY_API_KEY
docker compose up -d            # local Postgres + pgvector
pip install -r requirements.txt
```

Run `scripts/01` through `scripts/07` in order to populate the database and generate embeddings.

**Run an agent standalone:**
```bash
python src/agents/review_analysis.py
python -m src.agents.strategy   # calls all 4 other agents, then synthesizes
```

**Run the full graph:**
```bash
python -m src.orchestrator
```

**Run the app locally:**
```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # fill in real values
streamlit run app/streamlit_app.py
```

Gemini's free tier is limited (roughly 20 requests/day on the full Flash models); set `GEMINI_MODEL=gemini-3.5-flash-lite` in `.env` while iterating.

## Deployment

Database on **Supabase** (managed Postgres with pgvector built in, see `MIGRATE_TO_SUPABASE.md`), UI on **Streamlit Community Cloud** (free, connects to GitHub). Both required for a public, shareable link.

Public traffic is protected from exhausting the shared Gemini quota by:
- A dashboard tab with no LLM calls at all
- Per-question caching, so repeated questions don't repeat the underlying call
- A database-backed daily usage cap (`src/utils/rate_limit.py`), separate limits for single agent questions and full strategy runs

## Known Limitations

- The Kaggle dataset is synthetic; treat its specific numbers as directional, not factual.
- Segmentation clusters have modest silhouette scores (~0.12): real but overlapping segments.
- Reviews are a recent snapshot, not a long historical range.
- Investor PDF parsing extracts prose only, not structured financial tables.
- Strategy intentionally discounts synthetic-data findings in its recommendations. This is by design.