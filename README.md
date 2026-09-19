# Food Delivery Multi-Agent Insights Project

A multi-agent AI system analyzing Swiggy and Zomato/Eternal's position in the Indian food delivery and quick commerce market. Five specialized agents combine internal order and review data with live competitor research, orchestrated with LangGraph, to produce evidence-cited strategic recommendations. A Streamlit interface makes the system usable without reading code, deployable to a public URL.

## Architecture

Four agents run in parallel with no dependency between them. The Strategy Agent is the fan-in point that waits for all four before synthesizing.

```
    START
   /  |  |  \
  RA  DA SEG CR      (parallel, no dependency)
   \  |  |  /
    strategy          (fan-in: waits for all 4, does the synthesis)
       |
      END
```

- **Review Analysis Agent** (`review_analysis.py`): RAG over Play Store reviews for Swiggy and Zomato. Embeds a question, retrieves similar reviews via pgvector cosine similarity, asks Gemini for a structured sentiment and complaint summary grounded only in retrieved text.
- **Data Analyst Agent** (`data_analyst.py`): Text to SQL over the `orders` and `user_features` tables. Guardrails include read-only queries only, a keyword blocklist, row limits, and `SET TRANSACTION READ ONLY` at the database level. Generates SQL, executes it, then summarizes the actual returned rows.
- **Consumer Segmentation Agent** (`segmentation.py`): KMeans clustering on behavioral features such as order frequency, spend, repeat rate, discount usage, weekend behavior, and age. Cluster count is chosen automatically via silhouette score. Gemini names and describes each cluster using only aggregated cluster statistics, never individual user rows.
- **Competitor/Research Agent** (`competitor_research.py`): Live web search via Tavily, followed by a structured Gemini synthesis grounded only in the returned search results, with real source URLs attached to every finding.
- **Strategy Agent** (`strategy.py`): Synthesizes the outputs of the other four into prioritized, evidence-cited recommendations. Weights inputs by data quality: Review Analysis and Competitor Research are treated as high trust since they use real external data, while Data Analyst and Segmentation are treated as directional only, since they run on a confirmed synthetic dataset.

All LLM and database calls are wrapped in retry logic (`src/utils/retry.py`) with exponential backoff, distinguishing transient errors worth retrying from exhausted quota, which fails fast with a clear message instead of retrying pointlessly.

## Data Sources

1. **Kaggle order dataset** (`rhythmghai/food-ordering-behavior-india-50k-orders`): 50,000 synthetic orders across 4,000 users. Ratings and several categorical breakdowns such as city and cuisine show little to no real variation. Order-level spend and repeat-behavior metrics show more genuine variance and are what the Segmentation Agent relies on.
2. **Play Store reviews**: Scraped via `google-play-scraper` for both apps, no API key required. Real user-generated text, and the primary source of genuine sentiment signal in the project.
3. **Investor relations PDFs**: Annual reports and shareholder letters from Swiggy Corporate and Eternal Investor Relations, parsed with PyMuPDF and chunked for RAG.
4. **Live web search** (Tavily): Current competitor and market information not present in any static dataset.

## Tech Stack

Python, Pandas, SQL, LangGraph, LangChain, Gemini, pgvector, Sentence Transformers, scikit-learn, Tavily, PostgreSQL, Supabase, Docker, Pydantic, Streamlit, Plotly.

## Structure

```
food-delivery-agents/
├── data/                       # gitignored, regenerate via scripts/
├── notebooks/                  # exploratory analysis only
├── scripts/                    # numbered, one-shot data pipeline steps, run in order
├── src/
│   ├── agents/
│   │   ├── review_analysis.py
│   │   ├── data_analyst.py
│   │   ├── segmentation.py
│   │   ├── competitor_research.py
│   │   └── strategy.py
│   ├── utils/
│   │   ├── db.py                # centralized DB connection config
│   │   ├── retry.py             # shared LLM/DB retry decorators
│   │   └── rate_limit.py        # DB-backed daily usage limiter for the public app
│   └── orchestrator.py          # LangGraph wiring, full 5-agent graph
├── app/
│   └── streamlit_app.py         # public-facing dashboard and agent Q&A interface
├── api/                          # reserved for a FastAPI serving layer
├── tests/
├── .streamlit/
│   ├── config.toml               # theme colors and font
│   └── secrets.toml.example      # template, copy to secrets.toml locally, gitignored
├── schema.sql                     # Postgres schema, all tables and vector columns
├── docker-compose.yml              # local Postgres and pgvector container
├── MIGRATE_TO_SUPABASE.md
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

### Local development

1. `cp .env.example .env` and add your `GEMINI_API_KEY` and `TAVILY_API_KEY`, free at tavily.com, no card required.
2. `docker compose up -d` to start local Postgres with pgvector.
3. `pip install -r requirements.txt`
4. Run `scripts/01` through `scripts/06` in order to populate the database, then `scripts/07_generate_embeddings.py`.

### Run an agent standalone

```bash
python src/agents/review_analysis.py
python src/agents/data_analyst.py
python src/agents/segmentation.py
python src/agents/competitor_research.py
python -m src.agents.strategy
```

### Run the full orchestrated graph

```bash
python -m src.orchestrator
```

Gemini's free tier for the full Flash models is limited, roughly 20 requests a day at the time of writing, and a single full graph run makes 8 to 9 calls. Set `GEMINI_MODEL=gemini-3.5-flash-lite` in `.env` while iterating, which gets a higher free quota, and switch back to the full model for final or demo runs.

### Run the Streamlit app locally

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# fill in real credentials in .streamlit/secrets.toml, this file is gitignored
streamlit run app/streamlit_app.py
```

## Deployment

The project is designed to run publicly on two free services working together.

1. **Database: Supabase.** Managed Postgres with pgvector already enabled, so `schema.sql` and all queries work unchanged. See `MIGRATE_TO_SUPABASE.md` for exact steps to move your local data, including already-computed embeddings, without recomputing anything.
2. **App hosting: Streamlit Community Cloud.** Connects directly to a GitHub repo and free. Set the same values from `.streamlit/secrets.toml.example` in the app's Secrets settings in the Streamlit Cloud dashboard, real credentials are never committed to the repo.

Because a public app can receive traffic from many visitors, direct LLM calls per click could exhaust the shared Gemini quota quickly. The app addresses this with:
- A **Dashboard tab** that only queries the database directly, no LLM calls, safe for unlimited traffic.
- **Per-question caching** on all agent calls, so repeated identical questions do not repeat the underlying LLM call.
- A **database-backed daily usage counter** (`src/utils/rate_limit.py`), not an in-memory counter, so the limit holds even if the app restarts or redeploys. Individual agent questions and full strategy runs are capped separately, since a full strategy run costs roughly 8 to 9 calls versus 1 to 2 for a single agent question.

## Known Limitations

- The Kaggle orders dataset is synthetic. Ratings and several categorical breakdowns show little to no real variation. Order-level spend and repeat-behavior metrics carry more genuine signal.
- Segmentation clusters have modest silhouette scores, around 0.12, meaning the clusters are real but overlap substantially. This is consistent with the dataset's limited signal, not a flaw in the clustering approach.
- Play Store reviews were pulled as a recent snapshot, not a long historical range.
- Investor PDF parsing extracts prose text only, financial tables are not extracted as structured numeric data.
- The Strategy Agent explicitly discounts findings from the synthetic-data agents in its recommendations. This is intentional, not a bug, recommendations are meant to reflect what the evidence actually supports.
- Daily usage limits on the public app are shared across all visitors, not per-user, since the goal is protecting the shared free-tier quota rather than metering individual users.