# Food Delivery Multi-Agent Insights

Repo: [github.com/richakumari0311/Food-Delivery-Strategy-Agents](https://github.com/richakumari0311/Food-Delivery-Strategy-Agents)

Multi-agent AI system analyzing Swiggy and Zomato/Eternal's position in the Indian food delivery and quick commerce market. Five agents combine order data, app reviews, and live competitor research into evidence-cited strategic recommendations, orchestrated with LangGraph and served through Streamlit.

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

| Agent | Role |
|---|---|
| Review Analysis | RAG over Play Store reviews, pgvector similarity search |
| Data Analyst | Text-to-SQL over order data, read-only guardrails, self-corrects on query errors |
| Consumer Segmentation | KMeans clustering, LLM-named personas |
| Competitor Research | Live web search via Tavily |
| Strategy | Synthesizes the other four, weighted by data quality |

Strategy trusts Review Analysis and Competitor Research more than Data Analyst and Segmentation, since the latter two run on a synthetic dataset. All LLM and database calls retry transient failures and fail fast on exhausted quota.

## Tech Stack

Python, LangGraph, LangChain, Gemini, pgvector, Sentence Transformers, scikit-learn, Tavily, PostgreSQL, Supabase, Docker, Pydantic, Streamlit, Plotly, Ragas.

## Setup

```bash
cp .env.example .env          # GEMINI_API_KEY, TAVILY_API_KEY
docker compose up -d            # local Postgres + pgvector
pip install -r requirements.txt
```

Run `scripts/01` through `scripts/07` in order to populate the database and generate embeddings.

**Run an agent:**
```bash
python src/agents/review_analysis.py
python -m src.agents.strategy       # calls all 4 other agents, then synthesizes
```

**Run the full graph:**
```bash
python -m src.orchestrator
```

**Run the evaluation harness** (Ragas: faithfulness and answer relevancy):
```bash
python -m src.eval.run_evaluation
```

**Run the app:**
```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # fill in real values
streamlit run app/streamlit_app.py
```

Gemini's free tier is limited on the full Flash models; set `GEMINI_MODEL=gemini-3.5-flash-lite` in `.env` while iterating.

## Production Design

- **No public traffic hits raw LLM calls unmetered.** The dashboard tab is DB-only, no LLM calls. Agent questions are cached per-question. A database-backed daily usage counter (survives app restarts, unlike an in-memory counter) caps agent queries and full strategy runs separately.
- **No internal errors reach visitors.** Every LLM and database call is wrapped; failures show a generic message and error type, never a stack trace, host name, or SQL.
- **Fails fast on misconfiguration.** Missing secrets show one clear message on startup instead of an unpredictable crash later.
- **Self-correcting SQL.** The Data Analyst retries once with the actual database error fed back to the model, rather than crashing on a bad query.
- **Evaluated, not just tested by hand.** `src/eval/` scores agent outputs on faithfulness and answer relevancy against real retrieved context, including deliberately off-topic test cases to confirm agents flag what they don't know rather than guess.

## Deployment

Database on **Supabase** (managed Postgres with pgvector, see `MIGRATE_TO_SUPABASE.md`). UI on **Streamlit Community Cloud** (free, connects to GitHub) or an equivalent host. Both are required for a public link, since a deployed app needs a reachable database.

## Known Limitations

- The Kaggle orders dataset is synthetic; treat its specific numbers as directional, not factual. Confirmed by evaluation: the Data Analyst's lowest faithfulness score came from a question implying real company financials.
- Segmentation clusters have modest silhouette scores (~0.12): real but overlapping segments.
- Investor PDF parsing extracts prose only, not structured financial tables.