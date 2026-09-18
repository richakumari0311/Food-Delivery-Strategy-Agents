# Food Delivery Multi-Agent Insights Project

A multi-agent AI system analyzing Swiggy and Zomato/Eternal's position in the Indian food delivery and quick-commerce market. Five specialized agents combine internal order/review data with live competitor research, orchestrated with LangGraph, to produce evidence-cited strategic recommendations.

## Architecture

Four agents run in parallel (no dependency between them), and the Strategy Agent is the fan-in point that waits for all four before synthesizing:

```
    START
   /  |  |  \
  RA  DA SEG CR      (parallel, no dependency)
   \  |  |  /
    strategy          (fan-in: waits for all 4, does the synthesis)
       |
      END
```

- **Review Analysis Agent** (`review_analysis.py`): RAG over Play Store reviews (Swiggy, Zomato). Embeds a question, retrieves similar reviews via pgvector cosine similarity, asks Gemini for a structured sentiment/complaint summary grounded only in retrieved text.
- **Data Analyst Agent** (`data_analyst.py`): Text-to-SQL over the `orders` and `user_features` tables. Guardrails: read-only queries only, keyword blocklist, row limits, `SET TRANSACTION READ ONLY` at the database level. Generates SQL, executes it, then summarizes the actual returned rows.
- **Consumer Segmentation Agent** (`segmentation.py`): KMeans clustering on behavioral features (order frequency, spend, repeat rate, discount usage, weekend behavior, age). Cluster count chosen automatically via silhouette score. Gemini names and describes each cluster using only aggregated cluster statistics, never individual user rows.
- **Competitor/Research Agent** (`competitor_research.py`): Live web search via Tavily, followed by a structured Gemini synthesis grounded only in the returned search results, with real source URLs attached to every finding.
- **Strategy Agent** (`strategy.py`): Synthesizes the outputs of the other four into prioritized, evidence-cited recommendations. Explicitly weights inputs by data quality: Review Analysis and Competitor Research are treated as high-trust (real external data), while Data Analyst and Segmentation are treated as directional only, since they run on a confirmed-synthetic dataset.

All LLM and database calls are wrapped in retry logic (`src/utils/retry.py`) with exponential backoff, distinguishing transient errors (worth retrying) from exhausted quota (fails fast with a clear message rather than retrying pointlessly).

## Data Sources

1. **Kaggle order dataset** (`rhythmghai/food-ordering-behavior-india-50k-orders`): 50,000 synthetic orders across 4,000 users. Ratings and several categorical breakdowns (city, cuisine) show little to no real variation. Order-level spend and repeat-behavior metrics show more genuine variance and are what the Segmentation Agent relies on.
2. **Play Store reviews**: Scraped via `google-play-scraper` for both apps, no API key required. Real user-generated text, and the primary source of genuine sentiment signal in the project.
3. **Investor relations PDFs**: Annual reports and shareholder letters from Swiggy Corporate and Eternal Investor Relations, parsed with PyMuPDF and chunked for RAG.
4. **Live web search** (Tavily): Current competitor/market information not present in any static dataset.

## Tech Stack

Python, Pandas, SQL, LangGraph, LangChain, Gemini, pgvector, Sentence Transformers, scikit-learn, Tavily, PostgreSQL, Docker, Pydantic.

## Structure

```
food-delivery-agents/
├── data/                      # gitignored, regenerate via scripts/
├── notebooks/                 # exploratory analysis only
├── scripts/                   # numbered, one-shot data pipeline steps (run in order)
├── src/
│   ├── agents/
│   │   ├── review_analysis.py
│   │   ├── data_analyst.py
│   │   ├── segmentation.py
│   │   ├── competitor_research.py
│   │   └── strategy.py
│   ├── utils/
│   │   └── retry.py           # shared LLM/DB retry decorators
│   └── orchestrator.py        # LangGraph wiring, full 5-agent graph
├── api/                        # FastAPI serving layer
├── app/                        # Streamlit UI
├── tests/
├── schema.sql                   # Postgres schema (all tables + vector columns)
├── docker-compose.yml            # Postgres + pgvector container
├── .env.example
└── README.md
```

## Setup

1. `cp .env.example .env` and add your `GEMINI_API_KEY` and `TAVILY_API_KEY` (free at tavily.com, no card required)
2. `docker compose up -d` (Postgres + pgvector)
3. `pip install -r requirements.txt`
4. Run `scripts/01` through `scripts/06` in order to populate the database, then `scripts/07_generate_embeddings.py`

### Run an agent standalone

```bash
python src/agents/review_analysis.py
python src/agents/data_analyst.py
python src/agents/segmentation.py
python src/agents/competitor_research.py
python -m src.agents.strategy    # calls all 4 other agents, then synthesizes
```

### Run the full orchestrated graph

```bash
python -m src.orchestrator
```

Note on quota: Gemini's free tier for the full Flash models is limited (~20 requests/day at the time of writing), and a single full graph run makes 8-9 calls. Set `GEMINI_MODEL=gemini-3.5-flash-lite` in `.env` while iterating (~500/day free), and switch back to the full model for final/demo runs.

## Known Limitations

- The Kaggle orders dataset is synthetic. Ratings and several categorical breakdowns show little to no real variation. Order-level spend and repeat-behavior metrics carry more genuine signal.
- Segmentation clusters have modest silhouette scores (around 0.12), meaning the clusters are real but overlap substantially. Consistent with the dataset's limited signal, not a flaw in the clustering approach.
- Play Store reviews were pulled as a recent snapshot, not a long historical range.
- Investor PDF parsing extracts prose text only; financial tables are not extracted as structured numeric data.
- The Strategy Agent explicitly discounts findings from the synthetic-data agents in its recommendations. This is intentional, not a bug: recommendations are meant to reflect what the evidence actually supports.