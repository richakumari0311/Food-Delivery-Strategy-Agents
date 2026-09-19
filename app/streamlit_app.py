"""
Streamlit app for the Food Delivery Multi-Agent Insights project.

Two kinds of content, deliberately separated:
  1. Dashboard tab: direct DB queries only, no LLM calls. Always available,
     free to load repeatedly, safe for any amount of traffic.
  2. Ask an Agent / Full Strategy tabs: real LLM calls. Gated by a
     DB-backed daily usage counter (src/utils/rate_limit.py) so public
     traffic can't silently exhaust the shared Gemini quota.

SETUP (local):
   pip install streamlit plotly
   streamlit run app/streamlit_app.py

DEPLOY: push to GitHub, then on share.streamlit.io point at this file.
Set secrets in the Streamlit Cloud dashboard (see .streamlit/secrets.toml.example).
The visual theme lives in .streamlit/config.toml (colors/font) plus the
CSS block below (typography import, card styling, priority color-coding).
"""

import os
import sys
from pathlib import Path

import streamlit as st

# --- Bridge Streamlit secrets -> environment variables ---
# Must happen BEFORE importing anything from src/, since several modules
# read os.getenv(...) at import time (e.g. GEMINI_MODEL constants).
for _key in [
    "POSTGRES_HOST", "POSTGRES_PORT", "POSTGRES_USER", "POSTGRES_PASSWORD",
    "POSTGRES_DB", "POSTGRES_SSLMODE", "GEMINI_API_KEY", "GEMINI_MODEL",
    "TAVILY_API_KEY",
]:
    if _key in st.secrets:
        os.environ[_key] = str(st.secrets[_key])

# Make `src` importable when run as `streamlit run app/streamlit_app.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine, text

from src.utils.db import get_db_url
from src.utils.rate_limit import check_and_increment, get_current_count

st.set_page_config(page_title="Food Delivery Agent Insights", page_icon="📈", layout="wide")

# ---------- Design tokens (kept in one place so the palette stays consistent) ----------

COLOR_BG = "#0B0B0B"
COLOR_SURFACE = "#FFFFFF"
COLOR_TEXT = "#1A1D29"
COLOR_MUTED = "#6B7280"
COLOR_ACCENT = "#E8A33D"      # saffron - primary actions, medium priority
COLOR_HIGH = "#C1442E"        # terracotta-red - high priority / urgent
COLOR_LOW = "#1F6F6F"         # deep teal - low priority / secondary
PRIORITY_COLORS = {"high": COLOR_HIGH, "medium": COLOR_ACCENT, "low": COLOR_LOW}
CHART_SEQUENCE = [COLOR_ACCENT, COLOR_LOW, COLOR_HIGH, "#4A5568", "#8B6F47", "#2D5F5D"]

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@600;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    color: {COLOR_TEXT};
}}
h1, h2, h3, .kpi-value {{
    font-family: 'Sora', sans-serif !important;
    letter-spacing: -0.01em;
}}
[data-testid="stMetricValue"] {{
    font-family: 'Sora', sans-serif;
    color: {COLOR_TEXT};
}}
[data-testid="stMetricLabel"] {{
    color: {COLOR_MUTED};
}}
.stTabs [data-baseweb="tab"] {{
    font-weight: 500;
}}
.rec-card {{
    background: {COLOR_SURFACE};
    border-left: 4px solid var(--rec-color);
    border-radius: 4px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04);
}}
.rec-title {{
    font-family: 'Sora', sans-serif;
    font-weight: 600;
    font-size: 1.05rem;
    margin-bottom: 0.25rem;
}}
.rec-priority {{
    display: inline-block;
    font-size: 0.72rem;
    font-weight: 600;
    color: var(--rec-color);
    margin-bottom: 0.4rem;
}}
.evidence-item {{
    color: {COLOR_MUTED};
    font-size: 0.9rem;
    margin: 0.15rem 0;
}}
.app-subtitle {{
    color: {COLOR_MUTED};
    font-size: 1rem;
    margin-top: -0.5rem;
}}
#MainMenu {{visibility: visible;}}
footer {{visibility: hidden;}}
</style>
""", unsafe_allow_html=True)

# Daily caps - tuned to a Flash-Lite-class quota (~500/day). Lower these if
# you're on a model with the ~20/day free tier.
AGENT_QUERY_DAILY_LIMIT = 30
STRATEGY_RUN_DAILY_LIMIT = 5


# ---------- Data access (cached, no LLM calls) ----------

@st.cache_resource
def get_engine():
    return create_engine(get_db_url())


@st.cache_data(ttl=600)
def load_kpis() -> dict:
    with get_engine().connect() as conn:
        row = conn.execute(text("""
            SELECT COUNT(*) AS total_orders, COUNT(DISTINCT user_id) AS total_users,
                   ROUND(AVG(order_value), 2) AS avg_order_value,
                   ROUND(AVG(CASE WHEN is_repeat_order THEN 1.0 ELSE 0.0 END) * 100, 1) AS repeat_pct
            FROM orders
        """)).fetchone()
    return dict(row._mapping)


@st.cache_data(ttl=600)
def load_city_stats() -> pd.DataFrame:
    with get_engine().connect() as conn:
        return pd.read_sql(text("""
            SELECT city, COUNT(*) AS orders, ROUND(AVG(order_value), 2) AS avg_order_value,
                   ROUND(AVG(CASE WHEN is_repeat_order THEN 1.0 ELSE 0.0 END), 3) AS repeat_rate
            FROM orders GROUP BY city ORDER BY avg_order_value DESC
        """), conn)


@st.cache_data(ttl=600)
def load_review_rating_dist() -> pd.DataFrame:
    with get_engine().connect() as conn:
        return pd.read_sql(text("""
            SELECT app, rating, COUNT(*) AS n FROM reviews GROUP BY app, rating ORDER BY app, rating
        """), conn)


@st.cache_data(ttl=600)
def load_segments() -> pd.DataFrame:
    with get_engine().connect() as conn:
        try:
            return pd.read_sql(text("""
                SELECT persona_name, COUNT(*) AS n_users
                FROM user_segments GROUP BY persona_name ORDER BY n_users DESC
            """), conn)
        except Exception:
            return pd.DataFrame()  # segmentation agent may not have run yet


# ---------- LLM-backed calls (cached per question, gated by daily limit) ----------

@st.cache_data(ttl=3600, show_spinner=False)
def cached_review_analysis(question: str, app_filter: str):
    from src.agents.review_analysis import analyze
    result, reviews = analyze(question, app_filter=app_filter or None)
    return result.model_dump(), len(reviews)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_data_analysis(question: str):
    from src.agents.data_analyst import analyze
    result, df, sql = analyze(question)
    return result.model_dump(), sql, df.to_dict("records")


@st.cache_data(ttl=3600, show_spinner=False)
def cached_competitor_research(question: str):
    from src.agents.competitor_research import research
    result, _raw_search_results = research(question)
    return result.model_dump()


@st.cache_data(ttl=3600, show_spinner=False)
def cached_strategy(business_question: str, review_q: str, data_q: str, competitor_q: str):
    from src.agents.strategy import gather_inputs, synthesize
    inputs = gather_inputs(review_q, data_q, competitor_q)
    result = synthesize(business_question, inputs)
    return result.model_dump(), inputs


def render_recommendation_card(rec: dict):
    color = PRIORITY_COLORS.get(rec["priority"], COLOR_MUTED)
    evidence_html = "".join(f'<div class="evidence-item">- {e}</div>' for e in rec["supporting_evidence"])
    st.markdown(f"""
    <div class="rec-card" style="--rec-color: {color};">
        <div class="rec-priority" style="color: {color};">{rec['priority'].upper()} PRIORITY</div>
        <div class="rec-title">{rec['title']}</div>
        <div>{rec['rationale']}</div>
        <div style="margin-top: 0.5rem;">{evidence_html}</div>
    </div>
    """, unsafe_allow_html=True)


# ---------- Sidebar ----------

with st.sidebar:
    st.markdown("### Market Intelligence")
    st.caption("Swiggy vs. Zomato/Eternal competitive analysis")
    st.divider()
    st.markdown("**Today's usage**")
    st.progress(min(get_current_count("agent_query") / AGENT_QUERY_DAILY_LIMIT, 1.0),
                text=f"Agent queries: {get_current_count('agent_query')}/{AGENT_QUERY_DAILY_LIMIT}")
    st.progress(min(get_current_count("strategy_run") / STRATEGY_RUN_DAILY_LIMIT, 1.0),
                text=f"Strategy runs: {get_current_count('strategy_run')}/{STRATEGY_RUN_DAILY_LIMIT}")
    st.divider()
    st.caption("5 specialized agents: Review Analysis, Data Analyst, "
               "Consumer Segmentation, Competitor Research, and Strategy.")


# ---------- Main ----------

st.title("Food Delivery Multi-Agent Insights")
st.markdown('<p class="app-subtitle">Real reviews, order data, and live market research, synthesized by 5 AI agents</p>',
            unsafe_allow_html=True)
st.write("")

tab_dashboard, tab_agents, tab_strategy, tab_about = st.tabs(
    ["Dashboard", "Ask an Agent", "Full Strategy", "About"]
)

with tab_dashboard:
    kpis = load_kpis()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Orders", f"{kpis['total_orders']:,}")
    k2.metric("Unique Users", f"{kpis['total_users']:,}")
    k3.metric("Avg Order Value", f"₹{kpis['avg_order_value']:,.0f}")
    k4.metric("Repeat Order Rate", f"{kpis['repeat_pct']}%")

    st.write("")
    st.subheader("Order data by city")
    city_df = load_city_stats()
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(city_df, x="city", y="avg_order_value", color_discrete_sequence=[COLOR_ACCENT])
        fig.update_layout(plot_bgcolor=COLOR_SURFACE, paper_bgcolor=COLOR_SURFACE,
                           font_color=COLOR_TEXT, title="Average order value by city",
                           margin=dict(t=40, l=0, r=0, b=0))
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.bar(city_df, x="city", y="repeat_rate", color_discrete_sequence=[COLOR_LOW])
        fig.update_layout(plot_bgcolor=COLOR_SURFACE, paper_bgcolor=COLOR_SURFACE,
                           font_color=COLOR_TEXT, title="Repeat order rate by city",
                           margin=dict(t=40, l=0, r=0, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("View raw city data"):
        st.dataframe(city_df, use_container_width=True)

    st.subheader("Review rating distribution")
    review_df = load_review_rating_dist()
    if not review_df.empty:
        fig = px.bar(review_df, x="rating", y="n", color="app", barmode="group",
                     color_discrete_sequence=[COLOR_ACCENT, COLOR_LOW])
        fig.update_layout(plot_bgcolor=COLOR_SURFACE, paper_bgcolor=COLOR_SURFACE,
                           font_color=COLOR_TEXT, margin=dict(t=20, l=0, r=0, b=0))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Consumer segments")
    seg_df = load_segments()
    if not seg_df.empty:
        fig = px.bar(seg_df, x="n_users", y="persona_name", orientation="h",
                     color_discrete_sequence=[COLOR_ACCENT])
        fig.update_layout(plot_bgcolor=COLOR_SURFACE, paper_bgcolor=COLOR_SURFACE,
                           font_color=COLOR_TEXT, margin=dict(t=20, l=0, r=0, b=0),
                           yaxis_title=None, xaxis_title="Users")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Segmentation hasn't been run yet - run `src/agents/segmentation.py` to populate this.")

with tab_agents:
    st.write(
        "Ask a specific agent a question. Answers are grounded in real data "
        "(reviews, orders, or live web search) - agents say so when the data "
        "doesn't clearly answer your question, rather than guessing."
    )

    agent_choice = st.selectbox("Agent", ["Review Analysis", "Data Analyst", "Competitor Research"])

    if agent_choice == "Review Analysis":
        question = st.text_input("Your question about app reviews", "What do users complain about most?")
        app_filter = st.selectbox("App", ["", "swiggy", "zomato"], format_func=lambda x: x or "Both apps")
        if st.button("Ask Review Analysis Agent", type="primary"):
            allowed, count = check_and_increment("agent_query", AGENT_QUERY_DAILY_LIMIT)
            if not allowed:
                st.error(f"Daily query limit reached ({AGENT_QUERY_DAILY_LIMIT}/day). Try again tomorrow.")
            else:
                with st.spinner("Retrieving reviews and analyzing..."):
                    result, n_reviews = cached_review_analysis(question, app_filter)
                with st.chat_message("assistant"):
                    st.caption(f"Based on {n_reviews} retrieved reviews")
                    st.write(result["summary"])
                    st.markdown("**Top complaints:**")
                    for c in result["top_complaints"]:
                        st.write(f"- {c}")
                    if result.get("confidence_note"):
                        st.warning(result["confidence_note"])

    elif agent_choice == "Data Analyst":
        question = st.text_input("Your question about order data", "Which cities have the highest order values?")
        if st.button("Ask Data Analyst Agent", type="primary"):
            allowed, count = check_and_increment("agent_query", AGENT_QUERY_DAILY_LIMIT)
            if not allowed:
                st.error(f"Daily query limit reached ({AGENT_QUERY_DAILY_LIMIT}/day). Try again tomorrow.")
            else:
                with st.spinner("Generating SQL and analyzing..."):
                    result, sql, rows = cached_data_analysis(question)
                with st.chat_message("assistant"):
                    st.code(sql, language="sql")
                    st.write(result["summary"])
                    for f in result["key_findings"]:
                        st.write(f"- {f}")
                    if result.get("caveats"):
                        st.warning(result["caveats"])

    else:  # Competitor Research
        question = st.text_input(
            "Your question about the market",
            "What recent moves have Swiggy and Zomato made in quick commerce?",
        )
        if st.button("Ask Competitor Research Agent", type="primary"):
            allowed, count = check_and_increment("agent_query", AGENT_QUERY_DAILY_LIMIT)
            if not allowed:
                st.error(f"Daily query limit reached ({AGENT_QUERY_DAILY_LIMIT}/day). Try again tomorrow.")
            else:
                with st.spinner("Searching the web and analyzing..."):
                    result = cached_competitor_research(question)
                with st.chat_message("assistant"):
                    st.write(result["summary"])
                    for f in result["key_findings"]:
                        st.write(f"- {f}")
                    if result["sources"]:
                        st.markdown("**Sources:**")
                        for s in result["sources"]:
                            st.write(f"- {s}")

with tab_strategy:
    st.write(
        "Runs all 5 agents and synthesizes a full strategic recommendation. "
        "This is the most expensive operation (~8-9 AI calls), so it's capped "
        "more tightly than individual agent questions."
    )

    business_q = st.text_area(
        "Business question",
        "What should Swiggy prioritize over the next 1-2 quarters to improve "
        "customer retention and competitive position against Zomato/Eternal?",
    )

    if st.button("Run Full Strategy Analysis", type="primary"):
        allowed, count = check_and_increment("strategy_run", STRATEGY_RUN_DAILY_LIMIT)
        if not allowed:
            st.error(f"Daily strategy-run limit reached ({STRATEGY_RUN_DAILY_LIMIT}/day). Try again tomorrow.")
        else:
            with st.spinner("Running all 5 agents, this takes a minute..."):
                result, inputs = cached_strategy(
                    business_q,
                    review_q="What are the most common complaints about delivery time and order accuracy?",
                    data_q="Which cities have the highest average order value, and how does repeat order rate vary by city?",
                    competitor_q="What recent strategic moves have Swiggy and Zomato/Eternal made in the Indian quick-commerce or food delivery space?",
                )

            st.subheader("Executive Summary")
            st.write(result["executive_summary"])

            st.subheader("Recommendations")
            for rec in result["recommendations"]:
                render_recommendation_card(rec)

            st.subheader("Data Quality Caveats")
            for c in result["data_quality_caveats"]:
                st.warning(c)

            if result.get("confidence_note"):
                st.info(result["confidence_note"])

with tab_about:
    a1, a2 = st.columns([2, 1])
    with a1:
        st.markdown("""
This project analyzes Swiggy and Zomato/Eternal's position in the Indian food
delivery and quick-commerce market using 5 specialized AI agents:

- **Review Analysis** - RAG over real Play Store reviews
- **Data Analyst** - text-to-SQL over order data, with read-only guardrails
- **Consumer Segmentation** - KMeans clustering with LLM-generated personas
- **Competitor Research** - live web search via Tavily
- **Strategy** - synthesizes the other four, weighting real data above synthetic data

Full source and documentation: [GitHub repo link here]
        """)
    with a2:
        st.metric("Agents", "5")
        st.metric("Data sources", "4")