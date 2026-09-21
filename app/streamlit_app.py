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

st.set_page_config(
    page_title="Food Delivery Agent Insights",
    page_icon="📈",
    layout="wide",
    menu_items={"Get help": None, "Report a bug": None, "About": None},
)

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

_REQUIRED_SECRETS = ["POSTGRES_HOST", "POSTGRES_PASSWORD", "GEMINI_API_KEY", "TAVILY_API_KEY"]
_missing = [k for k in _REQUIRED_SECRETS if not os.getenv(k)]
if _missing:
    st.error(
        "This app is missing required configuration and cannot start.\n\n"
        f"Missing: {', '.join(_missing)}\n\n"
        "If you're the app owner, set these in Streamlit Cloud under "
        "Settings > Secrets, or in a local .streamlit/secrets.toml file."
    )
    st.stop()

import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine, text

from src.utils.db import get_db_url
from src.utils.rate_limit import check_and_increment, get_current_count

# ---------- Design tokens (kept in one place so the palette stays consistent) ----------

COLOR_BG = "#FAFAFA"
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
.block-container {{
    padding-top: 2rem;
    padding-bottom: 2rem;
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
#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}
[data-testid="stToolbar"] {{visibility: hidden;}}
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


def condense_question(history: list, new_question: str) -> str:
    """Rewrite a follow-up question into a standalone one using prior
    conversation turns, so the underlying agent (which has no memory of
    its own) still gets something searchable/queryable on its own.
    Skipped entirely if there's no history yet - first question in a
    session is always already standalone. NOT cached (unlike the agent
    calls below) - history is unique per session and not worth caching."""
    if not history:
        return new_question

    from langchain_google_genai import ChatGoogleGenerativeAI

    history_block = "\n".join(
        f"Q: {turn['question']}\nA: {turn['answer_summary']}" for turn in history[-3:]  # last 3 turns is enough context
    )
    prompt = (
        "Given this conversation history and a new follow-up question, rewrite "
        "the follow-up into a standalone question that includes all context "
        "needed to understand it without the history. If it's already "
        "standalone, return it completely unchanged. Return ONLY the question, "
        "nothing else.\n\n"
        f"CONVERSATION SO FAR:\n{history_block}\n\n"
        f"FOLLOW-UP: {new_question}"
    )
    llm = ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL", "gemini-3.6-flash"),
                                  temperature=0, timeout=30, google_api_key=os.getenv("GEMINI_API_KEY"))
    try:
        return llm.invoke(prompt).content.strip()
    except Exception:
        return new_question  # if condensing fails, fall back to the raw question rather than blocking


def render_agent_response(agent_choice: str, result: dict, sql: str = None, n_reviews: int = None):
    """Renders one assistant response with consistent formatting. Used for
    BOTH the fresh response right after asking AND when redrawing history -
    previously these used different code paths and looked inconsistent
    (history flattened to plain text, live response nicely formatted).
    Single source of truth now."""
    if agent_choice == "Review Analysis" and n_reviews is not None:
        st.caption(f"Based on {n_reviews} retrieved reviews")
    if agent_choice == "Data Analyst" and sql:
        st.code(sql, language="sql")

    st.write(result["summary"])

    key_list = result.get("top_complaints") or result.get("key_findings") or []
    label = "Top complaints:" if agent_choice == "Review Analysis" else "Key findings:"
    if key_list:
        st.markdown(f"**{label}**")
        for item in key_list:
            st.write(f"- {item}")

    note = result.get("confidence_note") or result.get("caveats")
    if note:
        st.warning(note)

    if agent_choice == "Competitor Research" and result.get("sources"):
        st.markdown("**Sources:**")
        for s in result["sources"]:
            st.write(f"- {s}")


def render_agent_chat_history(agent_key: str):
    """Render all prior turns for this agent as a chat thread, using the
    same render_agent_response function as live answers - so history looks
    identical to how it looked when it was first shown."""
    history = st.session_state.chat_history.get(agent_key, [])
    for turn in history:
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            render_agent_response(
                turn["agent_choice"], turn["result"],
                sql=turn.get("sql"), n_reviews=turn.get("n_reviews"),
            )


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


def show_friendly_error(exc: Exception):
    """Public-facing error message. Doesn't leak internal details (DB
    hosts, SQL, API error bodies) to visitors - just the failure type."""
    st.error(
        "Something went wrong processing this request. This is usually "
        "temporary (a busy API or a momentary connection issue). Please "
        "try again in a moment."
    )
    st.caption(f"Error type: {type(exc).__name__}")


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

if "chat_history" not in st.session_state:
    st.session_state.chat_history = {"review": [], "data": [], "competitor": []}

with st.sidebar:
    st.markdown("### Market Intelligence")
    st.caption("Swiggy vs. Zomato/Eternal competitive analysis")
    st.caption("5 agents: Review Analysis, Data Analyst, Consumer Segmentation, "
               "Competitor Research, Strategy.")

    with st.expander("Usage today", expanded=False):
        aq = get_current_count("agent_query")
        sr = get_current_count("strategy_run")
        st.caption(f"Agent queries: {aq}/{AGENT_QUERY_DAILY_LIMIT}")
        st.caption(f"Strategy runs: {sr}/{STRATEGY_RUN_DAILY_LIMIT}")


# ---------- Main ----------

st.title("Food Delivery Multi-Agent Insights")
st.markdown('<p class="app-subtitle">Real reviews, order data, and live market research, synthesized by 5 AI agents</p>',
            unsafe_allow_html=True)

tab_dashboard, tab_agents, tab_strategy, tab_about = st.tabs(
    ["Dashboard", "Ask an Agent", "Full Strategy", "About"]
)

with tab_dashboard:
    try:
        kpis = load_kpis()
    except Exception as e:
        st.error("Could not load dashboard data right now. Please try again shortly.")
        st.caption(f"Error type: {type(e).__name__}")
        st.stop()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Orders", f"{kpis['total_orders']:,}")
    k2.metric("Unique Users", f"{kpis['total_users']:,}")
    k3.metric("Avg Order Value", f"₹{kpis['avg_order_value']:,.0f}")
    k4.metric("Repeat Order Rate", f"{kpis['repeat_pct']}%")

    st.divider()
    st.subheader("Order data by city")
    city_df = load_city_stats()
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(city_df, x="city", y="avg_order_value", color_discrete_sequence=[COLOR_ACCENT])
        fig.update_layout(plot_bgcolor=COLOR_SURFACE, paper_bgcolor=COLOR_SURFACE,
                           font_color=COLOR_TEXT, title="Average order value by city",
                           margin=dict(t=40, l=0, r=0, b=0))
        st.plotly_chart(fig, width='stretch')
    with col2:
        fig = px.bar(city_df, x="city", y="repeat_rate", color_discrete_sequence=[COLOR_LOW])
        fig.update_layout(plot_bgcolor=COLOR_SURFACE, paper_bgcolor=COLOR_SURFACE,
                           font_color=COLOR_TEXT, title="Repeat order rate by city",
                           margin=dict(t=40, l=0, r=0, b=0))
        st.plotly_chart(fig, width='stretch')

    with st.expander("View raw city data"):
        st.dataframe(city_df, width='stretch')

    st.divider()
    st.subheader("Review rating distribution")
    review_df = load_review_rating_dist()
    if not review_df.empty:
        fig = px.bar(review_df, x="rating", y="n", color="app", barmode="group",
                     color_discrete_sequence=[COLOR_ACCENT, COLOR_LOW])
        fig.update_layout(plot_bgcolor=COLOR_SURFACE, paper_bgcolor=COLOR_SURFACE,
                           font_color=COLOR_TEXT, margin=dict(t=20, l=0, r=0, b=0))
        st.plotly_chart(fig, width='stretch')

    st.divider()
    st.subheader("Consumer segments")
    seg_df = load_segments()
    if not seg_df.empty:
        fig = px.bar(seg_df, x="n_users", y="persona_name", orientation="h",
                     color_discrete_sequence=[COLOR_ACCENT])
        fig.update_layout(plot_bgcolor=COLOR_SURFACE, paper_bgcolor=COLOR_SURFACE,
                           font_color=COLOR_TEXT, margin=dict(t=20, l=0, r=0, b=0),
                           yaxis_title=None, xaxis_title="Users")
        st.plotly_chart(fig, width='stretch')
    else:
        st.info("Segmentation hasn't been run yet - run `src/agents/segmentation.py` to populate this.")

@st.fragment
def render_ask_agent_tab():
    """Wrapped in st.fragment so asking a question only reruns THIS part of
    the page, not the whole app - meaningfully faster than a full rerun.
    NOTE: st.bottom cannot be used inside a fragment (Streamlit raises
    StreamlitFragmentWidgetsNotAllowedOutsideError - a fragment can't write
    widgets to the app's root-level bottom container, which is "outside"
    the fragment's own boundary). So chat_input renders inline here rather
    than pinned to the viewport edge; the bordered panel below keeps
    history and input visually together as one unit instead."""
    st.write("Pick an agent below and ask about customer reviews, order trends, "
              "or what competitors are doing. Feel free to ask follow-up questions.")

    agent_choice = st.selectbox("Agent", ["Review Analysis", "Data Analyst", "Competitor Research"])
    agent_key = {"Review Analysis": "review", "Data Analyst": "data", "Competitor Research": "competitor"}[agent_choice]

    if agent_choice == "Review Analysis":
        app_filter = st.selectbox("App", ["", "swiggy", "zomato"], format_func=lambda x: x or "Both apps",
                                   key="app_filter_review")

    history = st.session_state.chat_history[agent_key]

    # Single bordered panel holding history + input together, so they read
    # as one connected chat unit even though chat_input can't be pinned to
    # the viewport bottom inside a fragment (see note above).
    with st.container(border=True):
        chat_box = st.container(height=380)
        with chat_box:
            if not history:
                example = {
                    "Review Analysis": "What do people say about delivery speed?",
                    "Data Analyst": "What's the average spend per city?",
                    "Competitor Research": "How is Blinkit performing against Instamart?",
                }[agent_choice]
                st.caption(f'No messages yet. Try: "{example}"')
            else:
                render_agent_chat_history(agent_key)

        question = st.chat_input(f"Ask the {agent_choice} agent...", key=f"chat_input_{agent_key}")

    if st.button("Clear conversation", key=f"clear_{agent_key}"):
        st.session_state.chat_history[agent_key] = []
        st.rerun(scope="fragment")

    if question:
        allowed, count = check_and_increment("agent_query", AGENT_QUERY_DAILY_LIMIT)
        if not allowed:
            st.error("You've reached today's question limit for this app. Please try again tomorrow.")
            st.stop()

        # A follow-up (when there's prior history) costs a second call
        # against the same daily limit - one to understand context from
        # the conversation, one to actually answer. Not shown to the user;
        # this is an internal accounting detail, not something they need
        # to reason about while using the app.
        standalone_question = question
        if history:
            condense_allowed, _ = check_and_increment("agent_query", AGENT_QUERY_DAILY_LIMIT)
            if condense_allowed:
                standalone_question = condense_question(history, question)

        sql, n_reviews = None, None
        try:
            if agent_choice == "Review Analysis":
                with st.spinner("Retrieving reviews and analyzing..."):
                    result, n_reviews = cached_review_analysis(standalone_question, app_filter)
            elif agent_choice == "Data Analyst":
                with st.spinner("Generating SQL and analyzing..."):
                    result, sql, rows = cached_data_analysis(standalone_question)
            else:
                with st.spinner("Searching the web and analyzing..."):
                    result = cached_competitor_research(standalone_question)
        except Exception as e:
            show_friendly_error(e)
            st.stop()

        st.session_state.chat_history[agent_key].append({
            "question": question,
            "standalone_question": standalone_question,
            "answer_summary": result["summary"],
            "agent_choice": agent_choice,
            "result": result,
            "sql": sql,
            "n_reviews": n_reviews,
        })
        st.rerun(scope="fragment")


with tab_agents:
    render_ask_agent_tab()

with tab_strategy:
    st.write(
        "Get a complete strategic recommendation, combining review sentiment, "
        "order data, customer segments, and live market research into one "
        "prioritized analysis. Takes about a minute to run."
    )

    business_q = st.text_area(
        "Business question",
        "What should Swiggy prioritize over the next 1-2 quarters to improve "
        "customer retention and competitive position against Zomato/Eternal?",
    )

    run_clicked = st.button("Run Full Strategy Analysis", type="primary")

    if run_clicked:
        allowed, count = check_and_increment("strategy_run", STRATEGY_RUN_DAILY_LIMIT)
        if not allowed:
            st.error(f"Daily strategy-run limit reached ({STRATEGY_RUN_DAILY_LIMIT}/day). Try again tomorrow.")
        else:
            try:
                with st.spinner("Analyzing reviews, orders, customer segments, and the market..."):
                    result, inputs = cached_strategy(
                        business_q,
                        review_q="What are the most common complaints about delivery time and order accuracy?",
                        data_q="Which cities have the highest average order value, and how does repeat order rate vary by city?",
                        competitor_q="What recent strategic moves have Swiggy and Zomato/Eternal made in the Indian quick-commerce or food delivery space?",
                    )
            except Exception as e:
                show_friendly_error(e)
                st.stop()

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
delivery and quick commerce market using 5 specialized AI agents:

- **Review Analysis**: RAG over real Play Store reviews
- **Data Analyst**: text-to-SQL over order data, with read-only guardrails
- **Consumer Segmentation**: KMeans clustering with LLM-generated personas
- **Competitor Research**: live web search via Tavily
- **Strategy**: synthesizes the other four, weighting real data above synthetic data
        """)
    with a2:
        st.metric("Agents", "5")
        st.metric("Data sources", "4")