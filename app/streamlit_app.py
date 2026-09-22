"""Streamlit app for the Food Delivery Multi-Agent Insights project."""

import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import create_engine, text

st.set_page_config(
    page_title="Food Delivery Agent Insights",
    page_icon="📈",
    layout="wide",
    menu_items={"Get help": None, "Report a bug": None, "About": None},
)

# Secrets must be bridged to env vars before any src import, since several
# modules read os.getenv() at import time.
_SECRET_KEYS = [
    "POSTGRES_HOST", "POSTGRES_PORT", "POSTGRES_USER", "POSTGRES_PASSWORD",
    "POSTGRES_DB", "POSTGRES_SSLMODE", "GEMINI_API_KEY", "GEMINI_MODEL",
    "TAVILY_API_KEY",
]
for key in _SECRET_KEYS:
    if key in st.secrets:
        os.environ[key] = str(st.secrets[key])

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_REQUIRED_SECRETS = ["POSTGRES_HOST", "POSTGRES_PASSWORD", "GEMINI_API_KEY", "TAVILY_API_KEY"]
_missing = [key for key in _REQUIRED_SECRETS if not os.getenv(key)]
if _missing:
    st.error(
        "This app is missing required configuration and cannot start.\n\n"
        f"Missing: {', '.join(_missing)}\n\n"
        "If you're the app owner, set these in Streamlit Cloud under "
        "Settings > Secrets, or in a local .streamlit/secrets.toml file."
    )
    st.stop()

from src.utils.db import get_db_url
from src.utils.rate_limit import check_and_increment, get_current_count

COLOR_SURFACE = "#FFFFFF"
COLOR_TEXT = "#1A1D29"
COLOR_MUTED = "#6B7280"
COLOR_ACCENT = "#E8A33D"
COLOR_HIGH = "#C1442E"
COLOR_LOW = "#1F6F6F"
PRIORITY_COLORS = {"high": COLOR_HIGH, "medium": COLOR_ACCENT, "low": COLOR_LOW}

AGENT_QUERY_DAILY_LIMIT = 30
STRATEGY_RUN_DAILY_LIMIT = 5

_FONT_IMPORT_URL = (
    "https://fonts.googleapis.com/css2"
    "?family=Sora:wght@600;700&family=Inter:wght@400;500;600&display=swap"
)

st.markdown(f"""
<style>
@import url('{_FONT_IMPORT_URL}');

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
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
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


@st.cache_resource
def get_engine():
    return create_engine(get_db_url())


@st.cache_data(ttl=600)
def load_kpis() -> dict:
    query = text("""
        SELECT
            COUNT(*) AS total_orders,
            COUNT(DISTINCT user_id) AS total_users,
            ROUND(AVG(order_value), 2) AS avg_order_value,
            ROUND(AVG(CASE WHEN is_repeat_order THEN 1.0 ELSE 0.0 END) * 100, 1) AS repeat_pct
        FROM orders
    """)
    with get_engine().connect() as conn:
        row = conn.execute(query).fetchone()
    return dict(row._mapping)


@st.cache_data(ttl=600)
def load_city_stats() -> pd.DataFrame:
    query = text("""
        SELECT
            city,
            COUNT(*) AS orders,
            ROUND(AVG(order_value), 2) AS avg_order_value,
            ROUND(AVG(CASE WHEN is_repeat_order THEN 1.0 ELSE 0.0 END), 3) AS repeat_rate
        FROM orders
        GROUP BY city
        ORDER BY avg_order_value DESC
    """)
    with get_engine().connect() as conn:
        return pd.read_sql(query, conn)


@st.cache_data(ttl=600)
def load_review_rating_dist() -> pd.DataFrame:
    query = text("""
        SELECT app, rating, COUNT(*) AS n
        FROM reviews
        GROUP BY app, rating
        ORDER BY app, rating
    """)
    with get_engine().connect() as conn:
        return pd.read_sql(query, conn)


@st.cache_data(ttl=600)
def load_segments() -> pd.DataFrame:
    query = text("""
        SELECT persona_name, COUNT(*) AS n_users
        FROM user_segments
        GROUP BY persona_name
        ORDER BY n_users DESC
    """)
    with get_engine().connect() as conn:
        try:
            return pd.read_sql(query, conn)
        except Exception:
            return pd.DataFrame()


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
    result, _ = research(question)
    return result.model_dump()


@st.cache_data(ttl=3600, show_spinner=False)
def cached_strategy(business_question: str, review_q: str, data_q: str, competitor_q: str):
    from src.agents.strategy import gather_inputs, synthesize
    inputs = gather_inputs(review_q, data_q, competitor_q)
    result = synthesize(business_question, inputs)
    return result.model_dump(), inputs


def condense_question(history: list, new_question: str) -> str:
    """Rewrite a follow-up into a standalone question using prior turns.

    Not cached, since history is unique per session. Falls back to the
    raw question if condensing fails, rather than blocking the interaction.
    """
    if not history:
        return new_question

    from langchain_google_genai import ChatGoogleGenerativeAI

    history_block = "\n".join(
        f"Q: {turn['question']}\nA: {turn['answer_summary']}"
        for turn in history[-3:]
    )
    prompt = (
        "Given this conversation history and a new follow-up question, "
        "rewrite the follow-up into a standalone question that includes "
        "all context needed to understand it without the history. If "
        "it's already standalone, return it unchanged. Return ONLY the "
        "question, nothing else.\n\n"
        f"CONVERSATION SO FAR:\n{history_block}\n\n"
        f"FOLLOW-UP: {new_question}"
    )
    llm = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-3.6-flash"),
        temperature=0,
        timeout=30,
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )
    try:
        return llm.invoke(prompt).content.strip()
    except Exception:
        return new_question


def render_agent_response(
    agent_choice: str,
    result: dict,
    sql: str | None = None,
    n_reviews: int | None = None,
) -> None:
    """Render one assistant response. Shared by live answers and history
    redraws so both look identical.
    """
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
        for source in result["sources"]:
            st.write(f"- {source}")


def render_agent_chat_history(agent_key: str) -> None:
    for turn in st.session_state.chat_history.get(agent_key, []):
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            render_agent_response(
                turn["agent_choice"],
                turn["result"],
                sql=turn.get("sql"),
                n_reviews=turn.get("n_reviews"),
            )


def show_friendly_error(exc: Exception) -> None:
    """Public-facing error message. Never leaks internal details (DB
    hosts, SQL, API error bodies) to visitors.
    """
    st.error("Something went wrong processing this request. Please try again in a moment.")
    st.caption(f"Error type: {type(exc).__name__}")


def render_recommendation_card(rec: dict) -> None:
    color = PRIORITY_COLORS.get(rec["priority"], COLOR_MUTED)
    evidence_html = "".join(
        f'<div class="evidence-item">- {item}</div>'
        for item in rec["supporting_evidence"]
    )
    st.markdown(f"""
    <div class="rec-card" style="--rec-color: {color};">
        <div class="rec-priority" style="color: {color};">{rec['priority'].upper()} PRIORITY</div>
        <div class="rec-title">{rec['title']}</div>
        <div>{rec['rationale']}</div>
        <div style="margin-top: 0.5rem;">{evidence_html}</div>
    </div>
    """, unsafe_allow_html=True)


AGENT_KEYS = {
    "Review Analysis": "review",
    "Data Analyst": "data",
    "Competitor Research": "competitor",
}
EXAMPLE_QUESTIONS = {
    "Review Analysis": "What do people say about delivery speed?",
    "Data Analyst": "What's the average spend per city?",
    "Competitor Research": "How is Blinkit performing against Instamart?",
}


@st.fragment
def render_ask_agent_tab() -> None:
    """Scoped as a fragment so asking a question only reruns this tab,
    not the whole app. Note: st.bottom cannot be used inside a fragment,
    so chat_input renders inline within the bordered panel below rather
    than pinned to the viewport edge.
    """
    st.write(
        "Pick an agent below and ask about customer reviews, order "
        "trends, or what competitors are doing. Feel free to ask "
        "follow-up questions."
    )

    agent_choice = st.selectbox(
        "Agent", ["Review Analysis", "Data Analyst", "Competitor Research"]
    )
    agent_key = AGENT_KEYS[agent_choice]

    app_filter = ""
    if agent_choice == "Review Analysis":
        app_filter = st.selectbox(
            "App", ["", "swiggy", "zomato"],
            format_func=lambda x: x or "Both apps",
            key="app_filter_review",
        )

    history = st.session_state.chat_history[agent_key]

    with st.container(border=True):
        with st.container(height=380):
            if not history:
                st.caption(f'No messages yet. Try: "{EXAMPLE_QUESTIONS[agent_choice]}"')
            else:
                render_agent_chat_history(agent_key)

        question = st.chat_input(
            f"Ask the {agent_choice} agent...", key=f"chat_input_{agent_key}"
        )

    if st.button("Clear conversation", key=f"clear_{agent_key}"):
        st.session_state.chat_history[agent_key] = []
        st.rerun(scope="fragment")

    if not question:
        return

    allowed, _ = check_and_increment("agent_query", AGENT_QUERY_DAILY_LIMIT)
    if not allowed:
        st.error("You've reached today's question limit for this app. Please try again tomorrow.")
        st.stop()

    # A follow-up costs a second call against the same limit: one to
    # understand context, one to answer. If quota runs out on the
    # condensing step, fall back to the raw question instead of blocking.
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
                result, sql, _ = cached_data_analysis(standalone_question)
        else:
            with st.spinner("Searching the web and analyzing..."):
                result = cached_competitor_research(standalone_question)
    except Exception as exc:
        show_friendly_error(exc)
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


if "chat_history" not in st.session_state:
    st.session_state.chat_history = {key: [] for key in AGENT_KEYS.values()}

with st.sidebar:
    st.markdown("### Market Intelligence")
    st.caption("Swiggy vs. Zomato/Eternal competitive analysis")
    st.caption(
        "5 agents: Review Analysis, Data Analyst, Consumer Segmentation, "
        "Competitor Research, Strategy."
    )
    with st.expander("Usage today", expanded=False):
        st.caption(f"Agent queries: {get_current_count('agent_query')}/{AGENT_QUERY_DAILY_LIMIT}")
        st.caption(f"Strategy runs: {get_current_count('strategy_run')}/{STRATEGY_RUN_DAILY_LIMIT}")

st.title("Food Delivery Multi-Agent Insights")
st.markdown(
    '<p class="app-subtitle">Real reviews, order data, and live market '
    'research, synthesized by 5 AI agents</p>',
    unsafe_allow_html=True,
)

tab_dashboard, tab_agents, tab_strategy, tab_about = st.tabs(
    ["Dashboard", "Ask an Agent", "Full Strategy", "About"]
)

with tab_dashboard:
    try:
        kpis = load_kpis()
    except Exception as exc:
        st.error("Could not load dashboard data right now. Please try again shortly.")
        st.caption(f"Error type: {type(exc).__name__}")
        st.stop()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Orders", f"{kpis['total_orders']:,}")
    col2.metric("Unique Users", f"{kpis['total_users']:,}")
    col3.metric("Avg Order Value", f"₹{kpis['avg_order_value']:,.0f}")
    col4.metric("Repeat Order Rate", f"{kpis['repeat_pct']}%")

    st.divider()
    st.subheader("Order data by city")
    city_df = load_city_stats()
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        fig = px.bar(city_df, x="city", y="avg_order_value", color_discrete_sequence=[COLOR_ACCENT])
        fig.update_layout(
            plot_bgcolor=COLOR_SURFACE, paper_bgcolor=COLOR_SURFACE, font_color=COLOR_TEXT,
            title="Average order value by city", margin=dict(t=40, l=0, r=0, b=0),
        )
        st.plotly_chart(fig, width="stretch")

    with chart_col2:
        fig = px.bar(city_df, x="city", y="repeat_rate", color_discrete_sequence=[COLOR_LOW])
        fig.update_layout(
            plot_bgcolor=COLOR_SURFACE, paper_bgcolor=COLOR_SURFACE, font_color=COLOR_TEXT,
            title="Repeat order rate by city", margin=dict(t=40, l=0, r=0, b=0),
        )
        st.plotly_chart(fig, width="stretch")

    with st.expander("View raw city data"):
        st.dataframe(city_df, width="stretch")

    st.divider()
    st.subheader("Review rating distribution")
    review_df = load_review_rating_dist()
    if not review_df.empty:
        fig = px.bar(
            review_df, x="rating", y="n", color="app", barmode="group",
            color_discrete_sequence=[COLOR_ACCENT, COLOR_LOW],
        )
        fig.update_layout(
            plot_bgcolor=COLOR_SURFACE, paper_bgcolor=COLOR_SURFACE, font_color=COLOR_TEXT,
            margin=dict(t=20, l=0, r=0, b=0),
        )
        st.plotly_chart(fig, width="stretch")

    st.divider()
    st.subheader("Consumer segments")
    seg_df = load_segments()
    if not seg_df.empty:
        fig = px.bar(
            seg_df, x="n_users", y="persona_name", orientation="h",
            color_discrete_sequence=[COLOR_ACCENT],
        )
        fig.update_layout(
            plot_bgcolor=COLOR_SURFACE, paper_bgcolor=COLOR_SURFACE, font_color=COLOR_TEXT,
            margin=dict(t=20, l=0, r=0, b=0), yaxis_title=None, xaxis_title="Users",
        )
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("Segmentation hasn't been run yet - run `src/agents/segmentation.py` to populate this.")

with tab_agents:
    render_ask_agent_tab()

with tab_strategy:
    st.write(
        "Get a complete strategic recommendation, combining review "
        "sentiment, order data, customer segments, and live market "
        "research into one prioritized analysis. Takes about a minute "
        "to run."
    )

    business_q = st.text_area(
        "Business question",
        "What should Swiggy prioritize over the next 1-2 quarters to "
        "improve customer retention and competitive position against "
        "Zomato/Eternal?",
    )

    REVIEW_QUESTION = "What are the most common complaints about delivery time and order accuracy?"
    DATA_QUESTION = (
        "Which cities have the highest average order value, "
        "and how does repeat order rate vary by city?"
    )
    COMPETITOR_QUESTION = (
        "What recent strategic moves have Swiggy and Zomato/Eternal made "
        "in the Indian quick-commerce or food delivery space?"
    )

    if st.button("Run Full Strategy Analysis", type="primary"):
        allowed, _ = check_and_increment("strategy_run", STRATEGY_RUN_DAILY_LIMIT)
        if not allowed:
            st.error(
                f"Daily strategy-run limit reached ({STRATEGY_RUN_DAILY_LIMIT}/day). "
                "Try again tomorrow."
            )
        else:
            try:
                with st.spinner("Analyzing reviews, orders, customer segments, and the market..."):
                    result, _ = cached_strategy(
                        business_q,
                        review_q=REVIEW_QUESTION,
                        data_q=DATA_QUESTION,
                        competitor_q=COMPETITOR_QUESTION,
                    )
            except Exception as exc:
                show_friendly_error(exc)
                st.stop()

            st.subheader("Executive Summary")
            st.write(result["executive_summary"])

            st.subheader("Recommendations")
            for rec in result["recommendations"]:
                render_recommendation_card(rec)

            st.subheader("Data Quality Caveats")
            for caveat in result["data_quality_caveats"]:
                st.warning(caveat)

            if result.get("confidence_note"):
                st.info(result["confidence_note"])

with tab_about:
    about_col1, about_col2 = st.columns([2, 1])
    with about_col1:
        st.markdown("""
This project analyzes Swiggy and Zomato/Eternal's position in the Indian food
delivery and quick commerce market using 5 specialized AI agents:

- **Review Analysis**: RAG over real Play Store reviews
- **Data Analyst**: text-to-SQL over order data, with read-only guardrails
- **Consumer Segmentation**: KMeans clustering with LLM-generated personas
- **Competitor Research**: live web search via Tavily
- **Strategy**: synthesizes the other four, weighting real data above synthetic data
        """)
    with about_col2:
        st.metric("Agents", "5")
        st.metric("Data sources", "4")