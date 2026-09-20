"""Streamlit dashboard for the Food Delivery Multi-Agent Insights project."""

import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import create_engine, text


# Configuration

st.set_page_config(
    page_title="Food Delivery Agent Insights",
    page_icon="📈",
    layout="wide",
    menu_items={
        "Get help": None,
        "Report a bug": None,
        "About": None,
    },
)

REQUIRED_SECRETS = [
    "POSTGRES_HOST",
    "POSTGRES_PASSWORD",
    "GEMINI_API_KEY",
    "TAVILY_API_KEY",
]

ENV_KEYS = [
    "POSTGRES_HOST",
    "POSTGRES_PORT",
    "POSTGRES_USER",
    "POSTGRES_PASSWORD",
    "POSTGRES_DB",
    "POSTGRES_SSLMODE",
    "GEMINI_API_KEY",
    "GEMINI_MODEL",
    "TAVILY_API_KEY",
]

AGENT_QUERY_DAILY_LIMIT = 30
STRATEGY_RUN_DAILY_LIMIT = 5

# Theme

COLOR_BG = "#FAFAFA"
COLOR_SURFACE = "#FFFFFF"
COLOR_TEXT = "#1A1D29"
COLOR_MUTED = "#6B7280"
COLOR_ACCENT = "#E8A33D"
COLOR_HIGH = "#C1442E"
COLOR_LOW = "#1F6F6F"

PRIORITY_COLORS = {
    "high": COLOR_HIGH,
    "medium": COLOR_ACCENT,
    "low": COLOR_LOW,
}

CHART_SEQUENCE = [
    COLOR_ACCENT,
    COLOR_LOW,
    COLOR_HIGH,
    "#4A5568",
    "#8B6F47",
    "#2D5F5D",
]


# Environment setup

for key in ENV_KEYS:
    if key in st.secrets:
        os.environ[key] = str(st.secrets[key])

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

missing_secrets = [key for key in REQUIRED_SECRETS if not os.getenv(key)]

if missing_secrets:
    st.error(
        "This app is missing required configuration and cannot start.\n\n"
        f"Missing: {', '.join(missing_secrets)}\n\n"
        "Set these values in Streamlit Cloud under Settings > Secrets "
        "or in a local .streamlit/secrets.toml file."
    )
    st.stop()


# Local imports must happen after environment configuration.
from src.utils.db import get_db_url
from src.utils.rate_limit import check_and_increment, get_current_count


# Styling

st.markdown(
    f"""
    <style>
    @import url(
        'https://fonts.googleapis.com/css2?family=Sora:wght@600;700'
        '&family=Inter:wght@400;500;600&display=swap'
    );

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

    #MainMenu {{
        visibility: hidden;
    }}

    footer {{
        visibility: hidden;
    }}

    [data-testid="stToolbar"] {{
        visibility: hidden;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# Database access


@st.cache_resource
def get_engine():
    """Create and cache the database engine."""
    return create_engine(get_db_url())


@st.cache_data(ttl=600)
def load_kpis() -> dict:
    """Load dashboard KPI metrics."""
    query = text(
        """
        SELECT
            COUNT(*) AS total_orders,
            COUNT(DISTINCT user_id) AS total_users,
            ROUND(AVG(order_value), 2) AS avg_order_value,
            ROUND(
                AVG(
                    CASE
                        WHEN is_repeat_order THEN 1.0
                        ELSE 0.0
                    END
                ) * 100,
                1
            ) AS repeat_pct
        FROM orders
        """
    )

    with get_engine().connect() as connection:
        row = connection.execute(query).fetchone()

    return dict(row._mapping)


@st.cache_data(ttl=600)
def load_city_stats() -> pd.DataFrame:
    """Load order and repeat-rate statistics by city."""
    query = text(
        """
        SELECT
            city,
            COUNT(*) AS orders,
            ROUND(AVG(order_value), 2) AS avg_order_value,
            ROUND(
                AVG(
                    CASE
                        WHEN is_repeat_order THEN 1.0
                        ELSE 0.0
                    END
                ),
                3
            ) AS repeat_rate
        FROM orders
        GROUP BY city
        ORDER BY avg_order_value DESC
        """
    )

    with get_engine().connect() as connection:
        return pd.read_sql(query, connection)


@st.cache_data(ttl=600)
def load_review_rating_dist() -> pd.DataFrame:
    """Load review rating distribution by app."""
    query = text(
        """
        SELECT
            app,
            rating,
            COUNT(*) AS n
        FROM reviews
        GROUP BY app, rating
        ORDER BY app, rating
        """
    )

    with get_engine().connect() as connection:
        return pd.read_sql(query, connection)


@st.cache_data(ttl=600)
def load_segments() -> pd.DataFrame:
    """Load consumer segment counts if segmentation has been run."""
    query = text(
        """
        SELECT
            persona_name,
            COUNT(*) AS n_users
        FROM user_segments
        GROUP BY persona_name
        ORDER BY n_users DESC
        """
    )

    try:
        with get_engine().connect() as connection:
            return pd.read_sql(query, connection)
    except Exception:
        return pd.DataFrame()


# LLM-backed analysis


@st.cache_data(ttl=3600, show_spinner=False)
def cached_review_analysis(
    question: str,
    app_filter: str,
) -> tuple[dict, int]:
    """Run and cache review analysis for a question."""
    from src.agents.review_analysis import analyze

    result, reviews = analyze(
        question,
        app_filter=app_filter or None,
    )

    return result.model_dump(), len(reviews)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_data_analysis(
    question: str,
) -> tuple[dict, str, list[dict]]:
    """Run and cache data analysis for a question."""
    from src.agents.data_analyst import analyze

    result, df, sql = analyze(question)

    return result.model_dump(), sql, df.to_dict("records")


@st.cache_data(ttl=3600, show_spinner=False)
def cached_competitor_research(question: str) -> dict:
    """Run and cache competitor research for a question."""
    from src.agents.competitor_research import research

    result, _ = research(question)

    return result.model_dump()


@st.cache_data(ttl=3600, show_spinner=False)
def cached_strategy(
    business_question: str,
    review_question: str,
    data_question: str,
    competitor_question: str,
) -> tuple[dict, dict]:
    """Run and cache the complete strategy analysis using LangGraph."""
    from src.orchestrator import build_graph

    app = build_graph()

    initial_state = {
        "review_question": review_question,
        "data_question": data_question,
        "competitor_question": competitor_question,
        "business_question": business_question,
        "app_filter": None,
        "review_analysis": None,
        "data_analysis": None,
        "segmentation": None,
        "competitor_research": None,
        "strategy": None,
    }

    final_state = app.invoke(initial_state)

    return final_state["strategy"], final_state

# UI helpers

def show_friendly_error(exc: Exception) -> None:
    """Display a safe public-facing error message."""
    st.error(
        "Something went wrong while processing this request. "
        "Please try again in a moment."
    )
    st.caption(f"Error type: {type(exc).__name__}")


def render_recommendation_card(rec: dict) -> None:
    """Render a strategic recommendation card."""
    priority = rec["priority"]
    color = PRIORITY_COLORS.get(priority, COLOR_MUTED)

    evidence_html = "".join(
        f'<div class="evidence-item">- {evidence}</div>'
        for evidence in rec["supporting_evidence"]
    )

    st.markdown(
        f"""
        <div class="rec-card" style="--rec-color: {color};">
            <div class="rec-priority">
                {priority.upper()} PRIORITY
            </div>
            <div class="rec-title">{rec["title"]}</div>
            <div>{rec["rationale"]}</div>
            <div style="margin-top: 0.5rem;">
                {evidence_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def check_usage_limit(
    usage_type: str,
    limit: int,
) -> bool:
    """Check whether a request is within its daily usage limit."""
    allowed, _ = check_and_increment(usage_type, limit)

    if not allowed:
        st.error(
            f"Daily {usage_type.replace('_', ' ')} limit reached "
            f"({limit}/day). Try again tomorrow."
        )

    return allowed


# Sidebar

with st.sidebar:
    st.markdown("### Market Intelligence")
    st.caption("Swiggy vs. Zomato/Eternal competitive analysis")
    st.caption(
        "5 agents: Review Analysis, Data Analyst, Consumer "
        "Segmentation, Competitor Research, Strategy."
    )

    with st.expander("Usage today"):
        agent_queries = get_current_count("agent_query")
        strategy_runs = get_current_count("strategy_run")

        st.caption(
            f"Agent queries: {agent_queries}/{AGENT_QUERY_DAILY_LIMIT}"
        )
        st.caption(
            f"Strategy runs: {strategy_runs}/{STRATEGY_RUN_DAILY_LIMIT}"
        )


# Main application

st.title("Food Delivery Multi-Agent Insights")
st.markdown(
    '<p class="app-subtitle">'
    "Real reviews, order data, and live market research, "
    "synthesized by 5 AI agents"
    "</p>",
    unsafe_allow_html=True,
)

tab_dashboard, tab_agents, tab_strategy, tab_about = st.tabs(
    ["Dashboard", "Ask an Agent", "Full Strategy", "About"]
)


# Dashboard

with tab_dashboard:
    try:
        kpis = load_kpis()
    except Exception as exc:
        st.error(
            "Could not load dashboard data right now. "
            "Please try again shortly."
        )
        st.caption(f"Error type: {type(exc).__name__}")
        st.stop()

    k1, k2, k3, k4 = st.columns(4)

    k1.metric("Total Orders", f"{kpis['total_orders']:,}")
    k2.metric("Unique Users", f"{kpis['total_users']:,}")
    k3.metric("Avg Order Value", f"₹{kpis['avg_order_value']:,.0f}")
    k4.metric("Repeat Order Rate", f"{kpis['repeat_pct']}%")

    st.divider()
    st.subheader("Order data by city")

    city_df = load_city_stats()
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        fig = px.bar(
            city_df,
            x="city",
            y="avg_order_value",
            color_discrete_sequence=[COLOR_ACCENT],
        )
        fig.update_layout(
            plot_bgcolor=COLOR_SURFACE,
            paper_bgcolor=COLOR_SURFACE,
            font_color=COLOR_TEXT,
            title="Average order value by city",
            margin=dict(t=40, l=0, r=0, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    with chart_col2:
        fig = px.bar(
            city_df,
            x="city",
            y="repeat_rate",
            color_discrete_sequence=[COLOR_LOW],
        )
        fig.update_layout(
            plot_bgcolor=COLOR_SURFACE,
            paper_bgcolor=COLOR_SURFACE,
            font_color=COLOR_TEXT,
            title="Repeat order rate by city",
            margin=dict(t=40, l=0, r=0, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("View raw city data"):
        st.dataframe(city_df, use_container_width=True)

    st.divider()
    st.subheader("Review rating distribution")

    review_df = load_review_rating_dist()

    if not review_df.empty:
        fig = px.bar(
            review_df,
            x="rating",
            y="n",
            color="app",
            barmode="group",
            color_discrete_sequence=[COLOR_ACCENT, COLOR_LOW],
        )
        fig.update_layout(
            plot_bgcolor=COLOR_SURFACE,
            paper_bgcolor=COLOR_SURFACE,
            font_color=COLOR_TEXT,
            margin=dict(t=20, l=0, r=0, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Consumer segments")

    segment_df = load_segments()

    if not segment_df.empty:
        fig = px.bar(
            segment_df,
            x="n_users",
            y="persona_name",
            orientation="h",
            color_discrete_sequence=[COLOR_ACCENT],
        )
        fig.update_layout(
            plot_bgcolor=COLOR_SURFACE,
            paper_bgcolor=COLOR_SURFACE,
            font_color=COLOR_TEXT,
            margin=dict(t=20, l=0, r=0, b=0),
            yaxis_title=None,
            xaxis_title="Users",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(
            "Segmentation hasn't been run yet. "
            "Run `src/agents/segmentation.py` to populate this."
        )


# Agent queries

with tab_agents:
    st.write(
        "Ask a specific agent a question. Answers are grounded in real "
        "data from reviews, orders, or live web search."
    )

    agent_choice = st.selectbox(
        "Agent",
        ["Review Analysis", "Data Analyst", "Competitor Research"],
    )

    if agent_choice == "Review Analysis":
        question = st.text_input(
            "Your question about app reviews",
            "What do users complain about most?",
        )
        app_filter = st.selectbox(
            "App",
            ["", "swiggy", "zomato"],
            format_func=lambda value: value or "Both apps",
        )
        asked = st.button(
            "Ask Review Analysis Agent",
            type="primary",
        )

        if not asked:
            st.caption(
                'Try: "What do people say about delivery speed?" '
                'or "Are customers happy with order accuracy?"'
            )

        if asked and check_usage_limit(
            "agent_query",
            AGENT_QUERY_DAILY_LIMIT,
        ):
            try:
                with st.spinner("Retrieving reviews and analyzing..."):
                    result, review_count = cached_review_analysis(
                        question,
                        app_filter,
                    )
            except Exception as exc:
                show_friendly_error(exc)
                st.stop()

            with st.chat_message("assistant"):
                st.caption(
                    f"Based on {review_count} retrieved reviews"
                )
                st.write(result["summary"])

                st.markdown("**Top complaints:**")
                for complaint in result["top_complaints"]:
                    st.write(f"- {complaint}")

                if result.get("confidence_note"):
                    st.warning(result["confidence_note"])

    elif agent_choice == "Data Analyst":
        question = st.text_input(
            "Your question about order data",
            "Which cities have the highest order values?",
        )
        asked = st.button(
            "Ask Data Analyst Agent",
            type="primary",
        )

        if not asked:
            st.caption(
                'Try: "What\'s the average spend per city?" '
                'or "How does repeat rate vary by age group?"'
            )

        if asked and check_usage_limit(
            "agent_query",
            AGENT_QUERY_DAILY_LIMIT,
        ):
            try:
                with st.spinner("Generating SQL and analyzing..."):
                    result, sql, _ = cached_data_analysis(question)
            except Exception as exc:
                show_friendly_error(exc)
                st.stop()

            with st.chat_message("assistant"):
                st.code(sql, language="sql")
                st.write(result["summary"])

                for finding in result["key_findings"]:
                    st.write(f"- {finding}")

                if result.get("caveats"):
                    st.warning(result["caveats"])

    else:
        question = st.text_input(
            "Your question about the market",
            "What recent moves have Swiggy and Zomato made "
            "in quick commerce?",
        )
        asked = st.button(
            "Ask Competitor Research Agent",
            type="primary",
        )

        if not asked:
            st.caption(
                'Try: "How is Blinkit performing against Instamart?" '
                'or "What\'s the latest on Eternal\'s profitability?"'
            )

        if asked and check_usage_limit(
            "agent_query",
            AGENT_QUERY_DAILY_LIMIT,
        ):
            try:
                with st.spinner("Searching the web and analyzing..."):
                    result = cached_competitor_research(question)
            except Exception as exc:
                show_friendly_error(exc)
                st.stop()

            with st.chat_message("assistant"):
                st.write(result["summary"])

                for finding in result["key_findings"]:
                    st.write(f"- {finding}")

                if result["sources"]:
                    st.markdown("**Sources:**")
                    for source in result["sources"]:
                        st.write(f"- {source}")


# Full strategy

with tab_strategy:
    st.write(
        "Run all 5 agents and synthesize a full strategic recommendation. "
        "This is the most expensive operation and has a lower daily limit."
    )

    business_question = st.text_area(
        "Business question",
        "What should Swiggy prioritize over the next 1-2 quarters "
        "to improve customer retention and competitive position "
        "against Zomato/Eternal?",
    )

    run_clicked = st.button(
        "Run Full Strategy Analysis",
        type="primary",
    )

    if not run_clicked:
        st.caption(
            "Runs Review Analysis, Data Analyst, Consumer Segmentation, "
            "and Competitor Research in parallel, then synthesizes "
            "prioritized, evidence-cited recommendations."
        )

    if run_clicked and check_usage_limit(
        "strategy_run",
        STRATEGY_RUN_DAILY_LIMIT,
    ):
        try:
            with st.spinner("Running all 5 agents, this takes a minute..."):
                result, _ = cached_strategy(
                    business_question,
                    review_question=(
                        "What are the most common complaints about "
                        "delivery time and order accuracy?"
                    ),
                    data_question=(
                        "Which cities have the highest average order value, "
                        "and how does repeat order rate vary by city?"
                    ),
                    competitor_question=(
                        "What recent strategic moves have Swiggy and "
                        "Zomato/Eternal made in the Indian quick-commerce "
                        "or food delivery space?"
                    ),
                )
        except Exception as exc:
            show_friendly_error(exc)
            st.stop()

        st.subheader("Executive Summary")
        st.write(result["executive_summary"])

        st.subheader("Recommendations")
        for recommendation in result["recommendations"]:
            render_recommendation_card(recommendation)

        st.subheader("Data Quality Caveats")
        for caveat in result["data_quality_caveats"]:
            st.warning(caveat)

        if result.get("confidence_note"):
            st.info(result["confidence_note"])


# About

with tab_about:
    about_col, metrics_col = st.columns([2, 1])

    with about_col:
        st.markdown(
            """
            This project analyzes Swiggy and Zomato/Eternal's position in
            the Indian food delivery and quick commerce market using
            five specialized AI agents:

            - **Review Analysis:** RAG over real Play Store reviews
            - **Data Analyst:** Text-to-SQL over order data with read-only guardrails
            - **Consumer Segmentation:** KMeans clustering with LLM-generated personas
            - **Competitor Research:** Live web search via Tavily
            - **Strategy:** Synthesizes the other four agents
            """
        )

    with metrics_col:
        st.metric("Agents", "5")
        st.metric("Data sources", "4")