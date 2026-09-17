"""
Orchestrator (LangGraph) - parallel fan-out / fan-in.

Review Analysis and Data Analyst agents have no dependency on each other
(different tables, different questions), so they run in PARALLEL as two
branches from START, then a merge node combines their outputs. This is
the same pattern the eventual Strategy Agent will use once Segmentation
and Competitor/Research join in - proving it out now with 2 agents.

SETUP:
   pip install langgraph

RUN (from project root):
   python -m src.orchestrator
"""

import operator
from typing import Annotated, Optional, TypedDict

from langgraph.graph import StateGraph, START, END

from src.agents.review_analysis import analyze as run_review_analysis
from src.agents.data_analyst import analyze as run_data_analysis


# ---------- Shared graph state ----------
# Parallel branches write to DIFFERENT keys, so no merge conflicts.
# (If two parallel nodes ever need to write the SAME key, that key needs
# an Annotated reducer, e.g. Annotated[list, operator.add] - not needed yet.)

class GraphState(TypedDict):
    question: str
    app_filter: Optional[str]
    review_analysis: Optional[dict]
    retrieved_review_count: Optional[int]
    data_analysis: Optional[dict]
    data_analysis_sql: Optional[str]
    combined_summary: Optional[str]


# ---------- Nodes ----------

def review_analysis_node(state: GraphState) -> dict:
    result, retrieved = run_review_analysis(
        question=state["question"],
        app_filter=state.get("app_filter"),
    )
    return {
        "review_analysis": result.model_dump(),
        "retrieved_review_count": len(retrieved),
    }


def data_analyst_node(state: GraphState) -> dict:
    result, df, sql = run_data_analysis(question=state["question"])
    return {
        "data_analysis": result.model_dump(),
        "data_analysis_sql": sql,
    }


def merge_node(state: GraphState) -> dict:
    """Combine both branches' outputs. No synthesis/reasoning yet - that's
    the future Strategy Agent's job. This just confirms both branches
    landed correctly in shared state."""
    parts = []
    if state.get("review_analysis"):
        parts.append(f"[Reviews] {state['review_analysis']['summary']}")
    if state.get("data_analysis"):
        parts.append(f"[Orders data] {state['data_analysis']['summary']}")
    return {"combined_summary": "\n".join(parts)}


# ---------- Graph assembly ----------

def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("review_analysis", review_analysis_node)
    graph.add_node("data_analyst", data_analyst_node)
    graph.add_node("merge", merge_node)

    # fan-out: both branches start in parallel from START
    graph.add_edge(START, "review_analysis")
    graph.add_edge(START, "data_analyst")

    # fan-in: merge waits for BOTH branches before running
    graph.add_edge("review_analysis", "merge")
    graph.add_edge("data_analyst", "merge")

    graph.add_edge("merge", END)
    return graph.compile()


if __name__ == "__main__":
    app = build_graph()

    initial_state: GraphState = {
        "question": "What are the most common complaints about delivery time, and which cities have the highest order values?",
        "app_filter": "swiggy",
        "review_analysis": None,
        "retrieved_review_count": None,
        "data_analysis": None,
        "data_analysis_sql": None,
        "combined_summary": None,
    }

    final_state = app.invoke(initial_state)

    import json
    print("--- Review analysis ---")
    print(json.dumps(final_state["review_analysis"], indent=2))

    print("\n--- Data analysis ---")
    print(f"SQL used: {final_state['data_analysis_sql']}")
    print(json.dumps(final_state["data_analysis"], indent=2))

    print("\n--- Combined summary ---")
    print(final_state["combined_summary"])