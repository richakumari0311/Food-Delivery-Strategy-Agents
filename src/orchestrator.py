r"""
Orchestrator (LangGraph) - full 5-agent graph.

Four independent agents run in PARALLEL from START (no dependency between
them - different data sources, different question shapes). The Strategy
Agent is the fan-in point: LangGraph waits for all four parallel branches
to complete before running it, since it has four incoming edges. No
separate "merge" node needed - strategy IS the merge, since synthesis is
its actual job, not just a passthrough.

    START
   /  |  |  \
  RA  DA SEG CR      (parallel, no dependency)
   \  |  |  /
    strategy          (fan-in: waits for all 4)
       |
      END

SETUP:
   pip install langgraph

RUN (from project root):
   python -m src.orchestrator
"""

from typing import Optional, TypedDict

from langgraph.graph import StateGraph, START, END

from src.agents.review_analysis import analyze as review_analyze
from src.agents.data_analyst import analyze as data_analyze
from src.agents.segmentation import run_segmentation
from src.agents.competitor_research import research as competitor_research
from src.agents.strategy import synthesize


# ---------- Shared graph state ----------
# Each parallel node writes to its OWN key - no merge conflicts.
# strategy reads all four once they've landed.

class GraphState(TypedDict):
    # inputs
    review_question: str
    data_question: str
    competitor_question: str
    business_question: str
    app_filter: Optional[str]

    # per-agent outputs (written by the 4 parallel nodes)
    review_analysis: Optional[dict]
    data_analysis: Optional[dict]
    segmentation: Optional[dict]
    competitor_research: Optional[dict]

    # final output (written by strategy)
    strategy: Optional[dict]


# ---------- Nodes ----------

def review_analysis_node(state: GraphState) -> dict:
    result, _ = review_analyze(state["review_question"], app_filter=state.get("app_filter"))
    return {"review_analysis": result.model_dump()}


def data_analyst_node(state: GraphState) -> dict:
    result, _, _ = data_analyze(state["data_question"])
    return {"data_analysis": result.model_dump()}


def segmentation_node(state: GraphState) -> dict:
    persona_set, _ = run_segmentation()
    return {
        "segmentation": {
            "personas": [p.model_dump() for p in persona_set.personas],
            "note": "Based on KMeans clustering with modest silhouette scores (~0.12) - "
                    "segments are real but overlapping, on a synthetic dataset.",
        }
    }


def competitor_research_node(state: GraphState) -> dict:
    result, _ = competitor_research(state["competitor_question"])
    return {"competitor_research": result.model_dump()}


def strategy_node(state: GraphState) -> dict:
    """Fan-in point. Runs only after all 4 parallel branches above have
    written their keys, since it has an incoming edge from each."""
    inputs = {
        "review_analysis": state["review_analysis"],
        "data_analysis": state["data_analysis"],
        "segmentation": state["segmentation"],
        "competitor_research": state["competitor_research"],
    }
    result = synthesize(state["business_question"], inputs)
    return {"strategy": result.model_dump()}


# ---------- Graph assembly ----------

def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("review_analysis", review_analysis_node)
    graph.add_node("data_analyst", data_analyst_node)
    graph.add_node("segmentation", segmentation_node)
    graph.add_node("competitor_research", competitor_research_node)
    graph.add_node("strategy", strategy_node)

    # fan-out: all 4 start in parallel from START
    graph.add_edge(START, "review_analysis")
    graph.add_edge(START, "data_analyst")
    graph.add_edge(START, "segmentation")
    graph.add_edge(START, "competitor_research")

    # fan-in: strategy waits for ALL 4 branches before running
    graph.add_edge("review_analysis", "strategy")
    graph.add_edge("data_analyst", "strategy")
    graph.add_edge("segmentation", "strategy")
    graph.add_edge("competitor_research", "strategy")

    graph.add_edge("strategy", END)
    return graph.compile()


if __name__ == "__main__":
    app = build_graph()

    initial_state: GraphState = {
        "review_question": "What are the most common complaints about delivery time and order accuracy?",
        "data_question": "Which cities have the highest average order value, and how does repeat order rate vary by city?",
        "competitor_question": "What recent strategic moves have Swiggy and Zomato/Eternal made in the Indian quick-commerce or food delivery space?",
        "business_question": "What should Swiggy prioritize over the next 1-2 quarters to improve customer retention and competitive position against Zomato/Eternal?",
        "app_filter": "swiggy",
        "review_analysis": None,
        "data_analysis": None,
        "segmentation": None,
        "competitor_research": None,
        "strategy": None,
    }

    print("Running full 5-agent graph (4 parallel branches -> strategy synthesis) ...\n")
    final_state = app.invoke(initial_state)

    import json
    print("\n--- Final Strategy Output ---")
    print(json.dumps(final_state["strategy"], indent=2))