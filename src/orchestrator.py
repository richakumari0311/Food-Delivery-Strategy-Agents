"""LangGraph orchestrator for the four-agent parallel workflow and strategy synthesis."""

import json
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from src.agents.competitor_research import research as competitor_research
from src.agents.data_analyst import analyze as data_analyze
from src.agents.review_analysis import analyze as review_analyze
from src.agents.segmentation import run_segmentation
from src.agents.strategy import synthesize


class GraphState(TypedDict):
    """Shared state passed between LangGraph nodes."""

    review_question: str
    data_question: str
    competitor_question: str
    business_question: str
    app_filter: str | None

    review_analysis: dict | None
    data_analysis: dict | None
    segmentation: dict | None
    competitor_research: dict | None

    strategy: dict | None


def review_analysis_node(state: GraphState) -> dict:
    """Run the review analysis agent."""
    result, _ = review_analyze(
        state["review_question"],
        app_filter=state.get("app_filter"),
    )
    return {"review_analysis": result.model_dump()}


def data_analyst_node(state: GraphState) -> dict:
    """Run the data analyst agent."""
    result, _, _ = data_analyze(state["data_question"])
    return {"data_analysis": result.model_dump()}


def segmentation_node(state: GraphState) -> dict:
    """Run the consumer segmentation agent."""
    persona_set, _ = run_segmentation()

    return {
        "segmentation": {
            "personas": [
                persona.model_dump()
                for persona in persona_set.personas
            ],
            "note": (
                "Based on KMeans clustering with modest silhouette scores "
                "(~0.12). Segments are overlapping and use a synthetic dataset."
            ),
        }
    }


def competitor_research_node(state: GraphState) -> dict:
    """Run the competitor research agent."""
    result, _ = competitor_research(state["competitor_question"])
    return {"competitor_research": result.model_dump()}


def strategy_node(state: GraphState) -> dict:
    """Synthesize the outputs from all four specialist agents."""
    inputs = {
        "review_analysis": state["review_analysis"],
        "data_analysis": state["data_analysis"],
        "segmentation": state["segmentation"],
        "competitor_research": state["competitor_research"],
    }

    result = synthesize(state["business_question"], inputs)
    return {"strategy": result.model_dump()}


def build_graph():
    """Build and compile the five-agent LangGraph workflow."""
    graph = StateGraph(GraphState)

    graph.add_node("review_analysis", review_analysis_node)
    graph.add_node("data_analyst", data_analyst_node)
    graph.add_node("segmentation", segmentation_node)
    graph.add_node("competitor_research", competitor_research_node)
    graph.add_node("strategy", strategy_node)

    # Independent agents run in parallel; strategy fans in after all four complete.
    graph.add_edge(START, "review_analysis")
    graph.add_edge(START, "data_analyst")
    graph.add_edge(START, "segmentation")
    graph.add_edge(START, "competitor_research")

    graph.add_edge("review_analysis", "strategy")
    graph.add_edge("data_analyst", "strategy")
    graph.add_edge("segmentation", "strategy")
    graph.add_edge("competitor_research", "strategy")

    graph.add_edge("strategy", END)

    return graph.compile()


def main() -> None:
    """Run the complete five-agent workflow."""
    app = build_graph()

    initial_state: GraphState = {
        "review_question": (
            "What are the most common complaints about delivery time "
            "and order accuracy?"
        ),
        "data_question": (
            "Which cities have the highest average order value, and how "
            "does repeat order rate vary by city?"
        ),
        "competitor_question": (
            "What recent strategic moves have Swiggy and Zomato/Eternal "
            "made in the Indian quick-commerce or food delivery space?"
        ),
        "business_question": (
            "What should Swiggy prioritize over the next 1-2 quarters "
            "to improve customer retention and competitive position "
            "against Zomato/Eternal?"
        ),
        "app_filter": "swiggy",
        "review_analysis": None,
        "data_analysis": None,
        "segmentation": None,
        "competitor_research": None,
        "strategy": None,
    }

    print(
        "Running full 5-agent graph "
        "(4 parallel branches -> strategy synthesis) ...\n"
    )

    final_state = app.invoke(initial_state)

    print("\n--- Final Strategy Output ---")
    print(json.dumps(final_state["strategy"], indent=2))


if __name__ == "__main__":
    main()