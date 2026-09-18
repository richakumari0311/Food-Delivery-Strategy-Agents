"""
Strategy Agent (standalone, pre-LangGraph).

The one agent that doesn't retrieve anything itself - it takes the outputs
of the other four agents and synthesizes them into recommendations.

IMPORTANT DESIGN DECISION: the four inputs are NOT equally trustworthy, and
this agent is explicitly told that:
  - Review Analysis + Competitor/Research are grounded in REAL external data
    (actual user reviews, actual news/financial reporting) - higher trust.
  - Data Analyst + Segmentation run on the Kaggle dataset, which has been
    repeatedly confirmed (across every prior agent test) to be synthetic
    with weak/near-uniform signal in several fields - directional only,
    not a source of confident quantitative claims.
A naive synthesis would treat all four inputs as equally solid. This one
is instructed not to, and must say so in its output.

SETUP:
   pip install langchain-google-genai pydantic python-dotenv

RUN (imports and runs all 4 other agents live, then synthesizes):
   python src/agents/strategy.py

NOTE: update the import lines below if your other agent files aren't named
review_analysis.py / data_analyst.py / segmentation.py / competitor_research.py
"""

import os
from typing import Literal, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

from src.utils.retry import with_llm_retry

load_dotenv()

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

# --- Import the other four agents. Adjust these if your filenames differ. ---
from src.agents.review_analysis import analyze as review_analyze
from src.agents.data_analyst import analyze as data_analyze
from src.agents.segmentation import run_segmentation
from src.agents.competitor_research import research as competitor_research


class Recommendation(BaseModel):
    title: str = Field(description="Short, specific, actionable recommendation title")
    rationale: str = Field(description="Why this matters, referencing the specific evidence behind it")
    supporting_evidence: list[str] = Field(
        description="Which agent(s) this is based on and what specifically they found, "
                    "e.g. 'Review Analysis: 28% of Swiggy reviews are 1-star, dominated by delivery delay complaints'"
    )
    priority: Literal["high", "medium", "low"]


class StrategyResult(BaseModel):
    executive_summary: str = Field(description="3-4 sentence overview of the strategic situation and top priorities")
    recommendations: list[Recommendation] = Field(description="3-6 recommendations, ordered by priority")
    data_quality_caveats: list[str] = Field(
        description="Explicit notes on which inputs are weak/synthetic/thin and shouldn't be over-relied upon"
    )
    confidence_note: Optional[str] = Field(
        default=None,
        description="Note any conflicts between inputs, or areas where more/better data is needed before acting"
    )


def gather_inputs(review_question: str, data_question: str, competitor_question: str) -> dict:
    """Run all four specialist agents and collect their outputs."""
    print("Running Review Analysis Agent ...")
    review_result, _ = review_analyze(review_question)

    print("Running Data Analyst Agent ...")
    data_result, _, _ = data_analyze(data_question)

    print("Running Consumer Segmentation Agent ...")
    persona_set, profile = run_segmentation()

    print("Running Competitor/Research Agent ...")
    competitor_result, _ = competitor_research(competitor_question)

    return {
        "review_analysis": review_result.model_dump(),
        "data_analysis": data_result.model_dump(),
        "segmentation": {
            "personas": [p.model_dump() for p in persona_set.personas],
            "note": "Based on KMeans clustering with modest silhouette scores (~0.12) - "
                    "segments are real but overlapping, on a synthetic dataset.",
        },
        "competitor_research": competitor_result.model_dump(),
    }


def synthesize(business_question: str, inputs: dict) -> StrategyResult:
    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0.2)

    prompt = f"""You are a strategy analyst synthesizing findings from four specialist agents
to answer a business question. You must weigh the inputs according to their
data quality, not treat them as equally reliable:

- REVIEW ANALYSIS and COMPETITOR RESEARCH are grounded in REAL data (actual
  user reviews, actual market/financial reporting). Treat these as high-trust.
- DATA ANALYST and SEGMENTATION run on a confirmed-SYNTHETIC dataset with
  weak or near-uniform signal in several fields. Treat specific numbers from
  these as directional/illustrative only, not confident facts about the real
  business. Do not present synthetic-data numbers with the same confidence
  as real-world findings.

BUSINESS QUESTION: {business_question}

=== REVIEW ANALYSIS (real user review data) ===
{inputs['review_analysis']}

=== DATA ANALYST (synthetic order dataset) ===
{inputs['data_analysis']}

=== CONSUMER SEGMENTATION (synthetic dataset, weak cluster separation) ===
{inputs['segmentation']}

=== COMPETITOR RESEARCH (real web/news data) ===
{inputs['competitor_research']}

Synthesize these into recommendations. Every recommendation must cite which
agent(s)' findings it's based on. Explicitly flag in data_quality_caveats
which parts of your reasoning lean on the weaker synthetic-data inputs.
If any inputs conflict or leave gaps, say so in confidence_note rather than
papering over it."""

    return _invoke_synthesis(llm, prompt)


@with_llm_retry()
def _invoke_synthesis(llm, prompt: str) -> StrategyResult:
    return llm.with_structured_output(StrategyResult).invoke(prompt)


if __name__ == "__main__":
    business_question = (
        "What should Swiggy prioritize over the next 1-2 quarters to improve "
        "customer retention and competitive position against Zomato/Eternal?"
    )

    inputs = gather_inputs(
        review_question="What are the most common complaints about delivery time and order accuracy?",
        data_question="Which cities have the highest average order value, and how does repeat order rate vary by city?",
        competitor_question="What recent strategic moves have Swiggy and Zomato/Eternal made in the Indian quick-commerce or food delivery space?",
    )

    print("\nSynthesizing strategy ...")
    result = synthesize(business_question, inputs)

    print("\n--- Strategy Result ---")
    print(result.model_dump_json(indent=2))