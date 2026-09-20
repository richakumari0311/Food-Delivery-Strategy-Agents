"""Strategy synthesis agent combining specialist agent outputs."""

import os
from typing import Literal

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from src.agents.competitor_research import research as competitor_research
from src.agents.data_analyst import analyze as data_analyze
from src.agents.review_analysis import analyze as review_analyze
from src.agents.segmentation import run_segmentation
from src.utils.retry import with_llm_retry

load_dotenv()

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")


class Recommendation(BaseModel):
    """Structured strategic recommendation."""

    title: str = Field(
        description="Short, specific, actionable recommendation title."
    )
    rationale: str = Field(
        description="Why this matters, referencing the supporting evidence."
    )
    supporting_evidence: list[str] = Field(
        description=(
            "Agent findings supporting the recommendation, including the "
            "specific evidence used."
        )
    )
    priority: Literal["high", "medium", "low"]


class StrategyResult(BaseModel):
    """Structured strategy synthesis from specialist agent outputs."""

    executive_summary: str = Field(
        description=(
            "3-4 sentence overview of the strategic situation and top priorities."
        )
    )
    recommendations: list[Recommendation] = Field(
        description="3-6 actionable recommendations ordered by priority."
    )
    data_quality_caveats: list[str] = Field(
        description=(
            "Explicit notes on which inputs are weak, synthetic, or thin "
            "and should not be over-relied upon."
        )
    )
    confidence_note: str | None = Field(
        default=None,
        description=(
            "Note conflicts between inputs or areas where more data is needed "
            "before acting."
        )
    )


def gather_inputs(
    review_question: str,
    data_question: str,
    competitor_question: str,
) -> dict:
    """Run specialist agents and collect their outputs."""
    print("Running Review Analysis Agent ...")
    review_result, _ = review_analyze(review_question)

    print("Running Data Analyst Agent ...")
    data_result, _, _ = data_analyze(data_question)

    print("Running Consumer Segmentation Agent ...")
    persona_set, _ = run_segmentation()

    print("Running Competitor/Research Agent ...")
    competitor_result, _ = competitor_research(competitor_question)

    return {
        "review_analysis": review_result.model_dump(),
        "data_analysis": data_result.model_dump(),
        "segmentation": {
            "personas": [
                persona.model_dump()
                for persona in persona_set.personas
            ],
            "note": (
                "Based on KMeans clustering with modest silhouette scores "
                "(~0.12). Segments are overlapping and use a synthetic dataset."
            ),
        },
        "competitor_research": competitor_result.model_dump(),
    }


def synthesize(
    business_question: str,
    inputs: dict,
) -> StrategyResult:
    """Synthesize specialist findings into a strategy result."""
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=0.2,
    )

    prompt = f"""
You are a strategy analyst synthesizing findings from four specialist agents
to answer a business question. Weigh the inputs according to their data
quality rather than treating them as equally reliable.

- REVIEW ANALYSIS and COMPETITOR RESEARCH are grounded in REAL data, including
  actual user reviews and market or financial reporting. Treat these as
  higher-trust inputs.
- DATA ANALYST and SEGMENTATION use a confirmed SYNTHETIC dataset with weak
  or near-uniform signal in several fields. Treat specific numbers from these
  inputs as directional or illustrative only, not as confident facts about
  the real business.
- Do not present synthetic-data findings with the same confidence as
  real-world findings.

BUSINESS QUESTION:
{business_question}

=== REVIEW ANALYSIS (real user review data) ===
{inputs["review_analysis"]}

=== DATA ANALYST (synthetic order dataset) ===
{inputs["data_analysis"]}

=== CONSUMER SEGMENTATION (synthetic dataset, weak cluster separation) ===
{inputs["segmentation"]}

=== COMPETITOR RESEARCH (real web/news data) ===
{inputs["competitor_research"]}

Synthesize these findings into recommendations.

Every recommendation must identify which agent findings support it.
Explicitly flag in data_quality_caveats which parts of the reasoning rely on
the weaker synthetic-data inputs.

If inputs conflict or leave important gaps, state that in confidence_note
rather than presenting an unsupported conclusion.
"""

    return _invoke_synthesis(llm, prompt)


@with_llm_retry()
def _invoke_synthesis(
    llm: ChatGoogleGenerativeAI,
    prompt: str,
) -> StrategyResult:
    """Generate the structured strategy synthesis."""
    return llm.with_structured_output(StrategyResult).invoke(prompt)


def main() -> None:
    """Run the specialist agents and generate a strategy synthesis."""
    business_question = (
        "What should Swiggy prioritize over the next 1-2 quarters to improve "
        "customer retention and competitive position against Zomato/Eternal?"
    )

    inputs = gather_inputs(
        review_question=(
            "What are the most common complaints about delivery time "
            "and order accuracy?"
        ),
        data_question=(
            "Which cities have the highest average order value, and how "
            "does repeat order rate vary by city?"
        ),
        competitor_question=(
            "What recent strategic moves have Swiggy and Zomato/Eternal "
            "made in the Indian quick-commerce or food delivery space?"
        ),
    )

    print("\nSynthesizing strategy ...")
    result = synthesize(business_question, inputs)

    print("\n--- Strategy Result ---")
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()