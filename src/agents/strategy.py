"""Strategy agent that synthesizes specialist agent outputs."""

import os
from typing import Any, Literal, Optional

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
    """Actionable strategic recommendation."""

    title: str = Field(
        description="Short, specific, actionable recommendation title."
    )
    rationale: str = Field(
        description=(
            "Why the recommendation matters, referencing "
            "the supporting evidence."
        )
    )
    supporting_evidence: list[str] = Field(
        description=(
            "Agent findings supporting the recommendation, "
            "including the relevant evidence."
        )
    )
    priority: Literal["high", "medium", "low"]


class StrategyResult(BaseModel):
    """Structured output containing strategic recommendations."""

    executive_summary: str = Field(
        description=(
            "A 3-4 sentence overview of the strategic situation "
            "and top priorities."
        )
    )
    recommendations: list[Recommendation] = Field(
        description="3-6 strategic recommendations ordered by priority."
    )
    data_quality_caveats: list[str] = Field(
        description=(
            "Explicit notes about weak, synthetic, or limited inputs "
            "that should not be over-relied upon."
        )
    )
    confidence_note: Optional[str] = Field(
        default=None,
        description=(
            "Conflicts between inputs or areas where additional "
            "data is needed before acting."
        )
    )


def gather_inputs(
    review_question: str,
    data_question: str,
    competitor_question: str,
) -> dict[str, Any]:
    """Run specialist agents and collect their structured outputs."""

    print("Running Review Analysis Agent...")
    review_result, _ = review_analyze(review_question)

    print("Running Data Analyst Agent...")
    data_result, _, _ = data_analyze(data_question)

    print("Running Consumer Segmentation Agent...")
    persona_set, profile = run_segmentation()

    print("Running Competitor/Research Agent...")
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
                "(approximately 0.12). Segments overlap and are based on "
                "a synthetic dataset."
            ),
        },
        "competitor_research": competitor_result.model_dump(),
    }


def _create_llm() -> ChatGoogleGenerativeAI:
    """Create the configured Gemini client."""

    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=0.2,
        timeout=60,
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )


def _build_synthesis_prompt(
    business_question: str,
    inputs: dict[str, Any],
) -> str:
    """Build the strategy synthesis prompt."""

    return f"""
You are a strategy analyst synthesizing findings from four specialist
agents to answer a business question.

Weigh the inputs according to their data quality. Do not treat all sources
as equally reliable.

REVIEW ANALYSIS and COMPETITOR RESEARCH are grounded in real-world data,
including user reviews and current market reporting. Treat these as
higher-trust inputs.

DATA ANALYST and SEGMENTATION use a confirmed synthetic dataset with weak
or near-uniform signal in several fields. Treat specific numbers from
these sources as directional or illustrative only. Do not present
synthetic-data numbers with the same confidence as real-world findings.

BUSINESS QUESTION:
{business_question}

=== REVIEW ANALYSIS ===
{inputs["review_analysis"]}

=== DATA ANALYST ===
{inputs["data_analysis"]}

=== CONSUMER SEGMENTATION ===
{inputs["segmentation"]}

=== COMPETITOR RESEARCH ===
{inputs["competitor_research"]}

Synthesize the findings into strategic recommendations.

Every recommendation must identify the agent findings it is based on.
Explicitly identify reasoning that relies on weaker synthetic-data inputs
in data_quality_caveats.

If inputs conflict or leave important gaps, state this in confidence_note
rather than resolving the conflict without evidence.
""".strip()


@with_llm_retry()
def _invoke_synthesis(
    llm: ChatGoogleGenerativeAI,
    prompt: str,
) -> StrategyResult:
    """Generate the structured strategy synthesis."""

    return llm.with_structured_output(StrategyResult).invoke(prompt)


def synthesize(
    business_question: str,
    inputs: dict[str, Any],
) -> StrategyResult:
    """Synthesize specialist findings into strategic recommendations."""

    llm = _create_llm()
    prompt = _build_synthesis_prompt(
        business_question=business_question,
        inputs=inputs,
    )

    return _invoke_synthesis(llm, prompt)


if __name__ == "__main__":
    business_question = (
        "What should Swiggy prioritize over the next 1-2 quarters "
        "to improve customer retention and competitive position "
        "against Zomato/Eternal?"
    )

    inputs = gather_inputs(
        review_question=(
            "What are the most common complaints about delivery time "
            "and order accuracy?"
        ),
        data_question=(
            "Which cities have the highest average order value, and "
            "how does repeat order rate vary by city?"
        ),
        competitor_question=(
            "What recent strategic moves have Swiggy and Zomato/Eternal "
            "made in the Indian quick-commerce or food delivery space?"
        ),
    )

    print("\nSynthesizing strategy...")
    result = synthesize(business_question, inputs)

    print("\n--- Strategy Result ---")
    print(result.model_dump_json(indent=2))
