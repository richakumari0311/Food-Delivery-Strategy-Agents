"""Competitor research agent using Tavily and Gemini."""

import os
from typing import Optional

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from tavily import TavilyClient

from src.utils.retry import with_llm_retry


load_dotenv()

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
MAX_SEARCH_RESULTS = 6


class CompetitorResearchResult(BaseModel):
    """Structured output for competitor research."""

    summary: str = Field(
        description="A 2-3 sentence answer to the research question."
    )
    key_findings: list[str] = Field(
        description="3-5 specific, factual findings."
    )
    sources: list[str] = Field(
        default_factory=list,
        description="URLs supporting the findings.",
    )
    confidence_note: Optional[str] = Field(
        default=None,
        description=(
            "Note if the search results are thin, conflicting, "
            "or the question needs a narrower query."
        ),
    )


@with_llm_retry(max_attempts=3, initial_wait=3.0)
def search_web(
    query: str,
    max_results: int = MAX_SEARCH_RESULTS,
) -> list[dict]:
    """Search the web using Tavily."""

    client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    response = client.search(
        query=query,
        max_results=max_results,
        search_depth="advanced",
    )

    return [
        {
            "title": result.get("title", ""),
            "url": result.get("url", ""),
            "content": result.get("content", ""),
        }
        for result in response.get("results", [])
    ]


@with_llm_retry()
def _synthesize(
    llm: ChatGoogleGenerativeAI,
    prompt: str,
) -> CompetitorResearchResult:
    """Convert web search results into structured research output."""

    return llm.with_structured_output(
        CompetitorResearchResult
    ).invoke(prompt)


def research(
    question: str,
) -> tuple[CompetitorResearchResult, list[dict]]:
    """Research a question using live web results and Gemini."""

    results = search_web(question)

    if not results:
        return (
            CompetitorResearchResult(
                summary="No search results returned.",
                key_findings=[],
                sources=[],
                confidence_note=(
                    "Tavily returned no results for this query. "
                    "Try rephrasing the question."
                ),
            ),
            [],
        )

    results_block = "\n\n".join(
        f"[{result['title']}]({result['url']})\n"
        f"{result['content'][:500]}"
        for result in results
    )

    prompt = (
        f"QUESTION: {question}\n\n"
        f"WEB SEARCH RESULTS ({len(results)}):\n"
        f"{results_block}\n\n"
        "Turn this into a structured research finding. "
        "Use ONLY information present in the search results above. "
        "Do not add outside knowledge or make unsupported assumptions. "
        "Cite the specific source URLs that support each finding. "
        "If the results are thin, vague, conflicting, or do not clearly "
        "answer the question, explain this in confidence_note instead "
        "of filling gaps."
    )

    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=0,
        timeout=60,
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )

    result = _synthesize(llm, prompt)

    if not result.sources:
        result.sources = [result["url"] for result in results]

    return result, results


if __name__ == "__main__":
    question = (
        "What recent strategic moves (last few months) have Swiggy and "
        "Zomato/Eternal made in the Indian quick-commerce or food delivery space?"
    )

    print(f"Question: {question}\n")

    result, raw_results = research(question)

    print(f"--- Raw search results ({len(raw_results)}) ---")

    for result in raw_results:
        print(f"  {result['title']} - {result['url']}")

    print("\n--- Structured result ---")
    print(result.model_dump_json(indent=2))

