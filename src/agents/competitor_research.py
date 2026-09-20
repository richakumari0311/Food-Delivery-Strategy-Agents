"""Competitor research agent using Tavily and Gemini."""

import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from tavily import TavilyClient

from src.utils.retry import with_llm_retry


load_dotenv()


GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
MAX_SEARCH_RESULTS = 6
MAX_RESULT_CONTENT_LENGTH = 500


class CompetitorResearchResult(BaseModel):
    """Structured output returned by the competitor research agent."""

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
    confidence_note: str | None = Field(
        default=None,
        description=(
            "Note when search results are thin, conflicting, "
            "or the question requires a narrower query."
        ),
    )


@with_llm_retry(max_attempts=3, initial_wait=3.0)
def search_web(
    query: str,
    max_results: int = MAX_SEARCH_RESULTS,
) -> list[dict]:
    """Search the web using Tavily and return normalized results."""
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


def research(
    question: str,
) -> tuple[CompetitorResearchResult, list[dict]]:
    """Research a question and return the structured result and sources."""
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
        (
            f"[{result['title']}]({result['url']})\n"
            f"{result['content'][:MAX_RESULT_CONTENT_LENGTH]}"
        )
        for result in results
    )

    source_urls = [result["url"] for result in results]

    prompt = (
        f"QUESTION: {question}\n\n"
        f"WEB SEARCH RESULTS ({len(results)}):\n"
        f"{results_block}\n\n"
        "Turn this into a structured research finding. "
        "Use ONLY information present in the search results above. "
        "Do not add outside knowledge or make unsupported assumptions. "
        "Cite the specific source URLs that support each finding. "
        "If the results are thin, vague, conflicting, or do not clearly "
        "answer the question, explain that in confidence_note instead "
        "of filling the gaps."
    )

    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=0,
    )

    result = _synthesize(llm, prompt)

    if not result.sources:
        result.sources = source_urls

    return result, results


@with_llm_retry()
def _synthesize(
    llm: ChatGoogleGenerativeAI,
    prompt: str,
) -> CompetitorResearchResult:
    """Generate a structured research result from search results."""
    structured_llm = llm.with_structured_output(
        CompetitorResearchResult
    )
    return structured_llm.invoke(prompt)


def main() -> None:
    """Run a sample competitor research query."""
    question = (
        "What recent strategic moves (last few months) have "
        "Swiggy and Zomato/Eternal made in the Indian quick-commerce "
        "or food delivery space?"
    )

    print(f"Question: {question}\n")

    result, raw_results = research(question)

    print(f"--- Raw search results ({len(raw_results)}) ---")

    for search_result in raw_results:
        print(
            f"  {search_result['title']} - "
            f"{search_result['url']}"
        )

    print("\n--- Structured result ---")
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()