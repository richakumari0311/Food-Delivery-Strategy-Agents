"""
Competitor/Research Agent (standalone, pre-LangGraph).

Unlike the other agents, this one needs LIVE external information (market
news, competitor moves, pricing) that isn't in your Postgres tables at all.

Uses Tavily for search instead of Gemini's native Google Search grounding -
grounding turned out to not be available on Gemini's free API tier (confirmed
by testing: switching models didn't change the error, only the model name
in it), so this decouples search from Gemini's quota entirely.

Two-step flow, same pattern as the other agents:
  1. Tavily search -> real, current web results with real URLs (not
     LLM-guessed or metadata-extracted)
  2. Plain Gemini call (with_structured_output, no tools needed since
     Tavily already did the retrieval) -> turns those results into a
     validated Pydantic object, grounded only in what was actually returned

SETUP:
   pip install tavily-python langchain-google-genai pydantic python-dotenv
   Get a free API key at https://tavily.com (no credit card required,
   ~1000 searches/month free) and add to .env:
     TAVILY_API_KEY=your_key_here

RUN:
   python src/agents/competitor_research.py
"""

import os
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from tavily import TavilyClient
from langchain_google_genai import ChatGoogleGenerativeAI

from src.utils.retry import with_llm_retry

load_dotenv()

# Override in .env, e.g. GEMINI_MODEL=gemini-3.5-flash-lite for a higher
# free-tier daily quota while iterating. No grounding dependency now, so
# any Flash-class model works fine here.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
MAX_SEARCH_RESULTS = 6


class CompetitorResearchResult(BaseModel):
    summary: str = Field(description="2-3 sentence answer to the research question")
    key_findings: list[str] = Field(description="3-5 specific, factual findings")
    sources: list[str] = Field(default_factory=list, description="URLs supporting the findings")
    confidence_note: Optional[str] = Field(
        default=None,
        description="Note if the search results are thin, conflicting, or the question needs a narrower query"
    )


@with_llm_retry(max_attempts=3, initial_wait=3.0)
def search_web(query: str, max_results: int = MAX_SEARCH_RESULTS) -> list[dict]:
    client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    response = client.search(query=query, max_results=max_results, search_depth="advanced")
    return [
        {"title": r.get("title", ""), "url": r.get("url", ""), "content": r.get("content", "")}
        for r in response.get("results", [])
    ]


def research(question: str) -> tuple[CompetitorResearchResult, list[dict]]:
    results = search_web(question)

    if not results:
        return CompetitorResearchResult(
            summary="No search results returned.",
            key_findings=[],
            sources=[],
            confidence_note="Tavily returned no results for this query - try rephrasing.",
        ), []

    results_block = "\n\n".join(
        f"[{r['title']}]({r['url']})\n{r['content'][:500]}" for r in results
    )
    all_urls = [r["url"] for r in results]

    prompt = (
        f"QUESTION: {question}\n\n"
        f"WEB SEARCH RESULTS ({len(results)}):\n{results_block}\n\n"
        "Turn this into a structured research finding. Use ONLY information "
        "present in the search results above - do not add outside knowledge "
        "or assume anything beyond what's stated. Cite the specific source "
        "URLs that support each finding. If the results are thin, vague, or "
        "don't clearly answer the question, say so in confidence_note rather "
        "than filling gaps."
    )

    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0)
    result = _synthesize(llm, prompt)

    if not result.sources:
        result.sources = all_urls

    return result, results


@with_llm_retry()
def _synthesize(llm, prompt: str) -> CompetitorResearchResult:
    return llm.with_structured_output(CompetitorResearchResult).invoke(prompt)


if __name__ == "__main__":
    question = "What recent strategic moves (last few months) have Swiggy and Zomato/Eternal made in the Indian quick-commerce or food delivery space?"
    print(f"Question: {question}\n")

    result, raw_results = research(question)

    print(f"--- Raw search results ({len(raw_results)}) ---")
    for r in raw_results:
        print(f"  {r['title']} - {r['url']}")

    print("\n--- Structured result ---")
    print(result.model_dump_json(indent=2))