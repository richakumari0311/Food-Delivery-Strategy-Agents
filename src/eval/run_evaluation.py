"""Evaluate Q&A agents using Ragas faithfulness and answer relevancy metrics."""

import os
import sys
import types
from typing import Any

# Ragas compatibility shim.
try:
    from langchain_google_vertexai import ChatVertexAI

    shim = types.ModuleType("langchain_community.chat_models.vertexai")
    shim.ChatVertexAI = ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = shim
except ImportError:
    pass

from datasets import Dataset
from langchain_google_genai import ChatGoogleGenerativeAI
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, faithfulness

from src.agents.competitor_research import research as competitor_research
from src.agents.data_analyst import analyze as data_analyze
from src.agents.review_analysis import analyze as review_analyze
from src.eval.test_cases import (
    COMPETITOR_RESEARCH_CASES,
    DATA_ANALYST_CASES,
    REVIEW_ANALYSIS_CASES,
)

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
MAX_CONTEXT_ROWS = 20
MAX_SEARCH_RESULT_LENGTH = 500
MAX_JUDGE_WORKERS = 2
JUDGE_TIMEOUT = 180


def install_ragas_compatibility_shim() -> None:
    """Patch the legacy Vertex AI import expected by older Ragas versions."""
    try:
        from langchain_google_vertexai import ChatVertexAI

        module = types.ModuleType("langchain_community.chat_models.vertexai")
        module.ChatVertexAI = ChatVertexAI
        sys.modules["langchain_community.chat_models.vertexai"] = module
    except ImportError:
        pass


install_ragas_compatibility_shim()

try:
    answer_relevancy.strictness = 1
except (AttributeError, TypeError):
    pass


def get_judge_llm() -> Any:
    """Create the LLM used by Ragas as the evaluation judge."""
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=0,
    )
    return LangchainLLMWrapper(llm)


def get_judge_embeddings():
    """Create the embedding model used by Ragas answer relevancy."""
    from ragas.embeddings import LangchainEmbeddingsWrapper

    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return LangchainEmbeddingsWrapper(embeddings)


def _empty_dataset() -> dict[str, list]:
    """Create an empty Ragas dataset structure."""
    return {
        "question": [],
        "answer": [],
        "contexts": [],
    }


def _log_skipped_case(question: str, error: Exception) -> None:
    """Log a failed evaluation case without stopping the evaluation run."""
    print(
        f"  Skipping case: {question[:60]}... "
        f"({type(error).__name__})"
    )


def build_review_analysis_dataset() -> Dataset:
    """Build a Ragas dataset from Review Analysis agent outputs."""
    rows = _empty_dataset()

    for case in REVIEW_ANALYSIS_CASES:
        try:
            result, reviews = review_analyze(
                case["question"],
                app_filter=case["app_filter"],
            )
        except Exception as exc:
            _log_skipped_case(case["question"], exc)
            continue

        rows["question"].append(case["question"])
        rows["answer"].append(
            f"{result.summary} {' '.join(result.top_complaints)}"
        )
        rows["contexts"].append(
            [review["text"] for review in reviews]
            or ["(no reviews retrieved)"]
        )

    return Dataset.from_dict(rows)


def build_data_analyst_dataset() -> Dataset:
    """Build a Ragas dataset from Data Analyst agent outputs."""
    rows = _empty_dataset()

    for case in DATA_ANALYST_CASES:
        try:
            result, dataframe, sql = data_analyze(case["question"])
        except Exception as exc:
            _log_skipped_case(case["question"], exc)
            continue

        context = (
            f"SQL: {sql}\n"
            f"Results:\n"
            f"{dataframe.head(MAX_CONTEXT_ROWS).to_string(index=False)}"
            if not dataframe.empty
            else "(no rows returned)"
        )

        rows["question"].append(case["question"])
        rows["answer"].append(
            f"{result.summary} {' '.join(result.key_findings)}"
        )
        rows["contexts"].append([context])

    return Dataset.from_dict(rows)


def build_competitor_research_dataset() -> Dataset:
    """Build a Ragas dataset from Competitor Research agent outputs."""
    rows = _empty_dataset()

    for case in COMPETITOR_RESEARCH_CASES:
        try:
            result, raw_results = competitor_research(case["question"])
        except Exception as exc:
            _log_skipped_case(case["question"], exc)
            continue

        contexts = (
            [
                result["content"][:MAX_SEARCH_RESULT_LENGTH]
                for result in raw_results
            ]
            if raw_results
            else ["(no search results)"]
        )

        rows["question"].append(case["question"])
        rows["answer"].append(
            f"{result.summary} {' '.join(result.key_findings)}"
        )
        rows["contexts"].append(contexts)

    return Dataset.from_dict(rows)


def get_run_config():
    """Create Ragas execution settings with conservative concurrency."""
    try:
        from ragas.run_config import RunConfig

        return RunConfig(
            max_workers=MAX_JUDGE_WORKERS,
            timeout=JUDGE_TIMEOUT,
        )
    except ImportError:
        return None


def run_eval(
    agent_name: str,
    dataset: Dataset,
    judge_llm: Any,
    judge_embeddings,
) -> None:
    """Evaluate one agent dataset and persist detailed scores."""
    print(
        f"\n=== Evaluating: {agent_name} "
        f"({len(dataset)} cases) ==="
    )

    if not dataset:
        print("No evaluation cases available.")
        return

    run_config = get_run_config()

    eval_kwargs = {
        "metrics": [faithfulness, answer_relevancy],
        "llm": judge_llm,
        "embeddings": judge_embeddings,
    }

    if run_config is not None:
        eval_kwargs["run_config"] = run_config

    result = evaluate(dataset, **eval_kwargs)
    dataframe = result.to_pandas()

    score_columns = [
        column
        for column in ("faithfulness", "answer_relevancy")
        if column in dataframe.columns
    ]
    identifier_columns = [
        column
        for column in ("question", "user_input")
        if column in dataframe.columns
    ]

    columns = identifier_columns + score_columns
    print(f"Result columns: {dataframe.columns.tolist()}")

    if columns:
        print(dataframe[columns].to_string(index=False))

    for column in score_columns:
        print(f"Mean {column}: {dataframe[column].mean():.3f}")

    output_path = (
        f"eval_results_{agent_name.lower().replace(' ', '_')}.csv"
    )
    dataframe.to_csv(output_path, index=False)

    print(f"Saved results to {output_path}")


def main() -> None:
    """Run the evaluation suite for all supported Q&A agents."""
    judge_llm = get_judge_llm()
    judge_embeddings = get_judge_embeddings()

    datasets = [
        (
            "Review Analysis",
            build_review_analysis_dataset(),
        ),
        (
            "Data Analyst",
            build_data_analyst_dataset(),
        ),
        (
            "Competitor Research",
            build_competitor_research_dataset(),
        ),
    ]

    for agent_name, dataset in datasets:
        run_eval(
            agent_name,
            dataset,
            judge_llm,
            judge_embeddings,
        )

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()

