"""Test cases for the Ragas evaluation harness.

The fixed test set intentionally includes a mix of well-supported,
thin, and off-topic questions. This helps verify that faithfulness
and relevancy scores decrease appropriately for questions that are
poorly supported by the available data.
"""

REVIEW_ANALYSIS_CASES = [
    {
        "question": "What are the most common complaints about delivery time?",
        "app_filter": "swiggy",
    },
    {
        "question": "Are customers happy with order accuracy?",
        "app_filter": None,
    },
    {
        "question": "What do users say about customer support responsiveness?",
        "app_filter": "zomato",
    },
    {
        # Deliberately off-topic for this data source.
        "question": (
            "What do reviews say about the company's stock price performance?"
        ),
        "app_filter": None,
    },
]

DATA_ANALYST_CASES = [
    {
        "question": "Which cities have the highest average order value?",
    },
    {
        "question": "How does repeat order rate vary by age group?",
    },
    {
        "question": "What's the relationship between discount usage and order value?",
    },
    {
        # Deliberately outside what this table can answer.
        "question": "What is the company's total quarterly revenue?",
    },
]

COMPETITOR_RESEARCH_CASES = [
    {
        "question": "What recent moves has Swiggy made in quick commerce?",
    },
    {
        "question": "How does Eternal's profitability compare to Swiggy's?",
    },
    {
        "question": "What is Blinkit's market position versus Instamart?",
    },
]