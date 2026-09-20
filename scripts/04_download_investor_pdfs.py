"""Download investor-relations PDFs for the RAG corpus."""

import re
from pathlib import Path
from urllib.parse import urlparse

import pdfplumber
import requests


OUT_DIR = Path("data/raw/investor_pdfs")
MIN_PAGES_FOR_SUBSTANTIVE = 3
REQUEST_TIMEOUT = 30

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
}

SOURCES = [
    # Swiggy
    {
        "company": "swiggy",
        "doc_type": "sx_filing",
        "url": (
            "https://www.swiggy.com/corporate/wp-content/uploads/2026/08/"
            "SWIGGY_06082026090516_SE_Intimations_Capital_Markets_Day.pdf"
        ),
    },
    {
        "company": "swiggy",
        "doc_type": "corporate_deck",
        "url": (
            "https://www.swiggy.com/corporate/wp-content/uploads/2025/11/"
            "Swiggy-Corporate-Presentation_Nov-25.pdf"
        ),
    },
    {
        "company": "swiggy",
        "doc_type": "corporate_deck",
        "url": (
            "https://www.swiggy.com/corporate/wp-content/uploads/2025/08/"
            "Corporate-Deck-FY24-25.pdf"
        ),
    },
    {
        "company": "swiggy",
        "doc_type": "annual_report",
        "url": (
            "https://www.swiggy.com/corporate/wp-content/uploads/2026/07/"
            "annual-report-2025-26.pdf"
        ),
    },
    {
        "company": "swiggy",
        "doc_type": "annual_report",
        "url": (
            "https://www.swiggy.com/corporate/wp-content/uploads/2025/07/"
            "Swiggy-Annual-Report-FY-2024-25.pdf"
        ),
    },
    # Eternal / Zomato
    {
        "company": "eternal",
        "doc_type": "annual_report",
        "url": (
            "https://b.zmtcdn.com/investor-relations/"
            "Eternal_Annual_Report_2025-26.pdf"
        ),
    },
    {
        "company": "eternal",
        "doc_type": "annual_report",
        "url": (
            "http://b.zmtcdn.com/investor-relations/"
            "Eternal_Annual_Report_2024-25.pdf"
        ),
    },
    {
        "company": "eternal",
        "doc_type": "shareholder_letter",
        "url": (
            "https://b.zmtcdn.com/investor-relations/"
            "Eternal_Limited_Shareholders_Letter_Q1FY27_Results.pdf"
        ),
    },
    {
        "company": "eternal",
        "doc_type": "shareholder_letter",
        "url": (
            "https://b.zmtcdn.com/investor-relations/"
            "Eternal_Limited_Shareholders_Letter_Q4FY26_Results.pdf"
        ),
    },
    {
        "company": "eternal",
        "doc_type": "shareholder_letter",
        "url": (
            "https://b.zmtcdn.com/investor-relations/"
            "Eternal_Shareholders_Letter_Q3FY26_Results.pdf"
        ),
    },
    {
        "company": "eternal",
        "doc_type": "shareholder_letter",
        "url": (
            "https://b.zmtcdn.com/investor-relations/"
            "Eternal_Shareholders_Letter_Q2FY26_Results.pdf"
        ),
    },
    {
        "company": "eternal",
        "doc_type": "shareholder_letter",
        "url": (
            "https://b.zmtcdn.com/investor-relations/"
            "Eternal_Shareholders_Letter_Q1FY26_Results.pdf"
        ),
    },
]


def safe_filename(url: str) -> str:
    """Return a filesystem-safe filename derived from a URL."""
    filename = Path(urlparse(url).path).name
    return re.sub(r"[^A-Za-z0-9_.-]", "_", filename)


def download(url: str, destination: Path) -> bool:
    """Download a PDF and save it to the destination path."""
    try:
        response = requests.get(
            url,
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "").lower()
        is_pdf = "pdf" in content_type or response.content.startswith(
            b"%PDF"
        )

        if not is_pdf:
            print(
                "  WARNING: response does not look like a PDF "
                f"(content-type={content_type})"
            )
            return False

        destination.write_bytes(response.content)
        return True

    except requests.RequestException as exc:
        print(f"  FAILED: {exc}")
        return False


def page_count(pdf_path: Path) -> int:
    """Return the number of pages in a PDF, or -1 if unreadable."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            return len(pdf.pages)
    except (OSError, pdfplumber.PDFSyntaxError):
        return -1


def main() -> None:
    """Download, inspect, and summarize investor-relations PDFs."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for source in SOURCES:
        filename = f"{source['company']}_{safe_filename(source['url'])}"
        destination = OUT_DIR / filename

        print(
            f"Downloading [{source['company']}/{source['doc_type']}] "
            f"{source['url']}"
        )

        if not download(source["url"], destination):
            results.append(
                {
                    **source,
                    "filename": filename,
                    "status": "failed",
                    "pages": None,
                }
            )
            continue

        pages = page_count(destination)
        size_kb = destination.stat().st_size / 1024

        low_value = (
            source["doc_type"] == "sx_filing"
            or (
                pages != -1
                and pages < MIN_PAGES_FOR_SUBSTANTIVE
            )
        )

        flag = (
            "LOW-VALUE (review before using)"
            if low_value
            else "ok"
        )

        print(
            f"  saved -> {destination.name} "
            f"({size_kb:.0f} KB, {pages} pages) [{flag}]"
        )

        results.append(
            {
                **source,
                "filename": filename,
                "status": "ok",
                "pages": pages,
                "flag": flag,
            }
        )

    print("\n--- Summary ---")

    for result in results:
        if result["status"] == "failed":
            print(
                f"  FAILED  {result['company']:8s} "
                f"{result['url']}"
            )
            continue

        marker = (
            " <-- CHECK"
            if result.get("flag", "").startswith("LOW")
            else ""
        )

        print(
            f"  ok      {result['company']:8s} "
            f"{result['doc_type']:18s} "
            f"{result['pages']} pages  "
            f"{result['filename']}{marker}"
        )

    print(f"\nAll files in: {OUT_DIR}")
    print(
        "Review or exclude LOW-VALUE files before running the parser "
        "if they should not be included in the RAG corpus."
    )


if __name__ == "__main__":
    main()