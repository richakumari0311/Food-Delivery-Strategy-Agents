"""
Step 4: Download investor relations PDFs
directly from URLs into data/raw/investor_pdfs/, with a basic check to
flag likely-irrelevant files (e.g. short stock-exchange intimation notices)
before you bother feeding them into the RAG pipeline.

SETUP:
   pip install requests pdfplumber

RUN:
   python scripts/04_download_investor_pdfs.py
"""

import re
from pathlib import Path

import requests
import pdfplumber

OUT_DIR = Path("data/raw/investor_pdfs")

# doc_type is just metadata for later filtering/prioritization -
# "sx_filing" = stock exchange filing/intimation, usually low-value for RAG
SOURCES = [
    # --- Swiggy ---
    {"company": "swiggy", "doc_type": "sx_filing",
     "url": "https://www.swiggy.com/corporate/wp-content/uploads/2026/08/SWIGGY_06082026090516_SE_Intimations_Capital_Markets_Day.pdf"},
    {"company": "swiggy", "doc_type": "corporate_deck",
     "url": "https://www.swiggy.com/corporate/wp-content/uploads/2025/11/Swiggy-Corporate-Presentation_Nov-25.pdf"},
    {"company": "swiggy", "doc_type": "corporate_deck",
     "url": "https://www.swiggy.com/corporate/wp-content/uploads/2025/08/Corporate-Deck-FY24-25.pdf"},
    {"company": "swiggy", "doc_type": "annual_report",
     "url": "https://www.swiggy.com/corporate/wp-content/uploads/2026/07/annual-report-2025-26.pdf"},
    {"company": "swiggy", "doc_type": "annual_report",
     "url": "https://www.swiggy.com/corporate/wp-content/uploads/2025/07/Swiggy-Annual-Report-FY-2024-25.pdf"},

    # --- Eternal / Zomato ---
    {"company": "eternal", "doc_type": "annual_report",
     "url": "https://b.zmtcdn.com/investor-relations/Eternal_Annual_Report_2025-26.pdf"},
    {"company": "eternal", "doc_type": "annual_report",
     "url": "http://b.zmtcdn.com/investor-relations/Eternal_Annual_Report_2024-25.pdf"},
    {"company": "eternal", "doc_type": "shareholder_letter",
     "url": "https://b.zmtcdn.com/investor-relations/Eternal_Limited_Shareholders_Letter_Q1FY27_Results.pdf"},
    {"company": "eternal", "doc_type": "shareholder_letter",
     "url": "https://b.zmtcdn.com/investor-relations/Eternal_Limited_Shareholders_Letter_Q4FY26_Results.pdf"},
    {"company": "eternal", "doc_type": "shareholder_letter",
     "url": "https://b.zmtcdn.com/investor-relations/Eternal_Shareholders_Letter_Q3FY26_Results.pdf"},
    {"company": "eternal", "doc_type": "shareholder_letter",
     "url": "https://b.zmtcdn.com/investor-relations/Eternal_Shareholders_Letter_Q2FY26_Results.pdf"},
    {"company": "eternal", "doc_type": "shareholder_letter",
     "url": "https://b.zmtcdn.com/investor-relations/Eternal_Shareholders_Letter_Q1FY26_Results.pdf"},
]

MIN_PAGES_FOR_SUBSTANTIVE = 3  # fewer pages than this -> flagged as likely low-value

HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}


def safe_filename(url: str) -> str:
    name = url.split("/")[-1]
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def download(url: str, dest: Path) -> bool:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        if "pdf" not in resp.headers.get("Content-Type", "").lower() and not resp.content.startswith(b"%PDF"):
            print(f"  WARNING: response doesn't look like a PDF (content-type={resp.headers.get('Content-Type')})")
            return False
        dest.write_bytes(resp.content)
        return True
    except requests.RequestException as e:
        print(f"  FAILED: {e}")
        return False


def page_count(pdf_path: Path) -> int:
    try:
        with pdfplumber.open(pdf_path) as pdf:
            return len(pdf.pages)
    except Exception:
        return -1


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for src in SOURCES:
        filename = f"{src['company']}_{safe_filename(src['url'])}"
        dest = OUT_DIR / filename
        print(f"Downloading [{src['company']}/{src['doc_type']}] {src['url']}")
        ok = download(src["url"], dest)
        if not ok:
            results.append({**src, "filename": filename, "status": "failed", "pages": None})
            continue

        pages = page_count(dest)
        size_kb = dest.stat().st_size / 1024
        low_value = src["doc_type"] == "sx_filing" or (pages != -1 and pages < MIN_PAGES_FOR_SUBSTANTIVE)
        flag = "LOW-VALUE (review before using)" if low_value else "ok"
        print(f"  saved -> {dest.name}  ({size_kb:.0f} KB, {pages} pages)  [{flag}]")
        results.append({**src, "filename": filename, "status": "ok", "pages": pages, "flag": flag})

    print("\n--- Summary ---")
    for r in results:
        if r["status"] == "failed":
            print(f"  FAILED  {r['company']:8s} {r['url']}")
        else:
            marker = " <-- CHECK" if r.get("flag", "").startswith("LOW") else ""
            print(f"  ok      {r['company']:8s} {r['doc_type']:18s} {r['pages']} pages  {r['filename']}{marker}")

    print(f"\nAll files in: {OUT_DIR}")
    print("Delete/exclude any flagged as LOW-VALUE before running the parser if you don't want them in the RAG corpus.")