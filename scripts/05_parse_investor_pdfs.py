"""
Step 5: Extract & chunk text from Swiggy / Eternal investor relations PDFs,
ready for embedding into pgvector later (RAG source for Strategy/Research agents).

Run this AFTER 04_download_investor_pdfs.py has populated data/raw/investor_pdfs/.
Low-value files (SE intimation filings, or anything under MIN_PAGES_TO_KEEP pages
of extractable text) are skipped automatically - no manual deletion needed.

SETUP:
   pip install pymupdf pandas pyarrow tqdm

RUN:
   python scripts/05_parse_investor_pdfs.py

NOTE: uses PyMuPDF (fitz) instead of pdfplumber for extraction - pdfplumber
does detailed layout/table analysis per page which is very slow on large
(300-400+ page) PDFs. PyMuPDF is a much faster C-based extractor; it's a
bit less precise on complex table layouts, but for RAG chunking (where we
just need reasonably clean prose text) that tradeoff is worth it.
"""

import re
from pathlib import Path

import fitz  # PyMuPDF
import pandas as pd
from tqdm import tqdm

PDF_DIR = Path("data/raw/investor_pdfs")
OUT_PATH = Path("data/processed/investor_pdf_chunks.parquet")

CHUNK_SIZE_WORDS = 250   # rough chunk size, word-based
CHUNK_OVERLAP_WORDS = 40
MIN_PAGES_TO_KEEP = 3    # skip anything shorter - likely an intimation/filing, not real content

# extra belt-and-suspenders: skip by filename pattern even if page count check misses it
SKIP_FILENAME_PATTERNS = ["SE_Intimation", "sx_filing"]


def infer_company(filename: str) -> str:
    name = filename.lower()
    if "swiggy" in name:
        return "swiggy"
    if "eternal" in name or "zomato" in name:
        return "eternal"
    return "unknown"


def extract_text_by_page(pdf_path: Path) -> list[dict]:
    pages = []
    doc = fitz.open(pdf_path)
    for i in tqdm(range(len(doc)), desc=f"  extracting {pdf_path.name}", unit="page", leave=False):
        text = doc[i].get_text() or ""
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            pages.append({"page_number": i + 1, "text": text})
    doc.close()
    return pages


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = start + size
        chunks.append(" ".join(words[start:end]))
        start += size - overlap
    return chunks


def process_pdf(pdf_path: Path, pages: list[dict]) -> list[dict]:
    company = infer_company(pdf_path.name)
    print(f"Processing {pdf_path.name} (company={company}, {len(pages)} pages with text) ...")

    rows = []
    for page in pages:
        chunks = chunk_text(page["text"], CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS)
        for j, chunk in enumerate(chunks):
            rows.append({
                "source_file": pdf_path.name,
                "company": company,
                "page_number": page["page_number"],
                "chunk_index": j,
                "chunk_text": chunk,
                "word_count": len(chunk.split()),
            })
    print(f"  produced {len(rows)} chunks")
    return rows


if __name__ == "__main__":
    if not PDF_DIR.exists() or not any(PDF_DIR.glob("*.pdf")):
        raise SystemExit(
            f"No PDFs found in {PDF_DIR}. Download reports and place them there first."
        )

    pdf_paths = sorted(PDF_DIR.glob("*.pdf"))
    print(f"Found {len(pdf_paths)} PDFs to process\n")

    all_rows = []
    skipped = []
    for idx, pdf_path in enumerate(pdf_paths, 1):
        print(f"[{idx}/{len(pdf_paths)}] {pdf_path.name}")
        if any(pat.lower() in pdf_path.name.lower() for pat in SKIP_FILENAME_PATTERNS):
            print(f"Skipping {pdf_path.name} (matches low-value filename pattern)")
            skipped.append(pdf_path.name)
            continue

        pages = extract_text_by_page(pdf_path)
        if len(pages) < MIN_PAGES_TO_KEEP:
            print(f"Skipping {pdf_path.name} (only {len(pages)} pages with text - likely a filing, not a report)")
            skipped.append(pdf_path.name)
            continue

        all_rows.extend(process_pdf(pdf_path, pages))

    if skipped:
        print(f"\nSkipped {len(skipped)} low-value file(s): {skipped}")

    df = pd.DataFrame(all_rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"\nSaved {len(df)} chunks total -> {OUT_PATH}")

    print("\n--- Chunks per source file ---")
    print(df.groupby(["company", "source_file"]).size())

    print("\n--- Sample chunk ---")
    print(df.iloc[0]["chunk_text"][:500])