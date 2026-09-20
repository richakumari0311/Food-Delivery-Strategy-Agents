"""Extract and chunk investor-relations PDFs for the RAG pipeline."""

import re
from pathlib import Path

import fitz  # PyMuPDF
import pandas as pd
from tqdm import tqdm


PDF_DIR = Path("data/raw/investor_pdfs")
OUT_PATH = Path("data/processed/investor_pdf_chunks.parquet")

CHUNK_SIZE_WORDS = 250
CHUNK_OVERLAP_WORDS = 40
MIN_PAGES_TO_KEEP = 3

# Skip files that are unlikely to contain useful RAG content.
SKIP_FILENAME_PATTERNS = [
    "SE_Intimation",
    "sx_filing",
]


def infer_company(filename: str) -> str:
    """Infer the company name from the PDF filename."""
    name = filename.lower()

    if "swiggy" in name:
        return "swiggy"

    if "eternal" in name or "zomato" in name:
        return "eternal"

    return "unknown"


def extract_text_by_page(pdf_path: Path) -> list[dict]:
    """Extract normalized text from each non-empty PDF page."""
    pages = []

    with fitz.open(pdf_path) as document:
        for page_number, page in enumerate(
            tqdm(
                document,
                desc=f"  extracting {pdf_path.name}",
                unit="page",
                leave=False,
            ),
            start=1,
        ):
            text = page.get_text() or ""
            text = re.sub(r"\s+", " ", text).strip()

            if text:
                pages.append(
                    {
                        "page_number": page_number,
                        "text": text,
                    }
                )

    return pages


def chunk_text(
    text: str,
    size: int,
    overlap: int,
) -> list[str]:
    """Split text into overlapping word-based chunks."""
    if size <= 0:
        raise ValueError("Chunk size must be greater than zero.")

    if overlap < 0 or overlap >= size:
        raise ValueError(
            "Chunk overlap must be non-negative and smaller than "
            "the chunk size."
        )

    words = text.split()

    if not words:
        return []

    chunks = []
    step = size - overlap

    for start in range(0, len(words), step):
        chunk = " ".join(words[start:start + size])

        if chunk:
            chunks.append(chunk)

    return chunks


def process_pdf(
    pdf_path: Path,
    pages: list[dict],
) -> list[dict]:
    """Convert extracted PDF pages into RAG-ready chunk records."""
    company = infer_company(pdf_path.name)

    print(
        f"Processing {pdf_path.name} "
        f"(company={company}, {len(pages)} pages with text) ..."
    )

    rows = []

    for page in pages:
        chunks = chunk_text(
            page["text"],
            CHUNK_SIZE_WORDS,
            CHUNK_OVERLAP_WORDS,
        )

        for chunk_index, chunk in enumerate(chunks):
            rows.append(
                {
                    "source_file": pdf_path.name,
                    "company": company,
                    "page_number": page["page_number"],
                    "chunk_index": chunk_index,
                    "chunk_text": chunk,
                    "word_count": len(chunk.split()),
                }
            )

    print(f"  produced {len(rows)} chunks")

    return rows


def should_skip_file(pdf_path: Path) -> bool:
    """Return whether a PDF matches a known low-value filename pattern."""
    filename = pdf_path.name.lower()

    return any(
        pattern.lower() in filename
        for pattern in SKIP_FILENAME_PATTERNS
    )


def main() -> None:
    """Extract, filter, chunk, and save investor-relations PDFs."""
    if not PDF_DIR.exists() or not any(PDF_DIR.glob("*.pdf")):
        raise SystemExit(
            f"No PDFs found in {PDF_DIR}. "
            "Run the download script first."
        )

    pdf_paths = sorted(PDF_DIR.glob("*.pdf"))
    print(f"Found {len(pdf_paths)} PDFs to process\n")

    all_rows = []
    skipped = []

    for index, pdf_path in enumerate(pdf_paths, start=1):
        print(f"[{index}/{len(pdf_paths)}] {pdf_path.name}")

        if should_skip_file(pdf_path):
            print(
                "Skipping file "
                "(matches low-value filename pattern)"
            )
            skipped.append(pdf_path.name)
            continue

        pages = extract_text_by_page(pdf_path)

        if len(pages) < MIN_PAGES_TO_KEEP:
            print(
                f"Skipping file "
                f"(only {len(pages)} pages with text)"
            )
            skipped.append(pdf_path.name)
            continue

        all_rows.extend(process_pdf(pdf_path, pages))

    if skipped:
        print(
            f"\nSkipped {len(skipped)} low-value file(s): "
            f"{skipped}"
        )

    if not all_rows:
        print("\nNo chunks were produced. Nothing to save.")
        return

    dataframe = pd.DataFrame(all_rows)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_parquet(OUT_PATH, index=False)

    print(
        f"\nSaved {len(dataframe)} chunks total -> {OUT_PATH}"
    )

    print("\n--- Chunks per source file ---")
    print(
        dataframe.groupby(
            ["company", "source_file"]
        ).size()
    )

    print("\n--- Sample chunk ---")
    print(dataframe.iloc[0]["chunk_text"][:500])


if __name__ == "__main__":
    main()