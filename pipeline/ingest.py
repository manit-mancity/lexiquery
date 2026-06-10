# pipeline/ingest.py
# Extracts text from a PDF contract while preserving structure (section headers, clauses)

import pdfplumber
import re


def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """
    Extract text from a PDF page by page.
    Returns a list of dicts: { "page": int, "text": str }
    Preserves whitespace formatting to help detect headers and clause numbers.
    """
    pages = []

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text(x_tolerance=2, y_tolerance=2)
            if text and text.strip():
                pages.append({
                    "page": i + 1,
                    "text": text.strip()
                })

    if not pages:
        raise ValueError("No text could be extracted from this PDF. It may be scanned/image-based.")

    return pages


def extract_full_text(pdf_path: str) -> str:
    """
    Returns the entire contract as a single string.
    Used for clause extraction and risk flagging.
    """
    pages = extract_text_from_pdf(pdf_path)
    return "\n\n".join(p["text"] for p in pages)


def detect_structure(pages: list[dict]) -> list[dict]:
    """
    Tags each page's lines as 'header', 'clause', or 'body'.
    Headers: ALL CAPS lines or lines like "1.", "2.1", "SECTION 3"
    This metadata helps the chunker split on logical boundaries.
    """
    header_pattern = re.compile(
        r"^(\d+\.(\d+\.?)*|SECTION\s+\d+|ARTICLE\s+\d+|[A-Z\s]{5,})$"
    )

    structured_pages = []
    for page in pages:
        lines = page["text"].split("\n")
        tagged_lines = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if header_pattern.match(stripped):
                tagged_lines.append({"type": "header", "text": stripped})
            else:
                tagged_lines.append({"type": "body", "text": stripped})

        structured_pages.append({
            "page": page["page"],
            "lines": tagged_lines
        })

    return structured_pages


if __name__ == "__main__":
    # Quick test — replace with your contract path
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "sample_contracts/sample_nda.pdf"

    pages = extract_text_from_pdf(path)
    print(f"✅ Extracted {len(pages)} pages\n")
    print("--- First page preview ---")
    print(pages[0]["text"][:500])