# pipeline/chunker.py
# Splits contract text into chunks that respect clause boundaries.
# Legal docs have logical structure (clauses, sections) — splitting mid-clause
# loses context and hurts retrieval. We chunk by section, not by token count.

import re
from pipeline.ingest import extract_text_from_pdf


# Regex to detect clause/section headers like:
# "1.", "2.1", "3.1.2", "SECTION 4", "ARTICLE II", "DEFINITIONS"
CLAUSE_HEADER_PATTERN = re.compile(
    r"^(\d+\.(\d+\.?)*\s+[A-Z]|SECTION\s+\d+|ARTICLE\s+[IVXLC\d]+|[A-Z][A-Z\s]{4,}$)",
    re.MULTILINE
)


def split_into_clauses(text: str) -> list[str]:
    """
    Splits a contract's full text into clause-level chunks.
    Each chunk starts at a detected header and runs until the next one.
    Falls back to paragraph splitting if no headers are found.
    """
    # Find all header positions
    matches = list(CLAUSE_HEADER_PATTERN.finditer(text))

    if len(matches) < 2:
        # Fallback: split by double newline (paragraph-level)
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        return merge_short_chunks(paragraphs)

    chunks = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

    return merge_short_chunks(chunks)


def merge_short_chunks(chunks: list[str], min_chars: int = 100) -> list[str]:
    """
    Merges very short chunks (e.g. a standalone header with no body)
    with the next chunk. Prevents near-empty embeddings.
    """
    merged = []
    buffer = ""

    for chunk in chunks:
        buffer += "\n\n" + chunk if buffer else chunk
        if len(buffer) >= min_chars:
            merged.append(buffer.strip())
            buffer = ""

    if buffer:  # Flush any remaining text
        if merged:
            merged[-1] += "\n\n" + buffer  # Append to last chunk
        else:
            merged.append(buffer)

    return merged


def chunk_pdf(pdf_path: str) -> list[dict]:
    """
    Full pipeline: PDF → extracted text → clause-level chunks.
    Returns a list of dicts ready for embedding:
    { "chunk_id": str, "text": str, "source": str, "page_hint": int }
    """
    pages = extract_text_from_pdf(pdf_path)

    # Combine all pages into one string (contracts flow across pages)
    full_text = "\n\n".join(p["text"] for p in pages)

    clauses = split_into_clauses(full_text)

    chunks = []
    for i, clause_text in enumerate(clauses):
        chunks.append({
            "chunk_id": f"chunk_{i:03d}",
            "text": clause_text,
            "source": pdf_path,
            "char_count": len(clause_text)
        })

    return chunks


if __name__ == "__main__":
    import sys
    import json

    path = sys.argv[1] if len(sys.argv) > 1 else "sample_contracts/sample_nda.pdf"
    chunks = chunk_pdf(path)

    print(f"✅ Created {len(chunks)} chunks\n")
    for c in chunks[:3]:
        print(f"--- {c['chunk_id']} ({c['char_count']} chars) ---")
        print(c["text"][:300])
        print()