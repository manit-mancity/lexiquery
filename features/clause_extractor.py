# features/clause_extractor.py
# On upload, prompts the LLM to identify and summarize key clause types
# present in the contract — gives users an instant "table of contents".

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pipeline.ingest import extract_full_text
import json, re

load_dotenv()

CLAUSE_TYPES = [
    "Parties",
    "Term / Duration",
    "Termination",
    "Confidentiality / NDA",
    "Non-Compete",
    "Non-Solicitation",
    "Governing Law / Jurisdiction",
    "Dispute Resolution / Arbitration",
    "Liability / Indemnification",
    "Intellectual Property",
    "Payment / Compensation",
    "Auto-Renewal",
    "Force Majeure",
    "Amendment / Modification",
]

EXTRACTOR_PROMPT = """You are a legal document analyst. Given the contract text below, identify which of the following clause types are present and provide a one-sentence plain-English summary of each.

Clause types to look for:
{clause_types}

Respond ONLY in valid JSON format like this (include only clauses that are present):
{{
  "clauses": [
    {{"type": "Confidentiality / NDA", "summary": "Parties agree to keep shared information confidential for 2 years after termination."}},
    {{"type": "Termination", "summary": "Either party may terminate with 30 days written notice."}}
  ]
}}

CONTRACT TEXT (first 3000 characters):
{contract_text}
"""


def extract_clauses(pdf_path: str) -> list[dict]:
    """
    Extracts and summarizes key clauses from a contract PDF.
    Returns a list of { "type": str, "summary": str }
    """
    api_key = os.getenv("GROQ_API_KEY")
    full_text = extract_full_text(pdf_path)

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, api_key=api_key, max_tokens=800)
    prompt = ChatPromptTemplate.from_template(EXTRACTOR_PROMPT)
    chain = prompt | llm | StrOutputParser()

    raw = chain.invoke({
        "clause_types": "\n".join(f"- {c}" for c in CLAUSE_TYPES),
        "contract_text": full_text[:3000]
    })

    # Strip markdown fences if present
    raw = re.sub(r"```json|```", "", raw).strip()

    try:
        parsed = json.loads(raw)
        return parsed.get("clauses", [])
    except json.JSONDecodeError:
        return [{"type": "Parse Error", "summary": "Could not extract clauses automatically."}]