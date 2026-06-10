# features/risk_flagger.py
# Scans contract text for patterns commonly associated with unfavorable terms.
# Rule-based first (fast, no LLM cost), then optional LLM enrichment.

import re
from pipeline.ingest import extract_full_text

# Each risk has: a label, description, and regex patterns to search for
RISK_PATTERNS = [
    {
        "label": "🚨 Broad Non-Compete",
        "description": "Non-compete clauses with wide geography or duration may restrict your future employment significantly.",
        "patterns": [
            r"non[\-\s]?compete",
            r"shall not\s.{0,50}compet",
            r"covenant not to compete"
        ]
    },
    {
        "label": "⚠️ Auto-Renewal",
        "description": "Contract may renew automatically unless you cancel in a specific window — easy to miss.",
        "patterns": [
            r"auto[\-\s]?renew",
            r"automatically\s+renew",
            r"unless.{0,60}terminat.{0,30}notice"
        ]
    },
    {
        "label": "⚠️ Unilateral Amendment",
        "description": "One party can change contract terms without the other's consent.",
        "patterns": [
            r"reserves the right to (modify|amend|change)",
            r"may (modify|amend|update) (this agreement|these terms) at any time",
            r"unilateral(ly)?.{0,20}(amend|change|modify)"
        ]
    },
    {
        "label": "🚨 Unlimited Liability",
        "description": "No cap on damages — you could be liable for an uncapped amount.",
        "patterns": [
            r"without\s+limitation.{0,40}liab",
            r"unlimited\s+liabilit",
            r"no\s+cap\s+on\s+damages"
        ]
    },
    {
        "label": "⚠️ Broad IP Assignment",
        "description": "All intellectual property you create may belong to the other party, even outside work hours.",
        "patterns": [
            r"all\s+(intellectual property|inventions|works).{0,60}(assign|belong|vest)",
            r"work\s+made\s+for\s+hire",
            r"any\s+and\s+all\s+inventions"
        ]
    },
    {
        "label": "⚠️ Non-Solicitation",
        "description": "You may be restricted from hiring or working with clients/employees after leaving.",
        "patterns": [
            r"non[\-\s]?solicit",
            r"shall not\s.{0,50}solicit"
        ]
    },
    {
        "label": "⚠️ One-sided Termination",
        "description": "Only one party has the right to terminate at will.",
        "patterns": [
            r"(company|employer)\s+may\s+terminate\s+(at\s+any\s+time|without\s+cause|without\s+notice)",
            r"terminat.{0,30}sole\s+discretion"
        ]
    },
]


def flag_risks(pdf_path: str) -> list[dict]:
    """
    Scans the contract text for risk patterns.
    Returns a list of detected risks with evidence snippets.
    """
    full_text = extract_full_text(pdf_path).lower()
    detected = []

    for risk in RISK_PATTERNS:
        matched_snippets = []
        for pattern in risk["patterns"]:
            for match in re.finditer(pattern, full_text, re.IGNORECASE):
                start = max(0, match.start() - 80)
                end = min(len(full_text), match.end() + 80)
                snippet = "..." + full_text[start:end].replace("\n", " ").strip() + "..."
                matched_snippets.append(snippet)

        if matched_snippets:
            detected.append({
                "label": risk["label"],
                "description": risk["description"],
                "snippets": matched_snippets[:2]  # Max 2 snippets per risk
            })

    return detected


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "sample_contracts/sample_nda.pdf"
    risks = flag_risks(path)

    if not risks:
        print("✅ No common risk patterns detected.")
    else:
        print(f"⚠️ {len(risks)} potential risks found:\n")
        for r in risks:
            print(f"{r['label']}")
            print(f"  {r['description']}")
            for s in r["snippets"]:
                print(f"  → \"{s}\"")
            print()