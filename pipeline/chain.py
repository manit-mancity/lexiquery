# pipeline/chain.py
# Connects the retriever to Groq's LLM using a legal-specific prompt.
# Returns a grounded answer with source citations and a "not found" fallback.

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pipeline.retriever import retrieve_relevant_chunks, format_context
from pipeline.embedder import get_embedding_model, embed_pdf
from sentence_transformers import SentenceTransformer

load_dotenv()

# Legal-specific system prompt — instructs the model to:
# 1. Only answer from provided clauses
# 2. Cite the clause ID in its answer
# 3. Explicitly say if the answer isn't in the document
LEGAL_SYSTEM_PROMPT = """You are a legal contract analyst assistant. Your job is to answer questions about contracts clearly and accurately.

STRICT RULES:
1. Answer ONLY using information found in the provided contract clauses below.
2. Always cite the clause ID (e.g., [chunk_003]) where you found the answer.
3. If the answer is NOT present in the provided clauses, respond with exactly:
   "This information is not found in the provided contract. Please consult a lawyer for this question."
4. Do NOT guess, infer, or use outside legal knowledge.
5. Use plain English — avoid jargon where possible, but keep legal terms when precision matters.
6. Keep your answer concise (2-5 sentences unless the question requires more detail).

CONTRACT CLAUSES:
{context}
"""

USER_PROMPT = """Question: {question}

Answer (cite the relevant clause ID):"""


def build_chain() -> tuple:
    """
    Builds and returns the RAG chain components:
    - llm: Groq ChatGroq instance
    - prompt: ChatPromptTemplate
    - parser: StrOutputParser
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY not found. Check your .env file.")

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,           # Zero temp = deterministic, no hallucination
        api_key=api_key,
        max_tokens=512
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", LEGAL_SYSTEM_PROMPT),
        ("human", USER_PROMPT)
    ])

    parser = StrOutputParser()
    return llm, prompt, parser


def answer_question(
    question: str,
    collection_name: str,
    model: SentenceTransformer,
    chat_history: list[dict] = None
) -> dict:
    """
    Full RAG pipeline: question → retrieve → LLM → answer with citations.

    Args:
        question: User's natural language question
        collection_name: ChromaDB collection for this contract
        model: Loaded SentenceTransformer for embedding the question
        chat_history: Optional list of prior Q&A pairs for context

    Returns:
        {
            "answer": str,
            "sources": list[dict],   # Retrieved chunks used
            "found_in_doc": bool     # Whether an answer was found
        }
    """
    # Step 1: Retrieve relevant clauses
    chunks = retrieve_relevant_chunks(question, collection_name, model, top_k=4)
    context = format_context(chunks)

    # Step 2: Build and run the chain
    llm, prompt, parser = build_chain()
    chain = prompt | llm | parser

    raw_answer = chain.invoke({
        "context": context,
        "question": question
    })

    # Step 3: Detect "not found" fallback
    not_found = "not found in the provided contract" in raw_answer.lower()

    return {
        "answer": raw_answer,
        "sources": chunks,
        "found_in_doc": not not_found
    }


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "sample_contracts/sample_nda.pdf"
    question = sys.argv[2] if len(sys.argv) > 2 else "What is the confidentiality period?"

    model = get_embedding_model()
    collection_name = embed_pdf(path, model)

    print(f"\n❓ Question: {question}\n")
    result = answer_question(question, collection_name, model)

    print(f"💬 Answer:\n{result['answer']}\n")
    print(f"📌 Found in document: {result['found_in_doc']}")
    print(f"📎 Sources used: {[c['chunk_id'] for c in result['sources']]}")