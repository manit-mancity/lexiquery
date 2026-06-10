# pipeline/retriever.py
# Given a user question, retrieves the most relevant contract clauses
# from ChromaDB using semantic similarity search.

import chromadb
from sentence_transformers import SentenceTransformer
from pipeline.embedder import get_chroma_client, get_embedding_model


def retrieve_relevant_chunks(
    question: str,
    collection_name: str,
    model: SentenceTransformer,
    top_k: int = 4
) -> list[dict]:
    """
    Embeds the user's question and retrieves the top-k most similar
    contract clauses from ChromaDB.

    Returns a list of dicts:
    { "text": str, "chunk_id": str, "score": float }

    Lower distance = more similar (cosine distance, 0 to 2).
    """
    client = get_chroma_client()
    collection = client.get_collection(name=collection_name)

    # Embed the question using the same model as the chunks
    question_embedding = model.encode(question, normalize_embeddings=True).tolist()

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    chunks = []
    for i in range(len(results["documents"][0])):
        chunks.append({
            "chunk_id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "distance": results["distances"][0][i],
            "score": round(1 - results["distances"][0][i] / 2, 3)  # Normalize to 0-1
        })

    # Filter out very low-relevance chunks (score < 0.3)
    chunks = [c for c in chunks if c["score"] >= 0.3]

    return chunks


def format_context(chunks: list[dict]) -> str:
    """
    Formats retrieved chunks into a context block for the LLM prompt.
    Each chunk is clearly labeled with its ID for citation tracking.
    """
    if not chunks:
        return "No relevant clauses found."

    context_parts = []
    for chunk in chunks:
        context_parts.append(
            f"[{chunk['chunk_id']}] (relevance: {chunk['score']})\n{chunk['text']}"
        )

    return "\n\n---\n\n".join(context_parts)


if __name__ == "__main__":
    import sys
    from pipeline.embedder import embed_pdf

    path = sys.argv[1] if len(sys.argv) > 1 else "sample_contracts/sample_nda.pdf"
    question = sys.argv[2] if len(sys.argv) > 2 else "What is the termination clause?"

    model = get_embedding_model()
    collection_name = embed_pdf(path, model)

    print(f"\nQuestion: {question}\n")
    chunks = retrieve_relevant_chunks(question, collection_name, model)

    print(f"Retrieved {len(chunks)} relevant chunks:\n")
    for c in chunks:
        print(f"  [{c['chunk_id']}] score={c['score']}")
        print(f"  {c['text'][:200]}...\n")