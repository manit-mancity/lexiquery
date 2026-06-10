# pipeline/embedder.py
# Embeds contract chunks using a legal-domain sentence transformer
# and stores them in a local ChromaDB vector store.

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from pipeline.chunker import chunk_pdf
import hashlib
import os

# Using nlpaueb/legal-bert-base-uncased — trained on legal text, 
# significantly better than generic models for contract retrieval
EMBEDDING_MODEL = "nlpaueb/legal-bert-base-uncased"
CHROMA_DIR = "./chroma_store"


def get_embedding_model() -> SentenceTransformer:
    """Load the legal-BERT embedding model (downloads on first use ~400MB)."""
    print("Loading legal-BERT embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("✅ Model loaded")
    return model


def get_chroma_client() -> chromadb.Client:
    """Returns a persistent ChromaDB client stored locally."""
    return chromadb.PersistentClient(path=CHROMA_DIR)


def get_or_create_collection(client: chromadb.Client, collection_name: str):
    """Gets existing collection or creates a new one."""
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}  # Cosine similarity for semantic search
    )


def pdf_to_collection_name(pdf_path: str) -> str:
    """
    Generates a stable collection name from the PDF path.
    ChromaDB collection names must be alphanumeric + hyphens, 3-63 chars.
    """
    hash_str = hashlib.md5(pdf_path.encode()).hexdigest()[:16]
    base_name = os.path.basename(pdf_path).replace(".pdf", "").replace(" ", "-").lower()
    # Truncate and sanitize
    base_name = "".join(c if c.isalnum() or c == "-" else "" for c in base_name)[:30]
    return f"{base_name}-{hash_str}"


def embed_pdf(pdf_path: str, model: SentenceTransformer = None) -> str:
    """
    Full pipeline: PDF → chunks → embeddings → ChromaDB.
    Returns the collection name (used later for retrieval).
    Skips re-embedding if this PDF was already processed.
    """
    if model is None:
        model = get_embedding_model()

    client = get_chroma_client()
    collection_name = pdf_to_collection_name(pdf_path)
    collection = get_or_create_collection(client, collection_name)

    # Skip if already embedded
    if collection.count() > 0:
        print(f"✅ Found existing embeddings for this contract ({collection.count()} chunks)")
        return collection_name

    print(f"Chunking contract: {pdf_path}")
    chunks = chunk_pdf(pdf_path)
    print(f"  → {len(chunks)} chunks created")

    texts = [c["text"] for c in chunks]
    ids = [c["chunk_id"] for c in chunks]
    metadatas = [{"source": c["source"], "char_count": c["char_count"]} for c in chunks]

    print("Embedding chunks with legal-BERT...")
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    collection.add(
        documents=texts,
        embeddings=embeddings.tolist(),
        ids=ids,
        metadatas=metadatas
    )

    print(f"✅ Embedded and stored {len(chunks)} chunks in ChromaDB")
    return collection_name


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "sample_contracts/sample_nda.pdf"
    model = get_embedding_model()
    collection_name = embed_pdf(path, model)
    print(f"\nCollection name: {collection_name}")