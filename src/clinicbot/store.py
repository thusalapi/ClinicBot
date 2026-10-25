"""Step 3 of RAG: store vectors in a vector database and search them.

A vector DB stores (id, vector, text, metadata) rows and answers one question
fast: "which stored vectors are closest to this query vector?". Chroma uses an
HNSW index (an approximate nearest-neighbour graph) so it doesn't have to compare
against every row. We use *cosine distance*: 0 = same direction (very similar),
larger = less similar.
"""

from dataclasses import dataclass

import chromadb

from clinicbot.chunking import Chunk
from clinicbot.config import COLLECTION_NAME, VECTOR_DB_DIR
from clinicbot.embeddings import embed


@dataclass
class SearchResult:
    id: str
    text: str
    source: str
    section: str
    # Higher = more relevant. What the number means depends on who produced it
    # (cosine similarity, BM25 score, fusion score, reranker score), so only
    # compare scores from the same method.
    score: float


def _collection():
    client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
    return client.get_or_create_collection(
        COLLECTION_NAME,
        configuration={"hnsw": {"space": "cosine"}},
    )


def reset_collection() -> None:
    client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
    if COLLECTION_NAME in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION_NAME)


def add_chunks(chunks: list[Chunk]) -> None:
    # We compute embeddings ourselves (instead of letting Chroma do it silently)
    # so you can see this step happen.
    _collection().upsert(
        ids=[c.id for c in chunks],
        documents=[c.text for c in chunks],
        metadatas=[c.metadata for c in chunks],
        embeddings=embed([c.text for c in chunks]),
    )


def search(query: str, k: int) -> list[SearchResult]:
    result = _collection().query(query_embeddings=embed([query]), n_results=k)
    return [
        SearchResult(
            id=chunk_id,
            text=doc,
            source=meta["source"],
            section=meta["section"],
            score=1 - dist,  # cosine distance -> cosine similarity
        )
        for chunk_id, doc, meta, dist in zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        )
    ]


def all_chunks() -> list[SearchResult]:
    """Every stored chunk (used to build the BM25 keyword index)."""
    result = _collection().get(include=["documents", "metadatas"])
    return [
        SearchResult(
            id=chunk_id, text=doc, source=meta["source"], section=meta["section"], score=0.0
        )
        for chunk_id, doc, meta in zip(
            result["ids"], result["documents"], result["metadatas"]
        )
    ]


def count() -> int:
    return _collection().count()
