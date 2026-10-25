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
    text: str
    source: str
    section: str
    distance: float


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
            text=doc,
            source=meta["source"],
            section=meta["section"],
            distance=dist,
        )
        for doc, meta, dist in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0]
        )
    ]


def count() -> int:
    return _collection().count()
