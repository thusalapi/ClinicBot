"""One entry point for every retrieval strategy, so they're easy to compare.

    vector  semantic search only (Phase 1)
    bm25    keyword search only
    hybrid  both, merged with Reciprocal Rank Fusion
    rerank  hybrid candidates, re-ordered by a cross-encoder
"""

from functools import lru_cache

from clinicbot import store
from clinicbot.bm25 import BM25Index
from clinicbot.config import CANDIDATES, RRF_K

MODES = ("vector", "bm25", "hybrid", "rerank")


@lru_cache(maxsize=1)
def _bm25() -> BM25Index:
    return BM25Index(store.all_chunks())


def reciprocal_rank_fusion(
    ranked_lists: list[list[store.SearchResult]], k: int
) -> list[store.SearchResult]:
    """Merge rankings from different searches.

    BM25 scores (0–15) and cosine similarities (0–1) aren't on the same scale,
    so we can't simply add them. RRF ignores the scores and only uses each
    chunk's *rank* in each list:

        rrf(chunk) = sum over lists of 1 / (RRF_K + rank)

    A chunk ranked high by both searches wins. A chunk found by only one search
    still gets credit. RRF_K (usually 60) dampens the gap between rank 1 and rank 2.
    """
    fused: dict[str, float] = {}
    by_id: dict[str, store.SearchResult] = {}
    for results in ranked_lists:
        for rank, r in enumerate(results, start=1):
            fused[r.id] = fused.get(r.id, 0.0) + 1 / (RRF_K + rank)
            by_id[r.id] = r
    best = sorted(fused, key=fused.get, reverse=True)[:k]
    return [
        store.SearchResult(i, by_id[i].text, by_id[i].source, by_id[i].section, fused[i])
        for i in best
    ]


def retrieve(query: str, k: int, mode: str) -> list[store.SearchResult]:
    if mode == "vector":
        return store.search(query, k)
    if mode == "bm25":
        return _bm25().search(query, k)

    candidates = reciprocal_rank_fusion(
        [store.search(query, CANDIDATES), _bm25().search(query, CANDIDATES)],
        k=CANDIDATES,
    )
    if mode == "hybrid":
        return candidates[:k]
    if mode == "rerank":
        from clinicbot.rerank import rerank  # loads an extra model, so import lazily

        return rerank(query, candidates, k)
    raise ValueError(f"unknown retrieval mode {mode!r}; pick one of {MODES}")
