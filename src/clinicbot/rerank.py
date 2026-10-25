"""Reranking with a cross-encoder.

The embedding model is a *bi-encoder*: it embeds the question and each chunk
separately, then compares the two vectors. That's fast (chunk vectors are
computed once, at ingest time) but crude, because the model never sees the
question and the chunk together.

A *cross-encoder* reads "question + chunk" as one input and outputs a single
relevance score. It's much more accurate, but far too slow to run on every
chunk in a big collection. So the standard pattern is two-stage:

    cheap retrieval (vector/BM25/hybrid) -> top ~20 candidates
    expensive reranker                  -> best k of those 20

Model: ms-marco-MiniLM-L-6-v2 (~80 MB, runs on CPU), trained on Bing search
queries to judge whether a passage answers a query.
"""

import os
from functools import lru_cache

# Windows without Developer Mode can't create symlinks; the cache still works.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from fastembed.rerank.cross_encoder import TextCrossEncoder  # noqa: E402

from clinicbot.config import PROJECT_ROOT  # noqa: E402
from clinicbot.store import SearchResult  # noqa: E402

RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=1)
def _model() -> TextCrossEncoder:
    # Cache inside the project (default is the temp folder, which Windows may clean).
    return TextCrossEncoder(
        model_name=RERANK_MODEL, cache_dir=str(PROJECT_ROOT / "data" / "models")
    )


def rerank(query: str, candidates: list[SearchResult], k: int) -> list[SearchResult]:
    if not candidates:
        return []
    scores = list(_model().rerank(query, [c.text for c in candidates]))
    ranked = sorted(zip(scores, candidates), key=lambda pair: pair[0], reverse=True)
    return [
        SearchResult(c.id, c.text, c.source, c.section, float(s)) for s, c in ranked[:k]
    ]
