"""Step 2 of RAG: turn text into vectors (embeddings).

An embedding model maps a piece of text to a list of numbers (here 384 of them)
such that texts with similar *meaning* end up close together in that space.
"How much is a check-up?" and "Dental check-up: Rs. 2,500" share few words but
land near each other, which is why vector search beats plain keyword search
for many questions.

We use all-MiniLM-L6-v2, a small free model that runs locally on your CPU
(Chroma downloads it, ~80 MB, the first time). The important rule:
**documents and questions must be embedded with the same model**, otherwise
their vectors live in different spaces and distances are meaningless.
"""

from functools import lru_cache

from chromadb.utils.embedding_functions import DefaultEmbeddingFunction


@lru_cache(maxsize=1)
def _model() -> DefaultEmbeddingFunction:
    return DefaultEmbeddingFunction()


def embed(texts: list[str]) -> list[list[float]]:
    return [list(map(float, vector)) for vector in _model()(texts)]
