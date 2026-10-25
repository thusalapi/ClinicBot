"""Keyword search with BM25, written from scratch so you can see how it works.

Vector search matches *meaning* but is weak at exact tokens: prices ("500"),
codes ("HbA1c"), names ("LankaQR"). BM25 is the classic search-engine formula
that scores a chunk by the query words it contains:

    score(chunk) = sum over query words w of
                   IDF(w) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len / avg_len))

  tf      how often w appears in the chunk (more = better, with diminishing returns)
  IDF(w)  how *rare* w is across all chunks. "the" appears everywhere so it's
          worth ~0; "warfarin" appears once, so a match on it is worth a lot
  k1      how fast repeated words stop adding score (typical 1.2–2.0)
  b       how much to penalise long chunks, which contain more words by chance (0–1)
"""

import math
import re
from collections import Counter

from clinicbot.store import SearchResult

TOKEN_RE = re.compile(r"[a-z0-9]+")
# Very common words carry no signal; dropping them makes scores cleaner.
STOPWORDS = set(
    "a an and are as at be by can do does for from how i if in is it me my of on "
    "or the to what when where which who will with you your".split()
)


def tokenize(text: str) -> list[str]:
    # "Rs. 1,000" -> ["rs", "1", "000"]. Digits are kept on purpose.
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


class BM25Index:
    def __init__(self, chunks: list[SearchResult], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.term_freqs = [Counter(tokenize(c.text)) for c in chunks]
        self.lengths = [sum(tf.values()) for tf in self.term_freqs]
        self.avg_len = sum(self.lengths) / len(self.lengths)

        doc_freq = Counter(term for tf in self.term_freqs for term in tf)
        n = len(chunks)
        self.idf = {
            term: math.log(1 + (n - df + 0.5) / (df + 0.5)) for term, df in doc_freq.items()
        }

    def score(self, query: str, i: int) -> float:
        tf, length = self.term_freqs[i], self.lengths[i]
        total = 0.0
        for term in tokenize(query):
            if term not in tf:
                continue
            freq = tf[term]
            norm = self.k1 * (1 - self.b + self.b * length / self.avg_len)
            total += self.idf[term] * freq * (self.k1 + 1) / (freq + norm)
        return total

    def search(self, query: str, k: int) -> list[SearchResult]:
        scored = [(self.score(query, i), c) for i, c in enumerate(self.chunks)]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [
            SearchResult(c.id, c.text, c.source, c.section, s)
            for s, c in scored[:k]
            if s > 0
        ]
