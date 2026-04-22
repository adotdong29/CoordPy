"""Lexical (BM25) inverted index — Phase 20.

Companion to `core/retrieval_store.py`'s dense vector index. Solves a
specific failure mode of dense retrieval: rare-token queries.

The Phase-19 needle benchmark surfaced this clearly. A query like
"what ticket tracked the {city} {product} incident" contains a rare
literal — the city name. A semantic embedding maps the city to a vector
that's reasonably close to many other city sections, drowning the
relevant section in cosine-similar noise. A lexical index, by contrast,
treats the city literal as a discrete token whose presence/absence is
binary and unambiguous.

This is **not** a full search engine. It is the smallest BM25
implementation that:
  * tokenises Unicode text on alphanumeric boundaries
  * ignores English stopwords (a tiny set, kept inline so we have no
    dependencies)
  * uses Robertson-Spärck-Jones BM25 with k1=1.5, b=0.75 (the textbook
    defaults)
  * accepts unicode-clean queries — important because the corpora here
    use non-ASCII city names like "São Paulo" and "Reykjavík"

The interface mirrors `BruteForceVectorStore.knn()` so the
`ContextLedger` can fuse rankings from both indices uniformly via
reciprocal-rank fusion.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass


# Tiny English stopword set — keep tokens like "the", "of" out of the
# index where they would dominate the inverted lists without carrying
# discriminative information. Intentionally minimal: anything not on
# this list is indexed.
_STOPWORDS = frozenset({
    "the", "a", "an", "of", "to", "in", "on", "at", "by", "for",
    "with", "and", "or", "is", "was", "were", "be", "been", "being",
    "this", "that", "these", "those", "it", "its", "as", "from",
    "do", "does", "did", "you", "your", "we", "our", "they", "their",
})


_TOKEN_RE = re.compile(r"[A-Za-z\u0080-\uFFFF0-9][A-Za-z\u0080-\uFFFF0-9_\-]*")


def tokenize(text: str) -> list[str]:
    """Lowercased tokens. Keeps hyphens and underscores inside tokens
    (so "OS-2026-0001" stays one token; so does "Cape Town" → ["cape",
    "town"]). Strips Unicode case and accents only at the *case* level
    — accented characters survive ("são", not "sao") so the token still
    matches the literal in the corpus. Stopwords removed."""
    out: list[str] = []
    for m in _TOKEN_RE.finditer(text):
        t = m.group(0).lower()
        if t in _STOPWORDS:
            continue
        out.append(t)
    return out


def _ascii_fold(s: str) -> str:
    """Strip diacritics. Used as a *secondary* token form so a query for
    "Sao Paulo" still matches a corpus entry "São Paulo". The folded
    token is added to the doc *in addition to* the original (not in
    place of), so accented queries also match accented corpus."""
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )


@dataclass
class _Doc:
    cid: str          # foreign key into ContextLedger
    length: int       # number of tokens
    tf: Counter       # token → in-doc count


class LexicalIndex:
    """BM25 over a corpus of (cid, text) pairs.

    Usage parallels BruteForceVectorStore:
        idx = LexicalIndex()
        idx.add(cid, text)
        idx.knn(query_text, k=5)  # → [(cid, score), ...]

    The index is rebuilt incrementally on add(); IDF is recomputed once
    per query (cheap — N additions are O(unique tokens)). For
    write-heavy or very large corpora a deferred rebuild is the obvious
    optimisation; we keep it simple here so the test surface is small.

    Scoring uses the standard Okapi BM25 formula:

        score(D, Q) = sum_{t in Q} IDF(t) *
                      (tf(t,D)*(k1+1)) / (tf(t,D) + k1*(1 - b + b*|D|/avgdl))

    where IDF(t) = ln((N - df(t) + 0.5) / (df(t) + 0.5) + 1).
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = float(k1)
        self.b = float(b)
        self._docs: dict[str, _Doc] = {}            # cid → _Doc
        self._df: Counter = Counter()               # token → doc count
        self._total_length: int = 0
        # Inverted list — token → set of cids — built lazily for
        # candidate generation in knn(). We rebuild from _docs on
        # query if `_dirty`; this avoids per-add list churn.
        self._inv: dict[str, set[str]] = defaultdict(set)
        self._dirty: bool = True

    # ----- writes -----

    def add(self, cid: str, text: str) -> None:
        """Insert / overwrite a document by cid."""
        toks = tokenize(text)
        # Add accent-folded form too, so queries without diacritics still
        # match. We add the folded token *only when it differs* from the
        # original to avoid double-counting.
        folded = []
        for t in toks:
            f = _ascii_fold(t)
            if f != t:
                folded.append(f)
        all_toks = toks + folded

        # If the cid is already indexed, decrement old df entries first.
        if cid in self._docs:
            old = self._docs[cid]
            self._total_length -= old.length
            for tok in old.tf:
                self._df[tok] -= 1
                if self._df[tok] <= 0:
                    del self._df[tok]

        tf = Counter(all_toks)
        self._docs[cid] = _Doc(cid=cid, length=len(all_toks), tf=tf)
        self._total_length += len(all_toks)
        for tok in tf:
            self._df[tok] += 1
        self._dirty = True

    def remove(self, cid: str) -> None:
        """Delete a document. Idempotent — no-op if cid not present."""
        if cid not in self._docs:
            return
        old = self._docs.pop(cid)
        self._total_length -= old.length
        for tok in old.tf:
            self._df[tok] -= 1
            if self._df[tok] <= 0:
                del self._df[tok]
        self._dirty = True

    def __len__(self) -> int:
        return len(self._docs)

    # ----- reads -----

    @property
    def avgdl(self) -> float:
        return (self._total_length / len(self._docs)) if self._docs else 0.0

    def _rebuild_inv(self) -> None:
        if not self._dirty:
            return
        self._inv.clear()
        for cid, doc in self._docs.items():
            for tok in doc.tf:
                self._inv[tok].add(cid)
        self._dirty = False

    def idf(self, token: str) -> float:
        """IDF with +0.5/+0.5 smoothing per Okapi BM25."""
        n = len(self._docs)
        df = self._df.get(token, 0)
        # +1 inside the log so IDF is never negative (standard variant).
        return math.log(((n - df + 0.5) / (df + 0.5)) + 1.0)

    def knn(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        """Top-k cids by BM25, descending score. Empty if no overlap."""
        if k <= 0 or not self._docs:
            return []
        self._rebuild_inv()

        # Build query tokens (drop stopwords, fold accents).
        q_toks_orig = tokenize(query)
        q_folded = [_ascii_fold(t) for t in q_toks_orig if _ascii_fold(t) != t]
        q_toks = list(set(q_toks_orig + q_folded))
        if not q_toks:
            return []

        # Candidate set = union of inverted lists for query tokens.
        candidates: set[str] = set()
        for tok in q_toks:
            candidates |= self._inv.get(tok, set())
        if not candidates:
            return []

        avgdl = self.avgdl or 1.0
        scores: list[tuple[str, float]] = []
        for cid in candidates:
            doc = self._docs[cid]
            s = 0.0
            for tok in q_toks:
                tf = doc.tf.get(tok, 0)
                if tf == 0:
                    continue
                idf = self.idf(tok)
                num = tf * (self.k1 + 1.0)
                denom = tf + self.k1 * (
                    1.0 - self.b + self.b * doc.length / avgdl)
                s += idf * (num / denom)
            scores.append((cid, s))
        scores.sort(key=lambda x: -x[1])
        return scores[:k]


# ---------- Reciprocal Rank Fusion ----------------------------------------


def reciprocal_rank_fusion(
    rankings: list[list[tuple[str, float]]],
    k_rrf: int = 60,
    top_k: int = 10,
) -> list[tuple[str, float]]:
    """Combine multiple rankings via RRF.

    score_RRF(d) = sum_{ranking r} 1 / (k_rrf + rank_r(d))

    Standard k_rrf = 60 (Cormack et al., SIGIR 2009). Robust to score
    scale differences between rankings — only ranks are used.

    Returns the top_k items by aggregated RRF score, descending.
    """
    fused: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, (cid, _score) in enumerate(ranking, start=1):
            fused[cid] += 1.0 / (k_rrf + rank)
    items = sorted(fused.items(), key=lambda x: -x[1])
    return items[:top_k]
