"""Lossless context substrate — Phase 19.

The substrate that replaces summarise-then-pool with store-and-fetch.

Three pieces, layered:

1. **Exact byte store** — `core/merkle_dag.MerkleDAG`, content-addressed
   so the same artifact always gets the same CID. Used for the artifact
   bodies; nothing in the body is ever modified after `put`.
2. **Index** — embedding-based retrieval over `core/retrieval_store.
   BruteForceVectorStore`. Used to *find* artifacts without loading bodies.
3. **Handles** — fixed-size references that travel through the agent
   layer. They include the CID, an optional span, the embedding, and a
   short fingerprint preview so a recipient can see "what is this" without
   fetching the body.

The ledger ties the three together and adds provenance: every artifact
records the CIDs of the artifacts it was derived from, forming a DAG that
can be unwound to the original sources.

This module deliberately does NOT do compression, summarisation, or
truncation. Bodies that go in come out byte-equal. The only lossy operation
is *retrieval ranking* — top-k can miss the relevant artifact. When that
happens, the artifact still exists in the store; the worker just didn't
ask for it. That keeps the failure modes separable: index recall vs LLM
accuracy vs (zero) substrate loss.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import numpy as np

from .lexical_index import LexicalIndex, reciprocal_rank_fusion
from .merkle_dag import MerkleDAG, content_hash
from .retrieval_store import BruteForceVectorStore


# ---------- Exceptions ---------------------------------------------------


class LedgerCapacityError(Exception):
    """Raised when a `put` would exceed `max_artifacts` or
    `max_artifact_chars`. The ledger is in-memory and append-only;
    these guardrails exist so a runaway producer can't exhaust RAM
    silently."""


# ---------- Handle: the unit of exchange ---------------------------------


@dataclass(frozen=True)
class Handle:
    """A reference to an artifact in the ledger.

    Carries enough metadata for the recipient to decide whether to fetch:
        - cid: content hash of the FULL artifact body
        - span: optional (start, end) char range within the body; None means
                "the whole body"
        - embedding: dense embedding of the body (or of the span if span set)
                     — fixed dim, used for index search and similarity checks
        - fingerprint: short text preview, for human/LLM-readable routing
                       decisions before fetch. Default ≤ 80 chars.
        - metadata: free-form dict (doc_id, section_idx, kind, etc.).

    The handle's *prompt cost* is dominated by the fingerprint; the cid,
    span, and metadata together are O(100) bytes. The embedding does NOT
    enter the prompt — it stays in the ledger's index.
    """

    cid: str
    span: tuple[int, int] | None
    fingerprint: str
    metadata: tuple = ()    # tuple of (key, value) pairs for hashability
    # The embedding is intentionally NOT a field — it lives in the index,
    # not in the handle. Carrying it would inflate prompts.

    def metadata_dict(self) -> dict:
        return dict(self.metadata)

    def __repr__(self) -> str:
        m = self.metadata_dict()
        tag = m.get("doc_id") or m.get("kind") or m.get("section") or ""
        span_str = f"[{self.span[0]}:{self.span[1]}]" if self.span else ""
        return f"<Handle {self.cid[:8]}{span_str} {tag!r} {self.fingerprint[:40]!r}>"


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return v
    return v / n


def _fingerprint(text: str, max_chars: int = 80) -> str:
    """Short, deterministic preview of a body. First non-blank line, capped."""
    if not text:
        return ""
    # Take the first non-empty line, strip, cap.
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line[:max_chars]
    return text.strip()[:max_chars]


# ---------- ContextLedger ------------------------------------------------


@dataclass
class LedgerStats:
    n_put: int = 0
    n_search: int = 0
    n_search_dense: int = 0
    n_search_lexical: int = 0
    n_search_hybrid: int = 0
    n_fetch: int = 0
    bytes_fetched: int = 0
    bytes_stored: int = 0


@dataclass
class _Entry:
    """Internal record. Bodies live HERE (not in the handle)."""
    cid: str
    body: str
    embedding: np.ndarray
    parent_cids: tuple[str, ...]
    metadata: dict


class ContextLedger:
    """Exact artifact store + retrieval index + provenance.

    Use as a singleton per task. Multiple agents share one ledger.
    """

    def __init__(
        self,
        embed_dim: int,
        embed_fn: Callable[[str], np.ndarray],
        fingerprint_chars: int = 80,
        max_artifacts: int | None = None,
        max_artifact_chars: int | None = None,
    ):
        """Construct an empty ledger.

        Args:
            embed_dim: dim of the dense embeddings the ledger will store.
            embed_fn: text → embedding (called on `put` if no embedding given).
            fingerprint_chars: max chars in the preview line of a Handle.
            max_artifacts: optional safety bound on `len(self)`. Raises
                `LedgerCapacityError` on `put` once this many entries
                exist. Default None = unbounded.
            max_artifact_chars: optional bound on a single artifact's
                body length. Raises `LedgerCapacityError` if exceeded.
                Default None = unbounded.

        The two `max_*` knobs are operational guardrails: a Phase-19
        ledger is in-memory and append-only, and a runaway producer
        could exhaust RAM. The defaults are unbounded so existing
        experiments are unchanged; opt in by passing values."""
        if embed_dim < 1:
            raise ValueError("embed_dim must be ≥ 1")
        if max_artifacts is not None and max_artifacts < 1:
            raise ValueError("max_artifacts must be ≥ 1 if set")
        if max_artifact_chars is not None and max_artifact_chars < 1:
            raise ValueError("max_artifact_chars must be ≥ 1 if set")
        self.embed_dim = embed_dim
        self.embed_fn = embed_fn
        self.fingerprint_chars = fingerprint_chars
        self.max_artifacts = max_artifacts
        self.max_artifact_chars = max_artifact_chars

        self._dag = MerkleDAG()
        self._index = BruteForceVectorStore(dim=embed_dim)
        self._lexical = LexicalIndex()
        # cid → _Entry. Keeps O(1) lookups, separate from MerkleDAG (which
        # stores raw objects keyed by THEIR canonical encoding hash; we
        # store strings so the two hashes coincide).
        self._entries: dict[str, _Entry] = {}
        # cid → row index in the vector store, for span-aware fetching.
        self._cid_to_idx: dict[str, int] = {}
        # cid → list[child cid] — built lazily for lineage queries.
        self._children: dict[str, list[str]] = {}
        self.stats = LedgerStats()

    # ---------- Inserts ------------------------------------------------

    def put(
        self,
        body: str,
        *,
        embedding: np.ndarray | None = None,
        parent_cids: Iterable[str] = (),
        metadata: dict | None = None,
    ) -> Handle:
        """Insert an artifact and return a handle. Idempotent on body+metadata.

        If `embedding` is None, embed_fn(body) is called.
        Parents must already be in the ledger; an unknown parent CID raises.
        Bounded-size guardrails (set in __init__) are enforced here.
        """
        if not isinstance(body, str):
            raise TypeError("body must be a str")
        if self.max_artifact_chars is not None and len(body) > self.max_artifact_chars:
            raise LedgerCapacityError(
                f"artifact size {len(body)} > max_artifact_chars="
                f"{self.max_artifact_chars}")
        meta = dict(metadata) if metadata else {}

        # Idempotency: same (body, metadata) → same CID. We hash the
        # canonical encoding of (body, metadata) so adding the same chunk
        # twice with the same metadata returns the same handle.
        # Note: parent_cids is intentionally NOT in the CID — provenance
        # is annotation, not identity. Two derivations of the same content
        # collapse to one entry.
        cid = content_hash({"body": body, "meta": meta})

        if cid in self._entries:
            entry = self._entries[cid]
            return Handle(
                cid=cid,
                span=None,
                fingerprint=_fingerprint(body, self.fingerprint_chars),
                metadata=tuple(sorted(entry.metadata.items())),
            )

        if (self.max_artifacts is not None
                and len(self._entries) >= self.max_artifacts):
            raise LedgerCapacityError(
                f"ledger is full: {len(self._entries)} ≥ "
                f"max_artifacts={self.max_artifacts}")

        if embedding is None:
            embedding = self.embed_fn(body)
        emb = np.asarray(embedding, dtype=float).ravel()
        if emb.size != self.embed_dim:
            raise ValueError(
                f"embedding dim {emb.size} != ledger dim {self.embed_dim}")
        emb = _normalize(emb)

        for p in parent_cids:
            if p not in self._entries:
                raise KeyError(f"unknown parent_cid {p}")
            self._children.setdefault(p, []).append(cid)

        entry = _Entry(
            cid=cid, body=body, embedding=emb,
            parent_cids=tuple(parent_cids), metadata=meta,
        )
        self._entries[cid] = entry
        # Mirror in the Merkle DAG for inclusion proofs / external auditing.
        self._dag.put({"body": body, "meta": meta})
        idx = self._index.add(emb, payload=cid, metadata={"cid": cid})
        self._cid_to_idx[cid] = idx
        # Mirror into the lexical (BM25) index for hybrid retrieval.
        self._lexical.add(cid, body)

        self.stats.n_put += 1
        self.stats.bytes_stored += len(body)

        return Handle(
            cid=cid,
            span=None,
            fingerprint=_fingerprint(body, self.fingerprint_chars),
            metadata=tuple(sorted(meta.items())),
        )

    # ---------- Search -------------------------------------------------

    def _handle_for(self, cid: str) -> Handle:
        entry = self._entries[cid]
        return Handle(
            cid=cid,
            span=None,
            fingerprint=_fingerprint(entry.body, self.fingerprint_chars),
            metadata=tuple(sorted(entry.metadata.items())),
        )

    def search(
        self,
        query: str | np.ndarray,
        top_k: int = 5,
        embedding: np.ndarray | None = None,
        mode: str = "dense",
        rrf_k: int = 60,
    ) -> list[tuple[Handle, float]]:
        """Top-k handles. Bodies are NOT loaded. Three modes:

          * `mode="dense"`   — pure vector cosine (Phase-19 default;
                               unchanged behaviour).
          * `mode="lexical"` — pure BM25 over the inverted index.
                               Wins for rare-token queries (incident
                               IDs, ticket numbers, exact city names)
                               where dense embeddings smear meaning.
          * `mode="hybrid"`  — reciprocal-rank fusion of dense and
                               lexical rankings. Matches whichever
                               retriever wins on a given query, by
                               construction.

        Returns list of (handle, score) sorted by score descending.
        Score units differ across modes (cosine vs BM25 vs RRF) so
        compare ranks across modes, not raw values."""
        self.stats.n_search += 1

        if mode == "dense":
            self.stats.n_search_dense += 1
            return self._search_dense(query, top_k=top_k, embedding=embedding)
        if mode == "lexical":
            self.stats.n_search_lexical += 1
            return self._search_lexical(query, top_k=top_k)
        if mode == "hybrid":
            self.stats.n_search_hybrid += 1
            return self._search_hybrid(
                query, top_k=top_k, embedding=embedding, rrf_k=rrf_k)
        raise ValueError(
            f"unknown search mode {mode!r}; "
            "use 'dense' | 'lexical' | 'hybrid'")

    def _search_dense(
        self,
        query: str | np.ndarray,
        top_k: int,
        embedding: np.ndarray | None,
    ) -> list[tuple[Handle, float]]:
        if embedding is None:
            if isinstance(query, str):
                embedding = self.embed_fn(query)
            else:
                embedding = query
        q = np.asarray(embedding, dtype=float).ravel()
        if q.size != self.embed_dim:
            raise ValueError(f"query dim {q.size} != ledger dim {self.embed_dim}")
        q = _normalize(q)
        if len(self._index) == 0:
            return []
        hits = self._index.knn(q, k=top_k)
        return [(self._handle_for(record.payload), float(sim))
                for _idx, sim, record in hits]

    def _search_lexical(
        self,
        query: str,
        top_k: int,
    ) -> list[tuple[Handle, float]]:
        if not isinstance(query, str):
            raise TypeError("lexical search requires a string query")
        hits = self._lexical.knn(query, k=top_k)
        return [(self._handle_for(cid), float(score)) for cid, score in hits]

    def _search_hybrid(
        self,
        query: str | np.ndarray,
        top_k: int,
        embedding: np.ndarray | None,
        rrf_k: int,
    ) -> list[tuple[Handle, float]]:
        # Pull a wider candidate pool from each retriever before fusing.
        # 4× the requested top_k is the standard hybrid-search trick:
        # a relevant doc can be deeply ranked by one retriever but high
        # by the other; widening recall before fusion catches it.
        pool = max(top_k * 4, top_k)
        dense_hits = self._search_dense(query, top_k=pool, embedding=embedding)
        if isinstance(query, str):
            lex_hits = self._search_lexical(query, top_k=pool)
        else:
            lex_hits = []
        # Convert to (cid, score) pairs for RRF.
        d_rank = [(h.cid, s) for h, s in dense_hits]
        l_rank = [(h.cid, s) for h, s in lex_hits]
        fused = reciprocal_rank_fusion(
            [d_rank, l_rank], k_rrf=rrf_k, top_k=top_k)
        return [(self._handle_for(cid), float(score)) for cid, score in fused]

    # ---------- Fetch --------------------------------------------------

    def fetch(self, handle: Handle, span: tuple[int, int] | None = None) -> str:
        """Return the EXACT bytes of the artifact (or the requested span).

        Span override: if `span` is given, it overrides any span in the handle.
        Counts toward `bytes_fetched` accounting.

        Raises ValueError if the handle's fingerprint disagrees with the
        actual stored body — defends against handle tampering / cross-ledger
        confusion. Raises KeyError on unknown CID.
        """
        if handle.cid not in self._entries:
            raise KeyError(f"unknown cid {handle.cid}")
        body = self._entries[handle.cid].body
        # Validate the handle: its fingerprint must match what we'd
        # produce for the stored body. This catches handles forged in
        # one ledger and presented to another, and handles whose body
        # was mutated externally (which can't happen here but is the
        # invariant a real on-disk store would have to defend).
        if handle.fingerprint:
            expected_fp = _fingerprint(body, self.fingerprint_chars)
            if handle.fingerprint != expected_fp:
                raise ValueError(
                    f"handle fingerprint mismatch for cid {handle.cid[:8]}…: "
                    f"got {handle.fingerprint!r}, expected {expected_fp!r}")
        s = span if span is not None else handle.span
        if s is None:
            text = body
        else:
            start, end = int(s[0]), int(s[1])
            text = body[start:end]
        self.stats.n_fetch += 1
        self.stats.bytes_fetched += len(text)
        return text

    def verify_handle(self, handle: Handle) -> bool:
        """True iff the handle is well-formed and corresponds to a known
        artifact whose stored body still produces the handle's fingerprint.
        Does not count as a fetch."""
        if handle.cid not in self._entries:
            return False
        body = self._entries[handle.cid].body
        if handle.fingerprint and handle.fingerprint != _fingerprint(
                body, self.fingerprint_chars):
            return False
        return True

    def get_body(self, cid: str) -> str:
        """Direct CID → body lookup, no accounting (for internal/test use)."""
        return self._entries[cid].body

    # ---------- Provenance ---------------------------------------------

    def parents(self, handle_or_cid: Handle | str) -> list[Handle]:
        cid = handle_or_cid.cid if isinstance(handle_or_cid, Handle) else handle_or_cid
        if cid not in self._entries:
            raise KeyError(cid)
        out = []
        for p in self._entries[cid].parent_cids:
            entry = self._entries[p]
            out.append(Handle(
                cid=p, span=None,
                fingerprint=_fingerprint(entry.body, self.fingerprint_chars),
                metadata=tuple(sorted(entry.metadata.items())),
            ))
        return out

    def children(self, handle_or_cid: Handle | str) -> list[Handle]:
        cid = handle_or_cid.cid if isinstance(handle_or_cid, Handle) else handle_or_cid
        out = []
        for c in self._children.get(cid, []):
            entry = self._entries[c]
            out.append(Handle(
                cid=c, span=None,
                fingerprint=_fingerprint(entry.body, self.fingerprint_chars),
                metadata=tuple(sorted(entry.metadata.items())),
            ))
        return out

    def lineage(self, handle: Handle) -> list[Handle]:
        """Walk parents transitively, returning ancestors with no duplicates.

        Order: BFS from the handle, parents-before-grandparents. Useful for
        auditing "what derived this answer".
        """
        if handle.cid not in self._entries:
            raise KeyError(handle.cid)
        seen = {handle.cid}
        frontier = [handle]
        out: list[Handle] = []
        while frontier:
            nxt: list[Handle] = []
            for h in frontier:
                for p in self.parents(h):
                    if p.cid in seen:
                        continue
                    seen.add(p.cid)
                    out.append(p)
                    nxt.append(p)
            frontier = nxt
        return out

    # ---------- Inclusion proofs ---------------------------------------

    def merkle_root(self) -> tuple[str, list[list[str]]]:
        """Build a Merkle tree of all entries (in CID-sorted order) and return
        (root_hash, levels). Used for external audits of the artifact set."""
        cids = sorted(self._entries.keys())
        leaves = [{"body": self._entries[c].body,
                   "meta": self._entries[c].metadata}
                  for c in cids]
        return self._dag.build_merkle_tree(leaves)

    # ---------- Convenience --------------------------------------------

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, item: Handle | str) -> bool:
        cid = item.cid if isinstance(item, Handle) else item
        return cid in self._entries

    def all_handles(self) -> list[Handle]:
        return [
            Handle(
                cid=cid, span=None,
                fingerprint=_fingerprint(e.body, self.fingerprint_chars),
                metadata=tuple(sorted(e.metadata.items())),
            )
            for cid, e in self._entries.items()
        ]

    def stats_dict(self) -> dict:
        return {
            "n_artifacts": len(self._entries),
            "bytes_stored": self.stats.bytes_stored,
            "n_put": self.stats.n_put,
            "n_search": self.stats.n_search,
            "n_search_dense": self.stats.n_search_dense,
            "n_search_lexical": self.stats.n_search_lexical,
            "n_search_hybrid": self.stats.n_search_hybrid,
            "n_fetch": self.stats.n_fetch,
            "bytes_fetched": self.stats.bytes_fetched,
            "max_artifacts": self.max_artifacts,
            "max_artifact_chars": self.max_artifact_chars,
        }


# ---------- Helpers -------------------------------------------------------


def hash_embedding(text: str, dim: int = 64) -> np.ndarray:
    """Deterministic dummy embedding for tests / no-LLM smoke runs.

    Hashes 3-grams into a sparse vector then L2-normalises. This is NOT a
    semantic embedding — it's a stable function from text to ℝᵈ, which is
    enough to exercise the index plumbing without an LLM."""
    if dim < 1:
        raise ValueError("dim must be ≥ 1")
    vec = np.zeros(dim, dtype=float)
    text = text.lower()
    if len(text) == 0:
        return vec
    # 3-grams
    for i in range(max(0, len(text) - 2)):
        tri = text[i:i + 3]
        h = hashlib.sha1(tri.encode("utf-8")).digest()
        bucket = int.from_bytes(h[:4], "big") % dim
        sign = 1.0 if (h[4] & 1) else -1.0
        vec[bucket] += sign
    n = float(np.linalg.norm(vec))
    return vec / n if n > 1e-12 else vec
