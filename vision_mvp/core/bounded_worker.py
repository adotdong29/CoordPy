"""BoundedRetrievalWorker — Phase 19, extended in Phase 20.

A worker that answers a question by:
  1. Searching the ledger for top-k candidate handles.
  2. Fetching the EXACT bytes of the top spans, up to a per-handle byte cap.
  3. Composing a prompt with (question | retrieved spans), staying ≤ B chars.
  4. Calling the LLM once.
  5. Returning the answer + the CIDs whose bytes appeared in the prompt.

The point is to not trust an upstream summary. Every byte the LLM sees is
either (a) the question itself or (b) a verbatim slice of an artifact in
the ledger. The substrate is lossless; the worker's job is to choose
*which* artifacts to materialise.

Phase 20 additions:
  * `search_mode` ∈ {"dense", "lexical", "hybrid"} — selects the retrieval
    backend. Defaults to "dense" so existing Phase-19 callers are
    unaffected.
  * Multi-hop / cross-reference mode (`max_hops > 1`): after the first
    fetch batch, the worker pattern-matches structured cross-references
    (e.g. "see also REL-12345", "OS-2026-0017") in the fetched bytes,
    expands them into follow-up queries, retrieves the referenced
    artifacts, and re-composes the prompt with both batches before the
    single answer LLM call. **Critically: the worker does NOT call the
    LLM to decide the next query.** Hop expansion is a pure pattern
    match over the *exact* fetched bytes; this preserves substrate
    losslessness across hops.
  * `Hop` accounting in `WorkerResult.hops` — one entry per retrieval
    round, with the queries used and the cids returned.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Iterable

from .context_ledger import ContextLedger, Handle


# ---------- Result type ----------------------------------------------------


@dataclass
class Hop:
    """One retrieval round inside a multi-hop worker run."""
    query: str
    candidate_cids: list[str]
    extracted_refs: list[str] = field(default_factory=list)


@dataclass
class WorkerResult:
    """One question's worth of work, returned by `BoundedRetrievalWorker.answer`.

    Fields meant for the caller (humans / downstream agents / tests):
      - answer:           the LLM's reply text
      - cited_cids:       CIDs of the artifacts whose bytes appeared in the
                          prompt. The provenance trail for the answer.
      - prompt_chars:     how many chars the prompt actually occupied
                          (≤ prompt_budget_chars by construction)
      - fetch_count:      number of ledger.fetch calls
      - fetched_bytes:    sum of bytes returned by those fetches
      - exact_input:      True iff every byte in the prompt below the question
                          line came from an exact fetch of a ledger artifact
                          (no summarisation, no paraphrase). Phase 19/20 always
                          produces True; the field exists so future variants
                          (e.g. include LLM summaries) can flag themselves.
      - candidates:       (handle, similarity) pairs that were considered in
                          the FINAL hop. Useful for diagnosing recall.
      - llm_calls:        number of LLM calls (still 1 even in multi-hop:
                          hops do not call the LLM, they pattern-match
                          fetched bytes).
      - hops:             one Hop per retrieval round. Length 1 in single-
                          hop mode; up to max_hops in multi-hop.
      - search_mode:      backend used (dense / lexical / hybrid).
    """

    answer: str
    cited_cids: list[str]
    prompt_chars: int
    fetch_count: int
    fetched_bytes: int
    exact_input: bool
    candidates: list[tuple[Handle, float]] = field(default_factory=list)
    llm_calls: int = 1
    prompt: str = ""           # the full prompt as sent to the LLM (for audit)
    hops: list[Hop] = field(default_factory=list)
    search_mode: str = "dense"


# ---------- Worker ---------------------------------------------------------


_PROMPT_HEADER = (
    "You are answering a question using EXACT excerpts from source artifacts. "
    "Quote relevant facts verbatim. If the excerpts do not contain the "
    "answer, say so explicitly — do NOT invent facts.\n\n"
    "Question: {question}\n\n"
    "Excerpts (each begins with [CID]):"
)


# Cross-reference patterns extracted from fetched bytes for multi-hop
# retrieval. These are the structured anchors we use to expand the
# query in hop ≥ 2. Patterns are corpus-flavoured but generic — any
# ID-style token (uppercase prefix + digits) gets pulled in.
_REF_PATTERNS = [
    re.compile(r"\bOS-\d{4}-\d{4}\b"),     # incident IDs
    re.compile(r"\bREL-\d{3,7}\b"),        # ticket IDs
    re.compile(r"\bSection\s+\d+\b", re.IGNORECASE),
]


def extract_references(text: str) -> list[str]:
    """Find structured cross-references in `text`. Returns a deduped list
    in first-occurrence order.

    The patterns are conservative — they match ID-shaped tokens that have
    a real chance of being a reference to another artifact. Adding new
    patterns is the right path for new corpora; we don't try to learn
    references from prose."""
    seen: set[str] = set()
    out: list[str] = []
    for pat in _REF_PATTERNS:
        for m in pat.finditer(text):
            tok = m.group(0)
            if tok in seen:
                continue
            seen.add(tok)
            out.append(tok)
    return out


@dataclass
class BoundedRetrievalWorker:
    """Bounded-context worker over a ContextLedger.

    Constructor args:
        ledger:                the ContextLedger to retrieve from
        llm_call:              callable(prompt: str) -> str
        prompt_budget_chars:   max chars in the prompt sent to the LLM
        top_k:                 how many handles to consider per search round
        fetch_chars_per_handle: cap on bytes pulled from any one handle
                                (the body still EXISTS exactly in the ledger;
                                 this is just how much we put in the prompt)
        search_mode:           "dense" (default — Phase-19 behaviour),
                               "lexical" (BM25), or "hybrid" (RRF fusion)
        max_hops:              ≥1. >1 enables multi-hop: after the first
                               fetch, the worker pattern-matches structured
                               references in the fetched bytes and re-queries
                               for them. Each hop counts as one search; the
                               LLM is called exactly ONCE total.
        ref_extractor:         callable(text)→list[str] returning structured
                               references to expand. Default = extract_references.

    The worker NEVER calls a summarisation step. The only lossy operation
    is per-handle truncation; surfaced via `WorkerResult.fetched_bytes`.
    """

    ledger: ContextLedger
    llm_call: Callable[[str], str]
    prompt_budget_chars: int = 4000
    top_k: int = 5
    fetch_chars_per_handle: int = 600
    embed_query: bool = True
    search_mode: str = "dense"
    max_hops: int = 1
    ref_extractor: Callable[[str], list[str]] = field(
        default=extract_references)

    def answer(
        self,
        question: str,
        *,
        prefilter: Callable[[Handle, float], bool] | None = None,
    ) -> WorkerResult:
        """Retrieve (single- or multi-hop) and answer in one LLM call.

        `prefilter` lets the caller drop candidate handles before they
        enter the prompt (e.g. enforce a doc_id boundary)."""
        if self.max_hops < 1:
            raise ValueError("max_hops must be ≥ 1")
        if not question.strip():
            return WorkerResult(
                answer="", cited_cids=[], prompt_chars=0,
                fetch_count=0, fetched_bytes=0, exact_input=True,
                prompt="", search_mode=self.search_mode,
            )

        seen_cids: set[str] = set()
        all_excerpts: list[str] = []
        all_cited: list[str] = []
        hops: list[Hop] = []
        fetch_count = 0
        fetched_bytes = 0
        last_candidates: list[tuple[Handle, float]] = []

        header = _PROMPT_HEADER.format(question=question)
        budget = self.prompt_budget_chars - len(header) - 32

        # Hop 0 query is the question itself.
        queries: list[str] = [question]

        for hop_i in range(self.max_hops):
            hop_candidates: list[tuple[Handle, float]] = []
            seen_hop_cids: set[str] = set()
            for q in queries:
                cands = self.ledger.search(
                    query=q, top_k=self.top_k, mode=self.search_mode,
                )
                if prefilter is not None:
                    cands = [(h, s) for (h, s) in cands if prefilter(h, s)]
                for h, s in cands:
                    if h.cid in seen_hop_cids:
                        continue
                    seen_hop_cids.add(h.cid)
                    hop_candidates.append((h, s))
            last_candidates = hop_candidates

            # Materialise + collect refs for the next hop.
            extracted_refs: list[str] = []
            cids_this_hop: list[str] = []
            if budget > 0:
                for handle, _sim in hop_candidates:
                    if handle.cid in seen_cids:
                        continue
                    text = self.ledger.fetch(
                        handle, span=(0, self.fetch_chars_per_handle),
                    )
                    fetch_count += 1
                    fetched_bytes += len(text)
                    block = f"\n\n[{handle.cid[:8]}] {text}"
                    used = sum(len(e) for e in all_excerpts)
                    if used + len(block) > budget:
                        # Out of room. Don't add this excerpt to the
                        # prompt, but DO scan it for references — the
                        # next hop may surface a more compact answer.
                        for ref in self.ref_extractor(text):
                            if ref not in extracted_refs:
                                extracted_refs.append(ref)
                        continue
                    all_excerpts.append(block)
                    all_cited.append(handle.cid)
                    cids_this_hop.append(handle.cid)
                    seen_cids.add(handle.cid)
                    for ref in self.ref_extractor(text):
                        if ref not in extracted_refs:
                            extracted_refs.append(ref)

            hops.append(Hop(
                query=" | ".join(queries),
                candidate_cids=[h.cid for h, _ in hop_candidates],
                extracted_refs=extracted_refs,
            ))

            # Decide whether to take another hop.
            if hop_i + 1 >= self.max_hops:
                break
            # Use new refs as queries; if none survive, stop early.
            new_refs = [r for r in extracted_refs if r not in queries]
            if not new_refs:
                break
            queries = new_refs

        prompt = header + "".join(all_excerpts)
        ans = self.llm_call(prompt)

        return WorkerResult(
            answer=ans,
            cited_cids=all_cited,
            prompt_chars=len(prompt),
            fetch_count=fetch_count,
            fetched_bytes=fetched_bytes,
            exact_input=True,
            candidates=last_candidates,
            llm_calls=1,
            prompt=prompt,
            hops=hops,
            search_mode=self.search_mode,
        )
