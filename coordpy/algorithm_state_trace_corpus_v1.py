"""W136 / COO-9 — dedicated ALGORITHM-STATE-TRACE train/dev/eval/frontier corpus.

W136 tests whether MACHINE-STRUCTURED algorithm state (the dual optimal/naive trajectory + the marked
divergence step + the increment/branching deltas, ``coordpy.algorithm_state_trace_v1``) breaks the
wrong-algorithm capability ceiling that W135's PROSE structure witness could not.  It is tested on the
SAME 16 ``wa_*`` / ``se_*`` non-complexity templates W135 used (8 ``WRONG_ALGORITHM_ADMISSIBLE`` greedy-
vs-DP + 8 ``SEARCH_ENUM`` wrong-recurrence counting), but on FRESH W136 mint seeds (136_0xx),
seed-disjoint from the W135 corpus (135_0xx) — so the eval / frontier slices are a genuinely held-out
field minted for this milestone.

This module is a thin, explicit-import-only wrapper over the W135-validated corpus builder
(``noncomplexity_structure_corpus_v1.build_noncomplexity_corpus_v1``) — it changes ONLY the mint seeds
and the mint date, reusing the W132 per-problem gates, the cross-split content-dedup, the family-balanced
frontier slice, and the deterministic pickle cache verbatim (NO duplication).  The eval split CID and the
frontier 30-slice CID are LOCKED in ``docs/RUNBOOK_W136.md`` §3 before any eval / frontier NIM.

Lane-α SUCCESS = train >= 36, dev >= 36, eval >= 36, frontier >= 30 admitted (the W132-gate-admitted
counts on the fresh seeds).  The ADDITIONAL W136 characterization — how many admitted problems the trace
instrument can build a genuinely-new MACHINE-STRUCTURED state trace on (the naive/ref separation
faithfulness, computed in the self-test) — is reported, not used to gate admission.

See ``docs/RUNBOOK_W136.md`` (LOCKED §3).
"""
from __future__ import annotations

from typing import Optional, Sequence

from .noncomplexity_structure_corpus_v1 import (
    MIN_FRONTIER,
    MIN_PER_SPLIT,
    NonComplexityCorpusV1,
    build_noncomplexity_corpus_v1,
    load_corpus_v1,
    noncomplexity_slate_v1,
    save_corpus_v1,
    select_dev_bench_slice_v1,
    select_frontier_slice_v1,
)
from .resistant_by_construction_battlefield_v1 import DEFAULT_EXEC_TIMEOUT_S

W136_ALGORITHM_STATE_TRACE_CORPUS_V1_SCHEMA_VERSION: str = (
    "coordpy.algorithm_state_trace_corpus_v1.v1")

# ---- LOCKED W136 split seeds (all 20 distinct AND disjoint from the W135 135_0xx seeds ⇒
#      seed-disjoint splits + a genuinely fresh held-out field; docs/RUNBOOK_W136.md §3) ----
TRAIN_SEEDS: tuple[int, ...] = (136_011, 136_012, 136_013, 136_014, 136_015)
DEV_SEEDS: tuple[int, ...] = (136_021, 136_022, 136_023, 136_024, 136_025)
EVAL_SEEDS: tuple[int, ...] = (136_031, 136_032, 136_033, 136_034, 136_035)
FRONTIER_SEEDS: tuple[int, ...] = (136_041, 136_042, 136_043, 136_044, 136_045)
MINTED_DATE: str = "2026-06-04"


def build_algorithm_state_corpus_v1(*, minted_date: str = MINTED_DATE,
                                    train_seeds: Sequence[int] = TRAIN_SEEDS,
                                    dev_seeds: Sequence[int] = DEV_SEEDS,
                                    eval_seeds: Sequence[int] = EVAL_SEEDS,
                                    frontier_seeds: Sequence[int] = FRONTIER_SEEDS,
                                    timeout_s: float = DEFAULT_EXEC_TIMEOUT_S) -> NonComplexityCorpusV1:
    """Mint the fresh-seed W136 algorithm-state-trace corpus (the W135-validated non-complexity builder
    on the W136 136_0xx seeds; seed-disjoint train/dev/eval/frontier, content-deduped, family-balanced
    frontier slice)."""
    return build_noncomplexity_corpus_v1(
        minted_date=minted_date, train_seeds=train_seeds, dev_seeds=dev_seeds,
        eval_seeds=eval_seeds, frontier_seeds=frontier_seeds, timeout_s=timeout_s)


def load_or_build_algorithm_state_corpus_v1(path, *, expect_corpus_cid: Optional[str] = None,
                                            **build_kwargs) -> NonComplexityCorpusV1:
    """Load the cached W136 corpus if present (CID-matching when ``expect_corpus_cid`` is given), else
    mint it with the W136 seeds and cache it."""
    cached = load_corpus_v1(path)
    if cached is not None and (expect_corpus_cid is None
                               or cached.corpus_cid() == expect_corpus_cid):
        return cached
    corpus = build_algorithm_state_corpus_v1(**build_kwargs)
    save_corpus_v1(corpus, path)
    return corpus


__all__ = [
    "W136_ALGORITHM_STATE_TRACE_CORPUS_V1_SCHEMA_VERSION",
    "TRAIN_SEEDS", "DEV_SEEDS", "EVAL_SEEDS", "FRONTIER_SEEDS", "MINTED_DATE",
    "MIN_PER_SPLIT", "MIN_FRONTIER",
    "build_algorithm_state_corpus_v1", "load_or_build_algorithm_state_corpus_v1",
    "select_dev_bench_slice_v1", "select_frontier_slice_v1", "noncomplexity_slate_v1",
]
