"""W135 / COO-9 — dedicated NON-COMPLEXITY (wrong-algorithm + search-enum) train/dev/eval/frontier
corpus for the solution-structure witness.

W134's corpus was ENTIRELY ``COMPLEXITY_BLIND`` (the feedback-fixable timing sub-mode).  W135 targets
the modes W133 showed are CAPABILITY-bound under a bare counterexample (EW1 +0.00 over B0): the
``WRONG_ALGORITHM_ADMISSIBLE`` (greedy-vs-DP traps) and ``SEARCH_ENUM`` (wrong-recurrence counting)
families.  This corpus is ENTIRELY those two modes, so the structure witness is tested exactly where
the counterexample channel was flat — and the ≥2-mode earn condition is meaningful (WA ∧ SE).

Same held-out discipline as W134: 16 distinct algorithm families (8 ``wa_*`` + 8 ``se_*``) × multiple
FRESH seeds; novelty filtering WITHIN a seed only (never across seeds — same-family/different-seed is
the intended held-out diversity); each instance re-tagged ``rbc_<name>__s<seed>``; train / dev / eval
/ frontier draw from DISJOINT seed lists ⇒ seed-disjoint splits with pairwise-disjoint
``content_cid``.  All 16 templates are ``DISC_OUTPUT_MISMATCH`` with an INDEPENDENT exhaustive
``brute_source`` distinct from ``naive_source`` (a genuine oracle the structure witness can mine).

The eval split CID and the frontier 30-slice CID are LOCKED here before any eval / frontier NIM.
Lane-α SUCCESS = train >= 36, dev >= 36, eval >= 36, frontier >= 30 admitted.

Reuses (explicit-import only, NO duplication): ``mint_battlefield_v1`` + per-problem gates +
content-addressed records from ``resistant_by_construction_battlefield_v1``; the 33-template
``RBC_SLATE_V1`` (filtered to the two non-complexity modes).  Pure / deterministic except the
(audited) answer-key subprocess inside ``mint_battlefield_v1``; NO model inference here.  See
``docs/RUNBOOK_W135.md`` (LOCKED §3).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Optional, Sequence

from .resistant_by_construction_battlefield_v1 import (
    DEFAULT_EXEC_TIMEOUT_S,
    MODE_SEARCH_ENUM,
    MODE_WRONG_ALGORITHM,
    MintedProblemV1,
    MintedTemplateV1,
    mint_battlefield_v1,
)
from .resistant_by_construction_slate_v1 import RBC_SLATE_V1

W135_NONCOMPLEXITY_STRUCTURE_CORPUS_V1_SCHEMA_VERSION: str = (
    "coordpy.noncomplexity_structure_corpus_v1.v1")

# ---- LOCKED split seeds (all 20 distinct ⇒ seed-disjoint splits; docs/RUNBOOK_W135.md §3) --
TRAIN_SEEDS: tuple[int, ...] = (135_011, 135_012, 135_013, 135_014, 135_015)
DEV_SEEDS: tuple[int, ...] = (135_021, 135_022, 135_023, 135_024, 135_025)
EVAL_SEEDS: tuple[int, ...] = (135_031, 135_032, 135_033, 135_034, 135_035)
FRONTIER_SEEDS: tuple[int, ...] = (135_041, 135_042, 135_043, 135_044, 135_045)
MINTED_DATE: str = "2026-06-03"
MIN_PER_SPLIT: int = 36           # train / dev / eval floor
MIN_FRONTIER: int = 30            # frontier floor (and the locked frontier-slice size)
SPLIT_NAMES: tuple[str, ...] = ("train", "dev", "eval", "frontier")
NONCOMPLEXITY_MODES: tuple[str, ...] = (MODE_WRONG_ALGORITHM, MODE_SEARCH_ENUM)


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def noncomplexity_slate_v1() -> tuple[MintedTemplateV1, ...]:
    """The non-complexity sub-slate of RBC_SLATE_V1 (the 8 ``wa_*`` + 8 ``se_*`` templates)."""
    return tuple(t for t in RBC_SLATE_V1 if t.mode in NONCOMPLEXITY_MODES)


# ===================================================== one split (multi-seed pool)

@dataclasses.dataclass(frozen=True)
class NonComplexitySplitV1:
    split: str
    seeds: tuple[int, ...]
    problems: tuple[MintedProblemV1, ...]
    n_admitted: int
    family_histogram: dict[str, int]
    mode_histogram: dict[str, int]
    seed_histogram: dict[int, int]
    admitted_problem_ids: tuple[str, ...]
    content_cids: tuple[str, ...]
    split_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {"split": self.split, "seeds": list(self.seeds), "n_admitted": int(self.n_admitted),
                "family_histogram": dict(self.family_histogram),
                "mode_histogram": dict(self.mode_histogram),
                "seed_histogram": {str(k): v for k, v in sorted(self.seed_histogram.items())},
                "admitted_problem_ids": list(self.admitted_problem_ids),
                "split_cid": self.split_cid}


def _retag_seed(problem: MintedProblemV1, seed: int) -> MintedProblemV1:
    """Namespace a minted problem's id by its mint seed so same-family instances never collide.
    ``content_cid`` is name+statement+samples+secret+ref (NOT problem_id), so a seed-disjoint instance
    keeps a DISTINCT content_cid while getting a unique id."""
    return dataclasses.replace(problem, problem_id=f"{problem.problem_id}__s{int(seed)}")


def _finalize_split(split: str, seeds: Sequence[int], pooled: Sequence[MintedProblemV1],
                    seed_hist: dict[int, int]) -> NonComplexitySplitV1:
    """Build the split record (sort + histograms + content CIDs + split_cid) from a problem list.
    Used both at mint time and after cross-split dedup."""
    probs = sorted(pooled, key=lambda p: (p.mode, p.family, p.problem_id))
    fam_hist: dict[str, int] = {}
    mode_hist: dict[str, int] = {}
    for p in probs:
        fam_hist[p.family] = fam_hist.get(p.family, 0) + 1
        mode_hist[p.mode] = mode_hist.get(p.mode, 0) + 1
    content_cids = tuple(p.content_cid() for p in probs)
    split_cid = _sha256_hex({"kind": "w135_noncomplexity_split_v1", "split": str(split),
                             "seeds": [int(s) for s in seeds],
                             "content_cids": sorted(content_cids)})
    return NonComplexitySplitV1(
        split=str(split), seeds=tuple(int(s) for s in seeds), problems=tuple(probs),
        n_admitted=len(probs), family_histogram=dict(sorted(fam_hist.items())),
        mode_histogram=dict(sorted(mode_hist.items())), seed_histogram=dict(seed_hist),
        admitted_problem_ids=tuple(p.problem_id for p in probs), content_cids=content_cids,
        split_cid=split_cid)


def build_noncomplexity_split_v1(slate: Sequence[MintedTemplateV1], *, split: str,
                                 seeds: Sequence[int], minted_date: str,
                                 timeout_s: float = DEFAULT_EXEC_TIMEOUT_S) -> NonComplexitySplitV1:
    """Mint the non-complexity slate once per seed (novelty-filtered WITHIN each seed), re-tag by
    seed, and pool across seeds into one seed-disjoint split."""
    pooled: list[MintedProblemV1] = []
    seed_hist: dict[int, int] = {}
    for seed in seeds:
        bf = mint_battlefield_v1(slate, global_seed=int(seed), minted_date=str(minted_date),
                                 timeout_s=float(timeout_s), official_identities=())
        for p in bf.problems:
            pooled.append(_retag_seed(p, seed))
        seed_hist[int(seed)] = len(bf.problems)
    return _finalize_split(str(split), seeds, pooled, seed_hist)


def dedup_splits_across_v1(splits_in_order: list[NonComplexitySplitV1]) -> list[NonComplexitySplitV1]:
    """Enforce CONTENT-disjointness across splits: keep each ``content_cid`` in the FIRST split it
    appears (priority = list order, i.e. train > dev > eval > frontier), dropping later duplicates.

    A handful of SEARCH_ENUM families (e.g. ``fib_no_adjacent``) have a seed-INDEPENDENT tiny
    case set, so different mint seeds yield byte-identical problems (same content_cid) — seed
    disjointness alone then does not give held-out integrity.  Dropping the few cross-split
    duplicates makes the splits truly content-disjoint while staying far above the 36/30 floors."""
    seen: set[str] = set()
    out: list[NonComplexitySplitV1] = []
    for sp in splits_in_order:
        kept = [p for p in sp.problems if p.content_cid() not in seen]
        seen.update(p.content_cid() for p in kept)
        out.append(_finalize_split(sp.split, sp.seeds, kept, sp.seed_histogram))
    return out


def select_frontier_slice_v1(frontier: NonComplexitySplitV1, *, n: int = MIN_FRONTIER
                             ) -> tuple[MintedProblemV1, ...]:
    """Deterministic family-balanced frontier slice (largest-remainder apportionment over the 16
    families, then seed-tagged-id tie-break), so the frontier run spans all families AND both modes."""
    probs = list(frontier.problems)
    if len(probs) <= n:
        return tuple(probs)
    strata: dict[str, list[MintedProblemV1]] = {}
    for p in probs:
        strata.setdefault(p.family, []).append(p)
    for k in strata:
        strata[k].sort(key=lambda p: p.problem_id)
    counts = {k: len(v) for k, v in strata.items()}
    s = sum(counts.values()) or 1
    raw = {k: (n * v / s) for k, v in counts.items()}
    base = {k: int(x) for k, x in raw.items()}
    rem = int(n) - sum(base.values())
    order = sorted(counts, key=lambda k: (-(raw[k] - base[k]), k))
    for k in order[:max(0, rem)]:
        base[k] += 1
    chosen: list[MintedProblemV1] = []
    for k in sorted(strata):
        chosen.extend(strata[k][:base.get(k, 0)])
    if len(chosen) < n:
        ids = {p.problem_id for p in chosen}
        rest = sorted((p for p in probs if p.problem_id not in ids),
                      key=lambda p: (p.mode, p.family, p.problem_id))
        chosen.extend(rest[:n - len(chosen)])
    chosen = chosen[:n]
    chosen.sort(key=lambda p: (p.mode, p.family, p.problem_id))
    return tuple(chosen)


def select_dev_bench_slice_v1(dev: NonComplexitySplitV1, *, per_family: int = 1
                              ) -> tuple[MintedProblemV1, ...]:
    """Family-stratified DEV bench slice: ``per_family`` instances of each of the 16 families
    (deterministic by seed-tagged id), so the bench spans BOTH modes (WA ∧ SE) and all families at a
    bounded NIM cost."""
    strata: dict[str, list[MintedProblemV1]] = {}
    for p in dev.problems:
        strata.setdefault(p.family, []).append(p)
    chosen: list[MintedProblemV1] = []
    for fam in sorted(strata):
        picks = sorted(strata[fam], key=lambda p: p.problem_id)[:max(1, per_family)]
        chosen.extend(picks)
    chosen.sort(key=lambda p: (p.mode, p.family, p.problem_id))
    return tuple(chosen)


# ===================================================== disjointness audit + full corpus

def corpus_disjointness_report_v1(splits: dict[str, NonComplexitySplitV1]) -> dict[str, Any]:
    cid_sets = {k: set(v.content_cids) for k, v in splits.items()}
    names = list(splits)
    overlaps: dict[str, int] = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            overlaps[f"{a}__{b}"] = len(cid_sets[a] & cid_sets[b])
    seed_sets = {k: set(v.seeds) for k, v in splits.items()}
    seed_overlaps: dict[str, int] = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            seed_overlaps[f"{a}__{b}"] = len(seed_sets[a] & seed_sets[b])
    all_content_disjoint = all(v == 0 for v in overlaps.values())
    all_seed_disjoint = all(v == 0 for v in seed_overlaps.values())
    return {"content_cid_overlaps": overlaps,
            "all_content_cids_pairwise_disjoint": bool(all_content_disjoint),
            "seed_overlaps": seed_overlaps, "all_seeds_pairwise_disjoint": bool(all_seed_disjoint),
            "held_out_integrity": bool(all_content_disjoint and all_seed_disjoint)}


@dataclasses.dataclass(frozen=True)
class NonComplexityCorpusV1:
    schema: str
    minted_date: str
    train: NonComplexitySplitV1
    dev: NonComplexitySplitV1
    eval: NonComplexitySplitV1
    frontier: NonComplexitySplitV1
    frontier_slice_ids: tuple[str, ...]
    frontier_slice_cid: str
    meets_floors: bool
    disjointness: dict[str, Any]

    def corpus_cid(self) -> str:
        return _sha256_hex({"kind": "w135_noncomplexity_corpus_v1", "minted_date": self.minted_date,
                            "train": self.train.split_cid, "dev": self.dev.split_cid,
                            "eval": self.eval.split_cid, "frontier": self.frontier.split_cid})

    def template_by_problem_id(self) -> dict[str, MintedTemplateV1]:
        """Map each admitted (seed-tagged) problem id back to its generator template (harness-side;
        the structure-witness arms need the template to EXECUTE ref/naive/brute — never to render)."""
        by_name = {f"rbc_{t.name}": t for t in noncomplexity_slate_v1()}
        out: dict[str, MintedTemplateV1] = {}
        for split in (self.train, self.dev, self.eval, self.frontier):
            for p in split.problems:
                base = p.problem_id.split("__s")[0]
                if base in by_name:
                    out[p.problem_id] = by_name[base]
        return out

    def mode_by_problem_id(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for split in (self.train, self.dev, self.eval, self.frontier):
            for p in split.problems:
                out[p.problem_id] = p.mode
        return out

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.schema, "minted_date": self.minted_date,
                "corpus_cid": self.corpus_cid(),
                "n_noncomplexity_templates": len(noncomplexity_slate_v1()),
                "min_per_split": MIN_PER_SPLIT, "min_frontier": MIN_FRONTIER,
                "meets_floors": bool(self.meets_floors),
                "n_admitted": {"train": self.train.n_admitted, "dev": self.dev.n_admitted,
                               "eval": self.eval.n_admitted, "frontier": self.frontier.n_admitted},
                "splits": {"train": self.train.to_dict(), "dev": self.dev.to_dict(),
                           "eval": self.eval.to_dict(), "frontier": self.frontier.to_dict()},
                "frontier_slice_cid": self.frontier_slice_cid,
                "frontier_slice_ids": list(self.frontier_slice_ids),
                "disjointness": self.disjointness}


def frontier_slice_cid_v1(slice_problems: Sequence[MintedProblemV1]) -> str:
    return _sha256_hex({"kind": "w135_frontier_slice_v1",
                        "problem_ids": [p.problem_id for p in slice_problems]})


def build_noncomplexity_corpus_v1(*, minted_date: str = MINTED_DATE,
                                  train_seeds: Sequence[int] = TRAIN_SEEDS,
                                  dev_seeds: Sequence[int] = DEV_SEEDS,
                                  eval_seeds: Sequence[int] = EVAL_SEEDS,
                                  frontier_seeds: Sequence[int] = FRONTIER_SEEDS,
                                  timeout_s: float = DEFAULT_EXEC_TIMEOUT_S) -> NonComplexityCorpusV1:
    """Build the dedicated non-complexity corpus (seed-disjoint train/dev/eval/frontier)."""
    slate = noncomplexity_slate_v1()
    train = build_noncomplexity_split_v1(slate, split="train", seeds=train_seeds,
                                         minted_date=minted_date, timeout_s=timeout_s)
    dev = build_noncomplexity_split_v1(slate, split="dev", seeds=dev_seeds,
                                       minted_date=minted_date, timeout_s=timeout_s)
    eval_ = build_noncomplexity_split_v1(slate, split="eval", seeds=eval_seeds,
                                         minted_date=minted_date, timeout_s=timeout_s)
    frontier = build_noncomplexity_split_v1(slate, split="frontier", seeds=frontier_seeds,
                                            minted_date=minted_date, timeout_s=timeout_s)
    # enforce CONTENT-disjointness across splits (a few SEARCH_ENUM families have seed-independent
    # tiny case sets ⇒ byte-identical problems across seeds; keep each content_cid in the first
    # split it appears so dev/eval/frontier are truly held-out). Floors leave wide margin.
    train, dev, eval_, frontier = dedup_splits_across_v1([train, dev, eval_, frontier])
    fslice = select_frontier_slice_v1(frontier, n=MIN_FRONTIER)
    splits = {"train": train, "dev": dev, "eval": eval_, "frontier": frontier}
    meets = bool(train.n_admitted >= MIN_PER_SPLIT and dev.n_admitted >= MIN_PER_SPLIT
                 and eval_.n_admitted >= MIN_PER_SPLIT and frontier.n_admitted >= MIN_FRONTIER
                 and len(fslice) >= MIN_FRONTIER)
    return NonComplexityCorpusV1(
        schema=W135_NONCOMPLEXITY_STRUCTURE_CORPUS_V1_SCHEMA_VERSION, minted_date=str(minted_date),
        train=train, dev=dev, eval=eval_, frontier=frontier,
        frontier_slice_ids=tuple(p.problem_id for p in fslice),
        frontier_slice_cid=frontier_slice_cid_v1(fslice),
        meets_floors=meets, disjointness=corpus_disjointness_report_v1(splits))


# ===================================================== deterministic pickle cache

def save_corpus_v1(corpus: NonComplexityCorpusV1, path) -> str:
    import pickle
    from pathlib import Path as _P
    p = _P(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        pickle.dump(corpus, f)
    return corpus.corpus_cid()


def load_corpus_v1(path) -> Optional[NonComplexityCorpusV1]:
    import pickle
    from pathlib import Path as _P
    p = _P(path)
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


def load_or_build_corpus_v1(path, *, expect_corpus_cid: Optional[str] = None, **build_kwargs
                            ) -> NonComplexityCorpusV1:
    """Load the cached corpus if present (and CID-matching when ``expect_corpus_cid`` is given), else
    mint it and cache it."""
    cached = load_corpus_v1(path)
    if cached is not None and (expect_corpus_cid is None
                               or cached.corpus_cid() == expect_corpus_cid):
        return cached
    corpus = build_noncomplexity_corpus_v1(**build_kwargs)
    save_corpus_v1(corpus, path)
    return corpus


__all__ = [
    "W135_NONCOMPLEXITY_STRUCTURE_CORPUS_V1_SCHEMA_VERSION",
    "TRAIN_SEEDS", "DEV_SEEDS", "EVAL_SEEDS", "FRONTIER_SEEDS", "MINTED_DATE",
    "MIN_PER_SPLIT", "MIN_FRONTIER", "SPLIT_NAMES", "NONCOMPLEXITY_MODES",
    "noncomplexity_slate_v1", "NonComplexitySplitV1", "build_noncomplexity_split_v1",
    "dedup_splits_across_v1", "select_frontier_slice_v1", "select_dev_bench_slice_v1",
    "corpus_disjointness_report_v1", "NonComplexityCorpusV1", "frontier_slice_cid_v1",
    "build_noncomplexity_corpus_v1", "save_corpus_v1", "load_corpus_v1", "load_or_build_corpus_v1",
]
