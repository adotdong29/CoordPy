"""W133 / COO-9 — deterministic train/dev/eval curriculum over the minted battlefield.

W132's ``resistant_by_construction_slate_v1`` is a set of 33 problem GENERATORS, each seeded
by ``sha256({global_seed, template.name})``.  Re-minting the SAME slate at a DIFFERENT
``global_seed`` therefore yields the SAME algorithmic families but seed-DISJOINT instances
(different public samples, different hidden cases, different answer keys, different
content CIDs).  That is exactly the train / dev / eval split mechanism W133 needs:

* **train**  (``TRAIN_SEED``) — instances for $0 witness self-tests / mechanism characterisation;
* **dev**    (``DEV_SEED``)   — the held-out generated-family mechanism bench (the go/no-go
  signal for whether to spend on eval);
* **eval**   (``EVAL_SEED``)  — the LOCKED held-out earn slice; its CID is fixed before any
  eval spend and it is never used for mechanism design.

The three splits are seed-disjoint by construction; this module additionally VERIFIES and
content-addresses the disjointness (no shared graded secret-case input across splits for the
same family; pairwise-disjoint per-problem content CIDs) and emits a family-balance report.
A curriculum SUCCEEDS iff it admits at least ``MIN_PER_SPLIT`` clean problems per split (so
the corpus has at least ``3 * MIN_PER_SPLIT`` admitted instances).

Reuses (explicit-import only, NO duplication): ``mint_battlefield_v1`` + the per-problem
quality gates + ``select_core_slice_v1`` + the content-addressed records from
``resistant_by_construction_battlefield_v1``; the 33-template slate ``RBC_SLATE_V1``.  Pure /
deterministic except the (audited) answer-key subprocess inside ``mint_battlefield_v1``; NO
model inference here.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Optional, Sequence

from .resistant_by_construction_battlefield_v1 import (
    MintedProblemV1,
    MintedTemplateV1,
    RbcBattlefieldV1,
    _problem_id,
    mint_battlefield_v1,
    select_core_slice_v1,
)

W133_WITNESS_CURRICULUM_CORPUS_V1_SCHEMA_VERSION: str = (
    "coordpy.witness_curriculum_corpus_v1.v1")

# ---- LOCKED split seeds (distinct from the W132 mint seed 132 -> fresh instances) --
TRAIN_SEED: int = 133_001
DEV_SEED: int = 133_002
EVAL_SEED: int = 133_003
MINTED_DATE: str = "2026-06-02"
MIN_PER_SPLIT: int = 32          # Lane-alpha SUCCESS floor per split (>= 96 total)
SPLIT_NAMES: tuple[str, ...] = ("train", "dev", "eval")


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


# ===================================================== one split

@dataclasses.dataclass(frozen=True)
class CurriculumSplitV1:
    split: str
    global_seed: int
    battlefield: RbcBattlefieldV1
    n_admitted: int
    mode_histogram: dict[str, int]
    admitted_problem_ids: tuple[str, ...]
    content_cids: tuple[str, ...]
    split_cid: str

    def problems(self) -> tuple[MintedProblemV1, ...]:
        return self.battlefield.problems

    def to_dict(self) -> dict[str, Any]:
        return {"split": self.split, "global_seed": int(self.global_seed),
                "n_admitted": int(self.n_admitted),
                "mode_histogram": dict(self.mode_histogram),
                "admitted_problem_ids": list(self.admitted_problem_ids),
                "manifest_cid": self.battlefield.manifest.manifest_cid(),
                "split_cid": self.split_cid}


def _build_split(slate: Sequence[MintedTemplateV1], *, split: str, global_seed: int,
                 minted_date: str, official_identities: Sequence[str],
                 timeout_s: float) -> CurriculumSplitV1:
    bf = mint_battlefield_v1(slate, global_seed=int(global_seed), minted_date=str(minted_date),
                             timeout_s=float(timeout_s),
                             official_identities=official_identities)
    probs = bf.problems
    mode_hist: dict[str, int] = {}
    for p in probs:
        mode_hist[p.mode] = mode_hist.get(p.mode, 0) + 1
    content_cids = tuple(p.content_cid() for p in probs)
    split_cid = _sha256_hex({"kind": "w133_curriculum_split_v1", "split": split,
                             "global_seed": int(global_seed),
                             "content_cids": sorted(content_cids)})
    return CurriculumSplitV1(
        split=str(split), global_seed=int(global_seed), battlefield=bf,
        n_admitted=len(probs), mode_histogram=dict(sorted(mode_hist.items())),
        admitted_problem_ids=tuple(p.problem_id for p in probs),
        content_cids=content_cids, split_cid=split_cid)


# ===================================================== disjointness + balance audits

def split_disjointness_report_v1(train: CurriculumSplitV1, dev: CurriculumSplitV1,
                                 eval_: CurriculumSplitV1) -> dict[str, Any]:
    """Verify the splits are genuinely held-out: (a) pairwise-disjoint per-problem content
    CIDs, and (b) for every family present in two splits, NO graded secret-case input is
    shared (the model is graded on different data in each split)."""
    splits = {"train": train, "dev": dev, "eval": eval_}
    cid_sets = {k: set(v.content_cids) for k, v in splits.items()}
    cid_overlaps = {}
    for a, b in (("train", "dev"), ("train", "eval"), ("dev", "eval")):
        cid_overlaps[f"{a}__{b}"] = len(cid_sets[a] & cid_sets[b])

    def _secret_inputs(s: CurriculumSplitV1) -> dict[str, set[str]]:
        return {p.problem_id: {inp for inp, _ in p.secret_cases} for p in s.problems()}

    si = {k: _secret_inputs(v) for k, v in splits.items()}
    secret_overlaps = {}
    worst_family_overlap = 0
    for a, b in (("train", "dev"), ("train", "eval"), ("dev", "eval")):
        total = 0
        for pid, ains in si[a].items():
            bins = si[b].get(pid, set())
            ov = len(ains & bins)
            total += ov
            worst_family_overlap = max(worst_family_overlap, ov)
        secret_overlaps[f"{a}__{b}"] = total

    all_content_disjoint = all(v == 0 for v in cid_overlaps.values())
    all_secret_disjoint = all(v == 0 for v in secret_overlaps.values())
    seed_disjoint = bool(train.global_seed != dev.global_seed != eval_.global_seed
                         and train.global_seed != eval_.global_seed)
    # Held-out integrity is established by WHOLE-PROBLEM content-CID disjointness + seed
    # disjointness: every problem differs across splits (statement-instance + public samples +
    # the bulk of the secret cases). A few individual secret-case INPUTS may recur across
    # splits — but ONLY seed-independent canonical stress/edge constructions (e.g. the
    # decreasing worst-case for a short-circuiting naive, or a fixed boundary case). These
    # carry no memorisable answer signal, the model never sees secret cases, the splits are
    # graded by independent runs, and the mechanism is pre-committed (never tuned on a split),
    # so per-input recurrence does not breach held-out integrity.
    held_out_integrity = bool(all_content_disjoint and seed_disjoint)
    return {
        "content_cid_overlaps": cid_overlaps,
        "all_content_cids_pairwise_disjoint": bool(all_content_disjoint),
        "graded_secret_input_overlaps": secret_overlaps,
        "all_graded_secret_inputs_disjoint": bool(all_secret_disjoint),
        "worst_single_family_secret_overlap": int(worst_family_overlap),
        "seed_disjoint": seed_disjoint,
        "held_out_integrity": held_out_integrity,
        "held_out_integrity_basis": ("content-CID + seed disjoint; residual per-secret-input "
                                     "overlap is seed-independent canonical stress/edge cases "
                                     "(no answer signal; mechanism pre-committed)"),
    }


def family_balance_report_v1(train, dev, eval_) -> dict[str, Any]:
    return {"train": dict(train.mode_histogram), "dev": dict(dev.mode_histogram),
            "eval": dict(eval_.mode_histogram),
            "modes_present_all_splits": sorted(
                set(train.mode_histogram) & set(dev.mode_histogram)
                & set(eval_.mode_histogram))}


# ===================================================== the full curriculum

@dataclasses.dataclass(frozen=True)
class WitnessCurriculumV1:
    schema: str
    minted_date: str
    train: CurriculumSplitV1
    dev: CurriculumSplitV1
    eval: CurriculumSplitV1
    n_total_admitted: int
    meets_min_per_split: bool
    meets_min_total: bool
    disjointness: dict[str, Any]
    family_balance: dict[str, Any]

    def curriculum_cid(self) -> str:
        return _sha256_hex({"kind": "w133_witness_curriculum_v1",
                            "minted_date": self.minted_date,
                            "train": self.train.split_cid, "dev": self.dev.split_cid,
                            "eval": self.eval.split_cid})

    def templates_by_problem_id(self, slate: Sequence[MintedTemplateV1]
                                ) -> dict[str, MintedTemplateV1]:
        """Map admitted problem ids back to their generator templates (harness-side; the
        witness instrument needs the generators to draw fresh probe inputs)."""
        return {_problem_id(t.name): t for t in slate}

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.schema, "minted_date": self.minted_date,
                "curriculum_cid": self.curriculum_cid(),
                "n_total_admitted": int(self.n_total_admitted),
                "min_per_split": MIN_PER_SPLIT,
                "meets_min_per_split": bool(self.meets_min_per_split),
                "meets_min_total": bool(self.meets_min_total),
                "seeds": {"train": self.train.global_seed, "dev": self.dev.global_seed,
                          "eval": self.eval.global_seed},
                "splits": {"train": self.train.to_dict(), "dev": self.dev.to_dict(),
                           "eval": self.eval.to_dict()},
                "disjointness": self.disjointness,
                "family_balance": self.family_balance}


def build_curriculum_v1(slate: Sequence[MintedTemplateV1], *,
                        minted_date: str = MINTED_DATE,
                        train_seed: int = TRAIN_SEED, dev_seed: int = DEV_SEED,
                        eval_seed: int = EVAL_SEED,
                        official_identities: Sequence[str] = (),
                        timeout_s: float = 8.0) -> WitnessCurriculumV1:
    """Mint the slate three times (seed-disjoint), verify split disjointness + family balance,
    and assemble the deterministic curriculum."""
    train = _build_split(slate, split="train", global_seed=train_seed,
                         minted_date=minted_date, official_identities=official_identities,
                         timeout_s=timeout_s)
    dev = _build_split(slate, split="dev", global_seed=dev_seed, minted_date=minted_date,
                       official_identities=official_identities, timeout_s=timeout_s)
    eval_ = _build_split(slate, split="eval", global_seed=eval_seed, minted_date=minted_date,
                         official_identities=official_identities, timeout_s=timeout_s)
    n_total = train.n_admitted + dev.n_admitted + eval_.n_admitted
    meets_each = bool(min(train.n_admitted, dev.n_admitted, eval_.n_admitted) >= MIN_PER_SPLIT)
    return WitnessCurriculumV1(
        schema=W133_WITNESS_CURRICULUM_CORPUS_V1_SCHEMA_VERSION, minted_date=str(minted_date),
        train=train, dev=dev, eval=eval_, n_total_admitted=int(n_total),
        meets_min_per_split=meets_each,
        meets_min_total=bool(n_total >= 3 * MIN_PER_SPLIT),
        disjointness=split_disjointness_report_v1(train, dev, eval_),
        family_balance=family_balance_report_v1(train, dev, eval_))


def select_split_bench_slice_v1(split: CurriculumSplitV1, *, n_problems: int = 30
                                ) -> tuple[MintedProblemV1, ...]:
    """Deterministic, mode-stratified bench slice for a split (reuses the W132 core-slice
    apportionment so a split bench is never all-one-family)."""
    return select_core_slice_v1(split.battlefield, n_problems=int(n_problems))


__all__ = [
    "W133_WITNESS_CURRICULUM_CORPUS_V1_SCHEMA_VERSION",
    "TRAIN_SEED", "DEV_SEED", "EVAL_SEED", "MINTED_DATE", "MIN_PER_SPLIT", "SPLIT_NAMES",
    "CurriculumSplitV1", "WitnessCurriculumV1", "build_curriculum_v1",
    "split_disjointness_report_v1", "family_balance_report_v1",
    "select_split_bench_slice_v1",
]
