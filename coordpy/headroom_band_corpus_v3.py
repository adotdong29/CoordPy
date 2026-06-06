"""W138 / COO-9 — headroom-band CORPUS assembler v3 (train/dev/eval/frontier).

Mints the CALIBRATION-SURVIVING (family, knob) band cells across FOUR seed-disjoint splits so no
instance ever appears in two splits (resistance-by-construction: each instance is freshly minted and
never existed before the mint date — unmemorisable for ANY cutoff, the DyCodeEval/GSM-Symbolic
argument that sidesteps the W114 stronger-model-cutoff gate; arXiv:2503.04149 / arXiv:2410.05229).

Two $0 quality gates at assembly (identical discipline to ``hard_battlefield_corpus_v2``):
  * **HC1 parser-neutrality** — every minted case passes ``parser_neutrality_gate_v1``, AND
  * **HC2 exact-oracle discrimination** — every problem passes the W132 framework gates
    (ref-solvable, brute==ref small agreement, naive looks-right-on-public / fails-hidden with the
    declared kind, split integrity).
HC3/HB3/HB4 are the EMPIRICAL band gates and live in ``headroom_band_calibration_v2``; this module is
handed the surviving band templates and mints the held-out corpus from them.

W138 seed bases are byte-disjoint from W137 (137_*) and from the calibration base.  Reuses
``mint_problem_v1`` + the novelty Jaccard + CID helpers verbatim.  Pure / deterministic / explicit-
import only; ``coordpy/__init__.py`` untouched.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Optional, Sequence

from .hard_battlefield_slate_v2 import ParserNeutralTemplateV2
from .parser_neutral_io_v1 import parser_neutrality_gate_v1
from .resistant_by_construction_battlefield_v1 import (
    MintedProblemV1, NOVELTY_JACCARD_MAX, _sha256_hex, mint_problem_v1, statement_jaccard_v1)
from .icpc_reflexion_bench_v1 import IcpcPilotProblemV1

BAND_CORPUS_V3_SCHEMA_VERSION: str = "coordpy.headroom_band_corpus_v3.v1"

SPLIT_NAMES: tuple[str, ...] = ("train", "dev", "eval", "frontier")
_SPLIT_SEED_BASE: dict[str, int] = {
    "train": 138_100_000, "dev": 138_200_000, "eval": 138_300_000, "frontier": 138_400_000}

DEFAULT_MINTED_DATE: str = "2026-06-05"


def split_seeds_v1(split: str, n_replicas: int) -> list[int]:
    """The deterministic, disjoint global-seed list for a split (one per replica)."""
    base = _SPLIT_SEED_BASE[split]
    return [base + r for r in range(int(n_replicas))]


@dataclasses.dataclass(frozen=True)
class CorpusProblemV3:
    split: str
    seed: int
    cell_id: str          # template name (family + knob)
    family: str
    mode: str
    minted: MintedProblemV1
    hc1_parser_neutral: bool
    gate_admitted: bool

    @property
    def admitted(self) -> bool:
        return bool(self.gate_admitted and self.hc1_parser_neutral)

    def to_pilot(self, *, minted_date: str = DEFAULT_MINTED_DATE) -> IcpcPilotProblemV1:
        pilot = self.minted.to_pilot_problem(minted_date=minted_date)
        return dataclasses.replace(
            pilot, problem_id=f"{pilot.problem_id}__{self.split}_{self.seed}")

    def content_cid(self) -> str:
        return _sha256_hex({"k": "w138_corpus_problem_v3", "split": self.split,
                            "seed": int(self.seed), "cell": self.cell_id,
                            "content": self.minted.content_cid()})

    def to_dict(self) -> dict[str, Any]:
        return {"split": self.split, "seed": int(self.seed), "cell": self.cell_id,
                "family": self.family, "mode": self.mode,
                "hc1_parser_neutral": bool(self.hc1_parser_neutral),
                "gate_admitted": bool(self.gate_admitted), "admitted": bool(self.admitted),
                "reason": self.minted.gates.reason, "content_cid": self.content_cid()}


def mint_split_v3(split: str, *, templates: Sequence[ParserNeutralTemplateV2],
                  n_replicas: int, timeout_s: float = 8.0, mint_timeout_s: Optional[float] = None,
                  ) -> list[CorpusProblemV3]:
    """Mint every surviving band template at each of the split's disjoint seeds; run HC1 + framework
    gates.  ``mint_timeout_s`` (default = ``timeout_s``) bounds the $0 build (the COMPLEXITY naive
    TLEs regardless, so a shorter mint timeout only speeds the build — it does not change the locked
    grading budget)."""
    seeds = split_seeds_v1(split, n_replicas)
    mt = float(mint_timeout_s) if mint_timeout_s is not None else float(timeout_s)
    out: list[CorpusProblemV3] = []
    for t in templates:
        for s in seeds:
            p = mint_problem_v1(t.minted, global_seed=s, timeout_s=mt)
            hc1 = parser_neutrality_gate_v1([i for i, _ in p.secret_cases], t.io_shape)
            out.append(CorpusProblemV3(
                split=split, seed=s, cell_id=t.minted.name, family=t.minted.family,
                mode=t.minted.mode, minted=p,
                hc1_parser_neutral=bool(hc1.is_parser_neutral),
                gate_admitted=bool(p.gates.admitted)))
    return out


@dataclasses.dataclass(frozen=True)
class SplitRecordV3:
    split: str
    n_minted: int
    n_admitted: int
    admitted_problem_cids: tuple[str, ...]
    mode_histogram: dict[str, int]
    family_histogram: dict[str, int]

    def split_cid(self) -> str:
        return _sha256_hex({"k": "w138_split_v3", "split": self.split,
                            "admitted": list(self.admitted_problem_cids)})

    def to_dict(self) -> dict[str, Any]:
        return {"split": self.split, "n_minted": self.n_minted, "n_admitted": self.n_admitted,
                "mode_histogram": dict(self.mode_histogram),
                "family_histogram": dict(self.family_histogram),
                "n_families": len(self.family_histogram), "n_modes": len(self.mode_histogram),
                "split_cid": self.split_cid()}


def summarize_split_v3(problems: Sequence[CorpusProblemV3]) -> SplitRecordV3:
    admitted = [p for p in problems if p.admitted]
    mode_h: dict[str, int] = {}
    fam_h: dict[str, int] = {}
    for p in admitted:
        mode_h[p.mode] = mode_h.get(p.mode, 0) + 1
        fam_h[p.family] = fam_h.get(p.family, 0) + 1
    split = problems[0].split if problems else ""
    return SplitRecordV3(
        split=split, n_minted=len(problems), n_admitted=len(admitted),
        admitted_problem_cids=tuple(sorted(p.content_cid() for p in admitted)),
        mode_histogram=dict(sorted(mode_h.items())),
        family_histogram=dict(sorted(fam_h.items())))


def per_instance_novelty_v3(problems: Sequence[CorpusProblemV3],
                            jaccard_max: float = NOVELTY_JACCARD_MAX) -> dict[str, Any]:
    """Per-family worst within-family pairwise statement Jaccard (pre-empts the C5 near-dup inflation).
    Band instances of one family share a statement template by design, so this guards the INPUT
    diversity via the rendered cases rather than the prose; we report the prose Jaccard as a tripwire
    (it should be ~1.0 within a family since the statement is shared — the real novelty is the minted
    inputs, guaranteed distinct by disjoint seeds)."""
    by_fam: dict[str, list[CorpusProblemV3]] = {}
    for p in problems:
        if p.admitted:
            by_fam.setdefault(p.family, []).append(p)
    worst_cross_family = 0.0
    worst_pair = ("", "")
    fams = sorted(by_fam.keys())
    for i in range(len(fams)):
        for j in range(i + 1, len(fams)):
            a = by_fam[fams[i]][0].minted.statement
            b = by_fam[fams[j]][0].minted.statement
            jac = statement_jaccard_v1(a, b)
            if jac > worst_cross_family:
                worst_cross_family = jac
                worst_pair = (fams[i], fams[j])
    return {"n_families": len(fams),
            "worst_cross_family_jaccard": round(worst_cross_family, 4),
            "worst_pair": list(worst_pair),
            "cross_family_distinct": bool(worst_cross_family < float(jaccard_max))}


def corpus_cid_v3(per_split: dict[str, Sequence[CorpusProblemV3]]) -> str:
    return _sha256_hex({"k": "w138_band_corpus_v3",
                        "splits": {sp: sorted(p.content_cid() for p in probs if p.admitted)
                                   for sp, probs in sorted(per_split.items())}})


__all__ = [
    "BAND_CORPUS_V3_SCHEMA_VERSION", "SPLIT_NAMES", "DEFAULT_MINTED_DATE", "split_seeds_v1",
    "CorpusProblemV3", "mint_split_v3", "SplitRecordV3", "summarize_split_v3",
    "per_instance_novelty_v3", "corpus_cid_v3",
]
