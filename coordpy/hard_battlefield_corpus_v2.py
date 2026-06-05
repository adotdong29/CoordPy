"""W137 / COO-9 — parser-neutral hard-battlefield CORPUS assembler (train/dev/eval/frontier).

Mints every :mod:`coordpy.hard_battlefield_slate_v2` template across FOUR seed-disjoint splits so no
instance ever appears in two splits (resistance-by-construction: each instance is freshly minted and
never existed before the mint date — unmemorisable for ANY cutoff, the DyCodeEval/GSM-Symbolic-class
argument that sidesteps the W114 stronger-model-cutoff gate per Lane-γ research arXiv:2503.04149 /
arXiv:2410.05229).

Two $0 quality gates run at assembly time:

* **HC1 parser-neutrality** — every minted case passes
  :func:`coordpy.parser_neutral_io_v1.parser_neutrality_gate_v1` (the W136 confound provably cannot
  recur), AND
* **HC2 exact-oracle discrimination** — every problem passes the W132 framework gates
  (ref-solvable, brute==ref small-case agreement, naive looks-right-on-public / fails-hidden).

* **HC5 template diversity** — distinct templates must have statement char-5-gram Jaccard below the
  W132 novelty ceiling (so the field is not one family repeated).

The remaining two gates are EMPIRICAL and live in :mod:`coordpy.model_ladder_calibration_v1`:
HC3 (strong-anchor A0 headroom) and HC4 (small-model floor).  This module exposes
:func:`admit_by_template` so the calibration result selects the surviving (headroom-band) templates;
the admitted corpus and its split CIDs are then locked before any Lane-β / frontier spend.

Reuses ``mint_problem_v1`` + the novelty Jaccard + CID helpers verbatim.  Pure / deterministic /
explicit-import only; ``coordpy/__init__.py`` untouched.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Optional, Sequence

from .hard_battlefield_slate_v2 import ParserNeutralTemplateV2, build_hard_slate_v2
from .parser_neutral_io_v1 import parser_neutrality_gate_v1
from .resistant_by_construction_battlefield_v1 import (
    MintedProblemV1, NOVELTY_JACCARD_MAX, _sha256_hex, mint_problem_v1, statement_jaccard_v1)
from .icpc_reflexion_bench_v1 import IcpcPilotProblemV1

HARD_CORPUS_V2_SCHEMA_VERSION: str = "coordpy.hard_battlefield_corpus_v2.v1"

# split index -> disjoint global-seed base (instances = f(global_seed, template_name) so disjoint
# seed ranges across splits guarantee whole-instance cross-split disjointness)
SPLIT_NAMES: tuple[str, ...] = ("train", "dev", "eval", "frontier")
_SPLIT_SEED_BASE: dict[str, int] = {
    "train": 137_100_000, "dev": 137_200_000, "eval": 137_300_000, "frontier": 137_400_000}

DEFAULT_MINTED_DATE: str = "2026-06-04"


def split_seeds_v1(split: str, n_replicas: int) -> list[int]:
    """The deterministic, disjoint global-seed list for a split (one per replica)."""
    base = _SPLIT_SEED_BASE[split]
    return [base + r for r in range(int(n_replicas))]


@dataclasses.dataclass(frozen=True)
class CorpusProblemV2:
    split: str
    seed: int
    template_name: str
    family: str
    mode: str
    minted: MintedProblemV1
    hc1_parser_neutral: bool
    gate_admitted: bool

    @property
    def admitted(self) -> bool:
        return bool(self.gate_admitted and self.hc1_parser_neutral)

    def to_pilot(self, *, minted_date: str = DEFAULT_MINTED_DATE) -> IcpcPilotProblemV1:
        # carry the per-instance seed into the id so two replicas are distinct problems
        pilot = self.minted.to_pilot_problem(minted_date=minted_date)
        return dataclasses.replace(
            pilot, problem_id=f"{pilot.problem_id}__{self.split}_{self.seed}")

    def content_cid(self) -> str:
        return _sha256_hex({"k": "w137_corpus_problem_v2", "split": self.split,
                            "seed": int(self.seed), "tpl": self.template_name,
                            "content": self.minted.content_cid()})

    def to_dict(self) -> dict[str, Any]:
        return {"split": self.split, "seed": int(self.seed), "template": self.template_name,
                "family": self.family, "mode": self.mode,
                "hc1_parser_neutral": bool(self.hc1_parser_neutral),
                "gate_admitted": bool(self.gate_admitted), "admitted": bool(self.admitted),
                "reason": self.minted.gates.reason, "content_cid": self.content_cid()}


def mint_split_v2(split: str, *, n_replicas: int, timeout_s: float = 8.0,
                  templates: Optional[Sequence[ParserNeutralTemplateV2]] = None,
                  ) -> list[CorpusProblemV2]:
    """Mint every template at each of the split's disjoint seeds; run HC1 + the framework gates."""
    tpls = list(templates) if templates is not None else build_hard_slate_v2()
    seeds = split_seeds_v1(split, n_replicas)
    out: list[CorpusProblemV2] = []
    for t in tpls:
        for s in seeds:
            p = mint_problem_v1(t.minted, global_seed=s, timeout_s=timeout_s)
            hc1 = parser_neutrality_gate_v1([i for i, _ in p.secret_cases], t.io_shape)
            out.append(CorpusProblemV2(
                split=split, seed=s, template_name=t.minted.name, family=t.minted.family,
                mode=t.minted.mode, minted=p,
                hc1_parser_neutral=bool(hc1.is_parser_neutral),
                gate_admitted=bool(p.gates.admitted)))
    return out


# ===================================================== HC5 template diversity

@dataclasses.dataclass(frozen=True)
class TemplateDiversityRecordV1:
    n_templates: int
    max_pairwise_jaccard: float
    worst_pair: tuple[str, str]
    all_distinct: bool                 # every pairwise Jaccard < NOVELTY_JACCARD_MAX
    n_modes: int

    def to_dict(self) -> dict[str, Any]:
        return {"n_templates": self.n_templates,
                "max_pairwise_jaccard": round(float(self.max_pairwise_jaccard), 4),
                "worst_pair": list(self.worst_pair), "all_distinct": bool(self.all_distinct),
                "n_modes": int(self.n_modes)}


def template_diversity_v1(templates: Optional[Sequence[ParserNeutralTemplateV2]] = None,
                          jaccard_max: float = NOVELTY_JACCARD_MAX) -> TemplateDiversityRecordV1:
    tpls = list(templates) if templates is not None else build_hard_slate_v2()
    worst = 0.0
    pair = ("", "")
    for i in range(len(tpls)):
        for j in range(i + 1, len(tpls)):
            jac = statement_jaccard_v1(tpls[i].minted.statement, tpls[j].minted.statement)
            if jac > worst:
                worst = jac
                pair = (tpls[i].minted.name, tpls[j].minted.name)
    modes = {t.minted.mode for t in tpls}
    return TemplateDiversityRecordV1(
        n_templates=len(tpls), max_pairwise_jaccard=worst, worst_pair=pair,
        all_distinct=bool(worst < float(jaccard_max)), n_modes=len(modes))


# ===================================================== candidate corpus + CIDs

@dataclasses.dataclass(frozen=True)
class SplitRecordV2:
    split: str
    n_minted: int
    n_admitted: int
    admitted_problem_cids: tuple[str, ...]
    mode_histogram: dict[str, int]
    template_histogram: dict[str, int]

    def split_cid(self) -> str:
        return _sha256_hex({"k": "w137_split_v2", "split": self.split,
                            "admitted": list(self.admitted_problem_cids)})

    def to_dict(self) -> dict[str, Any]:
        return {"split": self.split, "n_minted": self.n_minted, "n_admitted": self.n_admitted,
                "mode_histogram": dict(self.mode_histogram),
                "template_histogram": dict(self.template_histogram),
                "split_cid": self.split_cid()}


def summarize_split_v2(problems: Sequence[CorpusProblemV2]) -> SplitRecordV2:
    admitted = [p for p in problems if p.admitted]
    mode_h: dict[str, int] = {}
    tpl_h: dict[str, int] = {}
    for p in admitted:
        mode_h[p.mode] = mode_h.get(p.mode, 0) + 1
        tpl_h[p.template_name] = tpl_h.get(p.template_name, 0) + 1
    split = problems[0].split if problems else ""
    return SplitRecordV2(
        split=split, n_minted=len(problems), n_admitted=len(admitted),
        admitted_problem_cids=tuple(sorted(p.content_cid() for p in admitted)),
        mode_histogram=dict(sorted(mode_h.items())),
        template_histogram=dict(sorted(tpl_h.items())))


def admit_by_template(problems: Sequence[CorpusProblemV2],
                      surviving_template_names: Sequence[str]) -> list[CorpusProblemV2]:
    """Restrict an already-gate-admitted split to the calibration-surviving (headroom-band)
    templates.  This is the HC3/HC4 admission applied AFTER the model-ladder calibration."""
    keep = set(surviving_template_names)
    return [p for p in problems if p.admitted and p.template_name in keep]


def corpus_cid_v2(per_split: dict[str, Sequence[CorpusProblemV2]]) -> str:
    return _sha256_hex({"k": "w137_hard_corpus_v2",
                        "splits": {sp: sorted(p.content_cid() for p in probs if p.admitted)
                                   for sp, probs in sorted(per_split.items())}})


__all__ = [
    "HARD_CORPUS_V2_SCHEMA_VERSION", "SPLIT_NAMES", "DEFAULT_MINTED_DATE", "split_seeds_v1",
    "CorpusProblemV2", "mint_split_v2",
    "TemplateDiversityRecordV1", "template_diversity_v1",
    "SplitRecordV2", "summarize_split_v2", "admit_by_template", "corpus_cid_v2",
]
