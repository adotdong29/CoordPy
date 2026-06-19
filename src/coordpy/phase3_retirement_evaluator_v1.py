"""W105 / COO-9 — Phase 3 retirement evaluator V1.

Reads a set of per-(model_class, seed) Phase 3 cell bench
reports + their provenance JSONs and emits a structured
per-class + cross-class retirement verdict.

The evaluator is a pure-Python module with no NIM dependency
and no model loading.  It is explicit-import only — it does NOT
register itself in ``coordpy/__init__.py``.

Inputs (per cell):

* The bench report dict (matches
  ``coordpy.humaneval_plus_reflexion_bench_v1.HumanEvalPlusBenchReportV1.to_dict``
  shape, with the W103+ pilot driver's added ``provenance``,
  ``mlb``, ``phase2_evaluation`` and ``wall_s`` envelope
  fields).
* The provenance dict (matches the W103+ pilot driver's
  ``provenance.json`` shape).

Inputs (per class set):

* Multiple cells, one per seed in the W105 retirement seed
  list ``(105 001, 105 002, 105 003)``.
* All cells share the same slice pack CID.

Per-class verdict labels (locked in W105 RUNBOOK):

* ``RETIRED`` — all 6 retirement bars PASS for that class.
* ``RETIRED_MARGIN_DRIVEN_NON_LOAD_BEARING`` — bars 1-6 all
  PASS BUT MLB-2 rescue rate < 33 % on that class (the W104
  lesson: margin can clear while mechanism load-bearingness
  erodes).
* ``FAIL_<reason>`` — at least one bar FAILs.

Cross-class entitlement (locked):

* Cross-class retirement claim entitled IFF:
  (i) both per-class verdicts are ``RETIRED`` (NOT
      ``RETIRED_MARGIN_DRIVEN_NON_LOAD_BEARING``);
  AND
  (ii) cross-class mean ``B − A1`` difference within ± 5 pp.

Refuses to run if (any of):

1. Cell count for a class is < 1 or > 3.
2. Two cells in the same class have the same seed.
3. Any cell's slice pack CID does not match the class set's
   first cell's slice pack CID.
4. Any cell's bench report schema is unrecognised.

Honest scope (W105)
-------------------

* ``W105-L-PHASE3-EVALUATOR-V1-SLICE-PACK-EQUAL-CAP`` — the
  evaluator REQUIRES byte-equal slice pack CIDs across all
  cells in a class.  Comparing two cells on DIFFERENT slice
  packs is structurally invalid.
* ``W105-L-PHASE3-EVALUATOR-V1-PER-CLASS-FIRST-CAP`` — the
  cross-class claim layer is ONLY emitted after both per-class
  verdicts have been computed independently.  No averaging
  across classes that hides a class-specific FAIL.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any


W105_PHASE3_RETIREMENT_EVALUATOR_V1_SCHEMA_VERSION: str = (
    "coordpy.phase3_retirement_evaluator_v1.v1")


# Schema versions the evaluator recognises.  W105 supports the
# W103+ HumanEval+ bench report shape (1-seed cells produced by
# the Phase 3 driver, one per seed).
_RECOGNISED_BENCH_REPORT_SCHEMAS: frozenset[str] = frozenset({
    "coordpy.humaneval_plus_reflexion_bench_v1.v1",
})


W105_PHASE3_MARGIN_FLOOR_PP: float = 5.0
W105_PHASE3_PER_PROBLEM_MAJORITY_FRACTION: float = 0.53
W105_PHASE3_PER_SEED_MAJORITY_MIN: int = 2
W105_PHASE3_A1_SATURATION_MAX_PCT: float = 90.0
W105_PHASE3_MLB2_FLOOR: float = 0.33
W105_PHASE3_CROSS_CLASS_DIFF_PP_ABS_MAX: float = 5.0


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


class Phase3RetirementEvaluatorError(ValueError):
    """Raised when the evaluator refuses to run because the
    inputs are structurally incompatible."""


@dataclasses.dataclass(frozen=True)
class Phase3CellSummaryV1:
    """One per-(class, seed) Phase 3 cell summary."""

    model_class_id: str
    seed: int
    n_problems: int
    a0_pct: float
    a1_pct: float
    b_pct: float
    b_minus_a1_pp: float
    n_b_ge_a1: int          # per-problem count where B did not lose to A1
    mlb1_invocation_rate: float
    mlb2_rescue_rate: float
    a1_lt_saturation_max: bool
    bench_merkle_root: str
    slice_pack_cid: str
    cell_run_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_class_id": str(self.model_class_id),
            "seed": int(self.seed),
            "n_problems": int(self.n_problems),
            "a0_pct": float(round(self.a0_pct, 4)),
            "a1_pct": float(round(self.a1_pct, 4)),
            "b_pct": float(round(self.b_pct, 4)),
            "b_minus_a1_pp": float(round(self.b_minus_a1_pp, 4)),
            "n_b_ge_a1": int(self.n_b_ge_a1),
            "mlb1_invocation_rate": float(round(
                self.mlb1_invocation_rate, 4)),
            "mlb2_rescue_rate": float(round(
                self.mlb2_rescue_rate, 4)),
            "a1_lt_saturation_max": bool(self.a1_lt_saturation_max),
            "bench_merkle_root": str(self.bench_merkle_root),
            "slice_pack_cid": str(self.slice_pack_cid),
            "cell_run_id": str(self.cell_run_id),
        }


@dataclasses.dataclass(frozen=True)
class Phase3PerClassVerdictV1:
    """Per-class Phase 3 verdict."""

    model_class_id: str
    cells: tuple[Phase3CellSummaryV1, ...]
    bar1_margin_mean_b_minus_a1_pp: float
    bar1_margin_passes: bool
    bar2_per_seed_majority_count: int
    bar2_per_seed_majority_passes: bool
    bar3_per_problem_majority_count: int
    bar3_per_problem_majority_floor: int
    bar3_per_problem_majority_passes: bool
    bar4_a1_not_saturated_passes: bool
    bar4_per_cell_a1_pcts: tuple[float, ...]
    bar5_audit_chain_passes: bool
    bar5_audit_chain_re_derives: int
    bar5_audit_chain_total: int
    bar6_executor_clean_passes: bool
    bar6_canonical_pass_rate: float
    mean_mlb1_invocation_rate: float
    mean_mlb2_rescue_rate: float
    mlb2_load_bearing: bool
    verdict_label: str
    n_bars_passed_of_6: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_class_id": str(self.model_class_id),
            "cells": [c.to_dict() for c in self.cells],
            "bar1_margin_mean_b_minus_a1_pp": float(round(
                self.bar1_margin_mean_b_minus_a1_pp, 4)),
            "bar1_margin_passes": bool(self.bar1_margin_passes),
            "bar2_per_seed_majority_count": int(
                self.bar2_per_seed_majority_count),
            "bar2_per_seed_majority_passes": bool(
                self.bar2_per_seed_majority_passes),
            "bar3_per_problem_majority_count": int(
                self.bar3_per_problem_majority_count),
            "bar3_per_problem_majority_floor": int(
                self.bar3_per_problem_majority_floor),
            "bar3_per_problem_majority_passes": bool(
                self.bar3_per_problem_majority_passes),
            "bar4_a1_not_saturated_passes": bool(
                self.bar4_a1_not_saturated_passes),
            "bar4_per_cell_a1_pcts": list(
                self.bar4_per_cell_a1_pcts),
            "bar5_audit_chain_passes": bool(
                self.bar5_audit_chain_passes),
            "bar5_audit_chain_re_derives": int(
                self.bar5_audit_chain_re_derives),
            "bar5_audit_chain_total": int(
                self.bar5_audit_chain_total),
            "bar6_executor_clean_passes": bool(
                self.bar6_executor_clean_passes),
            "bar6_canonical_pass_rate": float(round(
                self.bar6_canonical_pass_rate, 4)),
            "mean_mlb1_invocation_rate": float(round(
                self.mean_mlb1_invocation_rate, 4)),
            "mean_mlb2_rescue_rate": float(round(
                self.mean_mlb2_rescue_rate, 4)),
            "mlb2_load_bearing": bool(self.mlb2_load_bearing),
            "verdict_label": str(self.verdict_label),
            "n_bars_passed_of_6": int(self.n_bars_passed_of_6),
        }


@dataclasses.dataclass(frozen=True)
class Phase3CrossClassVerdictV1:
    """Cross-class entitlement layer (computed AFTER per-class)."""

    class_a_id: str
    class_b_id: str
    class_a_verdict_label: str
    class_b_verdict_label: str
    class_a_margin_pp: float
    class_b_margin_pp: float
    cross_class_b_minus_a1_diff_pp: float
    cross_class_diff_within_envelope: bool
    cross_class_retirement_entitled: bool
    cross_class_claim_label: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_a_id": str(self.class_a_id),
            "class_b_id": str(self.class_b_id),
            "class_a_verdict_label": str(
                self.class_a_verdict_label),
            "class_b_verdict_label": str(
                self.class_b_verdict_label),
            "class_a_margin_pp": float(round(
                self.class_a_margin_pp, 4)),
            "class_b_margin_pp": float(round(
                self.class_b_margin_pp, 4)),
            "cross_class_b_minus_a1_diff_pp": float(round(
                self.cross_class_b_minus_a1_diff_pp, 4)),
            "cross_class_diff_within_envelope": bool(
                self.cross_class_diff_within_envelope),
            "cross_class_retirement_entitled": bool(
                self.cross_class_retirement_entitled),
            "cross_class_claim_label": str(
                self.cross_class_claim_label),
        }


@dataclasses.dataclass(frozen=True)
class Phase3RetirementVerdictV1:
    """End-to-end Phase 3 retirement verdict."""

    schema: str
    slice_pack_cid: str
    corpus_sha256: str
    per_class: tuple[Phase3PerClassVerdictV1, ...]
    cross_class: Phase3CrossClassVerdictV1 | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "slice_pack_cid": str(self.slice_pack_cid),
            "corpus_sha256": str(self.corpus_sha256),
            "per_class": [c.to_dict() for c in self.per_class],
            "cross_class": (
                self.cross_class.to_dict()
                if self.cross_class is not None else None),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w105_phase3_retirement_verdict_v1",
            "verdict": self.to_dict(),
        })


def _validate_cell(
        *, bench_report: dict[str, Any],
        provenance: dict[str, Any], label: str) -> None:
    schema = str(bench_report.get("schema") or "")
    if schema not in _RECOGNISED_BENCH_REPORT_SCHEMAS:
        raise Phase3RetirementEvaluatorError(
            f"{label} bench report schema {schema!r} not in "
            f"recognised set "
            f"{sorted(_RECOGNISED_BENCH_REPORT_SCHEMAS)}")
    if not bench_report.get("mlb"):
        raise Phase3RetirementEvaluatorError(
            f"{label} bench report missing required `mlb` "
            "block")
    if "a0_mean_pass_at_1" not in bench_report:
        raise Phase3RetirementEvaluatorError(
            f"{label} bench report missing a0_mean_pass_at_1")
    if "a1_mean_pass_at_1" not in bench_report:
        raise Phase3RetirementEvaluatorError(
            f"{label} bench report missing a1_mean_pass_at_1")
    if "b_mean_pass_at_1" not in bench_report:
        raise Phase3RetirementEvaluatorError(
            f"{label} bench report missing b_mean_pass_at_1")
    for key in (
            "corpus_sha256",
            "slice_pack_cid",
            "seed",
            "model_id"):
        if provenance.get(key) in (None, ""):
            raise Phase3RetirementEvaluatorError(
                f"{label} provenance missing required field "
                f"{key!r}")


def _build_cell_summary(
        *, bench_report: dict[str, Any],
        provenance: dict[str, Any]) -> Phase3CellSummaryV1:
    per_seed_blocks = bench_report.get("per_seed") or []
    if len(per_seed_blocks) != 1:
        raise Phase3RetirementEvaluatorError(
            "Phase 3 cell bench report must have exactly one "
            f"per_seed block; got n_seeds={len(per_seed_blocks)}"
        )
    block = per_seed_blocks[0]
    n_problems = int(len(block.get("per_problem_b_passed", [])))
    if n_problems == 0:
        raise Phase3RetirementEvaluatorError(
            "Phase 3 cell bench report has zero problems")
    n_b_ge_a1 = 0
    for i in range(n_problems):
        b_p = bool(block["per_problem_b_passed"][i])
        a1_p = bool(block["per_problem_a1_passed"][i])
        if not (a1_p and not b_p):
            n_b_ge_a1 += 1
    a0_pct = float(bench_report["a0_mean_pass_at_1"]) * 100.0
    a1_pct = float(bench_report["a1_mean_pass_at_1"]) * 100.0
    b_pct = float(bench_report["b_mean_pass_at_1"]) * 100.0
    mlb = bench_report["mlb"]
    return Phase3CellSummaryV1(
        model_class_id=str(provenance.get("model_id") or ""),
        seed=int(provenance.get("seed") or 0),
        n_problems=int(n_problems),
        a0_pct=float(a0_pct),
        a1_pct=float(a1_pct),
        b_pct=float(b_pct),
        b_minus_a1_pp=float(b_pct - a1_pct),
        n_b_ge_a1=int(n_b_ge_a1),
        mlb1_invocation_rate=float(
            mlb["mlb1_invocation_rate"]),
        mlb2_rescue_rate=float(mlb["mlb2_rescue_rate"]),
        a1_lt_saturation_max=bool(
            a1_pct < W105_PHASE3_A1_SATURATION_MAX_PCT),
        bench_merkle_root=str(
            bench_report.get("bench_merkle_root") or ""),
        slice_pack_cid=str(
            provenance.get("slice_pack_cid") or ""),
        cell_run_id=str(provenance.get("cell_run_id") or ""))


def evaluate_per_class_verdict_v1(
        *,
        model_class_id: str,
        cells: tuple[Phase3CellSummaryV1, ...],
        audit_chain_re_derives: int,
        audit_chain_total: int,
        canonical_pass_rate: float,
) -> Phase3PerClassVerdictV1:
    """Apply the 6 retirement bars to a single class's cells.

    Refuses if cells empty / > 3 / duplicate seeds / slice pack
    CID mismatch.
    """
    if not cells:
        raise Phase3RetirementEvaluatorError(
            f"per-class evaluator: zero cells for "
            f"{model_class_id!r}")
    if len(cells) > 3:
        raise Phase3RetirementEvaluatorError(
            f"per-class evaluator: too many cells for "
            f"{model_class_id!r}: {len(cells)} > 3")
    seeds_seen: set[int] = set()
    for c in cells:
        if int(c.seed) in seeds_seen:
            raise Phase3RetirementEvaluatorError(
                f"per-class evaluator: duplicate seed "
                f"{c.seed} for class {model_class_id!r}")
        seeds_seen.add(int(c.seed))
    first_pack_cid = str(cells[0].slice_pack_cid)
    if not first_pack_cid:
        raise Phase3RetirementEvaluatorError(
            "per-class evaluator: cells missing slice_pack_cid")
    for c in cells:
        if str(c.slice_pack_cid) != first_pack_cid:
            raise Phase3RetirementEvaluatorError(
                f"per-class evaluator: cells have mismatched "
                f"slice_pack_cid for class {model_class_id!r}")
    margin_mean = sum(c.b_minus_a1_pp for c in cells) / float(
        len(cells))
    bar1 = bool(margin_mean >= W105_PHASE3_MARGIN_FLOOR_PP)
    per_seed_majority_count = sum(
        1 for c in cells if c.b_pct > c.a1_pct)
    bar2 = bool(
        per_seed_majority_count
        >= W105_PHASE3_PER_SEED_MAJORITY_MIN)
    per_problem_total = sum(
        int(c.n_problems) for c in cells)
    per_problem_majority_count = sum(
        int(c.n_b_ge_a1) for c in cells)
    floor3 = int(
        round(
            per_problem_total
            * W105_PHASE3_PER_PROBLEM_MAJORITY_FRACTION))
    bar3 = bool(per_problem_majority_count >= floor3)
    a1_pcts = tuple(float(c.a1_pct) for c in cells)
    bar4 = bool(all(
        x < W105_PHASE3_A1_SATURATION_MAX_PCT for x in a1_pcts))
    bar5 = bool(
        int(audit_chain_re_derives)
        >= max(int(audit_chain_total) - 1, 0))
    bar6 = bool(float(canonical_pass_rate) >= 1.0 - 1e-6)
    n_bars = int(bar1) + int(bar2) + int(bar3) + int(bar4) + (
        int(bar5) + int(bar6))
    mean_mlb1 = sum(
        c.mlb1_invocation_rate for c in cells) / float(len(cells))
    mean_mlb2 = sum(
        c.mlb2_rescue_rate for c in cells) / float(len(cells))
    mlb2_load_bearing = bool(
        mean_mlb2 >= W105_PHASE3_MLB2_FLOOR)
    if n_bars == 6:
        if mlb2_load_bearing:
            verdict_label = "RETIRED"
        else:
            verdict_label = (
                "RETIRED_MARGIN_DRIVEN_NON_LOAD_BEARING")
    else:
        fail_reasons: list[str] = []
        if not bar1:
            fail_reasons.append("MARGIN")
        if not bar2:
            fail_reasons.append("PER_SEED_MAJORITY")
        if not bar3:
            fail_reasons.append("PER_PROBLEM_MAJORITY")
        if not bar4:
            fail_reasons.append("A1_SATURATION")
        if not bar5:
            fail_reasons.append("AUDIT_CHAIN")
        if not bar6:
            fail_reasons.append("EXECUTOR_CLEAN")
        verdict_label = "FAIL_" + "_".join(fail_reasons)
    return Phase3PerClassVerdictV1(
        model_class_id=str(model_class_id),
        cells=tuple(cells),
        bar1_margin_mean_b_minus_a1_pp=float(margin_mean),
        bar1_margin_passes=bool(bar1),
        bar2_per_seed_majority_count=int(
            per_seed_majority_count),
        bar2_per_seed_majority_passes=bool(bar2),
        bar3_per_problem_majority_count=int(
            per_problem_majority_count),
        bar3_per_problem_majority_floor=int(floor3),
        bar3_per_problem_majority_passes=bool(bar3),
        bar4_a1_not_saturated_passes=bool(bar4),
        bar4_per_cell_a1_pcts=a1_pcts,
        bar5_audit_chain_passes=bool(bar5),
        bar5_audit_chain_re_derives=int(audit_chain_re_derives),
        bar5_audit_chain_total=int(audit_chain_total),
        bar6_executor_clean_passes=bool(bar6),
        bar6_canonical_pass_rate=float(canonical_pass_rate),
        mean_mlb1_invocation_rate=float(mean_mlb1),
        mean_mlb2_rescue_rate=float(mean_mlb2),
        mlb2_load_bearing=bool(mlb2_load_bearing),
        verdict_label=str(verdict_label),
        n_bars_passed_of_6=int(n_bars))


def evaluate_cross_class_verdict_v1(
        *,
        class_a: Phase3PerClassVerdictV1,
        class_b: Phase3PerClassVerdictV1,
) -> Phase3CrossClassVerdictV1:
    """Layer the cross-class claim on top of two per-class
    verdicts.  Cross-class retirement entitled IFF BOTH classes
    RETIRED (NOT _MARGIN_DRIVEN_NON_LOAD_BEARING) AND the
    cross-class B − A1 difference is within ± 5 pp."""
    a_margin = float(class_a.bar1_margin_mean_b_minus_a1_pp)
    b_margin = float(class_b.bar1_margin_mean_b_minus_a1_pp)
    diff = float(abs(a_margin - b_margin))
    within = bool(
        diff <= W105_PHASE3_CROSS_CLASS_DIFF_PP_ABS_MAX)
    both_retired = bool(
        str(class_a.verdict_label) == "RETIRED"
        and str(class_b.verdict_label) == "RETIRED")
    entitled = bool(both_retired and within)
    if entitled:
        claim_label = "CROSS_CLASS_RETIRED"
    elif both_retired and not within:
        claim_label = (
            "CROSS_CLASS_PARTIAL_RETIRED_MARGIN_GAP_EXCEEDS_ENVELOPE")
    elif (str(class_a.verdict_label) == "RETIRED"
          and str(class_b.verdict_label).startswith("FAIL")):
        claim_label = (
            "CLASS_A_RETIRED_CLASS_B_FAIL_BOUNDED_CLAIM")
    elif (str(class_b.verdict_label) == "RETIRED"
          and str(class_a.verdict_label).startswith("FAIL")):
        claim_label = (
            "CLASS_B_RETIRED_CLASS_A_FAIL_BOUNDED_CLAIM")
    elif (str(class_a.verdict_label)
          == "RETIRED_MARGIN_DRIVEN_NON_LOAD_BEARING"
          or str(class_b.verdict_label)
          == "RETIRED_MARGIN_DRIVEN_NON_LOAD_BEARING"):
        claim_label = (
            "AT_LEAST_ONE_CLASS_MARGIN_DRIVEN_NON_LOAD_BEARING")
    else:
        claim_label = "CROSS_CLASS_NOT_ENTITLED"
    return Phase3CrossClassVerdictV1(
        class_a_id=str(class_a.model_class_id),
        class_b_id=str(class_b.model_class_id),
        class_a_verdict_label=str(class_a.verdict_label),
        class_b_verdict_label=str(class_b.verdict_label),
        class_a_margin_pp=float(a_margin),
        class_b_margin_pp=float(b_margin),
        cross_class_b_minus_a1_diff_pp=float(a_margin - b_margin),
        cross_class_diff_within_envelope=bool(within),
        cross_class_retirement_entitled=bool(entitled),
        cross_class_claim_label=str(claim_label))


def build_phase3_retirement_verdict_v1(
        *,
        cells_by_class: dict[str,
                             list[tuple[dict[str, Any],
                                        dict[str, Any]]]],
        audit_chain_re_derives_by_class: dict[str, int],
        audit_chain_total_by_class: dict[str, int],
        canonical_pass_rate_by_class: dict[str, float],
) -> Phase3RetirementVerdictV1:
    """End-to-end Phase 3 retirement verdict builder.

    ``cells_by_class`` maps a model class id to a list of
    (bench_report_dict, provenance_dict) pairs (one per seed).
    Audit + canonical pass-rate are passed in by class (the
    driver records these).  Returns a structured verdict.
    """
    if not cells_by_class:
        raise Phase3RetirementEvaluatorError(
            "build_phase3_retirement_verdict_v1: empty "
            "cells_by_class")
    per_class_verdicts: list[Phase3PerClassVerdictV1] = []
    all_slice_pack_cids: set[str] = set()
    all_corpus_shas: set[str] = set()
    for class_id, cell_pairs in cells_by_class.items():
        cells: list[Phase3CellSummaryV1] = []
        for i, (br, prov) in enumerate(cell_pairs):
            _validate_cell(
                bench_report=br, provenance=prov,
                label=f"{class_id}#cell{i}")
            cells.append(
                _build_cell_summary(
                    bench_report=br, provenance=prov))
            all_slice_pack_cids.add(
                str(prov.get("slice_pack_cid") or ""))
            all_corpus_shas.add(
                str(prov.get("corpus_sha256") or ""))
        per_class_verdicts.append(
            evaluate_per_class_verdict_v1(
                model_class_id=str(class_id),
                cells=tuple(cells),
                audit_chain_re_derives=int(
                    audit_chain_re_derives_by_class.get(
                        class_id, 0)),
                audit_chain_total=int(
                    audit_chain_total_by_class.get(
                        class_id, 0)),
                canonical_pass_rate=float(
                    canonical_pass_rate_by_class.get(
                        class_id, 0.0))))
    if len(all_slice_pack_cids) != 1:
        raise Phase3RetirementEvaluatorError(
            "build_phase3_retirement_verdict_v1: cells span "
            f"multiple slice_pack_cids: {sorted(all_slice_pack_cids)}")
    if len(all_corpus_shas) != 1:
        raise Phase3RetirementEvaluatorError(
            "build_phase3_retirement_verdict_v1: cells span "
            f"multiple corpus_sha256: {sorted(all_corpus_shas)}")
    slice_pack_cid = next(iter(all_slice_pack_cids))
    corpus_sha = next(iter(all_corpus_shas))
    cross_class_verdict: Phase3CrossClassVerdictV1 | None = None
    if len(per_class_verdicts) == 2:
        cross_class_verdict = evaluate_cross_class_verdict_v1(
            class_a=per_class_verdicts[0],
            class_b=per_class_verdicts[1])
    return Phase3RetirementVerdictV1(
        schema=W105_PHASE3_RETIREMENT_EVALUATOR_V1_SCHEMA_VERSION,
        slice_pack_cid=str(slice_pack_cid),
        corpus_sha256=str(corpus_sha),
        per_class=tuple(per_class_verdicts),
        cross_class=cross_class_verdict)


def format_phase3_retirement_verdict_markdown_v1(
        *, verdict: Phase3RetirementVerdictV1) -> str:
    """Pretty-print the Phase 3 retirement verdict for the
    W105 docs."""
    out: list[str] = []
    out.append(
        f"## W105 Phase 3 retirement verdict (schema "
        f"`{verdict.schema}`)\n\n")
    out.append(
        f"* slice_pack_cid: `{verdict.slice_pack_cid}`\n"
        f"* corpus_sha256: `{verdict.corpus_sha256}`\n\n")
    for c in verdict.per_class:
        d = c.to_dict()
        out.append(
            f"### Class `{d['model_class_id']}` — verdict "
            f"**{d['verdict_label']}** "
            f"({d['n_bars_passed_of_6']}/6 bars PASS)\n\n")
        out.append(
            "| # | Bar | Value | Threshold | PASS |\n"
            "|---|---|---|---|---|\n"
            f"| 1 | mean B − A1 (pp) | "
            f"{d['bar1_margin_mean_b_minus_a1_pp']:+.2f} | "
            f"≥ +{W105_PHASE3_MARGIN_FLOOR_PP:.1f} | "
            f"{'YES' if d['bar1_margin_passes'] else 'NO'} |\n"
            f"| 2 | per-seed majority count | "
            f"{d['bar2_per_seed_majority_count']} | "
            f"≥ {W105_PHASE3_PER_SEED_MAJORITY_MIN} | "
            f"{'YES' if d['bar2_per_seed_majority_passes'] else 'NO'} |\n"
            f"| 3 | per-problem majority count | "
            f"{d['bar3_per_problem_majority_count']} | "
            f"≥ {d['bar3_per_problem_majority_floor']} | "
            f"{'YES' if d['bar3_per_problem_majority_passes'] else 'NO'} |\n"
            f"| 4 | A1 < {W105_PHASE3_A1_SATURATION_MAX_PCT:.0f} %"
            f" on each cell | "
            f"{', '.join(f'{x:.2f}' for x in d['bar4_per_cell_a1_pcts'])} | "
            f"all < {W105_PHASE3_A1_SATURATION_MAX_PCT:.0f} | "
            f"{'YES' if d['bar4_a1_not_saturated_passes'] else 'NO'} |\n"
            f"| 5 | audit chain re-derives | "
            f"{d['bar5_audit_chain_re_derives']}/"
            f"{d['bar5_audit_chain_total']} | "
            f"≥ {d['bar5_audit_chain_total']-1} | "
            f"{'YES' if d['bar5_audit_chain_passes'] else 'NO'} |\n"
            f"| 6 | canonical-solution pass rate | "
            f"{d['bar6_canonical_pass_rate']*100:.2f} % | "
            f"= 100 % | "
            f"{'YES' if d['bar6_executor_clean_passes'] else 'NO'} |\n"
            f"\n"
            f"* mean MLB-1 invocation: "
            f"{d['mean_mlb1_invocation_rate']*100:.2f} %\n"
            f"* mean MLB-2 rescue: "
            f"{d['mean_mlb2_rescue_rate']*100:.2f} % "
            f"({'load-bearing' if d['mlb2_load_bearing'] else 'NOT load-bearing'})\n\n")
        out.append("Per-cell summary:\n\n")
        out.append(
            "| seed | A0 % | A1 % | B % | B − A1 pp | "
            "MLB-1 % | MLB-2 % | bench Merkle |\n"
            "|---|---:|---:|---:|---:|---:|---:|---|\n")
        for cell in d["cells"]:
            out.append(
                f"| {cell['seed']} | "
                f"{cell['a0_pct']:.2f} | "
                f"{cell['a1_pct']:.2f} | "
                f"{cell['b_pct']:.2f} | "
                f"{cell['b_minus_a1_pp']:+.2f} | "
                f"{cell['mlb1_invocation_rate']*100:.2f} | "
                f"{cell['mlb2_rescue_rate']*100:.2f} | "
                f"`{cell['bench_merkle_root'][:16]}...` |\n")
        out.append("\n")
    if verdict.cross_class is not None:
        cc = verdict.cross_class.to_dict()
        out.append(
            f"### Cross-class verdict — "
            f"**{cc['cross_class_claim_label']}**\n\n"
            f"* class A (`{cc['class_a_id']}`) verdict: "
            f"`{cc['class_a_verdict_label']}` "
            f"(margin {cc['class_a_margin_pp']:+.2f} pp)\n"
            f"* class B (`{cc['class_b_id']}`) verdict: "
            f"`{cc['class_b_verdict_label']}` "
            f"(margin {cc['class_b_margin_pp']:+.2f} pp)\n"
            f"* cross-class B − A1 difference: "
            f"{cc['cross_class_b_minus_a1_diff_pp']:+.2f} pp "
            f"(envelope ± {W105_PHASE3_CROSS_CLASS_DIFF_PP_ABS_MAX:.1f}; "
            f"within = "
            f"{'YES' if cc['cross_class_diff_within_envelope'] else 'NO'})\n"
            f"* cross-class retirement entitled: "
            f"{'YES' if cc['cross_class_retirement_entitled'] else 'NO'}\n")
    return "".join(out)


__all__ = (
    "W105_PHASE3_RETIREMENT_EVALUATOR_V1_SCHEMA_VERSION",
    "W105_PHASE3_MARGIN_FLOOR_PP",
    "W105_PHASE3_PER_PROBLEM_MAJORITY_FRACTION",
    "W105_PHASE3_PER_SEED_MAJORITY_MIN",
    "W105_PHASE3_A1_SATURATION_MAX_PCT",
    "W105_PHASE3_MLB2_FLOOR",
    "W105_PHASE3_CROSS_CLASS_DIFF_PP_ABS_MAX",
    "Phase3RetirementEvaluatorError",
    "Phase3CellSummaryV1",
    "Phase3PerClassVerdictV1",
    "Phase3CrossClassVerdictV1",
    "Phase3RetirementVerdictV1",
    "evaluate_per_class_verdict_v1",
    "evaluate_cross_class_verdict_v1",
    "build_phase3_retirement_verdict_v1",
    "format_phase3_retirement_verdict_markdown_v1",
)
