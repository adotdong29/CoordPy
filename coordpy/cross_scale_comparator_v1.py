"""W104 / COO-9 — Cross-scale comparator V1.

Reads two HumanEval+ bench reports (+ their provenance JSONs)
and emits a structured cross-scale diff that catches 70B-vs-405B
mix-ups at write time, not at post-hoc autopsy time.

The comparator is a pure-Python module with no NIM dependency
and no model loading.  It is explicit-import only — it does NOT
register itself in ``coordpy/__init__.py``.

Inputs (per scale):

* The bench report dict (matches
  ``coordpy.humaneval_plus_reflexion_bench_v1.HumanEvalPlusBenchReportV1.to_dict``
  shape, with the W103+ pilot driver's added ``provenance``,
  ``mlb``, ``phase2_evaluation`` and ``wall_s`` envelope fields).
* The provenance dict (matches the W103+ pilot driver's
  ``provenance.json`` shape).

Outputs:

* A structured cross-scale comparator dict with per-problem
  cluster shifts (``stayed`` / ``improved`` / ``regressed`` /
  ``flipped``), aggregate cluster transitions, and aggregate
  arm deltas.

Refuses to run if:

1. The two reports do not share a slice CID.
2. The two reports do not share a corpus SHA-256.
3. Either report's MLB block is missing.
4. Either report's schema version is unrecognised.

Honest scope (W104)
-------------------

* ``W104-L-CROSS-SCALE-COMPARATOR-V1-SLICE-EQUAL-CAP`` — the
  comparator REQUIRES byte-equal slices.  Comparing two pilots
  on DIFFERENT slices is structurally invalid; the comparator
  refuses rather than silently emitting a misleading diff.
* ``W104-L-CROSS-SCALE-COMPARATOR-V1-CORPUS-PIN-CAP`` —
  corpus SHA mismatch refuses to run.  A future corpus update
  would be silent shift in the truth surface; the pin guards
  this.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any


W104_CROSS_SCALE_COMPARATOR_V1_SCHEMA_VERSION: str = (
    "coordpy.cross_scale_comparator_v1.v1")


# Schema versions the comparator recognises.  W104 supports the
# W103+ HumanEval+ bench report shape; future scales/benches
# extend this set explicitly.
_RECOGNISED_BENCH_REPORT_SCHEMAS: frozenset[str] = frozenset({
    "coordpy.humaneval_plus_reflexion_bench_v1.v1",
})


# Cross-scale cluster shift names.  Per-problem, comparing the
# (A0, A1, B) triple at scale A vs scale B.
CROSS_SCALE_CLUSTER_SHIFT_VALUES: tuple[str, ...] = (
    "stayed",       # same (A0, A1, B) outcome at both scales
    "improved",     # B improved at scale B (or A1 improved
                    # while B held) — strictly more passes
                    # at scale B
    "regressed",    # B regressed at scale B (or A1 regressed
                    # while B held) — strictly fewer passes
                    # at scale B
    "flipped",      # ambiguous transition (e.g., A0 lost but
                    # B gained, or vice versa)
)


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


class CrossScaleComparatorError(ValueError):
    """Raised when the comparator refuses to run because the
    two inputs are structurally incompatible."""


@dataclasses.dataclass(frozen=True)
class CrossScaleProblemRow:
    """One per-problem cross-scale row."""

    task_id: str
    a0_at_scale_a: bool
    a1_at_scale_a: bool
    b_at_scale_a: bool
    b_first_pass_idx_at_scale_a: int
    a0_at_scale_b: bool
    a1_at_scale_b: bool
    b_at_scale_b: bool
    b_first_pass_idx_at_scale_b: int
    cluster_shift: str  # one of CROSS_SCALE_CLUSTER_SHIFT_VALUES

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": str(self.task_id),
            "a0_at_scale_a": bool(self.a0_at_scale_a),
            "a1_at_scale_a": bool(self.a1_at_scale_a),
            "b_at_scale_a": bool(self.b_at_scale_a),
            "b_first_pass_idx_at_scale_a": int(
                self.b_first_pass_idx_at_scale_a),
            "a0_at_scale_b": bool(self.a0_at_scale_b),
            "a1_at_scale_b": bool(self.a1_at_scale_b),
            "b_at_scale_b": bool(self.b_at_scale_b),
            "b_first_pass_idx_at_scale_b": int(
                self.b_first_pass_idx_at_scale_b),
            "cluster_shift": str(self.cluster_shift),
        }


@dataclasses.dataclass(frozen=True)
class CrossScaleComparatorReportV1:
    """The cross-scale comparator report shape."""

    schema: str
    scale_a_model_id: str
    scale_b_model_id: str
    scale_a_bench_merkle: str
    scale_b_bench_merkle: str
    slice_cid: str
    corpus_sha256: str
    n_problems: int
    per_problem: tuple[CrossScaleProblemRow, ...]
    aggregate_arm_deltas_pp: dict[str, float]
    aggregate_cluster_shift_counts: dict[str, int]
    aggregate_b_minus_a1_pp_at_scale_a: float
    aggregate_b_minus_a1_pp_at_scale_b: float
    cross_scale_shift_on_b_minus_a1_pp: float
    aggregate_mlb1_invocation_rate_at_scale_a: float
    aggregate_mlb1_invocation_rate_at_scale_b: float
    aggregate_mlb2_rescue_rate_at_scale_a: float
    aggregate_mlb2_rescue_rate_at_scale_b: float
    cross_scale_shift_on_mlb2_pp: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "scale_a_model_id": str(self.scale_a_model_id),
            "scale_b_model_id": str(self.scale_b_model_id),
            "scale_a_bench_merkle": str(
                self.scale_a_bench_merkle),
            "scale_b_bench_merkle": str(
                self.scale_b_bench_merkle),
            "slice_cid": str(self.slice_cid),
            "corpus_sha256": str(self.corpus_sha256),
            "n_problems": int(self.n_problems),
            "per_problem": [
                r.to_dict() for r in self.per_problem],
            "aggregate_arm_deltas_pp": dict(
                self.aggregate_arm_deltas_pp),
            "aggregate_cluster_shift_counts": dict(
                self.aggregate_cluster_shift_counts),
            "aggregate_b_minus_a1_pp_at_scale_a": float(round(
                self.aggregate_b_minus_a1_pp_at_scale_a, 4)),
            "aggregate_b_minus_a1_pp_at_scale_b": float(round(
                self.aggregate_b_minus_a1_pp_at_scale_b, 4)),
            "cross_scale_shift_on_b_minus_a1_pp": float(round(
                self.cross_scale_shift_on_b_minus_a1_pp, 4)),
            "aggregate_mlb1_invocation_rate_at_scale_a": float(
                round(
                    self.aggregate_mlb1_invocation_rate_at_scale_a,
                    4)),
            "aggregate_mlb1_invocation_rate_at_scale_b": float(
                round(
                    self.aggregate_mlb1_invocation_rate_at_scale_b,
                    4)),
            "aggregate_mlb2_rescue_rate_at_scale_a": float(round(
                self.aggregate_mlb2_rescue_rate_at_scale_a, 4)),
            "aggregate_mlb2_rescue_rate_at_scale_b": float(round(
                self.aggregate_mlb2_rescue_rate_at_scale_b, 4)),
            "cross_scale_shift_on_mlb2_pp": float(round(
                self.cross_scale_shift_on_mlb2_pp, 4)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w104_cross_scale_comparator_report_v1",
            "report": self.to_dict(),
        })


def _per_problem_cluster_shift(
        *,
        a0_a: bool, a1_a: bool, b_a: bool,
        a0_b: bool, a1_b: bool, b_b: bool) -> str:
    """Classify the per-problem cross-scale shift.

    The rule (locked here, mirrors the W104 RUNBOOK § Hardening
    lane definition):

    * `stayed` — (a0, a1, b) tuple identical at both scales.
    * `improved` — at least one arm flipped F→T at scale B AND
      no arm flipped T→F.
    * `regressed` — at least one arm flipped T→F at scale B AND
      no arm flipped F→T.
    * `flipped` — at least one arm changed in EACH direction
      (truly ambiguous; "improved on some, regressed on others").
    """
    flipped_to_pass = 0
    flipped_to_fail = 0
    for sa, sb in (
            (bool(a0_a), bool(a0_b)),
            (bool(a1_a), bool(a1_b)),
            (bool(b_a), bool(b_b))):
        if sa == sb:
            continue
        if sb and not sa:
            flipped_to_pass += 1
        else:
            flipped_to_fail += 1
    if flipped_to_pass == 0 and flipped_to_fail == 0:
        return "stayed"
    if flipped_to_pass > 0 and flipped_to_fail == 0:
        return "improved"
    if flipped_to_fail > 0 and flipped_to_pass == 0:
        return "regressed"
    return "flipped"


def _extract_seed_block(report: dict[str, Any]) -> dict[str, Any]:
    """Pull the single per-seed block from a 1-seed bench
    report.  The W104 comparator assumes 1-seed cheap pilots
    on each scale (matches W103); multi-seed comparisons are
    out of scope for V1."""
    per_seed = report.get("per_seed") or []
    if not per_seed:
        raise CrossScaleComparatorError(
            "bench report missing per_seed block")
    if len(per_seed) != 1:
        raise CrossScaleComparatorError(
            "W104 cross-scale comparator V1 requires "
            "1-seed cheap pilots on each scale; got "
            f"n_seeds={len(per_seed)}")
    return per_seed[0]


def _validate_report(*, report: dict[str, Any], label: str) -> None:
    schema = str(report.get("schema") or "")
    if schema not in _RECOGNISED_BENCH_REPORT_SCHEMAS:
        raise CrossScaleComparatorError(
            f"{label} report schema {schema!r} not in "
            f"recognised set {sorted(_RECOGNISED_BENCH_REPORT_SCHEMAS)}")
    if not report.get("mlb"):
        raise CrossScaleComparatorError(
            f"{label} report missing required `mlb` block")
    if "a0_mean_pass_at_1" not in report:
        raise CrossScaleComparatorError(
            f"{label} report missing a0_mean_pass_at_1")
    if "a1_mean_pass_at_1" not in report:
        raise CrossScaleComparatorError(
            f"{label} report missing a1_mean_pass_at_1")
    if "b_mean_pass_at_1" not in report:
        raise CrossScaleComparatorError(
            f"{label} report missing b_mean_pass_at_1")


def _validate_provenance(
        *, provenance: dict[str, Any], label: str) -> None:
    for key in (
            "corpus_sha256",
            "slice_cid_bench_order"):
        if not provenance.get(key):
            raise CrossScaleComparatorError(
                f"{label} provenance missing required field "
                f"{key!r}")


def build_cross_scale_comparator_report_v1(
        *,
        scale_a_bench_report: dict[str, Any],
        scale_a_provenance: dict[str, Any],
        scale_b_bench_report: dict[str, Any],
        scale_b_provenance: dict[str, Any],
) -> CrossScaleComparatorReportV1:
    """Build the cross-scale comparator report from two
    HumanEval+ bench reports + provenance pairs.

    Refuses to run on schema / slice / corpus mismatch.
    """
    _validate_report(report=scale_a_bench_report, label="scale_a")
    _validate_report(report=scale_b_bench_report, label="scale_b")
    _validate_provenance(
        provenance=scale_a_provenance, label="scale_a")
    _validate_provenance(
        provenance=scale_b_provenance, label="scale_b")
    slice_a = str(
        scale_a_provenance.get("slice_cid_bench_order") or "")
    slice_b = str(
        scale_b_provenance.get("slice_cid_bench_order") or "")
    if slice_a != slice_b or not slice_a:
        raise CrossScaleComparatorError(
            "cross-scale comparator requires byte-equal slices; "
            f"slice_cid_bench_order differs: A={slice_a!r} "
            f"B={slice_b!r}")
    corpus_a = str(scale_a_provenance.get("corpus_sha256") or "")
    corpus_b = str(scale_b_provenance.get("corpus_sha256") or "")
    if corpus_a != corpus_b or not corpus_a:
        raise CrossScaleComparatorError(
            "cross-scale comparator requires matching "
            f"corpus_sha256; got A={corpus_a!r} B={corpus_b!r}")
    seed_block_a = _extract_seed_block(scale_a_bench_report)
    seed_block_b = _extract_seed_block(scale_b_bench_report)
    bench_iter_a = list(
        scale_a_provenance.get("bench_iteration_task_ids") or [])
    bench_iter_b = list(
        scale_b_provenance.get("bench_iteration_task_ids") or [])
    if bench_iter_a != bench_iter_b or not bench_iter_a:
        raise CrossScaleComparatorError(
            "cross-scale comparator requires identical "
            "bench_iteration_task_ids; differs at position "
            "{}".format(
                next(
                    (i for i in range(min(len(bench_iter_a),
                                           len(bench_iter_b)))
                     if bench_iter_a[i] != bench_iter_b[i]),
                    -1)))
    n = int(len(bench_iter_a))
    if (len(seed_block_a.get("per_problem_a0_passed", [])) != n
            or len(seed_block_b.get(
                "per_problem_a0_passed", [])) != n):
        raise CrossScaleComparatorError(
            "cross-scale comparator: per_problem_*_passed length "
            f"!= n_problems ({n})")
    rows: list[CrossScaleProblemRow] = []
    shift_counts: dict[str, int] = {
        k: 0 for k in CROSS_SCALE_CLUSTER_SHIFT_VALUES}
    for i in range(n):
        a0_a = bool(seed_block_a["per_problem_a0_passed"][i])
        a1_a = bool(seed_block_a["per_problem_a1_passed"][i])
        b_a = bool(seed_block_a["per_problem_b_passed"][i])
        fpi_a = int(
            seed_block_a["per_problem_b_first_pass_idx"][i])
        a0_b = bool(seed_block_b["per_problem_a0_passed"][i])
        a1_b = bool(seed_block_b["per_problem_a1_passed"][i])
        b_b = bool(seed_block_b["per_problem_b_passed"][i])
        fpi_b = int(
            seed_block_b["per_problem_b_first_pass_idx"][i])
        shift = _per_problem_cluster_shift(
            a0_a=a0_a, a1_a=a1_a, b_a=b_a,
            a0_b=a0_b, a1_b=a1_b, b_b=b_b)
        shift_counts[shift] = int(shift_counts.get(shift, 0)) + 1
        rows.append(CrossScaleProblemRow(
            task_id=str(bench_iter_a[i]),
            a0_at_scale_a=a0_a, a1_at_scale_a=a1_a,
            b_at_scale_a=b_a,
            b_first_pass_idx_at_scale_a=fpi_a,
            a0_at_scale_b=a0_b, a1_at_scale_b=a1_b,
            b_at_scale_b=b_b,
            b_first_pass_idx_at_scale_b=fpi_b,
            cluster_shift=shift))
    a0_a_pct = float(
        scale_a_bench_report["a0_mean_pass_at_1"]) * 100
    a1_a_pct = float(
        scale_a_bench_report["a1_mean_pass_at_1"]) * 100
    b_a_pct = float(
        scale_a_bench_report["b_mean_pass_at_1"]) * 100
    a0_b_pct = float(
        scale_b_bench_report["a0_mean_pass_at_1"]) * 100
    a1_b_pct = float(
        scale_b_bench_report["a1_mean_pass_at_1"]) * 100
    b_b_pct = float(
        scale_b_bench_report["b_mean_pass_at_1"]) * 100
    mlb_a = scale_a_bench_report["mlb"]
    mlb_b = scale_b_bench_report["mlb"]
    mlb1_a = float(mlb_a["mlb1_invocation_rate"])
    mlb1_b = float(mlb_b["mlb1_invocation_rate"])
    mlb2_a = float(mlb_a["mlb2_rescue_rate"])
    mlb2_b = float(mlb_b["mlb2_rescue_rate"])
    return CrossScaleComparatorReportV1(
        schema=W104_CROSS_SCALE_COMPARATOR_V1_SCHEMA_VERSION,
        scale_a_model_id=str(
            scale_a_bench_report.get("model_id") or ""),
        scale_b_model_id=str(
            scale_b_bench_report.get("model_id") or ""),
        scale_a_bench_merkle=str(
            scale_a_bench_report.get("bench_merkle_root") or ""),
        scale_b_bench_merkle=str(
            scale_b_bench_report.get("bench_merkle_root") or ""),
        slice_cid=str(slice_a),
        corpus_sha256=str(corpus_a),
        n_problems=n,
        per_problem=tuple(rows),
        aggregate_arm_deltas_pp={
            "a0_pp": float(round(a0_b_pct - a0_a_pct, 4)),
            "a1_pp": float(round(a1_b_pct - a1_a_pct, 4)),
            "b_pp": float(round(b_b_pct - b_a_pct, 4)),
        },
        aggregate_cluster_shift_counts=dict(shift_counts),
        aggregate_b_minus_a1_pp_at_scale_a=float(
            round(b_a_pct - a1_a_pct, 4)),
        aggregate_b_minus_a1_pp_at_scale_b=float(
            round(b_b_pct - a1_b_pct, 4)),
        cross_scale_shift_on_b_minus_a1_pp=float(
            round((b_b_pct - a1_b_pct) - (b_a_pct - a1_a_pct),
                  4)),
        aggregate_mlb1_invocation_rate_at_scale_a=mlb1_a,
        aggregate_mlb1_invocation_rate_at_scale_b=mlb1_b,
        aggregate_mlb2_rescue_rate_at_scale_a=mlb2_a,
        aggregate_mlb2_rescue_rate_at_scale_b=mlb2_b,
        cross_scale_shift_on_mlb2_pp=float(
            round((mlb2_b - mlb2_a) * 100, 4)),
    )


def format_cross_scale_comparator_markdown_v1(
        *, report: CrossScaleComparatorReportV1) -> str:
    """Pretty-print the comparator report for the W104 doc."""
    out: list[str] = []
    d = report.to_dict()
    out.append(
        "| Field | Scale A | Scale B |\n"
        "|---|---|---|\n"
        f"| Model | `{d['scale_a_model_id']}` | "
        f"`{d['scale_b_model_id']}` |\n"
        f"| Bench Merkle | `{d['scale_a_bench_merkle'][:16]}...` | "
        f"`{d['scale_b_bench_merkle'][:16]}...` |\n"
        f"| MLB-1 invocation | "
        f"{d['aggregate_mlb1_invocation_rate_at_scale_a']*100:.2f}% | "
        f"{d['aggregate_mlb1_invocation_rate_at_scale_b']*100:.2f}% |\n"
        f"| MLB-2 rescue | "
        f"{d['aggregate_mlb2_rescue_rate_at_scale_a']*100:.2f}% | "
        f"{d['aggregate_mlb2_rescue_rate_at_scale_b']*100:.2f}% |\n"
        f"| B − A1 (pp) | "
        f"{d['aggregate_b_minus_a1_pp_at_scale_a']:+.2f} | "
        f"{d['aggregate_b_minus_a1_pp_at_scale_b']:+.2f} |\n"
        f"\n**Cross-scale shift on B − A1**: "
        f"{d['cross_scale_shift_on_b_minus_a1_pp']:+.2f} pp\n"
        f"**Cross-scale shift on MLB-2**: "
        f"{d['cross_scale_shift_on_mlb2_pp']:+.2f} pp\n")
    out.append("\n### Cluster-shift aggregate\n\n")
    out.append("| Shift | Count |\n|---|---:|\n")
    for k in CROSS_SCALE_CLUSTER_SHIFT_VALUES:
        out.append(
            f"| `{k}` | "
            f"{d['aggregate_cluster_shift_counts'].get(k, 0)} |\n")
    out.append("\n### Per-problem cross-scale rows\n\n")
    out.append(
        "| idx | task_id | A0 A→B | A1 A→B | B A→B | "
        "bidx A→B | shift |\n"
        "|---|---|---|---|---|---|---|\n")
    for i, row in enumerate(d["per_problem"]):
        out.append(
            f"| {i+1} | {row['task_id']} | "
            f"{int(row['a0_at_scale_a'])}→{int(row['a0_at_scale_b'])} | "
            f"{int(row['a1_at_scale_a'])}→{int(row['a1_at_scale_b'])} | "
            f"{int(row['b_at_scale_a'])}→{int(row['b_at_scale_b'])} | "
            f"{int(row['b_first_pass_idx_at_scale_a'])}→"
            f"{int(row['b_first_pass_idx_at_scale_b'])} | "
            f"`{row['cluster_shift']}` |\n")
    return "".join(out)


__all__ = (
    "W104_CROSS_SCALE_COMPARATOR_V1_SCHEMA_VERSION",
    "CROSS_SCALE_CLUSTER_SHIFT_VALUES",
    "CrossScaleComparatorError",
    "CrossScaleProblemRow",
    "CrossScaleComparatorReportV1",
    "build_cross_scale_comparator_report_v1",
    "format_cross_scale_comparator_markdown_v1",
)
