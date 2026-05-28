"""W105 / COO-9 — Cross-class comparator V1 (per-seed-aligned).

Reads two SETS of Phase 3 cells (one per model class) and emits
a structured per-seed cross-class diff that catches class-A-vs-
class-B mix-ups at write time, not at post-hoc autopsy time.

This is the W105 fix for the W104 V1 cross-scale comparator
row-alignment failure mode: that comparator iterated
per-(problem-position) assuming both scales used the same bench
internal shuffle, but the W103 (seed 103 001) and W104 (seed
104 001) runs produced different per-seed shuffles and the
per-problem cluster-shift labels were arithmetically mis-aligned
(aggregate stats stayed correct).  W105 cross-class comparator
runs per-(matched seed) — the bench's internal shuffle is
identical on both sides of each pair.

The comparator is a pure-Python module with no NIM dependency
and no model loading.  It is explicit-import only — it does NOT
register itself in ``coordpy/__init__.py``.

Inputs (per class):

* A dict mapping each seed (one of the W105 retirement seeds
  ``105 001 / 105 002 / 105 003``) to a (bench_report_dict,
  provenance_dict) pair.

Refuses to run if (any of):

1. The two classes do not share the same set of seeds.
2. Any pair's slice pack CID does not match.
3. Any pair's corpus SHA does not match.
4. Any cell's bench report schema is unrecognised.
5. Any cell's per_seed_iteration_task_ids list differs between
   classes for the same seed (this is the W105 alignment check;
   the bench's seed-driven shuffle MUST be deterministic across
   model classes).

Honest scope (W105)
-------------------

* ``W105-L-CROSS-CLASS-COMPARATOR-V1-PER-SEED-ALIGN-CAP`` — the
  comparator iterates per-(matched seed) so each comparison
  pair has identical bench-internal shuffle order on both sides.
* ``W105-L-CROSS-CLASS-COMPARATOR-V1-SLICE-PACK-EQUAL-CAP`` —
  the comparator REQUIRES byte-equal slice pack CIDs across
  classes.  Comparing two classes on DIFFERENT slice packs is
  structurally invalid; the comparator refuses rather than
  silently emitting a misleading diff.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any


W105_CROSS_CLASS_COMPARATOR_V1_SCHEMA_VERSION: str = (
    "coordpy.cross_class_comparator_v1.v1")


_RECOGNISED_BENCH_REPORT_SCHEMAS: frozenset[str] = frozenset({
    "coordpy.humaneval_plus_reflexion_bench_v1.v1",
})


# Per-problem cross-class cluster shift names.
CROSS_CLASS_CLUSTER_SHIFT_VALUES: tuple[str, ...] = (
    "stayed",       # same (A0, A1, B) outcome at both classes
    "improved",     # strictly more passes at class B
    "regressed",    # strictly fewer passes at class B
    "flipped",      # ambiguous transition
)


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


class CrossClassComparatorError(ValueError):
    """Raised when the comparator refuses to run because the
    inputs are structurally incompatible."""


@dataclasses.dataclass(frozen=True)
class CrossClassProblemRow:
    """One per-problem cross-class row for a given seed pair."""

    seed: int
    task_id: str
    a0_at_class_a: bool
    a1_at_class_a: bool
    b_at_class_a: bool
    b_first_pass_idx_at_class_a: int
    a0_at_class_b: bool
    a1_at_class_b: bool
    b_at_class_b: bool
    b_first_pass_idx_at_class_b: int
    cluster_shift: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": int(self.seed),
            "task_id": str(self.task_id),
            "a0_at_class_a": bool(self.a0_at_class_a),
            "a1_at_class_a": bool(self.a1_at_class_a),
            "b_at_class_a": bool(self.b_at_class_a),
            "b_first_pass_idx_at_class_a": int(
                self.b_first_pass_idx_at_class_a),
            "a0_at_class_b": bool(self.a0_at_class_b),
            "a1_at_class_b": bool(self.a1_at_class_b),
            "b_at_class_b": bool(self.b_at_class_b),
            "b_first_pass_idx_at_class_b": int(
                self.b_first_pass_idx_at_class_b),
            "cluster_shift": str(self.cluster_shift),
        }


@dataclasses.dataclass(frozen=True)
class CrossClassSeedDeltaV1:
    """One per-(matched seed) cross-class summary."""

    seed: int
    n_problems: int
    class_a_a0_pct: float
    class_a_a1_pct: float
    class_a_b_pct: float
    class_a_b_minus_a1_pp: float
    class_a_mlb1: float
    class_a_mlb2: float
    class_b_a0_pct: float
    class_b_a1_pct: float
    class_b_b_pct: float
    class_b_b_minus_a1_pp: float
    class_b_mlb1: float
    class_b_mlb2: float
    cross_class_shift_on_b_minus_a1_pp: float
    cross_class_shift_on_mlb2_pp: float
    cluster_shift_counts: dict[str, int]
    per_problem: tuple[CrossClassProblemRow, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": int(self.seed),
            "n_problems": int(self.n_problems),
            "class_a_a0_pct": float(round(
                self.class_a_a0_pct, 4)),
            "class_a_a1_pct": float(round(
                self.class_a_a1_pct, 4)),
            "class_a_b_pct": float(round(self.class_a_b_pct, 4)),
            "class_a_b_minus_a1_pp": float(round(
                self.class_a_b_minus_a1_pp, 4)),
            "class_a_mlb1": float(round(self.class_a_mlb1, 4)),
            "class_a_mlb2": float(round(self.class_a_mlb2, 4)),
            "class_b_a0_pct": float(round(
                self.class_b_a0_pct, 4)),
            "class_b_a1_pct": float(round(
                self.class_b_a1_pct, 4)),
            "class_b_b_pct": float(round(self.class_b_b_pct, 4)),
            "class_b_b_minus_a1_pp": float(round(
                self.class_b_b_minus_a1_pp, 4)),
            "class_b_mlb1": float(round(self.class_b_mlb1, 4)),
            "class_b_mlb2": float(round(self.class_b_mlb2, 4)),
            "cross_class_shift_on_b_minus_a1_pp": float(round(
                self.cross_class_shift_on_b_minus_a1_pp, 4)),
            "cross_class_shift_on_mlb2_pp": float(round(
                self.cross_class_shift_on_mlb2_pp, 4)),
            "cluster_shift_counts": dict(
                self.cluster_shift_counts),
            "per_problem": [
                r.to_dict() for r in self.per_problem],
        }


@dataclasses.dataclass(frozen=True)
class CrossClassComparatorReportV1:
    """End-to-end cross-class comparator report."""

    schema: str
    class_a_model_id: str
    class_b_model_id: str
    slice_pack_cid: str
    corpus_sha256: str
    per_seed: tuple[CrossClassSeedDeltaV1, ...]
    aggregate_cross_class_shift_on_b_minus_a1_pp: float
    aggregate_cross_class_shift_on_mlb2_pp: float
    aggregate_cluster_shift_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "class_a_model_id": str(self.class_a_model_id),
            "class_b_model_id": str(self.class_b_model_id),
            "slice_pack_cid": str(self.slice_pack_cid),
            "corpus_sha256": str(self.corpus_sha256),
            "per_seed": [s.to_dict() for s in self.per_seed],
            "aggregate_cross_class_shift_on_b_minus_a1_pp": (
                float(round(
                    self.aggregate_cross_class_shift_on_b_minus_a1_pp,
                    4))),
            "aggregate_cross_class_shift_on_mlb2_pp": float(round(
                self.aggregate_cross_class_shift_on_mlb2_pp, 4)),
            "aggregate_cluster_shift_counts": dict(
                self.aggregate_cluster_shift_counts),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w105_cross_class_comparator_report_v1",
            "report": self.to_dict(),
        })


def _per_problem_cluster_shift(
        *,
        a0_a: bool, a1_a: bool, b_a: bool,
        a0_b: bool, a1_b: bool, b_b: bool) -> str:
    """Same classification rule as W104 cross-scale comparator
    V1.  Carried forward verbatim."""
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


def _validate_cell_pair(
        *, bench_report_a, prov_a, bench_report_b, prov_b,
        seed: int) -> None:
    for label, br in (
            ("class_a", bench_report_a),
            ("class_b", bench_report_b)):
        schema = str(br.get("schema") or "")
        if schema not in _RECOGNISED_BENCH_REPORT_SCHEMAS:
            raise CrossClassComparatorError(
                f"seed {seed} {label} bench report schema "
                f"{schema!r} not in recognised set "
                f"{sorted(_RECOGNISED_BENCH_REPORT_SCHEMAS)}")
        if not br.get("mlb"):
            raise CrossClassComparatorError(
                f"seed {seed} {label} bench report missing "
                "required `mlb` block")
    slice_a = str(prov_a.get("slice_pack_cid") or "")
    slice_b = str(prov_b.get("slice_pack_cid") or "")
    if slice_a != slice_b or not slice_a:
        raise CrossClassComparatorError(
            f"seed {seed} slice_pack_cid mismatch: "
            f"A={slice_a!r} B={slice_b!r}")
    corpus_a = str(prov_a.get("corpus_sha256") or "")
    corpus_b = str(prov_b.get("corpus_sha256") or "")
    if corpus_a != corpus_b or not corpus_a:
        raise CrossClassComparatorError(
            f"seed {seed} corpus_sha256 mismatch: "
            f"A={corpus_a!r} B={corpus_b!r}")
    iter_a = list(
        prov_a.get("per_seed_iteration_task_ids") or [])
    iter_b = list(
        prov_b.get("per_seed_iteration_task_ids") or [])
    if iter_a != iter_b or not iter_a:
        raise CrossClassComparatorError(
            f"seed {seed} per_seed_iteration_task_ids "
            "differs between classes; cross-class comparator "
            "REQUIRES bench-internal shuffle alignment")


def build_seed_delta_v1(
        *, seed: int,
        bench_report_a: dict[str, Any],
        prov_a: dict[str, Any],
        bench_report_b: dict[str, Any],
        prov_b: dict[str, Any]) -> CrossClassSeedDeltaV1:
    _validate_cell_pair(
        bench_report_a=bench_report_a, prov_a=prov_a,
        bench_report_b=bench_report_b, prov_b=prov_b,
        seed=int(seed))
    block_a = (bench_report_a.get("per_seed") or [{}])[0]
    block_b = (bench_report_b.get("per_seed") or [{}])[0]
    iter_task_ids = list(
        prov_a.get("per_seed_iteration_task_ids") or [])
    n = int(len(iter_task_ids))
    if n == 0:
        raise CrossClassComparatorError(
            f"seed {seed} per_seed_iteration_task_ids empty")
    if (len(block_a.get("per_problem_a0_passed", [])) != n
            or len(block_b.get("per_problem_a0_passed", [])) != n):
        raise CrossClassComparatorError(
            f"seed {seed} per_problem array length != "
            f"iteration n_problems ({n})")
    rows: list[CrossClassProblemRow] = []
    shift_counts: dict[str, int] = {
        k: 0 for k in CROSS_CLASS_CLUSTER_SHIFT_VALUES}
    for i in range(n):
        a0_a = bool(block_a["per_problem_a0_passed"][i])
        a1_a = bool(block_a["per_problem_a1_passed"][i])
        b_a = bool(block_a["per_problem_b_passed"][i])
        fpi_a = int(block_a["per_problem_b_first_pass_idx"][i])
        a0_b = bool(block_b["per_problem_a0_passed"][i])
        a1_b = bool(block_b["per_problem_a1_passed"][i])
        b_b = bool(block_b["per_problem_b_passed"][i])
        fpi_b = int(block_b["per_problem_b_first_pass_idx"][i])
        shift = _per_problem_cluster_shift(
            a0_a=a0_a, a1_a=a1_a, b_a=b_a,
            a0_b=a0_b, a1_b=a1_b, b_b=b_b)
        shift_counts[shift] = int(
            shift_counts.get(shift, 0)) + 1
        rows.append(CrossClassProblemRow(
            seed=int(seed),
            task_id=str(iter_task_ids[i]),
            a0_at_class_a=a0_a, a1_at_class_a=a1_a,
            b_at_class_a=b_a,
            b_first_pass_idx_at_class_a=fpi_a,
            a0_at_class_b=a0_b, a1_at_class_b=a1_b,
            b_at_class_b=b_b,
            b_first_pass_idx_at_class_b=fpi_b,
            cluster_shift=shift))
    a0_a_pct = float(
        bench_report_a["a0_mean_pass_at_1"]) * 100
    a1_a_pct = float(
        bench_report_a["a1_mean_pass_at_1"]) * 100
    b_a_pct = float(bench_report_a["b_mean_pass_at_1"]) * 100
    a0_b_pct = float(
        bench_report_b["a0_mean_pass_at_1"]) * 100
    a1_b_pct = float(
        bench_report_b["a1_mean_pass_at_1"]) * 100
    b_b_pct = float(bench_report_b["b_mean_pass_at_1"]) * 100
    mlb_a = bench_report_a["mlb"]
    mlb_b = bench_report_b["mlb"]
    return CrossClassSeedDeltaV1(
        seed=int(seed), n_problems=int(n),
        class_a_a0_pct=float(a0_a_pct),
        class_a_a1_pct=float(a1_a_pct),
        class_a_b_pct=float(b_a_pct),
        class_a_b_minus_a1_pp=float(b_a_pct - a1_a_pct),
        class_a_mlb1=float(mlb_a["mlb1_invocation_rate"]),
        class_a_mlb2=float(mlb_a["mlb2_rescue_rate"]),
        class_b_a0_pct=float(a0_b_pct),
        class_b_a1_pct=float(a1_b_pct),
        class_b_b_pct=float(b_b_pct),
        class_b_b_minus_a1_pp=float(b_b_pct - a1_b_pct),
        class_b_mlb1=float(mlb_b["mlb1_invocation_rate"]),
        class_b_mlb2=float(mlb_b["mlb2_rescue_rate"]),
        cross_class_shift_on_b_minus_a1_pp=float(
            (b_b_pct - a1_b_pct) - (b_a_pct - a1_a_pct)),
        cross_class_shift_on_mlb2_pp=float(
            (float(mlb_b["mlb2_rescue_rate"])
             - float(mlb_a["mlb2_rescue_rate"])) * 100),
        cluster_shift_counts=dict(shift_counts),
        per_problem=tuple(rows))


def build_cross_class_comparator_report_v1(
        *,
        class_a_id: str, class_b_id: str,
        class_a_by_seed: dict[int,
                              tuple[dict[str, Any],
                                    dict[str, Any]]],
        class_b_by_seed: dict[int,
                              tuple[dict[str, Any],
                                    dict[str, Any]]],
) -> CrossClassComparatorReportV1:
    """Build the per-seed-aligned cross-class comparator report.
    Refuses if seed sets differ or any pair fails the validator.
    """
    seeds_a = set(int(s) for s in class_a_by_seed.keys())
    seeds_b = set(int(s) for s in class_b_by_seed.keys())
    if seeds_a != seeds_b or not seeds_a:
        raise CrossClassComparatorError(
            f"class_a seeds {sorted(seeds_a)} != class_b seeds "
            f"{sorted(seeds_b)}")
    per_seed: list[CrossClassSeedDeltaV1] = []
    all_slice_packs: set[str] = set()
    all_corpus_shas: set[str] = set()
    aggregate_shift: dict[str, int] = {
        k: 0 for k in CROSS_CLASS_CLUSTER_SHIFT_VALUES}
    for seed in sorted(seeds_a):
        br_a, prov_a = class_a_by_seed[seed]
        br_b, prov_b = class_b_by_seed[seed]
        delta = build_seed_delta_v1(
            seed=int(seed),
            bench_report_a=br_a, prov_a=prov_a,
            bench_report_b=br_b, prov_b=prov_b)
        per_seed.append(delta)
        all_slice_packs.add(str(prov_a.get("slice_pack_cid") or ""))
        all_slice_packs.add(str(prov_b.get("slice_pack_cid") or ""))
        all_corpus_shas.add(str(prov_a.get("corpus_sha256") or ""))
        all_corpus_shas.add(str(prov_b.get("corpus_sha256") or ""))
        for k in CROSS_CLASS_CLUSTER_SHIFT_VALUES:
            aggregate_shift[k] = int(
                aggregate_shift.get(k, 0)) + int(
                delta.cluster_shift_counts.get(k, 0))
    if len(all_slice_packs) != 1:
        raise CrossClassComparatorError(
            f"slice_pack_cid spans multiple values: "
            f"{sorted(all_slice_packs)}")
    if len(all_corpus_shas) != 1:
        raise CrossClassComparatorError(
            f"corpus_sha256 spans multiple values: "
            f"{sorted(all_corpus_shas)}")
    n = len(per_seed)
    agg_b_minus_a1_shift = sum(
        d.cross_class_shift_on_b_minus_a1_pp
        for d in per_seed) / float(n)
    agg_mlb2_shift = sum(
        d.cross_class_shift_on_mlb2_pp for d in per_seed) / (
            float(n))
    return CrossClassComparatorReportV1(
        schema=W105_CROSS_CLASS_COMPARATOR_V1_SCHEMA_VERSION,
        class_a_model_id=str(class_a_id),
        class_b_model_id=str(class_b_id),
        slice_pack_cid=str(next(iter(all_slice_packs))),
        corpus_sha256=str(next(iter(all_corpus_shas))),
        per_seed=tuple(per_seed),
        aggregate_cross_class_shift_on_b_minus_a1_pp=float(
            agg_b_minus_a1_shift),
        aggregate_cross_class_shift_on_mlb2_pp=float(
            agg_mlb2_shift),
        aggregate_cluster_shift_counts=dict(aggregate_shift))


def format_cross_class_comparator_markdown_v1(
        *, report: CrossClassComparatorReportV1) -> str:
    out: list[str] = []
    d = report.to_dict()
    out.append(
        f"## Cross-class comparator (schema `{d['schema']}`)\n\n"
        f"* class A: `{d['class_a_model_id']}`\n"
        f"* class B: `{d['class_b_model_id']}`\n"
        f"* slice_pack_cid: `{d['slice_pack_cid']}`\n"
        f"* corpus_sha256: `{d['corpus_sha256']}`\n\n"
        f"### Aggregate (mean across seeds)\n\n"
        f"* cross-class shift on B − A1: "
        f"{d['aggregate_cross_class_shift_on_b_minus_a1_pp']:+.2f} pp\n"
        f"* cross-class shift on MLB-2: "
        f"{d['aggregate_cross_class_shift_on_mlb2_pp']:+.2f} pp\n"
        f"* aggregate cluster shifts: "
        f"{json.dumps(d['aggregate_cluster_shift_counts'])}\n\n")
    out.append("### Per-seed deltas\n\n")
    out.append(
        "| seed | A B-A1 pp | B B-A1 pp | shift on B-A1 pp | "
        "A MLB-2 % | B MLB-2 % | shift on MLB-2 pp | stayed | "
        "improved | regressed | flipped |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for s in d["per_seed"]:
        cs = s["cluster_shift_counts"]
        out.append(
            f"| {s['seed']} | "
            f"{s['class_a_b_minus_a1_pp']:+.2f} | "
            f"{s['class_b_b_minus_a1_pp']:+.2f} | "
            f"{s['cross_class_shift_on_b_minus_a1_pp']:+.2f} | "
            f"{s['class_a_mlb2']*100:.2f} | "
            f"{s['class_b_mlb2']*100:.2f} | "
            f"{s['cross_class_shift_on_mlb2_pp']:+.2f} | "
            f"{cs.get('stayed', 0)} | "
            f"{cs.get('improved', 0)} | "
            f"{cs.get('regressed', 0)} | "
            f"{cs.get('flipped', 0)} |\n")
    return "".join(out)


__all__ = (
    "W105_CROSS_CLASS_COMPARATOR_V1_SCHEMA_VERSION",
    "CROSS_CLASS_CLUSTER_SHIFT_VALUES",
    "CrossClassComparatorError",
    "CrossClassProblemRow",
    "CrossClassSeedDeltaV1",
    "CrossClassComparatorReportV1",
    "build_seed_delta_v1",
    "build_cross_class_comparator_report_v1",
    "format_cross_class_comparator_markdown_v1",
)
