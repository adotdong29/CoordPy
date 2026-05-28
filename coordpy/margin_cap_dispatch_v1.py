"""W106 / COO-9 — margin-cap dispatch decision rule V1.

Consumes a W105 Phase 3 retirement verdict (the dict emitted by
``coordpy.phase3_retirement_evaluator_v1`` /
``phase3_retirement_verdict.json``) and decides, for a class that
landed ``FAIL_MARGIN``, whether a cheap multi-seed confirmation is
**honestly earned** — i.e. whether it could change the retirement
verdict — using a pre-committed TWO-GATE rule.

The module is pure-Python with no NIM dependency and no model
loading.  It is explicit-import only — it does NOT register itself
in ``coordpy/__init__.py``.  It is a SIBLING of
``coordpy.phase3_retirement_evaluator_v1`` and
``coordpy.cross_class_comparator_v1`` (same refuse-to-run error
shape; consumes their verdict output).

Two-gate decision rule (locked in ``docs/RUNBOOK_W106.md`` § 2)
--------------------------------------------------------------

GATE 1 — Entitlement (pre-committed W104/W105 Branch C table).
  Classify the failed class's W105 signature into one of four
  rows and read the entitled next step + NIM ceiling.  Only the
  "0 ≤ margin < +5 pp AND MLB-2 ≥ 33 % AND A1 < 90 %" row entitles
  a *cheap multi-seed confirmation*.

GATE 2 — Verdict-changing power (the W106 margin-cap discipline).
  Entitlement is necessary, not sufficient.  A cheap confirmation
  is run only if it can HONESTLY convert the fair-slice
  ``FAIL_MARGIN`` into ``RETIRED``.  Requires ALL THREE:
    2a — FAIR BATTLEFIELD: the proposed slice is representative,
         NOT rescue-concentrated (a rescue-concentrated slice
         inflates B − A1 by construction = the W102 upper-bound
         anti-pattern).
    2b — NO AUTHORITATIVE FAIR RESULT ALREADY EXISTS: there is no
         fair broad-slice multi-seed Phase 3 verdict for the class
         already.  (If one exists, a cheaper re-run cannot overturn
         it; retirement is defined on the fair broad slice.)
    2c — FIXABLE CONFOUND: the FAIL is plausibly a fixable confound
         (executor/parser bug, budget mismatch, slice mismatch,
         mechanism collapse) rather than a true clean magnitude
         miss.

Decision: GO iff GATE 1 entitled AND (2a ∧ 2b ∧ 2c).  Else NO-GO
→ accept the bounded single-class claim, register the margin-cap
close carry-forward, spend $0 NIM.

Refuses to run if (any of):

1. The verdict schema is not the recognised Phase 3 evaluator
   schema.
2. There is not exactly one ``FAIL_<reason>`` class (the dispatch
   targets a single failed class).
3. There is no ``RETIRED`` class (a bounded single-class claim
   requires a retired class to bound to).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any


W106_MARGIN_CAP_DISPATCH_V1_SCHEMA_VERSION: str = (
    "coordpy.margin_cap_dispatch_v1.v1")

# The Phase 3 evaluator verdict schema this dispatcher consumes.
_RECOGNISED_VERDICT_SCHEMAS: frozenset[str] = frozenset({
    "coordpy.phase3_retirement_evaluator_v1.v1",
})

# Locked thresholds (byte-identical to the W105 RUNBOOK bars).
W106_MARGIN_FLOOR_PP: float = 5.0
W106_MLB2_FLOOR: float = 0.33
W106_A1_SATURATION_MAX_PCT: float = 90.0

# GATE 1 entitled next-step labels.
ENTITLE_LIVECODEBENCH_PREFLIGHT: str = "livecodebench_preflight_nim_free"
ENTITLE_APPS_PREFLIGHT: str = "apps_preflight_nim_free"
ENTITLE_CHEAP_CONFIRMATION: str = "multi_seed_cheap_confirmation_at_class"
ENTITLE_CROSS_SCALE_COLLAPSE_AUDIT: str = (
    "cross_scale_collapse_audit_plus_retired_class_confirmation")

# Proposed confirmation slice types (GATE 2a input).
SLICE_RESCUE_CONCENTRATED: str = "rescue_concentrated"
SLICE_FAIR_BROAD: str = "fair_broad"


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


class MarginCapDispatchError(ValueError):
    """Raised when the dispatcher refuses to run because the
    inputs are structurally incompatible."""


@dataclasses.dataclass(frozen=True)
class FailedClassSignatureV1:
    """The W105 Phase 3 failure signature of a single class."""

    model_class_id: str
    verdict_label: str
    margin_pp: float
    mlb2_rescue_rate: float
    a1_max_pct: float
    per_seed_majority_count: int
    n_seeds: int
    executor_clean: bool
    audit_chain_passes: bool
    n_bars_passed_of_6: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_class_id": str(self.model_class_id),
            "verdict_label": str(self.verdict_label),
            "margin_pp": float(round(self.margin_pp, 4)),
            "mlb2_rescue_rate": float(round(self.mlb2_rescue_rate, 4)),
            "a1_max_pct": float(round(self.a1_max_pct, 4)),
            "per_seed_majority_count": int(self.per_seed_majority_count),
            "n_seeds": int(self.n_seeds),
            "executor_clean": bool(self.executor_clean),
            "audit_chain_passes": bool(self.audit_chain_passes),
            "n_bars_passed_of_6": int(self.n_bars_passed_of_6),
        }


@dataclasses.dataclass(frozen=True)
class MarginCapDispatchDecisionV1:
    """The end-to-end GO / NO-GO dispatch decision."""

    schema: str
    retired_class_id: str
    failed_class: FailedClassSignatureV1
    proposed_confirmation_slice_type: str
    # GATE 1
    gate1_entitled_next_step: str
    gate1_entitles_cheap_confirmation: bool
    gate1_nim_ceiling_calls: int
    # GATE 2
    gate2a_fair_battlefield: bool
    gate2b_no_authoritative_fair_result: bool
    gate2c_fixable_confound: bool
    gate2_verdict_changing_power: bool
    # Decision
    decision: str               # "GO" | "NO_GO"
    reason: str
    carry_forward_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "retired_class_id": str(self.retired_class_id),
            "failed_class": self.failed_class.to_dict(),
            "proposed_confirmation_slice_type": str(
                self.proposed_confirmation_slice_type),
            "gate1_entitled_next_step": str(
                self.gate1_entitled_next_step),
            "gate1_entitles_cheap_confirmation": bool(
                self.gate1_entitles_cheap_confirmation),
            "gate1_nim_ceiling_calls": int(
                self.gate1_nim_ceiling_calls),
            "gate2a_fair_battlefield": bool(
                self.gate2a_fair_battlefield),
            "gate2b_no_authoritative_fair_result": bool(
                self.gate2b_no_authoritative_fair_result),
            "gate2c_fixable_confound": bool(
                self.gate2c_fixable_confound),
            "gate2_verdict_changing_power": bool(
                self.gate2_verdict_changing_power),
            "decision": str(self.decision),
            "reason": str(self.reason),
            "carry_forward_id": str(self.carry_forward_id),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w106_margin_cap_dispatch_decision_v1",
            "decision": self.to_dict(),
        })


def classify_entitlement_v1(
        *, sig: FailedClassSignatureV1) -> tuple[str, bool, int]:
    """GATE 1 — map a failure signature to the entitled next step.

    Returns (entitled_next_step, entitles_cheap_confirmation,
    nim_ceiling_calls).
    """
    margin = float(sig.margin_pp)
    mlb2 = float(sig.mlb2_rescue_rate)
    a1_saturated = bool(
        sig.a1_max_pct >= W106_A1_SATURATION_MAX_PCT)
    mlb2_load_bearing = bool(mlb2 >= W106_MLB2_FLOOR)
    if margin < 0.0 and not mlb2_load_bearing:
        return (ENTITLE_LIVECODEBENCH_PREFLIGHT, False, 0)
    if a1_saturated:
        return (ENTITLE_APPS_PREFLIGHT, False, 0)
    if 0.0 <= margin < W106_MARGIN_FLOOR_PP and mlb2_load_bearing:
        return (ENTITLE_CHEAP_CONFIRMATION, True, 990)
    if margin < 0.0 and mlb2_load_bearing:
        return (ENTITLE_CROSS_SCALE_COLLAPSE_AUDIT, False, 3300)
    # margin >= floor would not be a FAIL_MARGIN; treat as no
    # entitled cheap confirmation.
    return ("no_dispatch_margin_at_or_above_floor", False, 0)


def evaluate_gate2_v1(
        *,
        sig: FailedClassSignatureV1,
        proposed_confirmation_slice_type: str,
        fair_broad_phase3_result_exists: bool,
) -> tuple[bool, bool, bool, bool]:
    """GATE 2 — verdict-changing power.

    Returns (gate2a, gate2b, gate2c, gate2_overall).
    """
    # 2a — the proposed slice must be a fair battlefield.
    gate2a = bool(
        str(proposed_confirmation_slice_type) == SLICE_FAIR_BROAD)
    # 2b — there must be NO authoritative fair broad-slice Phase 3
    # result already.
    gate2b = bool(not fair_broad_phase3_result_exists)
    # 2c — the FAIL must be a fixable confound, NOT a clean true
    # magnitude miss.  A clean magnitude miss = executor clean AND
    # audit passes AND per-seed-majority positive AND MLB-2 healthy.
    clean_magnitude_miss = bool(
        sig.executor_clean
        and sig.audit_chain_passes
        and sig.per_seed_majority_count >= max(
            (sig.n_seeds // 2) + 1, 1)
        and sig.mlb2_rescue_rate >= W106_MLB2_FLOOR)
    gate2c = bool(not clean_magnitude_miss)
    gate2_overall = bool(gate2a and gate2b and gate2c)
    return (gate2a, gate2b, gate2c, gate2_overall)


def _extract_failed_class_signature(
        *, per_class_verdict: dict[str, Any]) -> FailedClassSignatureV1:
    cells = per_class_verdict.get("cells") or []
    a1_max = max(
        (float(c.get("a1_pct", 0.0)) for c in cells),
        default=0.0)
    return FailedClassSignatureV1(
        model_class_id=str(
            per_class_verdict.get("model_class_id") or ""),
        verdict_label=str(
            per_class_verdict.get("verdict_label") or ""),
        margin_pp=float(
            per_class_verdict.get(
                "bar1_margin_mean_b_minus_a1_pp", 0.0)),
        mlb2_rescue_rate=float(
            per_class_verdict.get("mean_mlb2_rescue_rate", 0.0)),
        a1_max_pct=float(a1_max),
        per_seed_majority_count=int(
            per_class_verdict.get(
                "bar2_per_seed_majority_count", 0)),
        n_seeds=int(len(cells)),
        executor_clean=bool(
            per_class_verdict.get("bar6_executor_clean_passes",
                                  False)),
        audit_chain_passes=bool(
            per_class_verdict.get("bar5_audit_chain_passes",
                                  False)),
        n_bars_passed_of_6=int(
            per_class_verdict.get("n_bars_passed_of_6", 0)))


def build_margin_cap_dispatch_decision_v1(
        *,
        phase3_verdict: dict[str, Any],
        proposed_confirmation_slice_type: str = (
            SLICE_RESCUE_CONCENTRATED),
        fair_broad_phase3_result_exists: bool = True,
) -> MarginCapDispatchDecisionV1:
    """End-to-end margin-cap dispatch decision builder.

    ``phase3_verdict`` is the dict emitted by
    ``coordpy.phase3_retirement_evaluator_v1`` (or read from
    ``phase3_retirement_verdict.json``).  Defaults model the
    realized W105 Verdict-C/C1 case: the only confirmation form on
    offer is rescue-concentrated, and a fair broad-slice Phase 3
    verdict already exists (this very verdict).

    Refuses to run on schema mismatch / not-exactly-one-FAIL-class
    / no-RETIRED-class.
    """
    schema = str(phase3_verdict.get("schema") or "")
    if schema not in _RECOGNISED_VERDICT_SCHEMAS:
        raise MarginCapDispatchError(
            f"phase3_verdict schema {schema!r} not in recognised "
            f"set {sorted(_RECOGNISED_VERDICT_SCHEMAS)}")
    per_class = phase3_verdict.get("per_class") or []
    if not per_class:
        raise MarginCapDispatchError(
            "phase3_verdict has no per_class verdicts")
    retired = [
        c for c in per_class
        if str(c.get("verdict_label") or "") == "RETIRED"]
    failed = [
        c for c in per_class
        if str(c.get("verdict_label") or "").startswith("FAIL")]
    if len(failed) != 1:
        raise MarginCapDispatchError(
            "margin-cap dispatch targets exactly one FAIL_<reason> "
            f"class; found {len(failed)} "
            f"({[c.get('verdict_label') for c in per_class]})")
    if not retired:
        raise MarginCapDispatchError(
            "margin-cap dispatch requires a RETIRED class to bound "
            "the single-class claim to; found none "
            f"({[c.get('verdict_label') for c in per_class]})")
    sig = _extract_failed_class_signature(
        per_class_verdict=failed[0])
    retired_class_id = str(retired[0].get("model_class_id") or "")

    (entitled_step, entitles_cheap,
     nim_ceiling) = classify_entitlement_v1(sig=sig)
    (g2a, g2b, g2c, g2) = evaluate_gate2_v1(
        sig=sig,
        proposed_confirmation_slice_type=(
            proposed_confirmation_slice_type),
        fair_broad_phase3_result_exists=bool(
            fair_broad_phase3_result_exists))

    go = bool(entitles_cheap and g2)
    if go:
        decision = "GO"
        carry_forward_id = ""
        reason = (
            f"GATE 1 entitled ({entitled_step}; ~{nim_ceiling} NIM "
            "calls) AND GATE 2 verdict-changing-power PASS (fair "
            "battlefield, no authoritative fair result yet, fixable "
            "confound) — a cheap confirmation can honestly change "
            "the verdict.")
    else:
        decision = "NO_GO"
        carry_forward_id = (
            "W106-L-HUMANEVAL-PLUS-LLAMA31-70B-MARGIN-CAP-"
            "CHEAP-CONFIRMATION-NOT-EARNED-CAP")
        why: list[str] = []
        if not entitles_cheap:
            why.append(
                f"GATE 1 does not entitle a cheap confirmation "
                f"(routed to {entitled_step})")
        if entitles_cheap and not g2a:
            why.append(
                "GATE 2a FAIL (proposed slice is "
                f"{proposed_confirmation_slice_type!r}, not a fair "
                "battlefield — rescue-concentrated is an upper "
                "bound, the W102 anti-pattern)")
        if entitles_cheap and not g2b:
            why.append(
                "GATE 2b FAIL (an authoritative fair broad-slice "
                "multi-seed Phase 3 verdict already exists for this "
                "class — a cheaper re-run cannot overturn it)")
        if entitles_cheap and not g2c:
            why.append(
                "GATE 2c FAIL (clean true magnitude miss: executor "
                "clean, audit passes, per-seed majority positive, "
                "MLB-2 healthy — no confound to fix)")
        reason = (
            "; ".join(why)
            + " ⇒ accept the bounded single-class claim on "
            + retired_class_id + "; $0 NIM.")
    return MarginCapDispatchDecisionV1(
        schema=W106_MARGIN_CAP_DISPATCH_V1_SCHEMA_VERSION,
        retired_class_id=str(retired_class_id),
        failed_class=sig,
        proposed_confirmation_slice_type=str(
            proposed_confirmation_slice_type),
        gate1_entitled_next_step=str(entitled_step),
        gate1_entitles_cheap_confirmation=bool(entitles_cheap),
        gate1_nim_ceiling_calls=int(nim_ceiling),
        gate2a_fair_battlefield=bool(g2a),
        gate2b_no_authoritative_fair_result=bool(g2b),
        gate2c_fixable_confound=bool(g2c),
        gate2_verdict_changing_power=bool(g2),
        decision=str(decision),
        reason=str(reason),
        carry_forward_id=str(carry_forward_id))


def format_margin_cap_dispatch_markdown_v1(
        *, decision: MarginCapDispatchDecisionV1) -> str:
    """Pretty-print the dispatch decision for the W106 docs."""
    d = decision.to_dict()
    fc = d["failed_class"]
    out: list[str] = []
    out.append(
        f"## W106 margin-cap dispatch decision (schema "
        f"`{d['schema']}`)\n\n")
    out.append(
        f"* failed class: `{fc['model_class_id']}` "
        f"(`{fc['verdict_label']}`; {fc['n_bars_passed_of_6']}/6 "
        f"bars)\n"
        f"* retired class (bounded-claim anchor): "
        f"`{d['retired_class_id']}`\n"
        f"* failed-class signature: margin "
        f"{fc['margin_pp']:+.2f} pp; MLB-2 "
        f"{fc['mlb2_rescue_rate']*100:.2f} %; A1 max "
        f"{fc['a1_max_pct']:.2f} %; per-seed majority "
        f"{fc['per_seed_majority_count']}/{fc['n_seeds']}; "
        f"executor clean {fc['executor_clean']}; audit "
        f"{fc['audit_chain_passes']}\n\n")
    out.append("### GATE 1 — entitlement\n\n")
    out.append(
        f"* entitled next step: `{d['gate1_entitled_next_step']}`\n"
        f"* entitles a cheap confirmation: "
        f"**{'YES' if d['gate1_entitles_cheap_confirmation'] else 'NO'}** "
        f"(NIM ceiling ~{d['gate1_nim_ceiling_calls']} calls)\n\n")
    out.append("### GATE 2 — verdict-changing power\n\n")
    out.append(
        f"| sub-gate | PASS |\n|---|---|\n"
        f"| 2a fair battlefield (not rescue-concentrated) | "
        f"{'YES' if d['gate2a_fair_battlefield'] else 'NO'} |\n"
        f"| 2b no authoritative fair result already exists | "
        f"{'YES' if d['gate2b_no_authoritative_fair_result'] else 'NO'} |\n"
        f"| 2c fixable confound (not clean magnitude miss) | "
        f"{'YES' if d['gate2c_fixable_confound'] else 'NO'} |\n"
        f"| **GATE 2 overall** | "
        f"**{'YES' if d['gate2_verdict_changing_power'] else 'NO'}** |\n\n")
    out.append(
        f"### Decision — **{d['decision']}**\n\n{d['reason']}\n")
    if d["carry_forward_id"]:
        out.append(
            f"\nCarry-forward registered: "
            f"`{d['carry_forward_id']}`\n")
    return "".join(out)


__all__ = (
    "W106_MARGIN_CAP_DISPATCH_V1_SCHEMA_VERSION",
    "W106_MARGIN_FLOOR_PP",
    "W106_MLB2_FLOOR",
    "W106_A1_SATURATION_MAX_PCT",
    "ENTITLE_LIVECODEBENCH_PREFLIGHT",
    "ENTITLE_APPS_PREFLIGHT",
    "ENTITLE_CHEAP_CONFIRMATION",
    "ENTITLE_CROSS_SCALE_COLLAPSE_AUDIT",
    "SLICE_RESCUE_CONCENTRATED",
    "SLICE_FAIR_BROAD",
    "MarginCapDispatchError",
    "FailedClassSignatureV1",
    "MarginCapDispatchDecisionV1",
    "classify_entitlement_v1",
    "evaluate_gate2_v1",
    "build_margin_cap_dispatch_decision_v1",
    "format_margin_cap_dispatch_markdown_v1",
)
