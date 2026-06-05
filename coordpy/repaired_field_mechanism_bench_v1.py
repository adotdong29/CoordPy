"""W137 / COO-9 — repaired-field mechanism-bench helpers (Lane β).

The Lane-β arms re-use the validated same-budget machinery verbatim — A0/A1/B0 from
``icpc_reflexion_bench_v1.run_icpc_reflexion_bench_v1``; C0 = the exact-oracle COMPLEXITY witness
(``exact_oracle_witness_v1`` arm ``C2``); M1 = the auto-routing counterexample-else-complexity
controller (arm ``C3``, the family-routed lead); M2 = the oracle-free deployable controller
(``deployable_complexity_witness_v1`` arm ``D3``).  This module adds only the genuinely-new W137
pieces:

* **M3 negative control** — a relabeled-reflexion fingerprint that the structural fake-different test
  must classify ``FAKE_DIFFERENT`` (so the discipline still bites on the repaired field); the real
  witness arms classify ``REAL``.  Empirically M3 ≡ B0 (blind reflexion) by construction, so it is
  scored at $0 (no separate NIM arm) — its job is to prove the bench is not rewarding decoration.
* **The §7a dev gate and §7b eval-earn rule** evaluators (RUNBOOK_W137 §7a/§7b), including the
  parsing/formatting-only exclusion (the W136 clause) and the ≥2-model-tier same-sign condition.

Reuses ``MechanismFingerprintV1`` from ``controller_native_code_mechanism_v1`` for the fake-different
classification.  Pure / deterministic / explicit-import only; ``coordpy/__init__.py`` untouched.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Optional, Sequence

from .controller_native_code_mechanism_v1 import (
    MechanismFingerprintV1, reflexion_b_fingerprint)

REPAIRED_FIELD_MECHANISM_BENCH_V1_SCHEMA_VERSION: str = (
    "coordpy.repaired_field_mechanism_bench_v1.v1")


@dataclasses.dataclass(frozen=True)
class ArmSpecV1:
    arm_id: str
    label: str
    consumes: str            # exactly which repaired-field signal it consumes
    survives_w136: str       # why it should survive the W136 confound correction
    is_negative_control: bool = False


# The locked arm slate (RUNBOOK_W137 §8).
ARM_SLATE_V1: tuple[ArmSpecV1, ...] = (
    ArmSpecV1("A0", "plain single-shot", "statement + public samples only",
              "baseline; no feedback"),
    ArmSpecV1("A1", "same-budget self-consistency", "statement + public samples (K i.i.d.)",
              "headroom baseline; no feedback"),
    ArmSpecV1("B0", "blind reflexion", "judge-reject bit + stderr + public-sample results",
              "the validated W120/W132 stack; parser-neutral field so the reject bit is honest"),
    ArmSpecV1("C0", "exact-oracle complexity witness (C2)",
              "owned-oracle timing curve on FRESH probes (no secret)",
              "W133 real+load-bearing on complexity; field is now parser-neutral so timing is the "
              "only signal, not an I/O artifact"),
    ArmSpecV1("M1", "family-routed counterexample-else-complexity controller (C3)",
              "owned-oracle minimal counterexample OR timing curve on FRESH probes, routed per "
              "candidate failure",
              "W136 REVISES W133-L/W135-L: the counterexample channel was tested on a confounded "
              "field; a clean field gives it a fair test"),
    ArmSpecV1("M2", "oracle-free deployable controller (D3)",
              "constraint-derived budget + public-format stress growth (NO oracle, NO secret)",
              "W134 deployable; oracle-free + parser-neutral ⇒ no parsing-only rescue possible"),
    ArmSpecV1("M3", "relabeled-reflexion NEGATIVE CONTROL", "B0 signal + a structurally-empty label",
              "must classify FAKE_DIFFERENT; proves the bench rewards real signal not decoration",
              is_negative_control=True),
)


# ===================================================== M3 fake-different discipline

def m3_relabeled_reflexion_fingerprint_v1() -> MechanismFingerprintV1:
    """The M3 negative control: blind reflexion with a cosmetic relabel — a single action type, no
    audited tool plane, a linear retry chain.  Must classify ``FAKE_DIFFERENT``."""
    return dataclasses.replace(reflexion_b_fingerprint(), name="M3_relabeled_reflexion")


def witness_arm_fingerprint_v1(arm_id: str) -> MechanismFingerprintV1:
    """Structural fingerprint of a real witness arm (C0/M1/M2): >=2 distinct action types, an
    audited owned-oracle plane, retry conditioned on the oracle digest, non-linear control flow."""
    return MechanismFingerprintV1(
        name=f"witness_{arm_id}", n_distinct_action_types=2, has_audited_tool_plane=True,
        retry_is_digest_conditioned=True, control_flow_is_linear_chain=False)


@dataclasses.dataclass(frozen=True)
class FakeDifferentReportV1:
    real_arms: tuple[str, ...]
    fake_arms: tuple[str, ...]
    bites: bool                      # the test classifies M3/B0 FAKE_DIFFERENT and >=1 arm REAL

    def to_dict(self) -> dict[str, Any]:
        return {"real_arms": list(self.real_arms), "fake_arms": list(self.fake_arms),
                "bites": bool(self.bites)}


def fake_different_report_v1(real_arm_ids: Sequence[str] = ("C0", "M1", "M2")
                             ) -> FakeDifferentReportV1:
    real, fake = [], []
    for a in real_arm_ids:
        (real if witness_arm_fingerprint_v1(a).classify() == "REAL" else fake).append(a)
    m3 = m3_relabeled_reflexion_fingerprint_v1().classify()
    b0 = reflexion_b_fingerprint().classify()
    if m3 == "FAKE_DIFFERENT":
        fake.append("M3")
    else:
        real.append("M3")
    if b0 == "FAKE_DIFFERENT":
        fake.append("B0")
    else:
        real.append("B0")
    bites = bool("M3" in fake and "B0" in fake and any(a in real for a in real_arm_ids))
    return FakeDifferentReportV1(real_arms=tuple(real), fake_arms=tuple(fake), bites=bites)


# ===================================================== §7a / §7b gate evaluators

def _rescues(per_lead: Sequence[bool], per_ref: Sequence[bool],
             modes: Sequence[str], families: Sequence[str]) -> tuple[list[int], set, set]:
    idx = [i for i in range(len(per_lead)) if per_lead[i] and not per_ref[i]]
    return idx, {modes[i] for i in idx}, {families[i] for i in idx}


@dataclasses.dataclass(frozen=True)
class GateVerdictV1:
    name: str                        # "dev_gate" | "eval_earn"
    lead_minus_a1_pp: float
    lead_minus_b0_pp: float
    n_rescues_vs_b0: int
    rescue_modes: tuple[str, ...]
    rescue_families: tuple[str, ...]
    span_ok: bool
    all_structural: bool
    no_net_regression: bool
    two_tier_same_sign: Optional[bool]
    passed: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "lead_minus_a1_pp": round(self.lead_minus_a1_pp, 2),
                "lead_minus_b0_pp": round(self.lead_minus_b0_pp, 2),
                "n_rescues_vs_b0": self.n_rescues_vs_b0,
                "rescue_modes": list(self.rescue_modes),
                "rescue_families": list(self.rescue_families), "span_ok": bool(self.span_ok),
                "all_structural": bool(self.all_structural),
                "no_net_regression": bool(self.no_net_regression),
                "two_tier_same_sign": self.two_tier_same_sign, "passed": bool(self.passed),
                "reason": self.reason}


def evaluate_gate_v1(*, name: str, per_lead: Sequence[bool], per_a1: Sequence[bool],
                     per_b0: Sequence[bool], modes: Sequence[str], families: Sequence[str],
                     rescue_is_structural: Sequence[bool], margin_pp: float,
                     two_tier_same_sign: Optional[bool] = None) -> GateVerdictV1:
    """RUNBOOK_W137 §7a (margin_pp=3.33) / §7b (margin_pp=5.00).  `rescue_is_structural[i]` is the
    arm's per-problem audit: True iff a rescue at i is algorithmic (NOT a parsing/formatting-only or
    complexity-only-single-family fix).  `two_tier_same_sign` (§7b only) is the cross-tier check."""
    n = float(len(per_lead)) or 1.0
    lead_acc = sum(per_lead) / n
    a1_acc = sum(per_a1) / n
    b0_acc = sum(per_b0) / n
    lead_m_a1 = (lead_acc - a1_acc) * 100.0
    lead_m_b0 = (lead_acc - b0_acc) * 100.0
    resc_idx, resc_modes, resc_fams = _rescues(per_lead, per_b0, modes, families)
    span_ok = bool(len(resc_modes) >= 2 or len(resc_fams) >= 3)
    all_structural = all(bool(rescue_is_structural[i]) for i in resc_idx) if resc_idx else False
    # no net regression: lead must not pass fewer than B0
    no_reg = bool(lead_acc >= b0_acc)
    margin_ok = bool(lead_m_a1 >= margin_pp and lead_m_b0 >= margin_pp)
    tier_ok = True if two_tier_same_sign is None else bool(two_tier_same_sign)
    passed = bool(margin_ok and span_ok and all_structural and no_reg and tier_ok)
    if passed:
        reason = "PASS"
    elif not margin_ok:
        reason = f"MARGIN_FAIL(lead-A1={lead_m_a1:+.2f},lead-B0={lead_m_b0:+.2f}<{margin_pp})"
    elif not span_ok:
        reason = f"SPAN_FAIL(modes={len(resc_modes)}<2 and families={len(resc_fams)}<3)"
    elif not all_structural:
        reason = "NON_STRUCTURAL_RESCUE(parsing/formatting/complexity-only excluded)"
    elif not no_reg:
        reason = "NET_REGRESSION_VS_B0"
    else:
        reason = "TWO_TIER_SAME_SIGN_FAIL"
    return GateVerdictV1(
        name=name, lead_minus_a1_pp=lead_m_a1, lead_minus_b0_pp=lead_m_b0,
        n_rescues_vs_b0=len(resc_idx), rescue_modes=tuple(sorted(resc_modes)),
        rescue_families=tuple(sorted(resc_fams)), span_ok=span_ok, all_structural=all_structural,
        no_net_regression=no_reg, two_tier_same_sign=two_tier_same_sign, passed=passed,
        reason=reason)


__all__ = [
    "REPAIRED_FIELD_MECHANISM_BENCH_V1_SCHEMA_VERSION", "ArmSpecV1", "ARM_SLATE_V1",
    "m3_relabeled_reflexion_fingerprint_v1", "witness_arm_fingerprint_v1",
    "FakeDifferentReportV1", "fake_different_report_v1",
    "GateVerdictV1", "evaluate_gate_v1",
]
