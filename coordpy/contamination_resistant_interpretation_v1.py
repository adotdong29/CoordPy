"""W110 / COO-9 — Lane β: contamination-resistant result interpretation rule.

The W110 verdict-changing question (``docs/RUNBOOK_W110.md`` § 6): the W89
mechanism FAILed on the FIRST contamination-RESISTANT benchmark (W108
LiveCodeBench 2025; B − A1 = −3.33 pp) and PASSed-on-margin on a
contamination-EXPOSED control (W109 APPS 2021; +16.67 pp,
PASS_NON_MECHANISM_DRIVEN). Is the W108 FAIL LiveCodeBench-SPECIFIC, or does the
mechanism fail GENERALLY on contamination-resistant code? W110 runs a SECOND,
genuinely-different contamination-resistant benchmark (BigCodeBench 2024). This
module pre-commits — BEFORE the pilot lands — exactly how each W110 outcome
moves the contamination-confound claim, so the interpretation is falsifiable
and back-fit-proof.

Two pure, deterministic functions (no NIM, no I/O):

* ``evaluate_phase2_gates_v1`` — the canonical 9 Phase-2 gates + MLB-1/MLB-2
  evaluator + verdict label, byte-identical to the rule the W103/W104/W105/
  W108/W109 pilot drivers applied inline. Single source of truth so the pilot
  driver and the interpretation rule cannot drift.
* ``interpret_second_resistant_result_v1`` — maps the verdict label + key
  metrics to the pre-committed contamination-confound implication (the Lane β
  claim-change branches).

This is the FAIL-side analogue of ``livecodebench_denoise_decision_v1`` (the
W109 Lane β rule): a falsifiable decision rule that keeps the programme from
back-fitting a comfortable story onto whatever number the pilot returns.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any

W110_CONTAMINATION_INTERP_V1_SCHEMA_VERSION: str = (
    "coordpy.contamination_resistant_interpretation_v1.v1")

PHASE2_MARGIN_BAR_PP: float = 5.0
PHASE2_A1_SATURATION: float = 0.90
PHASE2_MLB_FLOOR: float = 0.33

# Fixed prior facts (the W108/W109 pair) — held constant across the W110
# interpretation; they are NOT inputs the pilot can move.
W108_LCB_B_MINUS_A1_PP: float = -3.33
W109_APPS_B_MINUS_A1_PP: float = 16.67


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass(frozen=True)
class Phase2GateResultV1:
    schema: str
    gates: dict[str, bool]
    mlb1_invocation_rate: float
    mlb2_rescue_rate: float
    verdict_label: str  # PASS_MECHANISM_DRIVEN | PASS_NON_MECHANISM_DRIVEN | FAIL
    n_core_gates_pass: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "gates": {k: bool(v) for k, v in self.gates.items()},
            "mlb1_invocation_rate": float(round(self.mlb1_invocation_rate, 6)),
            "mlb2_rescue_rate": float(round(self.mlb2_rescue_rate, 6)),
            "verdict_label": str(self.verdict_label),
            "n_core_gates_pass": int(self.n_core_gates_pass),
        }

    def cid(self) -> str:
        return _sha256_hex({"kind": "w110_phase2_gate_result_v1",
                            "result": self.to_dict()})


def evaluate_phase2_gates_v1(
        *,
        n_problems: int,
        a0_pass_rate: float,
        a1_pass_rate: float,
        b_pass_rate: float,
        per_problem_b_not_worse_count: int,
        reflexion_invoked_count: int,
        reflexion_rescued_count: int,
        slice_pre_committed: bool,
        budget_byte_exact: bool,
        audit_chain_ok: bool,
        executor_clean: bool,
) -> Phase2GateResultV1:
    """Apply the 9 Phase-2 gates + MLB-1/MLB-2 (verbatim from W103→W109).

    * ``per_problem_b_not_worse_count`` — problems where B passed OR A1 also
      failed (B did not regress vs A1). Majority threshold = ``n//2 + 1``.
    * ``reflexion_invoked_count`` — problems where B's attempt-0 FAILED (so the
      reflexion loop was actually exercised). MLB-1 = invoked / n.
    * ``reflexion_rescued_count`` — of those invoked, problems B ultimately
      passed (a later attempt rescued it). MLB-2 = rescued / invoked.
    """
    n = max(1, int(n_problems))
    margin_pp = (float(b_pass_rate) - float(a1_pass_rate)) * 100.0
    vs_a0_pp = (float(b_pass_rate) - float(a0_pass_rate)) * 100.0
    majority_threshold = (n // 2) + 1
    invoked = int(reflexion_invoked_count)
    mlb1 = invoked / float(n)
    mlb2 = (int(reflexion_rescued_count) / float(invoked)) if invoked > 0 else 0.0
    gates = {
        "G1_slice_pre_committed": bool(slice_pre_committed),
        "G2_a1_below_saturation": bool(float(a1_pass_rate) < PHASE2_A1_SATURATION),
        "G3_b_gt_a1": bool(float(b_pass_rate) > float(a1_pass_rate)),
        "G4_margin_ge_5pp": bool(margin_pp >= PHASE2_MARGIN_BAR_PP),
        "G5_vs_a0_ge_5pp": bool(vs_a0_pp >= PHASE2_MARGIN_BAR_PP),
        "G6_per_problem_majority": bool(
            int(per_problem_b_not_worse_count) >= majority_threshold),
        "G7_budget_byte_exact": bool(budget_byte_exact),
        "G8_audit_chain_ok": bool(audit_chain_ok),
        "G9_executor_clean": bool(executor_clean),
    }
    n_core_pass = sum(1 for v in gates.values() if v)
    mlb1_pass = mlb1 >= PHASE2_MLB_FLOOR
    mlb2_pass = mlb2 >= PHASE2_MLB_FLOOR
    all_core = (n_core_pass == len(gates))
    if all_core and mlb1_pass and mlb2_pass:
        verdict = "PASS_MECHANISM_DRIVEN"
    elif all_core:
        verdict = "PASS_NON_MECHANISM_DRIVEN"
    else:
        verdict = "FAIL"
    return Phase2GateResultV1(
        schema=W110_CONTAMINATION_INTERP_V1_SCHEMA_VERSION,
        gates=gates, mlb1_invocation_rate=float(mlb1),
        mlb2_rescue_rate=float(mlb2), verdict_label=str(verdict),
        n_core_gates_pass=int(n_core_pass))


@dataclasses.dataclass(frozen=True)
class SecondResistantInterpretationV1:
    schema: str
    second_resistant_benchmark: str
    verdict_label: str
    b_minus_a1_pp: float
    mlb2_rescue_rate: float
    confound_direction: str  # STRENGTHENS | WEAKENS | UNCHANGED
    boundary_after_w110: str
    earns_phase3_retirement_bench: bool
    w111_branch: str
    claim_implication: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "second_resistant_benchmark": str(self.second_resistant_benchmark),
            "verdict_label": str(self.verdict_label),
            "b_minus_a1_pp": float(round(self.b_minus_a1_pp, 4)),
            "mlb2_rescue_rate": float(round(self.mlb2_rescue_rate, 6)),
            "confound_direction": str(self.confound_direction),
            "boundary_after_w110": str(self.boundary_after_w110),
            "earns_phase3_retirement_bench": bool(
                self.earns_phase3_retirement_bench),
            "w111_branch": str(self.w111_branch),
            "claim_implication": str(self.claim_implication),
        }

    def cid(self) -> str:
        return _sha256_hex({"kind": "w110_second_resistant_interpretation_v1",
                            "decision": self.to_dict()})


def interpret_second_resistant_result_v1(
        *,
        second_resistant_benchmark: str,
        verdict_label: str,
        b_minus_a1_pp: float,
        mlb2_rescue_rate: float,
) -> SecondResistantInterpretationV1:
    """Map the W110 second-resistant Phase-2 verdict to the pre-committed
    contamination-confound implication (RUNBOOK_W110 § 6). Falsifiable: the
    branch is selected purely by ``verdict_label`` (which is itself computed by
    ``evaluate_phase2_gates_v1``)."""
    v = str(verdict_label)
    if v == "PASS_MECHANISM_DRIVEN":
        direction = "WEAKENS"
        boundary = (
            "the W108 LiveCodeBench FAIL was LCB-SPECIFIC, not general to "
            "contamination-resistant code; a clean mechanism-driven same-budget "
            "win exists on contamination-RESISTANT BigCodeBench")
        earns = True
        w111 = (
            "W111 = BigCodeBench Phase-3 multi-seed retirement bench "
            "(3 seeds x 100 x K=5) — the earned path to a "
            "contamination-RESISTANT THIRD retirement")
        implication = (
            "Contamination-confound WEAKENS materially. The same-budget "
            "reflexion advantage replicates on a SECOND, genuinely-different "
            "contamination-resistant code benchmark, so the W108 FAIL is "
            "benchmark-idiosyncratic, not a contamination signal. NOT itself a "
            "retirement (Phase-3 is W111).")
    elif v == "PASS_NON_MECHANISM_DRIVEN":
        direction = "UNCHANGED"
        boundary = (
            "margin-without-clean-mechanism on the second resistant benchmark; "
            "the contamination-confound stays SUPPORTED-not-proven")
        earns = False
        w111 = (
            "W111 weighs a multi-seed BigCodeBench de-noise (low value) vs a "
            "THIRD genuinely-different resistant benchmark; no Phase-3 "
            "entitlement")
        implication = (
            "Weak/ambiguous: the margin exists but the mechanism is not "
            "cleanly load-bearing on invocation. Register the cap; the "
            "confound remains SUPPORTED-not-proven; does NOT earn a Phase-3 "
            "bench by itself.")
    else:  # FAIL
        direction = "STRENGTHENS"
        boundary = (
            "contamination-EXPOSED-specific at 70B: the mechanism now FAILs on "
            "TWO genuinely-different contamination-RESISTANT code benchmarks "
            "(LiveCodeBench 2025 + BigCodeBench 2024) while PASSing on "
            "contamination-EXPOSED HumanEval-family + APPS")
        earns = False
        w111 = (
            "W111 = register the tightened contamination-EXPOSED-specific "
            "boundary; honest next move is a DIFFERENT mechanism or acceptance "
            "of the bounded two-retirement contamination-exposed claim")
        implication = (
            "Contamination-confound STRENGTHENS toward a finding (still not "
            "proof — single-seed each; two resistant points). The W108 FAIL is "
            "shown GENERAL, not LCB-specific. COO-9's code-superiority "
            "GENERALISATION charter is substantially answered negatively for "
            "contamination-resistant code.")
    return SecondResistantInterpretationV1(
        schema=W110_CONTAMINATION_INTERP_V1_SCHEMA_VERSION,
        second_resistant_benchmark=str(second_resistant_benchmark),
        verdict_label=v, b_minus_a1_pp=float(b_minus_a1_pp),
        mlb2_rescue_rate=float(mlb2_rescue_rate),
        confound_direction=str(direction), boundary_after_w110=str(boundary),
        earns_phase3_retirement_bench=bool(earns), w111_branch=str(w111),
        claim_implication=str(implication))


__all__ = [
    "W110_CONTAMINATION_INTERP_V1_SCHEMA_VERSION",
    "PHASE2_MARGIN_BAR_PP",
    "PHASE2_A1_SATURATION",
    "PHASE2_MLB_FLOOR",
    "Phase2GateResultV1",
    "evaluate_phase2_gates_v1",
    "SecondResistantInterpretationV1",
    "interpret_second_resistant_result_v1",
]
