"""W138 / COO-9 — band mechanism-bench slate + dispatch (Lane β).

The W138 same-budget arm slate (RUNBOOK_W138 §8).  Every arm K=5, attempt-0 = byte-identical standard
prompt, one model call/attempt, no early stop, graded on secret; the witness/feedback is owned-oracle
on FRESH probes, $0 NIM, OUTSIDE the K budget.  Each arm is scored in the "B" slot so
``arm − A1 ≡ B − A1`` via the verbatim W108 ``_evaluate_phase2_gates`` / ``_mlb_rates``.

Reuses the validated arm-runners VERBATIM:
  * A0/A1/B0  ← ``icpc_reflexion_bench_v1.run_icpc_reflexion_bench_v1``
  * C0 (complexity witness, COMPLEXITY cells)  ← ``exact_oracle_witness_v1.ARM_C2_COMPLEXITY``
  * N0 (counterexample witness, NON-complexity cells — the 2nd-mode arm W138 must vindicate)
        ← ``exact_oracle_witness_v1.ARM_C1_COUNTEREXAMPLE``
  * X1 (family-routed counterexample-else-complexity controller, the LEAD)
        ← ``exact_oracle_witness_v1.ARM_C3_CONTROLLER`` (the M1 controller that scored +33pp in W137)
  * X2 (relabeled-reflexion NEGATIVE CONTROL, $0 ≡ B0) — the structural fake-different test must
        classify it FAKE_DIFFERENT (``repaired_field_mechanism_bench_v1.fake_different_report_v1``).

Re-exports ``evaluate_gate_v1`` (the §7a/§7b gate) + ``fake_different_report_v1`` verbatim so the W138
driver composes exactly the W137-validated discipline.  Pure / deterministic / explicit-import only;
``coordpy/__init__.py`` untouched.
"""
from __future__ import annotations

import dataclasses
from typing import Any

from .exact_oracle_witness_v1 import (
    ARM_C1_COUNTEREXAMPLE, ARM_C2_COMPLEXITY, ARM_C3_CONTROLLER)
from .repaired_field_mechanism_bench_v1 import (  # re-export verbatim
    GateVerdictV1, evaluate_gate_v1, fake_different_report_v1)

BAND_MECHANISM_BENCH_V1_SCHEMA_VERSION: str = "coordpy.band_mechanism_bench_v1.v1"


@dataclasses.dataclass(frozen=True)
class BandArmSpecV1:
    arm_id: str
    label: str
    consumes: str
    hypothesis: str
    scored_on: str               # "all" | "complexity" | "noncomplexity"
    is_lead: bool = False
    is_negative_control: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"arm_id": self.arm_id, "label": self.label, "consumes": self.consumes,
                "hypothesis": self.hypothesis, "scored_on": self.scored_on,
                "is_lead": bool(self.is_lead),
                "is_negative_control": bool(self.is_negative_control)}


# The locked W138 arm slate (RUNBOOK_W138 §8).
BAND_ARM_SLATE_V1: tuple[BandArmSpecV1, ...] = (
    BandArmSpecV1("A0", "plain single-shot", "statement + public samples",
                  "baseline pass@1; no feedback", "all"),
    BandArmSpecV1("A1", "same-budget self-consistency", "statement + public samples (K i.i.d.)",
                  "headroom baseline; beatable in principle iff 0<A1<1 (arXiv:2203.11171)", "all"),
    BandArmSpecV1("B0", "blind reflexion", "judge-reject bit + stderr + public-sample results",
                  "2nd baseline; excludes 'any feedback helps'", "all"),
    BandArmSpecV1("C0", "exact-oracle COMPLEXITY witness", "owned-oracle timing curve on FRESH probes",
                  "W133 proved REAL+load-bearing on complexity; clean parser-neutral field", "complexity"),
    BandArmSpecV1("N0", "exact-oracle COUNTEREXAMPLE witness", "minimal counterexample on FRESH probes",
                  "the 2nd-mode arm: W133 EW1 +0 was on a bimodal field; tests a designed-fixable band",
                  "noncomplexity"),
    BandArmSpecV1("X1", "family-routed counterexample-else-complexity controller",
                  "owned-oracle counterexample OR timing curve, routed per candidate failure",
                  "the M1 controller (W137 +33pp); tests whether it SPANS modes/families", "all",
                  is_lead=True),
    BandArmSpecV1("X2", "relabeled-reflexion NEGATIVE CONTROL", "B0 signal + a structurally-empty label",
                  "must classify FAKE_DIFFERENT; proves the bench rewards real signal not decoration",
                  "all", is_negative_control=True),
)

# arm id -> (kind, exact-oracle arm constant).  "deployable" reserved for an oracle-free variant.
BAND_ARM_DISPATCH_V1: dict[str, tuple[str, str]] = {
    "C0": ("witness", ARM_C2_COMPLEXITY),
    "N0": ("witness", ARM_C1_COUNTEREXAMPLE),
    "X1": ("witness", ARM_C3_CONTROLLER),
}

# the modes counted as COMPLEXITY for scored_on routing (C0 scored here; N0 scored elsewhere)
COMPLEXITY_MODES: tuple[str, ...] = ("COMPLEXITY_BLIND",)


def arm_scored_on_problem_v1(arm_id: str, problem_mode: str) -> bool:
    """Whether an arm contributes a scored attempt on a problem of this mode (RUNBOOK_W138 §8).
    C0 is scored on COMPLEXITY cells only; N0 on NON-complexity cells only; A*/B0/X1 on all."""
    spec = next((a for a in BAND_ARM_SLATE_V1 if a.arm_id == arm_id), None)
    if spec is None or spec.scored_on == "all":
        return True
    is_cx = problem_mode in COMPLEXITY_MODES
    return is_cx if spec.scored_on == "complexity" else (not is_cx)


__all__ = [
    "BAND_MECHANISM_BENCH_V1_SCHEMA_VERSION", "BandArmSpecV1", "BAND_ARM_SLATE_V1",
    "BAND_ARM_DISPATCH_V1", "COMPLEXITY_MODES", "arm_scored_on_problem_v1",
    # re-exported verbatim from repaired_field_mechanism_bench_v1
    "GateVerdictV1", "evaluate_gate_v1", "fake_different_report_v1",
]
