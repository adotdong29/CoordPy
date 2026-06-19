"""W113 / COO-9 — cross-scale resistant-for-Llama-4 interpretation rule.

The W112 +10.00pp Llama-4-Maverick BigCodeBench result was on a benchmark
contamination-EXPOSED for Maverick (Aug-2024 cutoff > 2024-06 release).  W113
asks the CLEAN question that W112 could not: does the +10pp SURVIVE on a
benchmark VERIFIABLY contamination-RESISTANT for Maverick — the date-filtered
LiveCodeBench ``release_v6`` functional slice, every problem dated 2025-01..04,
months strictly after the August-2024 cutoff?

This completes a 2x2 (model scale x slice resistance) on the SAME mechanism:

                   |  70B (Llama-3.3)        |  Maverick (Llama-4)
    ---------------+-------------------------+--------------------------
    EXPOSED  (BCB) |  +0.00pp  (W110)        |  +10.00pp (W112)
    RESISTANT(LCB) |  -3.33pp  (W108, FAIL)  |  ???  (W113 measures this)

The W113 cell reuses the EXACT W108 30-slice (CID
``2afc318cb9a24d9a52b8914082cfbddaa8e941ef85d77be5382981621f43aa82``), so the
ONLY variable vs W108 is model scale (70B -> Maverick) on identical resistant
problems — exactly as W112 reused the exact W110 slice to isolate scale on
exposed BigCodeBench.

Fixed prior facts (NOT inputs the W113 pilot can move):
* ``MAVERICK_EXPOSED_BCB_PP = +10.00`` — W112 Maverick on EXPOSED BigCodeBench.
* ``LLAMA33_RESISTANT_LCB_PP = -3.33`` — W108 70B on the SAME resistant slice.

This module pre-commits — BEFORE the Maverick pilot lands — exactly how each
W113 outcome moves the resistant-superiority / contamination-confound claim, so
the interpretation is falsifiable and back-fit-proof.  It is the cross-SCALE
analogue of ``contamination_resistant_interpretation_v1`` (the W110 cross-
BENCHMARK rule) and reuses that module's canonical ``evaluate_phase2_gates_v1``
verdict so the pilot driver and the interpretation rule cannot drift.

Bar (RUNBOOK_W113 § 1α): a CLEAN resistant reopening requires
``PASS_MECHANISM_DRIVEN`` — a margin alone (``PASS_NON_MECHANISM_DRIVEN``, the
W112 exposed shape) is explicitly NOT enough, and a FAIL confirms the W112 +10pp
was exposure.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any

W113_CROSS_SCALE_INTERP_V1_SCHEMA_VERSION: str = (
    "coordpy.cross_scale_resistant_interpretation_v1.v1")

# Fixed prior facts — held constant; NOT inputs the W113 pilot can move.
MAVERICK_EXPOSED_BCB_PP: float = 10.00     # W112 Maverick, EXPOSED BigCodeBench
LLAMA33_RESISTANT_LCB_PP: float = -3.33    # W108 70B, the SAME resistant slice

# The W108 resistant LiveCodeBench 30-slice CID the W113 Maverick pilot reuses.
W108_RESISTANT_LCB_SLICE_CID: str = (
    "2afc318cb9a24d9a52b8914082cfbddaa8e941ef85d77be5382981621f43aa82")


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass(frozen=True)
class CrossScaleResistantInterpretationV1:
    schema: str
    model_id: str
    resistant_benchmark: str
    verdict_label: str
    b_minus_a1_pp: float
    mlb2_rescue_rate: float
    # Derived interpretation:
    outcome: str               # RESISTANT_SUPERIORITY_REOPENS | ...
    clean_resistant_reopening: bool
    confound_direction: str    # STRENGTHENS | UNCHANGED | (n/a if reopens)
    entitled_stronger_superiority_claim: bool
    boundary_after_w113: str
    w114_branch: str
    claim_implication: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "model_id": str(self.model_id),
            "resistant_benchmark": str(self.resistant_benchmark),
            "verdict_label": str(self.verdict_label),
            "b_minus_a1_pp": float(round(self.b_minus_a1_pp, 4)),
            "mlb2_rescue_rate": float(round(self.mlb2_rescue_rate, 6)),
            "outcome": str(self.outcome),
            "clean_resistant_reopening": bool(self.clean_resistant_reopening),
            "confound_direction": str(self.confound_direction),
            "entitled_stronger_superiority_claim": bool(
                self.entitled_stronger_superiority_claim),
            "boundary_after_w113": str(self.boundary_after_w113),
            "w114_branch": str(self.w114_branch),
            "claim_implication": str(self.claim_implication),
            "fixed_priors": {
                "maverick_exposed_bcb_pp": MAVERICK_EXPOSED_BCB_PP,
                "llama33_resistant_lcb_pp": LLAMA33_RESISTANT_LCB_PP,
                "w108_resistant_lcb_slice_cid": W108_RESISTANT_LCB_SLICE_CID,
            },
        }

    def cid(self) -> str:
        return _sha256_hex({"kind": "w113_cross_scale_resistant_interp_v1",
                            "decision": self.to_dict()})


def interpret_cross_scale_resistant_result_v1(
        *,
        model_id: str,
        resistant_benchmark: str,
        verdict_label: str,
        b_minus_a1_pp: float,
        mlb2_rescue_rate: float,
) -> CrossScaleResistantInterpretationV1:
    """Map the W113 Maverick-on-RESISTANT-LCB Phase-2 verdict to the pre-
    committed resistant-superiority / contamination-confound implication
    (RUNBOOK_W113 § 1α + § 8).  Falsifiable: the branch is selected purely by
    ``verdict_label`` (computed by the canonical ``evaluate_phase2_gates_v1``)."""
    v = str(verdict_label)
    if v == "PASS_MECHANISM_DRIVEN":
        outcome = "RESISTANT_SUPERIORITY_REOPENS"
        clean = True
        direction = "n/a (clean resistant WIN; confound no longer the frame)"
        entitled = True
        boundary = (
            "the same-budget reflexion advantage HOLDS on a benchmark verifiably "
            "RESISTANT for Llama-4-Maverick (LiveCodeBench 2025, all dates >> the "
            "Aug-2024 cutoff), WITH a load-bearing mechanism — the first CLEAN "
            "contamination-resistant same-budget multi-agent superiority signal "
            "in the programme, at the STRONGER scale")
        w114 = (
            "W114 = Maverick x resistant-LiveCodeBench Phase-3 retirement bench "
            "(3 seeds x ~100 x K=5) — the earned path to a contamination-"
            "RESISTANT retirement (a genuinely NEW frontier beyond the two "
            "exposed-HumanEval-family retirements)")
        implication = (
            "STRONG: scale GENUINELY reopens resistant superiority. The W112 "
            "+10pp was NOT mere exposure — it survives on resistant code at the "
            "same scale with a clean mechanism. This is bigger than the confound "
            "question: it is a clean resistant superiority signal. NOT itself a "
            "retirement (single-seed Phase-2); Phase-3 is W114. W89 + W105 STAND "
            "and a THIRD, contamination-resistant retirement is now in reach.")
    elif v == "PASS_NON_MECHANISM_DRIVEN":
        outcome = "RESISTANT_MARGIN_NON_MECHANISM"
        clean = False
        direction = "UNCHANGED"
        entitled = False
        boundary = (
            "a +>=5pp margin survives on resistant LiveCodeBench at Maverick "
            "scale but WITHOUT a load-bearing mechanism (MLB sub-gate fails, the "
            "W112 exposed shape) — NOT a CLEAN resistant reopening (the § 1α bar "
            "requires PASS_MECHANISM_DRIVEN)")
        w114 = (
            "W114 weighs a multi-seed resistant de-noise (does the margin hold "
            "and does the mechanism become load-bearing across seeds?) vs "
            "accepting the bounded claim; no Phase-3 entitlement from a single "
            "non-mechanism-driven seed")
        implication = (
            "AMBIGUOUS: the margin exists on resistant code but the mechanism is "
            "not cleanly load-bearing, so this does NOT clear the clean-resistant-"
            "reopening bar. The programme is NOT entitled to a stronger "
            "superiority claim; the contamination-confound is UNCHANGED (a margin "
            "without mechanism is the same shape as the exposed W112 result). "
            "Register the cap; the honest move is a multi-seed de-noise.")
    else:  # FAIL
        outcome = "EXPOSURE_CONFIRMED"
        clean = False
        direction = "STRENGTHENS"
        entitled = False
        boundary = (
            "Maverick COLLAPSES on resistant LiveCodeBench (B-A1 < +5pp / "
            "regression / failed core gate) just as 70B did (W108 -3.33pp) — the "
            "W112 +10pp on EXPOSED BigCodeBench is shown to be CONTAMINATION "
            "EXPOSURE. The bounded two-retirement contamination-EXPOSED-at-70B "
            "claim is HARDENED, now with a same-model exposed/resistant "
            "dissociation at the stronger scale")
        w114 = (
            "W114 = accept the bounded contamination-EXPOSED claim as the honest "
            "code ceiling and pursue a GENUINELY DIFFERENT axis (not another "
            "exposed rerun, not another resistant reflexion pilot at the same "
            "scale); or a tier-2 follow-up ONLY if a per-model-resistant slice is "
            "fetched + certified")
        implication = (
            "DECISIVE NEGATIVE: the W112 +10pp WAS exposure. Within the SAME "
            "model + SAME mechanism, the margin flips +10.00pp (exposed BCB) -> "
            "FAIL (resistant LCB) purely on slice resistance — the SHARPEST "
            "contamination dissociation in the programme and a direct "
            "confirmation of W112's model-cutoff-relativity finding at scale. "
            "Resistant superiority remains 0 clean demonstrations across BOTH "
            "scales. W89 + W105 STAND (still exactly two, both exposed-HumanEval-"
            "family). The contamination-confound STRENGTHENS (still not proof — "
            "single-seed cells; capability-scale not excluded as a co-driver).")
    return CrossScaleResistantInterpretationV1(
        schema=W113_CROSS_SCALE_INTERP_V1_SCHEMA_VERSION,
        model_id=str(model_id),
        resistant_benchmark=str(resistant_benchmark),
        verdict_label=v,
        b_minus_a1_pp=float(b_minus_a1_pp),
        mlb2_rescue_rate=float(mlb2_rescue_rate),
        outcome=str(outcome),
        clean_resistant_reopening=bool(clean),
        confound_direction=str(direction),
        entitled_stronger_superiority_claim=bool(entitled),
        boundary_after_w113=str(boundary),
        w114_branch=str(w114),
        claim_implication=str(implication))


__all__ = [
    "W113_CROSS_SCALE_INTERP_V1_SCHEMA_VERSION",
    "MAVERICK_EXPOSED_BCB_PP",
    "LLAMA33_RESISTANT_LCB_PP",
    "W108_RESISTANT_LCB_SLICE_CID",
    "CrossScaleResistantInterpretationV1",
    "interpret_cross_scale_resistant_result_v1",
]
