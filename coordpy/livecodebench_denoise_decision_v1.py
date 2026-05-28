"""W109-β — LiveCodeBench multi-seed de-noise decision rule (NIM-free).

W108 ran a SINGLE-seed LiveCodeBench functional Phase-2 cheap pilot and it
FAILed cleanly: B − A1 = −3.33 pp (a 1-problem effect at n=30), MLB-2 = 25 %
(4/16 rescued), 7/9 gates.  Lane β must decide — BEFORE any further LCB NIM
spend — whether a multi-seed de-noise of that 1-problem margin is honestly
warranted, or whether the single-seed FAIL already bounds the claim.

The discipline (carried over from the W106 ``margin_cap_dispatch_v1`` NO-GO):
**the only honest reason to spend more on a negative is verdict-changing
power, NOT discomfort with the negative.**  A multi-seed bench REDUCES the
variance of the mean estimate; it does NOT shift the mean.  So a multi-seed
de-noise can only plausibly flip a FAIL→PASS when the single-seed point is a
MARGINAL POSITIVE miss near the bar (the W105-Llama-3.1 shape: +2.33 pp, 5/6
bars, MLB-2 healthy) — NOT when the point is on the WRONG side of zero or the
mechanism is weakly load-bearing.

Two-gate rule (both must hold for WARRANTED):

* **Gate 1 — marginal POSITIVE miss**: 0 < (B − A1) < +5 pp.  The point is
  positive and below the bar, so de-noising could plausibly carry it over.
  A negative (B < A1) point cannot be de-noised into a ≥ +5 pp PASS — that
  needs a MEAN SHIFT, which more seeds do not provide.
* **Gate 2 — mechanism load-bearing**: MLB-2 ≥ 33 %.  If the rescue rate is
  below the floor, the mechanism is not causally responsible for wins on this
  benchmark family (the W102 MBPP+ shape), and a tighter mean cannot
  manufacture mechanism-driven superiority.

This is NOT the closed Llama-3.1 rescue-concentrated branch under a new name:
it forbids rescue-concentrated slices (Gate 1 keys on the FAIR broad-slice
margin; a de-noise re-runs the SAME outcome-blind slice across more seeds, not
a cherry-picked rescue slice) and it requires a POSITIVE mechanism signal.

Applied to W108 (B − A1 = −3.33 pp; MLB-2 = 25 %): Gate 1 FAILS (negative) AND
Gate 2 FAILS (weak) ⇒ **NOT WARRANTED**.  The single-seed FAIL bounds the
claim; the contamination-confound question is answered by the W109 APPS
CONTROL contrast (verdict-changing for the confound hypothesis), not by more
LCB seeds (verdict-changing for nothing).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any

W109_LCB_DENOISE_DECISION_V1_SCHEMA_VERSION: str = (
    "coordpy.livecodebench_denoise_decision_v1.v1")

MARGIN_BAR_PP: float = 5.0
MLB2_FLOOR: float = 0.33


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass(frozen=True)
class LcbPhase2ResultV1:
    """The fair broad-slice single-seed Phase-2 summary to be judged."""
    b_minus_a1_pp: float
    mlb2_rescue_rate: float
    n_seeds: int
    n_problems: int
    a1_pct: float


@dataclasses.dataclass(frozen=True)
class LcbDenoiseDecisionV1:
    schema: str
    warranted: bool
    gate1_marginal_positive_miss: bool
    gate2_mechanism_load_bearing: bool
    required_mean_shift_pp: float
    reasoning: tuple[str, ...]
    recommended_followup: dict | None
    w110_implication: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "warranted": bool(self.warranted),
            "gate1_marginal_positive_miss": bool(
                self.gate1_marginal_positive_miss),
            "gate2_mechanism_load_bearing": bool(
                self.gate2_mechanism_load_bearing),
            "required_mean_shift_pp": float(round(
                self.required_mean_shift_pp, 4)),
            "reasoning": list(self.reasoning),
            "recommended_followup": self.recommended_followup,
            "w110_implication": str(self.w110_implication),
        }

    def cid(self) -> str:
        return _sha256_hex({"kind": "w109_lcb_denoise_decision_v1",
                            "decision": self.to_dict()})


def decide_livecodebench_denoise_v1(
        result: LcbPhase2ResultV1,
        *, margin_bar_pp: float = MARGIN_BAR_PP,
        mlb2_floor: float = MLB2_FLOOR,
) -> LcbDenoiseDecisionV1:
    """Return the WARRANTED / NOT-WARRANTED de-noise decision for a
    single-seed LCB Phase-2 FAIL, with the two-gate reasoning."""
    margin = float(result.b_minus_a1_pp)
    mlb2 = float(result.mlb2_rescue_rate)
    gate1 = bool(0.0 < margin < float(margin_bar_pp))
    gate2 = bool(mlb2 >= float(mlb2_floor))
    warranted = bool(gate1 and gate2)
    required_shift = max(0.0, float(margin_bar_pp) - margin)

    reasoning: list[str] = []
    if margin <= 0.0:
        reasoning.append(
            f"Gate 1 FAIL: B − A1 = {margin:+.2f} pp is on the WRONG side of "
            "zero (B does not beat A1). A multi-seed de-noise reduces the "
            "VARIANCE of the mean; it does not SHIFT the mean. Carrying this "
            f"to the +{margin_bar_pp:.0f} pp bar needs a +{required_shift:.2f} "
            "pp mean shift, which more seeds cannot supply.")
    elif margin >= margin_bar_pp:
        reasoning.append(
            f"Gate 1 N/A: B − A1 = {margin:+.2f} pp already clears the bar.")
    else:
        reasoning.append(
            f"Gate 1 PASS: B − A1 = {margin:+.2f} pp is a marginal POSITIVE "
            f"miss (0 < margin < +{margin_bar_pp:.0f} pp); de-noising could "
            "plausibly carry it over the bar.")
    if not gate2:
        reasoning.append(
            f"Gate 2 FAIL: MLB-2 = {mlb2*100:.2f} % < {mlb2_floor*100:.0f} % "
            "floor — reflexion is weakly load-bearing on this benchmark "
            "family (the W102 MBPP+ shape). A tighter mean cannot manufacture "
            "mechanism-driven superiority that is not present.")
    else:
        reasoning.append(
            f"Gate 2 PASS: MLB-2 = {mlb2*100:.2f} % ≥ {mlb2_floor*100:.0f} % "
            "floor — the mechanism is load-bearing.")

    if warranted:
        followup = {
            "shape": "multi-seed LCB de-noise on the BYTE-EQUAL W108 30-slice",
            "model_class": "same (meta/llama-3.3-70b-instruct) — NOT a new "
                           "class; cross-class is the closed W106 branch",
            "slice": "the SAME outcome-blind difficulty-stratified 30-slice "
                     "(CID 2afc318c…); NOT a rescue-concentrated slice",
            "n_seeds": 3,
            "approx_nim_calls": int(3 * result.n_problems * (1 + 5 + 5)),
            "verdict_changing_power": (
                "could flip FAIL→PASS iff the true mean is ≥ +5 pp and the "
                "single-seed point under-sampled it"),
        }
        reasoning.append(
            "BOTH gates hold ⇒ a multi-seed de-noise is WARRANTED (the "
            "W105-Llama-3.1 marginal-miss shape).")
    else:
        followup = None
        reasoning.append(
            "At least one gate FAILS ⇒ a multi-seed LCB de-noise is NOT "
            "warranted. The single-seed FAIL already bounds the claim; "
            "spending more LCB NIM to de-noise a clean negative is the "
            "discomfort-driven anti-pattern. The contamination-confound "
            "question is verdict-changing only via the W109 APPS CONTROL "
            "contrast, not via more LCB seeds.")

    if warranted:
        w110 = ("W110 could run the warranted multi-seed LCB de-noise IF the "
                "W109 APPS control did not already resolve the confound "
                "question.")
    else:
        w110 = ("W110 is decided by the W109 APPS control outcome (Lane α), "
                "NOT by an LCB de-noise: APPS PASS ⇒ confound-consistent ⇒ a "
                "SECOND contamination-resistant benchmark; APPS FAIL ⇒ "
                "confound weakened ⇒ register the HumanEval-family-specific "
                "boundary. No further LCB spend either way.")

    return LcbDenoiseDecisionV1(
        schema=W109_LCB_DENOISE_DECISION_V1_SCHEMA_VERSION,
        warranted=warranted,
        gate1_marginal_positive_miss=gate1,
        gate2_mechanism_load_bearing=gate2,
        required_mean_shift_pp=float(required_shift),
        reasoning=tuple(reasoning),
        recommended_followup=followup,
        w110_implication=w110)


# The W108 LCB Phase-2 result, as published in
# docs/RESULTS_W108_LIVECODEBENCH_PHASE2_70B_V1.md.
W108_LCB_RESULT = LcbPhase2ResultV1(
    b_minus_a1_pp=-3.33, mlb2_rescue_rate=0.25, n_seeds=1, n_problems=30,
    a1_pct=63.33)


__all__ = [
    "W109_LCB_DENOISE_DECISION_V1_SCHEMA_VERSION",
    "MARGIN_BAR_PP",
    "MLB2_FLOOR",
    "LcbPhase2ResultV1",
    "LcbDenoiseDecisionV1",
    "decide_livecodebench_denoise_v1",
    "W108_LCB_RESULT",
]
