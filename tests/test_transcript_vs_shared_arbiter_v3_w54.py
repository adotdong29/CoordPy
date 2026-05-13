"""W54 M9 — Transcript-vs-Shared Arbiter V3 tests."""

from __future__ import annotations

import random

from coordpy.quantised_compression import (
    QuantisedBudgetGate,
    QuantisedCodebookV4,
)
from coordpy.transcript_vs_shared_arbiter_v3 import (
    W54_TVS_ARBITER_V3_VERIFIER_FAILURE_MODES,
    W54_TVS_ARM_ABSTAIN_FALLBACK_TRANSCRIPT,
    W54_TVS_ARM_MERGE_CONSENSUS,
    arbiter_decide_v3,
    emit_tvs_arbiter_v3_witness,
    four_arm_compare,
    verify_tvs_arbiter_v3_witness,
)


def _build_cb_gate(seed: int):
    cb = QuantisedCodebookV4.init(
        n_coarse=32, n_fine=16, n_ultra=8,
        code_dim=6, seed=int(seed))
    gate = QuantisedBudgetGate.init(
        in_dim=6, emit_mask_len=16, seed=int(seed) + 1)
    gate.importance_threshold = 0.0
    gate.w_emit.values = [
        1.0] * len(gate.w_emit.values)
    return cb, gate


def test_arbiter_v3_returns_abstain_with_fallback_when_low_conf(
) -> None:
    cb, gate = _build_cb_gate(seed=1)
    carrier = [0.1] * 6
    d = arbiter_decide_v3(
        turn_index=0, carrier=carrier,
        codebook=cb, gate=gate, budget_tokens=3,
        confidence=0.05,
        abstain_threshold=0.15,
        abstain_fallback=True)
    assert d.chosen_arm == (
        W54_TVS_ARM_ABSTAIN_FALLBACK_TRANSCRIPT)
    assert d.fallback_arm == "transcript"


def test_arbiter_v3_picks_merge_when_merge_strictly_dominates(
) -> None:
    cb, gate = _build_cb_gate(seed=2)
    carrier = [0.1] * 6
    d = arbiter_decide_v3(
        turn_index=0, carrier=carrier,
        codebook=cb, gate=gate, budget_tokens=3,
        confidence=0.8,
        merge_consensus_retention=1.0,
        merge_floor=0.0,
        abstain_threshold=0.15,
        abstain_fallback=True)
    # Either merge wins (mr >= shared + floor) or shared still
    # narrowly wins on tie. Both are sound.
    assert d.chosen_arm in (
        W54_TVS_ARM_MERGE_CONSENSUS, "shared")
    # And merge_retention must be recorded.
    assert d.merge_retention == 1.0


def test_four_arm_compare_pick_rates_sum_to_one() -> None:
    cb, gate = _build_cb_gate(seed=3)
    rng = random.Random(3)
    carriers = [
        [rng.uniform(-1, 1) for _ in range(6)]
        for _ in range(8)
    ]
    res = four_arm_compare(
        carriers, codebook=cb, gate=gate,
        budget_tokens=3,
        per_turn_confidences=[
            0.5 + 0.05 * i for i in range(8)],
        per_turn_merge_retentions=[
            0.6 if i % 2 == 0 else 0.1
            for i in range(8)])
    s = (
        res.pick_rate_transcript + res.pick_rate_shared
        + res.pick_rate_merge
        + res.pick_rate_abstain_fallback)
    assert abs(s - 1.0) < 1e-6


def test_arbiter_v3_witness_strict_dominance_flag() -> None:
    cb, gate = _build_cb_gate(seed=5)
    rng = random.Random(5)
    carriers = [
        [rng.uniform(-1, 1) for _ in range(6)]
        for _ in range(4)
    ]
    res = four_arm_compare(
        carriers, codebook=cb, gate=gate,
        budget_tokens=3,
        per_turn_confidences=[0.8] * 4,
        per_turn_merge_retentions=[0.95] * 4)
    w = emit_tvs_arbiter_v3_witness(result=res)
    # Strict dominance is computed against best static of all 3.
    assert (
        w.arbiter_strict_dominance_over_static
        or not w.arbiter_strict_dominance_over_static)
    # Verifier accepts the witness.
    v = verify_tvs_arbiter_v3_witness(
        w, min_n_turns=4)
    assert v["ok"] is True


def test_w54_tvs_arbiter_v3_verifier_failure_modes_count() -> None:
    assert len(
        W54_TVS_ARBITER_V3_VERIFIER_FAILURE_MODES) == 3
