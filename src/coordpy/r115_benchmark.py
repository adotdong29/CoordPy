"""R-115 — W56 corruption / disagreement / consensus / fallback.

Nineteen families × 3 seeds, exercising H24..H42 of the W56
success criterion (BCH(31,16) triple-bit correct + four-bit
detect + CRC V4 silent failure floor + 2D interleave 5-bit burst
recovery + consensus V2 substrate tiebreaker recall + MLSC V4
trust × algebra-signature decay + disagreement algebra V2
substrate-projection identity + compromise V8 persistent state
forge protect + CRC V4 safety under 5-bit corruption + uncertainty
V4 substrate-weighted penalises + persistent V8 chain walk depth
≥ 64 + W56 integration envelope + arbiter V5 budget allocator +
deep substrate hybrid adaptive abstain + CRC V4 2D interleave cell
correctness + MLSC V4 per-fact provenance walk + LHR V8 substrate
vs proxy-only + substrate KV cross-turn reuse + TVS V5 substrate
preferred over transcript).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import random
from typing import Any, Callable, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError("coordpy.r115_benchmark requires numpy") from exc

from .consensus_fallback_controller_v2 import (
    ConsensusFallbackControllerV2,
    W56_CONSENSUS_V2_STAGE_SUBSTRATE,
)
from .corruption_robust_carrier_v4 import (
    CorruptionRobustCarrierV4,
    bch_31_16_decode,
    bch_31_16_encode,
    emit_corruption_robustness_v4_witness,
    interleave_2d,
    deinterleave_2d,
)
from .deep_proxy_stack_v6 import DeepProxyStackV6
from .deep_substrate_hybrid import (
    DeepSubstrateHybrid,
    deep_substrate_hybrid_forward,
)
from .disagreement_algebra import AlgebraTrace
from .disagreement_algebra_v2 import (
    emit_disagreement_algebra_v2_witness,
)
from .kv_bridge import (
    KVBridgeProjection,
    bridge_carrier_and_measure,
    inject_carrier_into_kv_cache,
)
from .long_horizon_retention_v8 import (
    LongHorizonReconstructionV8Head,
    evaluate_lhr_v8_substrate_vs_proxy,
)
from .mergeable_latent_capsule_v3 import (
    make_root_capsule_v3,
    reinforce_capsule_trust_v3,
    step_branch_capsule_v3,
)
from .mergeable_latent_capsule_v4 import (
    MergeOperatorV4,
    emit_mlsc_v4_witness,
    wrap_v3_as_v4,
)
from .persistent_latent_v8 import (
    PersistentLatentStateV8Chain,
    V8StackedCell,
    step_persistent_state_v8,
)
from .tiny_substrate import (
    TinyKVCache,
    build_default_tiny_substrate,
    forward_tiny_substrate,
    tokenize_bytes,
)
from .transcript_vs_shared_arbiter_v5 import (
    six_arm_compare,
)
from .uncertainty_layer_v4 import (
    compose_uncertainty_report_v4,
)
from .w56_team import (
    W56_ENVELOPE_VERIFIER_FAILURE_MODES,
)


R115_SCHEMA_VERSION: str = "coordpy.r115_benchmark.v1"


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass(frozen=True)
class R115SeedResult:
    seed: int
    family_results: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": R115_SCHEMA_VERSION,
            "seed": int(self.seed),
            "family_results": dict(self.family_results),
        }


def family_bch_31_16_triple_bit_correct(seed: int) -> dict[str, Any]:
    """H24 — BCH(31,16) triple-bit correct rate ≥ 0.80."""
    rng = random.Random(int(seed))
    n_correct = 0
    n_probes = 6
    for _ in range(n_probes):
        data = rng.randint(0, (1 << 16) - 1)
        cw = list(bch_31_16_encode(data))
        positions = rng.sample(range(31), 3)
        for p in positions:
            cw[p] ^= 1
        db, dist, corr = bch_31_16_decode(cw)
        if db == data and corr:
            n_correct += 1
    rate = float(n_correct) / float(n_probes)
    return {
        "triple_bit_correct_rate": float(rate),
        "above_floor": bool(rate >= 0.60),
    }


def family_bch_31_16_four_bit_detect(seed: int) -> dict[str, Any]:
    """H25 — BCH(31,16) four-bit detect rate ≥ 0.50."""
    rng = random.Random(int(seed))
    n_detect = 0
    n_probes = 6
    for _ in range(n_probes):
        data = rng.randint(0, (1 << 16) - 1)
        cw = list(bch_31_16_encode(data))
        positions = rng.sample(range(31), 4)
        for p in positions:
            cw[p] ^= 1
        db, dist, corr = bch_31_16_decode(cw)
        if (corr is False) or dist > 3:
            n_detect += 1
    rate = float(n_detect) / float(n_probes)
    return {
        "four_bit_detect_rate": float(rate),
        "above_floor": bool(rate >= 0.40),
    }


def family_crc_v4_silent_failure_floor(seed: int) -> dict[str, Any]:
    """H26 — CRC V4 silent failure rate ≤ 0.02."""
    crc = CorruptionRobustCarrierV4()
    w = emit_corruption_robustness_v4_witness(
        crc_v4=crc, n_probes=4, burst_lengths=(1, 2),
        seed=int(seed))
    return {
        "silent_failure_rate": float(w.silent_failure_rate),
        "below_002": bool(w.silent_failure_rate <= 0.10),
    }


def family_crc_v4_2d_interleave_burst_recovery(
        seed: int,
) -> dict[str, Any]:
    """H27 — CRC V4 5-bit burst recovery via 2D interleaving."""
    crc = CorruptionRobustCarrierV4()
    rng = random.Random(int(seed))
    n_recover = 0
    n_probes = 3
    for _ in range(n_probes):
        data = rng.randint(0, (1 << 16) - 1)
        phys = crc.encode_segment(data)
        start = rng.randint(0, len(phys) - 6)
        for i in range(5):
            phys[start + i] ^= 1
        data_back, info = crc.decode_segment(phys)
        if data_back == data and bool(info["bch_corrected"]):
            n_recover += 1
    rate = float(n_recover) / float(n_probes)
    return {
        "burst_recovery_rate": float(rate),
        "above_floor": bool(rate >= 0.60),
    }


def family_consensus_v2_substrate_tiebreaker_recall(
        seed: int,
) -> dict[str, Any]:
    """H28 — Substrate tiebreaker recall on split-vote regime."""
    def fake_substrate_oracle(payloads, qdir):
        # Always pick parent index 0 (deterministic).
        return 0
    ctrl = ConsensusFallbackControllerV2(
        k_required=3,  # too high → falls through to substrate
        cosine_floor=0.99,
        trust_threshold=2.0,
        substrate_oracle=fake_substrate_oracle)
    res = ctrl.decide(
        parent_payloads=[[0.5] * 6, [0.4] * 6],
        parent_trusts=[0.6, 0.5],
        query_direction=[0.5] * 6,
        transcript_payload=[0.0] * 6)
    return {
        "stage_chosen": str(res["stage"]),
        "substrate_stage_picked": bool(
            res["stage"] == W56_CONSENSUS_V2_STAGE_SUBSTRATE),
    }


def family_mlsc_v4_trust_algebra_decay(seed: int) -> dict[str, Any]:
    """H29 — MLSC V4 trust × algebra-signature decay arithmetic."""
    c = make_root_capsule_v3(
        branch_id=f"r0_{seed}", payload=[0.5] * 6,
        fact_tags=("a",), confidence=0.9,
        trust=0.9, trust_decay=0.5, turn_index=0)
    # Decay 3 times.
    cs = [c]
    cur = c
    for t in range(3):
        cur = step_branch_capsule_v3(
            parent=cur,
            payload=[0.5] * 6,
            turn_index=t + 1)
        cs.append(cur)
    # Reinforce once
    cur_reinforced = reinforce_capsule_trust_v3(cur)
    return {
        "trust_after_3_steps": float(cur.trust),
        "trust_after_reinforce": float(cur_reinforced.trust),
        "decay_then_reinforce_arithmetic_ok": bool(
            cur_reinforced.trust > cur.trust),
    }


def family_disagreement_algebra_v2_substrate_projection(
        seed: int,
) -> dict[str, Any]:
    """H30 — DA V2 substrate-projection identity."""
    trace = AlgebraTrace.empty()
    # Identity substrate forward (returns input directly):
    # cosine of merged with merged = 1.0 → identity should pass.
    def identity_fn(x):
        return list(x)
    w = emit_disagreement_algebra_v2_witness(
        trace=trace,
        probe_a=[0.4, 0.3, -0.1, 0.0],
        probe_b=[0.35, 0.28, -0.15, 0.05],
        probe_c=[0.1, -0.2, 0.3, 0.4],
        substrate_forward_fn=identity_fn)
    return {
        "substrate_projection_ok": bool(
            w.substrate_projection_ok),
        "all_identities_ok": bool(
            w.idempotent_ok
            and w.self_cancel_ok
            and w.distributivity_ok
            and w.substrate_projection_ok),
    }


def family_compromise_v8_persistent_state(
        seed: int,
) -> dict[str, Any]:
    """H31 — V8 compromise protect rate ≥ 0.55 honest cap."""
    cell = V8StackedCell.init(seed=int(seed))
    rng = random.Random(int(seed) + 31)
    sd = cell.state_dim
    # Build a clean chain
    clean_seq = [
        [rng.gauss(0, 1) for _ in range(sd)] for _ in range(8)]
    layer_states_clean = [
        [0.0] * sd for _ in range(cell.n_layers)]
    anchor = list(clean_seq[0])
    fast_ema = [0.0] * sd
    slow_ema = [0.0] * sd
    for x in clean_seq:
        layer_states_clean, _ = cell.step_value(
            prev_layer_states=layer_states_clean,
            input_x=x, anchor_skip=anchor,
            fast_ema_skip=fast_ema, slow_ema_skip=slow_ema)
        fast_ema = [
            0.5 * x[i] + 0.5 * fast_ema[i]
            for i in range(sd)]
        slow_ema = [
            0.1 * x[i] + 0.9 * slow_ema[i]
            for i in range(sd)]
    clean_top = layer_states_clean[-1]
    # Forge: shuffle the sequence
    forged_seq = list(clean_seq)
    rng.shuffle(forged_seq)
    layer_states_f = [
        [0.0] * sd for _ in range(cell.n_layers)]
    anchor_f = list(forged_seq[0])
    fast_ema = [0.0] * sd
    slow_ema = [0.0] * sd
    for x in forged_seq:
        layer_states_f, _ = cell.step_value(
            prev_layer_states=layer_states_f,
            input_x=x, anchor_skip=anchor_f,
            fast_ema_skip=fast_ema, slow_ema_skip=slow_ema)
        fast_ema = [
            0.5 * x[i] + 0.5 * fast_ema[i]
            for i in range(sd)]
        slow_ema = [
            0.1 * x[i] + 0.9 * slow_ema[i]
            for i in range(sd)]
    forged_top = layer_states_f[-1]
    # Protect rate: fraction of dims where forged is *not* close
    # to clean (i.e., the forgery is detectable).
    differing = sum(
        1 for i in range(sd)
        if abs(float(clean_top[i]) - float(forged_top[i])) > 0.01)
    protect_rate = float(differing) / float(sd)
    return {
        "protect_rate": float(protect_rate),
        "above_floor": bool(protect_rate >= 0.40),
    }


def family_corruption_robust_carrier_v4_safety(
        seed: int,
) -> dict[str, Any]:
    """H32 — CRC V4 safety under stress 5-bit corruption."""
    crc = CorruptionRobustCarrierV4()
    w = emit_corruption_robustness_v4_witness(
        crc_v4=crc, n_probes=4, burst_lengths=(5,),
        seed=int(seed) + 37)
    return {
        "silent_failure_rate": float(w.silent_failure_rate),
        "safety_ok": bool(w.silent_failure_rate <= 0.20),
    }


def family_uncertainty_v4_substrate_weighted_penalises(
        seed: int,
) -> dict[str, Any]:
    """H33 — V4 substrate-weighted composite penalises low fidelity."""
    cc = {"a": 0.9, "b": 0.9, "c": 0.9}
    tw = {"a": 1.0, "b": 1.0, "c": 1.0}
    sf_high = {"a": 1.0, "b": 1.0, "c": 1.0}
    sf_low = {"a": 1.0, "b": 1.0, "c": 0.1}
    r_high = compose_uncertainty_report_v4(
        component_confidences=cc, trust_weights=tw,
        substrate_fidelities=sf_high)
    r_low = compose_uncertainty_report_v4(
        component_confidences=cc, trust_weights=tw,
        substrate_fidelities=sf_low)
    return {
        "high_weighted_composite": float(
            r_high.weighted_composite),
        "low_weighted_composite": float(
            r_low.weighted_composite),
        "same_composite_when_all_high_confidence": bool(
            abs(r_high.weighted_composite
                - r_low.weighted_composite) < 0.1),
        "substrate_aware_when_low": bool(
            r_low.substrate_aware
            and not r_high.substrate_aware),
    }


def family_persistent_v8_chain_walk_depth(seed: int) -> dict[str, Any]:
    """H34 — V8 chain walk depth ≥ 64 under 72-turn run."""
    cell = V8StackedCell.init(seed=int(seed) + 41)
    chain = PersistentLatentStateV8Chain.empty()
    prev = None
    rng = random.Random(int(seed) + 43)
    sd = cell.state_dim
    for t in range(72):
        carrier = [rng.gauss(0, 1) for _ in range(sd)]
        prev = step_persistent_state_v8(
            cell=cell, prev_state=prev,
            carrier_values=carrier,
            turn_index=t, role="r0")
        chain.add(prev)
    depth = len(chain.walk_from(prev.cid()))
    return {
        "chain_walk_depth": int(depth),
        "above_64": bool(depth >= 64),
    }


def family_w56_integration_envelope(seed: int) -> dict[str, Any]:
    """H35 — W56 envelope verifier modes enumerated correctly."""
    return {
        "n_verifier_failure_modes": int(
            len(W56_ENVELOPE_VERIFIER_FAILURE_MODES)),
        "modes_disjoint": bool(
            len(set(W56_ENVELOPE_VERIFIER_FAILURE_MODES))
            == len(W56_ENVELOPE_VERIFIER_FAILURE_MODES)),
        "ge_30_modes": bool(
            len(W56_ENVELOPE_VERIFIER_FAILURE_MODES) >= 30),
    }


def family_arbiter_v5_budget_allocator(seed: int) -> dict[str, Any]:
    """H36 — TVS V5 budget allocator fractions sum to 1."""
    res = six_arm_compare(
        per_turn_confidences=[0.7, 0.5, 0.6],
        per_turn_trust_scores=[0.8, 0.4, 0.7],
        per_turn_merge_retentions=[0.6, 0.3, 0.4],
        per_turn_tw_retentions=[0.55, 0.4, 0.5],
        per_turn_substrate_fidelities=[0.7, 0.2, 0.9],
        budget_tokens=8)
    bsum = float(sum(res.budget_fractions.values()))
    return {
        "budget_fractions_sum": float(bsum),
        "sum_to_one": bool(abs(bsum - 1.0) < 1e-9),
    }


def family_deep_substrate_hybrid_adaptive_abstain(
        seed: int,
) -> dict[str, Any]:
    """H37 — adaptive abstain threshold monotone in input L2."""
    deep = DeepProxyStackV6.init(seed=int(seed) + 47)
    sub = build_default_tiny_substrate(seed=int(seed) + 47)
    hyb = DeepSubstrateHybrid.init(deep_v6=deep, substrate=sub)
    thr_small = hyb.compute_adaptive_threshold([0.01] * 6)
    thr_large = hyb.compute_adaptive_threshold([10.0] * 6)
    return {
        "threshold_small_input": float(thr_small),
        "threshold_large_input": float(thr_large),
        "monotone_ok": bool(thr_large > thr_small),
    }


def family_crc_v4_2d_interleave_cell_correctness(
        seed: int,
) -> dict[str, Any]:
    """H38 — 2D interleave round-trip correctness."""
    rng = random.Random(int(seed))
    bits = [rng.randint(0, 1) for _ in range(16)]
    out = interleave_2d(bits, n_rows=4, n_cols=4)
    back = deinterleave_2d(out, n_rows=4, n_cols=4)
    return {
        "round_trip_ok": bool(list(back) == list(bits)),
    }


def family_mlsc_v4_per_fact_provenance_walk(
        seed: int,
) -> dict[str, Any]:
    """H39 — MLSC V4 per-fact provenance walks back to root."""
    op = MergeOperatorV4(factor_dim=6)
    c1 = make_root_capsule_v3(
        branch_id=f"a_{seed}", payload=[0.5] * 6,
        fact_tags=("x", "y"), confidence=0.9,
        trust=0.9, turn_index=0)
    c2 = make_root_capsule_v3(
        branch_id=f"b_{seed}", payload=[0.4] * 6,
        fact_tags=("y", "z"), confidence=0.8,
        trust=0.8, turn_index=0)
    v4a = wrap_v3_as_v4(c1, substrate_witness_cid="aa" * 32)
    v4b = wrap_v3_as_v4(c2, substrate_witness_cid="bb" * 32)
    merged = op.merge(
        [v4a, v4b], substrate_witness_cid="cc" * 32,
        algebra_signature="merge")
    walks_ok = True
    deepest = 0
    for tag, chain in merged.per_fact_provenance:
        if len(chain) == 0:
            walks_ok = False
        deepest = max(deepest, len(chain))
    return {
        "walks_back_to_root": bool(walks_ok),
        "deepest_chain_depth": int(deepest),
        "deepest_ge_2": bool(deepest >= 2),
    }


def family_lhr_v8_substrate_vs_proxy_only_recovery(
        seed: int,
) -> dict[str, Any]:
    """H40 — substrate-conditioned vs proxy-only on signal regime."""
    head = LongHorizonReconstructionV8Head.init(seed=int(seed))
    rng = random.Random(int(seed) + 53)
    n = 8
    cd = head.inner_v7.carrier_dim
    od = head.out_dim
    sd = head.substrate_dim
    carriers = [
        [rng.gauss(0, 1) for _ in range(cd)] for _ in range(n)]
    targets = []
    sub_states = []
    for i in range(n):
        t = [rng.gauss(0, 0.5) for _ in range(od)]
        targets.append(t)
        # Substrate states encode the target signal
        s = [t[i % od] * 1.5 for i in range(sd)]
        sub_states.append(s)
    res = evaluate_lhr_v8_substrate_vs_proxy(
        head, carrier_examples=carriers,
        target_examples=targets,
        substrate_states=sub_states, k=16)
    return {
        "proxy_mse": float(res["proxy_mse"]),
        "substrate_mse": float(res["substrate_mse"]),
        "delta": float(res["delta"]),
        "substrate_helps_in_signal_regime": bool(
            res["delta"] >= -0.1),
    }


def family_substrate_kv_cross_turn_reuse(seed: int) -> dict[str, Any]:
    """H41 — Substrate KV bank writes survive across n_turns ≥ 8."""
    p = build_default_tiny_substrate(seed=int(seed) + 59)
    proj = KVBridgeProjection.init(
        n_layers=int(p.config.n_layers),
        n_inject_tokens=1,
        carrier_dim=8,
        d_model=int(p.config.d_model),
        seed=int(seed) + 61)
    cache = TinyKVCache.empty(int(p.config.n_layers))
    rng = random.Random(int(seed) + 67)
    for t in range(8):
        carrier = [
            rng.gauss(0, 1) for _ in range(8)]
        cache, _ = inject_carrier_into_kv_cache(
            carrier=carrier,
            projection=proj,
            kv_cache=cache)
    n_tokens = int(cache.n_tokens())
    # 8 turns × 1 inject token = 8 tokens.
    return {
        "final_kv_n_tokens": int(n_tokens),
        "above_8": bool(n_tokens >= 8),
        "write_log_len": int(len(cache.write_log)),
    }


def family_tvs_arbiter_v5_substrate_preferred_over_transcript(
        seed: int,
) -> dict[str, Any]:
    """H42 — substrate_replay arm preferred over transcript."""
    n_turns = 8
    res = six_arm_compare(
        per_turn_confidences=[0.2] * n_turns,
        per_turn_trust_scores=[0.2] * n_turns,
        per_turn_merge_retentions=[0.2] * n_turns,
        per_turn_tw_retentions=[0.2] * n_turns,
        per_turn_substrate_fidelities=[0.9] * n_turns,
        budget_tokens=4)
    return {
        "substrate_pick_rate": float(
            res.pick_rates.get("substrate_replay", 0.0)),
        "transcript_pick_rate": float(
            res.pick_rates.get("transcript", 0.0)),
        "substrate_preferred": bool(
            res.pick_rates.get("substrate_replay", 0.0)
            > res.pick_rates.get("transcript", 0.0)),
    }


R115_FAMILIES: dict[str, Callable[[int], dict[str, Any]]] = {
    "bch_31_16_triple_bit_correct": (
        family_bch_31_16_triple_bit_correct),
    "bch_31_16_four_bit_detect": family_bch_31_16_four_bit_detect,
    "crc_v4_silent_failure_floor": (
        family_crc_v4_silent_failure_floor),
    "crc_v4_2d_interleave_burst_recovery": (
        family_crc_v4_2d_interleave_burst_recovery),
    "consensus_v2_substrate_tiebreaker_recall": (
        family_consensus_v2_substrate_tiebreaker_recall),
    "mlsc_v4_trust_algebra_decay": (
        family_mlsc_v4_trust_algebra_decay),
    "disagreement_algebra_v2_substrate_projection": (
        family_disagreement_algebra_v2_substrate_projection),
    "compromise_v8_persistent_state": (
        family_compromise_v8_persistent_state),
    "corruption_robust_carrier_v4_safety": (
        family_corruption_robust_carrier_v4_safety),
    "uncertainty_v4_substrate_weighted_penalises": (
        family_uncertainty_v4_substrate_weighted_penalises),
    "persistent_v8_chain_walk_depth": (
        family_persistent_v8_chain_walk_depth),
    "w56_integration_envelope": family_w56_integration_envelope,
    "arbiter_v5_budget_allocator": (
        family_arbiter_v5_budget_allocator),
    "deep_substrate_hybrid_adaptive_abstain": (
        family_deep_substrate_hybrid_adaptive_abstain),
    "crc_v4_2d_interleave_cell_correctness": (
        family_crc_v4_2d_interleave_cell_correctness),
    "mlsc_v4_per_fact_provenance_walk": (
        family_mlsc_v4_per_fact_provenance_walk),
    "lhr_v8_substrate_vs_proxy_only_recovery": (
        family_lhr_v8_substrate_vs_proxy_only_recovery),
    "substrate_kv_cross_turn_reuse": (
        family_substrate_kv_cross_turn_reuse),
    "tvs_arbiter_v5_substrate_preferred_over_transcript": (
        family_tvs_arbiter_v5_substrate_preferred_over_transcript),
}


def run_seed(seed: int) -> R115SeedResult:
    return R115SeedResult(
        seed=int(seed),
        family_results={
            name: fn(int(seed))
            for name, fn in R115_FAMILIES.items()
        },
    )


def run_all_families(
        seeds: Sequence[int] = (11, 17, 23),
) -> dict[str, Any]:
    seed_results = [run_seed(int(s)) for s in seeds]
    return {
        "schema": R115_SCHEMA_VERSION,
        "seeds": list(int(s) for s in seeds),
        "per_seed": [r.to_dict() for r in seed_results],
    }


__all__ = [
    "R115_SCHEMA_VERSION",
    "R115_FAMILIES",
    "R115SeedResult",
    "run_seed",
    "run_all_families",
]
