"""W61 R-130 benchmark family — corruption / consensus / uncertainty
/ disagreement / TVS arbiter.

H158..H162b cell families.

* H158   crc_v9 kv512 fingerprint single-byte detect rate ≥ 0.95
* H158b  crc_v9 13-bit adversarial burst detect rate ≥ 0.95
* H158c  crc_v9 post-replay top-K Jaccard floor ≥ 0.5
* H159   consensus_v7 11-stage chain enumerated
* H159b  consensus_v7 attention_pattern_consensus stage fires
* H160   uncertainty_v9 8-axis weighted composite computes
* H160b  uncertainty_v9 attention-pattern-aware on a non-trivial axis
* H161   disagreement_algebra_v7 attention_pattern_equiv identity
* H161b  disagreement_algebra_v7 falsifier triggers
* H162   tvs_v10 11-arm pick rates sum to 1
* H162b  tvs_v10 attention_pattern_steer arm fires when dominant
* H162c  tvs_v10 reduces to V9 when attention_pattern_fidelity is 0
"""

from __future__ import annotations

from typing import Any, Sequence

from .consensus_fallback_controller_v7 import (
    ConsensusFallbackControllerV7,
    W61_CONSENSUS_V7_STAGE_ATTENTION_PATTERN,
    W61_CONSENSUS_V7_STAGES,
    emit_consensus_v7_witness,
)
from .corruption_robust_carrier_v8 import (
    CorruptionRobustCarrierV8,
)
from .corruption_robust_carrier_v9 import (
    CorruptionRobustCarrierV9,
    emit_corruption_robustness_v9_witness,
    kv_cache_fingerprint_512,
    post_replay_topk_jaccard,
)
from .disagreement_algebra import AlgebraTrace
from .disagreement_algebra_v7 import (
    emit_disagreement_algebra_v7_witness,
)
from .transcript_vs_shared_arbiter_v10 import (
    W61_TVS_V10_ARMS,
    eleven_arm_compare,
    emit_tvs_arbiter_v10_witness,
)
from .uncertainty_layer_v9 import (
    compose_uncertainty_report_v9,
)


R130_SCHEMA_VERSION: str = "coordpy.r130_benchmark.v1"


def family_h158_crc_v9_kv512_detect(seed: int) -> dict[str, Any]:
    crc = CorruptionRobustCarrierV9(
        inner_v8=CorruptionRobustCarrierV8())
    w = emit_corruption_robustness_v9_witness(
        crc_v9=crc, n_probes=32,
        seed=int(seed) + 30200)
    return {
        "schema": R130_SCHEMA_VERSION,
        "name": "h158_crc_v9_kv512_detect",
        "passed": bool(
            float(w.kv512_corruption_detect_rate) >= 0.95),
        "rate": float(w.kv512_corruption_detect_rate),
    }


def family_h158b_crc_v9_13bit_burst(seed: int) -> dict[str, Any]:
    crc = CorruptionRobustCarrierV9(
        inner_v8=CorruptionRobustCarrierV8())
    w = emit_corruption_robustness_v9_witness(
        crc_v9=crc, n_probes=32,
        seed=int(seed) + 30300)
    return {
        "schema": R130_SCHEMA_VERSION,
        "name": "h158b_crc_v9_13bit_burst",
        "passed": bool(
            float(w.adversarial_13bit_burst_detect_rate)
            >= 0.95),
        "rate": float(w.adversarial_13bit_burst_detect_rate),
    }


def family_h158c_crc_v9_post_replay_jaccard(seed: int) -> dict[str, Any]:
    crc = CorruptionRobustCarrierV9(
        inner_v8=CorruptionRobustCarrierV8())
    w = emit_corruption_robustness_v9_witness(
        crc_v9=crc, n_probes=32,
        seed=int(seed) + 30400)
    return {
        "schema": R130_SCHEMA_VERSION,
        "name": "h158c_crc_v9_post_replay_jaccard",
        "passed": bool(
            float(
                w.cache_retrieval_post_replay_topk_jaccard_floor)
            >= 0.5),
        "floor": float(
            w.cache_retrieval_post_replay_topk_jaccard_floor),
        "mean": float(
            w.cache_retrieval_post_replay_topk_jaccard_mean),
    }


def family_h159_consensus_v7_stages(seed: int) -> dict[str, Any]:
    rc = ConsensusFallbackControllerV7.init()
    return {
        "schema": R130_SCHEMA_VERSION,
        "name": "h159_consensus_v7_stages",
        "passed": bool(
            len(W61_CONSENSUS_V7_STAGES) == 11
            and W61_CONSENSUS_V7_STAGE_ATTENTION_PATTERN
            in W61_CONSENSUS_V7_STAGES),
        "n_stages": int(len(W61_CONSENSUS_V7_STAGES)),
    }


def family_h159b_consensus_v7_attention_pattern_fires(
        seed: int) -> dict[str, Any]:
    rc = ConsensusFallbackControllerV7.init(
        k_required=3, cosine_floor=0.99)
    # Two parents with same top-K positions; V6 chain falls through
    # to best_parent / abstain; V7 should pick attention_pattern.
    payloads = [
        [1.0, 0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]
    trusts = [0.9, 0.9, 0.6]
    decisions = ["choose_reuse", "choose_reuse", "choose_reuse"]
    top_k = [[0, 1], [0, 1], [2, 3]]
    out = rc.decide_v7(
        payloads=payloads, trusts=trusts,
        replay_decisions=decisions,
        attention_top_k_positions=top_k,
        attention_top_k_jaccard_floor=0.5,
        transcript_available=False)
    return {
        "schema": R130_SCHEMA_VERSION,
        "name": "h159b_consensus_v7_attention_pattern_fires",
        "passed": bool(
            str(out.get("stage", ""))
            == W61_CONSENSUS_V7_STAGE_ATTENTION_PATTERN),
        "stage": str(out.get("stage", "")),
    }


def family_h160_uncertainty_v9_eight_axis(seed: int) -> dict[str, Any]:
    confs = [0.6, 0.7, 0.5]
    trusts = [0.9, 0.8, 0.7]
    sfs = [0.95, 0.9, 0.85]
    hfs = [0.95, 0.92, 0.88]
    cfs = [0.95, 0.92, 0.86]
    rfs = [0.95, 0.93, 0.84]
    rps = [0.92, 0.90, 0.82]
    aps = [0.95, 0.90, 0.78]
    out = compose_uncertainty_report_v9(
        confidences=confs, trusts=trusts,
        substrate_fidelities=sfs,
        hidden_state_fidelities=hfs,
        cache_reuse_fidelities=cfs,
        retrieval_fidelities=rfs,
        replay_fidelities=rps,
        attention_pattern_fidelities=aps)
    return {
        "schema": R130_SCHEMA_VERSION,
        "name": "h160_uncertainty_v9_eight_axis",
        "passed": bool(
            out.attention_pattern_aware
            and out.n_components == 3
            and 0.0 <= out.weighted_composite <= 1.0),
        "composite": float(out.weighted_composite),
    }


def family_h160b_uncertainty_v9_attention_aware(
        seed: int) -> dict[str, Any]:
    confs = [0.7, 0.5]
    trusts = [1.0, 1.0]
    sfs = [1.0, 1.0]; hfs = [1.0, 1.0]
    cfs = [1.0, 1.0]; rfs = [1.0, 1.0]; rps = [1.0, 1.0]
    aps_high = [0.99, 0.99]
    aps_low = [0.2, 0.2]
    high = compose_uncertainty_report_v9(
        confidences=confs, trusts=trusts,
        substrate_fidelities=sfs,
        hidden_state_fidelities=hfs,
        cache_reuse_fidelities=cfs,
        retrieval_fidelities=rfs,
        replay_fidelities=rps,
        attention_pattern_fidelities=aps_high)
    low = compose_uncertainty_report_v9(
        confidences=confs, trusts=trusts,
        substrate_fidelities=sfs,
        hidden_state_fidelities=hfs,
        cache_reuse_fidelities=cfs,
        retrieval_fidelities=rfs,
        replay_fidelities=rps,
        attention_pattern_fidelities=aps_low)
    return {
        "schema": R130_SCHEMA_VERSION,
        "name": "h160b_uncertainty_v9_attention_aware",
        "passed": bool(
            high.attention_pattern_aware
            or low.attention_pattern_aware),
    }


def family_h161_disagreement_algebra_v7_identity(
        seed: int) -> dict[str, Any]:
    trace = AlgebraTrace(steps=[])
    probe = [0.1, 0.2, 0.3]
    w = emit_disagreement_algebra_v7_witness(
        trace=trace, probe_a=probe, probe_b=probe, probe_c=probe,
        attention_pattern_oracle=lambda: (True, 0.8))
    return {
        "schema": R130_SCHEMA_VERSION,
        "name": "h161_disagreement_algebra_v7_identity",
        "passed": bool(w.attention_pattern_equiv_ok),
    }


def family_h161b_disagreement_algebra_v7_falsifier(
        seed: int) -> dict[str, Any]:
    trace = AlgebraTrace(steps=[])
    probe = [0.1, 0.2, 0.3]
    w = emit_disagreement_algebra_v7_witness(
        trace=trace, probe_a=probe, probe_b=probe, probe_c=probe,
        attention_pattern_falsifier_oracle=lambda: (False, 0.1))
    return {
        "schema": R130_SCHEMA_VERSION,
        "name": "h161b_disagreement_algebra_v7_falsifier",
        "passed": bool(w.attention_pattern_falsifier_ok),
    }


def family_h162_tvs_v10_pick_rates(seed: int) -> dict[str, Any]:
    out = eleven_arm_compare(
        per_turn_confidences=[0.5, 0.6, 0.7, 0.8],
        per_turn_trust_scores=[0.5, 0.5, 0.5, 0.5],
        per_turn_merge_retentions=[0.5, 0.5, 0.5, 0.5],
        per_turn_tw_retentions=[0.5, 0.5, 0.5, 0.5],
        per_turn_substrate_fidelities=[0.4, 0.4, 0.4, 0.4],
        per_turn_hidden_fidelities=[0.3, 0.3, 0.3, 0.3],
        per_turn_cache_fidelities=[0.4, 0.4, 0.4, 0.4],
        per_turn_retrieval_fidelities=[0.5, 0.5, 0.5, 0.5],
        per_turn_replay_fidelities=[0.6, 0.6, 0.6, 0.6],
        per_turn_attention_pattern_fidelities=[
            0.7, 0.7, 0.7, 0.7])
    s = sum(out.pick_rates.values())
    return {
        "schema": R130_SCHEMA_VERSION,
        "name": "h162_tvs_v10_pick_rates",
        "passed": bool(
            abs(s - 1.0) < 1e-9
            and len(W61_TVS_V10_ARMS) == 11),
        "sum": float(s),
        "n_arms": int(len(W61_TVS_V10_ARMS)),
    }


def family_h162b_tvs_v10_attention_pattern_arm(seed: int) -> dict[str, Any]:
    out = eleven_arm_compare(
        per_turn_confidences=[0.0],
        per_turn_trust_scores=[0.0],
        per_turn_merge_retentions=[0.0],
        per_turn_tw_retentions=[0.0],
        per_turn_substrate_fidelities=[0.0],
        per_turn_hidden_fidelities=[0.0],
        per_turn_cache_fidelities=[0.0],
        per_turn_retrieval_fidelities=[0.0],
        per_turn_replay_fidelities=[0.0],
        per_turn_attention_pattern_fidelities=[0.9])
    return {
        "schema": R130_SCHEMA_VERSION,
        "name": "h162b_tvs_v10_attention_pattern_arm",
        "passed": bool(out.attention_pattern_used),
        "pick_rates": {
            k: float(v) for k, v in out.pick_rates.items()
            if v > 0.0},
    }


def family_h162c_tvs_v10_reduces_to_v9(seed: int) -> dict[str, Any]:
    out = eleven_arm_compare(
        per_turn_confidences=[0.7],
        per_turn_trust_scores=[0.5],
        per_turn_merge_retentions=[0.5],
        per_turn_tw_retentions=[0.5],
        per_turn_substrate_fidelities=[0.4],
        per_turn_hidden_fidelities=[0.3],
        per_turn_cache_fidelities=[0.4],
        per_turn_retrieval_fidelities=[0.5],
        per_turn_replay_fidelities=[0.6],
        per_turn_attention_pattern_fidelities=[0.0])
    return {
        "schema": R130_SCHEMA_VERSION,
        "name": "h162c_tvs_v10_reduces_to_v9",
        "passed": bool(not out.attention_pattern_used),
    }


_R130_FAMILIES: tuple[Any, ...] = (
    family_h158_crc_v9_kv512_detect,
    family_h158b_crc_v9_13bit_burst,
    family_h158c_crc_v9_post_replay_jaccard,
    family_h159_consensus_v7_stages,
    family_h159b_consensus_v7_attention_pattern_fires,
    family_h160_uncertainty_v9_eight_axis,
    family_h160b_uncertainty_v9_attention_aware,
    family_h161_disagreement_algebra_v7_identity,
    family_h161b_disagreement_algebra_v7_falsifier,
    family_h162_tvs_v10_pick_rates,
    family_h162b_tvs_v10_attention_pattern_arm,
    family_h162c_tvs_v10_reduces_to_v9,
)


def run_r130(
        seeds: Sequence[int] = (198, 298, 398),
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for s in seeds:
        results = {}
        for fn in _R130_FAMILIES:
            try:
                r = fn(int(s))
            except Exception as e:
                r = {
                    "schema": R130_SCHEMA_VERSION,
                    "name": getattr(fn, "__name__", "unknown"),
                    "passed": False,
                    "exception": str(e),
                }
            results[r["name"]] = r
        out.append({
            "schema": R130_SCHEMA_VERSION,
            "seed": int(s),
            "family_results": results,
        })
    return out


__all__ = [
    "R130_SCHEMA_VERSION",
    "run_r130",
]
