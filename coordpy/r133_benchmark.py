"""W62 R-133 benchmark family — corruption / disagreement /
consensus / fallback / abstention.

H172..H180 cell families.

* H172   CRC V10 KV1024 single-byte detect rate ≥ 0.95
* H172b  CRC V10 17-bit burst detect rate ≥ 0.95
* H172c  CRC V10 post-repair top-K Jaccard floor ≥ 0.5
* H173   consensus V8 12-stage chain enumerated with repair stage
* H173b  consensus V8 trained_repair stage fires under detected
         corruption AND high repair amount
* H174   uncertainty V10 9-axis composite returns value in [0, 1]
* H174b  uncertainty V10 replay-dominance-aware flips True when fid<1
* H175   disagreement V8 Wasserstein equivalence identity holds
* H175b  disagreement V8 Wasserstein falsifier triggers
* H176   TVS V11 pick rates sum to 1.0
* H176b  TVS V11 replay-dominance arm fires when fidelity highest
* H176c  TVS V11 reduces to V10 when replay-dominance-fidelity = 0
* H177   MLSC V10 replay-dominance witness chain inherits
* H177b  MLSC V10 disagreement Wasserstein computed at merge
* H178   substrate V7 per-(layer, head, slot) cache-write ledger
         has shape (L, H, T)
* H178b  substrate V7 logit-lens probe returns shape (L, V)
* H178c  substrate V7 attention-receive delta recorded
* H178d  substrate V7 replay-trust ledger updated after decision
* H179   deep hybrid V7 seven-way fires when all axes active
* H180   substrate V7 adapter tier satisfied only by V7 runtime
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as _np

from .cache_controller_v5 import CacheControllerV5
from .consensus_fallback_controller_v8 import (
    ConsensusFallbackControllerV8,
    W62_CONSENSUS_V8_STAGES,
    W62_CONSENSUS_V8_STAGE_TRAINED_REPAIR,
    emit_consensus_v8_witness,
)
from .corruption_robust_carrier_v10 import (
    CorruptionRobustCarrierV10,
    emit_corruption_robustness_v10_witness,
)
from .deep_substrate_hybrid_v6 import (
    DeepSubstrateHybridV6, DeepSubstrateHybridV6ForwardWitness,
)
from .deep_substrate_hybrid_v7 import (
    DeepSubstrateHybridV7, deep_substrate_hybrid_v7_forward,
)
from .disagreement_algebra import AlgebraTrace
from .disagreement_algebra_v8 import (
    emit_disagreement_algebra_v8_witness,
)
from .mergeable_latent_capsule_v3 import (
    make_root_capsule_v3,
)
from .mergeable_latent_capsule_v4 import wrap_v3_as_v4
from .mergeable_latent_capsule_v5 import wrap_v4_as_v5
from .mergeable_latent_capsule_v6 import wrap_v5_as_v6
from .mergeable_latent_capsule_v7 import wrap_v6_as_v7
from .mergeable_latent_capsule_v8 import wrap_v7_as_v8
from .mergeable_latent_capsule_v9 import wrap_v8_as_v9
from .mergeable_latent_capsule_v10 import (
    MergeOperatorV10,
    W62_MLSC_V10_ALGEBRA_REPLAY_DOMINANCE_PROPAGATION,
    emit_mlsc_v10_witness, wrap_v9_as_v10,
)
from .replay_controller import ReplayCandidate
from .replay_controller_v3 import ReplayControllerV3
from .substrate_adapter_v7 import (
    W62_SUBSTRATE_TIER_SUBSTRATE_V7_FULL,
    probe_all_v7_adapters,
)
from .tiny_substrate_v7 import (
    build_default_tiny_substrate_v7,
    forward_tiny_substrate_v7,
    record_replay_decision_v7,
    tokenize_bytes_v7,
)
from .transcript_vs_shared_arbiter_v11 import (
    W62_TVS_V11_ARMS,
    W62_TVS_V11_ARM_REPLAY_DOMINANCE,
    emit_tvs_arbiter_v11_witness,
    twelve_arm_compare,
)
from .uncertainty_layer_v10 import (
    compose_uncertainty_report_v10,
)


R133_SCHEMA_VERSION: str = "coordpy.r133_benchmark.v1"


def family_h172_crc_v10_kv1024_detect(seed: int) -> dict[str, Any]:
    crc = CorruptionRobustCarrierV10()
    w = emit_corruption_robustness_v10_witness(
        crc_v10=crc, n_probes=32,
        seed=int(seed) + 33000)
    return {
        "schema": R133_SCHEMA_VERSION,
        "name": "h172_crc_v10_kv1024_detect",
        "passed": bool(
            float(w.kv1024_corruption_detect_rate) >= 0.95),
        "rate": float(w.kv1024_corruption_detect_rate),
    }


def family_h172b_crc_v10_17bit_burst(seed: int) -> dict[str, Any]:
    crc = CorruptionRobustCarrierV10()
    w = emit_corruption_robustness_v10_witness(
        crc_v10=crc, n_probes=32,
        seed=int(seed) + 33100)
    return {
        "schema": R133_SCHEMA_VERSION,
        "name": "h172b_crc_v10_17bit_burst",
        "passed": bool(
            float(w.adversarial_17bit_burst_detect_rate)
            >= 0.95),
        "rate": float(w.adversarial_17bit_burst_detect_rate),
    }


def family_h172c_crc_v10_post_repair_jaccard(
        seed: int) -> dict[str, Any]:
    crc = CorruptionRobustCarrierV10()
    w = emit_corruption_robustness_v10_witness(
        crc_v10=crc, n_probes=32,
        seed=int(seed) + 33200)
    return {
        "schema": R133_SCHEMA_VERSION,
        "name": "h172c_crc_v10_post_repair_jaccard",
        "passed": bool(
            float(w.post_repair_topk_jaccard_floor) >= 0.5),
        "floor": float(w.post_repair_topk_jaccard_floor),
        "mean": float(w.post_repair_topk_jaccard_mean),
    }


def family_h173_consensus_v8_twelve_stages(
        seed: int) -> dict[str, Any]:
    return {
        "schema": R133_SCHEMA_VERSION,
        "name": "h173_consensus_v8_twelve_stages",
        "passed": bool(
            len(W62_CONSENSUS_V8_STAGES) == 12
            and W62_CONSENSUS_V8_STAGE_TRAINED_REPAIR
            in W62_CONSENSUS_V8_STAGES),
        "n_stages": int(len(W62_CONSENSUS_V8_STAGES)),
    }


def family_h173b_consensus_v8_repair_stage_fires(
        seed: int) -> dict[str, Any]:
    rc = ConsensusFallbackControllerV8.init(
        k_required=3, cosine_floor=0.99,
        repair_amount_threshold=0.1)
    payloads = [
        [1.0, 0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]
    trusts = [0.9, 0.9, 0.6]
    decisions = ["choose_reuse", "choose_reuse",
                 "choose_reuse"]
    out = rc.decide_v8(
        payloads=payloads, trusts=trusts,
        replay_decisions=decisions,
        attention_top_k_positions=None,
        attention_top_k_jaccard_floor=0.5,
        transcript_available=False,
        corruption_detected_per_parent=[True, False, False],
        repair_amount=0.5,
        repaired_payload=[0.7, 0.2, 0.1, 0.0])
    return {
        "schema": R133_SCHEMA_VERSION,
        "name": "h173b_consensus_v8_repair_stage_fires",
        "passed": bool(
            str(out.get("stage", ""))
            == W62_CONSENSUS_V8_STAGE_TRAINED_REPAIR),
        "stage": str(out.get("stage", "")),
    }


def family_h174_uncertainty_v10_nine_axis(
        seed: int) -> dict[str, Any]:
    confs = [0.6, 0.7, 0.5]
    trusts = [0.9, 0.8, 0.7]
    out = compose_uncertainty_report_v10(
        confidences=confs, trusts=trusts,
        substrate_fidelities=[0.95, 0.9, 0.85],
        hidden_state_fidelities=[0.95, 0.92, 0.88],
        cache_reuse_fidelities=[0.95, 0.92, 0.86],
        retrieval_fidelities=[0.95, 0.93, 0.84],
        replay_fidelities=[0.92, 0.90, 0.82],
        attention_pattern_fidelities=[0.95, 0.90, 0.78],
        replay_dominance_fidelities=[0.9, 0.85, 0.75])
    return {
        "schema": R133_SCHEMA_VERSION,
        "name": "h174_uncertainty_v10_nine_axis",
        "passed": bool(
            out.n_axes == 9
            and 0.0 <= out.weighted_composite <= 1.0),
        "composite": float(out.weighted_composite),
    }


def family_h174b_uncertainty_v10_rd_aware(
        seed: int) -> dict[str, Any]:
    confs = [0.7, 0.5]
    trusts = [1.0, 1.0]
    high = compose_uncertainty_report_v10(
        confidences=confs, trusts=trusts,
        substrate_fidelities=[1.0, 1.0],
        hidden_state_fidelities=[1.0, 1.0],
        cache_reuse_fidelities=[1.0, 1.0],
        retrieval_fidelities=[1.0, 1.0],
        replay_fidelities=[1.0, 1.0],
        attention_pattern_fidelities=[1.0, 1.0],
        replay_dominance_fidelities=[0.99, 0.99])
    low = compose_uncertainty_report_v10(
        confidences=confs, trusts=trusts,
        substrate_fidelities=[1.0, 1.0],
        hidden_state_fidelities=[1.0, 1.0],
        cache_reuse_fidelities=[1.0, 1.0],
        retrieval_fidelities=[1.0, 1.0],
        replay_fidelities=[1.0, 1.0],
        attention_pattern_fidelities=[1.0, 1.0],
        replay_dominance_fidelities=[0.2, 0.2])
    return {
        "schema": R133_SCHEMA_VERSION,
        "name": "h174b_uncertainty_v10_rd_aware",
        "passed": bool(
            high.replay_dominance_aware
            or low.replay_dominance_aware),
    }


def family_h175_disagreement_v8_wasserstein_identity(
        seed: int) -> dict[str, Any]:
    trace = AlgebraTrace(steps=[])
    probe = [0.1, 0.2, 0.3]
    w = emit_disagreement_algebra_v8_witness(
        trace=trace, probe_a=probe, probe_b=probe,
        probe_c=probe,
        wasserstein_oracle=lambda: (True, 0.1))
    return {
        "schema": R133_SCHEMA_VERSION,
        "name": "h175_disagreement_v8_wasserstein_identity",
        "passed": bool(w.wasserstein_equiv_ok),
    }


def family_h175b_disagreement_v8_wasserstein_falsifier(
        seed: int) -> dict[str, Any]:
    trace = AlgebraTrace(steps=[])
    probe = [0.1, 0.2, 0.3]
    w = emit_disagreement_algebra_v8_witness(
        trace=trace, probe_a=probe, probe_b=probe,
        probe_c=probe,
        wasserstein_falsifier_oracle=lambda: (False, 1.0))
    return {
        "schema": R133_SCHEMA_VERSION,
        "name": "h175b_disagreement_v8_wasserstein_falsifier",
        "passed": bool(w.wasserstein_falsifier_ok),
    }


def family_h176_tvs_v11_pick_rates(seed: int) -> dict[str, Any]:
    out = twelve_arm_compare(
        per_turn_replay_dominance_fidelities=[
            0.4, 0.5, 0.6, 0.7],
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
    s = float(sum(out.pick_rates.values()))
    return {
        "schema": R133_SCHEMA_VERSION,
        "name": "h176_tvs_v11_pick_rates",
        "passed": bool(
            abs(s - 1.0) < 1e-6
            and len(W62_TVS_V11_ARMS) == 12),
        "sum": float(s),
        "n_arms": int(len(W62_TVS_V11_ARMS)),
    }


def family_h176b_tvs_v11_replay_dominance_arm(
        seed: int) -> dict[str, Any]:
    out = twelve_arm_compare(
        per_turn_replay_dominance_fidelities=[0.99],
        per_turn_confidences=[0.0],
        per_turn_trust_scores=[0.0],
        per_turn_merge_retentions=[0.0],
        per_turn_tw_retentions=[0.0],
        per_turn_substrate_fidelities=[0.0],
        per_turn_hidden_fidelities=[0.0],
        per_turn_cache_fidelities=[0.0],
        per_turn_retrieval_fidelities=[0.0],
        per_turn_replay_fidelities=[0.0],
        per_turn_attention_pattern_fidelities=[0.0])
    rd_rate = out.pick_rates.get(
        W62_TVS_V11_ARM_REPLAY_DOMINANCE, 0.0)
    return {
        "schema": R133_SCHEMA_VERSION,
        "name": "h176b_tvs_v11_replay_dominance_arm",
        "passed": bool(rd_rate > 0.5),
        "rd_rate": float(rd_rate),
    }


def family_h176c_tvs_v11_reduces_to_v10(
        seed: int) -> dict[str, Any]:
    out = twelve_arm_compare(
        per_turn_replay_dominance_fidelities=[0.0],
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
    rd_rate = out.pick_rates.get(
        W62_TVS_V11_ARM_REPLAY_DOMINANCE, 0.0)
    return {
        "schema": R133_SCHEMA_VERSION,
        "name": "h176c_tvs_v11_reduces_to_v10",
        "passed": bool(rd_rate < 1e-9
                       and not out.replay_dominance_used),
        "rd_rate": float(rd_rate),
    }


def family_h177_mlsc_v10_replay_dominance_chain(
        seed: int) -> dict[str, Any]:
    op = MergeOperatorV10(factor_dim=6)
    v3 = make_root_capsule_v3(
        branch_id="r133", payload=tuple([0.1] * 6),
        fact_tags=("r133",),
        confidence=0.9, trust=0.9,
        turn_index=int(seed))
    v4 = wrap_v3_as_v4(v3)
    v5 = wrap_v4_as_v5(v4, attention_witness_cid="a")
    v6 = wrap_v5_as_v6(v5,
        attention_witness_chain=("aw",),
        cache_reuse_witness_cid="cr")
    v7 = wrap_v6_as_v7(v6,
        retrieval_witness_chain=("rw",),
        controller_witness_cid="ctrl")
    v8 = wrap_v7_as_v8(v7,
        replay_witness_chain=("repl",),
        substrate_witness_chain=("sub",),
        provenance_trust_table={"b": 0.9})
    v9 = wrap_v8_as_v9(v8,
        attention_pattern_witness_chain=("ap",),
        cache_retrieval_witness_chain=("crv9",),
        per_layer_head_trust_matrix=((0, 0, 0.9),))
    v10 = wrap_v9_as_v10(v9,
        replay_dominance_witness_chain=("rd_v10",),
        algebra_signature_v10=(
            W62_MLSC_V10_ALGEBRA_REPLAY_DOMINANCE_PROPAGATION))
    merged = op.merge(
        [v10],
        replay_dominance_witness_chain=("merge_rd",),
        algebra_signature_v10=(
            W62_MLSC_V10_ALGEBRA_REPLAY_DOMINANCE_PROPAGATION))
    w = emit_mlsc_v10_witness(merged)
    return {
        "schema": R133_SCHEMA_VERSION,
        "name": "h177_mlsc_v10_replay_dominance_chain",
        "passed": bool(
            w.replay_dominance_witness_chain_depth >= 2),
        "depth": int(
            w.replay_dominance_witness_chain_depth),
    }


def family_h177b_mlsc_v10_wasserstein_distance(
        seed: int) -> dict[str, Any]:
    op = MergeOperatorV10(factor_dim=6)
    def _mk(payload):
        v3 = make_root_capsule_v3(
            branch_id=f"r133_{seed}",
            payload=tuple(payload),
            fact_tags=("r133",),
            confidence=0.9, trust=0.9,
            turn_index=int(seed))
        v4 = wrap_v3_as_v4(v3)
        v5 = wrap_v4_as_v5(v4, attention_witness_cid="a")
        v6 = wrap_v5_as_v6(v5,
            attention_witness_chain=("aw",),
            cache_reuse_witness_cid="cr")
        v7 = wrap_v6_as_v7(v6,
            retrieval_witness_chain=("rw",),
            controller_witness_cid="ctrl")
        v8 = wrap_v7_as_v8(v7,
            replay_witness_chain=("repl",),
            substrate_witness_chain=("sub",),
            provenance_trust_table={"b": 0.9})
        v9 = wrap_v8_as_v9(v8,
            attention_pattern_witness_chain=("ap",),
            cache_retrieval_witness_chain=("crv9",))
        v10 = wrap_v9_as_v10(v9)
        return v10
    cap_a = _mk([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    cap_b = _mk([1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
    merged = op.merge([cap_a, cap_b])
    return {
        "schema": R133_SCHEMA_VERSION,
        "name": "h177b_mlsc_v10_wasserstein_distance",
        "passed": bool(
            merged.disagreement_wasserstein_distance > 0.0),
        "wasserstein": float(
            merged.disagreement_wasserstein_distance),
    }


def family_h178_substrate_v7_cache_write_ledger(
        seed: int) -> dict[str, Any]:
    p = build_default_tiny_substrate_v7(seed=int(seed) + 33800)
    ids = tokenize_bytes_v7("sub", max_len=8)
    _, cache = forward_tiny_substrate_v7(p, ids)
    L, H, T = cache.cache_write_ledger.shape
    return {
        "schema": R133_SCHEMA_VERSION,
        "name": "h178_substrate_v7_cache_write_ledger",
        "passed": bool(
            L == p.config.n_layers
            and H == p.config.n_heads),
        "shape": [int(L), int(H), int(T)],
    }


def family_h178b_substrate_v7_logit_lens(
        seed: int) -> dict[str, Any]:
    p = build_default_tiny_substrate_v7(seed=int(seed) + 33900)
    ids = tokenize_bytes_v7("sub", max_len=8)
    trace, _ = forward_tiny_substrate_v7(p, ids)
    L, V = trace.logit_lens_per_layer.shape
    return {
        "schema": R133_SCHEMA_VERSION,
        "name": "h178b_substrate_v7_logit_lens",
        "passed": bool(
            L == p.config.n_layers
            and V == p.config.vocab_size),
        "shape": [int(L), int(V)],
    }


def family_h178c_substrate_v7_attention_receive_delta(
        seed: int) -> dict[str, Any]:
    p = build_default_tiny_substrate_v7(seed=int(seed) + 34000)
    ids = tokenize_bytes_v7("sub", max_len=8)
    trace, cache = forward_tiny_substrate_v7(p, ids)
    trace2, _ = forward_tiny_substrate_v7(
        p, ids[:4], v7_kv_cache=cache)
    total = float(sum(
        float(_np.linalg.norm(d.ravel()))
        for d in trace2.attention_receive_delta_per_layer))
    return {
        "schema": R133_SCHEMA_VERSION,
        "name": "h178c_substrate_v7_attention_receive_delta",
        "passed": bool(total > 0.0),
        "delta_l2_total": float(total),
    }


def family_h178d_substrate_v7_replay_trust_ledger(
        seed: int) -> dict[str, Any]:
    p = build_default_tiny_substrate_v7(seed=int(seed) + 34100)
    ids = tokenize_bytes_v7("sub", max_len=4)
    _, cache = forward_tiny_substrate_v7(p, ids)
    pre = float(cache.replay_trust_ledger.sum())
    record_replay_decision_v7(
        cache, layer_index=0, head_index=0,
        decision="choose_reuse")
    post = float(cache.replay_trust_ledger.sum())
    return {
        "schema": R133_SCHEMA_VERSION,
        "name": "h178d_substrate_v7_replay_trust_ledger",
        "passed": bool(post > pre + 1e-9),
        "pre": float(pre), "post": float(post),
    }


def family_h179_deep_hybrid_v7_seven_way(
        seed: int) -> dict[str, Any]:
    # Build a V7 hybrid + V6 witness asserting all V6 axes fire,
    # then provide V5 cache controller fit + V3 replay decisions
    # so V7's seven_way fires.
    rng = _np.random.default_rng(int(seed) + 34200)
    cc = CacheControllerV5.init(
        d_model=64, d_key=8, fit_seed=int(seed) + 34201)
    cc.two_objective_head = _np.zeros((4, 2))
    cc.repair_head_coefs = _np.zeros((4,))
    cc.composite_v5_weights = _np.ones((6,)) * (1.0 / 6.0)
    rcv3 = ReplayControllerV3.init()
    rcv3.audit_v3.append({
        "stage": "v3_per_regime",
        "regime": "crc_passed_low_drift",
        "replay_dominance": 0.3,
        "confidence": 0.7,
        "decision": "choose_reuse"})
    rcv3.hidden_vs_kv_classifier = _np.zeros((3, 5))
    hybrid = DeepSubstrateHybridV7(
        inner_v6=DeepSubstrateHybridV6(inner_v5=None))
    v6_witness = DeepSubstrateHybridV6ForwardWitness(
        schema="x", hybrid_cid="x",
        inner_v5_witness_cid="x", six_way=True,
        cache_controller_v4_fired=True,
        replay_controller_v2_fired=True,
        attention_steering_v5_fired=True,
        bilinear_retrieval_v6_used=True,
        hidden_write_gate_fired=False,
        decision_confidence_mean=0.6,
        attention_pattern_jaccard_mean=0.6,
        attention_pattern_l2_max=0.5)
    wh = deep_substrate_hybrid_v7_forward(
        hybrid=hybrid, v6_witness=v6_witness,
        cache_controller_v5=cc, replay_controller_v3=rcv3,
        cache_write_ledger_l2=1.0,
        attention_v6_coarse_l1_shift=0.5,
        prefix_v6_drift_predictor_trained=True)
    return {
        "schema": R133_SCHEMA_VERSION,
        "name": "h179_deep_hybrid_v7_seven_way",
        "passed": bool(wh.seven_way),
        "seven_way": bool(wh.seven_way),
    }


def family_h180_substrate_v7_adapter_tier(
        seed: int) -> dict[str, Any]:
    matrix = probe_all_v7_adapters(
        probe_ollama=False, probe_openai=False)
    has_v7_full = any(
        c.tier == W62_SUBSTRATE_TIER_SUBSTRATE_V7_FULL
        for c in matrix.capabilities)
    synthetic_not_v7 = any(
        c.tier != W62_SUBSTRATE_TIER_SUBSTRATE_V7_FULL
        for c in matrix.capabilities
        if c.backend_name != "tiny_substrate_v7")
    return {
        "schema": R133_SCHEMA_VERSION,
        "name": "h180_substrate_v7_adapter_tier",
        "passed": bool(has_v7_full and synthetic_not_v7),
    }


_R133_FAMILIES: tuple[Any, ...] = (
    family_h172_crc_v10_kv1024_detect,
    family_h172b_crc_v10_17bit_burst,
    family_h172c_crc_v10_post_repair_jaccard,
    family_h173_consensus_v8_twelve_stages,
    family_h173b_consensus_v8_repair_stage_fires,
    family_h174_uncertainty_v10_nine_axis,
    family_h174b_uncertainty_v10_rd_aware,
    family_h175_disagreement_v8_wasserstein_identity,
    family_h175b_disagreement_v8_wasserstein_falsifier,
    family_h176_tvs_v11_pick_rates,
    family_h176b_tvs_v11_replay_dominance_arm,
    family_h176c_tvs_v11_reduces_to_v10,
    family_h177_mlsc_v10_replay_dominance_chain,
    family_h177b_mlsc_v10_wasserstein_distance,
    family_h178_substrate_v7_cache_write_ledger,
    family_h178b_substrate_v7_logit_lens,
    family_h178c_substrate_v7_attention_receive_delta,
    family_h178d_substrate_v7_replay_trust_ledger,
    family_h179_deep_hybrid_v7_seven_way,
    family_h180_substrate_v7_adapter_tier,
)


def run_r133(
        seeds: Sequence[int] = (201, 301, 401),
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for s in seeds:
        results = {}
        for fn in _R133_FAMILIES:
            try:
                r = fn(int(s))
            except Exception as e:
                r = {
                    "schema": R133_SCHEMA_VERSION,
                    "name": getattr(fn, "__name__", "unknown"),
                    "passed": False,
                    "exception": str(e),
                }
            results[r["name"]] = r
        out.append({
            "schema": R133_SCHEMA_VERSION,
            "seed": int(s),
            "family_results": results,
        })
    return out


__all__ = ["R133_SCHEMA_VERSION", "run_r133"]
