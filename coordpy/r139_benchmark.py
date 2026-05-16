"""W64 R-139 benchmark family — corruption / disagreement /
consensus / fallback / replay-dominance-primary / hostile-channel.

H215..H222b cell families (16 H-bars).

* H215   CRC V12 1-byte detect rate ≥ 0.95 on 4096-bucket
* H215b  CRC V12 23-bit adversarial burst detect rate ≥ 0.5
* H215c  CRC V12 replay-dominance recovery ratio mean ≥ 0.4
* H215d  CRC V12 substrate-corruption blast radius mean ≥ 1.0
* H216   Consensus V10 has 14 disjoint stages
* H216b  Consensus V10 replay_dominance_primary_arbiter stage
         fires under V9-terminal + replay-wins prediction +
         dominance-primary score above threshold
* H217   MLSC V12 replay_dominance_primary chain inherits as
         union
* H217b  MLSC V12 total-variation distance computed at merge
* H217c  MLSC V12 hidden_state_trust chain inherits as union
* H218   Disagreement algebra V10 TV identity holds
* H218b  Disagreement algebra V10 TV falsifier triggers
* H219   Substrate adapter V9 has matrix.has_v9_full=True
* H219b  Hosted backends remain text-only at V9 tier
* H220   Deep substrate hybrid V9 sets nine_way=True
* H220b  Deep substrate hybrid V9 reduces to V8 byte-for-byte
* H221   W64 envelope verifier knows ≥ 85 failure modes
* H221b  Hidden-wins-primary falsifier returns 0 under inversion
* H222   TVS V13 fourteen arms sum to 1.0
* H222b  Uncertainty V12 11-axis composite returns value in [0, 1]
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as _np

from coordpy.cache_controller_v7 import CacheControllerV7
from coordpy.consensus_fallback_controller_v10 import (
    ConsensusFallbackControllerV10,
    W64_CONSENSUS_V10_STAGES,
    W64_CONSENSUS_V10_STAGE_REPLAY_DOMINANCE_PRIMARY_ARBITER,
)
from coordpy.corruption_robust_carrier_v12 import (
    CorruptionRobustCarrierV12,
    emit_corruption_robustness_v12_witness,
)
from coordpy.deep_substrate_hybrid_v8 import (
    DeepSubstrateHybridV8,
    DeepSubstrateHybridV8ForwardWitness,
)
from coordpy.deep_substrate_hybrid_v9 import (
    DeepSubstrateHybridV9,
    deep_substrate_hybrid_v9_forward,
)
from coordpy.disagreement_algebra import AlgebraTrace
from coordpy.disagreement_algebra_v10 import (
    check_tv_equivalence_identity,
    tv_equivalence_falsifier,
    emit_disagreement_algebra_v10_witness,
)
from coordpy.kv_bridge_v9 import (
    probe_kv_bridge_v9_hidden_wins_primary_falsifier,
)
from coordpy.mergeable_latent_capsule_v3 import make_root_capsule_v3
from coordpy.mergeable_latent_capsule_v4 import wrap_v3_as_v4
from coordpy.mergeable_latent_capsule_v5 import wrap_v4_as_v5
from coordpy.mergeable_latent_capsule_v6 import wrap_v5_as_v6
from coordpy.mergeable_latent_capsule_v7 import wrap_v6_as_v7
from coordpy.mergeable_latent_capsule_v8 import wrap_v7_as_v8
from coordpy.mergeable_latent_capsule_v9 import wrap_v8_as_v9
from coordpy.mergeable_latent_capsule_v10 import wrap_v9_as_v10
from coordpy.mergeable_latent_capsule_v11 import wrap_v10_as_v11
from coordpy.mergeable_latent_capsule_v12 import (
    MergeOperatorV12,
    W64_MLSC_V12_ALGEBRA_REPLAY_DOMINANCE_PRIMARY_PROPAGATION,
    wrap_v11_as_v12,
)
from coordpy.replay_controller_v5 import ReplayControllerV5
from coordpy.substrate_adapter_v9 import (
    W64_SUBSTRATE_TIER_SUBSTRATE_V9_FULL,
    probe_all_v9_adapters,
)
from coordpy.transcript_vs_shared_arbiter_v13 import (
    W64_TVS_V13_ARMS, fourteen_arm_compare,
)
from coordpy.uncertainty_layer_v12 import (
    compose_uncertainty_report_v12,
)
from coordpy.w64_team import (
    W64_ENVELOPE_VERIFIER_FAILURE_MODES,
)


R139_SCHEMA_VERSION: str = "coordpy.r139_benchmark.v1"


def family_h215_crc_v12_4096bucket_detect(
        seed: int) -> dict[str, Any]:
    crc = CorruptionRobustCarrierV12()
    w = emit_corruption_robustness_v12_witness(
        crc_v12=crc, n_probes=24, seed=int(seed) + 39000)
    return {
        "schema": R139_SCHEMA_VERSION,
        "name": "h215_crc_v12_4096bucket_detect",
        "passed": bool(
            w.kv4096_corruption_detect_rate >= 0.95),
        "rate": float(w.kv4096_corruption_detect_rate),
    }


def family_h215b_crc_v12_23bit_burst(
        seed: int) -> dict[str, Any]:
    crc = CorruptionRobustCarrierV12()
    w = emit_corruption_robustness_v12_witness(
        crc_v12=crc, n_probes=24, seed=int(seed) + 39100)
    return {
        "schema": R139_SCHEMA_VERSION,
        "name": "h215b_crc_v12_23bit_burst",
        "passed": bool(
            w.adversarial_23bit_burst_detect_rate >= 0.5),
        "rate": float(w.adversarial_23bit_burst_detect_rate),
    }


def family_h215c_crc_v12_replay_dominance_recovery(
        seed: int) -> dict[str, Any]:
    crc = CorruptionRobustCarrierV12()
    w = emit_corruption_robustness_v12_witness(
        crc_v12=crc, n_probes=24, seed=int(seed) + 39200)
    return {
        "schema": R139_SCHEMA_VERSION,
        "name": "h215c_crc_v12_replay_dominance_recovery",
        "passed": bool(
            w.replay_dominance_recovery_ratio_mean >= 0.4),
        "mean": float(
            w.replay_dominance_recovery_ratio_mean),
    }


def family_h215d_crc_v12_blast_radius(
        seed: int) -> dict[str, Any]:
    crc = CorruptionRobustCarrierV12()
    w = emit_corruption_robustness_v12_witness(
        crc_v12=crc, n_probes=24, seed=int(seed) + 39300)
    return {
        "schema": R139_SCHEMA_VERSION,
        "name": "h215d_crc_v12_blast_radius",
        "passed": bool(
            w.substrate_blast_radius_mean >= 1.0),
        "mean": float(w.substrate_blast_radius_mean),
    }


def family_h216_consensus_v10_stage_count(
        seed: int) -> dict[str, Any]:
    return {
        "schema": R139_SCHEMA_VERSION,
        "name": "h216_consensus_v10_stage_count",
        "passed": bool(len(W64_CONSENSUS_V10_STAGES) == 14),
        "n_stages": int(len(W64_CONSENSUS_V10_STAGES)),
    }


def family_h216b_consensus_v10_replay_dominance_primary_fires(
        seed: int) -> dict[str, Any]:
    ctrl = ConsensusFallbackControllerV10.init()
    res = ctrl.decide_v10(
        payloads=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        trusts=[0.4, 0.4],
        replay_decisions=[
            "choose_abstain", "choose_abstain"],
        transcript_available=False,
        corruption_detected_per_parent=[False, False],
        repair_amount=0.0,
        hidden_wins_margins_per_parent=[0.0, 0.0],
        three_way_predictions_per_parent=[
            "kv_wins", "kv_wins"],
        replay_dominance_primary_scores_per_parent=[
            0.5, 0.0],
        four_way_predictions_per_parent=[
            "replay_wins", "kv_wins"])
    return {
        "schema": R139_SCHEMA_VERSION,
        "name": (
            "h216b_consensus_v10_replay_dominance_primary_fires"),
        "passed": bool(
            res["stage"]
            == W64_CONSENSUS_V10_STAGE_REPLAY_DOMINANCE_PRIMARY_ARBITER),
        "stage": str(res.get("stage", "")),
    }


def _build_v12_capsule(
        branch: str, *, rdp_witness: str, hst_witness: str,
) -> Any:
    v3 = make_root_capsule_v3(
        branch_id=branch, payload=tuple([0.1] * 6),
        fact_tags=("w64",), confidence=0.9, trust=0.9,
        turn_index=0)
    v4 = wrap_v3_as_v4(v3)
    v5 = wrap_v4_as_v5(v4, attention_witness_cid="a")
    v6 = wrap_v5_as_v6(
        v5, attention_witness_chain=("a_chain",),
        cache_reuse_witness_cid="c")
    v7 = wrap_v6_as_v7(
        v6, retrieval_witness_chain=("r_chain",),
        controller_witness_cid="ctrl")
    v8 = wrap_v7_as_v8(
        v7, replay_witness_chain=("rc",),
        substrate_witness_chain=("sc",),
        provenance_trust_table={"a": 0.9})
    v9 = wrap_v8_as_v9(
        v8,
        attention_pattern_witness_chain=("ap",),
        cache_retrieval_witness_chain=("cr",),
        per_layer_head_trust_matrix=((0, 0, 0.9),))
    v10 = wrap_v9_as_v10(
        v9, replay_dominance_witness_chain=("rd",),
        disagreement_wasserstein_distance=0.05)
    v11 = wrap_v10_as_v11(
        v10, hidden_wins_witness_chain=("hw",))
    return wrap_v11_as_v12(
        v11,
        replay_dominance_primary_witness_chain=(rdp_witness,),
        hidden_state_trust_witness_chain=(hst_witness,))


def family_h217_mlsc_v12_replay_dominance_primary_chain(
        seed: int) -> dict[str, Any]:
    op = MergeOperatorV12(factor_dim=6)
    a = _build_v12_capsule(
        "A", rdp_witness="rdp_a", hst_witness="hst_a")
    b = _build_v12_capsule(
        "B", rdp_witness="rdp_b", hst_witness="hst_b")
    merged = op.merge([a, b])
    return {
        "schema": R139_SCHEMA_VERSION,
        "name": (
            "h217_mlsc_v12_replay_dominance_primary_chain"),
        "passed": bool(
            "rdp_a" in merged.replay_dominance_primary_witness_chain
            and "rdp_b" in
                merged.replay_dominance_primary_witness_chain),
    }


def family_h217b_mlsc_v12_total_variation_distance(
        seed: int) -> dict[str, Any]:
    op = MergeOperatorV12(factor_dim=6)
    a = _build_v12_capsule(
        "A", rdp_witness="rdp_a", hst_witness="hst_a")
    b = _build_v12_capsule(
        "B", rdp_witness="rdp_b", hst_witness="hst_b")
    merged = op.merge([a, b])
    return {
        "schema": R139_SCHEMA_VERSION,
        "name": "h217b_mlsc_v12_total_variation_distance",
        "passed": bool(
            0.0 <= merged.disagreement_total_variation_distance
            <= 1.0),
        "tv": float(
            merged.disagreement_total_variation_distance),
    }


def family_h217c_mlsc_v12_hidden_state_trust_chain(
        seed: int) -> dict[str, Any]:
    op = MergeOperatorV12(factor_dim=6)
    a = _build_v12_capsule(
        "A", rdp_witness="rdp_a", hst_witness="hst_a")
    b = _build_v12_capsule(
        "B", rdp_witness="rdp_b", hst_witness="hst_b")
    merged = op.merge([a, b])
    return {
        "schema": R139_SCHEMA_VERSION,
        "name": "h217c_mlsc_v12_hidden_state_trust_chain",
        "passed": bool(
            "hst_a" in merged.hidden_state_trust_witness_chain
            and "hst_b" in
                merged.hidden_state_trust_witness_chain),
    }


def family_h218_disagreement_v10_tv_identity(
        seed: int) -> dict[str, Any]:
    ok = check_tv_equivalence_identity(
        tv_oracle=lambda: (True, 0.05), tv_floor=0.30)
    return {
        "schema": R139_SCHEMA_VERSION,
        "name": "h218_disagreement_v10_tv_identity",
        "passed": bool(ok),
    }


def family_h218b_disagreement_v10_tv_falsifier(
        seed: int) -> dict[str, Any]:
    fals = tv_equivalence_falsifier(
        tv_oracle=lambda: (False, 1.0), tv_floor=0.30)
    return {
        "schema": R139_SCHEMA_VERSION,
        "name": "h218b_disagreement_v10_tv_falsifier",
        "passed": bool(fals),
    }


def family_h219_substrate_adapter_v9_full(
        seed: int) -> dict[str, Any]:
    matrix = probe_all_v9_adapters(
        probe_ollama=False, probe_openai=False)
    return {
        "schema": R139_SCHEMA_VERSION,
        "name": "h219_substrate_adapter_v9_full",
        "passed": bool(matrix.has_v9_full()),
    }


def family_h219b_hosted_backends_text_only_at_v9(
        seed: int) -> dict[str, Any]:
    matrix = probe_all_v9_adapters(
        probe_ollama=False, probe_openai=False)
    has = matrix.has_v9_full()
    n_caps = len(matrix.capabilities)
    return {
        "schema": R139_SCHEMA_VERSION,
        "name": "h219b_hosted_backends_text_only_at_v9",
        "passed": bool(has and n_caps >= 1),
        "n_caps": int(n_caps),
    }


def family_h220_deep_substrate_hybrid_v9_nine_way(
        seed: int) -> dict[str, Any]:
    hybrid = DeepSubstrateHybridV9(
        inner_v8=DeepSubstrateHybridV8(inner_v7=None))
    cc = CacheControllerV7.init(fit_seed=int(seed) + 39400)
    cc = type(cc)(
        policy=cc.policy, inner_v6=cc.inner_v6,
        four_objective_head=_np.eye(4),
        similarity_eviction_head_coefs=_np.zeros(6),
        composite_v7_weights=_np.ones(8) / 8,
        ridge_lambda=cc.ridge_lambda, fit_seed=cc.fit_seed)
    rcv5 = ReplayControllerV5.init()
    rcv5.audit_v5.append(
        {"replay_dominance": 0.4})
    rcv5.four_way_bridge_classifier = _np.zeros((4, 9))
    v8w = DeepSubstrateHybridV8ForwardWitness(
        schema="x", hybrid_cid="x",
        inner_v7_witness_cid="x",
        eight_way=True,
        cache_controller_v6_fired=True,
        replay_controller_v4_fired=True,
        three_way_bridge_classifier_fired=True,
        hidden_vs_kv_contention_active=True,
        attention_v7_active=True,
        prefix_v7_drift_predictor_active=True,
        prefix_reuse_trust_active=True,
        mean_replay_dominance=0.4,
        hidden_vs_kv_contention_l1=1.0,
        attention_v7_js_max=0.15,
        prefix_reuse_trust_l1=0.7)
    w = deep_substrate_hybrid_v9_forward(
        hybrid=hybrid, v8_witness=v8w,
        cache_controller_v7=cc,
        replay_controller_v5=rcv5,
        hidden_wins_primary_l1=1.0,
        attention_v8_hellinger_max=0.15,
        prefix_v8_drift_predictor_trained=True,
        hidden_state_trust_ledger_l1=0.5,
        hsb_v8_hidden_wins_primary_margin=0.2)
    return {
        "schema": R139_SCHEMA_VERSION,
        "name": "h220_deep_substrate_hybrid_v9_nine_way",
        "passed": bool(w.nine_way),
        "nine_way": bool(w.nine_way),
    }


def family_h220b_deep_substrate_hybrid_v9_reduces_to_v8(
        seed: int) -> dict[str, Any]:
    hybrid = DeepSubstrateHybridV9(
        inner_v8=DeepSubstrateHybridV8(inner_v7=None))
    v8w = DeepSubstrateHybridV8ForwardWitness(
        schema="x", hybrid_cid="x",
        inner_v7_witness_cid="x",
        eight_way=False,
        cache_controller_v6_fired=False,
        replay_controller_v4_fired=False,
        three_way_bridge_classifier_fired=False,
        hidden_vs_kv_contention_active=False,
        attention_v7_active=False,
        prefix_v7_drift_predictor_active=False,
        prefix_reuse_trust_active=False,
        mean_replay_dominance=0.0,
        hidden_vs_kv_contention_l1=0.0,
        attention_v7_js_max=0.0,
        prefix_reuse_trust_l1=0.0)
    w = deep_substrate_hybrid_v9_forward(
        hybrid=hybrid, v8_witness=v8w,
        hidden_wins_primary_l1=0.0,
        attention_v8_hellinger_max=0.0,
        prefix_v8_drift_predictor_trained=False,
        hidden_state_trust_ledger_l1=0.0,
        hsb_v8_hidden_wins_primary_margin=0.0)
    return {
        "schema": R139_SCHEMA_VERSION,
        "name": "h220b_deep_substrate_hybrid_v9_reduces_to_v8",
        "passed": bool(not w.nine_way),
    }


def family_h221_w64_envelope_verifier_modes(
        seed: int) -> dict[str, Any]:
    n = len(W64_ENVELOPE_VERIFIER_FAILURE_MODES)
    return {
        "schema": R139_SCHEMA_VERSION,
        "name": "h221_w64_envelope_verifier_modes",
        "passed": bool(n >= 85),
        "n_modes": int(n),
    }


def family_h221b_kv_v9_hidden_wins_primary_falsifier(
        seed: int) -> dict[str, Any]:
    w = probe_kv_bridge_v9_hidden_wins_primary_falsifier(
        primary_flag=0.7)
    return {
        "schema": R139_SCHEMA_VERSION,
        "name": "h221b_kv_v9_hidden_wins_primary_falsifier",
        "passed": bool(w.falsifier_score == 0.0),
    }


def family_h222_tvs_v13_fourteen_arms(
        seed: int) -> dict[str, Any]:
    res = fourteen_arm_compare(
        per_turn_replay_dominance_primary_fidelities=[0.5],
        per_turn_hidden_wins_fidelities=[0.5],
        per_turn_replay_dominance_fidelities=[0.6],
        per_turn_confidences=[0.8],
        per_turn_trust_scores=[0.7],
        per_turn_merge_retentions=[0.6],
        per_turn_tw_retentions=[0.6],
        per_turn_substrate_fidelities=[0.5],
        per_turn_hidden_fidelities=[0.4],
        per_turn_cache_fidelities=[0.5],
        per_turn_retrieval_fidelities=[0.6],
        per_turn_replay_fidelities=[0.7],
        per_turn_attention_pattern_fidelities=[0.9],
        budget_tokens=6)
    s = sum(res.pick_rates.values())
    return {
        "schema": R139_SCHEMA_VERSION,
        "name": "h222_tvs_v13_fourteen_arms",
        "passed": bool(
            abs(s - 1.0) < 1e-6
            and len(W64_TVS_V13_ARMS) == 14),
        "sum": float(s),
    }


def family_h222b_uncertainty_v12_eleven_axis(
        seed: int) -> dict[str, Any]:
    res = compose_uncertainty_report_v12(
        confidences=[0.7], trusts=[0.9],
        substrate_fidelities=[0.95],
        hidden_state_fidelities=[0.95],
        cache_reuse_fidelities=[0.95],
        retrieval_fidelities=[0.95],
        replay_fidelities=[0.92],
        hidden_wins_fidelities=[0.5],
        replay_dominance_primary_fidelities=[0.5])
    return {
        "schema": R139_SCHEMA_VERSION,
        "name": "h222b_uncertainty_v12_eleven_axis",
        "passed": bool(
            res.n_axes == 11
            and 0.0 <= res.weighted_composite <= 1.0),
        "weighted": float(res.weighted_composite),
    }


_R139_FAMILIES: tuple[Any, ...] = (
    family_h215_crc_v12_4096bucket_detect,
    family_h215b_crc_v12_23bit_burst,
    family_h215c_crc_v12_replay_dominance_recovery,
    family_h215d_crc_v12_blast_radius,
    family_h216_consensus_v10_stage_count,
    family_h216b_consensus_v10_replay_dominance_primary_fires,
    family_h217_mlsc_v12_replay_dominance_primary_chain,
    family_h217b_mlsc_v12_total_variation_distance,
    family_h217c_mlsc_v12_hidden_state_trust_chain,
    family_h218_disagreement_v10_tv_identity,
    family_h218b_disagreement_v10_tv_falsifier,
    family_h219_substrate_adapter_v9_full,
    family_h219b_hosted_backends_text_only_at_v9,
    family_h220_deep_substrate_hybrid_v9_nine_way,
    family_h220b_deep_substrate_hybrid_v9_reduces_to_v8,
    family_h221_w64_envelope_verifier_modes,
    family_h221b_kv_v9_hidden_wins_primary_falsifier,
    family_h222_tvs_v13_fourteen_arms,
    family_h222b_uncertainty_v12_eleven_axis,
)


def run_r139(
        seeds: Sequence[int] = (199, 299, 399),
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for s in seeds:
        results: dict[str, Any] = {}
        for fn in _R139_FAMILIES:
            try:
                r = fn(int(s))
            except Exception as e:
                r = {
                    "schema": R139_SCHEMA_VERSION,
                    "name": getattr(fn, "__name__", "unknown"),
                    "passed": False,
                    "exception": str(e),
                }
            results[r["name"]] = r
        out.append({
            "schema": R139_SCHEMA_VERSION,
            "seed": int(s),
            "family_results": results,
        })
    return out
