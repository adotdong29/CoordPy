"""W67 R-151 benchmark family — corruption V15 / disagreement V13 /
consensus V13 / branch-merge / role-dropout.

H298..H306 cell families (18 H-bars):

* H298   CRC V15 32768-bucket fingerprint single-byte detect rate
         ≥ 0.95
* H298b  CRC V15 35-bit adversarial burst detect rate ≥ 0.4
* H298c  CRC V15 branch-merge reconciliation ratio floor ≥ 0.4
* H299   Consensus V13 has 20 stages
* H299b  Consensus V13 role_dropout_arbiter fires on positive
         scores
* H299c  Consensus V13 branch_merge_reconciliation_arbiter fires
         on conflicting branches
* H300   Disagreement Algebra V13 Bregman-equivalence identity
         holds on identical probes
* H300b  Disagreement Algebra V13 Bregman falsifier triggers on
         divergent probes
* H300c  Disagreement Algebra V13 squared-L2 Bregman of (a, a) = 0
* H301   MASC V3 V12 strictly beats V11 in team_consensus_under_
         budget regime ≥ 50% seeds
* H301b  MASC V3 V12 strictly beats V11 in team_failure_recovery
         regime ≥ 50% seeds
* H301c  MASC V3 V12 strictly beats V11 in role_dropout regime
         ≥ 50% seeds
* H301d  MASC V3 V12 strictly beats V11 in
         branch_merge_reconciliation regime ≥ 50% seeds
* H302   TVS V16 17 arms and pick rates sum to 1.0 under non-zero
         branch-merge fidelity
* H302b  TVS V16 branch_merge_reconciliation arm fires
* H303   Uncertainty V15 14-axis composite in [0, 1]
* H303b  Uncertainty V15 branch-merge-aware flag flips under
         non-unity fidelity
* H304   MLSC V15 wrap_v14_as_v15 records both new chains
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as _np

from coordpy.consensus_fallback_controller_v13 import (
    ConsensusFallbackControllerV13,
    W67_CONSENSUS_V13_STAGES,
    W67_CONSENSUS_V13_STAGE_BRANCH_MERGE_RECONCILIATION_ARBITER,
    W67_CONSENSUS_V13_STAGE_ROLE_DROPOUT_ARBITER,
    emit_consensus_v13_witness,
)
from coordpy.corruption_robust_carrier_v15 import (
    CorruptionRobustCarrierV15,
    emit_corruption_robustness_v15_witness,
)
from coordpy.disagreement_algebra import AlgebraTrace
from coordpy.disagreement_algebra_v13 import (
    bregman_divergence_squared_l2,
    emit_disagreement_algebra_v13_witness,
)
from coordpy.mergeable_latent_capsule_v3 import (
    make_root_capsule_v3,
)
from coordpy.mergeable_latent_capsule_v4 import wrap_v3_as_v4
from coordpy.mergeable_latent_capsule_v5 import wrap_v4_as_v5
from coordpy.mergeable_latent_capsule_v6 import wrap_v5_as_v6
from coordpy.mergeable_latent_capsule_v7 import wrap_v6_as_v7
from coordpy.mergeable_latent_capsule_v8 import wrap_v7_as_v8
from coordpy.mergeable_latent_capsule_v9 import wrap_v8_as_v9
from coordpy.mergeable_latent_capsule_v10 import wrap_v9_as_v10
from coordpy.mergeable_latent_capsule_v11 import wrap_v10_as_v11
from coordpy.mergeable_latent_capsule_v12 import wrap_v11_as_v12
from coordpy.mergeable_latent_capsule_v13 import wrap_v12_as_v13
from coordpy.mergeable_latent_capsule_v14 import wrap_v13_as_v14
from coordpy.mergeable_latent_capsule_v15 import (
    W67_MLSC_V15_ALGEBRA_BRANCH_MERGE_RECONCILIATION_PROPAGATION,
    emit_mlsc_v15_witness,
    wrap_v14_as_v15,
)
from coordpy.multi_agent_substrate_coordinator_v2 import (
    W66_MASC_V2_REGIME_TEAM_CONSENSUS_UNDER_BUDGET,
    W66_MASC_V2_REGIME_TEAM_FAILURE_RECOVERY,
)
from coordpy.multi_agent_substrate_coordinator_v3 import (
    MultiAgentSubstrateCoordinatorV3,
    W67_MASC_V3_REGIME_BRANCH_MERGE_RECONCILIATION,
    W67_MASC_V3_REGIME_ROLE_DROPOUT,
)
from coordpy.transcript_vs_shared_arbiter_v16 import (
    W67_TVS_V16_ARMS,
    W67_TVS_V16_ARM_BRANCH_MERGE_RECONCILIATION,
    seventeen_arm_compare,
)
from coordpy.uncertainty_layer_v15 import (
    compose_uncertainty_report_v15,
)


R151_SCHEMA_VERSION: str = "coordpy.r151_benchmark.v1"


def family_h298_crc_v15_kv32768_detect(
        seed: int) -> dict[str, Any]:
    crc = CorruptionRobustCarrierV15()
    w = emit_corruption_robustness_v15_witness(
        crc_v15=crc, n_probes=32, seed=int(seed) + 51000)
    return {
        "schema": R151_SCHEMA_VERSION,
        "name": "h298_crc_v15_kv32768_detect",
        "passed": bool(
            w.kv32768_corruption_detect_rate >= 0.95),
        "rate": float(w.kv32768_corruption_detect_rate),
    }


def family_h298b_crc_v15_35bit_burst(
        seed: int) -> dict[str, Any]:
    crc = CorruptionRobustCarrierV15()
    w = emit_corruption_robustness_v15_witness(
        crc_v15=crc, n_probes=32, seed=int(seed) + 51010)
    return {
        "schema": R151_SCHEMA_VERSION,
        "name": "h298b_crc_v15_35bit_burst",
        "passed": bool(
            w.adversarial_35bit_burst_detect_rate >= 0.4),
        "rate": float(w.adversarial_35bit_burst_detect_rate),
    }


def family_h298c_crc_v15_branch_merge_ratio(
        seed: int) -> dict[str, Any]:
    crc = CorruptionRobustCarrierV15()
    w = emit_corruption_robustness_v15_witness(
        crc_v15=crc, n_probes=16, seed=int(seed) + 51020)
    return {
        "schema": R151_SCHEMA_VERSION,
        "name": "h298c_crc_v15_branch_merge_ratio",
        "passed": bool(
            w.branch_merge_reconciliation_ratio_floor >= 0.4),
        "ratio_floor": float(
            w.branch_merge_reconciliation_ratio_floor),
    }


def family_h299_consensus_v13_n_stages(
        seed: int) -> dict[str, Any]:
    return {
        "schema": R151_SCHEMA_VERSION,
        "name": "h299_consensus_v13_n_stages",
        "passed": bool(len(W67_CONSENSUS_V13_STAGES) == 20),
        "n_stages": int(len(W67_CONSENSUS_V13_STAGES)),
    }


def family_h299b_consensus_v13_role_dropout_arbiter(
        seed: int) -> dict[str, Any]:
    ctrl = ConsensusFallbackControllerV13.init()
    res = ctrl.decide_v13(
        payloads=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        trusts=[0.2, 0.2],
        replay_decisions=["choose_abstain", "choose_abstain"],
        transcript_available=False,
        role_dropout_scores_per_parent=[0.6, 0.5],
        team_failure_recovery_scores_per_parent=[0.1, 0.1],
        team_consensus_under_budget_scores_per_parent=[0.1, 0.1],
        visible_token_budget_frac=0.3,
        team_substrate_coordination_scores_per_parent=[0.1, 0.1],
        multi_agent_abstain_score=0.4,
        corruption_detected_per_parent=[False, False],
        repair_amount=0.0,
        hidden_wins_margins_per_parent=[0.0, 0.0],
        three_way_predictions_per_parent=["kv_wins", "kv_wins"],
        replay_dominance_primary_scores_per_parent=[0.0, 0.0],
        four_way_predictions_per_parent=[
            "kv_wins", "kv_wins"])
    return {
        "schema": R151_SCHEMA_VERSION,
        "name": "h299b_consensus_v13_role_dropout_arbiter",
        "passed": bool(
            res.get("stage")
            == W67_CONSENSUS_V13_STAGE_ROLE_DROPOUT_ARBITER),
        "stage": str(res.get("stage", "")),
    }


def family_h299c_consensus_v13_branch_merge_arbiter(
        seed: int) -> dict[str, Any]:
    ctrl = ConsensusFallbackControllerV13.init()
    res = ctrl.decide_v13(
        payloads=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        trusts=[0.2, 0.2],
        replay_decisions=["choose_abstain", "choose_abstain"],
        transcript_available=False,
        role_dropout_scores_per_parent=[0.0, 0.0],
        branch_merge_scores_per_parent=[0.7, 0.55],
        n_conflicting_branches=2,
        team_failure_recovery_scores_per_parent=[0.1, 0.1],
        team_consensus_under_budget_scores_per_parent=[0.1, 0.1],
        visible_token_budget_frac=0.3,
        team_substrate_coordination_scores_per_parent=[0.1, 0.1],
        multi_agent_abstain_score=0.4,
        corruption_detected_per_parent=[False, False],
        repair_amount=0.0,
        hidden_wins_margins_per_parent=[0.0, 0.0],
        three_way_predictions_per_parent=["kv_wins", "kv_wins"],
        replay_dominance_primary_scores_per_parent=[0.0, 0.0],
        four_way_predictions_per_parent=[
            "kv_wins", "kv_wins"])
    return {
        "schema": R151_SCHEMA_VERSION,
        "name": "h299c_consensus_v13_branch_merge_arbiter",
        "passed": bool(
            res.get("stage")
            == (W67_CONSENSUS_V13_STAGE_BRANCH_MERGE_RECONCILIATION_ARBITER)),
        "stage": str(res.get("stage", "")),
    }


def family_h300_da_v13_bregman_identity(
        seed: int) -> dict[str, Any]:
    trace = AlgebraTrace(steps=[])
    probe = [0.1, 0.2, 0.3]
    w = emit_disagreement_algebra_v13_witness(
        trace=trace, probe_a=probe, probe_b=probe, probe_c=probe,
        tv_oracle=lambda: (True, 0.05),
        wasserstein_oracle=lambda: (True, 0.1),
        js_oracle=lambda: (True, 0.05),
        bregman_oracle=lambda: (True, 0.05))
    return {
        "schema": R151_SCHEMA_VERSION,
        "name": "h300_da_v13_bregman_identity",
        "passed": bool(w.bregman_equiv_ok),
    }


def family_h300b_da_v13_bregman_falsifier(
        seed: int) -> dict[str, Any]:
    trace = AlgebraTrace(steps=[])
    probe = [0.1, 0.2, 0.3]
    w = emit_disagreement_algebra_v13_witness(
        trace=trace, probe_a=probe, probe_b=probe, probe_c=probe,
        tv_oracle=lambda: (True, 0.05),
        wasserstein_oracle=lambda: (True, 0.1),
        js_oracle=lambda: (True, 0.05),
        bregman_falsifier_oracle=lambda: (False, 1.0))
    return {
        "schema": R151_SCHEMA_VERSION,
        "name": "h300b_da_v13_bregman_falsifier",
        "passed": bool(w.bregman_falsifier_ok),
    }


def family_h300c_da_v13_bregman_self_zero(
        seed: int) -> dict[str, Any]:
    probe = [0.1, 0.2, 0.3]
    br = bregman_divergence_squared_l2(probe, probe)
    return {
        "schema": R151_SCHEMA_VERSION,
        "name": "h300c_da_v13_bregman_self_zero",
        "passed": bool(abs(br) < 1e-12),
    }


def family_h301_masc_v3_v12_beats_v11_tcub(
        seed: int) -> dict[str, Any]:
    masc = MultiAgentSubstrateCoordinatorV3()
    _, agg = masc.run_batch(
        seeds=list(range(15)),
        regime=W66_MASC_V2_REGIME_TEAM_CONSENSUS_UNDER_BUDGET)
    return {
        "schema": R151_SCHEMA_VERSION,
        "name": "h301_masc_v3_v12_beats_v11_tcub",
        "passed": bool(agg.v12_beats_v11_rate >= 0.5),
        "rate": float(agg.v12_beats_v11_rate),
    }


def family_h301b_masc_v3_v12_beats_v11_tfr(
        seed: int) -> dict[str, Any]:
    masc = MultiAgentSubstrateCoordinatorV3()
    _, agg = masc.run_batch(
        seeds=list(range(15)),
        regime=W66_MASC_V2_REGIME_TEAM_FAILURE_RECOVERY)
    return {
        "schema": R151_SCHEMA_VERSION,
        "name": "h301b_masc_v3_v12_beats_v11_tfr",
        "passed": bool(agg.v12_beats_v11_rate >= 0.5),
        "rate": float(agg.v12_beats_v11_rate),
    }


def family_h301c_masc_v3_v12_beats_v11_role_dropout(
        seed: int) -> dict[str, Any]:
    masc = MultiAgentSubstrateCoordinatorV3()
    _, agg = masc.run_batch(
        seeds=list(range(15)),
        regime=W67_MASC_V3_REGIME_ROLE_DROPOUT)
    return {
        "schema": R151_SCHEMA_VERSION,
        "name": "h301c_masc_v3_v12_beats_v11_role_dropout",
        "passed": bool(agg.v12_beats_v11_rate >= 0.5),
        "rate": float(agg.v12_beats_v11_rate),
    }


def family_h301d_masc_v3_v12_beats_v11_branch_merge(
        seed: int) -> dict[str, Any]:
    masc = MultiAgentSubstrateCoordinatorV3()
    _, agg = masc.run_batch(
        seeds=list(range(15)),
        regime=W67_MASC_V3_REGIME_BRANCH_MERGE_RECONCILIATION)
    return {
        "schema": R151_SCHEMA_VERSION,
        "name": "h301d_masc_v3_v12_beats_v11_branch_merge",
        "passed": bool(agg.v12_beats_v11_rate >= 0.5),
        "rate": float(agg.v12_beats_v11_rate),
    }


def family_h302_tvs_v16_pick_rates(
        seed: int) -> dict[str, Any]:
    res = seventeen_arm_compare(
        per_turn_branch_merge_reconciliation_fidelities=[0.7],
        per_turn_team_failure_recovery_fidelities=[0.6],
        per_turn_team_substrate_coordination_fidelities=[0.7],
        per_turn_replay_dominance_primary_fidelities=[0.7],
        per_turn_hidden_wins_fidelities=[0.6],
        per_turn_replay_dominance_fidelities=[0.7],
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
        "schema": R151_SCHEMA_VERSION,
        "name": "h302_tvs_v16_pick_rates",
        "passed": bool(
            len(W67_TVS_V16_ARMS) == 17
            and abs(s - 1.0) < 1e-9),
        "pick_sum": float(s),
    }


def family_h302b_tvs_v16_branch_merge_arm(
        seed: int) -> dict[str, Any]:
    res = seventeen_arm_compare(
        per_turn_branch_merge_reconciliation_fidelities=[0.8],
        per_turn_team_failure_recovery_fidelities=[0.0],
        per_turn_team_substrate_coordination_fidelities=[0.0],
        per_turn_replay_dominance_primary_fidelities=[0.0],
        per_turn_hidden_wins_fidelities=[0.0],
        per_turn_replay_dominance_fidelities=[0.0],
        per_turn_confidences=[0.5],
        per_turn_trust_scores=[0.5],
        per_turn_merge_retentions=[0.0],
        per_turn_tw_retentions=[0.0],
        per_turn_substrate_fidelities=[0.0],
        per_turn_hidden_fidelities=[0.0],
        per_turn_cache_fidelities=[0.0],
        per_turn_retrieval_fidelities=[0.0],
        per_turn_replay_fidelities=[0.0],
        per_turn_attention_pattern_fidelities=[0.0],
        budget_tokens=6)
    return {
        "schema": R151_SCHEMA_VERSION,
        "name": "h302b_tvs_v16_branch_merge_arm",
        "passed": bool(
            res.branch_merge_reconciliation_used
            and res.pick_rates.get(
                W67_TVS_V16_ARM_BRANCH_MERGE_RECONCILIATION, 0.0
                ) > 0.0),
    }


def family_h303_uncertainty_v15_composite_range(
        seed: int) -> dict[str, Any]:
    comp = compose_uncertainty_report_v15(
        confidences=[0.7, 0.5],
        trusts=[0.9, 0.8],
        substrate_fidelities=[0.95, 0.9],
        hidden_state_fidelities=[0.95, 0.92],
        cache_reuse_fidelities=[0.95, 0.92],
        retrieval_fidelities=[0.95, 0.93],
        replay_fidelities=[0.92, 0.90],
        attention_pattern_fidelities=[0.95, 0.90],
        replay_dominance_fidelities=[0.88, 0.85],
        hidden_wins_fidelities=[0.86, 0.82],
        replay_dominance_primary_fidelities=[0.84, 0.80],
        team_coordination_fidelities=[0.80, 0.75],
        team_failure_recovery_fidelities=[0.78, 0.74],
        branch_merge_reconciliation_fidelities=[0.76, 0.72])
    val = float(comp.weighted_composite)
    return {
        "schema": R151_SCHEMA_VERSION,
        "name": "h303_uncertainty_v15_composite_range",
        "passed": bool(
            0.0 <= val <= 1.0 and comp.n_axes == 14),
        "value": float(val),
    }


def family_h303b_uncertainty_v15_branch_merge_aware(
        seed: int) -> dict[str, Any]:
    comp = compose_uncertainty_report_v15(
        confidences=[0.7, 0.5],
        trusts=[0.9, 0.8],
        substrate_fidelities=[1.0, 1.0],
        hidden_state_fidelities=[1.0, 1.0],
        cache_reuse_fidelities=[1.0, 1.0],
        retrieval_fidelities=[1.0, 1.0],
        replay_fidelities=[1.0, 1.0],
        attention_pattern_fidelities=[1.0, 1.0],
        replay_dominance_fidelities=[1.0, 1.0],
        hidden_wins_fidelities=[1.0, 1.0],
        replay_dominance_primary_fidelities=[1.0, 1.0],
        team_coordination_fidelities=[1.0, 1.0],
        team_failure_recovery_fidelities=[1.0, 1.0],
        branch_merge_reconciliation_fidelities=[0.8, 0.9])
    return {
        "schema": R151_SCHEMA_VERSION,
        "name": "h303b_uncertainty_v15_branch_merge_aware",
        "passed": bool(comp.branch_merge_aware),
    }


def family_h304_mlsc_v15_chains(
        seed: int) -> dict[str, Any]:
    v3 = make_root_capsule_v3(
        branch_id="r151", payload=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
        fact_tags=("r151",), confidence=0.9, trust=0.9,
        turn_index=int(seed))
    v4 = wrap_v3_as_v4(v3)
    v5 = wrap_v4_as_v5(v4, attention_witness_cid="a")
    v6 = wrap_v5_as_v6(
        v5, attention_witness_chain=("a_chain",),
        cache_reuse_witness_cid="c")
    v7 = wrap_v6_as_v7(
        v6, retrieval_witness_chain=("r_chain",),
        controller_witness_cid="ctrl")
    v8 = wrap_v7_as_v8(
        v7, replay_witness_chain=("rep_chain",),
        substrate_witness_chain=("sub_chain",),
        provenance_trust_table={"backend_a": 0.9})
    v9 = wrap_v8_as_v9(
        v8, attention_pattern_witness_chain=("ap_chain",),
        cache_retrieval_witness_chain=("cr_chain",),
        per_layer_head_trust_matrix=((0, 0, 0.9),))
    v10 = wrap_v9_as_v10(
        v9, replay_dominance_witness_chain=("rd_chain",),
        disagreement_wasserstein_distance=0.05)
    v11 = wrap_v10_as_v11(
        v10, hidden_wins_witness_chain=("hw_chain",))
    v12 = wrap_v11_as_v12(
        v11,
        replay_dominance_primary_witness_chain=("rdp_chain",),
        hidden_state_trust_witness_chain=("hst_chain",))
    v13 = wrap_v12_as_v13(
        v12,
        team_substrate_witness_chain=("ts_chain",),
        role_conditioned_witness_chain=("rc_chain",))
    v14 = wrap_v13_as_v14(
        v13,
        team_failure_recovery_witness_chain=("tfr_chain",),
        team_consensus_under_budget_witness_chain=("tcb_chain",))
    v15 = wrap_v14_as_v15(
        v14,
        role_dropout_recovery_witness_chain=("rdr_chain",),
        branch_merge_reconciliation_witness_chain=(
            "bmr_chain",),
        algebra_signature_v15=(
            W67_MLSC_V15_ALGEBRA_BRANCH_MERGE_RECONCILIATION_PROPAGATION))
    w = emit_mlsc_v15_witness(v15)
    return {
        "schema": R151_SCHEMA_VERSION,
        "name": "h304_mlsc_v15_chains",
        "passed": bool(
            w.role_dropout_recovery_chain_depth == 1
            and w.branch_merge_reconciliation_chain_depth == 1),
    }


_R151_FAMILIES: tuple[Any, ...] = (
    family_h298_crc_v15_kv32768_detect,
    family_h298b_crc_v15_35bit_burst,
    family_h298c_crc_v15_branch_merge_ratio,
    family_h299_consensus_v13_n_stages,
    family_h299b_consensus_v13_role_dropout_arbiter,
    family_h299c_consensus_v13_branch_merge_arbiter,
    family_h300_da_v13_bregman_identity,
    family_h300b_da_v13_bregman_falsifier,
    family_h300c_da_v13_bregman_self_zero,
    family_h301_masc_v3_v12_beats_v11_tcub,
    family_h301b_masc_v3_v12_beats_v11_tfr,
    family_h301c_masc_v3_v12_beats_v11_role_dropout,
    family_h301d_masc_v3_v12_beats_v11_branch_merge,
    family_h302_tvs_v16_pick_rates,
    family_h302b_tvs_v16_branch_merge_arm,
    family_h303_uncertainty_v15_composite_range,
    family_h303b_uncertainty_v15_branch_merge_aware,
    family_h304_mlsc_v15_chains,
)


def run_r151(
        seeds: Sequence[int] = (199, 299, 399),
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for s in seeds:
        results: dict[str, Any] = {}
        for fn in _R151_FAMILIES:
            try:
                r = fn(int(s))
            except Exception as e:
                r = {
                    "schema": R151_SCHEMA_VERSION,
                    "name": getattr(fn, "__name__", "unknown"),
                    "passed": False,
                    "exception": str(e),
                }
            results[r["name"]] = r
        out.append({
            "schema": R151_SCHEMA_VERSION,
            "seed": int(s),
            "family_results": results,
        })
    return out


__all__ = ["R151_SCHEMA_VERSION", "run_r151"]
