"""W65 R-145 benchmark family — corruption / consensus V11 /
disagreement V11 / uncertainty V13 / TVS V14 / W65 envelope.

H237..H244b cell families (16 H-bars).

* H237   CRC V13 8192-bucket detect rate ≥ 0.95
* H237b  CRC V13 31-bit adversarial burst detect rate ≥ 0.4
* H237c  CRC V13 team-coordination recovery ratio mean ≥ 0.4
* H238   Consensus V11 has 16 disjoint stages
* H238b  Consensus V11 team_substrate_coordination_arbiter
         stage fires
* H238c  Consensus V11 multi_agent_abstain_arbiter stage fires
         under high abstain score
* H239   Disagreement V11 Wasserstein identity holds
* H239b  Disagreement V11 Wasserstein falsifier triggers
* H240   Uncertainty V13 12-axis composite in [0, 1]
* H240b  Uncertainty V13 team_coordination_aware flips True under
         < 1.0 fidelity
* H241   TVS V14 fifteen arms sum to 1.0
* H241b  TVS V14 team_substrate_coordination arm fires under
         non-zero fidelity
* H242   W65 envelope verifier knows ≥ 100 failure modes
* H242b  W65 envelope chain-of-1 step verifies clean
* H243   MLSC V13 team_substrate chain inherits as union at merge
* H243b  MLSC V13 role_conditioned chain inherits as union at
         merge
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as _np

from coordpy.consensus_fallback_controller_v11 import (
    ConsensusFallbackControllerV11,
    W65_CONSENSUS_V11_STAGES,
    W65_CONSENSUS_V11_STAGE_MULTI_AGENT_ABSTAIN_ARBITER,
    W65_CONSENSUS_V11_STAGE_TEAM_SUBSTRATE_COORDINATION_ARBITER,
)
from coordpy.corruption_robust_carrier_v13 import (
    CorruptionRobustCarrierV13,
    emit_corruption_robustness_v13_witness,
)
from coordpy.disagreement_algebra_v11 import (
    check_wasserstein_equivalence_identity,
    wasserstein_equivalence_falsifier,
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
from coordpy.mergeable_latent_capsule_v13 import (
    MergeOperatorV13, wrap_v12_as_v13,
)
from coordpy.transcript_vs_shared_arbiter_v14 import (
    W65_TVS_V14_ARMS, fifteen_arm_compare,
)
from coordpy.uncertainty_layer_v13 import (
    compose_uncertainty_report_v13,
)
from coordpy.w65_team import (
    W65_ENVELOPE_VERIFIER_FAILURE_MODES,
    W65Params, W65Team, verify_w65_handoff,
)


R145_SCHEMA_VERSION: str = "coordpy.r145_benchmark.v1"


def family_h237_crc_v13_8192_detect(
        seed: int) -> dict[str, Any]:
    crc = CorruptionRobustCarrierV13()
    w = emit_corruption_robustness_v13_witness(
        crc_v13=crc, n_probes=24, seed=int(seed) + 45000)
    return {
        "schema": R145_SCHEMA_VERSION,
        "name": "h237_crc_v13_8192_detect",
        "passed": bool(
            w.kv8192_corruption_detect_rate >= 0.95),
        "rate": float(w.kv8192_corruption_detect_rate),
    }


def family_h237b_crc_v13_31bit_burst(
        seed: int) -> dict[str, Any]:
    crc = CorruptionRobustCarrierV13()
    w = emit_corruption_robustness_v13_witness(
        crc_v13=crc, n_probes=24, seed=int(seed) + 45100)
    return {
        "schema": R145_SCHEMA_VERSION,
        "name": "h237b_crc_v13_31bit_burst",
        "passed": bool(
            w.adversarial_31bit_burst_detect_rate >= 0.4),
        "rate": float(w.adversarial_31bit_burst_detect_rate),
    }


def family_h237c_crc_v13_team_coord_recovery(
        seed: int) -> dict[str, Any]:
    crc = CorruptionRobustCarrierV13()
    w = emit_corruption_robustness_v13_witness(
        crc_v13=crc, n_probes=24, seed=int(seed) + 45200)
    return {
        "schema": R145_SCHEMA_VERSION,
        "name": "h237c_crc_v13_team_coord_recovery",
        "passed": bool(
            w.team_coordination_recovery_ratio_mean >= 0.4),
        "mean": float(w.team_coordination_recovery_ratio_mean),
    }


def family_h238_consensus_v11_stage_count(
        seed: int) -> dict[str, Any]:
    return {
        "schema": R145_SCHEMA_VERSION,
        "name": "h238_consensus_v11_stage_count",
        "passed": bool(len(W65_CONSENSUS_V11_STAGES) == 16),
        "n_stages": int(len(W65_CONSENSUS_V11_STAGES)),
    }


def family_h238b_consensus_v11_team_substrate_stage(
        seed: int) -> dict[str, Any]:
    ctrl = ConsensusFallbackControllerV11.init()
    res = ctrl.decide_v11(
        payloads=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        trusts=[0.4, 0.4],
        replay_decisions=["choose_abstain", "choose_abstain"],
        transcript_available=False,
        team_substrate_coordination_scores_per_parent=[
            0.7, 0.4],
        multi_agent_abstain_score=0.0,
        corruption_detected_per_parent=[False, False],
        repair_amount=0.0,
        hidden_wins_margins_per_parent=[0.0, 0.0],
        three_way_predictions_per_parent=[
            "kv_wins", "kv_wins"],
        replay_dominance_primary_scores_per_parent=[0.0, 0.0],
        four_way_predictions_per_parent=[
            "kv_wins", "kv_wins"])
    return {
        "schema": R145_SCHEMA_VERSION,
        "name": "h238b_consensus_v11_team_substrate_stage",
        "passed": bool(
            res["stage"]
            == W65_CONSENSUS_V11_STAGE_TEAM_SUBSTRATE_COORDINATION_ARBITER),
        "stage": str(res.get("stage", "")),
    }


def family_h238c_consensus_v11_multi_agent_abstain_stage(
        seed: int) -> dict[str, Any]:
    ctrl = ConsensusFallbackControllerV11.init()
    res = ctrl.decide_v11(
        payloads=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        trusts=[0.4, 0.4],
        replay_decisions=["choose_abstain", "choose_abstain"],
        transcript_available=False,
        team_substrate_coordination_scores_per_parent=[
            0.0, 0.0],
        multi_agent_abstain_score=0.7,
        corruption_detected_per_parent=[False, False],
        repair_amount=0.0,
        hidden_wins_margins_per_parent=[0.0, 0.0],
        three_way_predictions_per_parent=[
            "kv_wins", "kv_wins"],
        replay_dominance_primary_scores_per_parent=[0.0, 0.0],
        four_way_predictions_per_parent=[
            "kv_wins", "kv_wins"])
    return {
        "schema": R145_SCHEMA_VERSION,
        "name": "h238c_consensus_v11_multi_agent_abstain_stage",
        "passed": bool(
            res["stage"]
            == W65_CONSENSUS_V11_STAGE_MULTI_AGENT_ABSTAIN_ARBITER),
        "stage": str(res.get("stage", "")),
    }


def family_h239_disagreement_v11_wasserstein_identity(
        seed: int) -> dict[str, Any]:
    ok = check_wasserstein_equivalence_identity(
        wasserstein_oracle=lambda: (True, 0.05),
        wasserstein_floor=0.20)
    return {
        "schema": R145_SCHEMA_VERSION,
        "name": "h239_disagreement_v11_wasserstein_identity",
        "passed": bool(ok),
    }


def family_h239b_disagreement_v11_wasserstein_falsifier(
        seed: int) -> dict[str, Any]:
    fals = wasserstein_equivalence_falsifier(
        wasserstein_oracle=lambda: (False, 1.0),
        wasserstein_floor=0.20)
    return {
        "schema": R145_SCHEMA_VERSION,
        "name": "h239b_disagreement_v11_wasserstein_falsifier",
        "passed": bool(fals),
    }


def family_h240_uncertainty_v13_twelve_axis(
        seed: int) -> dict[str, Any]:
    res = compose_uncertainty_report_v13(
        confidences=[0.7], trusts=[0.9],
        substrate_fidelities=[0.95],
        hidden_state_fidelities=[0.95],
        cache_reuse_fidelities=[0.95],
        retrieval_fidelities=[0.95],
        replay_fidelities=[0.92],
        team_coordination_fidelities=[0.8])
    return {
        "schema": R145_SCHEMA_VERSION,
        "name": "h240_uncertainty_v13_twelve_axis",
        "passed": bool(
            res.n_axes == 12
            and 0.0 <= res.weighted_composite <= 1.0),
        "weighted": float(res.weighted_composite),
    }


def family_h240b_uncertainty_v13_team_coord_aware(
        seed: int) -> dict[str, Any]:
    res_aware = compose_uncertainty_report_v13(
        confidences=[0.7], trusts=[0.9],
        substrate_fidelities=[0.95],
        hidden_state_fidelities=[0.95],
        cache_reuse_fidelities=[0.95],
        retrieval_fidelities=[0.95],
        replay_fidelities=[0.92],
        team_coordination_fidelities=[0.5])
    res_full = compose_uncertainty_report_v13(
        confidences=[0.7], trusts=[0.9],
        substrate_fidelities=[1.0],
        hidden_state_fidelities=[1.0],
        cache_reuse_fidelities=[1.0],
        retrieval_fidelities=[1.0],
        replay_fidelities=[1.0],
        team_coordination_fidelities=[1.0])
    return {
        "schema": R145_SCHEMA_VERSION,
        "name": "h240b_uncertainty_v13_team_coord_aware",
        "passed": bool(
            res_aware.team_coordination_aware
            and not res_full.team_coordination_aware),
    }


def family_h241_tvs_v14_fifteen_arms_sum_to_one(
        seed: int) -> dict[str, Any]:
    res = fifteen_arm_compare(
        per_turn_team_substrate_coordination_fidelities=[0.5],
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
        "schema": R145_SCHEMA_VERSION,
        "name": "h241_tvs_v14_fifteen_arms_sum_to_one",
        "passed": bool(
            abs(s - 1.0) < 1e-6
            and len(W65_TVS_V14_ARMS) == 15),
        "sum": float(s),
    }


def family_h241b_tvs_v14_team_substrate_arm_active(
        seed: int) -> dict[str, Any]:
    res = fifteen_arm_compare(
        per_turn_team_substrate_coordination_fidelities=[0.7],
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
    return {
        "schema": R145_SCHEMA_VERSION,
        "name": "h241b_tvs_v14_team_substrate_arm_active",
        "passed": bool(res.team_substrate_coordination_used),
    }


def family_h242_w65_envelope_verifier_modes(
        seed: int) -> dict[str, Any]:
    n = len(W65_ENVELOPE_VERIFIER_FAILURE_MODES)
    return {
        "schema": R145_SCHEMA_VERSION,
        "name": "h242_w65_envelope_verifier_modes",
        "passed": bool(n >= 100),
        "n_modes": int(n),
    }


def family_h242b_w65_envelope_chain_clean(
        seed: int) -> dict[str, Any]:
    p = W65Params.build_default(seed=int(seed) + 45100)
    team = W65Team(params=p)
    env = team.step(turn_index=0, role="planner",
                     w64_outer_cid="w64_r145")
    v = verify_w65_handoff(env)
    return {
        "schema": R145_SCHEMA_VERSION,
        "name": "h242b_w65_envelope_chain_clean",
        "passed": bool(
            v["ok"] and env.substrate_v10_used
            and env.ten_way_used),
        "n_failures": int(len(v["failures"])),
    }


def _build_v13_capsule(
        branch: str, *, ts_witness: str, rc_witness: str,
) -> Any:
    v3 = make_root_capsule_v3(
        branch_id=branch, payload=tuple([0.1] * 6),
        fact_tags=("w65",), confidence=0.9, trust=0.9,
        turn_index=0)
    v4 = wrap_v3_as_v4(v3)
    v5 = wrap_v4_as_v5(v4, attention_witness_cid="a")
    v6 = wrap_v5_as_v6(v5, attention_witness_chain=("ac",),
                       cache_reuse_witness_cid="c")
    v7 = wrap_v6_as_v7(v6, retrieval_witness_chain=("rc",),
                       controller_witness_cid="ctrl")
    v8 = wrap_v7_as_v8(
        v7, replay_witness_chain=("rc",),
        substrate_witness_chain=("sc",),
        provenance_trust_table={"a": 0.9})
    v9 = wrap_v8_as_v9(
        v8, attention_pattern_witness_chain=("ap",),
        cache_retrieval_witness_chain=("cr",),
        per_layer_head_trust_matrix=((0, 0, 0.9),))
    v10 = wrap_v9_as_v10(
        v9, replay_dominance_witness_chain=("rd",),
        disagreement_wasserstein_distance=0.05)
    v11 = wrap_v10_as_v11(
        v10, hidden_wins_witness_chain=("hw",))
    v12 = wrap_v11_as_v12(
        v11, replay_dominance_primary_witness_chain=("rdp",),
        hidden_state_trust_witness_chain=("hst",))
    return wrap_v12_as_v13(
        v12, team_substrate_witness_chain=(ts_witness,),
        role_conditioned_witness_chain=(rc_witness,))


def family_h243_mlsc_v13_team_substrate_chain(
        seed: int) -> dict[str, Any]:
    op = MergeOperatorV13(factor_dim=6)
    a = _build_v13_capsule("A", ts_witness="ts_a",
                            rc_witness="rc_a")
    b = _build_v13_capsule("B", ts_witness="ts_b",
                            rc_witness="rc_b")
    merged = op.merge([a, b])
    return {
        "schema": R145_SCHEMA_VERSION,
        "name": "h243_mlsc_v13_team_substrate_chain",
        "passed": bool(
            "ts_a" in merged.team_substrate_witness_chain
            and "ts_b" in merged.team_substrate_witness_chain),
    }


def family_h243b_mlsc_v13_role_conditioned_chain(
        seed: int) -> dict[str, Any]:
    op = MergeOperatorV13(factor_dim=6)
    a = _build_v13_capsule("A", ts_witness="ts_a",
                            rc_witness="rc_a")
    b = _build_v13_capsule("B", ts_witness="ts_b",
                            rc_witness="rc_b")
    merged = op.merge([a, b])
    return {
        "schema": R145_SCHEMA_VERSION,
        "name": "h243b_mlsc_v13_role_conditioned_chain",
        "passed": bool(
            "rc_a" in merged.role_conditioned_witness_chain
            and "rc_b" in merged.role_conditioned_witness_chain),
    }


_R145_FAMILIES: tuple[Any, ...] = (
    family_h237_crc_v13_8192_detect,
    family_h237b_crc_v13_31bit_burst,
    family_h237c_crc_v13_team_coord_recovery,
    family_h238_consensus_v11_stage_count,
    family_h238b_consensus_v11_team_substrate_stage,
    family_h238c_consensus_v11_multi_agent_abstain_stage,
    family_h239_disagreement_v11_wasserstein_identity,
    family_h239b_disagreement_v11_wasserstein_falsifier,
    family_h240_uncertainty_v13_twelve_axis,
    family_h240b_uncertainty_v13_team_coord_aware,
    family_h241_tvs_v14_fifteen_arms_sum_to_one,
    family_h241b_tvs_v14_team_substrate_arm_active,
    family_h242_w65_envelope_verifier_modes,
    family_h242b_w65_envelope_chain_clean,
    family_h243_mlsc_v13_team_substrate_chain,
    family_h243b_mlsc_v13_role_conditioned_chain,
)


def run_r145(
        seeds: Sequence[int] = (199, 299, 399),
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for s in seeds:
        results: dict[str, Any] = {}
        for fn in _R145_FAMILIES:
            try:
                r = fn(int(s))
            except Exception as e:
                r = {
                    "schema": R145_SCHEMA_VERSION,
                    "name": getattr(fn, "__name__", "unknown"),
                    "passed": False,
                    "exception": str(e),
                }
            results[r["name"]] = r
        out.append({
            "schema": R145_SCHEMA_VERSION,
            "seed": int(s),
            "family_results": results,
        })
    return out


__all__ = [
    "R145_SCHEMA_VERSION",
    "run_r145",
]
