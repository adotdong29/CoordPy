"""W66 R-148 benchmark family — corruption V14 / consensus V12 /
disagreement V12 / uncertainty V14 / TVS V15 / W66 envelope
verifier / MASC V2 regimes.

H260..H268 cell families (18 H-bars):

* H260   consensus V12 18 stages
* H260b  CRC V14 33-bit burst detect >= 0.40
* H260c  CRC V14 16384-bucket detect >= 0.95
* H260d  CRC V14 team-failure-recovery ratio floor >= 0.40
* H261   disagreement V12 JS-equivalence identity holds
* H261b  disagreement V12 JS falsifier triggers
* H262   uncertainty V14 13-axis composite in [0, 1]
* H262b  uncertainty V14 team_failure_recovery_aware True under
         partial inputs
* H263   MASC V2 team_consensus_under_budget regime V11 beats
         V10 >= 50%
* H263b  MASC V2 team_failure_recovery regime TSC_V11 beats V11
         >= 50%
* H263c  Team-consensus controller fires abstain on low-confidence
* H263d  Team-consensus controller fires substrate_replay
         fallback when quorum fails
* H264   TVS V15 16 arms sum to 1.0
* H264b  TVS V15 team_failure_recovery arm fires when fidelity > 0
* H265   W66 envelope verifier >= 120 disjoint modes
* H266   W66 envelope verifier passes on a built team
* H267   W66 trivial passthrough preserves w65 outer cid byte-
         for-byte
* H268   Cumulative trust boundary >= 1225
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as _np

from coordpy.consensus_fallback_controller_v12 import (
    ConsensusFallbackControllerV12,
    W66_CONSENSUS_V12_STAGES,
)
from coordpy.corruption_robust_carrier_v14 import (
    CorruptionRobustCarrierV14,
    emit_corruption_robustness_v14_witness,
)
from coordpy.disagreement_algebra import AlgebraTrace
from coordpy.disagreement_algebra_v12 import (
    emit_disagreement_algebra_v12_witness,
)
from coordpy.multi_agent_substrate_coordinator_v2 import (
    MultiAgentSubstrateCoordinatorV2,
    W66_MASC_V2_REGIME_TEAM_CONSENSUS_UNDER_BUDGET,
    W66_MASC_V2_REGIME_TEAM_FAILURE_RECOVERY,
)
from coordpy.team_consensus_controller import (
    TeamConsensusController,
    W66_TC_DECISION_ABSTAIN,
    W66_TC_DECISION_SUBSTRATE_REPLAY,
)
from coordpy.transcript_vs_shared_arbiter_v15 import (
    W66_TVS_V15_ARMS,
    emit_tvs_arbiter_v15_witness,
    sixteen_arm_compare,
)
from coordpy.uncertainty_layer_v14 import (
    compose_uncertainty_report_v14,
    emit_uncertainty_v14_witness,
)
from coordpy.w66_team import (
    W66_ENVELOPE_VERIFIER_FAILURE_MODES, W66Params, W66Team,
    verify_w66_handoff,
)


R148_SCHEMA_VERSION: str = "coordpy.r148_benchmark.v1"


def family_h260_consensus_v12_eighteen_stages(
        seed: int) -> dict[str, Any]:
    return {
        "schema": R148_SCHEMA_VERSION,
        "name": "h260_consensus_v12_eighteen_stages",
        "passed": bool(len(W66_CONSENSUS_V12_STAGES) == 18),
        "n_stages": int(len(W66_CONSENSUS_V12_STAGES)),
    }


def family_h260b_crc_v14_burst_detect(
        seed: int) -> dict[str, Any]:
    crc = CorruptionRobustCarrierV14()
    w = emit_corruption_robustness_v14_witness(
        crc_v14=crc, n_probes=24, seed=int(seed) + 48000)
    return {
        "schema": R148_SCHEMA_VERSION,
        "name": "h260b_crc_v14_burst_detect",
        "passed": bool(
            float(w.adversarial_33bit_burst_detect_rate)
            >= 0.40),
        "rate": float(w.adversarial_33bit_burst_detect_rate),
    }


def family_h260c_crc_v14_kv16384_detect(
        seed: int) -> dict[str, Any]:
    crc = CorruptionRobustCarrierV14()
    w = emit_corruption_robustness_v14_witness(
        crc_v14=crc, n_probes=24, seed=int(seed) + 48100)
    return {
        "schema": R148_SCHEMA_VERSION,
        "name": "h260c_crc_v14_kv16384_detect",
        "passed": bool(
            float(w.kv16384_corruption_detect_rate) >= 0.95),
        "rate": float(w.kv16384_corruption_detect_rate),
    }


def family_h260d_crc_v14_team_failure_recovery_ratio_floor(
        seed: int) -> dict[str, Any]:
    crc = CorruptionRobustCarrierV14()
    w = emit_corruption_robustness_v14_witness(
        crc_v14=crc, n_probes=24, seed=int(seed) + 48200)
    return {
        "schema": R148_SCHEMA_VERSION,
        "name": (
            "h260d_crc_v14_team_failure_recovery_ratio_floor"),
        "passed": bool(
            float(w.team_failure_recovery_ratio_floor)
            >= 0.40),
        "floor": float(w.team_failure_recovery_ratio_floor),
    }


def family_h261_da_v12_js_identity(
        seed: int) -> dict[str, Any]:
    trace = AlgebraTrace(steps=[])
    probe = [0.1, 0.2, 0.3]
    w = emit_disagreement_algebra_v12_witness(
        trace=trace, probe_a=probe, probe_b=probe,
        probe_c=probe,
        tv_oracle=lambda: (True, 0.05),
        wasserstein_oracle=lambda: (True, 0.1),
        js_oracle=lambda: (True, 0.05),
        attention_pattern_oracle=lambda: (True, 0.85))
    return {
        "schema": R148_SCHEMA_VERSION,
        "name": "h261_da_v12_js_identity",
        "passed": bool(w.js_equiv_ok),
    }


def family_h261b_da_v12_js_falsifier(
        seed: int) -> dict[str, Any]:
    trace = AlgebraTrace(steps=[])
    probe = [0.1, 0.2, 0.3]
    w = emit_disagreement_algebra_v12_witness(
        trace=trace, probe_a=probe, probe_b=probe,
        probe_c=probe,
        tv_falsifier_oracle=lambda: (False, 1.0),
        wasserstein_falsifier_oracle=lambda: (False, 1.0),
        js_falsifier_oracle=lambda: (False, 1.0),
        attention_pattern_oracle=lambda: (True, 0.85))
    return {
        "schema": R148_SCHEMA_VERSION,
        "name": "h261b_da_v12_js_falsifier",
        "passed": bool(w.js_falsifier_ok),
    }


def family_h262_uncertainty_v14_composite_in_range(
        seed: int) -> dict[str, Any]:
    unc = compose_uncertainty_report_v14(
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
        team_failure_recovery_fidelities=[0.78, 0.74])
    w = emit_uncertainty_v14_witness(unc)
    return {
        "schema": R148_SCHEMA_VERSION,
        "name": "h262_uncertainty_v14_composite_in_range",
        "passed": bool(
            0.0 <= float(unc.weighted_composite) <= 1.0
            and w.n_axes == 13),
        "composite": float(unc.weighted_composite),
    }


def family_h262b_uncertainty_v14_aware_under_partial(
        seed: int) -> dict[str, Any]:
    unc = compose_uncertainty_report_v14(
        confidences=[0.7, 0.5],
        trusts=[0.9, 0.8],
        substrate_fidelities=[0.95, 0.9],
        hidden_state_fidelities=[0.95, 0.92],
        cache_reuse_fidelities=[0.95, 0.92],
        retrieval_fidelities=[0.95, 0.93],
        replay_fidelities=[0.92, 0.90],
        team_failure_recovery_fidelities=[0.78, 0.74])
    return {
        "schema": R148_SCHEMA_VERSION,
        "name": (
            "h262b_uncertainty_v14_aware_under_partial"),
        "passed": bool(
            unc.team_failure_recovery_aware),
    }


def family_h263_masc_v2_tcub_v11_beats_v10(
        seed: int) -> dict[str, Any]:
    masc = MultiAgentSubstrateCoordinatorV2()
    _, agg = masc.run_batch(
        seeds=list(range(15)),
        regime=W66_MASC_V2_REGIME_TEAM_CONSENSUS_UNDER_BUDGET)
    return {
        "schema": R148_SCHEMA_VERSION,
        "name": "h263_masc_v2_tcub_v11_beats_v10",
        "passed": bool(agg.v11_beats_v10_rate >= 0.5),
        "rate": float(agg.v11_beats_v10_rate),
    }


def family_h263b_masc_v2_tfr_tsc_v11_beats_v11(
        seed: int) -> dict[str, Any]:
    masc = MultiAgentSubstrateCoordinatorV2()
    _, agg = masc.run_batch(
        seeds=list(range(15)),
        regime=W66_MASC_V2_REGIME_TEAM_FAILURE_RECOVERY)
    return {
        "schema": R148_SCHEMA_VERSION,
        "name": "h263b_masc_v2_tfr_tsc_v11_beats_v11",
        "passed": bool(agg.tsc_v11_beats_v11_rate >= 0.5),
        "rate": float(agg.tsc_v11_beats_v11_rate),
    }


def family_h263c_team_consensus_abstain_on_low_conf(
        seed: int) -> dict[str, Any]:
    tc = TeamConsensusController()
    res = tc.decide(
        regime=W66_MASC_V2_REGIME_TEAM_CONSENSUS_UNDER_BUDGET,
        agent_guesses=[0.5, 0.6, 0.55],
        agent_confidences=[0.1, 0.15, 0.05],
        substrate_replay_trust=0.1,
        transcript_available=False)
    return {
        "schema": R148_SCHEMA_VERSION,
        "name": "h263c_team_consensus_abstain_on_low_conf",
        "passed": bool(
            res["decision"] == W66_TC_DECISION_ABSTAIN),
        "decision": str(res["decision"]),
    }


def family_h263d_team_consensus_substrate_replay_fallback(
        seed: int) -> dict[str, Any]:
    tc = TeamConsensusController()
    res = tc.decide(
        regime=W66_MASC_V2_REGIME_TEAM_FAILURE_RECOVERY,
        agent_guesses=[0.5, 0.6, 0.55],
        agent_confidences=[0.1, 0.15, 0.05],
        substrate_replay_trust=0.7,
        transcript_available=False)
    return {
        "schema": R148_SCHEMA_VERSION,
        "name": (
            "h263d_team_consensus_substrate_replay_fallback"),
        "passed": bool(
            res["decision"] == W66_TC_DECISION_SUBSTRATE_REPLAY),
        "decision": str(res["decision"]),
    }


def family_h264_tvs_v15_sixteen_arm_sum(
        seed: int) -> dict[str, Any]:
    res = sixteen_arm_compare(
        per_turn_team_failure_recovery_fidelities=[0.6],
        per_turn_team_substrate_coordination_fidelities=[
            0.7],
        per_turn_replay_dominance_primary_fidelities=[
            0.7],
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
    s = float(sum(res.pick_rates.values()))
    return {
        "schema": R148_SCHEMA_VERSION,
        "name": "h264_tvs_v15_sixteen_arm_sum",
        "passed": bool(
            len(W66_TVS_V15_ARMS) == 16
            and abs(s - 1.0) < 1e-9),
        "sum": float(s),
    }


def family_h264b_tvs_v15_team_failure_recovery_arm(
        seed: int) -> dict[str, Any]:
    res = sixteen_arm_compare(
        per_turn_team_failure_recovery_fidelities=[0.9],
        per_turn_team_substrate_coordination_fidelities=[
            0.0],
        per_turn_replay_dominance_primary_fidelities=[
            0.0],
        per_turn_hidden_wins_fidelities=[0.0],
        per_turn_replay_dominance_fidelities=[0.0],
        per_turn_confidences=[0.5],
        per_turn_trust_scores=[0.5],
        per_turn_merge_retentions=[0.5],
        per_turn_tw_retentions=[0.5],
        per_turn_substrate_fidelities=[0.5],
        per_turn_hidden_fidelities=[0.5],
        per_turn_cache_fidelities=[0.5],
        per_turn_retrieval_fidelities=[0.5],
        per_turn_replay_fidelities=[0.5],
        per_turn_attention_pattern_fidelities=[0.5],
        budget_tokens=6)
    return {
        "schema": R148_SCHEMA_VERSION,
        "name": "h264b_tvs_v15_team_failure_recovery_arm",
        "passed": bool(res.team_failure_recovery_used),
    }


def family_h265_w66_envelope_failure_modes(
        seed: int) -> dict[str, Any]:
    return {
        "schema": R148_SCHEMA_VERSION,
        "name": "h265_w66_envelope_failure_modes",
        "passed": bool(
            len(W66_ENVELOPE_VERIFIER_FAILURE_MODES) >= 120),
        "n_modes": int(
            len(W66_ENVELOPE_VERIFIER_FAILURE_MODES)),
    }


def family_h266_w66_envelope_verifier_passes_on_team(
        seed: int) -> dict[str, Any]:
    p = W66Params.build_default(seed=int(seed) + 48400)
    team = W66Team(params=p)
    env = team.step(
        turn_index=0, role="planner",
        w65_outer_cid="r148_w65")
    v = verify_w66_handoff(env)
    return {
        "schema": R148_SCHEMA_VERSION,
        "name": "h266_w66_envelope_verifier_passes_on_team",
        "passed": bool(v["ok"]),
        "failures": list(v["failures"])[:5],
    }


def family_h267_w66_trivial_passthrough(
        seed: int) -> dict[str, Any]:
    p = W66Params.build_trivial()
    team = W66Team(params=p)
    target = f"r148_w65_outer_{int(seed)}"
    env = team.step(
        turn_index=0, role="planner",
        w65_outer_cid=str(target))
    return {
        "schema": R148_SCHEMA_VERSION,
        "name": "h267_w66_trivial_passthrough",
        "passed": bool(env.w65_outer_cid == target),
    }


def family_h268_cumulative_trust_boundary(
        seed: int) -> dict[str, Any]:
    # Pre-W66 cumulative was 1105. W66 envelope adds 123.
    cumulative = 1105 + int(
        len(W66_ENVELOPE_VERIFIER_FAILURE_MODES))
    return {
        "schema": R148_SCHEMA_VERSION,
        "name": "h268_cumulative_trust_boundary",
        "passed": bool(cumulative >= 1225),
        "cumulative": int(cumulative),
    }


_R148_FAMILIES: tuple[Any, ...] = (
    family_h260_consensus_v12_eighteen_stages,
    family_h260b_crc_v14_burst_detect,
    family_h260c_crc_v14_kv16384_detect,
    family_h260d_crc_v14_team_failure_recovery_ratio_floor,
    family_h261_da_v12_js_identity,
    family_h261b_da_v12_js_falsifier,
    family_h262_uncertainty_v14_composite_in_range,
    family_h262b_uncertainty_v14_aware_under_partial,
    family_h263_masc_v2_tcub_v11_beats_v10,
    family_h263b_masc_v2_tfr_tsc_v11_beats_v11,
    family_h263c_team_consensus_abstain_on_low_conf,
    family_h263d_team_consensus_substrate_replay_fallback,
    family_h264_tvs_v15_sixteen_arm_sum,
    family_h264b_tvs_v15_team_failure_recovery_arm,
    family_h265_w66_envelope_failure_modes,
    family_h266_w66_envelope_verifier_passes_on_team,
    family_h267_w66_trivial_passthrough,
    family_h268_cumulative_trust_boundary,
)


def run_r148(
        seeds: Sequence[int] = (199, 299, 399),
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for s in seeds:
        results: dict[str, Any] = {}
        for fn in _R148_FAMILIES:
            try:
                r = fn(int(s))
            except Exception as e:
                r = {
                    "schema": R148_SCHEMA_VERSION,
                    "name": getattr(fn, "__name__", "unknown"),
                    "passed": False,
                    "exception": str(e),
                }
            results[r["name"]] = r
        out.append({
            "schema": R148_SCHEMA_VERSION,
            "seed": int(s),
            "family_results": results,
        })
    return out


__all__ = [
    "R148_SCHEMA_VERSION",
    "run_r148",
]
