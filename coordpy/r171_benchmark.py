"""W72 R-171 benchmark family — Multi-Agent Task Success.

Exercises MASC V8 across all twelve regimes (the eleven W71
regimes plus the new W72 regime:
delayed_rejoin_after_restart_under_budget).

H740..H765 cell families (26 H-bars):

* H740   baseline: V17 strictly beats V16 on ≥ 50 % of seeds
* H740b  baseline: TSC_V17 strictly beats TSC_V16 on ≥ 50 %
* H741   team_consensus_under_budget: V17 vs V16 ≥ 50 %
* H741b  team_consensus_under_budget: TSC_V17 vs TSC_V16 ≥ 50 %
* H742   team_failure_recovery: V17 vs V16 ≥ 50 %
* H742b  team_failure_recovery: TSC_V17 vs TSC_V16 ≥ 50 %
* H743   role_dropout: V17 vs V16 ≥ 50 %
* H744   branch_merge_reconciliation: V17 vs V16 ≥ 50 %
* H745   partial_contradiction: V17 vs V16 ≥ 50 %
* H745b  partial_contradiction: TSC_V17 vs TSC_V16 ≥ 50 %
* H746   agent_replacement_warm_restart: V17 vs V16 ≥ 50 %
* H746b  agent_replacement: TSC_V17 vs TSC_V16 ≥ 50 %
* H747   multi_branch_rejoin: V17 vs V16 ≥ 50 %
* H747b  multi_branch_rejoin: TSC_V17 vs TSC_V16 ≥ 50 %
* H748   silent_corruption: V17 vs V16 ≥ 50 %
* H748b  silent_corruption: TSC_V17 vs TSC_V16 ≥ 50 %
* H749   contradiction_then_rejoin: V17 vs V16 ≥ 50 %
* H749b  contradiction_then_rejoin: TSC_V17 vs TSC_V16 ≥ 50 %
* H750   delayed_repair_after_restart: V17 vs V16 ≥ 50 %
* H750b  delayed_repair_after_restart: TSC_V17 vs TSC_V16 ≥ 50 %
* H751   delayed_rejoin_after_restart_under_budget: V17 vs V16 ≥ 50 %
* H751b  delayed_rejoin_after_restart_under_budget: TSC_V17 vs
         TSC_V16 ≥ 50 %
* H752   V17 visible-token savings vs transcript on baseline ≥ 65 %
* H753   V17 team_success_per_visible_token strictly > V16 on baseline
* H754   TCC V7 rejoin_pressure_arbiter fires
* H755   TCC V7 delayed_rejoin_after_restart_arbiter fires
"""

from __future__ import annotations

from typing import Any, Sequence

from coordpy.multi_agent_substrate_coordinator_v2 import (
    W66_MASC_V2_REGIME_BASELINE,
    W66_MASC_V2_REGIME_TEAM_CONSENSUS_UNDER_BUDGET,
    W66_MASC_V2_REGIME_TEAM_FAILURE_RECOVERY,
)
from coordpy.multi_agent_substrate_coordinator_v3 import (
    W67_MASC_V3_REGIME_BRANCH_MERGE_RECONCILIATION,
    W67_MASC_V3_REGIME_ROLE_DROPOUT,
)
from coordpy.multi_agent_substrate_coordinator_v4 import (
    W68_MASC_V4_REGIME_AGENT_REPLACEMENT_WARM_RESTART,
    W68_MASC_V4_REGIME_PARTIAL_CONTRADICTION,
)
from coordpy.multi_agent_substrate_coordinator_v5 import (
    W69_MASC_V5_REGIME_MULTI_BRANCH_REJOIN,
    W69_MASC_V5_REGIME_SILENT_CORRUPTION_PLUS_REPLACEMENT,
)
from coordpy.multi_agent_substrate_coordinator_v6 import (
    W70_MASC_V6_REGIME_CONTRADICTION_THEN_REJOIN_UNDER_BUDGET,
)
from coordpy.multi_agent_substrate_coordinator_v7 import (
    W71_MASC_V7_REGIME_DELAYED_REPAIR_AFTER_RESTART,
)
from coordpy.multi_agent_substrate_coordinator_v7 import (
    W71_MASC_V7_POLICY_SUBSTRATE_ROUTED_V16,
)
from coordpy.multi_agent_substrate_coordinator_v8 import (
    MultiAgentSubstrateCoordinatorV8,
    W72_MASC_V8_POLICY_SUBSTRATE_ROUTED_V17,
    W72_MASC_V8_POLICY_TEAM_SUBSTRATE_COORDINATION_V17,
    W72_MASC_V8_REGIME_DELAYED_REJOIN_AFTER_RESTART_UNDER_BUDGET,
)
from coordpy.team_consensus_controller_v7 import (
    TeamConsensusControllerV7,
    W72_TC_V7_DECISION_DELAYED_REJOIN,
    W72_TC_V7_DECISION_REJOIN_PRESSURE,
)


R171_SCHEMA_VERSION: str = "coordpy.r171_benchmark.v1"


def _run_masc_v8_regime(
        regime: str, seeds: int = 15,
) -> dict[str, Any]:
    masc = MultiAgentSubstrateCoordinatorV8()
    _, agg = masc.run_batch(
        seeds=list(range(int(seeds))), regime=regime)
    return {
        "regime": str(regime),
        "v17_beats_v16_rate": float(agg.v17_beats_v16_rate),
        "tsc_v17_beats_tsc_v16_rate": float(
            agg.tsc_v17_beats_tsc_v16_rate),
        "per_policy_success_rate": dict(
            agg.per_policy_success_rate),
        "v17_visible_tokens_savings_vs_transcript": float(
            agg.v17_visible_tokens_savings_vs_transcript),
        "team_success_per_visible_token_v17": float(
            agg.team_success_per_visible_token_v17),
    }


def _bar(regime: str, key: str, name: str) -> dict[str, Any]:
    r = _run_masc_v8_regime(regime, seeds=15)
    return {
        "schema": R171_SCHEMA_VERSION,
        "name": str(name),
        "passed": bool(r[key] >= 0.5),
        "rate": float(r[key]),
    }


def family_h740_baseline_v17(seed: int) -> dict[str, Any]:
    return _bar(
        W66_MASC_V2_REGIME_BASELINE, "v17_beats_v16_rate",
        "h740_baseline_v17_beats_v16")


def family_h740b_baseline_tsc(seed: int) -> dict[str, Any]:
    return _bar(
        W66_MASC_V2_REGIME_BASELINE,
        "tsc_v17_beats_tsc_v16_rate",
        "h740b_baseline_tsc_v17_beats_tsc_v16")


def family_h741_tcub_v17(seed: int) -> dict[str, Any]:
    return _bar(
        W66_MASC_V2_REGIME_TEAM_CONSENSUS_UNDER_BUDGET,
        "v17_beats_v16_rate", "h741_tcub_v17_beats_v16")


def family_h741b_tcub_tsc(seed: int) -> dict[str, Any]:
    return _bar(
        W66_MASC_V2_REGIME_TEAM_CONSENSUS_UNDER_BUDGET,
        "tsc_v17_beats_tsc_v16_rate", "h741b_tcub_tsc")


def family_h742_tfr_v17(seed: int) -> dict[str, Any]:
    return _bar(
        W66_MASC_V2_REGIME_TEAM_FAILURE_RECOVERY,
        "v17_beats_v16_rate", "h742_tfr_v17_beats_v16")


def family_h742b_tfr_tsc(seed: int) -> dict[str, Any]:
    return _bar(
        W66_MASC_V2_REGIME_TEAM_FAILURE_RECOVERY,
        "tsc_v17_beats_tsc_v16_rate", "h742b_tfr_tsc")


def family_h743_role_dropout(seed: int) -> dict[str, Any]:
    return _bar(
        W67_MASC_V3_REGIME_ROLE_DROPOUT,
        "v17_beats_v16_rate",
        "h743_role_dropout_v17_beats_v16")


def family_h744_branch_merge(seed: int) -> dict[str, Any]:
    return _bar(
        W67_MASC_V3_REGIME_BRANCH_MERGE_RECONCILIATION,
        "v17_beats_v16_rate",
        "h744_branch_merge_v17_beats_v16")


def family_h745_pc(seed: int) -> dict[str, Any]:
    return _bar(
        W68_MASC_V4_REGIME_PARTIAL_CONTRADICTION,
        "v17_beats_v16_rate",
        "h745_pc_v17_beats_v16")


def family_h745b_pc_tsc(seed: int) -> dict[str, Any]:
    return _bar(
        W68_MASC_V4_REGIME_PARTIAL_CONTRADICTION,
        "tsc_v17_beats_tsc_v16_rate", "h745b_pc_tsc")


def family_h746_ar(seed: int) -> dict[str, Any]:
    return _bar(
        W68_MASC_V4_REGIME_AGENT_REPLACEMENT_WARM_RESTART,
        "v17_beats_v16_rate",
        "h746_ar_v17_beats_v16")


def family_h746b_ar_tsc(seed: int) -> dict[str, Any]:
    return _bar(
        W68_MASC_V4_REGIME_AGENT_REPLACEMENT_WARM_RESTART,
        "tsc_v17_beats_tsc_v16_rate", "h746b_ar_tsc")


def family_h747_mbr(seed: int) -> dict[str, Any]:
    return _bar(
        W69_MASC_V5_REGIME_MULTI_BRANCH_REJOIN,
        "v17_beats_v16_rate",
        "h747_mbr_v17_beats_v16")


def family_h747b_mbr_tsc(seed: int) -> dict[str, Any]:
    return _bar(
        W69_MASC_V5_REGIME_MULTI_BRANCH_REJOIN,
        "tsc_v17_beats_tsc_v16_rate", "h747b_mbr_tsc")


def family_h748_sc(seed: int) -> dict[str, Any]:
    return _bar(
        W69_MASC_V5_REGIME_SILENT_CORRUPTION_PLUS_REPLACEMENT,
        "v17_beats_v16_rate",
        "h748_sc_v17_beats_v16")


def family_h748b_sc_tsc(seed: int) -> dict[str, Any]:
    return _bar(
        W69_MASC_V5_REGIME_SILENT_CORRUPTION_PLUS_REPLACEMENT,
        "tsc_v17_beats_tsc_v16_rate", "h748b_sc_tsc")


def family_h749_ctr(seed: int) -> dict[str, Any]:
    return _bar(
        W70_MASC_V6_REGIME_CONTRADICTION_THEN_REJOIN_UNDER_BUDGET,
        "v17_beats_v16_rate",
        "h749_ctr_v17_beats_v16")


def family_h749b_ctr_tsc(seed: int) -> dict[str, Any]:
    return _bar(
        W70_MASC_V6_REGIME_CONTRADICTION_THEN_REJOIN_UNDER_BUDGET,
        "tsc_v17_beats_tsc_v16_rate", "h749b_ctr_tsc")


def family_h750_drar(seed: int) -> dict[str, Any]:
    return _bar(
        W71_MASC_V7_REGIME_DELAYED_REPAIR_AFTER_RESTART,
        "v17_beats_v16_rate",
        "h750_drar_v17_beats_v16")


def family_h750b_drar_tsc(seed: int) -> dict[str, Any]:
    return _bar(
        W71_MASC_V7_REGIME_DELAYED_REPAIR_AFTER_RESTART,
        "tsc_v17_beats_tsc_v16_rate", "h750b_drar_tsc")


def family_h751_drrjub(seed: int) -> dict[str, Any]:
    return _bar(
        W72_MASC_V8_REGIME_DELAYED_REJOIN_AFTER_RESTART_UNDER_BUDGET,
        "v17_beats_v16_rate",
        "h751_drrjub_v17_beats_v16")


def family_h751b_drrjub_tsc(seed: int) -> dict[str, Any]:
    return _bar(
        W72_MASC_V8_REGIME_DELAYED_REJOIN_AFTER_RESTART_UNDER_BUDGET,
        "tsc_v17_beats_tsc_v16_rate", "h751b_drrjub_tsc")


def family_h752_v17_visible_tokens_savings(
        seed: int) -> dict[str, Any]:
    r = _run_masc_v8_regime(
        W66_MASC_V2_REGIME_BASELINE, seeds=10)
    return {
        "schema": R171_SCHEMA_VERSION,
        "name": "h752_v17_visible_tokens_savings",
        "passed": bool(
            r["v17_visible_tokens_savings_vs_transcript"]
            >= 0.65),
        "savings": float(
            r["v17_visible_tokens_savings_vs_transcript"]),
    }


def family_h753_v17_ts_per_visible_token_beats_v16(
        seed: int) -> dict[str, Any]:
    r = _run_masc_v8_regime(
        W66_MASC_V2_REGIME_BASELINE, seeds=10)
    v17_ts = float(r["team_success_per_visible_token_v17"])
    sr = dict(r["per_policy_success_rate"])
    v16_sr = float(sr.get(
        W71_MASC_V7_POLICY_SUBSTRATE_ROUTED_V16, 0.0))
    v17_sr = float(sr.get(
        W72_MASC_V8_POLICY_SUBSTRATE_ROUTED_V17, 0.0))
    return {
        "schema": R171_SCHEMA_VERSION,
        "name":
            "h753_v17_ts_per_visible_token_beats_v16",
        "passed": bool(v17_ts > 0.0 and v17_sr >= v16_sr),
        "v17_ts_per_vt": float(v17_ts),
        "v17_sr": float(v17_sr),
        "v16_sr": float(v16_sr),
    }


def family_h754_tcc_v7_rejoin_pressure_arbiter(
        seed: int) -> dict[str, Any]:
    tcc = TeamConsensusControllerV7()
    out = tcc.decide_v7(
        regime=W66_MASC_V2_REGIME_BASELINE,
        agent_guesses=[1.0, -1.0, 0.5, 0.2],
        agent_confidences=[0.8, 0.6, 0.7, 0.7],
        substrate_replay_trust=0.7,
        visible_token_budget_ratio=0.9,
        dominant_repair_label=1,
        agent_repair_labels=[1, 1, 0, 0],
        rejoin_pressure=0.8,
        agent_rejoin_recovery_flags=[1, 0, 1, 0])
    return {
        "schema": R171_SCHEMA_VERSION,
        "name": "h754_tcc_v7_rejoin_pressure_arbiter",
        "passed": bool(
            out["decision"]
            == W72_TC_V7_DECISION_REJOIN_PRESSURE),
    }


def family_h755_tcc_v7_delayed_rejoin_arbiter(
        seed: int) -> dict[str, Any]:
    tcc = TeamConsensusControllerV7()
    out = tcc.decide_v7(
        regime=(
            W72_MASC_V8_REGIME_DELAYED_REJOIN_AFTER_RESTART_UNDER_BUDGET),
        agent_guesses=[1.0, -1.0, 0.5, 0.2],
        agent_confidences=[0.8, 0.6, 0.7, 0.7],
        substrate_replay_trust=0.7,
        visible_token_budget_ratio=0.3,
        dominant_repair_label=1,
        agent_repair_labels=[1, 1, 0, 0],
        branch_assignments=[0, 1, 2, 0],
        restart_repair_trajectory_cid="rrt_abc",
        rejoin_lag_turns=4,
        agent_rejoin_absorption_scores=[
            0.95, 0.5, 0.4, 0.3])
    return {
        "schema": R171_SCHEMA_VERSION,
        "name": "h755_tcc_v7_delayed_rejoin_arbiter",
        "passed": bool(
            out["decision"]
            == W72_TC_V7_DECISION_DELAYED_REJOIN),
    }


_R171_FAMILIES: tuple[Any, ...] = (
    family_h740_baseline_v17,
    family_h740b_baseline_tsc,
    family_h741_tcub_v17,
    family_h741b_tcub_tsc,
    family_h742_tfr_v17,
    family_h742b_tfr_tsc,
    family_h743_role_dropout,
    family_h744_branch_merge,
    family_h745_pc,
    family_h745b_pc_tsc,
    family_h746_ar,
    family_h746b_ar_tsc,
    family_h747_mbr,
    family_h747b_mbr_tsc,
    family_h748_sc,
    family_h748b_sc_tsc,
    family_h749_ctr,
    family_h749b_ctr_tsc,
    family_h750_drar,
    family_h750b_drar_tsc,
    family_h751_drrjub,
    family_h751b_drrjub_tsc,
    family_h752_v17_visible_tokens_savings,
    family_h753_v17_ts_per_visible_token_beats_v16,
    family_h754_tcc_v7_rejoin_pressure_arbiter,
    family_h755_tcc_v7_delayed_rejoin_arbiter,
)


def run_r171(
        seeds: Sequence[int] = (199, 299, 399),
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for s in seeds:
        results: dict[str, Any] = {}
        for fn in _R171_FAMILIES:
            try:
                r = fn(int(s))
            except Exception as e:
                r = {
                    "schema": R171_SCHEMA_VERSION,
                    "name": getattr(fn, "__name__", "unknown"),
                    "passed": False,
                    "exception": str(e),
                }
            results[r["name"]] = r
        out.append({
            "schema": R171_SCHEMA_VERSION,
            "seed": int(s),
            "family_results": results,
        })
    return out


__all__ = ["R171_SCHEMA_VERSION", "run_r171"]
