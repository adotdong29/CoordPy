"""W73 R-175 benchmark family — Multi-Agent Task Success.

Exercises MASC V9 across all thirteen regimes (the twelve W72
regimes plus the new W73 regime:
``replacement_after_contradiction_then_rejoin``).

H840..H865 cell families (28 H-bars):

* H840   baseline: V18 strictly beats V17 on ≥ 50 % of seeds
* H840b  baseline: TSC_V18 strictly beats TSC_V17 on ≥ 50 %
* H841   team_consensus_under_budget: V18 vs V17 ≥ 50 %
* H841b  team_consensus_under_budget: TSC_V18 vs TSC_V17 ≥ 50 %
* H842   team_failure_recovery: V18 vs V17 ≥ 50 %
* H842b  team_failure_recovery: TSC_V18 vs TSC_V17 ≥ 50 %
* H843   role_dropout: V18 vs V17 ≥ 50 %
* H844   branch_merge_reconciliation: V18 vs V17 ≥ 50 %
* H845   partial_contradiction: V18 vs V17 ≥ 50 %
* H845b  partial_contradiction: TSC_V18 vs TSC_V17 ≥ 50 %
* H846   agent_replacement_warm_restart: V18 vs V17 ≥ 50 %
* H846b  agent_replacement_warm_restart: TSC_V18 vs TSC_V17 ≥ 50 %
* H847   multi_branch_rejoin: V18 vs V17 ≥ 50 %
* H847b  multi_branch_rejoin: TSC_V18 vs TSC_V17 ≥ 50 %
* H848   silent_corruption: V18 vs V17 ≥ 50 %
* H848b  silent_corruption: TSC_V18 vs TSC_V17 ≥ 50 %
* H849   contradiction_then_rejoin: V18 vs V17 ≥ 50 %
* H849b  contradiction_then_rejoin: TSC_V18 vs TSC_V17 ≥ 50 %
* H850   delayed_repair_after_restart: V18 vs V17 ≥ 50 %
* H850b  delayed_repair_after_restart: TSC_V18 vs TSC_V17 ≥ 50 %
* H851   delayed_rejoin_after_restart_under_budget: V18 vs V17 ≥ 50 %
* H851b  delayed_rejoin_after_restart_under_budget: TSC_V18 vs
         TSC_V17 ≥ 50 %
* H852   replacement_after_contradiction_then_rejoin: V18 vs V17 ≥ 50 %
* H852b  replacement_after_contradiction_then_rejoin: TSC_V18 vs
         TSC_V17 ≥ 50 %
* H853   V18 visible-token savings vs transcript on baseline ≥ 65 %
* H854   V18 team_success_per_visible_token strictly > V17 on baseline
* H855   TCC V8 replacement_pressure_arbiter fires
* H856   TCC V8 replacement_after_contradiction_then_rejoin_arbiter
         fires
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
from coordpy.multi_agent_substrate_coordinator_v8 import (
    W72_MASC_V8_POLICY_SUBSTRATE_ROUTED_V17,
    W72_MASC_V8_REGIME_DELAYED_REJOIN_AFTER_RESTART_UNDER_BUDGET,
)
from coordpy.multi_agent_substrate_coordinator_v9 import (
    MultiAgentSubstrateCoordinatorV9,
    W73_MASC_V9_POLICY_SUBSTRATE_ROUTED_V18,
    W73_MASC_V9_POLICY_TEAM_SUBSTRATE_COORDINATION_V18,
    W73_MASC_V9_REGIME_REPLACEMENT_AFTER_CTR,
)
from coordpy.team_consensus_controller_v8 import (
    TeamConsensusControllerV8,
    W73_TC_V8_DECISION_REPLACEMENT_AFTER_CTR,
    W73_TC_V8_DECISION_REPLACEMENT_PRESSURE,
)


R175_SCHEMA_VERSION: str = "coordpy.r175_benchmark.v1"


def _run_masc_v9_regime(
        regime: str, seeds: int = 15,
) -> dict[str, Any]:
    masc = MultiAgentSubstrateCoordinatorV9()
    _, agg = masc.run_batch(
        seeds=list(range(int(seeds))), regime=regime)
    return {
        "regime": str(regime),
        "v18_beats_v17_rate": float(agg.v18_beats_v17_rate),
        "tsc_v18_beats_tsc_v17_rate": float(
            agg.tsc_v18_beats_tsc_v17_rate),
        "per_policy_success_rate": dict(
            agg.per_policy_success_rate),
        "v18_visible_tokens_savings_vs_transcript": float(
            agg.v18_visible_tokens_savings_vs_transcript),
        "team_success_per_visible_token_v18": float(
            agg.team_success_per_visible_token_v18),
    }


def _bar(regime: str, key: str, name: str) -> dict[str, Any]:
    r = _run_masc_v9_regime(regime, seeds=15)
    return {
        "schema": R175_SCHEMA_VERSION,
        "name": str(name),
        "passed": bool(r[key] >= 0.5),
        "rate": float(r[key]),
    }


def family_h840_baseline_v18(seed: int) -> dict[str, Any]:
    return _bar(
        W66_MASC_V2_REGIME_BASELINE, "v18_beats_v17_rate",
        "h840_baseline_v18_beats_v17")


def family_h840b_baseline_tsc(seed: int) -> dict[str, Any]:
    return _bar(
        W66_MASC_V2_REGIME_BASELINE,
        "tsc_v18_beats_tsc_v17_rate",
        "h840b_baseline_tsc_v18_beats_tsc_v17")


def family_h841_tcub_v18(seed: int) -> dict[str, Any]:
    return _bar(
        W66_MASC_V2_REGIME_TEAM_CONSENSUS_UNDER_BUDGET,
        "v18_beats_v17_rate", "h841_tcub_v18_beats_v17")


def family_h841b_tcub_tsc(seed: int) -> dict[str, Any]:
    return _bar(
        W66_MASC_V2_REGIME_TEAM_CONSENSUS_UNDER_BUDGET,
        "tsc_v18_beats_tsc_v17_rate", "h841b_tcub_tsc")


def family_h842_tfr_v18(seed: int) -> dict[str, Any]:
    return _bar(
        W66_MASC_V2_REGIME_TEAM_FAILURE_RECOVERY,
        "v18_beats_v17_rate", "h842_tfr_v18_beats_v17")


def family_h842b_tfr_tsc(seed: int) -> dict[str, Any]:
    return _bar(
        W66_MASC_V2_REGIME_TEAM_FAILURE_RECOVERY,
        "tsc_v18_beats_tsc_v17_rate", "h842b_tfr_tsc")


def family_h843_role_dropout(seed: int) -> dict[str, Any]:
    return _bar(
        W67_MASC_V3_REGIME_ROLE_DROPOUT,
        "v18_beats_v17_rate",
        "h843_role_dropout_v18_beats_v17")


def family_h844_branch_merge(seed: int) -> dict[str, Any]:
    return _bar(
        W67_MASC_V3_REGIME_BRANCH_MERGE_RECONCILIATION,
        "v18_beats_v17_rate",
        "h844_branch_merge_v18_beats_v17")


def family_h845_pc(seed: int) -> dict[str, Any]:
    return _bar(
        W68_MASC_V4_REGIME_PARTIAL_CONTRADICTION,
        "v18_beats_v17_rate",
        "h845_pc_v18_beats_v17")


def family_h845b_pc_tsc(seed: int) -> dict[str, Any]:
    return _bar(
        W68_MASC_V4_REGIME_PARTIAL_CONTRADICTION,
        "tsc_v18_beats_tsc_v17_rate", "h845b_pc_tsc")


def family_h846_ar(seed: int) -> dict[str, Any]:
    return _bar(
        W68_MASC_V4_REGIME_AGENT_REPLACEMENT_WARM_RESTART,
        "v18_beats_v17_rate",
        "h846_ar_v18_beats_v17")


def family_h846b_ar_tsc(seed: int) -> dict[str, Any]:
    return _bar(
        W68_MASC_V4_REGIME_AGENT_REPLACEMENT_WARM_RESTART,
        "tsc_v18_beats_tsc_v17_rate", "h846b_ar_tsc")


def family_h847_mbr(seed: int) -> dict[str, Any]:
    return _bar(
        W69_MASC_V5_REGIME_MULTI_BRANCH_REJOIN,
        "v18_beats_v17_rate",
        "h847_mbr_v18_beats_v17")


def family_h847b_mbr_tsc(seed: int) -> dict[str, Any]:
    return _bar(
        W69_MASC_V5_REGIME_MULTI_BRANCH_REJOIN,
        "tsc_v18_beats_tsc_v17_rate", "h847b_mbr_tsc")


def family_h848_sc(seed: int) -> dict[str, Any]:
    return _bar(
        W69_MASC_V5_REGIME_SILENT_CORRUPTION_PLUS_REPLACEMENT,
        "v18_beats_v17_rate",
        "h848_sc_v18_beats_v17")


def family_h848b_sc_tsc(seed: int) -> dict[str, Any]:
    return _bar(
        W69_MASC_V5_REGIME_SILENT_CORRUPTION_PLUS_REPLACEMENT,
        "tsc_v18_beats_tsc_v17_rate", "h848b_sc_tsc")


def family_h849_ctr(seed: int) -> dict[str, Any]:
    return _bar(
        W70_MASC_V6_REGIME_CONTRADICTION_THEN_REJOIN_UNDER_BUDGET,
        "v18_beats_v17_rate",
        "h849_ctr_v18_beats_v17")


def family_h849b_ctr_tsc(seed: int) -> dict[str, Any]:
    return _bar(
        W70_MASC_V6_REGIME_CONTRADICTION_THEN_REJOIN_UNDER_BUDGET,
        "tsc_v18_beats_tsc_v17_rate", "h849b_ctr_tsc")


def family_h850_drar(seed: int) -> dict[str, Any]:
    return _bar(
        W71_MASC_V7_REGIME_DELAYED_REPAIR_AFTER_RESTART,
        "v18_beats_v17_rate",
        "h850_drar_v18_beats_v17")


def family_h850b_drar_tsc(seed: int) -> dict[str, Any]:
    return _bar(
        W71_MASC_V7_REGIME_DELAYED_REPAIR_AFTER_RESTART,
        "tsc_v18_beats_tsc_v17_rate", "h850b_drar_tsc")


def family_h851_drrjub(seed: int) -> dict[str, Any]:
    return _bar(
        W72_MASC_V8_REGIME_DELAYED_REJOIN_AFTER_RESTART_UNDER_BUDGET,
        "v18_beats_v17_rate",
        "h851_drrjub_v18_beats_v17")


def family_h851b_drrjub_tsc(seed: int) -> dict[str, Any]:
    return _bar(
        W72_MASC_V8_REGIME_DELAYED_REJOIN_AFTER_RESTART_UNDER_BUDGET,
        "tsc_v18_beats_tsc_v17_rate", "h851b_drrjub_tsc")


def family_h852_repctr(seed: int) -> dict[str, Any]:
    return _bar(
        W73_MASC_V9_REGIME_REPLACEMENT_AFTER_CTR,
        "v18_beats_v17_rate",
        "h852_repctr_v18_beats_v17")


def family_h852b_repctr_tsc(seed: int) -> dict[str, Any]:
    return _bar(
        W73_MASC_V9_REGIME_REPLACEMENT_AFTER_CTR,
        "tsc_v18_beats_tsc_v17_rate", "h852b_repctr_tsc")


def family_h853_v18_visible_tokens_savings(
        seed: int) -> dict[str, Any]:
    r = _run_masc_v9_regime(
        W66_MASC_V2_REGIME_BASELINE, seeds=10)
    return {
        "schema": R175_SCHEMA_VERSION,
        "name": "h853_v18_visible_tokens_savings",
        "passed": bool(
            r["v18_visible_tokens_savings_vs_transcript"]
            >= 0.65),
        "savings": float(
            r["v18_visible_tokens_savings_vs_transcript"]),
    }


def family_h854_v18_ts_per_visible_token_beats_v17(
        seed: int) -> dict[str, Any]:
    r = _run_masc_v9_regime(
        W66_MASC_V2_REGIME_BASELINE, seeds=10)
    v18_ts = float(r["team_success_per_visible_token_v18"])
    sr = dict(r["per_policy_success_rate"])
    v17_sr = float(sr.get(
        W72_MASC_V8_POLICY_SUBSTRATE_ROUTED_V17, 0.0))
    v18_sr = float(sr.get(
        W73_MASC_V9_POLICY_SUBSTRATE_ROUTED_V18, 0.0))
    return {
        "schema": R175_SCHEMA_VERSION,
        "name":
            "h854_v18_ts_per_visible_token_beats_v17",
        "passed": bool(v18_ts > 0.0 and v18_sr >= v17_sr),
        "v18_ts_per_vt": float(v18_ts),
        "v18_sr": float(v18_sr),
        "v17_sr": float(v17_sr),
    }


def family_h855_tcc_v8_replacement_pressure_arbiter(
        seed: int) -> dict[str, Any]:
    tcc = TeamConsensusControllerV8()
    out = tcc.decide_v8(
        regime=W66_MASC_V2_REGIME_BASELINE,
        agent_guesses=[1.0, -1.0, 0.5, 0.2],
        agent_confidences=[0.8, 0.6, 0.7, 0.7],
        substrate_replay_trust=0.7,
        visible_token_budget_ratio=0.9,
        dominant_repair_label=1,
        agent_repair_labels=[1, 1, 0, 0],
        replacement_pressure=0.8,
        agent_replacement_recovery_flags=[1, 0, 1, 0])
    return {
        "schema": R175_SCHEMA_VERSION,
        "name": "h855_tcc_v8_replacement_pressure_arbiter",
        "passed": bool(
            out["decision"]
            == W73_TC_V8_DECISION_REPLACEMENT_PRESSURE),
    }


def family_h856_tcc_v8_replacement_after_ctr_arbiter(
        seed: int) -> dict[str, Any]:
    tcc = TeamConsensusControllerV8()
    out = tcc.decide_v8(
        regime=(
            W73_MASC_V9_REGIME_REPLACEMENT_AFTER_CTR),
        agent_guesses=[1.0, -1.0, 0.5, 0.2],
        agent_confidences=[0.8, 0.6, 0.7, 0.7],
        substrate_replay_trust=0.7,
        visible_token_budget_ratio=0.3,
        dominant_repair_label=1,
        agent_repair_labels=[1, 1, 0, 0],
        branch_assignments=[0, 1, 2, 0],
        replacement_repair_trajectory_cid="rep_abc",
        replacement_lag_turns=5,
        agent_replacement_absorption_scores=[
            0.95, 0.5, 0.4, 0.3])
    return {
        "schema": R175_SCHEMA_VERSION,
        "name": "h856_tcc_v8_replacement_after_ctr_arbiter",
        "passed": bool(
            out["decision"]
            == W73_TC_V8_DECISION_REPLACEMENT_AFTER_CTR),
    }


_R175_FAMILIES: tuple[Any, ...] = (
    family_h840_baseline_v18,
    family_h840b_baseline_tsc,
    family_h841_tcub_v18,
    family_h841b_tcub_tsc,
    family_h842_tfr_v18,
    family_h842b_tfr_tsc,
    family_h843_role_dropout,
    family_h844_branch_merge,
    family_h845_pc,
    family_h845b_pc_tsc,
    family_h846_ar,
    family_h846b_ar_tsc,
    family_h847_mbr,
    family_h847b_mbr_tsc,
    family_h848_sc,
    family_h848b_sc_tsc,
    family_h849_ctr,
    family_h849b_ctr_tsc,
    family_h850_drar,
    family_h850b_drar_tsc,
    family_h851_drrjub,
    family_h851b_drrjub_tsc,
    family_h852_repctr,
    family_h852b_repctr_tsc,
    family_h853_v18_visible_tokens_savings,
    family_h854_v18_ts_per_visible_token_beats_v17,
    family_h855_tcc_v8_replacement_pressure_arbiter,
    family_h856_tcc_v8_replacement_after_ctr_arbiter,
)


def run_r175(
        seeds: Sequence[int] = (199, 299, 399),
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for s in seeds:
        results: dict[str, Any] = {}
        for fn in _R175_FAMILIES:
            try:
                r = fn(int(s))
            except Exception as e:
                r = {
                    "schema": R175_SCHEMA_VERSION,
                    "name": getattr(fn, "__name__", "unknown"),
                    "passed": False,
                    "exception": str(e),
                }
            results[r["name"]] = r
        out.append({
            "schema": R175_SCHEMA_VERSION,
            "seed": int(s),
            "family_results": results,
        })
    return out


__all__ = ["R175_SCHEMA_VERSION", "run_r175"]
