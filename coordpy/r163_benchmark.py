"""W70 R-163 benchmark family — Multi-Agent Task Success.

Exercises MASC V6 across all ten regimes (the nine W69 regimes plus
the new W70 regime: contradiction_then_rejoin_under_budget).

H490..H510 cell families (22 H-bars):

* H490   baseline: V15 strictly beats V14 on ≥ 50 % of seeds
* H490b  baseline: TSC_V15 strictly beats TSC_V14 on ≥ 50 %
* H491   team_consensus_under_budget: V15 vs V14 ≥ 50 %
* H491b  team_consensus_under_budget: TSC_V15 vs TSC_V14 ≥ 50 %
* H492   team_failure_recovery: V15 vs V14 ≥ 50 %
* H492b  team_failure_recovery: TSC_V15 vs TSC_V14 ≥ 50 %
* H493   role_dropout: V15 vs V14 ≥ 50 %
* H494   branch_merge_reconciliation: V15 vs V14 ≥ 50 %
* H495   partial_contradiction: V15 vs V14 ≥ 50 %
* H495b  partial_contradiction: TSC_V15 vs TSC_V14 ≥ 50 %
* H496   agent_replacement_warm_restart: V15 vs V14 ≥ 50 %
* H496b  agent_replacement: TSC_V15 vs TSC_V14 ≥ 50 %
* H497   multi_branch_rejoin: V15 vs V14 ≥ 50 %
* H497b  multi_branch_rejoin: TSC_V15 vs TSC_V14 ≥ 50 %
* H498   silent_corruption: V15 vs V14 ≥ 50 %
* H498b  silent_corruption: TSC_V15 vs TSC_V14 ≥ 50 %
* H499   contradiction_then_rejoin_under_budget: V15 vs V14 ≥ 50 %
* H499b  contradiction_then_rejoin: TSC_V15 vs TSC_V14 ≥ 50 %
* H500   V15 saves visible tokens vs transcript on baseline ≥ 60 %
* H501   V15 team_success_per_visible_token > 0 on baseline
* H502   TCC V5 repair_dominance_arbiter fires
* H503   TCC V5 budget_primary_arbiter fires
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
    MultiAgentSubstrateCoordinatorV6,
    W70_MASC_V6_POLICY_SUBSTRATE_ROUTED_V15,
    W70_MASC_V6_POLICY_TEAM_SUBSTRATE_COORDINATION_V15,
    W70_MASC_V6_REGIME_CONTRADICTION_THEN_REJOIN_UNDER_BUDGET,
)
from coordpy.team_consensus_controller_v5 import (
    TeamConsensusControllerV5,
    W70_TC_V5_DECISION_BUDGET_PRIMARY,
    W70_TC_V5_DECISION_REPAIR_DOMINANCE,
)


R163_SCHEMA_VERSION: str = "coordpy.r163_benchmark.v1"


def _run_masc_v6_regime(
        regime: str, seeds: int = 15,
) -> dict[str, Any]:
    masc = MultiAgentSubstrateCoordinatorV6()
    _, agg = masc.run_batch(
        seeds=list(range(int(seeds))), regime=regime)
    return {
        "regime": str(regime),
        "v15_beats_v14_rate": float(agg.v15_beats_v14_rate),
        "tsc_v15_beats_tsc_v14_rate": float(
            agg.tsc_v15_beats_tsc_v14_rate),
        "per_policy_success_rate": dict(
            agg.per_policy_success_rate),
        "v15_visible_tokens_savings_vs_transcript": float(
            agg.v15_visible_tokens_savings_vs_transcript),
        "team_success_per_visible_token_v15": float(
            agg.team_success_per_visible_token_v15),
    }


def _bar(regime: str, key: str, name: str) -> dict[str, Any]:
    r = _run_masc_v6_regime(regime, seeds=15)
    return {
        "schema": R163_SCHEMA_VERSION,
        "name": str(name),
        "passed": bool(r[key] >= 0.5),
        "rate": float(r[key]),
    }


def family_h490_baseline_v15_beats_v14(
        seed: int) -> dict[str, Any]:
    return _bar(
        W66_MASC_V2_REGIME_BASELINE, "v15_beats_v14_rate",
        "h490_baseline_v15_beats_v14")


def family_h490b_baseline_tsc(seed: int) -> dict[str, Any]:
    return _bar(
        W66_MASC_V2_REGIME_BASELINE,
        "tsc_v15_beats_tsc_v14_rate",
        "h490b_baseline_tsc_v15_beats_tsc_v14")


def family_h491_tcub_v15_beats_v14(
        seed: int) -> dict[str, Any]:
    return _bar(
        W66_MASC_V2_REGIME_TEAM_CONSENSUS_UNDER_BUDGET,
        "v15_beats_v14_rate",
        "h491_tcub_v15_beats_v14")


def family_h491b_tcub_tsc(seed: int) -> dict[str, Any]:
    return _bar(
        W66_MASC_V2_REGIME_TEAM_CONSENSUS_UNDER_BUDGET,
        "tsc_v15_beats_tsc_v14_rate", "h491b_tcub_tsc")


def family_h492_tfr_v15_beats_v14(
        seed: int) -> dict[str, Any]:
    return _bar(
        W66_MASC_V2_REGIME_TEAM_FAILURE_RECOVERY,
        "v15_beats_v14_rate", "h492_tfr_v15_beats_v14")


def family_h492b_tfr_tsc(seed: int) -> dict[str, Any]:
    return _bar(
        W66_MASC_V2_REGIME_TEAM_FAILURE_RECOVERY,
        "tsc_v15_beats_tsc_v14_rate", "h492b_tfr_tsc")


def family_h493_role_dropout(seed: int) -> dict[str, Any]:
    return _bar(
        W67_MASC_V3_REGIME_ROLE_DROPOUT,
        "v15_beats_v14_rate",
        "h493_role_dropout_v15_beats_v14")


def family_h494_branch_merge(seed: int) -> dict[str, Any]:
    return _bar(
        W67_MASC_V3_REGIME_BRANCH_MERGE_RECONCILIATION,
        "v15_beats_v14_rate",
        "h494_branch_merge_v15_beats_v14")


def family_h495_partial_contradiction(
        seed: int) -> dict[str, Any]:
    return _bar(
        W68_MASC_V4_REGIME_PARTIAL_CONTRADICTION,
        "v15_beats_v14_rate",
        "h495_pc_v15_beats_v14")


def family_h495b_partial_contradiction_tsc(
        seed: int) -> dict[str, Any]:
    return _bar(
        W68_MASC_V4_REGIME_PARTIAL_CONTRADICTION,
        "tsc_v15_beats_tsc_v14_rate", "h495b_pc_tsc")


def family_h496_agent_replacement(
        seed: int) -> dict[str, Any]:
    return _bar(
        W68_MASC_V4_REGIME_AGENT_REPLACEMENT_WARM_RESTART,
        "v15_beats_v14_rate",
        "h496_ar_v15_beats_v14")


def family_h496b_agent_replacement_tsc(
        seed: int) -> dict[str, Any]:
    return _bar(
        W68_MASC_V4_REGIME_AGENT_REPLACEMENT_WARM_RESTART,
        "tsc_v15_beats_tsc_v14_rate", "h496b_ar_tsc")


def family_h497_multi_branch_rejoin(
        seed: int) -> dict[str, Any]:
    return _bar(
        W69_MASC_V5_REGIME_MULTI_BRANCH_REJOIN,
        "v15_beats_v14_rate",
        "h497_mbr_v15_beats_v14")


def family_h497b_multi_branch_rejoin_tsc(
        seed: int) -> dict[str, Any]:
    return _bar(
        W69_MASC_V5_REGIME_MULTI_BRANCH_REJOIN,
        "tsc_v15_beats_tsc_v14_rate", "h497b_mbr_tsc")


def family_h498_silent_corruption(
        seed: int) -> dict[str, Any]:
    return _bar(
        W69_MASC_V5_REGIME_SILENT_CORRUPTION_PLUS_REPLACEMENT,
        "v15_beats_v14_rate",
        "h498_sc_v15_beats_v14")


def family_h498b_silent_corruption_tsc(
        seed: int) -> dict[str, Any]:
    return _bar(
        W69_MASC_V5_REGIME_SILENT_CORRUPTION_PLUS_REPLACEMENT,
        "tsc_v15_beats_tsc_v14_rate", "h498b_sc_tsc")


def family_h499_contradiction_then_rejoin(
        seed: int) -> dict[str, Any]:
    return _bar(
        W70_MASC_V6_REGIME_CONTRADICTION_THEN_REJOIN_UNDER_BUDGET,
        "v15_beats_v14_rate",
        "h499_ctr_v15_beats_v14")


def family_h499b_contradiction_then_rejoin_tsc(
        seed: int) -> dict[str, Any]:
    return _bar(
        W70_MASC_V6_REGIME_CONTRADICTION_THEN_REJOIN_UNDER_BUDGET,
        "tsc_v15_beats_tsc_v14_rate", "h499b_ctr_tsc")


def family_h500_v15_visible_tokens_savings(
        seed: int) -> dict[str, Any]:
    r = _run_masc_v6_regime(
        W66_MASC_V2_REGIME_BASELINE, seeds=10)
    return {
        "schema": R163_SCHEMA_VERSION,
        "name": "h500_v15_visible_tokens_savings",
        "passed": bool(
            r["v15_visible_tokens_savings_vs_transcript"]
            >= 0.6),
        "savings": float(
            r["v15_visible_tokens_savings_vs_transcript"]),
    }


def family_h501_team_success_per_visible_token(
        seed: int) -> dict[str, Any]:
    r = _run_masc_v6_regime(
        W66_MASC_V2_REGIME_BASELINE, seeds=10)
    return {
        "schema": R163_SCHEMA_VERSION,
        "name": "h501_team_success_per_visible_token",
        "passed": bool(
            r["team_success_per_visible_token_v15"] > 0.0),
        "ts_per_vt": float(
            r["team_success_per_visible_token_v15"]),
    }


def family_h502_tcc_v5_repair_dominance_arbiter(
        seed: int) -> dict[str, Any]:
    tcc = TeamConsensusControllerV5()
    out = tcc.decide_v5(
        regime=W66_MASC_V2_REGIME_BASELINE,
        agent_guesses=[1.0, -1.0, 0.5, 0.2],
        agent_confidences=[0.8, 0.6, 0.7, 0.7],
        substrate_replay_trust=0.7,
        visible_token_budget_ratio=0.9,
        dominant_repair_label=1,
        agent_repair_labels=[1, 1, 0, 0])
    return {
        "schema": R163_SCHEMA_VERSION,
        "name": "h502_tcc_v5_repair_dominance_arbiter",
        "passed": bool(
            out["decision"]
            == W70_TC_V5_DECISION_REPAIR_DOMINANCE),
    }


def family_h503_tcc_v5_budget_primary_arbiter(
        seed: int) -> dict[str, Any]:
    tcc = TeamConsensusControllerV5()
    out = tcc.decide_v5(
        regime=W66_MASC_V2_REGIME_BASELINE,
        agent_guesses=[0.5, 0.5, 0.4, 0.5],
        agent_confidences=[0.8, 0.6, 0.7, 0.7],
        substrate_replay_trust=0.7,
        visible_token_budget_ratio=0.3,
        dominant_repair_label=0)
    return {
        "schema": R163_SCHEMA_VERSION,
        "name": "h503_tcc_v5_budget_primary_arbiter",
        "passed": bool(
            out["decision"]
            == W70_TC_V5_DECISION_BUDGET_PRIMARY),
    }


_R163_FAMILIES: tuple[Any, ...] = (
    family_h490_baseline_v15_beats_v14,
    family_h490b_baseline_tsc,
    family_h491_tcub_v15_beats_v14,
    family_h491b_tcub_tsc,
    family_h492_tfr_v15_beats_v14,
    family_h492b_tfr_tsc,
    family_h493_role_dropout,
    family_h494_branch_merge,
    family_h495_partial_contradiction,
    family_h495b_partial_contradiction_tsc,
    family_h496_agent_replacement,
    family_h496b_agent_replacement_tsc,
    family_h497_multi_branch_rejoin,
    family_h497b_multi_branch_rejoin_tsc,
    family_h498_silent_corruption,
    family_h498b_silent_corruption_tsc,
    family_h499_contradiction_then_rejoin,
    family_h499b_contradiction_then_rejoin_tsc,
    family_h500_v15_visible_tokens_savings,
    family_h501_team_success_per_visible_token,
    family_h502_tcc_v5_repair_dominance_arbiter,
    family_h503_tcc_v5_budget_primary_arbiter,
)


def run_r163(
        seeds: Sequence[int] = (199, 299, 399),
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for s in seeds:
        results: dict[str, Any] = {}
        for fn in _R163_FAMILIES:
            try:
                r = fn(int(s))
            except Exception as e:
                r = {
                    "schema": R163_SCHEMA_VERSION,
                    "name": getattr(fn, "__name__", "unknown"),
                    "passed": False,
                    "exception": str(e),
                }
            results[r["name"]] = r
        out.append({
            "schema": R163_SCHEMA_VERSION,
            "seed": int(s),
            "family_results": results,
        })
    return out


__all__ = ["R163_SCHEMA_VERSION", "run_r163"]
