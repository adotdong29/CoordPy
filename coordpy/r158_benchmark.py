"""W69 R-158 benchmark family — Multi-Agent Task Success.

Exercises MASC V5 across all nine regimes (the seven W68 regimes
plus the two new W69 regimes: multi_branch_rejoin_after_divergent_work
and silent_corruption_plus_member_replacement).

H430..H447 cell families (18 H-bars):

* H430   baseline: V14 strictly beats V13 on ≥ 50 % of seeds
* H430b  baseline: TSC_V14 strictly beats TSC_V13 on ≥ 50 %
* H431   team_consensus_under_budget: V14 vs V13 ≥ 50 %
* H431b  team_consensus_under_budget: TSC_V14 vs TSC_V13 ≥ 50 %
* H432   team_failure_recovery: V14 vs V13 ≥ 50 %
* H432b  team_failure_recovery: TSC_V14 vs TSC_V13 ≥ 50 %
* H433   role_dropout: V14 vs V13 ≥ 50 %
* H434   branch_merge_reconciliation: V14 vs V13 ≥ 50 %
* H435   partial_contradiction: V14 vs V13 ≥ 50 %
* H435b  partial_contradiction: TSC_V14 vs TSC_V13 ≥ 50 %
* H436   agent_replacement_warm_restart: V14 vs V13 ≥ 50 %
* H436b  agent_replacement: TSC_V14 vs TSC_V13 ≥ 50 %
* H437   multi_branch_rejoin: V14 vs V13 ≥ 50 %
* H437b  multi_branch_rejoin: TSC_V14 vs TSC_V13 ≥ 50 %
* H438   silent_corruption: V14 vs V13 ≥ 50 %
* H438b  silent_corruption: TSC_V14 vs TSC_V13 ≥ 50 %
* H439   V14 saves visible tokens vs transcript on baseline ≥ 50%
* H440   TCC V4 fires multi_branch_rejoin arbiter
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
    MultiAgentSubstrateCoordinatorV5,
    W69_MASC_V5_POLICY_SUBSTRATE_ROUTED_V14,
    W69_MASC_V5_POLICY_TEAM_SUBSTRATE_COORDINATION_V14,
    W69_MASC_V5_REGIME_MULTI_BRANCH_REJOIN,
    W69_MASC_V5_REGIME_SILENT_CORRUPTION_PLUS_REPLACEMENT,
)
from coordpy.multi_agent_substrate_coordinator import (
    W65_MASC_POLICY_TRANSCRIPT_ONLY,
)
from coordpy.team_consensus_controller_v4 import (
    TeamConsensusControllerV4,
    W69_TC_V4_DECISION_MULTI_BRANCH_REJOIN,
    W69_TC_V4_DECISION_SILENT_CORRUPTION_PLUS_REPLACEMENT,
)


R158_SCHEMA_VERSION: str = "coordpy.r158_benchmark.v1"


def _run_masc_v5_regime(
        regime: str, seeds: int = 15,
) -> dict[str, Any]:
    masc = MultiAgentSubstrateCoordinatorV5()
    _, agg = masc.run_batch(
        seeds=list(range(int(seeds))), regime=regime)
    return {
        "regime": str(regime),
        "v14_beats_v13_rate": float(agg.v14_beats_v13_rate),
        "tsc_v14_beats_tsc_v13_rate": float(
            agg.tsc_v14_beats_tsc_v13_rate),
        "per_policy_success_rate": dict(
            agg.per_policy_success_rate),
        "v14_visible_tokens_savings_vs_transcript": float(
            agg.v14_visible_tokens_savings_vs_transcript),
    }


def _bar(regime: str, key: str, name: str) -> dict[str, Any]:
    r = _run_masc_v5_regime(regime, seeds=15)
    return {
        "schema": R158_SCHEMA_VERSION,
        "name": str(name),
        "passed": bool(r[key] >= 0.5),
        "rate": float(r[key]),
    }


def family_h430_baseline_v14_beats_v13(
        seed: int) -> dict[str, Any]:
    return _bar(
        W66_MASC_V2_REGIME_BASELINE, "v14_beats_v13_rate",
        "h430_baseline_v14_beats_v13")


def family_h430b_baseline_tsc(seed: int) -> dict[str, Any]:
    return _bar(
        W66_MASC_V2_REGIME_BASELINE,
        "tsc_v14_beats_tsc_v13_rate",
        "h430b_baseline_tsc_v14_beats_tsc_v13")


def family_h431_tcub_v14_beats_v13(
        seed: int) -> dict[str, Any]:
    return _bar(
        W66_MASC_V2_REGIME_TEAM_CONSENSUS_UNDER_BUDGET,
        "v14_beats_v13_rate", "h431_tcub_v14_beats_v13")


def family_h431b_tcub_tsc(seed: int) -> dict[str, Any]:
    return _bar(
        W66_MASC_V2_REGIME_TEAM_CONSENSUS_UNDER_BUDGET,
        "tsc_v14_beats_tsc_v13_rate", "h431b_tcub_tsc")


def family_h432_tfr_v14_beats_v13(seed: int) -> dict[str, Any]:
    return _bar(
        W66_MASC_V2_REGIME_TEAM_FAILURE_RECOVERY,
        "v14_beats_v13_rate", "h432_tfr_v14_beats_v13")


def family_h432b_tfr_tsc(seed: int) -> dict[str, Any]:
    return _bar(
        W66_MASC_V2_REGIME_TEAM_FAILURE_RECOVERY,
        "tsc_v14_beats_tsc_v13_rate", "h432b_tfr_tsc")


def family_h433_role_dropout(seed: int) -> dict[str, Any]:
    return _bar(
        W67_MASC_V3_REGIME_ROLE_DROPOUT, "v14_beats_v13_rate",
        "h433_role_dropout_v14_beats_v13")


def family_h434_branch_merge(seed: int) -> dict[str, Any]:
    return _bar(
        W67_MASC_V3_REGIME_BRANCH_MERGE_RECONCILIATION,
        "v14_beats_v13_rate",
        "h434_branch_merge_v14_beats_v13")


def family_h435_partial_contradiction(
        seed: int) -> dict[str, Any]:
    return _bar(
        W68_MASC_V4_REGIME_PARTIAL_CONTRADICTION,
        "v14_beats_v13_rate", "h435_pc_v14_beats_v13")


def family_h435b_partial_contradiction_tsc(
        seed: int) -> dict[str, Any]:
    return _bar(
        W68_MASC_V4_REGIME_PARTIAL_CONTRADICTION,
        "tsc_v14_beats_tsc_v13_rate", "h435b_pc_tsc")


def family_h436_agent_replacement(seed: int) -> dict[str, Any]:
    return _bar(
        W68_MASC_V4_REGIME_AGENT_REPLACEMENT_WARM_RESTART,
        "v14_beats_v13_rate", "h436_ar_v14_beats_v13")


def family_h436b_agent_replacement_tsc(
        seed: int) -> dict[str, Any]:
    return _bar(
        W68_MASC_V4_REGIME_AGENT_REPLACEMENT_WARM_RESTART,
        "tsc_v14_beats_tsc_v13_rate", "h436b_ar_tsc")


def family_h437_multi_branch_rejoin(
        seed: int) -> dict[str, Any]:
    return _bar(
        W69_MASC_V5_REGIME_MULTI_BRANCH_REJOIN,
        "v14_beats_v13_rate", "h437_mbr_v14_beats_v13")


def family_h437b_multi_branch_rejoin_tsc(
        seed: int) -> dict[str, Any]:
    return _bar(
        W69_MASC_V5_REGIME_MULTI_BRANCH_REJOIN,
        "tsc_v14_beats_tsc_v13_rate", "h437b_mbr_tsc")


def family_h438_silent_corruption(
        seed: int) -> dict[str, Any]:
    return _bar(
        W69_MASC_V5_REGIME_SILENT_CORRUPTION_PLUS_REPLACEMENT,
        "v14_beats_v13_rate", "h438_sc_v14_beats_v13")


def family_h438b_silent_corruption_tsc(
        seed: int) -> dict[str, Any]:
    return _bar(
        W69_MASC_V5_REGIME_SILENT_CORRUPTION_PLUS_REPLACEMENT,
        "tsc_v14_beats_tsc_v13_rate", "h438b_sc_tsc")


def family_h439_v14_visible_tokens_savings(
        seed: int) -> dict[str, Any]:
    r = _run_masc_v5_regime(
        W66_MASC_V2_REGIME_BASELINE, seeds=10)
    return {
        "schema": R158_SCHEMA_VERSION,
        "name": "h439_v14_visible_tokens_savings",
        "passed": bool(
            r["v14_visible_tokens_savings_vs_transcript"] > 0.5),
        "savings": float(
            r["v14_visible_tokens_savings_vs_transcript"]),
    }


def family_h440_tcc_v4_multi_branch_rejoin_arbiter(
        seed: int) -> dict[str, Any]:
    tcc = TeamConsensusControllerV4()
    out = tcc.decide_v4(
        regime=W69_MASC_V5_REGIME_MULTI_BRANCH_REJOIN,
        agent_guesses=[1.0, -1.0, 0.5, 0.2],
        agent_confidences=[0.8, 0.6, 0.7, 0.7],
        substrate_replay_trust=0.7,
        transcript_available=False,
        branch_assignments=[0, 1, 2, 0])
    return {
        "schema": R158_SCHEMA_VERSION,
        "name": "h440_tcc_v4_multi_branch_rejoin_arbiter",
        "passed": bool(
            out["decision"]
            == W69_TC_V4_DECISION_MULTI_BRANCH_REJOIN),
    }


_R158_FAMILIES: tuple[Any, ...] = (
    family_h430_baseline_v14_beats_v13,
    family_h430b_baseline_tsc,
    family_h431_tcub_v14_beats_v13,
    family_h431b_tcub_tsc,
    family_h432_tfr_v14_beats_v13,
    family_h432b_tfr_tsc,
    family_h433_role_dropout,
    family_h434_branch_merge,
    family_h435_partial_contradiction,
    family_h435b_partial_contradiction_tsc,
    family_h436_agent_replacement,
    family_h436b_agent_replacement_tsc,
    family_h437_multi_branch_rejoin,
    family_h437b_multi_branch_rejoin_tsc,
    family_h438_silent_corruption,
    family_h438b_silent_corruption_tsc,
    family_h439_v14_visible_tokens_savings,
    family_h440_tcc_v4_multi_branch_rejoin_arbiter,
)


def run_r158(
        seeds: Sequence[int] = (199, 299, 399),
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for s in seeds:
        results: dict[str, Any] = {}
        for fn in _R158_FAMILIES:
            try:
                r = fn(int(s))
            except Exception as e:
                r = {
                    "schema": R158_SCHEMA_VERSION,
                    "name": getattr(fn, "__name__", "unknown"),
                    "passed": False,
                    "exception": str(e),
                }
            results[r["name"]] = r
        out.append({
            "schema": R158_SCHEMA_VERSION,
            "seed": int(s),
            "family_results": results,
        })
    return out


__all__ = ["R158_SCHEMA_VERSION", "run_r158"]
