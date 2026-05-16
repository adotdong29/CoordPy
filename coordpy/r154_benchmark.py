"""W68 R-154 benchmark family — Multi-Agent Task Success.

Exercises MASC V4 across all seven regimes (baseline,
team_consensus_under_budget, team_failure_recovery, role_dropout,
branch_merge_reconciliation, partial_contradiction,
agent_replacement_warm_restart). Verifies that:

* V13 strictly beats V12 on ≥ 50 % of seeds in every regime.
* TSC_V13 strictly beats TSC_V12 on ≥ 50 % of seeds in every
  regime.
* V13 saves visible tokens vs transcript-only.
* TCC V3 fires the new arbiters when given the relevant signals.

H330..H343 cell families (14 H-bars):

* H330   baseline: V13 strictly beats V12 on ≥ 50 % of seeds
* H330b  baseline: TSC_V13 strictly beats TSC_V12 on ≥ 50 %
* H331   team_consensus_under_budget: V13 vs V12 ≥ 50 %
* H331b  team_consensus_under_budget: TSC_V13 vs TSC_V12 ≥ 50 %
* H332   team_failure_recovery: V13 vs V12 ≥ 50 %
* H332b  team_failure_recovery: TSC_V13 vs TSC_V12 ≥ 50 %
* H333   role_dropout: V13 vs V12 ≥ 50 %
* H334   branch_merge_reconciliation: V13 vs V12 ≥ 50 %
* H335   partial_contradiction: V13 vs V12 ≥ 50 %
* H335b  partial_contradiction: TSC_V13 vs TSC_V12 ≥ 50 %
* H336   agent_replacement_warm_restart: V13 vs V12 ≥ 50 %
* H336b  agent_replacement: TSC_V13 vs TSC_V12 ≥ 50 %
* H337   V13 saves visible tokens vs transcript-only on baseline
* H338   TCC V3 fires partial_contradiction arbiter
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
    MultiAgentSubstrateCoordinatorV4,
    W68_MASC_V4_POLICY_SUBSTRATE_ROUTED_V13,
    W68_MASC_V4_POLICY_TEAM_SUBSTRATE_COORDINATION_V13,
    W68_MASC_V4_REGIME_AGENT_REPLACEMENT_WARM_RESTART,
    W68_MASC_V4_REGIME_PARTIAL_CONTRADICTION,
    W68_MASC_V4_REGIMES,
)
from coordpy.multi_agent_substrate_coordinator import (
    W65_MASC_POLICY_TRANSCRIPT_ONLY,
)
from coordpy.team_consensus_controller_v3 import (
    TeamConsensusControllerV3,
    W68_TC_V3_DECISION_PARTIAL_CONTRADICTION,
)


R154_SCHEMA_VERSION: str = "coordpy.r154_benchmark.v1"


def _run_masc_v4_regime(
        regime: str, seeds: int = 15,
) -> dict[str, Any]:
    masc = MultiAgentSubstrateCoordinatorV4()
    _, agg = masc.run_batch(
        seeds=list(range(int(seeds))), regime=regime)
    return {
        "regime": str(regime),
        "v13_beats_v12_rate": float(agg.v13_beats_v12_rate),
        "tsc_v13_beats_tsc_v12_rate": float(
            agg.tsc_v13_beats_tsc_v12_rate),
        "per_policy_success_rate": dict(
            agg.per_policy_success_rate),
        "v13_visible_tokens_savings_vs_transcript": float(
            agg.v13_visible_tokens_savings_vs_transcript),
    }


def family_h330_baseline_v13_beats_v12(
        seed: int) -> dict[str, Any]:
    r = _run_masc_v4_regime(
        W66_MASC_V2_REGIME_BASELINE, seeds=15)
    return {
        "schema": R154_SCHEMA_VERSION,
        "name": "h330_baseline_v13_beats_v12",
        "passed": bool(r["v13_beats_v12_rate"] >= 0.5),
        "rate": float(r["v13_beats_v12_rate"]),
    }


def family_h330b_baseline_tsc_v13_beats_tsc_v12(
        seed: int) -> dict[str, Any]:
    r = _run_masc_v4_regime(
        W66_MASC_V2_REGIME_BASELINE, seeds=15)
    return {
        "schema": R154_SCHEMA_VERSION,
        "name": "h330b_baseline_tsc_v13_beats_tsc_v12",
        "passed": bool(r["tsc_v13_beats_tsc_v12_rate"] >= 0.5),
        "rate": float(r["tsc_v13_beats_tsc_v12_rate"]),
    }


def family_h331_tcub_v13_beats_v12(
        seed: int) -> dict[str, Any]:
    r = _run_masc_v4_regime(
        W66_MASC_V2_REGIME_TEAM_CONSENSUS_UNDER_BUDGET, seeds=15)
    return {
        "schema": R154_SCHEMA_VERSION,
        "name": "h331_tcub_v13_beats_v12",
        "passed": bool(r["v13_beats_v12_rate"] >= 0.5),
        "rate": float(r["v13_beats_v12_rate"]),
    }


def family_h331b_tcub_tsc(seed: int) -> dict[str, Any]:
    r = _run_masc_v4_regime(
        W66_MASC_V2_REGIME_TEAM_CONSENSUS_UNDER_BUDGET, seeds=15)
    return {
        "schema": R154_SCHEMA_VERSION,
        "name": "h331b_tcub_tsc",
        "passed": bool(r["tsc_v13_beats_tsc_v12_rate"] >= 0.5),
        "rate": float(r["tsc_v13_beats_tsc_v12_rate"]),
    }


def family_h332_tfr_v13_beats_v12(seed: int) -> dict[str, Any]:
    r = _run_masc_v4_regime(
        W66_MASC_V2_REGIME_TEAM_FAILURE_RECOVERY, seeds=15)
    return {
        "schema": R154_SCHEMA_VERSION,
        "name": "h332_tfr_v13_beats_v12",
        "passed": bool(r["v13_beats_v12_rate"] >= 0.5),
        "rate": float(r["v13_beats_v12_rate"]),
    }


def family_h332b_tfr_tsc(seed: int) -> dict[str, Any]:
    r = _run_masc_v4_regime(
        W66_MASC_V2_REGIME_TEAM_FAILURE_RECOVERY, seeds=15)
    return {
        "schema": R154_SCHEMA_VERSION,
        "name": "h332b_tfr_tsc",
        "passed": bool(r["tsc_v13_beats_tsc_v12_rate"] >= 0.5),
        "rate": float(r["tsc_v13_beats_tsc_v12_rate"]),
    }


def family_h333_role_dropout_v13_beats_v12(
        seed: int) -> dict[str, Any]:
    r = _run_masc_v4_regime(
        W67_MASC_V3_REGIME_ROLE_DROPOUT, seeds=15)
    return {
        "schema": R154_SCHEMA_VERSION,
        "name": "h333_role_dropout_v13_beats_v12",
        "passed": bool(r["v13_beats_v12_rate"] >= 0.5),
        "rate": float(r["v13_beats_v12_rate"]),
    }


def family_h334_branch_merge_v13_beats_v12(
        seed: int) -> dict[str, Any]:
    r = _run_masc_v4_regime(
        W67_MASC_V3_REGIME_BRANCH_MERGE_RECONCILIATION, seeds=15)
    return {
        "schema": R154_SCHEMA_VERSION,
        "name": "h334_branch_merge_v13_beats_v12",
        "passed": bool(r["v13_beats_v12_rate"] >= 0.5),
        "rate": float(r["v13_beats_v12_rate"]),
    }


def family_h335_pc_v13_beats_v12(seed: int) -> dict[str, Any]:
    r = _run_masc_v4_regime(
        W68_MASC_V4_REGIME_PARTIAL_CONTRADICTION, seeds=15)
    return {
        "schema": R154_SCHEMA_VERSION,
        "name": "h335_pc_v13_beats_v12",
        "passed": bool(r["v13_beats_v12_rate"] >= 0.5),
        "rate": float(r["v13_beats_v12_rate"]),
    }


def family_h335b_pc_tsc(seed: int) -> dict[str, Any]:
    r = _run_masc_v4_regime(
        W68_MASC_V4_REGIME_PARTIAL_CONTRADICTION, seeds=15)
    return {
        "schema": R154_SCHEMA_VERSION,
        "name": "h335b_pc_tsc",
        "passed": bool(r["tsc_v13_beats_tsc_v12_rate"] >= 0.5),
        "rate": float(r["tsc_v13_beats_tsc_v12_rate"]),
    }


def family_h336_ar_v13_beats_v12(seed: int) -> dict[str, Any]:
    r = _run_masc_v4_regime(
        W68_MASC_V4_REGIME_AGENT_REPLACEMENT_WARM_RESTART,
        seeds=15)
    return {
        "schema": R154_SCHEMA_VERSION,
        "name": "h336_ar_v13_beats_v12",
        "passed": bool(r["v13_beats_v12_rate"] >= 0.5),
        "rate": float(r["v13_beats_v12_rate"]),
    }


def family_h336b_ar_tsc(seed: int) -> dict[str, Any]:
    r = _run_masc_v4_regime(
        W68_MASC_V4_REGIME_AGENT_REPLACEMENT_WARM_RESTART,
        seeds=15)
    return {
        "schema": R154_SCHEMA_VERSION,
        "name": "h336b_ar_tsc",
        "passed": bool(r["tsc_v13_beats_tsc_v12_rate"] >= 0.5),
        "rate": float(r["tsc_v13_beats_tsc_v12_rate"]),
    }


def family_h337_v13_visible_tokens_savings(
        seed: int) -> dict[str, Any]:
    r = _run_masc_v4_regime(
        W66_MASC_V2_REGIME_BASELINE, seeds=10)
    # V13 routed and TSC V13 should each use < transcript-only.
    return {
        "schema": R154_SCHEMA_VERSION,
        "name": "h337_v13_visible_tokens_savings",
        "passed": bool(
            r["v13_visible_tokens_savings_vs_transcript"] > 0.5),
        "savings": float(
            r["v13_visible_tokens_savings_vs_transcript"]),
    }


def family_h338_tcc_v3_partial_contradiction_arbiter(
        seed: int) -> dict[str, Any]:
    tcc = TeamConsensusControllerV3()
    out = tcc.decide_v3(
        regime=W68_MASC_V4_REGIME_PARTIAL_CONTRADICTION,
        agent_guesses=[1.0, -1.0, 0.5],
        agent_confidences=[0.8, 0.4, 0.7],
        substrate_replay_trust=0.7,
        transcript_available=False,
        contradiction_indicators=[0.0, 1.0, 0.0],
        replaced_role_indices=None)
    return {
        "schema": R154_SCHEMA_VERSION,
        "name": "h338_tcc_v3_partial_contradiction_arbiter",
        "passed": bool(
            out["decision"]
            == W68_TC_V3_DECISION_PARTIAL_CONTRADICTION),
    }


_R154_FAMILIES: tuple[Any, ...] = (
    family_h330_baseline_v13_beats_v12,
    family_h330b_baseline_tsc_v13_beats_tsc_v12,
    family_h331_tcub_v13_beats_v12,
    family_h331b_tcub_tsc,
    family_h332_tfr_v13_beats_v12,
    family_h332b_tfr_tsc,
    family_h333_role_dropout_v13_beats_v12,
    family_h334_branch_merge_v13_beats_v12,
    family_h335_pc_v13_beats_v12,
    family_h335b_pc_tsc,
    family_h336_ar_v13_beats_v12,
    family_h336b_ar_tsc,
    family_h337_v13_visible_tokens_savings,
    family_h338_tcc_v3_partial_contradiction_arbiter,
)


def run_r154(
        seeds: Sequence[int] = (199, 299, 399),
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for s in seeds:
        results: dict[str, Any] = {}
        for fn in _R154_FAMILIES:
            try:
                r = fn(int(s))
            except Exception as e:
                r = {
                    "schema": R154_SCHEMA_VERSION,
                    "name": getattr(fn, "__name__", "unknown"),
                    "passed": False,
                    "exception": str(e),
                }
            results[r["name"]] = r
        out.append({
            "schema": R154_SCHEMA_VERSION,
            "seed": int(s),
            "family_results": results,
        })
    return out


__all__ = ["R154_SCHEMA_VERSION", "run_r154"]
