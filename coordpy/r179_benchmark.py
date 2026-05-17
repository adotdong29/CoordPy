"""W74 R-179 benchmark family — Multi-Agent Task Success.

Exercises MASC V10 across all fourteen regimes (the thirteen W73
regimes plus the new W74 regime:
``replacement_after_delayed_repair_under_budget``).

H920..H947 cell families (30 H-bars):

* H920   baseline: V19 strictly beats V18 on ≥ 50 % of seeds
* H920b  baseline: TSC_V19 strictly beats TSC_V18 on ≥ 50 %
* H921   team_consensus_under_budget: V19 vs V18 ≥ 50 %
* H921b  team_consensus_under_budget: TSC_V19 vs TSC_V18 ≥ 50 %
* H922   team_failure_recovery: V19 vs V18 ≥ 50 %
* H922b  team_failure_recovery: TSC_V19 vs TSC_V18 ≥ 50 %
* H923   role_dropout: V19 vs V18 ≥ 50 %
* H924   branch_merge_reconciliation: V19 vs V18 ≥ 50 %
* H925   partial_contradiction: V19 vs V18 ≥ 50 %
* H925b  partial_contradiction: TSC_V19 vs TSC_V18 ≥ 50 %
* H926   agent_replacement_warm_restart: V19 vs V18 ≥ 50 %
* H926b  agent_replacement_warm_restart: TSC_V19 vs TSC_V18 ≥ 50 %
* H927   multi_branch_rejoin: V19 vs V18 ≥ 50 %
* H927b  multi_branch_rejoin: TSC_V19 vs TSC_V18 ≥ 50 %
* H928   silent_corruption: V19 vs V18 ≥ 50 %
* H928b  silent_corruption: TSC_V19 vs TSC_V18 ≥ 50 %
* H929   contradiction_then_rejoin: V19 vs V18 ≥ 50 %
* H929b  contradiction_then_rejoin: TSC_V19 vs TSC_V18 ≥ 50 %
* H930   delayed_repair_after_restart: V19 vs V18 ≥ 50 %
* H930b  delayed_repair_after_restart: TSC_V19 vs TSC_V18 ≥ 50 %
* H931   delayed_rejoin_after_restart_under_budget: V19 vs V18 ≥ 50 %
* H931b  delayed_rejoin_after_restart_under_budget: TSC_V19 vs
         TSC_V18 ≥ 50 %
* H932   replacement_after_contradiction_then_rejoin: V19 vs V18 ≥ 50 %
* H932b  replacement_after_contradiction_then_rejoin: TSC_V19 vs
         TSC_V18 ≥ 50 %
* H933   replacement_after_delayed_repair_under_budget: V19 vs V18 ≥ 50 %
* H933b  replacement_after_delayed_repair_under_budget: TSC_V19 vs
         TSC_V18 ≥ 50 %
* H934   V19 visible-token savings vs transcript on baseline ≥ 65 %
* H935   V19 team_success_per_visible_token strictly > V18 on baseline
* H936   TCC V9 compound_pressure_arbiter fires
* H937   TCC V9 compound_repair_after_delayed_repair_then_replacement
         arbiter fires
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
    W72_MASC_V8_REGIME_DELAYED_REJOIN_AFTER_RESTART_UNDER_BUDGET,
)
from coordpy.multi_agent_substrate_coordinator_v9 import (
    W73_MASC_V9_POLICY_SUBSTRATE_ROUTED_V18,
    W73_MASC_V9_REGIME_REPLACEMENT_AFTER_CTR,
)
from coordpy.multi_agent_substrate_coordinator_v10 import (
    MultiAgentSubstrateCoordinatorV10,
    W74_MASC_V10_POLICY_SUBSTRATE_ROUTED_V19,
    W74_MASC_V10_POLICY_TEAM_SUBSTRATE_COORDINATION_V19,
    W74_MASC_V10_REGIME_REPLACEMENT_AFTER_DELAYED_REPAIR,
)
from coordpy.team_consensus_controller_v9 import (
    TeamConsensusControllerV9,
    W74_TC_V9_DECISION_COMPOUND_PRESSURE,
    W74_TC_V9_DECISION_COMPOUND_REPAIR_AFTER_DRTR,
)


R179_SCHEMA_VERSION: str = "coordpy.r179_benchmark.v1"


def _run_masc_v10_regime(
        regime: str, seeds: int = 15,
) -> dict[str, Any]:
    masc = MultiAgentSubstrateCoordinatorV10()
    _, agg = masc.run_batch(
        seeds=list(range(int(seeds))), regime=regime)
    return {
        "regime": str(regime),
        "v19_beats_v18_rate": float(agg.v19_beats_v18_rate),
        "tsc_v19_beats_tsc_v18_rate": float(
            agg.tsc_v19_beats_tsc_v18_rate),
        "per_policy_success_rate": dict(
            agg.per_policy_success_rate),
        "v19_visible_tokens_savings_vs_transcript": float(
            agg.v19_visible_tokens_savings_vs_transcript),
        "team_success_per_visible_token_v19": float(
            agg.team_success_per_visible_token_v19),
    }


def _bar(regime: str, key: str, name: str) -> dict[str, Any]:
    r = _run_masc_v10_regime(regime, seeds=15)
    return {
        "schema": R179_SCHEMA_VERSION,
        "name": str(name),
        "passed": bool(r[key] >= 0.5),
        "rate": float(r[key]),
    }


def family_h920_baseline_v19(seed: int) -> dict[str, Any]:
    return _bar(
        W66_MASC_V2_REGIME_BASELINE, "v19_beats_v18_rate",
        "h920_baseline_v19_beats_v18")


def family_h920b_baseline_tsc(seed: int) -> dict[str, Any]:
    return _bar(
        W66_MASC_V2_REGIME_BASELINE,
        "tsc_v19_beats_tsc_v18_rate",
        "h920b_baseline_tsc_v19_beats_tsc_v18")


def family_h921_tcub_v19(seed: int) -> dict[str, Any]:
    return _bar(
        W66_MASC_V2_REGIME_TEAM_CONSENSUS_UNDER_BUDGET,
        "v19_beats_v18_rate", "h921_tcub_v19_beats_v18")


def family_h921b_tcub_tsc(seed: int) -> dict[str, Any]:
    return _bar(
        W66_MASC_V2_REGIME_TEAM_CONSENSUS_UNDER_BUDGET,
        "tsc_v19_beats_tsc_v18_rate", "h921b_tcub_tsc")


def family_h922_tfr_v19(seed: int) -> dict[str, Any]:
    return _bar(
        W66_MASC_V2_REGIME_TEAM_FAILURE_RECOVERY,
        "v19_beats_v18_rate", "h922_tfr_v19_beats_v18")


def family_h922b_tfr_tsc(seed: int) -> dict[str, Any]:
    return _bar(
        W66_MASC_V2_REGIME_TEAM_FAILURE_RECOVERY,
        "tsc_v19_beats_tsc_v18_rate", "h922b_tfr_tsc")


def family_h923_role_dropout(seed: int) -> dict[str, Any]:
    return _bar(
        W67_MASC_V3_REGIME_ROLE_DROPOUT,
        "v19_beats_v18_rate",
        "h923_role_dropout_v19_beats_v18")


def family_h924_branch_merge(seed: int) -> dict[str, Any]:
    return _bar(
        W67_MASC_V3_REGIME_BRANCH_MERGE_RECONCILIATION,
        "v19_beats_v18_rate",
        "h924_branch_merge_v19_beats_v18")


def family_h925_pc(seed: int) -> dict[str, Any]:
    return _bar(
        W68_MASC_V4_REGIME_PARTIAL_CONTRADICTION,
        "v19_beats_v18_rate",
        "h925_pc_v19_beats_v18")


def family_h925b_pc_tsc(seed: int) -> dict[str, Any]:
    return _bar(
        W68_MASC_V4_REGIME_PARTIAL_CONTRADICTION,
        "tsc_v19_beats_tsc_v18_rate", "h925b_pc_tsc")


def family_h926_ar(seed: int) -> dict[str, Any]:
    return _bar(
        W68_MASC_V4_REGIME_AGENT_REPLACEMENT_WARM_RESTART,
        "v19_beats_v18_rate",
        "h926_ar_v19_beats_v18")


def family_h926b_ar_tsc(seed: int) -> dict[str, Any]:
    return _bar(
        W68_MASC_V4_REGIME_AGENT_REPLACEMENT_WARM_RESTART,
        "tsc_v19_beats_tsc_v18_rate", "h926b_ar_tsc")


def family_h927_mbr(seed: int) -> dict[str, Any]:
    return _bar(
        W69_MASC_V5_REGIME_MULTI_BRANCH_REJOIN,
        "v19_beats_v18_rate",
        "h927_mbr_v19_beats_v18")


def family_h927b_mbr_tsc(seed: int) -> dict[str, Any]:
    return _bar(
        W69_MASC_V5_REGIME_MULTI_BRANCH_REJOIN,
        "tsc_v19_beats_tsc_v18_rate", "h927b_mbr_tsc")


def family_h928_sc(seed: int) -> dict[str, Any]:
    return _bar(
        W69_MASC_V5_REGIME_SILENT_CORRUPTION_PLUS_REPLACEMENT,
        "v19_beats_v18_rate",
        "h928_sc_v19_beats_v18")


def family_h928b_sc_tsc(seed: int) -> dict[str, Any]:
    return _bar(
        W69_MASC_V5_REGIME_SILENT_CORRUPTION_PLUS_REPLACEMENT,
        "tsc_v19_beats_tsc_v18_rate", "h928b_sc_tsc")


def family_h929_ctr(seed: int) -> dict[str, Any]:
    return _bar(
        W70_MASC_V6_REGIME_CONTRADICTION_THEN_REJOIN_UNDER_BUDGET,
        "v19_beats_v18_rate",
        "h929_ctr_v19_beats_v18")


def family_h929b_ctr_tsc(seed: int) -> dict[str, Any]:
    return _bar(
        W70_MASC_V6_REGIME_CONTRADICTION_THEN_REJOIN_UNDER_BUDGET,
        "tsc_v19_beats_tsc_v18_rate", "h929b_ctr_tsc")


def family_h930_drar(seed: int) -> dict[str, Any]:
    return _bar(
        W71_MASC_V7_REGIME_DELAYED_REPAIR_AFTER_RESTART,
        "v19_beats_v18_rate",
        "h930_drar_v19_beats_v18")


def family_h930b_drar_tsc(seed: int) -> dict[str, Any]:
    return _bar(
        W71_MASC_V7_REGIME_DELAYED_REPAIR_AFTER_RESTART,
        "tsc_v19_beats_tsc_v18_rate", "h930b_drar_tsc")


def family_h931_drrjub(seed: int) -> dict[str, Any]:
    return _bar(
        W72_MASC_V8_REGIME_DELAYED_REJOIN_AFTER_RESTART_UNDER_BUDGET,
        "v19_beats_v18_rate",
        "h931_drrjub_v19_beats_v18")


def family_h931b_drrjub_tsc(seed: int) -> dict[str, Any]:
    return _bar(
        W72_MASC_V8_REGIME_DELAYED_REJOIN_AFTER_RESTART_UNDER_BUDGET,
        "tsc_v19_beats_tsc_v18_rate", "h931b_drrjub_tsc")


def family_h932_repctr(seed: int) -> dict[str, Any]:
    return _bar(
        W73_MASC_V9_REGIME_REPLACEMENT_AFTER_CTR,
        "v19_beats_v18_rate",
        "h932_repctr_v19_beats_v18")


def family_h932b_repctr_tsc(seed: int) -> dict[str, Any]:
    return _bar(
        W73_MASC_V9_REGIME_REPLACEMENT_AFTER_CTR,
        "tsc_v19_beats_tsc_v18_rate", "h932b_repctr_tsc")


def family_h933_cmpdr(seed: int) -> dict[str, Any]:
    return _bar(
        W74_MASC_V10_REGIME_REPLACEMENT_AFTER_DELAYED_REPAIR,
        "v19_beats_v18_rate",
        "h933_cmpdr_v19_beats_v18")


def family_h933b_cmpdr_tsc(seed: int) -> dict[str, Any]:
    return _bar(
        W74_MASC_V10_REGIME_REPLACEMENT_AFTER_DELAYED_REPAIR,
        "tsc_v19_beats_tsc_v18_rate", "h933b_cmpdr_tsc")


def family_h934_v19_visible_tokens_savings(
        seed: int) -> dict[str, Any]:
    r = _run_masc_v10_regime(
        W66_MASC_V2_REGIME_BASELINE, seeds=10)
    return {
        "schema": R179_SCHEMA_VERSION,
        "name": "h934_v19_visible_tokens_savings",
        "passed": bool(
            r["v19_visible_tokens_savings_vs_transcript"]
            >= 0.65),
        "savings": float(
            r["v19_visible_tokens_savings_vs_transcript"]),
    }


def family_h935_v19_ts_per_visible_token_beats_v18(
        seed: int) -> dict[str, Any]:
    r = _run_masc_v10_regime(
        W66_MASC_V2_REGIME_BASELINE, seeds=10)
    v19_ts = float(r["team_success_per_visible_token_v19"])
    sr = dict(r["per_policy_success_rate"])
    v18_sr = float(sr.get(
        W73_MASC_V9_POLICY_SUBSTRATE_ROUTED_V18, 0.0))
    v19_sr = float(sr.get(
        W74_MASC_V10_POLICY_SUBSTRATE_ROUTED_V19, 0.0))
    return {
        "schema": R179_SCHEMA_VERSION,
        "name":
            "h935_v19_ts_per_visible_token_beats_v18",
        "passed": bool(v19_ts > 0.0 and v19_sr >= v18_sr),
        "v19_ts_per_vt": float(v19_ts),
        "v19_sr": float(v19_sr),
        "v18_sr": float(v18_sr),
    }


def family_h936_tcc_v9_compound_pressure_arbiter(
        seed: int) -> dict[str, Any]:
    tcc = TeamConsensusControllerV9()
    out = tcc.decide_v9(
        regime=W66_MASC_V2_REGIME_BASELINE,
        agent_guesses=[1.0, -1.0, 0.5, 0.2],
        agent_confidences=[0.8, 0.6, 0.7, 0.7],
        substrate_replay_trust=0.7,
        compound_pressure=0.8,
        agent_compound_recovery_flags=[1, 0, 1, 0])
    return {
        "schema": R179_SCHEMA_VERSION,
        "name": "h936_tcc_v9_compound_pressure_arbiter",
        "passed": bool(
            out["decision"]
            == W74_TC_V9_DECISION_COMPOUND_PRESSURE),
    }


def family_h937_tcc_v9_compound_repair_drtr_arbiter(
        seed: int) -> dict[str, Any]:
    tcc = TeamConsensusControllerV9()
    out = tcc.decide_v9(
        regime=(
            W74_MASC_V10_REGIME_REPLACEMENT_AFTER_DELAYED_REPAIR),
        agent_guesses=[1.0, -1.0, 0.5, 0.2],
        agent_confidences=[0.8, 0.6, 0.7, 0.7],
        substrate_replay_trust=0.7,
        compound_repair_trajectory_cid="cmp_abc",
        compound_window_turns=8,
        agent_compound_absorption_scores=[
            0.95, 0.5, 0.4, 0.3])
    return {
        "schema": R179_SCHEMA_VERSION,
        "name": "h937_tcc_v9_compound_repair_drtr_arbiter",
        "passed": bool(
            out["decision"]
            == W74_TC_V9_DECISION_COMPOUND_REPAIR_AFTER_DRTR),
    }


_R179_FAMILIES: tuple[Any, ...] = (
    family_h920_baseline_v19,
    family_h920b_baseline_tsc,
    family_h921_tcub_v19,
    family_h921b_tcub_tsc,
    family_h922_tfr_v19,
    family_h922b_tfr_tsc,
    family_h923_role_dropout,
    family_h924_branch_merge,
    family_h925_pc,
    family_h925b_pc_tsc,
    family_h926_ar,
    family_h926b_ar_tsc,
    family_h927_mbr,
    family_h927b_mbr_tsc,
    family_h928_sc,
    family_h928b_sc_tsc,
    family_h929_ctr,
    family_h929b_ctr_tsc,
    family_h930_drar,
    family_h930b_drar_tsc,
    family_h931_drrjub,
    family_h931b_drrjub_tsc,
    family_h932_repctr,
    family_h932b_repctr_tsc,
    family_h933_cmpdr,
    family_h933b_cmpdr_tsc,
    family_h934_v19_visible_tokens_savings,
    family_h935_v19_ts_per_visible_token_beats_v18,
    family_h936_tcc_v9_compound_pressure_arbiter,
    family_h937_tcc_v9_compound_repair_drtr_arbiter,
)


def run_r179(
        seeds: Sequence[int] = (199, 299, 399),
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for s in seeds:
        results: dict[str, Any] = {}
        for fn in _R179_FAMILIES:
            try:
                r = fn(int(s))
            except Exception as e:
                r = {
                    "schema": R179_SCHEMA_VERSION,
                    "name": getattr(fn, "__name__", "unknown"),
                    "passed": False,
                    "exception": str(e),
                }
            results[r["name"]] = r
        out.append({
            "schema": R179_SCHEMA_VERSION,
            "seed": int(s),
            "family_results": results,
        })
    return out


__all__ = ["R179_SCHEMA_VERSION", "run_r179"]
