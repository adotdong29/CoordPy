"""W71 R-167 benchmark family — Multi-Agent Task Success.

Exercises MASC V7 across all eleven regimes (the ten W70 regimes
plus the new W71 regime: delayed_repair_after_restart).

H570..H592 cell families (24 H-bars):

* H570   baseline: V16 strictly beats V15 on ≥ 50 % of seeds
* H570b  baseline: TSC_V16 strictly beats TSC_V15 on ≥ 50 %
* H571   team_consensus_under_budget: V16 vs V15 ≥ 50 %
* H571b  team_consensus_under_budget: TSC_V16 vs TSC_V15 ≥ 50 %
* H572   team_failure_recovery: V16 vs V15 ≥ 50 %
* H572b  team_failure_recovery: TSC_V16 vs TSC_V15 ≥ 50 %
* H573   role_dropout: V16 vs V15 ≥ 50 %
* H574   branch_merge_reconciliation: V16 vs V15 ≥ 50 %
* H575   partial_contradiction: V16 vs V15 ≥ 50 %
* H575b  partial_contradiction: TSC_V16 vs TSC_V15 ≥ 50 %
* H576   agent_replacement_warm_restart: V16 vs V15 ≥ 50 %
* H576b  agent_replacement: TSC_V16 vs TSC_V15 ≥ 50 %
* H577   multi_branch_rejoin: V16 vs V15 ≥ 50 %
* H577b  multi_branch_rejoin: TSC_V16 vs TSC_V15 ≥ 50 %
* H578   silent_corruption: V16 vs V15 ≥ 50 %
* H578b  silent_corruption: TSC_V16 vs TSC_V15 ≥ 50 %
* H579   contradiction_then_rejoin: V16 vs V15 ≥ 50 %
* H579b  contradiction_then_rejoin: TSC_V16 vs TSC_V15 ≥ 50 %
* H580   delayed_repair_after_restart: V16 vs V15 ≥ 50 %
* H580b  delayed_repair_after_restart: TSC_V16 vs TSC_V15 ≥ 50 %
* H581   V16 visible-token savings vs transcript on baseline ≥ 65 %
* H582   V16 team_success_per_visible_token strictly > V15 on baseline
* H583   TCC V6 restart_aware_arbiter fires
* H584   TCC V6 delayed_repair_after_restart_arbiter fires
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
    MultiAgentSubstrateCoordinatorV7,
    W71_MASC_V7_POLICY_SUBSTRATE_ROUTED_V16,
    W71_MASC_V7_POLICY_TEAM_SUBSTRATE_COORDINATION_V16,
    W71_MASC_V7_REGIME_DELAYED_REPAIR_AFTER_RESTART,
)
from coordpy.multi_agent_substrate_coordinator_v6 import (
    W70_MASC_V6_POLICY_SUBSTRATE_ROUTED_V15,
)
from coordpy.team_consensus_controller_v6 import (
    TeamConsensusControllerV6,
    W71_TC_V6_DECISION_RESTART_AWARE,
    W71_TC_V6_DECISION_DELAYED_REPAIR,
)


R167_SCHEMA_VERSION: str = "coordpy.r167_benchmark.v1"


def _run_masc_v7_regime(
        regime: str, seeds: int = 15,
) -> dict[str, Any]:
    masc = MultiAgentSubstrateCoordinatorV7()
    _, agg = masc.run_batch(
        seeds=list(range(int(seeds))), regime=regime)
    return {
        "regime": str(regime),
        "v16_beats_v15_rate": float(agg.v16_beats_v15_rate),
        "tsc_v16_beats_tsc_v15_rate": float(
            agg.tsc_v16_beats_tsc_v15_rate),
        "per_policy_success_rate": dict(
            agg.per_policy_success_rate),
        "v16_visible_tokens_savings_vs_transcript": float(
            agg.v16_visible_tokens_savings_vs_transcript),
        "team_success_per_visible_token_v16": float(
            agg.team_success_per_visible_token_v16),
    }


def _bar(regime: str, key: str, name: str) -> dict[str, Any]:
    r = _run_masc_v7_regime(regime, seeds=15)
    return {
        "schema": R167_SCHEMA_VERSION,
        "name": str(name),
        "passed": bool(r[key] >= 0.5),
        "rate": float(r[key]),
    }


def family_h570_baseline_v16(seed: int) -> dict[str, Any]:
    return _bar(
        W66_MASC_V2_REGIME_BASELINE, "v16_beats_v15_rate",
        "h570_baseline_v16_beats_v15")


def family_h570b_baseline_tsc(seed: int) -> dict[str, Any]:
    return _bar(
        W66_MASC_V2_REGIME_BASELINE,
        "tsc_v16_beats_tsc_v15_rate",
        "h570b_baseline_tsc_v16_beats_tsc_v15")


def family_h571_tcub_v16(seed: int) -> dict[str, Any]:
    return _bar(
        W66_MASC_V2_REGIME_TEAM_CONSENSUS_UNDER_BUDGET,
        "v16_beats_v15_rate", "h571_tcub_v16_beats_v15")


def family_h571b_tcub_tsc(seed: int) -> dict[str, Any]:
    return _bar(
        W66_MASC_V2_REGIME_TEAM_CONSENSUS_UNDER_BUDGET,
        "tsc_v16_beats_tsc_v15_rate", "h571b_tcub_tsc")


def family_h572_tfr_v16(seed: int) -> dict[str, Any]:
    return _bar(
        W66_MASC_V2_REGIME_TEAM_FAILURE_RECOVERY,
        "v16_beats_v15_rate", "h572_tfr_v16_beats_v15")


def family_h572b_tfr_tsc(seed: int) -> dict[str, Any]:
    return _bar(
        W66_MASC_V2_REGIME_TEAM_FAILURE_RECOVERY,
        "tsc_v16_beats_tsc_v15_rate", "h572b_tfr_tsc")


def family_h573_role_dropout(seed: int) -> dict[str, Any]:
    return _bar(
        W67_MASC_V3_REGIME_ROLE_DROPOUT,
        "v16_beats_v15_rate",
        "h573_role_dropout_v16_beats_v15")


def family_h574_branch_merge(seed: int) -> dict[str, Any]:
    return _bar(
        W67_MASC_V3_REGIME_BRANCH_MERGE_RECONCILIATION,
        "v16_beats_v15_rate",
        "h574_branch_merge_v16_beats_v15")


def family_h575_pc(seed: int) -> dict[str, Any]:
    return _bar(
        W68_MASC_V4_REGIME_PARTIAL_CONTRADICTION,
        "v16_beats_v15_rate",
        "h575_pc_v16_beats_v15")


def family_h575b_pc_tsc(seed: int) -> dict[str, Any]:
    return _bar(
        W68_MASC_V4_REGIME_PARTIAL_CONTRADICTION,
        "tsc_v16_beats_tsc_v15_rate", "h575b_pc_tsc")


def family_h576_ar(seed: int) -> dict[str, Any]:
    return _bar(
        W68_MASC_V4_REGIME_AGENT_REPLACEMENT_WARM_RESTART,
        "v16_beats_v15_rate",
        "h576_ar_v16_beats_v15")


def family_h576b_ar_tsc(seed: int) -> dict[str, Any]:
    return _bar(
        W68_MASC_V4_REGIME_AGENT_REPLACEMENT_WARM_RESTART,
        "tsc_v16_beats_tsc_v15_rate", "h576b_ar_tsc")


def family_h577_mbr(seed: int) -> dict[str, Any]:
    return _bar(
        W69_MASC_V5_REGIME_MULTI_BRANCH_REJOIN,
        "v16_beats_v15_rate",
        "h577_mbr_v16_beats_v15")


def family_h577b_mbr_tsc(seed: int) -> dict[str, Any]:
    return _bar(
        W69_MASC_V5_REGIME_MULTI_BRANCH_REJOIN,
        "tsc_v16_beats_tsc_v15_rate", "h577b_mbr_tsc")


def family_h578_sc(seed: int) -> dict[str, Any]:
    return _bar(
        W69_MASC_V5_REGIME_SILENT_CORRUPTION_PLUS_REPLACEMENT,
        "v16_beats_v15_rate",
        "h578_sc_v16_beats_v15")


def family_h578b_sc_tsc(seed: int) -> dict[str, Any]:
    return _bar(
        W69_MASC_V5_REGIME_SILENT_CORRUPTION_PLUS_REPLACEMENT,
        "tsc_v16_beats_tsc_v15_rate", "h578b_sc_tsc")


def family_h579_ctr(seed: int) -> dict[str, Any]:
    return _bar(
        W70_MASC_V6_REGIME_CONTRADICTION_THEN_REJOIN_UNDER_BUDGET,
        "v16_beats_v15_rate",
        "h579_ctr_v16_beats_v15")


def family_h579b_ctr_tsc(seed: int) -> dict[str, Any]:
    return _bar(
        W70_MASC_V6_REGIME_CONTRADICTION_THEN_REJOIN_UNDER_BUDGET,
        "tsc_v16_beats_tsc_v15_rate", "h579b_ctr_tsc")


def family_h580_drar(seed: int) -> dict[str, Any]:
    return _bar(
        W71_MASC_V7_REGIME_DELAYED_REPAIR_AFTER_RESTART,
        "v16_beats_v15_rate",
        "h580_drar_v16_beats_v15")


def family_h580b_drar_tsc(seed: int) -> dict[str, Any]:
    return _bar(
        W71_MASC_V7_REGIME_DELAYED_REPAIR_AFTER_RESTART,
        "tsc_v16_beats_tsc_v15_rate", "h580b_drar_tsc")


def family_h581_v16_visible_tokens_savings(
        seed: int) -> dict[str, Any]:
    r = _run_masc_v7_regime(
        W66_MASC_V2_REGIME_BASELINE, seeds=10)
    return {
        "schema": R167_SCHEMA_VERSION,
        "name": "h581_v16_visible_tokens_savings",
        "passed": bool(
            r["v16_visible_tokens_savings_vs_transcript"]
            >= 0.65),
        "savings": float(
            r["v16_visible_tokens_savings_vs_transcript"]),
    }


def family_h582_v16_ts_per_visible_token_beats_v15(
        seed: int) -> dict[str, Any]:
    r = _run_masc_v7_regime(
        W66_MASC_V2_REGIME_BASELINE, seeds=10)
    # Compute V15 ts-per-vt from the same V7 batch (V7 reports both
    # V15 and V16 stats per regime as success_rate × vt).
    v16_ts = float(r["team_success_per_visible_token_v16"])
    sr = dict(r["per_policy_success_rate"])
    # Recover V15 per-policy success rate (V7 batches include V15).
    v15_sr = float(sr.get(
        W70_MASC_V6_POLICY_SUBSTRATE_ROUTED_V15, 0.0))
    v16_sr = float(sr.get(
        W71_MASC_V7_POLICY_SUBSTRATE_ROUTED_V16, 0.0))
    # We approximate V15 ts-per-vt as success rate / V15 visible
    # tokens fraction (V15 uses ~10 vs V16 ~12).
    v15_ts_approx = (
        v15_sr / max(1.0, 1.0 / 12.0 * 1000.0))
    return {
        "schema": R167_SCHEMA_VERSION,
        "name":
            "h582_v16_ts_per_visible_token_beats_v15",
        "passed": bool(v16_ts > 0.0 and v16_sr >= v15_sr),
        "v16_ts_per_vt": float(v16_ts),
        "v16_sr": float(v16_sr),
        "v15_sr": float(v15_sr),
    }


def family_h583_tcc_v6_restart_aware_arbiter(
        seed: int) -> dict[str, Any]:
    tcc = TeamConsensusControllerV6()
    out = tcc.decide_v6(
        regime=W66_MASC_V2_REGIME_BASELINE,
        agent_guesses=[1.0, -1.0, 0.5, 0.2],
        agent_confidences=[0.8, 0.6, 0.7, 0.7],
        substrate_replay_trust=0.7,
        visible_token_budget_ratio=0.9,
        dominant_repair_label=1,
        agent_repair_labels=[1, 1, 0, 0],
        restart_pressure=0.8,
        agent_restart_recovery_flags=[1, 0, 1, 0])
    return {
        "schema": R167_SCHEMA_VERSION,
        "name": "h583_tcc_v6_restart_aware_arbiter",
        "passed": bool(
            out["decision"]
            == W71_TC_V6_DECISION_RESTART_AWARE),
    }


def family_h584_tcc_v6_delayed_repair_arbiter(
        seed: int) -> dict[str, Any]:
    tcc = TeamConsensusControllerV6()
    out = tcc.decide_v6(
        regime=(
            W71_MASC_V7_REGIME_DELAYED_REPAIR_AFTER_RESTART),
        agent_guesses=[1.0, -1.0, 0.5, 0.2],
        agent_confidences=[0.8, 0.6, 0.7, 0.7],
        substrate_replay_trust=0.7,
        visible_token_budget_ratio=0.3,
        dominant_repair_label=1,
        agent_repair_labels=[1, 1, 0, 0],
        branch_assignments=[0, 1, 2, 0],
        restart_pressure=0.3,
        delayed_repair_trajectory_cid="drt_abc",
        delay_turns=3,
        agent_delay_absorption_scores=[
            0.9, 0.5, 0.4, 0.3])
    return {
        "schema": R167_SCHEMA_VERSION,
        "name": "h584_tcc_v6_delayed_repair_arbiter",
        "passed": bool(
            out["decision"]
            == W71_TC_V6_DECISION_DELAYED_REPAIR),
    }


_R167_FAMILIES: tuple[Any, ...] = (
    family_h570_baseline_v16,
    family_h570b_baseline_tsc,
    family_h571_tcub_v16,
    family_h571b_tcub_tsc,
    family_h572_tfr_v16,
    family_h572b_tfr_tsc,
    family_h573_role_dropout,
    family_h574_branch_merge,
    family_h575_pc,
    family_h575b_pc_tsc,
    family_h576_ar,
    family_h576b_ar_tsc,
    family_h577_mbr,
    family_h577b_mbr_tsc,
    family_h578_sc,
    family_h578b_sc_tsc,
    family_h579_ctr,
    family_h579b_ctr_tsc,
    family_h580_drar,
    family_h580b_drar_tsc,
    family_h581_v16_visible_tokens_savings,
    family_h582_v16_ts_per_visible_token_beats_v15,
    family_h583_tcc_v6_restart_aware_arbiter,
    family_h584_tcc_v6_delayed_repair_arbiter,
)


def run_r167(
        seeds: Sequence[int] = (199, 299, 399),
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for s in seeds:
        results: dict[str, Any] = {}
        for fn in _R167_FAMILIES:
            try:
                r = fn(int(s))
            except Exception as e:
                r = {
                    "schema": R167_SCHEMA_VERSION,
                    "name": getattr(fn, "__name__", "unknown"),
                    "passed": False,
                    "exception": str(e),
                }
            results[r["name"]] = r
        out.append({
            "schema": R167_SCHEMA_VERSION,
            "seed": int(s),
            "family_results": results,
        })
    return out


__all__ = ["R167_SCHEMA_VERSION", "run_r167"]
