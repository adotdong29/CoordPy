"""W75 tests — MASC V11 + TCC V10."""

from __future__ import annotations

from coordpy.multi_agent_substrate_coordinator_v11 import (
    MultiAgentSubstrateCoordinatorV11,
    W75_MASC_V11_POLICIES,
    W75_MASC_V11_REGIME_COMPOUND_CHAIN,
    W75_MASC_V11_REGIMES,
)
from coordpy.team_consensus_controller_v10 import (
    TeamConsensusControllerV10,
    W75_TC_V10_DECISION_COMPOUND_CHAIN_PRESSURE,
    W75_TC_V10_DECISION_COMPOUND_REPAIR_AFTER_RTR,
)


def test_masc_v11_has_fifteen_regimes() -> None:
    assert len(W75_MASC_V11_REGIMES) == 15
    assert W75_MASC_V11_REGIME_COMPOUND_CHAIN in (
        W75_MASC_V11_REGIMES)


def test_masc_v11_has_twenty_four_policies() -> None:
    assert (
        "substrate_routed_v20" in W75_MASC_V11_POLICIES
        and "team_substrate_coordination_v20"
        in W75_MASC_V11_POLICIES)


def test_masc_v11_v20_beats_v19_each_regime() -> None:
    masc = MultiAgentSubstrateCoordinatorV11()
    seeds = list(range(5))
    for regime in W75_MASC_V11_REGIMES:
        _, agg = masc.run_batch(seeds=seeds, regime=regime)
        assert (
            agg.v20_beats_v19_rate >= 0.5
        ), f"{regime}: v20 only beats v19 in {agg.v20_beats_v19_rate}"
        assert (
            agg.tsc_v20_beats_tsc_v19_rate >= 0.5
        ), f"{regime}: tsc_v20 only beats tsc_v19"


def test_tcc_v10_compound_chain_arbiter_fires() -> None:
    ctrl = TeamConsensusControllerV10()
    out = ctrl.decide_v10(
        regime="baseline",
        agent_guesses=[1.0, -1.0, 0.5, 0.2],
        agent_confidences=[0.8, 0.6, 0.7, 0.7],
        substrate_replay_trust=0.7,
        compound_chain_pressure=0.8,
        agent_compound_chain_recovery_flags=[1, 0, 1, 0])
    assert out["decision"] == (
        W75_TC_V10_DECISION_COMPOUND_CHAIN_PRESSURE)


def test_tcc_v10_compound_rtr_arbiter_fires() -> None:
    ctrl = TeamConsensusControllerV10()
    out = ctrl.decide_v10(
        regime=W75_MASC_V11_REGIME_COMPOUND_CHAIN,
        agent_guesses=[1.0, -1.0, 0.5, 0.2],
        agent_confidences=[0.8, 0.6, 0.7, 0.7],
        substrate_replay_trust=0.7,
        compound_chain_repair_trajectory_cid="x",
        compound_chain_window_turns=11,
        agent_compound_chain_absorption_scores=[
            0.95, 0.5, 0.4, 0.3])
    assert out["decision"] == (
        W75_TC_V10_DECISION_COMPOUND_REPAIR_AFTER_RTR)
