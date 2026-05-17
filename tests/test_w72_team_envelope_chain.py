"""W72 envelope chain test.

The W72 envelope's ``w71_outer_cid`` carries the supplied W71 outer
CID byte-for-byte and the verifier enumerates ≥ 50 disjoint failure
modes.
"""

from __future__ import annotations


def test_w72_team_envelope_chain():
    from coordpy.w72_team import (
        W72_FAILURE_MODES,
        W72Params,
        W72Team,
        verify_w72_handoff,
    )
    p = W72Params.build_default(seed=72000)
    team = W72Team(params=p)
    env = team.run_team_turn(
        w71_outer_cid="W71_OUTER_TEST_CID_72000",
        text="w72_envelope_chain_test")
    assert env.w71_outer_cid == "W71_OUTER_TEST_CID_72000"
    assert env.substrate_v17_used
    assert env.seventeen_way_used
    ok, fails = verify_w72_handoff(
        env, p, "W71_OUTER_TEST_CID_72000")
    assert ok, fails
    assert len(W72_FAILURE_MODES) >= 50
    assert (
        env.masc_v8_v17_beats_v16_rate >= 0.5
        and env.masc_v8_tsc_v17_beats_tsc_v16_rate >= 0.5)
    assert env.restart_repair_trajectory_cid != ""
    assert env.handoff_envelope_v4_chain_cid != ""
    assert env.provider_filter_v4_report_cid != ""
