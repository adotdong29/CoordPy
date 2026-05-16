"""W71 envelope chain test.

The W71 envelope's ``w70_outer_cid`` carries the supplied W70 outer
CID byte-for-byte and the verifier enumerates ≥ 50 disjoint failure
modes.
"""

from __future__ import annotations


def test_w71_team_envelope_chain():
    from coordpy.w71_team import (
        W71_FAILURE_MODES,
        W71Params,
        W71Team,
        verify_w71_handoff,
    )
    p = W71Params.build_default(seed=71000)
    team = W71Team(params=p)
    env = team.run_team_turn(
        w70_outer_cid="W70_OUTER_TEST_CID_71000",
        text="w71_envelope_chain_test")
    assert env.w70_outer_cid == "W70_OUTER_TEST_CID_71000"
    assert env.substrate_v16_used
    assert env.sixteen_way_used
    ok, fails = verify_w71_handoff(
        env, p, "W70_OUTER_TEST_CID_71000")
    assert ok, fails
    assert len(W71_FAILURE_MODES) >= 50
    assert (
        env.masc_v7_v16_beats_v15_rate >= 0.5
        and env.masc_v7_tsc_v16_beats_tsc_v15_rate >= 0.5)
    assert env.delayed_repair_trajectory_cid != ""
    assert env.handoff_envelope_v3_chain_cid != ""
    assert env.provider_filter_v3_report_cid != ""
