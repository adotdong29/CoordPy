"""W70 envelope chain test.

The W70 envelope's ``w69_outer_cid`` carries the supplied W69 outer
CID byte-for-byte and the verifier enumerates ≥ 50 disjoint failure
modes.
"""

from __future__ import annotations


def test_w70_team_envelope_chain():
    from coordpy.w70_team import (
        W70_FAILURE_MODES,
        W70Params,
        W70Team,
        verify_w70_handoff,
    )
    p = W70Params.build_default(seed=70000)
    team = W70Team(params=p)
    env = team.run_team_turn(
        w69_outer_cid="W69_OUTER_TEST_CID_70000",
        text="w70_envelope_chain_test")
    assert env.w69_outer_cid == "W69_OUTER_TEST_CID_70000"
    assert env.substrate_v15_used
    assert env.fifteen_way_used
    ok, fails = verify_w70_handoff(
        env, p, "W69_OUTER_TEST_CID_70000")
    assert ok, fails
    assert len(W70_FAILURE_MODES) >= 50
    assert (
        env.masc_v6_v15_beats_v14_rate >= 0.5
        and env.masc_v6_tsc_v15_beats_tsc_v14_rate >= 0.5)
    assert env.repair_trajectory_cid != ""
    assert env.handoff_envelope_v2_chain_cid != ""
