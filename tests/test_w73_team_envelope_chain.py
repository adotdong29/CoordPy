"""W73 envelope chain test.

The W73 envelope's ``w72_outer_cid`` carries the supplied W72 outer
CID byte-for-byte and the verifier enumerates ≥ 54 disjoint failure
modes.
"""

from __future__ import annotations


def test_w73_team_envelope_chain():
    from coordpy.w73_team import (
        W73_FAILURE_MODES,
        W73Params,
        W73Team,
        verify_w73_handoff,
    )
    p = W73Params.build_default(seed=73000)
    team = W73Team(params=p)
    env = team.run_team_turn(
        w72_outer_cid="W72_OUTER_TEST_CID_73000",
        text="w73_envelope_chain_test")
    assert env.w72_outer_cid == "W72_OUTER_TEST_CID_73000"
    assert env.substrate_v18_used
    assert env.eighteen_way_used
    ok, fails = verify_w73_handoff(
        env, p, "W72_OUTER_TEST_CID_73000")
    assert ok, fails
    assert len(W73_FAILURE_MODES) >= 54
    assert (
        env.masc_v9_v18_beats_v17_rate >= 0.5
        and env.masc_v9_tsc_v18_beats_tsc_v17_rate >= 0.5)
    assert env.replacement_repair_trajectory_cid != ""
    assert env.handoff_envelope_v5_chain_cid != ""
    assert env.provider_filter_v5_report_cid != ""
