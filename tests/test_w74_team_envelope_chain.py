"""W74 envelope chain test.

The W74 envelope's ``w73_outer_cid`` carries the supplied W73 outer
CID byte-for-byte and the verifier enumerates ≥ 55 disjoint failure
modes.
"""

from __future__ import annotations


def test_w74_team_envelope_chain():
    from coordpy.w74_team import (
        W74_FAILURE_MODES,
        W74Params,
        W74Team,
        verify_w74_handoff,
    )
    p = W74Params.build_default(seed=74000)
    team = W74Team(params=p)
    env = team.run_team_turn(
        w73_outer_cid="W73_OUTER_TEST_CID_74000",
        text="w74_envelope_chain_test")
    assert env.w73_outer_cid == "W73_OUTER_TEST_CID_74000"
    assert env.substrate_v19_used
    assert env.nineteen_way_used
    ok, fails = verify_w74_handoff(
        env, p, "W73_OUTER_TEST_CID_74000")
    assert ok, fails
    assert len(W74_FAILURE_MODES) >= 55
    assert (
        env.masc_v10_v19_beats_v18_rate >= 0.5
        and env.masc_v10_tsc_v19_beats_tsc_v18_rate >= 0.5)
    assert env.compound_repair_trajectory_cid != ""
    assert env.handoff_envelope_v6_chain_cid != ""
    assert env.provider_filter_v6_report_cid != ""
