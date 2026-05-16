"""W69 envelope chain test.

The W69 envelope's ``w68_outer_cid`` carries the supplied W68 outer
CID byte-for-byte and the verifier enumerates ≥ 42 disjoint failure
modes.
"""

from __future__ import annotations


def test_w69_team_envelope_chain():
    from coordpy.w69_team import (
        W69_FAILURE_MODES,
        W69Params,
        W69Team,
        verify_w69_handoff,
    )
    p = W69Params.build_default(seed=69000)
    team = W69Team(params=p)
    env = team.run_team_turn(
        w68_outer_cid="W68_OUTER_TEST_CID_69000",
        text="w69_envelope_chain_test")
    assert env.w68_outer_cid == "W68_OUTER_TEST_CID_69000"
    assert env.substrate_v14_used
    assert env.fourteen_way_used
    ok, fails = verify_w69_handoff(
        env, p, "W68_OUTER_TEST_CID_69000")
    assert ok, fails
    assert len(W69_FAILURE_MODES) >= 42
    assert (
        env.masc_v5_v14_beats_v13_rate >= 0.5
        and env.masc_v5_tsc_v14_beats_tsc_v13_rate >= 0.5)
    assert env.handoff_envelope_chain_cid != ""
