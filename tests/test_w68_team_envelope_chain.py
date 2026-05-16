"""W68 envelope chain test.

The W68 envelope's ``w67_outer_cid`` carries the supplied W67 outer
CID byte-for-byte and the verifier enumerates ≥ 40 disjoint failure
modes.
"""

from __future__ import annotations


def test_w68_team_envelope_chain():
    from coordpy.w68_team import (
        W68_FAILURE_MODES,
        W68Params,
        W68Team,
        verify_w68_handoff,
    )
    p = W68Params.build_default(seed=68000)
    team = W68Team(params=p)
    env = team.run_team_turn(
        w67_outer_cid="W67_OUTER_TEST_CID_68000",
        text="w68_envelope_chain_test")
    assert env.w67_outer_cid == "W67_OUTER_TEST_CID_68000"
    assert env.substrate_v13_used
    assert env.thirteen_way_used
    ok, fails = verify_w68_handoff(
        env, p, "W67_OUTER_TEST_CID_68000")
    assert ok, fails
    assert len(W68_FAILURE_MODES) >= 40
    assert (
        env.masc_v4_v13_beats_v12_rate >= 0.5
        and env.masc_v4_tsc_v13_beats_tsc_v12_rate >= 0.5)
