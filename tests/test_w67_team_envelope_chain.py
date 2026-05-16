"""W67 envelope chain test.

The W67 envelope's ``w66_outer_cid`` carries the supplied W66 outer
CID byte-for-byte and the verifier enumerates ≥ 140 disjoint failure
modes.
"""

from __future__ import annotations


def test_w67_team_envelope_chain():
    from coordpy.w67_team import (
        W67_ENVELOPE_VERIFIER_FAILURE_MODES,
        W67Params,
        W67Team,
        verify_w67_handoff,
    )
    p = W67Params.build_default(seed=67000)
    team = W67Team(params=p)
    env = team.step(
        turn_index=0, role="planner",
        w66_outer_cid="W66_OUTER_TEST_CID_67000")
    assert env.w66_outer_cid == "W66_OUTER_TEST_CID_67000"
    assert env.substrate_v12_used
    assert env.twelve_way_used
    res = verify_w67_handoff(env)
    assert res["ok"]
    assert len(W67_ENVELOPE_VERIFIER_FAILURE_MODES) >= 140
    assert (
        env.masc_v3_v12_beats_v11_rate >= 0.5
        and env.masc_v3_tsc_v12_beats_tsc_v11_rate >= 0.5)
