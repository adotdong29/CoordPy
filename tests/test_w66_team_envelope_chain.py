"""W66 team envelope-chain integration test.

The W66 envelope must:
* preserve the supplied w65_outer_cid byte-for-byte
* emit all W66-specific witness CIDs when params are default
* be verified clean by ``verify_w66_handoff`` (zero failures)
* enumerate ≥ 120 disjoint failure modes
* fire ``eleven_way_used`` when all eleven axes are active
* chain the W65 envelope: ``W65 envelope CID == W66.w65_outer_cid``
"""

from __future__ import annotations


def test_w66_envelope_chain_default():
    from coordpy.w66_team import (
        W66Params, W66Team,
        W66_ENVELOPE_VERIFIER_FAILURE_MODES,
        verify_w66_handoff,
    )
    p = W66Params.build_default(seed=66000)
    team = W66Team(params=p)
    env = team.step(
        turn_index=0, role="planner",
        w65_outer_cid="w65_test_chain_cid")
    v = verify_w66_handoff(env)
    assert v["ok"], f"verifier failures: {v['failures']}"
    assert env.w65_outer_cid == "w65_test_chain_cid"
    assert env.substrate_v11_used
    assert env.eleven_way_used
    assert len(W66_ENVELOPE_VERIFIER_FAILURE_MODES) >= 120
    assert env.masc_v2_v11_beats_v10_rate >= 0.5
    d = env.to_dict()
    for k, val in d.items():
        if (k.endswith("_witness_cid")
                or k.endswith("_matrix_cid")):
            assert str(val), f"{k} empty under default team"


def test_w66_envelope_chain_two_turns_distinct():
    from coordpy.w66_team import W66Params, W66Team
    p = W66Params.build_default(seed=66010)
    team = W66Team(params=p)
    env0 = team.step(turn_index=0, w65_outer_cid="w65_a")
    env1 = team.step(turn_index=1, w65_outer_cid="w65_a")
    assert env0.cid() != env1.cid()
    assert env0.persistent_v18_witness_cid != (
        env1.persistent_v18_witness_cid)


def test_w66_envelope_chain_with_real_w65():
    """W65 envelope CID flows verbatim into W66.w65_outer_cid."""
    from coordpy.w65_team import W65Params, W65Team
    from coordpy.w66_team import W66Params, W66Team
    p65 = W65Params.build_default(seed=65055)
    team65 = W65Team(params=p65)
    env65 = team65.step(
        turn_index=0, w64_outer_cid="w64_root_for_chain")
    w65_outer = env65.cid()
    p66 = W66Params.build_default(seed=66055)
    team66 = W66Team(params=p66)
    env66 = team66.step(
        turn_index=0, w65_outer_cid=w65_outer)
    assert env66.w65_outer_cid == w65_outer
