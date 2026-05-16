"""W65 team envelope-chain integration test.

The W65 envelope must:
* preserve the supplied w64_outer_cid byte-for-byte
* emit all W65-specific witness CIDs when params are default
* be verified clean by ``verify_w65_handoff`` (zero failures)
* enumerate ≥ 100 disjoint failure modes
* fire ``ten_way_used`` when all ten axes are active
* chain the W64 envelope: ``W64 envelope CID == W65.w64_outer_cid``
"""

from __future__ import annotations


def test_w65_envelope_chain_default():
    from coordpy.w65_team import (
        W65Params, W65Team,
        W65_ENVELOPE_VERIFIER_FAILURE_MODES,
        verify_w65_handoff,
    )
    p = W65Params.build_default(seed=65000)
    team = W65Team(params=p)
    env = team.step(
        turn_index=0, role="planner",
        w64_outer_cid="w64_test_chain_cid")
    v = verify_w65_handoff(env)
    assert v["ok"], f"verifier failures: {v['failures']}"
    assert env.w64_outer_cid == "w64_test_chain_cid"
    assert env.substrate_v10_used
    assert env.ten_way_used
    assert len(W65_ENVELOPE_VERIFIER_FAILURE_MODES) >= 100
    assert env.masc_v10_success_rate >= 0.5
    d = env.to_dict()
    for k, val in d.items():
        if (k.endswith("_witness_cid")
                or k.endswith("_matrix_cid")):
            assert str(val), f"{k} empty under default team"


def test_w65_envelope_chain_two_turns_distinct():
    from coordpy.w65_team import W65Params, W65Team
    p = W65Params.build_default(seed=65010)
    team = W65Team(params=p)
    env0 = team.step(turn_index=0, w64_outer_cid="w64_a")
    env1 = team.step(turn_index=1, w64_outer_cid="w64_a")
    assert env0.cid() != env1.cid()
    assert env0.persistent_v17_witness_cid != (
        env1.persistent_v17_witness_cid)


def test_w65_envelope_chain_with_real_w64():
    """W64 envelope CID flows verbatim into W65.w64_outer_cid."""
    from coordpy.w64_team import W64Params, W64Team
    from coordpy.w65_team import W65Params, W65Team
    p64 = W64Params.build_default(seed=64055)
    team64 = W64Team(params=p64)
    env64 = team64.step(
        turn_index=0, w63_outer_cid="w63_root_for_chain")
    w64_outer = env64.cid()
    p65 = W65Params.build_default(seed=65055)
    team65 = W65Team(params=p65)
    env65 = team65.step(
        turn_index=0, w64_outer_cid=w64_outer)
    assert env65.w64_outer_cid == w64_outer
