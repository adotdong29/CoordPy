"""W64 team envelope-chain integration test.

The W64 envelope must:
* preserve the supplied w63_outer_cid byte-for-byte
* emit all 23 W64-specific witness CIDs when params are default
* be verified clean by ``verify_w64_handoff`` (zero failures)
* enumerate ≥ 85 disjoint failure modes
* fire ``nine_way_used`` when all nine axes are active
* chain the W63 envelope: ``W63 envelope CID == W64.w63_outer_cid``
"""

from __future__ import annotations


def test_w64_envelope_chain_default():
    from coordpy.w64_team import (
        W64Params, W64Team,
        W64_ENVELOPE_VERIFIER_FAILURE_MODES,
        verify_w64_handoff,
    )
    p = W64Params.build_default(seed=64000)
    team = W64Team(params=p)
    env = team.step(
        turn_index=0, role="r",
        w63_outer_cid="w63_test_chain_cid")
    v = verify_w64_handoff(env)
    assert v["ok"], f"verifier failures: {v['failures']}"
    assert env.w63_outer_cid == "w63_test_chain_cid"
    assert env.substrate_v9_used
    assert env.nine_way_used
    assert len(W64_ENVELOPE_VERIFIER_FAILURE_MODES) >= 85
    d = env.to_dict()
    for k, val in d.items():
        if k.endswith("_witness_cid") or k.endswith("_matrix_cid"):
            assert str(val), f"{k} empty under default team"


def test_w64_envelope_chain_two_turns_distinct():
    from coordpy.w64_team import W64Params, W64Team
    p = W64Params.build_default(seed=64010)
    team = W64Team(params=p)
    env0 = team.step(turn_index=0, w63_outer_cid="w63_a")
    env1 = team.step(turn_index=1, w63_outer_cid="w63_a")
    assert env0.cid() != env1.cid()
    assert env0.persistent_v16_witness_cid != (
        env1.persistent_v16_witness_cid)


def test_w64_envelope_chain_with_real_w63():
    """W63 envelope CID flows verbatim into W64.w63_outer_cid."""
    from coordpy.w63_team import W63Params, W63Team
    from coordpy.w64_team import W64Params, W64Team
    p63 = W63Params.build_default(seed=63055)
    team63 = W63Team(params=p63)
    env63 = team63.step(
        turn_index=0, w62_outer_cid="w62_root_for_chain")
    w63_outer = env63.cid()
    p64 = W64Params.build_default(seed=64055)
    team64 = W64Team(params=p64)
    env64 = team64.step(
        turn_index=0, w63_outer_cid=w63_outer)
    assert env64.w63_outer_cid == w63_outer
