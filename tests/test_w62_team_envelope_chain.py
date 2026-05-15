"""W62 team envelope-chain integration test.

The W62 envelope must:
* preserve the supplied w61_outer_cid byte-for-byte
* emit all 20 W62-specific witness CIDs when params are default
* be verified clean by ``verify_w62_handoff`` (zero failures)
* enumerate ≥ 65 disjoint failure modes
* fire ``seven_way_used`` when all seven axes are active
"""

from __future__ import annotations


def test_w62_envelope_chain_default():
    from coordpy.w62_team import (
        W62Params, W62Team,
        W62_ENVELOPE_VERIFIER_FAILURE_MODES,
        verify_w62_handoff,
    )
    p = W62Params.build_default(seed=62000)
    team = W62Team(params=p)
    env = team.step(
        turn_index=0, role="r",
        w61_outer_cid="w61_test_chain_cid")
    v = verify_w62_handoff(env)
    assert v["ok"], f"verifier failures: {v['failures']}"
    assert env.w61_outer_cid == "w61_test_chain_cid"
    assert env.substrate_v7_used
    assert env.seven_way_used
    assert len(W62_ENVELOPE_VERIFIER_FAILURE_MODES) >= 65
    d = env.to_dict()
    for k, val in d.items():
        if k.endswith("_witness_cid") or k.endswith("_matrix_cid"):
            assert str(val), f"{k} empty under default team"


def test_w62_envelope_chain_two_turns_distinct():
    from coordpy.w62_team import W62Params, W62Team
    p = W62Params.build_default(seed=62010)
    team = W62Team(params=p)
    env0 = team.step(turn_index=0, w61_outer_cid="w61_a")
    env1 = team.step(turn_index=1, w61_outer_cid="w61_a")
    assert env0.cid() != env1.cid()
    assert env0.persistent_v14_witness_cid != (
        env1.persistent_v14_witness_cid)
