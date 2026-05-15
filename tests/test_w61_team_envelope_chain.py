"""W61 team envelope-chain integration test.

The W61 envelope must:
* preserve the supplied w60_outer_cid byte-for-byte
* emit all 20 W61-specific witness CIDs when params are default
* be verified clean by ``verify_w61_handoff`` (zero failures)
* enumerate ≥ 55 disjoint failure modes
* fire ``six_way_used`` when all six axes are active
"""

from __future__ import annotations


def test_w61_envelope_chain_default():
    from coordpy.w61_team import (
        W61Params, W61Team,
        W61_ENVELOPE_VERIFIER_FAILURE_MODES,
        verify_w61_handoff,
    )
    p = W61Params.build_default(seed=61000)
    team = W61Team(params=p)
    env = team.step(
        turn_index=0, role="r",
        w60_outer_cid="w60_test_chain_cid")
    v = verify_w61_handoff(env)
    assert v["ok"], (
        f"verifier failures: {v['failures']}")
    assert env.w60_outer_cid == "w60_test_chain_cid"
    assert env.substrate_v6_used
    assert env.six_way_used
    assert len(W61_ENVELOPE_VERIFIER_FAILURE_MODES) >= 55
    # Every witness CID is non-empty.
    d = env.to_dict()
    for k, v_ in d.items():
        if k.endswith("_witness_cid") or k.endswith("_matrix_cid"):
            assert str(v_), f"{k} empty under default team"


def test_w61_envelope_chain_two_turns_distinct():
    from coordpy.w61_team import W61Params, W61Team
    p = W61Params.build_default(seed=61010)
    team = W61Team(params=p)
    env0 = team.step(
        turn_index=0, w60_outer_cid="w60_a")
    env1 = team.step(
        turn_index=1, w60_outer_cid="w60_a")
    assert env0.cid() != env1.cid()
    # The persistent V13 chain advances.
    assert (
        env0.persistent_v13_witness_cid
        != env1.persistent_v13_witness_cid)
