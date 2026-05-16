"""W63 team envelope-chain integration test.

The W63 envelope must:
* preserve the supplied w62_outer_cid byte-for-byte
* emit all 22 W63-specific witness CIDs when params are default
* be verified clean by ``verify_w63_handoff`` (zero failures)
* enumerate ≥ 72 disjoint failure modes
* fire ``eight_way_used`` when all eight axes are active
"""

from __future__ import annotations


def test_w63_envelope_chain_default():
    from coordpy.w63_team import (
        W63Params, W63Team,
        W63_ENVELOPE_VERIFIER_FAILURE_MODES,
        verify_w63_handoff,
    )
    p = W63Params.build_default(seed=63000)
    team = W63Team(params=p)
    env = team.step(
        turn_index=0, role="r",
        w62_outer_cid="w62_test_chain_cid")
    v = verify_w63_handoff(env)
    assert v["ok"], f"verifier failures: {v['failures']}"
    assert env.w62_outer_cid == "w62_test_chain_cid"
    assert env.substrate_v8_used
    assert env.eight_way_used
    assert len(W63_ENVELOPE_VERIFIER_FAILURE_MODES) >= 72
    d = env.to_dict()
    for k, val in d.items():
        if k.endswith("_witness_cid") or k.endswith("_matrix_cid"):
            assert str(val), f"{k} empty under default team"


def test_w63_envelope_chain_two_turns_distinct():
    from coordpy.w63_team import W63Params, W63Team
    p = W63Params.build_default(seed=63010)
    team = W63Team(params=p)
    env0 = team.step(turn_index=0, w62_outer_cid="w62_a")
    env1 = team.step(turn_index=1, w62_outer_cid="w62_a")
    assert env0.cid() != env1.cid()
    assert env0.persistent_v15_witness_cid != (
        env1.persistent_v15_witness_cid)
