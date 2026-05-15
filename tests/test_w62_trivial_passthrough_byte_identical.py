"""W62 trivial-passthrough byte-identity test.

When ``W62Params.build_trivial()`` is used, the W62 envelope's
``w61_outer_cid`` must carry the supplied W61 outer CID byte-for-
byte, and the verifier must report exactly the expected missing-
witness failures.
"""

from __future__ import annotations


def test_w62_trivial_passthrough_w61_cid_preserved():
    from coordpy.w62_team import (
        W62Params, W62Team, verify_w62_handoff,
    )
    p = W62Params.build_trivial()
    team = W62Team(params=p)
    sample_w61_cid = (
        "w61_passthrough_byte_identical_test_2026_05_15")
    env = team.step(
        turn_index=0, w61_outer_cid=sample_w61_cid)
    assert env.w61_outer_cid == sample_w61_cid
    assert not env.substrate_v7_used
    assert not env.seven_way_used
    v = verify_w62_handoff(env)
    assert "missing_w61_outer_cid" not in v["failures"]
    assert "missing_substrate_v7_witness" in v["failures"]


def test_w62_trivial_envelope_idempotent():
    from coordpy.w62_team import W62Params, W62Team
    p = W62Params.build_trivial()
    team = W62Team(params=p)
    env1 = team.step(turn_index=0, w61_outer_cid="constant_cid")
    env2 = team.step(turn_index=0, w61_outer_cid="constant_cid")
    assert env1.cid() == env2.cid()
