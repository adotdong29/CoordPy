"""W63 trivial-passthrough byte-identity test.

When ``W63Params.build_trivial()`` is used, the W63 envelope's
``w62_outer_cid`` must carry the supplied W62 outer CID byte-for-
byte, and the verifier must report exactly the expected missing-
witness failures.
"""

from __future__ import annotations


def test_w63_trivial_passthrough_w62_cid_preserved():
    from coordpy.w63_team import (
        W63Params, W63Team, verify_w63_handoff,
    )
    p = W63Params.build_trivial()
    team = W63Team(params=p)
    sample_w62_cid = (
        "w62_passthrough_byte_identical_test_2026_05_15")
    env = team.step(
        turn_index=0, w62_outer_cid=sample_w62_cid)
    assert env.w62_outer_cid == sample_w62_cid
    assert not env.substrate_v8_used
    assert not env.eight_way_used
    v = verify_w63_handoff(env)
    assert "missing_w62_outer_cid" not in v["failures"]
    assert "missing_substrate_v8_witness" in v["failures"]


def test_w63_trivial_envelope_idempotent():
    from coordpy.w63_team import W63Params, W63Team
    p = W63Params.build_trivial()
    team = W63Team(params=p)
    env1 = team.step(turn_index=0, w62_outer_cid="constant_cid")
    env2 = team.step(turn_index=0, w62_outer_cid="constant_cid")
    assert env1.cid() == env2.cid()
