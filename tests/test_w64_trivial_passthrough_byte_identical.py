"""W64 trivial-passthrough byte-identity test.

When ``W64Params.build_trivial()`` is used, the W64 envelope's
``w63_outer_cid`` must carry the supplied W63 outer CID byte-for-
byte, and the verifier must report exactly the expected missing-
witness failures.
"""

from __future__ import annotations


def test_w64_trivial_passthrough_w63_cid_preserved():
    from coordpy.w64_team import (
        W64Params, W64Team, verify_w64_handoff,
    )
    p = W64Params.build_trivial()
    team = W64Team(params=p)
    sample_w63_cid = (
        "w63_passthrough_byte_identical_test_2026_05_15")
    env = team.step(
        turn_index=0, w63_outer_cid=sample_w63_cid)
    assert env.w63_outer_cid == sample_w63_cid
    assert not env.substrate_v9_used
    assert not env.nine_way_used
    v = verify_w64_handoff(env)
    assert "missing_w63_outer_cid" not in v["failures"]
    assert "missing_substrate_v9_witness" in v["failures"]


def test_w64_trivial_envelope_idempotent():
    from coordpy.w64_team import W64Params, W64Team
    p = W64Params.build_trivial()
    team = W64Team(params=p)
    env1 = team.step(turn_index=0, w63_outer_cid="constant_cid")
    env2 = team.step(turn_index=0, w63_outer_cid="constant_cid")
    assert env1.cid() == env2.cid()
