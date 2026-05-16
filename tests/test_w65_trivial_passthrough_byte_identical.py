"""W65 trivial-passthrough byte-identity test.

When ``W65Params.build_trivial()`` is used, the W65 envelope's
``w64_outer_cid`` must carry the supplied W64 outer CID byte-for-
byte, and the verifier must report exactly the expected missing-
witness failures.
"""

from __future__ import annotations


def test_w65_trivial_passthrough_w64_cid_preserved():
    from coordpy.w65_team import (
        W65Params, W65Team, verify_w65_handoff,
    )
    p = W65Params.build_trivial()
    team = W65Team(params=p)
    sample_w64_cid = (
        "w64_passthrough_byte_identical_test_2026_05_16")
    env = team.step(
        turn_index=0, w64_outer_cid=sample_w64_cid)
    assert env.w64_outer_cid == sample_w64_cid
    assert not env.substrate_v10_used
    assert not env.ten_way_used
    v = verify_w65_handoff(env)
    assert "missing_w64_outer_cid" not in v["failures"]
    assert "missing_substrate_v10_witness" in v["failures"]


def test_w65_trivial_envelope_idempotent():
    from coordpy.w65_team import W65Params, W65Team
    p = W65Params.build_trivial()
    team = W65Team(params=p)
    env1 = team.step(
        turn_index=0, w64_outer_cid="constant_cid")
    env2 = team.step(
        turn_index=0, w64_outer_cid="constant_cid")
    assert env1.cid() == env2.cid()
