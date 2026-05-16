"""W66 trivial-passthrough byte-identity test.

When ``W66Params.build_trivial()`` is used, the W66 envelope's
``w65_outer_cid`` must carry the supplied W65 outer CID byte-for-
byte, and the verifier must report exactly the expected missing-
witness failures.
"""

from __future__ import annotations


def test_w66_trivial_passthrough_w65_cid_preserved():
    from coordpy.w66_team import (
        W66Params, W66Team, verify_w66_handoff,
    )
    p = W66Params.build_trivial()
    team = W66Team(params=p)
    sample_w65_cid = (
        "w65_passthrough_byte_identical_test_2026_05_16")
    env = team.step(
        turn_index=0, w65_outer_cid=sample_w65_cid)
    assert env.w65_outer_cid == sample_w65_cid
    assert not env.substrate_v11_used
    assert not env.eleven_way_used
    v = verify_w66_handoff(env)
    assert "missing_w65_outer_cid" not in v["failures"]
    assert "missing_substrate_v11_witness" in v["failures"]


def test_w66_trivial_envelope_idempotent():
    from coordpy.w66_team import W66Params, W66Team
    p = W66Params.build_trivial()
    team = W66Team(params=p)
    env1 = team.step(
        turn_index=0, w65_outer_cid="constant_cid")
    env2 = team.step(
        turn_index=0, w65_outer_cid="constant_cid")
    assert env1.cid() == env2.cid()
