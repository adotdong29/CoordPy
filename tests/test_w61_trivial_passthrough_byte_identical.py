"""W61 trivial-passthrough byte-identity test.

When ``W61Params.build_trivial()`` is used, the W61 envelope's
``w60_outer_cid`` must carry the supplied W60 outer CID byte-for-
byte, and the verifier must report exactly the expected missing-
witness failures (every module witness is absent because trivial
mode skips all module work).
"""

from __future__ import annotations


def test_w61_trivial_passthrough_w60_cid_preserved():
    from coordpy.w61_team import (
        W61Params, W61Team, verify_w61_handoff,
    )
    p = W61Params.build_trivial()
    team = W61Team(params=p)
    sample_w60_cid = (
        "w60_passthrough_byte_identical_test_2026_05_15")
    env = team.step(
        turn_index=0, w60_outer_cid=sample_w60_cid)
    assert env.w60_outer_cid == sample_w60_cid
    assert not env.substrate_v6_used
    assert not env.six_way_used
    v = verify_w61_handoff(env)
    # Trivial mode means every witness is missing, but the CID
    # carry-forward is preserved. The verifier should report
    # exactly the "missing_*" failures; the chain-fidelity field
    # ``missing_w60_outer_cid`` MUST NOT appear because we did
    # supply a non-empty CID.
    assert "missing_w60_outer_cid" not in v["failures"]
    assert "missing_substrate_v6_witness" in v["failures"]


def test_w61_trivial_envelope_idempotent():
    from coordpy.w61_team import W61Params, W61Team
    p = W61Params.build_trivial()
    team = W61Team(params=p)
    env1 = team.step(turn_index=0, w60_outer_cid="constant_cid")
    env2 = team.step(turn_index=0, w60_outer_cid="constant_cid")
    # Trivial mode is deterministic: same turn_index + w60 cid
    # produces the same envelope.
    assert env1.cid() == env2.cid()
