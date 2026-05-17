"""W73 trivial passthrough test.

When ``W73Params.build_trivial()`` is used the W73 envelope must
preserve the supplied W72 outer CID byte-for-byte and emit zero
substrate witnesses (substrate is unused)."""

from __future__ import annotations


def test_w73_trivial_passthrough_byte_identical():
    from coordpy.w73_team import W73Params, W73Team
    p = W73Params.build_trivial()
    team = W73Team(params=p)
    env = team.run_team_turn(
        w72_outer_cid="UNIQUE_W72_CID_FOR_PASSTHROUGH_TEST",
        text="trivial")
    assert (
        env.w72_outer_cid
        == "UNIQUE_W72_CID_FOR_PASSTHROUGH_TEST")
    assert env.substrate_v18_witness_cid == ""
    assert env.kv_bridge_v18_witness_cid == ""
    assert env.handoff_coordinator_v5_witness_cid == ""
    assert env.handoff_envelope_v5_chain_cid == ""
    assert not env.substrate_v18_used
    assert not env.eighteen_way_used
