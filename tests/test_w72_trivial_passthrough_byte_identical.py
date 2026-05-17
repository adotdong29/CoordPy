"""W72 trivial passthrough test.

When ``W72Params.build_trivial()`` is used the W72 envelope must
preserve the supplied W71 outer CID byte-for-byte and emit zero
substrate witnesses (substrate is unused)."""

from __future__ import annotations


def test_w72_trivial_passthrough_byte_identical():
    from coordpy.w72_team import W72Params, W72Team
    p = W72Params.build_trivial()
    team = W72Team(params=p)
    env = team.run_team_turn(
        w71_outer_cid="UNIQUE_W71_CID_FOR_PASSTHROUGH_TEST",
        text="trivial")
    assert (
        env.w71_outer_cid
        == "UNIQUE_W71_CID_FOR_PASSTHROUGH_TEST")
    assert env.substrate_v17_witness_cid == ""
    assert env.kv_bridge_v17_witness_cid == ""
    assert env.handoff_coordinator_v4_witness_cid == ""
    assert env.handoff_envelope_v4_chain_cid == ""
    assert not env.substrate_v17_used
    assert not env.seventeen_way_used
