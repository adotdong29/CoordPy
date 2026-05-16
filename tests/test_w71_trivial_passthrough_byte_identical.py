"""W71 trivial passthrough test.

When ``W71Params.build_trivial()`` is used the W71 envelope must
preserve the supplied W70 outer CID byte-for-byte and emit zero
substrate witnesses (substrate is unused)."""

from __future__ import annotations


def test_w71_trivial_passthrough_byte_identical():
    from coordpy.w71_team import W71Params, W71Team
    p = W71Params.build_trivial()
    team = W71Team(params=p)
    env = team.run_team_turn(
        w70_outer_cid="UNIQUE_W70_CID_FOR_PASSTHROUGH_TEST",
        text="trivial")
    assert (
        env.w70_outer_cid
        == "UNIQUE_W70_CID_FOR_PASSTHROUGH_TEST")
    assert env.substrate_v16_witness_cid == ""
    assert env.kv_bridge_v16_witness_cid == ""
    assert env.handoff_coordinator_v3_witness_cid == ""
    assert env.handoff_envelope_v3_chain_cid == ""
    assert not env.substrate_v16_used
    assert not env.sixteen_way_used
