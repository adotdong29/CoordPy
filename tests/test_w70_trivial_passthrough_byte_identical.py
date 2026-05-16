"""W70 trivial passthrough test.

When ``W70Params.build_trivial()`` is used the W70 envelope must
preserve the supplied W69 outer CID byte-for-byte and emit zero
substrate witnesses (substrate is unused)."""

from __future__ import annotations


def test_w70_trivial_passthrough_byte_identical():
    from coordpy.w70_team import W70Params, W70Team
    p = W70Params.build_trivial()
    team = W70Team(params=p)
    env = team.run_team_turn(
        w69_outer_cid="UNIQUE_W69_CID_FOR_PASSTHROUGH_TEST",
        text="trivial")
    assert (
        env.w69_outer_cid
        == "UNIQUE_W69_CID_FOR_PASSTHROUGH_TEST")
    assert env.substrate_v15_witness_cid == ""
    assert env.kv_bridge_v15_witness_cid == ""
    assert env.handoff_coordinator_v2_witness_cid == ""
    assert env.handoff_envelope_v2_chain_cid == ""
    assert not env.substrate_v15_used
    assert not env.fifteen_way_used
