"""W69 trivial passthrough test.

When ``W69Params.build_trivial()`` is used the W69 envelope must
preserve the supplied W68 outer CID byte-for-byte and emit zero
substrate witnesses (substrate is unused)."""

from __future__ import annotations


def test_w69_trivial_passthrough_byte_identical():
    from coordpy.w69_team import W69Params, W69Team
    p = W69Params.build_trivial()
    team = W69Team(params=p)
    env = team.run_team_turn(
        w68_outer_cid="UNIQUE_W68_CID_FOR_PASSTHROUGH_TEST",
        text="trivial")
    assert (
        env.w68_outer_cid
        == "UNIQUE_W68_CID_FOR_PASSTHROUGH_TEST")
    assert env.substrate_v14_witness_cid == ""
    assert env.kv_bridge_v14_witness_cid == ""
    assert env.handoff_coordinator_witness_cid == ""
    assert not env.substrate_v14_used
    assert not env.fourteen_way_used
