"""W67 trivial passthrough test.

When ``W67Params.build_trivial()`` is used the W67 envelope must
preserve the supplied W66 outer CID byte-for-byte and emit zero
substrate witnesses (substrate is unused)."""

from __future__ import annotations


def test_w67_trivial_passthrough_byte_identical():
    from coordpy.w67_team import W67Params, W67Team
    p = W67Params.build_trivial()
    team = W67Team(params=p)
    env = team.step(
        turn_index=0, role="planner",
        w66_outer_cid="UNIQUE_W66_CID_FOR_PASSTHROUGH_TEST")
    assert (
        env.w66_outer_cid
        == "UNIQUE_W66_CID_FOR_PASSTHROUGH_TEST")
    assert env.substrate_v12_witness_cid == ""
    assert env.kv_bridge_v12_witness_cid == ""
    assert not env.substrate_v12_used
    assert not env.twelve_way_used
