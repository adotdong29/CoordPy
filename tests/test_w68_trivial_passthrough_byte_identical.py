"""W68 trivial passthrough test.

When ``W68Params.build_trivial()`` is used the W68 envelope must
preserve the supplied W67 outer CID byte-for-byte and emit zero
substrate witnesses (substrate is unused)."""

from __future__ import annotations


def test_w68_trivial_passthrough_byte_identical():
    from coordpy.w68_team import W68Params, W68Team
    p = W68Params.build_trivial()
    team = W68Team(params=p)
    env = team.run_team_turn(
        w67_outer_cid="UNIQUE_W67_CID_FOR_PASSTHROUGH_TEST",
        text="trivial")
    assert (
        env.w67_outer_cid
        == "UNIQUE_W67_CID_FOR_PASSTHROUGH_TEST")
    assert env.substrate_v13_witness_cid == ""
    assert env.kv_bridge_v13_witness_cid == ""
    assert not env.substrate_v13_used
    assert not env.thirteen_way_used
