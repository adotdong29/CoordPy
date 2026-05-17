"""W74 trivial passthrough test.

When ``W74Params.build_trivial()`` is used the W74 envelope must
preserve the supplied W73 outer CID byte-for-byte and emit zero
substrate witnesses (substrate is unused)."""

from __future__ import annotations


def test_w74_trivial_passthrough_byte_identical():
    from coordpy.w74_team import W74Params, W74Team
    p = W74Params.build_trivial()
    team = W74Team(params=p)
    env = team.run_team_turn(
        w73_outer_cid="UNIQUE_W73_CID_FOR_PASSTHROUGH_TEST",
        text="trivial")
    assert (
        env.w73_outer_cid
        == "UNIQUE_W73_CID_FOR_PASSTHROUGH_TEST")
    assert env.substrate_v19_witness_cid == ""
    assert env.kv_bridge_v19_witness_cid == ""
    assert env.handoff_coordinator_v6_witness_cid == ""
    assert env.handoff_envelope_v6_chain_cid == ""
    assert not env.substrate_v19_used
    assert not env.nineteen_way_used
