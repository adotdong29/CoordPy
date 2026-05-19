"""W83 — distributed gateway coordination tests."""

from __future__ import annotations


def test_w83_dist_gateway_two_gateways_agree_over_http():
    from coordpy.distributed_gateway_coordination_v1 import (
        run_distributed_envelope_over_http_v1,
    )
    rep = run_distributed_envelope_over_http_v1()
    assert int(rep.n_events_sent) == 2
    assert bool(rep.merkle_roots_match)
    assert all(int(c) == 200 for c in rep.http_status_codes)
    assert float(rep.transport_round_trip_seconds) > 0.0
    assert len(rep.cid()) == 64


def test_w83_dist_gateway_witness_emitted():
    from coordpy.distributed_gateway_coordination_v1 import (
        emit_distributed_gateway_coordination_witness_v1,
        run_distributed_envelope_over_http_v1,
    )
    rep = run_distributed_envelope_over_http_v1()
    w = emit_distributed_gateway_coordination_witness_v1(
        result=rep)
    assert len(w.cid()) == 64
    assert bool(w.merkle_roots_match) == bool(
        rep.merkle_roots_match)


def test_w83_dist_gateway_run_is_repeatable():
    from coordpy.distributed_gateway_coordination_v1 import (
        run_distributed_envelope_over_http_v1,
    )
    a = run_distributed_envelope_over_http_v1()
    b = run_distributed_envelope_over_http_v1()
    # Both must agree internally.
    assert bool(a.merkle_roots_match)
    assert bool(b.merkle_roots_match)
    # Sender root CIDs should match between runs (same prompt
    # + same default runtime params).
    assert a.sender_post_root_cid == b.sender_post_root_cid
