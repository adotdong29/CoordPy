"""W84 / P0 #29 — Cross-process distributed substrate tests."""

from __future__ import annotations

import pytest

from coordpy.cross_process_distributed_substrate_v1 import (
    W84_CROSS_PROCESS_DISTRIBUTED_V1_SCHEMA_VERSION,
    W84_MTLS_HEADER,
    TrustRootV1,
    build_trust_root_v1,
    run_cross_process_distributed_bench_v1,
)


def test_w84_trust_root_is_content_addressed():
    a = build_trust_root_v1(principal_id="alpha", seed=1)
    b = build_trust_root_v1(principal_id="alpha", seed=1)
    assert a.cid() == b.cid()
    assert len(a.cid()) == 64


def test_w84_trust_root_principal_differentiation():
    a = build_trust_root_v1(principal_id="alpha", seed=1)
    b = build_trust_root_v1(principal_id="beta", seed=1)
    # Different principals → different CIDs even with same seed.
    assert a.cid() != b.cid()


def test_w84_cross_process_bench_mTLS_unauth_refused():
    rep = run_cross_process_distributed_bench_v1(
        n_envelopes=2, partition_window_seconds=0.3)
    assert rep.mtls_unauthenticated_refused is True


def test_w84_cross_process_bench_bad_signature_refused():
    rep = run_cross_process_distributed_bench_v1(
        n_envelopes=2, partition_window_seconds=0.3)
    assert rep.mtls_bad_signature_refused is True


def test_w84_cross_process_bench_post_roots_match():
    rep = run_cross_process_distributed_bench_v1(
        n_envelopes=4, partition_window_seconds=0.3)
    assert rep.cross_process_post_root_match is True
    assert rep.sender_root_cid == rep.receiver_root_cid


def test_w84_cross_process_bench_partition_drops_traffic():
    rep = run_cross_process_distributed_bench_v1(
        n_envelopes=2, partition_window_seconds=0.3)
    assert rep.partition_drops_all_traffic is True


def test_w84_cross_process_bench_partition_heals():
    rep = run_cross_process_distributed_bench_v1(
        n_envelopes=2, partition_window_seconds=0.3)
    assert rep.partition_heals_and_recovers is True


def test_w84_cross_process_bench_skew_within_tolerance():
    rep = run_cross_process_distributed_bench_v1(
        n_envelopes=2, partition_window_seconds=0.3,
        clock_skew_alpha_s=2.0, clock_skew_beta_s=-2.0)
    assert rep.skew_injection_within_tolerance is True


def test_w84_cross_process_bench_idempotent_apply_holds():
    rep = run_cross_process_distributed_bench_v1(
        n_envelopes=2, partition_window_seconds=0.3)
    assert rep.idempotent_apply_holds is True


def test_w84_cross_process_bench_report_is_content_addressed():
    rep_a = run_cross_process_distributed_bench_v1(
        n_envelopes=2, partition_window_seconds=0.3)
    # Re-running with the same config produces a stable report
    # shape; the post-root-cid is deterministic on the envelope
    # CIDs.
    rep_b = run_cross_process_distributed_bench_v1(
        n_envelopes=2, partition_window_seconds=0.3)
    assert rep_a.sender_root_cid == rep_b.sender_root_cid
