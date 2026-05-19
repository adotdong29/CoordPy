"""W84 / P0 #29 — Real cross-host distributed substrate V2 tests.

Tests:

- mTLS handshake required on every connection;
  unauthenticated peer rejected.
- Migration envelope CID is content-addressed and matches
  across hosts byte-identically.
- Partition test: simulate accept-packets=False on host B,
  verify sender fails, heal succeeds, ``PartitionEventV2``
  records pre/post root CIDs.
- Skew test: inject 3s clock skew on host B; envelope still
  verifies within the 5s window.
- Idempotency: 10 replays of the same envelope leave host B's
  apply-log snapshot CID byte-identical.
- Cross-host envelope_wire_cid equal between source and
  target.

These tests run the V2 bench against ``127.0.0.1`` and
``127.0.0.2`` on the loopback subnet (different IPs, real TCP,
real TLS handshake). Production V2 deploys to two distinct
hosts; the wire format / mTLS handshake / audit chain are
identical across topologies.
"""

from __future__ import annotations

import socket

import pytest

import coordpy.real_distributed_substrate_v2 as rds


def _can_bind(host: str) -> bool:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind((host, 0))
        s.close()
        return True
    except OSError:
        return False


_CAN_BIND_127_0_0_2 = _can_bind("127.0.0.2")


def test_w84_real_distributed_module_exports():
    for name in (
        "W84_REAL_DISTRIBUTED_V2_SCHEMA_VERSION",
        "MtlsCertificateAuthorityV2",
        "MtlsHttpsHostV2",
        "MigrationEnvelopeWireFormatV2",
        "build_envelope_v2",
        "IdempotentApplyLogV2",
        "PartitionSimulatorV2",
        "ClockSkewSimulatorV2",
        "PartitionEventV2",
        "emit_partition_event_v2",
        "RealDistributedBenchReportV2",
        "run_real_distributed_bench_v2",
    ):
        assert name in rds.__all__
        assert hasattr(rds, name)


def test_w84_certificate_authority_signs_host_certs():
    ca = rds.MtlsCertificateAuthorityV2()
    cert_pem, key_pem = ca.issue_host_cert(
        principal_name="host-test",
        host_ip="127.0.0.1")
    assert cert_pem.startswith(b"-----BEGIN CERTIFICATE-----")
    assert (
        key_pem.startswith(b"-----BEGIN RSA PRIVATE KEY-----")
        or key_pem.startswith(b"-----BEGIN PRIVATE KEY-----"))
    # Two issued certs must differ (different keys).
    cert_pem2, key_pem2 = ca.issue_host_cert(
        principal_name="host-test",
        host_ip="127.0.0.1")
    assert cert_pem != cert_pem2 or key_pem != key_pem2


def test_w84_envelope_cid_is_content_addressed():
    env_a = rds.build_envelope_v2(
        envelope_id="env-1",
        source_principal="A", target_principal="B",
        source_ip="10.0.0.1", target_ip="10.0.0.2",
        payload={"x": 1, "y": [1, 2]},
        source_now_ns=12345)
    env_b = rds.build_envelope_v2(
        envelope_id="env-1",
        source_principal="A", target_principal="B",
        source_ip="10.0.0.1", target_ip="10.0.0.2",
        payload={"x": 1, "y": [1, 2]},
        source_now_ns=12345)
    assert env_a.envelope_wire_cid() == (
        env_b.envelope_wire_cid())
    env_c = rds.build_envelope_v2(
        envelope_id="env-1",
        source_principal="A", target_principal="B",
        source_ip="10.0.0.1", target_ip="10.0.0.2",
        payload={"x": 2, "y": [1, 2]},
        source_now_ns=12345)
    assert env_a.envelope_wire_cid() != (
        env_c.envelope_wire_cid())


def test_w84_idempotent_apply_log():
    env = rds.build_envelope_v2(
        envelope_id="e1",
        source_principal="A", target_principal="B",
        source_ip="10.0.0.1", target_ip="10.0.0.2",
        payload={"k": "v"},
        source_now_ns=1)
    log = rds.IdempotentApplyLogV2()
    # First apply: new.
    assert log.apply(envelope=env) is True
    snap1 = log.snapshot_cid()
    # Re-apply: idempotent.
    assert log.apply(envelope=env) is False
    snap2 = log.snapshot_cid()
    # Snapshot CID byte-identical across replays.
    assert snap1 == snap2
    # Apply more replays.
    for _ in range(8):
        assert log.apply(envelope=env) is False
    assert log.snapshot_cid() == snap1
    assert int(log.n_applied) == 1


def test_w84_partition_event_records_root_cids():
    sim = rds.PartitionSimulatorV2()
    sim.begin_partition(pre_root_cid="abc123")
    assert not sim.accept_packets
    sim.heal_partition(post_root_cid="def456")
    assert sim.accept_packets
    ev = rds.emit_partition_event_v2(simulator=sim)
    assert ev.pre_root_cid == "abc123"
    assert ev.post_root_cid == "def456"
    assert ev.duration_seconds >= 0.0
    assert isinstance(ev.cid(), str)
    assert len(ev.cid()) == 64


def test_w84_clock_skew_simulator_offsets_now():
    sim = rds.ClockSkewSimulatorV2(skew_seconds=2.0)
    base_ns = 1_000_000_000_000
    now = sim.now_ns(base_ns=base_ns)
    assert now == int(base_ns + 2.0 * 1e9)
    sim2 = rds.ClockSkewSimulatorV2(skew_seconds=-3.0)
    assert sim2.now_ns(base_ns=base_ns) == int(
        base_ns - 3.0 * 1e9)


def test_w84_verify_envelope_skew_within_window():
    env = rds.build_envelope_v2(
        envelope_id="e1",
        source_principal="A", target_principal="B",
        source_ip="10.0.0.1", target_ip="10.0.0.2",
        payload={},
        source_now_ns=10_000_000_000_000)
    # Target is 3s ahead → 3s skew, within 5s window.
    within, skew_s = rds.verify_envelope_skew_v2(
        envelope=env,
        target_now_ns=10_000_000_000_000 + int(3e9),
        max_acceptable_skew_seconds=5.0)
    assert within is True
    assert 2.99 < skew_s < 3.01
    # Outside window.
    within2, _ = rds.verify_envelope_skew_v2(
        envelope=env,
        target_now_ns=10_000_000_000_000 + int(7e9),
        max_acceptable_skew_seconds=5.0)
    assert within2 is False


@pytest.mark.skipif(
    not _CAN_BIND_127_0_0_2,
    reason="127.0.0.2 not bindable on this host")
def test_w84_real_distributed_bench_end_to_end():
    """End-to-end V2 bench: every load-bearing bar must pass."""
    r = rds.run_real_distributed_bench_v2(
        clock_skew_seconds=2.0,
        n_idempotent_replays=10,
        partition_window_seconds=0.3)
    # mTLS bar.
    assert r.mtls_authenticated_handshake_ok
    assert r.unauthenticated_peer_rejected
    # Cross-host CID equality.
    assert r.cross_host_cid_equal
    assert r.source_envelope_wire_cid == (
        r.target_envelope_wire_cid)
    assert len(r.source_envelope_wire_cid) == 64
    # Partition.
    assert r.sender_failed_during_partition
    assert r.heal_succeeded
    assert len(r.partition_event_cid) == 64
    # Skew.
    assert r.skew_within_window
    assert abs(float(r.measured_skew_seconds)) <= (
        float(r.skew_window_seconds))
    # Idempotency.
    assert r.n_idempotent_replays == 10
    assert r.log_snapshot_cid_byte_identical
    assert r.log_snapshot_cid_before_replays == (
        r.log_snapshot_cid_after_replays)
    # Real TCP packets exchanged.
    assert r.n_real_packets_exchanged >= 5
    # Topology: distinct IPs.
    assert r.host_a_ip != r.host_b_ip
    # Determinism: report cid is stable.
    assert isinstance(r.cid(), str)
    assert len(r.cid()) == 64


@pytest.mark.skipif(
    not _CAN_BIND_127_0_0_2,
    reason="127.0.0.2 not bindable on this host")
def test_w84_real_distributed_unauthenticated_client_rejected():
    """An unauthenticated TLS client must be rejected during
    the handshake (verify_mode=CERT_REQUIRED on the server)."""
    import ssl
    from urllib import error as _urlerror
    from urllib import request as _urlrequest
    ca = rds.MtlsCertificateAuthorityV2()
    a_cert, a_key = ca.issue_host_cert(
        principal_name="host-A", host_ip="127.0.0.1")
    host_a = rds.MtlsHttpsHostV2(
        principal_name="host-A",
        host_ip="127.0.0.1",
        cert_pem=a_cert, key_pem=a_key,
        ca_cert_pem=ca.ca_cert_pem())
    host_a.start()
    try:
        # Client with NO cert: should fail handshake.
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.load_verify_locations(
            cadata=ca.ca_cert_pem().decode("utf-8"))
        ctx.check_hostname = False
        rejected = False
        try:
            with _urlrequest.urlopen(
                    f"{host_a.url}/healthz",
                    timeout=1.0, context=ctx) as r:
                _ = r.read()
        except (
                ssl.SSLError, ssl.SSLCertVerificationError,
                _urlerror.URLError, OSError):
            rejected = True
        assert rejected
    finally:
        host_a.stop()
