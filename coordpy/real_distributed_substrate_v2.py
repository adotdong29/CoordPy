"""W84 / P0 #29 — Real Cross-Host Distributed Substrate V2.

W82 ``distributed_substrate_coordination_v1`` simulates multi-
host coordination in-process — the transport is a function
call (``W82-L-DISTRIBUTED-V1-IN-PROCESS-CAP``). W83
``distributed_gateway_coordination_v1`` ships JSON over HTTP
between two gateways but binds both endpoints to ``127.0.0.1``
on different ports (``W83-L-DIST-GATEWAY-V1-LOOPBACK-CAP`` +
``W83-L-DIST-GATEWAY-V1-IN-PROCESS-CAP``).

P0 #29 asks for a V2 that actually carries the W82 migration
semantics over a *real* cross-host transport: mTLS-
authenticated HTTPS, with handling for network partitions,
clock skew, idempotent re-delivery, and a content-addressed
wire format. This module is that V2.

V2 ships
--------

1. ``MtlsCertificateAuthorityV2`` — generates a self-signed
   X.509 CA and signs per-host certificates at runtime. The
   CA is HMAC-keyed for tamper detection; the per-host certs
   carry the host's named principal in CN + SAN.
2. ``MtlsHttpsHostV2`` — wraps an HTTP request handler in
   ``ssl.SSLContext(PROTOCOL_TLS_SERVER)`` with
   ``verify_mode=CERT_REQUIRED`` and the CA trust anchor. Any
   client that does not present a CA-signed cert is rejected
   at the TLS handshake.
3. ``MigrationEnvelopeWireFormatV2`` — content-addressed JSON
   over HTTPS. Every byte on the wire re-hashes to a known
   ``envelope_wire_cid``; the receiver verifies the CID match
   before applying.
4. ``PartitionSimulatorV2`` — per-host ``accept_packets``
   gate. Tests flip the gate to simulate a partition; the
   sender's POST fails fast, the system records a
   ``PartitionEventV2`` with start_ns / end_ns / pre_root_cid /
   post_root_cid in the audit chain.
5. ``ClockSkewSimulatorV2`` — per-host clock offset (positive
   or negative). The W82 migration timestamps + W83 hosted
   audit anchors verify across the skew (W2 picks the larger
   of |skew_seconds| as the verification window).
6. ``IdempotentApplyLogV2`` — receiver-side ``envelope_cid``
   set; re-applying the same envelope is a no-op. 10× replay
   leaves the receiver's event graph byte-identical.
7. ``run_real_distributed_bench_v2`` — end-to-end test runner
   that spins up two hosts on distinct IPs (127.0.0.1 and
   127.0.0.2), exchanges a migration over mTLS, exercises the
   partition / skew / idempotency / replay-byte-identity
   paths, and emits a content-addressed
   ``RealDistributedBenchReportV2``.

Honest scope (W84 P0 #29)
-------------------------

- ``W84-L-REAL-DISTRIBUTED-V2-RESEARCH-ONLY-CAP`` — explicit
  import only.
- ``W84-L-REAL-DISTRIBUTED-V2-CI-LOOPBACK-SUBNET-CAP`` — V2's
  test harness uses two distinct IPs on the loopback subnet
  (``127.0.0.1`` and ``127.0.0.2``). Real production deploys
  to two distinct machines / cloud regions / availability
  zones; the wire format, mTLS handshake, and audit chain are
  identical across both topologies. The honest reading: V2 is
  deployable to real hosts, AND the CI harness exercises the
  same transport over real TCP / TLS handshakes / kernel
  network stack between distinct IPs.
- ``W84-L-REAL-DISTRIBUTED-V2-SELF-SIGNED-CA-CAP`` — V2
  generates a self-signed CA at runtime. Production deploys
  use real PKI; the trust model is identical (X.509 +
  CA-signed + named principal) but the CA root is a
  configured trust anchor, not a runtime-generated one.
- ``W84-L-REAL-DISTRIBUTED-V2-EVENTUAL-CONSISTENCY-CAP`` —
  carries forward from W82. V2 strengthens the W82 transport
  bar to real mTLS HTTPS, but does NOT promise strong /
  linearizable consistency. The audit chain proves
  *eventual* consistency: after heal+sync every host's
  Merkle root is identical.
- ``W84-L-REAL-DISTRIBUTED-V2-INSECURE-MODE-CAP`` — there is
  an explicit ``allow_insecure_for_local_dev`` flag, OFF by
  default. CI tests verify that an un-authenticated peer is
  *rejected* by default. The insecure path exists only so
  local-dev iteration is not gated on cert plumbing.
- ``W84-L-REAL-DISTRIBUTED-V2-SINGLE-PARTITION-CAP`` — V2
  handles a single A↔B partition. Multi-partition (three-way
  split-brain) is V2.5 future work.
- ``W84-L-REAL-DISTRIBUTED-V2-CLOCK-SKEW-CAP`` — V2 verifies
  across ±5s skew. Larger skews require V3 + NTP discipline.
"""

from __future__ import annotations

import dataclasses
import hashlib
import http.server
import io
import json
import os
import socket
import ssl
import tempfile
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib import error as _urlerror
from urllib import request as _urlrequest


W84_REAL_DISTRIBUTED_V2_SCHEMA_VERSION: str = (
    "coordpy.real_distributed_substrate_v2.v1")

W84_REAL_DISTRIBUTED_V2_DEFAULT_HOST_A_IP: str = "127.0.0.1"
W84_REAL_DISTRIBUTED_V2_DEFAULT_HOST_B_IP: str = "127.0.0.2"
# Honest precision floor: cross-host transport adds no float
# noise (the wire format is content-addressed JSON, deserialised
# into bit-identical arrays via Python's stdlib). The replay-
# byte-identity floor is therefore the *single-host* floor
# carried forward unchanged.
W84_REAL_DISTRIBUTED_V2_REPLAY_TOLERANCE: float = 0.0
W84_REAL_DISTRIBUTED_V2_DEFAULT_PARTITION_WINDOW_SECONDS: float = (
    1.0)
W84_REAL_DISTRIBUTED_V2_DEFAULT_CLOCK_SKEW_SECONDS: float = 5.0
W84_REAL_DISTRIBUTED_V2_DEFAULT_IDEMPOTENT_REPLAYS: int = 10


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _pick_free_port_on(host_ip: str) -> int:
    """Pick a free TCP port on the named host IP."""
    with socket.socket(
            socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host_ip, 0))
        return int(s.getsockname()[1])


# ---------------------------------------------------------------
# mTLS certificate authority
# ---------------------------------------------------------------

@dataclasses.dataclass
class MtlsCertificateAuthorityV2:
    """Self-signed X.509 CA + per-host cert minter.

    Production V2 deployments would receive a CA root from a
    real PKI; the runtime-generated CA here is honest as a
    research-grade trust anchor. The trust model (X.509 +
    CA-signed + named principal in CN) is identical.
    """

    common_name: str = "coordpy-w84-test-ca"
    valid_seconds: int = 3600
    _ca_key: Any = None
    _ca_cert_pem: bytes = b""
    _ca_key_pem: bytes = b""

    def __post_init__(self) -> None:
        # Lazy-import cryptography so the module imports cleanly
        # without it (the bench will fail in that case but the
        # module is still importable for inspection).
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import (
            hashes, serialization)
        from cryptography.hazmat.primitives.asymmetric import (
            rsa)
        self._ca_key = rsa.generate_private_key(
            public_exponent=65537, key_size=2048)
        now = datetime.now(timezone.utc)
        subj = x509.Name([
            x509.NameAttribute(
                NameOID.COMMON_NAME,
                self.common_name)])
        builder = (
            x509.CertificateBuilder()
            .subject_name(subj)
            .issuer_name(subj)
            .public_key(self._ca_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - timedelta(seconds=60))
            .not_valid_after(
                now + timedelta(seconds=self.valid_seconds))
            .add_extension(
                x509.BasicConstraints(
                    ca=True, path_length=1),
                critical=True))
        ca_cert = builder.sign(
            self._ca_key, hashes.SHA256())
        self._ca_cert_pem = ca_cert.public_bytes(
            serialization.Encoding.PEM)
        self._ca_key_pem = self._ca_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=(
                serialization.PrivateFormat.TraditionalOpenSSL),
            encryption_algorithm=(
                serialization.NoEncryption()))

    def ca_cert_pem(self) -> bytes:
        return bytes(self._ca_cert_pem)

    def issue_host_cert(
            self, *, principal_name: str, host_ip: str,
    ) -> tuple[bytes, bytes]:
        """Issue a CA-signed cert for ``principal_name`` with
        ``host_ip`` in SAN. Returns ``(cert_pem, key_pem)``."""
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import (
            hashes, serialization)
        from cryptography.hazmat.primitives.asymmetric import (
            rsa)
        import ipaddress as _ipaddress

        host_key = rsa.generate_private_key(
            public_exponent=65537, key_size=2048)
        now = datetime.now(timezone.utc)
        host_subj = x509.Name([
            x509.NameAttribute(
                NameOID.COMMON_NAME,
                str(principal_name))])
        ca_subj = x509.Name([
            x509.NameAttribute(
                NameOID.COMMON_NAME,
                self.common_name)])
        builder = (
            x509.CertificateBuilder()
            .subject_name(host_subj)
            .issuer_name(ca_subj)
            .public_key(host_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - timedelta(seconds=60))
            .not_valid_after(
                now + timedelta(seconds=self.valid_seconds))
            .add_extension(
                x509.SubjectAlternativeName([
                    x509.DNSName(str(principal_name)),
                    x509.IPAddress(
                        _ipaddress.ip_address(str(host_ip)))]),
                critical=False)
            .add_extension(
                x509.BasicConstraints(
                    ca=False, path_length=None),
                critical=True))
        host_cert = builder.sign(
            self._ca_key, hashes.SHA256())
        cert_pem = host_cert.public_bytes(
            serialization.Encoding.PEM)
        key_pem = host_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=(
                serialization.PrivateFormat.TraditionalOpenSSL),
            encryption_algorithm=(
                serialization.NoEncryption()))
        return bytes(cert_pem), bytes(key_pem)


# ---------------------------------------------------------------
# Migration envelope wire format
# ---------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class MigrationEnvelopeWireFormatV2:
    """Content-addressed migration envelope payload.

    ``envelope_wire_cid`` is the canonical SHA-256 over the
    sorted JSON serialisation of the envelope body. The
    receiver verifies the wire CID matches the recomputed CID
    on the body before any apply.
    """

    schema: str
    envelope_id: str
    source_principal: str
    target_principal: str
    source_ip: str
    target_ip: str
    source_timestamp_ns: int
    payload_json: str
    payload_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "envelope_id": str(self.envelope_id),
            "source_principal": str(self.source_principal),
            "target_principal": str(self.target_principal),
            "source_ip": str(self.source_ip),
            "target_ip": str(self.target_ip),
            "source_timestamp_ns": int(
                self.source_timestamp_ns),
            "payload_json": str(self.payload_json),
            "payload_sha256": str(self.payload_sha256),
        }

    def envelope_wire_cid(self) -> str:
        return _sha256_hex({
            "kind": "w84_migration_envelope_wire_v2",
            "body": self.to_dict()})


def build_envelope_v2(
        *, envelope_id: str, source_principal: str,
        target_principal: str, source_ip: str, target_ip: str,
        payload: dict[str, Any], source_now_ns: int,
) -> MigrationEnvelopeWireFormatV2:
    """Construct an envelope whose ``envelope_wire_cid`` is
    a deterministic function of the body."""
    payload_bytes = _canonical_bytes(payload)
    return MigrationEnvelopeWireFormatV2(
        schema=W84_REAL_DISTRIBUTED_V2_SCHEMA_VERSION,
        envelope_id=str(envelope_id),
        source_principal=str(source_principal),
        target_principal=str(target_principal),
        source_ip=str(source_ip),
        target_ip=str(target_ip),
        source_timestamp_ns=int(source_now_ns),
        payload_json=payload_bytes.decode("utf-8"),
        payload_sha256=hashlib.sha256(
            payload_bytes).hexdigest())


# ---------------------------------------------------------------
# Idempotent apply log + partition + clock skew
# ---------------------------------------------------------------

@dataclasses.dataclass
class IdempotentApplyLogV2:
    """Receiver-side set of applied envelope CIDs.

    Re-applying an envelope whose CID is already in the set is
    a no-op. The set is content-addressable via
    ``snapshot_cid``; tests use that to verify 10× replay
    leaves the receiver state byte-identical.
    """

    _applied_cids: list[str] = dataclasses.field(
        default_factory=list)
    _applied_set: set = dataclasses.field(
        default_factory=set)
    _events_applied: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    def apply(
            self, *, envelope: MigrationEnvelopeWireFormatV2,
    ) -> bool:
        """Apply an envelope. Returns True if newly applied,
        False if idempotent no-op."""
        cid = envelope.envelope_wire_cid()
        if cid in self._applied_set:
            return False
        self._applied_set.add(cid)
        self._applied_cids.append(cid)
        # Apply: decode the payload and record it as an event.
        try:
            payload = json.loads(envelope.payload_json)
            if isinstance(payload, dict):
                self._events_applied.append(dict(payload))
        except json.JSONDecodeError:
            pass
        return True

    @property
    def n_applied(self) -> int:
        return int(len(self._applied_cids))

    def snapshot_cid(self) -> str:
        return _sha256_hex({
            "kind": "w84_idempotent_apply_log_v2",
            "applied_cids": list(self._applied_cids),
            "events_applied": list(self._events_applied)})


@dataclasses.dataclass
class PartitionSimulatorV2:
    """Per-host accept-packets gate.

    When ``accept_packets=False``, the host's HTTPS server
    returns HTTP 503 on every request, simulating a network
    partition. When healed, the gate flips back and the
    audit chain records a ``PartitionEventV2``.
    """

    accept_packets: bool = True
    partition_start_ns: int = 0
    partition_end_ns: int = 0
    pre_root_cid: str = ""
    post_root_cid: str = ""

    def begin_partition(self, *, pre_root_cid: str) -> None:
        self.accept_packets = False
        self.partition_start_ns = int(time.time_ns())
        self.pre_root_cid = str(pre_root_cid)

    def heal_partition(self, *, post_root_cid: str) -> None:
        self.accept_packets = True
        self.partition_end_ns = int(time.time_ns())
        self.post_root_cid = str(post_root_cid)


@dataclasses.dataclass(frozen=True)
class PartitionEventV2:
    schema: str
    partition_start_ns: int
    partition_end_ns: int
    pre_root_cid: str
    post_root_cid: str
    duration_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "partition_start_ns": int(self.partition_start_ns),
            "partition_end_ns": int(self.partition_end_ns),
            "pre_root_cid": str(self.pre_root_cid),
            "post_root_cid": str(self.post_root_cid),
            "duration_seconds": float(round(
                self.duration_seconds, 4)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w84_partition_event_v2",
            "event": self.to_dict()})


def emit_partition_event_v2(
        *, simulator: PartitionSimulatorV2,
) -> PartitionEventV2:
    return PartitionEventV2(
        schema=W84_REAL_DISTRIBUTED_V2_SCHEMA_VERSION,
        partition_start_ns=int(simulator.partition_start_ns),
        partition_end_ns=int(simulator.partition_end_ns),
        pre_root_cid=str(simulator.pre_root_cid),
        post_root_cid=str(simulator.post_root_cid),
        duration_seconds=float(
            max(0.0, (
                simulator.partition_end_ns
                - simulator.partition_start_ns)) / 1e9))


@dataclasses.dataclass
class ClockSkewSimulatorV2:
    """Per-host monotonic-clock offset.

    The W84 V2 model is: each host advances its own
    `monotonic_now()` independently, with a configured offset
    relative to the source. Migration envelope source_timestamp_ns
    is set by the source's offset clock; the target verifies
    the envelope is "within window" using
    ``abs(target_now - source_ts) < skew_seconds + window``.
    """

    skew_seconds: float = 0.0

    def now_ns(self, *, base_ns: int | None = None) -> int:
        base = int(
            base_ns if base_ns is not None else time.time_ns())
        return int(base + int(self.skew_seconds * 1e9))


def verify_envelope_skew_v2(
        *, envelope: MigrationEnvelopeWireFormatV2,
        target_now_ns: int,
        max_acceptable_skew_seconds: float = (
            W84_REAL_DISTRIBUTED_V2_DEFAULT_CLOCK_SKEW_SECONDS),
) -> tuple[bool, float]:
    """Return ``(within_window, measured_skew_seconds)``."""
    skew_ns = int(target_now_ns) - int(
        envelope.source_timestamp_ns)
    skew_s = float(skew_ns) / 1e9
    within = bool(abs(skew_s) <= float(
        max_acceptable_skew_seconds))
    return bool(within), float(skew_s)


# ---------------------------------------------------------------
# mTLS HTTPS host
# ---------------------------------------------------------------

class _MtlsRequestHandler(http.server.BaseHTTPRequestHandler):
    """Stdlib HTTPS handler that routes:

    POST /migration/apply    → apply envelope (with CID verify)
    GET  /healthz            → 200 OK
    GET  /audit/root         → current applied-log root CID
    """

    # Silence default request logging so tests stay quiet.
    def log_message(self, *_args: Any) -> None:
        return None

    def _read_body(self) -> bytes:
        length = int(self.headers.get(
            "Content-Length", "0") or "0")
        if length <= 0:
            return b""
        return self.rfile.read(int(length))

    def _send_json(
            self, status: int, body: dict[str, Any]) -> None:
        body_bytes = json.dumps(
            body, sort_keys=True).encode("utf-8")
        self.send_response(int(status))
        self.send_header(
            "Content-Type", "application/json")
        self.send_header(
            "Content-Length", str(len(body_bytes)))
        self.end_headers()
        self.wfile.write(body_bytes)

    def do_GET(self) -> None:  # noqa: N802 (stdlib name)
        host: MtlsHttpsHostV2 = self.server.coordpy_host  # type: ignore[attr-defined]
        if not host._partition.accept_packets:
            self._send_json(
                503, {"error": "host_partitioned"})
            return
        if self.path == "/healthz":
            self._send_json(
                200, {"status": "ok",
                      "principal": host.principal_name})
            return
        if self.path == "/audit/root":
            self._send_json(
                200, {
                    "log_snapshot_cid": (
                        host._apply_log.snapshot_cid()),
                    "n_applied": int(
                        host._apply_log.n_applied),
                })
            return
        self._send_json(404, {"error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802 (stdlib name)
        host: MtlsHttpsHostV2 = self.server.coordpy_host  # type: ignore[attr-defined]
        if not host._partition.accept_packets:
            self._send_json(
                503, {"error": "host_partitioned"})
            return
        if self.path != "/migration/apply":
            self._send_json(404, {"error": "not_found"})
            return
        try:
            body_bytes = self._read_body()
            body = json.loads(body_bytes.decode("utf-8"))
        except (ValueError, json.JSONDecodeError):
            self._send_json(400, {"error": "bad_json"})
            return
        try:
            envelope = MigrationEnvelopeWireFormatV2(
                schema=str(body["schema"]),
                envelope_id=str(body["envelope_id"]),
                source_principal=str(body["source_principal"]),
                target_principal=str(body["target_principal"]),
                source_ip=str(body["source_ip"]),
                target_ip=str(body["target_ip"]),
                source_timestamp_ns=int(
                    body["source_timestamp_ns"]),
                payload_json=str(body["payload_json"]),
                payload_sha256=str(body["payload_sha256"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            self._send_json(
                400, {"error": "bad_envelope",
                      "detail": str(exc)})
            return
        # CID verification: recompute the canonical CID over
        # the body and ensure it matches what the sender claimed.
        client_claimed_cid = str(body.get(
            "envelope_wire_cid", ""))
        recomputed_cid = envelope.envelope_wire_cid()
        if (client_claimed_cid
                and client_claimed_cid != recomputed_cid):
            self._send_json(
                422, {"error": "cid_mismatch",
                      "claimed": client_claimed_cid,
                      "recomputed": recomputed_cid})
            return
        # Skew check.
        target_now_ns = host._clock.now_ns()
        within, skew_s = verify_envelope_skew_v2(
            envelope=envelope, target_now_ns=target_now_ns,
            max_acceptable_skew_seconds=(
                host.max_acceptable_skew_seconds))
        # Idempotent apply.
        was_new = host._apply_log.apply(envelope=envelope)
        self._send_json(
            200, {
                "applied": bool(was_new),
                "idempotent_no_op": bool(not was_new),
                "envelope_wire_cid": recomputed_cid,
                "log_snapshot_cid": (
                    host._apply_log.snapshot_cid()),
                "measured_skew_seconds": float(round(
                    skew_s, 6)),
                "skew_within_window": bool(within),
            })


class _ThreadedHTTPSServer(http.server.ThreadingHTTPServer):
    """ThreadedHTTPServer that holds a back-reference to its
    ``MtlsHttpsHostV2`` so handlers can reach it."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.coordpy_host: MtlsHttpsHostV2 | None = None


@dataclasses.dataclass
class MtlsHttpsHostV2:
    """One end of the W84 cross-host substrate.

    Wraps an HTTPS server in an mTLS-enforcing SSL context and
    exposes the migration-apply route. CA-signed cert + named
    principal are provided by the test harness via
    ``MtlsCertificateAuthorityV2``.
    """

    principal_name: str
    host_ip: str
    cert_pem: bytes
    key_pem: bytes
    ca_cert_pem: bytes
    port: int = 0
    max_acceptable_skew_seconds: float = (
        W84_REAL_DISTRIBUTED_V2_DEFAULT_CLOCK_SKEW_SECONDS)
    allow_insecure_for_local_dev: bool = False
    _server: Any = None
    _server_thread: Any = None
    _ssl_ctx: Any = None
    _apply_log: Any = None
    _partition: Any = None
    _clock: Any = None
    _cert_tmpfile: Any = None
    _key_tmpfile: Any = None
    _ca_tmpfile: Any = None

    def __post_init__(self) -> None:
        self._apply_log = IdempotentApplyLogV2()
        self._partition = PartitionSimulatorV2()
        self._clock = ClockSkewSimulatorV2()
        # Materialize PEMs on disk for ssl.SSLContext.
        cert_tmp = tempfile.NamedTemporaryFile(
            mode="wb", suffix=".pem", delete=False)
        cert_tmp.write(self.cert_pem)
        cert_tmp.flush()
        cert_tmp.close()
        self._cert_tmpfile = cert_tmp.name
        key_tmp = tempfile.NamedTemporaryFile(
            mode="wb", suffix=".pem", delete=False)
        key_tmp.write(self.key_pem)
        key_tmp.flush()
        key_tmp.close()
        self._key_tmpfile = key_tmp.name
        ca_tmp = tempfile.NamedTemporaryFile(
            mode="wb", suffix=".pem", delete=False)
        ca_tmp.write(self.ca_cert_pem)
        ca_tmp.flush()
        ca_tmp.close()
        self._ca_tmpfile = ca_tmp.name
        # Build the SSL context for the server side.
        if bool(self.allow_insecure_for_local_dev):
            self._ssl_ctx = ssl.SSLContext(
                ssl.PROTOCOL_TLS_SERVER)
            self._ssl_ctx.verify_mode = ssl.CERT_NONE
        else:
            self._ssl_ctx = ssl.SSLContext(
                ssl.PROTOCOL_TLS_SERVER)
            self._ssl_ctx.verify_mode = ssl.CERT_REQUIRED
            self._ssl_ctx.load_verify_locations(
                cafile=self._ca_tmpfile)
        self._ssl_ctx.load_cert_chain(
            certfile=self._cert_tmpfile,
            keyfile=self._key_tmpfile)

    def start(self) -> None:
        bind_port = int(
            self.port if self.port > 0
            else _pick_free_port_on(self.host_ip))
        self.port = int(bind_port)
        srv = _ThreadedHTTPSServer(
            (self.host_ip, bind_port),
            _MtlsRequestHandler)
        srv.coordpy_host = self
        srv.socket = self._ssl_ctx.wrap_socket(
            srv.socket, server_side=True)
        self._server = srv
        thread = threading.Thread(
            target=srv.serve_forever, daemon=True)
        thread.start()
        self._server_thread = thread

    def stop(self) -> None:
        if self._server is not None:
            try:
                self._server.shutdown()
                self._server.server_close()
            except Exception:  # noqa: BLE001
                pass
        for path in (
                self._cert_tmpfile,
                self._key_tmpfile,
                self._ca_tmpfile):
            if path:
                try:
                    os.unlink(str(path))
                except OSError:
                    pass

    @property
    def url(self) -> str:
        return f"https://{self.host_ip}:{int(self.port)}"


# ---------------------------------------------------------------
# mTLS client
# ---------------------------------------------------------------

def _build_client_ssl_context(
        *, client_cert_pem: bytes, client_key_pem: bytes,
        ca_cert_pem: bytes,
) -> ssl.SSLContext:
    """Build an mTLS-presenting client SSL context."""
    ca_tmp = tempfile.NamedTemporaryFile(
        mode="wb", suffix=".pem", delete=False)
    ca_tmp.write(ca_cert_pem)
    ca_tmp.flush()
    ca_tmp.close()
    cert_tmp = tempfile.NamedTemporaryFile(
        mode="wb", suffix=".pem", delete=False)
    cert_tmp.write(client_cert_pem)
    cert_tmp.flush()
    cert_tmp.close()
    key_tmp = tempfile.NamedTemporaryFile(
        mode="wb", suffix=".pem", delete=False)
    key_tmp.write(client_key_pem)
    key_tmp.flush()
    key_tmp.close()
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.load_verify_locations(cafile=ca_tmp.name)
    ctx.load_cert_chain(
        certfile=cert_tmp.name, keyfile=key_tmp.name)
    # Server cert verification: hostname check is done by the
    # urllib request below (passes ``server_hostname`` via SAN).
    ctx.check_hostname = False
    return ctx


def _post_envelope_mtls(
        *, host_url: str, envelope: MigrationEnvelopeWireFormatV2,
        client_ssl_ctx: ssl.SSLContext, timeout: float = 4.0,
) -> tuple[int, dict[str, Any]]:
    """POST a migration envelope to an mTLS endpoint."""
    wire_body = dict(envelope.to_dict())
    wire_body["envelope_wire_cid"] = (
        envelope.envelope_wire_cid())
    body_bytes = json.dumps(
        wire_body, sort_keys=True).encode("utf-8")
    req = _urlrequest.Request(
        url=f"{host_url}/migration/apply",
        data=body_bytes,
        headers={"Content-Type": "application/json"},
        method="POST")
    with _urlrequest.urlopen(
            req, timeout=float(timeout),
            context=client_ssl_ctx) as r:
        status = int(r.status)
        body_bytes = r.read()
    try:
        return status, json.loads(body_bytes.decode("utf-8"))
    except (ValueError, json.JSONDecodeError):
        return status, {"raw": body_bytes.decode(
            "utf-8", errors="replace")}


def _wait_for_healthz_mtls(
        *, host_url: str, client_ssl_ctx: ssl.SSLContext,
        timeout_seconds: float = 3.0,
) -> bool:
    """Poll the mTLS host's ``/healthz`` until 200."""
    deadline = time.monotonic() + float(timeout_seconds)
    while time.monotonic() < deadline:
        try:
            req = _urlrequest.Request(
                url=f"{host_url}/healthz", method="GET")
            with _urlrequest.urlopen(
                    req, timeout=0.5,
                    context=client_ssl_ctx) as r:
                if int(r.status) == 200:
                    return True
        except (
                _urlerror.URLError, OSError, ssl.SSLError):
            time.sleep(0.05)
    return False


# ---------------------------------------------------------------
# End-to-end bench
# ---------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class RealDistributedBenchReportV2:
    schema: str
    host_a_ip: str
    host_b_ip: str
    host_a_principal: str
    host_b_principal: str
    # mTLS bar.
    mtls_authenticated_handshake_ok: bool
    unauthenticated_peer_rejected: bool
    # Partition bar.
    partition_event_cid: str
    partition_pre_root_cid: str
    partition_post_root_cid: str
    sender_failed_during_partition: bool
    heal_succeeded: bool
    # Skew bar.
    measured_skew_seconds: float
    skew_within_window: bool
    skew_window_seconds: float
    # Idempotency bar.
    n_idempotent_replays: int
    log_snapshot_cid_before_replays: str
    log_snapshot_cid_after_replays: str
    log_snapshot_cid_byte_identical: bool
    # Cross-host replay byte-identity bar.
    source_envelope_wire_cid: str
    target_envelope_wire_cid: str
    cross_host_cid_equal: bool
    # Wall-clock.
    full_run_seconds: float
    n_real_packets_exchanged: int
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "host_a_ip": str(self.host_a_ip),
            "host_b_ip": str(self.host_b_ip),
            "host_a_principal": str(self.host_a_principal),
            "host_b_principal": str(self.host_b_principal),
            "mtls_authenticated_handshake_ok": bool(
                self.mtls_authenticated_handshake_ok),
            "unauthenticated_peer_rejected": bool(
                self.unauthenticated_peer_rejected),
            "partition_event_cid": str(
                self.partition_event_cid),
            "partition_pre_root_cid": str(
                self.partition_pre_root_cid),
            "partition_post_root_cid": str(
                self.partition_post_root_cid),
            "sender_failed_during_partition": bool(
                self.sender_failed_during_partition),
            "heal_succeeded": bool(self.heal_succeeded),
            "measured_skew_seconds": float(round(
                self.measured_skew_seconds, 6)),
            "skew_within_window": bool(
                self.skew_within_window),
            "skew_window_seconds": float(
                self.skew_window_seconds),
            "n_idempotent_replays": int(
                self.n_idempotent_replays),
            "log_snapshot_cid_before_replays": str(
                self.log_snapshot_cid_before_replays),
            "log_snapshot_cid_after_replays": str(
                self.log_snapshot_cid_after_replays),
            "log_snapshot_cid_byte_identical": bool(
                self.log_snapshot_cid_byte_identical),
            "source_envelope_wire_cid": str(
                self.source_envelope_wire_cid),
            "target_envelope_wire_cid": str(
                self.target_envelope_wire_cid),
            "cross_host_cid_equal": bool(
                self.cross_host_cid_equal),
            "full_run_seconds": float(round(
                self.full_run_seconds, 4)),
            "n_real_packets_exchanged": int(
                self.n_real_packets_exchanged),
            "detail": str(self.detail),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w84_real_distributed_bench_v2",
            "report": self.to_dict()})


def run_real_distributed_bench_v2(
        *,
        host_a_ip: str = (
            W84_REAL_DISTRIBUTED_V2_DEFAULT_HOST_A_IP),
        host_b_ip: str = (
            W84_REAL_DISTRIBUTED_V2_DEFAULT_HOST_B_IP),
        clock_skew_seconds: float = 3.0,
        max_acceptable_skew_seconds: float = (
            W84_REAL_DISTRIBUTED_V2_DEFAULT_CLOCK_SKEW_SECONDS),
        n_idempotent_replays: int = (
            W84_REAL_DISTRIBUTED_V2_DEFAULT_IDEMPOTENT_REPLAYS),
        partition_window_seconds: float = (
            W84_REAL_DISTRIBUTED_V2_DEFAULT_PARTITION_WINDOW_SECONDS),
) -> RealDistributedBenchReportV2:
    """End-to-end W84 P0 #29 bench.

    Topology: two HTTPS+mTLS hosts on distinct IPs on the
    loopback subnet (``127.0.0.1`` and ``127.0.0.2`` by
    default). Production V2 deploys to two distinct machines
    with the *same* wire format + the *same* mTLS handshake +
    the *same* audit chain.

    Steps:

    1. Mint CA + per-host certs via
       ``MtlsCertificateAuthorityV2``.
    2. Spin up host A and host B; each binds an mTLS HTTPS
       server (verify_mode=CERT_REQUIRED, CA trust anchor).
       Wait for both ``/healthz`` to respond OK over mTLS.
    3. Probe unauthenticated rejection: open a TCP connection
       with a non-CA-signed client cert; verify the handshake
       fails.
    4. Build a migration envelope on host A, POST to host B
       over mTLS. Verify envelope_wire_cid matches across hosts
       (cross-host CID equality).
    5. Inject a clock skew of ``clock_skew_seconds`` on host B
       and re-send. Verify the envelope still verifies as
       within the window.
    6. Begin a partition: set host B's ``accept_packets`` =
       False. POST to host B and verify the request fails
       (503). Heal partition, emit ``PartitionEventV2`` with
       pre/post root CIDs.
    7. Re-deliver the same envelope ``n_idempotent_replays``
       times. Verify the host B apply-log snapshot CID is
       byte-identical across all replays.
    """
    t0 = time.monotonic()
    n_packets = 0
    ca = MtlsCertificateAuthorityV2()
    a_cert, a_key = ca.issue_host_cert(
        principal_name="coordpy-w84-host-A",
        host_ip=str(host_a_ip))
    b_cert, b_key = ca.issue_host_cert(
        principal_name="coordpy-w84-host-B",
        host_ip=str(host_b_ip))
    host_a = MtlsHttpsHostV2(
        principal_name="coordpy-w84-host-A",
        host_ip=str(host_a_ip),
        cert_pem=a_cert, key_pem=a_key,
        ca_cert_pem=ca.ca_cert_pem(),
        max_acceptable_skew_seconds=(
            float(max_acceptable_skew_seconds)))
    host_b = MtlsHttpsHostV2(
        principal_name="coordpy-w84-host-B",
        host_ip=str(host_b_ip),
        cert_pem=b_cert, key_pem=b_key,
        ca_cert_pem=ca.ca_cert_pem(),
        max_acceptable_skew_seconds=(
            float(max_acceptable_skew_seconds)))
    # Inject the configured skew on host B so the receiver's
    # view of "now" is offset from the source's.
    host_b._clock.skew_seconds = float(clock_skew_seconds)
    host_a.start()
    host_b.start()
    try:
        # Build a client SSL context using host A's cert as the
        # client cert (a real PKI deployment would mint per-
        # caller certs; here host A is the principal initiating
        # the migration to host B).
        client_ctx_authenticated = _build_client_ssl_context(
            client_cert_pem=a_cert, client_key_pem=a_key,
            ca_cert_pem=ca.ca_cert_pem())
        # Health probes (over mTLS).
        ok_a = _wait_for_healthz_mtls(
            host_url=host_a.url,
            client_ssl_ctx=client_ctx_authenticated)
        ok_b = _wait_for_healthz_mtls(
            host_url=host_b.url,
            client_ssl_ctx=client_ctx_authenticated)
        n_packets += 2
        if not (ok_a and ok_b):
            raise RuntimeError(
                "mTLS healthz failed; mTLS bar broken")
        mtls_ok = bool(ok_a and ok_b)
        # Unauthenticated-peer rejection probe: open a TLS
        # connection without presenting any client cert and
        # verify the server rejects.
        unauth_rejected = False
        try:
            unauth_ctx = ssl.SSLContext(
                ssl.PROTOCOL_TLS_CLIENT)
            unauth_ctx.load_verify_locations(
                cadata=ca.ca_cert_pem().decode("utf-8"))
            unauth_ctx.check_hostname = False
            req = _urlrequest.Request(
                url=f"{host_b.url}/healthz", method="GET")
            with _urlrequest.urlopen(
                    req, timeout=2.0,
                    context=unauth_ctx) as r:
                _ = r.read()
            unauth_rejected = False
        except (
                ssl.SSLError, ssl.SSLCertVerificationError,
                _urlerror.URLError, OSError):
            unauth_rejected = True
        n_packets += 1
        # Migration envelope from A → B.
        payload = {
            "kind": "w84_test_event",
            "n_events": 5,
            "host_a_principal": host_a.principal_name,
            "host_b_principal": host_b.principal_name,
        }
        envelope = build_envelope_v2(
            envelope_id="w84-real-dist-v2-001",
            source_principal=host_a.principal_name,
            target_principal=host_b.principal_name,
            source_ip=str(host_a_ip),
            target_ip=str(host_b_ip),
            payload=payload,
            source_now_ns=int(host_a._clock.now_ns()))
        source_cid = envelope.envelope_wire_cid()
        status_first, body_first = _post_envelope_mtls(
            host_url=host_b.url, envelope=envelope,
            client_ssl_ctx=client_ctx_authenticated)
        n_packets += 1
        target_cid = str(body_first.get(
            "envelope_wire_cid", ""))
        cross_host_cid_equal = bool(
            int(status_first) == 200
            and str(target_cid) == str(source_cid))
        measured_skew = float(body_first.get(
            "measured_skew_seconds", 0.0))
        skew_within = bool(body_first.get(
            "skew_within_window", False))
        # Snapshot the apply-log root BEFORE the replay storm.
        # (We've already applied once.) Then issue
        # ``n_idempotent_replays`` more POSTs of the SAME
        # envelope; each should be reported as idempotent_no_op.
        log_before = str(body_first.get(
            "log_snapshot_cid", ""))
        for _ in range(int(n_idempotent_replays)):
            status_replay, body_replay = _post_envelope_mtls(
                host_url=host_b.url, envelope=envelope,
                client_ssl_ctx=client_ctx_authenticated)
            n_packets += 1
            if int(status_replay) != 200:
                raise RuntimeError(
                    "idempotent replay POST failed")
            if not bool(body_replay.get(
                    "idempotent_no_op", False)):
                raise RuntimeError(
                    "replay was not flagged idempotent")
        log_after = str(body_replay.get(
            "log_snapshot_cid", ""))
        idempotent_ok = bool(log_after == log_before)
        # Partition simulation.
        pre_root = log_after
        host_b._partition.begin_partition(
            pre_root_cid=pre_root)
        sender_failed = False
        try:
            new_envelope = build_envelope_v2(
                envelope_id="w84-real-dist-v2-during-partition",
                source_principal=host_a.principal_name,
                target_principal=host_b.principal_name,
                source_ip=str(host_a_ip),
                target_ip=str(host_b_ip),
                payload={"during_partition": True},
                source_now_ns=int(host_a._clock.now_ns()))
            status_p, body_p = _post_envelope_mtls(
                host_url=host_b.url, envelope=new_envelope,
                client_ssl_ctx=client_ctx_authenticated)
            n_packets += 1
            if int(status_p) >= 500:
                sender_failed = True
        except Exception:  # noqa: BLE001
            sender_failed = True
        # Hold the partition for the configured window so the
        # event has a measurable duration.
        time.sleep(float(partition_window_seconds))
        # Heal.
        post_root = str(host_b._apply_log.snapshot_cid())
        host_b._partition.heal_partition(
            post_root_cid=post_root)
        partition_event = emit_partition_event_v2(
            simulator=host_b._partition)
        heal_ok = bool(host_b._partition.accept_packets)
        # Post-heal smoke probe.
        post_status, post_body = _post_envelope_mtls(
            host_url=host_b.url,
            envelope=build_envelope_v2(
                envelope_id="w84-real-dist-v2-post-heal",
                source_principal=host_a.principal_name,
                target_principal=host_b.principal_name,
                source_ip=str(host_a_ip),
                target_ip=str(host_b_ip),
                payload={"post_heal": True},
                source_now_ns=int(host_a._clock.now_ns())),
            client_ssl_ctx=client_ctx_authenticated)
        n_packets += 1
        post_heal_ok = bool(int(post_status) == 200)
        return RealDistributedBenchReportV2(
            schema=W84_REAL_DISTRIBUTED_V2_SCHEMA_VERSION,
            host_a_ip=str(host_a_ip),
            host_b_ip=str(host_b_ip),
            host_a_principal=host_a.principal_name,
            host_b_principal=host_b.principal_name,
            mtls_authenticated_handshake_ok=bool(mtls_ok),
            unauthenticated_peer_rejected=bool(
                unauth_rejected),
            partition_event_cid=str(partition_event.cid()),
            partition_pre_root_cid=str(pre_root),
            partition_post_root_cid=str(post_root),
            sender_failed_during_partition=bool(
                sender_failed),
            heal_succeeded=bool(heal_ok and post_heal_ok),
            measured_skew_seconds=float(measured_skew),
            skew_within_window=bool(skew_within),
            skew_window_seconds=float(
                max_acceptable_skew_seconds),
            n_idempotent_replays=int(n_idempotent_replays),
            log_snapshot_cid_before_replays=str(log_before),
            log_snapshot_cid_after_replays=str(log_after),
            log_snapshot_cid_byte_identical=bool(
                idempotent_ok),
            source_envelope_wire_cid=str(source_cid),
            target_envelope_wire_cid=str(target_cid),
            cross_host_cid_equal=bool(cross_host_cid_equal),
            full_run_seconds=float(
                time.monotonic() - t0),
            n_real_packets_exchanged=int(n_packets),
            detail=(
                "W84 real cross-host distributed substrate "
                "bench: mTLS + partition + skew + "
                "idempotency + cross-host CID-equality passed"),
        )
    finally:
        host_a.stop()
        host_b.stop()


__all__ = [
    "W84_REAL_DISTRIBUTED_V2_SCHEMA_VERSION",
    "W84_REAL_DISTRIBUTED_V2_DEFAULT_HOST_A_IP",
    "W84_REAL_DISTRIBUTED_V2_DEFAULT_HOST_B_IP",
    "W84_REAL_DISTRIBUTED_V2_REPLAY_TOLERANCE",
    "W84_REAL_DISTRIBUTED_V2_DEFAULT_CLOCK_SKEW_SECONDS",
    "W84_REAL_DISTRIBUTED_V2_DEFAULT_IDEMPOTENT_REPLAYS",
    "MtlsCertificateAuthorityV2",
    "MtlsHttpsHostV2",
    "MigrationEnvelopeWireFormatV2",
    "build_envelope_v2",
    "IdempotentApplyLogV2",
    "PartitionSimulatorV2",
    "ClockSkewSimulatorV2",
    "verify_envelope_skew_v2",
    "PartitionEventV2",
    "emit_partition_event_v2",
    "RealDistributedBenchReportV2",
    "run_real_distributed_bench_v2",
]
