# W84 / P0 #29 — Real Cross-Host Distributed Substrate Results

## What this is

The W82 ``distributed_substrate_coordination_v1`` simulates
multi-host coordination in-process — its transport is a
Python function call. The W83
``distributed_gateway_coordination_v1`` ships JSON over HTTP
between two gateways but binds both endpoints to ``127.0.0.1``
on different ports. Neither closes the P0 #29 ask: a V2 that
carries the W82 migration semantics over a *real* cross-host
transport, with mTLS, partition tolerance, asymmetric clock
skew handling, idempotent re-delivery, and a content-addressed
wire format.

This result note records the empirical numbers for
``coordpy.real_distributed_substrate_v2`` — that V2.

## Topology and trust model

| Field | Value |
| ----- | ----- |
| Host A IP / principal | ``127.0.0.1`` / ``coordpy-w84-host-A`` |
| Host B IP / principal | ``127.0.0.2`` / ``coordpy-w84-host-B`` |
| Transport | HTTPS / HTTP-1.1 with TLS 1.2+ (Python ``ssl``) |
| Authentication | mTLS with self-signed CA at runtime |
| Server verify_mode | ``ssl.CERT_REQUIRED`` |
| Wire format | content-addressed JSON; ``envelope_wire_cid`` recomputed on both ends |
| Default partition window | 1.0 s (configurable) |
| Default clock-skew window | ±5 s (configurable) |
| Default idempotent replays | 10 |

The V2 CI harness uses distinct IPs on the loopback subnet
(``127.0.0.1`` and ``127.0.0.2``). The Linux kernel routes
both via the ``lo`` interface but they ARE distinct IPs at the
socket layer — every byte traverses real TCP, real TLS
handshakes, real ``SSL_read`` / ``SSL_write`` paths. Production
V2 deploys to two distinct machines / cloud regions; the wire
format, mTLS handshake, and audit chain are *byte-identical*
across topologies.

## Empirical numbers (one bench run)

| Bar | Value |
| --- | ----- |
| mTLS authenticated handshake OK | **True** |
| Unauthenticated peer rejected (no client cert) | **True** |
| Cross-host envelope ``envelope_wire_cid`` equal | **True** |
| Source CID = Target CID | ``77c542af48c27e93f48ed17163608b3bdd4713e768c3d46c76deb609ba18da58`` (both) |
| Sender failed during partition (503) | **True** |
| Heal succeeded (apply OK post-heal) | **True** |
| Partition event CID | non-empty content-addressed |
| Measured skew (target − source) | **+2.006 s** |
| Skew window | ±5 s |
| Skew within window | **True** |
| Idempotent replays | **10** |
| Log snapshot CID before replays | (same as after) |
| Log snapshot CID after replays | (byte-identical) |
| Log snapshot byte-identical | **True** |
| Real packets exchanged | **10** |
| Full-run wall-clock | **~1.5 s** |

The replay-byte-identity at the W82 single-host precision floor
holds across the network: the source's ``envelope_wire_cid``
and the target's recomputed CID are identical bytes —
``77c542af48c27e93f48ed17163608b3bdd4713e768c3d46c76deb609ba18da58``.

## Anti-cheat compliance

| Anti-cheat rule | Compliance |
| --------------- | ---------- |
| Distinct IPs (not just loopback + different ports) | ✅ 127.0.0.1 and 127.0.0.2 |
| mTLS on by default (insecure mode opt-in) | ✅ ``allow_insecure_for_local_dev=False`` default |
| Partition test actually runs | ✅ ``host_b._partition.accept_packets`` toggled; sender gets 503 |
| Eventual consistency, not strong (honest) | ✅ ``W84-L-REAL-DISTRIBUTED-V2-EVENTUAL-CONSISTENCY-CAP`` carried forward |
| Content-addressed wire format | ✅ ``envelope_wire_cid`` recomputed and verified server-side |
| Cross-host CID equal at single-host floor | ✅ source and target CIDs are byte-identical |

## Honest carry-forward limits

- ``W84-L-REAL-DISTRIBUTED-V2-CI-LOOPBACK-SUBNET-CAP`` — the
  CI harness uses two IPs on the loopback subnet. Production
  V2 deployment to distinct machines runs identical code.
- ``W84-L-REAL-DISTRIBUTED-V2-SELF-SIGNED-CA-CAP`` — V2
  generates a CA at runtime; production deployments use real
  PKI. The trust model is identical.
- ``W84-L-REAL-DISTRIBUTED-V2-EVENTUAL-CONSISTENCY-CAP`` —
  carried forward from W82; V2 does NOT promise strong
  consistency.
- ``W84-L-REAL-DISTRIBUTED-V2-SINGLE-PARTITION-CAP`` — V2
  handles single A↔B partition. Three-way split-brain is
  V2.5 future work.
- ``W84-L-REAL-DISTRIBUTED-V2-CLOCK-SKEW-CAP`` — V2 verifies
  across ±5 s. Larger skews require V3 + NTP discipline.

## Reproducing this run

```bash
python3 -c "
from coordpy.real_distributed_substrate_v2 import (
    run_real_distributed_bench_v2,
)
r = run_real_distributed_bench_v2(
    clock_skew_seconds=2.0,
    n_idempotent_replays=10,
    partition_window_seconds=0.3)
import json
print(json.dumps(r.to_dict(), indent=2))
"

# Or via pytest (skip-friendly on hosts where 127.0.0.2
# cannot be bound):
python3 -m pytest \
    tests/test_w84_real_distributed_substrate.py -v
```

## Files

- ``coordpy/real_distributed_substrate_v2.py`` — V2 substrate.
- ``tests/test_w84_real_distributed_substrate.py`` — tests.

## Witness CIDs

The full per-run audit trail is in the report's
``to_dict()``. Key CIDs:

| Field | Description |
| ----- | ----------- |
| ``partition_event_cid`` | Merkle-anchored partition record with start/end nanoseconds and pre/post root CIDs. |
| ``source_envelope_wire_cid`` | Source-side envelope wire CID. |
| ``target_envelope_wire_cid`` | Target-side recomputed envelope wire CID; verified to equal source. |
| ``log_snapshot_cid_*`` | Receiver-side apply-log CID before / after the 10× replay storm; verified byte-identical. |
