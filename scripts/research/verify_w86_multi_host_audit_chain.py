#!/usr/bin/env python3
"""Offline re-verifier for the W86 multi-host bench audit chain.

Given a ``multi_host_distributed_bench_report.json`` produced by
``scripts/run_w86_multi_host_bench.py``, re-derives the
content-addressed report_cid from the report bytes, and prints a
PASS/FAIL per #29 DoD bullet. Exits 0 iff every bool the issue
asks for is True.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def _sha256(payload):
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True,
            separators=(",", ":"),
            default=str).encode("utf-8")).hexdigest()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--report", required=True,
        help="path to multi_host_distributed_bench_report.json")
    args = p.parse_args(argv)
    report = json.loads(
        Path(args.report).read_bytes().decode("utf-8"))

    notes: list[str] = []
    ok = True

    # 1. Re-derive the report_cid from the report bytes (with
    # the recorded report_cid stripped out, matching the
    # cid() method).
    derived = _sha256({
        "kind":
            "w86_multi_host_distributed_bench_report_v1",
        "report": report,
    })
    # The bench writes the report WITHOUT a top-level report_cid
    # field; the cid is computed by callers on demand. So the
    # match is between cid(json.loads(disk_bytes)) and the
    # caller's recorded value (printed by the orchestrator at
    # end of run). We just print it here so a third party can
    # confirm the value matches what their run reported.
    notes.append(
        f"INFO report_cid (re-derived) = {derived}")

    # 2. Headline bars.
    bars = [
        ("mtls_unauthenticated_refused",
         "#29 mTLS handshake required on every connection "
         "(unauthenticated refused)"),
        ("mtls_bad_signature_refused",
         "#29 mTLS bad-signature requests refused (refuses "
         "unsigned/badly-signed peers)"),
        ("cross_host_post_root_match",
         "#29 cross-host post-root match after N envelopes "
         "(both hosts agree on the destination state root)"),
        ("partition_drops_all_traffic",
         "#29 partition test: every request during the drop "
         "window fails"),
        ("partition_heals_and_recovers",
         "#29 partition test: proxy heals + post-heal "
         "envelope succeeds"),
        ("skew_injection_within_tolerance",
         "#29 skew test: ±s clock skew within W84 tolerance"),
        ("idempotent_apply_holds",
         "#29 idempotency: 10 replays produce 1 distinct "
         "destination digest"),
    ]
    for key, label in bars:
        val = bool(report.get(key, False))
        if not val:
            ok = False
        notes.append(
            f"{'PASS' if val else 'FAIL'} {label}: {val}")

    # 3. Numeric witnesses (informational).
    for key in (
            "partition_recovery_seconds",
            "n_idempotent_replays",
            "n_distinct_replay_digests",
            "rtt_host_a_ms", "rtt_host_b_ms",
            "wall_clock_seconds"):
        notes.append(
            f"INFO {key} = {report.get(key)}")

    # 4. Topology witness.
    topo = report.get("topology", {})
    notes.append(
        f"INFO topology host_a_ip={topo.get('host_a_hostname')} "
        f"host_b_ip={topo.get('host_b_hostname')} "
        f"proxy={topo.get('proxy_base_url')}")
    notes.append(
        f"INFO topology docker_network_id="
        f"{topo.get('docker_network_id', '')[:16]}...")

    # 5. Anti-cheat: refuse to claim closure if the hosts are
    # named identically (would indicate a loopback configuration).
    if (topo.get("host_a_hostname")
            == topo.get("host_b_hostname")):
        ok = False
        notes.append(
            "FAIL host-a and host-b have identical hostnames "
            "— this is loopback configuration, not multi-host")
    else:
        notes.append(
            "PASS host-a and host-b have distinct hostnames")

    for n in notes:
        print(n)
    print(f"\nOVERALL: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
