#!/usr/bin/env python3
"""W86 multi-host distributed substrate bench orchestrator.

End-to-end #29 closure run:

1. Mint fresh HMAC keys for alpha / beta / client principals.
2. Write a .env file the docker-compose stack consumes.
3. ``docker compose up -d`` the W86 multi-host stack.
4. Wait for host-a + host-b + partition-proxy healthchecks.
5. Run ``run_multi_host_distributed_bench_v1`` against the
   stack, exercising mTLS, partition, skew, idempotency,
   cross-host post-root.
6. Inspect ``docker inspect`` for the topology (hostnames,
   IPs, network id).
7. Write the content-addressed report to disk.
8. Tear the stack down.

All from the terminal; no manual steps. The DoD literal
"≥ 2 containers in docker-compose" bar is satisfied.
"""
from __future__ import annotations

import base64
import json
import os
import secrets
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from coordpy.cross_process_distributed_substrate_v1 import (  # noqa: E402
    build_trust_root_v1,
)
from coordpy.multi_host_distributed_substrate_v1 import (  # noqa: E402
    MultiHostTopologyV1,
    W86_MULTI_HOST_DISTRIBUTED_V1_SCHEMA_VERSION,
    run_multi_host_distributed_bench_v1,
)


COMPOSE_FILE = (
    ROOT / "docker" / "compose-w86-multi-host.yml")
ENV_FILE = ROOT / "docker" / ".env.w86"
PROJECT_NAME = "coordpy_w86"


def _run(
        cmd: list[str], *, cwd: Path | None = None,
        check: bool = True, capture: bool = True,
        env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    print(f"$ {' '.join(cmd)}", flush=True)
    proc = subprocess.run(
        cmd, cwd=str(cwd) if cwd else None, check=check,
        capture_output=bool(capture), text=True, env=env)
    if capture and proc.stdout:
        sys.stdout.write(proc.stdout)
    if capture and proc.stderr:
        sys.stderr.write(proc.stderr)
    return proc


def _mint_hmac_key_b64(n_bytes: int = 32) -> str:
    return base64.b64encode(secrets.token_bytes(n_bytes)).decode(
        "ascii")


def _write_env(env_file: Path, **kv: str) -> None:
    env_file.parent.mkdir(parents=True, exist_ok=True)
    env_file.write_text(
        "\n".join(f"{k}={v}" for k, v in kv.items()) + "\n",
        encoding="utf-8")
    print(f"wrote env to {env_file}")


def _docker_inspect_topology(
        *, host_a_ctr: str, host_b_ctr: str,
        proxy_ctr: str, network: str,
) -> dict[str, str]:
    def inspect(name: str, fmt: str) -> str:
        r = _run(
            ["docker", "inspect", name,
             "--format", fmt], check=False)
        return (r.stdout or "").strip()
    return {
        "host_a_hostname": inspect(host_a_ctr, "{{.Config.Hostname}}"),
        "host_a_ip": inspect(
            host_a_ctr,
            "{{(index .NetworkSettings.Networks \""
            + network + "\").IPAddress}}"),
        "host_b_hostname": inspect(host_b_ctr, "{{.Config.Hostname}}"),
        "host_b_ip": inspect(
            host_b_ctr,
            "{{(index .NetworkSettings.Networks \""
            + network + "\").IPAddress}}"),
        "proxy_hostname": inspect(proxy_ctr, "{{.Config.Hostname}}"),
        "proxy_ip": inspect(
            proxy_ctr,
            "{{(index .NetworkSettings.Networks \""
            + network + "\").IPAddress}}"),
        "network_id": (
            _run(["docker", "network", "inspect",
                  network, "--format", "{{.Id}}"],
                 check=False).stdout or "").strip(),
    }


def _wait_for_healthz(
        url: str, *, timeout_s: float = 90.0,
        poll: float = 0.5,
) -> bool:
    deadline = time.time() + float(timeout_s)
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(
                    url, timeout=2.0) as r:
                if int(r.status) == 200:
                    return True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(float(poll))
    return False


def main() -> int:
    out_path = Path(os.environ.get(
        "W86_MH_OUT_PATH",
        str(ROOT / "results" / "w86" / "multi_host" /
            "multi_host_distributed_bench_report.json")))

    # 1. Mint HMAC keys for the three principals.
    alpha_key = _mint_hmac_key_b64()
    beta_key = _mint_hmac_key_b64()
    client_key = _mint_hmac_key_b64()
    _write_env(
        ENV_FILE,
        ALPHA_KEY=alpha_key, BETA_KEY=beta_key,
        CLIENT_KEY=client_key)

    # 2. Bring the stack up.
    print()
    print("=== docker compose up ===", flush=True)
    _run([
        "docker", "compose",
        "-p", PROJECT_NAME,
        "-f", str(COMPOSE_FILE),
        "--env-file", str(ENV_FILE),
        "up", "-d", "--build",
    ])

    try:
        # 3. Wait for healthz (host-a, host-b, proxy).
        host_a_url = "http://127.0.0.1:18080"
        host_b_url = "http://127.0.0.1:18081"
        proxy_url = "http://127.0.0.1:19000"
        print()
        print("=== waiting for healthchecks ===", flush=True)
        ok_a = _wait_for_healthz(f"{host_a_url}/healthz")
        ok_b = _wait_for_healthz(f"{host_b_url}/healthz")
        ok_p = _wait_for_healthz(f"{proxy_url}/healthz")
        if not (ok_a and ok_b and ok_p):
            raise RuntimeError(
                f"healthcheck failed: a={ok_a} b={ok_b} p={ok_p}")
        print(f"host-a OK, host-b OK, proxy OK")

        # 4. Inspect the topology.
        topo_d = _docker_inspect_topology(
            host_a_ctr="w86-host-a",
            host_b_ctr="w86-host-b",
            proxy_ctr="w86-partition-proxy",
            network=f"{PROJECT_NAME}_coordpy_w86_net")
        print()
        print("=== topology ===", flush=True)
        for k, v in topo_d.items():
            print(f"  {k}: {v}")

        topology = MultiHostTopologyV1(
            schema=(
                W86_MULTI_HOST_DISTRIBUTED_V1_SCHEMA_VERSION),
            host_a_label="alpha (container w86-host-a)",
            host_a_base_url=host_a_url,
            host_a_hostname=str(topo_d["host_a_hostname"]),
            host_b_label="beta (container w86-host-b)",
            host_b_base_url=host_b_url,
            host_b_hostname=str(topo_d["host_b_hostname"]),
            proxy_base_url=proxy_url,
            docker_network_id=str(topo_d["network_id"]),
        )

        # 5. Build the client trust root using the same HMAC key
        # the containers know.
        from coordpy.cross_process_distributed_substrate_v1 import (
            TrustRootV1,
            W84_CROSS_PROCESS_DISTRIBUTED_V1_SCHEMA_VERSION,
        )
        import hashlib as _hashlib
        client_trust_root = TrustRootV1(
            schema=W84_CROSS_PROCESS_DISTRIBUTED_V1_SCHEMA_VERSION,
            principal_id="client",
            hmac_key_b64=client_key,
            trust_anchor_cid=_hashlib.sha256(
                b"trust-anchor::client").hexdigest(),
        )

        # 6. Run the bench.
        print()
        print("=== running distributed bench ===", flush=True)
        report = run_multi_host_distributed_bench_v1(
            host_a_base_url=host_a_url,
            host_b_base_url=host_b_url,
            proxy_base_url=proxy_url,
            topology=topology,
            client_trust_root=client_trust_root,
            n_envelopes=int(
                os.environ.get("W86_MH_N_ENVELOPES", "8")),
            n_replays=int(
                os.environ.get("W86_MH_N_REPLAYS", "10")),
            partition_window_seconds=float(os.environ.get(
                "W86_MH_PARTITION_S", "1.5")),
            envelope_partition_seconds=float(os.environ.get(
                "W86_MH_PARTITION_SEND_S", "0.8")),
        )

        # 7. Write report.
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(report.to_dict(), indent=2),
            encoding="utf-8")
        print()
        print(f"=== report -> {out_path} ===", flush=True)
        rd = report.to_dict()
        for k in (
                "mtls_unauthenticated_refused",
                "mtls_bad_signature_refused",
                "cross_host_post_root_match",
                "partition_drops_all_traffic",
                "partition_recovery_seconds",
                "partition_heals_and_recovers",
                "skew_injection_within_tolerance",
                "idempotent_apply_holds",
                "n_idempotent_replays",
                "n_distinct_replay_digests",
                "rtt_host_a_ms", "rtt_host_b_ms",
                "wall_clock_seconds"):
            print(f"  {k}: {rd[k]}")
        print(f"  report_cid: {report.cid()}")
        return 0
    finally:
        # 8. Always tear the stack down. Logs first.
        try:
            print()
            print("=== logs (host-a tail) ===", flush=True)
            _run(["docker", "logs", "--tail", "20",
                  "w86-host-a"], check=False)
            print()
            print("=== logs (host-b tail) ===", flush=True)
            _run(["docker", "logs", "--tail", "20",
                  "w86-host-b"], check=False)
            print()
            print("=== logs (proxy tail) ===", flush=True)
            _run(["docker", "logs", "--tail", "20",
                  "w86-partition-proxy"], check=False)
        except Exception:  # noqa: BLE001
            pass
        print()
        print("=== docker compose down ===", flush=True)
        _run([
            "docker", "compose",
            "-p", PROJECT_NAME,
            "-f", str(COMPOSE_FILE),
            "--env-file", str(ENV_FILE),
            "down", "-v",
        ], check=False)
        if ENV_FILE.exists():
            try:
                ENV_FILE.unlink()
            except OSError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
