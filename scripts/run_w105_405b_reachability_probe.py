#!/usr/bin/env python3
"""W105 — Cheap 405B reachability probe (sub-second NIM).

Independent of the main W105 Phase 3 run.  Re-probes
``meta/llama-3.1-405b-instruct`` reachability on NIM and
records the result in
``results/w105/405b_reachability_probe/<run_id>/probe.json``.

Either outcome (HTTP 200 reachable / HTTP 404 still unreachable
/ other) does NOT change the W105 core matrix unless the
RUNBOOK is explicitly re-locked.

Usage::

    export NVIDIA_API_KEY=...
    python scripts/run_w105_405b_reachability_probe.py
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

NIM_CHAT_URL: str = (
    "https://integrate.api.nvidia.com/v1/chat/completions")

W105_405B_TARGET: str = "meta/llama-3.1-405b-instruct"


def _probe(*, model: str, max_seconds: float) -> dict:
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        return {
            "status": "no_api_key",
            "model": str(model),
            "ts_utc": _dt.datetime.now(
                _dt.timezone.utc).isoformat(),
        }
    body = {
        "model": str(model),
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 4,
        "temperature": 0.0,
        "stream": False,
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        NIM_CHAT_URL, data=data,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}",
        }, method="POST")
    t0 = time.time()
    try:
        with urllib.request.urlopen(
                req, timeout=float(max_seconds)) as r:
            raw = r.read()
        wall_ms = int((time.time() - t0) * 1000)
        return {
            "status": "reachable",
            "http_status": 200,
            "model": str(model),
            "wall_ms": int(wall_ms),
            "ts_utc": _dt.datetime.now(
                _dt.timezone.utc).isoformat(),
            "response_len": int(len(raw)),
        }
    except urllib.error.HTTPError as e:
        return {
            "status": "http_error",
            "http_status": int(e.code),
            "reason": str(e.reason),
            "model": str(model),
            "wall_ms": int((time.time() - t0) * 1000),
            "ts_utc": _dt.datetime.now(
                _dt.timezone.utc).isoformat(),
        }
    except Exception as e:  # noqa: BLE001
        return {
            "status": "exception",
            "exc_type": type(e).__name__,
            "exc_msg": str(e),
            "model": str(model),
            "wall_ms": int((time.time() - t0) * 1000),
            "ts_utc": _dt.datetime.now(
                _dt.timezone.utc).isoformat(),
        }


def main() -> int:
    ap = argparse.ArgumentParser(description=(
        "W105 cheap 405B reachability probe"))
    ap.add_argument(
        "--model", default=W105_405B_TARGET,
        help=(
            "Target model id (default = W105 405B target "
            f"{W105_405B_TARGET})."))
    ap.add_argument(
        "--max-seconds", type=float, default=20.0,
        help="Probe timeout in seconds")
    ap.add_argument(
        "--out-root", default=str(
            ROOT / "results" / "w105"
            / "405b_reachability_probe"),
        help="Output root")
    args = ap.parse_args()
    run_id = _dt.datetime.now(
        _dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_root) / f"probe_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  W105 405B reachability probe: {args.model}")
    result = _probe(
        model=str(args.model),
        max_seconds=float(args.max_seconds))
    result["schema"] = "coordpy.w105_405b_reachability_probe.v1"
    with open(out_dir / "probe.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    latest = Path(args.out_root) / "latest_run.txt"
    with open(latest, "w") as f:
        f.write(out_dir.name + "\n")
    print(f"  result: {json.dumps(result, indent=2)}")
    print(f"  out_dir: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
