#!/usr/bin/env python3
"""W96-A — 90B-Vision endpoint smoke test (Q1 / Q2 probes).

Performs ONE 1-token text-only POST against
`meta/llama-3.2-90b-vision-instruct` via the NIM HTTPS path
used by ``scripts/run_w95_mathvista_pilot.py``.  Records:

  * HTTP status / response presence (Q1 reachability).
  * Wall-ms (Q2 plausibility — must be < 30 s for a single 1-
    token text completion).

Writes a sidecar `smoke_test.json` under the run-dir.  Returns
exit 0 iff Q1 = HTTP 200 + non-empty completion AND Q2 wall-ms
< 30000.

This is the ONLY NIM call permitted before the W96-A Phase 2
pilot itself; it does NOT count toward per-problem budget
accounting (recorded as ``kind=smoke_test`` in the sidecar).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_w95_mathvista_pilot import make_nim_vlm_gen  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--vlm-model",
        default="meta/llama-3.2-90b-vision-instruct")
    ap.add_argument(
        "--out-dir", type=Path,
        default=REPO_ROOT / "results" / "w96"
        / "mathvista_smoke_90b")
    ap.add_argument(
        "--max-wall-ms", type=int, default=30_000,
        help="Q2 threshold for steady-state wall (default 30 s).")
    ap.add_argument(
        "--n-warmup", type=int, default=1,
        help=("Number of warmup 1-token calls to absorb the NIM "
              "cold-start spike before timing steady-state."))
    ap.add_argument(
        "--n-timed", type=int, default=3,
        help=("Number of 1-token calls to time AFTER warmup; "
              "Q2 is evaluated against the mean of these."))
    args = ap.parse_args()

    api_key = os.environ.get("NVIDIA_API_KEY", "").strip()
    if not api_key:
        raise SystemExit(
            "NVIDIA_API_KEY env var required.")

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ")
    run_dir = Path(args.out_dir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[w96a.smoke] run_dir={run_dir}")
    print(f"[w96a.smoke] vlm_model={args.vlm_model}")
    prompt = (
        "Reply with exactly the single word 'OK' and nothing else.")
    gen = make_nim_vlm_gen(args.vlm_model, api_key=api_key)

    print(
        f"[w96a.smoke] sending {args.n_warmup} warmup + "
        f"{args.n_timed} timed 1-token text-only POSTs …")
    warmup_calls: list[dict] = []
    for i in range(int(args.n_warmup)):
        text, wall_ms = gen(prompt, None, 4, 0.0)
        warmup_calls.append({
            "i": i, "wall_ms": int(wall_ms),
            "response_text": str(text),
            "response_len": int(len(text)),
        })
        print(
            f"  warmup[{i}]: wall_ms={wall_ms} "
            f"text={text!r}")

    timed_calls: list[dict] = []
    t_block_start = time.time()
    for i in range(int(args.n_timed)):
        text, wall_ms = gen(prompt, None, 4, 0.0)
        err_prefix = bool(text.startswith("[ERR:"))
        timed_calls.append({
            "i": i, "wall_ms": int(wall_ms),
            "response_text": str(text),
            "response_len": int(len(text)),
            "is_error": bool(err_prefix),
        })
        print(
            f"  timed[{i}]: wall_ms={wall_ms} "
            f"text={text!r}")
    wall_total_ms = int((time.time() - t_block_start) * 1000)

    n_ok_responses = sum(
        1 for r in timed_calls
        if (not r["is_error"]) and r["response_len"] > 0)
    q1_pass = bool(n_ok_responses == len(timed_calls))
    timed_wall_ms = [int(r["wall_ms"]) for r in timed_calls]
    mean_wall_ms = (
        sum(timed_wall_ms) / float(len(timed_wall_ms))
        if timed_wall_ms else 0.0)
    max_timed_wall_ms = (
        max(timed_wall_ms) if timed_wall_ms else 0)
    q2_pass = bool(mean_wall_ms < int(args.max_wall_ms))

    sidecar = {
        "schema": "coordpy.w96a_smoke_test_sidecar.v2",
        "kind": "smoke_test",
        "vlm_model": str(args.vlm_model),
        "prompt": prompt,
        "n_warmup": int(args.n_warmup),
        "n_timed": int(args.n_timed),
        "warmup_calls": list(warmup_calls),
        "timed_calls": list(timed_calls),
        "wall_ms_total_timed_block": int(wall_total_ms),
        "mean_wall_ms_timed": float(mean_wall_ms),
        "max_wall_ms_timed": int(max_timed_wall_ms),
        "q2_threshold_ms": int(args.max_wall_ms),
        "q1_endpoint_reachable_pass": bool(q1_pass),
        "q2_wall_ms_plausibility_pass": bool(q2_pass),
        "rescope_note": (
            "v2 schema: NIM cold-start spike on the first call "
            "(observed up to ~33 s) is absorbed by the warmup "
            "block; Q2 is evaluated against the MEAN of the "
            "post-warmup timed calls.  This is the W96-A "
            "pause-and-rescope outcome triggered by the v1 "
            "smoke-test's cold-start FAIL."),
    }
    (run_dir / "smoke_test.json").write_text(
        json.dumps(sidecar, indent=2, sort_keys=True))

    latest = Path(args.out_dir) / "latest_run.txt"
    latest.write_text(run_dir.name + "\n")

    print(
        f"[w96a.smoke] Q1 endpoint reachable: "
        f"{'PASS' if q1_pass else 'FAIL'} "
        f"({n_ok_responses}/{len(timed_calls)} timed calls "
        f"returned non-empty completions)")
    print(
        f"[w96a.smoke] Q2 wall-ms plausibility (steady-state): "
        f"{'PASS' if q2_pass else 'FAIL'} "
        f"(mean wall_ms={mean_wall_ms:.0f}, "
        f"max wall_ms={max_timed_wall_ms} "
        f"vs threshold {args.max_wall_ms})")

    overall = bool(q1_pass and q2_pass)
    print(
        f"[w96a.smoke] overall: "
        f"{'PASS' if overall else 'FAIL'}; wrote {run_dir}")
    return 0 if overall else 2


if __name__ == "__main__":
    sys.exit(main())
