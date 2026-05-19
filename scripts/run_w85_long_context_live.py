"""W85 long-context live bench driver — Llama-3.1-8B-Instruct via NIM.

Runs the three-arm needle-in-haystack bench (A_FULL, A_BOUNDED_V3,
B_COMPOSED) on a live frontier-class model at horizons
{8k, 32k, 128k} characters. Reports per-horizon success and the
32k strict-beat verdict.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from coordpy.nim_frontier_text_runtime_v1 import NIMFrontierTextRuntimeV1
from coordpy.long_context_live_bench_v1 import (
    run_long_context_live_bench_v1,
)


def main() -> int:
    n_per_horizon = int(os.environ.get("W85_LC_N_PER_HORIZON", "3"))
    horizons = os.environ.get(
        "W85_LC_HORIZONS", "8000,32000,128000")
    horizons_chars = tuple(
        int(x) for x in horizons.split(",") if x)
    model_id = os.environ.get(
        "W85_LC_MODEL", "meta/llama-3.1-8b-instruct")
    out_path = os.environ.get(
        "W85_LC_OUT",
        str(ROOT / "results" / "w85" /
            "long_context_live_report.json"))

    runtime = NIMFrontierTextRuntimeV1(
        model_id=model_id, timeout=600.0)
    n_calls = 0
    calls_sidecar_path = out_path.replace(
        ".json", ".calls.jsonl")
    os.makedirs(os.path.dirname(calls_sidecar_path), exist_ok=True)
    calls_sidecar_f = open(calls_sidecar_path, "w")

    def gen(prompt: str, max_tokens: int, temperature: float):
        nonlocal n_calls
        n_calls += 1
        t0 = time.time()
        # Retry on transient
        last = None
        for attempt in range(3):
            try:
                cap, text = runtime.generate_capsule(
                    prompt=prompt, max_tokens=int(max_tokens),
                    temperature=float(temperature))
                # Persist (prompt_sha, prompt_chars, response_text,
                # ptokens, wall_ms) so a third party can verify
                # CIDs without re-calling the model
                rec = {
                    "model_id": model_id,
                    "n_call": n_calls,
                    "prompt_cid": cap.prompt_cid,
                    "prompt_chars": len(prompt),
                    "prompt_tokens": cap.prompt_tokens,
                    "response_cid": cap.response_cid,
                    "response_text": text,
                    "temperature": float(temperature),
                    "max_tokens": int(max_tokens),
                    "wall_ms": int((time.time()-t0)*1000),
                    "finish_reason": cap.response_finish_reason,
                }
                calls_sidecar_f.write(
                    json.dumps(rec, separators=(",", ":")) + "\n")
                calls_sidecar_f.flush()
                return text, int((time.time()-t0)*1000), \
                    cap.prompt_tokens
            except Exception as e:  # noqa: BLE001
                last = e
                time.sleep(3 * (attempt + 1))
        return f"[ERR: {type(last).__name__}: {last}]", \
            int((time.time()-t0)*1000), 0

    t0 = time.time()
    print(f"model: {model_id}")
    print(f"horizons_chars: {horizons_chars}")
    print(f"n_per_horizon: {n_per_horizon}")

    def on_prompt_start(pid):
        elapsed = time.time() - t0
        print(f"  prompt={pid} elapsed={elapsed:.0f}s "
              f"calls={n_calls}", flush=True)

    report = run_long_context_live_bench_v1(
        gen=gen, model_id=model_id,
        horizons_chars=horizons_chars,
        n_per_horizon=n_per_horizon,
        max_tokens=64,
        on_prompt_start=on_prompt_start)

    dt = time.time() - t0
    print(f"\ntotal wall: {dt:.0f}s, NIM calls: {n_calls}")
    for p in report.per_horizon:
        print(f"  horizon={p.horizon_chars}: "
              f"FULL={p.a_full_success_rate:.2f} "
              f"BOUNDED_V3={p.a_bounded_v3_success_rate:.2f} "
              f"COMPOSED={p.b_composed_success_rate:.2f} "
              f"composed>bounded={p.composed_strictly_beats_bounded}")
    print(f"composed_strictly_beats_bounded_at_32k: "
          f"{report.composed_strictly_beats_bounded_at_32k}")
    print(f"composed_strictly_beats_bounded_at_every_horizon: "
          f"{report.composed_strictly_beats_bounded_at_every_horizon}")
    print(f"Merkle root: {report.bench_merkle_root[:16]}")

    calls_sidecar_f.close()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)
    print(f"report written to {out_path}")
    print(f"per-call sidecar at {calls_sidecar_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
