"""W85 GSM8K real-task bench driver — Llama-3.1-8B-Instruct via NIM.

Runs the three-arm head-to-head (A0 single-shot CoT, A1 self-
consistency K=5, B CoordPy multi-agent K=5) on a deterministic
subset of GSM8K test problems. Reports per-seed accuracy, strict-
improvement bools, and a Merkle root over per-task capsules.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Ensure we import from the repo, not any site-packages
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from coordpy.nim_frontier_text_runtime_v1 import NIMFrontierTextRuntimeV1
from coordpy.gsm8k_real_bench_v1 import (
    load_gsm8k_test_corpus_v1, run_gsm8k_real_bench_v1,
    GSM8KBenchConfigV1,
)


def main() -> int:
    n_problems = int(os.environ.get("W85_N_PROBLEMS", "20"))
    n_seeds = int(os.environ.get("W85_N_SEEDS", "3"))
    seeds = tuple(85_001 + i for i in range(n_seeds))
    model_id = os.environ.get(
        "W85_NIM_MODEL", "meta/llama-3.1-8b-instruct")
    out_path = os.environ.get(
        "W85_OUT_PATH",
        str(ROOT / "results" / "w85" / "gsm8k_bench_report.json"))

    runtime = NIMFrontierTextRuntimeV1(
        model_id=model_id, timeout=240.0)

    n_calls = 0
    calls_sidecar_path = out_path.replace(
        ".json", ".calls.jsonl")
    os.makedirs(os.path.dirname(calls_sidecar_path), exist_ok=True)
    calls_sidecar_f = open(calls_sidecar_path, "w")

    def gen(prompt: str, max_tokens: int, temperature: float):
        nonlocal n_calls
        n_calls += 1
        t0 = time.time()
        # Up to 3 retries on 429/transient
        last = None
        for attempt in range(3):
            try:
                cap, text = runtime.generate_capsule(
                    prompt=prompt, max_tokens=int(max_tokens),
                    temperature=float(temperature))
                rec = {
                    "model_id": model_id,
                    "n_call": n_calls,
                    "prompt_cid": cap.prompt_cid,
                    "prompt": prompt,
                    "prompt_tokens": cap.prompt_tokens,
                    "response_cid": cap.response_cid,
                    "response_text": text,
                    "temperature": float(temperature),
                    "max_tokens": int(max_tokens),
                    "wall_ms": int((time.time() - t0) * 1000),
                    "finish_reason": cap.response_finish_reason,
                }
                calls_sidecar_f.write(
                    json.dumps(rec, separators=(",", ":")) + "\n")
                calls_sidecar_f.flush()
                return text, int((time.time() - t0) * 1000)
            except Exception as e:  # noqa: BLE001
                last = e
                time.sleep(2 * (attempt + 1))
        return f"[ERR: {type(last).__name__}: {last}]", \
            int((time.time() - t0) * 1000)

    corpus = load_gsm8k_test_corpus_v1()
    print(f"corpus: {len(corpus)} problems")
    print(f"config: n={n_problems}, seeds={seeds}, model={model_id}")

    cfg = GSM8KBenchConfigV1(
        n_problems=n_problems,
        K_multi_sample=5,
        seeds=seeds,
        sampling_temperature=0.7,
        max_tokens_per_call=384,
    )

    t0 = time.time()

    def progress(s, p, o):
        elapsed = time.time() - t0
        rate = n_calls / max(elapsed, 0.001)
        print(
            f"  seed={s} problem {p+1}/{n_problems} (corpus idx={o}) "
            f"elapsed={elapsed:.0f}s calls={n_calls} "
            f"rate={rate:.2f}/s",
            flush=True)

    report = run_gsm8k_real_bench_v1(
        gen=gen, model_id=model_id, corpus=corpus,
        config=cfg, on_problem_start=progress)
    dt = time.time() - t0

    print(f"\ntotal wall: {dt:.0f}s, NIM calls: {n_calls}")
    print(f"a0_mean_accuracy: {report.a0_mean_accuracy:.4f}")
    print(f"a1_mean_accuracy: {report.a1_mean_accuracy:.4f}")
    print(f"b_mean_accuracy:  {report.b_mean_accuracy:.4f}")
    print(f"B beats A0 per seed: {report.b_beats_a0_per_seed}")
    print(f"B beats A1 per seed: {report.b_beats_a1_per_seed}")
    print(f"B strictly > A0 on all seeds: "
          f"{report.b_strictly_beats_a0_on_all_seeds}")
    print(f"B strictly > A1 on all seeds: "
          f"{report.b_strictly_beats_a1_on_all_seeds}")
    print(f"bench Merkle root: {report.bench_merkle_root[:16]}")

    calls_sidecar_f.close()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)
    print(f"report written to {out_path}")
    print(f"per-call sidecar at {calls_sidecar_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
