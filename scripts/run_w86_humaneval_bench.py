"""W86 HumanEval real-task bench driver — Llama-3.1-8B-Instruct via NIM.

Three-arm head-to-head (A0 single-shot, A1 first-pass-among-K
self-consistency, B CoordPy multi-agent with executor-as-critic)
on a deterministic subset of HumanEval. Reports per-seed pass@1
+ strict-improvement bools + per-task Merkle audit chain.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Import from repo, not site-packages.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from coordpy.nim_frontier_text_runtime_v1 import (
    NIMFrontierTextRuntimeV1,
)
from coordpy.humaneval_real_bench_v1 import (
    HumanEvalBenchConfigV1, load_humaneval_corpus_v1,
    run_humaneval_real_bench_v1,
)


def main() -> int:
    n_problems = int(os.environ.get("W86_HE_N_PROBLEMS", "30"))
    n_seeds = int(os.environ.get("W86_HE_N_SEEDS", "3"))
    seeds = tuple(86_028_001 + i for i in range(n_seeds))
    model_id = os.environ.get(
        "W86_HE_NIM_MODEL", "meta/llama-3.1-8b-instruct")
    out_path = os.environ.get(
        "W86_HE_OUT_PATH",
        str(ROOT / "results" / "w86" /
            "humaneval_bench_report.json"))

    runtime = NIMFrontierTextRuntimeV1(
        model_id=model_id, timeout=240.0)

    calls_sidecar_path = out_path.replace(
        ".json", ".calls.jsonl")
    os.makedirs(
        os.path.dirname(calls_sidecar_path), exist_ok=True)
    calls_sidecar_f = open(calls_sidecar_path, "w")
    n_calls = 0

    def gen(prompt: str, max_tokens: int, temperature: float):
        nonlocal n_calls
        n_calls += 1
        t0 = time.time()
        last = None
        for attempt in range(3):
            try:
                cap, text = runtime.generate_capsule(
                    prompt=prompt,
                    max_tokens=int(max_tokens),
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
                    "finish_reason": (
                        cap.response_finish_reason),
                }
                calls_sidecar_f.write(
                    json.dumps(rec, separators=(",", ":"))
                    + "\n")
                calls_sidecar_f.flush()
                return text, int((time.time() - t0) * 1000)
            except Exception as e:  # noqa: BLE001
                last = e
                time.sleep(2 * (attempt + 1))
        return (
            f"[ERR: {type(last).__name__}: {last}]",
            int((time.time() - t0) * 1000))

    corpus = load_humaneval_corpus_v1()
    print(f"corpus: {len(corpus)} problems")
    print(f"config: n_problems={n_problems}, "
          f"seeds={seeds}, model={model_id}")

    cfg = HumanEvalBenchConfigV1(
        n_problems=n_problems,
        K_multi_sample=5,
        seeds=seeds,
        sampling_temperature=0.7,
        max_tokens_per_call=768,
    )

    t0 = time.time()

    def progress(seed: int, p_idx: int, task_id: str) -> None:
        elapsed = time.time() - t0
        rate = n_calls / max(elapsed, 0.001)
        print(
            f"  seed={seed} problem {p_idx+1}/{n_problems} "
            f"(task={task_id}) elapsed={elapsed:.0f}s "
            f"calls={n_calls} rate={rate:.2f}/s",
            flush=True)

    report = run_humaneval_real_bench_v1(
        gen=gen, model_id=model_id, corpus=corpus,
        config=cfg, on_problem_start=progress)
    dt = time.time() - t0

    print()
    print(f"total wall: {dt:.0f}s, NIM calls: {n_calls}")
    print(f"a0_mean_pass@1: {report.a0_mean_pass_at_1:.4f}")
    print(f"a1_mean_pass@1: {report.a1_mean_pass_at_1:.4f}")
    print(f"b_mean_pass@1:  {report.b_mean_pass_at_1:.4f}")
    print(f"B beats A0 per seed: {report.b_beats_a0_per_seed}")
    print(f"B beats A1 per seed: {report.b_beats_a1_per_seed}")
    print(
        f"B strictly > A0 on all seeds: "
        f"{report.b_strictly_beats_a0_on_all_seeds}")
    print(
        f"B strictly > A1 on all seeds: "
        f"{report.b_strictly_beats_a1_on_all_seeds}")
    print(
        f"B mean strictly > A0 mean: "
        f"{report.b_mean_strictly_beats_a0_mean}")
    print(
        f"B mean strictly > A1 mean: "
        f"{report.b_mean_strictly_beats_a1_mean}")
    print(f"bench Merkle root: "
          f"{report.bench_merkle_root[:16]}...")

    calls_sidecar_f.close()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)
    print(f"report -> {out_path}")
    print(f"per-call sidecar -> {calls_sidecar_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
