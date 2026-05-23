"""W90 MBPP sequential-reflexion bench driver — NIM Llama-3.3-70B
default.  Tests whether the W89 70B-HumanEval same-budget
multi-agent superiority claim GENERALISES to a second published
benchmark (MBPP-sanitized).

Same architecture as W88/W89's sequential-reflexion B-pipeline;
same K=5 budget; same A0 / A1 / B arm shape; same audit-chain
discipline.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
import hashlib
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from coordpy.mbpp_reflexion_bench_v1 import (
    MBPPBenchConfigV1, load_mbpp_corpus_v1,
    run_mbpp_reflexion_bench_v1,
)


def _make_nim_gen(model_id: str, *, api_key: str,
                  timeout: float = 240.0):
    url = "https://integrate.api.nvidia.com/v1/chat/completions"

    def gen(prompt: str, max_tokens: int,
            temperature: float) -> tuple[str, int]:
        body = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "stream": False,
        }
        if float(temperature) > 0.0:
            body["top_p"] = 0.95
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, method="POST",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            })
        t0 = time.time()
        last_exc: Exception | None = None
        for attempt in range(6):
            try:
                with urllib.request.urlopen(
                        req, timeout=float(timeout)) as r:
                    raw = r.read()
                obj = json.loads(raw.decode("utf-8"))
                choices = obj.get("choices") or []
                msg = choices[0].get("message") or {}
                text = str(msg.get("content") or "")
                wall = int((time.time() - t0) * 1000)
                return text, wall
            except urllib.error.HTTPError as e:
                last_exc = e
                # Back off harder on 429 (rate limit).
                if int(e.code) == 429:
                    time.sleep(15.0 + 15.0 * attempt)
                else:
                    time.sleep(2.0 * (attempt + 1))
            except Exception as e:  # noqa: BLE001
                last_exc = e
                time.sleep(2.0 * (attempt + 1))
        return (
            f"[ERR: {type(last_exc).__name__}: {last_exc}]",
            int((time.time() - t0) * 1000))

    return gen


def main() -> int:
    parser = argparse.ArgumentParser(
        description="W90 MBPP sequential-reflexion bench driver.")
    parser.add_argument(
        "--model", default=os.environ.get(
            "W90_MBPP_MODEL",
            "meta/llama-3.3-70b-instruct"))
    parser.add_argument(
        "--n-problems", type=int,
        default=int(os.environ.get("W90_MBPP_N_PROBLEMS", "30")))
    parser.add_argument(
        "--n-seeds", type=int,
        default=int(os.environ.get("W90_MBPP_N_SEEDS", "3")))
    parser.add_argument(
        "--out-dir", default=os.environ.get(
            "W90_MBPP_OUT_DIR",
            str(ROOT / "results" / "w90" /
                "mbpp_reflexion")))
    parser.add_argument(
        "--max-tokens", type=int, default=768)
    parser.add_argument(
        "--temperature", type=float, default=0.7)
    args = parser.parse_args()

    api_key = os.environ.get("NVIDIA_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("NVIDIA_API_KEY required")

    model_id = args.model
    n_problems = int(args.n_problems)
    n_seeds = int(args.n_seeds)
    seeds = tuple(90_001 + i for i in range(n_seeds))

    gen = _make_nim_gen(model_id, api_key=api_key)

    safe_model = model_id.replace("/", "_").replace(":", "_")
    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ")
    run_dir = (
        Path(args.out_dir)
        / f"w90_mbpp_nim_{safe_model}_{timestamp}")
    run_dir.mkdir(parents=True, exist_ok=True)

    calls_sidecar_path = run_dir / "mbpp_reflexion_calls.jsonl"
    calls_sidecar_f = open(calls_sidecar_path, "w")
    n_calls = 0

    inner_gen = gen

    def wrapped_gen(prompt: str, max_tokens: int,
                    temperature: float) -> tuple[str, int]:
        nonlocal n_calls
        n_calls += 1
        t0 = time.time()
        text, wall = inner_gen(prompt, int(max_tokens),
                               float(temperature))
        rec = {
            "model_id": model_id,
            "n_call": int(n_calls),
            "prompt_sha256": hashlib.sha256(
                prompt.encode("utf-8")).hexdigest(),
            "response_sha256": hashlib.sha256(
                text.encode("utf-8")).hexdigest(),
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "wall_ms": int(wall),
            "prompt": prompt, "response_text": text,
        }
        calls_sidecar_f.write(
            json.dumps(rec, separators=(",", ":")) + "\n")
        calls_sidecar_f.flush()
        return text, int(wall)

    corpus = load_mbpp_corpus_v1()
    print(f"mbpp corpus: {len(corpus)} problems")
    print(
        f"config: model={model_id} n_problems={n_problems} "
        f"seeds={seeds}")

    cfg = MBPPBenchConfigV1(
        n_problems=int(n_problems),
        K_multi_sample=5,
        seeds=seeds,
        sampling_temperature=float(args.temperature),
        max_tokens_per_call=int(args.max_tokens),
    )

    t0 = time.time()

    def progress(seed: int, p_idx: int, task_id: int) -> None:
        elapsed = time.time() - t0
        rate = n_calls / max(elapsed, 0.001)
        print(
            f"  seed={seed} problem {p_idx+1}/{n_problems} "
            f"(task={task_id}) elapsed={elapsed:.0f}s "
            f"calls={n_calls} rate={rate:.2f}/s",
            flush=True)

    report = run_mbpp_reflexion_bench_v1(
        gen=wrapped_gen, model_id=model_id, corpus=corpus,
        config=cfg, on_problem_start=progress)
    dt = time.time() - t0

    print()
    print(f"total wall: {dt:.0f}s, model calls: {n_calls}")
    print(f"a0_mean_pass@1: {report.a0_mean_pass_at_1:.4f}")
    print(f"a1_mean_pass@1: {report.a1_mean_pass_at_1:.4f}")
    print(f"b_mean_pass@1:  {report.b_mean_pass_at_1:.4f}")
    print(f"B beats A0 per seed: {report.b_beats_a0_per_seed}")
    print(f"B beats A1 per seed: {report.b_beats_a1_per_seed}")
    print(
        f"B mean strictly > A0 mean: "
        f"{report.b_mean_strictly_beats_a0_mean}")
    print(
        f"B mean strictly > A1 mean: "
        f"{report.b_mean_strictly_beats_a1_mean}")
    print(
        f"B mean − A1 mean: "
        f"{report.b_mean_minus_a1_mean_pp:+.2f} pp")
    print(
        f"bench Merkle root: "
        f"{report.bench_merkle_root[:16]}...")

    calls_sidecar_f.close()
    out_path = run_dir / "mbpp_reflexion_bench_report.json"
    with open(out_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)
    print(f"report -> {out_path}")

    latest_pointer = Path(args.out_dir) / "latest_run.txt"
    latest_pointer.parent.mkdir(parents=True, exist_ok=True)
    latest_pointer.write_text(run_dir.name + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
