"""W88 HumanEval sequential-reflexion bench driver.

Three-arm head-to-head (A0 single-shot, A1 first-pass-among-K
self-consistency, B sequential-reflexion-K=5) on a deterministic
subset of HumanEval.  Reports per-seed pass@1 + strict-improvement
bools + per-task Merkle audit chain.

Backends supported via the ``--backend`` flag:

* ``nim`` (default) — NVIDIA NIM with ``NVIDIA_API_KEY`` (the same
  path W86 used for direct comparability with the W86 negative
  result).
* ``ollama`` — local Ollama daemon (cross-model robustness check;
  no API key required).

Environment variables:

* ``W88_HE_N_PROBLEMS`` (default 30)
* ``W88_HE_N_SEEDS`` (default 3)
* ``W88_HE_MODEL`` (default depends on backend)
* ``W88_HE_OUT_DIR`` (default ``results/w88/humaneval_reflexion``)
* ``W88_HE_BACKEND`` (default ``nim``; overridden by ``--backend``)
* ``NVIDIA_API_KEY`` — required if backend is ``nim``.
* ``COORDPY_OLLAMA_URL`` (default ``http://localhost:11434``) —
  used if backend is ``ollama``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from coordpy.humaneval_real_bench_v1 import (
    load_humaneval_corpus_v1,
)
from coordpy.humaneval_reflexion_bench_v1 import (
    HumanEvalReflexionBenchConfigV1,
    run_humaneval_reflexion_bench_v1,
)


def _make_nim_gen(model_id: str, *, api_key: str,
                  timeout: float = 240.0):
    """Direct NIM HTTPS calls via stdlib; avoids any heavyweight
    SDK dependency. Mirrors the W86 ``NIMFrontierTextRuntimeV1``
    surface but at the lowest level so the bench is self-contained.
    """
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
        for attempt in range(3):
            try:
                with urllib.request.urlopen(
                        req, timeout=float(timeout)) as r:
                    raw = r.read()
                obj = json.loads(raw.decode("utf-8"))
                choices = obj.get("choices") or []
                if not choices:
                    raise RuntimeError(
                        f"NIM returned empty choices: {obj}")
                msg = choices[0].get("message") or {}
                text = str(msg.get("content") or "")
                wall = int((time.time() - t0) * 1000)
                return text, wall
            except (urllib.error.HTTPError,
                    urllib.error.URLError,
                    TimeoutError) as e:
                last_exc = e
                time.sleep(2.0 * (attempt + 1))
            except Exception as e:  # noqa: BLE001
                last_exc = e
                time.sleep(2.0 * (attempt + 1))
        return (
            f"[ERR: {type(last_exc).__name__}: {last_exc}]",
            int((time.time() - t0) * 1000))

    return gen


def _make_ollama_gen(model_id: str, *, base_url: str,
                     timeout: float = 240.0):
    """Local Ollama chat completion via stdlib HTTP.

    Uses the /api/chat endpoint which mirrors OpenAI-style chat.
    """
    url = base_url.rstrip("/") + "/api/chat"

    def gen(prompt: str, max_tokens: int,
            temperature: float) -> tuple[str, int]:
        body = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "temperature": float(temperature),
                "num_predict": int(max_tokens),
            },
        }
        if float(temperature) > 0.0:
            body["options"]["top_p"] = 0.95
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, method="POST",
            headers={"Content-Type": "application/json"})
        t0 = time.time()
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(
                        req, timeout=float(timeout)) as r:
                    raw = r.read()
                obj = json.loads(raw.decode("utf-8"))
                msg = obj.get("message") or {}
                text = str(msg.get("content") or "")
                wall = int((time.time() - t0) * 1000)
                return text, wall
            except (urllib.error.HTTPError,
                    urllib.error.URLError,
                    TimeoutError) as e:
                last_exc = e
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
        description="W88 HumanEval sequential-reflexion bench driver.")
    parser.add_argument(
        "--backend", default=os.environ.get(
            "W88_HE_BACKEND", "nim"),
        choices=("nim", "ollama"))
    parser.add_argument(
        "--model", default=None,
        help="Backend model id (defaults: NIM=meta/llama-3.1-8b-instruct, "
             "Ollama=qwen2.5-coder:7b)")
    parser.add_argument(
        "--n-problems", type=int,
        default=int(os.environ.get("W88_HE_N_PROBLEMS", "30")))
    parser.add_argument(
        "--n-seeds", type=int,
        default=int(os.environ.get("W88_HE_N_SEEDS", "3")))
    parser.add_argument(
        "--out-dir", default=os.environ.get(
            "W88_HE_OUT_DIR",
            str(ROOT / "results" / "w88" / "humaneval_reflexion")))
    parser.add_argument(
        "--max-tokens", type=int, default=768)
    parser.add_argument(
        "--temperature", type=float, default=0.7)
    parser.add_argument(
        "--executor-timeout-s", type=float, default=8.0)
    parser.add_argument(
        "--executor-kill-after-s", type=float, default=12.0)
    args = parser.parse_args()

    backend = args.backend
    if args.model is not None:
        model_id = args.model
    elif backend == "nim":
        model_id = "meta/llama-3.1-8b-instruct"
    elif backend == "ollama":
        model_id = "qwen2.5-coder:7b"
    else:
        raise SystemExit(f"unknown backend: {backend}")
    n_problems = int(args.n_problems)
    n_seeds = int(args.n_seeds)
    seeds = tuple(88_028_001 + i for i in range(n_seeds))

    if backend == "nim":
        api_key = os.environ.get("NVIDIA_API_KEY", "").strip()
        if not api_key:
            raise SystemExit(
                "NIM backend selected but NVIDIA_API_KEY is empty")
        gen = _make_nim_gen(model_id, api_key=api_key)
    elif backend == "ollama":
        base = os.environ.get(
            "COORDPY_OLLAMA_URL", "http://localhost:11434")
        gen = _make_ollama_gen(model_id, base_url=base)
    else:
        raise SystemExit(f"unknown backend: {backend}")

    # Backend-aware out dir suffix so runs do not clobber.
    safe_backend = backend
    safe_model = model_id.replace("/", "_").replace(":", "_")
    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ")
    run_dir = (
        Path(args.out_dir)
        / f"w88_{safe_backend}_{safe_model}_{timestamp}")
    run_dir.mkdir(parents=True, exist_ok=True)

    calls_sidecar_path = run_dir / "humaneval_reflexion_calls.jsonl"
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
            "backend": backend,
            "n_call": int(n_calls),
            "prompt_len": int(len(prompt)),
            "prompt_sha256": __import__(
                "hashlib").sha256(prompt.encode("utf-8")).hexdigest(),
            "response_len": int(len(text)),
            "response_sha256": __import__(
                "hashlib").sha256(text.encode("utf-8")).hexdigest(),
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "wall_ms": int(wall),
            "issued_wall_s": int(time.time() - t0),
            "prompt": prompt,
            "response_text": text,
        }
        calls_sidecar_f.write(
            json.dumps(rec, separators=(",", ":")) + "\n")
        calls_sidecar_f.flush()
        return text, int(wall)

    corpus = load_humaneval_corpus_v1()
    print(f"corpus: {len(corpus)} problems")
    print(
        f"config: backend={backend} model={model_id} "
        f"n_problems={n_problems} seeds={seeds}")

    cfg = HumanEvalReflexionBenchConfigV1(
        n_problems=int(n_problems),
        K_multi_sample=5,
        seeds=seeds,
        sampling_temperature=float(args.temperature),
        max_tokens_per_call=int(args.max_tokens),
        executor_timeout_s=float(args.executor_timeout_s),
        executor_kill_after_s=float(args.executor_kill_after_s),
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

    report = run_humaneval_reflexion_bench_v1(
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

    out_path = run_dir / "humaneval_reflexion_bench_report.json"
    with open(out_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)
    print(f"report -> {out_path}")
    print(f"per-call sidecar -> {calls_sidecar_path}")

    # Also write a "latest" pointer so the verifier can find this
    # run without timestamps.  Store as a path RELATIVE to the
    # pointer's directory so the pointer is portable across
    # checkouts.
    latest_pointer = (
        Path(args.out_dir) / "latest_run.txt")
    latest_pointer.parent.mkdir(parents=True, exist_ok=True)
    latest_pointer.write_text(run_dir.name + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
