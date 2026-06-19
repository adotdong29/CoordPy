"""W88 cross-modal code bench driver — VLM + code-LM via NIM.

Three-arm head-to-head on the HumanEval-Visual corpus:

* ``A0_text`` — text-only single-shot, no image access
* ``A1_vlm`` — single-agent VLM, K=5 first-pass-among-K
* ``B_cross`` — VLM-extract → code-LM-reflexion, K=5 budget

Reports per-seed pass@1 + strict-improvement bools + per-task
content-addressed audit chain.

Environment variables:

* ``W88_CM_VLM`` (default ``meta/llama-3.2-11b-vision-instruct``)
* ``W88_CM_CODE`` (default ``meta/llama-3.1-8b-instruct``)
* ``W88_CM_N_PROBLEMS`` (default 12)
* ``W88_CM_N_SEEDS`` (default 3)
* ``W88_CM_OUT_DIR`` (default ``results/w88/cross_modal_code``)
* ``NVIDIA_API_KEY`` — required.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
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
from coordpy.cross_modal_code_bench_v1 import (
    CrossModalCodeBenchConfigV1,
    run_cross_modal_code_bench_v1,
)


def _make_nim_text_gen(model_id: str, *, api_key: str,
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
        for attempt in range(3):
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
            except Exception as e:  # noqa: BLE001
                last_exc = e
                time.sleep(2.0 * (attempt + 1))
        return (
            f"[ERR: {type(last_exc).__name__}: {last_exc}]",
            int((time.time() - t0) * 1000))

    return gen


def _make_nim_vlm_gen(model_id: str, *, api_key: str,
                      timeout: float = 240.0):
    url = "https://integrate.api.nvidia.com/v1/chat/completions"

    def gen(prompt: str, image_bytes: bytes | None,
            max_tokens: int,
            temperature: float) -> tuple[str, int]:
        if image_bytes is not None:
            img_b64 = base64.b64encode(
                image_bytes).decode("ascii")
            content = [
                {"type": "text", "text": prompt},
                {"type": "image_url",
                 "image_url": {
                     "url": f"data:image/png;base64,{img_b64}"}},
            ]
        else:
            content = prompt
        body = {
            "model": model_id,
            "messages": [{"role": "user", "content": content}],
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
                msg = choices[0].get("message") or {}
                text = str(msg.get("content") or "")
                wall = int((time.time() - t0) * 1000)
                return text, wall
            except Exception as e:  # noqa: BLE001
                last_exc = e
                time.sleep(2.0 * (attempt + 1))
        return (
            f"[ERR: {type(last_exc).__name__}: {last_exc}]",
            int((time.time() - t0) * 1000))

    return gen


def main() -> int:
    parser = argparse.ArgumentParser(
        description=("W88 cross-modal code bench driver "
                     "(NIM VLM + NIM code-LM)."))
    parser.add_argument(
        "--vlm-model", default=os.environ.get(
            "W88_CM_VLM",
            "meta/llama-3.2-11b-vision-instruct"))
    parser.add_argument(
        "--code-model", default=os.environ.get(
            "W88_CM_CODE", "meta/llama-3.1-8b-instruct"))
    parser.add_argument(
        "--n-problems", type=int,
        default=int(os.environ.get("W88_CM_N_PROBLEMS", "12")))
    parser.add_argument(
        "--n-seeds", type=int,
        default=int(os.environ.get("W88_CM_N_SEEDS", "3")))
    parser.add_argument(
        "--out-dir", default=os.environ.get(
            "W88_CM_OUT_DIR",
            str(ROOT / "results" / "w88" / "cross_modal_code")))
    parser.add_argument(
        "--max-tokens", type=int, default=768)
    parser.add_argument(
        "--temperature", type=float, default=0.7)
    parser.add_argument(
        "--min-doctest-lines", type=int, default=2)
    parser.add_argument(
        "--strip-mode", default="doctest_only",
        choices=("doctest_only", "all_docstring"),
        help=(
            "Corpus synthesis stripping mode: 'doctest_only' "
            "(strip just the >>> lines; keep prose) or "
            "'all_docstring' (replace the entire docstring "
            "with a 'See image' stub)."))
    args = parser.parse_args()

    api_key = os.environ.get("NVIDIA_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("NVIDIA_API_KEY is empty")

    vlm_model_id = args.vlm_model
    code_model_id = args.code_model
    n_problems = int(args.n_problems)
    n_seeds = int(args.n_seeds)
    seeds = tuple(88_046_001 + i for i in range(n_seeds))

    text_gen = _make_nim_text_gen(
        code_model_id, api_key=api_key)
    vlm_gen = _make_nim_vlm_gen(
        vlm_model_id, api_key=api_key)

    safe_vlm = vlm_model_id.replace("/", "_").replace(":", "_")
    safe_code = code_model_id.replace("/", "_").replace(":", "_")
    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ")
    run_dir = (
        Path(args.out_dir)
        / f"w88_xm_{args.strip_mode}_"
          f"{safe_vlm}__{safe_code}_{timestamp}")
    run_dir.mkdir(parents=True, exist_ok=True)

    text_sidecar_path = run_dir / "text_calls.jsonl"
    vlm_sidecar_path = run_dir / "vlm_calls.jsonl"
    text_sidecar_f = open(text_sidecar_path, "w")
    vlm_sidecar_f = open(vlm_sidecar_path, "w")
    n_text_calls = 0
    n_vlm_calls = 0

    inner_text = text_gen
    inner_vlm = vlm_gen

    def wrapped_text(prompt, max_tokens, temperature):
        nonlocal n_text_calls
        n_text_calls += 1
        text, wall = inner_text(
            prompt, int(max_tokens), float(temperature))
        rec = {
            "model_id": code_model_id,
            "n_call": int(n_text_calls),
            "prompt_sha256": hashlib.sha256(
                prompt.encode("utf-8")).hexdigest(),
            "response_sha256": hashlib.sha256(
                text.encode("utf-8")).hexdigest(),
            "temperature": float(temperature),
            "wall_ms": int(wall),
            "prompt": prompt,
            "response_text": text,
        }
        text_sidecar_f.write(
            json.dumps(rec, separators=(",", ":")) + "\n")
        text_sidecar_f.flush()
        return text, int(wall)

    def wrapped_vlm(prompt, image_bytes, max_tokens,
                    temperature):
        nonlocal n_vlm_calls
        n_vlm_calls += 1
        text, wall = inner_vlm(
            prompt, image_bytes,
            int(max_tokens), float(temperature))
        img_cid = (
            hashlib.sha256(image_bytes).hexdigest()
            if image_bytes is not None else "")
        rec = {
            "model_id": vlm_model_id,
            "n_call": int(n_vlm_calls),
            "prompt_sha256": hashlib.sha256(
                prompt.encode("utf-8")).hexdigest(),
            "image_cid": img_cid,
            "image_bytes_len": (
                len(image_bytes)
                if image_bytes is not None else 0),
            "response_sha256": hashlib.sha256(
                text.encode("utf-8")).hexdigest(),
            "temperature": float(temperature),
            "wall_ms": int(wall),
            "prompt": prompt,
            "response_text": text,
        }
        vlm_sidecar_f.write(
            json.dumps(rec, separators=(",", ":")) + "\n")
        vlm_sidecar_f.flush()
        return text, int(wall)

    corpus = load_humaneval_corpus_v1()
    print(f"humaneval corpus: {len(corpus)} problems")
    print(
        f"config: vlm={vlm_model_id} code={code_model_id} "
        f"n_problems={n_problems} seeds={seeds} K=5")

    cfg = CrossModalCodeBenchConfigV1(
        n_problems=int(n_problems),
        K_multi_sample=5,
        seeds=seeds,
        sampling_temperature=float(args.temperature),
        max_tokens_per_call=int(args.max_tokens),
        min_doctest_lines=int(args.min_doctest_lines),
        strip_mode=str(args.strip_mode),
    )

    t0 = time.time()

    def progress(seed: int, p_idx: int, task_id: str) -> None:
        elapsed = time.time() - t0
        total_calls = n_text_calls + n_vlm_calls
        print(
            f"  seed={seed} problem {p_idx+1}/{n_problems} "
            f"(task={task_id}) elapsed={elapsed:.0f}s "
            f"text={n_text_calls} vlm={n_vlm_calls} "
            f"total={total_calls}",
            flush=True)

    report, cross_corpus = run_cross_modal_code_bench_v1(
        text_gen=wrapped_text, vlm_gen=wrapped_vlm,
        vlm_model_id=vlm_model_id,
        code_model_id=code_model_id,
        corpus=corpus, config=cfg,
        on_problem_start=progress)
    dt = time.time() - t0

    print()
    print(
        f"total wall: {dt:.0f}s, "
        f"text calls: {n_text_calls}, vlm calls: {n_vlm_calls}")
    print(
        f"A0_text  mean pass@1: "
        f"{report.a0_text_mean_pass_at_1:.4f}")
    print(
        f"A1_vlm   mean pass@1: "
        f"{report.a1_vlm_mean_pass_at_1:.4f}")
    print(
        f"B_cross  mean pass@1: "
        f"{report.b_cross_mean_pass_at_1:.4f}")
    print(
        f"B_cross beats A0_text per seed: "
        f"{report.b_cross_beats_a0_text_per_seed}")
    print(
        f"B_cross beats A1_vlm per seed: "
        f"{report.b_cross_beats_a1_vlm_per_seed}")
    print(
        f"B_cross mean strictly > A0_text mean: "
        f"{report.b_cross_mean_strictly_beats_a0_text_mean}")
    print(
        f"B_cross mean strictly > A1_vlm mean: "
        f"{report.b_cross_mean_strictly_beats_a1_vlm_mean}")
    print(
        f"B_cross − A0_text: "
        f"{report.b_cross_mean_minus_a0_text_mean_pp:+.2f} pp")
    print(
        f"B_cross − A1_vlm: "
        f"{report.b_cross_mean_minus_a1_vlm_mean_pp:+.2f} pp")
    print(
        f"bench Merkle root: "
        f"{report.bench_merkle_root[:16]}...")

    text_sidecar_f.close()
    vlm_sidecar_f.close()

    out_path = run_dir / "cross_modal_code_bench_report.json"
    with open(out_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)
    print(f"report -> {out_path}")
    print(f"text sidecar -> {text_sidecar_path}")
    print(f"vlm sidecar -> {vlm_sidecar_path}")

    latest_pointer = (
        Path(args.out_dir) / "latest_run.txt")
    latest_pointer.parent.mkdir(parents=True, exist_ok=True)
    latest_pointer.write_text(run_dir.name + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
