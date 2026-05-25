#!/usr/bin/env python3
"""W101 — MBPP+ cheap-pilot driver.

CONDITIONAL ON W101 PREFLIGHT P1 + P2 PASS at re-run after the
operator fetches the EvalPlus MBPP+ release artifact and records
its SHA pin (see `docs/RUNBOOK_W101.md` for the canonical fetch
command).

Runs the W89 sequential-reflexion B-pipeline + A0 + A1 baselines
against MBPP+ at 1 seed × 30 problems × K=5 ≈ 330 NIM calls at
the target model (default `meta/llama-3.3-70b-instruct`).
Evaluates the pre-committed 9 Phase 2 gates + MLB-1 + MLB-2
mechanism-load-bearingness sub-gates.

Requires `NVIDIA_API_KEY` in the environment.

Usage::

    export NVIDIA_API_KEY=...
    python scripts/run_w101_mbpp_plus_pilot.py \\
        --model meta/llama-3.3-70b-instruct \\
        --n-problems 30 --seed 101001
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from coordpy.mbpp_plus_loader_v1 import (  # noqa: E402
    load_mbpp_plus_corpus_v1,
)
from coordpy.mbpp_plus_reflexion_bench_v1 import (  # noqa: E402
    MbppPlusBenchConfigV1,
    run_mbpp_plus_reflexion_bench_v1,
    select_mbpp_plus_subset_v1,
)


NIM_CHAT_URL: str = (
    "https://integrate.api.nvidia.com/v1/chat/completions")


def _build_nim_gen(
        *,
        model: str,
        max_retries: int = 6,
        sidecar_writer=None,
):
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise SystemExit(
            "NVIDIA_API_KEY not set; W101 cheap pilot requires "
            "an authorised NIM endpoint.")

    def _gen(prompt: str, max_tokens: int,
              temperature: float) -> tuple[str, int]:
        body = {
            "model": str(model),
            "messages": [
                {"role": "user", "content": str(prompt)},
            ],
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "stream": False,
        }
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            NIM_CHAT_URL,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST")
        t0 = time.time()
        last_err: Exception | None = None
        for attempt in range(int(max_retries)):
            try:
                with urllib.request.urlopen(
                        req, timeout=180.0) as r:
                    raw = r.read()
                wall_ms = int((time.time() - t0) * 1000)
                payload = json.loads(
                    raw.decode("utf-8", errors="replace"))
                text = ""
                choices = payload.get("choices") or []
                if choices:
                    msg = (
                        choices[0].get("message") or {})
                    text = str(msg.get("content") or "")
                if sidecar_writer is not None:
                    sidecar_writer({
                        "model_id": str(model),
                        "backend": "nim",
                        "prompt_len": int(len(prompt)),
                        "prompt_sha256": hashlib.sha256(
                            prompt.encode("utf-8")
                        ).hexdigest(),
                        "response_len": int(len(text)),
                        "response_sha256": hashlib.sha256(
                            text.encode("utf-8")
                        ).hexdigest(),
                        "temperature": float(temperature),
                        "max_tokens": int(max_tokens),
                        "wall_ms": int(wall_ms),
                        "prompt": str(prompt),
                        "response_text": str(text),
                    })
                return str(text), int(wall_ms)
            except urllib.error.HTTPError as e:
                last_err = e
                if e.code == 429:
                    time.sleep(
                        min(2 ** attempt, 60.0))
                    continue
                raise
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(min(2 ** attempt, 30.0))
        raise RuntimeError(
            f"NIM call failed after {max_retries} attempts: "
            f"{last_err}")
    return _gen


def main() -> int:
    ap = argparse.ArgumentParser(description=(
        "W101 MBPP+ cheap-pilot driver"))
    ap.add_argument(
        "--model",
        default="meta/llama-3.3-70b-instruct",
        help="Target NIM model id (default 70B)")
    ap.add_argument(
        "--n-problems", type=int, default=30,
        help="Number of problems per seed (default 30)")
    ap.add_argument(
        "--seed", type=int, default=101_001,
        help="Seed (single-seed cheap pilot)")
    ap.add_argument(
        "--mbpp-plus-cache", default=None,
        help="MBPP+ JSONL cache path override")
    ap.add_argument(
        "--out-dir",
        default=str(
            ROOT / "results" / "w101"
            / "mbpp_plus_pilot"),
        help="Output root")
    ap.add_argument(
        "--dry-run", action="store_true",
        help=(
            "Do NOT actually call NIM; just validate the bench "
            "config + corpus subset selection."))
    args = ap.parse_args()

    print(f"  loading MBPP+ corpus ...")
    corpus = load_mbpp_plus_corpus_v1(
        cache_path=args.mbpp_plus_cache)
    print(f"  corpus = {len(corpus)} problems")
    subset = select_mbpp_plus_subset_v1(
        corpus=corpus,
        n_problems=int(args.n_problems),
        seed=int(args.seed))
    print(
        f"  slice seed={args.seed} "
        f"n_problems={len(subset)}")
    print(
        f"  first task_id = {subset[0].task_id}, "
        f"last = {subset[-1].task_id}")
    if args.dry_run:
        print("  --dry-run: stopping before any NIM call")
        return 0
    run_id = _dt.datetime.utcnow().strftime(
        "%Y%m%dT%H%M%SZ")
    safe_model = (
        str(args.model)
        .replace("/", "_")
        .replace("-", "-"))
    out_dir = (
        Path(args.out_dir)
        / f"w101_mbpp_plus_pilot_{safe_model}_{run_id}")
    out_dir.mkdir(parents=True, exist_ok=True)
    sidecar_path = (
        out_dir / "mbpp_plus_reflexion_calls.jsonl")
    sidecar_f = open(sidecar_path, "w")

    def sidecar_writer(rec):
        sidecar_f.write(
            json.dumps(rec, separators=(",", ":")) + "\n")

    print(f"  output: {out_dir}")
    gen = _build_nim_gen(
        model=str(args.model),
        sidecar_writer=sidecar_writer)
    cfg = MbppPlusBenchConfigV1(
        n_problems=int(args.n_problems),
        K_multi_sample=5,
        seeds=(int(args.seed),),
        sampling_temperature=0.7,
        max_tokens_per_call=768,
        executor_mode="base_and_plus",
    )
    print(f"  bench config = {cfg}")
    t0 = time.time()
    report = run_mbpp_plus_reflexion_bench_v1(
        gen=gen,
        model_id=str(args.model),
        corpus=corpus,
        config=cfg,
        on_problem_start=lambda s, i, t: print(
            f"  seed={s} p_idx={i+1}/{cfg.n_problems} "
            f"task_id={t}"))
    sidecar_f.close()
    wall_s = float(time.time() - t0)
    rep = report.to_dict()
    rep["wall_s"] = float(round(wall_s, 2))
    rep_path = (
        out_dir / "mbpp_plus_reflexion_bench_report.json")
    with open(rep_path, "w") as f:
        json.dump(rep, f, indent=2, default=str)
    latest = out_dir.parent / "latest_run.txt"
    with open(latest, "w") as f:
        f.write(out_dir.name + "\n")
    print()
    print(
        f"  WALL: {wall_s:.1f} s; "
        f"A0={report.a0_mean_pass_at_1*100:.2f}% "
        f"A1={report.a1_mean_pass_at_1*100:.2f}% "
        f"B={report.b_mean_pass_at_1*100:.2f}% "
        f"B-A1={report.b_mean_minus_a1_mean_pp:+.2f}pp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
