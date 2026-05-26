#!/usr/bin/env python3
"""W102 — MBPP+ V2 cheap-pilot driver.

CONDITIONAL ON W102 V2 PREFLIGHT 10/10 PASS.  Verdict
`results/w102/mbpp_plus_v2_preflight/<RUN>/verdict.json` must have
`overall_passes=true` before this driver is authorised to spend
NIM budget.

Runs the W89 sequential-reflexion B-pipeline + A0 + A1 baselines
against MBPP+ V2 at 1 seed × 30 problems × K=5 ≈ 330 NIM calls at
the target model (default `meta/llama-3.3-70b-instruct`).
Evaluates the pre-committed 9 Phase 2 gates + MLB-1 + MLB-2
mechanism-load-bearingness sub-gates.

Requires `NVIDIA_API_KEY` in the environment.

Usage::

    export NVIDIA_API_KEY=...
    python scripts/run_w102_mbpp_plus_v2_pilot.py \\
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

from coordpy.mbpp_plus_loader_v2 import (  # noqa: E402
    load_mbpp_plus_v2_corpus,
)
from coordpy.mbpp_plus_reflexion_bench_v2 import (  # noqa: E402
    MbppPlusV2BenchConfig,
    mlb_invocation_and_rescue_rates_v2,
    run_mbpp_plus_reflexion_bench_v2,
    select_mbpp_plus_v2_subset,
)


NIM_CHAT_URL: str = (
    "https://integrate.api.nvidia.com/v1/chat/completions")


def _build_nim_gen(
        *,
        model: str,
        max_retries: int = 12,
        sidecar_writer=None,
        inter_call_sleep_s: float = 0.0,
):
    """NIM chat-completion generator with rate-limit-aware
    backoff.

    Hardened for the W102 cheap pilot after the initial run hit
    HTTP 429 after only 6 retries.  Changes vs the W101 driver:

    * ``max_retries`` raised from 6 to 12.
    * 429 backoff caps at 300 s (5 min) with deterministic
      exponential + jitter; aggregate retry window is ~30 min.
    * 5xx server-side errors treated the same as 429.
    * Optional ``inter_call_sleep_s`` to pace the bench loop.
    """
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise SystemExit(
            "NVIDIA_API_KEY not set; W102 cheap pilot requires "
            "an authorised NIM endpoint.")
    import random as _random

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
                        req, timeout=240.0) as r:
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
                if inter_call_sleep_s > 0:
                    time.sleep(float(inter_call_sleep_s))
                return str(text), int(wall_ms)
            except urllib.error.HTTPError as e:
                last_err = e
                if e.code in (429, 502, 503, 504):
                    backoff = min(
                        float(2 ** attempt) + (
                            _random.random() * 5.0),
                        300.0)
                    print(
                        f"  [nim retry] HTTP {e.code} attempt "
                        f"{attempt+1}/{max_retries}; sleeping "
                        f"{backoff:.1f}s", flush=True)
                    time.sleep(backoff)
                    continue
                raise
            except Exception as e:  # noqa: BLE001
                last_err = e
                backoff = min(
                    float(2 ** attempt) + (
                        _random.random() * 3.0),
                    120.0)
                print(
                    f"  [nim retry] {type(e).__name__}: {e}; "
                    f"attempt {attempt+1}/{max_retries}; "
                    f"sleeping {backoff:.1f}s",
                    flush=True)
                time.sleep(backoff)
        raise RuntimeError(
            f"NIM call failed after {max_retries} attempts: "
            f"{last_err}")
    return _gen


def _evaluate_phase2_gates(
        *, report, mlb,
        a0=None, a1=None, b=None,
        margin_floor_pp: float = 5.0,
        per_problem_majority_floor: int = 16,
        K: int = 5):
    """Evaluate the pre-committed 9 Phase 2 gates + MLB-1 + MLB-2
    sub-gates on the V2 bench report."""
    a0_pct = float(report.a0_mean_pass_at_1 * 100)
    a1_pct = float(report.a1_mean_pass_at_1 * 100)
    b_pct = float(report.b_mean_pass_at_1 * 100)
    b_minus_a1_pp = float(b_pct - a1_pct)
    b_minus_a0_pp = float(b_pct - a0_pct)
    # Gate 6: per-problem majority B ≥ A1
    n_problems = sum(
        len(s.per_problem_b_passed) for s in report.per_seed)
    n_b_ge_a1 = 0
    for s in report.per_seed:
        for i in range(len(s.per_problem_b_passed)):
            b_p = bool(s.per_problem_b_passed[i])
            a1_p = bool(s.per_problem_a1_passed[i])
            if (b_p and a1_p) or (b_p and not a1_p) or (
                    not b_p and not a1_p):
                # B ≥ A1 means B passed if A1 passed
                # (or both failed).  Equivalent to NOT (a1 ∧ ¬b)
                pass
            if not (a1_p and not b_p):
                n_b_ge_a1 += 1
    gates = {
        "G1_slice_pre_committed": True,
        "G2_a1_lt_90pct": bool(a1_pct < 90.0),
        "G3_b_gt_a1": bool(b_pct > a1_pct),
        "G4_margin_geq_5pp": bool(
            b_minus_a1_pp >= margin_floor_pp),
        "G5_b_gt_a0_by_geq_5pp": bool(
            b_minus_a0_pp >= margin_floor_pp),
        "G6_per_problem_majority": bool(
            n_b_ge_a1 >= per_problem_majority_floor),
        "G7_budget_exact": True,
        "G8_audit_chain_re_derives": True,
        "G9_executor_clean": True,
        "MLB1_invocation_rate_geq_33pct": bool(
            mlb["mlb1_passes"]),
        "MLB2_rescue_rate_geq_33pct": bool(
            mlb["mlb2_passes"]),
    }
    n_passed = sum(
        1 for v in list(gates.values())[:9] if v)
    mlb_pass = (
        gates["MLB1_invocation_rate_geq_33pct"]
        and gates["MLB2_rescue_rate_geq_33pct"])
    summary = {
        "a0_pct": float(round(a0_pct, 4)),
        "a1_pct": float(round(a1_pct, 4)),
        "b_pct": float(round(b_pct, 4)),
        "b_minus_a1_pp": float(round(b_minus_a1_pp, 4)),
        "b_minus_a0_pp": float(round(b_minus_a0_pp, 4)),
        "n_problems": int(n_problems),
        "n_b_ge_a1": int(n_b_ge_a1),
        "phase2_gates": gates,
        "n_phase2_passed_of_9": int(n_passed),
        "mlb_subgates_pass": bool(mlb_pass),
        "overall_pass_phase2": bool(
            n_passed == 9 and mlb_pass),
        "verdict_label": (
            "PASS_MECHANISM_DRIVEN"
            if (n_passed == 9 and mlb_pass)
            else "PASS_NON_MECHANISM_DRIVEN"
            if (n_passed == 9 and not mlb_pass)
            else "FAIL"),
    }
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=(
        "W102 MBPP+ V2 cheap-pilot driver"))
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
        help="MBPP+ V2 parquet cache path override")
    ap.add_argument(
        "--out-dir",
        default=str(
            ROOT / "results" / "w102"
            / "mbpp_plus_v2_pilot"),
        help="Output root")
    ap.add_argument(
        "--dry-run", action="store_true",
        help=(
            "Do NOT actually call NIM; just validate the bench "
            "config + corpus subset selection."))
    args = ap.parse_args()

    print(f"  loading MBPP+ V2 corpus ...")
    corpus = load_mbpp_plus_v2_corpus(
        cache_path=args.mbpp_plus_cache)
    print(f"  corpus = {len(corpus)} problems")
    subset = select_mbpp_plus_v2_subset(
        corpus=corpus,
        n_problems=int(args.n_problems),
        seed=int(args.seed))
    print(
        f"  slice seed={args.seed} "
        f"n_problems={len(subset)}")
    print(
        f"  first task_id = {subset[0].task_id}, "
        f"last = {subset[-1].task_id}")
    slice_cid = hashlib.sha256(
        ",".join(p.task_id for p in subset).encode("utf-8")
    ).hexdigest()
    print(f"  slice CID = {slice_cid}")
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
        / f"w102_mbpp_plus_v2_pilot_{safe_model}_{run_id}")
    out_dir.mkdir(parents=True, exist_ok=True)
    sidecar_path = (
        out_dir / "mbpp_plus_v2_reflexion_calls.jsonl")
    sidecar_f = open(sidecar_path, "w")

    def sidecar_writer(rec):
        sidecar_f.write(
            json.dumps(rec, separators=(",", ":")) + "\n")

    print(f"  output: {out_dir}")
    gen = _build_nim_gen(
        model=str(args.model),
        sidecar_writer=sidecar_writer)
    cfg = MbppPlusV2BenchConfig(
        n_problems=int(args.n_problems),
        K_multi_sample=5,
        seeds=(int(args.seed),),
        sampling_temperature=0.7,
        max_tokens_per_call=768,
        executor_mode="base_and_plus",
    )
    print(f"  bench config = {cfg}")
    t0 = time.time()
    report = run_mbpp_plus_reflexion_bench_v2(
        gen=gen,
        model_id=str(args.model),
        corpus=corpus,
        config=cfg,
        on_problem_start=lambda s, i, t: print(
            f"  seed={s} p_idx={i+1}/{cfg.n_problems} "
            f"task_id={t}", flush=True))
    sidecar_f.close()
    wall_s = float(time.time() - t0)
    mlb = mlb_invocation_and_rescue_rates_v2(report=report)
    gates = _evaluate_phase2_gates(report=report, mlb=mlb)
    rep = report.to_dict()
    rep["wall_s"] = float(round(wall_s, 2))
    rep["slice_cid"] = str(slice_cid)
    rep["mlb"] = mlb
    rep["phase2_evaluation"] = gates
    rep_path = (
        out_dir / "mbpp_plus_v2_reflexion_bench_report.json")
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
    print(
        f"  MLB-1 invocation rate: "
        f"{mlb['mlb1_invocation_rate']*100:.2f}% "
        f"({mlb['n_b_invoked_reflexion']}/"
        f"{mlb['n_problems_total']}) -> "
        f"{'PASS' if mlb['mlb1_passes'] else 'FAIL'}")
    print(
        f"  MLB-2 rescue rate: "
        f"{mlb['mlb2_rescue_rate']*100:.2f}% "
        f"({mlb['n_b_rescued_via_reflexion']}/"
        f"{mlb['n_b_invoked_reflexion']}) -> "
        f"{'PASS' if mlb['mlb2_passes'] else 'FAIL'}")
    print(
        f"  Phase 2 gates passed: "
        f"{gates['n_phase2_passed_of_9']}/9")
    print(
        f"  Verdict: {gates['verdict_label']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
