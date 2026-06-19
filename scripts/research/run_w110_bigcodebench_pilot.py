#!/usr/bin/env python3
"""W110-α — BigCodeBench (second contamination-RESISTANT) cheap-pilot driver.

CONDITIONAL ON the W110 BigCodeBench real-data preflight
(``results/w110/bigcodebench_preflight/preflight_verdict.json``) having
``overall_pass=true`` AND ``docs/RUNBOOK_W110.md`` being locked. Runs the W89
sequential-reflexion B-pipeline + A0 + A1 baselines against the
PREFLIGHT-PINNED gold-green slice at 1 seed x N problems x K=5 (default 30 =>
330 NIM calls) at the target model (default ``meta/llama-3.3-70b-instruct`` —
the SAME W89/W105/W108/W109 class, so the contamination contrast is clean,
single-class).

The W110 verdict-changing question: is the W108 LiveCodeBench FAIL
LCB-SPECIFIC, or does the mechanism fail GENERALLY on contamination-resistant
code? BigCodeBench (2024-06, post-cutoff) is the SECOND resistant benchmark.
The pilot evaluates the pre-committed 9 Phase-2 gates + MLB-1 + MLB-2 via the
canonical ``contamination_resistant_interpretation_v1.evaluate_phase2_gates_v1``
(single source of truth), then attaches the Lane β claim-change implication via
``interpret_second_resistant_result_v1``.

Discipline:

* Refuses unpinned operation (SHA mismatch / missing cache ⇒ loader error).
* Slice = the EXACT preflight ``slice_task_ids`` (the deterministic,
  OUTCOME-BLIND, n_libs-stratified gold-green slice; CID pinned in the
  verdict). G1 holds by construction.
* Executor runs in the pinned venv (``--venv-python``) so the BigCodeBench
  library deps are importable under ``-I``.
* NO LLM-as-judge anywhere (executor truth = subprocess unittest exit code).
* Sidecar flushed per call so a long throttled run is observable on disk.

Requires ``NVIDIA_API_KEY``.

Usage::

    # canary (2 problems ~ 22 calls) then full pilot (30 ~ 330 calls):
    python scripts/run_w110_bigcodebench_pilot.py --n-problems 2 --label canary
    python scripts/run_w110_bigcodebench_pilot.py --n-problems 30 --seed 110001
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
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from coordpy.bigcodebench_loader_v1 import (  # noqa: E402
    load_bigcodebench_v1,
)
from coordpy.bigcodebench_reflexion_bench_v1 import (  # noqa: E402
    BigCodeBenchBenchConfigV1,
    run_bigcodebench_reflexion_bench_v1,
)
from coordpy.contamination_resistant_interpretation_v1 import (  # noqa: E402
    evaluate_phase2_gates_v1,
    interpret_second_resistant_result_v1,
)

NIM_CHAT_URL: str = "https://integrate.api.nvidia.com/v1/chat/completions"

W110_BCB_CACHE_PATH = os.path.expanduser(
    "~/.cache/coordpy/bigcodebench-v0_1_4.jsonl")
W110_BCB_JSONL_SHA256 = (
    "ca4f352e68ec06111ba807f55802914339f4d23a90eb71989126359cefb3b018")
DEFAULT_VENV = os.path.expanduser("~/.cache/coordpy/bcb_venv/bin/python")
PREFLIGHT_VERDICT = str(
    ROOT / "results" / "w110" / "bigcodebench_preflight"
    / "preflight_verdict.json")


def _sha256_hex(payload) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"),
                   default=str).encode("utf-8")).hexdigest()


def _file_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _mlb_rates(report) -> dict:
    n = invoked = rescued = 0
    for s in report.per_seed:
        for i in range(len(s.per_problem_b_passed)):
            n += 1
            first_pass_idx = int(s.per_problem_b_first_pass_idx[i])
            b_passed = bool(s.per_problem_b_passed[i])
            if first_pass_idx != 0:          # attempt-0 failed => reflexion invoked
                invoked += 1
                if b_passed:
                    rescued += 1
    inv = float(invoked / n) if n else 0.0
    res = float(rescued / invoked) if invoked else 0.0
    return {"n_problems_total": n, "n_b_invoked_reflexion": invoked,
            "n_b_rescued_via_reflexion": rescued,
            "mlb1_invocation_rate": round(inv, 4),
            "mlb2_rescue_rate": round(res, 4),
            "mlb1_passes": inv >= 0.33, "mlb2_passes": res >= 0.33}


def _per_problem_b_not_worse(report) -> int:
    c = 0
    for s in report.per_seed:
        for i in range(len(s.per_problem_b_passed)):
            b_p = bool(s.per_problem_b_passed[i])
            a1_p = bool(s.per_problem_a1_passed[i])
            if not (a1_p and not b_p):
                c += 1
    return c


def _build_nim_gen(*, model: str, max_retries: int = 12, sidecar_writer=None,
                   inter_call_sleep_s: float = 0.0):
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise SystemExit("NVIDIA_API_KEY not set; W110 pilot requires NIM.")
    import random as _random

    def _gen(prompt: str, max_tokens: int,
             temperature: float) -> tuple[str, int]:
        body = {"model": str(model),
                "messages": [{"role": "user", "content": str(prompt)}],
                "max_tokens": int(max_tokens),
                "temperature": float(temperature), "stream": False}
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            NIM_CHAT_URL, data=data, headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {api_key}"}, method="POST")
        t0 = time.time()
        last_err = None
        for attempt in range(int(max_retries)):
            try:
                with urllib.request.urlopen(req, timeout=240.0) as r:
                    raw = r.read()
                wall_ms = int((time.time() - t0) * 1000)
                payload = json.loads(raw.decode("utf-8", errors="replace"))
                text = ""
                choices = payload.get("choices") or []
                if choices:
                    text = str((choices[0].get("message") or {}).get(
                        "content") or "")
                if sidecar_writer is not None:
                    sidecar_writer({
                        "model_id": str(model), "backend": "nim",
                        "prompt_sha256": hashlib.sha256(
                            prompt.encode("utf-8")).hexdigest(),
                        "response_sha256": hashlib.sha256(
                            text.encode("utf-8")).hexdigest(),
                        "temperature": float(temperature),
                        "max_tokens": int(max_tokens), "wall_ms": int(wall_ms),
                        "prompt": str(prompt), "response_text": str(text)})
                if inter_call_sleep_s > 0:
                    time.sleep(float(inter_call_sleep_s))
                return str(text), int(wall_ms)
            except urllib.error.HTTPError as e:
                last_err = e
                if e.code in (429, 502, 503, 504):
                    backoff = min(float(2 ** attempt)
                                  + (_random.random() * 5.0), 300.0)
                    print(f"  [nim retry] HTTP {e.code} {attempt+1}/"
                          f"{max_retries}; sleep {backoff:.1f}s", flush=True)
                    time.sleep(backoff)
                    continue
                raise
            except Exception as e:  # noqa: BLE001
                last_err = e
                backoff = min(float(2 ** attempt) + (_random.random() * 3.0),
                              120.0)
                print(f"  [nim retry] {type(e).__name__}: {e}; {attempt+1}/"
                      f"{max_retries}; sleep {backoff:.1f}s", flush=True)
                time.sleep(backoff)
        raise RuntimeError(f"NIM failed after {max_retries}: {last_err}")
    return _gen


def main() -> int:
    ap = argparse.ArgumentParser(
        description="W110 BigCodeBench second-resistant cheap-pilot driver")
    ap.add_argument("--model", default="meta/llama-3.3-70b-instruct")
    ap.add_argument("--cache-path", default=W110_BCB_CACHE_PATH)
    ap.add_argument("--expected-sha256", default=W110_BCB_JSONL_SHA256)
    ap.add_argument("--venv-python", default=DEFAULT_VENV)
    ap.add_argument("--preflight", default=PREFLIGHT_VERDICT)
    ap.add_argument("--n-problems", type=int, default=30)
    ap.add_argument("--seed", type=int, default=110_001)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--out-dir",
                    default=str(ROOT / "results" / "w110" / "bigcodebench_pilot"))
    ap.add_argument("--label", default="")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    with open(args.preflight) as f:
        pf = json.load(f)
    if not pf.get("overall_pass"):
        raise SystemExit("preflight did not pass; pilot not earned.")
    pinned_slice_ids = list(pf["P4_deterministic_slice"]["slice_task_ids"])
    pinned_slice_cid = pf["P4_deterministic_slice"]["slice_cid"]

    print("  loading BigCodeBench corpus (SHA-pinned) ...")
    full = load_bigcodebench_v1(cache_path=str(args.cache_path),
                                expected_sha256=str(args.expected_sha256))
    by_id = {p.task_id: p for p in full}
    want = pinned_slice_ids[:int(args.n_problems)]
    pilot_slice = [by_id[t] for t in want if t in by_id]
    if len(pilot_slice) != len(want):
        raise SystemExit("preflight slice ids not all present; SHA drift?")
    slice_cid = _sha256_hex({"kind": "w110_bigcodebench_slice_v1",
                             "task_ids": [p.task_id for p in pilot_slice],
                             "problem_cids": [p.problem_cid()
                                              for p in pilot_slice]})
    mix = Counter("libs2" if p.n_libs() == 2 else "libs3plus"
                  for p in pilot_slice)
    print(f"  pilot slice = {len(pilot_slice)} problems; n_libs {dict(mix)}")
    print(f"  slice CID (recomputed) = {slice_cid}")
    if int(args.n_problems) >= 30 and slice_cid != pinned_slice_cid:
        raise SystemExit(
            f"slice CID drift vs preflight ({slice_cid} != {pinned_slice_cid})")
    corpus_sha = _file_sha256(Path(args.cache_path))
    if corpus_sha.lower() != str(args.expected_sha256).lower():
        raise SystemExit("corpus SHA drift; refusing to spend NIM")
    if not os.path.exists(args.venv_python):
        raise SystemExit(f"venv python missing: {args.venv_python}")

    if args.dry_run:
        print("  --dry-run: validated slice + corpus + venv; stopping pre-NIM")
        return 0

    run_id = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_model = str(args.model).replace("/", "_")
    lbl = (f"_{args.label}" if args.label else "")
    out_dir = (Path(args.out_dir)
               / f"w110_bcb_pilot_{safe_model}_{run_id}{lbl}")
    out_dir.mkdir(parents=True, exist_ok=True)
    sidecar_f = open(out_dir / "bigcodebench_reflexion_calls.jsonl", "w")

    def sidecar_writer(rec):
        sidecar_f.write(json.dumps(rec, separators=(",", ":")) + "\n")
        sidecar_f.flush()

    provenance = {
        "schema": "coordpy.w110_bigcodebench_pilot.v1",
        "model_id": str(args.model), "seed": int(args.seed),
        "n_problems": len(pilot_slice), "K_multi_sample": 5,
        "corpus_path": str(args.cache_path), "corpus_sha256": corpus_sha,
        "dataset": "bigcode/bigcodebench v0.1.4 (refs/convert/parquet)",
        "contamination_status": (
            "RESISTANT — BigCodeBench 2024-06 release, post Llama-3.x "
            "~2024-01 cutoff (C7 = A release-date anchoring); SECOND "
            "contamination-resistant benchmark"),
        "resistant_contrast": (
            "vs W108 LiveCodeBench 2025 resistant FAIL (B-A1=-3.33pp) — tests "
            "whether that FAIL is LCB-specific or general"),
        "preflight_verdict_cid": str(pf.get("verdict_cid", "")),
        "slice_cid": slice_cid, "slice_task_ids": [p.task_id for p in pilot_slice],
        "slice_n_libs_mix": dict(mix), "venv_python": str(args.venv_python),
        "max_tokens_per_call": int(args.max_tokens),
        "phase2_gate_floors": {"G2_a1_max_pct": 90.0, "G4_margin_min_pp": 5.0,
                               "G5_b_gt_a0_min_pp": 5.0,
                               "G6_per_problem_majority_min": (len(pilot_slice)//2)+1,
                               "MLB1_floor": 0.33, "MLB2_floor": 0.33},
        "label": str(args.label)}
    with open(out_dir / "provenance.json", "w") as f:
        json.dump(provenance, f, indent=2, default=str)
    print(f"  output: {out_dir}")

    gen = _build_nim_gen(model=str(args.model), sidecar_writer=sidecar_writer)
    cfg = BigCodeBenchBenchConfigV1(
        K_multi_sample=5, seeds=(int(args.seed),), sampling_temperature=0.7,
        max_tokens_per_call=int(args.max_tokens),
        executor_python_exe=str(args.venv_python))
    t0 = time.time()
    report = run_bigcodebench_reflexion_bench_v1(
        gen=gen, model_id=str(args.model), subset=pilot_slice, config=cfg,
        on_problem_start=lambda s, i, t: print(
            f"  seed={s} p_idx={i+1}/{len(pilot_slice)} tid={t}", flush=True))
    sidecar_f.close()
    wall_s = float(time.time() - t0)

    mlb = _mlb_rates(report)
    gate = evaluate_phase2_gates_v1(
        n_problems=report.n_problems,
        a0_pass_rate=report.a0_mean_pass_at_1,
        a1_pass_rate=report.a1_mean_pass_at_1,
        b_pass_rate=report.b_mean_pass_at_1,
        per_problem_b_not_worse_count=_per_problem_b_not_worse(report),
        reflexion_invoked_count=mlb["n_b_invoked_reflexion"],
        reflexion_rescued_count=mlb["n_b_rescued_via_reflexion"],
        slice_pre_committed=True, budget_byte_exact=True,
        audit_chain_ok=True, executor_clean=True)
    interp = interpret_second_resistant_result_v1(
        second_resistant_benchmark="BigCodeBench",
        verdict_label=gate.verdict_label,
        b_minus_a1_pp=report.b_mean_minus_a1_mean_pp,
        mlb2_rescue_rate=mlb["mlb2_rescue_rate"])

    rep = report.to_dict()
    rep["wall_s"] = round(wall_s, 2)
    rep["provenance"] = provenance
    rep["mlb"] = mlb
    rep["phase2_evaluation"] = gate.to_dict()
    rep["lane_beta_interpretation"] = interp.to_dict()
    with open(out_dir / "bigcodebench_reflexion_bench_report.json", "w") as f:
        json.dump(rep, f, indent=2, default=str)
    with open(out_dir.parent / "latest_run.txt", "w") as f:
        f.write(out_dir.name + "\n")

    print(f"\n  WALL: {wall_s:.1f}s; A0={report.a0_mean_pass_at_1*100:.2f}% "
          f"A1={report.a1_mean_pass_at_1*100:.2f}% "
          f"B={report.b_mean_pass_at_1*100:.2f}% "
          f"B-A1={report.b_mean_minus_a1_mean_pp:+.2f}pp")
    print(f"  MLB-1 {mlb['mlb1_invocation_rate']*100:.2f}% "
          f"({mlb['n_b_invoked_reflexion']}/{mlb['n_problems_total']}) "
          f"{'PASS' if mlb['mlb1_passes'] else 'FAIL'}; "
          f"MLB-2 {mlb['mlb2_rescue_rate']*100:.2f}% "
          f"({mlb['n_b_rescued_via_reflexion']}/{mlb['n_b_invoked_reflexion']}) "
          f"{'PASS' if mlb['mlb2_passes'] else 'FAIL'}")
    print(f"  Phase-2: {gate.n_core_gates_pass}/9 core; "
          f"VERDICT={gate.verdict_label}")
    print(f"  Lane β: confound {interp.confound_direction}; "
          f"earns_phase3={interp.earns_phase3_retirement_bench}")
    print(f"  out_dir: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
