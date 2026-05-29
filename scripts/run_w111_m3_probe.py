#!/usr/bin/env python3
"""W111-α — M3 (executor-grounded structured-failure patcher) smallest-decisive
cheap probe.

CONDITIONAL ON ``docs/RUNBOOK_W111.md`` being locked (it is) and M3 having
earned the live probe under § 4.3 (it has: weakness-coverage 81.6% ≥ 33%;
hard-core fair recoverable surface = 2 ≥ 1).

Runs A0 + A1 + M3 (the executor-grounded structured-failure patcher) on the
PINNED, hard-core-focused 13-problem slice of the W110 BigCodeBench gold-green
pool (3 both-pass controls + 8 both-A1+B-fail hard-core + /51 B-rescue + /26
B-regression), single seed, K=5 byte-exact. ~143 NIM calls.

This slice is rescue-CONCENTRATED by construction (the W102/W106 discipline) =>
it yields an UPPER BOUND on M3, NEVER a PASS claim. Its job is the § 4.3
KILL / EARN decision:

* KILL M3 (=> bounded-claim fallback, RUNBOOK § 6) iff M3 rescues 0 of the 2
  OUTPUT_VALUE hard-core problems (BigCodeBench/15, /20) — the only fair-regime
  targets where the expected value is in the executor stderr.
* M3 EARNS the fair 30-slice cheap pilot (RUNBOOK § 5) iff it rescues >= 1
  OUTPUT_VALUE hard-core problem AND holds BigCodeBench/51 AND regresses <= 1 of
  the 3 both-pass controls.

NO LLM-as-judge (executor truth = unittest exit code). Requires ``NVIDIA_API_KEY``.

Usage::

    python scripts/run_w111_m3_probe.py --dry-run          # validate, no NIM
    python scripts/run_w111_m3_probe.py --label canary --n-problems 2
    python scripts/run_w111_m3_probe.py                    # full 13-problem probe
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

from coordpy.bigcodebench_loader_v1 import load_bigcodebench_v1  # noqa: E402
from coordpy.executor_grounded_patcher_v1 import (  # noqa: E402
    PatcherBenchConfigV1,
    run_executor_grounded_patcher_bench_v1,
)
from coordpy.contamination_resistant_interpretation_v1 import (  # noqa: E402
    evaluate_phase2_gates_v1,
)

NIM_CHAT_URL: str = "https://integrate.api.nvidia.com/v1/chat/completions"
W111_BCB_CACHE_PATH = os.path.expanduser(
    "~/.cache/coordpy/bigcodebench-v0_1_4.jsonl")
W111_BCB_JSONL_SHA256 = (
    "ca4f352e68ec06111ba807f55802914339f4d23a90eb71989126359cefb3b018")
DEFAULT_VENV = os.path.expanduser("~/.cache/coordpy/bcb_venv/bin/python")

# Pinned hard-core-focused probe slice (RUNBOOK_W111 § 4.3), task_id order.
# 3 controls (/1,/2,/3) + 8 hard-core (both A1+B fail) + /26 (B-regress) + /51 (B-rescue).
PROBE_SLICE_IDS: tuple[str, ...] = (
    "BigCodeBench/1", "BigCodeBench/2", "BigCodeBench/3",
    "BigCodeBench/6", "BigCodeBench/10", "BigCodeBench/12", "BigCodeBench/13",
    "BigCodeBench/15", "BigCodeBench/17", "BigCodeBench/20",
    "BigCodeBench/26", "BigCodeBench/32", "BigCodeBench/51")
# M3 kill/earn targets (OUTPUT_VALUE hard-core: expected value IS in stderr).
M3_OUTPUT_VALUE_TARGETS = ("BigCodeBench/15", "BigCodeBench/20")
CONTROLS = ("BigCodeBench/1", "BigCodeBench/2", "BigCodeBench/3")
HOLD_CHECK = "BigCodeBench/51"


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


def _m3_mlb(report) -> dict:
    n = invoked = rescued = 0
    for s in report.per_seed:
        for i in range(len(s.per_problem_m3_passed)):
            n += 1
            fp = int(s.per_problem_m3_first_pass_idx[i])
            passed = bool(s.per_problem_m3_passed[i])
            if fp != 0:                       # attempt-0 failed => patch invoked
                invoked += 1
                if passed:
                    rescued += 1
    inv = float(invoked / n) if n else 0.0
    res = float(rescued / invoked) if invoked else 0.0
    return {"n_problems_total": n, "n_m3_invoked_patch": invoked,
            "n_m3_rescued_via_patch": rescued,
            "mlb1_invocation_rate": round(inv, 4),
            "mlb2_rescue_rate": round(res, 4),
            "mlb1_passes": inv >= 0.33, "mlb2_passes": res >= 0.33}


def _per_problem_m3_not_worse(report) -> int:
    c = 0
    for s in report.per_seed:
        for i in range(len(s.per_problem_m3_passed)):
            if not (bool(s.per_problem_a1_passed[i])
                    and not bool(s.per_problem_m3_passed[i])):
                c += 1
    return c


def _kill_earn_decision(report) -> dict:
    s = report.per_seed[0]
    pid_idx = {t: i for i, t in enumerate(s.problem_ids)}

    def m3(t):
        return bool(s.per_problem_m3_passed[pid_idx[t]]) if t in pid_idx else None

    def a1(t):
        return bool(s.per_problem_a1_passed[pid_idx[t]]) if t in pid_idx else None

    output_value_rescued = [t for t in M3_OUTPUT_VALUE_TARGETS
                            if t in pid_idx and m3(t)]
    hold_51 = m3(HOLD_CHECK) if HOLD_CHECK in pid_idx else None
    control_regressions = [t for t in CONTROLS
                           if t in pid_idx and a1(t) and not m3(t)]
    n_ov = len(output_value_rescued)
    earns = (n_ov >= 1 and (hold_51 is True or HOLD_CHECK not in pid_idx)
             and len(control_regressions) <= 1)
    kill = (n_ov == 0)
    verdict = ("EARNS_FAIR_30SLICE_PILOT" if earns
               else ("KILL_M3_BOUNDED_FALLBACK" if kill else "AMBIGUOUS"))
    return {
        "output_value_targets": list(M3_OUTPUT_VALUE_TARGETS),
        "output_value_rescued": output_value_rescued,
        "n_output_value_rescued": n_ov,
        "hold_check_51_held": hold_51,
        "control_regressions": control_regressions,
        "verdict": verdict,
        "rule": ("EARN iff >=1 OUTPUT_VALUE rescued AND /51 held AND <=1 control "
                 "regression; KILL iff 0 OUTPUT_VALUE rescued (RUNBOOK_W111 § 4.3)"),
    }


def _build_nim_gen(*, model, max_retries=12, sidecar_writer=None):
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise SystemExit("NVIDIA_API_KEY not set; W111 probe requires NIM.")
    import random as _random

    def _gen(prompt, max_tokens, temperature):
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
    ap = argparse.ArgumentParser(description="W111 M3 smallest-decisive probe")
    ap.add_argument("--model", default="meta/llama-3.3-70b-instruct")
    ap.add_argument("--cache-path", default=W111_BCB_CACHE_PATH)
    ap.add_argument("--expected-sha256", default=W111_BCB_JSONL_SHA256)
    ap.add_argument("--venv-python", default=DEFAULT_VENV)
    ap.add_argument("--n-problems", type=int, default=len(PROBE_SLICE_IDS))
    ap.add_argument("--seed", type=int, default=111_001)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--out-dir",
                    default=str(ROOT / "results" / "w111" / "m3_probe"))
    ap.add_argument("--label", default="")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print("  loading BigCodeBench corpus (SHA-pinned) ...")
    full = load_bigcodebench_v1(cache_path=str(args.cache_path),
                                expected_sha256=str(args.expected_sha256))
    by_id = {p.task_id: p for p in full}
    want = list(PROBE_SLICE_IDS)[:int(args.n_problems)]
    probe_slice = [by_id[t] for t in want if t in by_id]
    if len(probe_slice) != len(want):
        raise SystemExit("probe slice ids not all present; SHA drift?")
    slice_cid = _sha256_hex({"kind": "w111_m3_probe_slice_v1",
                             "task_ids": [p.task_id for p in probe_slice],
                             "problem_cids": [p.problem_cid()
                                              for p in probe_slice]})
    print(f"  probe slice = {len(probe_slice)} problems; CID = {slice_cid}")
    corpus_sha = _file_sha256(Path(args.cache_path))
    if corpus_sha.lower() != str(args.expected_sha256).lower():
        raise SystemExit("corpus SHA drift; refusing to spend NIM")
    if not os.path.exists(args.venv_python):
        raise SystemExit(f"venv python missing: {args.venv_python}")
    if args.dry_run:
        print("  --dry-run: validated slice + corpus + venv; stopping pre-NIM")
        print(f"  expected NIM calls = {len(probe_slice)} x 11 = "
              f"{len(probe_slice) * 11}")
        return 0

    run_id = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_model = str(args.model).replace("/", "_")
    lbl = (f"_{args.label}" if args.label else "")
    out_dir = (Path(args.out_dir) / f"w111_m3_probe_{safe_model}_{run_id}{lbl}")
    out_dir.mkdir(parents=True, exist_ok=True)
    sidecar_f = open(out_dir / "patcher_calls.jsonl", "w")

    def sidecar_writer(rec):
        sidecar_f.write(json.dumps(rec, separators=(",", ":")) + "\n")
        sidecar_f.flush()

    provenance = {
        "schema": "coordpy.w111_m3_probe.v1",
        "mechanism": "M3 executor-grounded structured-failure patcher",
        "model_id": str(args.model), "seed": int(args.seed),
        "n_problems": len(probe_slice), "K_multi_sample": 5,
        "corpus_path": str(args.cache_path), "corpus_sha256": corpus_sha,
        "dataset": "bigcode/bigcodebench v0.1.4 (the W110 pinned corpus)",
        "probe_slice_kind": (
            "RESCUE-CONCENTRATED hard-core-focused (NOT a fair slice => UPPER "
            "BOUND, never a PASS claim; KILL/EARN decision only)"),
        "slice_cid": slice_cid, "slice_task_ids": [p.task_id for p in probe_slice],
        "m3_output_value_targets": list(M3_OUTPUT_VALUE_TARGETS),
        "controls": list(CONTROLS), "hold_check": HOLD_CHECK,
        "venv_python": str(args.venv_python),
        "max_tokens_per_call": int(args.max_tokens),
        "runbook": "docs/RUNBOOK_W111.md § 4.3", "label": str(args.label)}
    with open(out_dir / "provenance.json", "w") as f:
        json.dump(provenance, f, indent=2, default=str)
    print(f"  output: {out_dir}")

    gen = _build_nim_gen(model=str(args.model), sidecar_writer=sidecar_writer)
    cfg = PatcherBenchConfigV1(
        K_multi_sample=5, seeds=(int(args.seed),), sampling_temperature=0.7,
        max_tokens_per_call=int(args.max_tokens),
        executor_python_exe=str(args.venv_python))
    t0 = time.time()
    report = run_executor_grounded_patcher_bench_v1(
        gen=gen, model_id=str(args.model), subset=probe_slice, config=cfg,
        on_problem_start=lambda s, i, t: print(
            f"  seed={s} p_idx={i+1}/{len(probe_slice)} tid={t}", flush=True))
    sidecar_f.close()
    wall_s = float(time.time() - t0)

    mlb = _m3_mlb(report)
    # Informational gate eval (M3 as "B"); slice is rescue-concentrated => NOT a
    # PASS claim, the KILL/EARN decision below is binding.
    gate = evaluate_phase2_gates_v1(
        n_problems=report.n_problems,
        a0_pass_rate=report.a0_mean_pass_at_1,
        a1_pass_rate=report.a1_mean_pass_at_1,
        b_pass_rate=report.m3_mean_pass_at_1,
        per_problem_b_not_worse_count=_per_problem_m3_not_worse(report),
        reflexion_invoked_count=mlb["n_m3_invoked_patch"],
        reflexion_rescued_count=mlb["n_m3_rescued_via_patch"],
        slice_pre_committed=True, budget_byte_exact=True,
        audit_chain_ok=True, executor_clean=True)
    decision = _kill_earn_decision(report)

    rep = report.to_dict()
    rep["wall_s"] = round(wall_s, 2)
    rep["provenance"] = provenance
    rep["m3_mlb"] = mlb
    rep["informational_gate_eval_NOT_a_pass_claim"] = gate.to_dict()
    rep["kill_earn_decision"] = decision
    with open(out_dir / "patcher_bench_report.json", "w") as f:
        json.dump(rep, f, indent=2, default=str)
    with open(out_dir.parent / "latest_run.txt", "w") as f:
        f.write(out_dir.name + "\n")

    print(f"\n  WALL: {wall_s:.1f}s; A0={report.a0_mean_pass_at_1*100:.2f}% "
          f"A1={report.a1_mean_pass_at_1*100:.2f}% "
          f"M3={report.m3_mean_pass_at_1*100:.2f}% "
          f"M3-A1={report.m3_mean_minus_a1_mean_pp:+.2f}pp "
          f"(rescue-concentrated UPPER BOUND)")
    print(f"  M3 MLB-1 {mlb['mlb1_invocation_rate']*100:.1f}% / "
          f"MLB-2 {mlb['mlb2_rescue_rate']*100:.1f}%")
    print(f"  OUTPUT_VALUE rescued: {decision['output_value_rescued']} "
          f"({decision['n_output_value_rescued']}/2); "
          f"/51 held: {decision['hold_check_51_held']}; "
          f"control regressions: {decision['control_regressions']}")
    print(f"  === KILL/EARN VERDICT: {decision['verdict']} ===")
    print(f"  out_dir: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
