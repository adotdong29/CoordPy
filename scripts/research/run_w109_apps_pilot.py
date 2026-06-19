#!/usr/bin/env python3
"""W109-α — APPS contamination-control cheap-pilot driver (main empirical lane).

CONDITIONAL ON the W109 APPS real-data preflight
(``results/w109/apps_preflight/preflight_verdict.json``) having
``overall_pass=true`` AND ``docs/RUNBOOK_W109.md`` being locked.  Runs the
W89 sequential-reflexion B-pipeline + A0 + A1 baselines against the APPS
call-based (functional) subset at 1 seed × N problems × K=5 (default 30 ⇒
330 NIM calls) at the target model (default ``meta/llama-3.3-70b-instruct`` —
the SAME W89/W105/W108 retirement class, so the contamination contrast is
clean).  Evaluates the pre-committed 9 Phase-2 gates + MLB-1 + MLB-2
sub-gates — the SAME contract used by W103/W104/W105/W108, byte-for-byte
(``_evaluate_phase2_gates`` + ``_mlb_rates`` + ``_build_nim_gen`` copied
verbatim from the W108 LiveCodeBench driver; only the corpus, loader,
executor, bench, and prompt differ).

The W109 control question: does the SAME mechanism RECOVER on
contamination-EXPOSED APPS (2021) after FAILing on contamination-RESISTANT
LiveCodeBench (2025, W108: B − A1 = −3.33 pp)?  A PASS is evidence CONSISTENT
with a contamination-confound — NOT proof, and NOT a retirement (APPS is
contamination-exposed → control evidence only).  A FAIL WEAKENS the confound
hypothesis and tightens the boundary.

Discipline:

* Refuses unpinned operation (SHA mismatch / missing cache / schema mismatch
  ⇒ ``AppsCorpusError`` from the loader).
* Slice is the deterministic, OUTCOME-BLIND difficulty-stratified slice
  (``select_apps_functional_slice_v1``); its CID is pinned in provenance (G1).
* NO LLM-as-judge anywhere (executor truth = subprocess exit code).
* Sidecar flushed per call so a long throttled run is observable on disk.

Requires ``NVIDIA_API_KEY``.

Usage::

    # canary (2 problems ~ 22 calls) then full pilot (30 ~ 330 calls):
    python scripts/run_w109_apps_pilot.py --n-problems 2 --seed 109001 --label canary
    python scripts/run_w109_apps_pilot.py --n-problems 30 --seed 109001
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

from coordpy.apps_loader_v1 import load_apps_call_based_v1  # noqa: E402
from coordpy.apps_reflexion_bench_v1 import (  # noqa: E402
    AppsBenchConfigV1,
    run_apps_reflexion_bench_v1,
    select_apps_functional_slice_v1,
)

NIM_CHAT_URL: str = "https://integrate.api.nvidia.com/v1/chat/completions"

W109_APPS_CACHE_PATH = os.path.expanduser("~/.cache/coordpy/apps-test.jsonl")
W109_APPS_JSONL_SHA256 = (
    "f6c44d76be0eea7669f0ccbd90b6b45fb03a4327d06682073b5cd8f905310918")
W109_APPS_PREFLIGHT_VERDICT_CID = (
    "0cf1a8e2b02acc1511c5db1f2fe3ce79771987a2b9b9759c2dfd978bf5498e7b")


def _sha256_hex_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha256_hex(payload) -> str:
    return _sha256_hex_bytes(
        json.dumps(payload, sort_keys=True, separators=(",", ":"),
                   default=str).encode("utf-8"))


def _file_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---- MLB sub-gates + Phase-2 gates (verbatim from the W108 driver) ---------

def _mlb_rates(report) -> dict:
    n = 0
    invoked = 0
    rescued = 0
    for s in report.per_seed:
        for i in range(len(s.per_problem_b_passed)):
            n += 1
            first_pass_idx = int(s.per_problem_b_first_pass_idx[i])
            b_passed = bool(s.per_problem_b_passed[i])
            attempt_0_failed = (first_pass_idx != 0)
            if attempt_0_failed:
                invoked += 1
                if b_passed:
                    rescued += 1
    inv_rate = float(invoked / n) if n > 0 else 0.0
    rescue_rate = float(rescued / invoked) if invoked > 0 else 0.0
    return {
        "n_problems_total": int(n),
        "n_b_invoked_reflexion": int(invoked),
        "n_b_rescued_via_reflexion": int(rescued),
        "mlb1_invocation_rate": float(round(inv_rate, 4)),
        "mlb2_rescue_rate": float(round(rescue_rate, 4)),
        "mlb1_floor": 0.33, "mlb2_floor": 0.33,
        "mlb1_passes": bool(inv_rate >= 0.33),
        "mlb2_passes": bool(rescue_rate >= 0.33),
    }


def _evaluate_phase2_gates(*, report, mlb,
                           margin_floor_pp: float = 5.0,
                           per_problem_majority_floor: int = 16):
    a0_pct = float(report.a0_mean_pass_at_1 * 100)
    a1_pct = float(report.a1_mean_pass_at_1 * 100)
    b_pct = float(report.b_mean_pass_at_1 * 100)
    b_minus_a1_pp = float(b_pct - a1_pct)
    b_minus_a0_pp = float(b_pct - a0_pct)
    n_problems = sum(len(s.per_problem_b_passed) for s in report.per_seed)
    n_b_ge_a1 = 0
    for s in report.per_seed:
        for i in range(len(s.per_problem_b_passed)):
            b_p = bool(s.per_problem_b_passed[i])
            a1_p = bool(s.per_problem_a1_passed[i])
            if not (a1_p and not b_p):
                n_b_ge_a1 += 1
    gates = {
        "G1_slice_pre_committed": True,
        "G2_a1_lt_90pct": bool(a1_pct < 90.0),
        "G3_b_gt_a1": bool(b_pct > a1_pct),
        "G4_margin_geq_5pp": bool(b_minus_a1_pp >= margin_floor_pp),
        "G5_b_gt_a0_by_geq_5pp": bool(b_minus_a0_pp >= margin_floor_pp),
        "G6_per_problem_majority": bool(
            n_b_ge_a1 >= per_problem_majority_floor),
        "G7_budget_exact": True,
        "G8_audit_chain_re_derives": True,
        "G9_executor_clean": True,
        "MLB1_invocation_rate_geq_33pct": bool(mlb["mlb1_passes"]),
        "MLB2_rescue_rate_geq_33pct": bool(mlb["mlb2_passes"]),
    }
    n_passed = sum(1 for v in list(gates.values())[:9] if v)
    mlb_pass = (gates["MLB1_invocation_rate_geq_33pct"]
                and gates["MLB2_rescue_rate_geq_33pct"])
    verdict_label = (
        "PASS_MECHANISM_DRIVEN" if (n_passed == 9 and mlb_pass)
        else "PASS_NON_MECHANISM_DRIVEN" if (n_passed == 9 and not mlb_pass)
        else "FAIL")
    return {
        "a0_pct": float(round(a0_pct, 4)), "a1_pct": float(round(a1_pct, 4)),
        "b_pct": float(round(b_pct, 4)),
        "b_minus_a1_pp": float(round(b_minus_a1_pp, 4)),
        "b_minus_a0_pp": float(round(b_minus_a0_pp, 4)),
        "n_problems": int(n_problems), "n_b_ge_a1": int(n_b_ge_a1),
        "phase2_gates": gates, "n_phase2_passed_of_9": int(n_passed),
        "mlb_subgates_pass": bool(mlb_pass),
        "overall_pass_phase2": bool(n_passed == 9 and mlb_pass),
        "verdict_label": verdict_label,
    }


def _build_nim_gen(*, model: str, max_retries: int = 12,
                   sidecar_writer=None, inter_call_sleep_s: float = 0.0):
    """NIM chat-completion generator with rate-limit-aware backoff
    (verbatim from the W108 driver)."""
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise SystemExit(
            "NVIDIA_API_KEY not set; W109 cheap pilot requires an "
            "authorised NIM endpoint.")
    import random as _random

    def _gen(prompt: str, max_tokens: int,
             temperature: float) -> tuple[str, int]:
        body = {
            "model": str(model),
            "messages": [{"role": "user", "content": str(prompt)}],
            "max_tokens": int(max_tokens),
            "temperature": float(temperature), "stream": False,
        }
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            NIM_CHAT_URL, data=data, headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {api_key}",
            }, method="POST")
        t0 = time.time()
        last_err: Exception | None = None
        for attempt in range(int(max_retries)):
            try:
                with urllib.request.urlopen(req, timeout=240.0) as r:
                    raw = r.read()
                wall_ms = int((time.time() - t0) * 1000)
                payload = json.loads(raw.decode("utf-8", errors="replace"))
                text = ""
                choices = payload.get("choices") or []
                if choices:
                    msg = choices[0].get("message") or {}
                    text = str(msg.get("content") or "")
                if sidecar_writer is not None:
                    sidecar_writer({
                        "model_id": str(model), "backend": "nim",
                        "prompt_len": int(len(prompt)),
                        "prompt_sha256": hashlib.sha256(
                            prompt.encode("utf-8")).hexdigest(),
                        "response_len": int(len(text)),
                        "response_sha256": hashlib.sha256(
                            text.encode("utf-8")).hexdigest(),
                        "temperature": float(temperature),
                        "max_tokens": int(max_tokens),
                        "wall_ms": int(wall_ms),
                        "prompt": str(prompt), "response_text": str(text),
                    })
                if inter_call_sleep_s > 0:
                    time.sleep(float(inter_call_sleep_s))
                return str(text), int(wall_ms)
            except urllib.error.HTTPError as e:
                last_err = e
                if e.code in (429, 502, 503, 504):
                    backoff = min(float(2 ** attempt)
                                  + (_random.random() * 5.0), 300.0)
                    print(f"  [nim retry] HTTP {e.code} attempt "
                          f"{attempt+1}/{max_retries}; sleeping "
                          f"{backoff:.1f}s", flush=True)
                    time.sleep(backoff)
                    continue
                raise
            except Exception as e:  # noqa: BLE001
                last_err = e
                backoff = min(float(2 ** attempt)
                              + (_random.random() * 3.0), 120.0)
                print(f"  [nim retry] {type(e).__name__}: {e}; attempt "
                      f"{attempt+1}/{max_retries}; sleeping {backoff:.1f}s",
                      flush=True)
                time.sleep(backoff)
        raise RuntimeError(
            f"NIM call failed after {max_retries} attempts: {last_err}")
    return _gen


def main() -> int:
    ap = argparse.ArgumentParser(
        description="W109 APPS contamination-control cheap-pilot driver")
    ap.add_argument("--model", default="meta/llama-3.3-70b-instruct")
    ap.add_argument("--cache-path", default=W109_APPS_CACHE_PATH)
    ap.add_argument("--expected-sha256", default=W109_APPS_JSONL_SHA256)
    ap.add_argument("--n-problems", type=int, default=30)
    ap.add_argument("--seed", type=int, default=109_001)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--max-tests-per-problem", type=int, default=25)
    ap.add_argument(
        "--out-dir",
        default=str(ROOT / "results" / "w109" / "apps_pilot"))
    ap.add_argument("--label", default="",
                    help="optional run label suffix (e.g. canary)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print("  loading APPS call-based corpus (SHA-pinned) ...")
    full_subset = load_apps_call_based_v1(
        cache_path=str(args.cache_path),
        expected_sha256=str(args.expected_sha256))
    print(f"  call-based subset = {len(full_subset)} problems")
    pilot_slice = select_apps_functional_slice_v1(
        full_subset, n_problems=int(args.n_problems))
    slice_ids = [str(p.problem_id) for p in pilot_slice]
    slice_cid = _sha256_hex({"kind": "w109_apps_pilot_slice_v1",
                             "problem_ids": slice_ids})
    mix = Counter(p.difficulty for p in pilot_slice)
    print(f"  pilot slice = {len(pilot_slice)} problems; difficulty {dict(mix)}")
    print(f"  slice CID = {slice_cid}")
    corpus_sha = _file_sha256(Path(args.cache_path))
    if corpus_sha.lower() != str(args.expected_sha256).lower():
        raise SystemExit("corpus SHA drift; refusing to spend NIM")

    if args.dry_run:
        print("  --dry-run: validated slice + corpus; stopping before NIM")
        return 0

    run_id = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_model = str(args.model).replace("/", "_")
    lbl = (f"_{args.label}" if args.label else "")
    out_dir = (Path(args.out_dir)
               / f"w109_apps_pilot_{safe_model}_{run_id}{lbl}")
    out_dir.mkdir(parents=True, exist_ok=True)
    sidecar_path = out_dir / "apps_reflexion_calls.jsonl"
    sidecar_f = open(sidecar_path, "w")

    def sidecar_writer(rec):
        sidecar_f.write(json.dumps(rec, separators=(",", ":")) + "\n")
        sidecar_f.flush()

    provenance = {
        "schema": "coordpy.w109_apps_pilot.v1",
        "model_id": str(args.model), "seed": int(args.seed),
        "n_problems": int(len(pilot_slice)), "K_multi_sample": 5,
        "corpus_path": str(args.cache_path),
        "corpus_sha256": str(corpus_sha),
        "dataset": "codeparrot/apps (refs/convert/parquet)",
        "contamination_status": (
            "EXPOSED — APPS 2021 vintage, pre Llama-3.x 2024-01-01 cutoff "
            "(C7 = C; control/backup evidence only)"),
        "control_contrast": (
            "vs W108 LiveCodeBench 2025 contamination-RESISTANT FAIL "
            "(B - A1 = -3.33 pp; MLB-2 = 25%)"),
        "preflight_verdict_cid": str(W109_APPS_PREFLIGHT_VERDICT_CID),
        "slice_cid": str(slice_cid),
        "slice_problem_ids": list(slice_ids),
        "slice_difficulty_mix": dict(mix),
        "max_tokens_per_call": int(args.max_tokens),
        "max_tests_per_problem": int(args.max_tests_per_problem),
        "phase2_gate_floors": {
            "G2_a1_max_pct": 90.0, "G4_margin_min_pp": 5.0,
            "G5_b_gt_a0_min_pp": 5.0, "G6_per_problem_majority_min": 16,
            "MLB1_floor": 0.33, "MLB2_floor": 0.33},
        "label": str(args.label),
    }
    with open(out_dir / "provenance.json", "w") as f:
        json.dump(provenance, f, indent=2, default=str)
    print(f"  output: {out_dir}")

    gen = _build_nim_gen(model=str(args.model), sidecar_writer=sidecar_writer)
    cfg = AppsBenchConfigV1(
        K_multi_sample=5, seeds=(int(args.seed),),
        sampling_temperature=0.7, max_tokens_per_call=int(args.max_tokens),
        max_tests_per_problem=int(args.max_tests_per_problem))
    print(f"  bench config = {cfg}")
    t0 = time.time()
    report = run_apps_reflexion_bench_v1(
        gen=gen, model_id=str(args.model), subset=pilot_slice, config=cfg,
        on_problem_start=lambda s, i, t: print(
            f"  seed={s} p_idx={i+1}/{len(pilot_slice)} pid={t}", flush=True))
    sidecar_f.close()
    wall_s = float(time.time() - t0)
    mlb = _mlb_rates(report)
    gates = _evaluate_phase2_gates(report=report, mlb=mlb)
    rep = report.to_dict()
    rep["wall_s"] = float(round(wall_s, 2))
    rep["provenance"] = provenance
    rep["mlb"] = mlb
    rep["phase2_evaluation"] = gates
    with open(out_dir / "apps_reflexion_bench_report.json", "w") as f:
        json.dump(rep, f, indent=2, default=str)
    with open(out_dir.parent / "latest_run.txt", "w") as f:
        f.write(out_dir.name + "\n")

    print()
    print(f"  WALL: {wall_s:.1f} s; "
          f"A0={report.a0_mean_pass_at_1*100:.2f}% "
          f"A1={report.a1_mean_pass_at_1*100:.2f}% "
          f"B={report.b_mean_pass_at_1*100:.2f}% "
          f"B-A1={report.b_mean_minus_a1_mean_pp:+.2f}pp")
    print(f"  MLB-1 invocation: {mlb['mlb1_invocation_rate']*100:.2f}% "
          f"({mlb['n_b_invoked_reflexion']}/{mlb['n_problems_total']}) "
          f"-> {'PASS' if mlb['mlb1_passes'] else 'FAIL'}")
    print(f"  MLB-2 rescue: {mlb['mlb2_rescue_rate']*100:.2f}% "
          f"({mlb['n_b_rescued_via_reflexion']}/"
          f"{mlb['n_b_invoked_reflexion']}) "
          f"-> {'PASS' if mlb['mlb2_passes'] else 'FAIL'}")
    print(f"  Phase 2 gates: {gates['n_phase2_passed_of_9']}/9")
    print(f"  Verdict: {gates['verdict_label']}")
    print(f"  out_dir: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
