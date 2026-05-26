#!/usr/bin/env python3
"""W102 — HumanEval+ NIM-free preflight harness (backup lane).

Runs `coordpy.humaneval_plus_preflight_v1.run_humaneval_plus_preflight_v1`
against the operator-fetched HumanEval+ corpus.  NIM-free.

Usage::

    python scripts/run_w102_humaneval_plus_preflight.py
    python scripts/run_w102_humaneval_plus_preflight.py \\
        --skip-executor-self-test
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from coordpy.humaneval_plus_preflight_v1 import (  # noqa: E402
    run_humaneval_plus_preflight_v1,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=(
        "W102 HumanEval+ NIM-free preflight harness"))
    ap.add_argument(
        "--humaneval-plus-cache",
        default=None,
        help=(
            "Optional HumanEval+ JSONL cache path; if not "
            "provided, uses COORDPY_HUMANEVAL_PLUS_CACHE env var "
            "or default ~/.cache/coordpy/humaneval-plus.jsonl"))
    ap.add_argument(
        "--bench-module-path",
        default=str(
            ROOT / "coordpy"
            / "humaneval_plus_reflexion_bench_v1.py"),
        help="Path to W102 HumanEval+ bench module")
    ap.add_argument(
        "--skip-executor-self-test", action="store_true")
    ap.add_argument(
        "--executor-self-test-sample",
        type=int, default=30)
    ap.add_argument(
        "--candidate-model",
        default="meta/llama-3.3-70b-instruct")
    ap.add_argument(
        "--out-dir",
        default=str(
            ROOT / "results" / "w102"
            / "humaneval_plus_preflight"),
        help="Output root")
    args = ap.parse_args()

    run_id = _dt.datetime.utcnow().strftime(
        "%Y%m%dT%H%M%SZ")
    out_dir = (
        Path(args.out_dir)
        / f"w102_humaneval_plus_preflight_{run_id}")
    out_dir.mkdir(parents=True, exist_ok=True)
    bench_path = Path(args.bench_module_path)
    verdict = run_humaneval_plus_preflight_v1(
        cache_path=args.humaneval_plus_cache,
        bench_module_path=bench_path,
        candidate=(
            "B (W89 sequential reflexion on HumanEval+)"),
        candidate_model=str(args.candidate_model),
        run_executor_self_test=(
            not args.skip_executor_self_test),
        executor_self_test_sample=int(
            args.executor_self_test_sample),
    )
    verdict_dict = verdict.to_dict()
    with open(out_dir / "verdict.json", "w") as f:
        json.dump(verdict_dict, f, indent=2, default=str)
    latest = out_dir.parent / "latest_run.txt"
    with open(latest, "w") as f:
        f.write(out_dir.name + "\n")
    print()
    print(
        f"  candidate: {verdict.candidate}")
    print(
        f"  candidate model: {verdict.candidate_model}")
    print()
    for p in verdict.probes:
        emoji = "PASS" if p.passed else "FAIL"
        print(f"  [{emoji}] {p.probe_id}: {p.summary}")
    print()
    print(
        f"  n_passed: {verdict.n_passed} / "
        f"{len(verdict.probes)} (required >= "
        f"{verdict.n_required})")
    print(
        f"  overall_passes: {verdict.overall_passes}")
    print(
        f"  verdict_cid: {verdict.verdict_cid}")
    print()
    print(f"  out_dir: {out_dir}")
    return 0 if verdict.overall_passes else 2


if __name__ == "__main__":
    raise SystemExit(main())
