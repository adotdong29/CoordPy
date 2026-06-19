"""W88 — per-task outcome inspector.

Reads a W88 HumanEval-reflexion or cross-modal-code bench report
+ per-call sidecar and prints a per-(seed, task) table of which
arm passed/failed.  Useful for postmortem analysis ("did B win
because of a few rescues, or across the board?").

Usage:

    python scripts/inspect_w88_per_task_outcomes.py \
        --run-dir <path-to-W88-run-dir>

If no --run-dir is given, the inspector reads the bench
``latest_run.txt`` pointer.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _outcomes_by_arm_task(report: dict) -> dict[
        str, dict[tuple[int, str], bool]]:
    """Group outcomes by (seed, arm) -> {task_id: passed}.

    The report does not carry the per-task pass flag directly —
    only per-task outcome CIDs.  We instead read the per-call
    sidecar's executor result entries (the bench writes those
    via the wrapped gen, not directly in the sidecar).  V1 of
    this inspector reads what's in the report only and reports
    the AGGREGATE per-seed pass rate; full per-task is
    reconstructed by re-running the executor offline.
    """
    out: dict[str, dict[tuple[int, str], bool]] = (
        defaultdict(dict))
    return out


def _print_per_seed_summary(report: dict) -> None:
    print()
    print(f"{'seed':>10} {'A0':>8} {'A1':>8} "
          f"{'B':>8} {'B-A1':>8}")
    print("-" * 50)
    for ps in report.get("per_seed", []):
        seed = int(ps["seed"])
        if "a0_pass_at_1" in ps:
            # HumanEval-reflexion report
            a0 = float(ps["a0_pass_at_1"])
            a1 = float(ps["a1_pass_at_1"])
            b = float(ps["b_pass_at_1"])
        else:
            # Cross-modal report
            a0 = float(ps["a0_text_pass_at_1"])
            a1 = float(ps["a1_vlm_pass_at_1"])
            b = float(ps["b_cross_pass_at_1"])
        print(
            f"{seed:>10} {a0:>8.4f} {a1:>8.4f} "
            f"{b:>8.4f} {(b - a1) * 100:+7.2f}pp")
    if "a0_mean_pass_at_1" in report:
        a0m = float(report["a0_mean_pass_at_1"])
        a1m = float(report["a1_mean_pass_at_1"])
        bm = float(report["b_mean_pass_at_1"])
    else:
        a0m = float(report["a0_text_mean_pass_at_1"])
        a1m = float(report["a1_vlm_mean_pass_at_1"])
        bm = float(report["b_cross_mean_pass_at_1"])
    print("-" * 50)
    print(
        f"{'MEAN':>10} {a0m:>8.4f} {a1m:>8.4f} {bm:>8.4f} "
        f"{(bm - a1m) * 100:+7.2f}pp")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="W88 per-task outcome inspector")
    parser.add_argument(
        "--run-dir",
        help="Run directory containing bench report JSON")
    parser.add_argument(
        "--report",
        help="Explicit report JSON path")
    args = parser.parse_args()

    if args.report:
        report_path = Path(args.report)
    elif args.run_dir:
        d = Path(args.run_dir)
        # Try both possible report names.
        candidates = [
            d / "humaneval_reflexion_bench_report.json",
            d / "cross_modal_code_bench_report.json",
        ]
        report_path = None
        for c in candidates:
            if c.exists():
                report_path = c
                break
        if report_path is None:
            raise SystemExit(f"no bench report found in {d}")
    else:
        # Try W88 humaneval reflexion first, then cross-modal
        for sub in (
                "humaneval_reflexion", "cross_modal_code"):
            ptr = (
                ROOT / "results" / "w88" / sub
                / "latest_run.txt")
            if ptr.exists():
                pointer_value = ptr.read_text().strip()
                candidate = Path(pointer_value)
                if not candidate.is_absolute():
                    candidate = ptr.parent / candidate
                d = candidate
                for c in (
                        d / "humaneval_reflexion_bench_report.json",
                        d / "cross_modal_code_bench_report.json"):
                    if c.exists():
                        report_path = c
                        break
                if report_path is not None:
                    break
        else:
            raise SystemExit(
                "no --run-dir / --report given and no W88 "
                "latest_run.txt found")

    print(f"report = {report_path}")
    with open(report_path) as f:
        report = json.load(f)
    print(
        f"schema = "
        f"{report.get('schema', '<unknown>')}")
    if "model_id" in report:
        print(f"model  = {report['model_id']}")
    elif "vlm_model_id" in report:
        print(
            f"vlm    = {report['vlm_model_id']}\n"
            f"code   = {report['code_model_id']}")
    print(
        f"n_seeds={report.get('n_seeds')} "
        f"n_problems={report.get('n_problems')} "
        f"K={report.get('K_multi_sample')}")
    _print_per_seed_summary(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
