"""Phase-45 one-command product runner.

Usage (as a module):

    python3 -m vision_mvp.product --profile bundled_57 \\
            --out-dir vision_mvp/artifacts/rc1

    python3 -m vision_mvp.product --profile local_smoke --out-dir /tmp/cz-smoke
    python3 -m vision_mvp.product --profile aspen_mac1_coder \\
            --out-dir vision_mvp/artifacts/mac1
    python3 -m vision_mvp.product --profile public_jsonl \\
            --jsonl /path/to/swe_bench_lite.jsonl \\
            --out-dir vision_mvp/artifacts/public_lite

The runner does three things, in order, and emits a single
``product_report.json`` + ``product_summary.txt`` into ``--out-dir``:

  1. **Readiness validation** (`phase44_public_readiness`).
     If any of the five checks fail on any row, the run stops
     unless ``--force-sweep`` is set.
  2. **Parser sweep** (`phase42`-shape, optionally with raw capture),
     if the profile declares a ``sweep`` block. Mock mode is run
     in-process; real mode is run in-process if `--mode real` is
     requested and an LLM client is reachable.
  3. **Report emission** — a machine-readable JSON verdict + a
     human-readable summary with pass/fail, parser/matcher/semantic
     attribution, model + profile metadata, and paths to all
     artifacts.

The runner never rewrites the per-phase experiment scripts; it
imports the same primitives and reports on them. Lower-level
scripts remain available and unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")))


from vision_mvp.product import profiles as _profiles
from vision_mvp.product import report as _report
from vision_mvp.experiments.phase44_public_readiness import run_readiness
from vision_mvp.tasks.swe_bench_bridge import (
    ALL_SWE_STRATEGIES, ParserComplianceCounter,
    build_synthetic_event_log, deterministic_oracle_generator,
    load_jsonl_bank, parse_unified_diff,
)
from vision_mvp.tasks.swe_patch_parser import (
    PARSER_STRICT, parse_patch_block,
)
from vision_mvp.tasks.swe_sandbox import (
    run_swe_loop_sandboxed, select_sandbox,
)


RUNNER_SCHEMA = "phase45.product_report.v1"


def _mock_sweep(sweep_cfg: dict) -> dict:
    """Run a mock-oracle parser sweep in-process. Returns a report dict
    suitable for embedding into the product report."""
    jsonl_path = sweep_cfg["jsonl"]
    nd_list = list(sweep_cfg["n_distractors"])
    parser_modes = list(sweep_cfg["parser_modes"])
    apply_modes = list(sweep_cfg["apply_modes"])
    strategies = tuple(ALL_SWE_STRATEGIES)
    sandbox = select_sandbox(sweep_cfg["sandbox"])

    cells = []
    for parser_mode in parser_modes:
        for apply_mode in apply_modes:
            for nd in nd_list:
                tasks, repo_files = load_jsonl_bank(
                    jsonl_path,
                    hidden_event_log_factory=(
                        lambda t, k=nd: build_synthetic_event_log(t, k)),
                    limit=sweep_cfg["n_instances"],
                )
                counter = ParserComplianceCounter()
                # oracle generator ignores parser_mode, but we still
                # capture the counter shape for schema stability.
                rep = run_swe_loop_sandboxed(
                    bank=tasks, repo_files=repo_files,
                    generator=deterministic_oracle_generator,
                    sandbox=sandbox, strategies=strategies,
                    timeout_s=15.0, apply_mode=apply_mode)
                pooled = rep.pooled_summary()
                cells.append({
                    "parser_mode": parser_mode,
                    "apply_mode": apply_mode,
                    "n_distractors": nd,
                    "pooled": pooled,
                    "parser_compliance": counter.as_dict(),
                    "n_instances": len(tasks),
                })
    return {"mode": "mock", "cells": cells,
            "jsonl": jsonl_path, "sandbox": sandbox.name()}


def _real_sweep_stub(sweep_cfg: dict) -> dict:
    """Real-LLM sweep is heavy — the runner records the resolved
    command line for the operator to launch on the ASPEN cluster,
    rather than silently forking a multi-hour job from inside a
    product-runner invocation. This is deliberate: we preserve the
    Phase-42/44 scripts as the *execution* surface for real-model
    runs and make the product runner the *orchestration* surface."""
    cmd = [
        sys.executable, "-m",
        "vision_mvp.experiments.phase42_parser_sweep",
        "--mode", "real",
        "--model", str(sweep_cfg["model"]),
        "--ollama-url", str(sweep_cfg["ollama_url"]),
        "--jsonl", sweep_cfg["jsonl"],
        "--parser-modes", *sweep_cfg["parser_modes"],
        "--apply-modes", *sweep_cfg["apply_modes"],
        "--n-distractors", *[str(x) for x in sweep_cfg["n_distractors"]],
        "--sandbox", sweep_cfg["sandbox"],
    ]
    if sweep_cfg.get("n_instances") is not None:
        cmd += ["--n-instances", str(sweep_cfg["n_instances"])]
    capture_cmd = None
    if sweep_cfg.get("enable_raw_capture"):
        capture_cmd = cmd[:]
        capture_cmd[2] = "vision_mvp.experiments.phase44_semantic_residue"
    return {
        "mode": "real",
        "executed_in_process": False,
        "launch_cmd": cmd,
        "raw_capture_launch_cmd": capture_cmd,
        "model_metadata": _profiles.model_availability(
            sweep_cfg.get("model")),
        "note": ("Heavy real-model runs are launched via the existing "
                  "phase42/phase44 CLI. The product runner records the "
                  "resolved command line so the operator can kick it off "
                  "on the correct cluster node."),
    }


def run_profile(profile_name: str, *,
                 out_dir: str,
                 jsonl_override: str | None = None,
                 force_sweep: bool = False,
                 skip_sweep: bool = False,
                 ) -> dict:
    t0 = time.time()
    prof = _profiles.get_profile(profile_name)
    if jsonl_override:
        prof["readiness"]["jsonl"] = jsonl_override
        if prof.get("sweep"):
            prof["sweep"]["jsonl"] = jsonl_override
    if not prof["readiness"]["jsonl"]:
        raise SystemExit(
            f"profile {profile_name!r} requires --jsonl <path>")

    os.makedirs(out_dir, exist_ok=True)

    readiness_verdict = run_readiness(
        prof["readiness"]["jsonl"],
        limit=prof["readiness"]["limit"],
        sandbox_name=prof["readiness"]["sandbox_name"])
    with open(os.path.join(out_dir, "readiness_verdict.json"),
               "w", encoding="utf-8") as fh:
        json.dump(readiness_verdict, fh, indent=2, default=str)

    sweep_result: dict[str, Any] | None = None
    sweep_cfg = prof.get("sweep")
    if sweep_cfg and not skip_sweep:
        if not readiness_verdict["ready"] and not force_sweep:
            sweep_result = {
                "skipped": True,
                "reason": "readiness_not_ready",
                "blockers": readiness_verdict["blockers"],
            }
        else:
            if sweep_cfg["mode"] == "mock":
                sweep_result = _mock_sweep(sweep_cfg)
                with open(os.path.join(out_dir,
                                          "sweep_result.json"),
                           "w", encoding="utf-8") as fh:
                    json.dump(sweep_result, fh, indent=2, default=str)
            else:
                sweep_result = _real_sweep_stub(sweep_cfg)
                with open(os.path.join(out_dir,
                                          "sweep_launch.json"),
                           "w", encoding="utf-8") as fh:
                    json.dump(sweep_result, fh, indent=2, default=str)

    product_report = {
        "schema": RUNNER_SCHEMA,
        "profile": profile_name,
        "profile_description": prof["description"],
        "readiness": readiness_verdict,
        "sweep": sweep_result,
        "wall_seconds": round(time.time() - t0, 2),
        "out_dir": os.path.abspath(out_dir),
    }
    # Write product_report.json + product_summary.txt first so the
    # declared artifact list reflects the final on-disk state.
    report_path = os.path.join(out_dir, "product_report.json")
    summary_path = os.path.join(out_dir, "product_summary.txt")
    # Snapshot present + upcoming artifacts.
    existing = set(os.listdir(out_dir))
    upcoming = {"product_report.json", "product_summary.txt"}
    product_report["artifacts"] = sorted(existing | upcoming)
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(product_report, fh, indent=2, default=str)
    summary_text = _report.render_summary(product_report)
    with open(summary_path, "w", encoding="utf-8") as fh:
        fh.write(summary_text)
    product_report["summary_text"] = summary_text
    return product_report


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Phase-45 one-command product runner.")
    ap.add_argument("--profile", required=True,
                     choices=_profiles.list_profiles())
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--jsonl", default=None,
                     help="override profile JSONL (required for "
                           "public_jsonl)")
    ap.add_argument("--skip-sweep", action="store_true",
                     help="run readiness only; skip the sweep block")
    ap.add_argument("--force-sweep", action="store_true",
                     help="run the sweep even if readiness is not ready")
    args = ap.parse_args()

    report = run_profile(
        args.profile, out_dir=args.out_dir,
        jsonl_override=args.jsonl,
        force_sweep=args.force_sweep,
        skip_sweep=args.skip_sweep)
    print(report["summary_text"])
    return 0 if report["readiness"]["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
