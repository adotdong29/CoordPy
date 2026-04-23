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


RUNNER_SCHEMA = "phase45.product_report.v2"
RUNNER_SCHEMA_V1 = "phase45.product_report.v1"


def _enforce_trust(prof: dict, profile_name: str,
                    *, allow_unsafe_sandbox: bool) -> None:
    """Enforce the Docker-first default for untrusted profiles.

    Rules:
      * Untrusted + readiness/sweep sandbox != "docker"
        → refuse, unless ``allow_unsafe_sandbox=True``.
      * Untrusted + sandbox == "docker" but Docker unavailable
        → refuse, unless ``allow_unsafe_sandbox=True`` (in which
          case downgrade to "subprocess" with an explicit note).

    Trusted profiles are unaffected — they retain whatever sandbox
    they declared (subprocess / in_process / docker).
    """
    if prof.get("trust") != _profiles.TRUST_UNTRUSTED:
        return
    rd = prof.get("readiness") or {}
    sw = prof.get("sweep") or {}
    sandboxes = {rd.get("sandbox_name"), sw.get("sandbox")} - {None}
    non_docker = [s for s in sandboxes if s != "docker"]
    if non_docker and not allow_unsafe_sandbox:
        raise SystemExit(
            f"profile {profile_name!r} is UNTRUSTED and must run "
            f"inside Docker; declared sandbox(es): {sorted(sandboxes)}. "
            f"Re-run with --allow-unsafe-sandbox only if you have "
            f"audited the JSONL yourself.")
    # Docker availability probe (only if we're actually going to use it).
    if "docker" in sandboxes:
        from vision_mvp.tasks.swe_sandbox import DockerSandbox
        if not DockerSandbox().is_available():
            if not allow_unsafe_sandbox:
                raise SystemExit(
                    f"profile {profile_name!r} requires Docker but no "
                    f"Docker daemon is reachable. Install / start "
                    f"Docker, or re-run with --allow-unsafe-sandbox "
                    f"to fall back to the subprocess sandbox (weaker: "
                    f"no network isolation, no read-only rootfs).")
            # Explicit opt-out: downgrade to subprocess.
            prof["_sandbox_downgrade"] = {
                "from": "docker", "to": "subprocess",
                "reason": "docker_unavailable_with_allow_unsafe",
            }
            if rd.get("sandbox_name") == "docker":
                rd["sandbox_name"] = "subprocess"
            if sw.get("sandbox") == "docker":
                sw["sandbox"] = "subprocess"


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
                 acknowledge_heavy: bool = False,
                 allow_unsafe_sandbox: bool = False,
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

    # Trust enforcement — untrusted profiles are Docker-first.
    _enforce_trust(prof, profile_name,
                    allow_unsafe_sandbox=allow_unsafe_sandbox)

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
            # Slice 2 — unified runtime path.
            from vision_mvp.wevra.runtime import (
                sweep_spec_from_profile, run_sweep,
            )
            spec = sweep_spec_from_profile(
                profile_name,
                acknowledge_heavy=acknowledge_heavy,
                jsonl_override=jsonl_override)
            if spec is not None:
                try:
                    sweep_result = run_sweep(spec)
                except Exception as ex:
                    sweep_result = {
                        "schema": "wevra.sweep.v2",
                        "mode": sweep_cfg["mode"],
                        "executed_in_process": False,
                        "requires_acknowledgement": False,
                        "error_kind": type(ex).__name__,
                        "error_detail": str(ex)[:400],
                    }
                sweep_result.update({
                    "model_metadata": _profiles.model_availability(
                        sweep_cfg.get("model")),
                })
                artifact_name = (
                    "sweep_result.json" if sweep_result.get(
                        "executed_in_process") else "sweep_launch.json")
                with open(os.path.join(out_dir, artifact_name),
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

    # Build provenance manifest — every run carries one. Kept in the
    # core runner (not just wevra.run) so legacy invocations of
    # `python -m vision_mvp.product` also get a reproducible stamp.
    from vision_mvp.wevra.provenance import build_manifest
    rd_cfg = prof.get("readiness") or {}
    sw_cfg = prof.get("sweep") or {}
    jsonl_path = (
        jsonl_override or sw_cfg.get("jsonl") or rd_cfg.get("jsonl"))
    manifest = build_manifest(
        profile_name=profile_name,
        profile_schema=_profiles.SCHEMA_VERSION,
        jsonl_path=jsonl_path,
        model=sw_cfg.get("model"),
        endpoint=sw_cfg.get("ollama_url"),
        sandbox=(sw_cfg.get("sandbox") or rd_cfg.get("sandbox_name")),
        out_dir=out_dir,
        artifacts=None,  # filled in below after artifact list settles
        argv=sys.argv,
        extra={
            "invocation_kwargs": {
                "skip_sweep": skip_sweep,
                "force_sweep": force_sweep,
                "jsonl_override": jsonl_override,
            },
        },
    )
    prov_path = os.path.join(out_dir, "provenance.json")
    with open(prov_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    product_report["provenance"] = manifest

    # Write product_report.json + product_summary.txt after provenance
    # lands on disk, so the declared artifact list reflects the final
    # on-disk state (including provenance.json). capsule_view.json
    # is the SDK-v3 capsule graph — it lands alongside the report so
    # its CID is self-contained.
    report_path = os.path.join(out_dir, "product_report.json")
    summary_path = os.path.join(out_dir, "product_summary.txt")
    view_path = os.path.join(out_dir, "capsule_view.json")
    existing = set(os.listdir(out_dir))
    upcoming = {"product_report.json", "product_summary.txt",
                "capsule_view.json"}
    artifacts = sorted(existing | upcoming)
    product_report["artifacts"] = artifacts
    manifest["output"]["artifacts"] = artifacts

    # Fold the finished report into a Capsule DAG. This is the
    # SDK-v3 load-bearing product surface: every boundary-crossing
    # artefact from the run becomes a sealed, content-addressed,
    # typed capsule, and the RUN_REPORT capsule's CID is the
    # durable identifier for the run.
    from vision_mvp.wevra.capsule import (
        build_report_ledger, render_view,
    )
    ledger, run_cid = build_report_ledger(
        product_report, profile_dict=prof)
    view = render_view(ledger, root_cid=run_cid).as_dict()
    product_report["capsules"] = view

    with open(prov_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(product_report, fh, indent=2, default=str)
    with open(view_path, "w", encoding="utf-8") as fh:
        json.dump(view, fh, indent=2, default=str)
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
    ap.add_argument("--acknowledge-heavy", action="store_true",
                     help=("acknowledge the cost of a real-LLM sweep and "
                            "run it in-process. Without this flag, real "
                            "sweeps stage a launch command instead."))
    ap.add_argument("--allow-unsafe-sandbox", action="store_true",
                     help=("opt out of the Docker-first default for "
                            "untrusted/public JSONL profiles. Only use "
                            "this when you have audited the JSONL "
                            "yourself — weaker sandbox has no network "
                            "isolation and no read-only rootfs."))
    args = ap.parse_args()

    report = run_profile(
        args.profile, out_dir=args.out_dir,
        jsonl_override=args.jsonl,
        force_sweep=args.force_sweep,
        skip_sweep=args.skip_sweep,
        acknowledge_heavy=args.acknowledge_heavy,
        allow_unsafe_sandbox=args.allow_unsafe_sandbox)
    print(report["summary_text"])
    return 0 if report["readiness"]["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
