"""CoordPy console entry points.

Three scripts, wired into ``pyproject.toml`` ``[project.scripts]``:

    coordpy        — run a profile, emit a provenance-stamped report
    coordpy-import — audit a public JSONL for SWE-bench-Lite compatibility
    coordpy-ci     — consume product_report.json, emit pass/fail verdict

These are thin wrappers over the already-stable product modules.
They intentionally do *not* expose experimental knobs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .config import CoordPyConfig
from .run import RunSpec, run as coordpy_run


def _cmd_run(argv: list[str] | None = None) -> int:
    from coordpy._internal.product import profiles as _profiles
    # ``--version`` is a pre-parse short-circuit so it works without
    # ``--profile`` / ``--out-dir`` (both of which are otherwise
    # required). This matches the shape exercised by the CI workflow.
    if argv is None:
        argv = sys.argv[1:]
    if "--version" in argv:
        from . import __version__, SDK_VERSION
        print(f"coordpy {__version__} ({SDK_VERSION})")
        return 0
    ap = argparse.ArgumentParser(
        prog="coordpy",
        description=(
            "CoordPy — run a profile and emit a reproducible, "
            "provenance-stamped product report.\n\n"
            "Trust model: profiles tagged ``untrusted`` "
            "(e.g. public_jsonl) execute inside Docker by default "
            "(--network=none, read-only rootfs). Pass "
            "--allow-unsafe-sandbox only for JSONL you have audited "
            "yourself.\n\n"
            "Provider config: prefer COORDPY_BACKEND, "
            "COORDPY_API_BASE_URL, COORDPY_API_KEY "
            "(legacy COORDPY_LLM_* names and OPENAI_* fallbacks "
            "are also supported for OpenAI-compatible providers)."))
    ap.add_argument("--profile", required=True,
                     choices=_profiles.list_profiles())
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--jsonl", default=None,
                     help="override profile JSONL")
    ap.add_argument("--model", default=None,
                     help="override the profile's model tag for real sweeps")
    ap.add_argument("--backend", default=None,
                     choices=("ollama", "openai", "openai_compatible",
                              "provider",
                              "mlx_distributed"),
                     help=("LLM backend for real sweeps. Defaults to "
                           "CoordPy env/config resolution."))
    ap.add_argument("--base-url", default=None,
                     help=("override the backend base URL. For OpenAI-"
                           "compatible providers this should be the "
                           "root URL hosting /v1/chat/completions."))
    ap.add_argument("--api-key-env", default=None,
                     help=("read the provider API key from this "
                           "environment variable instead of "
                           "COORDPY_API_KEY / OPENAI_API_KEY"))
    ap.add_argument("--skip-sweep", action="store_true")
    ap.add_argument("--force-sweep", action="store_true")
    ap.add_argument("--acknowledge-heavy", action="store_true",
                     help=("acknowledge the cost of a real-LLM sweep and "
                            "run it in-process. Without this flag, real "
                            "sweeps stage a launch command instead of "
                            "executing."))
    ap.add_argument("--allow-unsafe-sandbox", action="store_true",
                     help=("opt out of the Docker-first default for "
                            "untrusted/public JSONL profiles. Without "
                            "this flag, untrusted profiles refuse to "
                            "run on anything weaker than Docker."))
    ap.add_argument("--report-sink", action="append", default=[],
                     help=("name of a registered ReportSink extension "
                            "to emit to after the run. Repeatable. "
                            "Built-ins: stdout, jsonfile."))
    ap.add_argument("--version", action="store_true",
                     help="print CoordPy SDK version and exit")
    args = ap.parse_args(argv)
    if args.version:
        from . import __version__, SDK_VERSION
        print(f"coordpy {__version__} ({SDK_VERSION})")
        return 0
    api_key = None
    if args.api_key_env:
        api_key = os.environ.get(args.api_key_env)
        if api_key is None:
            ap.error(
                f"--api-key-env {args.api_key_env!r} is not set in the environment")
    config = CoordPyConfig.from_env(
        model=args.model,
        llm_backend=args.backend,
        llm_base_url=args.base_url,
        llm_api_key=api_key,
    )
    spec = RunSpec(
        profile=args.profile,
        out_dir=args.out_dir,
        jsonl_override=args.jsonl,
        skip_sweep=args.skip_sweep,
        force_sweep=args.force_sweep,
        acknowledge_heavy=args.acknowledge_heavy,
        allow_unsafe_sandbox=args.allow_unsafe_sandbox,
        report_sinks=tuple(args.report_sink),
        config=config,
    )
    report = coordpy_run(spec)
    print(report.get("summary_text", ""))
    return 0 if report["readiness"]["ready"] else 1


def _cmd_import(argv: list[str] | None = None) -> int:
    from coordpy._internal.product.import_data import audit_jsonl, _render_summary
    ap = argparse.ArgumentParser(
        prog="coordpy-import",
        description="Audit a public JSONL for SWE-bench-Lite compatibility.")
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sandbox", choices=("in_process", "subprocess"),
                     default="subprocess")
    ap.add_argument("--skip-readiness", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    report = audit_jsonl(
        args.jsonl, limit=args.limit,
        run_readiness_check=not args.skip_readiness,
        sandbox_name=args.sandbox)
    print(_render_summary(report))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, default=str)
        print(f"Wrote {args.out}")
    if report.get("error_kind") == "file_not_found":
        return 2
    return 0 if report["ok"] else 1


def _cmd_ci(argv: list[str] | None = None) -> int:
    from coordpy._internal.product.ci_gate import (
        evaluate_report, aggregate, _render_verdict,
    )
    ap = argparse.ArgumentParser(
        prog="coordpy-ci",
        description="Consume product_report.json, emit pass/fail verdict.")
    ap.add_argument("--report", nargs="+", required=True)
    ap.add_argument("--require-profile", nargs="+", default=None)
    ap.add_argument("--allow-profile", nargs="+", default=None)
    ap.add_argument("--min-ready-fraction", type=float, default=1.0)
    ap.add_argument("--min-pass-at-1", type=float, default=1.0)
    ap.add_argument("--allow-not-ready", action="store_true")
    ap.add_argument("--require-sweep-executed", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    verdicts = []
    for p in args.report:
        v = evaluate_report(
            p,
            allow_profiles=tuple(args.allow_profile)
                if args.allow_profile else None,
            require_profiles=tuple(args.require_profile)
                if args.require_profile else None,
            min_ready_fraction=args.min_ready_fraction,
            min_pass_at_1=args.min_pass_at_1,
            allow_not_ready=args.allow_not_ready,
            require_sweep_executed=args.require_sweep_executed)
        verdicts.append(v)
        print(_render_verdict(v))
    agg = aggregate(verdicts)
    print(f"=== aggregate ===\nok={agg['ok']}  "
           f"n_reports={agg['n_reports']}  "
           f"aggregate_blockers={len(agg['aggregate_blockers'])}")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(agg, fh, indent=2, default=str)
        print(f"Wrote {args.out}")
    return 0 if agg["ok"] else 1


def _cmd_capsule(argv: list[str] | None = None) -> int:
    """``coordpy-capsule`` — inspect / audit the capsule graph of a
    finished product report.

    Subcommands:
      view   — print a short capsule-graph summary (default).
      verify — re-hash the chain in capsule_view.json and print OK / BAD.
      cid    — print the RUN_REPORT capsule's CID for the report.
    """
    ap = argparse.ArgumentParser(
        prog="coordpy-capsule",
        description=(
            "Inspect the capsule graph that every CoordPy run emits. "
            "Capsules are the SDK-v3 load-bearing unit: every "
            "boundary-crossing artefact is a typed, content-addressed, "
            "lifecycle-bounded, provenance-stamped capsule."))
    sub = ap.add_subparsers(dest="sub")
    p_view = sub.add_parser(
        "view", help="summarise the capsule graph of a report")
    p_view.add_argument("--report", required=True,
                          help="path to product_report.json")
    p_view.add_argument("--full", action="store_true",
                          help="print every capsule header, not just stats")
    p_verify = sub.add_parser(
        "verify", help="re-hash the capsule chain and verify C5")
    p_verify.add_argument("--report", required=True)
    p_cid = sub.add_parser(
        "cid", help="print the RUN_REPORT capsule CID for a report")
    p_cid.add_argument("--report", required=True)
    p_audit = sub.add_parser(
        "audit",
        help=("run the SDK v3.3 lifecycle audit on a finished "
              "capsule view (eight invariants L-1..L-8). Prints "
              "OK / BAD plus typed counterexamples on violation."))
    p_audit.add_argument("--report", required=True)
    args = ap.parse_args(argv)

    if args.sub is None:
        ap.print_help()
        return 1

    try:
        with open(args.report, "r", encoding="utf-8") as fh:
            report = json.load(fh)
    except FileNotFoundError:
        print(f"error: report not found: {args.report}", file=sys.stderr)
        return 2
    except json.JSONDecodeError as e:
        print(f"error: report is not valid JSON: {args.report}: {e}",
              file=sys.stderr)
        return 2
    cv = report.get("capsules")
    if not isinstance(cv, dict) or cv.get("schema") != "coordpy.capsule_view.v1":
        print("error: report has no capsule view "
              "(pre-SDK-v3 report or malformed)",
              file=sys.stderr)
        return 2

    if args.sub == "cid":
        print(cv.get("root_cid") or "")
        return 0

    if args.sub == "audit":
        # SDK v3.3 lifecycle audit: mechanically verifies eight
        # invariants L-1..L-8 over the capsule view. Returns a
        # short summary + typed counterexamples on BAD.
        from .lifecycle_audit import audit_capsule_lifecycle_from_view
        report_audit = audit_capsule_lifecycle_from_view(cv)
        print(f"verdict        = {report_audit.verdict}")
        print(f"rules_passed   = "
               f"{len(report_audit.rules_passed)} / "
               f"{len(report_audit.rules_checked)}")
        print(f"violations     = {len(report_audit.violations)}")
        print(f"by_kind        = {report_audit.stats}")
        if report_audit.violations:
            print()
            print("counterexamples (first 8):")
            for v in report_audit.violations[:8]:
                print(f"  rule={v['rule']:<48s} cid="
                       f"{(v['capsule_cid'] or '-')[:16]:<16s} "
                       f"kind={v['capsule_kind']:<14s} "
                       f"detail={v['detail'][:140]}")
        return 0 if report_audit.verdict in ("OK", "EMPTY") else 4

    if args.sub == "verify":
        # SDK v3.2: stronger verify. Four independent on-disk
        # checks, each printed with its own verdict:
        #
        #   1. Chain-from-headers recompute. The embedded view's
        #      chain_head is recomputed from the headers on disk
        #      via ``verify_chain_from_view_dict`` — a tamper
        #      that flips a CID, a kind, or the order of capsules,
        #      or that rewrites chain_head, is detected.
        #   2. View on-disk vs embedded agreement. The on-disk
        #      ``capsule_view.json`` chain_head must match the
        #      report's embedded chain_head.
        #   3. ARTIFACT capsule on-disk re-hash. For each ARTIFACT
        #      capsule with a real sha256 in its payload, the
        #      on-disk file is re-read and re-hashed; any drift is
        #      reported as a mismatch.
        #   4. META_MANIFEST verification. If
        #      ``meta_manifest.json`` exists, each meta-artefact
        #      named in the manifest is re-read and re-hashed.
        #
        # Overall verdict is OK iff all four checks pass.
        from .capsule import verify_chain_from_view_dict
        from .capsule_runtime import (
            verify_artifacts_on_disk, verify_meta_manifest_on_disk,
        )
        out_dir = os.path.dirname(os.path.abspath(args.report))
        view_path = os.path.join(out_dir, "capsule_view.json")
        embedded_head = cv.get("chain_head")
        ok_embedded = bool(cv.get("chain_ok"))

        # Check 1: chain from headers (embedded).
        chain_recompute_ok = verify_chain_from_view_dict(cv)

        # Check 2: on-disk view agreement.
        agree = True
        disk_chain_recompute_ok = True
        if os.path.exists(view_path):
            with open(view_path, "r", encoding="utf-8") as fh:
                disk_view = json.load(fh)
            agree = disk_view.get("chain_head") == embedded_head
            disk_chain_recompute_ok = verify_chain_from_view_dict(
                disk_view)
        else:
            disk_view = cv

        # Check 3: ARTIFACT bytes on disk.
        artifact_check = verify_artifacts_on_disk(
            disk_view, base_dir=out_dir)

        # Check 4: META_MANIFEST.
        manifest_path = os.path.join(out_dir, "meta_manifest.json")
        manifest_check = None
        if os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as fh:
                manifest = json.load(fh)
            manifest_check = verify_meta_manifest_on_disk(
                manifest, base_dir=out_dir)

        all_checks_ok = (
            ok_embedded
            and chain_recompute_ok
            and disk_chain_recompute_ok
            and agree
            and artifact_check["verdict"] in ("OK", "EMPTY")
            and (manifest_check is None
                 or manifest_check["verdict"] in ("OK", "EMPTY"))
        )
        print(f"chain_ok_embedded         = {ok_embedded}")
        print(f"chain_recompute_embedded  = {chain_recompute_ok}")
        print(f"chain_recompute_on_disk   = {disk_chain_recompute_ok}")
        print(f"on_disk_view_agrees       = {agree}")
        print(f"artifacts_on_disk         = {artifact_check['verdict']}"
               f"  ({artifact_check['ok']}/{artifact_check['checked']}"
               f" matched, {len(artifact_check['mismatch'])} drifts,"
               f" {len(artifact_check['missing'])} missing)")
        if manifest_check is not None:
            print(f"meta_manifest_on_disk     = "
                   f"{manifest_check['verdict']}"
                   f"  ({manifest_check['ok']}/{manifest_check['checked']}"
                   f" matched, {len(manifest_check['mismatch'])} drifts,"
                   f" {len(manifest_check['missing'])} missing)")
        else:
            print("meta_manifest_on_disk     = ABSENT")
        print(f"verdict                   = "
               f"{'OK' if all_checks_ok else 'BAD'}")
        if artifact_check["mismatch"]:
            for d in artifact_check["mismatch"][:8]:
                print(f"  drift: {d['path']!r} "
                       f"sealed={d['sealed_sha256'][:16]}… "
                       f"on_disk={d['on_disk_sha256'][:16]}…")
        return 0 if all_checks_ok else 3

    # view
    stats = cv.get("stats") or {}
    print(f"=== coordpy capsule graph: {args.report} ===")
    print(f"  root_cid   : {cv.get('root_cid') or '-'}")
    print(f"  chain_head : {cv.get('chain_head') or '-'}")
    print(f"  chain_ok   : {cv.get('chain_ok')}")
    print(f"  n_entries  : {stats.get('n_entries', '-')}")
    print(f"  by_kind    : {stats.get('by_kind', {})}")
    print(f"  by_lifecyc : {stats.get('by_lifecycle', {})}")
    if args.full:
        print()
        print("  capsules:")
        for cap in cv.get("capsules", []):
            cid = (cap.get("cid") or "")[:16]
            kind = cap.get("kind", "-")
            life = cap.get("lifecycle", "-")
            nt = cap.get("n_tokens")
            nb = cap.get("n_bytes")
            parents = len(cap.get("parents") or [])
            print(f"    {cid:<18s} {kind:<18s} {life:<9s} "
                  f"parents={parents:<3d} "
                  f"tokens={nt!r:<6s} bytes={nb!r}")
    return 0


def main_run() -> int:
    return _cmd_run()


def main_import() -> int:
    return _cmd_import()


def main_ci() -> int:
    return _cmd_ci()


def main_capsule() -> int:
    return _cmd_capsule()


if __name__ == "__main__":
    raise SystemExit(main_run())
