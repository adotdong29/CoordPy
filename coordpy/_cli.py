"""CoordPy console entry points.

Five scripts, wired into ``pyproject.toml`` ``[project.scripts]``:

    coordpy-team    — run / replay / sweep / compare an AgentTeam preset
                      (the recommended front-door CLI for new users)
    coordpy-capsule — view / verify / verify-view / audit a sealed
                      capsule chain (works on both team and RunSpec runs)
    coordpy         — run a profile, emit a provenance-stamped report
    coordpy-import  — audit a public JSONL for SWE-bench-Lite compatibility
    coordpy-ci      — consume product_report.json, emit pass/fail verdict

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
    # Accept either:
    #   (a) a product_report.json with the embedded capsule view at
    #       ``report["capsules"]`` (the runner's output shape), or
    #   (b) a bare capsule_view.json / TeamResult.capsule_view dump
    #       (the file IS the view).
    if (isinstance(report, dict)
            and report.get("schema") == "coordpy.capsule_view.v1"):
        cv = report
    else:
        cv = report.get("capsules") if isinstance(report, dict) else None
    if not isinstance(cv, dict) or cv.get("schema") != "coordpy.capsule_view.v1":
        print("error: input has no capsule view "
              "(expected coordpy.capsule_view.v1 either at the top "
              "level or under ``capsules``; got pre-SDK-v3 report "
              "or malformed JSON)",
              file=sys.stderr)
        return 2

    if args.sub == "cid":
        root_cid = cv.get("root_cid")
        if not root_cid:
            print("error: capsule view has no RUN_REPORT root capsule "
                  "(this is normal for AgentTeam.run() outputs, which "
                  "seal a TEAM_HANDOFF chain rather than a full RUN_REPORT)",
                  file=sys.stderr)
            return 2
        print(root_cid)
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


def _cmd_team(argv: list[str] | None = None) -> int:
    """``coordpy-team`` — run / replay / sweep / compare an
    :class:`AgentTeam` from the CLI.

    Subcommands:
      run     — drive a curated preset (e.g. ``quant_desk``) over a
                scenario file or stdin, dump a re-verifiable bundle.
      replay  — re-run a sealed ``team_result.json`` against a new
                backend/model. Validates per-turn prompt SHAs.
      sweep   — run the same task at multiple
                ``--max-visible-handoffs`` settings; surfaces the
                bounded-context savings as a real-numbers table.
      compare — run the task on backend A, replay against backend B,
                report side-by-side telemetry and ACTION agreement.
    """
    from .agents import replay_team_result
    from .llm_backend import backend_from_env
    from . import presets as _presets_mod
    from . import _pretty as _pp

    ap = argparse.ArgumentParser(
        prog="coordpy-team",
        description=(
            "Run, replay, sweep, or compare a CoordPy AgentTeam "
            "from the CLI. The lightweight team surface is the "
            "easiest path for external builders; this CLI removes "
            "the need to write Python to drive it."
        ),
    )
    sub = ap.add_subparsers(dest="sub")

    def _add_backend_args(p, *, suffix: str = "") -> None:
        p.add_argument(
            f"--backend{suffix}", default=None,
            choices=["ollama", "openai", "openai_compatible",
                     "provider", "mlx_distributed"],
            help=f"override COORDPY_BACKEND{suffix}.")
        p.add_argument(f"--model{suffix}", default=None,
                        help=f"override COORDPY_MODEL{suffix}.")
        p.add_argument(f"--base-url{suffix}", default=None,
                        help=f"override the backend base URL{suffix}.")
        p.add_argument(
            f"--api-key-env{suffix}", default=None,
            help=("env var holding the API key (avoids leaking the "
                  "key into argv); falls back to COORDPY_API_KEY."))

    p_run = sub.add_parser(
        "run",
        help=(
            "drive an AgentTeam preset over a task. Use "
            "--task PATH (file) or --task - (stdin)."
        ),
    )
    p_run.add_argument(
        "--preset", default="quant_desk",
        choices=["quant_desk", "code_review", "research_writer"],
        help="curated preset to run (default: quant_desk).")
    p_run.add_argument(
        "--task", required=True,
        help="path to a task/scenario .txt; pass '-' to read stdin.")
    p_run.add_argument(
        "--out-dir", required=True,
        help=("output dir for final_output.txt + team_capsule_view.json "
              "+ team_result.json + team_report.md."))
    _add_backend_args(p_run)
    p_run.add_argument("--max-visible-handoffs", type=int, default=2)
    p_run.add_argument("--max-tokens", type=int, default=360)
    p_run.add_argument("--temperature", type=float, default=0.0)
    p_run.add_argument(
        "--quiet", action="store_true",
        help="skip the per-turn live progress display.")

    p_replay = sub.add_parser(
        "replay",
        help=(
            "re-run a sealed team_result.json against a new backend/"
            "model. Records new TEAM_HANDOFF capsules."
        ),
    )
    p_replay.add_argument(
        "--result", required=True,
        help="path to team_result.json (coordpy.team_result.v1).")
    p_replay.add_argument(
        "--out-dir", required=True,
        help="output dir for the replay's bundle.")
    _add_backend_args(p_replay)
    p_replay.add_argument(
        "--quiet", action="store_true",
        help="skip the per-turn live progress display.")

    p_sweep = sub.add_parser(
        "sweep",
        help=(
            "run the same task at multiple max_visible_handoffs "
            "settings and report the bounded-context savings."
        ),
    )
    p_sweep.add_argument(
        "--preset", default="quant_desk",
        choices=["quant_desk", "code_review", "research_writer"])
    p_sweep.add_argument("--task", required=True)
    p_sweep.add_argument(
        "--out-dir", required=True,
        help="output dir; one subdir per sweep config.")
    p_sweep.add_argument(
        "--handoffs", default="1,2,8",
        help="comma-separated max_visible_handoffs values "
             "(default: 1,2,8).")
    _add_backend_args(p_sweep)
    p_sweep.add_argument("--max-tokens", type=int, default=360)
    p_sweep.add_argument("--temperature", type=float, default=0.0)
    p_sweep.add_argument(
        "--quiet", action="store_true",
        help="skip the per-turn live progress display.")

    p_compare = sub.add_parser(
        "compare",
        help=(
            "run on backend A then replay against backend B; report "
            "side-by-side telemetry and ACTION agreement."
        ),
    )
    p_compare.add_argument(
        "--preset", default="quant_desk",
        choices=["quant_desk", "code_review", "research_writer"])
    p_compare.add_argument("--task", required=True)
    p_compare.add_argument(
        "--out-dir", required=True,
        help="output dir; will write subdirs original/ and replay/.")
    _add_backend_args(p_compare)
    p_compare.add_argument(
        "--replay-backend", default=None,
        choices=["ollama", "openai", "openai_compatible",
                 "provider", "mlx_distributed"],
        help="replay-side backend.")
    p_compare.add_argument("--replay-model", default=None)
    p_compare.add_argument("--replay-base-url", default=None)
    p_compare.add_argument("--replay-api-key-env", default=None)
    p_compare.add_argument("--max-visible-handoffs", type=int, default=2)
    p_compare.add_argument("--max-tokens", type=int, default=360)
    p_compare.add_argument("--temperature", type=float, default=0.0)
    p_compare.add_argument(
        "--quiet", action="store_true",
        help="skip the per-turn live progress display.")

    args = ap.parse_args(argv)
    if args.sub is None:
        ap.print_help()
        return 1

    def _resolve_api_key(env_name: str | None) -> str | None:
        return os.environ.get(env_name) if env_name else None

    def _live(turn) -> None:
        cid = (turn.capsule_cid or "")[:12] or "-"
        head = (turn.output or "").strip().splitlines()[0] if turn.output else ""
        head = head[:96] + ("…" if len(head) > 96 else "")
        print(
            "  "
            f"{_pp.style(turn.role.ljust(22), 'cyan')} "
            f"capsule={_pp.style(cid, 'dim')}  "
            f"in={turn.prompt_tokens:>5d}  out={turn.output_tokens:>5d}  "
            f"wall={turn.wall_ms/1000:6.2f}s  vis={turn.visible_handoffs}",
            flush=True,
        )
        if head:
            print(f"  {_pp.style('↳', 'dim')} {head}", flush=True)

    def _print_team_summary(result, *, paths=None, label: str = "RUN") -> None:
        print()
        print(_pp.header(label + " — final output"))
        print((result.final_output or "").strip())
        print()
        action = result.parse_action()
        cramming = result.cramming_estimate()
        items = [
            ("backend", f"{result.backend_model} @ "
                        f"{result.backend_base_url or '(default)'}"),
            ("turns", len(result.turns)),
            ("total_tokens", f"{result.total_tokens}  "
                              f"(in={result.total_prompt_tokens}, "
                              f"out={result.total_output_tokens})"),
            ("total_wall", f"{result.total_wall_ms / 1000.0:.2f} s"),
            ("capsule_root", result.root_cid or "-"),
        ]
        if action is not None:
            verdict_color = "green" if action.action == "EXECUTE" else (
                "yellow" if action.action == "EXECUTE-WITH-MODS"
                else ("red" if action.action == "NO-ACTION" else "cyan"))
            items.append(
                ("parsed_action",
                 _pp.style(action.action, "bold", verdict_color)))
        items.extend([
            ("bounded_words", cramming["bounded_words"]),
            ("naive_words", cramming["naive_words"]),
            ("savings", _pp.style(
                f"{cramming['saved_words']} words "
                f"({cramming['savings_pct']:.1f}%)  "
                f"~{cramming['estimated_tokens_saved']} tokens",
                "bold", "green",
            )),
        ])
        print(_pp.header(label + " — telemetry"))
        print(_pp.kv(items))
        if paths:
            print()
            print(_pp.header(label + " — artefacts"))
            for label2, p in paths.items():
                print(f"  {_pp.style(label2.ljust(14), 'dim')}  {p}")

    progress = None if getattr(args, "quiet", False) else _live

    preset_factory = {
        "quant_desk": _presets_mod.quant_desk_team,
        "code_review": _presets_mod.code_review_team,
        "research_writer": _presets_mod.research_writer_team,
    }

    if args.sub == "run":
        api_key = _resolve_api_key(getattr(args, "api_key_env", None))
        task = (sys.stdin.read() if args.task == "-"
                else open(args.task, "r", encoding="utf-8").read())
        team = preset_factory[args.preset](
            model=args.model,
            backend_name=args.backend,
            base_url=args.base_url,
            api_key=api_key,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            max_visible_handoffs=args.max_visible_handoffs,
        )
        backend = team.backend
        print(_pp.header(
            f"coordpy-team run — preset={args.preset}, "
            f"max_visible_handoffs={args.max_visible_handoffs}"))
        print(_pp.kv([
            ("backend", f"{type(backend).__name__}"),
            ("model", getattr(backend, "model", "")),
            ("base_url", getattr(backend, "base_url", None) or "(default)"),
            ("temperature", args.temperature),
            ("max_tokens", args.max_tokens),
        ]))
        print()
        print(_pp.header("RUN — live turns"))
        result = team.run(task, progress=progress)
        paths = result.dump(args.out_dir)
        _print_team_summary(result, paths=paths, label="RUN")
        return 0

    if args.sub == "replay":
        api_key = _resolve_api_key(getattr(args, "api_key_env", None))
        backend = backend_from_env(
            args.backend, model=args.model,
            base_url=args.base_url, api_key=api_key)
        print(_pp.header(
            f"coordpy-team replay — model={getattr(backend, 'model', '')}"))
        print(_pp.kv([
            ("source_manifest", args.result),
            ("replay_backend", type(backend).__name__),
            ("replay_model", getattr(backend, "model", "")),
            ("replay_base_url",
             getattr(backend, "base_url", None) or "(default)"),
        ]))
        print()
        print(_pp.header("REPLAY — live turns"))
        result = replay_team_result(
            args.result, backend=backend, progress=progress)
        paths = result.dump(args.out_dir)
        _print_team_summary(result, paths=paths, label="REPLAY")
        return 0

    if args.sub == "sweep":
        api_key = _resolve_api_key(getattr(args, "api_key_env", None))
        task = (sys.stdin.read() if args.task == "-"
                else open(args.task, "r", encoding="utf-8").read())
        try:
            handoff_settings = [
                int(x.strip()) for x in args.handoffs.split(",") if x.strip()
            ]
        except ValueError:
            print("error: --handoffs must be comma-separated ints",
                  file=sys.stderr)
            return 2
        if not handoff_settings:
            print("error: --handoffs is empty", file=sys.stderr)
            return 2
        out_root = os.path.abspath(args.out_dir)
        os.makedirs(out_root, exist_ok=True)
        rows: list[list[object]] = []
        for h in handoff_settings:
            print()
            print(_pp.header(
                f"sweep config: max_visible_handoffs={h}"))
            team = preset_factory[args.preset](
                model=args.model,
                backend_name=args.backend,
                base_url=args.base_url,
                api_key=api_key,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                max_visible_handoffs=h,
            )
            print(_pp.kv([
                ("model", getattr(team.backend, "model", "")),
                ("base_url",
                 getattr(team.backend, "base_url", None) or "(default)"),
            ]))
            print()
            print(_pp.header(f"sweep h={h} — live turns"))
            result = team.run(task, progress=progress)
            sub_dir = os.path.join(out_root, f"h{h}")
            paths = result.dump(sub_dir)
            action = result.parse_action()
            cramming = result.cramming_estimate()
            rows.append([
                f"h={h}",
                result.total_prompt_tokens,
                result.total_output_tokens,
                result.total_tokens,
                f"{result.total_wall_ms / 1000.0:.1f} s",
                cramming["bounded_words"],
                cramming["naive_words"],
                f"{cramming['savings_pct']:.1f}%",
                action.action if action else "-",
            ])
            print()
            print(_pp.kv([
                ("config_dir", sub_dir),
                ("savings",
                 _pp.style(
                    f"{cramming['saved_words']} words "
                    f"({cramming['savings_pct']:.1f}%)",
                    "bold", "green")),
                ("parsed_action",
                 action.action if action else "-"),
            ]))
        print()
        print(_pp.header("sweep — comparison"))
        print(_pp.table(
            ["config", "in_tok", "out_tok", "tok_total",
             "wall", "bounded_w", "naive_w", "saved_pct", "action"],
            rows,
            align=["l", "r", "r", "r", "r", "r", "r", "r", "l"],
        ))
        return 0

    if args.sub == "compare":
        api_key = _resolve_api_key(getattr(args, "api_key_env", None))
        replay_api_key = _resolve_api_key(
            getattr(args, "replay_api_key_env", None))
        task = (sys.stdin.read() if args.task == "-"
                else open(args.task, "r", encoding="utf-8").read())
        out_root = os.path.abspath(args.out_dir)
        os.makedirs(out_root, exist_ok=True)
        team = preset_factory[args.preset](
            model=args.model,
            backend_name=args.backend,
            base_url=args.base_url,
            api_key=api_key,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            max_visible_handoffs=args.max_visible_handoffs,
        )
        print(_pp.header(
            f"compare — original on {getattr(team.backend, 'model', '')}"))
        print(_pp.kv([
            ("model_A", getattr(team.backend, "model", "")),
            ("base_url_A",
             getattr(team.backend, "base_url", None) or "(default)"),
            ("max_visible_handoffs", args.max_visible_handoffs),
        ]))
        print()
        print(_pp.header("ORIG — live turns"))
        result_a = team.run(task, progress=progress)
        paths_a = result_a.dump(os.path.join(out_root, "original"))
        action_a = result_a.parse_action()
        print()
        print(_pp.kv([
            ("orig_root", result_a.root_cid),
            ("orig_action", action_a.action if action_a else "-"),
            ("orig_tokens", result_a.total_tokens),
            ("orig_wall", f"{result_a.total_wall_ms / 1000.0:.1f} s"),
        ]))
        replay_backend = backend_from_env(
            args.replay_backend or args.backend,
            model=args.replay_model,
            base_url=args.replay_base_url,
            api_key=replay_api_key or api_key,
        )
        print()
        print(_pp.header(
            f"compare — replay on {getattr(replay_backend, 'model', '')}"))
        print(_pp.kv([
            ("model_B", getattr(replay_backend, "model", "")),
            ("base_url_B",
             getattr(replay_backend, "base_url", None) or "(default)"),
            ("source_manifest", paths_a["team_result"]),
        ]))
        print()
        print(_pp.header("REPLAY — live turns"))
        from .agents import replay_team_result as _replay
        result_b = _replay(
            paths_a["team_result"], backend=replay_backend,
            progress=progress)
        paths_b = result_b.dump(os.path.join(out_root, "replay"))
        action_b = result_b.parse_action()
        prompt_match = all(
            t1.prompt_sha256 == t2.prompt_sha256
            for t1, t2 in zip(result_a.turns, result_b.turns)
        )
        action_match = (
            action_a is not None and action_b is not None
            and action_a.action == action_b.action
        )
        print()
        print(_pp.header("compare — side-by-side"))
        print(_pp.table(
            ["metric", "ORIG (A)", "REPLAY (B)"],
            [
                ["model", result_a.backend_model,
                 result_b.backend_model],
                ["prompt_tokens",
                 result_a.total_prompt_tokens,
                 result_b.total_prompt_tokens],
                ["output_tokens",
                 result_a.total_output_tokens,
                 result_b.total_output_tokens],
                ["total_tokens",
                 result_a.total_tokens, result_b.total_tokens],
                ["total_wall",
                 f"{result_a.total_wall_ms / 1000.0:.1f} s",
                 f"{result_b.total_wall_ms / 1000.0:.1f} s"],
                ["root_cid",
                 (result_a.root_cid or "-")[:18] + "…",
                 (result_b.root_cid or "-")[:18] + "…"],
                ["parsed_action",
                 action_a.action if action_a else "-",
                 action_b.action if action_b else "-"],
            ],
            align=["l", "r", "r"],
        ))
        print()
        print(_pp.kv([
            ("prompt_sha_match (all turns)",
             _pp.style("yes" if prompt_match else "NO",
                       "bold", "green" if prompt_match else "red")),
            ("action_agreement",
             _pp.style("yes" if action_match else "NO",
                       "bold", "green" if action_match else "yellow")),
            ("orig_dir", os.path.join(out_root, "original")),
            ("replay_dir", os.path.join(out_root, "replay")),
        ]))
        return 0

    ap.print_help()
    return 1


def main_run() -> int:
    return _cmd_run()


def main_import() -> int:
    return _cmd_import()


def main_ci() -> int:
    return _cmd_ci()


def main_capsule() -> int:
    return _cmd_capsule()


def main_team() -> int:
    return _cmd_team()


if __name__ == "__main__":
    raise SystemExit(main_run())
