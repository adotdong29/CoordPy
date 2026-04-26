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
                 capsule_native: bool = True,
                 deterministic: bool = False,
                 ) -> dict:
    """Execute one Wevra run.

    ``capsule_native`` (default True): drive the run through a
    ``CapsuleNativeRunContext`` so each boundary-crossing artefact
    becomes a sealed capsule *in flight*. The PROFILE capsule is
    sealed first; readiness, sweep_spec, sweep cells, provenance,
    and substantive on-disk artefacts seal under their parent CIDs
    as the run progresses; the RUN_REPORT capsule is sealed before
    the meta-artefacts (``product_report.json``,
    ``capsule_view.json``, ``product_summary.txt``) are written.
    Mid-run failure leaves a typed witness in
    ``ctx.in_flight_failures()``.

    When ``capsule_native=False``, the legacy post-hoc fold is
    used: the run executes as ordinary Python and
    ``build_report_ledger`` synthesises the capsule DAG after the
    fact. The two paths produce CID-equivalent ledgers for the
    non-artefact kinds (Theorem W3-34); they differ only in the
    ARTIFACT kind, where the in-flight path emits content-addressed
    capsules with real SHA-256 hashes and the post-hoc path emits
    capsules with ``sha256=None``.

    ``deterministic`` (default False, SDK v3.3): strip per-run
    timestamps from PROVENANCE / RUN_REPORT capsule payloads so
    two runs of the same deterministic profile produce identical
    full-DAG CIDs (Theorem W3-41). The legacy
    timestamp-bearing-CIDs path is the default for normal
    operations; deterministic mode is the audit / replay opt-in.
    """
    if capsule_native:
        return _run_profile_capsule_native(
            profile_name,
            out_dir=out_dir,
            jsonl_override=jsonl_override,
            force_sweep=force_sweep,
            skip_sweep=skip_sweep,
            acknowledge_heavy=acknowledge_heavy,
            allow_unsafe_sandbox=allow_unsafe_sandbox,
            deterministic=deterministic,
        )
    return _run_profile_post_hoc(
        profile_name,
        out_dir=out_dir,
        jsonl_override=jsonl_override,
        force_sweep=force_sweep,
        skip_sweep=skip_sweep,
        acknowledge_heavy=acknowledge_heavy,
        allow_unsafe_sandbox=allow_unsafe_sandbox,
        deterministic=deterministic,
    )


def _resolve_profile_and_setup(profile_name: str, *,
                                out_dir: str,
                                jsonl_override: str | None,
                                allow_unsafe_sandbox: bool,
                                ) -> dict:
    """Shared setup for both run paths: profile resolution, JSONL
    override, trust enforcement, output-directory creation."""
    prof = _profiles.get_profile(profile_name)
    if jsonl_override:
        prof["readiness"]["jsonl"] = jsonl_override
        if prof.get("sweep"):
            prof["sweep"]["jsonl"] = jsonl_override
    if not prof["readiness"]["jsonl"]:
        raise SystemExit(
            f"profile {profile_name!r} requires --jsonl <path>")
    _enforce_trust(prof, profile_name,
                    allow_unsafe_sandbox=allow_unsafe_sandbox)
    os.makedirs(out_dir, exist_ok=True)
    return prof


_DETERMINISTIC_TIMESTAMP_UTC = "1970-01-01T00:00:00+00:00"
_DETERMINISTIC_WALL = 0.0
_DETERMINISTIC_HOST = "deterministic"
_DETERMINISTIC_USER = "deterministic"
_DETERMINISTIC_OUT_DIR = "/deterministic/out_dir"


def _canonicalise_for_determinism(manifest: dict, *,
                                    profile_name: str,
                                    jsonl_path: str | None,
                                    ) -> dict:
    """Strip per-run, host-local, and wall-clock fields from a
    provenance manifest so two runs of the same deterministic
    profile produce byte-identical PROVENANCE capsule payloads.

    Theorem W3-41 (deterministic-mode CID determinism): the set of
    fields stripped here is exactly the set of fields whose values
    are not a deterministic function of (profile, JSONL bytes,
    parser/apply mode, n_distractors, sandbox name). After
    stripping, the PROVENANCE capsule's CID is a deterministic
    function of the run's logical inputs.
    """
    out = dict(manifest)
    out["timestamp_utc"] = _DETERMINISTIC_TIMESTAMP_UTC
    code = dict(out.get("code", {}))
    code["git_sha"] = "deterministic"
    code["git_dirty"] = False
    code["repo_dir"] = "/deterministic/repo"
    out["code"] = code
    runtime_block = dict(out.get("runtime", {}))
    runtime_block["python_version"] = "deterministic"
    runtime_block["python_implementation"] = "deterministic"
    runtime_block["platform"] = "deterministic"
    runtime_block["machine"] = "deterministic"
    runtime_block["system"] = "deterministic"
    out["runtime"] = runtime_block
    invocation = dict(out.get("invocation", {}))
    invocation["argv"] = ["deterministic"]
    invocation["cwd"] = "/deterministic"
    invocation["user"] = _DETERMINISTIC_USER
    invocation["hostname"] = _DETERMINISTIC_HOST
    out["invocation"] = invocation
    output_block = dict(out.get("output", {}))
    output_block["out_dir"] = _DETERMINISTIC_OUT_DIR
    out["output"] = output_block
    # Input — strip the absolute path while preserving the SHA
    # (which IS a deterministic function of the JSONL bytes).
    input_block = dict(out.get("input", {}))
    if input_block.get("jsonl_path") and jsonl_path:
        input_block["jsonl_path"] = os.path.basename(jsonl_path)
    out["input"] = input_block
    return out


def _canonicalise_run_report_headers(headers: dict) -> dict:
    """Strip ``wall_seconds`` from RUN_REPORT headers so two runs
    of the same profile produce byte-identical RUN_REPORT capsule
    payloads."""
    out = dict(headers)
    out["wall_seconds"] = _DETERMINISTIC_WALL
    return out


def _canonicalise_readiness_verdict(verdict: dict) -> dict:
    """Strip per-run wall-clock and absolute-path fields from a
    readiness verdict so two runs produce byte-identical verdict
    payloads (and therefore byte-identical READINESS_CHECK
    capsules)."""
    out = dict(verdict)
    out["wall_seconds"] = _DETERMINISTIC_WALL
    if isinstance(out.get("jsonl_path"), str):
        out["jsonl_path"] = os.path.basename(out["jsonl_path"])
    return out


def _canonicalise_sweep_result(sweep_result: dict) -> dict:
    """Strip per-run wall-clock fields from a unified-runtime sweep
    block so two runs of the same deterministic profile produce
    byte-identical sweep payloads.

    Wall-clock fields stripped (set to ``_DETERMINISTIC_WALL``):

      * top-level ``wall_seconds``
      * each cell's ``cell_wall_s`` (real-mode only)

    The ``jsonl`` field is reduced to its basename to remove
    host-local layout dependence; the JSONL's SHA-256 is recorded
    in the PROVENANCE manifest, so the basename is enough to
    cross-reference back.
    """
    out = dict(sweep_result)
    out["wall_seconds"] = _DETERMINISTIC_WALL
    if isinstance(out.get("jsonl"), str):
        out["jsonl"] = os.path.basename(out["jsonl"])
    cells = out.get("cells")
    if isinstance(cells, list):
        new_cells = []
        for cell in cells:
            if not isinstance(cell, dict):
                new_cells.append(cell)
                continue
            cc = dict(cell)
            if "cell_wall_s" in cc:
                cc["cell_wall_s"] = _DETERMINISTIC_WALL
            new_cells.append(cc)
        out["cells"] = new_cells
    return out


def _run_profile_capsule_native(profile_name: str, *,
                                 out_dir: str,
                                 jsonl_override: str | None,
                                 force_sweep: bool,
                                 skip_sweep: bool,
                                 acknowledge_heavy: bool,
                                 allow_unsafe_sandbox: bool,
                                 deterministic: bool = False,
                                 ) -> dict:
    """Capsule-native run path.

    Capsules drive execution through their lifecycle. Each stage
    seals one capsule; the next stage is gated by the parent CID
    being present in the ledger (Capsule Contract C5). Substantive
    artefacts are content-addressed at write time (Theorem W3-33);
    meta-artefacts (the report itself, the view, the summary) are
    sealed *after* the RUN_REPORT capsule and recorded as a
    separate post-RUN_REPORT artefact slice.
    """
    from vision_mvp.wevra.capsule_runtime import CapsuleNativeRunContext
    from vision_mvp.wevra.provenance import build_manifest
    from vision_mvp.wevra.runtime import (
        sweep_spec_from_profile, run_sweep,
    )

    t0 = time.time()
    prof = _resolve_profile_and_setup(
        profile_name, out_dir=out_dir,
        jsonl_override=jsonl_override,
        allow_unsafe_sandbox=allow_unsafe_sandbox)

    ctx = CapsuleNativeRunContext()
    # Stage 1: seal PROFILE before anything else can reference it.
    ctx.start_run(profile_name=profile_name, profile_dict=prof)

    # Stage 2: readiness. The verdict is computed by the legacy
    # primitive (unchanged); the capsule wrapping happens in flight.
    readiness_verdict = run_readiness(
        prof["readiness"]["jsonl"],
        limit=prof["readiness"]["limit"],
        sandbox_name=prof["readiness"]["sandbox_name"])
    if deterministic:
        readiness_verdict = _canonicalise_readiness_verdict(
            readiness_verdict)
    rd_cap = ctx.seal_readiness(readiness_verdict)
    # Substantive artefact: the readiness verdict on disk is
    # content-addressed at write time. Its parent is the
    # READINESS_CHECK capsule itself (tighter than the legacy
    # post-hoc fold's parent=profile, which is fine — the in-flight
    # path produces a stricter parent set).
    readiness_path = os.path.join(out_dir, "readiness_verdict.json")
    readiness_data = json.dumps(
        readiness_verdict, indent=2, default=str).encode("utf-8")
    # In deterministic mode, the ARTIFACT capsule's payload uses
    # the basename only, so two runs with different ``out_dir``
    # values produce byte-identical ARTIFACT payloads. The on-disk
    # bytes still land at the full path; the capsule just records
    # the path-anchor independently of host-local layout.
    artifact_path_for_capsule = (
        os.path.basename(readiness_path) if deterministic
        else readiness_path)
    ctx.seal_and_write_artifact(
        path=readiness_path, data=readiness_data,
        parents=(rd_cap.cid,),
        recorded_path=artifact_path_for_capsule)

    # Stage 3: sweep — runs through the unified runtime which
    # seals each cell in flight via ctx.
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
            spec = sweep_spec_from_profile(
                profile_name,
                acknowledge_heavy=acknowledge_heavy,
                jsonl_override=jsonl_override)
            if spec is not None:
                try:
                    sweep_result = run_sweep(spec, ctx=ctx)
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
                # Substantive artefact: sweep_result.json /
                # sweep_launch.json. Parent: the SWEEP_SPEC capsule
                # if one was sealed, else profile.
                artifact_name = (
                    "sweep_result.json" if sweep_result.get(
                        "executed_in_process") else "sweep_launch.json")
                sweep_path = os.path.join(out_dir, artifact_name)
                sweep_for_disk = (
                    _canonicalise_sweep_result(sweep_result)
                    if deterministic else sweep_result)
                sweep_data = json.dumps(
                    sweep_for_disk, indent=2, default=str).encode("utf-8")
                sweep_parent = (
                    (ctx.spec_cap.cid,) if ctx.spec_cap is not None
                    else (ctx.profile_cap.cid,))
                sweep_recorded = (
                    artifact_name if deterministic else sweep_path)
                ctx.seal_and_write_artifact(
                    path=sweep_path, data=sweep_data,
                    parents=sweep_parent,
                    recorded_path=sweep_recorded)

    # Stage 4: provenance manifest.
    #
    # Build with the FINAL artifact list (so the PROVENANCE
    # capsule's CID matches what a post-hoc fold of the same
    # report would produce — Theorem W3-34). The artifact list is
    # the union of files already on disk (substantive artefacts
    # we just wrote) and the upcoming meta-artefacts (which the
    # runner will write at the end).
    rd_cfg = prof.get("readiness") or {}
    sw_cfg = prof.get("sweep") or {}
    jsonl_path = (
        jsonl_override or sw_cfg.get("jsonl") or rd_cfg.get("jsonl"))
    upcoming_meta = {"product_report.json", "product_summary.txt",
                     "capsule_view.json", "provenance.json"}
    on_disk_now = set(os.listdir(out_dir))
    artifact_list = sorted(on_disk_now | upcoming_meta)
    manifest = build_manifest(
        profile_name=profile_name,
        profile_schema=_profiles.SCHEMA_VERSION,
        jsonl_path=jsonl_path,
        model=sw_cfg.get("model"),
        endpoint=sw_cfg.get("ollama_url"),
        sandbox=(sw_cfg.get("sandbox") or rd_cfg.get("sandbox_name")),
        out_dir=out_dir,
        artifacts=artifact_list,
        argv=sys.argv,
        extra={
            "invocation_kwargs": {
                "skip_sweep": skip_sweep,
                "force_sweep": force_sweep,
                "jsonl_override": jsonl_override,
            },
        },
    )
    if deterministic:
        # SDK v3.3 — strip per-run / host-local / wall-clock fields
        # from the manifest used to seal the PROVENANCE capsule so
        # two runs of the same deterministic profile produce
        # byte-identical PROVENANCE payloads (Theorem W3-41). The
        # on-disk ``provenance.json`` carries the canonicalised
        # manifest too, so the on-disk artefact's SHA is also
        # deterministic.
        manifest = _canonicalise_for_determinism(
            manifest, profile_name=profile_name,
            jsonl_path=jsonl_path)
    prov_cap = ctx.seal_provenance(manifest)
    prov_path = os.path.join(out_dir, "provenance.json")
    prov_data = json.dumps(
        manifest, indent=2, default=str).encode("utf-8")
    prov_recorded = (
        "provenance.json" if deterministic else prov_path)
    ctx.seal_and_write_artifact(
        path=prov_path, data=prov_data,
        parents=(prov_cap.cid,),
        recorded_path=prov_recorded)

    # Now build the in-memory product report.
    product_report = {
        "schema": RUNNER_SCHEMA,
        "profile": profile_name,
        "profile_description": prof["description"],
        "readiness": readiness_verdict,
        "sweep": sweep_result,
        "wall_seconds": round(time.time() - t0, 2),
        "out_dir": os.path.abspath(out_dir),
        "provenance": manifest,
        "artifacts": artifact_list,
    }

    # Stage 5: seal the RUN_REPORT capsule. Parents are every
    # other capsule sealed so far (PROFILE, READINESS_CHECK,
    # SWEEP_SPEC, SWEEP_CELL × N, PROVENANCE, ARTIFACT × K). The
    # RUN_REPORT's CID is the durable run identifier.
    headers = {
        "profile": profile_name,
        "schema": product_report["schema"],
        "wall_seconds": product_report["wall_seconds"],
        "ready": bool((product_report.get("readiness") or {})
                       .get("ready")),
        "executed_in_process": bool(
            (product_report.get("sweep") or {})
            .get("executed_in_process")),
    }
    if deterministic:
        headers = _canonicalise_run_report_headers(headers)
    ctx.seal_run_report(headers)

    # Stage 6: render the in-flight capsule view ONCE. Both the
    # embedded report["capsules"] and the on-disk capsule_view.json
    # are this exact same render — no chicken-and-egg, no
    # render-twice drift. The view's chain_head is stable from
    # this moment forward; ``wevra-capsule verify`` cross-checks it.
    view = ctx.render(include_payload=False)
    product_report["capsules"] = view

    # Stage 7: meta-artefacts (product_report.json,
    # capsule_view.json, product_summary.txt). These are
    # *post-view rendering* of the canonical capsule view; they
    # are NOT themselves capsule-tracked in the primary ledger.
    # Naming a meta-artefact under its own ARTIFACT capsule
    # within the primary ledger would require the capsule view
    # to include the capsule that hashes its own bytes — a
    # circular dependency formalised as Theorem W3-36
    # (``docs/CAPSULE_FORMALISM.md`` § 4.H).
    #
    # SDK v3.2 closes the gap with a *detached witness*: after
    # the meta-artefacts are written, the runner re-reads each
    # one, computes its on-disk SHA-256, and seals a META_MANIFEST
    # capsule in a SECONDARY ledger. The manifest is written to
    # ``meta_manifest.json`` and is the one-hop trust unit beyond
    # the primary view (Theorem W3-36's positive corollary —
    # detached authentication is achievable, and the limitation
    # is sharp: it cannot be in the primary ledger).
    summary_text = _report.render_summary(product_report)
    product_report["summary_text"] = summary_text

    summary_path = os.path.join(out_dir, "product_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as fh:
        fh.write(summary_text)

    view_path = os.path.join(out_dir, "capsule_view.json")
    with open(view_path, "w", encoding="utf-8") as fh:
        json.dump(view, fh, indent=2, default=str)

    report_path = os.path.join(out_dir, "product_report.json")
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(product_report, fh, indent=2, default=str)

    # Stage 8 (post-fixed-point): detached META_MANIFEST. Seal
    # the secondary ledger and write ``meta_manifest.json``.
    # Failures here do not invalidate the primary run — they are
    # logged into the report's ``meta_manifest_error`` field.
    try:
        meta_artifacts: list[dict[str, Any]] = []
        for name in ("product_report.json", "capsule_view.json",
                      "product_summary.txt"):
            full = os.path.join(out_dir, name)
            if not os.path.exists(full):
                continue
            with open(full, "rb") as fh:
                data = fh.read()
            import hashlib as _hashlib
            sha = _hashlib.sha256(data).hexdigest()
            meta_artifacts.append({
                "path": name,
                "sha256": sha,
                "n_bytes": len(data),
            })
        ctx.seal_meta_manifest(meta_artifacts=meta_artifacts)
        manifest_view = ctx.render_meta_manifest_view()
        manifest_path = os.path.join(out_dir, "meta_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as fh:
            json.dump(manifest_view, fh, indent=2, default=str)
    except Exception as ex:
        # The primary run is still valid; just record the issue.
        product_report["meta_manifest_error"] = (
            f"{type(ex).__name__}: {ex}")
    return product_report


def _run_profile_post_hoc(profile_name: str, *,
                           out_dir: str,
                           jsonl_override: str | None,
                           force_sweep: bool,
                           skip_sweep: bool,
                           acknowledge_heavy: bool,
                           allow_unsafe_sandbox: bool,
                           deterministic: bool = False,
                           ) -> dict:
    """Legacy post-hoc fold path. Retained so the SDK can still
    emit a v3-shape capsule view from a finished run dict (third
    parties who construct ``product_report`` outside the runtime
    can still call ``build_report_ledger``).

    The two paths produce CID-equivalent ledgers for the
    non-ARTIFACT kinds (Theorem W3-34).
    """
    t0 = time.time()
    prof = _resolve_profile_and_setup(
        profile_name, out_dir=out_dir,
        jsonl_override=jsonl_override,
        allow_unsafe_sandbox=allow_unsafe_sandbox)

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
        artifacts=None,
        argv=sys.argv,
        extra={
            "invocation_kwargs": {
                "skip_sweep": skip_sweep,
                "force_sweep": force_sweep,
                "jsonl_override": jsonl_override,
            },
        },
    )
    if deterministic:
        manifest = _canonicalise_for_determinism(
            manifest, profile_name=profile_name,
            jsonl_path=jsonl_path)
    prov_path = os.path.join(out_dir, "provenance.json")
    with open(prov_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    product_report["provenance"] = manifest

    report_path = os.path.join(out_dir, "product_report.json")
    summary_path = os.path.join(out_dir, "product_summary.txt")
    view_path = os.path.join(out_dir, "capsule_view.json")
    existing = set(os.listdir(out_dir))
    upcoming = {"product_report.json", "product_summary.txt",
                "capsule_view.json"}
    artifacts = sorted(existing | upcoming)
    product_report["artifacts"] = artifacts
    manifest["output"]["artifacts"] = artifacts

    from vision_mvp.wevra.capsule import (
        build_report_ledger, render_view,
    )
    if deterministic:
        # Strip wall_seconds from the report before the post-hoc
        # fold so the synthesised RUN_REPORT capsule's headers
        # match the in-flight builder's deterministic output.
        product_report["wall_seconds"] = _DETERMINISTIC_WALL
    ledger, run_cid = build_report_ledger(
        product_report, profile_dict=prof)
    view = render_view(ledger, root_cid=run_cid).as_dict()
    # Tag the post-hoc construction so consumers can tell.
    view["construction"] = "post_hoc"
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
    ap.add_argument("--legacy-post-hoc-capsules", action="store_true",
                     help=("opt out of the capsule-native runtime and "
                            "fall back to the legacy post-hoc fold "
                            "(``build_report_ledger`` after the run "
                            "completes). The two paths produce CID-"
                            "equivalent ledgers for non-ARTIFACT kinds; "
                            "the legacy path is retained only for "
                            "third parties whose product_report dicts "
                            "are constructed outside the runtime."))
    ap.add_argument("--deterministic", action="store_true",
                     help=("strip per-run / host-local / wall-clock "
                            "fields from the PROVENANCE / RUN_REPORT "
                            "capsule payloads so two runs of the same "
                            "deterministic profile produce identical "
                            "full-DAG CIDs (Theorem W3-41). The "
                            "default mode preserves wall-clock-bearing "
                            "CIDs for normal operations; use this for "
                            "audit / replay / CI."))
    args = ap.parse_args()

    report = run_profile(
        args.profile, out_dir=args.out_dir,
        jsonl_override=args.jsonl,
        force_sweep=args.force_sweep,
        skip_sweep=args.skip_sweep,
        acknowledge_heavy=args.acknowledge_heavy,
        allow_unsafe_sandbox=args.allow_unsafe_sandbox,
        capsule_native=not args.legacy_post_hoc_capsules,
        deterministic=args.deterministic)
    print(report["summary_text"])
    return 0 if report["readiness"]["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
