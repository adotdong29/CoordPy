"""Wevra unified mock / real runtime.

Slice 1 shipped a mock-executes-in-process, real-stages-a-launch-cmd
split. Slice 2 unifies the two behind ``run_sweep(spec)``:

  * **Mock mode**: deterministic oracle, no network, runs in-process.
    Same semantics as Slice 1.
  * **Real mode — acknowledged**: the LLM-backed sweep runs
    *in-process* against the configured Ollama endpoint. The report
    records ``executed_in_process=True``. This is the drop-in SDK
    path.
  * **Real mode — not acknowledged**: Wevra refuses to start the
    heavy run and emits the resolved launch command as a staging
    artifact. The report records ``executed_in_process=False`` and
    ``requires_acknowledgement=True``. This is the explicit cost
    gate; it is NOT a silent fallback.

``SweepSpec`` is the frozen, validated descriptor the runner consumes.
``RunSpec.acknowledge_heavy`` flips on the executed path for real
profiles.

Artifact model is unified: every run, mock or real, emits the same
``phase45.product_report.v2`` schema with a ``sweep`` block carrying
``mode``, ``executed_in_process``, ``cells`` (if executed),
``launch_cmd`` (if staged), ``wall_seconds``, and ``sandbox``.

This module intentionally does NOT rewrite the per-phase experiment
scripts. It imports the same primitives (``run_swe_loop_sandboxed``,
``ParserComplianceCounter``, ``load_jsonl_bank``) that the research
scripts use, so the unified path is a *caller* of the same substrate,
not a reimplementation.
"""

from __future__ import annotations

import dataclasses
import os
import time
from typing import Any


class HeavyRunNotAcknowledged(Exception):
    """Raised when ``run_sweep`` is asked to execute a real-LLM cell
    but ``RunSpec.acknowledge_heavy`` is False AND the caller asked
    for strict gating (``strict_cost_gate=True``). By default the
    runtime stages the launch instead of raising."""


@dataclasses.dataclass(frozen=True)
class SweepSpec:
    """Frozen descriptor for one unified sweep.

    Fields
    ------
    mode              : "mock" | "real"
    jsonl             : path to the SWE-bench-shape JSONL
    model             : LLM tag (real mode only; ignored for mock)
    endpoint          : LLM base URL (real mode only)
    sandbox           : "in_process" | "subprocess" | "docker"
    parser_modes      : list[str]
    apply_modes       : list[str]
    n_distractors     : list[int]
    n_instances       : int | None
    timeout_s         : per-task patch+test timeout
    llm_timeout       : per-LLM-call timeout (real mode)
    max_tokens        : LLM generation budget (real mode)
    acknowledge_heavy : operator cost gate for real runs
    """

    mode: str
    jsonl: str
    sandbox: str = "subprocess"
    parser_modes: tuple[str, ...] = ("strict", "robust")
    apply_modes: tuple[str, ...] = ("strict",)
    n_distractors: tuple[int, ...] = (6,)
    n_instances: int | None = None
    model: str | None = None
    endpoint: str | None = None
    timeout_s: float = 20.0
    llm_timeout: float = 300.0
    max_tokens: int = 400
    acknowledge_heavy: bool = False
    enable_raw_capture: bool = False

    def __post_init__(self) -> None:
        if self.mode not in ("mock", "real"):
            raise ValueError(f"mode must be mock|real, got {self.mode!r}")
        if self.mode == "real" and not self.model:
            raise ValueError("real mode requires model=<tag>")


def _load_bank(spec: SweepSpec, n_distractors: int):
    from vision_mvp.wevra.extensions.taskbank import get_task_bank
    from vision_mvp.tasks.swe_bench_bridge import build_synthetic_event_log
    loader = get_task_bank("jsonl")
    bundle = loader.load(
        spec.jsonl,
        hidden_event_log_factory=(
            lambda t, _k=n_distractors: build_synthetic_event_log(t, _k)),
        limit=spec.n_instances,
    )
    return bundle.tasks, bundle.repo_files


def _select_sandbox(spec: SweepSpec):
    from vision_mvp.wevra.extensions.sandbox import get_sandbox
    return get_sandbox(spec.sandbox)


def _mock_cells(spec: SweepSpec) -> list[dict[str, Any]]:
    from vision_mvp.tasks.swe_bench_bridge import (
        ALL_SWE_STRATEGIES, ParserComplianceCounter,
        deterministic_oracle_generator,
    )
    from vision_mvp.tasks.swe_sandbox import run_swe_loop_sandboxed

    sandbox = _select_sandbox(spec)
    cells: list[dict[str, Any]] = []
    for parser_mode in spec.parser_modes:
        for apply_mode in spec.apply_modes:
            for nd in spec.n_distractors:
                tasks, repo_files = _load_bank(spec, nd)
                counter = ParserComplianceCounter()
                rep = run_swe_loop_sandboxed(
                    bank=tasks, repo_files=repo_files,
                    generator=deterministic_oracle_generator,
                    sandbox=sandbox,
                    strategies=tuple(ALL_SWE_STRATEGIES),
                    timeout_s=spec.timeout_s, apply_mode=apply_mode)
                cells.append({
                    "parser_mode": parser_mode,
                    "apply_mode": apply_mode,
                    "n_distractors": nd,
                    "pooled": rep.pooled_summary(),
                    "parser_compliance": counter.as_dict(),
                    "n_instances": len(tasks),
                })
    return cells


def _real_cells(spec: SweepSpec) -> list[dict[str, Any]]:
    """Execute a real-LLM sweep in-process.

    Lives here in ``wevra.runtime`` (not in `experiments/phase42`)
    so the Wevra SDK owns the unified path. Reuses the exact same
    substrate primitives (``run_swe_loop_sandboxed``, parser, client).
    """
    from vision_mvp.core.llm_client import LLMClient
    from vision_mvp.tasks.swe_bench_bridge import (
        ALL_SWE_STRATEGIES, ParserComplianceCounter, ProposedPatch,
        build_patch_generator_prompt,
    )
    from vision_mvp.tasks.swe_patch_parser import (
        ALL_PARSER_MODES, parse_patch_block,
    )
    from vision_mvp.tasks.swe_bench_bridge import parse_unified_diff
    from vision_mvp.tasks.swe_sandbox import run_swe_loop_sandboxed

    sandbox = _select_sandbox(spec)
    client = LLMClient(
        model=spec.model, timeout=spec.llm_timeout,
        base_url=spec.endpoint)
    raw_cache: dict[tuple, str] = {}
    cells: list[dict[str, Any]] = []
    for parser_mode in spec.parser_modes:
        resolved_mode = None if parser_mode == "none" else parser_mode
        if resolved_mode is not None and resolved_mode not in ALL_PARSER_MODES:
            raise ValueError(
                f"unknown parser_mode {parser_mode!r}; "
                f"valid: {sorted(ALL_PARSER_MODES)} + 'none'")
        for apply_mode in spec.apply_modes:
            for nd in spec.n_distractors:
                tasks, repo_files = _load_bank(spec, nd)
                counter = ParserComplianceCounter()

                def _call(prompt: str,
                           _client=client,
                           _max=spec.max_tokens) -> str:
                    return _client.generate(
                        prompt, max_tokens=_max, temperature=0.0)

                def _gen(task, ctx, buggy_source, issue_summary,
                          _counter=counter, _pm=resolved_mode,
                          _nd=nd):
                    strat_proxy = ("substrate" if "hunk" in ctx
                                    else "naive_or_routing")
                    prompt = build_patch_generator_prompt(
                        task=task, ctx=ctx,
                        buggy_source=buggy_source,
                        issue_summary=issue_summary,
                        prompt_style="block")
                    key = (task.instance_id, strat_proxy, _nd)
                    if key in raw_cache:
                        text = raw_cache[key]
                    else:
                        text = _call(prompt)
                        raw_cache[key] = text
                    outcome = parse_patch_block(
                        text, mode=_pm,
                        unified_diff_parser=parse_unified_diff)
                    _counter.record(outcome)
                    if not outcome.ok:
                        return ProposedPatch(
                            patch=(),
                            rationale=f"parse_failed:{outcome.failure_kind}")
                    rat = "llm_proposed"
                    if outcome.recovery:
                        rat = f"llm_proposed:{outcome.recovery}"
                    return ProposedPatch(
                        patch=outcome.substitutions, rationale=rat)

                t0 = time.time()
                rep = run_swe_loop_sandboxed(
                    bank=tasks, repo_files=repo_files,
                    generator=_gen, sandbox=sandbox,
                    strategies=tuple(ALL_SWE_STRATEGIES),
                    timeout_s=spec.timeout_s,
                    apply_mode=apply_mode)
                wall = time.time() - t0
                cells.append({
                    "parser_mode": parser_mode,
                    "apply_mode": apply_mode,
                    "n_distractors": nd,
                    "pooled": rep.pooled_summary(),
                    "parser_compliance": counter.as_dict(),
                    "n_instances": len(tasks),
                    "cell_wall_s": round(wall, 2),
                })
    return cells


def _resolve_launch_cmd(spec: SweepSpec) -> list[str]:
    """Resolve the staging-mode launch command — the exact argv
    that would run this sweep via the existing phase42 CLI. Used
    when the heavy run is not acknowledged."""
    import sys
    cmd = [
        sys.executable, "-m",
        "vision_mvp.experiments.phase42_parser_sweep",
        "--mode", "real",
        "--model", str(spec.model),
        "--ollama-url", str(spec.endpoint or ""),
        "--jsonl", spec.jsonl,
        "--parser-modes", *spec.parser_modes,
        "--apply-modes", *spec.apply_modes,
        "--n-distractors", *[str(x) for x in spec.n_distractors],
        "--sandbox", spec.sandbox,
    ]
    if spec.n_instances is not None:
        cmd += ["--n-instances", str(spec.n_instances)]
    return cmd


def run_sweep(spec: SweepSpec,
              *, strict_cost_gate: bool = False) -> dict[str, Any]:
    """Execute ``spec`` and return a unified sweep block.

    Contract (``phase45.product_report.v2`` sweep sub-block):

      {
        "schema": "wevra.sweep.v2",
        "mode": "mock" | "real",
        "executed_in_process": bool,
        "requires_acknowledgement": bool,
        "sandbox": str,
        "jsonl": str,
        "cells": [ ... ]                 # present iff executed
        "launch_cmd": [ ... ]            # present iff staged
        "wall_seconds": float,
        "model": str | None,
        "endpoint": str | None,
      }

    ``strict_cost_gate=True`` makes a not-acknowledged real run raise
    ``HeavyRunNotAcknowledged`` instead of staging.
    """
    t0 = time.time()
    if spec.mode == "mock":
        cells = _mock_cells(spec)
        return {
            "schema": "wevra.sweep.v2",
            "mode": "mock", "executed_in_process": True,
            "requires_acknowledgement": False,
            "sandbox": spec.sandbox, "jsonl": spec.jsonl,
            "cells": cells,
            "launch_cmd": None,
            "wall_seconds": round(time.time() - t0, 3),
            "model": None, "endpoint": None,
        }
    # real mode
    if not spec.acknowledge_heavy:
        if strict_cost_gate:
            raise HeavyRunNotAcknowledged(
                "real-mode sweep requires "
                "RunSpec(acknowledge_heavy=True) or "
                "run_sweep(..., strict_cost_gate=False)")
        launch_cmd = _resolve_launch_cmd(spec)
        raw_capture_cmd = None
        if spec.enable_raw_capture:
            raw_capture_cmd = launch_cmd[:]
            # swap the experiment module to the raw-capture variant
            raw_capture_cmd[2] = (
                "vision_mvp.experiments.phase44_semantic_residue")
        return {
            "schema": "wevra.sweep.v2",
            "mode": "real", "executed_in_process": False,
            "requires_acknowledgement": True,
            "sandbox": spec.sandbox, "jsonl": spec.jsonl,
            "cells": None,
            "launch_cmd": launch_cmd,
            "raw_capture_launch_cmd": raw_capture_cmd,
            "wall_seconds": round(time.time() - t0, 3),
            "model": spec.model, "endpoint": spec.endpoint,
            "note": (
                "Heavy real-LLM run was NOT acknowledged. The SDK "
                "refused to start the run and resolved the launch "
                "command instead. Re-run with "
                "RunSpec(acknowledge_heavy=True) to execute in-process."),
        }
    cells = _real_cells(spec)
    return {
        "schema": "wevra.sweep.v2",
        "mode": "real", "executed_in_process": True,
        "requires_acknowledgement": False,
        "sandbox": spec.sandbox, "jsonl": spec.jsonl,
        "cells": cells, "launch_cmd": None,
        "wall_seconds": round(time.time() - t0, 3),
        "model": spec.model, "endpoint": spec.endpoint,
    }


def sweep_spec_from_profile(profile_name: str,
                              *, acknowledge_heavy: bool = False,
                              jsonl_override: str | None = None,
                              ) -> SweepSpec | None:
    """Build a ``SweepSpec`` from a profile's ``sweep`` block.

    Returns None if the profile has no sweep block. Endpoint is
    resolved via ``WEVRA_OLLAMA_URL_*`` env overrides when the
    profile's declared endpoint matches a known cluster node.
    """
    from vision_mvp.product import profiles as _profiles
    prof = _profiles.get_profile(profile_name)
    sw = prof.get("sweep")
    if sw is None:
        return None
    jsonl = jsonl_override or sw["jsonl"]
    endpoint = _resolve_endpoint(sw.get("ollama_url"))
    return SweepSpec(
        mode=sw["mode"],
        jsonl=jsonl,
        sandbox=sw["sandbox"],
        parser_modes=tuple(sw["parser_modes"]),
        apply_modes=tuple(sw["apply_modes"]),
        n_distractors=tuple(sw["n_distractors"]),
        n_instances=sw.get("n_instances"),
        model=sw.get("model"),
        endpoint=endpoint,
        acknowledge_heavy=acknowledge_heavy,
        enable_raw_capture=bool(sw.get("enable_raw_capture", False)),
    )


def _resolve_endpoint(declared: str | None) -> str | None:
    """Apply env-var overrides to a profile's declared endpoint.

      WEVRA_OLLAMA_URL_MAC1 — overrides profiles whose declared URL
                              matches ``192.168.12.191``
      WEVRA_OLLAMA_URL_MAC2 — overrides ``192.168.12.248``
      WEVRA_OLLAMA_URL      — overrides any endpoint (last resort)
    """
    if declared is None:
        return os.environ.get("WEVRA_OLLAMA_URL")
    if "192.168.12.191" in declared and os.environ.get(
            "WEVRA_OLLAMA_URL_MAC1"):
        return os.environ["WEVRA_OLLAMA_URL_MAC1"]
    if "192.168.12.248" in declared and os.environ.get(
            "WEVRA_OLLAMA_URL_MAC2"):
        return os.environ["WEVRA_OLLAMA_URL_MAC2"]
    if os.environ.get("WEVRA_OLLAMA_URL"):
        return os.environ["WEVRA_OLLAMA_URL"]
    return declared


__all__ = [
    "SweepSpec", "run_sweep", "HeavyRunNotAcknowledged",
    "sweep_spec_from_profile",
]
