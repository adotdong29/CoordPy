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
    # SDK v3.4 — synthetic mode uses the same LLM-backed code path
    # as real mode (and therefore exercises the PROMPT /
    # LLM_RESPONSE capsule slice), but the LLM client is a
    # deterministic in-process substitute (see
    # ``vision_mvp.wevra.synthetic_llm.SyntheticLLMClient``).
    # ``synthetic_model_tag`` selects one of the canned
    # distributions in ``SYNTHETIC_MODEL_PROFILES``. No network is
    # required; ``acknowledge_heavy`` is unused.
    synthetic_model_tag: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in ("mock", "real", "synthetic"):
            raise ValueError(
                f"mode must be mock|real|synthetic, got {self.mode!r}")
        if self.mode == "real" and not self.model:
            raise ValueError("real mode requires model=<tag>")
        if self.mode == "synthetic" and not self.synthetic_model_tag:
            raise ValueError(
                "synthetic mode requires synthetic_model_tag=<tag> "
                "(see vision_mvp.wevra.synthetic_llm.SYNTHETIC_MODEL_PROFILES)")


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


def _parse_outcome_from_rationale(
        rationale: str,
        *, n_substitutions: int,
        ) -> tuple[bool, str, str, str]:
    """Reverse-engineer the parser-axis structured outcome from
    a ``ProposedPatch.rationale`` string.

    The substrate stores the parser's ``failure_kind`` / ``recovery``
    in the rationale string of the ``ProposedPatch`` it returns:

      * ``"parse_failed:<kind>"`` for a parser failure (substitutions
        empty);
      * ``"llm_proposed"`` for a clean parse without recovery;
      * ``"llm_proposed:<recovery>"`` when a recovery heuristic
        fired;
      * any other string (e.g. issue-summary text from the
        deterministic oracle) is treated as the
        ``PARSE_OUTCOME_ORACLE`` sentinel.

    Returning ``(ok, failure_kind, recovery, detail)`` lets the
    runtime build a PARSE_OUTCOME capsule without coupling the
    capsule layer to substrate string formats.
    """
    rat = (rationale or "").strip()
    if rat.startswith("parse_failed"):
        # "parse_failed" or "parse_failed:<kind>"
        if ":" in rat:
            kind = rat.split(":", 1)[1]
        else:
            kind = "parse_failed"
        return False, kind, "", rat
    if rat.startswith("llm_proposed"):
        # "llm_proposed" or "llm_proposed:<recovery>"
        if ":" in rat:
            recovery = rat.split(":", 1)[1]
        else:
            recovery = ""
        return True, "ok", recovery, ""
    if rat.startswith("gen_error"):
        # Generator raised before parsing; surface as a typed
        # generator-error parse outcome.
        return False, "gen_error", "", rat[:200]
    # Default — deterministic oracle or any non-parser path.
    detail = rat[:200] if (n_substitutions > 0) else ""
    return True, "oracle", "", detail


def _make_intra_cell_hooks(ctx: "Any",
                              *,
                              parser_mode: str,
                              apply_mode: str,
                              n_distractors: int,
                              llm_io_state: "dict[Any, Any] | None" = None,
                              ):
    """Build the (on_patch_proposed, on_test_completed) hook pair
    that routes intra-cell transitions into capsule lifecycle
    transitions on ``ctx``.

    Returns ``(None, None)`` when ``ctx`` is None — the substrate
    loop runs byte-for-byte unchanged. Returns concrete closures
    otherwise. The closures capture the cell coordinates so the
    sealed PATCH_PROPOSAL / TEST_VERDICT capsules carry the
    full (parser_mode, apply_mode, n_distractors, instance_id,
    strategy) tuple needed to navigate the DAG without an
    external index.

    SDK v3.3: ``on_patch`` now also seals a PARSE_OUTCOME capsule
    (parent: SWEEP_SPEC) BEFORE the PATCH_PROPOSAL. The parse
    outcome's ``failure_kind`` / ``recovery`` are derived from the
    substrate's ``ProposedPatch.rationale`` string (see
    ``_parse_outcome_from_rationale``), so the capsule layer does
    not need to peek inside the substrate's parser. The
    PATCH_PROPOSAL is then parented on both SWEEP_SPEC and the
    PARSE_OUTCOME, giving the parse → patch → verdict chain a
    typed DAG witness.

    SDK v3.4 (``llm_io_state`` parameter): when the caller is the
    LLM-backed ``_real_cells`` path, it threads a per-cell mapping
    ``(task.instance_id, strategy_proxy) -> llm_response_cid``
    that ``_gen`` populates after sealing the PROMPT and
    LLM_RESPONSE capsules. The ``on_patch`` hook reads the
    mapping and, when an LLM_RESPONSE CID is available, parents
    the PARSE_OUTCOME on it (in addition to SWEEP_SPEC). This
    closes the prompt → response → parse → patch → verdict
    chain on the DAG. When ``llm_io_state`` is ``None`` (the
    mock-oracle path) the PARSE_OUTCOME is sealed as before, with
    parents = (SWEEP_SPEC,) only.
    """
    if ctx is None:
        return None, None

    def on_patch(task, strat, proposed):
        n_subs = len(tuple(proposed.patch))
        ok, failure_kind, recovery, detail = (
            _parse_outcome_from_rationale(
                proposed.rationale or "",
                n_substitutions=n_subs))
        # SDK v3.4: thread the LLM_RESPONSE CID into the
        # PARSE_OUTCOME capsule when one is available. The
        # strategy_proxy used to key llm_io_state is "substrate"
        # vs "naive_or_routing" because naive + routing share an
        # LLM call (raw_cache deduplicates by strategy_proxy).
        # The audit invariants (L-11) accept either parent shape
        # — single-parent (oracle / no LLM observation) or
        # two-parent (LLM-backed with response chain).
        llm_response_cid = None
        if llm_io_state is not None:
            strat_proxy = ("substrate" if strat == "substrate"
                            else "naive_or_routing")
            llm_response_cid = llm_io_state.get(
                (task.instance_id, strat_proxy))
        parse_cap = ctx.seal_parse_outcome(
            instance_id=task.instance_id,
            strategy=strat,
            parser_mode=parser_mode,
            apply_mode=apply_mode,
            n_distractors=int(n_distractors),
            ok=ok,
            failure_kind=failure_kind,
            recovery=recovery,
            substitutions_count=n_subs,
            detail=detail,
            llm_response_cid=llm_response_cid,
        )
        cap = ctx.seal_patch_proposal(
            instance_id=task.instance_id,
            strategy=strat,
            parser_mode=parser_mode,
            apply_mode=apply_mode,
            n_distractors=int(n_distractors),
            substitutions=tuple(proposed.patch),
            rationale=proposed.rationale or "",
            parse_outcome_cid=parse_cap.cid,
        )
        return cap.cid

    def on_verdict(task, strat, wr, patch_cid):
        if patch_cid is None:
            # Should not happen in normal flow — on_patch raises
            # rather than returning None.
            return
        ctx.seal_test_verdict(
            instance_id=task.instance_id,
            strategy=strat,
            parser_mode=parser_mode,
            apply_mode=apply_mode,
            n_distractors=int(n_distractors),
            patch_proposal_cid=patch_cid,
            patch_applied=bool(wr.patch_applied),
            syntax_ok=bool(wr.syntax_ok),
            test_passed=bool(wr.test_passed),
            error_kind=wr.error_kind or "",
            error_detail=(wr.error_detail or "")[:300],
        )

    return on_patch, on_verdict


def _seal_prompt_response_pair(ctx: "Any",
                                  *,
                                  task,
                                  strategy_proxy: str,
                                  parser_mode: str,
                                  apply_mode: str,
                                  n_distractors: int,
                                  model_tag: str,
                                  prompt_style: str,
                                  prompt_text: str,
                                  response_text: str,
                                  elapsed_ms: int | None,
                                  ) -> str | None:
    """Seal a PROMPT capsule + an LLM_RESPONSE capsule for one LLM
    call and return the LLM_RESPONSE CID.

    Returns ``None`` when ``ctx`` is None or the spec is not yet
    sealed (defensive — the runtime always seals SWEEP_SPEC
    before any cell runs). The two capsules are sealed in
    succession; the LLM_RESPONSE's parent is the PROMPT's CID.

    Idempotent on content (Capsule Contract C1): two calls with
    byte-identical prompt + response collapse to the same pair of
    CIDs. The runtime relies on this to deduplicate the
    naive + routing strategies' shared LLM call (they hit the
    raw_cache; the prompt and response bytes are identical).
    """
    if ctx is None or ctx.spec_cap is None:
        return None
    prompt_cap = ctx.seal_prompt(
        instance_id=task.instance_id,
        strategy=strategy_proxy,
        parser_mode=parser_mode,
        apply_mode=apply_mode,
        n_distractors=int(n_distractors),
        model_tag=str(model_tag),
        prompt_style=prompt_style,
        prompt_text=prompt_text,
    )
    response_cap = ctx.seal_llm_response(
        instance_id=task.instance_id,
        strategy=strategy_proxy,
        parser_mode=parser_mode,
        apply_mode=apply_mode,
        n_distractors=int(n_distractors),
        model_tag=str(model_tag),
        response_text=response_text,
        prompt_cid=prompt_cap.cid,
        elapsed_ms=elapsed_ms,
    )
    return response_cap.cid


def _mock_cells(spec: SweepSpec,
                 *, ctx: "Any" = None,
                 ) -> list[dict[str, Any]]:
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
                on_patch, on_verdict = _make_intra_cell_hooks(
                    ctx,
                    parser_mode=parser_mode,
                    apply_mode=apply_mode,
                    n_distractors=int(nd))
                rep = run_swe_loop_sandboxed(
                    bank=tasks, repo_files=repo_files,
                    generator=deterministic_oracle_generator,
                    sandbox=sandbox,
                    strategies=tuple(ALL_SWE_STRATEGIES),
                    timeout_s=spec.timeout_s, apply_mode=apply_mode,
                    on_patch_proposed=on_patch,
                    on_test_completed=on_verdict)
                cell = {
                    "parser_mode": parser_mode,
                    "apply_mode": apply_mode,
                    "n_distractors": nd,
                    "pooled": rep.pooled_summary(),
                    "parser_compliance": counter.as_dict(),
                    "n_instances": len(tasks),
                }
                cells.append(cell)
                # In-flight capsule: seal the SWEEP_CELL as soon as
                # its results land. If the ctx is provided, the
                # ledger learns about each cell during the sweep,
                # not after. A cell that fails to seal (over budget,
                # parent missing) raises here and the remaining
                # cells of this sweep do not run.
                if ctx is not None:
                    ctx.seal_sweep_cell(cell)
    return cells


def _real_cells(spec: SweepSpec,
                 *, ctx: "Any" = None,
                 llm_client: "Any" = None,
                 model_tag: str | None = None,
                 ) -> list[dict[str, Any]]:
    """Execute an LLM-backed sweep in-process.

    Lives here in ``wevra.runtime`` (not in `experiments/phase42`)
    so the Wevra SDK owns the unified path. Reuses the exact same
    substrate primitives (``run_swe_loop_sandboxed``, parser,
    client).

    SDK v3.4: a per-cell ``llm_io_state`` mapping keys
    ``(task.instance_id, strategy_proxy)`` to the LLM_RESPONSE
    CID just sealed by ``_gen``. The intra-cell hooks read this
    mapping and, when an LLM_RESPONSE is available, parent the
    PARSE_OUTCOME on it (in addition to SWEEP_SPEC). The
    prompt → response → parse → patch → verdict chain is
    therefore observable end-to-end on the DAG (Theorems
    W3-42 / W3-43 / W3-44).

    ``llm_client`` (optional, SDK v3.4): inject a duck-typed
    substitute for ``LLMClient``. Used by ``_synthetic_cells``
    (deterministic in-process LLM stand-in) and by tests that
    want to drive the LLM-backed path without an Ollama
    endpoint. When ``None`` the runtime instantiates a real
    ``LLMClient`` against ``spec.endpoint``.

    ``model_tag`` (optional, SDK v3.4): the string recorded as
    the PROMPT / LLM_RESPONSE capsules' ``model_tag`` field.
    Defaults to ``spec.model`` so the capsule layer can attribute
    each prompt / response to the LLM that produced it.
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
    if llm_client is None:
        client = LLMClient(
            model=spec.model, timeout=spec.llm_timeout,
            base_url=spec.endpoint)
    else:
        client = llm_client
    resolved_model_tag = (
        model_tag if model_tag is not None
        else (spec.model if spec.model is not None
              else getattr(client, "model", "unknown")))
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
                # Per-cell LLM-IO state: maps (instance_id,
                # strategy_proxy) → LLM_RESPONSE CID. Populated by
                # _gen when an LLM call seals PROMPT + LLM_RESPONSE
                # capsules; consumed by the on_patch hook to thread
                # the response CID into the PARSE_OUTCOME parent set.
                llm_io_state: dict[Any, Any] = {}

                def _call(prompt: str,
                           _client=client,
                           _max=spec.max_tokens) -> tuple[str, int]:
                    """Return ``(response_text, elapsed_ms)``.

                    The elapsed_ms is captured so the LLM_RESPONSE
                    capsule can record latency under non-deterministic
                    runs; it is stripped under deterministic-mode
                    canonicalisation.
                    """
                    t_call = time.time()
                    text = _client.generate(
                        prompt, max_tokens=_max, temperature=0.0)
                    return text, int(round((time.time() - t_call) * 1000))

                def _gen(task, gen_ctx, buggy_source, issue_summary,
                          _counter=counter, _pm=resolved_mode,
                          _nd=nd, _parser_mode=parser_mode,
                          _apply_mode=apply_mode,
                          _model_tag=resolved_model_tag,
                          _state=llm_io_state):
                    strat_proxy = ("substrate" if "hunk" in gen_ctx
                                    else "naive_or_routing")
                    prompt = build_patch_generator_prompt(
                        task=task, ctx=gen_ctx,
                        buggy_source=buggy_source,
                        issue_summary=issue_summary,
                        prompt_style="block")
                    key = (task.instance_id, strat_proxy, _nd)
                    if key in raw_cache:
                        text = raw_cache[key]
                        elapsed_ms_obs = 0
                    else:
                        text, elapsed_ms_obs = _call(prompt)
                        raw_cache[key] = text
                    # SDK v3.4: seal PROMPT + LLM_RESPONSE capsules
                    # for this LLM call. Idempotent on content —
                    # naive + routing strategies share a prompt and
                    # therefore a single PROMPT capsule and a single
                    # LLM_RESPONSE capsule. The capsule layer's
                    # admit_and_seal collapses duplicates by CID.
                    response_cid = _seal_prompt_response_pair(
                        ctx,
                        task=task,
                        strategy_proxy=strat_proxy,
                        parser_mode=_parser_mode,
                        apply_mode=_apply_mode,
                        n_distractors=int(_nd),
                        model_tag=_model_tag,
                        prompt_style="block",
                        prompt_text=prompt,
                        response_text=text,
                        elapsed_ms=elapsed_ms_obs,
                    )
                    if response_cid is not None:
                        _state[(task.instance_id, strat_proxy)] = (
                            response_cid)
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

                on_patch, on_verdict = _make_intra_cell_hooks(
                    ctx,
                    parser_mode=parser_mode,
                    apply_mode=apply_mode,
                    n_distractors=int(nd),
                    llm_io_state=llm_io_state)
                t0 = time.time()
                rep = run_swe_loop_sandboxed(
                    bank=tasks, repo_files=repo_files,
                    generator=_gen, sandbox=sandbox,
                    strategies=tuple(ALL_SWE_STRATEGIES),
                    timeout_s=spec.timeout_s,
                    apply_mode=apply_mode,
                    on_patch_proposed=on_patch,
                    on_test_completed=on_verdict)
                wall = time.time() - t0
                cell = {
                    "parser_mode": parser_mode,
                    "apply_mode": apply_mode,
                    "n_distractors": nd,
                    "pooled": rep.pooled_summary(),
                    "parser_compliance": counter.as_dict(),
                    "n_instances": len(tasks),
                    "cell_wall_s": round(wall, 2),
                }
                cells.append(cell)
                if ctx is not None:
                    ctx.seal_sweep_cell(cell)
    return cells


def _synthetic_cells(spec: SweepSpec,
                       *, ctx: "Any" = None,
                       ) -> list[dict[str, Any]]:
    """Execute a synthetic-LLM sweep in-process.

    Same code path as ``_real_cells`` (so the PROMPT /
    LLM_RESPONSE / PARSE_OUTCOME / PATCH_PROPOSAL / TEST_VERDICT
    capsule chain is exercised end-to-end), but the LLM client is
    a deterministic in-process substitute returning canned
    strings drawn from
    ``synthetic_llm.SYNTHETIC_MODEL_PROFILES[spec.synthetic_model_tag]``.

    Used by:
      * SDK v3.4 contract tests covering the PROMPT /
        LLM_RESPONSE slice without an Ollama endpoint.
      * The cross-model parser-boundary research experiment
        (``vision_mvp.experiments.parser_boundary_cross_model``)
        sweeping multiple synthetic distributions through the
        real parser.
    """
    from vision_mvp.wevra.synthetic_llm import (
        SyntheticLLMClient, make_synthetic_response_fn,
    )
    tasks, _repo_files = _load_bank(spec, spec.n_distractors[0])
    gold_patches: dict[str, tuple[str, str]] = {}
    for task in tasks:
        if task.gold_patch:
            gold_patches[task.instance_id] = task.gold_patch[0]
    fn = make_synthetic_response_fn(
        spec.synthetic_model_tag or "synthetic.clean", gold_patches)
    client = SyntheticLLMClient(
        model_tag=spec.synthetic_model_tag or "synthetic.clean",
        response_fn=fn)
    return _real_cells(
        spec, ctx=ctx, llm_client=client,
        model_tag=spec.synthetic_model_tag or "synthetic.clean")


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
              *, strict_cost_gate: bool = False,
              ctx: "Any" = None,
              llm_backend: "Any" = None,
              ) -> dict[str, Any]:
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
        "backend": str | None,           # SDK v3.6 — provenance only
      }

    ``strict_cost_gate=True`` makes a not-acknowledged real run raise
    ``HeavyRunNotAcknowledged`` instead of staging.

    ``ctx`` (optional) is a ``CapsuleNativeRunContext`` from
    ``vision_mvp.wevra.capsule_runtime``. When provided, the runtime
    seals each cell as a ``SWEEP_CELL`` capsule *in flight*: the
    SWEEP_SPEC capsule is sealed before any cell runs, and each cell
    is sealed as soon as its results land. Mid-run failure leaves an
    in-flight register entry that never reaches the ledger — the
    typed witness of which cell never completed. When ``ctx`` is
    None, the legacy behaviour is preserved (cells accumulate in
    memory; the post-hoc ``build_report_ledger`` folds them into
    capsules afterwards).

    ``llm_backend`` (optional, SDK v3.6) is any object satisfying
    the duck-typed :class:`vision_mvp.wevra.llm_backend.LLMBackend`
    Protocol. When provided in real mode, the inner loop calls
    ``llm_backend.generate`` instead of constructing a default
    Ollama ``LLMClient``. This is the integration point for the
    MLX-distributed two-Mac path: pass an
    ``MLXDistributedBackend(model=..., base_url=...)`` whose
    ``base_url`` is the head-rank ``mlx_lm.server`` started under
    ``mpirun``. The Wevra spine is unchanged — the PROMPT /
    LLM_RESPONSE capsules record the same SHA-256 + length +
    snippet shape regardless of which backend produced them.
    Backward compatible: ``llm_backend=None`` preserves byte-for-
    byte behaviour with prior SDK versions.
    """
    t0 = time.time()
    spec_payload = _spec_payload(spec)
    if ctx is not None and ctx.spec_cap is None:
        # Seal the SWEEP_SPEC before any cell runs. The capsule
        # contract makes "no cells before spec" a typed gate, not a
        # Python ordering convention.
        ctx.seal_sweep_spec({
            **spec_payload,
            # Filled in below once we know which path executed.
            "executed_in_process": (spec.mode == "mock"
                                     or spec.acknowledge_heavy),
        })
    if spec.mode == "mock":
        cells = _mock_cells(spec, ctx=ctx)
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
    if spec.mode == "synthetic":
        # SDK v3.4: deterministic in-process synthetic LLM. No
        # network, no acknowledge_heavy gate, no launch staging.
        # The PROMPT / LLM_RESPONSE / PARSE_OUTCOME chain seals
        # in flight for every (task, strategy) pair just as in the
        # real-LLM path.
        cells = _synthetic_cells(spec, ctx=ctx)
        return {
            "schema": "wevra.sweep.v2",
            "mode": "synthetic", "executed_in_process": True,
            "requires_acknowledgement": False,
            "sandbox": spec.sandbox, "jsonl": spec.jsonl,
            "cells": cells,
            "launch_cmd": None,
            "wall_seconds": round(time.time() - t0, 3),
            "model": spec.synthetic_model_tag,
            "endpoint": None,
            "synthetic_model_tag": spec.synthetic_model_tag,
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
    cells = _real_cells(spec, ctx=ctx, llm_client=llm_backend)
    backend_name = (
        type(llm_backend).__name__ if llm_backend is not None else None)
    return {
        "schema": "wevra.sweep.v2",
        "mode": "real", "executed_in_process": True,
        "requires_acknowledgement": False,
        "sandbox": spec.sandbox, "jsonl": spec.jsonl,
        "cells": cells, "launch_cmd": None,
        "wall_seconds": round(time.time() - t0, 3),
        "model": spec.model, "endpoint": spec.endpoint,
        "backend": backend_name,
    }


def _spec_payload(spec: SweepSpec) -> dict[str, Any]:
    """Canonical payload for the SWEEP_SPEC capsule. Matches the
    shape ``build_report_ledger`` uses for its own post-hoc fold so
    in-flight and post-hoc ledgers stay CID-equivalent on the
    SWEEP_SPEC kind (Theorem W3-34).

    SDK v3.4: synthetic-mode runs preserve the same shape (mode +
    sandbox + jsonl + model + endpoint). For synthetic mode,
    ``model`` carries the synthetic model tag (so two synthetic
    runs with different tags do not collide on SWEEP_SPEC CID).
    """
    if spec.mode == "synthetic":
        return {
            "mode": spec.mode,
            "sandbox": spec.sandbox,
            "jsonl": spec.jsonl,
            "model": spec.synthetic_model_tag,
            "endpoint": None,
        }
    return {
        "mode": spec.mode,
        "sandbox": spec.sandbox,
        "jsonl": spec.jsonl,
        "model": spec.model,
        "endpoint": spec.endpoint,
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
