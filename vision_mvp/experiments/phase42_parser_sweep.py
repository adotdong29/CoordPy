"""Phase 42 — parser-compliance axis + larger-bank rerun.

Phase 41 surfaced a new attribution layer the programme had not
named before: the LLM-output *parser*. On the Phase-41 bank
``gemma2:9b`` emitted the semantically correct fix on every instance
but failed to close the ``<<<`` delimiter of the bridge's OLD/NEW
block contract, so every patch parsed as the empty tuple and landed
as ``patch_no_match``. The matcher axis (strict vs permissive) sat
*below* the parser and could not help.

Phase 42 closes that layer:

  * **Parser axis** (``--parser-modes``). Route every LLM response
    through either the Phase-41 strict regex (baseline) or the new
    Phase-42 ``parse_patch_block`` with tolerant block closing at
    end-of-generation, a unified-diff fallback, and two labelled
    heuristics (``fenced_code_heuristic`` / ``label_prefix_
    heuristic``). Recovery is attributed explicitly — a downstream
    analyst sees which heuristic fired, and can separate parser-
    recovery lift from raw-strict compliance.
  * **Expanded bank** (``--jsonl``). The Phase-42 driver is bank-
    agnostic; the canonical Phase-42 bank is the 57-instance
    ``swe_lite_style_bank.jsonl`` (grown from the Phase-41
    28-instance bank during this phase).
  * **Cluster endpoint support** (``--ollama-url``). The ASPEN
    2-Mac cluster exposes Ollama on
    ``http://192.168.12.191:11434`` (macbook-1) and
    ``http://192.168.12.248:11434`` (macbook-2). The driver
    forwards ``--ollama-url`` to ``LLMClient(base_url=...)`` so
    coding/generation runs can sit on macbook-1 and comparison /
    secondary runs on macbook-2 in parallel.

Typical runs:

    # Phase-42 mock — oracle through the new parser axis. The
    # oracle's patch is byte-exact so parser recovery is vacuous;
    # this run is the null-control on the new axis.
    python3 -m vision_mvp.experiments.phase42_parser_sweep \
        --mode mock --sandbox subprocess \
        --apply-modes strict --parser-modes strict robust \
        --jsonl vision_mvp/tasks/data/swe_lite_style_bank.jsonl \
        --n-distractors 6 \
        --out vision_mvp/results_phase42_parser_mock.json

    # Phase-42 real LLM — macbook-1, qwen2.5-coder:14b,
    # all 57 instances, strict vs robust parser.
    python3 -m vision_mvp.experiments.phase42_parser_sweep \
        --mode real --model qwen2.5-coder:14b \
        --ollama-url http://192.168.12.191:11434 \
        --sandbox subprocess \
        --apply-modes strict --parser-modes strict robust \
        --jsonl vision_mvp/tasks/data/swe_lite_style_bank.jsonl \
        --n-distractors 6 \
        --out vision_mvp/results_phase42_parser_14b_coder.json

    # Phase-42 secondary — macbook-2, qwen3.5:35b (general),
    # all 57 instances, strict vs robust parser. Parallel to
    # the coder run above.
    python3 -m vision_mvp.experiments.phase42_parser_sweep \
        --mode real --model qwen3.5:35b \
        --ollama-url http://192.168.12.248:11434 \
        --sandbox subprocess \
        --apply-modes strict --parser-modes strict robust \
        --jsonl vision_mvp/tasks/data/swe_lite_style_bank.jsonl \
        --n-distractors 6 \
        --out vision_mvp/results_phase42_parser_35b_moe.json

Artifacts shipped alongside:
  * ``vision_mvp/results_phase42_swe_lite_mock.json`` —
    oracle saturation on the 57-instance bank (Theorem P41-1
    reproduction).
  * ``vision_mvp/results_phase42_parser_*.json`` — per-model
    parser sweep.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")))

from vision_mvp.tasks.swe_bench_bridge import (
    ALL_APPLY_MODES, ALL_SWE_STRATEGIES, APPLY_MODE_STRICT,
    STRATEGY_SUBSTRATE, build_synthetic_event_log,
    deterministic_oracle_generator, llm_patch_generator,
    load_jsonl_bank, ParserComplianceCounter,
    parse_unified_diff,
)
from vision_mvp.tasks.swe_patch_parser import (
    ALL_PARSER_MODES, PARSER_ROBUST, PARSER_STRICT, PARSER_UNIFIED,
)
from vision_mvp.tasks.swe_sandbox import (
    SubprocessSandbox, run_swe_loop_sandboxed, select_sandbox,
)


_DEFAULT_JSONL = os.path.join(
    os.path.dirname(__file__), "..", "tasks", "data",
    "swe_lite_style_bank.jsonl")


# ---------------------------------------------------------------
# Generator factory — wires the Phase-42 parser into llm_patch_generator
# ---------------------------------------------------------------


def _make_generator(mode: str, model: str | None,
                     parser_mode: str,
                     prompt_style: str,
                     ollama_url: str | None,
                     timeout: float = 300.0,
                     counter: ParserComplianceCounter | None = None,
                     ):
    """Return ``(generator_callable, client_or_None, counter)``.

    ``counter`` is a fresh ``ParserComplianceCounter`` unless the
    caller pre-allocated one to pool across cells.
    """
    if mode == "mock":
        # Oracle — no parser runs, so parser_mode is meaningless on
        # this path. We still return a fresh counter so the artifact
        # shape is stable.
        return (deterministic_oracle_generator, None,
                counter or ParserComplianceCounter())
    from vision_mvp.core.llm_client import LLMClient
    client = LLMClient(model=model, timeout=timeout,
                        base_url=ollama_url)

    def _call(prompt: str) -> str:
        return client.generate(prompt, max_tokens=400,
                                temperature=0.0)

    ctr = counter or ParserComplianceCounter()
    # Use the Phase-42 parser when parser_mode != "none".
    resolved_parser_mode: str | None
    if parser_mode == "none":
        resolved_parser_mode = None
    else:
        resolved_parser_mode = parser_mode
    gen = llm_patch_generator(
        _call, parser_mode=resolved_parser_mode,
        parser_counter=ctr, prompt_style=prompt_style)
    return gen, client, ctr


def _pretty_cell(pooled: dict, header: str) -> str:
    rows = [header]
    rows.append(f"{'strategy':>14} {'pass@1':>8} "
                 f"{'apply':>7} {'tok≈':>8} {'evts':>5} {'hand':>5}")
    for strat, p in pooled.items():
        rows.append(f"{strat:>14} {p['pass_at_1']:>8.3f} "
                    f"{p['patch_applied_rate']:>7.3f} "
                    f"{p['mean_patch_gen_prompt_tokens_approx']:>8.1f} "
                    f"{p['mean_events_to_patch_gen']:>5.1f} "
                    f"{p['mean_handoffs']:>5.1f}")
    return "\n".join(rows)


def _failure_taxonomy(measurements) -> dict:
    by: dict[str, collections.Counter] = {}
    for m in measurements:
        by.setdefault(m.strategy, collections.Counter())[
            m.error_kind or "ok"] += 1
    return {strat: dict(cnt) for strat, cnt in by.items()}


def _pass_flags(report) -> dict[tuple[str, str], bool]:
    out: dict[tuple[str, str], bool] = {}
    for m in report.measurements:
        out[(m.instance_id, m.strategy)] = bool(m.test_passed)
    return out


def _pair_delta(baseline_flags: dict, candidate_flags: dict) -> dict:
    """Attribution-table analogue for parser axis.

    ``baseline_flags`` is the strict-parser cell; ``candidate_flags``
    is the robust-parser (or unified, ...) cell. ``recovered`` are
    instances the candidate parses pass on but the baseline fails
    on. ``regressed`` are the inverse (a failure the new parser
    introduces) — empirically expected to be empty because recovery
    never fabricates content. Theorem P42-2 formalises that
    expectation.
    """
    by_strat: dict[str, dict] = {}
    for (iid, strat), pass_cand in candidate_flags.items():
        pass_base = baseline_flags.get((iid, strat), False)
        s = by_strat.setdefault(strat, {
            "recovered": [], "regressed": [],
            "unchanged_pass": 0, "unchanged_fail": 0,
        })
        if pass_base and pass_cand:
            s["unchanged_pass"] += 1
        elif pass_base and not pass_cand:
            s["regressed"].append(iid)
        elif (not pass_base) and pass_cand:
            s["recovered"].append(iid)
        else:
            s["unchanged_fail"] += 1
    return by_strat


def _materialise_bank(args):
    path = args.jsonl
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"--jsonl {path} not found; default ships at {_DEFAULT_JSONL}")
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("mock", "real"), default="mock")
    ap.add_argument("--model", default="qwen2.5-coder:7b")
    ap.add_argument("--ollama-url", default=None,
                      help=("Ollama base URL (e.g. http://192.168.12.191:11434 "
                             "for the ASPEN cluster macbook-1 node). When "
                             "None the client talks to localhost:11434."))
    ap.add_argument("--jsonl", default=_DEFAULT_JSONL,
                      help="path to a SWE-bench-shape JSONL bank")
    ap.add_argument("--n-distractors", nargs="+", type=int,
                      default=[6])
    ap.add_argument("--n-instances", type=int, default=None,
                      help="cap on instances loaded (None = all)")
    ap.add_argument("--strategies", nargs="+",
                      default=list(ALL_SWE_STRATEGIES))
    ap.add_argument("--apply-modes", nargs="+",
                      default=["strict"],
                      help=("matcher modes to sweep; default strict. "
                             "Phase 41 already demonstrated matcher "
                             "permissiveness is empirically null-gain on "
                             "the 28-instance bank, so the Phase-42 "
                             "default sweeps parser modes instead."))
    ap.add_argument("--parser-modes", nargs="+",
                      default=["strict", "robust"],
                      help=("parser modes to sweep. ``strict`` is the "
                             "Phase-41 baseline; ``robust`` is the "
                             "Phase-42 default with tolerant block "
                             "closing + heuristics."))
    ap.add_argument("--prompt-style", default="block",
                      choices=("block", "unified_diff"),
                      help=("prompt instructs the LLM to emit either an "
                             "``OLD>>>/<<<NEW>>>/<<<`` block (default) "
                             "or a unified diff."))
    ap.add_argument("--sandbox",
                      choices=("auto", "in_process", "subprocess",
                                "docker"),
                      default="subprocess")
    ap.add_argument("--docker-image", default="python:3.11-slim")
    ap.add_argument("--timeout-s", type=float, default=20.0)
    ap.add_argument("--llm-timeout", type=float, default=300.0)
    ap.add_argument("--think", choices=("on", "off", "default"),
                      default="default",
                      help=("Qwen3-class reasoning models expose a ``think`` "
                             "flag in Ollama's /api/generate. ``default`` omits "
                             "the field (Phase 42 byte-for-byte); ``off`` sends "
                             "``think: false`` so thinking tokens do not eat "
                             "the output budget on models like qwen3.5:35b."))
    ap.add_argument("--max-tokens", type=int, default=400,
                      help="LLM generation max tokens (num_predict).")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    for m in args.apply_modes:
        if m not in ALL_APPLY_MODES:
            raise SystemExit(
                f"--apply-modes: unknown mode {m!r}; valid: "
                f"{sorted(ALL_APPLY_MODES)}")
    if APPLY_MODE_STRICT not in args.apply_modes:
        args.apply_modes = [APPLY_MODE_STRICT] + [
            m for m in args.apply_modes if m != APPLY_MODE_STRICT]

    valid_parser = set(ALL_PARSER_MODES) | {"none"}
    for pm in args.parser_modes:
        if pm not in valid_parser:
            raise SystemExit(
                f"--parser-modes: unknown mode {pm!r}; valid: "
                f"{sorted(valid_parser)}")
    # Ensure strict is the first parser-mode cell so the attribution
    # table pairs robust against strict cleanly.
    if PARSER_STRICT not in args.parser_modes:
        args.parser_modes = [PARSER_STRICT] + [
            p for p in args.parser_modes if p != PARSER_STRICT]
    else:
        args.parser_modes = [PARSER_STRICT] + [
            p for p in args.parser_modes if p != PARSER_STRICT]

    jsonl_path = _materialise_bank(args)
    sandbox = select_sandbox(args.sandbox, docker_image=args.docker_image)
    print(f"[phase42] sandbox={sandbox.name()} (asked={args.sandbox})",
          flush=True)
    if args.sandbox == "docker" and not sandbox.is_available():
        print("[phase42] WARNING: docker requested but unavailable; "
              "falling back to subprocess.", flush=True)
        sandbox = SubprocessSandbox()

    overall_start = time.time()

    # Generation cache keyed by (instance_id, strategy_proxy,
    # n_distractors, prompt_style). The parser_mode is NOT in the
    # key because the raw LLM response is the same; we re-parse per
    # parser mode cheaply (Phase-41 caching discipline extended to
    # the parser axis).
    raw_response_cache: dict[tuple, str] = {}
    # Per-(parser_mode, nd) cells get their own compliance counter.
    counters: dict[tuple[str, int], ParserComplianceCounter] = {}
    # One shared LLMClient handles every cell so its stats (prompt /
    # output token totals, wall) sum across the whole sweep.
    shared_llm_client = None
    if args.mode == "real":
        from vision_mvp.core.llm_client import LLMClient
        think_arg: bool | None
        if args.think == "on":
            think_arg = True
        elif args.think == "off":
            think_arg = False
        else:
            think_arg = None
        shared_llm_client = LLMClient(
            model=args.model, timeout=args.llm_timeout,
            base_url=args.ollama_url, think=think_arg)

    def _make_wrapped_generator(parser_mode: str, nd: int,
                                  prompt_style: str):
        """Return a generator that memoises the LLM text per
        (instance_id, strategy_proxy, n_distractors, prompt_style)
        and runs the Phase-42 parser locally each call.
        """
        from vision_mvp.tasks.swe_bench_bridge import (
            ProposedPatch, build_patch_generator_prompt,
        )
        from vision_mvp.tasks.swe_patch_parser import parse_patch_block
        resolved_mode: str | None
        if parser_mode == "none":
            resolved_mode = None
        else:
            resolved_mode = parser_mode
        counter = counters.setdefault(
            (parser_mode, nd), ParserComplianceCounter())

        if args.mode == "mock":
            gen, _cli, _ctr = _make_generator(
                "mock", args.model, parser_mode,
                prompt_style, args.ollama_url,
                timeout=args.llm_timeout, counter=counter)
            return gen, counter, None

        client = shared_llm_client
        max_tokens = args.max_tokens

        def _call(prompt: str) -> str:
            return client.generate(prompt, max_tokens=max_tokens,
                                    temperature=0.0)

        def _gen(task, ctx, buggy_source, issue_summary):
            strat_proxy = "substrate" if "hunk" in ctx else (
                "naive_or_routing")
            prompt = build_patch_generator_prompt(
                task=task, ctx=ctx, buggy_source=buggy_source,
                issue_summary=issue_summary,
                prompt_style=prompt_style)
            key = (task.instance_id, strat_proxy, nd, prompt_style)
            if key in raw_response_cache:
                text = raw_response_cache[key]
            else:
                text = _call(prompt)
                raw_response_cache[key] = text
            if resolved_mode is None:
                import re
                m = re.search(r"OLD>>>(.*?)<<<NEW>>>(.*?)<<<",
                               text, flags=re.DOTALL)
                if m is None:
                    return ProposedPatch(patch=(),
                                          rationale="parse_failed")
                return ProposedPatch(
                    patch=((m.group(1).strip("\n"),
                            m.group(2).strip("\n")),),
                    rationale="llm_proposed")
            outcome = parse_patch_block(
                text, mode=resolved_mode,
                unified_diff_parser=parse_unified_diff)
            counter.record(outcome)
            if not outcome.ok:
                return ProposedPatch(
                    patch=(),
                    rationale=f"parse_failed:{outcome.failure_kind}")
            rat = "llm_proposed"
            if outcome.recovery:
                rat = f"llm_proposed:{outcome.recovery}"
            return ProposedPatch(
                patch=outcome.substitutions, rationale=rat)

        return _gen, counter, client

    # Outer loop: parser_mode × apply_mode × n_distractors.
    # Strategies are the inner axis handled by run_swe_loop_sandboxed.
    cells: list[dict] = []
    for parser_mode in args.parser_modes:
        for apply_mode in args.apply_modes:
            for n_distractors in args.n_distractors:
                tasks, repo_files = load_jsonl_bank(
                    jsonl_path,
                    hidden_event_log_factory=(
                        lambda t, k=n_distractors: build_synthetic_event_log(t, k)),
                    limit=args.n_instances,
                )
                cell_label = (f"parser={parser_mode} "
                               f"apply={apply_mode} "
                               f"nd={n_distractors}")
                print(f"\n[phase42] {cell_label} "
                      f"jsonl={os.path.basename(jsonl_path)} "
                      f"n_instances={len(tasks)} "
                      f"model={args.model if args.mode == 'real' else '-'} "
                      f"ollama={args.ollama_url or 'localhost'}",
                      flush=True)
                t0 = time.time()
                gen, counter, _client = _make_wrapped_generator(
                    parser_mode, n_distractors, args.prompt_style)

                rep = run_swe_loop_sandboxed(
                    bank=tasks, repo_files=repo_files,
                    generator=gen,
                    sandbox=sandbox,
                    strategies=tuple(args.strategies),
                    timeout_s=args.timeout_s,
                    apply_mode=apply_mode)
                pooled = rep.pooled_summary()
                wall = time.time() - t0
                print(_pretty_cell(pooled, f"  [{cell_label}] pooled:"))
                tax = _failure_taxonomy(rep.measurements)
                print(f"  [{cell_label}] taxonomy: {tax}")
                print(f"  [{cell_label}] parser_compliance: "
                      f"{counter.as_dict()}")
                cells.append({
                    "parser_mode": parser_mode,
                    "apply_mode": apply_mode,
                    "n_distractors": n_distractors,
                    "report": rep.as_dict(),
                    "failure_taxonomy": tax,
                    "parser_compliance": counter.as_dict(),
                    "cell_wall_s": round(wall, 2),
                })

    # Parser-axis attribution table: for each (apply_mode, nd),
    # compute the set delta between the strict-parser cell and each
    # non-strict parser cell.
    attribution: dict[str, dict[str, dict[str, dict]]] = {}
    cell_index: dict[tuple[str, str, int], dict] = {}
    for c in cells:
        key = (c["parser_mode"], c["apply_mode"], c["n_distractors"])
        flags: dict[tuple[str, str], bool] = {}
        for m in c["report"]["measurements"]:
            flags[(m["instance_id"], m["strategy"])] = bool(m["test_passed"])
        cell_index[key] = {
            "flags": flags, "compliance": c["parser_compliance"]}

    for apply_mode in args.apply_modes:
        for nd in args.n_distractors:
            strict_key = (PARSER_STRICT, apply_mode, nd)
            if strict_key not in cell_index:
                continue
            for parser_mode in args.parser_modes:
                if parser_mode == PARSER_STRICT:
                    continue
                key = (parser_mode, apply_mode, nd)
                if key not in cell_index:
                    continue
                delta = _pair_delta(
                    cell_index[strict_key]["flags"],
                    cell_index[key]["flags"])
                attribution.setdefault(
                    f"nd={nd}", {}
                ).setdefault(
                    f"apply={apply_mode}", {}
                )[f"parser={parser_mode}"] = delta

    overall = time.time() - overall_start

    print()
    print("=" * 72)
    print("PHASE 42 — parser-compliance axis + larger bank")
    print("=" * 72)
    for c in cells:
        head = (f"  parser={c['parser_mode']}  "
                f"apply={c['apply_mode']}  "
                f"nd={c['n_distractors']}  "
                f"wall={c['cell_wall_s']}s")
        print(head)
        for strat, p in c["report"]["pooled"].items():
            print(f"    {strat:>12}  pass@1={p['pass_at_1']:.3f}  "
                  f"apply={p['patch_applied_rate']:.3f}  "
                  f"tok≈{p['mean_patch_gen_prompt_tokens_approx']:.1f}")
        comp = c["parser_compliance"]
        print(f"    compliance={comp['compliance_rate']:.3f}  "
              f"raw={comp['raw_compliance_rate']:.3f}  "
              f"lift={comp['recovery_lift']:+.3f}")
    print()
    print("--- parser-axis attribution ---")
    for nd_key, by_apply in attribution.items():
        for apply_key, by_parser in by_apply.items():
            for parser_key, by_strat in by_parser.items():
                print(f"  {nd_key}  {apply_key}  "
                      f"{PARSER_STRICT} vs {parser_key}:")
                for strat, d in by_strat.items():
                    rec = d.get("recovered", [])
                    reg = d.get("regressed", [])
                    print(f"    {strat:>12}  recovered={len(rec)}  "
                          f"regressed={len(reg)}  "
                          f"up_pass={d.get('unchanged_pass', 0)}  "
                          f"up_fail={d.get('unchanged_fail', 0)}")
                    if rec:
                        print(f"      recovered_ids: {rec}")
                    if reg:
                        print(f"      regressed_ids: {reg}")
    print(f"\n  overall wall: {overall:.1f}s")

    payload = {
        "config": vars(args),
        "sandbox": sandbox.name(),
        "jsonl_path": jsonl_path,
        "cells": cells,
        "attribution": attribution,
        "wall_seconds": round(overall, 2),
    }
    if shared_llm_client is not None and hasattr(shared_llm_client, "stats"):
        payload["llm_client_stats"] = {
            "prompt_tokens": shared_llm_client.stats.prompt_tokens,
            "output_tokens": shared_llm_client.stats.output_tokens,
            "n_generate_calls": shared_llm_client.stats.n_generate_calls,
            "total_wall": round(shared_llm_client.stats.total_wall, 2),
        }
    if args.out:
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
