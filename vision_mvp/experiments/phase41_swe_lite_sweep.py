"""Phase 41 — Larger SWE-bench-Lite-style empirical sweep + matcher
permissiveness attribution study.

Phase 40 proved the real loop exists. Phase 41 moves the next
credibility step: scale. Three tightly coupled experiments run from
this one driver:

  * **Part A — larger empirical sweep.** ~20+ real-shape
    instances (``swe_lite_style_bank.jsonl``) through the same
    Phase-40 loader + sandbox + substrate pipeline. Naive,
    routing, and substrate compared at a disciplined distractor
    grid. The run washes out the 6-instance variance that
    surfaced as the Phase-40 substrate-vs-naive ranking
    inversion.

  * **Part B — matcher permissiveness attribution.** Every
    proposed patch is re-evaluated under two matcher modes —
    strict (Phase-40 byte-exact) and a permissive mode
    (``lstrip`` by default; ``line_anchored`` and
    ``ws_collapse`` also supported). Pass-rate *gains* under
    permissive matching attribute to "generator-side literal-
    text fidelity" rather than to the substrate; pass-rate
    *losses* (over-accepting bad patches) are reported
    explicitly.

  * **Part C — stronger-model datapoint.** The same sweep can
    be executed with a larger local model (e.g.
    ``gemma2:9b`` — the strongest local model in the Phase-39
    frontier ranking) on a representative subset so the
    substrate-advantage question has a second data point
    beyond the Phase-40 0.5B / 7B pair.

The driver is deliberately *compositional*: each (mode,
strategy, distractor) cell is one independent measurement;
the LLM is called exactly once per (task, strategy) and the
proposed patch is evaluated under both matcher modes without
re-generation (the compute-efficient form).

Reproducibility: fully hermetic on a local JSONL artifact,
runs offline in seconds under ``--mode mock``. Real-LLM runs
call a localhost Ollama at ``http://localhost:11434``.

Reproducible commands
---------------------

    # 1. Phase-41 mock sweep (oracle, all instances, both matchers).
    python3 -m vision_mvp.experiments.phase41_swe_lite_sweep \
        --mode mock --sandbox subprocess \
        --apply-modes strict lstrip \
        --jsonl vision_mvp/tasks/data/swe_lite_style_bank.jsonl \
        --n-distractors 0 6 12 24 \
        --out vision_mvp/results_phase41_swe_lite_mock.json

    # 2. Phase-41 real LLM — qwen2.5-coder:7b on the full larger bank.
    python3 -m vision_mvp.experiments.phase41_swe_lite_sweep \
        --mode real --model qwen2.5-coder:7b --sandbox subprocess \
        --apply-modes strict lstrip \
        --jsonl vision_mvp/tasks/data/swe_lite_style_bank.jsonl \
        --n-distractors 6 \
        --out vision_mvp/results_phase41_swe_lite_7b.json

    # 3. Phase-41 stronger-model datapoint — gemma2:9b on a subset.
    python3 -m vision_mvp.experiments.phase41_swe_lite_sweep \
        --mode real --model gemma2:9b --sandbox subprocess \
        --apply-modes strict lstrip \
        --jsonl vision_mvp/tasks/data/swe_lite_style_bank.jsonl \
        --n-distractors 6 --n-instances 12 \
        --out vision_mvp/results_phase41_swe_lite_9b.json
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
    load_jsonl_bank,
)
from vision_mvp.tasks.swe_sandbox import (
    SubprocessSandbox, run_swe_loop_sandboxed, select_sandbox,
)


_DEFAULT_JSONL = os.path.join(
    os.path.dirname(__file__), "..", "tasks", "data",
    "swe_lite_style_bank.jsonl")


def _make_generator(mode: str, model: str | None,
                     timeout: float = 300.0):
    """Return ``(generator_callable, client_or_None)``."""
    if mode == "mock":
        return deterministic_oracle_generator, None
    from vision_mvp.core.llm_client import LLMClient
    client = LLMClient(model=model, timeout=timeout)

    def _call(prompt: str) -> str:
        return client.generate(prompt, max_tokens=300,
                                temperature=0.0)

    return llm_patch_generator(_call), client


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
    """Per-strategy histogram of ``error_kind`` labels."""
    by: dict[str, collections.Counter] = {}
    for m in measurements:
        by.setdefault(m.strategy, collections.Counter())[
            m.error_kind or "ok"] += 1
    return {strat: dict(cnt) for strat, cnt in by.items()}


def _pass_flags(report) -> dict[tuple[str, str], bool]:
    """Map (instance_id, strategy) → test_passed bool for set-arithmetic
    attribution between strict and permissive matcher runs.
    """
    out: dict[tuple[str, str], bool] = {}
    for m in report.measurements:
        out[(m.instance_id, m.strategy)] = bool(m.test_passed)
    return out


def _pair_delta(strict_flags: dict, perm_flags: dict) -> dict:
    """Set-arithmetic delta between two per-(instance, strategy)
    pass maps. Returns {
        "recovered":         {strat: [instances ...]},
        "regressed":         {strat: [instances ...]},
        "unchanged_pass":    {strat: count},
        "unchanged_fail":    {strat: count},
    }
    ``recovered`` are instances that strict fails but permissive
    passes (the generator-side attribution question).
    ``regressed`` are instances that strict passes but permissive
    fails (the risk-of-over-accepting question).
    """
    by_strat: dict[str, dict] = {}
    for (iid, strat), pass_perm in perm_flags.items():
        pass_strict = strict_flags.get((iid, strat), False)
        s = by_strat.setdefault(strat, {
            "recovered": [], "regressed": [],
            "unchanged_pass": 0, "unchanged_fail": 0,
        })
        if pass_strict and pass_perm:
            s["unchanged_pass"] += 1
        elif pass_strict and not pass_perm:
            s["regressed"].append(iid)
        elif (not pass_strict) and pass_perm:
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
    ap.add_argument("--jsonl", default=_DEFAULT_JSONL,
                      help="path to a SWE-bench-shape JSONL bank")
    ap.add_argument("--n-distractors", nargs="+", type=int,
                      default=[6])
    ap.add_argument("--n-instances", type=int, default=None,
                      help="cap on instances loaded (None = all)")
    ap.add_argument("--strategies", nargs="+",
                      default=list(ALL_SWE_STRATEGIES))
    ap.add_argument("--apply-modes", nargs="+",
                      default=["strict", "lstrip"],
                      help="matcher modes to compare. Strict is always "
                            "reported as the attribution baseline.")
    ap.add_argument("--sandbox",
                      choices=("auto", "in_process", "subprocess",
                                "docker"),
                      default="subprocess")
    ap.add_argument("--docker-image", default="python:3.11-slim")
    ap.add_argument("--timeout-s", type=float, default=20.0)
    ap.add_argument("--llm-timeout", type=float, default=300.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    for m in args.apply_modes:
        if m not in ALL_APPLY_MODES:
            raise SystemExit(
                f"--apply-modes: unknown mode {m!r}; valid: "
                f"{sorted(ALL_APPLY_MODES)}")
    if APPLY_MODE_STRICT not in args.apply_modes:
        # The attribution study requires strict as the baseline cell.
        args.apply_modes = [APPLY_MODE_STRICT] + [
            m for m in args.apply_modes if m != APPLY_MODE_STRICT]

    jsonl_path = _materialise_bank(args)
    sandbox = select_sandbox(args.sandbox, docker_image=args.docker_image)
    print(f"[phase41] sandbox={sandbox.name()} (asked={args.sandbox})",
          flush=True)
    if args.sandbox == "docker" and not sandbox.is_available():
        print("[phase41] WARNING: docker requested but unavailable; "
              "falling back to subprocess.", flush=True)
        sandbox = SubprocessSandbox()

    overall_start = time.time()
    generator, client = _make_generator(
        args.mode, args.model, timeout=args.llm_timeout)

    # Run each (distractor, apply_mode) cell in turn; LLM call-count
    # is bounded by n_distractor_cells × n_instances × n_strategies
    # (the permissive modes reuse the LLM output, so the real cost
    # is just the extra sandbox runs).
    # To achieve that, we run mock-style: the generator emits once,
    # but we reuse the proposed patch under each matcher mode.
    # The current run_swe_loop_sandboxed calls the generator inline;
    # for a real-LLM path we memoize per (instance_id, strategy,
    # n_distractors) so re-running across apply_modes costs only
    # the (fast) sandbox.
    from vision_mvp.tasks.swe_bench_bridge import ProposedPatch

    gen_cache: dict[tuple, ProposedPatch] = {}

    def _cached_generator(task, ctx, buggy_source, issue_summary):
        strat = ctx.get("__strategy__") or "<no-strat>"
        nd = ctx.get("__nd__") or "<no-nd>"
        key = (task.instance_id, strat, nd)
        if key in gen_cache:
            return gen_cache[key]
        proposed = generator(task, ctx, buggy_source, issue_summary)
        gen_cache[key] = proposed
        return proposed

    # The bridge runner does not inject strategy/distractor into ctx
    # directly; we wrap it manually per outer loop instead.
    # For the cache to work we need to carry (strategy, n_distractors)
    # context. The simplest honest approach: memoize on (instance_id,
    # strategy) — strategies differ but n_distractors does not affect
    # the substrate's patch_generator prompt (Theorem P40-2) and
    # changes only the naive/routing raw-event stream.

    cells: list[dict] = []
    for apply_mode in args.apply_modes:
        for n_distractors in args.n_distractors:
            tasks, repo_files = load_jsonl_bank(
                jsonl_path,
                hidden_event_log_factory=(
                    lambda t, k=n_distractors: build_synthetic_event_log(t, k)),
                limit=args.n_instances,
            )
            cell_label = f"apply_mode={apply_mode} nd={n_distractors}"
            print(f"\n[phase41] {cell_label} "
                  f"jsonl={os.path.basename(jsonl_path)} "
                  f"n_instances={len(tasks)} "
                  f"model={args.model if args.mode == 'real' else '-'}",
                  flush=True)
            t0 = time.time()

            # We wrap the generator so it carries the current (strategy,
            # n_distractors) via ctx injection; the bridge calls this
            # via generator(task, ctx, buggy_source, issue_summary).
            # The bridge builds ctx per strategy and passes it; we
            # annotate it here inline.
            def _inject_and_call(task, ctx, buggy_source, issue_summary,
                                   _nd=n_distractors):
                # Strategy is inferable from ctx shape: substrate has
                # issue_summary+hunk; routing has empty ctx; naive has
                # empty ctx but delivered_events ≠ [] — we can't see
                # delivered_events here. Use a hash of ctx keys instead.
                strat_proxy = "substrate" if "hunk" in ctx else (
                    "naive_or_routing")
                # Refinement: the bridge re-builds prompts before
                # calling generator, and separately tracks delivered;
                # the generator sees only ctx. The strategy is stable
                # per outer loop cell so we use the outer binding.
                ctx2 = dict(ctx)
                ctx2["__strategy__"] = strat_proxy
                ctx2["__nd__"] = _nd
                return _cached_generator(task, ctx2, buggy_source,
                                          issue_summary)

            rep = run_swe_loop_sandboxed(
                bank=tasks, repo_files=repo_files,
                generator=_inject_and_call,
                sandbox=sandbox,
                strategies=tuple(args.strategies),
                timeout_s=args.timeout_s,
                apply_mode=apply_mode)
            pooled = rep.pooled_summary()
            wall = time.time() - t0
            print(_pretty_cell(pooled, f"  [{cell_label}] pooled:"))
            tax = _failure_taxonomy(rep.measurements)
            print(f"  [{cell_label}] taxonomy: {tax}")
            cells.append({
                "apply_mode": apply_mode,
                "n_distractors": n_distractors,
                "report": rep.as_dict(),
                "failure_taxonomy": tax,
                "cell_wall_s": round(wall, 2),
            })

    # Matcher-permissiveness attribution: per strategy, per distractor
    # cell, compute the set delta between strict and each permissive
    # mode. A positive "recovered" is a generator-side gain; a positive
    # "regressed" is a permissive-mode risk.
    attribution = {}
    # Re-index cells by (apply_mode, n_distractors) for easy lookup.
    cell_index: dict[tuple[str, int], object] = {}
    for c in cells:
        key = (c["apply_mode"], c["n_distractors"])
        # Reconstruct pass flags from the report dict.
        flags: dict[tuple[str, str], bool] = {}
        for m in c["report"]["measurements"]:
            flags[(m["instance_id"], m["strategy"])] = bool(m["test_passed"])
        cell_index[key] = flags
    for nd in args.n_distractors:
        strict_key = (APPLY_MODE_STRICT, nd)
        if strict_key not in cell_index:
            continue
        for mode in args.apply_modes:
            if mode == APPLY_MODE_STRICT:
                continue
            key = (mode, nd)
            if key not in cell_index:
                continue
            delta = _pair_delta(cell_index[strict_key], cell_index[key])
            attribution.setdefault(str(nd), {})[mode] = delta

    overall = time.time() - overall_start

    print()
    print("=" * 72)
    print("PHASE 41 — SWE-bench-Lite-style empirical sweep")
    print("=" * 72)
    # Flatten cell-pooled into a compact cross-cell summary.
    for c in cells:
        head = (f"  apply_mode={c['apply_mode']}  "
                f"nd={c['n_distractors']}  "
                f"wall={c['cell_wall_s']}s")
        print(head)
        for strat, p in c["report"]["pooled"].items():
            print(f"    {strat:>12}  pass@1={p['pass_at_1']:.3f}  "
                  f"apply={p['patch_applied_rate']:.3f}  "
                  f"tok≈{p['mean_patch_gen_prompt_tokens_approx']:.1f}")
    print()
    print("--- matcher-permissiveness attribution ---")
    for nd, by_mode in attribution.items():
        for mode, by_strat in by_mode.items():
            print(f"  nd={nd}  {APPLY_MODE_STRICT} vs {mode}:")
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
    if client is not None and hasattr(client, "stats"):
        payload["llm_client_stats"] = {
            "prompt_tokens": client.stats.prompt_tokens,
            "output_tokens": client.stats.output_tokens,
            "n_generate_calls": client.stats.n_generate_calls,
            "total_wall": round(client.stats.total_wall, 2),
        }
    if args.out:
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
