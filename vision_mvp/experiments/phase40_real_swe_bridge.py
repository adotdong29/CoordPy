"""Phase 40 — real SWE-bench-style loader + sandboxed evaluation.

Phase 39 shipped the bridge (``tasks/swe_bench_bridge``): a SWE-bench-
compatible schema, a four-instance hand-authored mini bank, and a
four-role team running on the unchanged Phase-31 ``HandoffRouter``.
Theorems P39-3 / P39-4 named the gap:

  * P39-4: every required SWE-bench field has a typed counterpart in
    ``SWEBenchStyleTask``; the only schema gap is ``gold_patch``
    representation (substitution vs unified diff). Mechanical.
  * P39-3: substrate bounded-context preservation extends to the
    SWE-style team. Architectural.

Phase 40 closes the *mechanical* gap end-to-end:

  * a unified-diff parser (``parse_unified_diff``);
  * a real-shape adapter (``SWEBenchAdapter.from_swe_bench_dict``);
  * a JSONL loader (``load_jsonl_bank``) that materialises a bank
    from local artifacts;
  * a sandboxed runner (``swe_sandbox``) with three backends
    (in-process / subprocess / docker), so candidate patches no
    longer execute inside the bridge process;
  * this driver, which composes loader + adapter + sandbox + the
    Phase-31 substrate, runs naive / routing / substrate strategies,
    and reports pass@1, prompt cost, and a failure taxonomy.

What this driver does NOT claim
-------------------------------

* It is not SWE-bench Lite end-to-end. The bundled JSONL
  (``vision_mvp/tasks/data/swe_real_shape_mini.jsonl``) is a
  *real-SWE-bench-shape* artifact (unified-diff patches, JSONL,
  in-bundle ``repo_files``) but the instances are six small,
  self-authored bugs. Pointing the driver at SWE-bench Lite is
  a ``--jsonl <path>`` parameter change; the loader / adapter /
  sandbox path is unchanged. The reason the bundled artifact is
  small is *credibility under reproducibility* — the entire
  Phase 40 evaluation runs in seconds on a laptop with no
  network, so every claim in RESULTS_PHASE40.md is rerunnable
  by anyone reading the diff.

* It does not claim a leaderboard pass@1. The Phase-39
  oracle-ceiling result extends to Phase 40 trivially (a
  deterministic oracle saturates pass@1 = 1.000 because the
  patch is correct by construction). The interesting numbers
  are the LLM-mode runs (``--mode real``), which surface the
  *transcription-bounded vs communication-bounded* split of
  Theorem P39-2 on a real-shape pipeline.

Reproducible commands
---------------------

    # Phase-40 mock — sandboxed deterministic oracle on the
    # bundled JSONL (sub-second).
    python3 -m vision_mvp.experiments.phase40_real_swe_bridge \\
        --mode mock --sandbox subprocess \\
        --jsonl vision_mvp/tasks/data/swe_real_shape_mini.jsonl \\
        --n-distractors 0 6 12 \\
        --out vision_mvp/results_phase40_real_swe_bridge_mock.json

    # Phase-40 real LLM — qwen2.5:0.5b on the same bank.
    python3 -m vision_mvp.experiments.phase40_real_swe_bridge \\
        --mode real --model qwen2.5:0.5b --sandbox subprocess \\
        --jsonl vision_mvp/tasks/data/swe_real_shape_mini.jsonl \\
        --n-distractors 6 \\
        --out vision_mvp/results_phase40_real_swe_bridge_0p5b.json

    # Phase-40 real LLM — qwen2.5-coder:7b spot check.
    python3 -m vision_mvp.experiments.phase40_real_swe_bridge \\
        --mode real --model qwen2.5-coder:7b --sandbox subprocess \\
        --jsonl vision_mvp/tasks/data/swe_real_shape_mini.jsonl \\
        --n-distractors 6 \\
        --out vision_mvp/results_phase40_real_swe_bridge_7b.json

    # Phase-40 docker run (only when a daemon is reachable).
    python3 -m vision_mvp.experiments.phase40_real_swe_bridge \\
        --mode mock --sandbox docker \\
        --jsonl vision_mvp/tasks/data/swe_real_shape_mini.jsonl \\
        --n-distractors 6 \\
        --out vision_mvp/results_phase40_real_swe_bridge_docker.json
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
    ALL_SWE_STRATEGIES, STRATEGY_SUBSTRATE,
    build_synthetic_event_log, deterministic_oracle_generator,
    llm_patch_generator, load_jsonl_bank,
)
from vision_mvp.tasks.swe_sandbox import (
    DockerSandbox, InProcessSandbox, SubprocessSandbox,
    run_swe_loop_sandboxed, select_sandbox,
)


_DEFAULT_JSONL = os.path.join(
    os.path.dirname(__file__), "..", "tasks", "data",
    "swe_real_shape_mini.jsonl")


def _make_generator(mode: str, model: str | None,
                     timeout: float = 300.0):
    if mode == "mock":
        return deterministic_oracle_generator, None
    from vision_mvp.core.llm_client import LLMClient
    client = LLMClient(model=model, timeout=timeout)

    def _call(prompt: str) -> str:
        return client.generate(prompt, max_tokens=200,
                                temperature=0.0)
    return llm_patch_generator(_call), client


def _pretty(pooled: dict) -> str:
    rows = []
    rows.append(f"{'strategy':>14} {'pass@1':>8} "
                 f"{'apply':>7} {'tok≈':>8} {'evts':>5} "
                 f"{'hand':>5}")
    for strat, p in pooled.items():
        rows.append(f"{strat:>14} {p['pass_at_1']:>8.3f} "
                    f"{p['patch_applied_rate']:>7.3f} "
                    f"{p['mean_patch_gen_prompt_tokens_approx']:>8.1f} "
                    f"{p['mean_events_to_patch_gen']:>5.1f} "
                    f"{p['mean_handoffs']:>5.1f}")
    return "\n".join(rows)


def _failure_taxonomy(measurements) -> dict:
    """Pooled failure-kind histogram per strategy.

    The taxonomy is the diagnostic surface the driver reports so
    the substrate-vs-LLM-vs-sandbox attribution is decidable from
    the artifact alone (no rerun required).
    """
    by: dict[str, collections.Counter] = {}
    for m in measurements:
        by.setdefault(m.strategy, collections.Counter())[m.error_kind] += 1
    return {strat: dict(cnt) for strat, cnt in by.items()}


def _materialise_bank(args):
    """Build the ``(tasks, repo_files)`` pair under the JSONL path
    or the bundled fallback. Returns the path used so the artifact
    can record provenance.
    """
    path = args.jsonl
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"--jsonl {path} not found; default ships at "
            f"{_DEFAULT_JSONL}")
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("mock", "real"),
                      default="mock")
    ap.add_argument("--model", default="qwen2.5:0.5b")
    ap.add_argument("--jsonl", default=_DEFAULT_JSONL,
                      help="path to a SWE-bench-shape JSONL bank")
    ap.add_argument("--n-distractors", nargs="+", type=int,
                      default=[6])
    ap.add_argument("--n-instances", type=int, default=None,
                      help="cap on instances loaded (None = all)")
    ap.add_argument("--strategies", nargs="+",
                      default=list(ALL_SWE_STRATEGIES))
    ap.add_argument("--sandbox",
                      choices=("auto", "in_process", "subprocess",
                                "docker"),
                      default="subprocess",
                      help="patch+test execution boundary")
    ap.add_argument("--docker-image", default="python:3.11-slim")
    ap.add_argument("--timeout-s", type=float, default=20.0)
    ap.add_argument("--llm-timeout", type=float, default=300.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    jsonl_path = _materialise_bank(args)

    sandbox = select_sandbox(args.sandbox, docker_image=args.docker_image)
    print(f"[phase40] sandbox={sandbox.name()} "
          f"(asked={args.sandbox})", flush=True)
    if args.sandbox == "docker" and not sandbox.is_available():
        print("[phase40] WARNING: docker requested but unavailable; "
              "falling back to subprocess.", flush=True)
        sandbox = SubprocessSandbox()

    overall_start = time.time()
    generator, client = _make_generator(
        args.mode, args.model, timeout=args.llm_timeout)

    per_distractor: list[dict] = []
    for n_distractors in args.n_distractors:
        # Reload the bank with the requested distractor count so the
        # event log shape matches what the substrate / naive prompts
        # consume.
        tasks, repo_files = load_jsonl_bank(
            jsonl_path,
            hidden_event_log_factory=(
                lambda t, k=n_distractors: build_synthetic_event_log(t, k)),
            limit=args.n_instances,
        )
        print(f"\n[phase40] mode={args.mode} "
              f"model={args.model if args.mode == 'real' else '-'} "
              f"jsonl={os.path.basename(jsonl_path)} "
              f"n_instances={len(tasks)} "
              f"n_distractors={n_distractors}",
              flush=True)
        rep = run_swe_loop_sandboxed(
            bank=tasks, repo_files=repo_files,
            generator=generator,
            sandbox=sandbox,
            strategies=tuple(args.strategies),
            timeout_s=args.timeout_s)
        pooled = rep.pooled_summary()
        print(_pretty(pooled))
        # Per-instance breakdown for the substrate strategy.
        sub_rows = [m for m in rep.measurements
                    if m.strategy == STRATEGY_SUBSTRATE]
        if sub_rows:
            print("  per-instance (substrate):")
            for m in sub_rows:
                print(f"    {m.instance_id:>20}  "
                      f"pass={int(m.test_passed)}  "
                      f"apply={int(m.patch_applied)}  "
                      f"err={m.error_kind!r}")
        per_distractor.append({
            "n_distractors": n_distractors,
            "report": rep.as_dict(),
            "failure_taxonomy": _failure_taxonomy(rep.measurements),
        })

    overall = time.time() - overall_start

    # Cross-distractor pooled summary.
    pooled_by_strat: dict[str, list[dict]] = {}
    for row in per_distractor:
        for strat, p in row["report"]["pooled"].items():
            pooled_by_strat.setdefault(strat, []).append(p)
    pooled_summary: dict[str, dict] = {}
    for strat, rows in pooled_by_strat.items():
        n = len(rows)
        pooled_summary[strat] = {
            "n_cells": n,
            "pass_at_1_mean": round(sum(
                r["pass_at_1"] for r in rows) / n, 4),
            "tokens_mean": round(sum(
                r["mean_patch_gen_prompt_tokens_approx"]
                for r in rows) / n, 1),
            "events_mean": round(sum(
                r["mean_events_to_patch_gen"]
                for r in rows) / n, 2),
        }

    print()
    print("=" * 72)
    print(f"PHASE 40 — REAL SWE-bridge pooled summary "
          f"[sandbox={sandbox.name()}]")
    print("=" * 72)
    for strat, p in pooled_summary.items():
        print(f"  {strat:>12}  pass@1={p['pass_at_1_mean']:.3f}  "
              f"tok≈{p['tokens_mean']:.1f}  "
              f"events={p['events_mean']:.2f}")
    print(f"  wall: {overall:.1f}s")

    payload = {
        "config": vars(args),
        "sandbox": sandbox.name(),
        "jsonl_path": jsonl_path,
        "per_distractor": per_distractor,
        "pooled_summary": pooled_summary,
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
