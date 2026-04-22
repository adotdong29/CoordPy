"""Phase 30 — external LLM-in-loop SWE substrate benchmark.

Runs the Phase-30 loop harness (``vision_mvp.tasks.swe_loop_harness``)
against a **real LLM** on a **real third-party Python corpus**, and
reports the cross-strategy accuracy-vs-prompt-tokens table that
distinguishes substrate-gated answering from naive-full-context
answering.

This is the programme's first evaluation that simultaneously:

  1. runs a real LLM (local Ollama, default ``qwen2.5-coder:7b``) on
     the answer path;
  2. uses a realistic **external** Python corpus — defaults to
     ``click`` (third-party CLI framework) + the stdlib ``json``
     module, both importable from the local Python install, neither
     inside this repo;
  3. decomposes task correctness against active context along the
     three delivery strategies (naive / routing / substrate_wrap)
     measured byte-accurately;
  4. produces a single reproducible JSON artifact that Phase-31 can
     diff against.

The goal is NOT to beat SWE-bench. SWE-bench is still ROADMAP
medium-term. The goal is to produce the strongest *external-
validity* evidence the programme has so far: on a corpus that the
authors did not write, with a model that the authors did not fine-
tune, the substrate should (a) shrink the LLM's active context by
≥100× on matched queries, and (b) not lose correctness relative to
naive full-context delivery on that slice.

Typical CI-friendly run (click only, one corpus):

    python3 -W ignore -m vision_mvp.experiments.phase30_llm_swe_benchmark \\
        --model qwen2.5-coder:7b --corpora click \\
        --out vision_mvp/results_phase30_click.json

Full run (click + json-stdlib + vision-core as an internal control):

    python3 -W ignore -m vision_mvp.experiments.phase30_llm_swe_benchmark \\
        --model qwen2.5-coder:7b --corpora click json-stdlib vision-core \\
        --out vision_mvp/results_phase30_full.json

Deterministic mock run (no LLM, exercises only the harness):

    python3 -W ignore -m vision_mvp.experiments.phase30_llm_swe_benchmark \\
        --mock --corpora vision-core --out /tmp/phase30_mock.json

Scope discipline:
    * no LLM fine-tuning
    * no RAG outside the substrate's own retrieval
    * no model-judged grading — every answer is graded by the
      deterministic ``grade_answer`` function from the harness
    * same seed → byte-identical result artifact, modulo network
      non-determinism in the Ollama generation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Callable

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")))

from vision_mvp.core.llm_client import LLMClient
from vision_mvp.tasks.python_corpus import PythonCorpus
from vision_mvp.tasks.swe_loop_harness import (
    MockAnswerLLM, compute_cross_strategy_deltas, run_loop,
)


# -----------------------------------------------------------------------------
# Corpus discovery
# -----------------------------------------------------------------------------


def _locate_external(name: str) -> str | None:
    """Return a best-effort path to a third-party corpus by import
    lookup. Falls back to None if the package is not installed.
    """
    if name == "click":
        try:
            import click as _m
            return os.path.dirname(_m.__file__)
        except ImportError:
            return None
    if name == "json-stdlib":
        try:
            import json as _j
            return os.path.dirname(_j.__file__)
        except ImportError:
            return None
    return None


def _locate_internal(name: str) -> str | None:
    repo = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", ".."))
    mapping = {
        "vision-core":        os.path.join(repo, "vision_mvp/core"),
        "vision-tasks":       os.path.join(repo, "vision_mvp/tasks"),
        "vision-tests":       os.path.join(repo, "vision_mvp/tests"),
        "vision-experiments": os.path.join(repo, "vision_mvp/experiments"),
    }
    root = mapping.get(name)
    if root and os.path.isdir(root):
        return root
    return None


def _resolve_corpus(name: str) -> tuple[str, str] | None:
    """Return (label, absolute_path) for ``name``."""
    root = _locate_internal(name) or _locate_external(name)
    if root is None:
        return None
    return name, root


# -----------------------------------------------------------------------------
# Aggregator wrapper
# -----------------------------------------------------------------------------


def _make_aggregator(model: str, mock: bool = False,
                     max_answer_tokens: int = 80) -> tuple[Callable[[str], str], object]:
    """Return (aggregator_callable, stats_object)."""
    if mock:
        mock_llm = MockAnswerLLM()
        return mock_llm, mock_llm
    client = LLMClient(model=model, timeout=300.0)

    def _call(prompt: str) -> str:
        return client.generate(
            prompt, max_tokens=max_answer_tokens, temperature=0.0)
    return _call, client.stats


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5-coder:7b")
    ap.add_argument("--mock", action="store_true",
                      help="Run with a deterministic mock LLM (no Ollama).")
    ap.add_argument("--corpora", nargs="+",
                      default=["vision-core"],
                      help="Corpus names (click / json-stdlib / "
                           "vision-core / vision-tasks / ...)")
    ap.add_argument("--seed", type=int, default=30)
    ap.add_argument("--n-agent-comments", type=int, default=6)
    ap.add_argument("--max-events-in-prompt", type=int, default=400)
    ap.add_argument("--max-answer-tokens", type=int, default=80)
    ap.add_argument("--max-files", type=int, default=None,
                      help="Optionally cap corpus size for fast CI runs.")
    ap.add_argument("--strategies", nargs="+",
                      default=["naive", "routing", "substrate_wrap"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # Resolve corpora up-front; fail fast if anything missing.
    resolved: list[tuple[str, str]] = []
    for nm in args.corpora:
        pair = _resolve_corpus(nm)
        if pair is None:
            print(f"[phase30] could not resolve corpus {nm!r}",
                  flush=True)
            continue
        resolved.append(pair)
    if not resolved:
        print("[phase30] no corpora resolved; exiting.", flush=True)
        return 2

    aggregator, stats = _make_aggregator(
        args.model, mock=args.mock,
        max_answer_tokens=args.max_answer_tokens)

    overall_start = time.time()
    per_corpus: list[dict] = []
    pooled_accuracy: dict[str, list[float]] = {
        s: [] for s in args.strategies
    }
    pooled_tokens: dict[str, list[float]] = {
        s: [] for s in args.strategies
    }
    pooled_substrate_correct_ratio: list[float] = []

    for (name, root) in resolved:
        if not os.path.isdir(root):
            print(f"[{name}] skip — {root} is not a directory",
                  flush=True)
            continue
        print(f"[{name}] building corpus at {root}", flush=True)
        t0 = time.time()
        c = PythonCorpus(root=root,
                          max_files=args.max_files,
                          seed=args.seed)
        c.build()
        build_s = time.time() - t0
        print(f"[{name}] corpus built in {build_s:.2f}s "
              f"(n_files={c.n_files} "
              f"n_functions={c.n_functions_total})",
              flush=True)

        rep = run_loop(
            corpus_name=name, corpus=c,
            aggregator=aggregator,
            strategies=tuple(args.strategies),
            seed=args.seed,
            n_agent_comments=args.n_agent_comments,
            max_events_in_prompt=args.max_events_in_prompt,
        )
        print(f"[{name}] n_tasks={rep.n_tasks} "
              f"n_events={rep.n_events}")
        pooled = rep.pooled_summary()
        for strat in args.strategies:
            if strat in pooled:
                print(f"  {strat:>18} "
                      f"acc={pooled[strat]['accuracy']:.3f}  "
                      f"tok={pooled[strat]['mean_prompt_tokens']:>8.1f}  "
                      f"sub_match={pooled[strat]['substrate_match_count']}  "
                      f"trunc={pooled[strat]['truncated_count']}  "
                      f"wall={pooled[strat]['mean_wall_seconds']:.2f}s")
                pooled_accuracy[strat].append(pooled[strat]["accuracy"])
                pooled_tokens[strat].append(
                    pooled[strat]["mean_prompt_tokens"])

        # Deltas per corpus.
        deltas = compute_cross_strategy_deltas(rep)
        for d in deltas:
            print(f"  {d.base:>18} → {d.comp:<18}  "
                  f"acc {d.accuracy_base:.3f}→{d.accuracy_comp:.3f} "
                  f"(+{d.accuracy_delta:+.3f})  "
                  f"tok_ratio {d.token_ratio:.2f}×")

        # Substrate-matched slice accuracy (substrate vs naive on that slice).
        matched_ids = {m.task_id for m in rep.measurements
                        if m.substrate_matched}
        if matched_ids:
            sub_correct = sum(
                1 for m in rep.measurements
                if m.task_id in matched_ids
                and m.strategy.startswith("substrate")
                and m.answer_correct)
            sub_total = sum(
                1 for m in rep.measurements
                if m.task_id in matched_ids
                and m.strategy.startswith("substrate"))
            ratio = sub_correct / max(1, sub_total)
            print(f"  substrate-matched-slice accuracy "
                  f"= {sub_correct}/{sub_total} = {ratio:.3f}")
            pooled_substrate_correct_ratio.append(ratio)

        per_corpus.append({
            "corpus_name": name,
            "root": root,
            "build_seconds": round(build_s, 3),
            "report": rep.as_dict(),
            "deltas": [
                {
                    "base": d.base, "comp": d.comp,
                    "accuracy_base": d.accuracy_base,
                    "accuracy_comp": d.accuracy_comp,
                    "accuracy_delta": d.accuracy_delta,
                    "token_ratio": d.token_ratio,
                    "mean_tokens_base": d.mean_tokens_base,
                    "mean_tokens_comp": d.mean_tokens_comp,
                } for d in deltas
            ],
        })

    overall = time.time() - overall_start

    # ---------- pooled summary ----------
    def _mean(xs):
        return round(sum(xs) / max(1, len(xs)), 4)
    pooled_summary = {
        "strategies": list(args.strategies),
        "n_corpora": len(per_corpus),
        "mean_accuracy": {s: _mean(pooled_accuracy[s])
                          for s in args.strategies},
        "mean_prompt_tokens": {s: _mean(pooled_tokens[s])
                                for s in args.strategies},
        "substrate_matched_slice_accuracy": _mean(
            pooled_substrate_correct_ratio),
        "wall_seconds": round(overall, 2),
    }

    print()
    print("=" * 72)
    print("PHASE 30 POOLED — across all runnable corpora")
    print("=" * 72)
    for s in args.strategies:
        print(f"  {s:>18}  acc={pooled_summary['mean_accuracy'][s]:.3f}  "
              f"tok={pooled_summary['mean_prompt_tokens'][s]:>8.1f}")
    print(f"  substrate-matched slice accuracy "
          f"(pooled across corpora) = "
          f"{pooled_summary['substrate_matched_slice_accuracy']:.3f}")
    print(f"  wall-time: {overall:.1f}s")

    llm_stats = None
    if hasattr(stats, "total_tokens"):
        llm_stats = {
            "prompt_tokens": stats.prompt_tokens,
            "output_tokens": stats.output_tokens,
            "embed_tokens": stats.embed_tokens,
            "n_generate_calls": stats.n_generate_calls,
            "n_embed_calls": stats.n_embed_calls,
            "total_wall": round(stats.total_wall, 2),
        }
        print(f"  llm total generate calls={stats.n_generate_calls}  "
              f"prompt_tokens(true)={stats.prompt_tokens}  "
              f"output_tokens={stats.output_tokens}  "
              f"wall={stats.total_wall:.1f}s")
    elif hasattr(stats, "n_calls"):
        llm_stats = {
            "n_calls": stats.n_calls,
            "total_prompt_chars": stats.total_prompt_chars,
        }
        print(f"  mock llm total calls={stats.n_calls}  "
              f"chars={stats.total_prompt_chars}")

    payload = {
        "config": {
            "model": args.model,
            "mock": args.mock,
            "corpora": args.corpora,
            "seed": args.seed,
            "n_agent_comments": args.n_agent_comments,
            "max_events_in_prompt": args.max_events_in_prompt,
            "max_answer_tokens": args.max_answer_tokens,
            "strategies": list(args.strategies),
            "max_files": args.max_files,
        },
        "per_corpus": per_corpus,
        "pooled": pooled_summary,
        "llm_stats": llm_stats,
    }
    if args.out:
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
