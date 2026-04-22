"""Phase 23 — Multi-corpus external validity.

Phase 22 established that the substrate's direct-exact render path
answers a battery of structural code questions on a single real
Python codebase (`vision_mvp/core/`) with zero LLM calls. Phase 23
extends this to *several* real Python codebases differing in family,
size, and metadata coverage, to separate "works on one corpus" from
"works across corpora."

What this benchmark does, per corpus:

  1. Build a `PythonCorpus` (gold answers computed deterministically
     from AST metadata).
  2. Ingest via `CodeIndexer` and report coverage statistics
     (`IngestionStats`).
  3. Run three conditions:
        * `lossless-multihop`   — Phase-20 retrieval-mediated path,
                                  over file source bodies.
        * `lossless-planner`    — Phase-21 planner with wrap-LLM.
        * `direct-exact`        — Phase-22 NEW. Planner result
                                  returned verbatim; zero LLM calls.
  4. Record per-question failure class and cost.
  5. Optional oracle is available but skipped by default because
     corpora exceed the default oracle budget; retrieval and direct
     paths are the scientifically interesting conditions.

Aggregate:

  * Per-corpus per-condition scoreboard.
  * Cross-corpus average (how robust is each condition?).
  * A small scaling sweep on the largest corpus (10, 20, 40, 80, 108
     files) measuring ingestion time, planner-matched fraction, and
     direct-exact accuracy.

Reproduce:
    python -m vision_mvp.experiments.phase23_multicorpus \\
        --mode mock --out vision_mvp/results_phase23_mock.json

    python -m vision_mvp.experiments.phase23_multicorpus \\
        --mode mock --extra-roots /path/to/click \\
        --out vision_mvp/results_phase23_mock_plus_click.json

    python -m vision_mvp.experiments.phase23_multicorpus \\
        --mode mock --scaling --out vision_mvp/results_phase23_scaling.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")))

from vision_mvp.core.bounded_worker import BoundedRetrievalWorker
from vision_mvp.core.code_index import CodeIndexer
from vision_mvp.core.code_planner import CodeQueryPlanner
from vision_mvp.core.context_ledger import ContextLedger, hash_embedding
from vision_mvp.experiments.phase19_lossless import MockLLM
from vision_mvp.experiments.phase22_codebase import (
    _adapt_phase19, _classify, CondReport, QResult,
    run_direct_exact, run_lossless_retrieval, run_planner_with_wrap,
)
from vision_mvp.tasks.corpus_registry import (
    CorpusEntry, CorpusSpec, default_phase23_registry,
)
from vision_mvp.tasks.needle_corpus import NeedleCorpus
from vision_mvp.tasks.python_corpus import PythonCorpus


# =============================================================================
# Per-corpus driver
# =============================================================================


@dataclass
class CorpusReport:
    """One corpus's results across conditions."""
    name: str
    family: str
    spec: dict
    coverage: dict
    reports_by_condition: dict[str, dict] = field(default_factory=dict)
    questions_by_condition: dict[str, list[dict]] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


def run_corpus(
    entry: CorpusEntry, args, llm_factory, embed_fn,
    progress: Callable[[str], None] = print,
) -> CorpusReport:
    corpus = entry.corpus
    cr = CorpusReport(
        name=entry.name,
        family=entry.spec.family,
        spec={"root": entry.spec.root, "max_files": entry.spec.max_files,
              "max_chars_per_file": entry.spec.max_chars_per_file,
              "seed": entry.spec.seed},
        coverage=entry.coverage.as_dict(),
    )

    # direct-exact (Phase-22 endpoint; zero LLM on planner-matched)
    progress(f"\n-- [{entry.name}] CONDITION direct-exact")
    rep_de = run_direct_exact(
        corpus=corpus, llm=llm_factory(), embed_fn=embed_fn,
        embed_dim=args.embed_dim,
        prompt_budget_chars=args.prompt_budget, top_k=args.top_k,
        fetch_chars_per_handle=args.fetch_chars, progress=progress)
    cr.reports_by_condition["direct-exact"] = rep_de.aggregate()
    cr.questions_by_condition["direct-exact"] = [asdict(q) for q in rep_de.questions]

    # lossless-planner (Phase-21 wrap path — LLM paraphrases operator output)
    if "lossless-planner" not in args.skip_conditions:
        progress(f"-- [{entry.name}] CONDITION lossless-planner")
        rep_lp = run_planner_with_wrap(
            corpus=corpus, llm=llm_factory(), embed_fn=embed_fn,
            embed_dim=args.embed_dim,
            prompt_budget_chars=args.prompt_budget, top_k=args.top_k,
            fetch_chars_per_handle=args.fetch_chars, progress=progress)
        cr.reports_by_condition["lossless-planner"] = rep_lp.aggregate()
        cr.questions_by_condition["lossless-planner"] = [
            asdict(q) for q in rep_lp.questions]

    # lossless-multihop (Phase-20 retrieval-only over file bodies)
    if "lossless-multihop" not in args.skip_conditions:
        progress(f"-- [{entry.name}] CONDITION lossless-multihop")
        rep_mh = run_lossless_retrieval(
            name="lossless-multihop", search_mode="hybrid", max_hops=3,
            corpus=corpus, llm=llm_factory(), embed_fn=embed_fn,
            embed_dim=args.embed_dim,
            prompt_budget_chars=args.prompt_budget, top_k=args.top_k,
            fetch_chars_per_handle=args.fetch_chars, progress=progress)
        cr.reports_by_condition["lossless-multihop"] = rep_mh.aggregate()
        cr.questions_by_condition["lossless-multihop"] = [
            asdict(q) for q in rep_mh.questions]

    return cr


# =============================================================================
# Scaling sweep on a single corpus
# =============================================================================


@dataclass
class ScalingPoint:
    n_files_requested: int
    n_files_ingested: int
    coverage: dict
    total_lines: int
    build_seconds: float
    index_seconds: float
    direct_exact: dict
    plan_matched_fraction: float


def run_scaling_sweep(
    root: str, args, llm_factory, embed_fn,
    points: tuple[int, ...] = (10, 20, 40, 80, 108),
    progress: Callable[[str], None] = print,
) -> list[ScalingPoint]:
    """Repeatedly ingest sub-samples of `root` and measure direct-exact
    accuracy + ingestion cost. The sweep exercises the hypothesis that
    direct-exact coverage is monotone non-decreasing in corpus size
    (adding files never reduces the planner's match rate)."""
    out: list[ScalingPoint] = []
    for pt in points:
        progress(f"\n== scaling sweep: n_files ≤ {pt}")
        t_build = time.time()
        corpus = PythonCorpus(root=root, max_files=pt, seed=args.seed)
        corpus.build()
        build_seconds = time.time() - t_build

        # Time the ingestion separately so the coverage pass isn't
        # confounded by the gold-answer computation.
        t_idx = time.time()
        ledger = ContextLedger(embed_dim=args.embed_dim, embed_fn=embed_fn,
                                max_artifacts=10_000,
                                max_artifact_chars=corpus.max_chars_per_file)
        indexer = CodeIndexer(
            root=root, max_files=pt,
            max_chars_per_file=corpus.max_chars_per_file)
        indexer.index_into(ledger)
        index_seconds = time.time() - t_idx

        # Run direct-exact. We reuse the already-built corpus to avoid
        # a second build.
        rep_de = run_direct_exact(
            corpus=corpus, llm=llm_factory(), embed_fn=embed_fn,
            embed_dim=args.embed_dim,
            prompt_budget_chars=args.prompt_budget, top_k=args.top_k,
            fetch_chars_per_handle=args.fetch_chars,
            progress=lambda _msg: None)
        agg = rep_de.aggregate()
        n_matched = sum(1 for q in rep_de.questions if q.planner_used)
        n_q = max(1, len(rep_de.questions))
        out.append(ScalingPoint(
            n_files_requested=pt,
            n_files_ingested=corpus.n_files,
            coverage=indexer.stats.as_dict(),
            total_lines=corpus.total_lines,
            build_seconds=round(build_seconds, 3),
            index_seconds=round(index_seconds, 3),
            direct_exact=agg,
            plan_matched_fraction=round(n_matched / n_q, 3),
        ))
    return out


# =============================================================================
# Scoreboards
# =============================================================================


def _print_scoreboard(corpus_reports: list[CorpusReport]) -> None:
    print("\n" + "=" * 96)
    print("PER-CORPUS SCOREBOARD")
    print("=" * 96)
    head = (f"{'corpus':>18} | {'cond':>20} | {'exact':>12} | "
            f"{'no_llm':>12} | {'plan':>6} | {'pmt':>7}")
    print(head)
    print("-" * len(head))
    cond_order = ["lossless-multihop", "lossless-planner", "direct-exact"]
    for cr in corpus_reports:
        for cn in cond_order:
            agg = cr.reports_by_condition.get(cn)
            if not agg:
                continue
            nq = agg["n_questions"]
            ex = f"{agg['exact_correct']}/{nq} ({agg['exact_correct_rate']*100:.1f}%)"
            no_llm = f"{agg['n_no_final_llm']}/{nq}"
            pl = f"{agg['n_planned']}/{nq}"
            print(f"{cr.name:>18} | {cn:>20} | {ex:>12} | "
                  f"{no_llm:>12} | {pl:>6} | {agg['mean_prompt_chars']:>7.0f}")
    print("-" * len(head))


def _aggregate_cross_corpus(corpus_reports: list[CorpusReport]) -> dict:
    """Flat average of exact-correct rates across corpora per condition."""
    out: dict[str, dict] = {}
    conds = set()
    for cr in corpus_reports:
        conds.update(cr.reports_by_condition.keys())
    for cn in sorted(conds):
        ex_rates: list[float] = []
        no_llm_rates: list[float] = []
        n_correct: int = 0
        n_total: int = 0
        prompts: list[float] = []
        for cr in corpus_reports:
            a = cr.reports_by_condition.get(cn)
            if not a:
                continue
            ex_rates.append(a["exact_correct_rate"])
            no_llm_rates.append(a["no_final_llm_rate"])
            n_correct += a["exact_correct"]
            n_total += a["n_questions"]
            prompts.append(a["mean_prompt_chars"])
        if not ex_rates:
            continue
        out[cn] = {
            "mean_exact_rate": round(sum(ex_rates) / len(ex_rates), 4),
            "min_exact_rate": round(min(ex_rates), 4),
            "max_exact_rate": round(max(ex_rates), 4),
            "stddev_exact_rate": (
                round(statistics.stdev(ex_rates), 4) if len(ex_rates) > 1 else 0.0),
            "pooled_exact_rate": round(n_correct / max(1, n_total), 4),
            "mean_no_llm_rate": round(sum(no_llm_rates) / len(no_llm_rates), 4),
            "mean_prompt_chars": round(sum(prompts) / len(prompts), 1),
            "n_corpora": len(ex_rates),
            "total_questions": n_total,
            "total_correct": n_correct,
        }
    return out


# =============================================================================
# LLM adapters (same as Phase 22)
# =============================================================================


def make_mock_llm() -> MockLLM:
    return MockLLM()


def make_ollama_llm(model: str, max_tokens: int):
    from vision_mvp.core.llm_client import LLMClient
    client = LLMClient(model=model)
    def llm(prompt: str) -> str:
        return client.generate(prompt, max_tokens=max_tokens, temperature=0.0)
    llm.client = client
    return llm


def make_ollama_embed(model: str, dim: int):
    from vision_mvp.core.llm_client import LLMClient
    client = LLMClient(model=model)
    def embed(text: str) -> np.ndarray:
        v = np.asarray(client.embed(text), dtype=float)
        if v.size >= dim:
            return v[:dim]
        out = np.zeros(dim, dtype=float)
        out[:v.size] = v
        return out
    embed.client = client
    return embed


# =============================================================================
# CLI
# =============================================================================


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["mock", "ollama"], default="mock")
    ap.add_argument("--extra-roots", nargs="*", default=[],
                    help="Additional local directories to treat as corpora")
    ap.add_argument("--only", nargs="*", default=[],
                    help="Run only these corpus names (by spec .name)")
    ap.add_argument("--max-files", type=int, default=None,
                    help="Optional cap applied UNIFORMLY to every corpus. "
                         "By default each corpus uses its spec's cap (None = all files).")
    ap.add_argument("--max-chars-per-file", type=int, default=64_000)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--model", default="qwen2.5:0.5b")
    ap.add_argument("--embed-dim", type=int, default=64)
    ap.add_argument("--prompt-budget", type=int, default=4000)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--fetch-chars", type=int, default=600)
    ap.add_argument("--skip-conditions", nargs="*",
                    default=[],
                    help="Conditions to skip. Subset of "
                         "{lossless-multihop, lossless-planner}; "
                         "direct-exact is always run as the Phase-23 anchor.")
    ap.add_argument("--scaling", action="store_true",
                    help="Also run a scaling sweep on vision_mvp/core")
    ap.add_argument("--scaling-root", default="vision_mvp/core")
    ap.add_argument("--scaling-points", nargs="*", type=int,
                    default=[10, 20, 40, 80, 108])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print(f"Phase-23 multi-corpus benchmark — mode={args.mode}", flush=True)

    # ---- LLM + embed adapters
    if args.mode == "mock":
        embed_fn = lambda t: hash_embedding(t, dim=args.embed_dim)
        llm_factory = make_mock_llm
    else:
        embed_fn = make_ollama_embed(args.model, dim=args.embed_dim)
        ollama_llm = make_ollama_llm(args.model, max_tokens=200)
        llm_factory = lambda: ollama_llm

    # ---- Build the registry
    reg = default_phase23_registry(extra_roots=args.extra_roots or None)
    if args.max_files is not None:
        # Re-spec every corpus with a uniform cap. This is how we run a
        # "fair" scaling comparison where every corpus contributes the
        # same number of files.
        new_specs = []
        for s in reg.specs:
            new_specs.append(CorpusSpec(
                name=s.name, root=s.root, family=s.family,
                max_files=args.max_files,
                max_chars_per_file=args.max_chars_per_file,
                seed=args.seed,
            ))
        reg.specs = new_specs

    only = set(args.only) if args.only else None
    entries = reg.build(only=only)
    if not entries:
        print("ERROR: no corpora built. Check registry roots.", flush=True)
        return 2

    print("\nCorpora to benchmark:")
    for e in entries:
        s = e.summary()
        print(f"  - {s['name']:>18} | {s['family']:>20} | "
              f"files={s['n_files']:>4} lines={s['total_lines']:>6} "
              f"fns={s['n_functions_total']:>4} cls={s['n_classes_total']:>4} "
              f"Q={s['n_questions']:>2} parse_cov="
              f"{s['coverage']['parse_coverage']*100:.0f}%")

    # ---- Run per-corpus benchmark
    reports: list[CorpusReport] = []
    for e in entries:
        cr = run_corpus(e, args, llm_factory, embed_fn)
        reports.append(cr)

    # ---- Scoreboard + cross-corpus aggregate
    _print_scoreboard(reports)
    cross = _aggregate_cross_corpus(reports)
    print("\n" + "=" * 96)
    print("CROSS-CORPUS AGGREGATE")
    print("=" * 96)
    for cn, s in cross.items():
        print(f"  {cn:>20}: mean_exact={s['mean_exact_rate']*100:.1f}% "
              f"(min={s['min_exact_rate']*100:.1f}% max={s['max_exact_rate']*100:.1f}% "
              f"σ={s['stddev_exact_rate']*100:.1f}) "
              f"pooled={s['pooled_exact_rate']*100:.1f}% "
              f"(n={s['total_correct']}/{s['total_questions']}) "
              f"no_llm={s['mean_no_llm_rate']*100:.1f}% "
              f"pmt={s['mean_prompt_chars']:.0f}")

    # ---- Scaling sweep
    scaling_points: list[ScalingPoint] = []
    if args.scaling:
        # Only run scaling if the root is actually on disk.
        if os.path.isdir(args.scaling_root):
            scaling_points = run_scaling_sweep(
                args.scaling_root, args, llm_factory, embed_fn,
                points=tuple(args.scaling_points))
            print("\n" + "=" * 96)
            print(f"SCALING SWEEP on {args.scaling_root}")
            print("=" * 96)
            head = (f"{'n_files':>8} | {'lines':>7} | {'build_s':>7} | "
                    f"{'index_s':>7} | {'parse_cov':>10} | "
                    f"{'direct_exact':>20} | {'plan_frac':>10}")
            print(head)
            print("-" * len(head))
            for s in scaling_points:
                ex = s.direct_exact
                exact_txt = (f"{ex['exact_correct']}/{ex['n_questions']} "
                             f"({ex['exact_correct_rate']*100:.0f}%)")
                print(f"{s.n_files_ingested:>8} | "
                      f"{s.total_lines:>7} | "
                      f"{s.build_seconds:>7.2f} | "
                      f"{s.index_seconds:>7.2f} | "
                      f"{s.coverage['parse_coverage']*100:>9.1f}% | "
                      f"{exact_txt:>20} | "
                      f"{s.plan_matched_fraction:>10.2f}")
        else:
            print(f"\n[scaling] skip — {args.scaling_root!r} not a directory")

    # ---- Persist
    if args.out:
        payload = {
            "config": {k: (v if not callable(v) else repr(v))
                       for k, v in vars(args).items()},
            "corpora_summary": [e.summary() for e in entries],
            "per_corpus": [cr.as_dict() for cr in reports],
            "cross_corpus_aggregate": cross,
            "scaling_points": [asdict(p) for p in scaling_points],
        }
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"\nWrote {args.out}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
