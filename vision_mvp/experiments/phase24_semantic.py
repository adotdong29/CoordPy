"""Phase 24 — Conservative semantic-code benchmark across multiple corpora.

Reuses the Phase-23 multi-corpus ingestion and the Phase-22 direct-exact
/ wrap-planner / lossless-multihop harness, but restricts the question
battery to the **Phase-24 conservative-semantic** slice:

  * count_may_raise / list_may_raise
  * count_is_recursive / list_is_recursive
  * count_may_write_global / list_may_write_global
  * count_calls_subprocess / list_calls_subprocess
  * count_calls_filesystem / list_calls_filesystem
  * count_calls_network / list_calls_network
  * count_calls_external_io

Three conditions per corpus:

  1. `lossless-multihop`   — Phase-20 retrieval-mediated baseline.
  2. `lossless-planner`    — Phase-21 wrap-LLM planner (operator
                             pipeline computes the answer, mock/real
                             LLM re-states it).
  3. `direct-exact`        — Phase-22 direct render — zero LLM on
                             planner-matched questions.

The headline metric is the per-corpus *semantic-only* direct-exact
rate and the gap to retrieval. On corpora where the analyzer flags
non-zero functions for every predicate, direct-exact SHOULD score
100 % (the gold comes from the same analyzer), while retrieval should
scatter based on whether the top-k file bodies happen to contain the
right token by coincidence.

Reproduce:
    python -m vision_mvp.experiments.phase24_semantic \\
        --mode mock --out vision_mvp/results_phase24_mock.json
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
    CondReport, QResult, _classify,
    run_direct_exact, run_lossless_retrieval, run_planner_with_wrap,
)
from vision_mvp.tasks.corpus_registry import (
    CorpusEntry, CorpusSpec, default_phase23_registry,
)
from vision_mvp.tasks.needle_corpus import NeedleQuestion
from vision_mvp.tasks.python_corpus import PythonCorpus


# =============================================================================
# Semantic-question filter
# =============================================================================


_SEMANTIC_KINDS: frozenset[str] = frozenset({
    "count_may_raise", "list_may_raise",
    "count_is_recursive", "list_is_recursive",
    "count_may_write_global", "list_may_write_global",
    "count_calls_subprocess", "list_calls_subprocess",
    "count_calls_filesystem", "list_calls_filesystem",
    "count_calls_network", "list_calls_network",
    "count_calls_external_io",
})


def _restrict_to_semantic(corpus: PythonCorpus) -> PythonCorpus:
    """Return `corpus` with its `.questions` filtered to the Phase-24
    semantic kinds only. This is a side-effecting convenience — we
    rebuild `corpus.questions` in place so the downstream runners
    (which iterate over `corpus.questions`) see only the Phase-24
    slice."""
    semantic = [q for q in corpus.questions if q.kind in _SEMANTIC_KINDS]
    corpus.questions = semantic
    return corpus


# =============================================================================
# Per-corpus driver — reuses Phase-23 shape but with semantic-only Q
# =============================================================================


@dataclass
class CorpusReport:
    name: str
    family: str
    spec: dict
    coverage: dict
    semantic_totals: dict = field(default_factory=dict)
    reports_by_condition: dict[str, dict] = field(default_factory=dict)
    questions_by_condition: dict[str, list[dict]] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


def _semantic_totals(corpus: PythonCorpus) -> dict:
    """Per-corpus gold counts — useful context for interpreting results."""
    return {
        "n_functions_may_raise": corpus.n_functions_may_raise,
        "n_functions_is_recursive": corpus.n_functions_is_recursive,
        "n_functions_may_write_global": corpus.n_functions_may_write_global,
        "n_functions_calls_subprocess": corpus.n_functions_calls_subprocess,
        "n_functions_calls_filesystem": corpus.n_functions_calls_filesystem,
        "n_functions_calls_network": corpus.n_functions_calls_network,
        "n_functions_calls_external_io": corpus.n_functions_calls_external_io,
    }


def run_corpus(
    entry: CorpusEntry, args, llm_factory, embed_fn,
    progress: Callable[[str], None] = print,
) -> CorpusReport:
    corpus = _restrict_to_semantic(entry.corpus)
    cr = CorpusReport(
        name=entry.name, family=entry.spec.family,
        spec={"root": entry.spec.root, "max_files": entry.spec.max_files,
              "max_chars_per_file": entry.spec.max_chars_per_file,
              "seed": entry.spec.seed},
        coverage=entry.coverage.as_dict(),
        semantic_totals=_semantic_totals(corpus),
    )
    if not corpus.questions:
        # The corpus has no semantic-flagged functions — nothing to ask.
        # Record empty reports and move on.
        progress(f"[{entry.name}] no semantic questions emitted "
                 f"(all predicates flagged 0) — skipping conditions")
        return cr

    progress(f"\n-- [{entry.name}] DIRECT-EXACT")
    rep_de = run_direct_exact(
        corpus=corpus, llm=llm_factory(), embed_fn=embed_fn,
        embed_dim=args.embed_dim,
        prompt_budget_chars=args.prompt_budget, top_k=args.top_k,
        fetch_chars_per_handle=args.fetch_chars, progress=progress)
    cr.reports_by_condition["direct-exact"] = rep_de.aggregate()
    cr.questions_by_condition["direct-exact"] = [
        asdict(q) for q in rep_de.questions]

    if "lossless-planner" not in args.skip_conditions:
        progress(f"-- [{entry.name}] LOSSLESS-PLANNER (wrap LLM)")
        rep_lp = run_planner_with_wrap(
            corpus=corpus, llm=llm_factory(), embed_fn=embed_fn,
            embed_dim=args.embed_dim,
            prompt_budget_chars=args.prompt_budget, top_k=args.top_k,
            fetch_chars_per_handle=args.fetch_chars, progress=progress)
        cr.reports_by_condition["lossless-planner"] = rep_lp.aggregate()
        cr.questions_by_condition["lossless-planner"] = [
            asdict(q) for q in rep_lp.questions]

    if "lossless-multihop" not in args.skip_conditions:
        progress(f"-- [{entry.name}] LOSSLESS-MULTIHOP")
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
# Scoreboards
# =============================================================================


def _print_scoreboard(corpus_reports: list[CorpusReport]) -> None:
    print("\n" + "=" * 112)
    print("PER-CORPUS PHASE-24 SEMANTIC SCOREBOARD")
    print("=" * 112)
    head = (f"{'corpus':>18} | {'cond':>20} | "
            f"{'exact':>14} | {'no_llm':>12} | {'plan':>7} | {'pmt':>7}")
    print(head)
    print("-" * len(head))
    cond_order = ["lossless-multihop", "lossless-planner", "direct-exact"]
    for cr in corpus_reports:
        if not cr.reports_by_condition:
            print(f"{cr.name:>18} | (no semantic questions emitted)")
            continue
        for cn in cond_order:
            agg = cr.reports_by_condition.get(cn)
            if not agg:
                continue
            nq = agg["n_questions"]
            ex = (f"{agg['exact_correct']}/{nq} "
                  f"({agg['exact_correct_rate']*100:.1f}%)")
            no_llm = f"{agg['n_no_final_llm']}/{nq}"
            pl = f"{agg['n_planned']}/{nq}"
            print(f"{cr.name:>18} | {cn:>20} | {ex:>14} | "
                  f"{no_llm:>12} | {pl:>7} | {agg['mean_prompt_chars']:>7.0f}")
    print("-" * len(head))


def _aggregate_cross_corpus(corpus_reports: list[CorpusReport]) -> dict:
    """Cross-corpus summary of exact-correct rates per condition."""
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
                round(statistics.stdev(ex_rates), 4) if len(ex_rates) > 1
                else 0.0),
            "pooled_exact_rate": round(n_correct / max(1, n_total), 4),
            "mean_no_llm_rate": round(
                sum(no_llm_rates) / len(no_llm_rates), 4),
            "mean_prompt_chars": round(sum(prompts) / len(prompts), 1),
            "n_corpora": len(ex_rates),
            "total_questions": n_total,
            "total_correct": n_correct,
        }
    return out


def _per_predicate_breakdown(corpus_reports: list[CorpusReport]) -> dict:
    """Flatten per-question results across corpora and split by
    predicate (`may_raise`, `is_recursive`, …). Each predicate gets
    {ok, n} per condition."""
    out: dict[str, dict[str, dict[str, int]]] = {}
    for cr in corpus_reports:
        for cn, qs in cr.questions_by_condition.items():
            for q in qs:
                kind = q["kind"]
                # Strip "count_" / "list_" prefix → predicate name
                if kind.startswith("count_"):
                    pred = kind[len("count_"):]
                elif kind.startswith("list_"):
                    pred = kind[len("list_"):]
                else:
                    pred = kind
                d = out.setdefault(pred, {})
                dc = d.setdefault(cn, {"ok": 0, "n": 0})
                dc["n"] += 1
                if q["exact_correct"]:
                    dc["ok"] += 1
    return out


# =============================================================================
# LLM adapters
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
    ap.add_argument("--extra-roots", nargs="*", default=[])
    ap.add_argument("--only", nargs="*", default=[])
    ap.add_argument("--max-files", type=int, default=None,
                    help="Uniform file-count cap across every corpus.")
    ap.add_argument("--max-chars-per-file", type=int, default=64_000)
    ap.add_argument("--seed", type=int, default=24)
    ap.add_argument("--model", default="qwen2.5:0.5b")
    ap.add_argument("--embed-dim", type=int, default=64)
    ap.add_argument("--prompt-budget", type=int, default=4000)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--fetch-chars", type=int, default=600)
    ap.add_argument("--skip-conditions", nargs="*", default=[])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print(f"Phase-24 semantic benchmark — mode={args.mode}", flush=True)

    if args.mode == "mock":
        embed_fn = lambda t: hash_embedding(t, dim=args.embed_dim)
        llm_factory = make_mock_llm
    else:
        embed_fn = make_ollama_embed(args.model, dim=args.embed_dim)
        ollama_llm = make_ollama_llm(args.model, max_tokens=200)
        llm_factory = lambda: ollama_llm

    reg = default_phase23_registry(extra_roots=args.extra_roots or None)
    if args.max_files is not None:
        # Re-spec every corpus with a uniform cap.
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

    print("\nCorpora to benchmark (Phase-24 semantic-only):")
    for e in entries:
        s = e.summary()
        tot = _semantic_totals(e.corpus)
        print(
            f"  - {s['name']:>18} | {s['family']:>20} | "
            f"files={s['n_files']:>4} fns={s['n_functions_total']:>4} "
            f"may_raise={tot['n_functions_may_raise']:>3} "
            f"rec={tot['n_functions_is_recursive']:>2} "
            f"gw={tot['n_functions_may_write_global']:>2} "
            f"sp={tot['n_functions_calls_subprocess']:>2} "
            f"fs={tot['n_functions_calls_filesystem']:>2} "
            f"net={tot['n_functions_calls_network']:>2} "
            f"io={tot['n_functions_calls_external_io']:>2}"
        )

    reports: list[CorpusReport] = []
    for e in entries:
        reports.append(run_corpus(e, args, llm_factory, embed_fn))

    _print_scoreboard(reports)
    cross = _aggregate_cross_corpus(reports)
    print("\n" + "=" * 112)
    print("PHASE-24 CROSS-CORPUS AGGREGATE (semantic slice only)")
    print("=" * 112)
    for cn, s in cross.items():
        print(f"  {cn:>20}: mean_exact={s['mean_exact_rate']*100:5.1f}% "
              f"(min={s['min_exact_rate']*100:5.1f}% "
              f"max={s['max_exact_rate']*100:5.1f}% "
              f"σ={s['stddev_exact_rate']*100:4.1f}) "
              f"pooled={s['pooled_exact_rate']*100:5.1f}% "
              f"(n={s['total_correct']}/{s['total_questions']}) "
              f"no_llm={s['mean_no_llm_rate']*100:5.1f}% "
              f"pmt={s['mean_prompt_chars']:.0f}")

    per_pred = _per_predicate_breakdown(reports)
    print("\nPER-PREDICATE BREAKDOWN (pooled across corpora)")
    print("-" * 80)
    hdr = f"{'predicate':>20} | {'condition':>20} | {'ok':>5} / {'n':>3} rate"
    print(hdr)
    for pred in sorted(per_pred):
        for cn, counts in sorted(per_pred[pred].items()):
            n = counts["n"]
            ok = counts["ok"]
            rate = (ok / n) if n else 0.0
            print(f"{pred:>20} | {cn:>20} | {ok:>5} / {n:>3}  "
                  f"{rate*100:>5.1f}%")

    if args.out:
        payload = {
            "config": {k: (v if not callable(v) else repr(v))
                       for k, v in vars(args).items()},
            "corpora_summary": [e.summary() for e in entries],
            "per_corpus": [cr.as_dict() for cr in reports],
            "cross_corpus_aggregate": cross,
            "per_predicate_breakdown": per_pred,
        }
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"\nWrote {args.out}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
