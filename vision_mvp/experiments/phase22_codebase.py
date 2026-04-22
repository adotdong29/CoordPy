"""Phase 22 — Codebase-scale exact-memory computation.

Moves the substrate from synthetic incident-review corpora (Phases
19/20/21) to a real Python codebase. The corpus is a directory of
`.py` files; the ground truth is computed deterministically from the
AST-derived metadata; the questions exercise both retrieval-style and
aggregation-style queries over typed code structure.

Six conditions:

  1. `map_reduce`         — text-only baseline; summarise each file,
                            answer from the pooled summaries
  2. `lossless-hybrid`    — Phase-20 dense + lexical retrieval over
                            file source bodies
  3. `lossless-multihop`  — Phase-20 hybrid + cross-reference expansion
                            (rarely fires here; cross-refs in code are
                            mostly imports, not regex IDs)
  4. `lossless-planner`   — Phase-21 planner with **wrap LLM** at the
                            end (operator pipeline computes the answer,
                            LLM re-states it)
  5. **`direct-exact`**   — Phase-22 NEW. When the planner has an
                            answer, return its render output verbatim
                            with provenance — NO LLM call. Falls
                            through to `lossless-multihop` only when
                            the planner has no plan.
  6. `oracle`             — full corpus in the prompt (skipped when
                            corpus exceeds budget)

Per question we record the same metrics as Phase 21 plus a new
`render_used_llm` flag distinguishing the wrap path from the direct
path. Failure decomposition extends to:

  ok / retrieval_miss / planning_error / render_error / llm_error

Reproduce:
    python -m vision_mvp.experiments.phase22_codebase \\
        --mode mock --root vision_mvp/core --max-files 24

    python -m vision_mvp.experiments.phase22_codebase \\
        --mode ollama --model qwen2.5:0.5b \\
        --root vision_mvp/core --max-files 16
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
from vision_mvp.experiments.phase19_lossless import (
    MockLLM, run_map_reduce, run_oracle,
)
from vision_mvp.tasks.needle_corpus import NeedleCorpus
from vision_mvp.tasks.python_corpus import PythonCorpus


# =============================================================================
# Result types
# =============================================================================


@dataclass
class QResult:
    question: str
    gold: str
    kind: str
    answer: str
    exact_correct: bool
    fact_in_input: bool
    failure_class: str       # ok / retrieval_miss / planning_error / render_error / llm_error
    prompt_chars: int
    fetch_count: int = 0
    fetched_bytes: int = 0
    cited_cids: list[str] = field(default_factory=list)
    n_hops: int = 1
    planner_pattern: str = ""
    planner_used: bool = False
    render_used_llm: bool = False
    extra: dict = field(default_factory=dict)


@dataclass
class CondReport:
    name: str
    questions: list[QResult] = field(default_factory=list)
    setup_seconds: float = 0.0
    answer_seconds: float = 0.0
    n_llm_setup_calls: int = 0
    n_llm_answer_calls: int = 0
    extras: dict = field(default_factory=dict)

    def aggregate(self) -> dict:
        n = len(self.questions)
        if n == 0:
            return {}
        n_exact = sum(1 for q in self.questions if q.exact_correct)
        n_in = sum(1 for q in self.questions if q.fact_in_input)
        prompts = [q.prompt_chars for q in self.questions]
        fc = {"ok": 0, "retrieval_miss": 0, "planning_error": 0,
              "render_error": 0, "llm_error": 0}
        for q in self.questions:
            fc[q.failure_class] = fc.get(q.failure_class, 0) + 1
        per_kind: dict[str, dict] = {}
        for q in self.questions:
            d = per_kind.setdefault(q.kind, {"n": 0, "exact": 0})
            d["n"] += 1
            d["exact"] += int(q.exact_correct)
        n_planner = sum(1 for q in self.questions if q.planner_used)
        n_no_llm = sum(1 for q in self.questions if not q.render_used_llm)
        return {
            "name": self.name,
            "n_questions": n,
            "exact_correct": n_exact,
            "exact_correct_rate": round(n_exact / n, 4),
            "fact_in_input": n_in,
            "fact_in_input_rate": round(n_in / n, 4),
            "mean_prompt_chars": round(sum(prompts) / n, 1),
            "max_prompt_chars": max(prompts),
            "mean_fetch_count": round(
                sum(q.fetch_count for q in self.questions) / n, 2),
            "failure_classes": fc,
            "per_kind_exact": per_kind,
            "n_planned": n_planner,
            "n_no_final_llm": n_no_llm,
            "no_final_llm_rate": round(n_no_llm / n, 4),
            "setup_seconds": round(self.setup_seconds, 2),
            "answer_seconds": round(self.answer_seconds, 2),
            "n_llm_setup_calls": self.n_llm_setup_calls,
            "n_llm_answer_calls": self.n_llm_answer_calls,
            "extras": self.extras,
        }


def _classify(*, exact: bool, fact_in: bool, planner_used: bool,
              render_used_llm: bool) -> str:
    if exact:
        return "ok"
    if planner_used and not render_used_llm:
        # Direct render produced a wrong answer — this is a planning_error
        # (the operator pipeline itself returned the wrong value). Should
        # be ≈ 0 if the planner's pattern matches the question correctly.
        return "planning_error"
    if planner_used and render_used_llm:
        # Operator chain returned an answer; LLM wrap then mangled it.
        return "render_error"
    if fact_in:
        return "llm_error"
    return "retrieval_miss"


# =============================================================================
# Adapters: map-reduce and oracle (text-body conditions over file sources)
# =============================================================================


def _adapt_phase19(name: str, rep19) -> CondReport:
    out = CondReport(
        name=name, setup_seconds=rep19.setup_seconds,
        answer_seconds=rep19.answer_seconds,
        n_llm_setup_calls=rep19.n_llm_setup_calls,
        n_llm_answer_calls=rep19.n_llm_answer_calls,
        extras=dict(rep19.extras))
    for q in rep19.questions:
        fc = _classify(exact=q.exact_correct, fact_in=q.fact_in_input,
                       planner_used=False, render_used_llm=True)
        out.questions.append(QResult(
            question=q.question, gold=q.gold, kind=q.kind, answer=q.answer,
            exact_correct=q.exact_correct, fact_in_input=q.fact_in_input,
            failure_class=fc, prompt_chars=q.prompt_chars,
            planner_pattern="(unmatched)", planner_used=False,
            render_used_llm=True,
        ))
    return out


# =============================================================================
# lossless-hybrid / lossless-multihop adapter (over code bodies)
# =============================================================================


def run_lossless_retrieval(
    *, name: str, search_mode: str, max_hops: int,
    corpus: PythonCorpus, llm, embed_fn, embed_dim: int,
    prompt_budget_chars: int, top_k: int, fetch_chars_per_handle: int,
    progress=print,
) -> CondReport:
    rep = CondReport(name=name)
    t0 = time.time()
    ledger = ContextLedger(embed_dim=embed_dim, embed_fn=embed_fn,
                            max_artifacts=10_000,
                            max_artifact_chars=corpus.max_chars_per_file)
    indexer = CodeIndexer(root=corpus.root,
                           max_files=corpus.max_files,
                           max_chars_per_file=corpus.max_chars_per_file)
    indexer.index_into(ledger, progress_cb=progress)
    rep.setup_seconds = time.time() - t0
    rep.extras["n_embed_calls"] = len(ledger)
    rep.extras["search_mode"] = search_mode
    rep.extras["max_hops"] = max_hops

    worker = BoundedRetrievalWorker(
        ledger=ledger, llm_call=llm,
        prompt_budget_chars=prompt_budget_chars,
        top_k=top_k, fetch_chars_per_handle=fetch_chars_per_handle,
        search_mode=search_mode, max_hops=max_hops,
    )

    t0 = time.time()
    if isinstance(llm, MockLLM):
        llm.mode = "extract"
    for q in corpus.questions:
        if isinstance(llm, MockLLM):
            llm.current_gold = q.gold
        r = worker.answer(q.question)
        rep.n_llm_answer_calls += 1
        fact_in = q.gold.lower() in r.prompt.lower()
        exact = NeedleCorpus.score_exact(r.answer, q)
        fc = _classify(exact=exact, fact_in=fact_in,
                       planner_used=False, render_used_llm=True)
        rep.questions.append(QResult(
            question=q.question, gold=q.gold, kind=q.kind,
            answer=r.answer, exact_correct=exact, fact_in_input=fact_in,
            failure_class=fc, prompt_chars=r.prompt_chars,
            fetch_count=r.fetch_count, fetched_bytes=r.fetched_bytes,
            cited_cids=r.cited_cids, n_hops=len(r.hops),
            planner_pattern="(unmatched)", planner_used=False,
            render_used_llm=True,
        ))
    rep.answer_seconds = time.time() - t0
    return rep


# =============================================================================
# lossless-planner (wrap LLM) — Phase 21 path on code corpus
# =============================================================================


_WRAP_PROMPT = (
    "You are answering a question. The system has already computed the "
    "EXACT answer using deterministic operators over typed code "
    "metadata. Quote the answer verbatim.\n\n"
    "Question: {q}\n"
    "Computed answer: {a}\n\n"
    "Final answer:")


def run_planner_with_wrap(
    *, corpus: PythonCorpus, llm, embed_fn, embed_dim: int,
    prompt_budget_chars: int, top_k: int, fetch_chars_per_handle: int,
    progress=print,
) -> CondReport:
    rep = CondReport(name="lossless-planner")
    t0 = time.time()
    ledger = ContextLedger(embed_dim=embed_dim, embed_fn=embed_fn,
                            max_artifacts=10_000,
                            max_artifact_chars=corpus.max_chars_per_file)
    CodeIndexer(root=corpus.root, max_files=corpus.max_files,
                 max_chars_per_file=corpus.max_chars_per_file
                 ).index_into(ledger, progress_cb=progress)
    rep.setup_seconds = time.time() - t0

    planner = CodeQueryPlanner()
    fallback = BoundedRetrievalWorker(
        ledger=ledger, llm_call=llm,
        prompt_budget_chars=prompt_budget_chars, top_k=top_k,
        fetch_chars_per_handle=fetch_chars_per_handle,
        search_mode="hybrid", max_hops=3,
    )

    t0 = time.time()
    if isinstance(llm, MockLLM):
        llm.mode = "extract"
    for q in corpus.questions:
        plan_res = planner.plan(q.question)
        if plan_res.plan is not None:
            stage, _trace = plan_res.plan.execute(ledger)
            computed = plan_res.plan.render(stage)
            wrap = _WRAP_PROMPT.format(q=q.question, a=computed)
            if isinstance(llm, MockLLM):
                llm.current_gold = q.gold
            try:
                wrapped = llm(wrap)
            except Exception:
                wrapped = computed
            rep.n_llm_answer_calls += 1
            full = wrapped + "\n[planner: " + computed + "]"
            exact = NeedleCorpus.score_exact(full, q)
            fact_in = q.gold.lower() in (computed + " " + wrap).lower()
            fc = _classify(exact=exact, fact_in=fact_in,
                           planner_used=True, render_used_llm=True)
            rep.questions.append(QResult(
                question=q.question, gold=q.gold, kind=q.kind,
                answer=full, exact_correct=exact, fact_in_input=fact_in,
                failure_class=fc, prompt_chars=len(wrap),
                planner_pattern=plan_res.pattern, planner_used=True,
                render_used_llm=True,
                extra={"computed": computed},
            ))
        else:
            if isinstance(llm, MockLLM):
                llm.current_gold = q.gold
            r = fallback.answer(q.question)
            rep.n_llm_answer_calls += 1
            fact_in = q.gold.lower() in r.prompt.lower()
            exact = NeedleCorpus.score_exact(r.answer, q)
            fc = _classify(exact=exact, fact_in=fact_in,
                           planner_used=False, render_used_llm=True)
            rep.questions.append(QResult(
                question=q.question, gold=q.gold, kind=q.kind,
                answer=r.answer, exact_correct=exact, fact_in_input=fact_in,
                failure_class=fc, prompt_chars=r.prompt_chars,
                fetch_count=r.fetch_count, fetched_bytes=r.fetched_bytes,
                cited_cids=r.cited_cids,
                planner_pattern="(unmatched)", planner_used=False,
                render_used_llm=True,
            ))
    rep.answer_seconds = time.time() - t0
    return rep


# =============================================================================
# direct-exact — Phase 22 NEW
# =============================================================================


def run_direct_exact(
    *, corpus: PythonCorpus, llm, embed_fn, embed_dim: int,
    prompt_budget_chars: int, top_k: int, fetch_chars_per_handle: int,
    progress=print,
) -> CondReport:
    """When the planner has a plan, return its render output VERBATIM.

    No LLM call. Provenance preserved as `cited_cids` (every cid that
    contributed to the operator pipeline). Falls through to the
    multihop hybrid worker only when the planner can't match."""
    rep = CondReport(name="direct-exact")
    t0 = time.time()
    ledger = ContextLedger(embed_dim=embed_dim, embed_fn=embed_fn,
                            max_artifacts=10_000,
                            max_artifact_chars=corpus.max_chars_per_file)
    CodeIndexer(root=corpus.root, max_files=corpus.max_files,
                 max_chars_per_file=corpus.max_chars_per_file
                 ).index_into(ledger, progress_cb=progress)
    rep.setup_seconds = time.time() - t0

    planner = CodeQueryPlanner()
    fallback = BoundedRetrievalWorker(
        ledger=ledger, llm_call=llm,
        prompt_budget_chars=prompt_budget_chars, top_k=top_k,
        fetch_chars_per_handle=fetch_chars_per_handle,
        search_mode="hybrid", max_hops=3,
    )

    t0 = time.time()
    if isinstance(llm, MockLLM):
        llm.mode = "extract"
    for q in corpus.questions:
        plan_res = planner.plan(q.question)
        if plan_res.plan is not None:
            stage, op_trace = plan_res.plan.execute(ledger)
            computed = plan_res.plan.render(stage)
            # Provenance: cids that the operators TOUCHED + cids of
            # contributing handles in the final stage (when applicable).
            cids_touched: set[str] = set()
            for t in op_trace:
                cids_touched.update(t.cids_touched)
            from vision_mvp.core.exact_ops import (
                StageGroups, StageList, StageScalar, StageValues,
            )
            if isinstance(stage, (StageScalar, StageGroups, StageList)):
                cids_touched.update(getattr(stage, "contributing_cids", []))
            elif isinstance(stage, StageValues):
                cids_touched.update(h.cid for h, _v in stage.pairs)
            answer = computed
            exact = NeedleCorpus.score_exact(answer, q)
            fact_in = q.gold.lower() in answer.lower()
            fc = _classify(exact=exact, fact_in=fact_in,
                           planner_used=True, render_used_llm=False)
            rep.questions.append(QResult(
                question=q.question, gold=q.gold, kind=q.kind,
                answer=answer, exact_correct=exact, fact_in_input=fact_in,
                failure_class=fc, prompt_chars=0,
                cited_cids=sorted(cids_touched),
                planner_pattern=plan_res.pattern, planner_used=True,
                render_used_llm=False,
                extra={"computed": computed,
                       "n_cids_touched": len(cids_touched)},
            ))
        else:
            # Fall through.
            if isinstance(llm, MockLLM):
                llm.current_gold = q.gold
            r = fallback.answer(q.question)
            rep.n_llm_answer_calls += 1
            fact_in = q.gold.lower() in r.prompt.lower()
            exact = NeedleCorpus.score_exact(r.answer, q)
            fc = _classify(exact=exact, fact_in=fact_in,
                           planner_used=False, render_used_llm=True)
            rep.questions.append(QResult(
                question=q.question, gold=q.gold, kind=q.kind,
                answer=r.answer, exact_correct=exact, fact_in_input=fact_in,
                failure_class=fc, prompt_chars=r.prompt_chars,
                fetch_count=r.fetch_count, fetched_bytes=r.fetched_bytes,
                cited_cids=r.cited_cids,
                planner_pattern="(unmatched)", planner_used=False,
                render_used_llm=True,
            ))
    rep.answer_seconds = time.time() - t0
    return rep


# =============================================================================
# Map-reduce / oracle adaptation (over file source bodies)
# =============================================================================


def _make_text_corpus_from(corpus: PythonCorpus):
    """Wrap the PythonCorpus so the existing Phase-19 map_reduce / oracle
    runners (which expect a NeedleCorpus interface) work over code."""
    # Create a thin shim that mimics the small subset of NeedleCorpus
    # the Phase-19 runners use: `.sections`, `.questions`, `.document`.
    class _Shim:
        sections = corpus.chunks()
        questions = corpus.questions
        document = "\n\n=== FILE BREAK ===\n\n".join(
            corpus.chunks())
    return _Shim()


# =============================================================================
# One pass + repeats
# =============================================================================


def run_one_pass(args, corpus: PythonCorpus, llm_factory, embed_fn,
                 progress=print) -> dict[str, CondReport]:
    out: dict[str, CondReport] = {}
    text_corpus = _make_text_corpus_from(corpus)

    if not args.skip_mr:
        progress("\n" + "=" * 78)
        progress("CONDITION: map_reduce baseline (over file bodies)")
        progress("=" * 78)
        rep_mr = run_map_reduce(
            text_corpus, llm_factory(),
            summary_prompt_chars_max=args.summary_prompt_cap,
            answer_prompt_chars_max=args.prompt_budget,
            progress=progress)
        out["map_reduce"] = _adapt_phase19("map_reduce", rep_mr)

    for name, search_mode, max_hops in [
        ("lossless-hybrid", "hybrid", 1),
        ("lossless-multihop", "hybrid", 3),
    ]:
        if name in args.skip_conditions:
            continue
        progress("\n" + "=" * 78)
        progress(f"CONDITION: {name}")
        progress("=" * 78)
        out[name] = run_lossless_retrieval(
            name=name, search_mode=search_mode, max_hops=max_hops,
            corpus=corpus, llm=llm_factory(), embed_fn=embed_fn,
            embed_dim=args.embed_dim,
            prompt_budget_chars=args.prompt_budget, top_k=args.top_k,
            fetch_chars_per_handle=args.fetch_chars, progress=progress)

    if "lossless-planner" not in args.skip_conditions:
        progress("\n" + "=" * 78)
        progress("CONDITION: lossless-planner (wrap LLM)")
        progress("=" * 78)
        out["lossless-planner"] = run_planner_with_wrap(
            corpus=corpus, llm=llm_factory(), embed_fn=embed_fn,
            embed_dim=args.embed_dim,
            prompt_budget_chars=args.prompt_budget, top_k=args.top_k,
            fetch_chars_per_handle=args.fetch_chars, progress=progress)

    if "direct-exact" not in args.skip_conditions:
        progress("\n" + "=" * 78)
        progress("CONDITION: direct-exact (Phase 22 NEW — no wrap LLM)")
        progress("=" * 78)
        out["direct-exact"] = run_direct_exact(
            corpus=corpus, llm=llm_factory(), embed_fn=embed_fn,
            embed_dim=args.embed_dim,
            prompt_budget_chars=args.prompt_budget, top_k=args.top_k,
            fetch_chars_per_handle=args.fetch_chars, progress=progress)

    if not args.skip_oracle:
        progress("\n" + "=" * 78)
        progress("CONDITION: oracle (full corpus in prompt)")
        progress("=" * 78)
        rep_o = run_oracle(text_corpus, llm_factory(),
                            prompt_budget_chars=args.oracle_budget)
        out["oracle"] = _adapt_phase19("oracle", rep_o)
    return out


def _stats(vals: list[float]) -> dict:
    if not vals:
        return {}
    d = {"mean": round(statistics.mean(vals), 4),
         "min": round(min(vals), 4),
         "max": round(max(vals), 4)}
    if len(vals) > 1:
        d["stddev"] = round(statistics.stdev(vals), 4)
    return d


def aggregate_repeats(per_rep_aggs: list[dict]) -> dict:
    if not per_rep_aggs:
        return {}
    cond_names = list(per_rep_aggs[0].keys())
    out: dict = {"n_repeats": len(per_rep_aggs), "by_condition": {}}
    for cn in cond_names:
        ex = [r[cn]["exact_correct_rate"] for r in per_rep_aggs
              if cn in r and r[cn]]
        no_llm = [r[cn]["no_final_llm_rate"] for r in per_rep_aggs
                   if cn in r and r[cn]]
        if not ex:
            continue
        out["by_condition"][cn] = {
            "exact_correct_rate": _stats(ex),
            "no_final_llm_rate": _stats(no_llm),
        }
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
    ap.add_argument("--root", default="vision_mvp/core",
                    help="root directory of the Python corpus")
    ap.add_argument("--max-files", type=int, default=24)
    ap.add_argument("--max-chars-per-file", type=int, default=64_000)
    ap.add_argument("--seed", type=int, default=22)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--model", default="qwen2.5:0.5b")
    ap.add_argument("--embed-dim", type=int, default=64)
    ap.add_argument("--prompt-budget", type=int, default=4000)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--fetch-chars", type=int, default=600)
    ap.add_argument("--summary-prompt-cap", type=int, default=4000)
    ap.add_argument("--oracle-budget", type=int, default=128_000)
    ap.add_argument("--skip-oracle", action="store_true")
    ap.add_argument("--skip-mr", action="store_true")
    ap.add_argument("--skip-conditions", nargs="*", default=[])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print(f"Phase-22 codebase benchmark — mode={args.mode} "
          f"root={args.root} max_files={args.max_files}", flush=True)
    if args.mode == "mock":
        embed_fn = lambda t: hash_embedding(t, dim=args.embed_dim)
        llm_factory = make_mock_llm
    else:
        embed_fn = make_ollama_embed(args.model, dim=args.embed_dim)
        ollama_llm = make_ollama_llm(args.model, max_tokens=200)
        llm_factory = lambda: ollama_llm

    per_rep_aggs: list[dict] = []
    per_rep_full: list[dict] = []

    for rep_i in range(args.repeats):
        rep_seed = args.seed + rep_i
        print(f"\n# REPEAT {rep_i+1} / {args.repeats}  (seed={rep_seed})",
              flush=True)
        corpus = PythonCorpus(root=args.root, max_files=args.max_files,
                               max_chars_per_file=args.max_chars_per_file,
                               seed=rep_seed)
        corpus.build()
        print(f"Corpus: {corpus.n_files} files, "
              f"{corpus.total_lines} lines, "
              f"{corpus.n_functions_total} functions, "
              f"{corpus.n_classes_total} classes, "
              f"{corpus.n_distinct_imports} distinct imports → "
              f"{len(corpus.questions)} questions.",
              flush=True)

        reports = run_one_pass(args, corpus, llm_factory, embed_fn)
        agg_by_cond = {name: r.aggregate() for name, r in reports.items()}
        per_rep_aggs.append(agg_by_cond)
        per_rep_full.append({
            "rep": rep_i, "seed": rep_seed,
            "aggregate": agg_by_cond,
            "questions_per_condition": {
                name: [asdict(q) for q in r.questions]
                for name, r in reports.items()
            },
        })

        # Per-rep scoreboard
        print("\n" + "-" * 86)
        print(f"REPEAT {rep_i+1} OVERALL SCOREBOARD")
        print("-" * 86)
        head = (f"{'condition':>22} | {'exact':>11} | {'fact_in':>11} | "
                f"{'no_llm':>7} | {'plan':>5} | {'pmt':>6}")
        print(head)
        print("-" * len(head))
        for name in ("map_reduce", "lossless-hybrid", "lossless-multihop",
                     "lossless-planner", "direct-exact", "oracle"):
            if name not in agg_by_cond:
                continue
            a = agg_by_cond[name]
            if not a:
                continue
            print(f"{name:>22} | "
                  f"{a['exact_correct']}/{a['n_questions']:>3} ({a['exact_correct_rate']*100:>4.1f}%) | "
                  f"{a['fact_in_input']}/{a['n_questions']:>3} ({a['fact_in_input_rate']*100:>4.1f}%) | "
                  f"{a['n_no_final_llm']}/{a['n_questions']:>3} | "
                  f"{a['n_planned']:>2}/{a['n_questions']:>2} | "
                  f"{a['mean_prompt_chars']:>6.0f}")

    cross = aggregate_repeats(per_rep_aggs)
    if args.repeats > 1:
        print("\n" + "=" * 86)
        print(f"AGGREGATE OVER {args.repeats} REPEATS")
        print("=" * 86)
        for cn, stats in cross["by_condition"].items():
            ex = stats["exact_correct_rate"]
            no_llm = stats["no_final_llm_rate"]
            print(f"  {cn:>22}: exact mean={ex['mean']*100:.1f}% "
                  f"(σ={ex.get('stddev', 0)*100:.1f})  "
                  f"no-LLM rate mean={no_llm['mean']*100:.1f}%")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({
                "config": vars(args),
                "per_repeat": per_rep_full,
                "cross_repeat_aggregate": cross,
            }, f, indent=2, default=str)
        print(f"\nWrote {args.out}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
