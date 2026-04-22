"""Phase 21 — Exact computation over external memory.

Extends `phase20_substrate` with one new condition that uses the Phase-21
query planner + exact operator pipeline to answer aggregation /
composition queries that the bounded retrieval worker cannot answer
well.

Conditions compared (in declaration order):

  1. `map_reduce`             — Phase-19 baseline, summarise → pool → answer
  2. `lossless-hybrid`        — Phase-20 hybrid (RRF) single-hop retrieval
  3. `lossless-multihop`      — Phase-20 hybrid + structural multi-hop
  4. `lossless-planner`       — Phase-21 NEW. For each question:
        - try the QueryPlanner first; if it matches a pattern, execute
          the deterministic operator pipeline against the ledger and
          render the result (no LLM in the inner loop)
        - else fall through to lossless-multihop
  5. `oracle`                 — full doc in prompt (Phase-19 baseline)

Per question we record the same fields as Phase 20, plus:
  - `planner_pattern`   — which Phase-21 pattern matched, or
                          "(unmatched)" if it fell through
  - `planner_used`      — bool, true iff the answer came from the
                          operator pipeline (no LLM in the inner loop)
  - `failure_class`     — extended to 4 categories:
        ok / retrieval_miss / reduction_error / llm_error
        (`reduction_error` = planner ran, returned a result, result wrong)

Repeat mode: `--repeats N` runs N corpora with different seeds; full
aggregate (mean/min/max/σ) reported per condition.

Reproduce:
    python -m vision_mvp.experiments.phase21_compute --mode mock --n 24 --repeats 3
    python -m vision_mvp.experiments.phase21_compute --mode ollama --model qwen2.5:0.5b --n 12
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
from vision_mvp.core.context_ledger import ContextLedger, hash_embedding
from vision_mvp.core.query_planner import QueryPlanner
from vision_mvp.experiments.phase19_lossless import (
    MockLLM, run_map_reduce, run_oracle,
)
from vision_mvp.tasks.needle_corpus import NeedleCorpus, NeedleQuestion


# =============================================================================
# Result types (extended from Phase 20)
# =============================================================================


@dataclass
class QResult:
    question: str
    gold: str
    kind: str
    answer: str
    exact_correct: bool
    fact_in_input: bool
    retrieval_hit_at_k: bool
    failure_class: str          # ok / retrieval_miss / reduction_error / llm_error
    prompt_chars: int
    fetch_count: int = 0
    fetched_bytes: int = 0
    cited_cids: list[str] = field(default_factory=list)
    source_section: tuple[int, ...] = field(default_factory=tuple)
    candidate_sections: list[int] = field(default_factory=list)
    n_hops: int = 1
    planner_pattern: str = ""
    planner_used: bool = False
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
        n_recall = sum(1 for q in self.questions if q.retrieval_hit_at_k)
        prompts = [q.prompt_chars for q in self.questions]
        # Failure breakdown.
        fc: dict[str, int] = {
            "ok": 0, "retrieval_miss": 0, "reduction_error": 0, "llm_error": 0}
        for q in self.questions:
            fc[q.failure_class] = fc.get(q.failure_class, 0) + 1
        # Per-kind exact rate.
        per_kind: dict[str, dict] = {}
        for q in self.questions:
            d = per_kind.setdefault(q.kind, {"n": 0, "exact": 0})
            d["n"] += 1
            d["exact"] += int(q.exact_correct)
        # Planner usage.
        n_planned = sum(1 for q in self.questions if q.planner_used)
        return {
            "name": self.name,
            "n_questions": n,
            "exact_correct": n_exact,
            "exact_correct_rate": round(n_exact / n, 4),
            "fact_in_input": n_in,
            "fact_in_input_rate": round(n_in / n, 4),
            "retrieval_hit_at_k": n_recall,
            "retrieval_hit_rate": round(n_recall / n, 4),
            "mean_prompt_chars": round(sum(prompts) / n, 1),
            "max_prompt_chars": max(prompts),
            "mean_fetch_count": round(
                sum(q.fetch_count for q in self.questions) / n, 2),
            "failure_classes": fc,
            "per_kind_exact": per_kind,
            "n_planned": n_planned,
            "planner_usage_rate": round(n_planned / n, 4),
            "setup_seconds": round(self.setup_seconds, 2),
            "answer_seconds": round(self.answer_seconds, 2),
            "n_llm_setup_calls": self.n_llm_setup_calls,
            "n_llm_answer_calls": self.n_llm_answer_calls,
            "extras": self.extras,
        }


def _classify(*, exact: bool, fact_in: bool, planner_used: bool) -> str:
    if exact:
        return "ok"
    if planner_used:
        # Planner ran and produced a wrong answer — that's a reduction error
        # (the planner pipeline itself, not the LLM).
        return "reduction_error"
    if fact_in:
        return "llm_error"
    return "retrieval_miss"


# =============================================================================
# The Phase-21 condition: planner-first, fall back to multihop worker
# =============================================================================


def run_planner_condition(
    *,
    name: str,
    corpus: NeedleCorpus,
    llm,
    embed_fn: Callable[[str], np.ndarray],
    embed_dim: int,
    prompt_budget_chars: int,
    top_k: int,
    fetch_chars_per_handle: int,
    progress: Callable[[str], None] = print,
) -> CondReport:
    """Try `QueryPlanner` first; if no plan, fall through to the
    Phase-20 multi-hop hybrid worker."""
    rep = CondReport(name=name)

    # ---- index
    t0 = time.time()
    ledger = ContextLedger(embed_dim=embed_dim, embed_fn=embed_fn,
                            max_artifacts=10_000,
                            max_artifact_chars=64_000)
    n_index = 0
    for i, sec in enumerate(corpus.sections):
        ledger.put(sec, metadata=corpus.section_meta[i])   # full metadata!
        n_index += 1
        if (i + 1) % max(1, len(corpus.sections) // 5) == 0:
            progress(f"  [{name}] indexed {i+1}/{len(corpus.sections)}")
    rep.setup_seconds = time.time() - t0
    rep.extras["n_embed_calls"] = n_index
    rep.extras["search_mode"] = "planner-first → hybrid-multihop fallback"

    planner = QueryPlanner()
    worker = BoundedRetrievalWorker(
        ledger=ledger, llm_call=llm,
        prompt_budget_chars=prompt_budget_chars,
        top_k=top_k, fetch_chars_per_handle=fetch_chars_per_handle,
        search_mode="hybrid", max_hops=3,
    )

    # ---- answer phase
    t0 = time.time()
    if isinstance(llm, MockLLM):
        llm.mode = "extract"
    for q in corpus.questions:
        plan_res = planner.plan(q.question)
        used_planner = plan_res.plan is not None

        if used_planner:
            # Pure operator pipeline. No LLM in the inner loop.
            stage, op_trace = plan_res.plan.execute(ledger)
            answer_text = plan_res.plan.render(stage)
            # We DO call the LLM once at the end to wrap the result in a
            # natural-language sentence — but we send the planner's exact
            # answer as the only authoritative input. This keeps the
            # interface symmetric with other conditions and lets the
            # `exact_correct` scoring rule see a model-style answer.
            wrap_prompt = (
                "You are answering a question. The system has already "
                "computed the EXACT answer using deterministic operators. "
                "Quote the answer verbatim.\n\n"
                f"Question: {q.question}\n"
                f"Computed answer: {answer_text}\n\n"
                "Final answer:")
            if isinstance(llm, MockLLM):
                llm.current_gold = q.gold
            try:
                wrapped = llm(wrap_prompt)
            except Exception:
                wrapped = answer_text
            rep.n_llm_answer_calls += 1
            # Score against both the wrapped and raw planner output —
            # the planner's exact answer should be what counts; the LLM
            # wrap is cosmetic.
            full = wrapped + "\n[planner: " + answer_text + "]"
            exact = NeedleCorpus.score_exact(full, q)
            fact_in = q.gold.lower() in (answer_text.lower() +
                                          " " + wrap_prompt.lower())
            # Retrieval recall: did the planner's pipeline touch the
            # source sections?
            touched = set()
            for t in op_trace:
                touched.update(t.cids_touched)
            # Map cid → section_idx
            cid_to_section = {h.cid: h.metadata_dict().get("section_idx")
                              for h in ledger.all_handles()}
            cand_sections = sorted({cid_to_section[cid]
                                     for cid in touched
                                     if cid in cid_to_section})
            # If the operator was metadata-only, no cids were "touched"
            # — but the operator scanned ALL handles. Treat that as a
            # full scan: every section was touched.
            if not cand_sections and any(
                    op.in_size > 0 for op in op_trace):
                cand_sections = list(range(len(corpus.sections)))
            retrieval_hit = any(s in cand_sections for s in q.source_section)

            fc = _classify(exact=exact, fact_in=fact_in,
                           planner_used=True)
            rep.questions.append(QResult(
                question=q.question, gold=q.gold, kind=q.kind,
                answer=full, exact_correct=exact, fact_in_input=fact_in,
                retrieval_hit_at_k=retrieval_hit, failure_class=fc,
                prompt_chars=len(wrap_prompt), fetch_count=0,
                fetched_bytes=0,
                cited_cids=[], source_section=q.source_section,
                candidate_sections=cand_sections, n_hops=1,
                planner_pattern=plan_res.pattern, planner_used=True,
                extra={"planner_rationale": plan_res.rationale,
                       "planner_op_trace": [
                           {"name": t.name, "in": t.in_size,
                            "out": t.out_size,
                            "cids_touched": len(t.cids_touched)}
                           for t in op_trace]},
            ))
        else:
            # Fall through to Phase-20 hybrid multi-hop worker.
            if isinstance(llm, MockLLM):
                llm.current_gold = q.gold
            r = worker.answer(q.question)
            rep.n_llm_answer_calls += 1
            cand_sections = []
            for hop in r.hops:
                for cid in hop.candidate_cids:
                    md = ledger._entries[cid].metadata    # type: ignore[attr-defined]
                    if "section_idx" in md and md["section_idx"] not in cand_sections:
                        cand_sections.append(md["section_idx"])
            terminal = q.source_section[-1]
            retrieval_hit = terminal in cand_sections
            fact_in = q.gold.lower() in r.prompt.lower()
            exact = NeedleCorpus.score_exact(r.answer, q)
            fc = _classify(exact=exact, fact_in=fact_in,
                           planner_used=False)
            rep.questions.append(QResult(
                question=q.question, gold=q.gold, kind=q.kind,
                answer=r.answer, exact_correct=exact, fact_in_input=fact_in,
                retrieval_hit_at_k=retrieval_hit, failure_class=fc,
                prompt_chars=r.prompt_chars, fetch_count=r.fetch_count,
                fetched_bytes=r.fetched_bytes, cited_cids=r.cited_cids,
                source_section=q.source_section,
                candidate_sections=cand_sections, n_hops=len(r.hops),
                planner_pattern="(unmatched)", planner_used=False,
            ))
    rep.answer_seconds = time.time() - t0
    return rep


# =============================================================================
# Adapters — reuse Phase 19 / 20 runners with a uniform result schema
# =============================================================================


def _adapt_phase19(name: str, rep19, planner_used_default=False) -> CondReport:
    """Convert a Phase-19-shaped report into the Phase-21 schema."""
    out = CondReport(
        name=name, setup_seconds=rep19.setup_seconds,
        answer_seconds=rep19.answer_seconds,
        n_llm_setup_calls=rep19.n_llm_setup_calls,
        n_llm_answer_calls=rep19.n_llm_answer_calls,
        extras=dict(rep19.extras))
    for q in rep19.questions:
        fc = _classify(exact=q.exact_correct, fact_in=q.fact_in_input,
                       planner_used=False)
        out.questions.append(QResult(
            question=q.question, gold=q.gold, kind=q.kind, answer=q.answer,
            exact_correct=q.exact_correct, fact_in_input=q.fact_in_input,
            retrieval_hit_at_k=q.fact_in_input,    # treat as same here
            failure_class=fc, prompt_chars=q.prompt_chars,
            source_section=q.source_section,
            planner_pattern="(unmatched)", planner_used=False,
        ))
    return out


def run_lossless_phase20(
    *, name: str, search_mode: str, max_hops: int,
    corpus: NeedleCorpus, llm, embed_fn, embed_dim: int,
    prompt_budget_chars: int, top_k: int, fetch_chars_per_handle: int,
    progress,
) -> CondReport:
    """Run the Phase-20 lossless-hybrid / lossless-multihop conditions
    under the Phase-21 result schema."""
    from vision_mvp.experiments.phase20_substrate import run_lossless
    rep20 = run_lossless(
        name=name, corpus=corpus, llm=llm, embed_fn=embed_fn,
        embed_dim=embed_dim, prompt_budget_chars=prompt_budget_chars,
        top_k=top_k, fetch_chars_per_handle=fetch_chars_per_handle,
        search_mode=search_mode, max_hops=max_hops, progress=progress)
    out = CondReport(
        name=name, setup_seconds=rep20.setup_seconds,
        answer_seconds=rep20.answer_seconds,
        n_llm_setup_calls=rep20.n_llm_setup_calls,
        n_llm_answer_calls=rep20.n_llm_answer_calls,
        extras=dict(rep20.extras))
    for q in rep20.questions:
        fc = _classify(exact=q.exact_correct, fact_in=q.fact_in_input,
                       planner_used=False)
        out.questions.append(QResult(
            question=q.question, gold=q.gold, kind=q.kind, answer=q.answer,
            exact_correct=q.exact_correct, fact_in_input=q.fact_in_input,
            retrieval_hit_at_k=q.retrieval_hit_at_k, failure_class=fc,
            prompt_chars=q.prompt_chars, fetch_count=q.fetch_count,
            fetched_bytes=q.fetched_bytes, cited_cids=q.cited_cids,
            source_section=q.source_section,
            candidate_sections=q.candidate_sections, n_hops=q.n_hops,
            planner_pattern="(unmatched)", planner_used=False,
        ))
    return out


# =============================================================================
# One pass + repeats
# =============================================================================


def run_one_pass(args, corpus: NeedleCorpus, llm_factory, embed_fn,
                 progress=print) -> dict[str, CondReport]:
    out: dict[str, CondReport] = {}

    if not args.skip_mr:
        progress("\n" + "=" * 78)
        progress("CONDITION: map_reduce baseline")
        progress("=" * 78)
        rep_mr = run_map_reduce(
            corpus, llm_factory(),
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
        out[name] = run_lossless_phase20(
            name=name, search_mode=search_mode, max_hops=max_hops,
            corpus=corpus, llm=llm_factory(), embed_fn=embed_fn,
            embed_dim=args.embed_dim,
            prompt_budget_chars=args.prompt_budget,
            top_k=args.top_k,
            fetch_chars_per_handle=args.fetch_chars,
            progress=progress)

    if "lossless-planner" not in args.skip_conditions:
        progress("\n" + "=" * 78)
        progress("CONDITION: lossless-planner (Phase 21 NEW)")
        progress("=" * 78)
        out["lossless-planner"] = run_planner_condition(
            name="lossless-planner", corpus=corpus, llm=llm_factory(),
            embed_fn=embed_fn, embed_dim=args.embed_dim,
            prompt_budget_chars=args.prompt_budget, top_k=args.top_k,
            fetch_chars_per_handle=args.fetch_chars, progress=progress)

    if not args.skip_oracle:
        progress("\n" + "=" * 78)
        progress("CONDITION: oracle (full doc, if it fits)")
        progress("=" * 78)
        rep_o = run_oracle(corpus, llm_factory(), prompt_budget_chars=128_000)
        out["oracle"] = _adapt_phase19("oracle", rep_o)
    return out


def aggregate_repeats(per_rep_aggs: list[dict]) -> dict:
    if not per_rep_aggs:
        return {}
    cond_names = list(per_rep_aggs[0].keys())
    out: dict = {"n_repeats": len(per_rep_aggs), "by_condition": {}}
    for cn in cond_names:
        vals_exact = [r[cn]["exact_correct_rate"] for r in per_rep_aggs
                      if cn in r and r[cn]]
        vals_recall = [r[cn]["retrieval_hit_rate"] for r in per_rep_aggs
                       if cn in r and r[cn]]
        if not vals_exact:
            continue
        out["by_condition"][cn] = {
            "exact_correct_rate": _stats(vals_exact),
            "retrieval_hit_rate": _stats(vals_recall),
        }
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


# =============================================================================
# Per-question-class breakdown
# =============================================================================


def _classify_question(q: NeedleQuestion) -> str:
    if q.kind in NeedleCorpus._AGG_KINDS:
        return "aggregation"
    if len(q.source_section) >= 2:
        return "multi_hop"
    return "single_hop"


def per_class_breakdown(corpus: NeedleCorpus,
                         reports: dict[str, CondReport]) -> dict:
    """For each (condition, question class), report exact_correct rate.

    Question classes:
      - single_hop:  needs one section
      - multi_hop:   needs 2 sections via cross-reference
      - aggregation: needs many sections (Phase-21 target)
    """
    out: dict[str, dict[str, dict]] = {}
    for name, rep in reports.items():
        per_class: dict[str, dict] = {}
        for q_result, q_spec in zip(rep.questions, corpus.questions):
            cls = _classify_question(q_spec)
            d = per_class.setdefault(cls, {"n": 0, "exact": 0})
            d["n"] += 1
            d["exact"] += int(q_result.exact_correct)
        out[name] = per_class
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


def make_ollama_embed(model: str, dim: int) -> Callable[[str], np.ndarray]:
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
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--model", default="qwen2.5:0.5b")
    ap.add_argument("--embed-dim", type=int, default=64)
    ap.add_argument("--prompt-budget", type=int, default=4000)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--fetch-chars", type=int, default=600)
    ap.add_argument("--summary-prompt-cap", type=int, default=4000)
    ap.add_argument("--skip-oracle", action="store_true")
    ap.add_argument("--skip-mr", action="store_true")
    ap.add_argument("--skip-conditions", nargs="*", default=[])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print(f"Phase-21 compute benchmark — mode={args.mode} n={args.n} "
          f"repeats={args.repeats}", flush=True)
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
        corpus = NeedleCorpus(n_sections=args.n, seed=rep_seed,
                               include_aggregation=True)
        corpus.build()
        agg_qs = corpus.aggregation_questions()
        single_qs = corpus.single_hop_questions()
        multi_qs = corpus.multi_hop_questions()
        print(f"Corpus: {corpus.word_count} words, "
              f"{len(corpus.sections)} sections, "
              f"{len(corpus.questions)} questions "
              f"({len(single_qs)} single, "
              f"{len(multi_qs)} multi-hop, "
              f"{len(agg_qs)} aggregation).", flush=True)

        reports = run_one_pass(args, corpus, llm_factory, embed_fn)
        agg_by_cond = {name: r.aggregate() for name, r in reports.items()}
        per_class = per_class_breakdown(corpus, reports)

        per_rep_aggs.append(agg_by_cond)
        per_rep_full.append({
            "rep": rep_i, "seed": rep_seed,
            "aggregate": agg_by_cond,
            "per_question_class": per_class,
            "questions_per_condition": {
                name: [asdict(q) for q in r.questions]
                for name, r in reports.items()
            },
        })

        # Per-rep scoreboard
        print("\n" + "-" * 78)
        print(f"REPEAT {rep_i+1} OVERALL SCOREBOARD")
        print("-" * 78)
        head = (f"{'condition':>22} | {'exact':>11} | {'recall@k':>11} | "
                f"{'planned':>8} | {'pmt':>7}")
        print(head)
        print("-" * len(head))
        for name in ("map_reduce", "lossless-hybrid", "lossless-multihop",
                     "lossless-planner", "oracle"):
            if name not in agg_by_cond:
                continue
            a = agg_by_cond[name]
            if not a:
                continue
            print(f"{name:>22} | "
                  f"{a['exact_correct']}/{a['n_questions']:>3} ({a['exact_correct_rate']*100:>4.1f}%) | "
                  f"{a['retrieval_hit_at_k']}/{a['n_questions']:>3} ({a['retrieval_hit_rate']*100:>4.1f}%) | "
                  f"{a['n_planned']:>3}/{a['n_questions']:>3} | "
                  f"{a['mean_prompt_chars']:>7.0f}")

        # Per-class scoreboard — the headline diagnostic for Phase 21
        print()
        print("PER-CLASS SCOREBOARD (exact_correct / n)")
        print("-" * 78)
        classes = ("single_hop", "multi_hop", "aggregation")
        head = f"{'condition':>22} | " + " | ".join(f"{c:>13}" for c in classes)
        print(head)
        print("-" * len(head))
        for name in ("map_reduce", "lossless-hybrid", "lossless-multihop",
                     "lossless-planner", "oracle"):
            if name not in per_class:
                continue
            row = [f"{name:>22}"]
            for c in classes:
                d = per_class[name].get(c, {"n": 0, "exact": 0})
                if d["n"] == 0:
                    row.append(f"{'-':>13}")
                else:
                    row.append(
                        f"{d['exact']}/{d['n']} ({100*d['exact']/d['n']:>4.0f}%)".rjust(13))
            print(" | ".join(row))

    cross = aggregate_repeats(per_rep_aggs)
    if args.repeats > 1:
        print("\n" + "=" * 78)
        print(f"AGGREGATE OVER {args.repeats} REPEATS")
        print("=" * 78)
        for cn, stats in cross["by_condition"].items():
            ex = stats["exact_correct_rate"]
            print(f"  {cn:>22}: exact mean={ex['mean']*100:.1f}% "
                  f"(σ={ex.get('stddev', 0)*100:.1f}, "
                  f"range=[{ex['min']*100:.1f},{ex['max']*100:.1f}])")

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
