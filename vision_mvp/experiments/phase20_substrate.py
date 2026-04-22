"""Phase 20 — Stronger context-substrate benchmark.

Extends `phase19_lossless` along two axes:

  1. **Retrieval upgrade.** New conditions `lossless-hybrid` (RRF over
     dense + BM25) and `lossless-multihop` (hybrid + structural
     cross-reference expansion). Both are zero-summarisation paths.

  2. **Multi-hop questions.** The needle corpus now embeds a
     "Related: OS-…" cross-reference into every section. New question
     kinds (`vendor_via_related`, `sla_via_related`) require following
     that reference to a second section.

Conditions compared:
  - `map_reduce`        — Phase-19 baseline (summarise → pool → answer)
  - `lossless-dense`    — Phase-19 lossless: dense-only single-hop
  - `lossless-hybrid`   — Phase-20 hybrid (RRF), single-hop
  - `lossless-multihop` — Phase-20 hybrid + structural cross-ref expansion
                         (max_hops=3)
  - `oracle`            — full-doc baseline; skipped if doc > budget

Per question we report:
  - exact_correct        — gold substring in answer
  - fact_in_input        — gold substring in answering LLM's prompt
  - retrieval_hit_at_k   — was the right source section in the top-k
                           returned by retrieval (terminal hop only)?
  - prompt_chars         — answering-prompt size
  - fetch_count          — how many ledger.fetch calls
  - failure_class        — one of: ok, retrieval_miss, llm_error, both

Repeat mode: `--repeats N` runs the benchmark N times with different
seeds; aggregate stats (mean / min / max / σ of exact accuracy) appear
in the JSON output. Useful for guarding against single-seed flukes.

Usage:
    # Smoke (no Ollama, deterministic):
    python -m vision_mvp.experiments.phase20_substrate --mode mock --n 32 --repeats 3

    # Real LLM:
    python -m vision_mvp.experiments.phase20_substrate --mode ollama \\
        --model qwen2.5:0.5b --n 16 --repeats 1
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
from vision_mvp.experiments.phase19_lossless import (
    MockLLM, run_map_reduce, run_oracle,
)
from vision_mvp.tasks.needle_corpus import NeedleCorpus, NeedleQuestion


# ---------- Result dataclasses (mirror phase19 + add retrieval recall) ---


@dataclass
class QuestionResult:
    question: str
    gold: str
    kind: str
    answer: str
    exact_correct: bool
    fact_in_input: bool
    retrieval_hit_at_k: bool
    failure_class: str
    prompt_chars: int
    fetch_count: int = 0
    fetched_bytes: int = 0
    cited_cids: list[str] = field(default_factory=list)
    source_section: tuple[int, ...] = field(default_factory=tuple)
    candidate_sections: list[int] = field(default_factory=list)
    n_hops: int = 1
    extra: dict = field(default_factory=dict)


@dataclass
class ConditionReport:
    name: str
    questions: list[QuestionResult] = field(default_factory=list)
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
        n_in_input = sum(1 for q in self.questions if q.fact_in_input)
        n_recall = sum(1 for q in self.questions if q.retrieval_hit_at_k)
        n_in_input_correct = sum(
            1 for q in self.questions
            if q.fact_in_input and q.exact_correct)
        prompts = [q.prompt_chars for q in self.questions]
        # Failure breakdown.
        fc_counts: dict[str, int] = {
            "ok": 0, "retrieval_miss": 0, "llm_error": 0, "both": 0}
        for q in self.questions:
            fc_counts[q.failure_class] = fc_counts.get(q.failure_class, 0) + 1
        # Per-kind.
        per_kind: dict[str, dict] = {}
        for q in self.questions:
            d = per_kind.setdefault(q.kind, {"n": 0, "exact": 0, "fact_in": 0,
                                              "recall": 0})
            d["n"] += 1
            d["exact"] += int(q.exact_correct)
            d["fact_in"] += int(q.fact_in_input)
            d["recall"] += int(q.retrieval_hit_at_k)
        return {
            "name": self.name,
            "n_questions": n,
            "exact_correct": n_exact,
            "exact_correct_rate": round(n_exact / n, 4),
            "fact_in_input": n_in_input,
            "fact_in_input_rate": round(n_in_input / n, 4),
            "retrieval_hit_at_k": n_recall,
            "retrieval_hit_rate": round(n_recall / n, 4),
            "llm_accuracy_given_fact": round(
                n_in_input_correct / max(n_in_input, 1), 4),
            "mean_prompt_chars": round(sum(prompts) / n, 1),
            "max_prompt_chars": max(prompts),
            "mean_fetch_count": round(
                sum(q.fetch_count for q in self.questions) / n, 2),
            "mean_fetched_bytes": round(
                sum(q.fetched_bytes for q in self.questions) / n, 1),
            "failure_classes": fc_counts,
            "per_kind": per_kind,
            "setup_seconds": round(self.setup_seconds, 2),
            "answer_seconds": round(self.answer_seconds, 2),
            "n_llm_setup_calls": self.n_llm_setup_calls,
            "n_llm_answer_calls": self.n_llm_answer_calls,
            "extras": self.extras,
        }


# ---------- Failure classification --------------------------------------


def _classify_failure(*, exact: bool, fact_in: bool) -> str:
    """Decompose the answer outcome into substrate vs LLM behaviour.

    - ok               : exact correct
    - retrieval_miss   : gold not in input, answer wrong
    - llm_error        : gold IS in input but answer wrong
    - both             : (defensive) gold not in input AND somehow correct
                         — happens when the gold token coincidentally
                         appears in another section the worker cited.
    """
    if exact and fact_in:
        return "ok"
    if exact and not fact_in:
        # The model produced the gold without seeing the source. Either a
        # coincidental collision (e.g. SLA value 30 appears elsewhere) or
        # a hallucination that happened to be right.
        return "ok"      # still mark as ok for accuracy purposes
    if not exact and fact_in:
        return "llm_error"
    return "retrieval_miss"


# ---------- Adapted lossless run (with recall instrumentation) -----------


def run_lossless(
    *,
    name: str,
    corpus: NeedleCorpus,
    llm,
    embed_fn: Callable[[str], np.ndarray],
    embed_dim: int,
    prompt_budget_chars: int,
    top_k: int,
    fetch_chars_per_handle: int,
    search_mode: str,
    max_hops: int,
    progress: Callable[[str], None] = print,
) -> ConditionReport:
    rep = ConditionReport(name=name)

    # ---- index
    t0 = time.time()
    ledger = ContextLedger(embed_dim=embed_dim, embed_fn=embed_fn,
                           max_artifacts=10_000,
                           max_artifact_chars=64_000)
    n_index = 0
    for i, sec in enumerate(corpus.sections):
        ledger.put(sec, metadata={
            "section_idx": i,
            "incident_id": corpus.section_meta[i]["incident_id"],
        })
        n_index += 1
        if (i + 1) % max(1, len(corpus.sections) // 5) == 0:
            progress(f"  [{name}] indexed {i+1}/{len(corpus.sections)}")
    rep.setup_seconds = time.time() - t0
    rep.extras["n_embed_calls"] = n_index
    rep.extras["search_mode"] = search_mode
    rep.extras["max_hops"] = max_hops
    rep.extras["ledger_bytes_stored"] = ledger.stats_dict()["bytes_stored"]

    worker = BoundedRetrievalWorker(
        ledger=ledger, llm_call=llm,
        prompt_budget_chars=prompt_budget_chars,
        top_k=top_k, fetch_chars_per_handle=fetch_chars_per_handle,
        search_mode=search_mode, max_hops=max_hops,
    )

    # ---- answer phase
    t0 = time.time()
    if isinstance(llm, MockLLM):
        llm.mode = "extract"
    for q in corpus.questions:
        if isinstance(llm, MockLLM):
            llm.current_gold = q.gold
        r = worker.answer(q.question)
        rep.n_llm_answer_calls += 1

        # Aggregate retrieval candidate sections across all hops.
        candidate_sections: list[int] = []
        for hop in r.hops:
            for cid in hop.candidate_cids:
                # Look up section_idx in the metadata we attached.
                # This requires the ledger's internal entry — use the
                # worker's reference path instead.
                m = ledger._entries[cid].metadata    # type: ignore[attr-defined]
                if "section_idx" in m:
                    si = m["section_idx"]
                    if si not in candidate_sections:
                        candidate_sections.append(si)
        terminal_section = q.source_section[-1]    # the section the gold lives in
        retrieval_hit = terminal_section in candidate_sections

        fact_in = q.gold.lower() in r.prompt.lower()
        exact = NeedleCorpus.score_exact(r.answer, q)
        fc = _classify_failure(exact=exact, fact_in=fact_in)

        rep.questions.append(QuestionResult(
            question=q.question, gold=q.gold, kind=q.kind,
            answer=r.answer, exact_correct=exact, fact_in_input=fact_in,
            retrieval_hit_at_k=retrieval_hit, failure_class=fc,
            prompt_chars=r.prompt_chars, fetch_count=r.fetch_count,
            fetched_bytes=r.fetched_bytes, cited_cids=r.cited_cids,
            source_section=q.source_section,
            candidate_sections=candidate_sections,
            n_hops=len(r.hops),
        ))
    rep.answer_seconds = time.time() - t0
    return rep


# ---------- Map-reduce / oracle adapters (reuse Phase-19 code) ----------


def run_map_reduce_phase20(corpus: NeedleCorpus, llm,
                           summary_cap: int, answer_cap: int,
                           progress) -> ConditionReport:
    """Wrap the Phase-19 map-reduce runner with Phase-20 result schema."""
    rep19 = run_map_reduce(corpus, llm,
                           summary_prompt_chars_max=summary_cap,
                           answer_prompt_chars_max=answer_cap,
                           progress=progress)
    rep = ConditionReport(name="map_reduce",
                          setup_seconds=rep19.setup_seconds,
                          answer_seconds=rep19.answer_seconds,
                          n_llm_setup_calls=rep19.n_llm_setup_calls,
                          n_llm_answer_calls=rep19.n_llm_answer_calls,
                          extras=dict(rep19.extras))
    for q19, qspec in zip(rep19.questions, corpus.questions):
        fc = _classify_failure(exact=q19.exact_correct,
                               fact_in=q19.fact_in_input)
        rep.questions.append(QuestionResult(
            question=q19.question, gold=q19.gold, kind=q19.kind,
            answer=q19.answer, exact_correct=q19.exact_correct,
            fact_in_input=q19.fact_in_input,
            # map-reduce doesn't have explicit retrieval — treat the
            # pooled summaries as a single "candidate", and "hit" means
            # the gold survived summarisation (== fact_in_input).
            retrieval_hit_at_k=q19.fact_in_input,
            failure_class=fc, prompt_chars=q19.prompt_chars,
            source_section=q19.source_section,
        ))
    return rep


def run_oracle_phase20(corpus: NeedleCorpus, llm, budget: int) -> ConditionReport:
    rep19 = run_oracle(corpus, llm, prompt_budget_chars=budget)
    rep = ConditionReport(name="oracle",
                          setup_seconds=rep19.setup_seconds,
                          answer_seconds=rep19.answer_seconds,
                          n_llm_setup_calls=rep19.n_llm_setup_calls,
                          n_llm_answer_calls=rep19.n_llm_answer_calls,
                          extras=dict(rep19.extras))
    for q19 in rep19.questions:
        fc = _classify_failure(exact=q19.exact_correct,
                               fact_in=q19.fact_in_input)
        rep.questions.append(QuestionResult(
            question=q19.question, gold=q19.gold, kind=q19.kind,
            answer=q19.answer, exact_correct=q19.exact_correct,
            fact_in_input=q19.fact_in_input,
            retrieval_hit_at_k=q19.fact_in_input,
            failure_class=fc, prompt_chars=q19.prompt_chars,
            source_section=q19.source_section,
        ))
    return rep


# ---------- One full pass over all conditions ----------------------------


def run_one_pass(args, corpus: NeedleCorpus, llm_factory: Callable[[], object],
                 embed_fn: Callable[[str], np.ndarray],
                 progress: Callable[[str], None] = print) -> dict[str, ConditionReport]:
    reports: dict[str, ConditionReport] = {}

    if not args.skip_mr:
        progress("\n" + "=" * 78)
        progress("CONDITION: map_reduce baseline")
        progress("=" * 78)
        llm_mr = llm_factory()
        reports["map_reduce"] = run_map_reduce_phase20(
            corpus, llm_mr, summary_cap=args.summary_prompt_cap,
            answer_cap=args.prompt_budget, progress=progress)

    for name, search_mode, max_hops in [
        ("lossless-dense", "dense", 1),
        ("lossless-hybrid", "hybrid", 1),
        ("lossless-multihop", "hybrid", 3),
    ]:
        if name in args.skip_conditions:
            progress(f"  [{name}] skipped via --skip-conditions")
            continue
        progress("\n" + "=" * 78)
        progress(f"CONDITION: {name} (search={search_mode} max_hops={max_hops})")
        progress("=" * 78)
        llm_ll = llm_factory()
        reports[name] = run_lossless(
            name=name, corpus=corpus, llm=llm_ll, embed_fn=embed_fn,
            embed_dim=args.embed_dim,
            prompt_budget_chars=args.prompt_budget, top_k=args.top_k,
            fetch_chars_per_handle=args.fetch_chars,
            search_mode=search_mode, max_hops=max_hops,
            progress=progress)

    if not args.skip_oracle:
        progress("\n" + "=" * 78)
        progress("CONDITION: oracle (full doc, if it fits)")
        progress("=" * 78)
        llm_o = llm_factory()
        reports["oracle"] = run_oracle_phase20(
            corpus, llm_o, budget=128_000)
    return reports


# ---------- Repeat / aggregate ------------------------------------------


def aggregate_repeats(per_rep_aggs: list[dict]) -> dict:
    """Per-condition mean/min/max/σ of headline metrics across repeats."""
    if not per_rep_aggs:
        return {}
    cond_names = list(per_rep_aggs[0].keys())
    out: dict = {"n_repeats": len(per_rep_aggs), "by_condition": {}}
    for cn in cond_names:
        vals_exact = [r[cn]["exact_correct_rate"] for r in per_rep_aggs
                      if cn in r and r[cn]]
        vals_recall = [r[cn]["retrieval_hit_rate"] for r in per_rep_aggs
                       if cn in r and r[cn]]
        vals_factin = [r[cn]["fact_in_input_rate"] for r in per_rep_aggs
                       if cn in r and r[cn]]
        vals_pmt = [r[cn]["mean_prompt_chars"] for r in per_rep_aggs
                    if cn in r and r[cn]]
        if not vals_exact:
            continue
        out["by_condition"][cn] = {
            "exact_correct_rate": _stats(vals_exact),
            "retrieval_hit_rate": _stats(vals_recall),
            "fact_in_input_rate": _stats(vals_factin),
            "mean_prompt_chars": _stats(vals_pmt),
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


# ---------- LLM adapters -------------------------------------------------


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


# ---------- CLI ----------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["mock", "ollama"], default="mock")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--seed", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=1,
                    help="Number of independent repeats with different seeds")
    ap.add_argument("--model", default="qwen2.5:0.5b")
    ap.add_argument("--embed-dim", type=int, default=64)
    ap.add_argument("--prompt-budget", type=int, default=4000)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--fetch-chars", type=int, default=600)
    ap.add_argument("--summary-prompt-cap", type=int, default=4000)
    ap.add_argument("--skip-oracle", action="store_true")
    ap.add_argument("--skip-mr", action="store_true")
    ap.add_argument("--skip-conditions", nargs="*", default=[],
                    help="Condition names to skip, e.g. lossless-multihop")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print(f"Phase-20 substrate benchmark — mode={args.mode} n={args.n} "
          f"repeats={args.repeats}", flush=True)
    if args.mode == "mock":
        embed_fn = lambda t: hash_embedding(t, dim=args.embed_dim)
        llm_factory = make_mock_llm
    else:
        embed_fn = make_ollama_embed(args.model, dim=args.embed_dim)
        ollama_llm = make_ollama_llm(args.model, max_tokens=200)
        # Same client per pass so token accounting is per-pass.
        llm_factory = lambda: ollama_llm

    per_rep_aggs: list[dict] = []
    per_rep_full: list[dict] = []

    for rep_i in range(args.repeats):
        rep_seed = args.seed + rep_i
        print(f"\n# REPEAT {rep_i+1} / {args.repeats}  (seed={rep_seed})",
              flush=True)
        corpus = NeedleCorpus(n_sections=args.n, seed=rep_seed)
        corpus.build()
        print(f"Corpus: {corpus.word_count} words, "
              f"{len(corpus.sections)} sections, "
              f"{len(corpus.questions)} questions "
              f"({len(corpus.single_hop_questions())} single-hop, "
              f"{len(corpus.multi_hop_questions())} multi-hop).", flush=True)

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
        print("\n" + "-" * 78)
        print(f"REPEAT {rep_i+1} SCOREBOARD")
        print("-" * 78)
        head = (f"{'condition':>20} | {'exact':>10} | {'fact_in':>10} | "
                f"{'recall@k':>10} | {'mean_pmt':>9}")
        print(head)
        print("-" * len(head))
        for name in ("map_reduce", "lossless-dense", "lossless-hybrid",
                     "lossless-multihop", "oracle"):
            if name not in agg_by_cond:
                continue
            a = agg_by_cond[name]
            if not a:
                continue
            print(f"{name:>20} | "
                  f"{a['exact_correct']}/{a['n_questions']:>3} ({a['exact_correct_rate']*100:>4.1f}%) | "
                  f"{a['fact_in_input']}/{a['n_questions']:>3} ({a['fact_in_input_rate']*100:>4.1f}%) | "
                  f"{a['retrieval_hit_at_k']}/{a['n_questions']:>3} ({a['retrieval_hit_rate']*100:>4.1f}%) | "
                  f"{a['mean_prompt_chars']:>9.0f}")

    # Cross-repeat aggregate
    cross = aggregate_repeats(per_rep_aggs)
    if args.repeats > 1:
        print("\n" + "=" * 78)
        print(f"AGGREGATE OVER {args.repeats} REPEATS")
        print("=" * 78)
        for cn, stats in cross["by_condition"].items():
            ex = stats["exact_correct_rate"]
            rc = stats["retrieval_hit_rate"]
            sd_ex = ex.get("stddev", 0)
            sd_rc = rc.get("stddev", 0)
            print(f"  {cn:>20}: exact mean={ex['mean']*100:.1f}% "
                  f"(σ={sd_ex*100:.1f}, range=[{ex['min']*100:.1f},{ex['max']*100:.1f}])  "
                  f"recall mean={rc['mean']*100:.1f}% (σ={sd_rc*100:.1f})")

    if args.out:
        out = {
            "config": vars(args),
            "per_repeat": per_rep_full,
            "cross_repeat_aggregate": cross,
        }
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"\nWrote {args.out}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
