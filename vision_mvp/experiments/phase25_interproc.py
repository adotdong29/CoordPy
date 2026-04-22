"""Phase 25 — Conservative interprocedural-semantic benchmark.

Phase 24 restricted the exact slice to intraprocedural predicates
(per-function conservative flags computed against each function's own
body). Phase 25 widens that slice to **interprocedural** predicates
computed by propagating each Phase-24 flag transitively across a
local call graph:

  - count_trans_may_raise            / list_trans_may_raise
  - count_trans_may_write_global     / list_trans_may_write_global
  - count_trans_calls_subprocess     / list_trans_calls_subprocess
  - count_trans_calls_filesystem     / list_trans_calls_filesystem
  - count_trans_calls_network        / list_trans_calls_network
  - count_trans_calls_external_io                    (union count)
  - count_participates_in_cycle      / list_participates_in_cycle
  - count_has_unresolved_callees

Conditions per corpus (same machinery as Phase 24):

  1. `lossless-multihop`   — Phase-20 retrieval baseline.
  2. `lossless-planner`    — Phase-21 wrap-LLM planner.
  3. `direct-exact`        — Phase-22 direct render — zero LLM.

Headline: on the interprocedural slice, the substrate's direct-exact
guarantee carries over to a new class of questions (wrapper → helper
chains, mutual recursion, effect propagation) that the Phase-24
intra-only slice could not express. The comparison artefact we emit
includes a "Phase-24 baseline" column on the same corpora so the
widening is visible per predicate.

Reproduce:

    python -m vision_mvp.experiments.phase25_interproc \\
        --mode mock --out vision_mvp/results_phase25_mock.json

    python -m vision_mvp.experiments.phase25_interproc \\
        --mode mock --extra-roots <click-path> <json-path> \\
        --out vision_mvp/results_phase25_mock_external.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")))

from vision_mvp.core.context_ledger import hash_embedding
from vision_mvp.experiments.phase19_lossless import MockLLM
from vision_mvp.experiments.phase22_codebase import (
    run_direct_exact, run_lossless_retrieval, run_planner_with_wrap,
)
from vision_mvp.tasks.corpus_registry import (
    CorpusEntry, CorpusSpec, default_phase23_registry,
)
from vision_mvp.tasks.python_corpus import PythonCorpus


# =============================================================================
# Slice filters — Phase-25 interproc kinds + Phase-24 baseline kinds
# =============================================================================


_INTERPROC_KINDS: frozenset[str] = frozenset({
    "count_trans_may_raise", "list_trans_may_raise",
    "count_trans_may_write_global", "list_trans_may_write_global",
    "count_trans_calls_subprocess", "list_trans_calls_subprocess",
    "count_trans_calls_filesystem", "list_trans_calls_filesystem",
    "count_trans_calls_network", "list_trans_calls_network",
    "count_trans_calls_external_io",
    "count_participates_in_cycle", "list_participates_in_cycle",
    "count_has_unresolved_callees",
})


_PHASE24_BASELINE_KINDS: frozenset[str] = frozenset({
    "count_may_raise", "list_may_raise",
    "count_is_recursive", "list_is_recursive",
    "count_may_write_global", "list_may_write_global",
    "count_calls_subprocess", "list_calls_subprocess",
    "count_calls_filesystem", "list_calls_filesystem",
    "count_calls_network", "list_calls_network",
    "count_calls_external_io",
})


def _restrict(corpus: PythonCorpus, allowed: frozenset[str]) -> PythonCorpus:
    """Filter corpus.questions in place to the kinds in `allowed`."""
    corpus.questions = [q for q in corpus.questions if q.kind in allowed]
    return corpus


# =============================================================================
# Per-corpus driver
# =============================================================================


@dataclass
class CorpusReport:
    name: str
    family: str
    spec: dict
    coverage: dict
    interproc_totals: dict = field(default_factory=dict)
    intra_totals: dict = field(default_factory=dict)
    reports_by_condition: dict[str, dict] = field(default_factory=dict)
    questions_by_condition: dict[str, list[dict]] = field(default_factory=dict)
    # Phase-24 baseline on the SAME corpus — so we can report the
    # per-corpus widening of the exact slice.
    reports_phase24_baseline: dict[str, dict] = field(default_factory=dict)
    questions_phase24_baseline: dict[str, list[dict]] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


def _interproc_totals(corpus: PythonCorpus) -> dict:
    return {
        "n_functions_trans_may_raise": corpus.n_functions_trans_may_raise,
        "n_functions_trans_may_write_global":
            corpus.n_functions_trans_may_write_global,
        "n_functions_trans_calls_subprocess":
            corpus.n_functions_trans_calls_subprocess,
        "n_functions_trans_calls_filesystem":
            corpus.n_functions_trans_calls_filesystem,
        "n_functions_trans_calls_network":
            corpus.n_functions_trans_calls_network,
        "n_functions_trans_calls_external_io":
            corpus.n_functions_trans_calls_external_io,
        "n_functions_participates_in_cycle":
            corpus.n_functions_participates_in_cycle,
        "n_functions_has_unresolved_callees":
            corpus.n_functions_has_unresolved_callees,
    }


def _intra_totals(corpus: PythonCorpus) -> dict:
    return {
        "n_functions_may_raise": corpus.n_functions_may_raise,
        "n_functions_is_recursive": corpus.n_functions_is_recursive,
        "n_functions_may_write_global": corpus.n_functions_may_write_global,
        "n_functions_calls_subprocess": corpus.n_functions_calls_subprocess,
        "n_functions_calls_filesystem": corpus.n_functions_calls_filesystem,
        "n_functions_calls_network": corpus.n_functions_calls_network,
        "n_functions_calls_external_io":
            corpus.n_functions_calls_external_io,
    }


def _run_three_conditions(
    corpus: PythonCorpus, args, llm_factory, embed_fn, progress,
) -> tuple[dict[str, "CondReport"], dict[str, list[dict]]]:
    """Run direct-exact / lossless-planner / lossless-multihop on the
    current `corpus.questions` slice. Shared by interproc + baseline
    passes so they report the same way."""
    reps: dict[str, "CondReport"] = {}
    qs: dict[str, list[dict]] = {}

    rep_de = run_direct_exact(
        corpus=corpus, llm=llm_factory(), embed_fn=embed_fn,
        embed_dim=args.embed_dim,
        prompt_budget_chars=args.prompt_budget, top_k=args.top_k,
        fetch_chars_per_handle=args.fetch_chars, progress=progress)
    reps["direct-exact"] = rep_de.aggregate()
    qs["direct-exact"] = [asdict(q) for q in rep_de.questions]

    if "lossless-planner" not in args.skip_conditions:
        rep_lp = run_planner_with_wrap(
            corpus=corpus, llm=llm_factory(), embed_fn=embed_fn,
            embed_dim=args.embed_dim,
            prompt_budget_chars=args.prompt_budget, top_k=args.top_k,
            fetch_chars_per_handle=args.fetch_chars, progress=progress)
        reps["lossless-planner"] = rep_lp.aggregate()
        qs["lossless-planner"] = [asdict(q) for q in rep_lp.questions]

    if "lossless-multihop" not in args.skip_conditions:
        rep_mh = run_lossless_retrieval(
            name="lossless-multihop", search_mode="hybrid", max_hops=3,
            corpus=corpus, llm=llm_factory(), embed_fn=embed_fn,
            embed_dim=args.embed_dim,
            prompt_budget_chars=args.prompt_budget, top_k=args.top_k,
            fetch_chars_per_handle=args.fetch_chars, progress=progress)
        reps["lossless-multihop"] = rep_mh.aggregate()
        qs["lossless-multihop"] = [asdict(q) for q in rep_mh.questions]

    return reps, qs


def run_corpus(
    entry: CorpusEntry, args, llm_factory, embed_fn,
    progress: Callable[[str], None] = print,
) -> CorpusReport:
    """Two passes per corpus:
      1. interprocedural slice (new Phase-25 questions).
      2. Phase-24 baseline slice, for apples-to-apples comparison
         on the same corpus (only if the skip flag is not set).

    Each pass uses a fresh corpus build (the runners rebuild ledgers
    anyway, but the `corpus.questions` filter is mutating so we copy
    the source PythonCorpus before restricting).
    """
    cr = CorpusReport(
        name=entry.name, family=entry.spec.family,
        spec={"root": entry.spec.root, "max_files": entry.spec.max_files,
              "max_chars_per_file": entry.spec.max_chars_per_file,
              "seed": entry.spec.seed},
        coverage=entry.coverage.as_dict(),
        interproc_totals=_interproc_totals(entry.corpus),
        intra_totals=_intra_totals(entry.corpus),
    )

    # ---- Interprocedural slice ---------------------------------------
    progress(f"\n-- [{entry.name}] INTERPROC SLICE "
             f"({sum(1 for q in entry.corpus.questions if q.kind in _INTERPROC_KINDS)}"
             f" questions)")
    ip_corpus = _build_fresh_corpus(entry.spec)
    _restrict(ip_corpus, _INTERPROC_KINDS)
    if ip_corpus.questions:
        reps, qs = _run_three_conditions(
            ip_corpus, args, llm_factory, embed_fn, progress)
        cr.reports_by_condition = reps
        cr.questions_by_condition = qs
    else:
        progress(f"[{entry.name}] no interproc questions emitted — skipping")

    # ---- Phase-24 baseline slice (same corpus) ------------------------
    if not args.skip_phase24_baseline:
        progress(f"-- [{entry.name}] PHASE-24 BASELINE SLICE")
        base_corpus = _build_fresh_corpus(entry.spec)
        _restrict(base_corpus, _PHASE24_BASELINE_KINDS)
        if base_corpus.questions:
            reps, qs = _run_three_conditions(
                base_corpus, args, llm_factory, embed_fn, progress)
            cr.reports_phase24_baseline = reps
            cr.questions_phase24_baseline = qs
        else:
            progress(f"[{entry.name}] no Phase-24 baseline questions — skip")

    return cr


def _build_fresh_corpus(spec: CorpusSpec) -> PythonCorpus:
    """Materialise a PythonCorpus from a CorpusSpec. Wanted fresh
    because we mutate `corpus.questions` in place per slice."""
    c = PythonCorpus(
        root=spec.root, max_files=spec.max_files,
        max_chars_per_file=spec.max_chars_per_file, seed=spec.seed,
    )
    c.build()
    return c


# =============================================================================
# Scoreboards
# =============================================================================


def _print_scoreboard(reports: list[CorpusReport]) -> None:
    print("\n" + "=" * 118)
    print("PER-CORPUS PHASE-25 INTERPROCEDURAL SCOREBOARD")
    print("=" * 118)
    head = (f"{'corpus':>18} | {'cond':>20} | "
            f"{'exact':>14} | {'no_llm':>10} | "
            f"{'plan':>7} | {'pmt':>7}")
    print(head)
    print("-" * len(head))
    cond_order = ["lossless-multihop", "lossless-planner", "direct-exact"]
    for cr in reports:
        if not cr.reports_by_condition:
            print(f"{cr.name:>18} | (no interproc questions emitted)")
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
                  f"{no_llm:>10} | {pl:>7} | {agg['mean_prompt_chars']:>7.0f}")
    print("-" * len(head))


def _aggregate_cross_corpus(reports: list[CorpusReport],
                             key: str = "reports_by_condition") -> dict:
    out: dict[str, dict] = {}
    conds: set[str] = set()
    for cr in reports:
        conds.update(getattr(cr, key).keys())
    for cn in sorted(conds):
        ex_rates: list[float] = []
        no_llm_rates: list[float] = []
        n_correct = 0
        n_total = 0
        prompts: list[float] = []
        for cr in reports:
            a = getattr(cr, key).get(cn)
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


def _per_predicate_breakdown(reports: list[CorpusReport],
                              key: str = "questions_by_condition") -> dict:
    out: dict[str, dict[str, dict[str, int]]] = {}
    for cr in reports:
        for cn, qs in getattr(cr, key).items():
            for q in qs:
                kind = q["kind"]
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


def _print_widening(reports: list[CorpusReport]) -> None:
    """Per-corpus Phase-24 → Phase-25 widening: how many new
    deterministically-answerable questions the interprocedural
    slice adds, and the per-predicate intra → trans count ratio.
    """
    print("\n" + "=" * 118)
    print("PHASE-24 → PHASE-25 EXACT-SLICE WIDENING PER CORPUS")
    print("=" * 118)
    head = (f"{'corpus':>18} | {'intra fns':>10} | {'trans fns':>10} | "
            f"{'delta':>10} | {'in_cycle':>10} | {'unres':>10}")
    print(head)
    print("-" * len(head))
    for cr in reports:
        intra = cr.intra_totals
        tr = cr.interproc_totals
        intra_io = intra.get("n_functions_calls_external_io", 0)
        trans_io = tr.get("n_functions_trans_calls_external_io", 0)
        intra_mr = intra.get("n_functions_may_raise", 0)
        trans_mr = tr.get("n_functions_trans_may_raise", 0)
        print(f"{cr.name:>18} | "
              f"{intra_io:>10} | {trans_io:>10} | "
              f"{trans_io - intra_io:+>10} | "
              f"{tr.get('n_functions_participates_in_cycle', 0):>10} | "
              f"{tr.get('n_functions_has_unresolved_callees', 0):>10}")
    print("-" * len(head))
    print(f"{'(external_io counts; also see trans_may_raise delta below)':>80}")
    for cr in reports:
        intra_mr = cr.intra_totals.get("n_functions_may_raise", 0)
        trans_mr = cr.interproc_totals.get("n_functions_trans_may_raise", 0)
        print(f"  {cr.name:>18}  may_raise: intra={intra_mr:>4}  "
              f"trans={trans_mr:>4}  Δ={trans_mr - intra_mr:+>3}")


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
    ap.add_argument("--seed", type=int, default=25)
    ap.add_argument("--model", default="qwen2.5:0.5b")
    ap.add_argument("--embed-dim", type=int, default=64)
    ap.add_argument("--prompt-budget", type=int, default=4000)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--fetch-chars", type=int, default=600)
    ap.add_argument("--skip-conditions", nargs="*", default=[])
    ap.add_argument("--skip-phase24-baseline", action="store_true",
                    help="Skip the per-corpus Phase-24 baseline pass "
                         "(interproc slice only).")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print(f"Phase-25 interprocedural benchmark — mode={args.mode}",
          flush=True)

    if args.mode == "mock":
        embed_fn = lambda t: hash_embedding(t, dim=args.embed_dim)
        llm_factory = make_mock_llm
    else:
        embed_fn = make_ollama_embed(args.model, dim=args.embed_dim)
        ollama_llm = make_ollama_llm(args.model, max_tokens=200)
        llm_factory = lambda: ollama_llm

    reg = default_phase23_registry(extra_roots=args.extra_roots or None)
    if args.max_files is not None:
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

    print("\nCorpora to benchmark (Phase-25 interproc slice):")
    for e in entries:
        s = e.summary()
        tot = _interproc_totals(e.corpus)
        print(
            f"  - {s['name']:>18} | {s['family']:>20} | "
            f"files={s['n_files']:>4} fns={s['n_functions_total']:>4} "
            f"trans_mr={tot['n_functions_trans_may_raise']:>4} "
            f"cyc={tot['n_functions_participates_in_cycle']:>3} "
            f"trans_sp={tot['n_functions_trans_calls_subprocess']:>2} "
            f"trans_fs={tot['n_functions_trans_calls_filesystem']:>2} "
            f"trans_net={tot['n_functions_trans_calls_network']:>2} "
            f"trans_io={tot['n_functions_trans_calls_external_io']:>3} "
            f"unres={tot['n_functions_has_unresolved_callees']:>4}"
        )

    reports: list[CorpusReport] = []
    for e in entries:
        reports.append(run_corpus(e, args, llm_factory, embed_fn))

    _print_scoreboard(reports)

    cross = _aggregate_cross_corpus(reports, key="reports_by_condition")
    print("\n" + "=" * 118)
    print("PHASE-25 CROSS-CORPUS AGGREGATE (interproc slice only)")
    print("=" * 118)
    for cn, s in cross.items():
        print(f"  {cn:>20}: mean_exact={s['mean_exact_rate']*100:5.1f}% "
              f"(min={s['min_exact_rate']*100:5.1f}% "
              f"max={s['max_exact_rate']*100:5.1f}% "
              f"σ={s['stddev_exact_rate']*100:4.1f}) "
              f"pooled={s['pooled_exact_rate']*100:5.1f}% "
              f"(n={s['total_correct']}/{s['total_questions']}) "
              f"no_llm={s['mean_no_llm_rate']*100:5.1f}% "
              f"pmt={s['mean_prompt_chars']:.0f}")

    per_pred = _per_predicate_breakdown(
        reports, key="questions_by_condition")
    print("\nPER-PREDICATE BREAKDOWN (pooled across corpora) — INTERPROC")
    print("-" * 86)
    hdr = f"{'predicate':>28} | {'condition':>20} | {'ok':>5} / {'n':>3} rate"
    print(hdr)
    for pred in sorted(per_pred):
        for cn, counts in sorted(per_pred[pred].items()):
            n = counts["n"]
            ok = counts["ok"]
            rate = (ok / n) if n else 0.0
            print(f"{pred:>28} | {cn:>20} | {ok:>5} / {n:>3}  "
                  f"{rate*100:>5.1f}%")

    if not args.skip_phase24_baseline:
        # Emit a baseline aggregate + widening summary for context.
        cross_base = _aggregate_cross_corpus(
            reports, key="reports_phase24_baseline")
        print("\n" + "=" * 118)
        print("PHASE-24 BASELINE (same corpora, intraprocedural slice)")
        print("=" * 118)
        for cn, s in cross_base.items():
            print(f"  {cn:>20}: mean_exact={s['mean_exact_rate']*100:5.1f}% "
                  f"pooled={s['pooled_exact_rate']*100:5.1f}% "
                  f"(n={s['total_correct']}/{s['total_questions']}) "
                  f"no_llm={s['mean_no_llm_rate']*100:5.1f}%")

    _print_widening(reports)

    if args.out:
        payload = {
            "config": {k: (v if not callable(v) else repr(v))
                       for k, v in vars(args).items()},
            "corpora_summary": [e.summary() for e in entries],
            "per_corpus": [cr.as_dict() for cr in reports],
            "cross_corpus_aggregate_interproc": cross,
            "cross_corpus_aggregate_phase24_baseline": (
                _aggregate_cross_corpus(
                    reports, key="reports_phase24_baseline")
                if not args.skip_phase24_baseline else {}),
            "per_predicate_breakdown_interproc": per_pred,
        }
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"\nWrote {args.out}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
