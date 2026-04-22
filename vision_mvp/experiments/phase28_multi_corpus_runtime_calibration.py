"""Phase 28 — Multi-corpus runtime calibration + explicit/implicit-raise axis.

Phase 27 ran the Phase-26 probes against a single real corpus
(``vision-core``) and two auxiliary corpora (``vision-tasks``), and
exposed a new boundary class: **implicit raises from builtin
operations on arguments outside the function's semantic domain.**
Phase 28 pushes that research in two directions simultaneously:

  1. **Multi-corpus**. Run the runtime calibration stack against
     every local Phase-23 corpus (``vision-core``, ``vision-tasks``,
     ``vision-tests``, ``vision-experiments``) in one benchmark,
     with per-corpus coverage + per-predicate metrics + pooled
     aggregates. Coverage is reported as a first-class variable
     alongside calibration — the witness-availability bound
     (Theorem P27-1) is the dominant effect and must be visible
     per corpus.

  2. **Explicit vs implicit raise separation**. ``may_raise`` (the
     Phase-24 contract) is joined by two new predicates — one
     matching the Phase-24 contract exactly
     (``may_raise_explicit``) and one covering the Phase-27-surfaced
     implicit-raise surface (``may_raise_implicit``). The analyzer
     flags are drawn from ``trans_may_raise`` and the new
     ``trans_may_raise_implicit`` respectively; runtime observation
     classifies each caught exception by traceback origin.

Reproduce the headline run:

    python -m vision_mvp.experiments.phase28_multi_corpus_runtime_calibration \\
        --seeds 0 1 2 --budget 0.08 \\
        --out vision_mvp/results_phase28_multi.json

Repeat-run variance:

    python -m vision_mvp.experiments.phase28_multi_corpus_runtime_calibration \\
        --seeds 0 1 2 3 4 --budget 0.12 \\
        --out vision_mvp/results_phase28_multi_5seeds.json

See ``vision_mvp/RESULTS_PHASE28.md`` for the research framing,
the Phase-28 theorem set, and the per-corpus tables.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")))

from vision_mvp.core.code_corpus_runtime import (
    RUNTIME_DECIDABLE_PREDICATES,
    calibrate_corpus, collect_divergences, rows_as_dict_list,
    summarise_corpus_calibration,
)
from vision_mvp.tasks.corpus_runtime_recipes import (
    DEFAULT_PHASE27_SKIP_FILES, build_default_recipe_registry,
)


# =============================================================================
# Phase-28 predicate set
# =============================================================================


# The six Phase-26 predicates PLUS the two Phase-28 raise-bucket
# predicates. The composite `may_raise` predicate is kept for
# backwards comparison with Phase-27 headline numbers; the
# explicit/implicit pair are the new research axis.
PHASE28_PREDICATES: tuple[str, ...] = (
    "calls_filesystem",
    "calls_network",
    "calls_subprocess",
    "may_raise",
    "may_raise_explicit",
    "may_raise_implicit",
    "may_write_global",
    "participates_in_cycle",
)


# =============================================================================
# Multi-corpus catalogue — local only, deterministic
# =============================================================================


def _default_corpora() -> list[tuple[str, str, str]]:
    """(name, root, importable_package) triples for every Phase-23
    local corpus the runtime probe can handle under the default
    recipe strategy. External corpora (e.g. ``click``, stdlib
    ``json``) are OQ-28a and handled by the ``--extra`` flag."""
    repo = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", ".."))
    candidates = [
        ("vision-core",
         os.path.join(repo, "vision_mvp/core"),
         "vision_mvp.core"),
        ("vision-tasks",
         os.path.join(repo, "vision_mvp/tasks"),
         "vision_mvp.tasks"),
        ("vision-tests",
         os.path.join(repo, "vision_mvp/tests"),
         "vision_mvp.tests"),
        ("vision-experiments",
         os.path.join(repo, "vision_mvp/experiments"),
         "vision_mvp.experiments"),
    ]
    return [(n, r, p) for (n, r, p) in candidates if os.path.isdir(r)]


# =============================================================================
# Per-corpus helper
# =============================================================================


def run_one_corpus(corpus_name: str, corpus_root: str,
                    corpus_package: str | None,
                    seeds: tuple[int, ...], budget_s: float,
                    predicates: tuple[str, ...],
                    max_files: int | None,
                    progress=print) -> dict:
    """Execute the Phase-28 multi-predicate probe pipeline on one
    corpus. Returns a serialisable per-corpus record.
    """
    progress(f"[{corpus_name}] corpus root: {corpus_root}")
    reg = build_default_recipe_registry()
    t0 = time.time()
    rows, cov = calibrate_corpus(
        corpus_name, corpus_root,
        corpus_package=corpus_package, recipe_registry=reg,
        predicates=predicates, seeds=seeds, budget_s=budget_s,
        skip_files=DEFAULT_PHASE27_SKIP_FILES,
        max_files=max_files,
        progress=progress,
    )
    metrics = summarise_corpus_calibration(rows, predicates=predicates)
    divergences = collect_divergences(rows, predicates=predicates)
    elapsed = time.time() - t0

    # Analyzer aggregates on the corpus: useful alongside per-
    # predicate metrics so a reader can separate the witness-
    # availability bound (coverage) from the analyzer's native
    # flag count.
    analyzer_counts: dict[str, int] = {}
    for p in predicates:
        n_true = 0
        for row in rows:
            if bool(row.static_flags.get(p, False)):
                n_true += 1
        analyzer_counts[p] = n_true

    return {
        "corpus_name": corpus_name,
        "corpus_root": corpus_root,
        "corpus_package": corpus_package,
        "elapsed_s": round(elapsed, 2),
        "coverage": cov.as_dict(),
        "analyzer_counts_per_predicate": analyzer_counts,
        "metrics_per_predicate": {p: m.as_dict() for p, m in metrics.items()},
        "divergences": [
            {"qname": d.qname, "predicate": d.predicate, "kind": d.kind,
             "static_flag": d.static_flag, "runtime_flag": d.runtime_flag,
             "witnesses": list(d.witnesses),
             "callable_status": d.callable_status,
             "module_name": d.module_name}
            for d in divergences
        ],
        "rows": rows_as_dict_list(rows),
    }


# =============================================================================
# Pretty-print helpers
# =============================================================================


def _print_coverage(entry: dict) -> None:
    c = entry["coverage"]
    print(f"\n--- {entry['corpus_name']} ({entry['corpus_root']}) ---")
    print(f"  elapsed: {entry['elapsed_s']}s")
    print(f"  n_total functions analysed:         {c['n_total']}")
    print(f"  ready_no_args + ready_typed + ready_curated: "
          f"{c['ready_no_args']} + {c['ready_typed']} + "
          f"{c['ready_curated']} = "
          f"{c['ready_no_args'] + c['ready_typed'] + c['ready_curated']}  "
          f"({c['ready_fraction']*100:.1f}%)")
    print(f"  probed / entered / calibrated:      "
          f"{c['n_probed']} / {c['n_entered']} / {c['n_calibrated']}  "
          f"({c['calibrated_fraction']*100:.1f}% of total)")
    print(f"  timeout                             {c['n_timeout']}")
    print(f"  unsupported_method / import / missing / untyped / varargs / "
          f"async / generator = "
          f"{c['unsupported_method']} / {c['unsupported_import']} / "
          f"{c['unsupported_missing']} / {c['unsupported_untyped']} / "
          f"{c['unsupported_varargs']} / {c['unsupported_async']} / "
          f"{c['unsupported_generator']}")


def _print_metrics(entry: dict) -> None:
    print(f"\n  PER-PREDICATE METRICS ({entry['corpus_name']})")
    head = (f"    {'predicate':>22} | {'applic':>6} | {'entered':>7} | "
            f"{'S_true':>6} | {'R_true':>6} | {'agree':>5} | "
            f"{'FP':>3} | {'FN':>3}")
    print(head)
    print("    " + "-" * (len(head) - 4))
    order = PHASE28_PREDICATES
    for p in order:
        m = entry["metrics_per_predicate"].get(p)
        if m is None:
            continue
        print(f"    {p:>22} | {m['n_applicable']:>6} | {m['n_entered']:>7} | "
              f"{m['n_static_true']:>6} | {m['n_runtime_true']:>6} | "
              f"{m['n_agree']:>5} | {m['n_false_positives']:>3} | "
              f"{m['n_false_negatives']:>3}")


def _print_divergences(entry: dict, cap: int = 15) -> None:
    divs = entry["divergences"]
    # Show counts per (predicate, kind) — the pattern is more useful
    # than enumerating every row.
    from collections import defaultdict
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for d in divs:
        counts[(d["predicate"], d["kind"])] += 1
    print(f"\n  DIVERGENCES summary ({entry['corpus_name']}, "
          f"{len(divs)} total):")
    if not divs:
        print("    (none)")
        return
    for (p, k), n in sorted(counts.items()):
        print(f"    {k:>14} {p:>22}: {n}")
    print(f"  Witness samples (first {cap}):")
    for d in divs[:cap]:
        print(f"    [{d['kind']:>14}] {d['predicate']:>22} "
              f"@ {d['qname']}  witnesses={d['witnesses']}")
    if len(divs) > cap:
        print(f"    ... and {len(divs) - cap} more.")


# =============================================================================
# CLI
# =============================================================================


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--budget", type=float, default=0.08,
                    help="Per-call wall-time budget in seconds.")
    ap.add_argument("--predicates", nargs="*", default=None,
                    help="Subset of predicates (defaults to the Phase-28 8).")
    ap.add_argument("--out", default=None,
                    help="Output JSON path (optional).")
    ap.add_argument("--corpora", nargs="*", default=None,
                    help="Subset of corpora to run (by name).")
    ap.add_argument("--max-files", type=int, default=None,
                    help="Cap files per corpus (debug / CI time-bound).")
    args = ap.parse_args()

    predicates = tuple(args.predicates) if args.predicates \
        else PHASE28_PREDICATES

    print(f"Phase-28 multi-corpus runtime calibration — "
          f"seeds={args.seeds} budget={args.budget}s "
          f"predicates={predicates}", flush=True)

    corpora = _default_corpora()
    if args.corpora is not None:
        want = set(args.corpora)
        corpora = [c for c in corpora if c[0] in want]
    if not corpora:
        print("No runnable corpora matched. Exiting.")
        return 1

    results: list[dict] = []
    t_all = time.time()
    for name, root, pkg in corpora:
        if not os.path.isdir(root):
            print(f"[{name}] skipping — {root} not a directory", flush=True)
            continue
        entry = run_one_corpus(
            name, root, pkg,
            seeds=tuple(args.seeds), budget_s=args.budget,
            predicates=predicates,
            max_files=args.max_files,
            progress=lambda s: print(s, flush=True),
        )
        results.append(entry)

    # Pooled metrics across corpora.
    pooled: dict[str, dict] = {p: {"n_applicable": 0, "n_entered": 0,
                                    "n_agree": 0, "n_false_positives": 0,
                                    "n_false_negatives": 0,
                                    "n_static_true": 0, "n_runtime_true": 0}
                                 for p in predicates}
    pooled_cov = {
        "n_total": 0, "ready_no_args": 0, "ready_typed": 0,
        "ready_curated": 0, "n_probed": 0, "n_entered": 0,
        "n_calibrated": 0, "n_timeout": 0,
        "unsupported_method": 0, "unsupported_import": 0,
        "unsupported_missing": 0, "unsupported_untyped": 0,
        "unsupported_varargs": 0, "unsupported_async": 0,
        "unsupported_generator": 0,
    }
    pooled_analyzer: dict[str, int] = {p: 0 for p in predicates}
    for entry in results:
        c = entry["coverage"]
        for k in pooled_cov:
            pooled_cov[k] += c.get(k, 0)
        for p, m in entry["metrics_per_predicate"].items():
            if p not in pooled:
                continue
            for k in pooled[p]:
                pooled[p][k] += m.get(k, 0)
        for p, n in entry["analyzer_counts_per_predicate"].items():
            if p in pooled_analyzer:
                pooled_analyzer[p] += n
    for p in pooled:
        st = pooled[p]["n_static_true"]
        rt = pooled[p]["n_runtime_true"]
        pooled[p]["fp_rate"] = (
            round(pooled[p]["n_false_positives"] / st, 4) if st else None)
        pooled[p]["fn_rate"] = (
            round(pooled[p]["n_false_negatives"] / rt, 4) if rt else None)

    # Per-corpus report.
    print("\n" + "=" * 100)
    print("PHASE-28 PER-CORPUS REPORT")
    print("=" * 100)
    for entry in results:
        _print_coverage(entry)
        _print_metrics(entry)
        _print_divergences(entry)

    # Pooled report.
    n_total = pooled_cov["n_total"]
    n_ready = (pooled_cov["ready_no_args"] + pooled_cov["ready_typed"]
               + pooled_cov["ready_curated"])
    print("\n" + "=" * 100)
    print("PHASE-28 POOLED METRICS (across all runnable corpora)")
    print("=" * 100)
    print(f"  corpora_run:    {len(results)}")
    print(f"  n_total:        {n_total}")
    print(f"  ready:          {n_ready} "
          f"({(n_ready / max(1, n_total)) * 100:.1f}%)")
    print(f"  entered:        {pooled_cov['n_entered']} "
          f"({(pooled_cov['n_entered'] / max(1, n_total)) * 100:.1f}% "
          f"of corpus)")
    print(f"  timeouts:       {pooled_cov['n_timeout']}")
    head = (f"  {'predicate':>22} | {'applic':>6} | {'entered':>7} | "
            f"{'S_true':>6} | {'R_true':>6} | {'agree':>5} | "
            f"{'FP':>3} | {'FN':>3} | {'analyzer_ct':>11}")
    print(head)
    print("  " + "-" * (len(head) - 2))
    for p in predicates:
        m = pooled.get(p)
        if m is None:
            continue
        analyzer_ct = pooled_analyzer.get(p, 0)
        print(f"  {p:>22} | {m['n_applicable']:>6} | {m['n_entered']:>7} | "
              f"{m['n_static_true']:>6} | {m['n_runtime_true']:>6} | "
              f"{m['n_agree']:>5} | {m['n_false_positives']:>3} | "
              f"{m['n_false_negatives']:>3} | {analyzer_ct:>11}")

    # Phase-28 headline: explicit vs implicit raise on the pooled
    # entered slice.
    m_exp = pooled.get("may_raise_explicit")
    m_imp = pooled.get("may_raise_implicit")
    if m_exp and m_imp:
        print()
        print(f"  Phase-28 headline (pooled, entered only):")
        print(f"    may_raise_explicit:  "
              f"analyzer_true={m_exp['n_static_true']}  "
              f"runtime_true={m_exp['n_runtime_true']}  "
              f"agree={m_exp['n_agree']}/{m_exp['n_entered']}  "
              f"FP={m_exp['n_false_positives']}  "
              f"FN={m_exp['n_false_negatives']}")
        print(f"    may_raise_implicit:  "
              f"analyzer_true={m_imp['n_static_true']}  "
              f"runtime_true={m_imp['n_runtime_true']}  "
              f"agree={m_imp['n_agree']}/{m_imp['n_entered']}  "
              f"FP={m_imp['n_false_positives']}  "
              f"FN={m_imp['n_false_negatives']}")

    print(f"\n  TOTAL WALL-TIME: {round(time.time() - t_all, 2)}s")

    payload = {
        "config": {
            "seeds": list(args.seeds),
            "budget_s": args.budget,
            "predicates": list(predicates),
            "skip_files": list(DEFAULT_PHASE27_SKIP_FILES),
        },
        "corpora": results,
        "pooled": {
            "coverage": pooled_cov,
            "metrics_per_predicate": pooled,
            "analyzer_counts_per_predicate": pooled_analyzer,
        },
    }
    if args.out:
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
