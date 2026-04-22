"""Phase 27 — Corpus-aware runtime-truth calibration of the
conservative analyzer.

Phase 26 measured analyzer-vs-runtime agreement on a 21-snippet
curated corpus. Phase 27 extends runtime truth to **real functions
drawn from real Python corpora** — here, the ``vision_mvp`` repo
itself. See ``vision_mvp/RESULTS_PHASE27.md`` for the research
framing and theorem set.

Three-axis report, same shape as Phase 26 but now at corpus scale:

  1. **Analyzer prediction** — Phase-24/25 static interprocedural flag.
  2. **Runtime-observed truth** — instrumented probe from
     ``core/code_corpus_runtime`` with entry-detection + per-call
     wall-time budget.
  3. **Direct-exact planner answer** — unchanged from Phase 26; the
     substrate guarantee (planner count == analyzer count) is
     independently verified per corpus.

Phase 27 adds a FOURTH reported axis Phase 26 did not need:

  4. **Callable coverage** — of the N functions the analyzer declared
     flags for, how many were *runtime-calibratable*: ready-for-
     invocation under the current recipe derivation strategy + the
     curated ``SafeRecipeRegistry``. This is Theorem P27-1 in numbers.

Reproduce:

    python -m vision_mvp.experiments.phase27_corpus_runtime_calibration \\
        --seeds 0 1 2 --budget 0.1 \\
        --out vision_mvp/results_phase27_corpus.json

Repeat-run variance:

    python -m vision_mvp.experiments.phase27_corpus_runtime_calibration \\
        --seeds 0 1 2 3 4 --budget 0.15 \\
        --out vision_mvp/results_phase27_corpus_5seeds.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict

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
# Corpus catalog — local-only, deterministic
# =============================================================================


# Each entry is (name, filesystem_root, importable_package_name).
# The importable-package name is what ``importlib.import_module`` is fed;
# we prefer it over file-based imports so any relative imports resolve
# correctly. Setting it to None falls back to spec_from_file_location.
def _default_corpora() -> list[tuple[str, str, str]]:
    repo = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", ".."))
    return [
        ("vision-core",        os.path.join(repo, "vision_mvp/core"),
         "vision_mvp.core"),
        ("vision-tasks",       os.path.join(repo, "vision_mvp/tasks"),
         "vision_mvp.tasks"),
    ]


# =============================================================================
# Planner round-trip — mirrors Phase 26 § D.3 but on a real corpus
# =============================================================================


def _planner_roundtrip_per_corpus(corpus_root: str,
                                   corpus_package: str | None,
                                   predicates: tuple[str, ...]) -> dict:
    """For each predicate, ingest the corpus via CodeIndexer (Phase 23/24/25
    path), run the matching trans-count planner query, compare with the
    analyzer's corpus-wide count. Returns a per-predicate match record.
    """
    import tempfile
    import shutil
    from vision_mvp.core.code_index import CodeIndexer
    from vision_mvp.core.context_ledger import ContextLedger, hash_embedding
    from vision_mvp.core.code_planner import CodeQueryPlanner
    from vision_mvp.core.exact_ops import StageHandles

    # Static counts from the analyzer (this is the gold the planner
    # should match). We re-ingest through `CodeIndexer` and read the
    # n_functions_trans_* aggregates from each handle's metadata.
    ledger = ContextLedger(
        embed_dim=16, embed_fn=lambda t: hash_embedding(t, dim=16))
    indexer = CodeIndexer(root=corpus_root)
    indexer.index_into(ledger)
    handles = list(ledger.all_handles())

    # Corpus-wide analyzer counts per predicate (sum over files).
    agg_field_by_pred = {
        "may_raise": "n_functions_trans_may_raise",
        "may_write_global": "n_functions_trans_may_write_global",
        "calls_subprocess": "n_functions_trans_calls_subprocess",
        "calls_filesystem": "n_functions_trans_calls_filesystem",
        "calls_network": "n_functions_trans_calls_network",
        "participates_in_cycle": "n_functions_participates_in_cycle",
    }

    q_map = {
        "may_raise":
            "How many functions may transitively raise an exception?",
        "may_write_global":
            "How many functions may transitively mutate module globals "
            "through a helper?",
        "calls_subprocess":
            "How many functions transitively invoke subprocess through a helper?",
        "calls_filesystem":
            "How many functions transitively touch the filesystem through a helper?",
        "calls_network":
            "How many functions transitively make network calls through a helper?",
        "participates_in_cycle":
            "How many functions participate in a recursion cycle?",
    }

    out: dict[str, dict] = {}
    planner = CodeQueryPlanner()
    for pred in predicates:
        field = agg_field_by_pred.get(pred)
        if field is None:
            continue
        expected = 0
        for h in handles:
            md = h.metadata_dict()
            expected += int(md.get(field, 0))
        question = q_map.get(pred)
        if question is None:
            continue
        result = planner.plan(question)
        if result is None or result.plan is None:
            out[pred] = {"expected": expected, "planner_val": None,
                         "matched": False}
            continue
        trace: list = []
        stage = StageHandles(handles=list(handles))
        for op in result.plan.ops:
            stage = op.execute(ledger, stage, trace)
        planner_val = getattr(stage, "value", None)
        planner_val = int(planner_val) if isinstance(
            planner_val, (int, float)) else None
        out[pred] = {
            "expected": expected, "planner_val": planner_val,
            "matched": planner_val == expected,
        }
    return out


# =============================================================================
# Run one corpus
# =============================================================================


def run_one_corpus(corpus_name: str, corpus_root: str,
                    corpus_package: str | None,
                    seeds: tuple[int, ...], budget_s: float,
                    predicates: tuple[str, ...],
                    progress=print) -> dict:
    progress(f"[{corpus_name}] corpus root: {corpus_root}")
    reg = build_default_recipe_registry()
    t0 = time.time()
    rows, cov = calibrate_corpus(
        corpus_name, corpus_root,
        corpus_package=corpus_package, recipe_registry=reg,
        predicates=predicates, seeds=seeds, budget_s=budget_s,
        skip_files=DEFAULT_PHASE27_SKIP_FILES,
        progress=progress,
    )
    metrics = summarise_corpus_calibration(rows, predicates=predicates)
    divergences = collect_divergences(rows, predicates=predicates)
    planner_rt = _planner_roundtrip_per_corpus(
        corpus_root, corpus_package, predicates)
    elapsed = time.time() - t0
    return {
        "corpus_name": corpus_name,
        "corpus_root": corpus_root,
        "corpus_package": corpus_package,
        "elapsed_s": round(elapsed, 2),
        "coverage": cov.as_dict(),
        "metrics_per_predicate": {p: m.as_dict() for p, m in metrics.items()},
        "divergences": [
            {"qname": d.qname, "predicate": d.predicate, "kind": d.kind,
             "static_flag": d.static_flag, "runtime_flag": d.runtime_flag,
             "witnesses": list(d.witnesses),
             "callable_status": d.callable_status,
             "module_name": d.module_name}
            for d in divergences
        ],
        "planner_direct_exact_roundtrip": planner_rt,
        "rows": rows_as_dict_list(rows),
    }


# =============================================================================
# Pretty-print
# =============================================================================


def _print_coverage(entry: dict) -> None:
    c = entry["coverage"]
    print(f"\n--- {entry['corpus_name']} ({entry['corpus_root']}) ---")
    print(f"  elapsed: {entry['elapsed_s']}s")
    print(f"  n_total functions analysed:        {c['n_total']}")
    print(f"  ready_no_args + ready_typed + ready_curated: "
          f"{c['ready_no_args']} + {c['ready_typed']} + "
          f"{c['ready_curated']} = "
          f"{c['ready_no_args'] + c['ready_typed'] + c['ready_curated']}  "
          f"({c['ready_fraction']*100:.1f}%)")
    print(f"  probed / entered / calibrated:     "
          f"{c['n_probed']} / {c['n_entered']} / {c['n_calibrated']}  "
          f"({c['calibrated_fraction']*100:.1f}% of total)")
    print(f"  timeout                            {c['n_timeout']}")
    print(f"  unsupported_* breakdown:")
    for k in ("unsupported_varargs", "unsupported_untyped",
              "unsupported_async", "unsupported_generator",
              "unsupported_method", "unsupported_import",
              "unsupported_missing"):
        if c.get(k, 0):
            print(f"    {k:>28}: {c[k]}")


def _print_metrics(entry: dict) -> None:
    print(f"\n  PER-PREDICATE METRICS ({entry['corpus_name']})")
    head = (f"    {'predicate':>22} | {'applic':>6} | {'entered':>7} | "
            f"{'S_true':>6} | {'R_true':>6} | {'agree':>5} | "
            f"{'FP':>3} | {'FN':>3}")
    print(head)
    print("    " + "-" * (len(head) - 4))
    for p, m in sorted(entry["metrics_per_predicate"].items()):
        print(f"    {p:>22} | {m['n_applicable']:>6} | {m['n_entered']:>7} | "
              f"{m['n_static_true']:>6} | {m['n_runtime_true']:>6} | "
              f"{m['n_agree']:>5} | {m['n_false_positives']:>3} | "
              f"{m['n_false_negatives']:>3}")


def _print_divergences(entry: dict) -> None:
    print(f"\n  DIVERGENCES (analyzer ≠ runtime) for {entry['corpus_name']}")
    divs = entry["divergences"]
    if not divs:
        print("    (none)")
        return
    for d in divs[:30]:
        print(f"    [{d['kind']:>14}] {d['predicate']:>22} "
              f"@ {d['qname']}  witnesses={d['witnesses']}")
    if len(divs) > 30:
        print(f"    ... and {len(divs) - 30} more.")


def _print_planner_roundtrip(entry: dict) -> None:
    print(f"\n  PLANNER DIRECT-EXACT ROUND-TRIP "
          f"({entry['corpus_name']}):")
    rt = entry["planner_direct_exact_roundtrip"]
    for p, info in sorted(rt.items()):
        ok = "OK" if info.get("matched") else "MISMATCH"
        print(f"    {p:>22}: expected={info.get('expected')}  "
              f"planner={info.get('planner_val')}  [{ok}]")


# =============================================================================
# CLI
# =============================================================================


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--budget", type=float, default=0.1,
                    help="Per-call wall-time budget in seconds.")
    ap.add_argument("--predicates", nargs="*", default=None)
    ap.add_argument("--out", default=None,
                    help="Output JSON path (optional).")
    ap.add_argument("--corpora", nargs="*", default=None,
                    help="Subset of corpora to run (by name).")
    ap.add_argument("--max-files", type=int, default=None,
                    help="Cap files per corpus (debug).")
    args = ap.parse_args()

    print(f"Phase-27 corpus-aware runtime calibration — "
          f"seeds={args.seeds} budget={args.budget}s", flush=True)

    predicates = tuple(args.predicates) if args.predicates else tuple(
        sorted(RUNTIME_DECIDABLE_PREDICATES))

    corpora = _default_corpora()
    if args.corpora is not None:
        want = set(args.corpora)
        corpora = [c for c in corpora if c[0] in want]

    results: list[dict] = []
    for name, root, pkg in corpora:
        if not os.path.isdir(root):
            print(f"[{name}] skipping — {root} not a directory", flush=True)
            continue
        entry = run_one_corpus(
            name, root, pkg,
            seeds=tuple(args.seeds), budget_s=args.budget,
            predicates=predicates,
            progress=lambda s: print(s, flush=True),
        )
        results.append(entry)

    # Aggregate across corpora: pooled metrics give the single-number
    # Phase-27 headline.
    pooled: dict[str, dict] = {p: {"n_applicable": 0, "n_entered": 0,
                                    "n_agree": 0, "n_false_positives": 0,
                                    "n_false_negatives": 0,
                                    "n_static_true": 0, "n_runtime_true": 0}
                                 for p in predicates}
    total_cov = {
        "n_total": 0, "ready_no_args": 0, "ready_typed": 0,
        "ready_curated": 0, "n_probed": 0, "n_entered": 0,
        "n_calibrated": 0, "n_timeout": 0,
    }
    for entry in results:
        c = entry["coverage"]
        for k in total_cov:
            total_cov[k] += c.get(k, 0)
        for p, m in entry["metrics_per_predicate"].items():
            if p not in pooled:
                continue
            for k in pooled[p]:
                pooled[p][k] += m.get(k, 0)
    for p in pooled:
        st = pooled[p]["n_static_true"]
        rt = pooled[p]["n_runtime_true"]
        pooled[p]["fp_rate"] = (
            round(pooled[p]["n_false_positives"] / st, 4) if st else None)
        pooled[p]["fn_rate"] = (
            round(pooled[p]["n_false_negatives"] / rt, 4) if rt else None)

    # Print report.
    print("\n" + "=" * 100)
    print("PHASE-27 CORPUS-AWARE RUNTIME CALIBRATION — PER-CORPUS REPORT")
    print("=" * 100)
    for entry in results:
        _print_coverage(entry)
        _print_metrics(entry)
        _print_divergences(entry)
        _print_planner_roundtrip(entry)

    print("\n" + "=" * 100)
    print("PHASE-27 POOLED METRICS (across all corpora)")
    print("=" * 100)
    print(f"  TOTAL COVERAGE: ready = "
          f"{total_cov['ready_no_args'] + total_cov['ready_typed'] + total_cov['ready_curated']}"
          f" / {total_cov['n_total']} "
          f"({(total_cov['ready_no_args'] + total_cov['ready_typed'] + total_cov['ready_curated']) / max(1, total_cov['n_total']) * 100:.1f}%)")
    print(f"  TOTAL ENTERED: {total_cov['n_entered']} "
          f"({total_cov['n_entered'] / max(1, total_cov['n_total']) * 100:.1f}% of functions)")
    print(f"  TOTAL TIMEOUTS: {total_cov['n_timeout']}")
    head = (f"  {'predicate':>22} | {'applic':>6} | {'entered':>7} | "
            f"{'S_true':>6} | {'R_true':>6} | {'agree':>5} | "
            f"{'FP':>3} | {'FN':>3}")
    print(head)
    print("  " + "-" * (len(head) - 2))
    for p, m in sorted(pooled.items()):
        fpr = f"{m['fp_rate']:.3f}" if m['fp_rate'] is not None else "   —   "
        fnr = f"{m['fn_rate']:.3f}" if m['fn_rate'] is not None else "   —   "
        print(f"  {p:>22} | {m['n_applicable']:>6} | {m['n_entered']:>7} | "
              f"{m['n_static_true']:>6} | {m['n_runtime_true']:>6} | "
              f"{m['n_agree']:>5} | {m['n_false_positives']:>3} | "
              f"{m['n_false_negatives']:>3}")

    payload = {
        "config": {
            "seeds": list(args.seeds),
            "budget_s": args.budget,
            "predicates": list(predicates),
        },
        "corpora": results,
        "pooled": {
            "coverage": total_cov,
            "metrics_per_predicate": pooled,
        },
    }

    if args.out:
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"\nWrote {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
