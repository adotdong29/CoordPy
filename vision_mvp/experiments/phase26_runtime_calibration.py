"""Phase 26 — Runtime-truth calibration of the conservative analyzer.

Phases 22–25 established analyzer-gold exactness (direct-exact at 100 %
by construction because the corpus gold is computed from the same
analyzer output the planner reads). Phase 26 measures a separate axis:
how the analyzer's conservative flags compare to RUNTIME-observed
truth on executable snippets.

Three-way comparison per snippet × predicate:

  1. **Analyzer prediction** — Phase-24/25 static flag.
  2. **Runtime-observed truth** — instrumented probe from
     `core/code_runtime_calibration`.
  3. **Direct-exact planner answer** — on a per-snippet ledger
     ingested via `CodeIndexer` + planner query. Same static flag
     the analyzer emits, surfaced through the normal query path.

The benchmark reports, per predicate:
  - n_applicable (snippets where the probe ran)
  - n_agree (static == runtime)
  - n_false_positives (static True, runtime False)
  - n_false_negatives (static False, runtime True — soundness break)
  - fp_rate, fn_rate
  - per-family breakdown

Repeat-run variance: `--seeds 0 1 2 3 4` drives the probe with five
independent fuzz seeds; observations are OR-ed (runtime_flag True iff
any seed observed the effect). Per-snippet `trigger_rate` is the
fraction of runs where the effect fired, exposing stochasticity.

Reproduce:

    python -m vision_mvp.experiments.phase26_runtime_calibration \\
        --seeds 0 1 2 --fuzz 8 \\
        --out vision_mvp/results_phase26_runtime.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import asdict

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")))

from vision_mvp.core.code_index import CodeIndexer
from vision_mvp.core.code_runtime_calibration import (
    RUNTIME_DECIDABLE_PREDICATES, calibrate_snippet,
    compute_static_flags_from_source, summarise_calibration,
)
from vision_mvp.core.context_ledger import ContextLedger, hash_embedding
from vision_mvp.core.code_planner import CodeQueryPlanner
from vision_mvp.tasks.executable_snippets import default_snippet_registry


# =============================================================================
# Planner-round-trip: verify the direct-exact path still answers
# "how many functions may raise" on a tiny ledger built from a single
# snippet source.
# =============================================================================


def _analyzer_count_for_snippet(source: str, predicate: str) -> int:
    """Return the number of functions in `source` that the Phase-
    24+25 analyzer flags for `predicate`. This is the corpus-wide
    count the planner should produce on a direct-exact count query.
    """
    import ast
    import textwrap
    from vision_mvp.core.code_interproc import (
        analyze_interproc, build_module_context)

    tree = ast.parse(textwrap.dedent(source))
    ctx, intra = build_module_context("snippet", tree)
    interproc, _ = analyze_interproc([ctx], intra)
    getter_map = {
        "may_raise": lambda s: s.trans_may_raise,
        "may_write_global": lambda s: s.trans_may_write_global,
        "calls_subprocess": lambda s: s.trans_calls_subprocess,
        "calls_filesystem": lambda s: s.trans_calls_filesystem,
        "calls_network": lambda s: s.trans_calls_network,
        "participates_in_cycle": lambda s: s.participates_in_cycle,
    }
    getter = getter_map.get(predicate)
    if getter is None:
        return 0
    return sum(1 for sem in interproc.values() if getter(sem))


def _planner_count_via_direct_exact(source: str, predicate: str,
                                     embed_dim: int = 32) -> int | None:
    """Ingest `source` into a fresh ledger, run the planner-matched
    count query for `predicate`, and return the numeric answer.
    Returns None when the planner didn't match the query.
    """
    # Write the snippet to a temp file and ingest via CodeIndexer.
    # Dedent first — snippet sources use triple-string indentation
    # that is fine for compile() / load_snippet_module but breaks
    # ast.parse when written verbatim to disk.
    import textwrap
    tmpdir = tempfile.mkdtemp(prefix="phase26_planner_")
    try:
        path = os.path.join(tmpdir, "snippet.py")
        with open(path, "w") as f:
            f.write(textwrap.dedent(source))
        ledger = ContextLedger(
            embed_dim=embed_dim,
            embed_fn=lambda t: hash_embedding(t, dim=embed_dim))
        indexer = CodeIndexer(root=tmpdir)
        indexer.index_into(ledger)
        planner = CodeQueryPlanner()
        # Map predicate → natural-language TRANS-count question so the
        # planner routes through the Phase-25 interprocedural pattern
        # (transitivity markers are the distinguishing phrasing; without
        # them the query would route to the Phase-24 intra pattern).
        q_map = {
            "may_raise":
                "How many functions may transitively raise an exception?",
            "may_write_global":
                "How many functions may transitively mutate module globals "
                "through a helper?",
            "calls_subprocess":
                "How many functions transitively invoke subprocess "
                "through a helper?",
            "calls_filesystem":
                "How many functions transitively touch the filesystem "
                "through a helper?",
            "calls_network":
                "How many functions transitively make network calls "
                "through a helper?",
            "participates_in_cycle":
                "How many functions participate in a recursion cycle?",
        }
        question = q_map.get(predicate)
        if question is None:
            return None
        result = planner.plan(question)
        if result is None or result.plan is None:
            return None
        # Execute the typed plan against the ledger.
        handles = ledger.all_handles()
        from vision_mvp.core.exact_ops import StageHandles
        trace: list = []
        out = StageHandles(handles=list(handles))
        for op in result.plan.ops:
            out = op.execute(ledger, out, trace)
        # Terminal op is Sum → StageScalar with `.value`.
        val = getattr(out, "value", None)
        if isinstance(val, (int, float)):
            return int(val)
        return None
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


# =============================================================================
# Per-snippet calibration driver
# =============================================================================


def run_snippets(seeds: tuple[int, ...], fuzz: int,
                 predicates: tuple[str, ...] | None = None,
                 progress=print) -> dict:
    """Run the full snippet calibration pass.

    Returns a JSON-serialisable dict with:
      - per-snippet observations (static / runtime / ground_truth)
      - per-predicate calibration summary
      - per-family per-predicate breakdown
      - per-predicate planner-direct-exact round-trip agreement
    """
    predicates = tuple(predicates or RUNTIME_DECIDABLE_PREDICATES)
    registry = default_snippet_registry()

    # Static flags + runtime observations per snippet.
    results = []
    for i, spec in enumerate(registry):
        spec_with_fuzz = spec
        if fuzz != spec.n_fuzz:
            # Re-issue with overridden fuzz count. Keep everything else.
            from dataclasses import replace
            spec_with_fuzz = replace(spec, n_fuzz=fuzz)
        static = compute_static_flags_from_source(
            spec.source, spec.target_qname)
        res = calibrate_snippet(
            spec_with_fuzz, predicates=predicates, seeds=seeds,
            static_flags=static)
        results.append(res)
        static_map = {p: static.get(p) for p in predicates}
        runtime_map = {
            p: res.runtime_observations[p].runtime_flag
            for p in predicates
        }
        progress(f"  [{i+1}/{len(registry)}] {spec.name:>32} "
                 f"family={spec.family:>9} "
                 f"static={static_map} runtime={runtime_map}")

    summary = summarise_calibration(results)

    # Per-family breakdown: same metrics as summary but keyed by family.
    family_index: dict[str, list] = {}
    for spec, res in zip(registry, results):
        family_index.setdefault(spec.family, []).append(res)
    per_family = {}
    for fam, rs in family_index.items():
        per_family[fam] = summarise_calibration(rs).per_predicate

    # Planner-vs-analyzer round-trip: for a small set of count queries,
    # check the direct-exact planner answer for each snippet equals
    # the analyzer's corpus-wide count (the number of functions in
    # the snippet that are flagged — not just the target function).
    planner_round_trip = {}
    for predicate in predicates:
        matches = 0
        n_checked = 0
        mismatches: list[dict] = []
        for spec in registry:
            expected = _analyzer_count_for_snippet(spec.source, predicate)
            planner_val = _planner_count_via_direct_exact(
                spec.source, predicate)
            if planner_val is None:
                continue
            n_checked += 1
            if planner_val == expected:
                matches += 1
            else:
                mismatches.append({
                    "snippet": spec.name, "expected": expected,
                    "planner_val": planner_val,
                })
        planner_round_trip[predicate] = {
            "n_checked": n_checked, "n_matches": matches,
            "match_rate": round(matches / max(1, n_checked), 4),
            "mismatches": mismatches,
        }

    payload = {
        "config": {"seeds": list(seeds), "fuzz": fuzz,
                   "predicates": list(predicates),
                   "n_snippets": len(registry)},
        "summary": summary.as_dict(),
        "per_family_calibration": per_family,
        "planner_direct_exact_roundtrip": planner_round_trip,
    }
    return payload


# =============================================================================
# Reporting
# =============================================================================


def _print_summary(payload: dict) -> None:
    cfg = payload["config"]
    print("\n" + "=" * 110)
    print("PHASE-26 RUNTIME CALIBRATION — PER-PREDICATE SUMMARY")
    print(f"seeds={cfg['seeds']} fuzz={cfg['fuzz']} "
          f"n_snippets={cfg['n_snippets']}")
    print("=" * 110)
    head = (f"{'predicate':>24} | {'applic':>6} | {'S_true':>6} | "
            f"{'R_true':>6} | {'agree':>5} | {'FP':>3} | {'FN':>3} | "
            f"{'fp_rate':>8} | {'fn_rate':>8}")
    print(head)
    print("-" * len(head))
    per = payload["summary"]["per_predicate"]
    for pred, m in sorted(per.items()):
        fp_r = f"{m['fp_rate']:.3f}" if m['fp_rate'] is not None else "   —   "
        fn_r = f"{m['fn_rate']:.3f}" if m['fn_rate'] is not None else "   —   "
        print(f"{pred:>24} | {m['n_applicable']:>6} | "
              f"{m['n_static_true']:>6} | {m['n_runtime_true']:>6} | "
              f"{m['n_agree']:>5} | {m['n_false_positives']:>3} | "
              f"{m['n_false_negatives']:>3} | {fp_r:>8} | {fn_r:>8}")
    print("-" * len(head))

    print("\nPER-FAMILY BREAKDOWN:")
    for fam, rows in payload["per_family_calibration"].items():
        print(f"\n  family={fam}")
        for pred, m in sorted(rows.items()):
            if m["n_applicable"] == 0:
                continue
            print(f"    {pred:>24}: applic={m['n_applicable']:>2} "
                  f"agree={m['n_agree']:>2}/{m['n_applicable']:<2} "
                  f"FP={m['n_false_positives']:>2} "
                  f"FN={m['n_false_negatives']:>2}")

    print("\nPLANNER DIRECT-EXACT ROUND-TRIP "
          "(planner count matches analyzer count per snippet):")
    for pred, info in payload["planner_direct_exact_roundtrip"].items():
        mr = info["match_rate"] * 100
        print(f"  {pred:>24}: {info['n_matches']}/"
              f"{info['n_checked']} ({mr:.1f}%) "
              f"mismatches={len(info['mismatches'])}")


def _print_divergences(payload: dict) -> None:
    print("\n" + "=" * 110)
    print("PER-SNIPPET DIVERGENCES (static ≠ runtime)")
    print("=" * 110)
    any_div = False
    for s in payload["summary"]["per_snippet"]:
        divs = s.get("divergences", [])
        if not divs:
            continue
        any_div = True
        print(f"  {s['snippet_name']:>36}  "
              f"family={s.get('target_qname',''):>12}  "
              f"divergences={divs}")
    if not any_div:
        print("  (none)")


# =============================================================================
# CLI
# =============================================================================


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--fuzz", type=int, default=6,
                    help="Per-seed fuzz-input count (ignored when a "
                         "snippet ships an explicit invoke).")
    ap.add_argument("--predicates", nargs="*", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print(f"Phase-26 runtime calibration — seeds={args.seeds} "
          f"fuzz={args.fuzz}", flush=True)

    predicates = tuple(args.predicates) if args.predicates else None
    payload = run_snippets(
        seeds=tuple(args.seeds), fuzz=args.fuzz, predicates=predicates)

    _print_summary(payload)
    _print_divergences(payload)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"\nWrote {args.out}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
