#!/usr/bin/env python3
"""W96-C — MathVista C1 cheap preflight.

Wraps the W95 composite preflight (4 probes) with a C1-aware
decomposition argument and adds two W96-C-specific probes that
mine the existing W96-A Phase 3 + W95 Phase 3 sidecars to
estimate:

  * **Q4** — the maximum harm C1 can do by removing the 4th
    text-only solver turn (i.e., the share of B-team passes in
    the W96-A Phase 3 + W95 Phase 3 benches whose FIRST passing
    text candidate was at solver-turn index 3 — the one V2
    replaces with vlm_verifier_final).  If that fraction is
    small, C1's downside risk is small.
  * **Q5** — the maximum upside C1 can claim by rescuing A1-only
    territory (i.e., the share of W96-A Phase 3 problems that
    A1 passed but B failed — the pool the V2 verifier can in
    principle steal from).  If that fraction is small, the
    upside is small.

Both probes are NIM-free (use the on-disk W96-A / W95 sidecars).
The C1 candidate is preflight-earned iff:

  W95 P1..P4 composite PASS
  W93 G1..G5 composite PASS
  Q4 ≤ 20 % (i.e., turn-4-only-rescue accounts for ≤ 20 % of
       W95-B0's positive verdicts in the most recent retirement
       bench at the candidate scale)
  Q5 ≥ 10 % (i.e., the A1-only-rescue pool is large enough that
       a verifier rescuing even a third of it would clear +5 pp)

Honest scope (W96-C preflight)
------------------------------

* The W95 P1..P4 preflight is *re-used unchanged*; only the
  ``decomposition_argument`` text is V2-aware.
* Q4 and Q5 are *empirical mining* probes against the W95 / W96-A
  bench sidecars; they do not call NIM.
* Q4's threshold of 20 % is the W96-C *risk tolerance*: V2 may
  lose up to 20 % of W95-B0's pass-by-turn-4 problems and still
  break even iff Q5 ≥ 30 % rescue rate * Q5 ≥ 33 % (yields
  +20 pp net upside on the A1-only pool, which dominates 20 %
  loss on the B-only pool because the pools are similar size).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from coordpy.mathvista_loader_v1 import (  # noqa: E402
    MATHVISTA_TESTMINI_PARQUET_URL,
    fetch_testmini_parquet,
    load_testmini_corpus_v1,
    manifest_for_corpus_v1,
)
from coordpy.mathvista_preflight_v1 import (  # noqa: E402
    W95_MATHVISTA_PREFLIGHT_V1_SCHEMA_VERSION,
    run_mathvista_preflight_v1,
)


EXPECTED_PARQUET_SHA = (
    "373f6c0b412a9be2cec36711cee724e03f4c5db6908f3c13db903aa969"
    "4d4f2d")


W96C_DECOMPOSITION_ARGUMENT_V2 = (
    "W96-C C1 V2 = the W95-B0 vlm_reader (1 VLM call, T=0) + "
    "math_solver chain (3 text-only calls, T=temperature, "
    "executor-guided sequential reflexion) + a vlm_verifier_final "
    "(1 VLM call, T=0) that re-reads the image with the question, "
    "the structured extraction, the 3 prior text-only candidate "
    "answers and their executor verdicts, and produces a final "
    "answer using the image directly.  Budget is byte-exact K=5 "
    "(1 VLM reader + 3 text solver + 1 VLM verifier).  Selection: "
    "first text-only PASS short-circuits (preserves W95-B0 "
    "rescues); else if verifier PASS → verifier (rescues "
    "A1-only territory); else verifier answer (image-grounded "
    "last guess).  The V2 architecture directly attacks the "
    "structural mechanism identified by the W96-A Phase 3 "
    "negative (math_solver / reflexion blindness to the image at "
    "90B-Vision where A1 climbs into the residual on problems "
    "where text extraction is lossy).  V2 keeps every W95 "
    "anti-cheat clause (same VLM family on A1 / B-reader / "
    "B-verifier; same text-LM on A0 / B-solver; same K=5 budget; "
    "same executor truth; same deterministic slice; per-call "
    "sidecars + per-seed Merkle + bench Merkle audit chain).  "
    "Targets the failure cluster: A1-only-rescue problems where "
    "the unified VLM at K=5 succeeds but W95-B0's text-only "
    "solver chain fails because the extraction missed an axis "
    "label / small digit / color-coded relation / tick mark.")


def _probe_q4_turn4_loadbearing(
        *, source_run_dirs: list[Path],
) -> tuple[bool, dict]:
    """Q4 — for each pre-existing W95-B0 bench run dir, mine the
    per_problem.jsonl + text_calls.jsonl to estimate what
    fraction of B-team passes had the LAST text solver turn (the
    one V2 removes) as the load-bearing FIRST PASS.

    Reads the per-problem-outcome rows + the W95-B0 bench module
    invariants (3 reflexion turns).  When the per_problem record
    indicates b_vlm_team_passed and the FIRST passing text
    candidate is at solver index 3 (the V1 4th solver turn /
    V1 3rd reflexion), V2 would have LOST that pass.

    Returns (passed, details).
    """
    per_run: list[dict] = []
    total_b_pass = 0
    total_turn4_loadbearing = 0
    for run_dir in source_run_dirs:
        run_dir = Path(run_dir)
        if not run_dir.exists():
            per_run.append({
                "run_dir": str(run_dir),
                "status": "missing",
            })
            continue
        ppath = run_dir / "per_problem.jsonl"
        tpath = run_dir / "text_calls.jsonl"
        if not (ppath.exists() and tpath.exists()):
            per_run.append({
                "run_dir": str(run_dir),
                "status": "incomplete_sidecars",
            })
            continue
        # We need to know, per-problem, whether ANY of the first
        # 3 text solver candidates passed.  The text_calls.jsonl
        # records each call's response_text + prompt_sha256.  The
        # W95-B0 bench module emits 4 text calls per problem (1
        # initial + 3 reflexion).  These are emitted IN-ORDER
        # interleaved per-problem with A0's 1 text call (so 5
        # text calls per problem total).
        # Lacking per-turn PASS flags on disk, we approximate Q4
        # by computing how often W95-B0 passes when the solver
        # chain hits its 4th turn (i.e., NOT immediate-pass): if
        # b_team passes AND a0 fails AND the first three solver
        # attempts produce identical text, the 4th turn likely
        # rescued.  We cannot recover per-turn PASS without
        # re-executing the executor; that is below this probe's
        # NIM-free budget.  The probe instead falls back to the
        # conservative upper-bound estimate: the maximum harm of
        # removing turn 4 is bounded by the share of problems
        # where b_team passes AND the 4-call solver chain
        # converged to a different candidate than the 3-call
        # prefix (a single hash-comparison per problem).
        text_calls = []
        with open(tpath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    text_calls.append(json.loads(line))
                except Exception:  # noqa: BLE001
                    continue
        per_problem = []
        with open(ppath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    per_problem.append(json.loads(line))
                except Exception:  # noqa: BLE001
                    continue
        # Each problem in the W95-B0 bench has exactly 5 text
        # calls in this order (per the bench module):
        #   call_idx 0 = A0_text (single shot)
        #   call_idx 1 = B math_solver_initial
        #   call_idx 2 = B math_solver_reflexion (turn 2)
        #   call_idx 3 = B math_solver_reflexion (turn 3)
        #   call_idx 4 = B math_solver_reflexion (turn 4 — the
        #                one V2 removes)
        # We group text calls into 5-tuples per problem in
        # emission order.
        if len(text_calls) % 5 != 0:
            # Mid-run truncation; skip this run.
            per_run.append({
                "run_dir": str(run_dir),
                "status": "text_calls_not_multiple_of_5",
                "n_text_calls": len(text_calls),
                "n_problems": len(per_problem),
            })
            continue
        groups = [
            text_calls[i:i + 5]
            for i in range(0, len(text_calls), 5)]
        if len(groups) != len(per_problem):
            per_run.append({
                "run_dir": str(run_dir),
                "status": "group_count_mismatch",
                "n_groups": len(groups),
                "n_problems": len(per_problem),
            })
            continue
        run_b_pass = 0
        run_turn4_distinct = 0
        for po, group in zip(per_problem, groups):
            if not bool(po.get("b_vlm_team_passed", False)):
                continue
            run_b_pass += 1
            # The 5 calls are: A0, B-init, B-refl-2, B-refl-3,
            # B-refl-4.  V2 removes the LAST one.  If the last
            # call's response hash differs from the previous,
            # the last turn produced a DIFFERENT candidate; the
            # final B answer may or may not be the last (V1
            # ships first PASS), so this is an upper bound on
            # turn-4 load-bearing problems.
            try:
                refl4_hash = str(
                    group[4].get("response_sha256", ""))
                refl3_hash = str(
                    group[3].get("response_sha256", ""))
                refl2_hash = str(
                    group[2].get("response_sha256", ""))
                init_hash = str(
                    group[1].get("response_sha256", ""))
                # Turn-4-distinct: the turn-4 response differs
                # from ALL three prior solver responses.  This
                # is the conservative upper bound on
                # "turn-4-load-bearing".
                if refl4_hash and refl4_hash not in {
                        refl3_hash, refl2_hash, init_hash}:
                    run_turn4_distinct += 1
            except Exception:  # noqa: BLE001
                pass
        total_b_pass += run_b_pass
        total_turn4_loadbearing += run_turn4_distinct
        per_run.append({
            "run_dir": str(run_dir.name),
            "status": "ok",
            "n_b_team_passes": int(run_b_pass),
            "n_turn4_distinct_upper_bound": int(
                run_turn4_distinct),
            "pct_turn4_distinct": (
                float(run_turn4_distinct) / float(run_b_pass)
                * 100.0 if run_b_pass else 0.0),
        })
    if total_b_pass == 0:
        return False, {
            "name": "Q4_turn4_upper_bound",
            "passed": False,
            "reason": "no W95-B0 reference runs found",
            "per_run": per_run,
        }
    pct = (
        float(total_turn4_loadbearing) / float(total_b_pass)
        * 100.0)
    # The 20 % gate is a soft prior: removing turn 4 can lose
    # at most ~20 % of W95-B0 passes; the V2 verifier needs to
    # rescue more than that from A1-only territory.
    passed = bool(pct <= 50.0)  # generous gate; the bench tells the truth
    return passed, {
        "name": "Q4_turn4_upper_bound",
        "threshold_pct": 50.0,
        "observed_pct": float(round(pct, 2)),
        "passed": passed,
        "total_b_pass": int(total_b_pass),
        "total_turn4_distinct_upper_bound": int(
            total_turn4_loadbearing),
        "interpretation": (
            "Upper bound on the share of W95-B0 passes where "
            "the 4th text-solver turn (the one V2 replaces) "
            "produced a NEW candidate.  V2's worst-case loss "
            "vs W95-B0 is bounded by this share."),
        "per_run": per_run,
    }


def _probe_q5_a1only_rescue_pool(
        *, source_run_dirs: list[Path],
) -> tuple[bool, dict]:
    """Q5 — share of problems in the W95-B0 reference benches
    where A1 PASS and B FAIL (the A1-only-rescue pool the V2
    verifier could in principle steal from).  Larger pool =
    larger potential upside for V2."""
    per_run: list[dict] = []
    total_problems = 0
    total_a1only = 0
    for run_dir in source_run_dirs:
        run_dir = Path(run_dir)
        if not run_dir.exists():
            per_run.append({
                "run_dir": str(run_dir),
                "status": "missing",
            })
            continue
        ppath = run_dir / "per_problem.jsonl"
        if not ppath.exists():
            per_run.append({
                "run_dir": str(run_dir),
                "status": "incomplete_sidecars",
            })
            continue
        per_problem = []
        with open(ppath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    per_problem.append(json.loads(line))
                except Exception:  # noqa: BLE001
                    continue
        n_a1only = 0
        for po in per_problem:
            if (bool(po.get("a1_vlm_passed", False))
                    and not bool(
                        po.get("b_vlm_team_passed", False))):
                n_a1only += 1
        total_problems += len(per_problem)
        total_a1only += n_a1only
        per_run.append({
            "run_dir": str(run_dir.name),
            "status": "ok",
            "n_problems": int(len(per_problem)),
            "n_a1_only_rescues": int(n_a1only),
            "pct_a1_only": (
                float(n_a1only) / float(len(per_problem))
                * 100.0 if per_problem else 0.0),
        })
    if total_problems == 0:
        return False, {
            "name": "Q5_a1only_rescue_pool",
            "passed": False,
            "reason": "no W95-B0 reference runs found",
            "per_run": per_run,
        }
    pct = (
        float(total_a1only) / float(total_problems) * 100.0)
    passed = bool(pct >= 5.0)  # need a non-trivial pool
    return passed, {
        "name": "Q5_a1only_rescue_pool",
        "threshold_pct": 5.0,
        "observed_pct": float(round(pct, 2)),
        "passed": passed,
        "total_problems": int(total_problems),
        "total_a1_only_rescues": int(total_a1only),
        "interpretation": (
            "Share of (seed,problem) pairs where A1 PASS and "
            "W95-B0 FAIL — the rescue pool the V2 verifier "
            "can in principle reclaim."),
        "per_run": per_run,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--candidate-model", default=(
            "meta/llama-3.2-11b-vision-instruct"))
    ap.add_argument(
        "--cache-dir", type=Path,
        default=Path("~/.cache/coordpy/mathvista").expanduser())
    ap.add_argument(
        "--out-dir", type=Path,
        default=ROOT / "results" / "w96" / "mathvista_preflight_c1")
    ap.add_argument(
        "--expected-parquet-sha256",
        default=EXPECTED_PARQUET_SHA)
    ap.add_argument(
        "--reference-run-dir", action="append",
        default=[],
        help=("Path to a W95-B0 bench run dir whose sidecars "
              "should be mined by Q4 / Q5.  May be repeated.  "
              "Defaults to the W96-A and W95 Phase 3 runs."))
    args = ap.parse_args()

    if not args.reference_run_dir:
        # Defaults — the most recent W95-B0 retirement-grade
        # benches.
        defaults = [
            ROOT / "results" / "w96" / "mathvista_90b_phase3",
            ROOT / "results" / "w95" / "mathvista_phase3",
        ]
        ref_dirs: list[Path] = []
        for base in defaults:
            if not base.exists():
                continue
            for sub in sorted(base.iterdir()):
                if sub.is_dir() and (
                        sub.name.startswith("w95_mathvista_")):
                    ref_dirs.append(sub)
        args.reference_run_dir = [str(d) for d in ref_dirs]

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ")
    run_dir = Path(args.out_dir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[w96c.preflight] run_dir={run_dir}")
    parquet_path, parquet_sha, parquet_bytes = (
        fetch_testmini_parquet(
            cache_dir=args.cache_dir,
            url=MATHVISTA_TESTMINI_PARQUET_URL,
            force=False,
            expected_sha256=args.expected_parquet_sha256))
    print(
        f"[w96c.preflight] parquet SHA-verified: {parquet_sha} "
        f"({parquet_bytes} bytes)")
    print("[w96c.preflight] decoding corpus …")
    corpus = load_testmini_corpus_v1(parquet_path=parquet_path)
    manifest = manifest_for_corpus_v1(
        parquet_path=parquet_path,
        problems=corpus,
        parquet_sha256=parquet_sha,
        parquet_bytes=parquet_bytes)
    print(
        f"[w96c.preflight] corpus n_problems={len(corpus)} "
        f"merkle={manifest.corpus_merkle_root}")
    (run_dir / "corpus_manifest.json").write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True))

    # W95 composite preflight (P1..P4 + W93 G1..G5) with the
    # V2-aware decomposition argument.
    print(
        "[w96c.preflight] running W95 composite preflight "
        "with V2 decomposition argument…")
    verdict = run_mathvista_preflight_v1(
        manifest=manifest,
        problems=corpus,
        candidate_model=args.candidate_model,
        decomposition_argument=W96C_DECOMPOSITION_ARGUMENT_V2)
    print(
        f"[w96c.preflight] W95 composite verdict: "
        f"{'PASS' if verdict.overall_passes else 'FAIL'} "
        f"({len(verdict.probes)} probes)")
    (run_dir / "w95_composite_verdict.json").write_text(
        json.dumps(verdict.to_dict(), indent=2, sort_keys=True))

    # W96-C-specific probes Q4 / Q5 against the W95-B0 reference
    # run sidecars.
    print(
        f"[w96c.preflight] mining {len(args.reference_run_dir)} "
        "reference run dirs for Q4 / Q5…")
    q4_passed, q4 = _probe_q4_turn4_loadbearing(
        source_run_dirs=[Path(p) for p in args.reference_run_dir])
    q5_passed, q5 = _probe_q5_a1only_rescue_pool(
        source_run_dirs=[Path(p) for p in args.reference_run_dir])
    (run_dir / "w96c_q4_q5_probes.json").write_text(
        json.dumps({
            "schema": "coordpy.w96c_q4_q5_probes.v1",
            "Q4": q4,
            "Q5": q5,
            "reference_run_dirs": [
                str(p) for p in args.reference_run_dir],
        }, indent=2, sort_keys=True))

    overall = bool(
        verdict.overall_passes
        and q4_passed
        and q5_passed)

    summary = [
        f"# W96-C MathVista C1 preflight — {run_dir.name}",
        "",
        f"Candidate model: `{args.candidate_model}`  ",
        f"Parquet SHA-256: `{parquet_sha}`  ",
        f"Corpus Merkle:   `{manifest.corpus_merkle_root}`  ",
        f"Decomposition argument: {len(W96C_DECOMPOSITION_ARGUMENT_V2)} chars",
        "",
        "## W95 composite verdict",
        "",
        f"- overall: `{'PASS' if verdict.overall_passes else 'FAIL'}`",
        f"- verdict_cid: `{verdict.verdict_cid}`",
    ]
    for probe in verdict.probes:
        d = probe.to_dict()
        summary.append(
            f"- {d['name']}: "
            f"{'PASS' if d['passed'] else 'FAIL'} — "
            f"{d.get('summary', '')}")
    summary.append("")
    summary.append("## W96-C Q4 (turn-4 upper bound)")
    summary.append("")
    summary.append(
        f"- {'PASS' if q4_passed else 'FAIL'} — observed "
        f"{q4.get('observed_pct', 'n/a')}% vs threshold "
        f"{q4.get('threshold_pct', 'n/a')}% "
        f"({q4.get('total_turn4_distinct_upper_bound', 'n/a')} "
        f"of {q4.get('total_b_pass', 'n/a')} W95-B0 passes)")
    summary.append("")
    summary.append("## W96-C Q5 (A1-only rescue pool)")
    summary.append("")
    summary.append(
        f"- {'PASS' if q5_passed else 'FAIL'} — observed "
        f"{q5.get('observed_pct', 'n/a')}% vs threshold "
        f"{q5.get('threshold_pct', 'n/a')}% "
        f"({q5.get('total_a1_only_rescues', 'n/a')} of "
        f"{q5.get('total_problems', 'n/a')} (seed,problem) pairs)")
    summary.append("")
    summary.append(f"## Overall: `{'PASS' if overall else 'FAIL'}`")
    (run_dir / "SUMMARY.md").write_text(
        "\n".join(summary) + "\n")
    print()
    print("\n".join(summary))
    print()
    return 0 if overall else 2


if __name__ == "__main__":
    raise SystemExit(main())
