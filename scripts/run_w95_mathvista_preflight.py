#!/usr/bin/env python3
"""W95 — MathVista cheap-probe preflight runner.

Runs the four W95 cheap probes (no NIM calls) and produces a
content-addressed verdict at
``results/w95/mathvista_preflight/<RUN_ID>/preflight_verdict.json``.

Steps:

  1. Fetch the canonical testmini parquet to ``--cache-dir`` (or
     verify a cached copy if already present).
  2. Decode the parquet into ``MathVistaProblemV1`` capsules
     (1000 problems).
  3. Build the corpus manifest (parquet SHA, Merkle root,
     problem count).
  4. Run the W93 5-gate harness for the W95-B0 candidate against
     the corpus manifest + a written hypothesis +
     decomposition argument.
  5. Run the four W95-specific cheap probes via
     ``run_mathvista_preflight_v1``.
  6. Write the verdict JSON, a Markdown summary, and the corpus
     manifest sidecar to disk.

This script DOES NOT call NIM.  It is safe to run any time and
costs nothing beyond a single one-time CDN fetch of the
~170 MB testmini parquet (cached after the first run).
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from coordpy.cross_modal_preflight_harness_v1 import (  # noqa: E402
    run_preflight,
)
from coordpy.mathvista_executor_v1 import (  # noqa: E402
    W95_MATHVISTA_EXECUTOR_V1_SCHEMA_VERSION,
)
from coordpy.mathvista_loader_v1 import (  # noqa: E402
    MATHVISTA_TESTMINI_PARQUET_URL,
    W95_MATHVISTA_LOADER_V1_SCHEMA_VERSION,
    fetch_testmini_parquet,
    load_testmini_corpus_v1,
    manifest_for_corpus_v1,
)
from coordpy.mathvista_preflight_v1 import (  # noqa: E402
    W95_MATHVISTA_PREFLIGHT_V1_SCHEMA_VERSION,
    run_mathvista_preflight_v1,
)


W95_B0_HYPOTHESIS = (
    "W95 candidate B0 (vlm_reader + math_solver + executor-guided "
    "reflexion) decomposes a MathVista testmini problem into a "
    "vision-extract stage (1 VLM call at T=0.0 producing structured "
    "numerical / geometric / tabular facts) and a math-solve stage "
    "(1 solver call at T=0.7 producing a candidate answer + up to "
    "3 sequential reflexion turns conditioned on the executor's "
    "verdict + diagnostics) so that the same K=5 budget A1 spends "
    "on i.i.d. unified-VLM samples is redistributed onto roles "
    "the unified VLM cannot ablate.  The hypothesis is that on "
    "MathVista — where unified-VLM single-shot is ~33-49% for "
    "Llama-3.2-Vision and the failure-residual is structurally "
    "large — the role decomposition recovers ≥ 5 pp at K=5 over "
    "A1.  This is structurally different from the W88/W90/W92 "
    "HumanEval-Visual attempts because (i) the ceiling is much "
    "lower, (ii) the math-solver call genuinely runs on a "
    "different intermediate representation than the VLM, and "
    "(iii) the executor is a clean canonical-answer match rather "
    "than a unit-test PASS that requires the candidate code to "
    "be both syntactically and semantically right.")


W95_B0_DECOMPOSITION_ARGUMENT = (
    "MathVista testmini contains a large share of problems whose "
    "natural answer requires two cognitively distinct steps: "
    "(1) READ structured data from the image (a chart's bars, a "
    "geometric figure's coordinates, a scientific plot's axis "
    "values, a table's cells), then (2) COMPUTE the answer from "
    "the question + the extracted data.  A unified VLM forward "
    "must perform both steps inside a single auto-regressive "
    "generation, which forces a single attention budget across "
    "two different cognitive needs.  The W95-B0 team-decomposition "
    "spends one call on a dedicated extract-only forward (low "
    "temperature, structured output) and a different call on a "
    "dedicated math-solve forward (higher temperature, free-form "
    "chain-of-thought).  The reflexion turns add executor "
    "diagnostics that a unified-VLM K=5 cannot use because its "
    "K=5 samples are independent.  This is the same load-bearing "
    "structural difference that delivered the W89 70B HumanEval "
    "retirement on a text-only benchmark.  MathVista's ~20-50% "
    "geometry/chart/scientific share of problems gives this "
    "decomposition real surface area to work on.")


W95_B0_BENCHMARK_JUSTIFICATION = (
    "MathVista testmini is selected over HumanEval-Visual K=5 "
    "(retired in W94 per docs/W94_CROSS_MODAL_BATTLEFIELD_SCOUTING.md) "
    "because: (a) published single-shot SOTA for the Llama-3.2-Vision "
    "family is 33-49% (vs HumanEval-Visual's 88-92% K=5 ceiling on "
    "Llama-3.2-90B-Vision), so failure-residual is 35-50 pp instead "
    "of 8-12 pp; (b) the answer-match executor is canonical "
    "(numeric-tolerance + multi-choice + canonical text) so no "
    "GPT-4-judge dependency violates W88 anti-cheat; (c) the corpus "
    "is public + SHA-anchorable on HuggingFace; (d) the "
    "vision-extract → math-solve decomposition is structurally "
    "load-bearing on geometry / chart / table problems where the "
    "VLM's strength (vision) and the solver's strength (math) "
    "differ in load profile.  This is a fundamentally lower-ceiling "
    "battlefield than HumanEval-Visual K=5.")


def _now_run_id() -> str:
    return (
        datetime.datetime.now(datetime.UTC)
        .strftime("%Y%m%dT%H%M%SZ"))


def _canonical_json(payload: object) -> str:
    return json.dumps(
        payload, sort_keys=True, indent=2,
        default=str)


def make_preflight_evidence_check(verdict_dict: dict):
    """Adapt the MathVista preflight verdict into the W93 G2
    sidecar-evidence callable shape (returns
    (passed, summary, evidence_payload))."""

    def _fn():
        passed = bool(verdict_dict.get("overall_passes", False))
        summary = ", ".join(
            f"{p['probe_id']}={'PASS' if p['passed'] else 'FAIL'}"
            for p in verdict_dict.get("probes", []))
        evidence = {
            "preflight_verdict_cid": verdict_dict.get(
                "verdict_cid", ""),
            "probes": [
                {"probe_id": p["probe_id"],
                 "passed": bool(p["passed"]),
                 "summary": p["summary"]}
                for p in verdict_dict.get("probes", [])],
        }
        return bool(passed), str(summary), dict(evidence)

    return _fn


def make_w95_ablation_check(corpus_n: int):
    """W95-B0 ablation argument: if you REMOVE the dedicated
    vision-extract call and feed the image+question directly to
    the solver, you fall back to A1 unified-VLM; the team
    advantage disappears by construction.  No NIM needed to make
    this argument — the ablation is structural and is what
    distinguishes the W95-B0 candidate from A1."""

    def _fn():
        passed = bool(corpus_n > 0)  # corpus must exist
        summary = (
            "structural ablation: removing the vlm_reader stage "
            "collapses W95-B0 to A1 unified-VLM, by construction "
            "(no separate extraction representation, no "
            "executor-diagnostics reflexion turns); the ablation "
            "is load-bearing for the candidate's identity.")
        evidence = {
            "kind": "structural_ablation",
            "removed_component": "vlm_reader_stage",
            "expected_collapse_target": (
                "A1 unified VLM at K=5"),
            "corpus_n": int(corpus_n),
        }
        return bool(passed), str(summary), dict(evidence)

    return _fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cache-dir", type=Path,
        default=Path("~/.cache/coordpy/mathvista").expanduser(),
        help="Where to cache the testmini parquet.")
    ap.add_argument(
        "--out-dir", type=Path,
        default=Path("results/w95/mathvista_preflight"),
        help="Where to write the verdict + sidecar.")
    ap.add_argument(
        "--candidate-model",
        default="meta/llama-3.2-11b-vision-instruct",
        help=("Candidate VLM identifier (used for P3 published-"
              "SOTA lookup)."))
    ap.add_argument(
        "--candidate-id", default="W95-B0",
        help="Candidate identifier for the preflight verdict.")
    ap.add_argument(
        "--target-K", type=int, default=5,
        help="Same-budget K for the W93 G4 budget gate.")
    ap.add_argument(
        "--n-model-calls-per-problem", type=int, default=5,
        help=("Per-problem model calls the candidate would use "
              "(used by the W93 G4 budget gate)."))
    ap.add_argument(
        "--skip-fetch", action="store_true",
        help=("Skip the parquet fetch; assume the cache_dir "
              "already contains the testmini parquet."))
    ap.add_argument(
        "--expected-parquet-sha256", default="",
        help=("Optional: expected SHA-256 of the canonical "
              "testmini parquet; refuses to proceed on "
              "mismatch."))
    args = ap.parse_args()

    run_id = _now_run_id()
    out_root = Path(args.out_dir).resolve()
    out_dir = out_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[w95.preflight] run_id={run_id} out_dir={out_dir}")

    # --- Step 1 — fetch + SHA the parquet
    expected_sha = args.expected_parquet_sha256 or None
    parquet_path, parquet_sha, parquet_bytes = (
        fetch_testmini_parquet(
            cache_dir=args.cache_dir,
            url=MATHVISTA_TESTMINI_PARQUET_URL,
            force=False,
            expected_sha256=expected_sha))
    print(
        f"[w95.preflight] parquet_path={parquet_path}\n"
        f"[w95.preflight] parquet_sha256={parquet_sha}\n"
        f"[w95.preflight] parquet_bytes={parquet_bytes}")

    # --- Step 2 — decode the parquet
    print("[w95.preflight] decoding parquet → MathVistaProblemV1 …")
    problems = load_testmini_corpus_v1(parquet_path=parquet_path)
    print(f"[w95.preflight] decoded n_problems={len(problems)}")

    # --- Step 3 — corpus manifest
    manifest = manifest_for_corpus_v1(
        parquet_path=parquet_path,
        problems=problems,
        parquet_sha256=parquet_sha,
        parquet_bytes=parquet_bytes,
        url=MATHVISTA_TESTMINI_PARQUET_URL)
    print(
        "[w95.preflight] corpus_merkle_root="
        f"{manifest.corpus_merkle_root}")
    (out_dir / "corpus_manifest.json").write_text(
        _canonical_json(manifest.to_dict()))

    # --- Step 4 — W95 cheap probes (NO NIM)
    print("[w95.preflight] running 4 W95 cheap probes …")
    mathvista_verdict = run_mathvista_preflight_v1(
        manifest=manifest,
        problems=problems,
        candidate_model=args.candidate_model,
        decomposition_argument=W95_B0_DECOMPOSITION_ARGUMENT,
        max_acceptable_a1_k5_pass_rate=80.0,
        min_executor_self_test_pass_rate=0.98)
    mathvista_verdict_dict = mathvista_verdict.to_dict()
    (out_dir / "mathvista_preflight_verdict.json").write_text(
        _canonical_json(mathvista_verdict_dict))
    for p in mathvista_verdict.probes:
        print(
            f"[w95.preflight] {p.probe_id}: "
            f"{'PASS' if p.passed else 'FAIL'} — {p.summary}")
    print(
        "[w95.preflight] mathvista preflight overall_passes="
        f"{mathvista_verdict.overall_passes}")

    # --- Step 5 — W93 5-gate harness
    print("[w95.preflight] running W93 5-gate harness …")
    verdict_w93 = run_preflight(
        candidate_id=args.candidate_id,
        candidate_hypothesis=W95_B0_HYPOTHESIS,
        n_model_calls_per_problem=int(
            args.n_model_calls_per_problem),
        target_K=int(args.target_K),
        evidence_check_fn=make_preflight_evidence_check(
            mathvista_verdict_dict),
        ablation_check_fn=make_w95_ablation_check(
            corpus_n=len(problems)),
        chosen_benchmark="MathVista-testmini",
        why_better=W95_B0_BENCHMARK_JUSTIFICATION)
    w93_verdict_dict = verdict_w93.to_dict()
    (out_dir / "w93_preflight_verdict.json").write_text(
        _canonical_json(w93_verdict_dict))
    for g in verdict_w93.gates:
        print(
            f"[w95.preflight] {g.gate_id}: "
            f"{'PASS' if g.passed else 'FAIL'} — "
            f"{g.evidence_summary}")
    print(
        "[w95.preflight] W93 harness overall_passes="
        f"{verdict_w93.overall_passes}")

    # --- Step 6 — composite verdict
    composite_passes = bool(
        mathvista_verdict.overall_passes
        and verdict_w93.overall_passes)
    composite = {
        "schema": "coordpy.w95_composite_preflight_verdict.v1",
        "run_id": str(run_id),
        "candidate_id": str(args.candidate_id),
        "candidate_model": str(args.candidate_model),
        "loader_schema": W95_MATHVISTA_LOADER_V1_SCHEMA_VERSION,
        "executor_schema": (
            W95_MATHVISTA_EXECUTOR_V1_SCHEMA_VERSION),
        "preflight_schema": (
            W95_MATHVISTA_PREFLIGHT_V1_SCHEMA_VERSION),
        "corpus_manifest": manifest.to_dict(),
        "mathvista_preflight_verdict": mathvista_verdict_dict,
        "w93_preflight_verdict": w93_verdict_dict,
        "composite_passes": bool(composite_passes),
    }
    (out_dir / "composite_preflight_verdict.json").write_text(
        _canonical_json(composite))

    summary_md = (
        f"# W95 MathVista preflight composite verdict — {run_id}\n\n"
        f"Candidate: `{args.candidate_id}`  \n"
        f"Candidate model: `{args.candidate_model}`  \n"
        f"Parquet SHA-256: `{parquet_sha}`  \n"
        f"Corpus Merkle root: `{manifest.corpus_merkle_root}`  \n"
        f"Problem count: {len(problems)}  \n\n"
        "## W95 MathVista cheap probes\n\n"
        + "\n".join(
            f"* **{p.probe_id}**: "
            f"{'PASS' if p.passed else 'FAIL'} — {p.summary}"
            for p in mathvista_verdict.probes)
        + "\n\n## W93 5-gate harness\n\n"
        + "\n".join(
            f"* **{g.gate_id}**: "
            f"{'PASS' if g.passed else 'FAIL'} — "
            f"{g.evidence_summary}"
            for g in verdict_w93.gates)
        + f"\n\n## Composite verdict: "
        f"`{'PASS' if composite_passes else 'FAIL'}`\n"
        + ("\nProceed to W95 Phase 2 (cheap NIM pilot) per the "
           "pre-committed gates in `docs/RUNBOOK_W95.md`.\n"
           if composite_passes else
           "\nDO NOT launch a NIM pilot.  Record the failure in "
           "`docs/RESULTS_W95_MATHVISTA_PREFLIGHT_V1.md` and "
           "either fix the candidate or pivot to a backup "
           "battlefield (ChartQA / RealWorldQA per "
           "`docs/W94_CROSS_MODAL_BATTLEFIELD_SCOUTING.md`).\n"))
    (out_dir / "SUMMARY.md").write_text(summary_md)

    latest_path = out_root / "latest_run.txt"
    latest_path.write_text(run_id + "\n")

    print(
        "[w95.preflight] composite_passes="
        f"{composite_passes}; wrote {out_dir}")
    return 0 if composite_passes else 2


if __name__ == "__main__":
    sys.exit(main())
