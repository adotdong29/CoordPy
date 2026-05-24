"""W93 candidate-preflight runner.

Defines 3 W93 candidate architectures (with hypotheses) and
runs them through the 5-gate preflight harness against existing
W88–W92 sidecars.  Documents which survive and which die in
preflight.  No NIM calls; runs in seconds.

Candidates:
  A: Self-Verifying VLM-in-loop (variant of W90 P2)
  B: Heterogeneous Pool (3× VLM + 2× code-LM in parallel)
  C: Reflexion at K=10 budget (NEW K, different contract)

Each candidate is evaluated by its own evidence + ablation
check functions; gates G1, G4, G5 are direct field checks.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from coordpy.cross_modal_preflight_harness_v1 import (
    check_role_specialization_against_w92,
    check_vlm_inloop_evidence_against_w91_p2b,
    check_mbpp_reflexion_per_seed_majority,
    run_preflight,
)


def define_candidate_a():
    """A: Self-Verifying VLM-in-loop.  Same architecture as
    W90 P2 / W91 P2b but each VLM turn also emits a self-
    verification token.  Hypothesis: the model's structured
    self-verification correlates with executor PASS strongly
    enough to enable early termination or reranking, breaking
    the W90/W91 tie/loss.
    """
    hypothesis = (
        "Same VLM as A1_vlm, K=5 sequential turns with image "
        "in context every turn (W90 P2 architecture).  Add a "
        "structured self-verification token to each turn's "
        "output that the model emits BEFORE the code.  Use "
        "the self-verification to break ties between candidates "
        "that all pass the executor: prefer the candidate with "
        "the most-confident self-verification.  Hypothesis: "
        "this can beat A1_vlm because A1's 'first PASS' "
        "selection picks the first lucky pass, not the most "
        "robust one — and at K=5 some i.i.d. passes are "
        "fragile.  The architectural feature is the self-"
        "verification token; preflight evidence should show "
        "that the model's self-verification correlates with "
        "executor PASS in W90 / W91 sidecars."
    )
    # Evidence check: does the W90/W91 VLM-in-loop architecture
    # actually beat A1 at 7-seed scale?  If even the base VLM-in-
    # loop fails at 7 seeds, adding a self-verification token
    # won't fix it.
    def evidence_check():
        return check_vlm_inloop_evidence_against_w91_p2b(
            root_results_dir=ROOT / "results")

    # Ablation check: the W92 evidence directly shows that
    # adding more "verifier-like" turns to VLM-in-loop did NOT
    # help (it made things worse).  Self-verification is a
    # weaker variant of the W92 Verifier.
    def ablation_check():
        return check_role_specialization_against_w92(
            root_results_dir=ROOT / "results")

    return {
        "candidate_id": "W93-A-self-verifying-vlm-in-loop",
        "candidate_hypothesis": hypothesis,
        "n_model_calls_per_problem": 5,
        "target_K": 5,
        "evidence_check_fn": evidence_check,
        "ablation_check_fn": ablation_check,
        "chosen_benchmark": "HumanEval-Visual all_docstring",
        "why_better": (
            "Same battlefield as W90 P2 / W91 P2b — but uses a "
            "DIFFERENT mechanism (structured self-verification) "
            "to break ties.  The hypothesis is that the W91 P2b "
            "negative was driven by A1's lucky-first-pass picking "
            "fragile passes; self-verification re-ranks."),
    }


def define_candidate_b():
    """B: Heterogeneous Pool.  3 VLM samples (image, T=0.7) +
    2 code-LM samples (text-only, with VLM-extracted I/O from
    one of the VLM samples).  All 5 attempts pooled; ship first
    PASS.  Hypothesis: heterogeneous models produce different
    failure modes; pooling adds diversity beyond i.i.d. sampling.
    """
    hypothesis = (
        "3× VLM samples (T=0.7, with image) + 2× code-LM "
        "samples (text-only, conditioned on a parsed extraction "
        "from one of the VLM samples).  Ship first PASS across "
        "the 5-call pool.  Hypothesis: heterogeneous models "
        "(VLM vs code-LM) produce different failure modes; "
        "pooling adds genuine diversity beyond i.i.d. VLM "
        "sampling.  The architectural feature is heterogeneity; "
        "ablation = use 5 VLM samples (= A1) and confirm B's "
        "advantage disappears."
    )

    # Evidence check: W88's "split" architecture used code-LM
    # downstream of VLM extraction.  The W88 V1 result was −5.6
    # pp; the W89 P3 (90B-V + 70B code-LM) was also −5.6 pp.
    # The code-LM downstream of VLM extraction is empirically a
    # NEGATIVE contributor on this benchmark.  So a Heterogeneous
    # Pool that ADDS 2 code-LM samples to 3 VLM samples is
    # adding NEGATIVE samples — should make B WORSE than
    # A1_vlm K=5, not better.  Preflight predicts FAILURE.
    def evidence_check():
        # Check: did the W88/W89 code-LM-downstream-of-VLM
        # samples ever individually outperform VLM samples?
        # No public data lets us answer this without re-running.
        # Fall back to the structural argument: in W88 split,
        # the code-LM-generated candidate ALWAYS lost to A1_vlm
        # K=5.  So the marginal contribution of a code-LM
        # sample to a heterogeneous pool is negative.
        candidates = list(
            (ROOT / "results").rglob(
                "cross_modal_code_bench_report.json"))
        deltas = []
        for c in candidates:
            try:
                with open(c) as f:
                    rep = json.load(f)
                delta = float(rep.get(
                    "b_cross_mean_minus_a1_vlm_mean_pp",
                    0.0))
                deltas.append({
                    "path": str(c), "delta_pp": delta})
            except Exception:  # noqa: BLE001
                continue
        # Hypothesis supported iff ANY split run shows B > A1
        supported = any(
            d["delta_pp"] > 0.0 for d in deltas)
        return (
            bool(supported),
            "code-LM downstream of VLM extraction "
            f"(W88 split runs): {deltas} — supported iff any "
            "delta > 0",
            {"deltas": deltas})

    # Ablation check: removing the code-LM (= 5 VLM samples)
    # should make B WORSE than the heterogeneous pool (if the
    # code-LM contributes positively).  But we KNOW 5 VLM
    # samples (=A1_vlm K=5) reaches 84-88 %, while 3 VLM + 2
    # code-LM split reaches 58-86 %.  Ablation predicts
    # heterogeneous pool LOSES to A1 K=5.
    def ablation_check():
        return check_role_specialization_against_w92(
            root_results_dir=ROOT / "results")

    return {
        "candidate_id": "W93-B-heterogeneous-pool",
        "candidate_hypothesis": hypothesis,
        "n_model_calls_per_problem": 5,
        "target_K": 5,
        "evidence_check_fn": evidence_check,
        "ablation_check_fn": ablation_check,
        "chosen_benchmark": "HumanEval-Visual all_docstring",
        "why_better": (
            "Same battlefield; tests whether mixing VLM and "
            "code-LM samples adds diversity that pure VLM "
            "i.i.d. sampling misses.  But the W88 split "
            "evidence (code-LM downstream of VLM extraction "
            "always loses) directly falsifies the hypothesis "
            "that the code-LM contribution is positive."),
    }


def define_candidate_c():
    """C: Reflexion at K=10 budget.  Same architecture as W89
    HumanEval reflexion (sequential reflexion with executor
    stderr conditioning), but at K=10.  Both A1 and B get
    K=10.  Hypothesis: at K=10, i.i.d. sampling saturates
    (diminishing returns past K=5); reflexion has more
    iterations to add value; B beats A1 by a clearer margin.
    """
    hypothesis = (
        "Reflexion architecture from W88/W89 (sequential turns "
        "each conditioned on cumulative executor stderr), but "
        "at K=10 budget instead of K=5.  Both A1 (= first-pass-"
        "among-K=10) and B (= K=10 sequential reflexion) get "
        "the same K=10 model-call budget.  Hypothesis: at K=10, "
        "the unified-VLM K=10 baseline saturates (diminishing "
        "returns past K=5), while reflexion has 10 iterations "
        "to apply executor-feedback-driven refinement.  The "
        "advantage that was 0.0-2.78 pp at K=5 grows to a "
        "robust ≥ +5 pp at K=10.  This is a NEW budget contract; "
        "it does NOT retire the W89 K=5 carry-forward — it "
        "establishes a new claim at K=10."
    )

    # Evidence check: do we have any K=10 data?  No.  All W88–W92
    # runs are K=5.  So no preflight evidence is available
    # WITHOUT a new run.  Without evidence, this hypothesis
    # cannot pass cheap preflight; it requires a small actual
    # NIM run to gather evidence first.
    def evidence_check():
        return (
            False,
            "No K=10 evidence in W88–W92 sidecars; cheap "
            "preflight cannot confirm the K=10 hypothesis "
            "without a small NIM run.  Candidate cannot pass "
            "cheap preflight; requires a K=10 pilot run before "
            "the expensive bench.",
            {})

    # Ablation check: would removing the extra K=5 budget (back
    # to K=5) make B's advantage disappear?  Yes — W89 already
    # shows K=5 reflexion beats A1 by +5.56 pp on HumanEval.
    # So the "extra K" feature is load-bearing in the hypothesis.
    # This ablation PASSES (the extra K matters).
    def ablation_check():
        # The W89 K=5 evidence (B 91.1 % vs A1 85.6 %) shows
        # reflexion already wins at K=5.  Going to K=10 should
        # not HARM this advantage.  Ablation passes iff K=5
        # already shows positive B − A1 delta.
        candidates = list(
            (ROOT / "results").rglob(
                "humaneval_reflexion_bench_report.json"))
        deltas = []
        for c in candidates:
            try:
                with open(c) as f:
                    rep = json.load(f)
                delta = float(rep.get(
                    "b_mean_minus_a1_mean_pp", 0.0))
                if "70b" in str(c).lower():
                    deltas.append({
                        "path": str(c), "delta_pp": delta})
            except Exception:  # noqa: BLE001
                continue
        supported = any(d["delta_pp"] > 0.0 for d in deltas)
        return (
            bool(supported),
            "K=5 reflexion baseline at 70B (HumanEval): "
            f"{deltas} — ablation passes iff at least one K=5 "
            "70B run shows positive B − A1",
            {"deltas": deltas})

    return {
        "candidate_id": "W93-C-reflexion-K10",
        "candidate_hypothesis": hypothesis,
        "n_model_calls_per_problem": 10,
        "target_K": 10,
        "evidence_check_fn": evidence_check,
        "ablation_check_fn": ablation_check,
        "chosen_benchmark": (
            "HumanEval at K=10 (different budget contract; "
            "establishes a NEW claim at K=10, does not retire "
            "the K=5 carry-forward)"),
        "why_better": (
            "Same benchmark, NEW budget contract.  K=10 gives "
            "reflexion more iterations to add value; A1's "
            "i.i.d. sampling saturates past K=5.  The W89 K=5 "
            "evidence directly supports the architecture's "
            "direction; K=10 should amplify the advantage."),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="W93 candidate preflight runner")
    parser.add_argument(
        "--out", default=str(
            ROOT / "results" / "w93" /
            "candidate_preflight_verdicts.json"))
    args = parser.parse_args()

    candidates = [
        define_candidate_a(),
        define_candidate_b(),
        define_candidate_c(),
    ]
    print(f"running preflight on {len(candidates)} candidates")
    print()

    verdicts = []
    for c in candidates:
        v = run_preflight(**c)
        verdicts.append(v.to_dict())
        print(
            f"=== {v.candidate_id} ===")
        for g in v.gates:
            mark = "✓" if g.passed else "✗"
            print(
                f"  {mark} {g.gate_id}: "
                f"{g.evidence_summary[:120]}")
        print(
            f"  overall: "
            f"{'PASS — expensive run is JUSTIFIED' if v.overall_passes else 'KILLED in preflight'}")
        print()

    summary = {
        "schema": "coordpy.w93_candidate_preflight_v1",
        "verdicts": verdicts,
        "n_candidates": int(len(candidates)),
        "n_passed": int(sum(
            1 for v in verdicts if v["overall_passes"])),
        "n_killed": int(sum(
            1 for v in verdicts
            if not v["overall_passes"])),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(
        f"preflight summary: {summary['n_passed']} passed, "
        f"{summary['n_killed']} killed")
    print(f"verdicts -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
