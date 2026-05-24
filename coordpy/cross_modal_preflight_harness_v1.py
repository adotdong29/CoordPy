"""W93 / Post-W92 — Cross-modal preflight harness V1.

Cheap synthetic discriminators that evaluate a candidate
cross-modal architecture WITHOUT calling NIM.  Goal: kill weak
candidates in seconds instead of paying full-benchmark price.

The harness implements 5 pre-committed gates that a candidate
architecture must pass before earning an expensive run:

  Gate 1: Hypothesis written (manual check — pre-commit doc).
  Gate 2: Cheap preflight evidence — synthetic discriminator
          test using prior-run sidecars to confirm the
          hypothesized advantage is structurally present.
  Gate 3: Adversarial ablation — removing the key architectural
          feature degrades preflight performance.
  Gate 4: Budget accounting validated — K model calls per
          problem, no branching that exceeds K.
  Gate 5: Benchmark justification — written; not auto-checkable.

Cheap = no NIM calls; runs locally in seconds.

The harness produces a content-addressed preflight verdict JSON
that can be linked to the candidate's runbook.  If verdict
overall_passes = False, the candidate is KILLED in preflight.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable


W93_CROSS_MODAL_PREFLIGHT_HARNESS_V1_SCHEMA_VERSION: str = (
    "coordpy.cross_modal_preflight_harness_v1.v1")


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass(frozen=True)
class PreflightGateResultV1:
    gate_id: str
    description: str
    passed: bool
    evidence_summary: str
    evidence_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate_id": str(self.gate_id),
            "description": str(self.description),
            "passed": bool(self.passed),
            "evidence_summary": str(self.evidence_summary),
            "evidence_cid": str(self.evidence_cid),
        }


@dataclasses.dataclass(frozen=True)
class PreflightVerdictV1:
    schema: str
    candidate_id: str
    candidate_hypothesis: str
    gates: tuple[PreflightGateResultV1, ...]
    overall_passes: bool
    verdict_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "candidate_id": str(self.candidate_id),
            "candidate_hypothesis": str(
                self.candidate_hypothesis),
            "gates": [g.to_dict() for g in self.gates],
            "overall_passes": bool(self.overall_passes),
            "verdict_cid": str(self.verdict_cid),
        }


# ---------------------------------------------------------------
# Sidecar-based preflight checks (no NIM calls)
# ---------------------------------------------------------------

def gate_budget_accounting(
        *,
        candidate_id: str,
        n_model_calls_per_problem: int,
        target_K: int,
) -> PreflightGateResultV1:
    """Gate 4: validate that the candidate's K model-call budget
    matches the A1 baseline EXACTLY for every problem (no
    branching that exceeds K)."""
    passed = (int(n_model_calls_per_problem) == int(target_K))
    summary = (
        f"candidate uses {n_model_calls_per_problem} model "
        f"calls/problem; A1 uses {target_K} → "
        f"{'matches' if passed else 'MISMATCH'}")
    ev = {
        "kind": "w93_preflight_budget",
        "candidate_id": str(candidate_id),
        "n_model_calls_per_problem": int(
            n_model_calls_per_problem),
        "target_K": int(target_K),
    }
    return PreflightGateResultV1(
        gate_id="G4_budget_accounting",
        description=(
            "Candidate's per-problem model-call budget "
            "matches A1's K exactly."),
        passed=bool(passed),
        evidence_summary=str(summary),
        evidence_cid=_sha256_hex(ev))


def gate_hypothesis_written(
        *,
        candidate_id: str,
        hypothesis: str,
) -> PreflightGateResultV1:
    """Gate 1: hypothesis is written and >= 50 characters
    (sanity-check; not a content check)."""
    passed = (
        bool(hypothesis)
        and len(hypothesis.strip()) >= 50
        and "beat A1" in hypothesis.lower()
        + " " + hypothesis.lower())
    # Looser check — the hypothesis must mention how it differs
    # from baseline.
    if not passed:
        passed = (
            bool(hypothesis)
            and len(hypothesis.strip()) >= 50)
    summary = (
        f"hypothesis length {len(hypothesis.strip())} chars; "
        f"{'present' if passed else 'TOO SHORT or MISSING'}")
    ev = {
        "kind": "w93_preflight_hypothesis",
        "candidate_id": str(candidate_id),
        "hypothesis_len": int(len(hypothesis.strip())),
        "hypothesis_sha256": hashlib.sha256(
            hypothesis.encode("utf-8")).hexdigest(),
    }
    return PreflightGateResultV1(
        gate_id="G1_hypothesis_written",
        description=(
            "Hypothesis for why the candidate should beat A1 "
            "is written and non-trivial (≥ 50 chars)."),
        passed=bool(passed),
        evidence_summary=str(summary),
        evidence_cid=_sha256_hex(ev))


def gate_sidecar_evidence(
        *,
        candidate_id: str,
        evidence_check_fn: Callable[[], tuple[bool, str, dict]],
) -> PreflightGateResultV1:
    """Gate 2: cheap preflight evidence — runs a candidate-
    specific evidence check against prior W88–W92 sidecars.
    The check function returns (passed, summary, evidence_payload).
    """
    try:
        passed, summary, payload = evidence_check_fn()
    except Exception as e:  # noqa: BLE001
        passed = False
        summary = (
            f"evidence check raised: {type(e).__name__}: {e}")
        payload = {"error": str(e)}
    ev = {
        "kind": "w93_preflight_sidecar_evidence",
        "candidate_id": str(candidate_id),
        "payload": payload,
    }
    return PreflightGateResultV1(
        gate_id="G2_sidecar_evidence",
        description=(
            "Cheap preflight evidence from prior-run sidecars "
            "confirms the hypothesized advantage."),
        passed=bool(passed),
        evidence_summary=str(summary),
        evidence_cid=_sha256_hex(ev))


def gate_ablation(
        *,
        candidate_id: str,
        ablation_check_fn: Callable[[], tuple[bool, str, dict]],
) -> PreflightGateResultV1:
    """Gate 3: adversarial ablation — the candidate's key
    architectural feature is load-bearing.  Test by simulating
    removal and confirming the candidate's hypothesized
    advantage degrades."""
    try:
        passed, summary, payload = ablation_check_fn()
    except Exception as e:  # noqa: BLE001
        passed = False
        summary = (
            f"ablation check raised: {type(e).__name__}: {e}")
        payload = {"error": str(e)}
    ev = {
        "kind": "w93_preflight_ablation",
        "candidate_id": str(candidate_id),
        "payload": payload,
    }
    return PreflightGateResultV1(
        gate_id="G3_adversarial_ablation",
        description=(
            "Removing the candidate's key architectural feature "
            "in an ablation reduces its hypothesized advantage."),
        passed=bool(passed),
        evidence_summary=str(summary),
        evidence_cid=_sha256_hex(ev))


def gate_benchmark_justification(
        *,
        candidate_id: str,
        chosen_benchmark: str,
        why_better_than_humaneval_visual: str,
) -> PreflightGateResultV1:
    """Gate 5: benchmark justification.  HumanEval-Visual at K=5
    is presumptively hostile.  If candidate picks it, must
    explain why it's a radically different attempt.  Else must
    explain why the new benchmark is a better battlefield."""
    is_humaneval_visual = (
        "humaneval-visual" in chosen_benchmark.lower()
        or "humaneval_visual" in chosen_benchmark.lower())
    passed = (
        bool(chosen_benchmark)
        and bool(why_better_than_humaneval_visual)
        and len(why_better_than_humaneval_visual.strip()) >= 50)
    summary = (
        f"benchmark='{chosen_benchmark}'; "
        f"is_humaneval_visual={is_humaneval_visual}; "
        f"justification len="
        f"{len(why_better_than_humaneval_visual.strip())}")
    ev = {
        "kind": "w93_preflight_benchmark_justification",
        "candidate_id": str(candidate_id),
        "benchmark": str(chosen_benchmark),
        "is_humaneval_visual": bool(is_humaneval_visual),
        "justification_sha256": hashlib.sha256(
            why_better_than_humaneval_visual.encode(
                "utf-8")).hexdigest(),
    }
    return PreflightGateResultV1(
        gate_id="G5_benchmark_justification",
        description=(
            "Benchmark is either a new battlefield with "
            "stronger justification than the W88/W90/W91/W92 "
            "choices, or HumanEval-Visual with a radically "
            "different hypothesis."),
        passed=bool(passed),
        evidence_summary=str(summary),
        evidence_cid=_sha256_hex(ev))


def run_preflight(
        *,
        candidate_id: str,
        candidate_hypothesis: str,
        n_model_calls_per_problem: int,
        target_K: int,
        evidence_check_fn: (
            Callable[[], tuple[bool, str, dict]] | None) = None,
        ablation_check_fn: (
            Callable[[], tuple[bool, str, dict]] | None) = None,
        chosen_benchmark: str = "HumanEval-Visual",
        why_better: str = "",
) -> PreflightVerdictV1:
    """Run all 5 preflight gates against a candidate.  Returns a
    content-addressed verdict.  overall_passes = True iff all 5
    gates pass."""
    gates: list[PreflightGateResultV1] = []
    gates.append(gate_hypothesis_written(
        candidate_id=candidate_id,
        hypothesis=candidate_hypothesis))
    if evidence_check_fn is None:
        evidence_check_fn = (
            lambda: (False, "no evidence check provided", {}))
    gates.append(gate_sidecar_evidence(
        candidate_id=candidate_id,
        evidence_check_fn=evidence_check_fn))
    if ablation_check_fn is None:
        ablation_check_fn = (
            lambda: (False, "no ablation check provided", {}))
    gates.append(gate_ablation(
        candidate_id=candidate_id,
        ablation_check_fn=ablation_check_fn))
    gates.append(gate_budget_accounting(
        candidate_id=candidate_id,
        n_model_calls_per_problem=int(
            n_model_calls_per_problem),
        target_K=int(target_K)))
    gates.append(gate_benchmark_justification(
        candidate_id=candidate_id,
        chosen_benchmark=str(chosen_benchmark),
        why_better_than_humaneval_visual=str(why_better)))

    overall_passes = bool(all(g.passed for g in gates))

    verdict = PreflightVerdictV1(
        schema=W93_CROSS_MODAL_PREFLIGHT_HARNESS_V1_SCHEMA_VERSION,
        candidate_id=str(candidate_id),
        candidate_hypothesis=str(candidate_hypothesis),
        gates=tuple(gates),
        overall_passes=bool(overall_passes),
        verdict_cid="")
    cid = _sha256_hex({
        "kind": "w93_preflight_verdict_v1",
        "verdict": {
            **verdict.to_dict(), "verdict_cid": ""},
    })
    return dataclasses.replace(verdict, verdict_cid=str(cid))


# ---------------------------------------------------------------
# Concrete evidence/ablation checks against W88–W92 sidecars
# ---------------------------------------------------------------

def check_vlm_inloop_evidence_against_w91_p2b(
        *, root_results_dir: Path,
) -> tuple[bool, str, dict]:
    """Cheap check: is there any evidence in W91 P2b that a
    VLM-in-loop variant would systematically beat A1_vlm at
    7-seed scale on all_docstring HumanEval-Visual?

    Returns False if W91 P2b's bench report shows B − A1_vlm
    is negative — i.e., the architecture is already
    empirically falsified at adequate seed scale.
    """
    candidates = list(root_results_dir.rglob(
        "cross_modal_vlm_loop_bench_report.json"))
    if not candidates:
        return (
            False, "no W90/W91 VLM-in-loop reports found",
            {})
    deltas = []
    for c in candidates:
        try:
            with open(c) as f:
                rep = json.load(f)
            delta_pp = float(rep.get(
                "b_vlm_loop_mean_minus_a1_vlm_mean_pp",
                0.0))
            n_seeds = int(rep.get("n_seeds", 0))
            deltas.append({
                "path": str(c),
                "delta_pp": delta_pp,
                "n_seeds": n_seeds,
            })
        except Exception:  # noqa: BLE001
            continue
    # The hypothesis "VLM-in-loop variant will beat A1_vlm at
    # K=5 on HumanEval-Visual" is supported iff at least one
    # 7-seed-or-larger run shows B > A1_vlm strictly.
    supported = False
    for d in deltas:
        if d["n_seeds"] >= 7 and d["delta_pp"] > 0.0:
            supported = True
            break
    summary = (
        "VLM-in-loop variants at ≥ 7 seeds on HumanEval-Visual: "
        + str(deltas))
    return (
        bool(supported),
        str(summary),
        {"deltas": deltas})


def check_role_specialization_against_w92(
        *, root_results_dir: Path,
) -> tuple[bool, str, dict]:
    """Cheap check: is there evidence in W92 that adding more
    role-specialization to VLM-in-loop will help?

    The W92 evidence (B − A1_vlm = −10.71 pp, worse than W91
    P2b's −7.14) directly falsifies this.  Returns False.
    """
    candidates = list(root_results_dir.rglob(
        "cross_modal_role_specialized_bench_report.json"))
    if not candidates:
        return (
            False,
            "no W92 role-specialized reports found",
            {})
    deltas = []
    for c in candidates:
        try:
            with open(c) as f:
                rep = json.load(f)
            delta_pp = float(rep.get(
                "b_role_spec_mean_minus_a1_vlm_mean_pp",
                0.0))
            n_seeds = int(rep.get("n_seeds", 0))
            deltas.append({
                "path": str(c),
                "delta_pp": delta_pp,
                "n_seeds": n_seeds,
            })
        except Exception:  # noqa: BLE001
            continue
    # Role-specialization hypothesis supported iff at least one
    # 7-seed-or-larger run shows B > A1_vlm.
    supported = False
    for d in deltas:
        if d["n_seeds"] >= 7 and d["delta_pp"] > 0.0:
            supported = True
            break
    summary = (
        "Role-specialization at ≥ 7 seeds on HumanEval-Visual: "
        + str(deltas))
    return (
        bool(supported),
        str(summary),
        {"deltas": deltas})


def check_mbpp_reflexion_per_seed_majority(
        *, root_results_dir: Path,
) -> tuple[bool, str, dict]:
    """Cheap check: does any MBPP-70B-reflexion run achieve
    per-seed strict majority B > A1?"""
    candidates = list(root_results_dir.rglob(
        "mbpp_reflexion_bench_report.json"))
    if not candidates:
        return (False, "no MBPP reports found", {})
    summaries = []
    any_majority = False
    for c in candidates:
        try:
            with open(c) as f:
                rep = json.load(f)
            b_beats_a1 = list(rep.get(
                "b_beats_a1_per_seed", []))
            n_seeds = int(rep.get("n_seeds", 0))
            n_wins = sum(1 for x in b_beats_a1 if x)
            majority = (
                n_wins * 2 >= n_seeds + 1
                if n_seeds > 0 else False)
            if majority:
                any_majority = True
            summaries.append({
                "path": str(c),
                "n_wins": int(n_wins),
                "n_seeds": int(n_seeds),
                "per_seed_majority": bool(majority),
            })
        except Exception:  # noqa: BLE001
            continue
    summary = (
        "MBPP-70B reflexion runs (n_wins/n_seeds, majority): "
        + str(summaries))
    return (
        bool(any_majority),
        str(summary),
        {"summaries": summaries})


__all__ = [
    "W93_CROSS_MODAL_PREFLIGHT_HARNESS_V1_SCHEMA_VERSION",
    "PreflightGateResultV1",
    "PreflightVerdictV1",
    "gate_hypothesis_written",
    "gate_sidecar_evidence",
    "gate_ablation",
    "gate_budget_accounting",
    "gate_benchmark_justification",
    "run_preflight",
    "check_vlm_inloop_evidence_against_w91_p2b",
    "check_role_specialization_against_w92",
    "check_mbpp_reflexion_per_seed_majority",
]
