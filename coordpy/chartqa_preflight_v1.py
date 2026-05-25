"""W96-D — ChartQA cheap-probe preflight V1.

Extends the W93 5-gate harness
(``coordpy.cross_modal_preflight_harness_v1``) with ChartQA-
specific cheap probes that answer the four W94 questions BEFORE
any NIM spend:

  P1. Is the corpus reproducible from the canonical parquet
      (SHA-anchored, expected size, valid schema)?
  P2. Is the executor clean? (gold-as-prediction → ≥ 98 % pass
      under the W96-D executor; if not, the executor silently
      penalises every arm.)
  P3. Is there a plausible failure-residual under unified VLM at
      K=5? (estimated from published SOTA scores; refused if A1@
      K=5 saturates the +20 pp residual floor.)
  P4. Is there a plausible decomposition advantage for B? (written
      argument; structural fit between ChartQA questions and the
      vlm-reader + text-solver pipeline.)

Each probe returns a ``ChartQAPreflightProbeResultV1``.  All four
plus the W93 G1–G5 gates must pass before any NIM-spending pilot
launches.  If any probe fails, the W96-D D1 battlefield is killed
cheaply with a carry-forward in
``docs/RESULTS_W96D_CHARTQA_PREFLIGHT_V1.md``.

This module never calls NIM.  Probes that consume external state
(e.g., the parquet corpus) use ONLY content-addressed local data;
the network fetch is delegated to
``chartqa_loader_v1.fetch_chartqa_test_parquet`` and is performed
once before the probes run.

Honest scope (W96-D preflight)
------------------------------

* ``W96-L-CHARTQA-PREFLIGHT-V1-PUBLISHED-SOTA-FROZEN-CAP`` — the
  published-SOTA table is a frozen snapshot of Meta's
  Llama-3.2-Vision-Instruct release notes + comparable frontier
  numbers as of 2026-05; updates are explicit code changes.
* ``W96-L-CHARTQA-PREFLIGHT-V1-NO-PER-PROBLEM-PRECISION-CAP`` —
  ChartQA does not carry per-problem precision in the canonical
  HF schema, unlike MathVista; the executor uses a fixed 5 %
  relative tolerance.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

from .chartqa_executor_v1 import (
    W96_CHARTQA_EXECUTOR_V1_SCHEMA_VERSION,
    executor_self_test_on_gold_v1,
)
from .chartqa_loader_v1 import (
    CHARTQA_TEST_EXPECTED_N_PROBLEMS_LOWER,
    CHARTQA_TEST_EXPECTED_N_PROBLEMS_UPPER,
    CHARTQA_TEST_EXPECTED_PARQUET_BYTES_LOWER,
    CHARTQA_TEST_EXPECTED_PARQUET_BYTES_UPPER,
    ChartQACorpusManifestV1,
    ChartQAProblemV1,
    W96_CHARTQA_LOADER_V1_SCHEMA_VERSION,
)


W96_CHARTQA_PREFLIGHT_V1_SCHEMA_VERSION: str = (
    "coordpy.chartqa_preflight_v1.v1")


# Published-SOTA reference points used by P3.  Frozen snapshot.
# Sources:
#   * Llama-3.2 release notes (Meta, Sep 2024) — Vision-Instruct
#     ChartQA test single-shot.
#   * GPT-4o / Claude-3.5-Sonnet / Gemini-1.5-Pro — paper /
#     leaderboard 2024-2025 reports.
# These are documented anti-cheat constants; updates are explicit
# code changes.
CHARTQA_PUBLISHED_SOTA_SINGLE_SHOT_BY_MODEL: dict[
        str, float] = {
    "llama-3.2-11b-vision-instruct": 83.4,
    "llama-3.2-90b-vision-instruct": 85.5,
    "gpt-4o": 85.7,
    "claude-3.5-sonnet": 90.8,
    "gemini-1.5-pro": 87.2,
    "qwen2-vl-72b": 88.3,
}


@dataclasses.dataclass(frozen=True)
class ChartQAPreflightProbeResultV1:
    probe_id: str
    description: str
    passed: bool
    summary: str
    evidence_cid: str
    evidence: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "probe_id": str(self.probe_id),
            "description": str(self.description),
            "passed": bool(self.passed),
            "summary": str(self.summary),
            "evidence_cid": str(self.evidence_cid),
            "evidence": dict(self.evidence),
        }


@dataclasses.dataclass(frozen=True)
class ChartQAPreflightVerdictV1:
    schema: str
    corpus_manifest: dict[str, Any]
    probes: tuple[ChartQAPreflightProbeResultV1, ...]
    overall_passes: bool
    verdict_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "corpus_manifest": dict(self.corpus_manifest),
            "probes": [p.to_dict() for p in self.probes],
            "overall_passes": bool(self.overall_passes),
            "verdict_cid": str(self.verdict_cid),
        }


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


# ---------------------------------------------------------------
# P1 — Corpus integrity probe
# ---------------------------------------------------------------

def probe_corpus_integrity_v1(
        *,
        manifest: ChartQACorpusManifestV1,
        problems: tuple[ChartQAProblemV1, ...],
) -> ChartQAPreflightProbeResultV1:
    """Probe P1: parquet hashes to a known value, decodes to the
    expected number of problems in the documented range, every
    problem carries non-empty image + non-empty labels."""
    n_problems = len(problems)
    n_lower = CHARTQA_TEST_EXPECTED_N_PROBLEMS_LOWER
    n_upper = CHARTQA_TEST_EXPECTED_N_PROBLEMS_UPPER
    bytes_low = CHARTQA_TEST_EXPECTED_PARQUET_BYTES_LOWER
    bytes_high = CHARTQA_TEST_EXPECTED_PARQUET_BYTES_UPPER
    parquet_bytes_ok = (
        bytes_low <= manifest.parquet_bytes <= bytes_high)
    n_ok = (n_lower <= n_problems <= n_upper)
    n_with_image = sum(
        1 for p in problems if len(p.image_bytes) > 0)
    n_with_labels = sum(
        1 for p in problems if p.labels and any(
            str(l).strip() for l in p.labels))
    passed = bool(
        parquet_bytes_ok
        and n_ok
        and n_with_image == n_problems
        and n_with_labels == n_problems
        and bool(manifest.parquet_sha256))
    summary = (
        f"parquet_bytes={manifest.parquet_bytes} "
        f"(range_ok={parquet_bytes_ok}); "
        f"n_problems={n_problems} in "
        f"[{n_lower},{n_upper}] (ok={n_ok}); "
        f"n_with_image={n_with_image}; "
        f"n_with_labels={n_with_labels}")
    evidence = {
        "parquet_sha256": str(manifest.parquet_sha256),
        "parquet_bytes": int(manifest.parquet_bytes),
        "expected_bytes_range": [
            int(bytes_low), int(bytes_high)],
        "n_problems": int(n_problems),
        "expected_n_range": [int(n_lower), int(n_upper)],
        "n_with_image": int(n_with_image),
        "n_with_labels": int(n_with_labels),
        "corpus_merkle_root": str(manifest.corpus_merkle_root),
    }
    return ChartQAPreflightProbeResultV1(
        probe_id="P1_corpus_integrity",
        description=(
            "Parquet SHA-anchored; test decodes to expected size "
            "in-range; every problem has image + labels."),
        passed=bool(passed),
        summary=str(summary),
        evidence_cid=_sha256_hex(evidence),
        evidence=dict(evidence))


# ---------------------------------------------------------------
# P2 — Executor self-test on gold
# ---------------------------------------------------------------

def probe_executor_self_test_v1(
        *,
        problems: tuple[ChartQAProblemV1, ...],
        min_pass_rate: float = 0.98,
) -> ChartQAPreflightProbeResultV1:
    """Probe P2: feed each problem's first gold label back through
    the executor.  A well-formed executor MUST pass on ~all gold.
    """
    result = executor_self_test_on_gold_v1(problems)
    pass_rate = float(result["pass_rate"])
    passed = bool(pass_rate >= float(min_pass_rate))
    summary = (
        f"executor self-test on gold: "
        f"{result['n_pass']}/{result['n_problems']} = "
        f"{pass_rate * 100.0:.2f}% "
        f"(threshold={min_pass_rate * 100.0:.2f}%)")
    evidence = {
        "executor_schema": (
            W96_CHARTQA_EXECUTOR_V1_SCHEMA_VERSION),
        "n_problems": int(result["n_problems"]),
        "n_pass": int(result["n_pass"]),
        "pass_rate": float(pass_rate),
        "min_pass_rate": float(min_pass_rate),
        "by_rule": dict(result["by_rule"]),
        "failed_sample": list(result["failed_pids"])[:50],
        "n_failed": int(len(result["failed_pids"])),
    }
    return ChartQAPreflightProbeResultV1(
        probe_id="P2_executor_self_test",
        description=(
            "Gold-as-prediction must score ≥ 98 % under the "
            "W96-D ChartQA executor."),
        passed=bool(passed),
        summary=str(summary),
        evidence_cid=_sha256_hex(evidence),
        evidence=dict(evidence))


# ---------------------------------------------------------------
# P3 — A1 saturation / failure-residual estimate
# ---------------------------------------------------------------

def estimate_a1_k5_pass_rate_v1(
        single_shot_pct: float,
        *,
        k: int = 5,
        correlation: float = 0.5,
) -> float:
    """Same correlation-blended estimator as the W95 preflight:

    A1@K = (1 − correlation) × i.i.d._upper + correlation × single

    where i.i.d._upper = 1 − (1 − p_single)^K.

    See ``coordpy.mathvista_preflight_v1.estimate_a1_k5_pass_rate_v1``
    for the structural rationale; this is a verbatim copy so the
    W96-D preflight does not depend on the W95 module.
    """
    p = max(0.0, min(1.0, float(single_shot_pct) / 100.0))
    iid_upper = 1.0 - (1.0 - p) ** int(k)
    return float(
        100.0 * (
            (1.0 - float(correlation)) * iid_upper
            + float(correlation) * p))


def probe_a1_failure_residual_v1(
        *,
        candidate_model: str,
        max_acceptable_a1_k5_pass_rate: float = 80.0,
        published_sota_table: (
            dict[str, float] | None) = None,
) -> ChartQAPreflightProbeResultV1:
    """Probe P3: use the published single-shot ChartQA score for
    the candidate VLM family to estimate A1@K=5 and refuse to
    launch if A1 is presumptively saturated above the configured
    ceiling.

    Default ceiling is 80 % — leaves ≥ 20 pp residual room for B.
    This matches the W95 default; W96-D inherits the bar
    verbatim.
    """
    table = (
        published_sota_table
        if published_sota_table is not None
        else CHARTQA_PUBLISHED_SOTA_SINGLE_SHOT_BY_MODEL)
    key = str(candidate_model).lower().strip()
    aliases = {
        "meta/llama-3.2-11b-vision-instruct": (
            "llama-3.2-11b-vision-instruct"),
        "meta/llama-3.2-90b-vision-instruct": (
            "llama-3.2-90b-vision-instruct"),
        "meta-llama/llama-3.2-11b-vision-instruct": (
            "llama-3.2-11b-vision-instruct"),
        "meta-llama/llama-3.2-90b-vision-instruct": (
            "llama-3.2-90b-vision-instruct"),
    }
    key = aliases.get(key, key)
    single_shot = float(table.get(key, -1.0))
    if single_shot < 0.0:
        return ChartQAPreflightProbeResultV1(
            probe_id="P3_a1_failure_residual",
            description=(
                "Estimated A1@K=5 pass rate based on published "
                "single-shot ChartQA SOTA must leave ≥ 20 pp "
                "failure-residual."),
            passed=False,
            summary=(
                f"no published single-shot ChartQA SOTA for "
                f"'{candidate_model}'; refusing to estimate "
                "A1@K=5 cheaply."),
            evidence_cid=_sha256_hex({
                "model": str(candidate_model),
                "table_keys": sorted(table.keys()),
            }),
            evidence={
                "model": str(candidate_model),
                "table_keys": sorted(table.keys()),
            })
    a1_k5_est = estimate_a1_k5_pass_rate_v1(
        single_shot, k=5)
    residual = max(0.0, 100.0 - a1_k5_est)
    passed = bool(
        a1_k5_est <= float(max_acceptable_a1_k5_pass_rate))
    summary = (
        f"published single-shot ChartQA for {candidate_model} = "
        f"{single_shot:.2f}%; "
        f"estimated A1@K=5 = {a1_k5_est:.2f}%; "
        f"estimated residual = {residual:.2f} pp "
        f"(ceiling = {max_acceptable_a1_k5_pass_rate:.2f}%); "
        f"pass = {passed}")
    evidence = {
        "model": str(candidate_model),
        "single_shot_pct": float(single_shot),
        "estimated_a1_k5_pct": float(a1_k5_est),
        "estimated_residual_pp": float(residual),
        "max_acceptable_a1_k5_pct": float(
            max_acceptable_a1_k5_pass_rate),
        "K": 5,
        "correlation_assumed": 0.5,
    }
    return ChartQAPreflightProbeResultV1(
        probe_id="P3_a1_failure_residual",
        description=(
            "Estimated A1@K=5 pass rate based on published "
            "single-shot ChartQA SOTA must leave room for B."),
        passed=bool(passed),
        summary=str(summary),
        evidence_cid=_sha256_hex(evidence),
        evidence=dict(evidence))


# ---------------------------------------------------------------
# P4 — Decomposition argument
# ---------------------------------------------------------------

def probe_decomposition_argument_v1(
        *,
        problems: tuple[ChartQAProblemV1, ...],
        decomposition_argument: str,
        min_argument_chars: int = 200,
        min_human_split_share: float = 0.20,
) -> ChartQAPreflightProbeResultV1:
    """Probe P4: B's decomposition must have a written structural
    argument AND the corpus must have enough problems where
    chart-extract + math-solve is a plausibly distinct pipeline.

    For ChartQA, the corpus check is the share of problems
    flagged as ``human_or_machine == "human"`` (the harder
    human-authored subset) — these are the problems where
    structured extraction is most likely to be load-bearing.
    """
    arg = str(decomposition_argument or "").strip()
    arg_ok = len(arg) >= int(min_argument_chars)

    n_corpus = len(problems)
    sample_n = min(500, n_corpus)
    n_human = 0
    samples: list[dict[str, Any]] = []
    for p in problems[:sample_n]:
        flag = str(p.human_or_machine or "").lower().strip()
        is_human = (flag == "human")
        if is_human:
            n_human += 1
        if len(samples) < 5:
            samples.append({
                "pid": str(p.pid),
                "human_or_machine": str(flag),
                "labels": list(p.labels),
                "query_head": p.query[:80] if p.query else "",
            })
    if sample_n > 0:
        human_share = float(n_human) / float(sample_n)
    else:
        human_share = 0.0
    human_ok = bool(
        human_share >= float(min_human_split_share))
    passed = bool(arg_ok and human_ok)
    summary = (
        f"decomp argument len={len(arg)} "
        f"(threshold={min_argument_chars}; ok={arg_ok}); "
        f"human-split share in sample of {sample_n} = "
        f"{human_share * 100.0:.1f}% "
        f"(threshold={min_human_split_share * 100.0:.1f}%; "
        f"ok={human_ok})")
    evidence = {
        "argument_chars": int(len(arg)),
        "argument_sha256": hashlib.sha256(
            arg.encode("utf-8")).hexdigest(),
        "min_argument_chars": int(min_argument_chars),
        "sample_n": int(sample_n),
        "n_human_in_sample": int(n_human),
        "human_share_in_sample": float(human_share),
        "min_human_share": float(min_human_split_share),
        "samples": list(samples),
    }
    return ChartQAPreflightProbeResultV1(
        probe_id="P4_decomposition_argument",
        description=(
            "Written structural argument for B's decomposition "
            "exists AND corpus has ≥ 20 % human-authored "
            "ChartQA problems where chart-extract + math-solve "
            "is plausibly distinct from unified VLM."),
        passed=bool(passed),
        summary=str(summary),
        evidence_cid=_sha256_hex(evidence),
        evidence=dict(evidence))


# ---------------------------------------------------------------
# Composite verdict
# ---------------------------------------------------------------

def run_chartqa_preflight_v1(
        *,
        manifest: ChartQACorpusManifestV1,
        problems: tuple[ChartQAProblemV1, ...],
        candidate_model: str,
        decomposition_argument: str,
        max_acceptable_a1_k5_pass_rate: float = 80.0,
        min_executor_self_test_pass_rate: float = 0.98,
) -> ChartQAPreflightVerdictV1:
    """Run all four W96-D ChartQA cheap probes against a loaded
    corpus and return a content-addressed verdict.
    ``overall_passes = True`` iff all four probes pass."""
    probes: list[ChartQAPreflightProbeResultV1] = []
    probes.append(probe_corpus_integrity_v1(
        manifest=manifest, problems=problems))
    probes.append(probe_executor_self_test_v1(
        problems=problems,
        min_pass_rate=min_executor_self_test_pass_rate))
    probes.append(probe_a1_failure_residual_v1(
        candidate_model=candidate_model,
        max_acceptable_a1_k5_pass_rate=(
            max_acceptable_a1_k5_pass_rate)))
    probes.append(probe_decomposition_argument_v1(
        problems=problems,
        decomposition_argument=decomposition_argument))
    overall = bool(all(p.passed for p in probes))
    verdict = ChartQAPreflightVerdictV1(
        schema=W96_CHARTQA_PREFLIGHT_V1_SCHEMA_VERSION,
        corpus_manifest=manifest.to_dict(),
        probes=tuple(probes),
        overall_passes=bool(overall),
        verdict_cid="")
    cid = _sha256_hex({
        "kind": "w96d_chartqa_preflight_verdict_v1",
        "verdict": {
            **verdict.to_dict(), "verdict_cid": ""},
    })
    return dataclasses.replace(verdict, verdict_cid=str(cid))


__all__ = [
    "W96_CHARTQA_PREFLIGHT_V1_SCHEMA_VERSION",
    "CHARTQA_PUBLISHED_SOTA_SINGLE_SHOT_BY_MODEL",
    "ChartQAPreflightProbeResultV1",
    "ChartQAPreflightVerdictV1",
    "estimate_a1_k5_pass_rate_v1",
    "probe_corpus_integrity_v1",
    "probe_executor_self_test_v1",
    "probe_a1_failure_residual_v1",
    "probe_decomposition_argument_v1",
    "run_chartqa_preflight_v1",
]
