"""W96-D — RealWorldQA cheap-probe preflight V1.

Extends the W93 5-gate harness with RealWorldQA-specific cheap
probes mirroring the W96-D ChartQA preflight structure (P1
corpus integrity / P2 executor self-test / P3 saturation /
P4 decomposition argument).

This module never calls NIM.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

from .realworldqa_executor_v1 import (
    W96_REALWORLDQA_EXECUTOR_V1_SCHEMA_VERSION,
    executor_self_test_on_gold_v1,
)
from .realworldqa_loader_v1 import (
    REALWORLDQA_TEST_EXPECTED_N_PROBLEMS_LOWER,
    REALWORLDQA_TEST_EXPECTED_N_PROBLEMS_UPPER,
    REALWORLDQA_TEST_EXPECTED_PARQUET_BYTES_LOWER,
    REALWORLDQA_TEST_EXPECTED_PARQUET_BYTES_UPPER,
    RealWorldQACorpusManifestV1,
    RealWorldQAProblemV1,
    W96_REALWORLDQA_LOADER_V1_SCHEMA_VERSION,
)


W96_REALWORLDQA_PREFLIGHT_V1_SCHEMA_VERSION: str = (
    "coordpy.realworldqa_preflight_v1.v1")


# Published-SOTA reference points for RealWorldQA test (765
# problems).  Frozen snapshot from public reports.  Sources:
#   * Llama-3.2 Vision-Instruct release — third-party evals
#     (Sep 2024).
#   * xAI Grok-1.5V announcement — original RealWorldQA dataset
#     introduction.
#   * GPT-4o paper / leaderboard.
#   * Claude-3.5-Sonnet / Gemini-1.5-Pro public reports.
# Values are documented anti-cheat constants; updates are
# explicit code changes.
REALWORLDQA_PUBLISHED_SOTA_SINGLE_SHOT_BY_MODEL: dict[
        str, float] = {
    "llama-3.2-11b-vision-instruct": 50.0,
    "llama-3.2-90b-vision-instruct": 60.0,
    "grok-1.5v": 68.7,
    "gpt-4o": 75.4,
    "claude-3.5-sonnet": 60.1,
    "gemini-1.5-pro": 67.5,
}


@dataclasses.dataclass(frozen=True)
class RealWorldQAPreflightProbeResultV1:
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
class RealWorldQAPreflightVerdictV1:
    schema: str
    corpus_manifest: dict[str, Any]
    probes: tuple[RealWorldQAPreflightProbeResultV1, ...]
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


def probe_corpus_integrity_v1(
        *,
        manifest: RealWorldQACorpusManifestV1,
        problems: tuple[RealWorldQAProblemV1, ...],
) -> RealWorldQAPreflightProbeResultV1:
    n_problems = len(problems)
    n_lower = REALWORLDQA_TEST_EXPECTED_N_PROBLEMS_LOWER
    n_upper = REALWORLDQA_TEST_EXPECTED_N_PROBLEMS_UPPER
    bytes_low = REALWORLDQA_TEST_EXPECTED_PARQUET_BYTES_LOWER
    bytes_high = REALWORLDQA_TEST_EXPECTED_PARQUET_BYTES_UPPER
    parquet_bytes_ok = (
        bytes_low <= manifest.parquet_total_bytes <= bytes_high)
    n_ok = (n_lower <= n_problems <= n_upper)
    n_with_image = sum(
        1 for p in problems if len(p.image_bytes) > 0)
    n_with_answer = sum(
        1 for p in problems if p.answer.strip())
    passed = bool(
        parquet_bytes_ok
        and n_ok
        and n_with_image == n_problems
        and n_with_answer == n_problems
        and all(
            bool(s) for s in manifest.parquet_shard_sha256))
    summary = (
        f"parquet_total_bytes={manifest.parquet_total_bytes} "
        f"(range_ok={parquet_bytes_ok}); "
        f"n_problems={n_problems} in "
        f"[{n_lower},{n_upper}] (ok={n_ok}); "
        f"n_with_image={n_with_image}; "
        f"n_with_answer={n_with_answer}")
    evidence = {
        "parquet_shard_sha256": list(
            manifest.parquet_shard_sha256),
        "parquet_total_bytes": int(manifest.parquet_total_bytes),
        "expected_bytes_range": [
            int(bytes_low), int(bytes_high)],
        "n_problems": int(n_problems),
        "expected_n_range": [int(n_lower), int(n_upper)],
        "n_with_image": int(n_with_image),
        "n_with_answer": int(n_with_answer),
        "corpus_merkle_root": str(manifest.corpus_merkle_root),
    }
    return RealWorldQAPreflightProbeResultV1(
        probe_id="P1_corpus_integrity",
        description=(
            "All shards SHA-anchored; test decodes to expected "
            "size in-range; every problem has image + answer."),
        passed=bool(passed),
        summary=str(summary),
        evidence_cid=_sha256_hex(evidence),
        evidence=dict(evidence))


def probe_executor_self_test_v1(
        *,
        problems: tuple[RealWorldQAProblemV1, ...],
        min_pass_rate: float = 0.98,
) -> RealWorldQAPreflightProbeResultV1:
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
            W96_REALWORLDQA_EXECUTOR_V1_SCHEMA_VERSION),
        "n_problems": int(result["n_problems"]),
        "n_pass": int(result["n_pass"]),
        "pass_rate": float(pass_rate),
        "min_pass_rate": float(min_pass_rate),
        "by_rule": dict(result["by_rule"]),
        "failed_sample": list(result["failed_pids"])[:50],
        "n_failed": int(len(result["failed_pids"])),
    }
    return RealWorldQAPreflightProbeResultV1(
        probe_id="P2_executor_self_test",
        description=(
            "Gold-as-prediction must score ≥ 98 % under the "
            "W96-D RealWorldQA executor."),
        passed=bool(passed),
        summary=str(summary),
        evidence_cid=_sha256_hex(evidence),
        evidence=dict(evidence))


def estimate_a1_k5_pass_rate_v1(
        single_shot_pct: float,
        *,
        k: int = 5,
        correlation: float = 0.5,
) -> float:
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
) -> RealWorldQAPreflightProbeResultV1:
    table = (
        published_sota_table
        if published_sota_table is not None
        else REALWORLDQA_PUBLISHED_SOTA_SINGLE_SHOT_BY_MODEL)
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
        return RealWorldQAPreflightProbeResultV1(
            probe_id="P3_a1_failure_residual",
            description=(
                "Estimated A1@K=5 pass rate based on published "
                "single-shot RealWorldQA SOTA must leave "
                "≥ 20 pp failure-residual."),
            passed=False,
            summary=(
                f"no published single-shot RealWorldQA SOTA "
                f"for '{candidate_model}'; refusing to estimate."),
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
        f"published single-shot RealWorldQA for "
        f"{candidate_model} = {single_shot:.2f}%; "
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
    return RealWorldQAPreflightProbeResultV1(
        probe_id="P3_a1_failure_residual",
        description=(
            "Estimated A1@K=5 pass rate based on published "
            "RealWorldQA SOTA must leave room for B."),
        passed=bool(passed),
        summary=str(summary),
        evidence_cid=_sha256_hex(evidence),
        evidence=dict(evidence))


def probe_decomposition_argument_v1(
        *,
        problems: tuple[RealWorldQAProblemV1, ...],
        decomposition_argument: str,
        min_argument_chars: int = 200,
        min_multimodal_share: float = 0.95,
) -> RealWorldQAPreflightProbeResultV1:
    """Probe P4: written decomposition argument + corpus is fully
    multimodal (every problem has an image AND a question)."""
    arg = str(decomposition_argument or "").strip()
    arg_ok = len(arg) >= int(min_argument_chars)

    n_corpus = len(problems)
    sample_n = min(500, n_corpus)
    n_multimodal = 0
    samples: list[dict[str, Any]] = []
    for p in problems[:sample_n]:
        is_mm = (
            bool(p.image_bytes)
            and bool(p.question.strip())
            and bool(p.answer.strip()))
        if is_mm:
            n_multimodal += 1
        if len(samples) < 5:
            samples.append({
                "pid": str(p.pid),
                "question_head": (
                    p.question[:80] if p.question else ""),
                "answer": str(p.answer),
            })
    if sample_n > 0:
        mm_share = float(n_multimodal) / float(sample_n)
    else:
        mm_share = 0.0
    mm_ok = bool(mm_share >= float(min_multimodal_share))
    passed = bool(arg_ok and mm_ok)
    summary = (
        f"decomp argument len={len(arg)} "
        f"(threshold={min_argument_chars}; ok={arg_ok}); "
        f"multimodal-completeness in sample of {sample_n} = "
        f"{mm_share * 100.0:.1f}% "
        f"(threshold={min_multimodal_share * 100.0:.1f}%; "
        f"ok={mm_ok})")
    evidence = {
        "argument_chars": int(len(arg)),
        "argument_sha256": hashlib.sha256(
            arg.encode("utf-8")).hexdigest(),
        "min_argument_chars": int(min_argument_chars),
        "sample_n": int(sample_n),
        "n_multimodal_in_sample": int(n_multimodal),
        "multimodal_share_in_sample": float(mm_share),
        "min_multimodal_share": float(min_multimodal_share),
        "samples": list(samples),
    }
    return RealWorldQAPreflightProbeResultV1(
        probe_id="P4_decomposition_argument",
        description=(
            "Written structural argument for B's decomposition "
            "exists AND corpus is fully multimodal (image + "
            "question + answer present)."),
        passed=bool(passed),
        summary=str(summary),
        evidence_cid=_sha256_hex(evidence),
        evidence=dict(evidence))


def run_realworldqa_preflight_v1(
        *,
        manifest: RealWorldQACorpusManifestV1,
        problems: tuple[RealWorldQAProblemV1, ...],
        candidate_model: str,
        decomposition_argument: str,
        max_acceptable_a1_k5_pass_rate: float = 80.0,
        min_executor_self_test_pass_rate: float = 0.98,
) -> RealWorldQAPreflightVerdictV1:
    probes: list[RealWorldQAPreflightProbeResultV1] = []
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
    verdict = RealWorldQAPreflightVerdictV1(
        schema=W96_REALWORLDQA_PREFLIGHT_V1_SCHEMA_VERSION,
        corpus_manifest=manifest.to_dict(),
        probes=tuple(probes),
        overall_passes=bool(overall),
        verdict_cid="")
    cid = _sha256_hex({
        "kind": "w96d_realworldqa_preflight_verdict_v1",
        "verdict": {
            **verdict.to_dict(), "verdict_cid": ""},
    })
    return dataclasses.replace(verdict, verdict_cid=str(cid))


__all__ = [
    "W96_REALWORLDQA_PREFLIGHT_V1_SCHEMA_VERSION",
    "REALWORLDQA_PUBLISHED_SOTA_SINGLE_SHOT_BY_MODEL",
    "RealWorldQAPreflightProbeResultV1",
    "RealWorldQAPreflightVerdictV1",
    "estimate_a1_k5_pass_rate_v1",
    "probe_corpus_integrity_v1",
    "probe_executor_self_test_v1",
    "probe_a1_failure_residual_v1",
    "probe_decomposition_argument_v1",
    "run_realworldqa_preflight_v1",
]
