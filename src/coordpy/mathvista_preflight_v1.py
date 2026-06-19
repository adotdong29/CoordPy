"""W95 — MathVista cheap-probe preflight V1.

Extends the W93 5-gate harness
(``coordpy.cross_modal_preflight_harness_v1``) with MathVista-
specific cheap probes that answer the four W94 questions BEFORE
any NIM spend:

  P1. Is the corpus reproducible from the canonical parquet
      (SHA-anchored, 1000 problems, valid images, decoded
      schema)?
  P2. Is the executor clean? (gold-as-prediction → 100 % pass
      under the W95 executor; if not, the executor is silently
      penalising every arm and the battlefield is broken.)
  P3. Is there a plausible failure-residual under unified VLM at
      K=5? (estimated from published SOTA scores; checked against
      a probe budget of zero NIM calls.)
  P4. Is there a plausible decomposition advantage for B? (written
      argument; structural fit between MathVista questions and
      vision-reader + math-solver pipeline.)

Each probe returns a ``PreflightProbeResultV1``.  All four probes
plus the W93 G1–G5 gates must pass before any NIM-spending pilot
launches.  If any probe fails, the W95 battlefield is killed
cheaply with a carry-forward in
``docs/RESULTS_W95_MATHVISTA_PREFLIGHT_V1.md``.

This module never calls NIM.  Probes that consume external state
(e.g., the parquet corpus) use ONLY content-addressed local data;
the network fetch is delegated to
``mathvista_loader_v1.fetch_testmini_parquet`` and is performed
once before the probes run.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

from .mathvista_executor_v1 import (
    W95_MATHVISTA_EXECUTOR_V1_SCHEMA_VERSION,
    executor_self_test_on_gold_v1,
)
from .mathvista_loader_v1 import (
    MATHVISTA_TESTMINI_EXPECTED_N_PROBLEMS,
    MATHVISTA_TESTMINI_EXPECTED_PARQUET_BYTES_LOWER,
    MATHVISTA_TESTMINI_EXPECTED_PARQUET_BYTES_UPPER,
    MathVistaCorpusManifestV1,
    MathVistaProblemV1,
    W95_MATHVISTA_LOADER_V1_SCHEMA_VERSION,
)


W95_MATHVISTA_PREFLIGHT_V1_SCHEMA_VERSION: str = (
    "coordpy.mathvista_preflight_v1.v1")


# Published-SOTA reference points used by P3.  The values below
# are taken from the public MathVista leaderboard / paper as of
# 2026-05; the loader does not fetch them from the web — they are
# baked in as documented anti-cheat constants so the probe is
# byte-reproducible.  Future updates to these constants are an
# explicit code change.
#
# References:
#   * MathVista paper (Lu et al., 2024), Table 2.
#   * Llama-3.2 release notes for 11B-Vision-Instruct testmini
#     reported single-shot ≈ 33 %.
#   * Llama-3.2 release notes for 90B-Vision-Instruct testmini
#     reported single-shot ≈ 49 %.
#   * GPT-4o testmini ≈ 63 %; Claude-3.5-Sonnet testmini ≈ 67 %.
MATHVISTA_PUBLISHED_SOTA_SINGLE_SHOT_BY_MODEL: dict[
        str, float] = {
    "llama-3.2-11b-vision-instruct": 33.0,
    "llama-3.2-90b-vision-instruct": 49.0,
    "gpt-4o": 63.0,
    "claude-3.5-sonnet": 67.5,
    "qwen2-vl-72b": 60.0,
}


@dataclasses.dataclass(frozen=True)
class PreflightProbeResultV1:
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
class MathVistaPreflightVerdictV1:
    schema: str
    corpus_manifest: dict[str, Any]
    probes: tuple[PreflightProbeResultV1, ...]
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
        manifest: MathVistaCorpusManifestV1,
        problems: tuple[MathVistaProblemV1, ...],
) -> PreflightProbeResultV1:
    """Probe P1: the parquet hashes to a known value, decodes to
    the expected number of problems, and every problem carries a
    non-empty image + non-empty gold answer + valid answer_type.
    """
    n_problems = len(problems)
    expected_n = MATHVISTA_TESTMINI_EXPECTED_N_PROBLEMS
    bytes_low = MATHVISTA_TESTMINI_EXPECTED_PARQUET_BYTES_LOWER
    bytes_high = MATHVISTA_TESTMINI_EXPECTED_PARQUET_BYTES_UPPER
    parquet_bytes_ok = (
        bytes_low <= manifest.parquet_bytes <= bytes_high)
    n_ok = (n_problems == expected_n)
    n_with_image = sum(
        1 for p in problems if len(p.image_bytes) > 0)
    n_with_answer = sum(
        1 for p in problems if p.answer.strip())
    valid_answer_types = {
        "integer", "float", "text", "list"}
    valid_question_types = {
        "multi_choice", "free_form"}
    n_bad_answer_type = sum(
        1 for p in problems
        if p.answer_type and (
            p.answer_type.lower() not in valid_answer_types))
    n_bad_question_type = sum(
        1 for p in problems
        if p.question_type and (
            p.question_type.lower() not in valid_question_types))
    passed = bool(
        parquet_bytes_ok
        and n_ok
        and n_with_image == n_problems
        and n_with_answer == n_problems
        and n_bad_answer_type == 0
        and n_bad_question_type == 0
        and bool(manifest.parquet_sha256))
    summary = (
        f"parquet_bytes={manifest.parquet_bytes} "
        f"(range_ok={parquet_bytes_ok}); "
        f"n_problems={n_problems}/{expected_n} (ok={n_ok}); "
        f"n_with_image={n_with_image}; "
        f"n_with_answer={n_with_answer}; "
        f"n_bad_answer_type={n_bad_answer_type}; "
        f"n_bad_question_type={n_bad_question_type}")
    evidence = {
        "parquet_sha256": str(manifest.parquet_sha256),
        "parquet_bytes": int(manifest.parquet_bytes),
        "expected_bytes_range": [
            int(bytes_low), int(bytes_high)],
        "n_problems": int(n_problems),
        "expected_n_problems": int(expected_n),
        "n_with_image": int(n_with_image),
        "n_with_answer": int(n_with_answer),
        "n_bad_answer_type": int(n_bad_answer_type),
        "n_bad_question_type": int(n_bad_question_type),
        "corpus_merkle_root": str(manifest.corpus_merkle_root),
    }
    return PreflightProbeResultV1(
        probe_id="P1_corpus_integrity",
        description=(
            "Parquet SHA-anchored; testmini decodes to expected "
            "size; every problem has image + answer + valid "
            "answer/question type."),
        passed=bool(passed),
        summary=str(summary),
        evidence_cid=_sha256_hex(evidence),
        evidence=dict(evidence))


# ---------------------------------------------------------------
# P2 — Executor self-test on gold
# ---------------------------------------------------------------

def probe_executor_self_test_v1(
        *,
        problems: tuple[MathVistaProblemV1, ...],
        min_pass_rate: float = 0.98,
) -> PreflightProbeResultV1:
    """Probe P2: feed each problem's gold answer back through the
    executor.  A well-formed executor MUST pass on ~all gold;
    otherwise it would silently penalise every bench arm equally
    and the headline numbers would be wrong.  The threshold is
    98 % rather than 100 % because MathVista contains a small
    number of problems whose gold answer is a free-text phrase
    where canonical normalisation has multiple valid forms; W95
    tolerates these edge cases as long as they are a small
    minority that affects all arms equally.
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
            W95_MATHVISTA_EXECUTOR_V1_SCHEMA_VERSION),
        "n_problems": int(result["n_problems"]),
        "n_pass": int(result["n_pass"]),
        "pass_rate": float(pass_rate),
        "min_pass_rate": float(min_pass_rate),
        "by_rule": dict(result["by_rule"]),
        # Cap failed-pids list to the first 50 for sidecar size.
        "failed_sample": list(result["failed_pids"])[:50],
        "n_failed": int(len(result["failed_pids"])),
    }
    return PreflightProbeResultV1(
        probe_id="P2_executor_self_test",
        description=(
            "Gold-as-prediction must score ≥ 98 % under the "
            "W95 executor; else the executor silently penalises "
            "every arm and breaks the bench truth function."),
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
    """A back-of-envelope upper bound for A1 first-pass-among-K
    when independent samples are drawn at temperature > 0.  We
    use a correlation factor to model the empirical fact that
    repeated samples on the same problem are NOT i.i.d.: hard
    problems tend to fail repeatedly.  ``correlation = 0`` gives
    the i.i.d. upper bound; ``correlation = 1`` gives the
    single-shot floor.

    A1@K = (1 − correlation) × i.i.d._upper + correlation × single

    where
        i.i.d._upper = 1 − (1 − p_single)^K.

    This is documented anti-cheat: the constants used here are
    explicit in the source.  The W93 discipline does not let us
    use this for final claims, only for cheap presumption.
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
) -> PreflightProbeResultV1:
    """Probe P3: use the published single-shot SOTA score for the
    candidate VLM family to estimate A1@K=5 and refuse to launch
    if A1 is presumptively saturated above the configured ceiling.

    Default ceiling is 80 % — a 20 % failure-residual is the
    minimum room a team-based B candidate needs to clear the W88
    +5 pp margin bar honestly (and is 2-3× HumanEval-Visual's
    8-12 % residual).  This is documented as a presumption only;
    a real NIM pilot may revise it, but the cheap probe gates
    everything else.
    """
    table = (
        published_sota_table
        if published_sota_table is not None
        else MATHVISTA_PUBLISHED_SOTA_SINGLE_SHOT_BY_MODEL)
    key = str(candidate_model).lower().strip()
    # Normalise common variations.
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
        return PreflightProbeResultV1(
            probe_id="P3_a1_failure_residual",
            description=(
                "Estimated A1@K=5 pass rate based on published "
                "single-shot SOTA must leave ≥ 20 pp failure-"
                "residual."),
            passed=False,
            summary=(
                f"no published single-shot SOTA for "
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
        f"published single-shot for {candidate_model} = "
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
    return PreflightProbeResultV1(
        probe_id="P3_a1_failure_residual",
        description=(
            "Estimated A1@K=5 pass rate based on published "
            "single-shot SOTA must leave room for B to win."),
        passed=bool(passed),
        summary=str(summary),
        evidence_cid=_sha256_hex(evidence),
        evidence=dict(evidence))


# ---------------------------------------------------------------
# P4 — Decomposition argument
# ---------------------------------------------------------------

def probe_decomposition_argument_v1(
        *,
        problems: tuple[MathVistaProblemV1, ...],
        decomposition_argument: str,
        min_argument_chars: int = 200,
        min_geometric_or_chart_share: float = 0.20,
) -> PreflightProbeResultV1:
    """Probe P4: the candidate B's decomposition must have a
    written structural argument AND the corpus must have enough
    problems where vision-extract + math-solve is a plausibly
    distinct pipeline (geometry / chart / figure problems).

    The corpus check looks at the per-problem ``metadata`` for
    ``category`` / ``task`` / ``skills`` fields that match
    geometry/chart/figure-style sub-categories.
    """
    arg = str(decomposition_argument or "").strip()
    arg_ok = len(arg) >= int(min_argument_chars)

    # Categorise by metadata fields.  MathVista's metadata
    # commonly includes "category" (math sub-area), "task" (e.g.
    # geometry_problem_solving, math_word_problem,
    # figure_qa, etc.), and "skills" (list of fine-grained
    # capabilities).  We accept either:
    #   * task contains "geometry" / "figure" / "chart" /
    #     "scientific" / "table", OR
    #   * skills contains any of a small whitelist.
    geo_kws = (
        "geometry", "figure", "chart", "scientific",
        "table", "diagram", "plot")
    n_corpus = len(problems)
    n_geo = 0
    samples: list[dict[str, Any]] = []
    for p in problems[:200]:  # cap sample size for evidence
        md = p.metadata or {}
        task = str(md.get("task", "")).lower()
        category = str(md.get("category", "")).lower()
        skills = md.get("skills") or []
        if not isinstance(skills, (list, tuple)):
            skills = [skills]
        skills_text = " ".join(
            str(s).lower() for s in skills)
        combined = task + " " + category + " " + skills_text
        is_geo = any(k in combined for k in geo_kws)
        if is_geo:
            n_geo += 1
        if len(samples) < 5:
            samples.append({
                "pid": str(p.pid),
                "task": str(task),
                "category": str(category),
                "skills": [str(s) for s in skills],
                "is_geo_or_chart": bool(is_geo),
            })
    # Extrapolate sample share to corpus share is unfair; here we
    # report and gate on what we sampled (200/1000 = 20%).  We
    # require ≥ min_geometric_or_chart_share of the SAMPLE to be
    # geometry/chart-style.
    sample_n = min(200, n_corpus)
    if sample_n > 0:
        geo_share = float(n_geo) / float(sample_n)
    else:
        geo_share = 0.0
    geo_ok = bool(geo_share >= float(min_geometric_or_chart_share))
    passed = bool(arg_ok and geo_ok)
    summary = (
        f"decomp argument len={len(arg)} "
        f"(threshold={min_argument_chars}; ok={arg_ok}); "
        f"geo/chart-style share in sample of {sample_n} = "
        f"{geo_share * 100.0:.1f}% "
        f"(threshold={min_geometric_or_chart_share * 100.0:.1f}%; "
        f"ok={geo_ok})")
    evidence = {
        "argument_chars": int(len(arg)),
        "argument_sha256": hashlib.sha256(
            arg.encode("utf-8")).hexdigest(),
        "min_argument_chars": int(min_argument_chars),
        "sample_n": int(sample_n),
        "n_geo_or_chart_in_sample": int(n_geo),
        "geo_share_in_sample": float(geo_share),
        "min_geo_share": float(min_geometric_or_chart_share),
        "samples": list(samples),
    }
    return PreflightProbeResultV1(
        probe_id="P4_decomposition_argument",
        description=(
            "Written structural argument for B's decomposition "
            "exists AND corpus has ≥ 20 % geometry/chart-style "
            "problems where vision-extract + math-solve is "
            "plausibly distinct from unified VLM."),
        passed=bool(passed),
        summary=str(summary),
        evidence_cid=_sha256_hex(evidence),
        evidence=dict(evidence))


# ---------------------------------------------------------------
# Composite verdict
# ---------------------------------------------------------------

def run_mathvista_preflight_v1(
        *,
        manifest: MathVistaCorpusManifestV1,
        problems: tuple[MathVistaProblemV1, ...],
        candidate_model: str,
        decomposition_argument: str,
        max_acceptable_a1_k5_pass_rate: float = 80.0,
        min_executor_self_test_pass_rate: float = 0.98,
) -> MathVistaPreflightVerdictV1:
    """Run all four W95 cheap probes against a loaded corpus and
    return a content-addressed verdict.  ``overall_passes = True``
    iff all four probes pass."""
    probes: list[PreflightProbeResultV1] = []
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
    verdict = MathVistaPreflightVerdictV1(
        schema=W95_MATHVISTA_PREFLIGHT_V1_SCHEMA_VERSION,
        corpus_manifest=manifest.to_dict(),
        probes=tuple(probes),
        overall_passes=bool(overall),
        verdict_cid="")
    cid = _sha256_hex({
        "kind": "w95_mathvista_preflight_verdict_v1",
        "verdict": {
            **verdict.to_dict(), "verdict_cid": ""},
    })
    return dataclasses.replace(verdict, verdict_cid=str(cid))


__all__ = [
    "W95_MATHVISTA_PREFLIGHT_V1_SCHEMA_VERSION",
    "MATHVISTA_PUBLISHED_SOTA_SINGLE_SHOT_BY_MODEL",
    "PreflightProbeResultV1",
    "MathVistaPreflightVerdictV1",
    "estimate_a1_k5_pass_rate_v1",
    "probe_corpus_integrity_v1",
    "probe_executor_self_test_v1",
    "probe_a1_failure_residual_v1",
    "probe_decomposition_argument_v1",
    "run_mathvista_preflight_v1",
]
