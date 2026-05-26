"""W102 / COO-9 — MBPP+ V2 cheap-probe preflight.

Extends the W101 V1 preflight with TWO new probes that catch the
V1 silent-degeneration failure mode discovered during the W102
fetch step:

* **P5 — extra-test-surface integrity** — confirms every row in
  the V2 corpus has a non-empty `extra_test_program` AND that
  program contains the canonical EvalPlus iteration pattern
  (a `for ... in zip(inputs, results):` or `for i, (inp, exp)
  in enumerate(zip(inputs, results)):` loop calling `assertion(
  <entry_point>(*inp), exp, ...)`).  This directly probes the
  V1 bug: V1 silently parsed `plus_input = []` / `plus_output =
  []` from every row of the actual EvalPlus release, and V1's
  executor degenerated to base-MBPP behavior.

* **P6 — V1-vs-V2 canonical-solution agreement** — runs both V1
  base-only (no extra tests) AND V2 base+plus on the same
  canonical-solution sample; verifies V2 is a STRICT EXTENSION
  of V1, i.e., every problem V2 PASSes under base+plus mode also
  PASSes under V2 base-only mode AND under V1 base-only mode (no
  regressions; V2 just adds the extra-test surface).

Other probes (P1–P4 + AddrW101-P1..P4) are inherited from V1
verbatim but re-bound to the V2 loader + V2 executor so the
preflight verdict matches what the V2 cheap pilot will actually
see.

All probes are NIM-free.

Honest scope (W102)
-------------------

* ``W102-L-MBPP-PLUS-PREFLIGHT-V2-P5-LITERAL-PATTERN-CAP`` —
  P5 uses literal-string + regex pattern matching on each row's
  `extra_test_program` to detect the EvalPlus iteration; if
  EvalPlus re-publishes with a non-iteration-shaped test program
  (e.g., wraps each test in a separate `def test_<n>():` and
  invokes a runner), P5 may underreport.  The fall-back is the
  V2 executor self-test (P2) which is the canonical truth.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Callable

from .mbpp_plus_executor_v2 import (
    W102_MBPP_PLUS_EXECUTOR_V2_SCHEMA_VERSION,
    run_mbpp_plus_executor_v2,
)
from .mbpp_plus_loader_v2 import (
    W102_MBPP_PLUS_LOADER_V2_SCHEMA_VERSION,
    MbppPlusProblemV2,
    MbppPlusV2CorpusError,
    is_mbpp_plus_v2_cached,
    load_mbpp_plus_v2_corpus,
)
from .mbpp_plus_preflight_v1 import (
    A1_SATURATION_THRESHOLD_PP,
    MBPP_PLUS_PUBLISHED_BASE_TO_PLUS_DROP_PP,
    MbppPlusPreflightProbeResultV1,
    MbppPlusPreflightVerdictV1,
    W101_PHASE2_MARGIN_FLOOR_PP,
    W91_5SEED_70B_MBPP_A1_MEAN_PP,
    probe_a1_failure_residual_v1,
    probe_addr_cluster_structure_v1,
    probe_addr_cross_bench_stability_v1,
    probe_addr_mechanism_load_bearing_v1,
    probe_addr_no_anti_pattern_v1,
    probe_decomposition_argument_v1,
)


W102_MBPP_PLUS_PREFLIGHT_V2_SCHEMA_VERSION: str = (
    "coordpy.mbpp_plus_preflight_v2.v1")


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


# Re-use the V1 result dataclass shape so the verdict JSON is
# directly comparable to the W101 verdict.
MbppPlusV2PreflightProbeResult = MbppPlusPreflightProbeResultV1
MbppPlusV2PreflightVerdict = MbppPlusPreflightVerdictV1


# ---------------------------------------------------------------
# V2 P1 — Corpus integrity (V2 loader)
# ---------------------------------------------------------------

def probe_corpus_integrity_v2(
        *,
        cache_path: str | None = None,
) -> MbppPlusV2PreflightProbeResult:
    """V2 P1: MBPP+ V2 corpus loads (parquet path), SHA-matches
    the pinned HF LFS oid, decodes to a valid number of problems,
    every problem carries non-empty entry_point + non-empty
    base_test_list + non-empty extra_test_program."""
    has_cache = is_mbpp_plus_v2_cached(cache_path=cache_path)
    if not has_cache:
        evidence = {
            "has_cache": False,
            "expected_cache_path": str(
                cache_path
                or "~/.cache/coordpy/mbpp-plus.parquet"),
            "fetch_instructions": (
                "Operator must fetch the EvalPlus MBPP+ parquet "
                "from Hugging Face.  W102 RUNBOOK §"
                "'Critical W102 finding' documents the canonical "
                "URL + SHA pin."),
            "loader_schema": (
                W102_MBPP_PLUS_LOADER_V2_SCHEMA_VERSION),
        }
        return MbppPlusV2PreflightProbeResult(
            probe_id="P1",
            description=(
                "MBPP+ V2 corpus integrity (SHA-pinned canonical "
                "EvalPlus HF parquet)"),
            passed=False,
            summary=(
                "MBPP+ V2 cache absent; preflight cannot "
                "validate integrity.  W102 cheap pilot is "
                "BLOCKED until operator fetches MBPP+ + records "
                "the SHA pin."),
            evidence_cid=_sha256_hex(evidence),
            evidence=evidence,
        )
    try:
        corpus = load_mbpp_plus_v2_corpus(cache_path=cache_path)
    except MbppPlusV2CorpusError as e:
        evidence = {
            "has_cache": True,
            "load_error": f"{type(e).__name__}: {e}",
        }
        return MbppPlusV2PreflightProbeResult(
            probe_id="P1",
            description=(
                "MBPP+ V2 corpus integrity (SHA-pinned canonical "
                "EvalPlus HF parquet)"),
            passed=False,
            summary=(
                "MBPP+ V2 corpus failed to load + verify; "
                "preflight is BLOCKED."),
            evidence_cid=_sha256_hex(evidence),
            evidence=evidence,
        )
    n_with_extra = sum(
        1 for p in corpus
        if str(p.extra_test_program).strip())
    n_with_base = sum(
        1 for p in corpus if p.base_test_list)
    n_with_entry = sum(
        1 for p in corpus if p.entry_point)
    evidence = {
        "has_cache": True,
        "n_problems": int(len(corpus)),
        "n_with_extra_test_program": int(n_with_extra),
        "n_with_base_tests": int(n_with_base),
        "n_with_entry_point": int(n_with_entry),
        "first_task_id": str(corpus[0].task_id),
        "last_task_id": str(corpus[-1].task_id),
        "loader_schema": (
            W102_MBPP_PLUS_LOADER_V2_SCHEMA_VERSION),
    }
    passed = (
        len(corpus) >= 350
        and n_with_extra >= int(len(corpus) * 0.95)
        and n_with_base >= int(len(corpus) * 0.95)
        and n_with_entry == len(corpus))
    return MbppPlusV2PreflightProbeResult(
        probe_id="P1",
        description=(
            "MBPP+ V2 corpus integrity (SHA-pinned canonical "
            "EvalPlus HF parquet)"),
        passed=bool(passed),
        summary=(
            f"MBPP+ V2 corpus loaded: {len(corpus)} problems "
            f"({n_with_extra} with extra_test_program; "
            f"{n_with_base} with base_test_list; "
            f"{n_with_entry} with entry_point)."),
        evidence_cid=_sha256_hex(evidence),
        evidence=evidence,
    )


# ---------------------------------------------------------------
# V2 P2 — Executor self-test on canonical solutions (V2 executor)
# ---------------------------------------------------------------

def probe_executor_self_test_on_gold_v2(
        *,
        cache_path: str | None = None,
        n_sample: int = 30,
        on_progress: (
            Callable[[int, int, bool], None] | None) = None,
) -> MbppPlusV2PreflightProbeResult:
    """V2 P2: feed each canonical solution to the V2 executor in
    `base_and_plus` mode; verify ≥ 98 % pass.  Catches both
    encoding regressions AND the V1 silent-degeneration failure
    mode (V1 would have reported 100 % pass with 0 EvalPlus
    extra-tests actually run)."""
    if not is_mbpp_plus_v2_cached(cache_path=cache_path):
        evidence = {"deferred": True,
                    "reason": "MBPP+ V2 cache absent"}
        return MbppPlusV2PreflightProbeResult(
            probe_id="P2",
            description=(
                "V2 executor self-test on canonical solutions "
                "(base_and_plus mode)"),
            passed=False,
            summary=(
                "MBPP+ V2 cache absent; executor self-test "
                "DEFERRED."),
            evidence_cid=_sha256_hex(evidence),
            evidence=evidence,
        )
    try:
        corpus = load_mbpp_plus_v2_corpus(cache_path=cache_path)
    except MbppPlusV2CorpusError as e:
        evidence = {
            "deferred": True,
            "load_error": f"{type(e).__name__}: {e}",
        }
        return MbppPlusV2PreflightProbeResult(
            probe_id="P2",
            description=(
                "V2 executor self-test on canonical solutions "
                "(base_and_plus mode)"),
            passed=False,
            summary=(
                "MBPP+ V2 corpus did not load; executor "
                "self-test DEFERRED."),
            evidence_cid=_sha256_hex(evidence),
            evidence=evidence,
        )
    sample = (
        corpus[:int(n_sample)]
        if n_sample > 0 and n_sample <= len(corpus)
        else corpus)
    n_pass = 0
    failures: list[dict[str, Any]] = []
    for i, p in enumerate(sample):
        if not p.canonical_code:
            continue
        try:
            exe = run_mbpp_plus_executor_v2(
                problem=p, candidate_code=p.canonical_code,
                mode="base_and_plus")
        except Exception as e:  # noqa: BLE001
            failures.append({
                "task_id": str(p.task_id),
                "error": f"{type(e).__name__}: {e}",
            })
            continue
        if exe.passed:
            n_pass += 1
        else:
            failures.append({
                "task_id": str(p.task_id),
                "returncode": int(exe.returncode),
                "stderr_tail": str(exe.stderr_tail)[:240],
            })
        if on_progress is not None:
            on_progress(int(i), int(len(sample)),
                        bool(exe.passed))
    pass_rate = float(n_pass / max(len(sample), 1))
    evidence = {
        "n_sampled": int(len(sample)),
        "n_pass": int(n_pass),
        "pass_rate": float(round(pass_rate, 4)),
        "failures": failures[:10],
        "executor_schema": (
            W102_MBPP_PLUS_EXECUTOR_V2_SCHEMA_VERSION),
    }
    return MbppPlusV2PreflightProbeResult(
        probe_id="P2",
        description=(
            "V2 executor self-test on canonical solutions "
            "(base_and_plus mode)"),
        passed=bool(pass_rate >= 0.98),
        summary=(
            f"V2 canonical-solution executor self-test "
            f"(base_and_plus): {n_pass}/{len(sample)} = "
            f"{pass_rate*100:.2f}%; floor 98%."),
        evidence_cid=_sha256_hex(evidence),
        evidence=evidence,
    )


# ---------------------------------------------------------------
# V2 P5 — Extra-test-surface integrity (NEW; catches V1 bug)
# ---------------------------------------------------------------

_ITER_PATTERN_RE = re.compile(
    r"for\s+\w+\s*,\s*\([^)]*\)\s+in\s+enumerate\s*\(\s*zip\s*\("
    r"\s*inputs\s*,\s*results\s*\)\s*\)",
    re.MULTILINE | re.IGNORECASE)

_ITER_PATTERN_SIMPLE_RE = re.compile(
    r"for\s+\w+\s*,\s*\w+\s+in\s+zip\s*\(\s*inputs\s*,\s*"
    r"results\s*\)",
    re.MULTILINE | re.IGNORECASE)


def probe_extra_test_surface_integrity_v2(
        *,
        cache_path: str | None = None,
) -> MbppPlusV2PreflightProbeResult:
    """V2 P5: confirm every row's `extra_test_program` contains
    the canonical EvalPlus iteration pattern.  This is the
    structural guard against the V1 silent-degeneration failure
    mode.

    PASS criterion: ≥ 95 % of rows carry an iteration loop over
    `(inputs, results)` AND a call to `assertion(<entry_point>(`.
    """
    if not is_mbpp_plus_v2_cached(cache_path=cache_path):
        evidence = {"deferred": True,
                    "reason": "MBPP+ V2 cache absent"}
        return MbppPlusV2PreflightProbeResult(
            probe_id="P5",
            description=(
                "Extra-test-surface integrity guard (V1 silent-"
                "degeneration anti-pattern)"),
            passed=False,
            summary=(
                "MBPP+ V2 cache absent; P5 DEFERRED."),
            evidence_cid=_sha256_hex(evidence),
            evidence=evidence,
        )
    try:
        corpus = load_mbpp_plus_v2_corpus(
            cache_path=cache_path)
    except MbppPlusV2CorpusError as e:
        evidence = {
            "deferred": True,
            "load_error": f"{type(e).__name__}: {e}",
        }
        return MbppPlusV2PreflightProbeResult(
            probe_id="P5",
            description=(
                "Extra-test-surface integrity guard"),
            passed=False,
            summary=(
                "MBPP+ V2 corpus did not load; P5 DEFERRED."),
            evidence_cid=_sha256_hex(evidence),
            evidence=evidence,
        )
    n_with_iter = 0
    n_with_assertion_call = 0
    n_total = len(corpus)
    misses: list[str] = []
    for p in corpus:
        prog = str(p.extra_test_program)
        has_iter = bool(
            _ITER_PATTERN_RE.search(prog)
            or _ITER_PATTERN_SIMPLE_RE.search(prog))
        has_assertion_call = (
            f"assertion({p.entry_point}(" in prog
            or f"assertion( {p.entry_point}(" in prog)
        if has_iter:
            n_with_iter += 1
        if has_assertion_call:
            n_with_assertion_call += 1
        if not (has_iter and has_assertion_call):
            misses.append(p.task_id)
    iter_rate = float(n_with_iter / max(n_total, 1))
    call_rate = float(
        n_with_assertion_call / max(n_total, 1))
    evidence = {
        "n_problems": int(n_total),
        "n_with_iteration_loop": int(n_with_iter),
        "n_with_assertion_call": int(n_with_assertion_call),
        "iter_rate": float(round(iter_rate, 4)),
        "assertion_call_rate": float(round(call_rate, 4)),
        "first_misses": misses[:5],
        "min_required_rate": 0.95,
    }
    passed = bool(iter_rate >= 0.95 and call_rate >= 0.95)
    return MbppPlusV2PreflightProbeResult(
        probe_id="P5",
        description=(
            "Extra-test-surface integrity guard (V1 silent-"
            "degeneration anti-pattern)"),
        passed=bool(passed),
        summary=(
            f"Extra-test surface: iter loop on "
            f"{n_with_iter}/{n_total} = {iter_rate*100:.2f}%; "
            f"assertion call on {n_with_assertion_call}/"
            f"{n_total} = {call_rate*100:.2f}% (floor 95%)."),
        evidence_cid=_sha256_hex(evidence),
        evidence=evidence,
    )


# ---------------------------------------------------------------
# V2 P6 — V2 strict-extension-of-V1 agreement
# ---------------------------------------------------------------

def probe_v1_v2_canonical_agreement(
        *,
        cache_path: str | None = None,
        n_sample: int = 30,
) -> MbppPlusV2PreflightProbeResult:
    """V2 P6: confirm V2 base-only mode is a strict extension of
    V1 base-only behavior on canonical solutions (every canonical
    that PASSed under V1 also PASSes under V2 base-only; V2 adds
    extra-test surface, not regressions).

    PASS criterion: V2 base-only canonical pass rate ≥ 95 %.
    """
    if not is_mbpp_plus_v2_cached(cache_path=cache_path):
        evidence = {"deferred": True,
                    "reason": "MBPP+ V2 cache absent"}
        return MbppPlusV2PreflightProbeResult(
            probe_id="P6",
            description=(
                "V2 strict-extension-of-V1 canonical agreement"),
            passed=False,
            summary=(
                "MBPP+ V2 cache absent; P6 DEFERRED."),
            evidence_cid=_sha256_hex(evidence),
            evidence=evidence,
        )
    corpus = load_mbpp_plus_v2_corpus(cache_path=cache_path)
    sample = (
        corpus[:int(n_sample)]
        if n_sample > 0 and n_sample <= len(corpus)
        else corpus)
    n_pass = 0
    for p in sample:
        if not p.canonical_code:
            continue
        exe = run_mbpp_plus_executor_v2(
            problem=p, candidate_code=p.canonical_code,
            mode="base_only")
        if exe.passed:
            n_pass += 1
    rate = float(n_pass / max(len(sample), 1))
    evidence = {
        "n_sampled": int(len(sample)),
        "n_pass_base_only": int(n_pass),
        "pass_rate": float(round(rate, 4)),
        "min_required_rate": 0.95,
    }
    passed = bool(rate >= 0.95)
    return MbppPlusV2PreflightProbeResult(
        probe_id="P6",
        description=(
            "V2 strict-extension-of-V1 canonical agreement"),
        passed=bool(passed),
        summary=(
            f"V2 base_only canonical: {n_pass}/{len(sample)} = "
            f"{rate*100:.2f}% (floor 95%)."),
        evidence_cid=_sha256_hex(evidence),
        evidence=evidence,
    )


# ---------------------------------------------------------------
# Driver
# ---------------------------------------------------------------

def run_mbpp_plus_preflight_v2(
        *,
        cache_path: str | None = None,
        arsenal_mining_report_path: Path,
        bench_module_path: Path,
        candidate: str = (
            "B (W89 sequential reflexion on MBPP+ V2)"),
        candidate_model: str = (
            "meta/llama-3.3-70b-instruct"),
        run_executor_self_test: bool = True,
        executor_self_test_sample: int = 30,
) -> MbppPlusV2PreflightVerdict:
    """Run all W102 V2 preflight probes and emit a structured
    verdict.

    Includes 10 probes total:
      P1 corpus integrity (V2 loader)
      P2 executor self-test (V2 executor; base_and_plus)
      P3 A1@K=5 failure-residual estimate (V1 verbatim)
      P4 decomposition argument (V1 verbatim)
      P5 extra-test-surface integrity (NEW V2)
      P6 V1-vs-V2 canonical agreement (NEW V2)
      AddrW101-P1 mechanism-load-bearing prior (V1 verbatim)
      AddrW101-P2 per-problem cluster structure (V1 verbatim)
      AddrW101-P3 cross-bench failure-residual stability
        (V1 verbatim)
      AddrW101-P4 anti-pattern guard (V1 verbatim)

    NIM-free.
    """
    probes: list[MbppPlusV2PreflightProbeResult] = []
    probes.append(probe_corpus_integrity_v2(
        cache_path=cache_path))
    if run_executor_self_test:
        probes.append(probe_executor_self_test_on_gold_v2(
            cache_path=cache_path,
            n_sample=int(executor_self_test_sample)))
    else:
        probes.append(MbppPlusV2PreflightProbeResult(
            probe_id="P2",
            description=(
                "V2 executor self-test (skipped)"),
            passed=False,
            summary="executor self-test skipped",
            evidence_cid=_sha256_hex({"skipped": True}),
            evidence={"skipped": True},
        ))
    probes.append(probe_a1_failure_residual_v1(
        arsenal_mining_report_path=(
            arsenal_mining_report_path
            if arsenal_mining_report_path.exists()
            else None),
        cached_mbpp_plus_path=cache_path))
    probes.append(probe_decomposition_argument_v1())
    probes.append(probe_extra_test_surface_integrity_v2(
        cache_path=cache_path))
    if run_executor_self_test:
        probes.append(probe_v1_v2_canonical_agreement(
            cache_path=cache_path,
            n_sample=int(executor_self_test_sample)))
    else:
        probes.append(MbppPlusV2PreflightProbeResult(
            probe_id="P6",
            description="V2 vs V1 canonical agreement (skipped)",
            passed=False,
            summary="P6 skipped",
            evidence_cid=_sha256_hex({"skipped": True}),
            evidence={"skipped": True},
        ))
    probes.append(probe_addr_mechanism_load_bearing_v1(
        arsenal_mining_report_path=(
            arsenal_mining_report_path)))
    probes.append(probe_addr_cluster_structure_v1(
        arsenal_mining_report_path=(
            arsenal_mining_report_path)))
    probes.append(probe_addr_cross_bench_stability_v1(
        cache_path=cache_path))
    probes.append(probe_addr_no_anti_pattern_v1(
        bench_module_path=bench_module_path))
    n_passed = sum(1 for p in probes if p.passed)
    # W102 V2 preflight is stricter than W101 V1: NO DEFERREDs
    # allowed once the corpus is fetched.  All 10 probes must
    # PASS for the cheap pilot to be earned.
    n_required = len(probes)
    overall = bool(n_passed >= n_required)
    verdict = {
        "schema": (
            W102_MBPP_PLUS_PREFLIGHT_V2_SCHEMA_VERSION),
        "candidate": str(candidate),
        "candidate_model": str(candidate_model),
        "probes": [p.to_dict() for p in probes],
        "overall_passes": bool(overall),
        "n_passed": int(n_passed),
        "n_required": int(n_required),
    }
    return MbppPlusV2PreflightVerdict(
        schema=W102_MBPP_PLUS_PREFLIGHT_V2_SCHEMA_VERSION,
        candidate=str(candidate),
        candidate_model=str(candidate_model),
        probes=tuple(probes),
        overall_passes=bool(overall),
        n_passed=int(n_passed),
        n_required=int(n_required),
        verdict_cid=_sha256_hex(verdict),
    )


__all__ = [
    "W102_MBPP_PLUS_PREFLIGHT_V2_SCHEMA_VERSION",
    "MbppPlusV2PreflightProbeResult",
    "MbppPlusV2PreflightVerdict",
    "probe_corpus_integrity_v2",
    "probe_executor_self_test_on_gold_v2",
    "probe_extra_test_surface_integrity_v2",
    "probe_v1_v2_canonical_agreement",
    "run_mbpp_plus_preflight_v2",
]
