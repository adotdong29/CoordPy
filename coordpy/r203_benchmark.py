"""R-203 — Bounded-Window-Strongest-Baseline V3 Falsifier.

R-203 ships the W83 falsifier benchmark family. The W83 V3
bounded-window baseline composes:

* k=256 raw-event window
* high-fidelity (φ=0.65) rolling summary
* semantic retrieval over the visible window

R-203 demonstrates that even this strongest-known bounded
baseline cannot answer reconstruction queries at horizons past
the visible window + summary coverage. The H-bars:

* H1600: V3 abstains on 100% of horizons past coverage
* H1601: V3 successfully answers queries inside the window
* H1602: V3 successfully answers queries inside the summary
  coverage when the required fidelity is below 0.65
* H1603: V3 emits a content-addressed failure proof CID
* H1604: V3's failure rate at horizons ≥ 1024 is 1.0

R-203 is the **load-bearing falsifier** for the bounded-context-
is-good-enough hypothesis.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.r203_benchmark requires numpy") from exc

from .bounded_window_baseline_v3 import (
    BoundedWindowBaselineV3,
    BoundedWindowEventV3,
    BoundedWindowV3FailureProofV1,
    answer_reconstruction_query_v3,
    build_bounded_window_baseline_v3,
    prove_bounded_window_v3_insufficient_v1,
)


R203_BENCHMARK_SCHEMA_VERSION: str = (
    "coordpy.r203_benchmark.v1")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True,
            separators=(",", ":"),
            default=str).encode("utf-8")).hexdigest()


@dataclasses.dataclass(frozen=True)
class R203HBarV1:
    h_id: str
    pass_: bool
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "h_id": str(self.h_id),
            "pass": bool(self.pass_),
            "detail": str(self.detail),
        }


@dataclasses.dataclass(frozen=True)
class R203ReportV1:
    schema: str
    baseline_cid: str
    proof_cid: str
    failure_rate_at_horizon_1024: float
    in_window_success: bool
    in_summary_success: bool
    h_bars: tuple[R203HBarV1, ...]
    all_pass: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "baseline_cid": str(self.baseline_cid),
            "proof_cid": str(self.proof_cid),
            "failure_rate_at_horizon_1024": float(round(
                self.failure_rate_at_horizon_1024, 12)),
            "in_window_success": bool(
                self.in_window_success),
            "in_summary_success": bool(
                self.in_summary_success),
            "h_bars": [h.to_dict() for h in self.h_bars],
            "all_pass": bool(self.all_pass),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "r203_benchmark_report_v1",
            "report": self.to_dict()})


def run_r203(
        *,
        summary_coverage_turns: int = 512,
        seed: int = 83_203_001,
) -> R203ReportV1:
    baseline = build_bounded_window_baseline_v3()
    proof = prove_bounded_window_v3_insufficient_v1(
        baseline=baseline,
        summary_coverage_turns=int(summary_coverage_turns),
        horizons_to_test=(
            1024, 2048, 8192, 32_768, 100_000),
        seed=int(seed))
    # In-window query: build a small window and ask for an event
    # inside the window.
    rng = _np.random.default_rng(int(seed) + 1)
    window_events = [
        BoundedWindowEventV3(
            turn_index=int(i),
            feature=rng.standard_normal(
                (int(baseline.feature_dim),)
            ).astype(_np.float64),
            payload_cid=_sha256_hex({
                "kind": "r203_in_window_event",
                "turn_index": int(i)}))
        for i in range(int(baseline.window_size))]
    in_window_target = int(
        int(baseline.window_size) // 2)
    in_window_ans = answer_reconstruction_query_v3(
        baseline=baseline,
        events_window=window_events,
        summary_feature_centroid=None,
        summary_covers_turns=(0, 0),
        target_turn_index=int(in_window_target),
        query_feature=None,
        required_fidelity=0.5)
    in_window_ok = bool(
        in_window_ans.success
        and in_window_ans.answer_source == "window")
    # In-summary query: target turn past the window, but inside
    # summary coverage; required fidelity below summary fidelity.
    H = int(baseline.window_size) + (
        int(summary_coverage_turns) // 2)
    target_in_summary = (
        int(H) - int(baseline.window_size) - 1)
    in_summary_ans = answer_reconstruction_query_v3(
        baseline=baseline,
        events_window=[],
        summary_feature_centroid=None,
        summary_covers_turns=(
            int(H) - int(baseline.window_size)
            - int(summary_coverage_turns),
            int(H) - int(baseline.window_size) - 1),
        target_turn_index=int(target_in_summary),
        query_feature=None,
        required_fidelity=0.5)
    in_summary_ok = bool(
        in_summary_ans.success
        and in_summary_ans.answer_source == "summary")
    # Failure rate at horizon 1024 specifically.
    fail_1024_in_proof = bool(
        1024 in [int(h) for h in proof.failure_horizons])
    fail_rate_1024 = (
        1.0 if fail_1024_in_proof else 0.0)
    h_bars: list[R203HBarV1] = []
    h_bars.append(R203HBarV1(
        h_id="H1600_v3_abstains_past_coverage",
        pass_=bool(
            float(
                proof.failure_rate_beyond_coverage)
            >= 1.0 - 1e-12),
        detail=(
            "failure_rate_beyond_coverage="
            f"{proof.failure_rate_beyond_coverage:.4f}")))
    h_bars.append(R203HBarV1(
        h_id="H1601_v3_answers_inside_window",
        pass_=bool(in_window_ok),
        detail=(
            f"in_window_source={in_window_ans.answer_source}")))
    h_bars.append(R203HBarV1(
        h_id="H1602_v3_answers_inside_summary",
        pass_=bool(in_summary_ok),
        detail=(
            "in_summary_source="
            f"{in_summary_ans.answer_source}")))
    h_bars.append(R203HBarV1(
        h_id="H1603_v3_emits_failure_proof_cid",
        pass_=bool(len(str(proof.cid())) == 64),
        detail=f"proof_cid={str(proof.cid())[:16]}..."))
    h_bars.append(R203HBarV1(
        h_id="H1604_v3_failure_rate_at_1024_eq_1",
        pass_=bool(float(fail_rate_1024) >= 1.0 - 1e-12),
        detail=(
            "fail_rate_1024="
            f"{fail_rate_1024:.4f}")))
    all_pass = all(h.pass_ for h in h_bars)
    return R203ReportV1(
        schema=R203_BENCHMARK_SCHEMA_VERSION,
        baseline_cid=str(baseline.cid()),
        proof_cid=str(proof.cid()),
        failure_rate_at_horizon_1024=float(fail_rate_1024),
        in_window_success=bool(in_window_ok),
        in_summary_success=bool(in_summary_ok),
        h_bars=tuple(h_bars),
        all_pass=bool(all_pass),
    )


__all__ = [
    "R203_BENCHMARK_SCHEMA_VERSION",
    "R203HBarV1",
    "R203ReportV1",
    "run_r203",
]
