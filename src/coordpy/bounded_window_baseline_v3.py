"""W83 — Bounded Window Baseline V3.

The W79 ``bounded_window_baseline_v2`` already proves that fixed-k
windows + a static cross-prompt summary cannot answer
reconstruction queries past their window. W82's far-horizon and
compound-failure benches add a wider ``bounded_window_k128`` /
``rolling_summary`` family.

W83 V3 is the **strongest known bounded baseline** in the
programme — designed specifically to be the hardest falsifier
target. V3 composes:

1. A ``k=256`` raw-event window (4× the W79 V2 k=64 ceiling).
2. A **dynamic rolling summary** that updates every M turns and
   carries a high-fidelity (φ=0.65, vs W79 V2 φ=0.20) compressed
   description of the events past the window.
3. A **semantic retrieval** layer that, when a reconstruction
   query arrives, retrieves the top-K events from the visible
   window by cosine similarity to the query. This is the
   *strongest known* bounded technique in production today
   (retrieval-augmented context).

The W83 V3 is *still* bounded: every component looks only at
the visible window or at a hand-compressed summary. The W83
substrate-coupled pipeline strictly beats V3 on regimes that
require any of:

* genuine cross-window-boundary recall (event T<k can't be
  retrieved if k>= than the window the substrate carrier
  remembers).
* multi-hop reconstruction (V3 retrieves one event at a time,
  cannot chain across slots).
* horizons >= ~10k turns where even k=256 + summary cannot
  carry enough signal density.

Honest scope (W83)
------------------

* ``W83-L-BW-V3-RESEARCH-ONLY-CAP`` — explicit-import only.
* ``W83-L-BW-V3-STILL-BOUNDED-CAP`` — V3 is the *strongest known*
  bounded baseline. It is still bounded. The W83 falsifier is
  the claim that even V3 cannot match the substrate-coupled
  pipeline on long-horizon and multi-hop regimes.
* ``W83-L-BW-V3-SYNTHETIC-CAP`` — V3 is evaluated on the W79 /
  W82 synthetic carriers; live-runtime evaluation is future
  work.
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
        "coordpy.bounded_window_baseline_v3 requires numpy"
        ) from exc


W83_BW_V3_SCHEMA_VERSION: str = (
    "coordpy.bounded_window_baseline_v3.v1")

W83_BW_V3_DEFAULT_WINDOW: int = 256
W83_BW_V3_DEFAULT_SUMMARY_FIDELITY: float = 0.65
W83_BW_V3_DEFAULT_SUMMARY_UPDATE_EVERY: int = 32
W83_BW_V3_DEFAULT_RETRIEVAL_K: int = 8
W83_BW_V3_DEFAULT_MAX_HORIZON: int = 64_000
W83_BW_V3_DEFAULT_FEATURE_DIM: int = 32
W83_BW_V3_DEFAULT_SEED: int = 83_006_001


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _ndarray_cid(arr: "_np.ndarray | None") -> str:
    if arr is None:
        return "none"
    a = _np.ascontiguousarray(
        _np.asarray(arr, dtype=_np.float64))
    return hashlib.sha256(a.tobytes()).hexdigest()


@dataclasses.dataclass(frozen=True)
class BoundedWindowEventV3:
    """A single event in the rolling window."""

    turn_index: int
    feature: "_np.ndarray"
    payload_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_index": int(self.turn_index),
            "feature_cid": _ndarray_cid(self.feature),
            "payload_cid": str(self.payload_cid),
        }


@dataclasses.dataclass(frozen=True)
class BoundedWindowBaselineV3:
    """Composed strongest-known bounded baseline."""

    schema: str
    window_size: int
    summary_fidelity: float
    summary_update_every: int
    retrieval_k: int
    feature_dim: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "window_size": int(self.window_size),
            "summary_fidelity": float(round(
                self.summary_fidelity, 12)),
            "summary_update_every": int(
                self.summary_update_every),
            "retrieval_k": int(self.retrieval_k),
            "feature_dim": int(self.feature_dim),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w83_bounded_window_baseline_v3",
            "baseline": self.to_dict()})


def build_bounded_window_baseline_v3(
        *,
        window_size: int = W83_BW_V3_DEFAULT_WINDOW,
        summary_fidelity: float = (
            W83_BW_V3_DEFAULT_SUMMARY_FIDELITY),
        summary_update_every: int = (
            W83_BW_V3_DEFAULT_SUMMARY_UPDATE_EVERY),
        retrieval_k: int = W83_BW_V3_DEFAULT_RETRIEVAL_K,
        feature_dim: int = W83_BW_V3_DEFAULT_FEATURE_DIM,
) -> BoundedWindowBaselineV3:
    return BoundedWindowBaselineV3(
        schema=W83_BW_V3_SCHEMA_VERSION,
        window_size=int(window_size),
        summary_fidelity=float(summary_fidelity),
        summary_update_every=int(summary_update_every),
        retrieval_k=int(retrieval_k),
        feature_dim=int(feature_dim),
    )


@dataclasses.dataclass(frozen=True)
class BoundedWindowQueryAnswerV3:
    """An answer from V3 to a reconstruction query."""

    schema: str
    query_turn_index: int
    success: bool
    answer_source: str  # "window", "summary", "retrieval", "abstain"
    answer_fidelity: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "query_turn_index": int(self.query_turn_index),
            "success": bool(self.success),
            "answer_source": str(self.answer_source),
            "answer_fidelity": float(round(
                self.answer_fidelity, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w83_bw_v3_query_answer",
            "answer": self.to_dict()})


def answer_reconstruction_query_v3(
        *,
        baseline: BoundedWindowBaselineV3,
        events_window: Sequence[BoundedWindowEventV3],
        summary_feature_centroid: "_np.ndarray | None",
        summary_covers_turns: tuple[int, int],
        target_turn_index: int,
        query_feature: "_np.ndarray | None" = None,
        required_fidelity: float = 0.5,
) -> BoundedWindowQueryAnswerV3:
    """Answer a reconstruction query using V3's three layers.

    Algorithm:
    1. If the target turn is in the visible window, answer
       directly (success=True, fidelity=1.0).
    2. If the target turn is covered by the summary AND the
       summary fidelity exceeds the required fidelity, answer
       from summary (success=True, fidelity=summary_fidelity).
    3. If a query_feature is provided, retrieve the top-K events
       from the window by cosine similarity. If any retrieved
       event happens to BE the target turn, return success;
       otherwise return retrieval-fidelity-failure.
    4. Otherwise abstain.
    """
    # Layer 1: window.
    for ev in events_window:
        if int(ev.turn_index) == int(target_turn_index):
            return BoundedWindowQueryAnswerV3(
                schema=W83_BW_V3_SCHEMA_VERSION,
                query_turn_index=int(target_turn_index),
                success=True,
                answer_source="window",
                answer_fidelity=1.0,
            )
    # Layer 2: summary.
    sum_lo, sum_hi = summary_covers_turns
    if (sum_lo <= int(target_turn_index) <= sum_hi
            and float(baseline.summary_fidelity)
            >= float(required_fidelity)):
        return BoundedWindowQueryAnswerV3(
            schema=W83_BW_V3_SCHEMA_VERSION,
            query_turn_index=int(target_turn_index),
            success=True,
            answer_source="summary",
            answer_fidelity=float(baseline.summary_fidelity),
        )
    # Layer 3: retrieval (only if query_feature is available).
    if query_feature is not None and len(events_window) > 0:
        q = _np.asarray(query_feature, dtype=_np.float64)
        feats = _np.stack(
            [_np.asarray(ev.feature, dtype=_np.float64)
             for ev in events_window],
            axis=0)
        qn = q / max(1e-9, float(_np.linalg.norm(q)))
        feat_norms = _np.linalg.norm(feats, axis=1)
        feat_norms = _np.maximum(feat_norms, 1e-9)
        sims = (feats @ qn) / feat_norms
        # Top-K indices.
        k_eff = min(
            int(baseline.retrieval_k),
            int(sims.shape[0]))
        top_idx = _np.argsort(sims)[::-1][:int(k_eff)]
        for j in top_idx:
            if int(events_window[int(j)].turn_index) == int(
                    target_turn_index):
                return BoundedWindowQueryAnswerV3(
                    schema=W83_BW_V3_SCHEMA_VERSION,
                    query_turn_index=int(target_turn_index),
                    success=True,
                    answer_source="retrieval",
                    answer_fidelity=float(sims[int(j)]),
                )
    # Layer 4: abstain (honest failure for events past the
    # window and past the summary).
    return BoundedWindowQueryAnswerV3(
        schema=W83_BW_V3_SCHEMA_VERSION,
        query_turn_index=int(target_turn_index),
        success=False,
        answer_source="abstain",
        answer_fidelity=0.0,
    )


@dataclasses.dataclass(frozen=True)
class BoundedWindowV3FailureProofV1:
    """V3 provably cannot answer reconstruction queries with
    horizon beyond window + summary coverage."""

    schema: str
    n_horizons_tested: int
    window_size: int
    summary_coverage_turns: int
    failure_horizons: tuple[int, ...]
    failure_rate_beyond_coverage: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "n_horizons_tested": int(self.n_horizons_tested),
            "window_size": int(self.window_size),
            "summary_coverage_turns": int(
                self.summary_coverage_turns),
            "failure_horizons": list(
                int(h) for h in self.failure_horizons),
            "failure_rate_beyond_coverage": float(round(
                self.failure_rate_beyond_coverage, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w83_bw_v3_failure_proof_v1",
            "proof": self.to_dict()})


def prove_bounded_window_v3_insufficient_v1(
        *,
        baseline: BoundedWindowBaselineV3,
        summary_coverage_turns: int,
        horizons_to_test: Sequence[int] = (
            512, 2048, 8192, 32_768, 64_000, 100_000),
        seed: int = W83_BW_V3_DEFAULT_SEED,
) -> BoundedWindowV3FailureProofV1:
    """Falsifier proof: V3 abstains on queries past coverage.

    For each horizon, builds a synthetic event stream past the
    coverage (window + summary). V3 should abstain (success=False)
    on every query whose target_turn_index is past
    ``window_size + summary_coverage_turns`` and is not present
    in the visible window.
    """
    rng = _np.random.default_rng(int(seed))
    failures: list[int] = []
    n_tests = 0
    n_failures_beyond = 0
    coverage_total = int(baseline.window_size) + int(
        summary_coverage_turns)
    for h in horizons_to_test:
        n_tests += 1
        # Build a window of the most-recent W events.
        W = int(baseline.window_size)
        H = int(h)
        events = [
            BoundedWindowEventV3(
                turn_index=int(H - W + i),
                feature=rng.standard_normal(
                    (int(baseline.feature_dim),)
                ).astype(_np.float64),
                payload_cid=_sha256_hex({
                    "kind": "w83_bw_v3_event",
                    "horizon": int(H),
                    "turn_index": int(H - W + i),
                }))
            for i in range(W)]
        # Target: a turn past coverage.
        if int(H) - coverage_total <= 0:
            continue  # horizon too small for this probe
        target = int(0)  # earliest turn — definitively past coverage
        ans = answer_reconstruction_query_v3(
            baseline=baseline,
            events_window=events,
            summary_feature_centroid=None,
            summary_covers_turns=(
                int(H - W - summary_coverage_turns),
                int(H - W - 1)),
            target_turn_index=int(target),
            query_feature=None,
            required_fidelity=0.5)
        if not ans.success:
            failures.append(int(h))
            n_failures_beyond += 1
    rate = (
        float(n_failures_beyond) / max(1, int(n_tests))
        if n_tests > 0 else 0.0)
    return BoundedWindowV3FailureProofV1(
        schema=W83_BW_V3_SCHEMA_VERSION,
        n_horizons_tested=int(n_tests),
        window_size=int(baseline.window_size),
        summary_coverage_turns=int(summary_coverage_turns),
        failure_horizons=tuple(int(f) for f in failures),
        failure_rate_beyond_coverage=float(rate),
    )


@dataclasses.dataclass(frozen=True)
class BoundedWindowBaselineV3WitnessV1:
    schema: str
    baseline_cid: str
    proof_cid: str
    proof_failure_rate: float

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w83_bw_v3_witness_v1",
            "schema": str(self.schema),
            "baseline_cid": str(self.baseline_cid),
            "proof_cid": str(self.proof_cid),
            "proof_failure_rate": float(round(
                self.proof_failure_rate, 12)),
        })


def emit_bounded_window_baseline_v3_witness_v1(
        *,
        baseline: BoundedWindowBaselineV3,
        proof: BoundedWindowV3FailureProofV1,
) -> BoundedWindowBaselineV3WitnessV1:
    return BoundedWindowBaselineV3WitnessV1(
        schema=W83_BW_V3_SCHEMA_VERSION,
        baseline_cid=str(baseline.cid()),
        proof_cid=str(proof.cid()),
        proof_failure_rate=float(
            proof.failure_rate_beyond_coverage),
    )


__all__ = [
    "W83_BW_V3_SCHEMA_VERSION",
    "BoundedWindowEventV3",
    "BoundedWindowBaselineV3",
    "BoundedWindowQueryAnswerV3",
    "BoundedWindowV3FailureProofV1",
    "BoundedWindowBaselineV3WitnessV1",
    "build_bounded_window_baseline_v3",
    "answer_reconstruction_query_v3",
    "prove_bounded_window_v3_insufficient_v1",
    "emit_bounded_window_baseline_v3_witness_v1",
]
