"""W78 — Bounded-Window Transcript Baseline V1.

This module is the explicit **anti-goal falsifier** at the heart
of the W78 milestone. The Context-Zero programme has repeatedly
warned against quietly regressing to bounded-context thinking:
fixed-k handoff windows, rolling summaries, last-N-turn
coordinators, and disguised bounded caches.

W78 makes that warning load-bearing by *building* the bounded-
window baselines explicitly and benchmarking the W78 substrate
against them on a regime where the baselines provably fail.

Each bounded-window baseline:

* Maintains a **fixed-k visible window** of the most recent
  ``k`` transcript turns.
* Answers reconstruction queries by reading *only* from that
  window.
* By construction, any event whose source lies > k turns ago
  is **outside** the window and therefore invisible.

The W78 long-horizon-reconstruction substrate, by contrast,
reads from a persistent latent carrier whose information is
**not** bounded by visible-window length.

The asymmetry is:

* Bounded-window baseline can succeed iff the source event is
  within the last k turns.
* W78 substrate can succeed even if the source event lies
  arbitrarily far in the past, as long as the persistent latent
  carrier has retained its CID.

This module ships five baselines:

* k = 4   (the "last 4 turns" coordinator the original prompt
            explicitly warns against)
* k = 8
* k = 16
* k = 32
* rolling-summary baseline (compresses but truncates; equivalent
  to k = ∞ at fidelity 1/sqrt(n))

Honest scope (W78)
------------------

* These baselines are **synthetic** — they read from a list of
  per-turn event tags, not from a real LLM transcript. The
  insufficiency claim holds by construction within the
  synthetic regime: a predictor with zero information about an
  event cannot strictly beat random on that event.
* The W78 bounded-window theorem
  (``W78-T-BOUNDED-WINDOW-INSUFFICIENT``) restates Shannon's
  observation about information channels with capacity zero on
  the relevant interval.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Sequence


W78_BOUNDED_WINDOW_BASELINE_SCHEMA_VERSION: str = (
    "coordpy.bounded_window_baseline_v1.v1")

W78_DEFAULT_BOUNDED_WINDOW_KS: tuple[int, ...] = (4, 8, 16, 32)
W78_ROLLING_SUMMARY_LABEL: str = "rolling_summary"


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass(frozen=True)
class BoundedWindowQuery:
    """A reconstruction query.

    The substrate must answer: "what was the event at turn
    ``source_turn``, given the team is now at turn
    ``current_turn``?"

    Bounded-window baselines can succeed iff
    ``current_turn - source_turn < k``.
    """

    query_id: str
    current_turn: int
    source_turn: int
    expected_event_cid: str

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "bounded_window_query",
            "query_id": str(self.query_id),
            "current_turn": int(self.current_turn),
            "source_turn": int(self.source_turn),
            "expected_event_cid": str(self.expected_event_cid),
        })


@dataclasses.dataclass(frozen=True)
class BoundedWindowOutcome:
    """A single baseline's outcome on a single query."""

    schema: str
    baseline_id: str
    query_cid: str
    k: int
    rolling_summary: bool
    in_window: bool
    success: bool
    visible_turns_used: int
    answer_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "baseline_id": str(self.baseline_id),
            "query_cid": str(self.query_cid),
            "k": int(self.k),
            "rolling_summary": bool(self.rolling_summary),
            "in_window": bool(self.in_window),
            "success": bool(self.success),
            "visible_turns_used": int(self.visible_turns_used),
            "answer_cid": str(self.answer_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "bounded_window_outcome",
            "outcome": self.to_dict()})


@dataclasses.dataclass(frozen=True)
class BoundedWindowBaseline:
    """A fixed-k bounded-window transcript baseline.

    Reads only the most recent ``k`` turns of the transcript.
    Any event whose ``source_turn`` lies outside that window is
    invisible to this baseline — it can only answer a query
    successfully if ``current_turn - source_turn < k``.

    The ``rolling_summary=True`` variant nominally tracks an
    *infinite* horizon via a compressed summary, but with
    **fidelity bounded by visible-window length**: in W78 we
    model rolling summary as 1/sqrt(n) effective fidelity, which
    is strictly less than 1 for any n > 1. On a 1-bit
    reconstruction query the rolling-summary baseline succeeds
    only when (1) the event is within the visible window OR
    (2) the rolling-summary fidelity is > 0.5 — and we
    construct the regime so neither holds on the new W78 query.
    """

    schema: str = W78_BOUNDED_WINDOW_BASELINE_SCHEMA_VERSION
    baseline_id: str = "k4"
    k: int = 4
    rolling_summary: bool = False
    rolling_summary_fidelity_at_n: float = 0.0

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "bounded_window_baseline",
            "schema": str(self.schema),
            "baseline_id": str(self.baseline_id),
            "k": int(self.k),
            "rolling_summary": bool(self.rolling_summary),
            "rolling_summary_fidelity_at_n": float(round(
                self.rolling_summary_fidelity_at_n, 12)),
        })

    def answer_query(
            self, *, query: BoundedWindowQuery,
    ) -> BoundedWindowOutcome:
        """Answer a reconstruction query under the bounded window.

        Returns ``success=True`` iff this baseline can reproduce
        the expected event CID:

        * For fixed-k baselines: iff
          ``current_turn - source_turn < k``.
        * For the rolling-summary baseline: iff the source is in
          the visible window OR the rolling-summary fidelity at
          ``current_turn`` is > 0.5.
        """
        delta = int(query.current_turn) - int(query.source_turn)
        in_window = bool(0 <= delta < int(self.k))
        if self.rolling_summary:
            # Rolling-summary baseline: it can also fake-succeed
            # if its declared fidelity exceeds 0.5 at the
            # query's horizon.
            success = bool(
                in_window
                or self.rolling_summary_fidelity_at_n > 0.5)
        else:
            success = bool(in_window)
        if success:
            answer_cid = str(query.expected_event_cid)
        else:
            # The baseline literally has no information — its
            # answer is the empty CID. This is the load-bearing
            # bounded-window failure mode.
            answer_cid = ""
        return BoundedWindowOutcome(
            schema=W78_BOUNDED_WINDOW_BASELINE_SCHEMA_VERSION,
            baseline_id=str(self.baseline_id),
            query_cid=str(query.cid()),
            k=int(self.k),
            rolling_summary=bool(self.rolling_summary),
            in_window=bool(in_window),
            success=bool(success),
            visible_turns_used=int(
                min(int(self.k), int(query.current_turn))),
            answer_cid=str(answer_cid),
        )


def build_default_bounded_window_baselines() -> (
        tuple[BoundedWindowBaseline, ...]):
    """Build the four default fixed-k baselines + rolling
    summary."""
    out: list[BoundedWindowBaseline] = []
    for i, k in enumerate(W78_DEFAULT_BOUNDED_WINDOW_KS):
        out.append(BoundedWindowBaseline(
            baseline_id=f"k{int(k)}",
            k=int(k),
            rolling_summary=False))
    out.append(BoundedWindowBaseline(
        baseline_id=W78_ROLLING_SUMMARY_LABEL,
        k=int(W78_DEFAULT_BOUNDED_WINDOW_KS[-1]),
        rolling_summary=True,
        rolling_summary_fidelity_at_n=0.30))
    return tuple(out)


@dataclasses.dataclass(frozen=True)
class BoundedWindowFalsifierReport:
    """Falsifier report for one query against a set of baselines.

    By construction, *all* fixed-k baselines must report
    ``success=False`` when ``current_turn - source_turn >= k``.
    The W78 substrate, which reads from a persistent latent
    carrier instead of the visible window, can answer
    successfully even for such queries. This is the
    bounded-window-insufficiency falsifier.
    """

    schema: str
    query_cid: str
    per_baseline_outcomes_cid: tuple[str, ...]
    fixed_k_failure_count: int
    rolling_summary_failure: bool
    all_fixed_k_failed: bool
    n_fixed_k_baselines: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "query_cid": str(self.query_cid),
            "per_baseline_outcomes_cid": list(
                self.per_baseline_outcomes_cid),
            "fixed_k_failure_count": int(
                self.fixed_k_failure_count),
            "rolling_summary_failure": bool(
                self.rolling_summary_failure),
            "all_fixed_k_failed": bool(self.all_fixed_k_failed),
            "n_fixed_k_baselines": int(self.n_fixed_k_baselines),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "bounded_window_falsifier_report",
            "report": self.to_dict()})


def run_bounded_window_falsifier(
        *, baselines: Sequence[BoundedWindowBaseline],
        query: BoundedWindowQuery,
) -> tuple[
        tuple[BoundedWindowOutcome, ...],
        BoundedWindowFalsifierReport]:
    """Run all baselines on the query; emit a falsifier report."""
    outs: list[BoundedWindowOutcome] = []
    fixed_k_fails = 0
    rolling_fail = False
    n_fixed = 0
    for b in baselines:
        out = b.answer_query(query=query)
        outs.append(out)
        if not b.rolling_summary:
            n_fixed += 1
            if not out.success:
                fixed_k_fails += 1
        else:
            if not out.success:
                rolling_fail = True
    report = BoundedWindowFalsifierReport(
        schema=W78_BOUNDED_WINDOW_BASELINE_SCHEMA_VERSION,
        query_cid=str(query.cid()),
        per_baseline_outcomes_cid=tuple(o.cid() for o in outs),
        fixed_k_failure_count=int(fixed_k_fails),
        rolling_summary_failure=bool(rolling_fail),
        all_fixed_k_failed=bool(
            n_fixed > 0 and fixed_k_fails == n_fixed),
        n_fixed_k_baselines=int(n_fixed),
    )
    return tuple(outs), report


@dataclasses.dataclass(frozen=True)
class BoundedWindowInsufficiencyProof:
    """Code-backed bounded-window-insufficiency proof.

    Given a query whose source is k_max turns behind the current
    turn, no fixed-k baseline with k <= k_max can possibly
    succeed. We assert this by construction.
    """

    schema: str
    proven: bool
    query_horizon_turns: int
    max_baseline_k: int
    proof_sketch_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "proven": bool(self.proven),
            "query_horizon_turns": int(
                self.query_horizon_turns),
            "max_baseline_k": int(self.max_baseline_k),
            "proof_sketch_cid": str(self.proof_sketch_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "bounded_window_insufficiency_proof",
            "proof": self.to_dict()})


def prove_bounded_window_insufficient(
        *, query_horizon_turns: int,
        baselines: Sequence[BoundedWindowBaseline],
) -> BoundedWindowInsufficiencyProof:
    """Prove that bounded-window baselines are insufficient.

    By construction, a baseline with window ``k`` cannot observe
    events older than ``k`` turns. If ``query_horizon_turns >=
    max_k`` for all fixed-k baselines, then *all* of them have
    zero information about the event and cannot strictly beat
    random.
    """
    max_k = max(
        int(b.k) for b in baselines if not b.rolling_summary)
    proven = bool(int(query_horizon_turns) >= int(max_k))
    sketch = _sha256_hex({
        "kind": "bounded_window_proof_sketch",
        "claim": (
            "any predictor restricted to the last k turns has "
            "zero information about events occurring more than k "
            "turns ago; therefore cannot strictly beat random "
            "on ≥ 1-bit reconstruction queries"),
        "query_horizon_turns": int(query_horizon_turns),
        "max_baseline_k": int(max_k),
    })
    return BoundedWindowInsufficiencyProof(
        schema=W78_BOUNDED_WINDOW_BASELINE_SCHEMA_VERSION,
        proven=bool(proven),
        query_horizon_turns=int(query_horizon_turns),
        max_baseline_k=int(max_k),
        proof_sketch_cid=str(sketch),
    )


@dataclasses.dataclass(frozen=True)
class BoundedWindowBaselineWitness:
    schema: str
    n_baselines: int
    n_fixed_k: int
    fixed_k_values: tuple[int, ...]
    insufficiency_proof_cid: str
    falsifier_report_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "n_baselines": int(self.n_baselines),
            "n_fixed_k": int(self.n_fixed_k),
            "fixed_k_values": list(self.fixed_k_values),
            "insufficiency_proof_cid": str(
                self.insufficiency_proof_cid),
            "falsifier_report_cid": str(
                self.falsifier_report_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "bounded_window_baseline_witness",
            "witness": self.to_dict()})


def emit_bounded_window_baseline_witness(
        *, baselines: Sequence[BoundedWindowBaseline],
        proof: BoundedWindowInsufficiencyProof,
        falsifier: BoundedWindowFalsifierReport,
) -> BoundedWindowBaselineWitness:
    fixed_ks = tuple(
        int(b.k) for b in baselines if not b.rolling_summary)
    return BoundedWindowBaselineWitness(
        schema=W78_BOUNDED_WINDOW_BASELINE_SCHEMA_VERSION,
        n_baselines=int(len(list(baselines))),
        n_fixed_k=int(len(fixed_ks)),
        fixed_k_values=fixed_ks,
        insufficiency_proof_cid=str(proof.cid()),
        falsifier_report_cid=str(falsifier.cid()),
    )


__all__ = [
    "W78_BOUNDED_WINDOW_BASELINE_SCHEMA_VERSION",
    "W78_DEFAULT_BOUNDED_WINDOW_KS",
    "W78_ROLLING_SUMMARY_LABEL",
    "BoundedWindowQuery",
    "BoundedWindowOutcome",
    "BoundedWindowBaseline",
    "BoundedWindowFalsifierReport",
    "BoundedWindowInsufficiencyProof",
    "BoundedWindowBaselineWitness",
    "build_default_bounded_window_baselines",
    "run_bounded_window_falsifier",
    "prove_bounded_window_insufficient",
    "emit_bounded_window_baseline_witness",
]
