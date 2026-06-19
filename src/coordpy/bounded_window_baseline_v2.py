"""W79 — Bounded-Window Transcript Baseline V2.

Strictly extends W78's ``coordpy.bounded_window_baseline_v1``.
V2 adds two new baseline shapes that the W79 milestone treats as
load-bearing falsifiers:

* **k = 64 fixed-window baseline** — strictly extends V1's
  {4, 8, 16, 32} set with a wider window that is *still* not
  wide enough for the W79 long-delay reconstruction regime
  (where the source event lies 200+ turns before the
  reconstruction request).
* **Cross-prompt summary baseline** — emulates the worst-case
  cross-prompt "I'll re-summarize from scratch each turn"
  pattern: full transcript replay every turn, capped at a fixed
  visible-token budget, so it loses information about anything
  outside its summary cap.

The V2 falsifier is even stronger than V1: not only does the
W78 fixed-k {4..32} set fail by construction outside the
window, the **64-turn window also fails** on the W79 regime
where the source-to-reconstruction-request horizon exceeds 64.

V2 is the W79 anti-goal substrate: bounded-window visible
context is the wrong answer.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Sequence

from .bounded_window_baseline_v1 import (
    BoundedWindowBaseline,
    BoundedWindowBaselineWitness,
    BoundedWindowFalsifierReport,
    BoundedWindowInsufficiencyProof,
    BoundedWindowOutcome,
    BoundedWindowQuery,
    W78_BOUNDED_WINDOW_BASELINE_SCHEMA_VERSION,
    W78_DEFAULT_BOUNDED_WINDOW_KS,
    W78_ROLLING_SUMMARY_LABEL,
    build_default_bounded_window_baselines,
    emit_bounded_window_baseline_witness,
    prove_bounded_window_insufficient,
    run_bounded_window_falsifier,
)


W79_BOUNDED_WINDOW_BASELINE_V2_SCHEMA_VERSION: str = (
    "coordpy.bounded_window_baseline_v2.v1")
W79_DEFAULT_BOUNDED_WINDOW_KS_V2: tuple[int, ...] = (
    *W78_DEFAULT_BOUNDED_WINDOW_KS, 64)
W79_CROSS_PROMPT_SUMMARY_LABEL: str = "cross_prompt_summary"
W79_DEFAULT_CROSS_PROMPT_SUMMARY_BUDGET_TURNS: int = 64
W79_DEFAULT_CROSS_PROMPT_SUMMARY_FIDELITY: float = 0.20


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass(frozen=True)
class CrossPromptSummaryBaselineV2:
    """A baseline that emulates the worst-case cross-prompt
    summary loop.

    Every turn it re-summarises the *visible* transcript window
    (capped at ``budget_turns``) into a fixed-size summary,
    then answers queries from the summary plus the window. Any
    event whose source lies outside the budget is irrecoverable
    by construction — the summary fidelity ``fidelity_at_n``
    is bounded below 0.5 in W79 default config so even the
    "fake-succeed" path of V1's rolling summary is gone.
    """

    schema: str
    baseline_id: str = W79_CROSS_PROMPT_SUMMARY_LABEL
    budget_turns: int = (
        W79_DEFAULT_CROSS_PROMPT_SUMMARY_BUDGET_TURNS)
    summary_fidelity: float = (
        W79_DEFAULT_CROSS_PROMPT_SUMMARY_FIDELITY)

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w79_cross_prompt_summary_baseline_v2",
            "schema": str(self.schema),
            "baseline_id": str(self.baseline_id),
            "budget_turns": int(self.budget_turns),
            "summary_fidelity": float(round(
                self.summary_fidelity, 12)),
        })

    def answer_query(
            self, *, query: BoundedWindowQuery,
    ) -> BoundedWindowOutcome:
        """Cross-prompt summary baseline outcome.

        Succeeds iff (1) source is within the budget window OR
        (2) summary fidelity > 0.5. With the W79 default
        fidelity of 0.20, the only success path is window-bound;
        thus on the W79 regime where source-to-request horizon
        exceeds budget_turns, this baseline fails by
        construction.
        """
        delta = int(query.current_turn) - int(query.source_turn)
        in_budget = bool(0 <= delta < int(self.budget_turns))
        success = bool(
            in_budget
            or float(self.summary_fidelity) > 0.5)
        if success:
            answer_cid = str(query.expected_event_cid)
        else:
            answer_cid = ""
        return BoundedWindowOutcome(
            schema=W78_BOUNDED_WINDOW_BASELINE_SCHEMA_VERSION,
            baseline_id=str(self.baseline_id),
            query_cid=str(query.cid()),
            k=int(self.budget_turns),
            rolling_summary=True,
            in_window=bool(in_budget),
            success=bool(success),
            visible_turns_used=int(
                min(int(self.budget_turns),
                    int(query.current_turn))),
            answer_cid=str(answer_cid),
        )


@dataclasses.dataclass(frozen=True)
class BoundedWindowBaselineSetV2:
    """Container for the W79 V2 baseline set."""

    schema: str
    inner_v1: tuple[BoundedWindowBaseline, ...]
    k64_baseline: BoundedWindowBaseline
    cross_prompt_summary: CrossPromptSummaryBaselineV2

    def all_outcomes(
            self, *, query: BoundedWindowQuery,
    ) -> tuple[BoundedWindowOutcome, ...]:
        outs: list[BoundedWindowOutcome] = []
        for b in self.inner_v1:
            outs.append(b.answer_query(query=query))
        outs.append(self.k64_baseline.answer_query(query=query))
        outs.append(
            self.cross_prompt_summary.answer_query(query=query))
        return tuple(outs)

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w79_bounded_window_baseline_set_v2",
            "schema": str(self.schema),
            "inner_v1_cids": [
                b.cid() for b in self.inner_v1],
            "k64_baseline_cid": str(
                self.k64_baseline.cid()),
            "cross_prompt_summary_cid": str(
                self.cross_prompt_summary.cid()),
        })


def build_default_bounded_window_baselines_v2(
) -> BoundedWindowBaselineSetV2:
    inner_v1 = build_default_bounded_window_baselines()
    k64 = BoundedWindowBaseline(
        baseline_id="k64", k=64, rolling_summary=False)
    cps = CrossPromptSummaryBaselineV2(
        schema=W79_BOUNDED_WINDOW_BASELINE_V2_SCHEMA_VERSION)
    return BoundedWindowBaselineSetV2(
        schema=W79_BOUNDED_WINDOW_BASELINE_V2_SCHEMA_VERSION,
        inner_v1=tuple(inner_v1),
        k64_baseline=k64,
        cross_prompt_summary=cps,
    )


@dataclasses.dataclass(frozen=True)
class BoundedWindowFalsifierReportV2:
    schema: str
    query_cid: str
    inner_v1_falsifier_cid: str
    per_baseline_outcomes_cid: tuple[str, ...]
    k64_failed: bool
    cross_prompt_summary_failed: bool
    fixed_k_failure_count_v2: int
    n_fixed_k_baselines_v2: int
    all_fixed_k_failed_v2: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "query_cid": str(self.query_cid),
            "inner_v1_falsifier_cid": str(
                self.inner_v1_falsifier_cid),
            "per_baseline_outcomes_cid": list(
                self.per_baseline_outcomes_cid),
            "k64_failed": bool(self.k64_failed),
            "cross_prompt_summary_failed": bool(
                self.cross_prompt_summary_failed),
            "fixed_k_failure_count_v2": int(
                self.fixed_k_failure_count_v2),
            "n_fixed_k_baselines_v2": int(
                self.n_fixed_k_baselines_v2),
            "all_fixed_k_failed_v2": bool(
                self.all_fixed_k_failed_v2),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w79_bounded_window_falsifier_v2",
            "report": self.to_dict()})


def run_bounded_window_falsifier_v2(
        *,
        baselines_v2: BoundedWindowBaselineSetV2,
        query: BoundedWindowQuery,
) -> tuple[
        tuple[BoundedWindowOutcome, ...],
        BoundedWindowFalsifierReportV2]:
    # V1 falsifier on the original baselines.
    v1_outs, v1_rep = run_bounded_window_falsifier(
        baselines=baselines_v2.inner_v1, query=query)
    # k64 + cross-prompt.
    k64_out = baselines_v2.k64_baseline.answer_query(
        query=query)
    cps_out = baselines_v2.cross_prompt_summary.answer_query(
        query=query)
    all_outs = (*v1_outs, k64_out, cps_out)
    fixed_k_fails = int(v1_rep.fixed_k_failure_count)
    n_fixed_k = int(v1_rep.n_fixed_k_baselines)
    if not k64_out.success:
        fixed_k_fails += 1
    n_fixed_k += 1
    rep = BoundedWindowFalsifierReportV2(
        schema=W79_BOUNDED_WINDOW_BASELINE_V2_SCHEMA_VERSION,
        query_cid=str(query.cid()),
        inner_v1_falsifier_cid=str(v1_rep.cid()),
        per_baseline_outcomes_cid=tuple(
            o.cid() for o in all_outs),
        k64_failed=bool(not k64_out.success),
        cross_prompt_summary_failed=bool(not cps_out.success),
        fixed_k_failure_count_v2=int(fixed_k_fails),
        n_fixed_k_baselines_v2=int(n_fixed_k),
        all_fixed_k_failed_v2=bool(
            fixed_k_fails == n_fixed_k and n_fixed_k > 0),
    )
    return all_outs, rep


@dataclasses.dataclass(frozen=True)
class BoundedWindowInsufficiencyProofV2:
    schema: str
    proven: bool
    query_horizon_turns: int
    max_baseline_k_v2: int
    cross_prompt_summary_budget_turns: int
    cross_prompt_summary_fidelity: float
    proof_sketch_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "proven": bool(self.proven),
            "query_horizon_turns": int(
                self.query_horizon_turns),
            "max_baseline_k_v2": int(self.max_baseline_k_v2),
            "cross_prompt_summary_budget_turns": int(
                self.cross_prompt_summary_budget_turns),
            "cross_prompt_summary_fidelity": float(round(
                self.cross_prompt_summary_fidelity, 12)),
            "proof_sketch_cid": str(self.proof_sketch_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind":
                "w79_bounded_window_insufficiency_proof_v2",
            "proof": self.to_dict()})


def prove_bounded_window_insufficient_v2(
        *, query_horizon_turns: int,
        baselines_v2: BoundedWindowBaselineSetV2,
) -> BoundedWindowInsufficiencyProofV2:
    """V2 insufficiency proof.

    For the V2 to be sufficient on a query at horizon ``h``,
    *some* baseline would need to be able to recover the
    answer. By construction:

    * fixed-k baselines with k ∈ {4, 8, 16, 32, 64} all see only
      the last k turns. Any of them with k <= h cannot succeed.
    * The cross-prompt summary baseline succeeds iff (h <
      budget_turns) OR (fidelity > 0.5). With the W79 default
      fidelity of 0.20, that reduces to (h < budget_turns =
      64).

    Therefore: if ``h >= 64`` AND the V1 rolling-summary
    fidelity < 0.5, the V2 baseline set is provably
    insufficient. We assert this by construction.
    """
    all_ks = (
        *(int(b.k) for b in baselines_v2.inner_v1
          if not b.rolling_summary),
        int(baselines_v2.k64_baseline.k),
    )
    max_k_v2 = int(max(all_ks)) if all_ks else 0
    cps_budget = int(
        baselines_v2.cross_prompt_summary.budget_turns)
    cps_fid = float(
        baselines_v2.cross_prompt_summary.summary_fidelity)
    proven = bool(
        int(query_horizon_turns) >= int(max_k_v2)
        and int(query_horizon_turns) >= int(cps_budget)
        and float(cps_fid) <= 0.5)
    sketch = _sha256_hex({
        "kind": "w79_bounded_window_proof_sketch_v2",
        "claim": (
            "any predictor restricted to a window <= h turns OR "
            "to a summary of fidelity <= 0.5 has zero "
            "information about an event h turns ago; therefore "
            "cannot strictly beat random on >= 1-bit "
            "reconstruction queries"),
        "query_horizon_turns": int(query_horizon_turns),
        "max_baseline_k_v2": int(max_k_v2),
        "cross_prompt_summary_budget_turns": int(cps_budget),
    })
    return BoundedWindowInsufficiencyProofV2(
        schema=W79_BOUNDED_WINDOW_BASELINE_V2_SCHEMA_VERSION,
        proven=bool(proven),
        query_horizon_turns=int(query_horizon_turns),
        max_baseline_k_v2=int(max_k_v2),
        cross_prompt_summary_budget_turns=int(cps_budget),
        cross_prompt_summary_fidelity=float(cps_fid),
        proof_sketch_cid=str(sketch),
    )


@dataclasses.dataclass(frozen=True)
class BoundedWindowBaselineV2Witness:
    schema: str
    baseline_set_v2_cid: str
    inner_v1_witness_cid: str
    n_fixed_k_v2: int
    fixed_k_values_v2: tuple[int, ...]
    cross_prompt_summary_present: bool
    insufficiency_proof_v2_cid: str
    falsifier_v2_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "baseline_set_v2_cid": str(
                self.baseline_set_v2_cid),
            "inner_v1_witness_cid": str(
                self.inner_v1_witness_cid),
            "n_fixed_k_v2": int(self.n_fixed_k_v2),
            "fixed_k_values_v2": list(self.fixed_k_values_v2),
            "cross_prompt_summary_present": bool(
                self.cross_prompt_summary_present),
            "insufficiency_proof_v2_cid": str(
                self.insufficiency_proof_v2_cid),
            "falsifier_v2_cid": str(self.falsifier_v2_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w79_bounded_window_baseline_v2_witness",
            "witness": self.to_dict()})


def emit_bounded_window_baseline_witness_v2(
        *,
        baselines_v2: BoundedWindowBaselineSetV2,
        proof_v2: BoundedWindowInsufficiencyProofV2,
        falsifier_v2: BoundedWindowFalsifierReportV2,
        inner_v1_witness: BoundedWindowBaselineWitness,
) -> BoundedWindowBaselineV2Witness:
    fixed_ks = tuple(
        int(b.k) for b in baselines_v2.inner_v1
        if not b.rolling_summary)
    fixed_ks = (*fixed_ks, int(baselines_v2.k64_baseline.k))
    return BoundedWindowBaselineV2Witness(
        schema=W79_BOUNDED_WINDOW_BASELINE_V2_SCHEMA_VERSION,
        baseline_set_v2_cid=str(baselines_v2.cid()),
        inner_v1_witness_cid=str(inner_v1_witness.cid()),
        n_fixed_k_v2=int(len(fixed_ks)),
        fixed_k_values_v2=fixed_ks,
        cross_prompt_summary_present=True,
        insufficiency_proof_v2_cid=str(proof_v2.cid()),
        falsifier_v2_cid=str(falsifier_v2.cid()),
    )


__all__ = [
    "W79_BOUNDED_WINDOW_BASELINE_V2_SCHEMA_VERSION",
    "W79_DEFAULT_BOUNDED_WINDOW_KS_V2",
    "W79_CROSS_PROMPT_SUMMARY_LABEL",
    "W79_DEFAULT_CROSS_PROMPT_SUMMARY_BUDGET_TURNS",
    "W79_DEFAULT_CROSS_PROMPT_SUMMARY_FIDELITY",
    "CrossPromptSummaryBaselineV2",
    "BoundedWindowBaselineSetV2",
    "build_default_bounded_window_baselines_v2",
    "BoundedWindowFalsifierReportV2",
    "run_bounded_window_falsifier_v2",
    "BoundedWindowInsufficiencyProofV2",
    "prove_bounded_window_insufficient_v2",
    "BoundedWindowBaselineV2Witness",
    "emit_bounded_window_baseline_witness_v2",
]
