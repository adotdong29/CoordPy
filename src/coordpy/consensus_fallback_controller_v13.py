"""W67 M13 — Consensus Fallback Controller V13.

Strictly extends W66's ``coordpy.consensus_fallback_controller_v12``.
V12 had an 18-stage chain. V13 adds two new stages:

  role_dropout_arbiter
  branch_merge_reconciliation_arbiter

placed between ``team_consensus_under_budget_arbiter`` and
``best_parent``. The new stages fire when:

* ``role_dropout_arbiter`` — at least one parent has a role-dropout
  recovery score above threshold;
* ``branch_merge_reconciliation_arbiter`` — at least two parents
  carry conflicting branch payloads and the branch-merge score is
  above threshold.

Honest scope (W67)
------------------

* The new stages rely on synthetic scores passed by the caller.
  ``W67-L-CONSENSUS-V13-SYNTHETIC-CAP``.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

from .consensus_fallback_controller_v5 import (
    W59_CONSENSUS_V5_STAGE_ABSTAIN,
    W59_CONSENSUS_V5_STAGE_BEST_PARENT,
    W59_CONSENSUS_V5_STAGE_TRANSCRIPT,
)
from .consensus_fallback_controller_v12 import (
    ConsensusFallbackControllerV12,
    W66_CONSENSUS_V12_STAGE_TEAM_CONSENSUS_UNDER_BUDGET_ARBITER,
    W66_CONSENSUS_V12_STAGES,
)
from .tiny_substrate_v3 import _sha256_hex


W67_CONSENSUS_V13_SCHEMA_VERSION: str = (
    "coordpy.consensus_fallback_controller_v13.v1")
W67_CONSENSUS_V13_STAGE_ROLE_DROPOUT_ARBITER: str = (
    "role_dropout_arbiter")
W67_CONSENSUS_V13_STAGE_BRANCH_MERGE_RECONCILIATION_ARBITER: str = (
    "branch_merge_reconciliation_arbiter")


def _build_v13_stages() -> tuple[str, ...]:
    out: list[str] = []
    inserted = False
    for s in W66_CONSENSUS_V12_STAGES:
        out.append(s)
        if (not inserted and s
                == W66_CONSENSUS_V12_STAGE_TEAM_CONSENSUS_UNDER_BUDGET_ARBITER):
            out.append(
                W67_CONSENSUS_V13_STAGE_ROLE_DROPOUT_ARBITER)
            out.append(
                W67_CONSENSUS_V13_STAGE_BRANCH_MERGE_RECONCILIATION_ARBITER)
            inserted = True
    if not inserted:
        idx = (out.index(W59_CONSENSUS_V5_STAGE_BEST_PARENT)
               if W59_CONSENSUS_V5_STAGE_BEST_PARENT in out
               else len(out))
        out.insert(
            idx,
            W67_CONSENSUS_V13_STAGE_BRANCH_MERGE_RECONCILIATION_ARBITER)
        out.insert(
            idx,
            W67_CONSENSUS_V13_STAGE_ROLE_DROPOUT_ARBITER)
    return tuple(out)


W67_CONSENSUS_V13_STAGES: tuple[str, ...] = _build_v13_stages()


@dataclasses.dataclass
class ConsensusFallbackControllerV13:
    inner_v12: ConsensusFallbackControllerV12
    role_dropout_threshold: float = 0.5
    branch_merge_reconciliation_threshold: float = 0.5
    audit_v13: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    @classmethod
    def init(
            cls, *,
            k_required: int = 2, cosine_floor: float = 0.6,
            trust_threshold: float = 0.5,
            team_failure_recovery_threshold: float = 0.5,
            team_consensus_under_budget_threshold: float = 0.5,
            visible_token_budget_floor: float = 0.4,
            role_dropout_threshold: float = 0.5,
            branch_merge_reconciliation_threshold: float = 0.5,
    ) -> "ConsensusFallbackControllerV13":
        inner = ConsensusFallbackControllerV12.init(
            k_required=int(k_required),
            cosine_floor=float(cosine_floor),
            trust_threshold=float(trust_threshold),
            team_failure_recovery_threshold=float(
                team_failure_recovery_threshold),
            team_consensus_under_budget_threshold=float(
                team_consensus_under_budget_threshold),
            visible_token_budget_floor=float(
                visible_token_budget_floor))
        return cls(
            inner_v12=inner,
            role_dropout_threshold=float(role_dropout_threshold),
            branch_merge_reconciliation_threshold=float(
                branch_merge_reconciliation_threshold))

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W67_CONSENSUS_V13_SCHEMA_VERSION,
            "kind": "consensus_v13_controller",
            "inner_v12_cid": str(self.inner_v12.cid()),
            "stages": list(W67_CONSENSUS_V13_STAGES),
            "role_dropout_threshold": float(round(
                self.role_dropout_threshold, 12)),
            "branch_merge_reconciliation_threshold": float(round(
                self.branch_merge_reconciliation_threshold, 12)),
        })

    def decide_v13(
            self, *, payloads: Sequence[Sequence[float]],
            trusts: Sequence[float],
            replay_decisions: Sequence[str],
            transcript_available: bool = False,
            role_dropout_scores_per_parent: (
                Sequence[float] | None) = None,
            branch_merge_scores_per_parent: (
                Sequence[float] | None) = None,
            n_conflicting_branches: int = 0,
            **v12_kwargs: Any,
    ) -> dict[str, Any]:
        v12_out = self.inner_v12.decide_v12(
            payloads=payloads, trusts=trusts,
            replay_decisions=replay_decisions,
            transcript_available=bool(transcript_available),
            **v12_kwargs)
        v12_stage = str(v12_out.get("stage", ""))
        terminal_stages = (
            W59_CONSENSUS_V5_STAGE_BEST_PARENT,
            W59_CONSENSUS_V5_STAGE_TRANSCRIPT,
            W59_CONSENSUS_V5_STAGE_ABSTAIN)
        # Stage 19: role-dropout arbiter.
        if (v12_stage in terminal_stages
                and role_dropout_scores_per_parent is not None):
            best_idx = -1
            best_score = -1.0
            for i, sc in enumerate(
                    role_dropout_scores_per_parent):
                if (float(sc)
                        >= float(self.role_dropout_threshold)
                        and float(sc) > best_score):
                    best_idx = int(i)
                    best_score = float(sc)
            if best_idx >= 0 and best_idx < len(payloads):
                self.audit_v13.append({
                    "stage":
                        W67_CONSENSUS_V13_STAGE_ROLE_DROPOUT_ARBITER,
                    "v12_terminal_stage": str(v12_stage),
                    "winning_parent": int(best_idx),
                    "score": float(round(best_score, 12)),
                })
                return {
                    "stage":
                        W67_CONSENSUS_V13_STAGE_ROLE_DROPOUT_ARBITER,
                    "payload": [
                        float(x) for x in payloads[best_idx]],
                    "v13_promoted": True,
                    "rationale": "role_dropout_applied",
                }
        # Stage 20: branch-merge reconciliation arbiter.
        if (v12_stage in terminal_stages
                and branch_merge_scores_per_parent is not None
                and int(n_conflicting_branches) >= 2):
            best_idx = -1
            best_score = -1.0
            for i, sc in enumerate(
                    branch_merge_scores_per_parent):
                if (float(sc)
                        >= float(
                            self.branch_merge_reconciliation_threshold)
                        and float(sc) > best_score):
                    best_idx = int(i)
                    best_score = float(sc)
            if best_idx >= 0 and best_idx < len(payloads):
                self.audit_v13.append({
                    "stage":
                        W67_CONSENSUS_V13_STAGE_BRANCH_MERGE_RECONCILIATION_ARBITER,
                    "v12_terminal_stage": str(v12_stage),
                    "winning_parent": int(best_idx),
                    "score": float(round(best_score, 12)),
                    "n_conflicting_branches": int(
                        n_conflicting_branches),
                })
                return {
                    "stage":
                        W67_CONSENSUS_V13_STAGE_BRANCH_MERGE_RECONCILIATION_ARBITER,
                    "payload": [
                        float(x) for x in payloads[best_idx]],
                    "v13_promoted": True,
                    "rationale": (
                        "branch_merge_reconciliation_applied"),
                }
        self.audit_v13.append({
            "stage": v12_stage, "v13_promoted": False})
        return v12_out


@dataclasses.dataclass(frozen=True)
class ConsensusV13Witness:
    schema: str
    controller_cid: str
    stages: tuple[str, ...]
    n_decisions: int
    role_dropout_stage_fired: int
    branch_merge_reconciliation_stage_fired: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "controller_cid": str(self.controller_cid),
            "stages": list(self.stages),
            "n_decisions": int(self.n_decisions),
            "role_dropout_stage_fired": int(
                self.role_dropout_stage_fired),
            "branch_merge_reconciliation_stage_fired": int(
                self.branch_merge_reconciliation_stage_fired),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "consensus_v13_witness",
            "witness": self.to_dict()})


def emit_consensus_v13_witness(
        controller: ConsensusFallbackControllerV13,
) -> ConsensusV13Witness:
    rd = sum(
        1 for e in controller.audit_v13
        if str(e.get("stage", ""))
            == W67_CONSENSUS_V13_STAGE_ROLE_DROPOUT_ARBITER)
    bm = sum(
        1 for e in controller.audit_v13
        if str(e.get("stage", ""))
            == W67_CONSENSUS_V13_STAGE_BRANCH_MERGE_RECONCILIATION_ARBITER)
    return ConsensusV13Witness(
        schema=W67_CONSENSUS_V13_SCHEMA_VERSION,
        controller_cid=str(controller.cid()),
        stages=tuple(W67_CONSENSUS_V13_STAGES),
        n_decisions=int(len(controller.audit_v13)),
        role_dropout_stage_fired=int(rd),
        branch_merge_reconciliation_stage_fired=int(bm),
    )


__all__ = [
    "W67_CONSENSUS_V13_SCHEMA_VERSION",
    "W67_CONSENSUS_V13_STAGE_ROLE_DROPOUT_ARBITER",
    "W67_CONSENSUS_V13_STAGE_BRANCH_MERGE_RECONCILIATION_ARBITER",
    "W67_CONSENSUS_V13_STAGES",
    "ConsensusFallbackControllerV13",
    "ConsensusV13Witness",
    "emit_consensus_v13_witness",
]
