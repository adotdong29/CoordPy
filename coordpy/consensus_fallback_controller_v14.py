"""W68 M11 — Consensus Fallback Controller V14.

Strictly extends W67's ``coordpy.consensus_fallback_controller_v13``.
V13 had a 20-stage chain. V14 adds two new stages:

  partial_contradiction_arbiter
  agent_replacement_warm_restart_arbiter

placed between
``branch_merge_reconciliation_arbiter`` and ``best_parent``. The
new stages fire when:

* ``partial_contradiction_arbiter`` — at least two parents carry
  conflicting payloads with delayed reconciliation and the
  partial-contradiction score is above threshold;
* ``agent_replacement_warm_restart_arbiter`` — at least one parent
  has an agent-replacement warm-restart score above threshold.

Honest scope (W68)
------------------

* The new stages rely on synthetic scores passed by the caller.
  ``W68-L-CONSENSUS-V14-SYNTHETIC-CAP``.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

from .consensus_fallback_controller_v5 import (
    W59_CONSENSUS_V5_STAGE_ABSTAIN,
    W59_CONSENSUS_V5_STAGE_BEST_PARENT,
    W59_CONSENSUS_V5_STAGE_TRANSCRIPT,
)
from .consensus_fallback_controller_v13 import (
    ConsensusFallbackControllerV13,
    W67_CONSENSUS_V13_STAGE_BRANCH_MERGE_RECONCILIATION_ARBITER,
    W67_CONSENSUS_V13_STAGES,
)
from .tiny_substrate_v3 import _sha256_hex


W68_CONSENSUS_V14_SCHEMA_VERSION: str = (
    "coordpy.consensus_fallback_controller_v14.v1")
W68_CONSENSUS_V14_STAGE_PARTIAL_CONTRADICTION_ARBITER: str = (
    "partial_contradiction_arbiter")
W68_CONSENSUS_V14_STAGE_AGENT_REPLACEMENT_WARM_RESTART_ARBITER: str = (
    "agent_replacement_warm_restart_arbiter")


def _build_v14_stages() -> tuple[str, ...]:
    out: list[str] = []
    inserted = False
    for s in W67_CONSENSUS_V13_STAGES:
        out.append(s)
        if (not inserted and s ==
                W67_CONSENSUS_V13_STAGE_BRANCH_MERGE_RECONCILIATION_ARBITER):
            out.append(
                W68_CONSENSUS_V14_STAGE_PARTIAL_CONTRADICTION_ARBITER)
            out.append(
                W68_CONSENSUS_V14_STAGE_AGENT_REPLACEMENT_WARM_RESTART_ARBITER)
            inserted = True
    if not inserted:
        idx = (out.index(W59_CONSENSUS_V5_STAGE_BEST_PARENT)
               if W59_CONSENSUS_V5_STAGE_BEST_PARENT in out
               else len(out))
        out.insert(
            idx,
            W68_CONSENSUS_V14_STAGE_AGENT_REPLACEMENT_WARM_RESTART_ARBITER)
        out.insert(
            idx,
            W68_CONSENSUS_V14_STAGE_PARTIAL_CONTRADICTION_ARBITER)
    return tuple(out)


W68_CONSENSUS_V14_STAGES: tuple[str, ...] = _build_v14_stages()


@dataclasses.dataclass
class ConsensusFallbackControllerV14:
    inner_v13: ConsensusFallbackControllerV13
    partial_contradiction_threshold: float = 0.5
    agent_replacement_warm_restart_threshold: float = 0.5
    audit_v14: list[dict[str, Any]] = dataclasses.field(
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
            partial_contradiction_threshold: float = 0.5,
            agent_replacement_warm_restart_threshold: float = 0.5,
    ) -> "ConsensusFallbackControllerV14":
        inner = ConsensusFallbackControllerV13.init(
            k_required=int(k_required),
            cosine_floor=float(cosine_floor),
            trust_threshold=float(trust_threshold),
            team_failure_recovery_threshold=float(
                team_failure_recovery_threshold),
            team_consensus_under_budget_threshold=float(
                team_consensus_under_budget_threshold),
            visible_token_budget_floor=float(
                visible_token_budget_floor),
            role_dropout_threshold=float(role_dropout_threshold),
            branch_merge_reconciliation_threshold=float(
                branch_merge_reconciliation_threshold))
        return cls(
            inner_v13=inner,
            partial_contradiction_threshold=float(
                partial_contradiction_threshold),
            agent_replacement_warm_restart_threshold=float(
                agent_replacement_warm_restart_threshold))

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W68_CONSENSUS_V14_SCHEMA_VERSION,
            "kind": "consensus_v14_controller",
            "inner_v13_cid": str(self.inner_v13.cid()),
            "stages": list(W68_CONSENSUS_V14_STAGES),
            "partial_contradiction_threshold": float(round(
                self.partial_contradiction_threshold, 12)),
            "agent_replacement_warm_restart_threshold": float(round(
                self.agent_replacement_warm_restart_threshold, 12)),
        })

    def decide_v14(
            self, *, payloads: Sequence[Sequence[float]],
            trusts: Sequence[float],
            replay_decisions: Sequence[str],
            transcript_available: bool = False,
            partial_contradiction_scores_per_parent: (
                Sequence[float] | None) = None,
            agent_replacement_scores_per_parent: (
                Sequence[float] | None) = None,
            n_conflicting_branches: int = 0,
            **v13_kwargs: Any,
    ) -> dict[str, Any]:
        v13_out = self.inner_v13.decide_v13(
            payloads=payloads, trusts=trusts,
            replay_decisions=replay_decisions,
            transcript_available=bool(transcript_available),
            n_conflicting_branches=int(n_conflicting_branches),
            **v13_kwargs)
        v13_stage = str(v13_out.get("stage", ""))
        terminal_stages = (
            W59_CONSENSUS_V5_STAGE_BEST_PARENT,
            W59_CONSENSUS_V5_STAGE_TRANSCRIPT,
            W59_CONSENSUS_V5_STAGE_ABSTAIN)
        # Stage 21: partial-contradiction arbiter.
        if (v13_stage in terminal_stages
                and partial_contradiction_scores_per_parent
                is not None
                and int(n_conflicting_branches) >= 2):
            best_idx = -1
            best_score = -1.0
            for i, sc in enumerate(
                    partial_contradiction_scores_per_parent):
                if (float(sc) >= float(
                        self.partial_contradiction_threshold)
                        and float(sc) > best_score):
                    best_idx = int(i)
                    best_score = float(sc)
            if best_idx >= 0 and best_idx < len(payloads):
                self.audit_v14.append({
                    "stage": (
                        W68_CONSENSUS_V14_STAGE_PARTIAL_CONTRADICTION_ARBITER),
                    "v13_terminal_stage": str(v13_stage),
                    "winning_parent": int(best_idx),
                    "score": float(round(best_score, 12)),
                })
                return {
                    "stage": (
                        W68_CONSENSUS_V14_STAGE_PARTIAL_CONTRADICTION_ARBITER),
                    "payload": [
                        float(x) for x in payloads[best_idx]],
                    "v14_promoted": True,
                    "rationale": "partial_contradiction_applied",
                }
        # Stage 22: agent-replacement-warm-restart arbiter.
        if (v13_stage in terminal_stages
                and agent_replacement_scores_per_parent
                is not None):
            best_idx = -1
            best_score = -1.0
            for i, sc in enumerate(
                    agent_replacement_scores_per_parent):
                if (float(sc) >= float(
                        self.agent_replacement_warm_restart_threshold)
                        and float(sc) > best_score):
                    best_idx = int(i)
                    best_score = float(sc)
            if best_idx >= 0 and best_idx < len(payloads):
                self.audit_v14.append({
                    "stage": (
                        W68_CONSENSUS_V14_STAGE_AGENT_REPLACEMENT_WARM_RESTART_ARBITER),
                    "v13_terminal_stage": str(v13_stage),
                    "winning_parent": int(best_idx),
                    "score": float(round(best_score, 12)),
                })
                return {
                    "stage": (
                        W68_CONSENSUS_V14_STAGE_AGENT_REPLACEMENT_WARM_RESTART_ARBITER),
                    "payload": [
                        float(x) for x in payloads[best_idx]],
                    "v14_promoted": True,
                    "rationale": (
                        "agent_replacement_warm_restart_applied"),
                }
        self.audit_v14.append({
            "stage": v13_stage, "v14_promoted": False})
        return v13_out


@dataclasses.dataclass(frozen=True)
class ConsensusV14Witness:
    schema: str
    controller_cid: str
    stages: tuple[str, ...]
    n_decisions: int
    partial_contradiction_stage_fired: int
    agent_replacement_warm_restart_stage_fired: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "controller_cid": str(self.controller_cid),
            "stages": list(self.stages),
            "n_decisions": int(self.n_decisions),
            "partial_contradiction_stage_fired": int(
                self.partial_contradiction_stage_fired),
            "agent_replacement_warm_restart_stage_fired": int(
                self.agent_replacement_warm_restart_stage_fired),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "consensus_v14_witness",
            "witness": self.to_dict()})


def emit_consensus_v14_witness(
        controller: ConsensusFallbackControllerV14,
) -> ConsensusV14Witness:
    pc = sum(
        1 for e in controller.audit_v14
        if str(e.get("stage", ""))
            == W68_CONSENSUS_V14_STAGE_PARTIAL_CONTRADICTION_ARBITER)
    ar = sum(
        1 for e in controller.audit_v14
        if str(e.get("stage", ""))
            == W68_CONSENSUS_V14_STAGE_AGENT_REPLACEMENT_WARM_RESTART_ARBITER)
    return ConsensusV14Witness(
        schema=W68_CONSENSUS_V14_SCHEMA_VERSION,
        controller_cid=str(controller.cid()),
        stages=tuple(W68_CONSENSUS_V14_STAGES),
        n_decisions=int(len(controller.audit_v14)),
        partial_contradiction_stage_fired=int(pc),
        agent_replacement_warm_restart_stage_fired=int(ar),
    )


__all__ = [
    "W68_CONSENSUS_V14_SCHEMA_VERSION",
    "W68_CONSENSUS_V14_STAGE_PARTIAL_CONTRADICTION_ARBITER",
    "W68_CONSENSUS_V14_STAGE_AGENT_REPLACEMENT_WARM_RESTART_ARBITER",
    "W68_CONSENSUS_V14_STAGES",
    "ConsensusFallbackControllerV14",
    "ConsensusV14Witness",
    "emit_consensus_v14_witness",
]
