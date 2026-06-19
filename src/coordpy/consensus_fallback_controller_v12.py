"""W66 M13 — Consensus Fallback Controller V12.

Strictly extends W65's ``coordpy.consensus_fallback_controller_v11``.
V11 had a 16-stage chain. V12 adds two new stages:

  team_failure_recovery_arbiter
  team_consensus_under_budget_arbiter

placed between ``multi_agent_abstain_arbiter`` and ``best_parent``.
The new stages fire when:

* ``team_failure_recovery_arbiter`` — at least one parent has a
  team-failure-recovery score above threshold;
* ``team_consensus_under_budget_arbiter`` — a parent has a
  team-consensus score above threshold AND the visible-token
  budget fraction is below threshold.

Honest scope (W66)
------------------

* The new stages rely on synthetic scores passed by the caller.
  ``W66-L-CONSENSUS-V12-SYNTHETIC-CAP``.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

from .consensus_fallback_controller_v5 import (
    W59_CONSENSUS_V5_STAGE_ABSTAIN,
    W59_CONSENSUS_V5_STAGE_BEST_PARENT,
    W59_CONSENSUS_V5_STAGE_TRANSCRIPT,
)
from .consensus_fallback_controller_v11 import (
    ConsensusFallbackControllerV11,
    W65_CONSENSUS_V11_STAGE_MULTI_AGENT_ABSTAIN_ARBITER,
    W65_CONSENSUS_V11_STAGES,
)
from .tiny_substrate_v3 import _sha256_hex


W66_CONSENSUS_V12_SCHEMA_VERSION: str = (
    "coordpy.consensus_fallback_controller_v12.v1")
W66_CONSENSUS_V12_STAGE_TEAM_FAILURE_RECOVERY_ARBITER: str = (
    "team_failure_recovery_arbiter")
W66_CONSENSUS_V12_STAGE_TEAM_CONSENSUS_UNDER_BUDGET_ARBITER: str = (
    "team_consensus_under_budget_arbiter")


def _build_v12_stages() -> tuple[str, ...]:
    out: list[str] = []
    inserted = False
    for s in W65_CONSENSUS_V11_STAGES:
        out.append(s)
        if (not inserted and s
                == W65_CONSENSUS_V11_STAGE_MULTI_AGENT_ABSTAIN_ARBITER):
            out.append(
                W66_CONSENSUS_V12_STAGE_TEAM_FAILURE_RECOVERY_ARBITER)
            out.append(
                W66_CONSENSUS_V12_STAGE_TEAM_CONSENSUS_UNDER_BUDGET_ARBITER)
            inserted = True
    if not inserted:
        idx = (out.index(W59_CONSENSUS_V5_STAGE_BEST_PARENT)
               if W59_CONSENSUS_V5_STAGE_BEST_PARENT in out
               else len(out))
        out.insert(
            idx,
            W66_CONSENSUS_V12_STAGE_TEAM_CONSENSUS_UNDER_BUDGET_ARBITER)
        out.insert(
            idx,
            W66_CONSENSUS_V12_STAGE_TEAM_FAILURE_RECOVERY_ARBITER)
    return tuple(out)


W66_CONSENSUS_V12_STAGES: tuple[str, ...] = _build_v12_stages()


@dataclasses.dataclass
class ConsensusFallbackControllerV12:
    inner_v11: ConsensusFallbackControllerV11
    team_failure_recovery_threshold: float = 0.5
    team_consensus_under_budget_threshold: float = 0.5
    visible_token_budget_floor: float = 0.4
    audit_v12: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    @classmethod
    def init(
            cls, *,
            k_required: int = 2, cosine_floor: float = 0.6,
            trust_threshold: float = 0.5,
            team_failure_recovery_threshold: float = 0.5,
            team_consensus_under_budget_threshold: float = 0.5,
            visible_token_budget_floor: float = 0.4,
    ) -> "ConsensusFallbackControllerV12":
        inner = ConsensusFallbackControllerV11.init(
            k_required=int(k_required),
            cosine_floor=float(cosine_floor),
            trust_threshold=float(trust_threshold))
        return cls(
            inner_v11=inner,
            team_failure_recovery_threshold=float(
                team_failure_recovery_threshold),
            team_consensus_under_budget_threshold=float(
                team_consensus_under_budget_threshold),
            visible_token_budget_floor=float(
                visible_token_budget_floor))

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W66_CONSENSUS_V12_SCHEMA_VERSION,
            "kind": "consensus_v12_controller",
            "inner_v11_cid": str(self.inner_v11.cid()),
            "stages": list(W66_CONSENSUS_V12_STAGES),
            "team_failure_recovery_threshold": float(round(
                self.team_failure_recovery_threshold, 12)),
            "team_consensus_under_budget_threshold": float(round(
                self.team_consensus_under_budget_threshold, 12)),
            "visible_token_budget_floor": float(round(
                self.visible_token_budget_floor, 12)),
        })

    def decide_v12(
            self, *, payloads: Sequence[Sequence[float]],
            trusts: Sequence[float],
            replay_decisions: Sequence[str],
            transcript_available: bool = False,
            team_failure_recovery_scores_per_parent: (
                Sequence[float] | None) = None,
            team_consensus_under_budget_scores_per_parent: (
                Sequence[float] | None) = None,
            visible_token_budget_frac: float = 1.0,
            **v11_kwargs: Any,
    ) -> dict[str, Any]:
        v11_out = self.inner_v11.decide_v11(
            payloads=payloads, trusts=trusts,
            replay_decisions=replay_decisions,
            transcript_available=bool(transcript_available),
            **v11_kwargs)
        v11_stage = str(v11_out.get("stage", ""))
        terminal_stages = (
            W59_CONSENSUS_V5_STAGE_BEST_PARENT,
            W59_CONSENSUS_V5_STAGE_TRANSCRIPT,
            W59_CONSENSUS_V5_STAGE_ABSTAIN)
        # Stage 17: team-failure-recovery arbiter.
        if (v11_stage in terminal_stages
                and team_failure_recovery_scores_per_parent
                is not None):
            best_idx = -1
            best_score = -1.0
            for i, sc in enumerate(
                    team_failure_recovery_scores_per_parent):
                if (float(sc)
                        >= float(
                            self.team_failure_recovery_threshold)
                        and float(sc) > best_score):
                    best_idx = int(i)
                    best_score = float(sc)
            if best_idx >= 0 and best_idx < len(payloads):
                self.audit_v12.append({
                    "stage":
                        W66_CONSENSUS_V12_STAGE_TEAM_FAILURE_RECOVERY_ARBITER,
                    "v11_terminal_stage": str(v11_stage),
                    "winning_parent": int(best_idx),
                    "score": float(round(best_score, 12)),
                })
                return {
                    "stage":
                        W66_CONSENSUS_V12_STAGE_TEAM_FAILURE_RECOVERY_ARBITER,
                    "payload": [
                        float(x) for x in payloads[best_idx]],
                    "v12_promoted": True,
                    "rationale": "team_failure_recovery_applied",
                }
        # Stage 18: team-consensus-under-budget arbiter.
        if (v11_stage in terminal_stages
                and team_consensus_under_budget_scores_per_parent
                is not None
                and float(visible_token_budget_frac)
                <= float(self.visible_token_budget_floor)):
            best_idx = -1
            best_score = -1.0
            for i, sc in enumerate(
                    team_consensus_under_budget_scores_per_parent):
                if (float(sc)
                        >= float(
                            self.team_consensus_under_budget_threshold)
                        and float(sc) > best_score):
                    best_idx = int(i)
                    best_score = float(sc)
            if best_idx >= 0 and best_idx < len(payloads):
                self.audit_v12.append({
                    "stage":
                        W66_CONSENSUS_V12_STAGE_TEAM_CONSENSUS_UNDER_BUDGET_ARBITER,
                    "v11_terminal_stage": str(v11_stage),
                    "winning_parent": int(best_idx),
                    "score": float(round(best_score, 12)),
                    "visible_token_budget_frac": float(round(
                        visible_token_budget_frac, 12)),
                })
                return {
                    "stage":
                        W66_CONSENSUS_V12_STAGE_TEAM_CONSENSUS_UNDER_BUDGET_ARBITER,
                    "payload": [
                        float(x) for x in payloads[best_idx]],
                    "v12_promoted": True,
                    "rationale": (
                        "team_consensus_under_budget_applied"),
                }
        self.audit_v12.append({
            "stage": v11_stage, "v12_promoted": False})
        return v11_out


@dataclasses.dataclass(frozen=True)
class ConsensusV12Witness:
    schema: str
    controller_cid: str
    stages: tuple[str, ...]
    n_decisions: int
    team_failure_recovery_stage_fired: int
    team_consensus_under_budget_stage_fired: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "controller_cid": str(self.controller_cid),
            "stages": list(self.stages),
            "n_decisions": int(self.n_decisions),
            "team_failure_recovery_stage_fired": int(
                self.team_failure_recovery_stage_fired),
            "team_consensus_under_budget_stage_fired": int(
                self.team_consensus_under_budget_stage_fired),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "consensus_v12_witness",
            "witness": self.to_dict()})


def emit_consensus_v12_witness(
        controller: ConsensusFallbackControllerV12,
) -> ConsensusV12Witness:
    tfr = sum(
        1 for e in controller.audit_v12
        if str(e.get("stage", ""))
            == W66_CONSENSUS_V12_STAGE_TEAM_FAILURE_RECOVERY_ARBITER)
    tcb = sum(
        1 for e in controller.audit_v12
        if str(e.get("stage", ""))
            == W66_CONSENSUS_V12_STAGE_TEAM_CONSENSUS_UNDER_BUDGET_ARBITER)
    return ConsensusV12Witness(
        schema=W66_CONSENSUS_V12_SCHEMA_VERSION,
        controller_cid=str(controller.cid()),
        stages=tuple(W66_CONSENSUS_V12_STAGES),
        n_decisions=int(len(controller.audit_v12)),
        team_failure_recovery_stage_fired=int(tfr),
        team_consensus_under_budget_stage_fired=int(tcb),
    )


__all__ = [
    "W66_CONSENSUS_V12_SCHEMA_VERSION",
    "W66_CONSENSUS_V12_STAGE_TEAM_FAILURE_RECOVERY_ARBITER",
    "W66_CONSENSUS_V12_STAGE_TEAM_CONSENSUS_UNDER_BUDGET_ARBITER",
    "W66_CONSENSUS_V12_STAGES",
    "ConsensusFallbackControllerV12",
    "ConsensusV12Witness",
    "emit_consensus_v12_witness",
]
