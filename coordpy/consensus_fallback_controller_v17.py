"""W71 — Consensus Fallback Controller V17.

Strictly extends W70's
``coordpy.consensus_fallback_controller_v16``. V16 had a 26-stage
chain. V17 adds two new stages:

  restart_aware_arbiter
  delayed_repair_after_restart_arbiter

placed between
``budget_primary_arbiter`` and ``best_parent``.

Honest scope (W71): ``W71-L-CONSENSUS-V17-SYNTHETIC-CAP``.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

from .consensus_fallback_controller_v5 import (
    W59_CONSENSUS_V5_STAGE_ABSTAIN,
    W59_CONSENSUS_V5_STAGE_BEST_PARENT,
    W59_CONSENSUS_V5_STAGE_TRANSCRIPT,
)
from .consensus_fallback_controller_v16 import (
    ConsensusFallbackControllerV16,
    W70_CONSENSUS_V16_STAGES,
    W70_CONSENSUS_V16_STAGE_BUDGET_PRIMARY_ARBITER,
)
from .tiny_substrate_v3 import _sha256_hex


W71_CONSENSUS_V17_SCHEMA_VERSION: str = (
    "coordpy.consensus_fallback_controller_v17.v1")
W71_CONSENSUS_V17_STAGE_RESTART_AWARE_ARBITER: str = (
    "restart_aware_arbiter")
W71_CONSENSUS_V17_STAGE_DELAYED_REPAIR_ARBITER: str = (
    "delayed_repair_after_restart_arbiter")


def _build_v17_stages() -> tuple[str, ...]:
    out: list[str] = []
    inserted = False
    for s in W70_CONSENSUS_V16_STAGES:
        out.append(s)
        if (not inserted and s ==
                W70_CONSENSUS_V16_STAGE_BUDGET_PRIMARY_ARBITER):
            out.append(
                W71_CONSENSUS_V17_STAGE_RESTART_AWARE_ARBITER)
            out.append(
                W71_CONSENSUS_V17_STAGE_DELAYED_REPAIR_ARBITER)
            inserted = True
    if not inserted:
        idx = (out.index(W59_CONSENSUS_V5_STAGE_BEST_PARENT)
               if W59_CONSENSUS_V5_STAGE_BEST_PARENT in out
               else len(out))
        out.insert(
            idx, W71_CONSENSUS_V17_STAGE_DELAYED_REPAIR_ARBITER)
        out.insert(
            idx, W71_CONSENSUS_V17_STAGE_RESTART_AWARE_ARBITER)
    return tuple(out)


W71_CONSENSUS_V17_STAGES: tuple[str, ...] = _build_v17_stages()


@dataclasses.dataclass
class ConsensusFallbackControllerV17:
    inner_v16: ConsensusFallbackControllerV16
    restart_aware_threshold: float = 0.5
    delayed_repair_threshold: float = 0.5
    audit_v17: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    @classmethod
    def init(
            cls, *,
            k_required: int = 2, cosine_floor: float = 0.6,
            trust_threshold: float = 0.5,
            multi_branch_rejoin_threshold: float = 0.5,
            silent_corruption_threshold: float = 0.5,
            repair_dominance_threshold: float = 0.5,
            budget_primary_threshold: float = 0.5,
            restart_aware_threshold: float = 0.5,
            delayed_repair_threshold: float = 0.5,
            **inner_kwargs: Any,
    ) -> "ConsensusFallbackControllerV17":
        inner = ConsensusFallbackControllerV16.init(
            k_required=int(k_required),
            cosine_floor=float(cosine_floor),
            trust_threshold=float(trust_threshold),
            multi_branch_rejoin_threshold=float(
                multi_branch_rejoin_threshold),
            silent_corruption_threshold=float(
                silent_corruption_threshold),
            repair_dominance_threshold=float(
                repair_dominance_threshold),
            budget_primary_threshold=float(
                budget_primary_threshold),
            **inner_kwargs)
        return cls(
            inner_v16=inner,
            restart_aware_threshold=float(
                restart_aware_threshold),
            delayed_repair_threshold=float(
                delayed_repair_threshold))

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W71_CONSENSUS_V17_SCHEMA_VERSION,
            "kind": "consensus_v17_controller",
            "inner_v16_cid": str(self.inner_v16.cid()),
            "stages": list(W71_CONSENSUS_V17_STAGES),
            "restart_aware_threshold": float(round(
                self.restart_aware_threshold, 12)),
            "delayed_repair_threshold": float(round(
                self.delayed_repair_threshold, 12)),
        })

    def decide_v17(
            self, *, payloads: Sequence[Sequence[float]],
            trusts: Sequence[float],
            replay_decisions: Sequence[str],
            transcript_available: bool = False,
            restart_aware_scores_per_parent: (
                Sequence[float] | None) = None,
            delayed_repair_scores_per_parent: (
                Sequence[float] | None) = None,
            **v16_kwargs: Any,
    ) -> dict[str, Any]:
        v16_out = self.inner_v16.decide_v16(
            payloads=payloads, trusts=trusts,
            replay_decisions=replay_decisions,
            transcript_available=bool(transcript_available),
            **v16_kwargs)
        v16_stage = str(v16_out.get("stage", ""))
        terminal_stages = (
            W59_CONSENSUS_V5_STAGE_BEST_PARENT,
            W59_CONSENSUS_V5_STAGE_TRANSCRIPT,
            W59_CONSENSUS_V5_STAGE_ABSTAIN)
        # Stage 27: restart-aware arbiter.
        if (v16_stage in terminal_stages
                and restart_aware_scores_per_parent is not None):
            best_idx = -1
            best_score = -1.0
            for i, sc in enumerate(
                    restart_aware_scores_per_parent):
                if (float(sc) >= float(
                        self.restart_aware_threshold)
                        and float(sc) > best_score):
                    best_idx = int(i)
                    best_score = float(sc)
            if best_idx >= 0 and best_idx < len(payloads):
                self.audit_v17.append({
                    "stage": (
                        W71_CONSENSUS_V17_STAGE_RESTART_AWARE_ARBITER),
                    "v16_terminal_stage": str(v16_stage),
                    "winning_parent": int(best_idx),
                    "score": float(round(best_score, 12)),
                })
                return {
                    "stage": (
                        W71_CONSENSUS_V17_STAGE_RESTART_AWARE_ARBITER),
                    "payload": [
                        float(x) for x in payloads[best_idx]],
                    "v17_promoted": True,
                    "rationale": "restart_aware_applied",
                }
        # Stage 28: delayed-repair-after-restart arbiter.
        if (v16_stage in terminal_stages
                and delayed_repair_scores_per_parent is not None):
            best_idx = -1
            best_score = -1.0
            for i, sc in enumerate(
                    delayed_repair_scores_per_parent):
                if (float(sc) >= float(
                        self.delayed_repair_threshold)
                        and float(sc) > best_score):
                    best_idx = int(i)
                    best_score = float(sc)
            if best_idx >= 0 and best_idx < len(payloads):
                self.audit_v17.append({
                    "stage": (
                        W71_CONSENSUS_V17_STAGE_DELAYED_REPAIR_ARBITER),
                    "v16_terminal_stage": str(v16_stage),
                    "winning_parent": int(best_idx),
                    "score": float(round(best_score, 12)),
                })
                return {
                    "stage": (
                        W71_CONSENSUS_V17_STAGE_DELAYED_REPAIR_ARBITER),
                    "payload": [
                        float(x) for x in payloads[best_idx]],
                    "v17_promoted": True,
                    "rationale": "delayed_repair_applied",
                }
        self.audit_v17.append({
            "stage": v16_stage, "v17_promoted": False})
        return v16_out


@dataclasses.dataclass(frozen=True)
class ConsensusV17Witness:
    schema: str
    controller_cid: str
    stages: tuple[str, ...]
    n_decisions: int
    restart_aware_stage_fired: int
    delayed_repair_stage_fired: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "controller_cid": str(self.controller_cid),
            "stages": list(self.stages),
            "n_decisions": int(self.n_decisions),
            "restart_aware_stage_fired": int(
                self.restart_aware_stage_fired),
            "delayed_repair_stage_fired": int(
                self.delayed_repair_stage_fired),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "consensus_v17_witness",
            "witness": self.to_dict()})


def emit_consensus_v17_witness(
        controller: ConsensusFallbackControllerV17,
) -> ConsensusV17Witness:
    ra = sum(
        1 for e in controller.audit_v17
        if str(e.get("stage", ""))
            == W71_CONSENSUS_V17_STAGE_RESTART_AWARE_ARBITER)
    dr = sum(
        1 for e in controller.audit_v17
        if str(e.get("stage", ""))
            == W71_CONSENSUS_V17_STAGE_DELAYED_REPAIR_ARBITER)
    return ConsensusV17Witness(
        schema=W71_CONSENSUS_V17_SCHEMA_VERSION,
        controller_cid=str(controller.cid()),
        stages=tuple(W71_CONSENSUS_V17_STAGES),
        n_decisions=int(len(controller.audit_v17)),
        restart_aware_stage_fired=int(ra),
        delayed_repair_stage_fired=int(dr),
    )


__all__ = [
    "W71_CONSENSUS_V17_SCHEMA_VERSION",
    "W71_CONSENSUS_V17_STAGE_RESTART_AWARE_ARBITER",
    "W71_CONSENSUS_V17_STAGE_DELAYED_REPAIR_ARBITER",
    "W71_CONSENSUS_V17_STAGES",
    "ConsensusFallbackControllerV17",
    "ConsensusV17Witness",
    "emit_consensus_v17_witness",
]
