"""W72 — Consensus Fallback Controller V18.

Strictly extends W71's
``coordpy.consensus_fallback_controller_v17``. V17 had a 28-stage
chain. V18 adds two new stages:

  rejoin_pressure_arbiter
  delayed_rejoin_after_restart_arbiter

placed between
``delayed_repair_after_restart_arbiter`` and ``best_parent``.

Honest scope (W72): ``W72-L-CONSENSUS-V18-SYNTHETIC-CAP``.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

from .consensus_fallback_controller_v5 import (
    W59_CONSENSUS_V5_STAGE_ABSTAIN,
    W59_CONSENSUS_V5_STAGE_BEST_PARENT,
    W59_CONSENSUS_V5_STAGE_TRANSCRIPT,
)
from .consensus_fallback_controller_v17 import (
    ConsensusFallbackControllerV17,
    W71_CONSENSUS_V17_STAGE_DELAYED_REPAIR_ARBITER,
    W71_CONSENSUS_V17_STAGES,
)
from .tiny_substrate_v3 import _sha256_hex


W72_CONSENSUS_V18_SCHEMA_VERSION: str = (
    "coordpy.consensus_fallback_controller_v18.v1")
W72_CONSENSUS_V18_STAGE_REJOIN_PRESSURE_ARBITER: str = (
    "rejoin_pressure_arbiter")
W72_CONSENSUS_V18_STAGE_DELAYED_REJOIN_ARBITER: str = (
    "delayed_rejoin_after_restart_arbiter")


def _build_v18_stages() -> tuple[str, ...]:
    out: list[str] = []
    inserted = False
    for s in W71_CONSENSUS_V17_STAGES:
        out.append(s)
        if (not inserted and s ==
                W71_CONSENSUS_V17_STAGE_DELAYED_REPAIR_ARBITER):
            out.append(
                W72_CONSENSUS_V18_STAGE_REJOIN_PRESSURE_ARBITER)
            out.append(
                W72_CONSENSUS_V18_STAGE_DELAYED_REJOIN_ARBITER)
            inserted = True
    if not inserted:
        idx = (out.index(W59_CONSENSUS_V5_STAGE_BEST_PARENT)
               if W59_CONSENSUS_V5_STAGE_BEST_PARENT in out
               else len(out))
        out.insert(
            idx, W72_CONSENSUS_V18_STAGE_DELAYED_REJOIN_ARBITER)
        out.insert(
            idx, W72_CONSENSUS_V18_STAGE_REJOIN_PRESSURE_ARBITER)
    return tuple(out)


W72_CONSENSUS_V18_STAGES: tuple[str, ...] = _build_v18_stages()


@dataclasses.dataclass
class ConsensusFallbackControllerV18:
    inner_v17: ConsensusFallbackControllerV17
    rejoin_pressure_threshold: float = 0.5
    delayed_rejoin_threshold: float = 0.5
    audit_v18: list[dict[str, Any]] = dataclasses.field(
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
            rejoin_pressure_threshold: float = 0.5,
            delayed_rejoin_threshold: float = 0.5,
            **inner_kwargs: Any,
    ) -> "ConsensusFallbackControllerV18":
        inner = ConsensusFallbackControllerV17.init(
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
            restart_aware_threshold=float(
                restart_aware_threshold),
            delayed_repair_threshold=float(
                delayed_repair_threshold),
            **inner_kwargs)
        return cls(
            inner_v17=inner,
            rejoin_pressure_threshold=float(
                rejoin_pressure_threshold),
            delayed_rejoin_threshold=float(
                delayed_rejoin_threshold))

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W72_CONSENSUS_V18_SCHEMA_VERSION,
            "kind": "consensus_v18_controller",
            "inner_v17_cid": str(self.inner_v17.cid()),
            "stages": list(W72_CONSENSUS_V18_STAGES),
            "rejoin_pressure_threshold": float(round(
                self.rejoin_pressure_threshold, 12)),
            "delayed_rejoin_threshold": float(round(
                self.delayed_rejoin_threshold, 12)),
        })

    def decide_v18(
            self, *, payloads: Sequence[Sequence[float]],
            trusts: Sequence[float],
            replay_decisions: Sequence[str],
            transcript_available: bool = False,
            rejoin_pressure_scores_per_parent: (
                Sequence[float] | None) = None,
            delayed_rejoin_scores_per_parent: (
                Sequence[float] | None) = None,
            **v17_kwargs: Any,
    ) -> dict[str, Any]:
        v17_out = self.inner_v17.decide_v17(
            payloads=payloads, trusts=trusts,
            replay_decisions=replay_decisions,
            transcript_available=bool(transcript_available),
            **v17_kwargs)
        v17_stage = str(v17_out.get("stage", ""))
        terminal_stages = (
            W59_CONSENSUS_V5_STAGE_BEST_PARENT,
            W59_CONSENSUS_V5_STAGE_TRANSCRIPT,
            W59_CONSENSUS_V5_STAGE_ABSTAIN)
        # Stage 29: rejoin-pressure arbiter.
        if (v17_stage in terminal_stages
                and rejoin_pressure_scores_per_parent
                is not None):
            best_idx = -1
            best_score = -1.0
            for i, sc in enumerate(
                    rejoin_pressure_scores_per_parent):
                if (float(sc) >= float(
                        self.rejoin_pressure_threshold)
                        and float(sc) > best_score):
                    best_idx = int(i)
                    best_score = float(sc)
            if best_idx >= 0 and best_idx < len(payloads):
                self.audit_v18.append({
                    "stage": (
                        W72_CONSENSUS_V18_STAGE_REJOIN_PRESSURE_ARBITER),
                    "v17_terminal_stage": str(v17_stage),
                    "winning_parent": int(best_idx),
                    "score": float(round(best_score, 12)),
                })
                return {
                    "stage": (
                        W72_CONSENSUS_V18_STAGE_REJOIN_PRESSURE_ARBITER),
                    "payload": [
                        float(x) for x in payloads[best_idx]],
                    "v18_promoted": True,
                    "rationale": "rejoin_pressure_applied",
                }
        # Stage 30: delayed-rejoin-after-restart arbiter.
        if (v17_stage in terminal_stages
                and delayed_rejoin_scores_per_parent is not None):
            best_idx = -1
            best_score = -1.0
            for i, sc in enumerate(
                    delayed_rejoin_scores_per_parent):
                if (float(sc) >= float(
                        self.delayed_rejoin_threshold)
                        and float(sc) > best_score):
                    best_idx = int(i)
                    best_score = float(sc)
            if best_idx >= 0 and best_idx < len(payloads):
                self.audit_v18.append({
                    "stage": (
                        W72_CONSENSUS_V18_STAGE_DELAYED_REJOIN_ARBITER),
                    "v17_terminal_stage": str(v17_stage),
                    "winning_parent": int(best_idx),
                    "score": float(round(best_score, 12)),
                })
                return {
                    "stage": (
                        W72_CONSENSUS_V18_STAGE_DELAYED_REJOIN_ARBITER),
                    "payload": [
                        float(x) for x in payloads[best_idx]],
                    "v18_promoted": True,
                    "rationale": "delayed_rejoin_applied",
                }
        self.audit_v18.append({
            "stage": v17_stage, "v18_promoted": False})
        return v17_out


@dataclasses.dataclass(frozen=True)
class ConsensusV18Witness:
    schema: str
    controller_cid: str
    stages: tuple[str, ...]
    n_decisions: int
    rejoin_pressure_stage_fired: int
    delayed_rejoin_stage_fired: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "controller_cid": str(self.controller_cid),
            "stages": list(self.stages),
            "n_decisions": int(self.n_decisions),
            "rejoin_pressure_stage_fired": int(
                self.rejoin_pressure_stage_fired),
            "delayed_rejoin_stage_fired": int(
                self.delayed_rejoin_stage_fired),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "consensus_v18_witness",
            "witness": self.to_dict()})


def emit_consensus_v18_witness(
        controller: ConsensusFallbackControllerV18,
) -> ConsensusV18Witness:
    rp = sum(
        1 for e in controller.audit_v18
        if str(e.get("stage", ""))
            == W72_CONSENSUS_V18_STAGE_REJOIN_PRESSURE_ARBITER)
    dr = sum(
        1 for e in controller.audit_v18
        if str(e.get("stage", ""))
            == W72_CONSENSUS_V18_STAGE_DELAYED_REJOIN_ARBITER)
    return ConsensusV18Witness(
        schema=W72_CONSENSUS_V18_SCHEMA_VERSION,
        controller_cid=str(controller.cid()),
        stages=tuple(W72_CONSENSUS_V18_STAGES),
        n_decisions=int(len(controller.audit_v18)),
        rejoin_pressure_stage_fired=int(rp),
        delayed_rejoin_stage_fired=int(dr),
    )


__all__ = [
    "W72_CONSENSUS_V18_SCHEMA_VERSION",
    "W72_CONSENSUS_V18_STAGE_REJOIN_PRESSURE_ARBITER",
    "W72_CONSENSUS_V18_STAGE_DELAYED_REJOIN_ARBITER",
    "W72_CONSENSUS_V18_STAGES",
    "ConsensusFallbackControllerV18",
    "ConsensusV18Witness",
    "emit_consensus_v18_witness",
]
