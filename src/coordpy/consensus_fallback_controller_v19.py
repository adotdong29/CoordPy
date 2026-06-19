"""W73 — Consensus Fallback Controller V19.

Strictly extends W72's
``coordpy.consensus_fallback_controller_v18``. V18 had a 30-stage
chain. V19 adds two new stages:

  replacement_pressure_arbiter
  replacement_after_contradiction_then_rejoin_arbiter

placed between
``delayed_rejoin_after_restart_arbiter`` and ``best_parent``.

Honest scope (W73): ``W73-L-CONSENSUS-V19-SYNTHETIC-CAP``.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

from .consensus_fallback_controller_v5 import (
    W59_CONSENSUS_V5_STAGE_ABSTAIN,
    W59_CONSENSUS_V5_STAGE_BEST_PARENT,
    W59_CONSENSUS_V5_STAGE_TRANSCRIPT,
)
from .consensus_fallback_controller_v18 import (
    ConsensusFallbackControllerV18,
    W72_CONSENSUS_V18_STAGE_DELAYED_REJOIN_ARBITER,
    W72_CONSENSUS_V18_STAGES,
)
from .tiny_substrate_v3 import _sha256_hex


W73_CONSENSUS_V19_SCHEMA_VERSION: str = (
    "coordpy.consensus_fallback_controller_v19.v1")
W73_CONSENSUS_V19_STAGE_REPLACEMENT_PRESSURE_ARBITER: str = (
    "replacement_pressure_arbiter")
W73_CONSENSUS_V19_STAGE_REPLACEMENT_AFTER_CTR_ARBITER: str = (
    "replacement_after_contradiction_then_rejoin_arbiter")


def _build_v19_stages() -> tuple[str, ...]:
    out: list[str] = []
    inserted = False
    for s in W72_CONSENSUS_V18_STAGES:
        out.append(s)
        if (not inserted and s ==
                W72_CONSENSUS_V18_STAGE_DELAYED_REJOIN_ARBITER):
            out.append(
                W73_CONSENSUS_V19_STAGE_REPLACEMENT_PRESSURE_ARBITER)
            out.append(
                W73_CONSENSUS_V19_STAGE_REPLACEMENT_AFTER_CTR_ARBITER)
            inserted = True
    if not inserted:
        idx = (out.index(W59_CONSENSUS_V5_STAGE_BEST_PARENT)
               if W59_CONSENSUS_V5_STAGE_BEST_PARENT in out
               else len(out))
        out.insert(
            idx,
            W73_CONSENSUS_V19_STAGE_REPLACEMENT_AFTER_CTR_ARBITER)
        out.insert(
            idx,
            W73_CONSENSUS_V19_STAGE_REPLACEMENT_PRESSURE_ARBITER)
    return tuple(out)


W73_CONSENSUS_V19_STAGES: tuple[str, ...] = _build_v19_stages()


@dataclasses.dataclass
class ConsensusFallbackControllerV19:
    inner_v18: ConsensusFallbackControllerV18
    replacement_pressure_threshold: float = 0.5
    replacement_after_ctr_threshold: float = 0.5
    audit_v19: list[dict[str, Any]] = dataclasses.field(
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
            replacement_pressure_threshold: float = 0.5,
            replacement_after_ctr_threshold: float = 0.5,
            **inner_kwargs: Any,
    ) -> "ConsensusFallbackControllerV19":
        inner = ConsensusFallbackControllerV18.init(
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
            rejoin_pressure_threshold=float(
                rejoin_pressure_threshold),
            delayed_rejoin_threshold=float(
                delayed_rejoin_threshold),
            **inner_kwargs)
        return cls(
            inner_v18=inner,
            replacement_pressure_threshold=float(
                replacement_pressure_threshold),
            replacement_after_ctr_threshold=float(
                replacement_after_ctr_threshold))

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W73_CONSENSUS_V19_SCHEMA_VERSION,
            "kind": "consensus_v19_controller",
            "inner_v18_cid": str(self.inner_v18.cid()),
            "stages": list(W73_CONSENSUS_V19_STAGES),
            "replacement_pressure_threshold": float(round(
                self.replacement_pressure_threshold, 12)),
            "replacement_after_ctr_threshold": float(round(
                self.replacement_after_ctr_threshold, 12)),
        })

    def decide_v19(
            self, *, payloads: Sequence[Sequence[float]],
            trusts: Sequence[float],
            replay_decisions: Sequence[str],
            transcript_available: bool = False,
            replacement_pressure_scores_per_parent: (
                Sequence[float] | None) = None,
            replacement_after_ctr_scores_per_parent: (
                Sequence[float] | None) = None,
            **v18_kwargs: Any,
    ) -> dict[str, Any]:
        v18_out = self.inner_v18.decide_v18(
            payloads=payloads, trusts=trusts,
            replay_decisions=replay_decisions,
            transcript_available=bool(transcript_available),
            **v18_kwargs)
        v18_stage = str(v18_out.get("stage", ""))
        terminal_stages = (
            W59_CONSENSUS_V5_STAGE_BEST_PARENT,
            W59_CONSENSUS_V5_STAGE_TRANSCRIPT,
            W59_CONSENSUS_V5_STAGE_ABSTAIN)
        # Stage 31: replacement-pressure arbiter.
        if (v18_stage in terminal_stages
                and replacement_pressure_scores_per_parent
                is not None):
            best_idx = -1
            best_score = -1.0
            for i, sc in enumerate(
                    replacement_pressure_scores_per_parent):
                if (float(sc) >= float(
                        self.replacement_pressure_threshold)
                        and float(sc) > best_score):
                    best_idx = int(i)
                    best_score = float(sc)
            if best_idx >= 0 and best_idx < len(payloads):
                self.audit_v19.append({
                    "stage": (
                        W73_CONSENSUS_V19_STAGE_REPLACEMENT_PRESSURE_ARBITER),
                    "v18_terminal_stage": str(v18_stage),
                    "winning_parent": int(best_idx),
                    "score": float(round(best_score, 12)),
                })
                return {
                    "stage": (
                        W73_CONSENSUS_V19_STAGE_REPLACEMENT_PRESSURE_ARBITER),
                    "payload": [
                        float(x) for x in payloads[best_idx]],
                    "v19_promoted": True,
                    "rationale": "replacement_pressure_applied",
                }
        # Stage 32: replacement-after-CTR arbiter.
        if (v18_stage in terminal_stages
                and replacement_after_ctr_scores_per_parent
                is not None):
            best_idx = -1
            best_score = -1.0
            for i, sc in enumerate(
                    replacement_after_ctr_scores_per_parent):
                if (float(sc) >= float(
                        self.replacement_after_ctr_threshold)
                        and float(sc) > best_score):
                    best_idx = int(i)
                    best_score = float(sc)
            if best_idx >= 0 and best_idx < len(payloads):
                self.audit_v19.append({
                    "stage": (
                        W73_CONSENSUS_V19_STAGE_REPLACEMENT_AFTER_CTR_ARBITER),
                    "v18_terminal_stage": str(v18_stage),
                    "winning_parent": int(best_idx),
                    "score": float(round(best_score, 12)),
                })
                return {
                    "stage": (
                        W73_CONSENSUS_V19_STAGE_REPLACEMENT_AFTER_CTR_ARBITER),
                    "payload": [
                        float(x) for x in payloads[best_idx]],
                    "v19_promoted": True,
                    "rationale": "replacement_after_ctr_applied",
                }
        self.audit_v19.append({
            "stage": v18_stage, "v19_promoted": False})
        return v18_out


@dataclasses.dataclass(frozen=True)
class ConsensusV19Witness:
    schema: str
    controller_cid: str
    stages: tuple[str, ...]
    n_decisions: int
    replacement_pressure_stage_fired: int
    replacement_after_ctr_stage_fired: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "controller_cid": str(self.controller_cid),
            "stages": list(self.stages),
            "n_decisions": int(self.n_decisions),
            "replacement_pressure_stage_fired": int(
                self.replacement_pressure_stage_fired),
            "replacement_after_ctr_stage_fired": int(
                self.replacement_after_ctr_stage_fired),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "consensus_v19_witness",
            "witness": self.to_dict()})


def emit_consensus_v19_witness(
        controller: ConsensusFallbackControllerV19,
) -> ConsensusV19Witness:
    rp = sum(
        1 for e in controller.audit_v19
        if str(e.get("stage", ""))
            == W73_CONSENSUS_V19_STAGE_REPLACEMENT_PRESSURE_ARBITER)
    rep = sum(
        1 for e in controller.audit_v19
        if str(e.get("stage", ""))
            == W73_CONSENSUS_V19_STAGE_REPLACEMENT_AFTER_CTR_ARBITER)
    return ConsensusV19Witness(
        schema=W73_CONSENSUS_V19_SCHEMA_VERSION,
        controller_cid=str(controller.cid()),
        stages=tuple(W73_CONSENSUS_V19_STAGES),
        n_decisions=int(len(controller.audit_v19)),
        replacement_pressure_stage_fired=int(rp),
        replacement_after_ctr_stage_fired=int(rep),
    )


__all__ = [
    "W73_CONSENSUS_V19_SCHEMA_VERSION",
    "W73_CONSENSUS_V19_STAGE_REPLACEMENT_PRESSURE_ARBITER",
    "W73_CONSENSUS_V19_STAGE_REPLACEMENT_AFTER_CTR_ARBITER",
    "W73_CONSENSUS_V19_STAGES",
    "ConsensusFallbackControllerV19",
    "ConsensusV19Witness",
    "emit_consensus_v19_witness",
]
