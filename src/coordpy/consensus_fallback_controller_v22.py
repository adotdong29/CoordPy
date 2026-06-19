"""W76 — Consensus Fallback Controller V22.

Strictly extends W75's
``coordpy.consensus_fallback_controller_v21``. V21 had a 36-stage
chain. V22 adds two new stages:

  chain_then_restart_repair_arbiter
  post_compound_chain_restart_best_parent_arbiter

placed between
``compound_repair_after_replacement_then_rejoin_arbiter`` and
``best_parent``.

Honest scope (W76): ``W76-L-CONSENSUS-V22-SYNTHETIC-CAP``.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

from .consensus_fallback_controller_v5 import (
    W59_CONSENSUS_V5_STAGE_ABSTAIN,
    W59_CONSENSUS_V5_STAGE_BEST_PARENT,
    W59_CONSENSUS_V5_STAGE_TRANSCRIPT,
)
from .consensus_fallback_controller_v21 import (
    ConsensusFallbackControllerV21,
    W75_CONSENSUS_V21_STAGE_COMPOUND_REPAIR_AFTER_RTR_ARBITER,
    W75_CONSENSUS_V21_STAGES,
)
from .tiny_substrate_v3 import _sha256_hex


W76_CONSENSUS_V22_SCHEMA_VERSION: str = (
    "coordpy.consensus_fallback_controller_v22.v1")
W76_CONSENSUS_V22_STAGE_CHAIN_THEN_RESTART_ARBITER: str = (
    "chain_then_restart_repair_arbiter")
W76_CONSENSUS_V22_STAGE_POST_COMPOUND_CHAIN_RESTART_BEST_PARENT: (
    str) = (
        "post_compound_chain_restart_best_parent_arbiter")


def _build_v22_stages() -> tuple[str, ...]:
    out: list[str] = []
    inserted = False
    for s in W75_CONSENSUS_V21_STAGES:
        out.append(s)
        if (not inserted and s ==
                W75_CONSENSUS_V21_STAGE_COMPOUND_REPAIR_AFTER_RTR_ARBITER):
            out.append(
                W76_CONSENSUS_V22_STAGE_CHAIN_THEN_RESTART_ARBITER)
            out.append(
                W76_CONSENSUS_V22_STAGE_POST_COMPOUND_CHAIN_RESTART_BEST_PARENT)
            inserted = True
    if not inserted:
        idx = (
            out.index(W59_CONSENSUS_V5_STAGE_BEST_PARENT)
            if W59_CONSENSUS_V5_STAGE_BEST_PARENT in out
            else len(out))
        out.insert(
            idx,
            W76_CONSENSUS_V22_STAGE_POST_COMPOUND_CHAIN_RESTART_BEST_PARENT)
        out.insert(
            idx,
            W76_CONSENSUS_V22_STAGE_CHAIN_THEN_RESTART_ARBITER)
    return tuple(out)


W76_CONSENSUS_V22_STAGES: tuple[str, ...] = _build_v22_stages()


@dataclasses.dataclass
class ConsensusFallbackControllerV22:
    inner_v21: ConsensusFallbackControllerV21
    chain_then_restart_threshold: float = 0.5
    post_compound_chain_restart_threshold: float = 0.5
    audit_v22: list[dict[str, Any]] = dataclasses.field(
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
            compound_repair_threshold: float = 0.5,
            compound_repair_drtr_threshold: float = 0.5,
            compound_chain_repair_threshold: float = 0.5,
            compound_repair_rtr_threshold: float = 0.5,
            chain_then_restart_threshold: float = 0.5,
            post_compound_chain_restart_threshold: float = 0.5,
            **inner_kwargs: Any,
    ) -> "ConsensusFallbackControllerV22":
        inner = ConsensusFallbackControllerV21.init(
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
            replacement_pressure_threshold=float(
                replacement_pressure_threshold),
            replacement_after_ctr_threshold=float(
                replacement_after_ctr_threshold),
            compound_repair_threshold=float(
                compound_repair_threshold),
            compound_repair_drtr_threshold=float(
                compound_repair_drtr_threshold),
            compound_chain_repair_threshold=float(
                compound_chain_repair_threshold),
            compound_repair_rtr_threshold=float(
                compound_repair_rtr_threshold),
            **inner_kwargs)
        return cls(
            inner_v21=inner,
            chain_then_restart_threshold=float(
                chain_then_restart_threshold),
            post_compound_chain_restart_threshold=float(
                post_compound_chain_restart_threshold))

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W76_CONSENSUS_V22_SCHEMA_VERSION,
            "kind": "consensus_v22_controller",
            "inner_v21_cid": str(self.inner_v21.cid()),
            "stages": list(W76_CONSENSUS_V22_STAGES),
            "chain_then_restart_threshold": float(round(
                self.chain_then_restart_threshold, 12)),
            "post_compound_chain_restart_threshold": float(round(
                self.post_compound_chain_restart_threshold, 12)),
        })

    def decide_v22(
            self, *, payloads: Sequence[Sequence[float]],
            trusts: Sequence[float],
            replay_decisions: Sequence[str],
            transcript_available: bool = False,
            chain_then_restart_scores_per_parent: (
                Sequence[float] | None) = None,
            post_compound_chain_restart_scores_per_parent: (
                Sequence[float] | None) = None,
            **v21_kwargs: Any,
    ) -> dict[str, Any]:
        v21_out = self.inner_v21.decide_v21(
            payloads=payloads, trusts=trusts,
            replay_decisions=replay_decisions,
            transcript_available=bool(transcript_available),
            **v21_kwargs)
        v21_stage = str(v21_out.get("stage", ""))
        terminal_stages = (
            W59_CONSENSUS_V5_STAGE_BEST_PARENT,
            W59_CONSENSUS_V5_STAGE_TRANSCRIPT,
            W59_CONSENSUS_V5_STAGE_ABSTAIN)
        # Stage 37: chain-then-restart-repair arbiter.
        if (v21_stage in terminal_stages
                and chain_then_restart_scores_per_parent
                is not None):
            best_idx = -1
            best_score = -1.0
            for i, sc in enumerate(
                    chain_then_restart_scores_per_parent):
                if (float(sc) >= float(
                        self.chain_then_restart_threshold)
                        and float(sc) > best_score):
                    best_idx = int(i)
                    best_score = float(sc)
            if best_idx >= 0 and best_idx < len(payloads):
                self.audit_v22.append({
                    "stage": (
                        W76_CONSENSUS_V22_STAGE_CHAIN_THEN_RESTART_ARBITER),
                    "v21_terminal_stage": str(v21_stage),
                    "winning_parent": int(best_idx),
                    "score": float(round(best_score, 12)),
                })
                return {
                    "stage": (
                        W76_CONSENSUS_V22_STAGE_CHAIN_THEN_RESTART_ARBITER),
                    "payload": [
                        float(x) for x in payloads[best_idx]],
                    "v22_promoted": True,
                    "rationale": "chain_then_restart_applied",
                }
        # Stage 38: post-compound-chain-restart-best-parent
        # arbiter.
        if (v21_stage in terminal_stages
                and post_compound_chain_restart_scores_per_parent
                is not None):
            best_idx = -1
            best_score = -1.0
            for i, sc in enumerate(
                    post_compound_chain_restart_scores_per_parent):
                if (float(sc) >= float(
                        self
                        .post_compound_chain_restart_threshold)
                        and float(sc) > best_score):
                    best_idx = int(i)
                    best_score = float(sc)
            if best_idx >= 0 and best_idx < len(payloads):
                self.audit_v22.append({
                    "stage": (
                        W76_CONSENSUS_V22_STAGE_POST_COMPOUND_CHAIN_RESTART_BEST_PARENT),
                    "v21_terminal_stage": str(v21_stage),
                    "winning_parent": int(best_idx),
                    "score": float(round(best_score, 12)),
                })
                return {
                    "stage": (
                        W76_CONSENSUS_V22_STAGE_POST_COMPOUND_CHAIN_RESTART_BEST_PARENT),
                    "payload": [
                        float(x) for x in payloads[best_idx]],
                    "v22_promoted": True,
                    "rationale":
                        "post_compound_chain_restart_applied",
                }
        self.audit_v22.append({
            "stage": v21_stage, "v22_promoted": False})
        return v21_out


@dataclasses.dataclass(frozen=True)
class ConsensusV22Witness:
    schema: str
    controller_cid: str
    stages: tuple[str, ...]
    n_decisions: int
    chain_then_restart_arbiter_fired: int
    post_compound_chain_restart_arbiter_fired: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "controller_cid": str(self.controller_cid),
            "stages": list(self.stages),
            "n_decisions": int(self.n_decisions),
            "chain_then_restart_arbiter_fired": int(
                self.chain_then_restart_arbiter_fired),
            "post_compound_chain_restart_arbiter_fired": int(
                self.post_compound_chain_restart_arbiter_fired),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "consensus_v22_witness",
            "witness": self.to_dict()})


def emit_consensus_v22_witness(
        controller: ConsensusFallbackControllerV22,
) -> ConsensusV22Witness:
    ctr = sum(
        1 for e in controller.audit_v22
        if str(e.get("stage", ""))
            == W76_CONSENSUS_V22_STAGE_CHAIN_THEN_RESTART_ARBITER)
    pcr = sum(
        1 for e in controller.audit_v22
        if str(e.get("stage", ""))
            == W76_CONSENSUS_V22_STAGE_POST_COMPOUND_CHAIN_RESTART_BEST_PARENT)
    return ConsensusV22Witness(
        schema=W76_CONSENSUS_V22_SCHEMA_VERSION,
        controller_cid=str(controller.cid()),
        stages=tuple(W76_CONSENSUS_V22_STAGES),
        n_decisions=int(len(controller.audit_v22)),
        chain_then_restart_arbiter_fired=int(ctr),
        post_compound_chain_restart_arbiter_fired=int(pcr),
    )


__all__ = [
    "W76_CONSENSUS_V22_SCHEMA_VERSION",
    "W76_CONSENSUS_V22_STAGE_CHAIN_THEN_RESTART_ARBITER",
    "W76_CONSENSUS_V22_STAGE_POST_COMPOUND_CHAIN_RESTART_BEST_PARENT",
    "W76_CONSENSUS_V22_STAGES",
    "ConsensusFallbackControllerV22",
    "ConsensusV22Witness",
    "emit_consensus_v22_witness",
]
