"""W65 M13 — Consensus Fallback Controller V11.

Strictly extends W64's ``coordpy.consensus_fallback_controller_v10``.
V10 had a 14-stage chain ending with `replay_dominance_primary_arbiter`.
V11 adds two new stages:

  team_substrate_coordination_arbiter
  multi_agent_abstain_arbiter

placed between ``replay_dominance_primary_arbiter`` and
``best_parent``. The new stages fire when:

* ``team_substrate_coordination_arbiter`` — at least one parent
  has a team_substrate_coordination score above threshold;
* ``multi_agent_abstain_arbiter`` — the multi-agent abstain
  feature exceeds threshold (returns abstain).

Honest scope (W65)
------------------

* The new stages rely on synthetic team-coordination scores
  passed in by the caller. ``W65-L-CONSENSUS-V11-SYNTHETIC-CAP``.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

from .consensus_fallback_controller_v5 import (
    W59_CONSENSUS_V5_STAGE_ABSTAIN,
    W59_CONSENSUS_V5_STAGE_BEST_PARENT,
    W59_CONSENSUS_V5_STAGE_TRANSCRIPT,
)
from .consensus_fallback_controller_v10 import (
    ConsensusFallbackControllerV10,
    W64_CONSENSUS_V10_STAGES,
    W64_CONSENSUS_V10_STAGE_REPLAY_DOMINANCE_PRIMARY_ARBITER,
)
from .tiny_substrate_v3 import _sha256_hex


W65_CONSENSUS_V11_SCHEMA_VERSION: str = (
    "coordpy.consensus_fallback_controller_v11.v1")
W65_CONSENSUS_V11_STAGE_TEAM_SUBSTRATE_COORDINATION_ARBITER: (
    str) = "team_substrate_coordination_arbiter"
W65_CONSENSUS_V11_STAGE_MULTI_AGENT_ABSTAIN_ARBITER: str = (
    "multi_agent_abstain_arbiter")


def _build_v11_stages() -> tuple[str, ...]:
    out: list[str] = []
    inserted = False
    for s in W64_CONSENSUS_V10_STAGES:
        out.append(s)
        if (not inserted and s
                == W64_CONSENSUS_V10_STAGE_REPLAY_DOMINANCE_PRIMARY_ARBITER):
            out.append(
                W65_CONSENSUS_V11_STAGE_TEAM_SUBSTRATE_COORDINATION_ARBITER)
            out.append(
                W65_CONSENSUS_V11_STAGE_MULTI_AGENT_ABSTAIN_ARBITER)
            inserted = True
    if not inserted:
        idx = (out.index(W59_CONSENSUS_V5_STAGE_BEST_PARENT)
               if W59_CONSENSUS_V5_STAGE_BEST_PARENT in out
               else len(out))
        out.insert(
            idx,
            W65_CONSENSUS_V11_STAGE_MULTI_AGENT_ABSTAIN_ARBITER)
        out.insert(
            idx,
            W65_CONSENSUS_V11_STAGE_TEAM_SUBSTRATE_COORDINATION_ARBITER)
    return tuple(out)


W65_CONSENSUS_V11_STAGES: tuple[str, ...] = _build_v11_stages()


@dataclasses.dataclass
class ConsensusFallbackControllerV11:
    inner_v10: ConsensusFallbackControllerV10
    team_substrate_coordination_threshold: float = 0.5
    multi_agent_abstain_threshold: float = 0.5
    audit_v11: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    @classmethod
    def init(
            cls, *,
            k_required: int = 2, cosine_floor: float = 0.6,
            trust_threshold: float = 0.5,
            repair_amount_threshold: float = 0.1,
            hidden_wins_margin_threshold: float = 0.05,
            replay_dominance_primary_threshold: float = 0.05,
            team_substrate_coordination_threshold: float = 0.5,
            multi_agent_abstain_threshold: float = 0.5,
    ) -> "ConsensusFallbackControllerV11":
        inner = ConsensusFallbackControllerV10.init(
            k_required=int(k_required),
            cosine_floor=float(cosine_floor),
            trust_threshold=float(trust_threshold),
            repair_amount_threshold=float(
                repair_amount_threshold),
            hidden_wins_margin_threshold=float(
                hidden_wins_margin_threshold),
            replay_dominance_primary_threshold=float(
                replay_dominance_primary_threshold))
        return cls(
            inner_v10=inner,
            team_substrate_coordination_threshold=float(
                team_substrate_coordination_threshold),
            multi_agent_abstain_threshold=float(
                multi_agent_abstain_threshold))

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W65_CONSENSUS_V11_SCHEMA_VERSION,
            "kind": "consensus_v11_controller",
            "inner_v10_cid": str(self.inner_v10.cid()),
            "stages": list(W65_CONSENSUS_V11_STAGES),
            "team_substrate_coordination_threshold": float(
                round(
                    self.team_substrate_coordination_threshold,
                    12)),
            "multi_agent_abstain_threshold": float(round(
                self.multi_agent_abstain_threshold, 12)),
        })

    def decide_v11(
            self, *, payloads: Sequence[Sequence[float]],
            trusts: Sequence[float],
            replay_decisions: Sequence[str],
            transcript_available: bool = False,
            team_substrate_coordination_scores_per_parent: (
                Sequence[float] | None) = None,
            multi_agent_abstain_score: float = 0.0,
            **v10_kwargs: Any,
    ) -> dict[str, Any]:
        v10_out = self.inner_v10.decide_v10(
            payloads=payloads, trusts=trusts,
            replay_decisions=replay_decisions,
            transcript_available=bool(transcript_available),
            **v10_kwargs)
        v10_stage = str(v10_out.get("stage", ""))
        terminal_stages = (
            W59_CONSENSUS_V5_STAGE_BEST_PARENT,
            W59_CONSENSUS_V5_STAGE_TRANSCRIPT,
            W59_CONSENSUS_V5_STAGE_ABSTAIN)
        # Stage 16: multi-agent abstain takes precedence.
        if (v10_stage in terminal_stages
                and float(multi_agent_abstain_score)
                >= float(self.multi_agent_abstain_threshold)):
            self.audit_v11.append({
                "stage":
                    W65_CONSENSUS_V11_STAGE_MULTI_AGENT_ABSTAIN_ARBITER,
                "v10_terminal_stage": str(v10_stage),
                "abstain_score": float(round(
                    multi_agent_abstain_score, 12)),
            })
            return {
                "stage":
                    W65_CONSENSUS_V11_STAGE_MULTI_AGENT_ABSTAIN_ARBITER,
                "payload": [],
                "v11_promoted": True,
                "rationale": "multi_agent_abstain_applied",
            }
        # Stage 15: team-substrate coordination arbiter.
        if (v10_stage in terminal_stages
                and team_substrate_coordination_scores_per_parent
                is not None):
            best_idx = -1
            best_score = -1.0
            for i, sc in enumerate(
                    team_substrate_coordination_scores_per_parent):
                if (float(sc)
                        >= float(
                            self.team_substrate_coordination_threshold)
                        and float(sc) > best_score):
                    best_idx = int(i)
                    best_score = float(sc)
            if best_idx >= 0 and best_idx < len(payloads):
                self.audit_v11.append({
                    "stage":
                        W65_CONSENSUS_V11_STAGE_TEAM_SUBSTRATE_COORDINATION_ARBITER,
                    "v10_terminal_stage": str(v10_stage),
                    "winning_parent": int(best_idx),
                    "score": float(round(best_score, 12)),
                })
                return {
                    "stage":
                        W65_CONSENSUS_V11_STAGE_TEAM_SUBSTRATE_COORDINATION_ARBITER,
                    "payload": [
                        float(x) for x in payloads[best_idx]],
                    "v11_promoted": True,
                    "rationale": (
                        "team_substrate_coordination_applied"),
                }
        self.audit_v11.append({
            "stage": v10_stage, "v11_promoted": False})
        return v10_out


@dataclasses.dataclass(frozen=True)
class ConsensusV11Witness:
    schema: str
    controller_cid: str
    stages: tuple[str, ...]
    n_decisions: int
    team_substrate_coordination_stage_fired: int
    multi_agent_abstain_stage_fired: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "controller_cid": str(self.controller_cid),
            "stages": list(self.stages),
            "n_decisions": int(self.n_decisions),
            "team_substrate_coordination_stage_fired": int(
                self.team_substrate_coordination_stage_fired),
            "multi_agent_abstain_stage_fired": int(
                self.multi_agent_abstain_stage_fired),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "consensus_v11_witness",
            "witness": self.to_dict()})


def emit_consensus_v11_witness(
        controller: ConsensusFallbackControllerV11,
) -> ConsensusV11Witness:
    ts = sum(
        1 for e in controller.audit_v11
        if str(e.get("stage", ""))
            == W65_CONSENSUS_V11_STAGE_TEAM_SUBSTRATE_COORDINATION_ARBITER)
    ab = sum(
        1 for e in controller.audit_v11
        if str(e.get("stage", ""))
            == W65_CONSENSUS_V11_STAGE_MULTI_AGENT_ABSTAIN_ARBITER)
    return ConsensusV11Witness(
        schema=W65_CONSENSUS_V11_SCHEMA_VERSION,
        controller_cid=str(controller.cid()),
        stages=tuple(W65_CONSENSUS_V11_STAGES),
        n_decisions=int(len(controller.audit_v11)),
        team_substrate_coordination_stage_fired=int(ts),
        multi_agent_abstain_stage_fired=int(ab),
    )


__all__ = [
    "W65_CONSENSUS_V11_SCHEMA_VERSION",
    "W65_CONSENSUS_V11_STAGE_TEAM_SUBSTRATE_COORDINATION_ARBITER",
    "W65_CONSENSUS_V11_STAGE_MULTI_AGENT_ABSTAIN_ARBITER",
    "W65_CONSENSUS_V11_STAGES",
    "ConsensusFallbackControllerV11",
    "ConsensusV11Witness",
    "emit_consensus_v11_witness",
]
