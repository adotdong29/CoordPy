"""W69 M11 — Consensus Fallback Controller V15.

Strictly extends W68's ``coordpy.consensus_fallback_controller_v14``.
V14 had a 22-stage chain. V15 adds two new stages:

  multi_branch_rejoin_arbiter
  silent_corruption_plus_member_replacement_arbiter

placed between
``agent_replacement_warm_restart_arbiter`` and ``best_parent``.

Honest scope (W69): ``W69-L-CONSENSUS-V15-SYNTHETIC-CAP``.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

from .consensus_fallback_controller_v5 import (
    W59_CONSENSUS_V5_STAGE_ABSTAIN,
    W59_CONSENSUS_V5_STAGE_BEST_PARENT,
    W59_CONSENSUS_V5_STAGE_TRANSCRIPT,
)
from .consensus_fallback_controller_v14 import (
    ConsensusFallbackControllerV14,
    W68_CONSENSUS_V14_STAGES,
    W68_CONSENSUS_V14_STAGE_AGENT_REPLACEMENT_WARM_RESTART_ARBITER,
)
from .tiny_substrate_v3 import _sha256_hex


W69_CONSENSUS_V15_SCHEMA_VERSION: str = (
    "coordpy.consensus_fallback_controller_v15.v1")
W69_CONSENSUS_V15_STAGE_MULTI_BRANCH_REJOIN_ARBITER: str = (
    "multi_branch_rejoin_arbiter")
W69_CONSENSUS_V15_STAGE_SILENT_CORRUPTION_ARBITER: str = (
    "silent_corruption_plus_member_replacement_arbiter")


def _build_v15_stages() -> tuple[str, ...]:
    out: list[str] = []
    inserted = False
    for s in W68_CONSENSUS_V14_STAGES:
        out.append(s)
        if (not inserted and s ==
                W68_CONSENSUS_V14_STAGE_AGENT_REPLACEMENT_WARM_RESTART_ARBITER):
            out.append(
                W69_CONSENSUS_V15_STAGE_MULTI_BRANCH_REJOIN_ARBITER)
            out.append(
                W69_CONSENSUS_V15_STAGE_SILENT_CORRUPTION_ARBITER)
            inserted = True
    if not inserted:
        idx = (out.index(W59_CONSENSUS_V5_STAGE_BEST_PARENT)
               if W59_CONSENSUS_V5_STAGE_BEST_PARENT in out
               else len(out))
        out.insert(
            idx, W69_CONSENSUS_V15_STAGE_SILENT_CORRUPTION_ARBITER)
        out.insert(
            idx, W69_CONSENSUS_V15_STAGE_MULTI_BRANCH_REJOIN_ARBITER)
    return tuple(out)


W69_CONSENSUS_V15_STAGES: tuple[str, ...] = _build_v15_stages()


@dataclasses.dataclass
class ConsensusFallbackControllerV15:
    inner_v14: ConsensusFallbackControllerV14
    multi_branch_rejoin_threshold: float = 0.5
    silent_corruption_threshold: float = 0.5
    audit_v15: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    @classmethod
    def init(
            cls, *,
            k_required: int = 2, cosine_floor: float = 0.6,
            trust_threshold: float = 0.5,
            multi_branch_rejoin_threshold: float = 0.5,
            silent_corruption_threshold: float = 0.5,
            **inner_kwargs: Any,
    ) -> "ConsensusFallbackControllerV15":
        inner = ConsensusFallbackControllerV14.init(
            k_required=int(k_required),
            cosine_floor=float(cosine_floor),
            trust_threshold=float(trust_threshold),
            **inner_kwargs)
        return cls(
            inner_v14=inner,
            multi_branch_rejoin_threshold=float(
                multi_branch_rejoin_threshold),
            silent_corruption_threshold=float(
                silent_corruption_threshold))

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W69_CONSENSUS_V15_SCHEMA_VERSION,
            "kind": "consensus_v15_controller",
            "inner_v14_cid": str(self.inner_v14.cid()),
            "stages": list(W69_CONSENSUS_V15_STAGES),
            "multi_branch_rejoin_threshold": float(round(
                self.multi_branch_rejoin_threshold, 12)),
            "silent_corruption_threshold": float(round(
                self.silent_corruption_threshold, 12)),
        })

    def decide_v15(
            self, *, payloads: Sequence[Sequence[float]],
            trusts: Sequence[float],
            replay_decisions: Sequence[str],
            transcript_available: bool = False,
            multi_branch_rejoin_scores_per_parent: (
                Sequence[float] | None) = None,
            silent_corruption_scores_per_parent: (
                Sequence[float] | None) = None,
            n_rejoining_branches: int = 0,
            **v14_kwargs: Any,
    ) -> dict[str, Any]:
        v14_out = self.inner_v14.decide_v14(
            payloads=payloads, trusts=trusts,
            replay_decisions=replay_decisions,
            transcript_available=bool(transcript_available),
            **v14_kwargs)
        v14_stage = str(v14_out.get("stage", ""))
        terminal_stages = (
            W59_CONSENSUS_V5_STAGE_BEST_PARENT,
            W59_CONSENSUS_V5_STAGE_TRANSCRIPT,
            W59_CONSENSUS_V5_STAGE_ABSTAIN)
        # Stage 23: multi-branch-rejoin arbiter.
        if (v14_stage in terminal_stages
                and multi_branch_rejoin_scores_per_parent
                is not None
                and int(n_rejoining_branches) >= 2):
            best_idx = -1
            best_score = -1.0
            for i, sc in enumerate(
                    multi_branch_rejoin_scores_per_parent):
                if (float(sc) >= float(
                        self.multi_branch_rejoin_threshold)
                        and float(sc) > best_score):
                    best_idx = int(i)
                    best_score = float(sc)
            if best_idx >= 0 and best_idx < len(payloads):
                self.audit_v15.append({
                    "stage": (
                        W69_CONSENSUS_V15_STAGE_MULTI_BRANCH_REJOIN_ARBITER),
                    "v14_terminal_stage": str(v14_stage),
                    "winning_parent": int(best_idx),
                    "score": float(round(best_score, 12)),
                })
                return {
                    "stage": (
                        W69_CONSENSUS_V15_STAGE_MULTI_BRANCH_REJOIN_ARBITER),
                    "payload": [
                        float(x) for x in payloads[best_idx]],
                    "v15_promoted": True,
                    "rationale": "multi_branch_rejoin_applied",
                }
        # Stage 24: silent-corruption-plus-member-replacement arbiter.
        if (v14_stage in terminal_stages
                and silent_corruption_scores_per_parent
                is not None):
            best_idx = -1
            best_score = -1.0
            for i, sc in enumerate(
                    silent_corruption_scores_per_parent):
                if (float(sc) >= float(
                        self.silent_corruption_threshold)
                        and float(sc) > best_score):
                    best_idx = int(i)
                    best_score = float(sc)
            if best_idx >= 0 and best_idx < len(payloads):
                self.audit_v15.append({
                    "stage": (
                        W69_CONSENSUS_V15_STAGE_SILENT_CORRUPTION_ARBITER),
                    "v14_terminal_stage": str(v14_stage),
                    "winning_parent": int(best_idx),
                    "score": float(round(best_score, 12)),
                })
                return {
                    "stage": (
                        W69_CONSENSUS_V15_STAGE_SILENT_CORRUPTION_ARBITER),
                    "payload": [
                        float(x) for x in payloads[best_idx]],
                    "v15_promoted": True,
                    "rationale": (
                        "silent_corruption_member_replacement_applied"),
                }
        self.audit_v15.append({
            "stage": v14_stage, "v15_promoted": False})
        return v14_out


@dataclasses.dataclass(frozen=True)
class ConsensusV15Witness:
    schema: str
    controller_cid: str
    stages: tuple[str, ...]
    n_decisions: int
    multi_branch_rejoin_stage_fired: int
    silent_corruption_stage_fired: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "controller_cid": str(self.controller_cid),
            "stages": list(self.stages),
            "n_decisions": int(self.n_decisions),
            "multi_branch_rejoin_stage_fired": int(
                self.multi_branch_rejoin_stage_fired),
            "silent_corruption_stage_fired": int(
                self.silent_corruption_stage_fired),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "consensus_v15_witness",
            "witness": self.to_dict()})


def emit_consensus_v15_witness(
        controller: ConsensusFallbackControllerV15,
) -> ConsensusV15Witness:
    mbr = sum(
        1 for e in controller.audit_v15
        if str(e.get("stage", ""))
            == W69_CONSENSUS_V15_STAGE_MULTI_BRANCH_REJOIN_ARBITER)
    sc = sum(
        1 for e in controller.audit_v15
        if str(e.get("stage", ""))
            == W69_CONSENSUS_V15_STAGE_SILENT_CORRUPTION_ARBITER)
    return ConsensusV15Witness(
        schema=W69_CONSENSUS_V15_SCHEMA_VERSION,
        controller_cid=str(controller.cid()),
        stages=tuple(W69_CONSENSUS_V15_STAGES),
        n_decisions=int(len(controller.audit_v15)),
        multi_branch_rejoin_stage_fired=int(mbr),
        silent_corruption_stage_fired=int(sc),
    )


__all__ = [
    "W69_CONSENSUS_V15_SCHEMA_VERSION",
    "W69_CONSENSUS_V15_STAGE_MULTI_BRANCH_REJOIN_ARBITER",
    "W69_CONSENSUS_V15_STAGE_SILENT_CORRUPTION_ARBITER",
    "W69_CONSENSUS_V15_STAGES",
    "ConsensusFallbackControllerV15",
    "ConsensusV15Witness",
    "emit_consensus_v15_witness",
]
