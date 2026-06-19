"""W77 — Consensus Fallback Controller V23.

Strictly extends W76's
``coordpy.consensus_fallback_controller_v22``. V22 had a 38-stage
chain. V23 adds two new stages:

  post_restart_replacement_repair_arbiter
  post_restart_replacement_best_parent_arbiter

placed between
``post_compound_chain_restart_best_parent_arbiter`` and
``best_parent``.

Honest scope (W77): ``W77-L-CONSENSUS-V23-SYNTHETIC-CAP``.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

from .consensus_fallback_controller_v5 import (
    W59_CONSENSUS_V5_STAGE_ABSTAIN,
    W59_CONSENSUS_V5_STAGE_BEST_PARENT,
    W59_CONSENSUS_V5_STAGE_TRANSCRIPT,
)
from .consensus_fallback_controller_v22 import (
    ConsensusFallbackControllerV22,
    W76_CONSENSUS_V22_STAGE_POST_COMPOUND_CHAIN_RESTART_BEST_PARENT,
    W76_CONSENSUS_V22_STAGES,
)
from .tiny_substrate_v3 import _sha256_hex


W77_CONSENSUS_V23_SCHEMA_VERSION: str = (
    "coordpy.consensus_fallback_controller_v23.v1")
W77_CONSENSUS_V23_STAGE_POST_RESTART_REPLACEMENT_ARBITER: str = (
    "post_restart_replacement_repair_arbiter")
W77_CONSENSUS_V23_STAGE_POST_RESTART_REPLACEMENT_BEST_PARENT: str = (
    "post_restart_replacement_best_parent_arbiter")


def _build_v23_stages() -> tuple[str, ...]:
    out: list[str] = []
    inserted = False
    for s in W76_CONSENSUS_V22_STAGES:
        out.append(s)
        if (not inserted and s ==
                W76_CONSENSUS_V22_STAGE_POST_COMPOUND_CHAIN_RESTART_BEST_PARENT):
            out.append(
                W77_CONSENSUS_V23_STAGE_POST_RESTART_REPLACEMENT_ARBITER)
            out.append(
                W77_CONSENSUS_V23_STAGE_POST_RESTART_REPLACEMENT_BEST_PARENT)
            inserted = True
    if not inserted:
        idx = (
            out.index(W59_CONSENSUS_V5_STAGE_BEST_PARENT)
            if W59_CONSENSUS_V5_STAGE_BEST_PARENT in out
            else len(out))
        out.insert(
            idx,
            W77_CONSENSUS_V23_STAGE_POST_RESTART_REPLACEMENT_BEST_PARENT)
        out.insert(
            idx,
            W77_CONSENSUS_V23_STAGE_POST_RESTART_REPLACEMENT_ARBITER)
    return tuple(out)


W77_CONSENSUS_V23_STAGES: tuple[str, ...] = _build_v23_stages()


@dataclasses.dataclass
class ConsensusFallbackControllerV23:
    inner_v22: ConsensusFallbackControllerV22
    post_restart_replacement_threshold: float = 0.5
    post_restart_replacement_best_parent_threshold: float = 0.5
    audit_v23: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    @classmethod
    def init(
            cls, *,
            post_restart_replacement_threshold: float = 0.5,
            post_restart_replacement_best_parent_threshold: (
                float) = 0.5,
            **inner_kwargs: Any,
    ) -> "ConsensusFallbackControllerV23":
        inner = ConsensusFallbackControllerV22.init(
            **inner_kwargs)
        return cls(
            inner_v22=inner,
            post_restart_replacement_threshold=float(
                post_restart_replacement_threshold),
            post_restart_replacement_best_parent_threshold=float(
                post_restart_replacement_best_parent_threshold))

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W77_CONSENSUS_V23_SCHEMA_VERSION,
            "kind": "consensus_v23_controller",
            "inner_v22_cid": str(self.inner_v22.cid()),
            "stages": list(W77_CONSENSUS_V23_STAGES),
            "post_restart_replacement_threshold": float(round(
                self.post_restart_replacement_threshold, 12)),
            "post_restart_replacement_best_parent_threshold":
                float(round(
                    self
                    .post_restart_replacement_best_parent_threshold,
                    12)),
        })

    def decide_v23(
            self, *, payloads: Sequence[Sequence[float]],
            trusts: Sequence[float],
            replay_decisions: Sequence[str],
            transcript_available: bool = False,
            post_restart_replacement_scores_per_parent: (
                Sequence[float] | None) = None,
            post_restart_replacement_best_parent_scores_per_parent: (
                Sequence[float] | None) = None,
            **v22_kwargs: Any,
    ) -> dict[str, Any]:
        v22_out = self.inner_v22.decide_v22(
            payloads=payloads, trusts=trusts,
            replay_decisions=replay_decisions,
            transcript_available=bool(transcript_available),
            **v22_kwargs)
        v22_stage = str(v22_out.get("stage", ""))
        terminal_stages = (
            W59_CONSENSUS_V5_STAGE_BEST_PARENT,
            W59_CONSENSUS_V5_STAGE_TRANSCRIPT,
            W59_CONSENSUS_V5_STAGE_ABSTAIN)
        # Stage 39: post-restart-replacement-repair arbiter.
        if (v22_stage in terminal_stages
                and post_restart_replacement_scores_per_parent
                is not None):
            best_idx = -1
            best_score = -1.0
            for i, sc in enumerate(
                    post_restart_replacement_scores_per_parent):
                if (float(sc) >= float(
                        self
                        .post_restart_replacement_threshold)
                        and float(sc) > best_score):
                    best_idx = int(i)
                    best_score = float(sc)
            if best_idx >= 0 and best_idx < len(payloads):
                self.audit_v23.append({
                    "stage": (
                        W77_CONSENSUS_V23_STAGE_POST_RESTART_REPLACEMENT_ARBITER),
                    "v22_terminal_stage": str(v22_stage),
                    "winning_parent": int(best_idx),
                    "score": float(round(best_score, 12)),
                })
                return {
                    "stage": (
                        W77_CONSENSUS_V23_STAGE_POST_RESTART_REPLACEMENT_ARBITER),
                    "payload": [
                        float(x) for x in payloads[best_idx]],
                    "v23_promoted": True,
                    "rationale":
                        "post_restart_replacement_applied",
                }
        # Stage 40: post-restart-replacement-best-parent arbiter.
        if (v22_stage in terminal_stages
                and post_restart_replacement_best_parent_scores_per_parent
                is not None):
            best_idx = -1
            best_score = -1.0
            for i, sc in enumerate(
                    post_restart_replacement_best_parent_scores_per_parent):
                if (float(sc) >= float(
                        self
                        .post_restart_replacement_best_parent_threshold)
                        and float(sc) > best_score):
                    best_idx = int(i)
                    best_score = float(sc)
            if best_idx >= 0 and best_idx < len(payloads):
                self.audit_v23.append({
                    "stage": (
                        W77_CONSENSUS_V23_STAGE_POST_RESTART_REPLACEMENT_BEST_PARENT),
                    "v22_terminal_stage": str(v22_stage),
                    "winning_parent": int(best_idx),
                    "score": float(round(best_score, 12)),
                })
                return {
                    "stage": (
                        W77_CONSENSUS_V23_STAGE_POST_RESTART_REPLACEMENT_BEST_PARENT),
                    "payload": [
                        float(x) for x in payloads[best_idx]],
                    "v23_promoted": True,
                    "rationale":
                        "post_restart_replacement_best_parent_applied",
                }
        self.audit_v23.append({
            "stage": v22_stage, "v23_promoted": False})
        return v22_out


@dataclasses.dataclass(frozen=True)
class ConsensusV23Witness:
    schema: str
    controller_cid: str
    stages: tuple[str, ...]
    n_decisions: int
    post_restart_replacement_arbiter_fired: int
    post_restart_replacement_best_parent_arbiter_fired: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "controller_cid": str(self.controller_cid),
            "stages": list(self.stages),
            "n_decisions": int(self.n_decisions),
            "post_restart_replacement_arbiter_fired": int(
                self.post_restart_replacement_arbiter_fired),
            "post_restart_replacement_best_parent_arbiter_fired":
                int(
                    self
                    .post_restart_replacement_best_parent_arbiter_fired),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "consensus_v23_witness",
            "witness": self.to_dict()})


def emit_consensus_v23_witness(
        controller: ConsensusFallbackControllerV23,
) -> ConsensusV23Witness:
    pr = sum(
        1 for e in controller.audit_v23
        if str(e.get("stage", ""))
            == W77_CONSENSUS_V23_STAGE_POST_RESTART_REPLACEMENT_ARBITER)
    pbp = sum(
        1 for e in controller.audit_v23
        if str(e.get("stage", ""))
            == W77_CONSENSUS_V23_STAGE_POST_RESTART_REPLACEMENT_BEST_PARENT)
    return ConsensusV23Witness(
        schema=W77_CONSENSUS_V23_SCHEMA_VERSION,
        controller_cid=str(controller.cid()),
        stages=tuple(W77_CONSENSUS_V23_STAGES),
        n_decisions=int(len(controller.audit_v23)),
        post_restart_replacement_arbiter_fired=int(pr),
        post_restart_replacement_best_parent_arbiter_fired=int(
            pbp),
    )


__all__ = [
    "W77_CONSENSUS_V23_SCHEMA_VERSION",
    "W77_CONSENSUS_V23_STAGE_POST_RESTART_REPLACEMENT_ARBITER",
    "W77_CONSENSUS_V23_STAGE_POST_RESTART_REPLACEMENT_BEST_PARENT",
    "W77_CONSENSUS_V23_STAGES",
    "ConsensusFallbackControllerV23",
    "ConsensusV23Witness",
    "emit_consensus_v23_witness",
]
