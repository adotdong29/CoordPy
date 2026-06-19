"""W78 — Consensus Fallback Controller V24.

Strictly extends W77's
``coordpy.consensus_fallback_controller_v23``. V23 had a 40-stage
chain. V24 adds two new stages:

  long_horizon_reconstruction_repair_arbiter
  long_horizon_reconstruction_best_parent_arbiter

placed between
``post_restart_replacement_best_parent_arbiter`` and ``best_parent``.

Honest scope (W78): ``W78-L-CONSENSUS-V24-SYNTHETIC-CAP``.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

from .consensus_fallback_controller_v5 import (
    W59_CONSENSUS_V5_STAGE_ABSTAIN,
    W59_CONSENSUS_V5_STAGE_BEST_PARENT,
    W59_CONSENSUS_V5_STAGE_TRANSCRIPT,
)
from .consensus_fallback_controller_v23 import (
    ConsensusFallbackControllerV23,
    W77_CONSENSUS_V23_STAGE_POST_RESTART_REPLACEMENT_BEST_PARENT,
    W77_CONSENSUS_V23_STAGES,
)
from .tiny_substrate_v3 import _sha256_hex


W78_CONSENSUS_V24_SCHEMA_VERSION: str = (
    "coordpy.consensus_fallback_controller_v24.v1")
W78_CONSENSUS_V24_STAGE_LONG_HORIZON_RECONSTRUCTION_ARBITER: str = (
    "long_horizon_reconstruction_repair_arbiter")
W78_CONSENSUS_V24_STAGE_LONG_HORIZON_RECONSTRUCTION_BEST_PARENT: (
    str) = (
    "long_horizon_reconstruction_best_parent_arbiter")


def _build_v24_stages() -> tuple[str, ...]:
    out: list[str] = []
    inserted = False
    for s in W77_CONSENSUS_V23_STAGES:
        out.append(s)
        if (not inserted and s ==
                W77_CONSENSUS_V23_STAGE_POST_RESTART_REPLACEMENT_BEST_PARENT):
            out.append(
                W78_CONSENSUS_V24_STAGE_LONG_HORIZON_RECONSTRUCTION_ARBITER)
            out.append(
                W78_CONSENSUS_V24_STAGE_LONG_HORIZON_RECONSTRUCTION_BEST_PARENT)
            inserted = True
    if not inserted:
        idx = (
            out.index(W59_CONSENSUS_V5_STAGE_BEST_PARENT)
            if W59_CONSENSUS_V5_STAGE_BEST_PARENT in out
            else len(out))
        out.insert(
            idx,
            W78_CONSENSUS_V24_STAGE_LONG_HORIZON_RECONSTRUCTION_BEST_PARENT)
        out.insert(
            idx,
            W78_CONSENSUS_V24_STAGE_LONG_HORIZON_RECONSTRUCTION_ARBITER)
    return tuple(out)


W78_CONSENSUS_V24_STAGES: tuple[str, ...] = _build_v24_stages()


@dataclasses.dataclass
class ConsensusFallbackControllerV24:
    inner_v23: ConsensusFallbackControllerV23
    long_horizon_reconstruction_threshold: float = 0.5
    long_horizon_reconstruction_best_parent_threshold: float = 0.5
    audit_v24: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    @classmethod
    def init(
            cls, *,
            long_horizon_reconstruction_threshold: float = 0.5,
            long_horizon_reconstruction_best_parent_threshold: (
                float) = 0.5,
            **inner_kwargs: Any,
    ) -> "ConsensusFallbackControllerV24":
        inner = ConsensusFallbackControllerV23.init(
            **inner_kwargs)
        return cls(
            inner_v23=inner,
            long_horizon_reconstruction_threshold=float(
                long_horizon_reconstruction_threshold),
            long_horizon_reconstruction_best_parent_threshold=(
                float(
                    long_horizon_reconstruction_best_parent_threshold)))

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W78_CONSENSUS_V24_SCHEMA_VERSION,
            "kind": "consensus_v24_controller",
            "inner_v23_cid": str(self.inner_v23.cid()),
            "stages": list(W78_CONSENSUS_V24_STAGES),
            "long_horizon_reconstruction_threshold": float(round(
                self.long_horizon_reconstruction_threshold, 12)),
            "long_horizon_reconstruction_best_parent_threshold":
                float(round(
                    self
                    .long_horizon_reconstruction_best_parent_threshold,
                    12)),
        })

    def decide_v24(
            self, *, payloads: Sequence[Sequence[float]],
            trusts: Sequence[float],
            replay_decisions: Sequence[str],
            transcript_available: bool = False,
            long_horizon_reconstruction_scores_per_parent: (
                Sequence[float] | None) = None,
            long_horizon_reconstruction_best_parent_scores_per_parent: (
                Sequence[float] | None) = None,
            **v23_kwargs: Any,
    ) -> dict[str, Any]:
        v23_out = self.inner_v23.decide_v23(
            payloads=payloads, trusts=trusts,
            replay_decisions=replay_decisions,
            transcript_available=bool(transcript_available),
            **v23_kwargs)
        v23_stage = str(v23_out.get("stage", ""))
        terminal_stages = (
            W59_CONSENSUS_V5_STAGE_BEST_PARENT,
            W59_CONSENSUS_V5_STAGE_TRANSCRIPT,
            W59_CONSENSUS_V5_STAGE_ABSTAIN)
        # Stage 41: long-horizon-reconstruction-repair arbiter.
        if (v23_stage in terminal_stages
                and long_horizon_reconstruction_scores_per_parent
                is not None):
            best_idx = -1
            best_score = -1.0
            for i, sc in enumerate(
                    long_horizon_reconstruction_scores_per_parent):
                if (float(sc) >= float(
                        self
                        .long_horizon_reconstruction_threshold)
                        and float(sc) > best_score):
                    best_idx = int(i)
                    best_score = float(sc)
            if best_idx >= 0 and best_idx < len(payloads):
                self.audit_v24.append({
                    "stage": (
                        W78_CONSENSUS_V24_STAGE_LONG_HORIZON_RECONSTRUCTION_ARBITER),
                    "v23_terminal_stage": str(v23_stage),
                    "winning_parent": int(best_idx),
                    "score": float(round(best_score, 12)),
                })
                return {
                    "stage": (
                        W78_CONSENSUS_V24_STAGE_LONG_HORIZON_RECONSTRUCTION_ARBITER),
                    "payload": [
                        float(x) for x in payloads[best_idx]],
                    "v24_promoted": True,
                    "rationale":
                        "long_horizon_reconstruction_applied",
                }
        # Stage 42: long-horizon-reconstruction-best-parent arbiter.
        if (v23_stage in terminal_stages
                and long_horizon_reconstruction_best_parent_scores_per_parent
                is not None):
            best_idx = -1
            best_score = -1.0
            for i, sc in enumerate(
                    long_horizon_reconstruction_best_parent_scores_per_parent):
                if (float(sc) >= float(
                        self
                        .long_horizon_reconstruction_best_parent_threshold)
                        and float(sc) > best_score):
                    best_idx = int(i)
                    best_score = float(sc)
            if best_idx >= 0 and best_idx < len(payloads):
                self.audit_v24.append({
                    "stage": (
                        W78_CONSENSUS_V24_STAGE_LONG_HORIZON_RECONSTRUCTION_BEST_PARENT),
                    "v23_terminal_stage": str(v23_stage),
                    "winning_parent": int(best_idx),
                    "score": float(round(best_score, 12)),
                })
                return {
                    "stage": (
                        W78_CONSENSUS_V24_STAGE_LONG_HORIZON_RECONSTRUCTION_BEST_PARENT),
                    "payload": [
                        float(x) for x in payloads[best_idx]],
                    "v24_promoted": True,
                    "rationale":
                        "long_horizon_reconstruction_best_parent_applied",
                }
        self.audit_v24.append({
            "stage": v23_stage, "v24_promoted": False})
        return v23_out


@dataclasses.dataclass(frozen=True)
class ConsensusV24Witness:
    schema: str
    controller_cid: str
    stages: tuple[str, ...]
    n_decisions: int
    long_horizon_reconstruction_arbiter_fired: int
    long_horizon_reconstruction_best_parent_arbiter_fired: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "controller_cid": str(self.controller_cid),
            "stages": list(self.stages),
            "n_decisions": int(self.n_decisions),
            "long_horizon_reconstruction_arbiter_fired": int(
                self.long_horizon_reconstruction_arbiter_fired),
            "long_horizon_reconstruction_best_parent_arbiter_fired":
                int(
                    self
                    .long_horizon_reconstruction_best_parent_arbiter_fired),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "consensus_v24_witness",
            "witness": self.to_dict()})


def emit_consensus_v24_witness(
        controller: ConsensusFallbackControllerV24,
) -> ConsensusV24Witness:
    lr = sum(
        1 for e in controller.audit_v24
        if str(e.get("stage", ""))
            == W78_CONSENSUS_V24_STAGE_LONG_HORIZON_RECONSTRUCTION_ARBITER)
    lbp = sum(
        1 for e in controller.audit_v24
        if str(e.get("stage", ""))
            == W78_CONSENSUS_V24_STAGE_LONG_HORIZON_RECONSTRUCTION_BEST_PARENT)
    return ConsensusV24Witness(
        schema=W78_CONSENSUS_V24_SCHEMA_VERSION,
        controller_cid=str(controller.cid()),
        stages=tuple(W78_CONSENSUS_V24_STAGES),
        n_decisions=int(len(controller.audit_v24)),
        long_horizon_reconstruction_arbiter_fired=int(lr),
        long_horizon_reconstruction_best_parent_arbiter_fired=int(
            lbp),
    )


__all__ = [
    "W78_CONSENSUS_V24_SCHEMA_VERSION",
    "W78_CONSENSUS_V24_STAGE_LONG_HORIZON_RECONSTRUCTION_ARBITER",
    "W78_CONSENSUS_V24_STAGE_LONG_HORIZON_RECONSTRUCTION_BEST_PARENT",
    "W78_CONSENSUS_V24_STAGES",
    "ConsensusFallbackControllerV24",
    "ConsensusV24Witness",
    "emit_consensus_v24_witness",
]
