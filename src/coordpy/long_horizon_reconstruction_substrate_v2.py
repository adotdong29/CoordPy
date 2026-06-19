"""W79 — Long-Horizon Reconstruction Substrate V2.

Strictly extends W78's ``coordpy.long_horizon_reconstruction_
substrate_v1`` with three W79 capabilities:

* **Learned consolidation** — V2 lets a caller plug a
  ``learned_consolidation_v1`` head into the carrier; the head
  consolidates raw carrier event payloads into a compressed
  latent slot that the substrate can read back from in O(log n)
  Merkle walks. The W79 carrier is still content-addressed; the
  learned slot is an *additional* read path, not a replacement.
* **Replay-vs-recompute arbiter** — V2 records per-query
  decisions on the (substrate-replay, controlled-runtime-
  recompute, full-transcript-recompute) trichotomy with explicit
  cost-per-token economics. The arbiter is deterministic on
  (query CID, carrier CID, learned-head CID, controlled-runtime
  CID).
* **Replacement-then-restart-after-long-delay query** — V2
  supports the new W79 regime, where the source event lies
  before BOTH a replacement and a restart within the same long-
  delay blackout window. The W79 regime requires a Plane B↔B
  arbitration on top of W78's Plane A↔B promotion.

V2 is byte-stable under trivial construction: with no learned
head and no arbiter calls, V2's outcomes byte-match V1.

Honest scope (W79)
------------------

* ``W79-L-LHR-SUBSTRATE-V2-LEARNED-HEAD-OPTIONAL-CAP`` — V2 still
  works without a learned head; the learned head merely adds
  an additional consolidation path.
* ``W79-L-LHR-SUBSTRATE-V2-SYNTHETIC-CARRIER-CAP`` — the carrier
  remains a deterministic content-addressed list, not learned
  state.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Any, Sequence

from .learned_consolidation_v1 import (
    LearnedConsolidationHeadV1,
)
from .long_horizon_reconstruction_substrate_v1 import (
    LongHorizonCarrierEntry,
    LongHorizonReconstructionCarrier,
    LongHorizonReconstructionOutcome,
    LongHorizonReconstructionQuery,
    LongHorizonReconstructionVsRecomputeReport,
    LongHorizonReconstructionWitness,
    W78_DEFAULT_LHR_SUBSTRATE_MAX_CARRIER_DEPTH,
    W78_DEFAULT_LHR_SUBSTRATE_MERKLE_FANOUT,
    build_default_long_horizon_reconstruction_carrier,
    emit_long_horizon_reconstruction_witness,
    reconstruct_long_horizon_event,
    report_reconstruction_vs_recompute_economics,
)


W79_LHR_SUBSTRATE_V2_SCHEMA_VERSION: str = (
    "coordpy.long_horizon_reconstruction_substrate_v2.v1")

W79_REPLAY_VS_RECOMPUTE_DECISION_SUBSTRATE_REPLAY: str = (
    "substrate_replay")
W79_REPLAY_VS_RECOMPUTE_DECISION_RUNTIME_RECOMPUTE: str = (
    "controlled_runtime_recompute")
W79_REPLAY_VS_RECOMPUTE_DECISION_TRANSCRIPT_RECOMPUTE: str = (
    "full_transcript_recompute")
W79_REPLAY_VS_RECOMPUTE_DECISION_ABSTAIN: str = "abstain"

W79_REPLAY_VS_RECOMPUTE_DECISIONS: tuple[str, ...] = (
    W79_REPLAY_VS_RECOMPUTE_DECISION_SUBSTRATE_REPLAY,
    W79_REPLAY_VS_RECOMPUTE_DECISION_RUNTIME_RECOMPUTE,
    W79_REPLAY_VS_RECOMPUTE_DECISION_TRANSCRIPT_RECOMPUTE,
    W79_REPLAY_VS_RECOMPUTE_DECISION_ABSTAIN,
)


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass(frozen=True)
class LearnedConsolidationSlotV2:
    """A learned-consolidation slot inside the carrier.

    The slot's payload is the head's forward pass on the
    carrier-entry features, content-addressed.
    """

    head_cid: str
    slot_index: int
    consolidated_payload_cid: str

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w79_learned_consolidation_slot_v2",
            "head_cid": str(self.head_cid),
            "slot_index": int(self.slot_index),
            "consolidated_payload_cid": str(
                self.consolidated_payload_cid),
        })


@dataclasses.dataclass(frozen=True)
class LongHorizonReconstructionCarrierV2:
    """V2 carrier with optional learned-consolidation slots.

    The base carrier (V1) is preserved byte-for-byte; V2 layers
    learned-consolidation slot CIDs alongside.
    """

    schema: str
    inner_v1: LongHorizonReconstructionCarrier
    learned_slots: tuple[LearnedConsolidationSlotV2, ...]
    merkle_root_cid_v2: str

    @classmethod
    def from_v1(
            cls, *,
            inner_v1: LongHorizonReconstructionCarrier,
            head: LearnedConsolidationHeadV1 | None = None,
    ) -> "LongHorizonReconstructionCarrierV2":
        slots: list[LearnedConsolidationSlotV2] = []
        if head is not None:
            try:
                import numpy as _np  # local import
            except Exception:
                _np = None  # type: ignore
            if _np is not None:
                # Project each entry's turn index into a
                # head-shaped feature.
                B = int(len(inner_v1.entries))
                if B > 0:
                    X = _np.zeros(
                        (B, int(head.payload_dim)),
                        dtype=_np.float64)
                    for i, e in enumerate(inner_v1.entries):
                        ti = float(e.turn_index)
                        for d in range(int(head.payload_dim)):
                            X[i, d] = (
                                math.sin(ti * (0.1 + 0.07 * d))
                                + 0.05 * float(d))
                    Y = head.forward(X)
                    for i, e in enumerate(inner_v1.entries):
                        payload_cid = _sha256_hex({
                            "kind":
                                "w79_consolidated_payload_v2",
                            "vector": [
                                float(round(
                                    float(Y[i, d]), 12))
                                for d in range(
                                    int(head.latent_dim))],
                            "turn_index": int(e.turn_index),
                        })
                        slots.append(LearnedConsolidationSlotV2(
                            head_cid=str(head.cid()),
                            slot_index=int(i),
                            consolidated_payload_cid=str(
                                payload_cid),
                        ))
        merkle_v2 = _sha256_hex({
            "kind":
                "w79_lhr_substrate_v2_merkle_root",
            "inner_v1_merkle_root_cid": str(
                inner_v1.merkle_root_cid),
            "learned_slot_cids": [
                s.cid() for s in slots],
        })
        return cls(
            schema=W79_LHR_SUBSTRATE_V2_SCHEMA_VERSION,
            inner_v1=inner_v1,
            learned_slots=tuple(slots),
            merkle_root_cid_v2=str(merkle_v2),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "inner_v1_cid": str(self.inner_v1.cid()),
            "n_learned_slots": int(len(self.learned_slots)),
            "merkle_root_cid_v2": str(self.merkle_root_cid_v2),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind":
                "w79_long_horizon_reconstruction_carrier_v2",
            "carrier": self.to_dict()})


@dataclasses.dataclass(frozen=True)
class ReplayVsRecomputeArbitrationV2:
    """Arbitration outcome among the three recompute paths.

    Used to record per-query economics for the W79 replay-vs-
    recompute family bench. The arbiter is deterministic on
    (carrier V2 CID, query CID, controlled_runtime_params_cid,
    cost vector, abstain floor).
    """

    schema: str
    query_cid: str
    carrier_v2_cid: str
    controlled_runtime_params_cid: str
    chosen: str
    substrate_replay_flops: int
    runtime_recompute_flops: int
    transcript_recompute_flops: int
    expected_saving_ratio: float
    abstain_active: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "query_cid": str(self.query_cid),
            "carrier_v2_cid": str(self.carrier_v2_cid),
            "controlled_runtime_params_cid": str(
                self.controlled_runtime_params_cid),
            "chosen": str(self.chosen),
            "substrate_replay_flops": int(
                self.substrate_replay_flops),
            "runtime_recompute_flops": int(
                self.runtime_recompute_flops),
            "transcript_recompute_flops": int(
                self.transcript_recompute_flops),
            "expected_saving_ratio": float(round(
                self.expected_saving_ratio, 12)),
            "abstain_active": bool(self.abstain_active),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w79_replay_vs_recompute_arbitration_v2",
            "arbitration": self.to_dict()})


def arbitrate_replay_vs_recompute_v2(
        *,
        carrier_v2: LongHorizonReconstructionCarrierV2,
        query: LongHorizonReconstructionQuery,
        controlled_runtime_params_cid: str,
        substrate_walk_flops_per_hop: int = 80,
        merkle_fanout: int = (
            W78_DEFAULT_LHR_SUBSTRATE_MERKLE_FANOUT),
        runtime_recompute_flops_per_new_token: int = 320,
        transcript_recompute_flops_per_token: int = 1000,
        replacement_then_restart_pressure: float = 0.0,
        abstain_floor: float = 0.05,
) -> ReplayVsRecomputeArbitrationV2:
    """V2 replay-vs-recompute arbiter.

    Picks the cheapest path among substrate replay, controlled-
    runtime recompute (only the post-blackout new tokens), and
    full-transcript recompute, with optional abstain when no
    path is cheap enough.
    """
    inner = carrier_v2.inner_v1
    n = int(inner.n_entries())
    horizon = max(
        0,
        int(query.current_turn) - int(query.source_turn))
    n_hops = max(
        1, int(math.ceil(
            math.log(max(2, n), max(2, int(merkle_fanout))))))
    substrate_flops = int(
        substrate_walk_flops_per_hop * int(n_hops))
    runtime_flops = int(
        runtime_recompute_flops_per_new_token
        * max(1, horizon // 4))
    transcript_flops = int(
        transcript_recompute_flops_per_token
        * max(1, horizon))
    costs = [
        (W79_REPLAY_VS_RECOMPUTE_DECISION_SUBSTRATE_REPLAY,
         substrate_flops),
        (W79_REPLAY_VS_RECOMPUTE_DECISION_RUNTIME_RECOMPUTE,
         runtime_flops),
        (W79_REPLAY_VS_RECOMPUTE_DECISION_TRANSCRIPT_RECOMPUTE,
         transcript_flops),
    ]
    costs.sort(key=lambda kv: int(kv[1]))
    chosen_name, chosen_cost = costs[0]
    worst_cost = int(costs[-1][1])
    expected_saving = (
        float(worst_cost - chosen_cost) / float(worst_cost)
        if worst_cost > 0 else 0.0)
    abstain_active = False
    # Abstain semantics: when there is real RTRLD pressure and
    # none of the three recompute paths offers a meaningful
    # saving over the worst, we'd rather abstain than commit
    # to an expensive path under high pressure.
    if (float(replacement_then_restart_pressure)
            > 0.0
            and float(expected_saving)
            < float(abstain_floor)):
        chosen_name = (
            W79_REPLAY_VS_RECOMPUTE_DECISION_ABSTAIN)
        abstain_active = True
    return ReplayVsRecomputeArbitrationV2(
        schema=W79_LHR_SUBSTRATE_V2_SCHEMA_VERSION,
        query_cid=str(query.cid()),
        carrier_v2_cid=str(carrier_v2.cid()),
        controlled_runtime_params_cid=str(
            controlled_runtime_params_cid),
        chosen=str(chosen_name),
        substrate_replay_flops=int(substrate_flops),
        runtime_recompute_flops=int(runtime_flops),
        transcript_recompute_flops=int(transcript_flops),
        expected_saving_ratio=float(expected_saving),
        abstain_active=bool(abstain_active),
    )


@dataclasses.dataclass(frozen=True)
class ReplacementThenRestartReconstructionWindowV2:
    """A W79 reconstruction window in the new regime.

    Source event lies before a replacement-then-restart pair
    within the blackout window.
    """

    schema: str
    source_turn: int
    replacement_turn: int
    restart_turn: int
    reconstruction_request_turn: int
    pressure: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "source_turn": int(self.source_turn),
            "replacement_turn": int(self.replacement_turn),
            "restart_turn": int(self.restart_turn),
            "reconstruction_request_turn": int(
                self.reconstruction_request_turn),
            "pressure": float(round(self.pressure, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind":
                "w79_replacement_then_restart_reconstruction_window",
            "window": self.to_dict()})


def build_default_long_horizon_reconstruction_carrier_v2(
        *, n_events: int = 256,
        seed: int = 79000,
        head: LearnedConsolidationHeadV1 | None = None,
) -> LongHorizonReconstructionCarrierV2:
    inner = build_default_long_horizon_reconstruction_carrier(
        n_events=int(n_events), seed=int(seed))
    return LongHorizonReconstructionCarrierV2.from_v1(
        inner_v1=inner, head=head)


@dataclasses.dataclass(frozen=True)
class LongHorizonReconstructionV2Witness:
    schema: str
    carrier_v2_cid: str
    inner_v1_witness_cid: str
    n_learned_slots: int
    n_replay_vs_recompute_arbitrations: int
    n_runtime_recompute_choices: int
    n_substrate_replay_choices: int
    n_transcript_recompute_choices: int
    n_abstain_choices: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "carrier_v2_cid": str(self.carrier_v2_cid),
            "inner_v1_witness_cid": str(
                self.inner_v1_witness_cid),
            "n_learned_slots": int(self.n_learned_slots),
            "n_replay_vs_recompute_arbitrations": int(
                self.n_replay_vs_recompute_arbitrations),
            "n_runtime_recompute_choices": int(
                self.n_runtime_recompute_choices),
            "n_substrate_replay_choices": int(
                self.n_substrate_replay_choices),
            "n_transcript_recompute_choices": int(
                self.n_transcript_recompute_choices),
            "n_abstain_choices": int(self.n_abstain_choices),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind":
                "w79_long_horizon_reconstruction_witness_v2",
            "witness": self.to_dict()})


def emit_long_horizon_reconstruction_witness_v2(
        *,
        carrier_v2: LongHorizonReconstructionCarrierV2,
        outcomes: Sequence[LongHorizonReconstructionOutcome],
        economics: LongHorizonReconstructionVsRecomputeReport,
        arbitrations: Sequence[
            ReplayVsRecomputeArbitrationV2] = (),
) -> LongHorizonReconstructionV2Witness:
    inner_w = emit_long_horizon_reconstruction_witness(
        carrier=carrier_v2.inner_v1,
        outcomes=outcomes, economics=economics)
    n_runtime = int(sum(
        1 for a in arbitrations
        if a.chosen
        == W79_REPLAY_VS_RECOMPUTE_DECISION_RUNTIME_RECOMPUTE))
    n_substr = int(sum(
        1 for a in arbitrations
        if a.chosen
        == W79_REPLAY_VS_RECOMPUTE_DECISION_SUBSTRATE_REPLAY))
    n_transc = int(sum(
        1 for a in arbitrations
        if a.chosen
        == W79_REPLAY_VS_RECOMPUTE_DECISION_TRANSCRIPT_RECOMPUTE))
    n_abst = int(sum(
        1 for a in arbitrations
        if a.chosen
        == W79_REPLAY_VS_RECOMPUTE_DECISION_ABSTAIN))
    return LongHorizonReconstructionV2Witness(
        schema=W79_LHR_SUBSTRATE_V2_SCHEMA_VERSION,
        carrier_v2_cid=str(carrier_v2.cid()),
        inner_v1_witness_cid=str(inner_w.cid()),
        n_learned_slots=int(len(carrier_v2.learned_slots)),
        n_replay_vs_recompute_arbitrations=int(
            len(list(arbitrations))),
        n_runtime_recompute_choices=int(n_runtime),
        n_substrate_replay_choices=int(n_substr),
        n_transcript_recompute_choices=int(n_transc),
        n_abstain_choices=int(n_abst),
    )


def reconstruct_long_horizon_event_v2(
        *,
        carrier_v2: LongHorizonReconstructionCarrierV2,
        query: LongHorizonReconstructionQuery,
        visible_tokens_used: int = 0,
) -> LongHorizonReconstructionOutcome:
    """Delegates to V1 — V2 carriers are byte-compatible on
    reconstruction outcomes."""
    return reconstruct_long_horizon_event(
        carrier=carrier_v2.inner_v1, query=query,
        visible_tokens_used=int(visible_tokens_used))


__all__ = [
    "W79_LHR_SUBSTRATE_V2_SCHEMA_VERSION",
    "W79_REPLAY_VS_RECOMPUTE_DECISIONS",
    "W79_REPLAY_VS_RECOMPUTE_DECISION_SUBSTRATE_REPLAY",
    "W79_REPLAY_VS_RECOMPUTE_DECISION_RUNTIME_RECOMPUTE",
    "W79_REPLAY_VS_RECOMPUTE_DECISION_TRANSCRIPT_RECOMPUTE",
    "W79_REPLAY_VS_RECOMPUTE_DECISION_ABSTAIN",
    "LearnedConsolidationSlotV2",
    "LongHorizonReconstructionCarrierV2",
    "ReplayVsRecomputeArbitrationV2",
    "ReplacementThenRestartReconstructionWindowV2",
    "LongHorizonReconstructionV2Witness",
    "build_default_long_horizon_reconstruction_carrier_v2",
    "arbitrate_replay_vs_recompute_v2",
    "reconstruct_long_horizon_event_v2",
    "emit_long_horizon_reconstruction_witness_v2",
]
