"""W64 M10 — Persistent Latent State V16.

Strictly extends W63's ``coordpy.persistent_latent_v15``. V16
adds:

* **15 layers** (vs V15's 14).
* **Thirteenth persistent skip-link** — V15's twelve plus a new
  *replay-dominance-witness EMA* that carries the W64 V9 substrate's
  replay_dominance_witness scalar.
* ``max_chain_walk_depth = 6144`` (vs V15's 4096).
* **Larger distractor basis** — V15 was rank-10; V16 is **rank-12**
  so the carriers survive even more aggressive distractor attacks.

V16 strictly extends V15: with ``replay_dominance_skip_v16 = None``,
the new EMA stays at the prior value (no-op) and V16 reduces to
V15 byte-for-byte.

Honest scope (W64)
------------------

* V16 wrapper still does NOT train the V13 outer GRU.
  ``W64-L-V16-OUTER-NOT-TRAINED-CAP`` documents the new cap.
* The new EMA shares the same shape and update law as V15's EMAs.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Sequence

from .persistent_latent_v12 import V12StackedCell
from .persistent_latent_v15 import (
    PersistentLatentStateV15,
    PersistentLatentStateV15Chain,
    W63_DEFAULT_V15_DISTRACTOR_RANK,
    W63_DEFAULT_V15_STATE_DIM,
    step_persistent_state_v15,
)
from .tiny_substrate_v3 import _sha256_hex


W64_PERSISTENT_V16_SCHEMA_VERSION: str = (
    "coordpy.persistent_latent_v16.v1")
W64_DEFAULT_V16_STATE_DIM: int = W63_DEFAULT_V15_STATE_DIM
W64_DEFAULT_V16_N_LAYERS: int = 15
W64_DEFAULT_V16_MAX_CHAIN_WALK_DEPTH: int = 6144
W64_DEFAULT_V16_DISTRACTOR_RANK: int = 12
W64_V16_NO_PARENT_STATE: str = "no_parent_v16_state"


@dataclasses.dataclass(frozen=True)
class PersistentLatentStateV16:
    inner_v15: PersistentLatentStateV15
    replay_dominance_witness_carrier: tuple[float, ...]
    distractor_rank: int

    @property
    def turn_index(self) -> int:
        return int(self.inner_v15.turn_index)

    @property
    def role(self) -> str:
        return str(self.inner_v15.role)

    @property
    def state_dim(self) -> int:
        return int(self.inner_v15.state_dim)

    @property
    def n_layers(self) -> int:
        return int(self.inner_v15.n_layers) + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W64_PERSISTENT_V16_SCHEMA_VERSION,
            "inner_v15_cid": str(self.inner_v15.cid()),
            "replay_dominance_witness_carrier": [
                float(round(float(x), 12))
                for x in self.replay_dominance_witness_carrier],
            "distractor_rank": int(self.distractor_rank),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w64_v16_persistent_state",
            "state": self.to_dict()})


@dataclasses.dataclass
class PersistentLatentStateV16Chain:
    states: dict[str, PersistentLatentStateV16] = (
        dataclasses.field(default_factory=dict))

    @classmethod
    def empty(cls) -> "PersistentLatentStateV16Chain":
        return cls(states={})

    def add(self, s: PersistentLatentStateV16) -> None:
        self.states[s.cid()] = s

    def get(
            self, cid: str,
    ) -> PersistentLatentStateV16 | None:
        return self.states.get(str(cid))

    def walk_from(
            self, leaf_cid: str, *,
            max_depth: int = W64_DEFAULT_V16_MAX_CHAIN_WALK_DEPTH,
    ) -> list[PersistentLatentStateV16]:
        out: list[PersistentLatentStateV16] = []
        cur = self.get(leaf_cid)
        seen: set[str] = set()
        steps = 0
        while cur is not None and steps < int(max_depth):
            out.append(cur)
            seen.add(cur.cid())
            inner_parent_cid = (
                cur.inner_v15.inner_v14.inner_v13
                .inner_v12.parent_state_cid)
            parent_inner = None
            for c in self.states.values():
                if (c.inner_v15.inner_v14.inner_v13.cid()
                        == inner_parent_cid):
                    parent_inner = c
                    break
            if parent_inner is None or parent_inner.cid() in seen:
                break
            cur = parent_inner
            steps += 1
        return out

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w64_v16_state_chain",
            "members": [
                {"cid": c, "state": s.to_dict()}
                for c, s in sorted(self.states.items())],
        })


def step_persistent_state_v16(
        *, cell: V12StackedCell,
        prev_state: PersistentLatentStateV16 | None,
        carrier_values: Sequence[float],
        turn_index: int, role: str,
        branch_id: str = "main",
        anchor_skip: Sequence[float] | None = None,
        substrate_skip: Sequence[float] | None = None,
        hidden_state_skip: Sequence[float] | None = None,
        attention_skip: Sequence[float] | None = None,
        retrieval_skip: Sequence[float] | None = None,
        replay_skip: Sequence[float] | None = None,
        replay_confidence_skip: Sequence[float] | None = None,
        replay_dominance_skip: Sequence[float] | None = None,
        hidden_wins_skip: Sequence[float] | None = None,
        prefix_reuse_skip: Sequence[float] | None = None,
        replay_dominance_witness_skip_v16: (
            Sequence[float] | None) = None,
        substrate_fidelity: float = 1.0,
        attention_fidelity: float = 1.0,
        retrieval_fidelity: float = 1.0,
        replay_fidelity: float = 1.0,
        hidden_wins_ema_alpha: float = 0.10,
        prefix_reuse_ema_alpha: float = 0.10,
        replay_dominance_witness_ema_alpha: float = 0.10,
) -> PersistentLatentStateV16:
    """V16 step: V15 step + replay-dominance-witness EMA."""
    prev_v15 = (
        prev_state.inner_v15 if prev_state is not None else None)
    new_v15 = step_persistent_state_v15(
        cell=cell, prev_state=prev_v15,
        carrier_values=list(carrier_values),
        turn_index=int(turn_index), role=str(role),
        branch_id=str(branch_id),
        anchor_skip=anchor_skip,
        substrate_skip=substrate_skip,
        hidden_state_skip=hidden_state_skip,
        attention_skip=attention_skip,
        retrieval_skip=retrieval_skip,
        replay_skip=replay_skip,
        replay_confidence_skip=replay_confidence_skip,
        replay_dominance_skip=replay_dominance_skip,
        hidden_wins_skip=hidden_wins_skip,
        prefix_reuse_skip=prefix_reuse_skip,
        substrate_fidelity=float(substrate_fidelity),
        attention_fidelity=float(attention_fidelity),
        retrieval_fidelity=float(retrieval_fidelity),
        replay_fidelity=float(replay_fidelity),
        hidden_wins_ema_alpha=float(hidden_wins_ema_alpha),
        prefix_reuse_ema_alpha=float(prefix_reuse_ema_alpha))
    sd = int(new_v15.state_dim)
    if prev_state is not None:
        prev_rdw = list(prev_state.replay_dominance_witness_carrier)
    else:
        prev_rdw = [0.0] * sd
    if replay_dominance_witness_skip_v16 is not None:
        rdws = list(replay_dominance_witness_skip_v16)[:sd]
        while len(rdws) < sd:
            rdws.append(0.0)
        a = float(max(0.0, min(
            1.0, float(replay_dominance_witness_ema_alpha))))
        new_rdw = [
            a * float(rdws[i]) + (1.0 - a) * float(
                prev_rdw[i] if i < len(prev_rdw) else 0.0)
            for i in range(sd)]
    else:
        new_rdw = list(prev_rdw)
        while len(new_rdw) < sd:
            new_rdw.append(0.0)
    return PersistentLatentStateV16(
        inner_v15=new_v15,
        replay_dominance_witness_carrier=tuple(new_rdw),
        distractor_rank=int(W64_DEFAULT_V16_DISTRACTOR_RANK),
    )


@dataclasses.dataclass(frozen=True)
class PersistentLatentStateV16Witness:
    schema: str
    chain_cid: str
    n_states: int
    chain_walk_depth_used: int
    n_layers: int
    distractor_rank: int
    thirteenth_skip_present: bool
    replay_dominance_witness_carrier_l1_sum: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "chain_cid": str(self.chain_cid),
            "n_states": int(self.n_states),
            "chain_walk_depth_used": int(
                self.chain_walk_depth_used),
            "n_layers": int(self.n_layers),
            "distractor_rank": int(self.distractor_rank),
            "thirteenth_skip_present": bool(
                self.thirteenth_skip_present),
            "replay_dominance_witness_carrier_l1_sum": float(
                round(
                    self.replay_dominance_witness_carrier_l1_sum,
                    12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w64_persistent_v16_witness",
            "witness": self.to_dict()})


def emit_persistent_v16_witness(
        chain: PersistentLatentStateV16Chain,
        last_leaf_cid: str,
        *, max_depth: int = (
            W64_DEFAULT_V16_MAX_CHAIN_WALK_DEPTH),
) -> PersistentLatentStateV16Witness:
    walk = chain.walk_from(
        last_leaf_cid, max_depth=int(max_depth))
    rdw_sum = float(sum(
        abs(float(v))
        for s in walk
        for v in s.replay_dominance_witness_carrier))
    return PersistentLatentStateV16Witness(
        schema=W64_PERSISTENT_V16_SCHEMA_VERSION,
        chain_cid=str(chain.cid()),
        n_states=int(len(chain.states)),
        chain_walk_depth_used=int(len(walk)),
        n_layers=int(W64_DEFAULT_V16_N_LAYERS),
        distractor_rank=int(W64_DEFAULT_V16_DISTRACTOR_RANK),
        thirteenth_skip_present=bool(rdw_sum > 0.0),
        replay_dominance_witness_carrier_l1_sum=float(rdw_sum),
    )


__all__ = [
    "W64_PERSISTENT_V16_SCHEMA_VERSION",
    "W64_DEFAULT_V16_STATE_DIM",
    "W64_DEFAULT_V16_N_LAYERS",
    "W64_DEFAULT_V16_MAX_CHAIN_WALK_DEPTH",
    "W64_DEFAULT_V16_DISTRACTOR_RANK",
    "W64_V16_NO_PARENT_STATE",
    "PersistentLatentStateV16",
    "PersistentLatentStateV16Chain",
    "PersistentLatentStateV16Witness",
    "step_persistent_state_v16",
    "emit_persistent_v16_witness",
]
