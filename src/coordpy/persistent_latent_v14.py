"""W62 M8 — Persistent Latent State V14.

Strictly extends W61's ``coordpy.persistent_latent_v13``. V14
adds:

* **12 layers** (vs V13's 11) — wrapper of V13's nonuple GRU stack.
* **Decuple persistent skip-link** — V13's nonuple plus a tenth:
  a **replay-dominance EMA** that carries the V62 replay
  controller V3's softmax margin (top - second-best) scalar across
  turns. The persistent cell uses it to amplify the replay channel
  when the controller is *strongly* dominant.
* ``max_chain_walk_depth = 2048`` (vs V13's 1536).
* **Larger distractor basis** — V13 was rank-6; V14 is **rank-8**
  so the replay-skip projection survives more aggressive
  distractor attacks.

V14 strictly extends V13: with ``replay_dominance_skip = None``,
the new tenth EMA stays at the prior value (no-op) and V14 reduces
to V13 byte-for-byte.

Honest scope
------------

* V14 wrapper still does NOT train the V13 outer GRU.
  ``W62-L-V14-OUTER-NOT-TRAINED-CAP`` documents the new cap.
* The tenth EMA shares the same shape and update law as V13's
  nine EMAs.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Sequence

from .persistent_latent_v13 import (
    PersistentLatentStateV13,
    PersistentLatentStateV13Chain,
    W61_DEFAULT_V13_STATE_DIM,
    W61_DEFAULT_V13_N_LAYERS,
    W61_DEFAULT_V13_DISTRACTOR_RANK,
    step_persistent_state_v13,
)
from .persistent_latent_v12 import V12StackedCell
from .tiny_substrate_v3 import _sha256_hex


W62_PERSISTENT_V14_SCHEMA_VERSION: str = (
    "coordpy.persistent_latent_v14.v1")
W62_DEFAULT_V14_STATE_DIM: int = W61_DEFAULT_V13_STATE_DIM
W62_DEFAULT_V14_N_LAYERS: int = 12
W62_DEFAULT_V14_MAX_CHAIN_WALK_DEPTH: int = 2048
W62_DEFAULT_V14_DISTRACTOR_RANK: int = 8
W62_V14_NO_PARENT_STATE: str = "no_parent_v14_state"


@dataclasses.dataclass(frozen=True)
class PersistentLatentStateV14:
    inner_v13: PersistentLatentStateV13
    replay_dominance_carrier: tuple[float, ...]
    distractor_rank: int

    @property
    def turn_index(self) -> int:
        return int(self.inner_v13.turn_index)

    @property
    def role(self) -> str:
        return str(self.inner_v13.role)

    @property
    def state_dim(self) -> int:
        return int(self.inner_v13.state_dim)

    @property
    def n_layers(self) -> int:
        return int(self.inner_v13.n_layers) + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W62_PERSISTENT_V14_SCHEMA_VERSION,
            "inner_v13_cid": str(self.inner_v13.cid()),
            "replay_dominance_carrier": [
                float(round(float(x), 12))
                for x in self.replay_dominance_carrier],
            "distractor_rank": int(self.distractor_rank),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w62_v14_persistent_state",
            "state": self.to_dict()})


@dataclasses.dataclass
class PersistentLatentStateV14Chain:
    states: dict[str, PersistentLatentStateV14] = (
        dataclasses.field(default_factory=dict))

    @classmethod
    def empty(cls) -> "PersistentLatentStateV14Chain":
        return cls(states={})

    def add(self, s: PersistentLatentStateV14) -> None:
        self.states[s.cid()] = s

    def get(
            self, cid: str,
    ) -> PersistentLatentStateV14 | None:
        return self.states.get(str(cid))

    def walk_from(
            self, leaf_cid: str, *,
            max_depth: int = W62_DEFAULT_V14_MAX_CHAIN_WALK_DEPTH,
    ) -> list[PersistentLatentStateV14]:
        out: list[PersistentLatentStateV14] = []
        cur = self.get(leaf_cid)
        seen: set[str] = set()
        steps = 0
        while cur is not None and steps < int(max_depth):
            out.append(cur)
            seen.add(cur.cid())
            inner_parent_cid = (
                cur.inner_v13.inner_v12.parent_state_cid)
            parent_inner = None
            for c in self.states.values():
                if c.inner_v13.cid() == inner_parent_cid:
                    parent_inner = c
                    break
            if parent_inner is None or parent_inner.cid() in seen:
                break
            cur = parent_inner
            steps += 1
        return out

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w62_v14_state_chain",
            "members": [
                {"cid": c, "state": s.to_dict()}
                for c, s in sorted(self.states.items())],
        })


def step_persistent_state_v14(
        *, cell: V12StackedCell,
        prev_state: PersistentLatentStateV14 | None,
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
        substrate_fidelity: float = 1.0,
        attention_fidelity: float = 1.0,
        retrieval_fidelity: float = 1.0,
        replay_fidelity: float = 1.0,
        replay_dominance_ema_alpha: float = 0.12,
) -> PersistentLatentStateV14:
    """Compute one V14 step: V13 step + replay-dominance EMA
    update."""
    prev_v13 = (
        prev_state.inner_v13 if prev_state is not None else None)
    new_v13 = step_persistent_state_v13(
        cell=cell, prev_state=prev_v13,
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
        substrate_fidelity=float(substrate_fidelity),
        attention_fidelity=float(attention_fidelity),
        retrieval_fidelity=float(retrieval_fidelity),
        replay_fidelity=float(replay_fidelity))
    sd = int(new_v13.state_dim)
    if prev_state is not None:
        prev_rd = list(prev_state.replay_dominance_carrier)
    else:
        prev_rd = [0.0] * sd
    if replay_dominance_skip is not None:
        rds = list(replay_dominance_skip)[:sd]
        while len(rds) < sd:
            rds.append(0.0)
        a = float(max(0.0, min(
            1.0, float(replay_dominance_ema_alpha))))
        new_rd = [
            a * float(rds[i]) + (1.0 - a) * float(
                prev_rd[i] if i < len(prev_rd) else 0.0)
            for i in range(sd)]
    else:
        new_rd = list(prev_rd)
        while len(new_rd) < sd:
            new_rd.append(0.0)
    return PersistentLatentStateV14(
        inner_v13=new_v13,
        replay_dominance_carrier=tuple(new_rd),
        distractor_rank=int(W62_DEFAULT_V14_DISTRACTOR_RANK),
    )


@dataclasses.dataclass(frozen=True)
class PersistentLatentStateV14Witness:
    schema: str
    chain_cid: str
    n_states: int
    chain_walk_depth_used: int
    n_layers: int
    distractor_rank: int
    decuple_skip_present: bool
    replay_dominance_carrier_l1_sum: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "chain_cid": str(self.chain_cid),
            "n_states": int(self.n_states),
            "chain_walk_depth_used": int(
                self.chain_walk_depth_used),
            "n_layers": int(self.n_layers),
            "distractor_rank": int(self.distractor_rank),
            "decuple_skip_present": bool(
                self.decuple_skip_present),
            "replay_dominance_carrier_l1_sum": float(round(
                self.replay_dominance_carrier_l1_sum, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w62_persistent_v14_witness",
            "witness": self.to_dict()})


def emit_persistent_v14_witness(
        chain: PersistentLatentStateV14Chain,
        last_leaf_cid: str,
        *, max_depth: int = W62_DEFAULT_V14_MAX_CHAIN_WALK_DEPTH,
) -> PersistentLatentStateV14Witness:
    walk = chain.walk_from(
        last_leaf_cid, max_depth=int(max_depth))
    rd_sum = float(sum(
        abs(float(v))
        for s in walk
        for v in s.replay_dominance_carrier))
    return PersistentLatentStateV14Witness(
        schema=W62_PERSISTENT_V14_SCHEMA_VERSION,
        chain_cid=str(chain.cid()),
        n_states=int(len(chain.states)),
        chain_walk_depth_used=int(len(walk)),
        n_layers=int(W62_DEFAULT_V14_N_LAYERS),
        distractor_rank=int(W62_DEFAULT_V14_DISTRACTOR_RANK),
        decuple_skip_present=bool(rd_sum > 0.0),
        replay_dominance_carrier_l1_sum=float(rd_sum),
    )


__all__ = [
    "W62_PERSISTENT_V14_SCHEMA_VERSION",
    "W62_DEFAULT_V14_STATE_DIM",
    "W62_DEFAULT_V14_N_LAYERS",
    "W62_DEFAULT_V14_MAX_CHAIN_WALK_DEPTH",
    "W62_DEFAULT_V14_DISTRACTOR_RANK",
    "W62_V14_NO_PARENT_STATE",
    "PersistentLatentStateV14",
    "PersistentLatentStateV14Chain",
    "PersistentLatentStateV14Witness",
    "step_persistent_state_v14",
    "emit_persistent_v14_witness",
]
