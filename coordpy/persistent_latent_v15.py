"""W63 M11 — Persistent Latent State V15.

Strictly extends W62's ``coordpy.persistent_latent_v14``. V15
adds:

* **14 layers** (vs V14's 12).
* **Twelfth persistent skip-link** — V14's decuple plus two new:
  a **hidden-wins EMA** that carries the W63 hidden-vs-KV
  contention margin scalar, and a **prefix-reuse trust EMA**
  that carries the V8 prefix-reuse trust ledger scalar.
* ``max_chain_walk_depth = 4096`` (vs V14's 2048).
* **Larger distractor basis** — V14 was rank-8; V15 is **rank-10**
  so the carriers survive even more aggressive distractor attacks.

V15 strictly extends V14: with ``hidden_wins_skip = None`` AND
``prefix_reuse_skip = None``, the two new EMAs stay at the prior
value (no-op) and V15 reduces to V14 byte-for-byte.

Honest scope
------------

* V15 wrapper still does NOT train the V13 outer GRU.
  ``W63-L-V15-OUTER-NOT-TRAINED-CAP`` documents the new cap.
* The two new EMAs share the same shape and update law as V14's
  ten EMAs.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Sequence

from .persistent_latent_v12 import V12StackedCell
from .persistent_latent_v14 import (
    PersistentLatentStateV14,
    PersistentLatentStateV14Chain,
    W62_DEFAULT_V14_DISTRACTOR_RANK,
    W62_DEFAULT_V14_STATE_DIM,
    step_persistent_state_v14,
)
from .tiny_substrate_v3 import _sha256_hex


W63_PERSISTENT_V15_SCHEMA_VERSION: str = (
    "coordpy.persistent_latent_v15.v1")
W63_DEFAULT_V15_STATE_DIM: int = W62_DEFAULT_V14_STATE_DIM
W63_DEFAULT_V15_N_LAYERS: int = 14
W63_DEFAULT_V15_MAX_CHAIN_WALK_DEPTH: int = 4096
W63_DEFAULT_V15_DISTRACTOR_RANK: int = 10
W63_V15_NO_PARENT_STATE: str = "no_parent_v15_state"


@dataclasses.dataclass(frozen=True)
class PersistentLatentStateV15:
    inner_v14: PersistentLatentStateV14
    hidden_wins_carrier: tuple[float, ...]
    prefix_reuse_carrier: tuple[float, ...]
    distractor_rank: int

    @property
    def turn_index(self) -> int:
        return int(self.inner_v14.turn_index)

    @property
    def role(self) -> str:
        return str(self.inner_v14.role)

    @property
    def state_dim(self) -> int:
        return int(self.inner_v14.state_dim)

    @property
    def n_layers(self) -> int:
        return int(self.inner_v14.n_layers) + 2

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W63_PERSISTENT_V15_SCHEMA_VERSION,
            "inner_v14_cid": str(self.inner_v14.cid()),
            "hidden_wins_carrier": [
                float(round(float(x), 12))
                for x in self.hidden_wins_carrier],
            "prefix_reuse_carrier": [
                float(round(float(x), 12))
                for x in self.prefix_reuse_carrier],
            "distractor_rank": int(self.distractor_rank),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w63_v15_persistent_state",
            "state": self.to_dict()})


@dataclasses.dataclass
class PersistentLatentStateV15Chain:
    states: dict[str, PersistentLatentStateV15] = (
        dataclasses.field(default_factory=dict))

    @classmethod
    def empty(cls) -> "PersistentLatentStateV15Chain":
        return cls(states={})

    def add(self, s: PersistentLatentStateV15) -> None:
        self.states[s.cid()] = s

    def get(
            self, cid: str,
    ) -> PersistentLatentStateV15 | None:
        return self.states.get(str(cid))

    def walk_from(
            self, leaf_cid: str, *,
            max_depth: int = W63_DEFAULT_V15_MAX_CHAIN_WALK_DEPTH,
    ) -> list[PersistentLatentStateV15]:
        out: list[PersistentLatentStateV15] = []
        cur = self.get(leaf_cid)
        seen: set[str] = set()
        steps = 0
        while cur is not None and steps < int(max_depth):
            out.append(cur)
            seen.add(cur.cid())
            inner_parent_cid = (
                cur.inner_v14.inner_v13.inner_v12.parent_state_cid)
            parent_inner = None
            for c in self.states.values():
                if c.inner_v14.inner_v13.cid() == inner_parent_cid:
                    parent_inner = c
                    break
            if parent_inner is None or parent_inner.cid() in seen:
                break
            cur = parent_inner
            steps += 1
        return out

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w63_v15_state_chain",
            "members": [
                {"cid": c, "state": s.to_dict()}
                for c, s in sorted(self.states.items())],
        })


def step_persistent_state_v15(
        *, cell: V12StackedCell,
        prev_state: PersistentLatentStateV15 | None,
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
        substrate_fidelity: float = 1.0,
        attention_fidelity: float = 1.0,
        retrieval_fidelity: float = 1.0,
        replay_fidelity: float = 1.0,
        hidden_wins_ema_alpha: float = 0.10,
        prefix_reuse_ema_alpha: float = 0.10,
) -> PersistentLatentStateV15:
    """V15 step: V14 step + hidden-wins EMA + prefix-reuse EMA."""
    prev_v14 = (
        prev_state.inner_v14 if prev_state is not None else None)
    new_v14 = step_persistent_state_v14(
        cell=cell, prev_state=prev_v14,
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
        substrate_fidelity=float(substrate_fidelity),
        attention_fidelity=float(attention_fidelity),
        retrieval_fidelity=float(retrieval_fidelity),
        replay_fidelity=float(replay_fidelity))
    sd = int(new_v14.state_dim)
    if prev_state is not None:
        prev_hw = list(prev_state.hidden_wins_carrier)
        prev_pr = list(prev_state.prefix_reuse_carrier)
    else:
        prev_hw = [0.0] * sd
        prev_pr = [0.0] * sd
    if hidden_wins_skip is not None:
        hws = list(hidden_wins_skip)[:sd]
        while len(hws) < sd:
            hws.append(0.0)
        a = float(max(0.0, min(
            1.0, float(hidden_wins_ema_alpha))))
        new_hw = [
            a * float(hws[i]) + (1.0 - a) * float(
                prev_hw[i] if i < len(prev_hw) else 0.0)
            for i in range(sd)]
    else:
        new_hw = list(prev_hw)
        while len(new_hw) < sd:
            new_hw.append(0.0)
    if prefix_reuse_skip is not None:
        prs = list(prefix_reuse_skip)[:sd]
        while len(prs) < sd:
            prs.append(0.0)
        a = float(max(0.0, min(
            1.0, float(prefix_reuse_ema_alpha))))
        new_pr = [
            a * float(prs[i]) + (1.0 - a) * float(
                prev_pr[i] if i < len(prev_pr) else 0.0)
            for i in range(sd)]
    else:
        new_pr = list(prev_pr)
        while len(new_pr) < sd:
            new_pr.append(0.0)
    return PersistentLatentStateV15(
        inner_v14=new_v14,
        hidden_wins_carrier=tuple(new_hw),
        prefix_reuse_carrier=tuple(new_pr),
        distractor_rank=int(W63_DEFAULT_V15_DISTRACTOR_RANK),
    )


@dataclasses.dataclass(frozen=True)
class PersistentLatentStateV15Witness:
    schema: str
    chain_cid: str
    n_states: int
    chain_walk_depth_used: int
    n_layers: int
    distractor_rank: int
    twelfth_skip_present: bool
    hidden_wins_carrier_l1_sum: float
    prefix_reuse_carrier_l1_sum: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "chain_cid": str(self.chain_cid),
            "n_states": int(self.n_states),
            "chain_walk_depth_used": int(
                self.chain_walk_depth_used),
            "n_layers": int(self.n_layers),
            "distractor_rank": int(self.distractor_rank),
            "twelfth_skip_present": bool(
                self.twelfth_skip_present),
            "hidden_wins_carrier_l1_sum": float(round(
                self.hidden_wins_carrier_l1_sum, 12)),
            "prefix_reuse_carrier_l1_sum": float(round(
                self.prefix_reuse_carrier_l1_sum, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w63_persistent_v15_witness",
            "witness": self.to_dict()})


def emit_persistent_v15_witness(
        chain: PersistentLatentStateV15Chain,
        last_leaf_cid: str,
        *, max_depth: int = W63_DEFAULT_V15_MAX_CHAIN_WALK_DEPTH,
) -> PersistentLatentStateV15Witness:
    walk = chain.walk_from(
        last_leaf_cid, max_depth=int(max_depth))
    hw_sum = float(sum(
        abs(float(v))
        for s in walk
        for v in s.hidden_wins_carrier))
    pr_sum = float(sum(
        abs(float(v))
        for s in walk
        for v in s.prefix_reuse_carrier))
    return PersistentLatentStateV15Witness(
        schema=W63_PERSISTENT_V15_SCHEMA_VERSION,
        chain_cid=str(chain.cid()),
        n_states=int(len(chain.states)),
        chain_walk_depth_used=int(len(walk)),
        n_layers=int(W63_DEFAULT_V15_N_LAYERS),
        distractor_rank=int(W63_DEFAULT_V15_DISTRACTOR_RANK),
        twelfth_skip_present=bool(hw_sum > 0.0 or pr_sum > 0.0),
        hidden_wins_carrier_l1_sum=float(hw_sum),
        prefix_reuse_carrier_l1_sum=float(pr_sum),
    )


__all__ = [
    "W63_PERSISTENT_V15_SCHEMA_VERSION",
    "W63_DEFAULT_V15_STATE_DIM",
    "W63_DEFAULT_V15_N_LAYERS",
    "W63_DEFAULT_V15_MAX_CHAIN_WALK_DEPTH",
    "W63_DEFAULT_V15_DISTRACTOR_RANK",
    "W63_V15_NO_PARENT_STATE",
    "PersistentLatentStateV15",
    "PersistentLatentStateV15Chain",
    "PersistentLatentStateV15Witness",
    "step_persistent_state_v15",
    "emit_persistent_v15_witness",
]
