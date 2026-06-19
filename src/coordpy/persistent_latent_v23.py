"""W71 — Persistent Latent State V23.

Strictly extends W70's ``coordpy.persistent_latent_v22``. V22 was
21 layers + nineteen skip carriers + ``max_chain_walk_depth=131072``.
V23 adds:

* **22 layers** (vs V22's 21).
* **Twentieth persistent skip-link** — V22's nineteen plus a new
  *restart-dominance EMA carrier*.
* ``max_chain_walk_depth = 262144`` (W71 doubles the W70 cap).
* **Larger distractor basis** — V23 is **rank-22** (V22 was 21).

V23 strictly extends V22: with ``restart_dominance_skip_v23 = None``,
the new EMA stays at the prior value (no-op) and V23 reduces to
V22 byte-for-byte.

Honest scope (W71): ``W71-L-V23-OUTER-NOT-TRAINED-CAP``.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

from .persistent_latent_v12 import V12StackedCell
from .persistent_latent_v22 import (
    PersistentLatentStateV22,
    W70_DEFAULT_V22_STATE_DIM,
    step_persistent_state_v22,
)
from .tiny_substrate_v3 import _sha256_hex


W71_PERSISTENT_V23_SCHEMA_VERSION: str = (
    "coordpy.persistent_latent_v23.v1")
W71_DEFAULT_V23_STATE_DIM: int = W70_DEFAULT_V22_STATE_DIM
W71_DEFAULT_V23_N_LAYERS: int = 22
W71_DEFAULT_V23_MAX_CHAIN_WALK_DEPTH: int = 262144
W71_DEFAULT_V23_DISTRACTOR_RANK: int = 22
W71_V23_NO_PARENT_STATE: str = "no_parent_v23_state"


@dataclasses.dataclass(frozen=True)
class PersistentLatentStateV23:
    inner_v22: PersistentLatentStateV22
    restart_dominance_carrier: tuple[float, ...]
    distractor_rank: int

    @property
    def turn_index(self) -> int:
        return int(self.inner_v22.turn_index)

    @property
    def role(self) -> str:
        return str(self.inner_v22.role)

    @property
    def state_dim(self) -> int:
        return int(self.inner_v22.state_dim)

    @property
    def n_layers(self) -> int:
        return int(self.inner_v22.n_layers) + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W71_PERSISTENT_V23_SCHEMA_VERSION,
            "inner_v22_cid": str(self.inner_v22.cid()),
            "restart_dominance_carrier": [
                float(round(float(x), 12))
                for x in self.restart_dominance_carrier],
            "distractor_rank": int(self.distractor_rank),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w71_v23_persistent_state",
            "state": self.to_dict()})


@dataclasses.dataclass
class PersistentLatentStateV23Chain:
    states: dict[str, PersistentLatentStateV23] = (
        dataclasses.field(default_factory=dict))

    @classmethod
    def empty(cls) -> "PersistentLatentStateV23Chain":
        return cls(states={})

    def add(self, s: PersistentLatentStateV23) -> None:
        self.states[s.cid()] = s

    def get(self, cid: str) -> PersistentLatentStateV23 | None:
        return self.states.get(str(cid))

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w71_v23_state_chain",
            "members": [
                {"cid": c, "state": s.to_dict()}
                for c, s in sorted(self.states.items())],
        })


def step_persistent_state_v23(
        *, cell: V12StackedCell,
        prev_state: PersistentLatentStateV23 | None,
        carrier_values: Sequence[float],
        turn_index: int, role: str,
        branch_id: str = "main",
        restart_dominance_skip_v23: (
            Sequence[float] | None) = None,
        restart_dominance_ema_alpha: float = 0.10,
        **kwargs: Any,
) -> PersistentLatentStateV23:
    prev_v22 = (
        prev_state.inner_v22 if prev_state is not None else None)
    new_v22 = step_persistent_state_v22(
        cell=cell, prev_state=prev_v22,
        carrier_values=list(carrier_values),
        turn_index=int(turn_index), role=str(role),
        branch_id=str(branch_id),
        **kwargs)
    sd = int(new_v22.state_dim)
    if prev_state is not None:
        prev_rd = list(prev_state.restart_dominance_carrier)
    else:
        prev_rd = [0.0] * sd
    if restart_dominance_skip_v23 is not None:
        rd = list(restart_dominance_skip_v23)[:sd]
        while len(rd) < sd:
            rd.append(0.0)
        a = float(max(0.0, min(
            1.0, float(restart_dominance_ema_alpha))))
        new_rd = [
            a * float(rd[i]) + (1.0 - a) * float(
                prev_rd[i] if i < len(prev_rd) else 0.0)
            for i in range(sd)]
    else:
        new_rd = list(prev_rd)
        while len(new_rd) < sd:
            new_rd.append(0.0)
    return PersistentLatentStateV23(
        inner_v22=new_v22,
        restart_dominance_carrier=tuple(new_rd),
        distractor_rank=int(W71_DEFAULT_V23_DISTRACTOR_RANK),
    )


@dataclasses.dataclass(frozen=True)
class PersistentLatentStateV23Witness:
    schema: str
    chain_cid: str
    n_states: int
    n_layers: int
    distractor_rank: int
    twentieth_skip_present: bool
    restart_dominance_carrier_l1_sum: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "chain_cid": str(self.chain_cid),
            "n_states": int(self.n_states),
            "n_layers": int(self.n_layers),
            "distractor_rank": int(self.distractor_rank),
            "twentieth_skip_present": bool(
                self.twentieth_skip_present),
            "restart_dominance_carrier_l1_sum": float(round(
                self.restart_dominance_carrier_l1_sum, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w71_persistent_v23_witness",
            "witness": self.to_dict()})


def emit_persistent_v23_witness(
        chain: PersistentLatentStateV23Chain,
) -> PersistentLatentStateV23Witness:
    rd_sum = float(sum(
        abs(float(v))
        for s in chain.states.values()
        for v in s.restart_dominance_carrier))
    return PersistentLatentStateV23Witness(
        schema=W71_PERSISTENT_V23_SCHEMA_VERSION,
        chain_cid=str(chain.cid()),
        n_states=int(len(chain.states)),
        n_layers=int(W71_DEFAULT_V23_N_LAYERS),
        distractor_rank=int(W71_DEFAULT_V23_DISTRACTOR_RANK),
        twentieth_skip_present=bool(rd_sum > 0.0),
        restart_dominance_carrier_l1_sum=float(rd_sum),
    )


__all__ = [
    "W71_PERSISTENT_V23_SCHEMA_VERSION",
    "W71_DEFAULT_V23_STATE_DIM",
    "W71_DEFAULT_V23_N_LAYERS",
    "W71_DEFAULT_V23_MAX_CHAIN_WALK_DEPTH",
    "W71_DEFAULT_V23_DISTRACTOR_RANK",
    "W71_V23_NO_PARENT_STATE",
    "PersistentLatentStateV23",
    "PersistentLatentStateV23Chain",
    "PersistentLatentStateV23Witness",
    "step_persistent_state_v23",
    "emit_persistent_v23_witness",
]
