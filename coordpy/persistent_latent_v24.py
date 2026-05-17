"""W72 — Persistent Latent State V24.

Strictly extends W71's ``coordpy.persistent_latent_v23``. V23 was
22 layers + twenty skip carriers + ``max_chain_walk_depth=262144``.
V24 adds:

* **23 layers** (vs V23's 22).
* **Twenty-first persistent skip-link** — V23's twenty plus a new
  *rejoin-pressure EMA carrier*.
* ``max_chain_walk_depth = 524288`` (W72 doubles the W71 cap).
* **Larger distractor basis** — V24 is **rank-23** (V23 was 22).

V24 strictly extends V23: with ``rejoin_pressure_skip_v24 = None``,
the new EMA stays at the prior value (no-op) and V24 reduces to
V23 byte-for-byte.

Honest scope (W72): ``W72-L-V24-OUTER-NOT-TRAINED-CAP``.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

from .persistent_latent_v12 import V12StackedCell
from .persistent_latent_v23 import (
    PersistentLatentStateV23,
    W71_DEFAULT_V23_STATE_DIM,
    step_persistent_state_v23,
)
from .tiny_substrate_v3 import _sha256_hex


W72_PERSISTENT_V24_SCHEMA_VERSION: str = (
    "coordpy.persistent_latent_v24.v1")
W72_DEFAULT_V24_STATE_DIM: int = W71_DEFAULT_V23_STATE_DIM
W72_DEFAULT_V24_N_LAYERS: int = 23
W72_DEFAULT_V24_MAX_CHAIN_WALK_DEPTH: int = 524288
W72_DEFAULT_V24_DISTRACTOR_RANK: int = 23
W72_V24_NO_PARENT_STATE: str = "no_parent_v24_state"


@dataclasses.dataclass(frozen=True)
class PersistentLatentStateV24:
    inner_v23: PersistentLatentStateV23
    rejoin_pressure_carrier: tuple[float, ...]
    distractor_rank: int

    @property
    def turn_index(self) -> int:
        return int(self.inner_v23.turn_index)

    @property
    def role(self) -> str:
        return str(self.inner_v23.role)

    @property
    def state_dim(self) -> int:
        return int(self.inner_v23.state_dim)

    @property
    def n_layers(self) -> int:
        return int(self.inner_v23.n_layers) + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W72_PERSISTENT_V24_SCHEMA_VERSION,
            "inner_v23_cid": str(self.inner_v23.cid()),
            "rejoin_pressure_carrier": [
                float(round(float(x), 12))
                for x in self.rejoin_pressure_carrier],
            "distractor_rank": int(self.distractor_rank),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w72_v24_persistent_state",
            "state": self.to_dict()})


@dataclasses.dataclass
class PersistentLatentStateV24Chain:
    states: dict[str, PersistentLatentStateV24] = (
        dataclasses.field(default_factory=dict))

    @classmethod
    def empty(cls) -> "PersistentLatentStateV24Chain":
        return cls(states={})

    def add(self, s: PersistentLatentStateV24) -> None:
        self.states[s.cid()] = s

    def get(self, cid: str) -> PersistentLatentStateV24 | None:
        return self.states.get(str(cid))

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w72_v24_state_chain",
            "members": [
                {"cid": c, "state": s.to_dict()}
                for c, s in sorted(self.states.items())],
        })


def step_persistent_state_v24(
        *, cell: V12StackedCell,
        prev_state: PersistentLatentStateV24 | None,
        carrier_values: Sequence[float],
        turn_index: int, role: str,
        branch_id: str = "main",
        rejoin_pressure_skip_v24: (
            Sequence[float] | None) = None,
        rejoin_pressure_ema_alpha: float = 0.10,
        **kwargs: Any,
) -> PersistentLatentStateV24:
    prev_v23 = (
        prev_state.inner_v23 if prev_state is not None else None)
    new_v23 = step_persistent_state_v23(
        cell=cell, prev_state=prev_v23,
        carrier_values=list(carrier_values),
        turn_index=int(turn_index), role=str(role),
        branch_id=str(branch_id),
        **kwargs)
    sd = int(new_v23.state_dim)
    if prev_state is not None:
        prev_rj = list(prev_state.rejoin_pressure_carrier)
    else:
        prev_rj = [0.0] * sd
    if rejoin_pressure_skip_v24 is not None:
        rj = list(rejoin_pressure_skip_v24)[:sd]
        while len(rj) < sd:
            rj.append(0.0)
        a = float(max(0.0, min(
            1.0, float(rejoin_pressure_ema_alpha))))
        new_rj = [
            a * float(rj[i]) + (1.0 - a) * float(
                prev_rj[i] if i < len(prev_rj) else 0.0)
            for i in range(sd)]
    else:
        new_rj = list(prev_rj)
        while len(new_rj) < sd:
            new_rj.append(0.0)
    return PersistentLatentStateV24(
        inner_v23=new_v23,
        rejoin_pressure_carrier=tuple(new_rj),
        distractor_rank=int(W72_DEFAULT_V24_DISTRACTOR_RANK),
    )


@dataclasses.dataclass(frozen=True)
class PersistentLatentStateV24Witness:
    schema: str
    chain_cid: str
    n_states: int
    n_layers: int
    distractor_rank: int
    twenty_first_skip_present: bool
    rejoin_pressure_carrier_l1_sum: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "chain_cid": str(self.chain_cid),
            "n_states": int(self.n_states),
            "n_layers": int(self.n_layers),
            "distractor_rank": int(self.distractor_rank),
            "twenty_first_skip_present": bool(
                self.twenty_first_skip_present),
            "rejoin_pressure_carrier_l1_sum": float(round(
                self.rejoin_pressure_carrier_l1_sum, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w72_persistent_v24_witness",
            "witness": self.to_dict()})


def emit_persistent_v24_witness(
        chain: PersistentLatentStateV24Chain,
) -> PersistentLatentStateV24Witness:
    rj_sum = float(sum(
        abs(float(v))
        for s in chain.states.values()
        for v in s.rejoin_pressure_carrier))
    return PersistentLatentStateV24Witness(
        schema=W72_PERSISTENT_V24_SCHEMA_VERSION,
        chain_cid=str(chain.cid()),
        n_states=int(len(chain.states)),
        n_layers=int(W72_DEFAULT_V24_N_LAYERS),
        distractor_rank=int(W72_DEFAULT_V24_DISTRACTOR_RANK),
        twenty_first_skip_present=bool(rj_sum > 0.0),
        rejoin_pressure_carrier_l1_sum=float(rj_sum),
    )


__all__ = [
    "W72_PERSISTENT_V24_SCHEMA_VERSION",
    "W72_DEFAULT_V24_STATE_DIM",
    "W72_DEFAULT_V24_N_LAYERS",
    "W72_DEFAULT_V24_MAX_CHAIN_WALK_DEPTH",
    "W72_DEFAULT_V24_DISTRACTOR_RANK",
    "W72_V24_NO_PARENT_STATE",
    "PersistentLatentStateV24",
    "PersistentLatentStateV24Chain",
    "PersistentLatentStateV24Witness",
    "step_persistent_state_v24",
    "emit_persistent_v24_witness",
]
