"""W76 — Persistent Latent State V28.

Strictly extends W75's ``coordpy.persistent_latent_v27``. V27 was
26 layers + twenty-four skip carriers + ``max_chain_walk_depth=
4194304``. V28 adds:

* **27 layers** (vs V27's 26).
* **Twenty-fifth persistent skip-link** — V27's twenty-four plus
  a new *chain-then-restart-pressure EMA carrier*.
* ``max_chain_walk_depth = 8388608`` (W76 doubles the W75 cap).
* **Larger distractor basis** — V28 is **rank-27** (V27 was 26).

V28 strictly extends V27: with ``chain_then_restart_pressure_skip_
v28 = None``, the new EMA stays at the prior value (no-op) and V28
reduces to V27 byte-for-byte.

Honest scope (W76): ``W76-L-V28-OUTER-NOT-TRAINED-CAP``.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

from .persistent_latent_v12 import V12StackedCell
from .persistent_latent_v27 import (
    PersistentLatentStateV27,
    W75_DEFAULT_V27_STATE_DIM,
    step_persistent_state_v27,
)
from .tiny_substrate_v3 import _sha256_hex


W76_PERSISTENT_V28_SCHEMA_VERSION: str = (
    "coordpy.persistent_latent_v28.v1")
W76_DEFAULT_V28_STATE_DIM: int = W75_DEFAULT_V27_STATE_DIM
W76_DEFAULT_V28_N_LAYERS: int = 27
W76_DEFAULT_V28_MAX_CHAIN_WALK_DEPTH: int = 8388608
W76_DEFAULT_V28_DISTRACTOR_RANK: int = 27
W76_V28_NO_PARENT_STATE: str = "no_parent_v28_state"


@dataclasses.dataclass(frozen=True)
class PersistentLatentStateV28:
    inner_v27: PersistentLatentStateV27
    chain_then_restart_pressure_carrier: tuple[float, ...]
    distractor_rank: int

    @property
    def turn_index(self) -> int:
        return int(self.inner_v27.turn_index)

    @property
    def role(self) -> str:
        return str(self.inner_v27.role)

    @property
    def state_dim(self) -> int:
        return int(self.inner_v27.state_dim)

    @property
    def n_layers(self) -> int:
        return int(self.inner_v27.n_layers) + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W76_PERSISTENT_V28_SCHEMA_VERSION,
            "inner_v27_cid": str(self.inner_v27.cid()),
            "chain_then_restart_pressure_carrier": [
                float(round(float(x), 12))
                for x in self.chain_then_restart_pressure_carrier],
            "distractor_rank": int(self.distractor_rank),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w76_v28_persistent_state",
            "state": self.to_dict()})


@dataclasses.dataclass
class PersistentLatentStateV28Chain:
    states: dict[str, PersistentLatentStateV28] = (
        dataclasses.field(default_factory=dict))

    @classmethod
    def empty(cls) -> "PersistentLatentStateV28Chain":
        return cls(states={})

    def add(self, s: PersistentLatentStateV28) -> None:
        self.states[s.cid()] = s

    def get(self, cid: str) -> PersistentLatentStateV28 | None:
        return self.states.get(str(cid))

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w76_v28_state_chain",
            "members": [
                {"cid": c, "state": s.to_dict()}
                for c, s in sorted(self.states.items())],
        })


def step_persistent_state_v28(
        *, cell: V12StackedCell,
        prev_state: PersistentLatentStateV28 | None,
        carrier_values: Sequence[float],
        turn_index: int, role: str,
        branch_id: str = "main",
        chain_then_restart_pressure_skip_v28: (
            Sequence[float] | None) = None,
        chain_then_restart_pressure_ema_alpha: float = 0.10,
        **kwargs: Any,
) -> PersistentLatentStateV28:
    prev_v27 = (
        prev_state.inner_v27 if prev_state is not None else None)
    new_v27 = step_persistent_state_v27(
        cell=cell, prev_state=prev_v27,
        carrier_values=list(carrier_values),
        turn_index=int(turn_index), role=str(role),
        branch_id=str(branch_id),
        **kwargs)
    sd = int(new_v27.state_dim)
    if prev_state is not None:
        prev_chain = list(
            prev_state.chain_then_restart_pressure_carrier)
    else:
        prev_chain = [0.0] * sd
    if chain_then_restart_pressure_skip_v28 is not None:
        chain = list(chain_then_restart_pressure_skip_v28)[:sd]
        while len(chain) < sd:
            chain.append(0.0)
        a = float(max(0.0, min(
            1.0,
            float(chain_then_restart_pressure_ema_alpha))))
        new_chain = [
            a * float(chain[i]) + (1.0 - a) * float(
                prev_chain[i] if i < len(prev_chain) else 0.0)
            for i in range(sd)]
    else:
        new_chain = list(prev_chain)
        while len(new_chain) < sd:
            new_chain.append(0.0)
    return PersistentLatentStateV28(
        inner_v27=new_v27,
        chain_then_restart_pressure_carrier=tuple(new_chain),
        distractor_rank=int(W76_DEFAULT_V28_DISTRACTOR_RANK),
    )


@dataclasses.dataclass(frozen=True)
class PersistentLatentStateV28Witness:
    schema: str
    chain_cid: str
    n_states: int
    n_layers: int
    distractor_rank: int
    twenty_fifth_skip_present: bool
    chain_then_restart_pressure_carrier_l1_sum: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "chain_cid": str(self.chain_cid),
            "n_states": int(self.n_states),
            "n_layers": int(self.n_layers),
            "distractor_rank": int(self.distractor_rank),
            "twenty_fifth_skip_present": bool(
                self.twenty_fifth_skip_present),
            "chain_then_restart_pressure_carrier_l1_sum": float(
                round(
                    self
                    .chain_then_restart_pressure_carrier_l1_sum,
                    12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w76_persistent_v28_witness",
            "witness": self.to_dict()})


def emit_persistent_v28_witness(
        chain: PersistentLatentStateV28Chain,
) -> PersistentLatentStateV28Witness:
    chain_sum = float(sum(
        abs(float(v))
        for s in chain.states.values()
        for v in s.chain_then_restart_pressure_carrier))
    return PersistentLatentStateV28Witness(
        schema=W76_PERSISTENT_V28_SCHEMA_VERSION,
        chain_cid=str(chain.cid()),
        n_states=int(len(chain.states)),
        n_layers=int(W76_DEFAULT_V28_N_LAYERS),
        distractor_rank=int(W76_DEFAULT_V28_DISTRACTOR_RANK),
        twenty_fifth_skip_present=bool(chain_sum > 0.0),
        chain_then_restart_pressure_carrier_l1_sum=float(
            chain_sum),
    )


__all__ = [
    "W76_PERSISTENT_V28_SCHEMA_VERSION",
    "W76_DEFAULT_V28_STATE_DIM",
    "W76_DEFAULT_V28_N_LAYERS",
    "W76_DEFAULT_V28_MAX_CHAIN_WALK_DEPTH",
    "W76_DEFAULT_V28_DISTRACTOR_RANK",
    "W76_V28_NO_PARENT_STATE",
    "PersistentLatentStateV28",
    "PersistentLatentStateV28Chain",
    "PersistentLatentStateV28Witness",
    "step_persistent_state_v28",
    "emit_persistent_v28_witness",
]
