"""W78 — Persistent Latent State V30.

Strictly extends W77's ``coordpy.persistent_latent_v29``. V29 was
28 layers + twenty-six skip carriers + ``max_chain_walk_depth=
16777216``. V30 adds:

* **29 layers** (vs V29's 28).
* **Twenty-seventh persistent skip-link** — V29's twenty-six plus
  a new *long-horizon-reconstruction-pressure EMA carrier*. This
  is the persistent latent the W78 long-horizon-reconstruction
  substrate reads from.
* ``max_chain_walk_depth = 33554432`` (W78 doubles the W77 cap;
  the chain walk depth doubles every milestone since W75).
* **Larger distractor basis** — V30 is **rank-29** (V29 was 28).

V30 strictly extends V29: with
``long_horizon_reconstruction_pressure_skip_v30 = None`` the new
EMA stays at the prior value and V30 reduces to V29 byte-for-byte.

Honest scope (W78): ``W78-L-V30-OUTER-NOT-TRAINED-CAP``.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

from .persistent_latent_v12 import V12StackedCell
from .persistent_latent_v29 import (
    PersistentLatentStateV29, W77_DEFAULT_V29_STATE_DIM,
    step_persistent_state_v29,
)
from .tiny_substrate_v3 import _sha256_hex


W78_PERSISTENT_V30_SCHEMA_VERSION: str = (
    "coordpy.persistent_latent_v30.v1")
W78_DEFAULT_V30_STATE_DIM: int = W77_DEFAULT_V29_STATE_DIM
W78_DEFAULT_V30_N_LAYERS: int = 29
W78_DEFAULT_V30_MAX_CHAIN_WALK_DEPTH: int = 33554432
W78_DEFAULT_V30_DISTRACTOR_RANK: int = 29
W78_V30_NO_PARENT_STATE: str = "no_parent_v30_state"


@dataclasses.dataclass(frozen=True)
class PersistentLatentStateV30:
    inner_v29: PersistentLatentStateV29
    long_horizon_reconstruction_pressure_carrier: tuple[
        float, ...]
    distractor_rank: int

    @property
    def turn_index(self) -> int:
        return int(self.inner_v29.turn_index)

    @property
    def role(self) -> str:
        return str(self.inner_v29.role)

    @property
    def state_dim(self) -> int:
        return int(self.inner_v29.state_dim)

    @property
    def n_layers(self) -> int:
        return int(self.inner_v29.n_layers) + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W78_PERSISTENT_V30_SCHEMA_VERSION,
            "inner_v29_cid": str(self.inner_v29.cid()),
            "long_horizon_reconstruction_pressure_carrier": [
                float(round(float(x), 12))
                for x in
                self
                .long_horizon_reconstruction_pressure_carrier],
            "distractor_rank": int(self.distractor_rank),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w78_v30_persistent_state",
            "state": self.to_dict()})


@dataclasses.dataclass
class PersistentLatentStateV30Chain:
    states: dict[str, PersistentLatentStateV30] = (
        dataclasses.field(default_factory=dict))

    @classmethod
    def empty(cls) -> "PersistentLatentStateV30Chain":
        return cls(states={})

    def add(self, s: PersistentLatentStateV30) -> None:
        self.states[s.cid()] = s

    def get(self, cid: str) -> PersistentLatentStateV30 | None:
        return self.states.get(str(cid))

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w78_v30_state_chain",
            "members": [
                {"cid": c, "state": s.to_dict()}
                for c, s in sorted(self.states.items())],
        })


def step_persistent_state_v30(
        *, cell: V12StackedCell,
        prev_state: PersistentLatentStateV30 | None,
        carrier_values: Sequence[float],
        turn_index: int, role: str,
        branch_id: str = "main",
        long_horizon_reconstruction_pressure_skip_v30: (
            Sequence[float] | None) = None,
        long_horizon_reconstruction_pressure_ema_alpha: float = (
            0.12),
        **kwargs: Any,
) -> PersistentLatentStateV30:
    prev_v29 = (
        prev_state.inner_v29 if prev_state is not None else None)
    new_v29 = step_persistent_state_v29(
        cell=cell, prev_state=prev_v29,
        carrier_values=list(carrier_values),
        turn_index=int(turn_index), role=str(role),
        branch_id=str(branch_id),
        **kwargs)
    sd = int(new_v29.state_dim)
    if prev_state is not None:
        prev_chain = list(
            prev_state
            .long_horizon_reconstruction_pressure_carrier)
    else:
        prev_chain = [0.0] * sd
    if long_horizon_reconstruction_pressure_skip_v30 is not None:
        chain = list(
            long_horizon_reconstruction_pressure_skip_v30)[:sd]
        while len(chain) < sd:
            chain.append(0.0)
        a = float(max(0.0, min(
            1.0,
            float(
                long_horizon_reconstruction_pressure_ema_alpha))))
        new_chain = [
            a * float(chain[i]) + (1.0 - a) * float(
                prev_chain[i] if i < len(prev_chain) else 0.0)
            for i in range(sd)]
    else:
        new_chain = list(prev_chain)
        while len(new_chain) < sd:
            new_chain.append(0.0)
    return PersistentLatentStateV30(
        inner_v29=new_v29,
        long_horizon_reconstruction_pressure_carrier=tuple(
            new_chain),
        distractor_rank=int(W78_DEFAULT_V30_DISTRACTOR_RANK),
    )


@dataclasses.dataclass(frozen=True)
class PersistentLatentStateV30Witness:
    schema: str
    chain_cid: str
    n_states: int
    n_layers: int
    distractor_rank: int
    twenty_seventh_skip_present: bool
    long_horizon_reconstruction_pressure_carrier_l1_sum: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "chain_cid": str(self.chain_cid),
            "n_states": int(self.n_states),
            "n_layers": int(self.n_layers),
            "distractor_rank": int(self.distractor_rank),
            "twenty_seventh_skip_present": bool(
                self.twenty_seventh_skip_present),
            "long_horizon_reconstruction_pressure_carrier_l1_sum":
                float(round(
                    self
                    .long_horizon_reconstruction_pressure_carrier_l1_sum,
                    12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w78_persistent_v30_witness",
            "witness": self.to_dict()})


def emit_persistent_v30_witness(
        chain: PersistentLatentStateV30Chain,
) -> PersistentLatentStateV30Witness:
    chain_sum = float(sum(
        abs(float(v))
        for s in chain.states.values()
        for v in s.long_horizon_reconstruction_pressure_carrier))
    return PersistentLatentStateV30Witness(
        schema=W78_PERSISTENT_V30_SCHEMA_VERSION,
        chain_cid=str(chain.cid()),
        n_states=int(len(chain.states)),
        n_layers=int(W78_DEFAULT_V30_N_LAYERS),
        distractor_rank=int(W78_DEFAULT_V30_DISTRACTOR_RANK),
        twenty_seventh_skip_present=bool(chain_sum > 0.0),
        long_horizon_reconstruction_pressure_carrier_l1_sum=float(
            chain_sum),
    )


__all__ = [
    "W78_PERSISTENT_V30_SCHEMA_VERSION",
    "W78_DEFAULT_V30_STATE_DIM",
    "W78_DEFAULT_V30_N_LAYERS",
    "W78_DEFAULT_V30_MAX_CHAIN_WALK_DEPTH",
    "W78_DEFAULT_V30_DISTRACTOR_RANK",
    "W78_V30_NO_PARENT_STATE",
    "PersistentLatentStateV30",
    "PersistentLatentStateV30Chain",
    "PersistentLatentStateV30Witness",
    "step_persistent_state_v30",
    "emit_persistent_v30_witness",
]
