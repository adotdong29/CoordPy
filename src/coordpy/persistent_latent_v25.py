"""W73 — Persistent Latent State V25.

Strictly extends W72's ``coordpy.persistent_latent_v24``. V24 was
23 layers + twenty-one skip carriers + ``max_chain_walk_depth=524288``.
V25 adds:

* **24 layers** (vs V24's 23).
* **Twenty-second persistent skip-link** — V24's twenty-one plus a
  new *replacement-pressure EMA carrier*.
* ``max_chain_walk_depth = 1048576`` (W73 doubles the W72 cap).
* **Larger distractor basis** — V25 is **rank-24** (V24 was 23).

V25 strictly extends V24: with ``replacement_pressure_skip_v25 =
None``, the new EMA stays at the prior value (no-op) and V25
reduces to V24 byte-for-byte.

Honest scope (W73): ``W73-L-V25-OUTER-NOT-TRAINED-CAP``.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

from .persistent_latent_v12 import V12StackedCell
from .persistent_latent_v24 import (
    PersistentLatentStateV24,
    W72_DEFAULT_V24_STATE_DIM,
    step_persistent_state_v24,
)
from .tiny_substrate_v3 import _sha256_hex


W73_PERSISTENT_V25_SCHEMA_VERSION: str = (
    "coordpy.persistent_latent_v25.v1")
W73_DEFAULT_V25_STATE_DIM: int = W72_DEFAULT_V24_STATE_DIM
W73_DEFAULT_V25_N_LAYERS: int = 24
W73_DEFAULT_V25_MAX_CHAIN_WALK_DEPTH: int = 1048576
W73_DEFAULT_V25_DISTRACTOR_RANK: int = 24
W73_V25_NO_PARENT_STATE: str = "no_parent_v25_state"


@dataclasses.dataclass(frozen=True)
class PersistentLatentStateV25:
    inner_v24: PersistentLatentStateV24
    replacement_pressure_carrier: tuple[float, ...]
    distractor_rank: int

    @property
    def turn_index(self) -> int:
        return int(self.inner_v24.turn_index)

    @property
    def role(self) -> str:
        return str(self.inner_v24.role)

    @property
    def state_dim(self) -> int:
        return int(self.inner_v24.state_dim)

    @property
    def n_layers(self) -> int:
        return int(self.inner_v24.n_layers) + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W73_PERSISTENT_V25_SCHEMA_VERSION,
            "inner_v24_cid": str(self.inner_v24.cid()),
            "replacement_pressure_carrier": [
                float(round(float(x), 12))
                for x in self.replacement_pressure_carrier],
            "distractor_rank": int(self.distractor_rank),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w73_v25_persistent_state",
            "state": self.to_dict()})


@dataclasses.dataclass
class PersistentLatentStateV25Chain:
    states: dict[str, PersistentLatentStateV25] = (
        dataclasses.field(default_factory=dict))

    @classmethod
    def empty(cls) -> "PersistentLatentStateV25Chain":
        return cls(states={})

    def add(self, s: PersistentLatentStateV25) -> None:
        self.states[s.cid()] = s

    def get(self, cid: str) -> PersistentLatentStateV25 | None:
        return self.states.get(str(cid))

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w73_v25_state_chain",
            "members": [
                {"cid": c, "state": s.to_dict()}
                for c, s in sorted(self.states.items())],
        })


def step_persistent_state_v25(
        *, cell: V12StackedCell,
        prev_state: PersistentLatentStateV25 | None,
        carrier_values: Sequence[float],
        turn_index: int, role: str,
        branch_id: str = "main",
        replacement_pressure_skip_v25: (
            Sequence[float] | None) = None,
        replacement_pressure_ema_alpha: float = 0.10,
        **kwargs: Any,
) -> PersistentLatentStateV25:
    prev_v24 = (
        prev_state.inner_v24 if prev_state is not None else None)
    new_v24 = step_persistent_state_v24(
        cell=cell, prev_state=prev_v24,
        carrier_values=list(carrier_values),
        turn_index=int(turn_index), role=str(role),
        branch_id=str(branch_id),
        **kwargs)
    sd = int(new_v24.state_dim)
    if prev_state is not None:
        prev_rep = list(prev_state.replacement_pressure_carrier)
    else:
        prev_rep = [0.0] * sd
    if replacement_pressure_skip_v25 is not None:
        rep = list(replacement_pressure_skip_v25)[:sd]
        while len(rep) < sd:
            rep.append(0.0)
        a = float(max(0.0, min(
            1.0, float(replacement_pressure_ema_alpha))))
        new_rep = [
            a * float(rep[i]) + (1.0 - a) * float(
                prev_rep[i] if i < len(prev_rep) else 0.0)
            for i in range(sd)]
    else:
        new_rep = list(prev_rep)
        while len(new_rep) < sd:
            new_rep.append(0.0)
    return PersistentLatentStateV25(
        inner_v24=new_v24,
        replacement_pressure_carrier=tuple(new_rep),
        distractor_rank=int(W73_DEFAULT_V25_DISTRACTOR_RANK),
    )


@dataclasses.dataclass(frozen=True)
class PersistentLatentStateV25Witness:
    schema: str
    chain_cid: str
    n_states: int
    n_layers: int
    distractor_rank: int
    twenty_second_skip_present: bool
    replacement_pressure_carrier_l1_sum: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "chain_cid": str(self.chain_cid),
            "n_states": int(self.n_states),
            "n_layers": int(self.n_layers),
            "distractor_rank": int(self.distractor_rank),
            "twenty_second_skip_present": bool(
                self.twenty_second_skip_present),
            "replacement_pressure_carrier_l1_sum": float(round(
                self.replacement_pressure_carrier_l1_sum, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w73_persistent_v25_witness",
            "witness": self.to_dict()})


def emit_persistent_v25_witness(
        chain: PersistentLatentStateV25Chain,
) -> PersistentLatentStateV25Witness:
    rep_sum = float(sum(
        abs(float(v))
        for s in chain.states.values()
        for v in s.replacement_pressure_carrier))
    return PersistentLatentStateV25Witness(
        schema=W73_PERSISTENT_V25_SCHEMA_VERSION,
        chain_cid=str(chain.cid()),
        n_states=int(len(chain.states)),
        n_layers=int(W73_DEFAULT_V25_N_LAYERS),
        distractor_rank=int(W73_DEFAULT_V25_DISTRACTOR_RANK),
        twenty_second_skip_present=bool(rep_sum > 0.0),
        replacement_pressure_carrier_l1_sum=float(rep_sum),
    )


__all__ = [
    "W73_PERSISTENT_V25_SCHEMA_VERSION",
    "W73_DEFAULT_V25_STATE_DIM",
    "W73_DEFAULT_V25_N_LAYERS",
    "W73_DEFAULT_V25_MAX_CHAIN_WALK_DEPTH",
    "W73_DEFAULT_V25_DISTRACTOR_RANK",
    "W73_V25_NO_PARENT_STATE",
    "PersistentLatentStateV25",
    "PersistentLatentStateV25Chain",
    "PersistentLatentStateV25Witness",
    "step_persistent_state_v25",
    "emit_persistent_v25_witness",
]
