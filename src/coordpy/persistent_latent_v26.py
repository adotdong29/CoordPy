"""W74 — Persistent Latent State V26.

Strictly extends W73's ``coordpy.persistent_latent_v25``. V25 was
24 layers + twenty-two skip carriers + ``max_chain_walk_depth=1048576``.
V26 adds:

* **25 layers** (vs V25's 24).
* **Twenty-third persistent skip-link** — V25's twenty-two plus a
  new *compound-pressure EMA carrier*.
* ``max_chain_walk_depth = 2097152`` (W74 doubles the W73 cap).
* **Larger distractor basis** — V26 is **rank-25** (V25 was 24).

V26 strictly extends V25: with ``compound_pressure_skip_v26 =
None``, the new EMA stays at the prior value (no-op) and V26
reduces to V25 byte-for-byte.

Honest scope (W74): ``W74-L-V26-OUTER-NOT-TRAINED-CAP``.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

from .persistent_latent_v12 import V12StackedCell
from .persistent_latent_v25 import (
    PersistentLatentStateV25,
    W73_DEFAULT_V25_STATE_DIM,
    step_persistent_state_v25,
)
from .tiny_substrate_v3 import _sha256_hex


W74_PERSISTENT_V26_SCHEMA_VERSION: str = (
    "coordpy.persistent_latent_v26.v1")
W74_DEFAULT_V26_STATE_DIM: int = W73_DEFAULT_V25_STATE_DIM
W74_DEFAULT_V26_N_LAYERS: int = 25
W74_DEFAULT_V26_MAX_CHAIN_WALK_DEPTH: int = 2097152
W74_DEFAULT_V26_DISTRACTOR_RANK: int = 25
W74_V26_NO_PARENT_STATE: str = "no_parent_v26_state"


@dataclasses.dataclass(frozen=True)
class PersistentLatentStateV26:
    inner_v25: PersistentLatentStateV25
    compound_pressure_carrier: tuple[float, ...]
    distractor_rank: int

    @property
    def turn_index(self) -> int:
        return int(self.inner_v25.turn_index)

    @property
    def role(self) -> str:
        return str(self.inner_v25.role)

    @property
    def state_dim(self) -> int:
        return int(self.inner_v25.state_dim)

    @property
    def n_layers(self) -> int:
        return int(self.inner_v25.n_layers) + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W74_PERSISTENT_V26_SCHEMA_VERSION,
            "inner_v25_cid": str(self.inner_v25.cid()),
            "compound_pressure_carrier": [
                float(round(float(x), 12))
                for x in self.compound_pressure_carrier],
            "distractor_rank": int(self.distractor_rank),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w74_v26_persistent_state",
            "state": self.to_dict()})


@dataclasses.dataclass
class PersistentLatentStateV26Chain:
    states: dict[str, PersistentLatentStateV26] = (
        dataclasses.field(default_factory=dict))

    @classmethod
    def empty(cls) -> "PersistentLatentStateV26Chain":
        return cls(states={})

    def add(self, s: PersistentLatentStateV26) -> None:
        self.states[s.cid()] = s

    def get(self, cid: str) -> PersistentLatentStateV26 | None:
        return self.states.get(str(cid))

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w74_v26_state_chain",
            "members": [
                {"cid": c, "state": s.to_dict()}
                for c, s in sorted(self.states.items())],
        })


def step_persistent_state_v26(
        *, cell: V12StackedCell,
        prev_state: PersistentLatentStateV26 | None,
        carrier_values: Sequence[float],
        turn_index: int, role: str,
        branch_id: str = "main",
        compound_pressure_skip_v26: (
            Sequence[float] | None) = None,
        compound_pressure_ema_alpha: float = 0.10,
        **kwargs: Any,
) -> PersistentLatentStateV26:
    prev_v25 = (
        prev_state.inner_v25 if prev_state is not None else None)
    new_v25 = step_persistent_state_v25(
        cell=cell, prev_state=prev_v25,
        carrier_values=list(carrier_values),
        turn_index=int(turn_index), role=str(role),
        branch_id=str(branch_id),
        **kwargs)
    sd = int(new_v25.state_dim)
    if prev_state is not None:
        prev_cmp = list(prev_state.compound_pressure_carrier)
    else:
        prev_cmp = [0.0] * sd
    if compound_pressure_skip_v26 is not None:
        cmp = list(compound_pressure_skip_v26)[:sd]
        while len(cmp) < sd:
            cmp.append(0.0)
        a = float(max(0.0, min(
            1.0, float(compound_pressure_ema_alpha))))
        new_cmp = [
            a * float(cmp[i]) + (1.0 - a) * float(
                prev_cmp[i] if i < len(prev_cmp) else 0.0)
            for i in range(sd)]
    else:
        new_cmp = list(prev_cmp)
        while len(new_cmp) < sd:
            new_cmp.append(0.0)
    return PersistentLatentStateV26(
        inner_v25=new_v25,
        compound_pressure_carrier=tuple(new_cmp),
        distractor_rank=int(W74_DEFAULT_V26_DISTRACTOR_RANK),
    )


@dataclasses.dataclass(frozen=True)
class PersistentLatentStateV26Witness:
    schema: str
    chain_cid: str
    n_states: int
    n_layers: int
    distractor_rank: int
    twenty_third_skip_present: bool
    compound_pressure_carrier_l1_sum: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "chain_cid": str(self.chain_cid),
            "n_states": int(self.n_states),
            "n_layers": int(self.n_layers),
            "distractor_rank": int(self.distractor_rank),
            "twenty_third_skip_present": bool(
                self.twenty_third_skip_present),
            "compound_pressure_carrier_l1_sum": float(round(
                self.compound_pressure_carrier_l1_sum, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w74_persistent_v26_witness",
            "witness": self.to_dict()})


def emit_persistent_v26_witness(
        chain: PersistentLatentStateV26Chain,
) -> PersistentLatentStateV26Witness:
    cmp_sum = float(sum(
        abs(float(v))
        for s in chain.states.values()
        for v in s.compound_pressure_carrier))
    return PersistentLatentStateV26Witness(
        schema=W74_PERSISTENT_V26_SCHEMA_VERSION,
        chain_cid=str(chain.cid()),
        n_states=int(len(chain.states)),
        n_layers=int(W74_DEFAULT_V26_N_LAYERS),
        distractor_rank=int(W74_DEFAULT_V26_DISTRACTOR_RANK),
        twenty_third_skip_present=bool(cmp_sum > 0.0),
        compound_pressure_carrier_l1_sum=float(cmp_sum),
    )


__all__ = [
    "W74_PERSISTENT_V26_SCHEMA_VERSION",
    "W74_DEFAULT_V26_STATE_DIM",
    "W74_DEFAULT_V26_N_LAYERS",
    "W74_DEFAULT_V26_MAX_CHAIN_WALK_DEPTH",
    "W74_DEFAULT_V26_DISTRACTOR_RANK",
    "W74_V26_NO_PARENT_STATE",
    "PersistentLatentStateV26",
    "PersistentLatentStateV26Chain",
    "PersistentLatentStateV26Witness",
    "step_persistent_state_v26",
    "emit_persistent_v26_witness",
]
