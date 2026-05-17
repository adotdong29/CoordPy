"""W75 — Persistent Latent State V27.

Strictly extends W74's ``coordpy.persistent_latent_v26``. V26 was
25 layers + twenty-three skip carriers + ``max_chain_walk_depth=
2097152``. V27 adds:

* **26 layers** (vs V26's 25).
* **Twenty-fourth persistent skip-link** — V26's twenty-three plus
  a new *compound-chain-pressure EMA carrier*.
* ``max_chain_walk_depth = 4194304`` (W75 doubles the W74 cap).
* **Larger distractor basis** — V27 is **rank-26** (V26 was 25).

V27 strictly extends V26: with ``compound_chain_pressure_skip_v27 =
None``, the new EMA stays at the prior value (no-op) and V27
reduces to V26 byte-for-byte.

Honest scope (W75): ``W75-L-V27-OUTER-NOT-TRAINED-CAP``.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

from .persistent_latent_v12 import V12StackedCell
from .persistent_latent_v26 import (
    PersistentLatentStateV26,
    W74_DEFAULT_V26_STATE_DIM,
    step_persistent_state_v26,
)
from .tiny_substrate_v3 import _sha256_hex


W75_PERSISTENT_V27_SCHEMA_VERSION: str = (
    "coordpy.persistent_latent_v27.v1")
W75_DEFAULT_V27_STATE_DIM: int = W74_DEFAULT_V26_STATE_DIM
W75_DEFAULT_V27_N_LAYERS: int = 26
W75_DEFAULT_V27_MAX_CHAIN_WALK_DEPTH: int = 4194304
W75_DEFAULT_V27_DISTRACTOR_RANK: int = 26
W75_V27_NO_PARENT_STATE: str = "no_parent_v27_state"


@dataclasses.dataclass(frozen=True)
class PersistentLatentStateV27:
    inner_v26: PersistentLatentStateV26
    compound_chain_pressure_carrier: tuple[float, ...]
    distractor_rank: int

    @property
    def turn_index(self) -> int:
        return int(self.inner_v26.turn_index)

    @property
    def role(self) -> str:
        return str(self.inner_v26.role)

    @property
    def state_dim(self) -> int:
        return int(self.inner_v26.state_dim)

    @property
    def n_layers(self) -> int:
        return int(self.inner_v26.n_layers) + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W75_PERSISTENT_V27_SCHEMA_VERSION,
            "inner_v26_cid": str(self.inner_v26.cid()),
            "compound_chain_pressure_carrier": [
                float(round(float(x), 12))
                for x in self.compound_chain_pressure_carrier],
            "distractor_rank": int(self.distractor_rank),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w75_v27_persistent_state",
            "state": self.to_dict()})


@dataclasses.dataclass
class PersistentLatentStateV27Chain:
    states: dict[str, PersistentLatentStateV27] = (
        dataclasses.field(default_factory=dict))

    @classmethod
    def empty(cls) -> "PersistentLatentStateV27Chain":
        return cls(states={})

    def add(self, s: PersistentLatentStateV27) -> None:
        self.states[s.cid()] = s

    def get(self, cid: str) -> PersistentLatentStateV27 | None:
        return self.states.get(str(cid))

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w75_v27_state_chain",
            "members": [
                {"cid": c, "state": s.to_dict()}
                for c, s in sorted(self.states.items())],
        })


def step_persistent_state_v27(
        *, cell: V12StackedCell,
        prev_state: PersistentLatentStateV27 | None,
        carrier_values: Sequence[float],
        turn_index: int, role: str,
        branch_id: str = "main",
        compound_chain_pressure_skip_v27: (
            Sequence[float] | None) = None,
        compound_chain_pressure_ema_alpha: float = 0.10,
        **kwargs: Any,
) -> PersistentLatentStateV27:
    prev_v26 = (
        prev_state.inner_v26 if prev_state is not None else None)
    new_v26 = step_persistent_state_v26(
        cell=cell, prev_state=prev_v26,
        carrier_values=list(carrier_values),
        turn_index=int(turn_index), role=str(role),
        branch_id=str(branch_id),
        **kwargs)
    sd = int(new_v26.state_dim)
    if prev_state is not None:
        prev_chain = list(
            prev_state.compound_chain_pressure_carrier)
    else:
        prev_chain = [0.0] * sd
    if compound_chain_pressure_skip_v27 is not None:
        chain = list(compound_chain_pressure_skip_v27)[:sd]
        while len(chain) < sd:
            chain.append(0.0)
        a = float(max(0.0, min(
            1.0, float(compound_chain_pressure_ema_alpha))))
        new_chain = [
            a * float(chain[i]) + (1.0 - a) * float(
                prev_chain[i] if i < len(prev_chain) else 0.0)
            for i in range(sd)]
    else:
        new_chain = list(prev_chain)
        while len(new_chain) < sd:
            new_chain.append(0.0)
    return PersistentLatentStateV27(
        inner_v26=new_v26,
        compound_chain_pressure_carrier=tuple(new_chain),
        distractor_rank=int(W75_DEFAULT_V27_DISTRACTOR_RANK),
    )


@dataclasses.dataclass(frozen=True)
class PersistentLatentStateV27Witness:
    schema: str
    chain_cid: str
    n_states: int
    n_layers: int
    distractor_rank: int
    twenty_fourth_skip_present: bool
    compound_chain_pressure_carrier_l1_sum: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "chain_cid": str(self.chain_cid),
            "n_states": int(self.n_states),
            "n_layers": int(self.n_layers),
            "distractor_rank": int(self.distractor_rank),
            "twenty_fourth_skip_present": bool(
                self.twenty_fourth_skip_present),
            "compound_chain_pressure_carrier_l1_sum": float(
                round(
                    self
                    .compound_chain_pressure_carrier_l1_sum,
                    12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w75_persistent_v27_witness",
            "witness": self.to_dict()})


def emit_persistent_v27_witness(
        chain: PersistentLatentStateV27Chain,
) -> PersistentLatentStateV27Witness:
    chain_sum = float(sum(
        abs(float(v))
        for s in chain.states.values()
        for v in s.compound_chain_pressure_carrier))
    return PersistentLatentStateV27Witness(
        schema=W75_PERSISTENT_V27_SCHEMA_VERSION,
        chain_cid=str(chain.cid()),
        n_states=int(len(chain.states)),
        n_layers=int(W75_DEFAULT_V27_N_LAYERS),
        distractor_rank=int(W75_DEFAULT_V27_DISTRACTOR_RANK),
        twenty_fourth_skip_present=bool(chain_sum > 0.0),
        compound_chain_pressure_carrier_l1_sum=float(chain_sum),
    )


__all__ = [
    "W75_PERSISTENT_V27_SCHEMA_VERSION",
    "W75_DEFAULT_V27_STATE_DIM",
    "W75_DEFAULT_V27_N_LAYERS",
    "W75_DEFAULT_V27_MAX_CHAIN_WALK_DEPTH",
    "W75_DEFAULT_V27_DISTRACTOR_RANK",
    "W75_V27_NO_PARENT_STATE",
    "PersistentLatentStateV27",
    "PersistentLatentStateV27Chain",
    "PersistentLatentStateV27Witness",
    "step_persistent_state_v27",
    "emit_persistent_v27_witness",
]
