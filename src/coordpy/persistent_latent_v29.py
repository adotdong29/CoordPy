"""W77 — Persistent Latent State V29.

Strictly extends W76's ``coordpy.persistent_latent_v28``. V28 was
27 layers + twenty-five skip carriers + ``max_chain_walk_depth=
8388608``. V29 adds:

* **28 layers** (vs V28's 27).
* **Twenty-sixth persistent skip-link** — V28's twenty-five plus
  a new *post-restart-replacement-pressure EMA carrier*.
* ``max_chain_walk_depth = 16777216`` (W77 doubles the W76 cap).
* **Larger distractor basis** — V29 is **rank-28** (V28 was 27).

V29 strictly extends V28: with
``post_restart_replacement_pressure_skip_v29 = None`` the new EMA
stays at the prior value and V29 reduces to V28 byte-for-byte.

Honest scope (W77): ``W77-L-V29-OUTER-NOT-TRAINED-CAP``.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

from .persistent_latent_v12 import V12StackedCell
from .persistent_latent_v28 import (
    PersistentLatentStateV28,
    W76_DEFAULT_V28_STATE_DIM,
    step_persistent_state_v28,
)
from .tiny_substrate_v3 import _sha256_hex


W77_PERSISTENT_V29_SCHEMA_VERSION: str = (
    "coordpy.persistent_latent_v29.v1")
W77_DEFAULT_V29_STATE_DIM: int = W76_DEFAULT_V28_STATE_DIM
W77_DEFAULT_V29_N_LAYERS: int = 28
W77_DEFAULT_V29_MAX_CHAIN_WALK_DEPTH: int = 16777216
W77_DEFAULT_V29_DISTRACTOR_RANK: int = 28
W77_V29_NO_PARENT_STATE: str = "no_parent_v29_state"


@dataclasses.dataclass(frozen=True)
class PersistentLatentStateV29:
    inner_v28: PersistentLatentStateV28
    post_restart_replacement_pressure_carrier: tuple[float, ...]
    distractor_rank: int

    @property
    def turn_index(self) -> int:
        return int(self.inner_v28.turn_index)

    @property
    def role(self) -> str:
        return str(self.inner_v28.role)

    @property
    def state_dim(self) -> int:
        return int(self.inner_v28.state_dim)

    @property
    def n_layers(self) -> int:
        return int(self.inner_v28.n_layers) + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W77_PERSISTENT_V29_SCHEMA_VERSION,
            "inner_v28_cid": str(self.inner_v28.cid()),
            "post_restart_replacement_pressure_carrier": [
                float(round(float(x), 12))
                for x in
                self.post_restart_replacement_pressure_carrier],
            "distractor_rank": int(self.distractor_rank),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w77_v29_persistent_state",
            "state": self.to_dict()})


@dataclasses.dataclass
class PersistentLatentStateV29Chain:
    states: dict[str, PersistentLatentStateV29] = (
        dataclasses.field(default_factory=dict))

    @classmethod
    def empty(cls) -> "PersistentLatentStateV29Chain":
        return cls(states={})

    def add(self, s: PersistentLatentStateV29) -> None:
        self.states[s.cid()] = s

    def get(self, cid: str) -> PersistentLatentStateV29 | None:
        return self.states.get(str(cid))

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w77_v29_state_chain",
            "members": [
                {"cid": c, "state": s.to_dict()}
                for c, s in sorted(self.states.items())],
        })


def step_persistent_state_v29(
        *, cell: V12StackedCell,
        prev_state: PersistentLatentStateV29 | None,
        carrier_values: Sequence[float],
        turn_index: int, role: str,
        branch_id: str = "main",
        post_restart_replacement_pressure_skip_v29: (
            Sequence[float] | None) = None,
        post_restart_replacement_pressure_ema_alpha: float = 0.10,
        **kwargs: Any,
) -> PersistentLatentStateV29:
    prev_v28 = (
        prev_state.inner_v28 if prev_state is not None else None)
    new_v28 = step_persistent_state_v28(
        cell=cell, prev_state=prev_v28,
        carrier_values=list(carrier_values),
        turn_index=int(turn_index), role=str(role),
        branch_id=str(branch_id),
        **kwargs)
    sd = int(new_v28.state_dim)
    if prev_state is not None:
        prev_chain = list(
            prev_state.post_restart_replacement_pressure_carrier)
    else:
        prev_chain = [0.0] * sd
    if post_restart_replacement_pressure_skip_v29 is not None:
        chain = list(
            post_restart_replacement_pressure_skip_v29)[:sd]
        while len(chain) < sd:
            chain.append(0.0)
        a = float(max(0.0, min(
            1.0,
            float(post_restart_replacement_pressure_ema_alpha))))
        new_chain = [
            a * float(chain[i]) + (1.0 - a) * float(
                prev_chain[i] if i < len(prev_chain) else 0.0)
            for i in range(sd)]
    else:
        new_chain = list(prev_chain)
        while len(new_chain) < sd:
            new_chain.append(0.0)
    return PersistentLatentStateV29(
        inner_v28=new_v28,
        post_restart_replacement_pressure_carrier=tuple(new_chain),
        distractor_rank=int(W77_DEFAULT_V29_DISTRACTOR_RANK),
    )


@dataclasses.dataclass(frozen=True)
class PersistentLatentStateV29Witness:
    schema: str
    chain_cid: str
    n_states: int
    n_layers: int
    distractor_rank: int
    twenty_sixth_skip_present: bool
    post_restart_replacement_pressure_carrier_l1_sum: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "chain_cid": str(self.chain_cid),
            "n_states": int(self.n_states),
            "n_layers": int(self.n_layers),
            "distractor_rank": int(self.distractor_rank),
            "twenty_sixth_skip_present": bool(
                self.twenty_sixth_skip_present),
            "post_restart_replacement_pressure_carrier_l1_sum":
                float(round(
                    self
                    .post_restart_replacement_pressure_carrier_l1_sum,
                    12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w77_persistent_v29_witness",
            "witness": self.to_dict()})


def emit_persistent_v29_witness(
        chain: PersistentLatentStateV29Chain,
) -> PersistentLatentStateV29Witness:
    chain_sum = float(sum(
        abs(float(v))
        for s in chain.states.values()
        for v in s.post_restart_replacement_pressure_carrier))
    return PersistentLatentStateV29Witness(
        schema=W77_PERSISTENT_V29_SCHEMA_VERSION,
        chain_cid=str(chain.cid()),
        n_states=int(len(chain.states)),
        n_layers=int(W77_DEFAULT_V29_N_LAYERS),
        distractor_rank=int(W77_DEFAULT_V29_DISTRACTOR_RANK),
        twenty_sixth_skip_present=bool(chain_sum > 0.0),
        post_restart_replacement_pressure_carrier_l1_sum=float(
            chain_sum),
    )


__all__ = [
    "W77_PERSISTENT_V29_SCHEMA_VERSION",
    "W77_DEFAULT_V29_STATE_DIM",
    "W77_DEFAULT_V29_N_LAYERS",
    "W77_DEFAULT_V29_MAX_CHAIN_WALK_DEPTH",
    "W77_DEFAULT_V29_DISTRACTOR_RANK",
    "W77_V29_NO_PARENT_STATE",
    "PersistentLatentStateV29",
    "PersistentLatentStateV29Chain",
    "PersistentLatentStateV29Witness",
    "step_persistent_state_v29",
    "emit_persistent_v29_witness",
]
