"""W69 M8 — Persistent Latent State V21.

Strictly extends W68's ``coordpy.persistent_latent_v20``. V21 adds:

* **20 layers** (vs V20's 19).
* **Eighteenth persistent skip-link** — V20's seventeen plus a new
  *multi-branch-rejoin EMA*.
* ``max_chain_walk_depth = 65536`` (W69 doubles the W68 cap).
* **Larger distractor basis** — V21 is **rank-20** (V20 was 19).

V21 strictly extends V20: with ``multi_branch_rejoin_skip_v21 =
None``, the new EMA stays at the prior value (no-op) and V21
reduces to V20 byte-for-byte.

Honest scope (W69): ``W69-L-V21-OUTER-NOT-TRAINED-CAP``.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

from .persistent_latent_v12 import V12StackedCell
from .persistent_latent_v20 import (
    PersistentLatentStateV20,
    W68_DEFAULT_V20_STATE_DIM,
    step_persistent_state_v20,
)
from .tiny_substrate_v3 import _sha256_hex


W69_PERSISTENT_V21_SCHEMA_VERSION: str = (
    "coordpy.persistent_latent_v21.v1")
W69_DEFAULT_V21_STATE_DIM: int = W68_DEFAULT_V20_STATE_DIM
W69_DEFAULT_V21_N_LAYERS: int = 20
W69_DEFAULT_V21_MAX_CHAIN_WALK_DEPTH: int = 65536
W69_DEFAULT_V21_DISTRACTOR_RANK: int = 20
W69_V21_NO_PARENT_STATE: str = "no_parent_v21_state"


@dataclasses.dataclass(frozen=True)
class PersistentLatentStateV21:
    inner_v20: PersistentLatentStateV20
    multi_branch_rejoin_carrier: tuple[float, ...]
    distractor_rank: int

    @property
    def turn_index(self) -> int:
        return int(self.inner_v20.turn_index)

    @property
    def role(self) -> str:
        return str(self.inner_v20.role)

    @property
    def state_dim(self) -> int:
        return int(self.inner_v20.state_dim)

    @property
    def n_layers(self) -> int:
        return int(self.inner_v20.n_layers) + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W69_PERSISTENT_V21_SCHEMA_VERSION,
            "inner_v20_cid": str(self.inner_v20.cid()),
            "multi_branch_rejoin_carrier": [
                float(round(float(x), 12))
                for x in self.multi_branch_rejoin_carrier],
            "distractor_rank": int(self.distractor_rank),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w69_v21_persistent_state",
            "state": self.to_dict()})


@dataclasses.dataclass
class PersistentLatentStateV21Chain:
    states: dict[str, PersistentLatentStateV21] = (
        dataclasses.field(default_factory=dict))

    @classmethod
    def empty(cls) -> "PersistentLatentStateV21Chain":
        return cls(states={})

    def add(self, s: PersistentLatentStateV21) -> None:
        self.states[s.cid()] = s

    def get(self, cid: str) -> PersistentLatentStateV21 | None:
        return self.states.get(str(cid))

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w69_v21_state_chain",
            "members": [
                {"cid": c, "state": s.to_dict()}
                for c, s in sorted(self.states.items())],
        })


def step_persistent_state_v21(
        *, cell: V12StackedCell,
        prev_state: PersistentLatentStateV21 | None,
        carrier_values: Sequence[float],
        turn_index: int, role: str,
        branch_id: str = "main",
        multi_branch_rejoin_skip_v21: (
            Sequence[float] | None) = None,
        multi_branch_rejoin_ema_alpha: float = 0.10,
        **kwargs: Any,
) -> PersistentLatentStateV21:
    prev_v20 = (
        prev_state.inner_v20 if prev_state is not None else None)
    new_v20 = step_persistent_state_v20(
        cell=cell, prev_state=prev_v20,
        carrier_values=list(carrier_values),
        turn_index=int(turn_index), role=str(role),
        branch_id=str(branch_id),
        **kwargs)
    sd = int(new_v20.state_dim)
    if prev_state is not None:
        prev_mbr = list(prev_state.multi_branch_rejoin_carrier)
    else:
        prev_mbr = [0.0] * sd
    if multi_branch_rejoin_skip_v21 is not None:
        mbr = list(multi_branch_rejoin_skip_v21)[:sd]
        while len(mbr) < sd:
            mbr.append(0.0)
        a = float(max(0.0, min(
            1.0, float(multi_branch_rejoin_ema_alpha))))
        new_mbr = [
            a * float(mbr[i]) + (1.0 - a) * float(
                prev_mbr[i] if i < len(prev_mbr) else 0.0)
            for i in range(sd)]
    else:
        new_mbr = list(prev_mbr)
        while len(new_mbr) < sd:
            new_mbr.append(0.0)
    return PersistentLatentStateV21(
        inner_v20=new_v20,
        multi_branch_rejoin_carrier=tuple(new_mbr),
        distractor_rank=int(W69_DEFAULT_V21_DISTRACTOR_RANK),
    )


@dataclasses.dataclass(frozen=True)
class PersistentLatentStateV21Witness:
    schema: str
    chain_cid: str
    n_states: int
    n_layers: int
    distractor_rank: int
    eighteenth_skip_present: bool
    multi_branch_rejoin_carrier_l1_sum: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "chain_cid": str(self.chain_cid),
            "n_states": int(self.n_states),
            "n_layers": int(self.n_layers),
            "distractor_rank": int(self.distractor_rank),
            "eighteenth_skip_present": bool(
                self.eighteenth_skip_present),
            "multi_branch_rejoin_carrier_l1_sum": float(round(
                self.multi_branch_rejoin_carrier_l1_sum, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w69_persistent_v21_witness",
            "witness": self.to_dict()})


def emit_persistent_v21_witness(
        chain: PersistentLatentStateV21Chain,
) -> PersistentLatentStateV21Witness:
    mbr_sum = float(sum(
        abs(float(v))
        for s in chain.states.values()
        for v in s.multi_branch_rejoin_carrier))
    return PersistentLatentStateV21Witness(
        schema=W69_PERSISTENT_V21_SCHEMA_VERSION,
        chain_cid=str(chain.cid()),
        n_states=int(len(chain.states)),
        n_layers=int(W69_DEFAULT_V21_N_LAYERS),
        distractor_rank=int(W69_DEFAULT_V21_DISTRACTOR_RANK),
        eighteenth_skip_present=bool(mbr_sum > 0.0),
        multi_branch_rejoin_carrier_l1_sum=float(mbr_sum),
    )


__all__ = [
    "W69_PERSISTENT_V21_SCHEMA_VERSION",
    "W69_DEFAULT_V21_STATE_DIM",
    "W69_DEFAULT_V21_N_LAYERS",
    "W69_DEFAULT_V21_MAX_CHAIN_WALK_DEPTH",
    "W69_DEFAULT_V21_DISTRACTOR_RANK",
    "W69_V21_NO_PARENT_STATE",
    "PersistentLatentStateV21",
    "PersistentLatentStateV21Chain",
    "PersistentLatentStateV21Witness",
    "step_persistent_state_v21",
    "emit_persistent_v21_witness",
]
