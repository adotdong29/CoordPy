"""W70 — Persistent Latent State V22.

Strictly extends W69's ``coordpy.persistent_latent_v21``. V22 adds:

* **21 layers** (vs V21's 20).
* **Nineteenth persistent skip-link** — V21's eighteen plus a new
  *repair-dominance EMA*.
* ``max_chain_walk_depth = 131072`` (W70 doubles the W69 cap).
* **Larger distractor basis** — V22 is **rank-21** (V21 was 20).

V22 strictly extends V21: with ``repair_dominance_skip_v22 = None``,
the new EMA stays at the prior value (no-op) and V22 reduces to
V21 byte-for-byte.

Honest scope (W70): ``W70-L-V22-OUTER-NOT-TRAINED-CAP``.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

from .persistent_latent_v12 import V12StackedCell
from .persistent_latent_v21 import (
    PersistentLatentStateV21,
    W69_DEFAULT_V21_DISTRACTOR_RANK,
    W69_DEFAULT_V21_STATE_DIM,
    step_persistent_state_v21,
)
from .tiny_substrate_v3 import _sha256_hex


W70_PERSISTENT_V22_SCHEMA_VERSION: str = (
    "coordpy.persistent_latent_v22.v1")
W70_DEFAULT_V22_STATE_DIM: int = W69_DEFAULT_V21_STATE_DIM
W70_DEFAULT_V22_N_LAYERS: int = 21
W70_DEFAULT_V22_MAX_CHAIN_WALK_DEPTH: int = 131072
W70_DEFAULT_V22_DISTRACTOR_RANK: int = 21
W70_V22_NO_PARENT_STATE: str = "no_parent_v22_state"


@dataclasses.dataclass(frozen=True)
class PersistentLatentStateV22:
    inner_v21: PersistentLatentStateV21
    repair_dominance_carrier: tuple[float, ...]
    distractor_rank: int

    @property
    def turn_index(self) -> int:
        return int(self.inner_v21.turn_index)

    @property
    def role(self) -> str:
        return str(self.inner_v21.role)

    @property
    def state_dim(self) -> int:
        return int(self.inner_v21.state_dim)

    @property
    def n_layers(self) -> int:
        return int(self.inner_v21.n_layers) + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W70_PERSISTENT_V22_SCHEMA_VERSION,
            "inner_v21_cid": str(self.inner_v21.cid()),
            "repair_dominance_carrier": [
                float(round(float(x), 12))
                for x in self.repair_dominance_carrier],
            "distractor_rank": int(self.distractor_rank),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w70_v22_persistent_state",
            "state": self.to_dict()})


@dataclasses.dataclass
class PersistentLatentStateV22Chain:
    states: dict[str, PersistentLatentStateV22] = (
        dataclasses.field(default_factory=dict))

    @classmethod
    def empty(cls) -> "PersistentLatentStateV22Chain":
        return cls(states={})

    def add(self, s: PersistentLatentStateV22) -> None:
        self.states[s.cid()] = s

    def get(self, cid: str) -> PersistentLatentStateV22 | None:
        return self.states.get(str(cid))

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w70_v22_state_chain",
            "members": [
                {"cid": c, "state": s.to_dict()}
                for c, s in sorted(self.states.items())],
        })


def step_persistent_state_v22(
        *, cell: V12StackedCell,
        prev_state: PersistentLatentStateV22 | None,
        carrier_values: Sequence[float],
        turn_index: int, role: str,
        branch_id: str = "main",
        repair_dominance_skip_v22: (
            Sequence[float] | None) = None,
        repair_dominance_ema_alpha: float = 0.10,
        **kwargs: Any,
) -> PersistentLatentStateV22:
    prev_v21 = (
        prev_state.inner_v21 if prev_state is not None else None)
    new_v21 = step_persistent_state_v21(
        cell=cell, prev_state=prev_v21,
        carrier_values=list(carrier_values),
        turn_index=int(turn_index), role=str(role),
        branch_id=str(branch_id),
        **kwargs)
    sd = int(new_v21.state_dim)
    if prev_state is not None:
        prev_rd = list(prev_state.repair_dominance_carrier)
    else:
        prev_rd = [0.0] * sd
    if repair_dominance_skip_v22 is not None:
        rd = list(repair_dominance_skip_v22)[:sd]
        while len(rd) < sd:
            rd.append(0.0)
        a = float(max(0.0, min(
            1.0, float(repair_dominance_ema_alpha))))
        new_rd = [
            a * float(rd[i]) + (1.0 - a) * float(
                prev_rd[i] if i < len(prev_rd) else 0.0)
            for i in range(sd)]
    else:
        new_rd = list(prev_rd)
        while len(new_rd) < sd:
            new_rd.append(0.0)
    return PersistentLatentStateV22(
        inner_v21=new_v21,
        repair_dominance_carrier=tuple(new_rd),
        distractor_rank=int(W70_DEFAULT_V22_DISTRACTOR_RANK),
    )


@dataclasses.dataclass(frozen=True)
class PersistentLatentStateV22Witness:
    schema: str
    chain_cid: str
    n_states: int
    n_layers: int
    distractor_rank: int
    nineteenth_skip_present: bool
    repair_dominance_carrier_l1_sum: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "chain_cid": str(self.chain_cid),
            "n_states": int(self.n_states),
            "n_layers": int(self.n_layers),
            "distractor_rank": int(self.distractor_rank),
            "nineteenth_skip_present": bool(
                self.nineteenth_skip_present),
            "repair_dominance_carrier_l1_sum": float(round(
                self.repair_dominance_carrier_l1_sum, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w70_persistent_v22_witness",
            "witness": self.to_dict()})


def emit_persistent_v22_witness(
        chain: PersistentLatentStateV22Chain,
) -> PersistentLatentStateV22Witness:
    rd_sum = float(sum(
        abs(float(v))
        for s in chain.states.values()
        for v in s.repair_dominance_carrier))
    return PersistentLatentStateV22Witness(
        schema=W70_PERSISTENT_V22_SCHEMA_VERSION,
        chain_cid=str(chain.cid()),
        n_states=int(len(chain.states)),
        n_layers=int(W70_DEFAULT_V22_N_LAYERS),
        distractor_rank=int(W70_DEFAULT_V22_DISTRACTOR_RANK),
        nineteenth_skip_present=bool(rd_sum > 0.0),
        repair_dominance_carrier_l1_sum=float(rd_sum),
    )


__all__ = [
    "W70_PERSISTENT_V22_SCHEMA_VERSION",
    "W70_DEFAULT_V22_STATE_DIM",
    "W70_DEFAULT_V22_N_LAYERS",
    "W70_DEFAULT_V22_MAX_CHAIN_WALK_DEPTH",
    "W70_DEFAULT_V22_DISTRACTOR_RANK",
    "W70_V22_NO_PARENT_STATE",
    "PersistentLatentStateV22",
    "PersistentLatentStateV22Chain",
    "PersistentLatentStateV22Witness",
    "step_persistent_state_v22",
    "emit_persistent_v22_witness",
]
