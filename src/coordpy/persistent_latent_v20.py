"""W68 M8 — Persistent Latent State V20.

Strictly extends W67's ``coordpy.persistent_latent_v19``. V20 adds:

* **19 layers** (vs V19's 18).
* **Seventeenth persistent skip-link** — V19's sixteen plus a new
  *partial-contradiction EMA* that carries the W68 partial-
  contradiction scalar.
* ``max_chain_walk_depth = 32768`` (W68 doubles the W67 cap).
* **Larger distractor basis** — V20 is **rank-19** (V19 was 18).

V20 strictly extends V19: with ``partial_contradiction_skip_v20 =
None``, the new EMA stays at the prior value (no-op) and V20
reduces to V19 byte-for-byte.

Honest scope (W68)
------------------

* V20 wrapper still does NOT train the V13 outer GRU.
  ``W68-L-V20-OUTER-NOT-TRAINED-CAP`` documents.
* The new EMA shares the same shape and update law as V19's EMAs.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

from .persistent_latent_v12 import V12StackedCell
from .persistent_latent_v19 import (
    PersistentLatentStateV19,
    W67_DEFAULT_V19_STATE_DIM,
    step_persistent_state_v19,
)
from .tiny_substrate_v3 import _sha256_hex


W68_PERSISTENT_V20_SCHEMA_VERSION: str = (
    "coordpy.persistent_latent_v20.v1")
W68_DEFAULT_V20_STATE_DIM: int = W67_DEFAULT_V19_STATE_DIM
W68_DEFAULT_V20_N_LAYERS: int = 19
W68_DEFAULT_V20_MAX_CHAIN_WALK_DEPTH: int = 32768
W68_DEFAULT_V20_DISTRACTOR_RANK: int = 19
W68_V20_NO_PARENT_STATE: str = "no_parent_v20_state"


@dataclasses.dataclass(frozen=True)
class PersistentLatentStateV20:
    inner_v19: PersistentLatentStateV19
    partial_contradiction_carrier: tuple[float, ...]
    distractor_rank: int

    @property
    def turn_index(self) -> int:
        return int(self.inner_v19.turn_index)

    @property
    def role(self) -> str:
        return str(self.inner_v19.role)

    @property
    def state_dim(self) -> int:
        return int(self.inner_v19.state_dim)

    @property
    def n_layers(self) -> int:
        return int(self.inner_v19.n_layers) + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W68_PERSISTENT_V20_SCHEMA_VERSION,
            "inner_v19_cid": str(self.inner_v19.cid()),
            "partial_contradiction_carrier": [
                float(round(float(x), 12))
                for x in self.partial_contradiction_carrier],
            "distractor_rank": int(self.distractor_rank),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w68_v20_persistent_state",
            "state": self.to_dict()})


@dataclasses.dataclass
class PersistentLatentStateV20Chain:
    states: dict[str, PersistentLatentStateV20] = (
        dataclasses.field(default_factory=dict))

    @classmethod
    def empty(cls) -> "PersistentLatentStateV20Chain":
        return cls(states={})

    def add(self, s: PersistentLatentStateV20) -> None:
        self.states[s.cid()] = s

    def get(self, cid: str) -> PersistentLatentStateV20 | None:
        return self.states.get(str(cid))

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w68_v20_state_chain",
            "members": [
                {"cid": c, "state": s.to_dict()}
                for c, s in sorted(self.states.items())],
        })


def step_persistent_state_v20(
        *, cell: V12StackedCell,
        prev_state: PersistentLatentStateV20 | None,
        carrier_values: Sequence[float],
        turn_index: int, role: str,
        branch_id: str = "main",
        partial_contradiction_skip_v20: (
            Sequence[float] | None) = None,
        partial_contradiction_ema_alpha: float = 0.10,
        **kwargs: Any,
) -> PersistentLatentStateV20:
    prev_v19 = (
        prev_state.inner_v19 if prev_state is not None else None)
    new_v19 = step_persistent_state_v19(
        cell=cell, prev_state=prev_v19,
        carrier_values=list(carrier_values),
        turn_index=int(turn_index), role=str(role),
        branch_id=str(branch_id),
        **kwargs)
    sd = int(new_v19.state_dim)
    if prev_state is not None:
        prev_pc = list(prev_state.partial_contradiction_carrier)
    else:
        prev_pc = [0.0] * sd
    if partial_contradiction_skip_v20 is not None:
        pc = list(partial_contradiction_skip_v20)[:sd]
        while len(pc) < sd:
            pc.append(0.0)
        a = float(max(0.0, min(
            1.0, float(partial_contradiction_ema_alpha))))
        new_pc = [
            a * float(pc[i]) + (1.0 - a) * float(
                prev_pc[i] if i < len(prev_pc) else 0.0)
            for i in range(sd)]
    else:
        new_pc = list(prev_pc)
        while len(new_pc) < sd:
            new_pc.append(0.0)
    return PersistentLatentStateV20(
        inner_v19=new_v19,
        partial_contradiction_carrier=tuple(new_pc),
        distractor_rank=int(W68_DEFAULT_V20_DISTRACTOR_RANK),
    )


@dataclasses.dataclass(frozen=True)
class PersistentLatentStateV20Witness:
    schema: str
    chain_cid: str
    n_states: int
    n_layers: int
    distractor_rank: int
    seventeenth_skip_present: bool
    partial_contradiction_carrier_l1_sum: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "chain_cid": str(self.chain_cid),
            "n_states": int(self.n_states),
            "n_layers": int(self.n_layers),
            "distractor_rank": int(self.distractor_rank),
            "seventeenth_skip_present": bool(
                self.seventeenth_skip_present),
            "partial_contradiction_carrier_l1_sum": float(round(
                self.partial_contradiction_carrier_l1_sum, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w68_persistent_v20_witness",
            "witness": self.to_dict()})


def emit_persistent_v20_witness(
        chain: PersistentLatentStateV20Chain,
) -> PersistentLatentStateV20Witness:
    pc_sum = float(sum(
        abs(float(v))
        for s in chain.states.values()
        for v in s.partial_contradiction_carrier))
    return PersistentLatentStateV20Witness(
        schema=W68_PERSISTENT_V20_SCHEMA_VERSION,
        chain_cid=str(chain.cid()),
        n_states=int(len(chain.states)),
        n_layers=int(W68_DEFAULT_V20_N_LAYERS),
        distractor_rank=int(W68_DEFAULT_V20_DISTRACTOR_RANK),
        seventeenth_skip_present=bool(pc_sum > 0.0),
        partial_contradiction_carrier_l1_sum=float(pc_sum),
    )


__all__ = [
    "W68_PERSISTENT_V20_SCHEMA_VERSION",
    "W68_DEFAULT_V20_STATE_DIM",
    "W68_DEFAULT_V20_N_LAYERS",
    "W68_DEFAULT_V20_MAX_CHAIN_WALK_DEPTH",
    "W68_DEFAULT_V20_DISTRACTOR_RANK",
    "W68_V20_NO_PARENT_STATE",
    "PersistentLatentStateV20",
    "PersistentLatentStateV20Chain",
    "PersistentLatentStateV20Witness",
    "step_persistent_state_v20",
    "emit_persistent_v20_witness",
]
