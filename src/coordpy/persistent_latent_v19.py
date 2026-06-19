"""W67 M10 — Persistent Latent State V19.

Strictly extends W66's ``coordpy.persistent_latent_v18``. V19 adds:

* **18 layers** (vs V18's 17).
* **Sixteenth persistent skip-link** — V18's fifteen plus a new
  *role-dropout-recovery EMA* that carries the W67 role-dropout
  scalar.
* ``max_chain_walk_depth = 16384`` (W67 doubles the W66 cap).
* **Larger distractor basis** — V19 is **rank-18** (V18 was 16).

V19 strictly extends V18: with ``role_dropout_recovery_skip_v19 =
None``, the new EMA stays at the prior value (no-op) and V19
reduces to V18 byte-for-byte.

Honest scope (W67)
------------------

* V19 wrapper still does NOT train the V13 outer GRU.
  ``W67-L-V19-OUTER-NOT-TRAINED-CAP`` documents.
* The new EMA shares the same shape and update law as V18's EMAs.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

from .persistent_latent_v12 import V12StackedCell
from .persistent_latent_v18 import (
    PersistentLatentStateV18,
    PersistentLatentStateV18Chain,
    W66_DEFAULT_V18_STATE_DIM,
    step_persistent_state_v18,
)
from .tiny_substrate_v3 import _sha256_hex


W67_PERSISTENT_V19_SCHEMA_VERSION: str = (
    "coordpy.persistent_latent_v19.v1")
W67_DEFAULT_V19_STATE_DIM: int = W66_DEFAULT_V18_STATE_DIM
W67_DEFAULT_V19_N_LAYERS: int = 18
W67_DEFAULT_V19_MAX_CHAIN_WALK_DEPTH: int = 16384
W67_DEFAULT_V19_DISTRACTOR_RANK: int = 18
W67_V19_NO_PARENT_STATE: str = "no_parent_v19_state"


@dataclasses.dataclass(frozen=True)
class PersistentLatentStateV19:
    inner_v18: PersistentLatentStateV18
    role_dropout_recovery_carrier: tuple[float, ...]
    distractor_rank: int

    @property
    def turn_index(self) -> int:
        return int(self.inner_v18.turn_index)

    @property
    def role(self) -> str:
        return str(self.inner_v18.role)

    @property
    def state_dim(self) -> int:
        return int(self.inner_v18.state_dim)

    @property
    def n_layers(self) -> int:
        return int(self.inner_v18.n_layers) + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W67_PERSISTENT_V19_SCHEMA_VERSION,
            "inner_v18_cid": str(self.inner_v18.cid()),
            "role_dropout_recovery_carrier": [
                float(round(float(x), 12))
                for x in self.role_dropout_recovery_carrier],
            "distractor_rank": int(self.distractor_rank),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w67_v19_persistent_state",
            "state": self.to_dict()})


@dataclasses.dataclass
class PersistentLatentStateV19Chain:
    states: dict[str, PersistentLatentStateV19] = (
        dataclasses.field(default_factory=dict))

    @classmethod
    def empty(cls) -> "PersistentLatentStateV19Chain":
        return cls(states={})

    def add(self, s: PersistentLatentStateV19) -> None:
        self.states[s.cid()] = s

    def get(self, cid: str) -> PersistentLatentStateV19 | None:
        return self.states.get(str(cid))

    def walk_from(
            self, leaf_cid: str, *,
            max_depth: int = (
                W67_DEFAULT_V19_MAX_CHAIN_WALK_DEPTH),
    ) -> list[PersistentLatentStateV19]:
        out: list[PersistentLatentStateV19] = []
        cur = self.get(leaf_cid)
        seen: set[str] = set()
        steps = 0
        while cur is not None and steps < int(max_depth):
            out.append(cur)
            seen.add(cur.cid())
            inner_parent_cid = (
                cur.inner_v18.inner_v17.inner_v16.inner_v15.inner_v14
                .inner_v13.inner_v12.parent_state_cid)
            parent_inner = None
            for c in self.states.values():
                if (c.inner_v18.inner_v17.inner_v16.inner_v15.inner_v14
                        .inner_v13.cid()
                        == inner_parent_cid):
                    parent_inner = c
                    break
            if parent_inner is None or parent_inner.cid() in seen:
                break
            cur = parent_inner
            steps += 1
        return out

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w67_v19_state_chain",
            "members": [
                {"cid": c, "state": s.to_dict()}
                for c, s in sorted(self.states.items())],
        })


def step_persistent_state_v19(
        *, cell: V12StackedCell,
        prev_state: PersistentLatentStateV19 | None,
        carrier_values: Sequence[float],
        turn_index: int, role: str,
        branch_id: str = "main",
        role_dropout_recovery_skip_v19: (
            Sequence[float] | None) = None,
        role_dropout_recovery_ema_alpha: float = 0.10,
        **kwargs: Any,
) -> PersistentLatentStateV19:
    prev_v18 = (
        prev_state.inner_v18 if prev_state is not None else None)
    new_v18 = step_persistent_state_v18(
        cell=cell, prev_state=prev_v18,
        carrier_values=list(carrier_values),
        turn_index=int(turn_index), role=str(role),
        branch_id=str(branch_id),
        **kwargs)
    sd = int(new_v18.state_dim)
    if prev_state is not None:
        prev_rd = list(prev_state.role_dropout_recovery_carrier)
    else:
        prev_rd = [0.0] * sd
    if role_dropout_recovery_skip_v19 is not None:
        rd = list(role_dropout_recovery_skip_v19)[:sd]
        while len(rd) < sd:
            rd.append(0.0)
        a = float(max(0.0, min(
            1.0, float(role_dropout_recovery_ema_alpha))))
        new_rd = [
            a * float(rd[i]) + (1.0 - a) * float(
                prev_rd[i] if i < len(prev_rd) else 0.0)
            for i in range(sd)]
    else:
        new_rd = list(prev_rd)
        while len(new_rd) < sd:
            new_rd.append(0.0)
    return PersistentLatentStateV19(
        inner_v18=new_v18,
        role_dropout_recovery_carrier=tuple(new_rd),
        distractor_rank=int(W67_DEFAULT_V19_DISTRACTOR_RANK),
    )


@dataclasses.dataclass(frozen=True)
class PersistentLatentStateV19Witness:
    schema: str
    chain_cid: str
    n_states: int
    chain_walk_depth_used: int
    n_layers: int
    distractor_rank: int
    sixteenth_skip_present: bool
    role_dropout_recovery_carrier_l1_sum: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "chain_cid": str(self.chain_cid),
            "n_states": int(self.n_states),
            "chain_walk_depth_used": int(
                self.chain_walk_depth_used),
            "n_layers": int(self.n_layers),
            "distractor_rank": int(self.distractor_rank),
            "sixteenth_skip_present": bool(
                self.sixteenth_skip_present),
            "role_dropout_recovery_carrier_l1_sum": float(round(
                self.role_dropout_recovery_carrier_l1_sum, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w67_persistent_v19_witness",
            "witness": self.to_dict()})


def emit_persistent_v19_witness(
        chain: PersistentLatentStateV19Chain,
        last_leaf_cid: str,
        *, max_depth: int = (
            W67_DEFAULT_V19_MAX_CHAIN_WALK_DEPTH),
) -> PersistentLatentStateV19Witness:
    walk = chain.walk_from(
        last_leaf_cid, max_depth=int(max_depth))
    rd_sum = float(sum(
        abs(float(v))
        for s in walk
        for v in s.role_dropout_recovery_carrier))
    return PersistentLatentStateV19Witness(
        schema=W67_PERSISTENT_V19_SCHEMA_VERSION,
        chain_cid=str(chain.cid()),
        n_states=int(len(chain.states)),
        chain_walk_depth_used=int(len(walk)),
        n_layers=int(W67_DEFAULT_V19_N_LAYERS),
        distractor_rank=int(W67_DEFAULT_V19_DISTRACTOR_RANK),
        sixteenth_skip_present=bool(rd_sum > 0.0),
        role_dropout_recovery_carrier_l1_sum=float(rd_sum),
    )


__all__ = [
    "W67_PERSISTENT_V19_SCHEMA_VERSION",
    "W67_DEFAULT_V19_STATE_DIM",
    "W67_DEFAULT_V19_N_LAYERS",
    "W67_DEFAULT_V19_MAX_CHAIN_WALK_DEPTH",
    "W67_DEFAULT_V19_DISTRACTOR_RANK",
    "W67_V19_NO_PARENT_STATE",
    "PersistentLatentStateV19",
    "PersistentLatentStateV19Chain",
    "PersistentLatentStateV19Witness",
    "step_persistent_state_v19",
    "emit_persistent_v19_witness",
]
