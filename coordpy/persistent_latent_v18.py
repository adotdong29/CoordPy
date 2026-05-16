"""W66 M10 — Persistent Latent State V18.

Strictly extends W65's ``coordpy.persistent_latent_v17``. V18 adds:

* **17 layers** (vs V17's 16).
* **Fifteenth persistent skip-link** — V17's fourteen plus a new
  *team-failure-recovery EMA* that carries the W66 multi-agent
  team-failure-recovery scalar.
* ``max_chain_walk_depth = 8192`` (carried forward; W66 keeps the
  W65 cap).
* **Larger distractor basis** — V18 is **rank-16** (V17 was 14).

V18 strictly extends V17: with ``team_failure_recovery_skip_v18 =
None``, the new EMA stays at the prior value (no-op) and V18
reduces to V17 byte-for-byte.

Honest scope (W66)
------------------

* V18 wrapper still does NOT train the V13 outer GRU.
  ``W66-L-V18-OUTER-NOT-TRAINED-CAP`` documents.
* The new EMA shares the same shape and update law as V17's EMAs.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

from .persistent_latent_v12 import V12StackedCell
from .persistent_latent_v17 import (
    PersistentLatentStateV17,
    PersistentLatentStateV17Chain,
    W65_DEFAULT_V17_MAX_CHAIN_WALK_DEPTH,
    W65_DEFAULT_V17_STATE_DIM,
    step_persistent_state_v17,
)
from .tiny_substrate_v3 import _sha256_hex


W66_PERSISTENT_V18_SCHEMA_VERSION: str = (
    "coordpy.persistent_latent_v18.v1")
W66_DEFAULT_V18_STATE_DIM: int = W65_DEFAULT_V17_STATE_DIM
W66_DEFAULT_V18_N_LAYERS: int = 17
W66_DEFAULT_V18_MAX_CHAIN_WALK_DEPTH: int = (
    W65_DEFAULT_V17_MAX_CHAIN_WALK_DEPTH)
W66_DEFAULT_V18_DISTRACTOR_RANK: int = 16
W66_V18_NO_PARENT_STATE: str = "no_parent_v18_state"


@dataclasses.dataclass(frozen=True)
class PersistentLatentStateV18:
    inner_v17: PersistentLatentStateV17
    team_failure_recovery_carrier: tuple[float, ...]
    distractor_rank: int

    @property
    def turn_index(self) -> int:
        return int(self.inner_v17.turn_index)

    @property
    def role(self) -> str:
        return str(self.inner_v17.role)

    @property
    def state_dim(self) -> int:
        return int(self.inner_v17.state_dim)

    @property
    def n_layers(self) -> int:
        return int(self.inner_v17.n_layers) + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W66_PERSISTENT_V18_SCHEMA_VERSION,
            "inner_v17_cid": str(self.inner_v17.cid()),
            "team_failure_recovery_carrier": [
                float(round(float(x), 12))
                for x in self.team_failure_recovery_carrier],
            "distractor_rank": int(self.distractor_rank),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w66_v18_persistent_state",
            "state": self.to_dict()})


@dataclasses.dataclass
class PersistentLatentStateV18Chain:
    states: dict[str, PersistentLatentStateV18] = (
        dataclasses.field(default_factory=dict))

    @classmethod
    def empty(cls) -> "PersistentLatentStateV18Chain":
        return cls(states={})

    def add(self, s: PersistentLatentStateV18) -> None:
        self.states[s.cid()] = s

    def get(self, cid: str) -> PersistentLatentStateV18 | None:
        return self.states.get(str(cid))

    def walk_from(
            self, leaf_cid: str, *,
            max_depth: int = (
                W66_DEFAULT_V18_MAX_CHAIN_WALK_DEPTH),
    ) -> list[PersistentLatentStateV18]:
        out: list[PersistentLatentStateV18] = []
        cur = self.get(leaf_cid)
        seen: set[str] = set()
        steps = 0
        while cur is not None and steps < int(max_depth):
            out.append(cur)
            seen.add(cur.cid())
            inner_parent_cid = (
                cur.inner_v17.inner_v16.inner_v15.inner_v14
                .inner_v13.inner_v12.parent_state_cid)
            parent_inner = None
            for c in self.states.values():
                if (c.inner_v17.inner_v16.inner_v15.inner_v14
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
            "kind": "w66_v18_state_chain",
            "members": [
                {"cid": c, "state": s.to_dict()}
                for c, s in sorted(self.states.items())],
        })


def step_persistent_state_v18(
        *, cell: V12StackedCell,
        prev_state: PersistentLatentStateV18 | None,
        carrier_values: Sequence[float],
        turn_index: int, role: str,
        branch_id: str = "main",
        team_failure_recovery_skip_v18: (
            Sequence[float] | None) = None,
        team_failure_recovery_ema_alpha: float = 0.10,
        **kwargs: Any,
) -> PersistentLatentStateV18:
    prev_v17 = (
        prev_state.inner_v17 if prev_state is not None else None)
    new_v17 = step_persistent_state_v17(
        cell=cell, prev_state=prev_v17,
        carrier_values=list(carrier_values),
        turn_index=int(turn_index), role=str(role),
        branch_id=str(branch_id),
        **kwargs)
    sd = int(new_v17.state_dim)
    if prev_state is not None:
        prev_tfr = list(prev_state.team_failure_recovery_carrier)
    else:
        prev_tfr = [0.0] * sd
    if team_failure_recovery_skip_v18 is not None:
        tfr = list(team_failure_recovery_skip_v18)[:sd]
        while len(tfr) < sd:
            tfr.append(0.0)
        a = float(max(0.0, min(
            1.0, float(team_failure_recovery_ema_alpha))))
        new_tfr = [
            a * float(tfr[i]) + (1.0 - a) * float(
                prev_tfr[i] if i < len(prev_tfr) else 0.0)
            for i in range(sd)]
    else:
        new_tfr = list(prev_tfr)
        while len(new_tfr) < sd:
            new_tfr.append(0.0)
    return PersistentLatentStateV18(
        inner_v17=new_v17,
        team_failure_recovery_carrier=tuple(new_tfr),
        distractor_rank=int(W66_DEFAULT_V18_DISTRACTOR_RANK),
    )


@dataclasses.dataclass(frozen=True)
class PersistentLatentStateV18Witness:
    schema: str
    chain_cid: str
    n_states: int
    chain_walk_depth_used: int
    n_layers: int
    distractor_rank: int
    fifteenth_skip_present: bool
    team_failure_recovery_carrier_l1_sum: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "chain_cid": str(self.chain_cid),
            "n_states": int(self.n_states),
            "chain_walk_depth_used": int(
                self.chain_walk_depth_used),
            "n_layers": int(self.n_layers),
            "distractor_rank": int(self.distractor_rank),
            "fifteenth_skip_present": bool(
                self.fifteenth_skip_present),
            "team_failure_recovery_carrier_l1_sum": float(round(
                self.team_failure_recovery_carrier_l1_sum, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w66_persistent_v18_witness",
            "witness": self.to_dict()})


def emit_persistent_v18_witness(
        chain: PersistentLatentStateV18Chain,
        last_leaf_cid: str,
        *, max_depth: int = (
            W66_DEFAULT_V18_MAX_CHAIN_WALK_DEPTH),
) -> PersistentLatentStateV18Witness:
    walk = chain.walk_from(
        last_leaf_cid, max_depth=int(max_depth))
    tfr_sum = float(sum(
        abs(float(v))
        for s in walk
        for v in s.team_failure_recovery_carrier))
    return PersistentLatentStateV18Witness(
        schema=W66_PERSISTENT_V18_SCHEMA_VERSION,
        chain_cid=str(chain.cid()),
        n_states=int(len(chain.states)),
        chain_walk_depth_used=int(len(walk)),
        n_layers=int(W66_DEFAULT_V18_N_LAYERS),
        distractor_rank=int(W66_DEFAULT_V18_DISTRACTOR_RANK),
        fifteenth_skip_present=bool(tfr_sum > 0.0),
        team_failure_recovery_carrier_l1_sum=float(tfr_sum),
    )


__all__ = [
    "W66_PERSISTENT_V18_SCHEMA_VERSION",
    "W66_DEFAULT_V18_STATE_DIM",
    "W66_DEFAULT_V18_N_LAYERS",
    "W66_DEFAULT_V18_MAX_CHAIN_WALK_DEPTH",
    "W66_DEFAULT_V18_DISTRACTOR_RANK",
    "W66_V18_NO_PARENT_STATE",
    "PersistentLatentStateV18",
    "PersistentLatentStateV18Chain",
    "PersistentLatentStateV18Witness",
    "step_persistent_state_v18",
    "emit_persistent_v18_witness",
]
