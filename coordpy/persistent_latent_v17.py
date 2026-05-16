"""W65 M10 — Persistent Latent State V17.

Strictly extends W64's ``coordpy.persistent_latent_v16``. V17
adds:

* **16 layers** (vs V16's 15).
* **Fourteenth persistent skip-link** — V16's thirteen plus a new
  *team-task-success EMA* that carries the W65 multi-agent team
  task-success scalar.
* ``max_chain_walk_depth = 8192`` (vs V16's 6144).
* **Larger distractor basis** — V16 was rank-12; V17 is **rank-14**.

V17 strictly extends V16: with ``team_task_success_skip_v17 = None``,
the new EMA stays at the prior value (no-op) and V17 reduces to
V16 byte-for-byte.

Honest scope (W65)
------------------

* V17 wrapper still does NOT train the V13 outer GRU.
  ``W65-L-V17-OUTER-NOT-TRAINED-CAP`` documents.
* The new EMA shares the same shape and update law as V16's EMAs.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

from .persistent_latent_v12 import V12StackedCell
from .persistent_latent_v16 import (
    PersistentLatentStateV16,
    PersistentLatentStateV16Chain,
    W64_DEFAULT_V16_STATE_DIM,
    step_persistent_state_v16,
)
from .tiny_substrate_v3 import _sha256_hex


W65_PERSISTENT_V17_SCHEMA_VERSION: str = (
    "coordpy.persistent_latent_v17.v1")
W65_DEFAULT_V17_STATE_DIM: int = W64_DEFAULT_V16_STATE_DIM
W65_DEFAULT_V17_N_LAYERS: int = 16
W65_DEFAULT_V17_MAX_CHAIN_WALK_DEPTH: int = 8192
W65_DEFAULT_V17_DISTRACTOR_RANK: int = 14
W65_V17_NO_PARENT_STATE: str = "no_parent_v17_state"


@dataclasses.dataclass(frozen=True)
class PersistentLatentStateV17:
    inner_v16: PersistentLatentStateV16
    team_task_success_carrier: tuple[float, ...]
    distractor_rank: int

    @property
    def turn_index(self) -> int:
        return int(self.inner_v16.turn_index)

    @property
    def role(self) -> str:
        return str(self.inner_v16.role)

    @property
    def state_dim(self) -> int:
        return int(self.inner_v16.state_dim)

    @property
    def n_layers(self) -> int:
        return int(self.inner_v16.n_layers) + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W65_PERSISTENT_V17_SCHEMA_VERSION,
            "inner_v16_cid": str(self.inner_v16.cid()),
            "team_task_success_carrier": [
                float(round(float(x), 12))
                for x in self.team_task_success_carrier],
            "distractor_rank": int(self.distractor_rank),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w65_v17_persistent_state",
            "state": self.to_dict()})


@dataclasses.dataclass
class PersistentLatentStateV17Chain:
    states: dict[str, PersistentLatentStateV17] = (
        dataclasses.field(default_factory=dict))

    @classmethod
    def empty(cls) -> "PersistentLatentStateV17Chain":
        return cls(states={})

    def add(self, s: PersistentLatentStateV17) -> None:
        self.states[s.cid()] = s

    def get(self, cid: str) -> PersistentLatentStateV17 | None:
        return self.states.get(str(cid))

    def walk_from(
            self, leaf_cid: str, *,
            max_depth: int = (
                W65_DEFAULT_V17_MAX_CHAIN_WALK_DEPTH),
    ) -> list[PersistentLatentStateV17]:
        out: list[PersistentLatentStateV17] = []
        cur = self.get(leaf_cid)
        seen: set[str] = set()
        steps = 0
        while cur is not None and steps < int(max_depth):
            out.append(cur)
            seen.add(cur.cid())
            inner_parent_cid = (
                cur.inner_v16.inner_v15.inner_v14.inner_v13
                .inner_v12.parent_state_cid)
            parent_inner = None
            for c in self.states.values():
                if (c.inner_v16.inner_v15.inner_v14.inner_v13.cid()
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
            "kind": "w65_v17_state_chain",
            "members": [
                {"cid": c, "state": s.to_dict()}
                for c, s in sorted(self.states.items())],
        })


def step_persistent_state_v17(
        *, cell: V12StackedCell,
        prev_state: PersistentLatentStateV17 | None,
        carrier_values: Sequence[float],
        turn_index: int, role: str,
        branch_id: str = "main",
        team_task_success_skip_v17: (
            Sequence[float] | None) = None,
        team_task_success_ema_alpha: float = 0.10,
        **kwargs: Any,
) -> PersistentLatentStateV17:
    prev_v16 = (
        prev_state.inner_v16 if prev_state is not None else None)
    new_v16 = step_persistent_state_v16(
        cell=cell, prev_state=prev_v16,
        carrier_values=list(carrier_values),
        turn_index=int(turn_index), role=str(role),
        branch_id=str(branch_id),
        **kwargs)
    sd = int(new_v16.state_dim)
    if prev_state is not None:
        prev_tts = list(prev_state.team_task_success_carrier)
    else:
        prev_tts = [0.0] * sd
    if team_task_success_skip_v17 is not None:
        ttss = list(team_task_success_skip_v17)[:sd]
        while len(ttss) < sd:
            ttss.append(0.0)
        a = float(max(0.0, min(
            1.0, float(team_task_success_ema_alpha))))
        new_tts = [
            a * float(ttss[i]) + (1.0 - a) * float(
                prev_tts[i] if i < len(prev_tts) else 0.0)
            for i in range(sd)]
    else:
        new_tts = list(prev_tts)
        while len(new_tts) < sd:
            new_tts.append(0.0)
    return PersistentLatentStateV17(
        inner_v16=new_v16,
        team_task_success_carrier=tuple(new_tts),
        distractor_rank=int(W65_DEFAULT_V17_DISTRACTOR_RANK),
    )


@dataclasses.dataclass(frozen=True)
class PersistentLatentStateV17Witness:
    schema: str
    chain_cid: str
    n_states: int
    chain_walk_depth_used: int
    n_layers: int
    distractor_rank: int
    fourteenth_skip_present: bool
    team_task_success_carrier_l1_sum: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "chain_cid": str(self.chain_cid),
            "n_states": int(self.n_states),
            "chain_walk_depth_used": int(
                self.chain_walk_depth_used),
            "n_layers": int(self.n_layers),
            "distractor_rank": int(self.distractor_rank),
            "fourteenth_skip_present": bool(
                self.fourteenth_skip_present),
            "team_task_success_carrier_l1_sum": float(round(
                self.team_task_success_carrier_l1_sum, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w65_persistent_v17_witness",
            "witness": self.to_dict()})


def emit_persistent_v17_witness(
        chain: PersistentLatentStateV17Chain,
        last_leaf_cid: str,
        *, max_depth: int = (
            W65_DEFAULT_V17_MAX_CHAIN_WALK_DEPTH),
) -> PersistentLatentStateV17Witness:
    walk = chain.walk_from(
        last_leaf_cid, max_depth=int(max_depth))
    tts_sum = float(sum(
        abs(float(v))
        for s in walk
        for v in s.team_task_success_carrier))
    return PersistentLatentStateV17Witness(
        schema=W65_PERSISTENT_V17_SCHEMA_VERSION,
        chain_cid=str(chain.cid()),
        n_states=int(len(chain.states)),
        chain_walk_depth_used=int(len(walk)),
        n_layers=int(W65_DEFAULT_V17_N_LAYERS),
        distractor_rank=int(W65_DEFAULT_V17_DISTRACTOR_RANK),
        fourteenth_skip_present=bool(tts_sum > 0.0),
        team_task_success_carrier_l1_sum=float(tts_sum),
    )


__all__ = [
    "W65_PERSISTENT_V17_SCHEMA_VERSION",
    "W65_DEFAULT_V17_STATE_DIM",
    "W65_DEFAULT_V17_N_LAYERS",
    "W65_DEFAULT_V17_MAX_CHAIN_WALK_DEPTH",
    "W65_DEFAULT_V17_DISTRACTOR_RANK",
    "W65_V17_NO_PARENT_STATE",
    "PersistentLatentStateV17",
    "PersistentLatentStateV17Chain",
    "PersistentLatentStateV17Witness",
    "step_persistent_state_v17",
    "emit_persistent_v17_witness",
]
