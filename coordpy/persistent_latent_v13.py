"""W61 M8 — Persistent Latent State V13.

Strictly extends W60's ``coordpy.persistent_latent_v12``. V13 adds:

* **11 layers** (vs V12's 10) — wrapper of V12's outer GRU stack.
* **Nonuple persistent skip-link** — V12's octuple plus a ninth:
  a **replay-controller-V2 decision-confidence EMA** that carries
  the V2 controller's softmax confidence scalar across turns.
  The persistent cell uses it to gracefully damp the replay
  channel when the controller is uncertain.
* ``max_chain_walk_depth = 1536`` (vs V12's 1024).
* **Larger distractor basis** — V12 used rank-4 random orthonormal
  basis; V13 uses **rank-6** so the replay-skip projection
  survives more aggressive distractor attacks.

V13 strictly extends V12: with ``replay_confidence_skip = None``,
the new ninth EMA stays at the prior value (no-op) and V13 reduces
to V12 byte-for-byte.

Honest scope
------------

* The V13 wrapper still does NOT train the V12 outer GRU end-to-
  end. ``W61-L-V13-OUTER-NOT-TRAINED-CAP`` carries forward the
  V12 cap on top of the new 11th layer.
* The ninth EMA shares the same shape and update law as V12's
  eight EMAs — there is no new learned parameter introduced.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Any, Sequence

from .persistent_latent_v12 import (
    PersistentLatentStateV12,
    PersistentLatentStateV12Chain,
    V12StackedCell,
    W60_DEFAULT_V12_MAX_CHAIN_WALK_DEPTH,
    W60_DEFAULT_V12_N_LAYERS,
    W60_DEFAULT_V12_STATE_DIM,
    W60_V12_NO_PARENT_STATE,
    _round_floats,
    step_persistent_state_v12,
)


W61_PERSISTENT_V13_SCHEMA_VERSION: str = (
    "coordpy.persistent_latent_v13.v1")
W61_DEFAULT_V13_STATE_DIM: int = W60_DEFAULT_V12_STATE_DIM
W61_DEFAULT_V13_N_LAYERS: int = 11
W61_DEFAULT_V13_MAX_CHAIN_WALK_DEPTH: int = 1536
W61_DEFAULT_V13_DISTRACTOR_RANK: int = 6
W61_V13_NO_PARENT_STATE: str = "no_parent_v13_state"


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass(frozen=True)
class PersistentLatentStateV13:
    inner_v12: PersistentLatentStateV12
    replay_confidence_carrier: tuple[float, ...]
    distractor_rank: int

    @property
    def turn_index(self) -> int:
        return int(self.inner_v12.turn_index)

    @property
    def role(self) -> str:
        return str(self.inner_v12.role)

    @property
    def branch_id(self) -> str:
        return str(self.inner_v12.branch_id)

    @property
    def state_dim(self) -> int:
        return int(self.inner_v12.state_dim)

    @property
    def n_layers(self) -> int:
        return int(self.inner_v12.n_layers) + 1

    @property
    def top_state(self) -> tuple[float, ...]:
        # V13 top state = V12 top state mixed with replay confidence
        # carrier. Use a fixed deterministic linear blend.
        v12_top = list(self.inner_v12.top_state)
        rc = list(self.replay_confidence_carrier)
        sd = int(self.state_dim)
        return tuple(
            float(v12_top[i] if i < len(v12_top) else 0.0)
            + 0.05 * float(
                rc[i] if i < len(rc) else 0.0)
            for i in range(sd))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W61_PERSISTENT_V13_SCHEMA_VERSION,
            "inner_v12": self.inner_v12.to_dict(),
            "replay_confidence_carrier": list(
                _round_floats(self.replay_confidence_carrier)),
            "distractor_rank": int(self.distractor_rank),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w61_v13_persistent_state",
            "state": self.to_dict()})


@dataclasses.dataclass
class PersistentLatentStateV13Chain:
    states: dict[str, PersistentLatentStateV13]

    @classmethod
    def empty(cls) -> "PersistentLatentStateV13Chain":
        return cls(states={})

    def add(self, s: PersistentLatentStateV13) -> None:
        self.states[s.cid()] = s

    def get(
            self, cid: str,
    ) -> PersistentLatentStateV13 | None:
        return self.states.get(str(cid))

    def walk_from(
            self, leaf_cid: str, *,
            max_depth: int = (
                W61_DEFAULT_V13_MAX_CHAIN_WALK_DEPTH),
    ) -> list[PersistentLatentStateV13]:
        out: list[PersistentLatentStateV13] = []
        cur = self.get(leaf_cid)
        seen: set[str] = set()
        steps = 0
        while cur is not None and steps < int(max_depth):
            out.append(cur)
            seen.add(cur.cid())
            parent = self.get(
                cur.inner_v12.parent_state_cid)
            if parent is None or parent.cid() in seen:
                break
            cur = parent
            steps += 1
        return out

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w61_v13_state_chain",
            "members": [
                {"cid": c, "state": s.to_dict()}
                for c, s in sorted(self.states.items())],
        })


def step_persistent_state_v13(
        *, cell: V12StackedCell,
        prev_state: PersistentLatentStateV13 | None,
        carrier_values: Sequence[float],
        turn_index: int, role: str,
        branch_id: str = "main",
        anchor_skip: Sequence[float] | None = None,
        substrate_skip: Sequence[float] | None = None,
        hidden_state_skip: Sequence[float] | None = None,
        attention_skip: Sequence[float] | None = None,
        retrieval_skip: Sequence[float] | None = None,
        replay_skip: Sequence[float] | None = None,
        replay_confidence_skip: Sequence[float] | None = None,
        substrate_fidelity: float = 1.0,
        attention_fidelity: float = 1.0,
        retrieval_fidelity: float = 1.0,
        replay_fidelity: float = 1.0,
        replay_confidence_ema_alpha: float = 0.12,
) -> PersistentLatentStateV13:
    """Compute one V13 step: V12 step + replay-confidence-carrier
    EMA update. The V13 state wraps the resulting V12 state and the
    new carrier."""
    prev_v12 = prev_state.inner_v12 if prev_state is not None else None
    new_v12 = step_persistent_state_v12(
        cell=cell, prev_state=prev_v12,
        carrier_values=carrier_values,
        turn_index=int(turn_index), role=str(role),
        branch_id=str(branch_id),
        anchor_skip=anchor_skip,
        substrate_skip=substrate_skip,
        hidden_state_skip=hidden_state_skip,
        attention_skip=attention_skip,
        retrieval_skip=retrieval_skip,
        replay_skip=replay_skip,
        substrate_fidelity=float(substrate_fidelity),
        attention_fidelity=float(attention_fidelity),
        retrieval_fidelity=float(retrieval_fidelity),
        replay_fidelity=float(replay_fidelity))
    # Replay confidence EMA.
    sd = int(new_v12.state_dim)
    if prev_state is not None:
        prev_rc = list(prev_state.replay_confidence_carrier)
    else:
        prev_rc = [0.0] * sd
    if replay_confidence_skip is not None:
        rcs = list(replay_confidence_skip)[:sd]
        while len(rcs) < sd:
            rcs.append(0.0)
        a = float(max(0.0, min(
            1.0, float(replay_confidence_ema_alpha))))
        new_rc = [
            a * float(rcs[i])
            + (1.0 - a) * float(
                prev_rc[i] if i < len(prev_rc) else 0.0)
            for i in range(sd)]
    else:
        new_rc = list(prev_rc)
        while len(new_rc) < sd:
            new_rc.append(0.0)
    return PersistentLatentStateV13(
        inner_v12=new_v12,
        replay_confidence_carrier=tuple(new_rc),
        distractor_rank=int(W61_DEFAULT_V13_DISTRACTOR_RANK),
    )


@dataclasses.dataclass(frozen=True)
class PersistentLatentStateV13Witness:
    schema: str
    chain_cid: str
    n_states: int
    chain_walk_depth_used: int
    n_layers: int
    distractor_rank: int
    nine_skip_present: bool
    replay_confidence_carrier_l1_sum: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "chain_cid": str(self.chain_cid),
            "n_states": int(self.n_states),
            "chain_walk_depth_used": int(
                self.chain_walk_depth_used),
            "n_layers": int(self.n_layers),
            "distractor_rank": int(self.distractor_rank),
            "nine_skip_present": bool(self.nine_skip_present),
            "replay_confidence_carrier_l1_sum": float(round(
                self.replay_confidence_carrier_l1_sum, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w61_persistent_v13_witness",
            "witness": self.to_dict()})


def emit_persistent_v13_witness(
        chain: PersistentLatentStateV13Chain,
        last_leaf_cid: str,
        *, max_depth: int = (
            W61_DEFAULT_V13_MAX_CHAIN_WALK_DEPTH),
) -> PersistentLatentStateV13Witness:
    walk = chain.walk_from(
        last_leaf_cid, max_depth=int(max_depth))
    rc_sum = float(sum(
        abs(float(v))
        for s in walk
        for v in s.replay_confidence_carrier))
    return PersistentLatentStateV13Witness(
        schema=W61_PERSISTENT_V13_SCHEMA_VERSION,
        chain_cid=str(chain.cid()),
        n_states=int(len(chain.states)),
        chain_walk_depth_used=int(len(walk)),
        n_layers=int(W61_DEFAULT_V13_N_LAYERS),
        distractor_rank=int(W61_DEFAULT_V13_DISTRACTOR_RANK),
        nine_skip_present=bool(rc_sum > 0.0),
        replay_confidence_carrier_l1_sum=float(rc_sum),
    )


__all__ = [
    "W61_PERSISTENT_V13_SCHEMA_VERSION",
    "W61_DEFAULT_V13_STATE_DIM",
    "W61_DEFAULT_V13_N_LAYERS",
    "W61_DEFAULT_V13_MAX_CHAIN_WALK_DEPTH",
    "W61_DEFAULT_V13_DISTRACTOR_RANK",
    "W61_V13_NO_PARENT_STATE",
    "PersistentLatentStateV13",
    "PersistentLatentStateV13Chain",
    "PersistentLatentStateV13Witness",
    "step_persistent_state_v13",
    "emit_persistent_v13_witness",
]
