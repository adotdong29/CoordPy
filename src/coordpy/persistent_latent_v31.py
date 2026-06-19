"""W79 — Persistent Latent State V31.

Strictly extends W78's ``coordpy.persistent_latent_v30``. V30 was
29 layers + twenty-seven skip carriers + ``max_chain_walk_depth=
33554432``. V31 adds:

* **30 layers** (vs V30's 29).
* **Twenty-eighth persistent skip-link** — a new replacement-then-
  restart-after-long-delay EMA carrier. This is the persistent
  latent the W79 long-horizon-reconstruction substrate V2 reads
  from on the new W79 regime.
* ``max_chain_walk_depth = 67108864`` (W79 doubles W78's cap).
* **Larger distractor basis** — V31 is **rank-30** (V30 was 29).

V31 strictly extends V30: with
``replacement_then_restart_after_long_delay_skip_v31 = None`` the
new EMA stays at the prior value and V31 reduces to V30 byte-for-
byte.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Sequence

from .persistent_latent_v30 import (
    PersistentLatentStateV30, W78_DEFAULT_V30_STATE_DIM,
    step_persistent_state_v30,
)


W79_PERSISTENT_V31_SCHEMA_VERSION: str = (
    "coordpy.persistent_latent_v31.v1")
W79_DEFAULT_V31_STATE_DIM: int = W78_DEFAULT_V30_STATE_DIM
W79_DEFAULT_V31_N_LAYERS: int = 30
W79_DEFAULT_V31_MAX_CHAIN_WALK_DEPTH: int = 67108864
W79_DEFAULT_V31_DISTRACTOR_RANK: int = 30
W79_V31_NO_PARENT_STATE: str = "no_parent_v31_state"


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass(frozen=True)
class PersistentLatentStateV31:
    inner_v30: PersistentLatentStateV30
    replacement_then_restart_after_long_delay_carrier: tuple[
        float, ...]
    distractor_rank: int

    @property
    def turn_index(self) -> int:
        return int(self.inner_v30.turn_index)

    @property
    def role(self) -> str:
        return str(self.inner_v30.role)

    @property
    def state_dim(self) -> int:
        return int(self.inner_v30.state_dim)

    @property
    def n_layers(self) -> int:
        return int(self.inner_v30.n_layers) + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W79_PERSISTENT_V31_SCHEMA_VERSION,
            "inner_v30_cid": str(self.inner_v30.cid()),
            "replacement_then_restart_after_long_delay_carrier": [
                float(round(float(x), 12))
                for x in
                self
                .replacement_then_restart_after_long_delay_carrier],
            "distractor_rank": int(self.distractor_rank),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w79_v31_persistent_state",
            "state": self.to_dict()})


@dataclasses.dataclass
class PersistentLatentStateV31Chain:
    states: dict[str, PersistentLatentStateV31] = (
        dataclasses.field(default_factory=dict))

    @classmethod
    def empty(cls) -> "PersistentLatentStateV31Chain":
        return cls(states={})

    def add(self, s: PersistentLatentStateV31) -> None:
        self.states[s.cid()] = s

    def get(
            self, cid: str) -> PersistentLatentStateV31 | None:
        return self.states.get(str(cid))

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w79_v31_state_chain",
            "members": [
                {"cid": c, "state": s.to_dict()}
                for c, s in sorted(self.states.items())],
        })


def step_persistent_state_v31(
        *,
        cell: Any,
        prev_state: PersistentLatentStateV31 | None,
        carrier_values: Sequence[float],
        turn_index: int, role: str,
        branch_id: str = "main",
        replacement_then_restart_after_long_delay_skip_v31: (
            Sequence[float] | None) = None,
        replacement_then_restart_after_long_delay_ema_alpha: (
            float) = 0.12,
        **kwargs: Any,
) -> PersistentLatentStateV31:
    prev_v30 = (
        prev_state.inner_v30 if prev_state is not None else None)
    new_v30 = step_persistent_state_v30(
        cell=cell, prev_state=prev_v30,
        carrier_values=list(carrier_values),
        turn_index=int(turn_index), role=str(role),
        branch_id=str(branch_id),
        **kwargs)
    sd = int(new_v30.state_dim)
    if prev_state is not None:
        prev_chain = list(
            prev_state
            .replacement_then_restart_after_long_delay_carrier)
    else:
        prev_chain = [0.0] * sd
    if (replacement_then_restart_after_long_delay_skip_v31
            is not None):
        chain = list(
            replacement_then_restart_after_long_delay_skip_v31)[
                :sd]
        while len(chain) < sd:
            chain.append(0.0)
        a = float(max(0.0, min(
            1.0,
            float(
                replacement_then_restart_after_long_delay_ema_alpha))))
        new_chain = [
            a * float(chain[i]) + (1.0 - a) * float(
                prev_chain[i] if i < len(prev_chain) else 0.0)
            for i in range(sd)]
    else:
        new_chain = list(prev_chain)
        while len(new_chain) < sd:
            new_chain.append(0.0)
    return PersistentLatentStateV31(
        inner_v30=new_v30,
        replacement_then_restart_after_long_delay_carrier=tuple(
            new_chain),
        distractor_rank=int(W79_DEFAULT_V31_DISTRACTOR_RANK),
    )


@dataclasses.dataclass(frozen=True)
class PersistentLatentStateV31Witness:
    schema: str
    chain_cid: str
    n_states: int
    n_layers: int
    distractor_rank: int
    twenty_eighth_skip_present: bool
    replacement_then_restart_after_long_delay_carrier_l1_sum: (
        float)

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w79_persistent_v31_witness",
            "schema": str(self.schema),
            "chain_cid": str(self.chain_cid),
            "n_states": int(self.n_states),
            "n_layers": int(self.n_layers),
            "distractor_rank": int(self.distractor_rank),
            "twenty_eighth_skip_present": bool(
                self.twenty_eighth_skip_present),
            "replacement_then_restart_after_long_delay_carrier_l1_sum": float(
                round(
                    self
                    .replacement_then_restart_after_long_delay_carrier_l1_sum,
                    12)),
        })


def emit_persistent_v31_witness(
        chain: PersistentLatentStateV31Chain,
) -> PersistentLatentStateV31Witness:
    chain_sum = float(sum(
        abs(float(v))
        for s in chain.states.values()
        for v in
        s.replacement_then_restart_after_long_delay_carrier))
    return PersistentLatentStateV31Witness(
        schema=W79_PERSISTENT_V31_SCHEMA_VERSION,
        chain_cid=str(chain.cid()),
        n_states=int(len(chain.states)),
        n_layers=int(W79_DEFAULT_V31_N_LAYERS),
        distractor_rank=int(W79_DEFAULT_V31_DISTRACTOR_RANK),
        twenty_eighth_skip_present=bool(chain_sum > 0.0),
        replacement_then_restart_after_long_delay_carrier_l1_sum=float(
            chain_sum),
    )


__all__ = [
    "W79_PERSISTENT_V31_SCHEMA_VERSION",
    "W79_DEFAULT_V31_STATE_DIM",
    "W79_DEFAULT_V31_N_LAYERS",
    "W79_DEFAULT_V31_MAX_CHAIN_WALK_DEPTH",
    "W79_DEFAULT_V31_DISTRACTOR_RANK",
    "W79_V31_NO_PARENT_STATE",
    "PersistentLatentStateV31",
    "PersistentLatentStateV31Chain",
    "PersistentLatentStateV31Witness",
    "step_persistent_state_v31",
    "emit_persistent_v31_witness",
]
