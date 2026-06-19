"""W57 M6 — Persistent Latent State V9.

Extends W56 V8 with:

* **7 layers** (vs V8's 6): adds one more GRU layer above V8.
* **Quintuple persistent skip-link** — V8 has (turn-0 anchor +
  fast EMA + slow EMA + substrate-conditioned EMA). V9 adds a
  fifth skip: a **hidden-state-conditioned** carrier derived from
  the substrate's *layer-l* hidden state (not just the final
  hidden state). When the W57 hidden-state bridge produces a
  measurable per-layer perturbation, the V9 layer reads it.
* **``max_chain_walk_depth = 384``** (vs V8's 256).
* **Substrate-fidelity weighting** — at each step, the
  substrate-skip and hidden-state-skip carriers are weighted by
  a per-turn substrate-fidelity scalar in ``[0, 1]``. When the
  substrate signal is weak (low fidelity), the skip is damped.

V9 strictly extends V8: when ``hidden_state_skip = None`` and
``substrate_fidelity = 1.0``, V9 reduces to V8 plus one extra
GRU layer with no contribution from hidden-state-skip (the new
linear is zero-initialised; that path produces a zero
contribution).

Honest scope:

* The V9 outer GRU + hidden-state-skip linear are *initialised
  but not trained*. ``W57-L-V9-OUTER-NOT-TRAINED-CAP`` documents
  this. The benchmark "V9 quint-skip beats V8 quad-skip" is a
  conjecture; we report it honestly without claiming strict
  dominance.
* Permutation invariance of V8 carriers carries forward. EMA
  carriers smooth out sequence order; V9's hidden-state skip
  carries the same property in this dimension.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError:  # pragma: no cover
    _np = None  # type: ignore

from .autograd_manifold import (
    ParamTensor,
    W47_DEFAULT_INIT_SCALE,
    W47_DEFAULT_TRAIN_SEED,
    _DeterministicLCG,
)
from .persistent_latent_v8 import (
    PersistentLatentStateV8,
    PersistentLatentStateV8Chain,
    V8StackedCell,
    W56_DEFAULT_V8_STATE_DIM,
    W56_DEFAULT_V8_N_LAYERS,
    W56_V8_NO_PARENT_STATE,
)
from .persistent_latent_v7 import (
    W55_DEFAULT_V7_INPUT_DIM,
    _round_floats,
    _stable_sigmoid,
)


W57_PERSISTENT_V9_SCHEMA_VERSION: str = (
    "coordpy.persistent_latent_v9.v1")
W57_DEFAULT_V9_STATE_DIM: int = W56_DEFAULT_V8_STATE_DIM
W57_DEFAULT_V9_N_LAYERS: int = 7
W57_DEFAULT_V9_MAX_CHAIN_WALK_DEPTH: int = 384
W57_V9_NO_PARENT_STATE: str = "no_parent_v9_state"


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass
class V9StackedCell:
    inner_v8: V8StackedCell
    w_z_top: ParamTensor
    b_z_top: ParamTensor
    w_h_top: ParamTensor
    b_h_top: ParamTensor
    w_hidden_skip: ParamTensor
    state_dim: int

    @classmethod
    def init(
            cls, *,
            state_dim: int = W57_DEFAULT_V9_STATE_DIM,
            input_dim: int = W55_DEFAULT_V7_INPUT_DIM,
            n_layers: int = W57_DEFAULT_V9_N_LAYERS,
            seed: int = W47_DEFAULT_TRAIN_SEED,
            init_scale: float = W47_DEFAULT_INIT_SCALE,
    ) -> "V9StackedCell":
        inner = V8StackedCell.init(
            state_dim=int(state_dim),
            input_dim=int(input_dim),
            n_layers=max(W56_DEFAULT_V8_N_LAYERS,
                          int(n_layers) - 1),
            seed=int(seed),
            init_scale=float(init_scale))
        rng = _DeterministicLCG(seed=int(seed) + 211)
        cat_d = int(state_dim) + int(state_dim)
        w_z = ParamTensor(
            shape=(int(state_dim), int(cat_d)), values=[])
        w_z.init_seed(
            seed=int(rng.next_uniform() * (1 << 30)),
            scale=float(init_scale))
        b_z = ParamTensor(
            shape=(int(state_dim),),
            values=[-1.0] * int(state_dim))
        w_h = ParamTensor(
            shape=(int(state_dim), int(cat_d)), values=[])
        w_h.init_seed(
            seed=int(rng.next_uniform() * (1 << 30)),
            scale=float(init_scale))
        b_h = ParamTensor(
            shape=(int(state_dim),),
            values=[0.0] * int(state_dim))
        w_hidden = ParamTensor(
            shape=(int(state_dim), int(state_dim)), values=[])
        vals = [0.0] * (int(state_dim) * int(state_dim))
        # Diagonal init so hidden-state skip starts informative.
        for i in range(int(state_dim)):
            vals[i * int(state_dim) + i] = 0.10
        w_hidden.values = vals
        return cls(
            inner_v8=inner,
            w_z_top=w_z, b_z_top=b_z,
            w_h_top=w_h, b_h_top=b_h,
            w_hidden_skip=w_hidden,
            state_dim=int(state_dim),
        )

    @property
    def n_layers(self) -> int:
        return int(self.inner_v8.n_layers) + 1

    def _hidden_project(
            self, hidden_skip: Sequence[float],
    ) -> list[float]:
        sd = int(self.state_dim)
        out = [0.0] * sd
        for i in range(sd):
            s = 0.0
            for j in range(sd):
                v = float(
                    hidden_skip[j]
                    if j < len(hidden_skip) else 0.0)
                s += float(
                    self.w_hidden_skip.values[i * sd + j]) * v
            out[i] = s
        return out

    def step_value(
            self, *,
            prev_layer_states: Sequence[Sequence[float]],
            input_x: Sequence[float],
            anchor_skip: Sequence[float] | None = None,
            fast_ema_skip: Sequence[float] | None = None,
            slow_ema_skip: Sequence[float] | None = None,
            substrate_skip: Sequence[float] | None = None,
            hidden_state_skip: Sequence[float] | None = None,
            substrate_fidelity: float = 1.0,
    ) -> tuple[list[list[float]], list[list[float]]]:
        # Damp substrate_skip and hidden_state_skip by fidelity.
        fid = float(max(0.0, min(1.0, float(substrate_fidelity))))
        damped_sub = (
            None if substrate_skip is None
            else [float(x) * fid for x in substrate_skip])
        damped_hidden = (
            None if hidden_state_skip is None
            else [float(x) * fid for x in hidden_state_skip])
        v8_layers, v8_gates = self.inner_v8.step_value(
            prev_layer_states=prev_layer_states[
                :self.inner_v8.n_layers],
            input_x=input_x,
            anchor_skip=anchor_skip,
            fast_ema_skip=fast_ema_skip,
            slow_ema_skip=slow_ema_skip,
            substrate_skip=damped_sub)
        sd = int(self.state_dim)
        top_below = (
            list(v8_layers[-1]) if v8_layers else [0.0] * sd)
        prev_top = (
            list(prev_layer_states[self.inner_v8.n_layers])
            if self.inner_v8.n_layers < len(prev_layer_states)
            else [0.0] * sd)
        hidden_proj = (
            self._hidden_project(damped_hidden)
            if damped_hidden is not None else [0.0] * sd)
        layer_input = [
            float(top_below[i] if i < len(top_below) else 0.0)
            + float(hidden_proj[i]
                    if i < len(hidden_proj) else 0.0)
            for i in range(sd)
        ]
        cat = [
            float(prev_top[i] if i < len(prev_top) else 0.0)
            for i in range(sd)
        ] + [
            float(layer_input[j] if j < len(layer_input) else 0.0)
            for j in range(sd)
        ]
        cat_d = 2 * sd
        wz = self.w_z_top.values
        bz = self.b_z_top.values
        wh = self.w_h_top.values
        bh = self.b_h_top.values
        z = [0.0] * sd
        h = [0.0] * sd
        for r in range(sd):
            base = r * cat_d
            sz = 0.0
            sh = 0.0
            for j in range(cat_d):
                cj = float(cat[j])
                sz += float(wz[base + j]) * cj
                sh += float(wh[base + j]) * cj
            sz += float(bz[r])
            sh += float(bh[r])
            z[r] = float(_stable_sigmoid(sz))
            h[r] = math.tanh(sh)
        top_next = [
            (1.0 - z[i]) * float(
                prev_top[i] if i < len(prev_top) else 0.0)
            + z[i] * h[i]
            for i in range(sd)
        ]
        next_layers = list(v8_layers) + [top_next]
        gates = list(v8_gates) + [z]
        return next_layers, gates

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": str(
                W57_PERSISTENT_V9_SCHEMA_VERSION),
            "inner_v8_cid": str(self.inner_v8.cid()),
            "state_dim": int(self.state_dim),
            "w_z_top": self.w_z_top.to_dict(),
            "b_z_top": self.b_z_top.to_dict(),
            "w_h_top": self.w_h_top.to_dict(),
            "b_h_top": self.b_h_top.to_dict(),
            "w_hidden_skip": self.w_hidden_skip.to_dict(),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w57_v9_stacked_cell",
            "cell": self.to_dict()})


@dataclasses.dataclass(frozen=True)
class PersistentLatentStateV9:
    turn_index: int
    role: str
    branch_id: str
    state_dim: int
    n_layers: int
    layer_states: tuple[tuple[float, ...], ...]
    fast_ema_carrier: tuple[float, ...]
    slow_ema_carrier: tuple[float, ...]
    substrate_carrier: tuple[float, ...]
    hidden_state_carrier: tuple[float, ...]
    anchor_carrier: tuple[float, ...]
    substrate_fidelity: float
    parent_state_cid: str
    cell_cid: str
    anchor_skip_cid: str
    fast_ema_skip_cid: str
    slow_ema_skip_cid: str
    substrate_skip_cid: str
    hidden_skip_cid: str
    update_gate_l1_sum: float
    is_merge: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_index": int(self.turn_index),
            "role": str(self.role),
            "branch_id": str(self.branch_id),
            "state_dim": int(self.state_dim),
            "n_layers": int(self.n_layers),
            "layer_states": [
                list(_round_floats(s))
                for s in self.layer_states],
            "fast_ema_carrier": list(_round_floats(
                self.fast_ema_carrier)),
            "slow_ema_carrier": list(_round_floats(
                self.slow_ema_carrier)),
            "substrate_carrier": list(_round_floats(
                self.substrate_carrier)),
            "hidden_state_carrier": list(_round_floats(
                self.hidden_state_carrier)),
            "anchor_carrier": list(_round_floats(
                self.anchor_carrier)),
            "substrate_fidelity": float(round(
                self.substrate_fidelity, 12)),
            "parent_state_cid": str(self.parent_state_cid),
            "cell_cid": str(self.cell_cid),
            "anchor_skip_cid": str(self.anchor_skip_cid),
            "fast_ema_skip_cid": str(self.fast_ema_skip_cid),
            "slow_ema_skip_cid": str(self.slow_ema_skip_cid),
            "substrate_skip_cid": str(self.substrate_skip_cid),
            "hidden_skip_cid": str(self.hidden_skip_cid),
            "update_gate_l1_sum": float(round(
                self.update_gate_l1_sum, 12)),
            "is_merge": bool(self.is_merge),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w57_v9_persistent_state",
            "state": self.to_dict()})

    @property
    def top_state(self) -> tuple[float, ...]:
        if not self.layer_states:
            return tuple([0.0] * int(self.state_dim))
        return tuple(self.layer_states[-1])


@dataclasses.dataclass
class PersistentLatentStateV9Chain:
    states: dict[str, PersistentLatentStateV9]

    @classmethod
    def empty(cls) -> "PersistentLatentStateV9Chain":
        return cls(states={})

    def add(self, state: PersistentLatentStateV9) -> None:
        self.states[state.cid()] = state

    def get(
            self, cid: str,
    ) -> PersistentLatentStateV9 | None:
        return self.states.get(str(cid))

    def walk_from(
            self, leaf_cid: str, *,
            max_depth: int = (
                W57_DEFAULT_V9_MAX_CHAIN_WALK_DEPTH),
    ) -> list[PersistentLatentStateV9]:
        out: list[PersistentLatentStateV9] = []
        cur = self.get(leaf_cid)
        seen: set[str] = set()
        steps = 0
        while cur is not None and steps < int(max_depth):
            out.append(cur)
            seen.add(cur.cid())
            parent = self.get(cur.parent_state_cid)
            if parent is None or parent.cid() in seen:
                break
            cur = parent
            steps += 1
        return out

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w57_v9_state_chain",
            "members": [
                {"cid": c, "state": s.to_dict()}
                for c, s in sorted(self.states.items())],
        })


def step_persistent_state_v9(
        *,
        cell: V9StackedCell,
        prev_state: PersistentLatentStateV9 | None,
        carrier_values: Sequence[float],
        turn_index: int,
        role: str,
        branch_id: str = "main",
        anchor_skip: Sequence[float] | None = None,
        substrate_skip: Sequence[float] | None = None,
        hidden_state_skip: Sequence[float] | None = None,
        substrate_fidelity: float = 1.0,
        fast_ema_alpha: float = 0.5,
        slow_ema_alpha: float = 0.10,
        substrate_ema_alpha: float = 0.25,
        hidden_ema_alpha: float = 0.20,
) -> PersistentLatentStateV9:
    sd = int(cell.state_dim)
    n_layers = int(cell.n_layers)
    if prev_state is None:
        prev_layers = [[0.0] * sd for _ in range(n_layers)]
        prev_fast_ema = [0.0] * sd
        prev_slow_ema = [0.0] * sd
        prev_substrate = [0.0] * sd
        prev_hidden = [0.0] * sd
        anchor = (
            list(anchor_skip)[:sd] if anchor_skip is not None
            else list(carrier_values)[:sd])
        while len(anchor) < sd:
            anchor.append(0.0)
        parent_cid = W57_V9_NO_PARENT_STATE
    else:
        prev_layers = [
            list(s) for s in prev_state.layer_states]
        while len(prev_layers) < n_layers:
            prev_layers.append([0.0] * sd)
        prev_fast_ema = list(prev_state.fast_ema_carrier)
        prev_slow_ema = list(prev_state.slow_ema_carrier)
        prev_substrate = list(prev_state.substrate_carrier)
        prev_hidden = list(prev_state.hidden_state_carrier)
        anchor = list(prev_state.anchor_carrier)
        parent_cid = prev_state.cid()
    fa = float(max(0.0, min(1.0, float(fast_ema_alpha))))
    sa = float(max(0.0, min(1.0, float(slow_ema_alpha))))
    ua = float(max(0.0, min(1.0, float(substrate_ema_alpha))))
    ha = float(max(0.0, min(1.0, float(hidden_ema_alpha))))
    fast_ema_next = [
        fa * float(
            carrier_values[i]
            if i < len(carrier_values) else 0.0)
        + (1.0 - fa) * float(
            prev_fast_ema[i]
            if i < len(prev_fast_ema) else 0.0)
        for i in range(sd)
    ]
    slow_ema_next = [
        sa * float(
            carrier_values[i]
            if i < len(carrier_values) else 0.0)
        + (1.0 - sa) * float(
            prev_slow_ema[i]
            if i < len(prev_slow_ema) else 0.0)
        for i in range(sd)
    ]
    substrate_next = list(prev_substrate)
    if substrate_skip is not None:
        substrate_next = [
            ua * float(
                substrate_skip[i]
                if i < len(substrate_skip) else 0.0)
            + (1.0 - ua) * float(
                prev_substrate[i]
                if i < len(prev_substrate) else 0.0)
            for i in range(sd)
        ]
    hidden_next = list(prev_hidden)
    if hidden_state_skip is not None:
        hidden_next = [
            ha * float(
                hidden_state_skip[i]
                if i < len(hidden_state_skip) else 0.0)
            + (1.0 - ha) * float(
                prev_hidden[i]
                if i < len(prev_hidden) else 0.0)
            for i in range(sd)
        ]
    next_layers, gates = cell.step_value(
        prev_layer_states=prev_layers,
        input_x=carrier_values,
        anchor_skip=anchor,
        fast_ema_skip=fast_ema_next,
        slow_ema_skip=slow_ema_next,
        substrate_skip=substrate_next,
        hidden_state_skip=hidden_next,
        substrate_fidelity=substrate_fidelity)
    anchor_cid = _sha256_hex({
        "kind": "w57_v9_anchor_skip",
        "values": _round_floats(anchor),
    })
    fast_ema_cid = _sha256_hex({
        "kind": "w57_v9_fast_ema_skip",
        "values": _round_floats(fast_ema_next),
        "turn_index": int(turn_index),
    })
    slow_ema_cid = _sha256_hex({
        "kind": "w57_v9_slow_ema_skip",
        "values": _round_floats(slow_ema_next),
        "turn_index": int(turn_index),
    })
    substrate_cid = _sha256_hex({
        "kind": "w57_v9_substrate_skip",
        "values": _round_floats(substrate_next),
        "turn_index": int(turn_index),
    })
    hidden_cid = _sha256_hex({
        "kind": "w57_v9_hidden_skip",
        "values": _round_floats(hidden_next),
        "turn_index": int(turn_index),
    })
    gate_l1_sum = float(
        sum(abs(float(g))
            for layer_z in gates for g in layer_z))
    return PersistentLatentStateV9(
        turn_index=int(turn_index),
        role=str(role),
        branch_id=str(branch_id),
        state_dim=int(sd),
        n_layers=int(n_layers),
        layer_states=tuple(
            tuple(_round_floats(s)) for s in next_layers),
        fast_ema_carrier=tuple(_round_floats(fast_ema_next)),
        slow_ema_carrier=tuple(_round_floats(slow_ema_next)),
        substrate_carrier=tuple(_round_floats(substrate_next)),
        hidden_state_carrier=tuple(_round_floats(hidden_next)),
        anchor_carrier=tuple(_round_floats(anchor)),
        substrate_fidelity=float(substrate_fidelity),
        parent_state_cid=str(parent_cid),
        cell_cid=str(cell.cid()),
        anchor_skip_cid=str(anchor_cid),
        fast_ema_skip_cid=str(fast_ema_cid),
        slow_ema_skip_cid=str(slow_ema_cid),
        substrate_skip_cid=str(substrate_cid),
        hidden_skip_cid=str(hidden_cid),
        update_gate_l1_sum=float(round(gate_l1_sum, 12)),
        is_merge=False,
    )


@dataclasses.dataclass(frozen=True)
class PersistentLatentStateV9Witness:
    schema: str
    cell_cid: str
    n_layers: int
    chain_walk_depth: int
    state_cid: str
    substrate_skip_cid: str
    hidden_skip_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "cell_cid": str(self.cell_cid),
            "n_layers": int(self.n_layers),
            "chain_walk_depth": int(self.chain_walk_depth),
            "state_cid": str(self.state_cid),
            "substrate_skip_cid": str(self.substrate_skip_cid),
            "hidden_skip_cid": str(self.hidden_skip_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w57_v9_witness",
            "witness": self.to_dict(),
        })


def emit_persistent_v9_witness(
        *,
        cell: V9StackedCell,
        state: PersistentLatentStateV9,
        chain: PersistentLatentStateV9Chain | None = None,
) -> PersistentLatentStateV9Witness:
    chain_depth = 0
    if chain is not None:
        chain_depth = len(chain.walk_from(state.cid()))
    return PersistentLatentStateV9Witness(
        schema=W57_PERSISTENT_V9_SCHEMA_VERSION,
        cell_cid=str(cell.cid()),
        n_layers=int(cell.n_layers),
        chain_walk_depth=int(chain_depth),
        state_cid=str(state.cid()),
        substrate_skip_cid=str(state.substrate_skip_cid),
        hidden_skip_cid=str(state.hidden_skip_cid),
    )


__all__ = [
    "W57_PERSISTENT_V9_SCHEMA_VERSION",
    "W57_DEFAULT_V9_STATE_DIM",
    "W57_DEFAULT_V9_N_LAYERS",
    "W57_DEFAULT_V9_MAX_CHAIN_WALK_DEPTH",
    "W57_V9_NO_PARENT_STATE",
    "V9StackedCell",
    "PersistentLatentStateV9",
    "PersistentLatentStateV9Chain",
    "PersistentLatentStateV9Witness",
    "step_persistent_state_v9",
    "emit_persistent_v9_witness",
]
