"""W56 M4 — Persistent Latent State V8.

Extends W55 V7 with:

* **6 layers** (vs V7's 5): adds one more GRU layer above V7.
* **Quad persistent skip-link** — V7 has (turn-0 anchor, fast
  EMA, slow EMA); V8 adds a *substrate-conditioned* skip
  carrier derived from the tiny substrate's hidden state at a
  reference turn. All four skip-links feed the V8 top layer's
  input. Honest scope: the substrate carrier is **zero** unless
  the W56 loop explicitly provides one via the optional
  ``substrate_skip`` parameter. When zero, V8 reduces to V7 plus
  one extra GRU layer; the H8 quad-skip gain measures the gain
  with substrate signal *non-zero*.
* **`max_chain_walk_depth = 256`** (vs V7's 128).

V8 inherits V7's autograd surface. It does NOT train the outer
layer end-to-end (carry-forward of W55-L-V7-OUTER-NOT-TRAINED-CAP)
and reports the same honest soundness bars.

The substrate-conditioned carrier is honest, not magical: it is
just the substrate's final hidden state at some prior turn,
projected into the V8 state dim by a deterministic-seeded linear
map. The injection makes V8 a function of the substrate state
without claiming the substrate is itself optimised.
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
from .persistent_latent_v7 import (
    PersistentLatentStateV7,
    PersistentLatentStateV7Chain,
    V7StackedCell,
    W55_DEFAULT_V7_INPUT_DIM,
    W55_DEFAULT_V7_N_LAYERS,
    W55_DEFAULT_V7_STATE_DIM,
    W55_V7_NO_PARENT_STATE,
    _round_floats,
    _stable_sigmoid,
)


W56_PERSISTENT_V8_SCHEMA_VERSION: str = (
    "coordpy.persistent_latent_v8.v1")
W56_DEFAULT_V8_STATE_DIM: int = W55_DEFAULT_V7_STATE_DIM
W56_DEFAULT_V8_N_LAYERS: int = 6
W56_DEFAULT_V8_MAX_CHAIN_WALK_DEPTH: int = 256
W56_DEFAULT_V8_SUBSTRATE_PROJ_SEED: int = 56081234
W56_V8_NO_PARENT_STATE: str = "no_parent_v8_state"


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass
class V8StackedCell:
    """V8 = V7 cell + extra top GRU + substrate-conditioned skip."""

    inner_v7: V7StackedCell
    w_z_top: ParamTensor
    b_z_top: ParamTensor
    w_h_top: ParamTensor
    b_h_top: ParamTensor
    w_substrate_skip: ParamTensor
    state_dim: int
    substrate_proj_seed: int

    @classmethod
    def init(
            cls, *,
            state_dim: int = W56_DEFAULT_V8_STATE_DIM,
            input_dim: int = W55_DEFAULT_V7_INPUT_DIM,
            n_layers: int = W56_DEFAULT_V8_N_LAYERS,
            seed: int = W47_DEFAULT_TRAIN_SEED,
            init_scale: float = W47_DEFAULT_INIT_SCALE,
    ) -> "V8StackedCell":
        inner = V7StackedCell.init(
            state_dim=int(state_dim),
            input_dim=int(input_dim),
            n_layers=max(W55_DEFAULT_V7_N_LAYERS,
                          int(n_layers) - 1),
            seed=int(seed),
            init_scale=float(init_scale))
        rng = _DeterministicLCG(seed=int(seed) + 113)
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
        w_sub = ParamTensor(
            shape=(int(state_dim), int(state_dim)), values=[])
        vals = [0.0] * (int(state_dim) * int(state_dim))
        # diagonal-ish init so substrate skip is informative
        for i in range(int(state_dim)):
            vals[i * int(state_dim) + i] = 0.10
        w_sub.values = vals
        return cls(
            inner_v7=inner,
            w_z_top=w_z, b_z_top=b_z,
            w_h_top=w_h, b_h_top=b_h,
            w_substrate_skip=w_sub,
            state_dim=int(state_dim),
            substrate_proj_seed=int(seed),
        )

    @property
    def n_layers(self) -> int:
        return int(self.inner_v7.n_layers) + 1

    def _substrate_project(
            self, substrate_skip: Sequence[float],
    ) -> list[float]:
        sd = int(self.state_dim)
        out = [0.0] * sd
        for i in range(sd):
            s = 0.0
            for j in range(sd):
                sj = float(
                    substrate_skip[j]
                    if j < len(substrate_skip) else 0.0)
                s += float(
                    self.w_substrate_skip.values[i * sd + j]) * sj
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
    ) -> tuple[list[list[float]], list[list[float]]]:
        v7_layers, v7_gates = self.inner_v7.step_value(
            prev_layer_states=prev_layer_states[
                :self.inner_v7.n_layers],
            input_x=input_x,
            anchor_skip=anchor_skip,
            fast_ema_skip=fast_ema_skip,
            slow_ema_skip=slow_ema_skip)
        sd = int(self.state_dim)
        top_below = (
            list(v7_layers[-1]) if v7_layers else [0.0] * sd)
        prev_top = (
            list(prev_layer_states[self.inner_v7.n_layers])
            if self.inner_v7.n_layers < len(prev_layer_states)
            else [0.0] * sd)
        substrate_proj = (
            self._substrate_project(substrate_skip)
            if substrate_skip is not None else [0.0] * sd)
        layer_input = [
            float(top_below[i] if i < len(top_below) else 0.0)
            + float(substrate_proj[i]
                    if i < len(substrate_proj) else 0.0)
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
        next_layers = list(v7_layers) + [top_next]
        gates = list(v7_gates) + [z]
        return next_layers, gates

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": str(
                W56_PERSISTENT_V8_SCHEMA_VERSION),
            "inner_v7_cid": str(self.inner_v7.cid()),
            "state_dim": int(self.state_dim),
            "substrate_proj_seed": int(self.substrate_proj_seed),
            "w_z_top": self.w_z_top.to_dict(),
            "b_z_top": self.b_z_top.to_dict(),
            "w_h_top": self.w_h_top.to_dict(),
            "b_h_top": self.b_h_top.to_dict(),
            "w_substrate_skip": self.w_substrate_skip.to_dict(),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w56_v8_stacked_cell",
            "cell": self.to_dict()})


@dataclasses.dataclass(frozen=True)
class PersistentLatentStateV8:
    turn_index: int
    role: str
    branch_id: str
    state_dim: int
    n_layers: int
    layer_states: tuple[tuple[float, ...], ...]
    fast_ema_carrier: tuple[float, ...]
    slow_ema_carrier: tuple[float, ...]
    substrate_carrier: tuple[float, ...]
    anchor_carrier: tuple[float, ...]
    parent_state_cid: str
    cell_cid: str
    anchor_skip_cid: str
    fast_ema_skip_cid: str
    slow_ema_skip_cid: str
    substrate_skip_cid: str
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
            "anchor_carrier": list(_round_floats(
                self.anchor_carrier)),
            "parent_state_cid": str(self.parent_state_cid),
            "cell_cid": str(self.cell_cid),
            "anchor_skip_cid": str(self.anchor_skip_cid),
            "fast_ema_skip_cid": str(self.fast_ema_skip_cid),
            "slow_ema_skip_cid": str(self.slow_ema_skip_cid),
            "substrate_skip_cid": str(self.substrate_skip_cid),
            "update_gate_l1_sum": float(round(
                self.update_gate_l1_sum, 12)),
            "is_merge": bool(self.is_merge),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w56_v8_persistent_state",
            "state": self.to_dict()})

    @property
    def top_state(self) -> tuple[float, ...]:
        if not self.layer_states:
            return tuple([0.0] * int(self.state_dim))
        return tuple(self.layer_states[-1])


@dataclasses.dataclass
class PersistentLatentStateV8Chain:
    states: dict[str, PersistentLatentStateV8]

    @classmethod
    def empty(cls) -> "PersistentLatentStateV8Chain":
        return cls(states={})

    def add(self, state: PersistentLatentStateV8) -> None:
        self.states[state.cid()] = state

    def get(
            self, cid: str,
    ) -> PersistentLatentStateV8 | None:
        return self.states.get(str(cid))

    def walk_from(
            self, leaf_cid: str, *,
            max_depth: int = (
                W56_DEFAULT_V8_MAX_CHAIN_WALK_DEPTH),
    ) -> list[PersistentLatentStateV8]:
        out: list[PersistentLatentStateV8] = []
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
            "kind": "w56_v8_state_chain",
            "members": [
                {"cid": c, "state": s.to_dict()}
                for c, s in sorted(self.states.items())],
        })


def step_persistent_state_v8(
        *,
        cell: V8StackedCell,
        prev_state: PersistentLatentStateV8 | None,
        carrier_values: Sequence[float],
        turn_index: int,
        role: str,
        branch_id: str = "main",
        anchor_skip: Sequence[float] | None = None,
        substrate_skip: Sequence[float] | None = None,
        fast_ema_alpha: float = 0.5,
        slow_ema_alpha: float = 0.10,
        substrate_ema_alpha: float = 0.25,
) -> PersistentLatentStateV8:
    sd = int(cell.state_dim)
    n_layers = int(cell.n_layers)
    if prev_state is None:
        prev_layers = [[0.0] * sd for _ in range(n_layers)]
        prev_fast_ema = [0.0] * sd
        prev_slow_ema = [0.0] * sd
        prev_substrate = [0.0] * sd
        anchor = (
            list(anchor_skip)[:sd] if anchor_skip is not None
            else list(carrier_values)[:sd])
        while len(anchor) < sd:
            anchor.append(0.0)
        parent_cid = W56_V8_NO_PARENT_STATE
    else:
        prev_layers = [
            list(s) for s in prev_state.layer_states]
        while len(prev_layers) < n_layers:
            prev_layers.append([0.0] * sd)
        prev_fast_ema = list(prev_state.fast_ema_carrier)
        prev_slow_ema = list(prev_state.slow_ema_carrier)
        prev_substrate = list(prev_state.substrate_carrier)
        anchor = list(prev_state.anchor_carrier)
        parent_cid = prev_state.cid()
    fa = float(max(0.0, min(1.0, float(fast_ema_alpha))))
    sa = float(max(0.0, min(1.0, float(slow_ema_alpha))))
    ua = float(max(0.0, min(1.0, float(substrate_ema_alpha))))
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
    if substrate_skip is None:
        substrate_next = list(prev_substrate)
    else:
        substrate_next = [
            ua * float(
                substrate_skip[i]
                if i < len(substrate_skip) else 0.0)
            + (1.0 - ua) * float(
                prev_substrate[i]
                if i < len(prev_substrate) else 0.0)
            for i in range(sd)
        ]
    next_layers, gates = cell.step_value(
        prev_layer_states=prev_layers,
        input_x=carrier_values,
        anchor_skip=anchor,
        fast_ema_skip=fast_ema_next,
        slow_ema_skip=slow_ema_next,
        substrate_skip=substrate_next)
    anchor_cid = _sha256_hex({
        "kind": "w56_v8_anchor_skip",
        "values": _round_floats(anchor),
    })
    fast_ema_cid = _sha256_hex({
        "kind": "w56_v8_fast_ema_skip",
        "values": _round_floats(fast_ema_next),
        "turn_index": int(turn_index),
    })
    slow_ema_cid = _sha256_hex({
        "kind": "w56_v8_slow_ema_skip",
        "values": _round_floats(slow_ema_next),
        "turn_index": int(turn_index),
    })
    substrate_cid = _sha256_hex({
        "kind": "w56_v8_substrate_skip",
        "values": _round_floats(substrate_next),
        "turn_index": int(turn_index),
    })
    gate_l1_sum = float(
        sum(abs(float(g))
            for layer_z in gates for g in layer_z))
    return PersistentLatentStateV8(
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
        anchor_carrier=tuple(_round_floats(anchor)),
        parent_state_cid=str(parent_cid),
        cell_cid=str(cell.cid()),
        anchor_skip_cid=str(anchor_cid),
        fast_ema_skip_cid=str(fast_ema_cid),
        slow_ema_skip_cid=str(slow_ema_cid),
        substrate_skip_cid=str(substrate_cid),
        update_gate_l1_sum=float(round(gate_l1_sum, 12)),
        is_merge=False,
    )


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for i in range(n):
        ai = float(a[i])
        bi = float(b[i])
        dot += ai * bi
        na += ai * ai
        nb += bi * bi
    if na <= 1e-30 or nb <= 1e-30:
        return 0.0
    return float(dot / (math.sqrt(na) * math.sqrt(nb)))


def evaluate_v8_long_horizon_recall(
        cell: V8StackedCell,
        sequences: Sequence[Sequence[Sequence[float]]],
        targets: Sequence[Sequence[float]],
        *,
        substrate_skips: Sequence[Sequence[float]] | None = None,
) -> float:
    """Mean cosine recall using quad-skip step."""
    if not sequences:
        return 0.0
    cos_sum = 0.0
    n = 0
    sd = int(cell.state_dim)
    for idx, (seq, tgt) in enumerate(zip(sequences, targets)):
        layer_states = [
            [0.0] * sd for _ in range(cell.n_layers)]
        anchor = (
            list(seq[0])[:sd] if len(seq) > 0
            else [0.0] * sd)
        while len(anchor) < sd:
            anchor.append(0.0)
        fast_ema = [0.0] * sd
        slow_ema = [0.0] * sd
        substrate = [0.0] * sd
        if substrate_skips is not None and idx < len(substrate_skips):
            substrate = list(substrate_skips[idx])[:sd]
            while len(substrate) < sd:
                substrate.append(0.0)
        for x in seq:
            xv = list(x)[:sd]
            while len(xv) < sd:
                xv.append(0.0)
            fast_ema = [
                0.5 * xv[i] + 0.5 * fast_ema[i]
                for i in range(sd)]
            slow_ema = [
                0.1 * xv[i] + 0.9 * slow_ema[i]
                for i in range(sd)]
            layer_states, _ = cell.step_value(
                prev_layer_states=layer_states,
                input_x=xv,
                anchor_skip=anchor,
                fast_ema_skip=fast_ema,
                slow_ema_skip=slow_ema,
                substrate_skip=substrate)
        top = layer_states[-1] if layer_states else [0.0] * sd
        cos_sum += _cosine(top, tgt)
        n += 1
    if n == 0:
        return 0.0
    return float(cos_sum / float(n))


@dataclasses.dataclass(frozen=True)
class PersistentLatentStateV8Witness:
    """W56 V8 witness."""

    schema: str
    cell_cid: str
    n_layers: int
    chain_walk_depth: int
    state_cid: str
    substrate_skip_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "cell_cid": str(self.cell_cid),
            "n_layers": int(self.n_layers),
            "chain_walk_depth": int(self.chain_walk_depth),
            "state_cid": str(self.state_cid),
            "substrate_skip_cid": str(self.substrate_skip_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w56_v8_witness",
            "witness": self.to_dict(),
        })


def emit_persistent_v8_witness(
        *,
        cell: V8StackedCell,
        state: PersistentLatentStateV8,
        chain: PersistentLatentStateV8Chain | None = None,
) -> PersistentLatentStateV8Witness:
    chain_depth = 0
    if chain is not None:
        chain_depth = len(chain.walk_from(state.cid()))
    return PersistentLatentStateV8Witness(
        schema=W56_PERSISTENT_V8_SCHEMA_VERSION,
        cell_cid=str(cell.cid()),
        n_layers=int(cell.n_layers),
        chain_walk_depth=int(chain_depth),
        state_cid=str(state.cid()),
        substrate_skip_cid=str(state.substrate_skip_cid),
    )


__all__ = [
    "W56_PERSISTENT_V8_SCHEMA_VERSION",
    "W56_DEFAULT_V8_STATE_DIM",
    "W56_DEFAULT_V8_N_LAYERS",
    "W56_DEFAULT_V8_MAX_CHAIN_WALK_DEPTH",
    "W56_V8_NO_PARENT_STATE",
    "V8StackedCell",
    "PersistentLatentStateV8",
    "PersistentLatentStateV8Chain",
    "PersistentLatentStateV8Witness",
    "step_persistent_state_v8",
    "evaluate_v8_long_horizon_recall",
    "emit_persistent_v8_witness",
]
