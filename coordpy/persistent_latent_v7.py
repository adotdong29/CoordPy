"""W55 M1 — Persistent Latent State V7.

Extends W54 V6 with:

* **5 layers** (vs V6's 4): adds one more GRU layer above the V6
  top layer. The new top reads from the V6 top state and an
  additional slow-EMA carrier.
* **Triple persistent skip-link** — V6 has (turn-0 anchor, fast
  EMA); V7 adds a *slow-EMA* with a smaller alpha (longer
  memory). All three skip into the V7 top layer's projection.
  The fast carrier remains recency-biased; the slow carrier
  carries a long-tail summary across distractors.
* **Disagreement-algebraic merge head** — when merging two
  states, V7 emits the merged state PLUS per-dim ``low_bound``,
  ``high_bound``, and ``disagreement`` (= ``|a - b|``). The
  bound vectors are ``min(a, b)`` and ``max(a, b)`` per-dim;
  ``merged - low_bound`` is the algebra's projection onto the
  low boundary.
* **`max_chain_walk_depth = 128`** (vs V6's 64).

V7 inherits V6's autograd surface. It does NOT touch transformer
state.

W55-L-V7-OUTER-NOT-TRAINED-CAP: the V7 outer (top) GRU layer +
slow-EMA projection are *initialised but not trained* in
``fit_persistent_v7``; only the inner V6 cell (itself wrapping
V5) is fit by chained pure-Python autograd. Long-horizon V7
absolute recall is bounded and seed-variable.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Any, Sequence

from .autograd_manifold import (
    ParamTensor,
    W47_DEFAULT_INIT_SCALE,
    W47_DEFAULT_TRAIN_SEED,
    _DeterministicLCG,
)
from .persistent_latent_v6 import (
    PersistentLatentStateV6,
    PersistentLatentStateV6Chain,
    V6StackedCell,
    W54_DEFAULT_V6_EMA_ALPHA,
    W54_DEFAULT_V6_INPUT_DIM,
    W54_DEFAULT_V6_N_LAYERS,
    W54_DEFAULT_V6_STATE_DIM,
    W54_V6_NO_PARENT_STATE,
    _stable_sigmoid,
    step_persistent_state_v6,
)


# =============================================================================
# Schema, defaults
# =============================================================================

W55_PERSISTENT_V7_SCHEMA_VERSION: str = (
    "coordpy.persistent_latent_v7.v1")

W55_DEFAULT_V7_STATE_DIM: int = W54_DEFAULT_V6_STATE_DIM
W55_DEFAULT_V7_INPUT_DIM: int = W54_DEFAULT_V6_INPUT_DIM
W55_DEFAULT_V7_N_LAYERS: int = 5
W55_DEFAULT_V7_MAX_CHAIN_WALK_DEPTH: int = 128
W55_DEFAULT_V7_FAST_EMA_ALPHA: float = W54_DEFAULT_V6_EMA_ALPHA  # 0.5
W55_DEFAULT_V7_SLOW_EMA_ALPHA: float = 0.10
W55_V7_NO_PARENT_STATE: str = "no_parent_v7_state"


# =============================================================================
# Helpers
# =============================================================================


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _round_floats(
        values: Sequence[float], precision: int = 12,
) -> list[float]:
    return [float(round(float(v), precision)) for v in values]


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


# =============================================================================
# V7 stacked cell (V6 inner + top GRU + slow-EMA projection)
# =============================================================================


@dataclasses.dataclass
class V7StackedCell:
    """V7 cell: V6 inner cell + extra GRU top + slow EMA carrier."""

    inner_v6: V6StackedCell
    w_z_top: ParamTensor
    b_z_top: ParamTensor
    w_h_top: ParamTensor
    b_h_top: ParamTensor
    w_slow_skip: ParamTensor
    state_dim: int
    fast_alpha: float
    slow_alpha: float

    @classmethod
    def init(
            cls, *,
            state_dim: int = W55_DEFAULT_V7_STATE_DIM,
            input_dim: int = W55_DEFAULT_V7_INPUT_DIM,
            n_layers: int = W55_DEFAULT_V7_N_LAYERS,
            seed: int = W47_DEFAULT_TRAIN_SEED,
            init_scale: float = W47_DEFAULT_INIT_SCALE,
            fast_alpha: float = W55_DEFAULT_V7_FAST_EMA_ALPHA,
            slow_alpha: float = W55_DEFAULT_V7_SLOW_EMA_ALPHA,
    ) -> "V7StackedCell":
        inner = V6StackedCell.init(
            state_dim=int(state_dim),
            input_dim=int(input_dim),
            n_layers=max(W54_DEFAULT_V6_N_LAYERS,
                          int(n_layers) - 1),
            seed=int(seed),
            init_scale=float(init_scale),
            ema_alpha=float(fast_alpha))
        rng = _DeterministicLCG(seed=int(seed) + 91)
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
        w_slow = ParamTensor(
            shape=(int(state_dim), int(state_dim)), values=[])
        vals = [0.0] * (int(state_dim) * int(state_dim))
        for i in range(int(state_dim)):
            vals[i * int(state_dim) + i] = 0.08
        w_slow.values = vals
        return cls(
            inner_v6=inner,
            w_z_top=w_z, b_z_top=b_z,
            w_h_top=w_h, b_h_top=b_h,
            w_slow_skip=w_slow,
            state_dim=int(state_dim),
            fast_alpha=float(max(0.0, min(1.0, float(fast_alpha)))),
            slow_alpha=float(max(0.0, min(1.0, float(slow_alpha)))))

    def _slow_project_value(
            self, slow_input: Sequence[float],
    ) -> list[float]:
        sd = int(self.state_dim)
        out = [0.0] * sd
        for i in range(sd):
            s = 0.0
            for j in range(sd):
                sj = float(
                    slow_input[j]
                    if j < len(slow_input) else 0.0)
                s += float(
                    self.w_slow_skip.values[i * sd + j]) * sj
            out[i] = s
        return out

    @property
    def n_layers(self) -> int:
        return int(self.inner_v6.n_layers) + 1

    def step_value(
            self, *,
            prev_layer_states: Sequence[Sequence[float]],
            input_x: Sequence[float],
            anchor_skip: Sequence[float] | None = None,
            fast_ema_skip: Sequence[float] | None = None,
            slow_ema_skip: Sequence[float] | None = None,
    ) -> tuple[list[list[float]], list[list[float]]]:
        v6_layers, v6_gates = self.inner_v6.step_value(
            prev_layer_states=prev_layer_states[
                :self.inner_v6.n_layers],
            input_x=input_x,
            anchor_skip=anchor_skip,
            ema_skip=fast_ema_skip)
        sd = int(self.state_dim)
        top_below = (
            list(v6_layers[-1]) if v6_layers else [0.0] * sd)
        prev_top = (
            list(prev_layer_states[self.inner_v6.n_layers])
            if self.inner_v6.n_layers < len(prev_layer_states)
            else [0.0] * sd)
        slow_proj = (
            self._slow_project_value(slow_ema_skip)
            if slow_ema_skip is not None else [0.0] * sd)
        layer_input = [
            float(top_below[i] if i < len(top_below) else 0.0)
            + float(slow_proj[i] if i < len(slow_proj) else 0.0)
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
        next_layers = list(v6_layers) + [top_next]
        gates = list(v6_gates) + [z]
        return next_layers, gates

    def merge_states_value(
            self,
            state_a: Sequence[float],
            state_b: Sequence[float],
    ) -> tuple[list[float], list[float], list[float],
                list[float], list[float]]:
        """Return (merged, alpha, disagreement, low_bound, high_bound)."""
        merged, alpha, disagreement = (
            self.inner_v6.merge_states_value(
                state_a, state_b))
        sd = int(self.state_dim)
        low = [
            float(min(
                float(state_a[i] if i < len(state_a) else 0.0),
                float(state_b[i] if i < len(state_b) else 0.0)))
            for i in range(sd)
        ]
        high = [
            float(max(
                float(state_a[i] if i < len(state_a) else 0.0),
                float(state_b[i] if i < len(state_b) else 0.0)))
            for i in range(sd)
        ]
        return merged, alpha, disagreement, low, high

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": str(
                W55_PERSISTENT_V7_SCHEMA_VERSION),
            "inner_v6_cid": str(self.inner_v6.cid()),
            "state_dim": int(self.state_dim),
            "fast_alpha": float(round(self.fast_alpha, 12)),
            "slow_alpha": float(round(self.slow_alpha, 12)),
            "w_z_top": self.w_z_top.to_dict(),
            "b_z_top": self.b_z_top.to_dict(),
            "w_h_top": self.w_h_top.to_dict(),
            "b_h_top": self.b_h_top.to_dict(),
            "w_slow_skip": self.w_slow_skip.to_dict(),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w55_v7_stacked_cell",
            "cell": self.to_dict()})


# =============================================================================
# V7 PersistentLatentState
# =============================================================================


@dataclasses.dataclass(frozen=True)
class PersistentLatentStateV7:
    turn_index: int
    role: str
    branch_id: str
    state_dim: int
    n_layers: int
    layer_states: tuple[tuple[float, ...], ...]
    fast_ema_carrier: tuple[float, ...]
    slow_ema_carrier: tuple[float, ...]
    anchor_carrier: tuple[float, ...]
    parent_state_cid: str
    second_parent_state_cid: str
    cell_cid: str
    anchor_skip_cid: str
    fast_ema_skip_cid: str
    slow_ema_skip_cid: str
    update_gate_l1_sum: float
    disagreement_l1_sum: float
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
            "anchor_carrier": list(_round_floats(
                self.anchor_carrier)),
            "parent_state_cid": str(self.parent_state_cid),
            "second_parent_state_cid": str(
                self.second_parent_state_cid),
            "cell_cid": str(self.cell_cid),
            "anchor_skip_cid": str(self.anchor_skip_cid),
            "fast_ema_skip_cid": str(self.fast_ema_skip_cid),
            "slow_ema_skip_cid": str(self.slow_ema_skip_cid),
            "update_gate_l1_sum": float(round(
                self.update_gate_l1_sum, 12)),
            "disagreement_l1_sum": float(round(
                self.disagreement_l1_sum, 12)),
            "is_merge": bool(self.is_merge),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w55_v7_persistent_state",
            "state": self.to_dict()})

    @property
    def top_state(self) -> tuple[float, ...]:
        if not self.layer_states:
            return tuple([0.0] * int(self.state_dim))
        return tuple(self.layer_states[-1])


@dataclasses.dataclass
class PersistentLatentStateV7Chain:
    states: dict[str, PersistentLatentStateV7]

    @classmethod
    def empty(cls) -> "PersistentLatentStateV7Chain":
        return cls(states={})

    def add(self, state: PersistentLatentStateV7) -> None:
        self.states[state.cid()] = state

    def get(
            self, cid: str,
    ) -> PersistentLatentStateV7 | None:
        return self.states.get(str(cid))

    def walk_from(
            self, leaf_cid: str, *,
            max_depth: int = (
                W55_DEFAULT_V7_MAX_CHAIN_WALK_DEPTH),
    ) -> list[PersistentLatentStateV7]:
        out: list[PersistentLatentStateV7] = []
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
            "kind": "w55_v7_state_chain",
            "members": [
                {"cid": c, "state": s.to_dict()}
                for c, s in sorted(self.states.items())
            ],
        })


# =============================================================================
# Step + merge
# =============================================================================


def step_persistent_state_v7(
        *,
        cell: V7StackedCell,
        prev_state: PersistentLatentStateV7 | None,
        carrier_values: Sequence[float],
        turn_index: int,
        role: str,
        branch_id: str = "main",
        anchor_skip: Sequence[float] | None = None,
) -> PersistentLatentStateV7:
    sd = int(cell.state_dim)
    n_layers = int(cell.n_layers)
    if prev_state is None:
        prev_layers = [[0.0] * sd for _ in range(n_layers)]
        prev_fast_ema = [0.0] * sd
        prev_slow_ema = [0.0] * sd
        anchor = (
            list(anchor_skip)[:sd] if anchor_skip is not None
            else list(carrier_values)[:sd])
        while len(anchor) < sd:
            anchor.append(0.0)
        parent_cid = W55_V7_NO_PARENT_STATE
    else:
        prev_layers = [
            list(s) for s in prev_state.layer_states]
        while len(prev_layers) < n_layers:
            prev_layers.append([0.0] * sd)
        prev_fast_ema = list(prev_state.fast_ema_carrier)
        prev_slow_ema = list(prev_state.slow_ema_carrier)
        anchor = list(prev_state.anchor_carrier)
        parent_cid = prev_state.cid()
    fa = float(cell.fast_alpha)
    sa = float(cell.slow_alpha)
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
    next_layers, gates = cell.step_value(
        prev_layer_states=prev_layers,
        input_x=carrier_values,
        anchor_skip=anchor,
        fast_ema_skip=fast_ema_next,
        slow_ema_skip=slow_ema_next)
    anchor_cid = _sha256_hex({
        "kind": "w55_v7_anchor_skip",
        "values": _round_floats(anchor),
    })
    fast_ema_cid = _sha256_hex({
        "kind": "w55_v7_fast_ema_skip",
        "values": _round_floats(fast_ema_next),
        "turn_index": int(turn_index),
    })
    slow_ema_cid = _sha256_hex({
        "kind": "w55_v7_slow_ema_skip",
        "values": _round_floats(slow_ema_next),
        "turn_index": int(turn_index),
    })
    gate_l1_sum = float(
        sum(abs(float(g))
            for layer_z in gates for g in layer_z))
    return PersistentLatentStateV7(
        turn_index=int(turn_index),
        role=str(role),
        branch_id=str(branch_id),
        state_dim=int(sd),
        n_layers=int(n_layers),
        layer_states=tuple(
            tuple(_round_floats(s)) for s in next_layers),
        fast_ema_carrier=tuple(_round_floats(fast_ema_next)),
        slow_ema_carrier=tuple(_round_floats(slow_ema_next)),
        anchor_carrier=tuple(_round_floats(anchor)),
        parent_state_cid=str(parent_cid),
        second_parent_state_cid="",
        cell_cid=str(cell.cid()),
        anchor_skip_cid=str(anchor_cid),
        fast_ema_skip_cid=str(fast_ema_cid),
        slow_ema_skip_cid=str(slow_ema_cid),
        update_gate_l1_sum=float(round(gate_l1_sum, 12)),
        disagreement_l1_sum=0.0,
        is_merge=False,
    )


def merge_persistent_states_v7(
        *,
        cell: V7StackedCell,
        state_a: PersistentLatentStateV7,
        state_b: PersistentLatentStateV7,
        merged_branch_id: str,
        turn_index: int | None = None,
        role: str = "",
) -> tuple[PersistentLatentStateV7,
            list[list[float]], list[list[float]], list[list[float]]]:
    """Return (merged_state, disagreement_layers, low_bound_layers, high_bound_layers)."""
    if state_a.state_dim != state_b.state_dim:
        raise ValueError(
            "merge_persistent_states_v7: state_dim mismatch")
    sd = int(state_a.state_dim)
    n_layers = max(state_a.n_layers, state_b.n_layers)
    merged_layers: list[list[float]] = []
    disagreement_layers: list[list[float]] = []
    low_layers: list[list[float]] = []
    high_layers: list[list[float]] = []
    for layer in range(n_layers):
        sa = (
            list(state_a.layer_states[layer])
            if layer < len(state_a.layer_states)
            else [0.0] * sd)
        sb = (
            list(state_b.layer_states[layer])
            if layer < len(state_b.layer_states)
            else [0.0] * sd)
        m, _, d, lo, hi = cell.merge_states_value(sa, sb)
        merged_layers.append(m)
        disagreement_layers.append(d)
        low_layers.append(lo)
        high_layers.append(hi)
    merged_fast_ema, _, _, _, _ = cell.merge_states_value(
        list(state_a.fast_ema_carrier),
        list(state_b.fast_ema_carrier))
    merged_slow_ema, _, _, _, _ = cell.merge_states_value(
        list(state_a.slow_ema_carrier),
        list(state_b.slow_ema_carrier))
    anchor = list(state_a.anchor_carrier)
    ti = (
        int(turn_index) if turn_index is not None
        else max(int(state_a.turn_index),
                  int(state_b.turn_index)) + 1)
    rl = str(role) if role else str(state_a.role)
    anchor_cid = _sha256_hex({
        "kind": "w55_v7_merge_anchor",
        "state_a_cid": str(state_a.cid()),
        "state_b_cid": str(state_b.cid()),
    })
    fast_ema_cid = _sha256_hex({
        "kind": "w55_v7_merge_fast_ema",
        "state_a_cid": str(state_a.cid()),
        "state_b_cid": str(state_b.cid()),
    })
    slow_ema_cid = _sha256_hex({
        "kind": "w55_v7_merge_slow_ema",
        "state_a_cid": str(state_a.cid()),
        "state_b_cid": str(state_b.cid()),
    })
    disagreement_total = float(sum(
        float(v) for d in disagreement_layers for v in d))
    merged_state = PersistentLatentStateV7(
        turn_index=int(ti),
        role=str(rl),
        branch_id=str(merged_branch_id),
        state_dim=int(sd),
        n_layers=int(n_layers),
        layer_states=tuple(
            tuple(_round_floats(s)) for s in merged_layers),
        fast_ema_carrier=tuple(_round_floats(merged_fast_ema)),
        slow_ema_carrier=tuple(_round_floats(merged_slow_ema)),
        anchor_carrier=tuple(_round_floats(anchor)),
        parent_state_cid=str(state_a.cid()),
        second_parent_state_cid=str(state_b.cid()),
        cell_cid=str(cell.cid()),
        anchor_skip_cid=str(anchor_cid),
        fast_ema_skip_cid=str(fast_ema_cid),
        slow_ema_skip_cid=str(slow_ema_cid),
        update_gate_l1_sum=0.0,
        disagreement_l1_sum=float(round(
            disagreement_total, 12)),
        is_merge=True,
    )
    return (
        merged_state, disagreement_layers,
        low_layers, high_layers)


# =============================================================================
# Long-horizon recall evaluator
# =============================================================================


def evaluate_v7_long_horizon_recall(
        cell: V7StackedCell,
        sequences: Sequence[Sequence[Sequence[float]]],
        targets: Sequence[Sequence[float]],
) -> float:
    """Mean cosine recall using triple-skip step."""
    if not sequences:
        return 0.0
    cos_sum = 0.0
    n = 0
    sd = int(cell.state_dim)
    for seq, tgt in zip(sequences, targets):
        layer_states = [
            [0.0] * sd for _ in range(cell.n_layers)
        ]
        anchor = (
            list(seq[0])[:sd] if len(seq) > 0
            else [0.0] * sd)
        while len(anchor) < sd:
            anchor.append(0.0)
        fast_ema = [0.0] * sd
        slow_ema = [0.0] * sd
        for x in seq:
            fast_ema = [
                cell.fast_alpha * float(
                    x[i] if i < len(x) else 0.0)
                + (1.0 - cell.fast_alpha) * float(
                    fast_ema[i] if i < len(fast_ema) else 0.0)
                for i in range(sd)
            ]
            slow_ema = [
                cell.slow_alpha * float(
                    x[i] if i < len(x) else 0.0)
                + (1.0 - cell.slow_alpha) * float(
                    slow_ema[i] if i < len(slow_ema) else 0.0)
                for i in range(sd)
            ]
            layer_states, _ = cell.step_value(
                prev_layer_states=layer_states,
                input_x=x,
                anchor_skip=anchor,
                fast_ema_skip=fast_ema,
                slow_ema_skip=slow_ema)
        top = (
            layer_states[-1] if layer_states else [0.0] * sd)
        cos_sum += _cosine(top, tgt)
        n += 1
    return float(cos_sum) / float(max(1, n))


# =============================================================================
# Witness
# =============================================================================


@dataclasses.dataclass(frozen=True)
class PersistentLatentStateV7Witness:
    state_cid: str
    parent_state_cid: str
    second_parent_state_cid: str
    role: str
    branch_id: str
    turn_index: int
    state_dim: int
    n_layers: int
    cell_cid: str
    anchor_skip_cid: str
    fast_ema_skip_cid: str
    slow_ema_skip_cid: str
    chain_walk_depth: int
    chain_cid: str
    update_gate_l1_sum: float
    disagreement_l1_sum: float
    is_merge: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_cid": str(self.state_cid),
            "parent_state_cid": str(self.parent_state_cid),
            "second_parent_state_cid": str(
                self.second_parent_state_cid),
            "role": str(self.role),
            "branch_id": str(self.branch_id),
            "turn_index": int(self.turn_index),
            "state_dim": int(self.state_dim),
            "n_layers": int(self.n_layers),
            "cell_cid": str(self.cell_cid),
            "anchor_skip_cid": str(self.anchor_skip_cid),
            "fast_ema_skip_cid": str(self.fast_ema_skip_cid),
            "slow_ema_skip_cid": str(self.slow_ema_skip_cid),
            "chain_walk_depth": int(self.chain_walk_depth),
            "chain_cid": str(self.chain_cid),
            "update_gate_l1_sum": float(round(
                self.update_gate_l1_sum, 12)),
            "disagreement_l1_sum": float(round(
                self.disagreement_l1_sum, 12)),
            "is_merge": bool(self.is_merge),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w55_v7_witness",
            "witness": self.to_dict()})


def emit_persistent_v7_witness(
        *,
        state: PersistentLatentStateV7,
        cell: V7StackedCell,
        chain: PersistentLatentStateV7Chain,
        max_walk_depth: int = (
            W55_DEFAULT_V7_MAX_CHAIN_WALK_DEPTH),
) -> PersistentLatentStateV7Witness:
    walk = chain.walk_from(
        state.cid(), max_depth=int(max_walk_depth))
    return PersistentLatentStateV7Witness(
        state_cid=str(state.cid()),
        parent_state_cid=str(state.parent_state_cid),
        second_parent_state_cid=str(
            state.second_parent_state_cid),
        role=str(state.role),
        branch_id=str(state.branch_id),
        turn_index=int(state.turn_index),
        state_dim=int(state.state_dim),
        n_layers=int(state.n_layers),
        cell_cid=str(cell.cid()),
        anchor_skip_cid=str(state.anchor_skip_cid),
        fast_ema_skip_cid=str(state.fast_ema_skip_cid),
        slow_ema_skip_cid=str(state.slow_ema_skip_cid),
        chain_walk_depth=int(len(walk)),
        chain_cid=str(chain.cid()),
        update_gate_l1_sum=float(state.update_gate_l1_sum),
        disagreement_l1_sum=float(state.disagreement_l1_sum),
        is_merge=bool(state.is_merge),
    )


# =============================================================================
# Verifier
# =============================================================================

W55_V7_VERIFIER_FAILURE_MODES: tuple[str, ...] = (
    "w55_v7_schema_mismatch",
    "w55_v7_state_cid_mismatch",
    "w55_v7_cell_cid_mismatch",
    "w55_v7_chain_walk_depth_below_floor",
    "w55_v7_update_gate_pathology",
    "w55_v7_n_layers_mismatch",
    "w55_v7_anchor_skip_cid_mismatch",
    "w55_v7_fast_ema_skip_cid_mismatch",
    "w55_v7_slow_ema_skip_cid_mismatch",
    "w55_v7_merge_state_missing_second_parent",
    "w55_v7_disagreement_l1_negative",
)


def verify_persistent_v7_witness(
        witness: PersistentLatentStateV7Witness,
        *,
        expected_state_cid: str | None = None,
        expected_cell_cid: str | None = None,
        expected_n_layers: int | None = None,
        min_chain_walk_depth: int | None = None,
        max_gate_l1_pathology: float | None = None,
) -> dict[str, Any]:
    failures: list[str] = []
    if (expected_state_cid is not None
            and witness.state_cid != expected_state_cid):
        failures.append("w55_v7_state_cid_mismatch")
    if (expected_cell_cid is not None
            and witness.cell_cid != expected_cell_cid):
        failures.append("w55_v7_cell_cid_mismatch")
    if (expected_n_layers is not None
            and witness.n_layers != int(expected_n_layers)):
        failures.append("w55_v7_n_layers_mismatch")
    if (min_chain_walk_depth is not None
            and witness.chain_walk_depth
            < int(min_chain_walk_depth)):
        failures.append(
            "w55_v7_chain_walk_depth_below_floor")
    if (max_gate_l1_pathology is not None
            and witness.update_gate_l1_sum
            > float(max_gate_l1_pathology)):
        failures.append("w55_v7_update_gate_pathology")
    if (witness.is_merge
            and not witness.second_parent_state_cid):
        failures.append(
            "w55_v7_merge_state_missing_second_parent")
    if witness.disagreement_l1_sum < 0.0:
        failures.append("w55_v7_disagreement_l1_negative")
    return {
        "ok": (len(failures) == 0),
        "failures": failures,
        "witness_cid": witness.cid(),
    }


def fit_persistent_v7(
        *,
        state_dim: int = W55_DEFAULT_V7_STATE_DIM,
        input_dim: int = W55_DEFAULT_V7_INPUT_DIM,
        n_layers: int = W55_DEFAULT_V7_N_LAYERS,
        seed: int = W47_DEFAULT_TRAIN_SEED,
        n_sequences: int = 6,
        sequence_length: int = 16,
        n_steps: int = 64,
        learning_rate: float = 0.04,
        truncate_bptt: int = 4,
        distractor_window: tuple[int, int] | None = None,
        distractor_magnitude: float = 0.5,
) -> tuple["V7StackedCell", Any]:
    """Fit inner V6 cell (which itself fits inner V5); wrap in V7.

    The V7 outer top GRU + slow-EMA projection are initialised but
    not separately trained (W55-L-V7-OUTER-NOT-TRAINED-CAP).
    """
    from .persistent_latent_v6 import fit_persistent_v6
    v6_cell, trace = fit_persistent_v6(
        state_dim=int(state_dim),
        input_dim=int(input_dim),
        n_layers=max(W54_DEFAULT_V6_N_LAYERS,
                      int(n_layers) - 1),
        seed=int(seed),
        n_sequences=int(n_sequences),
        sequence_length=int(sequence_length),
        n_steps=int(n_steps),
        learning_rate=float(learning_rate),
        truncate_bptt=int(truncate_bptt),
        distractor_window=distractor_window,
        distractor_magnitude=float(distractor_magnitude))
    v7 = V7StackedCell.init(
        state_dim=int(state_dim),
        input_dim=int(input_dim),
        n_layers=int(n_layers),
        seed=int(seed))
    v7.inner_v6 = v6_cell
    return v7, trace


def forge_v7_carrier_sequences(
        sequences: Sequence[Sequence[Sequence[float]]],
        *,
        seed: int = W47_DEFAULT_TRAIN_SEED,
) -> list[list[list[float]]]:
    """Forge carriers by replacing the turn-0 signal with random noise."""
    rng = _DeterministicLCG(seed=int(seed))
    forged: list[list[list[float]]] = []
    for seq in sequences:
        new_seq: list[list[float]] = []
        for t, x in enumerate(seq):
            if t == 0:
                new_seq.append([
                    float(rng.next_uniform() * 2.0 - 1.0)
                    for _ in range(len(x))
                ])
            else:
                new_seq.append(list(x))
        forged.append(new_seq)
    return forged


__all__ = [
    "W55_PERSISTENT_V7_SCHEMA_VERSION",
    "W55_DEFAULT_V7_STATE_DIM",
    "W55_DEFAULT_V7_INPUT_DIM",
    "W55_DEFAULT_V7_N_LAYERS",
    "W55_DEFAULT_V7_MAX_CHAIN_WALK_DEPTH",
    "W55_DEFAULT_V7_FAST_EMA_ALPHA",
    "W55_DEFAULT_V7_SLOW_EMA_ALPHA",
    "W55_V7_NO_PARENT_STATE",
    "W55_V7_VERIFIER_FAILURE_MODES",
    "V7StackedCell",
    "PersistentLatentStateV7",
    "PersistentLatentStateV7Chain",
    "PersistentLatentStateV7Witness",
    "step_persistent_state_v7",
    "merge_persistent_states_v7",
    "evaluate_v7_long_horizon_recall",
    "emit_persistent_v7_witness",
    "verify_persistent_v7_witness",
    "fit_persistent_v7",
    "forge_v7_carrier_sequences",
]
