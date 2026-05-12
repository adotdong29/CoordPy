"""W53 M1 — Persistent Latent State V5 (3-layer stack + skip-link
   + branch merge).

A three-layer stacked GRU-style persistent latent state with two
upgrades over W52 V4:

* an optional **persistent skip-link** that is applied at *every*
  step (not just turn 0), so the upper layer keeps direct access
  to the carrier signal across longer chains than V4
* a **state merge head** that maps a pair of states from
  different branches to a merged state under a learned blend
  ratio — separate from MLSC's payload merge, this is the
  GRU-state-internal merge

Mathematics::

    Layer 0:
        z0_t = sigmoid(W_z0 · [s0_{t-1}; x_t] + b_z0)
        h0_t = tanh(W_h0 · [s0_{t-1}; x_t] + b_h0)
        s0_t = (1 - z0_t) ⊙ s0_{t-1} + z0_t ⊙ h0_t

    Skip:
        skip_t = W_skip · skip_input_t  (any t, persistent)

    Layer 1:
        z1_t = sigmoid(W_z1 · [s1_{t-1}; s0_t + skip_t] + b_z1)
        h1_t = tanh(W_h1 · [s1_{t-1}; s0_t + skip_t] + b_h1)
        s1_t = (1 - z1_t) ⊙ s1_{t-1} + z1_t ⊙ h1_t

    Layer 2:
        z2_t = sigmoid(W_z2 · [s2_{t-1}; s1_t] + b_z2)
        h2_t = tanh(W_h2 · [s2_{t-1}; s1_t] + b_h2)
        s2_t = (1 - z2_t) ⊙ s2_{t-1} + z2_t ⊙ h2_t

State merge::

    merge(s_a, s_b, alpha) = alpha ⊙ s_a + (1 - alpha) ⊙ s_b
        where alpha = sigmoid(W_merge · [s_a; s_b] + b_merge)

Pure-Python only — reuses the W47 ``Variable`` + ``AdamOptimizer``
autograd engine. Honest scope: same as W52 V4; does NOT touch
transformer-internal state. The "stretch-32" target is honest:
the chain walks past 32 turns; cosine recall is bounded.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

from .autograd_manifold import (
    AdamOptimizer,
    ParamTensor,
    Variable,
    W47_DEFAULT_BETA1,
    W47_DEFAULT_BETA2,
    W47_DEFAULT_EPS,
    W47_DEFAULT_GRAD_CLIP,
    W47_DEFAULT_INIT_SCALE,
    W47_DEFAULT_LEARNING_RATE,
    W47_DEFAULT_TRAIN_SEED,
    _DeterministicLCG,
    vmatmul,
    vmean,
)
from .persistent_shared_latent import W51_DEFAULT_STATE_DIM


# =============================================================================
# Schema, defaults
# =============================================================================

W53_PERSISTENT_V5_SCHEMA_VERSION: str = (
    "coordpy.persistent_latent_v5.v1")

W53_DEFAULT_V5_STATE_DIM: int = W51_DEFAULT_STATE_DIM
W53_DEFAULT_V5_INPUT_DIM: int = W51_DEFAULT_STATE_DIM
W53_DEFAULT_V5_N_LAYERS: int = 3
W53_DEFAULT_V5_MAX_CHAIN_WALK_DEPTH: int = 32
W53_V5_NO_PARENT_STATE: str = "no_parent_v5_state"


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


def _stable_sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


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
# V5 layered cell
# =============================================================================


@dataclasses.dataclass
class V5StackedCell:
    """Three-layer stacked GRU cell with persistent skip-link
    and a state-merge head."""

    state_dim: int
    input_dim: int
    n_layers: int
    w_z: tuple[ParamTensor, ...]
    b_z: tuple[ParamTensor, ...]
    w_h: tuple[ParamTensor, ...]
    b_h: tuple[ParamTensor, ...]
    w_skip: ParamTensor
    w_merge: ParamTensor  # (state_dim, 2*state_dim)
    b_merge: ParamTensor  # (state_dim,)

    @classmethod
    def init(
            cls, *,
            state_dim: int = W53_DEFAULT_V5_STATE_DIM,
            input_dim: int = W53_DEFAULT_V5_INPUT_DIM,
            n_layers: int = W53_DEFAULT_V5_N_LAYERS,
            seed: int = W47_DEFAULT_TRAIN_SEED,
            init_scale: float = W47_DEFAULT_INIT_SCALE,
    ) -> "V5StackedCell":
        rng = _DeterministicLCG(seed=int(seed))
        w_z: list[ParamTensor] = []
        b_z: list[ParamTensor] = []
        w_h: list[ParamTensor] = []
        b_h: list[ParamTensor] = []
        for layer in range(int(n_layers)):
            in_d = (
                int(input_dim) if layer == 0
                else int(state_dim))
            cat_d = int(state_dim) + in_d
            wz = ParamTensor(
                shape=(int(state_dim), int(cat_d)), values=[])
            wz.init_seed(
                seed=int(rng.next_uniform() * (1 << 30)),
                scale=float(init_scale))
            bz = ParamTensor(
                shape=(int(state_dim),),
                values=[-1.0] * int(state_dim))
            wh = ParamTensor(
                shape=(int(state_dim), int(cat_d)), values=[])
            wh.init_seed(
                seed=int(rng.next_uniform() * (1 << 30)),
                scale=float(init_scale))
            bh = ParamTensor(
                shape=(int(state_dim),),
                values=[0.0] * int(state_dim))
            w_z.append(wz)
            b_z.append(bz)
            w_h.append(wh)
            b_h.append(bh)
        # Skip-link projector — initialize as identity*0.1.
        w_skip = ParamTensor(
            shape=(int(state_dim), int(state_dim)), values=[])
        vals = [0.0] * (int(state_dim) * int(state_dim))
        for i in range(int(state_dim)):
            vals[i * int(state_dim) + i] = 0.1
        w_skip.values = vals
        # Merge head: identity-like init so early merges == 0.5
        # blend by default.
        w_merge = ParamTensor(
            shape=(int(state_dim), 2 * int(state_dim)),
            values=[])
        w_merge.init_seed(
            seed=int(rng.next_uniform() * (1 << 30)),
            scale=float(init_scale) * 0.5)
        b_merge = ParamTensor(
            shape=(int(state_dim),),
            values=[0.0] * int(state_dim))
        return cls(
            state_dim=int(state_dim),
            input_dim=int(input_dim),
            n_layers=int(n_layers),
            w_z=tuple(w_z), b_z=tuple(b_z),
            w_h=tuple(w_h), b_h=tuple(b_h),
            w_skip=w_skip,
            w_merge=w_merge, b_merge=b_merge)

    def params(self) -> list[ParamTensor]:
        out: list[ParamTensor] = []
        for layer in range(self.n_layers):
            out.append(self.w_z[layer])
            out.append(self.b_z[layer])
            out.append(self.w_h[layer])
            out.append(self.b_h[layer])
        out.extend([self.w_skip, self.w_merge, self.b_merge])
        return out

    def _layer_step_value(
            self,
            layer: int,
            prev_state: Sequence[float],
            layer_input: Sequence[float],
    ) -> tuple[list[float], list[float]]:
        sd = self.state_dim
        in_d = (
            self.input_dim if layer == 0 else self.state_dim)
        cat_d = sd + in_d
        cat = [
            (float(prev_state[i]) if i < len(prev_state) else 0.0)
            for i in range(sd)
        ] + [
            (float(layer_input[j])
             if j < len(layer_input) else 0.0)
            for j in range(in_d)
        ]
        z = [0.0] * sd
        h = [0.0] * sd
        wz = self.w_z[layer].values
        bz = self.b_z[layer].values
        wh = self.w_h[layer].values
        bh = self.b_h[layer].values
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
        nxt = [
            (1.0 - z[i]) * (
                float(prev_state[i])
                if i < len(prev_state) else 0.0
            ) + z[i] * h[i]
            for i in range(sd)
        ]
        return nxt, z

    def _skip_project_value(
            self,
            skip_input: Sequence[float],
    ) -> list[float]:
        sd = self.state_dim
        out = [0.0] * sd
        for i in range(sd):
            s = 0.0
            for j in range(sd):
                sj = float(
                    skip_input[j]
                    if j < len(skip_input) else 0.0)
                s += float(self.w_skip.values[i * sd + j]) * sj
            out[i] = s
        return out

    def step_value(
            self, *,
            prev_layer_states: Sequence[Sequence[float]],
            input_x: Sequence[float],
            skip_input: Sequence[float] | None = None,
    ) -> tuple[list[list[float]], list[list[float]]]:
        sd = self.state_dim
        cur = list(input_x)
        next_states: list[list[float]] = []
        gates: list[list[float]] = []
        for layer in range(self.n_layers):
            prev = (
                list(prev_layer_states[layer])
                if layer < len(prev_layer_states)
                else [0.0] * sd)
            layer_input = list(cur)
            # Apply persistent skip-link to layer 1 (and 2 if 3+
            # layers exist) — every step, not just turn 0.
            if (layer == 1 and skip_input is not None
                    and self.n_layers >= 2):
                proj = self._skip_project_value(skip_input)
                layer_input = [
                    float(layer_input[i] if i < len(layer_input) else 0.0)
                    + float(proj[i] if i < len(proj) else 0.0)
                    for i in range(sd)
                ]
            nxt, z = self._layer_step_value(
                layer=layer, prev_state=prev,
                layer_input=layer_input)
            next_states.append(nxt)
            gates.append(z)
            cur = nxt
        return next_states, gates

    def merge_states_value(
            self,
            state_a: Sequence[float],
            state_b: Sequence[float],
    ) -> tuple[list[float], list[float]]:
        """Apply the state-merge head between two same-layer
        states. Returns (merged_state, alpha)."""
        sd = self.state_dim
        cat = [
            float(state_a[i] if i < len(state_a) else 0.0)
            for i in range(sd)
        ] + [
            float(state_b[i] if i < len(state_b) else 0.0)
            for i in range(sd)
        ]
        wm = self.w_merge.values
        bm = self.b_merge.values
        alpha = [0.0] * sd
        for r in range(sd):
            s = 0.0
            base = r * (2 * sd)
            for j in range(2 * sd):
                s += float(wm[base + j]) * float(cat[j])
            s += float(bm[r])
            alpha[r] = float(_stable_sigmoid(s))
        merged = [
            float(alpha[i]) * float(state_a[i] if i < len(state_a) else 0.0)
            + (1.0 - float(alpha[i])) * float(
                state_b[i] if i < len(state_b) else 0.0)
            for i in range(sd)
        ]
        return merged, alpha

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": str(
                W53_PERSISTENT_V5_SCHEMA_VERSION),
            "state_dim": int(self.state_dim),
            "input_dim": int(self.input_dim),
            "n_layers": int(self.n_layers),
            "w_z": [p.to_dict() for p in self.w_z],
            "b_z": [p.to_dict() for p in self.b_z],
            "w_h": [p.to_dict() for p in self.w_h],
            "b_h": [p.to_dict() for p in self.b_h],
            "w_skip": self.w_skip.to_dict(),
            "w_merge": self.w_merge.to_dict(),
            "b_merge": self.b_merge.to_dict(),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w53_v5_stacked_cell",
            "cell": self.to_dict()})


# =============================================================================
# V5 PersistentLatentState
# =============================================================================


@dataclasses.dataclass(frozen=True)
class PersistentLatentStateV5:
    turn_index: int
    role: str
    branch_id: str
    state_dim: int
    n_layers: int
    layer_states: tuple[tuple[float, ...], ...]
    cycle_state: tuple[float, ...]
    parent_state_cid: str
    second_parent_state_cid: str  # For merge nodes; "" otherwise.
    cell_cid: str
    skip_link_input_cid: str
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
            "cycle_state": list(_round_floats(
                self.cycle_state)),
            "parent_state_cid": str(self.parent_state_cid),
            "second_parent_state_cid": str(
                self.second_parent_state_cid),
            "cell_cid": str(self.cell_cid),
            "skip_link_input_cid": str(
                self.skip_link_input_cid),
            "update_gate_l1_sum": float(round(
                self.update_gate_l1_sum, 12)),
            "is_merge": bool(self.is_merge),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w53_v5_persistent_state",
            "state": self.to_dict()})

    @property
    def top_state(self) -> tuple[float, ...]:
        if not self.layer_states:
            return tuple([0.0] * int(self.state_dim))
        return tuple(self.layer_states[-1])


@dataclasses.dataclass
class PersistentLatentStateV5Chain:
    states: dict[str, PersistentLatentStateV5]

    @classmethod
    def empty(cls) -> "PersistentLatentStateV5Chain":
        return cls(states={})

    def add(self, state: PersistentLatentStateV5) -> None:
        self.states[state.cid()] = state

    def get(
            self, cid: str,
    ) -> PersistentLatentStateV5 | None:
        return self.states.get(str(cid))

    def walk_from(
            self, leaf_cid: str, *,
            max_depth: int = (
                W53_DEFAULT_V5_MAX_CHAIN_WALK_DEPTH),
    ) -> list[PersistentLatentStateV5]:
        """Walk back through parents (single or merge).

        For merge nodes, follows the *first* parent_state_cid.
        Use ``walk_all_ancestors_from`` for full DAG walk.
        """
        out: list[PersistentLatentStateV5] = []
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

    def walk_all_ancestors_from(
            self, leaf_cid: str, *,
            max_steps: int = 256,
    ) -> list[PersistentLatentStateV5]:
        """DAG walk including merge node second parents."""
        out: dict[str, PersistentLatentStateV5] = {}
        stack: list[str] = [str(leaf_cid)]
        steps = 0
        while stack and steps < int(max_steps):
            cid = stack.pop()
            if cid in out:
                continue
            cur = self.get(cid)
            if cur is None:
                steps += 1
                continue
            out[cid] = cur
            if cur.parent_state_cid:
                stack.append(str(cur.parent_state_cid))
            if cur.is_merge and cur.second_parent_state_cid:
                stack.append(
                    str(cur.second_parent_state_cid))
            steps += 1
        # Sorted by turn_index then cid for determinism.
        return sorted(
            out.values(),
            key=lambda s: (int(s.turn_index), s.cid()))

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w53_v5_state_chain",
            "members": [
                {"cid": c, "state": s.to_dict()}
                for c, s in sorted(self.states.items())
            ],
        })


# =============================================================================
# Step helper
# =============================================================================


def step_persistent_state_v5(
        *,
        cell: V5StackedCell,
        prev_state: PersistentLatentStateV5 | None,
        carrier_values: Sequence[float],
        turn_index: int,
        role: str,
        branch_id: str = "main",
        cycle_index: int = 0,
        skip_input: Sequence[float] | None = None,
) -> PersistentLatentStateV5:
    sd = cell.state_dim
    n_layers = cell.n_layers
    if prev_state is None:
        prev_layers = [
            [0.0] * sd for _ in range(n_layers)]
        prev_cycle = [0.0] * sd
        parent_cid = W53_V5_NO_PARENT_STATE
    else:
        prev_layers = [
            list(s) for s in prev_state.layer_states]
        while len(prev_layers) < n_layers:
            prev_layers.append([0.0] * sd)
        prev_cycle = list(prev_state.cycle_state)
        parent_cid = prev_state.cid()
    next_layers, gates = cell.step_value(
        prev_layer_states=prev_layers,
        input_x=carrier_values,
        skip_input=skip_input)
    decay = 0.8
    top = next_layers[-1] if next_layers else [0.0] * sd
    nxt_cycle = [
        decay * float(
            prev_cycle[i] if i < len(prev_cycle) else 0.0)
        + (1.0 - decay) * float(top[i])
        for i in range(sd)
    ]
    skip_cid = _sha256_hex({
        "kind": "w53_v5_skip_link_input",
        "values": _round_floats(skip_input or []),
        "turn_index": int(turn_index),
        "cycle_index": int(cycle_index),
    })
    gate_l1_sum = float(
        sum(abs(float(g))
            for layer_z in gates for g in layer_z))
    return PersistentLatentStateV5(
        turn_index=int(turn_index),
        role=str(role),
        branch_id=str(branch_id),
        state_dim=int(sd),
        n_layers=int(n_layers),
        layer_states=tuple(
            tuple(_round_floats(s)) for s in next_layers),
        cycle_state=tuple(_round_floats(nxt_cycle)),
        parent_state_cid=str(parent_cid),
        second_parent_state_cid="",
        cell_cid=str(cell.cid()),
        skip_link_input_cid=str(skip_cid),
        update_gate_l1_sum=float(round(gate_l1_sum, 12)),
        is_merge=False,
    )


def merge_persistent_states_v5(
        *,
        cell: V5StackedCell,
        state_a: PersistentLatentStateV5,
        state_b: PersistentLatentStateV5,
        merged_branch_id: str,
        turn_index: int | None = None,
        role: str = "",
) -> PersistentLatentStateV5:
    """Merge two states from different branches.

    The merge applies ``cell.merge_states_value`` per-layer; the
    cycle state is also merged. Returns a state with kind=merge,
    parent_state_cid=state_a.cid(), second_parent_state_cid=
    state_b.cid().
    """
    if state_a.state_dim != state_b.state_dim:
        raise ValueError(
            "merge_persistent_states_v5: state_dim mismatch")
    sd = state_a.state_dim
    n_layers = max(
        state_a.n_layers, state_b.n_layers)
    merged_layers: list[list[float]] = []
    for layer in range(n_layers):
        sa = (
            list(state_a.layer_states[layer])
            if layer < len(state_a.layer_states)
            else [0.0] * sd)
        sb = (
            list(state_b.layer_states[layer])
            if layer < len(state_b.layer_states)
            else [0.0] * sd)
        m, _ = cell.merge_states_value(sa, sb)
        merged_layers.append(m)
    merged_cycle, _ = cell.merge_states_value(
        list(state_a.cycle_state),
        list(state_b.cycle_state))
    ti = (
        int(turn_index) if turn_index is not None
        else max(int(state_a.turn_index),
                  int(state_b.turn_index)) + 1)
    rl = str(role) if role else str(state_a.role)
    skip_cid = _sha256_hex({
        "kind": "w53_v5_merge_skip_link_input",
        "state_a_cid": str(state_a.cid()),
        "state_b_cid": str(state_b.cid()),
    })
    return PersistentLatentStateV5(
        turn_index=int(ti),
        role=str(rl),
        branch_id=str(merged_branch_id),
        state_dim=int(sd),
        n_layers=int(n_layers),
        layer_states=tuple(
            tuple(_round_floats(s)) for s in merged_layers),
        cycle_state=tuple(_round_floats(merged_cycle)),
        parent_state_cid=str(state_a.cid()),
        second_parent_state_cid=str(state_b.cid()),
        cell_cid=str(cell.cid()),
        skip_link_input_cid=str(skip_cid),
        update_gate_l1_sum=0.0,
        is_merge=True,
    )


# =============================================================================
# Training set + fitter
# =============================================================================


@dataclasses.dataclass(frozen=True)
class V5Example:
    input_sequence: tuple[tuple[float, ...], ...]
    initial_state: tuple[float, ...]
    target_state: tuple[float, ...]


@dataclasses.dataclass(frozen=True)
class V5TrainingSet:
    examples: tuple[V5Example, ...]
    state_dim: int
    input_dim: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_dim": int(self.state_dim),
            "input_dim": int(self.input_dim),
            "examples": [
                {"input_sequence": [list(x) for x in e.input_sequence],
                 "initial_state": list(e.initial_state),
                 "target_state": list(e.target_state)}
                for e in self.examples],
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w53_v5_training_set",
            "set": self.to_dict()})


def synthesize_v5_training_set(
        *,
        n_sequences: int = 8,
        sequence_length: int = 32,
        state_dim: int = W53_DEFAULT_V5_STATE_DIM,
        input_dim: int = W53_DEFAULT_V5_INPUT_DIM,
        seed: int = W47_DEFAULT_TRAIN_SEED,
        signal_position: int = 0,
        noise_magnitude: float = 0.05,
        distractor_window: tuple[int, int] | None = None,
        distractor_magnitude: float = 0.5,
) -> V5TrainingSet:
    rng = _DeterministicLCG(seed=int(seed))
    dist_start, dist_end = (-1, -1)
    if distractor_window is not None:
        dist_start, dist_end = (
            int(distractor_window[0]),
            int(distractor_window[1]))
    examples: list[V5Example] = []
    for _ in range(int(n_sequences)):
        signal = [
            float(rng.next_uniform() * 2.0 - 1.0)
            for _ in range(int(input_dim))
        ]
        seq: list[tuple[float, ...]] = []
        for t in range(int(sequence_length)):
            if t == int(signal_position):
                seq.append(tuple(signal))
            elif (distractor_window is not None
                    and dist_start <= t < dist_end):
                dist = [
                    float(rng.next_uniform() * 2.0 - 1.0)
                    * float(distractor_magnitude)
                    for _ in range(int(input_dim))
                ]
                seq.append(tuple(dist))
            else:
                noise = [
                    float(rng.next_uniform() - 0.5)
                    * float(noise_magnitude)
                    for _ in range(int(input_dim))
                ]
                seq.append(tuple(noise))
        initial = [0.0] * int(state_dim)
        target = list(signal)[:int(state_dim)]
        while len(target) < int(state_dim):
            target.append(0.0)
        examples.append(V5Example(
            input_sequence=tuple(seq),
            initial_state=tuple(initial),
            target_state=tuple(target),
        ))
    return V5TrainingSet(
        examples=tuple(examples),
        state_dim=int(state_dim),
        input_dim=int(input_dim))


@dataclasses.dataclass(frozen=True)
class V5TrainingTrace:
    seed: int
    n_steps: int
    final_loss: float
    final_grad_norm: float
    loss_head: tuple[float, ...]
    loss_tail: tuple[float, ...]
    training_set_cid: str
    final_cell_cid: str
    diverged: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": int(self.seed),
            "n_steps": int(self.n_steps),
            "final_loss": float(round(
                self.final_loss, 12)),
            "final_grad_norm": float(round(
                self.final_grad_norm, 12)),
            "loss_head": [float(round(v, 12))
                          for v in self.loss_head],
            "loss_tail": [float(round(v, 12))
                          for v in self.loss_tail],
            "training_set_cid": str(self.training_set_cid),
            "final_cell_cid": str(self.final_cell_cid),
            "diverged": bool(self.diverged),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w53_v5_training_trace",
            "trace": self.to_dict()})


def fit_persistent_v5(
        training_set: V5TrainingSet,
        *,
        n_steps: int = 96,
        learning_rate: float = 0.04,
        seed: int = W47_DEFAULT_TRAIN_SEED,
        init_scale: float = W47_DEFAULT_INIT_SCALE,
        beta1: float = W47_DEFAULT_BETA1,
        beta2: float = W47_DEFAULT_BETA2,
        eps: float = W47_DEFAULT_EPS,
        grad_clip: float = W47_DEFAULT_GRAD_CLIP,
        history_head: int = 6,
        history_tail: int = 6,
        truncate_bptt: int = 4,
        use_skip_link: bool = True,
        n_layers: int = W53_DEFAULT_V5_N_LAYERS,
) -> tuple[V5StackedCell, V5TrainingTrace]:
    """Fit V5 stacked GRU via truncated BPTT + Adam.

    Skip-link uses the turn-0 carrier value at every step
    (persistent), giving the upper layers continual access to
    the signal even past long distractor windows.
    """
    cell = V5StackedCell.init(
        state_dim=int(training_set.state_dim),
        input_dim=int(training_set.input_dim),
        n_layers=int(n_layers),
        seed=int(seed),
        init_scale=float(init_scale))
    optim = AdamOptimizer(
        learning_rate=float(learning_rate),
        beta1=float(beta1), beta2=float(beta2),
        eps=float(eps), grad_clip=float(grad_clip))
    trainable = cell.params()
    loss_history: list[float] = []
    grad_norm_history: list[float] = []
    diverged = False
    n = max(1, len(training_set.examples))
    for step in range(int(n_steps)):
        for p in trainable:
            p.make_vars()
        idx = step % n
        ex = training_set.examples[idx]
        seq_len = len(ex.input_sequence)
        stop_truncate = max(0, seq_len - int(truncate_bptt))
        sd = int(training_set.state_dim)
        layer_states_val: list[list[float]] = [
            [0.0] * sd for _ in range(cell.n_layers)
        ]
        skip_val = (
            list(ex.input_sequence[0])
            if use_skip_link and len(ex.input_sequence) > 0
            else None)
        for t in range(stop_truncate):
            layer_states_val, _ = cell.step_value(
                prev_layer_states=layer_states_val,
                input_x=ex.input_sequence[t],
                skip_input=skip_val)
        layer_states_var = [
            [Variable(float(v)) for v in s]
            for s in layer_states_val
        ]
        skip_var = (
            [Variable(float(v)) for v in skip_val]
            if skip_val is not None else None)
        # Manual var-step. For tractability we rebuild gates per
        # step using the same primitives as V4 — but applying
        # only to the un-truncated tail.
        for t in range(stop_truncate, seq_len):
            x_var = [
                Variable(float(v))
                for v in ex.input_sequence[t]
            ]
            cur = list(x_var)
            next_states_var: list[list[Variable]] = []
            for layer in range(cell.n_layers):
                in_d = (
                    cell.input_dim if layer == 0
                    else cell.state_dim)
                cat_d = sd + in_d
                prev = (
                    list(layer_states_var[layer])
                    if layer < len(layer_states_var)
                    else [Variable(0.0)] * sd)
                layer_input = list(cur)
                if (layer == 1 and skip_var is not None
                        and cell.n_layers >= 2):
                    w_skip_vars = cell.w_skip.make_vars()
                    rows: list[list[Variable]] = []
                    for i in range(sd):
                        rows.append(list(
                            w_skip_vars[i * sd:i * sd + sd]))
                    proj = vmatmul(rows, list(skip_var))
                    merged: list[Variable] = []
                    for i in range(sd):
                        a = (
                            layer_input[i]
                            if i < len(layer_input)
                            else Variable(0.0))
                        b = (
                            proj[i] if i < len(proj)
                            else Variable(0.0))
                        merged.append(a + b)
                    layer_input = merged
                w_z_vars = cell.w_z[layer].make_vars()
                b_z_vars = cell.b_z[layer].make_vars()
                w_h_vars = cell.w_h[layer].make_vars()
                b_h_vars = cell.b_h[layer].make_vars()
                cat: list[Variable] = []
                for i in range(sd):
                    cat.append(
                        prev[i] if i < len(prev)
                        else Variable(0.0))
                for j in range(in_d):
                    cat.append(
                        layer_input[j]
                        if j < len(layer_input)
                        else Variable(0.0))
                rows_z: list[list[Variable]] = []
                rows_h: list[list[Variable]] = []
                for r in range(sd):
                    base = r * cat_d
                    rows_z.append(
                        list(w_z_vars[base:base + cat_d]))
                    rows_h.append(
                        list(w_h_vars[base:base + cat_d]))
                pre_z = vmatmul(rows_z, cat)
                pre_h = vmatmul(rows_h, cat)
                z_vars = [
                    (pre_z[i] + b_z_vars[i]).sigmoid()
                    for i in range(sd)
                ]
                h_vars = [
                    (pre_h[i] + b_h_vars[i]).tanh()
                    for i in range(sd)
                ]
                nxt: list[Variable] = []
                for i in range(sd):
                    p = (
                        prev[i] if i < len(prev)
                        else Variable(0.0))
                    one_minus_z = Variable(1.0) - z_vars[i]
                    nxt.append(one_minus_z * p + z_vars[i] * h_vars[i])
                next_states_var.append(nxt)
                cur = nxt
            layer_states_var = next_states_var
        top = layer_states_var[-1]
        terms = []
        for j in range(len(ex.target_state)):
            t = Variable(float(ex.target_state[j]))
            o = top[j] if j < len(top) else Variable(0.0)
            d = o - t
            terms.append(d * d)
        loss = vmean(terms)
        loss.backward()
        total_grad_sq = 0.0
        for p in trainable:
            for g in p.grads():
                total_grad_sq += float(g) * float(g)
        gn = math.sqrt(total_grad_sq)
        loss_history.append(float(loss.value))
        grad_norm_history.append(float(gn))
        lv = loss.value
        if (lv != lv or lv == float("inf")
                or lv == float("-inf")):
            diverged = True
            break
        optim.step(trainable)
    head_n = max(0, int(history_head))
    tail_n = max(0, int(history_tail))
    trace = V5TrainingTrace(
        seed=int(seed),
        n_steps=int(n_steps),
        final_loss=float(
            loss_history[-1] if loss_history else 0.0),
        final_grad_norm=float(
            grad_norm_history[-1]
            if grad_norm_history else 0.0),
        loss_head=tuple(loss_history[:head_n]),
        loss_tail=tuple(
            loss_history[-tail_n:] if tail_n > 0 else ()),
        training_set_cid=str(training_set.cid()),
        final_cell_cid=str(cell.cid()),
        diverged=bool(diverged),
    )
    return cell, trace


def evaluate_v5_long_horizon_recall(
        cell: V5StackedCell,
        examples: Sequence[V5Example],
        *,
        use_skip_link: bool = True,
) -> float:
    if not examples:
        return 0.0
    cos_sum = 0.0
    n = 0
    for ex in examples:
        sd = cell.state_dim
        layer_states = [
            [0.0] * sd for _ in range(cell.n_layers)]
        skip_val = (
            list(ex.input_sequence[0])
            if use_skip_link and len(ex.input_sequence) > 0
            else None)
        for t in range(len(ex.input_sequence)):
            layer_states, _ = cell.step_value(
                prev_layer_states=layer_states,
                input_x=ex.input_sequence[t],
                skip_input=skip_val)
        top = (
            layer_states[-1] if layer_states else [0.0] * sd)
        cos_sum += _cosine(top, ex.target_state)
        n += 1
    return float(cos_sum) / float(max(1, n))


# =============================================================================
# Witness
# =============================================================================


@dataclasses.dataclass(frozen=True)
class PersistentLatentStateV5Witness:
    state_cid: str
    parent_state_cid: str
    second_parent_state_cid: str
    role: str
    branch_id: str
    turn_index: int
    state_dim: int
    n_layers: int
    cell_cid: str
    skip_link_input_cid: str
    chain_walk_depth: int
    chain_cid: str
    update_gate_l1_sum: float
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
            "skip_link_input_cid": str(
                self.skip_link_input_cid),
            "chain_walk_depth": int(self.chain_walk_depth),
            "chain_cid": str(self.chain_cid),
            "update_gate_l1_sum": float(round(
                self.update_gate_l1_sum, 12)),
            "is_merge": bool(self.is_merge),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w53_v5_witness",
            "witness": self.to_dict()})


def emit_persistent_v5_witness(
        *,
        state: PersistentLatentStateV5,
        cell: V5StackedCell,
        chain: PersistentLatentStateV5Chain,
        max_walk_depth: int = (
            W53_DEFAULT_V5_MAX_CHAIN_WALK_DEPTH),
) -> PersistentLatentStateV5Witness:
    walk = chain.walk_from(
        state.cid(), max_depth=int(max_walk_depth))
    return PersistentLatentStateV5Witness(
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
        skip_link_input_cid=str(state.skip_link_input_cid),
        chain_walk_depth=int(len(walk)),
        chain_cid=str(chain.cid()),
        update_gate_l1_sum=float(state.update_gate_l1_sum),
        is_merge=bool(state.is_merge),
    )


# =============================================================================
# Verifier + compromise helper
# =============================================================================

W53_V5_VERIFIER_FAILURE_MODES: tuple[str, ...] = (
    "w53_v5_schema_mismatch",
    "w53_v5_state_cid_mismatch",
    "w53_v5_cell_cid_mismatch",
    "w53_v5_chain_walk_depth_below_floor",
    "w53_v5_update_gate_pathology",
    "w53_v5_n_layers_mismatch",
    "w53_v5_skip_link_cid_mismatch",
    "w53_v5_merge_state_missing_second_parent",
)


def verify_persistent_v5_witness(
        witness: PersistentLatentStateV5Witness,
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
        failures.append("w53_v5_state_cid_mismatch")
    if (expected_cell_cid is not None
            and witness.cell_cid != expected_cell_cid):
        failures.append("w53_v5_cell_cid_mismatch")
    if (expected_n_layers is not None
            and witness.n_layers != int(expected_n_layers)):
        failures.append("w53_v5_n_layers_mismatch")
    if (min_chain_walk_depth is not None
            and witness.chain_walk_depth
            < int(min_chain_walk_depth)):
        failures.append(
            "w53_v5_chain_walk_depth_below_floor")
    if (max_gate_l1_pathology is not None
            and witness.update_gate_l1_sum
            > float(max_gate_l1_pathology)):
        failures.append("w53_v5_update_gate_pathology")
    if (witness.is_merge
            and not witness.second_parent_state_cid):
        failures.append(
            "w53_v5_merge_state_missing_second_parent")
    return {
        "ok": (len(failures) == 0),
        "failures": failures,
        "witness_cid": witness.cid(),
    }


def forge_v5_training_set(
        original: V5TrainingSet,
        *,
        seed: int = W47_DEFAULT_TRAIN_SEED,
) -> V5TrainingSet:
    rng = _DeterministicLCG(seed=int(seed))
    forged: list[V5Example] = []
    for ex in original.examples:
        forged_target = tuple(
            float(rng.next_uniform() * 2.0 - 1.0)
            for _ in range(len(ex.target_state)))
        forged.append(V5Example(
            input_sequence=ex.input_sequence,
            initial_state=ex.initial_state,
            target_state=forged_target))
    return V5TrainingSet(
        examples=tuple(forged),
        state_dim=original.state_dim,
        input_dim=original.input_dim)


__all__ = [
    "W53_PERSISTENT_V5_SCHEMA_VERSION",
    "W53_DEFAULT_V5_STATE_DIM",
    "W53_DEFAULT_V5_INPUT_DIM",
    "W53_DEFAULT_V5_N_LAYERS",
    "W53_DEFAULT_V5_MAX_CHAIN_WALK_DEPTH",
    "W53_V5_NO_PARENT_STATE",
    "W53_V5_VERIFIER_FAILURE_MODES",
    "V5StackedCell",
    "PersistentLatentStateV5",
    "PersistentLatentStateV5Chain",
    "PersistentLatentStateV5Witness",
    "V5Example",
    "V5TrainingSet",
    "V5TrainingTrace",
    "synthesize_v5_training_set",
    "fit_persistent_v5",
    "evaluate_v5_long_horizon_recall",
    "step_persistent_state_v5",
    "merge_persistent_states_v5",
    "emit_persistent_v5_witness",
    "verify_persistent_v5_witness",
    "forge_v5_training_set",
]
