"""W52 M1 — Persistent Latent State V4 (stacked + skip-link).

A two-layer stacked GRU-style persistent latent state with a
signal skip-link that survives across longer horizons than
W51's single-layer V3. Mathematics::

    Layer 0:
        z0_t = sigmoid(W_z0 · [s0_{t-1}; x_t] + b_z0)
        h0_t = tanh(W_h0 · [s0_{t-1}; x_t] + b_h0)
        s0_t = (1 - z0_t) ⊙ s0_{t-1} + z0_t ⊙ h0_t

    Skip:
        skip_t = W_skip · turn0_hash_vec  (when t = 0, else 0)

    Layer 1:
        z1_t = sigmoid(W_z1 · [s1_{t-1}; s0_t + skip_t] + b_z1)
        h1_t = tanh(W_h1 · [s1_{t-1}; s0_t + skip_t] + b_h1)
        s1_t = (1 - z1_t) ⊙ s1_{t-1} + z1_t ⊙ h1_t

A separate *cycle-summary state* persists across cycles
(carrying cycle-stationary invariants).

Pure-Python only — reuses the W47 ``Variable`` +
``AdamOptimizer`` autograd engine. Honest scope: same as W51
M1; does NOT touch transformer-internal state.
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

W52_PERSISTENT_V4_SCHEMA_VERSION: str = (
    "coordpy.persistent_latent_v4.v1")

W52_DEFAULT_V4_STATE_DIM: int = W51_DEFAULT_STATE_DIM
W52_DEFAULT_V4_INPUT_DIM: int = W51_DEFAULT_STATE_DIM
W52_DEFAULT_V4_N_LAYERS: int = 2
W52_DEFAULT_V4_MAX_CHAIN_WALK_DEPTH: int = 24
W52_V4_NO_PARENT_STATE: str = "no_parent_v4_state"


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


def _payload_hash_vec(
        payload: Any, dim: int,
) -> list[float]:
    """Deterministic [-1, 1] dim-vector from any payload."""
    h = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    out: list[float] = []
    for i in range(int(dim)):
        nb = h[(i * 2) % len(h):(i * 2) % len(h) + 2]
        if not nb:
            nb = "00"
        v = (int(nb, 16) / 127.5) - 1.0
        out.append(float(round(v, 12)))
    return out


# =============================================================================
# V4 layered cell
# =============================================================================


@dataclasses.dataclass
class V4StackedCell:
    """Two-layer stacked GRU cell with optional skip-link.

    Layer 0 reads the raw carrier; layer 1 reads layer-0 state
    plus an optional skip-link projection of the turn-0 carrier
    hash (a "signal" hint that bypasses the chain of update
    gates).
    """

    state_dim: int
    input_dim: int
    n_layers: int
    # Per-layer params.
    w_z: tuple[ParamTensor, ...]
    b_z: tuple[ParamTensor, ...]
    w_h: tuple[ParamTensor, ...]
    b_h: tuple[ParamTensor, ...]
    w_skip: ParamTensor  # (state_dim, state_dim) for skip projection

    @classmethod
    def init(
            cls, *,
            state_dim: int = W52_DEFAULT_V4_STATE_DIM,
            input_dim: int = W52_DEFAULT_V4_INPUT_DIM,
            n_layers: int = W52_DEFAULT_V4_N_LAYERS,
            seed: int = W47_DEFAULT_TRAIN_SEED,
            init_scale: float = W47_DEFAULT_INIT_SCALE,
    ) -> "V4StackedCell":
        rng = _DeterministicLCG(seed=int(seed))
        w_z: list[ParamTensor] = []
        b_z: list[ParamTensor] = []
        w_h: list[ParamTensor] = []
        b_h: list[ParamTensor] = []
        for layer in range(int(n_layers)):
            in_d = int(input_dim) if layer == 0 else int(state_dim)
            cat_d = int(state_dim) + in_d
            wz = ParamTensor(
                shape=(int(state_dim), int(cat_d)), values=[])
            wz.init_seed(
                seed=int(rng.next_uniform() * (1 << 30)),
                scale=float(init_scale))
            # Init gate bias slightly negative so state persists.
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
        # Skip-link projector.
        w_skip = ParamTensor(
            shape=(int(state_dim), int(state_dim)), values=[])
        # Initialise as identity so an untrained skip-link is
        # still a useful pass-through of the turn-0 hash.
        vals = [0.0] * (int(state_dim) * int(state_dim))
        for i in range(int(state_dim)):
            vals[i * int(state_dim) + i] = 0.1
        w_skip.values = vals
        return cls(
            state_dim=int(state_dim),
            input_dim=int(input_dim),
            n_layers=int(n_layers),
            w_z=tuple(w_z),
            b_z=tuple(b_z),
            w_h=tuple(w_h),
            b_h=tuple(b_h),
            w_skip=w_skip)

    def params(self) -> list[ParamTensor]:
        out: list[ParamTensor] = []
        for layer in range(self.n_layers):
            out.append(self.w_z[layer])
            out.append(self.b_z[layer])
            out.append(self.w_h[layer])
            out.append(self.b_h[layer])
        out.append(self.w_skip)
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
            (float(layer_input[j]) if j < len(layer_input) else 0.0)
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
                float(prev_state[i]) if i < len(prev_state) else 0.0
            ) + z[i] * h[i]
            for i in range(sd)
        ]
        return nxt, z

    def _layer_step_vars(
            self,
            layer: int,
            prev_state: Sequence[Variable],
            layer_input: Sequence[Variable],
            w_z_vars: Sequence[Variable],
            b_z_vars: Sequence[Variable],
            w_h_vars: Sequence[Variable],
            b_h_vars: Sequence[Variable],
    ) -> list[Variable]:
        sd = self.state_dim
        in_d = (
            self.input_dim if layer == 0 else self.state_dim)
        cat_d = sd + in_d
        cat: list[Variable] = []
        for i in range(sd):
            cat.append(
                prev_state[i] if i < len(prev_state)
                else Variable(0.0))
        for j in range(in_d):
            cat.append(
                layer_input[j] if j < len(layer_input)
                else Variable(0.0))
        rows_z: list[list[Variable]] = []
        rows_h: list[list[Variable]] = []
        for r in range(sd):
            base = r * cat_d
            rows_z.append(list(w_z_vars[base:base + cat_d]))
            rows_h.append(list(w_h_vars[base:base + cat_d]))
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
            p = prev_state[i] if i < len(prev_state) else Variable(0.0)
            one_minus_z = Variable(1.0) - z_vars[i]
            nxt.append(one_minus_z * p + z_vars[i] * h_vars[i])
        return nxt

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
                    skip_input[j] if j < len(skip_input) else 0.0)
                s += float(self.w_skip.values[i * sd + j]) * sj
            out[i] = s
        return out

    def step_value(
            self, *,
            prev_layer_states: Sequence[Sequence[float]],
            input_x: Sequence[float],
            skip_input: Sequence[float] | None = None,
    ) -> tuple[list[list[float]], list[list[float]]]:
        """Run one stacked-step at value-only.

        Returns (per_layer_next_state, per_layer_update_gate).
        """
        sd = self.state_dim
        cur = list(input_x)
        next_states: list[list[float]] = []
        gates: list[list[float]] = []
        for layer in range(self.n_layers):
            prev = (
                list(prev_layer_states[layer])
                if layer < len(prev_layer_states)
                else [0.0] * sd)
            # Add skip-link to layer-1 input.
            layer_input = list(cur)
            if (layer == 1 and skip_input is not None
                    and self.n_layers >= 2):
                proj = self._skip_project_value(skip_input)
                # layer_input must be state_dim dim;
                # cur already is.
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

    def step_vars(
            self, *,
            prev_layer_states: Sequence[Sequence[Variable]],
            input_x: Sequence[Variable],
            skip_input: Sequence[Variable] | None = None,
    ) -> list[list[Variable]]:
        """Run one stacked-step under autograd tape."""
        sd = self.state_dim
        cur = list(input_x)
        next_states: list[list[Variable]] = []
        w_skip_vars = self.w_skip.make_vars()
        for layer in range(self.n_layers):
            prev = (
                list(prev_layer_states[layer])
                if layer < len(prev_layer_states)
                else [Variable(0.0)] * sd)
            layer_input = list(cur)
            if (layer == 1 and skip_input is not None
                    and self.n_layers >= 2):
                # Skip projection via vmatmul.
                rows: list[list[Variable]] = []
                for i in range(sd):
                    rows.append(list(
                        w_skip_vars[i * sd:i * sd + sd]))
                proj = vmatmul(rows, list(skip_input))
                merged: list[Variable] = []
                for i in range(sd):
                    a = layer_input[i] if i < len(layer_input) else Variable(0.0)
                    b = proj[i] if i < len(proj) else Variable(0.0)
                    merged.append(a + b)
                layer_input = merged
            w_z_vars = self.w_z[layer].make_vars()
            b_z_vars = self.b_z[layer].make_vars()
            w_h_vars = self.w_h[layer].make_vars()
            b_h_vars = self.b_h[layer].make_vars()
            nxt = self._layer_step_vars(
                layer=layer, prev_state=prev,
                layer_input=layer_input,
                w_z_vars=w_z_vars, b_z_vars=b_z_vars,
                w_h_vars=w_h_vars, b_h_vars=b_h_vars)
            next_states.append(nxt)
            cur = nxt
        return next_states

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": str(
                W52_PERSISTENT_V4_SCHEMA_VERSION),
            "state_dim": int(self.state_dim),
            "input_dim": int(self.input_dim),
            "n_layers": int(self.n_layers),
            "w_z": [p.to_dict() for p in self.w_z],
            "b_z": [p.to_dict() for p in self.b_z],
            "w_h": [p.to_dict() for p in self.w_h],
            "b_h": [p.to_dict() for p in self.b_h],
            "w_skip": self.w_skip.to_dict(),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w52_v4_stacked_cell",
            "cell": self.to_dict()})


# =============================================================================
# V4 PersistentLatentState
# =============================================================================


@dataclasses.dataclass(frozen=True)
class PersistentLatentStateV4:
    """Per-turn V4 persistent latent state.

    Stores all n_layers states + cycle-summary state + parent
    state CID + skip-link projection CID.
    """

    turn_index: int
    role: str
    state_dim: int
    n_layers: int
    layer_states: tuple[tuple[float, ...], ...]
    cycle_state: tuple[float, ...]
    parent_state_cid: str
    cell_cid: str
    skip_link_input_cid: str
    update_gate_l1_sum: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_index": int(self.turn_index),
            "role": str(self.role),
            "state_dim": int(self.state_dim),
            "n_layers": int(self.n_layers),
            "layer_states": [
                list(_round_floats(s))
                for s in self.layer_states
            ],
            "cycle_state": list(_round_floats(self.cycle_state)),
            "parent_state_cid": str(self.parent_state_cid),
            "cell_cid": str(self.cell_cid),
            "skip_link_input_cid": str(self.skip_link_input_cid),
            "update_gate_l1_sum": float(round(
                self.update_gate_l1_sum, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w52_v4_persistent_state",
            "state": self.to_dict()})

    @property
    def top_state(self) -> tuple[float, ...]:
        """Convenience: top-layer state used as downstream carrier."""
        if not self.layer_states:
            return tuple([0.0] * int(self.state_dim))
        return tuple(self.layer_states[-1])


@dataclasses.dataclass
class PersistentLatentStateV4Chain:
    """Content-addressed chain of V4 states."""

    states: dict[str, PersistentLatentStateV4]

    @classmethod
    def empty(cls) -> "PersistentLatentStateV4Chain":
        return cls(states={})

    def add(self, state: PersistentLatentStateV4) -> None:
        self.states[state.cid()] = state

    def get(
            self, cid: str,
    ) -> PersistentLatentStateV4 | None:
        return self.states.get(str(cid))

    def walk_from(
            self, leaf_cid: str, *,
            max_depth: int = W52_DEFAULT_V4_MAX_CHAIN_WALK_DEPTH,
    ) -> list[PersistentLatentStateV4]:
        out: list[PersistentLatentStateV4] = []
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
            "kind": "w52_v4_state_chain",
            "members": [
                {"cid": c, "state": s.to_dict()}
                for c, s in sorted(self.states.items())
            ],
        })


# =============================================================================
# Step helper
# =============================================================================


def step_persistent_state_v4(
        *,
        cell: V4StackedCell,
        prev_state: PersistentLatentStateV4 | None,
        carrier_values: Sequence[float],
        turn_index: int,
        role: str,
        cycle_index: int = 0,
        skip_input: Sequence[float] | None = None,
) -> PersistentLatentStateV4:
    """One forward step of V4. Skip-link applied at turn 0 by
    default; callers may pass a custom skip vector at later
    turns when convenient.
    """
    sd = cell.state_dim
    n_layers = cell.n_layers
    if prev_state is None:
        prev_layers = [
            [0.0] * sd for _ in range(n_layers)
        ]
        prev_cycle = [0.0] * sd
        parent_cid = W52_V4_NO_PARENT_STATE
    else:
        prev_layers = [list(s) for s in prev_state.layer_states]
        while len(prev_layers) < n_layers:
            prev_layers.append([0.0] * sd)
        prev_cycle = list(prev_state.cycle_state)
        parent_cid = prev_state.cid()
    next_layers, gates = cell.step_value(
        prev_layer_states=prev_layers,
        input_x=carrier_values,
        skip_input=skip_input)
    # Cycle state is the EMA of the top-layer state.
    decay = 0.8
    top = next_layers[-1] if next_layers else [0.0] * sd
    nxt_cycle = [
        decay * float(prev_cycle[i] if i < len(prev_cycle) else 0.0)
        + (1.0 - decay) * float(top[i])
        for i in range(sd)
    ]
    skip_cid = _sha256_hex({
        "kind": "w52_v4_skip_link_input",
        "values": _round_floats(skip_input or []),
        "turn_index": int(turn_index),
        "cycle_index": int(cycle_index),
    })
    gate_l1_sum = float(
        sum(abs(float(g)) for layer_z in gates for g in layer_z))
    return PersistentLatentStateV4(
        turn_index=int(turn_index),
        role=str(role),
        state_dim=int(sd),
        n_layers=int(n_layers),
        layer_states=tuple(
            tuple(_round_floats(s)) for s in next_layers),
        cycle_state=tuple(_round_floats(nxt_cycle)),
        parent_state_cid=str(parent_cid),
        cell_cid=str(cell.cid()),
        skip_link_input_cid=str(skip_cid),
        update_gate_l1_sum=float(round(gate_l1_sum, 12)),
    )


# =============================================================================
# Training set
# =============================================================================


@dataclasses.dataclass(frozen=True)
class V4Example:
    input_sequence: tuple[tuple[float, ...], ...]
    initial_state: tuple[float, ...]
    target_state: tuple[float, ...]


@dataclasses.dataclass(frozen=True)
class V4TrainingSet:
    examples: tuple[V4Example, ...]
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
            "kind": "w52_v4_training_set",
            "set": self.to_dict()})


def synthesize_v4_training_set(
        *,
        n_sequences: int = 8,
        sequence_length: int = 20,
        state_dim: int = W52_DEFAULT_V4_STATE_DIM,
        input_dim: int = W52_DEFAULT_V4_INPUT_DIM,
        seed: int = W47_DEFAULT_TRAIN_SEED,
        signal_position: int = 0,
        noise_magnitude: float = 0.05,
        distractor_window: tuple[int, int] | None = None,
        distractor_magnitude: float = 0.5,
) -> V4TrainingSet:
    """Long-horizon training set: signal at turn 0, noise after.

    ``distractor_window`` (start, end) inserts large distractor
    pulses in that window — these stress-test the GRU's gate
    chain and force the skip-link to carry the signal forward.
    """
    rng = _DeterministicLCG(seed=int(seed))
    dist_start, dist_end = (-1, -1)
    if distractor_window is not None:
        dist_start, dist_end = (
            int(distractor_window[0]),
            int(distractor_window[1]))
    examples: list[V4Example] = []
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
                # Distractor: larger-magnitude noise.
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
        examples.append(V4Example(
            input_sequence=tuple(seq),
            initial_state=tuple(initial),
            target_state=tuple(target),
        ))
    return V4TrainingSet(
        examples=tuple(examples),
        state_dim=int(state_dim),
        input_dim=int(input_dim))


@dataclasses.dataclass(frozen=True)
class V4TrainingTrace:
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
            "final_loss": float(round(self.final_loss, 12)),
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
            "kind": "w52_v4_training_trace",
            "trace": self.to_dict()})


def fit_persistent_v4(
        training_set: V4TrainingSet,
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
) -> tuple[V4StackedCell, V4TrainingTrace]:
    """Fit the V4 stacked GRU via truncated BPTT + Adam SGD.

    Skip-link uses the turn-0 carrier value as the skip input,
    so the upper layer always has direct access to the signal.
    """
    cell = V4StackedCell.init(
        state_dim=int(training_set.state_dim),
        input_dim=int(training_set.input_dim),
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
        # Forward early turns without autograd.
        sd = int(training_set.state_dim)
        layer_states_val: list[list[float]] = [
            [0.0] * sd for _ in range(cell.n_layers)
        ]
        skip_val = (
            list(ex.input_sequence[0])
            if use_skip_link and len(ex.input_sequence) > 0
            else None)
        # Apply skip-link at every step so the signal is
        # continuously available to the upper layer; this is the
        # whole point of the skip-link (W51 V3 could not see the
        # signal past turn 0 except through the chain of gates).
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
        for t in range(stop_truncate, seq_len):
            x_var = [
                Variable(float(v))
                for v in ex.input_sequence[t]
            ]
            layer_states_var = cell.step_vars(
                prev_layer_states=layer_states_var,
                input_x=x_var,
                skip_input=skip_var)
        # Loss on top-layer state vs target.
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
    trace = V4TrainingTrace(
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


def evaluate_v4_long_horizon_recall(
        cell: V4StackedCell,
        examples: Sequence[V4Example],
        *,
        use_skip_link: bool = True,
) -> float:
    """Mean cosine recall on the V4 stacked cell."""
    if not examples:
        return 0.0
    cos_sum = 0.0
    n = 0
    for ex in examples:
        sd = cell.state_dim
        layer_states = [
            [0.0] * sd for _ in range(cell.n_layers)
        ]
        skip_val = (
            list(ex.input_sequence[0])
            if use_skip_link and len(ex.input_sequence) > 0
            else None)
        for t in range(len(ex.input_sequence)):
            layer_states, _ = cell.step_value(
                prev_layer_states=layer_states,
                input_x=ex.input_sequence[t],
                skip_input=skip_val)
        top = layer_states[-1] if layer_states else [0.0] * sd
        cos_sum += _cosine(top, ex.target_state)
        n += 1
    return float(cos_sum) / float(max(1, n))


# =============================================================================
# Witness
# =============================================================================


@dataclasses.dataclass(frozen=True)
class PersistentLatentStateV4Witness:
    state_cid: str
    parent_state_cid: str
    role: str
    turn_index: int
    state_dim: int
    n_layers: int
    cell_cid: str
    skip_link_input_cid: str
    chain_walk_depth: int
    chain_cid: str
    update_gate_l1_sum: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_cid": str(self.state_cid),
            "parent_state_cid": str(self.parent_state_cid),
            "role": str(self.role),
            "turn_index": int(self.turn_index),
            "state_dim": int(self.state_dim),
            "n_layers": int(self.n_layers),
            "cell_cid": str(self.cell_cid),
            "skip_link_input_cid": str(self.skip_link_input_cid),
            "chain_walk_depth": int(self.chain_walk_depth),
            "chain_cid": str(self.chain_cid),
            "update_gate_l1_sum": float(round(
                self.update_gate_l1_sum, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w52_v4_witness",
            "witness": self.to_dict()})


def emit_persistent_v4_witness(
        *,
        state: PersistentLatentStateV4,
        cell: V4StackedCell,
        chain: PersistentLatentStateV4Chain,
        max_walk_depth: int = W52_DEFAULT_V4_MAX_CHAIN_WALK_DEPTH,
) -> PersistentLatentStateV4Witness:
    walk = chain.walk_from(
        state.cid(), max_depth=int(max_walk_depth))
    return PersistentLatentStateV4Witness(
        state_cid=str(state.cid()),
        parent_state_cid=str(state.parent_state_cid),
        role=str(state.role),
        turn_index=int(state.turn_index),
        state_dim=int(state.state_dim),
        n_layers=int(state.n_layers),
        cell_cid=str(cell.cid()),
        skip_link_input_cid=str(state.skip_link_input_cid),
        chain_walk_depth=int(len(walk)),
        chain_cid=str(chain.cid()),
        update_gate_l1_sum=float(state.update_gate_l1_sum),
    )


# =============================================================================
# Verifier
# =============================================================================

W52_V4_VERIFIER_FAILURE_MODES: tuple[str, ...] = (
    "w52_v4_schema_mismatch",
    "w52_v4_state_cid_mismatch",
    "w52_v4_cell_cid_mismatch",
    "w52_v4_chain_walk_depth_below_floor",
    "w52_v4_update_gate_pathology",
    "w52_v4_n_layers_mismatch",
    "w52_v4_skip_link_cid_mismatch",
)


def verify_persistent_v4_witness(
        witness: PersistentLatentStateV4Witness,
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
        failures.append("w52_v4_state_cid_mismatch")
    if (expected_cell_cid is not None
            and witness.cell_cid != expected_cell_cid):
        failures.append("w52_v4_cell_cid_mismatch")
    if (expected_n_layers is not None
            and witness.n_layers != int(expected_n_layers)):
        failures.append("w52_v4_n_layers_mismatch")
    if (min_chain_walk_depth is not None
            and witness.chain_walk_depth
            < int(min_chain_walk_depth)):
        failures.append(
            "w52_v4_chain_walk_depth_below_floor")
    if (max_gate_l1_pathology is not None
            and witness.update_gate_l1_sum
            > float(max_gate_l1_pathology)):
        failures.append("w52_v4_update_gate_pathology")
    return {
        "ok": (len(failures) == 0),
        "failures": failures,
        "witness_cid": witness.cid(),
    }


# =============================================================================
# Compromise helper (for the H20 falsifier)
# =============================================================================


def forge_v4_training_set(
        original: V4TrainingSet,
        *,
        seed: int = W47_DEFAULT_TRAIN_SEED,
) -> V4TrainingSet:
    """Adversarially scramble the targets."""
    rng = _DeterministicLCG(seed=int(seed))
    forged: list[V4Example] = []
    for ex in original.examples:
        forged_target = tuple(
            float(rng.next_uniform() * 2.0 - 1.0)
            for _ in range(len(ex.target_state)))
        forged.append(V4Example(
            input_sequence=ex.input_sequence,
            initial_state=ex.initial_state,
            target_state=forged_target))
    return V4TrainingSet(
        examples=tuple(forged),
        state_dim=original.state_dim,
        input_dim=original.input_dim)


__all__ = [
    "W52_PERSISTENT_V4_SCHEMA_VERSION",
    "W52_DEFAULT_V4_STATE_DIM",
    "W52_DEFAULT_V4_INPUT_DIM",
    "W52_DEFAULT_V4_N_LAYERS",
    "W52_DEFAULT_V4_MAX_CHAIN_WALK_DEPTH",
    "W52_V4_NO_PARENT_STATE",
    "W52_V4_VERIFIER_FAILURE_MODES",
    "V4StackedCell",
    "PersistentLatentStateV4",
    "PersistentLatentStateV4Chain",
    "PersistentLatentStateV4Witness",
    "V4Example",
    "V4TrainingSet",
    "V4TrainingTrace",
    "synthesize_v4_training_set",
    "fit_persistent_v4",
    "evaluate_v4_long_horizon_recall",
    "step_persistent_state_v4",
    "emit_persistent_v4_witness",
    "verify_persistent_v4_witness",
    "forge_v4_training_set",
]
