"""W51 M1 — Persistent Shared Latent State V3.

A trainable GRU-style persistent latent state that survives
across multiple turns, branches, and roles. The state at turn
``t`` is updated by a learned update gate over the prior state
and the current carrier:

    z_t = sigmoid(W_z · [s_{t-1}; x_t] + b_z)
    h_t = tanh(W_h · [s_{t-1}; x_t] + b_h)
    s_t = (1 - z_t) ⊙ s_{t-1} + z_t ⊙ h_t

A **cross-role mixer** sits on top: each role pulls a learned
convex combination of per-role views of the persistent state.
The mixer combines the per-role view with the team-shared
state via a learned blend coefficient in ``[0, 1]``.

Pure-Python only — reuses the W47 ``Variable`` + ``AdamOptimizer``
autograd engine, the W50 ``SharedLatentCarrierV2`` chain
abstraction.

Honest scope (do-not-overstate)
-------------------------------

This module does NOT alter transformer-internal hidden state,
KV cache bytes, attention weights, or embeddings. The
"persistent latent state" is a capsule-layer carrier evolving
across turns. The
``W51-L-NO-REAL-KV-CAP`` extends W50's no-real-KV cap
unchanged.

The H2 long-horizon gain claim is empirical on the R-100
length-12 regime; it does **not** claim a uniform persistent-
state advantage on real model behaviours.
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
    vdot,
    vmatmul,
    vmean,
    vsoftmax,
    vsum,
)
from .shared_latent_carrier import (
    W50_DEFAULT_CARRIER_DIM,
)


# =============================================================================
# Schema, defaults
# =============================================================================

W51_PERSISTENT_LATENT_SCHEMA_VERSION: str = (
    "coordpy.persistent_shared_latent.v1")

W51_DEFAULT_STATE_DIM: int = W50_DEFAULT_CARRIER_DIM
W51_DEFAULT_INPUT_DIM: int = W50_DEFAULT_CARRIER_DIM
W51_DEFAULT_MAX_CHAIN_WALK_DEPTH: int = 16
W51_DEFAULT_MIXER_BLEND_INIT: float = 0.5
W51_NO_PARENT_STATE: str = "no_parent_state"


# =============================================================================
# Canonicalisation helpers
# =============================================================================

def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str,
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


def _l2(values: Sequence[float]) -> float:
    return float(
        math.sqrt(sum(float(v) * float(v) for v in values)))


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
# PersistentLatentState — per-turn content-addressed payload
# =============================================================================

@dataclasses.dataclass(frozen=True)
class PersistentLatentState:
    """Per-turn persistent latent state.

    The state vector ``values`` is the GRU output at the turn.
    ``parent_state_cid`` is the prior turn's state CID
    (or ``W51_NO_PARENT_STATE`` at the chain root).
    ``mixer_cid`` is the cross-role mixer's CID.
    """

    turn_index: int
    role: str
    state_dim: int
    values: tuple[float, ...]
    parent_state_cid: str
    mixer_cid: str
    update_gate_l1: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_index": int(self.turn_index),
            "role": str(self.role),
            "state_dim": int(self.state_dim),
            "values": list(_round_floats(self.values)),
            "parent_state_cid": str(self.parent_state_cid),
            "mixer_cid": str(self.mixer_cid),
            "update_gate_l1": float(round(
                self.update_gate_l1, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w51_persistent_latent_state",
            "state": self.to_dict()})


# =============================================================================
# PersistentLatentStateChain — content-addressed chain index
# =============================================================================

@dataclasses.dataclass
class PersistentLatentStateChain:
    """A content-addressed index of seen persistent states.

    Indexed by state CID; chain-walk from any state recovers
    all ancestors via the parent_state_cid links.
    """

    states: dict[str, PersistentLatentState]

    @classmethod
    def empty(cls) -> "PersistentLatentStateChain":
        return cls(states={})

    def add(self, state: PersistentLatentState) -> None:
        self.states[state.cid()] = state

    def get(
            self, cid: str,
    ) -> PersistentLatentState | None:
        return self.states.get(str(cid))

    def walk_from(
            self, leaf_cid: str,
            *,
            max_depth: int = W51_DEFAULT_MAX_CHAIN_WALK_DEPTH,
    ) -> list[PersistentLatentState]:
        out: list[PersistentLatentState] = []
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
            "kind": "w51_persistent_latent_state_chain",
            "members": [
                {"cid": c, "state": s.to_dict()}
                for c, s in sorted(self.states.items())
            ],
        })


# =============================================================================
# Cross-role mixer
# =============================================================================

@dataclasses.dataclass
class CrossRoleMixer:
    """Per-role view + team-blend mixer.

    For each role ``r``, produces a learned linear projection
    of the state vector (a per-role view ``v_r``). The output
    is a learned convex combination of the per-role view and
    the team-shared state:

        out_r = (1 - blend) ⊙ s_team + blend ⊙ v_r

    where ``blend`` is a learned sigmoid in ``[0, 1]``.
    """

    state_dim: int
    role_universe: tuple[str, ...]
    w_view: ParamTensor   # (n_roles * state_dim, state_dim)
    b_view: ParamTensor   # (n_roles, state_dim)
    w_blend: ParamTensor  # (n_roles,) pre-sigmoid blend logits

    @classmethod
    def init(
            cls, *,
            role_universe: Sequence[str],
            state_dim: int = W51_DEFAULT_STATE_DIM,
            seed: int = W47_DEFAULT_TRAIN_SEED,
            init_scale: float = W47_DEFAULT_INIT_SCALE,
            blend_init: float = W51_DEFAULT_MIXER_BLEND_INIT,
    ) -> "CrossRoleMixer":
        roles = tuple(sorted({str(r) for r in role_universe}))
        n_roles = max(1, len(roles))
        rng = _DeterministicLCG(seed=int(seed))
        # Init view weights as identity per role + tiny noise.
        w_view = ParamTensor(
            shape=(int(n_roles) * int(state_dim),
                   int(state_dim)),
            values=[])
        vals = [0.0] * (int(n_roles) * int(state_dim)
                        * int(state_dim))
        for r in range(int(n_roles)):
            block_base = (
                r * int(state_dim) * int(state_dim))
            for i in range(int(state_dim)):
                vals[block_base + i * int(state_dim) + i] = 1.0
            for k in range(int(state_dim) * int(state_dim)):
                vals[block_base + k] += (
                    rng.next_uniform() - 0.5) * 0.01
        w_view.values = vals
        b_view = ParamTensor(
            shape=(int(n_roles), int(state_dim)),
            values=[0.0] * (
                int(n_roles) * int(state_dim)))
        # Blend logits initialised so sigmoid(logit) = blend_init.
        b = float(min(0.99, max(0.01, float(blend_init))))
        logit = math.log(b / (1.0 - b))
        w_blend = ParamTensor(
            shape=(int(n_roles),),
            values=[float(logit)] * int(n_roles))
        return cls(
            state_dim=int(state_dim),
            role_universe=roles,
            w_view=w_view,
            b_view=b_view,
            w_blend=w_blend)

    def params(self) -> list[ParamTensor]:
        return [self.w_view, self.b_view, self.w_blend]

    def _role_index(self, role: str) -> int:
        try:
            return int(self.role_universe.index(str(role)))
        except ValueError:
            return 0

    def view_value(
            self, *,
            role: str,
            state: Sequence[float],
    ) -> list[float]:
        r = self._role_index(role)
        sd = self.state_dim
        view = [0.0] * sd
        for i in range(sd):
            row_base = (r * sd * sd) + (i * sd)
            s = 0.0
            for j in range(sd):
                sj = float(state[j]) if j < len(state) else 0.0
                s += float(self.w_view.values[row_base + j]) * sj
            s += float(self.b_view.values[r * sd + i])
            view[i] = s
        return view

    def project_value(
            self, *,
            role: str,
            team_state: Sequence[float],
    ) -> tuple[list[float], float]:
        """Returns (per-role mixed state, blend coefficient)."""
        view = self.view_value(role=role, state=team_state)
        r = self._role_index(role)
        blend = float(_stable_sigmoid(
            float(self.w_blend.values[r])))
        out = [
            (1.0 - blend) * float(team_state[i] if i < len(team_state) else 0.0)
            + blend * float(view[i])
            for i in range(self.state_dim)
        ]
        return out, float(blend)

    def project_vars(
            self, *,
            role: str,
            team_state: Sequence[Variable],
    ) -> tuple[list[Variable], Variable]:
        r = self._role_index(role)
        w_view_vars = self.w_view.make_vars()
        b_view_vars = self.b_view.make_vars()
        w_blend_vars = self.w_blend.make_vars()
        sd = self.state_dim
        rows: list[list[Variable]] = []
        for i in range(sd):
            row_base = (r * sd * sd) + (i * sd)
            rows.append(list(
                w_view_vars[row_base:row_base + sd]))
        pre = vmatmul(rows, list(team_state))
        view = [
            pre[i] + b_view_vars[r * sd + i]
            for i in range(sd)
        ]
        blend = w_blend_vars[r].sigmoid()
        one_minus_blend = Variable(1.0) - blend
        out: list[Variable] = []
        ts = list(team_state)
        for i in range(sd):
            t = ts[i] if i < len(ts) else Variable(0.0)
            out.append(one_minus_blend * t + blend * view[i])
        return out, blend

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_dim": int(self.state_dim),
            "role_universe": list(self.role_universe),
            "w_view": self.w_view.to_dict(),
            "b_view": self.b_view.to_dict(),
            "w_blend": self.w_blend.to_dict(),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w51_cross_role_mixer",
            "mixer": self.to_dict()})


# =============================================================================
# Persistent State Cell (GRU)
# =============================================================================

@dataclasses.dataclass
class PersistentStateCell:
    """GRU-style trainable update cell.

    Maintains weights for the update gate ``W_z`` and the
    candidate state ``W_h``. Each step takes the prior state
    ``s_{t-1}`` and the current input carrier ``x_t`` and
    produces ``s_t``.
    """

    state_dim: int
    input_dim: int
    w_z: ParamTensor   # (state_dim, state_dim + input_dim)
    b_z: ParamTensor   # (state_dim,)
    w_h: ParamTensor   # (state_dim, state_dim + input_dim)
    b_h: ParamTensor   # (state_dim,)

    @classmethod
    def init(
            cls, *,
            state_dim: int = W51_DEFAULT_STATE_DIM,
            input_dim: int = W51_DEFAULT_INPUT_DIM,
            seed: int = W47_DEFAULT_TRAIN_SEED,
            init_scale: float = W47_DEFAULT_INIT_SCALE,
    ) -> "PersistentStateCell":
        cat_dim = int(state_dim) + int(input_dim)
        rng = _DeterministicLCG(seed=int(seed))
        w_z = ParamTensor(
            shape=(int(state_dim), int(cat_dim)), values=[])
        w_z.init_seed(
            seed=int(rng.next_uniform() * (1 << 30)),
            scale=float(init_scale))
        # Bias initialised slightly negative so update gate starts
        # closed (state persists by default — important for trivial
        # passthrough).
        b_z = ParamTensor(
            shape=(int(state_dim),),
            values=[-1.0] * int(state_dim))
        w_h = ParamTensor(
            shape=(int(state_dim), int(cat_dim)), values=[])
        w_h.init_seed(
            seed=int(rng.next_uniform() * (1 << 30)),
            scale=float(init_scale))
        b_h = ParamTensor(
            shape=(int(state_dim),),
            values=[0.0] * int(state_dim))
        return cls(
            state_dim=int(state_dim),
            input_dim=int(input_dim),
            w_z=w_z, b_z=b_z, w_h=w_h, b_h=b_h)

    def params(self) -> list[ParamTensor]:
        return [self.w_z, self.b_z, self.w_h, self.b_h]

    def _concat(
            self, prev_state: Sequence[float],
            input_x: Sequence[float],
    ) -> list[float]:
        out: list[float] = []
        for i in range(self.state_dim):
            out.append(
                float(prev_state[i]) if i < len(prev_state)
                else 0.0)
        for j in range(self.input_dim):
            out.append(
                float(input_x[j]) if j < len(input_x) else 0.0)
        return out

    def _concat_vars(
            self, prev_state: Sequence[Variable],
            input_x: Sequence[Variable],
    ) -> list[Variable]:
        out: list[Variable] = []
        for i in range(self.state_dim):
            out.append(
                prev_state[i] if i < len(prev_state)
                else Variable(0.0))
        for j in range(self.input_dim):
            out.append(
                input_x[j] if j < len(input_x)
                else Variable(0.0))
        return out

    def step_value(
            self, *,
            prev_state: Sequence[float],
            input_x: Sequence[float],
    ) -> tuple[list[float], list[float], list[float]]:
        """Returns (next_state, update_gate, candidate)."""
        cat = self._concat(prev_state, input_x)
        cat_dim = self.state_dim + self.input_dim
        sd = self.state_dim
        z = [0.0] * sd
        h_cand = [0.0] * sd
        for r in range(sd):
            base = r * cat_dim
            sz = 0.0
            sh = 0.0
            for j in range(cat_dim):
                wj_z = float(self.w_z.values[base + j])
                wj_h = float(self.w_h.values[base + j])
                cj = float(cat[j])
                sz += wj_z * cj
                sh += wj_h * cj
            sz += float(self.b_z.values[r])
            sh += float(self.b_h.values[r])
            z[r] = float(_stable_sigmoid(sz))
            h_cand[r] = math.tanh(sh)
        next_state = [
            (1.0 - z[i]) * (
                float(prev_state[i]) if i < len(prev_state)
                else 0.0)
            + z[i] * h_cand[i]
            for i in range(sd)
        ]
        return next_state, z, h_cand

    def step_vars(
            self, *,
            prev_state: Sequence[Variable],
            input_x: Sequence[Variable],
    ) -> tuple[list[Variable], list[Variable]]:
        """Returns (next_state, update_gate)."""
        cat = self._concat_vars(prev_state, input_x)
        w_z_vars = self.w_z.make_vars()
        b_z_vars = self.b_z.make_vars()
        w_h_vars = self.w_h.make_vars()
        b_h_vars = self.b_h.make_vars()
        cat_dim = self.state_dim + self.input_dim
        sd = self.state_dim
        rows_z: list[list[Variable]] = []
        rows_h: list[list[Variable]] = []
        for r in range(sd):
            base = r * cat_dim
            rows_z.append(list(w_z_vars[base:base + cat_dim]))
            rows_h.append(list(w_h_vars[base:base + cat_dim]))
        pre_z = vmatmul(rows_z, cat)
        pre_h = vmatmul(rows_h, cat)
        z_vars = [
            (pre_z[i] + b_z_vars[i]).sigmoid()
            for i in range(sd)
        ]
        h_cand_vars = [
            (pre_h[i] + b_h_vars[i]).tanh()
            for i in range(sd)
        ]
        ps = list(prev_state)
        next_state: list[Variable] = []
        for i in range(sd):
            p = ps[i] if i < len(ps) else Variable(0.0)
            one_minus_z = Variable(1.0) - z_vars[i]
            next_state.append(
                one_minus_z * p + z_vars[i] * h_cand_vars[i])
        return next_state, z_vars

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_dim": int(self.state_dim),
            "input_dim": int(self.input_dim),
            "w_z": self.w_z.to_dict(),
            "b_z": self.b_z.to_dict(),
            "w_h": self.w_h.to_dict(),
            "b_h": self.b_h.to_dict(),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w51_persistent_state_cell",
            "cell": self.to_dict()})


# =============================================================================
# Training set + fit
# =============================================================================

@dataclasses.dataclass(frozen=True)
class PersistentStateExample:
    """One training example for the persistent state cell.

    The cell receives the sequence of inputs ``input_sequence``
    and is supervised on the final state's distance to
    ``target_state`` (a fixed early-turn feature the cell must
    remember).
    """

    input_sequence: tuple[tuple[float, ...], ...]
    initial_state: tuple[float, ...]
    target_state: tuple[float, ...]


@dataclasses.dataclass(frozen=True)
class PersistentStateTrainingSet:
    examples: tuple[PersistentStateExample, ...]
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
            "kind": "w51_persistent_state_training_set",
            "set": self.to_dict()})


def synthesize_persistent_state_training_set(
        *,
        n_sequences: int = 8,
        sequence_length: int = 12,
        state_dim: int = W51_DEFAULT_STATE_DIM,
        input_dim: int = W51_DEFAULT_INPUT_DIM,
        seed: int = W47_DEFAULT_TRAIN_SEED,
        signal_position: int = 0,
) -> PersistentStateTrainingSet:
    """Synthesise a deterministic long-horizon training set.

    At ``signal_position`` the input carries a load-bearing
    signal (random vector); every subsequent input is small
    noise. The target_state at the final turn is the signal,
    i.e. the cell must remember a long-ago feature.
    """
    rng = _DeterministicLCG(seed=int(seed))
    examples: list[PersistentStateExample] = []
    for _ in range(int(n_sequences)):
        seq: list[tuple[float, ...]] = []
        signal: list[float] = [
            float(rng.next_uniform() * 2.0 - 1.0)
            for _ in range(int(input_dim))
        ]
        for t in range(int(sequence_length)):
            if t == int(signal_position):
                # Signal turn carries the load-bearing vector.
                seq.append(tuple(signal))
            else:
                # Noise turn carries small-magnitude noise.
                noise = [
                    float(rng.next_uniform() - 0.5) * 0.05
                    for _ in range(int(input_dim))
                ]
                seq.append(tuple(noise))
        initial = [0.0] * int(state_dim)
        # Target = signal padded/truncated to state_dim.
        target = list(signal)[:int(state_dim)]
        while len(target) < int(state_dim):
            target.append(0.0)
        examples.append(PersistentStateExample(
            input_sequence=tuple(seq),
            initial_state=tuple(initial),
            target_state=tuple(target),
        ))
    return PersistentStateTrainingSet(
        examples=tuple(examples),
        state_dim=int(state_dim),
        input_dim=int(input_dim))


@dataclasses.dataclass(frozen=True)
class PersistentStateTrainingTrace:
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
            "final_grad_norm": float(
                round(self.final_grad_norm, 12)),
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
            "kind": "w51_persistent_state_training_trace",
            "trace": self.to_dict()})


def fit_persistent_state_cell(
        training_set: PersistentStateTrainingSet,
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
) -> tuple[PersistentStateCell, PersistentStateTrainingTrace]:
    """Fit the GRU cell via truncated BPTT + Adam SGD.

    Pure-Python autograd is expensive, so we truncate the
    backward graph to the last ``truncate_bptt`` steps of each
    sequence; the final state of those last steps is supervised
    against ``target_state`` via MSE.
    """
    cell = PersistentStateCell.init(
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
        # Forward through sequence WITHOUT tape for early turns,
        # then turn on tape for the last ``truncate_bptt`` turns.
        seq_len = len(ex.input_sequence)
        stop_truncate = max(0, seq_len - int(truncate_bptt))
        # Run early turns without autograd.
        s_val = list(ex.initial_state)
        for t in range(stop_truncate):
            s_val, _, _ = cell.step_value(
                prev_state=s_val,
                input_x=ex.input_sequence[t])
        # Run last ``truncate_bptt`` turns with autograd.
        s_var = [Variable(float(v)) for v in s_val]
        for t in range(stop_truncate, seq_len):
            x_var = [
                Variable(float(v))
                for v in ex.input_sequence[t]
            ]
            s_var, _ = cell.step_vars(
                prev_state=s_var, input_x=x_var)
        # Loss = MSE(s_var, target_state)
        terms = []
        for j in range(len(ex.target_state)):
            t = Variable(float(ex.target_state[j]))
            o = s_var[j] if j < len(s_var) else Variable(0.0)
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
    trace = PersistentStateTrainingTrace(
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


def evaluate_long_horizon_recall(
        cell: PersistentStateCell,
        examples: Sequence[PersistentStateExample],
) -> float:
    """Mean cosine similarity between the cell's final state
    and the target state across the example set.
    """
    if not examples:
        return 0.0
    cos_sum = 0.0
    n = 0
    for ex in examples:
        s = list(ex.initial_state)
        for t in range(len(ex.input_sequence)):
            s, _, _ = cell.step_value(
                prev_state=s,
                input_x=ex.input_sequence[t])
        cos_sum += _cosine(s, ex.target_state)
        n += 1
    return float(cos_sum) / float(max(1, n))


# =============================================================================
# Witnesses
# =============================================================================

@dataclasses.dataclass(frozen=True)
class PersistentLatentStateWitness:
    """Sealed per-turn persistent latent state witness."""

    state_cid: str
    parent_state_cid: str
    role: str
    turn_index: int
    state_dim: int
    cell_cid: str
    mixer_cid: str
    update_gate_l1: float
    chain_walk_depth: int
    chain_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_cid": str(self.state_cid),
            "parent_state_cid": str(self.parent_state_cid),
            "role": str(self.role),
            "turn_index": int(self.turn_index),
            "state_dim": int(self.state_dim),
            "cell_cid": str(self.cell_cid),
            "mixer_cid": str(self.mixer_cid),
            "update_gate_l1": float(round(
                self.update_gate_l1, 12)),
            "chain_walk_depth": int(self.chain_walk_depth),
            "chain_cid": str(self.chain_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w51_persistent_latent_state_witness",
            "witness": self.to_dict()})


def emit_persistent_latent_state_witness(
        *,
        state: PersistentLatentState,
        cell: PersistentStateCell,
        mixer: CrossRoleMixer | None,
        chain: PersistentLatentStateChain,
        max_walk_depth: int = W51_DEFAULT_MAX_CHAIN_WALK_DEPTH,
) -> PersistentLatentStateWitness:
    walk = chain.walk_from(
        state.cid(), max_depth=int(max_walk_depth))
    return PersistentLatentStateWitness(
        state_cid=str(state.cid()),
        parent_state_cid=str(state.parent_state_cid),
        role=str(state.role),
        turn_index=int(state.turn_index),
        state_dim=int(state.state_dim),
        cell_cid=str(cell.cid()),
        mixer_cid=str(mixer.cid()) if mixer is not None else "",
        update_gate_l1=float(state.update_gate_l1),
        chain_walk_depth=int(len(walk)),
        chain_cid=str(chain.cid()),
    )


# =============================================================================
# Step helper (for the W51 forward pass)
# =============================================================================

def step_persistent_latent_state(
        *,
        cell: PersistentStateCell,
        mixer: CrossRoleMixer | None,
        prev_state_values: Sequence[float],
        carrier_values: Sequence[float],
        turn_index: int,
        role: str,
        parent_state_cid: str,
        state_dim: int = W51_DEFAULT_STATE_DIM,
) -> PersistentLatentState:
    """Compute one forward step of the persistent state.

    If ``mixer`` is provided, the output state is mixed through
    the cross-role mixer. The returned PersistentLatentState
    carries the update gate L1 norm so a verifier can detect a
    "stale-state" pathology where the gate never opens.
    """
    next_state, z, _ = cell.step_value(
        prev_state=prev_state_values, input_x=carrier_values)
    if mixer is not None:
        next_state, _ = mixer.project_value(
            role=str(role), team_state=next_state)
    gate_l1 = float(sum(abs(float(g)) for g in z))
    return PersistentLatentState(
        turn_index=int(turn_index),
        role=str(role),
        state_dim=int(state_dim),
        values=tuple(_round_floats(next_state)),
        parent_state_cid=str(parent_state_cid),
        mixer_cid=(
            str(mixer.cid()) if mixer is not None else ""),
        update_gate_l1=float(round(gate_l1, 12)),
    )


# =============================================================================
# Verifier
# =============================================================================

W51_PERSISTENT_LATENT_VERIFIER_FAILURE_MODES: tuple[str, ...] = (
    "w51_persistent_state_schema_mismatch",
    "w51_persistent_state_cid_mismatch",
    "w51_persistent_state_cell_cid_mismatch",
    "w51_persistent_state_mixer_cid_mismatch",
    "w51_persistent_state_chain_walk_depth_below_floor",
    "w51_persistent_state_update_gate_pathology",
)


def verify_persistent_latent_state_witness(
        witness: PersistentLatentStateWitness,
        *,
        expected_state_cid: str | None = None,
        expected_cell_cid: str | None = None,
        expected_mixer_cid: str | None = None,
        min_chain_walk_depth: int | None = None,
        max_gate_l1_pathology: float | None = None,
) -> dict[str, Any]:
    """Verify a sealed persistent-latent-state witness.

    Returns a dict with ``ok`` plus an enumerated failure list.
    """
    failures: list[str] = []
    if (expected_state_cid is not None
            and witness.state_cid != expected_state_cid):
        failures.append(
            "w51_persistent_state_cid_mismatch")
    if (expected_cell_cid is not None
            and witness.cell_cid != expected_cell_cid):
        failures.append(
            "w51_persistent_state_cell_cid_mismatch")
    if (expected_mixer_cid is not None
            and witness.mixer_cid != expected_mixer_cid):
        failures.append(
            "w51_persistent_state_mixer_cid_mismatch")
    if (min_chain_walk_depth is not None
            and witness.chain_walk_depth
            < int(min_chain_walk_depth)):
        failures.append(
            "w51_persistent_state_chain_walk_depth_below_floor")
    if (max_gate_l1_pathology is not None
            and witness.update_gate_l1
            > float(max_gate_l1_pathology)):
        failures.append(
            "w51_persistent_state_update_gate_pathology")
    return {
        "ok": (len(failures) == 0),
        "failures": failures,
        "witness_cid": witness.cid(),
    }


# =============================================================================
# Compromise helper (for the H10 falsifier)
# =============================================================================

def forge_persistent_state_training_set(
        original: PersistentStateTrainingSet,
        *,
        seed: int = W47_DEFAULT_TRAIN_SEED,
) -> PersistentStateTrainingSet:
    """Adversarially scramble the long-horizon targets.

    Produces a forged training set with random noise targets —
    the W51 cell trained on these cannot recover the signal.
    """
    rng = _DeterministicLCG(seed=int(seed))
    forged: list[PersistentStateExample] = []
    for ex in original.examples:
        forged_target = tuple(
            float(rng.next_uniform() * 2.0 - 1.0)
            for _ in range(len(ex.target_state)))
        forged.append(PersistentStateExample(
            input_sequence=ex.input_sequence,
            initial_state=ex.initial_state,
            target_state=forged_target))
    return PersistentStateTrainingSet(
        examples=tuple(forged),
        state_dim=original.state_dim,
        input_dim=original.input_dim)


__all__ = [
    "W51_PERSISTENT_LATENT_SCHEMA_VERSION",
    "W51_DEFAULT_STATE_DIM",
    "W51_DEFAULT_INPUT_DIM",
    "W51_DEFAULT_MAX_CHAIN_WALK_DEPTH",
    "W51_DEFAULT_MIXER_BLEND_INIT",
    "W51_NO_PARENT_STATE",
    "W51_PERSISTENT_LATENT_VERIFIER_FAILURE_MODES",
    "PersistentLatentState",
    "PersistentLatentStateChain",
    "PersistentStateCell",
    "CrossRoleMixer",
    "PersistentStateExample",
    "PersistentStateTrainingSet",
    "PersistentStateTrainingTrace",
    "PersistentLatentStateWitness",
    "synthesize_persistent_state_training_set",
    "fit_persistent_state_cell",
    "evaluate_long_horizon_recall",
    "step_persistent_latent_state",
    "emit_persistent_latent_state_witness",
    "verify_persistent_latent_state_witness",
    "forge_persistent_state_training_set",
]
