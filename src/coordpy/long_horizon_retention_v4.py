"""W52 M5 — Long-Horizon Reconstruction V4 (three heads).

Extends W51's :class:`LongHorizonReconstructionV3Head` with a
third (cycle) head:

* **causal head** — recovers turn ``t-k`` features (k ∈ {1..12})
* **branch head** — branch-conditioned residual
* **cycle head** — cycle-stationary invariants (k-independent)

V4 supports ``max_k=12`` (vs W51's max_k=8) for longer-horizon
reconstruction. Witness records MSE for each k per head plus a
``k ∈ {1..24}`` degradation curve.

Pure-Python only — reuses the W47 ``Variable`` autograd engine.
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


# =============================================================================
# Schema, defaults
# =============================================================================

W52_LHR_V4_SCHEMA_VERSION: str = (
    "coordpy.long_horizon_retention_v4.v1")

W52_DEFAULT_LHR_V4_FLAT_FEATURE_DIM: int = 4
W52_DEFAULT_LHR_V4_HIDDEN_DIM: int = 32
W52_DEFAULT_LHR_V4_MAX_K: int = 12
W52_DEFAULT_LHR_V4_N_BRANCHES: int = 2
W52_DEFAULT_LHR_V4_N_CYCLES: int = 2
W52_DEFAULT_LHR_V4_DROPOFF_PROBE_KS: tuple[int, ...] = tuple(
    range(1, 25))


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


def _mse(a: Sequence[float], b: Sequence[float]) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    s = 0.0
    for i in range(n):
        d = float(a[i]) - float(b[i])
        s += d * d
    return float(s) / float(n)


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
# V4 three-headed reconstruction head
# =============================================================================


@dataclasses.dataclass
class LongHorizonReconstructionV4Head:
    """Three-headed reconstruction: causal + branch + cycle.

    Input = ``[carrier; k_one_hot; branch_one_hot; cycle_one_hot]``
    of dimension ``carrier_dim + max_k + n_branches + n_cycles``.

    Output = sum of (causal, branch_residual, cycle_residual).
    """

    in_dim: int
    hidden_dim: int
    out_dim: int
    max_k: int
    n_branches: int
    n_cycles: int
    w1: ParamTensor    # (hidden_dim, in_dim)
    b1: ParamTensor    # (hidden_dim,)
    w2c: ParamTensor   # (out_dim, hidden_dim) — causal
    b2c: ParamTensor   # (out_dim,)
    w2b: ParamTensor   # (out_dim, hidden_dim) — branch
    b2b: ParamTensor   # (out_dim,)
    w2y: ParamTensor   # (out_dim, hidden_dim) — cycle
    b2y: ParamTensor   # (out_dim,)

    @classmethod
    def init(
            cls, *,
            carrier_dim: int,
            hidden_dim: int = W52_DEFAULT_LHR_V4_HIDDEN_DIM,
            out_dim: int = W52_DEFAULT_LHR_V4_FLAT_FEATURE_DIM,
            max_k: int = W52_DEFAULT_LHR_V4_MAX_K,
            n_branches: int = W52_DEFAULT_LHR_V4_N_BRANCHES,
            n_cycles: int = W52_DEFAULT_LHR_V4_N_CYCLES,
            seed: int = W47_DEFAULT_TRAIN_SEED,
            init_scale: float = W47_DEFAULT_INIT_SCALE,
    ) -> "LongHorizonReconstructionV4Head":
        in_dim = (int(carrier_dim) + int(max_k)
                  + int(n_branches) + int(n_cycles))
        rng = _DeterministicLCG(seed=int(seed))
        w1 = ParamTensor(
            shape=(int(hidden_dim), int(in_dim)), values=[])
        w1.init_seed(
            seed=int(rng.next_uniform() * (1 << 30)),
            scale=float(init_scale))
        b1 = ParamTensor(
            shape=(int(hidden_dim),),
            values=[0.0] * int(hidden_dim))
        w2c = ParamTensor(
            shape=(int(out_dim), int(hidden_dim)), values=[])
        w2c.init_seed(
            seed=int(rng.next_uniform() * (1 << 30)),
            scale=float(init_scale))
        b2c = ParamTensor(
            shape=(int(out_dim),),
            values=[0.0] * int(out_dim))
        w2b = ParamTensor(
            shape=(int(out_dim), int(hidden_dim)), values=[])
        w2b.init_seed(
            seed=int(rng.next_uniform() * (1 << 30)),
            scale=float(init_scale))
        b2b = ParamTensor(
            shape=(int(out_dim),),
            values=[0.0] * int(out_dim))
        w2y = ParamTensor(
            shape=(int(out_dim), int(hidden_dim)), values=[])
        w2y.init_seed(
            seed=int(rng.next_uniform() * (1 << 30)),
            scale=float(init_scale))
        b2y = ParamTensor(
            shape=(int(out_dim),),
            values=[0.0] * int(out_dim))
        return cls(
            in_dim=int(in_dim),
            hidden_dim=int(hidden_dim),
            out_dim=int(out_dim),
            max_k=int(max_k),
            n_branches=int(n_branches),
            n_cycles=int(n_cycles),
            w1=w1, b1=b1,
            w2c=w2c, b2c=b2c,
            w2b=w2b, b2b=b2b,
            w2y=w2y, b2y=b2y)

    def params(self) -> list[ParamTensor]:
        return [
            self.w1, self.b1,
            self.w2c, self.b2c,
            self.w2b, self.b2b,
            self.w2y, self.b2y]

    @property
    def carrier_dim(self) -> int:
        return (int(self.in_dim) - int(self.max_k)
                - int(self.n_branches) - int(self.n_cycles))

    def _make_input(
            self,
            carrier: Sequence[float],
            k: int, branch: int, cycle: int,
    ) -> list[float]:
        cd = self.carrier_dim
        out = list(carrier)[:cd]
        while len(out) < cd:
            out.append(0.0)
        for kk in range(1, self.max_k + 1):
            out.append(1.0 if (int(k) == kk) else 0.0)
        for bb in range(self.n_branches):
            out.append(1.0 if (int(branch) == bb) else 0.0)
        for cc in range(self.n_cycles):
            out.append(1.0 if (int(cycle) == cc) else 0.0)
        return out

    def _make_input_vars(
            self,
            carrier: Sequence[Variable],
            k: int, branch: int, cycle: int,
    ) -> list[Variable]:
        cd = self.carrier_dim
        out = list(carrier)[:cd]
        while len(out) < cd:
            out.append(Variable(0.0))
        for kk in range(1, self.max_k + 1):
            out.append(Variable(1.0 if (int(k) == kk) else 0.0))
        for bb in range(self.n_branches):
            out.append(
                Variable(1.0 if (int(branch) == bb) else 0.0))
        for cc in range(self.n_cycles):
            out.append(
                Variable(1.0 if (int(cycle) == cc) else 0.0))
        return out

    def forward_value(
            self, *,
            carrier: Sequence[float],
            k: int,
            branch: int = 0,
            cycle: int = 0,
    ) -> tuple[list[float], list[float], list[float], list[float]]:
        """Returns (final, causal, branch_res, cycle_res)."""
        x = self._make_input(
            carrier, int(k), int(branch), int(cycle))
        hidden = [0.0] * self.hidden_dim
        for r in range(self.hidden_dim):
            base = r * self.in_dim
            s = 0.0
            for j in range(self.in_dim):
                if j < len(x):
                    s += float(self.w1.values[base + j]) \
                        * float(x[j])
            s += float(self.b1.values[r])
            hidden[r] = math.tanh(s)
        causal = [0.0] * self.out_dim
        branch_res = [0.0] * self.out_dim
        cycle_res = [0.0] * self.out_dim
        for r in range(self.out_dim):
            base = r * self.hidden_dim
            sc = 0.0
            sb = 0.0
            sy = 0.0
            for j in range(self.hidden_dim):
                hj = float(hidden[j])
                sc += float(self.w2c.values[base + j]) * hj
                sb += float(self.w2b.values[base + j]) * hj
                sy += float(self.w2y.values[base + j]) * hj
            sc += float(self.b2c.values[r])
            sb += float(self.b2b.values[r])
            sy += float(self.b2y.values[r])
            causal[r] = sc
            branch_res[r] = sb
            cycle_res[r] = sy
        final = [
            causal[r] + branch_res[r] + cycle_res[r]
            for r in range(self.out_dim)
        ]
        return final, causal, branch_res, cycle_res

    def forward_vars(
            self, *,
            carrier: Sequence[Variable],
            k: int,
            branch: int = 0,
            cycle: int = 0,
    ) -> tuple[list[Variable], list[Variable], list[Variable]]:
        x_vars = self._make_input_vars(
            carrier, int(k), int(branch), int(cycle))
        w1_vars = self.w1.make_vars()
        b1_vars = self.b1.make_vars()
        w2c_vars = self.w2c.make_vars()
        b2c_vars = self.b2c.make_vars()
        w2b_vars = self.w2b.make_vars()
        b2b_vars = self.b2b.make_vars()
        w2y_vars = self.w2y.make_vars()
        b2y_vars = self.b2y.make_vars()
        rows_h: list[list[Variable]] = []
        for r in range(self.hidden_dim):
            base = r * self.in_dim
            rows_h.append(list(
                w1_vars[base:base + self.in_dim]))
        pre_h = vmatmul(rows_h, x_vars)
        hidden = [
            (pre_h[i] + b1_vars[i]).tanh()
            for i in range(self.hidden_dim)
        ]
        rows_c: list[list[Variable]] = []
        rows_b: list[list[Variable]] = []
        rows_y: list[list[Variable]] = []
        for r in range(self.out_dim):
            base = r * self.hidden_dim
            rows_c.append(list(
                w2c_vars[base:base + self.hidden_dim]))
            rows_b.append(list(
                w2b_vars[base:base + self.hidden_dim]))
            rows_y.append(list(
                w2y_vars[base:base + self.hidden_dim]))
        pre_c = vmatmul(rows_c, hidden)
        pre_b = vmatmul(rows_b, hidden)
        pre_y = vmatmul(rows_y, hidden)
        causal = [
            pre_c[i] + b2c_vars[i] for i in range(self.out_dim)
        ]
        branch_res = [
            pre_b[i] + b2b_vars[i] for i in range(self.out_dim)
        ]
        cycle_res = [
            pre_y[i] + b2y_vars[i] for i in range(self.out_dim)
        ]
        final = [
            causal[i] + branch_res[i] + cycle_res[i]
            for i in range(self.out_dim)
        ]
        return final, causal, cycle_res

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": str(
                W52_LHR_V4_SCHEMA_VERSION),
            "in_dim": int(self.in_dim),
            "hidden_dim": int(self.hidden_dim),
            "out_dim": int(self.out_dim),
            "max_k": int(self.max_k),
            "n_branches": int(self.n_branches),
            "n_cycles": int(self.n_cycles),
            "w1": self.w1.to_dict(),
            "b1": self.b1.to_dict(),
            "w2c": self.w2c.to_dict(),
            "b2c": self.b2c.to_dict(),
            "w2b": self.w2b.to_dict(),
            "b2b": self.b2b.to_dict(),
            "w2y": self.w2y.to_dict(),
            "b2y": self.b2y.to_dict(),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w52_long_horizon_reconstruction_v4_head",
            "head": self.to_dict()})


# =============================================================================
# Training set + fit
# =============================================================================


@dataclasses.dataclass(frozen=True)
class LongHorizonV4Example:
    carrier: tuple[float, ...]
    k: int
    branch_index: int
    cycle_index: int
    target_features: tuple[float, ...]


@dataclasses.dataclass(frozen=True)
class LongHorizonV4TrainingSet:
    examples: tuple[LongHorizonV4Example, ...]
    carrier_dim: int
    out_dim: int
    max_k: int
    n_branches: int
    n_cycles: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "carrier_dim": int(self.carrier_dim),
            "out_dim": int(self.out_dim),
            "max_k": int(self.max_k),
            "n_branches": int(self.n_branches),
            "n_cycles": int(self.n_cycles),
            "examples": [
                {"carrier": list(e.carrier),
                 "k": int(e.k),
                 "branch_index": int(e.branch_index),
                 "cycle_index": int(e.cycle_index),
                 "target_features": list(e.target_features)}
                for e in self.examples],
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w52_long_horizon_v4_training_set",
            "set": self.to_dict()})


def synthesize_long_horizon_v4_training_set(
        *,
        n_sequences: int = 6,
        sequence_length: int = 16,
        carrier_dim: int | None = None,
        out_dim: int = W52_DEFAULT_LHR_V4_FLAT_FEATURE_DIM,
        max_k: int = W52_DEFAULT_LHR_V4_MAX_K,
        n_branches: int = W52_DEFAULT_LHR_V4_N_BRANCHES,
        n_cycles: int = W52_DEFAULT_LHR_V4_N_CYCLES,
        seed: int = W47_DEFAULT_TRAIN_SEED,
        carrier_decay: float = 0.0,
) -> LongHorizonV4TrainingSet:
    """Synthesise a V4 training set with cycle-conditioned shift."""
    actual_carrier_dim = int(
        carrier_dim
        if carrier_dim is not None
        else int(max_k) * int(out_dim))
    if actual_carrier_dim < int(max_k) * int(out_dim):
        raise ValueError(
            "carrier_dim must be >= max_k * out_dim")
    rng = _DeterministicLCG(seed=int(seed))
    examples: list[LongHorizonV4Example] = []
    for s_idx in range(int(n_sequences)):
        branch = int(s_idx % int(n_branches))
        cycle = int((s_idx // int(n_branches))
                    % int(n_cycles))
        flats: list[list[float]] = []
        for _ in range(int(sequence_length)):
            v = [
                float(rng.next_uniform() * 2.0 - 1.0)
                for _ in range(int(out_dim))
            ]
            for j in range(int(out_dim)):
                v[j] += float(branch) * 0.1
                v[j] += float(cycle) * 0.05
            flats.append(v)
        carriers: list[list[float]] = []
        for t in range(int(sequence_length)):
            c = [0.0] * int(actual_carrier_dim)
            for k in range(1, int(max_k) + 1):
                if t - k < 0:
                    continue
                base = (k - 1) * int(out_dim)
                scale = float(
                    math.exp(-float(carrier_decay)
                              * float(k - 1)))
                for j in range(int(out_dim)):
                    c[base + j] = (
                        float(flats[t - k][j]) * scale)
            carriers.append(c)
        for t in range(int(sequence_length)):
            for k in range(1, int(max_k) + 1):
                if t - k < 0:
                    continue
                examples.append(LongHorizonV4Example(
                    carrier=tuple(carriers[t]),
                    k=int(k),
                    branch_index=int(branch),
                    cycle_index=int(cycle),
                    target_features=tuple(flats[t - k])))
    return LongHorizonV4TrainingSet(
        examples=tuple(examples),
        carrier_dim=int(actual_carrier_dim),
        out_dim=int(out_dim),
        max_k=int(max_k),
        n_branches=int(n_branches),
        n_cycles=int(n_cycles))


@dataclasses.dataclass(frozen=True)
class LongHorizonV4TrainingTrace:
    seed: int
    n_steps: int
    final_loss: float
    final_grad_norm: float
    loss_head: tuple[float, ...]
    loss_tail: tuple[float, ...]
    training_set_cid: str
    final_head_cid: str
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
            "final_head_cid": str(self.final_head_cid),
            "diverged": bool(self.diverged),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w52_long_horizon_v4_training_trace",
            "trace": self.to_dict()})


def fit_long_horizon_v4(
        training_set: LongHorizonV4TrainingSet,
        *,
        hidden_dim: int = W52_DEFAULT_LHR_V4_HIDDEN_DIM,
        n_steps: int = 288,
        learning_rate: float = 0.005,
        seed: int = W47_DEFAULT_TRAIN_SEED,
        init_scale: float = W47_DEFAULT_INIT_SCALE,
        beta1: float = W47_DEFAULT_BETA1,
        beta2: float = W47_DEFAULT_BETA2,
        eps: float = W47_DEFAULT_EPS,
        grad_clip: float = W47_DEFAULT_GRAD_CLIP,
        history_head: int = 6,
        history_tail: int = 6,
) -> tuple[LongHorizonReconstructionV4Head,
           LongHorizonV4TrainingTrace]:
    """Fit V4 head via Adam SGD on MSE loss across all examples."""
    head = LongHorizonReconstructionV4Head.init(
        carrier_dim=int(training_set.carrier_dim),
        hidden_dim=int(hidden_dim),
        out_dim=int(training_set.out_dim),
        max_k=int(training_set.max_k),
        n_branches=int(training_set.n_branches),
        n_cycles=int(training_set.n_cycles),
        seed=int(seed),
        init_scale=float(init_scale))
    optim = AdamOptimizer(
        learning_rate=float(learning_rate),
        beta1=float(beta1), beta2=float(beta2),
        eps=float(eps), grad_clip=float(grad_clip))
    trainable = head.params()
    loss_history: list[float] = []
    grad_norm_history: list[float] = []
    diverged = False
    n = max(1, len(training_set.examples))
    for step in range(int(n_steps)):
        for p in trainable:
            p.make_vars()
        idx = step % n
        ex = training_set.examples[idx]
        c_vars = [Variable(float(v)) for v in ex.carrier]
        out_vars, _, _ = head.forward_vars(
            carrier=c_vars,
            k=int(ex.k),
            branch=int(ex.branch_index),
            cycle=int(ex.cycle_index))
        terms = []
        for j in range(len(ex.target_features)):
            t = Variable(float(ex.target_features[j]))
            o = out_vars[j] if j < len(out_vars) else Variable(0.0)
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
    trace = LongHorizonV4TrainingTrace(
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
        final_head_cid=str(head.cid()),
        diverged=bool(diverged),
    )
    return head, trace


def evaluate_long_horizon_v4_mse_at_k(
        head: LongHorizonReconstructionV4Head,
        examples: Sequence[LongHorizonV4Example],
        k: int,
) -> float:
    if not examples:
        return 0.0
    mse_sum = 0.0
    n = 0
    for ex in examples:
        if int(ex.k) != int(k):
            continue
        pred, _, _, _ = head.forward_value(
            carrier=ex.carrier,
            k=int(ex.k),
            branch=int(ex.branch_index),
            cycle=int(ex.cycle_index))
        mse_sum += _mse(pred, ex.target_features)
        n += 1
    return float(mse_sum) / float(max(1, n))


def evaluate_long_horizon_v4_mse_curve(
        head: LongHorizonReconstructionV4Head,
        examples: Sequence[LongHorizonV4Example],
        ks: Sequence[int] = W52_DEFAULT_LHR_V4_DROPOFF_PROBE_KS,
) -> tuple[tuple[int, float], ...]:
    out: list[tuple[int, float]] = []
    for k in ks:
        clipped_k = int(k)
        if clipped_k > head.max_k:
            clipped_k = head.max_k + 1
        mse_sum = 0.0
        n = 0
        for ex in examples:
            if int(ex.k) != int(k):
                continue
            pred, _, _, _ = head.forward_value(
                carrier=ex.carrier,
                k=int(clipped_k),
                branch=int(ex.branch_index),
                cycle=int(ex.cycle_index))
            mse_sum += _mse(pred, ex.target_features)
            n += 1
        mean = (
            float(mse_sum) / float(n)) if n > 0 else 0.0
        out.append((int(k), float(mean)))
    return tuple(out)


# =============================================================================
# Witness
# =============================================================================


@dataclasses.dataclass(frozen=True)
class LongHorizonReconstructionV4Witness:
    head_cid: str
    training_trace_cid: str
    max_k: int
    n_branches: int
    n_cycles: int
    out_dim: int
    mse_at_k1: float
    mse_at_k4: float
    mse_at_k8: float
    mse_at_k12: float
    degradation_curve: tuple[tuple[int, float], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "head_cid": str(self.head_cid),
            "training_trace_cid": str(self.training_trace_cid),
            "max_k": int(self.max_k),
            "n_branches": int(self.n_branches),
            "n_cycles": int(self.n_cycles),
            "out_dim": int(self.out_dim),
            "mse_at_k1": float(round(self.mse_at_k1, 12)),
            "mse_at_k4": float(round(self.mse_at_k4, 12)),
            "mse_at_k8": float(round(self.mse_at_k8, 12)),
            "mse_at_k12": float(round(self.mse_at_k12, 12)),
            "degradation_curve": [
                [int(k), float(round(v, 12))]
                for k, v in self.degradation_curve],
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w52_long_horizon_reconstruction_v4_witness",
            "witness": self.to_dict()})


def emit_long_horizon_v4_witness(
        *,
        head: LongHorizonReconstructionV4Head,
        training_trace: LongHorizonV4TrainingTrace,
        examples: Sequence[LongHorizonV4Example] = (),
        probe_ks: Sequence[int] = (1, 4, 8, 12),
) -> LongHorizonReconstructionV4Witness:
    mse_map: dict[int, float] = {}
    for k in probe_ks:
        if examples:
            mse_map[int(k)] = evaluate_long_horizon_v4_mse_at_k(
                head, examples, int(k))
        else:
            mse_map[int(k)] = 0.0
    curve = (
        evaluate_long_horizon_v4_mse_curve(head, examples)
        if examples else ())
    return LongHorizonReconstructionV4Witness(
        head_cid=str(head.cid()),
        training_trace_cid=str(training_trace.cid()),
        max_k=int(head.max_k),
        n_branches=int(head.n_branches),
        n_cycles=int(head.n_cycles),
        out_dim=int(head.out_dim),
        mse_at_k1=float(mse_map.get(1, 0.0)),
        mse_at_k4=float(mse_map.get(4, 0.0)),
        mse_at_k8=float(mse_map.get(8, 0.0)),
        mse_at_k12=float(mse_map.get(12, 0.0)),
        degradation_curve=curve,
    )


# =============================================================================
# Verifier
# =============================================================================


W52_LHR_V4_VERIFIER_FAILURE_MODES: tuple[str, ...] = (
    "w52_lhr_v4_schema_mismatch",
    "w52_lhr_v4_head_cid_mismatch",
    "w52_lhr_v4_max_k_mismatch",
    "w52_lhr_v4_n_branches_mismatch",
    "w52_lhr_v4_n_cycles_mismatch",
)


def verify_long_horizon_v4_witness(
        witness: LongHorizonReconstructionV4Witness,
        *,
        expected_head_cid: str | None = None,
        expected_max_k: int | None = None,
) -> dict[str, Any]:
    failures: list[str] = []
    if (expected_head_cid is not None
            and witness.head_cid != expected_head_cid):
        failures.append("w52_lhr_v4_head_cid_mismatch")
    if (expected_max_k is not None
            and witness.max_k != int(expected_max_k)):
        failures.append("w52_lhr_v4_max_k_mismatch")
    return {
        "ok": (len(failures) == 0),
        "failures": failures,
        "witness_cid": witness.cid(),
    }


__all__ = [
    "W52_LHR_V4_SCHEMA_VERSION",
    "W52_DEFAULT_LHR_V4_FLAT_FEATURE_DIM",
    "W52_DEFAULT_LHR_V4_HIDDEN_DIM",
    "W52_DEFAULT_LHR_V4_MAX_K",
    "W52_DEFAULT_LHR_V4_N_BRANCHES",
    "W52_DEFAULT_LHR_V4_N_CYCLES",
    "W52_DEFAULT_LHR_V4_DROPOFF_PROBE_KS",
    "W52_LHR_V4_VERIFIER_FAILURE_MODES",
    "LongHorizonReconstructionV4Head",
    "LongHorizonV4Example",
    "LongHorizonV4TrainingSet",
    "LongHorizonV4TrainingTrace",
    "LongHorizonReconstructionV4Witness",
    "synthesize_long_horizon_v4_training_set",
    "fit_long_horizon_v4",
    "evaluate_long_horizon_v4_mse_at_k",
    "evaluate_long_horizon_v4_mse_curve",
    "emit_long_horizon_v4_witness",
    "verify_long_horizon_v4_witness",
]
