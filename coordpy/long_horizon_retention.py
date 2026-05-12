"""W51 M5 — Long-Horizon Reconstruction V3.

A trained two-headed reconstruction head:

* **causal head** — recovers turn ``t-k`` features from
  current carrier; ``k ∈ {1..max_k}``, default ``max_k=8``.
* **branch head** — recovers turn ``t-k`` features
  conditioned on the branch path index.

Reconstruction V3 extends W50's V2 along two axes: (1) longer
horizon (``max_k=8`` vs W50's ``max_k=3``), and (2) explicit
branch conditioning (so reconstruction of a branch-specific
fact is biased towards the right history).

A **degradation curve** witness records MSE across ``k ∈
{1..16}`` so the falloff past the training horizon is
auditable.

Pure-Python only — reuses the W47 ``Variable`` +
``AdamOptimizer`` autograd engine, the W50
``SharedLatentCarrierV2`` chain abstraction.

Honest scope (do-not-overstate)
-------------------------------

This module does NOT alter transformer-internal hidden state,
KV cache bytes, attention weights, or embeddings.

The H13 / H14 bars are empirical under the W47 pure-Python
autograd training cost cap. The
``W51-L-LONG-HORIZON-DROPOFF-CAP`` falsifier reproduces
honestly past ``k=8``.
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
    vsum,
)
from .shared_latent_carrier import W50_DEFAULT_CARRIER_DIM


# =============================================================================
# Schema, defaults
# =============================================================================

W51_LONG_HORIZON_RETENTION_SCHEMA_VERSION: str = (
    "coordpy.long_horizon_retention.v1")

W51_DEFAULT_LHR_CARRIER_DIM: int = W50_DEFAULT_CARRIER_DIM
W51_DEFAULT_LHR_HIDDEN_DIM: int = 18
W51_DEFAULT_LHR_MAX_K: int = 8
W51_DEFAULT_LHR_FLAT_FEATURE_DIM: int = 6
W51_DEFAULT_LHR_N_BRANCHES: int = 4
W51_DEFAULT_LHR_DROPOFF_PROBE_KS: tuple[int, ...] = (
    1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16)


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


def _mse(a: Sequence[float], b: Sequence[float]) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    s = 0.0
    for i in range(n):
        diff = float(a[i]) - float(b[i])
        s += diff * diff
    return float(s) / float(max(1, n))


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
# Two-headed reconstruction V3 head
# =============================================================================

@dataclasses.dataclass
class LongHorizonReconstructionV3Head:
    """Trainable two-headed reconstruction head.

    Input = ``[carrier; k_one_hot; branch_one_hot]`` of
    dimension ``carrier_dim + max_k + n_branches``.

    Output = predicted flat-feature vector of dimension
    ``out_dim``.
    """

    in_dim: int
    hidden_dim: int
    out_dim: int
    max_k: int
    n_branches: int
    # Shared trunk W1 / b1, then split heads W2c / b2c (causal)
    # and W2b / b2b (branch) — branch head is a residual on top
    # of causal head.
    w1: ParamTensor
    b1: ParamTensor
    w2c: ParamTensor
    b2c: ParamTensor
    w2b: ParamTensor
    b2b: ParamTensor

    @classmethod
    def init(
            cls, *,
            carrier_dim: int = W51_DEFAULT_LHR_CARRIER_DIM,
            hidden_dim: int = W51_DEFAULT_LHR_HIDDEN_DIM,
            out_dim: int = W51_DEFAULT_LHR_FLAT_FEATURE_DIM,
            max_k: int = W51_DEFAULT_LHR_MAX_K,
            n_branches: int = W51_DEFAULT_LHR_N_BRANCHES,
            seed: int = W47_DEFAULT_TRAIN_SEED,
            init_scale: float = W47_DEFAULT_INIT_SCALE,
    ) -> "LongHorizonReconstructionV3Head":
        in_dim = (int(carrier_dim) + int(max_k)
                  + int(n_branches))
        rng = _DeterministicLCG(seed=int(seed))
        w1 = ParamTensor(
            shape=(int(hidden_dim), int(in_dim)),
            values=[])
        w1.init_seed(
            seed=int(rng.next_uniform() * (1 << 30)),
            scale=float(init_scale))
        b1 = ParamTensor(
            shape=(int(hidden_dim),),
            values=[0.0] * int(hidden_dim))
        w2c = ParamTensor(
            shape=(int(out_dim), int(hidden_dim)),
            values=[])
        w2c.init_seed(
            seed=int(rng.next_uniform() * (1 << 30)),
            scale=float(init_scale))
        b2c = ParamTensor(
            shape=(int(out_dim),),
            values=[0.0] * int(out_dim))
        w2b = ParamTensor(
            shape=(int(out_dim), int(hidden_dim)),
            values=[])
        w2b.init_seed(
            seed=int(rng.next_uniform() * (1 << 30)),
            scale=float(init_scale))
        b2b = ParamTensor(
            shape=(int(out_dim),),
            values=[0.0] * int(out_dim))
        return cls(
            in_dim=int(in_dim),
            hidden_dim=int(hidden_dim),
            out_dim=int(out_dim),
            max_k=int(max_k),
            n_branches=int(n_branches),
            w1=w1, b1=b1,
            w2c=w2c, b2c=b2c,
            w2b=w2b, b2b=b2b)

    def params(self) -> list[ParamTensor]:
        return [
            self.w1, self.b1,
            self.w2c, self.b2c,
            self.w2b, self.b2b]

    @property
    def carrier_dim(self) -> int:
        return (int(self.in_dim) - int(self.max_k)
                - int(self.n_branches))

    def _make_input(
            self, carrier: Sequence[float],
            k: int, branch: int,
    ) -> list[float]:
        cd = self.carrier_dim
        out = list(carrier)[:cd]
        while len(out) < cd:
            out.append(0.0)
        # k one-hot in [1, max_k]
        for kk in range(1, self.max_k + 1):
            out.append(1.0 if (int(k) == kk) else 0.0)
        # branch one-hot in [0, n_branches-1]
        for bb in range(self.n_branches):
            out.append(1.0 if (int(branch) == bb) else 0.0)
        return out

    def _make_input_vars(
            self, carrier: Sequence[Variable],
            k: int, branch: int,
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
        return out

    def forward_value(
            self, *,
            carrier: Sequence[float],
            k: int,
            branch: int = 0,
    ) -> tuple[list[float], list[float], list[float]]:
        """Returns (final_prediction, causal_head_out,
        branch_residual)."""
        x = self._make_input(carrier, int(k), int(branch))
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
        for r in range(self.out_dim):
            base = r * self.hidden_dim
            sc = 0.0
            sb = 0.0
            for j in range(self.hidden_dim):
                sc += float(self.w2c.values[base + j]) \
                    * float(hidden[j])
                sb += float(self.w2b.values[base + j]) \
                    * float(hidden[j])
            sc += float(self.b2c.values[r])
            sb += float(self.b2b.values[r])
            causal[r] = sc
            branch_res[r] = sb
        final = [
            causal[r] + branch_res[r]
            for r in range(self.out_dim)
        ]
        return final, causal, branch_res

    def forward_vars(
            self, *,
            carrier: Sequence[Variable],
            k: int,
            branch: int = 0,
    ) -> tuple[list[Variable], list[Variable]]:
        x_vars = self._make_input_vars(
            carrier, int(k), int(branch))
        w1_vars = self.w1.make_vars()
        b1_vars = self.b1.make_vars()
        w2c_vars = self.w2c.make_vars()
        b2c_vars = self.b2c.make_vars()
        w2b_vars = self.w2b.make_vars()
        b2b_vars = self.b2b.make_vars()
        rows_h: list[list[Variable]] = []
        for r in range(self.hidden_dim):
            base = r * self.in_dim
            rows_h.append(list(w1_vars[base:base + self.in_dim]))
        pre_h = vmatmul(rows_h, x_vars)
        hidden = [
            (pre_h[i] + b1_vars[i]).tanh()
            for i in range(self.hidden_dim)
        ]
        rows_c: list[list[Variable]] = []
        rows_b: list[list[Variable]] = []
        for r in range(self.out_dim):
            base = r * self.hidden_dim
            rows_c.append(list(
                w2c_vars[base:base + self.hidden_dim]))
            rows_b.append(list(
                w2b_vars[base:base + self.hidden_dim]))
        pre_c = vmatmul(rows_c, hidden)
        pre_b = vmatmul(rows_b, hidden)
        causal = [
            pre_c[i] + b2c_vars[i] for i in range(self.out_dim)
        ]
        branch_res = [
            pre_b[i] + b2b_vars[i] for i in range(self.out_dim)
        ]
        final = [
            causal[i] + branch_res[i]
            for i in range(self.out_dim)
        ]
        return final, causal

    def to_dict(self) -> dict[str, Any]:
        return {
            "in_dim": int(self.in_dim),
            "hidden_dim": int(self.hidden_dim),
            "out_dim": int(self.out_dim),
            "max_k": int(self.max_k),
            "n_branches": int(self.n_branches),
            "w1": self.w1.to_dict(),
            "b1": self.b1.to_dict(),
            "w2c": self.w2c.to_dict(),
            "b2c": self.b2c.to_dict(),
            "w2b": self.w2b.to_dict(),
            "b2b": self.b2b.to_dict(),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w51_long_horizon_reconstruction_v3_head",
            "head": self.to_dict()})


# =============================================================================
# Training set + fit
# =============================================================================

@dataclasses.dataclass(frozen=True)
class LongHorizonReconstructionExample:
    """One training example.

    ``carrier`` is the turn-``t`` carrier; ``target_features``
    is the turn-``t-k`` flat features we want to recover;
    ``branch_index`` selects the branch path.
    """

    carrier: tuple[float, ...]
    k: int
    branch_index: int
    target_features: tuple[float, ...]


@dataclasses.dataclass(frozen=True)
class LongHorizonReconstructionTrainingSet:
    examples: tuple[LongHorizonReconstructionExample, ...]
    carrier_dim: int
    out_dim: int
    max_k: int
    n_branches: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "carrier_dim": int(self.carrier_dim),
            "out_dim": int(self.out_dim),
            "max_k": int(self.max_k),
            "n_branches": int(self.n_branches),
            "examples": [
                {"carrier": list(e.carrier),
                 "k": int(e.k),
                 "branch_index": int(e.branch_index),
                 "target_features": list(e.target_features)}
                for e in self.examples],
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": (
                "w51_long_horizon_reconstruction_training_set"),
            "set": self.to_dict()})


def synthesize_long_horizon_reconstruction_training_set(
        *,
        n_sequences: int = 6,
        sequence_length: int = 12,
        carrier_dim: int | None = None,
        out_dim: int = W51_DEFAULT_LHR_FLAT_FEATURE_DIM,
        max_k: int = W51_DEFAULT_LHR_MAX_K,
        n_branches: int = W51_DEFAULT_LHR_N_BRANCHES,
        seed: int = W47_DEFAULT_TRAIN_SEED,
        carrier_decay: float = 0.0,
) -> LongHorizonReconstructionTrainingSet:
    """Synthesise a deterministic dataset of multi-turn
    sequences where the carrier at turn ``t`` *explicitly
    stores* the flat features from turns
    ``t-1, t-2, ..., t-max_k``, with an optional per-slot
    exponential decay.

    Multi-branch: each sequence runs under a fixed
    ``branch_index`` in ``[0, n_branches)``. The branch
    influences a tiny additive shift on the stored features so
    the branch head can learn to discriminate.
    """
    actual_carrier_dim = int(
        carrier_dim
        if carrier_dim is not None
        else int(max_k) * int(out_dim))
    if actual_carrier_dim < int(max_k) * int(out_dim):
        raise ValueError(
            "carrier_dim must be >= max_k * out_dim to store "
            "the prior-turn lookback")
    rng = _DeterministicLCG(seed=int(seed))
    examples: list[LongHorizonReconstructionExample] = []
    for s_idx in range(int(n_sequences)):
        branch = int(s_idx % int(n_branches))
        flats: list[list[float]] = []
        for _ in range(int(sequence_length)):
            v = [
                float(rng.next_uniform() * 2.0 - 1.0)
                for _ in range(int(out_dim))
            ]
            # Branch-specific small additive shift
            for j in range(int(out_dim)):
                v[j] += float(branch) * 0.1
            flats.append(v)
        # Build carriers explicitly from past flats.
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
                examples.append(
                    LongHorizonReconstructionExample(
                        carrier=tuple(carriers[t]),
                        k=int(k),
                        branch_index=int(branch),
                        target_features=tuple(flats[t - k])))
    return LongHorizonReconstructionTrainingSet(
        examples=tuple(examples),
        carrier_dim=int(actual_carrier_dim),
        out_dim=int(out_dim),
        max_k=int(max_k),
        n_branches=int(n_branches))


@dataclasses.dataclass(frozen=True)
class LongHorizonReconstructionTrainingTrace:
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
            "final_grad_norm": float(
                round(self.final_grad_norm, 12)),
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
            "kind": (
                "w51_long_horizon_reconstruction_training_trace"),
            "trace": self.to_dict()})


def fit_long_horizon_reconstruction_v3(
        training_set: LongHorizonReconstructionTrainingSet,
        *,
        hidden_dim: int = W51_DEFAULT_LHR_HIDDEN_DIM,
        n_steps: int = 192,
        learning_rate: float = 0.03,
        seed: int = W47_DEFAULT_TRAIN_SEED,
        init_scale: float = W47_DEFAULT_INIT_SCALE,
        beta1: float = W47_DEFAULT_BETA1,
        beta2: float = W47_DEFAULT_BETA2,
        eps: float = W47_DEFAULT_EPS,
        grad_clip: float = W47_DEFAULT_GRAD_CLIP,
        history_head: int = 6,
        history_tail: int = 6,
) -> tuple[LongHorizonReconstructionV3Head,
           LongHorizonReconstructionTrainingTrace]:
    """Fit the reconstruction V3 head via Adam SGD on MSE loss."""
    head = LongHorizonReconstructionV3Head.init(
        carrier_dim=int(training_set.carrier_dim),
        hidden_dim=int(hidden_dim),
        out_dim=int(training_set.out_dim),
        max_k=int(training_set.max_k),
        n_branches=int(training_set.n_branches),
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
        out_vars, _ = head.forward_vars(
            carrier=c_vars,
            k=int(ex.k),
            branch=int(ex.branch_index))
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
    trace = LongHorizonReconstructionTrainingTrace(
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


def evaluate_long_horizon_mse_at_k(
        head: LongHorizonReconstructionV3Head,
        examples: Sequence[LongHorizonReconstructionExample],
        k: int,
) -> float:
    """Mean MSE at lookback ``k`` across the example set."""
    if not examples:
        return 0.0
    mse_sum = 0.0
    n = 0
    for ex in examples:
        if int(ex.k) != int(k):
            continue
        pred, _, _ = head.forward_value(
            carrier=ex.carrier,
            k=int(ex.k),
            branch=int(ex.branch_index))
        mse_sum += _mse(pred, ex.target_features)
        n += 1
    return float(mse_sum) / float(max(1, n))


def evaluate_long_horizon_cosine_at_k(
        head: LongHorizonReconstructionV3Head,
        examples: Sequence[LongHorizonReconstructionExample],
        k: int,
) -> float:
    """Mean cosine at lookback ``k`` across the example set."""
    if not examples:
        return 0.0
    cos_sum = 0.0
    n = 0
    for ex in examples:
        if int(ex.k) != int(k):
            continue
        pred, _, _ = head.forward_value(
            carrier=ex.carrier,
            k=int(ex.k),
            branch=int(ex.branch_index))
        cos_sum += _cosine(pred, ex.target_features)
        n += 1
    return float(cos_sum) / float(max(1, n))


def evaluate_long_horizon_mse_curve(
        head: LongHorizonReconstructionV3Head,
        examples: Sequence[LongHorizonReconstructionExample],
        ks: Sequence[int] = (
            W51_DEFAULT_LHR_DROPOFF_PROBE_KS),
) -> tuple[tuple[int, float], ...]:
    """Probe MSE curve across the requested k values.

    For k > head.max_k, the one-hot in the input is all-zero
    (out of the trained range); we expect MSE to rise as a
    natural drop-off — honest about the
    W51-L-LONG-HORIZON-DROPOFF-CAP falsifier.
    """
    out: list[tuple[int, float]] = []
    for k in ks:
        # Bound k for the one-hot; if k > max_k, the head sees
        # all-zero k input → degraded prediction.
        clipped_k = int(k)
        if clipped_k > head.max_k:
            clipped_k = head.max_k + 1  # out-of-range sentinel
        mse_sum = 0.0
        n = 0
        for ex in examples:
            if int(ex.k) != int(k):
                continue
            pred, _, _ = head.forward_value(
                carrier=ex.carrier,
                k=int(clipped_k),
                branch=int(ex.branch_index))
            mse_sum += _mse(pred, ex.target_features)
            n += 1
        mean = (float(mse_sum) / float(n)) if n > 0 else 0.0
        out.append((int(k), float(mean)))
    return tuple(out)


# =============================================================================
# Witness
# =============================================================================

@dataclasses.dataclass(frozen=True)
class LongHorizonReconstructionV3Witness:
    """Sealed per-turn long-horizon reconstruction V3 witness."""

    head_cid: str
    training_trace_cid: str
    carrier_dim: int
    out_dim: int
    max_k: int
    n_branches: int
    mse_per_k: tuple[float, ...]
    cosine_per_k: tuple[float, ...]
    mse_curve_extended: tuple[tuple[int, float], ...]
    recovered_target_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "head_cid": str(self.head_cid),
            "training_trace_cid": str(self.training_trace_cid),
            "carrier_dim": int(self.carrier_dim),
            "out_dim": int(self.out_dim),
            "max_k": int(self.max_k),
            "n_branches": int(self.n_branches),
            "mse_per_k": [
                float(round(v, 12)) for v in self.mse_per_k],
            "cosine_per_k": [
                float(round(v, 12))
                for v in self.cosine_per_k],
            "mse_curve_extended": [
                [int(k), float(round(m, 12))]
                for (k, m) in self.mse_curve_extended],
            "recovered_target_cid": str(
                self.recovered_target_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": (
                "w51_long_horizon_reconstruction_v3_witness"),
            "witness": self.to_dict()})


def emit_long_horizon_reconstruction_v3_witness(
        *,
        head: LongHorizonReconstructionV3Head,
        training_trace: (
            LongHorizonReconstructionTrainingTrace),
        examples: Sequence[LongHorizonReconstructionExample],
        extended_ks: Sequence[int] = (
            W51_DEFAULT_LHR_DROPOFF_PROBE_KS),
) -> LongHorizonReconstructionV3Witness:
    mse_per_k: list[float] = []
    cos_per_k: list[float] = []
    for k in range(1, head.max_k + 1):
        mse_per_k.append(
            evaluate_long_horizon_mse_at_k(
                head, examples, k=int(k)))
        cos_per_k.append(
            evaluate_long_horizon_cosine_at_k(
                head, examples, k=int(k)))
    curve = evaluate_long_horizon_mse_curve(
        head, examples, ks=extended_ks)
    target_payload = {
        "kind": "w51_long_horizon_recovered_target_features",
        "n_examples": int(len(examples)),
        "training_set_cid": str(
            training_trace.training_set_cid),
    }
    return LongHorizonReconstructionV3Witness(
        head_cid=str(head.cid()),
        training_trace_cid=str(training_trace.cid()),
        carrier_dim=int(head.carrier_dim),
        out_dim=int(head.out_dim),
        max_k=int(head.max_k),
        n_branches=int(head.n_branches),
        mse_per_k=tuple(mse_per_k),
        cosine_per_k=tuple(cos_per_k),
        mse_curve_extended=tuple(curve),
        recovered_target_cid=str(_sha256_hex(target_payload)),
    )


# =============================================================================
# Verifier
# =============================================================================

W51_LONG_HORIZON_RETENTION_VERIFIER_FAILURE_MODES: tuple[
        str, ...] = (
    "w51_lhr_schema_mismatch",
    "w51_lhr_head_cid_mismatch",
    "w51_lhr_training_trace_cid_mismatch",
    "w51_lhr_witness_cid_mismatch",
    "w51_lhr_mse_at_k5_above_floor",
    "w51_lhr_mse_at_k8_above_stretch_floor",
    "w51_lhr_dropoff_curve_invalid",
)


def verify_long_horizon_reconstruction_v3_witness(
        witness: LongHorizonReconstructionV3Witness,
        *,
        expected_head_cid: str | None = None,
        expected_trace_cid: str | None = None,
        mse_floor_at_k5: float | None = None,
        mse_floor_at_k8: float | None = None,
) -> dict[str, Any]:
    failures: list[str] = []
    if (expected_head_cid is not None
            and witness.head_cid != expected_head_cid):
        failures.append("w51_lhr_head_cid_mismatch")
    if (expected_trace_cid is not None
            and witness.training_trace_cid != expected_trace_cid):
        failures.append("w51_lhr_training_trace_cid_mismatch")
    if (mse_floor_at_k5 is not None
            and len(witness.mse_per_k) >= 5
            and witness.mse_per_k[4] > float(mse_floor_at_k5)):
        failures.append("w51_lhr_mse_at_k5_above_floor")
    if (mse_floor_at_k8 is not None
            and len(witness.mse_per_k) >= 8
            and witness.mse_per_k[7] > float(mse_floor_at_k8)):
        failures.append("w51_lhr_mse_at_k8_above_stretch_floor")
    return {
        "ok": (len(failures) == 0),
        "failures": failures,
        "witness_cid": witness.cid(),
    }


__all__ = [
    "W51_LONG_HORIZON_RETENTION_SCHEMA_VERSION",
    "W51_DEFAULT_LHR_CARRIER_DIM",
    "W51_DEFAULT_LHR_HIDDEN_DIM",
    "W51_DEFAULT_LHR_MAX_K",
    "W51_DEFAULT_LHR_FLAT_FEATURE_DIM",
    "W51_DEFAULT_LHR_N_BRANCHES",
    "W51_DEFAULT_LHR_DROPOFF_PROBE_KS",
    "W51_LONG_HORIZON_RETENTION_VERIFIER_FAILURE_MODES",
    "LongHorizonReconstructionV3Head",
    "LongHorizonReconstructionExample",
    "LongHorizonReconstructionTrainingSet",
    "LongHorizonReconstructionTrainingTrace",
    "LongHorizonReconstructionV3Witness",
    "synthesize_long_horizon_reconstruction_training_set",
    "fit_long_horizon_reconstruction_v3",
    "evaluate_long_horizon_mse_at_k",
    "evaluate_long_horizon_cosine_at_k",
    "evaluate_long_horizon_mse_curve",
    "emit_long_horizon_reconstruction_v3_witness",
    "verify_long_horizon_reconstruction_v3_witness",
]
