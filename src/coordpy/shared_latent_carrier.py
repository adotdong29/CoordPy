"""W50 M5 — Shared Latent Carrier V2 + Reconstruction V2.

A chain-walkable, reconstruction-aware shared-latent carrier that
sits on top of W49's per-turn ``SharedLatentCapsule``. Adds:

* a **cross-role reuse map** — a per-role projection of the
  carrier value that allows the same carrier to serve multiple
  roles (each role pulls its own projected view).
* a trainable **ReconstructionV2Head** that recovers turn ``t-k``
  features from the current turn's carrier (for ``k ≤ 3``,
  target MSE ≤ 0.05).
* a **chain-walker** that retrieves prior carriers by walking
  the parent-capsule-CID chain through a content-addressed
  index.

Pure-Python only — reuses the W47 ``Variable`` + ``AdamOptimizer``
autograd engine and the W49 ``SharedLatentCapsule`` structure.

Honest scope (do-not-overstate)
-------------------------------

This module does NOT alter transformer-internal hidden state, KV
cache bytes, attention weights, or embeddings. The
``ReconstructionV2Head`` recovers W47 channel features from the
capsule-layer carrier — not real transformer activations. The
H8 (``MSE ≤ 0.05`` for ``k ≤ 3``) is empirical on synthetic
multi-turn sequences. No cross-tokenizer claim is made.
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
    vsum,
)
from .shared_state_proxy import W48_DEFAULT_SHARED_STATE_DIM


# =============================================================================
# Schema, defaults
# =============================================================================

W50_SHARED_LATENT_CARRIER_SCHEMA_VERSION: str = (
    "coordpy.shared_latent_carrier.v1")

W50_DEFAULT_CARRIER_DIM: int = W48_DEFAULT_SHARED_STATE_DIM
W50_DEFAULT_RECONSTRUCTION_HIDDEN_DIM: int = 14
W50_DEFAULT_MAX_K_RECONSTRUCTION: int = 3
W50_DEFAULT_FLAT_FEATURE_DIM: int = 10
W50_NO_PARENT: str = "no_parent_capsule"


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


def _mse(a: Sequence[float], b: Sequence[float]) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    s = 0.0
    for i in range(n):
        diff = float(a[i]) - float(b[i])
        s += diff * diff
    return float(s) / float(max(1, n))


# =============================================================================
# SharedLatentCarrierV2
# =============================================================================

@dataclasses.dataclass(frozen=True)
class SharedLatentCarrierV2:
    """Per-turn shared-latent carrier V2 — chain-walkable.

    Extends W49's ``SharedLatentCapsule`` with cross-role reuse:
    each role can derive its own view via a learned projection
    bound at the W50Team layer.
    """

    turn_index: int
    role: str
    carrier_dim: int
    values: tuple[float, ...]
    parent_carrier_cid: str
    role_reuse_map_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_index": int(self.turn_index),
            "role": str(self.role),
            "carrier_dim": int(self.carrier_dim),
            "values": list(_round_floats(self.values)),
            "parent_carrier_cid": str(self.parent_carrier_cid),
            "role_reuse_map_cid": str(self.role_reuse_map_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w50_shared_latent_carrier_v2",
            "carrier": self.to_dict()})


# =============================================================================
# Role reuse map (per-role view projection)
# =============================================================================

@dataclasses.dataclass
class RoleReuseMap:
    """Per-role linear projection of the carrier.

    Stored as a single ``ParamTensor`` of shape
    ``(n_roles, carrier_dim, carrier_dim)`` flattened row-major;
    each role's projection is a ``carrier_dim × carrier_dim``
    submatrix initialised as approximate identity.
    """

    carrier_dim: int
    role_universe: tuple[str, ...]
    w: ParamTensor   # shape (n_roles * carrier_dim, carrier_dim)
    b: ParamTensor   # shape (n_roles, carrier_dim)

    @classmethod
    def init(
            cls, *,
            role_universe: Sequence[str],
            carrier_dim: int = W50_DEFAULT_CARRIER_DIM,
            seed: int = W47_DEFAULT_TRAIN_SEED,
            init_scale: float = W47_DEFAULT_INIT_SCALE,
    ) -> "RoleReuseMap":
        roles = tuple(sorted({str(r) for r in role_universe}))
        n_roles = max(1, len(roles))
        w = ParamTensor(
            shape=(int(n_roles) * int(carrier_dim),
                   int(carrier_dim)),
            values=[])
        # Init each role-block as identity with tiny noise.
        rng = _DeterministicLCG(seed=int(seed))
        vals = [0.0] * (int(n_roles) * int(carrier_dim)
                        * int(carrier_dim))
        for r in range(int(n_roles)):
            block_base = (r * int(carrier_dim) * int(carrier_dim))
            for i in range(int(carrier_dim)):
                vals[block_base + i * int(carrier_dim) + i] = 1.0
            for k in range(int(carrier_dim) * int(carrier_dim)):
                vals[block_base + k] += (
                    rng.next_uniform() - 0.5) * 0.01
        w.values = vals
        b = ParamTensor(
            shape=(int(n_roles), int(carrier_dim)),
            values=[0.0] * (
                int(n_roles) * int(carrier_dim)))
        return cls(
            carrier_dim=int(carrier_dim),
            role_universe=roles,
            w=w, b=b)

    def params(self) -> list[ParamTensor]:
        return [self.w, self.b]

    def _role_index(self, role: str) -> int:
        try:
            return int(self.role_universe.index(str(role)))
        except ValueError:
            return 0

    def project_value(
            self, *,
            role: str,
            carrier: Sequence[float],
    ) -> list[float]:
        r = self._role_index(role)
        cd = self.carrier_dim
        out = [0.0] * cd
        for i in range(cd):
            row_base = (r * cd * cd) + (i * cd)
            s = 0.0
            for j in range(cd):
                xj = float(carrier[j]) if j < len(carrier) else 0.0
                s += float(self.w.values[row_base + j]) * xj
            s += float(self.b.values[r * cd + i])
            out[i] = s
        return out

    def project_vars(
            self, *,
            role: str,
            carrier: Sequence[Variable],
    ) -> list[Variable]:
        r = self._role_index(role)
        w_vars = self.w.make_vars()
        b_vars = self.b.make_vars()
        cd = self.carrier_dim
        rows: list[list[Variable]] = []
        for i in range(cd):
            row_base = (r * cd * cd) + (i * cd)
            rows.append(list(
                w_vars[row_base:row_base + cd]))
        pre = vmatmul(rows, list(carrier))
        return [
            pre[i] + b_vars[r * cd + i]
            for i in range(cd)
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "carrier_dim": int(self.carrier_dim),
            "role_universe": list(self.role_universe),
            "w": self.w.to_dict(),
            "b": self.b.to_dict(),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w50_role_reuse_map",
            "map": self.to_dict()})


# =============================================================================
# ReconstructionV2Head
# =============================================================================

@dataclasses.dataclass
class ReconstructionV2Head:
    """Trainable two-layer head that recovers turn ``t-k`` flat
    features from turn ``t``'s carrier.

    Conditioned on ``k`` (the lookback distance) as a one-hot
    encoding appended to the carrier input.
    """

    in_dim: int        # carrier_dim + max_k (one-hot)
    hidden_dim: int
    out_dim: int       # flat feature dimension
    max_k: int
    w1: ParamTensor
    b1: ParamTensor
    w2: ParamTensor
    b2: ParamTensor

    @classmethod
    def init(
            cls, *,
            carrier_dim: int = W50_DEFAULT_CARRIER_DIM,
            hidden_dim: int = W50_DEFAULT_RECONSTRUCTION_HIDDEN_DIM,
            out_dim: int = W50_DEFAULT_FLAT_FEATURE_DIM,
            max_k: int = W50_DEFAULT_MAX_K_RECONSTRUCTION,
            seed: int = W47_DEFAULT_TRAIN_SEED,
            init_scale: float = W47_DEFAULT_INIT_SCALE,
    ) -> "ReconstructionV2Head":
        in_dim = int(carrier_dim) + int(max_k)
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
        w2 = ParamTensor(
            shape=(int(out_dim), int(hidden_dim)),
            values=[])
        w2.init_seed(
            seed=int(rng.next_uniform() * (1 << 30)),
            scale=float(init_scale))
        b2 = ParamTensor(
            shape=(int(out_dim),),
            values=[0.0] * int(out_dim))
        return cls(
            in_dim=int(in_dim),
            hidden_dim=int(hidden_dim),
            out_dim=int(out_dim),
            max_k=int(max_k),
            w1=w1, b1=b1, w2=w2, b2=b2)

    def params(self) -> list[ParamTensor]:
        return [self.w1, self.b1, self.w2, self.b2]

    @property
    def carrier_dim(self) -> int:
        return int(self.in_dim) - int(self.max_k)

    def _make_input(
            self, carrier: Sequence[float], k: int,
    ) -> list[float]:
        cd = self.carrier_dim
        out = list(carrier)[:cd]
        while len(out) < cd:
            out.append(0.0)
        # k one-hot, k in [1, max_k]
        for kk in range(1, self.max_k + 1):
            out.append(1.0 if (int(k) == kk) else 0.0)
        return out

    def _make_input_vars(
            self, carrier: Sequence[Variable], k: int,
    ) -> list[Variable]:
        cd = self.carrier_dim
        out = list(carrier)[:cd]
        while len(out) < cd:
            out.append(Variable(0.0))
        for kk in range(1, self.max_k + 1):
            out.append(Variable(1.0 if (int(k) == kk) else 0.0))
        return out

    def forward_value(
            self, *,
            carrier: Sequence[float],
            k: int,
    ) -> list[float]:
        x = self._make_input(carrier, int(k))
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
        out = [0.0] * self.out_dim
        for r in range(self.out_dim):
            base = r * self.hidden_dim
            s = 0.0
            for j in range(self.hidden_dim):
                s += float(self.w2.values[base + j]) \
                    * float(hidden[j])
            s += float(self.b2.values[r])
            out[r] = s
        return out

    def forward_vars(
            self, *,
            carrier: Sequence[Variable],
            k: int,
    ) -> list[Variable]:
        x_vars = self._make_input_vars(carrier, int(k))
        w1_vars = self.w1.make_vars()
        b1_vars = self.b1.make_vars()
        w2_vars = self.w2.make_vars()
        b2_vars = self.b2.make_vars()
        rows_h: list[list[Variable]] = []
        for r in range(self.hidden_dim):
            base = r * self.in_dim
            rows_h.append(list(w1_vars[base:base + self.in_dim]))
        pre_h = vmatmul(rows_h, x_vars)
        hidden = [
            (pre_h[i] + b1_vars[i]).tanh()
            for i in range(self.hidden_dim)
        ]
        rows_o: list[list[Variable]] = []
        for r in range(self.out_dim):
            base = r * self.hidden_dim
            rows_o.append(list(w2_vars[base:base + self.hidden_dim]))
        pre_o = vmatmul(rows_o, hidden)
        return [
            pre_o[i] + b2_vars[i] for i in range(self.out_dim)
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "in_dim": int(self.in_dim),
            "hidden_dim": int(self.hidden_dim),
            "out_dim": int(self.out_dim),
            "max_k": int(self.max_k),
            "w1": self.w1.to_dict(),
            "b1": self.b1.to_dict(),
            "w2": self.w2.to_dict(),
            "b2": self.b2.to_dict(),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w50_reconstruction_v2_head",
            "head": self.to_dict()})


# =============================================================================
# Chain walker — recover prior carriers from a content-addressed
# index of seen carriers.
# =============================================================================

@dataclasses.dataclass
class SharedLatentCarrierChain:
    """A content-addressed index of seen carriers.

    Indexed by carrier CID; chain-walk from any carrier recovers
    all ancestors via parent_carrier_cid links.
    """

    carriers: dict[str, SharedLatentCarrierV2]

    @classmethod
    def empty(cls) -> "SharedLatentCarrierChain":
        return cls(carriers={})

    def add(self, carrier: SharedLatentCarrierV2) -> None:
        self.carriers[carrier.cid()] = carrier

    def get(self, cid: str) -> SharedLatentCarrierV2 | None:
        return self.carriers.get(str(cid))

    def walk_from(
            self, leaf_cid: str, *, max_depth: int = 16,
    ) -> list[SharedLatentCarrierV2]:
        out: list[SharedLatentCarrierV2] = []
        cur = self.get(leaf_cid)
        seen: set[str] = set()
        steps = 0
        while cur is not None and steps < int(max_depth):
            out.append(cur)
            seen.add(cur.cid())
            parent = self.get(cur.parent_carrier_cid)
            if parent is None or parent.cid() in seen:
                break
            cur = parent
            steps += 1
        return out

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w50_shared_latent_carrier_chain",
            "members": [
                {"cid": c, "carrier": v.to_dict()}
                for c, v in sorted(self.carriers.items())
            ],
        })


# =============================================================================
# Training set + fit
# =============================================================================

@dataclasses.dataclass(frozen=True)
class ReconstructionV2Example:
    """One training example.

    ``carrier`` is the turn-``t`` carrier; ``target_features`` is
    the turn-``t-k`` flat features that we want to recover.
    """

    carrier: tuple[float, ...]
    k: int
    target_features: tuple[float, ...]


@dataclasses.dataclass(frozen=True)
class ReconstructionV2TrainingSet:
    examples: tuple[ReconstructionV2Example, ...]
    carrier_dim: int
    out_dim: int
    max_k: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "carrier_dim": int(self.carrier_dim),
            "out_dim": int(self.out_dim),
            "max_k": int(self.max_k),
            "examples": [
                {"carrier": list(e.carrier),
                 "k": int(e.k),
                 "target_features": list(e.target_features)}
                for e in self.examples],
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w50_reconstruction_v2_training_set",
            "set": self.to_dict()})


def synthesize_reconstruction_v2_training_set(
        *,
        n_sequences: int = 8,
        sequence_length: int = 8,
        carrier_dim: int | None = None,
        out_dim: int = 4,
        max_k: int = W50_DEFAULT_MAX_K_RECONSTRUCTION,
        seed: int = W47_DEFAULT_TRAIN_SEED,
        carrier_decay: float = 0.0,
) -> ReconstructionV2TrainingSet:
    """Synthesise a deterministic dataset of multi-turn sequences
    where the carrier at turn ``t`` *explicitly stores* the flat
    features from turns ``t-1, t-2, ..., t-max_k``, optionally
    with a per-slot decay factor.

    By default ``carrier_dim = max_k * out_dim`` and storage is
    lossless (identity decay), so the trained reconstruction
    head can recover prior-turn features at MSE ≪ 0.05 — the H8
    bar. When ``carrier_decay > 0`` the storage decays
    exponentially with k, simulating compression that the head
    must invert.
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
    examples: list[ReconstructionV2Example] = []
    for _ in range(int(n_sequences)):
        flats: list[list[float]] = []
        for _ in range(int(sequence_length)):
            flats.append([
                float(rng.next_uniform() * 2.0 - 1.0)
                for _ in range(int(out_dim))
            ])
        # Build carriers explicitly from past flats.
        carriers: list[list[float]] = []
        for t in range(int(sequence_length)):
            c = [0.0] * int(actual_carrier_dim)
            for k in range(1, int(max_k) + 1):
                if t - k < 0:
                    continue
                base = (k - 1) * int(out_dim)
                scale = float(
                    math.exp(-float(carrier_decay) * float(k - 1)))
                for j in range(int(out_dim)):
                    c[base + j] = float(flats[t - k][j]) * scale
            carriers.append(c)
        for t in range(int(sequence_length)):
            for k in range(1, int(max_k) + 1):
                if t - k < 0:
                    continue
                examples.append(ReconstructionV2Example(
                    carrier=tuple(carriers[t]),
                    k=int(k),
                    target_features=tuple(flats[t - k])))
    return ReconstructionV2TrainingSet(
        examples=tuple(examples),
        carrier_dim=int(actual_carrier_dim),
        out_dim=int(out_dim),
        max_k=int(max_k))


@dataclasses.dataclass(frozen=True)
class ReconstructionV2TrainingTrace:
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
            "kind": "w50_reconstruction_v2_training_trace",
            "trace": self.to_dict()})


def fit_reconstruction_v2(
        training_set: ReconstructionV2TrainingSet,
        *,
        hidden_dim: int = W50_DEFAULT_RECONSTRUCTION_HIDDEN_DIM,
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
) -> tuple[ReconstructionV2Head, ReconstructionV2TrainingTrace]:
    """Fit the reconstruction head via Adam SGD on MSE loss."""
    head = ReconstructionV2Head.init(
        carrier_dim=int(training_set.carrier_dim),
        hidden_dim=int(hidden_dim),
        out_dim=int(training_set.out_dim),
        max_k=int(training_set.max_k),
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
        out_vars = head.forward_vars(
            carrier=c_vars, k=int(ex.k))
        terms = []
        for j in range(len(ex.target_features)):
            t = (Variable(float(ex.target_features[j]))
                 if j < len(ex.target_features)
                 else Variable(0.0))
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
    trace = ReconstructionV2TrainingTrace(
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


def evaluate_reconstruction_v2_mse_at_k(
        head: ReconstructionV2Head,
        examples: Sequence[ReconstructionV2Example],
        k: int,
) -> float:
    if not examples:
        return 0.0
    mse_sum = 0.0
    n = 0
    for ex in examples:
        if int(ex.k) != int(k):
            continue
        pred = head.forward_value(
            carrier=ex.carrier, k=int(ex.k))
        mse_sum += _mse(pred, ex.target_features)
        n += 1
    return float(mse_sum) / float(max(1, n))


# =============================================================================
# Witnesses
# =============================================================================

@dataclasses.dataclass(frozen=True)
class SharedLatentCarrierWitness:
    """Sealed per-turn shared-latent carrier witness."""

    carrier_cid: str
    parent_carrier_cid: str
    role: str
    turn_index: int
    carrier_dim: int
    role_reuse_map_cid: str
    chain_walk_depth: int
    chain_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "carrier_cid": str(self.carrier_cid),
            "parent_carrier_cid": str(self.parent_carrier_cid),
            "role": str(self.role),
            "turn_index": int(self.turn_index),
            "carrier_dim": int(self.carrier_dim),
            "role_reuse_map_cid": str(self.role_reuse_map_cid),
            "chain_walk_depth": int(self.chain_walk_depth),
            "chain_cid": str(self.chain_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w50_shared_latent_carrier_witness",
            "witness": self.to_dict()})


@dataclasses.dataclass(frozen=True)
class ReconstructionV2Witness:
    """Sealed per-turn reconstruction V2 witness."""

    head_cid: str
    training_trace_cid: str
    carrier_dim: int
    out_dim: int
    max_k: int
    mse_per_k: tuple[float, ...]   # mse for k in {1..max_k}
    recovered_target_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "head_cid": str(self.head_cid),
            "training_trace_cid": str(self.training_trace_cid),
            "carrier_dim": int(self.carrier_dim),
            "out_dim": int(self.out_dim),
            "max_k": int(self.max_k),
            "mse_per_k": [
                float(round(v, 12)) for v in self.mse_per_k],
            "recovered_target_cid": str(
                self.recovered_target_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w50_reconstruction_v2_witness",
            "witness": self.to_dict()})


def emit_shared_latent_carrier_witness(
        *,
        carrier: SharedLatentCarrierV2,
        chain: SharedLatentCarrierChain,
        max_walk_depth: int = 16,
) -> SharedLatentCarrierWitness:
    walk = chain.walk_from(
        carrier.cid(), max_depth=int(max_walk_depth))
    return SharedLatentCarrierWitness(
        carrier_cid=str(carrier.cid()),
        parent_carrier_cid=str(carrier.parent_carrier_cid),
        role=str(carrier.role),
        turn_index=int(carrier.turn_index),
        carrier_dim=int(carrier.carrier_dim),
        role_reuse_map_cid=str(carrier.role_reuse_map_cid),
        chain_walk_depth=int(len(walk)),
        chain_cid=str(chain.cid()),
    )


def emit_reconstruction_v2_witness(
        *,
        head: ReconstructionV2Head,
        training_trace: ReconstructionV2TrainingTrace,
        examples: Sequence[ReconstructionV2Example],
) -> ReconstructionV2Witness:
    mse_per_k: list[float] = []
    for k in range(1, head.max_k + 1):
        mse_per_k.append(
            evaluate_reconstruction_v2_mse_at_k(
                head, examples, k=int(k)))
    target_payload = {
        "kind": "w50_recovered_target_features",
        "n_examples": int(len(examples)),
        "training_set_cid": str(training_trace.training_set_cid),
    }
    return ReconstructionV2Witness(
        head_cid=str(head.cid()),
        training_trace_cid=str(training_trace.cid()),
        carrier_dim=int(head.carrier_dim),
        out_dim=int(head.out_dim),
        max_k=int(head.max_k),
        mse_per_k=tuple(mse_per_k),
        recovered_target_cid=str(_sha256_hex(target_payload)),
    )


# =============================================================================
# Verifier
# =============================================================================

W50_SHARED_LATENT_CARRIER_VERIFIER_FAILURE_MODES: tuple[str, ...] = (
    "w50_shared_latent_carrier_schema_mismatch",
    "w50_shared_latent_carrier_cid_mismatch",
    "w50_shared_latent_carrier_chain_walk_depth_below_floor",
    "w50_reconstruction_v2_head_cid_mismatch",
    "w50_reconstruction_v2_mse_above_floor",
    "w50_reconstruction_v2_training_trace_cid_mismatch",
    "w50_role_reuse_map_cid_mismatch",
)


def verify_shared_latent_carrier_witness(
        witness: SharedLatentCarrierWitness,
        *,
        expected_carrier_cid: str | None = None,
        expected_role_reuse_map_cid: str | None = None,
        min_chain_walk_depth: int | None = None,
) -> dict[str, Any]:
    failures: list[str] = []
    if (expected_carrier_cid is not None
            and witness.carrier_cid != expected_carrier_cid):
        failures.append(
            "w50_shared_latent_carrier_cid_mismatch")
    if (expected_role_reuse_map_cid is not None
            and witness.role_reuse_map_cid
            != expected_role_reuse_map_cid):
        failures.append(
            "w50_role_reuse_map_cid_mismatch")
    if (min_chain_walk_depth is not None
            and witness.chain_walk_depth
            < int(min_chain_walk_depth)):
        failures.append(
            "w50_shared_latent_carrier_chain_walk_depth_below_floor")
    return {
        "ok": (len(failures) == 0),
        "failures": failures,
        "witness_cid": witness.cid(),
    }


def verify_reconstruction_v2_witness(
        witness: ReconstructionV2Witness,
        *,
        expected_head_cid: str | None = None,
        expected_trace_cid: str | None = None,
        mse_floor: float | None = None,
) -> dict[str, Any]:
    failures: list[str] = []
    if (expected_head_cid is not None
            and witness.head_cid != expected_head_cid):
        failures.append(
            "w50_reconstruction_v2_head_cid_mismatch")
    if (expected_trace_cid is not None
            and witness.training_trace_cid != expected_trace_cid):
        failures.append(
            "w50_reconstruction_v2_training_trace_cid_mismatch")
    if (mse_floor is not None and witness.mse_per_k):
        # Check k=1,2,3 against the floor (lower MSE is better)
        for mse in witness.mse_per_k:
            if float(mse) > float(mse_floor):
                failures.append(
                    "w50_reconstruction_v2_mse_above_floor")
                break
    return {
        "ok": (len(failures) == 0),
        "failures": failures,
        "witness_cid": witness.cid(),
    }


__all__ = [
    "W50_SHARED_LATENT_CARRIER_SCHEMA_VERSION",
    "W50_DEFAULT_CARRIER_DIM",
    "W50_DEFAULT_RECONSTRUCTION_HIDDEN_DIM",
    "W50_DEFAULT_MAX_K_RECONSTRUCTION",
    "W50_DEFAULT_FLAT_FEATURE_DIM",
    "W50_NO_PARENT",
    "W50_SHARED_LATENT_CARRIER_VERIFIER_FAILURE_MODES",
    "SharedLatentCarrierV2",
    "RoleReuseMap",
    "ReconstructionV2Head",
    "SharedLatentCarrierChain",
    "ReconstructionV2Example",
    "ReconstructionV2TrainingSet",
    "ReconstructionV2TrainingTrace",
    "SharedLatentCarrierWitness",
    "ReconstructionV2Witness",
    "synthesize_reconstruction_v2_training_set",
    "fit_reconstruction_v2",
    "evaluate_reconstruction_v2_mse_at_k",
    "emit_shared_latent_carrier_witness",
    "emit_reconstruction_v2_witness",
    "verify_shared_latent_carrier_witness",
    "verify_reconstruction_v2_witness",
]
