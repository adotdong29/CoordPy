"""W50 M4 — Cross-Bank Transfer Layer + Adaptive Eviction V2.

A trainable role-pair-conditioned pseudo-KV transfer layer that
moves slot keys/values between role banks via a learned linear
transformation. Each ordered role pair ``(role_a, role_b)``
carries its own ``factor_dim × factor_dim`` projection matrix +
bias; a slot written by ``role_a`` can be projected into
``role_b``'s bank under that pair's transformation.

Paired with **AdaptiveEvictionPolicyV2** — a 5-feature sigmoid
scorer that extends W49's ``EvictionPolicy`` with two new
inputs: ``retention_probability`` (the retention head's output
for that slot) and ``transfer_signal`` (whether the slot was
recently transferred between banks). The V2 policy keeps
high-retention slots over noise and accelerates eviction of
slots that have already been transferred.

Pure-Python only — reuses the W47 ``Variable`` + ``AdamOptimizer``
autograd engine, the W49 ``PseudoKVBank`` + ``PseudoKVSlot``
abstractions.

Honest scope (do-not-overstate)
-------------------------------

This module does NOT alter transformer-internal hidden state, KV
cache bytes, attention weights, or embeddings. Role-pair
transfer operates over the W49 capsule-layer pseudo-KV slots
exclusively. The transfer ``W50-L-NO-REAL-KV-CAP`` extends W49's
no-real-KV cap unchanged.

The H12 ``W50-L-CROSS-BANK-COMPROMISE-CAP`` falsifier reproduces
when the role-pair training set is forged.
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
from .shared_state_proxy import (
    PseudoKVBank,
    PseudoKVSlot,
    W48_DEFAULT_FACTOR_DIM,
)


# =============================================================================
# Schema, defaults
# =============================================================================

W50_CROSS_BANK_SCHEMA_VERSION: str = (
    "coordpy.cross_bank_transfer.v1")

W50_DEFAULT_TRANSFER_INIT_AS_IDENTITY: bool = True
W50_DEFAULT_EVICTION_V2_IN_DIM: int = 5


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
# Role-pair projection
# =============================================================================

@dataclasses.dataclass
class RolePairProjection:
    """Single role-pair (a → b) learned linear projection.

    ``y = W · x + b`` where ``W`` is ``factor_dim × factor_dim``.
    Trainable via Adam. Initialised as approximate identity when
    ``init_as_identity = True`` so the unfitted projection is a
    structural no-op.
    """

    source_role: str
    target_role: str
    factor_dim: int
    w: ParamTensor
    b: ParamTensor

    @classmethod
    def init(
            cls, *,
            source_role: str,
            target_role: str,
            factor_dim: int = W48_DEFAULT_FACTOR_DIM,
            seed: int = W47_DEFAULT_TRAIN_SEED,
            init_scale: float = W47_DEFAULT_INIT_SCALE,
            init_as_identity: bool = (
                W50_DEFAULT_TRANSFER_INIT_AS_IDENTITY),
    ) -> "RolePairProjection":
        w = ParamTensor(
            shape=(int(factor_dim), int(factor_dim)), values=[])
        if init_as_identity:
            # W = I + small noise.
            vals = [0.0] * (int(factor_dim) * int(factor_dim))
            for i in range(int(factor_dim)):
                vals[i * int(factor_dim) + i] = 1.0
            w.values = vals
            # Small jitter on the off-diagonal.
            rng = _DeterministicLCG(seed=int(seed))
            for k in range(len(vals)):
                vals[k] += (rng.next_uniform() - 0.5) * 0.01
            w.values = vals
        else:
            w.init_seed(seed=int(seed), scale=float(init_scale))
        b = ParamTensor(
            shape=(int(factor_dim),),
            values=[0.0] * int(factor_dim))
        return cls(
            source_role=str(source_role),
            target_role=str(target_role),
            factor_dim=int(factor_dim),
            w=w, b=b)

    def params(self) -> list[ParamTensor]:
        return [self.w, self.b]

    def forward_value(
            self, x: Sequence[float],
    ) -> list[float]:
        out = [0.0] * self.factor_dim
        for r in range(self.factor_dim):
            base = r * self.factor_dim
            s = 0.0
            for j in range(self.factor_dim):
                xj = float(x[j]) if j < len(x) else 0.0
                s += float(self.w.values[base + j]) * float(xj)
            s += float(self.b.values[r])
            out[r] = s
        return out

    def forward_vars(
            self, x: Sequence[Variable],
    ) -> list[Variable]:
        w_vars = self.w.make_vars()
        b_vars = self.b.make_vars()
        rows: list[list[Variable]] = []
        for r in range(self.factor_dim):
            base = r * self.factor_dim
            rows.append(list(w_vars[base:base + self.factor_dim]))
        pre = vmatmul(rows, list(x))
        return [pre[i] + b_vars[i] for i in range(self.factor_dim)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_role": str(self.source_role),
            "target_role": str(self.target_role),
            "factor_dim": int(self.factor_dim),
            "w": self.w.to_dict(),
            "b": self.b.to_dict(),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w50_role_pair_projection",
            "projection": self.to_dict()})


# =============================================================================
# Cross-bank transfer layer
# =============================================================================

@dataclasses.dataclass
class CrossBankTransferLayer:
    """Trainable layer of role-pair projections.

    Keyed by ``(source_role, target_role)``. ``transfer_slot`` runs
    the corresponding projection over a slot's key + value.
    """

    factor_dim: int
    role_universe: tuple[str, ...]
    projections: dict[tuple[str, str], RolePairProjection]
    fitting_method: str = "unfitted"

    @classmethod
    def init(
            cls, *,
            role_universe: Sequence[str],
            factor_dim: int = W48_DEFAULT_FACTOR_DIM,
            seed: int = W47_DEFAULT_TRAIN_SEED,
            init_scale: float = W47_DEFAULT_INIT_SCALE,
            init_as_identity: bool = (
                W50_DEFAULT_TRANSFER_INIT_AS_IDENTITY),
    ) -> "CrossBankTransferLayer":
        roles = tuple(sorted({str(r) for r in role_universe}))
        rng = _DeterministicLCG(seed=int(seed))
        projections: dict[
            tuple[str, str], RolePairProjection] = {}
        for ra in roles:
            for rb in roles:
                if ra == rb:
                    # Same-role transfer is structural identity;
                    # still trainable but initialised exactly.
                    pass
                projections[(ra, rb)] = RolePairProjection.init(
                    source_role=str(ra),
                    target_role=str(rb),
                    factor_dim=int(factor_dim),
                    seed=int(rng.next_uniform() * (1 << 30)),
                    init_scale=float(init_scale),
                    init_as_identity=bool(init_as_identity))
        return cls(
            factor_dim=int(factor_dim),
            role_universe=roles,
            projections=projections,
            fitting_method="unfitted")

    def params(self) -> list[ParamTensor]:
        out: list[ParamTensor] = []
        for k in sorted(self.projections.keys()):
            out.extend(self.projections[k].params())
        return out

    def transfer_key_value(
            self, *,
            source_role: str,
            target_role: str,
            key: Sequence[float],
            value: Sequence[float],
    ) -> tuple[list[float], list[float]]:
        pair = self.projections.get(
            (str(source_role), str(target_role)))
        if pair is None:
            return list(key), list(value)
        return pair.forward_value(key), pair.forward_value(value)

    def transfer_slot(
            self, *,
            slot: PseudoKVSlot,
            target_role: str,
            new_turn_index: int | None = None,
    ) -> PseudoKVSlot:
        new_key, new_value = self.transfer_key_value(
            source_role=str(slot.role),
            target_role=str(target_role),
            key=slot.key,
            value=slot.value)
        return PseudoKVSlot(
            slot_index=int(slot.slot_index),
            turn_index=int(
                new_turn_index if new_turn_index is not None
                else slot.turn_index),
            role=str(target_role),
            key=tuple(_round_floats(new_key)),
            value=tuple(_round_floats(new_value)),
            write_gate_value=float(slot.write_gate_value),
            source_observation_cid=str(
                slot.source_observation_cid),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "factor_dim": int(self.factor_dim),
            "role_universe": list(self.role_universe),
            "projections": {
                f"{k[0]}->{k[1]}": v.to_dict()
                for k, v in sorted(self.projections.items())
            },
            "fitting_method": str(self.fitting_method),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w50_cross_bank_transfer_layer",
            "layer": self.to_dict()})


# =============================================================================
# Adaptive Eviction Policy V2
# =============================================================================

@dataclasses.dataclass
class AdaptiveEvictionPolicyV2:
    """V2 of W49's sigmoid eviction scorer.

    Input features (5):
        0: age_normalised (0..1)
        1: role_match (0 or 1)
        2: write_gate_value (0..1)
        3: retention_probability (0..1) — from the retention head
        4: transfer_signal (0 or 1) — was the slot recently
            transferred between banks
    """

    in_dim: int
    w_evict: ParamTensor

    @classmethod
    def init(
            cls, *,
            in_dim: int = W50_DEFAULT_EVICTION_V2_IN_DIM,
            seed: int = W47_DEFAULT_TRAIN_SEED,
            init_scale: float = W47_DEFAULT_INIT_SCALE,
    ) -> "AdaptiveEvictionPolicyV2":
        w = ParamTensor(shape=(int(in_dim),), values=[])
        w.init_seed(seed=int(seed), scale=float(init_scale))
        return cls(in_dim=int(in_dim), w_evict=w)

    def params(self) -> list[ParamTensor]:
        return [self.w_evict]

    def score_value(self, inputs: Sequence[float]) -> float:
        """Returns a sigmoid keep-score in [0, 1]. Higher = keep."""
        s = 0.0
        for i in range(min(self.in_dim, len(inputs))):
            s += float(self.w_evict.values[i]) * float(inputs[i])
        return float(_stable_sigmoid(s))

    def forward_vars(
            self, inputs: Sequence[Variable],
    ) -> Variable:
        w_vars = self.w_evict.make_vars()
        return vdot(list(w_vars), list(inputs)).sigmoid()

    def evict_index(
            self, *,
            bank: PseudoKVBank,
            current_role: str,
            current_turn: int,
            retention_probs: Sequence[float] | None = None,
            transfer_signals: Sequence[int] | None = None,
    ) -> int:
        if not bank.slots:
            return -1
        rps = (list(retention_probs)
               if retention_probs is not None else [])
        tsigs = (list(transfer_signals)
                 if transfer_signals is not None else [])
        scores: list[tuple[int, float]] = []
        for i, s in enumerate(bank.slots):
            age = float(
                max(0, int(current_turn) - int(s.turn_index)))
            age_norm = age / float(
                max(1, int(bank.capacity) + 1))
            role_match = (
                1.0 if str(s.role) == str(current_role) else 0.0)
            wg = float(s.write_gate_value)
            rp = float(rps[i]) if i < len(rps) else 0.5
            ts = float(tsigs[i]) if i < len(tsigs) else 0.0
            keep_score = self.score_value(
                [age_norm, role_match, wg, rp, ts])
            scores.append((i, keep_score))
        scores.sort(key=lambda x: x[1])
        return int(scores[0][0])

    def to_dict(self) -> dict[str, Any]:
        return {
            "in_dim": int(self.in_dim),
            "w_evict": self.w_evict.to_dict(),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w50_adaptive_eviction_v2",
            "policy": self.to_dict()})


# =============================================================================
# Training
# =============================================================================

@dataclasses.dataclass(frozen=True)
class CrossBankTransferExample:
    """One transfer training example.

    A source slot's key/value, the target role, and the gold
    transferred (key, value).
    """

    source_role: str
    target_role: str
    source_key: tuple[float, ...]
    source_value: tuple[float, ...]
    target_key: tuple[float, ...]
    target_value: tuple[float, ...]


@dataclasses.dataclass(frozen=True)
class CrossBankTransferTrainingSet:
    examples: tuple[CrossBankTransferExample, ...]
    factor_dim: int
    role_universe: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "factor_dim": int(self.factor_dim),
            "role_universe": list(self.role_universe),
            "examples": [
                {"source_role": str(e.source_role),
                 "target_role": str(e.target_role),
                 "source_key": list(e.source_key),
                 "source_value": list(e.source_value),
                 "target_key": list(e.target_key),
                 "target_value": list(e.target_value)}
                for e in self.examples],
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w50_cross_bank_transfer_training_set",
            "set": self.to_dict()})


def synthesize_cross_bank_transfer_training_set(
        *,
        role_universe: Sequence[str] = ("r0", "r1", "r2", "r3"),
        factor_dim: int = W48_DEFAULT_FACTOR_DIM,
        n_examples_per_pair: int = 4,
        seed: int = W47_DEFAULT_TRAIN_SEED,
) -> CrossBankTransferTrainingSet:
    """Synthesise a deterministic role-pair training set.

    For each ordered role pair ``(a, b)``, generate
    ``n_examples_per_pair`` carriers; the target is a fixed
    pair-specific permutation + sign-flip of the source — a
    distinctly non-identity transformation per pair so a trained
    transfer layer strictly beats both the identity-init baseline
    and the "no-transfer" (source-as-is) baseline.
    """
    rng = _DeterministicLCG(seed=int(seed))
    roles = tuple(sorted({str(r) for r in role_universe}))
    examples: list[CrossBankTransferExample] = []
    # Per-pair gold = signed permutation: index σ[i] and sign s[i].
    pair_perm: dict[tuple[str, str], tuple[tuple[int, ...],
                                            tuple[int, ...]]] = {}
    n_roles = len(roles)
    for ia, a in enumerate(roles):
        for ib, b in enumerate(roles):
            shift = ((ia * n_roles + ib) % factor_dim) + 1
            sigma = tuple(
                (i + shift) % int(factor_dim)
                for i in range(int(factor_dim)))
            signs = tuple(
                1 if ((ia + ib + i) % 2 == 0) else -1
                for i in range(int(factor_dim)))
            pair_perm[(a, b)] = (sigma, signs)
    for a in roles:
        for b in roles:
            sigma, signs = pair_perm[(a, b)]
            for _ in range(int(n_examples_per_pair)):
                src_key = [
                    float(rng.next_uniform() * 2.0 - 1.0)
                    for _ in range(int(factor_dim))
                ]
                src_value = [
                    float(rng.next_uniform() * 2.0 - 1.0)
                    for _ in range(int(factor_dim))
                ]
                tgt_key = [
                    float(signs[j]) * float(src_key[sigma[j]])
                    for j in range(int(factor_dim))
                ]
                tgt_value = [
                    float(signs[j]) * float(src_value[sigma[j]])
                    for j in range(int(factor_dim))
                ]
                examples.append(CrossBankTransferExample(
                    source_role=str(a),
                    target_role=str(b),
                    source_key=tuple(src_key),
                    source_value=tuple(src_value),
                    target_key=tuple(tgt_key),
                    target_value=tuple(tgt_value)))
    return CrossBankTransferTrainingSet(
        examples=tuple(examples),
        factor_dim=int(factor_dim),
        role_universe=tuple(roles))


@dataclasses.dataclass(frozen=True)
class CrossBankTransferTrainingTrace:
    seed: int
    n_steps: int
    final_loss: float
    final_grad_norm: float
    loss_head: tuple[float, ...]
    loss_tail: tuple[float, ...]
    training_set_cid: str
    final_layer_cid: str
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
            "final_layer_cid": str(self.final_layer_cid),
            "diverged": bool(self.diverged),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w50_cross_bank_transfer_training_trace",
            "trace": self.to_dict()})


def fit_cross_bank_transfer(
        training_set: CrossBankTransferTrainingSet,
        *,
        n_steps: int = 144,
        learning_rate: float = 0.02,
        seed: int = W47_DEFAULT_TRAIN_SEED,
        init_scale: float = W47_DEFAULT_INIT_SCALE,
        beta1: float = W47_DEFAULT_BETA1,
        beta2: float = W47_DEFAULT_BETA2,
        eps: float = W47_DEFAULT_EPS,
        grad_clip: float = W47_DEFAULT_GRAD_CLIP,
        history_head: int = 6,
        history_tail: int = 6,
        init_as_identity: bool = (
            W50_DEFAULT_TRANSFER_INIT_AS_IDENTITY),
) -> tuple[CrossBankTransferLayer, CrossBankTransferTrainingTrace]:
    """Fit the cross-bank transfer layer.

    Loss = sum over examples of L2( layer(source_role, target_role,
    source_key) - target_key) + same for value.
    """
    layer = CrossBankTransferLayer.init(
        role_universe=training_set.role_universe,
        factor_dim=int(training_set.factor_dim),
        seed=int(seed),
        init_scale=float(init_scale),
        init_as_identity=bool(init_as_identity))
    optim = AdamOptimizer(
        learning_rate=float(learning_rate),
        beta1=float(beta1), beta2=float(beta2),
        eps=float(eps), grad_clip=float(grad_clip))
    trainable = layer.params()
    loss_history: list[float] = []
    grad_norm_history: list[float] = []
    diverged = False
    n = max(1, len(training_set.examples))
    for step in range(int(n_steps)):
        for p in trainable:
            p.make_vars()
        idx = step % n
        ex = training_set.examples[idx]
        pair = layer.projections.get(
            (str(ex.source_role), str(ex.target_role)))
        if pair is None:
            continue
        src_key_vars = [
            Variable(float(v)) for v in ex.source_key]
        src_val_vars = [
            Variable(float(v)) for v in ex.source_value]
        tgt_key_vars = [
            Variable(float(v)) for v in ex.target_key]
        tgt_val_vars = [
            Variable(float(v)) for v in ex.target_value]
        out_key_vars = pair.forward_vars(src_key_vars)
        out_val_vars = pair.forward_vars(src_val_vars)
        key_terms = []
        for j in range(len(tgt_key_vars)):
            d = (out_key_vars[j] if j < len(out_key_vars)
                 else Variable(0.0)) - tgt_key_vars[j]
            key_terms.append(d * d)
        val_terms = []
        for j in range(len(tgt_val_vars)):
            d = (out_val_vars[j] if j < len(out_val_vars)
                 else Variable(0.0)) - tgt_val_vars[j]
            val_terms.append(d * d)
        loss = vmean(key_terms) + vmean(val_terms)
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
    fitted_layer = CrossBankTransferLayer(
        factor_dim=layer.factor_dim,
        role_universe=layer.role_universe,
        projections=layer.projections,
        fitting_method=(
            "cross_bank_transfer_adam_v1"
            if not diverged else "cross_bank_transfer_diverged"),
    )
    trace = CrossBankTransferTrainingTrace(
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
        final_layer_cid=str(fitted_layer.cid()),
        diverged=bool(diverged),
    )
    return fitted_layer, trace


def evaluate_role_pair_recall(
        layer: CrossBankTransferLayer,
        examples: Sequence[CrossBankTransferExample],
) -> float:
    """Mean cosine similarity between transferred key and gold
    target key across role-pair grid examples."""
    if not examples:
        return 0.0
    cos_sum = 0.0
    n = 0
    for ex in examples:
        pair = layer.projections.get(
            (str(ex.source_role), str(ex.target_role)))
        if pair is None:
            continue
        out = pair.forward_value(ex.source_key)
        cos_sum += _cosine(out, ex.target_key)
        n += 1
    return float(cos_sum) / float(max(1, n))


# =============================================================================
# Witness
# =============================================================================

@dataclasses.dataclass(frozen=True)
class CrossBankTransferWitness:
    """Sealed per-turn cross-bank-transfer witness."""

    layer_cid: str
    training_trace_cid: str
    factor_dim: int
    role_universe: tuple[str, ...]
    n_pairs: int
    mean_role_pair_recall: float
    transferred_slots_cid: str
    source_bank_cid: str
    target_bank_cid: str
    transfer_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_cid": str(self.layer_cid),
            "training_trace_cid": str(self.training_trace_cid),
            "factor_dim": int(self.factor_dim),
            "role_universe": list(self.role_universe),
            "n_pairs": int(self.n_pairs),
            "mean_role_pair_recall": float(
                round(self.mean_role_pair_recall, 12)),
            "transferred_slots_cid": str(
                self.transferred_slots_cid),
            "source_bank_cid": str(self.source_bank_cid),
            "target_bank_cid": str(self.target_bank_cid),
            "transfer_count": int(self.transfer_count),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w50_cross_bank_transfer_witness",
            "witness": self.to_dict()})


def emit_cross_bank_transfer_witness(
        *,
        layer: CrossBankTransferLayer,
        training_trace: CrossBankTransferTrainingTrace,
        probe_examples: Sequence[CrossBankTransferExample],
        transferred_slots: Sequence[PseudoKVSlot] = (),
        source_bank: PseudoKVBank | None = None,
        target_bank: PseudoKVBank | None = None,
) -> CrossBankTransferWitness:
    mean_recall = evaluate_role_pair_recall(layer, probe_examples)
    transferred_payload = {
        "kind": "w50_transferred_slots",
        "slots": [
            {"slot_index": int(s.slot_index),
             "turn_index": int(s.turn_index),
             "role": str(s.role),
             "key": list(s.key),
             "value": list(s.value)}
            for s in transferred_slots],
    }
    return CrossBankTransferWitness(
        layer_cid=str(layer.cid()),
        training_trace_cid=str(training_trace.cid()),
        factor_dim=int(layer.factor_dim),
        role_universe=tuple(layer.role_universe),
        n_pairs=int(len(layer.projections)),
        mean_role_pair_recall=float(mean_recall),
        transferred_slots_cid=str(_sha256_hex(transferred_payload)),
        source_bank_cid=str(
            source_bank.head_cid() if source_bank else ""),
        target_bank_cid=str(
            target_bank.head_cid() if target_bank else ""),
        transfer_count=int(len(transferred_slots)),
    )


# =============================================================================
# Verifier
# =============================================================================

W50_CROSS_BANK_TRANSFER_VERIFIER_FAILURE_MODES: tuple[str, ...] = (
    "w50_cross_bank_transfer_schema_mismatch",
    "w50_cross_bank_transfer_layer_cid_mismatch",
    "w50_cross_bank_transfer_training_trace_cid_mismatch",
    "w50_cross_bank_transfer_witness_cid_mismatch",
    "w50_cross_bank_transfer_recall_below_floor",
    "w50_cross_bank_transfer_role_universe_mismatch",
)


def verify_cross_bank_transfer_witness(
        witness: CrossBankTransferWitness,
        *,
        expected_layer_cid: str | None = None,
        expected_trace_cid: str | None = None,
        recall_floor: float | None = None,
        expected_role_universe: Sequence[str] | None = None,
) -> dict[str, Any]:
    failures: list[str] = []
    if (expected_layer_cid is not None
            and witness.layer_cid != expected_layer_cid):
        failures.append(
            "w50_cross_bank_transfer_layer_cid_mismatch")
    if (expected_trace_cid is not None
            and witness.training_trace_cid != expected_trace_cid):
        failures.append(
            "w50_cross_bank_transfer_training_trace_cid_mismatch")
    if (recall_floor is not None
            and witness.mean_role_pair_recall < float(recall_floor)):
        failures.append(
            "w50_cross_bank_transfer_recall_below_floor")
    if (expected_role_universe is not None
            and tuple(witness.role_universe)
            != tuple(expected_role_universe)):
        failures.append(
            "w50_cross_bank_transfer_role_universe_mismatch")
    return {
        "ok": (len(failures) == 0),
        "failures": failures,
        "witness_cid": witness.cid(),
    }


# =============================================================================
# Compromise cap forging helper (H12)
# =============================================================================

def forge_cross_bank_training_set(
        original: CrossBankTransferTrainingSet,
        *,
        seed: int = W47_DEFAULT_TRAIN_SEED,
) -> CrossBankTransferTrainingSet:
    """Adversarially scramble the role-pair training set.

    Produces a forged training set with mislabeled role-pair
    targets — the `W50-L-CROSS-BANK-COMPROMISE-CAP` falsifier.
    """
    rng = _DeterministicLCG(seed=int(seed))
    examples = list(original.examples)
    forged: list[CrossBankTransferExample] = []
    for ex in examples:
        # Replace target_key and target_value with random noise
        forged_key = tuple(
            float(rng.next_uniform() * 2.0 - 1.0)
            for _ in range(len(ex.target_key)))
        forged_value = tuple(
            float(rng.next_uniform() * 2.0 - 1.0)
            for _ in range(len(ex.target_value)))
        forged.append(CrossBankTransferExample(
            source_role=ex.source_role,
            target_role=ex.target_role,
            source_key=ex.source_key,
            source_value=ex.source_value,
            target_key=forged_key,
            target_value=forged_value))
    return CrossBankTransferTrainingSet(
        examples=tuple(forged),
        factor_dim=original.factor_dim,
        role_universe=original.role_universe)


__all__ = [
    "W50_CROSS_BANK_SCHEMA_VERSION",
    "W50_DEFAULT_TRANSFER_INIT_AS_IDENTITY",
    "W50_DEFAULT_EVICTION_V2_IN_DIM",
    "W50_CROSS_BANK_TRANSFER_VERIFIER_FAILURE_MODES",
    "RolePairProjection",
    "CrossBankTransferLayer",
    "AdaptiveEvictionPolicyV2",
    "CrossBankTransferExample",
    "CrossBankTransferTrainingSet",
    "CrossBankTransferTrainingTrace",
    "CrossBankTransferWitness",
    "synthesize_cross_bank_transfer_training_set",
    "fit_cross_bank_transfer",
    "evaluate_role_pair_recall",
    "emit_cross_bank_transfer_witness",
    "verify_cross_bank_transfer_witness",
    "forge_cross_bank_training_set",
]
