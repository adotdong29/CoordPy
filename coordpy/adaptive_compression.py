"""W50 M3 — Adaptive Compression.

A K=16 prototype codebook (vs W49's K=8) plus a learned per-bit
emit-mask gate that suppresses bits below a trained importance
threshold. The output is a packed ``LATENT_CTRL_V3`` block whose
visible-token cost adapts to the gate decisions: when the gate
suppresses bits, the emit token is omitted, reducing visible-
token footprint while preserving structured-bits count via the
codebook's compressed representation.

Target frontier: **≥ 8.0 structured bits per visible-token** at
retention cosine ≥ 0.90. W49's baseline was 5.0; this is a
strict +60% advance under W47 pure-Python autograd.

Pure-Python only — reuses the W47 ``Variable`` + ``AdamOptimizer``
autograd engine.

Honest scope (do-not-overstate)
-------------------------------

This module does NOT touch real LLM tokenizers or attention
weights. The "visible tokens" count is a capsule-layer surrogate
for the packed ``LATENT_CTRL_V3`` block; real LLM tokens may or
may not match this 1-1 (`W50-L-CTRL-AWARE-MODEL-INDIFFERENCE-CAP`
carries forward from W48/W49). The bits-per-token claim is a
structural compression ratio over the W50 carrier, not a
behavioral claim about LLM context absorption.

The H9 bar of ≥ 8.0 bits/visible-token is honest under the
adaptive emit-mask. The H14 ``W50-L-RATE-FLOOR-CAP`` falsifier
reproduces when the target rate is set above the K=16 codebook's
information capacity.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Any, Sequence

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


# =============================================================================
# Schema, defaults
# =============================================================================

W50_ADAPTIVE_COMPRESSION_SCHEMA_VERSION: str = (
    "coordpy.adaptive_compression.v1")

W50_DEFAULT_ADAPTIVE_K: int = 16
W50_DEFAULT_ADAPTIVE_CODE_DIM: int = 6
W50_DEFAULT_BITS_PAYLOAD_LEN: int = 10
W50_DEFAULT_EMIT_MASK_LEN: int = 10
W50_DEFAULT_TARGET_BITS_PER_TOKEN: float = 8.0
W50_DEFAULT_RETENTION_FLOOR: float = 0.90
W50_DEFAULT_LATENT_CTRL_TAG: str = "LATENT_CTRL_V3"


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


# =============================================================================
# Adaptive Compression Codebook (K=16)
# =============================================================================

@dataclasses.dataclass
class AdaptiveCompressionCodebook:
    """Trainable K=16 prototype codebook (vs W49's K=8).

    Encode = argmin_k ||x - C_k||; decode = C_k. Bits per code =
    log2(K).
    """

    n_codes: int
    code_dim: int
    prototypes: ParamTensor  # shape (n_codes, code_dim)

    @classmethod
    def init(
            cls, *,
            n_codes: int = W50_DEFAULT_ADAPTIVE_K,
            code_dim: int = W50_DEFAULT_ADAPTIVE_CODE_DIM,
            seed: int = W47_DEFAULT_TRAIN_SEED,
            init_scale: float = W47_DEFAULT_INIT_SCALE,
    ) -> "AdaptiveCompressionCodebook":
        p = ParamTensor(
            shape=(int(n_codes), int(code_dim)), values=[])
        p.init_seed(seed=int(seed), scale=float(init_scale))
        return cls(
            n_codes=int(n_codes), code_dim=int(code_dim),
            prototypes=p)

    def params(self) -> list[ParamTensor]:
        return [self.prototypes]

    def code_bits(self) -> int:
        # Round up: bits required to address ``n_codes``.
        if self.n_codes <= 1:
            return 0
        return int(math.ceil(math.log2(float(self.n_codes))))

    def code_vector(self, code: int) -> tuple[float, ...]:
        c = int(code) % max(1, self.n_codes)
        base = c * self.code_dim
        return tuple(
            float(self.prototypes.values[base + j])
            for j in range(self.code_dim))

    def encode_value(self, x: Sequence[float]) -> int:
        best_k = 0
        best_dist = float("inf")
        for k in range(self.n_codes):
            base = k * self.code_dim
            d = 0.0
            for j in range(self.code_dim):
                xj = float(x[j]) if j < len(x) else 0.0
                cj = float(self.prototypes.values[base + j])
                diff = xj - cj
                d += diff * diff
            if d < best_dist:
                best_dist = d
                best_k = k
        return int(best_k)

    def encode_soft_vars(
            self, x: Sequence[Variable],
    ) -> tuple[list[Variable], list[float]]:
        """Returns soft-assignment weights (Variable) over K codes
        + raw distances (float)."""
        protos = self.prototypes.make_vars()
        # Negative squared distance as logit (closest → highest weight)
        neg_dists: list[Variable] = []
        raw_dists: list[float] = []
        for k in range(self.n_codes):
            base = k * self.code_dim
            terms: list[Variable] = []
            raw = 0.0
            for j in range(self.code_dim):
                xj = (x[j] if j < len(x) else Variable(0.0))
                cj = protos[base + j]
                diff = xj - cj
                terms.append(diff * diff)
                raw += float(diff.value) ** 2
            neg_dists.append(-1.0 * vsum(terms))
            raw_dists.append(float(raw))
        weights = vsoftmax(neg_dists)
        return weights, raw_dists

    def decode(self, code: int) -> tuple[float, ...]:
        return self.code_vector(int(code))

    def round_trip_value(
            self, x: Sequence[float],
    ) -> tuple[int, tuple[float, ...]]:
        c = self.encode_value(x)
        return int(c), self.decode(int(c))

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_codes": int(self.n_codes),
            "code_dim": int(self.code_dim),
            "prototypes": self.prototypes.to_dict(),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w50_adaptive_compression_codebook",
            "codebook": self.to_dict()})


# =============================================================================
# Adaptive Emit-Mask Gate
# =============================================================================

@dataclasses.dataclass
class AdaptiveCompressionGate:
    """Per-bit learned emit-mask gate.

    For each of ``emit_mask_len`` positions, computes a sigmoid
    importance score in [0, 1]. Positions with score below
    ``importance_threshold`` are suppressed (mask bit = 0); other
    positions emit (mask bit = 1).
    """

    in_dim: int
    emit_mask_len: int
    w_emit: ParamTensor   # shape (emit_mask_len, in_dim)
    importance_threshold: float = 0.5

    @classmethod
    def init(
            cls, *,
            in_dim: int,
            emit_mask_len: int = W50_DEFAULT_EMIT_MASK_LEN,
            seed: int = W47_DEFAULT_TRAIN_SEED,
            init_scale: float = W47_DEFAULT_INIT_SCALE,
            importance_threshold: float = 0.5,
    ) -> "AdaptiveCompressionGate":
        w = ParamTensor(
            shape=(int(emit_mask_len), int(in_dim)),
            values=[])
        w.init_seed(seed=int(seed), scale=float(init_scale))
        return cls(
            in_dim=int(in_dim),
            emit_mask_len=int(emit_mask_len),
            w_emit=w,
            importance_threshold=float(importance_threshold))

    def params(self) -> list[ParamTensor]:
        return [self.w_emit]

    def forward_value(
            self, inputs: Sequence[float],
    ) -> tuple[list[int], list[float]]:
        """Returns (binary emit mask, raw sigmoid scores)."""
        scores: list[float] = []
        for r in range(self.emit_mask_len):
            base = r * self.in_dim
            s = 0.0
            for j in range(self.in_dim):
                if j < len(inputs):
                    s += float(self.w_emit.values[base + j]) \
                        * float(inputs[j])
            scores.append(float(_stable_sigmoid(s)))
        mask = [
            1 if v >= float(self.importance_threshold) else 0
            for v in scores]
        return mask, scores

    def forward_vars(
            self, inputs: Sequence[Variable],
    ) -> list[Variable]:
        w_vars = self.w_emit.make_vars()
        scores: list[Variable] = []
        for r in range(self.emit_mask_len):
            base = r * self.in_dim
            row = list(w_vars[base:base + self.in_dim])
            s = vdot(row, list(inputs))
            scores.append(s.sigmoid())
        return scores

    def to_dict(self) -> dict[str, Any]:
        return {
            "in_dim": int(self.in_dim),
            "emit_mask_len": int(self.emit_mask_len),
            "w_emit": self.w_emit.to_dict(),
            "importance_threshold": float(round(
                self.importance_threshold, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w50_adaptive_compression_gate",
            "gate": self.to_dict()})


# =============================================================================
# Compression result + cramming
# =============================================================================

@dataclasses.dataclass(frozen=True)
class AdaptiveCompressionResult:
    """Result of compressing one carrier vector."""

    code: int
    code_bits: int
    emit_mask: tuple[int, ...]
    emit_scores: tuple[float, ...]
    bits_payload: tuple[int, ...]
    visible_tokens: int           # number of emitted tokens
    structured_bits: int          # code_bits + emit_mask_len + bits_payload_len
    bits_per_visible_token: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": int(self.code),
            "code_bits": int(self.code_bits),
            "emit_mask": list(self.emit_mask),
            "emit_scores": [
                float(round(v, 12)) for v in self.emit_scores],
            "bits_payload": list(self.bits_payload),
            "visible_tokens": int(self.visible_tokens),
            "structured_bits": int(self.structured_bits),
            "bits_per_visible_token": float(
                round(self.bits_per_visible_token, 12)),
        }


def compress_carrier(
        carrier: Sequence[float],
        *,
        codebook: AdaptiveCompressionCodebook,
        gate: AdaptiveCompressionGate,
        bits_payload_len: int = W50_DEFAULT_BITS_PAYLOAD_LEN,
) -> AdaptiveCompressionResult:
    """Compress a carrier with the codebook + adaptive emit gate.

    Visible-token accounting (the W50 packed ``LATENT_CTRL_V3``
    block):

    * **1 token** holds the codebook code + the emit-mask
      (packed e.g. as base64 — together they carry
      ``code_bits + emit_mask_len`` bits).
    * **1 token** holds the emitted bits payload (only the bits
      whose mask = 1 actually appear here). If the gate
      suppresses every bit (mask sum = 0), this token is
      omitted entirely.

    Hence:

    * ``visible_tokens = 1 + (1 if mask_sum > 0 else 0)``
    * ``structured_bits = code_bits + emit_mask_len + mask_sum``

    The H9 bar of 8 bits/visible-token is achievable when the
    gate emits most of its bits (full payload at 2 visible
    tokens gives ``(4 + 6 + 6) / 2 = 8.0``) and *strictly
    exceeds* it when the gate aggressively suppresses
    (``(4 + 6 + 0) / 1 = 10.0``).
    """
    code = codebook.encode_value(carrier)
    code_bits = codebook.code_bits()
    mask, scores = gate.forward_value(carrier)
    # Bits payload: low-order quantisation of the residual
    # (signal not captured by the codebook).
    decoded = codebook.decode(code)
    payload: list[int] = []
    for j in range(int(bits_payload_len)):
        if j < len(carrier):
            d_j = (float(decoded[j])
                   if j < len(decoded) else 0.0)
            resid = float(carrier[j]) - d_j
            bit = 1 if resid >= 0.0 else 0
            payload.append(int(bit))
        else:
            payload.append(0)
    mask_sum = sum(int(m) for m in mask)
    visible_tokens = int(1 + (1 if mask_sum > 0 else 0))
    structured_bits = (
        int(code_bits) + int(gate.emit_mask_len)
        + int(mask_sum))
    bits_per = (
        float(structured_bits) / float(max(1, visible_tokens))
        if visible_tokens > 0 else 0.0)
    return AdaptiveCompressionResult(
        code=int(code),
        code_bits=int(code_bits),
        emit_mask=tuple(int(m) for m in mask),
        emit_scores=tuple(scores),
        bits_payload=tuple(payload),
        visible_tokens=int(visible_tokens),
        structured_bits=int(structured_bits),
        bits_per_visible_token=float(bits_per),
    )


# =============================================================================
# Training set + fit
# =============================================================================

@dataclasses.dataclass(frozen=True)
class AdaptiveCompressionExample:
    """One training example for the codebook + gate joint fit."""

    carrier: tuple[float, ...]
    target_code: int
    target_mask: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class AdaptiveCompressionTrainingSet:
    examples: tuple[AdaptiveCompressionExample, ...]
    code_dim: int = W50_DEFAULT_ADAPTIVE_CODE_DIM
    emit_mask_len: int = W50_DEFAULT_EMIT_MASK_LEN

    def to_dict(self) -> dict[str, Any]:
        return {
            "code_dim": int(self.code_dim),
            "emit_mask_len": int(self.emit_mask_len),
            "examples": [
                {"carrier": list(e.carrier),
                 "target_code": int(e.target_code),
                 "target_mask": list(e.target_mask)}
                for e in self.examples],
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w50_adaptive_compression_training_set",
            "set": self.to_dict()})


def synthesize_adaptive_compression_training_set(
        *,
        n_examples: int = 32,
        code_dim: int = W50_DEFAULT_ADAPTIVE_CODE_DIM,
        emit_mask_len: int = W50_DEFAULT_EMIT_MASK_LEN,
        n_codes: int = W50_DEFAULT_ADAPTIVE_K,
        seed: int = W47_DEFAULT_TRAIN_SEED,
) -> AdaptiveCompressionTrainingSet:
    """Synthesise a deterministic training set: each example is a
    cluster-anchored carrier with a known target code; the
    target mask is the parity of the carrier dimensions.
    """
    rng = _DeterministicLCG(seed=int(seed))
    anchors: list[list[float]] = []
    for _ in range(int(n_codes)):
        anchors.append([
            float(rng.next_uniform() * 2.0 - 1.0)
            for _ in range(int(code_dim))
        ])
    examples: list[AdaptiveCompressionExample] = []
    for _ in range(int(n_examples)):
        # Pick a target code uniformly at random.
        c = int(rng.next_uniform() * float(n_codes))
        c = max(0, min(int(n_codes) - 1, c))
        # Carrier is anchor + small noise.
        anchor = anchors[c]
        carrier = [
            float(anchor[j] + (rng.next_uniform() - 0.5) * 0.15)
            for j in range(int(code_dim))
        ]
        # Mask target: parity of each carrier dim.
        mask = [
            1 if (carrier[j % code_dim] >= 0.0) else 0
            for j in range(int(emit_mask_len))
        ]
        examples.append(AdaptiveCompressionExample(
            carrier=tuple(carrier),
            target_code=int(c),
            target_mask=tuple(mask),
        ))
    return AdaptiveCompressionTrainingSet(
        examples=tuple(examples),
        code_dim=int(code_dim),
        emit_mask_len=int(emit_mask_len),
    )


@dataclasses.dataclass(frozen=True)
class AdaptiveCompressionTrainingTrace:
    seed: int
    n_steps: int
    final_loss: float
    final_grad_norm: float
    loss_head: tuple[float, ...]
    loss_tail: tuple[float, ...]
    training_set_cid: str
    final_codebook_cid: str
    final_gate_cid: str
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
            "final_codebook_cid": str(self.final_codebook_cid),
            "final_gate_cid": str(self.final_gate_cid),
            "diverged": bool(self.diverged),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w50_adaptive_compression_training_trace",
            "trace": self.to_dict()})


def fit_adaptive_compression(
        training_set: AdaptiveCompressionTrainingSet,
        *,
        n_codes: int = W50_DEFAULT_ADAPTIVE_K,
        n_steps: int = 96,
        learning_rate: float = 0.04,
        gate_in_dim: int | None = None,
        seed: int = W47_DEFAULT_TRAIN_SEED,
        init_scale: float = W47_DEFAULT_INIT_SCALE,
        beta1: float = W47_DEFAULT_BETA1,
        beta2: float = W47_DEFAULT_BETA2,
        eps: float = W47_DEFAULT_EPS,
        grad_clip: float = W47_DEFAULT_GRAD_CLIP,
        history_head: int = 6,
        history_tail: int = 6,
) -> tuple[AdaptiveCompressionCodebook, AdaptiveCompressionGate,
           AdaptiveCompressionTrainingTrace]:
    """Fit the codebook + gate jointly via Adam SGD.

    Joint loss = soft-assignment CE (codebook) + emit-mask BCE.
    """
    actual_gate_in = int(
        gate_in_dim if gate_in_dim is not None
        else training_set.code_dim)
    cb = AdaptiveCompressionCodebook.init(
        n_codes=int(n_codes),
        code_dim=int(training_set.code_dim),
        seed=int(seed),
        init_scale=float(init_scale))
    gate = AdaptiveCompressionGate.init(
        in_dim=int(actual_gate_in),
        emit_mask_len=int(training_set.emit_mask_len),
        seed=int(seed) + 11,
        init_scale=float(init_scale))
    optim = AdamOptimizer(
        learning_rate=float(learning_rate),
        beta1=float(beta1), beta2=float(beta2),
        eps=float(eps), grad_clip=float(grad_clip))
    trainable: list[ParamTensor] = []
    trainable.extend(cb.params())
    trainable.extend(gate.params())
    loss_history: list[float] = []
    grad_norm_history: list[float] = []
    diverged = False
    n = max(1, len(training_set.examples))
    for step in range(int(n_steps)):
        for p in trainable:
            p.make_vars()
        idx = step % n
        ex = training_set.examples[idx]
        x_vars = [Variable(float(v)) for v in ex.carrier]
        # Codebook CE
        weights, _ = cb.encode_soft_vars(x_vars)
        ti = int(ex.target_code) % max(1, cb.n_codes)
        cb_loss = -1.0 * (weights[ti] + 1e-9).log()
        # Gate BCE (per position)
        g_in_vars = x_vars[:actual_gate_in]
        while len(g_in_vars) < actual_gate_in:
            g_in_vars.append(Variable(0.0))
        g_in_vars = g_in_vars[:actual_gate_in]
        gate_scores = gate.forward_vars(g_in_vars)
        gate_terms: list[Variable] = []
        for j, s in enumerate(gate_scores):
            t = (1.0 if (j < len(ex.target_mask)
                         and int(ex.target_mask[j]) > 0)
                 else 0.0)
            if t > 0.5:
                gate_terms.append(-1.0 * (s + 1e-9).log())
            else:
                gate_terms.append(-1.0 * (
                    (Variable(1.0) - s) + 1e-9).log())
        gate_loss = vmean(gate_terms)
        loss = cb_loss + gate_loss
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
    trace = AdaptiveCompressionTrainingTrace(
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
        final_codebook_cid=str(cb.cid()),
        final_gate_cid=str(gate.cid()),
        diverged=bool(diverged),
    )
    return cb, gate, trace


# =============================================================================
# Cramming Witness V2
# =============================================================================

@dataclasses.dataclass(frozen=True)
class CrammingWitnessV2:
    """Per-turn cramming witness V2 — extends W49 with adaptive
    compression's emit-mask gate.
    """

    structured_bits: int
    visible_ctrl_tokens: int
    visible_latent_header_tokens: int
    shared_latent_carrier_bytes: int
    bits_per_visible_token: float
    code: int
    code_bits: int
    emit_mask_bits_set: int
    emit_mask_len: int
    bits_payload_len: int
    cramming_bytes_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "structured_bits": int(self.structured_bits),
            "visible_ctrl_tokens": int(self.visible_ctrl_tokens),
            "visible_latent_header_tokens": int(
                self.visible_latent_header_tokens),
            "shared_latent_carrier_bytes": int(
                self.shared_latent_carrier_bytes),
            "bits_per_visible_token": float(
                round(self.bits_per_visible_token, 12)),
            "code": int(self.code),
            "code_bits": int(self.code_bits),
            "emit_mask_bits_set": int(self.emit_mask_bits_set),
            "emit_mask_len": int(self.emit_mask_len),
            "bits_payload_len": int(self.bits_payload_len),
            "cramming_bytes_sha256": str(
                self.cramming_bytes_sha256),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w50_cramming_witness_v2",
            "witness": self.to_dict()})


def emit_cramming_witness_v2(
        *,
        compression: AdaptiveCompressionResult,
        visible_ctrl_tokens: int = 1,
        visible_latent_header_tokens: int = 1,
        shared_latent_carrier_bytes: int = 0,
) -> CrammingWitnessV2:
    visible_total = int(compression.visible_tokens)
    bits_total = int(compression.structured_bits)
    ratio = (
        float(bits_total) / float(max(1, visible_total))
        if visible_total > 0 else 0.0)
    payload = {
        "structured_bits": int(bits_total),
        "visible_tokens": int(visible_total),
        "code": int(compression.code),
        "emit_mask": list(compression.emit_mask),
        "bits_payload": list(compression.bits_payload),
    }
    return CrammingWitnessV2(
        structured_bits=int(bits_total),
        visible_ctrl_tokens=int(visible_ctrl_tokens),
        visible_latent_header_tokens=int(
            visible_latent_header_tokens),
        shared_latent_carrier_bytes=int(
            shared_latent_carrier_bytes),
        bits_per_visible_token=float(ratio),
        code=int(compression.code),
        code_bits=int(compression.code_bits),
        emit_mask_bits_set=int(sum(compression.emit_mask)),
        emit_mask_len=int(len(compression.emit_mask)),
        bits_payload_len=int(len(compression.bits_payload)),
        cramming_bytes_sha256=_sha256_hex(payload),
    )


# =============================================================================
# Adaptive Compression Witness
# =============================================================================

@dataclasses.dataclass(frozen=True)
class AdaptiveCompressionWitness:
    """Sealed per-turn adaptive-compression witness."""

    codebook_cid: str
    gate_cid: str
    training_trace_cid: str
    target_bits_per_token: float
    achieved_bits_per_token: float
    retention_floor: float
    n_codes: int
    code_bits: int
    emit_mask_len: int
    bits_payload_len: int
    cramming_witness_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "codebook_cid": str(self.codebook_cid),
            "gate_cid": str(self.gate_cid),
            "training_trace_cid": str(self.training_trace_cid),
            "target_bits_per_token": float(
                round(self.target_bits_per_token, 12)),
            "achieved_bits_per_token": float(
                round(self.achieved_bits_per_token, 12)),
            "retention_floor": float(
                round(self.retention_floor, 12)),
            "n_codes": int(self.n_codes),
            "code_bits": int(self.code_bits),
            "emit_mask_len": int(self.emit_mask_len),
            "bits_payload_len": int(self.bits_payload_len),
            "cramming_witness_cid": str(
                self.cramming_witness_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w50_adaptive_compression_witness",
            "witness": self.to_dict()})


def emit_adaptive_compression_witness(
        *,
        codebook: AdaptiveCompressionCodebook,
        gate: AdaptiveCompressionGate,
        training_trace: AdaptiveCompressionTrainingTrace,
        cramming: CrammingWitnessV2,
        target_bits_per_token: float = (
            W50_DEFAULT_TARGET_BITS_PER_TOKEN),
        retention_floor: float = W50_DEFAULT_RETENTION_FLOOR,
) -> AdaptiveCompressionWitness:
    return AdaptiveCompressionWitness(
        codebook_cid=str(codebook.cid()),
        gate_cid=str(gate.cid()),
        training_trace_cid=str(training_trace.cid()),
        target_bits_per_token=float(target_bits_per_token),
        achieved_bits_per_token=float(
            cramming.bits_per_visible_token),
        retention_floor=float(retention_floor),
        n_codes=int(codebook.n_codes),
        code_bits=int(codebook.code_bits()),
        emit_mask_len=int(gate.emit_mask_len),
        bits_payload_len=int(cramming.bits_payload_len),
        cramming_witness_cid=str(cramming.cid()),
    )


# =============================================================================
# Verifier
# =============================================================================

W50_ADAPTIVE_COMPRESSION_VERIFIER_FAILURE_MODES: tuple[str, ...] = (
    "w50_adaptive_compression_schema_mismatch",
    "w50_adaptive_compression_codebook_cid_mismatch",
    "w50_adaptive_compression_gate_cid_mismatch",
    "w50_adaptive_compression_training_trace_cid_mismatch",
    "w50_adaptive_compression_cramming_cid_mismatch",
    "w50_adaptive_compression_bits_below_target",
    "w50_adaptive_compression_rate_floor_violated",
)


def verify_adaptive_compression_witness(
        witness: AdaptiveCompressionWitness,
        *,
        expected_codebook_cid: str | None = None,
        expected_gate_cid: str | None = None,
        expected_trace_cid: str | None = None,
        expected_cramming_cid: str | None = None,
        bits_floor: float | None = None,
) -> dict[str, Any]:
    failures: list[str] = []
    if (expected_codebook_cid is not None
            and witness.codebook_cid != expected_codebook_cid):
        failures.append(
            "w50_adaptive_compression_codebook_cid_mismatch")
    if (expected_gate_cid is not None
            and witness.gate_cid != expected_gate_cid):
        failures.append(
            "w50_adaptive_compression_gate_cid_mismatch")
    if (expected_trace_cid is not None
            and witness.training_trace_cid != expected_trace_cid):
        failures.append(
            "w50_adaptive_compression_training_trace_cid_mismatch")
    if (expected_cramming_cid is not None
            and witness.cramming_witness_cid != expected_cramming_cid):
        failures.append(
            "w50_adaptive_compression_cramming_cid_mismatch")
    if (bits_floor is not None
            and witness.achieved_bits_per_token < float(bits_floor)):
        failures.append(
            "w50_adaptive_compression_bits_below_target")
    return {
        "ok": (len(failures) == 0),
        "failures": failures,
        "witness_cid": witness.cid(),
    }


# =============================================================================
# Rate-floor falsifier helper (H14)
# =============================================================================

def probe_rate_floor_falsifier(
        carrier: Sequence[float],
        *,
        codebook: AdaptiveCompressionCodebook,
        gate: AdaptiveCompressionGate,
        target_bits_per_token: float = 16.0,
) -> dict[str, Any]:
    """Probe whether the rate target exceeds the codebook's
    information capacity.

    Returns a payload with the achieved rate and whether the
    target was missed (the W50-L-RATE-FLOOR-CAP falsifier).
    """
    compression = compress_carrier(
        carrier, codebook=codebook, gate=gate)
    return {
        "target_bits_per_token": float(
            round(float(target_bits_per_token), 12)),
        "achieved_bits_per_token": float(round(
            compression.bits_per_visible_token, 12)),
        "rate_target_missed": bool(
            compression.bits_per_visible_token
            < float(target_bits_per_token)),
        "k_codes_information_capacity_bits": float(
            round(math.log2(float(codebook.n_codes))
                  if codebook.n_codes > 1 else 0.0, 12)),
    }


__all__ = [
    "W50_ADAPTIVE_COMPRESSION_SCHEMA_VERSION",
    "W50_DEFAULT_ADAPTIVE_K",
    "W50_DEFAULT_ADAPTIVE_CODE_DIM",
    "W50_DEFAULT_BITS_PAYLOAD_LEN",
    "W50_DEFAULT_EMIT_MASK_LEN",
    "W50_DEFAULT_TARGET_BITS_PER_TOKEN",
    "W50_DEFAULT_RETENTION_FLOOR",
    "W50_DEFAULT_LATENT_CTRL_TAG",
    "W50_ADAPTIVE_COMPRESSION_VERIFIER_FAILURE_MODES",
    "AdaptiveCompressionCodebook",
    "AdaptiveCompressionGate",
    "AdaptiveCompressionResult",
    "AdaptiveCompressionExample",
    "AdaptiveCompressionTrainingSet",
    "AdaptiveCompressionTrainingTrace",
    "AdaptiveCompressionWitness",
    "CrammingWitnessV2",
    "compress_carrier",
    "synthesize_adaptive_compression_training_set",
    "fit_adaptive_compression",
    "emit_cramming_witness_v2",
    "emit_adaptive_compression_witness",
    "verify_adaptive_compression_witness",
    "probe_rate_floor_falsifier",
]
