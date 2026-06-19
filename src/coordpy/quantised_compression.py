"""W52 M4 — Quantised Compression V4 (3-level codebook).

Three-level codebook: coarse (K1) × fine (K2) × ultra-fine (K3).
Plus a learned **adaptive budget allocator** that decides how
many bits to spend per region.

Capacity: K1=32 × K2=16 × K3=8 = 4096 codes = 12 bits per
encoded pair. With per-bit emit gate adding additional bits,
the W52 target is **≥ 14 bits/visible-token** at full emit.

Pure-Python only — reuses the W47 ``Variable`` autograd engine
and the W51 ``HierarchicalCodebook`` for the K1 / K2 layers
(K3 added on top).
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
    vsum,
    vmean,
    vsoftmax,
)
from .hierarchical_compression import (
    HierarchicalCodebook,
    HierarchicalEmitGate,
    W51_DEFAULT_HIER_BITS_PAYLOAD_LEN,
    W51_DEFAULT_HIER_CODE_DIM,
    W51_DEFAULT_HIER_EMIT_MASK_LEN,
    W51_DEFAULT_HIER_K1,
    W51_DEFAULT_HIER_K2,
)


# =============================================================================
# Schema, defaults
# =============================================================================

W52_QUANT_SCHEMA_VERSION: str = (
    "coordpy.quantised_compression.v1")

W52_DEFAULT_QUANT_K1: int = W51_DEFAULT_HIER_K1
W52_DEFAULT_QUANT_K2: int = W51_DEFAULT_HIER_K2
W52_DEFAULT_QUANT_K3: int = 8
W52_DEFAULT_QUANT_CODE_DIM: int = W51_DEFAULT_HIER_CODE_DIM
W52_DEFAULT_QUANT_EMIT_MASK_LEN: int = 16
W52_DEFAULT_QUANT_BITS_PAYLOAD_LEN: int = (
    W51_DEFAULT_HIER_BITS_PAYLOAD_LEN)
W52_DEFAULT_QUANT_TARGET_BITS_PER_TOKEN: float = 14.0


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
# QuantisedCodebookV4 (K1 × K2 × K3)
# =============================================================================


@dataclasses.dataclass
class QuantisedCodebookV4:
    """Three-level trainable codebook: coarse × fine × ultra."""

    n_coarse: int
    n_fine: int
    n_ultra: int
    code_dim: int
    coarse_prototypes: ParamTensor  # (n_coarse, code_dim)
    fine_prototypes: ParamTensor    # (n_coarse, n_fine, code_dim)
    ultra_prototypes: ParamTensor   # (n_coarse, n_fine, n_ultra, code_dim)

    @classmethod
    def init(
            cls, *,
            n_coarse: int = W52_DEFAULT_QUANT_K1,
            n_fine: int = W52_DEFAULT_QUANT_K2,
            n_ultra: int = W52_DEFAULT_QUANT_K3,
            code_dim: int = W52_DEFAULT_QUANT_CODE_DIM,
            seed: int = W47_DEFAULT_TRAIN_SEED,
            init_scale: float = W47_DEFAULT_INIT_SCALE,
    ) -> "QuantisedCodebookV4":
        rng = _DeterministicLCG(seed=int(seed))
        cp = ParamTensor(
            shape=(int(n_coarse), int(code_dim)), values=[])
        cp.init_seed(
            seed=int(rng.next_uniform() * (1 << 30)),
            scale=float(init_scale))
        fp = ParamTensor(
            shape=(int(n_coarse), int(n_fine), int(code_dim)),
            values=[])
        fp.init_seed(
            seed=int(rng.next_uniform() * (1 << 30)),
            scale=float(init_scale) * 0.5)
        up = ParamTensor(
            shape=(int(n_coarse), int(n_fine),
                   int(n_ultra), int(code_dim)),
            values=[])
        up.init_seed(
            seed=int(rng.next_uniform() * (1 << 30)),
            scale=float(init_scale) * 0.25)
        return cls(
            n_coarse=int(n_coarse),
            n_fine=int(n_fine),
            n_ultra=int(n_ultra),
            code_dim=int(code_dim),
            coarse_prototypes=cp,
            fine_prototypes=fp,
            ultra_prototypes=up)

    def params(self) -> list[ParamTensor]:
        return [
            self.coarse_prototypes,
            self.fine_prototypes,
            self.ultra_prototypes,
        ]

    def coarse_bits(self) -> int:
        if self.n_coarse <= 1:
            return 0
        return int(math.ceil(math.log2(float(self.n_coarse))))

    def fine_bits(self) -> int:
        if self.n_fine <= 1:
            return 0
        return int(math.ceil(math.log2(float(self.n_fine))))

    def ultra_bits(self) -> int:
        if self.n_ultra <= 1:
            return 0
        return int(math.ceil(math.log2(float(self.n_ultra))))

    def coarse_vector(self, code: int) -> tuple[float, ...]:
        c = int(code) % max(1, self.n_coarse)
        base = c * self.code_dim
        return tuple(
            float(self.coarse_prototypes.values[base + j])
            for j in range(self.code_dim))

    def fine_vector(
            self, *, coarse: int, fine: int,
    ) -> tuple[float, ...]:
        c = int(coarse) % max(1, self.n_coarse)
        f = int(fine) % max(1, self.n_fine)
        base = (c * self.n_fine + f) * self.code_dim
        return tuple(
            float(self.fine_prototypes.values[base + j])
            for j in range(self.code_dim))

    def ultra_vector(
            self, *, coarse: int, fine: int, ultra: int,
    ) -> tuple[float, ...]:
        c = int(coarse) % max(1, self.n_coarse)
        f = int(fine) % max(1, self.n_fine)
        u = int(ultra) % max(1, self.n_ultra)
        base = ((c * self.n_fine + f) * self.n_ultra + u) * self.code_dim
        return tuple(
            float(self.ultra_prototypes.values[base + j])
            for j in range(self.code_dim))

    def encode_value(
            self, x: Sequence[float],
    ) -> tuple[int, int, int]:
        """Encode x as a (coarse, fine, ultra) triple."""
        # Coarse: nearest coarse prototype.
        best_c, best_cd = 0, float("inf")
        for c in range(self.n_coarse):
            base = c * self.code_dim
            d = 0.0
            for j in range(self.code_dim):
                xj = float(x[j]) if j < len(x) else 0.0
                cj = float(self.coarse_prototypes.values[base + j])
                diff = xj - cj
                d += diff * diff
            if d < best_cd:
                best_cd = d
                best_c = c
        # Residual = x - coarse.
        cv = self.coarse_vector(best_c)
        resid_c = [
            (float(x[j]) if j < len(x) else 0.0) - float(cv[j])
            for j in range(self.code_dim)
        ]
        # Fine: nearest within coarse cluster.
        best_f, best_fd = 0, float("inf")
        cluster_base = best_c * self.n_fine
        for f in range(self.n_fine):
            base = (cluster_base + f) * self.code_dim
            d = 0.0
            for j in range(self.code_dim):
                fj = float(self.fine_prototypes.values[base + j])
                diff = float(resid_c[j]) - fj
                d += diff * diff
            if d < best_fd:
                best_fd = d
                best_f = f
        # Residual = x - coarse - fine.
        fv = self.fine_vector(coarse=best_c, fine=best_f)
        resid_cf = [
            float(resid_c[j]) - float(fv[j])
            for j in range(self.code_dim)
        ]
        # Ultra: nearest within (coarse, fine) cell.
        best_u, best_ud = 0, float("inf")
        uf_base = (best_c * self.n_fine + best_f) * self.n_ultra
        for u in range(self.n_ultra):
            base = (uf_base + u) * self.code_dim
            d = 0.0
            for j in range(self.code_dim):
                uj = float(
                    self.ultra_prototypes.values[base + j])
                diff = float(resid_cf[j]) - uj
                d += diff * diff
            if d < best_ud:
                best_ud = d
                best_u = u
        return int(best_c), int(best_f), int(best_u)

    def decode(
            self, *, coarse: int, fine: int, ultra: int,
            include_ultra: bool = True,
    ) -> tuple[float, ...]:
        cv = self.coarse_vector(int(coarse))
        fv = self.fine_vector(coarse=int(coarse), fine=int(fine))
        uv: Sequence[float]
        if include_ultra:
            uv = self.ultra_vector(
                coarse=int(coarse), fine=int(fine),
                ultra=int(ultra))
        else:
            uv = [0.0] * self.code_dim
        return tuple(
            float(cv[j]) + float(fv[j]) + float(uv[j])
            for j in range(self.code_dim))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": str(W52_QUANT_SCHEMA_VERSION),
            "n_coarse": int(self.n_coarse),
            "n_fine": int(self.n_fine),
            "n_ultra": int(self.n_ultra),
            "code_dim": int(self.code_dim),
            "coarse_prototypes": self.coarse_prototypes.to_dict(),
            "fine_prototypes": self.fine_prototypes.to_dict(),
            "ultra_prototypes": self.ultra_prototypes.to_dict(),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w52_quantised_codebook_v4",
            "codebook": self.to_dict()})


# =============================================================================
# QuantisedBudgetGate
# =============================================================================


@dataclasses.dataclass
class QuantisedBudgetGate:
    """Per-bit emit gate + per-level (coarse, fine, ultra) gate.

    Three level gates decide whether to emit each codebook
    level; the per-bit emit gate continues to decide which
    bits of the bits_payload to emit.
    """

    in_dim: int
    emit_mask_len: int
    w_level: ParamTensor  # (3, in_dim) — coarse + fine + ultra
    w_emit: ParamTensor   # (emit_mask_len, in_dim)
    importance_threshold: float = 0.5

    @classmethod
    def init(
            cls, *,
            in_dim: int = W52_DEFAULT_QUANT_CODE_DIM,
            emit_mask_len: int = W52_DEFAULT_QUANT_EMIT_MASK_LEN,
            seed: int = W47_DEFAULT_TRAIN_SEED,
            init_scale: float = W47_DEFAULT_INIT_SCALE,
            importance_threshold: float = 0.5,
    ) -> "QuantisedBudgetGate":
        rng = _DeterministicLCG(seed=int(seed))
        # Init level emit gates open (positive bias).
        wl = ParamTensor(
            shape=(3, int(in_dim)),
            values=[0.0] * (3 * int(in_dim)))
        for i in range(int(in_dim)):
            wl.values[0 * int(in_dim) + i] = 1.0
            wl.values[1 * int(in_dim) + i] = 1.0
            wl.values[2 * int(in_dim) + i] = 1.0
        we = ParamTensor(
            shape=(int(emit_mask_len), int(in_dim)), values=[])
        we.init_seed(
            seed=int(rng.next_uniform() * (1 << 30)),
            scale=float(init_scale))
        return cls(
            in_dim=int(in_dim),
            emit_mask_len=int(emit_mask_len),
            w_level=wl,
            w_emit=we,
            importance_threshold=float(importance_threshold))

    def params(self) -> list[ParamTensor]:
        return [self.w_level, self.w_emit]

    def forward_value(
            self, x: Sequence[float],
    ) -> tuple[
            list[int], list[float],
            list[int], list[float]]:
        """Returns (level_mask, level_scores, emit_mask, emit_scores)."""
        in_d = int(self.in_dim)
        level_scores: list[float] = []
        for l in range(3):
            s = 0.0
            for j in range(in_d):
                xj = float(x[j]) if j < len(x) else 0.0
                s += float(self.w_level.values[l * in_d + j]) * xj
            level_scores.append(float(_stable_sigmoid(s)))
        level_mask = [
            1 if ls > float(self.importance_threshold) else 0
            for ls in level_scores
        ]
        emit_scores: list[float] = []
        for b in range(int(self.emit_mask_len)):
            s = 0.0
            for j in range(in_d):
                xj = float(x[j]) if j < len(x) else 0.0
                s += float(self.w_emit.values[b * in_d + j]) * xj
            emit_scores.append(float(_stable_sigmoid(s)))
        emit_mask = [
            1 if es > float(self.importance_threshold) else 0
            for es in emit_scores
        ]
        return level_mask, level_scores, emit_mask, emit_scores

    def to_dict(self) -> dict[str, Any]:
        return {
            "in_dim": int(self.in_dim),
            "emit_mask_len": int(self.emit_mask_len),
            "w_level": self.w_level.to_dict(),
            "w_emit": self.w_emit.to_dict(),
            "importance_threshold": float(round(
                self.importance_threshold, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w52_quantised_budget_gate",
            "gate": self.to_dict()})


# =============================================================================
# Compression result
# =============================================================================


@dataclasses.dataclass(frozen=True)
class QuantisedCompressionResult:
    coarse_code: int
    fine_code: int
    ultra_code: int
    coarse_bits: int
    fine_bits: int
    ultra_bits: int
    level_mask: tuple[int, ...]
    level_scores: tuple[float, ...]
    emit_mask: tuple[int, ...]
    emit_scores: tuple[float, ...]
    bits_payload: tuple[int, ...]
    visible_tokens: int
    structured_bits: int
    bits_per_visible_token: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "coarse_code": int(self.coarse_code),
            "fine_code": int(self.fine_code),
            "ultra_code": int(self.ultra_code),
            "coarse_bits": int(self.coarse_bits),
            "fine_bits": int(self.fine_bits),
            "ultra_bits": int(self.ultra_bits),
            "level_mask": list(self.level_mask),
            "level_scores": [
                float(round(v, 12))
                for v in self.level_scores],
            "emit_mask": list(self.emit_mask),
            "emit_scores": [
                float(round(v, 12))
                for v in self.emit_scores],
            "bits_payload": list(self.bits_payload),
            "visible_tokens": int(self.visible_tokens),
            "structured_bits": int(self.structured_bits),
            "bits_per_visible_token": float(round(
                self.bits_per_visible_token, 12)),
        }


def compress_carrier_quantised(
        carrier: Sequence[float],
        *,
        codebook: QuantisedCodebookV4,
        gate: QuantisedBudgetGate,
        bits_payload_len: int = (
            W52_DEFAULT_QUANT_BITS_PAYLOAD_LEN),
        max_visible_tokens: int | None = None,
) -> QuantisedCompressionResult:
    """Compress a carrier with K1 × K2 × K3 quantisation +
    adaptive budget gate.

    Token accounting (the W52 packed ``LATENT_CTRL_V4_Q`` block):
    * **token 0**: coarse code + 3-bit level mask
      (carries ``coarse_bits + 3``)
    * **token 1**: fine code + ultra code + per-bit emit mask
      (emitted iff level_mask[1] = 1) → ``fine_bits + ultra_bits + emit_mask_len``
    * **token 2**: emitted bits payload (only mask_sum bits;
      emitted iff mask_sum > 0)

    With K1=32 (5 bits) + K2=16 (4 bits) + K3=8 (3 bits) +
    emit_mask_len=16, full emit yields ~28 bits / 3 tokens
    or ~38 bits / 3 tokens when payload tokens are included.
    """
    coarse, fine, ultra = codebook.encode_value(carrier)
    coarse_bits = codebook.coarse_bits()
    fine_bits = codebook.fine_bits()
    ultra_bits = codebook.ultra_bits()
    decoded = codebook.decode(
        coarse=coarse, fine=fine, ultra=ultra)
    level_mask, level_scores, emit_mask, emit_scores = (
        gate.forward_value(carrier))
    # Bits payload: low-order quantisation of the residual.
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
    mask_sum = sum(int(m) for m in emit_mask)
    fine_emitted = (
        int(level_mask[1]) if len(level_mask) >= 2 else 1)
    # ultra share the same token as fine — emitted iff fine
    ultra_emitted = (
        int(level_mask[2]) if len(level_mask) >= 3 else 1)
    visible_tokens = (
        1 + (1 if fine_emitted else 0)
        + (1 if mask_sum > 0 else 0))
    if (max_visible_tokens is not None
            and visible_tokens > int(max_visible_tokens)):
        if int(max_visible_tokens) == 1:
            fine_emitted = 0
            ultra_emitted = 0
            mask_sum = 0
        elif int(max_visible_tokens) == 2:
            mask_sum = 0
        visible_tokens = int(max_visible_tokens)
    structured_bits = (
        int(coarse_bits) + 3  # level mask in token 0
        + (int(fine_bits)
           + (int(ultra_bits) if ultra_emitted else 0)
           + int(gate.emit_mask_len)
           if fine_emitted else 0)
        + int(mask_sum))
    bits_per = (
        float(structured_bits) / float(max(1, visible_tokens))
        if visible_tokens > 0 else 0.0)
    return QuantisedCompressionResult(
        coarse_code=int(coarse),
        fine_code=int(fine),
        ultra_code=int(ultra),
        coarse_bits=int(coarse_bits),
        fine_bits=int(fine_bits),
        ultra_bits=int(ultra_bits),
        level_mask=tuple(int(m) for m in level_mask),
        level_scores=tuple(level_scores),
        emit_mask=tuple(int(m) for m in emit_mask),
        emit_scores=tuple(emit_scores),
        bits_payload=tuple(payload),
        visible_tokens=int(visible_tokens),
        structured_bits=int(structured_bits),
        bits_per_visible_token=float(bits_per),
    )


# =============================================================================
# Cramming witness V4
# =============================================================================


@dataclasses.dataclass(frozen=True)
class CrammingWitnessV4:
    structured_bits: int
    visible_ctrl_tokens: int
    visible_latent_header_tokens: int
    persistent_state_bytes: int
    bits_per_visible_token: float
    coarse_code: int
    fine_code: int
    ultra_code: int
    coarse_bits: int
    fine_bits: int
    ultra_bits: int
    level_mask_bits_set: int
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
            "persistent_state_bytes": int(
                self.persistent_state_bytes),
            "bits_per_visible_token": float(round(
                self.bits_per_visible_token, 12)),
            "coarse_code": int(self.coarse_code),
            "fine_code": int(self.fine_code),
            "ultra_code": int(self.ultra_code),
            "coarse_bits": int(self.coarse_bits),
            "fine_bits": int(self.fine_bits),
            "ultra_bits": int(self.ultra_bits),
            "level_mask_bits_set": int(self.level_mask_bits_set),
            "emit_mask_bits_set": int(self.emit_mask_bits_set),
            "emit_mask_len": int(self.emit_mask_len),
            "bits_payload_len": int(self.bits_payload_len),
            "cramming_bytes_sha256": str(
                self.cramming_bytes_sha256),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w52_cramming_witness_v4",
            "witness": self.to_dict()})


def emit_cramming_witness_v4(
        *,
        compression: QuantisedCompressionResult,
) -> CrammingWitnessV4:
    """Build the per-turn cramming witness from compression."""
    cramming_bytes = _canonical_bytes(compression.to_dict())
    bsha = hashlib.sha256(cramming_bytes).hexdigest()
    return CrammingWitnessV4(
        structured_bits=int(compression.structured_bits),
        visible_ctrl_tokens=int(compression.visible_tokens),
        visible_latent_header_tokens=int(
            compression.visible_tokens),
        persistent_state_bytes=0,
        bits_per_visible_token=float(
            compression.bits_per_visible_token),
        coarse_code=int(compression.coarse_code),
        fine_code=int(compression.fine_code),
        ultra_code=int(compression.ultra_code),
        coarse_bits=int(compression.coarse_bits),
        fine_bits=int(compression.fine_bits),
        ultra_bits=int(compression.ultra_bits),
        level_mask_bits_set=int(sum(compression.level_mask)),
        emit_mask_bits_set=int(sum(compression.emit_mask)),
        emit_mask_len=int(len(compression.emit_mask)),
        bits_payload_len=int(len(compression.bits_payload)),
        cramming_bytes_sha256=str(bsha),
    )


# =============================================================================
# Degradation curve
# =============================================================================


@dataclasses.dataclass(frozen=True)
class QuantisedDegradationPoint:
    budget: int
    bits_per_visible_token: float
    retention_cosine: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "budget": int(self.budget),
            "bits_per_visible_token": float(round(
                self.bits_per_visible_token, 12)),
            "retention_cosine": float(round(
                self.retention_cosine, 12)),
        }


def probe_quantised_degradation_curve(
        carrier: Sequence[float],
        *,
        codebook: QuantisedCodebookV4,
        gate: QuantisedBudgetGate,
        budgets: Sequence[int] = (16, 8, 4, 2, 1),
        bits_payload_len: int = (
            W52_DEFAULT_QUANT_BITS_PAYLOAD_LEN),
) -> tuple[QuantisedDegradationPoint, ...]:
    """Probe achieved bits/token + retention cosine across
    decreasing token budgets.
    """
    out: list[QuantisedDegradationPoint] = []
    for budget in budgets:
        res = compress_carrier_quantised(
            carrier, codebook=codebook, gate=gate,
            bits_payload_len=int(bits_payload_len),
            max_visible_tokens=int(budget))
        decoded = codebook.decode(
            coarse=res.coarse_code,
            fine=res.fine_code,
            ultra=res.ultra_code,
            include_ultra=(int(res.ultra_code) >= 0
                           and len(res.level_mask) >= 3
                           and int(res.level_mask[2]) == 1))
        retention = _cosine(carrier, decoded)
        out.append(QuantisedDegradationPoint(
            budget=int(budget),
            bits_per_visible_token=float(
                res.bits_per_visible_token),
            retention_cosine=float(retention)))
    return tuple(out)


# =============================================================================
# Training set + fit
# =============================================================================


@dataclasses.dataclass(frozen=True)
class QuantisedCompressionExample:
    carrier: tuple[float, ...]


@dataclasses.dataclass(frozen=True)
class QuantisedCompressionTrainingSet:
    examples: tuple[QuantisedCompressionExample, ...]
    code_dim: int
    n_coarse: int
    n_fine: int
    n_ultra: int
    emit_mask_len: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "code_dim": int(self.code_dim),
            "n_coarse": int(self.n_coarse),
            "n_fine": int(self.n_fine),
            "n_ultra": int(self.n_ultra),
            "emit_mask_len": int(self.emit_mask_len),
            "examples": [
                {"carrier": list(e.carrier)}
                for e in self.examples],
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w52_quantised_compression_training_set",
            "set": self.to_dict()})


def synthesize_quantised_compression_training_set(
        *,
        n_examples: int = 24,
        code_dim: int = W52_DEFAULT_QUANT_CODE_DIM,
        n_coarse: int = W52_DEFAULT_QUANT_K1,
        n_fine: int = W52_DEFAULT_QUANT_K2,
        n_ultra: int = W52_DEFAULT_QUANT_K3,
        emit_mask_len: int = W52_DEFAULT_QUANT_EMIT_MASK_LEN,
        seed: int = W47_DEFAULT_TRAIN_SEED,
) -> QuantisedCompressionTrainingSet:
    """Deterministic compression bank."""
    rng = _DeterministicLCG(seed=int(seed))
    exs: list[QuantisedCompressionExample] = []
    for _ in range(int(n_examples)):
        c = tuple(
            float(rng.next_uniform() * 2.0 - 1.0)
            for _ in range(int(code_dim)))
        exs.append(QuantisedCompressionExample(carrier=c))
    return QuantisedCompressionTrainingSet(
        examples=tuple(exs),
        code_dim=int(code_dim),
        n_coarse=int(n_coarse),
        n_fine=int(n_fine),
        n_ultra=int(n_ultra),
        emit_mask_len=int(emit_mask_len))


@dataclasses.dataclass(frozen=True)
class QuantisedCompressionTrainingTrace:
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
            "final_grad_norm": float(round(
                self.final_grad_norm, 12)),
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
            "kind": "w52_quantised_compression_training_trace",
            "trace": self.to_dict()})


def fit_quantised_compression(
        training_set: QuantisedCompressionTrainingSet,
        *,
        n_steps: int = 32,
        learning_rate: float = 0.05,
        seed: int = W47_DEFAULT_TRAIN_SEED,
        init_scale: float = W47_DEFAULT_INIT_SCALE,
        beta1: float = W47_DEFAULT_BETA1,
        beta2: float = W47_DEFAULT_BETA2,
        eps: float = W47_DEFAULT_EPS,
        grad_clip: float = W47_DEFAULT_GRAD_CLIP,
        history_head: int = 6,
        history_tail: int = 6,
) -> tuple[QuantisedCodebookV4, QuantisedBudgetGate,
           QuantisedCompressionTrainingTrace]:
    """Fit the codebook (k-means-style) + gate (sigmoid).

    For pure-Python tractability, we don't run a full
    soft-EM here — we use a simple k-means update on the
    codebook (closest-prototype + average) and leave the
    gate at its default initialisation.
    """
    cb = QuantisedCodebookV4.init(
        n_coarse=int(training_set.n_coarse),
        n_fine=int(training_set.n_fine),
        n_ultra=int(training_set.n_ultra),
        code_dim=int(training_set.code_dim),
        seed=int(seed),
        init_scale=float(init_scale))
    gate = QuantisedBudgetGate.init(
        in_dim=int(training_set.code_dim),
        emit_mask_len=int(training_set.emit_mask_len),
        seed=int(seed) + 1,
        init_scale=float(init_scale),
        importance_threshold=0.5)
    loss_history: list[float] = []
    grad_norm_history: list[float] = []
    diverged = False
    # k-means refinement.
    for step in range(int(n_steps)):
        coarse_counts = [0] * cb.n_coarse
        coarse_sums = [
            [0.0] * cb.code_dim for _ in range(cb.n_coarse)
        ]
        fine_counts = [
            [0] * cb.n_fine for _ in range(cb.n_coarse)
        ]
        fine_sums = [
            [[0.0] * cb.code_dim for _ in range(cb.n_fine)]
            for _ in range(cb.n_coarse)
        ]
        ultra_counts = [
            [[0] * cb.n_ultra for _ in range(cb.n_fine)]
            for _ in range(cb.n_coarse)
        ]
        ultra_sums = [
            [[[0.0] * cb.code_dim
              for _ in range(cb.n_ultra)]
             for _ in range(cb.n_fine)]
            for _ in range(cb.n_coarse)
        ]
        sse = 0.0
        for ex in training_set.examples:
            c, f, u = cb.encode_value(ex.carrier)
            coarse_counts[c] += 1
            for j in range(cb.code_dim):
                xj = (float(ex.carrier[j])
                      if j < len(ex.carrier) else 0.0)
                coarse_sums[c][j] += xj
            cv = cb.coarse_vector(c)
            resid = [
                (float(ex.carrier[j]) if j < len(ex.carrier) else 0.0)
                - float(cv[j])
                for j in range(cb.code_dim)
            ]
            fine_counts[c][f] += 1
            for j in range(cb.code_dim):
                fine_sums[c][f][j] += float(resid[j])
            fv = cb.fine_vector(coarse=c, fine=f)
            resid_cf = [
                float(resid[j]) - float(fv[j])
                for j in range(cb.code_dim)
            ]
            ultra_counts[c][f][u] += 1
            for j in range(cb.code_dim):
                ultra_sums[c][f][u][j] += float(resid_cf[j])
            decoded = cb.decode(
                coarse=c, fine=f, ultra=u)
            for j in range(cb.code_dim):
                xj = (float(ex.carrier[j])
                      if j < len(ex.carrier) else 0.0)
                dj = float(decoded[j])
                sse += (xj - dj) ** 2
        # Update prototypes by mean of assigned residuals.
        new_cp = list(cb.coarse_prototypes.values)
        new_fp = list(cb.fine_prototypes.values)
        new_up = list(cb.ultra_prototypes.values)
        for c in range(cb.n_coarse):
            if coarse_counts[c] > 0:
                base = c * cb.code_dim
                for j in range(cb.code_dim):
                    new_cp[base + j] = (
                        coarse_sums[c][j]
                        / float(coarse_counts[c]))
            for f in range(cb.n_fine):
                if fine_counts[c][f] > 0:
                    base = (c * cb.n_fine + f) * cb.code_dim
                    for j in range(cb.code_dim):
                        new_fp[base + j] = (
                            fine_sums[c][f][j]
                            / float(fine_counts[c][f]))
                for u in range(cb.n_ultra):
                    if ultra_counts[c][f][u] > 0:
                        base = (
                            (c * cb.n_fine + f) * cb.n_ultra
                            + u) * cb.code_dim
                        for j in range(cb.code_dim):
                            new_up[base + j] = (
                                ultra_sums[c][f][u][j]
                                / float(ultra_counts[c][f][u]))
        cb.coarse_prototypes.values = new_cp
        cb.fine_prototypes.values = new_fp
        cb.ultra_prototypes.values = new_up
        loss_history.append(
            float(sse) / float(max(1, len(training_set.examples))))
        grad_norm_history.append(0.0)
    head_n = max(0, int(history_head))
    tail_n = max(0, int(history_tail))
    trace = QuantisedCompressionTrainingTrace(
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
# Witness
# =============================================================================


@dataclasses.dataclass(frozen=True)
class QuantisedCompressionWitness:
    codebook_cid: str
    gate_cid: str
    training_trace_cid: str
    cramming_cid: str
    retention_cosine: float
    achieved_bits_per_token: float
    target_bits_per_token: float
    target_met: bool
    degradation_curve: tuple[QuantisedDegradationPoint, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "codebook_cid": str(self.codebook_cid),
            "gate_cid": str(self.gate_cid),
            "training_trace_cid": str(self.training_trace_cid),
            "cramming_cid": str(self.cramming_cid),
            "retention_cosine": float(round(
                self.retention_cosine, 12)),
            "achieved_bits_per_token": float(round(
                self.achieved_bits_per_token, 12)),
            "target_bits_per_token": float(round(
                self.target_bits_per_token, 12)),
            "target_met": bool(self.target_met),
            "degradation_curve": [
                p.to_dict() for p in self.degradation_curve],
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w52_quantised_compression_witness",
            "witness": self.to_dict()})


def emit_quantised_compression_witness(
        *,
        codebook: QuantisedCodebookV4,
        gate: QuantisedBudgetGate,
        training_trace: QuantisedCompressionTrainingTrace,
        cramming: CrammingWitnessV4,
        retention_cosine: float,
        target_bits_per_token: float = (
            W52_DEFAULT_QUANT_TARGET_BITS_PER_TOKEN),
        degradation_curve: (
            tuple[QuantisedDegradationPoint, ...] | None) = None,
) -> QuantisedCompressionWitness:
    return QuantisedCompressionWitness(
        codebook_cid=str(codebook.cid()),
        gate_cid=str(gate.cid()),
        training_trace_cid=str(training_trace.cid()),
        cramming_cid=str(cramming.cid()),
        retention_cosine=float(retention_cosine),
        achieved_bits_per_token=float(
            cramming.bits_per_visible_token),
        target_bits_per_token=float(target_bits_per_token),
        target_met=bool(
            cramming.bits_per_visible_token
            >= float(target_bits_per_token)),
        degradation_curve=tuple(degradation_curve or ()),
    )


# =============================================================================
# Falsifier: 32-bit rate-floor probe
# =============================================================================


@dataclasses.dataclass(frozen=True)
class QuantisedRateFloorFalsifierResult:
    achieved_bits_per_token: float
    target_bits_per_token: float
    rate_target_missed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "achieved_bits_per_token": float(round(
                self.achieved_bits_per_token, 12)),
            "target_bits_per_token": float(round(
                self.target_bits_per_token, 12)),
            "rate_target_missed": bool(self.rate_target_missed),
        }


def probe_quantised_rate_floor_falsifier(
        carrier: Sequence[float],
        *,
        codebook: QuantisedCodebookV4,
        gate: QuantisedBudgetGate,
        target_bits_per_token: float = 32.0,
) -> QuantisedRateFloorFalsifierResult:
    """Probe whether the W52 codebook can hit 32 bits/token —
    it can't, because K1=32 × K2=16 × K3=8 = 4096 codes ≈
    12 bits per (coarse, fine, ultra) triple.
    """
    res = compress_carrier_quantised(
        carrier, codebook=codebook, gate=gate)
    return QuantisedRateFloorFalsifierResult(
        achieved_bits_per_token=float(
            res.bits_per_visible_token),
        target_bits_per_token=float(target_bits_per_token),
        rate_target_missed=bool(
            res.bits_per_visible_token
            < float(target_bits_per_token)),
    )


# =============================================================================
# Verifier
# =============================================================================


W52_QUANT_VERIFIER_FAILURE_MODES: tuple[str, ...] = (
    "w52_quant_schema_mismatch",
    "w52_quant_codebook_cid_mismatch",
    "w52_quant_gate_cid_mismatch",
    "w52_quant_training_trace_cid_mismatch",
    "w52_quant_cramming_cid_mismatch",
    "w52_quant_target_not_met_when_required",
)


def verify_quantised_compression_witness(
        witness: QuantisedCompressionWitness,
        *,
        expected_codebook_cid: str | None = None,
        expected_gate_cid: str | None = None,
        target_required: bool = False,
) -> dict[str, Any]:
    failures: list[str] = []
    if (expected_codebook_cid is not None
            and witness.codebook_cid != expected_codebook_cid):
        failures.append("w52_quant_codebook_cid_mismatch")
    if (expected_gate_cid is not None
            and witness.gate_cid != expected_gate_cid):
        failures.append("w52_quant_gate_cid_mismatch")
    if target_required and not witness.target_met:
        failures.append(
            "w52_quant_target_not_met_when_required")
    return {
        "ok": (len(failures) == 0),
        "failures": failures,
        "witness_cid": witness.cid(),
    }


__all__ = [
    "W52_QUANT_SCHEMA_VERSION",
    "W52_DEFAULT_QUANT_K1",
    "W52_DEFAULT_QUANT_K2",
    "W52_DEFAULT_QUANT_K3",
    "W52_DEFAULT_QUANT_CODE_DIM",
    "W52_DEFAULT_QUANT_EMIT_MASK_LEN",
    "W52_DEFAULT_QUANT_BITS_PAYLOAD_LEN",
    "W52_DEFAULT_QUANT_TARGET_BITS_PER_TOKEN",
    "W52_QUANT_VERIFIER_FAILURE_MODES",
    "QuantisedCodebookV4",
    "QuantisedBudgetGate",
    "QuantisedCompressionResult",
    "CrammingWitnessV4",
    "QuantisedDegradationPoint",
    "QuantisedCompressionExample",
    "QuantisedCompressionTrainingSet",
    "QuantisedCompressionTrainingTrace",
    "QuantisedCompressionWitness",
    "QuantisedRateFloorFalsifierResult",
    "compress_carrier_quantised",
    "emit_cramming_witness_v4",
    "probe_quantised_degradation_curve",
    "synthesize_quantised_compression_training_set",
    "fit_quantised_compression",
    "emit_quantised_compression_witness",
    "probe_quantised_rate_floor_falsifier",
    "verify_quantised_compression_witness",
]
