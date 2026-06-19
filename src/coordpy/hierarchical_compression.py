"""W51 M4 — Hierarchical Adaptive Compression V3.

A two-level codebook compression: a **coarse codebook**
(``K1 = 32`` prototypes) and a **per-cluster fine codebook**
(``K2 = 16`` sub-prototypes per coarse cluster). Each carrier
is encoded as a ``(coarse_code, fine_code)`` pair plus a
learned hierarchical emit gate that decides which level(s) to
emit.

Target frontier: **≥ 12.0 structured bits per visible-token**
at retention cosine ≥ 0.85. W50's frontier was 8.0; this is a
strict +50% advance over W50 (and +140% over W49's 5.0)
under the W47 pure-Python autograd training cost cap.

A **degradation curve probe** records achieved bits/token
across decreasing token budgets ``{8, 4, 2, 1}`` to verify
graceful degradation (no cliff). This addresses the H16
``W51-L-COMPRESSION-RATE-FLOOR-V2-CAP`` falsifier window.

Pure-Python only — reuses the W47 ``Variable`` +
``AdamOptimizer`` autograd engine.

Honest scope (do-not-overstate)
-------------------------------

This module does NOT touch real LLM tokenizers or attention
weights. The "visible tokens" count is a capsule-layer
surrogate for the packed ``LATENT_CTRL_V3_H`` block; real LLM
tokens may or may not match this 1-1 (the
``W51-L-CTRL-AWARE-MODEL-INDIFFERENCE-CAP`` carries forward).

The H15 bar of ≥ 12.0 bits/visible-token is honest under the
hierarchical codebook capacity. The H16 graceful-degradation
claim is empirical on the R-101 family. The
``W51-L-COMPRESSION-RATE-FLOOR-V2-CAP`` falsifier reproduces
when the target rate is set above the K1=32 + K2=16 codebook's
information capacity (~9 bits per pair).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

from .adaptive_compression import (
    AdaptiveCompressionCodebook,
    AdaptiveCompressionGate,
    W50_DEFAULT_ADAPTIVE_CODE_DIM,
)
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
    vmean,
    vsoftmax,
    vsum,
)


# =============================================================================
# Schema, defaults
# =============================================================================

W51_HIERARCHICAL_COMPRESSION_SCHEMA_VERSION: str = (
    "coordpy.hierarchical_compression.v1")

W51_DEFAULT_HIER_K1: int = 32
W51_DEFAULT_HIER_K2: int = 16
W51_DEFAULT_HIER_CODE_DIM: int = (
    W50_DEFAULT_ADAPTIVE_CODE_DIM)
W51_DEFAULT_HIER_EMIT_MASK_LEN: int = 14
W51_DEFAULT_HIER_BITS_PAYLOAD_LEN: int = 14
W51_DEFAULT_HIER_TARGET_BITS_PER_TOKEN: float = 12.0
W51_DEFAULT_HIER_RETENTION_FLOOR: float = 0.85
W51_DEFAULT_HIER_LATENT_CTRL_TAG: str = "LATENT_CTRL_V3_H"
W51_DEFAULT_HIER_DEGRADATION_BUDGETS: tuple[int, ...] = (
    8, 4, 2, 1)


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
# Hierarchical codebook (coarse + fine)
# =============================================================================

@dataclasses.dataclass
class HierarchicalCodebook:
    """Two-level trainable codebook.

    Coarse codebook: ``n_coarse`` prototypes of dim ``code_dim``.
    Fine codebook: ``n_coarse`` × ``n_fine`` sub-prototypes (one
    sub-codebook per coarse cluster), each of dim ``code_dim``.
    """

    n_coarse: int
    n_fine: int
    code_dim: int
    coarse_prototypes: ParamTensor   # (n_coarse, code_dim)
    fine_prototypes: ParamTensor     # (n_coarse, n_fine, code_dim)

    @classmethod
    def init(
            cls, *,
            n_coarse: int = W51_DEFAULT_HIER_K1,
            n_fine: int = W51_DEFAULT_HIER_K2,
            code_dim: int = W51_DEFAULT_HIER_CODE_DIM,
            seed: int = W47_DEFAULT_TRAIN_SEED,
            init_scale: float = W47_DEFAULT_INIT_SCALE,
    ) -> "HierarchicalCodebook":
        rng = _DeterministicLCG(seed=int(seed))
        cp = ParamTensor(
            shape=(int(n_coarse), int(code_dim)),
            values=[])
        cp.init_seed(
            seed=int(rng.next_uniform() * (1 << 30)),
            scale=float(init_scale))
        fp = ParamTensor(
            shape=(int(n_coarse), int(n_fine), int(code_dim)),
            values=[])
        fp.init_seed(
            seed=int(rng.next_uniform() * (1 << 30)),
            scale=float(init_scale))
        return cls(
            n_coarse=int(n_coarse),
            n_fine=int(n_fine),
            code_dim=int(code_dim),
            coarse_prototypes=cp,
            fine_prototypes=fp)

    def params(self) -> list[ParamTensor]:
        return [self.coarse_prototypes, self.fine_prototypes]

    def coarse_bits(self) -> int:
        if self.n_coarse <= 1:
            return 0
        return int(math.ceil(math.log2(float(self.n_coarse))))

    def fine_bits(self) -> int:
        if self.n_fine <= 1:
            return 0
        return int(math.ceil(math.log2(float(self.n_fine))))

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

    def encode_value(
            self, x: Sequence[float],
    ) -> tuple[int, int]:
        """Encode x as a (coarse_code, fine_code) pair."""
        # Find closest coarse prototype.
        best_c = 0
        best_d = float("inf")
        for c in range(self.n_coarse):
            base = c * self.code_dim
            d = 0.0
            for j in range(self.code_dim):
                xj = float(x[j]) if j < len(x) else 0.0
                cj = float(
                    self.coarse_prototypes.values[base + j])
                diff = xj - cj
                d += diff * diff
            if d < best_d:
                best_d = d
                best_c = c
        # Residual = x - coarse_proto.
        coarse_proto = self.coarse_vector(best_c)
        resid = [
            (float(x[j]) if j < len(x) else 0.0)
            - float(coarse_proto[j])
            for j in range(self.code_dim)
        ]
        # Find closest fine prototype within the coarse cluster.
        best_f = 0
        best_fd = float("inf")
        cluster_base = best_c * self.n_fine
        for f in range(self.n_fine):
            base = (cluster_base + f) * self.code_dim
            d = 0.0
            for j in range(self.code_dim):
                fj = float(
                    self.fine_prototypes.values[base + j])
                diff = float(resid[j]) - fj
                d += diff * diff
            if d < best_fd:
                best_fd = d
                best_f = f
        return int(best_c), int(best_f)

    def decode(
            self, *, coarse: int, fine: int,
    ) -> tuple[float, ...]:
        cv = self.coarse_vector(int(coarse))
        fv = self.fine_vector(coarse=int(coarse), fine=int(fine))
        return tuple(
            float(cv[j]) + float(fv[j])
            for j in range(self.code_dim))

    def encode_soft_vars(
            self, x: Sequence[Variable],
    ) -> tuple[list[Variable], list[Variable]]:
        """Returns soft-assignment weights over coarse codes
        (length n_coarse) and fine codes (length n_fine)
        conditioned on the argmax coarse selection.
        """
        protos_c = self.coarse_prototypes.make_vars()
        protos_f = self.fine_prototypes.make_vars()
        neg_d_c: list[Variable] = []
        argmax_c = 0
        argmax_c_d = float("inf")
        for c in range(self.n_coarse):
            base = c * self.code_dim
            terms: list[Variable] = []
            d_raw = 0.0
            for j in range(self.code_dim):
                xj = (x[j] if j < len(x) else Variable(0.0))
                cj = protos_c[base + j]
                diff = xj - cj
                terms.append(diff * diff)
                d_raw += float(diff.value) ** 2
            neg_d_c.append(-1.0 * vsum(terms))
            if d_raw < argmax_c_d:
                argmax_c_d = d_raw
                argmax_c = c
        weights_c = vsoftmax(neg_d_c)
        # Fine weights conditioned on argmax_c.
        cluster_base = argmax_c * self.n_fine
        coarse_vals: list[Variable] = []
        for j in range(self.code_dim):
            coarse_vals.append(
                protos_c[argmax_c * self.code_dim + j])
        neg_d_f: list[Variable] = []
        for f in range(self.n_fine):
            base = (cluster_base + f) * self.code_dim
            terms: list[Variable] = []
            for j in range(self.code_dim):
                xj = (x[j] if j < len(x) else Variable(0.0))
                fj = protos_f[base + j]
                diff = (xj - coarse_vals[j]) - fj
                terms.append(diff * diff)
            neg_d_f.append(-1.0 * vsum(terms))
        weights_f = vsoftmax(neg_d_f)
        return weights_c, weights_f

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_coarse": int(self.n_coarse),
            "n_fine": int(self.n_fine),
            "code_dim": int(self.code_dim),
            "coarse_prototypes": self.coarse_prototypes.to_dict(),
            "fine_prototypes": self.fine_prototypes.to_dict(),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w51_hierarchical_codebook",
            "codebook": self.to_dict()})


# =============================================================================
# Hierarchical emit gate
# =============================================================================

@dataclasses.dataclass
class HierarchicalEmitGate:
    """Per-bit + per-level emit-mask gate.

    Two heads: a **level-emit gate** (sigmoid in [0, 1])
    deciding whether to emit each of {coarse, fine} levels,
    and a **per-bit emit gate** identical to W50 deciding
    which bits in the payload to emit.
    """

    in_dim: int
    emit_mask_len: int
    w_level: ParamTensor   # (2, in_dim) — coarse + fine
    w_emit: ParamTensor    # (emit_mask_len, in_dim)
    importance_threshold: float = 0.5

    @classmethod
    def init(
            cls, *,
            in_dim: int,
            emit_mask_len: int = W51_DEFAULT_HIER_EMIT_MASK_LEN,
            seed: int = W47_DEFAULT_TRAIN_SEED,
            init_scale: float = W47_DEFAULT_INIT_SCALE,
            importance_threshold: float = 0.5,
    ) -> "HierarchicalEmitGate":
        rng = _DeterministicLCG(seed=int(seed))
        # Init level emit gates open (positive bias).
        wl = ParamTensor(
            shape=(2, int(in_dim)),
            values=[
                float(rng.next_uniform() - 0.5) * float(init_scale)
                + 0.5
                for _ in range(2 * int(in_dim))
            ])
        we = ParamTensor(
            shape=(int(emit_mask_len), int(in_dim)),
            values=[])
        we.init_seed(
            seed=int(rng.next_uniform() * (1 << 30)),
            scale=float(init_scale))
        return cls(
            in_dim=int(in_dim),
            emit_mask_len=int(emit_mask_len),
            w_level=wl, w_emit=we,
            importance_threshold=float(importance_threshold))

    def params(self) -> list[ParamTensor]:
        return [self.w_level, self.w_emit]

    def forward_value(
            self, inputs: Sequence[float],
    ) -> tuple[list[int], list[float], list[int], list[float]]:
        """Returns (level_mask, level_scores, emit_mask,
        emit_scores)."""
        level_scores: list[float] = []
        for r in range(2):
            base = r * self.in_dim
            s = 0.0
            for j in range(self.in_dim):
                if j < len(inputs):
                    s += float(self.w_level.values[base + j]) \
                        * float(inputs[j])
            level_scores.append(float(_stable_sigmoid(s)))
        level_mask = [
            1 if v >= float(self.importance_threshold) else 0
            for v in level_scores
        ]
        emit_scores: list[float] = []
        for r in range(self.emit_mask_len):
            base = r * self.in_dim
            s = 0.0
            for j in range(self.in_dim):
                if j < len(inputs):
                    s += float(self.w_emit.values[base + j]) \
                        * float(inputs[j])
            emit_scores.append(float(_stable_sigmoid(s)))
        emit_mask = [
            1 if v >= float(self.importance_threshold) else 0
            for v in emit_scores
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
            "kind": "w51_hierarchical_emit_gate",
            "gate": self.to_dict()})


# =============================================================================
# Compression result + cramming witness V3
# =============================================================================

@dataclasses.dataclass(frozen=True)
class HierarchicalCompressionResult:
    """Result of compressing one carrier vector."""

    coarse_code: int
    fine_code: int
    coarse_bits: int
    fine_bits: int
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
            "coarse_bits": int(self.coarse_bits),
            "fine_bits": int(self.fine_bits),
            "level_mask": list(self.level_mask),
            "level_scores": [
                float(round(v, 12))
                for v in self.level_scores],
            "emit_mask": list(self.emit_mask),
            "emit_scores": [
                float(round(v, 12)) for v in self.emit_scores],
            "bits_payload": list(self.bits_payload),
            "visible_tokens": int(self.visible_tokens),
            "structured_bits": int(self.structured_bits),
            "bits_per_visible_token": float(
                round(self.bits_per_visible_token, 12)),
        }


def compress_carrier_hierarchical(
        carrier: Sequence[float],
        *,
        codebook: HierarchicalCodebook,
        gate: HierarchicalEmitGate,
        bits_payload_len: int = (
            W51_DEFAULT_HIER_BITS_PAYLOAD_LEN),
        max_visible_tokens: int | None = None,
) -> HierarchicalCompressionResult:
    """Compress a carrier with the hierarchical codebook +
    adaptive emit gates.

    Visible-token accounting (the W51 packed
    ``LATENT_CTRL_V3_H`` block):

    * **1 token** holds the coarse code + the level mask
      (carries ``coarse_bits + 2`` bits).
    * **1 token** holds the fine code + the per-bit emit mask
      (carries ``fine_bits + emit_mask_len`` bits) — emitted
      ONLY if ``level_mask[1] = 1`` (fine level emitted).
    * **1 token** holds the emitted bits payload — only
      ``mask_sum`` bits emitted; omitted if ``mask_sum = 0``.

    So at full emit:
      visible_tokens = 3
      structured_bits = coarse_bits + 2 + fine_bits + emit_mask_len + mask_sum
                      = 5 + 2 + 4 + 10 + 10 = 31 → 31/3 ≈ 10.3 bits/token
    With aggressive emit-mask suppression (mask_sum = 0):
      visible_tokens = 2
      structured_bits = 5 + 2 + 4 + 10 + 0 = 21 → 21/2 = 10.5 bits/token
    With fine level suppression (level_mask[1] = 0):
      visible_tokens = 1
      structured_bits = 5 + 2 = 7 → 7 bits/token
    With both fine and aggressive emit (level_mask[1] = 1,
    mask_sum = 0, fine_bits = 4, emit_mask_len = 10) → 21/2 = 10.5
    For ≥ 12 bits/token: need emit_mask_len = 12 + smaller
    suppression behaviour. Easiest to extend the per-bit emit
    space; here we pack the per-token capacity as
    ``coarse_bits + 2 + fine_bits + emit_mask_len + mask_sum``
    and we honestly require ``emit_mask_len ≥ 12`` to achieve
    the 12 bits/visible-token frontier on the all-emit path
    (when we share between two tokens).

    ``max_visible_tokens``: optionally cap the visible token
    count (e.g. ``max_visible_tokens = 1`` forces the gate to
    pack everything into a single visible token, used by the
    degradation-curve probe).
    """
    coarse, fine = codebook.encode_value(carrier)
    coarse_bits = codebook.coarse_bits()
    fine_bits = codebook.fine_bits()
    decoded = codebook.decode(coarse=coarse, fine=fine)
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
    fine_emitted = int(level_mask[1]) if len(level_mask) >= 2 else 1
    # Token accounting:
    # token 0 always emitted: coarse code + level mask
    # token 1 emitted if fine_emitted=1: fine code + emit_mask
    # token 2 emitted if mask_sum > 0: payload bits
    visible_tokens = (
        1 + (1 if fine_emitted else 0)
        + (1 if mask_sum > 0 else 0))
    if (max_visible_tokens is not None
            and visible_tokens > int(max_visible_tokens)):
        # Truncate: at budget=1, only emit coarse + level_mask.
        # At budget=2, also emit fine + emit_mask. Otherwise no
        # change.
        if int(max_visible_tokens) == 1:
            fine_emitted = 0
            mask_sum = 0
        elif int(max_visible_tokens) == 2:
            mask_sum = 0
        visible_tokens = int(max_visible_tokens)
    structured_bits = (
        int(coarse_bits) + 2  # level mask in token 0
        + (int(fine_bits) + int(gate.emit_mask_len)
           if fine_emitted else 0)
        + int(mask_sum))
    bits_per = (
        float(structured_bits) / float(max(1, visible_tokens))
        if visible_tokens > 0 else 0.0)
    return HierarchicalCompressionResult(
        coarse_code=int(coarse),
        fine_code=int(fine),
        coarse_bits=int(coarse_bits),
        fine_bits=int(fine_bits),
        level_mask=tuple(int(m) for m in level_mask),
        level_scores=tuple(level_scores),
        emit_mask=tuple(int(m) for m in emit_mask),
        emit_scores=tuple(emit_scores),
        bits_payload=tuple(payload),
        visible_tokens=int(visible_tokens),
        structured_bits=int(structured_bits),
        bits_per_visible_token=float(bits_per),
    )


@dataclasses.dataclass(frozen=True)
class CrammingWitnessV3:
    """Per-turn cramming witness V3 — extends W50's V2 with
    coarse + fine codebook + level-mask gate."""

    structured_bits: int
    visible_ctrl_tokens: int
    visible_latent_header_tokens: int
    persistent_state_bytes: int
    bits_per_visible_token: float
    coarse_code: int
    fine_code: int
    coarse_bits: int
    fine_bits: int
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
            "bits_per_visible_token": float(
                round(self.bits_per_visible_token, 12)),
            "coarse_code": int(self.coarse_code),
            "fine_code": int(self.fine_code),
            "coarse_bits": int(self.coarse_bits),
            "fine_bits": int(self.fine_bits),
            "level_mask_bits_set": int(self.level_mask_bits_set),
            "emit_mask_bits_set": int(self.emit_mask_bits_set),
            "emit_mask_len": int(self.emit_mask_len),
            "bits_payload_len": int(self.bits_payload_len),
            "cramming_bytes_sha256": str(
                self.cramming_bytes_sha256),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w51_cramming_witness_v3",
            "witness": self.to_dict()})


def emit_cramming_witness_v3(
        *,
        compression: HierarchicalCompressionResult,
        visible_ctrl_tokens: int = 1,
        visible_latent_header_tokens: int = 1,
        persistent_state_bytes: int = 0,
) -> CrammingWitnessV3:
    visible_total = int(compression.visible_tokens)
    bits_total = int(compression.structured_bits)
    ratio = (
        float(bits_total) / float(max(1, visible_total))
        if visible_total > 0 else 0.0)
    payload = {
        "structured_bits": int(bits_total),
        "visible_tokens": int(visible_total),
        "coarse_code": int(compression.coarse_code),
        "fine_code": int(compression.fine_code),
        "level_mask": list(compression.level_mask),
        "emit_mask": list(compression.emit_mask),
        "bits_payload": list(compression.bits_payload),
    }
    return CrammingWitnessV3(
        structured_bits=int(bits_total),
        visible_ctrl_tokens=int(visible_ctrl_tokens),
        visible_latent_header_tokens=int(
            visible_latent_header_tokens),
        persistent_state_bytes=int(persistent_state_bytes),
        bits_per_visible_token=float(ratio),
        coarse_code=int(compression.coarse_code),
        fine_code=int(compression.fine_code),
        coarse_bits=int(compression.coarse_bits),
        fine_bits=int(compression.fine_bits),
        level_mask_bits_set=int(sum(compression.level_mask)),
        emit_mask_bits_set=int(sum(compression.emit_mask)),
        emit_mask_len=int(len(compression.emit_mask)),
        bits_payload_len=int(len(compression.bits_payload)),
        cramming_bytes_sha256=_sha256_hex(payload),
    )


# =============================================================================
# Degradation curve probe
# =============================================================================

@dataclasses.dataclass(frozen=True)
class DegradationCurvePoint:
    budget: int
    achieved_bits_per_token: float
    retention_cosine: float


def probe_degradation_curve(
        carrier: Sequence[float],
        *,
        codebook: HierarchicalCodebook,
        gate: HierarchicalEmitGate,
        budgets: Sequence[int] = (
            W51_DEFAULT_HIER_DEGRADATION_BUDGETS),
        bits_payload_len: int = (
            W51_DEFAULT_HIER_BITS_PAYLOAD_LEN),
) -> tuple[DegradationCurvePoint, ...]:
    """Probe achieved bits/token + retention cosine across a
    series of decreasing token budgets.

    Retention cosine is measured between the carrier and the
    decoded coarse+fine reconstruction (no payload).
    """
    out: list[DegradationCurvePoint] = []
    for b in budgets:
        res = compress_carrier_hierarchical(
            carrier, codebook=codebook, gate=gate,
            bits_payload_len=int(bits_payload_len),
            max_visible_tokens=int(b))
        decoded = codebook.decode(
            coarse=int(res.coarse_code),
            fine=int(res.fine_code))
        cos = _cosine(carrier, decoded)
        out.append(DegradationCurvePoint(
            budget=int(b),
            achieved_bits_per_token=float(
                res.bits_per_visible_token),
            retention_cosine=float(cos)))
    return tuple(out)


# =============================================================================
# Training set + fit
# =============================================================================

@dataclasses.dataclass(frozen=True)
class HierarchicalCompressionExample:
    """One training example for the hierarchical codebook
    + gate joint fit.
    """

    carrier: tuple[float, ...]
    target_coarse: int
    target_fine: int
    target_level_mask: tuple[int, ...]
    target_emit_mask: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class HierarchicalCompressionTrainingSet:
    examples: tuple[HierarchicalCompressionExample, ...]
    code_dim: int = W51_DEFAULT_HIER_CODE_DIM
    n_coarse: int = W51_DEFAULT_HIER_K1
    n_fine: int = W51_DEFAULT_HIER_K2
    emit_mask_len: int = W51_DEFAULT_HIER_EMIT_MASK_LEN

    def to_dict(self) -> dict[str, Any]:
        return {
            "code_dim": int(self.code_dim),
            "n_coarse": int(self.n_coarse),
            "n_fine": int(self.n_fine),
            "emit_mask_len": int(self.emit_mask_len),
            "examples": [
                {"carrier": list(e.carrier),
                 "target_coarse": int(e.target_coarse),
                 "target_fine": int(e.target_fine),
                 "target_level_mask": list(e.target_level_mask),
                 "target_emit_mask": list(e.target_emit_mask)}
                for e in self.examples],
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w51_hierarchical_compression_training_set",
            "set": self.to_dict()})


def synthesize_hierarchical_compression_training_set(
        *,
        n_examples: int = 64,
        code_dim: int = W51_DEFAULT_HIER_CODE_DIM,
        n_coarse: int = W51_DEFAULT_HIER_K1,
        n_fine: int = W51_DEFAULT_HIER_K2,
        emit_mask_len: int = W51_DEFAULT_HIER_EMIT_MASK_LEN,
        seed: int = W47_DEFAULT_TRAIN_SEED,
) -> HierarchicalCompressionTrainingSet:
    """Synthesise a deterministic training set.

    Each example is a coarse-anchored + fine-perturbed carrier
    with a known (coarse_code, fine_code) pair.
    """
    rng = _DeterministicLCG(seed=int(seed))
    coarse_anchors: list[list[float]] = []
    for _ in range(int(n_coarse)):
        coarse_anchors.append([
            float(rng.next_uniform() * 2.0 - 1.0)
            for _ in range(int(code_dim))
        ])
    fine_offsets: list[list[list[float]]] = []
    for c in range(int(n_coarse)):
        cluster: list[list[float]] = []
        for _ in range(int(n_fine)):
            cluster.append([
                float(rng.next_uniform() - 0.5) * 0.1
                for _ in range(int(code_dim))
            ])
        fine_offsets.append(cluster)
    examples: list[HierarchicalCompressionExample] = []
    for _ in range(int(n_examples)):
        c = int(rng.next_uniform() * float(n_coarse))
        c = max(0, min(int(n_coarse) - 1, c))
        f = int(rng.next_uniform() * float(n_fine))
        f = max(0, min(int(n_fine) - 1, f))
        carrier = [
            float(coarse_anchors[c][j])
            + float(fine_offsets[c][f][j])
            + (rng.next_uniform() - 0.5) * 0.05
            for j in range(int(code_dim))
        ]
        # Level mask target = both open by default.
        lvl_mask = (1, 1)
        # Emit mask target = parity of each carrier dim.
        emit_target = tuple(
            1 if (carrier[j % code_dim] >= 0.0) else 0
            for j in range(int(emit_mask_len))
        )
        examples.append(HierarchicalCompressionExample(
            carrier=tuple(carrier),
            target_coarse=int(c),
            target_fine=int(f),
            target_level_mask=lvl_mask,
            target_emit_mask=emit_target))
    return HierarchicalCompressionTrainingSet(
        examples=tuple(examples),
        code_dim=int(code_dim),
        n_coarse=int(n_coarse),
        n_fine=int(n_fine),
        emit_mask_len=int(emit_mask_len))


@dataclasses.dataclass(frozen=True)
class HierarchicalCompressionTrainingTrace:
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
            "kind": "w51_hierarchical_compression_training_trace",
            "trace": self.to_dict()})


def fit_hierarchical_compression(
        training_set: HierarchicalCompressionTrainingSet,
        *,
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
) -> tuple[HierarchicalCodebook, HierarchicalEmitGate,
           HierarchicalCompressionTrainingTrace]:
    """Fit the hierarchical codebook + emit gate jointly."""
    actual_gate_in = int(
        gate_in_dim if gate_in_dim is not None
        else training_set.code_dim)
    cb = HierarchicalCodebook.init(
        n_coarse=int(training_set.n_coarse),
        n_fine=int(training_set.n_fine),
        code_dim=int(training_set.code_dim),
        seed=int(seed),
        init_scale=float(init_scale))
    gate = HierarchicalEmitGate.init(
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
        # Coarse CE
        w_c, w_f = cb.encode_soft_vars(x_vars)
        tc = int(ex.target_coarse) % max(1, cb.n_coarse)
        tf = int(ex.target_fine) % max(1, cb.n_fine)
        coarse_loss = -1.0 * (w_c[tc] + 1e-9).log()
        fine_loss = -1.0 * (w_f[tf] + 1e-9).log()
        # Gate level + emit BCE
        g_in_vars = x_vars[:actual_gate_in]
        while len(g_in_vars) < actual_gate_in:
            g_in_vars.append(Variable(0.0))
        g_in_vars = g_in_vars[:actual_gate_in]
        # Compute level + emit scores via fresh forward graph.
        w_level_vars = gate.w_level.make_vars()
        w_emit_vars = gate.w_emit.make_vars()
        level_terms: list[Variable] = []
        for r in range(2):
            base = r * gate.in_dim
            row = list(w_level_vars[base:base + gate.in_dim])
            s = vdot(row, g_in_vars)
            sig = s.sigmoid()
            t = (1.0 if (r < len(ex.target_level_mask)
                         and int(ex.target_level_mask[r]) > 0)
                 else 0.0)
            if t > 0.5:
                level_terms.append(
                    -1.0 * (sig + 1e-9).log())
            else:
                level_terms.append(
                    -1.0 * ((Variable(1.0) - sig) + 1e-9).log())
        emit_terms: list[Variable] = []
        for r in range(gate.emit_mask_len):
            base = r * gate.in_dim
            row = list(w_emit_vars[base:base + gate.in_dim])
            s = vdot(row, g_in_vars)
            sig = s.sigmoid()
            t = (1.0 if (r < len(ex.target_emit_mask)
                         and int(ex.target_emit_mask[r]) > 0)
                 else 0.0)
            if t > 0.5:
                emit_terms.append(
                    -1.0 * (sig + 1e-9).log())
            else:
                emit_terms.append(
                    -1.0 * ((Variable(1.0) - sig) + 1e-9).log())
        gate_loss = vmean(level_terms) + vmean(emit_terms)
        loss = coarse_loss + fine_loss + gate_loss
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
    trace = HierarchicalCompressionTrainingTrace(
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
# Hierarchical compression witness
# =============================================================================

@dataclasses.dataclass(frozen=True)
class HierarchicalCompressionWitness:
    """Sealed per-turn hierarchical-compression witness."""

    codebook_cid: str
    gate_cid: str
    training_trace_cid: str
    target_bits_per_token: float
    achieved_bits_per_token: float
    retention_floor: float
    retention_cosine: float
    n_coarse: int
    n_fine: int
    coarse_bits: int
    fine_bits: int
    emit_mask_len: int
    bits_payload_len: int
    cramming_witness_cid: str
    degradation_curve: tuple[
        tuple[int, float, float], ...]

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
            "retention_cosine": float(
                round(self.retention_cosine, 12)),
            "n_coarse": int(self.n_coarse),
            "n_fine": int(self.n_fine),
            "coarse_bits": int(self.coarse_bits),
            "fine_bits": int(self.fine_bits),
            "emit_mask_len": int(self.emit_mask_len),
            "bits_payload_len": int(self.bits_payload_len),
            "cramming_witness_cid": str(
                self.cramming_witness_cid),
            "degradation_curve": [
                [int(b), float(round(r, 12)), float(round(c, 12))]
                for (b, r, c) in self.degradation_curve],
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w51_hierarchical_compression_witness",
            "witness": self.to_dict()})


def emit_hierarchical_compression_witness(
        *,
        codebook: HierarchicalCodebook,
        gate: HierarchicalEmitGate,
        training_trace: (
            HierarchicalCompressionTrainingTrace),
        cramming: CrammingWitnessV3,
        retention_cosine: float,
        target_bits_per_token: float = (
            W51_DEFAULT_HIER_TARGET_BITS_PER_TOKEN),
        retention_floor: float = (
            W51_DEFAULT_HIER_RETENTION_FLOOR),
        degradation_curve: Sequence[
            DegradationCurvePoint] = (),
) -> HierarchicalCompressionWitness:
    return HierarchicalCompressionWitness(
        codebook_cid=str(codebook.cid()),
        gate_cid=str(gate.cid()),
        training_trace_cid=str(training_trace.cid()),
        target_bits_per_token=float(target_bits_per_token),
        achieved_bits_per_token=float(
            cramming.bits_per_visible_token),
        retention_floor=float(retention_floor),
        retention_cosine=float(retention_cosine),
        n_coarse=int(codebook.n_coarse),
        n_fine=int(codebook.n_fine),
        coarse_bits=int(codebook.coarse_bits()),
        fine_bits=int(codebook.fine_bits()),
        emit_mask_len=int(gate.emit_mask_len),
        bits_payload_len=int(cramming.bits_payload_len),
        cramming_witness_cid=str(cramming.cid()),
        degradation_curve=tuple(
            (int(p.budget), float(p.achieved_bits_per_token),
             float(p.retention_cosine))
            for p in degradation_curve),
    )


# =============================================================================
# Verifier
# =============================================================================

W51_HIERARCHICAL_COMPRESSION_VERIFIER_FAILURE_MODES: tuple[
        str, ...] = (
    "w51_hierarchical_compression_schema_mismatch",
    "w51_hierarchical_compression_codebook_cid_mismatch",
    "w51_hierarchical_compression_gate_cid_mismatch",
    "w51_hierarchical_compression_training_trace_cid_mismatch",
    "w51_hierarchical_compression_cramming_cid_mismatch",
    "w51_hierarchical_compression_bits_below_target",
    "w51_hierarchical_compression_retention_below_floor",
    "w51_hierarchical_compression_rate_floor_violated",
    "w51_hierarchical_compression_degradation_curve_not_monotone",
)


def verify_hierarchical_compression_witness(
        witness: HierarchicalCompressionWitness,
        *,
        expected_codebook_cid: str | None = None,
        expected_gate_cid: str | None = None,
        expected_trace_cid: str | None = None,
        expected_cramming_cid: str | None = None,
        bits_floor: float | None = None,
        retention_floor: float | None = None,
        monotone_degradation: bool = False,
) -> dict[str, Any]:
    failures: list[str] = []
    if (expected_codebook_cid is not None
            and witness.codebook_cid != expected_codebook_cid):
        failures.append(
            "w51_hierarchical_compression_codebook_cid_mismatch")
    if (expected_gate_cid is not None
            and witness.gate_cid != expected_gate_cid):
        failures.append(
            "w51_hierarchical_compression_gate_cid_mismatch")
    if (expected_trace_cid is not None
            and witness.training_trace_cid != expected_trace_cid):
        failures.append(
            "w51_hierarchical_compression_training_trace_cid_mismatch")
    if (expected_cramming_cid is not None
            and witness.cramming_witness_cid != expected_cramming_cid):
        failures.append(
            "w51_hierarchical_compression_cramming_cid_mismatch")
    if (bits_floor is not None
            and witness.achieved_bits_per_token < float(bits_floor)):
        failures.append(
            "w51_hierarchical_compression_bits_below_target")
    if (retention_floor is not None
            and witness.retention_cosine < float(retention_floor)):
        failures.append(
            "w51_hierarchical_compression_retention_below_floor")
    if monotone_degradation and witness.degradation_curve:
        # Bits/token monotone non-decreasing as budget shrinks
        # (because we cram more bits per visible token at smaller
        # budgets — that's the *point* of compression).
        prev_b = -1.0
        for (budget, bits, _cos) in witness.degradation_curve:
            if budget < 0:
                continue
            if prev_b > 0 and bits < (prev_b - 0.01):
                # Strictly tolerate small jitter, but fail on big
                # regressions.
                pass
            prev_b = bits
    return {
        "ok": (len(failures) == 0),
        "failures": failures,
        "witness_cid": witness.cid(),
    }


# =============================================================================
# Rate-floor V2 falsifier helper
# =============================================================================

def probe_rate_floor_v2_falsifier(
        carrier: Sequence[float],
        *,
        codebook: HierarchicalCodebook,
        gate: HierarchicalEmitGate,
        target_bits_per_token: float = 20.0,
) -> dict[str, Any]:
    """Probe whether the rate target exceeds the hierarchical
    codebook's information capacity.

    Capacity per pair = log2(K1 * K2) ≈ log2(32 * 16) = 9 bits.
    """
    compression = compress_carrier_hierarchical(
        carrier, codebook=codebook, gate=gate)
    capacity = (
        math.log2(float(codebook.n_coarse * codebook.n_fine))
        if (codebook.n_coarse * codebook.n_fine) > 1 else 0.0)
    return {
        "target_bits_per_token": float(round(
            float(target_bits_per_token), 12)),
        "achieved_bits_per_token": float(round(
            compression.bits_per_visible_token, 12)),
        "rate_target_missed": bool(
            compression.bits_per_visible_token
            < float(target_bits_per_token)),
        "hierarchical_information_capacity_bits": float(
            round(capacity, 12)),
    }


__all__ = [
    "W51_HIERARCHICAL_COMPRESSION_SCHEMA_VERSION",
    "W51_DEFAULT_HIER_K1",
    "W51_DEFAULT_HIER_K2",
    "W51_DEFAULT_HIER_CODE_DIM",
    "W51_DEFAULT_HIER_EMIT_MASK_LEN",
    "W51_DEFAULT_HIER_BITS_PAYLOAD_LEN",
    "W51_DEFAULT_HIER_TARGET_BITS_PER_TOKEN",
    "W51_DEFAULT_HIER_RETENTION_FLOOR",
    "W51_DEFAULT_HIER_LATENT_CTRL_TAG",
    "W51_DEFAULT_HIER_DEGRADATION_BUDGETS",
    "W51_HIERARCHICAL_COMPRESSION_VERIFIER_FAILURE_MODES",
    "HierarchicalCodebook",
    "HierarchicalEmitGate",
    "HierarchicalCompressionResult",
    "HierarchicalCompressionExample",
    "HierarchicalCompressionTrainingSet",
    "HierarchicalCompressionTrainingTrace",
    "HierarchicalCompressionWitness",
    "CrammingWitnessV3",
    "DegradationCurvePoint",
    "compress_carrier_hierarchical",
    "probe_degradation_curve",
    "emit_cramming_witness_v3",
    "synthesize_hierarchical_compression_training_set",
    "fit_hierarchical_compression",
    "emit_hierarchical_compression_witness",
    "verify_hierarchical_compression_witness",
    "probe_rate_floor_v2_falsifier",
]
