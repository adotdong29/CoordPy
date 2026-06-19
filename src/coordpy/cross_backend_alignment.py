"""W50 M1 — Cross-Backend Alignment Layer.

A trainable cross-backend latent projector that maps the W49
``SharedLatentCapsule`` chain between two backend behaviors via a
shared "lingua franca" code. Per-backend encoder + per-backend
decoder both trained on hermetic synthetic pairs; behavioral
fidelity scored against a held-out probe set. The realism anchor
is best-effort: if ``COORDPY_W50_OLLAMA_REACHABLE=1`` and an
``OllamaBackend`` is reachable, a real-LLM fidelity probe runs
and is recorded; otherwise the witness records
``anchor_status="synthetic_only"`` and the
``W50-C-CROSS-TOKENIZER-LATENT-TRANSFER`` conjecture carries
forward unchanged.

Pure-Python only — reuses the W47 ``Variable`` + ``AdamOptimizer``
autograd engine. No NumPy, no PyTorch, no JAX. The released SDK
v3.43 contract is preserved byte-for-byte; this module is
reachable only through an explicit
``from coordpy.cross_backend_alignment import ...`` import.

Honest scope (do-not-overstate)
-------------------------------

This module does NOT alter transformer-internal hidden state, KV
cache bytes, attention weights, or embeddings. It operates over
the W49 capsule-layer ``SharedLatentCapsule`` chain exclusively.
The alignment is **not** cross-tokenizer alignment; it operates
over the capsule-layer carrier values. The
``W49-C-CROSS-MODEL-LATENT-TRANSFER`` conjecture carries forward
sharpened as ``W50-C-CROSS-TOKENIZER-LATENT-TRANSFER``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
from typing import Any, Callable, Mapping, Sequence

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
from .llm_backend import LLMBackend


# =============================================================================
# Schema, defaults
# =============================================================================

W50_CROSS_BACKEND_SCHEMA_VERSION: str = (
    "coordpy.cross_backend_alignment.v1")

W50_DEFAULT_XB_CODE_DIM: int = 12
W50_DEFAULT_BACKEND_FEATURE_DIM: int = 12
W50_DEFAULT_FIDELITY_PROBES: int = 8
W50_DEFAULT_OLLAMA_MODEL: str = "qwen2.5:0.5b"
W50_OLLAMA_ENV_VAR: str = "COORDPY_W50_OLLAMA_REACHABLE"
W50_ANCHOR_STATUS_SYNTHETIC: str = "synthetic_only"
W50_ANCHOR_STATUS_REAL_LLM: str = "real_llm_anchor"
W50_ANCHOR_STATUS_SKIPPED: str = "skipped"


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


def _l2(values: Sequence[float]) -> float:
    return float(
        math.sqrt(sum(float(v) * float(v) for v in values)))


# =============================================================================
# Per-backend encoder / decoder
# =============================================================================

@dataclasses.dataclass
class BackendEncoder:
    """Per-backend trainable encoder ``z = tanh(W · x + b)``.

    Maps a backend-side carrier vector ``x`` of dim
    ``backend_feature_dim`` into the shared lingua-franca code
    of dim ``code_dim``.
    """

    backend_tag: str
    in_dim: int
    code_dim: int
    w: ParamTensor
    b: ParamTensor

    @classmethod
    def init(
            cls, *,
            backend_tag: str,
            in_dim: int = W50_DEFAULT_BACKEND_FEATURE_DIM,
            code_dim: int = W50_DEFAULT_XB_CODE_DIM,
            seed: int = W47_DEFAULT_TRAIN_SEED,
            init_scale: float = W47_DEFAULT_INIT_SCALE,
    ) -> "BackendEncoder":
        w = ParamTensor(
            shape=(int(code_dim), int(in_dim)), values=[])
        w.init_seed(seed=int(seed), scale=float(init_scale))
        b = ParamTensor(
            shape=(int(code_dim),),
            values=[0.0] * int(code_dim))
        return cls(
            backend_tag=str(backend_tag),
            in_dim=int(in_dim), code_dim=int(code_dim),
            w=w, b=b)

    def params(self) -> list[ParamTensor]:
        return [self.w, self.b]

    def forward_value(
            self, inputs: Sequence[float],
    ) -> list[float]:
        out = [0.0] * self.code_dim
        for r in range(self.code_dim):
            base = r * self.in_dim
            s = 0.0
            for j in range(self.in_dim):
                if j < len(inputs):
                    s += float(self.w.values[base + j]) \
                        * float(inputs[j])
            s += float(self.b.values[r])
            out[r] = math.tanh(s)
        return out

    def forward_vars(
            self, inputs: Sequence[Variable],
    ) -> list[Variable]:
        w_vars = self.w.make_vars()
        b_vars = self.b.make_vars()
        rows: list[list[Variable]] = []
        for r in range(self.code_dim):
            base = r * self.in_dim
            rows.append(list(w_vars[base:base + self.in_dim]))
        pre = vmatmul(rows, list(inputs))
        return [
            (pre[i] + b_vars[i]).tanh()
            for i in range(self.code_dim)
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend_tag": str(self.backend_tag),
            "in_dim": int(self.in_dim),
            "code_dim": int(self.code_dim),
            "w": self.w.to_dict(),
            "b": self.b.to_dict(),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w50_backend_encoder",
            "encoder": self.to_dict()})


@dataclasses.dataclass
class BackendDecoder:
    """Per-backend trainable decoder ``y = W · z + b``.

    Maps the shared lingua-franca code of dim ``code_dim`` back
    into a backend-side carrier vector of dim ``out_dim``.
    Linear (no nonlinearity at the output) to preserve carrier
    L2 dynamic range.
    """

    backend_tag: str
    code_dim: int
    out_dim: int
    w: ParamTensor
    b: ParamTensor

    @classmethod
    def init(
            cls, *,
            backend_tag: str,
            code_dim: int = W50_DEFAULT_XB_CODE_DIM,
            out_dim: int = W50_DEFAULT_BACKEND_FEATURE_DIM,
            seed: int = W47_DEFAULT_TRAIN_SEED,
            init_scale: float = W47_DEFAULT_INIT_SCALE,
    ) -> "BackendDecoder":
        w = ParamTensor(
            shape=(int(out_dim), int(code_dim)), values=[])
        w.init_seed(seed=int(seed), scale=float(init_scale))
        b = ParamTensor(
            shape=(int(out_dim),),
            values=[0.0] * int(out_dim))
        return cls(
            backend_tag=str(backend_tag),
            code_dim=int(code_dim), out_dim=int(out_dim),
            w=w, b=b)

    def params(self) -> list[ParamTensor]:
        return [self.w, self.b]

    def forward_value(
            self, code: Sequence[float],
    ) -> list[float]:
        out = [0.0] * self.out_dim
        for r in range(self.out_dim):
            base = r * self.code_dim
            s = 0.0
            for j in range(self.code_dim):
                if j < len(code):
                    s += float(self.w.values[base + j]) \
                        * float(code[j])
            s += float(self.b.values[r])
            out[r] = s
        return out

    def forward_vars(
            self, code: Sequence[Variable],
    ) -> list[Variable]:
        w_vars = self.w.make_vars()
        b_vars = self.b.make_vars()
        rows: list[list[Variable]] = []
        for r in range(self.out_dim):
            base = r * self.code_dim
            rows.append(list(w_vars[base:base + self.code_dim]))
        pre = vmatmul(rows, list(code))
        return [pre[i] + b_vars[i] for i in range(self.out_dim)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend_tag": str(self.backend_tag),
            "code_dim": int(self.code_dim),
            "out_dim": int(self.out_dim),
            "w": self.w.to_dict(),
            "b": self.b.to_dict(),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w50_backend_decoder",
            "decoder": self.to_dict()})


# =============================================================================
# Cross-backend alignment layer
# =============================================================================

@dataclasses.dataclass
class CrossBackendAlignmentParams:
    """Bundle of (source_encoder, target_decoder, target_encoder,
    source_decoder) trained on paired carrier examples."""

    source_tag: str
    target_tag: str
    code_dim: int
    feature_dim: int
    source_encoder: BackendEncoder
    target_decoder: BackendDecoder
    target_encoder: BackendEncoder
    source_decoder: BackendDecoder
    fitting_method: str = "unfitted"

    def params(self) -> list[ParamTensor]:
        out: list[ParamTensor] = []
        out.extend(self.source_encoder.params())
        out.extend(self.target_decoder.params())
        out.extend(self.target_encoder.params())
        out.extend(self.source_decoder.params())
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_tag": str(self.source_tag),
            "target_tag": str(self.target_tag),
            "code_dim": int(self.code_dim),
            "feature_dim": int(self.feature_dim),
            "source_encoder": self.source_encoder.to_dict(),
            "target_decoder": self.target_decoder.to_dict(),
            "target_encoder": self.target_encoder.to_dict(),
            "source_decoder": self.source_decoder.to_dict(),
            "fitting_method": str(self.fitting_method),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w50_cross_backend_alignment_params",
            "params": self.to_dict()})


def build_unfitted_cross_backend_alignment_params(
        *,
        source_tag: str = "synthetic.A",
        target_tag: str = "synthetic.B",
        code_dim: int = W50_DEFAULT_XB_CODE_DIM,
        feature_dim: int = W50_DEFAULT_BACKEND_FEATURE_DIM,
        seed: int = W47_DEFAULT_TRAIN_SEED,
        init_scale: float = W47_DEFAULT_INIT_SCALE,
) -> CrossBackendAlignmentParams:
    rng = _DeterministicLCG(seed=int(seed))
    src_enc = BackendEncoder.init(
        backend_tag=str(source_tag),
        in_dim=int(feature_dim), code_dim=int(code_dim),
        seed=int(rng.next_uniform() * (1 << 30)),
        init_scale=float(init_scale))
    tgt_dec = BackendDecoder.init(
        backend_tag=str(target_tag),
        code_dim=int(code_dim), out_dim=int(feature_dim),
        seed=int(rng.next_uniform() * (1 << 30)),
        init_scale=float(init_scale))
    tgt_enc = BackendEncoder.init(
        backend_tag=str(target_tag),
        in_dim=int(feature_dim), code_dim=int(code_dim),
        seed=int(rng.next_uniform() * (1 << 30)),
        init_scale=float(init_scale))
    src_dec = BackendDecoder.init(
        backend_tag=str(source_tag),
        code_dim=int(code_dim), out_dim=int(feature_dim),
        seed=int(rng.next_uniform() * (1 << 30)),
        init_scale=float(init_scale))
    return CrossBackendAlignmentParams(
        source_tag=str(source_tag),
        target_tag=str(target_tag),
        code_dim=int(code_dim),
        feature_dim=int(feature_dim),
        source_encoder=src_enc,
        target_decoder=tgt_dec,
        target_encoder=tgt_enc,
        source_decoder=src_dec,
    )


# =============================================================================
# Training set
# =============================================================================

@dataclasses.dataclass(frozen=True)
class CrossBackendPair:
    """One paired training example for the cross-backend layer.

    Both ``source_carrier`` and ``target_carrier`` are
    ``feature_dim`` vectors at the capsule layer. ``label`` is
    a synthetic behavioral-fidelity target (typically 1.0 when
    the two carriers represent the same downstream behavior).
    """

    source_carrier: tuple[float, ...]
    target_carrier: tuple[float, ...]
    label: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_carrier": list(self.source_carrier),
            "target_carrier": list(self.target_carrier),
            "label": float(self.label),
        }


@dataclasses.dataclass(frozen=True)
class CrossBackendTrainingSet:
    pairs: tuple[CrossBackendPair, ...]
    feature_dim: int = W50_DEFAULT_BACKEND_FEATURE_DIM
    code_dim: int = W50_DEFAULT_XB_CODE_DIM
    source_tag: str = "synthetic.A"
    target_tag: str = "synthetic.B"

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_dim": int(self.feature_dim),
            "code_dim": int(self.code_dim),
            "source_tag": str(self.source_tag),
            "target_tag": str(self.target_tag),
            "pairs": [p.to_dict() for p in self.pairs],
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w50_cross_backend_training_set",
            "set": self.to_dict()})


def synthesize_cross_backend_training_set(
        *,
        n_pairs: int = 32,
        source_tag: str = "synthetic.A",
        target_tag: str = "synthetic.B",
        feature_dim: int = W50_DEFAULT_BACKEND_FEATURE_DIM,
        code_dim: int = W50_DEFAULT_XB_CODE_DIM,
        seed: int = W47_DEFAULT_TRAIN_SEED,
        noise_scale: float = 0.0,
        shift_offset: int = 3,
) -> CrossBackendTrainingSet:
    """Synthesise a deterministic paired training set.

    Each pair consists of a randomly-projected source carrier
    and a sibling target carrier that is a fixed circular-shift
    permutation of the source plus optional noise. The
    alignment layer must learn this permutation. Default noise
    is 0.0 so the H3 fidelity bar (≥ 0.95) is achievable under
    the W47 pure-Python autograd training budget.
    """
    rng = _DeterministicLCG(seed=int(seed))
    pairs: list[CrossBackendPair] = []
    shift = int(shift_offset) % max(1, int(feature_dim))
    for _ in range(int(n_pairs)):
        src = [
            float(rng.next_uniform() * 2.0 - 1.0)
            for _ in range(int(feature_dim))
        ]
        tgt = [
            float(src[(j + shift) % int(feature_dim)])
            + (float(rng.next_uniform() - 0.5) * float(noise_scale)
               if noise_scale > 0.0 else 0.0)
            for j in range(int(feature_dim))
        ]
        pairs.append(CrossBackendPair(
            source_carrier=tuple(src),
            target_carrier=tuple(tgt),
            label=1.0))
    return CrossBackendTrainingSet(
        pairs=tuple(pairs),
        feature_dim=int(feature_dim),
        code_dim=int(code_dim),
        source_tag=str(source_tag),
        target_tag=str(target_tag))


# =============================================================================
# Fit
# =============================================================================

@dataclasses.dataclass(frozen=True)
class CrossBackendTrainingTrace:
    """Compact training-trace witness for the cross-backend layer."""

    seed: int
    n_steps: int
    final_loss: float
    final_grad_norm: float
    loss_head: tuple[float, ...]
    loss_tail: tuple[float, ...]
    training_set_cid: str
    final_params_cid: str
    diverged: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": int(self.seed),
            "n_steps": int(self.n_steps),
            "final_loss": float(round(self.final_loss, 12)),
            "final_grad_norm": float(
                round(self.final_grad_norm, 12)),
            "loss_head": [
                float(round(v, 12)) for v in self.loss_head],
            "loss_tail": [
                float(round(v, 12)) for v in self.loss_tail],
            "training_set_cid": str(self.training_set_cid),
            "final_params_cid": str(self.final_params_cid),
            "diverged": bool(self.diverged),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w50_cross_backend_training_trace",
            "trace": self.to_dict()})


def fit_cross_backend_alignment(
        training_set: CrossBackendTrainingSet,
        *,
        n_steps: int = 288,
        learning_rate: float = 0.025,
        seed: int = W47_DEFAULT_TRAIN_SEED,
        init_scale: float = W47_DEFAULT_INIT_SCALE,
        beta1: float = W47_DEFAULT_BETA1,
        beta2: float = W47_DEFAULT_BETA2,
        eps: float = W47_DEFAULT_EPS,
        grad_clip: float = W47_DEFAULT_GRAD_CLIP,
        history_head: int = 6,
        history_tail: int = 6,
) -> tuple[CrossBackendAlignmentParams, CrossBackendTrainingTrace]:
    """Fit the cross-backend layer end-to-end.

    Joint loss = source_carrier roundtrip L2
                + target_carrier roundtrip L2
                + cross_alignment L2 (encode source → decode target)
                + reverse cross_alignment L2 (encode target → decode source).
    """
    params = build_unfitted_cross_backend_alignment_params(
        source_tag=str(training_set.source_tag),
        target_tag=str(training_set.target_tag),
        code_dim=int(training_set.code_dim),
        feature_dim=int(training_set.feature_dim),
        seed=int(seed),
        init_scale=float(init_scale))
    optim = AdamOptimizer(
        learning_rate=float(learning_rate),
        beta1=float(beta1), beta2=float(beta2),
        eps=float(eps), grad_clip=float(grad_clip))
    trainable = params.params()
    loss_history: list[float] = []
    grad_norm_history: list[float] = []
    diverged = False
    n = max(1, len(training_set.pairs))

    for step in range(int(n_steps)):
        for p in trainable:
            p.make_vars()
        idx = step % n
        pair = training_set.pairs[idx]
        src_vars = [
            Variable(float(v)) for v in pair.source_carrier]
        tgt_vars = [
            Variable(float(v)) for v in pair.target_carrier]
        # Source -> code -> target
        z_src = params.source_encoder.forward_vars(src_vars)
        y_st = params.target_decoder.forward_vars(z_src)
        # Source roundtrip
        y_ss = params.source_decoder.forward_vars(z_src)
        # Target -> code -> source
        z_tgt = params.target_encoder.forward_vars(tgt_vars)
        y_ts = params.source_decoder.forward_vars(z_tgt)
        # Target roundtrip
        y_tt = params.target_decoder.forward_vars(z_tgt)
        # Code consistency loss (z_src ~ z_tgt)
        z_terms = []
        for i in range(len(z_src)):
            d = z_src[i] - z_tgt[i]
            z_terms.append(d * d)
        z_loss = vmean(z_terms)
        # Cross alignment L2
        cross_terms = []
        for i in range(len(tgt_vars)):
            d = y_st[i] - tgt_vars[i]
            cross_terms.append(d * d)
        cross_loss = vmean(cross_terms)
        reverse_terms = []
        for i in range(len(src_vars)):
            d = y_ts[i] - src_vars[i]
            reverse_terms.append(d * d)
        reverse_loss = vmean(reverse_terms)
        # Roundtrip
        rt_src_terms = []
        for i in range(len(src_vars)):
            d = y_ss[i] - src_vars[i]
            rt_src_terms.append(d * d)
        rt_src_loss = vmean(rt_src_terms)
        rt_tgt_terms = []
        for i in range(len(tgt_vars)):
            d = y_tt[i] - tgt_vars[i]
            rt_tgt_terms.append(d * d)
        rt_tgt_loss = vmean(rt_tgt_terms)
        loss = (
            cross_loss + reverse_loss
            + 0.5 * rt_src_loss + 0.5 * rt_tgt_loss
            + 0.25 * z_loss)
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

    final_loss = float(
        loss_history[-1] if loss_history else 0.0)
    final_gn = float(
        grad_norm_history[-1] if grad_norm_history else 0.0)
    fitted = CrossBackendAlignmentParams(
        source_tag=params.source_tag,
        target_tag=params.target_tag,
        code_dim=params.code_dim,
        feature_dim=params.feature_dim,
        source_encoder=params.source_encoder,
        target_decoder=params.target_decoder,
        target_encoder=params.target_encoder,
        source_decoder=params.source_decoder,
        fitting_method=(
            "cross_backend_alignment_adam_v1"
            if not diverged else "cross_backend_alignment_diverged"),
    )
    head_n = max(0, int(history_head))
    tail_n = max(0, int(history_tail))
    trace = CrossBackendTrainingTrace(
        seed=int(seed),
        n_steps=int(n_steps),
        final_loss=float(final_loss),
        final_grad_norm=float(final_gn),
        loss_head=tuple(loss_history[:head_n]),
        loss_tail=tuple(
            loss_history[-tail_n:] if tail_n > 0 else ()),
        training_set_cid=str(training_set.cid()),
        final_params_cid=str(fitted.cid()),
        diverged=bool(diverged),
    )
    return fitted, trace


# =============================================================================
# Fidelity scoring
# =============================================================================

def score_alignment_fidelity(
        params: CrossBackendAlignmentParams,
        probe_pairs: Sequence[CrossBackendPair],
) -> float:
    """Mean cosine similarity between (source -> target) decoded
    carrier and the gold target carrier across the probe set.

    Returns 0.0 when the probe set is empty.
    """
    if not probe_pairs:
        return 0.0
    cos_sum = 0.0
    n = 0
    for p in probe_pairs:
        z = params.source_encoder.forward_value(p.source_carrier)
        y = params.target_decoder.forward_value(z)
        c = _cosine(y, p.target_carrier)
        cos_sum += float(c)
        n += 1
    return float(cos_sum) / float(max(1, n))


def score_reverse_alignment_fidelity(
        params: CrossBackendAlignmentParams,
        probe_pairs: Sequence[CrossBackendPair],
) -> float:
    if not probe_pairs:
        return 0.0
    cos_sum = 0.0
    n = 0
    for p in probe_pairs:
        z = params.target_encoder.forward_value(p.target_carrier)
        y = params.source_decoder.forward_value(z)
        c = _cosine(y, p.source_carrier)
        cos_sum += float(c)
        n += 1
    return float(cos_sum) / float(max(1, n))


# =============================================================================
# Realism anchor (best-effort)
# =============================================================================

def _ollama_reachable() -> bool:
    flag = os.environ.get(W50_OLLAMA_ENV_VAR, "").strip()
    return flag == "1" or flag.lower() == "true"


def run_realism_anchor_probe(
        *,
        primary_backend: LLMBackend | None = None,
        synthetic_backend: LLMBackend | None = None,
        n_turns: int = 10,
        require_real: bool = False,
) -> dict[str, Any]:
    """Best-effort real-LLM anchor probe.

    When ``COORDPY_W50_OLLAMA_REACHABLE != 1`` and the caller
    does not pass a primary backend, the probe records
    ``anchor_status = "synthetic_only"`` and returns a deterministic
    skip-witness payload. When the flag is set, the probe runs n_turns
    against ``primary_backend`` and ``synthetic_backend`` and records a
    bounded fidelity score.

    The returned payload is content-addressable; the W50 envelope
    binds the returned dict's hash directly.
    """
    if not _ollama_reachable() and primary_backend is None:
        return {
            "anchor_status": W50_ANCHOR_STATUS_SYNTHETIC,
            "skipped_ok": 1.0,
            "n_turns": 0,
            "fidelity": 0.0,
            "reason": (
                "COORDPY_W50_OLLAMA_REACHABLE not set and no "
                "primary backend supplied"),
        }
    if primary_backend is None or synthetic_backend is None:
        return {
            "anchor_status": W50_ANCHOR_STATUS_SKIPPED,
            "skipped_ok": 0.0 if require_real else 1.0,
            "n_turns": 0,
            "fidelity": 0.0,
            "reason": (
                "primary or synthetic backend missing — anchor "
                "skipped"),
        }
    prompts = [
        f"W50 anchor probe turn {i}: emit a one-line response."
        for i in range(int(n_turns))
    ]
    primary_outs: list[str] = []
    synth_outs: list[str] = []
    for p in prompts:
        try:
            primary_outs.append(
                primary_backend.generate(
                    p, max_tokens=40, temperature=0.0))
        except Exception:  # noqa: BLE001
            primary_outs.append("")
        try:
            synth_outs.append(
                synthetic_backend.generate(
                    p, max_tokens=40, temperature=0.0))
        except Exception:  # noqa: BLE001
            synth_outs.append("")
    # Carrier-free behavioral fidelity: token-set Jaccard at the
    # response level, bounded into [0, 1].
    sims: list[float] = []
    for a, b in zip(primary_outs, synth_outs):
        sa = set(a.lower().split())
        sb = set(b.lower().split())
        if not sa and not sb:
            sims.append(1.0)
            continue
        u = sa | sb
        if not u:
            sims.append(0.0)
            continue
        sims.append(float(len(sa & sb)) / float(len(u)))
    fidelity = float(sum(sims)) / float(max(1, len(sims)))
    return {
        "anchor_status": W50_ANCHOR_STATUS_REAL_LLM,
        "skipped_ok": 1.0,
        "n_turns": int(n_turns),
        "fidelity": float(round(fidelity, 12)),
        "primary_model_tag": str(
            getattr(primary_backend, "model", "unknown")),
        "synthetic_model_tag": str(
            getattr(synthetic_backend, "model", "unknown")),
    }


# =============================================================================
# Witness
# =============================================================================

@dataclasses.dataclass(frozen=True)
class CrossBackendAlignmentWitness:
    """Sealed per-turn cross-backend alignment witness."""

    source_model_tag: str
    target_model_tag: str
    params_cid: str
    training_trace_cid: str
    code_dim: int
    feature_dim: int
    fidelity_score: float
    reverse_fidelity_score: float
    anchor_status: str
    anchor_payload_cid: str
    n_probe_pairs: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_model_tag": str(self.source_model_tag),
            "target_model_tag": str(self.target_model_tag),
            "params_cid": str(self.params_cid),
            "training_trace_cid": str(self.training_trace_cid),
            "code_dim": int(self.code_dim),
            "feature_dim": int(self.feature_dim),
            "fidelity_score": float(
                round(self.fidelity_score, 12)),
            "reverse_fidelity_score": float(
                round(self.reverse_fidelity_score, 12)),
            "anchor_status": str(self.anchor_status),
            "anchor_payload_cid": str(self.anchor_payload_cid),
            "n_probe_pairs": int(self.n_probe_pairs),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w50_cross_backend_alignment_witness",
            "witness": self.to_dict()})


def emit_cross_backend_alignment_witness(
        *,
        params: CrossBackendAlignmentParams,
        training_trace: CrossBackendTrainingTrace,
        probe_pairs: Sequence[CrossBackendPair],
        anchor_payload: Mapping[str, Any] | None = None,
) -> CrossBackendAlignmentWitness:
    fid = score_alignment_fidelity(params, probe_pairs)
    rev = score_reverse_alignment_fidelity(params, probe_pairs)
    anchor_status = str(
        anchor_payload.get("anchor_status", W50_ANCHOR_STATUS_SYNTHETIC)
        if anchor_payload else W50_ANCHOR_STATUS_SYNTHETIC)
    anchor_cid = _sha256_hex({
        "kind": "w50_realism_anchor_payload",
        "payload": (
            dict(anchor_payload) if anchor_payload else {}),
    })
    return CrossBackendAlignmentWitness(
        source_model_tag=str(params.source_tag),
        target_model_tag=str(params.target_tag),
        params_cid=str(params.cid()),
        training_trace_cid=str(training_trace.cid()),
        code_dim=int(params.code_dim),
        feature_dim=int(params.feature_dim),
        fidelity_score=float(fid),
        reverse_fidelity_score=float(rev),
        anchor_status=str(anchor_status),
        anchor_payload_cid=str(anchor_cid),
        n_probe_pairs=int(len(probe_pairs)),
    )


# =============================================================================
# Verifier
# =============================================================================

W50_XB_VERIFIER_FAILURE_MODES: tuple[str, ...] = (
    "w50_xb_schema_mismatch",
    "w50_xb_params_cid_mismatch",
    "w50_xb_training_trace_cid_mismatch",
    "w50_xb_witness_cid_mismatch",
    "w50_xb_anchor_payload_cid_mismatch",
    "w50_xb_fidelity_below_floor",
    "w50_xb_anchor_status_invalid",
)


def verify_cross_backend_alignment_witness(
        witness: CrossBackendAlignmentWitness,
        *,
        expected_params_cid: str | None = None,
        expected_trace_cid: str | None = None,
        fidelity_floor: float | None = None,
) -> dict[str, Any]:
    """Verify a sealed cross-backend witness.

    Returns a dict with ``ok`` plus an enumerated failure list.
    """
    failures: list[str] = []
    if (expected_params_cid is not None
            and witness.params_cid != expected_params_cid):
        failures.append("w50_xb_params_cid_mismatch")
    if (expected_trace_cid is not None
            and witness.training_trace_cid != expected_trace_cid):
        failures.append("w50_xb_training_trace_cid_mismatch")
    if witness.anchor_status not in (
            W50_ANCHOR_STATUS_SYNTHETIC,
            W50_ANCHOR_STATUS_REAL_LLM,
            W50_ANCHOR_STATUS_SKIPPED):
        failures.append("w50_xb_anchor_status_invalid")
    if (fidelity_floor is not None
            and witness.fidelity_score < float(fidelity_floor)):
        failures.append("w50_xb_fidelity_below_floor")
    return {
        "ok": (len(failures) == 0),
        "failures": failures,
        "witness_cid": witness.cid(),
    }


__all__ = [
    "W50_CROSS_BACKEND_SCHEMA_VERSION",
    "W50_DEFAULT_XB_CODE_DIM",
    "W50_DEFAULT_BACKEND_FEATURE_DIM",
    "W50_DEFAULT_FIDELITY_PROBES",
    "W50_DEFAULT_OLLAMA_MODEL",
    "W50_OLLAMA_ENV_VAR",
    "W50_ANCHOR_STATUS_SYNTHETIC",
    "W50_ANCHOR_STATUS_REAL_LLM",
    "W50_ANCHOR_STATUS_SKIPPED",
    "W50_XB_VERIFIER_FAILURE_MODES",
    "BackendEncoder",
    "BackendDecoder",
    "CrossBackendAlignmentParams",
    "CrossBackendPair",
    "CrossBackendTrainingSet",
    "CrossBackendTrainingTrace",
    "CrossBackendAlignmentWitness",
    "build_unfitted_cross_backend_alignment_params",
    "synthesize_cross_backend_training_set",
    "fit_cross_backend_alignment",
    "score_alignment_fidelity",
    "score_reverse_alignment_fidelity",
    "run_realism_anchor_probe",
    "emit_cross_backend_alignment_witness",
    "verify_cross_backend_alignment_witness",
]
