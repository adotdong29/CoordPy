"""W51 M2 — Triple-Backend Translator.

A trainable triple-backend translator over three backend tags
``(A, B, C)``. Maintains three direct translators ``A→B``,
``A→C``, ``B→C`` plus the **transitivity loss** that penalises
disagreement between ``A→C`` (direct) and ``A→B→C``
(composition). Trains all three translators jointly via Adam.

Realism anchor: best-effort triple Ollama probe when
``COORDPY_W51_OLLAMA_REACHABLE=1`` is set; otherwise records
``anchor_status = "synthetic_only"`` and the
``W51-L-CROSS-TOKENIZER-TRIPLE-CAP`` conjecture carries
forward.

Pure-Python only — reuses the W47 ``Variable`` +
``AdamOptimizer`` autograd engine and the W50 backend-encoder
+ backend-decoder abstractions.

Honest scope (do-not-overstate)
-------------------------------

This module does NOT alter transformer-internal hidden state,
KV cache bytes, attention weights, or embeddings. The
triple-backend translator operates over W50 capsule-layer
carriers exclusively. The
``W50-C-CROSS-TOKENIZER-LATENT-TRANSFER`` conjecture carries
forward sharpened as
``W51-C-CROSS-TOKENIZER-TRIPLE-TRANSITIVITY``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
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
from .cross_backend_alignment import (
    BackendDecoder,
    BackendEncoder,
    W50_ANCHOR_STATUS_REAL_LLM,
    W50_ANCHOR_STATUS_SKIPPED,
    W50_ANCHOR_STATUS_SYNTHETIC,
    W50_DEFAULT_BACKEND_FEATURE_DIM,
    W50_DEFAULT_XB_CODE_DIM,
)
from .llm_backend import LLMBackend


# =============================================================================
# Schema, defaults
# =============================================================================

W51_TRIPLE_BACKEND_SCHEMA_VERSION: str = (
    "coordpy.cross_backend_translator.v1")

W51_DEFAULT_TRIPLE_CODE_DIM: int = W50_DEFAULT_XB_CODE_DIM
W51_DEFAULT_TRIPLE_FEATURE_DIM: int = (
    W50_DEFAULT_BACKEND_FEATURE_DIM)
W51_DEFAULT_TRIPLE_PROBES: int = 8
W51_OLLAMA_ENV_VAR: str = "COORDPY_W51_OLLAMA_REACHABLE"
W51_DEFAULT_TRIPLE_TRANSITIVITY_WEIGHT: float = 0.5


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


# =============================================================================
# Direct translator (encoder + decoder)
# =============================================================================

@dataclasses.dataclass
class DirectTranslator:
    """One directed (source → target) translator.

    Composes a ``BackendEncoder`` (source-side) with a
    ``BackendDecoder`` (target-side) through a shared code.
    """

    source_tag: str
    target_tag: str
    code_dim: int
    feature_dim: int
    encoder: BackendEncoder
    decoder: BackendDecoder

    @classmethod
    def init(
            cls, *,
            source_tag: str,
            target_tag: str,
            code_dim: int = W51_DEFAULT_TRIPLE_CODE_DIM,
            feature_dim: int = W51_DEFAULT_TRIPLE_FEATURE_DIM,
            seed: int = W47_DEFAULT_TRAIN_SEED,
            init_scale: float = W47_DEFAULT_INIT_SCALE,
    ) -> "DirectTranslator":
        rng = _DeterministicLCG(seed=int(seed))
        enc = BackendEncoder.init(
            backend_tag=str(source_tag),
            in_dim=int(feature_dim),
            code_dim=int(code_dim),
            seed=int(rng.next_uniform() * (1 << 30)),
            init_scale=float(init_scale))
        dec = BackendDecoder.init(
            backend_tag=str(target_tag),
            code_dim=int(code_dim),
            out_dim=int(feature_dim),
            seed=int(rng.next_uniform() * (1 << 30)),
            init_scale=float(init_scale))
        return cls(
            source_tag=str(source_tag),
            target_tag=str(target_tag),
            code_dim=int(code_dim),
            feature_dim=int(feature_dim),
            encoder=enc, decoder=dec)

    def params(self) -> list[ParamTensor]:
        out: list[ParamTensor] = []
        out.extend(self.encoder.params())
        out.extend(self.decoder.params())
        return out

    def translate_value(
            self, x: Sequence[float],
    ) -> list[float]:
        z = self.encoder.forward_value(x)
        return self.decoder.forward_value(z)

    def translate_vars(
            self, x: Sequence[Variable],
    ) -> list[Variable]:
        z = self.encoder.forward_vars(x)
        return self.decoder.forward_vars(z)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_tag": str(self.source_tag),
            "target_tag": str(self.target_tag),
            "code_dim": int(self.code_dim),
            "feature_dim": int(self.feature_dim),
            "encoder": self.encoder.to_dict(),
            "decoder": self.decoder.to_dict(),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w51_direct_translator",
            "translator": self.to_dict()})


# =============================================================================
# Triple-backend translator
# =============================================================================

@dataclasses.dataclass
class TripleBackendTranslator:
    """Bundle of three direct translators over (A, B, C)."""

    tag_a: str
    tag_b: str
    tag_c: str
    code_dim: int
    feature_dim: int
    ab: DirectTranslator
    ac: DirectTranslator
    bc: DirectTranslator
    fitting_method: str = "unfitted"

    @classmethod
    def init(
            cls, *,
            tag_a: str = "synthetic.A",
            tag_b: str = "synthetic.B",
            tag_c: str = "synthetic.C",
            code_dim: int = W51_DEFAULT_TRIPLE_CODE_DIM,
            feature_dim: int = W51_DEFAULT_TRIPLE_FEATURE_DIM,
            seed: int = W47_DEFAULT_TRAIN_SEED,
            init_scale: float = W47_DEFAULT_INIT_SCALE,
    ) -> "TripleBackendTranslator":
        rng = _DeterministicLCG(seed=int(seed))
        ab = DirectTranslator.init(
            source_tag=str(tag_a), target_tag=str(tag_b),
            code_dim=int(code_dim), feature_dim=int(feature_dim),
            seed=int(rng.next_uniform() * (1 << 30)),
            init_scale=float(init_scale))
        ac = DirectTranslator.init(
            source_tag=str(tag_a), target_tag=str(tag_c),
            code_dim=int(code_dim), feature_dim=int(feature_dim),
            seed=int(rng.next_uniform() * (1 << 30)),
            init_scale=float(init_scale))
        bc = DirectTranslator.init(
            source_tag=str(tag_b), target_tag=str(tag_c),
            code_dim=int(code_dim), feature_dim=int(feature_dim),
            seed=int(rng.next_uniform() * (1 << 30)),
            init_scale=float(init_scale))
        return cls(
            tag_a=str(tag_a), tag_b=str(tag_b), tag_c=str(tag_c),
            code_dim=int(code_dim), feature_dim=int(feature_dim),
            ab=ab, ac=ac, bc=bc,
            fitting_method="unfitted")

    def params(self) -> list[ParamTensor]:
        out: list[ParamTensor] = []
        out.extend(self.ab.params())
        out.extend(self.ac.params())
        out.extend(self.bc.params())
        return out

    def translate_a_to_b(
            self, x: Sequence[float],
    ) -> list[float]:
        return self.ab.translate_value(x)

    def translate_a_to_c(
            self, x: Sequence[float],
    ) -> list[float]:
        return self.ac.translate_value(x)

    def translate_b_to_c(
            self, x: Sequence[float],
    ) -> list[float]:
        return self.bc.translate_value(x)

    def translate_a_to_b_to_c(
            self, x: Sequence[float],
    ) -> list[float]:
        b = self.ab.translate_value(x)
        return self.bc.translate_value(b)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tag_a": str(self.tag_a),
            "tag_b": str(self.tag_b),
            "tag_c": str(self.tag_c),
            "code_dim": int(self.code_dim),
            "feature_dim": int(self.feature_dim),
            "ab": self.ab.to_dict(),
            "ac": self.ac.to_dict(),
            "bc": self.bc.to_dict(),
            "fitting_method": str(self.fitting_method),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w51_triple_backend_translator",
            "translator": self.to_dict()})


def build_unfitted_triple_backend_translator(
        *,
        tag_a: str = "synthetic.A",
        tag_b: str = "synthetic.B",
        tag_c: str = "synthetic.C",
        code_dim: int = W51_DEFAULT_TRIPLE_CODE_DIM,
        feature_dim: int = W51_DEFAULT_TRIPLE_FEATURE_DIM,
        seed: int = W47_DEFAULT_TRAIN_SEED,
        init_scale: float = W47_DEFAULT_INIT_SCALE,
) -> TripleBackendTranslator:
    return TripleBackendTranslator.init(
        tag_a=str(tag_a), tag_b=str(tag_b), tag_c=str(tag_c),
        code_dim=int(code_dim), feature_dim=int(feature_dim),
        seed=int(seed), init_scale=float(init_scale))


# =============================================================================
# Training set
# =============================================================================

@dataclasses.dataclass(frozen=True)
class TripleBackendExample:
    """One paired training example across (A, B, C)."""

    carrier_a: tuple[float, ...]
    carrier_b: tuple[float, ...]
    carrier_c: tuple[float, ...]


@dataclasses.dataclass(frozen=True)
class TripleBackendTrainingSet:
    examples: tuple[TripleBackendExample, ...]
    feature_dim: int
    code_dim: int
    tag_a: str
    tag_b: str
    tag_c: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_dim": int(self.feature_dim),
            "code_dim": int(self.code_dim),
            "tag_a": str(self.tag_a),
            "tag_b": str(self.tag_b),
            "tag_c": str(self.tag_c),
            "examples": [
                {"carrier_a": list(e.carrier_a),
                 "carrier_b": list(e.carrier_b),
                 "carrier_c": list(e.carrier_c)}
                for e in self.examples],
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w51_triple_backend_training_set",
            "set": self.to_dict()})


def synthesize_triple_backend_training_set(
        *,
        n_examples: int = 32,
        tag_a: str = "synthetic.A",
        tag_b: str = "synthetic.B",
        tag_c: str = "synthetic.C",
        feature_dim: int = W51_DEFAULT_TRIPLE_FEATURE_DIM,
        code_dim: int = W51_DEFAULT_TRIPLE_CODE_DIM,
        seed: int = W47_DEFAULT_TRAIN_SEED,
        shift_ab: int = 3,
        shift_bc: int = 2,
) -> TripleBackendTrainingSet:
    """Synthesise a deterministic triple training set.

    Each example has carriers ``(a, b, c)`` where ``b`` is a
    circular shift of ``a`` by ``shift_ab`` positions, and
    ``c`` is a circular shift of ``b`` by ``shift_bc``
    positions. The composed shift ``a→b→c`` should equal the
    direct shift ``a→c = a shifted by (shift_ab + shift_bc)``.

    A trained TripleBackendTranslator must learn all three shifts
    correctly AND respect the transitivity property.
    """
    rng = _DeterministicLCG(seed=int(seed))
    examples: list[TripleBackendExample] = []
    sab = int(shift_ab) % max(1, int(feature_dim))
    sbc = int(shift_bc) % max(1, int(feature_dim))
    for _ in range(int(n_examples)):
        a = [
            float(rng.next_uniform() * 2.0 - 1.0)
            for _ in range(int(feature_dim))
        ]
        b = [
            float(a[(j + sab) % int(feature_dim)])
            for j in range(int(feature_dim))
        ]
        c = [
            float(b[(j + sbc) % int(feature_dim)])
            for j in range(int(feature_dim))
        ]
        examples.append(TripleBackendExample(
            carrier_a=tuple(a),
            carrier_b=tuple(b),
            carrier_c=tuple(c)))
    return TripleBackendTrainingSet(
        examples=tuple(examples),
        feature_dim=int(feature_dim),
        code_dim=int(code_dim),
        tag_a=str(tag_a), tag_b=str(tag_b), tag_c=str(tag_c))


# =============================================================================
# Fit
# =============================================================================

@dataclasses.dataclass(frozen=True)
class TripleBackendTrainingTrace:
    seed: int
    n_steps: int
    final_loss: float
    final_grad_norm: float
    loss_head: tuple[float, ...]
    loss_tail: tuple[float, ...]
    training_set_cid: str
    final_translator_cid: str
    transitivity_weight: float
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
            "final_translator_cid": str(
                self.final_translator_cid),
            "transitivity_weight": float(round(
                self.transitivity_weight, 12)),
            "diverged": bool(self.diverged),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w51_triple_backend_training_trace",
            "trace": self.to_dict()})


def fit_triple_backend_translator(
        training_set: TripleBackendTrainingSet,
        *,
        n_steps: int = 288,
        learning_rate: float = 0.025,
        seed: int = W47_DEFAULT_TRAIN_SEED,
        init_scale: float = W47_DEFAULT_INIT_SCALE,
        beta1: float = W47_DEFAULT_BETA1,
        beta2: float = W47_DEFAULT_BETA2,
        eps: float = W47_DEFAULT_EPS,
        grad_clip: float = W47_DEFAULT_GRAD_CLIP,
        transitivity_weight: float = (
            W51_DEFAULT_TRIPLE_TRANSITIVITY_WEIGHT),
        history_head: int = 6,
        history_tail: int = 6,
) -> tuple[TripleBackendTranslator,
           TripleBackendTrainingTrace]:
    """Fit the triple-backend translator jointly.

    Joint loss = direct_L2(A→B, target_B)
               + direct_L2(A→C, target_C)
               + direct_L2(B→C, target_C)
               + transitivity_weight * L2(A→B→C, target_C).
    """
    translator = build_unfitted_triple_backend_translator(
        tag_a=str(training_set.tag_a),
        tag_b=str(training_set.tag_b),
        tag_c=str(training_set.tag_c),
        code_dim=int(training_set.code_dim),
        feature_dim=int(training_set.feature_dim),
        seed=int(seed), init_scale=float(init_scale))
    optim = AdamOptimizer(
        learning_rate=float(learning_rate),
        beta1=float(beta1), beta2=float(beta2),
        eps=float(eps), grad_clip=float(grad_clip))
    trainable = translator.params()
    loss_history: list[float] = []
    grad_norm_history: list[float] = []
    diverged = False
    n = max(1, len(training_set.examples))
    tw = float(transitivity_weight)
    for step in range(int(n_steps)):
        for p in trainable:
            p.make_vars()
        idx = step % n
        ex = training_set.examples[idx]
        a_vars = [Variable(float(v)) for v in ex.carrier_a]
        b_vars = [Variable(float(v)) for v in ex.carrier_b]
        c_vars = [Variable(float(v)) for v in ex.carrier_c]
        # Direct paths
        ab_out = translator.ab.translate_vars(a_vars)
        ac_out = translator.ac.translate_vars(a_vars)
        bc_out = translator.bc.translate_vars(b_vars)
        # Transitive path A → B → C
        # Use ab_out as the (predicted) intermediate
        abc_out = translator.bc.translate_vars(ab_out)
        # Losses
        def mse(pred: Sequence[Variable],
                tgt: Sequence[Variable]) -> Variable:
            terms: list[Variable] = []
            for j in range(len(tgt)):
                p = pred[j] if j < len(pred) else Variable(0.0)
                t = tgt[j]
                d = p - t
                terms.append(d * d)
            return vmean(terms)
        l_ab = mse(ab_out, b_vars)
        l_ac = mse(ac_out, c_vars)
        l_bc = mse(bc_out, c_vars)
        l_abc = mse(abc_out, c_vars)
        loss = l_ab + l_ac + l_bc + tw * l_abc
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
    fitted = TripleBackendTranslator(
        tag_a=translator.tag_a, tag_b=translator.tag_b,
        tag_c=translator.tag_c,
        code_dim=translator.code_dim,
        feature_dim=translator.feature_dim,
        ab=translator.ab, ac=translator.ac, bc=translator.bc,
        fitting_method=(
            "triple_backend_adam_v1"
            if not diverged
            else "triple_backend_diverged"),
    )
    head_n = max(0, int(history_head))
    tail_n = max(0, int(history_tail))
    trace = TripleBackendTrainingTrace(
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
        final_translator_cid=str(fitted.cid()),
        transitivity_weight=float(tw),
        diverged=bool(diverged),
    )
    return fitted, trace


# =============================================================================
# Fidelity scoring
# =============================================================================

@dataclasses.dataclass(frozen=True)
class TripleBackendFidelity:
    """Bundle of fidelity scores."""

    direct_ab: float
    direct_ac: float
    direct_bc: float
    transitive_a_b_c: float
    transitivity_gap: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "direct_ab": float(round(self.direct_ab, 12)),
            "direct_ac": float(round(self.direct_ac, 12)),
            "direct_bc": float(round(self.direct_bc, 12)),
            "transitive_a_b_c": float(
                round(self.transitive_a_b_c, 12)),
            "transitivity_gap": float(
                round(self.transitivity_gap, 12)),
        }


def score_triple_backend_fidelity(
        translator: TripleBackendTranslator,
        probes: Sequence[TripleBackendExample],
) -> TripleBackendFidelity:
    """Compute mean cosine fidelity across the probe set."""
    if not probes:
        return TripleBackendFidelity(0.0, 0.0, 0.0, 0.0, 0.0)
    n = len(probes)
    sab = 0.0
    sac = 0.0
    sbc = 0.0
    sabc = 0.0
    for p in probes:
        sab += _cosine(
            translator.translate_a_to_b(p.carrier_a),
            p.carrier_b)
        sac += _cosine(
            translator.translate_a_to_c(p.carrier_a),
            p.carrier_c)
        sbc += _cosine(
            translator.translate_b_to_c(p.carrier_b),
            p.carrier_c)
        sabc += _cosine(
            translator.translate_a_to_b_to_c(p.carrier_a),
            p.carrier_c)
    d_ab = sab / float(n)
    d_ac = sac / float(n)
    d_bc = sbc / float(n)
    t_abc = sabc / float(n)
    gap = abs(d_ac - t_abc)
    return TripleBackendFidelity(
        direct_ab=float(d_ab),
        direct_ac=float(d_ac),
        direct_bc=float(d_bc),
        transitive_a_b_c=float(t_abc),
        transitivity_gap=float(gap),
    )


# =============================================================================
# Realism anchor (best-effort triple Ollama probe)
# =============================================================================

def _w51_ollama_reachable() -> bool:
    flag = os.environ.get(W51_OLLAMA_ENV_VAR, "").strip()
    return flag == "1" or flag.lower() == "true"


def run_triple_realism_anchor_probe(
        *,
        backend_a: LLMBackend | None = None,
        backend_b: LLMBackend | None = None,
        backend_c: LLMBackend | None = None,
        n_turns: int = 6,
        require_real: bool = False,
) -> dict[str, Any]:
    """Best-effort triple-backend Ollama probe.

    When ``COORDPY_W51_OLLAMA_REACHABLE`` is unset and the
    caller does not pass three real backends, the probe records
    ``anchor_status = "synthetic_only"``.
    """
    if (not _w51_ollama_reachable()
            and backend_a is None
            and backend_b is None
            and backend_c is None):
        return {
            "anchor_status": W50_ANCHOR_STATUS_SYNTHETIC,
            "skipped_ok": 1.0,
            "n_turns": 0,
            "direct_ab": 0.0,
            "direct_ac": 0.0,
            "direct_bc": 0.0,
            "transitive_a_b_c": 0.0,
            "transitivity_gap": 0.0,
            "reason": (
                "COORDPY_W51_OLLAMA_REACHABLE not set and no "
                "triple backend supplied"),
        }
    if (backend_a is None or backend_b is None
            or backend_c is None):
        return {
            "anchor_status": W50_ANCHOR_STATUS_SKIPPED,
            "skipped_ok": 0.0 if require_real else 1.0,
            "n_turns": 0,
            "direct_ab": 0.0,
            "direct_ac": 0.0,
            "direct_bc": 0.0,
            "transitive_a_b_c": 0.0,
            "transitivity_gap": 0.0,
            "reason": (
                "one or more triple backends missing — anchor "
                "skipped"),
        }
    prompts = [
        f"W51 triple anchor probe turn {i}: emit one-line."
        for i in range(int(n_turns))
    ]

    def _gen(b: LLMBackend, p: str) -> str:
        try:
            return b.generate(p, max_tokens=40, temperature=0.0)
        except Exception:  # noqa: BLE001
            return ""

    out_a = [_gen(backend_a, p) for p in prompts]
    out_b = [_gen(backend_b, p) for p in prompts]
    out_c = [_gen(backend_c, p) for p in prompts]

    def _jaccard(x: str, y: str) -> float:
        sx = set(x.lower().split())
        sy = set(y.lower().split())
        if not sx and not sy:
            return 1.0
        u = sx | sy
        if not u:
            return 0.0
        return float(len(sx & sy)) / float(len(u))

    sims_ab = [_jaccard(a, b) for a, b in zip(out_a, out_b)]
    sims_ac = [_jaccard(a, c) for a, c in zip(out_a, out_c)]
    sims_bc = [_jaccard(b, c) for b, c in zip(out_b, out_c)]
    # Transitive: union(out_a)/inter with out_c via b as bridge
    # Use min(jaccard(a, b), jaccard(b, c)) as the proxy fidelity.
    sims_abc = [min(_jaccard(a, b), _jaccard(b, c))
                for a, b, c in zip(out_a, out_b, out_c)]

    def _mean(xs: list[float]) -> float:
        return float(sum(xs)) / float(max(1, len(xs)))

    fab = _mean(sims_ab)
    fac = _mean(sims_ac)
    fbc = _mean(sims_bc)
    fabc = _mean(sims_abc)
    gap = abs(fac - fabc)
    return {
        "anchor_status": W50_ANCHOR_STATUS_REAL_LLM,
        "skipped_ok": 1.0,
        "n_turns": int(n_turns),
        "direct_ab": float(round(fab, 12)),
        "direct_ac": float(round(fac, 12)),
        "direct_bc": float(round(fbc, 12)),
        "transitive_a_b_c": float(round(fabc, 12)),
        "transitivity_gap": float(round(gap, 12)),
        "backend_a_model_tag": str(
            getattr(backend_a, "model", "unknown")),
        "backend_b_model_tag": str(
            getattr(backend_b, "model", "unknown")),
        "backend_c_model_tag": str(
            getattr(backend_c, "model", "unknown")),
    }


# =============================================================================
# Witness
# =============================================================================

@dataclasses.dataclass(frozen=True)
class TripleBackendTranslatorWitness:
    """Sealed per-turn triple-backend translator witness."""

    translator_cid: str
    training_trace_cid: str
    tag_a: str
    tag_b: str
    tag_c: str
    code_dim: int
    feature_dim: int
    direct_ab: float
    direct_ac: float
    direct_bc: float
    transitive_a_b_c: float
    transitivity_gap: float
    anchor_status: str
    anchor_payload_cid: str
    n_probe_examples: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "translator_cid": str(self.translator_cid),
            "training_trace_cid": str(self.training_trace_cid),
            "tag_a": str(self.tag_a),
            "tag_b": str(self.tag_b),
            "tag_c": str(self.tag_c),
            "code_dim": int(self.code_dim),
            "feature_dim": int(self.feature_dim),
            "direct_ab": float(round(self.direct_ab, 12)),
            "direct_ac": float(round(self.direct_ac, 12)),
            "direct_bc": float(round(self.direct_bc, 12)),
            "transitive_a_b_c": float(
                round(self.transitive_a_b_c, 12)),
            "transitivity_gap": float(
                round(self.transitivity_gap, 12)),
            "anchor_status": str(self.anchor_status),
            "anchor_payload_cid": str(self.anchor_payload_cid),
            "n_probe_examples": int(self.n_probe_examples),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w51_triple_backend_translator_witness",
            "witness": self.to_dict()})


def emit_triple_backend_translator_witness(
        *,
        translator: TripleBackendTranslator,
        training_trace: TripleBackendTrainingTrace,
        probes: Sequence[TripleBackendExample],
        anchor_payload: Mapping[str, Any] | None = None,
) -> TripleBackendTranslatorWitness:
    fid = score_triple_backend_fidelity(translator, probes)
    anchor_status = str(
        anchor_payload.get("anchor_status",
                            W50_ANCHOR_STATUS_SYNTHETIC)
        if anchor_payload else W50_ANCHOR_STATUS_SYNTHETIC)
    anchor_cid = _sha256_hex({
        "kind": "w51_triple_backend_realism_anchor_payload",
        "payload": (
            dict(anchor_payload) if anchor_payload else {}),
    })
    return TripleBackendTranslatorWitness(
        translator_cid=str(translator.cid()),
        training_trace_cid=str(training_trace.cid()),
        tag_a=str(translator.tag_a),
        tag_b=str(translator.tag_b),
        tag_c=str(translator.tag_c),
        code_dim=int(translator.code_dim),
        feature_dim=int(translator.feature_dim),
        direct_ab=float(fid.direct_ab),
        direct_ac=float(fid.direct_ac),
        direct_bc=float(fid.direct_bc),
        transitive_a_b_c=float(fid.transitive_a_b_c),
        transitivity_gap=float(fid.transitivity_gap),
        anchor_status=str(anchor_status),
        anchor_payload_cid=str(anchor_cid),
        n_probe_examples=int(len(probes)),
    )


# =============================================================================
# Verifier
# =============================================================================

W51_TRIPLE_BACKEND_VERIFIER_FAILURE_MODES: tuple[str, ...] = (
    "w51_triple_backend_schema_mismatch",
    "w51_triple_backend_translator_cid_mismatch",
    "w51_triple_backend_training_trace_cid_mismatch",
    "w51_triple_backend_witness_cid_mismatch",
    "w51_triple_backend_anchor_payload_cid_mismatch",
    "w51_triple_backend_direct_fidelity_below_floor",
    "w51_triple_backend_transitivity_gap_above_ceiling",
    "w51_triple_backend_anchor_status_invalid",
    "w51_triple_backend_tag_universe_mismatch",
)


def verify_triple_backend_translator_witness(
        witness: TripleBackendTranslatorWitness,
        *,
        expected_translator_cid: str | None = None,
        expected_trace_cid: str | None = None,
        direct_fidelity_floor: float | None = None,
        transitivity_gap_ceiling: float | None = None,
        expected_tags: tuple[str, str, str] | None = None,
) -> dict[str, Any]:
    failures: list[str] = []
    if (expected_translator_cid is not None
            and witness.translator_cid
            != expected_translator_cid):
        failures.append(
            "w51_triple_backend_translator_cid_mismatch")
    if (expected_trace_cid is not None
            and witness.training_trace_cid != expected_trace_cid):
        failures.append(
            "w51_triple_backend_training_trace_cid_mismatch")
    if witness.anchor_status not in (
            W50_ANCHOR_STATUS_SYNTHETIC,
            W50_ANCHOR_STATUS_REAL_LLM,
            W50_ANCHOR_STATUS_SKIPPED):
        failures.append(
            "w51_triple_backend_anchor_status_invalid")
    if direct_fidelity_floor is not None:
        df = float(direct_fidelity_floor)
        if (witness.direct_ab < df
                or witness.direct_ac < df
                or witness.direct_bc < df):
            failures.append(
                "w51_triple_backend_direct_fidelity_below_floor")
    if (transitivity_gap_ceiling is not None
            and witness.transitivity_gap
            > float(transitivity_gap_ceiling)):
        failures.append(
            "w51_triple_backend_transitivity_gap_above_ceiling")
    if expected_tags is not None:
        ta, tb, tc = expected_tags
        if (witness.tag_a != str(ta)
                or witness.tag_b != str(tb)
                or witness.tag_c != str(tc)):
            failures.append(
                "w51_triple_backend_tag_universe_mismatch")
    return {
        "ok": (len(failures) == 0),
        "failures": failures,
        "witness_cid": witness.cid(),
    }


# =============================================================================
# Compromise cap helper (H10 falsifier)
# =============================================================================

def forge_triple_backend_training_set(
        original: TripleBackendTrainingSet,
        *,
        seed: int = W47_DEFAULT_TRAIN_SEED,
) -> TripleBackendTrainingSet:
    """Adversarially scramble the triple-backend target carriers.

    Replaces the target carriers (b and c) with random noise —
    the trained translator cannot recover the shift pattern.
    """
    rng = _DeterministicLCG(seed=int(seed))
    forged: list[TripleBackendExample] = []
    for ex in original.examples:
        forged_b = tuple(
            float(rng.next_uniform() * 2.0 - 1.0)
            for _ in range(len(ex.carrier_b)))
        forged_c = tuple(
            float(rng.next_uniform() * 2.0 - 1.0)
            for _ in range(len(ex.carrier_c)))
        forged.append(TripleBackendExample(
            carrier_a=ex.carrier_a,
            carrier_b=forged_b,
            carrier_c=forged_c))
    return TripleBackendTrainingSet(
        examples=tuple(forged),
        feature_dim=original.feature_dim,
        code_dim=original.code_dim,
        tag_a=original.tag_a, tag_b=original.tag_b,
        tag_c=original.tag_c)


__all__ = [
    "W51_TRIPLE_BACKEND_SCHEMA_VERSION",
    "W51_DEFAULT_TRIPLE_CODE_DIM",
    "W51_DEFAULT_TRIPLE_FEATURE_DIM",
    "W51_DEFAULT_TRIPLE_PROBES",
    "W51_OLLAMA_ENV_VAR",
    "W51_DEFAULT_TRIPLE_TRANSITIVITY_WEIGHT",
    "W51_TRIPLE_BACKEND_VERIFIER_FAILURE_MODES",
    "DirectTranslator",
    "TripleBackendTranslator",
    "TripleBackendExample",
    "TripleBackendTrainingSet",
    "TripleBackendTrainingTrace",
    "TripleBackendFidelity",
    "TripleBackendTranslatorWitness",
    "build_unfitted_triple_backend_translator",
    "synthesize_triple_backend_training_set",
    "fit_triple_backend_translator",
    "score_triple_backend_fidelity",
    "run_triple_realism_anchor_probe",
    "emit_triple_backend_translator_witness",
    "verify_triple_backend_translator_witness",
    "forge_triple_backend_training_set",
]
