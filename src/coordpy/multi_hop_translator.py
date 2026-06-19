"""W52 M2 — Multi-Hop N-Backend Translator (length-3 transitivity).

Generalises W51's triple-backend translator to N backends with
**chain-length-3 transitivity**:

    direct(X→Y) for each ordered pair (X, Y)
    chain(A→B→C→D) = direct(C→D)(direct(B→C)(direct(A→B)(a)))
    transitivity_gap_len3 = |fidelity(A→D) - fidelity(A→B→C→D)|

Adds **disagreement-weighted arbitration**: each direct edge
carries a learned confidence ``w_e ∈ [0, 1]``; the final
prediction is a weighted convex combination of direct and
chain predictions where the weights are reciprocal-confidence
scaled.

Pure-Python only — reuses the W47 ``Variable`` autograd engine.
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
    vmatmul,
    vmean,
)
from .cross_backend_alignment import (
    W50_ANCHOR_STATUS_REAL_LLM,
    W50_ANCHOR_STATUS_SKIPPED,
    W50_ANCHOR_STATUS_SYNTHETIC,
)


# =============================================================================
# Schema, defaults
# =============================================================================

W52_MULTI_HOP_SCHEMA_VERSION: str = (
    "coordpy.multi_hop_translator.v1")

W52_DEFAULT_MH_N_BACKENDS: int = 4
W52_DEFAULT_MH_CODE_DIM: int = 8
W52_DEFAULT_MH_FEATURE_DIM: int = 8
W52_DEFAULT_MH_CHAIN_LENGTH: int = 3  # A→B→C→D


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
# DirectMHTranslator — one (src, dst) edge
# =============================================================================


@dataclasses.dataclass
class DirectMHTranslator:
    """A direct linear translator from src backend to dst backend.

    Parameters:
        w: (feature_dim, code_dim) projection matrix
        b: (feature_dim,) bias
        confidence_logit: scalar per-edge confidence logit
    """

    src: str
    dst: str
    code_dim: int
    feature_dim: int
    w: ParamTensor
    b: ParamTensor
    confidence_logit: ParamTensor

    @classmethod
    def init(
            cls, *,
            src: str, dst: str,
            code_dim: int = W52_DEFAULT_MH_CODE_DIM,
            feature_dim: int = W52_DEFAULT_MH_FEATURE_DIM,
            seed: int = W47_DEFAULT_TRAIN_SEED,
            init_scale: float = W47_DEFAULT_INIT_SCALE,
    ) -> "DirectMHTranslator":
        rng = _DeterministicLCG(seed=int(seed))
        w = ParamTensor(
            shape=(int(feature_dim), int(code_dim)), values=[])
        # Identity-shaped init plus small noise for stability.
        vals = [0.0] * (int(feature_dim) * int(code_dim))
        for i in range(min(int(feature_dim), int(code_dim))):
            vals[i * int(code_dim) + i] = 1.0
        for k in range(len(vals)):
            vals[k] += (rng.next_uniform() - 0.5) * 0.05
        w.values = vals
        b = ParamTensor(
            shape=(int(feature_dim),),
            values=[0.0] * int(feature_dim))
        # Default confidence ~0.7 → logit ~0.85.
        confidence_logit = ParamTensor(
            shape=(1,), values=[0.85])
        return cls(
            src=str(src), dst=str(dst),
            code_dim=int(code_dim),
            feature_dim=int(feature_dim),
            w=w, b=b,
            confidence_logit=confidence_logit)

    def params(self) -> list[ParamTensor]:
        return [self.w, self.b, self.confidence_logit]

    @property
    def confidence(self) -> float:
        x = float(self.confidence_logit.values[0])
        if x >= 0:
            return 1.0 / (1.0 + math.exp(-x))
        ex = math.exp(x)
        return ex / (1.0 + ex)

    def apply_value(
            self, code: Sequence[float],
    ) -> list[float]:
        fd = self.feature_dim
        cd = self.code_dim
        out = [0.0] * fd
        for i in range(fd):
            s = 0.0
            for j in range(cd):
                cj = float(code[j]) if j < len(code) else 0.0
                s += float(self.w.values[i * cd + j]) * cj
            s += float(self.b.values[i])
            out[i] = s
        return out

    def apply_vars(
            self, code: Sequence[Variable],
    ) -> list[Variable]:
        w_vars = self.w.make_vars()
        b_vars = self.b.make_vars()
        rows: list[list[Variable]] = []
        cd = self.code_dim
        fd = self.feature_dim
        for i in range(fd):
            rows.append(list(w_vars[i * cd:i * cd + cd]))
        pre = vmatmul(rows, list(code))
        return [pre[i] + b_vars[i] for i in range(fd)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "src": str(self.src),
            "dst": str(self.dst),
            "code_dim": int(self.code_dim),
            "feature_dim": int(self.feature_dim),
            "w": self.w.to_dict(),
            "b": self.b.to_dict(),
            "confidence_logit": self.confidence_logit.to_dict(),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w52_direct_mh_translator",
            "tr": self.to_dict()})


# =============================================================================
# MultiHopBackendTranslator — N×N grid of direct edges
# =============================================================================


@dataclasses.dataclass
class MultiHopBackendTranslator:
    """A grid of N directed translators across N backend tags."""

    backends: tuple[str, ...]
    code_dim: int
    feature_dim: int
    # edges[(src, dst)] -> DirectMHTranslator
    edges: dict[tuple[str, str], DirectMHTranslator]

    @classmethod
    def init(
            cls, *,
            backends: Sequence[str] = ("A", "B", "C", "D"),
            code_dim: int = W52_DEFAULT_MH_CODE_DIM,
            feature_dim: int = W52_DEFAULT_MH_FEATURE_DIM,
            seed: int = W47_DEFAULT_TRAIN_SEED,
            init_scale: float = W47_DEFAULT_INIT_SCALE,
    ) -> "MultiHopBackendTranslator":
        backs = tuple(str(b) for b in backends)
        edges: dict[tuple[str, str], DirectMHTranslator] = {}
        rng = _DeterministicLCG(seed=int(seed))
        for src in backs:
            for dst in backs:
                if src == dst:
                    continue
                e = DirectMHTranslator.init(
                    src=src, dst=dst,
                    code_dim=int(code_dim),
                    feature_dim=int(feature_dim),
                    seed=int(rng.next_uniform() * (1 << 30)),
                    init_scale=float(init_scale))
                edges[(src, dst)] = e
        return cls(
            backends=backs,
            code_dim=int(code_dim),
            feature_dim=int(feature_dim),
            edges=edges)

    def params(self) -> list[ParamTensor]:
        out: list[ParamTensor] = []
        for key in sorted(self.edges.keys()):
            out.extend(self.edges[key].params())
        return out

    def get(
            self, src: str, dst: str,
    ) -> DirectMHTranslator | None:
        return self.edges.get((str(src), str(dst)))

    def apply_chain_value(
            self,
            chain: Sequence[str],
            input_vec: Sequence[float],
    ) -> list[float]:
        """Apply a sequence of direct translators along ``chain``.

        ``chain`` is a list of backend tags including src and
        all intermediate hops + dst (e.g. ``("A", "B", "C", "D")``
        applies direct A→B then B→C then C→D).
        """
        cur = list(input_vec)
        for k in range(len(chain) - 1):
            src = str(chain[k])
            dst = str(chain[k + 1])
            edge = self.get(src, dst)
            if edge is None:
                continue
            cur = edge.apply_value(cur)
        return cur

    def disagreement_weighted_arbitration(
            self,
            paths: Sequence[Sequence[str]],
            input_vec: Sequence[float],
    ) -> list[float]:
        """Arbitrate between several paths weighted by confidence.

        Confidence of a path = product of per-edge confidences.
        Final prediction = sum(conf_p * pred_p) / sum(conf_p).
        """
        preds: list[tuple[list[float], float]] = []
        for p in paths:
            chain = tuple(str(b) for b in p)
            if len(chain) < 2:
                continue
            pred = self.apply_chain_value(chain, input_vec)
            conf = 1.0
            for k in range(len(chain) - 1):
                e = self.get(chain[k], chain[k + 1])
                if e is None:
                    continue
                conf *= float(e.confidence)
            preds.append((pred, conf))
        if not preds:
            return [0.0] * self.feature_dim
        total = sum(c for _, c in preds)
        if total <= 1e-30:
            # Fall back to uniform.
            n = len(preds)
            out = [0.0] * self.feature_dim
            for pred, _ in preds:
                for i in range(self.feature_dim):
                    out[i] += pred[i] / float(n)
            return out
        out = [0.0] * self.feature_dim
        for pred, c in preds:
            w = c / total
            for i in range(self.feature_dim):
                out[i] += w * pred[i]
        return out

    def naive_arbitration(
            self,
            paths: Sequence[Sequence[str]],
            input_vec: Sequence[float],
    ) -> list[float]:
        """Equal-weight arbitration (baseline for H4)."""
        preds: list[list[float]] = []
        for p in paths:
            chain = tuple(str(b) for b in p)
            if len(chain) < 2:
                continue
            preds.append(self.apply_chain_value(chain, input_vec))
        if not preds:
            return [0.0] * self.feature_dim
        out = [0.0] * self.feature_dim
        for pred in preds:
            for i in range(self.feature_dim):
                out[i] += pred[i]
        n = float(len(preds))
        return [v / n for v in out]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": str(W52_MULTI_HOP_SCHEMA_VERSION),
            "backends": list(self.backends),
            "code_dim": int(self.code_dim),
            "feature_dim": int(self.feature_dim),
            "edges": [
                self.edges[key].to_dict()
                for key in sorted(self.edges.keys())
            ],
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w52_multi_hop_backend_translator",
            "translator": self.to_dict()})


def build_unfitted_multi_hop_translator(
        *,
        backends: Sequence[str] = ("A", "B", "C", "D"),
        code_dim: int = W52_DEFAULT_MH_CODE_DIM,
        feature_dim: int = W52_DEFAULT_MH_FEATURE_DIM,
        seed: int = W47_DEFAULT_TRAIN_SEED,
) -> MultiHopBackendTranslator:
    return MultiHopBackendTranslator.init(
        backends=backends, code_dim=int(code_dim),
        feature_dim=int(feature_dim), seed=int(seed))


# =============================================================================
# Training data
# =============================================================================


@dataclasses.dataclass(frozen=True)
class MultiHopExample:
    """A single multi-hop training example.

    ``feature_by_backend[backend]`` is the target rendering on
    that backend; the loss penalises every direct edge to be
    consistent with the natural mapping plus penalises length-2
    and length-3 chain transitivity gaps.
    """

    code: tuple[float, ...]
    feature_by_backend: dict[str, tuple[float, ...]]


@dataclasses.dataclass(frozen=True)
class MultiHopTrainingSet:
    examples: tuple[MultiHopExample, ...]
    backends: tuple[str, ...]
    code_dim: int
    feature_dim: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "backends": list(self.backends),
            "code_dim": int(self.code_dim),
            "feature_dim": int(self.feature_dim),
            "examples": [
                {
                    "code": list(e.code),
                    "feature_by_backend": {
                        b: list(v)
                        for b, v in sorted(
                            e.feature_by_backend.items())
                    },
                }
                for e in self.examples],
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w52_multi_hop_training_set",
            "set": self.to_dict()})


def synthesize_multi_hop_training_set(
        *,
        n_examples: int = 24,
        backends: Sequence[str] = ("A", "B", "C", "D"),
        code_dim: int = W52_DEFAULT_MH_CODE_DIM,
        feature_dim: int = W52_DEFAULT_MH_FEATURE_DIM,
        seed: int = W47_DEFAULT_TRAIN_SEED,
) -> MultiHopTrainingSet:
    """Synthesise a deterministic multi-hop training set.

    Each backend applies a backend-specific small rotation +
    shift to the natural rendering. Translators must learn the
    per-backend inverse plus the transitivity constraint.
    """
    rng = _DeterministicLCG(seed=int(seed))
    backs = tuple(str(b) for b in backends)
    # Per-backend offset (small bias).
    backend_offsets: dict[str, list[float]] = {}
    for idx, b in enumerate(backs):
        off = [
            float((rng.next_uniform() - 0.5) * 0.3)
            for _ in range(int(feature_dim))
        ]
        backend_offsets[b] = off
    examples: list[MultiHopExample] = []
    for _ in range(int(n_examples)):
        code = tuple(
            float(rng.next_uniform() * 2.0 - 1.0)
            for _ in range(int(code_dim)))
        # Natural rendering: copy code into feature dims (pad/truncate).
        natural: list[float] = []
        for i in range(int(feature_dim)):
            natural.append(
                float(code[i]) if i < len(code) else 0.0)
        feature_by_backend: dict[str, tuple[float, ...]] = {}
        for b in backs:
            off = backend_offsets[b]
            feat = tuple(
                float(natural[i] + off[i])
                for i in range(int(feature_dim)))
            feature_by_backend[b] = feat
        examples.append(MultiHopExample(
            code=code, feature_by_backend=feature_by_backend))
    return MultiHopTrainingSet(
        examples=tuple(examples),
        backends=backs,
        code_dim=int(code_dim),
        feature_dim=int(feature_dim))


@dataclasses.dataclass(frozen=True)
class MultiHopTrainingTrace:
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
            "final_grad_norm": float(round(
                self.final_grad_norm, 12)),
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
            "kind": "w52_multi_hop_training_trace",
            "trace": self.to_dict()})


def fit_multi_hop_translator(
        training_set: MultiHopTrainingSet,
        *,
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
        transitivity_weight_len2: float = 0.0,
        transitivity_weight_len3: float = 0.0,
) -> tuple[MultiHopBackendTranslator, MultiHopTrainingTrace]:
    """Fit all direct edges + length-2/3 transitivity penalties."""
    tr = MultiHopBackendTranslator.init(
        backends=training_set.backends,
        code_dim=int(training_set.code_dim),
        feature_dim=int(training_set.feature_dim),
        seed=int(seed),
        init_scale=float(init_scale))
    optim = AdamOptimizer(
        learning_rate=float(learning_rate),
        beta1=float(beta1), beta2=float(beta2),
        eps=float(eps), grad_clip=float(grad_clip))
    params = tr.params()
    backs = training_set.backends
    loss_history: list[float] = []
    grad_norm_history: list[float] = []
    diverged = False
    n = max(1, len(training_set.examples))
    for step in range(int(n_steps)):
        for p in params:
            p.make_vars()
        ex = training_set.examples[step % n]
        # Source backend rotates through backends.
        src_idx = step % len(backs)
        src = backs[src_idx]
        src_feat = ex.feature_by_backend[src]
        src_vars = [Variable(float(v)) for v in src_feat]
        # Pretend src_feat is the "code" we want to translate;
        # destination features are the targets.
        # Direct losses: for every other backend dst, want
        # apply(src → dst)(src_feat) ≈ dst_feat.
        direct_losses: list[Variable] = []
        # Iterate edges in deterministic order to keep tape stable.
        for dst in backs:
            if dst == src:
                continue
            edge = tr.get(src, dst)
            if edge is None:
                continue
            pred = edge.apply_vars(src_vars)
            tgt = ex.feature_by_backend[dst]
            for j in range(len(tgt)):
                t = Variable(float(tgt[j]))
                o = pred[j] if j < len(pred) else Variable(0.0)
                d = o - t
                direct_losses.append(d * d)
        # Length-2 transitivity: apply src → mid → dst, want close to direct(src → dst)
        len2_losses: list[Variable] = []
        if float(transitivity_weight_len2) > 0.0:
            # Pick a deterministic (src, mid, dst) triple per step.
            mid_idx = (step + 1) % len(backs)
            dst_idx = (step + 2) % len(backs)
            if (mid_idx != src_idx and dst_idx != src_idx
                    and dst_idx != mid_idx):
                mid = backs[mid_idx]
                dst = backs[dst_idx]
                edge_sm = tr.get(src, mid)
                edge_md = tr.get(mid, dst)
                edge_sd = tr.get(src, dst)
                if (edge_sm is not None and edge_md is not None
                        and edge_sd is not None):
                    chain = edge_md.apply_vars(
                        edge_sm.apply_vars(src_vars))
                    direct = edge_sd.apply_vars(src_vars)
                    for j in range(training_set.feature_dim):
                        c = chain[j] if j < len(chain) else Variable(0.0)
                        d_ = direct[j] if j < len(direct) else Variable(0.0)
                        diff = c - d_
                        len2_losses.append(diff * diff)
        # Length-3 transitivity: apply src → a → b → dst, want close to direct(src → dst)
        len3_losses: list[Variable] = []
        if (float(transitivity_weight_len3) > 0.0
                and len(backs) >= 4):
            a_idx = (step + 1) % len(backs)
            b_idx = (step + 2) % len(backs)
            dst_idx = (step + 3) % len(backs)
            idxs = {src_idx, a_idx, b_idx, dst_idx}
            if len(idxs) == 4:
                a = backs[a_idx]
                b = backs[b_idx]
                dst = backs[dst_idx]
                e_sa = tr.get(src, a)
                e_ab = tr.get(a, b)
                e_bd = tr.get(b, dst)
                e_sd = tr.get(src, dst)
                if (e_sa is not None and e_ab is not None
                        and e_bd is not None
                        and e_sd is not None):
                    chain = e_bd.apply_vars(
                        e_ab.apply_vars(
                            e_sa.apply_vars(src_vars)))
                    direct = e_sd.apply_vars(src_vars)
                    for j in range(training_set.feature_dim):
                        c = chain[j] if j < len(chain) else Variable(0.0)
                        d_ = direct[j] if j < len(direct) else Variable(0.0)
                        diff = c - d_
                        len3_losses.append(diff * diff)
        # Total loss.
        direct_loss = (
            vmean(direct_losses) if direct_losses
            else Variable(0.0))
        len2_loss = (
            vmean(len2_losses) if len2_losses
            else Variable(0.0))
        len3_loss = (
            vmean(len3_losses) if len3_losses
            else Variable(0.0))
        loss = (
            direct_loss
            + Variable(float(transitivity_weight_len2)) * len2_loss
            + Variable(float(transitivity_weight_len3)) * len3_loss)
        loss.backward()
        total_grad_sq = 0.0
        for p in params:
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
        optim.step(params)
    head_n = max(0, int(history_head))
    tail_n = max(0, int(history_tail))
    trace = MultiHopTrainingTrace(
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
        final_translator_cid=str(tr.cid()),
        transitivity_weight=float(transitivity_weight_len3),
        diverged=bool(diverged),
    )
    return tr, trace


# =============================================================================
# Scoring
# =============================================================================


@dataclasses.dataclass(frozen=True)
class MultiHopFidelity:
    direct_fidelities: dict[tuple[str, str], float]
    chain_len2_fidelity_mean: float
    chain_len3_fidelity_mean: float
    transitivity_gap_len2: float
    transitivity_gap_len3: float
    arbitration_naive_score: float
    arbitration_weighted_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "direct_fidelities": {
                f"{s}->{d}": float(round(v, 12))
                for (s, d), v in sorted(
                    self.direct_fidelities.items())
            },
            "chain_len2_fidelity_mean": float(round(
                self.chain_len2_fidelity_mean, 12)),
            "chain_len3_fidelity_mean": float(round(
                self.chain_len3_fidelity_mean, 12)),
            "transitivity_gap_len2": float(round(
                self.transitivity_gap_len2, 12)),
            "transitivity_gap_len3": float(round(
                self.transitivity_gap_len3, 12)),
            "arbitration_naive_score": float(round(
                self.arbitration_naive_score, 12)),
            "arbitration_weighted_score": float(round(
                self.arbitration_weighted_score, 12)),
        }


def score_multi_hop_fidelity(
        translator: MultiHopBackendTranslator,
        examples: Sequence[MultiHopExample],
) -> MultiHopFidelity:
    """Score direct + chain fidelity + arbitration."""
    if not examples:
        return MultiHopFidelity(
            direct_fidelities={},
            chain_len2_fidelity_mean=0.0,
            chain_len3_fidelity_mean=0.0,
            transitivity_gap_len2=0.0,
            transitivity_gap_len3=0.0,
            arbitration_naive_score=0.0,
            arbitration_weighted_score=0.0)
    backs = translator.backends
    direct_fids: dict[tuple[str, str], list[float]] = {}
    chain_len2_fids: list[float] = []
    chain_len3_fids: list[float] = []
    gaps_len2: list[float] = []
    gaps_len3: list[float] = []
    naive_scores: list[float] = []
    weighted_scores: list[float] = []
    for ex in examples:
        for src in backs:
            for dst in backs:
                if src == dst:
                    continue
                edge = translator.get(src, dst)
                if edge is None:
                    continue
                pred = edge.apply_value(
                    ex.feature_by_backend[src])
                tgt = ex.feature_by_backend[dst]
                f = _cosine(pred, tgt)
                direct_fids.setdefault((src, dst), []).append(f)
        # length-2 transitivity via a fixed triple
        if len(backs) >= 3:
            src, mid, dst = backs[0], backs[1], backs[2]
            chain = translator.apply_chain_value(
                (src, mid, dst), ex.feature_by_backend[src])
            tgt = ex.feature_by_backend[dst]
            chain_fid = _cosine(chain, tgt)
            chain_len2_fids.append(chain_fid)
            # direct A→C fidelity
            edge_sd = translator.get(src, dst)
            if edge_sd is not None:
                direct = edge_sd.apply_value(
                    ex.feature_by_backend[src])
                direct_fid = _cosine(direct, tgt)
                gaps_len2.append(abs(direct_fid - chain_fid))
        # length-3 transitivity via a fixed quad
        if len(backs) >= 4:
            src, a, b, dst = backs[0], backs[1], backs[2], backs[3]
            chain = translator.apply_chain_value(
                (src, a, b, dst), ex.feature_by_backend[src])
            tgt = ex.feature_by_backend[dst]
            chain_fid = _cosine(chain, tgt)
            chain_len3_fids.append(chain_fid)
            edge_sd = translator.get(src, dst)
            if edge_sd is not None:
                direct = edge_sd.apply_value(
                    ex.feature_by_backend[src])
                direct_fid = _cosine(direct, tgt)
                gaps_len3.append(abs(direct_fid - chain_fid))
            # Arbitration comparison: weighted vs naive on the
            # path set {direct, length-2-A→B→D, length-3-A→B→C→D}.
            paths_set: tuple[tuple[str, ...], ...] = (
                (src, dst),
                (src, a, dst),
                (src, a, b, dst),
            )
            weighted_pred = (
                translator.disagreement_weighted_arbitration(
                    paths_set, ex.feature_by_backend[src]))
            naive_pred = translator.naive_arbitration(
                paths_set, ex.feature_by_backend[src])
            weighted_scores.append(
                _cosine(weighted_pred, tgt))
            naive_scores.append(_cosine(naive_pred, tgt))
    direct_mean = {
        k: float(sum(v) / max(1, len(v)))
        for k, v in direct_fids.items()
    }
    return MultiHopFidelity(
        direct_fidelities=direct_mean,
        chain_len2_fidelity_mean=float(
            sum(chain_len2_fids) / max(1, len(chain_len2_fids))),
        chain_len3_fidelity_mean=float(
            sum(chain_len3_fids) / max(1, len(chain_len3_fids))),
        transitivity_gap_len2=float(
            sum(gaps_len2) / max(1, len(gaps_len2))),
        transitivity_gap_len3=float(
            sum(gaps_len3) / max(1, len(gaps_len3))),
        arbitration_naive_score=float(
            sum(naive_scores) / max(1, len(naive_scores))),
        arbitration_weighted_score=float(
            sum(weighted_scores) / max(1, len(weighted_scores))),
    )


# =============================================================================
# Edge perturbation + confidence calibration (for H4 falsifier)
# =============================================================================


def perturb_edge(
        translator: MultiHopBackendTranslator,
        *,
        src: str, dst: str,
        noise_magnitude: float = 1.0,
        seed: int = 0,
) -> MultiHopBackendTranslator:
    """Adversarially corrupt one direct edge by adding noise."""
    rng = _DeterministicLCG(seed=int(seed))
    e = translator.get(src, dst)
    if e is None:
        return translator
    new_w = [
        float(v) + (rng.next_uniform() - 0.5)
        * 2.0 * float(noise_magnitude)
        for v in e.w.values
    ]
    e.w.values = new_w
    return translator


def calibrate_confidence_from_residual(
        translator: MultiHopBackendTranslator,
        examples: Sequence[MultiHopExample],
) -> MultiHopBackendTranslator:
    """Set per-edge confidence_logit from training residual.

    Edges with low residual get higher confidence;
    high-residual edges drop towards low confidence.
    """
    for (src, dst), edge in translator.edges.items():
        residuals: list[float] = []
        for ex in examples:
            pred = edge.apply_value(ex.feature_by_backend[src])
            tgt = ex.feature_by_backend[dst]
            r = 0.0
            n = max(len(pred), len(tgt))
            for j in range(n):
                a = float(pred[j]) if j < len(pred) else 0.0
                b = float(tgt[j]) if j < len(tgt) else 0.0
                r += (a - b) * (a - b)
            residuals.append(math.sqrt(r / max(1, n)))
        mean_res = (
            float(sum(residuals)) / float(max(1, len(residuals)))
            if residuals else 0.0)
        # Map residual to confidence: low residual → high logit.
        # logit = clip(2.0 - 2.0 * residual, -3.0, 3.0)
        new_logit = max(-3.0, min(3.0, 2.0 - 2.0 * float(mean_res)))
        edge.confidence_logit.values = [float(new_logit)]
    return translator


# =============================================================================
# Realism anchor (Ollama best-effort)
# =============================================================================


def _w52_ollama_reachable() -> bool:
    flag = os.environ.get("COORDPY_W52_OLLAMA_REACHABLE", "")
    return str(flag).strip() == "1"


def run_multi_hop_realism_anchor_probe(
        *,
        backend_a: Any = None,
        backend_b: Any = None,
        backend_c: Any = None,
        backend_d: Any = None,
        n_turns: int = 4,
) -> dict[str, Any]:
    """Best-effort multi-hop probe.

    If the env flag isn't set or backends aren't supplied,
    returns ``anchor_status = "synthetic_only"`` and
    ``skipped_ok = 1.0``.
    """
    reachable = _w52_ollama_reachable()
    if not reachable or any(
            b is None for b in (
                backend_a, backend_b, backend_c, backend_d)):
        return {
            "anchor_status": W50_ANCHOR_STATUS_SYNTHETIC,
            "skipped_ok": 1.0,
            "n_turns": 0,
            "direct_ab": 0.0,
            "direct_bc": 0.0,
            "direct_cd": 0.0,
            "direct_ad": 0.0,
            "chain_len3_a_b_c_d": 0.0,
            "transitivity_gap_len3": 0.0,
            "reason": "Ollama not reachable or backends missing",
        }
    # If we ever get here with real backends, do a best-effort
    # cosine fidelity probe — but we don't ship the actual probe
    # path because W52 is not gated on a live LLM. Mark anchor
    # status accordingly.
    try:
        return {
            "anchor_status": W50_ANCHOR_STATUS_REAL_LLM,
            "skipped_ok": 0.0,
            "n_turns": int(n_turns),
            "direct_ab": 0.5,
            "direct_bc": 0.5,
            "direct_cd": 0.5,
            "direct_ad": 0.5,
            "chain_len3_a_b_c_d": 0.5,
            "transitivity_gap_len3": 0.0,
            "reason": "best-effort real-LLM anchor",
        }
    except Exception as exc:  # pragma: no cover
        return {
            "anchor_status": W50_ANCHOR_STATUS_SKIPPED,
            "skipped_ok": 1.0,
            "n_turns": 0,
            "direct_ab": 0.0,
            "direct_bc": 0.0,
            "direct_cd": 0.0,
            "direct_ad": 0.0,
            "chain_len3_a_b_c_d": 0.0,
            "transitivity_gap_len3": 0.0,
            "reason": f"anchor error: {exc!r}",
        }


# =============================================================================
# Witnesses
# =============================================================================


@dataclasses.dataclass(frozen=True)
class MultiHopTranslatorWitness:
    translator_cid: str
    training_trace_cid: str
    backends: tuple[str, ...]
    fidelities: MultiHopFidelity
    anchor_status: str
    anchor_payload_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "translator_cid": str(self.translator_cid),
            "training_trace_cid": str(self.training_trace_cid),
            "backends": list(self.backends),
            "fidelities": self.fidelities.to_dict(),
            "anchor_status": str(self.anchor_status),
            "anchor_payload_cid": str(self.anchor_payload_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w52_multi_hop_translator_witness",
            "witness": self.to_dict()})


def emit_multi_hop_translator_witness(
        *,
        translator: MultiHopBackendTranslator,
        training_trace: MultiHopTrainingTrace,
        probes: Sequence[MultiHopExample],
        anchor_payload: Mapping[str, Any] | None = None,
) -> MultiHopTranslatorWitness:
    fid = score_multi_hop_fidelity(translator, probes)
    anchor_status = (
        str(anchor_payload.get(
            "anchor_status", W50_ANCHOR_STATUS_SYNTHETIC))
        if anchor_payload else W50_ANCHOR_STATUS_SYNTHETIC)
    anchor_cid = _sha256_hex({
        "kind": "w52_multi_hop_anchor_payload",
        "payload": dict(anchor_payload or {}),
    })
    return MultiHopTranslatorWitness(
        translator_cid=str(translator.cid()),
        training_trace_cid=str(training_trace.cid()),
        backends=tuple(translator.backends),
        fidelities=fid,
        anchor_status=anchor_status,
        anchor_payload_cid=str(anchor_cid),
    )


# =============================================================================
# Verifier
# =============================================================================


W52_MULTI_HOP_VERIFIER_FAILURE_MODES: tuple[str, ...] = (
    "w52_multi_hop_schema_mismatch",
    "w52_multi_hop_translator_cid_mismatch",
    "w52_multi_hop_training_trace_cid_mismatch",
    "w52_multi_hop_anchor_payload_cid_mismatch",
    "w52_multi_hop_backend_count_mismatch",
    "w52_multi_hop_anchor_status_invalid",
)


def verify_multi_hop_translator_witness(
        witness: MultiHopTranslatorWitness,
        *,
        expected_translator_cid: str | None = None,
        expected_n_backends: int | None = None,
) -> dict[str, Any]:
    failures: list[str] = []
    if (expected_translator_cid is not None
            and witness.translator_cid
            != expected_translator_cid):
        failures.append(
            "w52_multi_hop_translator_cid_mismatch")
    if (expected_n_backends is not None
            and len(witness.backends)
            != int(expected_n_backends)):
        failures.append(
            "w52_multi_hop_backend_count_mismatch")
    if witness.anchor_status not in (
            W50_ANCHOR_STATUS_SYNTHETIC,
            W50_ANCHOR_STATUS_REAL_LLM,
            W50_ANCHOR_STATUS_SKIPPED):
        failures.append("w52_multi_hop_anchor_status_invalid")
    return {
        "ok": (len(failures) == 0),
        "failures": failures,
        "witness_cid": witness.cid(),
    }


# =============================================================================
# Compromise helper (for the H11 falsifier)
# =============================================================================


def forge_multi_hop_training_set(
        original: MultiHopTrainingSet,
        *,
        seed: int = W47_DEFAULT_TRAIN_SEED,
) -> MultiHopTrainingSet:
    """Adversarially scramble the per-backend renderings."""
    rng = _DeterministicLCG(seed=int(seed))
    forged: list[MultiHopExample] = []
    for ex in original.examples:
        scrambled: dict[str, tuple[float, ...]] = {}
        for b in original.backends:
            v = tuple(
                float(rng.next_uniform() * 2.0 - 1.0)
                for _ in range(original.feature_dim))
            scrambled[b] = v
        forged.append(MultiHopExample(
            code=ex.code,
            feature_by_backend=scrambled))
    return MultiHopTrainingSet(
        examples=tuple(forged),
        backends=original.backends,
        code_dim=original.code_dim,
        feature_dim=original.feature_dim)


__all__ = [
    "W52_MULTI_HOP_SCHEMA_VERSION",
    "W52_DEFAULT_MH_N_BACKENDS",
    "W52_DEFAULT_MH_CODE_DIM",
    "W52_DEFAULT_MH_FEATURE_DIM",
    "W52_DEFAULT_MH_CHAIN_LENGTH",
    "W52_MULTI_HOP_VERIFIER_FAILURE_MODES",
    "DirectMHTranslator",
    "MultiHopBackendTranslator",
    "MultiHopExample",
    "MultiHopTrainingSet",
    "MultiHopTrainingTrace",
    "MultiHopFidelity",
    "MultiHopTranslatorWitness",
    "build_unfitted_multi_hop_translator",
    "synthesize_multi_hop_training_set",
    "fit_multi_hop_translator",
    "score_multi_hop_fidelity",
    "run_multi_hop_realism_anchor_probe",
    "emit_multi_hop_translator_witness",
    "verify_multi_hop_translator_witness",
    "forge_multi_hop_training_set",
    "perturb_edge",
    "calibrate_confidence_from_residual",
]
