"""W53 M2 — Multi-Hop Translator V3 (5-backend, chain-length-4,
   uncertainty arbitration).

Extends W52 ``MultiHopBackendTranslator`` to:

* **5 backends** ``(A, B, C, D, E)`` — 20 directed edges
* **chain-length-4** scoring (``A→B→C→D→E``) with explicit
  transitivity gap reporting
* **uncertainty-aware arbitration** that returns both a
  prediction AND a per-dim confidence interval derived from
  the weighted disagreement among paths

The new arbitration scheme:

    For path p with prediction y_p and confidence c_p,
    the prediction is the c_p-weighted convex combine of {y_p}.
    The per-dim std-dev is the c_p-weighted std-dev of y_p[i],
    interpreted as a 1-sigma confidence interval.

A high-confidence prediction has small disagreement among the
paths AND high path-confidences. A low-confidence prediction
either has large disagreement or low path-confidences.

Pure-Python only — wraps W52's ``MultiHopBackendTranslator``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Any, Sequence

from .multi_hop_translator import (
    MultiHopBackendTranslator,
    MultiHopExample,
    MultiHopFidelity,
    W52_DEFAULT_MH_CODE_DIM,
    W52_DEFAULT_MH_FEATURE_DIM,
    build_unfitted_multi_hop_translator,
    fit_multi_hop_translator,
    score_multi_hop_fidelity,
    synthesize_multi_hop_training_set,
)
from .autograd_manifold import (
    W47_DEFAULT_TRAIN_SEED,
)


# =============================================================================
# Schema, defaults
# =============================================================================

W53_MH_V3_SCHEMA_VERSION: str = (
    "coordpy.multi_hop_translator_v3.v1")

W53_DEFAULT_MH_V3_BACKENDS: tuple[str, ...] = (
    "A", "B", "C", "D", "E")
W53_DEFAULT_MH_V3_N_BACKENDS: int = 5
W53_DEFAULT_MH_V3_CODE_DIM: int = W52_DEFAULT_MH_CODE_DIM
W53_DEFAULT_MH_V3_FEATURE_DIM: int = W52_DEFAULT_MH_FEATURE_DIM
W53_DEFAULT_MH_V3_CHAIN_LENGTH: int = 4
W53_DEFAULT_MH_V3_TRANSITIVITY_FLOOR: float = 0.20


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
# Uncertainty-aware arbitration
# =============================================================================


@dataclasses.dataclass(frozen=True)
class UncertaintyArbitrationResult:
    """Prediction + per-dim 1-sigma confidence interval."""

    prediction: tuple[float, ...]
    per_dim_std: tuple[float, ...]
    aggregate_confidence: float
    n_paths: int
    paths: tuple[tuple[str, ...], ...]
    path_confidences: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "prediction": list(_round_floats(self.prediction)),
            "per_dim_std": list(_round_floats(
                self.per_dim_std)),
            "aggregate_confidence": float(round(
                self.aggregate_confidence, 12)),
            "n_paths": int(self.n_paths),
            "paths": [list(p) for p in self.paths],
            "path_confidences": list(_round_floats(
                self.path_confidences)),
        }


def uncertainty_aware_arbitration(
        translator: MultiHopBackendTranslator,
        *,
        paths: Sequence[Sequence[str]],
        input_vec: Sequence[float],
        feature_dim: int,
) -> UncertaintyArbitrationResult:
    """Compute weighted prediction + per-dim 1-sigma std.

    aggregate_confidence is the mean path confidence. A small
    aggregate_confidence (<0.2) means none of the paths are
    trustworthy; the caller should fall back to abstention.
    """
    preds: list[list[float]] = []
    confs: list[float] = []
    used_paths: list[tuple[str, ...]] = []
    for p in paths:
        chain = tuple(str(b) for b in p)
        if len(chain) < 2:
            continue
        pred = translator.apply_chain_value(chain, input_vec)
        # Path conf = product of per-edge confidences.
        c = 1.0
        for k in range(len(chain) - 1):
            e = translator.get(chain[k], chain[k + 1])
            if e is None:
                continue
            c *= float(e.confidence)
        preds.append(list(pred))
        confs.append(float(c))
        used_paths.append(chain)
    fd = int(feature_dim)
    if not preds:
        return UncertaintyArbitrationResult(
            prediction=tuple([0.0] * fd),
            per_dim_std=tuple([1.0] * fd),
            aggregate_confidence=0.0,
            n_paths=0,
            paths=(),
            path_confidences=(),
        )
    total = sum(confs)
    if total <= 1e-30:
        n = float(len(preds))
        weighted_pred = [
            sum(preds[k][i] for k in range(int(n))) / n
            for i in range(fd)
        ]
        weights = [1.0 / n for _ in preds]
    else:
        weighted_pred = [0.0] * fd
        for k, pred in enumerate(preds):
            w = float(confs[k]) / float(total)
            for i in range(fd):
                weighted_pred[i] += w * float(
                    pred[i] if i < len(pred) else 0.0)
        weights = [
            float(confs[k]) / float(total)
            for k in range(len(preds))
        ]
    # Per-dim weighted std-dev.
    per_dim_std = [0.0] * fd
    for i in range(fd):
        var_i = 0.0
        for k, pred in enumerate(preds):
            d = float(
                pred[i] if i < len(pred) else 0.0
            ) - float(weighted_pred[i])
            var_i += float(weights[k]) * d * d
        per_dim_std[i] = float(math.sqrt(max(0.0, var_i)))
    agg = (
        float(sum(confs)) / float(max(1, len(confs)))
        if confs else 0.0)
    return UncertaintyArbitrationResult(
        prediction=tuple(_round_floats(weighted_pred)),
        per_dim_std=tuple(_round_floats(per_dim_std)),
        aggregate_confidence=float(agg),
        n_paths=int(len(preds)),
        paths=tuple(used_paths),
        path_confidences=tuple(_round_floats(confs)),
    )


# =============================================================================
# Quint translator + chain-len-4 fidelity
# =============================================================================


@dataclasses.dataclass(frozen=True)
class QuintFidelity:
    """5-backend fidelity report."""

    direct_fidelity_a_e: float
    chain_len2_fid_mean: float
    chain_len3_fid_mean: float
    chain_len4_fid_mean: float
    transitivity_gap_len4: float
    arbitration_uncertainty_mean: float
    arbitration_uncertainty_max: float
    weighted_arbitration_mean: float
    naive_arbitration_mean: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "direct_fidelity_a_e": float(round(
                self.direct_fidelity_a_e, 12)),
            "chain_len2_fid_mean": float(round(
                self.chain_len2_fid_mean, 12)),
            "chain_len3_fid_mean": float(round(
                self.chain_len3_fid_mean, 12)),
            "chain_len4_fid_mean": float(round(
                self.chain_len4_fid_mean, 12)),
            "transitivity_gap_len4": float(round(
                self.transitivity_gap_len4, 12)),
            "arbitration_uncertainty_mean": float(round(
                self.arbitration_uncertainty_mean, 12)),
            "arbitration_uncertainty_max": float(round(
                self.arbitration_uncertainty_max, 12)),
            "weighted_arbitration_mean": float(round(
                self.weighted_arbitration_mean, 12)),
            "naive_arbitration_mean": float(round(
                self.naive_arbitration_mean, 12)),
        }


def score_quint_fidelity(
        translator: MultiHopBackendTranslator,
        examples: Sequence[MultiHopExample],
) -> QuintFidelity:
    """Score 5-backend chain-length-4 fidelity."""
    if not examples or len(translator.backends) < 5:
        return QuintFidelity(
            direct_fidelity_a_e=0.0,
            chain_len2_fid_mean=0.0,
            chain_len3_fid_mean=0.0,
            chain_len4_fid_mean=0.0,
            transitivity_gap_len4=0.0,
            arbitration_uncertainty_mean=0.0,
            arbitration_uncertainty_max=0.0,
            weighted_arbitration_mean=0.0,
            naive_arbitration_mean=0.0,
        )
    src, b, c, d, dst = translator.backends[:5]
    fd = int(translator.feature_dim)
    direct_a_e: list[float] = []
    chain_len2: list[float] = []
    chain_len3: list[float] = []
    chain_len4: list[float] = []
    gaps_len4: list[float] = []
    arb_uncerts: list[float] = []
    weighted_scores: list[float] = []
    naive_scores: list[float] = []
    for ex in examples:
        # Direct A→E.
        edge_se = translator.get(src, dst)
        tgt = ex.feature_by_backend[dst]
        if edge_se is not None:
            pred = edge_se.apply_value(
                ex.feature_by_backend[src])
            direct_a_e.append(_cosine(pred, tgt))
        # Chain length-2: A→B→E
        ch2 = translator.apply_chain_value(
            (src, b, dst), ex.feature_by_backend[src])
        chain_len2.append(_cosine(ch2, tgt))
        # Chain length-3: A→B→C→E
        ch3 = translator.apply_chain_value(
            (src, b, c, dst), ex.feature_by_backend[src])
        chain_len3.append(_cosine(ch3, tgt))
        # Chain length-4: A→B→C→D→E
        ch4 = translator.apply_chain_value(
            (src, b, c, d, dst), ex.feature_by_backend[src])
        chain_len4.append(_cosine(ch4, tgt))
        # Transitivity gap.
        if direct_a_e:
            gaps_len4.append(
                abs(direct_a_e[-1] - chain_len4[-1]))
        # Uncertainty-aware arbitration over 4 paths.
        paths = (
            (src, dst),
            (src, b, dst),
            (src, b, c, dst),
            (src, b, c, d, dst),
        )
        arb = uncertainty_aware_arbitration(
            translator, paths=paths,
            input_vec=ex.feature_by_backend[src],
            feature_dim=fd)
        weighted_scores.append(
            _cosine(arb.prediction, tgt))
        # Naive arbitration.
        naive_pred = translator.naive_arbitration(
            paths, ex.feature_by_backend[src])
        naive_scores.append(_cosine(naive_pred, tgt))
        # Mean per-dim std.
        if arb.per_dim_std:
            arb_uncerts.append(
                float(sum(arb.per_dim_std))
                / float(len(arb.per_dim_std)))
    return QuintFidelity(
        direct_fidelity_a_e=float(
            sum(direct_a_e) / max(1, len(direct_a_e))),
        chain_len2_fid_mean=float(
            sum(chain_len2) / max(1, len(chain_len2))),
        chain_len3_fid_mean=float(
            sum(chain_len3) / max(1, len(chain_len3))),
        chain_len4_fid_mean=float(
            sum(chain_len4) / max(1, len(chain_len4))),
        transitivity_gap_len4=float(
            sum(gaps_len4) / max(1, len(gaps_len4))),
        arbitration_uncertainty_mean=float(
            sum(arb_uncerts) / max(1, len(arb_uncerts))),
        arbitration_uncertainty_max=float(
            max(arb_uncerts) if arb_uncerts else 0.0),
        weighted_arbitration_mean=float(
            sum(weighted_scores) / max(1, len(weighted_scores))),
        naive_arbitration_mean=float(
            sum(naive_scores) / max(1, len(naive_scores))),
    )


# =============================================================================
# Witnesses
# =============================================================================


@dataclasses.dataclass(frozen=True)
class MultiHopV3Witness:
    translator_cid: str
    backends: tuple[str, ...]
    chain_length: int
    direct_fidelity_a_e: float
    chain_len4_fidelity_mean: float
    transitivity_gap_len4: float
    arbitration_uncertainty_mean: float
    arbitration_weighted_minus_naive: float
    n_examples: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "translator_cid": str(self.translator_cid),
            "backends": list(self.backends),
            "chain_length": int(self.chain_length),
            "direct_fidelity_a_e": float(round(
                self.direct_fidelity_a_e, 12)),
            "chain_len4_fidelity_mean": float(round(
                self.chain_len4_fidelity_mean, 12)),
            "transitivity_gap_len4": float(round(
                self.transitivity_gap_len4, 12)),
            "arbitration_uncertainty_mean": float(round(
                self.arbitration_uncertainty_mean, 12)),
            "arbitration_weighted_minus_naive": float(round(
                self.arbitration_weighted_minus_naive, 12)),
            "n_examples": int(self.n_examples),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w53_mh_v3_witness",
            "witness": self.to_dict()})


def emit_multi_hop_v3_witness(
        *,
        translator: MultiHopBackendTranslator,
        examples: Sequence[MultiHopExample],
) -> MultiHopV3Witness:
    fid = score_quint_fidelity(translator, examples)
    return MultiHopV3Witness(
        translator_cid=str(translator.cid()),
        backends=tuple(translator.backends),
        chain_length=W53_DEFAULT_MH_V3_CHAIN_LENGTH,
        direct_fidelity_a_e=float(
            fid.direct_fidelity_a_e),
        chain_len4_fidelity_mean=float(
            fid.chain_len4_fid_mean),
        transitivity_gap_len4=float(
            fid.transitivity_gap_len4),
        arbitration_uncertainty_mean=float(
            fid.arbitration_uncertainty_mean),
        arbitration_weighted_minus_naive=float(
            fid.weighted_arbitration_mean
            - fid.naive_arbitration_mean),
        n_examples=int(len(examples)),
    )


# =============================================================================
# Verifier
# =============================================================================

W53_MH_V3_VERIFIER_FAILURE_MODES: tuple[str, ...] = (
    "w53_mh_v3_translator_cid_mismatch",
    "w53_mh_v3_n_backends_below_floor",
    "w53_mh_v3_chain_length_mismatch",
    "w53_mh_v3_transitivity_gap_above_ceiling",
    "w53_mh_v3_arbitration_weighted_below_naive",
)


def verify_multi_hop_v3_witness(
        witness: MultiHopV3Witness,
        *,
        expected_translator_cid: str | None = None,
        expected_n_backends: int | None = None,
        expected_chain_length: int | None = None,
        max_transitivity_gap: float | None = None,
        require_weighted_ge_naive: bool = False,
) -> dict[str, Any]:
    failures: list[str] = []
    if (expected_translator_cid is not None
            and witness.translator_cid
            != str(expected_translator_cid)):
        failures.append("w53_mh_v3_translator_cid_mismatch")
    if (expected_n_backends is not None
            and len(witness.backends)
            < int(expected_n_backends)):
        failures.append("w53_mh_v3_n_backends_below_floor")
    if (expected_chain_length is not None
            and witness.chain_length
            != int(expected_chain_length)):
        failures.append("w53_mh_v3_chain_length_mismatch")
    if (max_transitivity_gap is not None
            and witness.transitivity_gap_len4
            > float(max_transitivity_gap)):
        failures.append(
            "w53_mh_v3_transitivity_gap_above_ceiling")
    if (require_weighted_ge_naive
            and witness.arbitration_weighted_minus_naive
            < 0.0):
        failures.append(
            "w53_mh_v3_arbitration_weighted_below_naive")
    return {
        "ok": (len(failures) == 0),
        "failures": failures,
        "witness_cid": witness.cid(),
    }


# =============================================================================
# Convenience helpers
# =============================================================================


def build_unfitted_quint_translator(
        *,
        backends: Sequence[str] = (
            W53_DEFAULT_MH_V3_BACKENDS),
        code_dim: int = W53_DEFAULT_MH_V3_CODE_DIM,
        feature_dim: int = W53_DEFAULT_MH_V3_FEATURE_DIM,
        seed: int = W47_DEFAULT_TRAIN_SEED,
) -> MultiHopBackendTranslator:
    return build_unfitted_multi_hop_translator(
        backends=tuple(backends),
        code_dim=int(code_dim),
        feature_dim=int(feature_dim),
        seed=int(seed))


def synthesize_quint_training_set(
        *,
        n_examples: int = 24,
        backends: Sequence[str] = (
            W53_DEFAULT_MH_V3_BACKENDS),
        code_dim: int = W53_DEFAULT_MH_V3_CODE_DIM,
        feature_dim: int = W53_DEFAULT_MH_V3_FEATURE_DIM,
        seed: int = W47_DEFAULT_TRAIN_SEED,
):
    return synthesize_multi_hop_training_set(
        n_examples=int(n_examples),
        backends=tuple(backends),
        code_dim=int(code_dim),
        feature_dim=int(feature_dim),
        seed=int(seed))


def fit_quint_translator(
        training_set,
        *,
        n_steps: int = 192,
        seed: int = W47_DEFAULT_TRAIN_SEED,
):
    return fit_multi_hop_translator(
        training_set, n_steps=int(n_steps), seed=int(seed))


__all__ = [
    "W53_MH_V3_SCHEMA_VERSION",
    "W53_DEFAULT_MH_V3_BACKENDS",
    "W53_DEFAULT_MH_V3_N_BACKENDS",
    "W53_DEFAULT_MH_V3_CODE_DIM",
    "W53_DEFAULT_MH_V3_FEATURE_DIM",
    "W53_DEFAULT_MH_V3_CHAIN_LENGTH",
    "W53_DEFAULT_MH_V3_TRANSITIVITY_FLOOR",
    "W53_MH_V3_VERIFIER_FAILURE_MODES",
    "UncertaintyArbitrationResult",
    "QuintFidelity",
    "MultiHopV3Witness",
    "uncertainty_aware_arbitration",
    "score_quint_fidelity",
    "emit_multi_hop_v3_witness",
    "verify_multi_hop_v3_witness",
    "build_unfitted_quint_translator",
    "synthesize_quint_training_set",
    "fit_quint_translator",
]
