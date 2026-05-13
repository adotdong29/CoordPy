"""W54 M2 — Multi-Hop Translator V4 (6-backend, chain-length-5,
   disagreement-aware compromise arbitration).

Extends W53 V3 to:

* **6 backends** ``(A, B, C, D, E, F)`` — 30 directed edges
* **chain-length-5** scoring ``A→B→C→D→E→F`` with explicit
  transitivity gap reporting
* **disagreement-aware compromise arbitration** that picks the
  largest sub-set of paths whose pairwise predictions agree
  within a configurable cosine floor; if no compatible sub-set
  exists, the arbiter abstains and returns the highest-confidence
  single-path prediction with a low aggregate_confidence

The compromise scheme is the W54 answer to: "five paths give
five different answers — which subset should we trust?"

Pure-Python only — wraps W52's ``MultiHopBackendTranslator``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Any, Sequence

from .autograd_manifold import W47_DEFAULT_TRAIN_SEED
from .multi_hop_translator import (
    MultiHopBackendTranslator,
    MultiHopExample,
    W52_DEFAULT_MH_CODE_DIM,
    W52_DEFAULT_MH_FEATURE_DIM,
    build_unfitted_multi_hop_translator,
    fit_multi_hop_translator,
    synthesize_multi_hop_training_set,
)


# =============================================================================
# Schema, defaults
# =============================================================================

W54_MH_V4_SCHEMA_VERSION: str = (
    "coordpy.multi_hop_translator_v4.v1")

W54_DEFAULT_MH_V4_BACKENDS: tuple[str, ...] = (
    "A", "B", "C", "D", "E", "F")
W54_DEFAULT_MH_V4_N_BACKENDS: int = 6
W54_DEFAULT_MH_V4_CODE_DIM: int = W52_DEFAULT_MH_CODE_DIM
W54_DEFAULT_MH_V4_FEATURE_DIM: int = W52_DEFAULT_MH_FEATURE_DIM
W54_DEFAULT_MH_V4_CHAIN_LENGTH: int = 5
W54_DEFAULT_MH_V4_COMPROMISE_FLOOR: float = 0.5


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
# Disagreement-aware compromise arbitration
# =============================================================================


@dataclasses.dataclass(frozen=True)
class CompromiseArbitrationResult:
    """Largest-agreeing subset prediction + abstain flag."""

    prediction: tuple[float, ...]
    selected_paths: tuple[tuple[str, ...], ...]
    n_paths_total: int
    n_paths_selected: int
    pairwise_floor: float
    aggregate_confidence: float
    abstain: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "prediction": list(_round_floats(self.prediction)),
            "selected_paths": [
                list(p) for p in self.selected_paths],
            "n_paths_total": int(self.n_paths_total),
            "n_paths_selected": int(self.n_paths_selected),
            "pairwise_floor": float(round(
                self.pairwise_floor, 12)),
            "aggregate_confidence": float(round(
                self.aggregate_confidence, 12)),
            "abstain": bool(self.abstain),
        }


def disagreement_compromise_arbitration(
        translator: MultiHopBackendTranslator,
        *,
        paths: Sequence[Sequence[str]],
        input_vec: Sequence[float],
        feature_dim: int,
        compromise_floor: float = (
            W54_DEFAULT_MH_V4_COMPROMISE_FLOOR),
) -> CompromiseArbitrationResult:
    """Largest pairwise-agreeing subset arbitration.

    For each path, compute prediction + confidence; build pairwise
    cosine matrix between predictions; greedily pick the centrally-
    most-connected subset where every pair has cosine >=
    compromise_floor. If no pair clears the floor, fall back to
    the highest-confidence single path and set ``abstain=True``.
    """
    fd = int(feature_dim)
    preds: list[list[float]] = []
    confs: list[float] = []
    used_paths: list[tuple[str, ...]] = []
    for p in paths:
        chain = tuple(str(b) for b in p)
        if len(chain) < 2:
            continue
        pred = translator.apply_chain_value(chain, input_vec)
        c = 1.0
        for k in range(len(chain) - 1):
            e = translator.get(chain[k], chain[k + 1])
            if e is None:
                continue
            c *= float(e.confidence)
        preds.append(list(pred))
        confs.append(float(c))
        used_paths.append(chain)
    n = len(preds)
    if n == 0:
        return CompromiseArbitrationResult(
            prediction=tuple([0.0] * fd),
            selected_paths=(),
            n_paths_total=0,
            n_paths_selected=0,
            pairwise_floor=float(compromise_floor),
            aggregate_confidence=0.0,
            abstain=True,
        )
    cos = [[1.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            c = _cosine(preds[i], preds[j])
            cos[i][j] = float(c)
            cos[j][i] = float(c)
    centrality = [
        sum(cos[i][j] for j in range(n) if j != i)
        for i in range(n)
    ]
    order = sorted(
        range(n), key=lambda i: -centrality[i])
    selected: list[int] = []
    for i in order:
        ok = True
        for s in selected:
            if cos[i][s] < float(compromise_floor):
                ok = False
                break
        if ok:
            selected.append(i)
    # If only 1 path survives, abstain — single-path doesn't qualify
    # as a compromise.
    if len(selected) < 2:
        best_i = max(range(n), key=lambda i: confs[i])
        return CompromiseArbitrationResult(
            prediction=tuple(_round_floats(preds[best_i])),
            selected_paths=(used_paths[best_i],),
            n_paths_total=int(n),
            n_paths_selected=1,
            pairwise_floor=float(compromise_floor),
            aggregate_confidence=float(confs[best_i]) * 0.5,
            abstain=True,
        )
    # Confidence-weighted mean over selected.
    sel_conf = [float(confs[i]) for i in selected]
    total = float(sum(sel_conf))
    weighted_pred = [0.0] * fd
    if total <= 1e-30:
        for i in selected:
            for j in range(fd):
                weighted_pred[j] += float(
                    preds[i][j] if j < len(preds[i]) else 0.0
                ) / float(len(selected))
    else:
        for i in selected:
            w = float(confs[i]) / float(total)
            for j in range(fd):
                weighted_pred[j] += w * float(
                    preds[i][j] if j < len(preds[i]) else 0.0)
    agg = float(sum(sel_conf)) / float(len(selected))
    return CompromiseArbitrationResult(
        prediction=tuple(_round_floats(weighted_pred)),
        selected_paths=tuple(used_paths[i] for i in selected),
        n_paths_total=int(n),
        n_paths_selected=int(len(selected)),
        pairwise_floor=float(compromise_floor),
        aggregate_confidence=float(agg),
        abstain=False,
    )


# =============================================================================
# Hex translator + chain-len-5 fidelity
# =============================================================================


@dataclasses.dataclass(frozen=True)
class HexFidelity:
    """6-backend fidelity report."""

    direct_fidelity_a_f: float
    chain_len2_fid_mean: float
    chain_len3_fid_mean: float
    chain_len4_fid_mean: float
    chain_len5_fid_mean: float
    transitivity_gap_len5: float
    compromise_pick_rate: float
    compromise_abstain_rate: float
    compromise_fidelity_mean: float
    naive_arbitration_mean: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "direct_fidelity_a_f": float(round(
                self.direct_fidelity_a_f, 12)),
            "chain_len2_fid_mean": float(round(
                self.chain_len2_fid_mean, 12)),
            "chain_len3_fid_mean": float(round(
                self.chain_len3_fid_mean, 12)),
            "chain_len4_fid_mean": float(round(
                self.chain_len4_fid_mean, 12)),
            "chain_len5_fid_mean": float(round(
                self.chain_len5_fid_mean, 12)),
            "transitivity_gap_len5": float(round(
                self.transitivity_gap_len5, 12)),
            "compromise_pick_rate": float(round(
                self.compromise_pick_rate, 12)),
            "compromise_abstain_rate": float(round(
                self.compromise_abstain_rate, 12)),
            "compromise_fidelity_mean": float(round(
                self.compromise_fidelity_mean, 12)),
            "naive_arbitration_mean": float(round(
                self.naive_arbitration_mean, 12)),
        }


def score_hex_fidelity(
        translator: MultiHopBackendTranslator,
        examples: Sequence[MultiHopExample],
        *,
        compromise_floor: float = (
            W54_DEFAULT_MH_V4_COMPROMISE_FLOOR),
) -> HexFidelity:
    """Score 6-backend chain-length-5 fidelity."""
    if not examples or len(translator.backends) < 6:
        return HexFidelity(
            direct_fidelity_a_f=0.0,
            chain_len2_fid_mean=0.0,
            chain_len3_fid_mean=0.0,
            chain_len4_fid_mean=0.0,
            chain_len5_fid_mean=0.0,
            transitivity_gap_len5=0.0,
            compromise_pick_rate=0.0,
            compromise_abstain_rate=0.0,
            compromise_fidelity_mean=0.0,
            naive_arbitration_mean=0.0,
        )
    src, b, c, d, e, dst = translator.backends[:6]
    fd = int(translator.feature_dim)
    direct: list[float] = []
    ch2: list[float] = []
    ch3: list[float] = []
    ch4: list[float] = []
    ch5: list[float] = []
    gaps: list[float] = []
    comp_scores: list[float] = []
    naive_scores: list[float] = []
    n_compromise = 0
    n_abstain = 0
    for ex in examples:
        tgt = ex.feature_by_backend[dst]
        edge_sf = translator.get(src, dst)
        if edge_sf is not None:
            pred = edge_sf.apply_value(
                ex.feature_by_backend[src])
            direct.append(_cosine(pred, tgt))
        p2 = translator.apply_chain_value(
            (src, b, dst), ex.feature_by_backend[src])
        ch2.append(_cosine(p2, tgt))
        p3 = translator.apply_chain_value(
            (src, b, c, dst), ex.feature_by_backend[src])
        ch3.append(_cosine(p3, tgt))
        p4 = translator.apply_chain_value(
            (src, b, c, d, dst), ex.feature_by_backend[src])
        ch4.append(_cosine(p4, tgt))
        p5 = translator.apply_chain_value(
            (src, b, c, d, e, dst),
            ex.feature_by_backend[src])
        ch5.append(_cosine(p5, tgt))
        if direct:
            gaps.append(abs(direct[-1] - ch5[-1]))
        paths = (
            (src, dst),
            (src, b, dst),
            (src, b, c, dst),
            (src, b, c, d, dst),
            (src, b, c, d, e, dst),
        )
        arb = disagreement_compromise_arbitration(
            translator, paths=paths,
            input_vec=ex.feature_by_backend[src],
            feature_dim=fd,
            compromise_floor=float(compromise_floor))
        comp_scores.append(_cosine(arb.prediction, tgt))
        if arb.abstain:
            n_abstain += 1
        else:
            n_compromise += 1
        naive_pred = translator.naive_arbitration(
            paths, ex.feature_by_backend[src])
        naive_scores.append(_cosine(naive_pred, tgt))
    n_t = max(1, len(examples))
    return HexFidelity(
        direct_fidelity_a_f=float(
            sum(direct) / max(1, len(direct))),
        chain_len2_fid_mean=float(
            sum(ch2) / max(1, len(ch2))),
        chain_len3_fid_mean=float(
            sum(ch3) / max(1, len(ch3))),
        chain_len4_fid_mean=float(
            sum(ch4) / max(1, len(ch4))),
        chain_len5_fid_mean=float(
            sum(ch5) / max(1, len(ch5))),
        transitivity_gap_len5=float(
            sum(gaps) / max(1, len(gaps))),
        compromise_pick_rate=float(n_compromise) / float(n_t),
        compromise_abstain_rate=float(n_abstain) / float(n_t),
        compromise_fidelity_mean=float(
            sum(comp_scores) / max(1, len(comp_scores))),
        naive_arbitration_mean=float(
            sum(naive_scores) / max(1, len(naive_scores))),
    )


# =============================================================================
# Witness
# =============================================================================


@dataclasses.dataclass(frozen=True)
class MultiHopV4Witness:
    translator_cid: str
    backends: tuple[str, ...]
    chain_length: int
    direct_fidelity_a_f: float
    chain_len5_fidelity_mean: float
    transitivity_gap_len5: float
    compromise_pick_rate: float
    compromise_abstain_rate: float
    compromise_fidelity_mean: float
    compromise_vs_naive_delta: float
    n_examples: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "translator_cid": str(self.translator_cid),
            "backends": list(self.backends),
            "chain_length": int(self.chain_length),
            "direct_fidelity_a_f": float(round(
                self.direct_fidelity_a_f, 12)),
            "chain_len5_fidelity_mean": float(round(
                self.chain_len5_fidelity_mean, 12)),
            "transitivity_gap_len5": float(round(
                self.transitivity_gap_len5, 12)),
            "compromise_pick_rate": float(round(
                self.compromise_pick_rate, 12)),
            "compromise_abstain_rate": float(round(
                self.compromise_abstain_rate, 12)),
            "compromise_fidelity_mean": float(round(
                self.compromise_fidelity_mean, 12)),
            "compromise_vs_naive_delta": float(round(
                self.compromise_vs_naive_delta, 12)),
            "n_examples": int(self.n_examples),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w54_mh_v4_witness",
            "witness": self.to_dict()})


def emit_multi_hop_v4_witness(
        *,
        translator: MultiHopBackendTranslator,
        examples: Sequence[MultiHopExample],
        compromise_floor: float = (
            W54_DEFAULT_MH_V4_COMPROMISE_FLOOR),
) -> MultiHopV4Witness:
    fid = score_hex_fidelity(
        translator, examples,
        compromise_floor=float(compromise_floor))
    return MultiHopV4Witness(
        translator_cid=str(translator.cid()),
        backends=tuple(translator.backends),
        chain_length=W54_DEFAULT_MH_V4_CHAIN_LENGTH,
        direct_fidelity_a_f=float(fid.direct_fidelity_a_f),
        chain_len5_fidelity_mean=float(
            fid.chain_len5_fid_mean),
        transitivity_gap_len5=float(
            fid.transitivity_gap_len5),
        compromise_pick_rate=float(fid.compromise_pick_rate),
        compromise_abstain_rate=float(
            fid.compromise_abstain_rate),
        compromise_fidelity_mean=float(
            fid.compromise_fidelity_mean),
        compromise_vs_naive_delta=float(
            fid.compromise_fidelity_mean
            - fid.naive_arbitration_mean),
        n_examples=int(len(examples)),
    )


# =============================================================================
# Verifier
# =============================================================================

W54_MH_V4_VERIFIER_FAILURE_MODES: tuple[str, ...] = (
    "w54_mh_v4_translator_cid_mismatch",
    "w54_mh_v4_n_backends_below_floor",
    "w54_mh_v4_chain_length_mismatch",
    "w54_mh_v4_transitivity_gap_above_ceiling",
    "w54_mh_v4_compromise_below_naive",
    "w54_mh_v4_pick_rate_invalid",
)


def verify_multi_hop_v4_witness(
        witness: MultiHopV4Witness,
        *,
        expected_translator_cid: str | None = None,
        expected_n_backends: int | None = None,
        expected_chain_length: int | None = None,
        max_transitivity_gap: float | None = None,
        require_compromise_ge_naive: bool = False,
) -> dict[str, Any]:
    failures: list[str] = []
    if (expected_translator_cid is not None
            and witness.translator_cid
            != str(expected_translator_cid)):
        failures.append("w54_mh_v4_translator_cid_mismatch")
    if (expected_n_backends is not None
            and len(witness.backends)
            < int(expected_n_backends)):
        failures.append("w54_mh_v4_n_backends_below_floor")
    if (expected_chain_length is not None
            and witness.chain_length
            != int(expected_chain_length)):
        failures.append("w54_mh_v4_chain_length_mismatch")
    if (max_transitivity_gap is not None
            and witness.transitivity_gap_len5
            > float(max_transitivity_gap)):
        failures.append(
            "w54_mh_v4_transitivity_gap_above_ceiling")
    if (require_compromise_ge_naive
            and witness.compromise_vs_naive_delta < 0.0):
        failures.append("w54_mh_v4_compromise_below_naive")
    pr = (
        float(witness.compromise_pick_rate)
        + float(witness.compromise_abstain_rate))
    if not (-1e-6 <= pr - 1.0 <= 1e-6 or pr == 0.0):
        failures.append("w54_mh_v4_pick_rate_invalid")
    return {
        "ok": (len(failures) == 0),
        "failures": failures,
        "witness_cid": witness.cid(),
    }


# =============================================================================
# Convenience helpers
# =============================================================================


def build_unfitted_hex_translator(
        *,
        backends: Sequence[str] = (
            W54_DEFAULT_MH_V4_BACKENDS),
        code_dim: int = W54_DEFAULT_MH_V4_CODE_DIM,
        feature_dim: int = W54_DEFAULT_MH_V4_FEATURE_DIM,
        seed: int = W47_DEFAULT_TRAIN_SEED,
) -> MultiHopBackendTranslator:
    return build_unfitted_multi_hop_translator(
        backends=tuple(backends),
        code_dim=int(code_dim),
        feature_dim=int(feature_dim),
        seed=int(seed))


def synthesize_hex_training_set(
        *,
        n_examples: int = 24,
        backends: Sequence[str] = (
            W54_DEFAULT_MH_V4_BACKENDS),
        code_dim: int = W54_DEFAULT_MH_V4_CODE_DIM,
        feature_dim: int = W54_DEFAULT_MH_V4_FEATURE_DIM,
        seed: int = W47_DEFAULT_TRAIN_SEED,
):
    return synthesize_multi_hop_training_set(
        n_examples=int(n_examples),
        backends=tuple(backends),
        code_dim=int(code_dim),
        feature_dim=int(feature_dim),
        seed=int(seed))


def fit_hex_translator(
        training_set,
        *,
        n_steps: int = 192,
        seed: int = W47_DEFAULT_TRAIN_SEED,
):
    return fit_multi_hop_translator(
        training_set, n_steps=int(n_steps), seed=int(seed))


__all__ = [
    "W54_MH_V4_SCHEMA_VERSION",
    "W54_DEFAULT_MH_V4_BACKENDS",
    "W54_DEFAULT_MH_V4_N_BACKENDS",
    "W54_DEFAULT_MH_V4_CODE_DIM",
    "W54_DEFAULT_MH_V4_FEATURE_DIM",
    "W54_DEFAULT_MH_V4_CHAIN_LENGTH",
    "W54_DEFAULT_MH_V4_COMPROMISE_FLOOR",
    "W54_MH_V4_VERIFIER_FAILURE_MODES",
    "CompromiseArbitrationResult",
    "HexFidelity",
    "MultiHopV4Witness",
    "disagreement_compromise_arbitration",
    "score_hex_fidelity",
    "emit_multi_hop_v4_witness",
    "verify_multi_hop_v4_witness",
    "build_unfitted_hex_translator",
    "synthesize_hex_training_set",
    "fit_hex_translator",
]
