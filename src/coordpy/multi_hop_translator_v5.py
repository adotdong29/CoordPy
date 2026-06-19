"""W55 M2 — Multi-Hop Translator V5 (7-backend, chain-length-6,
   trust-weighted compromise arbitration).

Extends W54 V4 (6-backend) to:

* **7 backends** ``(A, B, C, D, E, F, G)`` — 42 directed edges
* **chain-length-6** scoring ``A→B→C→D→E→F→G`` with transitivity
  gap reporting (compared against direct ``A→G``)
* **trust-weighted compromise arbitration** that augments the
  W54 disagreement-aware compromise with a per-backend trust
  scalar. The arbiter:
    1. Computes per-path predictions + per-path confidence.
    2. Computes per-path **trust weight** = ``product(trust_e)``
       over each edge in the path.
    3. Picks the largest pairwise-agreeing subset where every
       pair has cosine ≥ floor AND the per-path trust weight ≥
       a trust floor.
    4. Aggregates predictions weighted by ``(confidence ×
       trust_weight)``.
    5. If no agreeing subset exists, falls back to the
       highest-(confidence × trust_weight) single path and
       sets ``abstain=True``.

Trust signature monotonicity: when all backends have trust=1,
this reduces exactly to the W54 V4 disagreement-aware compromise
(equal-trust softmax).

Pure-Python only — wraps W54's hex translator.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

from .autograd_manifold import W47_DEFAULT_TRAIN_SEED
from .multi_hop_translator import (
    MultiHopBackendTranslator,
    MultiHopExample,
    MultiHopTrainingSet,
    W52_DEFAULT_MH_CODE_DIM,
    W52_DEFAULT_MH_FEATURE_DIM,
    build_unfitted_multi_hop_translator,
    fit_multi_hop_translator,
    synthesize_multi_hop_training_set,
)
from .multi_hop_translator_v4 import (
    CompromiseArbitrationResult,
    HexFidelity,
    W54_DEFAULT_MH_V4_COMPROMISE_FLOOR,
    disagreement_compromise_arbitration,
    score_hex_fidelity,
)


# =============================================================================
# Schema, defaults
# =============================================================================

W55_MH_V5_SCHEMA_VERSION: str = (
    "coordpy.multi_hop_translator_v5.v1")

W55_DEFAULT_MH_V5_BACKENDS: tuple[str, ...] = (
    "A", "B", "C", "D", "E", "F", "G")
W55_DEFAULT_MH_V5_N_BACKENDS: int = 7
W55_DEFAULT_MH_V5_CODE_DIM: int = W52_DEFAULT_MH_CODE_DIM
W55_DEFAULT_MH_V5_FEATURE_DIM: int = W52_DEFAULT_MH_FEATURE_DIM
W55_DEFAULT_MH_V5_CHAIN_LENGTH: int = 6
W55_DEFAULT_MH_V5_COMPROMISE_FLOOR: float = 0.5
W55_DEFAULT_MH_V5_TRUST_FLOOR: float = 0.3
W55_DEFAULT_MH_V5_DEFAULT_TRUST: float = 1.0


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
# Trust-weighted compromise arbitration
# =============================================================================


@dataclasses.dataclass(frozen=True)
class TrustWeightedCompromiseResult:
    """Trust-weighted largest-agreeing subset prediction."""

    prediction: tuple[float, ...]
    selected_paths: tuple[tuple[str, ...], ...]
    n_paths_total: int
    n_paths_selected: int
    pairwise_floor: float
    trust_floor: float
    aggregate_confidence: float
    aggregate_trust_weight: float
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
            "trust_floor": float(round(self.trust_floor, 12)),
            "aggregate_confidence": float(round(
                self.aggregate_confidence, 12)),
            "aggregate_trust_weight": float(round(
                self.aggregate_trust_weight, 12)),
            "abstain": bool(self.abstain),
        }


def trust_weighted_compromise_arbitration(
        translator: MultiHopBackendTranslator,
        *,
        paths: Sequence[Sequence[str]],
        input_vec: Sequence[float],
        feature_dim: int,
        trust_per_backend: Mapping[str, float] | None = None,
        compromise_floor: float = (
            W55_DEFAULT_MH_V5_COMPROMISE_FLOOR),
        trust_floor: float = W55_DEFAULT_MH_V5_TRUST_FLOOR,
) -> TrustWeightedCompromiseResult:
    """Trust-weighted compromise arbitration.

    Each path gets a confidence (product of edge confidences) AND
    a trust weight (product of backend trust scalars along the
    path). Paths with trust weight < trust_floor are excluded.
    The largest pairwise-agreeing subset is selected; predictions
    are aggregated by (confidence × trust_weight).
    """
    fd = int(feature_dim)
    tpb: dict[str, float] = {}
    if trust_per_backend is not None:
        for k, v in trust_per_backend.items():
            tpb[str(k)] = float(max(0.0, min(
                1.0, float(v))))
    preds: list[list[float]] = []
    confs: list[float] = []
    trust_w: list[float] = []
    used_paths: list[tuple[str, ...]] = []
    for p in paths:
        chain = tuple(str(b) for b in p)
        if len(chain) < 2:
            continue
        pred = translator.apply_chain_value(chain, input_vec)
        c = 1.0
        tw = 1.0
        for k in range(len(chain) - 1):
            e = translator.get(chain[k], chain[k + 1])
            if e is None:
                continue
            c *= float(e.confidence)
        # Trust weight: product over all backends visited
        # (excluding the source, since source trust is implicit).
        for b in chain[1:]:
            tw *= float(tpb.get(
                str(b), W55_DEFAULT_MH_V5_DEFAULT_TRUST))
        if tw < float(trust_floor):
            continue
        preds.append(list(pred))
        confs.append(float(c))
        trust_w.append(float(tw))
        used_paths.append(chain)
    n = len(preds)
    if n == 0:
        return TrustWeightedCompromiseResult(
            prediction=tuple([0.0] * fd),
            selected_paths=(),
            n_paths_total=int(len(paths)),
            n_paths_selected=0,
            pairwise_floor=float(compromise_floor),
            trust_floor=float(trust_floor),
            aggregate_confidence=0.0,
            aggregate_trust_weight=0.0,
            abstain=True,
        )
    # Pairwise cosine matrix.
    cos = [[1.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            cc = _cosine(preds[i], preds[j])
            cos[i][j] = float(cc)
            cos[j][i] = float(cc)
    # Trust-weighted centrality: rank by (trust × centrality).
    centrality = [
        sum(cos[i][j] for j in range(n) if j != i)
        for i in range(n)
    ]
    score = [
        float(trust_w[i]) * float(centrality[i])
        for i in range(n)
    ]
    order = sorted(
        range(n), key=lambda i: -score[i])
    selected: list[int] = []
    for i in order:
        ok = True
        for s in selected:
            if cos[i][s] < float(compromise_floor):
                ok = False
                break
        if ok:
            selected.append(i)
    if len(selected) < 2:
        # Fall back to highest (confidence × trust) single path.
        best_i = max(
            range(n),
            key=lambda i: float(confs[i]) * float(trust_w[i]))
        return TrustWeightedCompromiseResult(
            prediction=tuple(_round_floats(preds[best_i])),
            selected_paths=(used_paths[best_i],),
            n_paths_total=int(len(paths)),
            n_paths_selected=1,
            pairwise_floor=float(compromise_floor),
            trust_floor=float(trust_floor),
            aggregate_confidence=float(
                confs[best_i]) * 0.5,
            aggregate_trust_weight=float(trust_w[best_i]),
            abstain=True,
        )
    weights = [
        float(confs[i]) * float(trust_w[i])
        for i in selected
    ]
    total = float(sum(weights))
    weighted_pred = [0.0] * fd
    if total <= 1e-30:
        for i in selected:
            for j in range(fd):
                weighted_pred[j] += float(
                    preds[i][j] if j < len(preds[i]) else 0.0
                ) / float(len(selected))
    else:
        for k, i in enumerate(selected):
            w = float(weights[k]) / float(total)
            for j in range(fd):
                weighted_pred[j] += w * float(
                    preds[i][j] if j < len(preds[i]) else 0.0)
    agg_conf = float(sum(confs[i] for i in selected)) / float(
        len(selected))
    agg_trust = float(sum(trust_w[i] for i in selected)) / (
        float(len(selected)))
    return TrustWeightedCompromiseResult(
        prediction=tuple(_round_floats(weighted_pred)),
        selected_paths=tuple(used_paths[i] for i in selected),
        n_paths_total=int(len(paths)),
        n_paths_selected=int(len(selected)),
        pairwise_floor=float(compromise_floor),
        trust_floor=float(trust_floor),
        aggregate_confidence=float(agg_conf),
        aggregate_trust_weight=float(agg_trust),
        abstain=False,
    )


# =============================================================================
# Hept translator + chain-len-6 fidelity
# =============================================================================


@dataclasses.dataclass(frozen=True)
class HeptFidelity:
    """7-backend chain-length-6 fidelity report."""

    direct_fidelity_a_g: float
    chain_len2_fid_mean: float
    chain_len3_fid_mean: float
    chain_len4_fid_mean: float
    chain_len5_fid_mean: float
    chain_len6_fid_mean: float
    transitivity_gap_len6: float
    trust_compromise_pick_rate: float
    trust_compromise_abstain_rate: float
    trust_compromise_fidelity_mean: float
    naive_arbitration_mean: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "direct_fidelity_a_g": float(round(
                self.direct_fidelity_a_g, 12)),
            "chain_len2_fid_mean": float(round(
                self.chain_len2_fid_mean, 12)),
            "chain_len3_fid_mean": float(round(
                self.chain_len3_fid_mean, 12)),
            "chain_len4_fid_mean": float(round(
                self.chain_len4_fid_mean, 12)),
            "chain_len5_fid_mean": float(round(
                self.chain_len5_fid_mean, 12)),
            "chain_len6_fid_mean": float(round(
                self.chain_len6_fid_mean, 12)),
            "transitivity_gap_len6": float(round(
                self.transitivity_gap_len6, 12)),
            "trust_compromise_pick_rate": float(round(
                self.trust_compromise_pick_rate, 12)),
            "trust_compromise_abstain_rate": float(round(
                self.trust_compromise_abstain_rate, 12)),
            "trust_compromise_fidelity_mean": float(round(
                self.trust_compromise_fidelity_mean, 12)),
            "naive_arbitration_mean": float(round(
                self.naive_arbitration_mean, 12)),
        }


def score_hept_fidelity(
        translator: MultiHopBackendTranslator,
        examples: Sequence[MultiHopExample],
        *,
        trust_per_backend: Mapping[str, float] | None = None,
        compromise_floor: float = (
            W55_DEFAULT_MH_V5_COMPROMISE_FLOOR),
        trust_floor: float = W55_DEFAULT_MH_V5_TRUST_FLOOR,
) -> HeptFidelity:
    """Score 7-backend chain-length-6 fidelity."""
    if not examples or len(translator.backends) < 7:
        return HeptFidelity(
            direct_fidelity_a_g=0.0,
            chain_len2_fid_mean=0.0,
            chain_len3_fid_mean=0.0,
            chain_len4_fid_mean=0.0,
            chain_len5_fid_mean=0.0,
            chain_len6_fid_mean=0.0,
            transitivity_gap_len6=0.0,
            trust_compromise_pick_rate=0.0,
            trust_compromise_abstain_rate=0.0,
            trust_compromise_fidelity_mean=0.0,
            naive_arbitration_mean=0.0,
        )
    src, b, c, d, e, f, dst = translator.backends[:7]
    fd = int(translator.feature_dim)
    direct: list[float] = []
    ch2: list[float] = []
    ch3: list[float] = []
    ch4: list[float] = []
    ch5: list[float] = []
    ch6: list[float] = []
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
        p6 = translator.apply_chain_value(
            (src, b, c, d, e, f, dst),
            ex.feature_by_backend[src])
        ch6.append(_cosine(p6, tgt))
        if direct:
            gaps.append(abs(direct[-1] - ch6[-1]))
        paths = (
            (src, dst),
            (src, b, dst),
            (src, b, c, dst),
            (src, b, c, d, dst),
            (src, b, c, d, e, dst),
            (src, b, c, d, e, f, dst),
        )
        arb = trust_weighted_compromise_arbitration(
            translator, paths=paths,
            input_vec=ex.feature_by_backend[src],
            feature_dim=fd,
            trust_per_backend=trust_per_backend,
            compromise_floor=float(compromise_floor),
            trust_floor=float(trust_floor))
        comp_scores.append(_cosine(arb.prediction, tgt))
        if arb.abstain:
            n_abstain += 1
        else:
            n_compromise += 1
        naive_pred = translator.naive_arbitration(
            paths, ex.feature_by_backend[src])
        naive_scores.append(_cosine(naive_pred, tgt))
    n_t = max(1, len(examples))
    return HeptFidelity(
        direct_fidelity_a_g=float(
            sum(direct) / max(1, len(direct))),
        chain_len2_fid_mean=float(
            sum(ch2) / max(1, len(ch2))),
        chain_len3_fid_mean=float(
            sum(ch3) / max(1, len(ch3))),
        chain_len4_fid_mean=float(
            sum(ch4) / max(1, len(ch4))),
        chain_len5_fid_mean=float(
            sum(ch5) / max(1, len(ch5))),
        chain_len6_fid_mean=float(
            sum(ch6) / max(1, len(ch6))),
        transitivity_gap_len6=float(
            sum(gaps) / max(1, len(gaps))),
        trust_compromise_pick_rate=float(n_compromise) / float(n_t),
        trust_compromise_abstain_rate=float(n_abstain) / float(n_t),
        trust_compromise_fidelity_mean=float(
            sum(comp_scores) / max(1, len(comp_scores))),
        naive_arbitration_mean=float(
            sum(naive_scores) / max(1, len(naive_scores))),
    )


# =============================================================================
# Witness
# =============================================================================


@dataclasses.dataclass(frozen=True)
class MultiHopV5Witness:
    translator_cid: str
    backends: tuple[str, ...]
    chain_length: int
    direct_fidelity_a_g: float
    chain_len6_fidelity_mean: float
    transitivity_gap_len6: float
    trust_compromise_pick_rate: float
    trust_compromise_abstain_rate: float
    trust_compromise_fidelity_mean: float
    trust_vs_naive_delta: float
    n_examples: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "translator_cid": str(self.translator_cid),
            "backends": list(self.backends),
            "chain_length": int(self.chain_length),
            "direct_fidelity_a_g": float(round(
                self.direct_fidelity_a_g, 12)),
            "chain_len6_fidelity_mean": float(round(
                self.chain_len6_fidelity_mean, 12)),
            "transitivity_gap_len6": float(round(
                self.transitivity_gap_len6, 12)),
            "trust_compromise_pick_rate": float(round(
                self.trust_compromise_pick_rate, 12)),
            "trust_compromise_abstain_rate": float(round(
                self.trust_compromise_abstain_rate, 12)),
            "trust_compromise_fidelity_mean": float(round(
                self.trust_compromise_fidelity_mean, 12)),
            "trust_vs_naive_delta": float(round(
                self.trust_vs_naive_delta, 12)),
            "n_examples": int(self.n_examples),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w55_mh_v5_witness",
            "witness": self.to_dict()})


def emit_multi_hop_v5_witness(
        *,
        translator: MultiHopBackendTranslator,
        examples: Sequence[MultiHopExample],
        trust_per_backend: Mapping[str, float] | None = None,
        compromise_floor: float = (
            W55_DEFAULT_MH_V5_COMPROMISE_FLOOR),
        trust_floor: float = W55_DEFAULT_MH_V5_TRUST_FLOOR,
) -> MultiHopV5Witness:
    fid = score_hept_fidelity(
        translator, examples,
        trust_per_backend=trust_per_backend,
        compromise_floor=float(compromise_floor),
        trust_floor=float(trust_floor))
    return MultiHopV5Witness(
        translator_cid=str(translator.cid()),
        backends=tuple(translator.backends),
        chain_length=W55_DEFAULT_MH_V5_CHAIN_LENGTH,
        direct_fidelity_a_g=float(fid.direct_fidelity_a_g),
        chain_len6_fidelity_mean=float(
            fid.chain_len6_fid_mean),
        transitivity_gap_len6=float(
            fid.transitivity_gap_len6),
        trust_compromise_pick_rate=float(
            fid.trust_compromise_pick_rate),
        trust_compromise_abstain_rate=float(
            fid.trust_compromise_abstain_rate),
        trust_compromise_fidelity_mean=float(
            fid.trust_compromise_fidelity_mean),
        trust_vs_naive_delta=float(
            fid.trust_compromise_fidelity_mean
            - fid.naive_arbitration_mean),
        n_examples=int(len(examples)),
    )


# =============================================================================
# Verifier
# =============================================================================

W55_MH_V5_VERIFIER_FAILURE_MODES: tuple[str, ...] = (
    "w55_mh_v5_translator_cid_mismatch",
    "w55_mh_v5_n_backends_below_floor",
    "w55_mh_v5_chain_length_mismatch",
    "w55_mh_v5_transitivity_gap_above_ceiling",
    "w55_mh_v5_pick_rate_invalid",
    "w55_mh_v5_trust_floor_invalid",
)


def verify_multi_hop_v5_witness(
        witness: MultiHopV5Witness,
        *,
        expected_translator_cid: str | None = None,
        expected_n_backends: int | None = None,
        expected_chain_length: int | None = None,
        max_transitivity_gap: float | None = None,
) -> dict[str, Any]:
    failures: list[str] = []
    if (expected_translator_cid is not None
            and witness.translator_cid
            != str(expected_translator_cid)):
        failures.append("w55_mh_v5_translator_cid_mismatch")
    if (expected_n_backends is not None
            and len(witness.backends)
            < int(expected_n_backends)):
        failures.append("w55_mh_v5_n_backends_below_floor")
    if (expected_chain_length is not None
            and witness.chain_length
            != int(expected_chain_length)):
        failures.append("w55_mh_v5_chain_length_mismatch")
    if (max_transitivity_gap is not None
            and witness.transitivity_gap_len6
            > float(max_transitivity_gap)):
        failures.append(
            "w55_mh_v5_transitivity_gap_above_ceiling")
    pr = (
        float(witness.trust_compromise_pick_rate)
        + float(witness.trust_compromise_abstain_rate))
    if not (-1e-6 <= pr - 1.0 <= 1e-6 or pr == 0.0):
        failures.append("w55_mh_v5_pick_rate_invalid")
    return {
        "ok": (len(failures) == 0),
        "failures": failures,
        "witness_cid": witness.cid(),
    }


# =============================================================================
# Convenience helpers
# =============================================================================


def build_unfitted_hept_translator(
        *,
        backends: Sequence[str] = (
            W55_DEFAULT_MH_V5_BACKENDS),
        code_dim: int = W55_DEFAULT_MH_V5_CODE_DIM,
        feature_dim: int = W55_DEFAULT_MH_V5_FEATURE_DIM,
        seed: int = W47_DEFAULT_TRAIN_SEED,
) -> MultiHopBackendTranslator:
    return build_unfitted_multi_hop_translator(
        backends=tuple(backends),
        code_dim=int(code_dim),
        feature_dim=int(feature_dim),
        seed=int(seed))


def synthesize_hept_training_set(
        *,
        n_examples: int = 24,
        backends: Sequence[str] = (
            W55_DEFAULT_MH_V5_BACKENDS),
        code_dim: int = W55_DEFAULT_MH_V5_CODE_DIM,
        feature_dim: int = W55_DEFAULT_MH_V5_FEATURE_DIM,
        seed: int = W47_DEFAULT_TRAIN_SEED,
):
    return synthesize_multi_hop_training_set(
        n_examples=int(n_examples),
        backends=tuple(backends),
        code_dim=int(code_dim),
        feature_dim=int(feature_dim),
        seed=int(seed))


def fit_hept_translator(
        training_set,
        *,
        n_steps: int = 192,
        seed: int = W47_DEFAULT_TRAIN_SEED,
):
    return fit_multi_hop_translator(
        training_set, n_steps=int(n_steps), seed=int(seed))


__all__ = [
    "W55_MH_V5_SCHEMA_VERSION",
    "W55_DEFAULT_MH_V5_BACKENDS",
    "W55_DEFAULT_MH_V5_N_BACKENDS",
    "W55_DEFAULT_MH_V5_CODE_DIM",
    "W55_DEFAULT_MH_V5_FEATURE_DIM",
    "W55_DEFAULT_MH_V5_CHAIN_LENGTH",
    "W55_DEFAULT_MH_V5_COMPROMISE_FLOOR",
    "W55_DEFAULT_MH_V5_TRUST_FLOOR",
    "W55_DEFAULT_MH_V5_DEFAULT_TRUST",
    "W55_MH_V5_VERIFIER_FAILURE_MODES",
    "TrustWeightedCompromiseResult",
    "HeptFidelity",
    "MultiHopV5Witness",
    "trust_weighted_compromise_arbitration",
    "score_hept_fidelity",
    "emit_multi_hop_v5_witness",
    "verify_multi_hop_v5_witness",
    "build_unfitted_hept_translator",
    "synthesize_hept_training_set",
    "fit_hept_translator",
]
