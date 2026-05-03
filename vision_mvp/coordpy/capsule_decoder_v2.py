"""Phase 49 — stronger bundle-aware capsule DECODING.

Phase 48 (``capsule_decoder``) shipped a linear ``LearnedBundleDecoder``
over a 10-feature class-agnostic vocabulary.  On the Phase-31
noisy bench it breaks the 0.200 structural ceiling by
+15..+17.5 pp (Claim W3-19) but does NOT cross the explicit
paradigm-shift threshold W3-C7 of $\\ge 0.400$ test accuracy with
approximately-symmetric cross-domain transfer.

Phase 49 attacks W3-C7 on four tightly-coupled research fronts:

  * **A — stronger decoders.**  This module adds three decoder
    families, each a principled step up from the Phase-48 linear
    logistic baseline:
      * ``InteractionBundleDecoder`` — linear over pairwise
        feature crosses of the V2 feature vocabulary.  Captures
        conjunctions (e.g. "lone_top_priority AND
        zero_vote_for_rc") that a pure-linear model cannot.
      * ``MLPBundleDecoder`` — a small (input→hidden→1) MLP
        over the V2 aggregated features, shared across rc.
        Tanh hidden layer (default width 8); trained by
        full-batch softmax-cross-entropy with numpy.  ~150
        parameters.
      * ``DeepSetBundleDecoder`` — a proper Deep Sets
        architecture: a per-capsule parametric embedding
        $\\varphi(c, rc) \\in \\mathbb{R}^{d_\\varphi}$, summed
        over the bundle, concatenated with the V2 aggregated
        features, scored through a final MLP.  ~250
        parameters.  The richest hypothesis class under test.

  * **B — symmetric transfer.**  This module also ships two
    decoders aimed at closing the Phase-48 security→incident
    asymmetry:
      * A ``BUNDLE_DECODER_FEATURES_V2`` vocabulary that adds
        eight **domain-invariant relative features** (rank
        features, delta-vs-best-other features, ratios) whose
        structural sign is the same across task families.
        Hypothesis W3-C8 (below): a linear decoder over V2
        features alone has more symmetric transfer than the
        Phase-48 V1 decoder.
      * ``MultitaskBundleDecoder`` — one weight vector
        factorised as ``w = w_shared + w_domain[d]`` for a
        one-hot domain tag $d$.  Pooled training on
        (incident, security); at test time each domain picks
        its own domain-head, so the multitask decoder can
        absorb the ``lone_top_priority`` sign-flip in the
        per-domain head while keeping task-family-invariant
        signals in the shared head.

  * **C — formal decoder story.**  See
    ``docs/CAPSULE_FORMALISM.md`` § 4.D for:
      * Theorem W3-20 (Deep Sets sufficiency — conditional):
        on a decoder distribution whose bundle-shape signature
        is separable in the per-capsule embedding space,
        DeepSetBundleDecoder strictly dominates every linear
        decoder over aggregated features.
      * Theorem W3-21 (linear-class asymmetry — negative,
        proved): a linear decoder over class-agnostic
        aggregated features whose sign on the gold-rc signature
        flips across domains (Phase-48's empirical finding)
        CANNOT achieve symmetric transfer under any choice of
        training data, no matter the regularisation.  This
        proves the Phase-48 asymmetry is a structural property
        of the linear-class hypothesis space, not a training
        artefact.

  * **D — honest programme update.**  If the Phase-49 best
    cell crosses 0.400 AND symmetric transfer is restored by
    one of the V2 decoders, W3-C7 is satisfied and the centre
    earns the paradigm-shift label.  If either gate fails,
    the limitation is named explicitly.

This module keeps the Capsule Contract C1..C6 untouched — it is
a *reader* of the admitted ledger, not a mutator.  No capsule
is created, admitted, sealed, or retired by any decoder here.

All decoders are deterministic in ``seed`` after numpy is seeded
at module load.  Weights are inspectable dicts (linear
decoders) or ``numpy`` arrays (MLP, DeepSet) with explicit
shape documentation.
"""

from __future__ import annotations

import dataclasses
import math
import random
from collections import defaultdict
from typing import Any, Callable, Iterable, Sequence

try:
    import numpy as np
except ImportError as ex:
    raise ImportError(
        "vision_mvp.coordpy.capsule_decoder_v2 requires numpy "
        "for MLP and DeepSet gradient training") from ex

from .capsule import ContextCapsule
from .capsule_decoder import (
    BundleDecoder, UNKNOWN, _votes_per_root_cause,
    _sources_per_root_cause,
)
from .capsule_policy_bundle import (
    DEFAULT_CLAIM_TO_ROOT_CAUSE, DEFAULT_PRIORITY_ORDER,
    DEFAULT_HIGH_PRIORITY_CUTOFF,
)


# =============================================================================
# BUNDLE_DECODER_FEATURES_V2 — extended vocabulary
# =============================================================================


BUNDLE_DECODER_FEATURES_V2: tuple[str, ...] = (
    # --- V1 aggregated features (kept verbatim) ---
    "bias",
    "log1p_votes",
    "log1p_sources",
    "votes_share",
    "high_priority_votes",
    "has_top_priority_kind",
    "multi_source_flag",
    "has_multi_source_kind",
    "lone_top_priority_flag",
    "zero_vote_flag",
    # --- V2 domain-invariant relative features ---
    "votes_minus_max_other",      # votes_for_rc - max_over_other_rc
    "sources_minus_max_other",    # sources_for_rc - max_over_other_rc
    "high_priority_minus_max_other",
    "is_strict_top_by_votes",     # 1 if rc is unique argmax of votes
    "is_co_top_by_votes",         # 1 if rc is in argmax set (tied ok)
    "is_strict_top_by_sources",
    "frac_bundle_implies_rc",     # votes_for_rc / bundle_size
    "log1p_bundle_size",
    "top_priority_implies_other_rc",
    "frac_high_priority_for_rc",  # rc's share of all high-priority
                                   # votes across all rcs
)


def _bundle_vote_summary(
        bundle: Sequence[ContextCapsule],
        claim_to_root_cause: dict[str, str],
        priority_order: tuple[str, ...],
        high_priority_cutoff: int = DEFAULT_HIGH_PRIORITY_CUTOFF,
        ) -> dict[str, Any]:
    """Compute bundle-level summary statistics used by V2
    featurisation.  Shared across rcs for efficiency.

    Returns a dict with keys:
      * ``votes_by_rc``    : rc → int votes
      * ``sources_by_rc``  : rc → set of distinct source roles
      * ``high_priority_by_rc`` : rc → int high-priority votes
      * ``bundle_size``    : total number of capsules in bundle
      * ``has_top_priority_in_bundle`` : bool
      * ``top_priority_implied_rc`` : str (the rc implied by the
        top-priority kind present, if any)
    """
    votes_by_rc: dict[str, int] = defaultdict(int)
    sources_by_rc: dict[str, set[str]] = defaultdict(set)
    high_priority_by_rc: dict[str, int] = defaultdict(int)
    high_priority_kinds = frozenset(
        priority_order[:high_priority_cutoff])
    bundle_size = len(bundle)
    top_priority_implied_rc = ""
    has_top_priority = False
    top_priority_kind = (priority_order[0]
                          if priority_order else "")
    for c in bundle:
        md = c.metadata_dict()
        k = md.get("claim_kind")
        src = md.get("source_role") or ""
        if not isinstance(k, str):
            continue
        rc = claim_to_root_cause.get(k)
        if rc:
            votes_by_rc[rc] += 1
            if src:
                sources_by_rc[rc].add(src)
            if k in high_priority_kinds:
                high_priority_by_rc[rc] += 1
            if k == top_priority_kind:
                has_top_priority = True
                top_priority_implied_rc = rc
    return {
        "votes_by_rc": dict(votes_by_rc),
        "sources_by_rc": {r: set(s) for r, s
                           in sources_by_rc.items()},
        "high_priority_by_rc": dict(high_priority_by_rc),
        "bundle_size": bundle_size,
        "has_top_priority_in_bundle": has_top_priority,
        "top_priority_implied_rc": top_priority_implied_rc,
        "high_priority_kinds": high_priority_kinds,
    }


def _featurise_bundle_for_rc_v2(
        bundle: Sequence[ContextCapsule],
        rc: str,
        claim_to_root_cause: dict[str, str],
        priority_order: tuple[str, ...],
        high_priority_cutoff: int = DEFAULT_HIGH_PRIORITY_CUTOFF,
        summary: dict[str, Any] | None = None,
        ) -> dict[str, float]:
    """Compute V2 shape features of ``bundle`` for candidate rc.

    V1 features (10) are reproduced verbatim for comparability
    with Phase 48 results; V2 adds 10 domain-invariant relative
    features.  See ``BUNDLE_DECODER_FEATURES_V2`` for the ordered
    list.
    """
    if summary is None:
        summary = _bundle_vote_summary(
            bundle, claim_to_root_cause, priority_order,
            high_priority_cutoff)
    votes_by_rc = summary["votes_by_rc"]
    sources_by_rc = summary["sources_by_rc"]
    high_priority_by_rc = summary["high_priority_by_rc"]
    bundle_size = summary["bundle_size"]
    top_priority_implied = summary["top_priority_implied_rc"]
    has_top_priority = summary["has_top_priority_in_bundle"]
    high_priority_kinds = summary["high_priority_kinds"]

    # Per-rc raw counts.
    votes = votes_by_rc.get(rc, 0)
    srcs = sources_by_rc.get(rc, set())
    n_sources = len(srcs)
    high_priority_votes = high_priority_by_rc.get(rc, 0)
    # V1 aggregated features.
    total_votes = sum(votes_by_rc.values())
    votes_share = (votes / total_votes) if total_votes else 0.0
    frac_bundle_implies_rc = (votes / bundle_size) if bundle_size else 0.0
    # has_top_priority_kind: does the top-priority kind (priority_order[0])
    # imply this rc?
    has_top_priority_kind = (1.0 if (top_priority_implied == rc
                                      and top_priority_implied)
                              else 0.0)
    multi_source_flag = 1.0 if n_sources >= 2 else 0.0
    # has_multi_source_kind: any single claim_kind implying rc has
    # >= 2 distinct sources.
    per_kind_sources: dict[str, set[str]] = defaultdict(set)
    for c in bundle:
        md = c.metadata_dict()
        k = md.get("claim_kind")
        src = md.get("source_role") or ""
        if not isinstance(k, str) or not src:
            continue
        if claim_to_root_cause.get(k) == rc:
            per_kind_sources[k].add(src)
    has_multi_source_kind = 1.0 if any(
        len(s) >= 2 for s in per_kind_sources.values()) else 0.0
    # lone_top_priority_flag: rc implied by a top-priority kind
    # emitted by <=1 source.
    lone_top_priority = 0.0
    for k, ks in per_kind_sources.items():
        if k in high_priority_kinds and len(ks) <= 1:
            lone_top_priority = 1.0
            break
    zero_vote_flag = 1.0 if votes == 0 else 0.0

    # V2 relative features.
    # Max over OTHER rcs.
    other_votes = [v for r, v in votes_by_rc.items() if r != rc]
    max_other_votes = max(other_votes) if other_votes else 0
    votes_minus_max_other = float(votes - max_other_votes)
    other_srcs = [len(s) for r, s in sources_by_rc.items() if r != rc]
    max_other_sources = max(other_srcs) if other_srcs else 0
    sources_minus_max_other = float(n_sources - max_other_sources)
    other_hp = [v for r, v in high_priority_by_rc.items() if r != rc]
    max_other_hp = max(other_hp) if other_hp else 0
    high_priority_minus_max_other = float(
        high_priority_votes - max_other_hp)
    # Strict / co-top by votes.
    is_strict_top_by_votes = 1.0 if (
        votes > 0 and votes > max_other_votes) else 0.0
    is_co_top_by_votes = 1.0 if (
        votes > 0 and votes >= max_other_votes) else 0.0
    is_strict_top_by_sources = 1.0 if (
        n_sources > 0 and n_sources > max_other_sources) else 0.0
    log1p_bundle_size = math.log1p(bundle_size)
    # Top-priority implies some other rc (adversarial signal).
    top_priority_implies_other_rc = 1.0 if (
        has_top_priority and top_priority_implied != rc) else 0.0
    # Fraction of high-priority votes that support this rc.
    total_hp = sum(high_priority_by_rc.values())
    frac_high_priority_for_rc = (
        high_priority_votes / total_hp) if total_hp else 0.0

    return {
        # V1 ----------------------------------------------------
        "bias": 1.0,
        "log1p_votes": math.log1p(votes),
        "log1p_sources": math.log1p(n_sources),
        "votes_share": votes_share,
        "high_priority_votes": float(high_priority_votes),
        "has_top_priority_kind": has_top_priority_kind,
        "multi_source_flag": multi_source_flag,
        "has_multi_source_kind": has_multi_source_kind,
        "lone_top_priority_flag": lone_top_priority,
        "zero_vote_flag": zero_vote_flag,
        # V2 ----------------------------------------------------
        "votes_minus_max_other": votes_minus_max_other,
        "sources_minus_max_other": sources_minus_max_other,
        "high_priority_minus_max_other":
            high_priority_minus_max_other,
        "is_strict_top_by_votes": is_strict_top_by_votes,
        "is_co_top_by_votes": is_co_top_by_votes,
        "is_strict_top_by_sources": is_strict_top_by_sources,
        "frac_bundle_implies_rc": frac_bundle_implies_rc,
        "log1p_bundle_size": log1p_bundle_size,
        "top_priority_implies_other_rc":
            top_priority_implies_other_rc,
        "frac_high_priority_for_rc": frac_high_priority_for_rc,
    }


def _feature_vector_v2(
        bundle: Sequence[ContextCapsule],
        rc: str,
        claim_to_root_cause: dict[str, str],
        priority_order: tuple[str, ...],
        high_priority_cutoff: int = DEFAULT_HIGH_PRIORITY_CUTOFF,
        summary: dict[str, Any] | None = None,
        ) -> np.ndarray:
    f = _featurise_bundle_for_rc_v2(
        bundle, rc, claim_to_root_cause, priority_order,
        high_priority_cutoff, summary=summary)
    return np.array([f[k] for k in BUNDLE_DECODER_FEATURES_V2],
                      dtype=np.float64)


# =============================================================================
# LearnedBundleDecoderV2 — linear over V2 features
# =============================================================================


@dataclasses.dataclass
class LearnedBundleDecoderV2(BundleDecoder):
    """Linear multinomial-logistic decoder over the V2 feature
    vocabulary.  Drop-in replacement for Phase-48
    ``LearnedBundleDecoder`` with 10 additional domain-invariant
    relative features."""

    weights: dict[str, float] = dataclasses.field(default_factory=dict)
    rc_alphabet: tuple[str, ...] = ()
    claim_to_root_cause: dict[str, str] = dataclasses.field(
        default_factory=lambda: dict(DEFAULT_CLAIM_TO_ROOT_CAUSE))
    priority_order: tuple[str, ...] = DEFAULT_PRIORITY_ORDER
    high_priority_cutoff: int = DEFAULT_HIGH_PRIORITY_CUTOFF
    unknown_label: str = UNKNOWN

    name: str = "learned_bundle_decoder_v2"

    def _score(self, bundle, rc, summary=None):
        f = _featurise_bundle_for_rc_v2(
            bundle, rc, self.claim_to_root_cause,
            self.priority_order, self.high_priority_cutoff,
            summary=summary)
        return sum(self.weights.get(k, 0.0) * v for k, v in f.items())

    def decode(self, admitted):
        if not self.rc_alphabet:
            return self.unknown_label
        summary = _bundle_vote_summary(
            admitted, self.claim_to_root_cause,
            self.priority_order, self.high_priority_cutoff)
        scores = {rc: self._score(admitted, rc, summary=summary)
                   for rc in self.rc_alphabet}
        best_rc = self.rc_alphabet[0]
        best_s = scores[best_rc]
        for rc in self.rc_alphabet:
            s = scores[rc]
            if s > best_s or (s == best_s and rc < best_rc):
                best_rc, best_s = rc, s
        return best_rc

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "weights": dict(self.weights),
            "rc_alphabet": list(self.rc_alphabet),
            "claim_to_root_cause": dict(self.claim_to_root_cause),
            "priority_order": list(self.priority_order),
            "high_priority_cutoff": int(self.high_priority_cutoff),
        }


def train_learned_bundle_decoder_v2(
        examples: Sequence[tuple[Sequence[ContextCapsule], str]],
        *,
        rc_alphabet: Sequence[str],
        claim_to_root_cause: dict[str, str],
        priority_order: tuple[str, ...] = DEFAULT_PRIORITY_ORDER,
        high_priority_cutoff: int = DEFAULT_HIGH_PRIORITY_CUTOFF,
        n_epochs: int = 300,
        lr: float = 0.5,
        l2: float = 1e-3,
        seed: int = 0,
        ) -> LearnedBundleDecoderV2:
    """Train a ``LearnedBundleDecoderV2`` via full-batch multinomial
    logistic regression (numpy-vectorised)."""
    if not examples:
        raise ValueError("train_learned_bundle_decoder_v2: no examples")
    alphabet = tuple(rc_alphabet)
    d = len(BUNDLE_DECODER_FEATURES_V2)
    # Cache feature tensors: X[i, rc_index, :] ∈ R^d.
    n = len(examples)
    K = len(alphabet)
    X = np.zeros((n, K, d), dtype=np.float64)
    y = np.zeros(n, dtype=np.int64)
    for i, (bundle, gold_rc) in enumerate(examples):
        summary = _bundle_vote_summary(
            bundle, claim_to_root_cause, priority_order,
            high_priority_cutoff)
        for j, rc in enumerate(alphabet):
            X[i, j, :] = _feature_vector_v2(
                bundle, rc, claim_to_root_cause, priority_order,
                high_priority_cutoff, summary=summary)
        # Gold idx (fallback: -1 → no gradient contribution).
        if gold_rc in alphabet:
            y[i] = alphabet.index(gold_rc)
        else:
            y[i] = -1
    valid = y >= 0
    # Shared weight vector w ∈ R^d.
    w = np.zeros(d, dtype=np.float64)
    for ep in range(n_epochs):
        # Scores[i, j] = sum_k w_k X[i, j, k].
        scores = X @ w  # shape (n, K)
        # Softmax per example.
        m = scores.max(axis=1, keepdims=True)
        ez = np.exp(scores - m)
        probs = ez / ez.sum(axis=1, keepdims=True)
        # Gradient: sum over valid i of X[i, j, :] * (probs[i, j] -
        # 1{j == y[i]}).
        grad = np.zeros(d, dtype=np.float64)
        for i in np.where(valid)[0]:
            pi = probs[i]
            tgt = np.zeros(K)
            tgt[y[i]] = 1.0
            err = pi - tgt  # (K,)
            grad += X[i].T @ err  # (d, K) @ (K,) → (d,)
        grad = grad / max(1, valid.sum()) + l2 * w
        w -= lr * grad
    weights = {k: float(w[i])
                for i, k in enumerate(BUNDLE_DECODER_FEATURES_V2)}
    return LearnedBundleDecoderV2(
        weights=weights,
        rc_alphabet=alphabet,
        claim_to_root_cause=dict(claim_to_root_cause),
        priority_order=tuple(priority_order),
        high_priority_cutoff=int(high_priority_cutoff),
    )


# =============================================================================
# InteractionBundleDecoder — linear + pairwise feature crosses
# =============================================================================


def _interaction_feature_names(
        base: tuple[str, ...] = BUNDLE_DECODER_FEATURES_V2,
        ) -> tuple[str, ...]:
    """Named pairwise-interaction feature vocabulary.  Base
    features plus all unordered pairs (i < j) of non-bias
    features.  ``bias`` itself is the intercept; we do NOT cross
    bias with itself.
    """
    out: list[str] = list(base)
    non_bias = [f for f in base if f != "bias"]
    for i, a in enumerate(non_bias):
        for b in non_bias[i+1:]:
            out.append(f"cross:{a}*{b}")
    return tuple(out)


INTERACTION_FEATURES: tuple[str, ...] = _interaction_feature_names()


def _interaction_vector(
        bundle: Sequence[ContextCapsule],
        rc: str,
        claim_to_root_cause: dict[str, str],
        priority_order: tuple[str, ...],
        high_priority_cutoff: int = DEFAULT_HIGH_PRIORITY_CUTOFF,
        summary: dict[str, Any] | None = None,
        ) -> np.ndarray:
    base = _feature_vector_v2(
        bundle, rc, claim_to_root_cause, priority_order,
        high_priority_cutoff, summary=summary)
    non_bias_idx = [i for i, k
                     in enumerate(BUNDLE_DECODER_FEATURES_V2)
                     if k != "bias"]
    d = len(BUNDLE_DECODER_FEATURES_V2)
    out = np.zeros(len(INTERACTION_FEATURES), dtype=np.float64)
    out[:d] = base
    k = d
    for i, ia in enumerate(non_bias_idx):
        for ib in non_bias_idx[i+1:]:
            out[k] = base[ia] * base[ib]
            k += 1
    return out


@dataclasses.dataclass
class InteractionBundleDecoder(BundleDecoder):
    """Linear multinomial-logistic decoder over V2 features + all
    pairwise interactions.  Captures conjunctive signatures that a
    pure-linear decoder cannot express."""

    weights: dict[str, float] = dataclasses.field(default_factory=dict)
    rc_alphabet: tuple[str, ...] = ()
    claim_to_root_cause: dict[str, str] = dataclasses.field(
        default_factory=lambda: dict(DEFAULT_CLAIM_TO_ROOT_CAUSE))
    priority_order: tuple[str, ...] = DEFAULT_PRIORITY_ORDER
    high_priority_cutoff: int = DEFAULT_HIGH_PRIORITY_CUTOFF
    unknown_label: str = UNKNOWN

    name: str = "interaction_bundle_decoder"

    def _score(self, bundle, rc, summary=None):
        v = _interaction_vector(
            bundle, rc, self.claim_to_root_cause,
            self.priority_order, self.high_priority_cutoff,
            summary=summary)
        w = np.array([self.weights.get(k, 0.0)
                       for k in INTERACTION_FEATURES],
                      dtype=np.float64)
        return float(v @ w)

    def decode(self, admitted):
        if not self.rc_alphabet:
            return self.unknown_label
        summary = _bundle_vote_summary(
            admitted, self.claim_to_root_cause,
            self.priority_order, self.high_priority_cutoff)
        scores = {rc: self._score(admitted, rc, summary=summary)
                   for rc in self.rc_alphabet}
        best_rc = self.rc_alphabet[0]
        best_s = scores[best_rc]
        for rc in self.rc_alphabet:
            s = scores[rc]
            if s > best_s or (s == best_s and rc < best_rc):
                best_rc, best_s = rc, s
        return best_rc

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "weights": dict(self.weights),
            "rc_alphabet": list(self.rc_alphabet),
            "claim_to_root_cause": dict(self.claim_to_root_cause),
            "priority_order": list(self.priority_order),
            "high_priority_cutoff": int(self.high_priority_cutoff),
        }


def train_interaction_bundle_decoder(
        examples: Sequence[tuple[Sequence[ContextCapsule], str]],
        *,
        rc_alphabet: Sequence[str],
        claim_to_root_cause: dict[str, str],
        priority_order: tuple[str, ...] = DEFAULT_PRIORITY_ORDER,
        high_priority_cutoff: int = DEFAULT_HIGH_PRIORITY_CUTOFF,
        n_epochs: int = 300,
        lr: float = 0.5,
        l2: float = 1e-2,
        seed: int = 0,
        ) -> InteractionBundleDecoder:
    if not examples:
        raise ValueError(
            "train_interaction_bundle_decoder: no examples")
    alphabet = tuple(rc_alphabet)
    D = len(INTERACTION_FEATURES)
    n = len(examples)
    K = len(alphabet)
    X = np.zeros((n, K, D), dtype=np.float64)
    y = np.zeros(n, dtype=np.int64)
    for i, (bundle, gold_rc) in enumerate(examples):
        summary = _bundle_vote_summary(
            bundle, claim_to_root_cause, priority_order,
            high_priority_cutoff)
        for j, rc in enumerate(alphabet):
            X[i, j, :] = _interaction_vector(
                bundle, rc, claim_to_root_cause,
                priority_order, high_priority_cutoff,
                summary=summary)
        y[i] = (alphabet.index(gold_rc) if gold_rc in alphabet
                 else -1)
    valid = y >= 0
    w = np.zeros(D, dtype=np.float64)
    for ep in range(n_epochs):
        scores = X @ w
        m = scores.max(axis=1, keepdims=True)
        ez = np.exp(scores - m)
        probs = ez / ez.sum(axis=1, keepdims=True)
        grad = np.zeros(D, dtype=np.float64)
        for i in np.where(valid)[0]:
            tgt = np.zeros(K)
            tgt[y[i]] = 1.0
            grad += X[i].T @ (probs[i] - tgt)
        grad = grad / max(1, valid.sum()) + l2 * w
        w -= lr * grad
    weights = {k: float(w[i])
                for i, k in enumerate(INTERACTION_FEATURES)}
    return InteractionBundleDecoder(
        weights=weights,
        rc_alphabet=alphabet,
        claim_to_root_cause=dict(claim_to_root_cause),
        priority_order=tuple(priority_order),
        high_priority_cutoff=int(high_priority_cutoff),
    )


# =============================================================================
# MLPBundleDecoder — small MLP over V2 aggregated features
# =============================================================================


@dataclasses.dataclass
class MLPBundleDecoder(BundleDecoder):
    """One-hidden-layer MLP over V2 aggregated features, shared
    across rc.  Architecture: x ∈ R^d → h = tanh(W1 x + b1) ∈ R^H
    → s = w2 · h + b2 ∈ R.  Default H = 8.

    Weights are stored as numpy arrays; ``to_dict`` serialises
    them as nested lists for JSON portability.
    """

    W1: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros((1, 1)))
    b1: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(1))
    w2: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(1))
    b2: float = 0.0
    rc_alphabet: tuple[str, ...] = ()
    claim_to_root_cause: dict[str, str] = dataclasses.field(
        default_factory=lambda: dict(DEFAULT_CLAIM_TO_ROOT_CAUSE))
    priority_order: tuple[str, ...] = DEFAULT_PRIORITY_ORDER
    high_priority_cutoff: int = DEFAULT_HIGH_PRIORITY_CUTOFF
    unknown_label: str = UNKNOWN
    hidden_size: int = 8

    name: str = "mlp_bundle_decoder"

    def _score(self, bundle, rc, summary=None):
        x = _feature_vector_v2(
            bundle, rc, self.claim_to_root_cause,
            self.priority_order, self.high_priority_cutoff,
            summary=summary)
        h = np.tanh(self.W1 @ x + self.b1)
        return float(self.w2 @ h + self.b2)

    def decode(self, admitted):
        if not self.rc_alphabet:
            return self.unknown_label
        summary = _bundle_vote_summary(
            admitted, self.claim_to_root_cause,
            self.priority_order, self.high_priority_cutoff)
        scores = {rc: self._score(admitted, rc, summary=summary)
                   for rc in self.rc_alphabet}
        best_rc = self.rc_alphabet[0]
        best_s = scores[best_rc]
        for rc in self.rc_alphabet:
            s = scores[rc]
            if s > best_s or (s == best_s and rc < best_rc):
                best_rc, best_s = rc, s
        return best_rc

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "W1": self.W1.tolist(),
            "b1": self.b1.tolist(),
            "w2": self.w2.tolist(),
            "b2": float(self.b2),
            "hidden_size": int(self.hidden_size),
            "rc_alphabet": list(self.rc_alphabet),
            "claim_to_root_cause": dict(self.claim_to_root_cause),
            "priority_order": list(self.priority_order),
            "high_priority_cutoff": int(self.high_priority_cutoff),
        }


def train_mlp_bundle_decoder(
        examples: Sequence[tuple[Sequence[ContextCapsule], str]],
        *,
        rc_alphabet: Sequence[str],
        claim_to_root_cause: dict[str, str],
        priority_order: tuple[str, ...] = DEFAULT_PRIORITY_ORDER,
        high_priority_cutoff: int = DEFAULT_HIGH_PRIORITY_CUTOFF,
        hidden_size: int = 8,
        n_epochs: int = 400,
        lr: float = 0.1,
        l2: float = 1e-3,
        seed: int = 0,
        ) -> MLPBundleDecoder:
    """Train an ``MLPBundleDecoder`` by full-batch softmax-cross-
    entropy over rc-indexed per-example feature tensors."""
    if not examples:
        raise ValueError("train_mlp_bundle_decoder: no examples")
    alphabet = tuple(rc_alphabet)
    d = len(BUNDLE_DECODER_FEATURES_V2)
    H = int(hidden_size)
    n = len(examples)
    K = len(alphabet)
    X = np.zeros((n, K, d), dtype=np.float64)
    y = np.zeros(n, dtype=np.int64)
    for i, (bundle, gold_rc) in enumerate(examples):
        summary = _bundle_vote_summary(
            bundle, claim_to_root_cause, priority_order,
            high_priority_cutoff)
        for j, rc in enumerate(alphabet):
            X[i, j, :] = _feature_vector_v2(
                bundle, rc, claim_to_root_cause,
                priority_order, high_priority_cutoff,
                summary=summary)
        y[i] = (alphabet.index(gold_rc) if gold_rc in alphabet
                 else -1)
    valid = y >= 0
    rng = np.random.RandomState(seed)
    W1 = rng.randn(H, d) * 0.1
    b1 = np.zeros(H)
    w2 = rng.randn(H) * 0.1
    b2 = 0.0
    n_valid = max(1, int(valid.sum()))
    for ep in range(n_epochs):
        # Forward.
        # X: (n, K, d). W1: (H, d). Pre: (n, K, H).
        pre = np.einsum("nkd,hd->nkh", X, W1) + b1
        h = np.tanh(pre)                          # (n, K, H)
        scores = h @ w2 + b2                      # (n, K)
        m = scores.max(axis=1, keepdims=True)
        ez = np.exp(scores - m)
        probs = ez / ez.sum(axis=1, keepdims=True)
        # Backward — softmax-cross-entropy.
        grad_scores = probs.copy()
        for i in np.where(valid)[0]:
            grad_scores[i, y[i]] -= 1.0
        grad_scores[~valid] = 0.0
        grad_scores /= n_valid
        # grad_w2: sum_{n,k} grad_scores[n,k] * h[n,k,:]
        grad_w2 = np.einsum("nk,nkh->h", grad_scores, h)
        grad_b2 = grad_scores.sum()
        # grad_h = grad_scores * w2 (broadcast)
        grad_h = grad_scores[..., None] * w2     # (n, K, H)
        grad_pre = grad_h * (1.0 - h * h)        # tanh derivative
        # grad_W1: sum_{n,k} grad_pre[n,k,:] outer X[n,k,:]
        grad_W1 = np.einsum("nkh,nkd->hd", grad_pre, X)
        grad_b1 = grad_pre.sum(axis=(0, 1))
        # L2 regularisation.
        grad_W1 += l2 * W1
        grad_w2 += l2 * w2
        # SGD step.
        W1 -= lr * grad_W1
        b1 -= lr * grad_b1
        w2 -= lr * grad_w2
        b2 -= lr * grad_b2
    return MLPBundleDecoder(
        W1=W1, b1=b1, w2=w2, b2=float(b2),
        hidden_size=H,
        rc_alphabet=alphabet,
        claim_to_root_cause=dict(claim_to_root_cause),
        priority_order=tuple(priority_order),
        high_priority_cutoff=int(high_priority_cutoff),
    )


# =============================================================================
# DeepSetBundleDecoder — per-capsule φ + ρ MLP on aggregated
# =============================================================================


# Per-capsule features φ(c, rc) used by DeepSet.  Dim = 8.  Each
# feature is a {0, 1} indicator or a bounded scalar — the sum
# over a bundle is ≤ bundle_size, keeping the aggregated vector
# on the same scale as the V1 aggregated features.
DEEPSET_PHI_FEATURES: tuple[str, ...] = (
    "phi:implies_rc",
    "phi:implies_rc_and_top_priority",
    "phi:implies_rc_and_high_priority",
    "phi:implies_rc_and_log1p_tokens",
    "phi:is_top_priority_but_not_rc",
    "phi:is_high_priority_but_not_rc",
    "phi:is_known_kind",
    "phi:implies_rc_and_unique_source",
)


def _phi_capsule(
        c: ContextCapsule,
        rc: str,
        claim_to_root_cause: dict[str, str],
        priority_order: tuple[str, ...],
        high_priority_cutoff: int = DEFAULT_HIGH_PRIORITY_CUTOFF,
        source_role_counts: dict[str, int] | None = None,
        ) -> np.ndarray:
    """Per-capsule embedding φ(c, rc) ∈ R^8 used by DeepSet."""
    md = c.metadata_dict()
    k = md.get("claim_kind")
    src = md.get("source_role") or ""
    if not isinstance(k, str):
        return np.zeros(len(DEEPSET_PHI_FEATURES), dtype=np.float64)
    high_priority_kinds = frozenset(
        priority_order[:high_priority_cutoff])
    implied_rc = claim_to_root_cause.get(k, "")
    implies = 1.0 if implied_rc == rc else 0.0
    top_priority = priority_order[0] if priority_order else ""
    is_top = 1.0 if k == top_priority else 0.0
    is_high = 1.0 if k in high_priority_kinds else 0.0
    is_known = 1.0 if implied_rc else 0.0
    # Unique-source flag: 1 if src is the only source of the
    # capsule's claim_kind in the bundle.
    unique_source = 0.0
    if source_role_counts is not None:
        cnt = source_role_counts.get(k, 0)
        if cnt == 1:
            unique_source = 1.0
    return np.array([
        implies,
        implies * is_top,
        implies * is_high,
        implies * math.log1p(c.n_tokens or 0),
        (1.0 - implies) * is_top,
        (1.0 - implies) * is_high,
        is_known,
        implies * unique_source,
    ], dtype=np.float64)


def _phi_sum(
        bundle: Sequence[ContextCapsule],
        rc: str,
        claim_to_root_cause: dict[str, str],
        priority_order: tuple[str, ...],
        high_priority_cutoff: int = DEFAULT_HIGH_PRIORITY_CUTOFF,
        ) -> np.ndarray:
    """Sum over the bundle of φ(c, rc) — Deep Sets aggregator."""
    # Compute per-kind source counts once.
    per_kind_sources: dict[str, set[str]] = defaultdict(set)
    for c in bundle:
        md = c.metadata_dict()
        k = md.get("claim_kind")
        src = md.get("source_role") or ""
        if isinstance(k, str) and src:
            per_kind_sources[k].add(src)
    source_role_counts = {k: len(s) for k, s in per_kind_sources.items()}
    out = np.zeros(len(DEEPSET_PHI_FEATURES), dtype=np.float64)
    for c in bundle:
        out += _phi_capsule(
            c, rc, claim_to_root_cause, priority_order,
            high_priority_cutoff, source_role_counts=source_role_counts)
    return out


@dataclasses.dataclass
class DeepSetBundleDecoder(BundleDecoder):
    """Deep Sets decoder: per-capsule embedding φ(c, rc) summed
    over the bundle, concatenated with V2 aggregated features g,
    scored through a 1-hidden-layer MLP shared across rc.

    Input: x = concat(phi_sum, g_v2) ∈ R^{d_phi + d_v2} = R^{28}.
    Hidden: h = tanh(W1 x + b1) ∈ R^H (default H = 10).
    Output: s = w2 · h + b2 ∈ R.
    Predict: argmax_rc s.
    """

    W1: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros((1, 1)))
    b1: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(1))
    w2: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(1))
    b2: float = 0.0
    rc_alphabet: tuple[str, ...] = ()
    claim_to_root_cause: dict[str, str] = dataclasses.field(
        default_factory=lambda: dict(DEFAULT_CLAIM_TO_ROOT_CAUSE))
    priority_order: tuple[str, ...] = DEFAULT_PRIORITY_ORDER
    high_priority_cutoff: int = DEFAULT_HIGH_PRIORITY_CUTOFF
    unknown_label: str = UNKNOWN
    hidden_size: int = 10

    name: str = "deep_set_bundle_decoder"

    def _input_vector(self, bundle, rc, summary=None):
        phi = _phi_sum(
            bundle, rc, self.claim_to_root_cause,
            self.priority_order, self.high_priority_cutoff)
        g = _feature_vector_v2(
            bundle, rc, self.claim_to_root_cause,
            self.priority_order, self.high_priority_cutoff,
            summary=summary)
        return np.concatenate([phi, g])

    def _score(self, bundle, rc, summary=None):
        x = self._input_vector(bundle, rc, summary=summary)
        h = np.tanh(self.W1 @ x + self.b1)
        return float(self.w2 @ h + self.b2)

    def decode(self, admitted):
        if not self.rc_alphabet:
            return self.unknown_label
        summary = _bundle_vote_summary(
            admitted, self.claim_to_root_cause,
            self.priority_order, self.high_priority_cutoff)
        scores = {rc: self._score(admitted, rc, summary=summary)
                   for rc in self.rc_alphabet}
        best_rc = self.rc_alphabet[0]
        best_s = scores[best_rc]
        for rc in self.rc_alphabet:
            s = scores[rc]
            if s > best_s or (s == best_s and rc < best_rc):
                best_rc, best_s = rc, s
        return best_rc

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "W1": self.W1.tolist(),
            "b1": self.b1.tolist(),
            "w2": self.w2.tolist(),
            "b2": float(self.b2),
            "hidden_size": int(self.hidden_size),
            "phi_features": list(DEEPSET_PHI_FEATURES),
            "v2_features": list(BUNDLE_DECODER_FEATURES_V2),
            "rc_alphabet": list(self.rc_alphabet),
            "claim_to_root_cause": dict(self.claim_to_root_cause),
            "priority_order": list(self.priority_order),
            "high_priority_cutoff": int(self.high_priority_cutoff),
        }


def train_deep_set_bundle_decoder(
        examples: Sequence[tuple[Sequence[ContextCapsule], str]],
        *,
        rc_alphabet: Sequence[str],
        claim_to_root_cause: dict[str, str],
        priority_order: tuple[str, ...] = DEFAULT_PRIORITY_ORDER,
        high_priority_cutoff: int = DEFAULT_HIGH_PRIORITY_CUTOFF,
        hidden_size: int = 10,
        n_epochs: int = 500,
        lr: float = 0.1,
        l2: float = 1e-3,
        seed: int = 0,
        ) -> DeepSetBundleDecoder:
    """Train a ``DeepSetBundleDecoder`` by full-batch softmax-cross-
    entropy.  Architecture: per-capsule φ → sum → concat with V2
    aggregated → MLP → scalar score per rc."""
    if not examples:
        raise ValueError(
            "train_deep_set_bundle_decoder: no examples")
    alphabet = tuple(rc_alphabet)
    d_phi = len(DEEPSET_PHI_FEATURES)
    d_v2 = len(BUNDLE_DECODER_FEATURES_V2)
    d = d_phi + d_v2
    H = int(hidden_size)
    n = len(examples)
    K = len(alphabet)
    X = np.zeros((n, K, d), dtype=np.float64)
    y = np.zeros(n, dtype=np.int64)
    for i, (bundle, gold_rc) in enumerate(examples):
        summary = _bundle_vote_summary(
            bundle, claim_to_root_cause, priority_order,
            high_priority_cutoff)
        for j, rc in enumerate(alphabet):
            phi = _phi_sum(bundle, rc, claim_to_root_cause,
                            priority_order, high_priority_cutoff)
            g = _feature_vector_v2(
                bundle, rc, claim_to_root_cause,
                priority_order, high_priority_cutoff,
                summary=summary)
            X[i, j, :] = np.concatenate([phi, g])
        y[i] = (alphabet.index(gold_rc) if gold_rc in alphabet
                 else -1)
    valid = y >= 0
    rng = np.random.RandomState(seed)
    W1 = rng.randn(H, d) * 0.1
    b1 = np.zeros(H)
    w2 = rng.randn(H) * 0.1
    b2 = 0.0
    n_valid = max(1, int(valid.sum()))
    for ep in range(n_epochs):
        pre = np.einsum("nkd,hd->nkh", X, W1) + b1
        h = np.tanh(pre)
        scores = h @ w2 + b2
        m = scores.max(axis=1, keepdims=True)
        ez = np.exp(scores - m)
        probs = ez / ez.sum(axis=1, keepdims=True)
        grad_scores = probs.copy()
        for i in np.where(valid)[0]:
            grad_scores[i, y[i]] -= 1.0
        grad_scores[~valid] = 0.0
        grad_scores /= n_valid
        grad_w2 = np.einsum("nk,nkh->h", grad_scores, h)
        grad_b2 = grad_scores.sum()
        grad_h = grad_scores[..., None] * w2
        grad_pre = grad_h * (1.0 - h * h)
        grad_W1 = np.einsum("nkh,nkd->hd", grad_pre, X)
        grad_b1 = grad_pre.sum(axis=(0, 1))
        grad_W1 += l2 * W1
        grad_w2 += l2 * w2
        W1 -= lr * grad_W1
        b1 -= lr * grad_b1
        w2 -= lr * grad_w2
        b2 -= lr * grad_b2
    return DeepSetBundleDecoder(
        W1=W1, b1=b1, w2=w2, b2=float(b2),
        hidden_size=H,
        rc_alphabet=alphabet,
        claim_to_root_cause=dict(claim_to_root_cause),
        priority_order=tuple(priority_order),
        high_priority_cutoff=int(high_priority_cutoff),
    )


# =============================================================================
# MultitaskBundleDecoder — pooled training, domain-aware head
# =============================================================================


@dataclasses.dataclass
class MultitaskBundleDecoder(BundleDecoder):
    """Linear decoder with a domain tag ``d`` selecting a
    domain-specific head.  Effective weights:
    ``w_effective = w_shared + w_domain[d]``.

    Trained on a pooled dataset of (bundle, gold_label, domain)
    triples.  At decode time the caller supplies the domain tag
    (passed through ``set_domain``).  The shared head learns
    task-family-invariant signals; the per-domain heads learn
    domain-specific calibration (e.g. the
    ``lone_top_priority_flag`` sign-flip between incident and
    security).
    """

    w_shared: dict[str, float] = dataclasses.field(default_factory=dict)
    w_domain: dict[str, dict[str, float]] = dataclasses.field(
        default_factory=dict)
    domain_tag: str = ""  # set before decoding
    rc_alphabet: tuple[str, ...] = ()
    # Domain→rc_alphabet mapping is only used when the caller switches
    # rc_alphabet + claim_to_root_cause between domains at decode time.
    claim_to_root_cause: dict[str, str] = dataclasses.field(
        default_factory=lambda: dict(DEFAULT_CLAIM_TO_ROOT_CAUSE))
    priority_order: tuple[str, ...] = DEFAULT_PRIORITY_ORDER
    high_priority_cutoff: int = DEFAULT_HIGH_PRIORITY_CUTOFF
    unknown_label: str = UNKNOWN

    name: str = "multitask_bundle_decoder"

    def set_domain(self, domain: str,
                   rc_alphabet: Sequence[str],
                   claim_to_root_cause: dict[str, str],
                   priority_order: tuple[str, ...]) -> "MultitaskBundleDecoder":
        return dataclasses.replace(
            self,
            domain_tag=domain,
            rc_alphabet=tuple(rc_alphabet),
            claim_to_root_cause=dict(claim_to_root_cause),
            priority_order=tuple(priority_order),
        )

    def _score(self, bundle, rc, summary=None):
        f = _featurise_bundle_for_rc_v2(
            bundle, rc, self.claim_to_root_cause,
            self.priority_order, self.high_priority_cutoff,
            summary=summary)
        d_head = self.w_domain.get(self.domain_tag, {})
        z = 0.0
        for k, v in f.items():
            z += (self.w_shared.get(k, 0.0) + d_head.get(k, 0.0)) * v
        return z

    def decode(self, admitted):
        if not self.rc_alphabet:
            return self.unknown_label
        summary = _bundle_vote_summary(
            admitted, self.claim_to_root_cause,
            self.priority_order, self.high_priority_cutoff)
        scores = {rc: self._score(admitted, rc, summary=summary)
                   for rc in self.rc_alphabet}
        best_rc = self.rc_alphabet[0]
        best_s = scores[best_rc]
        for rc in self.rc_alphabet:
            s = scores[rc]
            if s > best_s or (s == best_s and rc < best_rc):
                best_rc, best_s = rc, s
        return best_rc

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "w_shared": dict(self.w_shared),
            "w_domain": {d: dict(w) for d, w in self.w_domain.items()},
            "domain_tag": self.domain_tag,
            "rc_alphabet": list(self.rc_alphabet),
        }


def train_multitask_bundle_decoder(
        pooled_examples: Sequence[tuple[
            Sequence[ContextCapsule], str, str]],
        *,
        domain_specs: dict[str, dict[str, Any]],
        n_epochs: int = 400,
        lr: float = 0.3,
        l2_shared: float = 1e-3,
        l2_domain: float = 5e-3,
        seed: int = 0,
        ) -> MultitaskBundleDecoder:
    """Train a ``MultitaskBundleDecoder`` by pooled full-batch
    softmax-cross-entropy.

    ``pooled_examples`` is a sequence of
    ``(bundle, gold_label, domain_name)`` triples.

    ``domain_specs`` maps ``domain_name`` → dict with keys:
      * ``rc_alphabet``       — tuple[str, ...]
      * ``claim_to_root_cause`` — dict[str, str]
      * ``priority_order``    — tuple[str, ...]
      * ``high_priority_cutoff`` — int (optional)

    Heavier L2 on the per-domain head than the shared head
    (``l2_domain >> l2_shared``) pushes the classifier to place
    task-family-invariant signals in the shared head and only
    domain-specific corrections in the per-domain head.  This is
    the inductive bias that makes "symmetric transfer" a
    meaningful target.
    """
    if not pooled_examples:
        raise ValueError(
            "train_multitask_bundle_decoder: no examples")
    if not domain_specs:
        raise ValueError(
            "train_multitask_bundle_decoder: no domain_specs")
    domains = tuple(sorted(domain_specs.keys()))
    d = len(BUNDLE_DECODER_FEATURES_V2)

    # Per-domain training tensors.
    per_domain_X: dict[str, np.ndarray] = {}
    per_domain_y: dict[str, np.ndarray] = {}
    for dom in domains:
        specs = domain_specs[dom]
        alphabet = tuple(specs["rc_alphabet"])
        ctrc = dict(specs["claim_to_root_cause"])
        porder = tuple(specs["priority_order"])
        hpc = int(specs.get("high_priority_cutoff",
                               DEFAULT_HIGH_PRIORITY_CUTOFF))
        ex = [(b, g) for (b, g, D) in pooled_examples if D == dom]
        n = len(ex)
        K = len(alphabet)
        X = np.zeros((n, K, d), dtype=np.float64)
        y = np.zeros(n, dtype=np.int64)
        for i, (bundle, gold_rc) in enumerate(ex):
            summary = _bundle_vote_summary(
                bundle, ctrc, porder, hpc)
            for j, rc in enumerate(alphabet):
                X[i, j, :] = _feature_vector_v2(
                    bundle, rc, ctrc, porder, hpc,
                    summary=summary)
            y[i] = (alphabet.index(gold_rc)
                     if gold_rc in alphabet else -1)
        per_domain_X[dom] = X
        per_domain_y[dom] = y

    # Shared + per-domain weights.
    w_shared = np.zeros(d, dtype=np.float64)
    w_dom = {dom: np.zeros(d, dtype=np.float64) for dom in domains}
    for ep in range(n_epochs):
        # Forward + backward per domain; accumulate gradients.
        g_shared = np.zeros(d)
        g_dom = {dom: np.zeros(d) for dom in domains}
        total_valid = 0
        for dom in domains:
            X = per_domain_X[dom]
            y = per_domain_y[dom]
            if X.size == 0:
                continue
            w_eff = w_shared + w_dom[dom]
            scores = X @ w_eff
            m = scores.max(axis=1, keepdims=True)
            ez = np.exp(scores - m)
            probs = ez / ez.sum(axis=1, keepdims=True)
            valid = y >= 0
            for i in np.where(valid)[0]:
                tgt = np.zeros(scores.shape[1])
                tgt[y[i]] = 1.0
                err = probs[i] - tgt
                contrib = X[i].T @ err
                g_shared += contrib
                g_dom[dom] += contrib
                total_valid += 1
        total_valid = max(1, total_valid)
        g_shared = g_shared / total_valid + l2_shared * w_shared
        w_shared -= lr * g_shared
        for dom in domains:
            # Only normalise by THIS domain's valid count.
            n_dom = max(1, int((per_domain_y[dom] >= 0).sum()))
            g = g_dom[dom] / n_dom + l2_domain * w_dom[dom]
            w_dom[dom] -= lr * g
    return MultitaskBundleDecoder(
        w_shared={k: float(w_shared[i])
                    for i, k in enumerate(BUNDLE_DECODER_FEATURES_V2)},
        w_domain={
            dom: {k: float(w_dom[dom][i])
                    for i, k in enumerate(BUNDLE_DECODER_FEATURES_V2)}
            for dom in domains
        },
    )


# =============================================================================
# Public surface
# =============================================================================


__all__ = [
    "BUNDLE_DECODER_FEATURES_V2",
    "INTERACTION_FEATURES",
    "DEEPSET_PHI_FEATURES",
    "LearnedBundleDecoderV2", "train_learned_bundle_decoder_v2",
    "InteractionBundleDecoder", "train_interaction_bundle_decoder",
    "MLPBundleDecoder", "train_mlp_bundle_decoder",
    "DeepSetBundleDecoder", "train_deep_set_bundle_decoder",
    "MultitaskBundleDecoder", "train_multitask_bundle_decoder",
]
