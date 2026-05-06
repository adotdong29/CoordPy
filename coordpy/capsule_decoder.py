"""Phase 48 — bundle-aware capsule DECODING.

Phase 47 shipped bundle-aware *admission* (``capsule_policy_bundle``)
and cleanly falsified Conjecture P46-C1 in its strong form — no
bundle-aware admission policy beats the **structural** Phase-31
priority-decoder ceiling of 0.200 on the noise-poisoned test
bench. The ceiling is a property of the decoder, not of
admission: a first-match-priority rule over ``claim_kind`` outputs
``disk_fill`` whenever any ``DISK_FILL_CRITICAL`` is admitted,
and under ``spurious_prob = 0.30`` that happens in ≈ 100 %
of test scenarios regardless of admission.

This module ships the next step: the admitted set is the same;
only the *decoder* changes.  A bundle-aware decoder treats the
admitted capsules as a **set** whose implied root-causes compete
— the output is a function of the full structure (vote
distribution + source corroboration), not of the first high-
priority kind that happens to be present.  Three decoders:

  * ``PriorityDecoder``              — first-match over a
    priority order.  The status-quo Phase-31 decoder, surfaced
    as a policy-comparable object so the baseline is explicit.

  * ``PluralityDecoder``             — argmax over implied-
    ``root_cause`` vote counts; ties broken by ``priority_order``.
    The "follow the majority" decoder.  Breaks the ceiling on
    scenarios whose causal chain has ≥ 2 coherent claims
    (disk_fill / memory_leak / deadlock in Phase-31).

  * ``SourceCorroboratedPriorityDecoder`` — first-match over the
    priority order, but a ``claim_kind`` only counts if ≥
    ``min_sources`` distinct source roles in the admitted set
    emit it.  Kills lone spurious high-priority claims; leaves
    the decoder's mechanism otherwise unchanged.

  * ``LearnedBundleDecoder`` (+ ``train_learned_bundle_decoder``)
    — a small multinomial logistic-regression classifier over
    class-agnostic bundle-shape features.  Features per
    candidate ``root_cause`` are *structural* (votes for r,
    distinct sources for r, high-priority-vote count for r,
    lone-vote flag, zero-vote flag), so the same weight vector
    is meaningful across domains with different claim-kind
    alphabets.  This is the decoder's analogue of the Phase-46
    learned admission policy; it is the natural carrier for the
    decoder-side cross-domain transfer study.

Theorems anchored by this module (see ``docs/CAPSULE_FORMALISM.md``
§ 4.C):

  * W3-17 (admission locality — negative): no admission rule
    that does not change the decoder can exceed the decoder's
    first-match ceiling in expectation under bundle poisoning
    by high-priority spurious claims.
  * W3-18 (bundle-aware decoder sufficiency — conditional):
    for any bundle in the *coherent-majority* regime (where
    the true root-cause is supported by strictly more implied
    votes than any spurious root-cause), plurality decoding
    strictly dominates priority decoding; in particular, it
    recovers the gold ``root_cause`` even when a priority-
    breaking spurious high-priority claim is admitted.
  * W3-C6 (decoder-side task-family transfer — conjectural):
    class-agnostic bundle features carry a learned decoder
    across operational-detection domains sharing a role cast
    and scenario-archetype structure, and fail across task
    families with different scenario archetypes.  Mirrors
    Phase-47's admission-side P47-C3.

The decoder keeps the Capsule Contract C1..C6 untouched — it is
a *reader* of the admitted ledger, not a mutator.  No capsule
is created, admitted, sealed, or retired by any decoder in this
module.
"""

from __future__ import annotations

import dataclasses
import math
import random
from collections import defaultdict
from typing import Any, Callable, Iterable, Sequence

from .capsule import ContextCapsule
from .capsule_policy_bundle import (
    DEFAULT_CLAIM_TO_ROOT_CAUSE, DEFAULT_PRIORITY_ORDER,
    DEFAULT_HIGH_PRIORITY_CUTOFF,
)


# =============================================================================
# BundleDecoder interface
# =============================================================================


UNKNOWN = "unknown"


class BundleDecoder:
    """Decode an admitted capsule bundle into a root-cause label.

    The decoder reads a list of ``ContextCapsule`` (the admitted
    set) and returns a string (the inferred ``root_cause``).  It
    does NOT modify the ledger; decoders are read-only consumers
    of the admitted set.

    Subclasses implement ``decode``.  They may inspect each
    capsule's ``metadata_dict()`` (``claim_kind``, ``source_role``)
    but MUST NOT touch the payload by default — that would
    violate the "header-level only" discipline of the Phase 48
    research question.
    """

    name: str = "abstract"

    def decode(self, admitted: Sequence[ContextCapsule]) -> str:
        raise NotImplementedError

    def __call__(self, admitted: Sequence[ContextCapsule]) -> str:
        return self.decode(admitted)


def _claim_kinds(admitted: Sequence[ContextCapsule]) -> list[str]:
    """Return the ordered list of ``claim_kind`` strings present
    on admitted capsules.  Preserves multiplicity."""
    out: list[str] = []
    for c in admitted:
        md = c.metadata_dict()
        k = md.get("claim_kind")
        if isinstance(k, str):
            out.append(k)
    return out


def _source_per_kind(admitted: Sequence[ContextCapsule],
                     ) -> dict[str, set[str]]:
    """``claim_kind`` → set of distinct source roles in the admitted
    set emitting that kind."""
    out: dict[str, set[str]] = defaultdict(set)
    for c in admitted:
        md = c.metadata_dict()
        k = md.get("claim_kind")
        src = md.get("source_role") or ""
        if isinstance(k, str) and src:
            out[k].add(src)
    return out


def _sources_per_root_cause(
        admitted: Sequence[ContextCapsule],
        claim_to_root_cause: dict[str, str],
        ) -> dict[str, set[str]]:
    """``root_cause`` → set of distinct source roles whose admitted
    capsules carry a claim_kind implying that root_cause."""
    out: dict[str, set[str]] = defaultdict(set)
    for c in admitted:
        md = c.metadata_dict()
        k = md.get("claim_kind")
        src = md.get("source_role") or ""
        if not isinstance(k, str) or not src:
            continue
        rc = claim_to_root_cause.get(k)
        if rc:
            out[rc].add(src)
    return out


def _votes_per_root_cause(
        admitted: Sequence[ContextCapsule],
        claim_to_root_cause: dict[str, str],
        ) -> dict[str, int]:
    """``root_cause`` → count of admitted capsules whose claim_kind
    implies that root_cause."""
    out: dict[str, int] = defaultdict(int)
    for c in admitted:
        md = c.metadata_dict()
        k = md.get("claim_kind")
        if not isinstance(k, str):
            continue
        rc = claim_to_root_cause.get(k)
        if rc:
            out[rc] += 1
    return out


# =============================================================================
# PriorityDecoder — baseline (the status quo)
# =============================================================================


@dataclasses.dataclass
class PriorityDecoder(BundleDecoder):
    """First-match over a priority order.  Mirrors the Phase-31
    ``_decoder_from_handoffs`` rule: walk ``priority_order`` and
    output the implied root_cause of the first kind present in
    the admitted set.

    This is the status-quo decoder.  Under bundle poisoning by a
    spurious high-priority claim, its accuracy is bounded above
    by ``Pr[gold = priority_top_rc]`` — see Theorem W3-17 in
    ``docs/CAPSULE_FORMALISM.md``.
    """

    priority_order: tuple[str, ...] = DEFAULT_PRIORITY_ORDER
    claim_to_root_cause: dict[str, str] = dataclasses.field(
        default_factory=lambda: dict(DEFAULT_CLAIM_TO_ROOT_CAUSE))
    unknown_label: str = UNKNOWN

    name: str = "priority"

    def decode(self, admitted):
        kinds = set(_claim_kinds(admitted))
        for k in self.priority_order:
            if k in kinds:
                rc = self.claim_to_root_cause.get(k)
                if rc:
                    return rc
        return self.unknown_label


# =============================================================================
# PluralityDecoder — argmax over implied-rc votes
# =============================================================================


@dataclasses.dataclass
class PluralityDecoder(BundleDecoder):
    """Argmax over implied-``root_cause`` vote counts.

    Each admitted capsule with a known ``claim_kind`` casts one
    vote for the root_cause implied by its kind.  The root_cause
    with the most votes wins; ties are broken by walking
    ``priority_order`` and returning the first kind whose implied
    root_cause ties for the max.  Empty votes → ``unknown_label``.

    Breaks the Phase-31 priority ceiling on scenarios whose
    causal chain has ≥ 2 coherent claims implying the same
    root_cause.  See Theorem W3-18 for the sufficient condition.
    """

    priority_order: tuple[str, ...] = DEFAULT_PRIORITY_ORDER
    claim_to_root_cause: dict[str, str] = dataclasses.field(
        default_factory=lambda: dict(DEFAULT_CLAIM_TO_ROOT_CAUSE))
    unknown_label: str = UNKNOWN

    name: str = "plurality"

    def decode(self, admitted):
        votes = _votes_per_root_cause(admitted, self.claim_to_root_cause)
        if not votes:
            return self.unknown_label
        max_v = max(votes.values())
        # Tie-break: walk priority_order, return the first kind
        # whose implied rc ties for max.
        for k in self.priority_order:
            rc = self.claim_to_root_cause.get(k, "")
            if rc and votes.get(rc, 0) == max_v:
                return rc
        # Fallback — an rc whose kind is not in priority_order but
        # did show up (shouldn't happen on Phase-31 but is defensive).
        return max(votes.items(), key=lambda kv: kv[1])[0]


# =============================================================================
# SourceCorroboratedPriorityDecoder — veto singletons, then priority
# =============================================================================


@dataclasses.dataclass
class SourceCorroboratedPriorityDecoder(BundleDecoder):
    """First-match over priority order, but only kinds emitted by
    ≥ ``min_sources`` distinct source roles count.

    This is the cheapest bundle-aware decoder that provably
    defeats a *lone* spurious high-priority injection.  Limit:
    if the causal chain's top-priority claim is itself emitted
    by only one source role (as in Phase-31's tls_expiry
    scenario, where TLS_EXPIRED comes from ``network`` only),
    the decoder vetoes the causal claim too and falls back.
    """

    priority_order: tuple[str, ...] = DEFAULT_PRIORITY_ORDER
    claim_to_root_cause: dict[str, str] = dataclasses.field(
        default_factory=lambda: dict(DEFAULT_CLAIM_TO_ROOT_CAUSE))
    min_sources: int = 2
    unknown_label: str = UNKNOWN

    name: str = "src_corroborated_priority"

    def decode(self, admitted):
        sources = _source_per_kind(admitted)
        for k in self.priority_order:
            if len(sources.get(k, set())) >= self.min_sources:
                rc = self.claim_to_root_cause.get(k)
                if rc:
                    return rc
        return self.unknown_label


# =============================================================================
# LearnedBundleDecoder — small multinomial logistic regression
# =============================================================================


# Closed class-agnostic feature vocabulary.  Each feature is a
# *shape* measurement of "how well does root_cause r fit this
# bundle?"; weights are shared across rc (a single shared weight
# vector is sufficient because the features themselves are
# rc-indexed).
BUNDLE_DECODER_FEATURES: tuple[str, ...] = (
    "bias",
    "log1p_votes",
    "log1p_sources",
    "votes_share",
    "high_priority_votes",
    "has_top_priority_kind",
    "multi_source_flag",          # 1 if sources_for_rc >= 2
    "has_multi_source_kind",      # 1 if any single kind implying r
                                   #  has >= 2 distinct sources
    "lone_top_priority_flag",     # 1 if a top-priority kind implies
                                   #  r AND only 1 source emits it
    "zero_vote_flag",             # 1 if no kind implying r is present
)


def _featurise_bundle_for_rc(
        bundle: Sequence[ContextCapsule],
        rc: str,
        claim_to_root_cause: dict[str, str],
        priority_order: tuple[str, ...],
        high_priority_cutoff: int = DEFAULT_HIGH_PRIORITY_CUTOFF,
        ) -> dict[str, float]:
    """Compute shape features of ``bundle`` for candidate
    root_cause ``rc``.  Features are structural (vote counts,
    source counts), class-agnostic (no per-kind indicator
    variables), and therefore transfer across domains that share
    the same feature vocabulary.
    """
    # Kinds whose implied rc is our candidate:
    kinds_implying_rc = tuple(
        k for k, r in claim_to_root_cause.items() if r == rc)
    high_priority_kinds = frozenset(
        priority_order[:high_priority_cutoff])

    votes = 0
    total_votes = 0
    sources: set[str] = set()
    high_priority_votes = 0
    per_kind_sources: dict[str, set[str]] = defaultdict(set)
    has_top_priority_kind = 0.0

    for c in bundle:
        md = c.metadata_dict()
        k = md.get("claim_kind")
        src = md.get("source_role") or ""
        if not isinstance(k, str):
            continue
        # Count only kinds that map to SOME root_cause (claim-kind
        # alphabet discipline).
        if k not in claim_to_root_cause:
            continue
        total_votes += 1
        if k in kinds_implying_rc:
            votes += 1
            if src:
                sources.add(src)
                per_kind_sources[k].add(src)
            if k in high_priority_kinds:
                high_priority_votes += 1
            if priority_order and k == priority_order[0]:
                has_top_priority_kind = 1.0
    any_multi_source_kind = 0.0
    lone_top_priority = 0.0
    for k, srcs in per_kind_sources.items():
        if len(srcs) >= 2:
            any_multi_source_kind = 1.0
        if k in high_priority_kinds and len(srcs) <= 1:
            lone_top_priority = 1.0

    return {
        "bias": 1.0,
        "log1p_votes": math.log1p(votes),
        "log1p_sources": math.log1p(len(sources)),
        "votes_share": (
            votes / total_votes if total_votes else 0.0),
        "high_priority_votes": float(high_priority_votes),
        "has_top_priority_kind": has_top_priority_kind,
        "multi_source_flag": (
            1.0 if len(sources) >= 2 else 0.0),
        "has_multi_source_kind": any_multi_source_kind,
        "lone_top_priority_flag": lone_top_priority,
        "zero_vote_flag": 1.0 if votes == 0 else 0.0,
    }


@dataclasses.dataclass
class LearnedBundleDecoder(BundleDecoder):
    """Small multinomial logistic-regression decoder.

    The decoder stores a single shared weight vector
    ``weights : dict[str, float]`` (keys drawn from
    ``BUNDLE_DECODER_FEATURES``).  At decode time it scores every
    candidate ``rc`` in ``rc_alphabet`` by
    ``w^T f(bundle, rc)`` and emits ``argmax_rc``.

    Interpretability: exactly ``len(BUNDLE_DECODER_FEATURES)``
    named floats; the weight vector fits on one screen.

    Hypothesis class: class-agnostic linear-in-features.  The
    same weight vector is meaningful across domains that share
    the feature vocabulary (votes + sources + priority shape);
    this is the structural reason the decoder is the natural
    carrier for cross-domain transfer.
    """

    weights: dict[str, float] = dataclasses.field(default_factory=dict)
    rc_alphabet: tuple[str, ...] = ()
    claim_to_root_cause: dict[str, str] = dataclasses.field(
        default_factory=lambda: dict(DEFAULT_CLAIM_TO_ROOT_CAUSE))
    priority_order: tuple[str, ...] = DEFAULT_PRIORITY_ORDER
    high_priority_cutoff: int = DEFAULT_HIGH_PRIORITY_CUTOFF
    unknown_label: str = UNKNOWN

    name: str = "learned_bundle_decoder"

    def _score(self, bundle: Sequence[ContextCapsule], rc: str) -> float:
        f = _featurise_bundle_for_rc(
            bundle, rc, self.claim_to_root_cause,
            self.priority_order, self.high_priority_cutoff)
        return sum(self.weights.get(k, 0.0) * v for k, v in f.items())

    def decode(self, admitted):
        if not self.rc_alphabet:
            return self.unknown_label
        scores = {rc: self._score(admitted, rc)
                   for rc in self.rc_alphabet}
        if not scores:
            return self.unknown_label
        # argmax with a stable tiebreak (alphabetical).
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


def _softmax(logits: dict[str, float]) -> dict[str, float]:
    if not logits:
        return {}
    m = max(logits.values())
    exps = {k: math.exp(v - m) for k, v in logits.items()}
    Z = sum(exps.values()) or 1.0
    return {k: v / Z for k, v in exps.items()}


def train_learned_bundle_decoder(
        examples: Sequence[tuple[Sequence[ContextCapsule], str]],
        *,
        rc_alphabet: Sequence[str],
        claim_to_root_cause: dict[str, str],
        priority_order: tuple[str, ...] = DEFAULT_PRIORITY_ORDER,
        high_priority_cutoff: int = DEFAULT_HIGH_PRIORITY_CUTOFF,
        n_epochs: int = 200,
        lr: float = 0.5,
        l2: float = 1e-3,
        seed: int = 0,
        ) -> LearnedBundleDecoder:
    """Train a ``LearnedBundleDecoder`` by full-batch multinomial
    logistic regression.

    ``examples`` — sequence of ``(admitted_bundle, gold_root_cause)``
    pairs.  Gold labels outside ``rc_alphabet`` are ignored (they
    contribute 0 to the gradient because the classifier can't
    output them anyway).

    Training is deterministic in ``seed``.  Features are NOT
    z-scored: every axis is already bounded (votes ≤ ~30, ratios
    ∈ [0, 1], flags ∈ {0, 1}) so the logistic surface is
    well-conditioned without normalisation, and skipping the
    normalisation makes cross-domain evaluation interpretable
    (no hidden per-domain statistic).
    """
    if not examples:
        raise ValueError("train_learned_bundle_decoder: no examples")
    alphabet = tuple(rc_alphabet)
    if not alphabet:
        raise ValueError("train_learned_bundle_decoder: empty alphabet")
    ctrc = dict(claim_to_root_cause)
    rng = random.Random(seed)

    # Pre-compute feature vectors for every (example, rc) pair.
    # Cache avoids re-featurising inside the GD loop.
    feats_by_example: list[dict[str, dict[str, float]]] = []
    gold_by_example: list[str] = []
    for (bundle, gold_rc) in examples:
        per_rc: dict[str, dict[str, float]] = {}
        for rc in alphabet:
            per_rc[rc] = _featurise_bundle_for_rc(
                bundle, rc, ctrc, priority_order,
                high_priority_cutoff)
        feats_by_example.append(per_rc)
        gold_by_example.append(gold_rc)

    weights: dict[str, float] = {k: 0.0 for k in BUNDLE_DECODER_FEATURES}
    n = len(examples)
    for ep in range(n_epochs):
        idxs = list(range(n))
        rng.shuffle(idxs)
        grad: dict[str, float] = {k: 0.0 for k in weights}
        for i in idxs:
            per_rc = feats_by_example[i]
            gold = gold_by_example[i]
            # Compute logits, softmax, and gradient contribution.
            logits = {rc: sum(weights.get(k, 0.0) * v
                               for k, v in per_rc[rc].items())
                       for rc in alphabet}
            probs = _softmax(logits)
            for rc in alphabet:
                coef = probs[rc] - (1.0 if rc == gold else 0.0)
                for k, v in per_rc[rc].items():
                    grad[k] = grad.get(k, 0.0) + coef * v
        for k in list(weights.keys()):
            g = grad[k] / n + l2 * weights[k]
            weights[k] -= lr * g
    return LearnedBundleDecoder(
        weights=weights,
        rc_alphabet=alphabet,
        claim_to_root_cause=ctrc,
        priority_order=tuple(priority_order),
        high_priority_cutoff=int(high_priority_cutoff),
    )


# =============================================================================
# Decoder evaluation helpers
# =============================================================================


@dataclasses.dataclass
class DecoderResult:
    """Result of a decoder sweep on a set of (admitted_bundle,
    gold_rc) instances."""
    name: str
    n_instances: int
    accuracy: float
    per_scenario_accuracy: dict[str, float]


def evaluate_decoder(
        instances: Sequence[tuple[Sequence[ContextCapsule], str]],
        decoder: BundleDecoder,
        scenario_keys: Sequence[str] | None = None,
        ) -> DecoderResult:
    """Run ``decoder`` over every ``(bundle, gold_rc)`` instance
    and return aggregate accuracy + per-scenario breakdown.

    ``scenario_keys`` — optional per-instance scenario labels
    (e.g. ``"disk_fill_cron"``) used to compute per-scenario
    accuracy.  Length must match ``instances`` if provided.
    """
    if scenario_keys is not None and len(scenario_keys) != len(instances):
        raise ValueError(
            "scenario_keys length != instances length")
    n = len(instances)
    if n == 0:
        return DecoderResult(
            name=decoder.name, n_instances=0, accuracy=0.0,
            per_scenario_accuracy={})
    n_correct = 0
    per_scenario_correct: dict[str, int] = defaultdict(int)
    per_scenario_total: dict[str, int] = defaultdict(int)
    for i, (bundle, gold) in enumerate(instances):
        out = decoder.decode(bundle)
        correct = int(out == gold)
        n_correct += correct
        if scenario_keys is not None:
            key = scenario_keys[i]
            per_scenario_total[key] += 1
            per_scenario_correct[key] += correct
    per_scenario = {
        k: per_scenario_correct[k] / per_scenario_total[k]
        for k in per_scenario_total}
    return DecoderResult(
        name=decoder.name, n_instances=n,
        accuracy=n_correct / n,
        per_scenario_accuracy=per_scenario,
    )


__all__ = [
    "UNKNOWN",
    "BundleDecoder",
    "PriorityDecoder",
    "PluralityDecoder",
    "SourceCorroboratedPriorityDecoder",
    "LearnedBundleDecoder",
    "train_learned_bundle_decoder",
    "BUNDLE_DECODER_FEATURES",
    "DecoderResult", "evaluate_decoder",
]
