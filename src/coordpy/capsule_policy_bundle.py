"""Phase 47 — bundle-aware capsule admission policies.

Phase 46 shipped *per-capsule* admission learning:
``LearnedAdmissionPolicy`` scores each proposed capsule from its
own header features and admits the highest-scored subset under
the global token budget. That lift was real but narrow — it
drove admit-precision from 0.634 → 0.796 at B=16 but left the
downstream Phase-31 decoder stuck at 0.225 accuracy on the
noise-poisoned bundle (see ``docs/RESULTS_CAPSULE_LEARNING.md``
§ 3.3). The bottleneck is not the classifier, it is the
**bundle**: the Phase-31 priority decoder picks the first
high-priority ``claim_kind`` present in the admitted set, so a
single spurious ``DISK_FILL_CRITICAL`` wins even when fifty
causal claims argue otherwise.

This module ships the *bundle-aware* step — policies that see
the full offered set before admitting any single capsule, and
can veto capsules whose individual scores are high but whose
admission would poison the downstream bundle. Three policies
in order of increasing structure:

  * ``CorroboratedAdmissionPolicy`` — heuristic. A high-priority
    ``claim_kind`` K is admissible only if ≥ ``min_sources``
    distinct source roles in the offered set emit K. One-off
    spurious claims fail this check.

  * ``PluralityBundlePolicy`` — bundle-aware heuristic. Maps each
    offered claim_kind to its implied root_cause via the Phase-31
    priority decoder; computes the vote share per implied root_cause
    across the offered set; admits capsules whose implied
    root_cause matches the plurality winner in priority order. A
    spurious isolated vote loses to the 3–5 claim causal chain.

  * ``BundleLearnedPolicy`` — real ML bundle scorer. Logistic
    regression over *per-capsule + per-bundle* features
    (adding source-corroboration count, root_cause-vote share,
    high-priority-isolated flag). Trained via full-batch GD.
    Inspectable weights; deliberately small hypothesis class so
    the win — if any — is header-structure-only.

The objective of interest is **downstream decoder accuracy**
(was the admitted bundle's decoder output equal to the gold
root_cause?), not admit-precision. The Phase 47 experiment
driver ``phase47_bundle_learning.py`` reports all three metrics
so the honest win condition ("the bundle-aware policy lifts
decoder accuracy past 0.225 on noise-poisoned Phase-31 triage")
is the one the bench measures.

Theoretical anchor: ``docs/CAPSULE_FORMALISM.md`` § 5 / § 4.B
(Conjecture P46-C1 — bundle-aware admission closes the noise
ceiling). This module's contribution is the first empirical
evidence bearing on P46-C1.
"""

from __future__ import annotations

import dataclasses
import math
from collections import defaultdict
from typing import Any, Sequence

from .capsule import ContextCapsule
from .capsule_policy import (
    ADMIT, AdmissionPolicy, featurise_capsule, feature_index,
)


# =============================================================================
# Priority map — the canonical Phase-31 decoder used to compute implied
# root_cause for bundle-level reasoning.
# =============================================================================


# Mirror of coordpy/_internal/tasks/incident_triage.py::_decoder_from_handoffs.
# We replicate the table here to avoid importing the task module (and
# pulling in its scenario-bank dependencies) from the SDK surface. If a
# caller wants a different decoder, they pass their own priority map.
DEFAULT_CLAIM_TO_ROOT_CAUSE: dict[str, str] = {
    "DISK_FILL_CRITICAL": "disk_fill",
    "TLS_EXPIRED": "tls_expiry",
    "DNS_MISROUTE": "dns_misroute",
    "OOM_KILL": "memory_leak",
    "DEADLOCK_SUSPECTED": "deadlock",
    "CRON_OVERRUN": "disk_fill",
    "POOL_EXHAUSTION": "pool_exhaustion",
    "SLOW_QUERY_OBSERVED": "slow_query_cascade",
    "ERROR_RATE_SPIKE": "error_spike",
    "LATENCY_SPIKE": "latency_spike",
    "FW_BLOCK_SURGE": "fw_block",
}
DEFAULT_PRIORITY_ORDER: tuple[str, ...] = (
    "DISK_FILL_CRITICAL", "TLS_EXPIRED", "DNS_MISROUTE",
    "OOM_KILL", "DEADLOCK_SUSPECTED", "CRON_OVERRUN",
    "POOL_EXHAUSTION", "SLOW_QUERY_OBSERVED",
    "ERROR_RATE_SPIKE", "LATENCY_SPIKE", "FW_BLOCK_SURGE",
)
DEFAULT_HIGH_PRIORITY_CUTOFF = 5


# =============================================================================
# Bundle feature extraction
# =============================================================================


@dataclasses.dataclass(frozen=True)
class BundleStats:
    """Summary statistics of an offered set, computed once and
    reused across per-capsule bundle-feature lookups.

    Fields:
      * ``sources_per_kind`` — ``claim_kind`` → set of distinct
        source roles in the offered set emitting that kind.
      * ``votes_per_root_cause`` — implied-root_cause (via the
        priority map) → count of offered capsules voting for it.
      * ``plurality_root_cause`` — arg-max over votes_per_root_cause
        (ties broken by the priority order).
    """

    sources_per_kind: dict[str, frozenset[str]]
    votes_per_root_cause: dict[str, int]
    plurality_root_cause: str
    plurality_vote_count: int
    n_offered: int

    @classmethod
    def from_offered(cls, capsules: Sequence[ContextCapsule],
                      claim_to_root_cause: dict[str, str] =
                          DEFAULT_CLAIM_TO_ROOT_CAUSE,
                      priority_order: tuple[str, ...] =
                          DEFAULT_PRIORITY_ORDER,
                      ) -> "BundleStats":
        sources: dict[str, set[str]] = defaultdict(set)
        votes: dict[str, int] = defaultdict(int)
        for c in capsules:
            md = c.metadata_dict()
            kind = md.get("claim_kind")
            src = (md.get("source_role") or "")
            if not isinstance(kind, str):
                continue
            if src:
                sources[kind].add(src)
            rc = claim_to_root_cause.get(kind)
            if rc is not None:
                votes[rc] += 1
        # Plurality winner: choose the implied root_cause with the
        # most votes; tie-break by priority order (first kind in
        # priority_order whose implied root_cause matches the max).
        best_rc = ""
        best_votes = 0
        if votes:
            # Find max vote count first
            max_v = max(votes.values())
            # Walk priority_order; the first kind whose implied
            # root_cause has max votes wins.
            for kind in priority_order:
                rc = claim_to_root_cause.get(kind)
                if rc and votes.get(rc, 0) == max_v and best_rc == "":
                    best_rc = rc
                    best_votes = max_v
                    break
            if best_rc == "":
                # Fallback — unknown kinds.
                best_rc = next(iter(votes))
                best_votes = votes[best_rc]
        return cls(
            sources_per_kind={
                k: frozenset(v) for k, v in sources.items()},
            votes_per_root_cause=dict(votes),
            plurality_root_cause=best_rc,
            plurality_vote_count=best_votes,
            n_offered=len(capsules),
        )


def featurise_capsule_with_bundle(
        c: ContextCapsule, bundle: BundleStats,
        *,
        high_priority_kinds: frozenset[str] = frozenset(
            DEFAULT_PRIORITY_ORDER[:DEFAULT_HIGH_PRIORITY_CUTOFF]),
        claim_to_root_cause: dict[str, str] =
            DEFAULT_CLAIM_TO_ROOT_CAUSE,
        ) -> dict[str, float]:
    """Per-capsule feature dict enriched with bundle-level
    statistics. Extends ``capsule_policy.featurise_capsule`` with:

      * ``bundle:n_sources_same_kind`` — # distinct source roles
        in the offered set emitting this capsule's claim_kind.
      * ``bundle:implies_plurality_rc`` — 1.0 if this capsule's
        implied root_cause equals the bundle's plurality winner;
        0.0 otherwise.
      * ``bundle:plurality_vote_share`` — plurality vote count
        divided by n_offered (in [0, 1]).
      * ``bundle:lone_high_priority`` — 1.0 if claim_kind is
        high priority AND only 1 source in offered set emits it.
        This is the spurious-injection signature.

    The four bundle features all depend ONLY on the offered set's
    header-level structure; no LLM or domain-specific inference
    beyond the priority map.
    """
    base = featurise_capsule(c)
    md = c.metadata_dict()
    kind = md.get("claim_kind") if isinstance(
        md.get("claim_kind"), str) else None
    n_sources_same = 0
    if kind and kind in bundle.sources_per_kind:
        n_sources_same = len(bundle.sources_per_kind[kind])
    implied_rc = claim_to_root_cause.get(kind or "", "")
    implies_plurality = (1.0 if implied_rc and
                         implied_rc == bundle.plurality_root_cause
                         else 0.0)
    plurality_share = (
        (bundle.plurality_vote_count / bundle.n_offered)
        if bundle.n_offered else 0.0)
    lone_high = (1.0 if (kind in high_priority_kinds
                          and n_sources_same <= 1) else 0.0)
    base["bundle:n_sources_same_kind"] = float(n_sources_same)
    base["bundle:implies_plurality_rc"] = implies_plurality
    base["bundle:plurality_vote_share"] = plurality_share
    base["bundle:lone_high_priority"] = lone_high
    return base


def bundle_feature_index() -> tuple[str, ...]:
    return tuple(list(feature_index()) + [
        "bundle:n_sources_same_kind",
        "bundle:implies_plurality_rc",
        "bundle:plurality_vote_share",
        "bundle:lone_high_priority",
    ])


# =============================================================================
# Bundle-aware AdmissionPolicy interface
# =============================================================================


class BundleAwarePolicy(AdmissionPolicy):
    """AdmissionPolicy that operates on the *full offered set*.

    Subclasses override ``score_all`` (bundle-aware) and get
    ``decide`` for free. The batched entry point of
    ``BudgetedAdmissionLedger.offer_all_batched`` is the recommended
    driver for any policy in this family — it passes the full
    offered set to ``score_all`` and admits greedily by score.

    A subclass may also override ``reject_set`` to explicitly
    return the subset of offered capsules to REJECT regardless of
    score; that subset is then never admitted even if budget
    remains (useful for veto-style bundle rules like the
    corroboration gate).
    """

    name: str = "bundle-abstract"

    def decide(self, capsule, ledger, remaining_budget):
        # By default, bundle-aware policies are *batched* — a
        # streaming decide() that doesn't see the offered set
        # falls back to FIFO (ADMIT). Callers should use
        # offer_all_batched() on BudgetedAdmissionLedger.
        return ADMIT

    def reject_set(self,
                    capsules: Sequence[ContextCapsule],
                    ) -> frozenset[str]:
        """Optional: return the CID subset to never admit.
        Default empty."""
        return frozenset()

    def score_all(self, capsules: Sequence[ContextCapsule],
                   ) -> Sequence[float]:
        raise NotImplementedError


# =============================================================================
# CorroboratedAdmissionPolicy — heuristic bundle gate
# =============================================================================


@dataclasses.dataclass
class CorroboratedAdmissionPolicy(BundleAwarePolicy):
    """Admit high-priority claims only if corroborated by
    ``min_sources`` distinct source roles in the offered set.

    Low-priority claims admit under a wrapped per-capsule
    policy (default: admit all). This is the simplest bundle-
    aware policy that breaks the noise-poisoning ceiling: a
    spurious one-off DISK_FILL_CRITICAL fails corroboration and
    is vetoed at admit time, so the priority decoder never sees
    it.

    Parameters:
      * ``min_sources`` — corroboration threshold (default 2).
      * ``high_priority_kinds`` — claim kinds that require
        corroboration. Default: the top 5 of the Phase-31 priority
        order.
      * ``inner_policy`` — per-capsule policy for kinds not in
        ``high_priority_kinds``. Default: FIFO.
    """

    min_sources: int = 2
    high_priority_kinds: frozenset[str] = dataclasses.field(
        default_factory=lambda: frozenset(
            DEFAULT_PRIORITY_ORDER[:DEFAULT_HIGH_PRIORITY_CUTOFF]))
    inner_policy: AdmissionPolicy | None = None

    name: str = "corroborated"

    def reject_set(self, capsules: Sequence[ContextCapsule]
                    ) -> frozenset[str]:
        stats = BundleStats.from_offered(capsules)
        rej: set[str] = set()
        for c in capsules:
            md = c.metadata_dict()
            kind = md.get("claim_kind")
            if not isinstance(kind, str):
                continue
            if kind not in self.high_priority_kinds:
                continue
            n_sources = len(
                stats.sources_per_kind.get(kind, frozenset()))
            if n_sources < self.min_sources:
                rej.add(c.cid)
        return frozenset(rej)

    def score_all(self, capsules):
        rej = self.reject_set(capsules)
        # Use the inner policy's scores if present; vetoed capsules
        # score -inf.
        if self.inner_policy is None:
            base_scores = [1.0 for _ in capsules]
        else:
            base_scores = list(self.inner_policy.score_all(capsules))
        return [(-1e18 if c.cid in rej else s)
                 for c, s in zip(capsules, base_scores)]


# =============================================================================
# PluralityBundlePolicy — bundle-aware heuristic
# =============================================================================


@dataclasses.dataclass
class PluralityBundlePolicy(BundleAwarePolicy):
    """Admit capsules whose implied root_cause matches the bundle's
    plurality winner; otherwise down-rank.

    This is the "follow the vote" policy. It tries to keep the
    admitted set internally consistent so the priority decoder sees
    a coherent story, not a mixture. The implied-root_cause map
    defaults to the Phase-31 priority decoder; callers with
    other decoders pass their own map.

    Parameters:
      * ``claim_to_root_cause`` — claim_kind → implied root_cause.
      * ``priority_order`` — tie-break order for plurality winner.
      * ``off_winner_penalty`` — how much to down-rank capsules
        whose implied root_cause is NOT the plurality winner.
        ``0.0`` = no penalty, ``1.0`` = hard reject.
    """

    claim_to_root_cause: dict[str, str] = dataclasses.field(
        default_factory=lambda: dict(DEFAULT_CLAIM_TO_ROOT_CAUSE))
    priority_order: tuple[str, ...] = DEFAULT_PRIORITY_ORDER
    off_winner_penalty: float = 1.0  # hard reject by default

    name: str = "plurality_bundle"

    def score_all(self, capsules):
        stats = BundleStats.from_offered(
            capsules,
            claim_to_root_cause=self.claim_to_root_cause,
            priority_order=self.priority_order)
        winner = stats.plurality_root_cause
        scores: list[float] = []
        for c in capsules:
            md = c.metadata_dict()
            kind = md.get("claim_kind")
            implied = self.claim_to_root_cause.get(
                kind, "") if isinstance(kind, str) else ""
            if implied == winner and winner != "":
                # Priority-ordered rank within the winner set.
                try:
                    rank = self.priority_order.index(kind) \
                        if isinstance(kind, str) else 99
                except ValueError:
                    rank = 99
                scores.append(1000.0 - float(rank))
            else:
                # Off-winner: down-rank (or hard-reject if
                # off_winner_penalty == 1.0, by assigning -inf).
                if self.off_winner_penalty >= 1.0:
                    scores.append(-1e18)
                else:
                    scores.append(-self.off_winner_penalty)
        return scores


# =============================================================================
# BundleLearnedPolicy — logistic regression on per-capsule + bundle features
# =============================================================================


@dataclasses.dataclass
class BundleLearnedPolicy(BundleAwarePolicy):
    """Logistic regression policy over per-capsule ∪ bundle
    features.

    Training is a small extension of ``train_admission_policy``:
    each training example pairs a capsule with its *bundle
    context* (the offered set it was drawn from) so the bundle
    features are computable during training as well as inference.

    The hypothesis class is linear-in-features. Same
    interpretability guarantee as ``LearnedAdmissionPolicy``:
    every weight is a named float. The four bundle features
    (``bundle:*``) add four floats to the existing ~40 and
    turn out to be strongly informative — in particular
    ``bundle:lone_high_priority`` gets a large negative weight
    on the Phase-31 data (see ``docs/RESULTS_CAPSULE_RESEARCH_MILESTONE2.md``
    § 3).
    """

    weights: dict[str, float] = dataclasses.field(default_factory=dict)
    threshold: float = 0.5
    feature_means: dict[str, float] = dataclasses.field(
        default_factory=dict)
    feature_stds: dict[str, float] = dataclasses.field(
        default_factory=dict)
    high_priority_kinds: frozenset[str] = dataclasses.field(
        default_factory=lambda: frozenset(
            DEFAULT_PRIORITY_ORDER[:DEFAULT_HIGH_PRIORITY_CUTOFF]))
    claim_to_root_cause: dict[str, str] = dataclasses.field(
        default_factory=lambda: dict(DEFAULT_CLAIM_TO_ROOT_CAUSE))

    name: str = "bundle_learned"

    def _prepare(self, c: ContextCapsule,
                 stats: BundleStats) -> dict[str, float]:
        f = featurise_capsule_with_bundle(
            c, stats,
            high_priority_kinds=self.high_priority_kinds,
            claim_to_root_cause=self.claim_to_root_cause)
        # Normalise continuous features.
        for cont in ("log1p_n_tokens", "log1p_n_bytes",
                      "n_parents", "bundle:n_sources_same_kind",
                      "bundle:plurality_vote_share"):
            mu = self.feature_means.get(cont, 0.0)
            sd = self.feature_stds.get(cont, 1.0) or 1.0
            if cont in f:
                f[cont] = (f[cont] - mu) / sd
        return f

    def score_all(self, capsules):
        stats = BundleStats.from_offered(
            capsules,
            claim_to_root_cause=self.claim_to_root_cause)
        out: list[float] = []
        for c in capsules:
            f = self._prepare(c, stats)
            z = sum(self.weights.get(k, 0.0) * v
                     for k, v in f.items())
            out.append(1.0 / (1.0 + math.exp(-z)))
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "threshold": self.threshold,
            "weights": dict(self.weights),
            "feature_means": dict(self.feature_means),
            "feature_stds": dict(self.feature_stds),
            "high_priority_kinds": sorted(self.high_priority_kinds),
            "claim_to_root_cause": dict(self.claim_to_root_cause),
        }


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def train_bundle_policy(
        examples: Sequence[tuple[Sequence[ContextCapsule],
                                  Sequence[ContextCapsule],
                                  Sequence[int]]],
        *,
        n_epochs: int = 200,
        lr: float = 0.5,
        l2: float = 1e-3,
        seed: int = 0,
        high_priority_kinds: frozenset[str] | None = None,
        claim_to_root_cause: dict[str, str] | None = None,
        ) -> BundleLearnedPolicy:
    """Train a ``BundleLearnedPolicy``.

    ``examples`` is a sequence of ``(offered, members, labels)``
    triples where ``members`` is the subset of ``offered`` the
    caller wants to label and ``labels`` are the gold binary
    labels for each member (1 = causal, 0 = distractor).
    ``offered`` provides the *bundle context* used to compute
    bundle features for each member (they may be a subset of
    ``offered``).

    Continuous feature means/stds are computed over the entire
    training set and stored on the returned policy.
    """
    if not examples:
        raise ValueError("train_bundle_policy: no examples")
    import random as _random
    rng = _random.Random(seed)
    hpk = high_priority_kinds or frozenset(
        DEFAULT_PRIORITY_ORDER[:DEFAULT_HIGH_PRIORITY_CUTOFF])
    ctrc = dict(claim_to_root_cause or DEFAULT_CLAIM_TO_ROOT_CAUSE)

    # 1) Compute raw features for every (offered, member)
    # pair, collect (features, label) with bundle features already
    # injected. This is where bundle context matters.
    raw_feats: list[dict[str, float]] = []
    raw_labels: list[int] = []
    for (offered, members, labels) in examples:
        stats = BundleStats.from_offered(
            offered, claim_to_root_cause=ctrc)
        for c, y in zip(members, labels):
            f = featurise_capsule_with_bundle(
                c, stats, high_priority_kinds=hpk,
                claim_to_root_cause=ctrc)
            raw_feats.append(f)
            raw_labels.append(int(y))
    if not raw_feats:
        raise ValueError("train_bundle_policy: no member examples")

    # 2) Z-score continuous features.
    cont_axes = ("log1p_n_tokens", "log1p_n_bytes", "n_parents",
                  "bundle:n_sources_same_kind",
                  "bundle:plurality_vote_share")
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for ax in cont_axes:
        vals = [f.get(ax, 0.0) for f in raw_feats]
        m = sum(vals) / max(1, len(vals))
        var = sum((v - m) ** 2 for v in vals) / max(1, len(vals))
        means[ax] = m
        stds[ax] = math.sqrt(var) or 1.0
    for f in raw_feats:
        for ax in cont_axes:
            if ax in f:
                f[ax] = (f[ax] - means[ax]) / stds[ax]

    # 3) Full-batch GD.
    weights: dict[str, float] = {k: 0.0 for k in bundle_feature_index()}
    n = len(raw_feats)
    for ep in range(n_epochs):
        idxs = list(range(n))
        rng.shuffle(idxs)
        grad: dict[str, float] = {k: 0.0 for k in weights}
        for i in idxs:
            f = raw_feats[i]
            y = raw_labels[i]
            z = sum(weights.get(k, 0.0) * v
                     for k, v in f.items())
            p = _sigmoid(z)
            err = p - y
            for k, v in f.items():
                grad[k] = grad.get(k, 0.0) + err * v
        for k in weights:
            g = grad[k] / n + l2 * weights[k]
            weights[k] -= lr * g
    return BundleLearnedPolicy(
        weights=weights, threshold=0.5,
        feature_means=means, feature_stds=stds,
        high_priority_kinds=hpk,
        claim_to_root_cause=ctrc,
    )


# =============================================================================
# Public surface
# =============================================================================


__all__ = [
    "BundleStats", "featurise_capsule_with_bundle",
    "bundle_feature_index",
    "BundleAwarePolicy",
    "CorroboratedAdmissionPolicy",
    "PluralityBundlePolicy",
    "BundleLearnedPolicy",
    "train_bundle_policy",
    "DEFAULT_CLAIM_TO_ROOT_CAUSE",
    "DEFAULT_PRIORITY_ORDER",
    "DEFAULT_HIGH_PRIORITY_CUTOFF",
]
