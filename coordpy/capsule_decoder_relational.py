"""Phase 51 — cohort-relational bundle-aware decoding.

Phase 50 closed the Phase-49 paradigm-shift-candidate question
under the refined ``W3-C9`` bar (gap reading of Gate 2
+ point-estimate Gate 1 at n=80).  Two structural theorems —
W3-24 (post-search winner's-curse bias, proved) and W3-29
(Bayes-divergence zero-shot risk lower bound for the linear
class-agnostic family, proved) — explain *why* the strict
reading of W3-C7 is blocked on the present benchmark family.

Phase 50's sign-stable DeepSet achieves gap = 0.000 at zero-shot
level 0.237 on (incident, security) — direction-invariance
without level-matching.  The open frontier that Phase 51
attacks is the **level** of direction-invariant zero-shot
transfer: can a hypothesis class **structurally outside** the
W3-29-bounded magnitude-monoid family raise the level
materially above 0.237?

The Phase-51 candidate is the **cohort-relational** decoder:
rather than aggregating per-capsule features over the whole
bundle, we partition the bundle by source-role, compute a
per-role aggregate ψ(E_r, rc), then aggregate over the set of
roles.  This is a strictly richer hypothesis class than
DeepSet (Theorem W3-30) because it can express predicates that
distinguish two bundles with the same (kind, rc) multiset but
different role assignments — for example, "≥ 2 *distinct*
source roles independently emit a capsule implying rc".  That
is a relational / pairwise predicate in the sense of W3-16's
limitation theorem.

The module ships one decoder:

  * ``CohortRelationalDecoder`` — 2-layer cohort-aggregate
    decoder.  Input: for each (source_role, rc) pair, a small
    fixed-dim feature vector ψ(E_r, rc) ∈ R^6 (role presence,
    role supports rc, role contradicts rc, role has top-priority
    capsule, role has top-priority capsule implying rc, count
    of rc-supporting capsules from this role).  Aggregation:
    sum over roles of ψ, producing a bundle-level vector
    ρ(E, rc) ∈ R^6.  Post-aggregation: concatenate with a
    6-dim **cross-role** feature vector that reads the set of
    roles (distinct-roles-supporting-rc, distinct-roles-
    contradicting-rc, is-strict-role-plurality-for-rc, pair-
    agreement count, role-contradiction count, bias), and score
    through a 1-hidden-layer tanh MLP.  ~180 parameters.

    Weights are numpy arrays; training is full-batch softmax-
    cross-entropy via ``train_cohort_relational_decoder``.
    Deterministic in ``seed``.

Phase-51 theorem anchors (see ``docs/CAPSULE_FORMALISM.md``
§ 4.F):

  * Theorem W3-30 (strict separation — proved, constructive):
    the cohort-relational class strictly contains the DeepSet
    class because role-identity is erased by per-capsule φ-sum
    but preserved by per-role cohort partition.

  * Claim W3-31 (empirical level lift — code-backed, named):
    on the (incident, security) pair, direction-invariant
    zero-shot transfer reaches a level materially above
    Phase-50's sign-stable DeepSet 0.237 floor — or, if
    falsified, the programme adopts W3-C10 (level-ceiling
    conjecture).

The module keeps the Capsule Contract C1..C6 untouched — the
decoder is a read-only consumer of the admitted ledger.
"""

from __future__ import annotations

import dataclasses
import math
from collections import defaultdict
from typing import Any, Sequence

try:
    import numpy as np
except ImportError as ex:
    raise ImportError(
        "vision_mvp.coordpy.capsule_decoder_relational requires numpy"
    ) from ex

from .capsule import ContextCapsule
from .capsule_decoder import BundleDecoder, UNKNOWN
from .capsule_policy_bundle import (
    DEFAULT_CLAIM_TO_ROOT_CAUSE, DEFAULT_PRIORITY_ORDER,
    DEFAULT_HIGH_PRIORITY_CUTOFF,
)


# =============================================================================
# Cohort-relational feature vocabulary
# =============================================================================


# Per-(role, rc) features ψ(E_r, rc) ∈ R^6.  Each feature is a
# scalar evaluated on the sub-bundle E_r of capsules whose
# source_role == r, conditioned on candidate rc.
#
# The feature choices are deliberately **sign-stable** across
# operational-detection domains (see W3-C8'):
#   * role_supports_rc — at least one capsule with claim_kind
#     implying rc (sign-stable: more is evidence for rc).
#   * log1p_role_support_count — magnitude of support (stable).
#   * role_has_top_priority — top-priority capsule from this
#     role present (competitor-structure-independent).
#   * role_top_priority_implies_rc — sign-stable conjunction.
#   * role_top_priority_contradicts_rc — sign-stable (negative
#     for rc).
#   * role_has_high_priority_supporting_rc — redundant with
#     role_supports_rc on high-priority sub-kinds; included
#     for gradient capacity at the role level.
COHORT_PSI_FEATURES: tuple[str, ...] = (
    "psi:role_supports_rc",
    "psi:log1p_role_support_count",
    "psi:role_has_top_priority",
    "psi:role_top_priority_implies_rc",
    "psi:role_top_priority_contradicts_rc",
    "psi:role_has_high_priority_supporting_rc",
)


# Post-aggregation cross-role features ρ(E, rc) ∈ R^6.  These
# are computed *after* partitioning by role and *cannot* be
# reduced to a per-capsule sum, making them outside the DeepSet
# class (W3-30 witness).
COHORT_RHO_FEATURES: tuple[str, ...] = (
    "rho:bias",
    "rho:distinct_roles_supporting_rc",
    "rho:distinct_roles_contradicting_rc",
    "rho:is_role_plurality_for_rc",      # 1 if rc maxes role-count
    "rho:pairs_of_supporting_roles",     # C(n_sup, 2) — quadratic
    "rho:distinct_roles_total",
)


def _per_role_support(
        bundle: Sequence[ContextCapsule],
        rc: str,
        claim_to_root_cause: dict[str, str],
        priority_order: tuple[str, ...],
        high_priority_cutoff: int,
        ) -> dict[str, dict[str, float]]:
    """Partition bundle by source_role and compute per-role
    feature vector ψ(E_r, rc).

    Returns dict: role -> {feature_name -> value}.  Only roles
    with ≥ 1 capsule in the bundle are returned.
    """
    high_priority_kinds = frozenset(
        priority_order[:high_priority_cutoff])
    top_priority_kind = (priority_order[0]
                          if priority_order else "")

    by_role_kinds: dict[str, list[str]] = defaultdict(list)
    for c in bundle:
        md = c.metadata_dict()
        k = md.get("claim_kind")
        r = md.get("source_role") or ""
        if not isinstance(k, str) or not r:
            continue
        by_role_kinds[r].append(k)

    out: dict[str, dict[str, float]] = {}
    for r, kinds in by_role_kinds.items():
        supports = 0
        has_top_priority = 0.0
        top_priority_implies_rc = 0.0
        top_priority_contradicts_rc = 0.0
        high_priority_supports_rc = 0.0
        for k in kinds:
            implied = claim_to_root_cause.get(k)
            if implied == rc:
                supports += 1
                if k in high_priority_kinds:
                    high_priority_supports_rc = 1.0
            if k == top_priority_kind:
                has_top_priority = 1.0
                if implied == rc:
                    top_priority_implies_rc = 1.0
                elif implied and implied != rc:
                    top_priority_contradicts_rc = 1.0
        out[r] = {
            "psi:role_supports_rc":
                1.0 if supports > 0 else 0.0,
            "psi:log1p_role_support_count":
                math.log1p(supports),
            "psi:role_has_top_priority": has_top_priority,
            "psi:role_top_priority_implies_rc":
                top_priority_implies_rc,
            "psi:role_top_priority_contradicts_rc":
                top_priority_contradicts_rc,
            "psi:role_has_high_priority_supporting_rc":
                high_priority_supports_rc,
        }
    return out


def _cohort_rho_vector(
        bundle: Sequence[ContextCapsule],
        rc: str,
        claim_to_root_cause: dict[str, str],
        priority_order: tuple[str, ...],
        high_priority_cutoff: int,
        rc_alphabet: tuple[str, ...],
        ) -> np.ndarray:
    """Compute the 6-dim cross-role feature ρ(E, rc).

    Requires computing the per-role support structure for
    EVERY rc in the alphabet (not just the candidate rc) to
    decide whether the candidate is the role-plurality.
    """
    # Per-role (role -> set of rcs supported by that role).
    role_supports: dict[str, set[str]] = defaultdict(set)
    role_top_priority_implied: dict[str, str] = {}
    top_priority_kind = (priority_order[0]
                          if priority_order else "")

    for c in bundle:
        md = c.metadata_dict()
        k = md.get("claim_kind")
        r = md.get("source_role") or ""
        if not isinstance(k, str) or not r:
            continue
        implied = claim_to_root_cause.get(k)
        if implied:
            role_supports[r].add(implied)
        if k == top_priority_kind and implied:
            role_top_priority_implied[r] = implied

    # n_supporting_roles[r] = count of roles supporting rc'.
    supp_count_by_rc: dict[str, int] = defaultdict(int)
    for r, supported in role_supports.items():
        for rc_sup in supported:
            supp_count_by_rc[rc_sup] += 1
    n_sup = supp_count_by_rc.get(rc, 0)

    # Contradicting roles: those whose top-priority-implied rc
    # is present and != rc.
    n_contra = sum(
        1 for r, tp_rc in role_top_priority_implied.items()
        if tp_rc and tp_rc != rc)

    # Is rc the role-plurality?
    max_other = max(
        (v for (k, v) in supp_count_by_rc.items() if k != rc),
        default=0)
    is_role_plurality = (1.0 if n_sup > 0 and n_sup > max_other
                          else 0.0)

    # Distinct-roles-total.
    distinct_roles_total = len(role_supports)

    return np.array([
        1.0,  # rho:bias
        float(n_sup),
        float(n_contra),
        is_role_plurality,
        float(n_sup * (n_sup - 1) // 2),  # pairs
        float(distinct_roles_total),
    ], dtype=np.float64)


def _cohort_psi_sum(
        bundle: Sequence[ContextCapsule],
        rc: str,
        claim_to_root_cause: dict[str, str],
        priority_order: tuple[str, ...],
        high_priority_cutoff: int,
        ) -> np.ndarray:
    """Sum over roles r of ψ(E_r, rc) — the cohort aggregator."""
    per_role = _per_role_support(
        bundle, rc, claim_to_root_cause, priority_order,
        high_priority_cutoff)
    out = np.zeros(len(COHORT_PSI_FEATURES), dtype=np.float64)
    for r, feats in per_role.items():
        for i, name in enumerate(COHORT_PSI_FEATURES):
            out[i] += feats[name]
    return out


def _cohort_relational_input_vector(
        bundle: Sequence[ContextCapsule],
        rc: str,
        claim_to_root_cause: dict[str, str],
        priority_order: tuple[str, ...],
        high_priority_cutoff: int,
        rc_alphabet: tuple[str, ...],
        ) -> np.ndarray:
    """Concatenation of ψ-sum (cohort aggregate) and ρ (cross-
    role features).  Dim = 6 + 6 = 12."""
    psi = _cohort_psi_sum(
        bundle, rc, claim_to_root_cause, priority_order,
        high_priority_cutoff)
    rho = _cohort_rho_vector(
        bundle, rc, claim_to_root_cause, priority_order,
        high_priority_cutoff, rc_alphabet)
    return np.concatenate([psi, rho])


COHORT_RELATIONAL_FEATURES: tuple[str, ...] = tuple(
    COHORT_PSI_FEATURES) + tuple(COHORT_RHO_FEATURES)


# =============================================================================
# CohortRelationalDecoder
# =============================================================================


@dataclasses.dataclass
class CohortRelationalDecoder(BundleDecoder):
    """Cohort-relational decoder.  Per-role aggregate + cross-
    role feature vector, scored through a 1-hidden-layer tanh
    MLP shared across rc.

    Input dim d = |ψ| + |ρ| = 6 + 6 = 12.
    Hidden dim H (default 10).  Parameters ≈ H·d + H + H + 1
    ≈ 131 at H=10.

    Weights stored as numpy arrays; ``to_dict`` serialises as
    nested lists for JSON portability.
    """

    W1: np.ndarray = dataclasses.field(
        default_factory=lambda: np.zeros((1, 1)))
    b1: np.ndarray = dataclasses.field(
        default_factory=lambda: np.zeros(1))
    w2: np.ndarray = dataclasses.field(
        default_factory=lambda: np.zeros(1))
    b2: float = 0.0
    rc_alphabet: tuple[str, ...] = ()
    claim_to_root_cause: dict[str, str] = dataclasses.field(
        default_factory=lambda: dict(DEFAULT_CLAIM_TO_ROOT_CAUSE))
    priority_order: tuple[str, ...] = DEFAULT_PRIORITY_ORDER
    high_priority_cutoff: int = DEFAULT_HIGH_PRIORITY_CUTOFF
    hidden_size: int = 10
    unknown_label: str = UNKNOWN
    name: str = "cohort_relational_decoder"

    def _input_vector(self, bundle, rc):
        return _cohort_relational_input_vector(
            bundle, rc, self.claim_to_root_cause,
            self.priority_order, self.high_priority_cutoff,
            self.rc_alphabet)

    def _score(self, bundle, rc):
        x = self._input_vector(bundle, rc)
        h = np.tanh(self.W1 @ x + self.b1)
        return float(self.w2 @ h + self.b2)

    def decode(self, admitted):
        if not self.rc_alphabet:
            return self.unknown_label
        scores = {rc: self._score(admitted, rc)
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
            "psi_features": list(COHORT_PSI_FEATURES),
            "rho_features": list(COHORT_RHO_FEATURES),
            "rc_alphabet": list(self.rc_alphabet),
            "claim_to_root_cause": dict(self.claim_to_root_cause),
            "priority_order": list(self.priority_order),
            "high_priority_cutoff": int(self.high_priority_cutoff),
        }


def train_cohort_relational_decoder(
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
        ) -> CohortRelationalDecoder:
    """Train a ``CohortRelationalDecoder`` by full-batch softmax-
    cross-entropy over per-rc input vectors.

    Deterministic in ``seed`` (numpy RandomState controls the
    W1/w2 initialisation).
    """
    if not examples:
        raise ValueError(
            "train_cohort_relational_decoder: no examples")
    alphabet = tuple(rc_alphabet)
    d = len(COHORT_RELATIONAL_FEATURES)
    H = int(hidden_size)
    n = len(examples)
    K = len(alphabet)
    X = np.zeros((n, K, d), dtype=np.float64)
    y = np.zeros(n, dtype=np.int64)
    for i, (bundle, gold_rc) in enumerate(examples):
        for j, rc in enumerate(alphabet):
            X[i, j, :] = _cohort_relational_input_vector(
                bundle, rc, claim_to_root_cause,
                priority_order, high_priority_cutoff,
                alphabet)
        y[i] = (alphabet.index(gold_rc)
                 if gold_rc in alphabet else -1)
    valid = y >= 0
    rng = np.random.RandomState(seed)
    W1 = rng.randn(H, d) * 0.1
    b1 = np.zeros(H)
    w2 = rng.randn(H) * 0.1
    b2 = 0.0
    n_valid = max(1, int(valid.sum()))
    for _ep in range(n_epochs):
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
    return CohortRelationalDecoder(
        W1=W1, b1=b1, w2=w2, b2=float(b2),
        hidden_size=H,
        rc_alphabet=alphabet,
        claim_to_root_cause=dict(claim_to_root_cause),
        priority_order=tuple(priority_order),
        high_priority_cutoff=int(high_priority_cutoff),
    )


__all__ = [
    "COHORT_PSI_FEATURES",
    "COHORT_RHO_FEATURES",
    "COHORT_RELATIONAL_FEATURES",
    "CohortRelationalDecoder",
    "train_cohort_relational_decoder",
]
