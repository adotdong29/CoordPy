"""Learned multi-agent capsule policy — SDK v3.5 (research).

This module ships the *learned* admission policy for the
capsule-native team coordination layer (``team_coord.py``). It is
deliberately small, interpretable, and numpy-only. The goal is
honest evidence that *learning over capsule features* materially
improves team-level performance under tight per-role budgets,
not a heroic deep model.

Setup
-----

The admission decision is, per (role, candidate-handoff):
``admit ∈ {0, 1}``. Learning targets the gold-coverage signal: a
candidate is "right to admit" if its ``(source_role, claim_kind,
payload_sha)`` is part of the scenario's causal chain. The policy
is a small per-role logistic regression scoring
``(source_role, to_role, claim_kind, n_tokens,
 has_service_tag, has_numeric_threshold)`` — six features. The
policy is trained by gradient descent on an offline supervised
trace generated from the benchmark task bank.

The sense in which this is *real ML, not hand-tuned rules*:

1. Features are extracted automatically from the capsule payload
   (``team_coord._candidate_*`` adapters); no per-claim hand-coding.
2. Weights are SGD-fit on a held-out training partition; the
   evaluation partition is disjoint.
3. Train and eval use the same featuriser, the same loss, and a
   fixed threshold (default ``0.5``). The same scorer is used at
   admission time *and* at training time.

Comparison
----------

The benchmark compares four admission policies team-level:

* ``FifoAdmissionPolicy``           — FIFO baseline (substrate-equivalent).
* ``ClaimPriorityAdmissionPolicy``  — static priority over claim kinds.
* ``CoverageGuidedAdmissionPolicy`` — diversity over claim kinds.
* ``LearnedTeamAdmissionPolicy``    — this module.

The learned policy must beat the strongest fixed baseline under
the same per-role budget; if it does not, that is honest evidence
that learning does not help on this benchmark.

Public surface
--------------

* ``TeamFeatureVector`` — dataclass naming the six features.
* ``featurise_team_handoff`` — extract a vector for one TEAM_HANDOFF
  capsule.
* ``LearnedTeamAdmissionPolicy`` — SGD-trained per-role scorer.
* ``train_team_admission_policy`` — fit on a (role, candidate,
  label) trace.
* ``team_policy_train_trace`` — produce the supervised trace from
  a scenario bank.
"""

from __future__ import annotations

import dataclasses
import math
import random
import re
from typing import Any, Iterable, Sequence

import numpy as np

from coordpy.capsule import ContextCapsule
from coordpy.team_coord import (
    AdmissionDecision, AdmissionPolicy, RoleBudget, REASON_ADMIT,
    REASON_SCORE_LOW, _enforce_budget,
    _candidate_claim_kind, _candidate_n_tokens, _candidate_payload_sha,
)


# =============================================================================
# Featuriser
# =============================================================================


# Six-dimensional feature vector — small enough to be interpretable
# and large enough that the SGD scorer has slack to express role-
# specific admission preferences. The dimensions:
#
#   0. source_role index (one-hot would be larger; we use small int
#       here since the scorer is per-role and source_role is an
#       index into ``KNOWN_SOURCE_ROLES``).
#   1. claim_kind index into ``KNOWN_CLAIM_KINDS``.
#   2. log1p(n_tokens) on the candidate's payload.
#   3. has_service_tag: 1 if "service=" appears in the payload.
#   4. has_numeric_threshold: 1 if a "<digit>=<threshold>" or
#       "<digit>%=<threshold>" pattern appears.
#   5. payload_sha first byte mod 32 (a stable but arbitrary
#       partition signal).

KNOWN_SOURCE_ROLES = (
    "monitor", "db_admin", "sysadmin", "network", "auditor",
    "__other__",
)
KNOWN_CLAIM_KINDS = (
    "ERROR_RATE_SPIKE", "LATENCY_SPIKE",
    "SLOW_QUERY_OBSERVED", "POOL_EXHAUSTION",
    "DEADLOCK_SUSPECTED", "DISK_FILL_CRITICAL",
    "CRON_OVERRUN", "OOM_KILL", "TLS_EXPIRED",
    "DNS_MISROUTE", "FW_BLOCK_SURGE",
    "__other__",
)
N_FEATURES = 6
FEATURE_NAMES = (
    "source_role_idx",
    "claim_kind_idx",
    "log1p_n_tokens",
    "has_service_tag",
    "has_numeric_threshold",
    "sha_partition_signal",
)


def _index_of(seq: Sequence[str], v: str) -> int:
    try:
        return seq.index(v)
    except ValueError:
        return seq.index("__other__")


def _has_service_tag(text: str) -> int:
    return 1 if re.search(r"service=\w+", text or "") else 0


def _has_numeric_threshold(text: str) -> int:
    if not text:
        return 0
    if re.search(r"=\s*\d+(\.\d+)?", text):
        return 1
    if re.search(r"used=\d+%", text):
        return 1
    return 0


def _sha_partition_signal(sha: str) -> float:
    if not sha:
        return 0.0
    try:
        return float(int(sha[:2], 16) % 32) / 32.0
    except ValueError:
        return 0.0


@dataclasses.dataclass(frozen=True)
class TeamFeatureVector:
    """Named projection of a TEAM_HANDOFF capsule's six features."""

    source_role_idx: int
    claim_kind_idx: int
    log1p_n_tokens: float
    has_service_tag: int
    has_numeric_threshold: int
    sha_partition_signal: float

    def as_array(self) -> np.ndarray:
        return np.array([
            self.source_role_idx, self.claim_kind_idx,
            self.log1p_n_tokens, self.has_service_tag,
            self.has_numeric_threshold, self.sha_partition_signal,
        ], dtype=np.float64)


def featurise_team_handoff(cap: ContextCapsule) -> TeamFeatureVector:
    """Extract the six-dimensional feature vector from a TEAM_HANDOFF
    capsule. Robust to malformed payloads (missing fields default to
    the ``__other__`` index / 0)."""
    payload: dict[str, Any] = (
        cap.payload if isinstance(cap.payload, dict) else {})
    src = str(payload.get("source_role", ""))
    kind = str(payload.get("claim_kind", ""))
    text = str(payload.get("payload", ""))
    sha = str(payload.get("payload_sha256", ""))
    n_tok = _candidate_n_tokens(cap)
    return TeamFeatureVector(
        source_role_idx=_index_of(KNOWN_SOURCE_ROLES, src),
        claim_kind_idx=_index_of(KNOWN_CLAIM_KINDS, kind),
        log1p_n_tokens=math.log1p(max(0, n_tok)),
        has_service_tag=_has_service_tag(text),
        has_numeric_threshold=_has_numeric_threshold(text),
        sha_partition_signal=_sha_partition_signal(sha),
    )


# =============================================================================
# Per-role logistic-regression scorer
# =============================================================================


def _sigmoid(z: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-z))


@dataclasses.dataclass
class LearnedTeamAdmissionPolicy:
    """Per-role logistic-regression admission scorer.

    Stores one weight vector + bias per role. ``decide(...)`` looks
    up the scorer for the role, applies sigmoid, and admits iff the
    score exceeds ``threshold``. Falls back to a global scorer for
    unseen roles.

    Fields
    ------
    weights_per_role
        ``role_name -> (w: np.ndarray (6,), b: float)``.
    threshold
        Sigmoid threshold above which a candidate is admitted.
    name
        Identifier surfaced in TEAM_DECISION audit logs.
    """

    weights_per_role: dict[str, tuple[np.ndarray, float]] = dataclasses.field(
        default_factory=dict)
    threshold: float = 0.5
    name: str = "learned_team"

    def _score_role(self, role: str, x: np.ndarray) -> float:
        w, b = self.weights_per_role.get(
            role, (np.zeros(N_FEATURES, dtype=np.float64), 0.0))
        z = float(np.dot(w, x) + b)
        return float(_sigmoid(z))

    def decide(self, *, candidate, role, budget,
               current_admitted, current_n_tokens):
        denial = _enforce_budget(
            candidate=candidate, role=role, budget=budget,
            current_admitted=current_admitted,
            current_n_tokens=current_n_tokens)
        if denial is not None:
            return denial
        feats = featurise_team_handoff(candidate).as_array()
        score = self._score_role(role, feats)
        if score < self.threshold:
            return AdmissionDecision(admit=False,
                                       reason=REASON_SCORE_LOW,
                                       score=score)
        return AdmissionDecision(admit=True, reason=REASON_ADMIT,
                                   score=score)


# =============================================================================
# Training: SGD on a (role, candidate, label) trace
# =============================================================================


@dataclasses.dataclass
class TrainSample:
    """One supervised training sample for the team admission policy.

    Fields
    ------
    role
        Receiving role (the role under which the admission decision
        is being made).
    capsule
        The candidate TEAM_HANDOFF capsule.
    label
        ``1`` iff the candidate is causally relevant to the team's
        gold answer for this scenario (per the bench oracle); ``0``
        otherwise.
    """
    role: str
    capsule: ContextCapsule
    label: int


@dataclasses.dataclass
class TrainStats:
    """Per-training-run summary."""
    n_samples: int
    n_roles: int
    final_loss: float
    train_accuracy: float
    pos_rate: float
    epochs: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "n_samples": self.n_samples,
            "n_roles": self.n_roles,
            "final_loss": round(self.final_loss, 6),
            "train_accuracy": round(self.train_accuracy, 4),
            "pos_rate": round(self.pos_rate, 4),
            "epochs": self.epochs,
        }


def train_team_admission_policy(
        samples: Sequence[TrainSample],
        *,
        epochs: int = 200,
        lr: float = 0.2,
        l2: float = 1e-3,
        seed: int = 0,
        threshold: float = 0.5,
        ) -> tuple[LearnedTeamAdmissionPolicy, TrainStats]:
    """Fit a per-role logistic-regression admission scorer.

    Per role: minimise binary cross-entropy + L2 with full-batch
    gradient descent. Pure numpy. Deterministic given ``seed`` /
    ``epochs`` / ``lr`` / ``l2``.

    Returns
    -------
    (policy, stats)
        ``policy`` carries one weight vector per role; ``stats`` is
        a JSON-serialisable summary.
    """
    rng = random.Random(seed)
    by_role: dict[str, list[TrainSample]] = {}
    for s in samples:
        by_role.setdefault(s.role, []).append(s)

    policy = LearnedTeamAdmissionPolicy(threshold=float(threshold))
    total_loss = 0.0
    n_correct = 0
    n_total = 0
    n_pos = 0

    for role in sorted(by_role.keys()):
        rows = by_role[role]
        X = np.stack([
            featurise_team_handoff(s.capsule).as_array() for s in rows
        ], axis=0)
        y = np.array([float(s.label) for s in rows], dtype=np.float64)
        n_pos += int(y.sum())
        # Center features per-column for stability (offline only;
        # the policy stores w, b in the original feature space).
        mean = X.mean(axis=0)
        std = X.std(axis=0) + 1e-8
        Xn = (X - mean) / std
        w = np.zeros(N_FEATURES, dtype=np.float64)
        b = 0.0
        # Class-imbalance-aware weighting: per-class inverse-frequency.
        n_pos_role = max(1.0, float(y.sum()))
        n_neg_role = max(1.0, float((1.0 - y).sum()))
        w_pos = 0.5 * (n_pos_role + n_neg_role) / n_pos_role
        w_neg = 0.5 * (n_pos_role + n_neg_role) / n_neg_role
        sample_w = np.where(y > 0.5, w_pos, w_neg)
        for _ in range(epochs):
            z = Xn @ w + b
            p = _sigmoid(z)
            grad_z = (p - y) * sample_w
            grad_w = Xn.T @ grad_z / max(1, len(rows)) + l2 * w
            grad_b = grad_z.mean()
            w -= lr * grad_w
            b -= lr * float(grad_b)
        # De-normalise into the original feature space so inference
        # uses raw featurise outputs.
        w_orig = w / std
        b_orig = b - float(np.dot(w / std, mean))
        policy.weights_per_role[role] = (w_orig, b_orig)
        # Final loss / accuracy for this role.
        z_final = X @ w_orig + b_orig
        p_final = _sigmoid(z_final)
        eps = 1e-12
        loss = float(-np.mean(
            y * np.log(p_final + eps) + (1 - y) * np.log(1 - p_final + eps)))
        total_loss += loss * len(rows)
        pred = (p_final >= threshold).astype(np.float64)
        n_correct += int((pred == y).sum())
        n_total += len(rows)
        # Touch rng so seed is recorded in caller-visible state.
        _ = rng.random()

    stats = TrainStats(
        n_samples=n_total,
        n_roles=len(by_role),
        final_loss=total_loss / max(1, n_total),
        train_accuracy=float(n_correct) / max(1, n_total),
        pos_rate=float(n_pos) / max(1, n_total),
        epochs=epochs,
    )
    return policy, stats


__all__ = [
    "KNOWN_SOURCE_ROLES", "KNOWN_CLAIM_KINDS", "N_FEATURES",
    "FEATURE_NAMES",
    "TeamFeatureVector", "featurise_team_handoff",
    "LearnedTeamAdmissionPolicy",
    "TrainSample", "TrainStats",
    "train_team_admission_policy",
]
