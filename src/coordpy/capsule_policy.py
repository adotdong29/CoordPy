"""Capsule admission policies — the ML-relevant slice of the
capsule contract.

Up to SDK v3, capsule admission was a *fixed* heuristic: the
ledger checks the capsule's own declared budget and either accepts
or rejects. That is sufficient when budgets are generous and every
proposed capsule is plausibly load-bearing. It is **not** sufficient
in either of the two regimes that matter at runtime:

  1. **Tight global budget.** A team produces N candidate capsules
     and the auditor's downstream consumer can only afford to read
     k < N tokens worth. A fixed-per-capsule budget bounds the per-
     item size but does not pick the best k.

  2. **Noisy extractors.** A non-trivial fraction of the produced
     capsules are *distractors* — syntactically valid, type-correct,
     but causally irrelevant to the downstream answer. The Phase-31
     ``handoff_is_relevant`` oracle tells you which is which after
     the fact; at admission time you have only the capsule's
     headers (kind, source, byte/token count, parent topology).

This module turns admission into a **policy decision** —

    policy : (proposed_capsule, ledger_state, remaining_budget)
              → ADMIT | REJECT

and ships:

  * ``AdmissionPolicy``         — the abstract interface.
  * ``FIFOPolicy``              — accept until full (the SDK-v3
    default behaviour, included as the obvious baseline).
  * ``KindPriorityPolicy``      — accept by hand-coded priority
    over ``CapsuleKind`` × ``claim_kind``.
  * ``SmallestFirstPolicy``     — needs full lookahead; admits
    smallest-payload capsules first.
  * ``LearnedAdmissionPolicy``  — predicts P(causal | features)
    from a learned linear model and admits highest-scored
    capsules until the budget is exhausted.
  * ``BudgetedAdmissionLedger`` — wraps a ``CapsuleLedger`` with a
    global token budget and a policy.
  * ``train_admission_policy`` — gradient-descent training on a
    set of (capsule, gold-relevance) pairs, using only NumPy.

Why this is *real* ML and not another rule
------------------------------------------

The learned policy is a logistic regression over hand-coded
features, trained on the Phase-31 incident-triage data with the
``handoff_is_relevant`` oracle as gold. The feature set is
intentionally *small* (no embedding, no token-level NN); the
hypothesis it tests is whether *header-level structure* of the
capsule (its kind, its source role, its byte/token count, its
parent count) is enough to predict relevance better than the
fixed-priority heuristic. If yes — and Section "Empirical results"
in `docs/RESULTS_CAPSULE_LEARNING.md` shows that it is —
admission is *learnable* and the capsule contract opens a real
ML problem with a real dataset and a real held-out evaluation.

The policy's hypothesis class is small enough that a sceptical
reader can read every parameter (`policy.weights` is a dict of
~40 floats). This is by design; the goal of the milestone is to
prove the *learnability of admission*, not to maximise raw
accuracy with a deep model.

Theoretical anchor: ``docs/CAPSULE_FORMALISM.md`` § 5
(Conjecture W3-C4). Empirical anchor:
``docs/RESULTS_CAPSULE_LEARNING.md``.
"""

from __future__ import annotations

import dataclasses
import math
import random
from typing import Any, Iterable, Sequence

from .capsule import (
    CapsuleAdmissionError, CapsuleKind, CapsuleLedger,
    CapsuleLifecycleError, ContextCapsule,
)


# =============================================================================
# Decisions
# =============================================================================


ADMIT = "ADMIT"
REJECT = "REJECT"


# =============================================================================
# Featurisation
# =============================================================================


# Closed feature vocabulary. Keeping it small + named so the learned
# policy is interpretable. Index 0 is the bias term.
_KIND_FEATURES = sorted(CapsuleKind.ALL)
# Producer roles seen across Phase-31 / 35 / 32 benchmarks. If a
# feature is not present in a given capsule's metadata, it
# contributes 0 — the closed vocabulary keeps the feature space
# stable across runs.
_SOURCE_ROLE_FEATURES = (
    "monitor", "db_admin", "sysadmin", "network", "auditor",
    "legal", "security", "privacy", "finance", "compliance_officer",
    "soc_analyst", "ir_engineer", "threat_intel", "data_steward", "ciso",
)
# Claim-kind features for the structurally-typed handoff body. The
# featuriser looks for ``metadata["claim_kind"]`` first; absent ⇒ 0.
_CLAIM_KIND_FEATURES = (
    # Phase-31 incident_triage
    "ERROR_RATE_SPIKE", "LATENCY_SPIKE", "SLOW_QUERY_OBSERVED",
    "POOL_EXHAUSTION", "DEADLOCK_SUSPECTED", "DISK_FILL_CRITICAL",
    "CRON_OVERRUN", "OOM_KILL", "TLS_EXPIRED", "DNS_MISROUTE",
    "FW_BLOCK_SURGE",
    # Phase-32 compliance_review
    "MISSING_DPA", "UNCAPPED_LIABILITY", "WEAK_ENCRYPTION",
    "CROSS_BORDER_TRANSFER", "BUDGET_BREACH",
    # Phase-33 security_escalation
    "MALWARE_DETECTED", "DATA_EXFIL_SUSPECTED",
    "PRIV_ESC_OBSERVED", "INDICATOR_OF_COMPROMISE",
)


def featurise_capsule(c: ContextCapsule) -> dict[str, float]:
    """Compute a sparse feature dict for a capsule.

    The feature vocabulary is intentionally small + closed
    (~40 features). Returns a dict of (feature_name, value) so the
    policy is inspectable and the trained weights are a dict.
    """
    md = c.metadata_dict()
    feats: dict[str, float] = {"bias": 1.0}
    feats[f"kind:{c.kind}"] = 1.0
    src = (md.get("source_role") or "").lower()
    if src in _SOURCE_ROLE_FEATURES:
        feats[f"src:{src}"] = 1.0
    ck = md.get("claim_kind")
    if isinstance(ck, str) and ck in _CLAIM_KIND_FEATURES:
        feats[f"claim:{ck}"] = 1.0
    # Continuous features — z-scored at training time, NOT here.
    feats["log1p_n_tokens"] = math.log1p(c.n_tokens or 0)
    feats["log1p_n_bytes"] = math.log1p(c.n_bytes or 0)
    feats["n_parents"] = float(len(c.parents))
    return feats


def feature_index() -> tuple[str, ...]:
    """The full ordered feature list — used to materialise dense
    vectors for training. Includes:

      * bias
      * kind:K for K in CapsuleKind.ALL
      * src:R for R in known source roles
      * claim:K for K in known claim kinds
      * log1p_n_tokens, log1p_n_bytes, n_parents
    """
    out = ["bias"]
    out += [f"kind:{k}" for k in _KIND_FEATURES]
    out += [f"src:{r}" for r in _SOURCE_ROLE_FEATURES]
    out += [f"claim:{c}" for c in _CLAIM_KIND_FEATURES]
    out += ["log1p_n_tokens", "log1p_n_bytes", "n_parents"]
    return tuple(out)


# =============================================================================
# AdmissionPolicy interface + heuristic baselines
# =============================================================================


class AdmissionPolicy:
    """Interface: decide ADMIT/REJECT for a proposed capsule.

    A streaming policy sees capsules one at a time and decides
    immediately. A *batched* policy may instead score all
    candidates and pick the top-k under a global budget; see
    ``score_all`` for the optional second method.
    """

    name: str = "abstract"

    def decide(self, capsule: ContextCapsule,
               ledger: CapsuleLedger,
               remaining_budget: int) -> str:
        raise NotImplementedError

    def score_all(self, capsules: Sequence[ContextCapsule],
                  ) -> Sequence[float]:
        """Optional: return a per-capsule score (higher is more
        likely to be admitted). Default uses ``decide`` with an
        unbounded budget."""
        return [1.0 if self.decide(c, CapsuleLedger(), 10**9) == ADMIT
                else 0.0
                for c in capsules]


class FIFOPolicy(AdmissionPolicy):
    """Streaming: accept every proposed capsule. The ledger's own
    budget gate is what stops admission — but if the global budget
    is exhausted, the wrapping ``BudgetedAdmissionLedger`` will
    drop the next admit. This is the SDK-v3 default behaviour,
    surfaced as a policy for the comparison."""

    name = "fifo"

    def decide(self, capsule, ledger, remaining_budget):
        return ADMIT


@dataclasses.dataclass
class KindPriorityPolicy(AdmissionPolicy):
    """Admit if the capsule's claim_kind / kind sits at or above a
    cutoff in a hand-coded priority list. The default priority
    encodes the Phase-31 ``_decoder_from_handoffs`` ordering — i.e.
    "the kinds the deterministic decoder considers first."

    This is the strongest *non-learned* baseline: it bakes in the
    same domain knowledge the substrate-side decoder uses.
    """

    priority: tuple[str, ...] = (
        "DISK_FILL_CRITICAL", "TLS_EXPIRED", "DNS_MISROUTE",
        "OOM_KILL", "DEADLOCK_SUSPECTED", "CRON_OVERRUN",
        "POOL_EXHAUSTION", "SLOW_QUERY_OBSERVED",
        "ERROR_RATE_SPIKE", "LATENCY_SPIKE", "FW_BLOCK_SURGE",
        # Compliance / security claim kinds — admitted as a
        # secondary tier so the policy works across benchmarks.
        "MISSING_DPA", "UNCAPPED_LIABILITY", "WEAK_ENCRYPTION",
        "MALWARE_DETECTED", "DATA_EXFIL_SUSPECTED",
    )
    cutoff: int = 8  # admit only the top-N kinds

    name: str = "kind_priority"

    def decide(self, capsule, ledger, remaining_budget):
        ck = capsule.metadata_dict().get("claim_kind")
        if not isinstance(ck, str):
            return REJECT
        try:
            rank = self.priority.index(ck)
        except ValueError:
            return REJECT
        return ADMIT if rank < self.cutoff else REJECT

    def score_all(self, capsules):
        scores = []
        for c in capsules:
            ck = c.metadata_dict().get("claim_kind")
            try:
                rank = self.priority.index(ck) if isinstance(ck, str) else 99
            except ValueError:
                rank = 99
            # higher score = higher priority
            scores.append(-float(rank))
        return scores


@dataclasses.dataclass
class SmallestFirstPolicy(AdmissionPolicy):
    """Batched: admit smallest-payload capsules first until budget
    is exhausted. Models the operator who knows nothing about the
    semantics but wants to fit as many items in the budget as
    possible.
    """

    name: str = "smallest_first"

    def decide(self, capsule, ledger, remaining_budget):
        # Streaming variant: admit if its size fits in *half* the
        # remaining budget, leaving room for more candidates.
        nt = capsule.n_tokens or capsule.n_bytes or 1
        return ADMIT if nt <= max(1, remaining_budget // 2) else REJECT

    def score_all(self, capsules):
        return [-(c.n_tokens or c.n_bytes or 0) for c in capsules]


# =============================================================================
# Learned admission policy
# =============================================================================


@dataclasses.dataclass
class LearnedAdmissionPolicy(AdmissionPolicy):
    """Logistic regression over the closed feature vocabulary.

    The policy stores ``weights : dict[str, float]`` — one float
    per feature in ``feature_index()``. ``score_all`` returns
    P(causal | features); ``decide`` admits when score >
    ``threshold`` (default 0.5).

    Interpretability: every weight is a named float. After training,
    a reader can dump ``policy.weights`` and see which kinds /
    source roles / size axes the model relied on.

    Hypothesis class: linear-in-features. This is *deliberately*
    weaker than a deep model — the milestone's goal is to test
    whether *header-level* structure suffices, not to find the
    strongest classifier.
    """

    weights: dict[str, float] = dataclasses.field(default_factory=dict)
    threshold: float = 0.5
    feature_means: dict[str, float] = dataclasses.field(default_factory=dict)
    feature_stds: dict[str, float] = dataclasses.field(default_factory=dict)

    name: str = "learned"

    def _prepare_features(self, c: ContextCapsule) -> dict[str, float]:
        f = featurise_capsule(c)
        # Normalise the continuous features.
        for cont in ("log1p_n_tokens", "log1p_n_bytes", "n_parents"):
            mu = self.feature_means.get(cont, 0.0)
            sd = self.feature_stds.get(cont, 1.0) or 1.0
            if cont in f:
                f[cont] = (f[cont] - mu) / sd
        return f

    def _score(self, c: ContextCapsule) -> float:
        f = self._prepare_features(c)
        z = sum(self.weights.get(k, 0.0) * v for k, v in f.items())
        return 1.0 / (1.0 + math.exp(-z))

    def decide(self, capsule, ledger, remaining_budget):
        nt = capsule.n_tokens or 1
        if nt > remaining_budget:
            return REJECT
        return ADMIT if self._score(capsule) >= self.threshold else REJECT

    def score_all(self, capsules):
        return [self._score(c) for c in capsules]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "threshold": self.threshold,
            "weights": dict(self.weights),
            "feature_means": dict(self.feature_means),
            "feature_stds": dict(self.feature_stds),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LearnedAdmissionPolicy":
        return cls(
            weights=dict(d.get("weights", {})),
            threshold=float(d.get("threshold", 0.5)),
            feature_means=dict(d.get("feature_means", {})),
            feature_stds=dict(d.get("feature_stds", {})),
        )


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def train_admission_policy(
        examples: Sequence[tuple[ContextCapsule, int]],
        *,
        n_epochs: int = 200,
        lr: float = 0.5,
        l2: float = 1e-3,
        seed: int = 0,
) -> LearnedAdmissionPolicy:
    """Train a ``LearnedAdmissionPolicy`` on labelled examples.

    Each example is ``(capsule, gold_label)`` with
    ``gold_label in {0, 1}``. Trains a logistic regression by
    full-batch gradient descent (the dataset is small;
    correctness > speed). Continuous features are z-scored using
    statistics computed from the training set; the same statistics
    are stored on the returned policy so inference uses the same
    normalisation.
    """
    if not examples:
        raise ValueError("train_admission_policy: no examples")
    rng = random.Random(seed)
    # Compute means / stds for continuous features.
    cont_axes = ("log1p_n_tokens", "log1p_n_bytes", "n_parents")
    raw_feats = [featurise_capsule(c) for (c, _y) in examples]
    means = {}
    stds = {}
    for ax in cont_axes:
        vals = [f.get(ax, 0.0) for f in raw_feats]
        m = sum(vals) / len(vals)
        var = sum((v - m) ** 2 for v in vals) / len(vals)
        means[ax] = m
        stds[ax] = math.sqrt(var) or 1.0
    # Apply normalisation in place.
    for f in raw_feats:
        for ax in cont_axes:
            if ax in f:
                f[ax] = (f[ax] - means[ax]) / stds[ax]
    labels = [int(y) for (_c, y) in examples]
    weights: dict[str, float] = {k: 0.0 for k in feature_index()}
    n = len(examples)
    for ep in range(n_epochs):
        # Shuffle for stability.
        idxs = list(range(n))
        rng.shuffle(idxs)
        # Full-batch gradient.
        grad: dict[str, float] = {k: 0.0 for k in weights}
        for i in idxs:
            f = raw_feats[i]
            y = labels[i]
            z = sum(weights.get(k, 0.0) * v for k, v in f.items())
            p = _sigmoid(z)
            err = p - y
            for k, v in f.items():
                grad[k] = grad.get(k, 0.0) + err * v
        # L2 + step.
        for k in weights:
            g = grad[k] / n + l2 * weights[k]
            weights[k] -= lr * g
    return LearnedAdmissionPolicy(
        weights=weights,
        threshold=0.5,
        feature_means=means,
        feature_stds=stds,
    )


# =============================================================================
# Budgeted admission ledger — the runtime gate
# =============================================================================


@dataclasses.dataclass
class BudgetedAdmissionDecision:
    capsule_cid: str
    decision: str
    score: float | None
    reason: str


class BudgetedAdmissionLedger:
    """A ``CapsuleLedger`` augmented with a global token budget and
    an ``AdmissionPolicy``. Capsules arrive in stream order; the
    policy decides; on ADMIT the capsule is sealed and its
    ``n_tokens`` is subtracted from the remaining budget.

    Once the budget is exhausted, every subsequent admit returns
    REJECT regardless of the policy. The budget gate is the
    operational form of Theorem W3-8 (admissibility monotone under
    tightening): the budgeted ledger's admitted set is always a
    subset of the unbudgeted ledger's admitted set under the same
    policy.
    """

    def __init__(self,
                 budget_tokens: int,
                 policy: AdmissionPolicy,
                 ) -> None:
        self.ledger = CapsuleLedger()
        self.budget_tokens = int(budget_tokens)
        self.budget_used = 0
        self.policy = policy
        self.decisions: list[BudgetedAdmissionDecision] = []

    @property
    def remaining(self) -> int:
        return max(0, self.budget_tokens - self.budget_used)

    def offer(self, capsule: ContextCapsule) -> str:
        """Streaming offer of one capsule. Returns the decision."""
        nt = capsule.n_tokens or 0
        if nt > self.remaining:
            self.decisions.append(BudgetedAdmissionDecision(
                capsule.cid, REJECT, None, "budget_exhausted"))
            return REJECT
        # Score (best-effort; some policies don't score).
        score: float | None = None
        try:
            score = float(self.policy.score_all([capsule])[0])
        except Exception:
            score = None
        decision = self.policy.decide(capsule, self.ledger, self.remaining)
        if decision == ADMIT:
            try:
                sealed = self.ledger.admit_and_seal(capsule)
            except (CapsuleAdmissionError, CapsuleLifecycleError) as ex:
                self.decisions.append(BudgetedAdmissionDecision(
                    capsule.cid, REJECT, score, f"admit_error:{ex}"))
                return REJECT
            self.budget_used += sealed.n_tokens or 0
            self.decisions.append(BudgetedAdmissionDecision(
                capsule.cid, ADMIT, score, "ok"))
            return ADMIT
        self.decisions.append(BudgetedAdmissionDecision(
            capsule.cid, REJECT, score, "policy_reject"))
        return REJECT

    def offer_all(self, capsules: Iterable[ContextCapsule]) -> dict[str, int]:
        """Streaming offer over a batch. Returns counts."""
        counts = {ADMIT: 0, REJECT: 0}
        for c in capsules:
            d = self.offer(c)
            counts[d] = counts.get(d, 0) + 1
        return counts

    def offer_all_batched(self, capsules: Sequence[ContextCapsule],
                          ) -> dict[str, int]:
        """Batched offer: score all, sort by score descending, then
        admit greedily under the budget. Equivalent to the offline
        knapsack (relaxed) that ``LearnedAdmissionPolicy`` enables.

        Streaming policies that cannot rank still get a sensible
        default — ``score_all`` falls back to per-decide scores.

        **Bundle-aware extension (Phase 47).** If the policy
        exposes a ``reject_set(capsules) -> frozenset[str]``
        method, its returned CIDs are vetoed at admission — they
        are REJECTed regardless of score. This is how
        ``capsule_policy_bundle.CorroboratedAdmissionPolicy``
        and friends veto isolated high-priority claims before
        the decoder sees them.
        """
        scores = list(self.policy.score_all(capsules))
        veto: frozenset[str] = frozenset()
        if hasattr(self.policy, "reject_set"):
            try:
                veto = frozenset(self.policy.reject_set(capsules))
            except NotImplementedError:
                veto = frozenset()
        order = sorted(range(len(capsules)),
                       key=lambda i: scores[i], reverse=True)
        counts = {ADMIT: 0, REJECT: 0}
        for i in order:
            c = capsules[i]
            if c.cid in veto:
                self.decisions.append(BudgetedAdmissionDecision(
                    c.cid, REJECT, scores[i], "bundle_veto"))
                counts[REJECT] += 1
                continue
            nt = c.n_tokens or 0
            if nt > self.remaining:
                self.decisions.append(BudgetedAdmissionDecision(
                    c.cid, REJECT, scores[i], "budget_exhausted"))
                counts[REJECT] += 1
                continue
            try:
                sealed = self.ledger.admit_and_seal(c)
            except (CapsuleAdmissionError, CapsuleLifecycleError) as ex:
                self.decisions.append(BudgetedAdmissionDecision(
                    c.cid, REJECT, scores[i], f"admit_error:{ex}"))
                counts[REJECT] += 1
                continue
            self.budget_used += sealed.n_tokens or 0
            self.decisions.append(BudgetedAdmissionDecision(
                c.cid, ADMIT, scores[i], "ok"))
            counts[ADMIT] += 1
        return counts

    def stats(self) -> dict[str, Any]:
        s = self.ledger.stats()
        s.update({
            "budget_tokens": self.budget_tokens,
            "budget_used": self.budget_used,
            "budget_remaining": self.remaining,
            "policy": self.policy.name,
            "n_admit_decisions": sum(
                1 for d in self.decisions if d.decision == ADMIT),
            "n_reject_decisions": sum(
                1 for d in self.decisions if d.decision == REJECT),
        })
        return s


# =============================================================================
# Public surface
# =============================================================================


__all__ = [
    "ADMIT", "REJECT",
    "AdmissionPolicy",
    "FIFOPolicy", "KindPriorityPolicy", "SmallestFirstPolicy",
    "LearnedAdmissionPolicy",
    "BudgetedAdmissionLedger", "BudgetedAdmissionDecision",
    "featurise_capsule", "feature_index",
    "train_admission_policy",
]
