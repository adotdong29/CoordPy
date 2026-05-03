"""Phase 50 — strict zero-shot Gate 2 study.

Phase 49 established two structural facts about zero-shot decoder
transfer on the (incident, security) pair:

  * **Theorem W3-21 (proved, negative).** No class-agnostic
    linear decoder over a feature whose gold-conditional sign
    flips across domains can achieve both per-domain optima
    simultaneously.
  * **Empirical W3-C8'.** Of the 20 V2 features, 9 (the V2
    relative-margin sub-family) have domain-dependent signs on
    (incident, security); the 8-feature sign-stable sub-family
    (absolute counts + base-rate features) has consistent
    signs.

Phase 49's DeepSet reduced the zero-shot gap from V1's 0.175 to
0.038 but left per-direction transfer penalty at +0.138 pp
($i \\to s$) and +0.137 pp ($s \\to i$) — both above the strict
5 pp bar.  This driver attacks the strict-zero-shot half of
Conjecture W3-C7 Gate 2 on three principled routes:

  1. **Sign-stable sub-family decoder** — a
     ``LearnedBundleDecoderV2`` restricted to the 8 features in
     ``SIGN_STABLE_FEATURES_V2``.  Under refined Conjecture
     W3-C8', its features have stable gold-conditional sign
     across domains, so W3-21's structural obstruction does not
     apply.  Prediction: lower within-domain accuracy but
     *symmetric* transfer.
  2. **Source-standardised V2 decoder** — each V2 feature is
     z-scored using statistics computed from the *source*
     domain's training split only.  This removes scale-level
     distribution shift (a consistent "shift + re-scale" is
     absorbed) without using any target-domain information —
     preserving the zero-shot discipline.  Prediction:
     closes scale-driven asymmetries but not sign-flip ones.
  3. **Sign-stable + per-capsule DeepSet** — a DeepSet whose
     per-capsule φ is restricted to sign-stable indicator
     features (``implies_rc``, ``implies_rc_and_high_priority``,
     ``implies_rc_and_unique_source``).  Combines the richer
     hypothesis class with the sign-stable restriction.

Each of the three is a genuine **zero-shot** decoder: trained on
one domain, deployed on the other with ZERO target-domain data,
labels, or statistics.

We also report the Phase-49 baselines for apples-to-apples:
V1 (Phase-48), V2-full (Phase-49), DeepSet-full (Phase-49).

Strict-reading question
-----------------------
Is there any decoder whose zero-shot weight transfer satisfies

$$
\\mathrm{acc}(B, w_A) \\;\\ge\\; \\mathrm{acc}(B, w_B^*)
   \\;-\\; 0.05
\\quad \\text{and} \\quad
\\mathrm{acc}(A, w_B) \\;\\ge\\; \\mathrm{acc}(A, w_A^*)
   \\;-\\; 0.05
$$

on the (incident, security) pair at
$n_{\\rm test} = 80$ per domain?

We also diagnose the fundamental obstruction: on what features
does each family's gold-conditional sign flip?  This is the
empirical half of Theorem W3-21's premise.

Theoretical anchor:
  * ``docs/CAPSULE_FORMALISM.md`` § 4.D Theorem W3-21, W3-22,
    Conjecture W3-C8'.
  * ``docs/CAPSULE_FORMALISM.md`` § 4.E Claim W3-25 and
    Theorem W3-26 (Phase-50 additions).
  * ``docs/RESULTS_CAPSULE_RESEARCH_MILESTONE5.md`` § 2.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
import time
from typing import Any, Callable, Sequence

try:
    import numpy as np
except ImportError as ex:
    raise ImportError("phase50_zero_shot_transfer requires numpy") from ex

if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execvpe(sys.executable, [sys.executable, *sys.argv],
                os.environ)

from vision_mvp.coordpy.capsule import (
    ContextCapsule, capsule_from_handoff,
)
from vision_mvp.coordpy.capsule_decoder import (
    LearnedBundleDecoder, PriorityDecoder,
    evaluate_decoder, train_learned_bundle_decoder,
)
from vision_mvp.coordpy.capsule_decoder_v2 import (
    BUNDLE_DECODER_FEATURES_V2, DEEPSET_PHI_FEATURES,
    _bundle_vote_summary, _featurise_bundle_for_rc_v2,
    _feature_vector_v2, _phi_capsule, _phi_sum,
    LearnedBundleDecoderV2, train_learned_bundle_decoder_v2,
    DeepSetBundleDecoder, train_deep_set_bundle_decoder,
)
from vision_mvp.experiments.phase50_gate1_ci import (
    SIGN_STABLE_FEATURES_V2, train_sign_stable_v2,
)


# =============================================================================
# Domain adapters (reused from Phase 49 symmetric-transfer)
# =============================================================================


@dataclasses.dataclass
class DomainSpec:
    name: str
    build_scenario_bank: Callable
    run_handoff_protocol: Callable
    extract_claims_for_role: Callable
    handoff_is_relevant: Callable
    noisy_known_kinds: Callable
    auditor_role: str
    claim_to_label: dict[str, str]
    priority_order: tuple[str, ...]
    gold_label_fn: Callable
    label_alphabet: tuple[str, ...]


def _incident_spec() -> DomainSpec:
    from vision_mvp.tasks.incident_triage import (
        build_scenario_bank, run_handoff_protocol,
        handoff_is_relevant, extract_claims_for_role,
        ROLE_AUDITOR,
    )
    from vision_mvp.core.extractor_noise import (
        incident_triage_known_kinds,
    )
    from vision_mvp.coordpy.capsule_policy_bundle import (
        DEFAULT_CLAIM_TO_ROOT_CAUSE, DEFAULT_PRIORITY_ORDER,
    )
    return DomainSpec(
        name="incident",
        build_scenario_bank=build_scenario_bank,
        run_handoff_protocol=run_handoff_protocol,
        extract_claims_for_role=extract_claims_for_role,
        handoff_is_relevant=handoff_is_relevant,
        noisy_known_kinds=incident_triage_known_kinds,
        auditor_role=ROLE_AUDITOR,
        claim_to_label=dict(DEFAULT_CLAIM_TO_ROOT_CAUSE),
        priority_order=DEFAULT_PRIORITY_ORDER,
        gold_label_fn=lambda s: s.gold_root_cause,
        label_alphabet=(
            "deadlock", "disk_fill", "dns_misroute",
            "memory_leak", "tls_expiry",
        ),
    )


def _security_spec() -> DomainSpec:
    from vision_mvp.tasks.security_escalation import (
        build_scenario_bank, run_handoff_protocol,
        handoff_is_relevant, extract_claims_for_role,
        ROLE_CISO,
    )
    from vision_mvp.core.extractor_noise import (
        security_escalation_known_kinds,
    )
    scenarios = build_scenario_bank(
        seed=31, distractors_per_role=0)
    claim_to_label: dict[str, str] = {}
    for s in scenarios:
        for (_role, kind, _payload, _evs) in s.causal_chain:
            if kind not in claim_to_label:
                claim_to_label[kind] = s.gold_classification
    order: list[str] = []
    for s in scenarios:
        for (_role, kind, _payload, _evs) in s.causal_chain:
            if kind not in order:
                order.append(kind)
    return DomainSpec(
        name="security",
        build_scenario_bank=build_scenario_bank,
        run_handoff_protocol=run_handoff_protocol,
        extract_claims_for_role=extract_claims_for_role,
        handoff_is_relevant=handoff_is_relevant,
        noisy_known_kinds=security_escalation_known_kinds,
        auditor_role=ROLE_CISO,
        claim_to_label=claim_to_label,
        priority_order=tuple(order),
        gold_label_fn=lambda s: s.gold_classification,
        label_alphabet=tuple(sorted({
            s.gold_classification for s in scenarios})),
    )


ALL_DOMAINS: tuple[DomainSpec, ...] = (
    _incident_spec(), _security_spec())


def collect_domain_instances(
        spec: DomainSpec, seeds: Sequence[int],
        distractor_grid: Sequence[int],
        spurious_prob: float, mislabel_prob: float,
        ) -> list[dict[str, Any]]:
    from vision_mvp.core.extractor_noise import (
        NoiseConfig, noisy_extractor,
    )
    known_kinds = spec.noisy_known_kinds()
    out: list[dict[str, Any]] = []
    for seed in seeds:
        for k in distractor_grid:
            scenarios = spec.build_scenario_bank(
                seed=seed, distractors_per_role=k)
            for scenario in scenarios:
                noise = NoiseConfig(
                    spurious_prob=spurious_prob,
                    mislabel_prob=mislabel_prob,
                    seed=seed * 17 + k
                         + hash(spec.name) % 1000)
                noisy = noisy_extractor(
                    spec.extract_claims_for_role,
                    known_kinds, noise)
                router = spec.run_handoff_protocol(
                    scenario, extractor=noisy)
                inbox = router.inboxes.get(spec.auditor_role)
                if inbox is None:
                    continue
                handoffs = list(inbox.peek())
                offered_capsules = []
                for h in handoffs:
                    cap = capsule_from_handoff(h)
                    cap = dataclasses.replace(
                        cap, n_tokens=h.n_tokens)
                    offered_capsules.append(cap)
                out.append({
                    "domain": spec.name,
                    "scenario_id": scenario.scenario_id,
                    "k": k, "seed": seed,
                    "offered_capsules": offered_capsules,
                    "gold_label": spec.gold_label_fn(scenario),
                })
    return out


def split_by_seed(instances, train_seeds, test_seeds):
    tr = [i for i in instances if i["seed"] in set(train_seeds)]
    te = [i for i in instances if i["seed"] in set(test_seeds)]
    return tr, te


def eval_decoder(instances, decoder):
    pairs = [(i["offered_capsules"], i["gold_label"])
              for i in instances]
    scen = [i["scenario_id"] for i in instances]
    r = evaluate_decoder(pairs, decoder, scenario_keys=scen)
    return r


# =============================================================================
# Source-standardised V2 decoder (Phase-50, Route 2)
# =============================================================================


@dataclasses.dataclass
class StandardisedBundleDecoderV2:
    """A ``LearnedBundleDecoderV2``-shaped decoder that
    z-scores every V2 feature using *source-domain-only*
    statistics at decode time.

    Training: compute per-feature mean/std on the source-domain
    training split (over all (bundle, rc) pairs).  Weights are
    trained on z-scored features.  Decode time: apply the same
    z-score (using the *stored* source-domain statistics) to
    every feature of every bundle, then take the linear score.

    Deployment on a **different** domain B: features are
    z-scored with domain A's statistics, not B's.  This is
    honestly zero-shot — no target-domain data or labels are
    consulted.

    Rationale: if domains A and B differ only by a per-feature
    affine transform $x_B = a_i x_A + b_i$ with *consistent
    sign* (a_i > 0), z-scoring against A kills the offset and
    scale; transfer becomes sign-preserving.  If a_i < 0 (sign
    flip), z-scoring makes the situation worse (flipped features
    are z-scored with the wrong-signed reference), confirming
    Theorem W3-21's obstruction.
    """
    weights: np.ndarray  # shape (d,)
    feat_mean: np.ndarray  # shape (d,)
    feat_std: np.ndarray  # shape (d,)
    rc_alphabet: tuple[str, ...]
    claim_to_root_cause: dict[str, str]
    priority_order: tuple[str, ...]
    high_priority_cutoff: int
    unknown_label: str = "unknown"
    name: str = "standardised_bundle_decoder_v2"

    def _score(self, bundle, rc, summary=None):
        x = _feature_vector_v2(
            bundle, rc, self.claim_to_root_cause,
            self.priority_order, self.high_priority_cutoff,
            summary=summary)
        z = (x - self.feat_mean) / np.maximum(self.feat_std, 1e-6)
        # Keep bias feature un-standardised (z-score of constant
        # is zero which kills the intercept).
        bias_idx = BUNDLE_DECODER_FEATURES_V2.index("bias")
        z[bias_idx] = 1.0
        return float(z @ self.weights)

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


def train_standardised_bundle_decoder_v2(
        examples, rc_alphabet, claim_to_root_cause,
        priority_order, high_priority_cutoff=4,
        n_epochs=500, lr=0.5, l2=1e-3, seed=0,
        ) -> StandardisedBundleDecoderV2:
    """Train a standardised V2 decoder.

    Standardisation uses *source-domain-only* statistics pooled
    over (bundle, rc) pairs.  At decode time (including cross-
    domain deployment) the same statistics are applied.
    """
    alphabet = tuple(rc_alphabet)
    d = len(BUNDLE_DECODER_FEATURES_V2)
    # Collect feature vectors.
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
        y[i] = (alphabet.index(gold_rc)
                 if gold_rc in alphabet else -1)
    # Compute per-feature stats over all (bundle, rc) pairs.
    flat = X.reshape(-1, d)
    feat_mean = flat.mean(axis=0)
    feat_std = flat.std(axis=0)
    # Z-score features.  Keep bias coordinate as 1 (its z-score
    # would be undefined or 0).
    Xz = (X - feat_mean) / np.maximum(feat_std, 1e-6)
    bias_idx = BUNDLE_DECODER_FEATURES_V2.index("bias")
    Xz[..., bias_idx] = 1.0
    # Multinomial logistic regression on z-scored features.
    w = np.zeros(d, dtype=np.float64)
    valid = y >= 0
    n_valid = max(1, int(valid.sum()))
    for _ep in range(n_epochs):
        scores = Xz @ w
        m = scores.max(axis=1, keepdims=True)
        ez = np.exp(scores - m)
        probs = ez / ez.sum(axis=1, keepdims=True)
        grad = np.zeros(d, dtype=np.float64)
        for i in np.where(valid)[0]:
            tgt = np.zeros(K); tgt[y[i]] = 1.0
            grad += Xz[i].T @ (probs[i] - tgt)
        grad = grad / n_valid + l2 * w
        w -= lr * grad
    return StandardisedBundleDecoderV2(
        weights=w,
        feat_mean=feat_mean,
        feat_std=feat_std,
        rc_alphabet=alphabet,
        claim_to_root_cause=dict(claim_to_root_cause),
        priority_order=tuple(priority_order),
        high_priority_cutoff=int(high_priority_cutoff),
    )


# =============================================================================
# Sign-stable DeepSet (Phase-50, Route 3)
# =============================================================================


# φ indices in DEEPSET_PHI_FEATURES.  We retain only the
# "implies_rc"-positive conjuncts, which are sign-stable by
# construction — their gold-conditional value is monotonically
# non-decreasing in "how well does this capsule support rc".
SIGN_STABLE_PHI_IDX: tuple[int, ...] = tuple(
    i for i, f in enumerate(DEEPSET_PHI_FEATURES)
    if f in (
        "phi:implies_rc",
        "phi:implies_rc_and_high_priority",
        "phi:implies_rc_and_unique_source",
        "phi:is_known_kind",
    )
)


@dataclasses.dataclass
class SignStableDeepSetDecoder:
    """Sign-stable DeepSet: per-capsule φ uses only the
    monotonically-rc-supporting conjuncts, concatenated with the
    sign-stable V2 sub-family.  MLP head unchanged.

    Input dim = |stable_phi| (4) + |stable_v2| (8) = 12.
    """
    W1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: float
    rc_alphabet: tuple[str, ...]
    claim_to_root_cause: dict[str, str]
    priority_order: tuple[str, ...]
    high_priority_cutoff: int
    hidden_size: int = 8
    unknown_label: str = "unknown"
    name: str = "sign_stable_deepset_decoder"

    def _input_vector(self, bundle, rc, summary=None):
        phi = _phi_sum(
            bundle, rc, self.claim_to_root_cause,
            self.priority_order, self.high_priority_cutoff)
        phi_stable = phi[list(SIGN_STABLE_PHI_IDX)]
        g = _feature_vector_v2(
            bundle, rc, self.claim_to_root_cause,
            self.priority_order, self.high_priority_cutoff,
            summary=summary)
        stable_v2_idx = [BUNDLE_DECODER_FEATURES_V2.index(f)
                          for f in SIGN_STABLE_FEATURES_V2]
        g_stable = g[stable_v2_idx]
        return np.concatenate([phi_stable, g_stable])

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


def train_sign_stable_deepset(
        examples, rc_alphabet, claim_to_root_cause,
        priority_order, high_priority_cutoff=4,
        hidden_size=8, n_epochs=500, lr=0.1, l2=1e-3, seed=0,
        ) -> SignStableDeepSetDecoder:
    alphabet = tuple(rc_alphabet)
    d_phi = len(SIGN_STABLE_PHI_IDX)
    d_v2 = len(SIGN_STABLE_FEATURES_V2)
    d = d_phi + d_v2
    H = int(hidden_size)
    n = len(examples)
    K = len(alphabet)
    X = np.zeros((n, K, d), dtype=np.float64)
    y = np.zeros(n, dtype=np.int64)
    stable_v2_idx = [BUNDLE_DECODER_FEATURES_V2.index(f)
                      for f in SIGN_STABLE_FEATURES_V2]
    for i, (bundle, gold_rc) in enumerate(examples):
        summary = _bundle_vote_summary(
            bundle, claim_to_root_cause, priority_order,
            high_priority_cutoff)
        for j, rc in enumerate(alphabet):
            phi = _phi_sum(
                bundle, rc, claim_to_root_cause,
                priority_order, high_priority_cutoff)
            phi_stable = phi[list(SIGN_STABLE_PHI_IDX)]
            g = _feature_vector_v2(
                bundle, rc, claim_to_root_cause,
                priority_order, high_priority_cutoff,
                summary=summary)
            g_stable = g[stable_v2_idx]
            X[i, j, :] = np.concatenate([phi_stable, g_stable])
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
    return SignStableDeepSetDecoder(
        W1=W1, b1=b1, w2=w2, b2=float(b2),
        hidden_size=H,
        rc_alphabet=alphabet,
        claim_to_root_cause=dict(claim_to_root_cause),
        priority_order=tuple(priority_order),
        high_priority_cutoff=int(high_priority_cutoff),
    )


# =============================================================================
# Cross-domain "re-target" constructors (copy weights, swap alphabets)
# =============================================================================


def make_cross_v1(src, tgt_spec: DomainSpec):
    return LearnedBundleDecoder(
        weights=dict(src.weights),
        rc_alphabet=tgt_spec.label_alphabet,
        claim_to_root_cause=dict(tgt_spec.claim_to_label),
        priority_order=tuple(tgt_spec.priority_order),
        high_priority_cutoff=src.high_priority_cutoff)


def make_cross_v2(src, tgt_spec: DomainSpec):
    return LearnedBundleDecoderV2(
        weights=dict(src.weights),
        rc_alphabet=tgt_spec.label_alphabet,
        claim_to_root_cause=dict(tgt_spec.claim_to_label),
        priority_order=tuple(tgt_spec.priority_order),
        high_priority_cutoff=src.high_priority_cutoff)


def make_cross_stable(src, tgt_spec: DomainSpec):
    # Same shape as V2 (mask is already baked into weights).
    return LearnedBundleDecoderV2(
        weights=dict(src.weights),
        rc_alphabet=tgt_spec.label_alphabet,
        claim_to_root_cause=dict(tgt_spec.claim_to_label),
        priority_order=tuple(tgt_spec.priority_order),
        high_priority_cutoff=src.high_priority_cutoff)


def make_cross_standardised(src, tgt_spec: DomainSpec):
    return StandardisedBundleDecoderV2(
        weights=src.weights.copy(),
        feat_mean=src.feat_mean.copy(),
        feat_std=src.feat_std.copy(),
        rc_alphabet=tgt_spec.label_alphabet,
        claim_to_root_cause=dict(tgt_spec.claim_to_label),
        priority_order=tuple(tgt_spec.priority_order),
        high_priority_cutoff=src.high_priority_cutoff)


def make_cross_deepset(src, tgt_spec: DomainSpec):
    return DeepSetBundleDecoder(
        W1=src.W1.copy(), b1=src.b1.copy(),
        w2=src.w2.copy(), b2=src.b2,
        hidden_size=src.hidden_size,
        rc_alphabet=tgt_spec.label_alphabet,
        claim_to_root_cause=dict(tgt_spec.claim_to_label),
        priority_order=tuple(tgt_spec.priority_order),
        high_priority_cutoff=src.high_priority_cutoff)


def make_cross_stable_deepset(src, tgt_spec: DomainSpec):
    return SignStableDeepSetDecoder(
        W1=src.W1.copy(), b1=src.b1.copy(),
        w2=src.w2.copy(), b2=src.b2,
        hidden_size=src.hidden_size,
        rc_alphabet=tgt_spec.label_alphabet,
        claim_to_root_cause=dict(tgt_spec.claim_to_label),
        priority_order=tuple(tgt_spec.priority_order),
        high_priority_cutoff=src.high_priority_cutoff)


# =============================================================================
# Main
# =============================================================================


def run_phase50_zero_shot(
        out_dir: str = ".",
        distractor_grid: Sequence[int] = (6, 20, 60, 120),
        train_seeds: Sequence[int] = tuple(range(31, 47)),
        test_seeds: Sequence[int] = tuple(range(47, 51)),
        spurious_prob: float = 0.30,
        mislabel_prob: float = 0.10,
        n_epochs: int = 500,
) -> dict[str, Any]:
    """Phase 50 strict-zero-shot Gate 2 study.

    Default: 16 train seeds (31..46) + 4 test seeds (47..50) =
    20 seeds total, matching the Phase-49 symmetric-transfer
    driver's train/test split.  n_test = 4 × 4 × 5 = 80 per
    domain.
    """
    t0 = time.time()
    all_seeds = tuple(sorted(set(train_seeds) | set(test_seeds)))

    instances_by_domain: dict[str, list[dict]] = {}
    base_rates: dict[str, float] = {}
    for spec in ALL_DOMAINS:
        print(f"[phase50-zs] collecting {spec.name}…")
        insts = collect_domain_instances(
            spec, seeds=all_seeds,
            distractor_grid=distractor_grid,
            spurious_prob=spurious_prob,
            mislabel_prob=mislabel_prob)
        print(f"  {spec.name}: {len(insts)} instances; "
              f"alphabet {spec.label_alphabet}")
        counts: dict[str, int] = {}
        for inst in insts:
            g = inst["gold_label"]
            counts[g] = counts.get(g, 0) + 1
        base = (max(counts.values()) / len(insts)
                 if insts else 0.0)
        base_rates[spec.name] = base
        instances_by_domain[spec.name] = insts

    splits: dict[str, tuple[list, list]] = {}
    for spec in ALL_DOMAINS:
        tr, te = split_by_seed(
            instances_by_domain[spec.name],
            train_seeds, test_seeds)
        splits[spec.name] = (tr, te)
        print(f"[phase50-zs] {spec.name}: "
              f"train={len(tr)} test={len(te)}")

    # Per-domain training.
    per_domain: dict[str, dict[str, Any]] = {}
    for spec in ALL_DOMAINS:
        tr, _te = splits[spec.name]
        pairs = [(i["offered_capsules"], i["gold_label"])
                  for i in tr]
        print(f"[phase50-zs] training 5 decoder families on "
              f"{spec.name}…")
        v1 = train_learned_bundle_decoder(
            pairs, rc_alphabet=spec.label_alphabet,
            claim_to_root_cause=spec.claim_to_label,
            priority_order=spec.priority_order,
            n_epochs=300, lr=0.5, l2=1e-3, seed=train_seeds[0])
        v1.name = f"v1_{spec.name}"

        v2 = train_learned_bundle_decoder_v2(
            pairs, rc_alphabet=spec.label_alphabet,
            claim_to_root_cause=spec.claim_to_label,
            priority_order=spec.priority_order,
            n_epochs=n_epochs, lr=0.5, l2=1e-3,
            seed=train_seeds[0])
        v2.name = f"v2_{spec.name}"

        stable = train_sign_stable_v2(
            pairs, rc_alphabet=spec.label_alphabet,
            claim_to_root_cause=spec.claim_to_label,
            priority_order=spec.priority_order,
            n_epochs=n_epochs, lr=0.5, l2=1e-3,
            seed=train_seeds[0])
        stable.name = f"stable_{spec.name}"

        std = train_standardised_bundle_decoder_v2(
            pairs, rc_alphabet=spec.label_alphabet,
            claim_to_root_cause=spec.claim_to_label,
            priority_order=spec.priority_order,
            n_epochs=n_epochs, lr=0.5, l2=1e-3,
            seed=train_seeds[0])
        std.name = f"std_{spec.name}"

        ds = train_deep_set_bundle_decoder(
            pairs, rc_alphabet=spec.label_alphabet,
            claim_to_root_cause=spec.claim_to_label,
            priority_order=spec.priority_order,
            hidden_size=10, n_epochs=n_epochs, lr=0.1,
            l2=1e-3, seed=train_seeds[0])
        ds.name = f"deepset_{spec.name}"

        stable_ds = train_sign_stable_deepset(
            pairs, rc_alphabet=spec.label_alphabet,
            claim_to_root_cause=spec.claim_to_label,
            priority_order=spec.priority_order,
            hidden_size=8, n_epochs=n_epochs, lr=0.1,
            l2=1e-3, seed=train_seeds[0])
        stable_ds.name = f"stable_deepset_{spec.name}"

        per_domain[spec.name] = {
            "v1": v1, "v2": v2, "stable": stable, "std": std,
            "deepset": ds, "stable_deepset": stable_ds,
        }

    # Transfer matrix.
    families = ("v1", "v2", "stable", "std", "deepset",
                 "stable_deepset")
    make_cross = {
        "v1": make_cross_v1,
        "v2": make_cross_v2,
        "stable": make_cross_stable,
        "std": make_cross_standardised,
        "deepset": make_cross_deepset,
        "stable_deepset": make_cross_stable_deepset,
    }
    cells: list[dict[str, Any]] = []
    for train_spec in ALL_DOMAINS:
        for test_spec in ALL_DOMAINS:
            _, te = splits[test_spec.name]
            within = (train_spec.name == test_spec.name)
            cell: dict[str, Any] = {
                "train_domain": train_spec.name,
                "test_domain": test_spec.name,
                "within_domain": within,
                "base_rate": base_rates[test_spec.name],
                "n_test_instances": len(te),
            }
            # Priority baseline on test.
            pri = PriorityDecoder(
                priority_order=test_spec.priority_order,
                claim_to_root_cause=dict(test_spec.claim_to_label))
            cell["priority_accuracy"] = eval_decoder(
                te, pri).accuracy
            for fam in families:
                src = per_domain[train_spec.name][fam]
                if within:
                    dec = src
                else:
                    dec = make_cross[fam](src, test_spec)
                r = eval_decoder(te, dec)
                cell[f"{fam}_accuracy"] = r.accuracy
            cells.append(cell)

    # Symmetry / transfer summary.
    def _cell(train, test, family):
        for c in cells:
            if (c["train_domain"] == train
                and c["test_domain"] == test):
                return c[f"{family}_accuracy"]
        return None

    summary = {}
    for fam in families:
        within_inc = _cell("incident", "incident", fam)
        within_sec = _cell("security", "security", fam)
        cross_i2s = _cell("incident", "security", fam)
        cross_s2i = _cell("security", "incident", fam)
        symmetry_gap = abs(cross_i2s - cross_s2i)
        transfer_penalty_i2s = within_sec - cross_i2s
        transfer_penalty_s2i = within_inc - cross_s2i
        max_penalty = max(transfer_penalty_i2s,
                           transfer_penalty_s2i)
        gate2_met = (transfer_penalty_i2s <= 0.05
                      and transfer_penalty_s2i <= 0.05)
        summary[fam] = {
            "within_incident": within_inc,
            "within_security": within_sec,
            "incident_to_security": cross_i2s,
            "security_to_incident": cross_s2i,
            "symmetry_gap": symmetry_gap,
            "transfer_penalty_i2s": transfer_penalty_i2s,
            "transfer_penalty_s2i": transfer_penalty_s2i,
            "max_transfer_penalty": max_penalty,
            "gate2_strict_met": bool(gate2_met),
        }

    # Feature-level sign-agreement for the V2 + stable-V2
    # decoders (diagnoses the sign-flip premise of Theorem W3-21).
    sign_comparison: list[dict[str, Any]] = []
    for f in BUNDLE_DECODER_FEATURES_V2:
        row: dict[str, Any] = {"feature": f,
                                "in_stable_subfamily": (
                                    f in SIGN_STABLE_FEATURES_V2)}
        for spec in ALL_DOMAINS:
            row[spec.name] = round(
                per_domain[spec.name]["v2"].weights.get(f, 0.0),
                4)
        ws = [row[spec.name] for spec in ALL_DOMAINS]
        row["sign_agreement"] = int(
            all(w >= 0 for w in ws) or all(w <= 0 for w in ws))
        sign_comparison.append(row)

    v2_rate = (sum(r["sign_agreement"] for r in sign_comparison)
                / max(1, len(sign_comparison)))
    stable_rows = [r for r in sign_comparison
                    if r["in_stable_subfamily"]]
    stable_rate = (sum(r["sign_agreement"] for r in stable_rows)
                    / max(1, len(stable_rows)))

    out = {
        "schema": "coordpy.phase50.zero_shot.v1",
        "train_seeds": list(train_seeds),
        "test_seeds": list(test_seeds),
        "distractor_grid": list(distractor_grid),
        "spurious_prob": spurious_prob,
        "mislabel_prob": mislabel_prob,
        "domains": [spec.name for spec in ALL_DOMAINS],
        "label_alphabets": {
            spec.name: list(spec.label_alphabet)
            for spec in ALL_DOMAINS},
        "base_rates": {k: round(v, 4)
                         for k, v in base_rates.items()},
        "sign_stable_features": list(SIGN_STABLE_FEATURES_V2),
        "transfer_cells": cells,
        "summary": summary,
        "v2_sign_agreement_rate": v2_rate,
        "stable_sign_agreement_rate": stable_rate,
        "sign_comparison": sign_comparison,
        "per_domain_v2_weights": {
            spec.name:
                dict(per_domain[spec.name]["v2"].weights)
            for spec in ALL_DOMAINS
        },
        "per_domain_stable_weights": {
            spec.name:
                dict(per_domain[spec.name]["stable"].weights)
            for spec in ALL_DOMAINS
        },
        "wall_seconds": round(time.time() - t0, 3),
    }
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir,
                         "results_phase50_zero_shot.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"[phase50-zs] wrote {path} "
          f"({out['wall_seconds']} s)")
    _print_summary(out)
    return out


def _print_summary(out):
    domains = out["domains"]
    families = ("v1", "v2", "stable", "std", "deepset",
                 "stable_deepset")
    print(f"\n[phase50-zs] === TRANSFER MATRICES ===")
    for fam in families:
        print(f"\n  --- {fam} ---")
        print(f"  {'train\\test':<20s}", end="")
        for d in domains:
            print(f"  {d:>14s}", end="")
        print()
        for td in domains:
            print(f"  {td:<20s}", end="")
            for ed in domains:
                cell = next((c for c in out["transfer_cells"]
                              if c["train_domain"] == td
                              and c["test_domain"] == ed), None)
                print(f"  {cell[f'{fam}_accuracy']:>14.3f}",
                       end="")
            print()

    print(f"\n[phase50-zs] === STRICT ZERO-SHOT GATE-2 SUMMARY ===")
    print(f"  {'family':<22s}  {'wi':>7s}  {'ws':>7s}  "
          f"{'i→s':>7s}  {'s→i':>7s}  {'gap':>7s}  "
          f"{'pen i→s':>9s}  {'pen s→i':>9s}  "
          f"{'max pen':>9s}  {'Gate2':>7s}")
    for fam in families:
        s = out["summary"][fam]
        met = "MET" if s["gate2_strict_met"] else "not"
        print(f"  {fam:<22s}  {s['within_incident']:>7.3f}  "
              f"{s['within_security']:>7.3f}  "
              f"{s['incident_to_security']:>7.3f}  "
              f"{s['security_to_incident']:>7.3f}  "
              f"{s['symmetry_gap']:>7.3f}  "
              f"{s['transfer_penalty_i2s']:>+9.3f}  "
              f"{s['transfer_penalty_s2i']:>+9.3f}  "
              f"{s['max_transfer_penalty']:>+9.3f}  "
              f"{met:>7s}")

    print(f"\n[phase50-zs] === FEATURE SIGN AGREEMENT ===")
    print(f"  V2 all-features sign-agreement rate   : "
          f"{out['v2_sign_agreement_rate']:.3f}")
    print(f"  Stable-subfamily sign-agreement rate  : "
          f"{out['stable_sign_agreement_rate']:.3f}")
    for r in out["sign_comparison"]:
        agree = "agree   " if r["sign_agreement"] else "DISAGREE"
        stable = ("[stable]" if r["in_stable_subfamily"]
                   else "[      ]")
        row = f"  {stable} {r['feature']:<32s}"
        for d in domains:
            row += f"  {d}={r[d]:+.3f}"
        row += f"  [{agree}]"
        print(row)

    any_met = any(out["summary"][fam]["gate2_strict_met"]
                   for fam in families)
    print(f"\n[phase50-zs] STRICT ZERO-SHOT GATE-2: "
          f"{'MET (at least one family)' if any_met else 'NOT MET (all families fail)'}")


def _cli() -> None:
    p = argparse.ArgumentParser(
        description="Phase 50: strict zero-shot Gate-2 study")
    p.add_argument("--out-dir", default=".")
    p.add_argument("--train-seeds", nargs="+", type=int,
                    default=list(range(31, 47)))
    p.add_argument("--test-seeds", nargs="+", type=int,
                    default=list(range(47, 51)))
    p.add_argument("--distractor-grid", nargs="+", type=int,
                    default=[6, 20, 60, 120])
    p.add_argument("--spurious-prob", type=float, default=0.30)
    p.add_argument("--mislabel-prob", type=float, default=0.10)
    p.add_argument("--n-epochs", type=int, default=500)
    args = p.parse_args()
    run_phase50_zero_shot(
        out_dir=args.out_dir,
        train_seeds=tuple(args.train_seeds),
        test_seeds=tuple(args.test_seeds),
        distractor_grid=tuple(args.distractor_grid),
        spurious_prob=args.spurious_prob,
        mislabel_prob=args.mislabel_prob,
        n_epochs=args.n_epochs)


if __name__ == "__main__":
    _cli()
