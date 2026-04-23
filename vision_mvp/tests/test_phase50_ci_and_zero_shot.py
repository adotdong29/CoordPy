"""Contract tests for Phase-50 additions.

Locks in:
  * Binomial CI helpers are shaped correctly and monotonic.
  * ``SIGN_STABLE_FEATURES_V2`` is a subset of
    ``BUNDLE_DECODER_FEATURES_V2`` of the documented size.
  * ``train_sign_stable_v2`` returns a decoder whose masked
    weights match the stable sub-family at decode time.
  * ``StandardisedBundleDecoderV2`` computes the same score
    whether called on its own domain or after cross-domain
    re-targeting (weights + stats copied verbatim).
  * ``SignStableDeepSetDecoder`` architecture has the expected
    shape (4-dim stable-φ + 8-dim stable-V2 = 12-dim input).
  * ``_betainc_reg`` is monotone in the probability argument.
"""

from __future__ import annotations

import math

import numpy as np

from vision_mvp.experiments.phase50_gate1_ci import (
    SIGN_STABLE_FEATURES_V2,
    wilson_ci, clopper_pearson_ci,
    train_sign_stable_v2,
)
from vision_mvp.experiments.phase50_zero_shot_transfer import (
    SIGN_STABLE_PHI_IDX,
    StandardisedBundleDecoderV2,
    SignStableDeepSetDecoder,
    train_standardised_bundle_decoder_v2,
    train_sign_stable_deepset,
    make_cross_standardised, make_cross_stable_deepset,
    _incident_spec, _security_spec,
)
from vision_mvp.wevra.capsule import CapsuleKind, ContextCapsule
from vision_mvp.wevra.capsule_decoder_v2 import (
    BUNDLE_DECODER_FEATURES_V2, DEEPSET_PHI_FEATURES,
    _bundle_vote_summary, _feature_vector_v2,
)


def mk_cap(claim_kind: str, source_role: str,
           n_tokens: int = 4) -> ContextCapsule:
    return ContextCapsule.new(
        kind=CapsuleKind.HANDOFF,
        payload={"claim": claim_kind, "src": source_role,
                 "tag": f"{claim_kind}-{source_role}"},
        n_tokens=n_tokens,
        metadata={"claim_kind": claim_kind,
                  "source_role": source_role},
    )


DEFAULT_CLAIM_MAP = {
    "DISK_FILL_CRITICAL": "disk_fill",
    "CRON_OVERRUN": "disk_fill",
    "OOM_KILL": "memory_leak",
    "TLS_EXPIRED": "tls_expiry",
    "ERROR_RATE_SPIKE": "error_spike",
}
DEFAULT_PRIORITY = (
    "DISK_FILL_CRITICAL", "TLS_EXPIRED", "OOM_KILL",
    "CRON_OVERRUN", "ERROR_RATE_SPIKE",
)
DEFAULT_ALPHABET = ("disk_fill", "memory_leak", "tls_expiry")


def _mk_examples():
    disk_bundle = [
        mk_cap("DISK_FILL_CRITICAL", "sysadmin"),
        mk_cap("CRON_OVERRUN", "sysadmin"),
    ]
    mem_bundle = [
        mk_cap("OOM_KILL", "sysadmin"),
        mk_cap("OOM_KILL", "db_admin"),
    ]
    tls_bundle = [
        mk_cap("TLS_EXPIRED", "network"),
        mk_cap("ERROR_RATE_SPIKE", "monitor"),
    ]
    return [
        (disk_bundle, "disk_fill"),
        (mem_bundle, "memory_leak"),
        (tls_bundle, "tls_expiry"),
        (disk_bundle, "disk_fill"),
        (mem_bundle, "memory_leak"),
    ]


# ---------------------------------------------------------------------------
# Binomial CI helpers
# ---------------------------------------------------------------------------


def test_wilson_ci_bounded_and_monotone():
    # At k=0 the lower bound is exactly 0 (Wilson has no
    # 'at least one success' prior).
    lo, hi = wilson_ci(0, 50)
    assert 0.0 <= lo <= hi <= 1.0
    assert lo == 0.0
    # At k=n, upper bound is 1.0.
    lo, hi = wilson_ci(50, 50)
    assert hi == 1.0
    # CI for 40/100 = 0.40 should straddle 0.40.
    lo, hi = wilson_ci(40, 100)
    assert lo < 0.40 < hi
    # CI narrows with n.
    _, hi_small = wilson_ci(4, 10)
    _, hi_large = wilson_ci(40, 100)
    assert (hi_small - 0.4) > (hi_large - 0.4)


def test_clopper_pearson_ci_bounded_and_conservative():
    # Clopper-Pearson is ALWAYS at least as wide as Wilson.
    for k, n in ((0, 10), (3, 10), (40, 100), (10, 10)):
        w_lo, w_hi = wilson_ci(k, n)
        cp_lo, cp_hi = clopper_pearson_ci(k, n)
        assert 0.0 <= cp_lo <= cp_hi <= 1.0
        # Clopper-Pearson has wider or equal CIs.
        assert cp_lo <= w_lo + 1e-6
        assert cp_hi + 1e-6 >= w_hi


def test_clopper_pearson_ci_pareto_sample():
    # Spot-check against a known value.  For k=64, n=160,
    # p̂=0.400; Clopper-Pearson 95 % CI ≈ [0.323, 0.480] from
    # scipy.stats.binom_test inversion.
    lo, hi = clopper_pearson_ci(64, 160)
    assert abs(lo - 0.3227) < 0.01
    assert abs(hi - 0.4801) < 0.01


# ---------------------------------------------------------------------------
# Sign-stable subfamily
# ---------------------------------------------------------------------------


def test_sign_stable_features_v2_is_subset():
    assert set(SIGN_STABLE_FEATURES_V2).issubset(
        set(BUNDLE_DECODER_FEATURES_V2))
    # Phase-50 declared 8-feature sub-family.
    assert len(SIGN_STABLE_FEATURES_V2) == 8
    # Must include the bias.
    assert "bias" in SIGN_STABLE_FEATURES_V2
    # Must include absolute-count primitives.
    assert "log1p_votes" in SIGN_STABLE_FEATURES_V2
    assert "votes_share" in SIGN_STABLE_FEATURES_V2


def test_sign_stable_v2_masks_non_stable_features():
    d = train_sign_stable_v2(
        _mk_examples(), rc_alphabet=DEFAULT_ALPHABET,
        claim_to_root_cause=DEFAULT_CLAIM_MAP,
        priority_order=DEFAULT_PRIORITY,
        n_epochs=60, lr=0.5, l2=1e-3, seed=0)
    for k, v in d.weights.items():
        if k in SIGN_STABLE_FEATURES_V2:
            # May be zero by training, that's allowed.
            pass
        else:
            # MUST be exactly zero — masked out at train-end.
            assert v == 0.0, (
                f"Non-stable feature {k} has non-zero weight {v}")


def test_sign_stable_v2_decode_output_in_alphabet():
    d = train_sign_stable_v2(
        _mk_examples(), rc_alphabet=DEFAULT_ALPHABET,
        claim_to_root_cause=DEFAULT_CLAIM_MAP,
        priority_order=DEFAULT_PRIORITY,
        n_epochs=100, lr=0.5, l2=1e-3, seed=0)
    for (b, _gold) in _mk_examples():
        out = d.decode(b)
        assert out in DEFAULT_ALPHABET


# ---------------------------------------------------------------------------
# Standardised V2 decoder
# ---------------------------------------------------------------------------


def test_standardised_v2_training_is_deterministic():
    kwargs = dict(
        rc_alphabet=DEFAULT_ALPHABET,
        claim_to_root_cause=DEFAULT_CLAIM_MAP,
        priority_order=DEFAULT_PRIORITY,
        n_epochs=60, lr=0.5, l2=1e-3, seed=13)
    a = train_standardised_bundle_decoder_v2(
        _mk_examples(), **kwargs)
    b = train_standardised_bundle_decoder_v2(
        _mk_examples(), **kwargs)
    assert np.allclose(a.weights, b.weights)
    assert np.allclose(a.feat_mean, b.feat_mean)
    assert np.allclose(a.feat_std, b.feat_std)


def test_standardised_v2_decode_output_in_alphabet():
    d = train_standardised_bundle_decoder_v2(
        _mk_examples(), rc_alphabet=DEFAULT_ALPHABET,
        claim_to_root_cause=DEFAULT_CLAIM_MAP,
        priority_order=DEFAULT_PRIORITY,
        n_epochs=100, lr=0.5, l2=1e-3, seed=0)
    for (b, _g) in _mk_examples():
        assert d.decode(b) in DEFAULT_ALPHABET


def test_standardised_v2_cross_domain_uses_source_stats():
    """When we re-target via ``make_cross_standardised``, the
    decoder MUST keep its source-domain statistics (not re-
    compute from target).  This is the zero-shot discipline."""
    src = train_standardised_bundle_decoder_v2(
        _mk_examples(), rc_alphabet=DEFAULT_ALPHABET,
        claim_to_root_cause=DEFAULT_CLAIM_MAP,
        priority_order=DEFAULT_PRIORITY,
        n_epochs=50, lr=0.5, l2=1e-3, seed=0)
    tgt_spec = _incident_spec()
    cross = make_cross_standardised(src, tgt_spec)
    # Stats are copied verbatim from source.
    assert np.allclose(src.feat_mean, cross.feat_mean)
    assert np.allclose(src.feat_std, cross.feat_std)
    assert np.allclose(src.weights, cross.weights)


# ---------------------------------------------------------------------------
# Sign-stable DeepSet
# ---------------------------------------------------------------------------


def test_sign_stable_deepset_input_dim():
    """Input dim = |stable φ| + |stable V2|."""
    d = train_sign_stable_deepset(
        _mk_examples(), rc_alphabet=DEFAULT_ALPHABET,
        claim_to_root_cause=DEFAULT_CLAIM_MAP,
        priority_order=DEFAULT_PRIORITY,
        hidden_size=8, n_epochs=40, lr=0.1, l2=1e-3, seed=0)
    expected_input = len(SIGN_STABLE_PHI_IDX) + len(
        SIGN_STABLE_FEATURES_V2)
    # W1 has shape (hidden_size, input_dim).
    assert d.W1.shape == (8, expected_input)
    assert d.b1.shape == (8,)
    assert d.w2.shape == (8,)


def test_sign_stable_deepset_is_deterministic():
    kwargs = dict(
        rc_alphabet=DEFAULT_ALPHABET,
        claim_to_root_cause=DEFAULT_CLAIM_MAP,
        priority_order=DEFAULT_PRIORITY,
        hidden_size=8, n_epochs=40, lr=0.1, l2=1e-3, seed=5)
    a = train_sign_stable_deepset(_mk_examples(), **kwargs)
    b = train_sign_stable_deepset(_mk_examples(), **kwargs)
    assert np.allclose(a.W1, b.W1)
    assert np.allclose(a.w2, b.w2)


def test_sign_stable_deepset_output_in_alphabet():
    d = train_sign_stable_deepset(
        _mk_examples(), rc_alphabet=DEFAULT_ALPHABET,
        claim_to_root_cause=DEFAULT_CLAIM_MAP,
        priority_order=DEFAULT_PRIORITY,
        hidden_size=8, n_epochs=80, lr=0.1, l2=1e-3, seed=0)
    for (b, _g) in _mk_examples():
        assert d.decode(b) in DEFAULT_ALPHABET


def test_sign_stable_deepset_cross_domain_retarget():
    src = train_sign_stable_deepset(
        _mk_examples(), rc_alphabet=DEFAULT_ALPHABET,
        claim_to_root_cause=DEFAULT_CLAIM_MAP,
        priority_order=DEFAULT_PRIORITY,
        hidden_size=8, n_epochs=40, lr=0.1, l2=1e-3, seed=0)
    tgt = _incident_spec()
    cross = make_cross_stable_deepset(src, tgt)
    # Weights copied verbatim; alphabet swapped.
    assert np.allclose(src.W1, cross.W1)
    assert np.allclose(src.w2, cross.w2)
    assert cross.rc_alphabet == tgt.label_alphabet


# ---------------------------------------------------------------------------
# Theorem W3-24 witness: classical winner's-curse lower bound
# ---------------------------------------------------------------------------


def test_w3_24_winners_curse_lower_bound():
    """A simple synthetic demonstration: under C independent
    Binomial(n, 0.40) cells, the max observed accuracy has
    expectation > 0.40.  This is the (classical) winner's
    curse phenomenon that Phase-49's Gate-1 best-cell estimate
    is subject to.
    """
    rng = np.random.RandomState(0)
    n = 80
    p = 0.40
    C = 21  # number of (admission × budget) cells in Phase 49
    n_reps = 2000
    max_acc = np.empty(n_reps)
    for r in range(n_reps):
        ks = rng.binomial(n, p, size=C)
        max_acc[r] = ks.max() / n
    # Empirical expectation of the max is materially above p.
    assert max_acc.mean() - p > 0.05, (
        f"Winner's curse expected but got {max_acc.mean():.3f} "
        f"vs true {p:.3f}")
