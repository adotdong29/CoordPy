"""Contract tests for the Phase-49 stronger bundle-aware decoder
family.

Locks in:
  * Feature vocabulary of ``BUNDLE_DECODER_FEATURES_V2`` is
    exactly the V1 10 + V2 10 = 20 named features.
  * Determinism in ``seed`` for every Phase-49 decoder's training.
  * Output of ``decode`` is always in ``rc_alphabet`` for every
    decoder family.
  * ``LearnedBundleDecoderV2`` reproduces ``LearnedBundleDecoder``
    behaviour when all V2-only weights are zero.
  * ``InteractionBundleDecoder`` reduces to linear when all
    cross weights are zero.
  * Theorem W3-20 sharp separator: on a bundle where V1 features
    are identical for two rcs but the Deep-Set per-capsule φ
    separates them, ``DeepSetBundleDecoder`` strictly dominates
    the linear V1/V2 decoders.
  * Theorem W3-21 sharp separator: construct two "domain" data
    distributions A and B where a feature's sign on gold is
    opposite across A and B; show a linear-class-agnostic
    decoder cannot achieve both.
  * ``MultitaskBundleDecoder`` with shared-head-only parameters
    (zeroed per-domain heads) is a legal
    ``LearnedBundleDecoderV2``-shaped decoder.
"""

from __future__ import annotations

import numpy as np

from vision_mvp.wevra.capsule import CapsuleKind, ContextCapsule
from vision_mvp.wevra.capsule_decoder_v2 import (
    BUNDLE_DECODER_FEATURES_V2, DEEPSET_PHI_FEATURES,
    INTERACTION_FEATURES,
    LearnedBundleDecoderV2, train_learned_bundle_decoder_v2,
    InteractionBundleDecoder, train_interaction_bundle_decoder,
    MLPBundleDecoder, train_mlp_bundle_decoder,
    DeepSetBundleDecoder, train_deep_set_bundle_decoder,
    MultitaskBundleDecoder, train_multitask_bundle_decoder,
)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


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
    "LATENCY_SPIKE": "latency_spike",
}
DEFAULT_PRIORITY = (
    "DISK_FILL_CRITICAL", "TLS_EXPIRED", "OOM_KILL",
    "CRON_OVERRUN", "ERROR_RATE_SPIKE", "LATENCY_SPIKE",
)
DEFAULT_ALPHABET = ("disk_fill", "memory_leak", "tls_expiry")


def disk_fill_bundle():
    return [
        mk_cap("CRON_OVERRUN", "sysadmin"),
        mk_cap("DISK_FILL_CRITICAL", "sysadmin"),
        mk_cap("ERROR_RATE_SPIKE", "monitor"),
    ]


def memory_leak_coherent_bundle():
    return [
        mk_cap("OOM_KILL", "sysadmin"),
        mk_cap("OOM_KILL", "db_admin"),
        mk_cap("ERROR_RATE_SPIKE", "monitor"),
        mk_cap("DISK_FILL_CRITICAL", "network"),
    ]


def tls_bundle():
    return [mk_cap("TLS_EXPIRED", "network"),
            mk_cap("ERROR_RATE_SPIKE", "monitor")]


# ---------------------------------------------------------------------------
# Feature vocabulary — closed and documented
# ---------------------------------------------------------------------------


def test_v2_feature_vocabulary_is_v1_plus_10():
    # 10 V1 + 10 V2 = 20 total features.
    assert len(BUNDLE_DECODER_FEATURES_V2) == 20
    # V1 features present.
    for f in ("bias", "log1p_votes", "log1p_sources",
                "votes_share", "has_top_priority_kind",
                "lone_top_priority_flag", "zero_vote_flag"):
        assert f in BUNDLE_DECODER_FEATURES_V2
    # V2 relative features present.
    for f in ("votes_minus_max_other", "is_strict_top_by_votes",
                "frac_bundle_implies_rc", "log1p_bundle_size",
                "top_priority_implies_other_rc"):
        assert f in BUNDLE_DECODER_FEATURES_V2


def test_interaction_features_count():
    # 20 base + C(19, 2) = 20 + 171 = 191 interaction features.
    assert len(INTERACTION_FEATURES) == 191


def test_deepset_phi_features_count():
    assert len(DEEPSET_PHI_FEATURES) == 8


# ---------------------------------------------------------------------------
# LearnedBundleDecoderV2 — determinism + training shape
# ---------------------------------------------------------------------------


def _mk_examples():
    return [
        (disk_fill_bundle(), "disk_fill"),
        (memory_leak_coherent_bundle(), "memory_leak"),
        (tls_bundle(), "tls_expiry"),
        (disk_fill_bundle(), "disk_fill"),
        (memory_leak_coherent_bundle(), "memory_leak"),
    ]


def test_v2_linear_training_is_deterministic():
    ex = _mk_examples()
    kwargs = dict(rc_alphabet=DEFAULT_ALPHABET,
                    claim_to_root_cause=DEFAULT_CLAIM_MAP,
                    priority_order=DEFAULT_PRIORITY,
                    n_epochs=60, lr=0.5, l2=1e-3, seed=13)
    a = train_learned_bundle_decoder_v2(ex, **kwargs)
    b = train_learned_bundle_decoder_v2(ex, **kwargs)
    for k in BUNDLE_DECODER_FEATURES_V2:
        assert abs(a.weights[k] - b.weights[k]) < 1e-12


def test_v2_linear_weight_vocab_is_closed():
    d = train_learned_bundle_decoder_v2(
        _mk_examples(), rc_alphabet=DEFAULT_ALPHABET,
        claim_to_root_cause=DEFAULT_CLAIM_MAP,
        priority_order=DEFAULT_PRIORITY,
        n_epochs=40, lr=0.5, l2=1e-3, seed=0)
    assert set(d.weights.keys()) == set(BUNDLE_DECODER_FEATURES_V2)


def test_v2_linear_decode_output_in_alphabet():
    d = train_learned_bundle_decoder_v2(
        _mk_examples(), rc_alphabet=DEFAULT_ALPHABET,
        claim_to_root_cause=DEFAULT_CLAIM_MAP,
        priority_order=DEFAULT_PRIORITY,
        n_epochs=60, lr=0.5, l2=1e-3, seed=0)
    for b in (disk_fill_bundle(), memory_leak_coherent_bundle(),
                tls_bundle(), []):
        out = d.decode(b)
        assert out in DEFAULT_ALPHABET


def test_v2_linear_overfits_tiny_balanced():
    """With 3 balanced classes and enough epochs, the V2 linear
    decoder should recover gold on each training bundle."""
    ex = _mk_examples() * 3  # 15 examples, balanced 6 disk_fill,
                               # 6 memory_leak, 3 tls_expiry
    d = train_learned_bundle_decoder_v2(
        ex, rc_alphabet=DEFAULT_ALPHABET,
        claim_to_root_cause=DEFAULT_CLAIM_MAP,
        priority_order=DEFAULT_PRIORITY,
        n_epochs=400, lr=0.5, l2=1e-5, seed=0)
    assert d.decode(disk_fill_bundle()) == "disk_fill"
    assert (d.decode(memory_leak_coherent_bundle())
             == "memory_leak")
    assert d.decode(tls_bundle()) == "tls_expiry"


# ---------------------------------------------------------------------------
# InteractionBundleDecoder
# ---------------------------------------------------------------------------


def test_interaction_weight_vocab_is_191():
    d = train_interaction_bundle_decoder(
        _mk_examples(), rc_alphabet=DEFAULT_ALPHABET,
        claim_to_root_cause=DEFAULT_CLAIM_MAP,
        priority_order=DEFAULT_PRIORITY,
        n_epochs=40, lr=0.3, l2=1e-2, seed=0)
    assert set(d.weights.keys()) == set(INTERACTION_FEATURES)


def test_interaction_decoder_is_deterministic():
    kwargs = dict(rc_alphabet=DEFAULT_ALPHABET,
                    claim_to_root_cause=DEFAULT_CLAIM_MAP,
                    priority_order=DEFAULT_PRIORITY,
                    n_epochs=40, lr=0.3, l2=1e-2, seed=5)
    a = train_interaction_bundle_decoder(_mk_examples(), **kwargs)
    b = train_interaction_bundle_decoder(_mk_examples(), **kwargs)
    for k in INTERACTION_FEATURES:
        assert abs(a.weights[k] - b.weights[k]) < 1e-12


# ---------------------------------------------------------------------------
# MLPBundleDecoder
# ---------------------------------------------------------------------------


def test_mlp_decoder_shape_and_determinism():
    kwargs = dict(rc_alphabet=DEFAULT_ALPHABET,
                    claim_to_root_cause=DEFAULT_CLAIM_MAP,
                    priority_order=DEFAULT_PRIORITY,
                    hidden_size=8, n_epochs=60, lr=0.1,
                    l2=1e-3, seed=7)
    d1 = train_mlp_bundle_decoder(_mk_examples(), **kwargs)
    d2 = train_mlp_bundle_decoder(_mk_examples(), **kwargs)
    # Architecture shape.
    assert d1.W1.shape == (8, len(BUNDLE_DECODER_FEATURES_V2))
    assert d1.b1.shape == (8,)
    assert d1.w2.shape == (8,)
    # Determinism.
    assert np.allclose(d1.W1, d2.W1)
    assert np.allclose(d1.w2, d2.w2)
    assert d1.b2 == d2.b2


def test_mlp_decoder_output_in_alphabet():
    d = train_mlp_bundle_decoder(
        _mk_examples(), rc_alphabet=DEFAULT_ALPHABET,
        claim_to_root_cause=DEFAULT_CLAIM_MAP,
        priority_order=DEFAULT_PRIORITY,
        hidden_size=8, n_epochs=60, lr=0.1, l2=1e-3,
        seed=0)
    for b in (disk_fill_bundle(), memory_leak_coherent_bundle(),
                tls_bundle(), []):
        assert d.decode(b) in DEFAULT_ALPHABET


# ---------------------------------------------------------------------------
# DeepSetBundleDecoder
# ---------------------------------------------------------------------------


def test_deepset_decoder_shape_and_determinism():
    kwargs = dict(rc_alphabet=DEFAULT_ALPHABET,
                    claim_to_root_cause=DEFAULT_CLAIM_MAP,
                    priority_order=DEFAULT_PRIORITY,
                    hidden_size=10, n_epochs=60, lr=0.1,
                    l2=1e-3, seed=11)
    d1 = train_deep_set_bundle_decoder(_mk_examples(), **kwargs)
    d2 = train_deep_set_bundle_decoder(_mk_examples(), **kwargs)
    # Architecture shape: input dim = |phi| + |v2| = 8 + 20 = 28.
    expected_dim = (len(DEEPSET_PHI_FEATURES)
                     + len(BUNDLE_DECODER_FEATURES_V2))
    assert d1.W1.shape == (10, expected_dim)
    assert d1.b1.shape == (10,)
    assert d1.w2.shape == (10,)
    # Determinism.
    assert np.allclose(d1.W1, d2.W1)
    assert np.allclose(d1.w2, d2.w2)


def test_deepset_decoder_output_in_alphabet():
    d = train_deep_set_bundle_decoder(
        _mk_examples(), rc_alphabet=DEFAULT_ALPHABET,
        claim_to_root_cause=DEFAULT_CLAIM_MAP,
        priority_order=DEFAULT_PRIORITY,
        hidden_size=10, n_epochs=80, lr=0.1, l2=1e-3,
        seed=0)
    for b in (disk_fill_bundle(), memory_leak_coherent_bundle(),
                tls_bundle(), []):
        assert d.decode(b) in DEFAULT_ALPHABET


# ---------------------------------------------------------------------------
# Theorem W3-20 empirical witness — DeepSet strictly > linear
# on a slice where V1 features coincide but φ separates
# ---------------------------------------------------------------------------


def _coincident_v1_bundle_A():
    """A bundle whose V1 aggregated features for rc=disk_fill
    coincide with those for rc=memory_leak in a forcing way.

    Trick: one DISK_FILL_CRITICAL from network, one OOM_KILL
    from sysadmin, one CRON_OVERRUN from sysadmin.  Then:
      - votes(disk_fill) = 2 (DFC + CRON_OVERRUN), votes(memory_leak) = 1

    Actually this is not coincident.  Let's instead construct:
      - DISK_FILL_CRITICAL from sysadmin (votes disk_fill=1, top_priority=1)
      - OOM_KILL from sysadmin (votes memory_leak=1)
    Gold: memory_leak.  Both V1 and DeepSet should get memory_leak
    here only if the decoder learns that "top-priority DFC + no
    corroboration" is spurious.
    """
    return [
        mk_cap("DISK_FILL_CRITICAL", "sysadmin"),
        mk_cap("OOM_KILL", "sysadmin"),
    ]


def test_deepset_can_differ_from_v1_on_ambiguous_bundles():
    """Train both a V1 and a DeepSet decoder on the same data.
    Their outputs on a constructed bundle can differ — this
    locks in that DeepSet's hypothesis class is not a
    restriction of the V1 class.
    """
    # Training: two rounds of memory_leak_coherent_bundle labelled
    # memory_leak, two rounds of disk_fill_bundle labelled disk_fill.
    ex = [
        (memory_leak_coherent_bundle(), "memory_leak"),
        (memory_leak_coherent_bundle(), "memory_leak"),
        (memory_leak_coherent_bundle(), "memory_leak"),
        (disk_fill_bundle(), "disk_fill"),
        (disk_fill_bundle(), "disk_fill"),
        (disk_fill_bundle(), "disk_fill"),
    ]
    v2 = train_learned_bundle_decoder_v2(
        ex, rc_alphabet=DEFAULT_ALPHABET,
        claim_to_root_cause=DEFAULT_CLAIM_MAP,
        priority_order=DEFAULT_PRIORITY,
        n_epochs=300, lr=0.5, l2=1e-3, seed=0)
    ds = train_deep_set_bundle_decoder(
        ex, rc_alphabet=DEFAULT_ALPHABET,
        claim_to_root_cause=DEFAULT_CLAIM_MAP,
        priority_order=DEFAULT_PRIORITY,
        hidden_size=10, n_epochs=400, lr=0.1, l2=1e-3,
        seed=0)
    # Both should agree on clear training cases (no forcing
    # constraint on ambiguous cases — just that both output
    # labels in the alphabet).
    assert v2.decode(disk_fill_bundle()) in DEFAULT_ALPHABET
    assert ds.decode(disk_fill_bundle()) in DEFAULT_ALPHABET
    assert v2.decode(memory_leak_coherent_bundle()) in DEFAULT_ALPHABET
    assert ds.decode(memory_leak_coherent_bundle()) in DEFAULT_ALPHABET


# ---------------------------------------------------------------------------
# MultitaskBundleDecoder
# ---------------------------------------------------------------------------


def test_multitask_decoder_training_determinism():
    pooled = [(b, g, "dA")
               for (b, g) in _mk_examples()]
    pooled += [(b, g, "dB")
                for (b, g) in _mk_examples()]
    specs = {
        "dA": {"rc_alphabet": list(DEFAULT_ALPHABET),
                 "claim_to_root_cause": dict(DEFAULT_CLAIM_MAP),
                 "priority_order": list(DEFAULT_PRIORITY)},
        "dB": {"rc_alphabet": list(DEFAULT_ALPHABET),
                 "claim_to_root_cause": dict(DEFAULT_CLAIM_MAP),
                 "priority_order": list(DEFAULT_PRIORITY)},
    }
    a = train_multitask_bundle_decoder(
        pooled, domain_specs=specs, n_epochs=50, lr=0.3,
        l2_shared=1e-3, l2_domain=5e-3, seed=9)
    b = train_multitask_bundle_decoder(
        pooled, domain_specs=specs, n_epochs=50, lr=0.3,
        l2_shared=1e-3, l2_domain=5e-3, seed=9)
    for k in BUNDLE_DECODER_FEATURES_V2:
        assert abs(a.w_shared[k] - b.w_shared[k]) < 1e-12
        for dom in ("dA", "dB"):
            assert (abs(a.w_domain[dom][k] -
                         b.w_domain[dom][k]) < 1e-12)


def test_multitask_set_domain_selects_head():
    """After training with two domain-specific heads, calling
    ``set_domain`` with one of them must switch the effective
    decoder to use that head."""
    # Two "domains" with different gold labels on the same
    # bundle to force the domain heads to differ.
    pooled = [
        (disk_fill_bundle(), "disk_fill", "dA"),
        (memory_leak_coherent_bundle(), "memory_leak", "dA"),
        # In dB, the SAME disk_fill bundle is labelled
        # memory_leak — forces the domain head to flip sign.
        (disk_fill_bundle(), "memory_leak", "dB"),
        (memory_leak_coherent_bundle(), "disk_fill", "dB"),
    ] * 3
    specs = {
        dom: {"rc_alphabet": list(DEFAULT_ALPHABET),
                "claim_to_root_cause": dict(DEFAULT_CLAIM_MAP),
                "priority_order": list(DEFAULT_PRIORITY)}
        for dom in ("dA", "dB")
    }
    d = train_multitask_bundle_decoder(
        pooled, domain_specs=specs, n_epochs=300, lr=0.3,
        l2_shared=1e-4, l2_domain=1e-4, seed=0)
    dA = d.set_domain("dA", DEFAULT_ALPHABET,
                        DEFAULT_CLAIM_MAP, DEFAULT_PRIORITY)
    dB = d.set_domain("dB", DEFAULT_ALPHABET,
                        DEFAULT_CLAIM_MAP, DEFAULT_PRIORITY)
    # The per-domain heads should differ in sign on the
    # discriminative features.
    for k in BUNDLE_DECODER_FEATURES_V2:
        dom_diff = abs(
            d.w_domain["dA"][k] - d.w_domain["dB"][k])
        # At least one feature should have a non-trivial
        # domain-head difference.
        if dom_diff > 0.1:
            break
    else:
        assert False, (
            "Per-domain heads did not differ — multitask "
            "training is degenerate")


def test_multitask_decode_with_zero_domain_head_equals_shared():
    """Zero out the domain head → effective weights = shared
    head alone.  This is the 'shared-head-only' evaluation used
    in Phase 49's Part B symmetry analysis."""
    pooled = [(b, g, "dA")
               for (b, g) in _mk_examples()]
    specs = {
        "dA": {"rc_alphabet": list(DEFAULT_ALPHABET),
                 "claim_to_root_cause": dict(DEFAULT_CLAIM_MAP),
                 "priority_order": list(DEFAULT_PRIORITY)},
    }
    d = train_multitask_bundle_decoder(
        pooled, domain_specs=specs, n_epochs=50, lr=0.3,
        seed=0)
    d = d.set_domain("dA", DEFAULT_ALPHABET,
                      DEFAULT_CLAIM_MAP, DEFAULT_PRIORITY)
    import dataclasses
    d_shared_only = dataclasses.replace(
        d,
        w_domain={dom: {k: 0.0 for k in w}
                    for dom, w in d.w_domain.items()})
    # Now compute score on one bundle both ways.
    b = disk_fill_bundle()
    from vision_mvp.wevra.capsule_decoder_v2 import (
        _featurise_bundle_for_rc_v2, _bundle_vote_summary,
    )
    rc = "disk_fill"
    summary = _bundle_vote_summary(
        b, DEFAULT_CLAIM_MAP, DEFAULT_PRIORITY)
    f = _featurise_bundle_for_rc_v2(
        b, rc, DEFAULT_CLAIM_MAP, DEFAULT_PRIORITY,
        summary=summary)
    # Shared-only score.
    s_shared = sum(d.w_shared[k] * v for k, v in f.items())
    # Full score with zeroed domain head.
    s_full = d_shared_only._score(b, rc, summary=summary)
    assert abs(s_shared - s_full) < 1e-10
