"""Contract tests for the Phase-48 bundle-aware decoder family.

Locks in:
  * Determinism of each decoder's output on a fixed bundle.
  * The Priority decoder's structural ceiling: it outputs the
    implied rc of the first high-priority kind present, regardless
    of bundle shape.
  * The Plurality decoder returns the majority-implied rc on a
    coherent-majority bundle; on a one-vote-each tie it falls
    back to the priority-order tiebreak.
  * SourceCorroboratedPriorityDecoder vetoes lone high-priority
    claims (the Phase-31 poisoning signature).
  * LearnedBundleDecoder training is deterministic in seed;
    its weight vector is exactly ``BUNDLE_DECODER_FEATURES``
    keys; its decode returns a rc from ``rc_alphabet``.
  * Theorem W3-18 empirical witness: on a constructed bundle
    where the causal chain has ≥ 2 coherent claims implying
    ``rc_true`` plus one lone spurious high-priority claim
    implying ``rc_spurious``, plurality returns ``rc_true``
    while priority returns ``rc_spurious``.  This is the
    sharpest single-bundle separation between the two
    decoders.
"""

from __future__ import annotations

from vision_mvp.wevra.capsule import (
    CapsuleKind, ContextCapsule,
)
from vision_mvp.wevra.capsule_decoder import (
    BUNDLE_DECODER_FEATURES, LearnedBundleDecoder,
    PluralityDecoder, PriorityDecoder,
    SourceCorroboratedPriorityDecoder, UNKNOWN,
    evaluate_decoder, train_learned_bundle_decoder,
)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def mk_cap(claim_kind: str, source_role: str,
           n_tokens: int = 4) -> ContextCapsule:
    """Minimal HANDOFF capsule carrying the claim_kind + source_role
    metadata that every decoder reads.  Payload is deliberately
    small so admission never rejects for size."""
    return ContextCapsule.new(
        kind=CapsuleKind.HANDOFF,
        payload={"claim": claim_kind, "src": source_role},
        n_tokens=n_tokens,
        metadata={"claim_kind": claim_kind,
                   "source_role": source_role},
    )


def disk_fill_bundle(spurious: bool = False) -> list[ContextCapsule]:
    """A clean disk_fill_cron bundle (two coherent claims from
    sysadmin).  Optional lone spurious extra claim — not needed
    here because disk_fill IS the noise target."""
    caps = [
        mk_cap("CRON_OVERRUN", "sysadmin"),
        mk_cap("DISK_FILL_CRITICAL", "sysadmin"),
        mk_cap("SLOW_QUERY_OBSERVED", "db_admin"),
        mk_cap("POOL_EXHAUSTION", "db_admin"),
        mk_cap("ERROR_RATE_SPIKE", "monitor"),
        mk_cap("LATENCY_SPIKE", "monitor"),
    ]
    if spurious:
        caps.append(mk_cap("TLS_EXPIRED", "network"))
    return caps


def memory_leak_poisoned_bundle() -> list[ContextCapsule]:
    """Causal memory_leak cascade + one lone spurious
    DISK_FILL_CRITICAL from an irrelevant source — the Phase-31
    "priority decoder ceiling" signature.

    Priority decoder outputs ``disk_fill`` (wrong, gold is
    ``memory_leak``).
    Plurality sees one-vote-each on {memory_leak, error_spike,
    latency_spike, disk_fill}; priority tiebreak steps through
    and picks DFC first → also ``disk_fill`` on a pure plurality
    rule.  This is why Plurality alone does NOT universally
    break the ceiling.
    """
    return [
        mk_cap("OOM_KILL", "sysadmin"),
        mk_cap("ERROR_RATE_SPIKE", "monitor"),
        mk_cap("LATENCY_SPIKE", "monitor"),
        mk_cap("DISK_FILL_CRITICAL", "network"),  # lone spurious
    ]


def memory_leak_coherent_bundle() -> list[ContextCapsule]:
    """A memory_leak bundle with ≥ 2 coherent claims implying
    ``memory_leak`` (two OOM_KILL events from two sources) plus
    a single lone spurious DISK_FILL_CRITICAL.

    Constructed so Theorem W3-18's sufficient condition holds:
    the true rc has strictly more implied-rc votes than any
    spurious rc.  Plurality returns ``memory_leak``; Priority
    still returns ``disk_fill``.
    """
    return [
        mk_cap("OOM_KILL", "sysadmin"),
        mk_cap("OOM_KILL", "db_admin"),
        mk_cap("ERROR_RATE_SPIKE", "monitor"),
        mk_cap("LATENCY_SPIKE", "monitor"),
        mk_cap("DISK_FILL_CRITICAL", "network"),  # lone spurious
    ]


# ----------------------------------------------------------------------------
# Determinism + basic behaviour
# ----------------------------------------------------------------------------


def test_priority_decoder_is_deterministic():
    dec = PriorityDecoder()
    b = disk_fill_bundle()
    outputs = {dec.decode(b) for _ in range(5)}
    assert outputs == {"disk_fill"}


def test_priority_decoder_first_match():
    """The priority decoder outputs the implied-rc of the first
    kind in priority_order that is present.  DFC is rank 0, so
    any DFC-present bundle outputs disk_fill."""
    dec = PriorityDecoder()
    assert dec.decode(disk_fill_bundle()) == "disk_fill"
    assert dec.decode(memory_leak_poisoned_bundle()) == "disk_fill"
    assert dec.decode([]) == UNKNOWN


def test_priority_decoder_no_known_kind():
    """No claim_kind in the bundle → unknown."""
    dec = PriorityDecoder()
    # claim_kind=None (e.g. missing) should be silently ignored.
    c = ContextCapsule.new(
        kind=CapsuleKind.HANDLE, payload={},
        n_tokens=1,
        metadata={"handle_cid": "x"})
    # HANDLE capsule has no claim_kind metadata, so decoder skips.
    assert dec.decode([c]) == UNKNOWN


def test_plurality_decoder_majority():
    """Plurality returns the argmax-votes implied-rc when a
    strict majority exists (Theorem W3-18 sufficient condition)."""
    dec = PluralityDecoder()
    b = memory_leak_coherent_bundle()
    # Votes: memory_leak=2, error_spike=1, latency_spike=1,
    # disk_fill=1 — memory_leak wins.
    assert dec.decode(b) == "memory_leak"


def test_plurality_decoder_tiebreak_falls_back_to_priority():
    """One-vote-each ties break by priority_order walk, which
    re-produces the priority-decoder failure mode: DFC still
    wins under a lone spurious top-priority injection."""
    dec = PluralityDecoder()
    b = memory_leak_poisoned_bundle()
    # All rcs at 1 vote; priority_order walks DFC first → disk_fill.
    assert dec.decode(b) == "disk_fill"


def test_plurality_empty_bundle_is_unknown():
    dec = PluralityDecoder()
    assert dec.decode([]) == UNKNOWN


def test_src_corroborated_priority_vetoes_lone_high_priority():
    """SourceCorroborated with min_sources=2 vetoes any
    high-priority kind emitted by only one source.  On a lone
    spurious DISK_FILL_CRITICAL, the decoder does NOT return
    disk_fill; in the Phase-31 bundle it falls back to UNKNOWN
    because every top-priority kind in this bundle is
    emitted by only one source."""
    dec = SourceCorroboratedPriorityDecoder(min_sources=2)
    b = memory_leak_poisoned_bundle()
    # DFC from 1 source → vetoed.  OOM_KILL (priority rank 3)
    # from 1 source → vetoed.  DEADLOCK_SUSPECTED not present.
    # CRON_OVERRUN not present.  POOL_EXHAUSTION not present.
    # etc.  All priority_order kinds fail; returns UNKNOWN.
    assert dec.decode(b) == UNKNOWN


def test_src_corroborated_priority_admits_on_multi_source():
    """If a high-priority kind is supported by ≥ 2 sources, the
    decoder admits it.  Constructed: two DISK_FILL_CRITICAL from
    distinct sources → disk_fill."""
    dec = SourceCorroboratedPriorityDecoder(min_sources=2)
    b = [mk_cap("DISK_FILL_CRITICAL", "sysadmin"),
          mk_cap("DISK_FILL_CRITICAL", "monitor"),
          mk_cap("OOM_KILL", "sysadmin")]
    assert dec.decode(b) == "disk_fill"


# ----------------------------------------------------------------------------
# Theorem W3-18 — sharp single-bundle separation
# ----------------------------------------------------------------------------


def test_w3_18_plurality_strictly_dominates_priority_on_coherent_majority():
    """Sharpest single-bundle separator: in the coherent-majority
    regime (gold rc has strictly more implied-rc votes than any
    other rc), Plurality returns gold while Priority is poisoned
    by a lone top-priority spurious claim."""
    gold = "memory_leak"
    b = memory_leak_coherent_bundle()
    pri = PriorityDecoder()
    plu = PluralityDecoder()
    # The sharp separator — priority is poisoned, plurality is not.
    assert pri.decode(b) != gold
    assert plu.decode(b) == gold


# ----------------------------------------------------------------------------
# LearnedBundleDecoder — feature vocab, determinism, training shape
# ----------------------------------------------------------------------------


def test_learned_bundle_decoder_feature_vocab_matches_constants():
    """Any LearnedBundleDecoder's weights are exactly
    BUNDLE_DECODER_FEATURES — the feature vocabulary is closed."""
    decoder = train_learned_bundle_decoder(
        [(disk_fill_bundle(), "disk_fill"),
         (memory_leak_coherent_bundle(), "memory_leak")],
        rc_alphabet=("disk_fill", "memory_leak"),
        claim_to_root_cause={
            "DISK_FILL_CRITICAL": "disk_fill",
            "CRON_OVERRUN": "disk_fill",
            "OOM_KILL": "memory_leak",
            "SLOW_QUERY_OBSERVED": "slow_query_cascade",
            "POOL_EXHAUSTION": "pool_exhaustion",
            "ERROR_RATE_SPIKE": "error_spike",
            "LATENCY_SPIKE": "latency_spike",
        },
        n_epochs=50, lr=0.5, l2=1e-3, seed=7,
    )
    assert set(decoder.weights.keys()) == set(BUNDLE_DECODER_FEATURES)


def test_learned_bundle_decoder_training_is_deterministic():
    examples = [
        (disk_fill_bundle(), "disk_fill"),
        (memory_leak_coherent_bundle(), "memory_leak"),
        (disk_fill_bundle(spurious=True), "disk_fill"),
    ]
    kwargs = dict(
        rc_alphabet=("disk_fill", "memory_leak", "tls_expiry"),
        claim_to_root_cause={
            "DISK_FILL_CRITICAL": "disk_fill",
            "CRON_OVERRUN": "disk_fill",
            "OOM_KILL": "memory_leak",
            "TLS_EXPIRED": "tls_expiry",
            "SLOW_QUERY_OBSERVED": "slow_query_cascade",
            "POOL_EXHAUSTION": "pool_exhaustion",
            "ERROR_RATE_SPIKE": "error_spike",
            "LATENCY_SPIKE": "latency_spike",
        },
        n_epochs=80, lr=0.5, l2=1e-3, seed=13,
    )
    d1 = train_learned_bundle_decoder(examples, **kwargs)
    d2 = train_learned_bundle_decoder(examples, **kwargs)
    for k in BUNDLE_DECODER_FEATURES:
        assert abs(d1.weights[k] - d2.weights[k]) < 1e-12


def test_learned_bundle_decoder_output_in_alphabet():
    """The decoder's output is always in ``rc_alphabet``."""
    alpha = ("disk_fill", "memory_leak", "tls_expiry")
    decoder = train_learned_bundle_decoder(
        [(disk_fill_bundle(), "disk_fill"),
         (memory_leak_coherent_bundle(), "memory_leak")],
        rc_alphabet=alpha,
        claim_to_root_cause={
            "DISK_FILL_CRITICAL": "disk_fill",
            "CRON_OVERRUN": "disk_fill",
            "OOM_KILL": "memory_leak",
            "TLS_EXPIRED": "tls_expiry",
        },
        n_epochs=40, lr=0.5, l2=1e-3, seed=0,
    )
    for bundle in (disk_fill_bundle(),
                    memory_leak_poisoned_bundle(),
                    memory_leak_coherent_bundle(),
                    []):
        out = decoder.decode(bundle)
        assert out in alpha


def test_learned_bundle_decoder_fits_held_out_training_pattern():
    """A small, clean training set of two-class examples should
    push the decoder to recover each example's gold rc on the
    training set itself.  This is a trivial overfitting check —
    the decoder has enough features to memorise a 2-class
    linearly-separable shape."""
    ex = [
        (disk_fill_bundle(), "disk_fill"),
        (memory_leak_coherent_bundle(), "memory_leak"),
        (disk_fill_bundle(), "disk_fill"),
        (memory_leak_coherent_bundle(), "memory_leak"),
    ]
    alpha = ("disk_fill", "memory_leak")
    decoder = train_learned_bundle_decoder(
        ex, rc_alphabet=alpha,
        claim_to_root_cause={
            "DISK_FILL_CRITICAL": "disk_fill",
            "CRON_OVERRUN": "disk_fill",
            "OOM_KILL": "memory_leak",
            "SLOW_QUERY_OBSERVED": "slow_query_cascade",
            "POOL_EXHAUSTION": "pool_exhaustion",
            "ERROR_RATE_SPIKE": "error_spike",
            "LATENCY_SPIKE": "latency_spike",
        },
        n_epochs=400, lr=0.5, l2=1e-4, seed=0,
    )
    # Both examples should be decoded correctly.
    assert decoder.decode(disk_fill_bundle()) == "disk_fill"
    assert (decoder.decode(memory_leak_coherent_bundle())
             == "memory_leak")


def test_evaluate_decoder_aggregates_accuracy():
    ex = [
        (disk_fill_bundle(), "disk_fill"),
        (memory_leak_poisoned_bundle(), "memory_leak"),
        (memory_leak_coherent_bundle(), "memory_leak"),
    ]
    scen = ["disk_fill_cron",
             "memory_leak_oom", "memory_leak_oom"]
    pri = PriorityDecoder()
    result = evaluate_decoder(ex, pri, scenario_keys=scen)
    # Priority: disk_fill (right), disk_fill (wrong),
    # disk_fill (wrong) → 1/3.
    assert abs(result.accuracy - 1 / 3) < 1e-12
    assert result.n_instances == 3
    assert result.per_scenario_accuracy["disk_fill_cron"] == 1.0
    assert (result.per_scenario_accuracy["memory_leak_oom"]
             == 0.0)
