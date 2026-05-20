"""Tests for ``coordpy.differential_privacy_v1``."""

from __future__ import annotations

import json

import pytest

from coordpy.differential_privacy_v1 import (
    DPBenchReportV1,
    DPBudgetBreachEventV1,
    DPBudgetSpecV1,
    DPBudgetTrackerV1,
    DPCapsuleV1,
    DPMechanism,
    DPMechanismParamsV1,
    PII_PATTERNS_V1,
    apply_dp_mechanism_v1,
    build_dp_capsule_v1,
    measure_utility_vs_privacy_curve_v1,
    redact_pii_v1,
    run_dp_composed_pipeline_v1,
    run_dp_v1_bench,
)


def test_laplace_scale_is_sensitivity_over_epsilon():
    p = DPMechanismParamsV1(
        mechanism=DPMechanism.LAPLACE,
        sensitivity=1.0, epsilon=0.5)
    assert abs(p.noise_scale() - 2.0) < 1e-9


def test_gaussian_scale_uses_delta_correctly():
    p = DPMechanismParamsV1(
        mechanism=DPMechanism.GAUSSIAN,
        sensitivity=1.0, epsilon=1.0, delta=1e-5)
    import math
    expected = math.sqrt(2.0 * math.log(1.25 / 1e-5)) / 1.0
    assert abs(p.noise_scale() - expected) < 1e-6


def test_laplace_rejects_nonzero_delta():
    p = DPMechanismParamsV1(
        mechanism=DPMechanism.LAPLACE,
        sensitivity=1.0, epsilon=0.5, delta=0.1)
    with pytest.raises(ValueError):
        p.noise_scale()


def test_gaussian_rejects_zero_delta():
    p = DPMechanismParamsV1(
        mechanism=DPMechanism.GAUSSIAN,
        sensitivity=1.0, epsilon=1.0, delta=0.0)
    with pytest.raises(ValueError):
        p.noise_scale()


def test_dp_capsule_does_not_store_raw_value():
    p = DPMechanismParamsV1(
        mechanism=DPMechanism.LAPLACE,
        sensitivity=1.0, epsilon=1.0)
    cap = build_dp_capsule_v1(
        value=42.0, params=p, noise_seed_bytes=b"\x00" * 32)
    d = cap.to_dict()
    # 42 (rounded perturbed_value) might coincidentally equal,
    # but the cleartext field must not exist.
    assert "raw_value" not in d
    assert "original_value" not in d
    assert "cleartext_value" not in d


def test_dp_capsule_cid_changes_with_noise_seed():
    p = DPMechanismParamsV1(
        mechanism=DPMechanism.LAPLACE,
        sensitivity=1.0, epsilon=0.1)
    cap1 = build_dp_capsule_v1(
        value=10.0, params=p, noise_seed_bytes=b"\x00" * 32)
    cap2 = build_dp_capsule_v1(
        value=10.0, params=p, noise_seed_bytes=b"\x01" * 32)
    # Different seeds → different noise → different CIDs.
    assert cap1.cid() != cap2.cid()


def test_dp_capsule_deterministic_under_same_seed():
    p = DPMechanismParamsV1(
        mechanism=DPMechanism.LAPLACE,
        sensitivity=1.0, epsilon=1.0)
    seed = b"\x42" * 32
    cap1 = build_dp_capsule_v1(
        value=10.0, params=p, noise_seed_bytes=seed)
    cap2 = build_dp_capsule_v1(
        value=10.0, params=p, noise_seed_bytes=seed)
    assert cap1.cid() == cap2.cid()
    assert (
        abs(cap1.perturbed_value - cap2.perturbed_value) < 1e-12)


def test_budget_tracker_refuses_overflow():
    t = DPBudgetTrackerV1(
        spec=DPBudgetSpecV1(total_epsilon=1.0))
    assert t.request_spend(0.5) is True
    assert t.request_spend(0.4) is True
    assert t.request_spend(0.3) is False  # 1.2 > 1.0
    assert len(t.refused_calls) == 1
    assert (
        t.refused_calls[0].refusal_reason == "epsilon_overflow")


def test_budget_tracker_refuses_delta_overflow():
    t = DPBudgetTrackerV1(
        spec=DPBudgetSpecV1(
            total_epsilon=10.0, total_delta=1e-5))
    assert t.request_spend(epsilon=0.1, delta=5e-6) is True
    assert t.request_spend(epsilon=0.1, delta=2e-5) is False
    assert (
        t.refused_calls[0].refusal_reason == "delta_overflow")


def test_pii_redactor_redacts_five_patterns():
    text = (
        "Email alice@x.com SSN 123-45-6789 Phone "
        "555-123-4567 IP 10.0.0.1 Card "
        "4111-1111-1111-1111")
    redacted, events = redact_pii_v1(text)
    redacted_kinds = {e.pattern_name for e in events}
    # At least 5 patterns hit.
    assert len(redacted_kinds) >= 5, f"got {redacted_kinds}"
    # Original PII not in redacted.
    assert "alice@x.com" not in redacted
    assert "123-45-6789" not in redacted
    assert "555-123-4567" not in redacted
    assert "10.0.0.1" not in redacted
    assert "4111-1111-1111-1111" not in redacted


def test_pii_redaction_event_does_not_carry_originals():
    text = "secret email alice@example.com"
    _, events = redact_pii_v1(text)
    # Inspect event dicts for the original substring.
    for ev in events:
        d_str = json.dumps(ev.to_dict())
        assert "alice@example.com" not in d_str


def test_utility_curve_monotonic_ε_increasing():
    curve = measure_utility_vs_privacy_curve_v1(
        epsilons=(0.1, 0.5, 1.0, 2.0, 5.0), n_samples=1000)
    # Mean error should strictly decrease as ε increases.
    for a, b in zip(curve, curve[1:]):
        assert a.mean_abs_error > b.mean_abs_error


def test_composed_pipeline_emits_both_dp_and_integrity_cids():
    tracker = DPBudgetTrackerV1(
        spec=DPBudgetSpecV1(total_epsilon=1.0))
    p = DPMechanismParamsV1(
        mechanism=DPMechanism.LAPLACE,
        sensitivity=1.0, epsilon=0.5)
    out = run_dp_composed_pipeline_v1(
        true_value=10.0, tracker=tracker, params=p,
        noise_seed_bytes=b"\x07" * 32)
    assert out.refused_due_to_budget is False
    assert len(out.dp_capsule_cid) == 64
    assert len(out.integrity_anchor_cid) == 64
    # Both must differ (different content domains).
    assert out.dp_capsule_cid != out.integrity_anchor_cid


def test_composed_pipeline_refuses_when_budget_exhausted():
    tracker = DPBudgetTrackerV1(
        spec=DPBudgetSpecV1(total_epsilon=0.5))
    p = DPMechanismParamsV1(
        mechanism=DPMechanism.LAPLACE,
        sensitivity=1.0, epsilon=1.0)
    out = run_dp_composed_pipeline_v1(
        true_value=10.0, tracker=tracker, params=p)
    assert out.refused_due_to_budget is True
    assert len(tracker.refused_calls) == 1


def test_full_bench_passes_all_dod_bars():
    rep = run_dp_v1_bench()
    assert rep.pii_redaction_pattern_count >= 5
    assert rep.pii_redactions_made >= 5
    assert rep.pii_not_in_output is True
    assert rep.dp_committed_value_within_3_sigma is True
    assert rep.budget_breach_refused is True
    assert rep.utility_curve_is_monotonic is True
    assert rep.raw_value_not_in_capsule_dict is True
    assert len(rep.utility_curve_points) >= 5


def test_bench_report_cid_deterministic():
    r1 = run_dp_v1_bench(seed=86_039)
    r2 = run_dp_v1_bench(seed=86_039)
    assert r1.report_cid == r2.report_cid


def test_dp_capsule_cid_does_not_carry_raw_value_in_string():
    """A small CID-string sanity: the capsule's CID is a 64-char
    hex; it should not 'leak' the value as a substring. (This is
    a probabilistic check; a true cryptographic guarantee
    requires SHA-256 preimage resistance.)
    """
    p = DPMechanismParamsV1(
        mechanism=DPMechanism.LAPLACE,
        sensitivity=1.0, epsilon=0.1)
    cap = build_dp_capsule_v1(
        value=12345.0, params=p,
        noise_seed_bytes=b"\x55" * 32)
    assert "12345" not in cap.cid()


def test_budget_breach_event_content_addressed():
    ev = DPBudgetBreachEventV1(
        budget_spec_cid="abc", epsilon_requested=0.5,
        delta_requested=0.0, epsilon_remaining=0.0,
        delta_remaining=0.0, refusal_reason="x", label="y")
    assert len(ev.cid()) == 64
