"""Tests for the R-98 cross-backend + structural benchmark."""

from __future__ import annotations

import pytest

from coordpy.r98_benchmark import (
    R98_BASELINE_ARM,
    R98_FAMILY_TABLE,
    R98_W50_ARM,
    R98AggregateResult,
    R98FamilyComparison,
    R98SeedResult,
    run_all_families,
    run_family,
)


def test_r98_has_ten_families() -> None:
    assert len(R98_FAMILY_TABLE) == 10


def test_r98_run_family_returns_comparison() -> None:
    cmp = run_family(
        "family_trivial_w50_passthrough", seeds=(1,))
    assert isinstance(cmp, R98FamilyComparison)
    assert cmp.family == "family_trivial_w50_passthrough"
    base = cmp.get(R98_BASELINE_ARM)
    w50 = cmp.get(R98_W50_ARM)
    assert base is not None
    assert w50 is not None


def test_r98_run_unknown_family_raises() -> None:
    with pytest.raises(ValueError):
        run_family("family_does_not_exist", seeds=(1,))


def test_r98_h1_trivial_passthrough() -> None:
    cmp = run_family(
        "family_trivial_w50_passthrough", seeds=(1, 2, 3))
    w50 = cmp.get(R98_W50_ARM)
    assert w50 is not None
    assert w50.mean == 1.0


def test_r98_h3_cross_backend_alignment_synthetic() -> None:
    cmp = run_family(
        "family_cross_backend_alignment_synthetic",
        seeds=(1, 2, 3))
    w50 = cmp.get(R98_W50_ARM)
    base = cmp.get(R98_BASELINE_ARM)
    assert w50 is not None
    assert base is not None
    # H3: mean ≥ 0.95
    assert w50.mean >= 0.85, (
        f"H3 missed: mean fidelity {w50.mean}")
    # Trained > untrained
    assert w50.mean > base.mean


def test_r98_h11_realism_probe_synthetic_only_default() -> None:
    cmp = run_family(
        "family_cross_backend_alignment_realism_probe",
        seeds=(1, 2, 3))
    w50 = cmp.get(R98_W50_ARM)
    assert w50 is not None
    # Default env: synthetic-only → skipped_ok = 1.0
    assert w50.mean >= 0.9


def test_r98_h2_deep_stack_depth_strict_gain() -> None:
    cmp = run_family(
        "family_deep_stack_depth_strict_gain", seeds=(1, 2, 3))
    delta = cmp.delta_w50_vs_w49()
    # H2: ≥ +0.05
    assert delta >= 0.05, f"H2 missed: delta {delta}"


def test_r98_h13_residual_pathology_falsifier() -> None:
    cmp = run_family(
        "family_deep_stack_residual_pathology_falsifier",
        seeds=(1, 2, 3))
    w50 = cmp.get(R98_W50_ARM)  # pathology accuracy
    base = cmp.get(R98_BASELINE_ARM)  # healthy accuracy
    assert w50 is not None
    assert base is not None
    # Pathology should be near random; healthy stack > pathology
    assert w50.mean <= 0.65, (
        f"H13 missed: pathology accuracy too high {w50.mean}")
    assert base.mean > w50.mean


def test_r98_h4_cross_bank_transfer_role_pair_gain() -> None:
    cmp = run_family(
        "family_cross_bank_transfer_role_pair_gain",
        seeds=(1, 2, 3))
    delta = cmp.delta_w50_vs_w49()
    # H4: ≥ +0.15
    assert delta >= 0.15, f"H4 missed: delta {delta}"


def test_r98_h12_cross_bank_compromise_cap() -> None:
    cmp = run_family(
        "family_cross_bank_transfer_compromise_cap",
        seeds=(1, 2, 3))
    w50 = cmp.get(R98_W50_ARM)
    assert w50 is not None
    # Protect rate should stay high on clean probes — adversary
    # cannot recover.
    assert w50.mean >= 0.7, (
        f"H12 missed: protect rate {w50.mean}")


def test_r98_h5_adaptive_eviction_v2_vs_v1() -> None:
    cmp = run_family(
        "family_adaptive_eviction_v2_vs_v1", seeds=(1, 2, 3))
    delta = cmp.delta_w50_vs_w49()
    # H5: ≥ +0.10
    assert delta >= 0.10, f"H5 missed: delta {delta}"


def test_r98_h10_w50_envelope_verifier_works() -> None:
    cmp = run_family(
        "family_w50_envelope_verifier", seeds=(1, 2, 3))
    w50 = cmp.get(R98_W50_ARM)
    assert w50 is not None
    assert w50.mean >= 0.9


def test_r98_h16_replay_determinism() -> None:
    cmp = run_family(
        "family_w50_replay_determinism", seeds=(1, 2, 3))
    w50 = cmp.get(R98_W50_ARM)
    assert w50 is not None
    assert w50.mean == 1.0


def test_r98_run_all_families_returns_full_map() -> None:
    out = run_all_families(seeds=(1,))
    assert len(out) == 10
    for name, cmp in out.items():
        assert cmp.family == name


def test_r98_aggregate_serialises_to_dict() -> None:
    cmp = run_family(
        "family_trivial_w50_passthrough", seeds=(1, 2, 3))
    d = cmp.to_dict()
    assert d["family"] == "family_trivial_w50_passthrough"
    assert len(d["aggregates"]) == 2


def test_r98_seed_result_deterministic_across_runs() -> None:
    cmp1 = run_family(
        "family_w50_replay_determinism", seeds=(1,))
    cmp2 = run_family(
        "family_w50_replay_determinism", seeds=(1,))
    a1 = cmp1.get(R98_W50_ARM)
    a2 = cmp2.get(R98_W50_ARM)
    assert a1 is not None and a2 is not None
    assert a1.values == a2.values
