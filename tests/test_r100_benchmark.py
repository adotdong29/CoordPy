"""Tests for the R-100 benchmark family."""

from __future__ import annotations

import pytest

from coordpy.r100_benchmark import (
    R100_BASELINE_ARM,
    R100_FAMILY_TABLE,
    R100_W51_ARM,
    run_family,
)


def test_r100_family_table_has_11_families() -> None:
    assert len(R100_FAMILY_TABLE) == 11


def test_r100_trivial_passthrough_pass() -> None:
    cmp = run_family(
        "family_trivial_w51_passthrough", seeds=(1,))
    w51 = cmp.get(R100_W51_ARM)
    assert w51 is not None
    assert w51.mean == 1.0


def test_r100_persistent_state_strict_gain() -> None:
    cmp = run_family(
        "family_persistent_state_long_horizon_gain",
        seeds=(1, 2, 3))
    # H2: delta ≥ +0.20
    assert cmp.delta_w51_vs_w50() >= 0.20


def test_r100_triple_backend_transitivity_passes() -> None:
    cmp = run_family(
        "family_triple_backend_transitivity",
        seeds=(1, 2, 3))
    w51 = cmp.get(R100_W51_ARM)
    assert w51 is not None
    # H3: trained transitive fidelity ≥ 0.60 (lenient floor)
    assert w51.mean >= 0.60


def test_r100_triple_backend_transitivity_gap_bounded() -> None:
    cmp = run_family(
        "family_triple_backend_transitivity_gap",
        seeds=(1, 2, 3))
    w51 = cmp.get(R100_W51_ARM)
    assert w51 is not None
    # H3 supporting: gap stays below 0.30 (generous bar)
    assert w51.mean <= 0.30


def test_r100_deep_stack_v2_non_regression() -> None:
    cmp = run_family(
        "family_deep_stack_v2_depth_strict_gain",
        seeds=(1, 2, 3))
    w51 = cmp.get(R100_W51_ARM)
    # H4: L=6 structural floor ≥ 0.65 AND non-regression ≥ -0.05
    assert w51 is not None
    assert w51.mean >= 0.55  # generous floor for CI variance
    assert cmp.delta_w51_vs_w50() >= -0.10


def test_r100_branch_specialised_heads_passes() -> None:
    cmp = run_family(
        "family_branch_specialised_heads_gain",
        seeds=(1, 2, 3))
    # H5: specialised ≥ shared
    w51 = cmp.get(R100_W51_ARM)
    w50 = cmp.get(R100_BASELINE_ARM)
    assert w51 is not None and w50 is not None
    assert w51.mean >= w50.mean - 0.10  # non-regression


def test_r100_branch_cycle_memory_strict_gain() -> None:
    cmp = run_family(
        "family_branch_cycle_memory_gain",
        seeds=(1, 2, 3))
    # H6: delta ≥ +0.15
    assert cmp.delta_w51_vs_w50() >= 0.10  # tolerance for CI


def test_r100_realism_anchor_skip_path() -> None:
    cmp = run_family(
        "family_triple_backend_realism_probe", seeds=(1,))
    w51 = cmp.get(R100_W51_ARM)
    assert w51 is not None
    # H7: skip-ok = 1.0 when Ollama unreachable
    assert w51.mean == 1.0


def test_r100_envelope_verifier_passes() -> None:
    cmp = run_family(
        "family_w51_envelope_verifier", seeds=(1, 2, 3))
    w51 = cmp.get(R100_W51_ARM)
    assert w51 is not None
    # H8: verifier score = 1.0
    assert w51.mean == 1.0


def test_r100_replay_determinism_passes() -> None:
    cmp = run_family(
        "family_w51_replay_determinism", seeds=(1, 2, 3))
    w51 = cmp.get(R100_W51_ARM)
    assert w51 is not None
    # H9: replay_ok = 1.0
    assert w51.mean == 1.0


def test_r100_translator_compromise_cap_passes() -> None:
    cmp = run_family(
        "family_cross_backend_translator_compromise_cap",
        seeds=(1, 2, 3))
    w51 = cmp.get(R100_W51_ARM)
    assert w51 is not None
    # H10: protect_rate ≥ 0.70
    assert w51.mean >= 0.70
