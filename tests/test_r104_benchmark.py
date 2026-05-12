"""R-104 benchmark family-level smoke tests."""

from __future__ import annotations

import pytest

from coordpy.r104_benchmark import (
    R104_BASELINE_ARM,
    R104_FAMILY_TABLE,
    R104_W53_ARM,
    run_family,
)


def test_r104_family_registry_has_12_families() -> None:
    assert len(R104_FAMILY_TABLE) == 12


def test_r104_trivial_passthrough() -> None:
    c = run_family(
        "family_trivial_w53_passthrough", seeds=(1, 2))
    assert c.get(R104_W53_ARM).mean == 1.0


def test_r104_w53_envelope_verifier() -> None:
    c = run_family(
        "family_w53_envelope_verifier", seeds=(1,))
    assert c.get(R104_W53_ARM).mean == 1.0


def test_r104_w53_replay_determinism() -> None:
    c = run_family(
        "family_w53_replay_determinism", seeds=(1,))
    assert c.get(R104_W53_ARM).mean == 1.0


def test_r104_mlsc_consensus_quorum() -> None:
    c = run_family(
        "family_mlsc_consensus_quorum", seeds=(1, 2))
    assert c.get(R104_W53_ARM).mean == 1.0


def test_r104_mlsc_audit_trail_integrity() -> None:
    c = run_family(
        "family_mlsc_audit_trail_integrity", seeds=(1, 2))
    assert c.get(R104_W53_ARM).mean == 1.0


def test_r104_uncertainty_layer_calibration() -> None:
    c = run_family(
        "family_uncertainty_layer_calibration",
        seeds=(1, 2, 3))
    assert c.get(R104_W53_ARM).mean > 0.5


def test_r104_quint_realism_probe_skip_ok() -> None:
    c = run_family(
        "family_quint_realism_probe", seeds=(1,))
    # synthetic_only path should report skipped_ok=1.0.
    assert c.get(R104_W53_ARM).mean == 1.0


def test_r104_deep_v4_corruption_aware() -> None:
    c = run_family(
        "family_deep_stack_v4_corruption_aware", seeds=(1,))
    assert c.get(R104_W53_ARM).mean == 1.0
