"""R-113 + R-114 + R-115 W56 benchmark soundness tests.

Confirms each family returns a dict and core H-bars pass on a
single seed. Heavier full-bench runs (3 seeds) are kept out of
the test suite for wall-clock reasons; see
``coordpy.r113_benchmark.run_all_families`` for the full sweep.
"""

from __future__ import annotations

from coordpy.r113_benchmark import (
    R113_FAMILIES,
    run_seed as r113_run_seed,
)
from coordpy.r114_benchmark import (
    R114_FAMILIES,
    run_seed as r114_run_seed,
)
from coordpy.r115_benchmark import (
    R115_FAMILIES,
    run_seed as r115_run_seed,
)


def test_r113_run_seed_returns_dict() -> None:
    r = r113_run_seed(11)
    assert len(r.family_results) == len(R113_FAMILIES)


def test_r114_run_seed_returns_dict() -> None:
    r = r114_run_seed(11)
    assert len(r.family_results) == len(R114_FAMILIES)


def test_r115_run_seed_returns_dict() -> None:
    r = r115_run_seed(11)
    assert len(r.family_results) == len(R115_FAMILIES)


def test_r113_h1_trivial_passthrough() -> None:
    r = r113_run_seed(11)
    assert (
        r.family_results["trivial_w56_passthrough"][
            "passthrough_ok"]
        is True)


def test_r113_h2_substrate_determinism() -> None:
    r = r113_run_seed(11)
    assert (
        r.family_results[
            "tiny_substrate_forward_determinism"][
                "determinism_ok"]
        is True)


def test_r113_h4_causal_mask() -> None:
    r = r113_run_seed(11)
    assert (
        r.family_results[
            "tiny_substrate_causal_mask_soundness"][
                "causal_mask_ok"]
        is True)


def test_r113_h6_kv_bridge_perturbs_logits() -> None:
    r = r113_run_seed(11)
    assert (
        r.family_results[
            "kv_bridge_injection_perturbs_logits"][
                "perturbation_above_threshold"]
        is True)


def test_r114_h13_substrate_hybrid_kv_read() -> None:
    r = r114_run_seed(11)
    assert (
        r.family_results[
            "deep_substrate_hybrid_kv_read"][
                "kv_read_load_bearing"]
        is True)


def test_r114_h20_ecc_v8_19_bits() -> None:
    r = r114_run_seed(11)
    assert (
        r.family_results["ecc_v8_compression_19_bits"][
            "target_met"]
        is True)


def test_r115_h28_substrate_tiebreaker() -> None:
    r = r115_run_seed(11)
    assert (
        r.family_results[
            "consensus_v2_substrate_tiebreaker_recall"][
                "substrate_stage_picked"]
        is True)


def test_r115_h35_w56_integration_envelope_modes() -> None:
    r = r115_run_seed(11)
    assert (
        r.family_results["w56_integration_envelope"][
            "ge_30_modes"]
        is True)
