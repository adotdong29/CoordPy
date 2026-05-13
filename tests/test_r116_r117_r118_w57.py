"""W57 R-116 / R-117 / R-118 family-level pass tests.

These tests verify that every cell family in R-116, R-117, R-118
passes 3/3 seeds. They are the W57 success-criterion gate.
"""

from __future__ import annotations

from coordpy.r116_benchmark import R116_FAMILIES, run_r116
from coordpy.r117_benchmark import R117_FAMILIES, run_r117
from coordpy.r118_benchmark import R118_FAMILIES, run_r118


def test_r116_all_families_pass_3_seeds() -> None:
    r = run_r116(seeds=(0, 1, 2))
    assert r["all_passed"] is True, (
        f"R-116 failed: {r['pass_counts']}")
    for name, _ in R116_FAMILIES:
        assert r["pass_counts"].get(name, 0) == 3, (
            f"R-116 {name} did not pass 3/3 seeds")


def test_r117_all_families_pass_3_seeds() -> None:
    r = run_r117(seeds=(0, 1, 2))
    assert r["all_passed"] is True, (
        f"R-117 failed: {r['pass_counts']}")
    for name, _ in R117_FAMILIES:
        assert r["pass_counts"].get(name, 0) == 3, (
            f"R-117 {name} did not pass 3/3 seeds")


def test_r118_all_families_pass_3_seeds() -> None:
    r = run_r118(seeds=(0, 1, 2))
    assert r["all_passed"] is True, (
        f"R-118 failed: {r['pass_counts']}")
    for name, _ in R118_FAMILIES:
        assert r["pass_counts"].get(name, 0) == 3, (
            f"R-118 {name} did not pass 3/3 seeds")


def test_r116_has_14_families() -> None:
    assert len(R116_FAMILIES) == 14


def test_r117_has_14_families() -> None:
    assert len(R117_FAMILIES) == 14


def test_r118_has_15_families() -> None:
    assert len(R118_FAMILIES) == 15


def test_w57_total_h_bars_at_least_43() -> None:
    """H43..H85 = 43 hypothesis bars."""
    total = (len(R116_FAMILIES)
              + len(R117_FAMILIES)
              + len(R118_FAMILIES))
    assert total >= 43
