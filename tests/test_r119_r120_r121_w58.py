"""Run R-119 + R-120 + R-121 across 3 seeds and assert all
families pass."""

from __future__ import annotations

from coordpy.r119_benchmark import run_r119
from coordpy.r120_benchmark import run_r120
from coordpy.r121_benchmark import run_r121


def test_r119_all_pass_3_seeds() -> None:
    res = run_r119(seeds=(0, 1, 2))
    fails = {
        k: v for k, v in res["pass_counts"].items()
        if v != 3}
    assert res["all_passed"], (
        f"R-119 failures: {fails}")


def test_r120_all_pass_3_seeds() -> None:
    res = run_r120(seeds=(0, 1, 2))
    fails = {
        k: v for k, v in res["pass_counts"].items()
        if v != 3}
    assert res["all_passed"], (
        f"R-120 failures: {fails}")


def test_r121_all_pass_3_seeds() -> None:
    res = run_r121(seeds=(0, 1, 2))
    fails = {
        k: v for k, v in res["pass_counts"].items()
        if v != 3}
    assert res["all_passed"], (
        f"R-121 failures: {fails}")
