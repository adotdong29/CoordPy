"""W60 R-125 / R-126 / R-127 integrated bench tests.

Runs each suite at a single seed and asserts every H-bar passes.
The full 3-seed sweep is exercised by the explicit ``run_*``
entry points.
"""

from __future__ import annotations

import pytest


def _check_all_pass(results: list[dict]) -> None:
    for sr in results:
        for name, r in sr["family_results"].items():
            assert r["passed"], (
                f"H-bar {name} failed at seed {sr['seed']}: "
                f"{r}")


def test_run_r125_seed():
    from coordpy.r125_benchmark import run_r125
    results = run_r125(seeds=(193,))
    _check_all_pass(results)


def test_run_r126_seed():
    from coordpy.r126_benchmark import run_r126
    results = run_r126(seeds=(194,))
    _check_all_pass(results)


def test_run_r127_seed():
    from coordpy.r127_benchmark import run_r127
    results = run_r127(seeds=(195,))
    _check_all_pass(results)
