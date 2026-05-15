"""W59 benchmark family acceptance tests.

R-122 (real-substrate / trained-controller / cache-retrieval /
partial-prefix), R-123 (long-horizon retention / reconstruction /
aggressive cramming), R-124 (corruption / disagreement /
consensus / fallback).
"""

from __future__ import annotations

import pytest


def test_r122_passes_at_three_seeds():
    from coordpy.r122_benchmark import run_r122
    results = run_r122(seeds=(190, 290, 390))
    for r in results:
        for name, fr in r["family_results"].items():
            assert fr["passed"], (
                f"R-122 {name} failed at seed {r['seed']}: "
                f"{fr}")


def test_r123_passes_at_three_seeds():
    from coordpy.r123_benchmark import run_r123
    results = run_r123(seeds=(191, 291, 391))
    for r in results:
        for name, fr in r["family_results"].items():
            assert fr["passed"], (
                f"R-123 {name} failed at seed {r['seed']}: "
                f"{fr}")


def test_r124_passes_at_three_seeds():
    from coordpy.r124_benchmark import run_r124
    results = run_r124(seeds=(192, 292, 392))
    for r in results:
        for name, fr in r["family_results"].items():
            assert fr["passed"], (
                f"R-124 {name} failed at seed {r['seed']}: "
                f"{fr}")
