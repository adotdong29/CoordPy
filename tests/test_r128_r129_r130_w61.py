"""W61 R-128 / R-129 / R-130 integrated bench tests.

Runs each suite at a single seed and asserts every H-bar passes.
The full 3-seed sweep is exercised by the explicit ``run_*``
entry points in each benchmark module.
"""

from __future__ import annotations


def _check_all_pass(results: list[dict]) -> None:
    for sr in results:
        for name, r in sr["family_results"].items():
            assert r["passed"], (
                f"H-bar {name} failed at seed {sr['seed']}: "
                f"{r}")


def test_run_r128_seed():
    from coordpy.r128_benchmark import run_r128
    results = run_r128(seeds=(196,))
    _check_all_pass(results)


def test_run_r129_seed():
    from coordpy.r129_benchmark import run_r129
    results = run_r129(seeds=(197,))
    _check_all_pass(results)


def test_run_r130_seed():
    from coordpy.r130_benchmark import run_r130
    results = run_r130(seeds=(198,))
    _check_all_pass(results)
