"""W62 R-131 / R-132 / R-133 integrated bench tests.

Runs each suite at a single seed and asserts every H-bar passes.
The full 3-seed sweep is exercised by the explicit ``run_*`` entry
points in each benchmark module.
"""

from __future__ import annotations


def _check_all_pass(results: list[dict]) -> None:
    for sr in results:
        for name, r in sr["family_results"].items():
            assert r["passed"], (
                f"H-bar {name} failed at seed {sr['seed']}: {r}")


def test_run_r131_seed():
    from coordpy.r131_benchmark import run_r131
    results = run_r131(seeds=(199,))
    _check_all_pass(results)


def test_run_r132_seed():
    from coordpy.r132_benchmark import run_r132
    results = run_r132(seeds=(200,))
    _check_all_pass(results)


def test_run_r133_seed():
    from coordpy.r133_benchmark import run_r133
    results = run_r133(seeds=(201,))
    _check_all_pass(results)
