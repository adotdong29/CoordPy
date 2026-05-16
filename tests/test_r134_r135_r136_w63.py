"""W63 R-134 + R-135 + R-136 H-bar regression at three seeds.

Asserts that all 49 H-bars pass 3/3 seeds (147/147 cells), matching
the W63 success criterion.
"""

from __future__ import annotations


def _all_passed(family_runner, seeds=(199, 299, 399)):
    out = family_runner(seeds=seeds)
    fails = []
    for seed_result in out:
        for name, cell in seed_result["family_results"].items():
            if not cell["passed"]:
                fails.append(
                    (int(seed_result["seed"]), name, cell))
    return fails


def test_r134_w63_all_pass():
    from coordpy.r134_benchmark import run_r134
    fails = _all_passed(run_r134)
    assert not fails, f"R-134 failures: {fails}"


def test_r135_w63_all_pass():
    from coordpy.r135_benchmark import run_r135
    fails = _all_passed(run_r135)
    assert not fails, f"R-135 failures: {fails}"


def test_r136_w63_all_pass():
    from coordpy.r136_benchmark import run_r136
    fails = _all_passed(run_r136)
    assert not fails, f"R-136 failures: {fails}"


def test_r134_r135_r136_count():
    """Verify total H-bars: R-134 (17) + R-135 (16) + R-136 (16)
    = 49 H-bars across 3 seeds = 147 cells."""
    from coordpy.r134_benchmark import run_r134
    from coordpy.r135_benchmark import run_r135
    from coordpy.r136_benchmark import run_r136
    r134 = run_r134(seeds=(199,))
    r135 = run_r135(seeds=(199,))
    r136 = run_r136(seeds=(199,))
    total_h_bars = (
        len(r134[0]["family_results"])
        + len(r135[0]["family_results"])
        + len(r136[0]["family_results"]))
    assert total_h_bars == 49, (
        f"Expected 49 H-bars, got {total_h_bars}")
