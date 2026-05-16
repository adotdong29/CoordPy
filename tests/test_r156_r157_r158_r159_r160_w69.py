"""W69 R-156/R-157/R-158/R-159/R-160 benchmark sweep.

Runs the five families at three seeds and asserts strong success
(every H-bar passes on every seed). 60 H-bars × 3 seeds = 186
cells.
"""

from __future__ import annotations


def test_r156_r157_r158_r159_r160_strong_success():
    from coordpy.r156_benchmark import run_r156
    from coordpy.r157_benchmark import run_r157
    from coordpy.r158_benchmark import run_r158
    from coordpy.r159_benchmark import run_r159
    from coordpy.r160_benchmark import run_r160
    fails: list[tuple[str, int, str, str]] = []
    total = 0
    for run, name in [
            (run_r156, "R-156"),
            (run_r157, "R-157"),
            (run_r158, "R-158"),
            (run_r159, "R-159"),
            (run_r160, "R-160")]:
        res = run(seeds=(199, 299, 399))
        for sr in res:
            for h_name, r in sr["family_results"].items():
                total += 1
                if not r["passed"]:
                    fails.append((
                        name, int(sr["seed"]), h_name,
                        str(r.get("exception", ""))))
    assert not fails, (
        f"failures: {fails}; total={total}")
    # 10 + 17 + 18 + 9 + 8 = 62 H-bars * 3 seeds = 186 cells.
    assert total == 186
