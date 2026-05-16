"""W68 R-152/R-153/R-154/R-155 benchmark sweep.

Runs the four families at three seeds and asserts strong success
(every H-bar passes on every seed).
"""

from __future__ import annotations


def test_r152_r153_r154_r155_strong_success():
    from coordpy.r152_benchmark import run_r152
    from coordpy.r153_benchmark import run_r153
    from coordpy.r154_benchmark import run_r154
    from coordpy.r155_benchmark import run_r155
    fails: list[tuple[str, int, str, str]] = []
    total = 0
    for run, name in [
            (run_r152, "R-152"),
            (run_r153, "R-153"),
            (run_r154, "R-154"),
            (run_r155, "R-155")]:
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
    # 10 + 16 + 14 + 6 = 46 H-bars * 3 seeds = 138 cells.
    assert total == 138
