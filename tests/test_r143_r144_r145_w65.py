"""W65 R-143/R-144/R-145 benchmark sweep.

Runs the three families at three seeds and asserts strong success
(every H-bar passes on every seed).
"""

from __future__ import annotations


def test_r143_r144_r145_strong_success():
    from coordpy.r143_benchmark import run_r143
    from coordpy.r144_benchmark import run_r144
    from coordpy.r145_benchmark import run_r145
    fails: list[tuple[str, int, str, str]] = []
    total = 0
    for run, name in [
            (run_r143, "R-143"), (run_r144, "R-144"),
            (run_r145, "R-145")]:
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
    # 50 H-bars * 3 seeds = 150 cells.
    assert total == 150
