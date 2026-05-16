"""W67 R-149/R-150/R-151 benchmark sweep.

Runs the three families at three seeds and asserts strong success
(every H-bar passes on every seed).
"""

from __future__ import annotations


def test_r149_r150_r151_strong_success():
    from coordpy.r149_benchmark import run_r149
    from coordpy.r150_benchmark import run_r150
    from coordpy.r151_benchmark import run_r151
    fails: list[tuple[str, int, str, str]] = []
    total = 0
    for run, name in [
            (run_r149, "R-149"), (run_r150, "R-150"),
            (run_r151, "R-151")]:
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
    # 56 H-bars * 3 seeds = 168 cells.
    assert total == 168
