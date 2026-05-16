"""W66 R-146/R-147/R-148 benchmark sweep.

Runs the three families at three seeds and asserts strong success
(every H-bar passes on every seed).
"""

from __future__ import annotations


def test_r146_r147_r148_strong_success():
    from coordpy.r146_benchmark import run_r146
    from coordpy.r147_benchmark import run_r147
    from coordpy.r148_benchmark import run_r148
    fails: list[tuple[str, int, str, str]] = []
    total = 0
    for run, name in [
            (run_r146, "R-146"), (run_r147, "R-147"),
            (run_r148, "R-148")]:
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
