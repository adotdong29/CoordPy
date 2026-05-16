"""W70 R-161/R-162/R-163/R-164 benchmark sweep.

Runs the four families at three seeds and asserts strong success
(every H-bar passes on every seed). 60 H-bars × 3 seeds = 180 cells.
"""

from __future__ import annotations


def test_r161_r162_r163_r164_strong_success():
    from coordpy.r161_benchmark import run_r161
    from coordpy.r162_benchmark import run_r162
    from coordpy.r163_benchmark import run_r163
    from coordpy.r164_benchmark import run_r164
    fails: list[tuple[str, int, str, str]] = []
    total = 0
    for run, name in [
            (run_r161, "R-161"),
            (run_r162, "R-162"),
            (run_r163, "R-163"),
            (run_r164, "R-164")]:
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
    # 10 + 16 + 22 + 12 = 60 H-bars * 3 seeds = 180 cells.
    assert total == 180
