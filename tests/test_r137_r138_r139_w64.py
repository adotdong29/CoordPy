"""W64 R-137/R-138/R-139 benchmark suites.

R-137: V9 substrate / replay-dominance / hidden-wins-primary /
       four-way bridge / nine-way hybrid / V9 substrate axes
R-138: long-horizon retention / persistent V16 / multi-hop V14 /
       LHR V16 / ECC V16
R-139: corruption / disagreement / consensus / fallback /
       replay-dominance-primary / hostile-channel
"""

from __future__ import annotations


def test_r137_passes_3_seeds():
    from coordpy.r137_benchmark import run_r137
    res = run_r137(seeds=(199, 299, 399))
    for seed_res in res:
        for name, r in seed_res["family_results"].items():
            assert r.get("passed"), (
                f"R-137 cell {name} seed "
                f"{seed_res['seed']} failed: {r}")


def test_r138_passes_3_seeds():
    from coordpy.r138_benchmark import run_r138
    res = run_r138(seeds=(199, 299, 399))
    for seed_res in res:
        for name, r in seed_res["family_results"].items():
            assert r.get("passed"), (
                f"R-138 cell {name} seed "
                f"{seed_res['seed']} failed: {r}")


def test_r139_passes_3_seeds():
    from coordpy.r139_benchmark import run_r139
    res = run_r139(seeds=(199, 299, 399))
    for seed_res in res:
        for name, r in seed_res["family_results"].items():
            assert r.get("passed"), (
                f"R-139 cell {name} seed "
                f"{seed_res['seed']} failed: {r}")
