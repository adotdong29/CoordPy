"""W71 benchmark family smoke tests.

Run each W71 benchmark family on three seeds and assert every cell
passes. Mirrors the W70 ``test_r161_r162_r163_r164_w70.py`` shape.
"""

from __future__ import annotations


def _assert_all_pass(results, name):
    for r in results:
        for cell_name, cell in r["family_results"].items():
            assert bool(cell.get("passed", False)), (
                f"{name} cell {cell_name} (seed "
                f"{r['seed']}) failed: {cell}")


def test_r165_hosted_v4_all_pass():
    from coordpy.r165_benchmark import run_r165
    _assert_all_pass(
        run_r165(seeds=(199, 299, 399)), "R-165")


def test_r166_substrate_v16_all_pass():
    from coordpy.r166_benchmark import run_r166
    _assert_all_pass(
        run_r166(seeds=(199, 299, 399)), "R-166")


def test_r167_masc_v7_all_pass():
    from coordpy.r167_benchmark import run_r167
    _assert_all_pass(
        run_r167(seeds=(199, 299, 399)), "R-167")


def test_r168_handoff_v3_falsifier_all_pass():
    from coordpy.r168_benchmark import run_r168
    _assert_all_pass(
        run_r168(seeds=(199, 299, 399)), "R-168")
