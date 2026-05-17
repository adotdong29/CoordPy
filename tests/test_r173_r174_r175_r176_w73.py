"""W73 benchmark family smoke tests.

Run each W73 benchmark family on three seeds and assert every cell
passes. Mirrors the W72 ``test_r169_r170_r171_r172_w72.py`` shape.
"""

from __future__ import annotations


def _assert_all_pass(results, name):
    for r in results:
        for cell_name, cell in r["family_results"].items():
            assert bool(cell.get("passed", False)), (
                f"{name} cell {cell_name} (seed "
                f"{r['seed']}) failed: {cell}")


def test_r173_hosted_v6_all_pass():
    from coordpy.r173_benchmark import run_r173
    _assert_all_pass(
        run_r173(seeds=(199, 299, 399)), "R-173")


def test_r174_substrate_v18_all_pass():
    from coordpy.r174_benchmark import run_r174
    _assert_all_pass(
        run_r174(seeds=(199, 299, 399)), "R-174")


def test_r175_masc_v9_all_pass():
    from coordpy.r175_benchmark import run_r175
    _assert_all_pass(
        run_r175(seeds=(199, 299, 399)), "R-175")


def test_r176_handoff_v5_falsifier_all_pass():
    from coordpy.r176_benchmark import run_r176
    _assert_all_pass(
        run_r176(seeds=(199, 299, 399)), "R-176")
