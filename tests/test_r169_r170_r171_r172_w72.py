"""W72 benchmark family smoke tests.

Run each W72 benchmark family on three seeds and assert every cell
passes. Mirrors the W71 ``test_r165_r166_r167_r168_w71.py`` shape.
"""

from __future__ import annotations


def _assert_all_pass(results, name):
    for r in results:
        for cell_name, cell in r["family_results"].items():
            assert bool(cell.get("passed", False)), (
                f"{name} cell {cell_name} (seed "
                f"{r['seed']}) failed: {cell}")


def test_r169_hosted_v5_all_pass():
    from coordpy.r169_benchmark import run_r169
    _assert_all_pass(
        run_r169(seeds=(199, 299, 399)), "R-169")


def test_r170_substrate_v17_all_pass():
    from coordpy.r170_benchmark import run_r170
    _assert_all_pass(
        run_r170(seeds=(199, 299, 399)), "R-170")


def test_r171_masc_v8_all_pass():
    from coordpy.r171_benchmark import run_r171
    _assert_all_pass(
        run_r171(seeds=(199, 299, 399)), "R-171")


def test_r172_handoff_v4_falsifier_all_pass():
    from coordpy.r172_benchmark import run_r172
    _assert_all_pass(
        run_r172(seeds=(199, 299, 399)), "R-172")
