"""W74 benchmark family smoke tests.

Run each W74 benchmark family on three seeds and assert every cell
passes. Mirrors the W73 ``test_r173_r174_r175_r176_w73.py`` shape.
"""

from __future__ import annotations


def _assert_all_pass(results, name):
    for r in results:
        for cell_name, cell in r["family_results"].items():
            assert bool(cell.get("passed", False)), (
                f"{name} cell {cell_name} (seed "
                f"{r['seed']}) failed: {cell}")


def test_r177_hosted_v7_all_pass():
    from coordpy.r177_benchmark import run_r177
    _assert_all_pass(
        run_r177(seeds=(199, 299, 399)), "R-177")


def test_r178_substrate_v19_all_pass():
    from coordpy.r178_benchmark import run_r178
    _assert_all_pass(
        run_r178(seeds=(199, 299, 399)), "R-178")


def test_r179_masc_v10_all_pass():
    from coordpy.r179_benchmark import run_r179
    _assert_all_pass(
        run_r179(seeds=(199, 299, 399)), "R-179")


def test_r180_handoff_v6_falsifier_all_pass():
    from coordpy.r180_benchmark import run_r180
    _assert_all_pass(
        run_r180(seeds=(199, 299, 399)), "R-180")
