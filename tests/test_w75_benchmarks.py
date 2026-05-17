"""W75 tests — R-181, R-182, R-183, R-184 benchmark families."""

from __future__ import annotations

from coordpy.r181_benchmark import run_r181
from coordpy.r182_benchmark import run_r182
from coordpy.r183_benchmark import run_r183
from coordpy.r184_benchmark import run_r184


def test_r181_all_pass_seed_set_a() -> None:
    r = run_r181(seeds=[1, 2, 3])
    assert r["all_pass"], r["cells"]


def test_r181_all_pass_seed_set_b() -> None:
    r = run_r181(seeds=[4, 5, 6])
    assert r["all_pass"], r["cells"]


def test_r182_all_pass_seed_set_a() -> None:
    r = run_r182(seeds=[1, 2, 3])
    assert r["all_pass"], r["cells"]


def test_r182_all_pass_seed_set_b() -> None:
    r = run_r182(seeds=[4, 5, 6])
    assert r["all_pass"], r["cells"]


def test_r183_all_pass_seed_set_a() -> None:
    r = run_r183(seeds=[1, 2, 3, 4, 5])
    assert r["all_pass"], r["cells"]


def test_r183_chain_regime_v20_beats_v19() -> None:
    r = run_r183(seeds=[1, 2, 3, 4, 5])
    chain_rate = r["per_regime_v20_beats"][
        "compound_repair_after_replacement_then_rejoin_under_budget"]
    assert chain_rate >= 0.5


def test_r184_all_pass_seed_set_a() -> None:
    r = run_r184(seeds=[1, 2, 3])
    assert r["all_pass"], r["cells"]


def test_r184_all_pass_seed_set_b() -> None:
    r = run_r184(seeds=[4, 5, 6])
    assert r["all_pass"], r["cells"]
