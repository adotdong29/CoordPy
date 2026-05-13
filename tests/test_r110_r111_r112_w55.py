"""W55 R-110 / R-111 / R-112 benchmark family tests."""

from __future__ import annotations

from coordpy.r110_benchmark import (
    R110_FAMILY_TABLE,
    R110_W55_ARM,
    run_all_families as run_all_r110,
)
from coordpy.r111_benchmark import (
    R111_FAMILY_TABLE,
    R111_W55_ARM,
    run_all_families as run_all_r111,
)
from coordpy.r112_benchmark import (
    R112_FAMILY_TABLE,
    R112_W55_ARM,
    run_all_families as run_all_r112,
)


def test_r110_family_table_has_12() -> None:
    assert len(R110_FAMILY_TABLE) == 12


def test_r111_family_table_has_10() -> None:
    assert len(R111_FAMILY_TABLE) == 10


def test_r112_family_table_has_16() -> None:
    assert len(R112_FAMILY_TABLE) == 16


def test_r110_w55_arm_passes_majority() -> None:
    out = run_all_r110(seeds=(1, 2, 3))
    n_pass = 0
    for name, c in out.items():
        w = c.get(R110_W55_ARM)
        # Trivial pass + bench bars: at least 9/12 should be
        # ≥ 0.5 mean (the floor for "passes").
        if w is not None and w.mean >= 0.5:
            n_pass += 1
    assert n_pass >= 9


def test_r111_w55_arm_passes_majority() -> None:
    out = run_all_r111(seeds=(1, 2, 3))
    n_pass = 0
    for name, c in out.items():
        w = c.get(R111_W55_ARM)
        if w is not None and w.mean >= 0.5:
            n_pass += 1
    assert n_pass >= 8


def test_r112_w55_arm_passes_majority() -> None:
    out = run_all_r112(seeds=(1, 2, 3))
    n_pass = 0
    for name, c in out.items():
        w = c.get(R112_W55_ARM)
        if w is not None and w.mean >= 0.5:
            n_pass += 1
    # 16 families; require ≥ 12 passing.
    assert n_pass >= 12


def test_r110_trivial_passthrough_at_1() -> None:
    out = run_all_r110(seeds=(1, 2, 3))
    fam = out["family_trivial_w55_passthrough"]
    w = fam.get(R110_W55_ARM)
    assert w.mean == 1.0


def test_r112_bch_double_correct_above_floor() -> None:
    out = run_all_r112(seeds=(1, 2, 3))
    fam = out["family_bch_double_bit_correct"]
    w = fam.get(R112_W55_ARM)
    assert w.mean >= 0.85


def test_r111_ecc_v7_18_bits() -> None:
    out = run_all_r111(seeds=(1, 2, 3))
    fam = out["family_ecc_v7_compression_18_bits"]
    w = fam.get(R111_W55_ARM)
    assert w.mean == 1.0
