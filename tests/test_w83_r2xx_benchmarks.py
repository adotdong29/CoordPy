"""W83 — R-2XX benchmark family tests."""

from __future__ import annotations


def test_r202_all_pass():
    from coordpy.r202_benchmark import run_r202
    rep = run_r202()
    assert bool(rep.all_pass), [
        h.to_dict() for h in rep.h_bars]
    assert len(rep.cid()) == 64
    # 7 H-bars in R-202.
    assert len(rep.h_bars) == 7


def test_r203_all_pass():
    from coordpy.r203_benchmark import run_r203
    rep = run_r203()
    assert bool(rep.all_pass), [
        h.to_dict() for h in rep.h_bars]
    assert len(rep.cid()) == 64
    # 5 H-bars in R-203.
    assert len(rep.h_bars) == 5


def test_r204_all_pass():
    from coordpy.r204_benchmark import run_r204
    rep = run_r204()
    assert bool(rep.all_pass), [
        h.to_dict() for h in rep.h_bars]
    assert len(rep.cid()) == 64
    # 8 H-bars in R-204 (7 original + competitive-with-ridge).
    assert len(rep.h_bars) >= 7


def test_r202_bench_report_content_addressed():
    from coordpy.r202_benchmark import run_r202
    a = run_r202()
    b = run_r202()
    assert str(a.cid()) == str(b.cid())


def test_r203_baseline_and_proof_cids_unique():
    from coordpy.r203_benchmark import run_r203
    rep = run_r203()
    assert rep.baseline_cid != rep.proof_cid


def test_r204_composed_strictly_beats_ridge():
    from coordpy.r204_benchmark import run_r204
    rep = run_r204()
    assert float(rep.composed_mse) < float(rep.ridge_mse)
