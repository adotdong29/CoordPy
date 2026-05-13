"""W55 M11 — disagreement algebra unit tests."""

from __future__ import annotations

from coordpy.disagreement_algebra import (
    AlgebraTrace,
    check_difference_self_cancellation,
    check_intersection_distributivity_on_agreement,
    check_merge_idempotent,
    difference_op,
    difference_op_traced,
    emit_disagreement_algebra_witness,
    intersection_op,
    intersection_op_traced,
    merge_op,
    merge_op_traced,
    verify_disagreement_algebra_witness,
)


def test_merge_idempotent_on_equal_inputs() -> None:
    a = [1.0, 0.5, -0.3, 0.0]
    r = check_merge_idempotent(a)
    assert r.ok
    assert r.residual_l2 < 1e-9


def test_difference_self_cancels() -> None:
    a = [1.0, -1.0, 0.5, 0.0]
    r = check_difference_self_cancellation(a)
    assert r.ok
    assert r.residual_l2 < 1e-9


def test_intersection_distributivity_on_agreement_subspace() -> None:
    a = [1.0, 0.0, 0.5, 0.0]
    b = [1.0, 1.0, 0.5, 0.0]  # disagrees on dim 1 only
    c = [1.0, 0.5, 0.5, 1.0]
    r = check_intersection_distributivity_on_agreement(
        a, b, c)
    # On dims 0, 2, 3 (agreement subspace) distributive identity
    # must hold.
    assert r.residual_l2 < 1e-6


def test_merge_is_convex_combination() -> None:
    a = [0.0, 0.0, 0.0]
    b = [1.0, 1.0, 1.0]
    m = merge_op(a, b)
    for v in m:
        assert 0.0 <= v <= 1.0


def test_difference_is_nonnegative() -> None:
    a = [1.0, -0.5, 0.3]
    b = [-0.5, 0.7, 0.0]
    d = difference_op(a, b)
    for v in d:
        assert v >= 0.0


def test_intersection_zero_off_agreement_subspace() -> None:
    a = [1.0, 5.0]
    b = [1.0, -5.0]  # huge disagreement on dim 1
    vec, mask = intersection_op(
        a, b, agreement_floor=0.1)
    assert mask[1] == 0
    assert vec[1] == 0.0
    assert mask[0] == 1


def test_algebra_trace_records_ops() -> None:
    trace = AlgebraTrace.empty()
    a = [0.5, 0.3]
    b = [0.4, 0.2]
    merge_op_traced(a, b, trace=trace)
    difference_op_traced(a, b, trace=trace)
    intersection_op_traced(a, b, trace=trace)
    assert len(trace.steps) == 3
    assert {s.op_kind for s in trace.steps} == {
        "merge", "difference", "intersection"}


def test_algebra_witness_emits_pass_count() -> None:
    trace = AlgebraTrace.empty()
    a = [0.5, 0.2, -0.3]
    r1 = check_merge_idempotent(a)
    r2 = check_difference_self_cancellation(a)
    w = emit_disagreement_algebra_witness(
        trace=trace, identity_results=(r1, r2))
    assert w.identity_check_pass_count == 2


def test_algebra_verifier_passes_on_clean_witness() -> None:
    trace = AlgebraTrace.empty()
    a = [0.5, 0.2, -0.3]
    r = check_merge_idempotent(a)
    w = emit_disagreement_algebra_witness(
        trace=trace, identity_results=(r,))
    res = verify_disagreement_algebra_witness(
        w, expected_trace_cid=trace.cid())
    assert res["ok"]
