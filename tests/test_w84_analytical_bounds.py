"""W84 / P1 #35 — Analytical Bounds tests.

Each test exercises ONE proved theorem and asserts the proof's
predicted bound is not violated by the empirical bench.
"""

from __future__ import annotations


def test_w84_proved_replay_from_kv_byte_identity_holds():
    """Theorem W84-T-REPLAY-FROM-KV-BYTE-IDENTICAL (proved).

    Empirical bound: 8 · fp64_eps · max(|logits|).
    Proof file: papers/proofs/W84_replay_from_kv_byte_identical.md
    """
    from coordpy.analytical_bounds_v1 import (
        check_replay_from_kv_byte_identity_v1,
    )
    chk = check_replay_from_kv_byte_identity_v1()
    assert bool(chk.bound_holds), chk.to_dict()
    # The observed diff must be very close to 0 (within fp64
    # ULP * magnitude).
    assert (
        float(chk.max_observed_diff)
        <= float(chk.bound_value))


def test_w84_proved_honest_consensus_error_bound_holds():
    """Theorem W84-T-HONEST-WITNESS-CONSENSUS-ERROR-BOUND
    (proved).

    Proved bound: E[||consensus - μ||²] = d σ² / h exactly.
    Empirical: Monte Carlo within 10% relative tolerance.
    Proof file: papers/proofs/W84_honest_witness_consensus_error_bound.md
    """
    from coordpy.analytical_bounds_v1 import (
        check_honest_witness_consensus_error_bound_v1,
    )
    chk = check_honest_witness_consensus_error_bound_v1(
        h=32, d=4, sigma=1.0, n_trials=400)
    assert bool(
        chk.bound_holds_within_tolerance), chk.to_dict()
    # The Monte Carlo measured MSE is close to the bound.
    assert (
        float(chk.relative_error) <= 0.10), chk.to_dict()


def test_w84_proved_integrity_filtering_variance_optimal_holds():
    """Theorem W84-T-INTEGRITY-FILTERING-VARIANCE-OPTIMAL
    (proved).

    Proved bound (filtered): E[||filtered - μ||²] = d σ²/h
    regardless of tamper.
    Proof file: papers/proofs/W84_integrity_filtering_variance_optimal.md
    """
    from coordpy.analytical_bounds_v1 import (
        check_integrity_filtering_variance_optimal_v1,
    )
    chk = check_integrity_filtering_variance_optimal_v1(
        h=16, t=8, d=4, sigma=1.0,
        tamper_min=5.0, tamper_max=10.0,
        n_trials=400)
    assert bool(
        chk.filtered_bound_holds_within_tolerance
    ), chk.to_dict()
    # The filtered MSE is much lower than the unfiltered MSE
    # (the adversarial tamper drives unfiltered up; filtered
    # stays at the honest noise floor).
    assert (
        float(chk.measured_unfiltered_mse)
        > float(chk.measured_filtered_mse) * 10.0
    ), chk.to_dict()


def test_w84_three_theorems_promoted_to_proved():
    """Meta-test: three theorems are promoted from empirical /
    proved-conditional to proved.

    Verifies:
    * proof file exists for each of the three theorems;
    * each proof's empirical bound is checked by a separate
      test in this file.
    """
    import os
    proofs = [
        "papers/proofs/W84_replay_from_kv_byte_identical.md",
        "papers/proofs/W84_honest_witness_consensus_error_bound.md",
        ("papers/proofs/"
         "W84_integrity_filtering_variance_optimal.md"),
    ]
    for p in proofs:
        assert os.path.exists(p), f"missing proof: {p}"
        # Each proof is at least 50 lines (non-trivial).
        with open(p, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
        assert len(lines) >= 50, (p, len(lines))


def test_w84_analytical_bounds_bench_passes_all_three():
    """End-to-end bench: all three proved bounds hold."""
    from coordpy.analytical_bounds_v1 import (
        run_analytical_bounds_bench_v1,
    )
    rep = run_analytical_bounds_bench_v1()
    assert bool(rep.all_three_bounds_hold), rep.to_dict()


def test_w84_proofs_state_explicit_assumptions():
    """Anti-cheat: each proof must state its assumptions
    explicitly. Spec: 'Every assumption is explicit; if the
    proof requires f < n/3, the theorem statement says so.'

    We check each proof file contains the word 'Assumption'
    (capitalised) or 'Assume' as a section / explicit
    declaration.
    """
    proofs = [
        "papers/proofs/W84_replay_from_kv_byte_identical.md",
        "papers/proofs/W84_honest_witness_consensus_error_bound.md",
        ("papers/proofs/"
         "W84_integrity_filtering_variance_optimal.md"),
    ]
    for p in proofs:
        with open(p, "r", encoding="utf-8") as f:
            content = f.read()
        assert ("Assumption" in content
                or "Assume" in content), (
            f"{p} lacks explicit assumptions")


def test_w84_bench_report_cid_stable():
    """The bench report is content-addressed."""
    from coordpy.analytical_bounds_v1 import (
        run_analytical_bounds_bench_v1,
    )
    rep = run_analytical_bounds_bench_v1()
    assert len(rep.cid()) == 64


def test_w84_integrity_filtering_strictly_below_unfiltered():
    """Anti-cheat: tampering MUST move the unfiltered error
    (otherwise the filter axis is decorative). Verify the
    unfiltered MSE is at least 10x the filtered MSE on the
    high-tamper regime."""
    from coordpy.analytical_bounds_v1 import (
        check_integrity_filtering_variance_optimal_v1,
    )
    chk = check_integrity_filtering_variance_optimal_v1(
        h=16, t=8, d=4, sigma=1.0,
        tamper_min=5.0, tamper_max=10.0,
        n_trials=200)
    assert bool(chk.filtered_strictly_below_unfiltered)
    assert (
        float(chk.measured_unfiltered_mse)
        / max(1e-12, float(chk.measured_filtered_mse))
        > 10.0)
