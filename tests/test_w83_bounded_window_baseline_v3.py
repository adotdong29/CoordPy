"""W83 — bounded-window baseline V3 falsifier tests."""

from __future__ import annotations


def test_w83_bw_v3_default_config_explicit():
    from coordpy.bounded_window_baseline_v3 import (
        build_bounded_window_baseline_v3,
    )
    b = build_bounded_window_baseline_v3()
    assert int(b.window_size) == 256
    assert float(b.summary_fidelity) >= 0.5
    assert int(b.retrieval_k) >= 1
    assert len(b.cid()) == 64


def test_w83_bw_v3_answers_inside_window():
    import numpy as np
    from coordpy.bounded_window_baseline_v3 import (
        BoundedWindowEventV3,
        answer_reconstruction_query_v3,
        build_bounded_window_baseline_v3,
    )
    b = build_bounded_window_baseline_v3()
    rng = np.random.default_rng(1)
    events = [
        BoundedWindowEventV3(
            turn_index=int(i),
            feature=rng.standard_normal((b.feature_dim,)),
            payload_cid=f"cid_{i}")
        for i in range(int(b.window_size))]
    ans = answer_reconstruction_query_v3(
        baseline=b, events_window=events,
        summary_feature_centroid=None,
        summary_covers_turns=(0, 0),
        target_turn_index=int(b.window_size) // 2)
    assert bool(ans.success)
    assert ans.answer_source == "window"


def test_w83_bw_v3_answers_inside_summary():
    from coordpy.bounded_window_baseline_v3 import (
        answer_reconstruction_query_v3,
        build_bounded_window_baseline_v3,
    )
    b = build_bounded_window_baseline_v3()
    ans = answer_reconstruction_query_v3(
        baseline=b, events_window=[],
        summary_feature_centroid=None,
        summary_covers_turns=(100, 500),
        target_turn_index=300,
        required_fidelity=0.5)
    assert bool(ans.success)
    assert ans.answer_source == "summary"


def test_w83_bw_v3_abstains_beyond_coverage_at_all_horizons():
    from coordpy.bounded_window_baseline_v3 import (
        build_bounded_window_baseline_v3,
        prove_bounded_window_v3_insufficient_v1,
    )
    b = build_bounded_window_baseline_v3()
    proof = prove_bounded_window_v3_insufficient_v1(
        baseline=b, summary_coverage_turns=512,
        horizons_to_test=(1024, 2048, 8192, 32_768, 100_000))
    assert float(proof.failure_rate_beyond_coverage) >= (
        1.0 - 1e-12)
    assert int(proof.n_horizons_tested) >= 5
    assert len(proof.failure_horizons) == 5


def test_w83_bw_v3_emits_witness():
    from coordpy.bounded_window_baseline_v3 import (
        build_bounded_window_baseline_v3,
        emit_bounded_window_baseline_v3_witness_v1,
        prove_bounded_window_v3_insufficient_v1,
    )
    b = build_bounded_window_baseline_v3()
    proof = prove_bounded_window_v3_insufficient_v1(
        baseline=b, summary_coverage_turns=512,
        horizons_to_test=(1024, 8192))
    w = emit_bounded_window_baseline_v3_witness_v1(
        baseline=b, proof=proof)
    assert len(w.cid()) == 64
    assert float(w.proof_failure_rate) >= 1.0 - 1e-12
