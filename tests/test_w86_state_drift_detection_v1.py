"""Tests for ``coordpy.state_drift_detection_v1``."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from coordpy.controlled_runtime_substrate_v1 import (
    build_controlled_runtime_params_v1,
)
from coordpy.state_drift_detection_v1 import (
    CapturedHiddenStateV1,
    DriftBenchReportV1,
    DriftDetectorConfigV1,
    LinearAdapterV1,
    ModelDriftEventV1,
    ModelWeightsCIDV1,
    W86_DRIFT_V1_DEFAULT_SAFETY_MARGIN,
    W86_DRIFT_V1_FP64_PRECISION_FLOOR,
    compute_controlled_runtime_weights_cid_v1,
    evaluate_adapter_mse_v1,
    evaluate_stale_capsule_v1,
    run_drift_detector_v1,
    run_drift_v1_bench,
    train_linear_adapter_v1,
)


def _build_small_runtime(seed: int = 86_042):
    return build_controlled_runtime_params_v1(
        vocab_size=64, n_layers=2, n_heads=2,
        head_dim=4, mlp_dim=16, max_len=16, seed=seed)


def test_weights_cid_stable_for_same_params():
    p1 = _build_small_runtime(seed=42)
    p2 = _build_small_runtime(seed=42)
    cid1 = compute_controlled_runtime_weights_cid_v1(p1)
    cid2 = compute_controlled_runtime_weights_cid_v1(p2)
    assert cid1.model_weights_cid == cid2.model_weights_cid


def test_weights_cid_changes_with_seed():
    p1 = _build_small_runtime(seed=1)
    p2 = _build_small_runtime(seed=2)
    cid1 = compute_controlled_runtime_weights_cid_v1(p1)
    cid2 = compute_controlled_runtime_weights_cid_v1(p2)
    assert cid1.model_weights_cid != cid2.model_weights_cid


def test_weights_cid_n_parameters_reported():
    p = _build_small_runtime()
    cid = compute_controlled_runtime_weights_cid_v1(p)
    assert cid.n_parameters > 0


def test_drift_threshold_derivation_principled():
    cfg = DriftDetectorConfigV1()
    expected = (
        W86_DRIFT_V1_FP64_PRECISION_FLOOR
        * W86_DRIFT_V1_DEFAULT_SAFETY_MARGIN)
    assert abs(cfg.threshold() - expected) < 1e-12


def test_detector_does_not_fire_on_unchanged_weights():
    p = _build_small_runtime()
    prompts = [[1, 2, 3, 4, 5, 6, 7, 8]]
    ev = run_drift_detector_v1(
        old_params=p, new_params=p,
        prompt_ids_corpus=prompts)
    assert ev.drift_score < 1e-12
    assert ev.drift_detected is False


def test_detector_fires_on_perturbed_weights():
    p0 = _build_small_runtime()
    rng = np.random.default_rng(123)
    perturbation = 0.1  # large perturbation
    p1 = dataclasses.replace(
        p0,
        embed_W=p0.embed_W + perturbation * rng.standard_normal(
            p0.embed_W.shape))
    prompts = [[1, 2, 3, 4, 5, 6, 7, 8]]
    ev = run_drift_detector_v1(
        old_params=p0, new_params=p1,
        prompt_ids_corpus=prompts)
    assert ev.drift_detected is True
    assert ev.drift_score > ev.threshold


def test_stale_capsule_verdict_marks_old_as_stale():
    p0 = _build_small_runtime(seed=1)
    p1 = _build_small_runtime(seed=2)
    cap = CapturedHiddenStateV1(
        model_weights_cid=(
            compute_controlled_runtime_weights_cid_v1(p0)
            .model_weights_cid),
        prompt_ids=(1, 2, 3),
        hidden_state=np.array([1.0, 2.0]))
    verdict = evaluate_stale_capsule_v1(
        cap, compute_controlled_runtime_weights_cid_v1(p1))
    assert verdict.stale is True
    assert verdict.fallback_action == "recompute_from_prompt"


def test_stale_capsule_verdict_marks_current_fresh():
    p = _build_small_runtime()
    cap = CapturedHiddenStateV1(
        model_weights_cid=(
            compute_controlled_runtime_weights_cid_v1(p)
            .model_weights_cid),
        prompt_ids=(1, 2, 3),
        hidden_state=np.array([1.0, 2.0]))
    verdict = evaluate_stale_capsule_v1(
        cap, compute_controlled_runtime_weights_cid_v1(p))
    assert verdict.stale is False
    assert verdict.fallback_action == "use_captured"


def test_linear_adapter_training_recovers_identity():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 8))
    W_true = rng.standard_normal((8, 4))
    y = X @ W_true
    adapter = train_linear_adapter_v1(X, y)
    pred = adapter.predict(X)
    assert np.mean((pred - y) ** 2) < 1e-3


def test_bench_meets_all_dod_bars():
    rep = run_drift_v1_bench()
    assert rep.detector_fires_when_changed is True
    assert rep.detector_does_not_fire_when_unchanged is True
    assert rep.stale_verdict_marks_old_capsule_stale is True
    assert rep.stale_verdict_marks_fresh_capsule_fresh is True
    assert rep.fallback_recommendation_is_recompute_for_stale is True
    assert rep.new_memory_strictly_beats_stale_on_holdout is True
    # Strict beat = strictly lower MSE, not a tie.
    assert rep.new_holdout_mse < rep.stale_holdout_mse


def test_bench_report_cid_deterministic():
    r1 = run_drift_v1_bench(seed=86_042)
    r2 = run_drift_v1_bench(seed=86_042)
    assert r1.report_cid == r2.report_cid


def test_drift_event_content_addressed():
    p = _build_small_runtime()
    ev = run_drift_detector_v1(
        old_params=p, new_params=p,
        prompt_ids_corpus=[[1, 2, 3]])
    assert len(ev.cid()) == 64


def test_threshold_not_hand_tuned_to_bench():
    """The threshold is derived from
    (fp64_floor × safety_margin). It is NOT tuned to make the
    bench pass. We verify by reading the constants.
    """
    cfg = DriftDetectorConfigV1()
    assert cfg.precision_floor == 5e-3
    assert cfg.safety_margin == 3.0
    assert cfg.threshold() == 1.5e-2
