"""W84 / P0 #26 — live LLM training of composed learned memory tests.

These tests are gated on the optional ``transformers`` + ``torch``
deps. When they are missing, the tests skip cleanly so CI on
lean environments stays green.

The full head-to-head test takes ~30s on CPU with distilgpt2 +
~70 training iterations; pytest skips it unless
``COORDPY_RUN_LIVE_TRAINING_BENCH=1`` is set.
"""

from __future__ import annotations

import os

import pytest

try:
    import torch  # noqa: F401
    import transformers  # noqa: F401
    _HAS_HF = True
except Exception:  # noqa: BLE001
    _HAS_HF = False


def test_w84_live_dataset_module_exports():
    from coordpy import live_hidden_state_dataset_v1 as lds
    for name in (
        "W84_LIVE_DATASET_V1_SCHEMA_VERSION",
        "LiveHiddenStateDatasetV1",
        "build_live_hidden_state_dataset_v1",
    ):
        assert name in lds.__all__
        assert hasattr(lds, name)


def test_w84_live_training_module_exports():
    from coordpy import live_trained_composed_memory_v1 as ltm
    for name in (
        "W84_LIVE_TRAINING_V1_SCHEMA_VERSION",
        "TrainingTraceWitnessV1",
        "train_composed_learned_memory_on_live_hidden_states",
        "train_synthetic_baseline_composed_memory",
        "LiveVsSyntheticHeadToHeadReportV1",
        "compare_live_trained_vs_synthetic_trained",
    ):
        assert name in ltm.__all__
        assert hasattr(ltm, name)


def test_w84_input_projection_deterministic_on_seed():
    from coordpy.live_hidden_state_dataset_v1 import (
        _input_projection, _target_projection,
    )
    a = _input_projection(
        vocab_size=100, input_dim=5, seed=42)
    b = _input_projection(
        vocab_size=100, input_dim=5, seed=42)
    import numpy as np
    assert np.array_equal(a, b)
    c = _input_projection(
        vocab_size=100, input_dim=5, seed=43)
    assert not np.array_equal(a, c)
    # target projection same property.
    d = _target_projection(
        hidden_dim=64, output_dim=3, seed=7)
    e = _target_projection(
        hidden_dim=64, output_dim=3, seed=7)
    assert np.array_equal(d, e)


def test_w84_training_witness_is_content_addressed():
    from coordpy.live_trained_composed_memory_v1 import (
        TrainingTraceWitnessV1,
        W84_LIVE_TRAINING_V1_SCHEMA_VERSION,
    )
    w = TrainingTraceWitnessV1(
        schema=W84_LIVE_TRAINING_V1_SCHEMA_VERSION,
        seed=1, n_iters=10, learning_rate=0.01,
        pre_module_cid="a" * 64,
        post_module_cid="b" * 64,
        pre_loss=1.0, post_loss=0.5,
        loss_curve_cid="c" * 64,
        dataset_cid="d" * 64,
        train_dataset_n_sequences=5)
    assert isinstance(w.cid(), str)
    assert len(w.cid()) == 64
    # Same inputs → same CID.
    w2 = TrainingTraceWitnessV1(**{
        k: getattr(w, k)
        for k in w.__dataclass_fields__})
    assert w.cid() == w2.cid()


@pytest.mark.skipif(
    not _HAS_HF,
    reason="transformers / torch not installed")
def test_w84_live_dataset_built_from_distilgpt2_is_content_addressed():
    """End-to-end check: build the dataset from distilgpt2,
    verify CID is deterministic + non-trivial."""
    from coordpy.transformers_runtime_v1 import (
        TransformersRuntimeV1,
    )
    from coordpy.live_hidden_state_dataset_v1 import (
        build_live_hidden_state_dataset_v1,
    )
    rt = TransformersRuntimeV1()
    prompts = [
        "the quick brown fox", "context zero"]
    d1 = build_live_hidden_state_dataset_v1(
        runtime=rt, prompts=prompts, layer_index=1,
        max_tokens=6)
    d2 = build_live_hidden_state_dataset_v1(
        runtime=rt, prompts=prompts, layer_index=1,
        max_tokens=6)
    assert d1.cid() == d2.cid()
    assert d1.n_sequences == 2
    assert len(d1.cid()) == 64
    # Different prompts → different CID.
    d3 = build_live_hidden_state_dataset_v1(
        runtime=rt, prompts=["a different prompt"],
        layer_index=1, max_tokens=6)
    assert d1.cid() != d3.cid()
    # Different layer → different CID.
    d4 = build_live_hidden_state_dataset_v1(
        runtime=rt, prompts=prompts, layer_index=2,
        max_tokens=6)
    assert d1.cid() != d4.cid()


@pytest.mark.skipif(
    not _HAS_HF,
    reason="transformers / torch not installed")
@pytest.mark.skipif(
    not os.environ.get(
        "COORDPY_RUN_LIVE_TRAINING_BENCH", ""),
    reason=(
        "set COORDPY_RUN_LIVE_TRAINING_BENCH=1 to run the "
        "live training head-to-head bench (~30s on CPU)"))
def test_w84_live_training_strictly_beats_synthetic():
    """The load-bearing P0 #26 test.

    Live-trained composed memory must strictly beat synthetic-
    trained composed memory on a held-out *live* evaluation
    set. Not a tie; not within noise.
    """
    from coordpy.transformers_runtime_v1 import (
        TransformersRuntimeV1,
    )
    from coordpy.live_hidden_state_dataset_v1 import (
        build_live_hidden_state_dataset_v1,
    )
    from coordpy.live_trained_composed_memory_v1 import (
        compare_live_trained_vs_synthetic_trained,
    )
    rt = TransformersRuntimeV1(
        model_name="distilbert/distilgpt2")
    train_prompts = [
        "The quick brown fox jumps over the lazy dog.",
        "Context Zero solves multi-agent coordination.",
        "Long-horizon retention requires substrate access.",
        "The W80 contract validates frontier-scale runtime.",
        "Composed learned memory outperforms ridge.",
        "Adversarial consensus repairs corrupted evidence.",
        "Replay-from-KV is byte-identical at fp32 on CPU.",
        "Bounded-window baselines fail on far-horizon recall.",
        "Differentiable memory enables addressed reads.",
        "Multi-host coordination requires mTLS auth.",
    ]
    eval_prompts = [
        "Frontier transformers carry KV cache structures.",
        "Hidden-state intercept moves the trace CID.",
        "Cross-runtime portability needs signatures.",
        "Audit chains record Merkle roots for verification.",
        "Distributed substrate spans cloud regions.",
    ]
    # Train and eval prompts must be disjoint.
    assert len(set(train_prompts) & set(eval_prompts)) == 0
    train_dataset = build_live_hidden_state_dataset_v1(
        runtime=rt, prompts=train_prompts,
        layer_index=2, max_tokens=10)
    eval_dataset = build_live_hidden_state_dataset_v1(
        runtime=rt, prompts=eval_prompts,
        layer_index=2, max_tokens=10)
    report = compare_live_trained_vs_synthetic_trained(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        n_iters=50)
    assert report.eval_train_disjoint
    assert report.live_strict_win, (
        f"live={report.live_trained_mse_on_live_eval} "
        f"synthetic={report.synthetic_trained_mse_on_live_eval}"
    )
    # The relative improvement must be material (NOT within
    # noise). >= 10% is the P0 #26 floor.
    assert float(report.relative_improvement) > 0.10
