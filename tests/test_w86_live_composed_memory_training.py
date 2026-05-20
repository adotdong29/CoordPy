"""W86 — live composed-learned-memory training surface tests.

Validates the *non-torch* surface of the W86 #26 closure module:
projection determinism, content-addressing, capsule disjointness,
report schema. The end-to-end live training run lives in the
Vertex AI execution and is recorded in
``results/w86/.../26_live_learned_memory.json``.
"""
from __future__ import annotations

import numpy as np
import pytest


def test_w86_projection_deterministic_from_seed():
    from coordpy.live_composed_memory_training_v1 import (
        build_hidden_state_projection_v1,
    )
    p1, m1 = build_hidden_state_projection_v1(
        in_dim=4096, out_dim=8, seed=42)
    p2, m2 = build_hidden_state_projection_v1(
        in_dim=4096, out_dim=8, seed=42)
    assert p1.cid() == p2.cid()
    assert np.array_equal(m1, m2)


def test_w86_projection_changes_with_seed():
    from coordpy.live_composed_memory_training_v1 import (
        build_hidden_state_projection_v1,
    )
    p1, _ = build_hidden_state_projection_v1(
        in_dim=4096, out_dim=8, seed=42)
    p2, _ = build_hidden_state_projection_v1(
        in_dim=4096, out_dim=8, seed=43)
    assert p1.cid() != p2.cid()


def test_w86_projection_columns_unit_norm():
    from coordpy.live_composed_memory_training_v1 import (
        build_hidden_state_projection_v1,
    )
    _, P = build_hidden_state_projection_v1(
        in_dim=4096, out_dim=8, seed=42)
    norms = np.linalg.norm(P, axis=0)
    assert np.allclose(norms, 1.0, atol=1e-10)


def test_w86_projection_in_dim_out_dim_round_trip():
    from coordpy.live_composed_memory_training_v1 import (
        build_hidden_state_projection_v1,
    )
    p, P = build_hidden_state_projection_v1(
        in_dim=4096, out_dim=8, seed=42)
    assert p.in_dim == 4096
    assert p.out_dim == 8
    assert P.shape == (4096, 8)


def test_w86_capsule_disjointness_enforced():
    """The W84 LiveHiddenStateDatasetCapsuleV1.__post_init__
    must refuse a capsule with overlapping train and eval
    prompt CIDs. This is the anti-cheat for the W86 live
    training: same prompt cannot appear in both splits."""
    from coordpy.live_hidden_state_dataset_v1 import (
        build_live_hidden_state_dataset_v1,
    )
    overlap_prompt = "this prompt is in both splits"
    with pytest.raises(ValueError, match="overlap"):
        build_live_hidden_state_dataset_v1(
            prompts_train=[overlap_prompt, "train only"],
            prompts_eval=[overlap_prompt, "eval only"],
            model_name="fake/model",
            layer_index=4,
            precision_tier="tier_fp32",
        )


def test_w86_training_blocked_on_hardware_when_no_torch(
        monkeypatch):
    """Without torch/transformers the live training entry point
    must raise LiveTrainingBlockedOnHardwareError honestly, not
    silently fall back to synthetic data."""
    from coordpy.live_composed_memory_training_v1 import (
        train_composed_learned_memory_on_live_hidden_states_v1,
    )
    from coordpy.live_hidden_state_dataset_v1 import (
        LiveTrainingBlockedOnHardwareError,
    )
    with pytest.raises(LiveTrainingBlockedOnHardwareError):
        train_composed_learned_memory_on_live_hidden_states_v1(
            prompts_train=[
                "train prompt one",
                "train prompt two",
            ],
            prompts_eval=[
                "eval prompt one",
            ],
            model_name="fake/model",
            device="cpu",
            precision_tier="tier_fp32",
        )


def test_w86_training_schema_versions_present():
    from coordpy.live_composed_memory_training_v1 import (
        W86_LIVE_CM_TRAIN_V1_SCHEMA_VERSION,
        W86_LIVE_CM_DEFAULT_LAYER,
        W86_LIVE_CM_DEFAULT_PROJECTION_DIM,
    )
    assert W86_LIVE_CM_TRAIN_V1_SCHEMA_VERSION == (
        "coordpy.live_composed_memory_training_v1.v1")
    assert int(W86_LIVE_CM_DEFAULT_LAYER) == 12
    assert int(W86_LIVE_CM_DEFAULT_PROJECTION_DIM) == 8
