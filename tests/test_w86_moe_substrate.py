"""W86 #31 — MoE substrate surface tests.

Non-GPU surface tests for ``coordpy.moe_runtime_substrate_v1``.
The empirical bars (forward + replay + routing + intercept on
a real open-weight MoE) require a CUDA host with HF weights
and are exercised by ``scripts/run_w86_moe_substrate_closure.py``
on the Colab notebook ``scripts/colab_moe_substrate_closure_w86
.ipynb``; CI here covers the contract surface that is
verifiable on CPU without torch.
"""
from __future__ import annotations

import hashlib
import json


def test_w86_moe_module_imports():
    from coordpy.moe_runtime_substrate_v1 import (
        ExpertRoutingSnapshotV1,
        MoECapabilityProbeV1,
        MoEForceRoutingInjectionV1,
        MoEInstrumentationAxis,
        MoERuntimeAdapterV1,
        MoESubstrateClosureBenchReportV1,
        PerLayerRoutingV1,
        W86_MOE_AXES_ALL,
        W86_MOE_BLOCK_CANDIDATE_NAMES,
        W86_MOE_SUBSTRATE_V1_SCHEMA_VERSION,
        moe_declared_axes,
        probe_moe_capability_v1,
        run_moe_substrate_closure_bench_v1,
    )
    assert W86_MOE_SUBSTRATE_V1_SCHEMA_VERSION == (
        "coordpy.moe_runtime_substrate_v1.v1")


def test_w86_moe_three_axes_declared():
    """Load-bearing #31 DoD bullet: 3 new MoE-specific axes on
    the W80 contract."""
    from coordpy.moe_runtime_substrate_v1 import (
        W86_MOE_AXES_ALL,
        moe_declared_axes,
    )
    assert len(W86_MOE_AXES_ALL) == 3
    assert "read_expert_routing_per_layer" in W86_MOE_AXES_ALL
    assert (
        "write_force_expert_routing_per_layer"
        in W86_MOE_AXES_ALL)
    assert (
        "read_expert_output_per_expert_per_layer"
        in W86_MOE_AXES_ALL)
    axes = moe_declared_axes()
    assert set(axes.keys()) == set(W86_MOE_AXES_ALL)
    for a in axes.values():
        assert a in (
            "available", "backend_specific", "best_effort",
            "unavailable")


def test_w86_moe_lean_env_probe_honest():
    """Without torch the probe MUST honestly report
    `transformers_available = False` rather than mocking. The
    declared axes are downgraded to `unavailable`."""
    from coordpy.moe_runtime_substrate_v1 import (
        probe_moe_capability_v1,
    )
    p = probe_moe_capability_v1(
        model_name="allenai/OLMoE-1B-7B-0924-Instruct")
    assert p.transformers_available is False
    for axis, tag in p.declared_moe_axes:
        assert tag == "unavailable"
    assert isinstance(p.cid(), str)
    assert len(p.cid()) == 64


def test_w86_moe_block_candidates_include_major_families():
    """The MoE block class registry MUST cover the four major
    open-weight MoE families enumerated by the issue."""
    from coordpy.moe_runtime_substrate_v1 import (
        W86_MOE_BLOCK_CANDIDATE_NAMES,
    )
    # Mixtral
    assert any(
        "Mixtral" in n
        for n in W86_MOE_BLOCK_CANDIDATE_NAMES)
    # OLMoE
    assert any(
        "Olmoe" in n
        for n in W86_MOE_BLOCK_CANDIDATE_NAMES)
    # Qwen-MoE
    assert any(
        "Qwen" in n and "Moe" in n
        for n in W86_MOE_BLOCK_CANDIDATE_NAMES)
    # DeepSeek
    assert any(
        "Deepseek" in n
        for n in W86_MOE_BLOCK_CANDIDATE_NAMES)


def test_w86_moe_per_layer_routing_round_trip():
    """PerLayerRoutingV1.to_dict() + cid() must be
    deterministic and content-addressed."""
    import numpy as np
    from coordpy.moe_runtime_substrate_v1 import (
        PerLayerRoutingV1,
        W86_MOE_SUBSTRATE_V1_SCHEMA_VERSION,
    )
    ids = np.array([[0, 3], [1, 2], [3, 0]], dtype=np.int32)
    weights = np.array(
        [[0.6, 0.4], [0.7, 0.3], [0.5, 0.5]],
        dtype=np.float32)
    p1 = PerLayerRoutingV1(
        schema=W86_MOE_SUBSTRATE_V1_SCHEMA_VERSION,
        layer_index=2, n_experts=4, top_k=2, seq_len=3,
        expert_ids_cid=hashlib.sha256(
            ids.tobytes()).hexdigest(),
        gate_weights_cid=hashlib.sha256(
            weights.tobytes()).hexdigest(),
        expert_ids=ids, gate_weights=weights,
    )
    p2 = PerLayerRoutingV1(
        schema=W86_MOE_SUBSTRATE_V1_SCHEMA_VERSION,
        layer_index=2, n_experts=4, top_k=2, seq_len=3,
        expert_ids_cid=p1.expert_ids_cid,
        gate_weights_cid=p1.gate_weights_cid,
        expert_ids=ids, gate_weights=weights,
    )
    assert p1.cid() == p2.cid()
    # Different routing → different cid.
    weights2 = np.array(
        [[0.5, 0.5], [0.7, 0.3], [0.5, 0.5]],
        dtype=np.float32)
    p3 = PerLayerRoutingV1(
        schema=W86_MOE_SUBSTRATE_V1_SCHEMA_VERSION,
        layer_index=2, n_experts=4, top_k=2, seq_len=3,
        expert_ids_cid=p1.expert_ids_cid,
        gate_weights_cid=hashlib.sha256(
            weights2.tobytes()).hexdigest(),
        expert_ids=ids, gate_weights=weights2,
    )
    assert p1.cid() != p3.cid()


def test_w86_moe_expert_routing_snapshot_aggregates():
    """ExpertRoutingSnapshotV1 must aggregate per-layer CIDs
    into one snapshot CID."""
    import numpy as np
    from coordpy.moe_runtime_substrate_v1 import (
        ExpertRoutingSnapshotV1,
        PerLayerRoutingV1,
        W86_MOE_SUBSTRATE_V1_SCHEMA_VERSION,
    )
    layers = tuple(
        PerLayerRoutingV1(
            schema=W86_MOE_SUBSTRATE_V1_SCHEMA_VERSION,
            layer_index=i, n_experts=4, top_k=2, seq_len=3,
            expert_ids_cid=hashlib.sha256(
                f"ids_{i}".encode()).hexdigest(),
            gate_weights_cid=hashlib.sha256(
                f"w_{i}".encode()).hexdigest(),
            expert_ids=None, gate_weights=None,
        )
        for i in range(4))
    snap = ExpertRoutingSnapshotV1(
        schema=W86_MOE_SUBSTRATE_V1_SCHEMA_VERSION,
        model_id="fake/moe",
        n_layers=4, n_layers_with_routing=4, seq_len=3,
        per_layer=layers,
    )
    assert len(snap.cid()) == 64
    snap2 = ExpertRoutingSnapshotV1(
        schema=W86_MOE_SUBSTRATE_V1_SCHEMA_VERSION,
        model_id="fake/moe",
        n_layers=4, n_layers_with_routing=4, seq_len=3,
        per_layer=layers,
    )
    assert snap.cid() == snap2.cid()


def test_w86_moe_runtime_adapter_refuses_non_moe_model_lean():
    """Without torch, MoERuntimeAdapterV1 construction must
    fail clearly rather than fake it."""
    import pytest
    from coordpy.moe_runtime_substrate_v1 import (
        MoERuntimeAdapterV1,
    )
    with pytest.raises(Exception):
        MoERuntimeAdapterV1(
            model_name="fake/non-moe",
            device="cpu",
            precision_tier="tier_fp32")
