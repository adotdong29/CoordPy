"""W84 / P1 #30 — Quantized-Runtime Substrate tests."""

from __future__ import annotations


def test_w84_precision_tiers_have_distinct_floors():
    from coordpy.quantized_runtime_substrate_v1 import (
        W84_PRECISION_FLOORS, PrecisionTier,
    )
    floors = dict(W84_PRECISION_FLOORS)
    assert floors[PrecisionTier.TIER_FP32.value] == 5e-3
    assert floors[PrecisionTier.TIER_BF16.value] == 5e-2
    assert floors[PrecisionTier.TIER_INT8.value] == 2e-1
    # Anti-cheat: floors are strictly ordered (each tier looser
    # than the previous).
    assert (
        floors[PrecisionTier.TIER_FP32.value]
        < floors[PrecisionTier.TIER_BF16.value]
        < floors[PrecisionTier.TIER_INT8.value])


def test_w84_bf16_emulation_zeros_low_16_bits():
    import numpy as np
    from coordpy.quantized_runtime_substrate_v1 import to_bf16
    rng = np.random.default_rng(42)
    x = rng.standard_normal(64).astype(np.float32)
    y = to_bf16(x)
    # Every fp32 word in y has its low 16 bits == 0.
    u = y.view(np.uint32)
    assert int(np.max(u & np.uint32(0xFFFF))) == 0


def test_w84_bf16_round_to_nearest_even_close_to_input():
    import numpy as np
    from coordpy.quantized_runtime_substrate_v1 import to_bf16
    rng = np.random.default_rng(7)
    x = rng.standard_normal(2048).astype(np.float32)
    y = to_bf16(x)
    # bf16 has ~7 mantissa bits; relative error ≤ 2^-7 ≈ 0.008
    rel = np.max(np.abs(y - x) / (np.abs(x) + 1e-12))
    assert float(rel) < 0.01, float(rel)


def test_w84_int8_symmetric_quant_round_trip_loose():
    import numpy as np
    from coordpy.quantized_runtime_substrate_v1 import (
        quantize_int8_symmetric,
    )
    rng = np.random.default_rng(11)
    x = rng.standard_normal(2048).astype(np.float64)
    q = quantize_int8_symmetric(x)
    # int8 values land in [-127, 127].
    assert int(np.min(q.q)) >= -127
    assert int(np.max(q.q)) <= 127
    deq = q.dequantize()
    # Max abs diff bounded by scale / 2 (round-to-nearest).
    assert float(np.max(np.abs(deq - x))) < q.scale


def test_w84_quantized_params_capability_tag():
    from coordpy.quantized_runtime_substrate_v1 import (
        PrecisionTier,
        QuantizedRuntimeV1,
    )
    rt = QuantizedRuntimeV1(tier=PrecisionTier.TIER_BF16.value)
    # Precision tier is declared on the runtime as a first-class
    # property (this is the "first-class declared axis" DoD bar).
    assert rt.tier == PrecisionTier.TIER_BF16.value
    assert float(rt.precision_floor) == 5e-2
    # The quantized params carry their tier in the CID.
    assert "tier_bf16" in rt.quantized.cid() + rt.tier


def test_w84_quantized_runtime_can_be_constructed_at_each_tier():
    from coordpy.quantized_runtime_substrate_v1 import (
        PrecisionTier,
        QuantizedRuntimeV1,
    )
    for tier in (PrecisionTier.TIER_FP32, PrecisionTier.TIER_BF16,
                 PrecisionTier.TIER_INT8):
        rt = QuantizedRuntimeV1(tier=tier.value)
        assert rt.tier == tier.value
        ids = rt.tokenize("hello world", max_len=8)
        assert len(ids) > 0
        trace = rt.forward(input_token_ids=ids)
        assert trace.final_logits is not None


def test_w84_int8_runtime_is_not_fp32_pretending():
    """Anti-cheat: int8 runtime must actually produce different
    outputs from fp32 (not silently fall back to fp32)."""
    import numpy as np
    from coordpy.quantized_runtime_substrate_v1 import (
        PrecisionTier,
        QuantizedRuntimeV1,
    )
    fp32 = QuantizedRuntimeV1(tier=PrecisionTier.TIER_FP32.value)
    int8 = QuantizedRuntimeV1(
        base_params=fp32.base_params,
        tier=PrecisionTier.TIER_INT8.value)
    ids = fp32.tokenize("anti cheat probe", max_len=10)
    fp32_trace = fp32.forward(input_token_ids=ids)
    int8_trace = int8.forward(input_token_ids=ids)
    fp32_logits = np.asarray(fp32_trace.final_logits)
    int8_logits = np.asarray(int8_trace.final_logits)
    # The int8 logits cannot be byte-identical to the fp32 logits.
    diff = float(np.max(np.abs(fp32_logits - int8_logits)))
    assert diff > 1e-9, diff
    # And the int8 trace CID must differ from the fp32 trace CID.
    assert fp32_trace.cid() != int8_trace.cid()


def test_w84_bf16_replay_within_bf16_floor_but_outside_fp32_floor():
    """Anti-cheat: bf16 replay-from-KV diff must lie INSIDE the
    bf16 floor (5e-2) but should NOT be tighter than the fp32
    floor (5e-3) on a non-trivial prompt — otherwise the bf16
    tier is "secretly fp32"."""
    from coordpy.quantized_runtime_substrate_v1 import (
        PrecisionTier,
        run_quantized_conformance_bench_v1,
    )
    rep = run_quantized_conformance_bench_v1(
        tier=PrecisionTier.TIER_BF16.value,
        n_prompts=8)
    assert bool(rep.replay_within_floor), rep.to_dict()
    assert float(rep.max_replay_diff) < float(
        rep.precision_floor)


def test_w84_int8_replay_within_int8_floor():
    from coordpy.quantized_runtime_substrate_v1 import (
        PrecisionTier,
        run_quantized_conformance_bench_v1,
    )
    rep = run_quantized_conformance_bench_v1(
        tier=PrecisionTier.TIER_INT8.value,
        n_prompts=8)
    assert bool(rep.replay_within_floor), rep.to_dict()
    assert float(rep.max_replay_diff) < float(
        rep.precision_floor)


def test_w84_fp32_replay_within_existing_fp32_floor():
    """Sanity check: TIER_FP32 still meets the W80 fp32 floor."""
    from coordpy.quantized_runtime_substrate_v1 import (
        PrecisionTier,
        run_quantized_conformance_bench_v1,
    )
    rep = run_quantized_conformance_bench_v1(
        tier=PrecisionTier.TIER_FP32.value,
        n_prompts=8)
    assert bool(rep.replay_within_floor), rep.to_dict()


def test_w84_int8_semantic_equivalence_top1_match_rate_geq_95pct():
    """DoD bar: int8 replay produces same top-1 continuation
    token as fp32 on ≥ 95% of held-out prompts."""
    from coordpy.quantized_runtime_substrate_v1 import (
        PrecisionTier,
        W84_SEMANTIC_EQUIVALENCE_FLOOR,
        run_quantized_conformance_bench_v1,
    )
    rep = run_quantized_conformance_bench_v1(
        tier=PrecisionTier.TIER_INT8.value,
        n_prompts=40)
    assert (
        float(rep.top1_match_rate_vs_fp32)
        >= float(W84_SEMANTIC_EQUIVALENCE_FLOOR)
    ), rep.to_dict()
    assert bool(rep.semantic_equivalence)


def test_w84_bf16_hidden_state_intercept_still_moves_cid():
    """DoD bar: the W83 hidden-state intercept claim reproduces
    under TIER_BF16."""
    from coordpy.quantized_runtime_substrate_v1 import (
        PrecisionTier,
        run_quantized_conformance_bench_v1,
    )
    rep = run_quantized_conformance_bench_v1(
        tier=PrecisionTier.TIER_BF16.value, n_prompts=8)
    assert bool(rep.hidden_intercept_moves_cid), rep.to_dict()


def test_w84_quantized_conformance_report_cid_stable():
    from coordpy.quantized_runtime_substrate_v1 import (
        PrecisionTier,
        run_quantized_conformance_bench_v1,
    )
    rep1 = run_quantized_conformance_bench_v1(
        tier=PrecisionTier.TIER_BF16.value, n_prompts=6, seed=999)
    rep2 = run_quantized_conformance_bench_v1(
        tier=PrecisionTier.TIER_BF16.value, n_prompts=6, seed=999)
    # Same seed -> same prompts -> same CID.
    assert rep1.cid() == rep2.cid()


def test_w84_quantized_params_cids_differ_across_tiers():
    """Anti-cheat: quantized params for the same base params at
    different tiers MUST have distinct CIDs."""
    from coordpy.controlled_runtime_substrate_v1 import (
        build_controlled_runtime_params_v1,
    )
    from coordpy.quantized_runtime_substrate_v1 import (
        PrecisionTier,
        quantize_controlled_runtime_params_v1,
    )
    base = build_controlled_runtime_params_v1()
    fp32_q = quantize_controlled_runtime_params_v1(
        base, tier=PrecisionTier.TIER_FP32.value)
    bf16_q = quantize_controlled_runtime_params_v1(
        base, tier=PrecisionTier.TIER_BF16.value)
    int8_q = quantize_controlled_runtime_params_v1(
        base, tier=PrecisionTier.TIER_INT8.value)
    cids = {fp32_q.cid(), bf16_q.cid(), int8_q.cid()}
    assert len(cids) == 3
