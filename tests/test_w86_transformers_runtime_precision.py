"""W86 — precision-tier extensions to TransformersRuntimeV1 tests.

Validates the W86 additions (precision_tier, skinny_trace,
W86_REPLAY_TOLERANCE_PER_TIER) without requiring a real torch /
transformers install. The live frontier-scale tests that exercise
these on a real Llama-3.1-8B-Instruct in bf16 on a CUDA GPU live
in the Vertex AI execution and are recorded in
``results/w86/frontier_closure_report.json``; this file only
covers the contract surface that is verifiable on CPU CI.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest


def test_w86_precision_tier_constants_present():
    import coordpy.transformers_runtime_v1 as tr
    assert tr.W86_PRECISION_TIER_FP32 == "tier_fp32"
    assert tr.W86_PRECISION_TIER_BF16 == "tier_bf16"
    assert tr.W86_PRECISION_TIER_FP16 == "tier_fp16"
    assert tr.W86_PRECISION_TIER_INT8 == "tier_int8"


def test_w86_replay_tolerance_per_tier_honest_widening():
    import coordpy.transformers_runtime_v1 as tr
    tol = tr.W86_REPLAY_TOLERANCE_PER_TIER
    # fp32 floor preserved (the W80 byte-identity bar).
    assert tol[tr.W86_PRECISION_TIER_FP32] == 0.005
    # bf16/fp16/int8 floors widen monotonically.
    assert tol[tr.W86_PRECISION_TIER_BF16] > tol[
        tr.W86_PRECISION_TIER_FP32]
    assert tol[tr.W86_PRECISION_TIER_FP16] > tol[
        tr.W86_PRECISION_TIER_FP32]
    assert tol[tr.W86_PRECISION_TIER_INT8] > tol[
        tr.W86_PRECISION_TIER_BF16]


def test_w86_runtime_rejects_unknown_precision_tier(monkeypatch):
    import coordpy.transformers_runtime_v1 as tr

    class _FakeTorch:
        float32 = "float32"
        bfloat16 = "bfloat16"
        float16 = "float16"

        @staticmethod
        def set_grad_enabled(_enabled: bool) -> None:
            return None

    class _FakeAutoModelForCausalLM:
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            return SimpleNamespace(
                config=SimpleNamespace(
                    num_hidden_layers=6,
                    num_attention_heads=12,
                    hidden_size=768,
                    model_type="gpt2",
                ),
                eval=lambda: None,
                named_parameters=lambda: [],
            )

    class _FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, model_name):
            return object()

    monkeypatch.setattr(
        tr, "_torch_modules",
        lambda: (_FakeTorch(), _FakeAutoModelForCausalLM,
                 _FakeAutoTokenizer))

    with pytest.raises(
            ValueError, match="unknown precision_tier"):
        tr.TransformersRuntimeV1(
            model_name="fake/model",
            precision_tier="tier_bogus")


def test_w86_runtime_accepts_bf16_tier(monkeypatch):
    import coordpy.transformers_runtime_v1 as tr

    captured: dict[str, object] = {}

    class _FakeTorch:
        float32 = "float32"
        bfloat16 = "bfloat16"
        float16 = "float16"

        @staticmethod
        def set_grad_enabled(_enabled: bool) -> None:
            return None

    class _FakeAutoModelForCausalLM:
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            captured["kwargs"] = dict(kwargs)
            return SimpleNamespace(
                config=SimpleNamespace(
                    num_hidden_layers=6,
                    num_attention_heads=12,
                    hidden_size=768,
                    model_type="gpt2",
                ),
                eval=lambda: None,
                named_parameters=lambda: [],
            )

    class _FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, model_name):
            return object()

    monkeypatch.setattr(
        tr, "_torch_modules",
        lambda: (_FakeTorch(), _FakeAutoModelForCausalLM,
                 _FakeAutoTokenizer))

    rt = tr.TransformersRuntimeV1(
        model_name="fake/model",
        precision_tier=tr.W86_PRECISION_TIER_BF16,
    )
    assert rt.precision_tier == tr.W86_PRECISION_TIER_BF16
    assert captured["kwargs"]["torch_dtype"] == "bfloat16"


def test_w86_runtime_cuda_device_uses_device_map(monkeypatch):
    import coordpy.transformers_runtime_v1 as tr

    captured: dict[str, object] = {}

    class _FakeTorch:
        float32 = "float32"
        bfloat16 = "bfloat16"
        float16 = "float16"

        @staticmethod
        def set_grad_enabled(_enabled: bool) -> None:
            return None

    class _FakeAutoModelForCausalLM:
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            captured["kwargs"] = dict(kwargs)
            return SimpleNamespace(
                config=SimpleNamespace(
                    num_hidden_layers=6,
                    num_attention_heads=12,
                    hidden_size=768,
                    model_type="gpt2",
                ),
                eval=lambda: None,
                named_parameters=lambda: [],
            )

    class _FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, model_name):
            return object()

    monkeypatch.setattr(
        tr, "_torch_modules",
        lambda: (_FakeTorch(), _FakeAutoModelForCausalLM,
                 _FakeAutoTokenizer))

    tr.TransformersRuntimeV1(
        model_name="fake/model",
        device="cuda:0",
        precision_tier=tr.W86_PRECISION_TIER_BF16,
    )
    kw = captured["kwargs"]
    assert kw["device_map"] == "cuda:0"
    assert kw["torch_dtype"] == "bfloat16"


def test_w86_runtime_skinny_trace_flag_default_false(monkeypatch):
    import coordpy.transformers_runtime_v1 as tr

    class _FakeTorch:
        float32 = "float32"

        @staticmethod
        def set_grad_enabled(_enabled: bool) -> None:
            return None

    class _FakeAutoModelForCausalLM:
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            return SimpleNamespace(
                config=SimpleNamespace(
                    num_hidden_layers=6,
                    num_attention_heads=12,
                    hidden_size=768,
                    model_type="gpt2",
                ),
                eval=lambda: None,
                named_parameters=lambda: [],
            )

    class _FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, model_name):
            return object()

    monkeypatch.setattr(
        tr, "_torch_modules",
        lambda: (_FakeTorch(), _FakeAutoModelForCausalLM,
                 _FakeAutoTokenizer))

    rt = tr.TransformersRuntimeV1(model_name="fake/model")
    assert rt.skinny_trace is False

    rt_skinny = tr.TransformersRuntimeV1(
        model_name="fake/model", skinny_trace=True)
    assert rt_skinny.skinny_trace is True


def test_w86_runtime_skinny_trace_uses_sdpa_attention(monkeypatch):
    """Skinny trace mode must pick attn_implementation=sdpa so a
    32 k-token forward fits in 24-40 GB of VRAM. Eager attention
    materialises the full (seq_len, seq_len) matrix and OOMs at
    long context. Caught by the first Colab Pro run."""
    import coordpy.transformers_runtime_v1 as tr

    captured: dict[str, object] = {}

    class _FakeTorch:
        float32 = "float32"
        bfloat16 = "bfloat16"
        float16 = "float16"

        @staticmethod
        def set_grad_enabled(_enabled: bool) -> None:
            return None

    class _FakeAutoModelForCausalLM:
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            captured["kwargs"] = dict(kwargs)
            return SimpleNamespace(
                config=SimpleNamespace(
                    num_hidden_layers=6,
                    num_attention_heads=12,
                    hidden_size=768,
                    model_type="gpt2",
                ),
                eval=lambda: None,
                named_parameters=lambda: [],
            )

    class _FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, model_name):
            return object()

    monkeypatch.setattr(
        tr, "_torch_modules",
        lambda: (_FakeTorch(), _FakeAutoModelForCausalLM,
                 _FakeAutoTokenizer))

    tr.TransformersRuntimeV1(
        model_name="fake/model",
        precision_tier=tr.W86_PRECISION_TIER_BF16,
        skinny_trace=True,
    )
    kw = captured["kwargs"]
    assert kw["attn_implementation"] == "sdpa", (
        "skinny trace must use sdpa to fit long context")
    assert kw["output_attentions"] is False, (
        "skinny trace must not request per-layer attention "
        "probabilities (they materialise the full attention "
        "matrix)")
    assert kw["output_hidden_states"] is False, (
        "skinny trace must not request per-layer hidden states "
        "(they retain full-sequence arrays internally)")


def test_w86_runtime_full_trace_keeps_eager_attention(
        monkeypatch):
    """The default (skinny_trace=False) must keep eager
    attention so the W80 conformance suite's READ_ATTENTION_PROBS
    axis still passes. Regression on the existing W80 contract."""
    import coordpy.transformers_runtime_v1 as tr

    captured: dict[str, object] = {}

    class _FakeTorch:
        float32 = "float32"
        bfloat16 = "bfloat16"
        float16 = "float16"

        @staticmethod
        def set_grad_enabled(_enabled: bool) -> None:
            return None

    class _FakeAutoModelForCausalLM:
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            captured["kwargs"] = dict(kwargs)
            return SimpleNamespace(
                config=SimpleNamespace(
                    num_hidden_layers=6,
                    num_attention_heads=12,
                    hidden_size=768,
                    model_type="gpt2",
                ),
                eval=lambda: None,
                named_parameters=lambda: [],
            )

    class _FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, model_name):
            return object()

    monkeypatch.setattr(
        tr, "_torch_modules",
        lambda: (_FakeTorch(), _FakeAutoModelForCausalLM,
                 _FakeAutoTokenizer))

    tr.TransformersRuntimeV1(model_name="fake/model")
    kw = captured["kwargs"]
    assert kw["attn_implementation"] == "eager"
    assert kw["output_attentions"] is True
    assert kw["output_hidden_states"] is True
