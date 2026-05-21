"""W87 / P3 #46 — Multi-Modal Substrate V1 tests.

Covers:

  * `MultiModalPayloadV1` contract: content-addressing, modality
    validation, embedding/encoder consistency.
  * AST-aware code substrate: function-def extraction across
    nested functions, classes, async, decorators.
  * Vision substrate stub: encoder fingerprint, payload identity.
  * Composed multi-modal pipeline: cross-modality Merkle root,
    independent re-verification.
  * Per-modality precision floor reporting.
  * Anti-cheat enforcement: stub fallback is honestly labelled;
    `require_real_vlm=True` raises BlockedOnHardwareError when
    transformers absent.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from coordpy.code_substrate_v1 import (
    W87_CODE_STUB_EMBEDDING_DIM,
    ASTFunctionBoundaryV1,
    CodeSubstrateAdapterV1,
    encode_source_with_stub_v1,
    extract_function_boundaries_v1,
)
from coordpy.composed_multimodal_pipeline_v1 import (
    W87_COMPOSED_MULTI_MODAL_PIPELINE_V1_SCHEMA_VERSION as W87CV,
    MultiModalAgentTurnV1,
    run_composed_multi_modal_pipeline_v1,
    verify_multi_modal_run_report_v1,
)
from coordpy.multi_modal_payload_v1 import (
    W87_MODALITIES_V1,
    W87_MODALITY_PRECISION_FLOOR_DEFAULTS_FP32,
    BlockedOnHardwareError,
    EncoderFingerprintV1,
    Modality,
    build_cross_modality_merkle_root_v1,
    build_multi_modal_payload_v1,
    measure_modality_precision_floor_fp32_v1,
)
from coordpy.vision_substrate_v1 import (
    W87_VISION_STUB_EMBEDDING_DIM,
    VisionSubstrateAdapterV1,
    make_tiny_png_v1,
    probe_vision_substrate_capability_v1,
)


# ---------------------------------------------------------------
# MultiModalPayloadV1
# ---------------------------------------------------------------

def _enc(modality: str, dim: int = 8) -> EncoderFingerprintV1:
    return EncoderFingerprintV1(
        schema="v1", modality=modality,
        encoder_kind="hf_causal_lm",
        model_name="test", model_revision="main",
        precision_tier="tier_fp32", embedding_dim=int(dim))


def test_w87_modalities_v1_set() -> None:
    assert W87_MODALITIES_V1 == ("text", "image", "code")


def test_w87_multi_modal_payload_cid_includes_raw_bytes() -> None:
    e = np.zeros((3, 8), dtype=np.float32)
    p1 = build_multi_modal_payload_v1(
        modality="text", raw_bytes=b"hello",
        embedding=e, encoder=_enc("text"))
    p2 = build_multi_modal_payload_v1(
        modality="text", raw_bytes=b"hello world",
        embedding=e, encoder=_enc("text"))
    assert p1.payload_cid() != p2.payload_cid()


def test_w87_multi_modal_payload_cid_includes_embedding() -> None:
    e1 = np.zeros((3, 8), dtype=np.float32)
    e2 = np.ones((3, 8), dtype=np.float32)
    p1 = build_multi_modal_payload_v1(
        modality="text", raw_bytes=b"hello",
        embedding=e1, encoder=_enc("text"))
    p2 = build_multi_modal_payload_v1(
        modality="text", raw_bytes=b"hello",
        embedding=e2, encoder=_enc("text"))
    assert p1.payload_cid() != p2.payload_cid()


def test_w87_multi_modal_payload_cid_includes_encoder() -> None:
    e = np.zeros((3, 8), dtype=np.float32)
    p1 = build_multi_modal_payload_v1(
        modality="text", raw_bytes=b"hello", embedding=e,
        encoder=_enc("text"))
    p2 = build_multi_modal_payload_v1(
        modality="text", raw_bytes=b"hello", embedding=e,
        encoder=EncoderFingerprintV1(
            schema="v1", modality="text", encoder_kind="other",
            model_name="other", model_revision="main",
            precision_tier="tier_bf16", embedding_dim=8))
    assert p1.payload_cid() != p2.payload_cid()


def test_w87_multi_modal_payload_validates_modality() -> None:
    with pytest.raises(ValueError, match="unknown modality"):
        build_multi_modal_payload_v1(
            modality="audio", raw_bytes=b"x")


def test_w87_multi_modal_payload_no_embedding_no_encoder() -> None:
    p = build_multi_modal_payload_v1(
        modality="text", raw_bytes=b"deferred encode")
    assert p.embedding is None
    assert p.embedding_cid == "none"
    assert p.encoder is None


def test_w87_multi_modal_payload_embedding_requires_encoder() -> None:
    e = np.zeros((3, 8), dtype=np.float32)
    with pytest.raises(
            ValueError, match="every embedding must identify"):
        build_multi_modal_payload_v1(
            modality="text", raw_bytes=b"x", embedding=e)


# ---------------------------------------------------------------
# Cross-modality Merkle root
# ---------------------------------------------------------------

def test_w87_cross_modality_merkle_root_spans_all_payloads() -> None:
    e = np.zeros((3, 8), dtype=np.float32)
    ptext = build_multi_modal_payload_v1(
        modality="text", raw_bytes=b"text",
        embedding=e, encoder=_enc("text"))
    pcode = build_multi_modal_payload_v1(
        modality="code", raw_bytes=b"def f(): pass",
        embedding=e, encoder=_enc("code"))
    pimg = build_multi_modal_payload_v1(
        modality="image", raw_bytes=b"\x89PNG\r\n\x1a\n",
        embedding=e, encoder=_enc("image"))
    root = build_cross_modality_merkle_root_v1(
        [ptext, pcode, pimg])
    assert len({m for m, _ in root.per_modality_payload_cids}) == 3
    # Changing any payload changes the root.
    pcode2 = build_multi_modal_payload_v1(
        modality="code", raw_bytes=b"def f(): pass  # changed",
        embedding=e, encoder=_enc("code"))
    root2 = build_cross_modality_merkle_root_v1(
        [ptext, pcode2, pimg])
    assert root.root_cid != root2.root_cid


def test_w87_cross_modality_root_order_invariant() -> None:
    e = np.zeros((3, 8), dtype=np.float32)
    ptext = build_multi_modal_payload_v1(
        modality="text", raw_bytes=b"text",
        embedding=e, encoder=_enc("text"))
    pcode = build_multi_modal_payload_v1(
        modality="code", raw_bytes=b"def f(): pass",
        embedding=e, encoder=_enc("code"))
    r1 = build_cross_modality_merkle_root_v1([ptext, pcode])
    r2 = build_cross_modality_merkle_root_v1([pcode, ptext])
    # Order-invariant: canonical sort by (modality, payload_cid).
    assert r1.root_cid == r2.root_cid


# ---------------------------------------------------------------
# Code substrate (AST-aware)
# ---------------------------------------------------------------

def test_w87_ast_function_boundaries_basic() -> None:
    src = ("def f(x):\n    return x + 1\n"
           "def g(y, z):\n    return y * z\n")
    bds = extract_function_boundaries_v1(src)
    assert len(bds) == 2
    assert bds[0].name == "f"
    assert bds[0].n_args == 1
    assert bds[1].name == "g"
    assert bds[1].n_args == 2


def test_w87_ast_function_boundaries_nested_and_async() -> None:
    src = (
        "class Outer:\n"
        "    def method(self):\n"
        "        def helper():\n"
        "            return 1\n"
        "        return helper()\n"
        "    async def amethod(self):\n"
        "        return 2\n"
    )
    bds = extract_function_boundaries_v1(src)
    names = {b.qualname for b in bds}
    assert "Outer.method" in names
    assert "Outer.amethod" in names
    assert any("Outer.method.helper" in n for n in names)
    assert any(b.is_async for b in bds)


def test_w87_code_substrate_payload_carries_ast_extras() -> None:
    src = "def hello():\n    return 'hi'\n"
    adapter = CodeSubstrateAdapterV1()
    p = adapter.encode(src)
    assert p.modality == "code"
    assert "ast_function_boundaries" in p.extras
    assert p.extras["n_functions"] == 1
    assert p.extras["ast_function_boundaries"][0]["name"] == "hello"


def test_w87_code_substrate_per_function_cids_differ() -> None:
    src = ("def a():\n    return 1\n"
           "def b():\n    return 2\n")
    adapter = CodeSubstrateAdapterV1()
    r = adapter.read_at_function_boundaries(src)
    assert len(r.per_function_embedding_cids) == 2
    assert r.per_function_embedding_cids[0] != (
        r.per_function_embedding_cids[1])


def test_w87_code_substrate_stub_encoder_clearly_named() -> None:
    """Anti-cheat: stub fallback MUST be clearly labelled."""
    adapter = CodeSubstrateAdapterV1()
    fp = adapter.encoder_fingerprint()
    assert fp.encoder_kind == "stub_sha256_v1"
    assert fp.model_name == "stub"


def test_w87_code_substrate_require_real_model_raises_without_torch() -> None:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        # torch + transformers installed; skip — the negative path
        # is only meaningful when transformers is absent.
        pytest.skip("torch/transformers installed; negative path NA")
    except ImportError:
        pass
    with pytest.raises(BlockedOnHardwareError):
        CodeSubstrateAdapterV1(
            model_name="distilgpt2",
            require_real_model=True)


def test_w87_code_substrate_deterministic_across_runs() -> None:
    src = "def stable():\n    return 'always the same'\n"
    a1 = CodeSubstrateAdapterV1().encode(src)
    a2 = CodeSubstrateAdapterV1().encode(src)
    assert a1.payload_cid() == a2.payload_cid()


# ---------------------------------------------------------------
# Vision substrate
# ---------------------------------------------------------------

def test_w87_vision_capability_probe_returns_structured_state() -> None:
    cap = probe_vision_substrate_capability_v1()
    assert hasattr(cap, "torch_available")
    assert hasattr(cap, "transformers_available")
    assert hasattr(cap, "pil_available")
    assert hasattr(cap, "can_load_vlm")
    # All bools.
    assert isinstance(cap.can_load_vlm, bool)


def test_w87_vision_substrate_stub_payload_identity() -> None:
    adapter = VisionSubstrateAdapterV1()
    png_a = make_tiny_png_v1(seed=10)
    png_b = make_tiny_png_v1(seed=11)
    pa = adapter.encode_image(png_a)
    pb = adapter.encode_image(png_b)
    assert pa.modality == "image"
    assert pa.payload_cid() != pb.payload_cid()


def test_w87_vision_substrate_stub_encoder_clearly_named() -> None:
    adapter = VisionSubstrateAdapterV1()
    fp = adapter.encoder_fingerprint()
    assert fp.encoder_kind == "stub_clip_style_v1"
    assert fp.model_name == "stub"


def test_w87_vision_substrate_require_real_vlm_raises_without_torch() -> None:
    cap = probe_vision_substrate_capability_v1()
    if cap.can_load_vlm:
        pytest.skip(
            "VLM can load on this host; negative path NA")
    with pytest.raises(BlockedOnHardwareError):
        VisionSubstrateAdapterV1(
            model_name="HuggingFaceTB/SmolVLM-256M-Instruct",
            require_real_vlm=True)


def test_w87_vision_substrate_stub_deterministic() -> None:
    png = make_tiny_png_v1(seed=42)
    e1 = VisionSubstrateAdapterV1().encode_image(png)
    e2 = VisionSubstrateAdapterV1().encode_image(png)
    assert e1.payload_cid() == e2.payload_cid()


def test_w87_vision_substrate_allow_stub_false_blocks() -> None:
    cap = probe_vision_substrate_capability_v1()
    if cap.can_load_vlm:
        pytest.skip(
            "VLM can load on this host; negative path NA")
    adapter = VisionSubstrateAdapterV1(allow_stub=False)
    with pytest.raises(BlockedOnHardwareError):
        adapter.encode_image(make_tiny_png_v1())


# ---------------------------------------------------------------
# Composed pipeline
# ---------------------------------------------------------------

def _three_modality_turns() -> list[MultiModalAgentTurnV1]:
    e = np.random.RandomState(7).randn(4, 8).astype(np.float32)
    t = build_multi_modal_payload_v1(
        modality="text", raw_bytes=b"prompt",
        embedding=e, encoder=_enc("text"))
    code_adapter = CodeSubstrateAdapterV1()
    c = code_adapter.encode("def f():\n    return 1\n")
    vision_adapter = VisionSubstrateAdapterV1()
    i = vision_adapter.encode_image(make_tiny_png_v1(seed=3))
    return [
        MultiModalAgentTurnV1(
            schema=W87CV, agent_id="reader", role="reader",
            payload=t),
        MultiModalAgentTurnV1(
            schema=W87CV, agent_id="vision", role="critic",
            payload=i),
        MultiModalAgentTurnV1(
            schema=W87CV, agent_id="coder", role="implementer",
            payload=c),
    ]


def test_w87_composed_pipeline_spans_three_modalities() -> None:
    turns = _three_modality_turns()
    rep = run_composed_multi_modal_pipeline_v1(
        run_label="three_mode", agent_turns=turns)
    assert rep.n_modalities == 3
    assert len(rep.agent_turns) == 3


def test_w87_composed_pipeline_independently_reverifies() -> None:
    turns = _three_modality_turns()
    rep = run_composed_multi_modal_pipeline_v1(
        run_label="rev", agent_turns=turns)
    ok, detail = verify_multi_modal_run_report_v1(rep)
    assert ok, f"verify failed: {detail}"


def test_w87_composed_pipeline_two_modalities_minimum() -> None:
    """DoD: ≥ 2 modalities."""
    e = np.zeros((3, 8), dtype=np.float32)
    p1 = build_multi_modal_payload_v1(
        modality="text", raw_bytes=b"x",
        embedding=e, encoder=_enc("text"))
    p2 = build_multi_modal_payload_v1(
        modality="code", raw_bytes=b"def y(): pass",
        embedding=e, encoder=_enc("code"))
    turns = [
        MultiModalAgentTurnV1(
            schema=W87CV, agent_id="a", role="r1", payload=p1),
        MultiModalAgentTurnV1(
            schema=W87CV, agent_id="b", role="r2", payload=p2),
    ]
    rep = run_composed_multi_modal_pipeline_v1(
        run_label="two_mode", agent_turns=turns)
    assert rep.n_modalities == 2


def test_w87_composed_pipeline_report_cid_changes_on_payload_change() -> None:
    turns = _three_modality_turns()
    rep1 = run_composed_multi_modal_pipeline_v1(
        run_label="x", agent_turns=turns)
    # Mutate one payload by replacing the code with different source.
    turns2 = list(turns)
    new_code = CodeSubstrateAdapterV1().encode(
        "def different():\n    return 99\n")
    turns2[2] = MultiModalAgentTurnV1(
        schema=W87CV, agent_id="coder", role="implementer",
        payload=new_code)
    rep2 = run_composed_multi_modal_pipeline_v1(
        run_label="x", agent_turns=turns2)
    assert rep1.report_cid != rep2.report_cid
    assert (rep1.cross_modality_root.root_cid !=
            rep2.cross_modality_root.root_cid)


# ---------------------------------------------------------------
# Per-modality precision floor
# ---------------------------------------------------------------

def test_w87_precision_floor_identical_arrays() -> None:
    a = np.array([[1.0, 2.0]], dtype=np.float32)
    f = measure_modality_precision_floor_fp32_v1(
        modality="text", encoder_cid="enc",
        embedding_a=a, embedding_b=a)
    assert f.floor_fp32 == 0.0
    assert f.floor_within_tolerance_fp32 is True


def test_w87_precision_floor_within_tolerance() -> None:
    a = np.array([[1.0, 2.0]], dtype=np.float32)
    b = np.array([[1.001, 2.0]], dtype=np.float32)
    f = measure_modality_precision_floor_fp32_v1(
        modality="image", encoder_cid="enc",
        embedding_a=a, embedding_b=b)
    assert 0.0 < f.floor_fp32 < 0.01
    assert f.floor_within_tolerance_fp32 is True


def test_w87_precision_floor_outside_tolerance() -> None:
    a = np.array([[1.0, 2.0]], dtype=np.float32)
    b = np.array([[1.5, 2.0]], dtype=np.float32)
    f = measure_modality_precision_floor_fp32_v1(
        modality="text", encoder_cid="enc",
        embedding_a=a, embedding_b=b, tolerance=1e-4)
    assert f.floor_within_tolerance_fp32 is False


def test_w87_precision_floor_per_modality_tolerance_set() -> None:
    """All 3 modalities must have a default tolerance defined."""
    for m in ("text", "image", "code"):
        assert m in W87_MODALITY_PRECISION_FLOOR_DEFAULTS_FP32
        assert (W87_MODALITY_PRECISION_FLOOR_DEFAULTS_FP32[m] > 0)


# ---------------------------------------------------------------
# Real-model paths (skip-gated on transformers presence)
# ---------------------------------------------------------------

def _transformers_present() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(
    not _transformers_present(),
    reason="W87 real-model code path requires transformers")
def test_w87_code_substrate_real_distilgpt2_reads_hidden_state(
) -> None:
    """When transformers is present, the code adapter MUST be
    able to load a real causal LM and surface real hidden state."""
    adapter = CodeSubstrateAdapterV1(
        model_name="distilgpt2", require_real_model=True)
    p = adapter.encode("def hello():\n    return 'hi'\n")
    # The encoder fingerprint must be a real model, not the stub.
    assert p.encoder.encoder_kind == "hf_causal_lm"
    assert p.encoder.model_name == "distilgpt2"
    # The embedding must have non-trivial values (not all zeros).
    assert np.any(p.embedding != 0.0)
    # AST extras must be present.
    assert p.extras["n_functions"] == 1


@pytest.mark.skipif(
    not _transformers_present(),
    reason="W87 real-model vision path requires transformers")
def test_w87_vision_substrate_real_moondream_reads_hidden_state(
) -> None:
    """When transformers is present, the vision adapter MUST be
    able to load a real VLM and surface real hidden state from
    its vision tower."""
    cap = probe_vision_substrate_capability_v1()
    if not cap.can_load_vlm:
        pytest.skip(f"can_load_vlm=False: {cap.note}")
    # Use moondream2 — small enough for CPU.
    adapter = VisionSubstrateAdapterV1(
        model_name="vikhyatk/moondream2",
        require_real_vlm=True,
        precision_tier="tier_fp32",
        embedding_dim=2048)
    png = make_tiny_png_v1(seed=99)
    p = adapter.encode_image(png, prompt="What is in this image?")
    # Encoder fingerprint must be hf_vlm, not stub.
    assert p.encoder.encoder_kind == "hf_vlm"
    assert "moondream" in p.encoder.model_name.lower()
    # Hidden-state shape: (n_patches, embedding_dim).
    # Moondream2 vision tower: n_patches=729 (27x27 grid),
    # embedding_dim=2048.
    assert p.embedding.shape == (729, 2048)
    # Non-trivial values.
    assert float(np.abs(p.embedding).max()) > 0.0
