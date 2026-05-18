"""W80 / P0 #5 — transformers runtime V1 tests.

These tests are gated on the optional ``transformers`` + ``torch``
deps. When they are missing, the tests skip cleanly so CI on
lean environments still passes.
"""

from __future__ import annotations

import numpy as np
import pytest
from types import SimpleNamespace

try:
    import torch  # type: ignore  # noqa: F401
    import transformers  # type: ignore  # noqa: F401
    _HAS_HF = True
except Exception:  # noqa: BLE001
    _HAS_HF = False


def test_w80_transformers_probe_honest_when_runtime_init_fails(
        monkeypatch):
    import coordpy.transformers_runtime_v1 as tr

    monkeypatch.setattr(
        tr,
        "_probe_transformers_runtime_instantiable",
        lambda model_name: (
            False,
            "transformers + torch importable but runtime "
            "instantiation failed: TypeError: broken",
        ),
    )
    p = tr.probe_transformers_runtime_v1()
    assert not bool(p.transformers_available)
    assert "instantiation failed" in " ".join(p.notes)


def test_w80_transformers_runtime_uses_torch_dtype_kwarg(
        monkeypatch):
    import coordpy.transformers_runtime_v1 as tr

    call: dict[str, object] = {}

    class _FakeTorch:
        float32 = "float32"

        @staticmethod
        def set_grad_enabled(_enabled: bool) -> None:
            return None

    class _FakeAutoModelForCausalLM:
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            call["model_name"] = str(model_name)
            call["kwargs"] = dict(kwargs)
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
            call["tokenizer_model_name"] = str(model_name)
            return object()

    monkeypatch.setattr(
        tr,
        "_torch_modules",
        lambda: (
            _FakeTorch(),
            _FakeAutoModelForCausalLM,
            _FakeAutoTokenizer,
        ),
    )

    tr.TransformersRuntimeV1(model_name="fake/model")

    kwargs = dict(call["kwargs"])
    assert call["model_name"] == "fake/model"
    assert call["tokenizer_model_name"] == "fake/model"
    assert kwargs["torch_dtype"] == "float32"
    assert "dtype" not in kwargs
    assert kwargs["attn_implementation"] == "eager"


@pytest.mark.skipif(
    not _HAS_HF,
    reason="transformers / torch not installed",
)
def test_w80_transformers_probe_records_availability():
    from coordpy.transformers_runtime_v1 import (
        probe_transformers_runtime_v1,
    )
    p = probe_transformers_runtime_v1()
    assert bool(p.transformers_available)
    assert p.backend_name == (
        "coordpy.transformers_runtime_v1")
    # Probe is content-addressed.
    p2 = probe_transformers_runtime_v1()
    assert p.cid() == p2.cid()


@pytest.mark.skipif(
    not _HAS_HF,
    reason="transformers / torch not installed",
)
def test_w80_transformers_runtime_loads_distilgpt2():
    from coordpy.transformers_runtime_v1 import (
        TransformersRuntimeV1,
    )
    rt = TransformersRuntimeV1()
    # distilgpt2 baseline shapes.
    assert int(rt.n_layers) == 6
    assert int(rt.n_heads) == 12
    assert int(rt.hidden_dim) == 768
    assert int(rt.head_dim) == 64


@pytest.mark.skipif(
    not _HAS_HF,
    reason="transformers / torch not installed",
)
def test_w80_transformers_runtime_deterministic_forward():
    from coordpy.transformers_runtime_v1 import (
        TransformersRuntimeV1,
    )
    rt = TransformersRuntimeV1()
    ids = rt.tokenize("w80 transformers determinism check",
                      max_len=12)
    t1 = rt.forward(input_token_ids=ids)
    t2 = rt.forward(input_token_ids=ids)
    assert t1.cid() == t2.cid()


@pytest.mark.skipif(
    not _HAS_HF,
    reason="transformers / torch not installed",
)
def test_w80_transformers_runtime_replay_byte_identical():
    from coordpy.runtime_instrumentation_v1 import (
        W80_REPLAY_FROM_KV_MAX_ABS_DIFF,
    )
    from coordpy.transformers_runtime_v1 import (
        TransformersRuntimeV1,
    )
    rt = TransformersRuntimeV1()
    ids = rt.tokenize(
        "w80 transformers replay byte identity check",
        max_len=12)
    full = rt.forward(input_token_ids=ids)
    old = rt.forward(input_token_ids=ids[:-1])
    replay = rt.replay_from_kv(
        kv=old.kv, new_token_ids=ids[-1:])
    full_last = np.asarray(full.final_logits)[-1]
    rep_last = np.asarray(replay.final_logits)[-1]
    diff = float(np.max(np.abs(full_last - rep_last)))
    # fp32 CPU replay: ≤ 5e-3 is honest under the W80 contract.
    # In practice on distilgpt2 we see < 1e-3.
    assert diff < float(W80_REPLAY_FROM_KV_MAX_ABS_DIFF), (
        f"transformers replay diff {diff} > 5e-3 — fp32 "
        "should suffice")


@pytest.mark.skipif(
    not _HAS_HF,
    reason="transformers / torch not installed",
)
def test_w80_transformers_runtime_passes_conformance():
    from coordpy.runtime_instrumentation_v1 import (
        run_instrumentation_conformance_v1,
    )
    from coordpy.transformers_runtime_v1 import (
        TransformersRuntimeV1,
    )
    rt = TransformersRuntimeV1()
    r = run_instrumentation_conformance_v1(
        rt, prompt="conformance smoke for transformers runtime")
    # The HF runtime claims every axis (some BACKEND_SPECIFIC).
    # Every claimed axis must pass.
    assert int(r.n_fail) == 0, (
        f"failures: "
        f"{[(a, s) for a, s in r.per_axis if s == 'fail']}")
    assert int(r.n_pass) >= 10


@pytest.mark.skipif(
    not _HAS_HF,
    reason="transformers / torch not installed",
)
def test_w80_transformers_runtime_hidden_inject_moves_cid():
    from coordpy.runtime_instrumentation_v1 import (
        InjectionPlanV1,
        W80_RUNTIME_INSTRUMENTATION_V1_SCHEMA_VERSION,
    )
    from coordpy.transformers_runtime_v1 import (
        TransformersRuntimeV1,
    )
    rt = TransformersRuntimeV1()
    ids = rt.tokenize("inject probe", max_len=8)
    t_base = rt.forward(input_token_ids=ids)
    inj_layer0 = np.full(
        (int(t_base.seq_len), int(rt.hidden_dim)),
        0.05, dtype=np.float64)
    per_layer = [None] * int(rt.n_layers)
    per_layer[0] = inj_layer0
    plan = InjectionPlanV1(
        schema=W80_RUNTIME_INSTRUMENTATION_V1_SCHEMA_VERSION,
        hidden_state_inject_per_layer=tuple(per_layer))
    t_inj = rt.forward(input_token_ids=ids, injection=plan)
    assert t_base.cid() != t_inj.cid()


@pytest.mark.skipif(
    not _HAS_HF,
    reason="transformers / torch not installed",
)
def test_w80_transformers_runtime_prefix_inject_moves_cid():
    from coordpy.runtime_instrumentation_v1 import (
        InjectionPlanV1,
        W80_RUNTIME_INSTRUMENTATION_V1_SCHEMA_VERSION,
    )
    from coordpy.transformers_runtime_v1 import (
        TransformersRuntimeV1,
    )
    rt = TransformersRuntimeV1()
    ids = rt.tokenize("prefix probe", max_len=8)
    t_base = rt.forward(input_token_ids=ids)
    prefix = np.full(
        (int(t_base.seq_len), int(rt.hidden_dim)),
        0.03, dtype=np.float64)
    plan = InjectionPlanV1(
        schema=W80_RUNTIME_INSTRUMENTATION_V1_SCHEMA_VERSION,
        prefix_state_inject=prefix)
    t_pref = rt.forward(input_token_ids=ids, injection=plan)
    assert t_base.cid() != t_pref.cid()


@pytest.mark.skipif(
    not _HAS_HF,
    reason="transformers / torch not installed",
)
def test_w80_transformers_runtime_attention_bias_moves_cid():
    from coordpy.runtime_instrumentation_v1 import (
        InjectionPlanV1,
        W80_RUNTIME_INSTRUMENTATION_V1_SCHEMA_VERSION,
    )
    from coordpy.transformers_runtime_v1 import (
        TransformersRuntimeV1,
    )
    rt = TransformersRuntimeV1()
    ids = rt.tokenize("attention bias probe", max_len=8)
    t_base = rt.forward(input_token_ids=ids)
    bias_shape = (
        int(rt.n_heads),
        int(t_base.attn.seq_q),
        int(t_base.attn.seq_k))
    bias = np.full(bias_shape, 0.05, dtype=np.float64)
    per_layer_bias = [None] * int(rt.n_layers)
    per_layer_bias[0] = bias
    plan = InjectionPlanV1(
        schema=W80_RUNTIME_INSTRUMENTATION_V1_SCHEMA_VERSION,
        attention_bias_per_layer=tuple(per_layer_bias))
    t_bias = rt.forward(input_token_ids=ids, injection=plan)
    assert t_base.cid() != t_bias.cid()
