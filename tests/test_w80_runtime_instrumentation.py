"""W80 / P0 #8 — runtime instrumentation contract tests."""

from __future__ import annotations

import numpy as np
import pytest


def test_w80_instrumentation_contract_schema_versioned():
    from coordpy.runtime_instrumentation_v1 import (
        W80_RUNTIME_INSTRUMENTATION_V1_SCHEMA_VERSION,
        W80_INSTRUMENTATION_AXES_ALL,
        W80_INSTRUMENTATION_AXES_READ,
        W80_INSTRUMENTATION_AXES_WRITE,
        W80_INSTRUMENTATION_AXES_WITNESS,
    )
    assert isinstance(
        W80_RUNTIME_INSTRUMENTATION_V1_SCHEMA_VERSION, str)
    assert (
        W80_RUNTIME_INSTRUMENTATION_V1_SCHEMA_VERSION
        .startswith("coordpy.runtime_instrumentation_v1"))
    assert len(W80_INSTRUMENTATION_AXES_ALL) == 12
    # Reads + writes + witness partition the axis set.
    assert set(W80_INSTRUMENTATION_AXES_READ).isdisjoint(
        set(W80_INSTRUMENTATION_AXES_WRITE))
    assert set(W80_INSTRUMENTATION_AXES_WRITE).isdisjoint(
        set(W80_INSTRUMENTATION_AXES_WITNESS))
    assert set(W80_INSTRUMENTATION_AXES_READ).isdisjoint(
        set(W80_INSTRUMENTATION_AXES_WITNESS))
    covered = (
        set(W80_INSTRUMENTATION_AXES_READ)
        | set(W80_INSTRUMENTATION_AXES_WRITE)
        | set(W80_INSTRUMENTATION_AXES_WITNESS))
    assert covered == set(W80_INSTRUMENTATION_AXES_ALL)


def test_w80_capability_tags_are_canonical():
    from coordpy.runtime_instrumentation_v1 import (
        CapabilityTag,
    )
    expected = {"available", "backend_specific",
                "best_effort", "unavailable"}
    actual = {t.value for t in CapabilityTag}
    assert actual == expected


def test_w80_replay_tolerance_matches_hf_contract():
    from coordpy.runtime_instrumentation_v1 import (
        W80_REPLAY_FROM_KV_MAX_ABS_DIFF,
    )
    assert float(W80_REPLAY_FROM_KV_MAX_ABS_DIFF) == pytest.approx(
        5e-3)


def test_w80_controlled_runtime_adapter_passes_conformance():
    from coordpy.runtime_instrumentation_v1 import (
        ControlledRuntimeInstrumentationAdapterV1,
        run_instrumentation_conformance_v1,
        InstrumentationAxis,
    )
    a = ControlledRuntimeInstrumentationAdapterV1()
    report = run_instrumentation_conformance_v1(
        a, prompt="controlled runtime conformance check")
    # The W79 numpy controlled runtime claims every axis as
    # AVAILABLE — so every axis should pass.
    assert int(report.n_pass) == 12
    assert int(report.n_skip) == 0
    assert int(report.n_fail) == 0
    assert bool(report.all_claimed_pass())
    # The per-axis dict carries the same axis names as the
    # canonical contract.
    per = report.per_axis_dict()
    for ax in InstrumentationAxis:
        assert ax.value in per


def test_w80_controlled_runtime_adapter_deterministic():
    from coordpy.runtime_instrumentation_v1 import (
        ControlledRuntimeInstrumentationAdapterV1,
    )
    a = ControlledRuntimeInstrumentationAdapterV1()
    ids = a.tokenize("w80 determinism check", max_len=16)
    t1 = a.forward(input_token_ids=ids)
    t2 = a.forward(input_token_ids=ids)
    assert t1.cid() == t2.cid()
    assert t1.hidden is not None
    assert t1.hidden.cid() == t2.hidden.cid()
    assert t1.kv is not None
    assert t1.kv.cid() == t2.kv.cid()
    assert t1.attn is not None
    assert t1.attn.cid() == t2.attn.cid()


def test_w80_controlled_runtime_adapter_injection_moves_cid():
    from coordpy.runtime_instrumentation_v1 import (
        ControlledRuntimeInstrumentationAdapterV1,
        InjectionPlanV1,
        W80_RUNTIME_INSTRUMENTATION_V1_SCHEMA_VERSION,
    )
    a = ControlledRuntimeInstrumentationAdapterV1()
    ids = a.tokenize("w80 injection check", max_len=16)
    t_base = a.forward(input_token_ids=ids)
    n_layers = int(t_base.hidden.n_layers)
    inj = np.full(
        t_base.hidden.per_layer[0].shape,
        0.05, dtype=np.float64)
    per_layer = [None] * n_layers
    per_layer[0] = inj
    plan = InjectionPlanV1(
        schema=W80_RUNTIME_INSTRUMENTATION_V1_SCHEMA_VERSION,
        hidden_state_inject_per_layer=tuple(per_layer))
    t_inj = a.forward(input_token_ids=ids, injection=plan)
    assert t_base.cid() != t_inj.cid()


def test_w80_controlled_runtime_adapter_replay_byte_identical():
    from coordpy.runtime_instrumentation_v1 import (
        ControlledRuntimeInstrumentationAdapterV1,
    )
    a = ControlledRuntimeInstrumentationAdapterV1()
    ids = a.tokenize(
        "w80 replay-from-kv byte-identity check", max_len=24)
    full = a.forward(input_token_ids=ids)
    old = a.forward(input_token_ids=ids[:-1])
    replay = a.replay_from_kv(
        kv=old.kv, new_token_ids=ids[-1:])
    full_last = np.asarray(full.final_logits)[-1]
    rep_last = np.asarray(replay.final_logits)[-1]
    diff = float(np.max(np.abs(full_last - rep_last)))
    assert diff < 1e-8, (
        f"controlled runtime replay must be byte-identical; "
        f"diff={diff}")


def test_w80_instrumentation_witness_emits_stable_cid():
    from coordpy.runtime_instrumentation_v1 import (
        ControlledRuntimeInstrumentationAdapterV1,
        emit_instrumentation_witness,
    )
    a = ControlledRuntimeInstrumentationAdapterV1()
    w1 = emit_instrumentation_witness(a)
    w2 = emit_instrumentation_witness(a)
    assert w1.cid() == w2.cid()
    assert (
        w1.schema.startswith(
            "coordpy.runtime_instrumentation_v1"))


def test_w80_conformance_skips_unavailable_axes_honestly():
    """A backend that claims fewer axes should produce skips."""
    from coordpy.runtime_instrumentation_v1 import (
        CapabilityTag,
        ControlledRuntimeInstrumentationAdapterV1,
        InstrumentationAxis,
        run_instrumentation_conformance_v1,
    )

    base = ControlledRuntimeInstrumentationAdapterV1()

    class _RestrictedBackend:
        """A backend that hides three axes; conformance must
        skip them rather than report failure."""

        def backend_id(self):
            return "test.restricted_backend"

        def backend_runtime_id(self):
            return "test.restricted_backend#v0"

        def declared_axes(self):
            d = dict(base.declared_axes())
            for ax in (
                    InstrumentationAxis.WRITE_ATTENTION_BIAS,
                    InstrumentationAxis.INJECT_PREFIX_STATE,
                    InstrumentationAxis.READ_PER_LAYER_LOGITS):
                d[ax.value] = CapabilityTag.UNAVAILABLE.value
            return d

        def tokenize(self, text, *, max_len=64):
            return base.tokenize(text, max_len=max_len)

        def forward(self, *, input_token_ids, injection=None):
            t = base.forward(
                input_token_ids=input_token_ids,
                injection=injection)
            return type(t)(
                **{**t.__dict__,
                   "declared_axes": tuple(
                       (str(k), str(v))
                       for k, v in
                       self.declared_axes().items())})

        def replay_from_kv(
                self, *, kv, new_token_ids):
            t = base.replay_from_kv(
                kv=kv, new_token_ids=new_token_ids)
            return type(t)(
                **{**t.__dict__,
                   "declared_axes": tuple(
                       (str(k), str(v))
                       for k, v in
                       self.declared_axes().items())})

    r = run_instrumentation_conformance_v1(
        _RestrictedBackend(),
        prompt="restricted backend conformance smoke")
    assert int(r.n_skip) >= 3
    assert int(r.n_fail) == 0
    # The skipped axes are exactly the ones marked UNAVAILABLE.
    per = r.per_axis_dict()
    assert per[
        InstrumentationAxis.WRITE_ATTENTION_BIAS.value
    ] == "skip"
    assert per[
        InstrumentationAxis.INJECT_PREFIX_STATE.value
    ] == "skip"
    assert per[
        InstrumentationAxis.READ_PER_LAYER_LOGITS.value
    ] == "skip"
