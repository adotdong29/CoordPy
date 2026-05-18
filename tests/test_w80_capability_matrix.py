"""W80 / P0 #17 — living capability matrix V1 tests."""

from __future__ import annotations

import pytest

try:
    import torch  # type: ignore  # noqa: F401
    import transformers  # type: ignore  # noqa: F401
    _HAS_HF = True
except Exception:  # noqa: BLE001
    _HAS_HF = False


def test_w80_capability_matrix_has_four_surfaces():
    from coordpy.capability_matrix_v1 import (
        W80_SURFACE_CONTROLLED_RUNTIME_NUMPY,
        W80_SURFACE_CONTROLLED_RUNTIME_TRANSFORMERS,
        W80_SURFACE_HOSTED_THIRD_PARTY,
        W80_SURFACE_LOCAL_OPENAI_FACADE,
        build_capability_matrix_v1,
    )
    m = build_capability_matrix_v1(
        include_transformers=_HAS_HF)
    assert len(m.surfaces) == 4
    ids = {s.surface_id for s in m.surfaces}
    assert W80_SURFACE_HOSTED_THIRD_PARTY in ids
    assert W80_SURFACE_LOCAL_OPENAI_FACADE in ids
    assert W80_SURFACE_CONTROLLED_RUNTIME_NUMPY in ids
    assert (
        W80_SURFACE_CONTROLLED_RUNTIME_TRANSFORMERS in ids)


def test_w80_capability_matrix_is_living():
    """Two calls must produce two distinct refresh timestamps
    and two CIDs that differ on the timestamp field."""
    from coordpy.capability_matrix_v1 import (
        build_capability_matrix_v1,
    )
    import time
    m1 = build_capability_matrix_v1(include_transformers=False)
    time.sleep(0.001)
    m2 = build_capability_matrix_v1(include_transformers=False)
    # The matrices share the *boundary* CID (deterministic
    # hosted plane) but differ in refreshed_at_ns and matrix
    # CID.
    assert (
        m1.hosted_boundary_v12_cid
        == m2.hosted_boundary_v12_cid)
    assert int(m2.refreshed_at_ns) > int(m1.refreshed_at_ns)
    assert m1.cid() != m2.cid()


def test_w80_capability_matrix_hosted_blocks_substrate_axes():
    from coordpy.capability_matrix_v1 import (
        W80_SURFACE_HOSTED_THIRD_PARTY,
        build_capability_matrix_v1,
    )
    from coordpy.runtime_instrumentation_v1 import (
        CapabilityTag,
    )
    m = build_capability_matrix_v1(
        include_transformers=False)
    hosted = next(
        s for s in m.surfaces
        if s.surface_id == W80_SURFACE_HOSTED_THIRD_PARTY)
    # The hosted plane must block the substrate axes that the
    # controlled runtime exposes. The W79 boundary V12 already
    # blocks 7 controlled-runtime axes.
    blocked = m.axes_blocked_on_hosted()
    assert len(blocked) >= 7
    # And the cross-link must hit at least the 7 W79 axes.
    cross = dict(m.axis_cross_link)
    blocked_keys = [
        ax for ax in blocked if ax in cross]
    assert len(blocked_keys) >= 7


def test_w80_capability_matrix_local_universal_axes_dominates():
    """The lean no-transformers path still has a meaningful
    local-universal surface."""
    from coordpy.capability_matrix_v1 import (
        build_capability_matrix_v1,
    )
    m = build_capability_matrix_v1(
        include_transformers=False)
    univ = m.axes_universal_on_local()
    assert len(univ) >= 5


def test_w80_capability_matrix_asymmetry_report_records_load_bearing_gap():
    from coordpy.capability_matrix_v1 import (
        build_capability_matrix_v1,
    )
    m = build_capability_matrix_v1(
        include_transformers=False)
    rep = m.asymmetry_report()
    # Hosted-blocked AND local-universal = asymmetry surface.
    # This is the load-bearing W80 bar.
    assert int(rep["n_asymmetry_axes"]) >= 5
    # The asymmetry-axes set is a subset of the hosted-blocked
    # set and a subset of the local-universal set.
    asym = set(rep["asymmetry_axes"])
    assert asym.issubset(set(rep["hosted_blocked_axes"]))
    assert asym.issubset(set(rep["local_universal_axes"]))


def test_w80_capability_matrix_facade_surface_not_overclaimed():
    from coordpy.capability_matrix_v1 import (
        W80_SURFACE_LOCAL_OPENAI_FACADE,
        build_capability_matrix_v1,
    )
    from coordpy.runtime_instrumentation_v1 import (
        CapabilityTag,
        InstrumentationAxis,
    )
    m = build_capability_matrix_v1(
        include_transformers=False)
    facade = next(
        s for s in m.surfaces
        if s.surface_id == W80_SURFACE_LOCAL_OPENAI_FACADE)
    assert facade.axis_tag(
        InstrumentationAxis.WRITE_KV_RESTORE.value
    ) == CapabilityTag.UNAVAILABLE.value
    assert facade.axis_tag(
        InstrumentationAxis.INJECT_PREFIX_STATE.value
    ) == CapabilityTag.UNAVAILABLE.value
    assert facade.axis_tag(
        InstrumentationAxis.REPLAY_FROM_KV.value
    ) == CapabilityTag.UNAVAILABLE.value
    assert facade.axis_tag(
        InstrumentationAxis.READ_PER_LAYER_LOGITS.value
    ) == CapabilityTag.UNAVAILABLE.value


@pytest.mark.skipif(
    not _HAS_HF,
    reason="transformers / torch not installed")
def test_w80_capability_matrix_local_universal_axes_with_transformers():
    from coordpy.capability_matrix_v1 import (
        build_capability_matrix_v1,
    )
    m = build_capability_matrix_v1(
        include_transformers=True)
    assert len(m.axes_universal_on_local()) >= 5


def test_w80_capability_matrix_renders_markdown():
    from coordpy.capability_matrix_v1 import (
        build_capability_matrix_v1,
    )
    m = build_capability_matrix_v1(
        include_transformers=False)
    md = m.render_markdown()
    assert "Axis" in md
    assert "hosted_third_party" in md
    assert "controlled_runtime_numpy" in md


def test_w80_capability_matrix_witness_is_content_addressed():
    from coordpy.capability_matrix_v1 import (
        build_capability_matrix_v1,
        emit_capability_matrix_v1_witness,
    )
    m = build_capability_matrix_v1(
        include_transformers=False)
    w1 = emit_capability_matrix_v1_witness(m)
    w2 = emit_capability_matrix_v1_witness(m)
    assert w1.cid() == w2.cid()
    assert w1.matrix_cid == m.cid()


def test_w80_capability_matrix_json_serialisable():
    from coordpy.capability_matrix_v1 import (
        build_capability_matrix_v1,
    )
    import json
    m = build_capability_matrix_v1(
        include_transformers=False)
    s = m.to_json()
    # Round-trip: the JSON must be parseable and carry the
    # surface count, axis count, and CID.
    parsed = json.loads(s)
    assert "surfaces" in parsed
    assert "axes_order" in parsed
    assert "hosted_boundary_v12_cid" in parsed
    assert len(parsed["surfaces"]) == 4
