"""W80 / P0 #6 — runtime parity matrix V1 tests."""

from __future__ import annotations

import pytest

try:
    import torch  # type: ignore  # noqa: F401
    import transformers  # type: ignore  # noqa: F401
    _HAS_HF = True
except Exception:  # noqa: BLE001
    _HAS_HF = False


def test_w80_parity_matrix_without_transformers_has_one_backend():
    from coordpy.runtime_parity_matrix_v1 import (
        build_runtime_parity_matrix_v1,
    )
    m = build_runtime_parity_matrix_v1(
        include_transformers=False)
    # Always: in-repo NumPy controlled runtime.
    assert len(m.backends) == 1
    assert m.backends[0].backend_id == (
        "coordpy.controlled_runtime_substrate_v1")
    assert bool(m.backends[0].available)
    # Conformance: all axes pass on the in-repo runtime.
    r = m.conformance_reports[0]
    assert int(r.n_fail) == 0
    assert int(r.n_pass) == 12
    # Markdown render.
    md = m.render_markdown_table()
    assert "coordpy.controlled_runtime_substrate_v1" in md


def test_w80_parity_matrix_is_content_addressed():
    from coordpy.runtime_parity_matrix_v1 import (
        build_runtime_parity_matrix_v1,
        emit_runtime_parity_matrix_witness,
    )
    m = build_runtime_parity_matrix_v1(
        include_transformers=False)
    w = emit_runtime_parity_matrix_witness(m)
    # The matrix and witness have stable CIDs (the matrix CID
    # encodes the probed_at_ns; the witness CID encodes the
    # matrix CID, so two consecutive matrix-witness pairs may
    # differ by timestamp).
    assert isinstance(m.cid(), str)
    assert len(m.cid()) == 64
    assert w.matrix_cid == m.cid()
    assert int(w.n_backends) == int(len(m.backends))


@pytest.mark.skipif(
    not _HAS_HF,
    reason="transformers / torch not installed")
def test_w80_parity_matrix_with_transformers_has_two_backends():
    from coordpy.runtime_parity_matrix_v1 import (
        build_runtime_parity_matrix_v1,
    )
    m = build_runtime_parity_matrix_v1(
        include_transformers=True,
        prompt="parity matrix conformance smoke")
    assert len(m.backends) == 2
    ids = {b.backend_id for b in m.backends}
    assert "coordpy.controlled_runtime_substrate_v1" in ids
    assert "coordpy.transformers_runtime_v1" in ids
    # Both backends conform.
    for r in m.conformance_reports:
        assert int(r.n_fail) == 0
    # Some axes are universal across both runtimes.
    assert len(m.axes_universal()) >= 5
    # Markdown renders both columns.
    md = m.render_markdown_table()
    assert "controlled_runtime_substrate_v1" in md
    assert "transformers_runtime_v1" in md


def test_w80_parity_matrix_records_unavailable_backend():
    """If transformers truly is missing in the build, the
    matrix surfaces the gap rather than skipping the column."""
    from coordpy.runtime_parity_matrix_v1 import (
        build_runtime_parity_matrix_v1,
    )
    import importlib
    if _HAS_HF:
        pytest.skip(
            "transformers available — this test exercises "
            "the unavailable path")
    m = build_runtime_parity_matrix_v1(
        include_transformers=True)
    # The matrix should still surface the transformers backend
    # row, marked unavailable.
    assert len(m.backends) == 2
    transformers_backend = next(
        b for b in m.backends
        if b.backend_id == (
            "coordpy.transformers_runtime_v1"))
    assert not bool(transformers_backend.available)


def test_w80_parity_matrix_axes_universal_includes_reads():
    from coordpy.runtime_parity_matrix_v1 import (
        build_runtime_parity_matrix_v1,
    )
    m = build_runtime_parity_matrix_v1(
        include_transformers=False)
    # With one backend declaring AVAILABLE everywhere, every
    # claimed axis is universal.
    univ = m.axes_universal()
    assert "read_hidden_state" in univ
    assert "read_kv_cache" in univ
    assert "read_attention_probs" in univ
