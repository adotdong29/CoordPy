"""W80 / P0 #5 — substrate adapter V25 integration tests.

V25 is the W80 extension of the W79 substrate adapter V24 line.
It registers the HF transformers controlled runtime V1 as a
first-class entry inside the existing substrate adapter
pipeline — closing the P0 #5 requirement that the new HF
backend integrates with the existing adapter line rather than
living in a parallel one-off module.
"""

from __future__ import annotations

import pytest

try:
    import torch  # type: ignore  # noqa: F401
    import transformers  # type: ignore  # noqa: F401
    _HAS_HF = True
except Exception:  # noqa: BLE001
    _HAS_HF = False


def test_w80_substrate_adapter_v25_schema_versioned():
    from coordpy.substrate_adapter_v25 import (
        W80_SUBSTRATE_ADAPTER_V25_SCHEMA_VERSION,
        W80_SUBSTRATE_TIER_TRANSFORMERS_RUNTIME_V1,
        W80_SUBSTRATE_V25_CAPABILITY_AXES,
    )
    assert isinstance(
        W80_SUBSTRATE_ADAPTER_V25_SCHEMA_VERSION, str)
    assert (
        W80_SUBSTRATE_ADAPTER_V25_SCHEMA_VERSION
        .startswith("coordpy.substrate_adapter_v25"))
    # V25 inherits V24's capability axis set — it adds a new
    # backend, not new axes.
    assert len(W80_SUBSTRATE_V25_CAPABILITY_AXES) > 0
    assert (
        W80_SUBSTRATE_TIER_TRANSFORMERS_RUNTIME_V1
        == "transformers_runtime_v1")


def test_w80_substrate_adapter_v25_matrix_extends_v24():
    """V25 adapter matrix must include the V24 backends AND
    the W80 HF backend (or a 'transformers unavailable' entry
    when the deps are missing)."""
    from coordpy.substrate_adapter_v25 import (
        probe_all_v25_adapters,
    )
    m = probe_all_v25_adapters()
    names = {c.backend_name for c in m.capabilities}
    # V24 backends carried forward.
    assert "tiny_substrate_v24" in names
    assert "controlled_runtime_substrate_v1" in names
    # V25 new backend.
    assert "transformers_runtime_v1" in names
    # Tamper-evident V24 chaining.
    assert isinstance(m.v24_matrix_cid, str)
    assert len(m.v24_matrix_cid) == 64


def test_w80_substrate_adapter_v25_unavailable_when_no_hf():
    """If transformers / torch are missing, the V25 probe
    surfaces the HF backend as 'unreachable' tier rather than
    silently dropping the row."""
    if _HAS_HF:
        pytest.skip(
            "transformers available — this exercises the "
            "missing-deps path")
    from coordpy.substrate_adapter_v25 import (
        probe_all_v25_adapters,
    )
    from coordpy.substrate_adapter import (
        SUBSTRATE_TIER_UNREACHABLE,
    )
    m = probe_all_v25_adapters()
    hf = next(
        c for c in m.capabilities
        if c.backend_name == "transformers_runtime_v1")
    assert hf.tier == SUBSTRATE_TIER_UNREACHABLE
    assert not bool(m.has_transformers_runtime())


@pytest.mark.skipif(
    not _HAS_HF,
    reason="transformers / torch not installed")
def test_w80_substrate_adapter_v25_records_hf_when_available():
    """When transformers + torch are installed, V25's HF
    backend hits the controlled-runtime tier, exposing the
    seven W79 controlled-runtime axes."""
    from coordpy.substrate_adapter_v25 import (
        probe_all_v25_adapters,
    )
    from coordpy.substrate_adapter_v24 import (
        W79_CONTROLLED_RUNTIME_AXES_AS_CAPABILITIES,
        W79_SUBSTRATE_TIER_CONTROLLED_RUNTIME_V1,
    )
    m = probe_all_v25_adapters()
    assert bool(m.has_transformers_runtime())
    hf = next(
        c for c in m.capabilities
        if c.backend_name == "transformers_runtime_v1")
    assert hf.tier == W79_SUBSTRATE_TIER_CONTROLLED_RUNTIME_V1
    cap_map = {ax: val for ax, val in hf.capabilities}
    for ax in W79_CONTROLLED_RUNTIME_AXES_AS_CAPABILITIES:
        assert cap_map.get(ax) == "yes", (
            f"HF backend must expose {ax}; got "
            f"{cap_map.get(ax)}")


@pytest.mark.skipif(
    not _HAS_HF,
    reason="transformers / torch not installed")
def test_w80_substrate_adapter_v25_has_two_controlled_runtimes():
    """The V25 matrix is the load-bearing P0 #6 evidence: two
    controlled-runtime backends sit in the same adapter
    line."""
    from coordpy.substrate_adapter_v25 import (
        probe_all_v25_adapters,
    )
    m = probe_all_v25_adapters()
    # In-repo NumPy controlled runtime + HF runtime + V24 full
    # tier all hit the controlled-runtime threshold.
    assert int(m.n_controlled_runtimes()) >= 2


def test_w80_substrate_adapter_v25_content_addressed():
    from coordpy.substrate_adapter_v25 import (
        probe_all_v25_adapters,
    )
    m = probe_all_v25_adapters()
    assert isinstance(m.cid(), str)
    assert len(m.cid()) == 64
    # to_dict round-trips through JSON.
    import json
    s = json.dumps(m.to_dict(), default=str)
    parsed = json.loads(s)
    assert "capabilities" in parsed
    assert "v24_matrix_cid" in parsed
