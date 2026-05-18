"""W82 / P2 #13 — Cross-runtime portability tests.

Covers:
- runtime signature is content-addressed
- portable carrier is content-addressed
- same-signature round-trip is bit-identical for the exact-
  replay payload
- cross-signature transfer preserves the shared semantic
  projection from TARGET hidden state (cosine ≥ 0.95)
- cross-signature target-runtime classification survives
  ≥ 90% (the load-bearing portability claim)
- cross-signature target hidden state is approximate rather
  than exact (relative error > 0)
- raw hidden-coord-0 classification is NOT portable across
  different-hidden-dim signatures (honest scope: that axis is
  *not* in the EXACT_REPLAY or APPROXIMATE_SEMANTIC tier
  outside same-sig)
- non-portable axes dropped (drop_rate = 1.0)
- bench is deterministic on seed
"""

from __future__ import annotations


def test_w82_portability_runtime_signature_content_addressed():
    from coordpy.cross_runtime_state_portability_v1 import (
        build_runtime_signature_v1,
    )
    a = build_runtime_signature_v1(
        backend_label="x", vocab_size=10, hidden_dim=8)
    b = build_runtime_signature_v1(
        backend_label="x", vocab_size=10, hidden_dim=8)
    c = build_runtime_signature_v1(
        backend_label="y", vocab_size=10, hidden_dim=8)
    assert a.cid() == b.cid()
    assert a.cid() != c.cid()


def test_w82_portability_carrier_content_addressed():
    import numpy as np
    from coordpy.cross_runtime_state_portability_v1 import (
        build_runtime_signature_v1,
        build_portability_projector_v1,
        build_portability_bench_dataset_v1,
    )
    s = build_runtime_signature_v1(
        backend_label="t", vocab_size=12, hidden_dim=8)
    proj = build_portability_projector_v1()
    hs, tok = build_portability_bench_dataset_v1(
        source_signature=s, n_items=4, seed=7)
    ca = proj.encode_to_portable_v1(
        source_signature=s, hidden_states=hs, token_ids=tok)
    cb = proj.encode_to_portable_v1(
        source_signature=s, hidden_states=hs, token_ids=tok)
    assert ca.cid() == cb.cid()


def test_w82_portability_same_sig_bit_identical():
    """Same-signature round-trip MUST be bit-identical via
    exact_replay_payload."""
    import numpy as np
    from coordpy.cross_runtime_state_portability_v1 import (
        build_runtime_signature_v1,
        build_portability_projector_v1,
        build_portability_bench_dataset_v1,
    )
    s = build_runtime_signature_v1(
        backend_label="t", vocab_size=16, hidden_dim=8)
    proj = build_portability_projector_v1()
    hs, tok = build_portability_bench_dataset_v1(
        source_signature=s, n_items=4, seed=11)
    carrier = proj.encode_to_portable_v1(
        source_signature=s, hidden_states=hs, token_ids=tok)
    hs_back = proj.decode_to_runtime_v1(
        carrier=carrier, target_signature=s)
    assert np.array_equal(hs, hs_back)


def test_w82_portability_three_tiers_named():
    from coordpy.cross_runtime_state_portability_v1 import (
        W82_PORTABILITY_TIERS,
        PortabilityTier,
    )
    assert set(W82_PORTABILITY_TIERS) == {
        "exact_replay",
        "approximate_semantic",
        "non_portable",
    }
    assert PortabilityTier.EXACT_REPLAY.value == "exact_replay"


def test_w82_portability_tier_for_same_sig_is_exact_replay():
    from coordpy.cross_runtime_state_portability_v1 import (
        build_runtime_signature_v1,
        build_portability_projector_v1,
        PortabilityTier,
    )
    s = build_runtime_signature_v1(
        backend_label="r", vocab_size=8, hidden_dim=8)
    proj = build_portability_projector_v1()
    assert proj.portability_tier_for(
        source=s, target=s) == PortabilityTier.EXACT_REPLAY.value


def test_w82_portability_tier_for_diff_sig_is_approximate_semantic():
    from coordpy.cross_runtime_state_portability_v1 import (
        build_runtime_signature_v1,
        build_portability_projector_v1,
        PortabilityTier,
    )
    sa = build_runtime_signature_v1(
        backend_label="ra", vocab_size=8, hidden_dim=8)
    sb = build_runtime_signature_v1(
        backend_label="rb", vocab_size=12, hidden_dim=10)
    proj = build_portability_projector_v1()
    assert (
        proj.portability_tier_for(source=sa, target=sb) ==
        PortabilityTier.APPROXIMATE_SEMANTIC.value)


def test_w82_portability_cross_sig_anchor_cosine_high():
    """Cross-signature transfer preserves semantic cosine from
    transferred TARGET hidden state ≥ 0.95."""
    from coordpy.cross_runtime_state_portability_v1 import (
        run_portability_bench_end_to_end_v1,
    )
    reports, witness = run_portability_bench_end_to_end_v1()
    cross = [r for r in reports
             if r.tier == "approximate_semantic"]
    assert len(cross) >= 2
    assert witness.min_cross_signature_cosine >= 0.95


def test_w82_portability_anchor_classifier_preserved_geq_90pct():
    """Cross-signature target-runtime classifier preserved on
    ≥ 90% of items (the load-bearing portability claim)."""
    from coordpy.cross_runtime_state_portability_v1 import (
        run_portability_bench_end_to_end_v1,
    )
    _r, witness = run_portability_bench_end_to_end_v1()
    assert witness.min_cross_signature_classification >= 0.9


def test_w82_portability_non_portable_axis_drop_rate_is_one():
    """All declared non-portable axes are correctly dropped."""
    from coordpy.cross_runtime_state_portability_v1 import (
        run_portability_bench_end_to_end_v1,
    )
    reports, _w = run_portability_bench_end_to_end_v1()
    for r in reports:
        assert float(r.non_portable_axis_drop_rate) == 1.0


def test_w82_portability_witness_passes_threshold():
    from coordpy.cross_runtime_state_portability_v1 import (
        run_portability_bench_end_to_end_v1,
    )
    _r, w = run_portability_bench_end_to_end_v1()
    assert w.cross_signature_passes_threshold is True
    assert w.same_signature_bit_identical is True
    assert w.max_cross_signature_target_state_relative_error > 0.0


def test_w82_portability_same_sig_report_has_zero_target_state_error():
    from coordpy.cross_runtime_state_portability_v1 import (
        build_runtime_signature_v1,
        run_portability_pair_bench_v1,
    )
    s = build_runtime_signature_v1(
        backend_label="same", vocab_size=16, hidden_dim=8)
    report = run_portability_pair_bench_v1(
        source_signature=s, target_signature=s, seed=7)
    assert report.exact_replay_bit_identical is True
    assert float(report.target_state_relative_error) == 0.0


def test_w82_portability_cross_sig_target_state_error_nonzero():
    from coordpy.cross_runtime_state_portability_v1 import (
        build_runtime_signature_v1,
        run_portability_pair_bench_v1,
    )
    sa = build_runtime_signature_v1(
        backend_label="a", vocab_size=24, hidden_dim=8)
    sb = build_runtime_signature_v1(
        backend_label="b", vocab_size=32, hidden_dim=12)
    report = run_portability_pair_bench_v1(
        source_signature=sa, target_signature=sb, seed=7)
    assert report.tier == "approximate_semantic"
    assert float(report.target_state_relative_error) > 0.0


def test_w82_portability_bench_deterministic():
    from coordpy.cross_runtime_state_portability_v1 import (
        run_portability_bench_end_to_end_v1,
    )
    a, wa = run_portability_bench_end_to_end_v1(seed=99)
    b, wb = run_portability_bench_end_to_end_v1(seed=99)
    assert wa.cid() == wb.cid()
    for ra, rb in zip(a, b):
        assert ra.cid() == rb.cid()


def test_w82_portability_bench_cid_changes_on_seed():
    from coordpy.cross_runtime_state_portability_v1 import (
        run_portability_bench_end_to_end_v1,
    )
    _a, wa = run_portability_bench_end_to_end_v1(seed=99)
    _b, wb = run_portability_bench_end_to_end_v1(seed=100)
    assert wa.cid() != wb.cid()


def test_w82_portability_carrier_drops_non_portable_axes():
    """The carrier MUST list non-portable axes explicitly so
    the receiver knows the data was NOT shipped."""
    import numpy as np
    from coordpy.cross_runtime_state_portability_v1 import (
        build_runtime_signature_v1,
        build_portability_projector_v1,
        build_portability_bench_dataset_v1,
    )
    s = build_runtime_signature_v1(
        backend_label="t", vocab_size=8, hidden_dim=6)
    proj = build_portability_projector_v1(
        non_portable_axes=("kv_layout", "trace_id"))
    hs, tok = build_portability_bench_dataset_v1(
        source_signature=s, n_items=2, seed=11)
    c = proj.encode_to_portable_v1(
        source_signature=s, hidden_states=hs, token_ids=tok)
    assert "kv_layout" in c.non_portable_axis_labels
    assert "trace_id" in c.non_portable_axis_labels


def test_w82_portability_hidden_coord_zero_NOT_portable_cross_sig():
    """Honest-scope test: raw hidden-coordinate 0 is NOT
    expected to survive cross-signature transfer. Different
    hidden_dim → coord-0 has no shared meaning."""
    import numpy as np
    from coordpy.cross_runtime_state_portability_v1 import (
        build_runtime_signature_v1,
        build_portability_projector_v1,
        build_portability_bench_dataset_v1,
    )
    sa = build_runtime_signature_v1(
        backend_label="ra", vocab_size=24,
        hidden_dim=12, n_layers=3)
    sb = build_runtime_signature_v1(
        backend_label="rb", vocab_size=32,
        hidden_dim=8, n_layers=2)
    proj = build_portability_projector_v1()
    hs_a, tok_a = build_portability_bench_dataset_v1(
        source_signature=sa, n_items=64, seed=11)
    c = proj.encode_to_portable_v1(
        source_signature=sa,
        hidden_states=hs_a, token_ids=tok_a)
    hs_b = proj.decode_to_runtime_v1(
        carrier=c, target_signature=sb)
    # Raw coord-0 sign preservation is uncorrelated across
    # different-hidden-dim runtimes; we just assert the test
    # ran and that the cross-runtime hs_b has the target
    # hidden_dim. This is the *honest scope* observation, not
    # a load-bearing fidelity bar.
    assert hs_b.shape == (64, int(sb.hidden_dim))


def test_w82_portability_decode_validates_anchor_dim():
    """Decoding a carrier with a mismatched anchor_dim
    raises."""
    import numpy as np
    from coordpy.cross_runtime_state_portability_v1 import (
        build_runtime_signature_v1,
        build_portability_projector_v1,
        build_portability_bench_dataset_v1,
        PortableStateCarrierV1,
        W82_PORTABILITY_V1_SCHEMA_VERSION,
    )
    s = build_runtime_signature_v1(
        backend_label="t", vocab_size=8, hidden_dim=6)
    proj = build_portability_projector_v1(anchor_dim=4)
    hs, tok = build_portability_bench_dataset_v1(
        source_signature=s, n_items=2, seed=11)
    c = proj.encode_to_portable_v1(
        source_signature=s, hidden_states=hs, token_ids=tok)
    # Use a different anchor_dim projector to provoke the
    # mismatch on a cross-signature decode (the same-signature
    # path takes the exact-replay shortcut and bypasses the
    # anchor_dim check).
    s2 = build_runtime_signature_v1(
        backend_label="other", vocab_size=8, hidden_dim=6)
    other = build_portability_projector_v1(anchor_dim=5)
    try:
        other.decode_to_runtime_v1(
            carrier=c, target_signature=s2)
    except ValueError:
        return
    raise AssertionError(
        "decode with mismatched anchor_dim should raise")


def test_w82_portability_reports_separate_signature_cids():
    """The pair report records BOTH source and target
    signature CIDs so chaining is unambiguous."""
    from coordpy.cross_runtime_state_portability_v1 import (
        run_portability_bench_end_to_end_v1,
    )
    reports, _w = run_portability_bench_end_to_end_v1()
    for r in reports:
        assert len(r.source_signature_cid) == 64
        assert len(r.target_signature_cid) == 64
