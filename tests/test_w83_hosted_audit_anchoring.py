"""W83 — hosted audit anchoring tests."""

from __future__ import annotations


def test_w83_hosted_segment_content_addressed():
    from coordpy.hosted_audit_anchoring_v1 import (
        build_hosted_transcript_segment_v1,
    )
    a = build_hosted_transcript_segment_v1(
        segment_id="s1", role="planner",
        provider_id="openai_paid", model_id="gpt-4o-mini",
        prompt_bytes=b"hello", response_bytes=b"world",
        cache_hit=False, logprob_observed=True,
        timestamp_ns=1234)
    b = build_hosted_transcript_segment_v1(
        segment_id="s1", role="planner",
        provider_id="openai_paid", model_id="gpt-4o-mini",
        prompt_bytes=b"hello", response_bytes=b"world",
        cache_hit=False, logprob_observed=True,
        timestamp_ns=1234)
    assert str(a.cid()) == str(b.cid())
    c = build_hosted_transcript_segment_v1(
        segment_id="s1", role="planner",
        provider_id="openai_paid", model_id="gpt-4o-mini",
        prompt_bytes=b"hello!", response_bytes=b"world",
        cache_hit=False, logprob_observed=True,
        timestamp_ns=1234)
    assert str(a.cid()) != str(c.cid())


def test_w83_hosted_audit_anchor_merkle_root_verifies():
    from coordpy.hosted_audit_anchoring_v1 import (
        build_hosted_audit_anchor_v1,
        build_synthetic_hosted_run_v1,
        verify_hosted_audit_anchor_v1,
    )
    segments = build_synthetic_hosted_run_v1(n_segments=8)
    anchor = build_hosted_audit_anchor_v1(segments=segments)
    rep = verify_hosted_audit_anchor_v1(
        anchor=anchor, segments=segments)
    assert bool(rep.merkle_root_matches)
    assert int(rep.n_segments_verified) == 8


def test_w83_hosted_audit_anchor_detects_tampering():
    from coordpy.hosted_audit_anchoring_v1 import (
        build_hosted_audit_anchor_v1,
        build_synthetic_hosted_run_v1,
        verify_hosted_audit_anchor_v1,
        HostedTranscriptSegmentV1,
        W83_HAA_V1_SCHEMA_VERSION,
    )
    segments = list(build_synthetic_hosted_run_v1(
        n_segments=6))
    anchor = build_hosted_audit_anchor_v1(segments=segments)
    # Replace one segment with a tampered version (different
    # response_cid).
    tampered = HostedTranscriptSegmentV1(
        schema=W83_HAA_V1_SCHEMA_VERSION,
        segment_id=segments[2].segment_id,
        role=segments[2].role,
        provider_id=segments[2].provider_id,
        model_id=segments[2].model_id,
        prompt_cid=segments[2].prompt_cid,
        response_cid="tampered-response-cid",
        cache_hit=segments[2].cache_hit,
        logprob_observed=segments[2].logprob_observed,
        timestamp_ns=segments[2].timestamp_ns,
    )
    tampered_segments = list(segments)
    tampered_segments[2] = tampered
    rep = verify_hosted_audit_anchor_v1(
        anchor=anchor, segments=tampered_segments)
    # The rebuilt Merkle root must NOT match.
    assert not bool(rep.merkle_root_matches)


def test_w83_hosted_audit_witness_emitted():
    from coordpy.hosted_audit_anchoring_v1 import (
        build_hosted_audit_anchor_v1,
        build_synthetic_hosted_run_v1,
        emit_hosted_audit_anchoring_witness_v1,
        verify_hosted_audit_anchor_v1,
    )
    segments = build_synthetic_hosted_run_v1(n_segments=4)
    anchor = build_hosted_audit_anchor_v1(segments=segments)
    rep = verify_hosted_audit_anchor_v1(
        anchor=anchor, segments=segments)
    w = emit_hosted_audit_anchoring_witness_v1(
        anchor=anchor, verification=rep)
    assert len(w.cid()) == 64
    assert bool(w.merkle_root_matches)
