"""W83 — Hosted Audit Anchoring V1.

CoordPy's hosted plane (``hosted_router_controller_v12``,
``hosted_logprob_router_v12``, ``hosted_real_handoff_coordinator
_v11``) cannot pierce hosted-model substrate by design. But the
hosted-plane is still load-bearing in deployment: it controls
routing decisions, cache hits, cost budgets, and abstain floors
for a multi-agent run that ultimately commits to a value visible
to the user.

W82 ships content-addressed primitives — ``MerkleHashTreeV1``,
``RollbackAnchorV1``, ``StateSnapshotV1`` — that have so far been
used only by the substrate / event-graph lines. W83 V1 makes a
*hosted-plane* advance by anchoring hosted transcripts in those
primitives. Concretely:

1. Each hosted-routing decision is wrapped in a content-
   addressed ``HostedTranscriptSegmentV1``.
2. A run's segments are gathered into a Merkle tree; the root
   provides an *audit anchor* for the hosted run.
3. A ``RollbackAnchorV1`` is emitted pointing at a labelled
   segment (e.g. the first commit, the last successful cache
   hit, etc.); any later replay can verify integrity against
   the anchor.

This is the W83 hosted-control-plane gain: hosted runs now ship
the same audit-verifiability we already had for the controlled
substrate. A consumer of a hosted run can re-hash the transcript
segments and verify they hash to the recorded root *without*
piercing hosted substrate — this is a deployment-realism win.

The W83 V1 hosted anchor:

* does NOT pierce hosted-model internals — the audit is over the
  *hosted text/logprob/prefix-cache surface* only.
* does NOT require the hosted provider to participate — the
  audit is constructed client-side from observed responses.
* DOES provide a tamper-evident chain that downstream consumers
  can verify.

Honest scope (W83)
------------------

* ``W83-L-HOSTED-AUDIT-ANCHORING-V1-RESEARCH-ONLY-CAP`` —
  explicit-import only.
* ``W83-L-HOSTED-AUDIT-ANCHORING-V1-CLIENT-SIDE-CAP`` — the
  audit is constructed from observed hosted responses; if the
  hosted provider returns inconsistent responses, the audit
  records the inconsistency but cannot prevent it.
* ``W83-L-HOSTED-AUDIT-ANCHORING-V1-NO-SUBSTRATE-CAP`` — the
  W79 "no third-party substrate coupling" cap carries forward.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import time as _time
from typing import Any, Mapping, Sequence

from .cryptographic_state_integrity_v1 import (
    IntegrityVerdict,
    MerkleHashTreeV1,
    RollbackAnchorV1,
    StateSnapshotV1,
    W82_INTEGRITY_V1_SCHEMA_VERSION,
    build_state_snapshot_v1,
    verify_snapshot_integrity_v1,
)


W83_HAA_V1_SCHEMA_VERSION: str = (
    "coordpy.hosted_audit_anchoring_v1.v1")

W83_HAA_DEFAULT_HMAC_KEY: bytes = (
    b"w83-hosted-audit-anchoring-v1-default-hmac-key")


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass(frozen=True)
class HostedTranscriptSegmentV1:
    """A single hosted-plane segment: a routing decision +
    observed response."""

    schema: str
    segment_id: str
    role: str
    provider_id: str
    model_id: str
    prompt_cid: str
    response_cid: str
    cache_hit: bool
    logprob_observed: bool
    timestamp_ns: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "segment_id": str(self.segment_id),
            "role": str(self.role),
            "provider_id": str(self.provider_id),
            "model_id": str(self.model_id),
            "prompt_cid": str(self.prompt_cid),
            "response_cid": str(self.response_cid),
            "cache_hit": bool(self.cache_hit),
            "logprob_observed": bool(self.logprob_observed),
            "timestamp_ns": int(self.timestamp_ns),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w83_hosted_transcript_segment_v1",
            "segment": self.to_dict()})

    def to_snapshot_payload(self) -> bytes:
        return _canonical_bytes({
            "kind": "w83_hosted_transcript_segment_v1_payload",
            "segment": self.to_dict()})


def build_hosted_transcript_segment_v1(
        *,
        segment_id: str,
        role: str,
        provider_id: str,
        model_id: str,
        prompt_bytes: bytes,
        response_bytes: bytes,
        cache_hit: bool,
        logprob_observed: bool,
        timestamp_ns: int,
) -> HostedTranscriptSegmentV1:
    return HostedTranscriptSegmentV1(
        schema=W83_HAA_V1_SCHEMA_VERSION,
        segment_id=str(segment_id),
        role=str(role),
        provider_id=str(provider_id),
        model_id=str(model_id),
        prompt_cid=hashlib.sha256(
            bytes(prompt_bytes)).hexdigest(),
        response_cid=hashlib.sha256(
            bytes(response_bytes)).hexdigest(),
        cache_hit=bool(cache_hit),
        logprob_observed=bool(logprob_observed),
        timestamp_ns=int(timestamp_ns),
    )


@dataclasses.dataclass(frozen=True)
class HostedAuditAnchorV1:
    """Aggregated hosted-plane audit anchor."""

    schema: str
    segment_cids: tuple[str, ...]
    segment_snapshot_cids: tuple[str, ...]
    merkle_root_cid: str
    rollback_anchor: RollbackAnchorV1
    chain_parent_cid: str
    n_segments: int
    n_cache_hits: int
    n_logprob_observed: int
    anchor_chain_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "n_segments": int(self.n_segments),
            "merkle_root_cid": str(self.merkle_root_cid),
            "rollback_anchor_cid": str(
                self.rollback_anchor.cid()),
            "chain_parent_cid": str(self.chain_parent_cid),
            "n_cache_hits": int(self.n_cache_hits),
            "n_logprob_observed": int(
                self.n_logprob_observed),
            "anchor_chain_cid": str(self.anchor_chain_cid),
            "segment_cids": list(self.segment_cids),
            "segment_snapshot_cids": list(
                self.segment_snapshot_cids),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w83_hosted_audit_anchor_v1",
            "anchor": self.to_dict()})


def build_hosted_audit_anchor_v1(
        *,
        segments: Sequence[HostedTranscriptSegmentV1],
        chain_parent_cid: str = "genesis",
        rollback_label: str = "w83_hosted_audit_anchor",
        hmac_key: bytes = W83_HAA_DEFAULT_HMAC_KEY,
) -> HostedAuditAnchorV1:
    """Aggregate a list of segments into a Merkle-rooted anchor.

    Each segment is wrapped in a content-addressed StateSnapshotV1
    (HMAC-signed if ``hmac_key`` is non-empty). The Merkle root
    is built over the segment snapshot CIDs. The rollback anchor
    points at the first segment in the chain.
    """
    if len(segments) == 0:
        empty_root = _sha256_hex({
            "kind": "w83_hosted_audit_anchor_v1_empty"})
        empty_anchor = RollbackAnchorV1(
            schema=W82_INTEGRITY_V1_SCHEMA_VERSION,
            label=str(rollback_label),
            snapshot_cid="empty",
            chain_root_cid=str(empty_root),
            created_at_ns=int(_time.time_ns()))
        return HostedAuditAnchorV1(
            schema=W83_HAA_V1_SCHEMA_VERSION,
            segment_cids=tuple(),
            segment_snapshot_cids=tuple(),
            merkle_root_cid=str(empty_root),
            rollback_anchor=empty_anchor,
            chain_parent_cid=str(chain_parent_cid),
            n_segments=0,
            n_cache_hits=0,
            n_logprob_observed=0,
            anchor_chain_cid=str(empty_root),
        )
    parent_cid = str(chain_parent_cid)
    snapshot_cids: list[str] = []
    segment_cids: list[str] = []
    n_cache_hits = 0
    n_logprob = 0
    for seg in segments:
        payload = seg.to_snapshot_payload()
        snap = build_state_snapshot_v1(
            snapshot_id=str(seg.segment_id),
            parent_cid=str(parent_cid),
            payload_bytes=bytes(payload),
            timestamp_ns=int(seg.timestamp_ns),
            hmac_key=(bytes(hmac_key) if hmac_key else None))
        snapshot_cids.append(str(snap.cid()))
        segment_cids.append(str(seg.cid()))
        parent_cid = str(snap.cid())
        if bool(seg.cache_hit):
            n_cache_hits += 1
        if bool(seg.logprob_observed):
            n_logprob += 1
    merkle = MerkleHashTreeV1.from_snapshot_cids(
        snapshot_cids)
    rollback_anchor = RollbackAnchorV1(
        schema=W82_INTEGRITY_V1_SCHEMA_VERSION,
        label=str(rollback_label),
        snapshot_cid=str(snapshot_cids[0]),
        chain_root_cid=str(merkle.root_cid),
        created_at_ns=int(_time.time_ns()))
    anchor_chain_cid = _sha256_hex({
        "kind": "w83_hosted_audit_anchor_chain_v1",
        "merkle_root_cid": str(merkle.root_cid),
        "rollback_anchor_cid": str(rollback_anchor.cid()),
        "segment_cids": list(segment_cids),
        "chain_parent_cid": str(chain_parent_cid),
    })
    return HostedAuditAnchorV1(
        schema=W83_HAA_V1_SCHEMA_VERSION,
        segment_cids=tuple(segment_cids),
        segment_snapshot_cids=tuple(snapshot_cids),
        merkle_root_cid=str(merkle.root_cid),
        rollback_anchor=rollback_anchor,
        chain_parent_cid=str(chain_parent_cid),
        n_segments=int(len(segments)),
        n_cache_hits=int(n_cache_hits),
        n_logprob_observed=int(n_logprob),
        anchor_chain_cid=str(anchor_chain_cid),
    )


@dataclasses.dataclass(frozen=True)
class HostedAuditVerificationReportV1:
    """Verification report for a hosted audit anchor."""

    schema: str
    anchor_cid: str
    n_segments_verified: int
    n_segments_corrupt: int
    n_segments_unsigned: int
    n_segments_provenance_violation: int
    merkle_root_matches: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "anchor_cid": str(self.anchor_cid),
            "n_segments_verified": int(
                self.n_segments_verified),
            "n_segments_corrupt": int(
                self.n_segments_corrupt),
            "n_segments_unsigned": int(
                self.n_segments_unsigned),
            "n_segments_provenance_violation": int(
                self.n_segments_provenance_violation),
            "merkle_root_matches": bool(
                self.merkle_root_matches),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind":
                "w83_hosted_audit_verification_report_v1",
            "report": self.to_dict()})


def verify_hosted_audit_anchor_v1(
        *,
        anchor: HostedAuditAnchorV1,
        segments: Sequence[HostedTranscriptSegmentV1],
        chain_parent_cid: str | None = None,
        hmac_key: bytes = W83_HAA_DEFAULT_HMAC_KEY,
) -> HostedAuditVerificationReportV1:
    """Re-verify the audit anchor from its segments.

    Rebuilds the per-segment snapshots from the segment payloads
    + the recorded chain parent, re-verifies each, and compares
    the rebuilt Merkle root to the anchor's recorded root.
    """
    parent_cid = (
        str(chain_parent_cid)
        if chain_parent_cid is not None
        else str(anchor.chain_parent_cid))
    n_ok = 0
    n_corrupt = 0
    n_unsigned = 0
    n_prov = 0
    rebuilt_snapshot_cids: list[str] = []
    for seg in segments:
        snap = build_state_snapshot_v1(
            snapshot_id=str(seg.segment_id),
            parent_cid=str(parent_cid),
            payload_bytes=bytes(
                seg.to_snapshot_payload()),
            timestamp_ns=int(seg.timestamp_ns),
            hmac_key=(bytes(hmac_key) if hmac_key else None))
        rep = verify_snapshot_integrity_v1(
            snapshot=snap,
            chain_root_cid=str(anchor.merkle_root_cid),
            expected_parent_cid=str(parent_cid),
            hmac_key=(bytes(hmac_key) if hmac_key else None))
        verdict = str(rep.verdict)
        if verdict == IntegrityVerdict.OK.value:
            n_ok += 1
        elif verdict == IntegrityVerdict.CORRUPT.value:
            n_corrupt += 1
        elif verdict == IntegrityVerdict.UNSIGNED.value:
            n_unsigned += 1
        elif verdict == (
                IntegrityVerdict.PROVENANCE_VIOLATION.value):
            n_prov += 1
        rebuilt_snapshot_cids.append(str(snap.cid()))
        parent_cid = str(snap.cid())
    rebuilt_merkle = MerkleHashTreeV1.from_snapshot_cids(
        rebuilt_snapshot_cids)
    return HostedAuditVerificationReportV1(
        schema=W83_HAA_V1_SCHEMA_VERSION,
        anchor_cid=str(anchor.cid()),
        n_segments_verified=int(n_ok),
        n_segments_corrupt=int(n_corrupt),
        n_segments_unsigned=int(n_unsigned),
        n_segments_provenance_violation=int(n_prov),
        merkle_root_matches=bool(
            str(rebuilt_merkle.root_cid)
            == str(anchor.merkle_root_cid)),
    )


def build_synthetic_hosted_run_v1(
        *,
        n_segments: int = 12,
        providers: Sequence[str] = (
            "openrouter_paid", "openai_paid"),
        model_ids: Sequence[str] = (
            "claude-haiku", "gpt-4o-mini"),
        seed: int = 83_008_001,
) -> list[HostedTranscriptSegmentV1]:
    """Build a deterministic synthetic hosted run for the bench."""
    import random
    rnd = random.Random(int(seed))
    out: list[HostedTranscriptSegmentV1] = []
    for i in range(int(n_segments)):
        provider = str(rnd.choice(list(providers)))
        model = str(rnd.choice(list(model_ids)))
        prompt_bytes = json.dumps({
            "role": f"role_{i % 3}",
            "step": int(i),
            "seed": int(seed),
        }).encode("utf-8")
        response_bytes = json.dumps({
            "step": int(i),
            "tokens": [int(rnd.randint(0, 5000)) for _ in range(8)],
        }).encode("utf-8")
        cache_hit = bool(rnd.random() < 0.4)
        logprob_observed = bool(rnd.random() < 0.6)
        out.append(build_hosted_transcript_segment_v1(
            segment_id=f"seg{i}",
            role=f"role_{i % 3}",
            provider_id=provider,
            model_id=model,
            prompt_bytes=prompt_bytes,
            response_bytes=response_bytes,
            cache_hit=bool(cache_hit),
            logprob_observed=bool(logprob_observed),
            timestamp_ns=int(_time.time_ns()) + int(i),
        ))
    return out


@dataclasses.dataclass(frozen=True)
class HostedAuditAnchoringWitnessV1:
    schema: str
    anchor_cid: str
    verification_cid: str
    merkle_root_matches: bool
    n_segments: int

    def cid(self) -> str:
        return _sha256_hex({
            "kind":
                "w83_hosted_audit_anchoring_witness_v1",
            "schema": str(self.schema),
            "anchor_cid": str(self.anchor_cid),
            "verification_cid": str(self.verification_cid),
            "merkle_root_matches": bool(
                self.merkle_root_matches),
            "n_segments": int(self.n_segments),
        })


def emit_hosted_audit_anchoring_witness_v1(
        *,
        anchor: HostedAuditAnchorV1,
        verification: HostedAuditVerificationReportV1,
) -> HostedAuditAnchoringWitnessV1:
    return HostedAuditAnchoringWitnessV1(
        schema=W83_HAA_V1_SCHEMA_VERSION,
        anchor_cid=str(anchor.cid()),
        verification_cid=str(verification.cid()),
        merkle_root_matches=bool(
            verification.merkle_root_matches),
        n_segments=int(anchor.n_segments),
    )


__all__ = [
    "W83_HAA_V1_SCHEMA_VERSION",
    "W83_HAA_DEFAULT_HMAC_KEY",
    "HostedTranscriptSegmentV1",
    "HostedAuditAnchorV1",
    "HostedAuditVerificationReportV1",
    "HostedAuditAnchoringWitnessV1",
    "build_hosted_transcript_segment_v1",
    "build_hosted_audit_anchor_v1",
    "verify_hosted_audit_anchor_v1",
    "build_synthetic_hosted_run_v1",
    "emit_hosted_audit_anchoring_witness_v1",
]
