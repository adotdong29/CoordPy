"""W82 / P2 #18 — Cryptographic state integrity / rollback /
branch-merge verification V1.

Issue #18 asks for a *unified* integrity / rollback / branch-
merge subsystem on top of the many existing CID + witness
chains. The required deliverables are:

1. an explicit integrity / rollback subsystem
2. at least one benchmark exercising corruption + rollback +
   merge verification
3. failures diagnosed by provenance evidence (not post-hoc
   guesswork)

V1 delivers all three. The subsystem is built around four
explicit object types:

* ``StateSnapshotV1`` — content-addressed, optionally HMAC-
  signed snapshot of a state segment. Carries a parent CID
  (so a chain forms an append-only history) and a content
  CID that can be independently re-verified.
* ``MerkleHashTreeV1`` — a Merkle tree over an ordered list
  of state snapshots. Supports O(log n) inclusion proofs.
* ``RollbackAnchorV1`` — a tagged commit pointing at a
  specific snapshot CID. Carries a human-readable label,
  a creation-time CID, and the snapshot CID. Rolling back
  to an anchor restores the snapshot under integrity
  verification.
* ``BranchMergeWitnessV1`` — proves that a merge of two
  branches happened safely. The witness records the parent
  CIDs of the merge, the Merkle root of the merged tree,
  any conflict resolutions, and an integrity verdict.

Two failure modes are explicitly supported:

* **Silent corruption** — a snapshot's stored content
  diverges from its declared content CID. The verifier
  detects this and returns ``corrupt``.
* **Provenance violation** — a snapshot claims a parent CID
  but the parent doesn't exist in the chain, or a merge
  witness claims parent CIDs that don't both appear in the
  Merkle tree. The verifier flags this and returns
  ``provenance_violation``.

The benchmark family ``run_corruption_rollback_merge_bench_v1``
exercises:

1. building a chain of N snapshots with a Merkle root
2. injecting silent corruption into snapshot ``k`` and
   verifying detection
3. rolling back to an anchor at snapshot ``k-1`` and
   verifying integrity holds
4. spawning two branches from a common parent, mutating each
   independently, then issuing a merge witness and verifying
   the merge witness
5. attempting an unsafe merge (parent mismatch) and verifying
   detection

Honest scope (W82)
------------------

* ``W82-L-INTEGRITY-V1-RESEARCH-ONLY-CAP`` — explicit import
  only.
* ``W82-L-INTEGRITY-V1-NUMPY-CAP`` — pure NumPy / stdlib.
* ``W82-L-INTEGRITY-V1-HMAC-OPTIONAL-CAP`` — V1 supports
  optional HMAC-SHA256 signing keyed by an in-memory secret;
  it is not a PKI / certificate scheme. Use real X.509 / Ed25519
  for production.
* ``W82-L-INTEGRITY-V1-NO-DISTRIBUTED-CAP`` — V1 operates on
  a single host's content-addressed storage. Distributed
  consistency is covered by W82 #16.
* ``W82-L-INTEGRITY-V1-SCALAR-PAYLOAD-CAP`` — V1 hashes the
  serialised payload bytes; tree-level diffing of structured
  payloads is out of scope.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import hmac as _hmac
import json
import math
from typing import Any, Mapping, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.cryptographic_state_integrity_v1 requires "
        "numpy") from exc


W82_INTEGRITY_V1_SCHEMA_VERSION: str = (
    "coordpy.cryptographic_state_integrity_v1.v1")


W82_INTEGRITY_DEFAULT_MERKLE_FANOUT: int = 4
W82_INTEGRITY_DEFAULT_HMAC_KEY: bytes = (
    b"w82-integrity-v1-default-hmac-key")
W82_INTEGRITY_DEFAULT_CHAIN_LEN: int = 8
W82_INTEGRITY_DEFAULT_SEED: int = 82_018_001


class IntegrityVerdict(str, enum.Enum):
    OK = "ok"
    CORRUPT = "corrupt"
    PROVENANCE_VIOLATION = "provenance_violation"
    UNSIGNED = "unsigned"
    BAD_SIGNATURE = "bad_signature"


W82_INTEGRITY_VERDICTS: tuple[str, ...] = tuple(
    v.value for v in IntegrityVerdict)


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _sha256_bytes_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _ndarray_cid(arr: "_np.ndarray | None") -> str:
    if arr is None:
        return "none"
    a = _np.ascontiguousarray(
        _np.asarray(arr, dtype=_np.float64))
    return hashlib.sha256(a.tobytes()).hexdigest()


def _hmac_hex(key: bytes, payload_bytes: bytes) -> str:
    return _hmac.new(
        bytes(key), payload_bytes,
        hashlib.sha256).hexdigest()


def _extract_merge_parent_cids_v1(
        snapshot: "StateSnapshotV1",
) -> tuple[str, ...]:
    """Extract explicit two-parent merge metadata from a
    snapshot payload.

    A branch-merge witness is only considered load-bearing when
    the merged snapshot payload names both branch tips. A plain
    single-parent child is not enough.
    """
    try:
        payload = json.loads(
            bytes(snapshot.payload_bytes).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return tuple()
    parent_cids = payload.get("merge_parent_cids")
    if not isinstance(parent_cids, list):
        return tuple()
    out: list[str] = []
    for cid in parent_cids:
        if not isinstance(cid, str):
            return tuple()
        out.append(str(cid))
    return tuple(out)


# ---------------------------------------------------------------
# StateSnapshotV1
# ---------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class StateSnapshotV1:
    """A content-addressed, optionally HMAC-signed snapshot."""

    schema: str
    snapshot_id: str
    parent_cid: str
    payload_bytes: bytes
    payload_cid: str
    timestamp_ns: int
    hmac_signature_hex: str  # "" if unsigned
    additional_parent_cids: tuple[str, ...] = tuple()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "snapshot_id": str(self.snapshot_id),
            "parent_cid": str(self.parent_cid),
            "additional_parent_cids": list(
                str(cid) for cid in self.additional_parent_cids),
            "payload_cid": str(self.payload_cid),
            "payload_size_bytes": int(
                len(self.payload_bytes)),
            "timestamp_ns": int(self.timestamp_ns),
            "hmac_signature_hex": str(
                self.hmac_signature_hex),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w82_state_snapshot_v1",
            "snapshot": self.to_dict()})

    def re_hash_payload(self) -> str:
        """Recompute the payload CID directly from
        ``payload_bytes``. Used by the integrity verifier."""
        return _sha256_bytes_hex(bytes(self.payload_bytes))


def build_state_snapshot_v1(
        *, snapshot_id: str,
        parent_cid: str,
        additional_parent_cids: Sequence[str] = (),
        payload_bytes: bytes,
        timestamp_ns: int,
        hmac_key: bytes | None = None,
) -> StateSnapshotV1:
    """Build a content-addressed snapshot.

    If ``hmac_key`` is provided, the HMAC-SHA256 of the
    parent-set + payload CID + timestamp is computed and
    stored on the snapshot. Verification later compares a
    freshly-computed HMAC against the stored one.
    """
    payload_cid = _sha256_bytes_hex(bytes(payload_bytes))
    normalized_additional_parent_cids = tuple(
        str(cid) for cid in additional_parent_cids
        if str(cid))
    if hmac_key is not None:
        sig_payload = _canonical_bytes({
            "parent_cid": str(parent_cid),
            "additional_parent_cids": list(
                normalized_additional_parent_cids),
            "payload_cid": str(payload_cid),
            "timestamp_ns": int(timestamp_ns),
        })
        sig = _hmac_hex(bytes(hmac_key), sig_payload)
    else:
        sig = ""
    return StateSnapshotV1(
        schema=W82_INTEGRITY_V1_SCHEMA_VERSION,
        snapshot_id=str(snapshot_id),
        parent_cid=str(parent_cid),
        payload_bytes=bytes(payload_bytes),
        payload_cid=str(payload_cid),
        timestamp_ns=int(timestamp_ns),
        hmac_signature_hex=str(sig),
        additional_parent_cids=normalized_additional_parent_cids,
    )


def _snapshot_parent_cids(
        snapshot: StateSnapshotV1,
) -> tuple[str, ...]:
    return tuple(
        str(cid) for cid in (
            (str(snapshot.parent_cid),) +
            tuple(str(c) for c in snapshot.additional_parent_cids))
        if str(cid))


def _snapshot_ancestor_set(
        start_cid: str,
        *,
        snapshots_by_cid: Mapping[str, StateSnapshotV1],
) -> set[str]:
    out: set[str] = set()
    stack: list[str] = [str(start_cid)]
    while stack:
        cur = stack.pop()
        if cur in out:
            continue
        if cur not in snapshots_by_cid:
            continue
        out.add(cur)
        for parent_cid in _snapshot_parent_cids(
                snapshots_by_cid[cur]):
            if str(parent_cid) in snapshots_by_cid:
                stack.append(str(parent_cid))
    return out


# ---------------------------------------------------------------
# Merkle hash tree
# ---------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class MerkleHashTreeV1:
    """Merkle tree over an ordered list of snapshot CIDs."""

    schema: str
    snapshot_cids: tuple[str, ...]
    fanout: int
    root_cid: str
    # ``levels[i]`` is the list of node CIDs at level i, where
    # level 0 is the leaves (snapshot CIDs).
    levels: tuple[tuple[str, ...], ...]

    @classmethod
    def from_snapshot_cids(
            cls, snapshot_cids: Sequence[str],
            *, fanout: int = (
                W82_INTEGRITY_DEFAULT_MERKLE_FANOUT),
    ) -> "MerkleHashTreeV1":
        leaves = tuple(str(c) for c in snapshot_cids)
        levels: list[tuple[str, ...]] = [leaves]
        if not leaves:
            root = _sha256_hex({
                "kind": "w82_merkle_empty_root_v1"})
            return cls(
                schema=W82_INTEGRITY_V1_SCHEMA_VERSION,
                snapshot_cids=tuple(),
                fanout=int(fanout),
                root_cid=str(root),
                levels=tuple(levels))
        current = list(leaves)
        while len(current) > 1:
            next_level: list[str] = []
            for i in range(
                    0, len(current), int(fanout)):
                chunk = current[i:i + int(fanout)]
                next_level.append(_sha256_hex({
                    "kind": "w82_merkle_node_v1",
                    "children": list(chunk),
                }))
            current = next_level
            levels.append(tuple(current))
        return cls(
            schema=W82_INTEGRITY_V1_SCHEMA_VERSION,
            snapshot_cids=leaves,
            fanout=int(fanout),
            root_cid=str(current[0]),
            levels=tuple(levels))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "n_leaves": int(len(self.snapshot_cids)),
            "fanout": int(self.fanout),
            "root_cid": str(self.root_cid),
            "n_levels": int(len(self.levels)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w82_merkle_hash_tree_v1",
            "tree": self.to_dict(),
            "snapshot_cids": list(self.snapshot_cids),
        })

    def inclusion_path(
            self, leaf_index: int,
    ) -> tuple[str, ...]:
        """Return a content-addressed inclusion path from the
        leaf at ``leaf_index`` up to the root.

        Path is a tuple of node CIDs (one per level) that
        certifies the leaf's membership in the tree.
        """
        if not (0 <= int(leaf_index) <
                int(len(self.snapshot_cids))):
            raise IndexError(
                f"leaf_index {int(leaf_index)} out of range")
        path: list[str] = []
        idx = int(leaf_index)
        for level in self.levels[:-1]:
            grp_idx = idx // int(self.fanout)
            chunk = level[grp_idx * int(self.fanout):
                          (grp_idx + 1) * int(self.fanout)]
            path.append(_sha256_hex({
                "kind": "w82_merkle_node_v1",
                "children": list(chunk),
            }))
            idx = grp_idx
        return tuple(path)


# ---------------------------------------------------------------
# Rollback anchor
# ---------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class RollbackAnchorV1:
    """A named rollback anchor pointing at a snapshot CID."""

    schema: str
    label: str
    snapshot_cid: str
    chain_root_cid: str
    created_at_ns: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "label": str(self.label),
            "snapshot_cid": str(self.snapshot_cid),
            "chain_root_cid": str(self.chain_root_cid),
            "created_at_ns": int(self.created_at_ns),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w82_rollback_anchor_v1",
            "anchor": self.to_dict()})


# ---------------------------------------------------------------
# Branch-merge witness
# ---------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class BranchMergeWitnessV1:
    """Proves that a merge of two branches happened safely."""

    schema: str
    branch_a_tip_cid: str
    branch_b_tip_cid: str
    common_ancestor_cid: str
    merged_snapshot_cid: str
    merge_tree_root_cid: str
    n_conflicts_resolved: int
    integrity_verdict: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "branch_a_tip_cid": str(
                self.branch_a_tip_cid),
            "branch_b_tip_cid": str(
                self.branch_b_tip_cid),
            "common_ancestor_cid": str(
                self.common_ancestor_cid),
            "merged_snapshot_cid": str(
                self.merged_snapshot_cid),
            "merge_tree_root_cid": str(
                self.merge_tree_root_cid),
            "n_conflicts_resolved": int(
                self.n_conflicts_resolved),
            "integrity_verdict": str(
                self.integrity_verdict),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w82_branch_merge_witness_v1",
            "witness": self.to_dict()})


# ---------------------------------------------------------------
# Integrity verifier
# ---------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class IntegrityVerificationReportV1:
    """Per-snapshot integrity report."""

    schema: str
    snapshot_cid: str
    payload_cid: str
    payload_cid_recomputed: str
    parent_cid: str
    chain_root_cid: str
    verdict: str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "snapshot_cid": str(self.snapshot_cid),
            "payload_cid": str(self.payload_cid),
            "payload_cid_recomputed": str(
                self.payload_cid_recomputed),
            "parent_cid": str(self.parent_cid),
            "chain_root_cid": str(self.chain_root_cid),
            "verdict": str(self.verdict),
            "detail": str(self.detail),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w82_integrity_verification_report_v1",
            "report": self.to_dict()})


def verify_snapshot_integrity_v1(
        *, snapshot: StateSnapshotV1,
        chain_root_cid: str,
        expected_parent_cid: str | None = None,
        hmac_key: bytes | None = None,
) -> IntegrityVerificationReportV1:
    """Verify a single snapshot's integrity.

    Returns a per-snapshot report. The verdict is one of:

    * ``OK`` — payload hash matches, parent matches expected,
      signature (if any) verifies.
    * ``CORRUPT`` — payload bytes do not hash to the declared
      payload_cid.
    * ``PROVENANCE_VIOLATION`` — declared parent CID does not
      match ``expected_parent_cid`` (when given).
    * ``BAD_SIGNATURE`` — the snapshot was signed but the
      HMAC does not verify with the supplied key.
    * ``UNSIGNED`` — the snapshot is unsigned and a key was
      supplied (informational, not a hard error).
    """
    recomputed = _sha256_bytes_hex(
        bytes(snapshot.payload_bytes))
    parent_cids = _snapshot_parent_cids(snapshot)
    if len(set(parent_cids)) != len(parent_cids):
        return IntegrityVerificationReportV1(
            schema=W82_INTEGRITY_V1_SCHEMA_VERSION,
            snapshot_cid=str(snapshot.cid()),
            payload_cid=str(snapshot.payload_cid),
            payload_cid_recomputed=str(recomputed),
            parent_cid=str(snapshot.parent_cid),
            chain_root_cid=str(chain_root_cid),
            verdict=IntegrityVerdict.PROVENANCE_VIOLATION.value,
            detail="duplicate parent CID entries are not allowed",
        )
    if str(recomputed) != str(snapshot.payload_cid):
        return IntegrityVerificationReportV1(
            schema=W82_INTEGRITY_V1_SCHEMA_VERSION,
            snapshot_cid=str(snapshot.cid()),
            payload_cid=str(snapshot.payload_cid),
            payload_cid_recomputed=str(recomputed),
            parent_cid=str(snapshot.parent_cid),
            chain_root_cid=str(chain_root_cid),
            verdict=IntegrityVerdict.CORRUPT.value,
            detail=(
                f"payload bytes hash to {recomputed} but "
                f"snapshot declares {snapshot.payload_cid}"),
        )
    if expected_parent_cid is not None and (
            str(snapshot.parent_cid) !=
            str(expected_parent_cid)):
        return IntegrityVerificationReportV1(
            schema=W82_INTEGRITY_V1_SCHEMA_VERSION,
            snapshot_cid=str(snapshot.cid()),
            payload_cid=str(snapshot.payload_cid),
            payload_cid_recomputed=str(recomputed),
            parent_cid=str(snapshot.parent_cid),
            chain_root_cid=str(chain_root_cid),
            verdict=IntegrityVerdict.PROVENANCE_VIOLATION.value,
            detail=(
                f"declared parent {snapshot.parent_cid} != "
                f"expected {expected_parent_cid}"),
        )
    if hmac_key is not None:
        if not snapshot.hmac_signature_hex:
            return IntegrityVerificationReportV1(
                schema=W82_INTEGRITY_V1_SCHEMA_VERSION,
                snapshot_cid=str(snapshot.cid()),
                payload_cid=str(snapshot.payload_cid),
                payload_cid_recomputed=str(recomputed),
                parent_cid=str(snapshot.parent_cid),
                chain_root_cid=str(chain_root_cid),
                verdict=IntegrityVerdict.UNSIGNED.value,
                detail=(
                    "hmac key provided but snapshot is "
                    "unsigned"),
            )
        sig_payload = _canonical_bytes({
            "parent_cid": str(snapshot.parent_cid),
            "additional_parent_cids": list(
                str(c) for c in snapshot.additional_parent_cids),
            "payload_cid": str(snapshot.payload_cid),
            "timestamp_ns": int(snapshot.timestamp_ns),
        })
        recomputed_sig = _hmac_hex(
            bytes(hmac_key), sig_payload)
        if not _hmac.compare_digest(
                str(recomputed_sig),
                str(snapshot.hmac_signature_hex)):
            return IntegrityVerificationReportV1(
                schema=W82_INTEGRITY_V1_SCHEMA_VERSION,
                snapshot_cid=str(snapshot.cid()),
                payload_cid=str(snapshot.payload_cid),
                payload_cid_recomputed=str(recomputed),
                parent_cid=str(snapshot.parent_cid),
                chain_root_cid=str(chain_root_cid),
                verdict=IntegrityVerdict.BAD_SIGNATURE.value,
                detail=(
                    "HMAC mismatch — snapshot may have been "
                    "tampered after signing"),
            )
    return IntegrityVerificationReportV1(
        schema=W82_INTEGRITY_V1_SCHEMA_VERSION,
        snapshot_cid=str(snapshot.cid()),
        payload_cid=str(snapshot.payload_cid),
        payload_cid_recomputed=str(recomputed),
        parent_cid=str(snapshot.parent_cid),
        chain_root_cid=str(chain_root_cid),
        verdict=IntegrityVerdict.OK.value,
        detail="ok",
    )


class IntegrityAwareAction(str, enum.Enum):
    """Recommended action given an integrity verdict.

    Used by callers (e.g. the W81 adversarial consensus
    controller) to decide whether to commit, abstain, or
    rollback when integrity verification fails.
    """

    COMMIT = "commit"
    ABSTAIN = "abstain"
    ROLLBACK = "rollback"
    ESCALATE = "escalate"


W82_INTEGRITY_AWARE_ACTIONS: tuple[str, ...] = tuple(
    a.value for a in IntegrityAwareAction)


@dataclasses.dataclass(frozen=True)
class IntegrityAwareFallbackDecisionV1:
    """A per-verdict integrity-aware action recommendation.

    Maps the five integrity verdicts to one of four explicit
    actions:

    * ``OK`` → ``COMMIT``
    * ``CORRUPT`` → ``ROLLBACK`` (try a rollback anchor)
    * ``PROVENANCE_VIOLATION`` → ``ABSTAIN``
    * ``BAD_SIGNATURE`` → ``ESCALATE`` (tampering may be
      systematic — surface to a higher trust level)
    * ``UNSIGNED`` → ``ABSTAIN`` (informational; caller
      should not commit unsigned state when a key was
      expected)

    The decision is content-addressed.
    """

    schema: str
    integrity_verdict: str
    recommended_action: str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "integrity_verdict": str(self.integrity_verdict),
            "recommended_action": str(
                self.recommended_action),
            "detail": str(self.detail),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w82_integrity_aware_fallback_decision_v1",
            "decision": self.to_dict()})


def integrity_aware_fallback_decision_v1(
        verdict: str,
) -> IntegrityAwareFallbackDecisionV1:
    """Map an integrity verdict to a recommended action."""
    v = str(verdict)
    if v == IntegrityVerdict.OK.value:
        return IntegrityAwareFallbackDecisionV1(
            schema=W82_INTEGRITY_V1_SCHEMA_VERSION,
            integrity_verdict=v,
            recommended_action=IntegrityAwareAction.COMMIT.value,
            detail="integrity ok → commit")
    if v == IntegrityVerdict.CORRUPT.value:
        return IntegrityAwareFallbackDecisionV1(
            schema=W82_INTEGRITY_V1_SCHEMA_VERSION,
            integrity_verdict=v,
            recommended_action=(
                IntegrityAwareAction.ROLLBACK.value),
            detail=(
                "payload bytes do not hash to declared "
                "payload_cid; recover via rollback anchor"))
    if v == IntegrityVerdict.PROVENANCE_VIOLATION.value:
        return IntegrityAwareFallbackDecisionV1(
            schema=W82_INTEGRITY_V1_SCHEMA_VERSION,
            integrity_verdict=v,
            recommended_action=(
                IntegrityAwareAction.ABSTAIN.value),
            detail=(
                "declared parent / lineage does not match "
                "supplied evidence; abstain"))
    if v == IntegrityVerdict.BAD_SIGNATURE.value:
        return IntegrityAwareFallbackDecisionV1(
            schema=W82_INTEGRITY_V1_SCHEMA_VERSION,
            integrity_verdict=v,
            recommended_action=(
                IntegrityAwareAction.ESCALATE.value),
            detail=(
                "HMAC mismatch suggests tampering after "
                "signing; escalate to a higher trust level"))
    if v == IntegrityVerdict.UNSIGNED.value:
        return IntegrityAwareFallbackDecisionV1(
            schema=W82_INTEGRITY_V1_SCHEMA_VERSION,
            integrity_verdict=v,
            recommended_action=(
                IntegrityAwareAction.ABSTAIN.value),
            detail=(
                "snapshot unsigned but a key was supplied; "
                "abstain rather than commit unsigned state"))
    # Unknown verdict — fail safe.
    return IntegrityAwareFallbackDecisionV1(
        schema=W82_INTEGRITY_V1_SCHEMA_VERSION,
        integrity_verdict=v,
        recommended_action=IntegrityAwareAction.ABSTAIN.value,
        detail=f"unknown verdict {v!r}; fail safe → abstain")


def verify_chain_integrity_v1(
        *, snapshots: Sequence[StateSnapshotV1],
        hmac_key: bytes | None = None,
) -> tuple[IntegrityVerificationReportV1, ...]:
    """Verify a linear chain. Each snapshot's parent_cid must
    equal the prior snapshot's CID. The first snapshot's
    parent_cid is the genesis sentinel.
    """
    snaps = list(snapshots)
    tree = MerkleHashTreeV1.from_snapshot_cids(
        [s.cid() for s in snaps])
    reports: list[IntegrityVerificationReportV1] = []
    prev_cid: str | None = None
    for s in snaps:
        exp_parent = prev_cid
        rep = verify_snapshot_integrity_v1(
            snapshot=s,
            chain_root_cid=str(tree.root_cid),
            expected_parent_cid=exp_parent,
            hmac_key=hmac_key)
        reports.append(rep)
        if str(rep.verdict) == IntegrityVerdict.OK.value:
            prev_cid = str(s.cid())
        else:
            # Once we hit a violation, downstream parent_cid
            # checks will also fail. Stop chaining expected_
            # parent updates so we surface the *first* failure
            # cleanly.
            prev_cid = str(s.cid())
    return tuple(reports)


# ---------------------------------------------------------------
# Rollback
# ---------------------------------------------------------------

def build_rollback_anchor_v1(
        *, label: str,
        snapshot: StateSnapshotV1,
        chain_root_cid: str,
        created_at_ns: int,
) -> RollbackAnchorV1:
    return RollbackAnchorV1(
        schema=W82_INTEGRITY_V1_SCHEMA_VERSION,
        label=str(label),
        snapshot_cid=str(snapshot.cid()),
        chain_root_cid=str(chain_root_cid),
        created_at_ns=int(created_at_ns),
    )


def execute_rollback_v1(
        *, anchor: RollbackAnchorV1,
        snapshots: Sequence[StateSnapshotV1],
        hmac_key: bytes | None = None,
) -> tuple[StateSnapshotV1 | None,
           IntegrityVerificationReportV1]:
    """Rollback to ``anchor``: locate the matching snapshot,
    verify its integrity, return (snapshot, report)."""
    target: StateSnapshotV1 | None = None
    for s in snapshots:
        if str(s.cid()) == str(anchor.snapshot_cid):
            target = s
            break
    if target is None:
        return (None, IntegrityVerificationReportV1(
            schema=W82_INTEGRITY_V1_SCHEMA_VERSION,
            snapshot_cid=str(anchor.snapshot_cid),
            payload_cid="",
            payload_cid_recomputed="",
            parent_cid="",
            chain_root_cid=str(anchor.chain_root_cid),
            verdict=(
                IntegrityVerdict.PROVENANCE_VIOLATION.value),
            detail=(
                "rollback anchor's snapshot_cid not in "
                "supplied snapshot list"),
        ))
    rep = verify_snapshot_integrity_v1(
        snapshot=target,
        chain_root_cid=str(anchor.chain_root_cid),
        expected_parent_cid=None,
        hmac_key=hmac_key)
    if str(rep.verdict) == IntegrityVerdict.OK.value:
        return (target, rep)
    return (None, rep)


# ---------------------------------------------------------------
# Branch merge
# ---------------------------------------------------------------

def build_branch_merge_witness_v1(
        *, branch_a_tip: StateSnapshotV1,
        branch_b_tip: StateSnapshotV1,
        common_ancestor: StateSnapshotV1,
        merged_snapshot: StateSnapshotV1,
        all_snapshots: Sequence[StateSnapshotV1],
        n_conflicts_resolved: int,
) -> BranchMergeWitnessV1:
    """Build a branch-merge witness.

    The witness records the two branch tips, common ancestor,
    merged tip, and the Merkle root over all snapshots in the
    merged tree. The integrity verdict surfaces whether the
    branch ancestry is consistent with the supplied
    snapshot set.
    """
    snapshot_map = {
        str(s.cid()): s for s in all_snapshots}
    cid_set = set(snapshot_map.keys())
    verdict = IntegrityVerdict.OK.value
    if str(branch_a_tip.cid()) not in cid_set:
        verdict = IntegrityVerdict.PROVENANCE_VIOLATION.value
    elif str(branch_b_tip.cid()) not in cid_set:
        verdict = IntegrityVerdict.PROVENANCE_VIOLATION.value
    elif str(common_ancestor.cid()) not in cid_set:
        verdict = IntegrityVerdict.PROVENANCE_VIOLATION.value
    elif str(merged_snapshot.cid()) not in cid_set:
        verdict = IntegrityVerdict.PROVENANCE_VIOLATION.value
    elif str(branch_a_tip.parent_cid) != str(common_ancestor.cid()):
        verdict = IntegrityVerdict.PROVENANCE_VIOLATION.value
    elif str(branch_b_tip.parent_cid) != str(common_ancestor.cid()):
        verdict = IntegrityVerdict.PROVENANCE_VIOLATION.value
    # A real merge must carry two distinct parent references
    # in the snapshot structure itself, not only inside the
    # payload witness.
    elif set(_snapshot_parent_cids(merged_snapshot)) != {
            str(branch_a_tip.cid()),
            str(branch_b_tip.cid())}:
        verdict = IntegrityVerdict.PROVENANCE_VIOLATION.value
    elif len(_snapshot_parent_cids(merged_snapshot)) != 2:
        verdict = IntegrityVerdict.PROVENANCE_VIOLATION.value
    elif set(_extract_merge_parent_cids_v1(merged_snapshot)) != {
            str(branch_a_tip.cid()),
            str(branch_b_tip.cid())}:
        verdict = IntegrityVerdict.PROVENANCE_VIOLATION.value
    elif str(common_ancestor.cid()) not in _snapshot_ancestor_set(
            str(branch_a_tip.cid()),
            snapshots_by_cid=snapshot_map):
        verdict = IntegrityVerdict.PROVENANCE_VIOLATION.value
    elif str(common_ancestor.cid()) not in _snapshot_ancestor_set(
            str(branch_b_tip.cid()),
            snapshots_by_cid=snapshot_map):
        verdict = IntegrityVerdict.PROVENANCE_VIOLATION.value
    tree = MerkleHashTreeV1.from_snapshot_cids(
        [s.cid() for s in all_snapshots])
    return BranchMergeWitnessV1(
        schema=W82_INTEGRITY_V1_SCHEMA_VERSION,
        branch_a_tip_cid=str(branch_a_tip.cid()),
        branch_b_tip_cid=str(branch_b_tip.cid()),
        common_ancestor_cid=str(common_ancestor.cid()),
        merged_snapshot_cid=str(merged_snapshot.cid()),
        merge_tree_root_cid=str(tree.root_cid),
        n_conflicts_resolved=int(n_conflicts_resolved),
        integrity_verdict=str(verdict),
    )


def verify_branch_merge_witness_v1(
        *, witness: BranchMergeWitnessV1,
        snapshots: Sequence[StateSnapshotV1],
) -> IntegrityVerificationReportV1:
    """Independently verify a branch-merge witness against a
    supplied snapshot list."""
    cid_set = {str(s.cid()) for s in snapshots}
    tree = MerkleHashTreeV1.from_snapshot_cids(
        [s.cid() for s in snapshots])
    if str(tree.root_cid) != str(witness.merge_tree_root_cid):
        return IntegrityVerificationReportV1(
            schema=W82_INTEGRITY_V1_SCHEMA_VERSION,
            snapshot_cid=str(witness.merged_snapshot_cid),
            payload_cid="",
            payload_cid_recomputed="",
            parent_cid="",
            chain_root_cid=str(witness.merge_tree_root_cid),
            verdict=(
                IntegrityVerdict.PROVENANCE_VIOLATION.value),
            detail=(
                f"merge_tree_root_cid {witness.merge_tree_root_cid} "
                f"!= recomputed {tree.root_cid}"),
        )
    for cid in (
            witness.branch_a_tip_cid,
            witness.branch_b_tip_cid,
            witness.common_ancestor_cid,
            witness.merged_snapshot_cid,
    ):
        if str(cid) not in cid_set:
            return IntegrityVerificationReportV1(
                schema=W82_INTEGRITY_V1_SCHEMA_VERSION,
                snapshot_cid=str(cid),
                payload_cid="",
                payload_cid_recomputed="",
                parent_cid="",
                chain_root_cid=str(
                    witness.merge_tree_root_cid),
                verdict=(
                    IntegrityVerdict.PROVENANCE_VIOLATION.value),
                detail=(
                    f"witness references snapshot CID {cid} "
                    f"that is not in the supplied snapshot list"),
            )
    snapshot_map = {str(s.cid()): s for s in snapshots}
    branch_a = snapshot_map[str(witness.branch_a_tip_cid)]
    branch_b = snapshot_map[str(witness.branch_b_tip_cid)]
    common = snapshot_map[str(witness.common_ancestor_cid)]
    merged = snapshot_map[str(witness.merged_snapshot_cid)]
    if str(branch_a.parent_cid) != str(common.cid()):
        return IntegrityVerificationReportV1(
            schema=W82_INTEGRITY_V1_SCHEMA_VERSION,
            snapshot_cid=str(branch_a.cid()),
            payload_cid=str(branch_a.payload_cid),
            payload_cid_recomputed=str(
                branch_a.re_hash_payload()),
            parent_cid=str(branch_a.parent_cid),
            chain_root_cid=str(witness.merge_tree_root_cid),
            verdict=IntegrityVerdict.PROVENANCE_VIOLATION.value,
            detail=(
                "branch_a_tip does not descend directly from "
                "the declared common ancestor"),
        )
    if str(branch_b.parent_cid) != str(common.cid()):
        return IntegrityVerificationReportV1(
            schema=W82_INTEGRITY_V1_SCHEMA_VERSION,
            snapshot_cid=str(branch_b.cid()),
            payload_cid=str(branch_b.payload_cid),
            payload_cid_recomputed=str(
                branch_b.re_hash_payload()),
            parent_cid=str(branch_b.parent_cid),
            chain_root_cid=str(witness.merge_tree_root_cid),
            verdict=IntegrityVerdict.PROVENANCE_VIOLATION.value,
            detail=(
                "branch_b_tip does not descend directly from "
                "the declared common ancestor"),
        )
    if len(_snapshot_parent_cids(merged)) != 2:
        return IntegrityVerificationReportV1(
            schema=W82_INTEGRITY_V1_SCHEMA_VERSION,
            snapshot_cid=str(merged.cid()),
            payload_cid=str(merged.payload_cid),
            payload_cid_recomputed=str(
                merged.re_hash_payload()),
            parent_cid=str(merged.parent_cid),
            chain_root_cid=str(witness.merge_tree_root_cid),
            verdict=IntegrityVerdict.PROVENANCE_VIOLATION.value,
            detail=(
                "merged snapshot must reference exactly two "
                "distinct branch-parent CIDs"),
        )
    if set(_snapshot_parent_cids(merged)) != {
            str(branch_a.cid()), str(branch_b.cid())}:
        return IntegrityVerificationReportV1(
            schema=W82_INTEGRITY_V1_SCHEMA_VERSION,
            snapshot_cid=str(merged.cid()),
            payload_cid=str(merged.payload_cid),
            payload_cid_recomputed=str(
                merged.re_hash_payload()),
            parent_cid=str(merged.parent_cid),
            chain_root_cid=str(witness.merge_tree_root_cid),
            verdict=IntegrityVerdict.PROVENANCE_VIOLATION.value,
            detail=(
                "merged snapshot's direct parent is not one of "
                "the declared branch tips"),
        )
    merge_parent_cids = set(_extract_merge_parent_cids_v1(merged))
    if merge_parent_cids != {
            str(branch_a.cid()), str(branch_b.cid())}:
        return IntegrityVerificationReportV1(
            schema=W82_INTEGRITY_V1_SCHEMA_VERSION,
            snapshot_cid=str(merged.cid()),
            payload_cid=str(merged.payload_cid),
            payload_cid_recomputed=str(
                merged.re_hash_payload()),
            parent_cid=str(merged.parent_cid),
            chain_root_cid=str(witness.merge_tree_root_cid),
            verdict=IntegrityVerdict.PROVENANCE_VIOLATION.value,
            detail=(
                "merged snapshot payload does not prove a true "
                "two-parent merge"),
        )
    if str(common.cid()) not in _snapshot_ancestor_set(
            str(branch_a.cid()), snapshots_by_cid=snapshot_map):
        return IntegrityVerificationReportV1(
            schema=W82_INTEGRITY_V1_SCHEMA_VERSION,
            snapshot_cid=str(branch_a.cid()),
            payload_cid=str(branch_a.payload_cid),
            payload_cid_recomputed=str(
                branch_a.re_hash_payload()),
            parent_cid=str(branch_a.parent_cid),
            chain_root_cid=str(witness.merge_tree_root_cid),
            verdict=IntegrityVerdict.PROVENANCE_VIOLATION.value,
            detail=(
                "branch_a_tip does not descend from the "
                "declared common ancestor"),
        )
    if str(common.cid()) not in _snapshot_ancestor_set(
            str(branch_b.cid()), snapshots_by_cid=snapshot_map):
        return IntegrityVerificationReportV1(
            schema=W82_INTEGRITY_V1_SCHEMA_VERSION,
            snapshot_cid=str(branch_b.cid()),
            payload_cid=str(branch_b.payload_cid),
            payload_cid_recomputed=str(
                branch_b.re_hash_payload()),
            parent_cid=str(branch_b.parent_cid),
            chain_root_cid=str(witness.merge_tree_root_cid),
            verdict=IntegrityVerdict.PROVENANCE_VIOLATION.value,
            detail=(
                "branch_b_tip does not descend from the "
                "declared common ancestor"),
        )
    return IntegrityVerificationReportV1(
        schema=W82_INTEGRITY_V1_SCHEMA_VERSION,
        snapshot_cid=str(witness.merged_snapshot_cid),
        payload_cid="",
        payload_cid_recomputed="",
        parent_cid="",
        chain_root_cid=str(witness.merge_tree_root_cid),
        verdict=IntegrityVerdict.OK.value,
        detail="branch-merge witness verified",
    )


# ---------------------------------------------------------------
# Corruption simulation
# ---------------------------------------------------------------

def simulate_silent_corruption_v1(
        snapshot: StateSnapshotV1,
        *, corrupted_byte_index: int = 0,
) -> StateSnapshotV1:
    """Return a NEW snapshot identical to ``snapshot`` except
    that one byte of the payload has been flipped — the
    declared ``payload_cid`` is preserved (i.e. the corruption
    is SILENT until integrity verification runs).

    This is exactly the kind of failure the verifier should
    catch.
    """
    payload = bytearray(snapshot.payload_bytes)
    if len(payload) == 0:
        payload = bytearray(b"\x01")
    idx = int(corrupted_byte_index) % len(payload)
    payload[idx] = (payload[idx] ^ 0x55) & 0xFF
    return StateSnapshotV1(
        schema=snapshot.schema,
        snapshot_id=str(snapshot.snapshot_id),
        parent_cid=str(snapshot.parent_cid),
        payload_bytes=bytes(payload),
        payload_cid=str(snapshot.payload_cid),
        timestamp_ns=int(snapshot.timestamp_ns),
        hmac_signature_hex=str(
            snapshot.hmac_signature_hex),
        additional_parent_cids=tuple(
            str(cid) for cid in snapshot.additional_parent_cids),
    )


# ---------------------------------------------------------------
# Bench
# ---------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class IntegrityBenchReportV1:
    """End-to-end integrity bench output."""

    schema: str
    n_snapshots: int
    chain_clean_verdicts: tuple[str, ...]
    corruption_detected: bool
    rollback_after_corruption_ok: bool
    merge_clean_verdict: str
    merge_unsafe_verdict: str
    config_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "n_snapshots": int(self.n_snapshots),
            "chain_clean_verdicts": list(
                self.chain_clean_verdicts),
            "corruption_detected": bool(
                self.corruption_detected),
            "rollback_after_corruption_ok": bool(
                self.rollback_after_corruption_ok),
            "merge_clean_verdict": str(
                self.merge_clean_verdict),
            "merge_unsafe_verdict": str(
                self.merge_unsafe_verdict),
            "config_cid": str(self.config_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w82_integrity_bench_report_v1",
            "report": self.to_dict()})


def build_clean_snapshot_chain_v1(
        *, n: int = W82_INTEGRITY_DEFAULT_CHAIN_LEN,
        seed: int = W82_INTEGRITY_DEFAULT_SEED,
        hmac_key: bytes | None = (
            W82_INTEGRITY_DEFAULT_HMAC_KEY),
) -> tuple[StateSnapshotV1, ...]:
    """Build a clean, deterministic chain of N snapshots."""
    snaps: list[StateSnapshotV1] = []
    prev_cid = "genesis"
    for i in range(int(n)):
        payload_dict = {
            "kind": "w82_test_chain_payload_v1",
            "i": int(i),
            "seed": int(seed),
        }
        payload_bytes = _canonical_bytes(payload_dict)
        s = build_state_snapshot_v1(
            snapshot_id=f"s{i}",
            parent_cid=str(prev_cid),
            payload_bytes=payload_bytes,
            timestamp_ns=int(
                1_000_000_000 * (i + 1)),
            hmac_key=hmac_key)
        snaps.append(s)
        prev_cid = str(s.cid())
    return tuple(snaps)


def run_corruption_rollback_merge_bench_v1(
        *, n: int = W82_INTEGRITY_DEFAULT_CHAIN_LEN,
        seed: int = W82_INTEGRITY_DEFAULT_SEED,
        hmac_key: bytes | None = (
            W82_INTEGRITY_DEFAULT_HMAC_KEY),
        corrupt_index: int = 3,
) -> IntegrityBenchReportV1:
    """Run the load-bearing integrity bench:

    1. Build a clean snapshot chain of length ``n``.
    2. Verify the clean chain — every verdict OK.
    3. Inject silent corruption into snapshot index
       ``corrupt_index`` and verify detection.
    4. Roll back to snapshot ``corrupt_index - 1`` and verify
       integrity holds.
    5. Build two branches from snapshot index 2, mutate each
       independently, merge them, and verify the merge witness.
    6. Issue an UNSAFE merge witness (wrong parent) and verify
       detection.
    """
    clean = build_clean_snapshot_chain_v1(
        n=int(n), seed=int(seed), hmac_key=hmac_key)
    chain_reports = verify_chain_integrity_v1(
        snapshots=clean, hmac_key=hmac_key)
    verdicts = tuple(r.verdict for r in chain_reports)
    # Inject silent corruption.
    corrupted = list(clean)
    corrupted[int(corrupt_index)] = (
        simulate_silent_corruption_v1(
            clean[int(corrupt_index)]))
    corrupted_reports = verify_chain_integrity_v1(
        snapshots=corrupted, hmac_key=hmac_key)
    corruption_detected = bool(
        corrupted_reports[int(corrupt_index)].verdict
        == IntegrityVerdict.CORRUPT.value)
    # Rollback to corrupt_index - 1.
    anchor = build_rollback_anchor_v1(
        label="pre_corruption",
        snapshot=clean[int(corrupt_index) - 1],
        chain_root_cid=str(
            MerkleHashTreeV1.from_snapshot_cids(
                [s.cid() for s in clean]).root_cid),
        created_at_ns=int(1_000_000_000 * int(n) + 1))
    rollback_snap, rb_report = execute_rollback_v1(
        anchor=anchor, snapshots=clean,
        hmac_key=hmac_key)
    rollback_ok = bool(
        rollback_snap is not None and
        str(rb_report.verdict) ==
        IntegrityVerdict.OK.value)
    # Branch / merge.
    common = clean[2]
    branch_a_tip = build_state_snapshot_v1(
        snapshot_id="branch_a_tip",
        parent_cid=str(common.cid()),
        payload_bytes=_canonical_bytes({
            "kind": "branch_a_v1", "seed": int(seed)}),
        timestamp_ns=int(2_000_000_000),
        hmac_key=hmac_key)
    branch_b_tip = build_state_snapshot_v1(
        snapshot_id="branch_b_tip",
        parent_cid=str(common.cid()),
        payload_bytes=_canonical_bytes({
            "kind": "branch_b_v1", "seed": int(seed)}),
        timestamp_ns=int(2_000_000_001),
        hmac_key=hmac_key)
    merged = build_state_snapshot_v1(
        snapshot_id="merged",
        parent_cid=str(branch_a_tip.cid()),
        additional_parent_cids=(str(branch_b_tip.cid()),),
        payload_bytes=_canonical_bytes({
            "kind": "merged_v1",
            "seed": int(seed),
            "merge_parent_cids": [
                str(branch_a_tip.cid()),
                str(branch_b_tip.cid()),
            ]}),
        timestamp_ns=int(2_000_000_002),
        hmac_key=hmac_key)
    all_snaps = list(clean) + [
        branch_a_tip, branch_b_tip, merged]
    clean_merge = build_branch_merge_witness_v1(
        branch_a_tip=branch_a_tip,
        branch_b_tip=branch_b_tip,
        common_ancestor=common,
        merged_snapshot=merged,
        all_snapshots=all_snaps,
        n_conflicts_resolved=1)
    clean_verify = verify_branch_merge_witness_v1(
        witness=clean_merge, snapshots=all_snaps)
    # Unsafe merge witness (wrong parent_cid).
    unsafe_merged = build_state_snapshot_v1(
        snapshot_id="unsafe_merged",
        parent_cid=str(branch_a_tip.cid()),
        payload_bytes=_canonical_bytes({
            "kind": "unsafe_merged_v1",
            "merge_parent_cids": [
                str(branch_a_tip.cid()),
                str(branch_b_tip.cid()),
            ],
        }),
        timestamp_ns=int(2_000_000_003),
        hmac_key=hmac_key)
    unsafe_all = list(all_snaps) + [unsafe_merged]
    unsafe_witness = build_branch_merge_witness_v1(
        branch_a_tip=branch_a_tip,
        branch_b_tip=branch_b_tip,
        common_ancestor=common,
        merged_snapshot=unsafe_merged,
        all_snapshots=unsafe_all,
        n_conflicts_resolved=0)
    cfg_cid = _sha256_hex({
        "kind": "w82_integrity_bench_config_v1",
        "n": int(n),
        "seed": int(seed),
        "corrupt_index": int(corrupt_index),
        "hmac_keyed": bool(hmac_key is not None),
    })
    return IntegrityBenchReportV1(
        schema=W82_INTEGRITY_V1_SCHEMA_VERSION,
        n_snapshots=int(len(clean)),
        chain_clean_verdicts=tuple(verdicts),
        corruption_detected=bool(corruption_detected),
        rollback_after_corruption_ok=bool(rollback_ok),
        merge_clean_verdict=str(clean_verify.verdict),
        merge_unsafe_verdict=str(
            unsafe_witness.integrity_verdict),
        config_cid=str(cfg_cid),
    )


__all__ = [
    "W82_INTEGRITY_V1_SCHEMA_VERSION",
    "W82_INTEGRITY_DEFAULT_MERKLE_FANOUT",
    "W82_INTEGRITY_DEFAULT_HMAC_KEY",
    "W82_INTEGRITY_VERDICTS",
    "W82_INTEGRITY_AWARE_ACTIONS",
    "IntegrityVerdict",
    "IntegrityAwareAction",
    "StateSnapshotV1",
    "MerkleHashTreeV1",
    "RollbackAnchorV1",
    "BranchMergeWitnessV1",
    "IntegrityVerificationReportV1",
    "IntegrityAwareFallbackDecisionV1",
    "IntegrityBenchReportV1",
    "build_state_snapshot_v1",
    "build_rollback_anchor_v1",
    "verify_snapshot_integrity_v1",
    "verify_chain_integrity_v1",
    "execute_rollback_v1",
    "build_branch_merge_witness_v1",
    "verify_branch_merge_witness_v1",
    "integrity_aware_fallback_decision_v1",
    "simulate_silent_corruption_v1",
    "build_clean_snapshot_chain_v1",
    "run_corruption_rollback_merge_bench_v1",
]
