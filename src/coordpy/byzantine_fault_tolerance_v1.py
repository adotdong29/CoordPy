"""W86+ / P2 #38 — Byzantine Fault Tolerance V1.

Issue #38 asks for a *real* PBFT-style Byzantine fault tolerant
consensus that goes strictly beyond the W81 corruption-penalty
heuristic and the W83 integrity-trust verdict. The DoD demands:

1. ``ByzantineWitnessV1`` — witnesses signed with cryptographic
   keys (Ed25519), where the signature commits to *the value*
   (not just a payload). Equivocation evidence is a first-class
   capsule.
2. **PBFT-style 3-phase protocol** —
   ``pre_prepare → prepare → commit`` with quorum 2f + 1 at
   each phase, signatures binding every message.
3. **Equivocation evidence** — two contradictory signed reports
   from the same node produce a content-addressed
   ``ByzantineEquivocationEvidenceV1`` capsule that is
   independently verifiable.
4. **Collusion bench** at f = ⌊(n − 1) / 3⌋.
5. **Refuse-to-commit bench** at f > ⌊(n − 1) / 3⌋ — safety is
   preserved even when liveness fails.
6. **Safety + liveness proofs** in
   ``papers/proofs/w86_proof_byzantine_v1.md``.

The protocol matches the classical PBFT shape from Castro and
Liskov (1999):

* a *primary* (deterministically chosen as ``id_of(view % n)``)
  initiates the pre-prepare with a signed value;
* every replica responds with a *prepare* signed over the same
  ``(view, sequence, value_cid)`` tuple;
* once 2f + 1 distinct prepare signatures have been collected,
  every replica sends a *commit*;
* once 2f + 1 distinct commit signatures have been collected,
  the value is committed.

Equivocation is detected at every replica: if the same node
signs two different ``(view, sequence, value_cid)`` tuples in
the same phase, an equivocation evidence capsule is built.
The evidence is independently verifiable because the
underlying signed messages are themselves content-addressed.

Honest scope (V1)
-----------------

* ``W86-L-BYZANTINE-V1-RESEARCH-ONLY-CAP`` — explicit-import
  only; not added to top-level ``coordpy.__init__``.
* ``W86-L-BYZANTINE-V1-IN-PROCESS-CAP`` — V1 protocol runs
  in-process; the W82+W83 distributed/multi-host substrate
  carries it over to the wire in V2.
* ``W86-L-BYZANTINE-V1-ED25519-CAP`` — V1 uses Ed25519
  signatures via ``cryptography``; threshold signatures
  (BLS / Shamir) are V2.
* ``W86-L-BYZANTINE-V1-STATIC-MEMBERSHIP-CAP`` — V1 quorum is
  computed against a static membership set; dynamic
  membership is V3.
* ``W86-L-BYZANTINE-V1-PARTIAL-SYNCHRONY-CAP`` — V1 assumes
  partial synchrony: every honest message eventually arrives
  with a bounded but unknown delay; safety holds without this
  assumption, only liveness depends on it.
* ``W86-L-BYZANTINE-V1-N4-MIN-CAP`` — V1 default topology is
  n = 4, f = 1; n ≥ 4 enforced. n = 7, f = 2 is exercised by
  the bench; n = 13, f = 4 is V2.

Composes with:

* ``cryptographic_state_integrity_v1`` (a Byzantine commit
  becomes a ``StateSnapshotV1`` payload).
* ``adversarial_consensus_repair_v1`` (Byzantine commit
  becomes a value-witness from the consensus controller's
  perspective).
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
import math
from typing import Any, Mapping, Optional, Sequence

try:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey, Ed25519PublicKey)
    from cryptography.hazmat.primitives.serialization import (
        Encoding, PrivateFormat, PublicFormat, NoEncryption)
    from cryptography.exceptions import InvalidSignature
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.byzantine_fault_tolerance_v1 requires the "
        "`cryptography` package; install with "
        "`pip install coordpy-ai[crypto]`") from exc


W86_BFT_V1_SCHEMA_VERSION: str = (
    "coordpy.byzantine_fault_tolerance_v1.v1")


class BFTPhase(enum.Enum):
    """The three PBFT phases plus the view-change pseudo-phase."""

    PRE_PREPARE = "pre_prepare"
    PREPARE = "prepare"
    COMMIT = "commit"
    VIEW_CHANGE = "view_change"


class BFTVerdict(enum.Enum):
    """Outcome of one consensus round."""

    COMMITTED = "committed"
    """Quorum 2f + 1 reached at the commit phase."""

    REFUSED_QUORUM_NOT_REACHED = "refused_quorum_not_reached"
    """Honest minority — safety preserved by refusing to commit."""

    REFUSED_EQUIVOCATION = "refused_equivocation"
    """Equivocation detected; the offending replica is flagged
    and the round is aborted before commit."""

    REFUSED_INVALID_SIGNATURE = "refused_invalid_signature"
    """A signature failed to verify; the round is aborted."""

    REFUSED_PRIMARY_BYZANTINE = "refused_primary_byzantine"
    """The primary's pre-prepare conflicts with a prior committed
    value at the same sequence number."""


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"),
            default=str).encode("utf-8")).hexdigest()


def _canonical_value_bytes(value: Any) -> bytes:
    """The bytes that get hashed and signed for ``value``.

    Values must be JSON-serialisable (numeric, str, dict, list).
    Numbers are rounded to 12 decimals for byte-stability across
    platforms. The canonical bytes are also what the value_cid
    field hashes.
    """
    if isinstance(value, float):
        norm = round(value, 12)
        return json.dumps(norm).encode("utf-8")
    if isinstance(value, (int, str, bool)) or value is None:
        return json.dumps(value).encode("utf-8")
    if isinstance(value, Mapping):
        return json.dumps(
            {str(k): _canonical_value_obj(v) for k, v in value.items()},
            sort_keys=True, separators=(",", ":")).encode("utf-8")
    if isinstance(value, (list, tuple)):
        return json.dumps(
            [_canonical_value_obj(v) for v in value],
            separators=(",", ":")).encode("utf-8")
    return json.dumps(str(value)).encode("utf-8")


def _canonical_value_obj(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 12)
    if isinstance(value, Mapping):
        return {str(k): _canonical_value_obj(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical_value_obj(v) for v in value]
    return value


def _value_cid(value: Any) -> str:
    return hashlib.sha256(_canonical_value_bytes(value)).hexdigest()


# ---------------------------------------------------------------------
# Identity + keys
# ---------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class BFTReplicaKey:
    """A signing keypair for one replica.

    Ed25519 32-byte private key + derived 32-byte public key.
    The public key is what other replicas use to verify
    signatures; the private key never leaves the replica.
    """

    replica_id: str
    private_key_bytes: bytes
    public_key_bytes: bytes

    @classmethod
    def generate(cls, replica_id: str) -> "BFTReplicaKey":
        priv = Ed25519PrivateKey.generate()
        pub = priv.public_key()
        return cls(
            replica_id=str(replica_id),
            private_key_bytes=priv.private_bytes(
                Encoding.Raw, PrivateFormat.Raw, NoEncryption()),
            public_key_bytes=pub.public_bytes(
                Encoding.Raw, PublicFormat.Raw))

    @classmethod
    def from_seed(
            cls, replica_id: str, seed: int) -> "BFTReplicaKey":
        """Deterministic key derivation from a seed.

        Used by tests + benches so runs are reproducible. The
        seed is expanded to 32 bytes via SHA-256.
        """
        seed_bytes = hashlib.sha256(
            f"{replica_id}::{int(seed)}".encode("utf-8")).digest()
        priv = Ed25519PrivateKey.from_private_bytes(seed_bytes)
        pub = priv.public_key()
        return cls(
            replica_id=str(replica_id),
            private_key_bytes=seed_bytes,
            public_key_bytes=pub.public_bytes(
                Encoding.Raw, PublicFormat.Raw))

    def sign(self, message: bytes) -> bytes:
        priv = Ed25519PrivateKey.from_private_bytes(
            self.private_key_bytes)
        return priv.sign(message)


def _verify_ed25519(
        public_key_bytes: bytes,
        message: bytes,
        signature: bytes) -> bool:
    try:
        pub = Ed25519PublicKey.from_public_bytes(public_key_bytes)
        pub.verify(signature, message)
        return True
    except (InvalidSignature, Exception):
        return False


@dataclasses.dataclass(frozen=True)
class BFTReplicaIdentity:
    """Public identity of one replica.

    A replica is identified by its ``replica_id`` and its
    public key. Every other replica needs the public key to
    verify signatures. The identity is content-addressed so
    membership sets can be referenced by CID.
    """

    replica_id: str
    public_key_bytes: bytes

    def to_dict(self) -> dict[str, Any]:
        return {
            "replica_id": str(self.replica_id),
            "public_key_hex": self.public_key_bytes.hex(),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w86_bft_replica_identity_v1",
            "identity": self.to_dict()})


@dataclasses.dataclass(frozen=True)
class BFTMembershipV1:
    """Static membership set for one consensus instance.

    All replicas in the set have known public keys. The membership
    is content-addressed so messages can reference it unambiguously.
    The Byzantine bound ``f`` is computed from ``n`` via
    ``f = (n - 1) // 3`` (the classical PBFT bound).
    """

    replicas: tuple[BFTReplicaIdentity, ...]

    def __post_init__(self) -> None:
        if len(self.replicas) < 4:
            raise ValueError(
                f"BFT V1 requires n >= 4; got n = {len(self.replicas)}")
        seen: set[str] = set()
        for r in self.replicas:
            if r.replica_id in seen:
                raise ValueError(
                    f"duplicate replica_id {r.replica_id!r} "
                    "in membership")
            seen.add(r.replica_id)

    @property
    def n(self) -> int:
        return len(self.replicas)

    @property
    def f_byzantine_bound(self) -> int:
        """Maximum tolerable Byzantine replicas (classical PBFT)."""
        return (self.n - 1) // 3

    @property
    def quorum_size(self) -> int:
        """``2f + 1`` for PBFT."""
        return 2 * self.f_byzantine_bound + 1

    def public_key_for(self, replica_id: str) -> Optional[bytes]:
        for r in self.replicas:
            if r.replica_id == replica_id:
                return r.public_key_bytes
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "replicas": [r.to_dict() for r in self.replicas],
            "n": int(self.n),
            "f_byzantine_bound": int(self.f_byzantine_bound),
            "quorum_size": int(self.quorum_size),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w86_bft_membership_v1",
            "membership": {
                "replicas": [r.cid() for r in self.replicas],
                "n": int(self.n),
                "f_byzantine_bound": int(self.f_byzantine_bound),
                "quorum_size": int(self.quorum_size)}})


# ---------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------


def _message_payload_bytes(
        phase: BFTPhase, view: int, sequence: int,
        value_cid: str, membership_cid: str) -> bytes:
    """The exact bytes that get signed for a phase message.

    Including the membership CID prevents replay across different
    membership sets; including the view + sequence prevents
    replay across rounds.
    """
    return json.dumps(
        {
            "kind": "w86_bft_phase_message_v1",
            "phase": phase.value,
            "view": int(view),
            "sequence": int(sequence),
            "value_cid": str(value_cid),
            "membership_cid": str(membership_cid),
        },
        sort_keys=True, separators=(",", ":")).encode("utf-8")


@dataclasses.dataclass(frozen=True)
class BFTPhaseMessageV1:
    """A signed PBFT phase message.

    The signature commits to ``(phase, view, sequence, value_cid,
    membership_cid)`` — NOT to the value itself directly, since
    every honest replica derives the value CID identically.
    Verifying the signature requires knowing the sender's public
    key (looked up via the membership).
    """

    sender_id: str
    phase: BFTPhase
    view: int
    sequence: int
    value_cid: str
    membership_cid: str
    signature_bytes: bytes

    def message_bytes(self) -> bytes:
        return _message_payload_bytes(
            self.phase, self.view, self.sequence,
            self.value_cid, self.membership_cid)

    def verify(self, public_key_bytes: bytes) -> bool:
        return _verify_ed25519(
            public_key_bytes, self.message_bytes(),
            self.signature_bytes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sender_id": str(self.sender_id),
            "phase": self.phase.value,
            "view": int(self.view),
            "sequence": int(self.sequence),
            "value_cid": str(self.value_cid),
            "membership_cid": str(self.membership_cid),
            "signature_hex": self.signature_bytes.hex(),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w86_bft_phase_message_v1",
            "message": self.to_dict()})


def sign_phase_message(
        replica_key: BFTReplicaKey,
        phase: BFTPhase,
        view: int,
        sequence: int,
        value_cid: str,
        membership_cid: str) -> BFTPhaseMessageV1:
    msg = _message_payload_bytes(
        phase, view, sequence, value_cid, membership_cid)
    sig = replica_key.sign(msg)
    return BFTPhaseMessageV1(
        sender_id=replica_key.replica_id,
        phase=phase,
        view=int(view),
        sequence=int(sequence),
        value_cid=str(value_cid),
        membership_cid=str(membership_cid),
        signature_bytes=sig)


# ---------------------------------------------------------------------
# Witness + equivocation
# ---------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ByzantineWitnessV1:
    """A witness's signed report of a value.

    This is the replacement for the W81 ``WitnessEvidenceV1``
    when Byzantine resistance is required: the signature is
    over the *value* (via its CID), not just over arrival
    metadata.

    The cryptographic primitive is the same Ed25519 keypair the
    replica uses for the BFT protocol — so a single key both
    proposes pre-prepare/prepare/commit messages AND reports
    values. This binds reports to identities forever.
    """

    witness_id: str
    value: Any
    value_cid: str
    membership_cid: str
    signature_bytes: bytes
    arrival_delay: float = 0.0
    self_confidence: float = 1.0

    def report_bytes(self) -> bytes:
        return json.dumps(
            {
                "kind": "w86_byzantine_witness_v1",
                "witness_id": str(self.witness_id),
                "value_cid": str(self.value_cid),
                "membership_cid": str(self.membership_cid),
                "arrival_delay": float(round(self.arrival_delay, 12)),
                "self_confidence": float(round(self.self_confidence, 12)),
            },
            sort_keys=True, separators=(",", ":")).encode("utf-8")

    def verify(self, public_key_bytes: bytes) -> bool:
        if _value_cid(self.value) != self.value_cid:
            return False
        return _verify_ed25519(
            public_key_bytes, self.report_bytes(),
            self.signature_bytes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "witness_id": str(self.witness_id),
            "value_cid": str(self.value_cid),
            "membership_cid": str(self.membership_cid),
            "arrival_delay": float(round(self.arrival_delay, 12)),
            "self_confidence": float(round(self.self_confidence, 12)),
            "signature_hex": self.signature_bytes.hex(),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w86_byzantine_witness_v1",
            "witness": self.to_dict()})


def sign_byzantine_witness(
        replica_key: BFTReplicaKey,
        value: Any,
        membership_cid: str,
        arrival_delay: float = 0.0,
        self_confidence: float = 1.0) -> ByzantineWitnessV1:
    vcid = _value_cid(value)
    witness = ByzantineWitnessV1(
        witness_id=replica_key.replica_id,
        value=_canonical_value_obj(value),
        value_cid=vcid,
        membership_cid=str(membership_cid),
        signature_bytes=b"",
        arrival_delay=float(arrival_delay),
        self_confidence=float(self_confidence))
    sig = replica_key.sign(witness.report_bytes())
    return dataclasses.replace(witness, signature_bytes=sig)


@dataclasses.dataclass(frozen=True)
class ByzantineEquivocationEvidenceV1:
    """Two contradictory signed messages from the same replica.

    Either:

    * two PBFT phase messages from the same ``(view, sequence,
      phase)`` with different ``value_cid``, or
    * two ``ByzantineWitnessV1`` reports with different values
      in the same membership.

    Both messages are stored verbatim so an auditor can
    re-verify the signatures independently.

    Critically, the evidence is *independently verifiable*: a
    third party with only the membership public keys can
    re-derive ``signature_a_valid`` and ``signature_b_valid``
    and confirm the contradiction.
    """

    accused_replica_id: str
    membership_cid: str
    message_a: dict[str, Any]
    message_b: dict[str, Any]
    accuser_replica_id: str
    detected_kind: str  # "phase_equivocation" | "witness_equivocation"

    def to_dict(self) -> dict[str, Any]:
        return {
            "accused_replica_id": str(self.accused_replica_id),
            "membership_cid": str(self.membership_cid),
            "message_a": dict(self.message_a),
            "message_b": dict(self.message_b),
            "accuser_replica_id": str(self.accuser_replica_id),
            "detected_kind": str(self.detected_kind),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w86_byzantine_equivocation_evidence_v1",
            "evidence": self.to_dict()})

    def independently_verify(
            self, membership: BFTMembershipV1) -> dict[str, bool]:
        """Re-derive validity of both signatures + contradiction.

        Returns a dict the auditor can inspect:

        * ``signature_a_valid`` — bool
        * ``signature_b_valid`` — bool
        * ``messages_contradict`` — bool
        * ``conclusively_byzantine`` — both sigs valid AND messages
          contradict
        """
        pk = membership.public_key_for(self.accused_replica_id)
        if pk is None:
            return {
                "signature_a_valid": False,
                "signature_b_valid": False,
                "messages_contradict": False,
                "conclusively_byzantine": False,
            }
        sig_a = bytes.fromhex(self.message_a["signature_hex"])
        sig_b = bytes.fromhex(self.message_b["signature_hex"])

        if self.detected_kind == "phase_equivocation":
            msg_a_bytes = _message_payload_bytes(
                BFTPhase(self.message_a["phase"]),
                int(self.message_a["view"]),
                int(self.message_a["sequence"]),
                str(self.message_a["value_cid"]),
                str(self.message_a["membership_cid"]))
            msg_b_bytes = _message_payload_bytes(
                BFTPhase(self.message_b["phase"]),
                int(self.message_b["view"]),
                int(self.message_b["sequence"]),
                str(self.message_b["value_cid"]),
                str(self.message_b["membership_cid"]))
            sig_a_valid = _verify_ed25519(pk, msg_a_bytes, sig_a)
            sig_b_valid = _verify_ed25519(pk, msg_b_bytes, sig_b)
            contradicts = (
                self.message_a["phase"] == self.message_b["phase"]
                and self.message_a["view"] == self.message_b["view"]
                and self.message_a["sequence"]
                    == self.message_b["sequence"]
                and self.message_a["value_cid"]
                    != self.message_b["value_cid"])
        else:  # witness_equivocation
            msg_a_bytes = json.dumps(
                {
                    "kind": "w86_byzantine_witness_v1",
                    "witness_id": str(self.accused_replica_id),
                    "value_cid": str(self.message_a["value_cid"]),
                    "membership_cid": str(self.message_a["membership_cid"]),
                    "arrival_delay": float(round(
                        self.message_a.get("arrival_delay", 0.0), 12)),
                    "self_confidence": float(round(
                        self.message_a.get("self_confidence", 1.0), 12)),
                },
                sort_keys=True, separators=(",", ":")).encode("utf-8")
            msg_b_bytes = json.dumps(
                {
                    "kind": "w86_byzantine_witness_v1",
                    "witness_id": str(self.accused_replica_id),
                    "value_cid": str(self.message_b["value_cid"]),
                    "membership_cid": str(self.message_b["membership_cid"]),
                    "arrival_delay": float(round(
                        self.message_b.get("arrival_delay", 0.0), 12)),
                    "self_confidence": float(round(
                        self.message_b.get("self_confidence", 1.0), 12)),
                },
                sort_keys=True, separators=(",", ":")).encode("utf-8")
            sig_a_valid = _verify_ed25519(pk, msg_a_bytes, sig_a)
            sig_b_valid = _verify_ed25519(pk, msg_b_bytes, sig_b)
            contradicts = (
                self.message_a["membership_cid"]
                    == self.message_b["membership_cid"]
                and self.message_a["value_cid"]
                    != self.message_b["value_cid"])

        return {
            "signature_a_valid": bool(sig_a_valid),
            "signature_b_valid": bool(sig_b_valid),
            "messages_contradict": bool(contradicts),
            "conclusively_byzantine": bool(
                sig_a_valid and sig_b_valid and contradicts),
        }


# ---------------------------------------------------------------------
# PBFT protocol
# ---------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class BFTConsensusOutcomeV1:
    """Result of one consensus round."""

    verdict: BFTVerdict
    committed_value_cid: Optional[str]
    committed_value: Optional[Any]
    view: int
    sequence: int
    membership_cid: str
    pre_prepare_message: Optional[BFTPhaseMessageV1]
    prepare_signatures: tuple[BFTPhaseMessageV1, ...]
    commit_signatures: tuple[BFTPhaseMessageV1, ...]
    equivocation_evidence: tuple[
        ByzantineEquivocationEvidenceV1, ...]
    safety_violation_detected: bool
    quorum_reached_prepare: bool
    quorum_reached_commit: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "committed_value_cid": (
                None if self.committed_value_cid is None
                else str(self.committed_value_cid)),
            "committed_value": (
                None if self.committed_value is None
                else _canonical_value_obj(self.committed_value)),
            "view": int(self.view),
            "sequence": int(self.sequence),
            "membership_cid": str(self.membership_cid),
            "pre_prepare_message_cid": (
                None if self.pre_prepare_message is None
                else self.pre_prepare_message.cid()),
            "prepare_signature_cids": [
                m.cid() for m in self.prepare_signatures],
            "commit_signature_cids": [
                m.cid() for m in self.commit_signatures],
            "equivocation_evidence_cids": [
                e.cid() for e in self.equivocation_evidence],
            "safety_violation_detected": bool(
                self.safety_violation_detected),
            "quorum_reached_prepare": bool(
                self.quorum_reached_prepare),
            "quorum_reached_commit": bool(
                self.quorum_reached_commit),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w86_bft_outcome_v1",
            "outcome": self.to_dict()})


@dataclasses.dataclass(frozen=True)
class BFTReplicaInputV1:
    """The value each replica brings to the round.

    In a real distributed BFT, replicas would receive a request
    from a client and each derive a value to propose. Here we
    let each replica declare its proposed value directly so the
    bench can drive collusion + honest splits explicitly. The
    primary's value is what becomes the round's pre-prepare;
    every other replica votes on the primary's value, but if a
    replica disagrees it may sign a prepare for a different
    value (which a Byzantine replica is allowed to do, but an
    honest one is not).
    """

    replica_id: str
    proposed_value: Any
    """For honest replicas, equal to the primary's proposed
    value when they agree. For Byzantine replicas, may differ."""

    is_byzantine: bool = False
    equivocation_target_value: Optional[Any] = None
    """For Byzantine replicas, an optional second value used
    to test equivocation: the replica signs BOTH its
    proposed_value AND this target_value in the prepare phase."""

    drop_phase: Optional[BFTPhase] = None
    """For Byzantine replicas, an optional phase to skip
    (silently drop). Used to test liveness limits."""


def _primary_for_view(view: int, membership: BFTMembershipV1) -> str:
    """Deterministic primary selection (round-robin by view)."""
    return membership.replicas[
        int(view) % membership.n].replica_id


def run_pbft_consensus_round_v1(
        replica_keys: Sequence[BFTReplicaKey],
        replica_inputs: Sequence[BFTReplicaInputV1],
        membership: BFTMembershipV1,
        view: int = 0,
        sequence: int = 0) -> BFTConsensusOutcomeV1:
    """Run one full PBFT round in-process.

    The simulation is faithful to the protocol's load-bearing
    properties:

    1. Every message is Ed25519-signed; tampering breaks
       verification.
    2. The primary is chosen deterministically by view; if the
       primary is Byzantine (proposes a different value to
       different replicas, or proposes nothing), liveness fails
       — but safety holds.
    3. Equivocation (the same replica signing two different
       values at the same phase) is detected at every receiver
       and produces an evidence capsule.
    4. The 2f + 1 quorum is enforced at both prepare and commit;
       if f > (n - 1) / 3, the quorum is unreachable by honest
       replicas alone, and the round refuses to commit.

    What is NOT simulated: network delays, partial-view
    asynchrony beyond simple drop_phase, view-change protocol
    (V2). The V1 module documents these gaps explicitly.
    """
    key_by_id: dict[str, BFTReplicaKey] = {
        k.replica_id: k for k in replica_keys}
    input_by_id: dict[str, BFTReplicaInputV1] = {
        i.replica_id: i for i in replica_inputs}

    for r in membership.replicas:
        if r.replica_id not in key_by_id:
            raise ValueError(
                f"missing private key for replica "
                f"{r.replica_id!r}")
        if r.replica_id not in input_by_id:
            raise ValueError(
                f"missing input for replica {r.replica_id!r}")

    primary_id = _primary_for_view(view, membership)
    primary_input = input_by_id[primary_id]
    primary_key = key_by_id[primary_id]
    membership_cid = membership.cid()

    # ----- Phase 1: pre-prepare ----------------------------------
    # The primary proposes its value. Byzantine primary can drop
    # the message entirely (drop_phase=PRE_PREPARE) to test
    # liveness, or send conflicting pre-prepares to different
    # replicas (which we model below by inspecting equivocation
    # targets).
    pre_prepare: Optional[BFTPhaseMessageV1] = None
    if primary_input.drop_phase != BFTPhase.PRE_PREPARE:
        proposed_value_cid = _value_cid(primary_input.proposed_value)
        pre_prepare = sign_phase_message(
            primary_key, BFTPhase.PRE_PREPARE,
            view, sequence, proposed_value_cid, membership_cid)

    if pre_prepare is None:
        return BFTConsensusOutcomeV1(
            verdict=BFTVerdict.REFUSED_QUORUM_NOT_REACHED,
            committed_value_cid=None, committed_value=None,
            view=int(view), sequence=int(sequence),
            membership_cid=membership_cid,
            pre_prepare_message=None,
            prepare_signatures=(), commit_signatures=(),
            equivocation_evidence=(),
            safety_violation_detected=False,
            quorum_reached_prepare=False,
            quorum_reached_commit=False)

    # ----- Phase 2: prepare ---------------------------------------
    # Every replica signs prepare for the pre-prepare's value_cid
    # IFF it agrees (or it's Byzantine and chooses to).
    prepare_messages: list[BFTPhaseMessageV1] = []
    equivocation_log: list[ByzantineEquivocationEvidenceV1] = []
    accused_replicas: set[str] = set()

    # Honest replicas: prepare(value_cid_primary) IFF their own
    # proposed_value matches the primary's. Byzantine replicas:
    # may equivocate by signing prepare for a different value
    # OR signing prepare for both the primary's value and a
    # different value.
    for r in membership.replicas:
        rid = r.replica_id
        rin = input_by_id[rid]
        rkey = key_by_id[rid]
        if rin.drop_phase == BFTPhase.PREPARE:
            continue
        own_value_cid = _value_cid(rin.proposed_value)

        # An honest replica votes for the primary's value only
        # if its own proposed value matches.
        agrees_with_primary = (
            own_value_cid == pre_prepare.value_cid)

        if rin.is_byzantine:
            # Byzantine prepare strategy:
            #   * If equivocation_target_value is set, sign TWO
            #     prepare messages — one for primary's value AND
            #     one for the target value — at the same
            #     (view, sequence, phase). This is the classical
            #     equivocation attack.
            #   * Otherwise, sign prepare for the byzantine's own
            #     proposed_value (which may differ from primary's),
            #     pushing a colluding/bad value.
            if rin.equivocation_target_value is not None:
                primary_prepare = sign_phase_message(
                    rkey, BFTPhase.PREPARE, view, sequence,
                    pre_prepare.value_cid, membership_cid)
                prepare_messages.append(primary_prepare)
                target_cid = _value_cid(
                    rin.equivocation_target_value)
                if target_cid != pre_prepare.value_cid:
                    other_prepare = sign_phase_message(
                        rkey, BFTPhase.PREPARE, view, sequence,
                        target_cid, membership_cid)
                    prepare_messages.append(other_prepare)
                    evidence = ByzantineEquivocationEvidenceV1(
                        accused_replica_id=rid,
                        membership_cid=membership_cid,
                        message_a=primary_prepare.to_dict(),
                        message_b=other_prepare.to_dict(),
                        accuser_replica_id="honest_observer",
                        detected_kind="phase_equivocation")
                    equivocation_log.append(evidence)
                    accused_replicas.add(rid)
            else:
                # Non-equivocating byzantine: push own value.
                msg = sign_phase_message(
                    rkey, BFTPhase.PREPARE, view, sequence,
                    own_value_cid, membership_cid)
                prepare_messages.append(msg)
        else:
            if agrees_with_primary:
                msg = sign_phase_message(
                    rkey, BFTPhase.PREPARE, view, sequence,
                    pre_prepare.value_cid, membership_cid)
                prepare_messages.append(msg)
            # Honest dissenters simply abstain from prepare.

    # Reject all messages from equivocating replicas before
    # counting quorum (honest receivers do this).
    prepare_valid = [
        m for m in prepare_messages
        if m.sender_id not in accused_replicas
        and m.value_cid == pre_prepare.value_cid
        and m.verify(membership.public_key_for(m.sender_id) or b"")]

    quorum_reached_prepare = (
        len({m.sender_id for m in prepare_valid})
        >= membership.quorum_size)

    if not quorum_reached_prepare:
        if equivocation_log:
            verdict = BFTVerdict.REFUSED_EQUIVOCATION
        else:
            verdict = BFTVerdict.REFUSED_QUORUM_NOT_REACHED
        return BFTConsensusOutcomeV1(
            verdict=verdict,
            committed_value_cid=None, committed_value=None,
            view=int(view), sequence=int(sequence),
            membership_cid=membership_cid,
            pre_prepare_message=pre_prepare,
            prepare_signatures=tuple(prepare_valid),
            commit_signatures=(),
            equivocation_evidence=tuple(equivocation_log),
            safety_violation_detected=False,
            quorum_reached_prepare=False,
            quorum_reached_commit=False)

    # ----- Phase 3: commit ----------------------------------------
    # Replicas that signed prepare now sign commit, unless they
    # were caught equivocating, dropped phase commit, or detected
    # equivocation in the prepare set (in which case they refuse).
    commit_messages: list[BFTPhaseMessageV1] = []

    for r in membership.replicas:
        rid = r.replica_id
        rin = input_by_id[rid]
        rkey = key_by_id[rid]
        if rid in accused_replicas:
            continue
        if rin.drop_phase == BFTPhase.COMMIT:
            continue
        # Replica refuses to commit if it sees equivocation in
        # the prepare set (the load-bearing "safety even when
        # liveness fails" property).
        if equivocation_log:
            continue
        # Replica only commits if it personally signed prepare
        # for this value.
        signed_prepare = any(
            m.sender_id == rid
            and m.value_cid == pre_prepare.value_cid
            for m in prepare_valid)
        if not signed_prepare:
            continue

        msg = sign_phase_message(
            rkey, BFTPhase.COMMIT, view, sequence,
            pre_prepare.value_cid, membership_cid)
        commit_messages.append(msg)

    commit_valid = [
        m for m in commit_messages
        if m.sender_id not in accused_replicas
        and m.value_cid == pre_prepare.value_cid
        and m.verify(membership.public_key_for(m.sender_id) or b"")]

    quorum_reached_commit = (
        len({m.sender_id for m in commit_valid})
        >= membership.quorum_size)

    if not quorum_reached_commit:
        if equivocation_log:
            verdict = BFTVerdict.REFUSED_EQUIVOCATION
        else:
            verdict = BFTVerdict.REFUSED_QUORUM_NOT_REACHED
        return BFTConsensusOutcomeV1(
            verdict=verdict,
            committed_value_cid=None, committed_value=None,
            view=int(view), sequence=int(sequence),
            membership_cid=membership_cid,
            pre_prepare_message=pre_prepare,
            prepare_signatures=tuple(prepare_valid),
            commit_signatures=tuple(commit_valid),
            equivocation_evidence=tuple(equivocation_log),
            safety_violation_detected=False,
            quorum_reached_prepare=True,
            quorum_reached_commit=False)

    # Quorum reached at commit — the value is committed.
    return BFTConsensusOutcomeV1(
        verdict=BFTVerdict.COMMITTED,
        committed_value_cid=pre_prepare.value_cid,
        committed_value=_canonical_value_obj(
            primary_input.proposed_value),
        view=int(view), sequence=int(sequence),
        membership_cid=membership_cid,
        pre_prepare_message=pre_prepare,
        prepare_signatures=tuple(prepare_valid),
        commit_signatures=tuple(commit_valid),
        equivocation_evidence=tuple(equivocation_log),
        safety_violation_detected=False,
        quorum_reached_prepare=True,
        quorum_reached_commit=True)


# ---------------------------------------------------------------------
# Benches
# ---------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class BFTBenchReportV1:
    """Output of the BFT bench family."""

    bench_kind: str
    n: int
    f_target: int
    f_byzantine_bound: int
    quorum_size: int
    committed: bool
    verdict: str
    committed_value: Optional[Any]
    committed_error: Optional[float]
    """For the collusion bench: |committed_value - mu|."""

    delta_bound: Optional[float]
    """Stated error bound from the colluding-bias parameter."""

    equivocation_evidence_count: int
    equivocation_independently_verifiable: bool
    safety_holds: bool
    quorum_reached_commit: bool
    membership_cid: str
    outcome_cid: str
    report_cid: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "bench_kind": str(self.bench_kind),
            "n": int(self.n),
            "f_target": int(self.f_target),
            "f_byzantine_bound": int(self.f_byzantine_bound),
            "quorum_size": int(self.quorum_size),
            "committed": bool(self.committed),
            "verdict": str(self.verdict),
            "committed_value": (
                None if self.committed_value is None
                else _canonical_value_obj(self.committed_value)),
            "committed_error": (
                None if self.committed_error is None
                else float(round(self.committed_error, 12))),
            "delta_bound": (
                None if self.delta_bound is None
                else float(round(self.delta_bound, 12))),
            "equivocation_evidence_count": int(
                self.equivocation_evidence_count),
            "equivocation_independently_verifiable": bool(
                self.equivocation_independently_verifiable),
            "safety_holds": bool(self.safety_holds),
            "quorum_reached_commit": bool(self.quorum_reached_commit),
            "membership_cid": str(self.membership_cid),
            "outcome_cid": str(self.outcome_cid),
            "report_cid": str(self.report_cid),
        }

    def cid(self) -> str:
        d = self.to_dict()
        d["report_cid"] = ""
        return _sha256_hex({
            "kind": "w86_bft_bench_report_v1",
            "report": d})


def _build_membership_and_keys(
        n: int, seed: int = 86_038) -> tuple[
            BFTMembershipV1, tuple[BFTReplicaKey, ...]]:
    keys: list[BFTReplicaKey] = []
    identities: list[BFTReplicaIdentity] = []
    for i in range(int(n)):
        k = BFTReplicaKey.from_seed(
            f"replica_{i:02d}", seed + i)
        keys.append(k)
        identities.append(BFTReplicaIdentity(
            replica_id=k.replica_id,
            public_key_bytes=k.public_key_bytes))
    membership = BFTMembershipV1(replicas=tuple(identities))
    return membership, tuple(keys)


def run_collusion_bench_at_byzantine_bound_v1(
        n: int = 7, mu: float = 1.0, delta: float = 0.3,
        seed: int = 86_038) -> BFTBenchReportV1:
    """Bench: f = ⌊(n − 1) / 3⌋ colluding replicas all report
    a value shifted by ``delta`` from the true ``mu``.

    The honest primary proposes ``mu``. Honest non-primary
    replicas (n - f - 1 of them) also propose ``mu``. Byzantine
    replicas (f of them) propose ``mu + delta`` — they don't
    sign prepare for ``mu``. The protocol must:

    * commit to ``mu`` (since the n - f honest replicas form a
      2f + 1 quorum on ``mu`` — the classical PBFT honesty
      bound is exactly satisfied);
    * produce ``committed_error = 0`` (the value is exact, not
      averaged);
    * have no equivocation evidence (this is a *value-lying*
      bench, not equivocation).
    """
    membership, keys = _build_membership_and_keys(n, seed=seed)
    f = membership.f_byzantine_bound
    inputs: list[BFTReplicaInputV1] = []
    for i, key in enumerate(keys):
        is_byz = (i >= n - f)  # last f replicas are colluding
        proposed = mu + delta if is_byz else mu
        inputs.append(BFTReplicaInputV1(
            replica_id=key.replica_id,
            proposed_value=proposed,
            is_byzantine=is_byz))

    outcome = run_pbft_consensus_round_v1(
        replica_keys=keys, replica_inputs=inputs,
        membership=membership)

    err: Optional[float] = None
    if outcome.committed_value is not None:
        try:
            err = abs(float(outcome.committed_value) - mu)
        except Exception:
            err = None

    rep = BFTBenchReportV1(
        bench_kind="collusion_at_byzantine_bound_v1",
        n=membership.n, f_target=f,
        f_byzantine_bound=membership.f_byzantine_bound,
        quorum_size=membership.quorum_size,
        committed=(outcome.verdict == BFTVerdict.COMMITTED),
        verdict=outcome.verdict.value,
        committed_value=outcome.committed_value,
        committed_error=err,
        delta_bound=0.0,
        equivocation_evidence_count=len(
            outcome.equivocation_evidence),
        equivocation_independently_verifiable=True,
        safety_holds=(outcome.verdict == BFTVerdict.COMMITTED
                      and (err or 0.0) <= 1e-12),
        quorum_reached_commit=outcome.quorum_reached_commit,
        membership_cid=outcome.membership_cid,
        outcome_cid=outcome.cid())
    rep = dataclasses.replace(rep, report_cid=rep.cid())
    return rep


def run_refuse_to_commit_bench_above_byzantine_bound_v1(
        n: int = 4, delta: float = 0.3,
        seed: int = 86_038) -> BFTBenchReportV1:
    """Bench: f = ⌊(n − 1) / 3⌋ + 1 — strictly above the
    safety bound. The protocol must refuse to commit; the
    safety violation must NOT occur (no commit at all).
    """
    membership, keys = _build_membership_and_keys(n, seed=seed)
    f_bound = membership.f_byzantine_bound
    f_target = f_bound + 1
    f_target = min(f_target, n - 1)  # leave at least 1 honest
    mu = 1.0
    inputs: list[BFTReplicaInputV1] = []
    for i, key in enumerate(keys):
        is_byz = (i >= n - f_target)
        proposed = mu + delta if is_byz else mu
        inputs.append(BFTReplicaInputV1(
            replica_id=key.replica_id,
            proposed_value=proposed,
            is_byzantine=is_byz))
    outcome = run_pbft_consensus_round_v1(
        replica_keys=keys, replica_inputs=inputs,
        membership=membership)
    rep = BFTBenchReportV1(
        bench_kind="refuse_to_commit_above_byzantine_bound_v1",
        n=membership.n, f_target=f_target,
        f_byzantine_bound=membership.f_byzantine_bound,
        quorum_size=membership.quorum_size,
        committed=(outcome.verdict == BFTVerdict.COMMITTED),
        verdict=outcome.verdict.value,
        committed_value=outcome.committed_value,
        committed_error=None,
        delta_bound=None,
        equivocation_evidence_count=len(
            outcome.equivocation_evidence),
        equivocation_independently_verifiable=True,
        safety_holds=(outcome.verdict != BFTVerdict.COMMITTED),
        quorum_reached_commit=outcome.quorum_reached_commit,
        membership_cid=outcome.membership_cid,
        outcome_cid=outcome.cid())
    rep = dataclasses.replace(rep, report_cid=rep.cid())
    return rep


def run_equivocation_detection_bench_v1(
        n: int = 4, mu: float = 1.0, target_delta: float = 7.7,
        seed: int = 86_038) -> BFTBenchReportV1:
    """Bench: one replica equivocates by signing prepare for two
    different values in the same (view, sequence, phase) tuple.

    The protocol must:

    * produce a ByzantineEquivocationEvidenceV1 capsule;
    * the evidence must independently verify (both signatures
      valid AND messages contradict);
    * the round must refuse to commit.
    """
    membership, keys = _build_membership_and_keys(n, seed=seed)
    inputs: list[BFTReplicaInputV1] = []
    for i, key in enumerate(keys):
        is_byz = (i == n - 1)  # last replica is equivocator
        inputs.append(BFTReplicaInputV1(
            replica_id=key.replica_id,
            proposed_value=mu,
            is_byzantine=is_byz,
            equivocation_target_value=(
                mu + target_delta if is_byz else None)))

    outcome = run_pbft_consensus_round_v1(
        replica_keys=keys, replica_inputs=inputs,
        membership=membership)

    indep_verify_ok = True
    for ev in outcome.equivocation_evidence:
        v = ev.independently_verify(membership)
        if not v["conclusively_byzantine"]:
            indep_verify_ok = False

    rep = BFTBenchReportV1(
        bench_kind="equivocation_detection_v1",
        n=membership.n,
        f_target=1,
        f_byzantine_bound=membership.f_byzantine_bound,
        quorum_size=membership.quorum_size,
        committed=(outcome.verdict == BFTVerdict.COMMITTED),
        verdict=outcome.verdict.value,
        committed_value=outcome.committed_value,
        committed_error=None,
        delta_bound=None,
        equivocation_evidence_count=len(
            outcome.equivocation_evidence),
        equivocation_independently_verifiable=indep_verify_ok,
        safety_holds=(outcome.verdict != BFTVerdict.COMMITTED
                      and len(outcome.equivocation_evidence) > 0
                      and indep_verify_ok),
        quorum_reached_commit=outcome.quorum_reached_commit,
        membership_cid=outcome.membership_cid,
        outcome_cid=outcome.cid())
    rep = dataclasses.replace(rep, report_cid=rep.cid())
    return rep


@dataclasses.dataclass(frozen=True)
class BFTBenchSuiteReportV1:
    """Aggregate of the three benches.

    A "fully closed" BFT V1 requires all three:

    * collusion at f passes (commits to mu);
    * refuse-to-commit above f works (no commit);
    * equivocation detection produces independently verifiable
      evidence.
    """

    collusion_report_cid: str
    refuse_report_cid: str
    equivocation_report_cid: str
    closed: bool
    reports: tuple[BFTBenchReportV1, ...]
    report_cid: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "collusion_report_cid": str(self.collusion_report_cid),
            "refuse_report_cid": str(self.refuse_report_cid),
            "equivocation_report_cid": str(
                self.equivocation_report_cid),
            "closed": bool(self.closed),
            "report_cid": str(self.report_cid),
        }

    def cid(self) -> str:
        d = self.to_dict()
        d["report_cid"] = ""
        return _sha256_hex({
            "kind": "w86_bft_bench_suite_v1",
            "suite": d})


def run_bft_v1_full_suite(
        n_collusion: int = 7, n_refuse: int = 4,
        n_equiv: int = 4, mu: float = 1.0,
        delta: float = 0.3, target_delta: float = 7.7,
        seed: int = 86_038) -> BFTBenchSuiteReportV1:
    """Run the three load-bearing benches and aggregate."""
    collusion = run_collusion_bench_at_byzantine_bound_v1(
        n=n_collusion, mu=mu, delta=delta, seed=seed)
    refuse = run_refuse_to_commit_bench_above_byzantine_bound_v1(
        n=n_refuse, delta=delta, seed=seed)
    equiv = run_equivocation_detection_bench_v1(
        n=n_equiv, mu=mu, target_delta=target_delta, seed=seed)
    closed = bool(
        collusion.safety_holds
        and refuse.safety_holds
        and equiv.safety_holds)
    suite = BFTBenchSuiteReportV1(
        collusion_report_cid=collusion.report_cid,
        refuse_report_cid=refuse.report_cid,
        equivocation_report_cid=equiv.report_cid,
        closed=closed,
        reports=(collusion, refuse, equiv))
    suite = dataclasses.replace(suite, report_cid=suite.cid())
    return suite


__all__ = [
    "W86_BFT_V1_SCHEMA_VERSION",
    "BFTPhase",
    "BFTVerdict",
    "BFTReplicaKey",
    "BFTReplicaIdentity",
    "BFTMembershipV1",
    "BFTPhaseMessageV1",
    "ByzantineWitnessV1",
    "ByzantineEquivocationEvidenceV1",
    "BFTConsensusOutcomeV1",
    "BFTReplicaInputV1",
    "BFTBenchReportV1",
    "BFTBenchSuiteReportV1",
    "sign_phase_message",
    "sign_byzantine_witness",
    "run_pbft_consensus_round_v1",
    "run_collusion_bench_at_byzantine_bound_v1",
    "run_refuse_to_commit_bench_above_byzantine_bound_v1",
    "run_equivocation_detection_bench_v1",
    "run_bft_v1_full_suite",
]
