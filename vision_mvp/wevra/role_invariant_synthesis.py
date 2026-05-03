"""W42 Cross-Role-Invariant Synthesis (SDK v3.43).

W42 is the first capsule-native synthesis layer that adds a *third
orthogonal evidence axis* on top of the W41 integrated synthesis.
The W41 axes are the producer axis (W7..W11 multi-round bundle
decoder) and the trust axis (W21..W40 trust-adjudication chain).
Both axes operate on signal that an adversary controlling the W21
oracle responses AND the W40 response signature bytes can
co-opt.  This is the W41-L-COMPOSITE-COLLUSION-CAP wall.

W42 adds a third axis -- the **cross-role-handoff invariance
axis** -- that operates on a channel orthogonal to the oracle
responses and the response signature bytes: the canonical
role-handoff structure of the cell's input handoffs.  An honest
controller pre-registers a :class:`RoleInvariancePolicyRegistry`
mapping ``role_handoff_signature_cid -> expected_services``.  The
W42 verifier compares the W41 integrated services to the
registered expected services and abstains via
``INVARIANCE_DIVERGED_ABSTAINED`` when they disagree.

When the adversary has compromised both the producer axis AND the
trust axis but has NOT poisoned the controller-side policy
registry, the role-handoff signature still pulls in the correct
expected services, and W42 strictly recovers trust precision over
W41 (the load-bearing :ref:`R-89-ROLE-INVARIANT-RECOVER`
+0.500 gain).

When the adversary ALSO poisons the controller-side policy
registry, W42 cannot recover at the capsule layer.  This is the
new proved-conditional ``W42-L-FULL-COMPOSITE-COLLUSION-CAP``
limitation theorem (the W42 analog of
``W41-L-COMPOSITE-COLLUSION-CAP``, sharper in adversary cost).

This module is a *strict superset* and *strictly additive* on top
of W41.  No W22..W41 internals are touched.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Mapping, Sequence

from vision_mvp.wevra.team_coord import (
    SchemaCapsule,
    LatentVerificationOutcome,
)
from vision_mvp.wevra.integrated_synthesis import (
    IntegratedSynthesisOrchestrator,
    IntegratedSynthesisRegistry,
    W41_BRANCH_INTEGRATED_BOTH_AXES,
    W41_BRANCH_INTEGRATED_PRODUCER_ONLY,
    W41_BRANCH_INTEGRATED_TRUST_ONLY,
    W41_BRANCH_INTEGRATED_NEITHER_AXIS,
    W41_BRANCH_INTEGRATED_AXES_DIVERGED_ABSTAINED,
    W41_BRANCH_TRIVIAL_INTEGRATED_PASSTHROUGH,
    W41_BRANCH_INTEGRATED_DISABLED,
    W41_BRANCH_INTEGRATED_REJECTED,
)


# =============================================================================
# W42 schema, branch constants
# =============================================================================

W42_ROLE_INVARIANT_SCHEMA_VERSION: str = (
    "wevra.role_invariant_synthesis.v1")

W42_BRANCH_TRIVIAL_INVARIANCE_PASSTHROUGH: str = (
    "trivial_invariance_passthrough")
W42_BRANCH_INVARIANCE_DISABLED: str = "invariance_disabled"
W42_BRANCH_INVARIANCE_REJECTED: str = "invariance_rejected"
W42_BRANCH_INVARIANCE_NO_TRIGGER: str = "invariance_no_trigger"
W42_BRANCH_INVARIANCE_RATIFIED: str = "invariance_ratified"
W42_BRANCH_INVARIANCE_DIVERGED_ABSTAINED: str = (
    "invariance_diverged_abstained")
W42_BRANCH_INVARIANCE_NO_POLICY: str = "invariance_no_policy"

W42_ALL_BRANCHES: tuple[str, ...] = (
    W42_BRANCH_TRIVIAL_INVARIANCE_PASSTHROUGH,
    W42_BRANCH_INVARIANCE_DISABLED,
    W42_BRANCH_INVARIANCE_REJECTED,
    W42_BRANCH_INVARIANCE_NO_TRIGGER,
    W42_BRANCH_INVARIANCE_RATIFIED,
    W42_BRANCH_INVARIANCE_DIVERGED_ABSTAINED,
    W42_BRANCH_INVARIANCE_NO_POLICY,
)


# =============================================================================
# Canonicalisation + role-handoff signature CID
# =============================================================================

def _w42_canonical_bytes(payload: dict[str, Any]) -> bytes:
    """Canonical JSON bytes for a CID input."""
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _canonical_payload_text(payload: Any) -> str:
    """Lower-case, whitespace-collapsed canonical text form."""
    if payload is None:
        return ""
    text = str(payload).strip().lower()
    return " ".join(text.split())


def _canonical_handoff_tuples(
        per_round_handoffs: Sequence[Sequence[Any]],
) -> tuple[tuple[str, str, str], ...]:
    """Canonical (role, kind, payload_text) tuples sorted.

    Permutation-invariant: the role-handoff signature does not
    depend on the order of arrival of handoffs within a cell.
    """
    out: list[tuple[str, str, str]] = []
    for r in per_round_handoffs or ():
        for h in r or ():
            role = str(getattr(h, "source_role", "") or "")
            kind = str(getattr(h, "claim_kind", "") or "")
            payload = _canonical_payload_text(
                getattr(h, "payload", "") or "")
            out.append((role, kind, payload))
    return tuple(sorted(out))


def compute_role_handoff_signature_cid(
        per_round_handoffs: Sequence[Sequence[Any]],
) -> str:
    """SHA-256 over canonical sorted (role, kind, payload) tuples.

    Namespaced as ``w42_role_handoff_signature``.  Independent of
    oracle responses and response signature bytes; compromising
    this CID requires controlling the upstream cell schema.
    """
    payload = {
        "kind": "w42_role_handoff_signature",
        "tuples": [
            list(t) for t in _canonical_handoff_tuples(
                per_round_handoffs)
        ],
    }
    return hashlib.sha256(
        _w42_canonical_bytes(payload)).hexdigest()


# =============================================================================
# Policy registry
# =============================================================================

@dataclasses.dataclass(frozen=True)
class RoleInvariancePolicyEntry:
    """Controller-side honest mapping signature -> expected services.

    The controller pre-registers one entry per known
    role-handoff signature.  An unknown signature falls through
    to ``INVARIANCE_NO_POLICY``.
    """

    role_handoff_signature_cid: str
    expected_services: tuple[str, ...]

    def cid(self) -> str:
        payload = {
            "kind": "w42_policy_entry",
            "role_handoff_signature_cid": str(
                self.role_handoff_signature_cid),
            "expected_services": [
                str(s) for s in self.expected_services],
        }
        return hashlib.sha256(
            _w42_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass
class RoleInvariancePolicyRegistry:
    """Controller-side registry of role-invariance policy entries.

    Key: ``role_handoff_signature_cid`` (str hex).
    Value: :class:`RoleInvariancePolicyEntry`.
    """

    entries: dict[str, RoleInvariancePolicyEntry] = (
        dataclasses.field(default_factory=dict))

    def register(self, entry: RoleInvariancePolicyEntry) -> None:
        self.entries[str(entry.role_handoff_signature_cid)] = entry

    def lookup(self, signature_cid: str) -> (
            "RoleInvariancePolicyEntry | None"):
        return self.entries.get(str(signature_cid))

    def cid(self) -> str:
        payload = {
            "kind": "w42_policy_registry",
            "entries": [
                {
                    "k": k,
                    "v": v.cid(),
                }
                for k, v in sorted(self.entries.items())
            ],
        }
        return hashlib.sha256(
            _w42_canonical_bytes(payload)).hexdigest()


# =============================================================================
# Decision selector
# =============================================================================

def select_role_invariance_decision(
        *,
        integrated_services: Sequence[str],
        expected_services: Sequence[str] | None,
        policy_match_found: bool,
) -> tuple[str, tuple[str, ...], float]:
    """Closed-form invariance decision.

    Returns ``(branch, output_services, invariance_score)``.

    * No policy match -> ``INVARIANCE_NO_POLICY`` (output =
      integrated_services unchanged; W42 falls through).
    * Empty integrated services -> ``INVARIANCE_NO_TRIGGER``
      (nothing to compare; preserve W41 byte-for-byte).
    * Match -> ``INVARIANCE_RATIFIED`` (output =
      integrated_services; score = 1.0).
    * Disagreement -> ``INVARIANCE_DIVERGED_ABSTAINED`` (output =
      empty; score = jaccard(integrated, expected)).
    """
    int_set = tuple(sorted({str(s) for s in integrated_services if s}))
    exp_set = tuple(sorted(
        {str(s) for s in (expected_services or ()) if s}))

    if not policy_match_found:
        return (W42_BRANCH_INVARIANCE_NO_POLICY,
                int_set, 0.0)
    if not int_set:
        return (W42_BRANCH_INVARIANCE_NO_TRIGGER,
                int_set, 0.0)
    if int_set == exp_set:
        return (W42_BRANCH_INVARIANCE_RATIFIED, int_set, 1.0)
    inter = set(int_set) & set(exp_set)
    union = set(int_set) | set(exp_set)
    score = (float(len(inter)) / float(len(union))
             if union else 0.0)
    return (W42_BRANCH_INVARIANCE_DIVERGED_ABSTAINED, (), score)


# =============================================================================
# CID computations (manifest-v12 component CIDs)
# =============================================================================

def _compute_w42_invariance_state_cid(
        *,
        invariance_branch: str,
        role_handoff_signature_cid: str,
        integrated_services_pre_w42: tuple[str, ...],
        integrated_services_post_w42: tuple[str, ...],
        cell_index: int,
) -> str:
    payload = {
        "kind": "w42_invariance_state",
        "invariance_branch": str(invariance_branch),
        "role_handoff_signature_cid": str(
            role_handoff_signature_cid),
        "integrated_services_pre_w42": [
            str(s) for s in integrated_services_pre_w42],
        "integrated_services_post_w42": [
            str(s) for s in integrated_services_post_w42],
        "cell_index": int(cell_index),
    }
    return hashlib.sha256(
        _w42_canonical_bytes(payload)).hexdigest()


def _compute_w42_invariance_decision_cid(
        *,
        invariance_branch: str,
        integrated_services_post_w42: tuple[str, ...],
        invariance_score: float,
        n_w41_visible_tokens: int,
        n_w42_visible_tokens: int,
        n_w42_overhead_tokens: int,
) -> str:
    payload = {
        "kind": "w42_invariance_decision",
        "invariance_branch": str(invariance_branch),
        "integrated_services_post_w42": [
            str(s) for s in integrated_services_post_w42],
        "invariance_score": float(invariance_score),
        "n_w41_visible_tokens": int(n_w41_visible_tokens),
        "n_w42_visible_tokens": int(n_w42_visible_tokens),
        "n_w42_overhead_tokens": int(n_w42_overhead_tokens),
    }
    return hashlib.sha256(
        _w42_canonical_bytes(payload)).hexdigest()


def _compute_w42_invariance_audit_cid(
        *,
        invariance_branch: str,
        role_handoff_signature_cid: str,
        policy_entry_cid: str,
        integrated_services_pre_w42: tuple[str, ...],
        expected_services: tuple[str, ...],
        integrated_services_post_w42: tuple[str, ...],
        invariance_score: float,
) -> str:
    payload = {
        "kind": "w42_invariance_audit",
        "invariance_branch": str(invariance_branch),
        "role_handoff_signature_cid": str(
            role_handoff_signature_cid),
        "policy_entry_cid": str(policy_entry_cid),
        "integrated_services_pre_w42": [
            str(s) for s in integrated_services_pre_w42],
        "expected_services": [
            str(s) for s in expected_services],
        "integrated_services_post_w42": [
            str(s) for s in integrated_services_post_w42],
        "invariance_score": float(invariance_score),
    }
    return hashlib.sha256(
        _w42_canonical_bytes(payload)).hexdigest()


def _compute_w42_invariance_witness_cid(
        *,
        invariance_branch: str,
        role_handoff_signature_cid: str,
        n_w41_visible_tokens: int,
        n_w42_visible_tokens: int,
        n_w42_overhead_tokens: int,
        n_structured_bits: int,
) -> str:
    payload = {
        "kind": "w42_invariance_witness",
        "invariance_branch": str(invariance_branch),
        "role_handoff_signature_cid": str(
            role_handoff_signature_cid),
        "n_w41_visible_tokens": int(n_w41_visible_tokens),
        "n_w42_visible_tokens": int(n_w42_visible_tokens),
        "n_w42_overhead_tokens": int(n_w42_overhead_tokens),
        "n_structured_bits": int(n_structured_bits),
    }
    return hashlib.sha256(
        _w42_canonical_bytes(payload)).hexdigest()


def _compute_w42_manifest_v12_cid(
        *,
        parent_w41_cid: str,
        invariance_state_cid: str,
        invariance_decision_cid: str,
        invariance_audit_cid: str,
        invariance_witness_cid: str,
        role_handoff_signature_cid: str,
) -> str:
    payload = {
        "kind": "w42_manifest_v12",
        "parent_w41_cid": str(parent_w41_cid),
        "invariance_state_cid": str(invariance_state_cid),
        "invariance_decision_cid": str(invariance_decision_cid),
        "invariance_audit_cid": str(invariance_audit_cid),
        "invariance_witness_cid": str(invariance_witness_cid),
        "role_handoff_signature_cid": str(
            role_handoff_signature_cid),
    }
    return hashlib.sha256(
        _w42_canonical_bytes(payload)).hexdigest()


def _compute_w42_outer_cid(
        *,
        schema_cid: str,
        parent_w41_cid: str,
        manifest_v12_cid: str,
        cell_index: int,
) -> str:
    payload = {
        "kind": "w42_outer",
        "schema_cid": str(schema_cid),
        "parent_w41_cid": str(parent_w41_cid),
        "manifest_v12_cid": str(manifest_v12_cid),
        "cell_index": int(cell_index),
    }
    return hashlib.sha256(
        _w42_canonical_bytes(payload)).hexdigest()


# =============================================================================
# Envelope, registry, result
# =============================================================================

@dataclasses.dataclass(frozen=True)
class RoleInvariantSynthesisRatificationEnvelope:
    """Sealed manifest-v12 envelope binding W41 + invariance state."""

    schema_version: str
    schema_cid: str
    parent_w41_cid: str
    cell_index: int

    invariance_branch: str
    role_handoff_signature_cid: str
    policy_entry_cid: str

    integrated_services_pre_w42: tuple[str, ...]
    expected_services: tuple[str, ...]
    integrated_services_post_w42: tuple[str, ...]
    invariance_score: float

    invariance_state_cid: str
    invariance_decision_cid: str
    invariance_audit_cid: str
    invariance_witness_cid: str
    manifest_v12_cid: str

    n_w41_visible_tokens: int
    n_w42_visible_tokens: int
    n_w42_overhead_tokens: int
    n_structured_bits: int

    w42_cid: str

    def recompute_w42_cid(self) -> str:
        return _compute_w42_outer_cid(
            schema_cid=self.schema_cid,
            parent_w41_cid=self.parent_w41_cid,
            manifest_v12_cid=self.manifest_v12_cid,
            cell_index=int(self.cell_index),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "schema_cid": self.schema_cid,
            "parent_w41_cid": self.parent_w41_cid,
            "cell_index": int(self.cell_index),
            "invariance_branch": self.invariance_branch,
            "role_handoff_signature_cid": (
                self.role_handoff_signature_cid),
            "policy_entry_cid": self.policy_entry_cid,
            "integrated_services_pre_w42": list(
                self.integrated_services_pre_w42),
            "expected_services": list(self.expected_services),
            "integrated_services_post_w42": list(
                self.integrated_services_post_w42),
            "invariance_score": float(self.invariance_score),
            "invariance_state_cid": self.invariance_state_cid,
            "invariance_decision_cid": (
                self.invariance_decision_cid),
            "invariance_audit_cid": self.invariance_audit_cid,
            "invariance_witness_cid": (
                self.invariance_witness_cid),
            "manifest_v12_cid": self.manifest_v12_cid,
            "n_w41_visible_tokens": int(
                self.n_w41_visible_tokens),
            "n_w42_visible_tokens": int(
                self.n_w42_visible_tokens),
            "n_w42_overhead_tokens": int(
                self.n_w42_overhead_tokens),
            "n_structured_bits": int(self.n_structured_bits),
            "w42_cid": self.w42_cid,
        }


@dataclasses.dataclass
class RoleInvariantSynthesisRegistry:
    """Registry for the W42 cross-role-invariance layer.

    When ``invariance_enabled = False`` AND
    ``manifest_v12_disabled = True`` AND
    ``abstain_on_invariance_diverged = False``, the W42
    orchestrator reduces to W41 byte-for-byte (the W42-L-
    TRIVIAL-PASSTHROUGH falsifier).
    """

    schema: SchemaCapsule
    inner_w41_registry: IntegratedSynthesisRegistry | None
    policy_registry: RoleInvariancePolicyRegistry = (
        dataclasses.field(
            default_factory=RoleInvariancePolicyRegistry))
    invariance_enabled: bool = True
    manifest_v12_disabled: bool = False
    abstain_on_invariance_diverged: bool = True

    @property
    def is_trivial(self) -> bool:
        return (not self.invariance_enabled
                and self.manifest_v12_disabled
                and not self.abstain_on_invariance_diverged)


@dataclasses.dataclass
class W42RoleInvariantResult:
    answer: dict[str, Any]
    inner_w41_decoder_branch: str
    decoder_branch: str
    invariance_branch: str
    role_handoff_signature_cid: str
    policy_entry_cid: str
    integrated_services_pre_w42: tuple[str, ...]
    expected_services: tuple[str, ...]
    integrated_services_post_w42: tuple[str, ...]
    invariance_score: float
    parent_w41_cid: str
    cell_index: int
    n_w41_visible_tokens: int
    n_w42_visible_tokens: int
    n_w42_overhead_tokens: int
    w42_cid: str
    manifest_v12_cid: str
    invariance_state_cid: str
    invariance_decision_cid: str
    invariance_audit_cid: str
    invariance_witness_cid: str
    ratified: bool
    verification_ok: bool
    verification_reason: str
    n_envelope_bytes: int
    n_structured_bits: int
    cram_factor_w42: float

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


# =============================================================================
# Verifier
# =============================================================================

def verify_role_invariant_synthesis_ratification(
        env: RoleInvariantSynthesisRatificationEnvelope | None,
        *,
        registered_schema: SchemaCapsule,
        registered_parent_w41_cid: str,
) -> LatentVerificationOutcome:
    """Pure-function verifier for the W42 envelope.

    Enumerates 14 disjoint W42 failure modes:

    1.  ``empty_w42_envelope``
    2.  ``w42_schema_version_unknown``
    3.  ``w42_schema_cid_mismatch``
    4.  ``w41_parent_cid_mismatch``
    5.  ``w42_invariance_branch_unknown``
    6.  ``w42_role_handoff_signature_cid_mismatch``
    7.  ``w42_invariance_state_cid_mismatch``
    8.  ``w42_invariance_decision_cid_mismatch``
    9.  ``w42_invariance_audit_cid_mismatch``
    10. ``w42_invariance_witness_cid_mismatch``
    11. ``w42_invariance_score_invalid``
    12. ``w42_token_accounting_invalid``
    13. ``w42_manifest_v12_cid_mismatch``
    14. ``w42_outer_cid_mismatch``
    """
    n_checks = 0
    if env is None:
        return LatentVerificationOutcome(
            ok=False, reason="empty_w42_envelope",
            n_checks=n_checks)
    n_checks += 1
    if env.schema_version != W42_ROLE_INVARIANT_SCHEMA_VERSION:
        return LatentVerificationOutcome(
            ok=False, reason="w42_schema_version_unknown",
            n_checks=n_checks)
    n_checks += 1
    if env.schema_cid != registered_schema.cid:
        return LatentVerificationOutcome(
            ok=False, reason="w42_schema_cid_mismatch",
            n_checks=n_checks)
    n_checks += 1
    if env.parent_w41_cid != str(registered_parent_w41_cid):
        return LatentVerificationOutcome(
            ok=False, reason="w41_parent_cid_mismatch",
            n_checks=n_checks)
    n_checks += 1
    if env.invariance_branch not in W42_ALL_BRANCHES:
        return LatentVerificationOutcome(
            ok=False, reason="w42_invariance_branch_unknown",
            n_checks=n_checks)
    n_checks += 1
    # Role-handoff signature CID is a 64-hex SHA-256.  Empty is
    # only allowed on the trivial-passthrough branch.
    if (env.invariance_branch
            != W42_BRANCH_TRIVIAL_INVARIANCE_PASSTHROUGH):
        if (not env.role_handoff_signature_cid
                or len(env.role_handoff_signature_cid) != 64):
            return LatentVerificationOutcome(
                ok=False,
                reason="w42_role_handoff_signature_cid_mismatch",
                n_checks=n_checks)
    n_checks += 1
    expected_state_cid = _compute_w42_invariance_state_cid(
        invariance_branch=env.invariance_branch,
        role_handoff_signature_cid=(
            env.role_handoff_signature_cid),
        integrated_services_pre_w42=(
            env.integrated_services_pre_w42),
        integrated_services_post_w42=(
            env.integrated_services_post_w42),
        cell_index=int(env.cell_index),
    )
    if expected_state_cid != env.invariance_state_cid:
        return LatentVerificationOutcome(
            ok=False, reason="w42_invariance_state_cid_mismatch",
            n_checks=n_checks)
    n_checks += 1
    expected_decision_cid = _compute_w42_invariance_decision_cid(
        invariance_branch=env.invariance_branch,
        integrated_services_post_w42=(
            env.integrated_services_post_w42),
        invariance_score=float(env.invariance_score),
        n_w41_visible_tokens=int(env.n_w41_visible_tokens),
        n_w42_visible_tokens=int(env.n_w42_visible_tokens),
        n_w42_overhead_tokens=int(env.n_w42_overhead_tokens),
    )
    if expected_decision_cid != env.invariance_decision_cid:
        return LatentVerificationOutcome(
            ok=False,
            reason="w42_invariance_decision_cid_mismatch",
            n_checks=n_checks)
    n_checks += 1
    expected_audit_cid = _compute_w42_invariance_audit_cid(
        invariance_branch=env.invariance_branch,
        role_handoff_signature_cid=(
            env.role_handoff_signature_cid),
        policy_entry_cid=env.policy_entry_cid,
        integrated_services_pre_w42=(
            env.integrated_services_pre_w42),
        expected_services=env.expected_services,
        integrated_services_post_w42=(
            env.integrated_services_post_w42),
        invariance_score=float(env.invariance_score),
    )
    if expected_audit_cid != env.invariance_audit_cid:
        return LatentVerificationOutcome(
            ok=False,
            reason="w42_invariance_audit_cid_mismatch",
            n_checks=n_checks)
    n_checks += 1
    expected_witness_cid = _compute_w42_invariance_witness_cid(
        invariance_branch=env.invariance_branch,
        role_handoff_signature_cid=(
            env.role_handoff_signature_cid),
        n_w41_visible_tokens=int(env.n_w41_visible_tokens),
        n_w42_visible_tokens=int(env.n_w42_visible_tokens),
        n_w42_overhead_tokens=int(env.n_w42_overhead_tokens),
        n_structured_bits=int(env.n_structured_bits),
    )
    if expected_witness_cid != env.invariance_witness_cid:
        return LatentVerificationOutcome(
            ok=False,
            reason="w42_invariance_witness_cid_mismatch",
            n_checks=n_checks)
    n_checks += 1
    # Score is a float in [0, 1].
    if not (0.0 <= float(env.invariance_score) <= 1.0):
        return LatentVerificationOutcome(
            ok=False, reason="w42_invariance_score_invalid",
            n_checks=n_checks)
    n_checks += 1
    if (env.n_w41_visible_tokens < 0
            or env.n_w42_overhead_tokens < 0
            or env.n_w42_visible_tokens != (
                int(env.n_w41_visible_tokens)
                + int(env.n_w42_overhead_tokens))):
        return LatentVerificationOutcome(
            ok=False, reason="w42_token_accounting_invalid",
            n_checks=n_checks)
    n_checks += 1
    expected_manifest = _compute_w42_manifest_v12_cid(
        parent_w41_cid=env.parent_w41_cid,
        invariance_state_cid=env.invariance_state_cid,
        invariance_decision_cid=env.invariance_decision_cid,
        invariance_audit_cid=env.invariance_audit_cid,
        invariance_witness_cid=env.invariance_witness_cid,
        role_handoff_signature_cid=(
            env.role_handoff_signature_cid),
    )
    if expected_manifest != env.manifest_v12_cid:
        return LatentVerificationOutcome(
            ok=False, reason="w42_manifest_v12_cid_mismatch",
            n_checks=n_checks)
    n_checks += 1
    if env.recompute_w42_cid() != env.w42_cid:
        return LatentVerificationOutcome(
            ok=False, reason="w42_outer_cid_mismatch",
            n_checks=n_checks)
    n_checks += 1
    return LatentVerificationOutcome(
        ok=True, reason="ok", n_checks=n_checks)


# =============================================================================
# Orchestrator
# =============================================================================

@dataclasses.dataclass
class RoleInvariantSynthesisOrchestrator:
    """W42 cross-role-invariant synthesis orchestrator.

    Wraps :class:`IntegratedSynthesisOrchestrator` (W41).  At each
    cell, computes a role-handoff-signature CID over the input
    handoffs, looks it up in the registered policy, and ratifies
    or abstains based on whether the W41 integrated services
    match the policy's expected services.
    """

    inner: IntegratedSynthesisOrchestrator
    registry: RoleInvariantSynthesisRegistry
    enabled: bool = True
    require_w42_verification: bool = True

    _last_result: "W42RoleInvariantResult | None" = None
    _last_envelope: (
        "RoleInvariantSynthesisRatificationEnvelope | None") = None
    _last_signature_cid: str = ""
    _cell_index: int = 0

    @property
    def schema(self) -> "SchemaCapsule | None":
        return self.registry.schema

    def reset_session(self) -> None:
        self.inner.reset_session()
        self._last_result = None
        self._last_envelope = None
        self._last_signature_cid = ""
        self._cell_index = 0

    def set_response_provider(self, provider: Any) -> None:
        self.inner.set_response_provider(provider)

    def decode(self, handoffs: Sequence[Any]) -> dict[str, Any]:
        return self.decode_rounds([handoffs])

    def decode_rounds(
            self,
            per_round_handoffs: Sequence[Sequence[Any]],
    ) -> dict[str, Any]:
        out = self.inner.decode_rounds(per_round_handoffs)

        # Compute the role-handoff signature now (we still need it
        # for state recording even on disabled / trivial paths).
        signature_cid = compute_role_handoff_signature_cid(
            per_round_handoffs)
        self._last_signature_cid = signature_cid

        if not self.enabled or self.schema is None:
            self._cell_index += 1
            return self._pack(
                out=out,
                decoder_branch=W42_BRANCH_INVARIANCE_DISABLED,
                envelope=None,
                ratified=False,
                verify_ok=False,
                verify_reason="disabled",
                n_w42_overhead=0,
                invariance_branch=W42_BRANCH_INVARIANCE_DISABLED,
                role_handoff_signature_cid=signature_cid,
                policy_entry_cid="",
                integrated_services_pre_w42=tuple(
                    out.get("services", ())),
                expected_services=(),
                integrated_services_post_w42=tuple(
                    out.get("services", ())),
                invariance_score=0.0,
                parent_w41_cid="",
                invariance_state_cid="",
                invariance_decision_cid="",
                invariance_audit_cid="",
                invariance_witness_cid="",
                manifest_v12_cid="",
                w42_cid="",
                n_envelope_bytes=0,
                n_structured_bits=0,
            )

        # Trivial passthrough: when the W42 layer is fully disabled,
        # reduce to W41 byte-for-byte.
        if self.registry.is_trivial:
            self._cell_index += 1
            return self._pack(
                out=out,
                decoder_branch=(
                    W42_BRANCH_TRIVIAL_INVARIANCE_PASSTHROUGH),
                envelope=None,
                ratified=True,
                verify_ok=True,
                verify_reason="trivial_passthrough",
                n_w42_overhead=0,
                invariance_branch=(
                    W42_BRANCH_TRIVIAL_INVARIANCE_PASSTHROUGH),
                role_handoff_signature_cid="",
                policy_entry_cid="",
                integrated_services_pre_w42=tuple(
                    out.get("services", ())),
                expected_services=(),
                integrated_services_post_w42=tuple(
                    out.get("services", ())),
                invariance_score=0.0,
                parent_w41_cid="",
                invariance_state_cid="",
                invariance_decision_cid="",
                invariance_audit_cid="",
                invariance_witness_cid="",
                manifest_v12_cid="",
                w42_cid="",
                n_envelope_bytes=0,
                n_structured_bits=0,
            )

        w41_result = self.inner.last_result
        w41_envelope = self.inner.last_envelope
        n_w41_visible = int(
            w41_result.n_w41_visible_tokens
            if w41_result is not None else 0)
        integrated_services_pre = tuple(
            sorted({str(s) for s in out.get("services", ()) if s}))

        # Look up policy.
        entry = self.registry.policy_registry.lookup(signature_cid)
        if entry is None:
            policy_match_found = False
            expected_services: tuple[str, ...] = ()
            policy_entry_cid = ""
        else:
            policy_match_found = True
            expected_services = tuple(
                sorted({str(s) for s in entry.expected_services
                        if s}))
            policy_entry_cid = entry.cid()

        invariance_branch, post_services, score = (
            select_role_invariance_decision(
                integrated_services=integrated_services_pre,
                expected_services=expected_services,
                policy_match_found=policy_match_found,
            ))

        # If abstention disabled and we'd abstain, fall back to
        # ratification on integrated services pre.
        if (invariance_branch
                == W42_BRANCH_INVARIANCE_DIVERGED_ABSTAINED
                and not self.registry.abstain_on_invariance_diverged):
            invariance_branch = W42_BRANCH_INVARIANCE_RATIFIED
            post_services = integrated_services_pre

        parent_w41_cid = (
            str(w41_envelope.w41_cid)
            if w41_envelope is not None else "")
        if not parent_w41_cid:
            parent_payload = _w42_canonical_bytes({
                "inner_w41_decoder_branch": str(
                    w41_result.decoder_branch
                    if w41_result is not None else ""),
                "n_w41_visible": int(n_w41_visible),
                "cell_index": int(self._cell_index),
            })
            parent_w41_cid = hashlib.sha256(
                parent_payload).hexdigest()

        # W42 overhead: 1 visible token per cell when active and
        # not on the no-trigger / no-policy / disabled / trivial
        # paths.
        if invariance_branch in (
                W42_BRANCH_INVARIANCE_RATIFIED,
                W42_BRANCH_INVARIANCE_DIVERGED_ABSTAINED):
            n_w42_overhead = 1
        else:
            n_w42_overhead = 0
        n_w42_visible = int(n_w41_visible) + int(n_w42_overhead)

        invariance_state_cid = _compute_w42_invariance_state_cid(
            invariance_branch=invariance_branch,
            role_handoff_signature_cid=signature_cid,
            integrated_services_pre_w42=integrated_services_pre,
            integrated_services_post_w42=post_services,
            cell_index=int(self._cell_index),
        )
        invariance_decision_cid = (
            _compute_w42_invariance_decision_cid(
                invariance_branch=invariance_branch,
                integrated_services_post_w42=post_services,
                invariance_score=float(score),
                n_w41_visible_tokens=n_w41_visible,
                n_w42_visible_tokens=n_w42_visible,
                n_w42_overhead_tokens=n_w42_overhead,
            ))
        invariance_audit_cid = _compute_w42_invariance_audit_cid(
            invariance_branch=invariance_branch,
            role_handoff_signature_cid=signature_cid,
            policy_entry_cid=policy_entry_cid,
            integrated_services_pre_w42=integrated_services_pre,
            expected_services=expected_services,
            integrated_services_post_w42=post_services,
            invariance_score=float(score),
        )

        # W41 envelope structured bits (if any) + 4*256 W42 CIDs +
        # 256 for the signature CID + 256 for the policy_entry CID.
        w41_structured_bits = (
            int(w41_envelope.n_structured_bits)
            if (w41_envelope is not None
                and hasattr(w41_envelope, "n_structured_bits"))
            else 0)
        n_structured_bits = w41_structured_bits + 6 * 256

        invariance_witness_cid = (
            _compute_w42_invariance_witness_cid(
                invariance_branch=invariance_branch,
                role_handoff_signature_cid=signature_cid,
                n_w41_visible_tokens=n_w41_visible,
                n_w42_visible_tokens=n_w42_visible,
                n_w42_overhead_tokens=n_w42_overhead,
                n_structured_bits=n_structured_bits,
            ))
        manifest_v12_cid = (
            "" if self.registry.manifest_v12_disabled
            else _compute_w42_manifest_v12_cid(
                parent_w41_cid=parent_w41_cid,
                invariance_state_cid=invariance_state_cid,
                invariance_decision_cid=invariance_decision_cid,
                invariance_audit_cid=invariance_audit_cid,
                invariance_witness_cid=invariance_witness_cid,
                role_handoff_signature_cid=signature_cid,
            ))

        if manifest_v12_cid:
            w42_cid = _compute_w42_outer_cid(
                schema_cid=str(self.schema.cid),
                parent_w41_cid=parent_w41_cid,
                manifest_v12_cid=manifest_v12_cid,
                cell_index=int(self._cell_index),
            )
        else:
            w42_cid = ""

        envelope = RoleInvariantSynthesisRatificationEnvelope(
            schema_version=W42_ROLE_INVARIANT_SCHEMA_VERSION,
            schema_cid=str(self.schema.cid),
            parent_w41_cid=str(parent_w41_cid),
            cell_index=int(self._cell_index),
            invariance_branch=str(invariance_branch),
            role_handoff_signature_cid=signature_cid,
            policy_entry_cid=str(policy_entry_cid),
            integrated_services_pre_w42=tuple(
                integrated_services_pre),
            expected_services=tuple(expected_services),
            integrated_services_post_w42=tuple(post_services),
            invariance_score=float(score),
            invariance_state_cid=invariance_state_cid,
            invariance_decision_cid=invariance_decision_cid,
            invariance_audit_cid=invariance_audit_cid,
            invariance_witness_cid=invariance_witness_cid,
            manifest_v12_cid=manifest_v12_cid,
            n_w41_visible_tokens=int(n_w41_visible),
            n_w42_visible_tokens=int(n_w42_visible),
            n_w42_overhead_tokens=int(n_w42_overhead),
            n_structured_bits=int(n_structured_bits),
            w42_cid=w42_cid,
        )

        outcome = verify_role_invariant_synthesis_ratification(
            envelope,
            registered_schema=self.schema,
            registered_parent_w41_cid=parent_w41_cid,
        )
        verify_ok = bool(outcome.ok)
        verify_reason = str(outcome.reason)
        if not verify_ok and self.require_w42_verification:
            self._cell_index += 1
            return self._pack(
                out=out,
                decoder_branch=W42_BRANCH_INVARIANCE_REJECTED,
                envelope=envelope,
                ratified=False,
                verify_ok=False,
                verify_reason=verify_reason,
                n_w42_overhead=0,
                invariance_branch=W42_BRANCH_INVARIANCE_REJECTED,
                role_handoff_signature_cid=signature_cid,
                policy_entry_cid=policy_entry_cid,
                integrated_services_pre_w42=integrated_services_pre,
                expected_services=expected_services,
                integrated_services_post_w42=(),
                invariance_score=float(score),
                parent_w41_cid=parent_w41_cid,
                invariance_state_cid=invariance_state_cid,
                invariance_decision_cid=invariance_decision_cid,
                invariance_audit_cid=invariance_audit_cid,
                invariance_witness_cid=invariance_witness_cid,
                manifest_v12_cid=manifest_v12_cid,
                w42_cid=w42_cid,
                n_envelope_bytes=0,
                n_structured_bits=int(n_structured_bits),
            )

        out_local = dict(out)
        out_local["services"] = tuple(post_services)
        self._cell_index += 1

        return self._pack(
            out=out_local,
            decoder_branch=invariance_branch,
            envelope=envelope,
            ratified=bool(post_services),
            verify_ok=verify_ok,
            verify_reason=verify_reason,
            n_w42_overhead=n_w42_overhead,
            invariance_branch=invariance_branch,
            role_handoff_signature_cid=signature_cid,
            policy_entry_cid=policy_entry_cid,
            integrated_services_pre_w42=integrated_services_pre,
            expected_services=expected_services,
            integrated_services_post_w42=post_services,
            invariance_score=float(score),
            parent_w41_cid=parent_w41_cid,
            invariance_state_cid=invariance_state_cid,
            invariance_decision_cid=invariance_decision_cid,
            invariance_audit_cid=invariance_audit_cid,
            invariance_witness_cid=invariance_witness_cid,
            manifest_v12_cid=manifest_v12_cid,
            w42_cid=w42_cid,
            n_envelope_bytes=0,
            n_structured_bits=int(n_structured_bits),
        )

    def _pack(
            self,
            *,
            out: dict[str, Any],
            decoder_branch: str,
            envelope: (
                "RoleInvariantSynthesisRatificationEnvelope | None"),
            ratified: bool,
            verify_ok: bool,
            verify_reason: str,
            n_w42_overhead: int,
            invariance_branch: str,
            role_handoff_signature_cid: str,
            policy_entry_cid: str,
            integrated_services_pre_w42: tuple[str, ...],
            expected_services: tuple[str, ...],
            integrated_services_post_w42: tuple[str, ...],
            invariance_score: float,
            parent_w41_cid: str,
            invariance_state_cid: str,
            invariance_decision_cid: str,
            invariance_audit_cid: str,
            invariance_witness_cid: str,
            manifest_v12_cid: str,
            w42_cid: str,
            n_envelope_bytes: int,
            n_structured_bits: int,
    ) -> dict[str, Any]:
        n_w41_visible = 0
        w41_state = out.get("integrated_synthesis")
        if w41_state is not None:
            n_w41_visible = int(
                w41_state.get("n_w41_visible_tokens", 0))
        n_w42_visible = (
            int(n_w41_visible) + int(n_w42_overhead))
        wire = max(1, int(n_w42_overhead))
        cram_factor = (
            float(n_structured_bits) / float(wire)
            if int(n_structured_bits) > 0 else 0.0)

        result = W42RoleInvariantResult(
            answer=dict(out),
            inner_w41_decoder_branch=str(
                w41_state.get("decoder_branch", "")
                if w41_state is not None else ""),
            decoder_branch=str(decoder_branch),
            invariance_branch=str(invariance_branch),
            role_handoff_signature_cid=str(
                role_handoff_signature_cid),
            policy_entry_cid=str(policy_entry_cid),
            integrated_services_pre_w42=tuple(
                integrated_services_pre_w42),
            expected_services=tuple(expected_services),
            integrated_services_post_w42=tuple(
                integrated_services_post_w42),
            invariance_score=float(invariance_score),
            parent_w41_cid=str(parent_w41_cid),
            cell_index=int(
                self._cell_index - 1
                if self._cell_index > 0 else 0),
            n_w41_visible_tokens=int(n_w41_visible),
            n_w42_visible_tokens=int(n_w42_visible),
            n_w42_overhead_tokens=int(n_w42_overhead),
            w42_cid=str(w42_cid),
            manifest_v12_cid=str(manifest_v12_cid),
            invariance_state_cid=str(invariance_state_cid),
            invariance_decision_cid=str(
                invariance_decision_cid),
            invariance_audit_cid=str(invariance_audit_cid),
            invariance_witness_cid=str(
                invariance_witness_cid),
            ratified=bool(ratified),
            verification_ok=bool(verify_ok),
            verification_reason=str(verify_reason),
            n_envelope_bytes=int(n_envelope_bytes),
            n_structured_bits=int(n_structured_bits),
            cram_factor_w42=float(cram_factor),
        )
        self._last_result = result
        self._last_envelope = envelope
        out_local = dict(out)
        out_local["role_invariant_synthesis"] = result.as_dict()
        if envelope is not None:
            out_local["role_invariant_synthesis_envelope"] = (
                envelope.as_dict())
        return out_local

    @property
    def last_result(self) -> "W42RoleInvariantResult | None":
        return self._last_result

    @property
    def last_envelope(self) -> (
            "RoleInvariantSynthesisRatificationEnvelope | None"):
        return self._last_envelope

    @property
    def last_signature_cid(self) -> str:
        return self._last_signature_cid


# =============================================================================
# Builders
# =============================================================================

def build_trivial_role_invariant_registry(
        *,
        schema: SchemaCapsule,
        inner_w41_registry: IntegratedSynthesisRegistry | None = None,
) -> RoleInvariantSynthesisRegistry:
    return RoleInvariantSynthesisRegistry(
        schema=schema,
        inner_w41_registry=inner_w41_registry,
        policy_registry=RoleInvariancePolicyRegistry(),
        invariance_enabled=False,
        manifest_v12_disabled=True,
        abstain_on_invariance_diverged=False,
    )


def build_role_invariant_registry(
        *,
        schema: SchemaCapsule,
        inner_w41_registry: IntegratedSynthesisRegistry,
        policy_entries: Sequence[RoleInvariancePolicyEntry] = (),
        invariance_enabled: bool = True,
        manifest_v12_disabled: bool = False,
        abstain_on_invariance_diverged: bool = True,
) -> RoleInvariantSynthesisRegistry:
    policy = RoleInvariancePolicyRegistry()
    for e in policy_entries:
        policy.register(e)
    return RoleInvariantSynthesisRegistry(
        schema=schema,
        inner_w41_registry=inner_w41_registry,
        policy_registry=policy,
        invariance_enabled=bool(invariance_enabled),
        manifest_v12_disabled=bool(manifest_v12_disabled),
        abstain_on_invariance_diverged=bool(
            abstain_on_invariance_diverged),
    )


__all__ = [
    "W42_ROLE_INVARIANT_SCHEMA_VERSION",
    "W42_BRANCH_TRIVIAL_INVARIANCE_PASSTHROUGH",
    "W42_BRANCH_INVARIANCE_DISABLED",
    "W42_BRANCH_INVARIANCE_REJECTED",
    "W42_BRANCH_INVARIANCE_NO_TRIGGER",
    "W42_BRANCH_INVARIANCE_RATIFIED",
    "W42_BRANCH_INVARIANCE_DIVERGED_ABSTAINED",
    "W42_BRANCH_INVARIANCE_NO_POLICY",
    "W42_ALL_BRANCHES",
    "compute_role_handoff_signature_cid",
    "select_role_invariance_decision",
    "verify_role_invariant_synthesis_ratification",
    "RoleInvariantSynthesisRatificationEnvelope",
    "RoleInvariancePolicyEntry",
    "RoleInvariancePolicyRegistry",
    "RoleInvariantSynthesisRegistry",
    "W42RoleInvariantResult",
    "RoleInvariantSynthesisOrchestrator",
    "build_trivial_role_invariant_registry",
    "build_role_invariant_registry",
]
