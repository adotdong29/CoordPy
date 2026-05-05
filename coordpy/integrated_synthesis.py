"""W41 Integrated Multi-Agent Context Synthesis (SDK v3.42).

W41 is the first capsule-native synthesis layer that *jointly*
binds the strongest old-line explicit-capsule trust adjudication
(W21..W40) and the strongest cross-role / multi-round / bundle-
aware producer-side decoder family (W7..W11) into a single
integrated orchestration path with one ``manifest-v11`` envelope
that mechanically binds both axes plus a content-addressed
cross-axis witness.

This module is a *strict superset* and *strictly additive* on top
of W40.  The W40 inner orchestrator's ``decode_rounds`` method is
the load-bearing inner step; W41 wraps the W40 result with a
cross-axis classification and seals the outcome under a new
manifest.  No W22..W40 internals are touched.

Honest scope (do-not-overstate)
-------------------------------

W41 does **not** add a new transformer-internal mechanism.  It is
an *integration* milestone: it composes the strongest capsule-
layer proxies the repo already supports into a single auditable
end-to-end path, and it provides the first end-to-end multi-agent
context benchmark family (R-88) where producer-side ambiguity
AND multi-host trust adjudication are simultaneously measurable
on the *same* cell.  W41 therefore does NOT close
``W40-L-COORDINATED-DIVERSE-RESPONSE-CAP``.  W41 introduces a new
proved-conditional limitation theorem,
``W41-L-COMPOSITE-COLLUSION-CAP``: when both the producer-side
ambiguity and the trust-side collusion are coordinated by an
adversary, integration cannot recover at the capsule layer.

What W41 adds beyond W40
------------------------

1. **Cross-axis decision branch.**  Per cell, W41 classifies which
   axis was load-bearing on this cell from the W40 inner result:

   * ``producer_axis_only``  — the multi-round bundle decoder
     produced a non-empty ``services`` set AND the trust axis did
     not trigger (W40 RESPONSE_SIGNATURE_NO_TRIGGER).
   * ``trust_axis_only``     — the trust axis fired (W40 ratified
     or abstained), AND the producer axis did not change the
     answer.
   * ``both_axes``           — both axes fired and agreed.
   * ``axes_diverged``       — both axes fired and disagreed; W41
     abstains via SYNTHESIS_AXIS_DIVERGENCE_ABSTAINED.
   * ``neither_axis``        — neither axis carried any signal;
     W41 falls through to the substrate path.
   The classification is mechanical and zero-parameter.

2. **manifest-v11 CID.**  SHA-256 over four component CIDs:
   ``parent_w40_cid``, ``integrated_decision_cid``,
   ``synthesis_audit_cid``, ``cross_axis_witness_cid``.  Detects
   cross-component swaps that the W40 manifest-v10 alone cannot
   detect (e.g. an adversary swapping the W40 envelope but
   leaving the integrated-synthesis decision intact).

3. **Synthesis audit.**  Content-addressed audit envelope that
   records the per-axis branches and the cross-axis decision in
   one canonical form, namespaced as ``w41_synthesis_audit`` so
   substituting a W22..W40 audit for it is mechanically
   detected.

4. **Cross-axis witness.**  Content-addressed per-cell witness
   that binds ``(producer_axis_branch, trust_axis_branch,
   integrated_branch, n_w40_visible_tokens, n_w41_visible_tokens,
   structured_bits)``, namespaced as ``w41_cross_axis_witness``.

5. **Verifier with 14 enumerated W41 failure modes** disjoint
   from W22..W40's 168 cumulative modes.

The W41 mechanism is closed-form, deterministic, zero-parameter,
controller-pre-registered, and audited at the capsule layer.  It
does not read transformer hidden states, KV cache, attention
weights, or embeddings; it does not transplant runtime state; it
does not claim native latent transfer.
"""

from __future__ import annotations

import dataclasses
import hashlib
from typing import Any, Sequence

from coordpy.team_coord import (
    SchemaCapsule,
    LatentVerificationOutcome,
    CrossHostResponseHeterogeneityOrchestrator,
    CrossHostResponseHeterogeneityRatificationEnvelope,
    CrossHostResponseHeterogeneityRegistry,
    W40_RESPONSE_HETEROGENEITY_SCHEMA_VERSION,
    W40_BRANCH_RESPONSE_SIGNATURE_RESOLVED,
    W40_BRANCH_RESPONSE_SIGNATURE_DIVERSE,
    W40_BRANCH_RESPONSE_SIGNATURE_COLLAPSE_ABSTAINED,
    W40_BRANCH_RESPONSE_SIGNATURE_NO_REFERENCES,
    W40_BRANCH_RESPONSE_SIGNATURE_NO_TRIGGER,
    W40_BRANCH_RESPONSE_SIGNATURE_INSUFFICIENT,
    W40_BRANCH_RESPONSE_SIGNATURE_INCOMPLETE,
    W40_BRANCH_RESPONSE_SIGNATURE_DISABLED,
    W40_BRANCH_TRIVIAL_RESPONSE_SIGNATURE_PASSTHROUGH,
    W40_BRANCH_RESPONSE_SIGNATURE_REJECTED,
    _DecodedHandoff,
)


# =============================================================================
# W41 schema, branch constants
# =============================================================================

W41_INTEGRATED_SYNTHESIS_SCHEMA_VERSION: str = (
    "coordpy.integrated_synthesis.v1")

# Producer-axis branches: outcome of the multi-round bundle decoder
# (W7..W11) layer for this cell.
W41_PRODUCER_AXIS_FIRED: str = "producer_axis_fired"
W41_PRODUCER_AXIS_NO_TRIGGER: str = "producer_axis_no_trigger"

# Trust-axis branches: outcome of the W21..W40 ratification chain
# for this cell.
W41_TRUST_AXIS_RATIFIED: str = "trust_axis_ratified"
W41_TRUST_AXIS_ABSTAINED: str = "trust_axis_abstained"
W41_TRUST_AXIS_NO_TRIGGER: str = "trust_axis_no_trigger"

# Cross-axis (integrated) decision branches.
W41_BRANCH_TRIVIAL_INTEGRATED_PASSTHROUGH: str = (
    "trivial_integrated_passthrough")
W41_BRANCH_INTEGRATED_DISABLED: str = "integrated_disabled"
W41_BRANCH_INTEGRATED_REJECTED: str = "integrated_rejected"
W41_BRANCH_INTEGRATED_PRODUCER_ONLY: str = (
    "integrated_producer_only")
W41_BRANCH_INTEGRATED_TRUST_ONLY: str = "integrated_trust_only"
W41_BRANCH_INTEGRATED_BOTH_AXES: str = "integrated_both_axes"
W41_BRANCH_INTEGRATED_AXES_DIVERGED_ABSTAINED: str = (
    "integrated_axes_diverged_abstained")
W41_BRANCH_INTEGRATED_NEITHER_AXIS: str = (
    "integrated_neither_axis")

W41_ALL_BRANCHES: tuple[str, ...] = (
    W41_BRANCH_TRIVIAL_INTEGRATED_PASSTHROUGH,
    W41_BRANCH_INTEGRATED_DISABLED,
    W41_BRANCH_INTEGRATED_REJECTED,
    W41_BRANCH_INTEGRATED_PRODUCER_ONLY,
    W41_BRANCH_INTEGRATED_TRUST_ONLY,
    W41_BRANCH_INTEGRATED_BOTH_AXES,
    W41_BRANCH_INTEGRATED_AXES_DIVERGED_ABSTAINED,
    W41_BRANCH_INTEGRATED_NEITHER_AXIS,
)

W41_PRODUCER_BRANCHES: frozenset[str] = frozenset({
    W41_PRODUCER_AXIS_FIRED, W41_PRODUCER_AXIS_NO_TRIGGER,
})
W41_TRUST_BRANCHES: frozenset[str] = frozenset({
    W41_TRUST_AXIS_RATIFIED,
    W41_TRUST_AXIS_ABSTAINED,
    W41_TRUST_AXIS_NO_TRIGGER,
})


# =============================================================================
# W41 deterministic helpers (zero-parameter; closed-form)
# =============================================================================

def _w41_canonical_bytes(payload: dict[str, Any]) -> bytes:
    """Canonical JSON bytes for a CID input.

    Sort keys; UTF-8; no extra whitespace.  Inputs are ``str``,
    ``int``, ``float``, ``bool``, ``None``, list, or dict; this
    matches the canonicalisation used throughout W22..W40.
    """
    import json
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def classify_producer_axis_branch(
        *,
        services: Sequence[str],
) -> str:
    """Closed-form classification of the producer axis.

    The producer axis (W7..W11 multi-round bundle decoder) has
    fired on this cell iff the decoded ``services`` set is
    non-empty.  Empty ``services`` means the producer axis did
    not produce a decision (FIFO substrate or W11 abstention).
    """
    nonempty = bool(tuple(services))
    return (W41_PRODUCER_AXIS_FIRED if nonempty
            else W41_PRODUCER_AXIS_NO_TRIGGER)


def classify_trust_axis_branch(
        *,
        w40_projection_branch: str,
) -> str:
    """Closed-form classification of the trust axis.

    The trust axis (W21..W40 ratification chain) has *fired* on
    this cell iff the W40 projection branch is one of the
    ratification or abstention branches.  No-trigger branches and
    insufficient/incomplete branches are tagged as no-trigger.
    """
    pb = str(w40_projection_branch)
    if pb == W40_BRANCH_RESPONSE_SIGNATURE_DIVERSE:
        return W41_TRUST_AXIS_RATIFIED
    if pb == W40_BRANCH_RESPONSE_SIGNATURE_COLLAPSE_ABSTAINED:
        return W41_TRUST_AXIS_ABSTAINED
    return W41_TRUST_AXIS_NO_TRIGGER


def select_integrated_synthesis_decision(
        *,
        producer_axis_branch: str,
        trust_axis_branch: str,
        producer_services: Sequence[str],
        trust_services: Sequence[str],
) -> tuple[str, tuple[str, ...]]:
    """Closed-form integrated-synthesis decision.

    Returns ``(integrated_branch, integrated_services)``.

    The classification is mechanical:

    * Producer fired AND trust fired AND services agree
      => INTEGRATED_BOTH_AXES (intersection used).
    * Producer fired AND trust fired AND services disagree
      => INTEGRATED_AXES_DIVERGED_ABSTAINED (empty services).
    * Producer fired AND trust no-trigger
      => INTEGRATED_PRODUCER_ONLY (producer services).
    * Producer fired AND trust abstained
      => INTEGRATED_TRUST_ONLY (empty services -- trust over-rides
      producer because trust abstention is a safety branch).
    * Producer no-trigger AND trust ratified
      => INTEGRATED_TRUST_ONLY (trust services).
    * Producer no-trigger AND trust abstained
      => INTEGRATED_TRUST_ONLY (empty services).
    * Producer no-trigger AND trust no-trigger
      => INTEGRATED_NEITHER_AXIS (empty services).
    """
    p = str(producer_axis_branch)
    t = str(trust_axis_branch)
    p_set = tuple(sorted({str(s) for s in producer_services if s}))
    t_set = tuple(sorted({str(s) for s in trust_services if s}))

    if p == W41_PRODUCER_AXIS_FIRED:
        if t == W41_TRUST_AXIS_RATIFIED:
            if p_set == t_set:
                return W41_BRANCH_INTEGRATED_BOTH_AXES, p_set
            inter = tuple(
                sorted(set(p_set) & set(t_set)))
            if inter:
                return (
                    W41_BRANCH_INTEGRATED_BOTH_AXES, inter)
            return (
                W41_BRANCH_INTEGRATED_AXES_DIVERGED_ABSTAINED,
                ())
        if t == W41_TRUST_AXIS_ABSTAINED:
            # Trust safety over-rides producer.
            return W41_BRANCH_INTEGRATED_TRUST_ONLY, ()
        # trust_no_trigger
        return W41_BRANCH_INTEGRATED_PRODUCER_ONLY, p_set

    # producer no-trigger
    if t == W41_TRUST_AXIS_RATIFIED:
        return W41_BRANCH_INTEGRATED_TRUST_ONLY, t_set
    if t == W41_TRUST_AXIS_ABSTAINED:
        return W41_BRANCH_INTEGRATED_TRUST_ONLY, ()
    return W41_BRANCH_INTEGRATED_NEITHER_AXIS, ()


# =============================================================================
# CID computations (manifest-v11 component CIDs)
# =============================================================================

def _compute_w41_synthesis_state_cid(
        *,
        producer_axis_branch: str,
        trust_axis_branch: str,
        integrated_branch: str,
        integrated_services: tuple[str, ...],
        cell_index: int,
) -> str:
    """SHA-256 over the canonical state tuple."""
    payload = {
        "kind": "w41_synthesis_state",
        "producer_axis_branch": str(producer_axis_branch),
        "trust_axis_branch": str(trust_axis_branch),
        "integrated_branch": str(integrated_branch),
        "integrated_services": [
            str(s) for s in integrated_services],
        "cell_index": int(cell_index),
    }
    return hashlib.sha256(
        _w41_canonical_bytes(payload)).hexdigest()


def _compute_w41_synthesis_decision_cid(
        *,
        integrated_branch: str,
        integrated_services: tuple[str, ...],
        n_w40_visible_tokens: int,
        n_w41_visible_tokens: int,
        n_w41_overhead_tokens: int,
) -> str:
    payload = {
        "kind": "w41_synthesis_decision",
        "integrated_branch": str(integrated_branch),
        "integrated_services": [
            str(s) for s in integrated_services],
        "n_w40_visible_tokens": int(n_w40_visible_tokens),
        "n_w41_visible_tokens": int(n_w41_visible_tokens),
        "n_w41_overhead_tokens": int(n_w41_overhead_tokens),
    }
    return hashlib.sha256(
        _w41_canonical_bytes(payload)).hexdigest()


def _compute_w41_synthesis_audit_cid(
        *,
        integrated_branch: str,
        producer_axis_branch: str,
        trust_axis_branch: str,
        producer_services: tuple[str, ...],
        trust_services: tuple[str, ...],
        integrated_services: tuple[str, ...],
        w40_projection_branch: str,
) -> str:
    payload = {
        "kind": "w41_synthesis_audit",
        "integrated_branch": str(integrated_branch),
        "producer_axis_branch": str(producer_axis_branch),
        "trust_axis_branch": str(trust_axis_branch),
        "producer_services": [str(s) for s in producer_services],
        "trust_services": [str(s) for s in trust_services],
        "integrated_services": [
            str(s) for s in integrated_services],
        "w40_projection_branch": str(w40_projection_branch),
    }
    return hashlib.sha256(
        _w41_canonical_bytes(payload)).hexdigest()


def _compute_w41_cross_axis_witness_cid(
        *,
        producer_axis_branch: str,
        trust_axis_branch: str,
        integrated_branch: str,
        n_w40_visible_tokens: int,
        n_w41_visible_tokens: int,
        n_w41_overhead_tokens: int,
        n_structured_bits: int,
) -> str:
    payload = {
        "kind": "w41_cross_axis_witness",
        "producer_axis_branch": str(producer_axis_branch),
        "trust_axis_branch": str(trust_axis_branch),
        "integrated_branch": str(integrated_branch),
        "n_w40_visible_tokens": int(n_w40_visible_tokens),
        "n_w41_visible_tokens": int(n_w41_visible_tokens),
        "n_w41_overhead_tokens": int(n_w41_overhead_tokens),
        "n_structured_bits": int(n_structured_bits),
    }
    return hashlib.sha256(
        _w41_canonical_bytes(payload)).hexdigest()


def _compute_w41_manifest_v11_cid(
        *,
        parent_w40_cid: str,
        synthesis_state_cid: str,
        synthesis_decision_cid: str,
        synthesis_audit_cid: str,
        cross_axis_witness_cid: str,
) -> str:
    payload = {
        "kind": "w41_manifest_v11",
        "parent_w40_cid": str(parent_w40_cid),
        "synthesis_state_cid": str(synthesis_state_cid),
        "synthesis_decision_cid": str(synthesis_decision_cid),
        "synthesis_audit_cid": str(synthesis_audit_cid),
        "cross_axis_witness_cid": str(cross_axis_witness_cid),
    }
    return hashlib.sha256(
        _w41_canonical_bytes(payload)).hexdigest()


def _compute_w41_outer_cid(
        *,
        schema_cid: str,
        parent_w40_cid: str,
        manifest_v11_cid: str,
        cell_index: int,
) -> str:
    payload = {
        "kind": "w41_outer",
        "schema_cid": str(schema_cid),
        "parent_w40_cid": str(parent_w40_cid),
        "manifest_v11_cid": str(manifest_v11_cid),
        "cell_index": int(cell_index),
    }
    return hashlib.sha256(
        _w41_canonical_bytes(payload)).hexdigest()


# =============================================================================
# Envelope, registry, result
# =============================================================================

@dataclasses.dataclass(frozen=True)
class IntegratedSynthesisRatificationEnvelope:
    """Sealed manifest-v11 envelope binding W40 + cross-axis state.

    Strict superset over the W40 envelope: every check the W40
    envelope passes also applies here, plus four new W41 checks
    (state/decision/audit/witness).
    """

    schema_version: str
    schema_cid: str
    parent_w40_cid: str
    cell_index: int

    producer_axis_branch: str
    trust_axis_branch: str
    integrated_branch: str

    producer_services: tuple[str, ...]
    trust_services: tuple[str, ...]
    integrated_services: tuple[str, ...]

    w40_projection_branch: str

    synthesis_state_cid: str
    synthesis_decision_cid: str
    synthesis_audit_cid: str
    cross_axis_witness_cid: str
    manifest_v11_cid: str

    n_w40_visible_tokens: int
    n_w41_visible_tokens: int
    n_w41_overhead_tokens: int
    n_structured_bits: int

    w41_cid: str

    def recompute_w41_cid(self) -> str:
        return _compute_w41_outer_cid(
            schema_cid=self.schema_cid,
            parent_w40_cid=self.parent_w40_cid,
            manifest_v11_cid=self.manifest_v11_cid,
            cell_index=int(self.cell_index),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "schema_cid": self.schema_cid,
            "parent_w40_cid": self.parent_w40_cid,
            "cell_index": int(self.cell_index),
            "producer_axis_branch": self.producer_axis_branch,
            "trust_axis_branch": self.trust_axis_branch,
            "integrated_branch": self.integrated_branch,
            "producer_services": list(self.producer_services),
            "trust_services": list(self.trust_services),
            "integrated_services": list(self.integrated_services),
            "w40_projection_branch": self.w40_projection_branch,
            "synthesis_state_cid": self.synthesis_state_cid,
            "synthesis_decision_cid": self.synthesis_decision_cid,
            "synthesis_audit_cid": self.synthesis_audit_cid,
            "cross_axis_witness_cid": self.cross_axis_witness_cid,
            "manifest_v11_cid": self.manifest_v11_cid,
            "n_w40_visible_tokens": int(self.n_w40_visible_tokens),
            "n_w41_visible_tokens": int(self.n_w41_visible_tokens),
            "n_w41_overhead_tokens": int(
                self.n_w41_overhead_tokens),
            "n_structured_bits": int(self.n_structured_bits),
            "w41_cid": self.w41_cid,
        }


@dataclasses.dataclass
class IntegratedSynthesisRegistry:
    """Registry for the W41 integrated-synthesis layer.

    The registry holds the SchemaCapsule (shared with W22..W40),
    a reference to the inner W40 registry (so the trust axis can
    be verified end-to-end), and the synthesis-layer enable
    flags.  When ``synthesis_enabled = False`` AND
    ``manifest_v11_disabled = True``, the W41 orchestrator
    reduces to W40 byte-for-byte (the W41-L-TRIVIAL-PASSTHROUGH
    falsifier).
    """

    schema: SchemaCapsule
    inner_w40_registry: CrossHostResponseHeterogeneityRegistry | None
    synthesis_enabled: bool = True
    manifest_v11_disabled: bool = False
    abstain_on_axes_diverged: bool = True
    require_both_axes_for_ratification: bool = False

    @property
    def is_trivial(self) -> bool:
        return (not self.synthesis_enabled
                and self.manifest_v11_disabled
                and not self.abstain_on_axes_diverged)


@dataclasses.dataclass
class W41IntegratedSynthesisResult:
    answer: dict[str, Any]
    inner_w40_decoder_branch: str
    decoder_branch: str
    integrated_branch: str
    producer_axis_branch: str
    trust_axis_branch: str
    producer_services: tuple[str, ...]
    trust_services: tuple[str, ...]
    integrated_services: tuple[str, ...]
    w40_projection_branch: str
    parent_w40_cid: str
    cell_index: int
    n_w40_visible_tokens: int
    n_w41_visible_tokens: int
    n_w41_overhead_tokens: int
    w41_cid: str
    manifest_v11_cid: str
    synthesis_state_cid: str
    synthesis_decision_cid: str
    synthesis_audit_cid: str
    cross_axis_witness_cid: str
    ratified: bool
    verification_ok: bool
    verification_reason: str
    n_envelope_bytes: int
    n_structured_bits: int
    cram_factor_w41: float

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


# =============================================================================
# Verifier
# =============================================================================

def verify_integrated_synthesis_ratification(
        env: IntegratedSynthesisRatificationEnvelope | None,
        *,
        registered_schema: SchemaCapsule,
        registered_parent_w40_cid: str,
) -> LatentVerificationOutcome:
    """Pure-function verifier for the W41 envelope.

    Enumerates 14 disjoint W41 failure modes, disjoint from
    W22..W40's 168 cumulative modes:

    1.  ``empty_w41_envelope``
    2.  ``w41_schema_version_unknown``
    3.  ``w41_schema_cid_mismatch``
    4.  ``w40_parent_cid_mismatch``  (W41-specific, namespaced)
    5.  ``w41_integrated_branch_unknown``
    6.  ``w41_producer_axis_branch_unknown``
    7.  ``w41_trust_axis_branch_unknown``
    8.  ``w41_synthesis_state_cid_mismatch``
    9.  ``w41_synthesis_decision_cid_mismatch``
    10. ``w41_synthesis_audit_cid_mismatch``
    11. ``w41_cross_axis_witness_cid_mismatch``
    12. ``w41_token_accounting_invalid``
    13. ``w41_manifest_v11_cid_mismatch``
    14. ``w41_outer_cid_mismatch``
    """
    n_checks = 0
    if env is None:
        return LatentVerificationOutcome(
            ok=False, reason="empty_w41_envelope",
            n_checks=n_checks)
    n_checks += 1
    if env.schema_version != (
            W41_INTEGRATED_SYNTHESIS_SCHEMA_VERSION):
        return LatentVerificationOutcome(
            ok=False, reason="w41_schema_version_unknown",
            n_checks=n_checks)
    n_checks += 1
    if env.schema_cid != registered_schema.cid:
        return LatentVerificationOutcome(
            ok=False, reason="w41_schema_cid_mismatch",
            n_checks=n_checks)
    n_checks += 1
    if env.parent_w40_cid != str(registered_parent_w40_cid):
        return LatentVerificationOutcome(
            ok=False, reason="w40_parent_cid_mismatch",
            n_checks=n_checks)
    n_checks += 1
    if env.integrated_branch not in W41_ALL_BRANCHES:
        return LatentVerificationOutcome(
            ok=False, reason="w41_integrated_branch_unknown",
            n_checks=n_checks)
    n_checks += 1
    if env.producer_axis_branch not in W41_PRODUCER_BRANCHES:
        return LatentVerificationOutcome(
            ok=False, reason="w41_producer_axis_branch_unknown",
            n_checks=n_checks)
    n_checks += 1
    if env.trust_axis_branch not in W41_TRUST_BRANCHES:
        return LatentVerificationOutcome(
            ok=False, reason="w41_trust_axis_branch_unknown",
            n_checks=n_checks)
    n_checks += 1
    expected_state_cid = _compute_w41_synthesis_state_cid(
        producer_axis_branch=env.producer_axis_branch,
        trust_axis_branch=env.trust_axis_branch,
        integrated_branch=env.integrated_branch,
        integrated_services=env.integrated_services,
        cell_index=int(env.cell_index),
    )
    if expected_state_cid != env.synthesis_state_cid:
        return LatentVerificationOutcome(
            ok=False, reason="w41_synthesis_state_cid_mismatch",
            n_checks=n_checks)
    n_checks += 1
    expected_decision_cid = _compute_w41_synthesis_decision_cid(
        integrated_branch=env.integrated_branch,
        integrated_services=env.integrated_services,
        n_w40_visible_tokens=int(env.n_w40_visible_tokens),
        n_w41_visible_tokens=int(env.n_w41_visible_tokens),
        n_w41_overhead_tokens=int(env.n_w41_overhead_tokens),
    )
    if expected_decision_cid != env.synthesis_decision_cid:
        return LatentVerificationOutcome(
            ok=False, reason="w41_synthesis_decision_cid_mismatch",
            n_checks=n_checks)
    n_checks += 1
    expected_audit_cid = _compute_w41_synthesis_audit_cid(
        integrated_branch=env.integrated_branch,
        producer_axis_branch=env.producer_axis_branch,
        trust_axis_branch=env.trust_axis_branch,
        producer_services=env.producer_services,
        trust_services=env.trust_services,
        integrated_services=env.integrated_services,
        w40_projection_branch=env.w40_projection_branch,
    )
    if expected_audit_cid != env.synthesis_audit_cid:
        return LatentVerificationOutcome(
            ok=False, reason="w41_synthesis_audit_cid_mismatch",
            n_checks=n_checks)
    n_checks += 1
    expected_witness_cid = _compute_w41_cross_axis_witness_cid(
        producer_axis_branch=env.producer_axis_branch,
        trust_axis_branch=env.trust_axis_branch,
        integrated_branch=env.integrated_branch,
        n_w40_visible_tokens=int(env.n_w40_visible_tokens),
        n_w41_visible_tokens=int(env.n_w41_visible_tokens),
        n_w41_overhead_tokens=int(env.n_w41_overhead_tokens),
        n_structured_bits=int(env.n_structured_bits),
    )
    if expected_witness_cid != env.cross_axis_witness_cid:
        return LatentVerificationOutcome(
            ok=False,
            reason="w41_cross_axis_witness_cid_mismatch",
            n_checks=n_checks)
    n_checks += 1
    # Token accounting must be self-consistent and non-negative.
    if (env.n_w40_visible_tokens < 0
            or env.n_w41_overhead_tokens < 0
            or env.n_w41_visible_tokens != (
                int(env.n_w40_visible_tokens)
                + int(env.n_w41_overhead_tokens))):
        return LatentVerificationOutcome(
            ok=False, reason="w41_token_accounting_invalid",
            n_checks=n_checks)
    n_checks += 1
    expected_manifest = _compute_w41_manifest_v11_cid(
        parent_w40_cid=env.parent_w40_cid,
        synthesis_state_cid=env.synthesis_state_cid,
        synthesis_decision_cid=env.synthesis_decision_cid,
        synthesis_audit_cid=env.synthesis_audit_cid,
        cross_axis_witness_cid=env.cross_axis_witness_cid,
    )
    if expected_manifest != env.manifest_v11_cid:
        return LatentVerificationOutcome(
            ok=False, reason="w41_manifest_v11_cid_mismatch",
            n_checks=n_checks)
    n_checks += 1
    if env.recompute_w41_cid() != env.w41_cid:
        return LatentVerificationOutcome(
            ok=False, reason="w41_outer_cid_mismatch",
            n_checks=n_checks)
    n_checks += 1
    return LatentVerificationOutcome(
        ok=True, reason="ok", n_checks=n_checks)


# =============================================================================
# Orchestrator
# =============================================================================

@dataclasses.dataclass
class IntegratedSynthesisOrchestrator:
    """W41 integrated multi-agent context synthesis orchestrator.

    Wraps a :class:`CrossHostResponseHeterogeneityOrchestrator`
    (W40) and adds a cross-axis classification with a manifest-v11
    envelope binding the W40 envelope CID + per-axis branches +
    integrated decision.  The W40 ``decode_rounds`` / ``decode``
    contract is preserved byte-for-byte when ``is_trivial``.
    """

    inner: CrossHostResponseHeterogeneityOrchestrator
    registry: IntegratedSynthesisRegistry
    enabled: bool = True
    require_w41_verification: bool = True

    _last_result: "W41IntegratedSynthesisResult | None" = None
    _last_envelope: (
        "IntegratedSynthesisRatificationEnvelope | None") = None
    _cell_index: int = 0

    @property
    def schema(self) -> "SchemaCapsule | None":
        return self.registry.schema

    def reset_session(self) -> None:
        self.inner.reset_session()
        self._last_result = None
        self._last_envelope = None
        self._cell_index = 0

    def set_response_provider(self, provider: Any) -> None:
        """Pass-through to the inner W40 orchestrator."""
        self.inner.set_response_provider(provider)

    def decode(
            self,
            handoffs: Sequence[_DecodedHandoff],
    ) -> dict[str, Any]:
        return self.decode_rounds([handoffs])

    def decode_rounds(
            self,
            per_round_handoffs: Sequence[Sequence[_DecodedHandoff]],
    ) -> dict[str, Any]:
        # Always run the inner W40 chain.
        out = self.inner.decode_rounds(per_round_handoffs)

        if not self.enabled or self.schema is None:
            self._cell_index += 1
            return self._pack(
                out=out,
                decoder_branch=W41_BRANCH_INTEGRATED_DISABLED,
                envelope=None,
                ratified=False,
                verify_ok=False,
                verify_reason="disabled",
                n_w41_overhead=0,
                integrated_branch=W41_BRANCH_INTEGRATED_DISABLED,
                producer_axis_branch=W41_PRODUCER_AXIS_NO_TRIGGER,
                trust_axis_branch=W41_TRUST_AXIS_NO_TRIGGER,
                producer_services=(),
                trust_services=(),
                integrated_services=(),
                w40_projection_branch="",
                parent_w40_cid="",
                synthesis_state_cid="",
                synthesis_decision_cid="",
                synthesis_audit_cid="",
                cross_axis_witness_cid="",
                manifest_v11_cid="",
                w41_cid="",
                n_envelope_bytes=0,
                n_structured_bits=0,
            )

        # Read the inner W40 result and envelope.
        w40_result = self.inner.last_result
        w40_envelope = self.inner.last_envelope
        n_w40_visible = int(
            w40_result.n_w40_visible_tokens
            if w40_result is not None else 0)
        w40_projection_branch = str(
            w40_result.projection_branch
            if w40_result is not None else "")
        w40_decoder_branch = str(
            w40_result.decoder_branch
            if w40_result is not None else "")

        if self.registry.is_trivial:
            self._cell_index += 1
            return self._pack(
                out=out,
                decoder_branch=(
                    W41_BRANCH_TRIVIAL_INTEGRATED_PASSTHROUGH),
                envelope=None,
                ratified=True,
                verify_ok=True,
                verify_reason="trivial_passthrough",
                n_w41_overhead=0,
                integrated_branch=(
                    W41_BRANCH_TRIVIAL_INTEGRATED_PASSTHROUGH),
                producer_axis_branch=W41_PRODUCER_AXIS_NO_TRIGGER,
                trust_axis_branch=W41_TRUST_AXIS_NO_TRIGGER,
                producer_services=(),
                trust_services=(),
                integrated_services=(),
                w40_projection_branch=w40_projection_branch,
                parent_w40_cid="",
                synthesis_state_cid="",
                synthesis_decision_cid="",
                synthesis_audit_cid="",
                cross_axis_witness_cid="",
                manifest_v11_cid="",
                w41_cid="",
                n_envelope_bytes=0,
                n_structured_bits=0,
            )

        # The producer-axis services are the inner ``out`` services
        # *before* the W40 layer over-rode them (i.e. the multi-
        # round bundle decoder's output).  We read them from the
        # nested inner-result dict if available; otherwise fall
        # back to the current ``out`` services.
        # The W40 over-ride only happens on COLLAPSE_ABSTAINED;
        # in every other branch the W40 services are the W11
        # services unchanged.
        w39_payload = out.get("multi_host_disjoint_quorum") or {}
        # The inner W11 services are reported by the bundle decoder
        # under the key ``services`` of the W11 result; the W22+
        # chain preserves them through ``answer["services"]`` until
        # the W40 layer either ratifies (kept) or abstains
        # (cleared).
        # Producer-side services: always the pre-W40 services, which
        # equals out["services"] when W40 did not abstain, OR
        # w39_payload's answer ``services`` when W40 abstained.
        if (w40_projection_branch
                == W40_BRANCH_RESPONSE_SIGNATURE_COLLAPSE_ABSTAINED):
            # On abstain, W40 cleared services; producer services are
            # whatever W39 carried before.
            inner_w39_services = tuple(sorted(
                str(s) for s in (
                    w40_result.w39_decision_top_set
                    if w40_result is not None else ())))
            producer_services = inner_w39_services
            trust_services = ()
        else:
            current_services = tuple(sorted(
                str(s) for s in out.get("services", ())))
            producer_services = current_services
            # Trust services: the W40 ratified set when DIVERSE.
            if (w40_projection_branch
                    == W40_BRANCH_RESPONSE_SIGNATURE_DIVERSE):
                trust_services = current_services
            else:
                trust_services = ()

        producer_axis_branch = classify_producer_axis_branch(
            services=producer_services)
        trust_axis_branch = classify_trust_axis_branch(
            w40_projection_branch=w40_projection_branch)
        integrated_branch, integrated_services = (
            select_integrated_synthesis_decision(
                producer_axis_branch=producer_axis_branch,
                trust_axis_branch=trust_axis_branch,
                producer_services=producer_services,
                trust_services=trust_services,
            ))

        # Optional: if both-axes required for ratification but
        # producer-only fired, abstain.
        if (self.registry.require_both_axes_for_ratification
                and integrated_branch == (
                    W41_BRANCH_INTEGRATED_PRODUCER_ONLY)):
            integrated_branch = (
                W41_BRANCH_INTEGRATED_AXES_DIVERGED_ABSTAINED)
            integrated_services = ()

        # Optional: if axes diverged and abstain disabled, fall
        # back to producer-only.
        if (integrated_branch == (
                W41_BRANCH_INTEGRATED_AXES_DIVERGED_ABSTAINED)
                and not self.registry.abstain_on_axes_diverged):
            integrated_branch = (
                W41_BRANCH_INTEGRATED_PRODUCER_ONLY)
            integrated_services = producer_services

        parent_w40_cid = (
            str(w40_envelope.w40_cid)
            if w40_envelope is not None else "")
        if not parent_w40_cid:
            parent_payload = _w41_canonical_bytes({
                "inner_w40_decoder_branch": w40_decoder_branch,
                "n_w40_visible": int(n_w40_visible),
                "cell_index": int(self._cell_index),
            })
            parent_w40_cid = hashlib.sha256(
                parent_payload).hexdigest()

        # W41 overhead: 1 visible token per cell when not trivial
        # and not disabled (matches W40 / W39 pattern).
        n_w41_overhead = 1
        n_w41_visible = int(n_w40_visible) + int(n_w41_overhead)

        synthesis_state_cid = _compute_w41_synthesis_state_cid(
            producer_axis_branch=producer_axis_branch,
            trust_axis_branch=trust_axis_branch,
            integrated_branch=integrated_branch,
            integrated_services=integrated_services,
            cell_index=int(self._cell_index),
        )
        synthesis_decision_cid = (
            _compute_w41_synthesis_decision_cid(
                integrated_branch=integrated_branch,
                integrated_services=integrated_services,
                n_w40_visible_tokens=n_w40_visible,
                n_w41_visible_tokens=n_w41_visible,
                n_w41_overhead_tokens=n_w41_overhead,
            ))
        synthesis_audit_cid = _compute_w41_synthesis_audit_cid(
            integrated_branch=integrated_branch,
            producer_axis_branch=producer_axis_branch,
            trust_axis_branch=trust_axis_branch,
            producer_services=tuple(producer_services),
            trust_services=tuple(trust_services),
            integrated_services=tuple(integrated_services),
            w40_projection_branch=w40_projection_branch,
        )

        # Structured-bits estimate for cross-axis witness.  We count
        # the W40 envelope structured bits + the W41 envelope's own
        # state/decision/audit/witness CIDs (4 * 256 bits).
        w40_structured_bits = (
            int(w40_envelope.n_structured_bits)
            if (w40_envelope is not None
                and hasattr(w40_envelope, "n_structured_bits"))
            else 0)
        n_structured_bits = w40_structured_bits + 4 * 256

        cross_axis_witness_cid = (
            _compute_w41_cross_axis_witness_cid(
                producer_axis_branch=producer_axis_branch,
                trust_axis_branch=trust_axis_branch,
                integrated_branch=integrated_branch,
                n_w40_visible_tokens=n_w40_visible,
                n_w41_visible_tokens=n_w41_visible,
                n_w41_overhead_tokens=n_w41_overhead,
                n_structured_bits=n_structured_bits,
            ))
        manifest_v11_cid = (
            "" if self.registry.manifest_v11_disabled
            else _compute_w41_manifest_v11_cid(
                parent_w40_cid=parent_w40_cid,
                synthesis_state_cid=synthesis_state_cid,
                synthesis_decision_cid=synthesis_decision_cid,
                synthesis_audit_cid=synthesis_audit_cid,
                cross_axis_witness_cid=cross_axis_witness_cid,
            ))

        # Outer CID.
        if manifest_v11_cid:
            w41_cid = _compute_w41_outer_cid(
                schema_cid=str(self.schema.cid),
                parent_w40_cid=parent_w40_cid,
                manifest_v11_cid=manifest_v11_cid,
                cell_index=int(self._cell_index),
            )
        else:
            w41_cid = ""

        envelope = IntegratedSynthesisRatificationEnvelope(
            schema_version=W41_INTEGRATED_SYNTHESIS_SCHEMA_VERSION,
            schema_cid=str(self.schema.cid),
            parent_w40_cid=str(parent_w40_cid),
            cell_index=int(self._cell_index),
            producer_axis_branch=str(producer_axis_branch),
            trust_axis_branch=str(trust_axis_branch),
            integrated_branch=str(integrated_branch),
            producer_services=tuple(producer_services),
            trust_services=tuple(trust_services),
            integrated_services=tuple(integrated_services),
            w40_projection_branch=str(w40_projection_branch),
            synthesis_state_cid=synthesis_state_cid,
            synthesis_decision_cid=synthesis_decision_cid,
            synthesis_audit_cid=synthesis_audit_cid,
            cross_axis_witness_cid=cross_axis_witness_cid,
            manifest_v11_cid=manifest_v11_cid,
            n_w40_visible_tokens=int(n_w40_visible),
            n_w41_visible_tokens=int(n_w41_visible),
            n_w41_overhead_tokens=int(n_w41_overhead),
            n_structured_bits=int(n_structured_bits),
            w41_cid=w41_cid,
        )

        # Verify.
        outcome = verify_integrated_synthesis_ratification(
            envelope,
            registered_schema=self.schema,
            registered_parent_w40_cid=parent_w40_cid,
        )
        verify_ok = bool(outcome.ok)
        verify_reason = str(outcome.reason)
        if not verify_ok and self.require_w41_verification:
            self._cell_index += 1
            return self._pack(
                out=out,
                decoder_branch=W41_BRANCH_INTEGRATED_REJECTED,
                envelope=envelope,
                ratified=False,
                verify_ok=False,
                verify_reason=verify_reason,
                n_w41_overhead=0,
                integrated_branch=W41_BRANCH_INTEGRATED_REJECTED,
                producer_axis_branch=producer_axis_branch,
                trust_axis_branch=trust_axis_branch,
                producer_services=producer_services,
                trust_services=trust_services,
                integrated_services=(),
                w40_projection_branch=w40_projection_branch,
                parent_w40_cid=parent_w40_cid,
                synthesis_state_cid=synthesis_state_cid,
                synthesis_decision_cid=synthesis_decision_cid,
                synthesis_audit_cid=synthesis_audit_cid,
                cross_axis_witness_cid=cross_axis_witness_cid,
                manifest_v11_cid=manifest_v11_cid,
                w41_cid=w41_cid,
                n_envelope_bytes=0,
                n_structured_bits=int(n_structured_bits),
            )

        # Apply the integrated services to the outer answer:
        # downstream consumers should observe the integrated
        # decision rather than the W40 ratified set.
        out_local = dict(out)
        out_local["services"] = tuple(integrated_services)
        self._cell_index += 1

        return self._pack(
            out=out_local,
            decoder_branch=integrated_branch,
            envelope=envelope,
            ratified=bool(integrated_services),
            verify_ok=verify_ok,
            verify_reason=verify_reason,
            n_w41_overhead=n_w41_overhead,
            integrated_branch=integrated_branch,
            producer_axis_branch=producer_axis_branch,
            trust_axis_branch=trust_axis_branch,
            producer_services=producer_services,
            trust_services=trust_services,
            integrated_services=integrated_services,
            w40_projection_branch=w40_projection_branch,
            parent_w40_cid=parent_w40_cid,
            synthesis_state_cid=synthesis_state_cid,
            synthesis_decision_cid=synthesis_decision_cid,
            synthesis_audit_cid=synthesis_audit_cid,
            cross_axis_witness_cid=cross_axis_witness_cid,
            manifest_v11_cid=manifest_v11_cid,
            w41_cid=w41_cid,
            n_envelope_bytes=0,
            n_structured_bits=int(n_structured_bits),
        )

    def _pack(
            self,
            *,
            out: dict[str, Any],
            decoder_branch: str,
            envelope: (
                IntegratedSynthesisRatificationEnvelope | None),
            ratified: bool,
            verify_ok: bool,
            verify_reason: str,
            n_w41_overhead: int,
            integrated_branch: str,
            producer_axis_branch: str,
            trust_axis_branch: str,
            producer_services: tuple[str, ...],
            trust_services: tuple[str, ...],
            integrated_services: tuple[str, ...],
            w40_projection_branch: str,
            parent_w40_cid: str,
            synthesis_state_cid: str,
            synthesis_decision_cid: str,
            synthesis_audit_cid: str,
            cross_axis_witness_cid: str,
            manifest_v11_cid: str,
            w41_cid: str,
            n_envelope_bytes: int,
            n_structured_bits: int,
    ) -> dict[str, Any]:
        n_w40_visible = int(
            out.get("multi_host_disjoint_quorum", {}).get(
                "n_w39_visible_tokens", 0))
        # If the W40 result reported n_w40_visible_tokens, prefer it.
        w40_state = out.get("cross_host_response_heterogeneity")
        if (w40_state is not None
                and "n_w40_visible_tokens" in w40_state):
            n_w40_visible = int(w40_state["n_w40_visible_tokens"])
        n_w41_visible = int(n_w40_visible) + int(n_w41_overhead)
        wire = max(1, int(n_w41_overhead))
        cram_factor = (
            float(n_structured_bits) / float(wire)
            if int(n_structured_bits) > 0 else 0.0)

        result = W41IntegratedSynthesisResult(
            answer=dict(out),
            inner_w40_decoder_branch=str(
                w40_state.get("decoder_branch", "")
                if w40_state is not None else ""),
            decoder_branch=str(decoder_branch),
            integrated_branch=str(integrated_branch),
            producer_axis_branch=str(producer_axis_branch),
            trust_axis_branch=str(trust_axis_branch),
            producer_services=tuple(producer_services),
            trust_services=tuple(trust_services),
            integrated_services=tuple(integrated_services),
            w40_projection_branch=str(w40_projection_branch),
            parent_w40_cid=str(parent_w40_cid),
            cell_index=int(
                self._cell_index - 1
                if self._cell_index > 0 else 0),
            n_w40_visible_tokens=int(n_w40_visible),
            n_w41_visible_tokens=int(n_w41_visible),
            n_w41_overhead_tokens=int(n_w41_overhead),
            w41_cid=str(w41_cid),
            manifest_v11_cid=str(manifest_v11_cid),
            synthesis_state_cid=str(synthesis_state_cid),
            synthesis_decision_cid=str(synthesis_decision_cid),
            synthesis_audit_cid=str(synthesis_audit_cid),
            cross_axis_witness_cid=str(cross_axis_witness_cid),
            ratified=bool(ratified),
            verification_ok=bool(verify_ok),
            verification_reason=str(verify_reason),
            n_envelope_bytes=int(n_envelope_bytes),
            n_structured_bits=int(n_structured_bits),
            cram_factor_w41=float(cram_factor),
        )
        self._last_result = result
        self._last_envelope = envelope
        out_local = dict(out)
        out_local["integrated_synthesis"] = result.as_dict()
        if envelope is not None:
            out_local["integrated_synthesis_envelope"] = (
                envelope.as_dict())
        return out_local

    @property
    def last_result(self) -> (
            "W41IntegratedSynthesisResult | None"):
        return self._last_result

    @property
    def last_envelope(self) -> (
            "IntegratedSynthesisRatificationEnvelope | None"):
        return self._last_envelope


# =============================================================================
# Builders
# =============================================================================

def build_trivial_integrated_synthesis_registry(
        *,
        schema: SchemaCapsule,
        inner_w40_registry: (
            CrossHostResponseHeterogeneityRegistry | None) = None,
) -> IntegratedSynthesisRegistry:
    return IntegratedSynthesisRegistry(
        schema=schema,
        inner_w40_registry=inner_w40_registry,
        synthesis_enabled=False,
        manifest_v11_disabled=True,
        abstain_on_axes_diverged=False,
        require_both_axes_for_ratification=False,
    )


def build_integrated_synthesis_registry(
        *,
        schema: SchemaCapsule,
        inner_w40_registry: CrossHostResponseHeterogeneityRegistry,
        synthesis_enabled: bool = True,
        manifest_v11_disabled: bool = False,
        abstain_on_axes_diverged: bool = True,
        require_both_axes_for_ratification: bool = False,
) -> IntegratedSynthesisRegistry:
    return IntegratedSynthesisRegistry(
        schema=schema,
        inner_w40_registry=inner_w40_registry,
        synthesis_enabled=bool(synthesis_enabled),
        manifest_v11_disabled=bool(manifest_v11_disabled),
        abstain_on_axes_diverged=bool(abstain_on_axes_diverged),
        require_both_axes_for_ratification=bool(
            require_both_axes_for_ratification),
    )


__all__ = [
    "W41_INTEGRATED_SYNTHESIS_SCHEMA_VERSION",
    "W41_PRODUCER_AXIS_FIRED",
    "W41_PRODUCER_AXIS_NO_TRIGGER",
    "W41_TRUST_AXIS_RATIFIED",
    "W41_TRUST_AXIS_ABSTAINED",
    "W41_TRUST_AXIS_NO_TRIGGER",
    "W41_BRANCH_TRIVIAL_INTEGRATED_PASSTHROUGH",
    "W41_BRANCH_INTEGRATED_DISABLED",
    "W41_BRANCH_INTEGRATED_REJECTED",
    "W41_BRANCH_INTEGRATED_PRODUCER_ONLY",
    "W41_BRANCH_INTEGRATED_TRUST_ONLY",
    "W41_BRANCH_INTEGRATED_BOTH_AXES",
    "W41_BRANCH_INTEGRATED_AXES_DIVERGED_ABSTAINED",
    "W41_BRANCH_INTEGRATED_NEITHER_AXIS",
    "W41_ALL_BRANCHES",
    "W41_PRODUCER_BRANCHES",
    "W41_TRUST_BRANCHES",
    "classify_producer_axis_branch",
    "classify_trust_axis_branch",
    "select_integrated_synthesis_decision",
    "verify_integrated_synthesis_ratification",
    "IntegratedSynthesisRatificationEnvelope",
    "IntegratedSynthesisRegistry",
    "W41IntegratedSynthesisResult",
    "IntegratedSynthesisOrchestrator",
    "build_trivial_integrated_synthesis_registry",
    "build_integrated_synthesis_registry",
]
