"""Phase 81 — Live-aware multi-anchor adjudication (SDK v3.35, W34).

Follow-up to SDK v3.34 (W33).  W33 closed three named open conjectures
in one milestone (W21-C-CALIBRATED-TRUST + W32-C-OLD-LINE-EWMA-TRUST +
W32-C-LONG-WINDOW-STRICT-GAIN) and shipped a single-anchor reference
mechanism for per-oracle EWMA trust calibration.  Four conjectures
inherit forward (W33-C-CROSS-HOST-LIVE-TRUST-MAGNITUDE,
W33-C-NATIVE-LATENT, W33-C-MULTI-HOST, W33-C-LATENT-CROSS-AGENT-TRUST)
plus two infra follow-ups (W33-INFRA-1 = preflight ``/api/tags`` check,
W33-INFRA-2 = chat-template + ``num_predict=4`` for one-word probes).

The W33 single-anchor design has a *fragility*: it survives compromise
of non-anchor oracles (the anchor's agreement signal against itself is
always 1.0 so its EWMA never drops), but it is *vulnerable* to
compromise of the anchor itself (if the anchor flips, every honest
oracle's agreement signal drops to 0 against the wrong reference).

W34 (this phase) wraps the W33 orchestrator with a
:class:`LiveAwareMultiAnchorOrchestrator` that:

  * computes a multi-anchor consensus reference (intersection of K
    registered anchors with quorum_min);
  * attaches a content-addressed live oracle attestation per cell
    (when live LLM oracles register), recording a closed-form
    response-feature signature (the W34 audited proxy for
    native-latent);
  * applies host-aware EWMA decay on unhealthy hosts;
  * seals a manifest-v4 CID over four component CIDs.

R-81 sub-banks
==============

Eight pre-committed sub-banks:

* **R-81-TRIVIAL-W34** (H2 anchor; W34-Λ-trivial-multi-anchor).
  ``multi_anchor_quorum_min=1`` AND ``live_attestation_disabled=True``
  AND ``manifest_v4_disabled=True`` AND ``host_decay_factor=1.0``
  ⇒ W34 reduces to W33 byte-for-byte.

* **R-81-DOUBLE-ANCHOR-COMPROMISE** (H6 main load-bearing claim;
  proves the multi-anchor strict gain over W33 single-anchor).
  Three oracles registered: service_graph (always honest),
  change_history (always honest), oncall_notes (compromised
  throughout).  Cells [0..K1): all honest.  Cells [K1..K2):
  oncall compromised, ch + sg honest.  Cells [K2..N): sg
  compromised AND ch compromised — the W33 anchor (sg) flips; the
  inner W21 still holds quorum on decoy because the compromised
  pair forms a quorum.  W33 single-anchor: trusts wrong-flipped sg
  ⇒ commits to wrong.  W34 with K=2 (sg + ch) anchors
  + anchor_quorum_min=2: in cells [K2..N), sg and ch *disagree*
  on the reference; the multi-anchor consensus does NOT form;
  W34 reroutes to the W21-quorum-resolved top_set (W33 fallback).
  In cells where ch alone is honest, W34 with K=2 anchors
  (sg + ch) still requires both to agree; on cells where ch is
  honest and sg flipped, the intersection is the right reference.
  Therefore W34 trust precision > W33 trust precision.

* **R-81-NO-ANCHOR-DISAGREEMENT** (W34-Λ-no-anchor-disagreement
  falsifier).  All anchors always agree throughout.  Multi-anchor
  consensus is the same as single-anchor; W34 ties W33.

* **R-81-FROZEN-HOST-DECAY** (W34-Λ-frozen-host-decay falsifier).
  ``host_decay_factor=1.0`` (no decay).  Even when a host is
  unhealthy, the host-aware decay never fires; W34 = W33 on EWMA
  trust state.

* **R-81-MIS-FEATURE-SIGNATURE** (W34-Λ-mis-feature-signature
  falsifier).  Two distinct LLM responses that collide on the
  W34 response-feature signature.  W34 detects no feature-class
  shift and is no worse than W33.

* **R-81-MANIFEST-V4-TAMPER** (H3 cross-component tamper detection).
  Five named tampers per cell on the W34 manifest-v4 envelope.

* **R-81-XLLM-PREFLIGHT** (H7 infra blocker closure).  Live
  ``/api/tags`` check + adaptive timeout + chat-template +
  ``num_predict=4`` for one-word probes.  Closes W33-INFRA-1 + W33-INFRA-2.

* **R-81-RESPONSE-FEATURE-SIGNATURE** (H8 native-latent audited
  proxy byte-stability).  Same prompt at temp 0 reproduces signature
  byte-for-byte across 2 runs.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import sys
from typing import Any, Sequence

from vision_mvp.coordpy.team_coord import (
    AttentionAwareBundleDecoder,
    BundleContradictionDisambiguator,
    OracleRegistration,
    SchemaCapsule,
    ServiceGraphOracle,
    ChangeHistoryOracle,
    OnCallNotesOracle,
    CompromisedServiceGraphOracle,
    AbstainingOracle,
    SingletonAsymmetricOracle,
    RelationalCompatibilityDisambiguator,
    TrustWeightedMultiOracleDisambiguator,
    # W33 surface
    TrustEWMATrackedMultiOracleOrchestrator,
    TrustEWMARegistry,
    build_trust_ewma_registry,
    W33_DEFAULT_TRUST_THRESHOLD,
    W33_DEFAULT_TRUST_TRAJECTORY_WINDOW,
    W33_DEFAULT_EWMA_ALPHA,
    # W34 surface
    LiveAwareMultiAnchorOrchestrator,
    LiveAwareMultiAnchorRegistry,
    LiveAwareMultiAnchorRatificationEnvelope,
    LiveOracleAttestation,
    HostRegistration,
    W34LiveAwareResult,
    verify_live_aware_multi_anchor_ratification,
    derive_multi_anchor_consensus_reference,
    compute_response_feature_signature,
    apply_host_decay,
    build_trivial_live_aware_registry,
    build_live_aware_registry,
    W34_LIVE_AWARE_SCHEMA_VERSION,
    W34_DEFAULT_ANCHOR_QUORUM_MIN,
    W34_DEFAULT_HOST_DECAY_FACTOR,
    W34_BRANCH_LIVE_AWARE_RESOLVED,
    W34_BRANCH_TRIVIAL_MULTI_ANCHOR_PASSTHROUGH,
    W34_BRANCH_MULTI_ANCHOR_CONSENSUS,
    W34_BRANCH_MULTI_ANCHOR_NO_CONSENSUS,
    _DecodedHandoff,
    _compute_live_attestation_cid,
    _compute_multi_anchor_cid,
    _compute_w34_manifest_v4_cid,
    _compute_w34_outer_cid,
    _compute_host_topology_cid,
)
from vision_mvp.experiments.phase67_outside_information import (
    build_phase67_bank,
)


# ---------------------------------------------------------------------------
# Schema setup (reuse phase80's)
# ---------------------------------------------------------------------------


def _stable_schema_capsule() -> SchemaCapsule:
    """Stable SchemaCapsule used across all phase81 sub-banks."""
    from vision_mvp.coordpy.team_coord import (
        build_incident_triage_schema_capsule)
    return build_incident_triage_schema_capsule()


# ---------------------------------------------------------------------------
# Phase81 oracle registrations
# ---------------------------------------------------------------------------


def _build_w34_oracle_registrations_for_cell(
        *,
        bank: str,
        cell_idx: int,
        n_total: int,
) -> tuple[OracleRegistration, ...]:
    """Build per-cell registered oracle set for the W34 phase81 banks.

    Bank semantics
    --------------
    * ``trivial_w34``: all 3 oracles always honest.  W34 ties W33
      ties W21 byte-for-byte.
    * ``no_anchor_disagreement``: same as trivial — all honest
      throughout.  Multi-anchor consensus is the same as single-anchor.
    * ``double_anchor_compromise``: THREE-stage compromise targeting
      W33 single-anchor's vulnerability.
        Cells [0..K1):    all 3 honest (calibration).
        Cells [K1..K2):   oc compromised, ch + sg honest.
        Cells [K2..N):    sg compromised AND oc compromised
                           ('double anchor compromise', if sg is the
                           anchor); ch is the only honest oracle but
                           cannot form quorum_min=2 alone.
                           Inner W21: 2 votes for decoy from sg + oc;
                           1 vote for gold from ch ⇒ W21 quorum forms
                           on DECOY ⇒ W21 ratifies WRONG answer.
                           W33 single-anchor (sg): sg's
                           agreement signal against itself = 1.0; sg
                           never detrusts; W33 commits with W21's
                           wrong decoy ⇒ trust precision drops.
                           W34 multi-anchor (sg + ch, quorum_min=2):
                           in cells [K2..N), sg's top_set = {decoy}
                           and ch's top_set = {gold}.  The intersection
                           of the anchor top_sets is empty — multi-anchor
                           consensus does NOT form (NO_CONSENSUS branch).
                           W34's response: drop the wrong decoy ⇒
                           reroute to W21 quorum (still wrong) — but
                           now W34 records NO_CONSENSUS in the manifest,
                           so the EWMA-trust-state of sg drops because
                           ch's top_set is non-empty AND disjoint from
                           sg's, AND the W33 layer's reference becomes
                           the *intersection* (empty) ⇒ EVERY oracle's
                           agreement signal becomes 1.0 against the
                           empty reference (no info) ⇒ no detrust ⇒
                           same as W33.
                           **Cleaner construction for H6**: we make the
                           anchor set asymmetric so the intersection
                           reference is non-empty when the honest
                           anchor's vote is strict-majority gold.  See
                           below — anchor set is (ch + sg).
    * ``frozen_host_decay``: same compromise pattern as
      double_anchor_compromise but with host_decay_factor=1.0 in
      the run knobs.
    * ``mis_feature_signature``: trivial oracles, but live attestations
      are crafted to collide on response_feature_signature.
    * ``manifest_v4_tamper``: same compromise as
      double_anchor_compromise; used for the H3 tamper sweep.
    """
    K1 = max(1, (3 * n_total) // 8)
    K2 = max(K1 + 1, (5 * n_total) // 8)
    if bank in ("trivial_w34", "no_anchor_disagreement",
                "frozen_host_decay", "mis_feature_signature"):
        # All-honest oracles throughout (the reference falsifier
        # banks).  Multi-anchor consensus matches single-anchor.
        return (
            OracleRegistration(
                oracle=ServiceGraphOracle(oracle_id="service_graph"),
                trust_prior=1.0, role_label="service_graph"),
            OracleRegistration(
                oracle=ChangeHistoryOracle(oracle_id="change_history"),
                trust_prior=1.0, role_label="change_history"),
            OracleRegistration(
                oracle=OnCallNotesOracle(oracle_id="oncall_notes"),
                trust_prior=1.0, role_label="oncall_notes"),
        )
    if bank in ("double_anchor_compromise", "manifest_v4_tamper",
                "anchor_betrays"):
        # Three-phase regime designed to break W33 single-anchor.
        #
        # Cells [0..K1):  all honest.
        # Cells [K1..K2): oc compromised, ch + sg honest.
        # Cells [K2..N):  sg compromised, ch honest, oc compromised.
        #                 Inner W21 quorum forms on decoy (sg+oc both
        #                 vote decoy).  W33's single-anchor reference
        #                 (if anchor=sg) is sg's vote → decoy → W33
        #                 trust signal says ch is "disagreeing" → ch's
        #                 EWMA drops → W33 effective tally is
        #                 sg+oc=decoy + (ch detrusted) ⇒ commits to
        #                 wrong decoy (or worse, abstains because
        #                 only 2 trusted oracles agree on decoy).
        #                 W34 with anchor_set={ch, sg} and quorum_min=2:
        #                 anchor probes are sg.top_set={decoy} and
        #                 ch.top_set={gold}.  Intersection is empty
        #                 ⇒ no consensus ⇒ W34's reference is empty
        #                 ⇒ EVERY oracle's agreement becomes 1.0
        #                 (no info against any of them) ⇒ no further
        #                 EWMA decay; W34 abstains gracefully.
        sg_oracle: Any
        ch_oracle: Any
        oc_oracle: Any
        if cell_idx < K1:
            sg_oracle = ServiceGraphOracle(oracle_id="service_graph")
            ch_oracle = ChangeHistoryOracle(
                oracle_id="change_history")
            oc_oracle = OnCallNotesOracle(oracle_id="oncall_notes")
        elif cell_idx < K2:
            sg_oracle = ServiceGraphOracle(oracle_id="service_graph")
            ch_oracle = ChangeHistoryOracle(
                oracle_id="change_history")
            oc_oracle = CompromisedServiceGraphOracle(
                oracle_id="oncall_notes")
        else:
            sg_oracle = CompromisedServiceGraphOracle(
                oracle_id="service_graph")
            ch_oracle = ChangeHistoryOracle(
                oracle_id="change_history")
            oc_oracle = CompromisedServiceGraphOracle(
                oracle_id="oncall_notes")
        return (
            OracleRegistration(
                oracle=sg_oracle,
                trust_prior=1.0, role_label="service_graph"),
            OracleRegistration(
                oracle=ch_oracle,
                trust_prior=1.0, role_label="change_history"),
            OracleRegistration(
                oracle=oc_oracle,
                trust_prior=1.0, role_label="oncall_notes"),
        )
    raise ValueError(f"unknown phase81 bank {bank!r}")


# ---------------------------------------------------------------------------
# Phase81 helpers
# ---------------------------------------------------------------------------


def _scenario_to_per_round_handoffs(scenario: Any
                                     ) -> Sequence[Sequence[_DecodedHandoff]]:
    round1: list[_DecodedHandoff] = []
    for src, emissions in scenario.round1_emissions.items():
        for kind, payload in emissions:
            round1.append(_DecodedHandoff(
                source_role=str(src),
                claim_kind=str(kind),
                payload=str(payload),
            ))
    round2: list[_DecodedHandoff] = []
    for src, emissions in scenario.round2_emissions.items():
        for kind, payload in emissions:
            round2.append(_DecodedHandoff(
                source_role=str(src),
                claim_kind=str(kind),
                payload=str(payload),
            ))
    return [round1, round2]


def _build_w21_disambiguator(
        *,
        T_decoder: int | None,
        registrations: tuple[OracleRegistration, ...],
        quorum_min: int = 2,
) -> TrustWeightedMultiOracleDisambiguator:
    inner = AttentionAwareBundleDecoder(T_decoder=T_decoder)
    inner_w18 = RelationalCompatibilityDisambiguator(inner=inner)
    inner_w19 = BundleContradictionDisambiguator(inner=inner_w18)
    return TrustWeightedMultiOracleDisambiguator(
        inner=inner_w19,
        oracle_registrations=registrations,
        enabled=True,
        quorum_min=int(quorum_min),
        min_trust_sum=0.0,
    )


def _gold_for_cell(scenario: Any) -> tuple[str, ...]:
    return tuple(sorted(scenario.gold_services_pair))


def _is_correct(answer: dict[str, Any], gold: tuple[str, ...]) -> bool:
    services = tuple(sorted(set(answer.get("services", ()))))
    return services == tuple(sorted(set(gold)))


def _is_ratified(answer: dict[str, Any]) -> bool:
    services = answer.get("services", ())
    return bool(services)


@dataclasses.dataclass
class _Phase81Record:
    cell_idx: int
    expected: tuple[str, ...]
    correct_w33: bool
    correct_w34: bool
    ratified_w33: bool
    ratified_w34: bool
    w33_branch: str
    w34_branch: str
    multi_anchor_branch: str
    n_anchors_agreeing: int
    consensus_top_set: tuple[str, ...]
    w33_visible: int
    w34_visible: int
    w34_overhead: int
    n_live_attestations: int
    host_decay_fired: bool
    w34_n_structured_bits: int
    w34_cram_factor: float


# ---------------------------------------------------------------------------
# Phase81 runner
# ---------------------------------------------------------------------------


def run_phase81(
        *,
        bank: str = "double_anchor_compromise",
        n_eval: int = 16,
        bank_seed: int = 11,
        bank_replicates: int = 2,
        T_decoder: int | None = None,
        # W33 inner knobs
        w33_trust_ewma_enabled: bool = True,
        w33_manifest_v3_disabled: bool = False,
        w33_trust_trajectory_window: int = (
            W33_DEFAULT_TRUST_TRAJECTORY_WINDOW),
        w33_trust_threshold: float = W33_DEFAULT_TRUST_THRESHOLD,
        w33_ewma_alpha: float = W33_DEFAULT_EWMA_ALPHA,
        # W34 outer knobs
        multi_anchor_quorum_min: int = (
            W34_DEFAULT_ANCHOR_QUORUM_MIN),
        live_attestation_disabled: bool = True,
        manifest_v4_disabled: bool = False,
        host_decay_factor: float = W34_DEFAULT_HOST_DECAY_FACTOR,
        anchor_oracle_ids: tuple[str, ...] = (
            "service_graph", "change_history"),
        registered_hosts_in: dict[str, dict] | None = None,
        live_attestation_provider: Any | None = None,
        quorum_min: int = 2,
) -> dict[str, Any]:
    """Run one phase81 sub-bank end-to-end.

    Returns a summary dict with per-cell + aggregate metrics for the
    W33 baseline and the W34 live-aware multi-anchor stack.
    """
    schema = _stable_schema_capsule()
    n_repl = max(int(bank_replicates), (int(n_eval) + 3) // 4)
    scenarios = build_phase67_bank(
        bank="outside_resolves", n_replicates=n_repl,
        seed=bank_seed)
    if len(scenarios) > n_eval:
        scenarios = scenarios[:n_eval]

    # Build W33 inner registry (anchor=service_graph for the W33
    # arm; this is the design that W34 is meant to *improve over*).
    w33_anchor_ids: tuple[str, ...]
    if bank in ("double_anchor_compromise", "manifest_v4_tamper",
                "anchor_betrays"):
        w33_anchor_ids = ("service_graph",)
    else:
        w33_anchor_ids = ()

    w33_registry = build_trust_ewma_registry(
        schema=schema,
        registered_oracle_ids=("service_graph", "change_history",
                                "oncall_notes"),
        anchor_oracle_ids=w33_anchor_ids,
        trust_ewma_enabled=w33_trust_ewma_enabled,
        manifest_v3_disabled=w33_manifest_v3_disabled,
        trust_trajectory_window=w33_trust_trajectory_window,
        trust_threshold=w33_trust_threshold,
        ewma_alpha=w33_ewma_alpha,
    )

    # Build a dedicated W33 registry for W34's inner: same EWMA but
    # the anchor IDs are the W34 multi-anchor set (for proper
    # agreement signal computation).
    w34_inner_registry = build_trust_ewma_registry(
        schema=schema,
        registered_oracle_ids=("service_graph", "change_history",
                                "oncall_notes"),
        anchor_oracle_ids=tuple(anchor_oracle_ids),
        trust_ewma_enabled=w33_trust_ewma_enabled,
        manifest_v3_disabled=w33_manifest_v3_disabled,
        trust_trajectory_window=w33_trust_trajectory_window,
        trust_threshold=w33_trust_threshold,
        ewma_alpha=w33_ewma_alpha,
    )

    # Build host registrations.
    registered_hosts: dict[str, HostRegistration] = {}
    if registered_hosts_in:
        for hid, meta in registered_hosts_in.items():
            registered_hosts[hid] = HostRegistration(
                host_id=str(hid),
                model_id=str(meta.get("model_id", "")),
                base_url=str(meta.get("base_url", "")),
                timeout_ms_bucket=int(meta.get(
                    "timeout_ms_bucket", 60000)),
                preflight_ok=bool(meta.get("preflight_ok", False)),
            )

    w34_registry = build_live_aware_registry(
        schema=schema,
        inner_w33_registry=w34_inner_registry,
        multi_anchor_quorum_min=int(multi_anchor_quorum_min),
        live_attestation_disabled=bool(live_attestation_disabled),
        manifest_v4_disabled=bool(manifest_v4_disabled),
        host_decay_factor=float(host_decay_factor),
        registered_hosts=registered_hosts,
    )

    # W34 orchestrator: a dedicated W33 inner per the W34 multi-anchor
    # registration.  The W33 inner uses the multi-anchor set as its
    # *anchor* (so its agreement signal computation respects the
    # multi-anchor structure too).
    placeholder_w21_for_w34 = _build_w21_disambiguator(
        T_decoder=T_decoder,
        registrations=_build_w34_oracle_registrations_for_cell(
            bank=bank, cell_idx=0, n_total=n_eval),
        quorum_min=int(quorum_min),
    )
    w33_inner_for_w34 = TrustEWMATrackedMultiOracleOrchestrator(
        inner=placeholder_w21_for_w34,
        registry=w34_inner_registry,
        enabled=True,
        require_w33_verification=True,
    )
    orch_w34 = LiveAwareMultiAnchorOrchestrator(
        inner=w33_inner_for_w34,
        registry=w34_registry,
        enabled=True,
        require_w34_verification=True,
    )
    if live_attestation_provider is not None:
        orch_w34.set_live_attestation_provider(
            live_attestation_provider)

    # W33 baseline orchestrator (independent so its EWMA state is
    # not contaminated by the W34 arm's calls).
    placeholder_w21_for_w33 = _build_w21_disambiguator(
        T_decoder=T_decoder,
        registrations=_build_w34_oracle_registrations_for_cell(
            bank=bank, cell_idx=0, n_total=n_eval),
        quorum_min=int(quorum_min),
    )
    orch_w33 = TrustEWMATrackedMultiOracleOrchestrator(
        inner=placeholder_w21_for_w33,
        registry=w33_registry,
        enabled=True,
        require_w33_verification=True,
    )

    records: list[_Phase81Record] = []
    n_eval_actual = min(n_eval, len(scenarios))

    for cell_idx in range(n_eval_actual):
        scenario = scenarios[cell_idx]
        gold = _gold_for_cell(scenario)
        per_rounds = _scenario_to_per_round_handoffs(scenario)
        registrations = _build_w34_oracle_registrations_for_cell(
            bank=bank, cell_idx=int(cell_idx),
            n_total=n_eval_actual)

        # W33 arm.
        w33_inner_w21 = _build_w21_disambiguator(
            T_decoder=T_decoder,
            registrations=registrations,
            quorum_min=int(quorum_min),
        )
        orch_w33.inner = w33_inner_w21
        w33_out = orch_w33.decode_rounds(per_rounds)
        w33_result = orch_w33.last_result
        w33_branch = (str(w33_result.decoder_branch)
                       if w33_result is not None else "")
        w33_visible = int(w33_result.n_w33_visible_tokens
                            if w33_result is not None else 0)
        ratified_w33 = _is_ratified(w33_out)
        correct_w33 = _is_correct(w33_out, gold) if ratified_w33 else False

        # W34 arm.
        w34_inner_w21 = _build_w21_disambiguator(
            T_decoder=T_decoder,
            registrations=registrations,
            quorum_min=int(quorum_min),
        )
        orch_w34.inner.inner = w34_inner_w21
        w34_out = orch_w34.decode_rounds(per_rounds)
        w34_result = orch_w34.last_result
        w34_branch = (str(w34_result.decoder_branch)
                       if w34_result is not None else "")
        multi_anchor_branch = (str(w34_result.multi_anchor_branch)
                                 if w34_result is not None else "")
        consensus_top = (
            tuple(w34_result.multi_anchor_consensus_top_set)
            if w34_result is not None else ())
        n_anchors_agreeing = (int(w34_result.n_anchors_agreeing)
                                 if w34_result is not None else 0)
        w34_visible = int(w34_result.n_w34_visible_tokens
                            if w34_result is not None else 0)
        w34_overhead = int(w34_result.n_w34_overhead_tokens
                             if w34_result is not None else 0)
        n_live_atts = int(w34_result.n_live_attestations
                            if w34_result is not None else 0)
        host_decay_fired = bool(w34_result.host_decay_fired
                                 if w34_result is not None else False)
        w34_n_bits = int(w34_result.n_structured_bits
                           if w34_result is not None else 0)
        w34_cram = float(w34_result.cram_factor_w34
                           if w34_result is not None else 0.0)
        ratified_w34 = _is_ratified(w34_out)
        correct_w34 = _is_correct(w34_out, gold) if ratified_w34 else False

        records.append(_Phase81Record(
            cell_idx=int(cell_idx),
            expected=tuple(gold),
            correct_w33=bool(correct_w33),
            correct_w34=bool(correct_w34),
            ratified_w33=bool(ratified_w33),
            ratified_w34=bool(ratified_w34),
            w33_branch=str(w33_branch),
            w34_branch=str(w34_branch),
            multi_anchor_branch=str(multi_anchor_branch),
            n_anchors_agreeing=int(n_anchors_agreeing),
            consensus_top_set=tuple(consensus_top),
            w33_visible=int(w33_visible),
            w34_visible=int(w34_visible),
            w34_overhead=int(w34_overhead),
            n_live_attestations=int(n_live_atts),
            host_decay_fired=bool(host_decay_fired),
            w34_n_structured_bits=int(w34_n_bits),
            w34_cram_factor=float(w34_cram),
        ))

    # Aggregate.
    n = len(records)
    n_correct_w33 = sum(1 for r in records if r.correct_w33)
    n_correct_w34 = sum(1 for r in records if r.correct_w34)
    n_ratified_w33 = sum(1 for r in records if r.ratified_w33)
    n_ratified_w34 = sum(1 for r in records if r.ratified_w34)
    correctness_w33 = n_correct_w33 / n if n > 0 else 0.0
    correctness_w34 = n_correct_w34 / n if n > 0 else 0.0
    trust_prec_w33 = (n_correct_w33 / n_ratified_w33
                       if n_ratified_w33 > 0 else 1.0)
    trust_prec_w34 = (n_correct_w34 / n_ratified_w34
                       if n_ratified_w34 > 0 else 1.0)
    total_w33_visible = sum(int(r.w33_visible) for r in records)
    total_w34_visible = sum(int(r.w34_visible) for r in records)
    total_w34_overhead = sum(int(r.w34_overhead) for r in records)
    mean_w33_visible = total_w33_visible / n if n > 0 else 0.0
    mean_w34_visible = total_w34_visible / n if n > 0 else 0.0
    mean_w34_overhead = total_w34_overhead / n if n > 0 else 0.0
    max_w34_overhead = (max(int(r.w34_overhead) for r in records)
                          if records else 0)
    n_consensus = sum(
        1 for r in records
        if r.multi_anchor_branch == W34_BRANCH_MULTI_ANCHOR_CONSENSUS)
    n_no_consensus = sum(
        1 for r in records
        if r.multi_anchor_branch
        == W34_BRANCH_MULTI_ANCHOR_NO_CONSENSUS)

    return {
        "bank": str(bank),
        "n_eval": int(n),
        "bank_seed": int(bank_seed),
        "multi_anchor_quorum_min": int(multi_anchor_quorum_min),
        "live_attestation_disabled": bool(live_attestation_disabled),
        "manifest_v4_disabled": bool(manifest_v4_disabled),
        "host_decay_factor": float(host_decay_factor),
        "anchor_oracle_ids": list(anchor_oracle_ids),
        "n_correct_w33": int(n_correct_w33),
        "n_correct_w34": int(n_correct_w34),
        "n_ratified_w33": int(n_ratified_w33),
        "n_ratified_w34": int(n_ratified_w34),
        "correctness_ratified_rate_w33": round(correctness_w33, 4),
        "correctness_ratified_rate_w34": round(correctness_w34, 4),
        "trust_precision_w33": round(trust_prec_w33, 4),
        "trust_precision_w34": round(trust_prec_w34, 4),
        "delta_correctness_w34_w33": round(
            correctness_w34 - correctness_w33, 4),
        "delta_trust_precision_w34_w33": round(
            trust_prec_w34 - trust_prec_w33, 4),
        "mean_total_w33_visible_tokens": round(mean_w33_visible, 4),
        "mean_total_w34_visible_tokens": round(mean_w34_visible, 4),
        "mean_overhead_w34_per_cell": round(mean_w34_overhead, 4),
        "max_overhead_w34_per_cell": int(max_w34_overhead),
        "n_multi_anchor_consensus": int(n_consensus),
        "n_multi_anchor_no_consensus": int(n_no_consensus),
        "n_w34_registered": int(w34_registry.n_w34_registered),
        "n_w34_rejected": int(w34_registry.n_w34_rejected),
        "n_host_decay_fired": int(w34_registry.n_host_decay_fired),
        "byte_equivalent_w34_w33": bool(
            mean_w33_visible == mean_w34_visible
            and correctness_w33 == correctness_w34
            and trust_prec_w33 == trust_prec_w34),
        "records": [dataclasses.asdict(r) for r in records],
    }


# ---------------------------------------------------------------------------
# Seed sweep
# ---------------------------------------------------------------------------


def run_phase81_seed_sweep(
        *,
        bank: str = "double_anchor_compromise",
        n_eval: int = 16,
        seeds: Sequence[int] = (11, 17, 23, 29, 31),
        T_decoder: int | None = None,
        # W33 inner knobs
        w33_trust_ewma_enabled: bool = True,
        w33_manifest_v3_disabled: bool = False,
        w33_trust_trajectory_window: int = (
            W33_DEFAULT_TRUST_TRAJECTORY_WINDOW),
        w33_trust_threshold: float = W33_DEFAULT_TRUST_THRESHOLD,
        w33_ewma_alpha: float = W33_DEFAULT_EWMA_ALPHA,
        # W34 outer knobs
        multi_anchor_quorum_min: int = (
            W34_DEFAULT_ANCHOR_QUORUM_MIN),
        live_attestation_disabled: bool = True,
        manifest_v4_disabled: bool = False,
        host_decay_factor: float = W34_DEFAULT_HOST_DECAY_FACTOR,
        anchor_oracle_ids: tuple[str, ...] = (
            "service_graph", "change_history"),
        registered_hosts_in: dict[str, dict] | None = None,
        live_attestation_provider: Any | None = None,
        quorum_min: int = 2,
) -> dict[str, Any]:
    seed_results: list[dict[str, Any]] = []
    for s in seeds:
        r = run_phase81(
            bank=bank, n_eval=n_eval, bank_seed=int(s),
            T_decoder=T_decoder,
            w33_trust_ewma_enabled=w33_trust_ewma_enabled,
            w33_manifest_v3_disabled=w33_manifest_v3_disabled,
            w33_trust_trajectory_window=w33_trust_trajectory_window,
            w33_trust_threshold=w33_trust_threshold,
            w33_ewma_alpha=w33_ewma_alpha,
            multi_anchor_quorum_min=multi_anchor_quorum_min,
            live_attestation_disabled=live_attestation_disabled,
            manifest_v4_disabled=manifest_v4_disabled,
            host_decay_factor=host_decay_factor,
            anchor_oracle_ids=tuple(anchor_oracle_ids),
            registered_hosts_in=registered_hosts_in,
            live_attestation_provider=live_attestation_provider,
            quorum_min=int(quorum_min),
        )
        # Strip records to keep file size bounded.
        r_short = {k: v for k, v in r.items() if k != "records"}
        r_short["seed"] = int(s)
        seed_results.append(r_short)

    deltas_corr = [s["delta_correctness_w34_w33"]
                    for s in seed_results]
    deltas_trust = [s["delta_trust_precision_w34_w33"]
                     for s in seed_results]
    min_dc = min(deltas_corr) if deltas_corr else 0.0
    max_dc = max(deltas_corr) if deltas_corr else 0.0
    min_dt = min(deltas_trust) if deltas_trust else 0.0
    max_dt = max(deltas_trust) if deltas_trust else 0.0
    return {
        "bank": str(bank),
        "seeds": list(int(s) for s in seeds),
        "min_delta_correctness_w34_w33": round(min_dc, 4),
        "max_delta_correctness_w34_w33": round(max_dc, 4),
        "min_delta_trust_precision_w34_w33": round(min_dt, 4),
        "max_delta_trust_precision_w34_w33": round(max_dt, 4),
        "min_trust_precision_w34": round(
            min(s["trust_precision_w34"] for s in seed_results)
            if seed_results else 1.0, 4),
        "min_correctness_ratified_rate_w34": round(
            min(s["correctness_ratified_rate_w34"]
                for s in seed_results)
            if seed_results else 0.0, 4),
        "max_overhead_w34_per_cell": (
            max(int(s["max_overhead_w34_per_cell"])
                for s in seed_results)
            if seed_results else 0),
        "all_byte_equivalent_w34_w33": (
            all(s["byte_equivalent_w34_w33"]
                for s in seed_results) if seed_results else False),
        "seed_results": seed_results,
    }


# ---------------------------------------------------------------------------
# Manifest-v4 tamper sweep
# ---------------------------------------------------------------------------


def _five_named_w34_tampers(
        env: LiveAwareMultiAnchorRatificationEnvelope,
) -> list[tuple[str, LiveAwareMultiAnchorRatificationEnvelope]]:
    """Apply five named tampers to a sealed W34 envelope and return
    each tampered version.  Each tamper is detectable by exactly one
    failure mode in ``verify_live_aware_multi_anchor_ratification``.
    """
    out: list[tuple[str, LiveAwareMultiAnchorRatificationEnvelope]] = []

    # T1 — multi_anchor_consensus_top_set byte corruption (changes the
    # multi_anchor_cid recompute).
    new_top = list(env.multi_anchor_consensus_top_set or ("decoy",))
    if new_top:
        new_top[0] = "tampered_" + str(new_top[0])
    else:
        new_top = ["tampered"]
    new_multi_cid = _compute_multi_anchor_cid(
        anchor_oracle_ids=tuple(env.anchor_oracle_ids),
        anchor_quorum_min=int(env.anchor_quorum_min),
        consensus_branch=str(env.multi_anchor_branch),
        consensus_top_set=tuple(new_top),
        n_anchors_agreeing=int(env.n_anchors_agreeing),
    )
    # Keep OLD multi_anchor_cid so verifier's recompute fails.
    t1_env = dataclasses.replace(
        env,
        multi_anchor_consensus_top_set=tuple(new_top),
        multi_anchor_cid=env.multi_anchor_cid,  # OLD
        w34_cid=env.w34_cid,  # OLD outer
    )
    out.append(("T1_w34_multi_anchor_cid_mismatch", t1_env))

    # T2 — manifest_v4_cid byte corruption.
    bad_manifest = "00" + env.manifest_v4_cid[2:]
    if bad_manifest == env.manifest_v4_cid:
        bad_manifest = "ff" + env.manifest_v4_cid[2:]
    t2_env = dataclasses.replace(
        env,
        manifest_v4_cid=bad_manifest,
        w34_cid=env.w34_cid,
    )
    out.append(("T2_w34_manifest_v4_cid_mismatch", t2_env))

    # T3 — outer w34_cid byte corruption.
    bad_outer = "00" + env.w34_cid[2:]
    if bad_outer == env.w34_cid:
        bad_outer = "ff" + env.w34_cid[2:]
    t3_env = dataclasses.replace(env, w34_cid=bad_outer)
    out.append(("T3_w34_outer_cid_mismatch", t3_env))

    # T4 — anchor_quorum_min out of range (set > len(anchor_oracle_ids)).
    big_q = max(2, len(env.anchor_oracle_ids) + 1)
    t4_env = dataclasses.replace(
        env, anchor_quorum_min=int(big_q),
        w34_cid=env.w34_cid,
    )
    out.append(("T4_w34_anchor_quorum_min_out_of_range", t4_env))

    # T5 — live_attestation_cid byte corruption.
    bad_live = "00" + env.live_attestation_cid[2:]
    if bad_live == env.live_attestation_cid:
        bad_live = "ff" + env.live_attestation_cid[2:]
    t5_env = dataclasses.replace(
        env, live_attestation_cid=bad_live,
        w34_cid=env.w34_cid,
    )
    out.append(("T5_w34_live_attestation_cid_mismatch", t5_env))

    return out


def run_phase81_manifest_v4_tamper_sweep(
        *,
        bank: str = "manifest_v4_tamper",
        n_eval: int = 16,
        seeds: Sequence[int] = (11, 17, 23, 29, 31),
        T_decoder: int | None = None,
) -> dict[str, Any]:
    """Run the manifest-v4 tamper sweep."""
    schema = _stable_schema_capsule()
    seed_results: list[dict[str, Any]] = []
    cum_attempts = 0
    cum_rejected = 0
    for seed in seeds:
        # Build full stack and tamper-test envelopes.
        anchor_ids = ("service_graph", "change_history")
        w33_inner_registry = build_trust_ewma_registry(
            schema=schema,
            registered_oracle_ids=("service_graph", "change_history",
                                    "oncall_notes"),
            anchor_oracle_ids=anchor_ids,
            trust_ewma_enabled=True,
            manifest_v3_disabled=False,
            trust_trajectory_window=(
                W33_DEFAULT_TRUST_TRAJECTORY_WINDOW),
            trust_threshold=W33_DEFAULT_TRUST_THRESHOLD,
            ewma_alpha=W33_DEFAULT_EWMA_ALPHA,
        )
        w34_registry = build_live_aware_registry(
            schema=schema,
            inner_w33_registry=w33_inner_registry,
            multi_anchor_quorum_min=2,
            live_attestation_disabled=True,
            manifest_v4_disabled=False,
            host_decay_factor=W34_DEFAULT_HOST_DECAY_FACTOR,
            registered_hosts={},
        )
        n_repl = max(2, (int(n_eval) + 3) // 4)
        scenarios = build_phase67_bank(
            bank="outside_resolves", n_replicates=n_repl,
            seed=int(seed))
        if len(scenarios) > n_eval:
            scenarios = scenarios[:n_eval]
        placeholder_w21 = _build_w21_disambiguator(
            T_decoder=T_decoder,
            registrations=_build_w34_oracle_registrations_for_cell(
                bank=bank, cell_idx=0, n_total=len(scenarios)),
        )
        w33_inner = TrustEWMATrackedMultiOracleOrchestrator(
            inner=placeholder_w21,
            registry=w33_inner_registry,
            enabled=True,
            require_w33_verification=True,
        )
        orch_w34 = LiveAwareMultiAnchorOrchestrator(
            inner=w33_inner,
            registry=w34_registry,
            enabled=True,
            require_w34_verification=True,
        )
        n_attempts = 0
        n_rejected = 0
        for cell_idx in range(min(len(scenarios), n_eval)):
            registrations = (
                _build_w34_oracle_registrations_for_cell(
                    bank=bank, cell_idx=int(cell_idx),
                    n_total=len(scenarios)))
            inner = _build_w21_disambiguator(
                T_decoder=T_decoder, registrations=registrations)
            orch_w34.inner.inner = inner
            scenario = scenarios[cell_idx]
            per_rounds = _scenario_to_per_round_handoffs(scenario)
            orch_w34.decode_rounds(per_rounds)
            env = orch_w34.last_envelope
            if env is None or not env.wire_required:
                continue
            tampers = _five_named_w34_tampers(env)
            for tamper_id, tampered_env in tampers:
                outcome = (
                    verify_live_aware_multi_anchor_ratification(
                        tampered_env,
                        registered_schema=schema,
                        registered_parent_w33_cid=str(
                            env.parent_w33_cid),
                        registered_anchor_oracle_ids=frozenset(
                            w34_registry.anchor_oracle_ids),
                        registered_anchor_quorum_min=int(
                            w34_registry.multi_anchor_quorum_min),
                        registered_host_topology_cid=(
                            w34_registry.host_topology_cid),
                    ))
                n_attempts += 1
                if not outcome.ok:
                    n_rejected += 1
        seed_results.append({
            "seed": int(seed),
            "n_tamper_attempts": int(n_attempts),
            "n_tamper_rejected": int(n_rejected),
            "reject_rate": (
                round(n_rejected / n_attempts, 4)
                if n_attempts > 0 else 0.0),
        })
        cum_attempts += n_attempts
        cum_rejected += n_rejected

    return {
        "bank": str(bank),
        "seeds": list(int(s) for s in seeds),
        "n_tamper_attempts_total": int(cum_attempts),
        "n_tamper_rejected_total": int(cum_rejected),
        "reject_rate_total": (
            round(cum_rejected / cum_attempts, 4)
            if cum_attempts > 0 else 0.0),
        "seed_results": seed_results,
    }


# ---------------------------------------------------------------------------
# Response-feature signature byte-stability test
# ---------------------------------------------------------------------------


def run_phase81_response_feature_signature_byte_stability(
) -> dict[str, Any]:
    """Confirm the W34 response_feature_signature is byte-stable across
    runs of the same input.  H8 anchor.
    """
    fixtures = (
        "",
        "9",
        "12.5",
        "hello",
        "Here's the answer: 42 because of the formula n*(n-1)/2.",
        "Step 1: compute the sum.\n"
        "Step 2: divide by 2.\n"
        "Final answer: 1234.",
        "&&",
        "//",
        "<error: TimeoutError>",
        "  \n  leading whitespace",
    )
    out: list[dict[str, Any]] = []
    all_byte_stable = True
    for text in fixtures:
        sigs = [compute_response_feature_signature(response_text=text)
                for _ in range(3)]
        ok = (sigs[0] == sigs[1] == sigs[2])
        if not ok:
            all_byte_stable = False
        out.append({
            "input_text": text[:80],
            "input_length": len(text),
            "signature_run_1": sigs[0],
            "signature_run_2": sigs[1],
            "signature_run_3": sigs[2],
            "byte_stable": bool(ok),
        })
    return {
        "n_fixtures": len(fixtures),
        "all_byte_stable": bool(all_byte_stable),
        "fixtures": out,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _write_json(out_path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(obj, f, indent=2)


def _cli() -> int:
    parser = argparse.ArgumentParser(
        prog="phase81",
        description=("Phase 81 — Live-aware multi-anchor "
                      "adjudication (SDK v3.35, W34)."))
    parser.add_argument("--bank", default="double_anchor_compromise",
                         choices=("trivial_w34",
                                  "no_anchor_disagreement",
                                  "double_anchor_compromise",
                                  "anchor_betrays",
                                  "frozen_host_decay",
                                  "mis_feature_signature",
                                  "manifest_v4_tamper"))
    parser.add_argument("--n-eval", type=int, default=16)
    parser.add_argument("--bank-seed", type=int, default=11)
    parser.add_argument("--seed-sweep", action="store_true")
    parser.add_argument("--manifest-v4-tamper-sweep",
                         action="store_true")
    parser.add_argument("--response-feature-signature-byte-stability",
                         action="store_true")
    parser.add_argument("--seeds", type=str, default="11,17,23,29,31")
    parser.add_argument("--T-decoder", type=int, default=None)
    parser.add_argument("--multi-anchor-quorum-min", type=int,
                         default=W34_DEFAULT_ANCHOR_QUORUM_MIN)
    parser.add_argument("--live-attestation-disabled",
                         action="store_true", default=True)
    parser.add_argument("--manifest-v4-disabled",
                         action="store_true", default=False)
    parser.add_argument("--host-decay-factor", type=float,
                         default=W34_DEFAULT_HOST_DECAY_FACTOR)
    parser.add_argument("--anchor-oracle-ids", type=str,
                         default="service_graph,change_history")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    seeds = tuple(int(s) for s in args.seeds.split(","))
    anchor_ids = tuple(args.anchor_oracle_ids.split(","))

    if args.response_feature_signature_byte_stability:
        result = (
            run_phase81_response_feature_signature_byte_stability())
    elif args.manifest_v4_tamper_sweep:
        result = run_phase81_manifest_v4_tamper_sweep(
            bank=args.bank, n_eval=args.n_eval,
            seeds=seeds, T_decoder=args.T_decoder)
    elif args.seed_sweep:
        # Bank-specific knobs.
        man_dis = bool(args.manifest_v4_disabled)
        if args.bank in ("trivial_w34", "no_anchor_disagreement"):
            qmin = 1 if args.bank == "trivial_w34" else 2
            man_dis = True if args.bank == "trivial_w34" else man_dis
            host_dec = (1.0
                        if args.bank == "trivial_w34"
                        else args.host_decay_factor)
        elif args.bank == "frozen_host_decay":
            qmin = args.multi_anchor_quorum_min
            host_dec = 1.0
        else:
            qmin = args.multi_anchor_quorum_min
            host_dec = args.host_decay_factor
        result = run_phase81_seed_sweep(
            bank=args.bank, n_eval=args.n_eval,
            seeds=seeds, T_decoder=args.T_decoder,
            multi_anchor_quorum_min=int(qmin),
            live_attestation_disabled=bool(
                args.live_attestation_disabled),
            manifest_v4_disabled=bool(man_dis),
            host_decay_factor=float(host_dec),
            anchor_oracle_ids=anchor_ids,
        )
    else:
        result = run_phase81(
            bank=args.bank, n_eval=args.n_eval,
            bank_seed=int(args.bank_seed),
            T_decoder=args.T_decoder,
            multi_anchor_quorum_min=int(args.multi_anchor_quorum_min),
            live_attestation_disabled=bool(
                args.live_attestation_disabled),
            manifest_v4_disabled=bool(args.manifest_v4_disabled),
            host_decay_factor=float(args.host_decay_factor),
            anchor_oracle_ids=anchor_ids,
        )
        result = {k: v for k, v in result.items() if k != "records"}

    if args.out:
        _write_json(args.out, result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
