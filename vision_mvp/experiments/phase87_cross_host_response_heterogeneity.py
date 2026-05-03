"""Phase 87 -- W40 cross-host response-signature heterogeneity
ratification.

W40 wraps W39's K-of-N mutually-disjoint quorum consensus-reference
adjudication with a *response-heterogeneity* layer that operates on
an evidence axis orthogonal to top_set: the per-member response
**text bytes** themselves.

W39's deepest open wall is W39-L-FULL-DISJOINT-QUORUM-COLLUSION-CAP:
when **all** K registered disjoint quorum probes are themselves
compromised in lock-step (each emits the same wrong top_set), W39
cannot recover at the capsule layer.  W40 raises the capsule-layer
adversary bar from "compromise 2 of N trajectory hosts AND
``quorum_min`` of the K mutually-disjoint registered consensus
references" to "compromise 2 of N trajectory hosts AND
``quorum_min`` of the K mutually-disjoint registered consensus
references AND inject K diverse response text bytes that all encode
the same wrong top_set".

W40 does NOT close W39-L-FULL-DISJOINT-QUORUM-COLLUSION-CAP in
general; it bounds the *naive* full-quorum-collusion attack (where
the colluder pushes K identical wrong response bytes) but cannot
recover when the adversary diversifies response bytes while holding
the wrong top_set in lock-step (the new
W40-L-COORDINATED-DIVERSE-RESPONSE-CAP limitation theorem).

This is still a capsule-layer audited proxy.  It does not read
transformer hidden states, transplant KV cache, or claim native
latent transfer.  The heterogeneity score is a closed-form Jaccard
divergence over canonical sorted token bags (zero parameters).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from typing import Any, Sequence

from vision_mvp.wevra.team_coord import (
    AttentionAwareBundleDecoder,
    BundleContradictionDisambiguator,
    OracleRegistration,
    ServiceGraphOracle,
    ChangeHistoryOracle,
    OnCallNotesOracle,
    RelationalCompatibilityDisambiguator,
    TrustWeightedMultiOracleDisambiguator,
    TrustEWMATrackedMultiOracleOrchestrator,
    build_trust_ewma_registry,
    W33_DEFAULT_TRUST_THRESHOLD,
    W33_DEFAULT_TRUST_TRAJECTORY_WINDOW,
    W33_DEFAULT_EWMA_ALPHA,
    LiveAwareMultiAnchorOrchestrator,
    LiveOracleAttestation,
    HostRegistration,
    build_live_aware_registry,
    W34_DEFAULT_ANCHOR_QUORUM_MIN,
    TrustSubspaceDenseControlOrchestrator,
    build_trivial_trust_subspace_registry,
    build_trust_subspace_dense_registry,
    W35_DEFAULT_BASIS_EWMA_ALPHA,
    W35_DEFAULT_PROJECTION_THRESHOLD,
    W35_DEFAULT_PROJECTION_MARGIN_MIN,
    W35_DEFAULT_BASIS_HISTORY_WINDOW,
    W35_DEFAULT_MIN_BASIS_OBSERVATIONS,
    HostDiverseTrustSubspaceOrchestrator,
    build_trivial_host_diverse_registry,
    build_host_diverse_registry,
    W36_DEFAULT_MIN_DISTINCT_HOSTS,
    W36_DEFAULT_HOST_DIVERSITY_THRESHOLD,
    W36_DEFAULT_HOST_DIVERSITY_MARGIN_MIN,
    CrossHostBasisTrajectoryOrchestrator,
    build_trivial_cross_host_trajectory_registry,
    build_cross_host_trajectory_registry,
    W37_DEFAULT_TRAJECTORY_EWMA_ALPHA,
    W37_DEFAULT_TRAJECTORY_THRESHOLD,
    W37_DEFAULT_TRAJECTORY_MARGIN_MIN,
    W37_DEFAULT_MIN_ANCHORED_OBSERVATIONS,
    W37_DEFAULT_MIN_TRAJECTORY_ANCHORED_HOSTS,
    compute_response_feature_signature,
    DisjointConsensusReferenceOrchestrator,
    ConsensusReferenceProbe,
    build_trivial_disjoint_consensus_registry,
    build_disjoint_consensus_registry,
    W38_DEFAULT_CONSENSUS_STRENGTH_MIN,
    W38_DEFAULT_DIVERGENCE_MARGIN_MIN,
    MultiHostDisjointQuorumProbe,
    MultiHostDisjointQuorumOrchestrator,
    build_trivial_multi_host_disjoint_quorum_registry,
    build_multi_host_disjoint_quorum_registry,
    W39_DEFAULT_QUORUM_MIN,
    W39_DEFAULT_MIN_QUORUM_PROBES,
    W39_DEFAULT_QUORUM_STRENGTH_MIN,
    W39_DEFAULT_QUORUM_DIVERGENCE_MARGIN_MIN,
    W39_BRANCH_QUORUM_RATIFIED,
    ResponseSignatureProbe,
    MultiHostResponseHeterogeneityProbe,
    CrossHostResponseHeterogeneityOrchestrator,
    build_trivial_cross_host_response_heterogeneity_registry,
    build_cross_host_response_heterogeneity_registry,
    W40_DEFAULT_RESPONSE_TEXT_DIVERSITY_MIN,
    W40_DEFAULT_MIN_RESPONSE_SIGNATURE_PROBES,
    W40_BRANCH_RESPONSE_SIGNATURE_DIVERSE,
    W40_BRANCH_RESPONSE_SIGNATURE_COLLAPSE_ABSTAINED,
    W40_BRANCH_RESPONSE_SIGNATURE_NO_REFERENCES,
    W40_BRANCH_RESPONSE_SIGNATURE_NO_TRIGGER,
    W40_BRANCH_RESPONSE_SIGNATURE_INSUFFICIENT,
    W40_BRANCH_RESPONSE_SIGNATURE_INCOMPLETE,
    W40_BRANCH_TRIVIAL_RESPONSE_SIGNATURE_PASSTHROUGH,
    _w40_canonical_token_bag,
    _w40_compute_response_signature_cid,
    _DecodedHandoff,
)
from vision_mvp.experiments.phase67_outside_information import (
    build_phase67_bank,
)


# Phase 87 banks.  Each bank exercises a different W40 contract.
#
#   * trivial_w40                         -- byte-for-W39 passthrough
#   * response_signature_collapse         -- the load-bearing W40-over-W39
#                                            gain regime: trajectory hosts
#                                            collude, the W38 single
#                                            consensus colludes, the W39
#                                            quorum colludes (W39-L-FULL-
#                                            DISJOINT-QUORUM-COLLUSION-CAP
#                                            fires => W39 ratifies wrong
#                                            top_set) AND every quorum
#                                            member emits IDENTICAL wrong
#                                            response text bytes; W40
#                                            detects collapse and abstains
#   * no_regression_diverse_agrees        -- W39 ratifies AND quorum members
#                                            emit naturally diverse response
#                                            bytes; W40 ratifies (no
#                                            regression vs W39)
#   * coordinated_diverse_response        -- W39 collusion attack succeeds
#                                            AND every quorum member emits
#                                            DIFFERENT response bytes (the
#                                            "smart" attacker who diversifies
#                                            response text while keeping the
#                                            same wrong top_set); W40 cannot
#                                            recover (W40-L-COORDINATED-
#                                            DIVERSE-RESPONSE-CAP fires)
#   * insufficient_response_signature     -- only 1 W40 member probe is
#                                            provided in the recovery half
#                                            (below min_response_signature_
#                                            probes=2); W40 reduces to W39
#                                            via RESPONSE_SIGNATURE_
#                                            INSUFFICIENT
_PHASE87_K1_FRACTION = 1.0 / 2.0


def _stable_schema_capsule():
    from vision_mvp.wevra.team_coord import (
        build_incident_triage_schema_capsule)
    return build_incident_triage_schema_capsule()


def _scenario_to_per_round_handoffs(
        scenario: Any,
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


def _build_w40_oracle_registrations_for_cell(
        *,
        bank: str,
        cell_idx: int,
        n_total: int,
) -> tuple[OracleRegistration, ...]:
    sg = ServiceGraphOracle(oracle_id="service_graph")
    ch = ChangeHistoryOracle(oracle_id="change_history")
    oc = OnCallNotesOracle(oracle_id="oncall_notes")
    return (
        OracleRegistration(
            oracle=sg, trust_prior=1.0, role_label="service_graph"),
        OracleRegistration(
            oracle=ch, trust_prior=1.0, role_label="change_history"),
        OracleRegistration(
            oracle=oc, trust_prior=1.0, role_label="oncall_notes"),
    )


def _registered_hosts() -> dict[str, HostRegistration]:
    return {
        "mac1": HostRegistration(
            host_id="mac1", model_id="local-qwen",
            base_url="http://localhost:11434", preflight_ok=True),
        "mac_remote": HostRegistration(
            host_id="mac_remote", model_id="remote-qwen14b",
            base_url="http://192.168.12.191:11434", preflight_ok=True),
        "mac_shadow": HostRegistration(
            host_id="mac_shadow", model_id="shadow-adjudicator",
            base_url="mock://shadow", preflight_ok=True),
    }


def _make_live_attestation_provider(*, bank: str, n_total: int):
    K1 = max(1, int(_PHASE87_K1_FRACTION * n_total))

    def provider(orch: Any, _w33_result: Any) -> tuple[
            LiveOracleAttestation, ...]:
        cell_idx = int(getattr(orch, "_cell_index", 0))
        if bank == "trivial_w40":
            return ()
        if cell_idx < K1:
            host_map = {
                "service_graph": (
                    "mac1", "local-qwen", True),
                "change_history": (
                    "mac_remote", "remote-qwen14b", True),
                "oncall_notes": (
                    "mac_shadow", "shadow-adjudicator", True),
            }
        else:
            host_map = {
                "service_graph": (
                    "mac1", "local-qwen", True),
                "change_history": (
                    "mac1", "local-qwen", True),
                "oncall_notes": (
                    "mac1", "local-qwen", True),
            }
        out: list[LiveOracleAttestation] = []
        for oracle_id in (
                "service_graph", "change_history", "oncall_notes"):
            host_id, model_id, ok = host_map[oracle_id]
            sig = compute_response_feature_signature(
                response_text=f"{oracle_id}:{host_id}:{model_id}")
            out.append(LiveOracleAttestation(
                oracle_id=oracle_id,
                host_id=host_id,
                model_id=model_id,
                response_feature_signature=sig,
                latency_ms_bucket="0..1k",
                preflight_ok=bool(ok),
            ))
        return tuple(out)

    return provider


def _make_consensus_reference_provider(
        *, bank: str, n_total: int,
        consensus_top_per_cell: dict[int, tuple[str, ...]],
):
    """W38 single consensus reference provider for Phase 87.

    In Phase 87 banks, the W38-style single disjoint consensus
    reference is COMPROMISED IN LOCK-STEP with the colluding
    trajectory hosts in the recovery half (the W38-L-CONSENSUS-
    COLLUSION-CAP regime).  The W39 quorum is the cross-check at
    the W39 layer, and the W40 response-heterogeneity is the
    cross-check at the W40 layer.
    """

    def provider(orch: Any, w37_result: Any) -> (
            ConsensusReferenceProbe | None):
        cell_idx = int(getattr(orch, "_cell_index", 0))
        if bank in ("trivial_w40",):
            return None
        top = consensus_top_per_cell.get(cell_idx, ())
        if not top:
            return None
        return ConsensusReferenceProbe(
            top_set=tuple(sorted(str(t) for t in top)),
            consensus_host_ids=("mac_consensus",),
            consensus_oracle_ids=("disjoint_change_history",
                                  "disjoint_oncall_notes"),
            consensus_strength=1.0,
            cell_idx=int(cell_idx),
        )

    return provider


def _make_quorum_provider(
        *, bank: str, n_total: int,
        quorum_top_per_cell: dict[int, tuple[tuple[str, ...], ...]],
):
    """W39 multi-host disjoint quorum provider for Phase 87.

    In every Phase 87 bank, the W39 quorum is constructed so that
    its top_set decision matches the colluded trajectory top_set
    in the recovery half (i.e. W39-L-FULL-DISJOINT-QUORUM-
    COLLUSION-CAP fires); W40's job is to detect or fail to detect
    this at the response-text bytes layer, not the top_set layer.
    """

    def provider(orch: Any, w38_result: Any) -> (
            MultiHostDisjointQuorumProbe | None):
        cell_idx = int(getattr(orch, "_cell_index", 0))
        if bank == "trivial_w40":
            return None
        per_cell = quorum_top_per_cell.get(cell_idx)
        if not per_cell:
            return None
        pool_host_ids = (
            ("mac_off_cluster_a",),
            ("mac_off_cluster_b",),
        )
        pool_oracle_ids = (
            ("disjoint_quorum_oracle_a",),
            ("disjoint_quorum_oracle_b",),
        )
        members: list[ConsensusReferenceProbe] = []
        for k, top in enumerate(per_cell):
            if k >= len(pool_host_ids):
                break
            members.append(ConsensusReferenceProbe(
                top_set=tuple(sorted(str(t) for t in top)),
                consensus_host_ids=pool_host_ids[k],
                consensus_oracle_ids=pool_oracle_ids[k],
                consensus_strength=1.0,
                cell_idx=int(cell_idx),
            ))
        if not members:
            return None
        return MultiHostDisjointQuorumProbe(
            member_probes=tuple(members),
            quorum_min=W39_DEFAULT_QUORUM_MIN,
            min_quorum_probes=W39_DEFAULT_MIN_QUORUM_PROBES,
            cell_idx=int(cell_idx),
        )

    return provider


def _make_response_provider(
        *, bank: str, n_total: int,
        response_signature_per_cell: dict[
            int, tuple[tuple[str, str], ...]],
):
    """W40 cross-host response-heterogeneity provider for Phase 87.

    For every cell, returns a
    :class:`MultiHostResponseHeterogeneityProbe` consisting of K
    member probes, one per registered disjoint physical-host pool.
    Each member probe carries ``(member_index, host_ids,
    oracle_ids, response_token_bag, response_signature_cid,
    cell_idx)``.

    Bank semantics on the recovery half:

    * ``no_regression_diverse_agrees`` -- naturally diverse
      response bytes (different paraphrases per host) AND W39
      ratifies => W40 ratifies (no regression).
    * ``response_signature_collapse`` -- all K member probes emit
      IDENTICAL wrong response bytes (the "naive" full-quorum-
      collusion attack with byte-identical response payloads); W40
      detects collapse and abstains.
    * ``coordinated_diverse_response`` -- all K member probes
      emit DIFFERENT wrong response bytes that nevertheless encode
      the same wrong top_set; W40 cannot recover at the capsule
      layer (W40-L-COORDINATED-DIVERSE-RESPONSE-CAP).
    * ``insufficient_response_signature`` -- only 1 member probe is
      provided in the recovery half (below
      ``min_response_signature_probes=2``); W40 reduces to W39 via
      RESPONSE_SIGNATURE_INSUFFICIENT.
    """

    def provider(orch: Any, w39_result: Any) -> (
            MultiHostResponseHeterogeneityProbe | None):
        cell_idx = int(getattr(orch, "_cell_index", 0))
        if bank == "trivial_w40":
            return None
        per_cell = response_signature_per_cell.get(cell_idx)
        if not per_cell:
            return None
        pool_host_ids = (
            ("mac_off_cluster_a",),
            ("mac_off_cluster_b",),
        )
        pool_oracle_ids = (
            ("disjoint_quorum_oracle_a",),
            ("disjoint_quorum_oracle_b",),
        )
        members: list[ResponseSignatureProbe] = []
        for k, (host_label, response_text) in enumerate(per_cell):
            if k >= len(pool_host_ids):
                break
            bag = _w40_canonical_token_bag(response_text)
            sig_cid = _w40_compute_response_signature_cid(
                response_text=response_text)
            members.append(ResponseSignatureProbe(
                member_index=int(k),
                host_ids=pool_host_ids[k],
                oracle_ids=pool_oracle_ids[k],
                response_token_bag=bag,
                response_signature_cid=sig_cid,
                cell_idx=int(cell_idx),
            ))
        if not members:
            return None
        return MultiHostResponseHeterogeneityProbe(
            member_probes=tuple(members),
            response_text_diversity_min=(
                W40_DEFAULT_RESPONSE_TEXT_DIVERSITY_MIN),
            min_response_signature_probes=(
                W40_DEFAULT_MIN_RESPONSE_SIGNATURE_PROBES),
            cell_idx=int(cell_idx),
        )

    return provider


def _gold_for_cell(scenario: Any) -> tuple[str, ...]:
    return tuple(sorted(scenario.gold_services_pair))


def _is_correct(answer: dict[str, Any], gold: tuple[str, ...]) -> bool:
    services = tuple(sorted(set(answer.get("services", ()))))
    return services == tuple(sorted(set(gold)))


def _is_ratified(answer: dict[str, Any]) -> bool:
    return bool(answer.get("services", ()))


@dataclasses.dataclass
class _Phase87Record:
    cell_idx: int
    expected: tuple[str, ...]
    correct_substrate_fifo: bool
    correct_w21: bool
    correct_w36: bool
    correct_w37: bool
    correct_w38: bool
    correct_w39: bool
    correct_w40: bool
    ratified_substrate_fifo: bool
    ratified_w21: bool
    ratified_w36: bool
    ratified_w37: bool
    ratified_w38: bool
    ratified_w39: bool
    ratified_w40: bool
    w37_decoder_branch: str
    w38_decoder_branch: str
    w39_decoder_branch: str
    w40_decoder_branch: str
    w40_projection_branch: str
    w39_decision_top_set: tuple[str, ...]
    w40_n_diverse_pairs: int
    w40_n_collapse_pairs: int
    w40_n_pairs: int
    w40_n_total: int
    w40_mean_pairwise_jaccard: float
    w39_visible: int
    w40_visible: int
    w40_overhead: int
    w40_structured_bits: int
    w40_cram_factor: float


def _make_w33(
        *, schema, anchor_ids, T_decoder, bank, n_eval, quorum_min):
    registry = build_trust_ewma_registry(
        schema=schema,
        registered_oracle_ids=(
            "service_graph", "change_history", "oncall_notes"),
        anchor_oracle_ids=anchor_ids,
        trust_ewma_enabled=True,
        manifest_v3_disabled=False,
        trust_trajectory_window=W33_DEFAULT_TRUST_TRAJECTORY_WINDOW,
        trust_threshold=W33_DEFAULT_TRUST_THRESHOLD,
        ewma_alpha=W33_DEFAULT_EWMA_ALPHA,
    )
    placeholder = _build_w21_disambiguator(
        T_decoder=T_decoder,
        registrations=_build_w40_oracle_registrations_for_cell(
            bank=bank, cell_idx=0, n_total=n_eval),
        quorum_min=quorum_min)
    return TrustEWMATrackedMultiOracleOrchestrator(
        inner=placeholder, registry=registry,
        enabled=True, require_w33_verification=True)


def _make_w34(*, schema, T_decoder, bank, n_eval, anchor_ids,
              quorum_min, live_attestation_disabled):
    w33 = _make_w33(
        schema=schema, anchor_ids=anchor_ids,
        T_decoder=T_decoder, bank=bank, n_eval=n_eval,
        quorum_min=quorum_min)
    w34_registry = build_live_aware_registry(
        schema=schema, inner_w33_registry=w33.registry,
        multi_anchor_quorum_min=W34_DEFAULT_ANCHOR_QUORUM_MIN,
        live_attestation_disabled=bool(live_attestation_disabled),
        manifest_v4_disabled=False, host_decay_factor=1.0,
        registered_hosts=_registered_hosts())
    orch = LiveAwareMultiAnchorOrchestrator(
        inner=w33, registry=w34_registry,
        enabled=True, require_w34_verification=True)
    if not live_attestation_disabled:
        orch.set_live_attestation_provider(
            _make_live_attestation_provider(
                bank=bank, n_total=n_eval))
    return orch


def _make_w35(*, schema, T_decoder, bank, n_eval, anchor_ids,
              quorum_min, live_attestation_disabled,
              trust_subspace_enabled, manifest_v5_disabled):
    w34 = _make_w34(
        schema=schema, T_decoder=T_decoder, bank=bank, n_eval=n_eval,
        anchor_ids=anchor_ids, quorum_min=quorum_min,
        live_attestation_disabled=live_attestation_disabled)
    if not trust_subspace_enabled and manifest_v5_disabled:
        registry = build_trivial_trust_subspace_registry(
            schema=schema,
            inner_w34_registry=w34.registry,
            registered_oracle_ids=(
                "service_graph", "change_history", "oncall_notes"))
    else:
        registry = build_trust_subspace_dense_registry(
            schema=schema,
            inner_w34_registry=w34.registry,
            registered_oracle_ids=(
                "service_graph", "change_history", "oncall_notes"),
            trust_subspace_enabled=bool(trust_subspace_enabled),
            manifest_v5_disabled=bool(manifest_v5_disabled),
            basis_history_window=W35_DEFAULT_BASIS_HISTORY_WINDOW,
            basis_ewma_alpha=W35_DEFAULT_BASIS_EWMA_ALPHA,
            projection_threshold=W35_DEFAULT_PROJECTION_THRESHOLD,
            projection_margin_min=W35_DEFAULT_PROJECTION_MARGIN_MIN,
            min_basis_observations=W35_DEFAULT_MIN_BASIS_OBSERVATIONS,
            abstain_on_unstable_consensus=True)
    return TrustSubspaceDenseControlOrchestrator(
        inner=w34, registry=registry,
        enabled=True, require_w35_verification=True)


def _make_w36(*, schema, T_decoder, bank, n_eval, anchor_ids,
              quorum_min, live_attestation_disabled,
              trust_subspace_enabled, manifest_v5_disabled,
              host_diversity_enabled, manifest_v6_disabled,
              min_distinct_hosts):
    w35 = _make_w35(
        schema=schema, T_decoder=T_decoder, bank=bank, n_eval=n_eval,
        anchor_ids=anchor_ids, quorum_min=quorum_min,
        live_attestation_disabled=live_attestation_disabled,
        trust_subspace_enabled=trust_subspace_enabled,
        manifest_v5_disabled=manifest_v5_disabled)
    if not host_diversity_enabled and manifest_v6_disabled:
        registry = build_trivial_host_diverse_registry(
            schema=schema,
            inner_w35_registry=w35.registry,
            registered_oracle_ids=(
                "service_graph", "change_history", "oncall_notes"))
    else:
        registry = build_host_diverse_registry(
            schema=schema,
            inner_w35_registry=w35.registry,
            registered_oracle_ids=(
                "service_graph", "change_history", "oncall_notes"),
            registered_hosts=_registered_hosts(),
            host_diversity_enabled=bool(host_diversity_enabled),
            manifest_v6_disabled=bool(manifest_v6_disabled),
            min_distinct_hosts=int(min_distinct_hosts),
            host_diversity_threshold=W36_DEFAULT_HOST_DIVERSITY_THRESHOLD,
            host_diversity_margin_min=(
                W36_DEFAULT_HOST_DIVERSITY_MARGIN_MIN),
            abstain_on_unverified_host_projection=True)
    return HostDiverseTrustSubspaceOrchestrator(
        inner=w35, registry=registry,
        enabled=True, require_w36_verification=True)


def _make_w37(*, schema, T_decoder, bank, n_eval, anchor_ids,
              quorum_min, live_attestation_disabled,
              trust_subspace_enabled, manifest_v5_disabled,
              host_diversity_enabled, manifest_v6_disabled,
              min_distinct_hosts,
              trajectory_enabled, manifest_v7_disabled,
              allow_single_host_trajectory_reroute,
              trajectory_threshold, trajectory_margin_min,
              min_anchored_observations,
              min_trajectory_anchored_hosts,
              registered_anchor_host_ids):
    w36 = _make_w36(
        schema=schema, T_decoder=T_decoder, bank=bank, n_eval=n_eval,
        anchor_ids=anchor_ids, quorum_min=quorum_min,
        live_attestation_disabled=live_attestation_disabled,
        trust_subspace_enabled=trust_subspace_enabled,
        manifest_v5_disabled=manifest_v5_disabled,
        host_diversity_enabled=host_diversity_enabled,
        manifest_v6_disabled=manifest_v6_disabled,
        min_distinct_hosts=min_distinct_hosts)
    if (not trajectory_enabled
            and manifest_v7_disabled
            and not allow_single_host_trajectory_reroute):
        registry = build_trivial_cross_host_trajectory_registry(
            schema=schema,
            inner_w36_registry=w36.registry,
            registered_oracle_ids=(
                "service_graph", "change_history", "oncall_notes"))
    else:
        registry = build_cross_host_trajectory_registry(
            schema=schema,
            inner_w36_registry=w36.registry,
            registered_oracle_ids=(
                "service_graph", "change_history", "oncall_notes"),
            registered_host_ids=tuple(_registered_hosts().keys()),
            registered_anchor_host_ids=tuple(
                registered_anchor_host_ids),
            trajectory_enabled=bool(trajectory_enabled),
            manifest_v7_disabled=bool(manifest_v7_disabled),
            allow_single_host_trajectory_reroute=bool(
                allow_single_host_trajectory_reroute),
            trajectory_ewma_alpha=W37_DEFAULT_TRAJECTORY_EWMA_ALPHA,
            trajectory_threshold=float(trajectory_threshold),
            trajectory_margin_min=float(trajectory_margin_min),
            min_anchored_observations=int(min_anchored_observations),
            min_trajectory_anchored_hosts=int(
                min_trajectory_anchored_hosts))
    return CrossHostBasisTrajectoryOrchestrator(
        inner=w36, registry=registry, enabled=True,
        require_w37_verification=True)


def _make_w38(*, schema, T_decoder, bank, n_eval, anchor_ids,
              quorum_min, live_attestation_disabled,
              trust_subspace_enabled, manifest_v5_disabled,
              host_diversity_enabled, manifest_v6_disabled,
              min_distinct_hosts,
              trajectory_enabled, manifest_v7_disabled,
              allow_single_host_trajectory_reroute,
              trajectory_threshold, trajectory_margin_min,
              min_anchored_observations,
              min_trajectory_anchored_hosts,
              registered_anchor_host_ids,
              consensus_enabled, manifest_v8_disabled,
              allow_consensus_reference_divergence_abstain,
              consensus_strength_min, divergence_margin_min,
              consensus_top_per_cell):
    w37 = _make_w37(
        schema=schema, T_decoder=T_decoder, bank=bank, n_eval=n_eval,
        anchor_ids=anchor_ids, quorum_min=quorum_min,
        live_attestation_disabled=live_attestation_disabled,
        trust_subspace_enabled=trust_subspace_enabled,
        manifest_v5_disabled=manifest_v5_disabled,
        host_diversity_enabled=host_diversity_enabled,
        manifest_v6_disabled=manifest_v6_disabled,
        min_distinct_hosts=min_distinct_hosts,
        trajectory_enabled=trajectory_enabled,
        manifest_v7_disabled=manifest_v7_disabled,
        allow_single_host_trajectory_reroute=(
            allow_single_host_trajectory_reroute),
        trajectory_threshold=trajectory_threshold,
        trajectory_margin_min=trajectory_margin_min,
        min_anchored_observations=min_anchored_observations,
        min_trajectory_anchored_hosts=min_trajectory_anchored_hosts,
        registered_anchor_host_ids=registered_anchor_host_ids)
    if (not consensus_enabled and manifest_v8_disabled
            and not allow_consensus_reference_divergence_abstain):
        registry = build_trivial_disjoint_consensus_registry(
            schema=schema, inner_w37_registry=w37.registry)
    else:
        registry = build_disjoint_consensus_registry(
            schema=schema,
            inner_w37_registry=w37.registry,
            registered_consensus_host_ids=("mac_consensus",),
            registered_consensus_oracle_ids=(
                "disjoint_change_history", "disjoint_oncall_notes"),
            registered_trajectory_host_ids=tuple(
                _registered_hosts().keys()),
            consensus_enabled=bool(consensus_enabled),
            manifest_v8_disabled=bool(manifest_v8_disabled),
            allow_consensus_reference_divergence_abstain=bool(
                allow_consensus_reference_divergence_abstain),
            consensus_strength_min=float(consensus_strength_min),
            divergence_margin_min=float(divergence_margin_min))
    orch = DisjointConsensusReferenceOrchestrator(
        inner=w37, registry=registry,
        enabled=True, require_w38_verification=True)
    if (consensus_enabled and bank not in ("trivial_w40",)):
        orch.set_consensus_reference_provider(
            _make_consensus_reference_provider(
                bank=bank, n_total=n_eval,
                consensus_top_per_cell=consensus_top_per_cell))
    return orch


def _make_w39(*, schema, T_decoder, bank, n_eval, anchor_ids,
              quorum_min, live_attestation_disabled,
              trust_subspace_enabled, manifest_v5_disabled,
              host_diversity_enabled, manifest_v6_disabled,
              min_distinct_hosts,
              trajectory_enabled, manifest_v7_disabled,
              allow_single_host_trajectory_reroute,
              trajectory_threshold, trajectory_margin_min,
              min_anchored_observations,
              min_trajectory_anchored_hosts,
              registered_anchor_host_ids,
              consensus_enabled, manifest_v8_disabled,
              allow_consensus_reference_divergence_abstain,
              consensus_strength_min, divergence_margin_min,
              consensus_top_per_cell,
              quorum_enabled, manifest_v9_disabled,
              allow_disjoint_quorum_divergence_abstain,
              w39_quorum_min, w39_min_quorum_probes,
              w39_consensus_strength_min,
              w39_divergence_margin_min,
              quorum_top_per_cell):
    w38 = _make_w38(
        schema=schema, T_decoder=T_decoder, bank=bank, n_eval=n_eval,
        anchor_ids=anchor_ids, quorum_min=quorum_min,
        live_attestation_disabled=live_attestation_disabled,
        trust_subspace_enabled=trust_subspace_enabled,
        manifest_v5_disabled=manifest_v5_disabled,
        host_diversity_enabled=host_diversity_enabled,
        manifest_v6_disabled=manifest_v6_disabled,
        min_distinct_hosts=min_distinct_hosts,
        trajectory_enabled=trajectory_enabled,
        manifest_v7_disabled=manifest_v7_disabled,
        allow_single_host_trajectory_reroute=(
            allow_single_host_trajectory_reroute),
        trajectory_threshold=trajectory_threshold,
        trajectory_margin_min=trajectory_margin_min,
        min_anchored_observations=min_anchored_observations,
        min_trajectory_anchored_hosts=min_trajectory_anchored_hosts,
        registered_anchor_host_ids=registered_anchor_host_ids,
        consensus_enabled=consensus_enabled,
        manifest_v8_disabled=manifest_v8_disabled,
        allow_consensus_reference_divergence_abstain=(
            allow_consensus_reference_divergence_abstain),
        consensus_strength_min=consensus_strength_min,
        divergence_margin_min=divergence_margin_min,
        consensus_top_per_cell=consensus_top_per_cell)
    if (not quorum_enabled and manifest_v9_disabled
            and not allow_disjoint_quorum_divergence_abstain):
        registry = build_trivial_multi_host_disjoint_quorum_registry(
            schema=schema, inner_w38_registry=w38.registry)
    else:
        registry = build_multi_host_disjoint_quorum_registry(
            schema=schema,
            inner_w38_registry=w38.registry,
            registered_quorum_pool_host_ids=(
                ("mac_off_cluster_a",),
                ("mac_off_cluster_b",),
            ),
            registered_quorum_pool_oracle_ids=(
                ("disjoint_quorum_oracle_a",),
                ("disjoint_quorum_oracle_b",),
            ),
            registered_trajectory_host_ids=tuple(
                _registered_hosts().keys()) + ("mac_consensus",),
            quorum_enabled=bool(quorum_enabled),
            manifest_v9_disabled=bool(manifest_v9_disabled),
            allow_disjoint_quorum_divergence_abstain=bool(
                allow_disjoint_quorum_divergence_abstain),
            quorum_min=int(w39_quorum_min),
            min_quorum_probes=int(w39_min_quorum_probes),
            consensus_strength_min=float(w39_consensus_strength_min),
            divergence_margin_min=float(w39_divergence_margin_min))
    orch = MultiHostDisjointQuorumOrchestrator(
        inner=w38, registry=registry,
        enabled=True, require_w39_verification=True)
    if quorum_enabled and bank not in ("trivial_w40",):
        orch.set_quorum_provider(
            _make_quorum_provider(
                bank=bank, n_total=n_eval,
                quorum_top_per_cell=quorum_top_per_cell))
    return orch


def _make_w40(*, schema, T_decoder, bank, n_eval, anchor_ids,
              quorum_min, live_attestation_disabled,
              trust_subspace_enabled, manifest_v5_disabled,
              host_diversity_enabled, manifest_v6_disabled,
              min_distinct_hosts,
              trajectory_enabled, manifest_v7_disabled,
              allow_single_host_trajectory_reroute,
              trajectory_threshold, trajectory_margin_min,
              min_anchored_observations,
              min_trajectory_anchored_hosts,
              registered_anchor_host_ids,
              consensus_enabled, manifest_v8_disabled,
              allow_consensus_reference_divergence_abstain,
              consensus_strength_min, divergence_margin_min,
              consensus_top_per_cell,
              quorum_enabled, manifest_v9_disabled,
              allow_disjoint_quorum_divergence_abstain,
              w39_quorum_min, w39_min_quorum_probes,
              w39_consensus_strength_min,
              w39_divergence_margin_min,
              quorum_top_per_cell,
              response_signature_enabled,
              manifest_v10_disabled,
              allow_response_signature_collapse_abstain,
              response_text_diversity_min,
              min_response_signature_probes,
              response_signature_per_cell):
    w39 = _make_w39(
        schema=schema, T_decoder=T_decoder, bank=bank, n_eval=n_eval,
        anchor_ids=anchor_ids, quorum_min=quorum_min,
        live_attestation_disabled=live_attestation_disabled,
        trust_subspace_enabled=trust_subspace_enabled,
        manifest_v5_disabled=manifest_v5_disabled,
        host_diversity_enabled=host_diversity_enabled,
        manifest_v6_disabled=manifest_v6_disabled,
        min_distinct_hosts=min_distinct_hosts,
        trajectory_enabled=trajectory_enabled,
        manifest_v7_disabled=manifest_v7_disabled,
        allow_single_host_trajectory_reroute=(
            allow_single_host_trajectory_reroute),
        trajectory_threshold=trajectory_threshold,
        trajectory_margin_min=trajectory_margin_min,
        min_anchored_observations=min_anchored_observations,
        min_trajectory_anchored_hosts=min_trajectory_anchored_hosts,
        registered_anchor_host_ids=registered_anchor_host_ids,
        consensus_enabled=consensus_enabled,
        manifest_v8_disabled=manifest_v8_disabled,
        allow_consensus_reference_divergence_abstain=(
            allow_consensus_reference_divergence_abstain),
        consensus_strength_min=consensus_strength_min,
        divergence_margin_min=divergence_margin_min,
        consensus_top_per_cell=consensus_top_per_cell,
        quorum_enabled=quorum_enabled,
        manifest_v9_disabled=manifest_v9_disabled,
        allow_disjoint_quorum_divergence_abstain=(
            allow_disjoint_quorum_divergence_abstain),
        w39_quorum_min=w39_quorum_min,
        w39_min_quorum_probes=w39_min_quorum_probes,
        w39_consensus_strength_min=w39_consensus_strength_min,
        w39_divergence_margin_min=w39_divergence_margin_min,
        quorum_top_per_cell=quorum_top_per_cell)
    if (not response_signature_enabled
            and manifest_v10_disabled
            and not allow_response_signature_collapse_abstain):
        registry = (
            build_trivial_cross_host_response_heterogeneity_registry(
                schema=schema,
                inner_w39_registry=w39.registry))
    else:
        registry = build_cross_host_response_heterogeneity_registry(
            schema=schema,
            inner_w39_registry=w39.registry,
            registered_member_pool_host_ids=(
                ("mac_off_cluster_a",),
                ("mac_off_cluster_b",),
            ),
            registered_member_pool_oracle_ids=(
                ("disjoint_quorum_oracle_a",),
                ("disjoint_quorum_oracle_b",),
            ),
            registered_trajectory_host_ids=tuple(
                _registered_hosts().keys())
                + ("mac_consensus",),
            response_signature_enabled=bool(
                response_signature_enabled),
            manifest_v10_disabled=bool(manifest_v10_disabled),
            allow_response_signature_collapse_abstain=bool(
                allow_response_signature_collapse_abstain),
            response_text_diversity_min=float(
                response_text_diversity_min),
            min_response_signature_probes=int(
                min_response_signature_probes))
    orch = CrossHostResponseHeterogeneityOrchestrator(
        inner=w39, registry=registry,
        enabled=True, require_w40_verification=True)
    if response_signature_enabled and bank not in ("trivial_w40",):
        orch.set_response_provider(
            _make_response_provider(
                bank=bank, n_total=n_eval,
                response_signature_per_cell=(
                    response_signature_per_cell)))
    return orch


def _interleave_by_family(scenarios: list, n_families: int = 4) -> list:
    n = len(scenarios)
    if n_families <= 1 or n % n_families != 0:
        return list(scenarios)
    n_replicates = n // n_families
    out = []
    for r in range(n_replicates):
        for f in range(n_families):
            idx = f * n_replicates + r
            if idx < n:
                out.append(scenarios[idx])
    return out


def run_phase87(
        *,
        bank: str = "response_signature_collapse",
        n_eval: int = 16,
        bank_seed: int = 11,
        bank_replicates: int = 4,
        T_decoder: int | None = None,
        anchor_oracle_ids: tuple[str, ...] = (
            "service_graph", "change_history"),
        quorum_min: int = 2,
        live_attestation_disabled: bool = False,
        trust_subspace_enabled: bool = True,
        manifest_v5_disabled: bool = False,
        host_diversity_enabled: bool = True,
        manifest_v6_disabled: bool = False,
        min_distinct_hosts: int = W36_DEFAULT_MIN_DISTINCT_HOSTS,
        trajectory_enabled: bool = True,
        manifest_v7_disabled: bool = False,
        allow_single_host_trajectory_reroute: bool = True,
        trajectory_threshold: float = (
            W37_DEFAULT_TRAJECTORY_THRESHOLD),
        trajectory_margin_min: float = (
            W37_DEFAULT_TRAJECTORY_MARGIN_MIN),
        min_anchored_observations: int = (
            W37_DEFAULT_MIN_ANCHORED_OBSERVATIONS),
        min_trajectory_anchored_hosts: int = (
            W37_DEFAULT_MIN_TRAJECTORY_ANCHORED_HOSTS),
        registered_anchor_host_ids: tuple[str, ...] = (
            "mac_remote", "mac_shadow", "mac1"),
        consensus_enabled: bool = True,
        manifest_v8_disabled: bool = False,
        allow_consensus_reference_divergence_abstain: bool = True,
        consensus_strength_min: float = (
            W38_DEFAULT_CONSENSUS_STRENGTH_MIN),
        divergence_margin_min: float = (
            W38_DEFAULT_DIVERGENCE_MARGIN_MIN),
        quorum_enabled: bool = True,
        manifest_v9_disabled: bool = False,
        allow_disjoint_quorum_divergence_abstain: bool = True,
        w39_quorum_min: int = W39_DEFAULT_QUORUM_MIN,
        w39_min_quorum_probes: int = W39_DEFAULT_MIN_QUORUM_PROBES,
        w39_consensus_strength_min: float = (
            W39_DEFAULT_QUORUM_STRENGTH_MIN),
        w39_divergence_margin_min: float = (
            W39_DEFAULT_QUORUM_DIVERGENCE_MARGIN_MIN),
        response_signature_enabled: bool = True,
        manifest_v10_disabled: bool = False,
        allow_response_signature_collapse_abstain: bool = True,
        response_text_diversity_min: float = (
            W40_DEFAULT_RESPONSE_TEXT_DIVERSITY_MIN),
        min_response_signature_probes: int = (
            W40_DEFAULT_MIN_RESPONSE_SIGNATURE_PROBES),
) -> dict[str, Any]:
    schema = _stable_schema_capsule()
    n_repl = max(int(bank_replicates), (int(n_eval) + 3) // 4)
    scenarios = build_phase67_bank(
        bank="outside_resolves", n_replicates=n_repl,
        seed=bank_seed)
    scenarios = _interleave_by_family(scenarios, n_families=4)
    if len(scenarios) > n_eval:
        scenarios = scenarios[:n_eval]
    n_eval_actual = min(n_eval, len(scenarios))

    if bank == "trivial_w40":
        live_attestation_disabled = True
        trust_subspace_enabled = False
        manifest_v5_disabled = True
        host_diversity_enabled = False
        manifest_v6_disabled = True
        min_distinct_hosts = 1
        trajectory_enabled = False
        manifest_v7_disabled = True
        allow_single_host_trajectory_reroute = False
        consensus_enabled = False
        manifest_v8_disabled = True
        allow_consensus_reference_divergence_abstain = False
        quorum_enabled = False
        manifest_v9_disabled = True
        allow_disjoint_quorum_divergence_abstain = False
        response_signature_enabled = False
        manifest_v10_disabled = True
        allow_response_signature_collapse_abstain = False

    K1 = max(1, int(_PHASE87_K1_FRACTION * n_eval_actual))
    consensus_top_per_cell: dict[int, tuple[str, ...]] = {}
    quorum_top_per_cell: dict[
        int, tuple[tuple[str, ...], ...]] = {}
    response_signature_per_cell: dict[
        int, tuple[tuple[str, str], ...]] = {}
    bench_gold_per_cell: dict[int, tuple[str, ...]] = {}

    for cell_idx in range(n_eval_actual):
        scenario = scenarios[cell_idx]
        scenario_gold = tuple(sorted(_gold_for_cell(scenario)))
        registrations = _build_w40_oracle_registrations_for_cell(
            bank=bank, cell_idx=cell_idx, n_total=n_eval_actual)
        per_rounds = _scenario_to_per_round_handoffs(scenario)
        w21 = _build_w21_disambiguator(
            T_decoder=T_decoder, registrations=registrations,
            quorum_min=quorum_min)
        w21_out = w21.decode_rounds(per_rounds)
        w21_top = tuple(
            sorted(set(str(s) for s in w21_out.get("services", ()))))
        if not w21_top:
            w21_top = scenario_gold

        if bank == "trivial_w40":
            consensus_top_per_cell[cell_idx] = ()
            quorum_top_per_cell[cell_idx] = ()
            response_signature_per_cell[cell_idx] = ()
            bench_gold_per_cell[cell_idx] = scenario_gold
            continue

        # Decoy: a stable family-specific "true" pair that the W37
        # trajectory hosts collude against in the recovery half.
        family_idx = (cell_idx - K1) % 4
        decoys = [
            ("payments_disjoint", "billing_disjoint"),
            ("api_disjoint", "frontend_disjoint"),
            ("storage_disjoint", "queue_disjoint"),
            ("db_query_disjoint", "search_disjoint"),
        ]
        true_recovery_top = tuple(sorted(decoys[family_idx]))

        if bank == "no_regression_diverse_agrees":
            # Consensus AGREES with W21; W39 quorum AGREES; W40
            # quorum members carry naturally diverse response bytes.
            # W40 ratifies (no regression).
            consensus_top_per_cell[cell_idx] = w21_top
            quorum_top_per_cell[cell_idx] = (w21_top, w21_top)
            response_signature_per_cell[cell_idx] = (
                ("mac_off_cluster_a",
                 "primary db slowdown observed at edge boundary"),
                ("mac_off_cluster_b",
                 "queue lag amplified across worker shards"),
            )
            bench_gold_per_cell[cell_idx] = w21_top
        elif bank == "response_signature_collapse":
            if cell_idx < K1:
                # Prefix half: trajectory + consensus + quorum +
                # response heterogeneity all agree honestly.
                consensus_top_per_cell[cell_idx] = w21_top
                quorum_top_per_cell[cell_idx] = (w21_top, w21_top)
                response_signature_per_cell[cell_idx] = (
                    ("mac_off_cluster_a",
                     "alpha bravo charlie delta echo"),
                    ("mac_off_cluster_b",
                     "foxtrot golf hotel india juliet"),
                )
                bench_gold_per_cell[cell_idx] = w21_top
            else:
                # Recovery half: W37 collude + W38 collude + W39
                # quorum collude (all K members emit w21_top wrong
                # top_set) AND every W40 quorum member emits
                # IDENTICAL wrong response bytes (the "naive" full-
                # quorum-collusion attack with byte-identical
                # response payloads).  W40 detects the collapse and
                # abstains via RESPONSE_SIGNATURE_COLLAPSE.
                consensus_top_per_cell[cell_idx] = w21_top
                quorum_top_per_cell[cell_idx] = (w21_top, w21_top)
                identical_response = (
                    "wrong consensus pinned to incident root cause")
                response_signature_per_cell[cell_idx] = (
                    ("mac_off_cluster_a", identical_response),
                    ("mac_off_cluster_b", identical_response),
                )
                bench_gold_per_cell[cell_idx] = true_recovery_top
        elif bank == "coordinated_diverse_response":
            if cell_idx < K1:
                consensus_top_per_cell[cell_idx] = w21_top
                quorum_top_per_cell[cell_idx] = (w21_top, w21_top)
                response_signature_per_cell[cell_idx] = (
                    ("mac_off_cluster_a",
                     "alpha bravo charlie delta echo"),
                    ("mac_off_cluster_b",
                     "foxtrot golf hotel india juliet"),
                )
                bench_gold_per_cell[cell_idx] = w21_top
            else:
                # Recovery half: W37 collude + W38 collude + W39
                # quorum collude (W39 ratifies wrong top_set).
                # The W40 member probes carry DIFFERENT response
                # text bytes that nevertheless all encode the same
                # wrong top_set (the "smart" attacker who
                # diversifies response text while keeping the
                # same wrong top_set).  W40 cannot recover at the
                # capsule layer (W40-L-COORDINATED-DIVERSE-RESPONSE
                # -CAP fires).
                consensus_top_per_cell[cell_idx] = w21_top
                quorum_top_per_cell[cell_idx] = (w21_top, w21_top)
                response_signature_per_cell[cell_idx] = (
                    ("mac_off_cluster_a",
                     "primary db slowdown observed at edge boundary "
                     "for the wrong consensus root cause we picked"),
                    ("mac_off_cluster_b",
                     "queue lag amplified across worker shards "
                     "associated with the wrong consensus root pick"),
                )
                bench_gold_per_cell[cell_idx] = true_recovery_top
        elif bank == "insufficient_response_signature":
            if cell_idx < K1:
                consensus_top_per_cell[cell_idx] = w21_top
                quorum_top_per_cell[cell_idx] = (w21_top, w21_top)
                response_signature_per_cell[cell_idx] = (
                    ("mac_off_cluster_a",
                     "alpha bravo charlie delta echo"),
                    ("mac_off_cluster_b",
                     "foxtrot golf hotel india juliet"),
                )
                bench_gold_per_cell[cell_idx] = w21_top
            else:
                # Only ONE W40 member probe is provided in the
                # recovery half (below
                # min_response_signature_probes=2).  W40 reduces to
                # W39 via RESPONSE_SIGNATURE_INSUFFICIENT.  W39
                # itself is in the W39-L-FULL-DISJOINT-QUORUM-
                # COLLUSION-CAP regime here, so this is also a
                # falsifier where W40 cannot help.
                consensus_top_per_cell[cell_idx] = w21_top
                quorum_top_per_cell[cell_idx] = (w21_top, w21_top)
                identical_response = (
                    "wrong consensus pinned to incident root cause")
                response_signature_per_cell[cell_idx] = (
                    ("mac_off_cluster_a", identical_response),
                )
                bench_gold_per_cell[cell_idx] = true_recovery_top
        else:
            consensus_top_per_cell[cell_idx] = ()
            quorum_top_per_cell[cell_idx] = ()
            response_signature_per_cell[cell_idx] = ()
            bench_gold_per_cell[cell_idx] = scenario_gold

    w36 = _make_w36(
        schema=schema, T_decoder=T_decoder, bank=bank,
        n_eval=n_eval_actual, anchor_ids=anchor_oracle_ids,
        quorum_min=quorum_min,
        live_attestation_disabled=live_attestation_disabled,
        trust_subspace_enabled=trust_subspace_enabled,
        manifest_v5_disabled=manifest_v5_disabled,
        host_diversity_enabled=host_diversity_enabled,
        manifest_v6_disabled=manifest_v6_disabled,
        min_distinct_hosts=min_distinct_hosts)
    w37 = _make_w37(
        schema=schema, T_decoder=T_decoder, bank=bank,
        n_eval=n_eval_actual, anchor_ids=anchor_oracle_ids,
        quorum_min=quorum_min,
        live_attestation_disabled=live_attestation_disabled,
        trust_subspace_enabled=trust_subspace_enabled,
        manifest_v5_disabled=manifest_v5_disabled,
        host_diversity_enabled=host_diversity_enabled,
        manifest_v6_disabled=manifest_v6_disabled,
        min_distinct_hosts=min_distinct_hosts,
        trajectory_enabled=trajectory_enabled,
        manifest_v7_disabled=manifest_v7_disabled,
        allow_single_host_trajectory_reroute=(
            allow_single_host_trajectory_reroute),
        trajectory_threshold=trajectory_threshold,
        trajectory_margin_min=trajectory_margin_min,
        min_anchored_observations=min_anchored_observations,
        min_trajectory_anchored_hosts=min_trajectory_anchored_hosts,
        registered_anchor_host_ids=registered_anchor_host_ids)
    w38 = _make_w38(
        schema=schema, T_decoder=T_decoder, bank=bank,
        n_eval=n_eval_actual, anchor_ids=anchor_oracle_ids,
        quorum_min=quorum_min,
        live_attestation_disabled=live_attestation_disabled,
        trust_subspace_enabled=trust_subspace_enabled,
        manifest_v5_disabled=manifest_v5_disabled,
        host_diversity_enabled=host_diversity_enabled,
        manifest_v6_disabled=manifest_v6_disabled,
        min_distinct_hosts=min_distinct_hosts,
        trajectory_enabled=trajectory_enabled,
        manifest_v7_disabled=manifest_v7_disabled,
        allow_single_host_trajectory_reroute=(
            allow_single_host_trajectory_reroute),
        trajectory_threshold=trajectory_threshold,
        trajectory_margin_min=trajectory_margin_min,
        min_anchored_observations=min_anchored_observations,
        min_trajectory_anchored_hosts=min_trajectory_anchored_hosts,
        registered_anchor_host_ids=registered_anchor_host_ids,
        consensus_enabled=consensus_enabled,
        manifest_v8_disabled=manifest_v8_disabled,
        allow_consensus_reference_divergence_abstain=(
            allow_consensus_reference_divergence_abstain),
        consensus_strength_min=consensus_strength_min,
        divergence_margin_min=divergence_margin_min,
        consensus_top_per_cell=consensus_top_per_cell)
    w39 = _make_w39(
        schema=schema, T_decoder=T_decoder, bank=bank,
        n_eval=n_eval_actual, anchor_ids=anchor_oracle_ids,
        quorum_min=quorum_min,
        live_attestation_disabled=live_attestation_disabled,
        trust_subspace_enabled=trust_subspace_enabled,
        manifest_v5_disabled=manifest_v5_disabled,
        host_diversity_enabled=host_diversity_enabled,
        manifest_v6_disabled=manifest_v6_disabled,
        min_distinct_hosts=min_distinct_hosts,
        trajectory_enabled=trajectory_enabled,
        manifest_v7_disabled=manifest_v7_disabled,
        allow_single_host_trajectory_reroute=(
            allow_single_host_trajectory_reroute),
        trajectory_threshold=trajectory_threshold,
        trajectory_margin_min=trajectory_margin_min,
        min_anchored_observations=min_anchored_observations,
        min_trajectory_anchored_hosts=min_trajectory_anchored_hosts,
        registered_anchor_host_ids=registered_anchor_host_ids,
        consensus_enabled=consensus_enabled,
        manifest_v8_disabled=manifest_v8_disabled,
        allow_consensus_reference_divergence_abstain=(
            allow_consensus_reference_divergence_abstain),
        consensus_strength_min=consensus_strength_min,
        divergence_margin_min=divergence_margin_min,
        consensus_top_per_cell=consensus_top_per_cell,
        quorum_enabled=quorum_enabled,
        manifest_v9_disabled=manifest_v9_disabled,
        allow_disjoint_quorum_divergence_abstain=(
            allow_disjoint_quorum_divergence_abstain),
        w39_quorum_min=w39_quorum_min,
        w39_min_quorum_probes=w39_min_quorum_probes,
        w39_consensus_strength_min=w39_consensus_strength_min,
        w39_divergence_margin_min=w39_divergence_margin_min,
        quorum_top_per_cell=quorum_top_per_cell)
    w40 = _make_w40(
        schema=schema, T_decoder=T_decoder, bank=bank,
        n_eval=n_eval_actual, anchor_ids=anchor_oracle_ids,
        quorum_min=quorum_min,
        live_attestation_disabled=live_attestation_disabled,
        trust_subspace_enabled=trust_subspace_enabled,
        manifest_v5_disabled=manifest_v5_disabled,
        host_diversity_enabled=host_diversity_enabled,
        manifest_v6_disabled=manifest_v6_disabled,
        min_distinct_hosts=min_distinct_hosts,
        trajectory_enabled=trajectory_enabled,
        manifest_v7_disabled=manifest_v7_disabled,
        allow_single_host_trajectory_reroute=(
            allow_single_host_trajectory_reroute),
        trajectory_threshold=trajectory_threshold,
        trajectory_margin_min=trajectory_margin_min,
        min_anchored_observations=min_anchored_observations,
        min_trajectory_anchored_hosts=min_trajectory_anchored_hosts,
        registered_anchor_host_ids=registered_anchor_host_ids,
        consensus_enabled=consensus_enabled,
        manifest_v8_disabled=manifest_v8_disabled,
        allow_consensus_reference_divergence_abstain=(
            allow_consensus_reference_divergence_abstain),
        consensus_strength_min=consensus_strength_min,
        divergence_margin_min=divergence_margin_min,
        consensus_top_per_cell=consensus_top_per_cell,
        quorum_enabled=quorum_enabled,
        manifest_v9_disabled=manifest_v9_disabled,
        allow_disjoint_quorum_divergence_abstain=(
            allow_disjoint_quorum_divergence_abstain),
        w39_quorum_min=w39_quorum_min,
        w39_min_quorum_probes=w39_min_quorum_probes,
        w39_consensus_strength_min=w39_consensus_strength_min,
        w39_divergence_margin_min=w39_divergence_margin_min,
        quorum_top_per_cell=quorum_top_per_cell,
        response_signature_enabled=response_signature_enabled,
        manifest_v10_disabled=manifest_v10_disabled,
        allow_response_signature_collapse_abstain=(
            allow_response_signature_collapse_abstain),
        response_text_diversity_min=response_text_diversity_min,
        min_response_signature_probes=(
            min_response_signature_probes),
        response_signature_per_cell=response_signature_per_cell)

    records: list[_Phase87Record] = []
    for cell_idx in range(n_eval_actual):
        scenario = scenarios[cell_idx]
        gold = bench_gold_per_cell.get(
            cell_idx, _gold_for_cell(scenario))
        per_rounds = _scenario_to_per_round_handoffs(scenario)
        registrations = _build_w40_oracle_registrations_for_cell(
            bank=bank, cell_idx=cell_idx, n_total=n_eval_actual)

        substrate_fifo = AttentionAwareBundleDecoder(T_decoder=T_decoder)
        substrate_fifo_out = substrate_fifo.decode_rounds(per_rounds)

        w21 = _build_w21_disambiguator(
            T_decoder=T_decoder, registrations=registrations,
            quorum_min=quorum_min)
        w21_out = w21.decode_rounds(per_rounds)

        w36.inner.inner.inner.inner = _build_w21_disambiguator(
            T_decoder=T_decoder, registrations=registrations,
            quorum_min=quorum_min)
        w36_out = w36.decode_rounds(per_rounds)

        w37.inner.inner.inner.inner.inner = _build_w21_disambiguator(
            T_decoder=T_decoder, registrations=registrations,
            quorum_min=quorum_min)
        w37_out = w37.decode_rounds(per_rounds)
        w37_result = w37.last_result

        w38.inner.inner.inner.inner.inner.inner = (
            _build_w21_disambiguator(
                T_decoder=T_decoder, registrations=registrations,
                quorum_min=quorum_min))
        w38_out = w38.decode_rounds(per_rounds)
        w38_result = w38.last_result

        w39.inner.inner.inner.inner.inner.inner.inner = (
            _build_w21_disambiguator(
                T_decoder=T_decoder, registrations=registrations,
                quorum_min=quorum_min))
        w39_out = w39.decode_rounds(per_rounds)
        w39_result = w39.last_result

        w40.inner.inner.inner.inner.inner.inner.inner.inner = (
            _build_w21_disambiguator(
                T_decoder=T_decoder, registrations=registrations,
                quorum_min=quorum_min))
        w40_out = w40.decode_rounds(per_rounds)
        w40_result = w40.last_result

        rat_fifo = _is_ratified(substrate_fifo_out)
        rat21 = _is_ratified(w21_out)
        rat36 = _is_ratified(w36_out)
        rat37 = _is_ratified(w37_out)
        rat38 = _is_ratified(w38_out)
        rat39 = _is_ratified(w39_out)
        rat40 = _is_ratified(w40_out)
        records.append(_Phase87Record(
            cell_idx=int(cell_idx), expected=tuple(gold),
            correct_substrate_fifo=(
                _is_correct(substrate_fifo_out, gold)
                if rat_fifo else False),
            correct_w21=_is_correct(w21_out, gold) if rat21 else False,
            correct_w36=_is_correct(w36_out, gold) if rat36 else False,
            correct_w37=_is_correct(w37_out, gold) if rat37 else False,
            correct_w38=_is_correct(w38_out, gold) if rat38 else False,
            correct_w39=_is_correct(w39_out, gold) if rat39 else False,
            correct_w40=_is_correct(w40_out, gold) if rat40 else False,
            ratified_substrate_fifo=bool(rat_fifo),
            ratified_w21=bool(rat21),
            ratified_w36=bool(rat36),
            ratified_w37=bool(rat37),
            ratified_w38=bool(rat38),
            ratified_w39=bool(rat39),
            ratified_w40=bool(rat40),
            w37_decoder_branch=(
                str(w37_result.decoder_branch)
                if w37_result is not None else ""),
            w38_decoder_branch=(
                str(w38_result.decoder_branch)
                if w38_result is not None else ""),
            w39_decoder_branch=(
                str(w39_result.decoder_branch)
                if w39_result is not None else ""),
            w40_decoder_branch=(
                str(w40_result.decoder_branch)
                if w40_result is not None else ""),
            w40_projection_branch=(
                str(w40_result.projection_branch)
                if w40_result is not None else ""),
            w39_decision_top_set=(
                tuple(w40_result.w39_decision_top_set)
                if w40_result is not None else ()),
            w40_n_diverse_pairs=int(
                w40_result.n_diverse_pairs
                if w40_result is not None else 0),
            w40_n_collapse_pairs=int(
                w40_result.n_collapse_pairs
                if w40_result is not None else 0),
            w40_n_pairs=int(w40_result.n_pairs
                            if w40_result is not None else 0),
            w40_n_total=int(w40_result.n_total
                            if w40_result is not None else 0),
            w40_mean_pairwise_jaccard=float(
                w40_result.mean_pairwise_jaccard
                if w40_result is not None else 0.0),
            w39_visible=int(w39_result.n_w39_visible_tokens
                            if w39_result is not None else 0),
            w40_visible=int(w40_result.n_w40_visible_tokens
                            if w40_result is not None else 0),
            w40_overhead=int(w40_result.n_w40_overhead_tokens
                             if w40_result is not None else 0),
            w40_structured_bits=int(
                w40_result.n_structured_bits
                if w40_result is not None else 0),
            w40_cram_factor=float(
                w40_result.cram_factor_w40
                if w40_result is not None else 0.0),
        ))

    def _rate(attr: str) -> float:
        return (sum(1 for r in records if getattr(r, attr))
                / len(records) if records else 0.0)

    def _trust(correct_attr: str, rat_attr: str) -> float:
        n_rat = sum(1 for r in records if getattr(r, rat_attr))
        if n_rat == 0:
            return 1.0
        return sum(1 for r in records
                   if getattr(r, correct_attr)) / n_rat

    total_w40_overhead = sum(r.w40_overhead for r in records)
    total_w40_bits = sum(r.w40_structured_bits for r in records)
    n_w40_diverse = sum(
        1 for r in records
        if r.w40_decoder_branch
            == W40_BRANCH_RESPONSE_SIGNATURE_DIVERSE)
    n_w40_collapse = sum(
        1 for r in records
        if r.w40_decoder_branch
            == W40_BRANCH_RESPONSE_SIGNATURE_COLLAPSE_ABSTAINED)
    n_w40_no_references = sum(
        1 for r in records
        if r.w40_decoder_branch
            == W40_BRANCH_RESPONSE_SIGNATURE_NO_REFERENCES)
    n_w40_no_trigger = sum(
        1 for r in records
        if r.w40_decoder_branch
            == W40_BRANCH_RESPONSE_SIGNATURE_NO_TRIGGER)
    n_w40_insufficient = sum(
        1 for r in records
        if r.w40_decoder_branch
            == W40_BRANCH_RESPONSE_SIGNATURE_INSUFFICIENT)
    n_w40_incomplete = sum(
        1 for r in records
        if r.w40_decoder_branch
            == W40_BRANCH_RESPONSE_SIGNATURE_INCOMPLETE)
    n_w40_trivial = sum(
        1 for r in records
        if r.w40_decoder_branch
            == W40_BRANCH_TRIVIAL_RESPONSE_SIGNATURE_PASSTHROUGH)

    return {
        "bank": str(bank),
        "n_eval": len(records),
        "bank_seed": int(bank_seed),
        "anchor_oracle_ids": list(anchor_oracle_ids),
        "live_attestation_disabled": bool(live_attestation_disabled),
        "trajectory_enabled": bool(trajectory_enabled),
        "consensus_enabled": bool(consensus_enabled),
        "quorum_enabled": bool(quorum_enabled),
        "response_signature_enabled": bool(
            response_signature_enabled),
        "allow_response_signature_collapse_abstain": bool(
            allow_response_signature_collapse_abstain),
        "response_text_diversity_min": float(
            response_text_diversity_min),
        "min_response_signature_probes": int(
            min_response_signature_probes),
        "correctness_ratified_rate_substrate_fifo": round(
            _rate("correct_substrate_fifo"), 4),
        "correctness_ratified_rate_w21": round(
            _rate("correct_w21"), 4),
        "correctness_ratified_rate_w36": round(
            _rate("correct_w36"), 4),
        "correctness_ratified_rate_w37": round(
            _rate("correct_w37"), 4),
        "correctness_ratified_rate_w38": round(
            _rate("correct_w38"), 4),
        "correctness_ratified_rate_w39": round(
            _rate("correct_w39"), 4),
        "correctness_ratified_rate_w40": round(
            _rate("correct_w40"), 4),
        "trust_precision_substrate_fifo": round(
            _trust("correct_substrate_fifo",
                   "ratified_substrate_fifo"), 4),
        "trust_precision_w21": round(
            _trust("correct_w21", "ratified_w21"), 4),
        "trust_precision_w36": round(
            _trust("correct_w36", "ratified_w36"), 4),
        "trust_precision_w37": round(
            _trust("correct_w37", "ratified_w37"), 4),
        "trust_precision_w38": round(
            _trust("correct_w38", "ratified_w38"), 4),
        "trust_precision_w39": round(
            _trust("correct_w39", "ratified_w39"), 4),
        "trust_precision_w40": round(
            _trust("correct_w40", "ratified_w40"), 4),
        "delta_correctness_w40_w39": round(
            _rate("correct_w40") - _rate("correct_w39"), 4),
        "delta_trust_precision_w40_w39": round(
            _trust("correct_w40", "ratified_w40")
            - _trust("correct_w39", "ratified_w39"), 4),
        "n_w40_diverse": int(n_w40_diverse),
        "n_w40_collapse_abstained": int(n_w40_collapse),
        "n_w40_no_references": int(n_w40_no_references),
        "n_w40_no_trigger": int(n_w40_no_trigger),
        "n_w40_insufficient": int(n_w40_insufficient),
        "n_w40_incomplete": int(n_w40_incomplete),
        "n_w40_trivial": int(n_w40_trivial),
        "mean_total_w39_visible_tokens": (
            round(sum(r.w39_visible for r in records) / len(records),
                  4)
            if records else 0.0),
        "mean_total_w40_visible_tokens": (
            round(sum(r.w40_visible for r in records) / len(records),
                  4)
            if records else 0.0),
        "mean_overhead_w40_per_cell": (
            round(total_w40_overhead / len(records), 4)
            if records else 0.0),
        "max_overhead_w40_per_cell": max(
            (r.w40_overhead for r in records), default=0),
        "total_w40_structured_bits": int(total_w40_bits),
        "structured_state_transferred_per_visible_token": round(
            total_w40_bits / max(1, total_w40_overhead), 4),
        "mean_w40_cram_factor": (
            round(sum(r.w40_cram_factor for r in records)
                  / len(records), 4) if records else 0.0),
        "byte_equivalent_w40_w39": bool(
            all(r.w40_visible == r.w39_visible for r in records)
            and _rate("correct_w40") == _rate("correct_w39")
            and _trust("correct_w40", "ratified_w40")
            == _trust("correct_w39", "ratified_w39")),
        "records": [dataclasses.asdict(r) for r in records],
    }


def run_phase87_seed_sweep(
        *,
        bank: str = "response_signature_collapse",
        n_eval: int = 16,
        seeds: Sequence[int] = (11, 17, 23, 29, 31),
        **kwargs: Any,
) -> dict[str, Any]:
    seed_results: list[dict[str, Any]] = []
    for seed in seeds:
        result = run_phase87(
            bank=bank, n_eval=n_eval, bank_seed=int(seed), **kwargs)
        short = {k: v for k, v in result.items() if k != "records"}
        short["seed"] = int(seed)
        seed_results.append(short)
    return {
        "bank": str(bank),
        "seeds": [int(s) for s in seeds],
        "min_delta_correctness_w40_w39": round(
            min(s["delta_correctness_w40_w39"]
                for s in seed_results)
            if seed_results else 0.0, 4),
        "max_delta_correctness_w40_w39": round(
            max(s["delta_correctness_w40_w39"]
                for s in seed_results)
            if seed_results else 0.0, 4),
        "min_delta_trust_precision_w40_w39": round(
            min(s["delta_trust_precision_w40_w39"]
                for s in seed_results)
            if seed_results else 0.0, 4),
        "max_delta_trust_precision_w40_w39": round(
            max(s["delta_trust_precision_w40_w39"]
                for s in seed_results)
            if seed_results else 0.0, 4),
        "min_trust_precision_w40": round(
            min(s["trust_precision_w40"] for s in seed_results)
            if seed_results else 1.0, 4),
        "max_overhead_w40_per_cell": max(
            (int(s["max_overhead_w40_per_cell"])
             for s in seed_results), default=0),
        "all_byte_equivalent_w40_w39": all(
            bool(s["byte_equivalent_w40_w39"]) for s in seed_results)
            if seed_results else False,
        "seed_results": seed_results,
    }


def _write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _cli() -> int:
    parser = argparse.ArgumentParser(
        prog="phase87",
        description=("Phase 87 -- W40 cross-host response-signature "
                     "heterogeneity ratification."))
    parser.add_argument(
        "--bank", default="response_signature_collapse",
        choices=("trivial_w40",
                 "response_signature_collapse",
                 "no_regression_diverse_agrees",
                 "coordinated_diverse_response",
                 "insufficient_response_signature"))
    parser.add_argument("--n-eval", type=int, default=16)
    parser.add_argument("--bank-seed", type=int, default=11)
    parser.add_argument("--seed-sweep", action="store_true")
    parser.add_argument("--seeds", type=str, default="11,17,23,29,31")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()
    seeds = tuple(int(s) for s in args.seeds.split(","))
    if args.seed_sweep:
        result = run_phase87_seed_sweep(
            bank=args.bank, n_eval=args.n_eval, seeds=seeds)
    else:
        result = run_phase87(
            bank=args.bank, n_eval=args.n_eval,
            bank_seed=args.bank_seed)
        result = {k: v for k, v in result.items() if k != "records"}
    if args.out:
        _write_json(args.out, result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
