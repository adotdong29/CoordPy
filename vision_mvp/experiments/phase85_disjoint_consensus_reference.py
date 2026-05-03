"""Phase 85 -- W38 disjoint cross-source consensus-reference adjudication.

W38 wraps W37's anchor-cross-host basis-trajectory ratification with a
disjoint cross-source consensus reference.  W37's deepest open wall is
W37-L-MULTI-HOST-COLLUSION-CAP: when two registered hosts simultaneously
emit a coordinated wrong top_set across enough cells, the trajectory
crosses the anchored thresholds and W37 can be made to reroute on the
wrong top_set.  W38 cross-checks the W37 candidate top_set against a
controller-pre-registered consensus reference probe whose host topology
is mechanically disjoint from W37's trajectory hosts; if the W37
candidate disagrees with the disjoint consensus reference within
``divergence_margin_min``, W38 abstains.

This is still a capsule-layer audited proxy.  It does not read
transformer hidden states, transplant KV cache, or claim native latent
transfer.  It does not close W37-L-MULTI-HOST-COLLUSION-CAP; it bounds
it by raising the adversary bar from "compromise 2 of N trajectory
hosts" to "compromise 2 of N trajectory hosts AND the disjoint
registered consensus reference".  The new W38-L-CONSENSUS-COLLUSION-CAP
limitation theorem records the residual cap.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from typing import Any, Sequence

from vision_mvp.coordpy.team_coord import (
    AttentionAwareBundleDecoder,
    BundleContradictionDisambiguator,
    OracleRegistration,
    ServiceGraphOracle,
    ChangeHistoryOracle,
    OnCallNotesOracle,
    CompromisedServiceGraphOracle,
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
    W36_BRANCH_HOST_DIVERSE_ABSTAINED,
    W36_BRANCH_HOST_DIVERSE_REROUTED,
    CrossHostBasisTrajectoryOrchestrator,
    build_trivial_cross_host_trajectory_registry,
    build_cross_host_trajectory_registry,
    W37_DEFAULT_TRAJECTORY_EWMA_ALPHA,
    W37_DEFAULT_TRAJECTORY_THRESHOLD,
    W37_DEFAULT_TRAJECTORY_MARGIN_MIN,
    W37_DEFAULT_MIN_ANCHORED_OBSERVATIONS,
    W37_DEFAULT_MIN_TRAJECTORY_ANCHORED_HOSTS,
    W37_BRANCH_TRAJECTORY_REROUTED,
    W37_BRANCH_TRAJECTORY_ABSTAINED,
    W37_BRANCH_TRAJECTORY_DISAGREEMENT,
    W37_BRANCH_TRAJECTORY_NO_HISTORY,
    W37_BRANCH_TRAJECTORY_NO_TRIGGER,
    W37_BRANCH_TRAJECTORY_REJECTED,
    W37_BRANCH_TRAJECTORY_RESOLVED,
    W37_BRANCH_TRIVIAL_TRAJECTORY_PASSTHROUGH,
    compute_response_feature_signature,
    DisjointConsensusReferenceOrchestrator,
    ConsensusReferenceProbe,
    build_trivial_disjoint_consensus_registry,
    build_disjoint_consensus_registry,
    W38_DEFAULT_CONSENSUS_STRENGTH_MIN,
    W38_DEFAULT_DIVERGENCE_MARGIN_MIN,
    W38_BRANCH_CONSENSUS_RATIFIED,
    W38_BRANCH_CONSENSUS_DIVERGENCE_ABSTAINED,
    W38_BRANCH_CONSENSUS_NO_REFERENCE,
    W38_BRANCH_CONSENSUS_NO_TRIGGER,
    W38_BRANCH_CONSENSUS_REFERENCE_WEAK,
    W38_BRANCH_TRIVIAL_CONSENSUS_PASSTHROUGH,
    W38_BRANCH_CONSENSUS_DISABLED,
    _DecodedHandoff,
)
from vision_mvp.experiments.phase67_outside_information import (
    build_phase67_bank,
)


# Phase 85 banks. Each bank exercises a different W38 contract.
#
#   * trivial_w38                       -- byte-for-W37 passthrough
#   * colluded_cross_host_trajectory    -- 2 of 3 trajectory hosts collude
#                                          on a wrong top_set; disjoint
#                                          consensus reference detects
#                                          divergence
#   * no_collusion_consensus_agrees     -- single-host recovery regime
#                                          where consensus reference and
#                                          W37 candidate agree
#   * consensus_also_compromised        -- W38-L-CONSENSUS-COLLUSION-CAP
#                                          limitation theorem in action
#   * no_consensus_reference            -- consensus probe is None;
#                                          W38 must reduce to W37
_PHASE85_K1_FRACTION = 1.0 / 2.0


def _stable_schema_capsule():
    from vision_mvp.coordpy.team_coord import (
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


def _build_w38_oracle_registrations_for_cell(
        *,
        bank: str,
        cell_idx: int,
        n_total: int,
) -> tuple[OracleRegistration, ...]:
    """Phase85 oracle registrations.

    All Phase85 banks use HONEST oracles in the host topology.  The W38
    threat model is *meta-level*: the trajectory hosts (mac1, mac_remote)
    are assumed to have been colluding off-path on the recovery cells,
    even though their on-path oracle outputs are identical to the
    prefix.  The disjoint consensus reference (registered separately on
    ``mac_consensus``) is the independent source that *knows* the
    recovery-half answer is different.  The bench correctness function
    measures against the consensus-reference top_set on every cell, so
    the W37 trajectory-anchored answer is *correct in the prefix*
    (consensus agrees) and *wrong in the recovery half* (consensus
    disagrees).  W38 must catch the recovery-half divergence and abstain.
    """
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
    """Phase85 live-attestation provider.

    All non-trivial banks use the R-84-style 3-hosts-prefix /
    1-host-recovery topology so the W37 trajectory accumulates
    cross-host anchored observations in the prefix half and reroutes
    on the historically anchored top_set in the recovery half.  The
    bank-specific behavior comes from the disjoint *consensus
    reference* probe, which either agrees with (no_collusion_consensus_
    agrees), disagrees with (colluded_cross_host_trajectory), or is
    itself compromised against (consensus_also_compromised) the W37
    reroute.
    """
    K1 = max(1, int(_PHASE85_K1_FRACTION * n_total))

    def provider(orch: Any, _w33_result: Any) -> tuple[
            LiveOracleAttestation, ...]:
        cell_idx = int(getattr(orch, "_cell_index", 0))
        if bank == "trivial_w38":
            return ()
        if cell_idx < K1:
            # Prefix half: mac1 + mac_remote + mac_shadow each attest
            # one oracle ⇒ W37 trajectory accumulates cross-host
            # anchored observations.
            host_map = {
                "service_graph": (
                    "mac1", "local-qwen", True),
                "change_history": (
                    "mac_remote", "remote-qwen14b", True),
                "oncall_notes": (
                    "mac_shadow", "shadow-adjudicator", True),
            }
        else:
            # Recovery half: only mac1 attests ⇒ W36 abstains on host
            # diversity; W37 single-host reroute is the load-bearing
            # path; W38 cross-checks the disjoint consensus reference.
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
    """Phase85 disjoint consensus reference probe provider.

    The consensus reference is a controller-pre-registered probe sourced
    from a disjoint registered host topology (``mac_consensus``).  It is
    NOT a runtime ground-truth oracle: it is an audited capsule-layer
    artefact whose host topology is mechanically disjoint from the W37
    trajectory hosts (mac1, mac_remote, mac_shadow).

    Per-cell consensus top_set is pre-computed by ``run_phase85`` and
    threaded through ``consensus_top_per_cell`` so the bank-specific
    behavior is auditable end-to-end.
    """

    def provider(orch: Any, w37_result: Any) -> (
            ConsensusReferenceProbe | None):
        cell_idx = int(getattr(orch, "_cell_index", 0))
        if bank in ("trivial_w38", "no_consensus_reference"):
            return None
        top = consensus_top_per_cell.get(cell_idx, ())
        if not top:
            # No registered probe for this cell ⇒ NO_REFERENCE branch.
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


def _gold_for_cell(scenario: Any) -> tuple[str, ...]:
    return tuple(sorted(scenario.gold_services_pair))


def _is_correct(answer: dict[str, Any], gold: tuple[str, ...]) -> bool:
    services = tuple(sorted(set(answer.get("services", ()))))
    return services == tuple(sorted(set(gold)))


def _is_ratified(answer: dict[str, Any]) -> bool:
    return bool(answer.get("services", ()))


@dataclasses.dataclass
class _Phase85Record:
    cell_idx: int
    expected: tuple[str, ...]
    correct_substrate_fifo: bool
    correct_w21: bool
    correct_w33: bool
    correct_w34: bool
    correct_w35: bool
    correct_w36: bool
    correct_w37: bool
    correct_w38: bool
    ratified_substrate_fifo: bool
    ratified_w21: bool
    ratified_w33: bool
    ratified_w34: bool
    ratified_w35: bool
    ratified_w36: bool
    ratified_w37: bool
    ratified_w38: bool
    w36_decoder_branch: str
    w37_decoder_branch: str
    w38_decoder_branch: str
    w38_projection_branch: str
    w37_top_set: tuple[str, ...]
    w38_consensus_top_set: tuple[str, ...]
    w38_divergence_score: float
    w37_visible: int
    w38_visible: int
    w38_overhead: int
    w38_structured_bits: int
    w38_cram_factor: float


def _make_w33(
        *, schema: Any, anchor_ids: tuple[str, ...],
        T_decoder: int | None, bank: str, n_eval: int, quorum_min: int,
) -> TrustEWMATrackedMultiOracleOrchestrator:
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
        registrations=_build_w38_oracle_registrations_for_cell(
            bank=bank, cell_idx=0, n_total=n_eval),
        quorum_min=quorum_min,
    )
    return TrustEWMATrackedMultiOracleOrchestrator(
        inner=placeholder, registry=registry,
        enabled=True, require_w33_verification=True)


def _make_w34(
        *, schema: Any, T_decoder: int | None, bank: str,
        n_eval: int, anchor_ids: tuple[str, ...], quorum_min: int,
        live_attestation_disabled: bool,
) -> LiveAwareMultiAnchorOrchestrator:
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


def _make_w35(
        *, schema: Any, T_decoder: int | None, bank: str,
        n_eval: int, anchor_ids: tuple[str, ...], quorum_min: int,
        live_attestation_disabled: bool,
        trust_subspace_enabled: bool, manifest_v5_disabled: bool,
) -> TrustSubspaceDenseControlOrchestrator:
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


def _make_w36(
        *, schema: Any, T_decoder: int | None, bank: str,
        n_eval: int, anchor_ids: tuple[str, ...], quorum_min: int,
        live_attestation_disabled: bool,
        trust_subspace_enabled: bool, manifest_v5_disabled: bool,
        host_diversity_enabled: bool, manifest_v6_disabled: bool,
        min_distinct_hosts: int,
) -> HostDiverseTrustSubspaceOrchestrator:
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


def _make_w37(
        *, schema: Any, T_decoder: int | None, bank: str,
        n_eval: int, anchor_ids: tuple[str, ...], quorum_min: int,
        live_attestation_disabled: bool,
        trust_subspace_enabled: bool, manifest_v5_disabled: bool,
        host_diversity_enabled: bool, manifest_v6_disabled: bool,
        min_distinct_hosts: int,
        trajectory_enabled: bool, manifest_v7_disabled: bool,
        allow_single_host_trajectory_reroute: bool,
        trajectory_threshold: float, trajectory_margin_min: float,
        min_anchored_observations: int,
        min_trajectory_anchored_hosts: int,
        registered_anchor_host_ids: tuple[str, ...],
) -> CrossHostBasisTrajectoryOrchestrator:
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
            registered_anchor_host_ids=tuple(registered_anchor_host_ids),
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


def _make_w38(
        *, schema: Any, T_decoder: int | None, bank: str,
        n_eval: int, anchor_ids: tuple[str, ...], quorum_min: int,
        live_attestation_disabled: bool,
        trust_subspace_enabled: bool, manifest_v5_disabled: bool,
        host_diversity_enabled: bool, manifest_v6_disabled: bool,
        min_distinct_hosts: int,
        trajectory_enabled: bool, manifest_v7_disabled: bool,
        allow_single_host_trajectory_reroute: bool,
        trajectory_threshold: float, trajectory_margin_min: float,
        min_anchored_observations: int,
        min_trajectory_anchored_hosts: int,
        registered_anchor_host_ids: tuple[str, ...],
        consensus_enabled: bool, manifest_v8_disabled: bool,
        allow_consensus_reference_divergence_abstain: bool,
        consensus_strength_min: float, divergence_margin_min: float,
        consensus_top_per_cell: dict[int, tuple[str, ...]],
) -> DisjointConsensusReferenceOrchestrator:
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
    if (consensus_enabled
            and bank not in ("trivial_w38", "no_consensus_reference")):
        orch.set_consensus_reference_provider(
            _make_consensus_reference_provider(
                bank=bank, n_total=n_eval,
                consensus_top_per_cell=consensus_top_per_cell))
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


def run_phase85(
        *,
        bank: str = "colluded_cross_host_trajectory",
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

    if bank == "trivial_w38":
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

    # Pre-compute the disjoint-consensus reference top_set per cell and
    # the bench's per-cell "true gold" -- both bank-driven.
    #
    # The W38 threat model treats the consensus reference as the
    # independent ground truth.  The bench therefore measures
    # correctness against the per-cell consensus top_set on every
    # bank where a consensus reference is registered.
    K1 = max(1, int(_PHASE85_K1_FRACTION * n_eval_actual))
    consensus_top_per_cell: dict[int, tuple[str, ...]] = {}
    bench_gold_per_cell: dict[int, tuple[str, ...]] = {}
    for cell_idx in range(n_eval_actual):
        scenario = scenarios[cell_idx]
        scenario_gold = tuple(sorted(_gold_for_cell(scenario)))
        # Compute the W21-quorum-resolved top_set on honest oracles
        # (this is what the W37 trajectory will anchor on).
        registrations = _build_w38_oracle_registrations_for_cell(
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
        if bank in ("trivial_w38", "no_consensus_reference"):
            consensus_top_per_cell[cell_idx] = ()
            bench_gold_per_cell[cell_idx] = scenario_gold
        elif bank == "no_collusion_consensus_agrees":
            # Consensus AGREES with the W37 reroute (same top_set).
            consensus_top_per_cell[cell_idx] = w21_top
            bench_gold_per_cell[cell_idx] = w21_top
        elif bank == "colluded_cross_host_trajectory":
            # Prefix half: consensus AGREES with the trajectory hosts
            # (no divergence).  Recovery half: the trajectory hosts are
            # COLLUDING (per the W38 threat-model meta-narrative); the
            # consensus reference is the disjoint independent source
            # and disagrees.  We model "consensus disagrees" by emitting
            # a bank-specific decoy top_set in recovery cells.  The
            # bench's per-cell "true gold" follows the consensus.
            if cell_idx < K1:
                consensus_top_per_cell[cell_idx] = w21_top
                bench_gold_per_cell[cell_idx] = w21_top
            else:
                # Decoy disjoint-consensus top_set: a stable family-
                # specific "true" pair the trajectory hosts collude
                # against.  Concrete payload doesn't matter for the
                # mechanism test; what matters is that it diverges
                # from w21_top by >= divergence_margin_min Jaccard.
                family_idx = (cell_idx - K1) % 4
                decoys = [
                    ("payments_disjoint", "billing_disjoint"),
                    ("api_disjoint", "frontend_disjoint"),
                    ("storage_disjoint", "queue_disjoint"),
                    ("db_query_disjoint", "search_disjoint"),
                ]
                consensus_top_per_cell[cell_idx] = tuple(
                    sorted(decoys[family_idx]))
                bench_gold_per_cell[cell_idx] = tuple(
                    sorted(decoys[family_idx]))
        elif bank == "consensus_also_compromised":
            # Consensus is COLLUDING with the trajectory hosts -- it
            # emits the SAME top_set the W37 trajectory rerouted to.
            # The bench's per-cell "true gold" is the disjoint decoy,
            # so both W37 and W38 commit to the wrong answer in the
            # recovery half (W38-L-CONSENSUS-COLLUSION-CAP fires).
            if cell_idx < K1:
                consensus_top_per_cell[cell_idx] = w21_top
                bench_gold_per_cell[cell_idx] = w21_top
            else:
                consensus_top_per_cell[cell_idx] = w21_top
                family_idx = (cell_idx - K1) % 4
                decoys = [
                    ("payments_disjoint", "billing_disjoint"),
                    ("api_disjoint", "frontend_disjoint"),
                    ("storage_disjoint", "queue_disjoint"),
                    ("db_query_disjoint", "search_disjoint"),
                ]
                bench_gold_per_cell[cell_idx] = tuple(
                    sorted(decoys[family_idx]))
        else:
            consensus_top_per_cell[cell_idx] = ()
            bench_gold_per_cell[cell_idx] = scenario_gold

    w33 = _make_w33(
        schema=schema, anchor_ids=("service_graph",),
        T_decoder=T_decoder, bank=bank, n_eval=n_eval_actual,
        quorum_min=quorum_min)
    w34 = _make_w34(
        schema=schema, T_decoder=T_decoder, bank=bank,
        n_eval=n_eval_actual, anchor_ids=anchor_oracle_ids,
        quorum_min=quorum_min,
        live_attestation_disabled=live_attestation_disabled)
    w35 = _make_w35(
        schema=schema, T_decoder=T_decoder, bank=bank,
        n_eval=n_eval_actual, anchor_ids=anchor_oracle_ids,
        quorum_min=quorum_min,
        live_attestation_disabled=live_attestation_disabled,
        trust_subspace_enabled=trust_subspace_enabled,
        manifest_v5_disabled=manifest_v5_disabled)
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

    records: list[_Phase85Record] = []
    for cell_idx in range(n_eval_actual):
        scenario = scenarios[cell_idx]
        gold = bench_gold_per_cell.get(
            cell_idx, _gold_for_cell(scenario))
        per_rounds = _scenario_to_per_round_handoffs(scenario)
        registrations = _build_w38_oracle_registrations_for_cell(
            bank=bank, cell_idx=cell_idx, n_total=n_eval_actual)

        substrate_fifo = AttentionAwareBundleDecoder(T_decoder=T_decoder)
        substrate_fifo_out = substrate_fifo.decode_rounds(per_rounds)

        w21 = _build_w21_disambiguator(
            T_decoder=T_decoder, registrations=registrations,
            quorum_min=quorum_min)
        w21_out = w21.decode_rounds(per_rounds)

        w33.inner = _build_w21_disambiguator(
            T_decoder=T_decoder, registrations=registrations,
            quorum_min=quorum_min)
        w33_out = w33.decode_rounds(per_rounds)

        w34.inner.inner = _build_w21_disambiguator(
            T_decoder=T_decoder, registrations=registrations,
            quorum_min=quorum_min)
        w34_out = w34.decode_rounds(per_rounds)

        w35.inner.inner.inner = _build_w21_disambiguator(
            T_decoder=T_decoder, registrations=registrations,
            quorum_min=quorum_min)
        w35_out = w35.decode_rounds(per_rounds)

        w36.inner.inner.inner.inner = _build_w21_disambiguator(
            T_decoder=T_decoder, registrations=registrations,
            quorum_min=quorum_min)
        w36_out = w36.decode_rounds(per_rounds)
        w36_result = w36.last_result

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

        rat_fifo = _is_ratified(substrate_fifo_out)
        rat21 = _is_ratified(w21_out)
        rat33 = _is_ratified(w33_out)
        rat34 = _is_ratified(w34_out)
        rat35 = _is_ratified(w35_out)
        rat36 = _is_ratified(w36_out)
        rat37 = _is_ratified(w37_out)
        rat38 = _is_ratified(w38_out)
        records.append(_Phase85Record(
            cell_idx=int(cell_idx), expected=tuple(gold),
            correct_substrate_fifo=(
                _is_correct(substrate_fifo_out, gold)
                if rat_fifo else False),
            correct_w21=_is_correct(w21_out, gold) if rat21 else False,
            correct_w33=_is_correct(w33_out, gold) if rat33 else False,
            correct_w34=_is_correct(w34_out, gold) if rat34 else False,
            correct_w35=_is_correct(w35_out, gold) if rat35 else False,
            correct_w36=_is_correct(w36_out, gold) if rat36 else False,
            correct_w37=_is_correct(w37_out, gold) if rat37 else False,
            correct_w38=_is_correct(w38_out, gold) if rat38 else False,
            ratified_substrate_fifo=bool(rat_fifo),
            ratified_w21=bool(rat21),
            ratified_w33=bool(rat33),
            ratified_w34=bool(rat34),
            ratified_w35=bool(rat35),
            ratified_w36=bool(rat36),
            ratified_w37=bool(rat37),
            ratified_w38=bool(rat38),
            w36_decoder_branch=(
                str(w36_result.decoder_branch)
                if w36_result is not None else ""),
            w37_decoder_branch=(
                str(w37_result.decoder_branch)
                if w37_result is not None else ""),
            w38_decoder_branch=(
                str(w38_result.decoder_branch)
                if w38_result is not None else ""),
            w38_projection_branch=(
                str(w38_result.projection_branch)
                if w38_result is not None else ""),
            w37_top_set=(
                tuple(w37_result.projection_top_set)
                if w37_result is not None else ()),
            w38_consensus_top_set=(
                tuple(w38_result.consensus_top_set)
                if w38_result is not None else ()),
            w38_divergence_score=float(
                w38_result.divergence_score
                if w38_result is not None else 0.0),
            w37_visible=int(w37_result.n_w37_visible_tokens
                            if w37_result is not None else 0),
            w38_visible=int(w38_result.n_w38_visible_tokens
                            if w38_result is not None else 0),
            w38_overhead=int(w38_result.n_w38_overhead_tokens
                             if w38_result is not None else 0),
            w38_structured_bits=int(
                w38_result.n_structured_bits
                if w38_result is not None else 0),
            w38_cram_factor=float(
                w38_result.cram_factor_w38
                if w38_result is not None else 0.0),
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

    total_w38_overhead = sum(r.w38_overhead for r in records)
    total_w38_bits = sum(r.w38_structured_bits for r in records)
    n_w38_divergence = sum(
        1 for r in records
        if r.w38_decoder_branch == W38_BRANCH_CONSENSUS_DIVERGENCE_ABSTAINED)
    n_w38_ratified = sum(
        1 for r in records
        if r.w38_decoder_branch == W38_BRANCH_CONSENSUS_RATIFIED)
    n_w38_no_reference = sum(
        1 for r in records
        if r.w38_decoder_branch == W38_BRANCH_CONSENSUS_NO_REFERENCE)
    return {
        "bank": str(bank),
        "n_eval": len(records),
        "bank_seed": int(bank_seed),
        "anchor_oracle_ids": list(anchor_oracle_ids),
        "live_attestation_disabled": bool(live_attestation_disabled),
        "trajectory_enabled": bool(trajectory_enabled),
        "consensus_enabled": bool(consensus_enabled),
        "allow_consensus_reference_divergence_abstain": bool(
            allow_consensus_reference_divergence_abstain),
        "consensus_strength_min": float(consensus_strength_min),
        "divergence_margin_min": float(divergence_margin_min),
        "correctness_ratified_rate_substrate_fifo": round(
            _rate("correct_substrate_fifo"), 4),
        "correctness_ratified_rate_w21": round(
            _rate("correct_w21"), 4),
        "correctness_ratified_rate_w33": round(
            _rate("correct_w33"), 4),
        "correctness_ratified_rate_w34": round(
            _rate("correct_w34"), 4),
        "correctness_ratified_rate_w35": round(
            _rate("correct_w35"), 4),
        "correctness_ratified_rate_w36": round(
            _rate("correct_w36"), 4),
        "correctness_ratified_rate_w37": round(
            _rate("correct_w37"), 4),
        "correctness_ratified_rate_w38": round(
            _rate("correct_w38"), 4),
        "trust_precision_substrate_fifo": round(
            _trust("correct_substrate_fifo",
                   "ratified_substrate_fifo"), 4),
        "trust_precision_w21": round(
            _trust("correct_w21", "ratified_w21"), 4),
        "trust_precision_w33": round(
            _trust("correct_w33", "ratified_w33"), 4),
        "trust_precision_w34": round(
            _trust("correct_w34", "ratified_w34"), 4),
        "trust_precision_w35": round(
            _trust("correct_w35", "ratified_w35"), 4),
        "trust_precision_w36": round(
            _trust("correct_w36", "ratified_w36"), 4),
        "trust_precision_w37": round(
            _trust("correct_w37", "ratified_w37"), 4),
        "trust_precision_w38": round(
            _trust("correct_w38", "ratified_w38"), 4),
        "delta_correctness_w38_w37": round(
            _rate("correct_w38") - _rate("correct_w37"), 4),
        "delta_trust_precision_w38_w37": round(
            _trust("correct_w38", "ratified_w38")
            - _trust("correct_w37", "ratified_w37"), 4),
        "n_w38_divergence_abstained": int(n_w38_divergence),
        "n_w38_ratified": int(n_w38_ratified),
        "n_w38_no_reference": int(n_w38_no_reference),
        "mean_total_w37_visible_tokens": (
            round(sum(r.w37_visible for r in records) / len(records), 4)
            if records else 0.0),
        "mean_total_w38_visible_tokens": (
            round(sum(r.w38_visible for r in records) / len(records), 4)
            if records else 0.0),
        "mean_overhead_w38_per_cell": (
            round(total_w38_overhead / len(records), 4)
            if records else 0.0),
        "max_overhead_w38_per_cell": max(
            (r.w38_overhead for r in records), default=0),
        "total_w38_structured_bits": int(total_w38_bits),
        "structured_state_transferred_per_visible_token": round(
            total_w38_bits / max(1, total_w38_overhead), 4),
        "mean_w38_cram_factor": (
            round(sum(r.w38_cram_factor for r in records)
                  / len(records), 4) if records else 0.0),
        "byte_equivalent_w38_w37": bool(
            all(r.w38_visible == r.w37_visible for r in records)
            and _rate("correct_w38") == _rate("correct_w37")
            and _trust("correct_w38", "ratified_w38")
            == _trust("correct_w37", "ratified_w37")),
        "records": [dataclasses.asdict(r) for r in records],
    }


def run_phase85_seed_sweep(
        *,
        bank: str = "colluded_cross_host_trajectory",
        n_eval: int = 16,
        seeds: Sequence[int] = (11, 17, 23, 29, 31),
        **kwargs: Any,
) -> dict[str, Any]:
    seed_results: list[dict[str, Any]] = []
    for seed in seeds:
        result = run_phase85(
            bank=bank, n_eval=n_eval, bank_seed=int(seed), **kwargs)
        short = {k: v for k, v in result.items() if k != "records"}
        short["seed"] = int(seed)
        seed_results.append(short)
    return {
        "bank": str(bank),
        "seeds": [int(s) for s in seeds],
        "min_delta_correctness_w38_w37": round(
            min(s["delta_correctness_w38_w37"]
                for s in seed_results)
            if seed_results else 0.0, 4),
        "max_delta_correctness_w38_w37": round(
            max(s["delta_correctness_w38_w37"]
                for s in seed_results)
            if seed_results else 0.0, 4),
        "min_delta_trust_precision_w38_w37": round(
            min(s["delta_trust_precision_w38_w37"]
                for s in seed_results)
            if seed_results else 0.0, 4),
        "max_delta_trust_precision_w38_w37": round(
            max(s["delta_trust_precision_w38_w37"]
                for s in seed_results)
            if seed_results else 0.0, 4),
        "min_trust_precision_w38": round(
            min(s["trust_precision_w38"] for s in seed_results)
            if seed_results else 1.0, 4),
        "max_overhead_w38_per_cell": max(
            (int(s["max_overhead_w38_per_cell"])
             for s in seed_results), default=0),
        "all_byte_equivalent_w38_w37": all(
            bool(s["byte_equivalent_w38_w37"]) for s in seed_results)
            if seed_results else False,
        "seed_results": seed_results,
    }


def _write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _cli() -> int:
    parser = argparse.ArgumentParser(
        prog="phase85",
        description=("Phase 85 -- W38 disjoint cross-source consensus-"
                     "reference adjudication."))
    parser.add_argument(
        "--bank", default="colluded_cross_host_trajectory",
        choices=("trivial_w38",
                 "colluded_cross_host_trajectory",
                 "no_collusion_consensus_agrees",
                 "consensus_also_compromised",
                 "no_consensus_reference"))
    parser.add_argument("--n-eval", type=int, default=16)
    parser.add_argument("--bank-seed", type=int, default=11)
    parser.add_argument("--seed-sweep", action="store_true")
    parser.add_argument("--seeds", type=str, default="11,17,23,29,31")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()
    seeds = tuple(int(s) for s in args.seeds.split(","))
    if args.seed_sweep:
        result = run_phase85_seed_sweep(
            bank=args.bank, n_eval=args.n_eval, seeds=seeds)
    else:
        result = run_phase85(
            bank=args.bank, n_eval=args.n_eval,
            bank_seed=args.bank_seed)
        result = {k: v for k, v in result.items() if k != "records"}
    if args.out:
        _write_json(args.out, result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
