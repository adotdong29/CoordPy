"""Phase 84 -- W37 anchor-cross-host basis-trajectory ratification.

W37 hardens W36 at the cross-cell, cross-host trajectory boundary.
W36 abstains whenever the current cell has fewer than
``min_distinct_hosts`` healthy attested hosts even if a remaining
single host has been independently anchored across earlier cells.
W37 maintains a closed-form per-(host_id, oracle_id, top_set) EWMA
over anchored historical observations and -- only when the current
cell's basis already includes the same trajectory-anchored top_set
on a healthy host -- converts a W36 abstention into a single-host
trajectory-trusted reroute.  Without an anchored trajectory it
preserves W36 behavior byte-for-byte.

This is still a capsule-layer audited proxy.  It does not read
transformer hidden states, transplant KV cache, or claim native
latent transfer.
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
    _DecodedHandoff,
)
from vision_mvp.experiments.phase67_outside_information import (
    build_phase67_bank,
)


# Phase84 banks divide a 16-cell budget into a prefix half and a
# recovery half.  In the prefix, all 3 oracles are healthy on three
# distinct hosts (mac1, mac_remote, mac_shadow) so the W37 trajectory
# accumulates cross-host anchored observations.  In the recovery half,
# the bank manipulates which hosts attest in each cell.
_PHASE84_K1_FRACTION = 1.0 / 2.0


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


def _build_w37_oracle_registrations_for_cell(
        *,
        bank: str,
        cell_idx: int,
        n_total: int,
) -> tuple[OracleRegistration, ...]:
    K1 = max(1, int(_PHASE84_K1_FRACTION * n_total))

    if bank in ("trivial_w37", "no_trajectory_history"):
        sg = ServiceGraphOracle(oracle_id="service_graph")
        ch = ChangeHistoryOracle(oracle_id="change_history")
        oc = OnCallNotesOracle(oracle_id="oncall_notes")
    elif bank == "single_host_trajectory_recover":
        # All cells: 3 honest oracles.  The recovery story is an
        # attestation/host story, not an oracle-honesty story.
        sg = ServiceGraphOracle(oracle_id="service_graph")
        ch = ChangeHistoryOracle(oracle_id="change_history")
        oc = OnCallNotesOracle(oracle_id="oncall_notes")
    elif bank == "trajectory_disagreement":
        # Prefix: 3 honest oracles, building a cross-host anchored
        # trajectory on the gold top_set.  Recovery: only mac1
        # attested AND service_graph compromised so its top_set
        # disagrees with the historically anchored top_set.  W37 must
        # detect disagreement (the two trusted top_sets) or fall
        # through to abstain.
        if cell_idx < K1:
            sg = ServiceGraphOracle(oracle_id="service_graph")
            ch = ChangeHistoryOracle(oracle_id="change_history")
            oc = OnCallNotesOracle(oracle_id="oncall_notes")
        else:
            sg = CompromisedServiceGraphOracle(oracle_id="service_graph")
            ch = CompromisedServiceGraphOracle(oracle_id="change_history")
            oc = CompromisedServiceGraphOracle(oracle_id="oncall_notes")
    elif bank == "poisoned_trajectory":
        # All cells: only mac1 (single host) attests an oracle whose
        # output is consistently wrong.  The trajectory builds up a
        # high *un-anchored* EWMA on a single host -- W37 must NOT
        # pick this up as trajectory-trusted because anchoring requires
        # cross-host attestation.
        sg = CompromisedServiceGraphOracle(oracle_id="service_graph")
        ch = CompromisedServiceGraphOracle(oracle_id="change_history")
        oc = CompromisedServiceGraphOracle(oracle_id="oncall_notes")
    else:
        raise ValueError(f"unknown phase84 bank {bank!r}")

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
        "mac_spoofed": HostRegistration(
            host_id="mac_spoofed", model_id="spoofed-oracle",
            base_url="mock://spoofed", preflight_ok=False),
    }


def _make_live_attestation_provider(*, bank: str, n_total: int):
    """Phase84 live-attestation provider.

    Trajectory regimes are *attestation* regimes; oracle outputs are
    determined by ``_build_w37_oracle_registrations_for_cell``.
    """
    K1 = max(1, int(_PHASE84_K1_FRACTION * n_total))

    def provider(orch: Any, _w33_result: Any) -> tuple[
            LiveOracleAttestation, ...]:
        cell_idx = int(getattr(orch, "_cell_index", 0))
        if bank == "trivial_w37":
            return ()
        if bank == "no_trajectory_history":
            # No trajectory history at all: every cell attests only
            # mac1.  W37 must fall back to W36 byte-for-byte.
            host_map = {
                "service_graph": ("mac1", "local-qwen", True),
                "change_history": ("mac1", "local-qwen", True),
                "oncall_notes": ("mac1", "local-qwen", True),
            }
        elif bank == "poisoned_trajectory":
            # Poisoned-single-host trajectory: every cell attests
            # only mac1.  Honest co-attestation is never observed; the
            # trajectory accumulates unanchored EWMA only.
            host_map = {
                "service_graph": ("mac1", "local-qwen", True),
                "change_history": ("mac1", "local-qwen", True),
                "oncall_notes": ("mac1", "local-qwen", True),
            }
        elif bank == "single_host_trajectory_recover":
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
                # Recovery cells: only mac1 attests.
                host_map = {
                    "service_graph": (
                        "mac1", "local-qwen", True),
                    "change_history": (
                        "mac1", "local-qwen", True),
                    "oncall_notes": (
                        "mac1", "local-qwen", True),
                }
        elif bank == "trajectory_disagreement":
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
                # Recovery cells: only mac1 attests; oracle outputs
                # are now compromised (different top_set), so the
                # current basis disagrees with the historically
                # anchored trajectory.
                host_map = {
                    "service_graph": (
                        "mac1", "local-qwen", True),
                    "change_history": (
                        "mac1", "local-qwen", True),
                    "oncall_notes": (
                        "mac1", "local-qwen", True),
                }
        else:
            raise ValueError(f"unknown phase84 bank {bank!r}")
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


def _gold_for_cell(scenario: Any) -> tuple[str, ...]:
    return tuple(sorted(scenario.gold_services_pair))


def _is_correct(answer: dict[str, Any], gold: tuple[str, ...]) -> bool:
    services = tuple(sorted(set(answer.get("services", ()))))
    return services == tuple(sorted(set(gold)))


def _is_ratified(answer: dict[str, Any]) -> bool:
    return bool(answer.get("services", ()))


@dataclasses.dataclass
class _Phase84Record:
    cell_idx: int
    expected: tuple[str, ...]
    correct_substrate_fifo: bool
    correct_w21: bool
    correct_w33: bool
    correct_w34: bool
    correct_w35: bool
    correct_w36: bool
    correct_w37: bool
    ratified_substrate_fifo: bool
    ratified_w21: bool
    ratified_w33: bool
    ratified_w34: bool
    ratified_w35: bool
    ratified_w36: bool
    ratified_w37: bool
    w36_decoder_branch: str
    w37_decoder_branch: str
    w37_projection_branch: str
    w37_supporting_host_ids: tuple[str, ...]
    w37_anchor_host_ids: tuple[str, ...]
    w36_visible: int
    w37_visible: int
    w37_overhead: int
    w37_structured_bits: int
    w37_cram_factor: float


def _make_w33(
        *,
        schema: Any,
        anchor_ids: tuple[str, ...],
        T_decoder: int | None,
        bank: str,
        n_eval: int,
        quorum_min: int,
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
        registrations=_build_w37_oracle_registrations_for_cell(
            bank=bank, cell_idx=0, n_total=n_eval),
        quorum_min=quorum_min,
    )
    return TrustEWMATrackedMultiOracleOrchestrator(
        inner=placeholder,
        registry=registry,
        enabled=True,
        require_w33_verification=True,
    )


def _make_w34(
        *,
        schema: Any,
        T_decoder: int | None,
        bank: str,
        n_eval: int,
        anchor_ids: tuple[str, ...],
        quorum_min: int,
        live_attestation_disabled: bool,
) -> LiveAwareMultiAnchorOrchestrator:
    w33 = _make_w33(
        schema=schema, anchor_ids=anchor_ids,
        T_decoder=T_decoder, bank=bank, n_eval=n_eval,
        quorum_min=quorum_min)
    w34_registry = build_live_aware_registry(
        schema=schema,
        inner_w33_registry=w33.registry,
        multi_anchor_quorum_min=W34_DEFAULT_ANCHOR_QUORUM_MIN,
        live_attestation_disabled=bool(live_attestation_disabled),
        manifest_v4_disabled=False,
        host_decay_factor=1.0,
        registered_hosts=_registered_hosts(),
    )
    orch = LiveAwareMultiAnchorOrchestrator(
        inner=w33, registry=w34_registry,
        enabled=True, require_w34_verification=True)
    if not live_attestation_disabled:
        orch.set_live_attestation_provider(
            _make_live_attestation_provider(bank=bank, n_total=n_eval))
    return orch


def _make_w35(
        *,
        schema: Any,
        T_decoder: int | None,
        bank: str,
        n_eval: int,
        anchor_ids: tuple[str, ...],
        quorum_min: int,
        live_attestation_disabled: bool,
        trust_subspace_enabled: bool,
        manifest_v5_disabled: bool,
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
                "service_graph", "change_history", "oncall_notes"),
        )
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
            abstain_on_unstable_consensus=True,
        )
    return TrustSubspaceDenseControlOrchestrator(
        inner=w34, registry=registry,
        enabled=True, require_w35_verification=True)


def _make_w36(
        *,
        schema: Any,
        T_decoder: int | None,
        bank: str,
        n_eval: int,
        anchor_ids: tuple[str, ...],
        quorum_min: int,
        live_attestation_disabled: bool,
        trust_subspace_enabled: bool,
        manifest_v5_disabled: bool,
        host_diversity_enabled: bool,
        manifest_v6_disabled: bool,
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
                "service_graph", "change_history", "oncall_notes"),
        )
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
            abstain_on_unverified_host_projection=True,
        )
    return HostDiverseTrustSubspaceOrchestrator(
        inner=w35, registry=registry,
        enabled=True, require_w36_verification=True)


def _make_w37(
        *,
        schema: Any,
        T_decoder: int | None,
        bank: str,
        n_eval: int,
        anchor_ids: tuple[str, ...],
        quorum_min: int,
        live_attestation_disabled: bool,
        trust_subspace_enabled: bool,
        manifest_v5_disabled: bool,
        host_diversity_enabled: bool,
        manifest_v6_disabled: bool,
        min_distinct_hosts: int,
        trajectory_enabled: bool,
        manifest_v7_disabled: bool,
        allow_single_host_trajectory_reroute: bool,
        trajectory_threshold: float,
        trajectory_margin_min: float,
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
                "service_graph", "change_history", "oncall_notes"),
        )
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
                min_trajectory_anchored_hosts),
        )
    return CrossHostBasisTrajectoryOrchestrator(
        inner=w36, registry=registry, enabled=True,
        require_w37_verification=True)


def _interleave_by_family(scenarios: list, n_families: int = 4) -> list:
    """Reorder a family-blocked phase67 bank so families interleave.

    ``build_phase67_bank`` emits scenarios in family-major order
    (family 0 reps, then family 1 reps, ...).  Phase84 needs prefix
    cells to cover the *same* gold pairs as recovery cells so the W37
    trajectory has matching top_set entries to look up.  Round-robin
    permutation by family index achieves that.
    """
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


def run_phase84(
        *,
        bank: str = "single_host_trajectory_recover",
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

    if bank == "trivial_w37":
        live_attestation_disabled = True
        trust_subspace_enabled = False
        manifest_v5_disabled = True
        host_diversity_enabled = False
        manifest_v6_disabled = True
        min_distinct_hosts = 1
        trajectory_enabled = False
        manifest_v7_disabled = True
        allow_single_host_trajectory_reroute = False

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

    records: list[_Phase84Record] = []
    for cell_idx in range(n_eval_actual):
        scenario = scenarios[cell_idx]
        gold = _gold_for_cell(scenario)
        per_rounds = _scenario_to_per_round_handoffs(scenario)
        registrations = _build_w37_oracle_registrations_for_cell(
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

        rat_fifo = _is_ratified(substrate_fifo_out)
        rat21 = _is_ratified(w21_out)
        rat33 = _is_ratified(w33_out)
        rat34 = _is_ratified(w34_out)
        rat35 = _is_ratified(w35_out)
        rat36 = _is_ratified(w36_out)
        rat37 = _is_ratified(w37_out)
        records.append(_Phase84Record(
            cell_idx=int(cell_idx),
            expected=tuple(gold),
            correct_substrate_fifo=(
                _is_correct(substrate_fifo_out, gold) if rat_fifo
                else False),
            correct_w21=_is_correct(w21_out, gold) if rat21 else False,
            correct_w33=_is_correct(w33_out, gold) if rat33 else False,
            correct_w34=_is_correct(w34_out, gold) if rat34 else False,
            correct_w35=_is_correct(w35_out, gold) if rat35 else False,
            correct_w36=_is_correct(w36_out, gold) if rat36 else False,
            correct_w37=_is_correct(w37_out, gold) if rat37 else False,
            ratified_substrate_fifo=bool(rat_fifo),
            ratified_w21=bool(rat21),
            ratified_w33=bool(rat33),
            ratified_w34=bool(rat34),
            ratified_w35=bool(rat35),
            ratified_w36=bool(rat36),
            ratified_w37=bool(rat37),
            w36_decoder_branch=(
                str(w36_result.decoder_branch)
                if w36_result is not None else ""),
            w37_decoder_branch=(
                str(w37_result.decoder_branch)
                if w37_result is not None else ""),
            w37_projection_branch=(
                str(w37_result.projection_branch)
                if w37_result is not None else ""),
            w37_supporting_host_ids=(
                tuple(w37_result.supporting_host_ids)
                if w37_result is not None else ()),
            w37_anchor_host_ids=(
                tuple(w37_result.trajectory_anchor_host_ids)
                if w37_result is not None else ()),
            w36_visible=int(w36_result.n_w36_visible_tokens
                            if w36_result is not None else 0),
            w37_visible=int(w37_result.n_w37_visible_tokens
                            if w37_result is not None else 0),
            w37_overhead=int(w37_result.n_w37_overhead_tokens
                             if w37_result is not None else 0),
            w37_structured_bits=int(w37_result.n_structured_bits
                                    if w37_result is not None else 0),
            w37_cram_factor=float(w37_result.cram_factor_w37
                                  if w37_result is not None else 0.0),
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

    total_w37_overhead = sum(r.w37_overhead for r in records)
    total_w37_bits = sum(r.w37_structured_bits for r in records)
    n_w37_rerouted = sum(
        1 for r in records
        if r.w37_decoder_branch == W37_BRANCH_TRAJECTORY_REROUTED)
    n_w37_abstained = sum(
        1 for r in records
        if r.w37_decoder_branch == W37_BRANCH_TRAJECTORY_ABSTAINED)
    n_w37_disagreement = sum(
        1 for r in records
        if r.w37_decoder_branch == W37_BRANCH_TRAJECTORY_DISAGREEMENT)
    n_w36_abstained_w37_recovered = sum(
        1 for r in records
        if r.w36_decoder_branch == W36_BRANCH_HOST_DIVERSE_ABSTAINED
        and r.w37_decoder_branch == W37_BRANCH_TRAJECTORY_REROUTED
        and r.correct_w37)
    return {
        "bank": str(bank),
        "n_eval": len(records),
        "bank_seed": int(bank_seed),
        "anchor_oracle_ids": list(anchor_oracle_ids),
        "live_attestation_disabled": bool(live_attestation_disabled),
        "trajectory_enabled": bool(trajectory_enabled),
        "allow_single_host_trajectory_reroute": bool(
            allow_single_host_trajectory_reroute),
        "min_anchored_observations": int(min_anchored_observations),
        "min_trajectory_anchored_hosts": int(
            min_trajectory_anchored_hosts),
        "trajectory_threshold": float(trajectory_threshold),
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
        "delta_correctness_w37_w36": round(
            _rate("correct_w37") - _rate("correct_w36"), 4),
        "delta_trust_precision_w37_w36": round(
            _trust("correct_w37", "ratified_w37")
            - _trust("correct_w36", "ratified_w36"), 4),
        "n_w37_trajectory_rerouted": int(n_w37_rerouted),
        "n_w37_trajectory_abstained": int(n_w37_abstained),
        "n_w37_trajectory_disagreement": int(n_w37_disagreement),
        "n_w36_abstained_w37_recovered": int(
            n_w36_abstained_w37_recovered),
        "mean_total_w36_visible_tokens": (
            round(sum(r.w36_visible for r in records) / len(records), 4)
            if records else 0.0),
        "mean_total_w37_visible_tokens": (
            round(sum(r.w37_visible for r in records) / len(records), 4)
            if records else 0.0),
        "mean_overhead_w37_per_cell": (
            round(total_w37_overhead / len(records), 4)
            if records else 0.0),
        "max_overhead_w37_per_cell": max(
            (r.w37_overhead for r in records), default=0),
        "total_w37_structured_bits": int(total_w37_bits),
        "structured_state_transferred_per_visible_token": round(
            total_w37_bits / max(1, total_w37_overhead), 4),
        "mean_w37_cram_factor": (
            round(sum(r.w37_cram_factor for r in records)
                  / len(records), 4) if records else 0.0),
        "byte_equivalent_w37_w36": bool(
            all(r.w37_visible == r.w36_visible for r in records)
            and _rate("correct_w37") == _rate("correct_w36")
            and _trust("correct_w37", "ratified_w37")
            == _trust("correct_w36", "ratified_w36")),
        "records": [dataclasses.asdict(r) for r in records],
    }


def run_phase84_seed_sweep(
        *,
        bank: str = "single_host_trajectory_recover",
        n_eval: int = 16,
        seeds: Sequence[int] = (11, 17, 23, 29, 31),
        **kwargs: Any,
) -> dict[str, Any]:
    seed_results: list[dict[str, Any]] = []
    for seed in seeds:
        result = run_phase84(
            bank=bank, n_eval=n_eval, bank_seed=int(seed), **kwargs)
        short = {k: v for k, v in result.items() if k != "records"}
        short["seed"] = int(seed)
        seed_results.append(short)
    return {
        "bank": str(bank),
        "seeds": [int(s) for s in seeds],
        "min_delta_correctness_w37_w36": round(
            min(s["delta_correctness_w37_w36"]
                for s in seed_results)
            if seed_results else 0.0, 4),
        "max_delta_correctness_w37_w36": round(
            max(s["delta_correctness_w37_w36"]
                for s in seed_results)
            if seed_results else 0.0, 4),
        "min_delta_trust_precision_w37_w36": round(
            min(s["delta_trust_precision_w37_w36"]
                for s in seed_results)
            if seed_results else 0.0, 4),
        "min_trust_precision_w37": round(
            min(s["trust_precision_w37"] for s in seed_results)
            if seed_results else 1.0, 4),
        "max_overhead_w37_per_cell": max(
            (int(s["max_overhead_w37_per_cell"])
             for s in seed_results), default=0),
        "all_byte_equivalent_w37_w36": all(
            bool(s["byte_equivalent_w37_w36"]) for s in seed_results)
            if seed_results else False,
        "seed_results": seed_results,
    }


def _write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _cli() -> int:
    parser = argparse.ArgumentParser(
        prog="phase84",
        description=("Phase 84 -- W37 anchor-cross-host basis-"
                     "trajectory ratification."))
    parser.add_argument(
        "--bank", default="single_host_trajectory_recover",
        choices=("trivial_w37",
                 "single_host_trajectory_recover",
                 "trajectory_disagreement",
                 "no_trajectory_history",
                 "poisoned_trajectory"))
    parser.add_argument("--n-eval", type=int, default=16)
    parser.add_argument("--bank-seed", type=int, default=11)
    parser.add_argument("--seed-sweep", action="store_true")
    parser.add_argument("--seeds", type=str, default="11,17,23,29,31")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()
    seeds = tuple(int(s) for s in args.seeds.split(","))
    if args.seed_sweep:
        result = run_phase84_seed_sweep(
            bank=args.bank, n_eval=args.n_eval, seeds=seeds)
    else:
        result = run_phase84(
            bank=args.bank, n_eval=args.n_eval,
            bank_seed=args.bank_seed)
        result = {k: v for k, v in result.items() if k != "records"}
    if args.out:
        _write_json(args.out, result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
