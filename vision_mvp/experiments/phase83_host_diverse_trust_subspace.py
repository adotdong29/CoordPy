"""Phase 83 — host-diverse trust-subspace guard (SDK v3.37, W36).

W36 hardens W35 at the remaining host/live trust boundary.  W35 can
only reason over capsule-visible basis directions; if every direction
moves wrong together, W35 cannot recover.  W36 adds a typed host-
diversity envelope and refuses to ratify dense-control projections
whose support is not independently host-attested.

This is still a capsule-layer proxy.  It does not read transformer
hidden states, transplant KV cache, or claim native latent transfer.
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
    compute_response_feature_signature,
    _DecodedHandoff,
)
from vision_mvp.experiments.phase67_outside_information import (
    build_phase67_bank,
)


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


def _build_w36_oracle_registrations_for_cell(
        *,
        bank: str,
        cell_idx: int,
        n_total: int,
) -> tuple[OracleRegistration, ...]:
    K1 = max(1, (3 * n_total) // 8)
    K2 = max(K1 + 1, (5 * n_total) // 8)

    if bank in ("trivial_w36", "no_live_attestation"):
        sg = ServiceGraphOracle(oracle_id="service_graph")
        ch = ChangeHistoryOracle(oracle_id="change_history")
        oc = OnCallNotesOracle(oracle_id="oncall_notes")
    elif bank == "host_diverse_recover":
        if cell_idx < K1:
            sg = ServiceGraphOracle(oracle_id="service_graph")
            ch = ChangeHistoryOracle(oracle_id="change_history")
            oc = OnCallNotesOracle(oracle_id="oncall_notes")
        elif cell_idx < K2:
            sg = ServiceGraphOracle(oracle_id="service_graph")
            ch = ChangeHistoryOracle(oracle_id="change_history")
            oc = CompromisedServiceGraphOracle(oracle_id="oncall_notes")
        else:
            sg = CompromisedServiceGraphOracle(oracle_id="service_graph")
            ch = ChangeHistoryOracle(oracle_id="change_history")
            oc = OnCallNotesOracle(oracle_id="oncall_notes")
    elif bank == "host_spoofed_consensus":
        if cell_idx < K1:
            sg = ServiceGraphOracle(oracle_id="service_graph")
            ch = ChangeHistoryOracle(oracle_id="change_history")
            oc = OnCallNotesOracle(oracle_id="oncall_notes")
        elif cell_idx < K2:
            sg = ServiceGraphOracle(oracle_id="service_graph")
            ch = ChangeHistoryOracle(oracle_id="change_history")
            oc = CompromisedServiceGraphOracle(oracle_id="oncall_notes")
        else:
            sg = CompromisedServiceGraphOracle(oracle_id="service_graph")
            ch = CompromisedServiceGraphOracle(oracle_id="change_history")
            oc = CompromisedServiceGraphOracle(oracle_id="oncall_notes")
    else:
        raise ValueError(f"unknown phase83 bank {bank!r}")

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
    K2 = max(max(1, (3 * n_total) // 8) + 1, (5 * n_total) // 8)

    def provider(orch: Any, _w33_result: Any) -> tuple[
            LiveOracleAttestation, ...]:
        cell_idx = int(getattr(orch, "_cell_index", 0))
        if bank == "no_live_attestation":
            return ()
        if bank == "host_spoofed_consensus" and cell_idx >= K2:
            host_map = {
                "service_graph": ("mac_spoofed", "spoofed-oracle", False),
                "change_history": ("mac_spoofed", "spoofed-oracle", False),
                "oncall_notes": ("mac_spoofed", "spoofed-oracle", False),
            }
        else:
            host_map = {
                "service_graph": ("mac1", "local-qwen", True),
                "change_history": ("mac_remote", "remote-qwen14b", True),
                "oncall_notes": ("mac_shadow", "shadow-adjudicator", True),
            }
        out: list[LiveOracleAttestation] = []
        for oracle_id in ("service_graph", "change_history", "oncall_notes"):
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
class _Phase83Record:
    cell_idx: int
    expected: tuple[str, ...]
    correct_substrate_fifo: bool
    correct_w21: bool
    correct_w33: bool
    correct_w34: bool
    correct_w35: bool
    correct_w36: bool
    ratified_substrate_fifo: bool
    ratified_w21: bool
    ratified_w33: bool
    ratified_w34: bool
    ratified_w35: bool
    ratified_w36: bool
    w35_decoder_branch: str
    w36_decoder_branch: str
    w36_projection_branch: str
    w36_supporting_host_ids: tuple[str, ...]
    w35_visible: int
    w36_visible: int
    w36_overhead: int
    w36_structured_bits: int
    w36_cram_factor: float


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
        registrations=_build_w36_oracle_registrations_for_cell(
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
        schema=schema,
        anchor_ids=anchor_ids,
        T_decoder=T_decoder,
        bank=bank,
        n_eval=n_eval,
        quorum_min=quorum_min,
    )
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
        inner=w33,
        registry=w34_registry,
        enabled=True,
        require_w34_verification=True,
    )
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
        schema=schema,
        T_decoder=T_decoder,
        bank=bank,
        n_eval=n_eval,
        anchor_ids=anchor_ids,
        quorum_min=quorum_min,
        live_attestation_disabled=live_attestation_disabled,
    )
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
        inner=w34,
        registry=registry,
        enabled=True,
        require_w35_verification=True,
    )


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
        schema=schema,
        T_decoder=T_decoder,
        bank=bank,
        n_eval=n_eval,
        anchor_ids=anchor_ids,
        quorum_min=quorum_min,
        live_attestation_disabled=live_attestation_disabled,
        trust_subspace_enabled=trust_subspace_enabled,
        manifest_v5_disabled=manifest_v5_disabled,
    )
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
            host_diversity_margin_min=W36_DEFAULT_HOST_DIVERSITY_MARGIN_MIN,
            abstain_on_unverified_host_projection=True,
        )
    return HostDiverseTrustSubspaceOrchestrator(
        inner=w35,
        registry=registry,
        enabled=True,
        require_w36_verification=True,
    )


def run_phase83(
        *,
        bank: str = "host_spoofed_consensus",
        n_eval: int = 16,
        bank_seed: int = 11,
        bank_replicates: int = 2,
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
) -> dict[str, Any]:
    schema = _stable_schema_capsule()
    n_repl = max(int(bank_replicates), (int(n_eval) + 3) // 4)
    scenarios = build_phase67_bank(
        bank="outside_resolves", n_replicates=n_repl,
        seed=bank_seed)
    if len(scenarios) > n_eval:
        scenarios = scenarios[:n_eval]
    n_eval_actual = min(n_eval, len(scenarios))

    if bank == "trivial_w36":
        live_attestation_disabled = True
        trust_subspace_enabled = False
        manifest_v5_disabled = True
        host_diversity_enabled = False
        manifest_v6_disabled = True
        min_distinct_hosts = 1
    if bank == "no_live_attestation":
        live_attestation_disabled = True

    w33 = _make_w33(
        schema=schema,
        anchor_ids=("service_graph",),
        T_decoder=T_decoder,
        bank=bank,
        n_eval=n_eval_actual,
        quorum_min=quorum_min,
    )
    w34 = _make_w34(
        schema=schema,
        T_decoder=T_decoder,
        bank=bank,
        n_eval=n_eval_actual,
        anchor_ids=anchor_oracle_ids,
        quorum_min=quorum_min,
        live_attestation_disabled=live_attestation_disabled,
    )
    w35 = _make_w35(
        schema=schema,
        T_decoder=T_decoder,
        bank=bank,
        n_eval=n_eval_actual,
        anchor_ids=anchor_oracle_ids,
        quorum_min=quorum_min,
        live_attestation_disabled=live_attestation_disabled,
        trust_subspace_enabled=trust_subspace_enabled,
        manifest_v5_disabled=manifest_v5_disabled,
    )
    w36 = _make_w36(
        schema=schema,
        T_decoder=T_decoder,
        bank=bank,
        n_eval=n_eval_actual,
        anchor_ids=anchor_oracle_ids,
        quorum_min=quorum_min,
        live_attestation_disabled=live_attestation_disabled,
        trust_subspace_enabled=trust_subspace_enabled,
        manifest_v5_disabled=manifest_v5_disabled,
        host_diversity_enabled=host_diversity_enabled,
        manifest_v6_disabled=manifest_v6_disabled,
        min_distinct_hosts=min_distinct_hosts,
    )

    records: list[_Phase83Record] = []
    for cell_idx in range(n_eval_actual):
        scenario = scenarios[cell_idx]
        gold = _gold_for_cell(scenario)
        per_rounds = _scenario_to_per_round_handoffs(scenario)
        registrations = _build_w36_oracle_registrations_for_cell(
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
        w35_result = w35.last_result

        w36.inner.inner.inner.inner = _build_w21_disambiguator(
            T_decoder=T_decoder, registrations=registrations,
            quorum_min=quorum_min)
        w36_out = w36.decode_rounds(per_rounds)
        w36_result = w36.last_result

        rat_fifo = _is_ratified(substrate_fifo_out)
        rat21 = _is_ratified(w21_out)
        rat33 = _is_ratified(w33_out)
        rat34 = _is_ratified(w34_out)
        rat35 = _is_ratified(w35_out)
        rat36 = _is_ratified(w36_out)
        records.append(_Phase83Record(
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
            ratified_substrate_fifo=bool(rat_fifo),
            ratified_w21=bool(rat21),
            ratified_w33=bool(rat33),
            ratified_w34=bool(rat34),
            ratified_w35=bool(rat35),
            ratified_w36=bool(rat36),
            w35_decoder_branch=(
                str(w35_result.decoder_branch)
                if w35_result is not None else ""),
            w36_decoder_branch=(
                str(w36_result.decoder_branch)
                if w36_result is not None else ""),
            w36_projection_branch=(
                str(w36_result.projection_branch)
                if w36_result is not None else ""),
            w36_supporting_host_ids=(
                tuple(w36_result.supporting_host_ids)
                if w36_result is not None else ()),
            w35_visible=int(w35_result.n_w35_visible_tokens
                            if w35_result is not None else 0),
            w36_visible=int(w36_result.n_w36_visible_tokens
                            if w36_result is not None else 0),
            w36_overhead=int(w36_result.n_w36_overhead_tokens
                             if w36_result is not None else 0),
            w36_structured_bits=int(w36_result.n_structured_bits
                                    if w36_result is not None else 0),
            w36_cram_factor=float(w36_result.cram_factor_w36
                                  if w36_result is not None else 0.0),
        ))

    def _rate(attr: str) -> float:
        return (sum(1 for r in records if getattr(r, attr)) / len(records)
                if records else 0.0)

    def _trust(correct_attr: str, rat_attr: str) -> float:
        n_rat = sum(1 for r in records if getattr(r, rat_attr))
        if n_rat == 0:
            return 1.0
        return sum(1 for r in records if getattr(r, correct_attr)) / n_rat

    total_w36_overhead = sum(r.w36_overhead for r in records)
    total_w36_bits = sum(r.w36_structured_bits for r in records)
    n_w36_abstained = sum(
        1 for r in records
        if r.w36_decoder_branch == W36_BRANCH_HOST_DIVERSE_ABSTAINED)
    n_w36_rerouted = sum(
        1 for r in records
        if r.w36_decoder_branch == W36_BRANCH_HOST_DIVERSE_REROUTED)
    n_w35_wrong_ratified_w36_abstained = sum(
        1 for r in records
        if r.ratified_w35 and not r.correct_w35
        and r.w36_decoder_branch == W36_BRANCH_HOST_DIVERSE_ABSTAINED)
    return {
        "bank": str(bank),
        "n_eval": len(records),
        "bank_seed": int(bank_seed),
        "anchor_oracle_ids": list(anchor_oracle_ids),
        "live_attestation_disabled": bool(live_attestation_disabled),
        "host_diversity_enabled": bool(host_diversity_enabled),
        "min_distinct_hosts": int(min_distinct_hosts),
        "correctness_ratified_rate_substrate_fifo": round(
            _rate("correct_substrate_fifo"), 4),
        "correctness_ratified_rate_w21": round(_rate("correct_w21"), 4),
        "correctness_ratified_rate_w33": round(_rate("correct_w33"), 4),
        "correctness_ratified_rate_w34": round(_rate("correct_w34"), 4),
        "correctness_ratified_rate_w35": round(_rate("correct_w35"), 4),
        "correctness_ratified_rate_w36": round(_rate("correct_w36"), 4),
        "trust_precision_substrate_fifo": round(
            _trust("correct_substrate_fifo", "ratified_substrate_fifo"),
            4),
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
        "delta_correctness_w36_w35": round(
            _rate("correct_w36") - _rate("correct_w35"), 4),
        "delta_trust_precision_w36_w35": round(
            _trust("correct_w36", "ratified_w36")
            - _trust("correct_w35", "ratified_w35"), 4),
        "n_w36_host_diverse_rerouted": int(n_w36_rerouted),
        "n_w36_host_diverse_abstained": int(n_w36_abstained),
        "n_w35_wrong_ratified_w36_abstained": int(
            n_w35_wrong_ratified_w36_abstained),
        "mean_total_w35_visible_tokens": round(
            sum(r.w35_visible for r in records) / len(records), 4)
            if records else 0.0,
        "mean_total_w36_visible_tokens": round(
            sum(r.w36_visible for r in records) / len(records), 4)
            if records else 0.0,
        "mean_overhead_w36_per_cell": round(
            total_w36_overhead / len(records), 4) if records else 0.0,
        "max_overhead_w36_per_cell": max(
            (r.w36_overhead for r in records), default=0),
        "total_w36_structured_bits": int(total_w36_bits),
        "structured_state_transferred_per_visible_token": round(
            total_w36_bits / max(1, total_w36_overhead), 4),
        "mean_w36_cram_factor": round(
            sum(r.w36_cram_factor for r in records) / len(records), 4)
            if records else 0.0,
        "byte_equivalent_w36_w35": bool(
            all(r.w36_visible == r.w35_visible for r in records)
            and _rate("correct_w36") == _rate("correct_w35")
            and _trust("correct_w36", "ratified_w36")
            == _trust("correct_w35", "ratified_w35")),
        "records": [dataclasses.asdict(r) for r in records],
    }


def run_phase83_seed_sweep(
        *,
        bank: str = "host_spoofed_consensus",
        n_eval: int = 16,
        seeds: Sequence[int] = (11, 17, 23, 29, 31),
        **kwargs: Any,
) -> dict[str, Any]:
    seed_results: list[dict[str, Any]] = []
    for seed in seeds:
        result = run_phase83(
            bank=bank, n_eval=n_eval, bank_seed=int(seed), **kwargs)
        short = {k: v for k, v in result.items() if k != "records"}
        short["seed"] = int(seed)
        seed_results.append(short)
    return {
        "bank": str(bank),
        "seeds": [int(s) for s in seeds],
        "min_delta_trust_precision_w36_w35": round(
            min(s["delta_trust_precision_w36_w35"] for s in seed_results)
            if seed_results else 0.0, 4),
        "min_trust_precision_w36": round(
            min(s["trust_precision_w36"] for s in seed_results)
            if seed_results else 1.0, 4),
        "max_delta_correctness_w36_w35": round(
            max(s["delta_correctness_w36_w35"] for s in seed_results)
            if seed_results else 0.0, 4),
        "min_delta_correctness_w36_w35": round(
            min(s["delta_correctness_w36_w35"] for s in seed_results)
            if seed_results else 0.0, 4),
        "max_overhead_w36_per_cell": max(
            (int(s["max_overhead_w36_per_cell"])
             for s in seed_results), default=0),
        "all_byte_equivalent_w36_w35": all(
            bool(s["byte_equivalent_w36_w35"]) for s in seed_results)
            if seed_results else False,
        "seed_results": seed_results,
    }


def _write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _cli() -> int:
    parser = argparse.ArgumentParser(
        prog="phase83",
        description="Phase 83 — W36 host-diverse trust-subspace guard.")
    parser.add_argument("--bank", default="host_spoofed_consensus",
                        choices=("trivial_w36",
                                 "host_diverse_recover",
                                 "host_spoofed_consensus",
                                 "no_live_attestation"))
    parser.add_argument("--n-eval", type=int, default=16)
    parser.add_argument("--bank-seed", type=int, default=11)
    parser.add_argument("--seed-sweep", action="store_true")
    parser.add_argument("--seeds", type=str, default="11,17,23,29,31")
    parser.add_argument("--min-distinct-hosts", type=int,
                        default=W36_DEFAULT_MIN_DISTINCT_HOSTS)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()
    seeds = tuple(int(s) for s in args.seeds.split(","))
    kwargs = dict(min_distinct_hosts=int(args.min_distinct_hosts))
    if args.seed_sweep:
        result = run_phase83_seed_sweep(
            bank=args.bank, n_eval=args.n_eval, seeds=seeds, **kwargs)
    else:
        result = run_phase83(
            bank=args.bank, n_eval=args.n_eval,
            bank_seed=args.bank_seed, **kwargs)
        result = {k: v for k, v in result.items() if k != "records"}
    if args.out:
        _write_json(args.out, result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
