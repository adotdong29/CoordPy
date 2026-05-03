"""Phase 88 -- W41 Integrated Multi-Agent Context Synthesis.

R-88 is the first benchmark in the Context Zero programme that
measures the *full integrated synthesis* over the strongest
explicit-capsule trust adjudication chain (W21..W40) AND the
strongest cross-role / multi-round bundle decoder family
(W7..W11) at the same time, under one manifest-v11 envelope.

The R-88 bench is a strict superset of R-87.  Every R-88 cell is
also a valid R-87 cell (the inner W40 chain is the same), but
R-88 *additionally* records the W41 cross-axis classification
(producer_axis_branch + trust_axis_branch + integrated_branch),
the integrated services, the manifest-v11 CID, and the
synthesis_audit / cross_axis_witness / state / decision CIDs.

Banks
-----

* ``trivial_w41`` -- the W40 inner stack is fully disabled
  (matches R-87 ``trivial_w40``); W41 must reduce to W40
  byte-for-byte (decoder branch
  ``trivial_integrated_passthrough``).
* ``both_axes`` -- W21 produces gold AND W40 ratifies via
  ``response_signature_diverse``; W41 ratifies via
  ``integrated_both_axes`` on the gold subset.  W41 trust
  precision and correctness must match W40 (no regression).
* ``trust_only_safety`` -- W21 produces a wrong (colluded)
  top_set; W40 detects collapse and abstains via
  ``response_signature_collapse_abstained``; W41 routes through
  the trust-side safety branch ``integrated_trust_only`` with
  empty integrated services.  W41 trust precision must equal W40
  trust precision (= 1.000 on the recovery half of the
  collapse bench).
* ``composite_collusion`` -- W21 produces wrong top_set; W40
  ratifies via ``response_signature_diverse`` because the K
  member probes were injected with diverse response bytes that
  encode the same wrong top_set in lock-step (W40-L-COORDINATED-
  DIVERSE-RESPONSE-CAP).  W41 cannot recover: the integrated
  branch is ``integrated_both_axes`` on the wrong set; this is
  the new W41-L-COMPOSITE-COLLUSION-CAP limitation theorem.
* ``insufficient_response_signature`` -- only one W40 member
  probe is provided; W40 reduces to W39 via INSUFFICIENT.  W41
  routes through ``integrated_producer_only`` (the W21 set is
  the producer-axis; the trust axis no-trigger), preserving W40
  byte-for-W39 semantics on the answer.

The bench reuses the entire W40 phase87 builder chain so the
inner W21..W40 decisions are identical byte-for-byte to R-87.
The only addition is the W41 wrapper (`IntegratedSynthesisOrch
estrator`) and the new R-88 record class which captures the
W41-specific measurement axes.

Honest scope: W41 is an integration-and-measurement milestone,
NOT a new transformer-internal mechanism.  The R-88 bench
does not claim correctness gains over W40 on the recovery half
of the collapse bench (W41 ties W40 there by design); it claims
*meta-level* gains: a single content-addressed envelope binding
both axes, a measured cross-axis branch distribution, and an
explicit integration-layer falsifier (W41-L-COMPOSITE-COLLUSION
-CAP) that names the residual wall.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from typing import Any, Sequence

from vision_mvp.experiments.phase87_cross_host_response_heterogeneity import (
    _build_w21_disambiguator,
    _build_w40_oracle_registrations_for_cell,
    _gold_for_cell,
    _is_correct,
    _is_ratified,
    _make_w40,
    _scenario_to_per_round_handoffs,
    _stable_schema_capsule,
    _interleave_by_family,
    _PHASE87_K1_FRACTION,
)
from vision_mvp.experiments.phase67_outside_information import (
    build_phase67_bank,
)
from vision_mvp.coordpy.team_coord import (
    AttentionAwareBundleDecoder,
    W36_DEFAULT_MIN_DISTINCT_HOSTS,
    W37_DEFAULT_TRAJECTORY_THRESHOLD,
    W37_DEFAULT_TRAJECTORY_MARGIN_MIN,
    W37_DEFAULT_MIN_ANCHORED_OBSERVATIONS,
    W37_DEFAULT_MIN_TRAJECTORY_ANCHORED_HOSTS,
    W38_DEFAULT_CONSENSUS_STRENGTH_MIN,
    W38_DEFAULT_DIVERGENCE_MARGIN_MIN,
    W39_DEFAULT_QUORUM_MIN,
    W39_DEFAULT_MIN_QUORUM_PROBES,
    W39_DEFAULT_QUORUM_STRENGTH_MIN,
    W39_DEFAULT_QUORUM_DIVERGENCE_MARGIN_MIN,
    W40_DEFAULT_RESPONSE_TEXT_DIVERSITY_MIN,
    W40_DEFAULT_MIN_RESPONSE_SIGNATURE_PROBES,
    W40_BRANCH_RESPONSE_SIGNATURE_DIVERSE,
    W40_BRANCH_RESPONSE_SIGNATURE_COLLAPSE_ABSTAINED,
    W40_BRANCH_RESPONSE_SIGNATURE_NO_TRIGGER,
    W40_BRANCH_RESPONSE_SIGNATURE_INSUFFICIENT,
)
from vision_mvp.coordpy.integrated_synthesis import (
    IntegratedSynthesisOrchestrator,
    build_integrated_synthesis_registry,
    build_trivial_integrated_synthesis_registry,
    W41_BRANCH_TRIVIAL_INTEGRATED_PASSTHROUGH,
    W41_BRANCH_INTEGRATED_BOTH_AXES,
    W41_BRANCH_INTEGRATED_TRUST_ONLY,
    W41_BRANCH_INTEGRATED_PRODUCER_ONLY,
    W41_BRANCH_INTEGRATED_AXES_DIVERGED_ABSTAINED,
    W41_BRANCH_INTEGRATED_NEITHER_AXIS,
    W41_BRANCH_INTEGRATED_DISABLED,
    W41_BRANCH_INTEGRATED_REJECTED,
    W41_PRODUCER_AXIS_FIRED,
    W41_PRODUCER_AXIS_NO_TRIGGER,
    W41_TRUST_AXIS_RATIFIED,
    W41_TRUST_AXIS_ABSTAINED,
    W41_TRUST_AXIS_NO_TRIGGER,
)


# Map R-88 bank name -> R-87 inner bank name + W41 wrapping config.
_R88_BANK_TO_R87 = {
    "trivial_w41": "trivial_w40",
    "both_axes": "no_regression_diverse_agrees",
    "trust_only_safety": "response_signature_collapse",
    "composite_collusion": "coordinated_diverse_response",
    "insufficient_response_signature": "insufficient_response_signature",
}


@dataclasses.dataclass
class _Phase88Record:
    cell_idx: int
    expected: tuple[str, ...]
    correct_substrate_fifo: bool
    correct_w21: bool
    correct_w40: bool
    correct_w41: bool
    ratified_substrate_fifo: bool
    ratified_w21: bool
    ratified_w40: bool
    ratified_w41: bool
    w40_decoder_branch: str
    w40_projection_branch: str
    w41_decoder_branch: str
    w41_integrated_branch: str
    w41_producer_axis_branch: str
    w41_trust_axis_branch: str
    w41_producer_services: tuple[str, ...]
    w41_trust_services: tuple[str, ...]
    w41_integrated_services: tuple[str, ...]
    w40_visible: int
    w41_visible: int
    w41_overhead: int
    w41_structured_bits: int
    w41_cram_factor: float
    w41_cid: str
    w41_manifest_v11_cid: str
    w41_verification_ok: bool


def _make_w41(*, schema, w40_orchestrator,
              synthesis_enabled: bool = True,
              manifest_v11_disabled: bool = False,
              abstain_on_axes_diverged: bool = True,
              ) -> IntegratedSynthesisOrchestrator:
    if (not synthesis_enabled and manifest_v11_disabled
            and not abstain_on_axes_diverged):
        registry = build_trivial_integrated_synthesis_registry(
            schema=schema,
            inner_w40_registry=w40_orchestrator.registry)
    else:
        registry = build_integrated_synthesis_registry(
            schema=schema,
            inner_w40_registry=w40_orchestrator.registry,
            synthesis_enabled=bool(synthesis_enabled),
            manifest_v11_disabled=bool(manifest_v11_disabled),
            abstain_on_axes_diverged=bool(abstain_on_axes_diverged),
        )
    return IntegratedSynthesisOrchestrator(
        inner=w40_orchestrator,
        registry=registry,
        enabled=True,
        require_w41_verification=True,
    )


def _build_phase87_per_cell_state(
        *, scenarios, n_eval_actual: int,
        bank_r87: str, T_decoder, quorum_min,
):
    """Re-build the per-cell W40 provider state used by phase87.

    Mirrors the data-construction half of
    :func:`run_phase87`; the W40 provider expects (consensus,
    quorum, response-signature) provider state per cell.
    """
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
            bank=bank_r87, cell_idx=cell_idx, n_total=n_eval_actual)
        per_rounds = _scenario_to_per_round_handoffs(scenario)
        w21 = _build_w21_disambiguator(
            T_decoder=T_decoder, registrations=registrations,
            quorum_min=quorum_min)
        w21_out = w21.decode_rounds(per_rounds)
        w21_top = tuple(
            sorted(set(str(s) for s in w21_out.get("services", ()))))
        if not w21_top:
            w21_top = scenario_gold

        if bank_r87 == "trivial_w40":
            consensus_top_per_cell[cell_idx] = ()
            quorum_top_per_cell[cell_idx] = ()
            response_signature_per_cell[cell_idx] = ()
            bench_gold_per_cell[cell_idx] = scenario_gold
            continue

        family_idx = (cell_idx - K1) % 4
        decoys = [
            ("payments_disjoint", "billing_disjoint"),
            ("api_disjoint", "frontend_disjoint"),
            ("storage_disjoint", "queue_disjoint"),
            ("db_query_disjoint", "search_disjoint"),
        ]
        true_recovery_top = tuple(sorted(decoys[family_idx]))

        if bank_r87 == "no_regression_diverse_agrees":
            consensus_top_per_cell[cell_idx] = w21_top
            quorum_top_per_cell[cell_idx] = (w21_top, w21_top)
            response_signature_per_cell[cell_idx] = (
                ("mac_off_cluster_a",
                 "primary db slowdown observed at edge boundary"),
                ("mac_off_cluster_b",
                 "queue lag amplified across worker shards"),
            )
            bench_gold_per_cell[cell_idx] = w21_top
        elif bank_r87 == "response_signature_collapse":
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
                consensus_top_per_cell[cell_idx] = w21_top
                quorum_top_per_cell[cell_idx] = (w21_top, w21_top)
                identical_response = (
                    "wrong consensus pinned to incident root cause")
                response_signature_per_cell[cell_idx] = (
                    ("mac_off_cluster_a", identical_response),
                    ("mac_off_cluster_b", identical_response),
                )
                bench_gold_per_cell[cell_idx] = true_recovery_top
        elif bank_r87 == "coordinated_diverse_response":
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
        elif bank_r87 == "insufficient_response_signature":
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

    return (consensus_top_per_cell, quorum_top_per_cell,
            response_signature_per_cell, bench_gold_per_cell)


def run_phase88(
        *,
        bank: str = "both_axes",
        n_eval: int = 16,
        bank_seed: int = 11,
        bank_replicates: int = 4,
        T_decoder: int | None = None,
        anchor_oracle_ids: tuple[str, ...] = (
            "service_graph", "change_history"),
        quorum_min: int = 2,
        synthesis_enabled: bool = True,
        manifest_v11_disabled: bool = False,
        abstain_on_axes_diverged: bool = True,
) -> dict[str, Any]:
    """Run one R-88 cell sweep on a single bank/seed.

    Returns a dict containing per-cell records and the aggregate
    rates (correctness / trust precision / branch distribution /
    visible-token totals).
    """
    if bank not in _R88_BANK_TO_R87:
        raise ValueError(
            f"unknown R-88 bank {bank!r}; "
            f"valid: {sorted(_R88_BANK_TO_R87.keys())!r}")
    bank_r87 = _R88_BANK_TO_R87[bank]
    schema = _stable_schema_capsule()
    n_repl = max(int(bank_replicates), (int(n_eval) + 3) // 4)
    scenarios = build_phase67_bank(
        bank="outside_resolves", n_replicates=n_repl,
        seed=bank_seed)
    scenarios = _interleave_by_family(scenarios, n_families=4)
    if len(scenarios) > n_eval:
        scenarios = scenarios[:n_eval]
    n_eval_actual = min(n_eval, len(scenarios))

    # Build the inner-bank W40 provider state.
    (consensus_top_per_cell, quorum_top_per_cell,
     response_signature_per_cell, bench_gold_per_cell) = (
        _build_phase87_per_cell_state(
            scenarios=scenarios, n_eval_actual=n_eval_actual,
            bank_r87=bank_r87, T_decoder=T_decoder,
            quorum_min=quorum_min))

    # Per-bank W40 enable flags.
    if bank == "trivial_w41":
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
    else:
        live_attestation_disabled = False
        trust_subspace_enabled = True
        manifest_v5_disabled = False
        host_diversity_enabled = True
        manifest_v6_disabled = False
        min_distinct_hosts = W36_DEFAULT_MIN_DISTINCT_HOSTS
        trajectory_enabled = True
        manifest_v7_disabled = False
        allow_single_host_trajectory_reroute = True
        consensus_enabled = True
        manifest_v8_disabled = False
        allow_consensus_reference_divergence_abstain = True
        quorum_enabled = True
        manifest_v9_disabled = False
        allow_disjoint_quorum_divergence_abstain = True
        response_signature_enabled = True
        manifest_v10_disabled = False
        allow_response_signature_collapse_abstain = True

    if bank == "trivial_w41":
        # When the inner W40 chain is fully disabled, force the
        # W41 wrapper to also be trivial.  This is the W41-L-
        # TRIVIAL-PASSTHROUGH falsifier branch.
        synthesis_enabled = False
        manifest_v11_disabled = True
        abstain_on_axes_diverged = False

    w40 = _make_w40(
        schema=schema, T_decoder=T_decoder, bank=bank_r87,
        n_eval=n_eval_actual,
        anchor_ids=anchor_oracle_ids,
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
        trajectory_threshold=W37_DEFAULT_TRAJECTORY_THRESHOLD,
        trajectory_margin_min=W37_DEFAULT_TRAJECTORY_MARGIN_MIN,
        min_anchored_observations=(
            W37_DEFAULT_MIN_ANCHORED_OBSERVATIONS),
        min_trajectory_anchored_hosts=(
            W37_DEFAULT_MIN_TRAJECTORY_ANCHORED_HOSTS),
        registered_anchor_host_ids=(
            "mac_remote", "mac_shadow", "mac1"),
        consensus_enabled=consensus_enabled,
        manifest_v8_disabled=manifest_v8_disabled,
        allow_consensus_reference_divergence_abstain=(
            allow_consensus_reference_divergence_abstain),
        consensus_strength_min=W38_DEFAULT_CONSENSUS_STRENGTH_MIN,
        divergence_margin_min=W38_DEFAULT_DIVERGENCE_MARGIN_MIN,
        consensus_top_per_cell=consensus_top_per_cell,
        quorum_enabled=quorum_enabled,
        manifest_v9_disabled=manifest_v9_disabled,
        allow_disjoint_quorum_divergence_abstain=(
            allow_disjoint_quorum_divergence_abstain),
        w39_quorum_min=W39_DEFAULT_QUORUM_MIN,
        w39_min_quorum_probes=W39_DEFAULT_MIN_QUORUM_PROBES,
        w39_consensus_strength_min=(
            W39_DEFAULT_QUORUM_STRENGTH_MIN),
        w39_divergence_margin_min=(
            W39_DEFAULT_QUORUM_DIVERGENCE_MARGIN_MIN),
        quorum_top_per_cell=quorum_top_per_cell,
        response_signature_enabled=response_signature_enabled,
        manifest_v10_disabled=manifest_v10_disabled,
        allow_response_signature_collapse_abstain=(
            allow_response_signature_collapse_abstain),
        response_text_diversity_min=(
            W40_DEFAULT_RESPONSE_TEXT_DIVERSITY_MIN),
        min_response_signature_probes=(
            W40_DEFAULT_MIN_RESPONSE_SIGNATURE_PROBES),
        response_signature_per_cell=response_signature_per_cell)

    w41 = _make_w41(
        schema=schema, w40_orchestrator=w40,
        synthesis_enabled=synthesis_enabled,
        manifest_v11_disabled=manifest_v11_disabled,
        abstain_on_axes_diverged=abstain_on_axes_diverged)

    records: list[_Phase88Record] = []
    for cell_idx in range(n_eval_actual):
        scenario = scenarios[cell_idx]
        gold = bench_gold_per_cell.get(
            cell_idx, _gold_for_cell(scenario))
        per_rounds = _scenario_to_per_round_handoffs(scenario)
        registrations = _build_w40_oracle_registrations_for_cell(
            bank=bank_r87, cell_idx=cell_idx, n_total=n_eval_actual)

        substrate_fifo = AttentionAwareBundleDecoder(
            T_decoder=T_decoder)
        substrate_fifo_out = substrate_fifo.decode_rounds(per_rounds)

        w21 = _build_w21_disambiguator(
            T_decoder=T_decoder, registrations=registrations,
            quorum_min=quorum_min)
        w21_out = w21.decode_rounds(per_rounds)

        # Reset the W40 inner chain's W21 cell state (mirrors
        # phase87 driver).
        try:
            w40.inner.inner.inner.inner.inner.inner.inner.inner = (
                _build_w21_disambiguator(
                    T_decoder=T_decoder,
                    registrations=registrations,
                    quorum_min=quorum_min))
        except AttributeError:
            pass

        w41_out = w41.decode_rounds(per_rounds)
        w40_result = w40.last_result
        w41_result = w41.last_result

        rat_fifo = _is_ratified(substrate_fifo_out)
        rat21 = _is_ratified(w21_out)
        rat40 = (
            bool(w40_result.answer.get("services", ()))
            if w40_result is not None else False)
        rat41 = bool(w41_out.get("services", ()))

        records.append(_Phase88Record(
            cell_idx=int(cell_idx),
            expected=tuple(gold),
            correct_substrate_fifo=(
                _is_correct(substrate_fifo_out, gold)
                if rat_fifo else False),
            correct_w21=(
                _is_correct(w21_out, gold) if rat21 else False),
            correct_w40=(
                _is_correct(w40_result.answer, gold)
                if (rat40 and w40_result is not None) else False),
            correct_w41=(
                _is_correct(w41_out, gold) if rat41 else False),
            ratified_substrate_fifo=bool(rat_fifo),
            ratified_w21=bool(rat21),
            ratified_w40=bool(rat40),
            ratified_w41=bool(rat41),
            w40_decoder_branch=str(
                w40_result.decoder_branch
                if w40_result is not None else ""),
            w40_projection_branch=str(
                w40_result.projection_branch
                if w40_result is not None else ""),
            w41_decoder_branch=str(
                w41_result.decoder_branch
                if w41_result is not None else ""),
            w41_integrated_branch=str(
                w41_result.integrated_branch
                if w41_result is not None else ""),
            w41_producer_axis_branch=str(
                w41_result.producer_axis_branch
                if w41_result is not None else ""),
            w41_trust_axis_branch=str(
                w41_result.trust_axis_branch
                if w41_result is not None else ""),
            w41_producer_services=(
                tuple(w41_result.producer_services)
                if w41_result is not None else ()),
            w41_trust_services=(
                tuple(w41_result.trust_services)
                if w41_result is not None else ()),
            w41_integrated_services=(
                tuple(w41_result.integrated_services)
                if w41_result is not None else ()),
            w40_visible=int(
                w40_result.n_w40_visible_tokens
                if w40_result is not None else 0),
            w41_visible=int(
                w41_result.n_w41_visible_tokens
                if w41_result is not None else 0),
            w41_overhead=int(
                w41_result.n_w41_overhead_tokens
                if w41_result is not None else 0),
            w41_structured_bits=int(
                w41_result.n_structured_bits
                if w41_result is not None else 0),
            w41_cram_factor=float(
                w41_result.cram_factor_w41
                if w41_result is not None else 0.0),
            w41_cid=str(
                w41_result.w41_cid
                if w41_result is not None else ""),
            w41_manifest_v11_cid=str(
                w41_result.manifest_v11_cid
                if w41_result is not None else ""),
            w41_verification_ok=bool(
                w41_result.verification_ok
                if w41_result is not None else False),
        ))

    n = len(records) or 1

    def _rate(attr: str) -> float:
        return sum(1 for r in records if getattr(r, attr)) / n

    def _trust(correct_attr: str, rat_attr: str) -> float:
        n_rat = sum(1 for r in records if getattr(r, rat_attr))
        if n_rat == 0:
            return 1.0
        return sum(
            1 for r in records if getattr(r, correct_attr)) / n_rat

    branch_hist: dict[str, int] = {}
    for r in records:
        branch_hist[r.w41_integrated_branch] = (
            branch_hist.get(r.w41_integrated_branch, 0) + 1)
    producer_axis_hist: dict[str, int] = {}
    trust_axis_hist: dict[str, int] = {}
    for r in records:
        producer_axis_hist[r.w41_producer_axis_branch] = (
            producer_axis_hist.get(
                r.w41_producer_axis_branch, 0) + 1)
        trust_axis_hist[r.w41_trust_axis_branch] = (
            trust_axis_hist.get(r.w41_trust_axis_branch, 0) + 1)

    total_w40_visible = sum(r.w40_visible for r in records)
    total_w41_visible = sum(r.w41_visible for r in records)
    total_w41_overhead = sum(r.w41_overhead for r in records)
    total_w41_bits = sum(r.w41_structured_bits for r in records)

    # Byte-equivalence check between W41 and W40 on trivial bank.
    w41_w40_byte_equivalent = all(
        r.w40_decoder_branch in (
            "trivial_response_signature_passthrough",
            "response_signature_disabled",
            "response_signature_no_references",
            "response_signature_no_trigger",
        ) and r.w41_overhead == 0
        for r in records
    ) if bank == "trivial_w41" else False

    summary = {
        "bank": bank,
        "bank_r87": bank_r87,
        "bank_seed": int(bank_seed),
        "n_eval": int(n_eval_actual),
        "T_decoder": T_decoder,
        "synthesis_enabled": bool(synthesis_enabled),
        "manifest_v11_disabled": bool(manifest_v11_disabled),
        "abstain_on_axes_diverged": bool(
            abstain_on_axes_diverged),
        # Correctness rates (overall) per layer.
        "correctness_substrate_fifo": _rate(
            "correct_substrate_fifo"),
        "correctness_w21": _rate("correct_w21"),
        "correctness_w40": _rate("correct_w40"),
        "correctness_w41": _rate("correct_w41"),
        # Trust precision (correct/ratified) per layer.
        "trust_precision_substrate_fifo": _trust(
            "correct_substrate_fifo", "ratified_substrate_fifo"),
        "trust_precision_w21": _trust(
            "correct_w21", "ratified_w21"),
        "trust_precision_w40": _trust(
            "correct_w40", "ratified_w40"),
        "trust_precision_w41": _trust(
            "correct_w41", "ratified_w41"),
        # Ratification rates.
        "ratification_substrate_fifo": _rate(
            "ratified_substrate_fifo"),
        "ratification_w21": _rate("ratified_w21"),
        "ratification_w40": _rate("ratified_w40"),
        "ratification_w41": _rate("ratified_w41"),
        # Cross-axis branch distributions.
        "w41_integrated_branch_hist": branch_hist,
        "w41_producer_axis_hist": producer_axis_hist,
        "w41_trust_axis_hist": trust_axis_hist,
        # Token / structured-bits accounting.
        "total_w40_visible": int(total_w40_visible),
        "total_w41_visible": int(total_w41_visible),
        "total_w41_overhead": int(total_w41_overhead),
        "total_w41_structured_bits": int(total_w41_bits),
        "mean_w41_overhead_per_cell": float(
            total_w41_overhead / n),
        "mean_w41_structured_bits_per_cell": float(
            total_w41_bits / n),
        # Equivalence (trivial bank only).
        "w41_w40_byte_equivalent": bool(w41_w40_byte_equivalent),
        # Verification.
        "n_w41_verified_ok": sum(
            1 for r in records if r.w41_verification_ok),
        "all_w41_verified_ok": all(
            r.w41_verification_ok for r in records),
    }
    return {
        "summary": summary,
        "records": [dataclasses.asdict(r) for r in records],
    }


def run_phase88_seed_sweep(
        *, bank: str = "both_axes",
        n_eval: int = 16,
        seeds: tuple[int, ...] = (11, 17, 23, 29, 31),
        T_decoder: int | None = None,
) -> dict[str, Any]:
    """Run one R-88 bank across multiple seeds and aggregate.

    Returns: per-seed summaries + aggregate min/max/mean across
    the seeds for correctness / trust precision /
    integrated-branch distribution.
    """
    per_seed: list[dict[str, Any]] = []
    for seed in seeds:
        result = run_phase88(
            bank=bank, n_eval=n_eval, bank_seed=int(seed),
            T_decoder=T_decoder)
        per_seed.append(result["summary"])

    def _vals(key: str) -> list[float]:
        return [float(s[key]) for s in per_seed]

    correctness_w41 = _vals("correctness_w41")
    correctness_w40 = _vals("correctness_w40")
    trust_w41 = _vals("trust_precision_w41")
    trust_w40 = _vals("trust_precision_w40")
    delta_corr = [a - b for a, b in zip(
        correctness_w41, correctness_w40)]
    delta_trust = [a - b for a, b in zip(trust_w41, trust_w40)]

    # Aggregate integrated-branch histogram (sum across seeds).
    branch_hist: dict[str, int] = {}
    for s in per_seed:
        for k, v in s.get(
                "w41_integrated_branch_hist", {}).items():
            branch_hist[k] = branch_hist.get(k, 0) + int(v)

    return {
        "bank": bank,
        "n_eval": int(n_eval),
        "seeds": list(seeds),
        "per_seed": per_seed,
        "min_correctness_w41": min(correctness_w41),
        "max_correctness_w41": max(correctness_w41),
        "mean_correctness_w41": (
            sum(correctness_w41) / len(correctness_w41)),
        "min_correctness_w40": min(correctness_w40),
        "max_correctness_w40": max(correctness_w40),
        "min_trust_precision_w41": min(trust_w41),
        "max_trust_precision_w41": max(trust_w41),
        "min_trust_precision_w40": min(trust_w40),
        "max_trust_precision_w40": max(trust_w40),
        "min_delta_correctness_w41_w40": min(delta_corr),
        "max_delta_correctness_w41_w40": max(delta_corr),
        "min_delta_trust_precision_w41_w40": min(delta_trust),
        "max_delta_trust_precision_w41_w40": max(delta_trust),
        "aggregate_w41_integrated_branch_hist": branch_hist,
        "all_w41_verified_ok": all(
            s["all_w41_verified_ok"] for s in per_seed),
        "all_byte_equivalent_w41_w40": all(
            s.get("w41_w40_byte_equivalent", False)
            for s in per_seed)
        if bank == "trivial_w41" else False,
    }


def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(
        description="Phase 88 / W41 R-88 driver.")
    parser.add_argument(
        "--bank", default="both_axes",
        choices=sorted(_R88_BANK_TO_R87.keys()))
    parser.add_argument("--n-eval", type=int, default=16)
    parser.add_argument(
        "--seeds", type=int, nargs="+",
        default=[11, 17, 23, 29, 31])
    parser.add_argument(
        "--out-dir", default=os.path.join(
            "vision_mvp", "experiments", "artifacts",
            "phase88"))
    args = parser.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    summary = run_phase88_seed_sweep(
        bank=args.bank, n_eval=args.n_eval,
        seeds=tuple(args.seeds))
    out_path = os.path.join(
        out_dir, f"{args.bank}_seed_sweep.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
