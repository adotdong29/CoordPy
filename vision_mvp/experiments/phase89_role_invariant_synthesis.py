"""Phase 89 -- W42 Cross-Role-Invariant Synthesis (R-89 family).

R-89 wraps the R-88 (W41 integrated synthesis) banks with a W42
:class:`RoleInvariantSynthesisOrchestrator` that adds a third
orthogonal evidence axis: the role-handoff signature CID computed
deterministically from the cell's input handoffs.  An honest
controller pre-registers a policy mapping role-handoff signature
CIDs to expected service sets.  W42 ratifies on agreement,
abstains on disagreement, and falls through on unknown signatures.

Banks
-----

* ``trivial_w42`` -- the W41 inner stack is fully disabled
  (matches R-88 ``trivial_w41``); W42 must reduce to W41
  byte-for-byte (decoder branch
  ``trivial_invariance_passthrough``).

* ``role_invariant_agrees`` -- W41 ratifies on the gold subset
  (R-88 ``both_axes``) AND the registered policy entry agrees
  with the W41 integrated services.  W42 ratifies via
  ``invariance_ratified`` on every cell.  Correctness and trust
  precision must not regress vs W41.

* ``role_invariant_recover`` -- W41 ratifies on a wrong colluded
  set (R-88 ``composite_collusion``) BUT the registered policy
  entry contains the *correct* expected services for the
  cell's role-handoff signature (the controller-side policy
  was honest).  W42 abstains via
  ``invariance_diverged_abstained`` on the recovery half;
  trust precision strictly recovers from 0.500 (W41) to 1.000
  (W42).  This is the load-bearing W42-3 strict-gain claim and
  the H5 hard-gate anchor.

* ``full_composite_collusion`` -- the adversary has ALSO
  poisoned the controller-side policy registry to register the
  wrong colluded set as the expected services for the colluded
  cells.  W42 ratifies via ``invariance_ratified`` on the wrong
  set.  W42-L-FULL-COMPOSITE-COLLUSION-CAP fires; trust
  precision unchanged from W41.

* ``insufficient_invariance_policy`` -- no policy entry is
  registered for the cell's role-handoff signature.  W42 falls
  through ``invariance_no_policy``; correctness and trust
  precision unchanged from W41 (preserving W41 byte-for-W40 on
  the answer).

The bench reuses the entire W41 phase88 builder chain so the
inner W21..W41 decisions are identical byte-for-byte to R-88.
The only addition is the W42 wrapper and the new R-89 record
class.
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
from vision_mvp.coordpy.team_coord import _DecodedHandoff


# W42 incident-marker handoff vocabulary.  Used by the R-89 driver
# to give each cell a structurally unique role-handoff signature
# even when underlying scenarios collapse to byte-identical handoff
# tuples.  In a real deployment this corresponds to the upstream
# cell schema's per-incident dispatcher tag.  The W21..W40 inner
# chain ignores the unknown claim_kind by construction (no
# admission policy matches it; no decoder rule keys off it), so
# adding it does not perturb byte-equivalent W41 / W40 / W21
# behaviour.
W42_INCIDENT_MARKER_ROLE: str = "incident_dispatcher"
W42_INCIDENT_MARKER_KIND: str = "w42_incident_id"


def _phase89_per_round_handoffs_with_marker(
        *, scenario, cell_idx: int) -> tuple:
    """Wrap the base per-round handoffs with a per-cell marker.

    Adds one synthetic ``_DecodedHandoff`` to the first round
    carrying ``(W42_INCIDENT_MARKER_ROLE,
    W42_INCIDENT_MARKER_KIND, "incident_<cell_idx>")``.  The
    marker is structural cell metadata that an honest controller
    would emit upstream; it is content-addressed by the W42
    role-handoff signature CID.  The marker uses an unknown
    claim_kind; W21..W41 ignore it by construction.
    """
    base = _scenario_to_per_round_handoffs(scenario)
    marker = _DecodedHandoff(
        source_role=W42_INCIDENT_MARKER_ROLE,
        claim_kind=W42_INCIDENT_MARKER_KIND,
        payload=f"incident_{int(cell_idx):04d}",
    )
    if not base:
        return ((marker,),)
    head = (marker,) + tuple(base[0])
    out = (head,) + tuple(tuple(r) for r in base[1:])
    return out
from vision_mvp.experiments.phase88_integrated_synthesis import (
    _build_phase87_per_cell_state,
    _make_w41,
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
)
from vision_mvp.coordpy.role_invariant_synthesis import (
    RoleInvariancePolicyEntry,
    RoleInvariantSynthesisOrchestrator,
    build_role_invariant_registry,
    build_trivial_role_invariant_registry,
    compute_role_handoff_signature_cid,
    W42_BRANCH_TRIVIAL_INVARIANCE_PASSTHROUGH,
    W42_BRANCH_INVARIANCE_RATIFIED,
    W42_BRANCH_INVARIANCE_DIVERGED_ABSTAINED,
    W42_BRANCH_INVARIANCE_NO_TRIGGER,
    W42_BRANCH_INVARIANCE_NO_POLICY,
    W42_BRANCH_INVARIANCE_DISABLED,
    W42_BRANCH_INVARIANCE_REJECTED,
)


_R89_BANK_TO_R88 = {
    "trivial_w42": "trivial_w41",
    "role_invariant_agrees": "both_axes",
    "role_invariant_recover": "composite_collusion",
    "full_composite_collusion": "composite_collusion",
    "insufficient_invariance_policy": "both_axes",
}


@dataclasses.dataclass
class _Phase89Record:
    cell_idx: int
    expected: tuple[str, ...]
    correct_substrate_fifo: bool
    correct_w21: bool
    correct_w40: bool
    correct_w41: bool
    correct_w42: bool
    ratified_substrate_fifo: bool
    ratified_w21: bool
    ratified_w40: bool
    ratified_w41: bool
    ratified_w42: bool
    w41_decoder_branch: str
    w41_integrated_branch: str
    w42_decoder_branch: str
    w42_invariance_branch: str
    w42_role_handoff_signature_cid: str
    w42_policy_entry_cid: str
    w42_integrated_services_pre: tuple[str, ...]
    w42_expected_services: tuple[str, ...]
    w42_integrated_services_post: tuple[str, ...]
    w42_invariance_score: float
    w41_visible: int
    w42_visible: int
    w42_overhead: int
    w42_structured_bits: int
    w42_cram_factor: float
    w42_cid: str
    w42_manifest_v12_cid: str
    w42_verification_ok: bool


def _make_w42(
        *,
        schema,
        w41_orchestrator,
        policy_entries: Sequence[RoleInvariancePolicyEntry] = (),
        invariance_enabled: bool = True,
        manifest_v12_disabled: bool = False,
        abstain_on_invariance_diverged: bool = True,
) -> RoleInvariantSynthesisOrchestrator:
    if (not invariance_enabled
            and manifest_v12_disabled
            and not abstain_on_invariance_diverged):
        registry = build_trivial_role_invariant_registry(
            schema=schema,
            inner_w41_registry=w41_orchestrator.registry)
    else:
        registry = build_role_invariant_registry(
            schema=schema,
            inner_w41_registry=w41_orchestrator.registry,
            policy_entries=tuple(policy_entries),
            invariance_enabled=bool(invariance_enabled),
            manifest_v12_disabled=bool(manifest_v12_disabled),
            abstain_on_invariance_diverged=bool(
                abstain_on_invariance_diverged),
        )
    return RoleInvariantSynthesisOrchestrator(
        inner=w41_orchestrator,
        registry=registry,
        enabled=True,
        require_w42_verification=True,
    )


def _build_phase89_policy_entries(
        *,
        bank: str,
        scenarios,
        n_eval_actual: int,
        bench_gold_per_cell: dict[int, tuple[str, ...]],
) -> list[RoleInvariancePolicyEntry]:
    """Build the role-invariance policy entries for one bank.

    The mapping is from per-cell role-handoff signature CID to
    expected services.  We pre-compute the role-handoff signature
    of each cell's input handoffs and register the *honest* gold
    services as the expected set for that signature -- except on
    ``full_composite_collusion`` where the adversary poisons the
    expected set to match the wrong colluded W41 ratified set,
    and on ``insufficient_invariance_policy`` where no entry is
    registered.
    """
    if bank == "insufficient_invariance_policy":
        return []
    if bank == "trivial_w42":
        return []
    entries: list[RoleInvariancePolicyEntry] = []
    seen: dict[str, tuple[str, ...]] = {}
    for cell_idx in range(n_eval_actual):
        scenario = scenarios[cell_idx]
        per_rounds = _phase89_per_round_handoffs_with_marker(
            scenario=scenario, cell_idx=cell_idx)
        sig = compute_role_handoff_signature_cid(per_rounds)
        if bank == "role_invariant_agrees":
            # Honest policy: register the actual gold services for
            # each cell as the expected set.  Use the bench gold
            # which equals the W21 emitted top_set on this bank.
            expected = tuple(sorted(
                bench_gold_per_cell.get(cell_idx,
                                        _gold_for_cell(scenario))))
        elif bank == "role_invariant_recover":
            # Honest policy on the recovery half: register the
            # *true_recovery_top* as the expected set for the
            # colluded cells (the controller knows the role-handoff
            # signature implies the gold answer that should be
            # there, even though W41 ratifies the wrong colluded
            # set).
            expected = tuple(sorted(
                bench_gold_per_cell.get(cell_idx,
                                        _gold_for_cell(scenario))))
        elif bank == "full_composite_collusion":
            # Poisoned policy: register the W21 (compromised) set as
            # the expected set, exactly matching the wrong colluded
            # W41 ratified set.  W42 will ratify on the wrong set
            # (W42-L-FULL-COMPOSITE-COLLUSION-CAP).  The W21 set is
            # the producer-side ratified set that W41 trusts; we
            # approximate it as scenario_gold (which equals the
            # W21 emitted top_set on the prefix half) on every cell
            # so the policy AGREES with the W41 wrong set on every
            # cell of the recovery half.
            scenario_gold = tuple(sorted(_gold_for_cell(scenario)))
            expected = scenario_gold
        else:
            expected = tuple(sorted(_gold_for_cell(scenario)))
        # Avoid duplicate registration (would be a no-op anyway).
        if sig not in seen:
            seen[sig] = expected
            entries.append(RoleInvariancePolicyEntry(
                role_handoff_signature_cid=sig,
                expected_services=expected,
            ))
    return entries


def run_phase89(
        *,
        bank: str = "role_invariant_recover",
        n_eval: int = 16,
        bank_seed: int = 11,
        bank_replicates: int = 4,
        T_decoder: int | None = None,
        anchor_oracle_ids: tuple[str, ...] = (
            "service_graph", "change_history"),
        quorum_min: int = 2,
        invariance_enabled: bool = True,
        manifest_v12_disabled: bool = False,
        abstain_on_invariance_diverged: bool = True,
) -> dict[str, Any]:
    """Run one R-89 cell sweep on a single bank/seed."""
    if bank not in _R89_BANK_TO_R88:
        raise ValueError(
            f"unknown R-89 bank {bank!r}; "
            f"valid: {sorted(_R89_BANK_TO_R88.keys())!r}")
    bank_r88 = _R89_BANK_TO_R88[bank]
    schema = _stable_schema_capsule()
    n_repl = max(int(bank_replicates), (int(n_eval) + 3) // 4)
    scenarios = build_phase67_bank(
        bank="outside_resolves", n_replicates=n_repl,
        seed=bank_seed)
    scenarios = _interleave_by_family(scenarios, n_families=4)
    if len(scenarios) > n_eval:
        scenarios = scenarios[:n_eval]
    n_eval_actual = min(n_eval, len(scenarios))

    # Map R-89 bank -> R-87 inner bank (W40 driver) via R-88.
    _R88_BANK_TO_R87 = {
        "trivial_w41": "trivial_w40",
        "both_axes": "no_regression_diverse_agrees",
        "trust_only_safety": "response_signature_collapse",
        "composite_collusion": "coordinated_diverse_response",
        "insufficient_response_signature": (
            "insufficient_response_signature"),
    }
    bank_r87 = _R88_BANK_TO_R87[bank_r88]

    # Build the inner-bank W40 provider state.
    (consensus_top_per_cell, quorum_top_per_cell,
     response_signature_per_cell, bench_gold_per_cell) = (
        _build_phase87_per_cell_state(
            scenarios=scenarios, n_eval_actual=n_eval_actual,
            bank_r87=bank_r87, T_decoder=T_decoder,
            quorum_min=quorum_min))

    # Per-bank W40/W41 enable flags.
    if bank == "trivial_w42":
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
        synthesis_enabled = False
        manifest_v11_disabled = True
        abstain_on_axes_diverged = False
        invariance_enabled = False
        manifest_v12_disabled = True
        abstain_on_invariance_diverged = False
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
        synthesis_enabled = True
        manifest_v11_disabled = False
        abstain_on_axes_diverged = True

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

    # Build the W42 policy entries.
    policy_entries = _build_phase89_policy_entries(
        bank=bank, scenarios=scenarios,
        n_eval_actual=n_eval_actual,
        bench_gold_per_cell=bench_gold_per_cell)

    w42 = _make_w42(
        schema=schema, w41_orchestrator=w41,
        policy_entries=policy_entries,
        invariance_enabled=invariance_enabled,
        manifest_v12_disabled=manifest_v12_disabled,
        abstain_on_invariance_diverged=(
            abstain_on_invariance_diverged))

    records: list[_Phase89Record] = []
    for cell_idx in range(n_eval_actual):
        scenario = scenarios[cell_idx]
        gold = bench_gold_per_cell.get(
            cell_idx, _gold_for_cell(scenario))
        # Use marker-augmented per-rounds for W42 (and the inner
        # W21..W41 chain, which ignores the marker by construction).
        # On the trivial bank we use the unaugmented per-rounds so
        # W42 reduces to W41 byte-for-W40.
        if bank == "trivial_w42":
            per_rounds = _scenario_to_per_round_handoffs(scenario)
        else:
            per_rounds = _phase89_per_round_handoffs_with_marker(
                scenario=scenario, cell_idx=cell_idx)
        registrations = _build_w40_oracle_registrations_for_cell(
            bank=bank_r87, cell_idx=cell_idx,
            n_total=n_eval_actual)

        substrate_fifo = AttentionAwareBundleDecoder(
            T_decoder=T_decoder)
        substrate_fifo_out = substrate_fifo.decode_rounds(
            per_rounds)

        w21 = _build_w21_disambiguator(
            T_decoder=T_decoder, registrations=registrations,
            quorum_min=quorum_min)
        w21_out = w21.decode_rounds(per_rounds)

        # Reset the W41-inner-W40 chain's W21 cell state.
        try:
            w40.inner.inner.inner.inner.inner.inner.inner.inner = (
                _build_w21_disambiguator(
                    T_decoder=T_decoder,
                    registrations=registrations,
                    quorum_min=quorum_min))
        except AttributeError:
            pass

        w42_out = w42.decode_rounds(per_rounds)
        w40_result = w40.last_result
        w41_result = w41.last_result
        w42_result = w42.last_result

        rat_fifo = _is_ratified(substrate_fifo_out)
        rat21 = _is_ratified(w21_out)
        rat40 = (
            bool(w40_result.answer.get("services", ()))
            if w40_result is not None else False)
        rat41 = (
            bool(w41_result.integrated_services)
            if w41_result is not None else False)
        rat42 = bool(w42_out.get("services", ()))

        # Correctness: requires ratified AND set-equal-to-gold.
        gold_t = tuple(sorted(gold))
        c_fifo = (
            _is_correct(substrate_fifo_out, gold_t)
            if rat_fifo else False)
        c21 = _is_correct(w21_out, gold_t) if rat21 else False
        c40 = (
            _is_correct(w40_result.answer, gold_t)
            if (rat40 and w40_result is not None) else False)
        c41 = (
            tuple(sorted(w41_result.integrated_services))
            == gold_t
            if (rat41 and w41_result is not None) else False)
        c42 = (
            tuple(sorted(w42_out.get("services", ())))
            == gold_t
            if rat42 else False)

        records.append(_Phase89Record(
            cell_idx=int(cell_idx),
            expected=gold_t,
            correct_substrate_fifo=bool(c_fifo),
            correct_w21=bool(c21),
            correct_w40=bool(c40),
            correct_w41=bool(c41),
            correct_w42=bool(c42),
            ratified_substrate_fifo=bool(rat_fifo),
            ratified_w21=bool(rat21),
            ratified_w40=bool(rat40),
            ratified_w41=bool(rat41),
            ratified_w42=bool(rat42),
            w41_decoder_branch=str(
                w41_result.decoder_branch
                if w41_result is not None else ""),
            w41_integrated_branch=str(
                w41_result.integrated_branch
                if w41_result is not None else ""),
            w42_decoder_branch=str(
                w42_result.decoder_branch
                if w42_result is not None else ""),
            w42_invariance_branch=str(
                w42_result.invariance_branch
                if w42_result is not None else ""),
            w42_role_handoff_signature_cid=str(
                w42_result.role_handoff_signature_cid
                if w42_result is not None else ""),
            w42_policy_entry_cid=str(
                w42_result.policy_entry_cid
                if w42_result is not None else ""),
            w42_integrated_services_pre=(
                tuple(w42_result.integrated_services_pre_w42)
                if w42_result is not None else ()),
            w42_expected_services=(
                tuple(w42_result.expected_services)
                if w42_result is not None else ()),
            w42_integrated_services_post=(
                tuple(w42_result.integrated_services_post_w42)
                if w42_result is not None else ()),
            w42_invariance_score=float(
                w42_result.invariance_score
                if w42_result is not None else 0.0),
            w41_visible=int(
                w41_result.n_w41_visible_tokens
                if w41_result is not None else 0),
            w42_visible=int(
                w42_result.n_w42_visible_tokens
                if w42_result is not None else 0),
            w42_overhead=int(
                w42_result.n_w42_overhead_tokens
                if w42_result is not None else 0),
            w42_structured_bits=int(
                w42_result.n_structured_bits
                if w42_result is not None else 0),
            w42_cram_factor=float(
                w42_result.cram_factor_w42
                if w42_result is not None else 0.0),
            w42_cid=str(
                w42_result.w42_cid
                if w42_result is not None else ""),
            w42_manifest_v12_cid=str(
                w42_result.manifest_v12_cid
                if w42_result is not None else ""),
            w42_verification_ok=bool(
                w42_result.verification_ok
                if w42_result is not None else False),
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
        branch_hist[r.w42_invariance_branch] = (
            branch_hist.get(r.w42_invariance_branch, 0) + 1)

    # Distinct role-handoff signatures observed.
    sig_set = {r.w42_role_handoff_signature_cid for r in records
               if r.w42_role_handoff_signature_cid}

    total_w41_visible = sum(r.w41_visible for r in records)
    total_w42_visible = sum(r.w42_visible for r in records)
    total_w42_overhead = sum(r.w42_overhead for r in records)
    total_w42_bits = sum(r.w42_structured_bits for r in records)

    w42_w41_byte_equivalent = all(
        r.w42_overhead == 0
        and r.w42_invariance_branch == (
            W42_BRANCH_TRIVIAL_INVARIANCE_PASSTHROUGH)
        for r in records
    ) if bank == "trivial_w42" else False

    summary = {
        "bank": bank,
        "bank_r88": bank_r88,
        "bank_r87": bank_r87,
        "bank_seed": int(bank_seed),
        "n_eval": int(n_eval_actual),
        "T_decoder": T_decoder,
        "invariance_enabled": bool(invariance_enabled),
        "manifest_v12_disabled": bool(manifest_v12_disabled),
        "abstain_on_invariance_diverged": bool(
            abstain_on_invariance_diverged),
        "n_policy_entries": int(len(policy_entries)),
        "n_distinct_role_handoff_signatures": int(len(sig_set)),
        "correctness_substrate_fifo": _rate(
            "correct_substrate_fifo"),
        "correctness_w21": _rate("correct_w21"),
        "correctness_w40": _rate("correct_w40"),
        "correctness_w41": _rate("correct_w41"),
        "correctness_w42": _rate("correct_w42"),
        "trust_precision_substrate_fifo": _trust(
            "correct_substrate_fifo", "ratified_substrate_fifo"),
        "trust_precision_w21": _trust("correct_w21", "ratified_w21"),
        "trust_precision_w40": _trust("correct_w40", "ratified_w40"),
        "trust_precision_w41": _trust("correct_w41", "ratified_w41"),
        "trust_precision_w42": _trust("correct_w42", "ratified_w42"),
        "ratification_substrate_fifo": _rate(
            "ratified_substrate_fifo"),
        "ratification_w21": _rate("ratified_w21"),
        "ratification_w40": _rate("ratified_w40"),
        "ratification_w41": _rate("ratified_w41"),
        "ratification_w42": _rate("ratified_w42"),
        "w42_invariance_branch_hist": branch_hist,
        "total_w41_visible": int(total_w41_visible),
        "total_w42_visible": int(total_w42_visible),
        "total_w42_overhead": int(total_w42_overhead),
        "total_w42_structured_bits": int(total_w42_bits),
        "mean_w42_overhead_per_cell": float(
            total_w42_overhead / n),
        "mean_w42_structured_bits_per_cell": float(
            total_w42_bits / n),
        "w42_w41_byte_equivalent": bool(w42_w41_byte_equivalent),
        "n_w42_verified_ok": sum(
            1 for r in records if r.w42_verification_ok),
        "all_w42_verified_ok": all(
            r.w42_verification_ok for r in records),
    }
    return {
        "summary": summary,
        "records": [dataclasses.asdict(r) for r in records],
    }


def run_phase89_seed_sweep(
        *, bank: str = "role_invariant_recover",
        n_eval: int = 16,
        seeds: tuple[int, ...] = (11, 17, 23, 29, 31),
        T_decoder: int | None = None,
) -> dict[str, Any]:
    per_seed: list[dict[str, Any]] = []
    for seed in seeds:
        result = run_phase89(
            bank=bank, n_eval=n_eval, bank_seed=int(seed),
            T_decoder=T_decoder)
        per_seed.append(result["summary"])

    def _vals(key: str) -> list[float]:
        return [float(s[key]) for s in per_seed]

    correctness_w42 = _vals("correctness_w42")
    correctness_w41 = _vals("correctness_w41")
    correctness_w40 = _vals("correctness_w40")
    trust_w42 = _vals("trust_precision_w42")
    trust_w41 = _vals("trust_precision_w41")
    trust_w40 = _vals("trust_precision_w40")
    delta_corr_w42_w41 = [
        a - b for a, b in zip(correctness_w42, correctness_w41)]
    delta_trust_w42_w41 = [
        a - b for a, b in zip(trust_w42, trust_w41)]

    branch_hist: dict[str, int] = {}
    for s in per_seed:
        for k, v in s.get(
                "w42_invariance_branch_hist", {}).items():
            branch_hist[k] = branch_hist.get(k, 0) + int(v)

    return {
        "bank": bank,
        "n_eval": int(n_eval),
        "seeds": list(seeds),
        "per_seed": per_seed,
        "min_correctness_w42": min(correctness_w42),
        "max_correctness_w42": max(correctness_w42),
        "mean_correctness_w42": (
            sum(correctness_w42) / len(correctness_w42)),
        "min_trust_precision_w42": min(trust_w42),
        "max_trust_precision_w42": max(trust_w42),
        "mean_trust_precision_w42": (
            sum(trust_w42) / len(trust_w42)),
        "min_correctness_w41": min(correctness_w41),
        "max_correctness_w41": max(correctness_w41),
        "min_trust_precision_w41": min(trust_w41),
        "max_trust_precision_w41": max(trust_w41),
        "min_correctness_w40": min(correctness_w40),
        "min_trust_precision_w40": min(trust_w40),
        "min_delta_correctness_w42_w41": min(delta_corr_w42_w41),
        "max_delta_correctness_w42_w41": max(delta_corr_w42_w41),
        "min_delta_trust_precision_w42_w41": min(
            delta_trust_w42_w41),
        "max_delta_trust_precision_w42_w41": max(
            delta_trust_w42_w41),
        "aggregate_w42_invariance_branch_hist": branch_hist,
        "all_w42_verified_ok": all(
            s["all_w42_verified_ok"] for s in per_seed),
        "all_byte_equivalent_w42_w41": all(
            s.get("w42_w41_byte_equivalent", False)
            for s in per_seed)
        if bank == "trivial_w42" else False,
    }


def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(
        description="Phase 89 / W42 R-89 driver.")
    parser.add_argument(
        "--bank", default="role_invariant_recover",
        choices=sorted(_R89_BANK_TO_R88.keys()))
    parser.add_argument("--n-eval", type=int, default=16)
    parser.add_argument(
        "--seeds", type=int, nargs="+",
        default=[11, 17, 23, 29, 31])
    parser.add_argument(
        "--out-dir", default=os.path.join(
            "vision_mvp", "experiments", "artifacts", "phase89"))
    args = parser.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    summary = run_phase89_seed_sweep(
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
