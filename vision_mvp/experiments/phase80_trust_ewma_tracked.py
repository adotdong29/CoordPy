"""Phase 80 — Trust-EWMA-tracked multi-oracle adjudication (SDK v3.34, W33).

Follow-up to SDK v3.33 (W32).  W32 closed the long-window scaling-
stability axis (W31-C-LONG-WINDOW-CONVERGENCE) and characterised the
prompt-class-dependent cross-architecture disagreement frontier at
temp 0.  W32 was a PARTIAL SUCCESS but five named open conjectures
inherit forward (W32-C-LONG-WINDOW-STRICT-GAIN, W32-C-CROSS-HOST-LIVE-
GOLD-MAGNITUDE, W32-C-OLD-LINE-EWMA-TRUST, W32-C-NATIVE-LATENT,
W32-C-MULTI-HOST).

W33 (this phase) integrates the W32 EWMA primitive with the W21
multi-oracle adjudicator to give per-oracle online trust calibration
— the closest honest discharge of W32-C-OLD-LINE-EWMA-TRUST.  At
each cell, the inner W21 produces a quorum-resolved decision; W33
derives a per-oracle agreement signal (closed-form: 1.0 if probe.top_set
⊆ resolved.top_set; 0.0 if disjoint; 0.5 if partial) and EWMA-updates
each oracle's trust.  When an oracle's EWMA-tracked trust falls below
the registered ``trust_threshold``, its vote is *excluded* from the
W33 effective tally; on regimes where an honest oracle becomes
compromised mid-session, the W33 layer correctly de-trusts the
compromised oracle and abstains (or reroutes) where W21 would commit
to a wrong answer.

Phase-80 sub-banks
==================

Six pre-committed sub-banks:

* **R-80-TRIVIAL-W33** (H2 anchor; W33-Λ-trivial-trust-ewma).
  ``trust_ewma_enabled=False`` AND ``manifest_v3_disabled=True`` AND
  ``trust_trajectory_window=0`` ⇒ W33 reduces to W21 byte-for-byte.
* **R-80-COMPROMISED-SHIFT** (H6 main load-bearing claim;
  discharges W32-C-OLD-LINE-EWMA-TRUST and W21-C-CALIBRATED-TRUST).
  Three oracles registered: service_graph (always honest),
  change_history (honest in prefix, compromised in shift),
  oncall_notes (compromised throughout).  Cells 0..K: W21
  quorum forms on gold (2 honest votes); oncall's EWMA drops to ~0.
  Cells K..N: ch + oncall both vote decoy → W21 quorum forms on
  decoy → W21 ratifies WRONG answer.  W33 with EWMA-tracked trust:
  oncall is detrusted (EWMA below threshold) → effective votes
  exclude oncall → ch+sg disagree → no quorum → W33 abstains
  (correct conservative move; trust precision improves).
* **R-80-NO-TRUST-SHIFT** (W33-Λ-no-trust-shift falsifier).  All
  oracles agree with the consortium quorum throughout.  Every
  EWMA stays at 1.0; W33 ties W21 byte-for-byte.
* **R-80-FROZEN-TRUST-THRESHOLD** (W33-Λ-frozen-threshold falsifier).
  ``trust_threshold = 0.0`` (every EWMA-tracked oracle counts
  regardless).  W33 ties W21 byte-for-byte even on R-80-COMPROMISED-
  SHIFT (the EWMA drops, but the threshold gate never fires).
* **R-80-MIS-TRUST-SHIFT** (W33-Λ-mis-trust-shift falsifier).  An
  honest oracle is mis-classified as a "trust-shifted" oracle (its
  EWMA dropped because of a transient disagreement, not a true
  compromise).  W33 may regress vs W21 by losing a critical vote.
* **R-80-MANIFEST-V3-TAMPER** (H8 cross-component tamper detection).
  Cross-component swap that affects ``oracle_trust_state_cid`` but
  NOT the W21 / W22 manifest's component set.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import random
import sys
from typing import Any, Sequence

from vision_mvp.wevra.team_coord import (
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
    TrustEWMARatificationEnvelope,
    TrustEWMARegistry,
    TrustTrajectoryEntry,
    W33TrustEWMAResult,
    verify_trust_ewma_ratification,
    derive_per_oracle_agreement_signal,
    build_trivial_trust_ewma_registry,
    build_trust_ewma_registry,
    W33_TRUST_EWMA_SCHEMA_VERSION,
    W33_DEFAULT_TRUST_THRESHOLD,
    W33_DEFAULT_TRUST_TRAJECTORY_WINDOW,
    W33_DEFAULT_EWMA_ALPHA,
    W33_BRANCH_TRUST_EWMA_RESOLVED,
    W33_BRANCH_TRIVIAL_TRUST_EWMA_PASSTHROUGH,
    W33_BRANCH_TRUST_EWMA_DETRUSTED_ABSTAIN,
    W33_BRANCH_TRUST_EWMA_DETRUSTED_REROUTE,
    W21_BRANCH_QUORUM_RESOLVED,
    _DecodedHandoff,
    _compute_oracle_trust_state_cid,
    _compute_trust_trajectory_cid,
    _compute_w33_manifest_v3_cid,
    _compute_w33_outer_cid,
)
from vision_mvp.experiments.phase67_outside_information import (
    build_phase67_bank,
)


# ---------------------------------------------------------------------------
# Schema setup (reuse phase68's R-66-OUTSIDE-REQUIRED bundle shape)
# ---------------------------------------------------------------------------


def _stable_schema_capsule() -> SchemaCapsule:
    """Stable SchemaCapsule used across all phase80 sub-banks.

    The W33 envelope's schema_cid is a registered-vs-env check; we use
    a single stable schema across the entire phase80 bench so the
    registry's ``schema`` is always the same and the verifier never
    fires the ``w33_schema_cid_mismatch`` mode by accident.
    """
    from vision_mvp.experiments.phase76_geometry_partitioned_product_manifold import (
        INCIDENT_TRIAGE_AMBIENT_VOCABULARY)
    from vision_mvp.wevra.team_coord import (
        build_incident_triage_schema_capsule)
    return build_incident_triage_schema_capsule()


# ---------------------------------------------------------------------------
# Compromised oracle factories — for the trust-shift regimes.
# ---------------------------------------------------------------------------


def _build_compromised_change_history_oracle() -> ChangeHistoryOracle:
    """Build a ChangeHistoryOracle whose change_log points at the
    *wrong* gold pair for each root-cause family.

    Used in R-80-COMPROMISED-SHIFT to model an oracle that becomes
    compromised mid-session (e.g. registry-poisoning attack) by
    flipping the change-log to the decoy pair instead of the gold
    pair.
    """
    return ChangeHistoryOracle(
        oracle_id="change_history_compromised",
        change_log={
            # Wrong: deadlock root_cause should map to (orders, payments)
            # but we map it to (db, api) — points to the gold pair of
            # the *next* family, which on R-66-OUTSIDE-REQUIRED is the
            # decoy.
            "deadlock":             ("api", "db"),
            "pool_exhaustion":      ("orders", "payments"),
            "disk_fill":            ("web", "db_query"),
            "slow_query_cascade":   ("storage", "logs_pipeline"),
        })


def _build_compromised_oncall_notes_oracle() -> OnCallNotesOracle:
    """Build an OnCallNotesOracle whose notes point at the wrong pair.

    Used in R-80-COMPROMISED-SHIFT as the oracle that is compromised
    *throughout* the session.  After K cells where it consistently
    disagreed with the quorum, the W33 EWMA-tracked trust will drop
    below threshold and the oracle's vote will be excluded from the
    effective tally.
    """
    return OnCallNotesOracle(
        oracle_id="oncall_notes_compromised",
        notes_log={
            "deadlock":             ("api", "db"),
            "pool_exhaustion":      ("orders", "payments"),
            "disk_fill":            ("web", "db_query"),
            "slow_query_cascade":   ("storage", "logs_pipeline"),
        })


def _build_oracle_registrations_for_cell(
        *,
        bank: str,
        cell_idx: int,
        n_total: int,
) -> tuple[OracleRegistration, ...]:
    """Build the per-cell registered oracle set.

    The W33 trust-EWMA layer requires *consistent* oracle IDs across
    cells (the EWMA state is keyed by oracle_id).  So when an oracle
    "becomes compromised" mid-session, we keep the SAME oracle_id but
    swap in a compromised implementation.

    Bank semantics
    --------------
    * ``trivial_w33``: all 3 oracles always honest.  W21 quorum
      forms on gold every cell; W33 ties W21 byte-for-byte.
    * ``no_trust_shift``: same as trivial_w33 — all 3 always agree.
      Used to confirm W33-Λ-no-trust-shift falsifier.
    * ``compromised_shift``: TWO-stage compromise.  service_graph is
      always honest.  change_history is HONEST in cells [0..K) and
      COMPROMISED in cells [K..N).  oncall_notes is COMPROMISED
      throughout.
        Cells [0..K): votes={sg:gold, ch:gold, oc:decoy}; quorum on
          gold (2 vs 1) → W21 correct.  oc disagrees with quorum
          every cell → oc's EWMA drops toward 0.
        Cells [K..N): votes={sg:gold, ch:decoy, oc:decoy}; quorum on
          decoy (2 vs 1) → W21 commits to DECOY (wrong).
          With W33 EWMA-tracked trust at threshold=0.5: oc has been
          de-trusted (EWMA < 0.5); effective tally excludes oc →
          gold(1) vs decoy(1) → no quorum → W33 abstains (correct
          conservative move).
    * ``frozen_trust_threshold``: same regime as compromised_shift,
      but trust_threshold=0.0 in run_phase80 — gate never fires;
      W33 ties W21 byte-for-byte.
    * ``mis_trust_shift``: service_graph emits a TRANSIENTLY wrong
      vote on cells [0..3) (a flaky oracle) but is honest afterwards.
      W33 incorrectly de-trusts it; on later cells where its vote
      is critical, W33 may regress vs W21.
    * ``manifest_v3_tamper``: same regime as compromised_shift; used
      for the H8 cross-component tamper-detection sweep.
    """
    # Three-phase regime: calibration / single-compromise / double-
    # compromise.  Phase boundaries are 3/8 and 5/8 of the bench.
    K1 = max(1, (3 * n_total) // 8)
    K2 = max(K1 + 1, (5 * n_total) // 8)
    if bank in ("trivial_w33", "no_trust_shift"):
        # All-honest setup.  service_graph is the canonical honest
        # registry; change_history mirrors it for the gold pair;
        # oncall_notes also mirrors the gold pair.  All three vote
        # for the same gold tags every cell ⇒ W33's EWMA stays at 1.0.
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
    if bank in ("compromised_shift", "frozen_trust_threshold",
                "manifest_v3_tamper"):
        # Three-phase regime:
        # Cells [0..K1):    all honest (calibration window).
        # Cells [K1..K2):   oc compromised (single compromise).
        # Cells [K2..N):    ch + oc both compromised (double).
        sg_oracle: Any = ServiceGraphOracle(oracle_id="service_graph")
        ch_oracle: Any
        oc_oracle: Any
        if cell_idx < K1:
            # Calibration: all 3 honest.
            ch_oracle = ChangeHistoryOracle(
                oracle_id="change_history")
            oc_oracle = OnCallNotesOracle(oracle_id="oncall_notes")
        elif cell_idx < K2:
            # Single compromise: oc bad, ch+sg honest.
            ch_oracle = ChangeHistoryOracle(
                oracle_id="change_history")
            oc_oracle = CompromisedServiceGraphOracle(
                oracle_id="oncall_notes")
        else:
            # Double compromise: ch + oc both bad, sg honest.
            ch_oracle = CompromisedServiceGraphOracle(
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
    if bank == "mis_trust_shift":
        # service_graph: TRANSIENTLY emits a wrong vote on cell 0..2
        # (flaky oracle) but recovers afterwards.  W33 may
        # incorrectly de-trust it because of the transient noise.
        sg_oracle = (
            CompromisedServiceGraphOracle(oracle_id="service_graph")
            if cell_idx < 3
            else ServiceGraphOracle(oracle_id="service_graph"))
        return (
            OracleRegistration(
                oracle=sg_oracle,
                trust_prior=1.0, role_label="service_graph"),
            OracleRegistration(
                oracle=ChangeHistoryOracle(oracle_id="change_history"),
                trust_prior=1.0, role_label="change_history"),
            OracleRegistration(
                oracle=OnCallNotesOracle(oracle_id="oncall_notes"),
                trust_prior=1.0, role_label="oncall_notes"),
        )
    raise ValueError(f"unknown phase80 bank {bank!r}")


# ---------------------------------------------------------------------------
# Per-cell record + correctness helper
# ---------------------------------------------------------------------------


def _gold_for_cell(scenario: Any) -> tuple[str, ...]:
    """Extract the gold (services) tuple for a phase67 multi-round
    scenario.
    """
    return tuple(sorted(scenario.gold_services_pair))


def _is_correct(answer: dict[str, Any], gold: tuple[str, ...]) -> bool:
    services = tuple(sorted(set(answer.get("services", ()))))
    return services == tuple(sorted(set(gold)))


def _is_ratified(answer: dict[str, Any]) -> bool:
    """Returns True if the W33 / W21 answer commits to a non-empty
    proper subset of the admitted tags (i.e. ratified, not abstained).
    """
    services = answer.get("services", ())
    return bool(services)


@dataclasses.dataclass
class _Phase80Record:
    cell_idx: int
    expected: tuple[str, ...]
    correct_w21: bool
    correct_w33: bool
    ratified_w21: bool
    ratified_w33: bool
    w21_branch: str
    w33_branch: str
    w21_visible: int
    w33_visible: int
    w33_overhead: int
    w33_n_detrusted_oracles: int
    w33_oracle_trust_state: tuple[tuple[str, float], ...]
    w33_n_structured_bits: int
    w33_cram_factor: float


# ---------------------------------------------------------------------------
# Build the W33 + W21 stacks for a given bank
# ---------------------------------------------------------------------------


def _build_phase80_stacks(
        *,
        bank: str,
        T_decoder: int | None,
        n_eval: int,
        bank_seed: int,
        bank_replicates: int,
        # W33 knobs
        trust_ewma_enabled: bool,
        manifest_v3_disabled: bool,
        trust_trajectory_window: int,
        trust_threshold: float,
        ewma_alpha: float,
) -> dict[str, Any]:
    """Build the W21 baseline and W33-wrapped stacks on the same
    scenarios.

    bank_replicates controls how many replicates per family the
    underlying phase67 bench produces; we set it large enough that
    n_eval * (4 families) replicates cover the requested n_eval cells.
    """
    schema = _stable_schema_capsule()
    # phase67's outside_resolves produces 4 families × n_replicates
    # scenarios; we need len(scenarios) >= n_eval.  Ensure
    # bank_replicates >= ceil(n_eval / 4).
    n_repl = max(int(bank_replicates), (int(n_eval) + 3) // 4)
    scenarios = build_phase67_bank(
        bank="outside_resolves", n_replicates=n_repl,
        seed=bank_seed)
    if len(scenarios) > n_eval:
        scenarios = scenarios[:n_eval]
    return {
        "schema": schema,
        "scenarios": scenarios,
    }


# ---------------------------------------------------------------------------
# Run one cell for both W21 and W33 (W33 wraps a fresh inner W21 per
# cell to keep oracle registrations bank-cell-specific).
# ---------------------------------------------------------------------------


def _scenario_to_per_round_handoffs(scenario: Any
                                    ) -> Sequence[Sequence[_DecodedHandoff]]:
    """Convert a phase67 MultiRoundScenario to the per_round_handoffs
    shape expected by the W21 disambiguator.  Routes only the handoffs
    addressed to the auditor role (which is what the W19/W20/W21
    decoders consume).
    """
    from vision_mvp.experiments.phase52_team_coord import (
        ROLE_AUDITOR)
    round1: list[_DecodedHandoff] = []
    for src, emissions in scenario.round1_emissions.items():
        for kind, payload in emissions:
            # Phase58 emissions are tuples (kind, payload) — every
            # emission is implicitly directed at the auditor.
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
    """Build a fresh W21 disambiguator with the given registrations."""
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


# ---------------------------------------------------------------------------
# The phase80 runner
# ---------------------------------------------------------------------------


def run_phase80(
        *,
        bank: str = "compromised_shift",
        n_eval: int = 16,
        bank_seed: int = 11,
        bank_replicates: int = 2,
        T_decoder: int | None = None,
        # W33 knobs
        trust_ewma_enabled: bool = True,
        manifest_v3_disabled: bool = False,
        trust_trajectory_window: int = (
            W33_DEFAULT_TRUST_TRAJECTORY_WINDOW),
        trust_threshold: float = W33_DEFAULT_TRUST_THRESHOLD,
        ewma_alpha: float = W33_DEFAULT_EWMA_ALPHA,
        quorum_min: int = 2,
) -> dict[str, Any]:
    """Run one phase80 sub-bank end-to-end.

    Returns a summary dict with per-cell + aggregate metrics for the
    W21 baseline and the W33 trust-EWMA stack.
    """
    stacks = _build_phase80_stacks(
        bank=bank, T_decoder=T_decoder, n_eval=n_eval,
        bank_seed=bank_seed, bank_replicates=bank_replicates,
        trust_ewma_enabled=trust_ewma_enabled,
        manifest_v3_disabled=manifest_v3_disabled,
        trust_trajectory_window=trust_trajectory_window,
        trust_threshold=trust_threshold,
        ewma_alpha=ewma_alpha,
    )
    schema = stacks["schema"]
    scenarios = stacks["scenarios"]

    # Build the W33 orchestrator's persistent registry (shared across
    # cells so the EWMA state accumulates).
    # On compromised_shift, the calibration anchor is service_graph
    # — the always-honest oracle whose vote forms the reference for
    # the per-oracle agreement signal.  This makes the trust signal
    # robust to double-compromise of the other two oracles.
    if bank in ("compromised_shift", "frozen_trust_threshold",
                "manifest_v3_tamper", "mis_trust_shift"):
        anchor_ids: tuple[str, ...] = ("service_graph",)
    else:
        anchor_ids = ()
    if (trust_ewma_enabled or not manifest_v3_disabled
            or trust_trajectory_window > 0):
        registry_w33 = build_trust_ewma_registry(
            schema=schema,
            registered_oracle_ids=("service_graph", "change_history",
                                    "oncall_notes"),
            anchor_oracle_ids=anchor_ids,
            trust_ewma_enabled=trust_ewma_enabled,
            manifest_v3_disabled=manifest_v3_disabled,
            trust_trajectory_window=trust_trajectory_window,
            trust_threshold=trust_threshold,
            ewma_alpha=ewma_alpha,
        )
    else:
        registry_w33 = build_trivial_trust_ewma_registry(
            schema=schema,
            registered_oracle_ids=("service_graph", "change_history",
                                    "oncall_notes"),
        )

    # Build a SHARED W33 orchestrator with a placeholder W21 inner;
    # we will swap in the per-cell W21 inner each iteration.
    placeholder_w21 = _build_w21_disambiguator(
        T_decoder=T_decoder,
        registrations=_build_oracle_registrations_for_cell(
            bank=bank, cell_idx=0, n_total=n_eval),
        quorum_min=int(quorum_min),
    )
    orch_w33 = TrustEWMATrackedMultiOracleOrchestrator(
        inner=placeholder_w21,
        registry=registry_w33,
        enabled=True,
        require_w33_verification=True,
    )

    records: list[_Phase80Record] = []
    n_eval_actual = min(n_eval, len(scenarios))

    for cell_idx in range(n_eval_actual):
        scenario = scenarios[cell_idx]
        gold = _gold_for_cell(scenario)
        per_rounds = _scenario_to_per_round_handoffs(scenario)

        # Build per-cell oracle registrations (bank-specific).
        registrations = _build_oracle_registrations_for_cell(
            bank=bank, cell_idx=int(cell_idx), n_total=n_eval_actual)

        # W21 baseline arm (fresh disambiguator per cell — no state).
        w21_inner = _build_w21_disambiguator(
            T_decoder=T_decoder,
            registrations=registrations,
            quorum_min=int(quorum_min),
        )
        w21_out = w21_inner.decode_rounds(per_rounds)
        w21_result = w21_inner.last_result
        w21_branch = (str(w21_result.decoder_branch)
                       if w21_result is not None else "")
        w21_visible = int(w21_result.n_outside_tokens_total
                            if w21_result is not None else 0)
        ratified_w21 = _is_ratified(w21_out)
        correct_w21 = _is_correct(w21_out, gold) if ratified_w21 else False

        # W33 arm — swap in the same per-cell W21 inner so the
        # registrations are correct.
        w33_inner = _build_w21_disambiguator(
            T_decoder=T_decoder,
            registrations=registrations,
            quorum_min=int(quorum_min),
        )
        orch_w33.inner = w33_inner
        w33_out = orch_w33.decode_rounds(per_rounds)
        w33_result = orch_w33.last_result
        w33_branch = (str(w33_result.decoder_branch)
                       if w33_result is not None else "")
        w33_visible = int(w33_result.n_w33_visible_tokens
                            if w33_result is not None else 0)
        w33_overhead = int(w33_result.n_w33_overhead_tokens
                            if w33_result is not None else 0)
        w33_n_detrusted = int(w33_result.n_detrusted_oracles
                                 if w33_result is not None else 0)
        w33_oracle_trust = (
            tuple(w33_result.oracle_trust_state)
            if w33_result is not None else ())
        w33_n_bits = int(w33_result.n_structured_bits
                           if w33_result is not None else 0)
        w33_cram = float(w33_result.cram_factor_w33
                           if w33_result is not None else 0.0)
        ratified_w33 = _is_ratified(w33_out)
        correct_w33 = _is_correct(w33_out, gold) if ratified_w33 else False

        records.append(_Phase80Record(
            cell_idx=int(cell_idx),
            expected=tuple(gold),
            correct_w21=bool(correct_w21),
            correct_w33=bool(correct_w33),
            ratified_w21=bool(ratified_w21),
            ratified_w33=bool(ratified_w33),
            w21_branch=str(w21_branch),
            w33_branch=str(w33_branch),
            w21_visible=int(w21_visible),
            w33_visible=int(w33_visible),
            w33_overhead=int(w33_overhead),
            w33_n_detrusted_oracles=int(w33_n_detrusted),
            w33_oracle_trust_state=w33_oracle_trust,
            w33_n_structured_bits=int(w33_n_bits),
            w33_cram_factor=float(w33_cram),
        ))

    # Aggregate metrics.
    n = len(records)
    n_correct_w21 = sum(1 for r in records if r.correct_w21)
    n_correct_w33 = sum(1 for r in records if r.correct_w33)
    n_ratified_w21 = sum(1 for r in records if r.ratified_w21)
    n_ratified_w33 = sum(1 for r in records if r.ratified_w33)
    correctness_w21 = n_correct_w21 / n if n > 0 else 0.0
    correctness_w33 = n_correct_w33 / n if n > 0 else 0.0
    # Trust precision = correct/ratified (ratified-only).
    trust_prec_w21 = (n_correct_w21 / n_ratified_w21
                       if n_ratified_w21 > 0 else 1.0)
    trust_prec_w33 = (n_correct_w33 / n_ratified_w33
                       if n_ratified_w33 > 0 else 1.0)
    total_w21_visible = sum(int(r.w21_visible) for r in records)
    total_w33_visible = sum(int(r.w33_visible) for r in records)
    total_w33_overhead = sum(int(r.w33_overhead) for r in records)
    mean_w21_visible = total_w21_visible / n if n > 0 else 0.0
    mean_w33_visible = total_w33_visible / n if n > 0 else 0.0
    mean_w33_overhead = total_w33_overhead / n if n > 0 else 0.0
    max_w33_overhead = (max(int(r.w33_overhead) for r in records)
                          if records else 0)

    return {
        "bank": str(bank),
        "n_eval": int(n),
        "bank_seed": int(bank_seed),
        "trust_ewma_enabled": bool(trust_ewma_enabled),
        "manifest_v3_disabled": bool(manifest_v3_disabled),
        "trust_trajectory_window": int(trust_trajectory_window),
        "trust_threshold": float(trust_threshold),
        "ewma_alpha": float(ewma_alpha),
        "quorum_min": int(quorum_min),
        "n_correct_w21": int(n_correct_w21),
        "n_correct_w33": int(n_correct_w33),
        "n_ratified_w21": int(n_ratified_w21),
        "n_ratified_w33": int(n_ratified_w33),
        "correctness_ratified_rate_w21": round(correctness_w21, 4),
        "correctness_ratified_rate_w33": round(correctness_w33, 4),
        "trust_precision_w21": round(trust_prec_w21, 4),
        "trust_precision_w33": round(trust_prec_w33, 4),
        "delta_correctness_w33_w21": round(
            correctness_w33 - correctness_w21, 4),
        "delta_trust_precision_w33_w21": round(
            trust_prec_w33 - trust_prec_w21, 4),
        "mean_total_w21_visible_tokens": round(mean_w21_visible, 4),
        "mean_total_w33_visible_tokens": round(mean_w33_visible, 4),
        "mean_overhead_w33_per_cell": round(mean_w33_overhead, 4),
        "max_overhead_w33_per_cell": int(max_w33_overhead),
        "n_oracles_detrusted": int(
            registry_w33.n_oracles_detrusted),
        "n_trust_ewma_updates": int(
            registry_w33.n_trust_ewma_updates),
        "n_w33_registered": int(registry_w33.n_w33_registered),
        "n_w33_rejected": int(registry_w33.n_w33_rejected),
        "byte_equivalent_w33_w21": bool(
            mean_w21_visible == mean_w33_visible
            and correctness_w21 == correctness_w33
            and trust_prec_w21 == trust_prec_w33),
        "records": [dataclasses.asdict(r) for r in records],
        "final_oracle_trust_state": (
            list(records[-1].w33_oracle_trust_state)
            if records else []),
    }


# ---------------------------------------------------------------------------
# Seed sweep — five seeds (matches phase79 convention)
# ---------------------------------------------------------------------------


def run_phase80_seed_sweep(
        *,
        bank: str = "compromised_shift",
        n_eval: int = 16,
        seeds: Sequence[int] = (11, 17, 23, 29, 31),
        T_decoder: int | None = None,
        # W33 knobs
        trust_ewma_enabled: bool = True,
        manifest_v3_disabled: bool = False,
        trust_trajectory_window: int = (
            W33_DEFAULT_TRUST_TRAJECTORY_WINDOW),
        trust_threshold: float = W33_DEFAULT_TRUST_THRESHOLD,
        ewma_alpha: float = W33_DEFAULT_EWMA_ALPHA,
        quorum_min: int = 2,
) -> dict[str, Any]:
    seed_results: list[dict[str, Any]] = []
    for s in seeds:
        r = run_phase80(
            bank=bank, n_eval=n_eval, bank_seed=int(s),
            T_decoder=T_decoder,
            trust_ewma_enabled=trust_ewma_enabled,
            manifest_v3_disabled=manifest_v3_disabled,
            trust_trajectory_window=trust_trajectory_window,
            trust_threshold=trust_threshold,
            ewma_alpha=ewma_alpha,
            quorum_min=int(quorum_min),
        )
        # Drop the per-cell records from the seed-sweep summary to
        # keep file size sane; the per-cell records are still in r.
        seed_results.append({k: v for k, v in r.items()
                              if k != "records"})
    deltas_corr = [s["delta_correctness_w33_w21"]
                    for s in seed_results]
    deltas_trust = [s["delta_trust_precision_w33_w21"]
                     for s in seed_results]
    min_delta_corr = min(deltas_corr) if deltas_corr else 0.0
    max_delta_corr = max(deltas_corr) if deltas_corr else 0.0
    min_delta_trust = min(deltas_trust) if deltas_trust else 0.0
    max_delta_trust = max(deltas_trust) if deltas_trust else 0.0
    return {
        "bank": str(bank),
        "seeds": list(int(s) for s in seeds),
        "min_delta_correctness_w33_w21": round(min_delta_corr, 4),
        "max_delta_correctness_w33_w21": round(max_delta_corr, 4),
        "min_delta_trust_precision_w33_w21": round(min_delta_trust, 4),
        "max_delta_trust_precision_w33_w21": round(max_delta_trust, 4),
        "min_trust_precision_w33": round(
            min(s["trust_precision_w33"]
                for s in seed_results) if seed_results else 1.0, 4),
        "min_correctness_ratified_rate_w33": round(
            min(s["correctness_ratified_rate_w33"]
                for s in seed_results) if seed_results else 0.0, 4),
        "max_overhead_w33_per_cell": (
            max(int(s["max_overhead_w33_per_cell"])
                for s in seed_results) if seed_results else 0),
        "all_byte_equivalent_w33_w21": (
            all(s["byte_equivalent_w33_w21"]
                for s in seed_results) if seed_results else False),
        "seed_results": seed_results,
    }


# ---------------------------------------------------------------------------
# Manifest-v3 tamper sweep
# ---------------------------------------------------------------------------


def run_phase80_manifest_v3_tamper_sweep(
        *,
        bank: str = "manifest_v3_tamper",
        n_eval: int = 16,
        seeds: Sequence[int] = (11, 17, 23, 29, 31),
        T_decoder: int | None = None,
) -> dict[str, Any]:
    """Run the manifest-v3 tamper sweep.  For each ratified cell, we
    apply five named tampers and verify each is rejected by the
    `verify_trust_ewma_ratification` verifier.
    """
    schema = _stable_schema_capsule()
    seed_results: list[dict[str, Any]] = []
    cumulative_attempts = 0
    cumulative_rejected = 0
    for seed in seeds:
        r = run_phase80(
            bank=bank, n_eval=n_eval, bank_seed=int(seed),
            T_decoder=T_decoder,
            trust_ewma_enabled=True,
            manifest_v3_disabled=False,
            trust_trajectory_window=(
                W33_DEFAULT_TRUST_TRAJECTORY_WINDOW),
            trust_threshold=W33_DEFAULT_TRUST_THRESHOLD,
            ewma_alpha=W33_DEFAULT_EWMA_ALPHA,
        )
        # Reconstruct the W33 orchestrator's per-cell envelopes by
        # re-running with the same configuration but capturing the
        # envelopes.  Since the original orch is gone, we re-run.
        from vision_mvp.experiments.phase80_trust_ewma_tracked import (
            _build_phase80_stacks,
            _build_w21_disambiguator,
            _build_oracle_registrations_for_cell,
            _scenario_to_per_round_handoffs)
        stacks = _build_phase80_stacks(
            bank=bank, T_decoder=T_decoder, n_eval=n_eval,
            bank_seed=int(seed), bank_replicates=2,
            trust_ewma_enabled=True, manifest_v3_disabled=False,
            trust_trajectory_window=(
                W33_DEFAULT_TRUST_TRAJECTORY_WINDOW),
            trust_threshold=W33_DEFAULT_TRUST_THRESHOLD,
            ewma_alpha=W33_DEFAULT_EWMA_ALPHA,
        )
        scenarios = stacks["scenarios"]
        if len(scenarios) > n_eval:
            scenarios = scenarios[:n_eval]
        registry = build_trust_ewma_registry(
            schema=schema,
            registered_oracle_ids=("service_graph", "change_history",
                                    "oncall_notes"),
            anchor_oracle_ids=("service_graph",),
            trust_ewma_enabled=True,
            manifest_v3_disabled=False,
            trust_trajectory_window=(
                W33_DEFAULT_TRUST_TRAJECTORY_WINDOW),
            trust_threshold=W33_DEFAULT_TRUST_THRESHOLD,
            ewma_alpha=W33_DEFAULT_EWMA_ALPHA,
        )
        placeholder_w21 = _build_w21_disambiguator(
            T_decoder=T_decoder,
            registrations=_build_oracle_registrations_for_cell(
                bank=bank, cell_idx=0, n_total=len(scenarios)),
        )
        orch = TrustEWMATrackedMultiOracleOrchestrator(
            inner=placeholder_w21, registry=registry,
            enabled=True, require_w33_verification=True,
        )
        n_attempts = 0
        n_rejected = 0
        for cell_idx in range(min(len(scenarios), n_eval)):
            registrations = _build_oracle_registrations_for_cell(
                bank=bank, cell_idx=int(cell_idx),
                n_total=len(scenarios))
            inner = _build_w21_disambiguator(
                T_decoder=T_decoder, registrations=registrations)
            orch.inner = inner
            scenario = scenarios[cell_idx]
            per_rounds = _scenario_to_per_round_handoffs(scenario)
            orch.decode_rounds(per_rounds)
            env = orch.last_envelope
            if env is None or not env.wire_required:
                continue
            # Apply 5 named tampers.
            tampers = _five_named_tampers(env)
            for tamper_id, tampered_env in tampers:
                outcome = verify_trust_ewma_ratification(
                    tampered_env,
                    registered_schema=schema,
                    registered_parent_cid=str(env.parent_cid),
                    registered_oracle_ids=frozenset(
                        registry.registered_oracle_ids),
                    registered_trust_trajectory_window=(
                        int(registry.trust_trajectory_window)),
                    registered_oracle_trust_state_cid=(
                        env.oracle_trust_state_cid),
                )
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
        cumulative_attempts += n_attempts
        cumulative_rejected += n_rejected

    return {
        "bank": str(bank),
        "seeds": list(int(s) for s in seeds),
        "n_tamper_attempts_total": int(cumulative_attempts),
        "n_tamper_rejected_total": int(cumulative_rejected),
        "reject_rate_total": (
            round(cumulative_rejected / cumulative_attempts, 4)
            if cumulative_attempts > 0 else 0.0),
        "seed_results": seed_results,
    }


def _five_named_tampers(
        env: TrustEWMARatificationEnvelope,
) -> list[tuple[str, TrustEWMARatificationEnvelope]]:
    """Apply five named tampers to a sealed W33 envelope and return
    each tampered version.  Each tamper is detectable by exactly one
    failure mode.
    """
    out: list[tuple[str, TrustEWMARatificationEnvelope]] = []

    # T1 — oracle_trust_state byte corruption
    t1_state = list(env.oracle_trust_state)
    if t1_state:
        oid, t = t1_state[0]
        t1_state[0] = (oid, max(0.0, min(1.0, float(t) - 0.001)))
    new_state_cid = _compute_oracle_trust_state_cid(
        oracle_to_trust=tuple(t1_state))
    new_manifest_v3 = _compute_w33_manifest_v3_cid(
        parent_cid=env.parent_cid,
        oracle_trust_state_cid=new_state_cid,
        trust_trajectory_cid=env.trust_trajectory_cid,
        trust_route_audit_cid=env.trust_route_audit_cid,
    )
    new_outer = _compute_w33_outer_cid(
        schema_version=env.schema_version,
        schema_cid=env.schema_cid,
        parent_cid=env.parent_cid,
        oracle_trust_state_cid=new_state_cid,
        trust_trajectory_cid=env.trust_trajectory_cid,
        manifest_v3_cid=new_manifest_v3,
        cell_index=int(env.cell_index),
    )
    t1_env = dataclasses.replace(
        env,
        oracle_trust_state=tuple(t1_state),
        oracle_trust_state_cid=env.oracle_trust_state_cid,  # leave OLD
        manifest_v3_cid=new_manifest_v3,  # consistent with new state
        w33_cid=env.w33_cid,
    )
    out.append(("T1_oracle_trust_state_cid_mismatch", t1_env))

    # T2 — manifest_v3_cid byte corruption
    t2_env = dataclasses.replace(
        env,
        manifest_v3_cid=("0" * 64),
    )
    out.append(("T2_manifest_v3_cid_mismatch", t2_env))

    # T3 — trust_trajectory entry observed_quorum_agreement out of range
    if env.trust_trajectory:
        bad_entry = dataclasses.replace(
            env.trust_trajectory[0],
            observed_quorum_agreement=2.0,
        )
        new_traj = (bad_entry,) + tuple(env.trust_trajectory[1:])
        t3_env = dataclasses.replace(env, trust_trajectory=new_traj)
        out.append(("T3_trust_trajectory_observed_out_of_range",
                     t3_env))
    else:
        # Fallback — use a fresh out-of-range entry
        bad_entry = TrustTrajectoryEntry(
            cell_idx=0, oracle_id="service_graph",
            observed_quorum_agreement=2.0,
            ewma_trust_after=1.0)
        t3_env = dataclasses.replace(
            env, trust_trajectory=(bad_entry,))
        out.append(("T3_trust_trajectory_observed_out_of_range",
                     t3_env))

    # T4 — oracle_trust_state ewma out of range
    t4_state = list(env.oracle_trust_state)
    if t4_state:
        oid, _ = t4_state[0]
        t4_state[0] = (oid, 2.0)
    else:
        t4_state = [("service_graph", 2.0)]
    t4_env = dataclasses.replace(
        env, oracle_trust_state=tuple(t4_state))
    out.append(("T4_oracle_trust_state_ewma_out_of_range", t4_env))

    # T5 — outer w33_cid byte corruption
    t5_env = dataclasses.replace(env, w33_cid="0" * 64)
    out.append(("T5_w33_outer_cid_mismatch", t5_env))

    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Phase 80 — Trust-EWMA-tracked multi-oracle "
                    "adjudication (W33).")
    p.add_argument("--bank", default="compromised_shift",
                    choices=("trivial_w33", "compromised_shift",
                             "no_trust_shift", "frozen_trust_threshold",
                             "mis_trust_shift", "manifest_v3_tamper"))
    p.add_argument("--n-eval", type=int, default=16)
    p.add_argument("--bank-seed", type=int, default=11)
    p.add_argument("--seed-sweep", action="store_true")
    p.add_argument("--manifest-v3-tamper-sweep", action="store_true")
    p.add_argument("--trust-threshold", type=float,
                    default=W33_DEFAULT_TRUST_THRESHOLD)
    p.add_argument("--ewma-alpha", type=float,
                    default=W33_DEFAULT_EWMA_ALPHA)
    p.add_argument("--quorum-min", type=int, default=2)
    p.add_argument("--out", default=None)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _arg_parser().parse_args(argv)
    if args.manifest_v3_tamper_sweep:
        result = run_phase80_manifest_v3_tamper_sweep(
            bank=args.bank, n_eval=int(args.n_eval))
    elif args.seed_sweep:
        # Trivial / frozen_trust_threshold: trust_ewma_enabled=False
        # OR trust_threshold=0.0 (ties W21 byte-for-byte by design).
        if args.bank == "trivial_w33":
            kwargs = dict(
                trust_ewma_enabled=False,
                manifest_v3_disabled=True,
                trust_trajectory_window=0,
                trust_threshold=W33_DEFAULT_TRUST_THRESHOLD,
                ewma_alpha=W33_DEFAULT_EWMA_ALPHA,
            )
        elif args.bank == "frozen_trust_threshold":
            kwargs = dict(
                trust_ewma_enabled=True,
                manifest_v3_disabled=False,
                trust_trajectory_window=(
                    W33_DEFAULT_TRUST_TRAJECTORY_WINDOW),
                trust_threshold=0.0,
                ewma_alpha=W33_DEFAULT_EWMA_ALPHA,
            )
        else:
            kwargs = dict(
                trust_ewma_enabled=True,
                manifest_v3_disabled=False,
                trust_trajectory_window=(
                    W33_DEFAULT_TRUST_TRAJECTORY_WINDOW),
                trust_threshold=float(args.trust_threshold),
                ewma_alpha=float(args.ewma_alpha),
            )
        result = run_phase80_seed_sweep(
            bank=args.bank, n_eval=int(args.n_eval),
            quorum_min=int(args.quorum_min),
            **kwargs)
    else:
        result = run_phase80(
            bank=args.bank, n_eval=int(args.n_eval),
            bank_seed=int(args.bank_seed),
            quorum_min=int(args.quorum_min),
            trust_threshold=float(args.trust_threshold),
            ewma_alpha=float(args.ewma_alpha),
        )
    out_path = args.out
    if out_path is None:
        os.makedirs("vision_mvp/experiments/artifacts/phase80",
                     exist_ok=True)
        suffix = ("manifest_v3_tamper_seed_sweep"
                   if args.manifest_v3_tamper_sweep
                   else "seed_sweep" if args.seed_sweep
                   else "single_run")
        out_path = (f"vision_mvp/experiments/artifacts/phase80/"
                     f"{args.bank}_{suffix}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"phase80: wrote {out_path}")
    print(json.dumps({k: v for k, v in result.items()
                       if k not in ("records", "seed_results")},
                      indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
