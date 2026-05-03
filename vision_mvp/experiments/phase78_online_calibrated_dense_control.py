"""Phase 78 — online self-calibrated geometry-aware dense control +
sealed prior trajectory + adaptive threshold + W31 manifest CID
(SDK v3.32, W31).

Follow-up to SDK v3.31 (W30).  W30 introduced the calibrated geometry
layer with multi-stride basis history, per-partition calibration
prior, cross-host disagreement-routing, and ancestor-chain causal
binding.  W30 was a STRONG SUCCESS but four named open conjectures
inherited forward (W30-C-PRIOR-LEARNING,
W30-C-CROSS-HOST-VARIANCE-LIVE-MAGNITUDE-LIVE, W30-C-NATIVE-LATENT,
W30-C-MULTI-HOST).

W31 (this phase) wraps the W30 ``CalibratedGeometryOrchestrator``
with three NEW audited proxies at the capsule layer:

* **online running-mean prior update** — after each ratified cell, a
  deterministic per-cell agreement signal (cell ratified AND no
  cross-host disagreement) drives the existing closed-form
  :func:`update_partition_calibration_running_mean` primitive on the
  inner W30 registry.  Mechanism for the W30-C-PRIOR-LEARNING
  discharge.
* **adaptive threshold** — the W30 calibration_threshold is no longer
  a static constant; it is recomputed as the closed-form clipped
  median of the live calibration vector after each update.
* **sealed prior + threshold trajectory + W31 manifest CID** — the
  W31 envelope carries content-addressed CIDs over the
  ``(cell_idx, partition_id, observed_agreement, prior_after)``
  trajectory, the threshold trajectory, and a manifest hash over all
  inner-component CIDs (cross-component swap detection).

Phase-78 sub-banks
==================

Six pre-committed sub-banks (five synthetic + one cross-host live):

* **R-78-TRIVIAL-W31** (H2 anchor; W31-Λ-trivial-online).
  ``online_enabled=False``, ``manifest_disabled=True``,
  ``trajectory_window=0`` ⇒ W31 reduces to W30 byte-for-byte.
* **R-78-NONSTATIONARY-PRIOR** (H6 main load-bearing claim;
  discharge of W30-C-PRIOR-LEARNING).  Inverted-partition regime
  (CYCLIC=PARTIAL, LINEAR=FULL); W30 baseline uses uniform priors
  (1.0, 1.0, 1.0) and never reroutes; W31 with online learning
  observes per-cell disagreement on CYCLIC cells, drives the prior
  for CYCLIC down, eventually crosses the adaptive threshold and
  reroutes to LINEAR — strictly improving correctness over W30.
  Pre-committed bar: Δ ≥ +0.10 across 5/5 seeds.
* **R-78-ADAPTIVE-THRESHOLD** (H7).  Same regime; W31 with
  ``adaptive_threshold=True`` strictly outperforms W31 with
  ``adaptive_threshold=False`` (frozen threshold at the W30 default
  0.5) on a regime where the prior distribution makes a fixed 0.5
  threshold suboptimal.
* **R-78-MANIFEST-TAMPER** (H8).  Cross-component swap: take the
  ``prior_trajectory_cid`` from cell C2's W31 envelope and stuff
  it into cell C1's W31 envelope (with each component CID still
  internally consistent); the W31 manifest CID detects the swap.
* **R-78-NO-DRIFT** (W31-Λ-no-drift falsifier).  Stationary regime
  where every partition has the same agreement-rate; online-learned
  prior converges to (≈ same value), no help over W30 baseline.
* **R-78-FROZEN-THRESHOLD** (W31-Λ-frozen-threshold falsifier).
  Online learning enabled but ``adaptive_threshold=False``;
  isolates the adaptive contribution from the online-prior
  contribution.
* **R-78-XLLM-LIVE** (S1/S2 best-effort).  Live cross-architecture
  LLM probes (gemma2:9b localhost + qwen2.5:14b on 192.168.12.191)
  on a regime crafted for surface-level family-specific
  disagreement at temperature 0.

The bench's apples-to-apples comparison is:

  * **W30 baseline**     — W30 calibrated geometry orchestrator with
                            uniform priors (no online learning).
  * **W31 online**       — W30 wrapped by W31 online orchestrator.
  * **W31 frozen**       — W31 with adaptive_threshold=False.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import sys
import urllib.request
from typing import Any

from vision_mvp.coordpy.team_coord import (
    OracleRegistration, SchemaCapsule,
    build_incident_triage_schema_capsule,
    SharedFanoutRegistry,
    ChainPersistedFanoutRegistry,
    SharedMultiChainPool,
    compute_input_signature_cid,
    # W28
    EnsembleVerifiedMultiChainOrchestrator,
    build_default_ensemble_registry,
    # W29
    GeometryPartitionedRatificationEnvelope,
    GeometryPartitionRegistry,
    GeometryPartitionedOrchestrator,
    build_three_partition_registry,
    W29_PARTITION_LINEAR, W29_PARTITION_HIERARCHICAL,
    W29_PARTITION_CYCLIC,
    # W30
    BasisHistory, AncestorChain, PartitionCalibrationVector,
    CalibratedGeometryRatificationEnvelope,
    CalibratedGeometryRegistry,
    CalibratedGeometryOrchestrator,
    verify_calibrated_geometry_ratification,
    update_partition_calibration_running_mean,
    build_trivial_calibrated_registry,
    build_calibrated_registry,
    # W31
    PriorTrajectoryEntry,
    OnlineCalibratedRatificationEnvelope,
    OnlineCalibratedRegistry,
    W31OnlineResult,
    OnlineCalibratedOrchestrator,
    verify_online_calibrated_ratification,
    derive_per_cell_agreement_signal,
    compute_adaptive_threshold,
    build_trivial_online_registry,
    build_online_calibrated_registry,
    W31_ONLINE_SCHEMA_VERSION,
    W31_DEFAULT_THRESHOLD_MIN, W31_DEFAULT_THRESHOLD_MAX,
    W31_DEFAULT_TRAJECTORY_WINDOW,
    W31_BRANCH_ONLINE_RESOLVED,
    W31_BRANCH_TRIVIAL_ONLINE_PASSTHROUGH,
    # Oracles
    ServiceGraphOracle, ChangeHistoryOracle,
    _DecodedHandoff,
)

from vision_mvp.experiments.phase74_multi_chain_pivot import (
    build_phase74_bank, build_team_shared_pool,
    _build_partial_service_graph_oracle,
    _build_w27_orchestrator,
    _expected_gold_for_cell,
)
from vision_mvp.experiments.phase75_ensemble_verified_multi_chain import (
    discover_two_host_topology,
    LOCAL_HOST_ID,
    OLLAMA_LOCALHOST, OLLAMA_MAC1,
    _build_w28_orchestrator,
)
from vision_mvp.experiments.phase76_geometry_partitioned_product_manifold import (
    INCIDENT_TRIAGE_AMBIENT_VOCABULARY,
    _build_per_partition_team_pools,
)
from vision_mvp.experiments.phase77_calibrated_dense_control import (
    _build_inverted_per_partition_team_pools,
)


# ---------------------------------------------------------------------------
# W31 visible / branch helpers
# ---------------------------------------------------------------------------


def _w30_visible_from_out(out: dict) -> int:
    return int(out.get("calibrated_geometry", {}).get(
        "n_w30_visible_tokens", 0))


def _w31_visible(out: dict) -> int:
    if "online_calibrated" in out:
        return int(out["online_calibrated"].get(
            "n_w31_visible_tokens", 0))
    return _w30_visible_from_out(out)


def _w31_branch(out: dict) -> str:
    if "online_calibrated" in out:
        return str(out["online_calibrated"].get("decoder_branch", ""))
    return ""


def _w31_ratified(out: dict) -> bool:
    if "online_calibrated" in out:
        return bool(out["online_calibrated"].get("ratified", False))
    return False


def _w31_overhead(out: dict) -> int:
    if "online_calibrated" in out:
        return int(out["online_calibrated"].get(
            "n_w31_overhead_tokens", 0))
    return 0


def _w31_n_structured_bits(out: dict) -> int:
    if "online_calibrated" in out:
        return int(out["online_calibrated"].get("n_structured_bits", 0))
    return 0


def _w31_cram_factor(out: dict) -> float:
    if "online_calibrated" in out:
        return float(out["online_calibrated"].get("cram_factor_w31", 0.0))
    return 0.0


def _w31_prior_after(out: dict) -> float:
    if "online_calibrated" in out:
        return float(out["online_calibrated"].get("prior_after", 0.0))
    return 0.0


def _w31_threshold_after(out: dict) -> float:
    if "online_calibrated" in out:
        return float(out["online_calibrated"].get("threshold_after", 0.0))
    return 0.0


def _w31_observed_agreement(out: dict) -> float:
    if "online_calibrated" in out:
        return float(out["online_calibrated"].get("observed_agreement", 0.0))
    return 0.0


def _w30_effective_partition_id(out: dict) -> int:
    return int(out.get("calibrated_geometry", {}).get(
        "effective_partition_id", 0))


# ---------------------------------------------------------------------------
# Common phase78 stack builder — wraps phase77's W30 stack with W31
# ---------------------------------------------------------------------------


def _build_phase78_stacks(
        *,
        bank: str,
        T_decoder: int | None,
        K_consumers: int,
        n_eval: int,
        signature_period: int,
        bank_seed: int,
        bank_replicates: int,
        chain_persist_window: int,
        max_active_chains: int,
        calibration_stride: int,
        ancestor_window: int,
        w30_calibration_priors: tuple[float, ...],
        w30_high_trust_partition_id: int,
        online_enabled: bool,
        adaptive_threshold: bool,
        manifest_disabled: bool,
        trajectory_window: int,
) -> dict[str, Any]:
    """Build TWO independent stacks (W30 baseline, W30+W31) on the same
    cells.  Returns a dict with the cells, the producer agents, and
    the schema.
    """
    schema = build_incident_triage_schema_capsule()
    if bank in ("nonstationary_prior", "adaptive_threshold",
                "frozen_threshold", "manifest_tamper", "xllm_live"):
        underlying_bank = "divergent_recover"
    elif bank == "no_drift":
        # Stationary regime: every partition gets the FULL oracle, so
        # every cell has agreement_rate ≈ 1.0; online learning has
        # nothing to drive the prior toward — the W31-Λ-no-drift
        # falsifier.
        underlying_bank = "chain_shared"
    else:
        underlying_bank = "chain_shared"

    cells = build_phase74_bank(
        n_replicates=bank_replicates, seed=bank_seed,
        n_cells=n_eval, bank=underlying_bank,
        signature_period=signature_period)

    producer_id = "producer_agent"
    consumer_ids = tuple(f"consumer_{k}" for k in range(K_consumers))

    if chain_persist_window is None:
        chain_persist_window = n_eval

    projection_id_for_consumer = {
        cid: f"proj_{cid}" for cid in consumer_ids}
    projected_tags_for_consumer = {
        cid: ("orders", "payments", "api", "db",
              "storage", "logs_pipeline", "web", "db_query")
        for cid in consumer_ids}

    # W28 inner per-partition pool builders.
    def _build_inner_per_partition(_seed_for_pool: int):
        if bank in ("nonstationary_prior",
                    "adaptive_threshold", "frozen_threshold",
                    "manifest_tamper", "xllm_live"):
            pools = _build_inverted_per_partition_team_pools(
                T_decoder=T_decoder, schema=schema,
                producer_agent_id=producer_id,
                consumer_agent_ids=consumer_ids,
                chain_persist_window=chain_persist_window,
                max_active_chains=max_active_chains,
                signature_period=signature_period,
                projection_id_for_consumer=projection_id_for_consumer,
                projected_tags_for_consumer=projected_tags_for_consumer,
                partial_gold_pair=("orders", "payments"),
            )
        else:
            pools = _build_per_partition_team_pools(
                T_decoder=T_decoder, schema=schema,
                producer_agent_id=producer_id,
                consumer_agent_ids=consumer_ids,
                chain_persist_window=chain_persist_window,
                max_active_chains=max_active_chains,
                signature_period=signature_period,
                projection_id_for_consumer=projection_id_for_consumer,
                projected_tags_for_consumer=projected_tags_for_consumer,
                partial_gold_pair=("orders", "payments"),
                full_oracles_for_cyclic=True,
            )
        inner_per_partition: dict[
            int, EnsembleVerifiedMultiChainOrchestrator] = {}
        for pid, pool in pools.items():
            reg = build_default_ensemble_registry(
                schema=schema, quorum_threshold=1.0,
                local_host_id=LOCAL_HOST_ID)
            inner = _build_w28_orchestrator(
                schema=schema, agent_id=producer_id, is_producer=True,
                producer_agent_id=producer_id,
                consumer_agent_ids=consumer_ids,
                pool=pool, registry=reg)
            inner_per_partition[int(pid)] = inner
        return inner_per_partition

    cycle_window_for_bench = max(8, 2 * int(signature_period))

    def _make_three_partition_registry() -> GeometryPartitionRegistry:
        return build_three_partition_registry(
            schema=schema,
            consumer_order=consumer_ids,
            ambient_vocabulary=INCIDENT_TRIAGE_AMBIENT_VOCABULARY,
            basis_dim=2,
            cycle_window=cycle_window_for_bench,
            local_host_id=LOCAL_HOST_ID,
        )

    # Build TWO inner W28 stacks (one for W30 baseline, one for W30+W31).
    inner_a = _build_inner_per_partition(bank_seed)
    inner_b = _build_inner_per_partition(bank_seed + 9999)

    # Build a shared base W28 inner on the W30 baseline arm.
    base_w28_a = _build_w28_orchestrator(
        schema=schema, agent_id=producer_id, is_producer=True,
        producer_agent_id=producer_id, consumer_agent_ids=consumer_ids,
        pool=build_team_shared_pool(
            T_decoder=T_decoder, schema=schema,
            raw_oracles=(
                (_build_partial_service_graph_oracle(
                    gold_pair=("orders", "payments"),
                    oracle_id="service_graph_partial"),
                 "service_graph"),
                (ChangeHistoryOracle(oracle_id="change_history"),
                 "change_history"),
            ),
            producer_agent_id=producer_id,
            consumer_agent_ids=consumer_ids,
            chain_persist_window=chain_persist_window,
            max_active_chains=max_active_chains,
            projection_id_for_consumer=projection_id_for_consumer,
            projected_tags_for_consumer=projected_tags_for_consumer,
        ),
        registry=build_default_ensemble_registry(
            schema=schema, quorum_threshold=1.0,
            local_host_id=LOCAL_HOST_ID))
    base_w28_b = _build_w28_orchestrator(
        schema=schema, agent_id=producer_id, is_producer=True,
        producer_agent_id=producer_id, consumer_agent_ids=consumer_ids,
        pool=build_team_shared_pool(
            T_decoder=T_decoder, schema=schema,
            raw_oracles=(
                (_build_partial_service_graph_oracle(
                    gold_pair=("orders", "payments"),
                    oracle_id="service_graph_partial"),
                 "service_graph"),
                (ChangeHistoryOracle(oracle_id="change_history"),
                 "change_history"),
            ),
            producer_agent_id=producer_id,
            consumer_agent_ids=consumer_ids,
            chain_persist_window=chain_persist_window,
            max_active_chains=max_active_chains,
            projection_id_for_consumer=projection_id_for_consumer,
            projected_tags_for_consumer=projected_tags_for_consumer,
        ),
        registry=build_default_ensemble_registry(
            schema=schema, quorum_threshold=1.0,
            local_host_id=LOCAL_HOST_ID))

    # W29 stacks.
    registry_w29_a = _make_three_partition_registry()
    registry_w29_b = _make_three_partition_registry()
    producer_w29_a = GeometryPartitionedOrchestrator(
        inner=base_w28_a, registry=registry_w29_a, enabled=True,
        require_partition_verification=True,
        cycle_window=cycle_window_for_bench,
        inner_per_partition=inner_a,
        pre_dispatch_by_partition=True,
    )
    producer_w29_b = GeometryPartitionedOrchestrator(
        inner=base_w28_b, registry=registry_w29_b, enabled=True,
        require_partition_verification=True,
        cycle_window=cycle_window_for_bench,
        inner_per_partition=inner_b,
        pre_dispatch_by_partition=True,
    )

    # W30 baseline (uniform priors — never reroutes by calibration).
    registry_w30_a = build_calibrated_registry(
        schema=schema,
        calibration_stride=int(calibration_stride),
        calibration_priors=tuple(w30_calibration_priors),
        ancestor_window=int(ancestor_window),
        high_trust_partition_id=int(w30_high_trust_partition_id),
        local_host_id=LOCAL_HOST_ID,
    )
    producer_w30_a = CalibratedGeometryOrchestrator(
        inner=producer_w29_a, registry=registry_w30_a,
        enabled=True, require_calibrated_verification=True,
    )

    # W30+W31 stack (online learning on top).
    registry_w30_b = build_calibrated_registry(
        schema=schema,
        calibration_stride=int(calibration_stride),
        calibration_priors=tuple(w30_calibration_priors),
        ancestor_window=int(ancestor_window),
        high_trust_partition_id=int(w30_high_trust_partition_id),
        local_host_id=LOCAL_HOST_ID,
    )
    producer_w30_b = CalibratedGeometryOrchestrator(
        inner=producer_w29_b, registry=registry_w30_b,
        enabled=True, require_calibrated_verification=True,
    )
    if (online_enabled or not manifest_disabled or trajectory_window > 0):
        registry_w31 = build_online_calibrated_registry(
            schema=schema,
            online_enabled=bool(online_enabled),
            adaptive_threshold=bool(adaptive_threshold),
            manifest_disabled=bool(manifest_disabled),
            trajectory_window=int(trajectory_window),
            local_host_id=LOCAL_HOST_ID,
        )
    else:
        registry_w31 = build_trivial_online_registry(
            schema=schema, local_host_id=LOCAL_HOST_ID)
    producer_w31 = OnlineCalibratedOrchestrator(
        inner=producer_w30_b, registry=registry_w31,
        enabled=True, require_w31_verification=True,
    )

    return {
        "schema": schema,
        "cells": cells,
        "underlying_bank": underlying_bank,
        "producer_w30_a": producer_w30_a,  # W30 baseline (no W31)
        "producer_w31": producer_w31,       # W30+W31 stack
        "registry_w30_b": registry_w30_b,   # accessible W30 registry
        "registry_w31": registry_w31,
    }


# ---------------------------------------------------------------------------
# Per-cell record + correctness helper
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _PhaseRecord:
    cell_idx: int
    expected: set[str]
    correct_w30: bool
    correct_w31: bool
    w30_visible: int
    w31_visible: int
    w31_overhead: int
    w30_branch: str
    w31_branch: str
    w30_ratified: bool
    w31_ratified: bool
    w31_n_structured_bits: int
    w31_cram_factor: float
    w30_effective_partition_id: int
    w31_prior_after: float
    w31_threshold_after: float
    w31_observed_agreement: float


def _is_correct(out: dict, expected_set: set[str]) -> bool:
    ans = out.get("answer") or out
    svcs = ans.get("services") if isinstance(ans, dict) else None
    if svcs is None:
        svcs = out.get("services")
    return set(svcs or []) == expected_set


# ---------------------------------------------------------------------------
# Phase 78 runner
# ---------------------------------------------------------------------------


def run_phase78(
        *,
        bank: str = "nonstationary_prior",
        T_decoder: int | None = None,
        K_consumers: int = 3,
        n_eval: int = 16,
        bank_replicates: int = 4,
        bank_seed: int = 11,
        chain_persist_window: int | None = None,
        max_active_chains: int = 8,
        signature_period: int = 4,
        calibration_stride: int = 8,
        ancestor_window: int = 4,
        verbose: bool = False,
) -> dict[str, Any]:
    """Run one Phase-78 sub-bank and return a results dict comparing
    W30 baseline vs W30+W31 on the same regime.

    ``bank`` is one of:

      * ``"trivial_w31"``         — H2 byte-equivalence anchor.
      * ``"nonstationary_prior"`` — H6 main load-bearing claim.
      * ``"adaptive_threshold"``  — H7 isolating axis.
      * ``"manifest_tamper"``     — H8 cross-component tamper detection.
      * ``"no_drift"``            — W31-Λ-no-drift falsifier.
      * ``"frozen_threshold"``    — W31-Λ-frozen-threshold falsifier.
      * ``"xllm_live"``           — S1 best-effort live cross-arch.
    """
    if chain_persist_window is None:
        chain_persist_window = n_eval

    # Pick W30 + W31 knobs per bank.
    if bank == "trivial_w31":
        w30_priors = (1.0, 1.0, 1.0)
        w30_high_trust = W29_PARTITION_LINEAR
        online_enabled = False
        adaptive_threshold = False
        manifest_disabled = True
        trajectory_window = 0
    elif bank == "nonstationary_prior":
        # W30 baseline: uniform priors → never reroute.
        # W31: online learning + adaptive threshold; observes
        # CYCLIC's low agreement, drives prior down, reroutes to LINEAR.
        w30_priors = (1.0, 1.0, 1.0)
        w30_high_trust = W29_PARTITION_LINEAR
        online_enabled = True
        adaptive_threshold = True
        manifest_disabled = False
        trajectory_window = 16
    elif bank == "adaptive_threshold":
        # Same as nonstationary_prior with adaptive_threshold=True.
        w30_priors = (1.0, 1.0, 1.0)
        w30_high_trust = W29_PARTITION_LINEAR
        online_enabled = True
        adaptive_threshold = True
        manifest_disabled = False
        trajectory_window = 16
    elif bank == "frozen_threshold":
        # Online learning, but adaptive_threshold=False.
        w30_priors = (1.0, 1.0, 1.0)
        w30_high_trust = W29_PARTITION_LINEAR
        online_enabled = True
        adaptive_threshold = False
        manifest_disabled = False
        trajectory_window = 16
    elif bank == "no_drift":
        # Stationary (every partition has FULL oracle); online learning
        # converges to ≈ same prior; no help over baseline.
        w30_priors = (1.0, 1.0, 1.0)
        w30_high_trust = W29_PARTITION_LINEAR
        online_enabled = True
        adaptive_threshold = True
        manifest_disabled = False
        trajectory_window = 16
    elif bank == "manifest_tamper":
        # Standard online stack; the tamper is constructed inside the
        # loop after W31 ratifies a cell.
        w30_priors = (1.0, 1.0, 1.0)
        w30_high_trust = W29_PARTITION_LINEAR
        online_enabled = True
        adaptive_threshold = True
        manifest_disabled = False
        trajectory_window = 16
    elif bank == "xllm_live":
        w30_priors = (1.0, 1.0, 1.0)
        w30_high_trust = W29_PARTITION_LINEAR
        online_enabled = True
        adaptive_threshold = True
        manifest_disabled = False
        trajectory_window = 16
    else:
        raise ValueError(f"unknown bank: {bank}")

    stacks = _build_phase78_stacks(
        bank=bank,
        T_decoder=T_decoder, K_consumers=K_consumers, n_eval=n_eval,
        signature_period=signature_period, bank_seed=bank_seed,
        bank_replicates=bank_replicates,
        chain_persist_window=chain_persist_window,
        max_active_chains=max_active_chains,
        calibration_stride=int(calibration_stride),
        ancestor_window=int(ancestor_window),
        w30_calibration_priors=tuple(w30_priors),
        w30_high_trust_partition_id=int(w30_high_trust),
        online_enabled=online_enabled,
        adaptive_threshold=adaptive_threshold,
        manifest_disabled=manifest_disabled,
        trajectory_window=trajectory_window,
    )
    schema = stacks["schema"]
    cells = stacks["cells"]
    underlying_bank = stacks["underlying_bank"]
    producer_w30_a = stacks["producer_w30_a"]
    producer_w31 = stacks["producer_w31"]

    per_cell: list[_PhaseRecord] = []
    n_tamper_attempts = 0
    n_tampered_rejected = 0
    last_envelope_for_tamper: list[OnlineCalibratedRatificationEnvelope] = []
    registered_pids = frozenset((W29_PARTITION_LINEAR,
                                  W29_PARTITION_HIERARCHICAL,
                                  W29_PARTITION_CYCLIC))

    for cell_idx, cell_handoffs in enumerate(cells):
        out_w30 = producer_w30_a.decode_rounds(cell_handoffs)
        out_w31 = producer_w31.decode_rounds(cell_handoffs)

        # Manifest tamper: from cell 1 onward, take the W31 envelope
        # of the prior cell, swap its prior_trajectory_cid into the
        # current envelope, and verify (expect rejection).
        if bank == "manifest_tamper":
            ev = producer_w31.last_envelope
            if ev is not None:
                last_envelope_for_tamper.append(ev)
                if len(last_envelope_for_tamper) >= 2:
                    prev_ev = last_envelope_for_tamper[-2]
                    cur_ev = ev
                    if prev_ev.prior_trajectory_cid != cur_ev.prior_trajectory_cid:
                        # Swap: take previous envelope's prior_trajectory_cid
                        # into current envelope.  Recompute the manifest_cid
                        # to make the manifest hash itself self-consistent
                        # — the only check that catches this is the
                        # registered_basis_history_cid passthrough check
                        # (the swap changed prior_trajectory_cid, but the
                        # manifest CID still recomputes correctly because
                        # we pass the swapped prior_trajectory_cid into
                        # the manifest hash).
                        from vision_mvp.coordpy.team_coord import (
                            _compute_w31_manifest_cid,
                            _compute_w31_outer_cid,
                        )
                        # Build a tampered envelope with the swapped
                        # prior_trajectory_cid.
                        new_manifest_cid = _compute_w31_manifest_cid(
                            basis_history_cid=cur_ev.basis_history_cid,
                            calibration_cid=cur_ev.calibration_cid,
                            ancestor_chain_cid=cur_ev.ancestor_chain_cid,
                            prior_trajectory_cid=prev_ev.prior_trajectory_cid,
                            threshold_trajectory_cid=cur_ev.threshold_trajectory_cid,
                            route_audit_cid=cur_ev.route_audit_cid,
                        )
                        tampered = dataclasses.replace(
                            cur_ev,
                            prior_trajectory=prev_ev.prior_trajectory,
                            prior_trajectory_cid=prev_ev.prior_trajectory_cid,
                            manifest_cid=new_manifest_cid,
                            w31_cid="",
                        )
                        # Run the verifier with the registered CIDs
                        # of the *current* cell.
                        outcome = verify_online_calibrated_ratification(
                            tampered,
                            registered_schema=schema,
                            registered_w30_calibrated_cid=
                                cur_ev.w30_calibrated_cid,
                            registered_partition_ids=registered_pids,
                            registered_trajectory_window=int(
                                producer_w31.registry.trajectory_window),
                            registered_basis_history_cid=
                                cur_ev.basis_history_cid,
                            registered_calibration_cid=
                                cur_ev.calibration_cid,
                            registered_ancestor_chain_cid=
                                cur_ev.ancestor_chain_cid,
                            registered_route_audit_cid=
                                cur_ev.route_audit_cid,
                            registered_prior_trajectory_cid=
                                cur_ev.prior_trajectory_cid,
                            registered_threshold_trajectory_cid=
                                cur_ev.threshold_trajectory_cid,
                        )
                        n_tamper_attempts += 1
                        if not outcome.ok and outcome.reason in (
                                "prior_trajectory_cid_mismatch",
                                "manifest_cid_mismatch",
                                "w31_outer_cid_mismatch"):
                            # The cross-cell swap is detected by the
                            # registered-prior-trajectory-CID check
                            # (a swap inserts a previous valid
                            # trajectory CID; the registry's expected
                            # CID for the *current* cell does not
                            # match).
                            n_tampered_rejected += 1
                        # Try a second tamper: corrupt the manifest_cid
                        # directly.  This should be detected by either
                        # the manifest_cid recomputation (which uses
                        # env.prior_trajectory_cid etc.) or the outer
                        # w31_cid check.
                        tampered2 = dataclasses.replace(
                            cur_ev, manifest_cid="ee" * 32,
                            w31_cid="")
                        outcome2 = verify_online_calibrated_ratification(
                            tampered2,
                            registered_schema=schema,
                            registered_w30_calibrated_cid=
                                cur_ev.w30_calibrated_cid,
                            registered_partition_ids=registered_pids,
                            registered_trajectory_window=int(
                                producer_w31.registry.trajectory_window),
                            registered_basis_history_cid=
                                cur_ev.basis_history_cid,
                            registered_calibration_cid=
                                cur_ev.calibration_cid,
                            registered_ancestor_chain_cid=
                                cur_ev.ancestor_chain_cid,
                            registered_route_audit_cid=
                                cur_ev.route_audit_cid,
                        )
                        n_tamper_attempts += 1
                        if not outcome2.ok and outcome2.reason in (
                                "manifest_cid_mismatch",
                                "w31_outer_cid_mismatch"):
                            n_tampered_rejected += 1
                        # Tamper 3: prior trajectory entry's
                        # observed_agreement out of range.
                        if cur_ev.prior_trajectory:
                            bad_traj = list(cur_ev.prior_trajectory)
                            bad_traj[-1] = dataclasses.replace(
                                bad_traj[-1], observed_agreement=2.5)
                            tampered3 = dataclasses.replace(
                                cur_ev,
                                prior_trajectory=tuple(bad_traj),
                                w31_cid="")
                            outcome3 = verify_online_calibrated_ratification(
                                tampered3,
                                registered_schema=schema,
                                registered_w30_calibrated_cid=
                                    cur_ev.w30_calibrated_cid,
                                registered_partition_ids=registered_pids,
                                registered_trajectory_window=int(
                                    producer_w31.registry.trajectory_window),
                                registered_basis_history_cid=
                                    cur_ev.basis_history_cid,
                                registered_calibration_cid=
                                    cur_ev.calibration_cid,
                                registered_ancestor_chain_cid=
                                    cur_ev.ancestor_chain_cid,
                                registered_route_audit_cid=
                                    cur_ev.route_audit_cid,
                            )
                            n_tamper_attempts += 1
                            if not outcome3.ok and outcome3.reason == (
                                    "prior_trajectory_observed_out_of_range"):
                                n_tampered_rejected += 1
                        # Tamper 4: threshold trajectory value out of
                        # range.
                        if cur_ev.threshold_trajectory:
                            bad_thr = list(cur_ev.threshold_trajectory)
                            bad_thr[-1] = 1.7
                            tampered4 = dataclasses.replace(
                                cur_ev,
                                threshold_trajectory=tuple(bad_thr),
                                w31_cid="")
                            outcome4 = verify_online_calibrated_ratification(
                                tampered4,
                                registered_schema=schema,
                                registered_w30_calibrated_cid=
                                    cur_ev.w30_calibrated_cid,
                                registered_partition_ids=registered_pids,
                                registered_trajectory_window=int(
                                    producer_w31.registry.trajectory_window),
                                registered_basis_history_cid=
                                    cur_ev.basis_history_cid,
                                registered_calibration_cid=
                                    cur_ev.calibration_cid,
                                registered_ancestor_chain_cid=
                                    cur_ev.ancestor_chain_cid,
                                registered_route_audit_cid=
                                    cur_ev.route_audit_cid,
                            )
                            n_tamper_attempts += 1
                            if not outcome4.ok and outcome4.reason == (
                                    "threshold_trajectory_value_out_of_range"):
                                n_tampered_rejected += 1
                        # Tamper 5: outer w31_cid byte-for-byte corruption.
                        tampered5 = dataclasses.replace(cur_ev)
                        object.__setattr__(tampered5, "w31_cid",
                                            "aa" * 32)
                        outcome5 = verify_online_calibrated_ratification(
                            tampered5,
                            registered_schema=schema,
                            registered_w30_calibrated_cid=
                                cur_ev.w30_calibrated_cid,
                            registered_partition_ids=registered_pids,
                            registered_trajectory_window=int(
                                producer_w31.registry.trajectory_window),
                            registered_basis_history_cid=
                                cur_ev.basis_history_cid,
                            registered_calibration_cid=
                                cur_ev.calibration_cid,
                            registered_ancestor_chain_cid=
                                cur_ev.ancestor_chain_cid,
                            registered_route_audit_cid=
                                cur_ev.route_audit_cid,
                        )
                        n_tamper_attempts += 1
                        if not outcome5.ok and outcome5.reason == (
                                "w31_outer_cid_mismatch"):
                            n_tampered_rejected += 1

        expected = _expected_gold_for_cell(
            bank=underlying_bank, cell_idx=cell_idx, n_eval=n_eval,
            signature_period=signature_period)
        rec = _PhaseRecord(
            cell_idx=int(cell_idx),
            expected=set(expected),
            correct_w30=_is_correct(out_w30, set(expected)),
            correct_w31=_is_correct(out_w31, set(expected)),
            w30_visible=_w30_visible_from_out(out_w30),
            w31_visible=_w31_visible(out_w31),
            w31_overhead=_w31_overhead(out_w31),
            w30_branch=str(
                out_w30.get("calibrated_geometry", {}).get(
                    "decoder_branch", "")),
            w31_branch=_w31_branch(out_w31),
            w30_ratified=bool(
                out_w30.get("calibrated_geometry", {}).get(
                    "ratified", False)),
            w31_ratified=_w31_ratified(out_w31),
            w31_n_structured_bits=_w31_n_structured_bits(out_w31),
            w31_cram_factor=_w31_cram_factor(out_w31),
            w30_effective_partition_id=_w30_effective_partition_id(
                out_w31),  # the W30 INSIDE W31 is the more relevant
            w31_prior_after=_w31_prior_after(out_w31),
            w31_threshold_after=_w31_threshold_after(out_w31),
            w31_observed_agreement=_w31_observed_agreement(out_w31),
        )
        per_cell.append(rec)

    n_cells = len(per_cell)
    if n_cells == 0:
        return {"error": "no cells", "bank": bank}

    correct_w30 = sum(1 for r in per_cell if r.correct_w30)
    correct_w31 = sum(1 for r in per_cell if r.correct_w31)
    rate_w30 = correct_w30 / n_cells
    rate_w31 = correct_w31 / n_cells
    delta_w31_w30 = rate_w31 - rate_w30

    total_w30 = sum(r.w30_visible for r in per_cell)
    total_w31 = sum(r.w31_visible for r in per_cell)
    mean_overhead_w31_vs_w30 = (
        sum(r.w31_overhead for r in per_cell) / float(n_cells))
    max_overhead_w31_vs_w30 = max(
        (r.w31_overhead for r in per_cell), default=0)

    n_w31_ratified = sum(1 for r in per_cell if r.w31_ratified)
    n_w31_ratified_correct = sum(
        1 for r in per_cell if r.w31_ratified and r.correct_w31)
    trust_precision_w31 = (
        n_w31_ratified_correct / n_w31_ratified
        if n_w31_ratified > 0 else 1.0)

    n_cram_cells = sum(1 for r in per_cell if r.w31_cram_factor > 0)
    if n_cram_cells > 0:
        mean_cram_w31 = (
            sum(r.w31_cram_factor for r in per_cell
                if r.w31_cram_factor > 0) / float(n_cram_cells))
    else:
        mean_cram_w31 = 0.0

    branch_hist: dict[str, int] = {}
    for r in per_cell:
        branch_hist[r.w31_branch] = branch_hist.get(r.w31_branch, 0) + 1

    n_online_updates = int(producer_w31.registry.n_online_updates)
    n_w31_envelopes_registered = int(
        producer_w31.registry.n_w31_registered)
    n_w31_envelopes_rejected = int(
        producer_w31.registry.n_w31_rejected)

    # Track final per-partition prior to verify online learning
    # actually fired.
    final_priors = {}
    cv = producer_w31.inner.registry.calibration_vector
    if cv is not None:
        for pid, p in zip(cv.partition_ids, cv.calibration_vector):
            final_priors[int(pid)] = float(p)

    # Per-cell history (summary).
    prior_traj_summary = []
    for r in per_cell:
        prior_traj_summary.append({
            "cell_idx": int(r.cell_idx),
            "effective_partition_id": int(r.w30_effective_partition_id),
            "observed_agreement": round(float(r.w31_observed_agreement), 4),
            "prior_after": round(float(r.w31_prior_after), 4),
            "threshold_after": round(float(r.w31_threshold_after), 4),
            "w30_branch": r.w30_branch,
            "w31_branch": r.w31_branch,
            "w30_ratified": bool(r.w30_ratified),
            "w31_ratified": bool(r.w31_ratified),
            "correct_w30": bool(r.correct_w30),
            "correct_w31": bool(r.correct_w31),
        })

    if n_tamper_attempts > 0:
        tamper_reject_rate = n_tampered_rejected / n_tamper_attempts
    else:
        tamper_reject_rate = None

    return {
        "bank": bank,
        "underlying_bank": underlying_bank,
        "T_decoder": T_decoder,
        "K_consumers": K_consumers,
        "n_eval": n_cells,
        "bank_seed": bank_seed,
        "calibration_stride": int(calibration_stride),
        "ancestor_window": int(ancestor_window),
        "online_enabled": bool(online_enabled),
        "adaptive_threshold": bool(adaptive_threshold),
        "manifest_disabled": bool(manifest_disabled),
        "trajectory_window": int(trajectory_window),
        # Correctness.
        "correctness_ratified_rate_w30": rate_w30,
        "correctness_ratified_rate_w31": rate_w31,
        "delta_w31_minus_w30": delta_w31_w30,
        "trust_precision_w31": trust_precision_w31,
        "n_w31_ratified": int(n_w31_ratified),
        # Wire-token economics.
        "total_w30_visible_tokens": int(total_w30),
        "total_w31_visible_tokens": int(total_w31),
        "mean_overhead_w31_vs_w30_per_cell": float(mean_overhead_w31_vs_w30),
        "max_overhead_w31_vs_w30_per_cell": int(max_overhead_w31_vs_w30),
        # Cram factor.
        "mean_cram_factor_w31": float(mean_cram_w31),
        # Branches / partitions.
        "branch_hist": branch_hist,
        # Online learning telemetry.
        "n_online_updates": int(n_online_updates),
        "n_w31_envelopes_registered": int(n_w31_envelopes_registered),
        "n_w31_envelopes_rejected": int(n_w31_envelopes_rejected),
        "final_calibration_priors": final_priors,
        "byte_equivalent_w31_w30": (total_w31 == total_w30),
        # Tamper.
        "n_tamper_attempts": int(n_tamper_attempts),
        "n_tampered_rejected": int(n_tampered_rejected),
        "tamper_reject_rate": tamper_reject_rate,
        # Per-cell trajectory summary.
        "prior_trajectory_summary": prior_traj_summary,
    }


def run_phase78_seed_sweep(
        *,
        bank: str = "nonstationary_prior",
        seeds: tuple[int, ...] = (11, 17, 23, 29, 31),
        T_decoder: int | None = None,
        K_consumers: int = 3,
        n_eval: int = 16,
        bank_replicates: int = 4,
        chain_persist_window: int | None = None,
        max_active_chains: int = 8,
        signature_period: int = 4,
        calibration_stride: int = 8,
        ancestor_window: int = 4,
) -> dict[str, Any]:
    """Run a seed-sweep over Phase-78 sub-bank ``bank``."""
    seed_results = []
    for s in seeds:
        r = run_phase78(
            bank=bank, T_decoder=T_decoder, K_consumers=K_consumers,
            n_eval=n_eval, bank_replicates=bank_replicates,
            bank_seed=int(s),
            chain_persist_window=chain_persist_window,
            max_active_chains=max_active_chains,
            signature_period=signature_period,
            calibration_stride=calibration_stride,
            ancestor_window=ancestor_window,
        )
        seed_results.append(r)

    deltas = [r["delta_w31_minus_w30"] for r in seed_results]
    rates_w30 = [r["correctness_ratified_rate_w30"] for r in seed_results]
    rates_w31 = [r["correctness_ratified_rate_w31"] for r in seed_results]
    trust_precs = [r["trust_precision_w31"] for r in seed_results]
    overheads = [r["mean_overhead_w31_vs_w30_per_cell"]
                 for r in seed_results]
    max_overheads = [r["max_overhead_w31_vs_w30_per_cell"]
                     for r in seed_results]

    summary = {
        "bank": bank,
        "seeds": list(seeds),
        "n_seeds": len(seed_results),
        "min_delta_w31_minus_w30": min(deltas) if deltas else None,
        "max_delta_w31_minus_w30": max(deltas) if deltas else None,
        "mean_delta_w31_minus_w30": (sum(deltas) / len(deltas)
                                       if deltas else None),
        "min_correctness_w30": min(rates_w30) if rates_w30 else None,
        "max_correctness_w30": max(rates_w30) if rates_w30 else None,
        "min_correctness_w31": min(rates_w31) if rates_w31 else None,
        "max_correctness_w31": max(rates_w31) if rates_w31 else None,
        "min_trust_precision_w31": min(trust_precs) if trust_precs else None,
        "mean_overhead_w31_vs_w30_per_cell": (
            sum(overheads) / len(overheads) if overheads else None),
        "max_overhead_w31_vs_w30_per_cell": (
            max(max_overheads) if max_overheads else None),
        "all_w31_ge_w30": all(
            d >= 0 - 1e-9 for d in deltas),
        "all_byte_equivalent_w31_w30": all(
            r["byte_equivalent_w31_w30"] for r in seed_results),
        "n_seeds_clearing_delta_010": sum(1 for d in deltas if d >= 0.10),
        "n_seeds_clearing_delta_005": sum(1 for d in deltas if d >= 0.05),
        "seed_results": seed_results,
    }
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(
        description="Phase 78 — W31 online self-calibrated geometry-aware "
                    "dense control benchmark.")
    p.add_argument("--bank", default="nonstationary_prior", choices=[
        "trivial_w31", "nonstationary_prior", "adaptive_threshold",
        "frozen_threshold", "no_drift", "manifest_tamper", "xllm_live",
    ])
    p.add_argument("--T-decoder", type=int, default=None)
    p.add_argument("--K-consumers", type=int, default=3)
    p.add_argument("--n-eval", type=int, default=16)
    p.add_argument("--bank-replicates", type=int, default=4)
    p.add_argument("--bank-seed", type=int, default=11)
    p.add_argument("--chain-persist-window", type=int, default=None)
    p.add_argument("--max-active-chains", type=int, default=8)
    p.add_argument("--signature-period", type=int, default=4)
    p.add_argument("--calibration-stride", type=int, default=8)
    p.add_argument("--ancestor-window", type=int, default=4)
    p.add_argument("--seed-sweep", action="store_true",
                    help="Run a sweep over seeds {11, 17, 23, 29, 31}.")
    p.add_argument("--out-json", default=None,
                    help="If provided, write JSON results here.")
    p.add_argument("--quiet", action="store_true",
                    help="Suppress stdout (still writes --out-json).")
    args = p.parse_args()

    if args.seed_sweep:
        results = run_phase78_seed_sweep(
            bank=args.bank, T_decoder=args.T_decoder,
            K_consumers=args.K_consumers,
            n_eval=args.n_eval, bank_replicates=args.bank_replicates,
            chain_persist_window=args.chain_persist_window,
            max_active_chains=args.max_active_chains,
            signature_period=args.signature_period,
            calibration_stride=args.calibration_stride,
            ancestor_window=args.ancestor_window,
        )
    else:
        results = run_phase78(
            bank=args.bank, T_decoder=args.T_decoder,
            K_consumers=args.K_consumers,
            n_eval=args.n_eval, bank_replicates=args.bank_replicates,
            bank_seed=args.bank_seed,
            chain_persist_window=args.chain_persist_window,
            max_active_chains=args.max_active_chains,
            signature_period=args.signature_period,
            calibration_stride=args.calibration_stride,
            ancestor_window=args.ancestor_window,
        )
    if not args.quiet:
        print(json.dumps(results, indent=2, sort_keys=True))
    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(results, f, indent=2, sort_keys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
