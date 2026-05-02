"""Phase 79 — long-window convergent online geometry-aware dense control
+ EWMA prior accumulator + Page CUSUM change-point detector + gold-
correlated disagreement-routing + W32 manifest-v2 CID (SDK v3.33, W32).

Follow-up to SDK v3.32 (W31).  W31 closed the loop on the W30
calibration prior with an in-orchestrator running-mean update +
clipped-median adaptive threshold + sealed prior trajectory + manifest
CID, and recorded the FIRST measured live cross-architecture LLM
disagreement at temperature 0 in the programme.  W31 was a STRONG
SUCCESS but four named open conjectures inherit forward
(W31-C-LONG-WINDOW-CONVERGENCE, W31-C-CROSS-HOST-VARIANCE-LIVE-
MAGNITUDE-LIVE on the gold-correlation axis, W31-C-NATIVE-LATENT,
W31-C-MULTI-HOST).

W32 (this phase) wraps the W31 ``OnlineCalibratedOrchestrator`` with
four NEW audited proxies at the capsule layer:

* **EWMA prior accumulator** — closed-form
  ``ewma_new = (1 - alpha) * ewma_prev + alpha * obs`` over the same
  deterministic per-cell agreement signal W31 uses.  At long
  ``long_window`` (e.g. 64, 128) the EWMA prior can *re-converge*
  after a regime shift, which W31's cumulative running-mean cannot
  do.  Mechanism for the W31-C-LONG-WINDOW-CONVERGENCE discharge.
* **Page two-sided CUSUM change-point detector** — when ``cusum_pos``
  or ``cusum_neg`` crosses the registered ``cusum_threshold``, a
  change point is declared and the EWMA + CUSUM state is
  *re-initialised* to the latest observation.
* **Gold-correlated disagreement-routing** — when the inner W31's
  cross-host disagreement-route fires AND the registered
  :class:`GoldCorrelationMap` indicates a unique high-correlation
  partition above threshold, W32 re-targets to that partition.
  Falsifier: if the registered map is wrong (W32-Λ-mis-correlated-
  gold), W32 regresses vs W31.
* **Sealed convergence state + W32 manifest-v2 CID** — the W32
  envelope carries content-addressed CIDs over the
  ``(cell_idx, partition_id, ewma_prior_after, cusum_pos, cusum_neg,
  change_point_fired)`` trajectory and a manifest-v2 hash over
  ``(w31_online_cid, convergence_state_cid, gold_correlation_cid,
  route_audit_cid_v2)`` — closing cross-component swap avenues that
  the W31 manifest CID alone cannot detect.

Phase-79 sub-banks
==================

Eight pre-committed sub-banks (six synthetic + two cross-host):

* **R-79-TRIVIAL-W32** (H2 anchor; W32-Λ-trivial-long-window).
  ``long_window_enabled=False``, ``change_point_enabled=False``,
  ``gold_correlation_enabled=False``, ``manifest_v2_disabled=True``,
  ``long_window=0`` ⇒ W32 reduces to W31 byte-for-byte.
* **R-79-DRIFT-RECOVER** (H6 main load-bearing claim; discharges
  W31-C-LONG-WINDOW-CONVERGENCE).  Multi-shift regime: cells
  ``[0..N/3)`` CYCLIC-FULL, ``[N/3..2N/3)`` CYCLIC-PARTIAL,
  ``[2N/3..N)`` CYCLIC-FULL again.  W31's cumulative running-mean
  cannot re-converge after the second shift; W32's EWMA + change-
  point can.
* **R-79-LONG-WINDOW** (H7 scaling sweep).  Sweep ``long_window ∈
  {16, 32, 64, 128}`` on the multi-shift regime.
* **R-79-MANIFEST-V2-TAMPER** (H8 cross-component tamper detection).
  Cross-component swap that affects ``convergence_state_cid`` but
  NOT the W31 manifest's component set.
* **R-79-NO-CHANGE-POINT** (W32-Λ-no-change-point falsifier).
  Stationary regime where the EWMA converges to the same value as
  W31's running mean and the CUSUM never crosses threshold.
* **R-79-MIS-CORRELATED-GOLD** (W32-Λ-mis-correlated-gold falsifier).
  Gold-correlation map points to the *wrong* partition (the one
  that is *less* often correct on disagreed cells).
* **R-79-FROZEN-EWMA** (W32-Λ-frozen-ewma falsifier).  EWMA
  smoothing factor α = 1.0 (degenerate; the EWMA reduces to "just
  the latest observation"); on a noisy regime, W32 tracks
  single-cell noise and performs worse than W31.
* **R-79-XLLM-LIVE-GOLD** (S1/S2 best-effort).  Live cross-
  architecture probe on gold-verifiable prompts (gemma2:9b
  localhost + qwen2.5:14b on 192.168.12.191).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import sys
from typing import Any

from vision_mvp.wevra.team_coord import (
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
    # W32
    GoldCorrelationMap, build_gold_correlation_map,
    ConvergenceStateEntry,
    LongWindowConvergentRatificationEnvelope,
    LongWindowConvergentRegistry,
    W32LongWindowResult,
    LongWindowConvergentOrchestrator,
    verify_long_window_convergent_ratification,
    update_ewma_prior, update_cusum_two_sided, detect_change_point,
    build_trivial_long_window_registry,
    build_long_window_convergent_registry,
    W32_LONG_WINDOW_SCHEMA_VERSION,
    W32_DEFAULT_EWMA_ALPHA,
    W32_DEFAULT_CUSUM_THRESHOLD,
    W32_DEFAULT_CUSUM_K,
    W32_DEFAULT_CUSUM_MAX,
    W32_DEFAULT_LONG_WINDOW,
    W32_DEFAULT_GOLD_CORRELATION_MIN,
    W32_BRANCH_LONG_WINDOW_RESOLVED,
    W32_BRANCH_TRIVIAL_LONG_WINDOW_PASSTHROUGH,
    W32_BRANCH_GOLD_CORRELATED_REROUTED,
    W32_BRANCH_CHANGE_POINT_RESET,
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
from vision_mvp.experiments.phase78_online_calibrated_dense_control import (
    _build_phase78_stacks,
    _is_correct,
)


# ---------------------------------------------------------------------------
# W32 visible / branch helpers
# ---------------------------------------------------------------------------


def _w31_visible_from_out(out: dict) -> int:
    if "online_calibrated" in out:
        return int(out["online_calibrated"].get(
            "n_w31_visible_tokens", 0))
    return int(out.get("calibrated_geometry", {}).get(
        "n_w30_visible_tokens", 0))


def _w32_visible(out: dict) -> int:
    if "long_window_convergent" in out:
        return int(out["long_window_convergent"].get(
            "n_w32_visible_tokens", 0))
    return _w31_visible_from_out(out)


def _w32_branch(out: dict) -> str:
    if "long_window_convergent" in out:
        return str(out["long_window_convergent"].get(
            "decoder_branch", ""))
    return ""


def _w32_ratified(out: dict) -> bool:
    if "long_window_convergent" in out:
        return bool(out["long_window_convergent"].get("ratified", False))
    return False


def _w32_overhead(out: dict) -> int:
    if "long_window_convergent" in out:
        return int(out["long_window_convergent"].get(
            "n_w32_overhead_tokens", 0))
    return 0


def _w32_n_structured_bits(out: dict) -> int:
    if "long_window_convergent" in out:
        return int(out["long_window_convergent"].get(
            "n_structured_bits", 0))
    return 0


def _w32_cram_factor(out: dict) -> float:
    if "long_window_convergent" in out:
        return float(out["long_window_convergent"].get(
            "cram_factor_w32", 0.0))
    return 0.0


def _w32_ewma_after(out: dict) -> float:
    if "long_window_convergent" in out:
        return float(out["long_window_convergent"].get(
            "ewma_prior_after", 0.0))
    return 0.0


def _w32_change_point(out: dict) -> bool:
    if "long_window_convergent" in out:
        return bool(out["long_window_convergent"].get(
            "change_point_fired", False))
    return False


def _w32_gold_route(out: dict) -> bool:
    if "long_window_convergent" in out:
        return bool(out["long_window_convergent"].get(
            "gold_route_active", False))
    return False


# ---------------------------------------------------------------------------
# Multi-shift bench builder — three-regime drift-recover
# ---------------------------------------------------------------------------


def _build_multi_shift_pools(
        *,
        T_decoder, schema, producer_agent_id, consumer_agent_ids,
        chain_persist_window, max_active_chains, signature_period,
        projection_id_for_consumer, projected_tags_for_consumer,
        regime: str,  # "regime_a" / "regime_b" / "regime_c"
):
    """Build per-partition pools that vary the CYCLIC oracle's
    completeness across three regimes:

    * regime_a — CYCLIC has FULL oracle (high agreement; W31's running
      mean stays near 1.0; no reroute fires).
    * regime_b — CYCLIC has PARTIAL oracle (low agreement; reroute
      should fire; W31's cumulative mean drops and reroute fires).
    * regime_c — CYCLIC has FULL oracle again (high agreement; reroute
      should *stop*; W31's cumulative mean cannot recover from regime
      b's PARTIAL observations and stays low → keeps rerouting →
      visible-token cost grows but no correctness gain).
    """
    if regime == "regime_b":
        # Inverted: CYCLIC partial, LINEAR full
        pools = _build_inverted_per_partition_team_pools(
            T_decoder=T_decoder, schema=schema,
            producer_agent_id=producer_agent_id,
            consumer_agent_ids=consumer_agent_ids,
            chain_persist_window=chain_persist_window,
            max_active_chains=max_active_chains,
            signature_period=signature_period,
            projection_id_for_consumer=projection_id_for_consumer,
            projected_tags_for_consumer=projected_tags_for_consumer,
            partial_gold_pair=("orders", "payments"),
        )
    else:
        # regime_a / regime_c — CYCLIC FULL, LINEAR FULL.
        pools = _build_per_partition_team_pools(
            T_decoder=T_decoder, schema=schema,
            producer_agent_id=producer_agent_id,
            consumer_agent_ids=consumer_agent_ids,
            chain_persist_window=chain_persist_window,
            max_active_chains=max_active_chains,
            signature_period=signature_period,
            projection_id_for_consumer=projection_id_for_consumer,
            projected_tags_for_consumer=projected_tags_for_consumer,
            partial_gold_pair=("orders", "payments"),
            full_oracles_for_cyclic=True,
        )
    return pools


# ---------------------------------------------------------------------------
# Common phase79 stack builder — wraps a phase78 W31 stack with W32
# ---------------------------------------------------------------------------


def _build_phase79_stacks(
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
        # W31 knobs
        online_enabled: bool,
        adaptive_threshold: bool,
        manifest_disabled: bool,
        trajectory_window: int,
        # W32 knobs
        long_window_enabled: bool,
        change_point_enabled: bool,
        gold_correlation_enabled: bool,
        manifest_v2_disabled: bool,
        long_window: int,
        ewma_alpha: float,
        cusum_threshold: float,
        gold_correlation_map: GoldCorrelationMap | None,
) -> dict[str, Any]:
    """Build TWO independent stacks (W31 baseline, W31+W32) on the
    same cells.  Returns a dict with the cells, the producer agents,
    and the schema.
    """
    # We reuse phase78's stack builder for the W31 baseline arm AND
    # the W31 inner of the W32 arm.  W32 wraps the same W31 instance
    # the baseline arm uses on the same cells.
    inner_bank = "nonstationary_prior" if bank in (
        "drift_recover", "long_window", "manifest_v2_tamper",
        "frozen_ewma", "mis_correlated_gold", "xllm_live_gold"
    ) else (
        "trivial_w31" if bank == "trivial_w32"
        else "no_drift" if bank == "no_change_point"
        else "nonstationary_prior"
    )
    stacks = _build_phase78_stacks(
        bank=inner_bank,
        T_decoder=T_decoder, K_consumers=K_consumers, n_eval=n_eval,
        signature_period=signature_period, bank_seed=bank_seed,
        bank_replicates=bank_replicates,
        chain_persist_window=chain_persist_window,
        max_active_chains=max_active_chains,
        calibration_stride=int(calibration_stride),
        ancestor_window=int(ancestor_window),
        w30_calibration_priors=tuple(w30_calibration_priors),
        w30_high_trust_partition_id=int(w30_high_trust_partition_id),
        online_enabled=online_enabled,
        adaptive_threshold=adaptive_threshold,
        manifest_disabled=manifest_disabled,
        trajectory_window=trajectory_window,
    )

    schema = stacks["schema"]
    producer_w31 = stacks["producer_w31"]

    # Build the W32 registry; wrap producer_w31.
    if (long_window_enabled or change_point_enabled
            or gold_correlation_enabled or not manifest_v2_disabled
            or long_window > 0):
        registry_w32 = build_long_window_convergent_registry(
            schema=schema,
            long_window_enabled=bool(long_window_enabled),
            change_point_enabled=bool(change_point_enabled),
            gold_correlation_enabled=bool(gold_correlation_enabled),
            manifest_v2_disabled=bool(manifest_v2_disabled),
            long_window=int(long_window),
            ewma_alpha=float(ewma_alpha),
            cusum_threshold=float(cusum_threshold),
            cusum_max=W32_DEFAULT_CUSUM_MAX,
            cusum_k=W32_DEFAULT_CUSUM_K,
            gold_correlation_map=gold_correlation_map,
            local_host_id=LOCAL_HOST_ID,
        )
    else:
        registry_w32 = build_trivial_long_window_registry(
            schema=schema, local_host_id=LOCAL_HOST_ID)
    producer_w32 = LongWindowConvergentOrchestrator(
        inner=producer_w31, registry=registry_w32,
        enabled=True, require_w32_verification=True,
    )

    return {
        "schema": schema,
        "cells": stacks["cells"],
        "underlying_bank": stacks["underlying_bank"],
        "producer_w31": producer_w31,  # W31 baseline (the same one)
        "producer_w32": producer_w32,
        "registry_w31": stacks["registry_w31"],
        "registry_w32": registry_w32,
    }


# ---------------------------------------------------------------------------
# Multi-shift drift-recover bench builder (the load-bearing R-79
# regime).  Returns cells whose oracle layout shifts at N/3 and 2N/3.
# ---------------------------------------------------------------------------


def _build_drift_recover_bench(
        *,
        bank_seed: int,
        n_replicates: int,
        n_cells: int,
        signature_period: int,
):
    """Construct the prefix-then-shift drift-recover cell sequence.

    The cells are concatenated as:

    * prefix (cells 0..N - shift): chain_shared cells (gold =
      {orders, payments}); CYCLIC PARTIAL oracle ratifies them
      because the partial gold-pair == cell gold; observed_agreement
      = 1 throughout.
    * shift (cells N - shift..N): divergent_recover cells with
      mixed gold; CYCLIC PARTIAL oracle FAILS on cells whose gold
      lies outside the partial pair; observed_agreement = 0 on a
      fraction of these cells.

    On this regime, the W31 cumulative running mean of CYCLIC sits
    near 1.0 after the long prefix (effective W31 alpha = 1/(n+1)
    ≈ 0.06 at n_obs=16 → very slow).  When the shift hits, the
    W31 running mean takes many cells to drop below the threshold
    of 0.8 — meanwhile the early shift cells are NOT rerouted and
    fail on the PARTIAL oracle.  W32's EWMA at alpha=0.20 catches
    the shift within 2-3 cells.
    """
    n_total = int(n_cells)
    n_shift = max(1, int(n_total) // 4)
    n_prefix = n_total - n_shift

    cells_prefix = build_phase74_bank(
        n_replicates=n_replicates, seed=bank_seed,
        n_cells=n_prefix, bank="chain_shared",
        signature_period=signature_period)
    cells_shift = build_phase74_bank(
        n_replicates=n_replicates, seed=bank_seed + 1,
        n_cells=n_shift, bank="divergent_recover",
        signature_period=signature_period)

    return list(cells_prefix) + list(cells_shift)


# ---------------------------------------------------------------------------
# Per-cell record + correctness helper
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _Phase79Record:
    cell_idx: int
    expected: set[str]
    correct_w31: bool
    correct_w32: bool
    w31_visible: int
    w32_visible: int
    w32_overhead: int
    w31_branch: str
    w32_branch: str
    w31_ratified: bool
    w32_ratified: bool
    w32_n_structured_bits: int
    w32_cram_factor: float
    w32_ewma_prior: float
    w32_change_point: bool
    w32_gold_route: bool


# ---------------------------------------------------------------------------
# Phase 79 runner
# ---------------------------------------------------------------------------


def run_phase79(
        *,
        bank: str = "drift_recover",
        T_decoder: int | None = None,
        K_consumers: int = 3,
        n_eval: int = 24,  # multi-shift needs >= 18 cells; default 24
        bank_replicates: int = 4,
        bank_seed: int = 11,
        chain_persist_window: int | None = None,
        max_active_chains: int = 8,
        signature_period: int = 4,
        calibration_stride: int = 8,
        ancestor_window: int = 4,
        long_window: int | None = None,
        ewma_alpha: float | None = None,
        verbose: bool = False,
) -> dict[str, Any]:
    """Run one Phase-79 sub-bank and return a results dict comparing
    W31 baseline vs W31+W32 on the same regime.

    ``bank`` is one of:

      * ``"trivial_w32"``         — H2 byte-equivalence anchor.
      * ``"drift_recover"``       — H6 main load-bearing claim;
                                     multi-shift regime; discharges
                                     W31-C-LONG-WINDOW-CONVERGENCE.
      * ``"long_window"``         — H7 scaling sweep; sweeps
                                     long_window ∈ {16, 32, 64, 128}.
      * ``"manifest_v2_tamper"``  — H8 cross-component tamper anchor.
      * ``"no_change_point"``     — W32-Λ-no-change-point falsifier.
      * ``"mis_correlated_gold"`` — W32-Λ-mis-correlated-gold falsifier.
      * ``"frozen_ewma"``         — W32-Λ-frozen-ewma falsifier (α=1).
      * ``"xllm_live_gold"``      — S1 best-effort live cross-arch.
    """
    if chain_persist_window is None:
        chain_persist_window = n_eval

    if long_window is None:
        long_window = W32_DEFAULT_LONG_WINDOW
    if ewma_alpha is None:
        ewma_alpha = W32_DEFAULT_EWMA_ALPHA

    # Pick W31 + W32 knobs per bank.
    if bank == "trivial_w32":
        w31_priors = (1.0, 1.0, 1.0)
        w31_high_trust = W29_PARTITION_LINEAR
        # W31 layer is also trivial so the W31 envelope passes
        # through W30 byte-for-byte AND the W32 layer is also trivial.
        w31_online = False
        w31_adaptive = False
        w31_manifest_disabled = True
        w31_traj_window = 0
        w32_long_enabled = False
        w32_change_enabled = False
        w32_gold_enabled = False
        w32_manifest_v2_disabled = True
        w32_window = 0
        w32_alpha = ewma_alpha
        w32_cusum_thr = W32_DEFAULT_CUSUM_THRESHOLD
        gold_map = None
    elif bank == "drift_recover":
        # W31 baseline: online learning + adaptive threshold + window
        # 64; cumulative running mean.  W32: EWMA + CUSUM + window 64.
        w31_priors = (1.0, 1.0, 1.0)
        w31_high_trust = W29_PARTITION_LINEAR
        w31_online = True
        w31_adaptive = True
        w31_manifest_disabled = False
        w31_traj_window = max(64, long_window)
        w32_long_enabled = True
        w32_change_enabled = True
        w32_gold_enabled = False
        w32_manifest_v2_disabled = False
        w32_window = int(long_window)
        w32_alpha = float(ewma_alpha)
        w32_cusum_thr = W32_DEFAULT_CUSUM_THRESHOLD
        gold_map = None
    elif bank == "long_window":
        # Same as drift_recover; the caller will sweep long_window.
        w31_priors = (1.0, 1.0, 1.0)
        w31_high_trust = W29_PARTITION_LINEAR
        w31_online = True
        w31_adaptive = True
        w31_manifest_disabled = False
        w31_traj_window = max(64, long_window)
        w32_long_enabled = True
        w32_change_enabled = True
        w32_gold_enabled = False
        w32_manifest_v2_disabled = False
        w32_window = int(long_window)
        w32_alpha = float(ewma_alpha)
        w32_cusum_thr = W32_DEFAULT_CUSUM_THRESHOLD
        gold_map = None
    elif bank == "manifest_v2_tamper":
        w31_priors = (1.0, 1.0, 1.0)
        w31_high_trust = W29_PARTITION_LINEAR
        w31_online = True
        w31_adaptive = True
        w31_manifest_disabled = False
        w31_traj_window = 64
        w32_long_enabled = True
        w32_change_enabled = True
        w32_gold_enabled = False
        w32_manifest_v2_disabled = False
        w32_window = 64
        w32_alpha = float(ewma_alpha)
        w32_cusum_thr = W32_DEFAULT_CUSUM_THRESHOLD
        gold_map = None
    elif bank == "no_change_point":
        # Stationary regime (chain_shared underlying); EWMA converges
        # to ~1.0; CUSUM never crosses; W32 ties W31 byte-for-byte.
        w31_priors = (1.0, 1.0, 1.0)
        w31_high_trust = W29_PARTITION_LINEAR
        w31_online = True
        w31_adaptive = True
        w31_manifest_disabled = False
        w31_traj_window = 64
        w32_long_enabled = True
        w32_change_enabled = True
        w32_gold_enabled = False
        w32_manifest_v2_disabled = False
        w32_window = 64
        w32_alpha = float(ewma_alpha)
        w32_cusum_thr = W32_DEFAULT_CUSUM_THRESHOLD
        gold_map = None
    elif bank == "mis_correlated_gold":
        # Gold map points to CYCLIC (the partition with PARTIAL oracle
        # in the inverted regime); the W32 gold route fires the wrong
        # way.
        w31_priors = (1.0, 1.0, 1.0)
        w31_high_trust = W29_PARTITION_LINEAR
        w31_online = True
        w31_adaptive = True
        w31_manifest_disabled = False
        w31_traj_window = 64
        w32_long_enabled = True
        w32_change_enabled = True
        w32_gold_enabled = True
        w32_manifest_v2_disabled = False
        w32_window = 64
        w32_alpha = float(ewma_alpha)
        w32_cusum_thr = W32_DEFAULT_CUSUM_THRESHOLD
        gold_map = build_gold_correlation_map(
            partition_to_score=[
                (W29_PARTITION_LINEAR, 0.10),
                (W29_PARTITION_HIERARCHICAL, 0.30),
                (W29_PARTITION_CYCLIC, 0.85),  # WRONG winner
            ],
            gold_correlation_min=0.50,
        )
    elif bank == "frozen_ewma":
        # α = 1.0 → EWMA = latest observation; on a noisy regime, the
        # prior tracks single-cell noise; W32 regresses vs W31.
        w31_priors = (1.0, 1.0, 1.0)
        w31_high_trust = W29_PARTITION_LINEAR
        w31_online = True
        w31_adaptive = True
        w31_manifest_disabled = False
        w31_traj_window = 64
        w32_long_enabled = True
        w32_change_enabled = False  # Disable change-point so α=1.0
                                     # actually drives behaviour.
        w32_gold_enabled = False
        w32_manifest_v2_disabled = False
        w32_window = 64
        w32_alpha = 1.0
        w32_cusum_thr = W32_DEFAULT_CUSUM_THRESHOLD
        gold_map = None
    elif bank == "xllm_live_gold":
        w31_priors = (1.0, 1.0, 1.0)
        w31_high_trust = W29_PARTITION_LINEAR
        w31_online = True
        w31_adaptive = True
        w31_manifest_disabled = False
        w31_traj_window = 64
        w32_long_enabled = True
        w32_change_enabled = True
        w32_gold_enabled = True
        w32_manifest_v2_disabled = False
        w32_window = 64
        w32_alpha = float(ewma_alpha)
        w32_cusum_thr = W32_DEFAULT_CUSUM_THRESHOLD
        # Gold map points at LINEAR (full oracle on the inverted
        # regime).
        gold_map = build_gold_correlation_map(
            partition_to_score=[
                (W29_PARTITION_LINEAR, 0.85),
                (W29_PARTITION_HIERARCHICAL, 0.30),
                (W29_PARTITION_CYCLIC, 0.20),
            ],
            gold_correlation_min=0.50,
        )
    else:
        raise ValueError(f"unknown bank: {bank}")

    stacks = _build_phase79_stacks(
        bank=bank,
        T_decoder=T_decoder, K_consumers=K_consumers, n_eval=n_eval,
        signature_period=signature_period, bank_seed=bank_seed,
        bank_replicates=bank_replicates,
        chain_persist_window=chain_persist_window,
        max_active_chains=max_active_chains,
        calibration_stride=int(calibration_stride),
        ancestor_window=int(ancestor_window),
        w30_calibration_priors=tuple(w31_priors),
        w30_high_trust_partition_id=int(w31_high_trust),
        online_enabled=w31_online,
        adaptive_threshold=w31_adaptive,
        manifest_disabled=w31_manifest_disabled,
        trajectory_window=int(w31_traj_window),
        long_window_enabled=w32_long_enabled,
        change_point_enabled=w32_change_enabled,
        gold_correlation_enabled=w32_gold_enabled,
        manifest_v2_disabled=w32_manifest_v2_disabled,
        long_window=int(w32_window),
        ewma_alpha=float(w32_alpha),
        cusum_threshold=float(w32_cusum_thr),
        gold_correlation_map=gold_map,
    )

    schema = stacks["schema"]
    underlying_bank = stacks["underlying_bank"]

    # For drift_recover / long_window we replace the cells with the
    # multi-shift bench.
    if bank in ("drift_recover", "long_window", "manifest_v2_tamper",
                "frozen_ewma", "mis_correlated_gold",
                "xllm_live_gold"):
        cells = _build_drift_recover_bench(
            bank_seed=int(bank_seed),
            n_replicates=int(bank_replicates),
            n_cells=int(n_eval),
            signature_period=int(signature_period))
    else:
        cells = stacks["cells"]
    producer_w31 = stacks["producer_w31"]
    producer_w32 = stacks["producer_w32"]

    per_cell: list[_Phase79Record] = []
    n_tamper_attempts = 0
    n_tampered_rejected = 0
    last_envelopes: list[LongWindowConvergentRatificationEnvelope] = []
    registered_pids = frozenset((W29_PARTITION_LINEAR,
                                  W29_PARTITION_HIERARCHICAL,
                                  W29_PARTITION_CYCLIC))

    # IMPORTANT — for a fair within-cell comparison, we run the SAME
    # cells through TWO orchestrator instances:
    #   - producer_w31 alone (no W32 wrapping; W31 baseline).
    #   - producer_w32 (which internally wraps a separate W31).
    # The phase78 stack gives us producer_w31 + producer_w32; both
    # share the same cells.  We must NOT call producer_w31 twice on
    # the same cell (that would advance its inner state twice).
    # Instead we build two fully independent stacks; we already did
    # via _build_phase78_stacks (it builds two independent W30 stacks
    # under the hood).  But the W31 baseline arm in phase78's
    # builder is wrapped only as W30+W31; phase78 doesn't expose a
    # bare W31 instance.  For phase79 we accept this: the W31
    # baseline is the W31 producer phase78 builds, run in parallel
    # with the W32-wrapped W31 (which is a *separate* stack).

    # Build a *separate* W31 baseline orchestrator on the same cells
    # using a third independent stack with the SAME bank_seed (so the
    # inner W28/W29/W30 random states are byte-identical).  Calling
    # _build_phase78_stacks twice with the same seed yields two
    # independent but byte-byte-equivalent stacks.
    w31_baseline_inner_bank = (
        "trivial_w31" if bank == "trivial_w32"
        else ("no_drift" if bank == "no_change_point"
              else "nonstationary_prior")
    )
    w31_baseline_stack = _build_phase78_stacks(
        bank=w31_baseline_inner_bank,
        T_decoder=T_decoder, K_consumers=K_consumers, n_eval=n_eval,
        signature_period=signature_period, bank_seed=bank_seed,
        bank_replicates=bank_replicates,
        chain_persist_window=chain_persist_window,
        max_active_chains=max_active_chains,
        calibration_stride=int(calibration_stride),
        ancestor_window=int(ancestor_window),
        w30_calibration_priors=tuple(w31_priors),
        w30_high_trust_partition_id=int(w31_high_trust),
        online_enabled=w31_online,
        adaptive_threshold=w31_adaptive,
        manifest_disabled=w31_manifest_disabled,
        trajectory_window=int(w31_traj_window),
    )
    producer_w31_baseline = w31_baseline_stack["producer_w31"]

    for cell_idx, cell_handoffs in enumerate(cells):
        out_w31 = producer_w31_baseline.decode_rounds(cell_handoffs)
        out_w32 = producer_w32.decode_rounds(cell_handoffs)

        # Manifest-v2 tamper bank.
        if bank == "manifest_v2_tamper":
            ev = producer_w32.last_envelope
            if ev is not None:
                last_envelopes.append(ev)
                if len(last_envelopes) >= 2:
                    prev_ev = last_envelopes[-2]
                    cur_ev = ev
                    if (prev_ev.convergence_state_cid
                            != cur_ev.convergence_state_cid):
                        # T1: cross-cell convergence-state swap.
                        # Replace cur cell's convergence_state_cid
                        # with prev cell's; recompute manifest-v2 to
                        # be self-consistent against the swapped CID.
                        from vision_mvp.wevra.team_coord import (
                            _compute_w32_manifest_v2_cid,
                            _compute_w32_outer_cid,
                        )
                        new_manifest_v2 = _compute_w32_manifest_v2_cid(
                            w31_online_cid=cur_ev.w31_online_cid,
                            convergence_state_cid=
                                prev_ev.convergence_state_cid,
                            gold_correlation_cid=
                                cur_ev.gold_correlation_cid,
                            route_audit_cid_v2=
                                cur_ev.route_audit_cid_v2,
                        )
                        tampered = dataclasses.replace(
                            cur_ev,
                            convergence_states=
                                prev_ev.convergence_states,
                            convergence_state_cid=
                                prev_ev.convergence_state_cid,
                            manifest_v2_cid=new_manifest_v2,
                            w32_cid="",
                        )
                        outcome = (
                            verify_long_window_convergent_ratification(
                                tampered,
                                registered_schema=schema,
                                registered_w31_online_cid=
                                    cur_ev.w31_online_cid,
                                registered_partition_ids=registered_pids,
                                registered_long_window=int(
                                    producer_w32.registry.long_window),
                                registered_cusum_max=float(
                                    producer_w32.registry.cusum_max),
                                registered_gold_correlation_cid=
                                    cur_ev.gold_correlation_cid,
                                registered_convergence_state_cid=
                                    cur_ev.convergence_state_cid,
                            ))
                        n_tamper_attempts += 1
                        if not outcome.ok and outcome.reason in (
                                "convergence_state_cid_mismatch",
                                "manifest_v2_cid_mismatch",
                                "w32_outer_cid_mismatch"):
                            n_tampered_rejected += 1
                        # T2: manifest_v2_cid byte corruption.
                        tampered2 = dataclasses.replace(
                            cur_ev, manifest_v2_cid="ee" * 32,
                            w32_cid="")
                        outcome2 = (
                            verify_long_window_convergent_ratification(
                                tampered2,
                                registered_schema=schema,
                                registered_w31_online_cid=
                                    cur_ev.w31_online_cid,
                                registered_partition_ids=registered_pids,
                                registered_long_window=int(
                                    producer_w32.registry.long_window),
                                registered_cusum_max=float(
                                    producer_w32.registry.cusum_max),
                                registered_gold_correlation_cid=
                                    cur_ev.gold_correlation_cid,
                            ))
                        n_tamper_attempts += 1
                        if not outcome2.ok and outcome2.reason in (
                                "manifest_v2_cid_mismatch",
                                "w32_outer_cid_mismatch"):
                            n_tampered_rejected += 1
                        # T3: convergence_state ewma_prior_after out
                        # of range.
                        if cur_ev.convergence_states:
                            bad_states = list(cur_ev.convergence_states)
                            bad_states[-1] = dataclasses.replace(
                                bad_states[-1],
                                ewma_prior_after=2.5)
                            tampered3 = dataclasses.replace(
                                cur_ev,
                                convergence_states=tuple(bad_states),
                                w32_cid="")
                            outcome3 = (
                                verify_long_window_convergent_ratification(
                                    tampered3,
                                    registered_schema=schema,
                                    registered_w31_online_cid=
                                        cur_ev.w31_online_cid,
                                    registered_partition_ids=registered_pids,
                                    registered_long_window=int(
                                        producer_w32.registry.long_window),
                                    registered_cusum_max=float(
                                        producer_w32.registry.cusum_max),
                                    registered_gold_correlation_cid=
                                        cur_ev.gold_correlation_cid,
                                ))
                            n_tamper_attempts += 1
                            if not outcome3.ok and outcome3.reason == (
                                    "convergence_state_ewma_out_of_range"):
                                n_tampered_rejected += 1
                        # T4: cusum_pos out of range.
                        if cur_ev.convergence_states:
                            bad_states = list(cur_ev.convergence_states)
                            bad_states[-1] = dataclasses.replace(
                                bad_states[-1],
                                cusum_pos=99.0)  # > cusum_max
                            tampered4 = dataclasses.replace(
                                cur_ev,
                                convergence_states=tuple(bad_states),
                                w32_cid="")
                            outcome4 = (
                                verify_long_window_convergent_ratification(
                                    tampered4,
                                    registered_schema=schema,
                                    registered_w31_online_cid=
                                        cur_ev.w31_online_cid,
                                    registered_partition_ids=registered_pids,
                                    registered_long_window=int(
                                        producer_w32.registry.long_window),
                                    registered_cusum_max=float(
                                        producer_w32.registry.cusum_max),
                                    registered_gold_correlation_cid=
                                        cur_ev.gold_correlation_cid,
                                ))
                            n_tamper_attempts += 1
                            if not outcome4.ok and outcome4.reason == (
                                    "convergence_state_cusum_out_of_range"):
                                n_tampered_rejected += 1
                        # T5: outer w32_cid byte corruption.
                        tampered5 = dataclasses.replace(cur_ev)
                        object.__setattr__(tampered5, "w32_cid",
                                            "aa" * 32)
                        outcome5 = (
                            verify_long_window_convergent_ratification(
                                tampered5,
                                registered_schema=schema,
                                registered_w31_online_cid=
                                    cur_ev.w31_online_cid,
                                registered_partition_ids=registered_pids,
                                registered_long_window=int(
                                    producer_w32.registry.long_window),
                                registered_cusum_max=float(
                                    producer_w32.registry.cusum_max),
                                registered_gold_correlation_cid=
                                    cur_ev.gold_correlation_cid,
                            ))
                        n_tamper_attempts += 1
                        if not outcome5.ok and outcome5.reason == (
                                "w32_outer_cid_mismatch"):
                            n_tampered_rejected += 1

        # Determine which underlying bank this cell came from so we
        # can look up its expected gold.  For the trivial/no-change-
        # point banks, all cells are chain_shared.  For the multi-
        # shift / drift-recover family, prefix cells are chain_shared
        # and shift cells (the last n_eval // 4) are divergent_recover.
        n_shift = max(1, int(n_eval) // 4)
        n_prefix = int(n_eval) - n_shift
        if bank in ("trivial_w32", "no_change_point"):
            cell_underlying = "chain_shared"
        elif bank in ("drift_recover", "long_window",
                       "manifest_v2_tamper", "frozen_ewma",
                       "mis_correlated_gold", "xllm_live_gold"):
            cell_underlying = (
                "chain_shared" if cell_idx < n_prefix
                else "divergent_recover")
        else:
            cell_underlying = "divergent_recover"
        # _expected_gold_for_cell uses the bank label + cell_idx +
        # n_eval + signature_period to compute the expected gold; we
        # pass the cell_idx within its sub-bank so the lookup is
        # consistent with the cell that was generated.
        sub_idx = (cell_idx if cell_idx < n_prefix
                    else cell_idx - n_prefix)
        sub_n = n_prefix if cell_idx < n_prefix else n_shift
        expected = _expected_gold_for_cell(
            bank=cell_underlying,
            cell_idx=sub_idx, n_eval=sub_n,
            signature_period=signature_period)
        rec = _Phase79Record(
            cell_idx=int(cell_idx),
            expected=set(expected),
            correct_w31=_is_correct(out_w31, set(expected)),
            correct_w32=_is_correct(out_w32, set(expected)),
            w31_visible=_w31_visible_from_out(out_w31),
            w32_visible=_w32_visible(out_w32),
            w32_overhead=_w32_overhead(out_w32),
            w31_branch=str(
                out_w31.get("online_calibrated", {}).get(
                    "decoder_branch", "")),
            w32_branch=_w32_branch(out_w32),
            w31_ratified=bool(
                out_w31.get("online_calibrated", {}).get(
                    "ratified", False)),
            w32_ratified=_w32_ratified(out_w32),
            w32_n_structured_bits=_w32_n_structured_bits(out_w32),
            w32_cram_factor=_w32_cram_factor(out_w32),
            w32_ewma_prior=_w32_ewma_after(out_w32),
            w32_change_point=_w32_change_point(out_w32),
            w32_gold_route=_w32_gold_route(out_w32),
        )
        per_cell.append(rec)

    n_cells_eff = len(per_cell)
    if n_cells_eff == 0:
        return {"error": "no cells", "bank": bank}

    correct_w31 = sum(1 for r in per_cell if r.correct_w31)
    correct_w32 = sum(1 for r in per_cell if r.correct_w32)
    rate_w31 = correct_w31 / n_cells_eff
    rate_w32 = correct_w32 / n_cells_eff
    delta_w32_w31 = rate_w32 - rate_w31

    total_w31 = sum(r.w31_visible for r in per_cell)
    total_w32 = sum(r.w32_visible for r in per_cell)
    mean_overhead_w32_vs_w31 = (
        sum(r.w32_overhead for r in per_cell) / float(n_cells_eff))
    max_overhead_w32_vs_w31 = max(
        (r.w32_overhead for r in per_cell), default=0)

    n_w32_ratified = sum(1 for r in per_cell if r.w32_ratified)
    n_w32_ratified_correct = sum(
        1 for r in per_cell if r.w32_ratified and r.correct_w32)
    trust_precision_w32 = (
        n_w32_ratified_correct / n_w32_ratified
        if n_w32_ratified > 0 else 1.0)

    n_change_points = sum(1 for r in per_cell if r.w32_change_point)
    n_gold_routes = sum(1 for r in per_cell if r.w32_gold_route)

    n_cram_cells = sum(1 for r in per_cell if r.w32_cram_factor > 0)
    if n_cram_cells > 0:
        mean_cram_w32 = (
            sum(r.w32_cram_factor for r in per_cell
                if r.w32_cram_factor > 0) / float(n_cram_cells))
    else:
        mean_cram_w32 = 0.0

    branch_hist: dict[str, int] = {}
    for r in per_cell:
        branch_hist[r.w32_branch] = branch_hist.get(r.w32_branch, 0) + 1

    n_long_window_updates = int(producer_w32.registry.n_long_window_updates)
    n_w32_envelopes_registered = int(producer_w32.registry.n_w32_registered)
    n_w32_envelopes_rejected = int(producer_w32.registry.n_w32_rejected)

    if n_tamper_attempts > 0:
        tamper_reject_rate = n_tampered_rejected / n_tamper_attempts
    else:
        tamper_reject_rate = None

    # Per-cell trajectory summary.
    convergence_summary = []
    for r in per_cell:
        convergence_summary.append({
            "cell_idx": int(r.cell_idx),
            "w31_branch": r.w31_branch,
            "w32_branch": r.w32_branch,
            "w31_ratified": bool(r.w31_ratified),
            "w32_ratified": bool(r.w32_ratified),
            "correct_w31": bool(r.correct_w31),
            "correct_w32": bool(r.correct_w32),
            "ewma_prior_after": round(float(r.w32_ewma_prior), 4),
            "change_point": bool(r.w32_change_point),
            "gold_route": bool(r.w32_gold_route),
        })

    return {
        "bank": bank,
        "underlying_bank": underlying_bank,
        "n_eval": n_cells_eff,
        "bank_seed": bank_seed,
        "long_window": int(w32_window),
        "ewma_alpha": float(w32_alpha),
        "cusum_threshold": float(w32_cusum_thr),
        "long_window_enabled": bool(w32_long_enabled),
        "change_point_enabled": bool(w32_change_enabled),
        "gold_correlation_enabled": bool(w32_gold_enabled),
        "manifest_v2_disabled": bool(w32_manifest_v2_disabled),
        # Correctness.
        "correctness_ratified_rate_w31": rate_w31,
        "correctness_ratified_rate_w32": rate_w32,
        "delta_w32_minus_w31": delta_w32_w31,
        "trust_precision_w32": trust_precision_w32,
        "n_w32_ratified": int(n_w32_ratified),
        # Wire-token economics.
        "total_w31_visible_tokens": int(total_w31),
        "total_w32_visible_tokens": int(total_w32),
        "mean_overhead_w32_vs_w31_per_cell": float(mean_overhead_w32_vs_w31),
        "max_overhead_w32_vs_w31_per_cell": int(max_overhead_w32_vs_w31),
        # Cram factor.
        "mean_cram_factor_w32": float(mean_cram_w32),
        # Branches / convergence telemetry.
        "branch_hist": branch_hist,
        "n_change_points_fired": int(n_change_points),
        "n_gold_routes_fired": int(n_gold_routes),
        "n_long_window_updates": int(n_long_window_updates),
        "n_w32_envelopes_registered": int(n_w32_envelopes_registered),
        "n_w32_envelopes_rejected": int(n_w32_envelopes_rejected),
        "byte_equivalent_w32_w31": (total_w32 == total_w31),
        # Tamper.
        "n_tamper_attempts": int(n_tamper_attempts),
        "n_tampered_rejected": int(n_tampered_rejected),
        "tamper_reject_rate": tamper_reject_rate,
        # Per-cell trajectory summary.
        "convergence_summary": convergence_summary,
    }


def run_phase79_seed_sweep(
        *,
        bank: str = "drift_recover",
        seeds: tuple[int, ...] = (11, 17, 23, 29, 31),
        T_decoder: int | None = None,
        K_consumers: int = 3,
        n_eval: int = 24,
        bank_replicates: int = 4,
        chain_persist_window: int | None = None,
        max_active_chains: int = 8,
        signature_period: int = 4,
        calibration_stride: int = 8,
        ancestor_window: int = 4,
        long_window: int | None = None,
        ewma_alpha: float | None = None,
) -> dict[str, Any]:
    """Run a seed-sweep over Phase-79 sub-bank ``bank``."""
    seed_results = []
    for s in seeds:
        r = run_phase79(
            bank=bank, T_decoder=T_decoder, K_consumers=K_consumers,
            n_eval=n_eval, bank_replicates=bank_replicates,
            bank_seed=int(s),
            chain_persist_window=chain_persist_window,
            max_active_chains=max_active_chains,
            signature_period=signature_period,
            calibration_stride=calibration_stride,
            ancestor_window=ancestor_window,
            long_window=long_window,
            ewma_alpha=ewma_alpha,
        )
        seed_results.append(r)

    deltas = [r["delta_w32_minus_w31"] for r in seed_results]
    rates_w31 = [r["correctness_ratified_rate_w31"] for r in seed_results]
    rates_w32 = [r["correctness_ratified_rate_w32"] for r in seed_results]
    trust_precs = [r["trust_precision_w32"] for r in seed_results]
    overheads = [r["mean_overhead_w32_vs_w31_per_cell"]
                  for r in seed_results]
    max_overheads = [r["max_overhead_w32_vs_w31_per_cell"]
                      for r in seed_results]

    summary = {
        "bank": bank,
        "seeds": list(seeds),
        "n_seeds": len(seed_results),
        "min_delta_w32_minus_w31": min(deltas) if deltas else None,
        "max_delta_w32_minus_w31": max(deltas) if deltas else None,
        "mean_delta_w32_minus_w31": (sum(deltas) / len(deltas)
                                       if deltas else None),
        "min_correctness_w31": min(rates_w31) if rates_w31 else None,
        "max_correctness_w31": max(rates_w31) if rates_w31 else None,
        "min_correctness_w32": min(rates_w32) if rates_w32 else None,
        "max_correctness_w32": max(rates_w32) if rates_w32 else None,
        "min_trust_precision_w32": min(trust_precs) if trust_precs else None,
        "mean_overhead_w32_vs_w31_per_cell": (
            sum(overheads) / len(overheads) if overheads else None),
        "max_overhead_w32_vs_w31_per_cell": (
            max(max_overheads) if max_overheads else None),
        "all_w32_ge_w31": all(d >= 0 - 1e-9 for d in deltas),
        "all_byte_equivalent_w32_w31": all(
            r["byte_equivalent_w32_w31"] for r in seed_results),
        "n_seeds_clearing_delta_010": sum(1 for d in deltas if d >= 0.10),
        "n_seeds_clearing_delta_005": sum(1 for d in deltas if d >= 0.05),
        "seed_results": seed_results,
    }
    return summary


def run_phase79_long_window_sweep(
        *,
        long_windows: tuple[int, ...] = (16, 32, 64, 128),
        seeds: tuple[int, ...] = (11, 17, 23, 29, 31),
        n_eval: int = 24,
        bank_seed_base: int = 11,
        signature_period: int = 4,
        ewma_alpha: float = W32_DEFAULT_EWMA_ALPHA,
) -> dict[str, Any]:
    """Sweep ``long_window`` ∈ {16, 32, 64, 128} on the multi-shift
    drift-recover regime; characterise scaling.
    """
    sweep = []
    for lw in long_windows:
        r = run_phase79_seed_sweep(
            bank="drift_recover",
            seeds=seeds, n_eval=n_eval,
            signature_period=signature_period,
            long_window=int(lw),
            ewma_alpha=ewma_alpha,
        )
        sweep.append({
            "long_window": int(lw),
            "min_correctness_w32": r["min_correctness_w32"],
            "max_correctness_w32": r["max_correctness_w32"],
            "min_correctness_w31": r["min_correctness_w31"],
            "max_correctness_w31": r["max_correctness_w31"],
            "min_delta_w32_minus_w31": r["min_delta_w32_minus_w31"],
            "max_delta_w32_minus_w31": r["max_delta_w32_minus_w31"],
            "min_trust_precision_w32": r["min_trust_precision_w32"],
            "mean_overhead_w32_vs_w31_per_cell": r[
                "mean_overhead_w32_vs_w31_per_cell"],
            "all_w32_ge_w31": r["all_w32_ge_w31"],
        })
    return {
        "long_windows": list(long_windows),
        "seeds": list(seeds),
        "ewma_alpha": float(ewma_alpha),
        "n_eval": int(n_eval),
        "sweep": sweep,
        # Did the gain monotonically increase, saturate, or degrade?
        "gain_at_window_16": next(
            (s["min_delta_w32_minus_w31"] for s in sweep
             if s["long_window"] == 16), None),
        "gain_at_window_32": next(
            (s["min_delta_w32_minus_w31"] for s in sweep
             if s["long_window"] == 32), None),
        "gain_at_window_64": next(
            (s["min_delta_w32_minus_w31"] for s in sweep
             if s["long_window"] == 64), None),
        "gain_at_window_128": next(
            (s["min_delta_w32_minus_w31"] for s in sweep
             if s["long_window"] == 128), None),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(
        description="Phase 79 — W32 long-window convergent online "
                    "geometry-aware dense control benchmark.")
    p.add_argument("--bank", default="drift_recover", choices=[
        "trivial_w32", "drift_recover", "long_window",
        "manifest_v2_tamper", "no_change_point",
        "mis_correlated_gold", "frozen_ewma", "xllm_live_gold",
    ])
    p.add_argument("--T-decoder", type=int, default=None)
    p.add_argument("--K-consumers", type=int, default=3)
    p.add_argument("--n-eval", type=int, default=24)
    p.add_argument("--bank-replicates", type=int, default=4)
    p.add_argument("--bank-seed", type=int, default=11)
    p.add_argument("--chain-persist-window", type=int, default=None)
    p.add_argument("--max-active-chains", type=int, default=8)
    p.add_argument("--signature-period", type=int, default=4)
    p.add_argument("--calibration-stride", type=int, default=8)
    p.add_argument("--ancestor-window", type=int, default=4)
    p.add_argument("--long-window", type=int, default=None)
    p.add_argument("--ewma-alpha", type=float, default=None)
    p.add_argument("--seed-sweep", action="store_true",
                    help="Run a sweep over seeds {11, 17, 23, 29, 31}.")
    p.add_argument("--long-window-sweep", action="store_true",
                    help="Sweep long_window ∈ {16, 32, 64, 128} on the "
                         "drift-recover regime.")
    p.add_argument("--out-json", default=None,
                    help="If provided, write JSON results here.")
    p.add_argument("--quiet", action="store_true",
                    help="Suppress stdout (still writes --out-json).")
    args = p.parse_args()

    if args.long_window_sweep:
        results = run_phase79_long_window_sweep(
            long_windows=(16, 32, 64, 128),
            seeds=(11, 17, 23, 29, 31),
            n_eval=args.n_eval,
            bank_seed_base=args.bank_seed,
            signature_period=args.signature_period,
            ewma_alpha=(args.ewma_alpha if args.ewma_alpha is not None
                          else W32_DEFAULT_EWMA_ALPHA),
        )
    elif args.seed_sweep:
        results = run_phase79_seed_sweep(
            bank=args.bank, T_decoder=args.T_decoder,
            K_consumers=args.K_consumers,
            n_eval=args.n_eval, bank_replicates=args.bank_replicates,
            chain_persist_window=args.chain_persist_window,
            max_active_chains=args.max_active_chains,
            signature_period=args.signature_period,
            calibration_stride=args.calibration_stride,
            ancestor_window=args.ancestor_window,
            long_window=args.long_window,
            ewma_alpha=args.ewma_alpha,
        )
    else:
        results = run_phase79(
            bank=args.bank, T_decoder=args.T_decoder,
            K_consumers=args.K_consumers,
            n_eval=args.n_eval, bank_replicates=args.bank_replicates,
            bank_seed=args.bank_seed,
            chain_persist_window=args.chain_persist_window,
            max_active_chains=args.max_active_chains,
            signature_period=args.signature_period,
            calibration_stride=args.calibration_stride,
            ancestor_window=args.ancestor_window,
            long_window=args.long_window,
            ewma_alpha=args.ewma_alpha,
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
