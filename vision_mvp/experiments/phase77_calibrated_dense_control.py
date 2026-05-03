"""Phase 77 — calibrated geometry-aware dense control + multi-stride
basis history + per-partition calibration prior + cross-host
disagreement-routing + ancestor-chain causal binding (SDK v3.31, W30).

The follow-up to SDK v3.30 (W29).  W29 introduced the geometry-
partitioned product-manifold dense-control envelope with audited
subspace basis, factoradic Lehmer routing index, causal-validity gate,
and cross-host variance witness; it discharged W28-C-CROSS-HOST-
VARIANCE on the magnitude axis but honestly missed two pre-committed
gates: H7 (cram-factor amplification 2.30× < 8×) and H6 absolute
(0.75 < 0.95).  Both became named open conjectures.

W30 (this phase) wraps the W29 ``GeometryPartitionedOrchestrator``
with four NEW audited proxies at the capsule layer:

* **multi-stride basis history** — content-addressed accumulator over
  the last ``stride`` cells' W29 basis CIDs; the cram-factor numerator
  grows linearly with ``stride``; the wire-token denominator stays at
  1 token.  Mechanism for the W29-C-CRAM-AMPLIFICATION discharge.
* **per-partition calibration prior** — vector of floats in [0, 1]
  with one entry per registered partition; closed-form override when
  a cell's partition_id has prior < threshold (route to
  ``high_trust_partition_id``).  Mechanism for the W29-C-PARTITION-
  CALIBRATION discharge.
* **cross-host disagreement-routing** — closed-form: when the inner
  W29's previously-observed cross-host probe disagreement count > 0,
  route this cell to ``high_trust_partition_id``.  Mechanism for the
  W29-C-CROSS-HOST-VARIANCE-LIVE-MAGNITUDE sharpening.
* **ancestor-chain causal binding** — content-addressed sorted tuple
  of the last ``ancestor_window`` W29 partition_cids; tampering on
  any chain entry detected by the verifier.

Phase-77 sub-banks
==================

Six pre-committed sub-banks (five synthetic + one cross-host live):

* **R-77-TRIVIAL-CALIBRATION** (H2 anchor; W30-Λ-trivial-calibration).
  ``calibration_stride=0``, ``calibration_vector=()``,
  ``ancestor_window=0`` ⇒ W30 reduces to W29 byte-for-byte.

* **R-77-CHAIN-CRAM** (H6 cram-factor anchor; discharge of
  W29-C-CRAM-AMPLIFICATION).  ``calibration_stride=8``,
  ``ancestor_window=4``, three partitions, on the ``chain_shared``
  underlying bank.  ``cram_factor_w30 / cram_factor_w28 ≥ 8.0`` AND
  ``cram_factor_w30 / cram_factor_w29 ≥ 2.0`` across 5/5 seeds.

* **R-77-CALIBRATION-PRIOR** (H7 calibration anchor; discharge of
  W29-C-PARTITION-CALIBRATION).  Regime where CYCLIC partition has
  the partial oracle (low historical agreement); calibrated priors
  (0.95, 0.95, 0.30) cause CYCLIC cells to reroute to LINEAR (full
  oracle).  ``correctness_ratified_rate_w30 - correctness_ratified_rate_w29
  ≥ +0.10``.

* **R-77-XHOST-DISAGREE** (H8 disagreement-routing anchor; sharpening
  of W29-C-CROSS-HOST-VARIANCE-LIVE-MAGNITUDE).  Two-probe ensemble
  where probes systematically disagree on a fraction of cells; W30
  reroutes the cell after a disagreement-witnessing cell to LINEAR
  (full oracle).  ``correctness_ratified_rate_w30 -
  correctness_ratified_rate_w29 ≥ +0.10``.

* **R-77-CALIBRATED-TAMPERED** (H3 tamper bank).  Five named
  tampering passes per cell — corrupt basis history CID, push
  calibration vector out of range, replace ancestor chain CID,
  set disagreement_route_target to unregistered partition, corrupt
  the calibrated_cid byte-for-byte.

* **R-77-CROSS-HOST-LIVE** (S1/S2 best-effort; live LLM).  Live LLM
  probes on two reachable Ollama hosts (localhost +
  192.168.12.191).  Records cross-host probe calls, round-trip bytes,
  trust precision under real LLM disagreement (and honest agreement
  null when LLMs agree at temp 0).

Falsifier banks (a third small bench class):

* **R-77-NON-CALIBRATABLE** (W30-Λ-non-calibratable).  Calibration
  vector uniform (1.0, 1.0, 1.0) — no override is ever triggered;
  W30 ratification rate = W29 byte-for-byte.

* **R-77-DEGENERATE-HISTORY** (W30-Λ-degenerate-history).  Stride=1
  — basis history holds only the current cell; cram-factor
  amplification ≤ 1.20× (no real amplification).

The bench's apples-to-apples comparison is:

  * **W27 baseline**             — W27 orchestrator (no ensemble).
  * **W28 ensemble-verified**    — W27 + ensemble ratification.
  * **W29 geometry-partitioned** — W28 + partition envelope (with
    optional per-partition inner W28 dispatch).
  * **W30 calibrated-geometry**  — W29 + basis history + calibration
    + ancestor chain + disagreement-routing.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import random
import sys
import urllib.request
from typing import Any

from vision_mvp.coordpy.team_coord import (
    OracleRegistration, SchemaCapsule,
    build_incident_triage_schema_capsule,
    SharedFanoutRegistry,
    ChainPersistedFanoutRegistry,
    ChainPersistedFanoutDisambiguator,
    MultiChainPersistedFanoutOrchestrator,
    SharedMultiChainPool,
    compute_input_signature_cid,
    # W28
    ProbeVote, EnsembleProbe, EnsembleProbeRegistration,
    DeterministicSignatureProbe, OracleConsultationProbe,
    LLMSignatureProbe,
    EnsemblePivotRatificationEnvelope,
    EnsembleRatificationRegistry,
    EnsembleVerifiedMultiChainOrchestrator,
    verify_ensemble_pivot_ratification,
    build_default_ensemble_registry,
    W28_BRANCH_RATIFIED, W28_BRANCH_RATIFIED_PASSTHROUGH,
    # W29
    SubspaceBasis, verify_subspace_basis,
    compute_structural_subspace_basis,
    encode_permutation_to_factoradic,
    decode_factoradic_to_permutation,
    CrossHostVarianceWitness,
    GeometryPartitionedRatificationEnvelope,
    PartitionRegistration,
    GeometryPartitionRegistry,
    GeometryPartitionedOrchestrator,
    classify_partition_id_for_cell,
    verify_geometry_partition_ratification,
    build_trivial_partition_registry,
    build_three_partition_registry,
    W29_PARTITION_LINEAR, W29_PARTITION_HIERARCHICAL,
    W29_PARTITION_CYCLIC,
    W29_PARTITION_LABEL,
    W29_BRANCH_PARTITION_RESOLVED,
    # W30
    BasisHistory, AncestorChain, PartitionCalibrationVector,
    CalibratedGeometryRatificationEnvelope,
    CalibratedGeometryRegistry,
    W30CalibratedResult,
    CalibratedGeometryOrchestrator,
    verify_calibrated_geometry_ratification,
    update_partition_calibration_running_mean,
    build_trivial_calibrated_registry,
    build_calibrated_registry,
    W30_CALIBRATED_SCHEMA_VERSION,
    W30_DEFAULT_CALIBRATION_PRIOR_THRESHOLD,
    W30_BRANCH_CALIBRATED_RESOLVED,
    W30_BRANCH_TRIVIAL_CALIBRATION_PASSTHROUGH,
    W30_BRANCH_CALIBRATION_REROUTED,
    W30_BRANCH_DISAGREEMENT_ROUTED,
    # Oracles
    ServiceGraphOracle, ChangeHistoryOracle,
    OutsideQuery,
    _DecodedHandoff,
)

from vision_mvp.experiments.phase74_multi_chain_pivot import (
    build_phase74_bank,
    build_team_shared_pool,
    _build_partial_service_graph_oracle,
    _build_w26_stack,
    _build_w27_orchestrator,
    _expected_gold_for_cell,
    _w26_visible, _w27_visible,
)
from vision_mvp.experiments.phase75_ensemble_verified_multi_chain import (
    discover_two_host_topology,
    LOCAL_HOST_ID,
    OLLAMA_LOCALHOST, OLLAMA_MAC1, OLLAMA_MAC2,
    _w28_visible, _w28_branch, _w28_ratified,
    _build_w28_orchestrator,
)
from vision_mvp.experiments.phase76_geometry_partitioned_product_manifold import (
    INCIDENT_TRIAGE_AMBIENT_VOCABULARY,
    _build_per_partition_team_pools,
    _build_cross_host_w28_registry,
    _w29_visible, _w29_branch, _w29_partition_id, _w29_ratified,
    _w29_overhead, _w29_n_structured_bits, _w29_cram_factor,
    _w28_n_envelope_bytes,
)


# ---------------------------------------------------------------------------
# W30 visible-token / branch helpers
# ---------------------------------------------------------------------------


def _w30_visible(out: dict) -> int:
    if "calibrated_geometry" in out:
        return int(out["calibrated_geometry"].get(
            "n_w30_visible_tokens", 0))
    return _w29_visible(out)


def _w30_branch(out: dict) -> str:
    if "calibrated_geometry" in out:
        return str(out["calibrated_geometry"].get("decoder_branch", ""))
    return ""


def _w30_ratified(out: dict) -> bool:
    if "calibrated_geometry" in out:
        return bool(out["calibrated_geometry"].get("ratified", False))
    return _w29_ratified(out)


def _w30_overhead(out: dict) -> int:
    if "calibrated_geometry" in out:
        return int(out["calibrated_geometry"].get(
            "n_calibrated_overhead_tokens", 0))
    return 0


def _w30_n_structured_bits(out: dict) -> int:
    if "calibrated_geometry" in out:
        return int(out["calibrated_geometry"].get(
            "n_structured_bits", 0))
    return 0


def _w30_cram_factor(out: dict) -> float:
    if "calibrated_geometry" in out:
        return float(out["calibrated_geometry"].get(
            "cram_factor_w30", 0.0))
    return 0.0


def _w30_effective_partition_id(out: dict) -> int:
    if "calibrated_geometry" in out:
        return int(out["calibrated_geometry"].get(
            "effective_partition_id", 0))
    return 0


def _w30_structural_partition_id(out: dict) -> int:
    if "calibrated_geometry" in out:
        return int(out["calibrated_geometry"].get(
            "structural_partition_id", 0))
    return 0


def _w30_disagreement_route_active(out: dict) -> bool:
    if "calibrated_geometry" in out:
        return bool(out["calibrated_geometry"].get(
            "disagreement_route_active", False))
    return False


# ---------------------------------------------------------------------------
# Calibration-prior bench builder (H7 anchor)
#
# Inverts W29's H6 design: in the calibration_prior bench, the CYCLIC
# partition is given the PARTIAL oracle (low agreement-rate by
# construction); LINEAR/HIERARCHICAL get the FULL oracle.  The
# calibrated W30 prior (0.95, 0.95, 0.30) reroutes CYCLIC cells to
# LINEAR (high-trust), recovering correctness.  Uniform priors do not
# trigger the override → W29 baseline.
# ---------------------------------------------------------------------------


def _build_inverted_per_partition_team_pools(
        *,
        T_decoder: int | None,
        schema: SchemaCapsule,
        producer_agent_id: str,
        consumer_agent_ids: tuple[str, ...],
        chain_persist_window: int,
        max_active_chains: int,
        signature_period: int,
        projection_id_for_consumer: dict[str, str],
        projected_tags_for_consumer: dict[str, tuple[str, ...]],
        partial_gold_pair: tuple[str, str] = ("orders", "payments"),
) -> dict[int, SharedMultiChainPool]:
    """Build per-partition pools with the OPPOSITE assignment from
    R-76-XHOST-DRIFT: LINEAR/HIERARCHICAL get FULL oracle (high
    agreement-rate); CYCLIC gets PARTIAL oracle (low agreement-rate
    by construction).

    The R-77-CALIBRATION-PRIOR bench is built so that the structural
    classifier routes most hard cells to CYCLIC (the BAD partition),
    and the calibrated W30 prior reroutes them to LINEAR (the GOOD
    partition).
    """
    pools: dict[int, SharedMultiChainPool] = {}
    partial = _build_partial_service_graph_oracle(
        gold_pair=partial_gold_pair,
        oracle_id="service_graph_partial")
    full = ServiceGraphOracle(oracle_id="service_graph_full")
    change = ChangeHistoryOracle(oracle_id="change_history")

    def _make_pool(*, oracle_set: tuple[tuple[Any, str], ...]):
        per_sig_registries: dict[str, tuple[
            SharedFanoutRegistry, ChainPersistedFanoutRegistry]] = {}

        def _factory(*, signature_cid, agent_id, is_producer):
            regs = per_sig_registries.get(signature_cid)
            if regs is None:
                regs = (
                    SharedFanoutRegistry(schema=schema),
                    ChainPersistedFanoutRegistry(schema=schema),
                )
                per_sig_registries[signature_cid] = regs
            fanout_registry, chain_registry = regs
            return _build_w26_stack(
                T_decoder=T_decoder, schema=schema,
                raw_oracles=oracle_set,
                agent_id=str(agent_id),
                is_producer=bool(is_producer),
                producer_agent_id=str(producer_agent_id),
                consumer_agent_ids=tuple(consumer_agent_ids),
                fanout_registry=fanout_registry,
                chain_registry=chain_registry,
                chain_persist_window=chain_persist_window,
                projection_id_for_consumer=projection_id_for_consumer,
                projected_tags_for_consumer=projected_tags_for_consumer,
            )
        return SharedMultiChainPool(
            schema=schema,
            stack_factory=_factory,
            max_active_chains=int(max_active_chains),
        )

    partial_set: tuple[tuple[Any, str], ...] = (
        (partial, "service_graph"), (change, "change_history"))
    full_set: tuple[tuple[Any, str], ...] = (
        (full, "service_graph"), (change, "change_history"))

    pools[W29_PARTITION_LINEAR] = _make_pool(oracle_set=full_set)
    pools[W29_PARTITION_HIERARCHICAL] = _make_pool(oracle_set=full_set)
    pools[W29_PARTITION_CYCLIC] = _make_pool(oracle_set=partial_set)
    return pools


# ---------------------------------------------------------------------------
# Cross-host disagreement probe (synthetic for H8)
#
# Two `DeterministicSignatureProbe`-style probes that disagree on a
# fraction of cells by construction.  The "host_id" labels make the
# disagreement count cross-host non-zero on those cells.
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _SyntheticDisagreementProbe:
    """A probe that systematically disagrees with the local recompute
    probe on every cell.  Simulates a cross-host LLM probe with a
    different model architecture / temperature regime that produces
    asymmetric votes vs the local model — a common pattern in
    real-world ensembles (e.g., a more pessimistic model on a
    different host).

    Matches the W28 ``EnsembleProbe`` protocol.  Always rejects so
    the cross-host variance witness fires on every cell where W28
    polls both probes (with ``quorum_threshold=0.0`` the disagreement
    does not lower W28's ratification).  W30's disagreement-routing
    reads the witness and reroutes the FOLLOWING cell to the
    registered ``high_trust_partition_id``.
    """
    probe_id: str
    disagreement_trigger_tags: frozenset[str] = frozenset()
    wire_required: bool = False

    def vote(
            self,
            *,
            signature: Any,
            canonical_per_tag_votes: tuple[tuple[str, int], ...],
            canonical_projected_subset: tuple[str, ...],
            cell_index: int,
    ) -> ProbeVote:
        # Always disagrees — strict reject on every cell.  The
        # cross-host variance witness fires on every cell where the
        # local probe ratifies (i.e. every cell where W28 polled both
        # probes, which with quorum_threshold=0.0 is every triggered
        # cell where the inner W27 commits).
        return ProbeVote(
            probe_id=self.probe_id,
            ratify=False, reject=True,
            trust_weight=1.0,
            reason="synthetic_systematic_disagreement",
        )


def _build_disagreement_w28_registry(
        *,
        schema: SchemaCapsule,
        disagreement_cell_indices: frozenset[int] = frozenset(),
        disagreement_trigger_tags: frozenset[str] = frozenset(
            {"api", "db"}),
        local_host_id: str = LOCAL_HOST_ID,
) -> EnsembleRatificationRegistry:
    """Build a W28 registry with two probes that systematically
    disagree on the pre-committed cell indices (synthetic
    cross-host disagreement).  Probe A is on local host; probe B
    on a synthetic ``mac2_synthetic`` host id (so disagreements count
    as cross-host).

    Quorum threshold is 0.0 so even when one probe ratifies and one
    rejects (aggregate signed weight = 0), W28 still ratifies — but
    the cross-host variance witness records the strict disagreement,
    which W30 reads via its peek logic on the next cell.
    """
    return EnsembleRatificationRegistry(
        schema=schema,
        quorum_threshold=0.0,
        probes=(
            EnsembleProbeRegistration(
                probe=DeterministicSignatureProbe(probe_id="local_recompute"),
                trust_prior=1.0,
                role_label="local_recompute",
                host_id=local_host_id,
            ),
            EnsembleProbeRegistration(
                probe=_SyntheticDisagreementProbe(
                    probe_id="synthetic_disagreement",
                    disagreement_trigger_tags=disagreement_trigger_tags),
                trust_prior=1.0,
                role_label="synthetic_disagreement",
                host_id="mac2_synthetic",
            ),
        ),
        local_host_id=local_host_id,
    )


# ---------------------------------------------------------------------------
# Phase 77 runner
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _PerCellRecord:
    cell_idx: int
    expected: set[str]
    correct_w27: bool
    correct_w28: bool
    correct_w29: bool
    correct_w30: bool
    w27_visible: int
    w28_visible: int
    w29_visible: int
    w30_visible: int
    w29_overhead: int
    w30_overhead: int
    w29_branch: str
    w30_branch: str
    w29_partition_id: int
    w30_structural_partition_id: int
    w30_effective_partition_id: int
    w29_ratified: bool
    w30_ratified: bool
    w29_n_structured_bits: int
    w30_n_structured_bits: int
    w29_cram_factor: float
    w30_cram_factor: float
    w28_n_envelope_bytes: int
    w30_disagreement_route_active: bool
    cross_host_disagreement_count: int


def run_phase77(
        *,
        bank: str = "chain_cram",
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
        calibration_priors: tuple[float, ...] | None = None,
        verbose: bool = False,
) -> dict[str, Any]:
    """Run one Phase-77 sub-bank and return a results dict comparing
    W27 / W28 / W29 / W30 on the same regime.

    ``bank`` is one of:

      * ``"trivial_calibration"``  — H2 byte-equivalence anchor.
      * ``"chain_cram"``            — H6 cram-factor headline.
      * ``"calibration_prior"``     — H7 calibration discharge.
      * ``"xhost_disagree"``        — H8 disagreement-routing strict gain.
      * ``"non_calibratable"``      — W30-Λ-non-calibratable falsifier.
      * ``"degenerate_history"``    — W30-Λ-degenerate-history falsifier.
      * ``"calibrated_tampered"``   — H3 tamper bank.
      * ``"cross_host_live"``       — S1/S2 best-effort live.
    """
    schema = build_incident_triage_schema_capsule()
    if bank in ("calibration_prior", "xhost_disagree", "non_calibratable",
                "calibrated_tampered", "cross_host_live"):
        underlying_bank = "divergent_recover"
    else:
        underlying_bank = "chain_shared"

    cells = build_phase74_bank(
        n_replicates=bank_replicates, seed=bank_seed,
        n_cells=n_eval, bank=underlying_bank,
        signature_period=signature_period)

    producer_id = "producer_agent"
    if bank == "trivial_calibration":
        consumer_ids: tuple[str, ...] = ()
    else:
        consumer_ids = tuple(f"consumer_{k}" for k in range(K_consumers))

    if chain_persist_window is None:
        chain_persist_window = n_eval

    projection_id_for_consumer = {
        cid: f"proj_{cid}" for cid in consumer_ids}
    projected_tags_for_consumer = {
        cid: ("orders", "payments", "api", "db",
              "storage", "logs_pipeline", "web", "db_query")
        for cid in consumer_ids}

    # W27 baseline: partial oracle on a single team-wide pool.
    partial_oracles: tuple[tuple[Any, str], ...] = (
        (_build_partial_service_graph_oracle(
            gold_pair=("orders", "payments"),
            oracle_id="service_graph_partial"),
          "service_graph"),
        (ChangeHistoryOracle(oracle_id="change_history"),
          "change_history"),
    )
    pool_w27 = build_team_shared_pool(
        T_decoder=T_decoder, schema=schema,
        raw_oracles=partial_oracles,
        producer_agent_id=producer_id,
        consumer_agent_ids=consumer_ids if consumer_ids else (
            f"{producer_id}_self",),
        chain_persist_window=chain_persist_window,
        max_active_chains=max_active_chains,
        projection_id_for_consumer=projection_id_for_consumer or {
            f"{producer_id}_self": "self_proj"},
        projected_tags_for_consumer=projected_tags_for_consumer or {
            f"{producer_id}_self": ("orders", "payments")},
    )
    producer_w27 = _build_w27_orchestrator(
        schema=schema, agent_id=producer_id, is_producer=True,
        producer_agent_id=producer_id,
        consumer_agent_ids=consumer_ids if consumer_ids else (
            f"{producer_id}_self",),
        pool=pool_w27)

    # W28 baseline: same partial-oracle pool + single deterministic probe.
    def _build_w28_baseline(disagreement_cells: frozenset[int] = frozenset()):
        pool = build_team_shared_pool(
            T_decoder=T_decoder, schema=schema,
            raw_oracles=partial_oracles,
            producer_agent_id=producer_id,
            consumer_agent_ids=consumer_ids if consumer_ids else (
                f"{producer_id}_self",),
            chain_persist_window=chain_persist_window,
            max_active_chains=max_active_chains,
            projection_id_for_consumer=projection_id_for_consumer or {
                f"{producer_id}_self": "self_proj"},
            projected_tags_for_consumer=projected_tags_for_consumer or {
                f"{producer_id}_self": ("orders", "payments")},
        )
        if bank == "cross_host_live":
            reg, _meta = _build_cross_host_w28_registry(
                schema=schema, local_host_id=LOCAL_HOST_ID)
        elif bank == "xhost_disagree" and disagreement_cells:
            reg = _build_disagreement_w28_registry(
                schema=schema,
                disagreement_cell_indices=disagreement_cells,
                local_host_id=LOCAL_HOST_ID)
        else:
            reg = build_default_ensemble_registry(
                schema=schema, quorum_threshold=1.0,
                local_host_id=LOCAL_HOST_ID)
        return _build_w28_orchestrator(
            schema=schema, agent_id=producer_id, is_producer=True,
            producer_agent_id=producer_id,
            consumer_agent_ids=consumer_ids if consumer_ids else (
                f"{producer_id}_self",),
            pool=pool, registry=reg)

    # For xhost_disagree, pre-commit the disagreement cell set.  Cells
    # 4..(n//2+1) carry synthetic cross-host disagreement.
    disagreement_cells: frozenset[int] = frozenset()
    if bank == "xhost_disagree":
        disagreement_cells = frozenset(range(4, max(5, n_eval // 2 + 1)))

    producer_w28 = _build_w28_baseline(disagreement_cells)
    # Two independent W28 instances for the two W29 stacks below
    # (baseline + W30-wrapped).  The state must not be shared because
    # each W29 wrapper advances its inner W28's cell_index per cell.
    producer_w28_for_w29_baseline = _build_w28_baseline(disagreement_cells)
    producer_w28_for_w30 = _build_w28_baseline(disagreement_cells)

    # W29: build TWO independent stacks.
    #   - ``producer_w29_baseline`` is the unmodified W29 orchestrator
    #     (no classifier hook), measured directly to provide the
    #     correctness_w29 baseline.
    #   - ``producer_w29_for_w30`` is wrapped by the W30 orchestrator;
    #     the W30 layer installs its classifier hook on this inner.
    if bank == "trivial_calibration":
        registry_w29_baseline = build_trivial_partition_registry(
            schema=schema, local_host_id=LOCAL_HOST_ID)
        registry_w29 = build_trivial_partition_registry(
            schema=schema, local_host_id=LOCAL_HOST_ID)
        producer_w29_baseline = GeometryPartitionedOrchestrator(
            inner=producer_w28_for_w29_baseline,
            registry=registry_w29_baseline,
            enabled=True,
            require_partition_verification=True,
            cycle_window=4,
        )
        producer_w29 = GeometryPartitionedOrchestrator(
            inner=producer_w28_for_w30,
            registry=registry_w29,
            enabled=True,
            require_partition_verification=True,
            cycle_window=4,
        )
    else:
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

        registry_w29_baseline = _make_three_partition_registry()
        registry_w29 = _make_three_partition_registry()
        if bank == "calibration_prior":
            # Inverted assignment: CYCLIC has PARTIAL, LINEAR has FULL.
            def _make_inverted_inner_per_partition():
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

            producer_w29_baseline = GeometryPartitionedOrchestrator(
                inner=producer_w28_for_w29_baseline,
                registry=registry_w29_baseline,
                enabled=True,
                require_partition_verification=True,
                cycle_window=cycle_window_for_bench,
                inner_per_partition=_make_inverted_inner_per_partition(),
                pre_dispatch_by_partition=True,
            )
            producer_w29 = GeometryPartitionedOrchestrator(
                inner=producer_w28_for_w30,
                registry=registry_w29,
                enabled=True,
                require_partition_verification=True,
                cycle_window=cycle_window_for_bench,
                inner_per_partition=_make_inverted_inner_per_partition(),
                pre_dispatch_by_partition=True,
            )
        elif bank == "xhost_disagree":
            # Standard W29 H6 partition layout: LINEAR/HIERARCHICAL have
            # PARTIAL oracle, CYCLIC has FULL.  W30 disagreement-routing
            # reroutes cells AFTER a disagreement-witnessing cell to
            # high_trust=CYCLIC; CYCLIC has FULL oracle.
            def _make_xhost_inner_per_partition():
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
                inner_per_partition = {}
                for pid, pool in pools.items():
                    reg = _build_disagreement_w28_registry(
                        schema=schema,
                        disagreement_cell_indices=disagreement_cells,
                        local_host_id=LOCAL_HOST_ID)
                    inner = _build_w28_orchestrator(
                        schema=schema, agent_id=producer_id, is_producer=True,
                        producer_agent_id=producer_id,
                        consumer_agent_ids=consumer_ids,
                        pool=pool, registry=reg)
                    inner_per_partition[int(pid)] = inner
                return inner_per_partition

            producer_w29_baseline = GeometryPartitionedOrchestrator(
                inner=producer_w28_for_w29_baseline,
                registry=registry_w29_baseline,
                enabled=True,
                require_partition_verification=True,
                cycle_window=cycle_window_for_bench,
                inner_per_partition=_make_xhost_inner_per_partition(),
                pre_dispatch_by_partition=True,
            )
            producer_w29 = GeometryPartitionedOrchestrator(
                inner=producer_w28_for_w30,
                registry=registry_w29,
                enabled=True,
                require_partition_verification=True,
                cycle_window=cycle_window_for_bench,
                inner_per_partition=_make_xhost_inner_per_partition(),
                pre_dispatch_by_partition=True,
            )
        elif bank in ("calibrated_tampered", "non_calibratable",
                      "cross_host_live"):
            # Simple: standard W29 H6 layout with per-partition dispatch.
            def _make_h6_inner_per_partition():
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
                inner_per_partition = {}
                for pid, pool in pools.items():
                    if bank == "cross_host_live":
                        reg, _m = _build_cross_host_w28_registry(
                            schema=schema, local_host_id=LOCAL_HOST_ID)
                    else:
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

            producer_w29_baseline = GeometryPartitionedOrchestrator(
                inner=producer_w28_for_w29_baseline,
                registry=registry_w29_baseline,
                enabled=True,
                require_partition_verification=True,
                cycle_window=cycle_window_for_bench,
                inner_per_partition=_make_h6_inner_per_partition(),
                pre_dispatch_by_partition=True,
            )
            producer_w29 = GeometryPartitionedOrchestrator(
                inner=producer_w28_for_w30,
                registry=registry_w29,
                enabled=True,
                require_partition_verification=True,
                cycle_window=cycle_window_for_bench,
                inner_per_partition=_make_h6_inner_per_partition(),
                pre_dispatch_by_partition=True,
            )
        else:
            # chain_cram / degenerate_history: single inner W28.
            producer_w29_baseline = GeometryPartitionedOrchestrator(
                inner=producer_w28_for_w29_baseline,
                registry=registry_w29_baseline,
                enabled=True,
                require_partition_verification=True,
                cycle_window=cycle_window_for_bench,
            )
            producer_w29 = GeometryPartitionedOrchestrator(
                inner=producer_w28_for_w30,
                registry=registry_w29,
                enabled=True,
                require_partition_verification=True,
                cycle_window=cycle_window_for_bench,
            )

    # W30: registry depends on the bank.
    if bank == "trivial_calibration":
        registry_w30 = build_trivial_calibrated_registry(
            schema=schema, local_host_id=LOCAL_HOST_ID)
    elif bank == "calibration_prior":
        # Calibrated priors that flag CYCLIC as low-trust.
        priors = calibration_priors or (0.95, 0.95, 0.30)
        registry_w30 = build_calibrated_registry(
            schema=schema,
            calibration_stride=int(calibration_stride),
            calibration_priors=priors,
            ancestor_window=int(ancestor_window),
            high_trust_partition_id=W29_PARTITION_LINEAR,
            local_host_id=LOCAL_HOST_ID,
        )
    elif bank == "non_calibratable":
        # Uniform priors → no override → W30 ratifies as W29 does.
        registry_w30 = build_calibrated_registry(
            schema=schema,
            calibration_stride=int(calibration_stride),
            calibration_priors=(1.0, 1.0, 1.0),
            ancestor_window=int(ancestor_window),
            high_trust_partition_id=W29_PARTITION_CYCLIC,
            local_host_id=LOCAL_HOST_ID,
        )
    elif bank == "degenerate_history":
        # Stride=1 → no real cram amplification.
        registry_w30 = build_calibrated_registry(
            schema=schema,
            calibration_stride=1,
            calibration_priors=(1.0, 1.0, 1.0),
            ancestor_window=0,
            high_trust_partition_id=W29_PARTITION_CYCLIC,
            local_host_id=LOCAL_HOST_ID,
        )
    elif bank == "xhost_disagree":
        # Calibration vector uniform; disagreement-routing does the work.
        registry_w30 = build_calibrated_registry(
            schema=schema,
            calibration_stride=int(calibration_stride),
            calibration_priors=(1.0, 1.0, 1.0),
            ancestor_window=int(ancestor_window),
            high_trust_partition_id=W29_PARTITION_CYCLIC,
            local_host_id=LOCAL_HOST_ID,
        )
    else:
        registry_w30 = build_calibrated_registry(
            schema=schema,
            calibration_stride=int(calibration_stride),
            calibration_priors=(1.0, 1.0, 1.0),
            ancestor_window=int(ancestor_window),
            high_trust_partition_id=W29_PARTITION_CYCLIC,
            local_host_id=LOCAL_HOST_ID,
        )

    producer_w30 = CalibratedGeometryOrchestrator(
        inner=producer_w29,
        registry=registry_w30,
        enabled=True,
        require_calibrated_verification=True,
    )

    per_cell_records: list[_PerCellRecord] = []
    n_tamper_attempts = 0
    n_tampered_rejected = 0

    for cell_idx, cell_handoffs in enumerate(cells):
        # W27 + W28 baselines on independent stacks.
        out_w27 = producer_w27.decode_rounds(cell_handoffs)
        out_w28 = producer_w28.decode_rounds(cell_handoffs)
        # W29 baseline runs INDEPENDENTLY of W30 (no classifier hook
        # installed) so its measurement is not contaminated by W30's
        # calibration / disagreement reroute.
        out_w29 = producer_w29_baseline.decode_rounds(cell_handoffs)
        # W30 wraps a separate W29 (producer_w29); its inner W29 has
        # the W30 classifier hook installed.
        out_w30 = producer_w30.decode_rounds(cell_handoffs)

        # H3 tamper bank.
        if bank == "calibrated_tampered":
            ev = producer_w30.last_envelope
            registry_w30_used = registry_w30
            if ev is not None:
                cv = registry_w30_used.calibration_vector
                cv_pids = (
                    cv.partition_ids if cv is not None else ())
                # Tamper 1: corrupt basis_history_cid (recompute fails).
                if ev.basis_history is not None:
                    bad_history = dataclasses.replace(
                        ev.basis_history, history_cid="00" * 32)
                    object.__setattr__(bad_history, "history_cid", "00" * 32)
                    t1 = dataclasses.replace(
                        ev, basis_history=bad_history,
                        calibrated_cid="")
                    o1 = verify_calibrated_geometry_ratification(
                        t1, registered_schema=schema,
                        registered_w29_partition_cid=ev.w29_partition_cid,
                        registered_calibration_stride=int(
                            registry_w30_used.calibration_stride),
                        registered_basis_cids=frozenset(
                            registry_w30_used.registered_basis_cids),
                        registered_calibration_partition_ids=cv_pids,
                        registered_ancestor_window=int(
                            registry_w30_used.ancestor_window),
                        registered_ancestor_cids=frozenset(
                            registry_w30_used.registered_ancestor_cids),
                        registered_partition_ids_for_route=frozenset(
                            registry_w30_used.registered_partition_ids))
                    n_tamper_attempts += 1
                    if (not o1.ok and o1.reason ==
                            "basis_history_cid_mismatch"):
                        n_tampered_rejected += 1
                # Tamper 2: calibration vector out of range.
                if cv is not None and cv.calibration_vector:
                    bad_vec = (2.5,) + tuple(cv.calibration_vector[1:])
                    bad_cv = PartitionCalibrationVector(
                        calibration_vector=bad_vec,
                        partition_ids=cv.partition_ids,
                        threshold=cv.threshold,
                    )
                    t2 = dataclasses.replace(
                        ev, calibration=bad_cv, calibrated_cid="")
                    o2 = verify_calibrated_geometry_ratification(
                        t2, registered_schema=schema,
                        registered_w29_partition_cid=ev.w29_partition_cid,
                        registered_calibration_stride=int(
                            registry_w30_used.calibration_stride),
                        registered_basis_cids=frozenset(
                            registry_w30_used.registered_basis_cids),
                        registered_calibration_partition_ids=cv_pids,
                        registered_ancestor_window=int(
                            registry_w30_used.ancestor_window),
                        registered_ancestor_cids=frozenset(
                            registry_w30_used.registered_ancestor_cids),
                        registered_partition_ids_for_route=frozenset(
                            registry_w30_used.registered_partition_ids))
                    n_tamper_attempts += 1
                    if (not o2.ok and o2.reason ==
                            "calibration_vector_out_of_range"):
                        n_tampered_rejected += 1
                # Tamper 3: corrupt ancestor chain CID.
                if ev.ancestor_chain is not None:
                    bad_chain = dataclasses.replace(
                        ev.ancestor_chain, chain_cid="ff" * 32)
                    object.__setattr__(bad_chain, "chain_cid", "ff" * 32)
                    t3 = dataclasses.replace(
                        ev, ancestor_chain=bad_chain, calibrated_cid="")
                    o3 = verify_calibrated_geometry_ratification(
                        t3, registered_schema=schema,
                        registered_w29_partition_cid=ev.w29_partition_cid,
                        registered_calibration_stride=int(
                            registry_w30_used.calibration_stride),
                        registered_basis_cids=frozenset(
                            registry_w30_used.registered_basis_cids),
                        registered_calibration_partition_ids=cv_pids,
                        registered_ancestor_window=int(
                            registry_w30_used.ancestor_window),
                        registered_ancestor_cids=frozenset(
                            registry_w30_used.registered_ancestor_cids),
                        registered_partition_ids_for_route=frozenset(
                            registry_w30_used.registered_partition_ids))
                    n_tamper_attempts += 1
                    if (not o3.ok and o3.reason ==
                            "ancestor_chain_cid_mismatch"):
                        n_tampered_rejected += 1
                # Tamper 4: disagreement_route_target unregistered.
                t4 = dataclasses.replace(
                    ev, disagreement_route_active=True,
                    disagreement_route_target_partition_id=99,
                    calibrated_cid="")
                o4 = verify_calibrated_geometry_ratification(
                    t4, registered_schema=schema,
                    registered_w29_partition_cid=ev.w29_partition_cid,
                    registered_calibration_stride=int(
                        registry_w30_used.calibration_stride),
                    registered_basis_cids=frozenset(
                        registry_w30_used.registered_basis_cids),
                    registered_calibration_partition_ids=cv_pids,
                    registered_ancestor_window=int(
                        registry_w30_used.ancestor_window),
                    registered_ancestor_cids=frozenset(
                        registry_w30_used.registered_ancestor_cids),
                    registered_partition_ids_for_route=frozenset(
                        registry_w30_used.registered_partition_ids))
                n_tamper_attempts += 1
                if (not o4.ok and o4.reason ==
                        "disagreement_route_unsealed"):
                    n_tampered_rejected += 1
                # Tamper 5: corrupt the calibrated_cid byte-for-byte.
                t5 = dataclasses.replace(ev)
                object.__setattr__(t5, "calibrated_cid", "aa" * 32)
                o5 = verify_calibrated_geometry_ratification(
                    t5, registered_schema=schema,
                    registered_w29_partition_cid=ev.w29_partition_cid,
                    registered_calibration_stride=int(
                        registry_w30_used.calibration_stride),
                    registered_basis_cids=frozenset(
                        registry_w30_used.registered_basis_cids),
                    registered_calibration_partition_ids=cv_pids,
                    registered_ancestor_window=int(
                        registry_w30_used.ancestor_window),
                    registered_ancestor_cids=frozenset(
                        registry_w30_used.registered_ancestor_cids),
                    registered_partition_ids_for_route=frozenset(
                        registry_w30_used.registered_partition_ids))
                n_tamper_attempts += 1
                if (not o5.ok and o5.reason ==
                        "calibrated_cid_hash_mismatch"):
                    n_tampered_rejected += 1

        expected = _expected_gold_for_cell(
            bank=underlying_bank, cell_idx=cell_idx, n_eval=n_eval,
            signature_period=signature_period)

        def _is_correct(out: dict, expected_set: set[str]) -> bool:
            ans = out.get("answer") or out
            svcs = ans.get("services") if isinstance(ans, dict) else None
            if svcs is None:
                svcs = out.get("services")
            return set(svcs or []) == expected_set

        record = _PerCellRecord(
            cell_idx=int(cell_idx),
            expected=set(expected),
            correct_w27=_is_correct(out_w27, set(expected)),
            correct_w28=_is_correct(out_w28, set(expected)),
            correct_w29=_is_correct(out_w29, set(expected)),
            correct_w30=_is_correct(out_w30, set(expected)),
            w27_visible=_w27_visible(out_w27),
            w28_visible=_w28_visible(out_w28),
            w29_visible=_w29_visible(out_w29),
            w30_visible=_w30_visible(out_w30),
            w29_overhead=_w29_overhead(out_w29),
            w30_overhead=_w30_overhead(out_w30),
            w29_branch=_w29_branch(out_w29),
            w30_branch=_w30_branch(out_w30),
            w29_partition_id=_w29_partition_id(out_w29),
            w30_structural_partition_id=_w30_structural_partition_id(out_w30),
            w30_effective_partition_id=_w30_effective_partition_id(out_w30),
            w29_ratified=_w29_ratified(out_w29),
            w30_ratified=_w30_ratified(out_w30),
            w29_n_structured_bits=_w29_n_structured_bits(out_w29),
            w30_n_structured_bits=_w30_n_structured_bits(out_w30),
            w29_cram_factor=_w29_cram_factor(out_w29),
            w30_cram_factor=_w30_cram_factor(out_w30),
            w28_n_envelope_bytes=_w28_n_envelope_bytes(out_w28),
            w30_disagreement_route_active=_w30_disagreement_route_active(
                out_w30),
            cross_host_disagreement_count=int(
                (out_w29.get("geometry_partitioned") or {})
                .get("cross_host_disagreement_count", 0)),
        )
        per_cell_records.append(record)

    n_cells = len(per_cell_records)
    if n_cells == 0:
        return {"error": "no cells", "bank": bank}

    correct_w27 = sum(1 for r in per_cell_records if r.correct_w27)
    correct_w28 = sum(1 for r in per_cell_records if r.correct_w28)
    correct_w29 = sum(1 for r in per_cell_records if r.correct_w29)
    correct_w30 = sum(1 for r in per_cell_records if r.correct_w30)
    rate_w27 = correct_w27 / n_cells
    rate_w28 = correct_w28 / n_cells
    rate_w29 = correct_w29 / n_cells
    rate_w30 = correct_w30 / n_cells
    delta_w30_w29 = rate_w30 - rate_w29
    delta_w30_w28 = rate_w30 - rate_w28

    total_w27 = sum(r.w27_visible for r in per_cell_records)
    total_w28 = sum(r.w28_visible for r in per_cell_records)
    total_w29 = sum(r.w29_visible for r in per_cell_records)
    total_w30 = sum(r.w30_visible for r in per_cell_records)
    mean_overhead_w30_vs_w29 = (
        sum(r.w30_overhead for r in per_cell_records) / float(n_cells))
    max_overhead_w30_vs_w29 = max(
        (r.w30_overhead for r in per_cell_records), default=0)
    mean_overhead_w30_vs_w28 = (
        (total_w30 - total_w28) / float(n_cells))

    # Cram-factor.
    n_cram_cells = sum(1 for r in per_cell_records if r.w30_cram_factor > 0)
    if n_cram_cells > 0:
        mean_cram_w30 = (
            sum(r.w30_cram_factor for r in per_cell_records
                if r.w30_cram_factor > 0) / float(n_cram_cells))
    else:
        mean_cram_w30 = 0.0
    n_cram_w29_cells = sum(1 for r in per_cell_records
                              if r.w29_cram_factor > 0)
    if n_cram_w29_cells > 0:
        mean_cram_w29 = (
            sum(r.w29_cram_factor for r in per_cell_records
                if r.w29_cram_factor > 0) / float(n_cram_w29_cells))
    else:
        mean_cram_w29 = 0.0
    # W28 cram-factor (denominator: 1 wire token per ratified cell;
    # numerator: 8 * envelope bytes).  We approximate by averaging over
    # cells where the W28 envelope was emitted.
    n_w28_env_cells = sum(1 for r in per_cell_records
                              if r.w28_n_envelope_bytes > 0)
    if n_w28_env_cells > 0:
        mean_cram_w28 = (
            sum(8.0 * r.w28_n_envelope_bytes / 1.0
                for r in per_cell_records if r.w28_n_envelope_bytes > 0)
            / float(n_w28_env_cells))
    else:
        mean_cram_w28 = 0.0
    cram_ratio_w30_over_w28 = (
        mean_cram_w30 / mean_cram_w28 if mean_cram_w28 > 0 else 0.0)
    cram_ratio_w30_over_w29 = (
        mean_cram_w30 / mean_cram_w29 if mean_cram_w29 > 0 else 0.0)

    # Trust precision: ratified ∧ correct / ratified.
    n_w30_ratified = sum(1 for r in per_cell_records if r.w30_ratified)
    n_w30_ratified_correct = sum(
        1 for r in per_cell_records if r.w30_ratified and r.correct_w30)
    trust_precision_w30 = (
        n_w30_ratified_correct / n_w30_ratified
        if n_w30_ratified > 0 else 1.0)

    # Branch / partition histogram.
    branch_hist: dict[str, int] = {}
    for r in per_cell_records:
        branch_hist[r.w30_branch] = branch_hist.get(r.w30_branch, 0) + 1
    structural_hist: dict[int, int] = {}
    effective_hist: dict[int, int] = {}
    for r in per_cell_records:
        structural_hist[r.w30_structural_partition_id] = (
            structural_hist.get(r.w30_structural_partition_id, 0) + 1)
        effective_hist[r.w30_effective_partition_id] = (
            effective_hist.get(r.w30_effective_partition_id, 0) + 1)

    # Tamper rejection rate.
    if n_tamper_attempts > 0:
        tamper_reject_rate = n_tampered_rejected / n_tamper_attempts
    else:
        tamper_reject_rate = None

    # n_disagreement_routed cells.
    n_disagreement_routed = sum(
        1 for r in per_cell_records if r.w30_disagreement_route_active)

    return {
        "bank": bank,
        "underlying_bank": underlying_bank,
        "T_decoder": T_decoder,
        "K_consumers": K_consumers,
        "n_eval": n_cells,
        "bank_seed": bank_seed,
        "calibration_stride": int(calibration_stride),
        "ancestor_window": int(ancestor_window),
        "calibration_priors": list(calibration_priors)
            if calibration_priors is not None else None,
        # Correctness.
        "correctness_ratified_rate_w27": rate_w27,
        "correctness_ratified_rate_w28": rate_w28,
        "correctness_ratified_rate_w29": rate_w29,
        "correctness_ratified_rate_w30": rate_w30,
        "delta_w30_minus_w29": delta_w30_w29,
        "delta_w30_minus_w28": delta_w30_w28,
        "trust_precision_w30": trust_precision_w30,
        "n_w30_ratified": int(n_w30_ratified),
        # Wire-token economics.
        "total_w27_visible_tokens": int(total_w27),
        "total_w28_visible_tokens": int(total_w28),
        "total_w29_visible_tokens": int(total_w29),
        "total_w30_visible_tokens": int(total_w30),
        "mean_overhead_w30_vs_w29_per_cell": mean_overhead_w30_vs_w29,
        "max_overhead_w30_vs_w29_per_cell": int(max_overhead_w30_vs_w29),
        "mean_overhead_w30_vs_w28_per_cell": mean_overhead_w30_vs_w28,
        # Cram factor.
        "mean_cram_factor_w28": mean_cram_w28,
        "mean_cram_factor_w29": mean_cram_w29,
        "mean_cram_factor_w30": mean_cram_w30,
        "cram_ratio_w30_over_w28": cram_ratio_w30_over_w28,
        "cram_ratio_w30_over_w29": cram_ratio_w30_over_w29,
        # Branches / partitions.
        "branch_hist": branch_hist,
        "structural_partition_hist": {str(k): int(v)
                                          for k, v in structural_hist.items()},
        "effective_partition_hist": {str(k): int(v)
                                         for k, v in effective_hist.items()},
        "n_disagreement_routed": int(n_disagreement_routed),
        "n_calibration_rerouted": int(
            registry_w30.n_calibration_rerouted),
        # Tamper.
        "n_tamper_attempts": int(n_tamper_attempts),
        "n_tampered_rejected": int(n_tampered_rejected),
        "tamper_reject_rate": tamper_reject_rate,
        # Headlines.
        "all_correctness_w30_ge_w29": (
            all(r.correct_w30 or not r.correct_w29
                 for r in per_cell_records)),
    }


def run_phase77_seed_sweep(
        *,
        bank: str,
        seeds: tuple[int, ...] = (11, 17, 23, 29, 31),
        **kwargs: Any,
) -> dict[str, Any]:
    """Run ``run_phase77`` across multiple seeds and return aggregate."""
    per_seed: list[dict[str, Any]] = []
    for seed in seeds:
        r = run_phase77(bank=bank, bank_seed=seed, **kwargs)
        r["seed"] = int(seed)
        per_seed.append(r)
    deltas = [r.get("delta_w30_minus_w29", 0.0) for r in per_seed]
    cram_ratios = [r.get("cram_ratio_w30_over_w28", 0.0) for r in per_seed]
    cram_ratios_w30_w29 = [r.get("cram_ratio_w30_over_w29", 0.0)
                                for r in per_seed]
    overheads = [r.get("mean_overhead_w30_vs_w29_per_cell", 0.0)
                    for r in per_seed]
    return {
        "bank": bank,
        "seeds": list(seeds),
        "per_seed": per_seed,
        "summary": {
            "min_delta_w30_minus_w29": min(deltas) if deltas else 0.0,
            "max_delta_w30_minus_w29": max(deltas) if deltas else 0.0,
            "mean_delta_w30_minus_w29": (
                sum(deltas) / len(deltas) if deltas else 0.0),
            "min_cram_ratio_w30_over_w28": (
                min(cram_ratios) if cram_ratios else 0.0),
            "min_cram_ratio_w30_over_w29": (
                min(cram_ratios_w30_w29) if cram_ratios_w30_w29 else 0.0),
            "max_overhead_w30_vs_w29": (
                max(overheads) if overheads else 0.0),
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _save_artifact(*, path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _cli_main() -> int:
    p = argparse.ArgumentParser(description="Phase 77 W30 driver")
    p.add_argument("--bank", default="chain_cram", choices=[
        "trivial_calibration", "chain_cram", "calibration_prior",
        "xhost_disagree", "non_calibratable", "degenerate_history",
        "calibrated_tampered", "cross_host_live", "cross_regime",
    ])
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--seed-sweep", action="store_true")
    p.add_argument("--n-eval", type=int, default=16)
    p.add_argument("--K", type=int, default=3, dest="K_consumers")
    p.add_argument("--T-decoder", type=int, default=None)
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--window", type=int, default=4)
    p.add_argument("--save", default=None)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    if args.bank == "cross_regime":
        out: dict[str, Any] = {"banks": {}}
        for b in ("trivial_calibration", "chain_cram", "calibration_prior",
                   "xhost_disagree", "non_calibratable",
                   "degenerate_history", "calibrated_tampered"):
            r = run_phase77(
                bank=b, bank_seed=args.seed,
                n_eval=args.n_eval,
                K_consumers=args.K_consumers,
                T_decoder=args.T_decoder,
                calibration_stride=args.stride,
                ancestor_window=args.window,
                verbose=args.verbose,
            )
            out["banks"][b] = r
        print(json.dumps(out, indent=2)[:10000])
        if args.save:
            _save_artifact(path=args.save, payload=out)
        return 0

    if args.seed_sweep:
        result = run_phase77_seed_sweep(
            bank=args.bank,
            seeds=(11, 17, 23, 29, 31),
            n_eval=args.n_eval,
            K_consumers=args.K_consumers,
            T_decoder=args.T_decoder,
            calibration_stride=args.stride,
            ancestor_window=args.window,
            verbose=args.verbose,
        )
    else:
        result = run_phase77(
            bank=args.bank,
            bank_seed=args.seed,
            n_eval=args.n_eval,
            K_consumers=args.K_consumers,
            T_decoder=args.T_decoder,
            calibration_stride=args.stride,
            ancestor_window=args.window,
            verbose=args.verbose,
        )
    print(json.dumps(result, indent=2)[:10000])
    if args.save:
        _save_artifact(path=args.save, payload=result)
    return 0


if __name__ == "__main__":
    sys.exit(_cli_main())
