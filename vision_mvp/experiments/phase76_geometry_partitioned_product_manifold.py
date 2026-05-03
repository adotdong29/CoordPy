"""Phase 76 — geometry-partitioned product-manifold dense control +
audited subspace-basis payload + factoradic routing index +
causal-validity gate + cross-host variance witness (SDK v3.30, W29 family).

The follow-up to SDK v3.29 (W28).  W28 wraps the W27 multi-chain
pivot decision in a controller-side trust-weighted probe quorum and
seals the result in a content-addressed
:class:`EnsemblePivotRatificationEnvelope` with 11 enumerated failure
modes.  Its strongest concrete result is the first cross-host live
evidence in 23 milestones — but it left
**W28-C-CROSS-HOST-VARIANCE** open: the variance-reduction *magnitude*
on a regime where W27 alone makes correctness mistakes was unmeasured
because every R-75 sub-bank had W27 correctness = 1.000 by
construction.

W29 (this phase) inserts a **structural geometry-partitioning** step
between the cell handoffs and the W28 ensemble decision.  Each cell is
classified into one of three pre-committed partition labels — LINEAR
(extends most-recent signature), HIERARCHICAL (fresh anchor), CYCLIC
(re-visited signature) — by the producer using the W27 input signature
CID and a bounded signature history.  Per-partition inner W28 stacks
get their own oracle config / probe table / pool, so cells in
different partitions route through structurally different
compartments.  The W29 envelope additionally carries an audited
**subspace basis** over the closed-vocabulary tag space (Grassmannian-
style, capsule-layer audited proxy), a **factoradic routing index**
(Lehmer code over the registered consumer order), a **causal-validity
signature** binding to the parent W28 envelope's CID and the cell's
declared predecessors, and an optional **cross-host variance witness**
emitted when ensemble probes from different hosts disagree.

Phase-76 sub-banks
==================

Seven pre-committed sub-banks (six synthetic + one cross-host live):

* **R-76-TRIVIAL-PARTITION** (H2 anchor; W29-Λ-trivial-partition).
  Single LINEAR partition, ``basis_dim = 0``, K=1 single consumer,
  no predecessors, single inner W28 — W29 must reduce to W28
  byte-for-byte across all cells (5/5 seeds).  Strict success bar:
  ``mean_total_w29_visible_tokens == mean_total_w28_visible_tokens``
  AND every cell yields
  ``w29_decoder_branch == "trivial_partition_passthrough"``.

* **R-76-CHAIN-SHARED** (H7 cram-factor anchor).
  Three partitions, ``basis_dim = 2``, K=3 consumers, on the
  ``chain_shared`` underlying bank.  The W29 layer charges 1 wire
  token per triggered cell but packs partition_id + factoradic
  permutation + 2-D subspace basis + causal-validity signature +
  cross-host variance witness CID into that single token.  Cram-
  factor headline: ``cram_factor_w29 ≥ 8 × cram_factor_w28``.

* **R-76-XHOST-DRIFT** (H6 load-bearing).
  The synthetic regime where W27 alone fails: a partial
  ServiceGraphOracle that knows GOLD_A but only PARTIALLY knows
  GOLD_B.  W27/W28 baselines fall to the GOLD_B blind spot on cells
  in the CYCLIC partition.  W29 with three partitions routes CYCLIC
  cells to a partition-2 inner W28 stack that uses a FULL oracle
  AND the cross-host LLM probes.  The H6 gate requires
  ``correctness_ratified_rate_w29 - correctness_ratified_rate_w28
  ≥ 0.10`` AND ``correctness_ratified_rate_w29 ≥ 0.95`` AND
  ``trust_precision_w29 ≥ 0.95`` across 5/5 seeds.

* **R-76-NON-ORTHOGONAL-BASIS** (H5; W29-Λ-non-orthogonal-basis).
  The producer emits a non-orthogonal subspace basis (two parallel
  unit vectors) on every cell; the verifier MUST reject every cell
  with ``subspace_basis_non_orthogonal``.

* **R-76-COORDINATED-DRIFT-XHOST** (H5;
  W29-Λ-coordinated-drift-cross-host).  Every cross-host probe
  receives the same prompt at temperature 0 ⇒ identical replies ⇒
  zero observed cross-host disagreement.  The witness must record
  ``cross_host_disagreements = 0`` AND
  ``correctness_ratified_rate_w29 == correctness_ratified_rate_w28``
  on this bank — cross-host telemetry cannot fix coordinated drift.

* **R-76-PARTITION-TAMPERED** (H3 tamper bank).
  Five named tampering passes per cell — flip ``ratified`` on the
  partition envelope, corrupt one basis coefficient with NaN, push
  the factoradic_route_index to ``K!``, replace the
  causal_validity_signature with garbage, unregister a partition_id.
  The verifier must reject ≥ 95% of attempted tampers across 5/5
  seeds.

* **R-76-CROSS-HOST-LIVE** (S1/S2 best-effort; live LLM).
  Live LLM probe table on two reachable Ollama hosts (localhost +
  192.168.12.191).  Records cross-host probe calls, round-trip
  bytes, witness CIDs, trust precision under real LLM disagreement.

The bench's apples-to-apples comparison is:

  * **W27 baseline**            — W27 orchestrator (no ensemble)
  * **W28 ensemble-verified**   — W27 + ensemble ratification
  * **W29 geometry-partitioned** — W28 + partition envelope (+
    optional per-partition inner W28 dispatch on R-76-XHOST-DRIFT)
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import random
import socket
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
    W27_BRANCH_PIVOTED, W27_BRANCH_ANCHORED_NEW,
    W27_BRANCH_FALLBACK_W26,
    # W28
    ProbeVote, EnsembleProbe, EnsembleProbeRegistration,
    DeterministicSignatureProbe, OracleConsultationProbe,
    LLMSignatureProbe,
    EnsemblePivotRatificationEnvelope,
    EnsembleRatificationRegistry,
    EnsembleVerifiedMultiChainOrchestrator,
    W28EnsembleResult,
    verify_ensemble_pivot_ratification,
    build_default_ensemble_registry,
    W28_RATIFICATION_SCHEMA_VERSION,
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
    W29PartitionResult,
    GeometryPartitionedOrchestrator,
    classify_partition_id_for_cell,
    verify_geometry_partition_ratification,
    build_trivial_partition_registry,
    build_three_partition_registry,
    W29_PARTITION_SCHEMA_VERSION,
    W29_PARTITION_LINEAR, W29_PARTITION_HIERARCHICAL,
    W29_PARTITION_CYCLIC,
    W29_DEFAULT_TRIGGER_BRANCHES, W29_PARTITION_LABEL,
    W29_BRANCH_PARTITION_RESOLVED,
    W29_BRANCH_TRIVIAL_PARTITION_PASSTHROUGH,
    W29_BRANCH_PARTITION_REJECTED,
    W29_BRANCH_CROSS_HOST_VARIANCE_WITNESSED,
    W29_BRANCH_NO_PARTITION_NEEDED,
    W29_BRANCH_FALLBACK_W28,
    # Oracles
    ServiceGraphOracle, ChangeHistoryOracle,
    OutsideQuery,
    _DecodedHandoff,
)

# Re-use the phase74 stack builders so W26/W27/W28 baselines are
# byte-for-byte identical to the W27/W28 milestones' benches.
from vision_mvp.experiments.phase74_multi_chain_pivot import (
    build_phase74_bank,
    build_team_shared_pool,
    build_team_shared_pool_xoracle,
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


# Closed-vocabulary services in the incident-triage schema (must match
# build_incident_triage_schema_capsule output sorted order).  Used as
# the ambient dimension of the W29 subspace basis.
INCIDENT_TRIAGE_AMBIENT_VOCABULARY: tuple[str, ...] = (
    "api", "db", "db_query", "logs_pipeline",
    "orders", "payments", "storage", "web",
)


# ---------------------------------------------------------------------------
# W29 visible-token / branch helpers
# ---------------------------------------------------------------------------


def _w29_visible(out: dict) -> int:
    if "geometry_partitioned" in out:
        return int(out["geometry_partitioned"].get(
            "n_w29_visible_tokens", 0))
    return _w28_visible(out)


def _w29_branch(out: dict) -> str:
    if "geometry_partitioned" in out:
        return str(out["geometry_partitioned"].get("decoder_branch", ""))
    return ""


def _w29_partition_id(out: dict) -> int:
    if "geometry_partitioned" in out:
        return int(out["geometry_partitioned"].get("partition_id", 0))
    return 0


def _w29_partition_label(out: dict) -> str:
    return W29_PARTITION_LABEL.get(_w29_partition_id(out), "unknown")


def _w29_ratified(out: dict) -> bool:
    if "geometry_partitioned" in out:
        return bool(out["geometry_partitioned"].get("ratified", False))
    return False


def _w29_overhead(out: dict) -> int:
    if "geometry_partitioned" in out:
        return int(out["geometry_partitioned"].get(
            "n_partition_overhead_tokens", 0))
    return 0


def _w29_n_structured_bits(out: dict) -> int:
    if "geometry_partitioned" in out:
        return int(out["geometry_partitioned"].get("n_structured_bits", 0))
    return 0


def _w29_cram_factor(out: dict) -> float:
    if "geometry_partitioned" in out:
        return float(out["geometry_partitioned"].get(
            "cram_factor_w29", 0.0))
    return 0.0


def _w28_n_envelope_bytes(out: dict) -> int:
    """Approximate W28 envelope-bytes-per-cell — used for the W28
    cram-factor denominator.  Falls back to 0 when no envelope was
    emitted.
    """
    if "ensemble_ratification_envelope" in out:
        return int(out["ensemble_ratification_envelope"].get(
            "n_envelope_bytes", 0))
    return 0


# ---------------------------------------------------------------------------
# Per-partition stack builders (R-76-XHOST-DRIFT)
# ---------------------------------------------------------------------------


def _build_per_partition_team_pools(
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
        full_oracles_for_cyclic: bool = True,
) -> dict[int, SharedMultiChainPool]:
    """Build one SharedMultiChainPool per geometry partition.

    Construction (mirrors the H6 honest framing):

    * **LINEAR partition (id 0)** — uses a partial ServiceGraphOracle
      that knows only ``partial_gold_pair``.  Cells routed here that
      need GOLD_A succeed; cells needing GOLD_B fall through.
    * **HIERARCHICAL partition (id 1)** — uses the same partial oracle
      (fresh-anchor cells on this regime are still mostly GOLD_A in
      the synthetic bank, so this partition's correctness matches
      LINEAR's).
    * **CYCLIC partition (id 2)** — uses a FULL oracle (the default
      :class:`ServiceGraphOracle` which knows BOTH gold pairs).  This
      is the partition where W29 routes the cells W27/W28 alone get
      wrong.  When ``full_oracles_for_cyclic`` is False, the cyclic
      partition shares the partial oracle (used for the
      coordinated-drift falsifier where partition gating cannot help).
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

        def _factory(
                *,
                signature_cid: str,
                agent_id: str,
                is_producer: bool,
        ) -> ChainPersistedFanoutDisambiguator:
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

    pools[W29_PARTITION_LINEAR] = _make_pool(oracle_set=partial_set)
    pools[W29_PARTITION_HIERARCHICAL] = _make_pool(oracle_set=partial_set)
    if full_oracles_for_cyclic:
        pools[W29_PARTITION_CYCLIC] = _make_pool(oracle_set=full_set)
    else:
        pools[W29_PARTITION_CYCLIC] = _make_pool(oracle_set=partial_set)
    return pools


# ---------------------------------------------------------------------------
# Falsifier helpers — R-76 named falsifiers
# ---------------------------------------------------------------------------


def _make_non_orthogonal_basis(
        *, ambient_vocabulary: tuple[str, ...]) -> SubspaceBasis:
    """Two basis vectors that are intentionally parallel (Gram off-
    diagonal = 1) — used by R-76-NON-ORTHOGONAL-BASIS to confirm the
    verifier rejects with ``subspace_basis_non_orthogonal``.

    The CID still computes; the rejection comes from the orthogonality
    check, not the hash check.
    """
    n = len(ambient_vocabulary)
    if n < 1:
        raise ValueError("ambient vocabulary must be non-empty")
    v = [0.0] * n
    v[0] = 1.0
    return SubspaceBasis(
        dim=2,
        ambient_dim=n,
        ambient_vocabulary=tuple(ambient_vocabulary),
        basis_vectors=(tuple(v), tuple(v)),  # parallel ⇒ Gram off-diag 1.0
    )


# ---------------------------------------------------------------------------
# Cross-host live registry builder (R-76-CROSS-HOST-LIVE)
# ---------------------------------------------------------------------------


def _build_cross_host_w28_registry(
        *,
        schema: SchemaCapsule,
        local_host_id: str = "localhost",
) -> tuple[EnsembleRatificationRegistry, dict[str, Any]]:
    """Build a W28 registry with a deterministic local probe + N
    cross-host LLM probes (one per reachable host with a selected
    model).  Returns the registry plus a metadata dict.
    """
    topo = discover_two_host_topology()
    meta: dict[str, Any] = {"topology": topo["topology"], "hosts_used": [
        {"host_id": h["host_id"], "selected_model": h.get(
            "selected_model", "")}
        for h in topo["hosts"]]}
    if topo["topology"] == "unreachable":
        return (
            build_default_ensemble_registry(
                schema=schema, quorum_threshold=1.0,
                local_host_id=local_host_id),
            {**meta, "n_probes": 1,
              "topology_fallback": "deterministic_single"},
        )
    from vision_mvp.coordpy.llm_backend import OllamaBackend
    backends_with_hosts: list[tuple[Any, str, str, float]] = []
    for h in topo["hosts"]:
        if not h.get("selected_model"):
            continue
        backend = OllamaBackend(
            model=h["selected_model"],
            base_url=h["url"],
            timeout=10.0,
        )
        backends_with_hosts.append((
            backend, h["host_id"],
            f"{h['host_id']}_{h['selected_model']}",
            0.5,
        ))
    registry = EnsembleRatificationRegistry(
        schema=schema,
        quorum_threshold=1.0,
        probes=(
            EnsembleProbeRegistration(
                probe=DeterministicSignatureProbe(probe_id="local_recompute"),
                trust_prior=1.0,
                role_label="local_recompute",
                host_id=local_host_id,
            ),
            *(
                EnsembleProbeRegistration(
                    probe=LLMSignatureProbe(
                        backend=b, probe_id=f"llm_{role_label}"),
                    trust_prior=trust,
                    role_label=role_label,
                    host_id=host_id,
                )
                for b, host_id, role_label, trust in backends_with_hosts
            ),
        ),
        local_host_id=local_host_id,
    )
    meta["n_probes"] = 1 + len(backends_with_hosts)
    return registry, meta


# ---------------------------------------------------------------------------
# Phase 76 runner
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _PerCellRecord:
    cell_idx: int
    expected: set[str]
    correct_w27: bool
    correct_w28: bool
    correct_w29: bool
    w27_visible: int
    w28_visible: int
    w29_visible: int
    w29_overhead: int
    w29_branch: str
    w29_partition_id: int
    w29_partition_label: str
    w29_ratified: bool
    w29_n_structured_bits: int
    w29_cram_factor: float
    w28_n_envelope_bytes: int
    cross_host_disagreement_count: int


def run_phase76(
        *,
        bank: str = "xhost_drift",
        T_decoder: int | None = None,
        K_consumers: int = 3,
        n_eval: int = 16,
        bank_replicates: int = 4,
        bank_seed: int = 11,
        chain_persist_window: int | None = None,
        max_active_chains: int = 8,
        signature_period: int = 4,
        verbose: bool = False,
) -> dict[str, Any]:
    """Run one Phase-76 sub-bank and return a results dict comparing
    W27 / W28 / W29 on the same regime.

    ``bank`` is one of:

      * ``"trivial_partition"``       — H2 byte-equivalence anchor.
      * ``"chain_shared"``            — H7 cram-factor headline.
      * ``"xhost_drift"``             — H6 load-bearing
                                          cross-host-variance discharge.
      * ``"non_orthogonal_basis"``    — H5 W29-Λ-non-orthogonal-basis.
      * ``"coordinated_drift_xhost"`` — H5 W29-Λ-coordinated-drift-cross-host.
      * ``"partition_tampered"``      — H3 tamper bank.
      * ``"cross_host_live"``         — S1/S2 best-effort live.
    """
    schema = build_incident_triage_schema_capsule()
    # Underlying phase74 bank shape.
    if bank in ("xhost_drift", "non_orthogonal_basis",
                "coordinated_drift_xhost", "partition_tampered",
                "cross_host_live"):
        underlying_bank = "divergent_recover"
    else:
        underlying_bank = "chain_shared"

    cells = build_phase74_bank(
        n_replicates=bank_replicates, seed=bank_seed,
        n_cells=n_eval, bank=underlying_bank,
        signature_period=signature_period)

    producer_id = "producer_agent"
    if bank == "trivial_partition":
        consumer_ids: tuple[str, ...] = ()
    else:
        consumer_ids = tuple(f"consumer_{k}" for k in range(K_consumers))

    if chain_persist_window is None:
        chain_persist_window = n_eval

    projection_id_for_consumer = {
        cid: f"proj_{cid}" for cid in consumer_ids
    }
    projected_tags_for_consumer = {
        cid: ("orders", "payments", "api", "db",
              "storage", "logs_pipeline", "web", "db_query")
        for cid in consumer_ids
    }

    # ---- W27 baseline: partial oracle on a single team-wide pool ----
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

    # ---- W28 baseline: same partial-oracle pool + single det probe ----
    # NOTE: a SEPARATE pool / registry from the one wrapped by W29 below
    # so that running the W28 baseline does not pollute W29's inner
    # state (each run advances cell_index in the inner stack).
    def _build_w28_baseline() -> EnsembleVerifiedMultiChainOrchestrator:
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
            reg, _m = _build_cross_host_w28_registry(
                schema=schema, local_host_id=LOCAL_HOST_ID)
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

    producer_w28 = _build_w28_baseline()
    # Separate W28 instance for W29's default inner (wrapped below).
    producer_w28_for_w29_default = _build_w28_baseline()
    registry_w28 = producer_w28.registry  # exposed for telemetry

    # ---- W29: registry + per-partition pools (when needed) ----
    if bank == "trivial_partition":
        registry_w29 = build_trivial_partition_registry(
            schema=schema, local_host_id=LOCAL_HOST_ID)
        # Trivial path: W29 wraps a SEPARATE W28 inner (own pool +
        # registry) so W29's per-cell state does not collide with the
        # W28 baseline.  Both paths run on byte-identical handoffs ⇒
        # the byte-for-W28 invariant holds.
        producer_w29 = GeometryPartitionedOrchestrator(
            inner=producer_w28_for_w29_default,
            registry=registry_w29,
            enabled=True,
            require_partition_verification=True,
            cycle_window=4,
        )
        per_partition_pools: dict[int, SharedMultiChainPool] = {}
    else:
        # Three-partition registry with basis_dim 2.  cycle_window
        # needs to be >= 2 * signature_period so the structural
        # classifier can detect the bench's alternation.
        cycle_window_for_bench = max(8, 2 * int(signature_period))
        registry_w29 = build_three_partition_registry(
            schema=schema,
            consumer_order=consumer_ids,
            ambient_vocabulary=INCIDENT_TRIAGE_AMBIENT_VOCABULARY,
            basis_dim=2,
            cycle_window=cycle_window_for_bench,
            local_host_id=LOCAL_HOST_ID,
        )
        if bank in ("xhost_drift", "cross_host_live", "partition_tampered",
                    "non_orthogonal_basis", "coordinated_drift_xhost"):
            # Per-partition inner W28 dispatch.  H6: cyclic gets full
            # oracle.  Coordinated-drift falsifier: cyclic shares
            # partial oracle (so partition routing cannot help).
            full_for_cyclic = bank not in ("coordinated_drift_xhost",)
            per_partition_pools = _build_per_partition_team_pools(
                T_decoder=T_decoder, schema=schema,
                producer_agent_id=producer_id,
                consumer_agent_ids=consumer_ids,
                chain_persist_window=chain_persist_window,
                max_active_chains=max_active_chains,
                signature_period=signature_period,
                projection_id_for_consumer=projection_id_for_consumer,
                projected_tags_for_consumer=projected_tags_for_consumer,
                partial_gold_pair=("orders", "payments"),
                full_oracles_for_cyclic=full_for_cyclic,
            )
            inner_per_partition: dict[
                int, EnsembleVerifiedMultiChainOrchestrator] = {}
            for pid, pool in per_partition_pools.items():
                # Fresh W28 registry per partition (one local
                # deterministic probe; H6 isolates the *oracle* signal,
                # not the probe topology).
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

            producer_w29 = GeometryPartitionedOrchestrator(
                inner=producer_w28_for_w29_default,
                registry=registry_w29,
                enabled=True,
                require_partition_verification=True,
                cycle_window=cycle_window_for_bench,
                inner_per_partition=inner_per_partition,
                pre_dispatch_by_partition=True,
            )
        else:
            # chain_shared / cram-factor: single inner W28 + W29
            # envelope (no per-partition dispatch — non-trivial
            # registry just to charge the wire token and pack the
            # structured payload).  Wraps a separate W28 instance so
            # the baseline runs cleanly.
            per_partition_pools = {}
            producer_w29 = GeometryPartitionedOrchestrator(
                inner=producer_w28_for_w29_default,
                registry=registry_w29,
                enabled=True,
                require_partition_verification=True,
                cycle_window=cycle_window_for_bench,
            )

    per_cell_records: list[_PerCellRecord] = []
    n_tamper_attempts = 0
    n_tampered_rejected = 0

    for cell_idx, cell_handoffs in enumerate(cells):
        # Run W27 baseline.
        out_w27 = producer_w27.decode_rounds(cell_handoffs)
        out_w28 = producer_w28.decode_rounds(cell_handoffs)
        out_w29 = producer_w29.decode_rounds(cell_handoffs)

        # R-76-NON-ORTHOGONAL-BASIS: replace the W29 envelope's basis
        # with a deliberately non-orthogonal one and re-verify.
        # Demonstrates the verifier's structural rejection rather
        # than the orchestrator's runtime path.
        if bank == "non_orthogonal_basis":
            ev = producer_w29.last_envelope
            if ev is not None:
                bad = _make_non_orthogonal_basis(
                    ambient_vocabulary=INCIDENT_TRIAGE_AMBIENT_VOCABULARY)
                tampered = dataclasses.replace(
                    ev, basis=bad, basis_dim=2,
                    partition_cid="")
                outcome = verify_geometry_partition_ratification(
                    tampered,
                    registered_schema=schema,
                    registered_w28_ratification_cid=ev.w28_ratification_cid,
                    registered_partition_table=registry_w29.partition_table,
                    registered_basis_dim=int(registry_w29.basis_dim),
                    registered_ambient_dim=int(registry_w29.ambient_dim),
                    registered_consumer_order=registry_w29.consumer_order,
                    registered_predecessor_cids=(
                        registry_w29.registered_predecessor_cids),
                )
                n_tamper_attempts += 1
                if (not outcome.ok
                        and outcome.reason ==
                            "subspace_basis_non_orthogonal"):
                    n_tampered_rejected += 1

        # R-76-PARTITION-TAMPERED: five named tampers per cell.
        if bank == "partition_tampered":
            ev = producer_w29.last_envelope
            if ev is not None:
                # Tamper 1: flip partition_id to one not registered.
                t1 = dataclasses.replace(
                    ev, partition_id=99, partition_cid="")
                o1 = verify_geometry_partition_ratification(
                    t1, registered_schema=schema,
                    registered_w28_ratification_cid=ev.w28_ratification_cid,
                    registered_partition_table=registry_w29.partition_table,
                    registered_basis_dim=int(registry_w29.basis_dim),
                    registered_ambient_dim=int(registry_w29.ambient_dim),
                    registered_consumer_order=registry_w29.consumer_order)
                n_tamper_attempts += 1
                if not o1.ok and o1.reason == "partition_id_unregistered":
                    n_tampered_rejected += 1
                # Tamper 2: factoradic_route_index out of range.
                K = ev.factoradic_route_n_factors
                if K > 0:
                    t2 = dataclasses.replace(
                        ev, factoradic_route_index=math.factorial(K),
                        partition_cid="")
                    o2 = verify_geometry_partition_ratification(
                        t2, registered_schema=schema,
                        registered_w28_ratification_cid=ev.w28_ratification_cid,
                        registered_partition_table=registry_w29.partition_table,
                        registered_basis_dim=int(registry_w29.basis_dim),
                        registered_ambient_dim=int(registry_w29.ambient_dim),
                        registered_consumer_order=registry_w29.consumer_order)
                    n_tamper_attempts += 1
                    if (not o2.ok
                            and o2.reason ==
                                "factoradic_index_out_of_range"):
                        n_tampered_rejected += 1
                # Tamper 3: replace causal_validity_signature.
                t3 = dataclasses.replace(
                    ev, causal_validity_signature="00" * 32,
                    partition_cid="")
                o3 = verify_geometry_partition_ratification(
                    t3, registered_schema=schema,
                    registered_w28_ratification_cid=ev.w28_ratification_cid,
                    registered_partition_table=registry_w29.partition_table,
                    registered_basis_dim=int(registry_w29.basis_dim),
                    registered_ambient_dim=int(registry_w29.ambient_dim),
                    registered_consumer_order=registry_w29.consumer_order)
                n_tamper_attempts += 1
                if (not o3.ok
                        and o3.reason ==
                            "causal_validity_signature_invalid"):
                    n_tampered_rejected += 1
                # Tamper 4: corrupt the partition_cid hash directly.
                t4 = dataclasses.replace(
                    ev, partition_cid="ff" * 32)
                # Force-skip recompute by reaching past frozen guard.
                object.__setattr__(t4, "partition_cid", "ff" * 32)
                o4 = verify_geometry_partition_ratification(
                    t4, registered_schema=schema,
                    registered_w28_ratification_cid=ev.w28_ratification_cid,
                    registered_partition_table=registry_w29.partition_table,
                    registered_basis_dim=int(registry_w29.basis_dim),
                    registered_ambient_dim=int(registry_w29.ambient_dim),
                    registered_consumer_order=registry_w29.consumer_order)
                n_tamper_attempts += 1
                if (not o4.ok
                        and o4.reason == "partition_cid_hash_mismatch"):
                    n_tampered_rejected += 1
                # Tamper 5: switch parent W28 CID.
                t5 = dataclasses.replace(
                    ev, w28_ratification_cid="11" * 32,
                    partition_cid="")
                o5 = verify_geometry_partition_ratification(
                    t5, registered_schema=schema,
                    registered_w28_ratification_cid=ev.w28_ratification_cid,
                    registered_partition_table=registry_w29.partition_table,
                    registered_basis_dim=int(registry_w29.basis_dim),
                    registered_ambient_dim=int(registry_w29.ambient_dim),
                    registered_consumer_order=registry_w29.consumer_order)
                n_tamper_attempts += 1
                if not o5.ok and o5.reason == "w28_parent_cid_mismatch":
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
            w27_visible=_w27_visible(out_w27),
            w28_visible=_w28_visible(out_w28),
            w29_visible=_w29_visible(out_w29),
            w29_overhead=_w29_overhead(out_w29),
            w29_branch=_w29_branch(out_w29),
            w29_partition_id=_w29_partition_id(out_w29),
            w29_partition_label=_w29_partition_label(out_w29),
            w29_ratified=_w29_ratified(out_w29),
            w29_n_structured_bits=_w29_n_structured_bits(out_w29),
            w29_cram_factor=_w29_cram_factor(out_w29),
            w28_n_envelope_bytes=_w28_n_envelope_bytes(out_w28),
            cross_host_disagreement_count=int(
                (out_w29.get("geometry_partitioned") or {})
                .get("cross_host_disagreement_count", 0)),
        )
        per_cell_records.append(record)

    n_cells = len(per_cell_records)

    def _mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    def _stddev(xs: list[float]) -> float:
        if len(xs) < 2:
            return 0.0
        m = _mean(xs)
        return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5

    correct_w27 = sum(1 for r in per_cell_records if r.correct_w27)
    correct_w28 = sum(1 for r in per_cell_records if r.correct_w28)
    correct_w29 = sum(1 for r in per_cell_records if r.correct_w29)
    correct_w27_rate = correct_w27 / n_cells if n_cells else 0.0
    correct_w28_rate = correct_w28 / n_cells if n_cells else 0.0
    correct_w29_rate = correct_w29 / n_cells if n_cells else 0.0

    # Trust precision: ratified ∧ correct / ratified.
    n_w29_ratified = sum(1 for r in per_cell_records if r.w29_ratified)
    n_w29_ratified_correct = sum(
        1 for r in per_cell_records
        if r.w29_ratified and r.correct_w29)
    trust_precision_w29 = (
        n_w29_ratified_correct / n_w29_ratified
        if n_w29_ratified > 0 else 0.0)
    trust_coverage_w29 = (
        n_w29_ratified / n_cells if n_cells else 0.0)

    overhead = [float(r.w29_overhead) for r in per_cell_records]
    structured_bits_per_cell = [
        float(r.w29_n_structured_bits) for r in per_cell_records]
    w28_envelope_bytes_per_cell = [
        float(r.w28_n_envelope_bytes) for r in per_cell_records]
    # Cram-factor proper:
    #   cram_factor = bits_of_structured_state / max(1, n_w29_overhead).
    # Per-cell mean across cells with non-zero structured payload.
    cram_factors = [
        r.w29_cram_factor for r in per_cell_records
        if r.w29_n_structured_bits > 0
    ]
    mean_cram_factor_w29 = _mean(cram_factors) if cram_factors else 0.0
    # W28 cram-factor: bits in W28 envelope over the W28 wire (1 token
    # iff W28 charged it; else 0 ⇒ infinite cram by the same convention,
    # which would make the ratio trivially infinite — instead we use
    # the W28 envelope's bytes-per-token = bytes / max(1, w28_overhead),
    # where w28_overhead is implicitly 1 token when W28 ratifies with
    # wire_required=True).  When W28 rides the wire-free path
    # (deterministic single probe) the W28 cram-factor is the bits
    # itself (1 conceptual token).
    w28_cram_factors: list[float] = []
    for r in per_cell_records:
        bits = 8 * float(r.w28_n_envelope_bytes)
        # W28's wire is at most 1 token per ratified cell.
        wire = 1.0
        w28_cram_factors.append(bits / wire if bits > 0 else 0.0)
    mean_cram_factor_w28 = _mean(
        [c for c in w28_cram_factors if c > 0]) if any(
        c > 0 for c in w28_cram_factors) else 0.0
    cram_ratio = (
        mean_cram_factor_w29 / mean_cram_factor_w28
        if mean_cram_factor_w28 > 0 else 0.0
    )

    # Branch counts.
    branch_counts: dict[str, int] = {}
    partition_routing: dict[int, int] = {}
    for r in per_cell_records:
        branch_counts[r.w29_branch] = branch_counts.get(r.w29_branch, 0) + 1
        partition_routing[r.w29_partition_id] = (
            partition_routing.get(r.w29_partition_id, 0) + 1)

    # Cross-host telemetry.
    n_cross_host_calls = int(registry_w28.n_cross_host_probe_calls)
    cross_host_bytes = int(registry_w28.cross_host_round_trip_bytes)
    n_cross_host_disagreements_total = sum(
        r.cross_host_disagreement_count for r in per_cell_records)

    # Tokens — total per cell (producer-only on this bench).
    mean_w27 = _mean([float(r.w27_visible) for r in per_cell_records])
    mean_w28 = _mean([float(r.w28_visible) for r in per_cell_records])
    mean_w29 = _mean([float(r.w29_visible) for r in per_cell_records])

    results: dict[str, Any] = {
        "bank": bank,
        "T_decoder": T_decoder,
        "K_consumers": K_consumers,
        "n_cells": n_cells,
        "bank_seed": bank_seed,
        "chain_persist_window": chain_persist_window,
        "max_active_chains": max_active_chains,
        "signature_period": signature_period,
        # Visible tokens.
        "mean_total_w27_visible_tokens": round(mean_w27, 4),
        "mean_total_w28_visible_tokens": round(mean_w28, 4),
        "mean_total_w29_visible_tokens": round(mean_w29, 4),
        "mean_overhead_w29_vs_w28_per_cell": round(_mean(overhead), 4),
        "max_overhead_w29_vs_w28_per_cell": int(
            max(overhead) if overhead else 0),
        # Equivalence at trivial path.
        "byte_equivalent_w29_w28": (mean_w29 == mean_w28),
        # Correctness.
        "correctness_ratified_rate_w27": round(correct_w27_rate, 4),
        "correctness_ratified_rate_w28": round(correct_w28_rate, 4),
        "correctness_ratified_rate_w29": round(correct_w29_rate, 4),
        "delta_w29_minus_w28": round(
            correct_w29_rate - correct_w28_rate, 4),
        "delta_w29_minus_w27": round(
            correct_w29_rate - correct_w27_rate, 4),
        # Trust.
        "n_w29_ratified": n_w29_ratified,
        "n_w29_ratified_correct": n_w29_ratified_correct,
        "trust_precision_w29": round(trust_precision_w29, 4),
        "trust_coverage_w29": round(trust_coverage_w29, 4),
        # Cram.
        "mean_n_structured_bits_w29": round(
            _mean(structured_bits_per_cell), 2),
        "mean_w28_envelope_bytes": round(
            _mean(w28_envelope_bytes_per_cell), 2),
        "mean_cram_factor_w29": round(mean_cram_factor_w29, 2),
        "mean_cram_factor_w28": round(mean_cram_factor_w28, 2),
        "cram_ratio_w29_over_w28": round(cram_ratio, 2),
        # Branches.
        "branch_counts_w29": branch_counts,
        "partition_routing_counts": {
            int(k): int(v) for k, v in partition_routing.items()},
        "n_partitions_registered": int(
            len(registry_w29.partition_table) if registry_w29 else 0),
        # Cross-host.
        "n_cross_host_probe_calls": n_cross_host_calls,
        "cross_host_round_trip_bytes": cross_host_bytes,
        "n_cross_host_disagreements_total": int(
            n_cross_host_disagreements_total),
        # Tampering.
        "n_tamper_attempts": int(n_tamper_attempts),
        "n_tampered_rejected": int(n_tampered_rejected),
        # Registry counters (for forensics).
        "n_partitions_envelope_registered": int(
            registry_w29.n_partitions_registered),
        "n_partitions_envelope_rejected": int(
            registry_w29.n_partitions_rejected),
        "n_cross_host_variance_witnessed": int(
            registry_w29.n_cross_host_variance_witnessed),
    }

    if verbose:
        print(f"\n=== Phase 76 — R-76-{bank.upper()} (seed={bank_seed}) ===")
        print(f"K={K_consumers}  n_cells={n_cells}  "
              f"max_active_chains={max_active_chains}")
        print(f"W27={mean_w27:.2f}  W28={mean_w28:.2f}  W29={mean_w29:.2f}  "
              f"overhead/cell={_mean(overhead):.2f}")
        print(f"correctness: W27={correct_w27_rate:.4f}  "
              f"W28={correct_w28_rate:.4f}  W29={correct_w29_rate:.4f}  "
              f"Δw29-w28={correct_w29_rate - correct_w28_rate:+.4f}")
        print(f"trust_precision_w29={trust_precision_w29:.4f}  "
              f"coverage={trust_coverage_w29:.4f}")
        print(f"cram_factor: W28={mean_cram_factor_w28:.1f}  "
              f"W29={mean_cram_factor_w29:.1f}  "
              f"ratio={cram_ratio:.1f}")
        print(f"partitions_routed={partition_routing}  "
              f"branches={branch_counts}")
        print(f"cross_host_calls={n_cross_host_calls}  "
              f"bytes={cross_host_bytes}  "
              f"disagree={n_cross_host_disagreements_total}")
        if bank in ("partition_tampered", "non_orthogonal_basis"):
            print(f"tampered_rejected={n_tampered_rejected}/"
                  f"{n_tamper_attempts}")

    return results


def run_phase76_seed_stability_sweep(
        *,
        seeds: tuple[int, ...] = (11, 17, 23, 29, 31),
        bank: str = "xhost_drift",
        T_decoder: int | None = None,
        K_consumers: int = 3,
        n_eval: int = 16,
        bank_replicates: int = 4,
        chain_persist_window: int | None = None,
        max_active_chains: int = 8,
        signature_period: int = 4,
        verbose: bool = False,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        r = run_phase76(
            bank=bank, T_decoder=T_decoder, K_consumers=K_consumers,
            n_eval=n_eval, bank_replicates=bank_replicates,
            bank_seed=seed,
            chain_persist_window=chain_persist_window,
            max_active_chains=max_active_chains,
            signature_period=signature_period,
            verbose=verbose)
        rows.append(r)
    overhead = [r["mean_overhead_w29_vs_w28_per_cell"] for r in rows]
    correctness_w29 = [r["correctness_ratified_rate_w29"] for r in rows]
    correctness_w28 = [r["correctness_ratified_rate_w28"] for r in rows]
    correctness_w27 = [r["correctness_ratified_rate_w27"] for r in rows]
    delta_w29_w28 = [r["delta_w29_minus_w28"] for r in rows]
    trust_prec = [r["trust_precision_w29"] for r in rows]
    cram = [r["mean_cram_factor_w29"] for r in rows]
    cram_ratio = [r["cram_ratio_w29_over_w28"] for r in rows]
    return {
        "seeds": list(seeds),
        "bank": bank,
        "T_decoder": T_decoder,
        "K_consumers": K_consumers,
        "max_active_chains": max_active_chains,
        "rows": rows,
        # Stability bands.
        "min_overhead_w29_vs_w28": min(overhead) if overhead else 0,
        "mean_overhead_w29_vs_w28": (
            sum(overhead) / len(overhead) if overhead else 0),
        "max_overhead_w29_vs_w28": max(overhead) if overhead else 0,
        "min_correctness_w29": min(correctness_w29)
            if correctness_w29 else 0,
        "min_correctness_w28": min(correctness_w28)
            if correctness_w28 else 0,
        "min_correctness_w27": min(correctness_w27)
            if correctness_w27 else 0,
        "min_delta_w29_minus_w28": min(delta_w29_w28)
            if delta_w29_w28 else 0,
        "all_correctness_w29_ge_w28": all(
            cw29 >= cw28
            for cw29, cw28 in zip(correctness_w29, correctness_w28)
        ),
        "all_delta_ge_0_10": all(d >= 0.10 for d in delta_w29_w28),
        "min_trust_precision_w29": min(trust_prec) if trust_prec else 0,
        "min_cram_factor_w29": min(cram) if cram else 0,
        "min_cram_ratio_w29_over_w28": min(cram_ratio) if cram_ratio else 0,
        "all_cram_ratio_ge_8": all(c >= 8.0 for c in cram_ratio),
    }


def run_cross_regime_p76(
        *,
        K_consumers: int = 3,
        n_eval: int = 16,
        bank_replicates: int = 4,
        bank_seed: int = 11,
        verbose: bool = False,
) -> dict[str, Any]:
    """Run the full R-76 family (synthetic + tamper).  cross_host_live
    is run separately by the CLI.
    """
    sub_banks = (
        "trivial_partition", "chain_shared",
        "xhost_drift", "non_orthogonal_basis",
        "coordinated_drift_xhost", "partition_tampered",
    )
    out: dict[str, Any] = {}
    for b in sub_banks:
        out[b] = run_phase76(
            bank=b, T_decoder=None,
            K_consumers=K_consumers, n_eval=n_eval,
            bank_replicates=bank_replicates,
            bank_seed=bank_seed, verbose=verbose)
    return out


def _cli() -> None:
    ap = argparse.ArgumentParser(
        description="Phase 76 — W29 geometry-partitioned product-manifold "
                    "dense control")
    ap.add_argument("--bank", default="xhost_drift",
                     choices=["trivial_partition", "chain_shared",
                                "xhost_drift", "non_orthogonal_basis",
                                "coordinated_drift_xhost",
                                "partition_tampered",
                                "cross_host_live",
                                "cross_regime", "topology_probe"])
    ap.add_argument("--K-consumers", type=int, default=3)
    ap.add_argument("--decoder-budget", type=int, default=-1)
    ap.add_argument("--chain-persist-window", type=int, default=-1)
    ap.add_argument("--max-active-chains", type=int, default=8)
    ap.add_argument("--signature-period", type=int, default=4)
    ap.add_argument("--n-eval", type=int, default=16)
    ap.add_argument("--bank-replicates", type=int, default=4)
    ap.add_argument("--bank-seed", type=int, default=11)
    ap.add_argument("--seed-sweep", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", default="-")
    args = ap.parse_args()

    T_decoder: int | None = (None if args.decoder_budget < 0
                              else args.decoder_budget)
    chain_persist_window: int | None = (
        None if args.chain_persist_window < 0
        else args.chain_persist_window)

    if args.bank == "cross_regime":
        result: Any = run_cross_regime_p76(
            K_consumers=args.K_consumers, n_eval=args.n_eval,
            bank_replicates=args.bank_replicates,
            bank_seed=args.bank_seed, verbose=args.verbose)
    elif args.bank == "topology_probe":
        result = discover_two_host_topology()
    elif args.seed_sweep:
        result = run_phase76_seed_stability_sweep(
            bank=args.bank, T_decoder=T_decoder,
            K_consumers=args.K_consumers, n_eval=args.n_eval,
            bank_replicates=args.bank_replicates,
            chain_persist_window=chain_persist_window,
            max_active_chains=args.max_active_chains,
            signature_period=args.signature_period,
            verbose=args.verbose)
    else:
        result = run_phase76(
            bank=args.bank, T_decoder=T_decoder,
            K_consumers=args.K_consumers, n_eval=args.n_eval,
            bank_replicates=args.bank_replicates,
            bank_seed=args.bank_seed,
            chain_persist_window=chain_persist_window,
            max_active_chains=args.max_active_chains,
            signature_period=args.signature_period,
            verbose=args.verbose)

    out_text = json.dumps(result, indent=2, default=str)
    if args.out == "-":
        print(out_text)
    else:
        with open(args.out, "w") as fh:
            fh.write(out_text)


if __name__ == "__main__":
    _cli()
