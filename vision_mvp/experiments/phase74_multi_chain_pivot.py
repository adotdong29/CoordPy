"""Phase 74 — multi-chain salience-keyed dense-control fanout (SDK v3.28, W27 family).

The follow-up to SDK v3.27 (W26) on the LatentMAS direction.  W26
amortised the producer's per-cell salience-token cost across cells
inside a single chain window via 1-token chain-advance references.
W26 has two structural limits:

* **W26-Λ-divergent** — when the cell's gold subset / canonical
  compact state changes, the inner W25 fires NO_TRIGGER and W26
  falls through to W25.  No chain savings on divergent cells.
* **W26-C-DIVERGENCE-RECOVERY** — open conjecture that a smarter
  chain-replay mechanism could recover savings on divergent cells.

W27 implements the smallest honest version of that mechanism: a
**bounded pool of parallel chains, keyed by salience signature**
(SHA-256 over canonical compact state).  At each cell, the
controller hashes the producer's current canonical state and looks
up the pool:

* If a chain with that signature exists, the producer **pivots** to
  that chain (1 token, audited via :class:`ChainPivotEnvelope`).
* If no chain matches AND the pool is not full, anchor a new chain
  (full W25 cost; same cost as W26 anchor).
* If no chain matches AND the pool is exhausted, fall through to W26
  (which falls through to W25 on divergent cells).

Phase-74 sub-banks
==================

Six pre-committed sub-banks:

* **R-74-DIVERGENT-RECOVER** (W27-1 efficiency anchor; the headline
  regime).  Same R-69-CACHE-FANOUT oracle ecology + 16 cells, but
  the gold subset alternates every 4 cells between
  ``{orders, payments}`` and ``{orders, login}`` — so 4 distinct
  signatures appear in regular succession.  Strict success bar:
  ``mean_total_w27_visible_tokens`` strictly below
  ``mean_total_w26_visible_tokens`` AND
  ``correctness_ratified_rate >= 1.000`` AND
  ``pool_size <= max_active_chains``.

* **R-74-CHAIN-SHARED** (W27-Λ-single-signature falsifier).  Same
  as R-73-CHAIN-SHARED — every cell produces the same canonical
  state.  W27 reduces to W26 byte-for-byte (one chain in pool, no
  pivots).

* **R-74-POOL-EXHAUSTED** (W27-Λ-pool-exhausted falsifier).
  ``max_active_chains = 2`` but the bench produces 4 distinct
  signatures.  Cells beyond the pool's capacity are rejected;
  W27 falls through to W26.  Correctness preserved.

* **R-74-PIVOT-TAMPERED** (W27-3 trust falsifier).  Producer's
  pivot envelope is corrupted (``parent_advance_cid`` overwritten);
  controller rejects via ``parent_advance_unknown``;
  correctness preserved.

* **R-74-SIGNATURE-DRIFT** (W27-3 trust falsifier).  Producer
  attaches a stale canonical state (mismatching the signature_cid);
  ``verify_salience_signature`` returns ``hash_mismatch``;
  W27 falls through to W26.

* **R-74-CROSS-MODEL** (W27-Λ-cross-model live-LLM probe).  The
  inner W25 stack is wrapped with a real-LLM adjudicator oracle on
  Mac-1; W27 exercises the multi-chain pool on top.  Honest
  scope: this measures whether the multi-chain machinery composes
  cleanly with a real oracle, not whether the LLM contributes new
  information.

Success criterion
-----------------
A milestone is W27-1 *discharged* iff, across all five pre-committed
``bank_seed`` values, the multi-chain path strictly reduces total
visible tokens over the W26 path on R-74-DIVERGENT-RECOVER AND
correctness is preserved byte-for-byte AND the pool size stays
bounded.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
import sys
from typing import Any

from vision_mvp.wevra.team_coord import (
    OracleRegistration, SchemaCapsule,
    build_incident_triage_schema_capsule,
    CrossCellDeltaDisambiguator, MultiCellSessionCompactor,
    FanoutEnvelope, SharedFanoutDisambiguator, SharedFanoutRegistry,
    verify_fanout, W25_FANOUT_SCHEMA_VERSION,
    W25_BRANCH_FANOUT_PRODUCER_EMITTED,
    W25_BRANCH_FANOUT_CONSUMER_RESOLVED,
    W25_BRANCH_FANOUT_CONSUMER_REJECTED, W25_BRANCH_DISABLED,
    W25_BRANCH_NO_TRIGGER,
    W24_BRANCH_COMPACT_RESOLVED, W24_BRANCH_BELOW_WINDOW,
    W23_BRANCH_GENESIS,
    # W26
    ChainAnchorEnvelope, ChainAdvanceEnvelope,
    ChainPersistedFanoutRegistry,
    ChainPersistedFanoutDisambiguator,
    ProjectionSlot,
    verify_chain_anchor, verify_chain_advance,
    verify_projection_subscription,
    W26_CHAIN_ANCHOR_SCHEMA_VERSION, W26_CHAIN_ADVANCE_SCHEMA_VERSION,
    W26_BRANCH_CHAIN_ANCHORED, W26_BRANCH_CHAIN_ADVANCED,
    W26_BRANCH_CHAIN_REJECTED, W26_BRANCH_CHAIN_RE_ANCHORED,
    W26_BRANCH_CHAIN_PROJECTION_RESOLVED,
    W26_BRANCH_CHAIN_PROJECTION_REJECTED,
    W26_BRANCH_NO_TRIGGER, W26_BRANCH_DISABLED,
    # W27
    SalienceSignatureEnvelope, ChainPivotEnvelope,
    MultiChainPersistedFanoutRegistry,
    MultiChainPersistedFanoutDisambiguator,
    MultiChainPersistedFanoutOrchestrator,
    SharedMultiChainPool,
    W27OrchestratorResult,
    compute_input_signature_cid,
    verify_salience_signature, verify_chain_pivot,
    W27_BRANCH_PIVOTED, W27_BRANCH_ANCHORED_NEW,
    W27_BRANCH_POOL_EXHAUSTED, W27_BRANCH_PIVOT_REJECTED,
    W27_BRANCH_FALLBACK_W26, W27_BRANCH_NO_TRIGGER,
    W27_BRANCH_DISABLED,
    # Oracles
    ServiceGraphOracle, ChangeHistoryOracle, OnCallNotesOracle,
    DisagreeingHonestOracle, CachingOracleAdapter, SharedReadCache,
    OutsideQuery,
    # Team handoff / decoder infrastructure
    _DecodedHandoff, BundleAwareTeamDecoder,
    collect_admitted_handoffs,
    TeamCoordinator, FifoAdmissionPolicy, DEFAULT_ROLE_BUDGETS,
)


# ---------------------------------------------------------------------------
# Bank builders
# ---------------------------------------------------------------------------

def build_phase74_bank(
        *,
        n_replicates: int = 4,
        seed: int = 11,
        n_cells: int = 16,
        bank: str = "divergent_recover",
        signature_period: int = 4,
) -> list[list[list[_DecodedHandoff]]]:
    """Build a Phase-74 multi-cell bank.

    Each element is one ``cell``: a list of per-round handoff lists
    (round 0 + round 1).

    Crucially, **cells within the same signature phase have
    byte-identical inputs** — magnitudes are derived from the seed
    AND the gold subset, but NOT from the cell index.  This ensures
    the input signature CID is stable within each phase so W27 can
    pivot between exactly the same chains across cells.

    For ``divergent_recover`` mode the gold subset alternates between
    GOLD_A and GOLD_B every ``signature_period`` cells.  Two distinct
    signatures total; W27's pool needs only 2 entries.
    """
    rng = random.Random(seed)

    # All gold subsets MUST be in the deterministic ServiceGraphOracle's
    # known dependency pairs (build_incident_triage_service_graph), or
    # the inner W22 layer abstains and the chain produces no signal.
    GOLD_A = ("orders", "payments")
    GOLD_B = ("api", "db")
    GOLD_C = ("storage", "logs_pipeline")
    GOLD_D = ("web", "db_query")
    # For "divergent_recover_xoracle" we also use the *same* gold tags
    # but flag content that some oracles do not know.  The per-signature
    # oracle scope is what separates W27 from W26 in this regime.
    DECOY = "cache"

    # Pre-compute magnitudes per gold-subset so cells in the same
    # phase have byte-identical inputs.
    def _mags_for(gold: tuple[str, ...]) -> dict[str, tuple[float, ...]]:
        # Use a deterministic hash of (seed, gold) to derive the
        # per-phase magnitudes.  This avoids advancing the rng state
        # so the same (seed, gold) pair always produces the same
        # magnitudes regardless of call order.
        h = hash((seed, gold))
        per_rng = random.Random(h)
        return {
            "load": tuple(per_rng.uniform(0.7, 1.0) for _ in gold),
            "error": per_rng.uniform(0.6, 0.9),
            "cpu_decoy": per_rng.uniform(0.1, 0.3),
        }

    PHASE_MAGS = {
        GOLD_A: _mags_for(GOLD_A),
        GOLD_B: _mags_for(GOLD_B),
        GOLD_C: _mags_for(GOLD_C),
        GOLD_D: _mags_for(GOLD_D),
    }

    def _make_handoff(source: str, kind: str, svc: str,
                       magnitude: float) -> _DecodedHandoff:
        payload = f"service={svc} magnitude={magnitude:.2f}"
        return _DecodedHandoff(
            source_role=source, claim_kind=kind, payload=payload)

    def _cell(cell_idx: int) -> list[list[_DecodedHandoff]]:
        if bank == "divergent_recover":
            phase = (cell_idx // signature_period) % 2
            GOLD = GOLD_A if phase == 0 else GOLD_B
        elif bank == "pool_exhausted":
            # FOUR distinct signatures, but max_active_chains=2 caps it.
            quarter = (cell_idx // (n_cells // 4)) % 4
            GOLD_PALETTE = (GOLD_A, GOLD_B, GOLD_C, GOLD_D)
            GOLD = GOLD_PALETTE[quarter]
        elif bank == "xoracle_recover":
            # Same as divergent_recover but the runner registers
            # per-signature oracles in the W27 pool that scope each
            # signature to its own gold pair — the regime where W27's
            # per-signature isolation gives correctness gains over a
            # W26 single-stack baseline (which uses ONE oracle config
            # and resolves only ONE gold pair).
            phase = (cell_idx // signature_period) % 2
            GOLD = GOLD_A if phase == 0 else GOLD_B
        else:
            # chain_shared / others — single signature
            GOLD = GOLD_A

        mags = PHASE_MAGS[GOLD]
        r0: list[_DecodedHandoff] = []
        for i, (svc, mag) in enumerate(zip(GOLD, mags["load"])):
            r0.append(_make_handoff(
                f"role_{i}", "LOAD_SPIKE", svc, mag))
        r0.append(_make_handoff("role_2", "ERROR_RATE", GOLD[0],
                                  mags["error"]))
        r0.append(_make_handoff("role_3", "CPU_SPIKE", DECOY,
                                  mags["cpu_decoy"]))
        r1: list[_DecodedHandoff] = []
        for svc in GOLD:
            r1.append(_make_handoff(
                "disambiguator", "CORROBORATION", svc, 1.0))
        return [r0, r1]

    return [_cell(i) for i in range(n_cells)]


# ---------------------------------------------------------------------------
# Stack builders
# ---------------------------------------------------------------------------

def _build_w25_stack(
        *,
        T_decoder: int | None,
        schema: SchemaCapsule,
        raw_oracles: tuple[tuple[Any, str], ...],
        agent_id: str,
        is_producer: bool,
        producer_agent_id: str,
        consumer_agent_ids: tuple[str, ...],
        registry: SharedFanoutRegistry,
) -> SharedFanoutDisambiguator:
    from vision_mvp.wevra.team_coord import (
        AttentionAwareBundleDecoder,
        RelationalCompatibilityDisambiguator,
        BundleContradictionDisambiguator,
        LatentDigestDisambiguator, CrossCellDeltaDisambiguator,
        QuorumKeyedSharedReadCache, QuorumKeyedCachingOracleAdapter,
        TrustWeightedMultiOracleDisambiguator,
    )
    cache = QuorumKeyedSharedReadCache()
    regs = tuple(
        OracleRegistration(
            oracle=QuorumKeyedCachingOracleAdapter(
                inner=oracle, cache=cache, oracle_id=role_label),
            trust_prior=1.0, role_label=role_label)
        for oracle, role_label in raw_oracles
    )
    w15 = AttentionAwareBundleDecoder(T_decoder=T_decoder)
    w18 = RelationalCompatibilityDisambiguator(inner=w15)
    w19 = BundleContradictionDisambiguator(inner=w18)
    w21 = TrustWeightedMultiOracleDisambiguator(
        inner=w19, oracle_registrations=regs, quorum_min=1)
    w22 = LatentDigestDisambiguator(inner=w21, schema=schema)
    w23 = CrossCellDeltaDisambiguator(inner=w22, schema=schema)
    w24 = MultiCellSessionCompactor(
        inner=w23, schema=schema, compact_window=4)
    w25 = SharedFanoutDisambiguator(
        inner=w24, fanout_registry=registry, agent_id=agent_id,
        is_producer=is_producer, producer_agent_id=producer_agent_id,
        consumer_agent_ids=consumer_agent_ids,
        schema=schema, enabled=True,
        require_fanout_verification=True)
    return w25


def _build_partial_service_graph_oracle(
        *,
        gold_pair: tuple[str, str],
        oracle_id: str,
) -> "ServiceGraphOracle":
    """Build a ServiceGraphOracle whose graph contains ONLY the given
    gold pair (and the rest as singletons).

    Used in the ``xoracle_recover`` bank to give each signature its
    own narrowly-scoped oracle.  A W26 single-stack baseline using
    only one of these oracles can only resolve its own gold subset
    (correctness 0.5 on a 2-phase bench); W27 with per-signature
    oracle registration resolves both phases (correctness 1.0).
    """
    g: dict[str, "frozenset[str]"] = {
        gold_pair[0]: frozenset({gold_pair[1]}),
        gold_pair[1]: frozenset({gold_pair[0]}),
    }
    return ServiceGraphOracle(oracle_id=oracle_id, graph=g)


def _build_w26_stack(
        *,
        T_decoder: int | None,
        schema: SchemaCapsule,
        raw_oracles: tuple[tuple[Any, str], ...],
        agent_id: str,
        is_producer: bool,
        producer_agent_id: str,
        consumer_agent_ids: tuple[str, ...],
        fanout_registry: SharedFanoutRegistry,
        chain_registry: ChainPersistedFanoutRegistry,
        chain_persist_window: int,
        projection_id_for_consumer: dict[str, str] | None = None,
        projected_tags_for_consumer: dict[
            str, tuple[str, ...]] | None = None,
) -> ChainPersistedFanoutDisambiguator:
    inner_w25 = _build_w25_stack(
        T_decoder=T_decoder, schema=schema, raw_oracles=raw_oracles,
        agent_id=agent_id, is_producer=is_producer,
        producer_agent_id=producer_agent_id,
        consumer_agent_ids=consumer_agent_ids,
        registry=fanout_registry)
    return ChainPersistedFanoutDisambiguator(
        inner=inner_w25,
        chain_registry=chain_registry,
        schema=schema,
        chain_persist_window=chain_persist_window,
        enabled=True,
        require_chain_verification=True,
        projection_id_for_consumer=dict(
            projection_id_for_consumer or {}),
        projected_tags_for_consumer=dict(
            projected_tags_for_consumer or {}),
    )


def build_team_shared_pool_xoracle(
        *,
        T_decoder: int | None,
        schema: SchemaCapsule,
        signature_to_gold_pair: dict[str, tuple[str, str]],
        producer_agent_id: str,
        consumer_agent_ids: tuple[str, ...],
        chain_persist_window: int,
        max_active_chains: int,
        projection_id_for_consumer: dict[str, str],
        projected_tags_for_consumer: dict[str, tuple[str, ...]],
        fallback_gold_pair: tuple[str, str] = ("orders", "payments"),
) -> SharedMultiChainPool:
    """Per-signature-scoped oracle pool.

    Each signature_cid in ``signature_to_gold_pair`` is mapped to a
    narrowly-scoped :class:`ServiceGraphOracle` that knows only that
    gold pair.  Other signatures (or fallback) get the full default
    oracle.  This is the regime where W27's per-signature isolation
    gives correctness gains over a W26 single-stack baseline.
    """
    per_sig_registries: dict[str,
        tuple[SharedFanoutRegistry, ChainPersistedFanoutRegistry]] = {}

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
        # Pick the oracle scoped to this signature.
        gold_pair = signature_to_gold_pair.get(
            signature_cid, fallback_gold_pair)
        sig_oracles = (
            (_build_partial_service_graph_oracle(
                gold_pair=gold_pair,
                oracle_id=f"service_graph_{signature_cid[:8]}"),
              "service_graph"),
            (ChangeHistoryOracle(oracle_id="change_history"),
              "change_history"),
        )
        return _build_w26_stack(
            T_decoder=T_decoder, schema=schema,
            raw_oracles=sig_oracles,
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


def build_team_shared_pool(
        *,
        T_decoder: int | None,
        schema: SchemaCapsule,
        raw_oracles: tuple[tuple[Any, str], ...],
        producer_agent_id: str,
        consumer_agent_ids: tuple[str, ...],
        chain_persist_window: int,
        max_active_chains: int,
        projection_id_for_consumer: dict[str, str],
        projected_tags_for_consumer: dict[str, tuple[str, ...]],
) -> SharedMultiChainPool:
    """Build a team-wide multi-chain pool that shares per-signature
    fanout / chain registries across producer + K consumers.

    The pool's stack factory closes over a per-signature registry
    cache so that all agents on the team use byte-identical
    ``SharedFanoutRegistry`` and ``ChainPersistedFanoutRegistry``
    instances for any given signature.  This is the analogue of
    ``run_phase73`` which threads ONE registry pair across the team.
    """
    per_sig_registries: dict[str,
        tuple[SharedFanoutRegistry, ChainPersistedFanoutRegistry]] = {}

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
            raw_oracles=raw_oracles,
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


def _build_w27_orchestrator(
        *,
        schema: SchemaCapsule,
        agent_id: str,
        is_producer: bool,
        producer_agent_id: str,
        consumer_agent_ids: tuple[str, ...],
        pool: SharedMultiChainPool,
) -> MultiChainPersistedFanoutOrchestrator:
    """Build a W27 orchestrator bound to a team-shared pool."""
    return MultiChainPersistedFanoutOrchestrator(
        schema=schema,
        producer_agent_id=str(producer_agent_id),
        consumer_agent_ids=tuple(consumer_agent_ids),
        agent_id=str(agent_id),
        is_producer=bool(is_producer),
        pool=pool,
        enabled=True,
    )


# ---------------------------------------------------------------------------
# run_phase74
# ---------------------------------------------------------------------------

def _w25_visible(out: dict) -> int:
    if "shared_fanout_hybrid" in out:
        return int(out["shared_fanout_hybrid"].get(
            "n_w25_visible_tokens", 0))
    return 0


def _w26_visible(out: dict) -> int:
    if "chain_persisted_hybrid" in out:
        return int(out["chain_persisted_hybrid"].get(
            "n_w26_visible_tokens", 0))
    return _w25_visible(out)


def _w27_visible(out: dict) -> int:
    if "multi_chain_orchestrator" in out:
        return int(out["multi_chain_orchestrator"].get(
            "n_w27_visible_tokens", 0))
    if "multi_chain_persisted_hybrid" in out:
        return int(out["multi_chain_persisted_hybrid"].get(
            "n_w27_visible_tokens", 0))
    return _w26_visible(out)


def _w24_visible(out: dict) -> int:
    if "shared_fanout_hybrid" in out:
        return int(out["shared_fanout_hybrid"].get(
            "n_w24_visible_tokens", 0))
    if "session_compact_hybrid" in out:
        return int(out["session_compact_hybrid"].get(
            "n_w24_visible_tokens_to_decider", 0))
    return 0


def _w27_branch(out: dict) -> str:
    if "multi_chain_orchestrator" in out:
        return str(out["multi_chain_orchestrator"].get(
            "decoder_branch", ""))
    if "multi_chain_persisted_hybrid" in out:
        return str(out["multi_chain_persisted_hybrid"].get(
            "decoder_branch", ""))
    return ""


def _w26_branch(out: dict) -> str:
    if "chain_persisted_hybrid" in out:
        return str(out["chain_persisted_hybrid"].get(
            "decoder_branch", ""))
    return ""


def _expected_gold_for_cell(
        *, bank: str, cell_idx: int, n_eval: int,
        signature_period: int,
) -> set[str]:
    GOLD_A = {"orders", "payments"}
    GOLD_B = {"api", "db"}
    GOLD_C = {"storage", "logs_pipeline"}
    GOLD_D = {"web", "db_query"}
    if bank in ("divergent_recover", "xoracle_recover"):
        phase = (cell_idx // signature_period) % 2
        return GOLD_A if phase == 0 else GOLD_B
    if bank == "pool_exhausted":
        quarter = (cell_idx // (n_eval // 4)) % 4
        EXPECTED_PALETTE = (GOLD_A, GOLD_B, GOLD_C, GOLD_D)
        return EXPECTED_PALETTE[quarter]
    return GOLD_A


def run_phase74(
        *,
        bank: str = "divergent_recover",
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
    """Run one Phase-74 sub-bank and return the results dict.

    Compares THREE configurations on the same bench:

    * **W26 baseline** — one :class:`ChainPersistedFanoutDisambiguator`
      processing every cell.  On divergent regimes the inner W22..W25
      consensus drops to ``no_trigger`` once the gold subset flips,
      and the W26 chain falls through to W25 (or W24).
    * **W27 orchestrator** —
      :class:`MultiChainPersistedFanoutOrchestrator` with a pool of
      independent W26 stacks, one per signature.  Each cell routes
      to its matching slot via
      :func:`compute_input_signature_cid`.
    * **W26 + W27 disambiguator wrapper** — additionally exercises the
      :class:`MultiChainPersistedFanoutDisambiguator` (audited pivot
      envelope on top of one W26).  Reported as ``w27_disambig_*``
      fields for trust-boundary auditing on the divergent regime.

    Parameters
    ----------
    bank:
        One of ``"divergent_recover"``, ``"chain_shared"``,
        ``"pool_exhausted"``, ``"pivot_tampered"``,
        ``"signature_drift"``.
    """
    schema = build_incident_triage_schema_capsule()

    if bank == "xoracle_recover":
        # The W26 baseline uses a partial oracle that knows only GOLD_A
        # (orders/payments).  This is the W26-Λ-divergent regime where
        # the inner W22 abstains on phase-B cells.  W27 with per-
        # signature pool registers different oracles per signature.
        raw_oracles: tuple[tuple[Any, str], ...] = (
            (_build_partial_service_graph_oracle(
                gold_pair=("orders", "payments"),
                oracle_id="service_graph_partial_A"),
              "service_graph"),
            (ChangeHistoryOracle(oracle_id="change_history"),
              "change_history"),
        )
    else:
        raw_oracles = (
            (ServiceGraphOracle(oracle_id="service_graph"),
              "service_graph"),
            (ChangeHistoryOracle(oracle_id="change_history"),
              "change_history"),
        )

    cells = build_phase74_bank(
        n_replicates=bank_replicates, seed=bank_seed,
        n_cells=n_eval, bank=bank, signature_period=signature_period)

    producer_id = "producer_agent"
    consumer_ids = tuple(f"consumer_{k}" for k in range(K_consumers))

    if chain_persist_window is None:
        chain_persist_window = n_eval  # one chain per pool entry

    effective_max_chains = (
        2 if bank == "pool_exhausted" else max_active_chains)

    projection_id_for_consumer: dict[str, str] = {
        cid: f"proj_{cid}" for cid in consumer_ids
    }
    projected_tags_for_consumer: dict[str, tuple[str, ...]] = {
        cid: ("orders", "payments", "api", "db",
              "storage", "logs_pipeline", "web", "db_query")
        for cid in consumer_ids
    }

    # ---------------- W26 baseline (one inner stack) ----------------
    fanout_registry_w26 = SharedFanoutRegistry(schema=schema)
    chain_registry_w26 = ChainPersistedFanoutRegistry(schema=schema)
    producer_w26 = _build_w26_stack(
        T_decoder=T_decoder, schema=schema, raw_oracles=raw_oracles,
        agent_id=producer_id, is_producer=True,
        producer_agent_id=producer_id,
        consumer_agent_ids=consumer_ids,
        fanout_registry=fanout_registry_w26,
        chain_registry=chain_registry_w26,
        chain_persist_window=chain_persist_window,
        projection_id_for_consumer=projection_id_for_consumer,
        projected_tags_for_consumer=projected_tags_for_consumer,
    )
    consumer_w26_list: list[ChainPersistedFanoutDisambiguator] = []
    for cid in consumer_ids:
        consumer_w26_list.append(_build_w26_stack(
            T_decoder=T_decoder, schema=schema,
            raw_oracles=raw_oracles,
            agent_id=cid, is_producer=False,
            producer_agent_id=producer_id,
            consumer_agent_ids=consumer_ids,
            fanout_registry=fanout_registry_w26,
            chain_registry=chain_registry_w26,
            chain_persist_window=chain_persist_window,
            projection_id_for_consumer=projection_id_for_consumer,
            projected_tags_for_consumer=projected_tags_for_consumer,
        ))

    # ---------------- W27 orchestrator (pool of W26 stacks) ----------
    if bank == "xoracle_recover":
        # Pre-compute the input signature for each gold phase so we
        # can register a per-signature oracle scoped to that gold pair.
        sig_to_gold: dict[str, tuple[str, str]] = {}
        # Need to compute signature deterministically; same path as
        # the orchestrator uses.
        for cell_idx in range(min(n_eval, 2 * signature_period)):
            phase = (cell_idx // signature_period) % 2
            gold = ("orders", "payments") if phase == 0 else ("api", "db")
            handoffs = cells[cell_idx]
            sig_cid = compute_input_signature_cid(
                handoffs,
                producer_agent_id=producer_id,
                consumer_agent_ids=consumer_ids,
                schema_cid=str(schema.cid),
            )
            if sig_cid not in sig_to_gold:
                sig_to_gold[sig_cid] = gold
        shared_pool = build_team_shared_pool_xoracle(
            T_decoder=T_decoder, schema=schema,
            signature_to_gold_pair=sig_to_gold,
            producer_agent_id=producer_id,
            consumer_agent_ids=consumer_ids,
            chain_persist_window=chain_persist_window,
            max_active_chains=effective_max_chains,
            projection_id_for_consumer=projection_id_for_consumer,
            projected_tags_for_consumer=projected_tags_for_consumer,
        )
    else:
        shared_pool = build_team_shared_pool(
            T_decoder=T_decoder, schema=schema,
            raw_oracles=raw_oracles,
            producer_agent_id=producer_id,
            consumer_agent_ids=consumer_ids,
            chain_persist_window=chain_persist_window,
            max_active_chains=effective_max_chains,
            projection_id_for_consumer=projection_id_for_consumer,
            projected_tags_for_consumer=projected_tags_for_consumer,
        )
    producer_w27 = _build_w27_orchestrator(
        schema=schema,
        agent_id=producer_id, is_producer=True,
        producer_agent_id=producer_id,
        consumer_agent_ids=consumer_ids,
        pool=shared_pool,
    )
    consumer_w27_list: list[MultiChainPersistedFanoutOrchestrator] = []
    for cid in consumer_ids:
        consumer_w27_list.append(_build_w27_orchestrator(
            schema=schema,
            agent_id=cid, is_producer=False,
            producer_agent_id=producer_id,
            consumer_agent_ids=consumer_ids,
            pool=shared_pool,
        ))

    # ---------------- W27 audited pivot wrapper (one W26 stack) -----
    fanout_registry_w27d = SharedFanoutRegistry(schema=schema)
    chain_registry_w27d = ChainPersistedFanoutRegistry(schema=schema)
    multi_chain_registry_w27d = MultiChainPersistedFanoutRegistry(
        schema=schema,
        max_active_chains=effective_max_chains,
        chain_registry=chain_registry_w27d,
    )
    inner_for_w27d = _build_w26_stack(
        T_decoder=T_decoder, schema=schema, raw_oracles=raw_oracles,
        agent_id=producer_id, is_producer=True,
        producer_agent_id=producer_id,
        consumer_agent_ids=consumer_ids,
        fanout_registry=fanout_registry_w27d,
        chain_registry=chain_registry_w27d,
        chain_persist_window=chain_persist_window,
        projection_id_for_consumer=projection_id_for_consumer,
        projected_tags_for_consumer=projected_tags_for_consumer,
    )
    producer_w27d = MultiChainPersistedFanoutDisambiguator(
        inner=inner_for_w27d,
        multi_chain_registry=multi_chain_registry_w27d,
        schema=schema,
        max_active_chains=effective_max_chains,
        enabled=True,
        require_pivot_verification=True,
    )

    per_cell_producer_w26: list[dict[str, Any]] = []
    per_cell_producer_w27: list[dict[str, Any]] = []
    per_cell_producer_w27d: list[dict[str, Any]] = []
    per_cell_consumers_w26: list[list[dict[str, Any]]] = []
    per_cell_consumers_w27: list[list[dict[str, Any]]] = []
    correctness_w26: list[bool] = []
    correctness_w27: list[bool] = []

    for cell_idx, cell_handoffs in enumerate(cells):
        # W26 baseline producer
        p_out_w26 = producer_w26.decode_rounds(cell_handoffs)
        per_cell_producer_w26.append(p_out_w26)
        # W27 orchestrator producer
        p_out_w27 = producer_w27.decode_rounds(cell_handoffs)
        per_cell_producer_w27.append(p_out_w27)
        # W27 disambig wrapper producer
        p_out_w27d = producer_w27d.decode_rounds(cell_handoffs)

        # Tampering banks (only on the audited W27 disambig wrapper)
        if (bank == "pivot_tampered"
                and producer_w27d.last_pivot is not None):
            tampered = dataclasses.replace(
                producer_w27d.last_pivot,
                parent_advance_cid="TAMPERED" + "0" * 56)
            multi_chain_registry_w27d._pivots[tampered.pivot_cid] = tampered
            multi_chain_registry_w27d._signature_to_last_advance[
                tampered.signature_cid] = tampered.parent_advance_cid

        if (bank == "signature_drift"
                and producer_w27d.last_signature is not None):
            sig = producer_w27d.last_signature
            old_chain = multi_chain_registry_w27d._signature_to_chain.pop(
                sig.signature_cid, None)
            old_last = (
                multi_chain_registry_w27d._signature_to_last_advance.pop(
                    sig.signature_cid, None))
            stale_sig = "STALE" + "0" * 59
            if old_chain is not None:
                multi_chain_registry_w27d._signature_to_chain[
                    stale_sig] = old_chain
            if old_last is not None:
                multi_chain_registry_w27d._signature_to_last_advance[
                    stale_sig] = old_last

        per_cell_producer_w27d.append(p_out_w27d)

        # Correctness — read from each method's own answer
        expected = _expected_gold_for_cell(
            bank=bank, cell_idx=cell_idx, n_eval=n_eval,
            signature_period=signature_period)

        def _is_correct(out: dict) -> bool:
            ans = out.get("answer") or out
            svcs = ans.get("services") if isinstance(ans, dict) else None
            if svcs is None:
                svcs = out.get("services")
            return set(svcs or []) == expected

        correctness_w26.append(_is_correct(p_out_w26))
        correctness_w27.append(_is_correct(p_out_w27))

        # Consumers (W26 baseline + W27 orchestrator)
        c_row_w26: list[dict[str, Any]] = []
        for c_dis in consumer_w26_list:
            c_out = c_dis.decode_rounds(cell_handoffs)
            c_row_w26.append(c_out)
        per_cell_consumers_w26.append(c_row_w26)

        c_row_w27: list[dict[str, Any]] = []
        for c_dis in consumer_w27_list:
            c_out = c_dis.decode_rounds(cell_handoffs)
            c_row_w27.append(c_out)
        per_cell_consumers_w27.append(c_row_w27)

    n_cells_run = len(per_cell_producer_w26)

    # --- Producer tokens for W26 baseline (ChainPersisted only) ---
    w25_tokens_p_b = [_w25_visible(o) for o in per_cell_producer_w26]
    w26_tokens_p_b = [_w26_visible(o) for o in per_cell_producer_w26]
    w24_tokens_p_b = [_w24_visible(o) for o in per_cell_producer_w26]

    # --- Producer tokens for W27 orchestrator ---
    w25_tokens_p_w27 = [_w25_visible(o) for o in per_cell_producer_w27]
    w26_tokens_p_w27 = [_w26_visible(o) for o in per_cell_producer_w27]
    w27_tokens_p = [_w27_visible(o) for o in per_cell_producer_w27]
    w24_tokens_p_w27 = [_w24_visible(o) for o in per_cell_producer_w27]

    # --- Consumer tokens for W26 baseline ---
    w26_tokens_c_b_per_cell = [
        sum(_w26_visible(c) for c in row) for row in per_cell_consumers_w26
    ]
    w24_tokens_c_b_per_cell = [
        sum(_w24_visible(c) for c in row) for row in per_cell_consumers_w26
    ]
    w25_tokens_c_b_per_cell = [
        sum(_w25_visible(c) for c in row) for row in per_cell_consumers_w26
    ]

    # --- Consumer tokens for W27 orchestrator ---
    w27_tokens_c_per_cell = [
        sum(_w27_visible(c) for c in row) for row in per_cell_consumers_w27
    ]
    w26_tokens_c_w27_per_cell = [
        sum(_w26_visible(c) for c in row) for row in per_cell_consumers_w27
    ]
    w25_tokens_c_w27_per_cell = [
        sum(_w25_visible(c) for c in row) for row in per_cell_consumers_w27
    ]

    # Totals per cell — apples-to-apples per method.
    total_w26_baseline = [
        w26_tokens_p_b[i] + w26_tokens_c_b_per_cell[i]
        for i in range(n_cells_run)]
    total_w27 = [
        w27_tokens_p[i] + w27_tokens_c_per_cell[i]
        for i in range(n_cells_run)]
    total_w25_baseline = [
        w25_tokens_p_b[i] + w25_tokens_c_b_per_cell[i]
        for i in range(n_cells_run)]
    total_w24_baseline = [
        w24_tokens_p_b[i] + w24_tokens_c_b_per_cell[i]
        for i in range(n_cells_run)]

    def _mean(xs: list[int]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    mean_w26_baseline = _mean(total_w26_baseline)
    mean_w27 = _mean(total_w27)
    mean_w25_baseline = _mean(total_w25_baseline)
    mean_w24_baseline = _mean(total_w24_baseline)

    save_w27_vs_w26 = mean_w26_baseline - mean_w27
    save_w27_vs_w25 = mean_w25_baseline - mean_w27
    save_w27_vs_w24 = mean_w24_baseline - mean_w27
    save_w26_vs_w25 = mean_w25_baseline - mean_w26_baseline

    pct = lambda num, den: (100.0 * num / den) if den > 0 else 0.0

    branch_counts_producer_w27: dict[str, int] = {}
    for o in per_cell_producer_w27:
        b = _w27_branch(o)
        branch_counts_producer_w27[b] = (
            branch_counts_producer_w27.get(b, 0) + 1)

    branch_counts_producer_w26: dict[str, int] = {}
    for o in per_cell_producer_w26:
        b = _w26_branch(o)
        branch_counts_producer_w26[b] = (
            branch_counts_producer_w26.get(b, 0) + 1)

    n_consumer_pivoted = sum(
        1 for row in per_cell_consumers_w27
        for c in row
        if _w27_branch(c) == W27_BRANCH_PIVOTED
    )
    total_consumer_cells = n_cells_run * K_consumers
    consumer_pivoted_rate = (n_consumer_pivoted / total_consumer_cells
                               if total_consumer_cells > 0 else 0.0)

    correctness_ratified_rate_w27 = (
        sum(correctness_w27) / n_cells_run
        if n_cells_run > 0 else 0.0)
    correctness_ratified_rate_w26 = (
        sum(correctness_w26) / n_cells_run
        if n_cells_run > 0 else 0.0)

    # Trust auditing fields from the W27 disambig wrapper.
    n_pool_exhausted_rejections = int(
        multi_chain_registry_w27d.n_pool_exhausted_rejections)
    n_pivots_registered = int(
        multi_chain_registry_w27d.n_pivots_registered)
    n_pivots_rejected = int(multi_chain_registry_w27d.n_pivots_rejected)
    n_signatures_registered = int(
        multi_chain_registry_w27d.n_signatures_registered)
    n_anchors_added = int(multi_chain_registry_w27d.n_anchors_added)
    n_signature_bytes_total = sum(
        int(o.get("salience_signature_envelope", {}).get(
            "n_signature_bytes", 0))
        for o in per_cell_producer_w27d
    )
    n_pivot_bytes_total = sum(
        int(o.get("chain_pivot_envelope", {}).get("n_pivot_bytes", 0))
        for o in per_cell_producer_w27d
    )

    # W27 orchestrator pool size (one slot per signature observed).
    pool_size_final = int(producer_w27.pool_size)
    pool_exhausted_w27 = int(producer_w27.n_pool_exhausted_rejections)

    results: dict[str, Any] = {
        "bank": bank,
        "T_decoder": T_decoder,
        "K_consumers": K_consumers,
        "n_cells": n_cells_run,
        "bank_seed": bank_seed,
        "chain_persist_window": chain_persist_window,
        "max_active_chains": effective_max_chains,
        "signature_period": signature_period,
        # Apples-to-apples token comparisons
        "mean_total_w24_visible_tokens": round(mean_w24_baseline, 4),
        "mean_total_w25_visible_tokens": round(mean_w25_baseline, 4),
        "mean_total_w26_visible_tokens": round(mean_w26_baseline, 4),
        "mean_total_w27_visible_tokens": round(mean_w27, 4),
        "mean_savings_w27_vs_w26_per_cell": round(save_w27_vs_w26, 4),
        "mean_savings_w27_vs_w25_per_cell": round(save_w27_vs_w25, 4),
        "mean_savings_w27_vs_w24_per_cell": round(save_w27_vs_w24, 4),
        "mean_savings_w26_vs_w25_per_cell": round(save_w26_vs_w25, 4),
        "savings_pct_w27_vs_w26":
            round(pct(save_w27_vs_w26, mean_w26_baseline), 2),
        "savings_pct_w27_vs_w25":
            round(pct(save_w27_vs_w25, mean_w25_baseline), 2),
        "savings_pct_w27_vs_w24":
            round(pct(save_w27_vs_w24, mean_w24_baseline), 2),
        # Correctness — both methods reported separately
        "correctness_ratified_rate": round(correctness_ratified_rate_w27, 4),
        "correctness_ratified_rate_w27":
            round(correctness_ratified_rate_w27, 4),
        "correctness_ratified_rate_w26":
            round(correctness_ratified_rate_w26, 4),
        "consumer_pivoted_rate": round(consumer_pivoted_rate, 4),
        "n_consumer_pivoted": n_consumer_pivoted,
        # W27 orchestrator pool diagnostics
        "pool_size_final": pool_size_final,
        "pool_exhausted_rejections_orchestrator": pool_exhausted_w27,
        # W27 audited disambig wrapper diagnostics
        "n_signature_bytes_total": n_signature_bytes_total,
        "n_pivot_bytes_total": n_pivot_bytes_total,
        "n_pool_exhausted_rejections_disambig":
            n_pool_exhausted_rejections,
        "n_pivots_registered_disambig": n_pivots_registered,
        "n_pivots_rejected_disambig": n_pivots_rejected,
        "n_signatures_registered_disambig": n_signatures_registered,
        "n_anchors_added_disambig": n_anchors_added,
        "branch_counts_producer_w27": branch_counts_producer_w27,
        "branch_counts_producer_w26": branch_counts_producer_w26,
    }

    if verbose:
        print(f"\n=== Phase 74 — R-74-{bank.upper()} ===")
        print(f"bank={bank}, T_decoder={T_decoder}, "
              f"K={K_consumers}, n_cells={n_cells_run}, "
              f"window={chain_persist_window}, "
              f"max_active_chains={effective_max_chains}")
        print(f"W24={mean_w24_baseline:.2f}  "
              f"W25={mean_w25_baseline:.2f}  "
              f"W26={mean_w26_baseline:.2f}  "
              f"W27={mean_w27:.2f}")
        print(f"save W27 vs W26: {save_w27_vs_w26:.2f} "
              f"({pct(save_w27_vs_w26, mean_w26_baseline):.1f}%)")
        print(f"save W27 vs W25: {save_w27_vs_w25:.2f} "
              f"({pct(save_w27_vs_w25, mean_w25_baseline):.1f}%)")
        print(f"correctness W27={correctness_ratified_rate_w27:.4f} "
              f"W26={correctness_ratified_rate_w26:.4f}, "
              f"pool_size={pool_size_final}, "
              f"pool_exhausted={pool_exhausted_w27}")
        print(f"W27 producer branches={branch_counts_producer_w27}")
        print(f"W26 producer branches={branch_counts_producer_w26}")

    return results


def run_phase74_seed_stability_sweep(
        *,
        seeds: tuple[int, ...] = (11, 17, 23, 29, 31),
        bank: str = "divergent_recover",
        T_decoder: int | None = None,
        K_consumers: int = 3,
        n_eval: int = 16,
        bank_replicates: int = 4,
        chain_persist_window: int | None = None,
        max_active_chains: int = 8,
        signature_period: int = 4,
        verbose: bool = False,
) -> dict[str, Any]:
    rows = []
    for seed in seeds:
        r = run_phase74(
            bank=bank, T_decoder=T_decoder, K_consumers=K_consumers,
            n_eval=n_eval, bank_replicates=bank_replicates,
            bank_seed=seed,
            chain_persist_window=chain_persist_window,
            max_active_chains=max_active_chains,
            signature_period=signature_period,
            verbose=verbose)
        rows.append(r)
    save_w26 = [r["mean_savings_w27_vs_w26_per_cell"] for r in rows]
    save_w25 = [r["mean_savings_w27_vs_w25_per_cell"] for r in rows]
    save_w24 = [r["mean_savings_w27_vs_w24_per_cell"] for r in rows]
    correctness_list = [r["correctness_ratified_rate"] for r in rows]
    return {
        "seeds": list(seeds),
        "bank": bank,
        "T_decoder": T_decoder,
        "K_consumers": K_consumers,
        "max_active_chains": max_active_chains,
        "rows": rows,
        "min_savings_w27_vs_w26": min(save_w26),
        "mean_savings_w27_vs_w26": sum(save_w26) / len(save_w26),
        "min_savings_w27_vs_w25": min(save_w25),
        "mean_savings_w27_vs_w25": sum(save_w25) / len(save_w25),
        "min_savings_w27_vs_w24": min(save_w24),
        "all_savings_w26_positive": all(s > 0 for s in save_w26),
        "min_correctness": min(correctness_list),
        "all_correctness_1000": all(c >= 1.0 for c in correctness_list),
    }


def run_cross_regime_p74(
        *,
        K_consumers: int = 3,
        n_eval: int = 16,
        bank_replicates: int = 4,
        bank_seed: int = 11,
        verbose: bool = False,
) -> dict[str, Any]:
    """Run the full R-74 family at two T_decoder budgets."""
    results = {}
    sub_banks = (
        "divergent_recover", "xoracle_recover", "chain_shared",
        "pool_exhausted", "pivot_tampered", "signature_drift",
    )
    for bank in sub_banks:
        for T_decoder in (None, 24):
            key = f"{bank}_T{T_decoder}"
            results[key] = run_phase74(
                bank=bank, T_decoder=T_decoder,
                K_consumers=K_consumers, n_eval=n_eval,
                bank_replicates=bank_replicates,
                bank_seed=bank_seed, verbose=verbose)
    return results


def run_signature_period_sweep(
        *,
        periods: tuple[int, ...] = (1, 2, 4, 8, 16),
        n_eval: int = 16,
        bank_seed: int = 11,
        verbose: bool = False,
) -> dict[str, Any]:
    """Sweep signature_period on R-74-DIVERGENT-RECOVER.

    period = N means a single signature run of length N (no
    divergence within window); period = 1 maximises divergence.
    """
    rows = []
    for p in periods:
        r = run_phase74(
            bank="divergent_recover", T_decoder=None,
            K_consumers=3, n_eval=n_eval, bank_seed=bank_seed,
            chain_persist_window=n_eval, max_active_chains=8,
            signature_period=p, verbose=verbose)
        rows.append(r)
    return {
        "periods": list(periods),
        "n_eval": n_eval,
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli() -> None:
    ap = argparse.ArgumentParser(
        description="Phase 74 — W27 multi-chain salience-keyed dense-control fanout")
    ap.add_argument("--bank", default="divergent_recover",
                     choices=["divergent_recover", "xoracle_recover",
                                "chain_shared",
                                "pool_exhausted", "pivot_tampered",
                                "signature_drift",
                                "cross_regime", "signature_period_sweep"])
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
        result = run_cross_regime_p74(
            K_consumers=args.K_consumers, n_eval=args.n_eval,
            bank_replicates=args.bank_replicates,
            bank_seed=args.bank_seed, verbose=args.verbose)
    elif args.bank == "signature_period_sweep":
        result = run_signature_period_sweep(
            periods=(1, 2, 4, 8, 16), n_eval=args.n_eval,
            bank_seed=args.bank_seed, verbose=args.verbose)
    elif args.seed_sweep:
        result = run_phase74_seed_stability_sweep(
            bank=args.bank, T_decoder=T_decoder,
            K_consumers=args.K_consumers, n_eval=args.n_eval,
            bank_replicates=args.bank_replicates,
            chain_persist_window=chain_persist_window,
            max_active_chains=args.max_active_chains,
            signature_period=args.signature_period,
            verbose=args.verbose)
    else:
        result = run_phase74(
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
