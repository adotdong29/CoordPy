"""Phase 73 — chain-persisted dense-control fanout (SDK v3.27, W26 family).

The follow-up to SDK v3.26 (W25) on the LatentMAS "shared KV pool +
cross-cell amortisation" direction. W25 reduced multi-agent fanout
from K × C tokens/cell to C + K tokens/cell.  W25 still requires the
producer to emit the full ``n_w15_kept`` salience-token block on every
cell, even when the salience content is structurally repeated.  W26
chains the producer's compact state across cells inside a bounded
chain window:

* Anchor cell      — producer pays full W25 cost (``C`` ≈ 14.6 tokens).
* In-window cells  — producer pays a single ``<chain_advance:DDDD>``
                     token (1 token).
* Re-anchor        — every ``chain_persist_window`` cells.

Phase-73 sub-banks
==================

Five pre-committed sub-banks:

* **R-73-CHAIN-SHARED** (W26-1 efficiency anchor, default config).
  1 producer + K=3 consumers running R-69-CACHE-FANOUT for 16 cells
  with ``chain_persist_window = 16``.  Strict success bar:
  ``mean_total_w26_visible_tokens`` strictly below
  ``mean_total_w25_visible_tokens`` AND
  ``correctness_ratified_rate = 1.000`` AND
  ``chain_consumer_resolved_rate = 1.000``.

* **R-73-CHAIN-WINDOWED** (W26-1 windowed efficiency anchor).
  Same as R-73-CHAIN-SHARED but ``chain_persist_window = 4``;
  forces 4 anchor cells over 16 cells.  Savings should be smaller
  but still strictly positive over W25.

* **R-73-NO-CHAIN** (W26-Λ-no-chain named falsifier).
  ``chain_persist_window = 1`` → every cell is an anchor →
  W26 reduces to W25 byte-for-byte.

* **R-73-CHAIN-TAMPERED** (W26-Λ-tampered named falsifier).
  Producer's chain advance is corrupted (``advance_cid`` overwritten);
  controller rejects every advance; W26 falls through to W25.

* **R-73-PROJECTION-MISMATCH** (W26-Λ-projection-mismatch named
  falsifier).  One consumer asks for a projection_id not in their
  slot; controller rejects.  Strict requirement:
  rejected on every cell for that consumer; the other two consumers
  still resolve correctly.

* **R-73-DIVERGENT** (W26-Λ-divergent stress regime).  Half the
  cells flip gold subset (``orders, payments`` → ``orders, login``);
  re-anchor at the divergence point.  Savings collapse on the
  re-anchor cell but should remain positive overall.

Success criterion
-----------------
A milestone is W26-1 *discharged* iff, across all five pre-committed
bank_seed values, the chain-persisted path strictly reduces total
visible tokens over the W25 path on R-73-CHAIN-SHARED AND
correctness is preserved byte-for-byte.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
import sys
from typing import Any

from vision_mvp.coordpy.team_coord import (
    # W22 stack
    OracleRegistration, SchemaCapsule,
    build_incident_triage_schema_capsule,
    # W23/W24 stack
    CrossCellDeltaDisambiguator, MultiCellSessionCompactor,
    # W25 stack
    FanoutEnvelope, SharedFanoutDisambiguator, SharedFanoutRegistry,
    verify_fanout, W25_FANOUT_SCHEMA_VERSION,
    W25_BRANCH_FANOUT_PRODUCER_EMITTED,
    W25_BRANCH_FANOUT_CONSUMER_RESOLVED,
    W25_BRANCH_FANOUT_CONSUMER_REJECTED, W25_BRANCH_DISABLED,
    W25_BRANCH_NO_TRIGGER,
    W24_BRANCH_COMPACT_RESOLVED, W24_BRANCH_BELOW_WINDOW,
    W23_BRANCH_GENESIS,
    # W26 new
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
    # Existing oracle helpers
    ServiceGraphOracle, ChangeHistoryOracle, OnCallNotesOracle,
    DisagreeingHonestOracle, CachingOracleAdapter, SharedReadCache,
    OutsideQuery,
    # Team handoff / decoder infrastructure
    _DecodedHandoff, BundleAwareTeamDecoder,
    collect_admitted_handoffs,
    TeamCoordinator, FifoAdmissionPolicy, DEFAULT_ROLE_BUDGETS,
    audit_team_lifecycle, CapsuleLedger,
)


# ---------------------------------------------------------------------------
# Bank builders
# ---------------------------------------------------------------------------

def build_phase73_bank(
        *,
        n_replicates: int = 4,
        seed: int = 11,
        n_cells: int = 16,
        bank: str = "chain_shared",
) -> list[list[list[_DecodedHandoff]]]:
    """Build a Phase-73 multi-cell bank.

    Each element is one ``cell``: a list of per-round handoff lists
    (round 0 + round 1).  Returns ``n_cells`` cells.

    The oracle ecology is the same R-69-CACHE-FANOUT incident-triage
    setup used by W22/W23/W24/W25.  ``divergent`` mode flips the
    second-half cells to a different gold subset to stress re-anchor.
    """
    rng = random.Random(seed)

    GOLD_PRIMARY = ("orders", "payments")
    GOLD_DIVERGENT = ("orders", "login")
    DECOY = "cache"

    def _make_handoff(source: str, kind: str, svc: str,
                       magnitude: float) -> _DecodedHandoff:
        payload = f"service={svc} magnitude={magnitude:.2f}"
        return _DecodedHandoff(
            source_role=source, claim_kind=kind, payload=payload)

    def _cell(cell_idx: int) -> list[list[_DecodedHandoff]]:
        if bank == "divergent" and cell_idx >= n_cells // 2:
            GOLD = GOLD_DIVERGENT
        else:
            GOLD = GOLD_PRIMARY
        r0: list[_DecodedHandoff] = []
        for i, svc in enumerate(GOLD):
            mag = rng.uniform(0.7, 1.0)
            r0.append(_make_handoff(
                f"role_{i}", "LOAD_SPIKE", svc, mag))
        r0.append(_make_handoff("role_2", "ERROR_RATE", GOLD[0],
                                  rng.uniform(0.6, 0.9)))
        r0.append(_make_handoff("role_3", "CPU_SPIKE", DECOY,
                                  rng.uniform(0.1, 0.3)))
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
    """Build a W25 SharedFanoutDisambiguator (W22→W23→W24→W25)."""
    from vision_mvp.coordpy.team_coord import (
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
    """Build a W26 ChainPersistedFanoutDisambiguator on top of W25."""
    inner = _build_w25_stack(
        T_decoder=T_decoder, schema=schema, raw_oracles=raw_oracles,
        agent_id=agent_id, is_producer=is_producer,
        producer_agent_id=producer_agent_id,
        consumer_agent_ids=consumer_agent_ids,
        registry=fanout_registry)
    dis = ChainPersistedFanoutDisambiguator(
        inner=inner,
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
    return dis


# ---------------------------------------------------------------------------
# run_phase73
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


def _w24_visible(out: dict) -> int:
    if "shared_fanout_hybrid" in out:
        return int(out["shared_fanout_hybrid"].get(
            "n_w24_visible_tokens", 0))
    if "session_compact_hybrid" in out:
        return int(out["session_compact_hybrid"].get(
            "n_w24_visible_tokens_to_decider", 0))
    return 0


def _w26_branch(out: dict) -> str:
    if "chain_persisted_hybrid" in out:
        return str(out["chain_persisted_hybrid"].get(
            "decoder_branch", ""))
    return ""


def run_phase73(
        *,
        bank: str = "chain_shared",
        T_decoder: int | None = None,
        K_consumers: int = 3,
        n_eval: int = 16,
        bank_replicates: int = 4,
        bank_seed: int = 11,
        chain_persist_window: int | None = None,
        verbose: bool = False,
) -> dict[str, Any]:
    """Run one Phase-73 sub-bank and return the results dict.

    Parameters
    ----------
    bank:
        One of ``"chain_shared"``, ``"chain_windowed"``,
        ``"no_chain"``, ``"chain_tampered"``, ``"projection_mismatch"``,
        ``"divergent"``.
    chain_persist_window:
        Override default chain window (16 for chain_shared, 4 for
        chain_windowed, 1 for no_chain).
    """
    schema = build_incident_triage_schema_capsule()

    raw_oracles: tuple[tuple[Any, str], ...] = (
        (ServiceGraphOracle(oracle_id="service_graph"), "service_graph"),
        (ChangeHistoryOracle(oracle_id="change_history"), "change_history"),
    )

    # Determine bench cells (bank may need divergent layout)
    cells_bank = ("divergent" if bank == "divergent" else "chain_shared")
    cells = build_phase73_bank(
        n_replicates=bank_replicates, seed=bank_seed,
        n_cells=n_eval, bank=cells_bank)

    producer_id = "producer_agent"
    consumer_ids = tuple(f"consumer_{k}" for k in range(K_consumers))

    if chain_persist_window is None:
        if bank == "no_chain":
            chain_persist_window = 1
        elif bank == "chain_windowed":
            chain_persist_window = 4
        else:
            chain_persist_window = n_eval  # one anchor

    fanout_registry = SharedFanoutRegistry(schema=schema)
    chain_registry = ChainPersistedFanoutRegistry(schema=schema)

    # Per-consumer projection slot policy.
    # By default each consumer gets a unique projection_id named after
    # itself with the canonical projected_subset (gold services).
    projection_id_for_consumer: dict[str, str] = {
        cid: f"proj_{cid}" for cid in consumer_ids
    }
    projected_tags_for_consumer: dict[str, tuple[str, ...]] = {
        cid: ("orders", "payments") for cid in consumer_ids
    }

    # Build agents.
    producer_dis = _build_w26_stack(
        T_decoder=T_decoder, schema=schema, raw_oracles=raw_oracles,
        agent_id=producer_id, is_producer=True,
        producer_agent_id=producer_id,
        consumer_agent_ids=consumer_ids,
        fanout_registry=fanout_registry,
        chain_registry=chain_registry,
        chain_persist_window=chain_persist_window,
        projection_id_for_consumer=projection_id_for_consumer,
        projected_tags_for_consumer=projected_tags_for_consumer,
    )

    # Set up consumer disambiguators.  In `projection_mismatch` bank,
    # the FIRST consumer requests a projection_id NOT in its slot.
    consumer_dis_list: list[ChainPersistedFanoutDisambiguator] = []
    for k, cid in enumerate(consumer_ids):
        per_consumer_pid = dict(projection_id_for_consumer)
        if bank == "projection_mismatch" and k == 0:
            per_consumer_pid[cid] = "WRONG_PROJECTION_ID"
        consumer_dis_list.append(
            _build_w26_stack(
                T_decoder=T_decoder, schema=schema,
                raw_oracles=raw_oracles,
                agent_id=cid, is_producer=False,
                producer_agent_id=producer_id,
                consumer_agent_ids=consumer_ids,
                fanout_registry=fanout_registry,
                chain_registry=chain_registry,
                chain_persist_window=chain_persist_window,
                projection_id_for_consumer=per_consumer_pid,
                projected_tags_for_consumer=projected_tags_for_consumer,
            )
        )

    # Per-cell metrics
    per_cell_producer: list[dict[str, Any]] = []
    per_cell_consumers: list[list[dict[str, Any]]] = []
    correctness_producer: list[bool] = []

    for cell_idx, cell_handoffs in enumerate(cells):
        # Producer first (registers chain anchor / advance)
        p_out = producer_dis.decode_rounds(cell_handoffs)

        # Optionally tamper the latest advance — chain-tampered bank
        if bank == "chain_tampered":
            adv = producer_dis.last_advance
            if adv is not None:
                # Overwrite the advance_cid in the registry to break parent linkage
                if adv.advance_cid in chain_registry._advances:
                    tampered = dataclasses.replace(
                        adv, advance_cid="TAMPERED" + "0" * 56)
                    chain_registry._advances[
                        tampered.advance_cid] = tampered
                    # Reset chain_state so next advance's expected parent is the tampered one
                    chain_registry._chain_state[adv.chain_root_cid] = {
                        "parent_cid": tampered.advance_cid,
                        "cell_in_chain": int(adv.cell_in_chain),
                    }
                    # Also point producer to a wrong parent so it
                    # tries to advance with a stale parent_cid.
                    producer_dis._last_advance = adv  # producer's view stale

        per_cell_producer.append(p_out)

        ans = p_out.get("answer") or p_out
        svcs = ans.get("services") if isinstance(ans, dict) else None
        if svcs is None:
            svcs = p_out.get("services")
        # Determine expected gold for this cell
        if bank == "divergent" and cell_idx >= n_eval // 2:
            expected = {"orders", "login"}
        else:
            expected = {"orders", "payments"}
        correct = (set(svcs or []) == expected if svcs is not None
                    else False)
        correctness_producer.append(correct)

        c_row: list[dict[str, Any]] = []
        for c_dis in consumer_dis_list:
            c_out = c_dis.decode_rounds(cell_handoffs)
            c_row.append(c_out)
        per_cell_consumers.append(c_row)

    n_cells_run = len(per_cell_producer)

    # Producer tokens
    w25_tokens_producer = [_w25_visible(o) for o in per_cell_producer]
    w26_tokens_producer = [_w26_visible(o) for o in per_cell_producer]
    w24_tokens_producer = [_w24_visible(o) for o in per_cell_producer]

    # Consumer tokens
    w25_tokens_consumers_per_cell = [
        sum(_w25_visible(c) for c in row) for row in per_cell_consumers
    ]
    w26_tokens_consumers_per_cell = [
        sum(_w26_visible(c) for c in row) for row in per_cell_consumers
    ]
    w24_tokens_consumers_per_cell = [
        sum(_w24_visible(c) for c in row) for row in per_cell_consumers
    ]

    total_w25_per_cell = [
        w25_tokens_producer[i] + w25_tokens_consumers_per_cell[i]
        for i in range(n_cells_run)
    ]
    total_w26_per_cell = [
        w26_tokens_producer[i] + w26_tokens_consumers_per_cell[i]
        for i in range(n_cells_run)
    ]
    total_w24_per_cell = [
        w24_tokens_producer[i] + w24_tokens_consumers_per_cell[i]
        for i in range(n_cells_run)
    ]

    mean_total_w25 = (sum(total_w25_per_cell) / n_cells_run
                       if n_cells_run > 0 else 0.0)
    mean_total_w26 = (sum(total_w26_per_cell) / n_cells_run
                       if n_cells_run > 0 else 0.0)
    mean_total_w24 = (sum(total_w24_per_cell) / n_cells_run
                       if n_cells_run > 0 else 0.0)

    mean_savings_w26_vs_w25 = mean_total_w25 - mean_total_w26
    mean_savings_w26_vs_w24 = mean_total_w24 - mean_total_w26

    # Branch counts on consumers
    n_consumer_resolved = sum(
        1 for row in per_cell_consumers
        for c in row
        if _w26_branch(c) == W26_BRANCH_CHAIN_PROJECTION_RESOLVED
    )
    n_consumer_rejected = sum(
        1 for row in per_cell_consumers
        for c in row
        if _w26_branch(c) in (
            W26_BRANCH_CHAIN_REJECTED,
            W26_BRANCH_CHAIN_PROJECTION_REJECTED)
    )
    n_consumer_no_trigger = sum(
        1 for row in per_cell_consumers
        for c in row
        if _w26_branch(c) in (W26_BRANCH_NO_TRIGGER, W26_BRANCH_DISABLED, "")
    )
    total_consumer_cells = n_cells_run * K_consumers
    chain_consumer_resolved_rate = (
        n_consumer_resolved / total_consumer_cells
        if total_consumer_cells > 0 else 0.0)
    chain_consumer_rejected_rate = (
        n_consumer_rejected / total_consumer_cells
        if total_consumer_cells > 0 else 0.0)

    # Producer branch counts
    branch_counts_producer: dict[str, int] = {}
    for o in per_cell_producer:
        b = _w26_branch(o)
        branch_counts_producer[b] = branch_counts_producer.get(b, 0) + 1

    # Anchor / advance bytes
    n_anchor_bytes_total = sum(
        int(o.get("chain_anchor_envelope", {}).get("n_anchor_bytes", 0))
        for o in per_cell_producer
    )
    n_advance_bytes_total = sum(
        int(o.get("chain_advance_envelope", {}).get("n_advance_bytes", 0))
        for o in per_cell_producer
    )

    # Registry stats
    reg_anchors = int(chain_registry.n_anchors_registered)
    reg_advances = int(chain_registry.n_advances_registered)
    reg_advances_rejected = int(chain_registry.n_advances_rejected)
    reg_proj_resolved = int(chain_registry.n_projections_resolved)
    reg_proj_rejected = int(chain_registry.n_projections_rejected)

    correctness_ratified_rate = (
        sum(correctness_producer) / n_cells_run
        if n_cells_run > 0 else 0.0)

    savings_pct_w26_vs_w25 = (
        100.0 * mean_savings_w26_vs_w25 / mean_total_w25
        if mean_total_w25 > 0 else 0.0)
    savings_pct_w26_vs_w24 = (
        100.0 * mean_savings_w26_vs_w24 / mean_total_w24
        if mean_total_w24 > 0 else 0.0)

    results: dict[str, Any] = {
        "bank": bank,
        "T_decoder": T_decoder,
        "K_consumers": K_consumers,
        "n_cells": n_cells_run,
        "bank_seed": bank_seed,
        "chain_persist_window": chain_persist_window,
        "mean_total_w24_visible_tokens": round(mean_total_w24, 4),
        "mean_total_w25_visible_tokens": round(mean_total_w25, 4),
        "mean_total_w26_visible_tokens": round(mean_total_w26, 4),
        "mean_savings_w26_vs_w25_per_cell":
            round(mean_savings_w26_vs_w25, 4),
        "mean_savings_w26_vs_w24_per_cell":
            round(mean_savings_w26_vs_w24, 4),
        "savings_pct_w26_vs_w25": round(savings_pct_w26_vs_w25, 2),
        "savings_pct_w26_vs_w24": round(savings_pct_w26_vs_w24, 2),
        "correctness_ratified_rate": round(correctness_ratified_rate, 4),
        "chain_consumer_resolved_rate":
            round(chain_consumer_resolved_rate, 4),
        "chain_consumer_rejected_rate":
            round(chain_consumer_rejected_rate, 4),
        "n_consumer_resolved": n_consumer_resolved,
        "n_consumer_rejected": n_consumer_rejected,
        "n_consumer_no_trigger": n_consumer_no_trigger,
        "n_anchor_bytes_total": n_anchor_bytes_total,
        "n_advance_bytes_total": n_advance_bytes_total,
        "registry_n_anchors": reg_anchors,
        "registry_n_advances": reg_advances,
        "registry_n_advances_rejected": reg_advances_rejected,
        "registry_n_projections_resolved": reg_proj_resolved,
        "registry_n_projections_rejected": reg_proj_rejected,
        "branch_counts_producer": branch_counts_producer,
    }

    if verbose:
        print(f"\n=== Phase 73 — R-73-{bank.upper()} ===")
        print(f"bank={bank}, T_decoder={T_decoder}, "
              f"K_consumers={K_consumers}, n_cells={n_cells_run}, "
              f"chain_persist_window={chain_persist_window}")
        print(f"W24 total={mean_total_w24:.2f} "
              f"W25 total={mean_total_w25:.2f} "
              f"W26 total={mean_total_w26:.2f}")
        print(f"savings W26 vs W25: {mean_savings_w26_vs_w25:.2f} "
              f"({savings_pct_w26_vs_w25:.1f}%)")
        print(f"savings W26 vs W24: {mean_savings_w26_vs_w24:.2f} "
              f"({savings_pct_w26_vs_w24:.1f}%)")
        print(f"correctness_ratified_rate={correctness_ratified_rate:.4f}")
        print(f"chain_consumer_resolved_rate={chain_consumer_resolved_rate:.4f}")
        print(f"chain_consumer_rejected_rate={chain_consumer_rejected_rate:.4f}")
        print(f"branches producer={branch_counts_producer}")
        print(f"registry anchors={reg_anchors} advances={reg_advances} "
              f"adv_rejected={reg_advances_rejected} "
              f"proj_rejected={reg_proj_rejected}")

    return results


def run_phase73_seed_stability_sweep(
        *,
        seeds: tuple[int, ...] = (11, 17, 23, 29, 31),
        bank: str = "chain_shared",
        T_decoder: int | None = None,
        K_consumers: int = 3,
        n_eval: int = 16,
        bank_replicates: int = 4,
        chain_persist_window: int | None = None,
        verbose: bool = False,
) -> dict[str, Any]:
    rows = []
    for seed in seeds:
        r = run_phase73(
            bank=bank, T_decoder=T_decoder, K_consumers=K_consumers,
            n_eval=n_eval, bank_replicates=bank_replicates,
            bank_seed=seed,
            chain_persist_window=chain_persist_window,
            verbose=verbose)
        rows.append(r)
    savings_w25 = [r["mean_savings_w26_vs_w25_per_cell"] for r in rows]
    savings_w24 = [r["mean_savings_w26_vs_w24_per_cell"] for r in rows]
    correctness_list = [r["correctness_ratified_rate"] for r in rows]
    return {
        "seeds": list(seeds),
        "bank": bank,
        "T_decoder": T_decoder,
        "K_consumers": K_consumers,
        "rows": rows,
        "min_savings_w26_vs_w25": min(savings_w25),
        "mean_savings_w26_vs_w25": sum(savings_w25) / len(savings_w25),
        "min_savings_w26_vs_w24": min(savings_w24),
        "mean_savings_w26_vs_w24": sum(savings_w24) / len(savings_w24),
        "all_savings_w25_positive": all(s > 0 for s in savings_w25),
        "all_savings_w24_positive": all(s > 0 for s in savings_w24),
        "min_correctness": min(correctness_list),
        "all_correctness_1000": all(c >= 1.0 for c in correctness_list),
    }


def run_cross_regime_p73(
        *,
        K_consumers: int = 3,
        n_eval: int = 16,
        bank_replicates: int = 4,
        bank_seed: int = 11,
        verbose: bool = False,
) -> dict[str, Any]:
    """Run the full R-73 family at two T_decoder budgets."""
    results = {}
    sub_banks = ("chain_shared", "chain_windowed", "no_chain",
                  "chain_tampered", "projection_mismatch", "divergent")
    for bank in sub_banks:
        for T_decoder in (None, 24):
            key = f"{bank}_T{T_decoder}"
            results[key] = run_phase73(
                bank=bank, T_decoder=T_decoder,
                K_consumers=K_consumers, n_eval=n_eval,
                bank_replicates=bank_replicates,
                bank_seed=bank_seed, verbose=verbose)
    return results


def run_k_scaling_sweep(
        *,
        K_values: tuple[int, ...] = (3, 5, 8, 10),
        n_eval: int = 16,
        bank_replicates: int = 4,
        bank_seed: int = 11,
        T_decoder: int | None = None,
        verbose: bool = False,
) -> dict[str, Any]:
    """Sweep K (number of consumers) on R-73-CHAIN-SHARED.

    Discharges W25-C-K-SCALING and the analogous W26-C-K-SCALING
    conjecture by measuring savings at K ∈ {3, 5, 8, 10}.
    """
    rows = []
    for K in K_values:
        r = run_phase73(
            bank="chain_shared", T_decoder=T_decoder,
            K_consumers=K, n_eval=n_eval,
            bank_replicates=bank_replicates, bank_seed=bank_seed,
            verbose=verbose)
        rows.append(r)
    return {
        "K_values": list(K_values),
        "n_eval": n_eval,
        "T_decoder": T_decoder,
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli() -> None:
    ap = argparse.ArgumentParser(
        description="Phase 73 — W26 chain-persisted dense-control fanout")
    ap.add_argument("--bank", default="chain_shared",
                     choices=["chain_shared", "chain_windowed",
                                "no_chain", "chain_tampered",
                                "projection_mismatch", "divergent",
                                "cross_regime", "k_scaling"])
    ap.add_argument("--K-consumers", type=int, default=3)
    ap.add_argument("--decoder-budget", type=int, default=-1)
    ap.add_argument("--chain-persist-window", type=int, default=-1)
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
        result = run_cross_regime_p73(
            K_consumers=args.K_consumers, n_eval=args.n_eval,
            bank_replicates=args.bank_replicates,
            bank_seed=args.bank_seed, verbose=args.verbose)
    elif args.bank == "k_scaling":
        result = run_k_scaling_sweep(
            K_values=(3, 5, 8, 10), n_eval=args.n_eval,
            bank_replicates=args.bank_replicates,
            bank_seed=args.bank_seed,
            T_decoder=T_decoder, verbose=args.verbose)
    elif args.seed_sweep:
        result = run_phase73_seed_stability_sweep(
            bank=args.bank, T_decoder=T_decoder,
            K_consumers=args.K_consumers, n_eval=args.n_eval,
            bank_replicates=args.bank_replicates,
            chain_persist_window=chain_persist_window,
            verbose=args.verbose)
    else:
        result = run_phase73(
            bank=args.bank, T_decoder=T_decoder,
            K_consumers=args.K_consumers, n_eval=args.n_eval,
            bank_replicates=args.bank_replicates,
            bank_seed=args.bank_seed,
            chain_persist_window=chain_persist_window,
            verbose=args.verbose)

    out_text = json.dumps(result, indent=2, default=str)
    if args.out == "-":
        print(out_text)
    else:
        with open(args.out, "w") as fh:
            fh.write(out_text)


if __name__ == "__main__":
    _cli()
