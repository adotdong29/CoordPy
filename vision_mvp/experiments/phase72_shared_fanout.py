"""Phase 72 — shared-fanout dense-control capsule + cross-agent state reuse
(SDK v3.26, W25 family anchor).

The follow-up to SDK v3.25 (W24) on the LatentMAS "hardware pooling /
shared KV pool" direction.  W24 reduced per-agent session-context cost to
one compact token per cell via bounded-window compaction.  W25 extends that
to the MULTI-AGENT case: when K consumer agents all need the SAME cross-cell
session state produced by one producer agent, W24 would still require K
independent compact envelopes (K × C tokens total).  W25 replaces those K
envelopes with a single FanoutEnvelope registered by the producer + K
single-token <fanout_ref:DDDD> references for the consumers.

Phase-72 sub-banks
==================

Three pre-committed cells:

* **R-72-FANOUT-SHARED** (W25-1 efficiency anchor, default config).
  1 producer + K=3 consumers (4 agents total) running the same
  R-69-CACHE-FANOUT oracle ecology for 16 cells.  Strict success bar:
  ``mean_n_w25_visible_tokens_total`` strictly below
  ``mean_n_w24_visible_tokens_total`` AND
  ``correctness_ratified_rate = 1.000`` on the producer.  The saving
  per cell is K × (n_w24_compact_tokens − 1).

* **R-72-DISJOINT** (W25-Λ-disjoint named falsifier).  Each agent has a
  *separate* ``SharedFanoutDisambiguator`` with no shared registry; W25
  fires ``W25_BRANCH_NO_TRIGGER`` (disabled path) and reduces to W24
  per-agent on every agent.  Strict requirement:
  ``mean_n_w25_visible_tokens_total == mean_n_w24_visible_tokens_total``.

* **R-72-FANOUT-POISONED** (W25-3 trust falsifier).  The producer
  registers the fanout normally but a consumer tries to resolve with an
  *unauthorised* consumer_id (not in consumer_agent_ids).  Strict
  requirement: ``fanout_consumer_rejected_rate = 1.000`` on the
  unauthorised consumer.

Success criterion
-----------------
A milestone is W25-1 *discharged* iff, across all five pre-committed
bank_seed values, the shared-fanout path strictly reduces total visible
tokens over the W24 per-agent path on R-72-FANOUT-SHARED AND
correctness is preserved byte-for-byte.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
import sys
from typing import Any

from vision_mvp.wevra.team_coord import (
    # W22 stack
    OracleRegistration, SchemaCapsule,
    build_incident_triage_schema_capsule,
    # W23/W24 stack
    CrossCellDeltaDisambiguator, MultiCellSessionCompactor,
    # W25 new
    FanoutEnvelope, SharedFanoutDisambiguator, SharedFanoutRegistry,
    verify_fanout,
    W25_FANOUT_SCHEMA_VERSION,
    W25_BRANCH_FANOUT_PRODUCER_EMITTED, W25_BRANCH_FANOUT_CONSUMER_RESOLVED,
    W25_BRANCH_FANOUT_CONSUMER_REJECTED, W25_BRANCH_DISABLED,
    W25_BRANCH_NO_TRIGGER,
    # Branch constants for inner layers
    W24_BRANCH_COMPACT_RESOLVED, W24_BRANCH_BELOW_WINDOW,
    W23_BRANCH_GENESIS,
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
# Bank builder
# ---------------------------------------------------------------------------

def build_phase72_bank(
        *,
        n_replicates: int = 4,
        seed: int = 11,
        n_cells: int = 16,
        bank: str = "fanout_shared",
) -> list[list[list[_DecodedHandoff]]]:
    """Build a Phase-72 multi-cell bank.

    Each element is one ``cell``: a list of per-round handoff lists
    (round 0 + round 1).  Returns ``n_cells`` cells.

    The oracle ecology is the same R-69-CACHE-FANOUT incident-triage
    setup used by W22/W23/W24, so the correctness baseline is known
    to be W22-1 = 1.000.
    """
    rng = random.Random(seed)

    # Gold services + decoy
    GOLD = ("orders", "payments")
    DECOY = "cache"
    ALL_SERVICES = GOLD + (DECOY,)

    def _make_handoff(source: str, kind: str, svc: str,
                       magnitude: float) -> _DecodedHandoff:
        payload = f"service={svc} magnitude={magnitude:.2f}"
        return _DecodedHandoff(
            source_role=source,
            claim_kind=kind,
            payload=payload,
        )

    def _cell(cell_idx: int) -> list[list[_DecodedHandoff]]:
        r0: list[_DecodedHandoff] = []
        # Gold: 3 mentions from distinct roles
        for i, svc in enumerate(GOLD):
            mag = rng.uniform(0.7, 1.0)
            r0.append(_make_handoff(
                f"role_{i}", "LOAD_SPIKE", svc, mag))
        # Extra gold corroboration
        r0.append(_make_handoff("role_2", "ERROR_RATE", GOLD[0],
                                  rng.uniform(0.6, 0.9)))
        # Decoy: 1 mention
        r0.append(_make_handoff("role_3", "CPU_SPIKE", DECOY,
                                  rng.uniform(0.1, 0.3)))
        # Round 1: disambiguator hint
        r1: list[_DecodedHandoff] = []
        for svc in GOLD:
            r1.append(_make_handoff(
                "disambiguator", "CORROBORATION", svc, 1.0))
        return [r0, r1]

    return [_cell(i) for i in range(n_cells)]


# ---------------------------------------------------------------------------
# Strategy runners
# ---------------------------------------------------------------------------

def _build_w24_compactor(
        *,
        T_decoder: int | None,
        schema: SchemaCapsule,
        raw_oracles: tuple[tuple[Any, str], ...],
) -> MultiCellSessionCompactor:
    """Build a per-agent W24 MultiCellSessionCompactor.

    Parameters
    ----------
    raw_oracles:
        Tuple of ``(oracle_object, role_label)`` pairs.  Each oracle is
        wrapped in a fresh ``QuorumKeyedCachingOracleAdapter`` sharing a
        per-agent ``QuorumKeyedSharedReadCache``.
    """
    from vision_mvp.wevra.team_coord import (
        AttentionAwareBundleDecoder,
        RelationalCompatibilityDisambiguator,
        BundleContradictionDisambiguator,
        LatentDigestDisambiguator, CrossCellDeltaDisambiguator,
        QuorumKeyedSharedReadCache, QuorumKeyedCachingOracleAdapter,
        TrustWeightedMultiOracleDisambiguator,
        CACHE_FRESHNESS_PER_CELL_NONCE,
    )
    cache = QuorumKeyedSharedReadCache()
    regs = tuple(
        OracleRegistration(
            oracle=QuorumKeyedCachingOracleAdapter(
                inner=oracle, cache=cache,
                oracle_id=role_label),
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
    w24 = MultiCellSessionCompactor(inner=w23, schema=schema,
                                      compact_window=4)
    return w24


def _build_fanout_disambiguator(
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
    """Build a W25 SharedFanoutDisambiguator."""
    inner = _build_w24_compactor(
        T_decoder=T_decoder, schema=schema,
        raw_oracles=raw_oracles)
    dis = SharedFanoutDisambiguator(
        inner=inner,
        fanout_registry=registry,
        agent_id=agent_id,
        is_producer=is_producer,
        producer_agent_id=producer_agent_id,
        consumer_agent_ids=consumer_agent_ids,
        schema=schema,
        enabled=True,
        require_fanout_verification=True,
    )
    return dis


# ---------------------------------------------------------------------------
# run_phase72
# ---------------------------------------------------------------------------

def run_phase72(
        *,
        bank: str = "fanout_shared",
        T_decoder: int | None = None,
        K_consumers: int = 3,
        n_eval: int = 16,
        bank_replicates: int = 4,
        bank_seed: int = 11,
        verbose: bool = False,
) -> dict[str, Any]:
    """Run one Phase-72 sub-bank and return the results dict.

    Parameters
    ----------
    bank:
        One of ``"fanout_shared"`` (default), ``"disjoint"``,
        ``"fanout_poisoned"``.
    T_decoder:
        Decoder budget passed through.  ``None`` = loose (no cap).
    K_consumers:
        Number of consumer agents.  Default 3 (+ 1 producer = 4 total).
    n_eval:
        Cells per session.
    bank_replicates:
        Number of per-round handoff replicates.
    bank_seed:
        RNG seed for the bank.
    """
    schema = build_incident_triage_schema_capsule()

    # Oracle ecology (same as R-69/70/71): (oracle_object, role_label) pairs
    raw_oracles: tuple[tuple[Any, str], ...] = (
        (ServiceGraphOracle(oracle_id="service_graph"), "service_graph"),
        (ChangeHistoryOracle(oracle_id="change_history"), "change_history"),
    )

    cells = build_phase72_bank(
        n_replicates=bank_replicates, seed=bank_seed,
        n_cells=n_eval, bank=bank)

    producer_id = "producer_agent"
    consumer_ids = tuple(f"consumer_{k}" for k in range(K_consumers))

    if bank == "disjoint":
        # Each agent has no shared registry — W25 disabled effectively
        registry = None
    elif bank == "fanout_poisoned":
        # Registry shared but one consumer uses wrong id
        registry = SharedFanoutRegistry(schema=schema)
    else:
        registry = SharedFanoutRegistry(schema=schema)

    # Build all agent disambiguators
    if bank == "disjoint" or registry is None:
        # All agents use plain W24 (no shared fanout)
        producer_dis = _build_w24_compactor(
            T_decoder=T_decoder, schema=schema, raw_oracles=raw_oracles)
        consumer_dis_list = [
            _build_w24_compactor(T_decoder=T_decoder, schema=schema,
                                   raw_oracles=raw_oracles)
            for _ in consumer_ids
        ]
        use_fanout = False
    else:
        producer_dis = _build_fanout_disambiguator(
            T_decoder=T_decoder, schema=schema, raw_oracles=raw_oracles,
            agent_id=producer_id, is_producer=True,
            producer_agent_id=producer_id,
            consumer_agent_ids=consumer_ids,
            registry=registry)
        if bank == "fanout_poisoned":
            # First consumer uses wrong id (not in consumer_agent_ids)
            poisoned_id = "UNAUTHORIZED_AGENT"
            consumer_dis_list = [
                _build_fanout_disambiguator(
                    T_decoder=T_decoder, schema=schema, raw_oracles=raw_oracles,
                    agent_id=(poisoned_id if k == 0 else consumer_ids[k]),
                    is_producer=False,
                    producer_agent_id=producer_id,
                    consumer_agent_ids=consumer_ids,
                    registry=registry)
                for k in range(K_consumers)
            ]
        else:
            consumer_dis_list = [
                _build_fanout_disambiguator(
                    T_decoder=T_decoder, schema=schema, raw_oracles=raw_oracles,
                    agent_id=consumer_ids[k], is_producer=False,
                    producer_agent_id=producer_id,
                    consumer_agent_ids=consumer_ids,
                    registry=registry)
                for k in range(K_consumers)
            ]
        use_fanout = True

    # Per-cell metrics
    per_cell_producer: list[dict[str, Any]] = []
    per_cell_consumers: list[list[dict[str, Any]]] = []
    correctness_producer: list[bool] = []

    for cell_handoffs in cells:
        # Flatten multi-round handoffs for decoder consumption
        flat = [h for rnd in cell_handoffs for h in rnd]

        # Producer decode
        if use_fanout and hasattr(producer_dis, "decode_rounds"):
            p_out = producer_dis.decode_rounds(cell_handoffs)
        elif hasattr(producer_dis, "decode_rounds"):
            p_out = producer_dis.decode_rounds(cell_handoffs)
        else:
            p_out = producer_dis.decode(flat)

        per_cell_producer.append(p_out)

        # Check correctness: did the producer answer correctly?
        ans = p_out.get("answer") or p_out
        svcs = ans.get("services") if isinstance(ans, dict) else None
        if svcs is None:
            svcs = p_out.get("services")
        correct = (set(svcs or []) == {"orders", "payments"}
                    if svcs is not None else False)
        correctness_producer.append(correct)

        # Consumer decodes (each consumer independently)
        c_row: list[dict[str, Any]] = []
        for c_dis in consumer_dis_list:
            if hasattr(c_dis, "decode_rounds"):
                c_out = c_dis.decode_rounds(cell_handoffs)
            else:
                c_out = c_dis.decode(flat)
            c_row.append(c_out)
        per_cell_consumers.append(c_row)

    # Aggregate metrics
    def _w24_tokens(out: dict) -> int:
        if "session_compact_hybrid" in out:
            return int(out["session_compact_hybrid"].get(
                "n_w24_visible_tokens_to_decider", 0))
        if "shared_fanout_hybrid" in out:
            return int(out["shared_fanout_hybrid"].get(
                "n_w24_visible_tokens", 0))
        return 0

    def _w25_tokens(out: dict) -> int:
        if "shared_fanout_hybrid" in out:
            return int(out["shared_fanout_hybrid"].get(
                "n_w25_visible_tokens", 0))
        # For plain W24 agents in disjoint mode, W25 = W24
        return _w24_tokens(out)

    def _w25_branch(out: dict) -> str:
        if "shared_fanout_hybrid" in out:
            return str(out["shared_fanout_hybrid"].get(
                "decoder_branch", ""))
        return ""

    n_cells_run = len(per_cell_producer)

    # Producer tokens (same in W24 and W25; producer cost unchanged)
    w24_tokens_producer = [_w24_tokens(o) for o in per_cell_producer]
    w25_tokens_producer = [_w25_tokens(o) for o in per_cell_producer]

    # Consumer tokens
    w24_tokens_consumers_per_cell = [
        sum(_w24_tokens(c) for c in row)
        for row in per_cell_consumers
    ]
    w25_tokens_consumers_per_cell = [
        sum(_w25_tokens(c) for c in row)
        for row in per_cell_consumers
    ]

    # Total across all agents per cell
    total_w24_per_cell = [
        w24_tokens_producer[i] + w24_tokens_consumers_per_cell[i]
        for i in range(n_cells_run)
    ]
    total_w25_per_cell = [
        w25_tokens_producer[i] + w25_tokens_consumers_per_cell[i]
        for i in range(n_cells_run)
    ]

    mean_total_w24 = (sum(total_w24_per_cell) / n_cells_run
                       if n_cells_run > 0 else 0.0)
    mean_total_w25 = (sum(total_w25_per_cell) / n_cells_run
                       if n_cells_run > 0 else 0.0)
    mean_savings = mean_total_w24 - mean_total_w25

    # Branch counts for consumers
    n_consumer_resolved = sum(
        1 for row in per_cell_consumers
        for c in row
        if _w25_branch(c) == W25_BRANCH_FANOUT_CONSUMER_RESOLVED
    )
    n_consumer_rejected = sum(
        1 for row in per_cell_consumers
        for c in row
        if _w25_branch(c) == W25_BRANCH_FANOUT_CONSUMER_REJECTED
    )
    n_consumer_no_trigger = sum(
        1 for row in per_cell_consumers
        for c in row
        if _w25_branch(c) in (W25_BRANCH_NO_TRIGGER, W25_BRANCH_DISABLED, "")
    )
    total_consumer_cells = n_cells_run * K_consumers
    fanout_consumer_resolved_rate = (
        n_consumer_resolved / total_consumer_cells
        if total_consumer_cells > 0 else 0.0)
    fanout_consumer_rejected_rate = (
        n_consumer_rejected / total_consumer_cells
        if total_consumer_cells > 0 else 0.0)

    # Fanout bytes
    n_fanout_bytes_total = sum(
        int(o.get("shared_fanout_hybrid", {}).get("n_fanout_bytes", 0))
        for o in per_cell_producer
    )

    # Registry stats
    if registry is not None:
        reg_registered = int(registry.n_registered)
        reg_resolved = int(registry.n_resolved)
        reg_rejected_register = int(registry.n_rejected_register)
        reg_rejected_resolve = int(registry.n_rejected_resolve)
    else:
        reg_registered = reg_resolved = reg_rejected_register = 0
        reg_rejected_resolve = 0

    # Correctness
    correctness_ratified_rate = (
        sum(correctness_producer) / n_cells_run
        if n_cells_run > 0 else 0.0)

    savings_pct = (
        100.0 * mean_savings / mean_total_w24
        if mean_total_w24 > 0 else 0.0)

    results: dict[str, Any] = {
        "bank": bank,
        "T_decoder": T_decoder,
        "K_consumers": K_consumers,
        "n_cells": n_cells_run,
        "bank_seed": bank_seed,
        "mean_total_w24_visible_tokens": round(mean_total_w24, 4),
        "mean_total_w25_visible_tokens": round(mean_total_w25, 4),
        "mean_savings_tokens_per_cell": round(mean_savings, 4),
        "savings_pct": round(savings_pct, 2),
        "correctness_ratified_rate": round(correctness_ratified_rate, 4),
        "fanout_consumer_resolved_rate": round(fanout_consumer_resolved_rate, 4),
        "fanout_consumer_rejected_rate": round(fanout_consumer_rejected_rate, 4),
        "n_consumer_resolved": n_consumer_resolved,
        "n_consumer_rejected": n_consumer_rejected,
        "n_consumer_no_trigger": n_consumer_no_trigger,
        "n_fanout_bytes_total": n_fanout_bytes_total,
        "registry_n_registered": reg_registered,
        "registry_n_resolved": reg_resolved,
        "registry_n_rejected_register": reg_rejected_register,
        "registry_n_rejected_resolve": reg_rejected_resolve,
        "use_fanout": use_fanout,
    }

    if verbose:
        print(f"\n=== Phase 72 — R-72-{bank.upper()} ===")
        print(f"bank={bank}, T_decoder={T_decoder}, "
              f"K_consumers={K_consumers}, n_cells={n_cells_run}")
        print(f"mean_total_w24={mean_total_w24:.2f} "
              f"mean_total_w25={mean_total_w25:.2f} "
              f"savings={mean_savings:.2f} ({savings_pct:.1f}%)")
        print(f"correctness_ratified_rate={correctness_ratified_rate:.4f}")
        print(f"fanout_consumer_resolved_rate={fanout_consumer_resolved_rate:.4f}")
        print(f"fanout_consumer_rejected_rate={fanout_consumer_rejected_rate:.4f}")
        print(f"registry registered={reg_registered} resolved={reg_resolved}")
        print(f"n_fanout_bytes_total={n_fanout_bytes_total}")

    return results


def run_phase72_seed_stability_sweep(
        *,
        seeds: tuple[int, ...] = (11, 17, 23, 29, 31),
        bank: str = "fanout_shared",
        T_decoder: int | None = None,
        K_consumers: int = 3,
        n_eval: int = 16,
        bank_replicates: int = 4,
        verbose: bool = False,
) -> dict[str, Any]:
    """Run the same sub-bank across multiple seeds and report stability."""
    rows = []
    for seed in seeds:
        r = run_phase72(
            bank=bank, T_decoder=T_decoder, K_consumers=K_consumers,
            n_eval=n_eval, bank_replicates=bank_replicates,
            bank_seed=seed, verbose=verbose)
        rows.append(r)

    savings_list = [r["mean_savings_tokens_per_cell"] for r in rows]
    correctness_list = [r["correctness_ratified_rate"] for r in rows]

    return {
        "seeds": list(seeds),
        "bank": bank,
        "T_decoder": T_decoder,
        "K_consumers": K_consumers,
        "rows": rows,
        "min_savings": min(savings_list),
        "mean_savings": sum(savings_list) / len(savings_list),
        "all_savings_positive": all(s > 0 for s in savings_list),
        "min_correctness": min(correctness_list),
        "all_correctness_1000": all(c >= 1.0 for c in correctness_list),
    }


def run_cross_regime_p72(
        *,
        K_consumers: int = 3,
        n_eval: int = 16,
        bank_replicates: int = 4,
        bank_seed: int = 11,
        verbose: bool = False,
) -> dict[str, Any]:
    """Run all three R-72 sub-banks and return a summary dict."""
    results = {}
    for bank in ("fanout_shared", "disjoint", "fanout_poisoned"):
        for T_decoder in (None, 24):
            key = f"{bank}_T{T_decoder}"
            results[key] = run_phase72(
                bank=bank, T_decoder=T_decoder,
                K_consumers=K_consumers, n_eval=n_eval,
                bank_replicates=bank_replicates, bank_seed=bank_seed,
                verbose=verbose)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli() -> None:
    ap = argparse.ArgumentParser(
        description="Phase 72 — W25 shared-fanout benchmark")
    ap.add_argument("--bank", default="fanout_shared",
                     choices=["fanout_shared", "disjoint", "fanout_poisoned",
                               "cross_regime"],
                     help="Which sub-bank to run")
    ap.add_argument("--K-consumers", type=int, default=3,
                     help="Number of consumer agents")
    ap.add_argument("--decoder-budget", type=int, default=-1,
                     help="T_decoder (-1 = None)")
    ap.add_argument("--n-eval", type=int, default=16)
    ap.add_argument("--bank-replicates", type=int, default=4)
    ap.add_argument("--bank-seed", type=int, default=11)
    ap.add_argument("--seed-sweep", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", default="-",
                     help="Output JSON path ('-' = stdout)")
    args = ap.parse_args()

    T_decoder: int | None = None if args.decoder_budget < 0 else args.decoder_budget

    if args.bank == "cross_regime":
        result = run_cross_regime_p72(
            K_consumers=args.K_consumers, n_eval=args.n_eval,
            bank_replicates=args.bank_replicates,
            bank_seed=args.bank_seed, verbose=args.verbose)
    elif args.seed_sweep:
        result = run_phase72_seed_stability_sweep(
            bank=args.bank, T_decoder=T_decoder,
            K_consumers=args.K_consumers, n_eval=args.n_eval,
            bank_replicates=args.bank_replicates, verbose=args.verbose)
    else:
        result = run_phase72(
            bank=args.bank, T_decoder=T_decoder,
            K_consumers=args.K_consumers, n_eval=args.n_eval,
            bank_replicates=args.bank_replicates,
            bank_seed=args.bank_seed, verbose=args.verbose)

    out_text = json.dumps(result, indent=2, default=str)
    if args.out == "-":
        print(out_text)
    else:
        with open(args.out, "w") as fh:
            fh.write(out_text)


if __name__ == "__main__":
    _cli()
