"""Phase 58 — multi-round, delayed-causal-evidence decoder benchmark.

SDK v3.12, W11 family anchor.

This phase is the **next harder regime** the SDK v3.12 milestone
introduces. It directly attacks the W10 single-round bundle decoder's
structural limit (W10-Λ at the *temporal* axis) by constructing a
deterministic regime where:

* **Round 1** carries only *generic-tier* (non-causal) evidence
  ``LATENCY_SPIKE`` / ``ERROR_RATE_SPIKE`` / ``FW_BLOCK_SURGE`` on
  every candidate service — gold_A, gold_B, and a decoy. In round 1
  alone:
    - the W10 single-round bundle decoder picks a *generic* root_cause
      (``latency_spike`` or ``error_spike``) from the priority decoder,
    - the chosen CCK is itself generic (``{LATENCY_SPIKE,
      ERROR_RATE_SPIKE}``), so the CCK filter cannot exclude the decoy,
    - every service-blind admission policy ties FIFO at 0.000.
* **Round 2** carries one *specific-tier* disambiguating claim per
  gold service (e.g. ``DEADLOCK_SUSPECTED`` on gold_A,
  ``POOL_EXHAUSTION`` on gold_B) emitted by the ``db_admin`` role.
  The decoy gets nothing in round 2.

W11-Λ (named structural limit)
==============================

Every SDK strategy that decodes round-by-round (substrate, FIFO,
priority, coverage, W7-2, W8, W9, W10 *single-round*) ties FIFO at
0.000 on Phase 58 default — the round-1 admitted bundle is
generic-only, the round-2 admitted bundle is specific-only, and the
round-2 bundle's chosen root_cause picks ``deadlock`` whose CCK
filter, by itself, has nothing of round-1's services to project.

W11-1 (sufficiency)
===================

Pairing W9 admission with the
``MultiRoundBundleDecoder`` (which decodes the *union* of round-1
and round-2 admitted handoffs) achieves ``accuracy_full = 1.000`` on
Phase 58 default — the union promotes the chosen root_cause to
``deadlock`` (a specific-tier label) AND retains round-1's full
service-tag inventory, so the CCK filter can exclude the decoy.

W11-4 (named falsifier)
=======================

If round-1 fills the auditor's per-round budget so that round-2's
admission is forced to drop the disambiguating specific-tier claim,
the union is still all-generic and W11 cannot help. The Phase-58
falsifier bank instantiates this regime by setting ``K_auditor``
small enough that round-1's noise floods the budget.

CLI
---

::

    # Default Phase-58 default config (W11-1 anchor):
    python3 -m vision_mvp.experiments.phase58_multi_round_decoder \\
        --K-auditor 8 --n-eval 8 --out -

    # Falsifier (W11-4 anchor):
    python3 -m vision_mvp.experiments.phase58_multi_round_decoder \\
        --falsifier --K-auditor 4 --n-eval 8 --out -

    # Cross-regime (R-54..R-58 default at K_auditor=4):
    python3 -m vision_mvp.experiments.phase58_multi_round_decoder \\
        --cross-regime --n-eval 8 --out -
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import random
import re
import sys
from typing import Any, Sequence

from vision_mvp.tasks.incident_triage import (
    ALL_ROLES, ROLE_AUDITOR, ROLE_DB_ADMIN, ROLE_MONITOR,
    ROLE_NETWORK, ROLE_SYSADMIN,
    IncidentScenario,
    build_role_subscriptions, grade_answer,
    _decoder_from_handoffs as _phase31_decoder_from_handoffs,
)
from vision_mvp.wevra.capsule import CapsuleKind, CapsuleLedger
from vision_mvp.wevra.team_coord import (
    AdmissionPolicy, BundleAwareTeamDecoder,
    ClaimPriorityAdmissionPolicy,
    CohortCoherenceAdmissionPolicy,
    CoverageGuidedAdmissionPolicy,
    CrossRoleCorroborationAdmissionPolicy,
    FifoAdmissionPolicy, MultiRoundBundleDecoder,
    MultiServiceCorroborationAdmissionPolicy,
    RoleBudget, TeamCoordinator, audit_team_lifecycle,
    collect_admitted_handoffs, _DecodedHandoff,
)
from vision_mvp.experiments.phase52_team_coord import (
    StrategyResult, _format_canonical_answer,
    claim_priorities, make_team_budgets, pool,
)


# =============================================================================
# Phase 58 scenario family
# =============================================================================


@dataclasses.dataclass(frozen=True)
class MultiRoundScenario:
    """One Phase-58 scenario.

    Round 1 emissions are exclusively generic-tier
    (LATENCY_SPIKE/ERROR_RATE_SPIKE/FW_BLOCK_SURGE) on
    {gold_A, gold_B, decoy}. Round 2 emissions are exclusively
    specific-tier on {gold_A, gold_B} only.
    """

    scenario_id: str
    description: str
    gold_services_pair: tuple[str, str]
    decoy_storm_service: str
    gold_root_cause: str
    gold_remediation: str
    round1_emissions: dict[str, tuple[tuple[str, str], ...]]
    round2_emissions: dict[str, tuple[tuple[str, str], ...]]


_REMEDIATION = {
    "deadlock":             "enforce_lock_ordering_in_orders",
    "pool_exhaustion":      "raise_pool_cap_or_fix_upstream",
    "slow_query_cascade":   "index_or_split_slow_query",
    "disk_fill":            "rotate_logs_and_clear_backup",
}


def _emit(kind: str, payload: str) -> tuple[str, str]:
    return (kind, payload)


def _build_deadlock(decoy: str = "cache") -> MultiRoundScenario:
    A, B = "orders", "payments"
    return MultiRoundScenario(
        scenario_id=f"deadlock_orders_payments__delayed_{decoy}",
        description=(
            f"Round-1: gold {A}/{B} mentioned via single-role generic "
            f"noise; {decoy} corroborated via cross-role generic noise. "
            f"Round-2: db_admin emits DEADLOCK_SUSPECTED with relation "
            f"tag (no service= token); decoy gets nothing."),
        gold_services_pair=(A, B),
        decoy_storm_service=decoy,
        gold_root_cause="deadlock",
        gold_remediation=_REMEDIATION["deadlock"],
        round1_emissions={
            ROLE_MONITOR: (
                _emit("LATENCY_SPIKE", f"p95_ms=2100 service={A}"),
                _emit("ERROR_RATE_SPIKE", f"error_rate=0.22 service={B}"),
                _emit("LATENCY_SPIKE", f"p95_ms=180 service={decoy}"),
                _emit("ERROR_RATE_SPIKE", f"error_rate=0.05 service={decoy}"),
            ),
            ROLE_NETWORK: (
                _emit("FW_BLOCK_SURGE", f"rule=deny count=10 service={decoy}"),
                _emit("FW_BLOCK_SURGE", f"rule=deny count=11 service={decoy}"),
            ),
            ROLE_DB_ADMIN: (),
            ROLE_SYSADMIN: (),
        },
        round2_emissions={
            ROLE_MONITOR: (),
            ROLE_NETWORK: (),
            ROLE_DB_ADMIN: (
                _emit("DEADLOCK_SUSPECTED",
                       f"deadlock relation={A}_{B}_join wait_chain=2"),
            ),
            ROLE_SYSADMIN: (),
        },
    )


def _build_pool(decoy: str = "cache") -> MultiRoundScenario:
    A, B = "api", "db"
    return MultiRoundScenario(
        scenario_id=f"pool_api_db__delayed_{decoy}",
        description=(
            f"Round-1: gold {A}/{B} mentioned via single-role generic "
            f"noise; {decoy} corroborated via cross-role generic noise. "
            f"Round-2: db_admin emits POOL_EXHAUSTION (no service= token); "
            f"decoy gets nothing."),
        gold_services_pair=(A, B),
        decoy_storm_service=decoy,
        gold_root_cause="pool_exhaustion",
        gold_remediation=_REMEDIATION["pool_exhaustion"],
        round1_emissions={
            ROLE_MONITOR: (
                _emit("LATENCY_SPIKE", f"p95_ms=4200 service={A}"),
                _emit("ERROR_RATE_SPIKE", f"error_rate=0.18 service={B}"),
                _emit("LATENCY_SPIKE", f"p95_ms=210 service={decoy}"),
                _emit("ERROR_RATE_SPIKE", f"error_rate=0.04 service={decoy}"),
            ),
            ROLE_NETWORK: (
                _emit("FW_BLOCK_SURGE", f"rule=deny count=12 service={decoy}"),
                _emit("FW_BLOCK_SURGE", f"rule=deny count=14 service={decoy}"),
            ),
            ROLE_DB_ADMIN: (),
            ROLE_SYSADMIN: (),
        },
        round2_emissions={
            ROLE_MONITOR: (),
            ROLE_NETWORK: (),
            ROLE_DB_ADMIN: (
                _emit("POOL_EXHAUSTION",
                       "pool active=200/200 waiters=145 cluster=primary"),
            ),
            ROLE_SYSADMIN: (),
        },
    )


def _build_disk(decoy: str = "archival") -> MultiRoundScenario:
    A, B = "storage", "logs_pipeline"
    return MultiRoundScenario(
        scenario_id=f"disk_storage_logs__delayed_{decoy}",
        description=(
            f"Round-1: gold {A}/{B} mentioned via single-role generic "
            f"noise; {decoy} corroborated via cross-role generic noise. "
            f"Round-2: sysadmin emits DISK_FILL_CRITICAL on the host "
            f"(no service= token); decoy gets nothing."),
        gold_services_pair=(A, B),
        decoy_storm_service=decoy,
        gold_root_cause="disk_fill",
        gold_remediation=_REMEDIATION["disk_fill"],
        round1_emissions={
            ROLE_MONITOR: (
                _emit("ERROR_RATE_SPIKE", f"error_rate=0.41 service={A}"),
                _emit("LATENCY_SPIKE", f"p95_ms=4500 service={B}"),
                _emit("LATENCY_SPIKE", f"p95_ms=200 service={decoy}"),
                _emit("ERROR_RATE_SPIKE", f"error_rate=0.06 service={decoy}"),
            ),
            ROLE_NETWORK: (
                _emit("FW_BLOCK_SURGE", f"rule=deny count=7 service={decoy}"),
                _emit("FW_BLOCK_SURGE", f"rule=deny count=8 service={decoy}"),
            ),
            ROLE_DB_ADMIN: (),
            ROLE_SYSADMIN: (),
        },
        round2_emissions={
            ROLE_MONITOR: (),
            ROLE_NETWORK: (),
            ROLE_SYSADMIN: (
                _emit("DISK_FILL_CRITICAL",
                       "/var/log used=99% fs=/ host=primary"),
            ),
            ROLE_DB_ADMIN: (),
        },
    )


def _build_slow_query(decoy: str = "metrics") -> MultiRoundScenario:
    A, B = "web", "db"
    return MultiRoundScenario(
        scenario_id=f"slow_query_web_db__delayed_{decoy}",
        description=(
            f"Round-1: gold {A}/{B} mentioned via single-role generic "
            f"noise; {decoy} corroborated via cross-role generic noise. "
            f"Round-2: db_admin emits SLOW_QUERY_OBSERVED (no service= "
            f"token); decoy gets nothing."),
        gold_services_pair=(A, B),
        decoy_storm_service=decoy,
        gold_root_cause="slow_query_cascade",
        gold_remediation=_REMEDIATION["slow_query_cascade"],
        round1_emissions={
            ROLE_MONITOR: (
                _emit("LATENCY_SPIKE", f"p95_ms=4100 service={A}"),
                _emit("ERROR_RATE_SPIKE", f"error_rate=0.15 service={B}"),
                _emit("LATENCY_SPIKE", f"p95_ms=200 service={decoy}"),
                _emit("ERROR_RATE_SPIKE", f"error_rate=0.05 service={decoy}"),
            ),
            ROLE_NETWORK: (
                _emit("FW_BLOCK_SURGE", f"rule=deny count=4 service={decoy}"),
                _emit("FW_BLOCK_SURGE", f"rule=deny count=5 service={decoy}"),
            ),
            ROLE_DB_ADMIN: (),
            ROLE_SYSADMIN: (),
        },
        round2_emissions={
            ROLE_MONITOR: (),
            ROLE_NETWORK: (),
            ROLE_DB_ADMIN: (
                _emit("SLOW_QUERY_OBSERVED",
                       "q#9 mean_ms=4100 cluster=primary"),
            ),
            ROLE_SYSADMIN: (),
        },
    )


_BASE_BUILDERS = (
    _build_deadlock, _build_pool, _build_disk, _build_slow_query,
)

_KNOWN_DECOYS = (
    "cache", "archival", "metrics", "telemetry", "audit_jobs",
    "sessions", "search_index", "scratch_pool",
)


def build_phase58_bank(*, n_replicates: int = 2,
                          seed: int = 11) -> list[MultiRoundScenario]:
    rng = random.Random(seed)
    out: list[MultiRoundScenario] = []
    for builder in _BASE_BUILDERS:
        for r in range(n_replicates):
            i = rng.randrange(0, 1 << 16)
            chosen = _KNOWN_DECOYS[(i + r) % len(_KNOWN_DECOYS)]
            sc = builder(decoy=chosen)
            out.append(dataclasses.replace(
                sc, scenario_id=f"{sc.scenario_id}__rep{r}"))
    return out


def build_phase58_falsifier_bank(*, n_replicates: int = 2,
                                    seed: int = 11
                                    ) -> list[MultiRoundScenario]:
    """W11-4 falsifier bank: round-1 noise count is doubled so a small
    K_auditor (default 4) is fully consumed in round 1 and round-2's
    specific-tier claims are admission-dropped.
    """
    base = build_phase58_bank(n_replicates=n_replicates, seed=seed)
    out: list[MultiRoundScenario] = []
    for sc in base:
        r1 = dict(sc.round1_emissions)
        for role in (ROLE_MONITOR, ROLE_NETWORK):
            existing = list(r1.get(role, ()))
            extras: list[tuple[str, str]] = []
            for (kind, payload) in existing:
                # duplicate every noise emission with a small payload
                # bump so the CID is distinct.
                extras.append(_emit(kind, payload + " variant=2"))
                extras.append(_emit(kind, payload + " variant=3"))
            r1[role] = tuple(existing + extras)
        out.append(dataclasses.replace(
            sc, round1_emissions=r1,
            scenario_id=f"{sc.scenario_id}__falsifier"))
    return out


# =============================================================================
# IncidentScenario adapter
# =============================================================================


def _as_incident_scenario(sc: MultiRoundScenario) -> IncidentScenario:
    chain: list[tuple[str, str, str, tuple[int, ...]]] = []
    A, B = sc.gold_services_pair
    for round_emissions in (sc.round1_emissions, sc.round2_emissions):
        for role, emissions in round_emissions.items():
            for (kind, payload) in emissions:
                if (f"service={A}" in payload
                        or f"service={B}" in payload):
                    chain.append((role, kind, payload, (0,)))
    return IncidentScenario(
        scenario_id=sc.scenario_id,
        description=sc.description,
        gold_root_cause=sc.gold_root_cause,
        gold_services=tuple(sorted(sc.gold_services_pair)),
        gold_remediation=sc.gold_remediation,
        causal_chain=tuple(chain),
        per_role_events={r: () for r in ALL_ROLES},
    )


def _build_round_candidates(
        emissions: dict[str, tuple[tuple[str, str], ...]],
        ) -> list[tuple[str, str, str, str, tuple[int, ...]]]:
    subs = build_role_subscriptions()
    out: list[tuple[str, str, str, str, tuple[int, ...]]] = []
    for role in (ROLE_MONITOR, ROLE_DB_ADMIN, ROLE_SYSADMIN, ROLE_NETWORK):
        for (kind, payload) in emissions.get(role, ()):
            consumers = subs.consumers(role, kind)
            if not consumers:
                continue
            for to_role in sorted(consumers):
                out.append((role, to_role, kind, payload, (0,)))
    return out


# =============================================================================
# Per-strategy run drivers
# =============================================================================


def _run_capsule_strategy(
        sc: MultiRoundScenario,
        budgets: dict[str, RoleBudget],
        policy_per_role_factory,
        strategy_name: str,
        *,
        decoder_mode: str,
        ) -> StrategyResult:
    """Run two coordination rounds of the team capsule layer.

    decoder_mode:
      * ``"per_round"`` — substrate decoder is run on round-2's
        admitted handoffs only (legacy behaviour).
      * ``"single_round_bundle"`` — W10 bundle decoder run on round-2
        admitted handoffs only (W11-Λ witness for W10).
      * ``"multi_round_bundle"`` — W11 multi-round bundle decoder
        decodes the union of round-1 and round-2 admitted handoffs.
    """
    incident_sc = _as_incident_scenario(sc)
    ledger = CapsuleLedger()
    coord = TeamCoordinator(
        ledger=ledger, role_budgets=budgets,
        policy_per_role=policy_per_role_factory(),
        team_tag="multi_round_phase58",
    )

    # ---- round 1 -------------------------------------------------------
    coord.advance_round(1)
    cands_r1 = _build_round_candidates(sc.round1_emissions)
    for (src, to, kind, payload, _evs) in cands_r1:
        coord.emit_handoff(
            source_role=src, to_role=to, claim_kind=kind, payload=payload)
    coord.seal_all_role_views()
    rv1 = coord.role_view_cid(ROLE_AUDITOR)

    # ---- round 2 -------------------------------------------------------
    coord.advance_round(1)
    # Reset auditor admission state via a fresh policy_per_role each
    # round if the policy is buffered (cohort/corroboration/multi-
    # service rely on a fitted dominant set; we re-fit per round on
    # the round's candidate stream).
    cands_r2 = _build_round_candidates(sc.round2_emissions)
    coord.policy_per_role = policy_per_role_factory(round_idx=2,
                                                       cands=cands_r2)
    for (src, to, kind, payload, _evs) in cands_r2:
        coord.emit_handoff(
            source_role=src, to_role=to, claim_kind=kind, payload=payload)
    coord.seal_all_role_views()
    rv2 = coord.role_view_cid(ROLE_AUDITOR)

    # ---- decode --------------------------------------------------------
    if decoder_mode == "multi_round_bundle":
        union = collect_admitted_handoffs(ledger, [rv1, rv2])
        decoder = MultiRoundBundleDecoder()
        answer = decoder.decode_rounds([union])
    elif decoder_mode == "single_round_bundle":
        # Decode round-2 admitted handoffs only.
        r2_handoffs = collect_admitted_handoffs(ledger, [rv2])
        decoder = BundleAwareTeamDecoder(
            cck_filter=True, role_corroboration_floor=1,
            fallback_admitted_size_threshold=2)
        answer = decoder.decode(r2_handoffs)
    else:  # "per_round" — substrate-style priority decoder on round-2
        r2_handoffs = collect_admitted_handoffs(ledger, [rv2])
        # Use a tiny shim to feed the priority decoder.
        @dataclasses.dataclass(frozen=True)
        class _Shim:
            source_role: str
            claim_kind: str
            payload: str
            n_tokens: int = 1
        shimmed = [_Shim(h.source_role, h.claim_kind, h.payload)
                    for h in r2_handoffs]
        answer = _phase31_decoder_from_handoffs(shimmed)

    coord.seal_team_decision(
        team_role=ROLE_AUDITOR, decision=answer,
        extra_role_view_cids=[rv1] if rv1 and rv1 != rv2 else ())
    audit = audit_team_lifecycle(ledger)
    grading = grade_answer(incident_sc, _format_canonical_answer(answer))

    # Accounting on round-2 RV (or the union-equivalent).
    rv2_cap = ledger.get(rv2) if rv2 else None
    n_admitted_r2 = (rv2_cap.payload.get("n_admitted")
                       if rv2_cap is not None
                       and isinstance(rv2_cap.payload, dict) else 0)
    n_tokens_r2 = (rv2_cap.payload.get("n_tokens_admitted", 0)
                     if rv2_cap is not None
                     and isinstance(rv2_cap.payload, dict) else 0)
    admitted_kinds: set[tuple[str, str]] = set()
    for cid in (rv1, rv2):
        if not cid or cid not in ledger:
            continue
        cap = ledger.get(cid)
        for p in cap.parents:
            if p in ledger:
                handoff = ledger.get(p)
                if handoff.kind != CapsuleKind.TEAM_HANDOFF:
                    continue
                payload = (handoff.payload
                            if isinstance(handoff.payload, dict) else {})
                admitted_kinds.add((str(payload.get("source_role", "")),
                                     str(payload.get("claim_kind", ""))))
    required = {(role, kind)
                 for (role, kind, _p, _evs) in incident_sc.causal_chain}
    if grading["full_correct"]:
        failure_kind = "none"
    elif required - admitted_kinds:
        failure_kind = "missing_handoff"
    else:
        failure_kind = "decoder_error"
    return StrategyResult(
        strategy=strategy_name,
        scenario_id=sc.scenario_id,
        answer=answer,
        grading=grading,
        failure_kind=failure_kind,
        n_admitted_auditor=int(n_admitted_r2 or 0),
        n_dropped_auditor_budget=0,
        n_dropped_auditor_capacity=0,
        n_dropped_auditor_unknown_kind=0,
        n_team_handoff=coord.stats()["n_team_handoff"],
        n_role_view=coord.stats()["n_role_view"],
        n_team_decision=coord.stats()["n_team_decision"],
        audit_ok=audit.is_ok(),
        n_tokens_admitted=int(n_tokens_r2 or 0),
    )


def _run_substrate_strategy(
        sc: MultiRoundScenario,
        inbox_capacity: int,
        ) -> StrategyResult:
    """Substrate baseline: ROUTE round-1 + round-2 candidates through
    the substrate inbox-router and decode round-2 holdings only —
    same shape as Phase 57's substrate baseline (no cross-round
    union). Round-1 noise pre-fills the auditor's inbox up to
    ``inbox_capacity``; round-2 specific-tier claims arrive after
    and may be dropped if the inbox is full (FIFO drop-tail).
    """
    incident_sc = _as_incident_scenario(sc)
    from vision_mvp.core.role_handoff import HandoffRouter, RoleInbox
    subs = build_role_subscriptions()
    router = HandoffRouter(subs=subs)
    for role in ALL_ROLES:
        router.register_inbox(RoleInbox(role=role, capacity=inbox_capacity))
    for round_idx, emissions in (
            (1, sc.round1_emissions), (2, sc.round2_emissions)):
        cands = _build_round_candidates(emissions)
        for (src, _to, kind, payload, evids) in cands:
            router.emit(
                source_role=src,
                source_agent_id=ALL_ROLES.index(src),
                claim_kind=kind, payload=payload,
                source_event_ids=evids, round=round_idx)
    auditor_inbox = router.inboxes.get(ROLE_AUDITOR)
    held = tuple(auditor_inbox.peek()) if auditor_inbox else ()
    answer = _phase31_decoder_from_handoffs(held)
    grading = grade_answer(incident_sc, _format_canonical_answer(answer))
    admitted_kinds = {(h.source_role, h.claim_kind) for h in held}
    required = {(role, kind)
                 for (role, kind, _p, _evs) in incident_sc.causal_chain}
    if grading["full_correct"]:
        failure_kind = "none"
    elif required - admitted_kinds:
        failure_kind = "missing_handoff"
    else:
        failure_kind = "decoder_error"
    return StrategyResult(
        strategy="substrate",
        scenario_id=sc.scenario_id,
        answer=answer,
        grading=grading,
        failure_kind=failure_kind,
        n_admitted_auditor=len(held),
        n_dropped_auditor_budget=auditor_inbox.n_overflow if auditor_inbox else 0,
        n_dropped_auditor_capacity=auditor_inbox.n_dedup if auditor_inbox else 0,
        n_dropped_auditor_unknown_kind=0,
        n_team_handoff=0, n_role_view=0, n_team_decision=0,
        audit_ok=False,
        n_tokens_admitted=sum(h.n_tokens for h in held),
    )


# =============================================================================
# Bench-property metrics
# =============================================================================


_SERVICE_TAG_RE = re.compile(r"service=(\w+)")
_GENERIC_NOISE_KINDS = frozenset({
    "LATENCY_SPIKE", "ERROR_RATE_SPIKE", "FW_BLOCK_SURGE",
})


def _bench_property(sc: MultiRoundScenario) -> dict[str, Any]:
    """Pre-committed mechanical witnesses of the Phase-58 bench
    property:

      * ``round1_only_generic_noise`` — every round-1 emission has
        ``claim_kind`` in the generic-noise set.
      * ``round2_only_specific`` — every round-2 emission has
        ``claim_kind`` NOT in the generic-noise set.
      * ``decoy_only_in_round1`` — decoy is only mentioned in round 1.
      * ``round1_decoy_corroborated`` — decoy in round 1 is mentioned
        by ≥ 2 distinct producer roles.
    """
    cands_r1 = _build_round_candidates(sc.round1_emissions)
    cands_r2 = _build_round_candidates(sc.round2_emissions)
    r1_kinds = {kind for (_s, _t, kind, _p, _e) in cands_r1
                  if _t == ROLE_AUDITOR}
    r2_kinds = {kind for (_s, _t, kind, _p, _e) in cands_r2
                  if _t == ROLE_AUDITOR}
    round1_only_generic = bool(r1_kinds) and r1_kinds.issubset(
        _GENERIC_NOISE_KINDS)
    round2_only_specific = bool(r2_kinds) and not (
        r2_kinds & _GENERIC_NOISE_KINDS)
    decoy = sc.decoy_storm_service
    r2_mentions_decoy = any(
        f"service={decoy}" in payload
        for (_s, _t, _k, payload, _e) in cands_r2)
    r1_decoy_roles: set[str] = set()
    for (src, to, kind, payload, _e) in cands_r1:
        if to == ROLE_AUDITOR and f"service={decoy}" in payload:
            r1_decoy_roles.add(src)
    return {
        "round1_only_generic_noise": round1_only_generic,
        "round2_only_specific": round2_only_specific,
        "decoy_only_in_round1": (not r2_mentions_decoy),
        "round1_decoy_corroborated": len(r1_decoy_roles) >= 2,
        "delayed_causal_evidence_property_holds": (
            round1_only_generic and round2_only_specific
            and (not r2_mentions_decoy)
            and len(r1_decoy_roles) >= 2),
        "n_round1_to_auditor": sum(
            1 for c in cands_r1 if c[1] == ROLE_AUDITOR),
        "n_round2_to_auditor": sum(
            1 for c in cands_r2 if c[1] == ROLE_AUDITOR),
    }


# =============================================================================
# Top-level driver
# =============================================================================


def _make_factory(name: str, priorities, budgets):
    """Return a ``policy_per_role_factory(round_idx, cands) -> dict``
    for the given strategy. The factory takes optional kwargs so
    buffered policies can re-fit on the round's candidates.
    """
    def fac(round_idx: int = 1,
             cands: Sequence[tuple[str, str, str, str, tuple[int, ...]]] | None = None
             ) -> dict[str, AdmissionPolicy]:
        if name == "capsule_fifo":
            return {r: FifoAdmissionPolicy() for r in budgets}
        if name == "capsule_priority":
            return {r: ClaimPriorityAdmissionPolicy(
                priorities=priorities, threshold=0.65) for r in budgets}
        if name == "capsule_coverage":
            return {r: CoverageGuidedAdmissionPolicy() for r in budgets}
        cands = cands or []
        cands_aud = [c for c in cands if c[1] == ROLE_AUDITOR]
        if name == "capsule_cohort_buffered":
            policy = (CohortCoherenceAdmissionPolicy
                      .from_candidate_payloads([c[3] for c in cands_aud]))
            return {r: policy for r in budgets}
        if name == "capsule_corroboration":
            policy = (CrossRoleCorroborationAdmissionPolicy
                      .from_candidate_stream(
                          [(c[0], c[3]) for c in cands_aud]))
            return {r: policy for r in budgets}
        if name in ("capsule_multi_service",
                     "capsule_bundle_decoder"):
            policy = (MultiServiceCorroborationAdmissionPolicy
                      .from_candidate_stream(
                          [(c[0], c[3]) for c in cands_aud],
                          top_k=3, min_corroborated_roles=2))
            return {r: policy for r in budgets}
        if name == "capsule_multi_round":
            # The W11 method is a *decoder-side* move; admission is
            # FIFO so the decoder receives the complete admitted set
            # (round-1 services + round-2 specific claim).
            return {r: FifoAdmissionPolicy() for r in budgets}
        raise ValueError(f"unknown strategy {name!r}")
    return fac


def run_phase58(*,
                  n_eval: int | None = None,
                  K_auditor: int = 8,
                  T_auditor: int = 256,
                  K_producer: int = 6,
                  T_producer: int = 96,
                  inbox_capacity: int | None = None,
                  bank_seed: int = 11,
                  bank_replicates: int = 2,
                  use_falsifier_bank: bool = False,
                  verbose: bool = False,
                  ) -> dict[str, Any]:
    """Run Phase 58 end-to-end and return a JSON-serialisable report.

    Pre-committed default: ``K_auditor=8, T_auditor=256, n_eval=8,
    bank_seed=11, bank_replicates=2``.
    """
    if use_falsifier_bank:
        bank = build_phase58_falsifier_bank(
            n_replicates=bank_replicates, seed=bank_seed)
    else:
        bank = build_phase58_bank(
            n_replicates=bank_replicates, seed=bank_seed)
    if n_eval is not None:
        bank = bank[:n_eval]
    if inbox_capacity is None:
        inbox_capacity = K_auditor

    budgets = make_team_budgets(
        K_producer=K_producer, T_producer=T_producer,
        K_auditor=K_auditor, T_auditor=T_auditor)
    priorities = claim_priorities()

    strategies: list[tuple[str, str]] = [
        # (strategy_name, decoder_mode)
        ("capsule_fifo", "per_round"),
        ("capsule_priority", "per_round"),
        ("capsule_coverage", "per_round"),
        ("capsule_cohort_buffered", "per_round"),
        ("capsule_corroboration", "per_round"),
        ("capsule_multi_service", "per_round"),
        ("capsule_bundle_decoder", "single_round_bundle"),
        ("capsule_multi_round", "multi_round_bundle"),
    ]

    results: list[StrategyResult] = []
    bench_property: dict[str, dict[str, Any]] = {}
    for sc in bank:
        bench_property[sc.scenario_id] = _bench_property(sc)
        results.append(_run_substrate_strategy(sc, inbox_capacity))
        for (sname, dmode) in strategies:
            fac = _make_factory(sname, priorities, budgets)
            results.append(_run_capsule_strategy(
                sc=sc, budgets=budgets, policy_per_role_factory=fac,
                strategy_name=sname, decoder_mode=dmode))

    strategy_names = ("substrate",) + tuple(s[0] for s in strategies)
    pooled = {s: pool(results, s).as_dict() for s in strategy_names}

    def gap(a: str, b: str) -> float:
        return round(pooled[a]["accuracy_full"]
                      - pooled[b]["accuracy_full"], 4)

    headline_gap = {
        "multi_round_minus_fifo_accuracy_full":
            gap("capsule_multi_round", "capsule_fifo"),
        "multi_round_minus_substrate_accuracy_full":
            gap("capsule_multi_round", "substrate"),
        "multi_round_minus_bundle_decoder_accuracy_full":
            gap("capsule_multi_round", "capsule_bundle_decoder"),
        "multi_round_minus_multi_service_accuracy_full":
            gap("capsule_multi_round", "capsule_multi_service"),
        "max_single_round_accuracy_full": max(
            pooled[s]["accuracy_full"]
            for s in strategy_names if s != "capsule_multi_round"),
    }

    audit_ok_grid = {}
    for s in strategy_names:
        if s == "substrate":
            audit_ok_grid[s] = False
            continue
        rs = [r for r in results if r.strategy == s]
        audit_ok_grid[s] = bool(rs) and all(r.audit_ok for r in rs)

    bench_summary = {
        "scenarios_with_property": sum(
            1 for v in bench_property.values()
            if v["delayed_causal_evidence_property_holds"]),
        "n_scenarios": len(bench_property),
        "K_auditor": K_auditor,
    }

    if verbose:
        print(f"[phase58] n_eval={len(bank)}, K_auditor={K_auditor}, "
              f"falsifier={use_falsifier_bank}",
              file=sys.stderr, flush=True)
        print(f"[phase58] property holds in "
              f"{bench_summary['scenarios_with_property']}/{len(bank)}",
              file=sys.stderr, flush=True)
        for s in strategy_names:
            p = pooled[s]
            print(f"[phase58]   {s:30s} "
                  f"acc_full={p['accuracy_full']:.3f} "
                  f"acc_root_cause={p['accuracy_root_cause']:.3f} "
                  f"acc_services={p['accuracy_services']:.3f}",
                  file=sys.stderr, flush=True)
        print(f"[phase58] gap multi_round−fifo: "
              f"{headline_gap['multi_round_minus_fifo_accuracy_full']:+.3f}",
              file=sys.stderr, flush=True)
        print(f"[phase58] max non-multi-round acc_full = "
              f"{headline_gap['max_single_round_accuracy_full']:.3f} "
              f"(W11-Λ witness)",
              file=sys.stderr, flush=True)

    return {
        "schema": "phase58.multi_round.v1",
        "config": {
            "n_eval": len(bank), "K_auditor": K_auditor,
            "T_auditor": T_auditor, "K_producer": K_producer,
            "T_producer": T_producer, "inbox_capacity": inbox_capacity,
            "bank_seed": bank_seed, "bank_replicates": bank_replicates,
            "use_falsifier_bank": use_falsifier_bank,
        },
        "pooled": pooled,
        "audit_ok_grid": audit_ok_grid,
        "bench_summary": bench_summary,
        "bench_property_per_scenario": bench_property,
        "headline_gap": headline_gap,
        "scenarios_evaluated": [sc.scenario_id for sc in bank],
        "n_results": len(results),
    }


def run_phase58_seed_stability_sweep(
        *, seeds: Sequence[int] = (11, 17, 23, 29, 31),
        n_eval: int = 8, K_auditor: int = 8, T_auditor: int = 256,
        ) -> dict[str, Any]:
    per_seed: dict[int, dict[str, Any]] = {}
    for seed in seeds:
        rep = run_phase58(
            n_eval=n_eval, K_auditor=K_auditor, T_auditor=T_auditor,
            bank_seed=seed, bank_replicates=2, verbose=False)
        per_seed[seed] = {
            "headline_gap": rep["headline_gap"],
            "pooled": rep["pooled"],
            "audit_ok_grid": rep["audit_ok_grid"],
            "bench_summary": rep["bench_summary"],
        }
    return {
        "schema": "phase58.multi_round_seed_sweep.v1",
        "seeds": list(seeds),
        "K_auditor": K_auditor, "T_auditor": T_auditor,
        "n_eval": n_eval, "per_seed": per_seed,
    }


def run_cross_regime_summary(*, n_eval: int = 8, bank_seed: int = 11,
                                bank_replicates: int = 2,
                                ) -> dict[str, Any]:
    from vision_mvp.experiments.phase54_cross_role_coherence import (
        run_phase54)
    from vision_mvp.experiments.phase55_decoy_plurality import (
        run_phase55)
    from vision_mvp.experiments.phase56_multi_service_corroboration import (
        run_phase56)
    from vision_mvp.experiments.phase57_decoder_forcing import (
        run_phase57)
    p54 = run_phase54(n_eval=n_eval, K_auditor=4, T_auditor=128,
                       bank_seed=bank_seed,
                       bank_replicates=bank_replicates, verbose=False)
    p55 = run_phase55(n_eval=n_eval, K_auditor=4, T_auditor=128,
                       bank_seed=bank_seed,
                       bank_replicates=bank_replicates,
                       use_falsifier_bank=False, verbose=False)
    p56 = run_phase56(n_eval=n_eval, K_auditor=4, T_auditor=128,
                       bank_seed=bank_seed,
                       bank_replicates=bank_replicates,
                       use_falsifier_bank=False, verbose=False)
    p57 = run_phase57(n_eval=n_eval, K_auditor=8, T_auditor=256,
                       bank_seed=bank_seed, bank_replicates=3,
                       use_falsifier_bank=False, verbose=False)
    p58 = run_phase58(n_eval=n_eval, K_auditor=8, T_auditor=256,
                       bank_seed=bank_seed,
                       bank_replicates=bank_replicates, verbose=False)
    p58_fal = run_phase58(n_eval=n_eval, K_auditor=4, T_auditor=128,
                            bank_seed=bank_seed,
                            bank_replicates=bank_replicates,
                            use_falsifier_bank=True, verbose=False)
    return {
        "schema": "phase58.cross_regime.v1",
        "config": {"n_eval": n_eval, "bank_seed": bank_seed,
                    "bank_replicates": bank_replicates},
        "phase54_default": p54,
        "phase55_default": p55,
        "phase56_default": p56,
        "phase57_default": p57,
        "phase58_default": p58,
        "phase58_falsifier": p58_fal,
    }


# =============================================================================
# CLI
# =============================================================================


def _arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Phase 58 — multi-round delayed-causal-evidence "
                    "decoder benchmark (SDK v3.12 / W11 family).")
    p.add_argument("--K-auditor", type=int, default=8)
    p.add_argument("--T-auditor", type=int, default=256)
    p.add_argument("--n-eval", type=int, default=8)
    p.add_argument("--bank-seed", type=int, default=11)
    p.add_argument("--bank-replicates", type=int, default=2)
    p.add_argument("--falsifier", action="store_true")
    p.add_argument("--cross-regime", action="store_true")
    p.add_argument("--seed-sweep", action="store_true")
    p.add_argument("--out", type=str, default="")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _arg_parser().parse_args(argv)
    if args.cross_regime:
        report = run_cross_regime_summary(
            n_eval=args.n_eval, bank_seed=args.bank_seed,
            bank_replicates=args.bank_replicates)
    elif args.seed_sweep:
        report = run_phase58_seed_stability_sweep(
            n_eval=args.n_eval, K_auditor=args.K_auditor,
            T_auditor=args.T_auditor)
    else:
        report = run_phase58(
            n_eval=args.n_eval, K_auditor=args.K_auditor,
            T_auditor=args.T_auditor, bank_seed=args.bank_seed,
            bank_replicates=args.bank_replicates,
            use_falsifier_bank=args.falsifier,
            verbose=not args.quiet,
        )
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out == "-":
        print(text)
    elif args.out:
        d = os.path.dirname(os.path.abspath(args.out))
        if d:
            os.makedirs(d, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
        if not args.quiet:
            print(f"[phase58] wrote {args.out}", file=sys.stderr)
    else:
        if not args.quiet:
            print(json.dumps(report.get("pooled", report),
                              indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
