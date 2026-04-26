"""Phase 55 — decoy-plurality + cross-role-corroborated multi-agent
benchmark (SDK v3.9, W8 family anchor).

This phase is the **harder, fairer regime** the SDK v3.9 milestone
introduces. It directly attacks the W7-2 falsifier:

* SDK v3.8 W7-2 (buffered cohort) wins on Phase 54 because the
  candidate stream has the *gold-plurality property* (gold service
  carries strictly more raw mentions than any decoy).
* But on a stream where some decoy service has strictly more *raw
  mentions* than gold, W7-2 picks the decoy plurality and ties
  FIFO at 0.000 (this is the named W7-2 falsifier in
  ``HOW_NOT_TO_OVERSTATE.md``).
* Phase 55 instantiates exactly that falsifier — and at the same
  time arranges the gold service to be *cross-role corroborated*:
  the gold service is mentioned by **strictly more distinct
  producer roles** than any decoy service.

A policy that aggregates over (role, tag) — the
``CrossRoleCorroborationAdmissionPolicy`` shipped in this SDK — can
exploit the corroboration signal that W7-2 cannot see.

Why this is "fair", not benchmark-shopping
==========================================

* **Realistic distractors.** Each producer role sees telemetry
  from many services in the wild; one role can plausibly emit
  many mentions of one decoy service (e.g. an archival cron storm
  hitting only the sysadmin role).
* **Cross-role evidence is the natural signal.** When multiple
  independent roles all mention the same service, the chance that
  it is causal is materially higher than when a single role
  mentions it many times. This is *not* a contrived fingerprint;
  it mirrors how a human SRE would correlate signals across
  monitoring dashboards.
* **Locally-misleading distractors are real.** The harder property
  is that *each role's local plurality of `service=<tag>` mentions
  is a decoy* — i.e. if the auditor naively read the role-local
  view of any single producer, it would pick a *different* decoy
  service for that role. Only when aggregating across ≥ 3 roles
  does the gold emerge. This is the "local cleanliness is
  misleading but cross-role coherence is decisive" property the
  SDK v3.9 brief asks for.
* **Pre-committed bench property + falsifier.** The
  *cross-role-corroborated gold* property is mechanically tested in
  ``test_wevra_cross_role_corroboration.Phase55BankShapeTests``. A
  *decoy-corroborated decoy* falsifier regime is also defined in
  this file (see ``build_phase55_falsifier_bank``) and shows the
  W8-1 win does *not* hold when the decoy is the corroborated tag
  (W8-1 falsifier).

What the bench tests
====================

Three policies are compared at ``K_auditor=4`` with
``|candidates_to_auditor| ∈ {6, 8, 10}``:

* **substrate**                 — Phase-31 typed-handoff baseline.
* **capsule_fifo**              — capsule-native FIFO admission.
* **capsule_priority**          — claim-priority admission.
* **capsule_coverage**          — coverage-guided admission.
* **capsule_cohort_buffered**   — SDK v3.8 W7-2 buffered cohort
                                    (single-tag plurality).
* **capsule_corroboration**     — SDK v3.9 W8 buffered cross-role
                                    corroboration (this milestone).

Headline expected reading:

* On Phase 55 default (decoy-plurality + cross-role-corroborated
  gold), capsule_corroboration wins; capsule_cohort_buffered ties
  FIFO at 0.000 (W7-2 falsifier).
* On Phase 54 default (gold-plurality + foreign-decoys),
  capsule_corroboration matches W7-2 buffered cohort
  (backward-compatible W8-3).
* On Phase 53 default (low-surplus, real-LLM), no admission policy
  beats FIFO (W7-1 still holds — extraction-floor regime).

Theorem cross-reference (W8 family — minted by this milestone)
==============================================================

* **W8-1** — strict separation: on Phase 55, capsule_corroboration
  beats both substrate FIFO and SDK v3.8 W7-2 buffered cohort by
  ``≥ 0.50`` on ``accuracy_full`` at the pre-committed default,
  stable across ``≥ 3`` bank seeds.
* **W8-2** — structural: the corroboration score function strictly
  orders cross-role-corroborated gold above raw-plurality decoy when
  ``role_weight > raw_count_difference``.
* **W8-3** — backward-compatible: on Phase 54 default,
  capsule_corroboration ties or matches W7-2.
* **W8-4** — corroboration falsifier: when both raw plurality AND
  distinct-role coverage favour a decoy, capsule_corroboration ties
  FIFO and the W8-1 win does not hold.

Honest scope
============

* The bench is **deterministic** by design — the candidate streams
  are produced by code, not by an LLM. This isolates the *admission
  decision axis* from the *producer extraction quality axis*.
* A real-LLM variant is provided as an opt-in mode
  (``--llm-backend ollama --llm-model qwen2.5:14b-32k``) for
  empirical re-confirmation. CI does not invoke the LLM.
* The structural separation is *conditional* on the cross-role
  corroboration signal being present. The W8-4 falsifier regime
  is the explicit named counterexample.

CLI
---

::

    # Default deterministic Phase-55 default config (W8-1 anchor):
    python3 -m vision_mvp.experiments.phase55_decoy_plurality \\
        --K-auditor 4 --n-eval 10 \\
        --out docs/data/phase55_decoy_plurality_K4_n10.json

    # Falsifier regime sweep (W8-4 anchor):
    python3 -m vision_mvp.experiments.phase55_decoy_plurality \\
        --falsifier --n-eval 10 \\
        --out docs/data/phase55_falsifier_K4_n10.json

    # Budget sweep (W8-1-conditional anchor):
    python3 -m vision_mvp.experiments.phase55_decoy_plurality \\
        --budget-sweep --n-eval 10 \\
        --out docs/data/phase55_budget_sweep.json

The default configuration is the **pre-committed** evaluation
setting; do not retroactively widen it after seeing the result.
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
    AdmissionPolicy, ClaimPriorityAdmissionPolicy,
    CohortCoherenceAdmissionPolicy,
    CoverageGuidedAdmissionPolicy,
    CrossRoleCorroborationAdmissionPolicy,
    FifoAdmissionPolicy, RoleBudget, TeamCoordinator,
    audit_team_lifecycle,
)
from vision_mvp.experiments.phase52_team_coord import (
    PooledStrategy, StrategyResult, _format_canonical_answer,
    claim_priorities, make_team_budgets, pool,
)


# =============================================================================
# Phase 55 scenario family — decoy-plurality + cross-role corroboration
# =============================================================================


_KNOWN_DECOYS = (
    "archival", "backup", "cache", "logs", "audit_jobs",
    "users", "sessions", "catalog", "web", "metrics",
)


@dataclasses.dataclass(frozen=True)
class DecoyPluralityScenario:
    """One Phase-55 scenario.

    Fields
    ------
    scenario_id
        Short string key.
    description
        Human-readable one-liner.
    real_service
        The gold target service.
    decoy_storm_service
        The decoy that has *raw plurality* (strictly more total
        mentions than gold). Only one role emits a "storm" of this
        decoy.
    decoy_secondary_service
        A second decoy with one or two mentions. Strengthens the
        local-cleanliness-is-misleading property: each role's
        per-role plurality differs.
    gold_root_cause / gold_remediation
        Canonical decoder outputs the auditor must produce.
    role_emissions
        ``role -> [(claim_kind, payload), ...]`` ordered list. Each
        producer role's emissions are designed so that the role's
        *local* plurality of service-tagged mentions is a *decoy*,
        but the gold service appears in at least one role's
        emissions.
    """

    scenario_id: str
    description: str
    real_service: str
    decoy_storm_service: str
    decoy_secondary_service: str
    gold_root_cause: str
    gold_remediation: str
    role_emissions: dict[str, tuple[tuple[str, str], ...]]


_REMEDIATION = {
    "memory_leak":          "rollback_app_to_prev_release",
    "tls_expiry":           "renew_tls_and_reload",
    "dns_misroute":         "restore_internal_dns_zone",
    "disk_fill":            "rotate_logs_and_clear_backup",
    "deadlock":             "enforce_lock_ordering_in_orders",
    "pool_exhaustion":      "raise_pool_cap_or_fix_upstream",
    "slow_query_cascade":   "index_or_split_slow_query",
    "error_spike":          "roll_back_recent_deploy",
    "latency_spike":        "scale_up_api_pool",
    "fw_block":             "rescind_spurious_deny_rule",
}


def _emit(kind: str, payload: str) -> tuple[str, str]:
    return (kind, payload)


def _build_oom_api_with_archival_storm(
        decoy_storm: str = "archival",
        decoy_secondary: str = "logs") -> DecoyPluralityScenario:
    """OOM kill on api app; concurrent archival storm produces the
    raw plurality of mentions but only sysadmin sees archival."""
    real = "api"
    return DecoyPluralityScenario(
        scenario_id=f"oom_api__storm_{decoy_storm}",
        description=(
            f"OOM kill on {real} (memory_leak); concurrent {decoy_storm} "
            f"cron storm dominates sysadmin telemetry; benign "
            f"{decoy_secondary} chatter on monitor. Decoy raw plurality = "
            f"{decoy_storm}; gold cross-role corroboration = {real}."),
        real_service=real,
        decoy_storm_service=decoy_storm,
        decoy_secondary_service=decoy_secondary,
        gold_root_cause="memory_leak",
        gold_remediation=_REMEDIATION["memory_leak"],
        role_emissions={
            ROLE_MONITOR: (
                # monitor's local plurality is the secondary decoy
                _emit("LATENCY_SPIKE",
                       f"p95_ms=1100 service={decoy_secondary}"),
                _emit("LATENCY_SPIKE",
                       f"p95_ms=1200 service={decoy_secondary}"),
                _emit("LATENCY_SPIKE",
                       f"p95_ms=4700 service={real}"),
            ),
            ROLE_DB_ADMIN: (
                # db_admin emits one gold + one decoy_storm
                _emit("POOL_EXHAUSTION",
                       f"pool active=200/200 waiters=145 service={real}"),
                _emit("SLOW_QUERY_OBSERVED",
                       f"q#12 mean_ms=4210 service={decoy_storm}"),
            ),
            ROLE_SYSADMIN: (
                # sysadmin's local plurality IS the decoy_storm
                _emit("OOM_KILL",
                       f"oom_kill comm=cron rss=2.1G service={decoy_storm}"),
                _emit("OOM_KILL",
                       f"oom_kill comm=worker rss=1.8G "
                       f"service={decoy_storm}"),
                _emit("OOM_KILL",
                       f"oom_kill comm=archiver rss=1.6G "
                       f"service={decoy_storm}"),
                _emit("OOM_KILL",
                       f"oom_kill comm=app.py rss=8.1G service={real}"),
            ),
            ROLE_NETWORK: (
                # network's local plurality is the secondary decoy
                _emit("FW_BLOCK_SURGE",
                       f"rule=deny src=hc-probe count=12 "
                       f"service={decoy_secondary}"),
            ),
        },
    )


def _build_disk_orders_with_backup_storm(
        decoy_storm: str = "backup",
        decoy_secondary: str = "metrics") -> DecoyPluralityScenario:
    """Disk fill cascading to orders; backup storm dominates."""
    real = "orders"
    return DecoyPluralityScenario(
        scenario_id=f"disk_orders__storm_{decoy_storm}",
        description=(
            f"Disk fill cascading to {real}; concurrent {decoy_storm} "
            f"sweep produces raw plurality on monitor + sysadmin; "
            f"benign {decoy_secondary} chatter on db_admin. Gold "
            f"cross-role corroboration = {real}."),
        real_service=real,
        decoy_storm_service=decoy_storm,
        decoy_secondary_service=decoy_secondary,
        gold_root_cause="disk_fill",
        gold_remediation=_REMEDIATION["disk_fill"],
        role_emissions={
            ROLE_MONITOR: (
                _emit("ERROR_RATE_SPIKE",
                       f"error_rate=0.06 service={decoy_storm}"),
                _emit("ERROR_RATE_SPIKE",
                       f"error_rate=0.07 service={decoy_storm}"),
                _emit("ERROR_RATE_SPIKE",
                       f"error_rate=0.41 service={real}"),
            ),
            ROLE_DB_ADMIN: (
                _emit("SLOW_QUERY_OBSERVED",
                       f"q#7 mean_ms=900 service={decoy_secondary}"),
                _emit("POOL_EXHAUSTION",
                       f"pool active=200/200 service={real}"),
            ),
            ROLE_SYSADMIN: (
                _emit("CRON_OVERRUN",
                       f"backup.sh exit=137 duration_s=5400 "
                       f"service={decoy_storm}"),
                _emit("CRON_OVERRUN",
                       f"backup.sh exit=137 duration_s=7000 "
                       f"service={decoy_storm}"),
                _emit("DISK_FILL_CRITICAL",
                       f"/var/log used=99% fs=/ service={real}"),
            ),
            ROLE_NETWORK: (
                _emit("FW_BLOCK_SURGE",
                       f"rule=deny count=8 service={decoy_secondary}"),
            ),
        },
    )


def _build_dns_orders_with_users_storm(
        decoy_storm: str = "users",
        decoy_secondary: str = "cache") -> DecoyPluralityScenario:
    """DNS misroute cascade; users decoy storm on db_admin."""
    real = "orders"
    return DecoyPluralityScenario(
        scenario_id=f"dns_orders__storm_{decoy_storm}",
        description=(
            f"DNS misroute on db.internal cascading to {real}; "
            f"concurrent {decoy_storm} reconnect chatter dominates "
            f"db_admin local view. Gold cross-role corroboration = "
            f"{real}."),
        real_service=real,
        decoy_storm_service=decoy_storm,
        decoy_secondary_service=decoy_secondary,
        gold_root_cause="dns_misroute",
        gold_remediation=_REMEDIATION["dns_misroute"],
        role_emissions={
            ROLE_MONITOR: (
                _emit("ERROR_RATE_SPIKE",
                       f"error_rate=0.10 service={decoy_secondary}"),
                _emit("ERROR_RATE_SPIKE",
                       f"error_rate=0.88 service={real}"),
            ),
            ROLE_DB_ADMIN: (
                _emit("POOL_EXHAUSTION",
                       f"pool active=0/200 service={decoy_storm}"),
                _emit("POOL_EXHAUSTION",
                       f"pool active=0/200 service={decoy_storm}"),
                _emit("POOL_EXHAUSTION",
                       f"pool active=0/200 service={decoy_storm}"),
                _emit("POOL_EXHAUSTION",
                       f"pool active=0/200 service={real}"),
            ),
            ROLE_SYSADMIN: (
                _emit("CRON_OVERRUN",
                       f"sweep.sh duration_s=600 service={decoy_secondary}"),
            ),
            ROLE_NETWORK: (
                _emit("DNS_MISROUTE",
                       f"q={decoy_storm}.internal rc=NOERROR "
                       f"service={decoy_storm}"),
                _emit("DNS_MISROUTE",
                       f"q=db.internal rc=SERVFAIL service={real}"),
            ),
        },
    )


def _build_tls_api_with_cache_storm(
        decoy_storm: str = "cache",
        decoy_secondary: str = "metrics") -> DecoyPluralityScenario:
    """Expired TLS; cache evict storm dominates monitor + sysadmin."""
    real = "api"
    return DecoyPluralityScenario(
        scenario_id=f"tls_api__storm_{decoy_storm}",
        description=(
            f"Expired TLS cert on {real}; concurrent {decoy_storm} "
            f"evict storm produces raw plurality on monitor + sysadmin; "
            f"benign {decoy_secondary} chatter on db_admin."),
        real_service=real,
        decoy_storm_service=decoy_storm,
        decoy_secondary_service=decoy_secondary,
        gold_root_cause="tls_expiry",
        gold_remediation=_REMEDIATION["tls_expiry"],
        role_emissions={
            ROLE_MONITOR: (
                _emit("ERROR_RATE_SPIKE",
                       f"uptime_pct=92 service={decoy_storm}"),
                _emit("ERROR_RATE_SPIKE",
                       f"uptime_pct=88 service={decoy_storm}"),
                _emit("ERROR_RATE_SPIKE",
                       f"uptime_pct=41 service={real}"),
            ),
            ROLE_NETWORK: (
                _emit("FW_BLOCK_SURGE",
                       f"rule=deny count=20 service={decoy_secondary}"),
                _emit("TLS_EXPIRED",
                       f"tls handshake fail reason=expired service={real}"),
            ),
            ROLE_DB_ADMIN: (
                _emit("SLOW_QUERY_OBSERVED",
                       f"q#3 mean_ms=900 service={decoy_secondary}"),
                _emit("POOL_EXHAUSTION",
                       f"pool active=120/200 service={real}"),
            ),
            ROLE_SYSADMIN: (
                _emit("CRON_OVERRUN",
                       f"evict.sh duration_s=300 service={decoy_storm}"),
                _emit("CRON_OVERRUN",
                       f"evict.sh duration_s=400 service={decoy_storm}"),
            ),
        },
    )


def _build_deadlock_orders_with_logs_storm(
        decoy_storm: str = "logs",
        decoy_secondary: str = "audit_jobs") -> DecoyPluralityScenario:
    """Deadlock storm; logs decoy storm on monitor + network. The
    gold service is corroborated by 3 roles (monitor + db_admin +
    sysadmin) so the corroboration policy can lock onto it."""
    real = "orders"
    return DecoyPluralityScenario(
        scenario_id=f"deadlock_orders__storm_{decoy_storm}",
        description=(
            f"Lock-order bug on {real}_payments; concurrent {decoy_storm} "
            f"emission storm dominates monitor + network local view; "
            f"benign {decoy_secondary} chatter on sysadmin (with one "
            f"gold-tagged sweep). Gold cross-role corroboration = "
            f"3 roles."),
        real_service=real,
        decoy_storm_service=decoy_storm,
        decoy_secondary_service=decoy_secondary,
        gold_root_cause="deadlock",
        gold_remediation=_REMEDIATION["deadlock"],
        role_emissions={
            ROLE_MONITOR: (
                _emit("ERROR_RATE_SPIKE",
                       f"error_rate=0.04 service={decoy_storm}"),
                _emit("ERROR_RATE_SPIKE",
                       f"error_rate=0.05 service={decoy_storm}"),
                _emit("ERROR_RATE_SPIKE",
                       f"error_rate=0.22 service={real}"),
            ),
            ROLE_DB_ADMIN: (
                _emit("DEADLOCK_SUSPECTED",
                       f"deadlock relation={real}_payments "
                       f"service={real}"),
                _emit("POOL_EXHAUSTION",
                       f"pool active=200/200 waiters=88 service={real}"),
            ),
            ROLE_SYSADMIN: (
                _emit("CRON_OVERRUN",
                       f"audit.sh duration_s=120 service={decoy_secondary}"),
                # third role corroborating gold so the bench-property
                # gold_corroboration holds (3 roles > 2 decoy roles).
                _emit("CRON_OVERRUN",
                       f"sweep.sh duration_s=60 service={real}"),
            ),
            ROLE_NETWORK: (
                _emit("FW_BLOCK_SURGE",
                       f"rule=deny count=10 service={decoy_storm}"),
                _emit("FW_BLOCK_SURGE",
                       f"rule=deny count=12 service={decoy_storm}"),
                _emit("FW_BLOCK_SURGE",
                       f"rule=deny count=14 service={decoy_storm}"),
            ),
        },
    )


_BASE_BUILDERS = (
    _build_oom_api_with_archival_storm,
    _build_disk_orders_with_backup_storm,
    _build_dns_orders_with_users_storm,
    _build_tls_api_with_cache_storm,
    _build_deadlock_orders_with_logs_storm,
)


def build_phase55_bank(*,
                         n_replicates: int = 2,
                         seed: int = 11,
                         ) -> list[DecoyPluralityScenario]:
    """Return a deterministic Phase-55 scenario bank.

    Each base scenario is replicated ``n_replicates`` times under a
    deterministic permutation of the ``(decoy_storm, decoy_secondary)``
    tags. With ``n_replicates=2`` and 5 base builders, this yields a
    10-scenario bank.

    The decoy permutation NEVER picks the gold service or any of the
    ``("api", "orders")`` shared services (which are real_service
    candidates in other scenarios).
    """
    rng = random.Random(seed)
    pool_decoys = [s for s in _KNOWN_DECOYS]
    out: list[DecoyPluralityScenario] = []
    for builder in _BASE_BUILDERS:
        for r in range(n_replicates):
            i = rng.randrange(0, 1 << 16)
            chosen_storm = pool_decoys[(i + r) % len(pool_decoys)]
            chosen_secondary = pool_decoys[(i + r + 1) % len(pool_decoys)]
            if chosen_secondary == chosen_storm:
                chosen_secondary = pool_decoys[(i + r + 2)
                                                  % len(pool_decoys)]
            sc = builder(decoy_storm=chosen_storm,
                          decoy_secondary=chosen_secondary)
            out.append(dataclasses.replace(
                sc,
                scenario_id=(f"{sc.scenario_id}__rep{r}__"
                              f"sec_{chosen_secondary}")))
    return out


def build_phase55_falsifier_bank(*,
                                    n_replicates: int = 2,
                                    seed: int = 11,
                                    ) -> list[DecoyPluralityScenario]:
    """Return a *falsifier* bank where the decoy is ALSO
    cross-role corroborated (the W8-1 falsifier regime).

    Construction: take the Phase-55 default bank and *swap* the role
    that emits the gold-marked claim to a foreign role, so the gold
    only appears under one role while the decoy_storm appears under
    multiple roles. This breaks the cross-role-corroborated-gold
    property; capsule_corroboration ties FIFO on this bank.
    """
    base = build_phase55_bank(n_replicates=n_replicates, seed=seed)
    out: list[DecoyPluralityScenario] = []
    for sc in base:
        # Strip every gold-tagged emission down to ONE role; promote
        # one decoy_storm-tagged emission to a *second* role.
        role_emissions: dict[str, tuple[tuple[str, str], ...]] = {}
        gold_kept = False
        for role, ems in sc.role_emissions.items():
            new_ems: list[tuple[str, str]] = []
            for (kind, payload) in ems:
                if f"service={sc.real_service}" in payload:
                    if not gold_kept and role == ROLE_DB_ADMIN:
                        new_ems.append((kind, payload))
                        gold_kept = True
                    else:
                        # Demote gold-tagged emissions: replace tag.
                        repl = payload.replace(
                            f"service={sc.real_service}",
                            f"service={sc.decoy_storm_service}")
                        new_ems.append((kind, repl))
                else:
                    new_ems.append((kind, payload))
            role_emissions[role] = tuple(new_ems)
        out.append(dataclasses.replace(
            sc, role_emissions=role_emissions,
            scenario_id=f"{sc.scenario_id}__falsifier"))
    return out


# =============================================================================
# IncidentScenario adapter
# =============================================================================


def _as_incident_scenario(sc: DecoyPluralityScenario) -> IncidentScenario:
    """Adapter: turn a Phase-55 ``DecoyPluralityScenario`` into the
    Phase-31 ``IncidentScenario`` shape so we can reuse ``grade_answer``.

    Causal chain: every gold (role, claim_kind, payload) tuple, where
    "gold" = role_emissions entries whose payload contains
    ``service=<real_service>``.
    """
    chain: list[tuple[str, str, str, tuple[int, ...]]] = []
    for role, emissions in sc.role_emissions.items():
        for (kind, payload) in emissions:
            if f"service={sc.real_service}" in payload:
                chain.append((role, kind, payload, (0,)))
    return IncidentScenario(
        scenario_id=sc.scenario_id,
        description=sc.description,
        gold_root_cause=sc.gold_root_cause,
        gold_services=(sc.real_service,),
        gold_remediation=sc.gold_remediation,
        causal_chain=tuple(chain),
        per_role_events={r: () for r in ALL_ROLES},
    )


# =============================================================================
# Candidate-handoff stream construction
# =============================================================================


def build_candidate_stream(
        sc: DecoyPluralityScenario,
        ) -> list[tuple[str, str, str, str, tuple[int, ...]]]:
    """Materialise role-emissions into a per-scenario candidate
    stream ``[(source_role, to_role, kind, payload, source_event_ids),
    ...]`` routed via the standard subscription table.

    Order is **role-major**, then within-role original order.
    """
    subs = build_role_subscriptions()
    out: list[tuple[str, str, str, str, tuple[int, ...]]] = []
    for role in (ROLE_MONITOR, ROLE_DB_ADMIN, ROLE_SYSADMIN, ROLE_NETWORK):
        for (kind, payload) in sc.role_emissions.get(role, ()):
            consumers = subs.consumers(role, kind)
            if not consumers:
                continue
            for to_role in sorted(consumers):
                out.append((role, to_role, kind, payload, (0,)))
    return out


# =============================================================================
# Per-strategy run driver
# =============================================================================


@dataclasses.dataclass(frozen=True)
class _DecoderHandoffShim:
    source_role: str
    claim_kind: str
    payload: str
    n_tokens: int = 1


def _run_capsule_strategy(
        sc: DecoyPluralityScenario,
        candidates: Sequence[tuple[str, str, str, str, tuple[int, ...]]],
        budgets: dict[str, RoleBudget],
        policy_per_role: dict[str, AdmissionPolicy],
        strategy_name: str,
        ) -> StrategyResult:
    incident_sc = _as_incident_scenario(sc)
    ledger = CapsuleLedger()
    coord = TeamCoordinator(
        ledger=ledger, role_budgets=budgets,
        policy_per_role=policy_per_role,
        team_tag="decoy_plurality_phase55",
    )
    coord.advance_round(1)
    for (src, to, kind, payload, _evs) in candidates:
        coord.emit_handoff(
            source_role=src, to_role=to, claim_kind=kind, payload=payload)
    coord.seal_all_role_views()
    rv_cid = coord.role_view_cid(ROLE_AUDITOR)
    handoffs: list[_DecoderHandoffShim] = []
    if rv_cid and rv_cid in ledger:
        rv = ledger.get(rv_cid)
        for p in rv.parents:
            if p in ledger:
                cap = ledger.get(p)
                if cap.kind != CapsuleKind.TEAM_HANDOFF:
                    continue
                payload = (cap.payload if isinstance(cap.payload, dict)
                            else {})
                handoffs.append(_DecoderHandoffShim(
                    source_role=str(payload.get("source_role", "")),
                    claim_kind=str(payload.get("claim_kind", "")),
                    payload=str(payload.get("payload", "")),
                    n_tokens=int(payload.get("n_tokens", 1)),
                ))
    answer = _phase31_decoder_from_handoffs(handoffs)
    coord.seal_team_decision(team_role=ROLE_AUDITOR, decision=answer)
    audit = audit_team_lifecycle(ledger)
    grading = grade_answer(incident_sc, _format_canonical_answer(answer))

    rv = ledger.get(rv_cid) if rv_cid else None
    admitted_kinds: set[tuple[str, str]] = set()
    if rv is not None:
        for p in rv.parents:
            if p in ledger:
                cap = ledger.get(p)
                payload = (cap.payload if isinstance(cap.payload, dict)
                            else {})
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
    stats = coord.stats()
    dropped = stats["per_role_dropped"].get(ROLE_AUDITOR, {})
    return StrategyResult(
        strategy=strategy_name,
        scenario_id=sc.scenario_id,
        answer=answer,
        grading=grading,
        failure_kind=failure_kind,
        n_admitted_auditor=(rv.payload.get("n_admitted")
                              if rv is not None
                              and isinstance(rv.payload, dict) else 0),
        n_dropped_auditor_budget=int(dropped.get("budget_full", 0)),
        n_dropped_auditor_capacity=int(
            dropped.get("tokens_full", 0) + dropped.get("duplicate", 0)),
        n_dropped_auditor_unknown_kind=int(
            dropped.get("unknown_kind", 0) + dropped.get("score_low", 0)),
        n_team_handoff=stats["n_team_handoff"],
        n_role_view=stats["n_role_view"],
        n_team_decision=stats["n_team_decision"],
        audit_ok=audit.is_ok(),
        n_tokens_admitted=int(rv.payload.get("n_tokens_admitted", 0)
                                 if rv is not None
                                 and isinstance(rv.payload, dict) else 0),
    )


def _run_substrate_strategy(
        sc: DecoyPluralityScenario,
        candidates: Sequence[tuple[str, str, str, str, tuple[int, ...]]],
        inbox_capacity: int,
        ) -> StrategyResult:
    """Phase-31 typed-handoff substrate at matched inbox capacity."""
    incident_sc = _as_incident_scenario(sc)
    from vision_mvp.core.role_handoff import HandoffRouter, RoleInbox
    subs = build_role_subscriptions()
    router = HandoffRouter(subs=subs)
    for role in ALL_ROLES:
        router.register_inbox(RoleInbox(role=role, capacity=inbox_capacity))
    for (src, _to, kind, payload, evids) in candidates:
        router.emit(
            source_role=src,
            source_agent_id=ALL_ROLES.index(src),
            claim_kind=kind, payload=payload,
            source_event_ids=evids, round=1)
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
        n_team_handoff=0,
        n_role_view=0,
        n_team_decision=0,
        audit_ok=False,
        n_tokens_admitted=sum(h.n_tokens for h in held),
    )


# =============================================================================
# Bench-property metrics — pre-committed mechanical witnesses
# =============================================================================


_SERVICE_TAG_RE = re.compile(r"service=(\w+)")


def _candidate_stats(
        candidates: Sequence[tuple[str, str, str, str, tuple[int, ...]]],
        real_service: str,
        ) -> dict[str, Any]:
    """Compute the per-scenario bench-property metrics: raw counts and
    distinct-role counts for each service tag in the auditor stream."""
    cands_aud = [c for c in candidates if c[1] == ROLE_AUDITOR]
    raw: dict[str, int] = {}
    roles_per_tag: dict[str, set[str]] = {}
    for (src, _to, _kind, payload, _evs) in cands_aud:
        m = _SERVICE_TAG_RE.search(payload)
        if not m:
            continue
        tag = m.group(1)
        raw[tag] = raw.get(tag, 0) + 1
        roles_per_tag.setdefault(tag, set()).add(src)
    role_counts = {t: len(rs) for t, rs in roles_per_tag.items()}
    real_raw = raw.get(real_service, 0)
    real_roles = role_counts.get(real_service, 0)
    other_raw_max = max(
        (c for t, c in raw.items() if t != real_service),
        default=0)
    other_roles_max = max(
        (c for t, c in role_counts.items() if t != real_service),
        default=0)
    return {
        "raw_counts": dict(raw),
        "role_counts": dict(role_counts),
        "real_service": real_service,
        "real_raw": real_raw,
        "real_roles": real_roles,
        "max_decoy_raw": other_raw_max,
        "max_decoy_roles": other_roles_max,
        "decoy_plurality_holds": other_raw_max > real_raw,
        "gold_corroboration_holds": real_roles > other_roles_max,
        "n_candidates_to_auditor": len(cands_aud),
    }


# =============================================================================
# Top-level driver
# =============================================================================


def run_phase55(*,
                  n_eval: int | None = None,
                  K_auditor: int = 4,
                  T_auditor: int = 128,
                  K_producer: int = 6,
                  T_producer: int = 96,
                  inbox_capacity: int | None = None,
                  bank_seed: int = 11,
                  bank_replicates: int = 2,
                  use_falsifier_bank: bool = False,
                  verbose: bool = False,
                  ) -> dict[str, Any]:
    """Run Phase 55 end-to-end and return a JSON-serialisable report.

    The pre-committed default config is ``K_auditor=4, T_auditor=128,
    n_eval=10`` — do not retroactively widen.
    """
    if use_falsifier_bank:
        bank = build_phase55_falsifier_bank(
            n_replicates=bank_replicates, seed=bank_seed)
    else:
        bank = build_phase55_bank(
            n_replicates=bank_replicates, seed=bank_seed)
    if n_eval is not None:
        bank = bank[:n_eval]
    if inbox_capacity is None:
        inbox_capacity = K_auditor

    budgets = make_team_budgets(
        K_producer=K_producer, T_producer=T_producer,
        K_auditor=K_auditor, T_auditor=T_auditor)
    priorities = claim_priorities()

    results: list[StrategyResult] = []
    bench_property: dict[str, dict[str, Any]] = {}
    for sc in bank:
        cands = build_candidate_stream(sc)
        bench_property[sc.scenario_id] = _candidate_stats(
            cands, sc.real_service)
        cands_to_auditor = [c for c in cands if c[1] == ROLE_AUDITOR]
        # Buffered W7-2 cohort: pre-fitted single-tag plurality.
        cohort_buffered = CohortCoherenceAdmissionPolicy.from_candidate_payloads(
            [c[3] for c in cands_to_auditor])
        # Buffered W8 corroboration: pre-fitted (role, tag) score.
        corroboration = (
            CrossRoleCorroborationAdmissionPolicy.from_candidate_stream(
                [(c[0], c[3]) for c in cands_to_auditor]))
        scenario_strategies: dict[str, dict[str, AdmissionPolicy]] = {
            "capsule_fifo": {
                r: FifoAdmissionPolicy() for r in budgets
            },
            "capsule_priority": {
                r: ClaimPriorityAdmissionPolicy(priorities=priorities,
                                                  threshold=0.65)
                for r in budgets
            },
            "capsule_coverage": {
                r: CoverageGuidedAdmissionPolicy() for r in budgets
            },
            "capsule_cohort_buffered": {
                r: cohort_buffered for r in budgets
            },
            "capsule_corroboration": {
                r: corroboration for r in budgets
            },
        }
        results.append(_run_substrate_strategy(sc, cands, inbox_capacity))
        for sname, policy_per_role in scenario_strategies.items():
            results.append(_run_capsule_strategy(
                sc=sc, candidates=cands, budgets=budgets,
                policy_per_role=policy_per_role, strategy_name=sname))

    strategies = ("capsule_fifo", "capsule_priority", "capsule_coverage",
                   "capsule_cohort_buffered", "capsule_corroboration")
    strategy_names = ("substrate",) + strategies
    pooled = {s: pool(results, s).as_dict() for s in strategy_names}

    # Headline gaps:
    gap_corr_vs_fifo = round(
        pooled["capsule_corroboration"]["accuracy_full"]
        - pooled["capsule_fifo"]["accuracy_full"], 4)
    gap_corr_vs_cohort = round(
        pooled["capsule_corroboration"]["accuracy_full"]
        - pooled["capsule_cohort_buffered"]["accuracy_full"], 4)
    gap_corr_vs_substrate = round(
        pooled["capsule_corroboration"]["accuracy_full"]
        - pooled["substrate"]["accuracy_full"], 4)
    gap_cohort_vs_fifo = round(
        pooled["capsule_cohort_buffered"]["accuracy_full"]
        - pooled["capsule_fifo"]["accuracy_full"], 4)

    audit_ok_grid = {}
    for s in strategy_names:
        if s == "substrate":
            audit_ok_grid[s] = False
            continue
        rs = [r for r in results if r.strategy == s]
        audit_ok_grid[s] = bool(rs) and all(r.audit_ok for r in rs)

    # Bench-property witnesses (pre-committed mechanical checks).
    bench_summary = {
        "scenarios_with_decoy_plurality": sum(
            1 for v in bench_property.values()
            if v["decoy_plurality_holds"]),
        "scenarios_with_gold_corroboration": sum(
            1 for v in bench_property.values()
            if v["gold_corroboration_holds"]),
        "scenarios_with_surplus": sum(
            1 for v in bench_property.values()
            if v["n_candidates_to_auditor"] > K_auditor),
        "n_scenarios": len(bench_property),
        "K_auditor": K_auditor,
    }

    if verbose:
        print(f"[phase55] n_eval={len(bank)}, K_auditor={K_auditor}, "
              f"falsifier={use_falsifier_bank}",
              file=sys.stderr, flush=True)
        print(f"[phase55] bench_props: "
              f"decoy_plurality={bench_summary['scenarios_with_decoy_plurality']}/"
              f"{len(bank)}, "
              f"gold_corroborated={bench_summary['scenarios_with_gold_corroboration']}/"
              f"{len(bank)}, "
              f"surplus={bench_summary['scenarios_with_surplus']}/{len(bank)}",
              file=sys.stderr, flush=True)
        for s in strategy_names:
            p = pooled[s]
            print(f"[phase55]   {s:30s} "
                  f"acc_full={p['accuracy_full']:.3f} "
                  f"acc_root_cause={p['accuracy_root_cause']:.3f} "
                  f"acc_services={p['accuracy_services']:.3f} "
                  f"adm={p['mean_n_admitted_auditor']:.2f}",
                  file=sys.stderr, flush=True)
        print(f"[phase55] gap corroboration−fifo: {gap_corr_vs_fifo:+.3f}",
              file=sys.stderr, flush=True)
        print(f"[phase55] gap corroboration−cohort_buffered: "
              f"{gap_corr_vs_cohort:+.3f}",
              file=sys.stderr, flush=True)
        print(f"[phase55] gap cohort_buffered−fifo: "
              f"{gap_cohort_vs_fifo:+.3f}",
              file=sys.stderr, flush=True)

    return {
        "schema": "phase55.decoy_plurality.v1",
        "config": {
            "n_eval": len(bank),
            "K_auditor": K_auditor,
            "T_auditor": T_auditor,
            "K_producer": K_producer,
            "T_producer": T_producer,
            "inbox_capacity": inbox_capacity,
            "bank_seed": bank_seed,
            "bank_replicates": bank_replicates,
            "use_falsifier_bank": use_falsifier_bank,
        },
        "pooled": pooled,
        "audit_ok_grid": audit_ok_grid,
        "bench_summary": bench_summary,
        "bench_property_per_scenario": bench_property,
        "headline_gap": {
            "corroboration_minus_fifo_accuracy_full": gap_corr_vs_fifo,
            "corroboration_minus_cohort_buffered_accuracy_full":
                gap_corr_vs_cohort,
            "corroboration_minus_substrate_accuracy_full":
                gap_corr_vs_substrate,
            "cohort_buffered_minus_fifo_accuracy_full":
                gap_cohort_vs_fifo,
        },
        "scenarios_evaluated": [sc.scenario_id for sc in bank],
        "n_results": len(results),
    }


def run_phase55_budget_sweep(*,
                                K_values: Sequence[int] = (2, 3, 4, 5, 6, 8),
                                n_eval: int = 10,
                                bank_seed: int = 11,
                                bank_replicates: int = 2,
                                ) -> dict[str, Any]:
    """W8-1-conditional: sweep K_auditor to identify the structure-win
    window for the corroboration policy."""
    pooled_per_K: dict[int, dict[str, Any]] = {}
    headline_gap_per_K: dict[int, dict[str, float]] = {}
    audit_ok_per_K: dict[int, dict[str, bool]] = {}
    for K in K_values:
        rep = run_phase55(
            n_eval=n_eval, K_auditor=K, T_auditor=max(64, 32 * K),
            inbox_capacity=K, bank_seed=bank_seed,
            bank_replicates=bank_replicates, verbose=False)
        pooled_per_K[K] = rep["pooled"]
        headline_gap_per_K[K] = rep["headline_gap"]
        audit_ok_per_K[K] = rep["audit_ok_grid"]
    return {
        "schema": "phase55.decoy_plurality_budget_sweep.v1",
        "K_values": list(K_values),
        "n_eval": n_eval,
        "pooled_per_K": pooled_per_K,
        "headline_gap_per_K": headline_gap_per_K,
        "audit_ok_per_K": audit_ok_per_K,
    }


def run_cross_regime_summary(*, n_eval: int = 10, bank_seed: int = 11,
                                bank_replicates: int = 2,
                                ) -> dict[str, Any]:
    """Run Phase 55 default + Phase 55 falsifier + Phase 54 default
    side-by-side under matched K_auditor=4. The W8-1 / W8-3 / W8-4
    contracts read from this report.
    """
    from vision_mvp.experiments.phase54_cross_role_coherence import (
        run_phase54,
    )
    p55_default = run_phase55(
        n_eval=n_eval, K_auditor=4, T_auditor=128,
        bank_seed=bank_seed, bank_replicates=bank_replicates,
        use_falsifier_bank=False, verbose=False)
    p55_falsifier = run_phase55(
        n_eval=n_eval, K_auditor=4, T_auditor=128,
        bank_seed=bank_seed, bank_replicates=bank_replicates,
        use_falsifier_bank=True, verbose=False)
    p54_default = run_phase54(
        n_eval=n_eval, K_auditor=4, T_auditor=128,
        bank_seed=bank_seed, bank_replicates=bank_replicates,
        verbose=False)
    return {
        "schema": "phase55.cross_regime.v1",
        "config": {
            "K_auditor": 4, "T_auditor": 128,
            "n_eval": n_eval, "bank_seed": bank_seed,
            "bank_replicates": bank_replicates,
        },
        "phase55_default": p55_default,
        "phase55_falsifier": p55_falsifier,
        "phase54_default": p54_default,
    }


def run_seed_stability_sweep(*, seeds: Sequence[int] = (11, 17, 23, 29, 31),
                                n_eval: int = 10,
                                K_auditor: int = 4,
                                T_auditor: int = 128,
                                ) -> dict[str, Any]:
    """W8-1 stability anchor: run Phase 55 default at multiple
    bank seeds and report the headline gap for each."""
    per_seed: dict[int, dict[str, Any]] = {}
    for seed in seeds:
        rep = run_phase55(
            n_eval=n_eval, K_auditor=K_auditor, T_auditor=T_auditor,
            bank_seed=seed, bank_replicates=2,
            verbose=False)
        per_seed[seed] = {
            "headline_gap": rep["headline_gap"],
            "pooled": rep["pooled"],
            "audit_ok_grid": rep["audit_ok_grid"],
            "bench_summary": rep["bench_summary"],
        }
    return {
        "schema": "phase55.decoy_plurality_seed_sweep.v1",
        "seeds": list(seeds),
        "K_auditor": K_auditor,
        "T_auditor": T_auditor,
        "n_eval": n_eval,
        "per_seed": per_seed,
    }


# =============================================================================
# CLI
# =============================================================================


def _arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Phase 55 — decoy-plurality + cross-role-corroboration "
                    "multi-agent benchmark (SDK v3.9 / W8 family).")
    p.add_argument("--K-auditor", type=int, default=4)
    p.add_argument("--T-auditor", type=int, default=128)
    p.add_argument("--n-eval", type=int, default=10)
    p.add_argument("--bank-seed", type=int, default=11)
    p.add_argument("--bank-replicates", type=int, default=2)
    p.add_argument("--falsifier", action="store_true",
                    help="run the W8-1 falsifier bank instead.")
    p.add_argument("--budget-sweep", action="store_true")
    p.add_argument("--cross-regime", action="store_true",
                    help="run Phase 53/54/55 side by side at K=4.")
    p.add_argument("--seed-sweep", action="store_true",
                    help="run Phase 55 default at multiple bank seeds.")
    p.add_argument("--out", type=str, default="")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _arg_parser().parse_args(argv)
    if args.budget_sweep:
        report = run_phase55_budget_sweep(
            n_eval=args.n_eval, bank_seed=args.bank_seed,
            bank_replicates=args.bank_replicates)
    elif args.cross_regime:
        report = run_cross_regime_summary(
            n_eval=args.n_eval, bank_seed=args.bank_seed,
            bank_replicates=args.bank_replicates)
    elif args.seed_sweep:
        report = run_seed_stability_sweep(
            n_eval=args.n_eval, K_auditor=args.K_auditor,
            T_auditor=args.T_auditor)
    else:
        report = run_phase55(
            n_eval=args.n_eval,
            K_auditor=args.K_auditor,
            T_auditor=args.T_auditor,
            bank_seed=args.bank_seed,
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
            print(f"[phase55] wrote {args.out}", file=sys.stderr)
    else:
        if not args.quiet:
            print(json.dumps(report.get("pooled", report),
                              indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
