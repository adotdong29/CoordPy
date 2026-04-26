"""Phase 56 — multi-service-gold + cross-role-corroborated multi-agent
benchmark (SDK v3.10, W9 family anchor).

This phase is the **stronger, harder regime** the SDK v3.10 milestone
introduces. It directly attacks the named falsifier of W8 / SDK v3.9:

* SDK v3.9 W8-1 wins on Phase 55 because the gold answer's
  ``services`` set is a *singleton* — the auditor only needs to
  identify *one* causal service, and the buffered cross-role
  corroboration policy admits exactly the candidates carrying the
  top-1 corroborated service tag.
* But on a *multi-service-gold* candidate stream (gold answer has
  ``gold_services = (A, B)`` — both services are causal), the W8
  policy admits only candidates with the top-1 corroborated tag,
  yielding ``services = {A}`` (or ``{B}``); the decoder's set-equality
  ``services_correct`` check fails because the auditor is missing
  half of the gold service set. This is the pre-committed W8
  multi-service-gold falsifier (named in
  ``HOW_NOT_TO_OVERSTATE.md``).
* Phase 56 instantiates exactly that falsifier — every scenario has
  *two* gold services, each independently corroborated by ≥ 2
  distinct producer roles. A *single-role* decoy storm (raw plurality
  > role_weight × 1) is layered on top so substrate FIFO and W7-2
  buffered cohort also fail.

A policy that admits the **top-K** cross-role-corroborated tags above
a min-role threshold — the
``MultiServiceCorroborationAdmissionPolicy`` shipped in this SDK —
can recover the multi-service gold answer without admitting the
single-role decoy storm.

Why this is fair, not benchmark-shopping
========================================

* **Multi-service incidents are the canonical realistic case.** Every
  real production incident the SRE community catalogues
  (db-pool-cascade, deadlock-payments-orders, latency-fanout,
  disk-fill-cron) involves **two or more** affected services
  simultaneously. The single-service Phase-55 regime is the *easy*
  case; multi-service is the regime where corroboration alone is
  necessary but not sufficient.
* **Cross-role corroboration is still the right signal — top-K is
  the right *aggregator*.** The bench keeps every Phase-55 fairness
  property (locally-misleading distractors, budget pressure,
  realistic claim_kinds) and adds **multi-service gold** as the
  axis along which W8 becomes inadequate.
* **Pre-committed bench property + falsifier.** The
  *multi-service-gold-corroborated* property is mechanically tested
  in ``test_wevra_multi_service_corroboration.Phase56BankShapeTests``.
  A *decoy-corroborated decoy* falsifier regime
  (``build_phase56_falsifier_bank``) shows the W9-1 win does *not*
  hold when a decoy is also corroborated above the min-role
  threshold (W9-4 falsifier).

What the bench tests
====================

Six policies are compared at ``K_auditor=4`` with
``|candidates_to_auditor| ∈ {6, 7, 8}``:

* **substrate**                 — Phase-31 typed-handoff baseline.
* **capsule_fifo**              — capsule-native FIFO admission.
* **capsule_priority**          — claim-priority admission.
* **capsule_coverage**          — coverage-guided admission.
* **capsule_cohort_buffered**   — SDK v3.8 W7-2 buffered cohort
                                    (single-tag plurality).
* **capsule_corroboration**     — SDK v3.9 W8 buffered cross-role
                                    corroboration (single-tag).
* **capsule_multi_service**     — SDK v3.10 W9 buffered multi-service
                                    corroboration (top-K with min-role
                                    threshold).

Headline expected reading:

* On Phase 56 default (multi-service-gold + corroborated, decoy storm),
  capsule_multi_service wins. capsule_corroboration ties FIFO
  at 0.000 (W8 multi-service falsifier).
  capsule_cohort_buffered ties FIFO at 0.000 (W7-2 falsifier).
* On Phase 55 default, capsule_multi_service ties capsule_corroboration
  at 1.000 (backward-compat — W9-3).
* On Phase 54 default, capsule_multi_service ties W7-2 at 1.000
  (single-service-gold case collapses to single-tag W7-2; W9-3'
  backward-compat).
* On Phase 53 default (low-surplus, real-LLM-shaped), no admission
  policy beats FIFO (W7-1 still holds — extraction-floor regime).

Theorem cross-reference (W9 family — minted by this milestone)
==============================================================

* **W9-1** — strict separation: on Phase 56, capsule_multi_service
  beats both substrate FIFO, SDK v3.8 W7-2, and SDK v3.9 W8 by
  ``≥ 0.50`` on ``accuracy_full`` at the pre-committed default,
  stable across ``≥ 3`` bank seeds.
* **W9-2** — structural: with role_weight=100, top_k=2, and
  min_corroborated_roles=2, the dominant set excludes any
  single-role decoy regardless of raw count, and includes any tag
  with ≥2 distinct producer roles when the decoy is single-role.
* **W9-3** — backward-compatible on Phase 55 default (only one tag
  passes the role threshold; W9 collapses to W8).
* **W9-4** — multi-service falsifier: when both raw plurality AND
  ≥ min_corroborated_roles distinct-role coverage favour a decoy,
  capsule_multi_service ties FIFO and the W9-1 win does not hold.

Honest scope
============

* Bench is **deterministic** by design — candidate streams are
  produced by code, not by an LLM. This isolates the *admission
  decision axis* (single-tag vs top-K) from extraction quality.
* The structural separation is **conditional** on the bench
  properties (multi-service gold corroborated by ≥2 roles each,
  single-role decoy storms only). The W9-4 falsifier regime is
  the explicit named counterexample.

CLI
---

::

    # Default deterministic Phase-56 default config (W9-1 anchor):
    python3 -m vision_mvp.experiments.phase56_multi_service_corroboration \\
        --K-auditor 4 --n-eval 10 \\
        --out docs/data/phase56_multi_service_K4_n10.json

    # Falsifier regime (W9-4 anchor):
    python3 -m vision_mvp.experiments.phase56_multi_service_corroboration \\
        --falsifier --n-eval 10 \\
        --out docs/data/phase56_falsifier_K4_n10.json

    # Cross-regime summary (Phase 53/54/55/56 side by side at K=4):
    python3 -m vision_mvp.experiments.phase56_multi_service_corroboration \\
        --cross-regime --n-eval 10 \\
        --out docs/data/phase56_cross_regime.json

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
    FifoAdmissionPolicy, MultiServiceCorroborationAdmissionPolicy,
    RoleBudget, TeamCoordinator, audit_team_lifecycle,
)
from vision_mvp.experiments.phase52_team_coord import (
    StrategyResult, _format_canonical_answer,
    claim_priorities, make_team_budgets, pool,
)


# =============================================================================
# Phase 56 scenario family — multi-service-gold + cross-role corroboration
# =============================================================================


@dataclasses.dataclass(frozen=True)
class MultiServiceScenario:
    """One Phase-56 scenario.

    Fields
    ------
    scenario_id
        Short string key.
    description
        Human-readable one-liner.
    gold_services_pair
        Tuple ``(A, B)`` of the two gold service tags. Both must be
        corroborated by ≥ 2 distinct producer roles in
        ``role_emissions``. The decoder must produce
        ``services = {A, B}`` for ``services_correct``.
    decoy_storm_service
        The decoy that has *raw plurality* (strictly more total
        mentions than each gold service). It must be corroborated
        by **exactly one** producer role (Phase-56 default property).
    gold_root_cause / gold_remediation
        Canonical decoder outputs the auditor must produce.
    role_emissions
        ``role -> [(claim_kind, payload), ...]`` ordered list. Each
        gold service is mentioned by ≥ 2 distinct roles via causal
        claim_kinds; the decoy_storm appears only on one role.
    """

    scenario_id: str
    description: str
    gold_services_pair: tuple[str, str]
    decoy_storm_service: str
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


# =============================================================================
# Five base scenario builders — multi-service-gold + 1-role decoy storm
# =============================================================================


def _build_pool_api_db_with_archival_storm(
        decoy_storm: str = "archival") -> MultiServiceScenario:
    """Pool-exhaustion cascade across api + db; archival cron storm
    on monitor produces raw plurality (single-role) decoy."""
    A, B = "api", "db"
    return MultiServiceScenario(
        scenario_id=f"pool_api_db__storm_{decoy_storm}",
        description=(
            f"db pool exhaustion cascading to {A} latency; archival "
            f"cron storm on monitor produces raw-plurality decoy. "
            f"Gold = ({A}, {B}); each corroborated by 2 roles."),
        gold_services_pair=(A, B),
        decoy_storm_service=decoy_storm,
        gold_root_cause="pool_exhaustion",
        gold_remediation=_REMEDIATION["pool_exhaustion"],
        role_emissions={
            ROLE_MONITOR: (
                # decoy storm — single-role corroboration only
                _emit("ERROR_RATE_SPIKE",
                       f"error_rate=0.04 service={decoy_storm}"),
                _emit("ERROR_RATE_SPIKE",
                       f"error_rate=0.05 service={decoy_storm}"),
                _emit("ERROR_RATE_SPIKE",
                       f"error_rate=0.06 service={decoy_storm}"),
                # gold corroboration — monitor side
                _emit("LATENCY_SPIKE",
                       f"p95_ms=4200 service={A}"),
                _emit("ERROR_RATE_SPIKE",
                       f"error_rate=0.18 service={B}"),
            ),
            ROLE_DB_ADMIN: (
                # gold corroboration — db_admin side
                _emit("POOL_EXHAUSTION",
                       f"pool active=200/200 waiters=145 service={A}"),
                _emit("SLOW_QUERY_OBSERVED",
                       f"q#12 mean_ms=4210 service={B}"),
            ),
            ROLE_SYSADMIN: (),
            ROLE_NETWORK: (),
        },
    )


def _build_deadlock_orders_payments_with_cache_storm(
        decoy_storm: str = "cache") -> MultiServiceScenario:
    """Deadlock cascade orders ↔ payments; cache evict storm on
    monitor (single-role decoy)."""
    A, B = "orders", "payments"
    return MultiServiceScenario(
        scenario_id=f"deadlock_orders_payments__storm_{decoy_storm}",
        description=(
            f"Lock-order bug between {A} and {B}; concurrent "
            f"{decoy_storm} evict storm on monitor. Gold = "
            f"({A}, {B}); each corroborated by 2 roles."),
        gold_services_pair=(A, B),
        decoy_storm_service=decoy_storm,
        gold_root_cause="deadlock",
        gold_remediation=_REMEDIATION["deadlock"],
        role_emissions={
            ROLE_MONITOR: (
                _emit("ERROR_RATE_SPIKE",
                       f"error_rate=0.07 service={decoy_storm}"),
                _emit("ERROR_RATE_SPIKE",
                       f"error_rate=0.08 service={decoy_storm}"),
                _emit("ERROR_RATE_SPIKE",
                       f"error_rate=0.09 service={decoy_storm}"),
                _emit("LATENCY_SPIKE",
                       f"p95_ms=2100 service={A}"),
                _emit("ERROR_RATE_SPIKE",
                       f"error_rate=0.22 service={B}"),
            ),
            ROLE_DB_ADMIN: (
                _emit("DEADLOCK_SUSPECTED",
                       f"deadlock relation={A}_payments service={A}"),
                _emit("POOL_EXHAUSTION",
                       f"pool active=200/200 waiters=88 service={B}"),
            ),
            ROLE_SYSADMIN: (),
            ROLE_NETWORK: (),
        },
    )


def _build_slow_query_web_db_with_metrics_storm(
        decoy_storm: str = "metrics") -> MultiServiceScenario:
    """Slow-query cascade db → web frontend; metrics scrape storm on
    monitor (single-role decoy). Gold corroboration: web by monitor +
    network; db by monitor + db_admin. SLOW_QUERY_OBSERVED is the
    highest-priority causal claim_kind in the admitted set, so the
    decoder lands on slow_query_cascade."""
    A, B = "web", "db"
    return MultiServiceScenario(
        scenario_id=f"slow_query_web_db__storm_{decoy_storm}",
        description=(
            f"Slow-query cascade {B} → {A} frontend; concurrent "
            f"{decoy_storm} scrape storm on monitor. Gold = "
            f"({A}, {B}); {A} corroborated by monitor + network, "
            f"{B} corroborated by monitor + db_admin."),
        gold_services_pair=(A, B),
        decoy_storm_service=decoy_storm,
        gold_root_cause="slow_query_cascade",
        gold_remediation=_REMEDIATION["slow_query_cascade"],
        role_emissions={
            ROLE_MONITOR: (
                _emit("LATENCY_SPIKE",
                       f"p95_ms=200 service={decoy_storm}"),
                _emit("LATENCY_SPIKE",
                       f"p95_ms=180 service={decoy_storm}"),
                _emit("LATENCY_SPIKE",
                       f"p95_ms=210 service={decoy_storm}"),
                _emit("LATENCY_SPIKE",
                       f"p95_ms=4100 service={A}"),
                _emit("ERROR_RATE_SPIKE",
                       f"error_rate=0.15 service={B}"),
            ),
            ROLE_DB_ADMIN: (
                _emit("SLOW_QUERY_OBSERVED",
                       f"q#7 mean_ms=3900 service={B}"),
            ),
            ROLE_SYSADMIN: (),
            # Network FW_BLOCK is the lowest-priority kind in the
            # decoder priority table, so it cannot override
            # SLOW_QUERY_OBSERVED for the root_cause inference. It
            # simply provides the second role for web corroboration.
            ROLE_NETWORK: (
                _emit("FW_BLOCK_SURGE",
                       f"rule=deny count=4 service={A}"),
            ),
        },
    )


def _build_error_api_mobile_with_logs_storm(
        decoy_storm: str = "logs") -> MultiServiceScenario:
    """Error spike cascading to api + mobile; logs ingestion storm
    on monitor (single-role decoy). FW noise from network on gold
    services provides 2nd-role corroboration."""
    A, B = "api", "mobile"
    return MultiServiceScenario(
        scenario_id=f"error_api_mobile__storm_{decoy_storm}",
        description=(
            f"Error spike across {A} + {B} after deploy; concurrent "
            f"{decoy_storm} ingestion storm on monitor. Gold = "
            f"({A}, {B}); each corroborated by 2 roles "
            f"(monitor + network)."),
        gold_services_pair=(A, B),
        decoy_storm_service=decoy_storm,
        gold_root_cause="error_spike",
        gold_remediation=_REMEDIATION["error_spike"],
        role_emissions={
            ROLE_MONITOR: (
                _emit("LATENCY_SPIKE",
                       f"p95_ms=300 service={decoy_storm}"),
                _emit("LATENCY_SPIKE",
                       f"p95_ms=290 service={decoy_storm}"),
                _emit("LATENCY_SPIKE",
                       f"p95_ms=320 service={decoy_storm}"),
                _emit("ERROR_RATE_SPIKE",
                       f"error_rate=0.31 service={A}"),
                _emit("ERROR_RATE_SPIKE",
                       f"error_rate=0.27 service={B}"),
            ),
            ROLE_DB_ADMIN: (),
            ROLE_SYSADMIN: (),
            ROLE_NETWORK: (
                _emit("FW_BLOCK_SURGE",
                       f"rule=deny count=18 service={A}"),
                _emit("FW_BLOCK_SURGE",
                       f"rule=deny count=14 service={B}"),
            ),
        },
    )


def _build_disk_storage_logs_with_backup_storm(
        decoy_storm: str = "backup") -> MultiServiceScenario:
    """Disk fill cascading to storage + logs services; backup sweep
    storm on db_admin (single-role decoy)."""
    A, B = "storage", "logs_pipeline"
    return MultiServiceScenario(
        scenario_id=f"disk_storage_logs__storm_{decoy_storm}",
        description=(
            f"Disk fill on /var/log cascading to {A} + {B}; concurrent "
            f"{decoy_storm} sweep storm on db_admin only. Gold = "
            f"({A}, {B}); each corroborated by 2 roles "
            f"(sysadmin + monitor)."),
        gold_services_pair=(A, B),
        decoy_storm_service=decoy_storm,
        gold_root_cause="disk_fill",
        gold_remediation=_REMEDIATION["disk_fill"],
        role_emissions={
            ROLE_MONITOR: (
                _emit("ERROR_RATE_SPIKE",
                       f"error_rate=0.41 service={A}"),
                _emit("LATENCY_SPIKE",
                       f"p95_ms=4500 service={B}"),
            ),
            ROLE_DB_ADMIN: (
                _emit("SLOW_QUERY_OBSERVED",
                       f"q#3 mean_ms=8000 service={decoy_storm}"),
                _emit("SLOW_QUERY_OBSERVED",
                       f"q#4 mean_ms=8100 service={decoy_storm}"),
                _emit("SLOW_QUERY_OBSERVED",
                       f"q#5 mean_ms=8200 service={decoy_storm}"),
            ),
            ROLE_SYSADMIN: (
                _emit("DISK_FILL_CRITICAL",
                       f"/var/log used=99% fs=/ service={A}"),
                _emit("DISK_FILL_CRITICAL",
                       f"/var/log used=98% fs=/ service={B}"),
            ),
            ROLE_NETWORK: (),
        },
    )


_BASE_BUILDERS = (
    _build_pool_api_db_with_archival_storm,
    _build_deadlock_orders_payments_with_cache_storm,
    _build_slow_query_web_db_with_metrics_storm,
    _build_error_api_mobile_with_logs_storm,
    _build_disk_storage_logs_with_backup_storm,
)


_KNOWN_DECOYS = (
    "archival", "cache", "metrics", "logs", "backup",
    "audit_jobs", "sessions", "catalog", "users", "telemetry",
)


def build_phase56_bank(*,
                         n_replicates: int = 2,
                         seed: int = 11,
                         ) -> list[MultiServiceScenario]:
    """Return a deterministic Phase-56 scenario bank.

    Each base scenario is replicated ``n_replicates`` times under a
    deterministic permutation of the decoy_storm tag. With
    ``n_replicates=2`` and 5 base builders, this yields a 10-scenario
    bank.

    The decoy permutation NEVER picks any of the gold services
    actually used in the scenario family.
    """
    rng = random.Random(seed)
    pool_decoys = list(_KNOWN_DECOYS)
    out: list[MultiServiceScenario] = []
    for builder in _BASE_BUILDERS:
        for r in range(n_replicates):
            i = rng.randrange(0, 1 << 16)
            chosen = pool_decoys[(i + r) % len(pool_decoys)]
            sc = builder(decoy_storm=chosen)
            out.append(dataclasses.replace(
                sc, scenario_id=f"{sc.scenario_id}__rep{r}"))
    return out


def build_phase56_falsifier_bank(*,
                                    n_replicates: int = 2,
                                    seed: int = 11,
                                    ) -> list[MultiServiceScenario]:
    """Return a *falsifier* bank where the decoy is ALSO
    cross-role corroborated (the W9-1 falsifier regime, W9-4).

    Construction: take the Phase-56 default bank and *promote* the
    decoy_storm to a second role by replacing one of the gold-tagged
    emissions on a different role with a same-kind decoy-tagged one.
    Concretely, we add a second-role decoy emission so that the
    decoy's ``|distinct_roles|`` becomes 2 (≥ min_corroborated_roles)
    and its score equals or exceeds at least one gold tag. This
    breaks the multi-service-gold-corroborated property; the W9
    policy admits a decoy-tagged candidate and ``services_correct``
    fails.
    """
    base = build_phase56_bank(n_replicates=n_replicates, seed=seed)
    out: list[MultiServiceScenario] = []
    for sc in base:
        role_emissions = dict(sc.role_emissions)
        # Find which role currently emits the decoy storm (single-role
        # by construction of the default bank), and a different role
        # that emits gold-tagged claims; we replicate one decoy-tagged
        # emission onto the second role so the decoy becomes 2-role
        # corroborated. Choose a high-priority-safe kind: ERROR_RATE
        # (monitor), LATENCY (monitor), FW_BLOCK (network), POOL
        # (db_admin), or SLOW_QUERY (db_admin) — pick the first
        # available subscribed kind for the second role.
        decoy = sc.decoy_storm_service
        # Identify the producer role that emits the decoy storm:
        decoy_emitter = None
        for role, ems in sc.role_emissions.items():
            if any(f"service={decoy}" in p for (_k, p) in ems):
                decoy_emitter = role
                break
        # Pick a different role whose subscribed kinds let us emit a
        # decoy-tagged claim that won't override the gold root_cause.
        # Map of safe-emitter→safe kind:
        safe_choices = (
            (ROLE_NETWORK, "FW_BLOCK_SURGE"),
            (ROLE_MONITOR, "LATENCY_SPIKE"),
            (ROLE_DB_ADMIN, "POOL_EXHAUSTION"),
            (ROLE_SYSADMIN, "CRON_OVERRUN"),
        )
        promoted = False
        for (role, kind) in safe_choices:
            if role == decoy_emitter:
                continue
            # CRON maps to disk_fill → only safe if gold_root_cause is
            # disk_fill or memory_leak/tls_expiry/dns_misroute (higher
            # priority would still need DISK to admit, but CRON also
            # → disk_fill which clashes with non-disk_fill golds).
            if (kind == "CRON_OVERRUN"
                    and sc.gold_root_cause != "disk_fill"):
                continue
            existing = list(role_emissions.get(role, ()))
            existing.append(_emit(
                kind, f"falsifier-promotion service={decoy}"))
            role_emissions[role] = tuple(existing)
            promoted = True
            break
        if not promoted:
            # Final fallback: append a network FW_BLOCK_SURGE
            # tagged decoy. FW_BLOCK is the lowest-priority kind in
            # the catalogue and cannot override any gold root_cause.
            existing = list(role_emissions.get(ROLE_NETWORK, ()))
            existing.append(_emit("FW_BLOCK_SURGE",
                                   f"rule=deny count=8 service={decoy}"))
            role_emissions[ROLE_NETWORK] = tuple(existing)
        out.append(dataclasses.replace(
            sc, role_emissions=role_emissions,
            scenario_id=f"{sc.scenario_id}__falsifier"))
    return out


# =============================================================================
# IncidentScenario adapter
# =============================================================================


def _as_incident_scenario(sc: MultiServiceScenario) -> IncidentScenario:
    """Adapter: turn a Phase-56 scenario into the Phase-31 shape so
    we can reuse ``grade_answer``.

    Causal chain: every (role, claim_kind, payload) tuple where the
    payload contains a service tag matching either gold service.
    """
    chain: list[tuple[str, str, str, tuple[int, ...]]] = []
    A, B = sc.gold_services_pair
    for role, emissions in sc.role_emissions.items():
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


# =============================================================================
# Candidate-handoff stream construction
# =============================================================================


def build_candidate_stream(
        sc: MultiServiceScenario,
        ) -> list[tuple[str, str, str, str, tuple[int, ...]]]:
    """Materialise role-emissions into a per-scenario candidate
    stream ``[(source_role, to_role, kind, payload, source_event_ids),
    ...]`` routed via the standard subscription table.

    Order is **role-major**, then within-role original order. This
    mirrors phase55's stream construction.
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
        sc: MultiServiceScenario,
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
        team_tag="multi_service_phase56",
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
        sc: MultiServiceScenario,
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
        gold_services_pair: tuple[str, str],
        decoy_storm_service: str,
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
    A, B = gold_services_pair
    gold_set = {A, B}
    n_gold_corroborated = sum(
        1 for g in gold_set if role_counts.get(g, 0) >= 2)
    decoy_roles = role_counts.get(decoy_storm_service, 0)
    decoy_raw = raw.get(decoy_storm_service, 0)
    other_max = max((c for t, c in role_counts.items() if t not in gold_set),
                     default=0)
    return {
        "raw_counts": dict(raw),
        "role_counts": dict(role_counts),
        "gold_services_pair": list(sorted(gold_set)),
        "decoy_storm_service": decoy_storm_service,
        "n_gold_corroborated_roles_geq2": n_gold_corroborated,
        "decoy_roles": decoy_roles,
        "decoy_raw": decoy_raw,
        "max_decoy_role_count": other_max,
        # Phase-56 default bench property: BOTH gold services
        # corroborated by ≥2 roles AND every decoy ≤1 role.
        "multi_service_gold_property_holds": (
            n_gold_corroborated == 2
            and other_max <= 1),
        # Decoy-corroborated falsifier property: at least one decoy
        # has ≥2 distinct producer roles (W9-4 falsifier).
        "decoy_corroboration_holds": other_max >= 2,
        "n_candidates_to_auditor": len(cands_aud),
    }


# =============================================================================
# Top-level driver
# =============================================================================


def run_phase56(*,
                  n_eval: int | None = None,
                  K_auditor: int = 4,
                  T_auditor: int = 128,
                  K_producer: int = 6,
                  T_producer: int = 96,
                  inbox_capacity: int | None = None,
                  bank_seed: int = 11,
                  bank_replicates: int = 2,
                  use_falsifier_bank: bool = False,
                  ms_top_k: int = 2,
                  ms_min_corroborated_roles: int = 2,
                  verbose: bool = False,
                  ) -> dict[str, Any]:
    """Run Phase 56 end-to-end and return a JSON-serialisable report.

    The pre-committed default config is ``K_auditor=4, T_auditor=128,
    n_eval=10, ms_top_k=2, ms_min_corroborated_roles=2`` — do not
    retroactively widen.
    """
    if use_falsifier_bank:
        bank = build_phase56_falsifier_bank(
            n_replicates=bank_replicates, seed=bank_seed)
    else:
        bank = build_phase56_bank(
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
            cands, sc.gold_services_pair, sc.decoy_storm_service)
        cands_to_auditor = [c for c in cands if c[1] == ROLE_AUDITOR]
        cohort_buffered = (
            CohortCoherenceAdmissionPolicy.from_candidate_payloads(
                [c[3] for c in cands_to_auditor]))
        corroboration = (
            CrossRoleCorroborationAdmissionPolicy.from_candidate_stream(
                [(c[0], c[3]) for c in cands_to_auditor]))
        multi_service = (
            MultiServiceCorroborationAdmissionPolicy.from_candidate_stream(
                [(c[0], c[3]) for c in cands_to_auditor],
                top_k=ms_top_k,
                min_corroborated_roles=ms_min_corroborated_roles))
        scenario_strategies: dict[str, dict[str, AdmissionPolicy]] = {
            "capsule_fifo": {
                r: FifoAdmissionPolicy() for r in budgets},
            "capsule_priority": {
                r: ClaimPriorityAdmissionPolicy(priorities=priorities,
                                                  threshold=0.65)
                for r in budgets},
            "capsule_coverage": {
                r: CoverageGuidedAdmissionPolicy() for r in budgets},
            "capsule_cohort_buffered": {
                r: cohort_buffered for r in budgets},
            "capsule_corroboration": {
                r: corroboration for r in budgets},
            "capsule_multi_service": {
                r: multi_service for r in budgets},
        }
        results.append(_run_substrate_strategy(sc, cands, inbox_capacity))
        for sname, policy_per_role in scenario_strategies.items():
            results.append(_run_capsule_strategy(
                sc=sc, candidates=cands, budgets=budgets,
                policy_per_role=policy_per_role, strategy_name=sname))

    strategies = ("capsule_fifo", "capsule_priority", "capsule_coverage",
                   "capsule_cohort_buffered", "capsule_corroboration",
                   "capsule_multi_service")
    strategy_names = ("substrate",) + strategies
    pooled = {s: pool(results, s).as_dict() for s in strategy_names}

    gap_ms_vs_fifo = round(
        pooled["capsule_multi_service"]["accuracy_full"]
        - pooled["capsule_fifo"]["accuracy_full"], 4)
    gap_ms_vs_cohort = round(
        pooled["capsule_multi_service"]["accuracy_full"]
        - pooled["capsule_cohort_buffered"]["accuracy_full"], 4)
    gap_ms_vs_corroboration = round(
        pooled["capsule_multi_service"]["accuracy_full"]
        - pooled["capsule_corroboration"]["accuracy_full"], 4)
    gap_ms_vs_substrate = round(
        pooled["capsule_multi_service"]["accuracy_full"]
        - pooled["substrate"]["accuracy_full"], 4)
    gap_corr_vs_fifo = round(
        pooled["capsule_corroboration"]["accuracy_full"]
        - pooled["capsule_fifo"]["accuracy_full"], 4)

    audit_ok_grid = {}
    for s in strategy_names:
        if s == "substrate":
            audit_ok_grid[s] = False
            continue
        rs = [r for r in results if r.strategy == s]
        audit_ok_grid[s] = bool(rs) and all(r.audit_ok for r in rs)

    bench_summary = {
        "scenarios_with_multi_service_gold_corroboration": sum(
            1 for v in bench_property.values()
            if v["multi_service_gold_property_holds"]),
        "scenarios_with_decoy_corroboration": sum(
            1 for v in bench_property.values()
            if v["decoy_corroboration_holds"]),
        "scenarios_with_surplus": sum(
            1 for v in bench_property.values()
            if v["n_candidates_to_auditor"] > K_auditor),
        "n_scenarios": len(bench_property),
        "K_auditor": K_auditor,
    }

    if verbose:
        print(f"[phase56] n_eval={len(bank)}, K_auditor={K_auditor}, "
              f"falsifier={use_falsifier_bank}, top_k={ms_top_k}, "
              f"min_corroborated_roles={ms_min_corroborated_roles}",
              file=sys.stderr, flush=True)
        print(f"[phase56] bench_props: "
              f"multi_service_gold={bench_summary['scenarios_with_multi_service_gold_corroboration']}/"
              f"{len(bank)}, "
              f"decoy_corroborated={bench_summary['scenarios_with_decoy_corroboration']}/"
              f"{len(bank)}, "
              f"surplus={bench_summary['scenarios_with_surplus']}/{len(bank)}",
              file=sys.stderr, flush=True)
        for s in strategy_names:
            p = pooled[s]
            print(f"[phase56]   {s:30s} "
                  f"acc_full={p['accuracy_full']:.3f} "
                  f"acc_root_cause={p['accuracy_root_cause']:.3f} "
                  f"acc_services={p['accuracy_services']:.3f} "
                  f"adm={p['mean_n_admitted_auditor']:.2f}",
                  file=sys.stderr, flush=True)
        print(f"[phase56] gap multi_service−fifo: {gap_ms_vs_fifo:+.3f}",
              file=sys.stderr, flush=True)
        print(f"[phase56] gap multi_service−cohort_buffered: "
              f"{gap_ms_vs_cohort:+.3f}",
              file=sys.stderr, flush=True)
        print(f"[phase56] gap multi_service−corroboration: "
              f"{gap_ms_vs_corroboration:+.3f}",
              file=sys.stderr, flush=True)
        print(f"[phase56] gap corroboration−fifo: "
              f"{gap_corr_vs_fifo:+.3f}",
              file=sys.stderr, flush=True)

    return {
        "schema": "phase56.multi_service_corroboration.v1",
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
            "ms_top_k": ms_top_k,
            "ms_min_corroborated_roles": ms_min_corroborated_roles,
        },
        "pooled": pooled,
        "audit_ok_grid": audit_ok_grid,
        "bench_summary": bench_summary,
        "bench_property_per_scenario": bench_property,
        "headline_gap": {
            "multi_service_minus_fifo_accuracy_full": gap_ms_vs_fifo,
            "multi_service_minus_cohort_buffered_accuracy_full":
                gap_ms_vs_cohort,
            "multi_service_minus_corroboration_accuracy_full":
                gap_ms_vs_corroboration,
            "multi_service_minus_substrate_accuracy_full":
                gap_ms_vs_substrate,
            "corroboration_minus_fifo_accuracy_full":
                gap_corr_vs_fifo,
        },
        "scenarios_evaluated": [sc.scenario_id for sc in bank],
        "n_results": len(results),
    }


def run_phase56_seed_stability_sweep(
        *, seeds: Sequence[int] = (11, 17, 23, 29, 31),
        n_eval: int = 10,
        K_auditor: int = 4,
        T_auditor: int = 128,
        ) -> dict[str, Any]:
    """W9-1 stability anchor: run Phase 56 default at multiple bank
    seeds and report the headline gap for each."""
    per_seed: dict[int, dict[str, Any]] = {}
    for seed in seeds:
        rep = run_phase56(
            n_eval=n_eval, K_auditor=K_auditor, T_auditor=T_auditor,
            bank_seed=seed, bank_replicates=2, verbose=False)
        per_seed[seed] = {
            "headline_gap": rep["headline_gap"],
            "pooled": rep["pooled"],
            "audit_ok_grid": rep["audit_ok_grid"],
            "bench_summary": rep["bench_summary"],
        }
    return {
        "schema": "phase56.multi_service_corroboration_seed_sweep.v1",
        "seeds": list(seeds),
        "K_auditor": K_auditor,
        "T_auditor": T_auditor,
        "n_eval": n_eval,
        "per_seed": per_seed,
    }


def run_cross_regime_summary(*, n_eval: int = 10, bank_seed: int = 11,
                                bank_replicates: int = 2,
                                ) -> dict[str, Any]:
    """Run Phase 53/54/55/56 default + Phase 56 falsifier side-by-side
    under matched K_auditor=4. The W9-1 / W9-3 / W9-4 contracts read
    from this report."""
    from vision_mvp.experiments.phase54_cross_role_coherence import (
        run_phase54)
    from vision_mvp.experiments.phase55_decoy_plurality import (
        run_phase55)
    p54_default = run_phase54(
        n_eval=n_eval, K_auditor=4, T_auditor=128,
        bank_seed=bank_seed, bank_replicates=bank_replicates,
        verbose=False)
    p55_default = run_phase55(
        n_eval=n_eval, K_auditor=4, T_auditor=128,
        bank_seed=bank_seed, bank_replicates=bank_replicates,
        use_falsifier_bank=False, verbose=False)
    p56_default = run_phase56(
        n_eval=n_eval, K_auditor=4, T_auditor=128,
        bank_seed=bank_seed, bank_replicates=bank_replicates,
        use_falsifier_bank=False, verbose=False)
    p56_falsifier = run_phase56(
        n_eval=n_eval, K_auditor=4, T_auditor=128,
        bank_seed=bank_seed, bank_replicates=bank_replicates,
        use_falsifier_bank=True, verbose=False)
    return {
        "schema": "phase56.cross_regime.v1",
        "config": {
            "K_auditor": 4, "T_auditor": 128,
            "n_eval": n_eval, "bank_seed": bank_seed,
            "bank_replicates": bank_replicates,
        },
        "phase54_default": p54_default,
        "phase55_default": p55_default,
        "phase56_default": p56_default,
        "phase56_falsifier": p56_falsifier,
    }


# =============================================================================
# CLI
# =============================================================================


def _arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Phase 56 — multi-service-gold + cross-role-corroborated "
                    "multi-agent benchmark (SDK v3.10 / W9 family).")
    p.add_argument("--K-auditor", type=int, default=4)
    p.add_argument("--T-auditor", type=int, default=128)
    p.add_argument("--n-eval", type=int, default=10)
    p.add_argument("--bank-seed", type=int, default=11)
    p.add_argument("--bank-replicates", type=int, default=2)
    p.add_argument("--ms-top-k", type=int, default=2)
    p.add_argument("--ms-min-corroborated-roles", type=int, default=2)
    p.add_argument("--falsifier", action="store_true",
                    help="run the W9-1 falsifier bank instead.")
    p.add_argument("--cross-regime", action="store_true",
                    help="run Phase 54/55/56 side by side at K=4.")
    p.add_argument("--seed-sweep", action="store_true",
                    help="run Phase 56 default at multiple bank seeds.")
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
        report = run_phase56_seed_stability_sweep(
            n_eval=args.n_eval, K_auditor=args.K_auditor,
            T_auditor=args.T_auditor)
    else:
        report = run_phase56(
            n_eval=args.n_eval,
            K_auditor=args.K_auditor,
            T_auditor=args.T_auditor,
            bank_seed=args.bank_seed,
            bank_replicates=args.bank_replicates,
            use_falsifier_bank=args.falsifier,
            ms_top_k=args.ms_top_k,
            ms_min_corroborated_roles=args.ms_min_corroborated_roles,
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
            print(f"[phase56] wrote {args.out}", file=sys.stderr)
    else:
        if not args.quiet:
            print(json.dumps(report.get("pooled", report),
                              indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
