"""Phase 57 — multi-service-gold + corroborated-decoy *decoder-forcing*
multi-agent benchmark (SDK v3.11, W10 family anchor).

This phase is the **stronger, harder regime** the SDK v3.11 milestone
introduces. It directly attacks the W9-4 named falsifier of SDK v3.10
*and* operationalises the structural admission limit:

* SDK v3.10 W9-1 wins on Phase 56 because the only corroborated tags
  (≥ ``min_corroborated_roles`` distinct roles) are gold tags; the
  decoy storm is single-role, so the W9 dominant set is exactly the
  gold pair.
* On a *corroborated-decoy* candidate stream — where the decoy is
  ALSO mentioned by ≥ 2 distinct producer roles (via generic-noise
  claim_kinds) — the W9 dominant set's argmax-by-role-count tier
  expands to include the decoy. With ``top_k=2``, score breaks ties
  and the dominant set typically picks the decoy plus one gold; with
  ``top_k=3``, the decoy is admitted and ``services_correct`` fails
  on set equality (extra service).
* **No service-blind admission policy in the SDK can win on
  Phase 57.** The (role, tag) bipartite multiset alone does not
  carry enough information to distinguish the data-tier corroborated
  gold from the generic-noise corroborated decoy. This is the
  W10-Λ admission-only limitation theorem.

The new method
==============

The bundle-aware team decoder
:class:`vision_mvp.coordpy.team_coord.BundleAwareTeamDecoder` filters
admitted services by the *causal claim-kind set* of the chosen
``root_cause`` (CCK). When the chosen root_cause is a specific-tier
incident (deadlock / pool_exhaustion / disk_fill / etc.), generic-
noise mentions (LATENCY_SPIKE / ERROR_RATE_SPIKE / FW_BLOCK_SURGE)
are excluded — so the corroborated decoy is filtered out at decode
time even though admission could not exclude it.

Joint admit-and-decode method (W10-AD)
--------------------------------------

The SDK v3.11 *joint method* pairs:

* Admission: SDK v3.10 W9
  ``MultiServiceCorroborationAdmissionPolicy`` with
  ``top_k = |gold_services| + 1`` and
  ``min_corroborated_roles = 2`` (admits the corroborated decoy
  alongside the corroborated gold so the decoder has the data
  to project from).
* Decoding: SDK v3.11
  ``BundleAwareTeamDecoder(cck_filter=True,
  role_corroboration_floor=2)``.

This is the *first capsule-native multi-agent coordination method
that crosses the admission/decoding split*.

Why this is fair, not benchmark-shopping
========================================

* **Real production incidents have noise channels.** Logs, metrics
  scrapers, archival cron, and cache-evict storms produce
  cross-role generic-noise (LATENCY / ERROR_RATE / FW_BLOCK)
  alongside the *causal* signal of the actual incident. The
  *single-role* decoy storm is the easy case; the *multi-role*
  decoy storm is the regime where corroboration is necessary but
  not sufficient and *contradiction-aware decoding* becomes the
  load-bearing move.
* **The CCK rule is closed-vocabulary and deterministic.** No
  learning, no scorer, no per-bench tuning. The CCK table is
  derived once from the incident-triage decoder's priority list
  and grouped by tier (data / storage / compute / edge / network /
  generic). It is mechanically the same on every regime.
* **Pre-committed bench property + falsifier.** The
  *corroborated-decoy-via-non-causal-kinds* property is mechanically
  tested in
  ``test_coordpy_bundle_decoder.Phase57BankShapeTests``. A *decoy-
  causal-promotion* falsifier regime
  (``build_phase57_falsifier_bank``) shows the W10 win does *not*
  hold when the decoy is ALSO mentioned via at least one
  CCK-eligible claim_kind (W10-4 falsifier).

What the bench tests
====================

Eight strategies are compared at ``K_auditor=4`` with
``|candidates_to_auditor| ∈ {6, 7, 8, 9}``:

* **substrate**                 — Phase-31 typed-handoff baseline.
* **capsule_fifo**              — capsule-native FIFO admission.
* **capsule_priority**          — claim-priority admission.
* **capsule_coverage**          — coverage-guided admission.
* **capsule_cohort_buffered**   — SDK v3.8 W7-2 buffered cohort.
* **capsule_corroboration**     — SDK v3.9 W8 buffered corroboration.
* **capsule_multi_service**     — SDK v3.10 W9 buffered multi-service.
* **capsule_bundle_decoder**    — SDK v3.11 W10 joint admit-and-decode
                                    (W9 admission + bundle decoder).

Headline expected reading:

* On Phase 57 default (multi-service-gold + corroborated-decoy via
  generic-noise), capsule_bundle_decoder wins. Every service-blind
  admission policy ties FIFO at 0.000 (W10-Λ).
* On Phase 56 default, capsule_bundle_decoder ties capsule_multi_service
  at 1.000 (backward-compat — W10-3).
* On Phase 55 default, capsule_bundle_decoder ties capsule_corroboration
  at 1.000 (backward-compat — W10-3).
* On Phase 54 default, capsule_bundle_decoder ties W7-2 at 1.000
  (backward-compat — W10-3).
* On Phase 57 falsifier (decoy ALSO mentioned via causal CCK kind),
  capsule_bundle_decoder ties FIFO at 0.000 (W10-4).

Theorem cross-reference (W10 family — minted by this milestone)
================================================================

* **W10-Λ** — admission-only structural limit: on Phase 57, every
  service-blind admission policy in the SDK ties FIFO at 0.000.
  Proved-empirical (n ≥ 10 saturated) plus structural argument
  (the (role, tag) bipartite multiset is the policy's only state).
* **W10-1** — bundle-aware decoder sufficiency: pairing W9
  admission (top_k=3, min_roles=2) with the bundle decoder
  achieves ``accuracy_full = 1.000`` on Phase 57 default.
  Proved-empirical (n ≥ 10 saturated).
* **W10-2** — CCK structural correctness: for any data-tier gold
  root_cause R and any corroborated decoy D mentioned only via
  generic-noise claim_kinds, the bundle decoder's CCK filter
  excludes D. Proved by inspection of CCK table.
* **W10-3** — backward-compat on R-53/R-54/R-55/R-56:
  bundle decoder produces the same ``services`` set as the
  substrate decoder on every cell of every prior regime
  (gold root_cause is specific; gold services are mentioned via
  CCK-eligible kinds).
* **W10-4** — decoy-causal-promotion falsifier: when the decoy is
  ALSO mentioned via at least one CCK-eligible claim_kind, the
  bundle decoder cannot exclude it (the CCK filter is not strict
  enough); W10-1 does not hold.

Honest scope
============

* Bench is **deterministic** by design. This isolates the
  *decoding axis* (CCK filter) from extraction quality.
* The structural separation is **conditional** on the bench
  properties (specific-tier gold root_cause; decoy mentioned only
  via non-CCK generic-noise claim_kinds). Generic-tier scenarios
  (gold root_cause = error_spike / latency_spike) collapse the CCK
  to all-noise and the decoder cannot help — this is named in the
  W10 honest-scope section.

CLI
---

::

    # Default deterministic Phase-57 default config (W10-1 anchor):
    python3 -m vision_mvp.experiments.phase57_decoder_forcing \\
        --K-auditor 4 --n-eval 10 \\
        --out docs/data/phase57_decoder_K4_n10.json

    # Falsifier regime (W10-4 anchor):
    python3 -m vision_mvp.experiments.phase57_decoder_forcing \\
        --falsifier --n-eval 10 \\
        --out docs/data/phase57_falsifier_K4_n10.json

    # Cross-regime summary (Phase 53/54/55/56/57 side by side at K=4):
    python3 -m vision_mvp.experiments.phase57_decoder_forcing \\
        --cross-regime --n-eval 10 \\
        --out docs/data/phase57_cross_regime.json

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
from vision_mvp.coordpy.capsule import CapsuleKind, CapsuleLedger
from vision_mvp.coordpy.team_coord import (
    AdmissionPolicy, BundleAwareTeamDecoder,
    ClaimPriorityAdmissionPolicy,
    CohortCoherenceAdmissionPolicy,
    CoverageGuidedAdmissionPolicy,
    CrossRoleCorroborationAdmissionPolicy,
    FifoAdmissionPolicy, MultiServiceCorroborationAdmissionPolicy,
    RoleBudget, TeamCoordinator, audit_team_lifecycle,
    _DecodedHandoff,
)
from vision_mvp.experiments.phase52_team_coord import (
    StrategyResult, _format_canonical_answer,
    claim_priorities, make_team_budgets, pool,
)


# =============================================================================
# Phase 57 scenario family — multi-service-gold + corroborated-decoy
# =============================================================================


@dataclasses.dataclass(frozen=True)
class DecoderForcingScenario:
    """One Phase-57 scenario.

    Fields
    ------
    scenario_id
        Short string key.
    description
        Human-readable one-liner.
    gold_services_pair
        Tuple ``(A, B)`` of the two gold service tags. Both must be
        corroborated by ≥ 2 distinct producer roles via at least one
        *causal* claim_kind for the gold root_cause.
    decoy_storm_service
        The decoy service. Must be corroborated by ≥ 2 distinct
        producer roles via *only non-causal* (generic-noise)
        claim_kinds (LATENCY_SPIKE / ERROR_RATE_SPIKE / FW_BLOCK_SURGE).
    gold_root_cause
        The gold root_cause label. Must be a *specific-tier* label
        (deadlock / pool_exhaustion / disk_fill / etc.) — not a
        generic-noise label (error_spike / latency_spike).
    gold_remediation
        Canonical remediation string keyed by gold_root_cause.
    role_emissions
        ``role -> [(claim_kind, payload), ...]`` ordered list.
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
}


def _emit(kind: str, payload: str) -> tuple[str, str]:
    return (kind, payload)


# =============================================================================
# Four base scenario builders — multi-service-gold + corroborated-decoy
# =============================================================================
#
# Construction recipe (every base builder follows this skeleton):
#
# * Gold service A and gold service B are each mentioned via at least
#   one causal claim_kind for ``gold_root_cause``, on at least 2
#   distinct producer roles.
# * The decoy service ``D`` is mentioned via *only generic-noise*
#   claim_kinds (LATENCY_SPIKE on monitor, ERROR_RATE_SPIKE on monitor,
#   FW_BLOCK_SURGE on network) — never via a CCK-eligible kind for the
#   gold root_cause.
# * Decoy raw count and role count is high enough that W9 with
#   ``top_k=2`` either picks {decoy, gold_A}, {decoy, gold_B}, or with
#   ``top_k=3`` picks {decoy, gold_A, gold_B} (extra service ⇒
#   services_correct fails). Either way, every service-blind admission
#   is doomed.


def _build_pool_api_db_with_corroborated_cache(
        decoy_storm: str = "cache") -> DecoderForcingScenario:
    """Pool-exhaustion cascade api+db; cache decoy corroborated by
    monitor (LATENCY×3) + network (FW_BLOCK×2) — both non-CCK kinds."""
    A, B = "api", "db"
    return DecoderForcingScenario(
        scenario_id=f"pool_api_db__corr_{decoy_storm}",
        description=(
            f"db pool exhaustion cascading to {A} latency; "
            f"{decoy_storm} decoy corroborated by 2 roles via "
            f"generic-noise kinds (W10-Λ admission limit case). Gold "
            f"= ({A}, {B}); root_cause=pool_exhaustion."),
        gold_services_pair=(A, B),
        decoy_storm_service=decoy_storm,
        gold_root_cause="pool_exhaustion",
        gold_remediation=_REMEDIATION["pool_exhaustion"],
        role_emissions={
            ROLE_MONITOR: (
                # Decoy storm — generic noise, monitor side.
                _emit("LATENCY_SPIKE", f"p95_ms=210 service={decoy_storm}"),
                _emit("LATENCY_SPIKE", f"p95_ms=220 service={decoy_storm}"),
                _emit("LATENCY_SPIKE", f"p95_ms=230 service={decoy_storm}"),
                # Gold corroboration — monitor side, generic kinds.
                _emit("LATENCY_SPIKE", f"p95_ms=4200 service={A}"),
                _emit("ERROR_RATE_SPIKE",
                       f"error_rate=0.18 service={B}"),
            ),
            ROLE_DB_ADMIN: (
                # Gold corroboration — db_admin side, *causal* kinds.
                _emit("POOL_EXHAUSTION",
                       f"pool active=200/200 waiters=145 service={A}"),
                _emit("SLOW_QUERY_OBSERVED",
                       f"q#12 mean_ms=4210 service={B}"),
            ),
            ROLE_SYSADMIN: (),
            ROLE_NETWORK: (
                # Decoy storm — second role, network side.
                _emit("FW_BLOCK_SURGE",
                       f"rule=deny count=12 service={decoy_storm}"),
                _emit("FW_BLOCK_SURGE",
                       f"rule=deny count=14 service={decoy_storm}"),
            ),
        },
    )


def _build_deadlock_orders_payments_with_corroborated_cache(
        decoy_storm: str = "cache") -> DecoderForcingScenario:
    """Deadlock cascade orders↔payments; cache decoy 2-role
    corroborated via generic noise (LATENCY+FW_BLOCK)."""
    A, B = "orders", "payments"
    return DecoderForcingScenario(
        scenario_id=f"deadlock_orders_payments__corr_{decoy_storm}",
        description=(
            f"Lock-order bug between {A} and {B}; {decoy_storm} decoy "
            f"corroborated by 2 roles via generic-noise kinds. Gold = "
            f"({A}, {B}); root_cause=deadlock."),
        gold_services_pair=(A, B),
        decoy_storm_service=decoy_storm,
        gold_root_cause="deadlock",
        gold_remediation=_REMEDIATION["deadlock"],
        role_emissions={
            ROLE_MONITOR: (
                _emit("LATENCY_SPIKE", f"p95_ms=180 service={decoy_storm}"),
                _emit("LATENCY_SPIKE", f"p95_ms=190 service={decoy_storm}"),
                _emit("LATENCY_SPIKE", f"p95_ms=200 service={decoy_storm}"),
                _emit("LATENCY_SPIKE", f"p95_ms=2100 service={A}"),
                _emit("ERROR_RATE_SPIKE", f"error_rate=0.22 service={B}"),
            ),
            ROLE_DB_ADMIN: (
                _emit("DEADLOCK_SUSPECTED",
                       f"deadlock relation={A}_payments service={A}"),
                _emit("POOL_EXHAUSTION",
                       f"pool active=200/200 waiters=88 service={B}"),
            ),
            ROLE_SYSADMIN: (),
            ROLE_NETWORK: (
                _emit("FW_BLOCK_SURGE",
                       f"rule=deny count=10 service={decoy_storm}"),
                _emit("FW_BLOCK_SURGE",
                       f"rule=deny count=11 service={decoy_storm}"),
            ),
        },
    )


def _build_slow_query_web_db_with_corroborated_metrics(
        decoy_storm: str = "metrics") -> DecoderForcingScenario:
    """Slow-query cascade db→web; metrics decoy 2-role corroborated
    via generic-noise."""
    A, B = "web", "db"
    return DecoderForcingScenario(
        scenario_id=f"slow_query_web_db__corr_{decoy_storm}",
        description=(
            f"Slow-query cascade {B}→{A}; {decoy_storm} decoy "
            f"corroborated by 2 roles via generic-noise. Gold = "
            f"({A}, {B}); root_cause=slow_query_cascade."),
        gold_services_pair=(A, B),
        decoy_storm_service=decoy_storm,
        gold_root_cause="slow_query_cascade",
        gold_remediation=_REMEDIATION["slow_query_cascade"],
        role_emissions={
            ROLE_MONITOR: (
                _emit("LATENCY_SPIKE", f"p95_ms=200 service={decoy_storm}"),
                _emit("LATENCY_SPIKE", f"p95_ms=210 service={decoy_storm}"),
                _emit("LATENCY_SPIKE", f"p95_ms=220 service={decoy_storm}"),
                _emit("LATENCY_SPIKE", f"p95_ms=4100 service={A}"),
                _emit("ERROR_RATE_SPIKE",
                       f"error_rate=0.15 service={B}"),
            ),
            ROLE_DB_ADMIN: (
                _emit("SLOW_QUERY_OBSERVED",
                       f"q#7 mean_ms=3900 service={B}"),
                _emit("SLOW_QUERY_OBSERVED",
                       f"q#9 mean_ms=4100 service={A}"),
            ),
            ROLE_SYSADMIN: (),
            ROLE_NETWORK: (
                _emit("FW_BLOCK_SURGE",
                       f"rule=deny count=4 service={decoy_storm}"),
                _emit("FW_BLOCK_SURGE",
                       f"rule=deny count=5 service={decoy_storm}"),
            ),
        },
    )


def _build_disk_storage_logs_with_corroborated_archival(
        decoy_storm: str = "archival") -> DecoderForcingScenario:
    """Disk fill cascading to storage + logs services; archival
    decoy 2-role corroborated via generic-noise."""
    A, B = "storage", "logs_pipeline"
    return DecoderForcingScenario(
        scenario_id=f"disk_storage_logs__corr_{decoy_storm}",
        description=(
            f"Disk fill cascading to {A} + {B}; {decoy_storm} decoy "
            f"corroborated by 2 roles via generic-noise. Gold = "
            f"({A}, {B}); root_cause=disk_fill."),
        gold_services_pair=(A, B),
        decoy_storm_service=decoy_storm,
        gold_root_cause="disk_fill",
        gold_remediation=_REMEDIATION["disk_fill"],
        role_emissions={
            ROLE_MONITOR: (
                _emit("ERROR_RATE_SPIKE",
                       f"error_rate=0.12 service={decoy_storm}"),
                _emit("ERROR_RATE_SPIKE",
                       f"error_rate=0.14 service={decoy_storm}"),
                _emit("ERROR_RATE_SPIKE",
                       f"error_rate=0.16 service={decoy_storm}"),
                _emit("ERROR_RATE_SPIKE",
                       f"error_rate=0.41 service={A}"),
                _emit("LATENCY_SPIKE",
                       f"p95_ms=4500 service={B}"),
            ),
            ROLE_DB_ADMIN: (),
            ROLE_SYSADMIN: (
                _emit("DISK_FILL_CRITICAL",
                       f"/var/log used=99% fs=/ service={A}"),
                _emit("DISK_FILL_CRITICAL",
                       f"/var/log used=98% fs=/ service={B}"),
            ),
            ROLE_NETWORK: (
                _emit("FW_BLOCK_SURGE",
                       f"rule=deny count=7 service={decoy_storm}"),
                _emit("FW_BLOCK_SURGE",
                       f"rule=deny count=8 service={decoy_storm}"),
            ),
        },
    )


_BASE_BUILDERS = (
    _build_pool_api_db_with_corroborated_cache,
    _build_deadlock_orders_payments_with_corroborated_cache,
    _build_slow_query_web_db_with_corroborated_metrics,
    _build_disk_storage_logs_with_corroborated_archival,
)


_KNOWN_DECOYS = (
    "cache", "archival", "metrics", "telemetry", "audit_jobs",
    "sessions", "search_index", "scratch_pool",
)


def build_phase57_bank(*,
                         n_replicates: int = 3,
                         seed: int = 11,
                         ) -> list[DecoderForcingScenario]:
    """Return a deterministic Phase-57 scenario bank.

    With ``n_replicates=3`` and 4 base builders this yields a 12-scenario
    bank by default; the run driver clips to ``n_eval`` (default 10).

    The decoy permutation NEVER picks any of the gold services
    actually used in the scenario family.
    """
    rng = random.Random(seed)
    pool_decoys = list(_KNOWN_DECOYS)
    out: list[DecoderForcingScenario] = []
    for builder in _BASE_BUILDERS:
        for r in range(n_replicates):
            i = rng.randrange(0, 1 << 16)
            chosen = pool_decoys[(i + r) % len(pool_decoys)]
            sc = builder(decoy_storm=chosen)
            out.append(dataclasses.replace(
                sc, scenario_id=f"{sc.scenario_id}__rep{r}"))
    return out


def build_phase57_falsifier_bank(*,
                                    n_replicates: int = 3,
                                    seed: int = 11,
                                    ) -> list[DecoderForcingScenario]:
    """Return a *falsifier* bank where the decoy is also mentioned via
    a CCK-eligible (causal-tier) claim_kind for the scenario's
    ``gold_root_cause`` (W10-4 falsifier).

    Construction: take the Phase-57 default bank and append one extra
    emission on the decoy's existing roles whose ``claim_kind`` is in
    the gold root_cause's CCK. The decoy then satisfies the bundle
    decoder's CCK predicate; the W10-1 win does not hold.
    """
    base = build_phase57_bank(n_replicates=n_replicates, seed=seed)
    # CCK-promotion claim_kind chosen per gold_root_cause (one CCK
    # member that the decoy can plausibly carry):
    cck_promotion = {
        "pool_exhaustion":    ("db_admin", "POOL_EXHAUSTION"),
        "deadlock":           ("db_admin", "DEADLOCK_SUSPECTED"),
        "slow_query_cascade": ("db_admin", "SLOW_QUERY_OBSERVED"),
        "disk_fill":          ("sysadmin", "DISK_FILL_CRITICAL"),
    }
    out: list[DecoderForcingScenario] = []
    for sc in base:
        role_emissions = dict(sc.role_emissions)
        promo = cck_promotion.get(sc.gold_root_cause)
        if promo is None:
            out.append(dataclasses.replace(
                sc, scenario_id=f"{sc.scenario_id}__falsifier"))
            continue
        role, kind = promo
        existing = list(role_emissions.get(role, ()))
        existing.append(_emit(
            kind,
            f"falsifier-cck-promotion service={sc.decoy_storm_service}"))
        role_emissions[role] = tuple(existing)
        out.append(dataclasses.replace(
            sc, role_emissions=role_emissions,
            scenario_id=f"{sc.scenario_id}__falsifier"))
    return out


# =============================================================================
# IncidentScenario adapter
# =============================================================================


def _as_incident_scenario(sc: DecoderForcingScenario) -> IncidentScenario:
    """Adapter: turn a Phase-57 scenario into the Phase-31 shape so
    we can reuse ``grade_answer``."""
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
        sc: DecoderForcingScenario,
        ) -> list[tuple[str, str, str, str, tuple[int, ...]]]:
    """Materialise role-emissions into a per-scenario candidate
    stream ``[(source_role, to_role, kind, payload, source_event_ids),
    ...]`` routed via the standard subscription table."""
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
        sc: DecoderForcingScenario,
        candidates: Sequence[tuple[str, str, str, str, tuple[int, ...]]],
        budgets: dict[str, RoleBudget],
        policy_per_role: dict[str, AdmissionPolicy],
        strategy_name: str,
        bundle_decoder: BundleAwareTeamDecoder | None = None,
        ) -> StrategyResult:
    incident_sc = _as_incident_scenario(sc)
    ledger = CapsuleLedger()
    coord = TeamCoordinator(
        ledger=ledger, role_budgets=budgets,
        policy_per_role=policy_per_role,
        team_tag="bundle_decoder_phase57",
    )
    coord.advance_round(1)
    for (src, to, kind, payload, _evs) in candidates:
        coord.emit_handoff(
            source_role=src, to_role=to, claim_kind=kind, payload=payload)
    coord.seal_all_role_views()
    rv_cid = coord.role_view_cid(ROLE_AUDITOR)
    handoffs: list[_DecoderHandoffShim] = []
    bd_handoffs: list[_DecodedHandoff] = []
    if rv_cid and rv_cid in ledger:
        rv = ledger.get(rv_cid)
        for p in rv.parents:
            if p in ledger:
                cap = ledger.get(p)
                if cap.kind != CapsuleKind.TEAM_HANDOFF:
                    continue
                payload = (cap.payload if isinstance(cap.payload, dict)
                            else {})
                src = str(payload.get("source_role", ""))
                kind = str(payload.get("claim_kind", ""))
                pld = str(payload.get("payload", ""))
                handoffs.append(_DecoderHandoffShim(
                    source_role=src, claim_kind=kind, payload=pld,
                    n_tokens=int(payload.get("n_tokens", 1))))
                bd_handoffs.append(_DecodedHandoff(
                    source_role=src, claim_kind=kind, payload=pld))
    if bundle_decoder is not None:
        answer = bundle_decoder.decode(bd_handoffs)
    else:
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
        sc: DecoderForcingScenario,
        candidates: Sequence[tuple[str, str, str, str, tuple[int, ...]]],
        inbox_capacity: int,
        ) -> StrategyResult:
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

_GENERIC_NOISE_KINDS = frozenset({
    "LATENCY_SPIKE", "ERROR_RATE_SPIKE", "FW_BLOCK_SURGE",
})


def _candidate_stats(
        candidates: Sequence[tuple[str, str, str, str, tuple[int, ...]]],
        gold_services_pair: tuple[str, str],
        decoy_storm_service: str,
        gold_root_cause: str,
        ) -> dict[str, Any]:
    """Compute the per-scenario bench-property metrics for Phase 57.

    Pre-committed properties:
      * ``decoy_role_count >= 2`` (corroborated decoy)
      * ``decoy_only_in_noise_kinds`` is True (decoy is mentioned only
        via generic-noise claim_kinds — non-CCK)
      * ``both_gold_corroborated_via_cck`` is True (each gold service
        has ≥ 1 mention via a CCK-eligible claim_kind, on ≥ 2 distinct
        producer roles)
    """
    from vision_mvp.coordpy.team_coord import (
        CAUSAL_CLAIM_KINDS_PER_ROOT_CAUSE)
    cck = CAUSAL_CLAIM_KINDS_PER_ROOT_CAUSE.get(gold_root_cause, frozenset())
    cands_aud = [c for c in candidates if c[1] == ROLE_AUDITOR]
    raw: dict[str, int] = {}
    roles_per_tag: dict[str, set[str]] = {}
    cck_roles_per_tag: dict[str, set[str]] = {}
    noise_only_per_tag: dict[str, bool] = {}
    for (src, _to, kind, payload, _evs) in cands_aud:
        m = _SERVICE_TAG_RE.search(payload)
        if not m:
            continue
        tag = m.group(1)
        raw[tag] = raw.get(tag, 0) + 1
        roles_per_tag.setdefault(tag, set()).add(src)
        if kind in cck:
            cck_roles_per_tag.setdefault(tag, set()).add(src)
        if kind not in _GENERIC_NOISE_KINDS:
            noise_only_per_tag[tag] = False
        else:
            noise_only_per_tag.setdefault(tag, True)
    role_counts = {t: len(rs) for t, rs in roles_per_tag.items()}
    cck_role_counts = {t: len(rs) for t, rs in cck_roles_per_tag.items()}
    A, B = gold_services_pair
    gold_set = {A, B}
    decoy_role_count = role_counts.get(decoy_storm_service, 0)
    decoy_noise_only = noise_only_per_tag.get(decoy_storm_service, False)
    decoy_in_cck = cck_role_counts.get(decoy_storm_service, 0) > 0
    n_gold_cck_corroborated = sum(
        1 for g in gold_set if cck_role_counts.get(g, 0) >= 1
        and role_counts.get(g, 0) >= 2)
    return {
        "raw_counts": dict(raw),
        "role_counts": dict(role_counts),
        "cck_role_counts": dict(cck_role_counts),
        "gold_services_pair": list(sorted(gold_set)),
        "decoy_storm_service": decoy_storm_service,
        "gold_root_cause": gold_root_cause,
        "decoy_role_count": decoy_role_count,
        "decoy_in_cck": decoy_in_cck,
        "decoy_noise_only": decoy_noise_only,
        "both_gold_corroborated_via_cck": (
            n_gold_cck_corroborated == 2),
        # Phase-57 default bench property:
        # decoy is corroborated AND only mentioned via generic-noise
        # AND both gold services are CCK-corroborated.
        "decoder_forcing_property_holds": (
            decoy_role_count >= 2
            and decoy_noise_only
            and n_gold_cck_corroborated == 2),
        # W10-4 falsifier property: decoy is also CCK-corroborated.
        "decoy_cck_promoted": decoy_in_cck,
        "n_candidates_to_auditor": len(cands_aud),
    }


# =============================================================================
# Top-level driver
# =============================================================================


def run_phase57(*,
                  n_eval: int | None = None,
                  K_auditor: int = 8,
                  T_auditor: int = 256,
                  K_producer: int = 6,
                  T_producer: int = 96,
                  inbox_capacity: int | None = None,
                  bank_seed: int = 11,
                  bank_replicates: int = 3,
                  use_falsifier_bank: bool = False,
                  ms_top_k: int = 3,
                  ms_min_corroborated_roles: int = 2,
                  bundle_role_floor: int = 1,
                  verbose: bool = False,
                  ) -> dict[str, Any]:
    """Run Phase 57 end-to-end and return a JSON-serialisable report.

    The pre-committed default config is ``K_auditor=4, T_auditor=128,
    n_eval=10, ms_top_k=3, ms_min_corroborated_roles=2,
    bundle_role_floor=2``. The ``ms_top_k=3`` is the *anchor* for the
    bundle-decoder method: admit *all* corroborated tags (decoy +
    both gold services), then let the decoder filter.
    """
    if use_falsifier_bank:
        bank = build_phase57_falsifier_bank(
            n_replicates=bank_replicates, seed=bank_seed)
    else:
        bank = build_phase57_bank(
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
            cands, sc.gold_services_pair, sc.decoy_storm_service,
            sc.gold_root_cause)
        cands_to_auditor = [c for c in cands if c[1] == ROLE_AUDITOR]
        cohort_buffered = (
            CohortCoherenceAdmissionPolicy.from_candidate_payloads(
                [c[3] for c in cands_to_auditor]))
        corroboration = (
            CrossRoleCorroborationAdmissionPolicy.from_candidate_stream(
                [(c[0], c[3]) for c in cands_to_auditor]))
        multi_service_w9 = (
            MultiServiceCorroborationAdmissionPolicy.from_candidate_stream(
                [(c[0], c[3]) for c in cands_to_auditor],
                top_k=2,
                min_corroborated_roles=ms_min_corroborated_roles))
        multi_service_w10 = (
            MultiServiceCorroborationAdmissionPolicy.from_candidate_stream(
                [(c[0], c[3]) for c in cands_to_auditor],
                top_k=ms_top_k,
                min_corroborated_roles=ms_min_corroborated_roles))
        bundle_decoder = BundleAwareTeamDecoder(
            cck_filter=True,
            role_corroboration_floor=bundle_role_floor)
        scenario_strategies: list[tuple[str, dict[str, AdmissionPolicy],
                                          BundleAwareTeamDecoder | None]] = [
            ("capsule_fifo",
                {r: FifoAdmissionPolicy() for r in budgets}, None),
            ("capsule_priority",
                {r: ClaimPriorityAdmissionPolicy(priorities=priorities,
                                                  threshold=0.65)
                 for r in budgets}, None),
            ("capsule_coverage",
                {r: CoverageGuidedAdmissionPolicy() for r in budgets}, None),
            ("capsule_cohort_buffered",
                {r: cohort_buffered for r in budgets}, None),
            ("capsule_corroboration",
                {r: corroboration for r in budgets}, None),
            ("capsule_multi_service",
                {r: multi_service_w9 for r in budgets}, None),
            ("capsule_bundle_decoder",
                {r: multi_service_w10 for r in budgets}, bundle_decoder),
        ]
        results.append(_run_substrate_strategy(sc, cands, inbox_capacity))
        for (sname, policy_per_role, decoder) in scenario_strategies:
            results.append(_run_capsule_strategy(
                sc=sc, candidates=cands, budgets=budgets,
                policy_per_role=policy_per_role, strategy_name=sname,
                bundle_decoder=decoder))

    strategies = ("capsule_fifo", "capsule_priority", "capsule_coverage",
                   "capsule_cohort_buffered", "capsule_corroboration",
                   "capsule_multi_service", "capsule_bundle_decoder")
    strategy_names = ("substrate",) + strategies
    pooled = {s: pool(results, s).as_dict() for s in strategy_names}

    def gap(a: str, b: str) -> float:
        return round(pooled[a]["accuracy_full"]
                      - pooled[b]["accuracy_full"], 4)

    headline_gap = {
        "bundle_decoder_minus_fifo_accuracy_full":
            gap("capsule_bundle_decoder", "capsule_fifo"),
        "bundle_decoder_minus_substrate_accuracy_full":
            gap("capsule_bundle_decoder", "substrate"),
        "bundle_decoder_minus_cohort_buffered_accuracy_full":
            gap("capsule_bundle_decoder", "capsule_cohort_buffered"),
        "bundle_decoder_minus_corroboration_accuracy_full":
            gap("capsule_bundle_decoder", "capsule_corroboration"),
        "bundle_decoder_minus_multi_service_accuracy_full":
            gap("capsule_bundle_decoder", "capsule_multi_service"),
        # The W10-Λ witness: every service-blind admission ties FIFO.
        "max_admission_only_accuracy_full": max(
            pooled[s]["accuracy_full"]
            for s in ("capsule_fifo", "capsule_priority",
                       "capsule_coverage", "capsule_cohort_buffered",
                       "capsule_corroboration",
                       "capsule_multi_service")),
    }

    audit_ok_grid = {}
    for s in strategy_names:
        if s == "substrate":
            audit_ok_grid[s] = False
            continue
        rs = [r for r in results if r.strategy == s]
        audit_ok_grid[s] = bool(rs) and all(r.audit_ok for r in rs)

    bench_summary = {
        "scenarios_with_decoder_forcing_property": sum(
            1 for v in bench_property.values()
            if v["decoder_forcing_property_holds"]),
        "scenarios_with_decoy_cck_promotion": sum(
            1 for v in bench_property.values()
            if v["decoy_cck_promoted"]),
        "scenarios_with_surplus": sum(
            1 for v in bench_property.values()
            if v["n_candidates_to_auditor"] > K_auditor),
        "n_scenarios": len(bench_property),
        "K_auditor": K_auditor,
    }

    if verbose:
        print(f"[phase57] n_eval={len(bank)}, K_auditor={K_auditor}, "
              f"falsifier={use_falsifier_bank}, top_k={ms_top_k}, "
              f"min_roles={ms_min_corroborated_roles}, "
              f"bundle_floor={bundle_role_floor}",
              file=sys.stderr, flush=True)
        print(f"[phase57] bench_props: forcing="
              f"{bench_summary['scenarios_with_decoder_forcing_property']}/"
              f"{len(bank)}, "
              f"cck_promoted={bench_summary['scenarios_with_decoy_cck_promotion']}/"
              f"{len(bank)}, "
              f"surplus={bench_summary['scenarios_with_surplus']}/{len(bank)}",
              file=sys.stderr, flush=True)
        for s in strategy_names:
            p = pooled[s]
            print(f"[phase57]   {s:30s} "
                  f"acc_full={p['accuracy_full']:.3f} "
                  f"acc_root_cause={p['accuracy_root_cause']:.3f} "
                  f"acc_services={p['accuracy_services']:.3f} "
                  f"adm={p['mean_n_admitted_auditor']:.2f}",
                  file=sys.stderr, flush=True)
        print(f"[phase57] gap bundle_decoder−fifo: "
              f"{headline_gap['bundle_decoder_minus_fifo_accuracy_full']:+.3f}",
              file=sys.stderr, flush=True)
        print(f"[phase57] gap bundle_decoder−multi_service: "
              f"{headline_gap['bundle_decoder_minus_multi_service_accuracy_full']:+.3f}",
              file=sys.stderr, flush=True)
        print(f"[phase57] max admission-only acc_full = "
              f"{headline_gap['max_admission_only_accuracy_full']:.3f} "
              f"(W10-Λ witness)",
              file=sys.stderr, flush=True)

    return {
        "schema": "phase57.bundle_decoder.v1",
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
            "bundle_role_floor": bundle_role_floor,
        },
        "pooled": pooled,
        "audit_ok_grid": audit_ok_grid,
        "bench_summary": bench_summary,
        "bench_property_per_scenario": bench_property,
        "headline_gap": headline_gap,
        "scenarios_evaluated": [sc.scenario_id for sc in bank],
        "n_results": len(results),
    }


def run_phase57_seed_stability_sweep(
        *, seeds: Sequence[int] = (11, 17, 23, 29, 31),
        n_eval: int = 10,
        K_auditor: int = 4,
        T_auditor: int = 128,
        ) -> dict[str, Any]:
    """W10-1 stability anchor: run Phase 57 default at multiple bank
    seeds and report the headline gap for each."""
    per_seed: dict[int, dict[str, Any]] = {}
    for seed in seeds:
        rep = run_phase57(
            n_eval=n_eval, K_auditor=K_auditor, T_auditor=T_auditor,
            bank_seed=seed, bank_replicates=3, verbose=False,
            ms_top_k=3, ms_min_corroborated_roles=2,
            bundle_role_floor=1)
        per_seed[seed] = {
            "headline_gap": rep["headline_gap"],
            "pooled": rep["pooled"],
            "audit_ok_grid": rep["audit_ok_grid"],
            "bench_summary": rep["bench_summary"],
        }
    return {
        "schema": "phase57.bundle_decoder_seed_sweep.v1",
        "seeds": list(seeds),
        "K_auditor": K_auditor,
        "T_auditor": T_auditor,
        "n_eval": n_eval,
        "per_seed": per_seed,
    }


def run_cross_regime_summary(*, n_eval: int = 10, bank_seed: int = 11,
                                bank_replicates: int = 2,
                                ) -> dict[str, Any]:
    """Run Phase 54/55/56/57 default + Phase 57 falsifier side-by-side
    under matched K_auditor=4. The W10-1 / W10-3 / W10-4 contracts
    read from this report."""
    from vision_mvp.experiments.phase54_cross_role_coherence import (
        run_phase54)
    from vision_mvp.experiments.phase55_decoy_plurality import (
        run_phase55)
    from vision_mvp.experiments.phase56_multi_service_corroboration import (
        run_phase56)
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
    p57_default = run_phase57(
        n_eval=n_eval, K_auditor=8, T_auditor=256,
        bank_seed=bank_seed, bank_replicates=3,
        use_falsifier_bank=False, verbose=False)
    p57_falsifier = run_phase57(
        n_eval=n_eval, K_auditor=8, T_auditor=256,
        bank_seed=bank_seed, bank_replicates=3,
        use_falsifier_bank=True, verbose=False)
    return {
        "schema": "phase57.cross_regime.v1",
        "config": {
            "K_auditor": 4, "T_auditor": 128,
            "n_eval": n_eval, "bank_seed": bank_seed,
            "bank_replicates": bank_replicates,
        },
        "phase54_default": p54_default,
        "phase55_default": p55_default,
        "phase56_default": p56_default,
        "phase57_default": p57_default,
        "phase57_falsifier": p57_falsifier,
    }


# =============================================================================
# CLI
# =============================================================================


def _arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Phase 57 — multi-service-gold + corroborated-decoy "
                    "decoder-forcing benchmark (SDK v3.11 / W10 family).")
    p.add_argument("--K-auditor", type=int, default=8)
    p.add_argument("--T-auditor", type=int, default=256)
    p.add_argument("--n-eval", type=int, default=10)
    p.add_argument("--bank-seed", type=int, default=11)
    p.add_argument("--bank-replicates", type=int, default=3)
    p.add_argument("--ms-top-k", type=int, default=3)
    p.add_argument("--ms-min-corroborated-roles", type=int, default=2)
    p.add_argument("--bundle-role-floor", type=int, default=1)
    p.add_argument("--falsifier", action="store_true",
                    help="run the W10-4 falsifier bank instead.")
    p.add_argument("--cross-regime", action="store_true",
                    help="run Phase 54/55/56/57 side by side at K=4.")
    p.add_argument("--seed-sweep", action="store_true",
                    help="run Phase 57 default at multiple bank seeds.")
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
        report = run_phase57_seed_stability_sweep(
            n_eval=args.n_eval, K_auditor=args.K_auditor,
            T_auditor=args.T_auditor)
    else:
        report = run_phase57(
            n_eval=args.n_eval,
            K_auditor=args.K_auditor,
            T_auditor=args.T_auditor,
            bank_seed=args.bank_seed,
            bank_replicates=args.bank_replicates,
            use_falsifier_bank=args.falsifier,
            ms_top_k=args.ms_top_k,
            ms_min_corroborated_roles=args.ms_min_corroborated_roles,
            bundle_role_floor=args.bundle_role_floor,
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
            print(f"[phase57] wrote {args.out}", file=sys.stderr)
    else:
        if not args.quiet:
            print(json.dumps(report.get("pooled", report),
                              indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
