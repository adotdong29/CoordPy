"""Phase 54 — cross-role cohort-coherence multi-agent benchmark.

This is the SDK v3.8 reference benchmark. It directly attacks the
*main scientific weakness* surfaced by SDK v3.7's Phase-53 result:
**at the Phase-53 default config (K_auditor=4, real-LLM producer
extractor), every fixed admission policy is trivially equivalent
to "admit all" because the candidate stream is below the budget cap
(mean_n_admitted_auditor ∈ {2.2, 3.0, 3.2} < K_auditor=4 in every
regime).** With no budget pressure, FIFO necessarily ties or beats
every other admission policy by construction. Phase 53 was therefore
not a fair test of whether cross-role / cohort / relational
admission can ever help.

This phase fixes the test, not the answer
=========================================

Phase 54 builds the **smallest deterministic multi-agent benchmark
where structure can plausibly matter**:

* Each scenario has one ``real_service`` (the gold target) and 1–2
  ``decoy_services`` (benign services that share the role's allowed
  claim kinds).
* Each producer role emits **2–3 candidates**: at least one gold
  (with ``service=real``) and at least one decoy (with
  ``service=decoy_X``). Decoys are emitted *first* in role-local
  order, mirroring the realistic "background telemetry has been
  there longer than the incident" distribution.
* The auditor's K_role budget is **tight** (default ``K_auditor=4``)
  while the candidate stream per scenario is **8–12** candidates —
  so the budget *bites* and the admission policy must select.
* The decoy claims are **locally clean**: they pass kind whitelists,
  carry well-formed payloads, and have a single
  ``service=<decoy_tag>`` token. The only thing distinguishing
  them from gold candidates is the **service tag** — a relational
  signal that no per-candidate fixed policy can exploit.

This is **not benchmark shopping** for a structural win:

* Decoys are realistic role-local background, not adversarial
  injections. Every real-world incident-response role sees telemetry
  from many services, only one of which is causal in any given
  incident. (A monitor role sees latency on api, web, archival,
  catalog, etc., simultaneously.)
* The gold answer is achievable by *any* admission policy that can
  reason about the cohort signature; FIFO simply lacks the
  vocabulary to do so.
* The success condition is meaningful and falsifiable: cohort
  coherence must beat FIFO on ``accuracy_full`` by ``≥ 0.20`` pooled
  over the eval bank, *and* the lifecycle audit must hold (T-1..T-7
  unchanged).

Theorem cross-reference (W7 family — proposed)
==============================================

This benchmark anchors the empirical evidence for:

* **W7-1** (proposed proved-empirical, FIFO unbeatability under low
  surplus): when the producer pre-filters such that
  ``|candidates(scenario)| ≤ K_role``, every fixed admission policy
  is permutation-equivalent to FIFO; structure_gain = 0 by
  construction. Falsifier: a regime where ``|candidates| ≤ K_role``
  yet some admission policy strictly outperforms FIFO. (Phase-53 is
  the empirical anchor in the *positive* direction: fixed strategies
  tie at 0.800 across all regimes when |candidates| < K_auditor.)

* **W7-2** (proposed proved-empirical, structure win condition):
  when the producer emits role-complementary evidence with foreign-
  service decoys *and* ``|candidates| > K_role``, cohort-coherence
  admission strictly outperforms FIFO. Falsifier: a Phase-54 default
  config where ``capsule_cohort_coherence_acc_full ≤
  capsule_fifo_acc_full + 0.05``. (Phase-54 is the empirical anchor
  in the *negative* direction; if cohort coherence does *not* clear
  this bar, W7-2 is falsified and the audit-only reading from
  SDK v3.7 stands.)

* **W7-3** (proposed proved-negative, extraction floor): no
  per-role admission policy can recover a missing causal claim
  that the producer never emitted. (This specialises W4-3 to the
  "producer omission" case and isolates admission-fixable failures
  from extraction-fixable ones.)

Honest scope
============

* The benchmark is **deterministic** by design — the candidate
  streams are produced by code, not by an LLM. This is the right
  knob to isolate **admission/decoder coordination** from
  **producer extraction quality**. Phase 53 already measured the
  real-LLM regime; the inferences from this phase plug back into
  the real-LLM picture as "what could the auditor do *if* the
  producer were noisier?".
* The new admission policy ``CohortCoherenceAdmissionPolicy`` is
  small and interpretable (one regex, one counter); it does not
  require training. This is intentional: a learning move would
  re-introduce the OOD generalisation risk that bit the SDK v3.5
  learned policy under SDK v3.7.
* The bench is small (default ``n_eval=10``). Saturation is
  acceptable; the hypothesis is *direction* (cohort-coherence
  beats FIFO at appropriate K), not magnitude.

CLI
---

::

    python3 -m vision_mvp.experiments.phase54_cross_role_coherence \
        --K-auditor 4 --n-eval 10 \
        --out /tmp/coordpy-distributed/phase54_cross_role_coherence_K4.json

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
    IncidentScenario, IncidentEvent,
    build_role_subscriptions, grade_answer,
    _decoder_from_handoffs as _phase31_decoder_from_handoffs,
)
from vision_mvp.coordpy.capsule import CapsuleKind, CapsuleLedger
from vision_mvp.coordpy.team_coord import (
    AdmissionPolicy, ClaimPriorityAdmissionPolicy,
    CohortCoherenceAdmissionPolicy,
    CoverageGuidedAdmissionPolicy, FifoAdmissionPolicy,
    RoleBudget, TeamCoordinator, audit_team_lifecycle,
)
from vision_mvp.experiments.phase52_team_coord import (
    PooledStrategy, StrategyResult, _format_canonical_answer,
    claim_priorities, make_team_budgets, pool,
)


# =============================================================================
# Phase 54 scenario family — deterministic, role-complementary, decoy-rich
# =============================================================================


_KNOWN_SERVICES = (
    "api", "web", "orders", "catalog", "users", "sessions",
    "archival", "backup", "cache", "logs",
)


@dataclasses.dataclass(frozen=True)
class CrossRoleScenario:
    """One Phase-54 scenario.

    Fields
    ------
    scenario_id
        Short string key.
    description
        Human-readable one-liner.
    real_service
        The gold target service. Every causal candidate carries
        ``service=<real_service>``.
    decoy_services
        Tuple of foreign service tags. Each producer role may emit
        decoys carrying one of these tags (typically the first one).
    gold_root_cause
        Canonical root-cause label produced by the
        :func:`_phase31_decoder_from_handoffs` priority table for
        the gold cohort's claim_kinds.
    gold_remediation
        Remediation string keyed by ``gold_root_cause``.
    role_emissions
        ``role -> [(claim_kind, payload), ...]`` ordered list of
        what each producer role emits, in role-local order. By
        convention the *first* entry per role is a decoy and the
        rest are gold; this models "background has been there
        longer than the incident".
    """

    scenario_id: str
    description: str
    real_service: str
    decoy_services: tuple[str, ...]
    gold_root_cause: str
    gold_remediation: str
    role_emissions: dict[str, tuple[tuple[str, str], ...]]


# Closed-vocabulary remediation map keyed by gold_root_cause; mirrors
# the Phase-31 ``_decoder_from_handoffs`` priority table so the
# auditor's deterministic decoder produces the gold remediation when
# the gold cohort is admitted cleanly.
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


def _emit(role: str, kind: str, payload: str) -> tuple[str, str]:
    return (kind, payload)


def _build_scenario_oom_api(decoy: str = "archival") -> CrossRoleScenario:
    """OOM kill on api app; gold has plurality (4 gold vs 2 decoy)."""
    real = "api"
    return CrossRoleScenario(
        scenario_id="oom_api",
        description=(
            f"OOM kill on {real} app; concurrent benign cron {decoy} "
            f"chatter. Gold: memory_leak/{real}. Gold plurality = 2."),
        real_service=real,
        decoy_services=(decoy,),
        gold_root_cause="memory_leak",
        gold_remediation=_REMEDIATION["memory_leak"],
        role_emissions={
            ROLE_MONITOR: (
                _emit(ROLE_MONITOR, "LATENCY_SPIKE",
                       f"p95_ms=1100 service={decoy}"),
                _emit(ROLE_MONITOR, "LATENCY_SPIKE",
                       f"p95_ms=4700 service={real}"),
            ),
            ROLE_DB_ADMIN: (
                _emit(ROLE_DB_ADMIN, "SLOW_QUERY_OBSERVED",
                       f"q#12 mean_ms=4210 service={real}"),
                _emit(ROLE_DB_ADMIN, "POOL_EXHAUSTION",
                       f"pool active=200/200 waiters=145 service={real}"),
            ),
            ROLE_SYSADMIN: (
                _emit(ROLE_SYSADMIN, "OOM_KILL",
                       f"oom_kill comm=cron rss=2.1G service={decoy}"),
                _emit(ROLE_SYSADMIN, "OOM_KILL",
                       f"oom_kill comm=app.py rss=8.1G service={real}"),
            ),
            ROLE_NETWORK: (),
        },
    )


def _build_scenario_disk_with_backup_decoy(
        decoy: str = "backup") -> CrossRoleScenario:
    """Cron-driven /var/log fill cascading into pool exhaustion;
    gold has plurality (5 gold vs 2 decoy)."""
    real = "orders"
    return CrossRoleScenario(
        scenario_id="disk_fill_orders",
        description=(
            f"Cron-driven /var/log fill cascading into {real} pool "
            f"exhaustion; concurrent benign {decoy} chatter. Gold "
            f"plurality = 3."),
        real_service=real,
        decoy_services=(decoy,),
        gold_root_cause="disk_fill",
        gold_remediation=_REMEDIATION["disk_fill"],
        role_emissions={
            ROLE_MONITOR: (
                _emit(ROLE_MONITOR, "ERROR_RATE_SPIKE",
                       f"error_rate=0.12 service={decoy}"),
                _emit(ROLE_MONITOR, "ERROR_RATE_SPIKE",
                       f"error_rate=0.41 service={real}"),
            ),
            ROLE_DB_ADMIN: (
                _emit(ROLE_DB_ADMIN, "POOL_EXHAUSTION",
                       f"pool active=200/200 waiters=145 service={real}"),
                _emit(ROLE_DB_ADMIN, "SLOW_QUERY_OBSERVED",
                       f"q#12 mean_ms=4210 service={real}"),
            ),
            ROLE_SYSADMIN: (
                _emit(ROLE_SYSADMIN, "DISK_FILL_CRITICAL",
                       f"/srv/{decoy} used=72% fs=/data"),
                _emit(ROLE_SYSADMIN, "DISK_FILL_CRITICAL",
                       f"/var/log used=99% fs=/ service={real}"),
                _emit(ROLE_SYSADMIN, "CRON_OVERRUN",
                       f"backup.sh exit=137 duration_s=5400 service={real}"),
            ),
            ROLE_NETWORK: (),
        },
    )


def _build_scenario_tls_with_cache_decoy(
        decoy: str = "cache") -> CrossRoleScenario:
    """Expired TLS cert; gold has plurality (3 gold vs 2 decoy).

    No DISK_FILL_CRITICAL emission anywhere — that kind would be
    higher in the decoder's root-cause priority list than
    TLS_EXPIRED, so any DISK_FILL appearance (even with the gold
    service tag) would derail the decoder. The benchmark only tests
    cohort coherence, not decoder priority interactions.
    """
    real = "api"
    return CrossRoleScenario(
        scenario_id="tls_expiry_api",
        description=(
            f"Expired TLS cert on {real} drives healthcheck loop; "
            f"benign {decoy} chatter is concurrent. Gold "
            f"plurality = 1."),
        real_service=real,
        decoy_services=(decoy,),
        gold_root_cause="tls_expiry",
        gold_remediation=_REMEDIATION["tls_expiry"],
        role_emissions={
            ROLE_MONITOR: (
                _emit(ROLE_MONITOR, "ERROR_RATE_SPIKE",
                       f"uptime_pct=92 service={decoy}"),
                _emit(ROLE_MONITOR, "ERROR_RATE_SPIKE",
                       f"uptime_pct=41 service={real}"),
            ),
            ROLE_NETWORK: (
                _emit(ROLE_NETWORK, "TLS_EXPIRED",
                       f"tls handshake fail reason=expired "
                       f"service={real}"),
                _emit(ROLE_NETWORK, "FW_BLOCK_SURGE",
                       f"rule=deny src=hc-probe count=1200 "
                       f"service={real}"),
            ),
            ROLE_DB_ADMIN: (
                _emit(ROLE_DB_ADMIN, "SLOW_QUERY_OBSERVED",
                       f"q#3 mean_ms=900 service={decoy}"),
            ),
            ROLE_SYSADMIN: (),
        },
    )


def _build_scenario_dns_with_users_decoy(
        decoy: str = "users") -> CrossRoleScenario:
    """DNS misroute cascade; gold has plurality (4 gold vs 2 decoy)."""
    real = "orders"
    return CrossRoleScenario(
        scenario_id="dns_misroute_orders",
        description=(
            f"DNS misroute on db.internal cascades to {real} "
            f"reconnect storm; benign {decoy} chatter is concurrent. "
            f"Gold plurality = 2."),
        real_service=real,
        decoy_services=(decoy,),
        gold_root_cause="dns_misroute",
        gold_remediation=_REMEDIATION["dns_misroute"],
        role_emissions={
            ROLE_NETWORK: (
                _emit(ROLE_NETWORK, "DNS_MISROUTE",
                       f"q={decoy}.internal rc=NOERROR rtt_ms=4"),
                _emit(ROLE_NETWORK, "DNS_MISROUTE",
                       f"q=db.internal rc=SERVFAIL service={real}"),
            ),
            ROLE_DB_ADMIN: (
                _emit(ROLE_DB_ADMIN, "POOL_EXHAUSTION",
                       f"pool active=0/200 reconnect_attempts=842 "
                       f"service={real}"),
                _emit(ROLE_DB_ADMIN, "SLOW_QUERY_OBSERVED",
                       f"q#7 mean_ms=2400 service={real}"),
            ),
            ROLE_MONITOR: (
                _emit(ROLE_MONITOR, "ERROR_RATE_SPIKE",
                       f"error_rate=0.14 service={decoy}"),
                _emit(ROLE_MONITOR, "ERROR_RATE_SPIKE",
                       f"error_rate=0.88 service={real}"),
            ),
            ROLE_SYSADMIN: (),
        },
    )


def _build_scenario_deadlock_with_logs_decoy(
        decoy: str = "logs") -> CrossRoleScenario:
    """Deadlock storm; gold has plurality (4 gold vs 2 decoy)."""
    real = "orders"
    return CrossRoleScenario(
        scenario_id="deadlock_orders",
        description=(
            f"Lock-order bug on {real}_payments triggers deadlock "
            f"storm; benign {decoy} chatter is concurrent. Gold "
            f"plurality = 2."),
        real_service=real,
        decoy_services=(decoy,),
        gold_root_cause="deadlock",
        gold_remediation=_REMEDIATION["deadlock"],
        role_emissions={
            ROLE_DB_ADMIN: (
                _emit(ROLE_DB_ADMIN, "SLOW_QUERY_OBSERVED",
                       f"q#3 mean_ms=1100 service={decoy}"),
                _emit(ROLE_DB_ADMIN, "DEADLOCK_SUSPECTED",
                       f"deadlock relation={real}_payments service={real}"),
                _emit(ROLE_DB_ADMIN, "POOL_EXHAUSTION",
                       f"pool active=200/200 waiters=88 service={real}"),
            ),
            ROLE_MONITOR: (
                _emit(ROLE_MONITOR, "ERROR_RATE_SPIKE",
                       f"error_rate=0.11 service={decoy}"),
                _emit(ROLE_MONITOR, "ERROR_RATE_SPIKE",
                       f"error_rate=0.22 service={real}"),
            ),
            ROLE_SYSADMIN: (),
            ROLE_NETWORK: (),
        },
    )


# Five "base" scenarios under five real services + decoy services.
# Replicating each across (decoy_choice × seed) gives a small but
# multi-instance bench at the cost of being deterministic.
_BASE_BUILDERS = (
    _build_scenario_oom_api,
    _build_scenario_disk_with_backup_decoy,
    _build_scenario_tls_with_cache_decoy,
    _build_scenario_dns_with_users_decoy,
    _build_scenario_deadlock_with_logs_decoy,
)


def build_phase54_bank(*,
                         n_replicates: int = 2,
                         seed: int = 11,
                         ) -> list[CrossRoleScenario]:
    """Return a deterministic Phase-54 scenario bank.

    Each base scenario is replicated ``n_replicates`` times under a
    deterministic permutation of the decoy_service token. With the
    default ``n_replicates=2`` and 5 base builders, this yields a
    10-scenario bank.
    """
    rng = random.Random(seed)
    decoys_pool = [s for s in _KNOWN_SERVICES if s not in
                    ("api", "orders")]
    out: list[CrossRoleScenario] = []
    for builder in _BASE_BUILDERS:
        for r in range(n_replicates):
            chosen_decoy = decoys_pool[(rng.randrange(0, 1 << 16) + r)
                                          % len(decoys_pool)]
            sc = builder(decoy=chosen_decoy)
            out.append(dataclasses.replace(
                sc, scenario_id=f"{sc.scenario_id}__rep{r}__decoy_{chosen_decoy}"))
    return out


# =============================================================================
# IncidentScenario adapter — convert a CrossRoleScenario into the
# tagged scenario shape the existing ``grade_answer`` accepts
# =============================================================================


def _as_incident_scenario(sc: CrossRoleScenario) -> IncidentScenario:
    """Adapter: turn a Phase-54 ``CrossRoleScenario`` into the
    Phase-31 ``IncidentScenario`` shape so we can reuse
    ``grade_answer`` and the ``StrategyResult`` plumbing.

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
# Candidate-handoff stream from a CrossRoleScenario
# =============================================================================


def build_candidate_stream(
        sc: CrossRoleScenario,
        ) -> list[tuple[str, str, str, str, tuple[int, ...]]]:
    """Materialise the role-emissions into a per-scenario candidate
    stream ``[(source_role, to_role, kind, payload, source_event_ids),
    ...]`` routed via the standard subscription table.

    Order is **role-major, then within-role original order** (decoy
    first, gold second, by Phase-54 convention). This mirrors the
    Phase-52/53 stream-construction semantics.
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
        sc: CrossRoleScenario,
        candidates: Sequence[tuple[str, str, str, str, tuple[int, ...]]],
        budgets: dict[str, RoleBudget],
        policy_per_role: dict[str, AdmissionPolicy],
        strategy_name: str,
        ) -> StrategyResult:
    """Drive one (scenario, candidates, policy) combo and return a
    StrategyResult. Uses the standard ``TeamCoordinator`` admission
    path so the team-lifecycle audit (T-1..T-7) is exercised on every
    cell."""
    incident_sc = _as_incident_scenario(sc)
    ledger = CapsuleLedger()
    coord = TeamCoordinator(
        ledger=ledger, role_budgets=budgets,
        policy_per_role=policy_per_role,
        team_tag="cross_role_coherence",
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
        sc: CrossRoleScenario,
        candidates: Sequence[tuple[str, str, str, str, tuple[int, ...]]],
        inbox_capacity: int,
        ) -> StrategyResult:
    """Drive the Phase-31 typed-handoff substrate at matched inbox
    capacity. The substrate inbox is FIFO with deduplication on
    handoff identity; with K_auditor=4 and ~10 candidates the
    substrate admits the first 4 that arrive."""
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
# Top-level driver
# =============================================================================


def run_phase54(*,
                  n_eval: int | None = None,
                  K_auditor: int = 4,
                  T_auditor: int = 128,
                  K_producer: int = 6,
                  T_producer: int = 96,
                  inbox_capacity: int | None = None,
                  bank_seed: int = 11,
                  bank_replicates: int = 2,
                  verbose: bool = False,
                  ) -> dict[str, Any]:
    """Run Phase 54 end-to-end and return a JSON-serialisable report.

    The pre-committed default config is ``K_auditor=4, T_auditor=128,
    n_eval=10`` — do not retroactively widen.
    """
    bank = build_phase54_bank(n_replicates=bank_replicates, seed=bank_seed)
    if n_eval is not None:
        bank = bank[:n_eval]
    if inbox_capacity is None:
        inbox_capacity = K_auditor

    budgets = make_team_budgets(
        K_producer=K_producer, T_producer=T_producer,
        K_auditor=K_auditor, T_auditor=T_auditor)
    priorities = claim_priorities()

    # Per-scenario admission strategies. The buffered cohort policy
    # is pre-fitted on the auditor-routed candidate stream of each
    # scenario; constructing it inside the loop makes the dependency
    # explicit (W7-2 anchor: pre-fit before streaming admission).
    results: list[StrategyResult] = []
    surplus_per_scenario: dict[str, int] = {}
    candidate_count_per_scenario: dict[str, int] = {}
    for sc in bank:
        cands = build_candidate_stream(sc)
        cands_to_auditor = [c for c in cands if c[1] == ROLE_AUDITOR]
        candidate_count_per_scenario[sc.scenario_id] = len(cands_to_auditor)
        surplus_per_scenario[sc.scenario_id] = max(
            0, len(cands_to_auditor) - K_auditor)
        # Pre-fit the buffered cohort policy on the auditor stream.
        cohort_buffered = CohortCoherenceAdmissionPolicy.from_candidate_payloads(
            [c[3] for c in cands_to_auditor])
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
            # Streaming cohort baseline — anchors W7-1-aux limitation.
            "capsule_cohort_streaming": {
                r: CohortCoherenceAdmissionPolicy() for r in budgets
            },
            # Buffered cohort policy — anchors W7-2.
            "capsule_cohort_buffered": {
                r: cohort_buffered for r in budgets
            },
        }
        results.append(_run_substrate_strategy(sc, cands, inbox_capacity))
        for sname, policy_per_role in scenario_strategies.items():
            results.append(_run_capsule_strategy(
                sc=sc, candidates=cands, budgets=budgets,
                policy_per_role=policy_per_role, strategy_name=sname))
    # Strategy name closure for pooling and audit_ok grid below.
    strategies = {
        "capsule_fifo": None,
        "capsule_priority": None,
        "capsule_coverage": None,
        "capsule_cohort_streaming": None,
        "capsule_cohort_buffered": None,
    }

    strategy_names = ("substrate",) + tuple(strategies.keys())
    pooled = {s: pool(results, s).as_dict() for s in strategy_names}

    # Headline gap: cohort_buffered vs fifo + cohort_streaming vs fifo.
    gap_full = round(
        pooled["capsule_cohort_buffered"]["accuracy_full"]
        - pooled["capsule_fifo"]["accuracy_full"], 4)
    gap_root_cause = round(
        pooled["capsule_cohort_buffered"]["accuracy_root_cause"]
        - pooled["capsule_fifo"]["accuracy_root_cause"], 4)
    gap_services = round(
        pooled["capsule_cohort_buffered"]["accuracy_services"]
        - pooled["capsule_fifo"]["accuracy_services"], 4)
    gap_vs_substrate = round(
        pooled["capsule_cohort_buffered"]["accuracy_full"]
        - pooled["substrate"]["accuracy_full"], 4)
    gap_streaming_full = round(
        pooled["capsule_cohort_streaming"]["accuracy_full"]
        - pooled["capsule_fifo"]["accuracy_full"], 4)

    audit_ok_grid = {}
    for s in strategy_names:
        if s == "substrate":
            audit_ok_grid[s] = False
            continue
        rs = [r for r in results if r.strategy == s]
        audit_ok_grid[s] = bool(rs) and all(r.audit_ok for r in rs)

    surplus_summary = {
        "mean_candidates_to_auditor": round(
            sum(candidate_count_per_scenario.values())
            / max(1, len(candidate_count_per_scenario)), 3),
        "max_candidates_to_auditor":
            max(candidate_count_per_scenario.values(), default=0),
        "min_candidates_to_auditor":
            min(candidate_count_per_scenario.values(), default=0),
        "scenarios_above_K": sum(
            1 for v in candidate_count_per_scenario.values() if v > K_auditor),
        "K_auditor": K_auditor,
        "n_scenarios": len(candidate_count_per_scenario),
    }

    if verbose:
        print(f"[phase54] n_eval={len(bank)}, K_auditor={K_auditor}, "
              f"surplus={surplus_summary['scenarios_above_K']}/"
              f"{len(bank)} above K", file=sys.stderr, flush=True)
        for s in strategy_names:
            p = pooled[s]
            print(f"[phase54]   {s:25s} acc_full={p['accuracy_full']:.3f} "
                  f"adm={p['mean_n_admitted_auditor']:.2f}",
                  file=sys.stderr, flush=True)
        print(f"[phase54] cohort_buffered − fifo: full={gap_full:+.3f} "
              f"root_cause={gap_root_cause:+.3f} services={gap_services:+.3f}",
              file=sys.stderr, flush=True)
        print(f"[phase54] cohort_streaming − fifo: full={gap_streaming_full:+.3f}",
              file=sys.stderr, flush=True)

    return {
        "schema": "phase54.cross_role_coherence.v1",
        "config": {
            "n_eval": len(bank),
            "K_auditor": K_auditor,
            "T_auditor": T_auditor,
            "K_producer": K_producer,
            "T_producer": T_producer,
            "inbox_capacity": inbox_capacity,
            "bank_seed": bank_seed,
            "bank_replicates": bank_replicates,
        },
        "pooled": pooled,
        "audit_ok_grid": audit_ok_grid,
        "surplus_summary": surplus_summary,
        "candidate_count_per_scenario": candidate_count_per_scenario,
        "headline_gap": {
            "cohort_buffered_minus_fifo_accuracy_full": gap_full,
            "cohort_buffered_minus_fifo_accuracy_root_cause":
                gap_root_cause,
            "cohort_buffered_minus_fifo_accuracy_services": gap_services,
            "cohort_buffered_minus_substrate_accuracy_full":
                gap_vs_substrate,
            "cohort_streaming_minus_fifo_accuracy_full":
                gap_streaming_full,
        },
        "scenarios_evaluated": [sc.scenario_id for sc in bank],
        "n_results": len(results),
    }


def run_phase54_budget_sweep(*,
                              K_values: Sequence[int] = (2, 3, 4, 5, 6, 8),
                              n_eval: int = 10,
                              bank_seed: int = 11,
                              bank_replicates: int = 2,
                              ) -> dict[str, Any]:
    """W7-2 falsifier search: sweep ``K_auditor`` to identify the
    regime where buffered cohort coherence strictly outperforms FIFO.

    Three regimes are expected (W7-2-conditional):

    * **K too tight** (``K < gold_count``): cohort and FIFO both
      fail because the gold cohort doesn't fit; no admission policy
      can rescue.
    * **K in the structure-win window**
      (``gold_count ≤ K < total_candidates``): cohort beats FIFO
      because admission must select between gold and decoys.
    * **K too loose** (``K ≥ total_candidates``): cohort and FIFO
      tie because no admission is needed.

    Returns: ``{schema, K_values, pooled_per_K, n_eval}``.
    """
    pooled_per_K: dict[int, dict[str, Any]] = {}
    headline_gap_per_K: dict[int, dict[str, float]] = {}
    audit_ok_per_K: dict[int, dict[str, bool]] = {}
    for K in K_values:
        rep = run_phase54(
            n_eval=n_eval, K_auditor=K, T_auditor=max(64, 32 * K),
            inbox_capacity=K, bank_seed=bank_seed,
            bank_replicates=bank_replicates, verbose=False)
        pooled_per_K[K] = rep["pooled"]
        headline_gap_per_K[K] = rep["headline_gap"]
        audit_ok_per_K[K] = rep["audit_ok_grid"]
    return {
        "schema": "phase54.cross_role_coherence_budget_sweep.v1",
        "K_values": list(K_values),
        "n_eval": n_eval,
        "pooled_per_K": pooled_per_K,
        "headline_gap_per_K": headline_gap_per_K,
        "audit_ok_per_K": audit_ok_per_K,
    }


# =============================================================================
# CLI
# =============================================================================


def _arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Phase 54 — cross-role cohort-coherence multi-agent "
                    "benchmark.")
    p.add_argument("--K-auditor", type=int, default=4,
                    help="auditor's K_role (max admitted handoffs).")
    p.add_argument("--T-auditor", type=int, default=128,
                    help="auditor's T_role (max admitted token total).")
    p.add_argument("--n-eval", type=int, default=10,
                    help="number of eval scenarios.")
    p.add_argument("--bank-seed", type=int, default=11)
    p.add_argument("--bank-replicates", type=int, default=2)
    p.add_argument("--out", type=str, default="",
                    help="output JSON path; '-' for stdout.")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--budget-sweep", action="store_true",
                    help="run K_auditor budget sweep (W7-2 falsifier).")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _arg_parser().parse_args(argv)
    if args.budget_sweep:
        report = run_phase54_budget_sweep(
            n_eval=args.n_eval, bank_seed=args.bank_seed,
            bank_replicates=args.bank_replicates)
    else:
        report = run_phase54(
            n_eval=args.n_eval,
            K_auditor=args.K_auditor,
            T_auditor=args.T_auditor,
            bank_seed=args.bank_seed,
            bank_replicates=args.bank_replicates,
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
            print(f"[phase54] wrote {args.out}", file=sys.stderr)
    else:
        print(json.dumps(report["pooled"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
