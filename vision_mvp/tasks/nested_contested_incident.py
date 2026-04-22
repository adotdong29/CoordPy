"""Phase 37 Part C — nested-contest incident task family.

Phase 36 § D.3 established empirical equivalence of dynamic
threads and bounded adaptive subscriptions on the Phase-35
contested bank: 0 pp gap on every noise cell. The honest
caveat was that the family was narrow — every scenario
resolves in *one* round of producer-local typed replies, so
the thread's stateful affordances (current_round, frozen
member set across rounds) were never load-bearing.

This module builds a harder family: **nested contests where
round-1 replies are insufficient.** The scenarios are
constructed so that producers *cannot* classify their own
candidate as INDEPENDENT_ROOT or DOWNSTREAM_SYMPTOM from
role-local evidence alone in round 1. They need to see at
least one peer's round-1 reply before they can classify in
round 2. Typical pattern:

  * Three roles X, Y, Z each emit a candidate claim.
  * Role X's event mentions "either a or b caused this,
    depending on whether Y saw z" — i.e. X's own extractor
    returns UNCERTAIN in round 1.
  * Role Y's event reveals whether z happened. After round 1,
    Y has posted an IR / DS reply on its own claim.
  * In round 2, X has access to Y's posted reply (via the
    thread's ``replies`` list — a native thread affordance)
    and can now classify its own claim.

Under the thread primitive, this is a max_rounds=2 thread
with per-round reply posting. Under bounded adaptive
subscriptions, the analogue requires either (i) an auditor-
to-producer briefing edge between rounds carrying the
summary of round-1 hypotheses, OR (ii) a role-local inbox
that the producer's extractor reads before emitting its
round-2 hypothesis.

The Phase-37 experiment compares:

  * ``dynamic``          — max_rounds=2 thread with
    per-round typed replies and round-2 replies conditioned
    on round-1 thread state.
  * ``adaptive_sub_2r``  — two install/emit/tick cycles
    with an inter-round auditor briefing edge;
    implementation lives in
    ``run_adaptive_sub_two_round_coordination``.
  * ``adaptive_sub_1r``  — the Phase-36 single-round
    adaptive-sub, kept in the comparison as a dominated
    baseline.
  * ``static_handoff``   — Phase-31 baseline.

The aim is falsifiability: if ``adaptive_sub_2r`` matches
``dynamic`` at equal accuracy, Conjecture C35-5 is
*strongly* reinforced. If ``dynamic`` dominates
``adaptive_sub_2r`` on this family while both dominate
``adaptive_sub_1r``, we have located a real separation.

Scope discipline (what this module does NOT do)
-----------------------------------------------

  * Does NOT rewrite the thread primitive. The Phase-35
    ``EscalationThread`` already supports max_rounds ≥ 2 and
    per-round reply posting; we only use that affordance.
  * Does NOT rewrite ``AdaptiveSubRouter``. Two-round
    adaptive-sub is built on top of its existing
    install/tick surface.
  * Does NOT introduce free-form chat. Round-2 replies are
    still in the Phase-35 typed-reply enum; briefings under
    adaptive-sub-2r are one-line typed handoffs with a
    constrained payload shape.
  * Does NOT attempt ensemble defenses on top of the harder
    family — Phase-37 Part B covers the reply-axis
    ensemble question on the Phase-35 bank; combining the
    two is future work.

Theoretical anchor: RESULTS_PHASE37.md § B.3 (Theorem P37-5,
Conjecture C37-3).
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

from vision_mvp.core.role_handoff import (
    HandoffRouter, RoleInbox, RoleSubscriptionTable, TypedHandoff,
)
from vision_mvp.core.dynamic_comm import (
    DynamicCommRouter,
    REPLY_DOWNSTREAM_SYMPTOM, REPLY_INDEPENDENT_ROOT,
    REPLY_UNCERTAIN, RESOLUTION_CONFLICT, RESOLUTION_NO_CONSENSUS,
    RESOLUTION_SINGLE_INDEPENDENT_ROOT,
    THREAD_ISSUE_ROOT_CAUSE_CONFLICT,
    build_resolution_subscriptions,
)
from vision_mvp.tasks.contested_incident import (
    ContestedScenario, _distractors, _mk_event, _STATIC_PRIORITY,
    build_phase35_subscriptions, claim_kind_to_label,
    decoder_from_handoffs_phase35, detect_contested_top,
)
from vision_mvp.tasks.incident_triage import (
    ALL_ROLES, CLAIM_CRON_OVERRUN, CLAIM_DEADLOCK_SUSPECTED,
    CLAIM_DISK_FILL_CRITICAL, CLAIM_DNS_MISROUTE,
    CLAIM_ERROR_RATE_SPIKE, CLAIM_OOM_KILL, CLAIM_POOL_EXHAUSTION,
    CLAIM_TLS_EXPIRED,
    EVENT_DNS_QUERY, EVENT_FW_RULE_HIT, EVENT_LOG_LINE,
    EVENT_METRIC_SAMPLE, EVENT_NET_FLOW, EVENT_OS_EVENT,
    EVENT_SQL_STAT, IncidentEvent, IncidentScenario,
    ROLE_AUDITOR, ROLE_DB_ADMIN, ROLE_MONITOR, ROLE_NETWORK,
    ROLE_SYSADMIN, extract_claims_for_role,
)


# =============================================================================
# Nested scenario model — round-dependent causality
# =============================================================================


# Round-dependent causality: a mapping
#   (producer_role, claim_kind) -> {
#       1: initial_answer_without_peer_context,
#       2: refined_answer_given_peer_context,
#   }
# The scenario's ``round_causality`` stores this shape.
# If round 2 is not specified, it defaults to the round-1 answer.
NestedCausalityMap = dict[tuple[str, str], dict[int, str]]


@dataclass(frozen=True)
class NestedScenario:
    """Phase-37 Part C nested scenario.

    Fields:
      * ``base``                   — wrapped ``IncidentScenario``.
      * ``round_causality``        — per-(role, kind) dict of
        round → causality class. Round 1 is the "role-local only"
        answer (usually UNCERTAIN for at least one producer).
        Round 2 is the refined answer after the thread / adaptive
        sub has exposed peer round-1 replies.
      * ``peer_witness_gate``      — (role, kind) producer whose
        round-1 reply is the gate: if that producer's round-1
        reply equals ``gate_reply_kind``, the conditional producer
        upgrades from UNCERTAIN to their conditional answer.
      * ``conditional_producers``  — list of
        (role, kind, expected_upgrade_class) tuples — producers
        that start UNCERTAIN and refine once the gate fires.
      * ``gold_root_cause_kind``   — ``claim_kind`` of the true
        root. Matched to the priority decoder.
      * ``gold_claim_idx_name``    — short tag.
    """

    base: IncidentScenario
    round_causality: NestedCausalityMap
    peer_witness_gate: tuple[str, str]
    gate_reply_kind: str
    conditional_producers: tuple[tuple[str, str, str], ...]
    gold_root_cause_kind: str
    gold_claim_idx_name: str

    @property
    def scenario_id(self) -> str:
        return self.base.scenario_id

    @property
    def description(self) -> str:
        return self.base.description

    @property
    def gold_root_cause(self) -> str:
        return self.base.gold_root_cause

    @property
    def gold_services(self) -> tuple[str, ...]:
        return self.base.gold_services

    @property
    def gold_remediation(self) -> str:
        return self.base.gold_remediation

    @property
    def causal_chain(self):
        return self.base.causal_chain

    @property
    def per_role_events(self):
        return self.base.per_role_events


# =============================================================================
# Nested bank — three hand-designed scenarios
# =============================================================================


def make_nested_tls_requires_sysadmin_witness(
        rng: random.Random, start_id: int = 0,
        distractors_per_role: int = 6,
        ) -> NestedScenario:
    """Real cause: TLS expired on api. The twist: network's own
    evidence is ambiguous — the FW_RULE_HIT shows a
    ``tls handshake`` but does not resolve whether the issue is
    a cert expiry (network's root) or a downstream symptom of
    sysadmin's partial disk-fill (which could have corrupted the
    TLS keystore). In round 1, network replies UNCERTAIN. In
    round 2, once network sees sysadmin's round-1 reply
    classifying DISK_FILL as UNCERTAIN (the disk is on
    /var/archive, not the api FS), network can confidently
    reply INDEPENDENT_ROOT.

    So: the gate is ``(sysadmin, DISK_FILL_CRITICAL)`` with gate
    reply UNCERTAIN. The conditional producer is
    ``(network, TLS_EXPIRED)``, upgrading to INDEPENDENT_ROOT.

    Under a single-round thread or adaptive-sub, all producers
    reply UNCERTAIN → NO_CONSENSUS → decoder falls back to
    static priority → picks DISK_FILL_CRITICAL (wrong).
    """
    nid = start_id
    per_role: dict[str, list[IncidentEvent]] = {r: [] for r in ALL_ROLES}
    chain: list[tuple[str, str, str, tuple[int, ...]]] = []

    # --- NETWORK (gated; round-1 UNCERTAIN, round-2 IR) ---
    ev, nid = _mk_event(nid, EVENT_FW_RULE_HIT, ROLE_NETWORK,
                         "tls handshake fail service=api "
                         "reason=expired cert_path_ambiguous",
                         tags=("api",), causal=True)
    per_role[ROLE_NETWORK].append(ev)
    tls_ev = ev.event_id
    chain.append((ROLE_NETWORK, CLAIM_TLS_EXPIRED,
                  "tls service=api reason=expired cert_path_ambiguous",
                  (tls_ev,)))

    # --- SYSADMIN (gate; archival disk fill, UNCERTAIN by role) ---
    ev, nid = _mk_event(nid, EVENT_OS_EVENT, ROLE_SYSADMIN,
                         "/var/archive used=93% fs=/var/archive "
                         "service=archival",
                         tags=("archival",), causal=True)
    per_role[ROLE_SYSADMIN].append(ev)
    disk_ev = ev.event_id
    chain.append((ROLE_SYSADMIN, CLAIM_DISK_FILL_CRITICAL,
                  "/var/archive used=93% fs=/var/archive "
                  "service=archival",
                  (disk_ev,)))

    # --- MONITOR --- downstream symptom, not part of the gate.
    ev, nid = _mk_event(nid, EVENT_METRIC_SAMPLE, ROLE_MONITOR,
                         "uptime_pct=41 service=api window=5m",
                         tags=("api",), causal=True)
    per_role[ROLE_MONITOR].append(ev)
    m_ev = ev.event_id
    chain.append((ROLE_MONITOR, CLAIM_ERROR_RATE_SPIKE,
                  "uptime_pct=41 service=api", (m_ev,)))

    for role, kinds in ((ROLE_MONITOR, [EVENT_METRIC_SAMPLE]),
                         (ROLE_DB_ADMIN, [EVENT_SQL_STAT,
                                           EVENT_LOG_LINE]),
                         (ROLE_SYSADMIN, [EVENT_OS_EVENT]),
                         (ROLE_NETWORK, [EVENT_NET_FLOW,
                                          EVENT_DNS_QUERY])):
        dist, nid = _distractors(rng, nid, role, kinds,
                                   k=distractors_per_role)
        per_role[role].extend(dist)

    base = IncidentScenario(
        scenario_id="nested_tls_requires_sysadmin_witness",
        description="tls expired on api is the real cause; network "
                    "cannot tell in round-1 whether the tls fail is "
                    "an isolated expiry or downstream of a /var "
                    "disk issue, so it posts UNCERTAIN. Once sysadmin "
                    "clarifies the disk fill is on /var/archive "
                    "(unrelated to api), network refines to "
                    "INDEPENDENT_ROOT in round-2.",
        gold_root_cause="tls_expiry",
        gold_services=("api",),
        gold_remediation="renew_tls_and_reload",
        causal_chain=tuple(chain),
        per_role_events={r: tuple(evs) for r, evs in per_role.items()},
    )
    round_causality: NestedCausalityMap = {
        (ROLE_NETWORK, CLAIM_TLS_EXPIRED): {
            1: "UNCERTAIN",
            2: "INDEPENDENT_ROOT",
        },
        (ROLE_SYSADMIN, CLAIM_DISK_FILL_CRITICAL): {
            1: "UNCERTAIN",
            2: "UNCERTAIN",
        },
        (ROLE_MONITOR, CLAIM_ERROR_RATE_SPIKE): {
            1: f"DOWNSTREAM_SYMPTOM_OF:{CLAIM_TLS_EXPIRED}",
            2: f"DOWNSTREAM_SYMPTOM_OF:{CLAIM_TLS_EXPIRED}",
        },
    }
    return NestedScenario(
        base=base,
        round_causality=round_causality,
        peer_witness_gate=(ROLE_SYSADMIN, CLAIM_DISK_FILL_CRITICAL),
        gate_reply_kind=REPLY_UNCERTAIN,
        conditional_producers=(
            (ROLE_NETWORK, CLAIM_TLS_EXPIRED, REPLY_INDEPENDENT_ROOT),
        ),
        gold_root_cause_kind=CLAIM_TLS_EXPIRED,
        gold_claim_idx_name="tls_expiry",
    )


def make_nested_deadlock_requires_network_witness(
        rng: random.Random, start_id: int = 0,
        distractors_per_role: int = 6,
        ) -> NestedScenario:
    """Real cause: application-level deadlock on orders. Twist:
    dba's role-local evidence is a warning that *could* be a
    deadlock OR could be a symptom of a DNS misroute upstream
    (db.internal unreachable causes a race that looks like a
    deadlock). In round 1, dba replies UNCERTAIN. Network's
    round-1 reply is UNCERTAIN (no DNS events in this scenario),
    which is the gate: once dba sees "network=UNCERTAIN" in
    round 2, dba can rule out the DNS symptom story and reply
    INDEPENDENT_ROOT.

    Gate: ``(network, CLAIM_DNS_MISROUTE)`` with gate
    reply = UNCERTAIN. Conditional producer:
    ``(db_admin, CLAIM_DEADLOCK_SUSPECTED)`` upgrades to
    INDEPENDENT_ROOT.

    A single-round thread or adaptive-sub returns NO_CONSENSUS
    → static priority falls back to CRON_OVERRUN (wrong: real
    cause is DEADLOCK).
    """
    nid = start_id
    per_role: dict[str, list[IncidentEvent]] = {r: [] for r in ALL_ROLES}
    chain: list[tuple[str, str, str, tuple[int, ...]]] = []

    # --- DBA (gated) ---
    ev, nid = _mk_event(nid, EVENT_LOG_LINE, ROLE_DB_ADMIN,
                         "warning pg deadlock_or_rpc_stall pid=8823 "
                         "relation=orders_payments "
                         "cause=ambiguous_app_or_rpc",
                         tags=("orders",), causal=True)
    per_role[ROLE_DB_ADMIN].append(ev)
    dl_ev = ev.event_id
    chain.append((ROLE_DB_ADMIN, CLAIM_DEADLOCK_SUSPECTED,
                  "deadlock_or_rpc_stall relation=orders_payments "
                  "cause=ambiguous_app_or_rpc", (dl_ev,)))

    # --- SYSADMIN confounder: cron overrun on archival host ---
    ev, nid = _mk_event(nid, EVENT_OS_EVENT, ROLE_SYSADMIN,
                         "cron backup.sh exit=137 duration_s=5400 "
                         "service=archival host=backup01",
                         tags=("archival",), causal=True)
    per_role[ROLE_SYSADMIN].append(ev)
    cron_ev = ev.event_id
    chain.append((ROLE_SYSADMIN, CLAIM_CRON_OVERRUN,
                  "backup.sh exit=137 duration_s=5400 "
                  "service=archival", (cron_ev,)))

    # --- MONITOR downstream ---
    ev, nid = _mk_event(nid, EVENT_METRIC_SAMPLE, ROLE_MONITOR,
                         "error_rate=0.22 service=orders window=5m",
                         tags=("orders",), causal=True)
    per_role[ROLE_MONITOR].append(ev)
    m_ev = ev.event_id
    chain.append((ROLE_MONITOR, CLAIM_ERROR_RATE_SPIKE,
                  "error_rate=0.22 service=orders", (m_ev,)))

    # Note: network has NO causal claim here. The *absence* of a
    # DNS_MISROUTE claim is the gate signal: network participates
    # with a sentinel ``no_dns_issue`` emission so the thread has
    # a peer reply to condition on.

    for role, kinds in ((ROLE_MONITOR, [EVENT_METRIC_SAMPLE]),
                         (ROLE_DB_ADMIN, [EVENT_SQL_STAT,
                                           EVENT_LOG_LINE]),
                         (ROLE_SYSADMIN, [EVENT_OS_EVENT]),
                         (ROLE_NETWORK, [EVENT_NET_FLOW,
                                          EVENT_DNS_QUERY])):
        dist, nid = _distractors(rng, nid, role, kinds,
                                   k=distractors_per_role)
        per_role[role].extend(dist)

    base = IncidentScenario(
        scenario_id="nested_deadlock_requires_network_witness",
        description="pg deadlock on orders_payments is the real "
                    "cause; dba cannot tell in round-1 whether "
                    "the warning is a deadlock or a downstream "
                    "of an upstream DNS misroute. Once network "
                    "reports no DNS issue in round-1 (UNCERTAIN "
                    "on DNS_MISROUTE because they have no DNS "
                    "events), dba refines to INDEPENDENT_ROOT "
                    "in round-2.",
        gold_root_cause="deadlock",
        gold_services=("orders",),
        gold_remediation="enforce_lock_ordering_in_orders",
        causal_chain=tuple(chain),
        per_role_events={r: tuple(evs) for r, evs in per_role.items()},
    )
    round_causality: NestedCausalityMap = {
        (ROLE_DB_ADMIN, CLAIM_DEADLOCK_SUSPECTED): {
            1: "UNCERTAIN",
            2: "INDEPENDENT_ROOT",
        },
        (ROLE_SYSADMIN, CLAIM_CRON_OVERRUN): {
            1: "UNCERTAIN",
            2: "UNCERTAIN",
        },
        (ROLE_MONITOR, CLAIM_ERROR_RATE_SPIKE): {
            1: f"DOWNSTREAM_SYMPTOM_OF:{CLAIM_DEADLOCK_SUSPECTED}",
            2: f"DOWNSTREAM_SYMPTOM_OF:{CLAIM_DEADLOCK_SUSPECTED}",
        },
    }
    return NestedScenario(
        base=base,
        round_causality=round_causality,
        peer_witness_gate=(ROLE_NETWORK, CLAIM_DNS_MISROUTE),
        gate_reply_kind=REPLY_UNCERTAIN,
        conditional_producers=(
            (ROLE_DB_ADMIN, CLAIM_DEADLOCK_SUSPECTED,
             REPLY_INDEPENDENT_ROOT),
        ),
        gold_root_cause_kind=CLAIM_DEADLOCK_SUSPECTED,
        gold_claim_idx_name="deadlock",
    )


def make_nested_oom_requires_dba_witness(
        rng: random.Random, start_id: int = 0,
        distractors_per_role: int = 6,
        ) -> NestedScenario:
    """Real cause: OOM kill on api service from a memory leak.
    Twist: sysadmin's OOM log could be either a genuine app
    leak (INDEPENDENT_ROOT) or a symptom of an upstream DNS
    misroute (where a DNS outage backlog piles up memory in
    retry buffers). In round 1, sysadmin replies UNCERTAIN.
    In round 2, after network's round-1 reply classifies its
    own DNS_MISROUTE as UNCERTAIN (transient SERVFAIL, not
    an outage), sysadmin can refine to INDEPENDENT_ROOT.

    Static priority picks DNS_MISROUTE (#3 before OOM_KILL
    #4) → static is wrong.

    Gate: ``(network, CLAIM_DNS_MISROUTE)`` with gate
    reply = UNCERTAIN. Conditional producer:
    ``(sysadmin, CLAIM_OOM_KILL)`` upgrades to INDEPENDENT_ROOT.
    """
    nid = start_id
    per_role: dict[str, list[IncidentEvent]] = {r: [] for r in ALL_ROLES}
    chain: list[tuple[str, str, str, tuple[int, ...]]] = []

    # --- SYSADMIN (gated) ---
    ev, nid = _mk_event(nid, EVENT_OS_EVENT, ROLE_SYSADMIN,
                         "oom_kill pid=42 process=api-worker "
                         "rss_mb=2048 service=api "
                         "cause=ambiguous_leak_or_contention",
                         tags=("api",), causal=True)
    per_role[ROLE_SYSADMIN].append(ev)
    oom_ev = ev.event_id
    chain.append((ROLE_SYSADMIN, CLAIM_OOM_KILL,
                  "oom_kill pid=42 process=api-worker "
                  "rss_mb=2048 service=api "
                  "cause=ambiguous_leak_or_contention",
                  (oom_ev,)))

    # --- NETWORK (gate role emits a root-bearing DNS_MISROUTE) ---
    # NETWORK's DNS_QUERY carries "SERVFAIL" so the Phase-31
    # extractor emits DNS_MISROUTE (root-bearing, priority #3,
    # beats OOM_KILL at #4 under static). NETWORK's round-1
    # verdict is UNCERTAIN because the rc=SERVFAIL is a single
    # transient, not an outage pattern.
    ev, nid = _mk_event(nid, EVENT_DNS_QUERY, ROLE_NETWORK,
                         "q=cache.internal rc=SERVFAIL rtt_ms=50 "
                         "pattern=transient",
                         tags=("cache",), causal=True)
    per_role[ROLE_NETWORK].append(ev)
    dns_ev = ev.event_id
    chain.append((ROLE_NETWORK, CLAIM_DNS_MISROUTE,
                  "q=cache.internal rc=SERVFAIL rtt_ms=50 "
                  "pattern=transient",
                  (dns_ev,)))

    # --- MONITOR downstream ---
    ev, nid = _mk_event(nid, EVENT_METRIC_SAMPLE, ROLE_MONITOR,
                         "error_rate=0.34 service=api window=5m",
                         tags=("api",), causal=True)
    per_role[ROLE_MONITOR].append(ev)
    m_ev = ev.event_id
    chain.append((ROLE_MONITOR, CLAIM_ERROR_RATE_SPIKE,
                  "error_rate=0.34 service=api", (m_ev,)))

    for role, kinds in ((ROLE_MONITOR, [EVENT_METRIC_SAMPLE]),
                         (ROLE_DB_ADMIN, [EVENT_SQL_STAT,
                                           EVENT_LOG_LINE]),
                         (ROLE_SYSADMIN, [EVENT_OS_EVENT]),
                         (ROLE_NETWORK, [EVENT_NET_FLOW,
                                          EVENT_DNS_QUERY])):
        dist, nid = _distractors(rng, nid, role, kinds,
                                   k=distractors_per_role)
        per_role[role].extend(dist)

    base = IncidentScenario(
        scenario_id="nested_oom_requires_dba_witness",
        description="oom kill on api is the real cause; sysadmin "
                    "cannot tell in round-1 whether the OOM is a "
                    "genuine app leak or a symptom of dba's pool "
                    "exhaustion. Once dba reports pool is normal "
                    "in round-1 (UNCERTAIN on POOL_EXHAUSTION), "
                    "sysadmin refines to INDEPENDENT_ROOT in "
                    "round-2.",
        gold_root_cause="memory_leak",
        gold_services=("api",),
        gold_remediation="rollback_app_to_prev_release",
        causal_chain=tuple(chain),
        per_role_events={r: tuple(evs) for r, evs in per_role.items()},
    )
    round_causality: NestedCausalityMap = {
        (ROLE_SYSADMIN, CLAIM_OOM_KILL): {
            1: "UNCERTAIN",
            2: "INDEPENDENT_ROOT",
        },
        (ROLE_NETWORK, CLAIM_DNS_MISROUTE): {
            1: "UNCERTAIN",
            2: "UNCERTAIN",
        },
        (ROLE_MONITOR, CLAIM_ERROR_RATE_SPIKE): {
            1: f"DOWNSTREAM_SYMPTOM_OF:{CLAIM_OOM_KILL}",
            2: f"DOWNSTREAM_SYMPTOM_OF:{CLAIM_OOM_KILL}",
        },
    }
    return NestedScenario(
        base=base,
        round_causality=round_causality,
        peer_witness_gate=(ROLE_NETWORK, CLAIM_DNS_MISROUTE),
        gate_reply_kind=REPLY_UNCERTAIN,
        conditional_producers=(
            (ROLE_SYSADMIN, CLAIM_OOM_KILL,
             REPLY_INDEPENDENT_ROOT),
        ),
        gold_root_cause_kind=CLAIM_OOM_KILL,
        gold_claim_idx_name="memory_leak",
    )


NESTED_SCENARIO_BUILDERS: tuple[
    Callable[..., NestedScenario], ...] = (
    make_nested_tls_requires_sysadmin_witness,
    make_nested_deadlock_requires_network_witness,
    make_nested_oom_requires_dba_witness,
)


def build_nested_bank(seed: int = 37,
                       distractors_per_role: int = 6,
                       ) -> list[NestedScenario]:
    rng = random.Random(seed)
    return [b(rng, 0, distractors_per_role=distractors_per_role)
            for b in NESTED_SCENARIO_BUILDERS]


# =============================================================================
# Round-dependent causality oracle (used by both strategies)
# =============================================================================


def nested_round_oracle(scenario: NestedScenario,
                          round_no: int,
                          role: str,
                          kind: str,
                          payload: str,
                          ) -> str:
    """Return the round-dependent causality class for (role, kind)
    on ``scenario`` at round ``round_no``.

    Round numbering is 1-based; rounds > max stored round fall back
    to the max stored round's answer.
    """
    rounds = scenario.round_causality.get((role, kind))
    if not rounds:
        return "UNCERTAIN"
    if round_no in rounds:
        return rounds[round_no]
    max_r = max(rounds.keys())
    return rounds[max_r]


# =============================================================================
# Thread-based two-round coordination
# =============================================================================


@dataclass
class NestedCoordinationDebug:
    """Per-scenario debug surface.

    Captures the thread / adaptive-sub transcript so the Phase-37
    driver can report per-strategy diagnostics uniformly.
    """

    strategy: str
    contested_candidates: list[tuple[str, str, str]] = field(
        default_factory=list)
    thread_id: str | None = None
    round1_replies: list[tuple[str, str, int]] = field(default_factory=list)
    round2_replies: list[tuple[str, str, int]] = field(default_factory=list)
    n_briefings_installed: int = 0
    n_hypothesis_edges_installed: int = 0
    resolution_kind: str | None = None
    resolution_winner: tuple[str, str] | None = None
    resolved_claim_idx: int | None = None
    round_used: int = 0


def _run_handoff_prelude(scenario: NestedScenario,
                         router,
                         max_events_per_role: int,
                         ) -> None:
    """Emit the standard per-role typed handoffs into the router.

    Matches Phase-35 ``run_contested_handoff_protocol``'s body but
    works against either a ``DynamicCommRouter`` or an
    ``AdaptiveSubRouter``.
    """
    for role in (ROLE_MONITOR, ROLE_DB_ADMIN, ROLE_SYSADMIN,
                 ROLE_NETWORK):
        evs = list(scenario.per_role_events.get(role, ()))
        if max_events_per_role:
            evs = evs[:max_events_per_role]
        claims = extract_claims_for_role(role, evs, scenario.base)
        for (kind, payload, evids) in claims:
            router.emit(
                source_role=role,
                source_agent_id=ALL_ROLES.index(role),
                claim_kind=kind, payload=payload,
                source_event_ids=evids, round=1)


def run_nested_two_round_thread(
        scenario: NestedScenario,
        max_rounds: int = 2,
        witness_token_cap: int = 12,
        max_events_per_role: int = 200,
        inbox_capacity: int = 32,
        ) -> tuple[DynamicCommRouter, tuple[TypedHandoff, ...],
                   NestedCoordinationDebug]:
    """Run the Phase-35 thread primitive with max_rounds=2 on a
    nested scenario.

    Round 1: each producer posts a round-1 reply from the
    round-1 oracle (often UNCERTAIN).
    Round 2: producers whose (role, kind) is a conditional
    producer upgrade to their round-2 oracle class, conditioned
    on having observed the gate producer's round-1 reply. If
    the gate reply does not appear in round-1, the round-2
    reply stays at round-1's class.
    Close.
    """
    subs = build_phase35_subscriptions()
    base = HandoffRouter(subs=subs)
    for role in ALL_ROLES:
        base.register_inbox(RoleInbox(role=role,
                                        capacity=inbox_capacity))
    router = DynamicCommRouter(base_router=base)
    _run_handoff_prelude(scenario, router, max_events_per_role)

    auditor_inbox = router.inboxes.get(ROLE_AUDITOR)
    pre_handoffs = tuple(auditor_inbox.peek()) if auditor_inbox else ()
    top = detect_contested_top(pre_handoffs)

    debug = NestedCoordinationDebug(strategy="dynamic_nested_2r")
    if len(top) < 2:
        return router, pre_handoffs, debug

    candidates = [(h.source_role, h.claim_kind, h.payload)
                   for h in top]
    debug.contested_candidates = list(candidates)
    producer_roles = frozenset(h.source_role for h in top)
    members = producer_roles | {ROLE_AUDITOR}

    thread = router.open_thread(
        opener_role=ROLE_AUDITOR,
        issue_kind=THREAD_ISSUE_ROOT_CAUSE_CONFLICT,
        members=members,
        candidate_claims=candidates,
        max_rounds=max_rounds,
        max_replies_per_member=len(candidates) * max_rounds,
        quorum=1,
        witness_token_cap=witness_token_cap,
        round=2,
    )
    debug.thread_id = thread.thread_id

    # Round 1: each producer posts based on round-1 oracle.
    for idx, (prod_role, kind, payload) in enumerate(candidates):
        cls = nested_round_oracle(scenario, 1, prod_role,
                                    kind, payload)
        reply_kind = _class_to_reply_kind(cls)
        witness = " ".join(payload.split()[:witness_token_cap])
        router.post_reply(
            thread_id=thread.thread_id,
            replier_role=prod_role,
            reply_kind=reply_kind,
            referenced_claim_idx=idx,
            witness=witness,
            round=2,
        )
        debug.round1_replies.append((prod_role, reply_kind, idx))

    # Round 2: inspect round-1 state for the gate. Each
    # conditional producer reads the thread state to see if the
    # gate fired; if so, post its round-2 reply.
    state = router.get_state(thread.thread_id)
    round1_by_role: dict[str, list[str]] = {}
    for r in state.replies:
        round1_by_role.setdefault(r.replier_role, []).append(r.reply_kind)

    gate_role, gate_kind = scenario.peer_witness_gate
    # The gate role may not be in the candidates; in
    # ``nested_deadlock_requires_network_witness`` the network role
    # has no candidate claim. In that case, we treat "network
    # silence on DNS_MISROUTE" as equivalent to "network round-1
    # UNCERTAIN on DNS_MISROUTE" — semantically the peer has no
    # evidence.
    gate_fired = True
    if gate_role in round1_by_role:
        # If the gate role has replied, take the mode.
        gate_fired = (scenario.gate_reply_kind
                        in round1_by_role[gate_role])
    # If the gate role has no candidate claim in this scenario,
    # the "gate fires" (meaning: the gate role's silence is
    # semantically UNCERTAIN and the conditional producer can
    # upgrade).

    if gate_fired:
        for (cprod, ckind, _expected) in scenario.conditional_producers:
            # Find the candidate index for (cprod, ckind).
            for idx, (prod_role, kind, payload) in enumerate(candidates):
                if prod_role == cprod and kind == ckind:
                    cls2 = nested_round_oracle(
                        scenario, 2, prod_role, kind, payload)
                    reply2 = _class_to_reply_kind(cls2)
                    witness = " ".join(
                        payload.split()[:witness_token_cap])
                    router.post_reply(
                        thread_id=thread.thread_id,
                        replier_role=prod_role,
                        reply_kind=reply2,
                        referenced_claim_idx=idx,
                        witness=witness,
                        round=3,
                    )
                    debug.round2_replies.append(
                        (prod_role, reply2, idx))
                    break

    resolution = router.close_thread(thread.thread_id, round=4)
    debug.resolution_kind = resolution.resolution_kind
    debug.resolved_claim_idx = resolution.resolved_claim_idx
    debug.round_used = 2
    if resolution.resolved_claim_idx is not None:
        cc = thread.candidate_claims[resolution.resolved_claim_idx]
        debug.resolution_winner = (cc.producer_role, cc.claim_kind)
    handoffs_for_decoder = tuple(auditor_inbox.peek()) \
        if auditor_inbox else ()
    return router, handoffs_for_decoder, debug


# =============================================================================
# Adaptive-sub two-round coordination
# =============================================================================


# A dedicated claim kind for the inter-round briefing. Installed
# by the auditor as (auditor → producer, BRIEFING, ttl=1) between
# rounds. The payload encodes round-1 replies so the conditional
# producer's extractor can condition.
CLAIM_COORDINATION_BRIEFING = "COORDINATION_BRIEFING"


def _format_briefing(round1_replies: Sequence[tuple[str, str, int]]
                      ) -> str:
    parts = []
    for (role, kind, idx) in round1_replies:
        parts.append(f"{role}/{kind}@idx={idx}")
    return "round1=" + ",".join(parts)


def _parse_briefing(payload: str) -> list[tuple[str, str, int]]:
    if not payload.startswith("round1="):
        return []
    body = payload[len("round1="):]
    out: list[tuple[str, str, int]] = []
    for tok in body.split(","):
        tok = tok.strip()
        if not tok:
            continue
        # role/kind@idx=I
        try:
            rk, rest = tok.split("@", 1)
            role, kind = rk.split("/", 1)
            if rest.startswith("idx="):
                idx = int(rest[len("idx="):])
            else:
                idx = -1
        except ValueError:
            continue
        out.append((role, kind, idx))
    return out


def run_nested_two_round_adaptive_sub(
        scenario: NestedScenario,
        max_active_edges: int = 8,
        witness_token_cap: int = 12,
        max_events_per_role: int = 200,
        inbox_capacity: int = 32,
        ) -> tuple[object, tuple[TypedHandoff, ...],
                   NestedCoordinationDebug]:
    """Adaptive-sub analogue of ``run_nested_two_round_thread``.

    Implementation: two install/emit/tick cycles with an inter-
    round auditor-to-producer briefing edge.

    Round 1:
      1. For each producer in the contested set, install
         ``(producer → auditor, CAUSALITY_HYPOTHESIS, ttl=1)``.
      2. Each producer emits its round-1 hypothesis.
      3. Tick → edges expire.
    Inter-round briefing:
      4. For each conditional producer, install
         ``(auditor → producer, COORDINATION_BRIEFING, ttl=1)``.
      5. Auditor emits one BRIEFING handoff per conditional
         producer carrying the round-1 reply summary.
      6. Tick → edges expire.
    Round 2:
      7. For each conditional producer, install
         ``(producer → auditor, CAUSALITY_HYPOTHESIS, ttl=1)``.
      8. Producer reads its inbox for the BRIEFING; if present,
         its round-2 extractor is invoked (gate-aware); else
         it falls back to its round-1 answer.
      9. Tick → edges expire.
    """
    from vision_mvp.core.adaptive_sub import (
        AdaptiveSubRouter, CLAIM_CAUSALITY_HYPOTHESIS,
        format_hypothesis_payload,
    )
    subs = build_phase35_subscriptions()
    # Ensure the base subscription table knows about the
    # briefing kind (AdaptiveSubRouter installs will override,
    # but we pre-register for cleanliness / auditability).
    base = HandoffRouter(subs=subs)
    for role in ALL_ROLES:
        base.register_inbox(RoleInbox(role=role,
                                        capacity=inbox_capacity))
    router = AdaptiveSubRouter(
        base_router=base, max_active_edges=max_active_edges)

    _run_handoff_prelude(scenario, router, max_events_per_role)

    auditor_inbox = router.inboxes.get(ROLE_AUDITOR)
    pre_handoffs = tuple(auditor_inbox.peek()) if auditor_inbox else ()
    top = detect_contested_top(pre_handoffs)

    debug = NestedCoordinationDebug(strategy="adaptive_sub_2r")
    if len(top) < 2:
        return router, pre_handoffs, debug

    candidates = [(h.source_role, h.claim_kind, h.payload)
                   for h in top]
    debug.contested_candidates = list(candidates)

    # ---- Round 1 ----
    installed: list = []
    for prod in sorted({h.source_role for h in top}):
        edge = router.install_edge(
            source_role=prod,
            claim_kind=CLAIM_CAUSALITY_HYPOTHESIS,
            consumer_roles=[ROLE_AUDITOR],
            ttl_rounds=1,
        )
        installed.append(edge)
    debug.n_hypothesis_edges_installed = len(installed)
    round1: list[tuple[str, str, int]] = []
    for idx, (prod_role, kind, payload) in enumerate(candidates):
        cls = nested_round_oracle(scenario, 1, prod_role,
                                    kind, payload)
        reply_kind = _class_to_reply_kind(cls)
        witness = " ".join(payload.split()[:witness_token_cap])
        router.emit(
            source_role=prod_role,
            source_agent_id=ALL_ROLES.index(prod_role),
            claim_kind=CLAIM_CAUSALITY_HYPOTHESIS,
            payload=format_hypothesis_payload(
                reply_kind, idx, upstream_kind=kind,
                witness=witness),
            source_event_ids=(), round=2,
        )
        round1.append((prod_role, reply_kind, idx))
    debug.round1_replies = list(round1)
    router.tick(1)

    # ---- Inter-round briefing: auditor → conditional producer ----
    gate_role, _gate_kind = scenario.peer_witness_gate
    round1_by_role: dict[str, list[str]] = {}
    for (role, reply_kind, _idx) in round1:
        round1_by_role.setdefault(role, []).append(reply_kind)
    gate_fired = True
    if gate_role in round1_by_role:
        gate_fired = (scenario.gate_reply_kind
                        in round1_by_role[gate_role])

    briefings_sent = 0
    if gate_fired:
        for (cprod, _ckind, _expected) in scenario.conditional_producers:
            edge_b = router.install_edge(
                source_role=ROLE_AUDITOR,
                claim_kind=CLAIM_COORDINATION_BRIEFING,
                consumer_roles=[cprod],
                ttl_rounds=1,
            )
            installed.append(edge_b)
            router.emit(
                source_role=ROLE_AUDITOR,
                source_agent_id=ALL_ROLES.index(ROLE_AUDITOR),
                claim_kind=CLAIM_COORDINATION_BRIEFING,
                payload=_format_briefing(round1),
                source_event_ids=(), round=3,
            )
            briefings_sent += 1
    debug.n_briefings_installed = briefings_sent
    router.tick(1)

    # ---- Round 2: conditional producers re-emit ----
    round2: list[tuple[str, str, int]] = []
    if gate_fired and briefings_sent > 0:
        for (cprod, ckind, _expected) in scenario.conditional_producers:
            cprod_inbox = router.inboxes.get(cprod)
            has_briefing = False
            if cprod_inbox is not None:
                for h in cprod_inbox.peek():
                    if h.claim_kind == CLAIM_COORDINATION_BRIEFING:
                        has_briefing = True
                        break
            edge_r2 = router.install_edge(
                source_role=cprod,
                claim_kind=CLAIM_CAUSALITY_HYPOTHESIS,
                consumer_roles=[ROLE_AUDITOR],
                ttl_rounds=1,
            )
            installed.append(edge_r2)
            for idx, (prod_role, kind, payload) in enumerate(candidates):
                if prod_role != cprod or kind != ckind:
                    continue
                if has_briefing:
                    cls2 = nested_round_oracle(
                        scenario, 2, prod_role, kind, payload)
                else:
                    cls2 = nested_round_oracle(
                        scenario, 1, prod_role, kind, payload)
                reply2 = _class_to_reply_kind(cls2)
                witness = " ".join(
                    payload.split()[:witness_token_cap])
                router.emit(
                    source_role=prod_role,
                    source_agent_id=ALL_ROLES.index(prod_role),
                    claim_kind=CLAIM_CAUSALITY_HYPOTHESIS,
                    payload=format_hypothesis_payload(
                        reply2, idx, upstream_kind=kind,
                        witness=witness),
                    source_event_ids=(), round=4,
                )
                round2.append((prod_role, reply2, idx))
                break
    debug.round2_replies = list(round2)
    router.tick(1)

    # Apply the resolution rule over all CAUSALITY_HYPOTHESIS
    # handoffs the auditor has seen. Later round-2 IR claims
    # override round-1 UNCERTAIN on the same idx.
    from vision_mvp.core.adaptive_sub import (
        parse_hypothesis_payload,
    )
    post_handoffs = tuple(auditor_inbox.peek()) \
        if auditor_inbox else ()
    latest_by_idx: dict[int, tuple[str, str]] = {}
    for h in post_handoffs:
        if h.claim_kind != CLAIM_CAUSALITY_HYPOTHESIS:
            continue
        parsed = parse_hypothesis_payload(h.payload)
        try:
            idx = int(parsed.get("idx", "-1"))
        except ValueError:
            idx = -1
        if idx < 0:
            continue
        latest_by_idx[idx] = (h.source_role,
                                parsed.get("kind", ""))
    # Count IR hypotheses on latest-per-idx.
    ir_idxs = [idx for idx, (_r, k) in latest_by_idx.items()
                if k == REPLY_INDEPENDENT_ROOT]
    if len(ir_idxs) == 1:
        resolved_idx = ir_idxs[0]
        debug.resolution_kind = RESOLUTION_SINGLE_INDEPENDENT_ROOT
        debug.resolved_claim_idx = resolved_idx
        prod, kind, _payload = candidates[resolved_idx]
        debug.resolution_winner = (prod, kind)
    elif len(ir_idxs) >= 2:
        debug.resolution_kind = RESOLUTION_CONFLICT
    else:
        debug.resolution_kind = RESOLUTION_NO_CONSENSUS

    debug.round_used = 2
    handoffs_for_decoder = post_handoffs
    return router, handoffs_for_decoder, debug


# Single-round adaptive-sub for the nested bank — inherited
# shape from Phase-36 Part C, round-1 only. Dominated baseline.


def run_nested_one_round_adaptive_sub(
        scenario: NestedScenario,
        max_active_edges: int = 4,
        witness_token_cap: int = 12,
        max_events_per_role: int = 200,
        inbox_capacity: int = 32,
        ) -> tuple[object, tuple[TypedHandoff, ...],
                   NestedCoordinationDebug]:
    """Single-round adaptive-sub on nested scenarios.

    This is the Phase-36 primitive evaluated on a family it is
    specified against: every conditional producer replies
    UNCERTAIN in round-1, and there is no second round to
    recover. Resolution falls back to static priority.
    """
    from vision_mvp.core.adaptive_sub import (
        AdaptiveSubRouter, CLAIM_CAUSALITY_HYPOTHESIS,
        format_hypothesis_payload,
    )
    subs = build_phase35_subscriptions()
    base = HandoffRouter(subs=subs)
    for role in ALL_ROLES:
        base.register_inbox(RoleInbox(role=role,
                                        capacity=inbox_capacity))
    router = AdaptiveSubRouter(
        base_router=base, max_active_edges=max_active_edges)
    _run_handoff_prelude(scenario, router, max_events_per_role)
    auditor_inbox = router.inboxes.get(ROLE_AUDITOR)
    pre_handoffs = tuple(auditor_inbox.peek()) if auditor_inbox else ()
    top = detect_contested_top(pre_handoffs)
    debug = NestedCoordinationDebug(strategy="adaptive_sub_1r")
    if len(top) < 2:
        return router, pre_handoffs, debug
    candidates = [(h.source_role, h.claim_kind, h.payload)
                   for h in top]
    debug.contested_candidates = list(candidates)
    for prod in sorted({h.source_role for h in top}):
        router.install_edge(
            source_role=prod,
            claim_kind=CLAIM_CAUSALITY_HYPOTHESIS,
            consumer_roles=[ROLE_AUDITOR],
            ttl_rounds=1,
        )
        debug.n_hypothesis_edges_installed += 1
    round1: list[tuple[str, str, int]] = []
    for idx, (prod_role, kind, payload) in enumerate(candidates):
        cls = nested_round_oracle(scenario, 1, prod_role,
                                    kind, payload)
        reply_kind = _class_to_reply_kind(cls)
        witness = " ".join(payload.split()[:witness_token_cap])
        router.emit(
            source_role=prod_role,
            source_agent_id=ALL_ROLES.index(prod_role),
            claim_kind=CLAIM_CAUSALITY_HYPOTHESIS,
            payload=format_hypothesis_payload(
                reply_kind, idx, upstream_kind=kind,
                witness=witness),
            source_event_ids=(), round=2,
        )
        round1.append((prod_role, reply_kind, idx))
    router.tick(1)
    debug.round1_replies = list(round1)

    from vision_mvp.core.adaptive_sub import (
        parse_hypothesis_payload,
    )
    post_handoffs = tuple(auditor_inbox.peek()) \
        if auditor_inbox else ()
    ir_idxs: list[int] = []
    for h in post_handoffs:
        if h.claim_kind != CLAIM_CAUSALITY_HYPOTHESIS:
            continue
        parsed = parse_hypothesis_payload(h.payload)
        try:
            idx = int(parsed.get("idx", "-1"))
        except ValueError:
            idx = -1
        if idx < 0:
            continue
        if parsed.get("kind") == REPLY_INDEPENDENT_ROOT:
            ir_idxs.append(idx)
    if len(ir_idxs) == 1:
        debug.resolution_kind = RESOLUTION_SINGLE_INDEPENDENT_ROOT
        debug.resolved_claim_idx = ir_idxs[0]
        prod, kind, _payload = candidates[ir_idxs[0]]
        debug.resolution_winner = (prod, kind)
    elif len(ir_idxs) >= 2:
        debug.resolution_kind = RESOLUTION_CONFLICT
    else:
        debug.resolution_kind = RESOLUTION_NO_CONSENSUS
    debug.round_used = 1
    return router, post_handoffs, debug


# =============================================================================
# Helpers
# =============================================================================


def _class_to_reply_kind(cls: str) -> str:
    if cls == "INDEPENDENT_ROOT":
        return REPLY_INDEPENDENT_ROOT
    if cls.startswith("DOWNSTREAM_SYMPTOM_OF:"):
        return REPLY_DOWNSTREAM_SYMPTOM
    return REPLY_UNCERTAIN


# =============================================================================
# Grading / scoring
# =============================================================================


def grade_nested(scenario: NestedScenario,
                 handoffs_for_decoder: Sequence[TypedHandoff]) -> dict:
    """Apply the Phase-35 decoder on the delivered handoff bundle
    and compare to the scenario's gold.

    Returns a dict with the standard grading keys plus the
    decoder mode and selected claim.
    """
    cue = decoder_from_handoffs_phase35(handoffs_for_decoder)
    lbl = cue["root_cause"]
    svc = set(cue["services"])
    rem = cue["remediation"]
    rc_ok = lbl == scenario.gold_root_cause
    svc_ok = svc == set(scenario.gold_services)
    rem_ok = rem == scenario.gold_remediation
    return {
        "root_cause_correct": rc_ok,
        "services_correct": svc_ok,
        "remediation_correct": rem_ok,
        "full_correct": rc_ok and svc_ok and rem_ok,
        "decoder_mode": cue["decoder_mode"],
        "selected_claim_kind": cue.get("selected_claim_kind"),
    }


# Strategy tags
STRATEGY_NESTED_DYNAMIC = "dynamic_nested_2r"
STRATEGY_NESTED_ADAPTIVE_2R = "adaptive_sub_2r"
STRATEGY_NESTED_ADAPTIVE_1R = "adaptive_sub_1r"
STRATEGY_NESTED_STATIC = "static_handoff"

NESTED_ALL_STRATEGIES = (
    STRATEGY_NESTED_STATIC, STRATEGY_NESTED_ADAPTIVE_1R,
    STRATEGY_NESTED_ADAPTIVE_2R, STRATEGY_NESTED_DYNAMIC,
)


@dataclass
class NestedMeasurement:
    scenario_id: str
    strategy: str
    grading: dict
    debug: dict
    handoff_log_length: int
    wall_seconds: float

    def as_dict(self) -> dict:
        return {
            "scenario_id": self.scenario_id,
            "strategy": self.strategy,
            "grading": {k: v for k, v in self.grading.items()
                         if k != "parsed"},
            "debug": self.debug,
            "handoff_log_length": self.handoff_log_length,
            "wall_seconds": round(self.wall_seconds, 3),
        }


def _static_nested(scenario: NestedScenario,
                    max_events_per_role: int,
                    inbox_capacity: int,
                    ) -> tuple[object, tuple[TypedHandoff, ...]]:
    subs = build_phase35_subscriptions()
    base = HandoffRouter(subs=subs)
    for role in ALL_ROLES:
        base.register_inbox(RoleInbox(role=role,
                                        capacity=inbox_capacity))
    _run_handoff_prelude(scenario, base, max_events_per_role)
    auditor_inbox = base.inboxes.get(ROLE_AUDITOR)
    handoffs = tuple(auditor_inbox.peek()) if auditor_inbox else ()
    return base, handoffs


def run_nested_bank(scenarios: Sequence[NestedScenario],
                     strategies: Sequence[str] = NESTED_ALL_STRATEGIES,
                     max_events_per_role: int = 200,
                     inbox_capacity: int = 32,
                     witness_token_cap: int = 12,
                     ) -> list[NestedMeasurement]:
    out: list[NestedMeasurement] = []
    for scenario in scenarios:
        for strat in strategies:
            t0 = time.time()
            if strat == STRATEGY_NESTED_STATIC:
                _, handoffs = _static_nested(
                    scenario, max_events_per_role, inbox_capacity)
                debug = {"strategy": strat}
                log_len = 0
            elif strat == STRATEGY_NESTED_ADAPTIVE_1R:
                router, handoffs, dbg = \
                    run_nested_one_round_adaptive_sub(
                        scenario,
                        witness_token_cap=witness_token_cap,
                        max_events_per_role=max_events_per_role,
                        inbox_capacity=inbox_capacity)
                debug = _debug_to_dict(dbg)
                log_len = router.log_length()
            elif strat == STRATEGY_NESTED_ADAPTIVE_2R:
                router, handoffs, dbg = \
                    run_nested_two_round_adaptive_sub(
                        scenario,
                        witness_token_cap=witness_token_cap,
                        max_events_per_role=max_events_per_role,
                        inbox_capacity=inbox_capacity)
                debug = _debug_to_dict(dbg)
                log_len = router.log_length()
            elif strat == STRATEGY_NESTED_DYNAMIC:
                router, handoffs, dbg = \
                    run_nested_two_round_thread(
                        scenario,
                        witness_token_cap=witness_token_cap,
                        max_events_per_role=max_events_per_role,
                        inbox_capacity=inbox_capacity)
                debug = _debug_to_dict(dbg)
                log_len = router.log_length()
            else:
                raise ValueError(f"unknown strategy {strat!r}")
            grading = grade_nested(scenario, handoffs)
            wall = time.time() - t0
            out.append(NestedMeasurement(
                scenario_id=scenario.scenario_id,
                strategy=strat, grading=grading, debug=debug,
                handoff_log_length=log_len,
                wall_seconds=wall,
            ))
    return out


def _debug_to_dict(d: NestedCoordinationDebug) -> dict:
    return {
        "strategy": d.strategy,
        "contested_candidates": list(d.contested_candidates),
        "thread_id": d.thread_id,
        "round1_replies": [list(r) for r in d.round1_replies],
        "round2_replies": [list(r) for r in d.round2_replies],
        "n_hypothesis_edges_installed": d.n_hypothesis_edges_installed,
        "n_briefings_installed": d.n_briefings_installed,
        "resolution_kind": d.resolution_kind,
        "resolution_winner": list(d.resolution_winner)
            if d.resolution_winner else None,
        "resolved_claim_idx": d.resolved_claim_idx,
        "round_used": d.round_used,
    }


__all__ = [
    "NestedScenario", "NestedCausalityMap",
    "NESTED_SCENARIO_BUILDERS", "build_nested_bank",
    "nested_round_oracle",
    "run_nested_two_round_thread",
    "run_nested_one_round_adaptive_sub",
    "run_nested_two_round_adaptive_sub",
    "CLAIM_COORDINATION_BRIEFING",
    "STRATEGY_NESTED_STATIC", "STRATEGY_NESTED_ADAPTIVE_1R",
    "STRATEGY_NESTED_ADAPTIVE_2R", "STRATEGY_NESTED_DYNAMIC",
    "NESTED_ALL_STRATEGIES",
    "NestedMeasurement", "NestedCoordinationDebug",
    "run_nested_bank", "grade_nested",
]
