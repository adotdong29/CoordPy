"""Phase 31 — multi-role operational incident triage benchmark.

This module is the programme's first task-scale benchmark whose subject
is NOT a code corpus. It instantiates the general-agent-team thesis on
an operational incident-analysis scenario: a team of five role-typed
agents, each holding a different slice of telemetry, must collaborate
to identify the root cause of a cascading outage, enumerate the
affected services, and name a remediation action.

Why this task family

  * Each role owns *different* observables — the SRE holds uptime
    metrics, the DBA holds SQL query statistics, the sysadmin holds
    OS / disk events, the network engineer holds firewall + DNS
    traces, the auditor owns the final report.
  * The correct final report depends on cross-role *communication*:
    no single role can reconstruct the root cause from its own
    telemetry alone.
  * The task is structurally typed: each role's concern is a
    pre-declared set of *claim kinds* (e.g. the DBA cares about
    ``SLOW_QUERY_OBSERVED`` and ``DEADLOCK_SUSPECTED``). This is the
    precondition of Theorem P30-1 (structural-typing irrelevance
    bound) on non-code data.
  * The task does not collapse to code indexing — it is driven by a
    deterministic *incident generator* that manufactures per-role
    telemetry streams with a hidden causal chain.

Scenario family

  Each incident is one of a small catalogue of realistic cascade
  patterns (``disk_fill_cron`` / ``slow_query_pool_exhaustion`` /
  ``dns_misroute_leak`` / ``tls_expiry_healthcheck_loop`` /
  ``memory_leak_oom``). Every pattern defines:

    * ``root_cause_claim``        — the claim the auditor must emit.
    * ``affected_services``        — the set of services impacted.
    * ``remediation_kind``          — the action the auditor must name.
    * ``per_role_relevant_events`` — the events across roles that are
      oracle-causally-relevant to the gold answer.
    * ``distractor_events``        — per-role background of normal-
      operation events that look plausible under naive broadcast but
      are not causally load-bearing.

Scope discipline (what this module does NOT claim)

  * Not a substitute for SWE-bench. The task family is synthetic and
    structurally typed by construction; it is deliberately the kind
    of task Conjecture P30-5 / C31-6 predict the substrate should
    dominate on.
  * Not a claim about real-world incident investigations: agents
    here are deterministic oracles / small LLMs, not human SREs.
  * Not a claim that typed handoffs are always sufficient: the
    claim catalogue is finite and task-specific. An adversarial task
    ("summarise every event you have ever seen") would still
    bottom out on the naive cost (§ 1.4 / OQ-11 of the master plan).

Theoretical anchor: RESULTS_PHASE31.md § B.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

from ..core.role_handoff import (
    DeliveryAccount, HandoffLog, HandoffRouter, RoleInbox,
    RoleSubscriptionTable, TypedHandoff,
)


# =============================================================================
# Role registry
# =============================================================================


ROLE_MONITOR = "monitor"
ROLE_DB_ADMIN = "db_admin"
ROLE_SYSADMIN = "sysadmin"
ROLE_NETWORK = "network"
ROLE_AUDITOR = "auditor"

ALL_ROLES = (ROLE_MONITOR, ROLE_DB_ADMIN, ROLE_SYSADMIN,
             ROLE_NETWORK, ROLE_AUDITOR)


# =============================================================================
# Event schema (non-code — this is an *operational* event stream)
# =============================================================================


# Raw event kinds — what a role *observes* before any claim extraction.
# Kept short + enumerated so the Bloom-filter routing condition is clean.
EVENT_METRIC_SAMPLE = "METRIC_SAMPLE"
EVENT_LOG_LINE = "LOG_LINE"
EVENT_SQL_STAT = "SQL_STAT"
EVENT_OS_EVENT = "OS_EVENT"
EVENT_NET_FLOW = "NET_FLOW"
EVENT_FW_RULE_HIT = "FW_RULE_HIT"
EVENT_DNS_QUERY = "DNS_QUERY"
EVENT_TASK_GOAL = "TASK_GOAL"
EVENT_FINAL_ANSWER = "FINAL_ANSWER"

FIXED_POINT_EVENT_TYPES = frozenset({EVENT_TASK_GOAL, EVENT_FINAL_ANSWER})


# Role-specific observable types — who can ever see what.
ROLE_OBSERVABLE_TYPES: dict[str, frozenset[str]] = {
    ROLE_MONITOR: frozenset(
        {EVENT_METRIC_SAMPLE} | FIXED_POINT_EVENT_TYPES),
    ROLE_DB_ADMIN: frozenset(
        {EVENT_SQL_STAT, EVENT_LOG_LINE} | FIXED_POINT_EVENT_TYPES),
    ROLE_SYSADMIN: frozenset(
        {EVENT_OS_EVENT, EVENT_LOG_LINE} | FIXED_POINT_EVENT_TYPES),
    ROLE_NETWORK: frozenset(
        {EVENT_NET_FLOW, EVENT_FW_RULE_HIT, EVENT_DNS_QUERY}
        | FIXED_POINT_EVENT_TYPES),
    ROLE_AUDITOR: frozenset(FIXED_POINT_EVENT_TYPES),  # no raw telemetry
}


@dataclass(frozen=True)
class IncidentEvent:
    """One raw observed event in the incident timeline.

    ``origin_role`` is the role whose telemetry slice this event
    appears in; under naive broadcast every role still receives
    every event, but under routing the Bloom-filter subscription
    cuts by event type + role.
    """

    event_id: int
    event_type: str
    origin_role: str
    body: str
    timestamp: float = 0.0
    tags: tuple[str, ...] = ()     # load-bearing tags (service name, etc.)
    is_causal: bool = False        # oracle: True iff part of the causal chain

    @property
    def n_tokens(self) -> int:
        if not self.body:
            return 0
        return max(1, len(self.body.split()))

    @property
    def is_fixed_point(self) -> bool:
        return self.event_type in FIXED_POINT_EVENT_TYPES


# =============================================================================
# Claim taxonomy — what a role can *hand off* to another role
# =============================================================================


# Claims live at the *typed-handoff* layer. The claim kind is short,
# enumerated, and designed to be the granularity at which downstream
# roles subscribe.
CLAIM_ERROR_RATE_SPIKE = "ERROR_RATE_SPIKE"
CLAIM_LATENCY_SPIKE = "LATENCY_SPIKE"
CLAIM_SLOW_QUERY_OBSERVED = "SLOW_QUERY_OBSERVED"
CLAIM_POOL_EXHAUSTION = "POOL_EXHAUSTION"
CLAIM_DEADLOCK_SUSPECTED = "DEADLOCK_SUSPECTED"
CLAIM_DISK_FILL_CRITICAL = "DISK_FILL_CRITICAL"
CLAIM_CRON_OVERRUN = "CRON_OVERRUN"
CLAIM_OOM_KILL = "OOM_KILL"
CLAIM_TLS_EXPIRED = "TLS_EXPIRED"
CLAIM_DNS_MISROUTE = "DNS_MISROUTE"
CLAIM_FW_BLOCK_SURGE = "FW_BLOCK_SURGE"

ALL_CLAIMS = (
    CLAIM_ERROR_RATE_SPIKE, CLAIM_LATENCY_SPIKE,
    CLAIM_SLOW_QUERY_OBSERVED, CLAIM_POOL_EXHAUSTION,
    CLAIM_DEADLOCK_SUSPECTED, CLAIM_DISK_FILL_CRITICAL,
    CLAIM_CRON_OVERRUN, CLAIM_OOM_KILL, CLAIM_TLS_EXPIRED,
    CLAIM_DNS_MISROUTE, CLAIM_FW_BLOCK_SURGE,
)


def build_role_subscriptions() -> RoleSubscriptionTable:
    """The default "who should know what" declaration for this task
    family. Every claim has the auditor as at least one consumer;
    some claims flow laterally (e.g. ``DISK_FILL_CRITICAL`` also goes
    to the DBA, because a full disk can cause ``SLOW_QUERY_OBSERVED``
    downstream).

    The subscription table is part of the task specification: the
    protocol contract agents implement. Under the typed-handoff
    strategy, the ``HandoffRouter`` uses this table to decide which
    role's inbox each handoff enters.
    """
    subs = RoleSubscriptionTable()
    # Monitor → auditor (always)
    subs.subscribe(ROLE_MONITOR, CLAIM_ERROR_RATE_SPIKE, [ROLE_AUDITOR])
    subs.subscribe(ROLE_MONITOR, CLAIM_LATENCY_SPIKE, [ROLE_AUDITOR])
    # DB admin → auditor; disk-fill pressure also interesting to DBA
    subs.subscribe(ROLE_DB_ADMIN, CLAIM_SLOW_QUERY_OBSERVED, [ROLE_AUDITOR])
    subs.subscribe(ROLE_DB_ADMIN, CLAIM_POOL_EXHAUSTION, [ROLE_AUDITOR])
    subs.subscribe(ROLE_DB_ADMIN, CLAIM_DEADLOCK_SUSPECTED, [ROLE_AUDITOR])
    # Sysadmin → auditor; disk-fill / cron-overrun also relevant to DBA
    subs.subscribe(ROLE_SYSADMIN, CLAIM_DISK_FILL_CRITICAL,
                   [ROLE_AUDITOR, ROLE_DB_ADMIN])
    subs.subscribe(ROLE_SYSADMIN, CLAIM_CRON_OVERRUN, [ROLE_AUDITOR])
    subs.subscribe(ROLE_SYSADMIN, CLAIM_OOM_KILL, [ROLE_AUDITOR])
    # Network → auditor; DNS misroute also relevant to SRE
    subs.subscribe(ROLE_NETWORK, CLAIM_TLS_EXPIRED, [ROLE_AUDITOR])
    subs.subscribe(ROLE_NETWORK, CLAIM_DNS_MISROUTE,
                   [ROLE_AUDITOR, ROLE_MONITOR])
    subs.subscribe(ROLE_NETWORK, CLAIM_FW_BLOCK_SURGE, [ROLE_AUDITOR])
    return subs


# =============================================================================
# Scenario catalogue — deterministic generators
# =============================================================================


@dataclass(frozen=True)
class IncidentScenario:
    """One deterministic scenario in the catalogue.

    Fields:
      * ``scenario_id``      — short string key.
      * ``description``      — human-readable one-liner.
      * ``gold_root_cause``   — canonical string the auditor must
        produce. Used by ``grade_answer``.
      * ``gold_services``     — sorted tuple of affected service
        names. Gold list answer.
      * ``gold_remediation``  — canonical remediation string.
      * ``causal_chain``      — ordered list of (role, claim_kind,
        payload, source_event_ids) tuples describing the causal
        handoffs that jointly witness the root cause. Every entry
        is load-bearing: removing one breaks the chain.
      * ``per_role_events``   — role → list[IncidentEvent] slices.
        Each slice contains the causal events flagged ``is_causal``
        plus role-specific distractors; the oracle counts
        ``is_causal`` events per role as the ground-truth relevance
        set.
    """

    scenario_id: str
    description: str
    gold_root_cause: str
    gold_services: tuple[str, ...]
    gold_remediation: str
    causal_chain: tuple[tuple[str, str, str, tuple[int, ...]], ...]
    per_role_events: dict[str, tuple[IncidentEvent, ...]]


def _mk_event(next_id: int, et: str, role: str, body: str,
              *, tags: Sequence[str] = (), causal: bool = False,
              ) -> tuple[IncidentEvent, int]:
    ev = IncidentEvent(
        event_id=next_id, event_type=et, origin_role=role,
        body=body, tags=tuple(tags), is_causal=causal)
    return ev, next_id + 1


def _distractors(rng: random.Random, next_id: int,
                 role: str, kinds: Sequence[str],
                 k: int) -> tuple[list[IncidentEvent], int]:
    """Deterministic background chatter for role ``role`` — ``k``
    events drawn from ``kinds`` with benign payloads."""
    benign = {
        EVENT_METRIC_SAMPLE: [
            "cpu_pct=12 service=web", "cpu_pct=9 service=api",
            "rps=480 service=web ok=true",
            "rps=302 service=api ok=true",
            "mem_pct=44 service=web", "p95_ms=95 service=api",
        ],
        EVENT_LOG_LINE: [
            "info nightly_backup starting",
            "info user_login=u42 ok=true",
            "debug cache_hit=0.91 service=web",
            "info shard_rebalance step=3/10",
        ],
        EVENT_SQL_STAT: [
            "q#3 mean_ms=8 service=catalog ok=true",
            "q#7 mean_ms=15 service=orders ok=true",
            "q#11 mean_ms=12 service=users ok=true",
            "q#14 mean_ms=6 service=sessions ok=true",
        ],
        EVENT_OS_EVENT: [
            "kworker cpu=0 usage=6%",
            "cron backup exit=0 service=archival",
            "systemd unit=app.service active=yes",
        ],
        EVENT_NET_FLOW: [
            "src=10.0.0.4 dst=10.0.0.12 bytes=2100 ok=true",
            "src=10.0.0.5 dst=10.0.0.14 bytes=1900 ok=true",
        ],
        EVENT_FW_RULE_HIT: [
            "rule=allow src=10.0.0.0/24 action=accept",
        ],
        EVENT_DNS_QUERY: [
            "q=api.internal type=A rc=NOERROR rtt_ms=3",
            "q=db.internal type=A rc=NOERROR rtt_ms=2",
        ],
    }
    out: list[IncidentEvent] = []
    for _ in range(k):
        et = rng.choice(list(kinds))
        body = rng.choice(benign.get(et, ["benign"]))
        ev, next_id = _mk_event(next_id, et, role, body, causal=False)
        out.append(ev)
    return out, next_id


def make_scenario_disk_fill(rng: random.Random, start_id: int = 0,
                             distractors_per_role: int = 6,
                             ) -> IncidentScenario:
    """A ``disk_fill_cron`` scenario.

    Causal chain:
      sysadmin: cron job overran → disk fill
      db_admin: logs → slow query → pool exhaustion
      monitor:  error-rate spike
      network:  (not causal — distractors only)
      auditor:  gold = ``disk_fill``, services = [web, api, orders],
                remediation = ``rotate_logs_and_clear_backup``.
    """
    nid = start_id
    per_role: dict[str, list[IncidentEvent]] = {r: [] for r in ALL_ROLES}
    chain: list[tuple[str, str, str, tuple[int, ...]]] = []

    # sysadmin causal events — payload carries cron / disk facts only.
    # Producer-role "service" tags (e.g. archival, host) are NOT added
    # because under the decoder they would leak into the auditor's
    # impacted-services set; by convention only downstream symptoms
    # contribute service tags.
    ev, nid = _mk_event(nid, EVENT_OS_EVENT, ROLE_SYSADMIN,
                         "cron backup.sh exit=137 duration_s=5400",
                         causal=True)
    per_role[ROLE_SYSADMIN].append(ev)
    cron_ev_id = ev.event_id
    ev, nid = _mk_event(nid, EVENT_OS_EVENT, ROLE_SYSADMIN,
                         "/var/log used=99% fs=/",
                         causal=True)
    per_role[ROLE_SYSADMIN].append(ev)
    disk_ev_id = ev.event_id
    chain.append((ROLE_SYSADMIN, CLAIM_CRON_OVERRUN,
                  "backup.sh exit=137 duration_s=5400",
                  (cron_ev_id,)))
    chain.append((ROLE_SYSADMIN, CLAIM_DISK_FILL_CRITICAL,
                  "/var/log used=99% fs=/",
                  (disk_ev_id,)))

    # db_admin causal events
    ev, nid = _mk_event(nid, EVENT_LOG_LINE, ROLE_DB_ADMIN,
                         "error pg could not write file /var/log: "
                         "No space left on device",
                         tags=("db",), causal=True)
    per_role[ROLE_DB_ADMIN].append(ev)
    db_err = ev.event_id
    ev, nid = _mk_event(nid, EVENT_SQL_STAT, ROLE_DB_ADMIN,
                         "q#12 mean_ms=4210 service=orders calls=800",
                         tags=("orders",), causal=True)
    per_role[ROLE_DB_ADMIN].append(ev)
    slow_q = ev.event_id
    ev, nid = _mk_event(nid, EVENT_SQL_STAT, ROLE_DB_ADMIN,
                         "pool active=200/200 waiters=145 service=api",
                         tags=("api",), causal=True)
    per_role[ROLE_DB_ADMIN].append(ev)
    pool_ev = ev.event_id
    chain.append((ROLE_DB_ADMIN, CLAIM_SLOW_QUERY_OBSERVED,
                  "q#12 mean_ms=4210 service=orders",
                  (db_err, slow_q)))
    chain.append((ROLE_DB_ADMIN, CLAIM_POOL_EXHAUSTION,
                  "pool active=200/200 waiters=145 service=api",
                  (pool_ev,)))

    # monitor causal events
    ev, nid = _mk_event(nid, EVENT_METRIC_SAMPLE, ROLE_MONITOR,
                         "error_rate=0.37 service=web window=5m",
                         tags=("web",), causal=True)
    per_role[ROLE_MONITOR].append(ev)
    err_ev = ev.event_id
    ev, nid = _mk_event(nid, EVENT_METRIC_SAMPLE, ROLE_MONITOR,
                         "p95_ms=9800 service=api window=5m",
                         tags=("api",), causal=True)
    per_role[ROLE_MONITOR].append(ev)
    lat_ev = ev.event_id
    chain.append((ROLE_MONITOR, CLAIM_ERROR_RATE_SPIKE,
                  "error_rate=0.37 service=web",
                  (err_ev,)))
    chain.append((ROLE_MONITOR, CLAIM_LATENCY_SPIKE,
                  "p95_ms=9800 service=api",
                  (lat_ev,)))

    # Distractors — role-specific background. Keep the scenario at a
    # realistic signal / noise ratio.
    for role, kinds in ((ROLE_MONITOR, [EVENT_METRIC_SAMPLE]),
                         (ROLE_DB_ADMIN, [EVENT_SQL_STAT,
                                           EVENT_LOG_LINE]),
                         (ROLE_SYSADMIN, [EVENT_OS_EVENT,
                                           EVENT_LOG_LINE]),
                         (ROLE_NETWORK, [EVENT_NET_FLOW,
                                          EVENT_FW_RULE_HIT,
                                          EVENT_DNS_QUERY])):
        dist, nid = _distractors(rng, nid, role, kinds,
                                   k=distractors_per_role)
        per_role[role].extend(dist)

    return IncidentScenario(
        scenario_id="disk_fill_cron",
        description="cron backup filled /var/log → postgres write failure "
                    "→ slow queries → pool exhaustion → frontend spike",
        gold_root_cause="disk_fill",
        gold_services=("api", "orders", "web"),
        gold_remediation="rotate_logs_and_clear_backup",
        causal_chain=tuple(chain),
        per_role_events={r: tuple(evs) for r, evs in per_role.items()},
    )


def make_scenario_tls_expiry(rng: random.Random, start_id: int = 0,
                              distractors_per_role: int = 6,
                              ) -> IncidentScenario:
    """``tls_expiry_healthcheck_loop`` scenario."""
    nid = start_id
    per_role: dict[str, list[IncidentEvent]] = {r: [] for r in ALL_ROLES}
    chain: list[tuple[str, str, str, tuple[int, ...]]] = []

    # network causal
    ev, nid = _mk_event(nid, EVENT_FW_RULE_HIT, ROLE_NETWORK,
                         "tls handshake fail service=api reason=expired",
                         tags=("api",), causal=True)
    per_role[ROLE_NETWORK].append(ev)
    tls_ev = ev.event_id
    ev, nid = _mk_event(nid, EVENT_FW_RULE_HIT, ROLE_NETWORK,
                         "rule=deny src=hc-probe dst=api action=drop "
                         "count=1200 window=5m",
                         tags=("api",), causal=True)
    per_role[ROLE_NETWORK].append(ev)
    fw_ev = ev.event_id
    chain.append((ROLE_NETWORK, CLAIM_TLS_EXPIRED,
                  "tls service=api reason=expired",
                  (tls_ev,)))
    chain.append((ROLE_NETWORK, CLAIM_FW_BLOCK_SURGE,
                  "rule=deny src=hc-probe dst=api count=1200",
                  (fw_ev,)))

    # monitor
    ev, nid = _mk_event(nid, EVENT_METRIC_SAMPLE, ROLE_MONITOR,
                         "uptime_pct=41 service=api window=5m",
                         tags=("api",), causal=True)
    per_role[ROLE_MONITOR].append(ev)
    up_ev = ev.event_id
    chain.append((ROLE_MONITOR, CLAIM_ERROR_RATE_SPIKE,
                  "uptime_pct=41 service=api",
                  (up_ev,)))

    # distractors
    for role, kinds in ((ROLE_MONITOR, [EVENT_METRIC_SAMPLE]),
                         (ROLE_DB_ADMIN, [EVENT_SQL_STAT,
                                           EVENT_LOG_LINE]),
                         (ROLE_SYSADMIN, [EVENT_OS_EVENT]),
                         (ROLE_NETWORK, [EVENT_NET_FLOW,
                                          EVENT_DNS_QUERY])):
        dist, nid = _distractors(rng, nid, role, kinds,
                                   k=distractors_per_role)
        per_role[role].extend(dist)

    return IncidentScenario(
        scenario_id="tls_expiry_healthcheck_loop",
        description="expired TLS cert on api → healthcheck probes fail "
                    "→ firewall rule triggers rate-limit → uptime drop",
        gold_root_cause="tls_expiry",
        gold_services=("api",),
        gold_remediation="renew_tls_and_reload",
        causal_chain=tuple(chain),
        per_role_events={r: tuple(evs) for r, evs in per_role.items()},
    )


def make_scenario_dns_misroute(rng: random.Random, start_id: int = 0,
                                distractors_per_role: int = 6,
                                ) -> IncidentScenario:
    """``dns_misroute_leak`` scenario."""
    nid = start_id
    per_role: dict[str, list[IncidentEvent]] = {r: [] for r in ALL_ROLES}
    chain: list[tuple[str, str, str, tuple[int, ...]]] = []

    ev, nid = _mk_event(nid, EVENT_DNS_QUERY, ROLE_NETWORK,
                         "q=db.internal rc=SERVFAIL rtt_ms=2999",
                         tags=("db",), causal=True)
    per_role[ROLE_NETWORK].append(ev)
    dns_ev = ev.event_id
    chain.append((ROLE_NETWORK, CLAIM_DNS_MISROUTE,
                  "q=db.internal rc=SERVFAIL", (dns_ev,)))

    ev, nid = _mk_event(nid, EVENT_LOG_LINE, ROLE_DB_ADMIN,
                         "error psycopg2 could not connect host=db.internal",
                         tags=("db",), causal=True)
    per_role[ROLE_DB_ADMIN].append(ev)
    db_err = ev.event_id
    ev, nid = _mk_event(nid, EVENT_SQL_STAT, ROLE_DB_ADMIN,
                         "pool active=0/200 reconnect_attempts=842 "
                         "service=orders",
                         tags=("orders",), causal=True)
    per_role[ROLE_DB_ADMIN].append(ev)
    db_pool = ev.event_id
    chain.append((ROLE_DB_ADMIN, CLAIM_POOL_EXHAUSTION,
                  "reconnect_attempts=842 service=orders",
                  (db_err, db_pool)))

    ev, nid = _mk_event(nid, EVENT_METRIC_SAMPLE, ROLE_MONITOR,
                         "error_rate=0.88 service=orders window=5m",
                         tags=("orders",), causal=True)
    per_role[ROLE_MONITOR].append(ev)
    m_ev = ev.event_id
    chain.append((ROLE_MONITOR, CLAIM_ERROR_RATE_SPIKE,
                  "error_rate=0.88 service=orders", (m_ev,)))

    for role, kinds in ((ROLE_MONITOR, [EVENT_METRIC_SAMPLE]),
                         (ROLE_DB_ADMIN, [EVENT_SQL_STAT,
                                           EVENT_LOG_LINE]),
                         (ROLE_SYSADMIN, [EVENT_OS_EVENT]),
                         (ROLE_NETWORK, [EVENT_NET_FLOW,
                                          EVENT_FW_RULE_HIT])):
        dist, nid = _distractors(rng, nid, role, kinds,
                                   k=distractors_per_role)
        per_role[role].extend(dist)

    return IncidentScenario(
        scenario_id="dns_misroute_leak",
        description="dns config drift → db.internal SERVFAIL → orders "
                    "service reconnect storm → error-rate spike",
        gold_root_cause="dns_misroute",
        gold_services=("orders",),
        gold_remediation="restore_internal_dns_zone",
        causal_chain=tuple(chain),
        per_role_events={r: tuple(evs) for r, evs in per_role.items()},
    )


def make_scenario_memory_leak(rng: random.Random, start_id: int = 0,
                               distractors_per_role: int = 6,
                               ) -> IncidentScenario:
    """``memory_leak_oom`` scenario."""
    nid = start_id
    per_role: dict[str, list[IncidentEvent]] = {r: [] for r in ALL_ROLES}
    chain: list[tuple[str, str, str, tuple[int, ...]]] = []

    ev, nid = _mk_event(nid, EVENT_OS_EVENT, ROLE_SYSADMIN,
                         "oom_kill pid=8842 comm=app.py rss=8.1G "
                         "service=api",
                         tags=("api",), causal=True)
    per_role[ROLE_SYSADMIN].append(ev)
    oom_ev = ev.event_id
    chain.append((ROLE_SYSADMIN, CLAIM_OOM_KILL,
                  "oom_kill comm=app.py rss=8.1G service=api",
                  (oom_ev,)))

    ev, nid = _mk_event(nid, EVENT_METRIC_SAMPLE, ROLE_MONITOR,
                         "mem_pct=97 service=api window=5m",
                         tags=("api",), causal=True)
    per_role[ROLE_MONITOR].append(ev)
    m_ev = ev.event_id
    ev, nid = _mk_event(nid, EVENT_METRIC_SAMPLE, ROLE_MONITOR,
                         "p95_ms=4700 service=api window=5m",
                         tags=("api",), causal=True)
    per_role[ROLE_MONITOR].append(ev)
    m2 = ev.event_id
    chain.append((ROLE_MONITOR, CLAIM_LATENCY_SPIKE,
                  "p95_ms=4700 service=api", (m_ev, m2)))

    for role, kinds in ((ROLE_MONITOR, [EVENT_METRIC_SAMPLE]),
                         (ROLE_DB_ADMIN, [EVENT_SQL_STAT,
                                           EVENT_LOG_LINE]),
                         (ROLE_SYSADMIN, [EVENT_OS_EVENT,
                                           EVENT_LOG_LINE]),
                         (ROLE_NETWORK, [EVENT_NET_FLOW,
                                          EVENT_DNS_QUERY])):
        dist, nid = _distractors(rng, nid, role, kinds,
                                   k=distractors_per_role)
        per_role[role].extend(dist)

    return IncidentScenario(
        scenario_id="memory_leak_oom",
        description="slow memory leak in app.py → OOM-killer → api "
                    "latency spike",
        gold_root_cause="memory_leak",
        gold_services=("api",),
        gold_remediation="rollback_app_to_prev_release",
        causal_chain=tuple(chain),
        per_role_events={r: tuple(evs) for r, evs in per_role.items()},
    )


def make_scenario_deadlock(rng: random.Random, start_id: int = 0,
                            distractors_per_role: int = 6,
                            ) -> IncidentScenario:
    """``deadlock_pool_exhaustion`` scenario."""
    nid = start_id
    per_role: dict[str, list[IncidentEvent]] = {r: [] for r in ALL_ROLES}
    chain: list[tuple[str, str, str, tuple[int, ...]]] = []

    ev, nid = _mk_event(nid, EVENT_LOG_LINE, ROLE_DB_ADMIN,
                         "warning pg deadlock detected pid=8823 "
                         "relation=orders_payments",
                         tags=("orders",), causal=True)
    per_role[ROLE_DB_ADMIN].append(ev)
    dl_ev = ev.event_id
    ev, nid = _mk_event(nid, EVENT_SQL_STAT, ROLE_DB_ADMIN,
                         "pool active=200/200 waiters=88 service=orders",
                         tags=("orders",), causal=True)
    per_role[ROLE_DB_ADMIN].append(ev)
    pool_ev = ev.event_id
    chain.append((ROLE_DB_ADMIN, CLAIM_DEADLOCK_SUSPECTED,
                  "deadlock relation=orders_payments", (dl_ev,)))
    chain.append((ROLE_DB_ADMIN, CLAIM_POOL_EXHAUSTION,
                  "pool active=200/200 waiters=88 service=orders",
                  (pool_ev,)))

    ev, nid = _mk_event(nid, EVENT_METRIC_SAMPLE, ROLE_MONITOR,
                         "error_rate=0.22 service=orders window=5m",
                         tags=("orders",), causal=True)
    per_role[ROLE_MONITOR].append(ev)
    m_ev = ev.event_id
    chain.append((ROLE_MONITOR, CLAIM_ERROR_RATE_SPIKE,
                  "error_rate=0.22 service=orders", (m_ev,)))

    for role, kinds in ((ROLE_MONITOR, [EVENT_METRIC_SAMPLE]),
                         (ROLE_DB_ADMIN, [EVENT_SQL_STAT,
                                           EVENT_LOG_LINE]),
                         (ROLE_SYSADMIN, [EVENT_OS_EVENT]),
                         (ROLE_NETWORK, [EVENT_NET_FLOW,
                                          EVENT_DNS_QUERY])):
        dist, nid = _distractors(rng, nid, role, kinds,
                                   k=distractors_per_role)
        per_role[role].extend(dist)

    return IncidentScenario(
        scenario_id="deadlock_pool_exhaustion",
        description="lock-order bug between orders and payments → "
                    "deadlock storm → pool exhaustion",
        gold_root_cause="deadlock",
        gold_services=("orders",),
        gold_remediation="enforce_lock_ordering_in_orders",
        causal_chain=tuple(chain),
        per_role_events={r: tuple(evs) for r, evs in per_role.items()},
    )


SCENARIO_BUILDERS: tuple[
    Callable[..., IncidentScenario], ...] = (
    make_scenario_disk_fill,
    make_scenario_tls_expiry,
    make_scenario_dns_misroute,
    make_scenario_memory_leak,
    make_scenario_deadlock,
)


def build_scenario_bank(seed: int = 31,
                         distractors_per_role: int = 6,
                         ) -> list[IncidentScenario]:
    """Return the deterministic scenario bank (one scenario per
    builder). A caller can select a subset by ``scenario_id``.

    ``distractors_per_role`` controls the signal/noise ratio: each
    role accumulates this many background (benign) events in
    addition to the causal events. The Phase-31 benchmark sweeps
    this to measure the substrate's resilience as the naive stream
    grows without bound.
    """
    rng = random.Random(seed)
    out: list[IncidentScenario] = []
    next_id = 0
    for builder in SCENARIO_BUILDERS:
        s = builder(rng, next_id,
                    distractors_per_role=distractors_per_role)
        out.append(s)
        # The scenario event ids live in its own namespace; we
        # reset next_id across scenarios so the event-id sequence
        # restarts. This keeps each scenario self-contained in the
        # ``deliver`` path.
        next_id = 0
    return out


# =============================================================================
# Global event-stream assembly (naive broadcast view)
# =============================================================================


def fixed_point_events(scenario: IncidentScenario
                        ) -> list[IncidentEvent]:
    """Return the task-goal + final-answer fixed-points for a
    scenario. Ids are chosen to be disjoint from role-event ids
    (``-1`` / ``-2``)."""
    task_goal = IncidentEvent(
        event_id=-1, event_type=EVENT_TASK_GOAL,
        origin_role="__system__",
        body=(f"[task] incident {scenario.scenario_id}: identify root "
              f"cause, affected services, and remediation"))
    final = IncidentEvent(
        event_id=-2, event_type=EVENT_FINAL_ANSWER,
        origin_role="__system__",
        body="[final] <placeholder>")
    return [task_goal, final]


def naive_event_stream(scenario: IncidentScenario) -> list[IncidentEvent]:
    """The event stream a naive broadcast bus would emit across the
    entire team — every role's telemetry plus the fixed points.

    Event-id ordering is (fixed points) ++ (role events in
    ALL_ROLES order); the aggregator's prompt truncation thresholds
    use this ordering.
    """
    out: list[IncidentEvent] = list(fixed_point_events(scenario))
    for role in ALL_ROLES:
        out.extend(scenario.per_role_events.get(role, ()))
    return out


# =============================================================================
# Oracle — per-(scenario, role, event) causal relevance
# =============================================================================


def _event_is_causally_relevant_to_auditor(
        ev: IncidentEvent, scenario: IncidentScenario) -> bool:
    if ev.is_fixed_point:
        return True
    return bool(ev.is_causal)


def oracle_relevance(ev: IncidentEvent,
                     role: str,
                     scenario: IncidentScenario) -> bool:
    """Per-(event, role, scenario) causal relevance.

    * Fixed-point events are relevant to every role.
    * A raw event is relevant to the role that owns its observable
      type *only if* it is part of the scenario's causal chain.
    * The auditor role has no raw telemetry — relevance is decided
      at the *handoff* layer (see ``handoff_is_relevant``), not at
      the raw-event layer. For the auditor, raw events are
      relevant only when ``is_causal`` is True (as a structural
      upper bound on the handoff's ``source_event_ids`` set).
    """
    if ev.is_fixed_point:
        return True
    obs = ROLE_OBSERVABLE_TYPES.get(role, frozenset())
    if ev.event_type not in obs and role != ROLE_AUDITOR:
        return False
    if role == ROLE_AUDITOR:
        # Auditor only "sees" a raw event iff it is load-bearing.
        return bool(ev.is_causal)
    # Role can observe this type — but only causal observations are
    # relevant to the role's own gold. Distractors are irrelevant.
    return bool(ev.is_causal) and ev.origin_role == role


def handoff_is_relevant(h: TypedHandoff,
                        scenario: IncidentScenario) -> bool:
    """Oracle: is a given typed handoff load-bearing for the gold?

    A handoff is relevant iff its (source_role, claim_kind) appears
    in the scenario's ``causal_chain``. This is the typed-handoff
    analogue of the Phase-29 causal-relevance predicate.
    """
    chain_pairs = {(role, kind)
                   for (role, kind, _p, _evs) in scenario.causal_chain}
    return (h.source_role, h.claim_kind) in chain_pairs


# =============================================================================
# Claim extractors — deterministic, per-role
# =============================================================================


def extract_claims_for_role(role: str,
                             events: Sequence[IncidentEvent],
                             scenario: IncidentScenario,
                             ) -> list[tuple[str, str, tuple[int, ...]]]:
    """Deterministic rules for converting a role's delivered events
    into typed claims.

    The extractor is per-role and keyed on raw-event tags; a claim
    is emitted iff at least one causal event with the scenario's
    signature appears in ``events``. Distractors never produce a
    claim because their body does not match the extractor's regex.
    """
    out: list[tuple[str, str, tuple[int, ...]]] = []
    ev_by_id = {ev.event_id: ev for ev in events}

    def _emit(kind: str, body: str, evids: Sequence[int]) -> None:
        out.append((kind, body, tuple(evids)))

    if role == ROLE_MONITOR:
        for ev in events:
            if ev.event_type != EVENT_METRIC_SAMPLE:
                continue
            m = re.search(r"error_rate=(\d+\.\d+)", ev.body)
            if m and float(m.group(1)) >= 0.10:
                _emit(CLAIM_ERROR_RATE_SPIKE, ev.body, [ev.event_id])
            m = re.search(r"p95_ms=(\d+)", ev.body)
            if m and int(m.group(1)) >= 1000:
                _emit(CLAIM_LATENCY_SPIKE, ev.body, [ev.event_id])
            m = re.search(r"uptime_pct=(\d+)", ev.body)
            if m and int(m.group(1)) <= 80:
                _emit(CLAIM_ERROR_RATE_SPIKE, ev.body, [ev.event_id])
    elif role == ROLE_DB_ADMIN:
        for ev in events:
            if ev.event_type == EVENT_SQL_STAT:
                m = re.search(r"mean_ms=(\d+)", ev.body)
                if m and int(m.group(1)) >= 1000:
                    _emit(CLAIM_SLOW_QUERY_OBSERVED, ev.body,
                          [ev.event_id])
                if "active=200/200" in ev.body or \
                        "reconnect_attempts=" in ev.body:
                    _emit(CLAIM_POOL_EXHAUSTION, ev.body, [ev.event_id])
            if ev.event_type == EVENT_LOG_LINE:
                if "deadlock" in ev.body:
                    _emit(CLAIM_DEADLOCK_SUSPECTED, ev.body,
                          [ev.event_id])
                if "No space left on device" in ev.body:
                    _emit(CLAIM_SLOW_QUERY_OBSERVED, ev.body,
                          [ev.event_id])
    elif role == ROLE_SYSADMIN:
        for ev in events:
            if ev.event_type != EVENT_OS_EVENT:
                continue
            if "oom_kill" in ev.body:
                _emit(CLAIM_OOM_KILL, ev.body, [ev.event_id])
            if re.search(r"used=(9\d|100)%", ev.body):
                _emit(CLAIM_DISK_FILL_CRITICAL, ev.body, [ev.event_id])
            if re.search(r"cron .* exit=(13\d|1)", ev.body) or \
                    re.search(r"cron .* duration_s=\d{4,}", ev.body):
                _emit(CLAIM_CRON_OVERRUN, ev.body, [ev.event_id])
    elif role == ROLE_NETWORK:
        for ev in events:
            if ev.event_type == EVENT_FW_RULE_HIT:
                if "reason=expired" in ev.body:
                    _emit(CLAIM_TLS_EXPIRED, ev.body, [ev.event_id])
                if re.search(r"count=(\d{3,})", ev.body) and \
                        "action=drop" in ev.body:
                    _emit(CLAIM_FW_BLOCK_SURGE, ev.body, [ev.event_id])
            if ev.event_type == EVENT_DNS_QUERY and "SERVFAIL" in ev.body:
                _emit(CLAIM_DNS_MISROUTE, ev.body, [ev.event_id])
    # Auditor has no raw telemetry and therefore emits no claims.
    return out


# =============================================================================
# Delivery + per-strategy aggregator prompt assembly
# =============================================================================


STRATEGY_NAIVE = "naive"
STRATEGY_ROUTING = "routing"
STRATEGY_SUBSTRATE = "substrate"         # typed handoffs, no LLM wrap
STRATEGY_SUBSTRATE_WRAP = "substrate_wrap"   # typed handoffs + LLM wrap

ALL_STRATEGIES = (STRATEGY_NAIVE, STRATEGY_ROUTING,
                  STRATEGY_SUBSTRATE, STRATEGY_SUBSTRATE_WRAP)


def role_subscribed_events(events: Sequence[IncidentEvent],
                            role: str) -> list[IncidentEvent]:
    """Bloom-filter-routing view for one role.

    Mirrors the role-keyed subscription in ``task_scale_swe`` — by
    observable type only, no content filtering.
    """
    obs = ROLE_OBSERVABLE_TYPES.get(role, frozenset())
    return [ev for ev in events if ev.event_type in obs]


def run_handoff_protocol(scenario: IncidentScenario,
                          max_events_per_role: int = 200,
                          inbox_capacity: int = 32,
                          extractor: Callable[
                              [str, Sequence[IncidentEvent],
                               IncidentScenario],
                              list[tuple[str, str, tuple[int, ...]]],
                          ] | None = None,
                          ) -> HandoffRouter:
    """Drive the typed-handoff substrate for one scenario.

    Each producer role extracts claims from its role-subscribed
    events (i.e. its own telemetry slice, as the routing layer would
    deliver) and emits each claim via a ``HandoffRouter``. The
    auditor role receives only the typed handoffs subscribed to its
    role — not raw events.

    ``extractor`` defaults to ``extract_claims_for_role``; the Phase-32
    noisy-extractor sweep (``core/extractor_noise``) injects a wrapped
    extractor to exercise graceful-degradation bounds.
    """
    subs = build_role_subscriptions()
    router = HandoffRouter(subs=subs)
    for role in ALL_ROLES:
        router.register_inbox(RoleInbox(role=role, capacity=inbox_capacity))

    extractor = extractor or extract_claims_for_role
    for role in (ROLE_MONITOR, ROLE_DB_ADMIN, ROLE_SYSADMIN, ROLE_NETWORK):
        evs = list(scenario.per_role_events.get(role, ()))
        if max_events_per_role:
            evs = evs[:max_events_per_role]
        claims = extractor(role, evs, scenario)
        for (kind, payload, evids) in claims:
            router.emit(
                source_role=role, source_agent_id=ALL_ROLES.index(role),
                claim_kind=kind, payload=payload,
                source_event_ids=evids, round=1)
    return router


# =============================================================================
# Aggregator prompt assembly
# =============================================================================


def _decoder_from_handoffs(
        handoffs: Sequence[TypedHandoff]) -> dict[str, object]:
    """Derive the auditor's structured answer from a bundle of
    typed handoffs.

    Answer fields:
      * ``root_cause``   — most load-bearing claim_kind mapped to a
        canonical root-cause label (per the scenario gold). The
        mapping is {claim_kind → root_cause} and is deterministic.
      * ``services``     — sorted tuple of service names mentioned
        in any handoff's payload.
      * ``remediation``  — canonical remediation string keyed by the
        root-cause label.
    """
    claim_kinds = {h.claim_kind for h in handoffs}
    # Root-cause inference — priority order encodes which claim is
    # causal-primary vs downstream. The priority reflects the
    # scenario catalogue: a DISK_FILL is always upstream of a
    # SLOW_QUERY it causes; likewise OOM_KILL is upstream of latency.
    priority = (
        (CLAIM_DISK_FILL_CRITICAL, "disk_fill",
         "rotate_logs_and_clear_backup"),
        (CLAIM_TLS_EXPIRED, "tls_expiry", "renew_tls_and_reload"),
        (CLAIM_DNS_MISROUTE, "dns_misroute",
         "restore_internal_dns_zone"),
        (CLAIM_OOM_KILL, "memory_leak",
         "rollback_app_to_prev_release"),
        (CLAIM_DEADLOCK_SUSPECTED, "deadlock",
         "enforce_lock_ordering_in_orders"),
        (CLAIM_CRON_OVERRUN, "disk_fill",
         "rotate_logs_and_clear_backup"),
        (CLAIM_POOL_EXHAUSTION, "pool_exhaustion",
         "raise_pool_cap_or_fix_upstream"),
        (CLAIM_SLOW_QUERY_OBSERVED, "slow_query_cascade",
         "index_or_split_slow_query"),
        (CLAIM_ERROR_RATE_SPIKE, "error_spike",
         "roll_back_recent_deploy"),
        (CLAIM_LATENCY_SPIKE, "latency_spike",
         "scale_up_api_pool"),
        (CLAIM_FW_BLOCK_SURGE, "fw_block",
         "rescind_spurious_deny_rule"),
    )
    root_cause = "unknown"
    remediation = "investigate"
    for (kind, label, remed) in priority:
        if kind in claim_kinds:
            root_cause = label
            remediation = remed
            break
    services: set[str] = set()
    for h in handoffs:
        for tok in h.payload.split():
            m = re.search(r"service=(\w+)", tok)
            if m:
                services.add(m.group(1))
    return {
        "root_cause": root_cause,
        "services": tuple(sorted(services)),
        "remediation": remediation,
    }


def _decoder_from_events(events: Sequence[IncidentEvent]
                          ) -> dict[str, object]:
    """Fallback decoder under naive / routing — the aggregator
    tries to infer the answer directly from raw events.

    Emulates a perfect reader-of-delivered-events: it runs the
    per-role claim extractors (for the roles whose events appear
    in ``events``) and then calls ``_decoder_from_handoffs`` on a
    synthetic bundle. Under naive delivery this sees the union of
    all causal events and produces the correct answer — *but only*
    if those causal events survived truncation. Under routing
    delivery the auditor sees only fixed-point events (no raw
    telemetry), which causes the decoder to return ``unknown``.

    This is intentional: the decoder is a *reader*, not an
    overrider of delivery. Any gap between what the auditor could
    do with unlimited context and what it does with the bounded
    delivery falls on the strategy.
    """
    # Split events by origin role and run per-role extractors.
    by_role: dict[str, list[IncidentEvent]] = {r: [] for r in ALL_ROLES}
    for ev in events:
        if ev.origin_role in by_role:
            by_role[ev.origin_role].append(ev)
    synth_handoffs: list[TypedHandoff] = []
    next_id = 0
    for role in (ROLE_MONITOR, ROLE_DB_ADMIN, ROLE_SYSADMIN, ROLE_NETWORK):
        claims = extract_claims_for_role(role, by_role[role],
                                         _DUMMY_SCENARIO)
        for (kind, payload, evids) in claims:
            # Not a real handoff — just the shape the decoder expects.
            synth_handoffs.append(TypedHandoff(
                handoff_id=next_id, source_role=role,
                source_agent_id=ALL_ROLES.index(role),
                to_role=ROLE_AUDITOR, claim_kind=kind, payload=payload,
                source_event_ids=tuple(evids), round=1,
                payload_cid="", prev_chain_hash="", chain_hash=""))
            next_id += 1
    return _decoder_from_handoffs(synth_handoffs)


# A placeholder scenario that the `extract_claims_for_role` caller
# signature expects but does not actually read into (the extractors
# operate on events, not on `scenario`).
_DUMMY_SCENARIO = IncidentScenario(
    scenario_id="__dummy__", description="",
    gold_root_cause="", gold_services=(), gold_remediation="",
    causal_chain=(),
    per_role_events={r: () for r in ALL_ROLES})


# =============================================================================
# Prompt assembly for the auditor
# =============================================================================


def build_auditor_prompt(scenario: IncidentScenario,
                          strategy: str,
                          events: Sequence[IncidentEvent],
                          handoffs: Sequence[TypedHandoff] = (),
                          substrate_cue: dict | None = None,
                          max_events_in_prompt: int = 200,
                          ) -> tuple[str, list[IncidentEvent], bool]:
    """Assemble the auditor's prompt under ``strategy``.

    Returns ``(prompt, delivered_events, truncated)``. Truncation
    is a first-class metric: when the delivered event stream
    exceeds ``max_events_in_prompt`` we clip and set
    ``truncated=True``.
    """
    if strategy == STRATEGY_NAIVE:
        delivered = list(events)
    elif strategy == STRATEGY_ROUTING:
        # Auditor has no raw telemetry subscription — only fixed points.
        delivered = [ev for ev in events if ev.is_fixed_point]
    elif strategy in (STRATEGY_SUBSTRATE, STRATEGY_SUBSTRATE_WRAP):
        delivered = [ev for ev in events if ev.is_fixed_point]
    else:
        raise ValueError(f"unknown strategy {strategy!r}")
    truncated = False
    if len(delivered) > max_events_in_prompt:
        truncated = True
        delivered = delivered[:max_events_in_prompt]
    lines = [
        "You are the AUDITOR in a multi-role incident-response team.",
        "Identify the root cause, list the affected services, and "
        "name a remediation action.",
        ("Respond in three lines exactly, with each line prefixed as "
         "shown:\n  ROOT_CAUSE: <label>\n  SERVICES: "
         "<comma-separated list>\n  REMEDIATION: <label>"),
        "",
        f"SCENARIO: {scenario.description}",
    ]
    if strategy in (STRATEGY_SUBSTRATE, STRATEGY_SUBSTRATE_WRAP):
        if substrate_cue:
            lines.append("")
            lines.append("SUBSTRATE_ANSWER:")
            lines.append(f"  ROOT_CAUSE: {substrate_cue['root_cause']}")
            lines.append(
                f"  SERVICES: {','.join(substrate_cue['services'])}")
            lines.append(f"  REMEDIATION: {substrate_cue['remediation']}")
            if strategy == STRATEGY_SUBSTRATE_WRAP:
                lines.append(
                    ("The SUBSTRATE_ANSWER above was computed "
                     "deterministically from typed handoffs between "
                     "roles. Return it verbatim — do not revise it."))
        lines.append("")
        lines.append("DELIVERED HANDOFFS:")
        for h in handoffs:
            lines.append(f"- [{h.source_role}/{h.claim_kind}] "
                         f"{h.payload}")
    else:
        lines.append("")
        lines.append("DELIVERED EVENTS:")
        for ev in delivered:
            lines.append(f"- [{ev.event_type} by {ev.origin_role}] "
                         f"{ev.body}")
        if truncated:
            lines.append(
                "... (event stream truncated; delivered subset is the "
                "first window)")
    lines.append("")
    lines.append("ANSWER:")
    prompt = "\n".join(lines)
    return prompt, delivered, truncated


# =============================================================================
# Answer grading
# =============================================================================


_ROOT_CAUSE_RE = re.compile(r"ROOT[_ ]CAUSE\s*[:\-]?\s*([^\n]+)",
                             re.IGNORECASE)
_SERVICES_RE = re.compile(r"SERVICES?\s*[:\-]?\s*([^\n]+)",
                           re.IGNORECASE)
_REMED_RE = re.compile(r"REMEDIATION\s*[:\-]?\s*([^\n]+)",
                        re.IGNORECASE)


def parse_answer(text: str) -> dict[str, object]:
    """Parse an auditor's freeform output into structured fields."""
    rc = ""
    svc: tuple[str, ...] = ()
    rem = ""
    m = _ROOT_CAUSE_RE.search(text)
    if m:
        rc = m.group(1).strip().lower()
        rc = re.sub(r"[^a-z0-9_]+", "_", rc).strip("_")
    m = _SERVICES_RE.search(text)
    if m:
        raw = m.group(1).strip()
        parts = [p.strip().lower() for p in re.split(r"[,\s]+", raw)
                 if p.strip()]
        svc = tuple(sorted(set(parts)))
    m = _REMED_RE.search(text)
    if m:
        rem = m.group(1).strip().lower()
        rem = re.sub(r"[^a-z0-9_]+", "_", rem).strip("_")
    return {"root_cause": rc, "services": svc, "remediation": rem}


def grade_answer(scenario: IncidentScenario,
                 answer_text: str) -> dict[str, object]:
    """Deterministic three-field grader.

    Returns a dict with fields:
      * ``root_cause_correct``   — bool.
      * ``services_correct``     — bool (set equality).
      * ``remediation_correct``  — bool.
      * ``full_correct``          — all three.
      * ``parsed``                — parsed answer.
    """
    parsed = parse_answer(answer_text)
    gold_rc = scenario.gold_root_cause
    gold_svc = set(scenario.gold_services)
    gold_rem = scenario.gold_remediation
    rc_ok = parsed["root_cause"] == gold_rc
    svc_ok = set(parsed["services"]) == gold_svc
    rem_ok = parsed["remediation"] == gold_rem
    return {
        "root_cause_correct": rc_ok,
        "services_correct": svc_ok,
        "remediation_correct": rem_ok,
        "full_correct": rc_ok and svc_ok and rem_ok,
        "parsed": parsed,
    }


# =============================================================================
# Failure attribution
# =============================================================================


FAILURE_MISSING_HANDOFF = "missing_handoff"
FAILURE_RETRIEVAL_MISS = "retrieval_miss"
FAILURE_LLM_ERROR = "llm_error"
FAILURE_TRUNCATION = "truncation"
FAILURE_NONE = "none"


def attribute_failure(
        scenario: IncidentScenario,
        grading: dict,
        strategy: str,
        handoffs: Sequence[TypedHandoff],
        truncated: bool,
        ) -> str:
    """Attribute a failing answer to a single category.

    Order of dispatch:
      1. full_correct     → FAILURE_NONE
      2. truncated=True   → FAILURE_TRUNCATION
      3. missing required claim in handoffs
                           → FAILURE_MISSING_HANDOFF
         (this is the typed-handoff layer's own failure mode)
      4. substrate-matched path but wrong parse → FAILURE_LLM_ERROR
      5. retrieval/decoder miss under naive/routing
                           → FAILURE_RETRIEVAL_MISS

    This matches the five-way decomposition the programme uses at
    the substrate layer (``ok / retrieval_miss / planning_error /
    render_error / llm_error``), re-scoped to the team-
    communication primitive.
    """
    if grading["full_correct"]:
        return FAILURE_NONE
    if truncated and strategy in (STRATEGY_NAIVE, STRATEGY_ROUTING):
        return FAILURE_TRUNCATION
    # Missing causal handoff — look at the scenario's chain and
    # check whether all required (role, kind) pairs are present.
    required = {(role, kind)
                for (role, kind, _p, _evs) in scenario.causal_chain}
    seen = {(h.source_role, h.claim_kind) for h in handoffs}
    missing = required - seen
    if strategy in (STRATEGY_SUBSTRATE, STRATEGY_SUBSTRATE_WRAP) and missing:
        return FAILURE_MISSING_HANDOFF
    if strategy in (STRATEGY_SUBSTRATE, STRATEGY_SUBSTRATE_WRAP):
        return FAILURE_LLM_ERROR
    return FAILURE_RETRIEVAL_MISS


# =============================================================================
# Harness driver
# =============================================================================


@dataclass
class IncidentMeasurement:
    """Per-(scenario, strategy) record."""

    scenario_id: str
    strategy: str
    # Prompt / delivery
    n_events_delivered: int
    n_handoffs_delivered: int
    n_prompt_chars: int
    n_prompt_tokens_approx: int
    truncated: bool
    # Oracle
    n_events_total: int
    n_events_causal: int
    n_events_causal_to_auditor: int
    n_causal_events_delivered: int
    aggregator_relevance_fraction: float
    n_required_claims: int
    n_required_claims_delivered: int
    handoff_recall: float
    # Answer
    llm_answer: str
    grading: dict
    failure_kind: str
    # Logs
    handoff_log_length: int
    handoff_chain_ok: bool
    delivery_account: dict
    wall_seconds: float

    def as_dict(self) -> dict:
        return {
            "scenario_id": self.scenario_id,
            "strategy": self.strategy,
            "n_events_delivered": self.n_events_delivered,
            "n_handoffs_delivered": self.n_handoffs_delivered,
            "n_prompt_chars": self.n_prompt_chars,
            "n_prompt_tokens_approx": self.n_prompt_tokens_approx,
            "truncated": self.truncated,
            "n_events_total": self.n_events_total,
            "n_events_causal": self.n_events_causal,
            "n_events_causal_to_auditor": self.n_events_causal_to_auditor,
            "n_causal_events_delivered":
                self.n_causal_events_delivered,
            "aggregator_relevance_fraction": round(
                self.aggregator_relevance_fraction, 4),
            "n_required_claims": self.n_required_claims,
            "n_required_claims_delivered":
                self.n_required_claims_delivered,
            "handoff_recall": round(self.handoff_recall, 4),
            "llm_answer": self.llm_answer[:500],
            "grading": {k: (bool(v) if isinstance(v, bool) else v)
                         for k, v in self.grading.items()
                         if k != "parsed"},
            "parsed": self.grading.get("parsed"),
            "failure_kind": self.failure_kind,
            "handoff_log_length": self.handoff_log_length,
            "handoff_chain_ok": self.handoff_chain_ok,
            "delivery_account": self.delivery_account,
            "wall_seconds": round(self.wall_seconds, 3),
        }


@dataclass
class IncidentReport:
    """Aggregate per-scenario-bank report."""

    scenario_ids: tuple[str, ...]
    strategies: tuple[str, ...]
    measurements: list[IncidentMeasurement]
    config: dict

    def as_dict(self) -> dict:
        return {
            "scenario_ids": list(self.scenario_ids),
            "strategies": list(self.strategies),
            "config": self.config,
            "measurements": [m.as_dict() for m in self.measurements],
            "pooled": self.pooled(),
        }

    def pooled(self) -> dict:
        out: dict[str, dict] = {}
        for strat in self.strategies:
            ms = [m for m in self.measurements if m.strategy == strat]
            if not ms:
                continue
            n = len(ms)
            correct = sum(1 for m in ms if m.grading["full_correct"])
            rc = sum(1 for m in ms if m.grading["root_cause_correct"])
            svc = sum(1 for m in ms if m.grading["services_correct"])
            rem = sum(1 for m in ms if m.grading["remediation_correct"])
            mean_tok = sum(m.n_prompt_tokens_approx for m in ms) / n
            trunc = sum(1 for m in ms if m.truncated)
            mean_rel = sum(m.aggregator_relevance_fraction for m in ms) / n
            mean_recall = sum(m.handoff_recall for m in ms) / n
            f_hist: dict[str, int] = {}
            for m in ms:
                f_hist[m.failure_kind] = f_hist.get(m.failure_kind, 0) + 1
            out[strat] = {
                "n": n,
                "accuracy_full": round(correct / n, 4),
                "accuracy_root_cause": round(rc / n, 4),
                "accuracy_services": round(svc / n, 4),
                "accuracy_remediation": round(rem / n, 4),
                "mean_prompt_tokens": round(mean_tok, 2),
                "truncated_count": trunc,
                "mean_aggregator_relevance_fraction": round(mean_rel, 4),
                "mean_handoff_recall": round(mean_recall, 4),
                "failure_hist": f_hist,
            }
        return out


# =============================================================================
# Main loop
# =============================================================================


def run_incident_loop(
        scenarios: Sequence[IncidentScenario],
        auditor: Callable[[str], str],
        strategies: Sequence[str] = ALL_STRATEGIES,
        seed: int = 31,
        max_events_in_prompt: int = 200,
        inbox_capacity: int = 32,
        extractor: Callable[
            [str, Sequence[IncidentEvent], IncidentScenario],
            list[tuple[str, str, tuple[int, ...]]],
        ] | None = None,
        ) -> IncidentReport:
    """Run one (scenario × strategy) grid.

    The ``auditor`` callable is the aggregator LLM (or a mock); same
    shape as ``swe_loop_harness.AnswerLLM``. ``extractor`` defaults to
    ``extract_claims_for_role``; the Phase-32 noisy-extractor sweep
    overrides it to inject controlled degradation.
    """
    import time as _time
    measurements: list[IncidentMeasurement] = []

    for scenario in scenarios:
        events = naive_event_stream(scenario)
        n_events_total = len(events)
        n_causal = sum(1 for ev in events if ev.is_causal)
        n_causal_to_auditor = sum(
            1 for ev in events
            if oracle_relevance(ev, ROLE_AUDITOR, scenario))
        required_claims = {(r, k)
                           for (r, k, _p, _evs) in scenario.causal_chain}

        # Run handoff protocol once per scenario (strategy-independent).
        handoff_router = run_handoff_protocol(
            scenario, max_events_per_role=max_events_in_prompt,
            inbox_capacity=inbox_capacity, extractor=extractor)

        auditor_inbox = handoff_router.inboxes.get(ROLE_AUDITOR)
        auditor_handoffs = tuple(auditor_inbox.peek()) \
            if auditor_inbox else ()
        # Substrate cue: decoded answer from the auditor's inbox.
        substrate_cue = _decoder_from_handoffs(auditor_handoffs)

        seen_pairs = {(h.source_role, h.claim_kind)
                      for h in auditor_handoffs}
        recall_required = (len(required_claims & seen_pairs)
                           / max(1, len(required_claims)))

        for strat in strategies:
            # What handoffs does this strategy deliver to the auditor?
            if strat in (STRATEGY_SUBSTRATE, STRATEGY_SUBSTRATE_WRAP):
                delivered_handoffs = auditor_handoffs
            else:
                delivered_handoffs = ()
            prompt, delivered, truncated = build_auditor_prompt(
                scenario, strat, events,
                handoffs=delivered_handoffs,
                substrate_cue=substrate_cue if strat in (
                    STRATEGY_SUBSTRATE, STRATEGY_SUBSTRATE_WRAP) else None,
                max_events_in_prompt=max_events_in_prompt,
            )
            delivered_causal_count = sum(
                1 for ev in delivered
                if oracle_relevance(ev, ROLE_AUDITOR, scenario))
            rel_frac = (delivered_causal_count / max(1, len(delivered))
                        if strat in (STRATEGY_NAIVE, STRATEGY_ROUTING)
                        else (sum(1 for h in delivered_handoffs
                                   if handoff_is_relevant(h, scenario))
                               / max(1, len(delivered_handoffs))))
            t0 = _time.time()
            answer = auditor(prompt)
            wall = _time.time() - t0
            grading = grade_answer(scenario, answer)
            failure = attribute_failure(
                scenario, grading, strat, delivered_handoffs, truncated)
            measurements.append(IncidentMeasurement(
                scenario_id=scenario.scenario_id, strategy=strat,
                n_events_delivered=len(delivered),
                n_handoffs_delivered=len(delivered_handoffs),
                n_prompt_chars=len(prompt),
                n_prompt_tokens_approx=max(1, len(prompt) // 4),
                truncated=truncated,
                n_events_total=n_events_total,
                n_events_causal=n_causal,
                n_events_causal_to_auditor=n_causal_to_auditor,
                n_causal_events_delivered=delivered_causal_count,
                aggregator_relevance_fraction=rel_frac,
                n_required_claims=len(required_claims),
                n_required_claims_delivered=len(
                    required_claims & seen_pairs),
                handoff_recall=recall_required,
                llm_answer=answer,
                grading=grading,
                failure_kind=failure,
                handoff_log_length=handoff_router.log_length(),
                handoff_chain_ok=handoff_router.verify(),
                delivery_account=handoff_router.account.summary(),
                wall_seconds=wall,
            ))

    return IncidentReport(
        scenario_ids=tuple(s.scenario_id for s in scenarios),
        strategies=tuple(strategies),
        measurements=measurements,
        config={
            "seed": seed,
            "max_events_in_prompt": max_events_in_prompt,
            "inbox_capacity": inbox_capacity,
        },
    )


# =============================================================================
# Mock aggregator — deterministic reader of the delivered prompt
# =============================================================================


class MockIncidentAuditor:
    """Deterministic auditor that saturates the *upper bound* of
    what a perfect prompt reader could infer.

    Behaviour:
      * If the prompt contains ``SUBSTRATE_ANSWER:`` block, copy it.
      * Else, re-run the event decoder on the ``DELIVERED EVENTS``
        lines.
      * Else, return "UNKNOWN".

    Separating the mock from the real LLM lets us attribute any
    shortfall under the real LLM to model-specific failure modes
    (Theorem P30-3 analogue for this task family), while the mock
    scores the delivery-strategy ceiling.
    """

    def __init__(self) -> None:
        self.last_prompt = ""
        self.last_answer = ""
        self.n_calls = 0
        self.total_prompt_chars = 0

    def __call__(self, prompt: str) -> str:
        self.n_calls += 1
        self.last_prompt = prompt
        self.total_prompt_chars += len(prompt)
        # Substrate shortcut — verbatim copy.
        m = re.search(
            r"SUBSTRATE_ANSWER:\s*\n\s*ROOT_CAUSE:\s*([^\n]+)\s*\n"
            r"\s*SERVICES:\s*([^\n]+)\s*\n\s*REMEDIATION:\s*([^\n]+)",
            prompt, re.IGNORECASE)
        if m:
            rc = m.group(1).strip()
            svc = m.group(2).strip()
            rem = m.group(3).strip()
            self.last_answer = (f"ROOT_CAUSE: {rc}\nSERVICES: {svc}\n"
                                f"REMEDIATION: {rem}\n")
            return self.last_answer
        # Try to decode from DELIVERED EVENTS.
        lines = prompt.splitlines()
        events: list[IncidentEvent] = []
        next_id = 0
        collecting = False
        for line in lines:
            line_s = line.strip()
            if line_s.startswith("DELIVERED EVENTS:"):
                collecting = True
                continue
            if line_s.startswith("ANSWER:"):
                break
            if not collecting:
                continue
            m2 = re.match(r"^-\s*\[([A-Z_]+)\s+by\s+(\w+)\]\s+(.+)$",
                          line_s)
            if not m2:
                continue
            et = m2.group(1).strip()
            role = m2.group(2).strip()
            body = m2.group(3).strip()
            # A reader emulation cannot know ``is_causal`` — it
            # runs the extractors *as if* every delivered event
            # could be causal. The extractor regexes filter
            # distractors by design.
            events.append(IncidentEvent(
                event_id=next_id, event_type=et,
                origin_role=role, body=body))
            next_id += 1
        dec = _decoder_from_events(events) if events else {
            "root_cause": "unknown", "services": (), "remediation": "investigate"}
        self.last_answer = (f"ROOT_CAUSE: {dec['root_cause']}\n"
                            f"SERVICES: {','.join(dec['services'])}\n"
                            f"REMEDIATION: {dec['remediation']}\n")
        return self.last_answer
