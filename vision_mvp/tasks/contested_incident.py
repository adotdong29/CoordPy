"""Phase 35 — contested-incident benchmark.

This task family is the smallest empirical instance in the programme
for which *static typed handoffs are not enough* to recover the
correct answer. It extends the Phase-31 incident-triage role cast
(monitor / db_admin / sysadmin / network / auditor) and claim
taxonomy, but every scenario is deliberately constructed to have
**two or more plausible root-cause claims** in the auditor's inbox
after the typed-handoff layer fires.

Under a *static-priority* decoder (Phase 31's
``_decoder_from_handoffs``) the author-defined ordering picks a
specific candidate regardless of which candidate is actually the
isolated root cause on this scenario. On three of the five
scenarios the static priority picks the wrong candidate; on the
fifth (a control) it picks correctly.

Under the **dynamic substrate** (``core/dynamic_comm``) the auditor
detects the contested-claim pattern, opens an
``EscalationThread`` over the two producer roles, each producer
posts a typed reply derived from its own per-role evidence
(``INDEPENDENT_ROOT`` if the producer can witness its claim is an
isolated cause, ``DOWNSTREAM_SYMPTOM`` if the producer can name an
upstream claim it is a symptom of, ``UNCERTAIN`` otherwise), and
the thread's single public resolution handoff tells the auditor
which candidate to pick. The decoder then uses the thread-resolved
claim rather than static priority.

This is the smallest test of the Phase-35 expressivity gap: every
scenario has a decoder-reachable right answer, but only under
*producer-local reflection* delivered through the thread — never
through any static subscription graph on the Phase-31 substrate.

Scope discipline (what this benchmark does NOT claim)
-----------------------------------------------------

1. **Not a claim that dynamic coordination beats static handoffs
   on Phase-31 incident-triage.** Phase 31 is uncontested by
   construction; the static priority always matches the gold. The
   point of this benchmark is precisely the *contested* region.
2. **Not a claim that unrestricted chat would help.** Broadcast
   does not give the auditor any more signal than naive broadcast
   already gives; the *missing bit* is producer-local reflection
   ("is my claim a root cause or a symptom?"), not raw telemetry
   volume.
3. **Not a scaling claim.** R* per scenario is 2 or 3; the
   benchmark is about *shape*, not size.
4. **Not a claim about real LLM-driven reflection.** Each
   producer's reply is computed by a deterministic role-local
   witness extractor — ``infer_causality_hypothesis`` — whose
   precision and recall on the Phase-35 scenario bank are 1.00
   by construction. The Phase-34 per-role noise wrapper can be
   composed on top; that's mechanical follow-up.
5. **Not a new typed-handoff substrate.** The Phase-31 substrate
   (``core/role_handoff``) and the Phase-31 subscription table are
   reused unchanged; only a new claim kind
   (``CLAIM_THREAD_RESOLUTION``) and a dynamic-comm router are
   added.

Theoretical anchor: RESULTS_PHASE35.md § B.
"""

from __future__ import annotations

import random
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

from vision_mvp.core.role_handoff import (
    HandoffLog, HandoffRouter, RoleInbox, RoleSubscriptionTable,
    TypedHandoff,
)
from vision_mvp.core.dynamic_comm import (
    CLAIM_THREAD_RESOLUTION, DynamicCommRouter,
    REPLY_DISAGREE, REPLY_DOWNSTREAM_SYMPTOM,
    REPLY_INDEPENDENT_ROOT,
    REPLY_UNCERTAIN, RESOLUTION_SINGLE_INDEPENDENT_ROOT,
    THREAD_ISSUE_ROOT_CAUSE_CONFLICT,
    build_resolution_subscriptions, parse_resolution_payload,
)
from vision_mvp.tasks.incident_triage import (
    ALL_CLAIMS, ALL_ROLES,
    CLAIM_CRON_OVERRUN, CLAIM_DEADLOCK_SUSPECTED,
    CLAIM_DISK_FILL_CRITICAL, CLAIM_DNS_MISROUTE,
    CLAIM_ERROR_RATE_SPIKE, CLAIM_FW_BLOCK_SURGE,
    CLAIM_LATENCY_SPIKE, CLAIM_OOM_KILL, CLAIM_POOL_EXHAUSTION,
    CLAIM_SLOW_QUERY_OBSERVED, CLAIM_TLS_EXPIRED,
    EVENT_DNS_QUERY, EVENT_FW_RULE_HIT, EVENT_LOG_LINE,
    EVENT_METRIC_SAMPLE, EVENT_NET_FLOW, EVENT_OS_EVENT,
    EVENT_SQL_STAT, FIXED_POINT_EVENT_TYPES, IncidentEvent,
    IncidentScenario,
    ROLE_AUDITOR, ROLE_DB_ADMIN, ROLE_MONITOR, ROLE_NETWORK,
    ROLE_SYSADMIN, build_role_subscriptions, extract_claims_for_role,
    fixed_point_events, naive_event_stream,
)


# =============================================================================
# Strategy names — exported for the driver
# =============================================================================


STRATEGY_NAIVE = "naive"
STRATEGY_STATIC_HANDOFF = "static_handoff"
STRATEGY_DYNAMIC = "dynamic"
STRATEGY_DYNAMIC_WRAP = "dynamic_wrap"
# Phase-36 Part C — bounded adaptive subscriptions as an
# alternative to dynamic threads. See core/adaptive_sub.py.
STRATEGY_ADAPTIVE_SUB = "adaptive_sub"

ALL_STRATEGIES = (STRATEGY_NAIVE, STRATEGY_STATIC_HANDOFF,
                  STRATEGY_DYNAMIC, STRATEGY_DYNAMIC_WRAP,
                  STRATEGY_ADAPTIVE_SUB)


# =============================================================================
# Scenario gold oracle — extended with 'contested' flag and causality map
# =============================================================================


@dataclass(frozen=True)
class ContestedScenario:
    """Phase-35 scenario: extends Phase-31 incident scenarios with
    a per-claim causality map and a 'contested' flag.

    Fields:
      * ``base``                   — wrapped Phase-31 ``IncidentScenario``.
      * ``contested``              — True iff the scenario contains
        ≥ 2 plausible root-cause claims with different priorities.
      * ``claim_causality``        — map from ``(producer_role,
        claim_kind)`` → one of ``"INDEPENDENT_ROOT"`` /
        ``"DOWNSTREAM_SYMPTOM_OF:<claim_kind>"`` / ``"UNCERTAIN"``.
        This is the per-role ground-truth used by the Phase-35
        causality extractor (``infer_causality_hypothesis``).
      * ``gold_root_cause_kind``    — the claim_kind whose decoder
        label is the true root cause. (The Phase-31 gold_root_cause
        string is a *label*; here we pin down which claim_kind a
        correct decoder should pick.)
      * ``gold_claim_idx_name``     — short tag naming the isolated
        causal claim (for attribution in reports).
    """

    base: IncidentScenario
    contested: bool
    claim_causality: dict[tuple[str, str], str]
    gold_root_cause_kind: str
    gold_claim_idx_name: str

    # Phase-31 delegation surface
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
# Scenario helpers (mirror of incident_triage._mk_event, _distractors)
# =============================================================================


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
    """Background chatter, mirror of the Phase-31 distractors.

    Kept independent so Phase-35 scenarios remain self-contained —
    no cross-module regex changes.
    """
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


# =============================================================================
# Scenario catalogue — contested root-cause family
# =============================================================================


def make_scenario_deadlock_vs_shadow_cron(
        rng: random.Random, start_id: int = 0,
        distractors_per_role: int = 6,
        ) -> ContestedScenario:
    """Real cause: application-level deadlock.
    Confounding: a coincident `cron backup.sh exit=137 duration=5400`
    on the archival host (NOT the app path). Under Phase-31 static
    priority, ``CLAIM_CRON_OVERRUN`` beats ``CLAIM_DEADLOCK_SUSPECTED``
    (because CRON_OVERRUN maps to ``disk_fill``) — wrong answer.

    Under dynamic coordination, SYSADMIN can witness that its cron
    overrun is on service=archival with no DB interaction, so it
    replies ``UNCERTAIN`` on the top-priority slot. DBA can witness
    the deadlock is ``cause=application_bug`` on the orders service
    and replies ``INDEPENDENT_ROOT``. Thread resolves deadlock.
    """
    nid = start_id
    per_role: dict[str, list[IncidentEvent]] = {r: [] for r in ALL_ROLES}
    chain: list[tuple[str, str, str, tuple[int, ...]]] = []

    # --- DBA causal: deadlock is the real root cause ---
    ev, nid = _mk_event(nid, EVENT_LOG_LINE, ROLE_DB_ADMIN,
                         "warning pg deadlock detected pid=8823 "
                         "relation=orders_payments cause=application_bug",
                         tags=("orders",), causal=True)
    per_role[ROLE_DB_ADMIN].append(ev)
    dl_ev = ev.event_id
    ev, nid = _mk_event(nid, EVENT_SQL_STAT, ROLE_DB_ADMIN,
                         "pool active=200/200 waiters=88 service=orders",
                         tags=("orders",), causal=True)
    per_role[ROLE_DB_ADMIN].append(ev)
    pool_ev = ev.event_id
    chain.append((ROLE_DB_ADMIN, CLAIM_DEADLOCK_SUSPECTED,
                  "deadlock relation=orders_payments "
                  "cause=application_bug", (dl_ev,)))
    chain.append((ROLE_DB_ADMIN, CLAIM_POOL_EXHAUSTION,
                  "pool active=200/200 waiters=88 service=orders",
                  (pool_ev,)))

    # --- SYSADMIN confounding: cron overrun, archival-only ---
    ev, nid = _mk_event(nid, EVENT_OS_EVENT, ROLE_SYSADMIN,
                         "cron backup.sh exit=137 duration_s=5400 "
                         "service=archival host=backup01",
                         tags=("archival",), causal=True)
    per_role[ROLE_SYSADMIN].append(ev)
    cron_ev = ev.event_id
    chain.append((ROLE_SYSADMIN, CLAIM_CRON_OVERRUN,
                  "backup.sh exit=137 duration_s=5400 service=archival",
                  (cron_ev,)))

    # --- MONITOR: error spike on orders ---
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

    base = IncidentScenario(
        scenario_id="contested_deadlock_vs_shadow_cron",
        description="db deadlock on orders_payments is the real "
                    "cause; a coincident backup.sh cron on the "
                    "archival host produces a CRON_OVERRUN that "
                    "outranks the deadlock under static priority",
        gold_root_cause="deadlock",
        gold_services=("orders",),
        gold_remediation="enforce_lock_ordering_in_orders",
        causal_chain=tuple(chain),
        per_role_events={r: tuple(evs) for r, evs in per_role.items()},
    )
    claim_causality = {
        (ROLE_DB_ADMIN, CLAIM_DEADLOCK_SUSPECTED): "INDEPENDENT_ROOT",
        (ROLE_DB_ADMIN, CLAIM_POOL_EXHAUSTION):
            f"DOWNSTREAM_SYMPTOM_OF:{CLAIM_DEADLOCK_SUSPECTED}",
        (ROLE_SYSADMIN, CLAIM_CRON_OVERRUN): "UNCERTAIN",
        (ROLE_MONITOR, CLAIM_ERROR_RATE_SPIKE):
            f"DOWNSTREAM_SYMPTOM_OF:{CLAIM_DEADLOCK_SUSPECTED}",
    }
    return ContestedScenario(
        base=base, contested=True,
        claim_causality=claim_causality,
        gold_root_cause_kind=CLAIM_DEADLOCK_SUSPECTED,
        gold_claim_idx_name="deadlock",
    )


def make_scenario_tls_vs_disk_shadow(
        rng: random.Random, start_id: int = 0,
        distractors_per_role: int = 6,
        ) -> ContestedScenario:
    """Real cause: TLS expired on api. Confounding: a partial
    disk-fill warning (``used=93%``) on the archival FS that fires
    the Phase-31 DISK_FILL extractor. Under Phase-31 static priority,
    ``DISK_FILL_CRITICAL`` outranks ``TLS_EXPIRED`` → decoder picks
    ``disk_fill`` (wrong).

    SYSADMIN's causality witness: its disk-fill is fs=/var/archive
    — not the FS the API uses. Replies ``UNCERTAIN``. NETWORK can
    witness ``reason=expired`` directly — replies ``INDEPENDENT_ROOT``.
    """
    nid = start_id
    per_role: dict[str, list[IncidentEvent]] = {r: [] for r in ALL_ROLES}
    chain: list[tuple[str, str, str, tuple[int, ...]]] = []

    # --- NETWORK causal: tls expired ---
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
                  "tls service=api reason=expired", (tls_ev,)))
    chain.append((ROLE_NETWORK, CLAIM_FW_BLOCK_SURGE,
                  "rule=deny src=hc-probe dst=api count=1200",
                  (fw_ev,)))

    # --- MONITOR: uptime drop on api ---
    ev, nid = _mk_event(nid, EVENT_METRIC_SAMPLE, ROLE_MONITOR,
                         "uptime_pct=41 service=api window=5m",
                         tags=("api",), causal=True)
    per_role[ROLE_MONITOR].append(ev)
    up_ev = ev.event_id
    chain.append((ROLE_MONITOR, CLAIM_ERROR_RATE_SPIKE,
                  "uptime_pct=41 service=api", (up_ev,)))

    # --- SYSADMIN confounding: disk fill on archival FS ---
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
        scenario_id="contested_tls_vs_disk_shadow",
        description="tls expired on api is the real cause; a "
                    "partial disk-fill warning on /var/archive "
                    "triggers DISK_FILL_CRITICAL which outranks "
                    "TLS_EXPIRED under static priority",
        gold_root_cause="tls_expiry",
        gold_services=("api",),
        gold_remediation="renew_tls_and_reload",
        causal_chain=tuple(chain),
        per_role_events={r: tuple(evs) for r, evs in per_role.items()},
    )
    claim_causality = {
        (ROLE_NETWORK, CLAIM_TLS_EXPIRED): "INDEPENDENT_ROOT",
        (ROLE_NETWORK, CLAIM_FW_BLOCK_SURGE):
            f"DOWNSTREAM_SYMPTOM_OF:{CLAIM_TLS_EXPIRED}",
        (ROLE_MONITOR, CLAIM_ERROR_RATE_SPIKE):
            f"DOWNSTREAM_SYMPTOM_OF:{CLAIM_TLS_EXPIRED}",
        (ROLE_SYSADMIN, CLAIM_DISK_FILL_CRITICAL): "UNCERTAIN",
    }
    return ContestedScenario(
        base=base, contested=True,
        claim_causality=claim_causality,
        gold_root_cause_kind=CLAIM_TLS_EXPIRED,
        gold_claim_idx_name="tls_expiry",
    )


def make_scenario_dns_vs_pool_symptom(
        rng: random.Random, start_id: int = 0,
        distractors_per_role: int = 6,
        ) -> ContestedScenario:
    """Real cause: DNS misroute on db.internal. Confounding: DBA
    sees POOL_EXHAUSTION (which is *downstream* of the DNS outage).
    Under Phase-31 static priority, DNS_MISROUTE is actually higher
    than POOL_EXHAUSTION so in the *default* priority list this
    scenario is in fact resolved correctly without a thread. The
    contested aspect here is added by pinning the *label*: the
    decoder below ranks ``POOL_EXHAUSTION`` with a *tied* priority
    to ``DNS_MISROUTE``, and the tiebreak goes by claim_kind string
    sort — which picks ``DNS_MISROUTE`` correctly.

    We keep this scenario in the bank as a **partial control**:
    static handoffs win here; the dynamic thread confirms the same
    answer; both strategies should produce the correct answer. It
    exercises the 'thread reaches the right answer even when the
    static answer also happens to be right' corner.
    """
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
                         "error psycopg2 could not connect "
                         "host=db.internal",
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
                  "reconnect_attempts=842 host=db.internal "
                  "service=orders",
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

    base = IncidentScenario(
        scenario_id="contested_dns_vs_pool_symptom",
        description="dns misroute on db.internal is the real "
                    "cause; pool exhaustion on orders is a "
                    "downstream symptom — static priority and "
                    "dynamic coordination both reach the right "
                    "answer (control scenario)",
        gold_root_cause="dns_misroute",
        gold_services=("orders",),
        gold_remediation="restore_internal_dns_zone",
        causal_chain=tuple(chain),
        per_role_events={r: tuple(evs) for r, evs in per_role.items()},
    )
    claim_causality = {
        (ROLE_NETWORK, CLAIM_DNS_MISROUTE): "INDEPENDENT_ROOT",
        (ROLE_DB_ADMIN, CLAIM_POOL_EXHAUSTION):
            f"DOWNSTREAM_SYMPTOM_OF:{CLAIM_DNS_MISROUTE}",
        (ROLE_MONITOR, CLAIM_ERROR_RATE_SPIKE):
            f"DOWNSTREAM_SYMPTOM_OF:{CLAIM_DNS_MISROUTE}",
    }
    return ContestedScenario(
        base=base, contested=False,
        claim_causality=claim_causality,
        gold_root_cause_kind=CLAIM_DNS_MISROUTE,
        gold_claim_idx_name="dns_misroute",
    )


def make_scenario_cron_vs_oom_shadow(
        rng: random.Random, start_id: int = 0,
        distractors_per_role: int = 6,
        ) -> ContestedScenario:
    """Real cause: SYSADMIN's cron overran, producing the actual
    disk-fill cascade on the app FS. Confounding: an unrelated
    `oom_kill comm=analytics.py service=batch` (a batch workload,
    not the app path). Under Phase-31 static priority,
    ``CLAIM_OOM_KILL`` outranks ``CLAIM_CRON_OVERRUN`` /
    ``CLAIM_DISK_FILL_CRITICAL`` is not present (only the upstream
    CRON_OVERRUN is on the causal chain here) so the decoder
    picks ``memory_leak`` — wrong.

    SYSADMIN can witness both events but the OOM is ``service=batch``
    (a separate workload). On dynamic coordination SYSADMIN replies
    ``UNCERTAIN`` on OOM and ``INDEPENDENT_ROOT`` on CRON_OVERRUN.
    """
    nid = start_id
    per_role: dict[str, list[IncidentEvent]] = {r: [] for r in ALL_ROLES}
    chain: list[tuple[str, str, str, tuple[int, ...]]] = []

    # --- SYSADMIN causal: real cron overrun on app-path FS ---
    ev, nid = _mk_event(nid, EVENT_OS_EVENT, ROLE_SYSADMIN,
                         "cron rotate_app_logs.sh exit=1 "
                         "duration_s=7200 service=app",
                         tags=("app",), causal=True)
    per_role[ROLE_SYSADMIN].append(ev)
    cron_ev = ev.event_id
    chain.append((ROLE_SYSADMIN, CLAIM_CRON_OVERRUN,
                  "rotate_app_logs.sh exit=1 duration_s=7200 "
                  "service=app",
                  (cron_ev,)))

    # --- SYSADMIN confounding: oom on a different service ---
    ev, nid = _mk_event(nid, EVENT_OS_EVENT, ROLE_SYSADMIN,
                         "oom_kill pid=9911 comm=analytics.py "
                         "rss=8.1G service=batch",
                         tags=("batch",), causal=True)
    per_role[ROLE_SYSADMIN].append(ev)
    oom_ev = ev.event_id
    chain.append((ROLE_SYSADMIN, CLAIM_OOM_KILL,
                  "oom_kill comm=analytics.py service=batch",
                  (oom_ev,)))

    # --- MONITOR: latency on app ---
    ev, nid = _mk_event(nid, EVENT_METRIC_SAMPLE, ROLE_MONITOR,
                         "p95_ms=5400 service=app window=5m",
                         tags=("app",), causal=True)
    per_role[ROLE_MONITOR].append(ev)
    m_ev = ev.event_id
    chain.append((ROLE_MONITOR, CLAIM_LATENCY_SPIKE,
                  "p95_ms=5400 service=app", (m_ev,)))

    for role, kinds in ((ROLE_MONITOR, [EVENT_METRIC_SAMPLE]),
                         (ROLE_DB_ADMIN, [EVENT_SQL_STAT,
                                           EVENT_LOG_LINE]),
                         (ROLE_SYSADMIN, [EVENT_OS_EVENT]),
                         (ROLE_NETWORK, [EVENT_NET_FLOW,
                                          EVENT_FW_RULE_HIT])):
        dist, nid = _distractors(rng, nid, role, kinds,
                                   k=distractors_per_role)
        per_role[role].extend(dist)

    base = IncidentScenario(
        scenario_id="contested_cron_vs_oom_shadow",
        description="a cron overrun on the app path is the real "
                    "cause; an unrelated oom_kill on an analytics "
                    "batch workload produces an OOM_KILL that "
                    "outranks CRON_OVERRUN under static priority",
        gold_root_cause="disk_fill",
        gold_services=("app",),
        gold_remediation="rotate_logs_and_clear_backup",
        causal_chain=tuple(chain),
        per_role_events={r: tuple(evs) for r, evs in per_role.items()},
    )
    claim_causality = {
        (ROLE_SYSADMIN, CLAIM_CRON_OVERRUN): "INDEPENDENT_ROOT",
        (ROLE_SYSADMIN, CLAIM_OOM_KILL): "UNCERTAIN",
        (ROLE_MONITOR, CLAIM_LATENCY_SPIKE):
            f"DOWNSTREAM_SYMPTOM_OF:{CLAIM_CRON_OVERRUN}",
    }
    return ContestedScenario(
        base=base, contested=True,
        claim_causality=claim_causality,
        gold_root_cause_kind=CLAIM_CRON_OVERRUN,
        gold_claim_idx_name="cron_overrun",
    )


def make_scenario_dns_vs_tls_shadow(
        rng: random.Random, start_id: int = 0,
        distractors_per_role: int = 6,
        ) -> ContestedScenario:
    """Real cause: DNS misroute on api.internal. Confounding: a
    stale TLS cert warning on an *unrelated* host (``tls service=mail
    reason=expired``) that triggers the NETWORK TLS_EXPIRED
    extractor. Static priority ranks TLS_EXPIRED above DNS_MISROUTE,
    so the decoder picks ``tls_expiry`` — wrong.

    NETWORK is the producer of both claims. On dynamic
    coordination NETWORK can witness which one is an isolated
    cause: DNS has a SERVFAIL on api.internal, TLS is on the mail
    service (unrelated to the api path). Replies
    ``INDEPENDENT_ROOT`` on DNS, ``UNCERTAIN`` on TLS.
    """
    nid = start_id
    per_role: dict[str, list[IncidentEvent]] = {r: [] for r in ALL_ROLES}
    chain: list[tuple[str, str, str, tuple[int, ...]]] = []

    # --- NETWORK causal: DNS misroute on the api-path name ---
    ev, nid = _mk_event(nid, EVENT_DNS_QUERY, ROLE_NETWORK,
                         "q=api.internal rc=SERVFAIL rtt_ms=2999",
                         tags=("api",), causal=True)
    per_role[ROLE_NETWORK].append(ev)
    dns_ev = ev.event_id
    chain.append((ROLE_NETWORK, CLAIM_DNS_MISROUTE,
                  "q=api.internal rc=SERVFAIL service=api",
                  (dns_ev,)))

    # --- NETWORK confounding: stale TLS on mail (unrelated) ---
    ev, nid = _mk_event(nid, EVENT_FW_RULE_HIT, ROLE_NETWORK,
                         "tls handshake fail service=mail "
                         "reason=expired",
                         tags=("mail",), causal=True)
    per_role[ROLE_NETWORK].append(ev)
    tls_ev = ev.event_id
    chain.append((ROLE_NETWORK, CLAIM_TLS_EXPIRED,
                  "tls service=mail reason=expired", (tls_ev,)))

    # --- MONITOR: error_rate on api ---
    ev, nid = _mk_event(nid, EVENT_METRIC_SAMPLE, ROLE_MONITOR,
                         "error_rate=0.65 service=api window=5m",
                         tags=("api",), causal=True)
    per_role[ROLE_MONITOR].append(ev)
    m_ev = ev.event_id
    chain.append((ROLE_MONITOR, CLAIM_ERROR_RATE_SPIKE,
                  "error_rate=0.65 service=api", (m_ev,)))

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
        scenario_id="contested_dns_vs_tls_shadow",
        description="dns misroute on api.internal is the real "
                    "cause; a stale tls cert on an unrelated mail "
                    "service triggers TLS_EXPIRED which outranks "
                    "DNS_MISROUTE under static priority",
        gold_root_cause="dns_misroute",
        gold_services=("api",),
        gold_remediation="restore_internal_dns_zone",
        causal_chain=tuple(chain),
        per_role_events={r: tuple(evs) for r, evs in per_role.items()},
    )
    claim_causality = {
        (ROLE_NETWORK, CLAIM_DNS_MISROUTE): "INDEPENDENT_ROOT",
        (ROLE_NETWORK, CLAIM_TLS_EXPIRED): "UNCERTAIN",
        (ROLE_MONITOR, CLAIM_ERROR_RATE_SPIKE):
            f"DOWNSTREAM_SYMPTOM_OF:{CLAIM_DNS_MISROUTE}",
    }
    return ContestedScenario(
        base=base, contested=True,
        claim_causality=claim_causality,
        gold_root_cause_kind=CLAIM_DNS_MISROUTE,
        gold_claim_idx_name="dns_misroute",
    )


def make_scenario_concordant_disk_fill(
        rng: random.Random, start_id: int = 0,
        distractors_per_role: int = 6,
        ) -> ContestedScenario:
    """Canonical disk-fill scenario — the Phase-31 control.
    Every claim aligns. Static priority is correct; dynamic
    coordination reaches the same answer. Included to keep the
    benchmark honest: a contested benchmark that cannot recognise
    a concordant scenario would be a spec bug.
    """
    nid = start_id
    per_role: dict[str, list[IncidentEvent]] = {r: [] for r in ALL_ROLES}
    chain: list[tuple[str, str, str, tuple[int, ...]]] = []

    ev, nid = _mk_event(nid, EVENT_OS_EVENT, ROLE_SYSADMIN,
                         "cron backup.sh exit=137 duration_s=5400 "
                         "service=api",
                         tags=("api",), causal=True)
    per_role[ROLE_SYSADMIN].append(ev)
    cron_ev = ev.event_id
    ev, nid = _mk_event(nid, EVENT_OS_EVENT, ROLE_SYSADMIN,
                         "/var/log used=99% fs=/ service=api",
                         tags=("api",), causal=True)
    per_role[ROLE_SYSADMIN].append(ev)
    disk_ev = ev.event_id
    chain.append((ROLE_SYSADMIN, CLAIM_CRON_OVERRUN,
                  "backup.sh exit=137 duration_s=5400 service=api",
                  (cron_ev,)))
    chain.append((ROLE_SYSADMIN, CLAIM_DISK_FILL_CRITICAL,
                  "/var/log used=99% fs=/ service=api",
                  (disk_ev,)))

    ev, nid = _mk_event(nid, EVENT_METRIC_SAMPLE, ROLE_MONITOR,
                         "error_rate=0.30 service=api window=5m",
                         tags=("api",), causal=True)
    per_role[ROLE_MONITOR].append(ev)
    m_ev = ev.event_id
    chain.append((ROLE_MONITOR, CLAIM_ERROR_RATE_SPIKE,
                  "error_rate=0.30 service=api", (m_ev,)))

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
        scenario_id="concordant_disk_fill",
        description="control scenario — disk fill is the real "
                    "cause and every claim aligns with it; "
                    "static priority and dynamic coordination "
                    "should produce the same right answer",
        gold_root_cause="disk_fill",
        gold_services=("api",),
        gold_remediation="rotate_logs_and_clear_backup",
        causal_chain=tuple(chain),
        per_role_events={r: tuple(evs) for r, evs in per_role.items()},
    )
    claim_causality = {
        (ROLE_SYSADMIN, CLAIM_DISK_FILL_CRITICAL): "INDEPENDENT_ROOT",
        (ROLE_SYSADMIN, CLAIM_CRON_OVERRUN):
            f"DOWNSTREAM_SYMPTOM_OF:{CLAIM_DISK_FILL_CRITICAL}",
        (ROLE_MONITOR, CLAIM_ERROR_RATE_SPIKE):
            f"DOWNSTREAM_SYMPTOM_OF:{CLAIM_DISK_FILL_CRITICAL}",
    }
    return ContestedScenario(
        base=base, contested=False,
        claim_causality=claim_causality,
        gold_root_cause_kind=CLAIM_DISK_FILL_CRITICAL,
        gold_claim_idx_name="disk_fill",
    )


SCENARIO_BUILDERS: tuple[
    Callable[..., ContestedScenario], ...] = (
    make_scenario_deadlock_vs_shadow_cron,
    make_scenario_tls_vs_disk_shadow,
    make_scenario_dns_vs_pool_symptom,
    make_scenario_cron_vs_oom_shadow,
    make_scenario_dns_vs_tls_shadow,
    make_scenario_concordant_disk_fill,
)


def build_contested_bank(seed: int = 35,
                          distractors_per_role: int = 6,
                          ) -> list[ContestedScenario]:
    rng = random.Random(seed)
    out: list[ContestedScenario] = []
    for builder in SCENARIO_BUILDERS:
        out.append(builder(rng, 0,
                            distractors_per_role=distractors_per_role))
    return out


# =============================================================================
# Priority table — reused and extended for Phase-35
# =============================================================================


# (claim_kind, decoder_label, remediation). Mirror of Phase-31
# priority list in ``incident_triage._decoder_from_handoffs`` but
# with ``CLAIM_THREAD_RESOLUTION`` taking *absolute precedence*
# when it carries a resolved winner.
_STATIC_PRIORITY: tuple[tuple[str, str, str], ...] = (
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


def claim_kind_to_label(kind: str) -> tuple[str, str]:
    for k, lbl, rem in _STATIC_PRIORITY:
        if k == kind:
            return lbl, rem
    return "unknown", "investigate"


# =============================================================================
# Role-local causality extractor — the Phase-35 per-role signal
# =============================================================================


def infer_causality_hypothesis(
        scenario: ContestedScenario,
        role: str,
        claim_kind: str,
        payload: str,
        ) -> str:
    """Deterministic per-role witness examiner.

    Given a role ``role`` asked about one of its claims
    ``(claim_kind, payload)``, returns one of

      * ``"INDEPENDENT_ROOT"`` — the role has evidence the claim
        is an isolated cause on its own data.
      * ``"DOWNSTREAM_SYMPTOM_OF:<upstream_kind>"`` — the role has
        evidence the claim is a symptom of an upstream claim.
      * ``"UNCERTAIN"`` — the role cannot tell from its own data.

    The decision is strictly role-local: it reads only the role's
    own events and the witness payload. The oracle ground truth
    lives in ``scenario.claim_causality``; here the extractor is
    **matched** to the oracle by construction (Phase-35 analogue
    of Phase-31's regex extractor being matched to the scenario
    catalogue by construction). A noisy-extractor follow-up can
    compose ``core/extractor_noise`` on top.
    """
    # The scenario's claim_causality is keyed on (producer_role,
    # claim_kind). When the role asked IS the producer, the answer
    # is exactly the oracle bit. When the role asked is NOT the
    # producer, we conservatively return UNCERTAIN — a non-producer
    # cannot witness another role's data.
    key = (role, claim_kind)
    if key in scenario.claim_causality:
        return scenario.claim_causality[key]
    return "UNCERTAIN"


# =============================================================================
# Phase-35 subscription table — Phase-31 table + CLAIM_THREAD_RESOLUTION
# =============================================================================


def build_phase35_subscriptions() -> RoleSubscriptionTable:
    """Phase-31 subscriptions + one extra row: the auditor
    subscribes to its own ``CLAIM_THREAD_RESOLUTION`` — so a
    thread opened by the auditor and closed by the auditor emits
    its resolution into the auditor's inbox via the standard
    typed-handoff delivery path.
    """
    subs = build_role_subscriptions()
    build_resolution_subscriptions(
        subs, opener_roles=[ROLE_AUDITOR],
        consumer_roles=[ROLE_AUDITOR])
    return subs


# =============================================================================
# Run the handoff protocol + optional dynamic coordination round
# =============================================================================


def _extract_contested_candidates(
        inbox_handoffs: Sequence[TypedHandoff]) -> list[int]:
    """Indices into the static priority list for every claim
    *currently in the auditor's inbox*. Used to decide whether a
    scenario is contested from the auditor's viewpoint.
    """
    seen = {h.claim_kind for h in inbox_handoffs}
    return [i for i, (k, _, _) in enumerate(_STATIC_PRIORITY)
            if k in seen]


def detect_contested_top(
        inbox_handoffs: Sequence[TypedHandoff],
        top_k: int = 2) -> list[TypedHandoff]:
    """Return the top-``top_k`` candidate handoffs for the
    root-cause slot, in static-priority order.

    Used by the dynamic strategy to decide whether to open a
    thread. If the auditor's inbox has < 2 claims eligible for
    the root-cause slot, there is no contest and no thread is
    opened.
    """
    priority_idx = {k: i for i, (k, _, _) in enumerate(_STATIC_PRIORITY)}
    # The root-cause-bearing claim kinds — anything mapped to a
    # non-trivial label in the priority list.
    root_bearing_kinds = {
        CLAIM_DISK_FILL_CRITICAL, CLAIM_TLS_EXPIRED,
        CLAIM_DNS_MISROUTE, CLAIM_OOM_KILL,
        CLAIM_DEADLOCK_SUSPECTED, CLAIM_CRON_OVERRUN,
    }
    candidates = [h for h in inbox_handoffs
                  if h.claim_kind in root_bearing_kinds]
    # Dedup by (source_role, claim_kind) — same claim from two
    # agents should not inflate the contested count.
    seen: set[tuple[str, str]] = set()
    deduped: list[TypedHandoff] = []
    for h in candidates:
        key = (h.source_role, h.claim_kind)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(h)
    deduped.sort(key=lambda h: priority_idx.get(h.claim_kind, 99))
    return deduped[:top_k]


def run_dynamic_coordination(
        scenario: ContestedScenario,
        router: DynamicCommRouter,
        auditor_handoffs: Sequence[TypedHandoff],
        max_rounds: int = 2,
        witness_token_cap: int = 12,
        causality_extractor: Callable[
            [object, str, str, str], str] | None = None,
        ) -> tuple[object, dict]:
    """Run a single dynamic-coordination round if the auditor's
    inbox contains contested candidates.

    ``causality_extractor`` is a pluggable replacement for the
    deterministic ``infer_causality_hypothesis``. Its signature is
    ``(scenario, role, kind, payload) -> str`` returning one of
    ``"INDEPENDENT_ROOT"`` / ``"DOWNSTREAM_SYMPTOM_OF:<kind>"`` /
    ``"UNCERTAIN"``. Used by Phase-36 Part A to inject reply-axis
    noise and by Phase-36 Part B to drive replies via a real LLM
    (see ``core/reply_noise`` / ``core/llm_thread_replier``).

    Returns ``(thread_or_None, debug_dict)``. When a thread is
    opened, the members post causality-hypothesis replies and the
    thread is closed before return; the resolution's handoff is
    already in the auditor's inbox on return.

    The debug dict carries:

      * ``contested_candidates`` — list of ``(producer, kind,
        payload)``.
      * ``thread_id``            — str or None.
      * ``thread_replies``       — list[(role, reply_kind, idx)].
      * ``resolution_kind``      — str or None.
      * ``resolved_claim_idx``   — int or None.
      * ``resolution_winner``    — ``(producer, kind)`` or None.
    """
    debug: dict = {
        "contested_candidates": [],
        "thread_id": None,
        "thread_replies": [],
        "resolution_kind": None,
        "resolved_claim_idx": None,
        "resolution_winner": None,
    }
    extractor = causality_extractor or infer_causality_hypothesis
    top = detect_contested_top(auditor_handoffs)
    if len(top) < 2:
        return None, debug
    candidates = [(h.source_role, h.claim_kind, h.payload) for h in top]
    debug["contested_candidates"] = list(candidates)
    producer_roles = frozenset(h.source_role for h in top)
    members = producer_roles | {ROLE_AUDITOR}
    # Each producer may own more than one of the contested
    # candidates (e.g. sysadmin owns both OOM_KILL and CRON_OVERRUN
    # in ``contested_cron_vs_oom_shadow``). Cap replies per member
    # to exactly the number of candidates — this is still bounded
    # by task-spec constants (Theorem P35-2).
    thread = router.open_thread(
        opener_role=ROLE_AUDITOR,
        issue_kind=THREAD_ISSUE_ROOT_CAUSE_CONFLICT,
        members=members,
        candidate_claims=candidates,
        max_rounds=max_rounds,
        max_replies_per_member=len(candidates),
        quorum=1,
        witness_token_cap=witness_token_cap,
        round=2,
    )
    debug["thread_id"] = thread.thread_id
    # Each producer posts one typed reply by examining its own
    # evidence on the candidate claims that *it* produced.
    for idx, (prod_role, kind, payload) in enumerate(candidates):
        hypothesis = extractor(scenario, prod_role, kind, payload)
        if hypothesis == "INDEPENDENT_ROOT":
            reply_kind = REPLY_INDEPENDENT_ROOT
        elif hypothesis.startswith("DOWNSTREAM_SYMPTOM_OF:"):
            reply_kind = REPLY_DOWNSTREAM_SYMPTOM
        else:
            reply_kind = REPLY_UNCERTAIN
        # Witness string: first few tokens of the payload, bounded.
        witness_tokens = payload.split()[:witness_token_cap]
        witness = " ".join(witness_tokens)
        outcome = router.post_reply(
            thread_id=thread.thread_id,
            replier_role=prod_role,
            reply_kind=reply_kind,
            referenced_claim_idx=idx,
            witness=witness,
            round=2,
        )
        debug["thread_replies"].append(
            (prod_role, reply_kind, idx, outcome))
    resolution = router.close_thread(thread.thread_id, round=3)
    debug["resolution_kind"] = resolution.resolution_kind
    debug["resolved_claim_idx"] = resolution.resolved_claim_idx
    if resolution.resolved_claim_idx is not None:
        cc = thread.candidate_claims[resolution.resolved_claim_idx]
        debug["resolution_winner"] = (cc.producer_role, cc.claim_kind)
    return thread, debug


# =============================================================================
# Phase-36 Part C — adaptive-subscription coordination
# =============================================================================


def run_adaptive_sub_coordination(
        scenario: ContestedScenario,
        max_active_edges: int = 4,
        witness_token_cap: int = 12,
        causality_extractor: Callable[
            [object, str, str, str], str] | None = None,
        max_events_per_role: int = 200,
        inbox_capacity: int = 32,
        claim_extractor: Callable[
            [str, Sequence[IncidentEvent], IncidentScenario],
            list[tuple[str, str, tuple[int, ...]]],
        ] | None = None,
        ) -> tuple[object, tuple[TypedHandoff, ...], dict]:
    """Run the Phase-36 Part C bounded-adaptive-subscription path.

    Builds a fresh AdaptiveSubRouter, runs the same Phase-31 static
    handoffs as Phase-35, then — on contested inboxes — the auditor
    installs one temporary edge per producer role from
    ``CLAIM_CAUSALITY_HYPOTHESIS`` to ``ROLE_AUDITOR`` with
    ``ttl_rounds=1``. Each producer emits exactly one causality-
    hypothesis handoff on its own candidate claim. The edges are
    then ticked to expire, and the auditor's inbox is returned
    alongside the standard handoffs for the Phase-35 decoder to
    consume.

    This is the alternative to opening an escalation thread. The
    auditor never calls ``open_thread`` / ``post_reply`` /
    ``close_thread``. Bounded-context is enforced by
    ``max_active_edges`` and ``ttl_rounds`` at runtime, not by
    the thread primitive's type-level frozen member set.

    Returns ``(router, handoffs_for_decoder, debug)`` where
    ``debug`` is a dict with keys:
      * ``contested_candidates`` — list of (role, kind, payload).
      * ``n_edges_installed`` / ``n_edges_expired``.
      * ``n_hypotheses_delivered`` — number of
        ``CLAIM_CAUSALITY_HYPOTHESIS`` handoffs that reached the
        auditor.
      * ``hypotheses_by_role`` — dict[role] → parsed hypothesis.
      * ``resolved_claim_idx`` / ``resolution_kind`` /
        ``resolution_winner`` — the adaptive-sub analogue of a
        thread resolution, computed from the collected
        hypotheses under the same counting rule.
    """
    # Local imports to avoid circular dependency (adaptive_sub
    # depends on role_handoff; contested_incident depends on
    # dynamic_comm; we don't want a hard dep cycle).
    from vision_mvp.core.adaptive_sub import (
        AdaptiveSubRouter, CLAIM_CAUSALITY_HYPOTHESIS,
        format_hypothesis_payload, parse_hypothesis_payload,
    )
    from vision_mvp.core.dynamic_comm import (
        RESOLUTION_SINGLE_INDEPENDENT_ROOT as _RES_SINGLE,
        RESOLUTION_CONFLICT as _RES_CONFLICT,
        RESOLUTION_NO_CONSENSUS as _RES_NO,
    )

    extractor = causality_extractor or infer_causality_hypothesis
    claim_ext = claim_extractor or extract_claims_for_role

    subs = build_phase35_subscriptions()
    base = HandoffRouter(subs=subs)
    for role in ALL_ROLES:
        base.register_inbox(RoleInbox(role=role,
                                       capacity=inbox_capacity))
    router = AdaptiveSubRouter(
        base_router=base, max_active_edges=max_active_edges)

    # Emit the standard Phase-31 static handoffs.
    for role in (ROLE_MONITOR, ROLE_DB_ADMIN, ROLE_SYSADMIN,
                 ROLE_NETWORK):
        evs = list(scenario.per_role_events.get(role, ()))
        if max_events_per_role:
            evs = evs[:max_events_per_role]
        claims = claim_ext(role, evs, scenario.base)
        for (kind, payload, evids) in claims:
            router.emit(
                source_role=role,
                source_agent_id=ALL_ROLES.index(role),
                claim_kind=kind, payload=payload,
                source_event_ids=evids, round=1)

    auditor_inbox = router.inboxes.get(ROLE_AUDITOR)
    pre_handoffs = tuple(auditor_inbox.peek()) if auditor_inbox else ()
    top = detect_contested_top(pre_handoffs)

    debug: dict = {
        "contested_candidates": [],
        "n_edges_installed": 0,
        "n_edges_expired": 0,
        "n_hypotheses_delivered": 0,
        "hypotheses_by_role": {},
        "resolved_claim_idx": None,
        "resolution_kind": None,
        "resolution_winner": None,
    }

    if len(top) < 2:
        handoffs_for_decoder = pre_handoffs
        return router, handoffs_for_decoder, debug

    candidates = [(h.source_role, h.claim_kind, h.payload) for h in top]
    debug["contested_candidates"] = list(candidates)

    # Install one temporary edge per distinct producer role.
    installed_edges: list = []
    producer_roles = sorted({h.source_role for h in top})
    for prod in producer_roles:
        edge = router.install_edge(
            source_role=prod,
            claim_kind=CLAIM_CAUSALITY_HYPOTHESIS,
            consumer_roles=[ROLE_AUDITOR],
            ttl_rounds=1,
        )
        installed_edges.append(edge)
    debug["n_edges_installed"] = len(installed_edges)

    # Producers emit exactly one hypothesis claim per their own
    # candidate — symmetric with the Phase-35 thread reply.
    for idx, (prod_role, kind, payload) in enumerate(candidates):
        hypothesis = extractor(scenario, prod_role, kind, payload)
        if hypothesis == "INDEPENDENT_ROOT":
            reply_kind = REPLY_INDEPENDENT_ROOT
        elif hypothesis.startswith("DOWNSTREAM_SYMPTOM_OF:"):
            reply_kind = REPLY_DOWNSTREAM_SYMPTOM
        else:
            reply_kind = REPLY_UNCERTAIN
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
        debug["n_hypotheses_delivered"] += 1
        debug["hypotheses_by_role"][prod_role] = reply_kind

    # Tick — removes the installed edges.
    expired = router.tick(1)
    debug["n_edges_expired"] = len(expired)

    # Re-snapshot auditor inbox (now contains static + hypothesis
    # handoffs).
    auditor_inbox = router.inboxes.get(ROLE_AUDITOR)
    post_handoffs = tuple(auditor_inbox.peek()) if auditor_inbox else ()

    # Apply the same resolution rule the thread uses, but read
    # from CAUSALITY_HYPOTHESIS payloads:
    ir_by_idx: dict[int, list[str]] = {}
    dis_by_idx: dict[int, list[str]] = {}
    for h in post_handoffs:
        if h.claim_kind != CLAIM_CAUSALITY_HYPOTHESIS:
            continue
        parsed = parse_hypothesis_payload(h.payload)
        idx_str = parsed.get("idx", "-1")
        try:
            idx = int(idx_str)
        except ValueError:
            idx = -1
        if idx < 0:
            continue
        if parsed.get("kind") == REPLY_INDEPENDENT_ROOT:
            ir_by_idx.setdefault(idx, []).append(h.source_role)
        elif parsed.get("kind") == REPLY_DISAGREE:
            dis_by_idx.setdefault(idx, []).append(h.source_role)
    resolved_idx: int | None = None
    res_kind: str
    if len(ir_by_idx) == 1:
        only_idx = next(iter(ir_by_idx.keys()))
        if dis_by_idx.get(only_idx):
            res_kind = _RES_CONFLICT
        else:
            res_kind = _RES_SINGLE
            resolved_idx = only_idx
    elif len(ir_by_idx) >= 2:
        res_kind = _RES_CONFLICT
    else:
        res_kind = _RES_NO
    debug["resolved_claim_idx"] = resolved_idx
    debug["resolution_kind"] = res_kind
    if resolved_idx is not None:
        prod, kind, _payload = candidates[resolved_idx]
        debug["resolution_winner"] = (prod, kind)
    return router, post_handoffs, debug


# =============================================================================
# Decoder — static priority with optional thread-resolution override
# =============================================================================


def decoder_from_handoffs_phase35(
        handoffs: Sequence[TypedHandoff],
        ) -> dict:
    """Phase-35/36 decoder. Priority order is:

      1. If a ``CLAIM_THREAD_RESOLUTION`` handoff is present AND
         its ``resolution_kind`` is ``SINGLE_INDEPENDENT_ROOT`` or
         ``QUORUM_AGREE``, and the winner kind is recognised,
         pick it. Services are aggregated from handoffs *excluding*
         the thread's ``losers`` — so a shadow claim that named
         an out-of-band service (``service=mail`` on an unrelated
         TLS expiry while the real issue is DNS) does not
         contaminate the services list.
      2. Phase-36 extension: if one or more
         ``CLAIM_CAUSALITY_HYPOTHESIS`` handoffs are present (from
         the adaptive-subscription strategy) AND exactly one of
         them is ``reply_kind=INDEPENDENT_ROOT``, pick that
         producer's claim kind. Services are aggregated from
         handoffs *excluding* the producers whose hypothesis was
         ``UNCERTAIN`` / ``DOWNSTREAM_SYMPTOM`` — mirror of the
         thread loser-list filter.
      3. Otherwise, apply the static priority list to the claim
         kinds in the bundle (Phase-31 behaviour) with no loser
         filter.
    """
    # Parse any thread resolution first; collect the loser
    # (producer_role, claim_kind) tuples to filter from services.
    thread_winner_kind: str | None = None
    thread_lbl: str | None = None
    thread_rem: str | None = None
    loser_pairs: set[tuple[str, str]] = set()
    for h in handoffs:
        if h.claim_kind != CLAIM_THREAD_RESOLUTION:
            continue
        parsed = parse_resolution_payload(h.payload)
        if parsed.get("kind") not in (
                RESOLUTION_SINGLE_INDEPENDENT_ROOT,
                "QUORUM_AGREE"):
            continue
        winner = parsed.get("winner", "")
        if "/" in winner:
            _, kind = winner.split("/", 1)
            lbl, rem = claim_kind_to_label(kind)
            if lbl != "unknown":
                thread_winner_kind = kind
                thread_lbl = lbl
                thread_rem = rem
        losers_str = parsed.get("losers", "none")
        if losers_str and losers_str != "none":
            for part in losers_str.split(","):
                if "/" in part:
                    role, kind = part.split("/", 1)
                    loser_pairs.add((role, kind))

    # Phase-36 Part C — if no thread resolution fired, check for
    # CAUSALITY_HYPOTHESIS handoffs and apply the same rule.
    hypothesis_winner_kind: str | None = None
    hypothesis_lbl: str | None = None
    hypothesis_rem: str | None = None
    if thread_winner_kind is None:
        # Lazy import to avoid cyclic dependency at module load.
        try:
            from vision_mvp.core.adaptive_sub import (
                CLAIM_CAUSALITY_HYPOTHESIS as _CLAIM_HYP,
                parse_hypothesis_payload as _parse_hyp,
            )
        except ImportError:
            _CLAIM_HYP = None  # type: ignore[assignment]
        if _CLAIM_HYP is not None:
            ir_hypotheses: list[tuple[str, str]] = []
            non_ir_hypotheses: list[tuple[str, str]] = []
            for h in handoffs:
                if h.claim_kind != _CLAIM_HYP:
                    continue
                parsed = _parse_hyp(h.payload)
                try:
                    idx = int(parsed.get("idx", "-1"))
                except ValueError:
                    idx = -1
                if idx < 0:
                    continue
                upstream_kind = parsed.get("upstream_kind", "")
                if not upstream_kind:
                    continue
                row = (h.source_role, upstream_kind)
                if parsed.get("kind") == REPLY_INDEPENDENT_ROOT:
                    ir_hypotheses.append(row)
                else:
                    non_ir_hypotheses.append(row)
            if len(ir_hypotheses) == 1:
                winner_role, upstream_kind = ir_hypotheses[0]
                lbl, rem = claim_kind_to_label(upstream_kind)
                if lbl != "unknown":
                    hypothesis_winner_kind = upstream_kind
                    hypothesis_lbl = lbl
                    hypothesis_rem = rem
                    # Add every non-winning producer's claim to
                    # the loser filter so shadow-service tokens
                    # drop out of the services aggregation.
                    for (role, ukind) in non_ir_hypotheses:
                        loser_pairs.add((role, ukind))
                    # Also filter *any other root-bearing claim*
                    # produced by the winner role that is NOT the
                    # adjudicated upstream — the shadow-same-role
                    # case (e.g. NETWORK holds both TLS_EXPIRED
                    # and DNS_MISROUTE and the hypothesis picked
                    # DNS_MISROUTE).
                    for h in handoffs:
                        if h.source_role != winner_role:
                            continue
                        if h.claim_kind in (_CLAIM_HYP,
                                             CLAIM_THREAD_RESOLUTION):
                            continue
                        if h.claim_kind == upstream_kind:
                            continue
                        lbl2, _r2 = claim_kind_to_label(h.claim_kind)
                        if lbl2 == "unknown":
                            continue
                        loser_pairs.add((winner_role, h.claim_kind))
    services: set[str] = set()
    for h in handoffs:
        if (h.source_role, h.claim_kind) in loser_pairs:
            continue
        # Skip hypothesis and thread-resolution handoffs —
        # services live on the payloads of the upstream claims,
        # not the coordination messages.
        try:
            from vision_mvp.core.adaptive_sub import (
                CLAIM_CAUSALITY_HYPOTHESIS as _HYP,
            )
        except ImportError:
            _HYP = None  # type: ignore[assignment]
        if h.claim_kind == CLAIM_THREAD_RESOLUTION:
            continue
        if _HYP is not None and h.claim_kind == _HYP:
            continue
        for tok in h.payload.split():
            m = re.search(r"service=(\w+)", tok)
            if not m:
                continue
            svc = m.group(1)
            services.add(svc)
    services_tuple = tuple(sorted(services))
    if thread_winner_kind is not None:
        return {
            "root_cause": thread_lbl,
            "services": services_tuple,
            "remediation": thread_rem,
            "decoder_mode": "thread_resolution",
            "selected_claim_kind": thread_winner_kind,
        }
    if hypothesis_winner_kind is not None:
        return {
            "root_cause": hypothesis_lbl,
            "services": services_tuple,
            "remediation": hypothesis_rem,
            "decoder_mode": "adaptive_sub_hypothesis",
            "selected_claim_kind": hypothesis_winner_kind,
        }
    # Static priority fallback — no loser filter because no thread
    # ran. This matches Phase-31's services-aggregation behaviour
    # byte-for-byte.
    claim_kinds = {h.claim_kind for h in handoffs}
    for (kind, label, remediation) in _STATIC_PRIORITY:
        if kind in claim_kinds:
            return {
                "root_cause": label,
                "services": services_tuple,
                "remediation": remediation,
                "decoder_mode": "static_priority",
                "selected_claim_kind": kind,
            }
    return {
        "root_cause": "unknown",
        "services": services_tuple,
        "remediation": "investigate",
        "decoder_mode": "static_priority",
        "selected_claim_kind": None,
    }


# =============================================================================
# Harness per strategy
# =============================================================================


def run_contested_handoff_protocol(
        scenario: ContestedScenario,
        max_events_per_role: int = 200,
        inbox_capacity: int = 32,
        claim_extractor: Callable[
            [str, Sequence[IncidentEvent], IncidentScenario],
            list[tuple[str, str, tuple[int, ...]]],
        ] | None = None,
        ) -> DynamicCommRouter:
    """Run the static handoff layer AND register CLAIM_THREAD_RESOLUTION
    routing so a subsequent dynamic round emits cleanly.

    Returns a ``DynamicCommRouter`` whose inboxes and handoff log
    already contain the Phase-31 typed-handoff delivery for
    ``scenario``. No thread has been opened yet.

    ``claim_extractor`` is an optional replacement for the Phase-31
    ``extract_claims_for_role`` — used by Phase-38 Part A to inject
    an adversarial / ensemble extractor at the *claim-extraction*
    boundary (strictly upstream of the reply-axis layer). Signature:
    ``(role, events, scenario.base) -> list[(claim_kind, payload,
    evids)]``. Defaults to ``extract_claims_for_role``.
    """
    extractor = claim_extractor or extract_claims_for_role
    subs = build_phase35_subscriptions()
    base = HandoffRouter(subs=subs)
    for role in ALL_ROLES:
        base.register_inbox(RoleInbox(role=role,
                                       capacity=inbox_capacity))
    router = DynamicCommRouter(base_router=base)
    for role in (ROLE_MONITOR, ROLE_DB_ADMIN, ROLE_SYSADMIN, ROLE_NETWORK):
        evs = list(scenario.per_role_events.get(role, ()))
        if max_events_per_role:
            evs = evs[:max_events_per_role]
        claims = extractor(role, evs, scenario.base)
        for (kind, payload, evids) in claims:
            router.emit(
                source_role=role,
                source_agent_id=ALL_ROLES.index(role),
                claim_kind=kind, payload=payload,
                source_event_ids=evids, round=1)
    return router


def build_auditor_prompt_p35(
        scenario: ContestedScenario,
        strategy: str,
        events: Sequence[IncidentEvent],
        handoffs: Sequence[TypedHandoff] = (),
        substrate_cue: dict | None = None,
        max_events_in_prompt: int = 200,
        ) -> tuple[str, list[IncidentEvent], bool]:
    """Phase-35 prompt assembly.

    Under ``dynamic`` / ``dynamic_wrap``, the auditor's bundle
    includes the ``CLAIM_THREAD_RESOLUTION`` handoff alongside the
    static claims; the prompt template identifies it so a wrap-path
    model can copy the resolution verbatim.
    """
    if strategy == STRATEGY_NAIVE:
        delivered = list(events)
    else:
        delivered = [ev for ev in events if ev.is_fixed_point]
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
    if strategy in (STRATEGY_STATIC_HANDOFF, STRATEGY_DYNAMIC,
                    STRATEGY_DYNAMIC_WRAP, STRATEGY_ADAPTIVE_SUB):
        if substrate_cue:
            lines.append("")
            lines.append("SUBSTRATE_ANSWER:")
            lines.append(f"  ROOT_CAUSE: {substrate_cue['root_cause']}")
            lines.append(
                f"  SERVICES: {','.join(substrate_cue['services'])}")
            lines.append(f"  REMEDIATION: {substrate_cue['remediation']}")
            if strategy == STRATEGY_DYNAMIC_WRAP:
                lines.append(
                    ("The SUBSTRATE_ANSWER above incorporates any "
                     "thread-resolution summary and was computed "
                     "deterministically from typed handoffs + the "
                     "dynamic-coordination resolution. Return it "
                     "verbatim — do not revise it."))
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
# Answer grading (mirror of incident_triage.grade_answer)
# =============================================================================


_ROOT_CAUSE_RE = re.compile(r"ROOT[_ ]CAUSE\s*[:\-]?\s*([^\n]+)",
                             re.IGNORECASE)
_SERVICES_RE = re.compile(r"SERVICES?\s*[:\-]?\s*([^\n]+)",
                           re.IGNORECASE)
_REMED_RE = re.compile(r"REMEDIATION\s*[:\-]?\s*([^\n]+)",
                        re.IGNORECASE)


def parse_answer_p35(text: str) -> dict:
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


def grade_answer_p35(scenario: ContestedScenario,
                     answer_text: str) -> dict:
    parsed = parse_answer_p35(answer_text)
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
# Mock auditor — deterministic reader (mirror of MockIncidentAuditor)
# =============================================================================


class MockContestedAuditor:
    """Deterministic auditor that mirrors ``MockIncidentAuditor``:
    copies the SUBSTRATE_ANSWER when present, else decodes from
    DELIVERED EVENTS using the Phase-31 event-decoder fallback.
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
        # Naive fallback: run extractors on the delivered events
        # block. Mirror of incident_triage._decoder_from_events.
        lines = prompt.splitlines()
        collecting = False
        events: list[IncidentEvent] = []
        next_id = 0
        for line in lines:
            s = line.strip()
            if s.startswith("DELIVERED EVENTS:"):
                collecting = True
                continue
            if s.startswith("ANSWER:"):
                break
            if not collecting:
                continue
            m2 = re.match(r"^-\s*\[([A-Z_]+)\s+by\s+(\w+)\]\s+(.+)$", s)
            if not m2:
                continue
            et, role, body = m2.group(1), m2.group(2), m2.group(3)
            events.append(IncidentEvent(
                event_id=next_id, event_type=et,
                origin_role=role, body=body))
            next_id += 1
        # Run per-role extractors + static-priority decoder.
        by_role: dict[str, list[IncidentEvent]] = {
            r: [] for r in ALL_ROLES}
        for ev in events:
            if ev.origin_role in by_role:
                by_role[ev.origin_role].append(ev)
        synth_handoffs: list[TypedHandoff] = []
        next_id = 0
        for role in (ROLE_MONITOR, ROLE_DB_ADMIN, ROLE_SYSADMIN,
                     ROLE_NETWORK):
            claims = extract_claims_for_role(
                role, by_role[role], None)  # type: ignore[arg-type]
            for (kind, payload, evids) in claims:
                synth_handoffs.append(TypedHandoff(
                    handoff_id=next_id, source_role=role,
                    source_agent_id=ALL_ROLES.index(role),
                    to_role=ROLE_AUDITOR, claim_kind=kind,
                    payload=payload,
                    source_event_ids=tuple(evids), round=1,
                    payload_cid="", prev_chain_hash="",
                    chain_hash=""))
                next_id += 1
        dec = decoder_from_handoffs_phase35(synth_handoffs)
        self.last_answer = (f"ROOT_CAUSE: {dec['root_cause']}\n"
                            f"SERVICES: {','.join(dec['services'])}\n"
                            f"REMEDIATION: {dec['remediation']}\n")
        return self.last_answer


# =============================================================================
# Per-strategy measurement record + harness driver
# =============================================================================


@dataclass
class ContestedMeasurement:
    scenario_id: str
    strategy: str
    contested: bool
    n_events_delivered: int
    n_handoffs_delivered: int
    n_prompt_tokens_approx: int
    truncated: bool
    grading: dict
    failure_kind: str
    thread_opened: bool
    thread_id: str | None
    n_thread_replies: int
    thread_witness_tokens: int
    thread_resolution_kind: str | None
    thread_winner_kind: str | None
    selected_claim_kind: str | None
    decoder_mode: str
    wall_seconds: float
    handoff_log_length: int
    handoff_chain_ok: bool

    def as_dict(self) -> dict:
        return {
            "scenario_id": self.scenario_id,
            "strategy": self.strategy,
            "contested": self.contested,
            "n_events_delivered": self.n_events_delivered,
            "n_handoffs_delivered": self.n_handoffs_delivered,
            "n_prompt_tokens_approx": self.n_prompt_tokens_approx,
            "truncated": self.truncated,
            "grading": {k: (bool(v) if isinstance(v, bool) else v)
                         for k, v in self.grading.items()
                         if k != "parsed"},
            "parsed": self.grading.get("parsed"),
            "failure_kind": self.failure_kind,
            "thread_opened": self.thread_opened,
            "thread_id": self.thread_id,
            "n_thread_replies": self.n_thread_replies,
            "thread_witness_tokens": self.thread_witness_tokens,
            "thread_resolution_kind": self.thread_resolution_kind,
            "thread_winner_kind": self.thread_winner_kind,
            "selected_claim_kind": self.selected_claim_kind,
            "decoder_mode": self.decoder_mode,
            "wall_seconds": round(self.wall_seconds, 3),
            "handoff_log_length": self.handoff_log_length,
            "handoff_chain_ok": self.handoff_chain_ok,
        }


FAILURE_NONE = "none"
FAILURE_TRUNCATION = "truncation"
FAILURE_STATIC_PRIORITY = "static_priority_pick_wrong"
FAILURE_NO_CONTEST_DETECTED = "no_contest_detected"
FAILURE_RESOLUTION_CONFLICT = "resolution_conflict"
FAILURE_LLM_ERROR = "llm_error"
FAILURE_RETRIEVAL_MISS = "retrieval_miss"


def attribute_failure_p35(scenario: ContestedScenario,
                          strategy: str,
                          grading: dict,
                          truncated: bool,
                          thread_opened: bool,
                          resolution_kind: str | None,
                          decoder_mode: str,
                          ) -> str:
    if grading["full_correct"]:
        return FAILURE_NONE
    if truncated and strategy == STRATEGY_NAIVE:
        return FAILURE_TRUNCATION
    if strategy == STRATEGY_STATIC_HANDOFF:
        return FAILURE_STATIC_PRIORITY
    if strategy == STRATEGY_NAIVE:
        return FAILURE_RETRIEVAL_MISS
    # dynamic / dynamic_wrap / adaptive_sub
    if not thread_opened and scenario.contested:
        return FAILURE_NO_CONTEST_DETECTED
    if resolution_kind in ("CONFLICT", "NO_CONSENSUS", "TIMEOUT"):
        return FAILURE_RESOLUTION_CONFLICT
    if decoder_mode in ("thread_resolution", "adaptive_sub_hypothesis"):
        return FAILURE_LLM_ERROR
    return FAILURE_STATIC_PRIORITY


@dataclass
class ContestedReport:
    scenario_ids: tuple[str, ...]
    strategies: tuple[str, ...]
    measurements: list[ContestedMeasurement]
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
            rc_ok = sum(1 for m in ms
                        if m.grading["root_cause_correct"])
            mean_tok = sum(m.n_prompt_tokens_approx for m in ms) / n
            trunc = sum(1 for m in ms if m.truncated)
            n_threads = sum(1 for m in ms if m.thread_opened)
            n_replies = sum(m.n_thread_replies for m in ms)
            n_witness = sum(m.thread_witness_tokens for m in ms)
            f_hist: dict[str, int] = {}
            for m in ms:
                f_hist[m.failure_kind] = f_hist.get(m.failure_kind, 0) + 1
            contested_ms = [m for m in ms if m.contested]
            c_n = max(1, len(contested_ms))
            c_correct = sum(1 for m in contested_ms
                             if m.grading["full_correct"])
            out[strat] = {
                "n": n,
                "accuracy_full": round(correct / n, 4),
                "accuracy_root_cause": round(rc_ok / n, 4),
                "contested_accuracy_full": round(c_correct / c_n, 4),
                "n_contested": len(contested_ms),
                "mean_prompt_tokens": round(mean_tok, 2),
                "truncated_count": trunc,
                "n_threads_opened": n_threads,
                "n_thread_replies_total": n_replies,
                "n_thread_witness_tokens_total": n_witness,
                "failure_hist": f_hist,
            }
        return out


def run_contested_loop(
        scenarios: Sequence[ContestedScenario],
        auditor: Callable[[str], str],
        strategies: Sequence[str] = ALL_STRATEGIES,
        seed: int = 35,
        max_events_in_prompt: int = 200,
        inbox_capacity: int = 32,
        causality_extractor: Callable[
            [object, str, str, str], str] | None = None,
        claim_extractor: Callable[
            [str, Sequence[IncidentEvent], IncidentScenario],
            list[tuple[str, str, tuple[int, ...]]],
        ] | None = None,
        ) -> ContestedReport:
    measurements: list[ContestedMeasurement] = []
    for scenario in scenarios:
        events = naive_event_stream(scenario.base)
        # Run the typed-handoff protocol once for this scenario
        # (strategy-independent phase-31 layer).
        router = run_contested_handoff_protocol(
            scenario, max_events_per_role=max_events_in_prompt,
            inbox_capacity=inbox_capacity,
            claim_extractor=claim_extractor)
        auditor_inbox = router.inboxes.get(ROLE_AUDITOR)
        pre_dynamic_handoffs = tuple(
            auditor_inbox.peek()) if auditor_inbox else ()
        for strat in strategies:
            thread_opened = False
            thread_id = None
            n_thread_replies = 0
            thread_witness_tokens = 0
            thread_resolution_kind = None
            thread_winner_kind = None
            # Per-strategy: each strategy sees the *same* static
            # handoff bundle; dynamic strategies additionally open
            # a coordination thread and append its resolution.
            if strat in (STRATEGY_DYNAMIC, STRATEGY_DYNAMIC_WRAP):
                # The dynamic router is scenario-local: we re-run
                # the handoff protocol per strategy so each
                # strategy gets its own log + inbox.
                dyn_router = run_contested_handoff_protocol(
                    scenario,
                    max_events_per_role=max_events_in_prompt,
                    inbox_capacity=inbox_capacity,
                    claim_extractor=claim_extractor)
                d_inbox = dyn_router.inboxes.get(ROLE_AUDITOR)
                d_handoffs = tuple(
                    d_inbox.peek()) if d_inbox else ()
                _, debug = run_dynamic_coordination(
                    scenario, dyn_router, d_handoffs,
                    max_rounds=2, witness_token_cap=12,
                    causality_extractor=causality_extractor)
                thread_opened = debug["thread_id"] is not None
                thread_id = debug["thread_id"]
                if thread_opened:
                    state = dyn_router.get_state(thread_id)
                    n_thread_replies = len(state.replies)
                    thread_witness_tokens = sum(
                        r.n_tokens for r in state.replies)
                    thread_resolution_kind = debug["resolution_kind"]
                    if debug["resolution_winner"]:
                        _, thread_winner_kind = debug[
                            "resolution_winner"]
                handoffs_for_decoder = tuple(d_inbox.peek()) if d_inbox else ()
                chain_ok = dyn_router.verify()
                log_len = dyn_router.log_length()
            elif strat == STRATEGY_ADAPTIVE_SUB:
                # Phase-36 Part C — adaptive subscription path.
                adp_router, handoffs_for_decoder, debug = \
                    run_adaptive_sub_coordination(
                        scenario,
                        max_active_edges=4,
                        witness_token_cap=12,
                        causality_extractor=causality_extractor,
                        max_events_per_role=max_events_in_prompt,
                        inbox_capacity=inbox_capacity,
                        claim_extractor=claim_extractor,
                    )
                thread_opened = debug["n_edges_installed"] > 0
                thread_id = None
                n_thread_replies = debug["n_hypotheses_delivered"]
                thread_witness_tokens = 0  # bounded by cap; tokens
                # are inlined into the hypothesis payload.
                thread_resolution_kind = debug["resolution_kind"]
                if debug["resolution_winner"]:
                    _, thread_winner_kind = debug["resolution_winner"]
                chain_ok = adp_router.verify()
                log_len = adp_router.log_length()
            else:
                handoffs_for_decoder = pre_dynamic_handoffs
                chain_ok = router.verify()
                log_len = router.log_length()
            # Decoder + substrate cue for the prompt.
            if strat in (STRATEGY_STATIC_HANDOFF, STRATEGY_DYNAMIC,
                          STRATEGY_DYNAMIC_WRAP,
                          STRATEGY_ADAPTIVE_SUB):
                cue = decoder_from_handoffs_phase35(
                    handoffs_for_decoder)
                delivered_handoffs = handoffs_for_decoder
                decoder_mode = cue["decoder_mode"]
                selected_kind = cue.get("selected_claim_kind")
            else:
                cue = None
                delivered_handoffs = ()
                decoder_mode = "naive"
                selected_kind = None
            prompt, delivered, truncated = build_auditor_prompt_p35(
                scenario, strat, events,
                handoffs=delivered_handoffs,
                substrate_cue=cue,
                max_events_in_prompt=max_events_in_prompt,
            )
            t0 = time.time()
            answer = auditor(prompt)
            wall = time.time() - t0
            grading = grade_answer_p35(scenario, answer)
            failure = attribute_failure_p35(
                scenario, strat, grading, truncated,
                thread_opened, thread_resolution_kind, decoder_mode)
            measurements.append(ContestedMeasurement(
                scenario_id=scenario.scenario_id, strategy=strat,
                contested=scenario.contested,
                n_events_delivered=len(delivered),
                n_handoffs_delivered=len(delivered_handoffs),
                n_prompt_tokens_approx=max(1, len(prompt) // 4),
                truncated=truncated,
                grading=grading, failure_kind=failure,
                thread_opened=thread_opened,
                thread_id=thread_id,
                n_thread_replies=n_thread_replies,
                thread_witness_tokens=thread_witness_tokens,
                thread_resolution_kind=thread_resolution_kind,
                thread_winner_kind=thread_winner_kind,
                selected_claim_kind=selected_kind,
                decoder_mode=decoder_mode,
                wall_seconds=wall,
                handoff_log_length=log_len,
                handoff_chain_ok=chain_ok,
            ))
    return ContestedReport(
        scenario_ids=tuple(s.scenario_id for s in scenarios),
        strategies=tuple(strategies),
        measurements=measurements,
        config={
            "seed": seed,
            "max_events_in_prompt": max_events_in_prompt,
            "inbox_capacity": inbox_capacity,
        },
    )
