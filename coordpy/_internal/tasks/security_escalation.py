"""Phase 33 — multi-role security-audit escalation benchmark.

The programme's third non-code task-scale benchmark and the first whose
subject is a *security incident* under a regulatory / escalation clock.
A team of five role-typed agents — SOC analyst, IR engineer, threat-
intelligence analyst, data steward, CISO (the aggregator) — investigates
a cross-system alert cascade and decides:

  * ``severity``          — informational / low / medium / high / critical
    (a *priority-monotone ordinal*, not just a binary; this is the axis
    that distinguishes Phase 33 from Phase 31/32's decoder shape).
  * ``classification``    — attack family label (phishing / ransomware /
    data_exfil / supply_chain / insider_threat).
  * ``containment``       — canonical action (isolate_host / rotate_keys
    / block_domain / quarantine_code / revoke_access).
  * ``notify``            — set of stakeholder classes that must be
    notified within the regulatory clock (legal / dpo / exec / customer).

Why this task family

  * Each role owns a *different* slice of evidence: SIEM alerts
    (SOC), host forensics (IR), indicator matches (threat intel),
    data-classification inventory (data steward). The CISO has no
    raw telemetry — only role handoffs.
  * The correct answer requires *cross-role* information: a
    phishing-derived credential (SOC) meets a data-exfil staging
    event (IR) meets a known bad-IP (threat_intel) meets a PII
    store (data steward) → severity=critical, classification=
    data_exfil, notify={legal, dpo, exec, customer}.
  * **Ordinal severity** is a different decoder shape than Phase
    31 (priority-order root cause) or Phase 32 (monotone verdict
    + strict-set flags). Phase 33 tests whether the substrate's
    guarantees hold under a *max-ordinal* decoder — where a
    spurious high-severity claim can *escalate* the verdict and
    cause a false-positive in severity.

Scope discipline (what this module does NOT claim)

  * Not a SOC playbook. The five scenarios are hand-crafted
    *compound* incidents chosen to stress the substrate; real
    security escalation involves judgement and negotiation.
  * Not an adversarial-noise claim. The Phase-33 noisy-extractor
    sweep injects i.i.d. Bernoulli noise at the extractor boundary;
    a targeted-drop-of-load-bearing attacker is not modelled.
  * Not a cross-language claim. All payloads are English +
    canonical security jargon, regex-extractable by design.
  * Not a regulatory-compliance claim. The notify set is a
    bench-defined canonical label; real DPO / breach-disclosure
    triggers require legal review.

Theoretical anchor: RESULTS_PHASE33.md § D.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Callable, Sequence

from ..core.role_handoff import (
    HandoffRouter, RoleInbox, RoleSubscriptionTable, TypedHandoff,
)


# =============================================================================
# Role registry
# =============================================================================


ROLE_SOC_ANALYST = "soc_analyst"
ROLE_IR_ENGINEER = "ir_engineer"
ROLE_THREAT_INTEL = "threat_intel"
ROLE_DATA_STEWARD = "data_steward"
ROLE_CISO = "ciso"

ALL_ROLES = (ROLE_SOC_ANALYST, ROLE_IR_ENGINEER, ROLE_THREAT_INTEL,
             ROLE_DATA_STEWARD, ROLE_CISO)


# =============================================================================
# Event schema
# =============================================================================


EVENT_SIEM_ALERT = "SIEM_ALERT"
EVENT_HOST_FORENSIC = "HOST_FORENSIC"
EVENT_IOC_MATCH = "IOC_MATCH"
EVENT_DATA_CLASSIFICATION = "DATA_CLASSIFICATION"
EVENT_TASK_GOAL = "TASK_GOAL"
EVENT_FINAL_ANSWER = "FINAL_ANSWER"

FIXED_POINT_TYPES = frozenset({EVENT_TASK_GOAL, EVENT_FINAL_ANSWER})

ROLE_OBSERVABLE_TYPES: dict[str, frozenset[str]] = {
    ROLE_SOC_ANALYST: frozenset({EVENT_SIEM_ALERT} | FIXED_POINT_TYPES),
    ROLE_IR_ENGINEER: frozenset({EVENT_HOST_FORENSIC} | FIXED_POINT_TYPES),
    ROLE_THREAT_INTEL: frozenset({EVENT_IOC_MATCH} | FIXED_POINT_TYPES),
    ROLE_DATA_STEWARD: frozenset({EVENT_DATA_CLASSIFICATION}
                                 | FIXED_POINT_TYPES),
    ROLE_CISO: frozenset(FIXED_POINT_TYPES),
}


@dataclass(frozen=True)
class SecurityEvent:
    event_id: int
    event_type: str
    origin_role: str
    body: str
    tags: tuple[str, ...] = ()
    is_causal: bool = False

    @property
    def n_tokens(self) -> int:
        if not self.body:
            return 0
        return max(1, len(self.body.split()))

    @property
    def is_fixed_point(self) -> bool:
        return self.event_type in FIXED_POINT_TYPES


# =============================================================================
# Claim taxonomy — short, enumerated, subscription-granularity
# =============================================================================


# SOC (alerts / SIEM signals)
CLAIM_AUTH_SPIKE = "AUTH_SPIKE"
CLAIM_PHISHING_DETECTED = "PHISHING_DETECTED"
CLAIM_LATERAL_MOVEMENT = "LATERAL_MOVEMENT"
CLAIM_BRUTE_FORCE = "BRUTE_FORCE"

# IR (host forensics)
CLAIM_PERSISTENCE_INSTALLED = "PERSISTENCE_INSTALLED"
CLAIM_DATA_STAGING = "DATA_STAGING"
CLAIM_MALWARE_DETECTED = "MALWARE_DETECTED"
CLAIM_PRIV_ESCALATION = "PRIV_ESCALATION"

# Threat intel (IOC)
CLAIM_IOC_KNOWN_BAD_IP = "IOC_KNOWN_BAD_IP"
CLAIM_IOC_MALICIOUS_DOMAIN = "IOC_MALICIOUS_DOMAIN"
CLAIM_TTP_ATTRIBUTED = "TTP_ATTRIBUTED"
CLAIM_SUPPLY_CHAIN_IOC = "SUPPLY_CHAIN_IOC"

# Data steward (data exposure)
CLAIM_REGULATED_DATA_EXPOSED = "REGULATED_DATA_EXPOSED"
CLAIM_PII_AT_RISK = "PII_AT_RISK"
CLAIM_CROSS_TENANT_LEAK = "CROSS_TENANT_LEAK"

ALL_CLAIMS = (
    CLAIM_AUTH_SPIKE, CLAIM_PHISHING_DETECTED,
    CLAIM_LATERAL_MOVEMENT, CLAIM_BRUTE_FORCE,
    CLAIM_PERSISTENCE_INSTALLED, CLAIM_DATA_STAGING,
    CLAIM_MALWARE_DETECTED, CLAIM_PRIV_ESCALATION,
    CLAIM_IOC_KNOWN_BAD_IP, CLAIM_IOC_MALICIOUS_DOMAIN,
    CLAIM_TTP_ATTRIBUTED, CLAIM_SUPPLY_CHAIN_IOC,
    CLAIM_REGULATED_DATA_EXPOSED, CLAIM_PII_AT_RISK,
    CLAIM_CROSS_TENANT_LEAK,
)


def build_role_subscriptions() -> RoleSubscriptionTable:
    """Default subscription table — every claim reaches the CISO; a
    few escalate laterally to other roles (persistence-installed
    copies to threat intel to get IOC correlation; regulated-data
    exposure copies to the SOC for immediate containment)."""
    subs = RoleSubscriptionTable()
    # SOC → CISO
    subs.subscribe(ROLE_SOC_ANALYST, CLAIM_AUTH_SPIKE, [ROLE_CISO])
    subs.subscribe(ROLE_SOC_ANALYST, CLAIM_PHISHING_DETECTED, [ROLE_CISO])
    subs.subscribe(ROLE_SOC_ANALYST, CLAIM_LATERAL_MOVEMENT,
                   [ROLE_CISO, ROLE_IR_ENGINEER])
    subs.subscribe(ROLE_SOC_ANALYST, CLAIM_BRUTE_FORCE, [ROLE_CISO])
    # IR → CISO; persistence copies to threat_intel for IOC enrichment
    subs.subscribe(ROLE_IR_ENGINEER, CLAIM_PERSISTENCE_INSTALLED,
                   [ROLE_CISO, ROLE_THREAT_INTEL])
    subs.subscribe(ROLE_IR_ENGINEER, CLAIM_DATA_STAGING,
                   [ROLE_CISO, ROLE_DATA_STEWARD])
    subs.subscribe(ROLE_IR_ENGINEER, CLAIM_MALWARE_DETECTED,
                   [ROLE_CISO, ROLE_THREAT_INTEL])
    subs.subscribe(ROLE_IR_ENGINEER, CLAIM_PRIV_ESCALATION, [ROLE_CISO])
    # Threat intel → CISO
    subs.subscribe(ROLE_THREAT_INTEL, CLAIM_IOC_KNOWN_BAD_IP, [ROLE_CISO])
    subs.subscribe(ROLE_THREAT_INTEL, CLAIM_IOC_MALICIOUS_DOMAIN,
                   [ROLE_CISO])
    subs.subscribe(ROLE_THREAT_INTEL, CLAIM_TTP_ATTRIBUTED, [ROLE_CISO])
    subs.subscribe(ROLE_THREAT_INTEL, CLAIM_SUPPLY_CHAIN_IOC,
                   [ROLE_CISO, ROLE_IR_ENGINEER])
    # Data steward → CISO; regulated-data exposure also copies SOC
    subs.subscribe(ROLE_DATA_STEWARD, CLAIM_REGULATED_DATA_EXPOSED,
                   [ROLE_CISO, ROLE_SOC_ANALYST])
    subs.subscribe(ROLE_DATA_STEWARD, CLAIM_PII_AT_RISK,
                   [ROLE_CISO])
    subs.subscribe(ROLE_DATA_STEWARD, CLAIM_CROSS_TENANT_LEAK,
                   [ROLE_CISO, ROLE_SOC_ANALYST])
    return subs


# =============================================================================
# Scenario catalogue — five compound security incidents
# =============================================================================


# Severity ordinal: lower index = less severe.
SEVERITY_INFORMATIONAL = "informational"
SEVERITY_LOW = "low"
SEVERITY_MEDIUM = "medium"
SEVERITY_HIGH = "high"
SEVERITY_CRITICAL = "critical"

SEVERITY_ORDER = (SEVERITY_INFORMATIONAL, SEVERITY_LOW,
                  SEVERITY_MEDIUM, SEVERITY_HIGH, SEVERITY_CRITICAL)

SEVERITY_INDEX = {s: i for i, s in enumerate(SEVERITY_ORDER)}


@dataclass(frozen=True)
class SecurityScenario:
    scenario_id: str
    description: str
    gold_severity: str
    gold_classification: str
    gold_containment: str
    gold_notify: tuple[str, ...]
    causal_chain: tuple[tuple[str, str, str, tuple[int, ...]], ...]
    per_role_events: dict[str, tuple[SecurityEvent, ...]]


def _mk(nid: int, et: str, role: str, body: str,
        *, tags: Sequence[str] = (), causal: bool = False,
        ) -> tuple[SecurityEvent, int]:
    ev = SecurityEvent(event_id=nid, event_type=et, origin_role=role,
                       body=body, tags=tuple(tags), is_causal=causal)
    return ev, nid + 1


def _distractors(rng: random.Random, nid: int, role: str,
                  kinds: Sequence[str], k: int,
                  ) -> tuple[list[SecurityEvent], int]:
    benign = {
        EVENT_SIEM_ALERT: [
            "rule=info source=edr msg=routine_scan_complete",
            "rule=info source=ids msg=signature_db_updated",
            "rule=info source=dns msg=resolver_healthy",
            "rule=low source=vpn msg=user=u22 login=ok",
            "rule=low source=edr msg=file_access user=u11 ok=true",
        ],
        EVENT_HOST_FORENSIC: [
            "process ppid=1 pid=3302 cmd=/usr/sbin/cron state=running",
            "file hash=sha256:abcd123 path=/usr/bin/bash trust=known",
            "network conn src=10.0.1.4 dst=10.0.1.11 dport=443 ok=true",
            "persistence registry_key=none path=/etc/cron.daily ok=true",
        ],
        EVENT_IOC_MATCH: [
            "ioc=ip:203.0.113.7 type=dns hit=known-good rep=95",
            "ioc=domain:internal.intra type=dns hit=allowed rep=100",
            "ioc=sha256:aa11 type=hash hit=known-good",
        ],
        EVENT_DATA_CLASSIFICATION: [
            "class=public store=marketing_blog volume=small",
            "class=internal store=design_docs volume=medium owner=eng",
            "class=confidential store=finance_reports volume=small "
            "owner=cfo ok=true",
        ],
    }
    out: list[SecurityEvent] = []
    for _ in range(k):
        et = rng.choice(list(kinds))
        body = rng.choice(benign.get(et, ["benign"]))
        ev, nid = _mk(nid, et, role, body, causal=False)
        out.append(ev)
    return out, nid


def make_scenario_phishing_exfil(rng: random.Random,
                                   start_id: int = 0,
                                   distractors_per_role: int = 6,
                                   ) -> SecurityScenario:
    """Phishing → credential theft → data staging → exfil.

    CRITICAL severity, data_exfil classification, isolate_host
    containment, notify {legal, dpo, exec, customer}.
    """
    nid = start_id
    per_role: dict[str, list[SecurityEvent]] = {r: [] for r in ALL_ROLES}
    chain: list[tuple[str, str, str, tuple[int, ...]]] = []

    # SOC: phishing alert
    ev, nid = _mk(nid, EVENT_SIEM_ALERT, ROLE_SOC_ANALYST,
                   "rule=phishing source=email msg=credential_harvest "
                   "user=alice vendor=acme",
                   tags=("alice",), causal=True)
    per_role[ROLE_SOC_ANALYST].append(ev)
    ph_id = ev.event_id
    ev, nid = _mk(nid, EVENT_SIEM_ALERT, ROLE_SOC_ANALYST,
                   "rule=auth source=sso msg=login_burst "
                   "user=alice country=XX count=42",
                   tags=("alice",), causal=True)
    per_role[ROLE_SOC_ANALYST].append(ev)
    au_id = ev.event_id
    chain.append((ROLE_SOC_ANALYST, CLAIM_PHISHING_DETECTED,
                   "credential_harvest user=alice", (ph_id,)))
    chain.append((ROLE_SOC_ANALYST, CLAIM_AUTH_SPIKE,
                   "login_burst user=alice count=42", (au_id,)))

    # IR: data staging
    ev, nid = _mk(nid, EVENT_HOST_FORENSIC, ROLE_IR_ENGINEER,
                   "process pid=8842 cmd=tar czf /tmp/exfil.tgz "
                   "/mnt/customer_db",
                   tags=("customer_db",), causal=True)
    per_role[ROLE_IR_ENGINEER].append(ev)
    st_id = ev.event_id
    chain.append((ROLE_IR_ENGINEER, CLAIM_DATA_STAGING,
                   "tar /mnt/customer_db /tmp/exfil.tgz", (st_id,)))

    # Threat intel: known-bad IP
    ev, nid = _mk(nid, EVENT_IOC_MATCH, ROLE_THREAT_INTEL,
                   "ioc=ip:185.220.101.45 type=dns hit=known-bad "
                   "actor=group_x rep=10",
                   tags=("group_x",), causal=True)
    per_role[ROLE_THREAT_INTEL].append(ev)
    io_id = ev.event_id
    chain.append((ROLE_THREAT_INTEL, CLAIM_IOC_KNOWN_BAD_IP,
                   "ip:185.220.101.45 actor=group_x", (io_id,)))

    # Data steward: regulated data
    ev, nid = _mk(nid, EVENT_DATA_CLASSIFICATION, ROLE_DATA_STEWARD,
                   "class=regulated_pii store=customer_db volume=large "
                   "regulation=gdpr",
                   tags=("customer_db",), causal=True)
    per_role[ROLE_DATA_STEWARD].append(ev)
    ds_id = ev.event_id
    chain.append((ROLE_DATA_STEWARD, CLAIM_REGULATED_DATA_EXPOSED,
                   "store=customer_db regulation=gdpr", (ds_id,)))

    for role, kinds in ((ROLE_SOC_ANALYST, [EVENT_SIEM_ALERT]),
                         (ROLE_IR_ENGINEER, [EVENT_HOST_FORENSIC]),
                         (ROLE_THREAT_INTEL, [EVENT_IOC_MATCH]),
                         (ROLE_DATA_STEWARD,
                          [EVENT_DATA_CLASSIFICATION])):
        dist, nid = _distractors(rng, nid, role, kinds,
                                   k=distractors_per_role)
        per_role[role].extend(dist)

    return SecurityScenario(
        scenario_id="phishing_exfil",
        description=("credential phishing → login burst → data "
                      "staging to /tmp/exfil.tgz → outbound to "
                      "known-bad IP"),
        gold_severity=SEVERITY_CRITICAL,
        gold_classification="data_exfil",
        gold_containment="isolate_host_and_revoke_credentials",
        gold_notify=("customer", "dpo", "exec", "legal"),
        causal_chain=tuple(chain),
        per_role_events={r: tuple(evs) for r, evs in per_role.items()},
    )


def make_scenario_ransomware_precursor(rng: random.Random,
                                         start_id: int = 0,
                                         distractors_per_role: int = 6,
                                         ) -> SecurityScenario:
    """Ransomware precursor — persistence + lateral movement + malware.

    HIGH severity, ransomware classification, quarantine_code
    containment, notify {legal, exec}.
    """
    nid = start_id
    per_role: dict[str, list[SecurityEvent]] = {r: [] for r in ALL_ROLES}
    chain: list[tuple[str, str, str, tuple[int, ...]]] = []

    ev, nid = _mk(nid, EVENT_HOST_FORENSIC, ROLE_IR_ENGINEER,
                   "persistence registry_key=HKLM\\Run\\backup "
                   "path=C:\\temp\\svc.exe hash=sha256:bad123",
                   tags=("endpoint-a",), causal=True)
    per_role[ROLE_IR_ENGINEER].append(ev)
    ps_id = ev.event_id
    ev, nid = _mk(nid, EVENT_HOST_FORENSIC, ROLE_IR_ENGINEER,
                   "malware_detection hash=sha256:bad123 "
                   "family=ransomware.conti score=99",
                   tags=("endpoint-a",), causal=True)
    per_role[ROLE_IR_ENGINEER].append(ev)
    mw_id = ev.event_id
    chain.append((ROLE_IR_ENGINEER, CLAIM_PERSISTENCE_INSTALLED,
                   "registry_key=HKLM\\Run\\backup hash=sha256:bad123",
                   (ps_id,)))
    chain.append((ROLE_IR_ENGINEER, CLAIM_MALWARE_DETECTED,
                   "family=ransomware.conti hash=sha256:bad123",
                   (mw_id,)))

    ev, nid = _mk(nid, EVENT_SIEM_ALERT, ROLE_SOC_ANALYST,
                   "rule=lateral source=edr msg=smb_share_enum "
                   "src=endpoint-a dst=fileserver",
                   tags=("endpoint-a",), causal=True)
    per_role[ROLE_SOC_ANALYST].append(ev)
    lm_id = ev.event_id
    chain.append((ROLE_SOC_ANALYST, CLAIM_LATERAL_MOVEMENT,
                   "smb_share_enum src=endpoint-a dst=fileserver",
                   (lm_id,)))

    ev, nid = _mk(nid, EVENT_IOC_MATCH, ROLE_THREAT_INTEL,
                   "ioc=sha256:bad123 type=hash hit=known-bad "
                   "actor=conti_group rep=5",
                   tags=("conti_group",), causal=True)
    per_role[ROLE_THREAT_INTEL].append(ev)
    tp_id = ev.event_id
    chain.append((ROLE_THREAT_INTEL, CLAIM_TTP_ATTRIBUTED,
                   "actor=conti_group hash=sha256:bad123", (tp_id,)))

    for role, kinds in ((ROLE_SOC_ANALYST, [EVENT_SIEM_ALERT]),
                         (ROLE_IR_ENGINEER, [EVENT_HOST_FORENSIC]),
                         (ROLE_THREAT_INTEL, [EVENT_IOC_MATCH]),
                         (ROLE_DATA_STEWARD,
                          [EVENT_DATA_CLASSIFICATION])):
        dist, nid = _distractors(rng, nid, role, kinds,
                                   k=distractors_per_role)
        per_role[role].extend(dist)

    return SecurityScenario(
        scenario_id="ransomware_precursor",
        description=("ransomware precursor — conti persistence on "
                      "endpoint-a → SMB-share enum lateral → malware "
                      "hash attributed to known actor"),
        gold_severity=SEVERITY_HIGH,
        gold_classification="ransomware",
        gold_containment="quarantine_code_and_block_actor",
        gold_notify=("exec", "legal"),
        causal_chain=tuple(chain),
        per_role_events={r: tuple(evs) for r, evs in per_role.items()},
    )


def make_scenario_supply_chain(rng: random.Random,
                                 start_id: int = 0,
                                 distractors_per_role: int = 6,
                                 ) -> SecurityScenario:
    """Supply-chain compromise — vendor dependency compromised.

    HIGH severity, supply_chain classification, quarantine_code
    containment, notify {legal, exec}.
    """
    nid = start_id
    per_role: dict[str, list[SecurityEvent]] = {r: [] for r in ALL_ROLES}
    chain: list[tuple[str, str, str, tuple[int, ...]]] = []

    ev, nid = _mk(nid, EVENT_IOC_MATCH, ROLE_THREAT_INTEL,
                   "ioc=package:acme-logger@4.2.1 type=supply_chain "
                   "hit=known-bad rep=2 advisory=CVE-2026-0042",
                   tags=("acme-logger",), causal=True)
    per_role[ROLE_THREAT_INTEL].append(ev)
    sc_id = ev.event_id
    chain.append((ROLE_THREAT_INTEL, CLAIM_SUPPLY_CHAIN_IOC,
                   "package:acme-logger@4.2.1 advisory=CVE-2026-0042",
                   (sc_id,)))

    ev, nid = _mk(nid, EVENT_HOST_FORENSIC, ROLE_IR_ENGINEER,
                   "process pid=1002 cmd=node server.js "
                   "module_path=acme-logger module_version=4.2.1",
                   tags=("api-server",), causal=True)
    per_role[ROLE_IR_ENGINEER].append(ev)
    mw_id = ev.event_id
    ev, nid = _mk(nid, EVENT_HOST_FORENSIC, ROLE_IR_ENGINEER,
                   "malware_detection hash=sha256:scbad score=90 "
                   "origin=acme-logger@4.2.1",
                   tags=("api-server",), causal=True)
    per_role[ROLE_IR_ENGINEER].append(ev)
    md_id = ev.event_id
    chain.append((ROLE_IR_ENGINEER, CLAIM_MALWARE_DETECTED,
                   "hash=sha256:scbad origin=acme-logger@4.2.1",
                   (md_id,)))

    ev, nid = _mk(nid, EVENT_SIEM_ALERT, ROLE_SOC_ANALYST,
                   "rule=lateral source=edr msg=abnormal_outbound "
                   "src=api-server dst=185.220.101.88",
                   tags=("api-server",), causal=True)
    per_role[ROLE_SOC_ANALYST].append(ev)
    lm_id = ev.event_id
    chain.append((ROLE_SOC_ANALYST, CLAIM_LATERAL_MOVEMENT,
                   "abnormal_outbound src=api-server", (lm_id,)))

    for role, kinds in ((ROLE_SOC_ANALYST, [EVENT_SIEM_ALERT]),
                         (ROLE_IR_ENGINEER, [EVENT_HOST_FORENSIC]),
                         (ROLE_THREAT_INTEL, [EVENT_IOC_MATCH]),
                         (ROLE_DATA_STEWARD,
                          [EVENT_DATA_CLASSIFICATION])):
        dist, nid = _distractors(rng, nid, role, kinds,
                                   k=distractors_per_role)
        per_role[role].extend(dist)

    return SecurityScenario(
        scenario_id="supply_chain",
        description=("supply-chain compromise — acme-logger npm "
                      "package 4.2.1 compromised (CVE-2026-0042) → "
                      "running on api-server → outbound to bad IP"),
        gold_severity=SEVERITY_HIGH,
        gold_classification="supply_chain",
        gold_containment="quarantine_code_and_block_actor",
        gold_notify=("exec", "legal"),
        causal_chain=tuple(chain),
        per_role_events={r: tuple(evs) for r, evs in per_role.items()},
    )


def make_scenario_insider_threat(rng: random.Random,
                                    start_id: int = 0,
                                    distractors_per_role: int = 6,
                                    ) -> SecurityScenario:
    """Insider threat — employee staging cross-tenant data.

    HIGH severity, insider_threat classification, revoke_access
    containment, notify {legal, dpo, exec}.
    """
    nid = start_id
    per_role: dict[str, list[SecurityEvent]] = {r: [] for r in ALL_ROLES}
    chain: list[tuple[str, str, str, tuple[int, ...]]] = []

    ev, nid = _mk(nid, EVENT_HOST_FORENSIC, ROLE_IR_ENGINEER,
                   "process pid=4101 cmd=rsync "
                   "/mnt/tenant_a/export.csv s3://personal-bucket/ "
                   "user=bob",
                   tags=("bob",), causal=True)
    per_role[ROLE_IR_ENGINEER].append(ev)
    st_id = ev.event_id
    chain.append((ROLE_IR_ENGINEER, CLAIM_DATA_STAGING,
                   "rsync /mnt/tenant_a/export.csv s3://personal-bucket",
                   (st_id,)))

    ev, nid = _mk(nid, EVENT_DATA_CLASSIFICATION, ROLE_DATA_STEWARD,
                   "class=cross_tenant store=tenant_a volume=large "
                   "actor=bob tenant_of_record=tenant_b",
                   tags=("bob",), causal=True)
    per_role[ROLE_DATA_STEWARD].append(ev)
    ct_id = ev.event_id
    ev, nid = _mk(nid, EVENT_DATA_CLASSIFICATION, ROLE_DATA_STEWARD,
                   "class=regulated_pii store=tenant_a regulation=ccpa "
                   "volume=large",
                   tags=("tenant_a",), causal=True)
    per_role[ROLE_DATA_STEWARD].append(ev)
    pi_id = ev.event_id
    chain.append((ROLE_DATA_STEWARD, CLAIM_CROSS_TENANT_LEAK,
                   "tenant_a actor=bob tenant_of_record=tenant_b",
                   (ct_id,)))
    chain.append((ROLE_DATA_STEWARD, CLAIM_PII_AT_RISK,
                   "tenant_a regulation=ccpa", (pi_id,)))

    ev, nid = _mk(nid, EVENT_SIEM_ALERT, ROLE_SOC_ANALYST,
                   "rule=auth source=sso msg=off_hours_access "
                   "user=bob count=12 hour=03",
                   tags=("bob",), causal=True)
    per_role[ROLE_SOC_ANALYST].append(ev)
    au_id = ev.event_id
    chain.append((ROLE_SOC_ANALYST, CLAIM_AUTH_SPIKE,
                   "off_hours_access user=bob hour=03", (au_id,)))

    for role, kinds in ((ROLE_SOC_ANALYST, [EVENT_SIEM_ALERT]),
                         (ROLE_IR_ENGINEER, [EVENT_HOST_FORENSIC]),
                         (ROLE_THREAT_INTEL, [EVENT_IOC_MATCH]),
                         (ROLE_DATA_STEWARD,
                          [EVENT_DATA_CLASSIFICATION])):
        dist, nid = _distractors(rng, nid, role, kinds,
                                   k=distractors_per_role)
        per_role[role].extend(dist)

    return SecurityScenario(
        scenario_id="insider_threat",
        description=("insider threat — bob rsyncs tenant_a export.csv "
                      "to personal-bucket at 03:00, cross-tenant "
                      "regulated PII"),
        gold_severity=SEVERITY_HIGH,
        gold_classification="insider_threat",
        gold_containment="revoke_access_and_forensic_preserve",
        gold_notify=("dpo", "exec", "legal"),
        causal_chain=tuple(chain),
        per_role_events={r: tuple(evs) for r, evs in per_role.items()},
    )


def make_scenario_brute_force_low(rng: random.Random,
                                    start_id: int = 0,
                                    distractors_per_role: int = 6,
                                    ) -> SecurityScenario:
    """Brute-force attempts, no compromise — MEDIUM severity.

    Tests the *ordinal* decoder: the scenario is serious enough to
    escalate past LOW, but the absence of compromise keeps it below
    HIGH. A spurious MALWARE_DETECTED claim from a noisy extractor
    would wrongly escalate to CRITICAL in a strict decoder — this is
    the scenario that lights up the severity-monotone failure mode.
    """
    nid = start_id
    per_role: dict[str, list[SecurityEvent]] = {r: [] for r in ALL_ROLES}
    chain: list[tuple[str, str, str, tuple[int, ...]]] = []

    ev, nid = _mk(nid, EVENT_SIEM_ALERT, ROLE_SOC_ANALYST,
                   "rule=auth source=sso msg=brute_force "
                   "target=admin_portal count=5200 window=10m "
                   "src=185.220.101.55",
                   tags=("admin_portal",), causal=True)
    per_role[ROLE_SOC_ANALYST].append(ev)
    bf_id = ev.event_id
    chain.append((ROLE_SOC_ANALYST, CLAIM_BRUTE_FORCE,
                   "target=admin_portal count=5200 window=10m",
                   (bf_id,)))

    ev, nid = _mk(nid, EVENT_IOC_MATCH, ROLE_THREAT_INTEL,
                   "ioc=ip:185.220.101.55 type=dns hit=known-bad "
                   "actor=scanner_cluster rep=15",
                   tags=("scanner_cluster",), causal=True)
    per_role[ROLE_THREAT_INTEL].append(ev)
    io_id = ev.event_id
    chain.append((ROLE_THREAT_INTEL, CLAIM_IOC_KNOWN_BAD_IP,
                   "ip:185.220.101.55 actor=scanner_cluster",
                   (io_id,)))

    for role, kinds in ((ROLE_SOC_ANALYST, [EVENT_SIEM_ALERT]),
                         (ROLE_IR_ENGINEER, [EVENT_HOST_FORENSIC]),
                         (ROLE_THREAT_INTEL, [EVENT_IOC_MATCH]),
                         (ROLE_DATA_STEWARD,
                          [EVENT_DATA_CLASSIFICATION])):
        dist, nid = _distractors(rng, nid, role, kinds,
                                   k=distractors_per_role)
        per_role[role].extend(dist)

    return SecurityScenario(
        scenario_id="brute_force_blocked",
        description=("5200 brute-force attempts from a known-bad IP "
                      "hit the admin portal; firewall blocked, no "
                      "compromise observed"),
        gold_severity=SEVERITY_MEDIUM,
        gold_classification="reconnaissance",
        gold_containment="block_domain_and_rate_limit",
        gold_notify=("exec",),
        causal_chain=tuple(chain),
        per_role_events={r: tuple(evs) for r, evs in per_role.items()},
    )


SCENARIO_BUILDERS: tuple[
    Callable[..., SecurityScenario], ...] = (
    make_scenario_phishing_exfil,
    make_scenario_ransomware_precursor,
    make_scenario_supply_chain,
    make_scenario_insider_threat,
    make_scenario_brute_force_low,
)


def build_scenario_bank(seed: int = 33,
                         distractors_per_role: int = 6,
                         ) -> list[SecurityScenario]:
    rng = random.Random(seed)
    out: list[SecurityScenario] = []
    for builder in SCENARIO_BUILDERS:
        s = builder(rng, 0, distractors_per_role=distractors_per_role)
        out.append(s)
    return out


# =============================================================================
# Global event-stream assembly
# =============================================================================


def fixed_point_events(scenario: SecurityScenario
                         ) -> list[SecurityEvent]:
    goal = SecurityEvent(
        event_id=-1, event_type=EVENT_TASK_GOAL,
        origin_role="__system__",
        body=(f"[task] security-escalation {scenario.scenario_id}: "
              f"set severity, classify, name containment action, "
              f"list notify stakeholders"))
    final = SecurityEvent(
        event_id=-2, event_type=EVENT_FINAL_ANSWER,
        origin_role="__system__",
        body="[final] <placeholder>")
    return [goal, final]


def naive_event_stream(scenario: SecurityScenario
                         ) -> list[SecurityEvent]:
    out: list[SecurityEvent] = list(fixed_point_events(scenario))
    for role in ALL_ROLES:
        out.extend(scenario.per_role_events.get(role, ()))
    return out


# =============================================================================
# Oracle
# =============================================================================


def oracle_relevance(ev: SecurityEvent, role: str,
                      scenario: SecurityScenario) -> bool:
    if ev.is_fixed_point:
        return True
    obs = ROLE_OBSERVABLE_TYPES.get(role, frozenset())
    if role == ROLE_CISO:
        return bool(ev.is_causal)
    if ev.event_type not in obs:
        return False
    return bool(ev.is_causal) and ev.origin_role == role


def handoff_is_relevant(h: TypedHandoff,
                         scenario: SecurityScenario) -> bool:
    pairs = {(role, kind)
             for (role, kind, _p, _e) in scenario.causal_chain}
    return (h.source_role, h.claim_kind) in pairs


# =============================================================================
# Extractors — regex-based (deterministic baseline)
# =============================================================================


def extract_claims_for_role(role: str,
                             events: Sequence[SecurityEvent],
                             scenario: SecurityScenario,
                             ) -> list[tuple[str, str, tuple[int, ...]]]:
    out: list[tuple[str, str, tuple[int, ...]]] = []

    def _emit(kind: str, body: str, evids: Sequence[int]) -> None:
        out.append((kind, body, tuple(evids)))

    if role == ROLE_SOC_ANALYST:
        for ev in events:
            if ev.event_type != EVENT_SIEM_ALERT:
                continue
            if re.search(r"rule=phishing|credential_harvest", ev.body):
                _emit(CLAIM_PHISHING_DETECTED, ev.body, [ev.event_id])
            if re.search(r"login_burst|off_hours_access", ev.body):
                _emit(CLAIM_AUTH_SPIKE, ev.body, [ev.event_id])
            if re.search(r"brute_force.*count=(\d+)", ev.body):
                m = re.search(r"count=(\d+)", ev.body)
                if m and int(m.group(1)) >= 100:
                    _emit(CLAIM_BRUTE_FORCE, ev.body, [ev.event_id])
            if re.search(r"smb_share_enum|abnormal_outbound",
                          ev.body):
                _emit(CLAIM_LATERAL_MOVEMENT, ev.body, [ev.event_id])
    elif role == ROLE_IR_ENGINEER:
        for ev in events:
            if ev.event_type != EVENT_HOST_FORENSIC:
                continue
            if re.search(r"persistence\s+registry_key=HK", ev.body):
                _emit(CLAIM_PERSISTENCE_INSTALLED, ev.body,
                      [ev.event_id])
            if re.search(r"malware_detection", ev.body):
                _emit(CLAIM_MALWARE_DETECTED, ev.body, [ev.event_id])
            if re.search(r"rsync .*s3://|tar\s+\w*z\w*\s+/tmp/exfil",
                          ev.body):
                _emit(CLAIM_DATA_STAGING, ev.body, [ev.event_id])
            if re.search(r"privilege_escalation|setuid", ev.body):
                _emit(CLAIM_PRIV_ESCALATION, ev.body, [ev.event_id])
    elif role == ROLE_THREAT_INTEL:
        for ev in events:
            if ev.event_type != EVENT_IOC_MATCH:
                continue
            if re.search(r"type=supply_chain.*hit=known-bad",
                          ev.body):
                _emit(CLAIM_SUPPLY_CHAIN_IOC, ev.body, [ev.event_id])
            elif "type=hash" in ev.body and "hit=known-bad" in ev.body:
                _emit(CLAIM_TTP_ATTRIBUTED, ev.body, [ev.event_id])
            elif re.search(r"ioc=ip:.*hit=known-bad", ev.body):
                _emit(CLAIM_IOC_KNOWN_BAD_IP, ev.body, [ev.event_id])
            elif re.search(r"ioc=domain:.*hit=known-bad", ev.body):
                _emit(CLAIM_IOC_MALICIOUS_DOMAIN, ev.body,
                      [ev.event_id])
    elif role == ROLE_DATA_STEWARD:
        for ev in events:
            if ev.event_type != EVENT_DATA_CLASSIFICATION:
                continue
            if re.search(r"class=regulated_pii", ev.body):
                _emit(CLAIM_REGULATED_DATA_EXPOSED, ev.body,
                      [ev.event_id])
                if "regulation=" in ev.body:
                    _emit(CLAIM_PII_AT_RISK, ev.body, [ev.event_id])
            if re.search(r"class=cross_tenant", ev.body):
                _emit(CLAIM_CROSS_TENANT_LEAK, ev.body, [ev.event_id])
    return out


# =============================================================================
# Delivery strategies
# =============================================================================


STRATEGY_NAIVE = "naive"
STRATEGY_ROUTING = "routing"
STRATEGY_SUBSTRATE = "substrate"
STRATEGY_SUBSTRATE_WRAP = "substrate_wrap"

ALL_STRATEGIES = (STRATEGY_NAIVE, STRATEGY_ROUTING,
                  STRATEGY_SUBSTRATE, STRATEGY_SUBSTRATE_WRAP)


def run_handoff_protocol(scenario: SecurityScenario,
                          max_events_per_role: int = 400,
                          inbox_capacity: int = 64,
                          extractor: Callable[
                              [str, Sequence[SecurityEvent],
                               SecurityScenario],
                              list[tuple[str, str, tuple[int, ...]]],
                          ] | None = None,
                          ) -> HandoffRouter:
    subs = build_role_subscriptions()
    router = HandoffRouter(subs=subs)
    for role in ALL_ROLES:
        router.register_inbox(RoleInbox(role=role,
                                         capacity=inbox_capacity))
    extractor = extractor or extract_claims_for_role
    for role in (ROLE_SOC_ANALYST, ROLE_IR_ENGINEER, ROLE_THREAT_INTEL,
                  ROLE_DATA_STEWARD):
        evs = list(scenario.per_role_events.get(role, ()))
        if max_events_per_role:
            evs = evs[:max_events_per_role]
        claims = extractor(role, evs, scenario)
        for (kind, payload, evids) in claims:
            router.emit(source_role=role,
                         source_agent_id=ALL_ROLES.index(role),
                         claim_kind=kind, payload=payload,
                         source_event_ids=evids, round=1)
    return router


# =============================================================================
# Decoder — max-ordinal severity + claim-set classification
# =============================================================================


# Severity per claim kind — the decoder takes the MAX over delivered
# claim kinds (plus a compound rule for scenarios that need multiple
# claim kinds to reach CRITICAL).
_CLAIM_SEVERITY: dict[str, str] = {
    CLAIM_PHISHING_DETECTED: SEVERITY_HIGH,
    CLAIM_AUTH_SPIKE: SEVERITY_LOW,
    CLAIM_LATERAL_MOVEMENT: SEVERITY_HIGH,
    CLAIM_BRUTE_FORCE: SEVERITY_MEDIUM,
    CLAIM_PERSISTENCE_INSTALLED: SEVERITY_HIGH,
    CLAIM_MALWARE_DETECTED: SEVERITY_HIGH,
    CLAIM_DATA_STAGING: SEVERITY_HIGH,
    CLAIM_PRIV_ESCALATION: SEVERITY_HIGH,
    CLAIM_IOC_KNOWN_BAD_IP: SEVERITY_MEDIUM,
    CLAIM_IOC_MALICIOUS_DOMAIN: SEVERITY_MEDIUM,
    CLAIM_TTP_ATTRIBUTED: SEVERITY_HIGH,
    CLAIM_SUPPLY_CHAIN_IOC: SEVERITY_HIGH,
    CLAIM_REGULATED_DATA_EXPOSED: SEVERITY_HIGH,
    CLAIM_PII_AT_RISK: SEVERITY_MEDIUM,
    CLAIM_CROSS_TENANT_LEAK: SEVERITY_HIGH,
}


# Classification rules — (set of required claim kinds) → classification
# label. Evaluated in priority order; the first match wins.
_CLASSIFICATION_RULES: tuple[
    tuple[frozenset[str], str, str, tuple[str, ...], str], ...] = (
    # phishing + data staging + regulated data + known-bad IP → data exfil
    (frozenset({CLAIM_PHISHING_DETECTED, CLAIM_DATA_STAGING,
                CLAIM_REGULATED_DATA_EXPOSED}),
     "data_exfil", "isolate_host_and_revoke_credentials",
     ("customer", "dpo", "exec", "legal"), SEVERITY_CRITICAL),
    # persistence + malware + lateral movement → ransomware
    (frozenset({CLAIM_PERSISTENCE_INSTALLED, CLAIM_MALWARE_DETECTED,
                CLAIM_LATERAL_MOVEMENT}),
     "ransomware", "quarantine_code_and_block_actor",
     ("exec", "legal"), SEVERITY_HIGH),
    # supply chain IoC + malware detected → supply_chain
    (frozenset({CLAIM_SUPPLY_CHAIN_IOC, CLAIM_MALWARE_DETECTED}),
     "supply_chain", "quarantine_code_and_block_actor",
     ("exec", "legal"), SEVERITY_HIGH),
    # cross-tenant + data staging + PII at risk → insider threat
    (frozenset({CLAIM_CROSS_TENANT_LEAK, CLAIM_DATA_STAGING,
                CLAIM_PII_AT_RISK}),
     "insider_threat", "revoke_access_and_forensic_preserve",
     ("dpo", "exec", "legal"), SEVERITY_HIGH),
    # brute force + known-bad IP (no compromise) → reconnaissance
    (frozenset({CLAIM_BRUTE_FORCE, CLAIM_IOC_KNOWN_BAD_IP}),
     "reconnaissance", "block_domain_and_rate_limit",
     ("exec",), SEVERITY_MEDIUM),
)


def decode_from_handoffs(handoffs: Sequence[TypedHandoff]
                           ) -> dict[str, object]:
    """Max-ordinal severity + claim-set classification decoder.

    * severity: max-ordinal over ``_CLAIM_SEVERITY`` for every
      delivered claim kind. An empty inbox → ``informational``.
    * classification: the first ``_CLASSIFICATION_RULES`` rule whose
      required claim set is a subset of the delivered claim kinds.
      No rule matches → ``unclassified``.
    * containment + notify: from the matching rule; fallback
      ``investigate`` / empty set.

    Monotonicity note: severity is *non-monotone* under spurious
    claims (a spurious high-severity claim escalates the verdict).
    This is the Phase-33 test for the Theorem-P32-2 strict-decoder
    regime on a different decoder shape — the severity axis is
    essentially a max-reduction, which by construction is sensitive
    to precision failures.
    """
    kinds = {h.claim_kind for h in handoffs}
    # Severity — max ordinal over delivered claim kinds.
    sev = SEVERITY_INFORMATIONAL
    for k in kinds:
        s = _CLAIM_SEVERITY.get(k)
        if s and SEVERITY_INDEX[s] > SEVERITY_INDEX[sev]:
            sev = s
    # Classification — first matching rule.
    classification = "unclassified"
    containment = "investigate"
    notify: tuple[str, ...] = ()
    for (required, cls, cont, nt, min_sev) in _CLASSIFICATION_RULES:
        if required.issubset(kinds):
            classification = cls
            containment = cont
            notify = nt
            if SEVERITY_INDEX[min_sev] > SEVERITY_INDEX[sev]:
                sev = min_sev
            break
    return {
        "severity": sev,
        "classification": classification,
        "containment": containment,
        "notify": tuple(sorted(notify)),
    }


def _decoder_from_events(events: Sequence[SecurityEvent]
                          ) -> dict[str, object]:
    by_role: dict[str, list[SecurityEvent]] = {r: [] for r in ALL_ROLES}
    for ev in events:
        if ev.origin_role in by_role:
            by_role[ev.origin_role].append(ev)
    synth: list[TypedHandoff] = []
    nid = 0
    for role in (ROLE_SOC_ANALYST, ROLE_IR_ENGINEER,
                  ROLE_THREAT_INTEL, ROLE_DATA_STEWARD):
        claims = extract_claims_for_role(role, by_role[role],
                                          _DUMMY_SCENARIO)
        for (kind, payload, evids) in claims:
            synth.append(TypedHandoff(
                handoff_id=nid, source_role=role,
                source_agent_id=ALL_ROLES.index(role),
                to_role=ROLE_CISO, claim_kind=kind,
                payload=payload, source_event_ids=tuple(evids),
                round=1, payload_cid="", prev_chain_hash="",
                chain_hash=""))
            nid += 1
    return decode_from_handoffs(synth)


_DUMMY_SCENARIO = SecurityScenario(
    scenario_id="__dummy__", description="", gold_severity="",
    gold_classification="", gold_containment="", gold_notify=(),
    causal_chain=(),
    per_role_events={r: () for r in ALL_ROLES})


# =============================================================================
# Prompt assembly
# =============================================================================


def build_auditor_prompt(scenario: SecurityScenario,
                          strategy: str,
                          events: Sequence[SecurityEvent],
                          handoffs: Sequence[TypedHandoff] = (),
                          substrate_cue: dict | None = None,
                          max_events_in_prompt: int = 200,
                          ) -> tuple[str, list[SecurityEvent], bool]:
    if strategy == STRATEGY_NAIVE:
        delivered = list(events)
    elif strategy == STRATEGY_ROUTING:
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
        "You are the CISO for a multi-team security-escalation "
        "review.",
        ("Produce exactly four lines:\n"
         "  SEVERITY: informational|low|medium|high|critical\n"
         "  CLASSIFICATION: <attack family>\n"
         "  CONTAINMENT: <canonical action>\n"
         "  NOTIFY: <comma-separated stakeholder list>"),
        "",
        f"SCENARIO: {scenario.description}",
    ]
    if strategy in (STRATEGY_SUBSTRATE, STRATEGY_SUBSTRATE_WRAP):
        if substrate_cue:
            lines.append("")
            lines.append("SUBSTRATE_ANSWER:")
            lines.append(f"  SEVERITY: {substrate_cue['severity']}")
            lines.append(
                f"  CLASSIFICATION: {substrate_cue['classification']}")
            lines.append(
                f"  CONTAINMENT: {substrate_cue['containment']}")
            lines.append(
                f"  NOTIFY: {','.join(substrate_cue['notify'])}")
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
    return "\n".join(lines), delivered, truncated


# =============================================================================
# Grader
# =============================================================================


_SEVERITY_RE = re.compile(r"SEVERITY\s*[:\-]?\s*([^\n]+)", re.IGNORECASE)
_CLASS_RE = re.compile(r"CLASSIFICATION\s*[:\-]?\s*([^\n]+)",
                        re.IGNORECASE)
_CONT_RE = re.compile(r"CONTAINMENT\s*[:\-]?\s*([^\n]+)", re.IGNORECASE)
_NOTIFY_RE = re.compile(r"NOTIFY\s*[:\-]?\s*([^\n]+)", re.IGNORECASE)


def parse_answer(text: str) -> dict[str, object]:
    sev = ""
    cls = ""
    cont = ""
    notify: tuple[str, ...] = ()
    m = _SEVERITY_RE.search(text)
    if m:
        sev = m.group(1).strip().lower()
        sev = re.sub(r"[^a-z0-9_]+", "_", sev).strip("_")
    m = _CLASS_RE.search(text)
    if m:
        cls = m.group(1).strip().lower()
        cls = re.sub(r"[^a-z0-9_]+", "_", cls).strip("_")
    m = _CONT_RE.search(text)
    if m:
        cont = m.group(1).strip().lower()
        cont = re.sub(r"[^a-z0-9_]+", "_", cont).strip("_")
    m = _NOTIFY_RE.search(text)
    if m:
        raw = m.group(1).strip()
        parts = [p.strip().lower()
                 for p in re.split(r"[,\s]+", raw) if p.strip()]
        notify = tuple(sorted(set(parts)))
    return {
        "severity": sev,
        "classification": cls,
        "containment": cont,
        "notify": notify,
    }


def grade_answer(scenario: SecurityScenario,
                 answer_text: str) -> dict[str, object]:
    parsed = parse_answer(answer_text)
    gold_sev = scenario.gold_severity
    gold_cls = scenario.gold_classification
    gold_cont = scenario.gold_containment
    gold_notify = set(scenario.gold_notify)
    sev_ok = parsed["severity"] == gold_sev
    cls_ok = parsed["classification"] == gold_cls
    cont_ok = parsed["containment"] == gold_cont
    nt_ok = set(parsed["notify"]) == gold_notify
    return {
        "severity_correct": sev_ok,
        "classification_correct": cls_ok,
        "containment_correct": cont_ok,
        "notify_correct": nt_ok,
        "full_correct": sev_ok and cls_ok and cont_ok and nt_ok,
        "parsed": parsed,
    }


# =============================================================================
# Failure attribution
# =============================================================================


FAILURE_NONE = "none"
FAILURE_TRUNCATION = "truncation"
FAILURE_MISSING_HANDOFF = "missing_handoff"
FAILURE_LLM_ERROR = "llm_error"
FAILURE_RETRIEVAL_MISS = "retrieval_miss"
FAILURE_SPURIOUS_CLAIM = "spurious_claim"


def attribute_failure(scenario: SecurityScenario, grading: dict,
                       strategy: str,
                       handoffs: Sequence[TypedHandoff],
                       truncated: bool,
                       ) -> str:
    if grading["full_correct"]:
        return FAILURE_NONE
    if truncated and strategy in (STRATEGY_NAIVE, STRATEGY_ROUTING):
        return FAILURE_TRUNCATION
    required = {(role, kind)
                for (role, kind, _p, _e) in scenario.causal_chain}
    seen = {(h.source_role, h.claim_kind) for h in handoffs}
    missing = required - seen
    if strategy in (STRATEGY_SUBSTRATE, STRATEGY_SUBSTRATE_WRAP):
        if missing:
            return FAILURE_MISSING_HANDOFF
        spurious = seen - required
        if spurious:
            return FAILURE_SPURIOUS_CLAIM
        return FAILURE_LLM_ERROR
    return FAILURE_RETRIEVAL_MISS


# =============================================================================
# Harness
# =============================================================================


@dataclass
class SecurityMeasurement:
    scenario_id: str
    strategy: str
    n_events_delivered: int
    n_handoffs_delivered: int
    n_prompt_chars: int
    n_prompt_tokens_approx: int
    truncated: bool
    n_events_total: int
    n_events_causal: int
    n_events_causal_to_auditor: int
    n_causal_events_delivered: int
    aggregator_relevance_fraction: float
    n_required_claims: int
    n_required_claims_delivered: int
    n_spurious_handoffs: int
    handoff_recall: float
    handoff_precision: float
    llm_answer: str
    grading: dict
    failure_kind: str
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
            "n_causal_events_delivered": self.n_causal_events_delivered,
            "aggregator_relevance_fraction": round(
                self.aggregator_relevance_fraction, 4),
            "n_required_claims": self.n_required_claims,
            "n_required_claims_delivered":
                self.n_required_claims_delivered,
            "n_spurious_handoffs": self.n_spurious_handoffs,
            "handoff_recall": round(self.handoff_recall, 4),
            "handoff_precision": round(self.handoff_precision, 4),
            "llm_answer": self.llm_answer[:500],
            "grading": {k: bool(v) for k, v in self.grading.items()
                        if k != "parsed"},
            "parsed": self.grading.get("parsed"),
            "failure_kind": self.failure_kind,
            "handoff_log_length": self.handoff_log_length,
            "handoff_chain_ok": self.handoff_chain_ok,
            "delivery_account": self.delivery_account,
            "wall_seconds": round(self.wall_seconds, 3),
        }


@dataclass
class SecurityReport:
    scenario_ids: tuple[str, ...]
    strategies: tuple[str, ...]
    measurements: list[SecurityMeasurement]
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
            sev = sum(1 for m in ms if m.grading["severity_correct"])
            cls = sum(1 for m in ms
                      if m.grading["classification_correct"])
            cont = sum(1 for m in ms
                       if m.grading["containment_correct"])
            nt = sum(1 for m in ms if m.grading["notify_correct"])
            mean_tok = sum(m.n_prompt_tokens_approx for m in ms) / n
            trunc = sum(1 for m in ms if m.truncated)
            mean_rel = sum(m.aggregator_relevance_fraction
                            for m in ms) / n
            mean_recall = sum(m.handoff_recall for m in ms) / n
            mean_prec = sum(m.handoff_precision for m in ms) / n
            f_hist: dict[str, int] = {}
            for m in ms:
                f_hist[m.failure_kind] = f_hist.get(m.failure_kind,
                                                     0) + 1
            out[strat] = {
                "n": n,
                "accuracy_full": round(correct / n, 4),
                "accuracy_severity": round(sev / n, 4),
                "accuracy_classification": round(cls / n, 4),
                "accuracy_containment": round(cont / n, 4),
                "accuracy_notify": round(nt / n, 4),
                "mean_prompt_tokens": round(mean_tok, 2),
                "truncated_count": trunc,
                "mean_aggregator_relevance_fraction": round(
                    mean_rel, 4),
                "mean_handoff_recall": round(mean_recall, 4),
                "mean_handoff_precision": round(mean_prec, 4),
                "failure_hist": f_hist,
            }
        return out


def run_security_loop(
        scenarios: Sequence[SecurityScenario],
        auditor: Callable[[str], str],
        strategies: Sequence[str] = ALL_STRATEGIES,
        seed: int = 33,
        max_events_in_prompt: int = 200,
        inbox_capacity: int = 64,
        extractor: Callable[
            [str, Sequence[SecurityEvent], SecurityScenario],
            list[tuple[str, str, tuple[int, ...]]],
        ] | None = None,
        ) -> SecurityReport:
    import time as _time
    measurements: list[SecurityMeasurement] = []

    for scenario in scenarios:
        events = naive_event_stream(scenario)
        n_total = len(events)
        n_causal = sum(1 for ev in events if ev.is_causal)
        n_causal_to_auditor = sum(
            1 for ev in events
            if oracle_relevance(ev, ROLE_CISO, scenario))
        required = {(role, kind)
                    for (role, kind, _p, _e) in scenario.causal_chain}

        router = run_handoff_protocol(
            scenario, max_events_per_role=max_events_in_prompt,
            inbox_capacity=inbox_capacity, extractor=extractor)
        inbox = router.inboxes.get(ROLE_CISO)
        auditor_handoffs = tuple(inbox.peek()) if inbox else ()
        cue = decode_from_handoffs(auditor_handoffs)

        seen_pairs = {(h.source_role, h.claim_kind)
                      for h in auditor_handoffs}
        required_delivered = len(required & seen_pairs)
        recall = required_delivered / max(1, len(required))
        n_spurious = sum(1 for h in auditor_handoffs
                          if not handoff_is_relevant(h, scenario))
        precision = ((len(auditor_handoffs) - n_spurious)
                     / max(1, len(auditor_handoffs)))

        for strat in strategies:
            if strat in (STRATEGY_SUBSTRATE, STRATEGY_SUBSTRATE_WRAP):
                delivered_handoffs = auditor_handoffs
            else:
                delivered_handoffs = ()
            prompt, delivered, truncated = build_auditor_prompt(
                scenario, strat, events,
                handoffs=delivered_handoffs,
                substrate_cue=cue if strat in (
                    STRATEGY_SUBSTRATE, STRATEGY_SUBSTRATE_WRAP) else None,
                max_events_in_prompt=max_events_in_prompt,
            )
            delivered_causal = sum(
                1 for ev in delivered
                if oracle_relevance(ev, ROLE_CISO, scenario))
            rel = (delivered_causal / max(1, len(delivered))
                    if strat in (STRATEGY_NAIVE, STRATEGY_ROUTING)
                    else (sum(1 for h in delivered_handoffs
                              if handoff_is_relevant(h, scenario))
                          / max(1, len(delivered_handoffs))))
            t0 = _time.time()
            ans = auditor(prompt)
            wall = _time.time() - t0
            g = grade_answer(scenario, ans)
            fk = attribute_failure(scenario, g, strat,
                                    delivered_handoffs, truncated)
            measurements.append(SecurityMeasurement(
                scenario_id=scenario.scenario_id, strategy=strat,
                n_events_delivered=len(delivered),
                n_handoffs_delivered=len(delivered_handoffs),
                n_prompt_chars=len(prompt),
                n_prompt_tokens_approx=max(1, len(prompt) // 4),
                truncated=truncated,
                n_events_total=n_total,
                n_events_causal=n_causal,
                n_events_causal_to_auditor=n_causal_to_auditor,
                n_causal_events_delivered=delivered_causal,
                aggregator_relevance_fraction=rel,
                n_required_claims=len(required),
                n_required_claims_delivered=required_delivered,
                n_spurious_handoffs=n_spurious,
                handoff_recall=recall,
                handoff_precision=precision,
                llm_answer=ans, grading=g, failure_kind=fk,
                handoff_log_length=router.log_length(),
                handoff_chain_ok=router.verify(),
                delivery_account=router.account.summary(),
                wall_seconds=wall,
            ))

    return SecurityReport(
        scenario_ids=tuple(s.scenario_id for s in scenarios),
        strategies=tuple(strategies),
        measurements=measurements,
        config={"seed": seed,
                "max_events_in_prompt": max_events_in_prompt,
                "inbox_capacity": inbox_capacity},
    )


# =============================================================================
# Mock auditor
# =============================================================================


class MockSecurityAuditor:
    """Deterministic reader of the delivered prompt.

    Same shape as ``MockIncidentAuditor`` / ``MockComplianceAuditor``:
    if the prompt has a ``SUBSTRATE_ANSWER`` block, copy it; else
    run the per-role extractors on DELIVERED EVENTS and emit the
    decoded structure; else return UNKNOWN.
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
            r"SUBSTRATE_ANSWER:\s*\n\s*SEVERITY:\s*([^\n]+)\s*\n"
            r"\s*CLASSIFICATION:\s*([^\n]+)\s*\n"
            r"\s*CONTAINMENT:\s*([^\n]+)\s*\n"
            r"\s*NOTIFY:\s*([^\n]+)",
            prompt, re.IGNORECASE)
        if m:
            sev = m.group(1).strip()
            cls = m.group(2).strip()
            cont = m.group(3).strip()
            nt = m.group(4).strip()
            self.last_answer = (f"SEVERITY: {sev}\n"
                                f"CLASSIFICATION: {cls}\n"
                                f"CONTAINMENT: {cont}\n"
                                f"NOTIFY: {nt}\n")
            return self.last_answer
        lines = prompt.splitlines()
        events: list[SecurityEvent] = []
        nid = 0
        collecting = False
        for line in lines:
            s = line.strip()
            if s.startswith("DELIVERED EVENTS:"):
                collecting = True
                continue
            if s.startswith("ANSWER:"):
                break
            if not collecting:
                continue
            m2 = re.match(
                r"^-\s*\[([A-Z_]+)\s+by\s+([\w_]+)\]\s+(.+)$", s)
            if not m2:
                continue
            et = m2.group(1).strip()
            role = m2.group(2).strip()
            body = m2.group(3).strip()
            events.append(SecurityEvent(
                event_id=nid, event_type=et, origin_role=role,
                body=body))
            nid += 1
        dec = _decoder_from_events(events) if events else {
            "severity": "unknown",
            "classification": "unclassified",
            "containment": "investigate", "notify": ()}
        nt_str = ",".join(dec["notify"])
        self.last_answer = (f"SEVERITY: {dec['severity']}\n"
                            f"CLASSIFICATION: {dec['classification']}\n"
                            f"CONTAINMENT: {dec['containment']}\n"
                            f"NOTIFY: {nt_str}\n")
        return self.last_answer
