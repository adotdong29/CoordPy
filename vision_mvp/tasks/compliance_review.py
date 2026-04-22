"""Phase 32 — multi-role vendor-onboarding compliance-review benchmark.

The second non-code task-scale benchmark in the programme, and the first
one whose subject is *document review* rather than operational telemetry.
A team of five role-typed agents — legal, security, privacy, finance, and
a compliance officer (the aggregator) — evaluates a new vendor onboarding
request. Each role inspects a *different* class of document:

  * ``legal``        — contract clauses (MSA, DPA, termination, renewal,
    liability cap).
  * ``security``     — security-questionnaire answers (encryption, SSO,
    incident-response SLA, penetration test vintage).
  * ``privacy``      — data-processing inventory (PII categories,
    retention schedule, cross-border transfers).
  * ``finance``      — spend approval (amount, budget authority, payment
    terms).
  * ``compliance_officer`` — the auditor; receives role handoffs and
    emits the structured final verdict.

Why this task family (and why it is not a repo knowledge graph)

  * Each role owns a *different* slice of evidence; no role has enough
    information alone. A missing DPA (privacy) with uncapped liability
    (legal) is a compound blocker that no single role will flag.
  * The aggregator's output depends on cross-role *communication*: it
    is a structured verdict + flag list + remediation, not a summary.
  * The task is deliberately NOT a code corpus and NOT an index
    traversal. It is the master-plan § 1.5 differentiation made
    operational on a second domain: typed handoffs between roles are
    the mechanism, and the substrate's correctness is a property of
    the *subscription table + extractor set*, not of any corpus.
  * The scenario catalogue is a small, hand-crafted set of *compound*
    compliance issues (missing DPA; uncapped liability + aggressive
    terms; weak encryption + cross-border transfer; payment-terms
    escalation; pen-test stale + no incident SLA). Each scenario
    chooses a *verdict* (approved / conditional / blocked), a set of
    flags that must be raised, and a single remediation.

Scope discipline (what this module does NOT claim)

  * Not a real compliance engine. Real vendor review involves legal
    judgement, contextual negotiation, and opinion. We model only the
    slice where a structured multi-role team must produce a
    structured answer from role-partitioned evidence.
  * Not an adversarial-scenarios benchmark. The five scenarios are
    structurally typed by construction (Theorem P32-1); a scenario
    whose answer requires integrating information across arbitrary
    role combinations without typed claims falls outside the covered
    regime.
  * Not a replacement for ``incident_triage``. It reuses the same
    substrate primitive (``role_handoff``) and the same harness
    shape; the two benchmarks together are the *evidence* that the
    substrate generalises across non-code domains (Phase 32 Part A).

Theoretical anchor: RESULTS_PHASE32.md § B (Theorems P32-1, P32-2).
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


ROLE_LEGAL = "legal"
ROLE_SECURITY = "security"
ROLE_PRIVACY = "privacy"
ROLE_FINANCE = "finance"
ROLE_COMPLIANCE = "compliance_officer"

ALL_ROLES = (ROLE_LEGAL, ROLE_SECURITY, ROLE_PRIVACY,
             ROLE_FINANCE, ROLE_COMPLIANCE)


# =============================================================================
# Document (event) schema
# =============================================================================


# Raw document kinds — every producer role has its own subset.
DOC_CONTRACT_CLAUSE = "CONTRACT_CLAUSE"
DOC_SECURITY_QA = "SECURITY_QUESTIONNAIRE"
DOC_PRIVACY_INVENTORY = "PRIVACY_INVENTORY"
DOC_FINANCE_LINEITEM = "FINANCE_LINE_ITEM"
DOC_TASK_GOAL = "TASK_GOAL"
DOC_FINAL_ANSWER = "FINAL_ANSWER"

FIXED_POINT_TYPES = frozenset({DOC_TASK_GOAL, DOC_FINAL_ANSWER})

ROLE_OBSERVABLE_TYPES: dict[str, frozenset[str]] = {
    ROLE_LEGAL: frozenset({DOC_CONTRACT_CLAUSE} | FIXED_POINT_TYPES),
    ROLE_SECURITY: frozenset({DOC_SECURITY_QA} | FIXED_POINT_TYPES),
    ROLE_PRIVACY: frozenset({DOC_PRIVACY_INVENTORY} | FIXED_POINT_TYPES),
    ROLE_FINANCE: frozenset({DOC_FINANCE_LINEITEM} | FIXED_POINT_TYPES),
    ROLE_COMPLIANCE: frozenset(FIXED_POINT_TYPES),
}


@dataclass(frozen=True)
class VendorDoc:
    """One document observed by one role.

    ``origin_role`` is the role that owns the document class; a naive
    broadcast hands every document to every role, a role-keyed routing
    passes only its own observable types, and typed handoffs lift
    claims out of the body.
    """

    doc_id: int
    doc_type: str
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
        return self.doc_type in FIXED_POINT_TYPES


# =============================================================================
# Claim taxonomy
# =============================================================================


# Claim kinds — short, enumerated, subscription-granularity.
# Legal
CLAIM_LIABILITY_CAP_MISSING = "LIABILITY_CAP_MISSING"
CLAIM_AUTO_RENEWAL_UNFAVOURABLE = "AUTO_RENEWAL_UNFAVOURABLE"
CLAIM_TERMINATION_RESTRICTIVE = "TERMINATION_RESTRICTIVE"

# Security
CLAIM_ENCRYPTION_AT_REST_MISSING = "ENCRYPTION_AT_REST_MISSING"
CLAIM_SSO_NOT_SUPPORTED = "SSO_NOT_SUPPORTED"
CLAIM_PENTEST_STALE = "PENTEST_STALE"
CLAIM_INCIDENT_SLA_INADEQUATE = "INCIDENT_SLA_INADEQUATE"

# Privacy
CLAIM_DPA_MISSING = "DPA_MISSING"
CLAIM_CROSS_BORDER_UNAUTHORIZED = "CROSS_BORDER_UNAUTHORIZED"
CLAIM_RETENTION_UNCAPPED = "RETENTION_UNCAPPED"
CLAIM_PII_CATEGORY_UNDISCLOSED = "PII_CATEGORY_UNDISCLOSED"

# Finance
CLAIM_BUDGET_THRESHOLD_BREACH = "BUDGET_THRESHOLD_BREACH"
CLAIM_PAYMENT_TERMS_AGGRESSIVE = "PAYMENT_TERMS_AGGRESSIVE"

ALL_CLAIMS = (
    CLAIM_LIABILITY_CAP_MISSING, CLAIM_AUTO_RENEWAL_UNFAVOURABLE,
    CLAIM_TERMINATION_RESTRICTIVE,
    CLAIM_ENCRYPTION_AT_REST_MISSING, CLAIM_SSO_NOT_SUPPORTED,
    CLAIM_PENTEST_STALE, CLAIM_INCIDENT_SLA_INADEQUATE,
    CLAIM_DPA_MISSING, CLAIM_CROSS_BORDER_UNAUTHORIZED,
    CLAIM_RETENTION_UNCAPPED, CLAIM_PII_CATEGORY_UNDISCLOSED,
    CLAIM_BUDGET_THRESHOLD_BREACH, CLAIM_PAYMENT_TERMS_AGGRESSIVE,
)


def build_role_subscriptions() -> RoleSubscriptionTable:
    """The default "who should know what" declaration for the vendor-
    onboarding review. Every claim reaches the compliance_officer;
    the most severe claims (DPA missing, cross-border unauthorized,
    encryption at rest missing) also flow laterally to the *privacy*
    role so it can escalate. Budget breaches also reach legal (because
    spend threshold affects contract signing authority).
    """
    subs = RoleSubscriptionTable()
    # Legal → compliance
    subs.subscribe(ROLE_LEGAL, CLAIM_LIABILITY_CAP_MISSING,
                   [ROLE_COMPLIANCE])
    subs.subscribe(ROLE_LEGAL, CLAIM_AUTO_RENEWAL_UNFAVOURABLE,
                   [ROLE_COMPLIANCE])
    subs.subscribe(ROLE_LEGAL, CLAIM_TERMINATION_RESTRICTIVE,
                   [ROLE_COMPLIANCE])
    # Security → compliance; encryption issue also cc privacy.
    subs.subscribe(ROLE_SECURITY, CLAIM_ENCRYPTION_AT_REST_MISSING,
                   [ROLE_COMPLIANCE, ROLE_PRIVACY])
    subs.subscribe(ROLE_SECURITY, CLAIM_SSO_NOT_SUPPORTED,
                   [ROLE_COMPLIANCE])
    subs.subscribe(ROLE_SECURITY, CLAIM_PENTEST_STALE,
                   [ROLE_COMPLIANCE])
    subs.subscribe(ROLE_SECURITY, CLAIM_INCIDENT_SLA_INADEQUATE,
                   [ROLE_COMPLIANCE])
    # Privacy → compliance; DPA-missing also cc legal (needs new DPA
    # drafted); cross-border also cc legal.
    subs.subscribe(ROLE_PRIVACY, CLAIM_DPA_MISSING,
                   [ROLE_COMPLIANCE, ROLE_LEGAL])
    subs.subscribe(ROLE_PRIVACY, CLAIM_CROSS_BORDER_UNAUTHORIZED,
                   [ROLE_COMPLIANCE, ROLE_LEGAL])
    subs.subscribe(ROLE_PRIVACY, CLAIM_RETENTION_UNCAPPED,
                   [ROLE_COMPLIANCE])
    subs.subscribe(ROLE_PRIVACY, CLAIM_PII_CATEGORY_UNDISCLOSED,
                   [ROLE_COMPLIANCE])
    # Finance → compliance; budget breach also cc legal.
    subs.subscribe(ROLE_FINANCE, CLAIM_BUDGET_THRESHOLD_BREACH,
                   [ROLE_COMPLIANCE, ROLE_LEGAL])
    subs.subscribe(ROLE_FINANCE, CLAIM_PAYMENT_TERMS_AGGRESSIVE,
                   [ROLE_COMPLIANCE])
    return subs


# =============================================================================
# Scenario catalogue
# =============================================================================


VERDICT_APPROVED = "approved"
VERDICT_CONDITIONAL = "conditional"
VERDICT_BLOCKED = "blocked"


@dataclass(frozen=True)
class VendorScenario:
    """One deterministic vendor-onboarding scenario.

    * ``gold_verdict``    — ``approved`` / ``conditional`` / ``blocked``.
    * ``gold_flags``      — sorted tuple of flag labels (a canonical
      lowercase form of the load-bearing claim kinds).
    * ``gold_remediation`` — canonical remediation string.
    * ``causal_chain``    — (role, claim_kind, payload, evids) tuples.
    * ``per_role_docs``   — role → tuple of VendorDoc events.
    """

    scenario_id: str
    description: str
    gold_verdict: str
    gold_flags: tuple[str, ...]
    gold_remediation: str
    causal_chain: tuple[tuple[str, str, str, tuple[int, ...]], ...]
    per_role_docs: dict[str, tuple[VendorDoc, ...]]


def _mk(nid: int, dt: str, role: str, body: str, *,
        tags: Sequence[str] = (), causal: bool = False,
        ) -> tuple[VendorDoc, int]:
    d = VendorDoc(doc_id=nid, doc_type=dt, origin_role=role,
                  body=body, tags=tuple(tags), is_causal=causal)
    return d, nid + 1


def _distractors(rng: random.Random, nid: int, role: str,
                  kinds: Sequence[str], k: int,
                  ) -> tuple[list[VendorDoc], int]:
    """Benign role-specific chatter."""
    benign = {
        DOC_CONTRACT_CLAUSE: [
            "definitions section vendor means the party providing services",
            "notices shall be given in writing to the address of record",
            "governing law state of delaware",
            "force majeure excusable delay for acts of god",
            "assignment requires written consent",
            "entire agreement supersedes prior negotiations",
            "severability invalid clauses do not void the agreement",
        ],
        DOC_SECURITY_QA: [
            "password_policy min_length=14 rotation_days=90",
            "logging aggregated to siem retention=180d",
            "vulnerability management patching sla=30d",
            "backup frequency daily retention=30d",
            "network segmentation production isolated from corp",
            "dev_tooling secrets stored in vault",
        ],
        DOC_PRIVACY_INVENTORY: [
            "data_category=service_metadata retention=90d pii=no",
            "audit_log export format=json retention=365d",
            "employee_contact retention=employment_plus_2y pii=yes",
            "data_subject_access_sla days=30",
        ],
        DOC_FINANCE_LINEITEM: [
            "budget_line=cloud_infra code=IT-042 amount_usd=12000",
            "existing_vendor yes code=VEN-120 amount_usd=8000",
            "procurement_card approved amount_usd=500",
            "expense_report receipt attached amount_usd=240",
        ],
    }
    out: list[VendorDoc] = []
    for _ in range(k):
        dt = rng.choice(list(kinds))
        body = rng.choice(benign.get(dt, ["benign"]))
        d, nid = _mk(nid, dt, role, body, causal=False)
        out.append(d)
    return out, nid


# -----------------------------------------------------------------------------
# Five scenarios
# -----------------------------------------------------------------------------


def make_scenario_missing_dpa(rng: random.Random, start_id: int = 0,
                               distractors_per_role: int = 6,
                               ) -> VendorScenario:
    """Privacy DPA missing → BLOCKED, remediate = require_signed_dpa."""
    nid = start_id
    per_role: dict[str, list[VendorDoc]] = {r: [] for r in ALL_ROLES}
    chain: list[tuple[str, str, str, tuple[int, ...]]] = []

    d, nid = _mk(nid, DOC_PRIVACY_INVENTORY, ROLE_PRIVACY,
                  "data_processing_agreement=missing vendor=newdataco",
                  tags=("newdataco",), causal=True)
    per_role[ROLE_PRIVACY].append(d)
    dpa_id = d.doc_id
    d, nid = _mk(nid, DOC_PRIVACY_INVENTORY, ROLE_PRIVACY,
                  ("pii_category=user_email retention=unbounded "
                   "vendor=newdataco"),
                  tags=("newdataco",), causal=True)
    per_role[ROLE_PRIVACY].append(d)
    ret_id = d.doc_id
    chain.append((ROLE_PRIVACY, CLAIM_DPA_MISSING,
                  "data_processing_agreement=missing vendor=newdataco",
                  (dpa_id,)))
    chain.append((ROLE_PRIVACY, CLAIM_RETENTION_UNCAPPED,
                  "retention=unbounded vendor=newdataco",
                  (ret_id,)))

    for role, kinds in ((ROLE_LEGAL, [DOC_CONTRACT_CLAUSE]),
                         (ROLE_SECURITY, [DOC_SECURITY_QA]),
                         (ROLE_PRIVACY, [DOC_PRIVACY_INVENTORY]),
                         (ROLE_FINANCE, [DOC_FINANCE_LINEITEM])):
        dist, nid = _distractors(rng, nid, role, kinds,
                                   k=distractors_per_role)
        per_role[role].extend(dist)

    return VendorScenario(
        scenario_id="missing_dpa",
        description=("vendor onboarding: privacy review finds no DPA "
                      "on file and an uncapped retention for PII "
                      "category user_email"),
        gold_verdict=VERDICT_BLOCKED,
        gold_flags=("dpa_missing", "retention_uncapped"),
        gold_remediation="require_signed_dpa_and_cap_retention",
        causal_chain=tuple(chain),
        per_role_docs={r: tuple(v) for r, v in per_role.items()},
    )


def make_scenario_uncapped_liability(rng: random.Random,
                                       start_id: int = 0,
                                       distractors_per_role: int = 6,
                                       ) -> VendorScenario:
    """Legal: liability cap missing + auto-renewal unfavourable →
    CONDITIONAL, remediate = negotiate_liability_cap."""
    nid = start_id
    per_role: dict[str, list[VendorDoc]] = {r: [] for r in ALL_ROLES}
    chain: list[tuple[str, str, str, tuple[int, ...]]] = []

    d, nid = _mk(nid, DOC_CONTRACT_CLAUSE, ROLE_LEGAL,
                  "liability clause limits=none vendor=bigcontractor",
                  tags=("bigcontractor",), causal=True)
    per_role[ROLE_LEGAL].append(d)
    lib_id = d.doc_id
    d, nid = _mk(nid, DOC_CONTRACT_CLAUSE, ROLE_LEGAL,
                  ("auto_renewal term=36mo notice_window_days=7 "
                   "vendor=bigcontractor"),
                  tags=("bigcontractor",), causal=True)
    per_role[ROLE_LEGAL].append(d)
    ren_id = d.doc_id
    chain.append((ROLE_LEGAL, CLAIM_LIABILITY_CAP_MISSING,
                  "liability limits=none vendor=bigcontractor",
                  (lib_id,)))
    chain.append((ROLE_LEGAL, CLAIM_AUTO_RENEWAL_UNFAVOURABLE,
                  "auto_renewal term=36mo notice_window_days=7",
                  (ren_id,)))

    for role, kinds in ((ROLE_LEGAL, [DOC_CONTRACT_CLAUSE]),
                         (ROLE_SECURITY, [DOC_SECURITY_QA]),
                         (ROLE_PRIVACY, [DOC_PRIVACY_INVENTORY]),
                         (ROLE_FINANCE, [DOC_FINANCE_LINEITEM])):
        dist, nid = _distractors(rng, nid, role, kinds,
                                   k=distractors_per_role)
        per_role[role].extend(dist)

    return VendorScenario(
        scenario_id="uncapped_liability",
        description=("vendor onboarding: MSA liability uncapped and "
                      "auto-renewal term is 36 months with a 7-day "
                      "notice window"),
        gold_verdict=VERDICT_CONDITIONAL,
        gold_flags=("auto_renewal_unfavourable", "liability_cap_missing"),
        gold_remediation="negotiate_liability_cap_and_renewal",
        causal_chain=tuple(chain),
        per_role_docs={r: tuple(v) for r, v in per_role.items()},
    )


def make_scenario_weak_encryption(rng: random.Random,
                                    start_id: int = 0,
                                    distractors_per_role: int = 6,
                                    ) -> VendorScenario:
    """Security: encryption-at-rest missing + pentest stale → BLOCKED."""
    nid = start_id
    per_role: dict[str, list[VendorDoc]] = {r: [] for r in ALL_ROLES}
    chain: list[tuple[str, str, str, tuple[int, ...]]] = []

    d, nid = _mk(nid, DOC_SECURITY_QA, ROLE_SECURITY,
                  ("encryption_at_rest=no service=datastore "
                   "vendor=oldcloud"),
                  tags=("oldcloud",), causal=True)
    per_role[ROLE_SECURITY].append(d)
    enc_id = d.doc_id
    d, nid = _mk(nid, DOC_SECURITY_QA, ROLE_SECURITY,
                  ("pentest_report vintage_days=900 vendor=oldcloud"),
                  tags=("oldcloud",), causal=True)
    per_role[ROLE_SECURITY].append(d)
    pen_id = d.doc_id
    chain.append((ROLE_SECURITY, CLAIM_ENCRYPTION_AT_REST_MISSING,
                  "encryption_at_rest=no service=datastore",
                  (enc_id,)))
    chain.append((ROLE_SECURITY, CLAIM_PENTEST_STALE,
                  "pentest_report vintage_days=900", (pen_id,)))

    for role, kinds in ((ROLE_LEGAL, [DOC_CONTRACT_CLAUSE]),
                         (ROLE_SECURITY, [DOC_SECURITY_QA]),
                         (ROLE_PRIVACY, [DOC_PRIVACY_INVENTORY]),
                         (ROLE_FINANCE, [DOC_FINANCE_LINEITEM])):
        dist, nid = _distractors(rng, nid, role, kinds,
                                   k=distractors_per_role)
        per_role[role].extend(dist)

    return VendorScenario(
        scenario_id="weak_encryption",
        description=("vendor onboarding: security questionnaire reports "
                      "no encryption-at-rest and the most recent "
                      "pentest is 900 days old"),
        gold_verdict=VERDICT_BLOCKED,
        gold_flags=("encryption_at_rest_missing", "pentest_stale"),
        gold_remediation="require_encryption_and_fresh_pentest",
        causal_chain=tuple(chain),
        per_role_docs={r: tuple(v) for r, v in per_role.items()},
    )


def make_scenario_cross_border(rng: random.Random,
                                 start_id: int = 0,
                                 distractors_per_role: int = 6,
                                 ) -> VendorScenario:
    """Privacy: cross-border PII transfer unauthorized +
    encryption at rest missing → BLOCKED."""
    nid = start_id
    per_role: dict[str, list[VendorDoc]] = {r: [] for r in ALL_ROLES}
    chain: list[tuple[str, str, str, tuple[int, ...]]] = []

    d, nid = _mk(nid, DOC_PRIVACY_INVENTORY, ROLE_PRIVACY,
                  ("cross_border_transfer=yes region_from=eu "
                   "region_to=third_country sccs=no vendor=globo"),
                  tags=("globo",), causal=True)
    per_role[ROLE_PRIVACY].append(d)
    cb_id = d.doc_id
    d, nid = _mk(nid, DOC_SECURITY_QA, ROLE_SECURITY,
                  ("encryption_at_rest=no service=analytics "
                   "vendor=globo"),
                  tags=("globo",), causal=True)
    per_role[ROLE_SECURITY].append(d)
    enc_id = d.doc_id
    chain.append((ROLE_PRIVACY, CLAIM_CROSS_BORDER_UNAUTHORIZED,
                  "cross_border_transfer=yes sccs=no", (cb_id,)))
    chain.append((ROLE_SECURITY, CLAIM_ENCRYPTION_AT_REST_MISSING,
                  "encryption_at_rest=no service=analytics", (enc_id,)))

    for role, kinds in ((ROLE_LEGAL, [DOC_CONTRACT_CLAUSE]),
                         (ROLE_SECURITY, [DOC_SECURITY_QA]),
                         (ROLE_PRIVACY, [DOC_PRIVACY_INVENTORY]),
                         (ROLE_FINANCE, [DOC_FINANCE_LINEITEM])):
        dist, nid = _distractors(rng, nid, role, kinds,
                                   k=distractors_per_role)
        per_role[role].extend(dist)

    return VendorScenario(
        scenario_id="cross_border_transfer_unauthorized",
        description=("vendor onboarding: PII flows EU → third country "
                      "with no SCCs, and analytics data store has no "
                      "encryption at rest"),
        gold_verdict=VERDICT_BLOCKED,
        gold_flags=("cross_border_unauthorized",
                    "encryption_at_rest_missing"),
        gold_remediation="require_sccs_and_encryption_at_rest",
        causal_chain=tuple(chain),
        per_role_docs={r: tuple(v) for r, v in per_role.items()},
    )


def make_scenario_budget_breach(rng: random.Random,
                                 start_id: int = 0,
                                 distractors_per_role: int = 6,
                                 ) -> VendorScenario:
    """Finance: spend above authority + payment-terms aggressive →
    CONDITIONAL."""
    nid = start_id
    per_role: dict[str, list[VendorDoc]] = {r: [] for r in ALL_ROLES}
    chain: list[tuple[str, str, str, tuple[int, ...]]] = []

    d, nid = _mk(nid, DOC_FINANCE_LINEITEM, ROLE_FINANCE,
                  ("proposed_spend amount_usd=420000 "
                   "budget_authority_cap_usd=250000 vendor=bigdatavendor"),
                  tags=("bigdatavendor",), causal=True)
    per_role[ROLE_FINANCE].append(d)
    sp_id = d.doc_id
    d, nid = _mk(nid, DOC_FINANCE_LINEITEM, ROLE_FINANCE,
                  ("payment_terms net_days=7 prepay_pct=50 "
                   "vendor=bigdatavendor"),
                  tags=("bigdatavendor",), causal=True)
    per_role[ROLE_FINANCE].append(d)
    pt_id = d.doc_id
    chain.append((ROLE_FINANCE, CLAIM_BUDGET_THRESHOLD_BREACH,
                  "proposed_spend amount_usd=420000 cap_usd=250000",
                  (sp_id,)))
    chain.append((ROLE_FINANCE, CLAIM_PAYMENT_TERMS_AGGRESSIVE,
                  "payment_terms net_days=7 prepay_pct=50", (pt_id,)))

    for role, kinds in ((ROLE_LEGAL, [DOC_CONTRACT_CLAUSE]),
                         (ROLE_SECURITY, [DOC_SECURITY_QA]),
                         (ROLE_PRIVACY, [DOC_PRIVACY_INVENTORY]),
                         (ROLE_FINANCE, [DOC_FINANCE_LINEITEM])):
        dist, nid = _distractors(rng, nid, role, kinds,
                                   k=distractors_per_role)
        per_role[role].extend(dist)

    return VendorScenario(
        scenario_id="budget_threshold_breach",
        description=("vendor onboarding: proposed spend 420k exceeds "
                      "the requester's authority cap of 250k and the "
                      "payment terms require 50%% prepay net-7"),
        gold_verdict=VERDICT_CONDITIONAL,
        gold_flags=("budget_threshold_breach",
                    "payment_terms_aggressive"),
        gold_remediation="require_cfo_approval_and_renegotiate_terms",
        causal_chain=tuple(chain),
        per_role_docs={r: tuple(v) for r, v in per_role.items()},
    )


SCENARIO_BUILDERS: tuple[
    Callable[..., VendorScenario], ...] = (
    make_scenario_missing_dpa,
    make_scenario_uncapped_liability,
    make_scenario_weak_encryption,
    make_scenario_cross_border,
    make_scenario_budget_breach,
)


def build_scenario_bank(seed: int = 32,
                         distractors_per_role: int = 6,
                         ) -> list[VendorScenario]:
    rng = random.Random(seed)
    out: list[VendorScenario] = []
    for builder in SCENARIO_BUILDERS:
        s = builder(rng, 0, distractors_per_role=distractors_per_role)
        out.append(s)
    return out


# =============================================================================
# Global document-stream assembly
# =============================================================================


def fixed_point_docs(scenario: VendorScenario) -> list[VendorDoc]:
    goal = VendorDoc(
        doc_id=-1, doc_type=DOC_TASK_GOAL, origin_role="__system__",
        body=(f"[task] vendor-onboarding review {scenario.scenario_id}: "
              f"decide verdict, list flags, name remediation"))
    final = VendorDoc(
        doc_id=-2, doc_type=DOC_FINAL_ANSWER, origin_role="__system__",
        body="[final] <placeholder>")
    return [goal, final]


def naive_doc_stream(scenario: VendorScenario) -> list[VendorDoc]:
    out = list(fixed_point_docs(scenario))
    for role in ALL_ROLES:
        out.extend(scenario.per_role_docs.get(role, ()))
    return out


# =============================================================================
# Oracle (per-(doc, role) relevance)
# =============================================================================


def oracle_relevance(doc: VendorDoc, role: str,
                     scenario: VendorScenario) -> bool:
    if doc.is_fixed_point:
        return True
    obs = ROLE_OBSERVABLE_TYPES.get(role, frozenset())
    if role == ROLE_COMPLIANCE:
        return bool(doc.is_causal)
    if doc.doc_type not in obs:
        return False
    return bool(doc.is_causal) and doc.origin_role == role


def handoff_is_relevant(h: TypedHandoff,
                         scenario: VendorScenario) -> bool:
    pairs = {(role, kind)
             for (role, kind, _p, _e) in scenario.causal_chain}
    return (h.source_role, h.claim_kind) in pairs


# =============================================================================
# Extractors (deterministic, per-role)
# =============================================================================


def extract_claims_for_role(role: str,
                             docs: Sequence[VendorDoc],
                             scenario: VendorScenario,
                             ) -> list[tuple[str, str, tuple[int, ...]]]:
    """Regex-based extractors per role.

    Precision on causal events is 1 by construction; recall on the
    scenario's causal chain is 1. Distractors are filtered by regex.
    """
    out: list[tuple[str, str, tuple[int, ...]]] = []

    def _emit(kind: str, body: str, evids: Sequence[int]) -> None:
        out.append((kind, body, tuple(evids)))

    if role == ROLE_LEGAL:
        for d in docs:
            if d.doc_type != DOC_CONTRACT_CLAUSE:
                continue
            if re.search(r"liability.*limits=none", d.body):
                _emit(CLAIM_LIABILITY_CAP_MISSING, d.body,
                      [d.doc_id])
            if re.search(r"auto_renewal.*term=\d+mo.*notice_window_days=\d",
                          d.body):
                m = re.search(r"notice_window_days=(\d+)", d.body)
                term_m = re.search(r"term=(\d+)mo", d.body)
                if m and term_m and (int(m.group(1)) < 30
                                      or int(term_m.group(1)) >= 24):
                    _emit(CLAIM_AUTO_RENEWAL_UNFAVOURABLE, d.body,
                          [d.doc_id])
            if re.search(r"termination.*notice=\d+mo", d.body):
                m = re.search(r"notice=(\d+)mo", d.body)
                if m and int(m.group(1)) >= 12:
                    _emit(CLAIM_TERMINATION_RESTRICTIVE, d.body,
                          [d.doc_id])
    elif role == ROLE_SECURITY:
        for d in docs:
            if d.doc_type != DOC_SECURITY_QA:
                continue
            if re.search(r"encryption_at_rest=no", d.body):
                _emit(CLAIM_ENCRYPTION_AT_REST_MISSING, d.body,
                      [d.doc_id])
            if re.search(r"\bsso=no\b", d.body):
                _emit(CLAIM_SSO_NOT_SUPPORTED, d.body, [d.doc_id])
            m = re.search(r"pentest.*vintage_days=(\d+)", d.body)
            if m and int(m.group(1)) >= 365:
                _emit(CLAIM_PENTEST_STALE, d.body, [d.doc_id])
            m = re.search(r"incident_sla_hours=(\d+)", d.body)
            if m and int(m.group(1)) >= 72:
                _emit(CLAIM_INCIDENT_SLA_INADEQUATE, d.body,
                      [d.doc_id])
    elif role == ROLE_PRIVACY:
        for d in docs:
            if d.doc_type != DOC_PRIVACY_INVENTORY:
                continue
            if re.search(r"data_processing_agreement=missing", d.body):
                _emit(CLAIM_DPA_MISSING, d.body, [d.doc_id])
            if re.search(r"cross_border_transfer=yes.*sccs=no", d.body):
                _emit(CLAIM_CROSS_BORDER_UNAUTHORIZED, d.body,
                      [d.doc_id])
            if "retention=unbounded" in d.body:
                _emit(CLAIM_RETENTION_UNCAPPED, d.body, [d.doc_id])
            if re.search(r"pii_category=\w+\s+.*undeclared", d.body):
                _emit(CLAIM_PII_CATEGORY_UNDISCLOSED, d.body,
                      [d.doc_id])
    elif role == ROLE_FINANCE:
        for d in docs:
            if d.doc_type != DOC_FINANCE_LINEITEM:
                continue
            m_amount = re.search(r"amount_usd=(\d+)", d.body)
            m_cap = re.search(r"cap_usd=(\d+)", d.body)
            if (m_amount and m_cap and
                    int(m_amount.group(1)) > int(m_cap.group(1))):
                _emit(CLAIM_BUDGET_THRESHOLD_BREACH, d.body,
                      [d.doc_id])
            else:
                m_ak = re.search(r"budget_authority_cap_usd=(\d+)", d.body)
                if m_amount and m_ak and \
                        int(m_amount.group(1)) > int(m_ak.group(1)):
                    _emit(CLAIM_BUDGET_THRESHOLD_BREACH, d.body,
                          [d.doc_id])
            m_net = re.search(r"net_days=(\d+)", d.body)
            m_pp = re.search(r"prepay_pct=(\d+)", d.body)
            if (m_net and m_pp and
                    int(m_net.group(1)) <= 15 and
                    int(m_pp.group(1)) >= 30):
                _emit(CLAIM_PAYMENT_TERMS_AGGRESSIVE, d.body,
                      [d.doc_id])
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


def run_handoff_protocol(scenario: VendorScenario,
                          max_docs_per_role: int = 400,
                          inbox_capacity: int = 64,
                          extractor: Callable[
                              [str, Sequence[VendorDoc], VendorScenario],
                              list[tuple[str, str, tuple[int, ...]]],
                          ] | None = None,
                          ) -> HandoffRouter:
    """Drive typed handoffs for one scenario.

    ``extractor`` defaults to ``extract_claims_for_role``; tests and
    Phase 32's noisy-extractor sweep inject alternate extractors to
    exercise graceful-degradation bounds (Theorem P32-2).
    """
    subs = build_role_subscriptions()
    router = HandoffRouter(subs=subs)
    for role in ALL_ROLES:
        router.register_inbox(RoleInbox(role=role, capacity=inbox_capacity))

    extractor = extractor or extract_claims_for_role
    for role in (ROLE_LEGAL, ROLE_SECURITY, ROLE_PRIVACY, ROLE_FINANCE):
        docs = list(scenario.per_role_docs.get(role, ()))
        if max_docs_per_role:
            docs = docs[:max_docs_per_role]
        claims = extractor(role, docs, scenario)
        for (kind, payload, evids) in claims:
            router.emit(
                source_role=role,
                source_agent_id=ALL_ROLES.index(role),
                claim_kind=kind, payload=payload,
                source_event_ids=evids, round=1)
    return router


# =============================================================================
# Decoder — handoff bundle → structured verdict
# =============================================================================


# Ordered priority. Any scenario whose causal chain contains a higher-
# priority blocker is a BLOCKED verdict; conditional-only chains
# produce CONDITIONAL; empty chain → unknown.
_BLOCKING_KINDS = frozenset({
    CLAIM_DPA_MISSING, CLAIM_CROSS_BORDER_UNAUTHORIZED,
    CLAIM_ENCRYPTION_AT_REST_MISSING,
    CLAIM_PII_CATEGORY_UNDISCLOSED,
})
_CONDITIONAL_KINDS = frozenset({
    CLAIM_LIABILITY_CAP_MISSING, CLAIM_AUTO_RENEWAL_UNFAVOURABLE,
    CLAIM_TERMINATION_RESTRICTIVE, CLAIM_SSO_NOT_SUPPORTED,
    CLAIM_PENTEST_STALE, CLAIM_INCIDENT_SLA_INADEQUATE,
    CLAIM_RETENTION_UNCAPPED, CLAIM_BUDGET_THRESHOLD_BREACH,
    CLAIM_PAYMENT_TERMS_AGGRESSIVE,
})


# Remediation map — keyed by sorted tuple of load-bearing claim
# kinds. Chosen to match the five gold scenarios exactly; an
# unrecognised combination falls through to the priority-single-claim
# remediation.
_REMEDIATION_MAP: dict[frozenset[str], str] = {
    frozenset({CLAIM_DPA_MISSING, CLAIM_RETENTION_UNCAPPED}):
        "require_signed_dpa_and_cap_retention",
    frozenset({CLAIM_LIABILITY_CAP_MISSING,
               CLAIM_AUTO_RENEWAL_UNFAVOURABLE}):
        "negotiate_liability_cap_and_renewal",
    frozenset({CLAIM_ENCRYPTION_AT_REST_MISSING,
               CLAIM_PENTEST_STALE}):
        "require_encryption_and_fresh_pentest",
    frozenset({CLAIM_CROSS_BORDER_UNAUTHORIZED,
               CLAIM_ENCRYPTION_AT_REST_MISSING}):
        "require_sccs_and_encryption_at_rest",
    frozenset({CLAIM_BUDGET_THRESHOLD_BREACH,
               CLAIM_PAYMENT_TERMS_AGGRESSIVE}):
        "require_cfo_approval_and_renegotiate_terms",
}


_SINGLE_REMEDIATION: dict[str, str] = {
    CLAIM_DPA_MISSING: "require_signed_dpa",
    CLAIM_CROSS_BORDER_UNAUTHORIZED: "require_sccs",
    CLAIM_ENCRYPTION_AT_REST_MISSING: "require_encryption_at_rest",
    CLAIM_PII_CATEGORY_UNDISCLOSED: "require_pii_disclosure",
    CLAIM_LIABILITY_CAP_MISSING: "negotiate_liability_cap",
    CLAIM_AUTO_RENEWAL_UNFAVOURABLE: "renegotiate_auto_renewal",
    CLAIM_TERMINATION_RESTRICTIVE: "renegotiate_termination",
    CLAIM_PENTEST_STALE: "require_fresh_pentest",
    CLAIM_INCIDENT_SLA_INADEQUATE: "require_incident_sla",
    CLAIM_RETENTION_UNCAPPED: "cap_retention",
    CLAIM_BUDGET_THRESHOLD_BREACH: "require_cfo_approval",
    CLAIM_PAYMENT_TERMS_AGGRESSIVE: "renegotiate_payment_terms",
    CLAIM_SSO_NOT_SUPPORTED: "require_sso",
}


def _claim_kind_to_flag(kind: str) -> str:
    return kind.lower()


def decode_from_handoffs(handoffs: Sequence[TypedHandoff],
                          ) -> dict[str, object]:
    kinds = {h.claim_kind for h in handoffs}
    verdict = VERDICT_APPROVED
    if kinds & _BLOCKING_KINDS:
        verdict = VERDICT_BLOCKED
    elif kinds & _CONDITIONAL_KINDS:
        verdict = VERDICT_CONDITIONAL
    # Remediation — prefer exact-pair match, otherwise fall back on
    # highest-priority single-claim remediation.
    remediation = "investigate"
    if kinds:
        if frozenset(kinds) in _REMEDIATION_MAP:
            remediation = _REMEDIATION_MAP[frozenset(kinds)]
        else:
            # Highest-priority ordered list of kinds we know; use the
            # first match.
            priority_order = (
                CLAIM_DPA_MISSING, CLAIM_CROSS_BORDER_UNAUTHORIZED,
                CLAIM_ENCRYPTION_AT_REST_MISSING,
                CLAIM_PII_CATEGORY_UNDISCLOSED,
                CLAIM_LIABILITY_CAP_MISSING,
                CLAIM_AUTO_RENEWAL_UNFAVOURABLE,
                CLAIM_TERMINATION_RESTRICTIVE,
                CLAIM_PENTEST_STALE, CLAIM_INCIDENT_SLA_INADEQUATE,
                CLAIM_RETENTION_UNCAPPED,
                CLAIM_BUDGET_THRESHOLD_BREACH,
                CLAIM_PAYMENT_TERMS_AGGRESSIVE,
                CLAIM_SSO_NOT_SUPPORTED,
            )
            for k in priority_order:
                if k in kinds:
                    remediation = _SINGLE_REMEDIATION.get(
                        k, "investigate")
                    break
    flags = tuple(sorted(_claim_kind_to_flag(k) for k in kinds))
    return {
        "verdict": verdict,
        "flags": flags,
        "remediation": remediation,
    }


def _decoder_from_docs(docs: Sequence[VendorDoc]) -> dict[str, object]:
    """Fallback decoder — runs per-role extractors on the delivered
    docs and routes the synthetic claims through the handoff decoder.
    Used by the mock auditor under naive delivery.
    """
    by_role: dict[str, list[VendorDoc]] = {r: [] for r in ALL_ROLES}
    for d in docs:
        if d.origin_role in by_role:
            by_role[d.origin_role].append(d)
    synth: list[TypedHandoff] = []
    nid = 0
    for role in (ROLE_LEGAL, ROLE_SECURITY, ROLE_PRIVACY, ROLE_FINANCE):
        claims = extract_claims_for_role(role, by_role[role],
                                          _DUMMY_SCENARIO)
        for (kind, payload, evids) in claims:
            synth.append(TypedHandoff(
                handoff_id=nid, source_role=role,
                source_agent_id=ALL_ROLES.index(role),
                to_role=ROLE_COMPLIANCE, claim_kind=kind,
                payload=payload, source_event_ids=tuple(evids),
                round=1, payload_cid="", prev_chain_hash="",
                chain_hash=""))
            nid += 1
    return decode_from_handoffs(synth)


_DUMMY_SCENARIO = VendorScenario(
    scenario_id="__dummy__", description="",
    gold_verdict="", gold_flags=(), gold_remediation="",
    causal_chain=(), per_role_docs={r: () for r in ALL_ROLES})


# =============================================================================
# Prompt assembly
# =============================================================================


def build_auditor_prompt(scenario: VendorScenario, strategy: str,
                          docs: Sequence[VendorDoc],
                          handoffs: Sequence[TypedHandoff] = (),
                          substrate_cue: dict | None = None,
                          max_docs_in_prompt: int = 200,
                          ) -> tuple[str, list[VendorDoc], bool]:
    if strategy == STRATEGY_NAIVE:
        delivered = list(docs)
    elif strategy == STRATEGY_ROUTING:
        delivered = [d for d in docs if d.is_fixed_point]
    elif strategy in (STRATEGY_SUBSTRATE, STRATEGY_SUBSTRATE_WRAP):
        delivered = [d for d in docs if d.is_fixed_point]
    else:
        raise ValueError(f"unknown strategy {strategy!r}")
    truncated = False
    if len(delivered) > max_docs_in_prompt:
        truncated = True
        delivered = delivered[:max_docs_in_prompt]
    lines = [
        "You are the COMPLIANCE OFFICER for a vendor-onboarding review.",
        ("Produce exactly three lines:\n"
         "  VERDICT: approved|conditional|blocked\n"
         "  FLAGS: <comma-separated flag labels>\n"
         "  REMEDIATION: <label>"),
        "",
        f"SCENARIO: {scenario.description}",
    ]
    if strategy in (STRATEGY_SUBSTRATE, STRATEGY_SUBSTRATE_WRAP):
        if substrate_cue:
            lines.append("")
            lines.append("SUBSTRATE_ANSWER:")
            lines.append(f"  VERDICT: {substrate_cue['verdict']}")
            lines.append(
                f"  FLAGS: {','.join(substrate_cue['flags'])}")
            lines.append(
                f"  REMEDIATION: {substrate_cue['remediation']}")
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
        lines.append("DELIVERED DOCUMENTS:")
        for d in delivered:
            lines.append(f"- [{d.doc_type} by {d.origin_role}] "
                         f"{d.body}")
        if truncated:
            lines.append(
                "... (document stream truncated; delivered subset is the"
                " first window)")
    lines.append("")
    lines.append("ANSWER:")
    return "\n".join(lines), delivered, truncated


# =============================================================================
# Answer grading
# =============================================================================


_VERDICT_RE = re.compile(r"VERDICT\s*[:\-]?\s*([^\n]+)", re.IGNORECASE)
_FLAGS_RE = re.compile(r"FLAGS?\s*[:\-]?\s*([^\n]+)", re.IGNORECASE)
_REMED_RE = re.compile(r"REMEDIATION\s*[:\-]?\s*([^\n]+)",
                       re.IGNORECASE)


def parse_answer(text: str) -> dict[str, object]:
    v = ""
    flags: tuple[str, ...] = ()
    rem = ""
    m = _VERDICT_RE.search(text)
    if m:
        v = m.group(1).strip().lower()
        v = re.sub(r"[^a-z0-9_]+", "_", v).strip("_")
    m = _FLAGS_RE.search(text)
    if m:
        raw = m.group(1).strip()
        parts = [p.strip().lower() for p in re.split(r"[,\s]+", raw)
                 if p.strip()]
        flags = tuple(sorted(set(parts)))
    m = _REMED_RE.search(text)
    if m:
        rem = m.group(1).strip().lower()
        rem = re.sub(r"[^a-z0-9_]+", "_", rem).strip("_")
    return {"verdict": v, "flags": flags, "remediation": rem}


def grade_answer(scenario: VendorScenario,
                 answer_text: str) -> dict[str, object]:
    parsed = parse_answer(answer_text)
    gold_v = scenario.gold_verdict
    gold_flags = set(scenario.gold_flags)
    gold_rem = scenario.gold_remediation
    v_ok = parsed["verdict"] == gold_v
    f_ok = set(parsed["flags"]) == gold_flags
    r_ok = parsed["remediation"] == gold_rem
    return {
        "verdict_correct": v_ok,
        "flags_correct": f_ok,
        "remediation_correct": r_ok,
        "full_correct": v_ok and f_ok and r_ok,
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


def attribute_failure(scenario: VendorScenario, grading: dict,
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
    # Spurious (over-emission) attribution — if the delivered inbox
    # contains handoffs whose (role, kind) is NOT in the causal chain,
    # AND the grading is wrong, *and* required claims ARE all present,
    # attribute to spurious_claim (a noisy-extractor effect Phase 32
    # tracks as a first-class failure mode).
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
class ComplianceMeasurement:
    scenario_id: str
    strategy: str
    n_docs_delivered: int
    n_handoffs_delivered: int
    n_prompt_chars: int
    n_prompt_tokens_approx: int
    truncated: bool
    n_docs_total: int
    n_docs_causal: int
    n_docs_causal_to_auditor: int
    n_causal_docs_delivered: int
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
            "n_docs_delivered": self.n_docs_delivered,
            "n_handoffs_delivered": self.n_handoffs_delivered,
            "n_prompt_chars": self.n_prompt_chars,
            "n_prompt_tokens_approx": self.n_prompt_tokens_approx,
            "truncated": self.truncated,
            "n_docs_total": self.n_docs_total,
            "n_docs_causal": self.n_docs_causal,
            "n_docs_causal_to_auditor": self.n_docs_causal_to_auditor,
            "n_causal_docs_delivered": self.n_causal_docs_delivered,
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
class ComplianceReport:
    scenario_ids: tuple[str, ...]
    strategies: tuple[str, ...]
    measurements: list[ComplianceMeasurement]
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
            vc = sum(1 for m in ms if m.grading["verdict_correct"])
            fc = sum(1 for m in ms if m.grading["flags_correct"])
            rc = sum(1 for m in ms if m.grading["remediation_correct"])
            mean_tok = sum(m.n_prompt_tokens_approx for m in ms) / n
            trunc = sum(1 for m in ms if m.truncated)
            mean_rel = sum(m.aggregator_relevance_fraction for m in ms) / n
            mean_recall = sum(m.handoff_recall for m in ms) / n
            mean_prec = sum(m.handoff_precision for m in ms) / n
            f_hist: dict[str, int] = {}
            for m in ms:
                f_hist[m.failure_kind] = f_hist.get(m.failure_kind, 0) + 1
            out[strat] = {
                "n": n,
                "accuracy_full": round(correct / n, 4),
                "accuracy_verdict": round(vc / n, 4),
                "accuracy_flags": round(fc / n, 4),
                "accuracy_remediation": round(rc / n, 4),
                "mean_prompt_tokens": round(mean_tok, 2),
                "truncated_count": trunc,
                "mean_aggregator_relevance_fraction": round(mean_rel, 4),
                "mean_handoff_recall": round(mean_recall, 4),
                "mean_handoff_precision": round(mean_prec, 4),
                "failure_hist": f_hist,
            }
        return out


def run_compliance_loop(
        scenarios: Sequence[VendorScenario],
        auditor: Callable[[str], str],
        strategies: Sequence[str] = ALL_STRATEGIES,
        seed: int = 32,
        max_docs_in_prompt: int = 200,
        inbox_capacity: int = 64,
        extractor: Callable[
            [str, Sequence[VendorDoc], VendorScenario],
            list[tuple[str, str, tuple[int, ...]]],
        ] | None = None,
        ) -> ComplianceReport:
    import time as _time
    measurements: list[ComplianceMeasurement] = []

    for scenario in scenarios:
        docs = naive_doc_stream(scenario)
        n_total = len(docs)
        n_causal = sum(1 for d in docs if d.is_causal)
        n_causal_to_auditor = sum(
            1 for d in docs
            if oracle_relevance(d, ROLE_COMPLIANCE, scenario))
        required = {(role, kind)
                    for (role, kind, _p, _e) in scenario.causal_chain}

        router = run_handoff_protocol(
            scenario, max_docs_per_role=max_docs_in_prompt,
            inbox_capacity=inbox_capacity, extractor=extractor)
        inbox = router.inboxes.get(ROLE_COMPLIANCE)
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
                scenario, strat, docs,
                handoffs=delivered_handoffs,
                substrate_cue=cue if strat in (
                    STRATEGY_SUBSTRATE, STRATEGY_SUBSTRATE_WRAP) else None,
                max_docs_in_prompt=max_docs_in_prompt,
            )
            delivered_causal = sum(
                1 for d in delivered
                if oracle_relevance(d, ROLE_COMPLIANCE, scenario))
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
            measurements.append(ComplianceMeasurement(
                scenario_id=scenario.scenario_id, strategy=strat,
                n_docs_delivered=len(delivered),
                n_handoffs_delivered=len(delivered_handoffs),
                n_prompt_chars=len(prompt),
                n_prompt_tokens_approx=max(1, len(prompt) // 4),
                truncated=truncated,
                n_docs_total=n_total,
                n_docs_causal=n_causal,
                n_docs_causal_to_auditor=n_causal_to_auditor,
                n_causal_docs_delivered=delivered_causal,
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

    return ComplianceReport(
        scenario_ids=tuple(s.scenario_id for s in scenarios),
        strategies=tuple(strategies),
        measurements=measurements,
        config={"seed": seed,
                "max_docs_in_prompt": max_docs_in_prompt,
                "inbox_capacity": inbox_capacity},
    )


# =============================================================================
# Mock auditor
# =============================================================================


class MockComplianceAuditor:
    """Deterministic reader of the delivered prompt.

    Mirrors ``MockIncidentAuditor``: if the prompt has a
    ``SUBSTRATE_ANSWER`` block, copy it; else, run the per-role
    extractors on DELIVERED DOCUMENTS and emit the decoded verdict;
    else, return UNKNOWN.
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
            r"SUBSTRATE_ANSWER:\s*\n\s*VERDICT:\s*([^\n]+)\s*\n"
            r"\s*FLAGS:\s*([^\n]+)\s*\n\s*REMEDIATION:\s*([^\n]+)",
            prompt, re.IGNORECASE)
        if m:
            v = m.group(1).strip()
            fl = m.group(2).strip()
            rm = m.group(3).strip()
            self.last_answer = (f"VERDICT: {v}\nFLAGS: {fl}\n"
                                f"REMEDIATION: {rm}\n")
            return self.last_answer
        lines = prompt.splitlines()
        docs: list[VendorDoc] = []
        nid = 0
        collecting = False
        for line in lines:
            s = line.strip()
            if s.startswith("DELIVERED DOCUMENTS:"):
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
            dt = m2.group(1).strip()
            role = m2.group(2).strip()
            body = m2.group(3).strip()
            docs.append(VendorDoc(
                doc_id=nid, doc_type=dt, origin_role=role, body=body))
            nid += 1
        dec = _decoder_from_docs(docs) if docs else {
            "verdict": "unknown", "flags": (), "remediation": "investigate"}
        flags_str = ",".join(dec["flags"])
        self.last_answer = (f"VERDICT: {dec['verdict']}\n"
                            f"FLAGS: {flags_str}\n"
                            f"REMEDIATION: {dec['remediation']}\n")
        return self.last_answer
