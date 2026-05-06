"""Phase 32 — parameterised claim-extractor noise wrappers.
Phase 34 — per-role and adversarial noise extensions.

Phase 31's typed-handoff substrate (``role_handoff``) was evaluated
with *effectively perfect* extractors: the claim-regex only matched
scenario-specific causal payloads, so extractor recall and precision
on causal events were both 1.0 by construction. Real-world agent teams
do not have perfect extractors — LLM-based claim extraction has recall
and precision strictly below 1, and the typed-handoff correctness
guarantee (Theorem P31-4) was proved *under the assumption* that the
extractor is sound and complete. Phase 32 Part B falsifies or confirms
the graceful-degradation claim (Conjecture C31-7 → Theorem P32-2) by
injecting controlled noise into the extractor and measuring how far
the substrate's correctness survives.

Phase 34 adds two extensions on the same boundary:

  * ``PerRoleNoiseConfig`` — a wrapper that carries a *different*
    ``NoiseConfig`` per role, so the sweep can model role-heterogeneous
    extractors (e.g. a 0.5b LLM that is 100 % reliable on one role and
    100 % unreliable on another). This is the instrument for Phase 34
    Conjecture C33-3 → Theorem P34-1.
  * ``adversarial_extractor`` — an *adversarial* noise wrapper that
    selectively targets load-bearing claims along the causal chain
    (rather than random Bernoulli drops). Under a matched nominal
    budget, the adversarial wrapper provably dominates i.i.d.
    degradation (Theorem P34-2).

Noise knobs
-----------

The module exposes ``noisy_extractor`` (i.i.d. Bernoulli noise) and
``adversarial_extractor`` (targeted noise). Both wrap any extractor of
the shape

    extractor(role, events, scenario) -> list[(claim_kind, payload, evids)]

and return a *noisy* version with the same shape. The i.i.d. noise is
controlled by ``NoiseConfig`` with five first-class parameters:

  * ``drop_prob``       — per-claim probability of *silently dropping*
    a correctly-extracted causal claim. Models extractor recall drop.
  * ``spurious_prob``   — per-event probability of *emitting* a
    spurious claim (kind drawn from the extractor's known kinds) on
    an event that should not have produced one. Models extractor
    precision drop.
  * ``mislabel_prob``   — per-emission probability of *relabelling*
    an otherwise-correct emission's claim_kind to a different kind
    from the same role's claim set. Models extractor-type confusion.
  * ``payload_corrupt_prob`` — per-emission probability of a
    content-level corruption of the payload (e.g. drop a key token).
    Mostly affects grader / decoder downstream — included for
    completeness.
  * ``rng``             — optional ``random.Random`` seed for
    reproducible sweeps.

Composition semantics
---------------------

``noisy_extractor`` applies noise in a *fixed order* so the resulting
claim stream is deterministic per seed:

  1. Extract the ground-truth emissions (the original extractor).
  2. For each emission, decide with prob ``drop_prob`` whether to
     drop it. Dropped emissions contribute to *recall loss*.
  3. For each kept emission, decide with prob ``mislabel_prob``
     whether to swap its kind for another kind from the extractor's
     known-kind pool. Mislabels are *still valid claims*, just
     wrong-kinded.
  4. For each role-event pair that did *not* produce a ground-truth
     emission, decide with prob ``spurious_prob`` whether to emit a
     spurious claim (random kind from the role's pool, body is the
     event's body unchanged — so the *claim kind* is wrong, not the
     evidence).
  5. Payload corruption is the final pass.

No guarantees are made about the statistical independence of these
steps beyond the seeding — the implementation is deliberately simple
so that the sweep can be reproduced byte-for-byte.

Adversarial composition semantics
---------------------------------

``adversarial_extractor`` chooses targets deterministically from the
scenario's causal chain (if available via ``scenario.causal_chain``)
rather than flipping coins per emission:

  1. Identify the *load-bearing* claims — every ``(role, kind)`` pair
     present in the gold causal chain.
  2. Under ``AdversarialConfig.target_mode == "load_bearing_drop"``,
     drop every matching emission up to a per-scenario budget
     (``drop_budget``) — preferring higher-priority kinds first if a
     priority function is supplied. This targets the exact claims the
     aggregator's decoder requires.
  3. Under ``target_mode == "role_silencing"``, drop *every* emission
     from roles listed in ``target_roles``. Models the production risk
     of one extractor instance failing completely (network flake, OOM,
     prompt-injection).
  4. Under ``target_mode == "severity_escalation"``, inject a spurious
     high-severity claim (kind drawn from ``escalation_kinds``) on the
     first distractor event. Targets max-ordinal decoders like the
     security-escalation domain.

The adversarial wrapper is an *upper bound* on damage at its nominal
budget: an i.i.d. extractor with matched pooled drop/spurious rates is
structurally unable to hit the load-bearing claims as reliably.

Scope discipline (what this module does NOT do)
-----------------------------------------------

  * It does NOT change the subscription table, the inbox, or the
    router. Noise is strictly at the extractor boundary.
  * It does NOT claim to model the noise profile of any specific
    LLM-based extractor. The parameters are a parsimonious surrogate
    chosen to stress the substrate's graceful-degradation bound;
    empirical LLM-extractor noise can be fit to the same knobs by
    matched-pair experiments (Phase 33 calibration).
  * It is NOT intended to be used in the production delivery path.
    Production runs use the real extractor; this module exists to
    measure substrate robustness under *controlled degradation*.
  * The adversarial wrapper does NOT model a *cryptographically*
    adversarial attacker (signature spoofing, payload tampering under
    the chain hash). It models an *extractor-boundary* adversary —
    which is the realistic threat model for agent-team products.

Theoretical anchor: RESULTS_PHASE32.md § B (Theorem P32-2);
RESULTS_PHASE34.md § B (Theorems P34-1, P34-2).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence, TypeVar


# =============================================================================
# Type aliases — ``EventT`` is any per-role observable (IncidentEvent in
# phase 31, VendorDoc in phase 32). ``ScenarioT`` is the enclosing
# scenario dataclass. The noise wrapper does not look into either type
# beyond what the original extractor does.
# =============================================================================


EventT = TypeVar("EventT")
ScenarioT = TypeVar("ScenarioT")

Extractor = Callable[
    [str, Sequence[EventT], ScenarioT],
    list[tuple[str, str, tuple[int, ...]]],
]


@dataclass(frozen=True)
class NoiseConfig:
    """Noise knobs for a Phase-32 extractor sweep.

    Every probability is in ``[0, 1]``; 0 means the noise axis is
    disabled, 1 means it fires on every eligible item. The default
    constructor is an identity transform: ``NoiseConfig()`` produces
    zero noise and yields the original extractor output.
    """

    drop_prob: float = 0.0
    spurious_prob: float = 0.0
    mislabel_prob: float = 0.0
    payload_corrupt_prob: float = 0.0
    seed: int = 0

    def is_identity(self) -> bool:
        return (self.drop_prob == 0.0 and self.spurious_prob == 0.0
                and self.mislabel_prob == 0.0
                and self.payload_corrupt_prob == 0.0)

    def as_dict(self) -> dict:
        return {
            "drop_prob": self.drop_prob,
            "spurious_prob": self.spurious_prob,
            "mislabel_prob": self.mislabel_prob,
            "payload_corrupt_prob": self.payload_corrupt_prob,
            "seed": self.seed,
        }


def _payload_corrupt(body: str, rng: random.Random) -> str:
    """Drop one non-first, non-last whitespace token.

    A small corruption that leaves the body parseable by the grader
    at the structural level but can break content-level checks. Kept
    minimal so that the cost of a single corruption is measurable;
    aggressive corruption is out of scope for Phase 32.
    """
    tokens = body.split()
    if len(tokens) < 3:
        return body
    idx = rng.randrange(1, len(tokens) - 1)
    tokens.pop(idx)
    return " ".join(tokens)


def noisy_extractor(base: Extractor,
                     known_kinds_by_role: dict[str, Sequence[str]],
                     noise: NoiseConfig,
                     ) -> Extractor:
    """Return a noisy wrapper around ``base``.

    ``known_kinds_by_role`` is the pool used for spurious claim
    emission and mislabel — each role maps to the set of claim kinds
    it *can* produce. This is a *specification* of the extractor's
    domain, not a soft guess; the wrapper must not emit claim kinds
    outside this set.

    The returned extractor has the same signature as ``base`` and is
    deterministic in ``(noise.seed, scenario, role, events)``. Calling
    it twice on the same inputs yields the same output.
    """
    if noise.is_identity():
        return base

    def _wrapped(role: str, events: Sequence[EventT],
                 scenario: ScenarioT,
                 ) -> list[tuple[str, str, tuple[int, ...]]]:
        # Per-call deterministic RNG so that parallel invocations do
        # not clobber each other and so that a sweep over roles on
        # the same seed still differs per-role.
        key = (noise.seed, role, getattr(scenario, "scenario_id", ""))
        rng = random.Random(hash(key) & 0xFFFFFFFF)
        base_emissions = base(role, events, scenario)
        pool = list(known_kinds_by_role.get(role, ()))

        out: list[tuple[str, str, tuple[int, ...]]] = []
        # Step 1: drop emissions (recall drop).
        for (kind, payload, evids) in base_emissions:
            if rng.random() < noise.drop_prob:
                continue
            # Step 2: mislabel (keep emission but change kind).
            if pool and rng.random() < noise.mislabel_prob:
                others = [k for k in pool if k != kind]
                if others:
                    kind = rng.choice(others)
            # Step 3: payload corruption.
            if rng.random() < noise.payload_corrupt_prob:
                payload = _payload_corrupt(payload, rng)
            out.append((kind, payload, tuple(evids)))

        # Step 4: spurious emissions. We iterate over events the
        # base extractor did NOT emit from; we compute an "emitted
        # ids" set by collecting the first evid of each base
        # emission. In practice most extractors emit claim from one
        # event per claim; we still use the *event* as the spurious
        # anchor.
        emitted_ids: set[int] = set()
        for (_k, _p, evids) in base_emissions:
            for eid in evids:
                emitted_ids.add(eid)
        if pool:
            for ev in events:
                eid = getattr(ev, "event_id", None)
                if eid is None:
                    eid = getattr(ev, "doc_id", None)
                if eid is None:
                    continue
                if eid in emitted_ids:
                    continue
                if getattr(ev, "is_fixed_point", False):
                    continue
                if rng.random() < noise.spurious_prob:
                    kind = rng.choice(pool)
                    body = getattr(ev, "body", "")
                    out.append((kind, body, (eid,)))
        return out

    return _wrapped


# =============================================================================
# Domain knowledge — known-kinds-by-role for the programme's task
# families. These are imported only where the noise wrapper is wired in,
# to avoid a circular import.
# =============================================================================


def incident_triage_known_kinds() -> dict[str, Sequence[str]]:
    """Known claim kinds for the Phase-31 incident-triage benchmark,
    keyed by producer role. A centralized location because the noise
    wrapper needs this specification and the tests exercise it.
    """
    from ..tasks.incident_triage import (
        ROLE_MONITOR, ROLE_DB_ADMIN, ROLE_SYSADMIN, ROLE_NETWORK,
        CLAIM_ERROR_RATE_SPIKE, CLAIM_LATENCY_SPIKE,
        CLAIM_SLOW_QUERY_OBSERVED, CLAIM_POOL_EXHAUSTION,
        CLAIM_DEADLOCK_SUSPECTED, CLAIM_DISK_FILL_CRITICAL,
        CLAIM_CRON_OVERRUN, CLAIM_OOM_KILL, CLAIM_TLS_EXPIRED,
        CLAIM_DNS_MISROUTE, CLAIM_FW_BLOCK_SURGE,
    )
    return {
        ROLE_MONITOR: (CLAIM_ERROR_RATE_SPIKE, CLAIM_LATENCY_SPIKE),
        ROLE_DB_ADMIN: (CLAIM_SLOW_QUERY_OBSERVED,
                         CLAIM_POOL_EXHAUSTION,
                         CLAIM_DEADLOCK_SUSPECTED),
        ROLE_SYSADMIN: (CLAIM_DISK_FILL_CRITICAL,
                         CLAIM_CRON_OVERRUN, CLAIM_OOM_KILL),
        ROLE_NETWORK: (CLAIM_TLS_EXPIRED, CLAIM_DNS_MISROUTE,
                        CLAIM_FW_BLOCK_SURGE),
    }


def security_escalation_known_kinds() -> dict[str, Sequence[str]]:
    """Known claim kinds for the Phase-33 security-escalation benchmark."""
    from ..tasks.security_escalation import (
        ROLE_SOC_ANALYST, ROLE_IR_ENGINEER, ROLE_THREAT_INTEL,
        ROLE_DATA_STEWARD, CLAIM_AUTH_SPIKE,
        CLAIM_PHISHING_DETECTED, CLAIM_LATERAL_MOVEMENT,
        CLAIM_BRUTE_FORCE, CLAIM_PERSISTENCE_INSTALLED,
        CLAIM_DATA_STAGING, CLAIM_MALWARE_DETECTED,
        CLAIM_PRIV_ESCALATION,
        CLAIM_IOC_KNOWN_BAD_IP, CLAIM_IOC_MALICIOUS_DOMAIN,
        CLAIM_TTP_ATTRIBUTED, CLAIM_SUPPLY_CHAIN_IOC,
        CLAIM_REGULATED_DATA_EXPOSED, CLAIM_PII_AT_RISK,
        CLAIM_CROSS_TENANT_LEAK,
    )
    return {
        ROLE_SOC_ANALYST: (CLAIM_AUTH_SPIKE, CLAIM_PHISHING_DETECTED,
                            CLAIM_LATERAL_MOVEMENT,
                            CLAIM_BRUTE_FORCE),
        ROLE_IR_ENGINEER: (CLAIM_PERSISTENCE_INSTALLED,
                            CLAIM_DATA_STAGING,
                            CLAIM_MALWARE_DETECTED,
                            CLAIM_PRIV_ESCALATION),
        ROLE_THREAT_INTEL: (CLAIM_IOC_KNOWN_BAD_IP,
                             CLAIM_IOC_MALICIOUS_DOMAIN,
                             CLAIM_TTP_ATTRIBUTED,
                             CLAIM_SUPPLY_CHAIN_IOC),
        ROLE_DATA_STEWARD: (CLAIM_REGULATED_DATA_EXPOSED,
                             CLAIM_PII_AT_RISK,
                             CLAIM_CROSS_TENANT_LEAK),
    }


def compliance_review_known_kinds() -> dict[str, Sequence[str]]:
    """Known claim kinds for the Phase-32 compliance-review benchmark."""
    from ..tasks.compliance_review import (
        ROLE_LEGAL, ROLE_SECURITY, ROLE_PRIVACY, ROLE_FINANCE,
        CLAIM_LIABILITY_CAP_MISSING, CLAIM_AUTO_RENEWAL_UNFAVOURABLE,
        CLAIM_TERMINATION_RESTRICTIVE,
        CLAIM_ENCRYPTION_AT_REST_MISSING, CLAIM_SSO_NOT_SUPPORTED,
        CLAIM_PENTEST_STALE, CLAIM_INCIDENT_SLA_INADEQUATE,
        CLAIM_DPA_MISSING, CLAIM_CROSS_BORDER_UNAUTHORIZED,
        CLAIM_RETENTION_UNCAPPED, CLAIM_PII_CATEGORY_UNDISCLOSED,
        CLAIM_BUDGET_THRESHOLD_BREACH, CLAIM_PAYMENT_TERMS_AGGRESSIVE,
    )
    return {
        ROLE_LEGAL: (CLAIM_LIABILITY_CAP_MISSING,
                      CLAIM_AUTO_RENEWAL_UNFAVOURABLE,
                      CLAIM_TERMINATION_RESTRICTIVE),
        ROLE_SECURITY: (CLAIM_ENCRYPTION_AT_REST_MISSING,
                         CLAIM_SSO_NOT_SUPPORTED,
                         CLAIM_PENTEST_STALE,
                         CLAIM_INCIDENT_SLA_INADEQUATE),
        ROLE_PRIVACY: (CLAIM_DPA_MISSING,
                        CLAIM_CROSS_BORDER_UNAUTHORIZED,
                        CLAIM_RETENTION_UNCAPPED,
                        CLAIM_PII_CATEGORY_UNDISCLOSED),
        ROLE_FINANCE: (CLAIM_BUDGET_THRESHOLD_BREACH,
                        CLAIM_PAYMENT_TERMS_AGGRESSIVE),
    }


# =============================================================================
# Phase 34 — per-role noise wrapper
# =============================================================================


@dataclass(frozen=True)
class PerRoleNoiseConfig:
    """Role → NoiseConfig mapping + a fallback pooled NoiseConfig.

    Constructed by Phase 34's calibration layer to fit a measured real-
    LLM extractor's per-role noise profile. Passed to
    ``per_role_noisy_extractor`` which dispatches on ``role`` at call
    time — each role sees its own Bernoulli parameters.

    The fallback is used when a role is not in ``by_role`` (useful if
    the sweep explicitly ignores the aggregator role). The default
    fallback is identity noise.
    """

    by_role: dict[str, NoiseConfig] = field(default_factory=dict)
    fallback: NoiseConfig = field(default_factory=NoiseConfig)
    seed: int = 0

    def for_role(self, role: str) -> NoiseConfig:
        return self.by_role.get(role, self.fallback)

    def is_identity(self) -> bool:
        if not self.fallback.is_identity():
            return False
        return all(c.is_identity() for c in self.by_role.values())

    def as_dict(self) -> dict:
        return {
            "by_role": {r: c.as_dict() for r, c in self.by_role.items()},
            "fallback": self.fallback.as_dict(),
            "seed": self.seed,
        }


def per_role_noisy_extractor(base: Extractor,
                              known_kinds_by_role: dict[str, Sequence[str]],
                              per_role: PerRoleNoiseConfig,
                              ) -> Extractor:
    """Role-dispatched i.i.d. noise wrapper.

    For each incoming ``(role, events, scenario)`` call, this picks the
    role-specific ``NoiseConfig`` from ``per_role.by_role`` (or the
    fallback) and applies the standard Phase-32 wrapper at that config.
    The per-role seed is derived from ``per_role.seed`` and the role
    string so different roles are stochastically independent on the
    same scenario.
    """
    if per_role.is_identity():
        return base

    # Pre-compile one wrapper per role for stable per-role semantics.
    wrapped_by_role: dict[str, Extractor] = {}
    for role, cfg in per_role.by_role.items():
        # Derive a per-role seed so that role A's RNG stream does not
        # depend on role B's.
        role_cfg = NoiseConfig(
            drop_prob=cfg.drop_prob,
            spurious_prob=cfg.spurious_prob,
            mislabel_prob=cfg.mislabel_prob,
            payload_corrupt_prob=cfg.payload_corrupt_prob,
            seed=(cfg.seed ^ per_role.seed
                  ^ (hash(role) & 0xFFFFFFFF)) & 0xFFFFFFFF,
        )
        wrapped_by_role[role] = noisy_extractor(
            base, known_kinds_by_role, role_cfg)
    fallback_cfg = NoiseConfig(
        drop_prob=per_role.fallback.drop_prob,
        spurious_prob=per_role.fallback.spurious_prob,
        mislabel_prob=per_role.fallback.mislabel_prob,
        payload_corrupt_prob=per_role.fallback.payload_corrupt_prob,
        seed=(per_role.fallback.seed ^ per_role.seed) & 0xFFFFFFFF,
    )
    fallback_wrapped = noisy_extractor(base, known_kinds_by_role,
                                        fallback_cfg)

    def _wrapped(role: str, events: Sequence[EventT],
                 scenario: ScenarioT,
                 ) -> list[tuple[str, str, tuple[int, ...]]]:
        fn = wrapped_by_role.get(role, fallback_wrapped)
        return fn(role, events, scenario)

    return _wrapped


# =============================================================================
# Phase 34 — adversarial extractor wrapper
# =============================================================================


ADVERSARIAL_MODE_LOAD_BEARING_DROP = "load_bearing_drop"
ADVERSARIAL_MODE_ROLE_SILENCING = "role_silencing"
ADVERSARIAL_MODE_SEVERITY_ESCALATION = "severity_escalation"
ADVERSARIAL_MODE_COMBINED = "combined"


@dataclass(frozen=True)
class AdversarialConfig:
    """Knobs for the Phase-34 adversarial extractor wrapper.

    Unlike ``NoiseConfig`` whose degradation is distribution-wide, this
    wrapper chooses *which* claims to hurt based on the scenario's gold
    causal chain. It is the realistic-production threat model for
    Phase 34 Part B.

    * ``target_mode`` — one of ``ADVERSARIAL_MODE_*``:
        * ``load_bearing_drop`` — drop load-bearing emissions (those
          whose ``(role, kind)`` is in the gold causal chain), up to
          ``drop_budget`` per (role, scenario) call.
        * ``role_silencing`` — drop every emission from roles listed
          in ``target_roles``. Equivalent to a per-role extractor
          outage.
        * ``severity_escalation`` — inject a spurious claim with a
          kind drawn from ``escalation_kinds`` on the first distractor
          event. Targets max-ordinal decoders.
        * ``combined`` — apply all three passes in the order above.
    * ``drop_budget`` — maximum number of load-bearing claims to drop
      per (role, scenario). Default 1. Value ``-1`` means "drop all
      load-bearing claims for this role in this scenario".
    * ``target_roles`` — list of roles to silence in
      ``role_silencing`` / ``combined``. Empty tuple = silence no
      role (pass-through).
    * ``escalation_kinds`` — claim kinds to emit in
      ``severity_escalation`` / ``combined``. If empty, the wrapper
      picks the first kind from ``known_kinds_by_role[role]``.
    * ``priority_order`` — optional sequence of claim kinds from
      highest-priority to lowest. When set, the ``load_bearing_drop``
      pass prefers dropping higher-priority kinds first (hurts the
      decoder more under a small budget).
    * ``seed`` — seed for tie-breaking. The adversarial wrapper is
      mostly deterministic given the scenario; the seed only matters
      when the causal chain has ties.
    """

    target_mode: str = ADVERSARIAL_MODE_LOAD_BEARING_DROP
    drop_budget: int = 1
    target_roles: tuple[str, ...] = ()
    escalation_kinds: tuple[str, ...] = ()
    priority_order: tuple[str, ...] = ()
    seed: int = 0

    def as_dict(self) -> dict:
        return {
            "target_mode": self.target_mode,
            "drop_budget": self.drop_budget,
            "target_roles": list(self.target_roles),
            "escalation_kinds": list(self.escalation_kinds),
            "priority_order": list(self.priority_order),
            "seed": self.seed,
        }


def _scenario_causal_pairs(scenario: ScenarioT
                             ) -> set[tuple[str, str]]:
    """Return ``{(role, claim_kind)}`` from ``scenario.causal_chain``
    if present, else an empty set (wrapper degrades gracefully)."""
    chain = getattr(scenario, "causal_chain", ())
    pairs: set[tuple[str, str]] = set()
    for item in chain:
        # item shape is (role, kind, payload, evids)
        if len(item) >= 2:
            pairs.add((item[0], item[1]))
    return pairs


def _priority_rank(priority_order: Sequence[str], kind: str) -> int:
    """Higher number → higher priority (dropped first)."""
    for i, k in enumerate(priority_order):
        if k == kind:
            return len(priority_order) - i
    return 0


def adversarial_extractor(base: Extractor,
                           known_kinds_by_role: dict[str, Sequence[str]],
                           adv: AdversarialConfig,
                           ) -> Extractor:
    """Return an adversarial wrapper around ``base``.

    The returned extractor has the same signature as ``base``. It
    reads the scenario's causal chain to identify load-bearing claims
    and selectively damages them; if the scenario has no
    ``causal_chain`` attribute the wrapper degrades to a pass-through
    (and the caller will see no noise).
    """
    if adv.target_mode not in (
            ADVERSARIAL_MODE_LOAD_BEARING_DROP,
            ADVERSARIAL_MODE_ROLE_SILENCING,
            ADVERSARIAL_MODE_SEVERITY_ESCALATION,
            ADVERSARIAL_MODE_COMBINED):
        raise ValueError(f"unknown adversarial mode {adv.target_mode!r}")

    target_roles_set = set(adv.target_roles)
    prio = tuple(adv.priority_order)

    def _wrapped(role: str, events: Sequence[EventT],
                 scenario: ScenarioT,
                 ) -> list[tuple[str, str, tuple[int, ...]]]:
        base_emissions = base(role, events, scenario)
        causal_pairs = _scenario_causal_pairs(scenario)

        out: list[tuple[str, str, tuple[int, ...]]] = []

        # Role silencing: emit nothing from targeted roles.
        if adv.target_mode in (ADVERSARIAL_MODE_ROLE_SILENCING,
                                 ADVERSARIAL_MODE_COMBINED):
            if role in target_roles_set:
                # Still allow severity_escalation pass below if
                # combined — but drop all normal emissions.
                base_emissions = []

        # Load-bearing drop: remove load-bearing emissions up to budget.
        if adv.target_mode in (ADVERSARIAL_MODE_LOAD_BEARING_DROP,
                                 ADVERSARIAL_MODE_COMBINED):
            load_bearing_idx: list[int] = []
            for i, (kind, _p, _e) in enumerate(base_emissions):
                if (role, kind) in causal_pairs:
                    load_bearing_idx.append(i)
            # Sort by priority descending (higher priority dropped first).
            if prio:
                load_bearing_idx.sort(
                    key=lambda i: _priority_rank(
                        prio, base_emissions[i][0]),
                    reverse=True)
            budget = (len(load_bearing_idx)
                        if adv.drop_budget < 0 else adv.drop_budget)
            to_drop = set(load_bearing_idx[:budget])
            for i, em in enumerate(base_emissions):
                if i in to_drop:
                    continue
                out.append((em[0], em[1], tuple(em[2])))
        else:
            for em in base_emissions:
                out.append((em[0], em[1], tuple(em[2])))

        # Severity escalation: inject a spurious claim on the first
        # non-causal event.
        if adv.target_mode in (ADVERSARIAL_MODE_SEVERITY_ESCALATION,
                                 ADVERSARIAL_MODE_COMBINED):
            role_kinds = tuple(known_kinds_by_role.get(role, ()))
            candidates = tuple(
                k for k in adv.escalation_kinds if k in role_kinds)
            if not candidates and role_kinds:
                candidates = (role_kinds[0],)
            if candidates:
                emitted_ids: set[int] = set()
                for (_k, _p, evids) in base_emissions:
                    for eid in evids:
                        emitted_ids.add(eid)
                for ev in events:
                    eid = getattr(ev, "event_id", None)
                    if eid is None:
                        eid = getattr(ev, "doc_id", None)
                    if eid is None:
                        continue
                    if eid in emitted_ids:
                        continue
                    if getattr(ev, "is_fixed_point", False):
                        continue
                    if getattr(ev, "is_causal", False):
                        continue
                    injected_kind = candidates[0]
                    body = getattr(ev, "body", "")
                    out.append((injected_kind, body, (eid,)))
                    break

        return out

    return _wrapped


# =============================================================================
# Phase 34 — convenience: per-role identity and uniform builders
# =============================================================================


def build_uniform_per_role(roles: Sequence[str],
                             config: NoiseConfig,
                             ) -> PerRoleNoiseConfig:
    """Every role shares the same ``NoiseConfig``. Used by sanity tests
    to show per-role wrapper == pooled wrapper when the configs agree.
    """
    return PerRoleNoiseConfig(
        by_role={r: config for r in roles}, fallback=config,
        seed=config.seed)


def build_from_audit_per_role(audit_by_role: dict[str, dict[str, float]],
                                seed: int = 0,
                                ) -> PerRoleNoiseConfig:
    """Turn an ``ExtractorAudit.by_role`` dict into a
    ``PerRoleNoiseConfig`` for re-simulation.

    The audit's per-role quadruple (drop_rate, spurious_per_event,
    mislabel_rate, payload_corrupt_rate) becomes each role's
    ``NoiseConfig``. Used to ask "if we replay the Phase-32 sweep with
    per-role noise calibrated to the real LLM, does substrate accuracy
    match better than the pooled match?"
    """
    by_role: dict[str, NoiseConfig] = {}
    for role, rates in audit_by_role.items():
        by_role[role] = NoiseConfig(
            drop_prob=float(rates.get("drop_rate", 0.0)),
            spurious_prob=float(rates.get("spurious_per_event", 0.0)),
            mislabel_prob=float(rates.get("mislabel_rate", 0.0)),
            payload_corrupt_prob=float(
                rates.get("payload_corrupt_rate", 0.0)),
            seed=seed,
        )
    return PerRoleNoiseConfig(by_role=by_role, seed=seed)
