"""Phase 38 Part A — adversarial claim extractor for the Phase-35 bank.

Phase 34 shipped ``UnionExtractor`` (``core/ensemble_extractor``)
and an LLM-extractor scenario family where regex and LLM paths
genuinely compete. The Phase-35 contested bank was left
*uncontaminated* at the extractor axis: its regexes have
precision and recall 1 on every causal event by construction.
That was right for Phase-35's focus — isolating the reply-axis
gap. It is wrong for Phase 38 Part A, which needs a *conjunction*
of extractor-axis and reply-axis noise to distinguish single-
layer from two-layer ensemble defenses.

This module ships three lightweight extractor-axis perturbations
that compose on top of ``extract_claims_for_role`` and drop in
at the ``claim_extractor`` parameter surface of
``run_contested_handoff_protocol`` /
``run_adaptive_sub_coordination`` / ``run_contested_loop``:

  * ``DropGoldClaimExtractor``   — adversarial layer-1 attack:
    drops the scenario's gold-root-cause claim from one role's
    emissions, up to ``budget`` times per scenario. Models a
    compromised / silenced producer.
  * ``NarrativeSecondaryExtractor`` — a small, noisy second
    extractor that emits the Phase-35 claims by matching on
    *service tags* (``service=api``, ``service=orders``, ...)
    rather than the primary regex. Recovers the gold claim
    under the drop-gold attack. Does NOT emit non-causal
    claims (distractors stay distractor).
  * ``UnionClaimExtractor``      — convenience wrapper over
    ``core.ensemble_extractor.UnionExtractor`` that exposes the
    Phase-35 claim-tuple shape.

Scope discipline
----------------

  * The narrative secondary is intentionally NOT a full clone of
    the primary regex — it has a narrower coverage (it knows
    only the claim kinds relevant to the Phase-35 bank). A
    realistic deployment would use an LLM extractor; this module
    ships a deterministic stand-in so Phase-38 headlines are
    reproducible sub-second without an Ollama dependency.
  * This module does NOT modify Phase-34's ``UnionExtractor``
    or ``ensemble_extractor`` shape. It reuses them directly.
  * Budgets are per-scenario and per-(role, claim_kind) — so
    ``budget=1`` on ``DropGoldClaimExtractor`` drops exactly
    one gold emission per scenario-role pair, matching the
    Phase-37 adversarial-reply budget semantics.

Theoretical anchor: RESULTS_PHASE38.md § B.1 (Theorem P38-1,
Theorem P38-2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

from vision_mvp.tasks.incident_triage import (
    CLAIM_CRON_OVERRUN, CLAIM_DEADLOCK_SUSPECTED,
    CLAIM_DISK_FILL_CRITICAL, CLAIM_DNS_MISROUTE,
    CLAIM_ERROR_RATE_SPIKE, CLAIM_FW_BLOCK_SURGE,
    CLAIM_LATENCY_SPIKE, CLAIM_OOM_KILL, CLAIM_POOL_EXHAUSTION,
    CLAIM_SLOW_QUERY_OBSERVED, CLAIM_TLS_EXPIRED,
    IncidentEvent, IncidentScenario, ROLE_DB_ADMIN, ROLE_MONITOR,
    ROLE_NETWORK, ROLE_SYSADMIN,
    extract_claims_for_role,
)


ClaimTuple = tuple[str, str, tuple[int, ...]]
ClaimExtractor = Callable[
    [str, Sequence[IncidentEvent], IncidentScenario],
    list[ClaimTuple]]


# =============================================================================
# Adversarial drop-gold extractor
# =============================================================================


@dataclass
class DropGoldClaimExtractor:
    """Wrap an underlying extractor and drop a named ``(role,
    claim_kind)`` pair's emissions, up to ``budget`` per scenario.

    Fields:

      * ``base``            — underlying extractor (default
        ``extract_claims_for_role``).
      * ``target_role``     — role whose emissions to damage.
      * ``target_kind``     — claim kind to drop.
      * ``budget``          — per-scenario drop budget.
      * ``n_dropped``       — running counter.
      * ``n_calls``         — running counter of extractor calls.
    """

    target_role: str
    target_kind: str
    base: ClaimExtractor = field(default=extract_claims_for_role)
    # ``budget`` is kept in the signature for parity with Phase-37
    # adversarial wrappers, but the semantic is "drop every matching
    # emission up to ``budget`` per call". On the Phase-35 bank this
    # is equivalent to a per-scenario budget (each role emits at most
    # one (kind) per call). The adversary is stateless across
    # repeated extractor invocations in the same scenario — this
    # matches a *persistent* adversary that drops on every emission,
    # which is the worst-case Phase-37-style attack.
    budget: int = 1
    n_dropped: int = 0
    n_calls: int = 0

    def __call__(self, role: str,
                 events: Sequence[IncidentEvent],
                 scenario: IncidentScenario,
                 ) -> list[ClaimTuple]:
        self.n_calls += 1
        out = list(self.base(role, events, scenario))
        if role != self.target_role:
            return out
        if self.budget == 0:
            return out
        max_drop = self.budget if self.budget >= 0 else 1_000_000
        filtered: list[ClaimTuple] = []
        dropped_this_call = 0
        for (kind, payload, evids) in out:
            if (kind == self.target_kind
                    and dropped_this_call < max_drop):
                self.n_dropped += 1
                dropped_this_call += 1
                continue
            filtered.append((kind, payload, evids))
        return filtered

    def stats(self) -> dict:
        return {
            "n_calls": self.n_calls,
            "n_dropped": self.n_dropped,
            "target_role": self.target_role,
            "target_kind": self.target_kind,
        }


# =============================================================================
# Narrative secondary extractor (deterministic mock of an LLM extractor)
# =============================================================================


@dataclass
class NarrativeSecondaryExtractor:
    """A deterministic second extractor path keyed on *tags* in
    event bodies rather than the Phase-31 regex set.

    Used as the secondary of an ``UnionExtractor`` in Phase-38 to
    recover an adversarially-dropped gold claim. It has three
    intentional differences from the primary ``extract_claims_for_
    role``:

      1. It reads the ``service=<tag>`` token in event bodies;
      2. It maps ``(role, service_tag)`` → claim_kind via a
         small deterministic table keyed to the Phase-35 bank;
      3. It emits a narrative-style payload that is distinct
         from the primary payload (dedup at the handoff layer
         still catches identical-evid tuples; this exercises
         the Phase-34 UnionExtractor's dedup-by-(kind, evids)
         rule).

    It is strictly weaker than the primary on non-Phase-35
    scenarios — out of scope by design.
    """

    n_calls: int = 0
    n_emitted: int = 0

    # Hand-tuned (role, service_tag_or_event_tag, body_signal) →
    # claim_kind for the six Phase-35 scenarios. Signals are
    # intentionally narrow — each is drawn from the *causal* event
    # bodies in ``tasks/contested_incident``. Distractor events
    # (Phase-31 background chatter) do not carry these signals, so
    # the narrative extractor's precision on the contested bank is
    # 1 by construction.
    _table: tuple[tuple[str, str, str, str], ...] = (
        # db_admin → DEADLOCK (the root-bearing gold we most
        # often drop; narrative matches "deadlock" substring when
        # the event is tagged with the orders service).
        (ROLE_DB_ADMIN, "orders", "deadlock",
         CLAIM_DEADLOCK_SUSPECTED),
        # sysadmin → OOM_KILL (matches oom_kill literal on any
        # service-tagged causal event).
        (ROLE_SYSADMIN, "api", "oom_kill", CLAIM_OOM_KILL),
        (ROLE_SYSADMIN, "app", "oom_kill", CLAIM_OOM_KILL),
        # sysadmin → DISK_FILL (matches used>=90% in body; the
        # primary regex uses used=9X|100% — the substring is
        # tight enough to avoid distractor emission).
        (ROLE_SYSADMIN, "api", "used=99",
         CLAIM_DISK_FILL_CRITICAL),
        (ROLE_SYSADMIN, "api", "used=100",
         CLAIM_DISK_FILL_CRITICAL),
        (ROLE_SYSADMIN, "app", "used=99",
         CLAIM_DISK_FILL_CRITICAL),
        # sysadmin → CRON on exit=137 — Phase-31 causal marker.
        (ROLE_SYSADMIN, "api", "exit=137",
         CLAIM_CRON_OVERRUN),
        (ROLE_SYSADMIN, "app", "exit=137",
         CLAIM_CRON_OVERRUN),
        # network → TLS on reason=expired literal.
        (ROLE_NETWORK, "api", "reason=expired",
         CLAIM_TLS_EXPIRED),
        (ROLE_NETWORK, "mail", "reason=expired",
         CLAIM_TLS_EXPIRED),
        # network → DNS on SERVFAIL literal.
        (ROLE_NETWORK, "api", "SERVFAIL", CLAIM_DNS_MISROUTE),
        (ROLE_NETWORK, "orders", "SERVFAIL",
         CLAIM_DNS_MISROUTE),
        (ROLE_NETWORK, "users", "SERVFAIL",
         CLAIM_DNS_MISROUTE),
    )

    def __call__(self, role: str,
                 events: Sequence[IncidentEvent],
                 scenario: IncidentScenario,
                 ) -> list[ClaimTuple]:
        self.n_calls += 1
        out: list[ClaimTuple] = []
        seen: set[tuple[str, tuple[int, ...]]] = set()
        for ev in events:
            # The narrative extractor reads either a
            # ``service=<tag>`` inline marker in the body OR the
            # event's ``tags`` field — the Phase-35 bank's
            # canonical tag surface. Accepting both makes the
            # narrative extractor recover from events whose body
            # does not embed the service token inline (e.g.
            # ``relation=orders_payments``) but whose tag declares
            # the service.
            service = _extract_service(ev.body)
            tags_lower = tuple(t.lower() for t in (ev.tags or ()))
            body_lower = ev.body.lower()
            # Candidate service tokens: inline ``service=``, then
            # each event tag. A distractor without any service
            # signal at all is skipped — keeps the narrative
            # extractor's precision tied to tagged/tagged-adjacent
            # events.
            candidates = []
            if service is not None:
                candidates.append(service)
            candidates.extend(tags_lower)
            for (r, svc, signal, kind) in self._table:
                if role != r:
                    continue
                if svc not in candidates:
                    continue
                if signal.lower() not in body_lower:
                    continue
                # Emit a narrative-phrased payload — distinct from
                # the primary regex's payload. Encodes the
                # scenario-relevant fields in a prose-like shape.
                payload = _narrative_payload(ev, kind, svc)
                key = (kind, (ev.event_id,))
                if key in seen:
                    continue
                seen.add(key)
                out.append((kind, payload, (ev.event_id,)))
                self.n_emitted += 1
        return out

    def stats(self) -> dict:
        return {
            "n_calls": self.n_calls,
            "n_emitted": self.n_emitted,
        }


def _extract_service(body: str) -> str | None:
    marker = "service="
    idx = body.find(marker)
    if idx < 0:
        return None
    tail = body[idx + len(marker):]
    end = 0
    while end < len(tail) and (tail[end].isalnum() or tail[end] == "_"):
        end += 1
    svc = tail[:end]
    return svc or None


def _narrative_payload(ev: IncidentEvent, claim_kind: str,
                         service: str) -> str:
    # Narrative-style payload. Distinct from the primary regex
    # payload so a dedup on (kind, evids) still merges them, but
    # a payload-content inspection shows the two shapes.
    parts = [f"narr_kind={claim_kind}", f"service={service}"]
    body = ev.body
    for tok in body.split():
        if tok.startswith("service="):
            continue
        parts.append(tok)
    return " ".join(parts)


# =============================================================================
# Union claim extractor — thin wrapper over Phase-34 UnionExtractor
# =============================================================================


@dataclass
class UnionClaimExtractor:
    """Compose two ``ClaimExtractor`` paths with dedup on
    ``(kind, sorted evids)``. Wraps the Phase-34 ``UnionExtractor``
    with the Phase-35 claim-tuple shape.

    Fields:

      * ``primary``   — primary extractor (typically regex,
        possibly wrapped with an adversary).
      * ``secondary`` — secondary extractor (narrative mock, or
        a real LLM extractor).

    This is a minimal bridge; for the full orchestration surface
    (per-call stats, precision/recall reporting) see Phase-34's
    ``UnionExtractor``.
    """

    primary: ClaimExtractor
    secondary: ClaimExtractor
    n_calls: int = 0
    n_primary_only: int = 0
    n_secondary_only: int = 0
    n_shared: int = 0
    n_emitted: int = 0

    def __call__(self, role: str,
                 events: Sequence[IncidentEvent],
                 scenario: IncidentScenario,
                 ) -> list[ClaimTuple]:
        self.n_calls += 1
        p_emit = list(self.primary(role, events, scenario))
        s_emit = list(self.secondary(role, events, scenario))
        seen: dict[tuple[str, tuple[int, ...]], ClaimTuple] = {}
        pk: set[tuple[str, tuple[int, ...]]] = set()
        sk: set[tuple[str, tuple[int, ...]]] = set()
        out: list[ClaimTuple] = []
        for (kind, payload, evids) in p_emit:
            key = (kind, tuple(sorted(evids)))
            pk.add(key)
            if key not in seen:
                seen[key] = (kind, payload, tuple(evids))
                out.append(seen[key])
        for (kind, payload, evids) in s_emit:
            key = (kind, tuple(sorted(evids)))
            sk.add(key)
            if key not in seen:
                seen[key] = (kind, payload, tuple(evids))
                out.append(seen[key])
        shared = pk & sk
        self.n_primary_only += len(pk - shared)
        self.n_secondary_only += len(sk - shared)
        self.n_shared += len(shared)
        self.n_emitted += len(out)
        return out

    def stats(self) -> dict:
        return {
            "n_calls": self.n_calls,
            "n_primary_only": self.n_primary_only,
            "n_secondary_only": self.n_secondary_only,
            "n_shared": self.n_shared,
            "n_emitted": self.n_emitted,
        }


# =============================================================================
# Convenience constructor
# =============================================================================


def build_union_extractor(
        target_role: str,
        target_kind: str,
        drop_budget: int = 1,
        ) -> tuple[UnionClaimExtractor, DropGoldClaimExtractor,
                   NarrativeSecondaryExtractor]:
    """Construct a canonical Phase-38 two-layer-extractor stack:

      primary   = DropGoldClaimExtractor (adversarial wrapper on
                   the Phase-31 regex)
      secondary = NarrativeSecondaryExtractor

    Returns the union, the adversarial primary, and the narrative
    secondary — the caller may want all three for per-layer
    reporting.
    """
    primary = DropGoldClaimExtractor(
        target_role=target_role, target_kind=target_kind,
        budget=drop_budget)
    secondary = NarrativeSecondaryExtractor()
    union = UnionClaimExtractor(primary=primary, secondary=secondary)
    return union, primary, secondary


__all__ = [
    "DropGoldClaimExtractor", "NarrativeSecondaryExtractor",
    "UnionClaimExtractor", "build_union_extractor",
]
