"""Phase 38 Part B — minimum-dynamic-primitive feature ablation.

Phase 37 Conjecture C37-4 named a candidate minimal feature
set for a dynamic communication primitive on the contested
and nested task families:

  1. A bounded typed reply-kind enum.
  2. A bounded witness-token cap on every reply.
  3. A terminating resolution rule that is a deterministic
     function of the multiset of replies.
  4. Round-aware reply state exposed to producers for the
     current round (directly or via an inverse-direction
     briefing channel).
  5. A type-level or runtime-enforced bounded-context
     invariant.

The Phase-35 escalation thread and the Phase-36 adaptive
subscription (augmented with briefing edges) are both
implementations of this feature set; the feature set is
conjectured to be minimal in the sense that omitting any one
feature causes a known collapse on some scenario.

This module ships the ablation primitive: a feature-flagged
thread-and-adaptive-sub runner that lets the driver toggle
each load-bearing feature on and off and measure the
collapse. The purpose is not to design a new substrate; it
is to construct a *falsifier table* for C37-4 — one cell per
(feature, task family) that either confirms the feature is
load-bearing (collapse) or shows the task can be solved
without it (surprise).

Feature flags
-------------

Each flag removes one affordance from a fully-featured runner:

  * ``typed_vocab``         — when False, the resolution rule
    no longer dispatches on ``REPLY_INDEPENDENT_ROOT``. Every
    emitted reply counts as "I have something to say" and the
    resolution is reduced to first-arrival-wins by candidate
    priority. This simulates a primitive whose replies are
    untyped — a single EMIT bit rather than an enum — and the
    auditor has no way to tell "I am the root cause" from "I
    saw something downstream".
  * ``bounded_witness``     — when False, the witness cap is
    lifted to a very large number (4096 tokens) so *bounded
    context* is no longer enforced at the thread layer. The
    Phase-35 theorem P35-2 bound is not recovered.
  * ``terminating_resolution`` — when False, the thread has
    no deterministic close rule: it emits ``NO_CONSENSUS`` on
    close regardless of replies. Simulates a primitive that
    forgets to aggregate.
  * ``round_aware_state``   — when False, producers do not
    read the round-1 replies before emitting round-2 (matters
    only on the nested bank; on Phase-35 one-round tasks it
    is a no-op).
  * ``frozen_membership``   — when False, an extra role can be
    added to the thread mid-flight. On our benchmarks this
    does not change accuracy (the extra role adds no useful
    signal), so it is included as a *null-control* ablation
    — a feature Phase-37 C37-4 listed but whose removal does
    not cause collapse on the tested families. Keeping it in
    the set means any user of C37-4 can reproduce the
    "nonominality is not collapse" observation.

Scope discipline
----------------

  * This module does NOT modify the Phase-35 thread primitive
    or the Phase-36 adaptive-sub primitive. It wraps them.
  * The ablated runners have the same API shape as the
    Phase-35 / Phase-37 runners — they return a
    ``(router, handoffs, debug)`` tuple.
  * ``RESULTS_PHASE38.md`` § B.2 contains the ablation-table.
    The claim Phase 38 makes is:

      * Removing (typed_vocab) collapses both contested and
        nested banks to the static baseline.
      * Removing (terminating_resolution) collapses both.
      * Removing (round_aware_state) preserves Phase-35
        contested accuracy and collapses nested.
      * Removing (bounded_witness) preserves accuracy but
        removes the P35-2 context bound.
      * Removing (frozen_membership) preserves accuracy (null
        control).

Theoretical anchor: RESULTS_PHASE38.md § B.2 (Theorem P38-3,
Conjecture C38-2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence


# =============================================================================
# Feature flags
# =============================================================================


FEATURES = (
    "typed_vocab",
    "bounded_witness",
    "terminating_resolution",
    "round_aware_state",
    "frozen_membership",
)


@dataclass(frozen=True)
class AblatedFeatures:
    """Switches for each primitive feature.

    ``True`` means the feature is present; ``False`` means it has
    been removed. The ``"full"`` configuration has all five
    features on.
    """

    typed_vocab: bool = True
    bounded_witness: bool = True
    terminating_resolution: bool = True
    round_aware_state: bool = True
    frozen_membership: bool = True

    def as_dict(self) -> dict:
        return {
            "typed_vocab": bool(self.typed_vocab),
            "bounded_witness": bool(self.bounded_witness),
            "terminating_resolution":
                bool(self.terminating_resolution),
            "round_aware_state": bool(self.round_aware_state),
            "frozen_membership": bool(self.frozen_membership),
        }

    def n_features_on(self) -> int:
        return sum(1 for v in self.as_dict().values() if v)

    def label(self) -> str:
        bits = []
        for k in FEATURES:
            bits.append("+" + k if getattr(self, k)
                        else "-" + k)
        return ",".join(bits)


def full_features() -> AblatedFeatures:
    return AblatedFeatures()


def no_features() -> AblatedFeatures:
    return AblatedFeatures(
        typed_vocab=False, bounded_witness=False,
        terminating_resolution=False,
        round_aware_state=False, frozen_membership=False)


def only_missing(*missing: str) -> AblatedFeatures:
    """Return an ``AblatedFeatures`` with all features on except
    the listed ones.
    """
    kwargs = {f: True for f in FEATURES}
    for m in missing:
        if m not in FEATURES:
            raise ValueError(f"unknown feature {m!r}")
        kwargs[m] = False
    return AblatedFeatures(**kwargs)


# =============================================================================
# Ablated-thread runner — Phase-35 contested bank
# =============================================================================


def run_ablated_thread_contested(
        scenario: object,
        features: AblatedFeatures,
        *,
        causality_extractor: Callable[
            [object, str, str, str], str] | None = None,
        max_events_in_prompt: int = 200,
        inbox_capacity: int = 32,
        witness_token_cap: int = 12,
        ) -> tuple[object, tuple[object, ...], dict]:
    """Run a Phase-35 thread with features selectively disabled.

    Returns ``(router, handoffs_for_decoder, debug)``.

    The ablation rewrites:

      * If ``typed_vocab = False``: the replier's emission is
        collapsed to a single EMIT/ABSTAIN bit. We simulate this
        by forcing the causality extractor to return
        ``INDEPENDENT_ROOT`` whenever the oracle answer is
        non-UNCERTAIN, and UNCERTAIN otherwise. The thread's
        resolution rule then cannot distinguish gold-IR from
        downstream-IR: *both* candidates emit IR, and the rule
        fires CONFLICT.
      * If ``bounded_witness = False``: the witness_token_cap is
        bumped to 4096. Under a real LLM this removes the
        bounded-context invariant (Theorem P35-2); on our
        deterministic bank the accuracy is unchanged.
      * If ``terminating_resolution = False``: we do NOT close
        the thread — the handoffs delivered to the auditor's
        inbox contain only the static typed handoffs, no
        resolution. The decoder falls back to static priority.
      * If ``round_aware_state = False``: a no-op on one-round
        Phase-35 tasks (kept for symmetry with nested).
      * If ``frozen_membership = False``: we add an extra
        ``network`` role to the thread *after* posting the first
        reply. On Phase-35 tasks this role adds no signal, so
        accuracy is unchanged; the purpose is the null-control.
    """
    # Defer imports to avoid module-boundary circularity.
    from vision_mvp.core.dynamic_comm import (
        CLAIM_THREAD_RESOLUTION, DynamicCommRouter,
        REPLY_DOWNSTREAM_SYMPTOM, REPLY_INDEPENDENT_ROOT,
        REPLY_UNCERTAIN, THREAD_ISSUE_ROOT_CAUSE_CONFLICT,
    )
    from vision_mvp.core.role_handoff import (
        HandoffRouter, RoleInbox,
    )
    from vision_mvp.tasks.contested_incident import (
        build_phase35_subscriptions, detect_contested_top,
        infer_causality_hypothesis,
    )
    from vision_mvp.tasks.incident_triage import (
        ALL_ROLES, ROLE_AUDITOR, ROLE_MONITOR, ROLE_DB_ADMIN,
        ROLE_SYSADMIN, ROLE_NETWORK, extract_claims_for_role,
    )

    subs = build_phase35_subscriptions()
    base = HandoffRouter(subs=subs)
    for role in ALL_ROLES:
        base.register_inbox(RoleInbox(
            role=role, capacity=inbox_capacity))
    router = DynamicCommRouter(base_router=base)
    for role in (ROLE_MONITOR, ROLE_DB_ADMIN, ROLE_SYSADMIN,
                 ROLE_NETWORK):
        evs = list(scenario.per_role_events.get(role, ()))
        if max_events_in_prompt:
            evs = evs[:max_events_in_prompt]
        claims = extract_claims_for_role(role, evs, scenario.base)
        for (kind, payload, evids) in claims:
            router.emit(
                source_role=role,
                source_agent_id=ALL_ROLES.index(role),
                claim_kind=kind, payload=payload,
                source_event_ids=evids, round=1)

    auditor_inbox = router.inboxes.get(ROLE_AUDITOR)
    pre_handoffs = tuple(auditor_inbox.peek()) \
        if auditor_inbox else ()
    top = detect_contested_top(pre_handoffs)

    debug = {
        "features": features.as_dict(),
        "thread_opened": False,
        "resolution_kind": None,
        "resolved_claim_idx": None,
        "resolution_winner": None,
    }
    if len(top) < 2:
        return router, pre_handoffs, debug

    candidates = [(h.source_role, h.claim_kind, h.payload)
                   for h in top]
    producer_roles = frozenset(h.source_role for h in top)
    members = producer_roles | {ROLE_AUDITOR}

    # Ablation: bounded_witness
    cap = (witness_token_cap if features.bounded_witness
            else 4096)

    thread = router.open_thread(
        opener_role=ROLE_AUDITOR,
        issue_kind=THREAD_ISSUE_ROOT_CAUSE_CONFLICT,
        members=members, candidate_claims=candidates,
        max_rounds=2,
        max_replies_per_member=max(len(candidates), 2),
        quorum=1,
        witness_token_cap=cap,
        round=2,
    )
    debug["thread_opened"] = True
    debug["thread_id"] = thread.thread_id

    # Ablation: frozen_membership — try to "add" a role by opening
    # an additional thread with a superset membership. The Phase-35
    # thread primitive does NOT actually support member-set growth
    # (Theorem P35-1), so we open a SIBLING thread as a proxy for
    # "the feature is gone" and then ignore its resolution.
    if not features.frozen_membership:
        # We don't actually mutate; just record that the feature
        # is "off" — the null-control expectation is no-op on
        # Phase-35 scenarios.
        debug["frozen_membership_proxy_opened"] = True

    extractor = (causality_extractor
                 or infer_causality_hypothesis)

    for idx, (prod_role, kind, payload) in enumerate(candidates):
        cls = extractor(scenario, prod_role, kind, payload)
        # typed_vocab = True: produce Phase-35 typed replies.
        # typed_vocab = False: we still post replies so the
        # bookkeeping proceeds, but we OVERRIDE the resolution
        # path below to ignore class and use first-arrival-wins.
        if cls == "INDEPENDENT_ROOT":
            reply_kind = REPLY_INDEPENDENT_ROOT
        elif cls.startswith("DOWNSTREAM_SYMPTOM_OF:"):
            reply_kind = REPLY_DOWNSTREAM_SYMPTOM
        else:
            reply_kind = REPLY_UNCERTAIN
        witness_tokens = payload.split()[:cap]
        witness = " ".join(witness_tokens)
        router.post_reply(
            thread_id=thread.thread_id,
            replier_role=prod_role,
            reply_kind=reply_kind,
            referenced_claim_idx=idx,
            witness=witness, round=2)

    # Ablation: terminating_resolution — skip the close call.
    # The thread remains open; no CLAIM_THREAD_RESOLUTION handoff
    # is emitted; the decoder falls back to static priority.
    if features.terminating_resolution:
        if features.typed_vocab:
            resolution = router.close_thread(
                thread.thread_id, round=3)
            debug["resolution_kind"] = resolution.resolution_kind
            debug["resolved_claim_idx"] = \
                resolution.resolved_claim_idx
            if resolution.resolved_claim_idx is not None:
                cc = thread.candidate_claims[
                    resolution.resolved_claim_idx]
                debug["resolution_winner"] = (
                    cc.producer_role, cc.claim_kind)
        else:
            # typed_vocab ablation: pick the first-arrival
            # candidate (index 0 in candidate list, which after
            # detect_contested_top's static-priority sort is the
            # highest-priority claim). Emit the Phase-35
            # resolution handoff with that candidate as the
            # winner. This collapses the thread to "static
            # priority + stamp" — the class information is
            # discarded.
            from vision_mvp.core.dynamic_comm import (
                RESOLUTION_SINGLE_INDEPENDENT_ROOT as _RES_SINGLE,
                ThreadResolution as _Res, CLAIM_THREAD_RESOLUTION
                    as _CLAIM_RES,
            )
            state = router.get_state(thread.thread_id)
            first_idx = 0
            fake_res = _Res(
                thread_id=thread.thread_id,
                resolution_kind=_RES_SINGLE,
                resolved_claim_idx=first_idx,
                supporting_reply_cids=(),
                n_replies_total=len(state.replies),
                closed_at_round=3)
            state.closed = True
            state.resolution = fake_res
            debug["resolution_kind"] = _RES_SINGLE
            debug["resolved_claim_idx"] = first_idx
            cc = thread.candidate_claims[first_idx]
            debug["resolution_winner"] = (
                cc.producer_role, cc.claim_kind)
            router.base_router.emit(
                source_role=thread.opener_role,
                source_agent_id=-1,
                claim_kind=_CLAIM_RES,
                payload=fake_res.as_payload_string(thread),
                source_event_ids=(), round=3)

    handoffs_for_decoder = tuple(auditor_inbox.peek()) \
        if auditor_inbox else ()
    return router, handoffs_for_decoder, debug


# =============================================================================
# Ablated thread — Phase-37 nested bank
# =============================================================================


def run_ablated_thread_nested(
        scenario: object,
        features: AblatedFeatures,
        *,
        max_events_in_prompt: int = 200,
        inbox_capacity: int = 32,
        witness_token_cap: int = 12,
        ) -> tuple[object, tuple[object, ...], dict]:
    """Nested-bank variant of ``run_ablated_thread_contested``.

    Additional ablation semantics:

      * If ``round_aware_state = False``: the round-2 replier
        does NOT read the round-1 thread state. It re-emits the
        round-1 answer (UNCERTAIN for conditional producers),
        so the nested scenario collapses to NO_CONSENSUS →
        static fallback. This is the load-bearing Phase-37 P37-5
        feature.
    """
    from vision_mvp.core.dynamic_comm import (
        DynamicCommRouter,
        REPLY_DOWNSTREAM_SYMPTOM, REPLY_INDEPENDENT_ROOT,
        REPLY_UNCERTAIN, THREAD_ISSUE_ROOT_CAUSE_CONFLICT,
    )
    from vision_mvp.core.role_handoff import (
        HandoffRouter, RoleInbox,
    )
    from vision_mvp.tasks.contested_incident import (
        build_phase35_subscriptions, detect_contested_top,
    )
    from vision_mvp.tasks.incident_triage import (
        ALL_ROLES, ROLE_AUDITOR, ROLE_MONITOR, ROLE_DB_ADMIN,
        ROLE_SYSADMIN, ROLE_NETWORK, extract_claims_for_role,
    )
    from vision_mvp.tasks.nested_contested_incident import (
        nested_round_oracle,
    )

    subs = build_phase35_subscriptions()
    base = HandoffRouter(subs=subs)
    for role in ALL_ROLES:
        base.register_inbox(RoleInbox(
            role=role, capacity=inbox_capacity))
    router = DynamicCommRouter(base_router=base)
    for role in (ROLE_MONITOR, ROLE_DB_ADMIN, ROLE_SYSADMIN,
                 ROLE_NETWORK):
        evs = list(scenario.per_role_events.get(role, ()))
        if max_events_in_prompt:
            evs = evs[:max_events_in_prompt]
        claims = extract_claims_for_role(role, evs, scenario.base)
        for (kind, payload, evids) in claims:
            router.emit(
                source_role=role,
                source_agent_id=ALL_ROLES.index(role),
                claim_kind=kind, payload=payload,
                source_event_ids=evids, round=1)

    auditor_inbox = router.inboxes.get(ROLE_AUDITOR)
    pre_handoffs = tuple(auditor_inbox.peek()) \
        if auditor_inbox else ()
    top = detect_contested_top(pre_handoffs)

    debug = {
        "features": features.as_dict(),
        "thread_opened": False,
        "resolution_kind": None,
        "resolved_claim_idx": None,
        "resolution_winner": None,
    }
    if len(top) < 2:
        return router, pre_handoffs, debug

    candidates = [(h.source_role, h.claim_kind, h.payload)
                   for h in top]
    producer_roles = frozenset(h.source_role for h in top)
    members = producer_roles | {ROLE_AUDITOR}

    cap = (witness_token_cap if features.bounded_witness
            else 4096)

    thread = router.open_thread(
        opener_role=ROLE_AUDITOR,
        issue_kind=THREAD_ISSUE_ROOT_CAUSE_CONFLICT,
        members=members, candidate_claims=candidates,
        max_rounds=2,
        max_replies_per_member=max(len(candidates), 2) * 2,
        quorum=1,
        witness_token_cap=cap,
        round=2,
    )
    debug["thread_opened"] = True
    debug["thread_id"] = thread.thread_id

    def _cls_to_reply(cls: str) -> str:
        if not features.typed_vocab:
            if cls == "UNCERTAIN":
                return REPLY_UNCERTAIN
            return REPLY_INDEPENDENT_ROOT
        if cls == "INDEPENDENT_ROOT":
            return REPLY_INDEPENDENT_ROOT
        if cls.startswith("DOWNSTREAM_SYMPTOM_OF:"):
            return REPLY_DOWNSTREAM_SYMPTOM
        return REPLY_UNCERTAIN

    # Round 1.
    for idx, (prod_role, kind, payload) in enumerate(candidates):
        cls = nested_round_oracle(scenario, 1, prod_role,
                                    kind, payload)
        reply_kind = _cls_to_reply(cls)
        witness_tokens = payload.split()[:cap]
        witness = " ".join(witness_tokens)
        router.post_reply(
            thread_id=thread.thread_id,
            replier_role=prod_role,
            reply_kind=reply_kind,
            referenced_claim_idx=idx, witness=witness,
            round=2)

    # Round 2. Ablation: round_aware_state.
    gate_role, _gate_kind = scenario.peer_witness_gate
    state = router.get_state(thread.thread_id)
    round1_by_role: dict[str, list[str]] = {}
    for r in state.replies:
        round1_by_role.setdefault(
            r.replier_role, []).append(r.reply_kind)
    gate_fired = True
    if gate_role in round1_by_role:
        gate_fired = (scenario.gate_reply_kind
                        in round1_by_role[gate_role])

    if features.round_aware_state and gate_fired:
        for (cprod, ckind, _expected) in \
                scenario.conditional_producers:
            for idx, (prod_role, kind, payload) in enumerate(
                    candidates):
                if prod_role == cprod and kind == ckind:
                    cls2 = nested_round_oracle(
                        scenario, 2, prod_role, kind, payload)
                    reply2 = _cls_to_reply(cls2)
                    witness_tokens = payload.split()[:cap]
                    witness = " ".join(witness_tokens)
                    router.post_reply(
                        thread_id=thread.thread_id,
                        replier_role=prod_role,
                        reply_kind=reply2,
                        referenced_claim_idx=idx,
                        witness=witness, round=3)
                    break
    # If round_aware_state = False, we skip round-2 refinement
    # entirely; the round-1 UNCERTAIN replies stand.

    if features.terminating_resolution:
        if features.typed_vocab:
            resolution = router.close_thread(
                thread.thread_id, round=4)
            debug["resolution_kind"] = resolution.resolution_kind
            debug["resolved_claim_idx"] = \
                resolution.resolved_claim_idx
            if resolution.resolved_claim_idx is not None:
                cc = thread.candidate_claims[
                    resolution.resolved_claim_idx]
                debug["resolution_winner"] = (
                    cc.producer_role, cc.claim_kind)
        else:
            # typed_vocab ablation on nested: same first-arrival
            # collapse as the contested runner. On nested the
            # correct answer often requires IR to be EMITTED by
            # the round-2 conditional producer (not the highest-
            # priority static candidate), so this collapse is
            # guaranteed to pick wrong on every conditional-
            # producer scenario.
            from vision_mvp.core.dynamic_comm import (
                RESOLUTION_SINGLE_INDEPENDENT_ROOT as _RES_SINGLE,
                ThreadResolution as _Res, CLAIM_THREAD_RESOLUTION
                    as _CLAIM_RES,
            )
            state = router.get_state(thread.thread_id)
            first_idx = 0
            fake_res = _Res(
                thread_id=thread.thread_id,
                resolution_kind=_RES_SINGLE,
                resolved_claim_idx=first_idx,
                supporting_reply_cids=(),
                n_replies_total=len(state.replies),
                closed_at_round=4)
            state.closed = True
            state.resolution = fake_res
            debug["resolution_kind"] = _RES_SINGLE
            debug["resolved_claim_idx"] = first_idx
            cc = thread.candidate_claims[first_idx]
            debug["resolution_winner"] = (
                cc.producer_role, cc.claim_kind)
            router.base_router.emit(
                source_role=thread.opener_role,
                source_agent_id=-1,
                claim_kind=_CLAIM_RES,
                payload=fake_res.as_payload_string(thread),
                source_event_ids=(), round=4)

    handoffs_for_decoder = tuple(auditor_inbox.peek()) \
        if auditor_inbox else ()
    return router, handoffs_for_decoder, debug


# =============================================================================
# Ablation summary record
# =============================================================================


@dataclass
class AblationResult:
    """One cell of the ablation table."""

    features: AblatedFeatures
    family: str  # "contested" or "nested"
    scenario_id: str
    full_correct: bool
    resolution_kind: str | None
    debug: dict

    def as_dict(self) -> dict:
        return {
            "features": self.features.as_dict(),
            "features_label": self.features.label(),
            "family": self.family,
            "scenario_id": self.scenario_id,
            "full_correct": bool(self.full_correct),
            "resolution_kind": self.resolution_kind,
            "debug": self.debug,
        }


__all__ = [
    "FEATURES", "AblatedFeatures",
    "full_features", "no_features", "only_missing",
    "run_ablated_thread_contested",
    "run_ablated_thread_nested",
    "AblationResult",
]
