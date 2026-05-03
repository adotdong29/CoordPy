"""Phase 53 — model-scale vs capsule-structure on multi-agent team coordination.

This is the SDK v3.7 reference benchmark. It directly attacks the
*main scientific question* surfaced by SDK v3.6's two-Mac-distributed
integration: **does scaling the underlying LLM narrow or preserve
the capsule-structure advantage measured by Phase-52?**

The Phase-52 benchmark (``phase52_team_coord``) drives the team
coordinator with a deterministic, symbol-level claim extractor and
a synthetic noise wrapper. Phase 53 replaces the deterministic
extractor with a **real LLM producer-role extractor** for each of
the four producer roles (monitor, db_admin, sysadmin, network) and
runs the same five admission strategies (substrate, capsule_fifo,
capsule_priority, capsule_coverage, capsule_learned) on the
LLM-generated candidate handoff stream. We sweep the *model regime*
(synthetic / qwen2.5:14b dense / qwen3.5:35b MoE) holding everything
else fixed and decompose:

* ``structure_gain[M]`` := capsule_learned_acc[M] - substrate_acc[M]
* ``scale_gain[S]``     := acc(35b)[S] - acc(14b)[S]

The headline question: does ``structure_gain`` collapse, persist,
or grow as the model gets larger?

Honest scope
------------

* This benchmark uses the **single-Mac stronger model class
  available today**: qwen2.5:14b-32k (14.8B dense, Q4_K_M) and
  qwen3.5:35b (36.0B MoE, Q4_K_M, ``think=False``) served by Mac 1
  Ollama at ``192.168.12.191:11434``. Mac 2 is offline at the time
  of this milestone (ARP "incomplete"); the two-Mac sharded
  70B-class run remains the operator step described in
  ``docs/MLX_DISTRIBUTED_RUNBOOK.md``. The W5-1 result says nothing
  in this measurement contradicts the prediction that 70B will
  produce a third data point on the same axis when Mac 2 returns.
* The bench size is small on purpose (``n_eval = 6`` scenarios per
  model regime by default; the Phase-52 default of 30 is too
  expensive at 35b). The signal we want is whether the
  ``structure_gain`` direction is preserved, not a tight magnitude
  estimate. Saturation is acceptable; reproducibility under the
  LLM's own determinism is the falsifier.
* The learned admission policy is trained ONCE on the Phase-52
  *synthetic* train partition, then evaluated out-of-distribution
  on every model regime's candidate stream. This is intentional:
  it tests whether the capsule-feature scorer transfers from
  symbol noise to real-LLM noise, which is itself a research
  question (W6-C2).
* ``temperature=0.0`` and ``max_tokens=128``. A larger budget
  changes nothing at the saturation level we observe; we keep it
  tight so the wall-clock is honest.
* The LLM is given a closed-vocabulary list of allowed claim kinds
  for its role. We do NOT teach the LLM the bench's ground-truth
  triggering rules; the LLM must infer the right kinds from
  natural-language event bodies.

Theorem cross-reference (W6 family)
-----------------------------------

This benchmark is the empirical evidence anchor for:

* **W6-1** (proved-empirical) — capsule-team lifecycle audit
  T-1..T-7 holds for every (model, strategy, scenario) triple.
* **W6-2** (proved-empirical) — capsule-native runtime accepts
  real-LLM candidate streams without spine modification.
* **W6-C1** (empirical-research) — structure_gain is preserved
  (or grows) when the underlying LLM scales up; falsifier:
  structure_gain[35b] ≤ 0 with substrate_acc[35b] ≥
  capsule_learned_acc[35b] - 0.05.
* **W6-C2** (research, empirical) — the per-role admission scorer
  trained on synthetic data transfers usefully (better than FIFO)
  to real-LLM candidate streams; falsifier: capsule_learned beats
  capsule_fifo by < 0.05 on average across all model regimes.
* **W6-C3** (empirical-research) — 36B-MoE producer roles emit a
  *different* candidate-handoff distribution than 14.8B-dense
  producer roles on the same scenario bank under matched prompt
  and temperature; falsifier: per-role TVD between candidate-kind
  histograms < 0.10.

See ``docs/THEOREM_REGISTRY.md`` for status updates after a run.

CLI
---

::

    python3 -m vision_mvp.experiments.phase53_scale_vs_structure \\
        --endpoint http://192.168.12.191:11434 \\
        --models synthetic,qwen2.5:14b-32k,qwen3.5:35b \\
        --n-eval 6 \\
        --out /tmp/coordpy-distributed/phase53_scale_vs_structure.json

The synthetic regime falls back to ``extract_claims_for_role`` and
makes zero LLM calls; it is the SDK v3.5 baseline for cross-check.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import random
import re
import sys
import time
from typing import Any, Sequence

from vision_mvp.core.extractor_noise import (
    NoiseConfig, incident_triage_known_kinds,
)
from vision_mvp.tasks.incident_triage import (
    ALL_ROLES, ROLE_AUDITOR, ROLE_DB_ADMIN, ROLE_MONITOR,
    ROLE_NETWORK, ROLE_SYSADMIN,
    IncidentEvent, IncidentScenario,
    build_scenario_bank, build_role_subscriptions,
    extract_claims_for_role, grade_answer,
)
from vision_mvp.coordpy.capsule import CapsuleKind, CapsuleLedger
from vision_mvp.coordpy.llm_backend import LLMBackend, OllamaBackend
from vision_mvp.coordpy.team_coord import (
    AdmissionPolicy, ClaimPriorityAdmissionPolicy,
    CoverageGuidedAdmissionPolicy, FifoAdmissionPolicy,
    RoleBudget, TeamCoordinator, audit_team_lifecycle,
    capsule_team_handoff,
)
from vision_mvp.coordpy.team_policy import (
    LearnedTeamAdmissionPolicy, TrainSample,
    train_team_admission_policy,
)
from vision_mvp.experiments.phase52_team_coord import (
    PooledStrategy, StrategyResult, _format_canonical_answer,
    build_candidate_handoff_stream as _phase52_build_candidates,
    build_training_samples as _phase52_build_training_samples,
    claim_priorities, expand_bank, make_team_budgets, pool,
    run_strategy as _phase52_run_strategy,
    run_substrate_baseline as _phase52_run_substrate_baseline,
    split_bank,
)


# =============================================================================
# Per-role allowed claim kinds — closed vocabulary
# =============================================================================

# One-line description per claim kind. The LLM is given this so it
# can interpret the event bodies. Wording is kept neutral and
# symmetric across kinds so neither model is given a leg up.
CLAIM_KIND_DESCRIPTIONS: dict[str, str] = {
    "ERROR_RATE_SPIKE":
        "the application error_rate metric jumped above the noise floor",
    "LATENCY_SPIKE":
        "the application p95 latency exceeded its SLO",
    "SLOW_QUERY_OBSERVED":
        "a database query is much slower than its baseline mean_ms",
    "POOL_EXHAUSTION":
        "the database connection pool is fully saturated",
    "DEADLOCK_SUSPECTED":
        "a database log mentions a deadlock condition",
    "DISK_FILL_CRITICAL":
        "an OS disk-usage event shows ≥90% full or out-of-space",
    "CRON_OVERRUN":
        "a cron job ran much longer than expected or exited error",
    "OOM_KILL":
        "the OS killed a process due to out-of-memory",
    "TLS_EXPIRED":
        "a firewall or network event indicates an expired TLS cert",
    "DNS_MISROUTE":
        "DNS queries returned SERVFAIL or are misrouted",
    "FW_BLOCK_SURGE":
        "a firewall rule is dropping a surge of packets",
}


def _role_prompt(role: str, events: Sequence[IncidentEvent],
                 known_kinds: Sequence[str],
                 max_claims: int = 6) -> str:
    """Render the producer-role LLM prompt for the real-LLM extractor.

    The prompt is symmetric across roles modulo the kind list and
    role identifier. Bodies are quoted verbatim so the LLM has the
    raw evidence needed to disambiguate noise from signal.
    """
    kind_lines = []
    for k in known_kinds:
        kind_lines.append(f"  - {k}: {CLAIM_KIND_DESCRIPTIONS.get(k, '')}")
    event_lines = []
    for i, ev in enumerate(events, start=1):
        body = (ev.body or "").replace("\n", " ").strip()
        if len(body) > 200:
            body = body[:200]
        event_lines.append(
            f"  [{i}] type={ev.event_type} body=\"{body}\"")
    return (
        f"You are the {role!r} agent in an incident response team. "
        f"You observe operational events and must emit structured "
        f"claims to the incident auditor.\n\n"
        f"Allowed claim kinds for the {role!r} role:\n"
        + "\n".join(kind_lines) + "\n\n"
        f"Events you observed:\n"
        + ("\n".join(event_lines) if event_lines else "  (none)") + "\n\n"
        f"For each claim you wish to emit, output ONE LINE in this "
        f"exact format:\n"
        f"KIND | one-line evidence summary including any "
        f"\"service=<name>\" token from the source event\n\n"
        f"Output rules:\n"
        f"- Use only KINDs from the allowed list above.\n"
        f"- One claim per line. Maximum {max_claims} lines.\n"
        f"- If no event warrants a claim, output exactly: NONE\n"
        f"- Output nothing except the claim lines (no preamble, "
        f"no explanation).\n\n"
        f"Begin output now:\n"
    )


_LINE_RE = re.compile(
    r"^\s*([A-Z][A-Z0-9_]*)\s*[|:\-–—]\s*(.+?)\s*$"
)


def parse_role_response(response: str,
                          allowed_kinds: Sequence[str],
                          max_claims: int = 6,
                          ) -> list[tuple[str, str]]:
    """Parse one role's LLM response into ``[(kind, payload), ...]``.

    Robust to verbose preamble, multiple separators (``|`` / ``:`` /
    ``-`` / em dash), repeated kinds, and the literal ``NONE``
    sentinel. Drops:

    * lines that do not match the ``KIND <sep> payload`` shape;
    * kinds outside the allowed list (closed vocabulary);
    * duplicate kinds (first wins).

    Caps the returned list at ``max_claims``.
    """
    if not response:
        return []
    allowed = set(allowed_kinds)
    out: list[tuple[str, str]] = []
    seen_kinds: set[str] = set()
    for raw in response.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.lstrip().startswith(("#", ">", "//")):
            continue
        if line.upper() == "NONE":
            # Sentinel meaning "no claim from this role". Skip rather
            # than early-return: if the LLM mixes NONE with real
            # claims (a verbose / confused output), the real claims
            # still survive. If NONE is the ONLY meaningful line,
            # the loop terminates with `out` empty, which is the
            # intended behaviour.
            continue
        m = _LINE_RE.match(line)
        if m is None:
            continue
        kind = m.group(1).strip().upper()
        payload = m.group(2).strip()
        if kind not in allowed:
            continue
        if kind in seen_kinds:
            continue
        if len(payload) > 240:
            payload = payload[:240]
        out.append((kind, payload))
        seen_kinds.add(kind)
        if len(out) >= max_claims:
            break
    return out


# =============================================================================
# Real-LLM extractor — per role
# =============================================================================


@dataclasses.dataclass
class LLMExtractorStats:
    """Per-(model, role) stats over an entire run."""

    model_tag: str
    n_calls: int = 0
    n_total_response_chars: int = 0
    n_total_response_lines: int = 0
    n_total_emitted_claims: int = 0
    n_total_dropped_unknown_kind: int = 0
    n_total_dropped_malformed: int = 0
    total_wall_s: float = 0.0
    n_failed_calls: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_tag": self.model_tag,
            "n_calls": self.n_calls,
            "n_total_response_chars": self.n_total_response_chars,
            "n_total_response_lines": self.n_total_response_lines,
            "n_total_emitted_claims": self.n_total_emitted_claims,
            "n_total_dropped_unknown_kind":
                self.n_total_dropped_unknown_kind,
            "n_total_dropped_malformed": self.n_total_dropped_malformed,
            "total_wall_s": round(self.total_wall_s, 3),
            "n_failed_calls": self.n_failed_calls,
        }


def extract_claims_for_role_via_llm(
        role: str,
        events: Sequence[IncidentEvent],
        scenario: IncidentScenario,
        *,
        backend: LLMBackend,
        max_tokens: int = 128,
        max_claims: int = 6,
        stats: LLMExtractorStats | None = None,
        ) -> list[tuple[str, str, tuple[int, ...]]]:
    """Real-LLM analogue of ``extract_claims_for_role``.

    Calls ``backend.generate(prompt)`` with the role's prompt
    (closed-vocabulary kind list + raw events). Parses the response
    via :func:`parse_role_response`. Attribution: the source event
    IDs of every emitted claim are the full event ID list of
    ``events``. This is consistent with the substrate baseline's
    coarse attribution and is sufficient for the team-level audit.

    Robustness: any HTTP / parse exception is caught and counted
    in ``stats.n_failed_calls``; the role then emits zero claims for
    the scenario. This degrades gracefully — the auditor will see
    a smaller candidate stream, which is itself a measurable
    consequence of LLM unreliability.
    """
    known = incident_triage_known_kinds()
    allowed = tuple(known.get(role, ()))
    if not allowed or not events:
        return []
    prompt = _role_prompt(role, events, allowed, max_claims=max_claims)
    t0 = time.time()
    try:
        response = backend.generate(
            prompt, max_tokens=max_tokens, temperature=0.0)
    except Exception:
        if stats is not None:
            stats.n_calls += 1
            stats.total_wall_s += time.time() - t0
            stats.n_failed_calls += 1
        return []
    elapsed = time.time() - t0
    if stats is not None:
        stats.n_calls += 1
        stats.total_wall_s += elapsed
        stats.n_total_response_chars += len(response or "")
        stats.n_total_response_lines += len(
            (response or "").splitlines())
    parsed = parse_role_response(
        response or "", allowed_kinds=allowed, max_claims=max_claims)
    if stats is not None:
        stats.n_total_emitted_claims += len(parsed)
        # Crude split: every line that started with an unknown kind
        # is an unknown-kind drop; every line that didn't match
        # the regex is a malformed drop.
        for raw in (response or "").splitlines():
            line = raw.strip()
            if not line or line.upper() == "NONE":
                continue
            m = _LINE_RE.match(line)
            if m is None:
                stats.n_total_dropped_malformed += 1
                continue
            kind = m.group(1).strip().upper()
            if kind not in set(allowed):
                stats.n_total_dropped_unknown_kind += 1
    evids = tuple(ev.event_id for ev in events)
    out: list[tuple[str, str, tuple[int, ...]]] = []
    for (kind, payload) in parsed:
        out.append((kind, payload, evids))
    return out


# =============================================================================
# Per-scenario candidate handoff stream — using the real-LLM extractor
# =============================================================================


def build_candidate_handoff_stream_via_llm(
        scenario: IncidentScenario,
        *,
        backend: LLMBackend,
        max_tokens: int = 128,
        max_claims: int = 6,
        stats: LLMExtractorStats | None = None,
        ) -> list[tuple[str, str, str, str, tuple[int, ...]]]:
    """Real-LLM analogue of
    :func:`vision_mvp.experiments.phase52_team_coord.build_candidate_handoff_stream`.

    Returns ``[(source_role, to_role, claim_kind, payload,
    source_event_ids), ...]``. For each producer role, calls the
    LLM extractor on the scenario's role-local events; routes the
    emitted claims to every consumer the static subscription table
    designates (typically the auditor, plus DBA for
    DISK_FILL_CRITICAL).
    """
    subs = build_role_subscriptions()
    out: list[tuple[str, str, str, str, tuple[int, ...]]] = []
    for role in (ROLE_MONITOR, ROLE_DB_ADMIN, ROLE_SYSADMIN, ROLE_NETWORK):
        evs = scenario.per_role_events.get(role, ())
        claims = extract_claims_for_role_via_llm(
            role, evs, scenario, backend=backend,
            max_tokens=max_tokens, max_claims=max_claims, stats=stats,
        )
        for (kind, payload, evids) in claims:
            consumers = subs.consumers(role, kind)
            if not consumers:
                continue
            for to_role in sorted(consumers):
                out.append((role, to_role, kind, payload, evids))
    return out


# =============================================================================
# Per-(model, scenario, strategy) run driver
# =============================================================================


def _run_capsule_strategy_with_candidates(
        scenario: IncidentScenario,
        candidates: Sequence[tuple[str, str, str, str, tuple[int, ...]]],
        budgets: dict[str, RoleBudget],
        policy_per_role: dict[str, AdmissionPolicy],
        strategy_name: str,
        ) -> StrategyResult:
    """Drive one (scenario, candidates, policy) combo and return a
    StrategyResult. Mirrors phase52.run_strategy but operates on a
    pre-built candidate stream so multiple strategies can share one
    LLM call set.
    """
    ledger = CapsuleLedger()
    coord = TeamCoordinator(
        ledger=ledger, role_budgets=budgets,
        policy_per_role=policy_per_role,
        team_tag="incident_triage",
    )
    coord.advance_round(1)
    for (src, to, kind, payload, _evs) in candidates:
        coord.emit_handoff(
            source_role=src, to_role=to, claim_kind=kind, payload=payload)
    coord.seal_all_role_views()
    # Build the auditor's decision the same way phase52 does.
    rv_cid = coord.role_view_cid(ROLE_AUDITOR)
    handoffs: list[Any] = []
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
    from vision_mvp.tasks.incident_triage import (
        _decoder_from_handoffs as _phase52_decoder_from_handoffs,
    )
    answer = _phase52_decoder_from_handoffs(handoffs)
    coord.seal_team_decision(team_role=ROLE_AUDITOR, decision=answer)
    audit = audit_team_lifecycle(ledger)
    grading = grade_answer(scenario, _format_canonical_answer(answer))

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
                 for (role, kind, _p, _evs) in scenario.causal_chain}
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
        scenario_id=scenario.scenario_id,
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


@dataclasses.dataclass(frozen=True)
class _DecoderHandoffShim:
    source_role: str
    claim_kind: str
    payload: str
    n_tokens: int = 1


def _run_substrate_with_candidates(
        scenario: IncidentScenario,
        candidates: Sequence[tuple[str, str, str, str, tuple[int, ...]]],
        inbox_capacity: int,
        ) -> StrategyResult:
    """Drive the Phase-31 typed-handoff substrate at matched inbox
    capacity, replaying a pre-built candidate stream. Functionally
    equivalent to ``phase52.run_substrate_baseline`` modulo the
    extractor (here: real-LLM, externally pre-baked into
    ``candidates``).
    """
    from vision_mvp.core.role_handoff import (
        HandoffRouter, RoleInbox,
    )
    from vision_mvp.tasks.incident_triage import (
        _decoder_from_handoffs as _phase52_decoder_from_handoffs,
    )
    subs = build_role_subscriptions()
    router = HandoffRouter(subs=subs)
    for role in ALL_ROLES:
        router.register_inbox(RoleInbox(role=role, capacity=inbox_capacity))
    for (src, _to, kind, payload, evids) in candidates:
        router.emit(
            source_role=src,
            source_agent_id=ALL_ROLES.index(src),
            claim_kind=kind, payload=payload,
            source_event_ids=evids, round=1,
        )
    auditor_inbox = router.inboxes.get(ROLE_AUDITOR)
    held = tuple(auditor_inbox.peek()) if auditor_inbox else ()
    answer = _phase52_decoder_from_handoffs(held)
    grading = grade_answer(scenario, _format_canonical_answer(answer))
    admitted_kinds = {(h.source_role, h.claim_kind) for h in held}
    required = {(role, kind)
                 for (role, kind, _p, _evs) in scenario.causal_chain}
    if grading["full_correct"]:
        failure_kind = "none"
    elif required - admitted_kinds:
        failure_kind = "missing_handoff"
    else:
        failure_kind = "decoder_error"
    return StrategyResult(
        strategy="substrate",
        scenario_id=scenario.scenario_id,
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
# Cross-model TVD on candidate-kind histograms (W6-C3 evidence)
# =============================================================================


def _candidate_kind_hist(
        candidates_by_scenario: dict[str, list[
            tuple[str, str, str, str, tuple[int, ...]]]],
        ) -> dict[tuple[str, str], int]:
    """``(source_role, claim_kind) -> count`` over the full
    candidate stream pooled across scenarios.
    """
    hist: dict[tuple[str, str], int] = {}
    for cands in candidates_by_scenario.values():
        for (src, _to, kind, _p, _evs) in cands:
            hist[(src, kind)] = hist.get((src, kind), 0) + 1
    return hist


def _hist_tvd(a: dict[Any, int], b: dict[Any, int]) -> float:
    """Total variation distance between two count distributions
    over the same key space."""
    keys = set(a) | set(b)
    if not keys:
        return 0.0
    sa = max(1, sum(a.values()))
    sb = max(1, sum(b.values()))
    s = 0.0
    for k in keys:
        s += abs(a.get(k, 0) / sa - b.get(k, 0) / sb)
    return 0.5 * s


# =============================================================================
# Top-level driver — sweep across model regimes
# =============================================================================


@dataclasses.dataclass
class ModelRegime:
    """One row of the (model regime × strategy) grid."""

    name: str         # "synthetic" | concrete model tag
    backend: LLMBackend | None    # None for synthetic
    is_real_llm: bool
    cold_load_skipped: bool = False  # set True if the bench skipped this regime
    extractor_stats: LLMExtractorStats | None = None


def _build_synthetic_candidates(
        scenario: IncidentScenario,
        ) -> list[tuple[str, str, str, str, tuple[int, ...]]]:
    """Mirror Phase-52's deterministic extractor over the same
    candidate stream shape (no noise wrapper — pure deterministic
    baseline)."""
    out: list[tuple[str, str, str, str, tuple[int, ...]]] = []
    subs = build_role_subscriptions()
    for role in (ROLE_MONITOR, ROLE_DB_ADMIN, ROLE_SYSADMIN, ROLE_NETWORK):
        evs = scenario.per_role_events.get(role, ())
        for (kind, payload, evids) in extract_claims_for_role(
                role, evs, scenario):
            consumers = subs.consumers(role, kind)
            if not consumers:
                continue
            for to_role in sorted(consumers):
                out.append((role, to_role, kind, payload, tuple(evids)))
    return out


def run_phase53(
        endpoint: str = "http://192.168.12.191:11434",
        model_tags: Sequence[str] = ("synthetic", "qwen2.5:14b-32k",
                                       "qwen3.5:35b"),
        n_eval: int = 5,
        seeds: Sequence[int] = (31, 32, 33),
        distractors_per_role: Sequence[int] = (8,),
        K_auditor: int = 4,
        T_auditor: int = 128,
        train_seed: int = 0,
        train_epochs: int = 200,
        train_lr: float = 0.5,
        max_tokens: int = 128,
        max_claims: int = 6,
        timeout_s: float = 600.0,
        verbose: bool = True,
        ) -> dict[str, Any]:
    """Drive Phase 53 end-to-end across model regimes.

    Pipeline (per model regime):

    1. Train the learned admission policy ONCE on the Phase-52
       *synthetic* train partition (out-of-distribution test of
       transfer).
    2. For each evaluation scenario:
       (a) Build candidate-handoff stream via the regime-specific
            extractor (synthetic for ``"synthetic"``, real LLM
            otherwise).
       (b) For each strategy ∈ {substrate, capsule_fifo,
            capsule_priority, capsule_coverage, capsule_learned},
            replay the same candidate stream and grade the
            auditor's decision.
    3. Pool per-strategy stats; record per-(model, role) extractor
       stats; record cross-model candidate-kind histograms.
    4. Compute scale-vs-structure decomposition.

    Returns a JSON-serialisable report.
    """
    if verbose:
        print(f"[phase53] starting; endpoint={endpoint}, "
              f"models={list(model_tags)}, n_eval={n_eval}",
              file=sys.stderr, flush=True)

    # --- Train the learned policy ONCE on synthetic Phase-52 train data ---
    bank = expand_bank(seeds=seeds,
                        distractors_per_role=distractors_per_role)
    train, evald = split_bank(bank, train_fraction=0.66, seed=train_seed)
    if n_eval and len(evald) > n_eval:
        evald = list(evald)[:n_eval]
    if verbose:
        print(f"[phase53] train_n={len(train)}, eval_n={len(evald)}",
              file=sys.stderr, flush=True)
    noise = NoiseConfig(
        drop_prob=0.10, spurious_prob=0.30, mislabel_prob=0.05, seed=11)
    train_samples = _phase52_build_training_samples(train, noise)
    learned, train_stats = train_team_admission_policy(
        train_samples, epochs=train_epochs, lr=train_lr, seed=train_seed)

    budgets = make_team_budgets(K_auditor=K_auditor, T_auditor=T_auditor)
    priorities = claim_priorities()

    # --- Build the model regimes ---
    regimes: list[ModelRegime] = []
    for tag in model_tags:
        if tag == "synthetic":
            regimes.append(ModelRegime(
                name="synthetic", backend=None, is_real_llm=False))
        else:
            backend = OllamaBackend(
                model=tag, base_url=endpoint, timeout=timeout_s,
                think=(False if "qwen3" in tag else None),
            )
            regimes.append(ModelRegime(
                name=tag, backend=backend, is_real_llm=True,
                extractor_stats=LLMExtractorStats(model_tag=tag),
            ))

    # --- Per-regime candidate streams (LLM calls happen here once) ---
    cands_by_regime: dict[str, dict[str, list[
        tuple[str, str, str, str, tuple[int, ...]]]]] = {}
    for regime in regimes:
        cands_by_regime[regime.name] = {}
        if regime.is_real_llm:
            if verbose:
                print(f"[phase53] regime={regime.name}: "
                      f"calling LLM for {len(evald)} scenarios x "
                      f"4 producer roles…",
                      file=sys.stderr, flush=True)
            t0 = time.time()
            for sc in evald:
                cands = build_candidate_handoff_stream_via_llm(
                    sc, backend=regime.backend, max_tokens=max_tokens,
                    max_claims=max_claims, stats=regime.extractor_stats,
                )
                cands_by_regime[regime.name][sc.scenario_id] = cands
                if verbose:
                    print(
                        f"[phase53]   scenario={sc.scenario_id}: "
                        f"emitted {len(cands)} candidate handoffs",
                        file=sys.stderr, flush=True)
            if verbose:
                print(
                    f"[phase53] regime={regime.name}: "
                    f"LLM wall {time.time()-t0:.1f}s",
                    file=sys.stderr, flush=True)
        else:
            for sc in evald:
                cands_by_regime[regime.name][sc.scenario_id] = (
                    _build_synthetic_candidates(sc))

    # --- Strategies ---
    strategies: dict[str, dict[str, AdmissionPolicy]] = {
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
        "capsule_learned": {
            r: learned for r in budgets
        },
    }

    # --- Run all (regime × scenario × strategy) ---
    results_by_regime: dict[str, list[StrategyResult]] = {}
    for regime in regimes:
        results: list[StrategyResult] = []
        regime_cands = cands_by_regime[regime.name]
        for sc in evald:
            cands = regime_cands.get(sc.scenario_id, [])
            results.append(_run_substrate_with_candidates(
                sc, cands, K_auditor))
            for sname, policy_per_role in strategies.items():
                results.append(_run_capsule_strategy_with_candidates(
                    scenario=sc, candidates=cands,
                    budgets=budgets,
                    policy_per_role=policy_per_role,
                    strategy_name=sname,
                ))
        results_by_regime[regime.name] = results

    # --- Pool per (regime, strategy) ---
    pooled_per_regime: dict[str, dict[str, dict[str, Any]]] = {}
    strategy_names = ("substrate",) + tuple(strategies.keys())
    for regime in regimes:
        regime_results = results_by_regime[regime.name]
        pooled_per_regime[regime.name] = {
            s: pool(regime_results, s).as_dict()
            for s in strategy_names
        }

    # --- Decompose: scale_gain[strategy], structure_gain[regime] ---
    decomposition: dict[str, Any] = {
        "structure_gain": {},
        "scale_gain": {},
        "delta_with_scale": {},
    }
    for regime in regimes:
        sub_acc = pooled_per_regime[regime.name][
            "substrate"]["accuracy_full"]
        cap_acc = pooled_per_regime[regime.name][
            "capsule_learned"]["accuracy_full"]
        cap_root = pooled_per_regime[regime.name][
            "capsule_learned"]["accuracy_root_cause"]
        sub_root = pooled_per_regime[regime.name][
            "substrate"]["accuracy_root_cause"]
        decomposition["structure_gain"][regime.name] = {
            "accuracy_full": round(cap_acc - sub_acc, 4),
            "accuracy_root_cause": round(cap_root - sub_root, 4),
        }

    real_regimes = [r for r in regimes if r.is_real_llm]
    if len(real_regimes) >= 2:
        small = real_regimes[0].name
        large = real_regimes[-1].name
        for s in strategy_names:
            small_acc = pooled_per_regime[small][s]["accuracy_full"]
            large_acc = pooled_per_regime[large][s]["accuracy_full"]
            decomposition["scale_gain"][s] = {
                "small_model": small,
                "large_model": large,
                "accuracy_full_delta": round(large_acc - small_acc, 4),
            }
        small_struct = decomposition["structure_gain"][small][
            "accuracy_full"]
        large_struct = decomposition["structure_gain"][large][
            "accuracy_full"]
        decomposition["delta_with_scale"] = {
            "small_model": small,
            "large_model": large,
            "structure_gain_small": small_struct,
            "structure_gain_large": large_struct,
            "delta": round(large_struct - small_struct, 4),
        }

    # --- Cross-regime candidate-kind TVD (W6-C3) ---
    cross_tvd: dict[str, float] = {}
    if len(real_regimes) >= 2:
        for i, ra in enumerate(real_regimes):
            for rb in real_regimes[i + 1:]:
                ha = _candidate_kind_hist(cands_by_regime[ra.name])
                hb = _candidate_kind_hist(cands_by_regime[rb.name])
                cross_tvd[f"{ra.name}__vs__{rb.name}"] = round(
                    _hist_tvd(ha, hb), 4)

    # --- Audit OK rate across capsule strategies (W6-1) ---
    audit_ok_grid: dict[str, dict[str, bool]] = {}
    for regime in regimes:
        audit_ok_grid[regime.name] = {}
        for s in strategy_names:
            if s == "substrate":
                audit_ok_grid[regime.name][s] = False  # substrate not audited
                continue
            rs = [r for r in results_by_regime[regime.name]
                   if r.strategy == s]
            audit_ok_grid[regime.name][s] = (
                bool(rs) and all(r.audit_ok for r in rs))

    extractor_stats: dict[str, Any] = {}
    for regime in regimes:
        if regime.extractor_stats is not None:
            extractor_stats[regime.name] = (
                regime.extractor_stats.as_dict())

    return {
        "schema": "phase53.scale_vs_structure.v1",
        "config": {
            "endpoint": endpoint,
            "model_tags": list(model_tags),
            "n_eval": len(evald),
            "n_train": len(train_samples),
            "K_auditor": K_auditor,
            "T_auditor": T_auditor,
            "max_tokens": max_tokens,
            "max_claims": max_claims,
            "train_seed": train_seed,
            "train_epochs": train_epochs,
            "train_lr": train_lr,
            "seeds": list(seeds),
            "distractors_per_role": list(distractors_per_role),
            "noise_used_for_training": noise.as_dict(),
        },
        "train_stats": train_stats.as_dict(),
        "pooled_per_regime": pooled_per_regime,
        "decomposition": decomposition,
        "cross_regime_candidate_tvd": cross_tvd,
        "audit_ok_grid": audit_ok_grid,
        "extractor_stats": extractor_stats,
        "scenarios_evaluated": [sc.scenario_id for sc in evald],
        "n_results": sum(len(rs) for rs in results_by_regime.values()),
    }


# =============================================================================
# CLI
# =============================================================================


def _arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Phase 53 — model-scale vs capsule-structure on "
                    "multi-agent team coordination.")
    p.add_argument("--endpoint", type=str,
                    default="http://192.168.12.191:11434",
                    help="Ollama / OpenAI-compat endpoint base URL.")
    p.add_argument(
        "--models", type=str,
        default="synthetic,qwen2.5:14b-32k,qwen3.5:35b",
        help="comma-separated model regimes; 'synthetic' uses "
              "Phase-52's deterministic extractor.")
    p.add_argument("--n-eval", type=int, default=5,
                    help="number of eval scenarios per regime.")
    p.add_argument("--K-auditor", type=int, default=4)
    p.add_argument("--T-auditor", type=int, default=128)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--max-claims", type=int, default=6)
    p.add_argument("--out", type=str, default="",
                    help="output JSON path; '-' for stdout.")
    p.add_argument("--quiet", action="store_true",
                    help="silence progress output.")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _arg_parser().parse_args(argv)
    model_tags = tuple(t.strip() for t in args.models.split(",") if t.strip())
    report = run_phase53(
        endpoint=args.endpoint,
        model_tags=model_tags,
        n_eval=args.n_eval,
        K_auditor=args.K_auditor,
        T_auditor=args.T_auditor,
        max_tokens=args.max_tokens,
        max_claims=args.max_claims,
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
            print(f"[phase53] wrote {args.out}", file=sys.stderr)
    else:
        print(json.dumps(report["pooled_per_regime"], indent=2,
                          sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
