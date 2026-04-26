"""Phase 52 — capsule-native multi-agent team coordination benchmark.

This is the reference benchmark for SDK v3.5 (the multi-agent
capsule coordination research slice). It instantiates the
``TeamCoordinator`` from ``vision_mvp.wevra.team_coord`` on the
Phase-31 incident-triage bank under a controlled noise wrapper
(``vision_mvp.core.extractor_noise``) and compares four admission
policies head-to-head against the substrate typed-handoff baseline.

Why this benchmark
==================

The capsule layer's claim is *team-level coordination through
content-addressed, lifecycle-bounded, budget-bounded capsules*.
The natural falsification is:

    "Capsule-native coordination doesn't measurably help over
     typed handoffs at matched per-role budgets, so the new layer
     is bookkeeping without a result."

This benchmark tests that claim. Under noisy extraction (so each
role receives a real budget-pressured candidate stream), we
measure pooled team-decision accuracy + per-role budget use for:

* **substrate**           — Phase-31 typed handoff baseline (no
                              capsule layer admission). Substrate
                              FIFO inbox at the matched ``K_role``.
* **capsule_fifo**        — capsule-native + FIFO admission.
* **capsule_priority**    — capsule-native + claim-priority admission.
* **capsule_coverage**    — capsule-native + coverage-guided admission.
* **capsule_learned**     — capsule-native + learned admission policy
                              (per-role logistic regression trained on
                              a held-out scenario partition).

The benchmark reports:

* Pooled root-cause / services / remediation accuracy by strategy.
* Mean per-role admitted-handoff count and admitted-token total.
* Mean capsule-DAG header counts (n_team_handoff, n_role_view,
  n_team_decision).
* Mean ``team_lifecycle_audit.verdict == "OK"`` rate (always 1.0 for
  capsule strategies; substrate is not audited because it is not in
  the capsule ledger).
* Pooled failure-mode histogram.

Honest scope
============

* The benchmark is *team-internal*. It does not measure latency
  against a real LLM, network IO, or a real production inbox.
* Distractor noise is a parsimonious surrogate for real-LLM
  extractor noise; the same knobs apply with calibrated noise
  (Phase-32). The result *direction* is robust; the *magnitude*
  varies with calibration.
* The bank has 5 base scenarios, each replicated under multiple
  seeds + distractor counts to reach ``n_train = 60`` and
  ``n_eval = 30`` for the learned policy. The bank is not a
  proxy for real-world incident triage — it is a deterministic
  substrate for measuring the team-layer's *mechanism*.

Theorem cross-reference
=======================

Empirical evidence supporting **W4-1** (mechanically-checked
audit), **W4-2** (coverage-implies-correctness), **W4-3**
(local-view limitation), and the conjecture **W4-C1** (learned
policy beats best fixed policy at matched budgets). Statuses live
in ``docs/THEOREM_REGISTRY.md``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
from typing import Any, Sequence

from vision_mvp.core.extractor_noise import (
    NoiseConfig, incident_triage_known_kinds, noisy_extractor,
)
from vision_mvp.core.role_handoff import (
    HandoffRouter, RoleInbox, RoleSubscriptionTable,
)
from vision_mvp.tasks.incident_triage import (
    ALL_ROLES, ROLE_AUDITOR, ROLE_DB_ADMIN, ROLE_MONITOR,
    ROLE_NETWORK, ROLE_SYSADMIN,
    IncidentScenario, _decoder_from_handoffs,
    build_role_subscriptions, build_scenario_bank,
    extract_claims_for_role, fixed_point_events,
    grade_answer, attribute_failure,
    naive_event_stream,
)
from vision_mvp.wevra.capsule import (
    CapsuleKind, CapsuleLedger,
)
from vision_mvp.wevra.team_coord import (
    AdmissionPolicy, ClaimPriorityAdmissionPolicy,
    CoverageGuidedAdmissionPolicy, FifoAdmissionPolicy,
    RoleBudget, TeamCoordinator, audit_team_lifecycle,
    capsule_team_handoff,
)
from vision_mvp.wevra.team_policy import (
    LearnedTeamAdmissionPolicy, TrainSample,
    train_team_admission_policy,
)


# =============================================================================
# Per-role budget — uniform tight cap so policies are budget-bound
# =============================================================================


def make_team_budgets(K_producer: int = 6, T_producer: int = 96,
                       K_auditor: int = 8, T_auditor: int = 256,
                       ) -> dict[str, RoleBudget]:
    """Default tight per-role budgets for the team benchmark.

    The auditor budget is the load-bearing knob: it is the role that
    derives the team decision and therefore must admit a coverage-
    sufficient subset of handoffs. Producers are given small budgets
    only for symmetry; producers do not consume handoffs in this
    bench (only claim-emit), so their budgets do not affect outcomes
    materially.
    """
    return {
        ROLE_MONITOR:  RoleBudget(role=ROLE_MONITOR,  K_role=K_producer,
                                    T_role=T_producer),
        ROLE_DB_ADMIN: RoleBudget(role=ROLE_DB_ADMIN, K_role=K_producer,
                                    T_role=T_producer),
        ROLE_SYSADMIN: RoleBudget(role=ROLE_SYSADMIN, K_role=K_producer,
                                    T_role=T_producer),
        ROLE_NETWORK:  RoleBudget(role=ROLE_NETWORK,  K_role=K_producer,
                                    T_role=T_producer),
        ROLE_AUDITOR:  RoleBudget(role=ROLE_AUDITOR,  K_role=K_auditor,
                                    T_role=T_auditor),
    }


# =============================================================================
# Train/eval scenario bank — a deterministic expansion of the 5-base bank
# under multiple seeds + distractor counts so the learned policy has a
# meaningful train partition
# =============================================================================


def expand_bank(seeds: Sequence[int] = (31, 32, 33, 34, 35, 36),
                  distractors_per_role: Sequence[int] = (4, 8, 12),
                  ) -> list[IncidentScenario]:
    """Cartesian product of the base bank over (seed × distractors).

    With defaults: 5 scenarios × 6 seeds × 3 distractor levels = 90
    scenarios. Sufficient for the learned-policy contrast; small
    enough to run end-to-end in seconds.
    """
    out: list[IncidentScenario] = []
    for d in distractors_per_role:
        for s in seeds:
            out.extend(build_scenario_bank(
                seed=s, distractors_per_role=d))
    return out


def split_bank(bank: Sequence[IncidentScenario], train_fraction: float = 0.66,
                seed: int = 0) -> tuple[list[IncidentScenario],
                                          list[IncidentScenario]]:
    """Deterministic shuffle + split."""
    rng = random.Random(seed)
    idx = list(range(len(bank)))
    rng.shuffle(idx)
    n_train = int(round(train_fraction * len(bank)))
    train_idx = idx[:n_train]
    eval_idx = idx[n_train:]
    train = [bank[i] for i in train_idx]
    evald = [bank[i] for i in eval_idx]
    return train, evald


# =============================================================================
# Build the candidate handoff stream for one scenario
# =============================================================================


def build_candidate_handoff_stream(
        scenario: IncidentScenario,
        noise: NoiseConfig,
        ) -> list[tuple[str, str, str, str, tuple[int, ...]]]:
    """Return ``[(source_role, to_role, claim_kind, payload,
    source_event_ids), ...]`` — the candidate handoffs that each
    producer role would emit under noisy extraction, all destined
    for the auditor (the deciding role).

    Implementation: run ``extract_claims_for_role`` (wrapped by
    ``noisy_extractor`` if ``noise`` is non-identity) for each
    producer role on the scenario's events, then route every claim
    to the auditor (canonical for this bench). Producers also
    forward ``DISK_FILL_CRITICAL`` to the DBA per the static
    subscription table; we model that as a second handoff with
    ``to_role=db_admin``.
    """
    known = incident_triage_known_kinds()
    extractor = (noisy_extractor(extract_claims_for_role, known, noise)
                 if not noise.is_identity()
                 else extract_claims_for_role)
    subs = build_role_subscriptions()
    out: list[tuple[str, str, str, str, tuple[int, ...]]] = []
    for role in (ROLE_MONITOR, ROLE_DB_ADMIN, ROLE_SYSADMIN, ROLE_NETWORK):
        evs = scenario.per_role_events.get(role, ())
        for (kind, payload, evids) in extractor(role, evs, scenario):
            consumers = subs.consumers(role, kind)
            if not consumers:
                continue
            for to_role in sorted(consumers):
                out.append((role, to_role, kind, payload, tuple(evids)))
    return out


def is_causal_handoff(source_role: str, claim_kind: str,
                       payload: str,
                       scenario: IncidentScenario) -> bool:
    """Per-handoff oracle: True iff this (role, kind, payload) is
    on the scenario's causal chain."""
    pairs = {(role, kind, p)
             for (role, kind, p, _evs) in scenario.causal_chain}
    if (source_role, claim_kind, payload) in pairs:
        return True
    # Loose membership: same (source_role, claim_kind) is enough to
    # consider the handoff causally-relevant.
    role_kind_pairs = {(role, kind)
                       for (role, kind, _p, _evs) in scenario.causal_chain}
    return (source_role, claim_kind) in role_kind_pairs


# =============================================================================
# Derive the deterministic team decision from a coordinator's auditor view
# =============================================================================


# A tiny shim that gives ``_decoder_from_handoffs`` something it can
# consume without depending on the substrate ``TypedHandoff`` shape.
@dataclasses.dataclass(frozen=True)
class _DecoderHandoff:
    source_role: str
    claim_kind: str
    payload: str
    n_tokens: int = 1


def _decision_from_capsule_view(coord: TeamCoordinator,
                                 ledger: CapsuleLedger,
                                 ) -> dict[str, Any]:
    """Build the auditor's deterministic decision from the auditor's
    sealed ROLE_VIEW + the parent TEAM_HANDOFF capsules."""
    rv_cid = coord.role_view_cid(ROLE_AUDITOR)
    if not rv_cid or rv_cid not in ledger:
        return {"root_cause": "unknown", "services": (),
                 "remediation": "investigate"}
    rv = ledger.get(rv_cid)
    handoffs: list[_DecoderHandoff] = []
    for p in rv.parents:
        if p in ledger:
            cap = ledger.get(p)
            if cap.kind != CapsuleKind.TEAM_HANDOFF:
                continue
            payload = cap.payload if isinstance(cap.payload, dict) else {}
            handoffs.append(_DecoderHandoff(
                source_role=str(payload.get("source_role", "")),
                claim_kind=str(payload.get("claim_kind", "")),
                payload=str(payload.get("payload", "")),
                n_tokens=int(payload.get("n_tokens", 1)),
            ))
    return _decoder_from_handoffs(handoffs)


# =============================================================================
# Per-strategy run driver
# =============================================================================


@dataclasses.dataclass
class StrategyResult:
    strategy: str
    scenario_id: str
    answer: dict[str, Any]
    grading: dict[str, Any]
    failure_kind: str
    n_admitted_auditor: int
    n_dropped_auditor_budget: int
    n_dropped_auditor_capacity: int
    n_dropped_auditor_unknown_kind: int
    n_team_handoff: int
    n_role_view: int
    n_team_decision: int
    audit_ok: bool
    n_tokens_admitted: int


def _format_evidence(handoffs: Sequence[_DecoderHandoff]) -> str:
    return "; ".join(f"{h.source_role}/{h.claim_kind}" for h in handoffs)


def run_strategy(scenario: IncidentScenario,
                  noise: NoiseConfig,
                  budgets: dict[str, RoleBudget],
                  policy_per_role: dict[str, AdmissionPolicy],
                  strategy_name: str,
                  ) -> StrategyResult:
    """Drive one (scenario, policy) combo and return a
    StrategyResult."""
    candidates = build_candidate_handoff_stream(scenario, noise)
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
    answer = _decision_from_capsule_view(coord, ledger)
    decision_caps = coord.seal_team_decision(
        team_role=ROLE_AUDITOR, decision=answer)
    audit = audit_team_lifecycle(ledger)
    grading = grade_answer(scenario, _format_canonical_answer(answer))
    # Failure attribution — adapted to capsule-native:
    # If full_correct: NONE. Else: missing handoff iff a required
    # (role, kind) pair is missing from the auditor's admitted set;
    # else "decoder_error".
    rv_cid = coord.role_view_cid(ROLE_AUDITOR)
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


def _format_canonical_answer(answer: dict[str, Any]) -> str:
    return (f"ROOT_CAUSE: {answer.get('root_cause', '')}\n"
             f"SERVICES: {','.join(answer.get('services', ()))}\n"
             f"REMEDIATION: {answer.get('remediation', '')}\n")


# =============================================================================
# Substrate baseline — runs the typed-handoff substrate exactly as
# Phase-31 ships it, so we have an apples-to-apples reference
# =============================================================================


def run_substrate_baseline(scenario: IncidentScenario,
                            noise: NoiseConfig,
                            inbox_capacity: int,
                            ) -> StrategyResult:
    """Drive the Phase-31 typed-handoff substrate at matched inbox
    capacity. Failure attribution is adapted to the capsule
    failure-mode vocabulary."""
    known = incident_triage_known_kinds()
    extractor = (noisy_extractor(extract_claims_for_role, known, noise)
                 if not noise.is_identity()
                 else extract_claims_for_role)
    subs = build_role_subscriptions()
    router = HandoffRouter(subs=subs)
    for role in ALL_ROLES:
        router.register_inbox(RoleInbox(role=role, capacity=inbox_capacity))
    for role in (ROLE_MONITOR, ROLE_DB_ADMIN, ROLE_SYSADMIN, ROLE_NETWORK):
        evs = scenario.per_role_events.get(role, ())
        for (kind, payload, evids) in extractor(role, evs, scenario):
            router.emit(
                source_role=role,
                source_agent_id=ALL_ROLES.index(role),
                claim_kind=kind, payload=payload,
                source_event_ids=evids, round=1)
    auditor_inbox = router.inboxes.get(ROLE_AUDITOR)
    held = tuple(auditor_inbox.peek()) if auditor_inbox else ()
    answer = _decoder_from_handoffs(held)
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
        n_team_handoff=0,  # substrate is not in the capsule ledger
        n_role_view=0,
        n_team_decision=0,
        audit_ok=False,    # substrate has no team-lifecycle audit
        n_tokens_admitted=sum(h.n_tokens for h in held),
    )


# =============================================================================
# Build labelled training samples for the learned policy
# =============================================================================


def build_training_samples(scenarios: Sequence[IncidentScenario],
                            noise: NoiseConfig,
                            ) -> list[TrainSample]:
    """For each scenario × candidate-handoff: build a TrainSample
    labelled by the causal-chain oracle. Only auditor-targeted
    handoffs are kept (the deciding role)."""
    out: list[TrainSample] = []
    for sc in scenarios:
        candidates = build_candidate_handoff_stream(sc, noise)
        for (src, to, kind, payload, _evs) in candidates:
            if to != ROLE_AUDITOR:
                continue
            cap = capsule_team_handoff(
                source_role=src, to_role=to,
                claim_kind=kind, payload=payload, round=1)
            label = 1 if is_causal_handoff(src, kind, payload, sc) else 0
            out.append(TrainSample(
                role=ROLE_AUDITOR, capsule=cap, label=label))
    return out


# =============================================================================
# Static priorities — used by ClaimPriorityAdmissionPolicy
# =============================================================================


def claim_priorities() -> dict[str, float]:
    """Reflect the priority order in
    ``incident_triage._decoder_from_handoffs``: causal-primary
    claims get higher scores. Hand-tuned but kept honest by the
    benchmark — if the learned policy doesn't beat this, the
    learning was a no-op."""
    return {
        "DISK_FILL_CRITICAL": 1.0,
        "TLS_EXPIRED": 1.0,
        "DNS_MISROUTE": 1.0,
        "OOM_KILL": 1.0,
        "DEADLOCK_SUSPECTED": 0.95,
        "CRON_OVERRUN": 0.9,
        "POOL_EXHAUSTION": 0.85,
        "SLOW_QUERY_OBSERVED": 0.8,
        "ERROR_RATE_SPIKE": 0.7,
        "LATENCY_SPIKE": 0.7,
        "FW_BLOCK_SURGE": 0.5,
    }


# =============================================================================
# Pooled-result aggregation
# =============================================================================


@dataclasses.dataclass
class PooledStrategy:
    strategy: str
    n: int
    accuracy_full: float
    accuracy_root_cause: float
    accuracy_services: float
    accuracy_remediation: float
    mean_n_admitted_auditor: float
    mean_tokens_admitted: float
    mean_n_team_handoff: float
    mean_n_role_view: float
    audit_ok_rate: float
    failure_hist: dict[str, int]

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "n": self.n,
            "accuracy_full": round(self.accuracy_full, 4),
            "accuracy_root_cause": round(self.accuracy_root_cause, 4),
            "accuracy_services": round(self.accuracy_services, 4),
            "accuracy_remediation": round(self.accuracy_remediation, 4),
            "mean_n_admitted_auditor": round(self.mean_n_admitted_auditor, 3),
            "mean_tokens_admitted": round(self.mean_tokens_admitted, 3),
            "mean_n_team_handoff": round(self.mean_n_team_handoff, 3),
            "mean_n_role_view": round(self.mean_n_role_view, 3),
            "audit_ok_rate": round(self.audit_ok_rate, 4),
            "failure_hist": dict(self.failure_hist),
        }


def pool(results: Sequence[StrategyResult],
          strategy: str) -> PooledStrategy:
    rs = [r for r in results if r.strategy == strategy]
    n = max(1, len(rs))
    f_hist: dict[str, int] = {}
    for r in rs:
        f_hist[r.failure_kind] = f_hist.get(r.failure_kind, 0) + 1
    return PooledStrategy(
        strategy=strategy,
        n=len(rs),
        accuracy_full=sum(1 for r in rs if r.grading["full_correct"]) / n,
        accuracy_root_cause=sum(
            1 for r in rs if r.grading["root_cause_correct"]) / n,
        accuracy_services=sum(
            1 for r in rs if r.grading["services_correct"]) / n,
        accuracy_remediation=sum(
            1 for r in rs if r.grading["remediation_correct"]) / n,
        mean_n_admitted_auditor=sum(r.n_admitted_auditor for r in rs) / n,
        mean_tokens_admitted=sum(r.n_tokens_admitted for r in rs) / n,
        mean_n_team_handoff=sum(r.n_team_handoff for r in rs) / n,
        mean_n_role_view=sum(r.n_role_view for r in rs) / n,
        audit_ok_rate=sum(1 for r in rs if r.audit_ok) / n,
        failure_hist=f_hist,
    )


# =============================================================================
# Top-level driver
# =============================================================================


def run_phase52(seeds: Sequence[int] = (31, 32, 33, 34, 35, 36),
                  distractors_per_role: Sequence[int] = (4, 8, 12),
                  noise: NoiseConfig | None = None,
                  K_auditor: int = 8,
                  T_auditor: int = 256,
                  inbox_capacity: int | None = None,
                  train_seed: int = 0,
                  train_epochs: int = 200,
                  train_lr: float = 0.5,
                  ) -> dict[str, Any]:
    """Run the full benchmark and return a JSON-serialisable
    report dict."""
    if noise is None:
        noise = NoiseConfig(
            drop_prob=0.10, spurious_prob=0.30, mislabel_prob=0.05,
            seed=11)
    if inbox_capacity is None:
        inbox_capacity = K_auditor
    bank = expand_bank(seeds=seeds,
                        distractors_per_role=distractors_per_role)
    train, evald = split_bank(bank, train_fraction=0.66, seed=train_seed)

    # Train the learned policy on the train partition.
    samples = build_training_samples(train, noise)
    learned, train_stats = train_team_admission_policy(
        samples, epochs=train_epochs, lr=train_lr, seed=train_seed)
    budgets = make_team_budgets(K_auditor=K_auditor, T_auditor=T_auditor)
    priorities = claim_priorities()

    # Strategies to evaluate on the eval partition.
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

    results: list[StrategyResult] = []
    for sc in evald:
        # Substrate baseline (no capsule layer).
        results.append(run_substrate_baseline(sc, noise, inbox_capacity))
        # Capsule strategies.
        for sname, policy_per_role in strategies.items():
            results.append(run_strategy(
                scenario=sc, noise=noise, budgets=budgets,
                policy_per_role=policy_per_role,
                strategy_name=sname))
    pooled = {s: pool(results, s).as_dict()
              for s in ("substrate",) + tuple(strategies.keys())}
    return {
        "schema": "phase52.team_coord.v1",
        "config": {
            "n_train": len(train),
            "n_eval": len(evald),
            "noise": noise.as_dict(),
            "K_auditor": K_auditor,
            "T_auditor": T_auditor,
            "inbox_capacity": inbox_capacity,
            "train_seed": train_seed,
            "train_epochs": train_epochs,
            "train_lr": train_lr,
        },
        "train_stats": train_stats.as_dict(),
        "pooled": pooled,
        "n_results": len(results),
    }


# =============================================================================
# CLI
# =============================================================================


def run_phase52_budget_sweep(
        K_values: Sequence[int] = (4, 6, 8, 12, 16),
        seeds: Sequence[int] = (31, 32, 33, 34),
        distractors_per_role: Sequence[int] = (8, 12),
        noise: NoiseConfig | None = None,
        ) -> dict[str, Any]:
    """Sweep ``K_auditor`` to provide empirical support for W4-3
    (local-view limitation): per-role budget below the role's
    causal-share floor admits sound runs that fail the team gate.

    Returns: {schema, K_values, pooled_per_K, n_eval}.
    """
    if noise is None:
        noise = NoiseConfig(
            drop_prob=0.10, spurious_prob=0.30, mislabel_prob=0.05,
            seed=11)
    pooled_per_K: dict[int, dict[str, dict[str, Any]]] = {}
    n_eval = 0
    for K in K_values:
        rep = run_phase52(
            seeds=seeds, distractors_per_role=distractors_per_role,
            noise=noise, K_auditor=K, T_auditor=max(64, 32 * K),
            inbox_capacity=K)
        pooled_per_K[K] = rep["pooled"]
        n_eval = rep["config"]["n_eval"]
    return {
        "schema": "phase52.team_coord_budget_sweep.v1",
        "K_values": list(K_values),
        "pooled_per_K": pooled_per_K,
        "n_eval": n_eval,
        "noise": noise.as_dict(),
    }


def _arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Phase 52 — capsule-native team coordination benchmark.")
    p.add_argument("--K-auditor", type=int, default=8,
                    help="auditor's K_role (max admitted handoffs).")
    p.add_argument("--T-auditor", type=int, default=256,
                    help="auditor's T_role (max admitted token total).")
    p.add_argument("--noise-spurious", type=float, default=0.30)
    p.add_argument("--noise-drop", type=float, default=0.10)
    p.add_argument("--noise-mislabel", type=float, default=0.05)
    p.add_argument("--out", type=str, default="",
                    help="output JSON path; '-' for stdout, '' for none.")
    p.add_argument("--budget-sweep", action="store_true",
                    help="run a K_auditor budget sweep (W4-3 evidence).")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _arg_parser().parse_args(argv)
    noise = NoiseConfig(
        drop_prob=args.noise_drop,
        spurious_prob=args.noise_spurious,
        mislabel_prob=args.noise_mislabel,
        seed=11,
    )
    if args.budget_sweep:
        report = run_phase52_budget_sweep(noise=noise)
    else:
        report = run_phase52(
            K_auditor=args.K_auditor, T_auditor=args.T_auditor, noise=noise)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out == "-":
        print(text)
    elif args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        print(text if args.budget_sweep
              else json.dumps(report["pooled"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
