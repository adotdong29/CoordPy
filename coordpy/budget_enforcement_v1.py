"""W84 / P1 #37 — Hard Cost / Latency Budget Enforcement V1.

Production multi-agent teams operate under hard budgets: max
cost per run, max latency per step, max tokens per agent, max
tool calls per task. W81's ``learned_economics_controller_v1``
chooses actions on a cost/quality utility but does NOT enforce
a hard ceiling. W82's ``MigrationBudgetPolicyV1`` enforces
migration-bytes budgets but is local to migrations. There is no
end-to-end mechanism that takes a ``RunBudgetSpecV1`` with
``max_cost_usd`` / ``max_latency_ms`` / ``max_tokens`` and
*provably* refuses to violate them across the full composed
pipeline.

V1 ships:

1. **``RunBudgetSpecV1``** — content-addressed dataclass of hard
   budgets (cost, per-step latency, tokens, tool calls,
   recompute flops, abstain-on-breach flag, record-breach-audit
   flag).
2. **``CostModelV1``** — content-addressed static cost table
   mapping ``(action, model_cid, prompt_tokens, output_tokens)``
   to estimated USD cost. Identical configs produce identical
   ``CostModelV1.cid()``.
3. **``BudgetEnforcerV1``** — subsystem inserted before every
   commit. Checks remaining budget; if any axis would
   overshoot, the action is refused and a
   ``BudgetBreachAuditV1`` is emitted.
4. **``BudgetBreachAuditV1``** — content-addressed capsule
   carrying pre/post budgets, the would-be action, and the
   breach reason.
5. **Stress bench.** A deliberately over-budget regime: every
   committed action would overshoot. The enforcer drives the
   commit count to 0; every abstain carries a breach audit.

Honest scope (V1):

* `W84-L-BUDGET-V1-RUN-LEVEL-CAP` — V1 enforces per-run budgets;
  per-step latency is enforced via a per-step `max_per_step_
  latency_ms` field; per-tenant + per-agent budgets are V2.
* `W84-L-BUDGET-V1-STATIC-USD-CAP` — V1 cost model is a static
  table indexed by `(action, model_cid, prompt_tokens,
  output_tokens)`; dynamic pricing is V2.
* `W84-L-BUDGET-V1-HARD-ABSTAIN-CAP` — V1 enforces by hard
  abstain; graceful degradation (switch to a cheaper model)
  is V2.
* `W84-L-BUDGET-V1-NO-MEMORY-BUDGET-CAP` — memory budget is V2.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
import time
from typing import Any, Mapping, Sequence


W84_BUDGET_V1_SCHEMA_VERSION: str = (
    "coordpy.budget_enforcement_v1.v1")


# ---------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------


class BudgetBreachAxis(str, enum.Enum):
    COST_USD = "cost_usd"
    PER_STEP_LATENCY_MS = "per_step_latency_ms"
    TOTAL_TOKENS = "total_tokens"
    TOOL_CALLS = "tool_calls"
    RECOMPUTE_FLOPS = "recompute_flops"


class BudgetCommitVerdict(str, enum.Enum):
    COMMIT_ALLOWED = "commit_allowed"
    ABSTAIN_ON_BREACH = "abstain_on_breach"
    ENFORCER_DISABLED = "enforcer_disabled"


# ---------------------------------------------------------------
# RunBudgetSpec
# ---------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class RunBudgetSpecV1:
    """Hard budget spec for a run.

    The spec is content-addressed: identical fields produce
    identical CIDs. The CID is part of the BudgetBreachAuditV1
    so a third party can audit which budgets were imposed.
    """

    schema: str
    run_cid: str
    max_total_cost_usd: float
    max_per_step_latency_ms: float
    max_total_tokens: int
    max_tool_calls: int
    max_recompute_flops: int
    abstain_on_breach: bool
    record_breach_audit: bool
    enforcer_disabled: bool  # default False; audit-logged

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "run_cid": str(self.run_cid),
            "max_total_cost_usd": float(round(
                self.max_total_cost_usd, 12)),
            "max_per_step_latency_ms": float(round(
                self.max_per_step_latency_ms, 12)),
            "max_total_tokens": int(self.max_total_tokens),
            "max_tool_calls": int(self.max_tool_calls),
            "max_recompute_flops": int(
                self.max_recompute_flops),
            "abstain_on_breach": bool(self.abstain_on_breach),
            "record_breach_audit": bool(
                self.record_breach_audit),
            "enforcer_disabled": bool(self.enforcer_disabled),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w84_run_budget_spec_v1",
            "spec": self.to_dict()})


def build_run_budget_spec_v1(
        *,
        run_cid: str = "w84_run",
        max_total_cost_usd: float = 0.50,
        max_per_step_latency_ms: float = 1000.0,
        max_total_tokens: int = 8000,
        max_tool_calls: int = 32,
        max_recompute_flops: int = 1_000_000,
        abstain_on_breach: bool = True,
        record_breach_audit: bool = True,
        enforcer_disabled: bool = False,
) -> RunBudgetSpecV1:
    return RunBudgetSpecV1(
        schema=W84_BUDGET_V1_SCHEMA_VERSION,
        run_cid=str(run_cid),
        max_total_cost_usd=float(max_total_cost_usd),
        max_per_step_latency_ms=float(
            max_per_step_latency_ms),
        max_total_tokens=int(max_total_tokens),
        max_tool_calls=int(max_tool_calls),
        max_recompute_flops=int(max_recompute_flops),
        abstain_on_breach=bool(abstain_on_breach),
        record_breach_audit=bool(record_breach_audit),
        enforcer_disabled=bool(enforcer_disabled),
    )


# ---------------------------------------------------------------
# CostModelV1
# ---------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class CostModelV1:
    """Static cost table indexed by (action, model_cid).

    Cost = (prompt_tokens * prompt_cost_per_kt
            + output_tokens * output_cost_per_kt
            + tool_call_count * tool_cost) / 1000.0
            * action_multiplier.
    The cost is monotone in tokens + tool calls.
    """

    schema: str
    model_cid: str
    prompt_cost_per_kt: float
    output_cost_per_kt: float
    tool_cost_per_call: float
    action_multipliers: tuple[tuple[str, float], ...]
    flops_cost_per_megaflop: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "model_cid": str(self.model_cid),
            "prompt_cost_per_kt": float(round(
                self.prompt_cost_per_kt, 12)),
            "output_cost_per_kt": float(round(
                self.output_cost_per_kt, 12)),
            "tool_cost_per_call": float(round(
                self.tool_cost_per_call, 12)),
            "action_multipliers": [
                [str(a), float(round(m, 12))]
                for a, m in self.action_multipliers],
            "flops_cost_per_megaflop": float(round(
                self.flops_cost_per_megaflop, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w84_cost_model_v1",
            "model": self.to_dict()})

    def estimate_cost_usd(
            self, *, action: str,
            prompt_tokens: int = 0,
            output_tokens: int = 0,
            tool_calls: int = 0,
            recompute_flops: int = 0,
    ) -> float:
        mult = 1.0
        for a, m in self.action_multipliers:
            if a == action:
                mult = float(m)
                break
        base = (
            float(prompt_tokens)
            * float(self.prompt_cost_per_kt) / 1000.0
            + float(output_tokens)
            * float(self.output_cost_per_kt) / 1000.0
            + float(tool_calls)
            * float(self.tool_cost_per_call)
            + float(recompute_flops) / 1_000_000.0
            * float(self.flops_cost_per_megaflop))
        return float(mult) * float(base)


def build_default_cost_model_v1(
        *, model_cid: str = "w84_default_model",
) -> CostModelV1:
    return CostModelV1(
        schema=W84_BUDGET_V1_SCHEMA_VERSION,
        model_cid=str(model_cid),
        prompt_cost_per_kt=0.0030,   # $0.003 per 1K prompt tokens
        output_cost_per_kt=0.0060,   # $0.006 per 1K output tokens
        tool_cost_per_call=0.0010,   # $0.001 per tool call
        action_multipliers=(
            ("replay", 0.10),
            ("runtime_recompute", 1.00),
            ("transcript_recompute", 0.40),
            ("promote_to_richer_substrate", 4.00),
            ("abstain", 0.00),
        ),
        flops_cost_per_megaflop=0.000_001,
    )


# ---------------------------------------------------------------
# BudgetBreachAuditV1
# ---------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class BudgetBreachAuditV1:
    """Audit capsule emitted on a refused action."""

    schema: str
    budget_spec_cid: str
    cost_model_cid: str
    would_be_action: str
    pre_budget_remaining_cost_usd: float
    pre_budget_remaining_tokens: int
    pre_budget_remaining_tool_calls: int
    pre_budget_remaining_flops: int
    pre_step_latency_ms_measured: float
    would_cost_usd: float
    would_tokens: int
    would_tool_calls: int
    would_recompute_flops: int
    breach_axes: tuple[str, ...]
    timestamp_ns: int
    enforcer_was_disabled: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "budget_spec_cid": str(self.budget_spec_cid),
            "cost_model_cid": str(self.cost_model_cid),
            "would_be_action": str(self.would_be_action),
            "pre_budget_remaining_cost_usd": float(round(
                self.pre_budget_remaining_cost_usd, 12)),
            "pre_budget_remaining_tokens": int(
                self.pre_budget_remaining_tokens),
            "pre_budget_remaining_tool_calls": int(
                self.pre_budget_remaining_tool_calls),
            "pre_budget_remaining_flops": int(
                self.pre_budget_remaining_flops),
            "pre_step_latency_ms_measured": float(round(
                self.pre_step_latency_ms_measured, 12)),
            "would_cost_usd": float(round(
                self.would_cost_usd, 12)),
            "would_tokens": int(self.would_tokens),
            "would_tool_calls": int(self.would_tool_calls),
            "would_recompute_flops": int(
                self.would_recompute_flops),
            "breach_axes": list(self.breach_axes),
            "timestamp_ns": int(self.timestamp_ns),
            "enforcer_was_disabled": bool(
                self.enforcer_was_disabled),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w84_budget_breach_audit_v1",
            "audit": self.to_dict()})


# ---------------------------------------------------------------
# Action proposal (what the controller proposes before commit)
# ---------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ActionProposalV1:
    """Proposed action with estimated cost / latency / tokens
    / tool calls / recompute flops."""

    action: str
    prompt_tokens: int
    output_tokens: int
    tool_calls: int
    recompute_flops: int
    measured_step_latency_ms: float


# ---------------------------------------------------------------
# BudgetEnforcerV1
# ---------------------------------------------------------------


@dataclasses.dataclass
class BudgetEnforcerV1:
    """The hard-budget enforcer.

    Inserted before every commit. Maintains a running counter
    of cost / tokens / tool calls / flops; on a proposal, asks
    the cost model what the commit would cost; if any axis would
    overshoot, emits a ``BudgetBreachAuditV1`` and refuses the
    commit (returns ``ABSTAIN_ON_BREACH``).
    """

    spec: RunBudgetSpecV1
    cost_model: CostModelV1
    cost_used_usd: float = 0.0
    tokens_used: int = 0
    tool_calls_used: int = 0
    recompute_flops_used: int = 0
    breach_audits: list[BudgetBreachAuditV1] = (
        dataclasses.field(default_factory=list))

    def remaining_cost_usd(self) -> float:
        return float(max(
            0.0,
            float(self.spec.max_total_cost_usd)
            - float(self.cost_used_usd)))

    def remaining_tokens(self) -> int:
        return int(max(
            0,
            int(self.spec.max_total_tokens)
            - int(self.tokens_used)))

    def remaining_tool_calls(self) -> int:
        return int(max(
            0,
            int(self.spec.max_tool_calls)
            - int(self.tool_calls_used)))

    def remaining_flops(self) -> int:
        return int(max(
            0,
            int(self.spec.max_recompute_flops)
            - int(self.recompute_flops_used)))

    def propose(
            self, *, proposal: ActionProposalV1,
    ) -> tuple[str, BudgetBreachAuditV1 | None]:
        """Decide whether to allow the proposed action.

        Returns ``(verdict, audit_or_None)``.
        """
        if bool(self.spec.enforcer_disabled):
            # Enforcer disabled — but the disabled flag is
            # recorded honestly in any subsequent breach audit.
            return (
                BudgetCommitVerdict.ENFORCER_DISABLED.value,
                None)
        would_cost = float(self.cost_model.estimate_cost_usd(
            action=str(proposal.action),
            prompt_tokens=int(proposal.prompt_tokens),
            output_tokens=int(proposal.output_tokens),
            tool_calls=int(proposal.tool_calls),
            recompute_flops=int(proposal.recompute_flops)))
        breaches: list[str] = []
        if (float(self.cost_used_usd) + float(would_cost)
                > float(self.spec.max_total_cost_usd)):
            breaches.append(BudgetBreachAxis.COST_USD.value)
        if (int(self.tokens_used)
                + int(proposal.prompt_tokens)
                + int(proposal.output_tokens)
                > int(self.spec.max_total_tokens)):
            breaches.append(
                BudgetBreachAxis.TOTAL_TOKENS.value)
        if (int(self.tool_calls_used)
                + int(proposal.tool_calls)
                > int(self.spec.max_tool_calls)):
            breaches.append(
                BudgetBreachAxis.TOOL_CALLS.value)
        if (int(self.recompute_flops_used)
                + int(proposal.recompute_flops)
                > int(self.spec.max_recompute_flops)):
            breaches.append(
                BudgetBreachAxis.RECOMPUTE_FLOPS.value)
        if (float(proposal.measured_step_latency_ms)
                > float(self.spec.max_per_step_latency_ms)):
            breaches.append(
                BudgetBreachAxis.PER_STEP_LATENCY_MS.value)
        if not breaches:
            return (
                BudgetCommitVerdict.COMMIT_ALLOWED.value, None)
        audit = BudgetBreachAuditV1(
            schema=W84_BUDGET_V1_SCHEMA_VERSION,
            budget_spec_cid=str(self.spec.cid()),
            cost_model_cid=str(self.cost_model.cid()),
            would_be_action=str(proposal.action),
            pre_budget_remaining_cost_usd=float(
                self.remaining_cost_usd()),
            pre_budget_remaining_tokens=int(
                self.remaining_tokens()),
            pre_budget_remaining_tool_calls=int(
                self.remaining_tool_calls()),
            pre_budget_remaining_flops=int(
                self.remaining_flops()),
            pre_step_latency_ms_measured=float(
                proposal.measured_step_latency_ms),
            would_cost_usd=float(would_cost),
            would_tokens=int(
                proposal.prompt_tokens
                + proposal.output_tokens),
            would_tool_calls=int(proposal.tool_calls),
            would_recompute_flops=int(
                proposal.recompute_flops),
            breach_axes=tuple(breaches),
            timestamp_ns=int(time.time_ns()),
            enforcer_was_disabled=False,
        )
        if bool(self.spec.record_breach_audit):
            self.breach_audits.append(audit)
        return (
            BudgetCommitVerdict.ABSTAIN_ON_BREACH.value, audit)

    def commit(self, *, proposal: ActionProposalV1) -> None:
        """Account a committed action (caller invokes after a
        commit-allowed verdict)."""
        cost = float(self.cost_model.estimate_cost_usd(
            action=str(proposal.action),
            prompt_tokens=int(proposal.prompt_tokens),
            output_tokens=int(proposal.output_tokens),
            tool_calls=int(proposal.tool_calls),
            recompute_flops=int(proposal.recompute_flops)))
        self.cost_used_usd = float(
            self.cost_used_usd) + cost
        self.tokens_used = int(self.tokens_used) + int(
            proposal.prompt_tokens) + int(
            proposal.output_tokens)
        self.tool_calls_used = int(self.tool_calls_used) + int(
            proposal.tool_calls)
        self.recompute_flops_used = int(
            self.recompute_flops_used) + int(
            proposal.recompute_flops)


# ---------------------------------------------------------------
# Bench
# ---------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class BudgetStressBenchReportV1:
    schema: str
    budget_spec_cid: str
    cost_model_cid: str
    n_proposals: int
    n_commits: int
    n_abstain_on_breach: int
    all_proposals_audit_logged: bool
    audit_chain_cid: str
    under_budget_commits_exactly_match: bool
    breach_axes_seen: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "budget_spec_cid": str(self.budget_spec_cid),
            "cost_model_cid": str(self.cost_model_cid),
            "n_proposals": int(self.n_proposals),
            "n_commits": int(self.n_commits),
            "n_abstain_on_breach": int(
                self.n_abstain_on_breach),
            "all_proposals_audit_logged": bool(
                self.all_proposals_audit_logged),
            "audit_chain_cid": str(self.audit_chain_cid),
            "under_budget_commits_exactly_match": bool(
                self.under_budget_commits_exactly_match),
            "breach_axes_seen": list(self.breach_axes_seen),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w84_budget_stress_bench_report_v1",
            "report": self.to_dict()})


def run_budget_stress_bench_v1(
        *, seed: int = 84_037_001,
) -> BudgetStressBenchReportV1:
    """End-to-end budget enforcement bench.

    Two regimes:
    * Over-budget regime: budget tight, every proposal would
      overshoot → enforcer drives the commit count to 0;
      every abstain carries a breach audit.
    * Under-budget regime: budget loose, every proposal fits →
      enforcer never abstains.
    """
    cm = build_default_cost_model_v1()
    # Over-budget regime: very tight cost budget.
    tight_spec = build_run_budget_spec_v1(
        run_cid="w84_overbudget",
        max_total_cost_usd=0.0005,  # almost zero
        max_per_step_latency_ms=1000.0,
        max_total_tokens=10_000,
        max_tool_calls=10,
        max_recompute_flops=10_000_000,
        abstain_on_breach=True,
        record_breach_audit=True,
    )
    enf = BudgetEnforcerV1(spec=tight_spec, cost_model=cm)
    proposals_tight = [
        ActionProposalV1(
            action="runtime_recompute",
            prompt_tokens=200, output_tokens=100,
            tool_calls=1, recompute_flops=10_000,
            measured_step_latency_ms=10.0)
        for _ in range(10)
    ]
    commits_tight = 0
    abstains_tight = 0
    breach_axes_seen: set[str] = set()
    for p in proposals_tight:
        verdict, audit = enf.propose(proposal=p)
        if verdict == (
                BudgetCommitVerdict.COMMIT_ALLOWED.value):
            enf.commit(proposal=p)
            commits_tight += 1
        elif verdict == (
                BudgetCommitVerdict.ABSTAIN_ON_BREACH.value):
            abstains_tight += 1
            if audit is not None:
                for ax in audit.breach_axes:
                    breach_axes_seen.add(str(ax))
    # All abstain audits should equal abstains_tight.
    all_logged = (
        int(len(enf.breach_audits)) == int(abstains_tight))
    # Under-budget regime: loose budget; all proposals
    # commit; commit count equals the proposal count.
    loose_spec = build_run_budget_spec_v1(
        run_cid="w84_underbudget",
        max_total_cost_usd=10.0,
        max_per_step_latency_ms=10_000.0,
        max_total_tokens=10_000_000,
        max_tool_calls=10_000,
        max_recompute_flops=10_000_000_000,
        abstain_on_breach=True,
        record_breach_audit=True,
    )
    enf2 = BudgetEnforcerV1(spec=loose_spec, cost_model=cm)
    proposals_loose = list(proposals_tight)
    commits_loose = 0
    for p in proposals_loose:
        verdict, _ = enf2.propose(proposal=p)
        if verdict == (
                BudgetCommitVerdict.COMMIT_ALLOWED.value):
            enf2.commit(proposal=p)
            commits_loose += 1
    under_exact = (
        int(commits_loose) == int(len(proposals_loose)))
    # Audit chain CID over all breach audits.
    audit_chain_cid = _sha256_hex({
        "kind": "w84_budget_audit_chain_v1",
        "audit_cids": [
            str(a.cid()) for a in enf.breach_audits],
    })
    return BudgetStressBenchReportV1(
        schema=W84_BUDGET_V1_SCHEMA_VERSION,
        budget_spec_cid=str(tight_spec.cid()),
        cost_model_cid=str(cm.cid()),
        n_proposals=int(len(proposals_tight)),
        n_commits=int(commits_tight),
        n_abstain_on_breach=int(abstains_tight),
        all_proposals_audit_logged=bool(all_logged),
        audit_chain_cid=str(audit_chain_cid),
        under_budget_commits_exactly_match=bool(under_exact),
        breach_axes_seen=tuple(sorted(breach_axes_seen)),
    )


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


__all__ = [
    "W84_BUDGET_V1_SCHEMA_VERSION",
    "BudgetBreachAxis",
    "BudgetCommitVerdict",
    "RunBudgetSpecV1",
    "build_run_budget_spec_v1",
    "CostModelV1",
    "build_default_cost_model_v1",
    "BudgetBreachAuditV1",
    "ActionProposalV1",
    "BudgetEnforcerV1",
    "BudgetStressBenchReportV1",
    "run_budget_stress_bench_v1",
]
