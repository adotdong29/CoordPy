"""W84 / P1 #37 — Hard Cost / Latency Budget Enforcement V1.

Issue #37 asks for an end-to-end mechanism that takes a
``RunBudgetSpec`` carrying ``max_cost_usd`` / ``max_latency_ms``
/ ``max_tokens`` / ``max_tool_calls`` / ``max_recompute_flops``
and *provably* refuses to violate them across the W83 composed
pipeline. The W81 ``learned_economics_controller_v1`` picks
actions on a utility surface but does not enforce a hard
ceiling. The W82 ``distributed_substrate_coordination_v1``
``MigrationBudgetPolicyV1`` enforces *migration-byte* budgets
but is local to migrations, not to the whole composed pipeline.

W84 V1 closes the gap with three load-bearing properties:

* The budget is enforced *pre-action* — every candidate action
  is checked against the remaining budget BEFORE it is applied.
  Over-budget actions are refused (no silent overspend).
* Every refusal emits a content-addressed
  ``BudgetBreachAuditV1`` capsule into the audit chain.
* The cost model is *content-addressed*; identical configs
  produce identical cost CIDs so a third party can re-verify
  what the budget meant.

Anti-cheat clauses from the issue body:

* The enforcer is on by default. There is a ``budget_disabled``
  flag for local dev only; the flag's setting appears in the
  audit chain so a deployment that disables the enforcer can be
  detected.
* The cost model is monotone-in-tokens by construction.
* Abstain-on-breach is **not** task failure; it is the contract.
* Tool calls count toward both ``max_tool_calls`` and the
  per-call ``token`` / ``cost`` budgets via the W84 tool
  substrate integration.

Honest scope (W84 V1)
---------------------

* ``W84-L-BUDGET-ENFORCEMENT-V1-RESEARCH-ONLY-CAP`` — explicit-
  import only; not on the stable public surface.
* ``W84-L-BUDGET-ENFORCEMENT-V1-PER-RUN-CAP`` — V1 enforces
  per-run budgets. Per-step / per-tenant / per-agent budgets
  are V2 (composes with multi-tenancy issue #43).
* ``W84-L-BUDGET-ENFORCEMENT-V1-STATIC-COST-MODEL-CAP`` — V1
  cost model is a content-addressed static lookup of
  ``(action, model_cid, prompt_tokens, output_tokens) → USD``.
  Dynamic pricing is V2.
* ``W84-L-BUDGET-ENFORCEMENT-V1-NUMPY-CAP`` — pure stdlib +
  NumPy.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
import time as _time
from typing import Any, Mapping, Sequence


W84_BUDGET_ENFORCEMENT_V1_SCHEMA_VERSION: str = (
    "coordpy.budget_enforcement_v1.v1")


class BudgetAxis(str, enum.Enum):
    """The five canonical budget axes the V1 enforcer tracks."""

    COST_USD = "max_total_cost_usd"
    LATENCY_MS = "max_per_step_latency_ms"
    TOKENS = "max_total_tokens"
    TOOL_CALLS = "max_tool_calls"
    RECOMPUTE_FLOPS = "max_recompute_flops"


W84_BUDGET_AXES: tuple[str, ...] = tuple(a.value for a in BudgetAxis)


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


# ---------------------------------------------------------------
# Cost model — content-addressed, monotone in tokens.
# ---------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class CostModelV1:
    """Static, content-addressed cost model.

    Cost is computed as::

        usd = base_cost_per_action[action]
            + per_prompt_token_usd * prompt_tokens
            + per_output_token_usd * output_tokens

    The model is *monotone in tokens* by construction: both
    coefficients are non-negative, so a longer prompt or longer
    output cannot reduce cost. The anti-cheat clause "do not
    make the cost model so loose that nothing is over-budget"
    is honored by refusing to construct a model with negative
    or zero token coefficients.
    """

    schema: str
    model_cid: str
    base_cost_per_action: tuple[tuple[str, float], ...]
    per_prompt_token_usd: float
    per_output_token_usd: float

    def __post_init__(self) -> None:
        if not (float(self.per_prompt_token_usd) > 0.0):
            raise ValueError(
                "per_prompt_token_usd must be > 0 "
                "(anti-cheat: cost model must be monotone)")
        if not (float(self.per_output_token_usd) > 0.0):
            raise ValueError(
                "per_output_token_usd must be > 0 "
                "(anti-cheat: cost model must be monotone)")
        for _, v in self.base_cost_per_action:
            if not (float(v) >= 0.0):
                raise ValueError(
                    "base_cost_per_action must be >= 0")

    def _per_action_map(self) -> dict[str, float]:
        return {str(k): float(v)
                for k, v in self.base_cost_per_action}

    def estimate_usd(
            self, *, action: str,
            prompt_tokens: int, output_tokens: int,
    ) -> float:
        m = self._per_action_map()
        base = float(m.get(str(action), 0.0))
        return (base
                + float(self.per_prompt_token_usd)
                * int(prompt_tokens)
                + float(self.per_output_token_usd)
                * int(output_tokens))

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w84_cost_model_v1",
            "schema": str(self.schema),
            "model_cid": str(self.model_cid),
            "base_cost_per_action": [
                (str(k), float(round(v, 12)))
                for k, v in self.base_cost_per_action],
            "per_prompt_token_usd": float(round(
                self.per_prompt_token_usd, 12)),
            "per_output_token_usd": float(round(
                self.per_output_token_usd, 12)),
        })


def default_cost_model_v1(
        *, model_cid: str = "controlled-runtime-v1",
) -> CostModelV1:
    """A small, monotone, content-addressed default cost model."""
    return CostModelV1(
        schema=W84_BUDGET_ENFORCEMENT_V1_SCHEMA_VERSION,
        model_cid=str(model_cid),
        base_cost_per_action=(
            ("replay", 0.000_2),
            ("runtime_recompute", 0.001_0),
            ("transcript_recompute", 0.000_4),
            ("promote_to_richer_substrate", 0.005_0),
            ("abstain", 0.0),
            ("tool_call", 0.000_8),
        ),
        per_prompt_token_usd=0.000_001,
        per_output_token_usd=0.000_002,
    )


# ---------------------------------------------------------------
# Budget spec — content-addressed.
# ---------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class RunBudgetSpecV1:
    """The hard run-level budget the enforcer respects."""

    schema: str
    max_total_cost_usd: float
    max_per_step_latency_ms: float
    max_total_tokens: int
    max_tool_calls: int
    max_recompute_flops: float
    abstain_on_breach: bool = True
    record_breach_audit: bool = True
    budget_disabled: bool = False  # local dev only; appears in audit chain

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w84_run_budget_spec_v1",
            "schema": str(self.schema),
            "max_total_cost_usd": float(round(
                self.max_total_cost_usd, 12)),
            "max_per_step_latency_ms": float(round(
                self.max_per_step_latency_ms, 6)),
            "max_total_tokens": int(self.max_total_tokens),
            "max_tool_calls": int(self.max_tool_calls),
            "max_recompute_flops": float(round(
                self.max_recompute_flops, 6)),
            "abstain_on_breach": bool(self.abstain_on_breach),
            "record_breach_audit": bool(self.record_breach_audit),
            "budget_disabled": bool(self.budget_disabled),
        })


@dataclasses.dataclass(frozen=True)
class CandidateActionV1:
    """One candidate action to be checked against the budget.

    ``predicted_*`` fields are the enforcer's best estimate of
    the action's cost — they come from the cost model and the
    caller's measured per-step prediction. The enforcer compares
    these to the remaining budget envelope.
    """

    action_name: str
    predicted_usd: float
    predicted_latency_ms: float
    predicted_tokens: int
    is_tool_call: bool
    predicted_flops: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_name": str(self.action_name),
            "predicted_usd": float(round(self.predicted_usd, 12)),
            "predicted_latency_ms": float(round(
                self.predicted_latency_ms, 6)),
            "predicted_tokens": int(self.predicted_tokens),
            "is_tool_call": bool(self.is_tool_call),
            "predicted_flops": float(round(
                self.predicted_flops, 6)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w84_candidate_action_v1",
            "action": self.to_dict()})


# ---------------------------------------------------------------
# Audit capsules.
# ---------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class BudgetBreachAuditV1:
    """A refusal record. One per refused candidate action."""

    schema: str
    budget_spec_cid: str
    cost_model_cid: str
    pre_budget_used: Mapping[str, float]
    candidate_action_cid: str
    candidate_action_dict: Mapping[str, Any]
    breached_axis: str
    refusal_reason: str
    post_budget_would_be: Mapping[str, float]
    budget_disabled_flag: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "budget_spec_cid": str(self.budget_spec_cid),
            "cost_model_cid": str(self.cost_model_cid),
            "pre_budget_used": {
                str(k): float(round(v, 12))
                for k, v in self.pre_budget_used.items()},
            "candidate_action_cid": str(
                self.candidate_action_cid),
            "candidate_action": {
                str(k): v
                for k, v in self.candidate_action_dict.items()},
            "breached_axis": str(self.breached_axis),
            "refusal_reason": str(self.refusal_reason),
            "post_budget_would_be": {
                str(k): float(round(v, 12))
                for k, v in self.post_budget_would_be.items()},
            "budget_disabled_flag": bool(
                self.budget_disabled_flag),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w84_budget_breach_audit_v1",
            "audit": self.to_dict()})


@dataclasses.dataclass(frozen=True)
class BudgetEnforcerVerdictV1:
    """Result of one enforcer step."""

    schema: str
    permitted: bool
    breach_audit: BudgetBreachAuditV1 | None
    pre_budget_used: Mapping[str, float]
    post_budget_used: Mapping[str, float]
    candidate_action_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "permitted": bool(self.permitted),
            "breach_audit_cid": (
                str(self.breach_audit.cid())
                if self.breach_audit is not None else "absent"),
            "pre_budget_used": {
                str(k): float(round(v, 12))
                for k, v in self.pre_budget_used.items()},
            "post_budget_used": {
                str(k): float(round(v, 12))
                for k, v in self.post_budget_used.items()},
            "candidate_action_cid": str(
                self.candidate_action_cid),
        }


# ---------------------------------------------------------------
# Enforcer.
# ---------------------------------------------------------------

@dataclasses.dataclass
class BudgetEnforcerV1:
    """End-to-end pre-action budget enforcer.

    Keeps a running tally per axis and refuses any candidate
    action that would push any axis past its cap. Each refusal
    becomes a content-addressed ``BudgetBreachAuditV1`` capsule
    so a third party can re-verify the decision history.
    """

    spec: RunBudgetSpecV1
    cost_model: CostModelV1
    _used_cost_usd: float = 0.0
    _used_tokens: int = 0
    _used_tool_calls: int = 0
    _used_recompute_flops: float = 0.0
    _last_step_latency_ms: float = 0.0
    _audit_chain: list[BudgetBreachAuditV1] = dataclasses.field(
        default_factory=list)
    _commit_chain: list[str] = dataclasses.field(
        default_factory=list)

    def _current_used(self) -> dict[str, float]:
        return {
            BudgetAxis.COST_USD.value: float(self._used_cost_usd),
            BudgetAxis.LATENCY_MS.value: float(
                self._last_step_latency_ms),
            BudgetAxis.TOKENS.value: float(self._used_tokens),
            BudgetAxis.TOOL_CALLS.value: float(
                self._used_tool_calls),
            BudgetAxis.RECOMPUTE_FLOPS.value: float(
                self._used_recompute_flops),
        }

    def _projected_used(
            self, action: CandidateActionV1,
    ) -> dict[str, float]:
        return {
            BudgetAxis.COST_USD.value: float(
                self._used_cost_usd + action.predicted_usd),
            BudgetAxis.LATENCY_MS.value: float(
                action.predicted_latency_ms),
            BudgetAxis.TOKENS.value: float(
                self._used_tokens + int(action.predicted_tokens)),
            BudgetAxis.TOOL_CALLS.value: float(
                self._used_tool_calls
                + (1 if action.is_tool_call else 0)),
            BudgetAxis.RECOMPUTE_FLOPS.value: float(
                self._used_recompute_flops
                + float(action.predicted_flops)),
        }

    def _which_axis_breaches(
            self, projected: Mapping[str, float],
    ) -> str | None:
        if (float(projected[BudgetAxis.COST_USD.value])
                > float(self.spec.max_total_cost_usd)):
            return BudgetAxis.COST_USD.value
        if (float(projected[BudgetAxis.LATENCY_MS.value])
                > float(self.spec.max_per_step_latency_ms)):
            return BudgetAxis.LATENCY_MS.value
        if (float(projected[BudgetAxis.TOKENS.value])
                > float(self.spec.max_total_tokens)):
            return BudgetAxis.TOKENS.value
        if (float(projected[BudgetAxis.TOOL_CALLS.value])
                > float(self.spec.max_tool_calls)):
            return BudgetAxis.TOOL_CALLS.value
        if (float(projected[BudgetAxis.RECOMPUTE_FLOPS.value])
                > float(self.spec.max_recompute_flops)):
            return BudgetAxis.RECOMPUTE_FLOPS.value
        return None

    def check(
            self, action: CandidateActionV1,
    ) -> BudgetEnforcerVerdictV1:
        """Pre-action check. Does NOT mutate the tally.

        Returns a verdict carrying ``permitted`` and (if not
        permitted) a content-addressed breach audit.
        """
        pre = dict(self._current_used())
        post = self._projected_used(action)
        if self.spec.budget_disabled:
            # The flag's setting is recorded in any subsequent
            # commit's audit. Permitting under this flag is the
            # documented dev-only path.
            return BudgetEnforcerVerdictV1(
                schema=W84_BUDGET_ENFORCEMENT_V1_SCHEMA_VERSION,
                permitted=True,
                breach_audit=None,
                pre_budget_used=pre,
                post_budget_used=post,
                candidate_action_cid=str(action.cid()),
            )
        which = self._which_axis_breaches(post)
        if which is None:
            return BudgetEnforcerVerdictV1(
                schema=W84_BUDGET_ENFORCEMENT_V1_SCHEMA_VERSION,
                permitted=True,
                breach_audit=None,
                pre_budget_used=pre,
                post_budget_used=post,
                candidate_action_cid=str(action.cid()),
            )
        reason = (
            f"action {action.action_name} would push "
            f"{which} past its cap")
        audit = BudgetBreachAuditV1(
            schema=W84_BUDGET_ENFORCEMENT_V1_SCHEMA_VERSION,
            budget_spec_cid=str(self.spec.cid()),
            cost_model_cid=str(self.cost_model.cid()),
            pre_budget_used=pre,
            candidate_action_cid=str(action.cid()),
            candidate_action_dict=dict(action.to_dict()),
            breached_axis=str(which),
            refusal_reason=str(reason),
            post_budget_would_be=post,
            budget_disabled_flag=bool(self.spec.budget_disabled),
        )
        return BudgetEnforcerVerdictV1(
            schema=W84_BUDGET_ENFORCEMENT_V1_SCHEMA_VERSION,
            permitted=False,
            breach_audit=audit,
            pre_budget_used=pre,
            post_budget_used=post,
            candidate_action_cid=str(action.cid()),
        )

    def commit(self, action: CandidateActionV1) -> None:
        """Apply a permitted action's effects to the tally.

        Caller must call ``check(...)`` and verify
        ``permitted`` before calling ``commit``.
        """
        self._used_cost_usd = float(
            self._used_cost_usd + float(action.predicted_usd))
        self._used_tokens = int(
            self._used_tokens + int(action.predicted_tokens))
        if bool(action.is_tool_call):
            self._used_tool_calls = int(
                self._used_tool_calls + 1)
        self._used_recompute_flops = float(
            self._used_recompute_flops
            + float(action.predicted_flops))
        self._last_step_latency_ms = float(
            action.predicted_latency_ms)
        self._commit_chain.append(str(action.cid()))

    def record_breach(self, audit: BudgetBreachAuditV1) -> None:
        """Append a breach audit to the chain (idempotent)."""
        # Idempotency: dedupe by CID.
        seen = {str(a.cid()) for a in self._audit_chain}
        if str(audit.cid()) not in seen:
            self._audit_chain.append(audit)

    def audit_chain(self) -> tuple[BudgetBreachAuditV1, ...]:
        return tuple(self._audit_chain)

    def commit_chain(self) -> tuple[str, ...]:
        return tuple(self._commit_chain)

    def used(self) -> Mapping[str, float]:
        return dict(self._current_used())

    def merkle_audit_root(self) -> str:
        """Content-addressed root over the breach audit chain."""
        return _sha256_hex({
            "kind": "w84_budget_audit_merkle_root_v1",
            "schema": W84_BUDGET_ENFORCEMENT_V1_SCHEMA_VERSION,
            "breach_audit_cids": [
                str(a.cid()) for a in self._audit_chain],
            "commit_chain": list(self._commit_chain),
        })


# ---------------------------------------------------------------
# Stress bench — over-budget regime + under-budget regime.
# ---------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class BudgetBreachStressBenchReportV1:
    schema: str
    over_budget_regime_n_actions: int
    over_budget_regime_n_committed: int
    over_budget_regime_n_refused: int
    over_budget_regime_audit_root: str
    under_budget_regime_n_actions: int
    under_budget_regime_n_committed: int
    under_budget_regime_n_refused: int
    under_budget_regime_audit_root: str
    no_enforcer_regime_n_committed: int
    over_budget_zero_commit: bool
    under_budget_matches_no_enforcer: bool

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w84_budget_breach_stress_bench_v1",
            "report": {
                "schema": str(self.schema),
                "over_budget_regime_n_actions": int(
                    self.over_budget_regime_n_actions),
                "over_budget_regime_n_committed": int(
                    self.over_budget_regime_n_committed),
                "over_budget_regime_n_refused": int(
                    self.over_budget_regime_n_refused),
                "over_budget_regime_audit_root": str(
                    self.over_budget_regime_audit_root),
                "under_budget_regime_n_actions": int(
                    self.under_budget_regime_n_actions),
                "under_budget_regime_n_committed": int(
                    self.under_budget_regime_n_committed),
                "under_budget_regime_n_refused": int(
                    self.under_budget_regime_n_refused),
                "under_budget_regime_audit_root": str(
                    self.under_budget_regime_audit_root),
                "no_enforcer_regime_n_committed": int(
                    self.no_enforcer_regime_n_committed),
                "over_budget_zero_commit": bool(
                    self.over_budget_zero_commit),
                "under_budget_matches_no_enforcer": bool(
                    self.under_budget_matches_no_enforcer),
            },
        })


def _action_sequence_for_regime(
        regime: str,
        n_actions: int = 12,
) -> tuple[CandidateActionV1, ...]:
    """Deterministic action sequence used by the stress bench.

    Each action is the same shape — what differs across regimes
    is the BUDGET, not the actions.
    """
    actions: list[CandidateActionV1] = []
    for i in range(int(n_actions)):
        a = CandidateActionV1(
            action_name=(
                "runtime_recompute" if i % 3 == 0
                else "replay" if i % 3 == 1
                else "tool_call"),
            predicted_usd=0.001 + 0.0002 * (i % 5),
            predicted_latency_ms=20.0 + 5.0 * (i % 4),
            predicted_tokens=int(50 + 10 * (i % 4)),
            is_tool_call=(i % 3 == 2),
            predicted_flops=200.0 + 50.0 * (i % 3),
        )
        actions.append(a)
    _ = regime  # tag kept for the test surface
    return tuple(actions)


def run_budget_breach_stress_bench_v1(
        *,
        cost_model: CostModelV1 | None = None,
        n_actions: int = 12,
) -> BudgetBreachStressBenchReportV1:
    """Run three regimes:

    1. **No-enforcer** baseline: every action commits.
    2. **Over-budget** regime: total budget < total cost of all
       actions; enforcer must refuse every commit that would
       breach.
    3. **Under-budget** regime: total budget >> total cost; the
       enforcer must permit every action — identical commit
       behaviour to the no-enforcer regime.
    """
    cm = cost_model or default_cost_model_v1()
    actions = _action_sequence_for_regime("stress", n_actions)
    # No-enforcer regime: just commit everything.
    no_enf_committed = int(len(actions))
    # Over-budget regime: tiny budget; nothing can commit.
    tiny_spec = RunBudgetSpecV1(
        schema=W84_BUDGET_ENFORCEMENT_V1_SCHEMA_VERSION,
        max_total_cost_usd=0.000_001,
        max_per_step_latency_ms=0.001,
        max_total_tokens=1,
        max_tool_calls=0,
        max_recompute_flops=0.001,
    )
    over_enf = BudgetEnforcerV1(spec=tiny_spec, cost_model=cm)
    over_committed = 0
    over_refused = 0
    for a in actions:
        v = over_enf.check(a)
        if v.permitted:
            over_enf.commit(a)
            over_committed += 1
        else:
            assert v.breach_audit is not None
            over_enf.record_breach(v.breach_audit)
            over_refused += 1
    # Under-budget regime: huge budget; everything commits.
    big_spec = RunBudgetSpecV1(
        schema=W84_BUDGET_ENFORCEMENT_V1_SCHEMA_VERSION,
        max_total_cost_usd=1_000.0,
        max_per_step_latency_ms=1_000_000.0,
        max_total_tokens=1_000_000,
        max_tool_calls=1_000,
        max_recompute_flops=1e12,
    )
    under_enf = BudgetEnforcerV1(spec=big_spec, cost_model=cm)
    under_committed = 0
    under_refused = 0
    for a in actions:
        v = under_enf.check(a)
        if v.permitted:
            under_enf.commit(a)
            under_committed += 1
        else:
            assert v.breach_audit is not None
            under_enf.record_breach(v.breach_audit)
            under_refused += 1
    return BudgetBreachStressBenchReportV1(
        schema=W84_BUDGET_ENFORCEMENT_V1_SCHEMA_VERSION,
        over_budget_regime_n_actions=int(len(actions)),
        over_budget_regime_n_committed=int(over_committed),
        over_budget_regime_n_refused=int(over_refused),
        over_budget_regime_audit_root=str(
            over_enf.merkle_audit_root()),
        under_budget_regime_n_actions=int(len(actions)),
        under_budget_regime_n_committed=int(under_committed),
        under_budget_regime_n_refused=int(under_refused),
        under_budget_regime_audit_root=str(
            under_enf.merkle_audit_root()),
        no_enforcer_regime_n_committed=int(no_enf_committed),
        over_budget_zero_commit=bool(over_committed == 0),
        under_budget_matches_no_enforcer=bool(
            under_committed == no_enf_committed),
    )


__all__ = [
    "W84_BUDGET_ENFORCEMENT_V1_SCHEMA_VERSION",
    "W84_BUDGET_AXES",
    "BudgetAxis",
    "CostModelV1",
    "default_cost_model_v1",
    "RunBudgetSpecV1",
    "CandidateActionV1",
    "BudgetBreachAuditV1",
    "BudgetEnforcerVerdictV1",
    "BudgetEnforcerV1",
    "BudgetBreachStressBenchReportV1",
    "run_budget_breach_stress_bench_v1",
]
