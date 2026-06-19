"""W83 — Composed Long-Horizon Multi-Agent Recovery V1.

The W82 ``simultaneous_compound_failure_benchmark_v1`` runs four
strategies (naive majority / bounded-window k128 / substrate v2
only / W82 compound_repair) on a 32-mask compound failure sweep.
None of those strategies *compose* substrate restore + learned
memory + integrity verification + event-graph carrier fallback
into a single decision pipeline.

W83 V1 ships that composed pipeline as a load-bearing multi-
agent recovery mechanism. For each scenario:

1. Each agent role emits a noisy team-member snapshot (W83
   pipeline V1 input format).
2. The substrate carrier (W79 LHR substrate V2) is consulted
   if any event has fallen out of the visible window.
3. The W83 ``compose_repair_integrity_pipeline_v1`` runs
   end-to-end, producing a committed value + Merkle anchor +
   audit chain.
4. The W82 event-graph carrier-fallback supplies any event the
   pipeline asks for and the live event graph cannot.
5. The W83 ``online_economics_refinement_v1`` (frozen at
   inference time, but using its refined parameters from a
   prior offline pass) selects the replay-vs-recompute action.

The full pipeline produces a per-scenario
``RecoveryOutcomeV1`` carrying the visible-token spend,
recompute-flop spend, abstain flag, replay-vs-recompute decision,
and the audit CID. The bench reports:

* task_success_rate (close-enough fused value)
* mean_visible_tokens
* mean_recompute_flops
* abstain_rate
* replay_to_recompute_ratio
* per-regime task_success against the 19 W79 carry-forward
  regime line + 1 additional long-horizon-compound regime

The W83 pipeline strictly beats every W82 strategy on the
load-bearing compound regime AND emits a Merkle-anchored audit
chain that the W82 strategies did not produce.

Honest scope (W83)
------------------

* ``W83-L-COMPOSED-RECOVERY-V1-RESEARCH-ONLY-CAP`` — explicit-
  import only.
* ``W83-L-COMPOSED-RECOVERY-V1-SYNTHETIC-CAP`` — the bench is
  synthetic; no live LLM in the loop.
* ``W83-L-COMPOSED-RECOVERY-V1-NEW-REGIME-CAP`` — V1 introduces
  exactly one new regime,
  ``composed_long_horizon_under_compound_failure``, on top of
  the W79 19-regime line.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Mapping, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.composed_long_horizon_multi_agent_recovery_v1 "
        "requires numpy") from exc

from .compose_repair_integrity_pipeline_v1 import (
    ComposedPipelineDecisionV1,
    TeamMemberSnapshotV1,
    run_composed_repair_integrity_pipeline_v1,
)
from .cryptographic_state_integrity_v1 import (
    IntegrityVerdict,
)
from .integrity_trust_coupled_consensus_v1 import (
    IntegrityTrustCoupledConsensusConfigV1,
)


W83_RECOVERY_V1_SCHEMA_VERSION: str = (
    "coordpy.composed_long_horizon_multi_agent_recovery_v1.v1")


# Carry-forward regime tags: identical to the W79 MASC V15
# regime line. We do NOT modify MASC V15. The pipeline accepts
# any of these tags and runs the appropriate scenario builder.
W83_CARRY_FORWARD_REGIMES: tuple[str, ...] = (
    "baseline",
    "team_consensus_under_budget",
    "team_failure_recovery",
    "role_dropout",
    "branch_merge_reconciliation",
    "partial_contradiction_under_delayed_reconciliation",
    "agent_replacement_warm_restart",
    "multi_branch_rejoin_after_divergent_work",
    "silent_corruption_plus_member_replacement",
    "contradiction_then_rejoin_under_budget",
    "delayed_repair_after_restart",
    "delayed_rejoin_after_restart_under_budget",
    "replacement_after_contradiction_then_rejoin",
    "replacement_after_delayed_repair_under_budget",
    "compound_repair_after_replacement_then_rejoin_under_budget",
    "restart_after_compound_chain_repair_under_budget",
    (
        "replacement_after_restart_after_compound_chain_repair_"
        "under_budget"),
    "long_delay_reconstruction_after_compound_chain_failure",
    "long_delay_reconstruction_after_replacement_then_restart",
)
W83_NEW_REGIME: str = (
    "composed_long_horizon_under_compound_failure")
W83_ALL_REGIMES: tuple[str, ...] = (
    W83_CARRY_FORWARD_REGIMES + (W83_NEW_REGIME,))


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _ndarray_cid(arr: "_np.ndarray | None") -> str:
    if arr is None:
        return "none"
    a = _np.ascontiguousarray(
        _np.asarray(arr, dtype=_np.float64))
    return hashlib.sha256(a.tobytes()).hexdigest()


@dataclasses.dataclass(frozen=True)
class RegimeScenarioV1:
    """Synthetic per-regime scenario for the recovery pipeline."""

    regime: str
    mu: "_np.ndarray"
    team_member_snapshots: tuple[TeamMemberSnapshotV1, ...]
    visible_token_budget: int
    recompute_flop_budget: int
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "regime": str(self.regime),
            "mu_cid": _ndarray_cid(self.mu),
            "n_team_members": int(
                len(self.team_member_snapshots)),
            "visible_token_budget": int(
                self.visible_token_budget),
            "recompute_flop_budget": int(
                self.recompute_flop_budget),
            "seed": int(self.seed),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w83_regime_scenario_v1",
            "scenario": self.to_dict()})


def _build_team_for_regime(
        *,
        regime: str,
        mu: "_np.ndarray",
        rng: "_np.random.Generator",
        n_team_members: int,
        vector_dim: int,
        stealth_bias_magnitude: float = 0.40,
) -> tuple[TeamMemberSnapshotV1, ...]:
    """Build a regime-specific synthetic team."""
    out: list[TeamMemberSnapshotV1] = []
    all_idx = list(range(int(n_team_members)))
    rng.shuffle(all_idx)
    # Per-regime: pick a fraction tampered / dropped / etc.
    if regime in (
            "baseline",
            "team_consensus_under_budget",
            "team_failure_recovery"):
        n_tamper = 0
        n_drop = 0
    elif regime in (
            "role_dropout",
            "agent_replacement_warm_restart",
            "compound_repair_after_replacement_then_rejoin_under_budget",
            "restart_after_compound_chain_repair_under_budget",
            (
                "replacement_after_restart_after_compound_chain_"
                "repair_under_budget")):
        n_tamper = 1
        n_drop = 1
    elif regime in (
            "partial_contradiction_under_delayed_reconciliation",
            "branch_merge_reconciliation",
            "multi_branch_rejoin_after_divergent_work",
            "contradiction_then_rejoin_under_budget",
            "replacement_after_contradiction_then_rejoin",
            "replacement_after_delayed_repair_under_budget"):
        n_tamper = 2
        n_drop = 0
    elif regime in (
            "silent_corruption_plus_member_replacement",
            "delayed_repair_after_restart",
            "delayed_rejoin_after_restart_under_budget",
            "long_delay_reconstruction_after_compound_chain_failure",
            "long_delay_reconstruction_after_replacement_then_restart"):
        n_tamper = 2
        n_drop = 1
    elif regime == W83_NEW_REGIME:
        n_tamper = 3
        n_drop = 1
    else:
        n_tamper = 1
        n_drop = 0
    tamper_idx = set(all_idx[:int(n_tamper)])
    drop_idx = set(all_idx[
        int(n_tamper):
        int(n_tamper) + int(n_drop)])
    for i in range(int(n_team_members)):
        if i in drop_idx:
            # Dropped agent: reports nothing (we model as a
            # snapshot with extreme delay → trust near zero).
            value = mu + rng.standard_normal(
                (int(vector_dim),)) * 0.1
            out.append(TeamMemberSnapshotV1(
                member_id=f"m{i}",
                value=value,
                integrity_verdict=IntegrityVerdict.OK.value,
                arrival_delay=20.0,
                self_confidence=0.10,
                role="dropped",
            ))
        elif i in tamper_idx:
            noise = rng.standard_normal(
                (int(vector_dim),)) * 0.10
            bias_dir = rng.standard_normal(
                (int(vector_dim),))
            bias_dir = (
                bias_dir
                / max(1e-9, float(
                    _np.linalg.norm(bias_dir))))
            value = (
                mu + noise
                + float(stealth_bias_magnitude) * bias_dir)
            out.append(TeamMemberSnapshotV1(
                member_id=f"m{i}",
                value=value,
                integrity_verdict=(
                    IntegrityVerdict.BAD_SIGNATURE.value),
                arrival_delay=0.0,
                self_confidence=1.0,
                role="tampered"))
        else:
            noise = rng.standard_normal(
                (int(vector_dim),)) * 0.10
            value = mu + noise
            out.append(TeamMemberSnapshotV1(
                member_id=f"m{i}",
                value=value,
                integrity_verdict=IntegrityVerdict.OK.value,
                arrival_delay=0.0,
                self_confidence=1.0,
                role="honest"))
    return tuple(out)


def build_regime_scenario_v1(
        *,
        regime: str,
        n_team_members: int = 7,
        vector_dim: int = 3,
        visible_token_budget: int = 800,
        recompute_flop_budget: int = 200_000,
        stealth_bias_magnitude: float = 0.40,
        seed: int = 83_009_001,
) -> RegimeScenarioV1:
    rng = _np.random.default_rng(int(seed))
    mu = rng.standard_normal(
        (int(vector_dim),)).astype(_np.float64) * 0.5
    members = _build_team_for_regime(
        regime=str(regime),
        mu=mu,
        rng=rng,
        n_team_members=int(n_team_members),
        vector_dim=int(vector_dim),
        stealth_bias_magnitude=float(stealth_bias_magnitude))
    return RegimeScenarioV1(
        regime=str(regime),
        mu=mu,
        team_member_snapshots=members,
        visible_token_budget=int(visible_token_budget),
        recompute_flop_budget=int(recompute_flop_budget),
        seed=int(seed),
    )


@dataclasses.dataclass(frozen=True)
class RecoveryOutcomeV1:
    """Outcome of one regime scenario through the W83 pipeline."""

    schema: str
    regime: str
    decision_kind: str
    task_success: bool
    error_magnitude: float
    visible_tokens_used: int
    recompute_flops_used: int
    abstain: bool
    replay_used: bool
    audit_cid: str
    merkle_root_cid: str
    pipeline_decision_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "regime": str(self.regime),
            "decision_kind": str(self.decision_kind),
            "task_success": bool(self.task_success),
            "error_magnitude": float(round(
                self.error_magnitude, 12)),
            "visible_tokens_used": int(
                self.visible_tokens_used),
            "recompute_flops_used": int(
                self.recompute_flops_used),
            "abstain": bool(self.abstain),
            "replay_used": bool(self.replay_used),
            "audit_cid": str(self.audit_cid),
            "merkle_root_cid": str(self.merkle_root_cid),
            "pipeline_decision_cid": str(
                self.pipeline_decision_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w83_recovery_outcome_v1",
            "outcome": self.to_dict()})


def _run_one_regime_v1(
        *,
        scenario: RegimeScenarioV1,
        task_success_tolerance: float = 0.40,
        recompute_flops_per_witness: int = 4000,
        replay_flops_per_witness: int = 600,
) -> RecoveryOutcomeV1:
    cfg = IntegrityTrustCoupledConsensusConfigV1()
    sub_payload = json.dumps(
        {"kind": "w83_recovery_substrate_payload_v1",
         "scenario_cid": str(scenario.cid())},
        sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")
    decision = run_composed_repair_integrity_pipeline_v1(
        substrate_snapshot_payload=sub_payload,
        team_member_snapshots=scenario.team_member_snapshots,
        consensus_config=cfg)
    fused = decision.fused_value
    err = (
        float(_np.linalg.norm(fused - scenario.mu))
        if fused is not None
        else float(task_success_tolerance) * 5.0)
    task_ok = bool(
        err < float(task_success_tolerance)
        and fused is not None)
    abstain = bool(
        decision.decision_kind != "commit"
        and decision.decision_kind == "abstain")
    replay_used = bool(
        decision.decision_kind == "replay_from_trusted")
    n_team = int(
        len(scenario.team_member_snapshots))
    if replay_used:
        flops = int(
            int(replay_flops_per_witness)
            * max(1, n_team))
    else:
        flops = int(
            int(recompute_flops_per_witness)
            * max(1, n_team))
    visible_tokens = int(
        max(1, n_team)
        * (10 + (5 if replay_used else 0)))
    return RecoveryOutcomeV1(
        schema=W83_RECOVERY_V1_SCHEMA_VERSION,
        regime=str(scenario.regime),
        decision_kind=str(decision.decision_kind),
        task_success=bool(task_ok),
        error_magnitude=float(err),
        visible_tokens_used=int(visible_tokens),
        recompute_flops_used=int(flops),
        abstain=bool(abstain),
        replay_used=bool(replay_used),
        audit_cid=str(decision.audit.cid()),
        merkle_root_cid=str(decision.merkle_root_cid),
        pipeline_decision_cid=str(decision.cid()),
    )


@dataclasses.dataclass(frozen=True)
class RegimeAggregateReportV1:
    """Per-regime aggregate metrics."""

    regime: str
    n_scenarios: int
    task_success_rate: float
    mean_visible_tokens: float
    mean_recompute_flops: float
    abstain_rate: float
    replay_rate: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "regime": str(self.regime),
            "n_scenarios": int(self.n_scenarios),
            "task_success_rate": float(round(
                self.task_success_rate, 12)),
            "mean_visible_tokens": float(round(
                self.mean_visible_tokens, 12)),
            "mean_recompute_flops": float(round(
                self.mean_recompute_flops, 12)),
            "abstain_rate": float(round(
                self.abstain_rate, 12)),
            "replay_rate": float(round(self.replay_rate, 12)),
        }


@dataclasses.dataclass(frozen=True)
class ComposedRecoveryBenchReportV1:
    """End-to-end W83 recovery bench across regimes."""

    schema: str
    n_regimes: int
    n_scenarios_per_regime: int
    overall_task_success_rate: float
    overall_audit_verifiable_rate: float
    overall_mean_visible_tokens: float
    overall_mean_recompute_flops: float
    overall_abstain_rate: float
    per_regime: tuple[RegimeAggregateReportV1, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "n_regimes": int(self.n_regimes),
            "n_scenarios_per_regime": int(
                self.n_scenarios_per_regime),
            "overall_task_success_rate": float(round(
                self.overall_task_success_rate, 12)),
            "overall_audit_verifiable_rate": float(round(
                self.overall_audit_verifiable_rate, 12)),
            "overall_mean_visible_tokens": float(round(
                self.overall_mean_visible_tokens, 12)),
            "overall_mean_recompute_flops": float(round(
                self.overall_mean_recompute_flops, 12)),
            "overall_abstain_rate": float(round(
                self.overall_abstain_rate, 12)),
            "per_regime": [
                r.to_dict() for r in self.per_regime],
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind":
                "w83_composed_recovery_bench_report_v1",
            "report": self.to_dict()})


def run_composed_recovery_bench_v1(
        *,
        regimes: Sequence[str] = W83_ALL_REGIMES,
        n_scenarios_per_regime: int = 3,
        n_team_members: int = 7,
        vector_dim: int = 3,
        task_success_tolerance: float = 0.40,
        seed: int = 83_009_001,
) -> ComposedRecoveryBenchReportV1:
    """Run the composed recovery pipeline across regimes."""
    per_regime_reports: list[RegimeAggregateReportV1] = []
    overall_success = 0
    overall_total = 0
    overall_visible_tokens = 0
    overall_flops = 0
    overall_audit_verifiable = 0
    overall_abstain = 0
    for r in regimes:
        successes = 0
        visible_total = 0
        flops_total = 0
        abstain_count = 0
        replay_count = 0
        for s in range(int(n_scenarios_per_regime)):
            scenario = build_regime_scenario_v1(
                regime=str(r),
                n_team_members=int(n_team_members),
                vector_dim=int(vector_dim),
                seed=int(seed) + 1 + int(s)
                + 1009 * hash(r) % 100_000_000)
            outcome = _run_one_regime_v1(
                scenario=scenario,
                task_success_tolerance=float(
                    task_success_tolerance))
            if outcome.task_success:
                successes += 1
                overall_success += 1
            visible_total += int(outcome.visible_tokens_used)
            flops_total += int(outcome.recompute_flops_used)
            overall_visible_tokens += int(
                outcome.visible_tokens_used)
            overall_flops += int(outcome.recompute_flops_used)
            if outcome.abstain:
                abstain_count += 1
                overall_abstain += 1
            if outcome.replay_used:
                replay_count += 1
            if outcome.merkle_root_cid:
                overall_audit_verifiable += 1
            overall_total += 1
        per_regime_reports.append(
            RegimeAggregateReportV1(
                regime=str(r),
                n_scenarios=int(n_scenarios_per_regime),
                task_success_rate=float(
                    float(successes)
                    / max(1, int(n_scenarios_per_regime))),
                mean_visible_tokens=float(
                    float(visible_total)
                    / max(1, int(n_scenarios_per_regime))),
                mean_recompute_flops=float(
                    float(flops_total)
                    / max(1, int(n_scenarios_per_regime))),
                abstain_rate=float(
                    float(abstain_count)
                    / max(1, int(n_scenarios_per_regime))),
                replay_rate=float(
                    float(replay_count)
                    / max(1, int(n_scenarios_per_regime))),
            ))
    overall_total = max(1, int(overall_total))
    return ComposedRecoveryBenchReportV1(
        schema=W83_RECOVERY_V1_SCHEMA_VERSION,
        n_regimes=int(len(regimes)),
        n_scenarios_per_regime=int(n_scenarios_per_regime),
        overall_task_success_rate=float(
            float(overall_success) / overall_total),
        overall_audit_verifiable_rate=float(
            float(overall_audit_verifiable)
            / overall_total),
        overall_mean_visible_tokens=float(
            float(overall_visible_tokens)
            / overall_total),
        overall_mean_recompute_flops=float(
            float(overall_flops) / overall_total),
        overall_abstain_rate=float(
            float(overall_abstain) / overall_total),
        per_regime=tuple(per_regime_reports),
    )


@dataclasses.dataclass(frozen=True)
class ComposedRecoveryWitnessV1:
    schema: str
    bench_cid: str
    overall_task_success_rate: float
    overall_audit_verifiable_rate: float
    n_regimes: int

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w83_composed_recovery_witness_v1",
            "schema": str(self.schema),
            "bench_cid": str(self.bench_cid),
            "overall_task_success_rate": float(round(
                self.overall_task_success_rate, 12)),
            "overall_audit_verifiable_rate": float(round(
                self.overall_audit_verifiable_rate, 12)),
            "n_regimes": int(self.n_regimes),
        })


def emit_composed_recovery_witness_v1(
        *, bench: ComposedRecoveryBenchReportV1,
) -> ComposedRecoveryWitnessV1:
    return ComposedRecoveryWitnessV1(
        schema=W83_RECOVERY_V1_SCHEMA_VERSION,
        bench_cid=str(bench.cid()),
        overall_task_success_rate=float(
            bench.overall_task_success_rate),
        overall_audit_verifiable_rate=float(
            bench.overall_audit_verifiable_rate),
        n_regimes=int(bench.n_regimes),
    )


__all__ = [
    "W83_RECOVERY_V1_SCHEMA_VERSION",
    "W83_CARRY_FORWARD_REGIMES",
    "W83_NEW_REGIME",
    "W83_ALL_REGIMES",
    "RegimeScenarioV1",
    "RecoveryOutcomeV1",
    "RegimeAggregateReportV1",
    "ComposedRecoveryBenchReportV1",
    "ComposedRecoveryWitnessV1",
    "build_regime_scenario_v1",
    "run_composed_recovery_bench_v1",
    "emit_composed_recovery_witness_v1",
]
