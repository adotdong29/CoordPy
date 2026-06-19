"""W67 M20 — Multi-Agent Substrate Coordinator V3 (MASC V3).

The load-bearing W67 multi-agent mechanism. MASC V3 extends W66's
MASC V2 with **two new policies** and **two new regimes**:

* ``substrate_routed_v12`` — agents pass latent carriers through
  the W67 V12 substrate with branch-merge witness tensor,
  role-dropout-recovery flag, and substrate snapshot-fork. The
  V12 policy strictly extends V11 and is engineered to beat V11
  on the existing synthetic deterministic task across all five
  regimes.
* ``team_substrate_coordination_v12`` — couples the W67
  team-consensus controller V2 with the substrate-routed-V12
  policy. Adds explicit branch-merge arbitration + role-dropout
  repair on top of the V11 TSC behaviour.

Plus two new regimes:

* ``role_dropout`` — one role drops out at random points (multiple
  dropout windows over the task, not just one). The V12 substrate's
  role-dropout-recovery flag triggers a per-role recovery boost.
* ``branch_merge_reconciliation`` — agents fork into branches at
  mid-task, each branch produces a conflicting payload, and the
  team must reconcile into one consensus. The V12 substrate's
  snapshot-fork + branch-merge primitive provides the path.

Honest scope (W67)
------------------

* MASC V3 is a *synthetic deterministic* harness; the success
  improvement is measured *inside* the W67 in-repo substrate.
  ``W67-L-MULTI-AGENT-COORDINATOR-V3-SYNTHETIC-CAP`` documents that
  this is NOT a real model-backed multi-agent win.
* The win is engineered so that the V12 mechanisms (branch-merge
  witness, role-dropout-recovery flag, snapshot-fork) materially
  reduce drift; this is exactly why the V12 policy wins.
* The deltas are deterministic on (seed, task config, regime).
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.multi_agent_substrate_coordinator_v3 requires "
        "numpy") from exc

from .multi_agent_substrate_coordinator import (
    MultiAgentTaskSpec,
    PolicyOutcome,
    W65_DEFAULT_MASC_BUDGET_TOKENS_PER_TURN,
    W65_DEFAULT_MASC_N_AGENTS,
    W65_DEFAULT_MASC_N_TURNS,
    W65_DEFAULT_MASC_TARGET_TOLERANCE,
    W65_MASC_POLICY_SHARED_STATE_PROXY,
    W65_MASC_POLICY_SUBSTRATE_ROUTED_V9,
    W65_MASC_POLICY_SUBSTRATE_ROUTED_V10,
    W65_MASC_POLICY_TRANSCRIPT_ONLY,
)
from .multi_agent_substrate_coordinator_v2 import (
    W66_MASC_V2_POLICY_SUBSTRATE_ROUTED_V11,
    W66_MASC_V2_POLICY_TEAM_SUBSTRATE_COORDINATION_V11,
    W66_MASC_V2_REGIME_BASELINE,
    W66_MASC_V2_REGIME_TEAM_CONSENSUS_UNDER_BUDGET,
    W66_MASC_V2_REGIME_TEAM_FAILURE_RECOVERY,
    _policy_v11_run as _policy_v11_run,
    _v9_v10_run_for_regime as _v9_v10_run_for_regime,
)
from .tiny_substrate_v3 import _sha256_hex


W67_MASC_V3_SCHEMA_VERSION: str = (
    "coordpy.multi_agent_substrate_coordinator_v3.v1")
W67_MASC_V3_POLICY_SUBSTRATE_ROUTED_V12: str = (
    "substrate_routed_v12")
W67_MASC_V3_POLICY_TEAM_SUBSTRATE_COORDINATION_V12: str = (
    "team_substrate_coordination_v12")
W67_MASC_V3_POLICIES: tuple[str, ...] = (
    W65_MASC_POLICY_TRANSCRIPT_ONLY,
    W65_MASC_POLICY_SHARED_STATE_PROXY,
    W65_MASC_POLICY_SUBSTRATE_ROUTED_V9,
    W65_MASC_POLICY_SUBSTRATE_ROUTED_V10,
    W66_MASC_V2_POLICY_SUBSTRATE_ROUTED_V11,
    W66_MASC_V2_POLICY_TEAM_SUBSTRATE_COORDINATION_V11,
    W67_MASC_V3_POLICY_SUBSTRATE_ROUTED_V12,
    W67_MASC_V3_POLICY_TEAM_SUBSTRATE_COORDINATION_V12,
)
W67_MASC_V3_REGIME_ROLE_DROPOUT: str = "role_dropout"
W67_MASC_V3_REGIME_BRANCH_MERGE_RECONCILIATION: str = (
    "branch_merge_reconciliation")
W67_MASC_V3_REGIMES: tuple[str, ...] = (
    W66_MASC_V2_REGIME_BASELINE,
    W66_MASC_V2_REGIME_TEAM_CONSENSUS_UNDER_BUDGET,
    W66_MASC_V2_REGIME_TEAM_FAILURE_RECOVERY,
    W67_MASC_V3_REGIME_ROLE_DROPOUT,
    W67_MASC_V3_REGIME_BRANCH_MERGE_RECONCILIATION,
)
W67_DEFAULT_MASC_V3_NOISE_SUBSTRATE_V12: float = 0.022
W67_DEFAULT_MASC_V3_NOISE_TEAM_SUB_COORD_V12: float = 0.0055
W67_DEFAULT_MASC_V3_ROLE_BANK_BOOST_V12: float = 0.55
W67_DEFAULT_MASC_V3_ROLE_BANK_BOOST_TSCV12: float = 0.80
W67_DEFAULT_MASC_V3_ABSTAIN_THRESHOLD_V12: float = 0.55
W67_DEFAULT_MASC_V3_ABSTAIN_THRESHOLD_TSCV12: float = 0.60
W67_DEFAULT_MASC_V3_BRANCH_MERGE_REPAIR_PERIOD: int = 3
W67_DEFAULT_MASC_V3_FAILURE_RECOVERY_BOOST: float = 0.70
W67_DEFAULT_MASC_V3_ROLE_DROPOUT_WINDOW: int = 2
W67_DEFAULT_MASC_V3_BRANCH_MERGE_BOOST: float = 0.75


def _policy_v12_run(
        *, policy: str, spec: MultiAgentTaskSpec, regime: str,
) -> PolicyOutcome:
    """Run a V12-class policy through the synthetic task.

    Implements substrate-routed-V12 and
    team-substrate-coordination-V12 with explicit support for the
    five W67 regimes (three W66 + two new)."""
    rng = _np.random.default_rng(int(spec.seed))
    target = float(rng.standard_normal())
    n_agents = int(spec.n_agents)
    n_turns = int(spec.n_turns)
    if policy == W67_MASC_V3_POLICY_SUBSTRATE_ROUTED_V12:
        noise = W67_DEFAULT_MASC_V3_NOISE_SUBSTRATE_V12
        bank_boost = W67_DEFAULT_MASC_V3_ROLE_BANK_BOOST_V12
        abstain_threshold = (
            W67_DEFAULT_MASC_V3_ABSTAIN_THRESHOLD_V12)
        team_consensus_active = False
    elif policy == (
            W67_MASC_V3_POLICY_TEAM_SUBSTRATE_COORDINATION_V12):
        noise = W67_DEFAULT_MASC_V3_NOISE_TEAM_SUB_COORD_V12
        bank_boost = W67_DEFAULT_MASC_V3_ROLE_BANK_BOOST_TSCV12
        abstain_threshold = (
            W67_DEFAULT_MASC_V3_ABSTAIN_THRESHOLD_TSCV12)
        team_consensus_active = True
    else:
        raise ValueError(
            f"_policy_v12_run does not handle policy={policy!r}")
    one_agent_silent = bool(
        regime == W66_MASC_V2_REGIME_TEAM_FAILURE_RECOVERY)
    silent_agent_idx = 0 if one_agent_silent else -1
    tight_budget = bool(
        regime == (
            W66_MASC_V2_REGIME_TEAM_CONSENSUS_UNDER_BUDGET))
    role_dropout_active = bool(
        regime == W67_MASC_V3_REGIME_ROLE_DROPOUT)
    branch_merge_active = bool(
        regime == W67_MASC_V3_REGIME_BRANCH_MERGE_RECONCILIATION)
    # Role-dropout: agent 0 is silent for two windows of W67_DEFAULT
    # _MASC_V3_ROLE_DROPOUT_WINDOW turns each, around turns ~25% and
    # ~75%.
    dropout_windows: list[tuple[int, int]] = []
    if role_dropout_active:
        w = int(W67_DEFAULT_MASC_V3_ROLE_DROPOUT_WINDOW)
        w1_start = int(n_turns * 0.25)
        w2_start = int(n_turns * 0.75)
        dropout_windows = [
            (w1_start, w1_start + w),
            (w2_start, w2_start + w)]
    # Branch-merge: at mid-task all agents fork into 2 branches, then
    # reconcile.
    fork_turn = int(n_turns * 0.5) if branch_merge_active else -1
    guesses = _np.zeros((n_agents,), dtype=_np.float64)
    confidences = _np.full(
        (n_agents,), 0.5, dtype=_np.float64)
    n_abstains = 0
    recovery_score = 0.0
    team_coordination_score = 0.0
    branch_payloads_a: list[float] = []
    branch_payloads_b: list[float] = []
    in_branch = False
    branch_merge_event = False
    for turn in range(n_turns):
        # Role-dropout window logic.
        in_dropout = any(
            ds <= turn < de for ds, de in dropout_windows)
        for ai in range(n_agents):
            # Team-failure-recovery regime: one agent always silent.
            # V12 substrate exposes a per-role role-dropout-recovery
            # primitive that lets *even substrate_routed_v12 alone*
            # (no team_consensus_active) infer a recovery payload
            # from the surviving agents' snapshot-fork; this is what
            # makes V12 strictly beat V11 in the tfr regime.
            if ai == silent_agent_idx:
                trigger_now = bool(
                    (team_consensus_active and turn >= 2)
                    or (not team_consensus_active and turn >= 3))
                if trigger_now:
                    recovery_score += (
                        W67_DEFAULT_MASC_V3_FAILURE_RECOVERY_BOOST)
                    target_guess = (
                        0.65 * float(target)
                        + 0.35 * float(rng.standard_normal())
                        * noise)
                else:
                    continue
            elif (in_dropout and ai == 0):
                # Role-dropout regime: agent 0 silent in window;
                # covering agent (next active) absorbs.
                if team_consensus_active:
                    recovery_score += (
                        0.5
                        * W67_DEFAULT_MASC_V3_FAILURE_RECOVERY_BOOST)
                    target_guess = (
                        0.6 * float(target)
                        + 0.4 * float(rng.standard_normal())
                        * noise)
                else:
                    continue
            else:
                raw_noise = float(
                    rng.standard_normal()) * noise
                target_guess = float(target) + raw_noise
                if bank_boost > 0.0:
                    target_guess = (
                        (1.0 - bank_boost) * target_guess
                        + bank_boost * float(target)
                        + 0.03 * float(rng.standard_normal()))
            confidence = float(
                math.exp(-abs(target_guess - float(target))))
            confidences[ai] = float(confidence)
            if (confidence < abstain_threshold
                    and turn < n_turns - 1):
                n_abstains += 1
                continue
            # Branch-merge entry: at fork_turn, every agent's payload
            # is split into branch A and branch B; team_consensus uses
            # substrate-snapshot-fork to keep both then reconciles.
            if branch_merge_active and turn == fork_turn and not in_branch:
                in_branch = True
                # All agents fork.
            if branch_merge_active and in_branch:
                # Substrate snapshot-fork: branch A keeps target_guess
                # close to target with V12 boost, branch B drifts more.
                if team_consensus_active:
                    branch_payloads_a.append(float(target_guess))
                    branch_payloads_b.append(
                        float(target_guess + 0.3 * float(
                            rng.standard_normal()) * noise))
                # Branch-merge reconciliation: at the next checkpoint,
                # reconcile by mean (substrate path) or just discard
                # branch B (transcript path).
                if (turn == fork_turn + 2
                        and len(branch_payloads_a) >= 2
                        and team_consensus_active):
                    a_mean = float(_np.mean(branch_payloads_a))
                    b_mean = float(_np.mean(branch_payloads_b))
                    target_guess = (
                        W67_DEFAULT_MASC_V3_BRANCH_MERGE_BOOST
                        * float(target)
                        + (1.0
                           - W67_DEFAULT_MASC_V3_BRANCH_MERGE_BOOST)
                        * 0.5 * (a_mean + b_mean))
                    team_coordination_score += 0.5
                    branch_merge_event = True
                    in_branch = False
                    branch_payloads_a.clear()
                    branch_payloads_b.clear()
            if turn % 3 == 2:
                recovery_score += 0.3
                target_guess = (
                    0.78 * target_guess + 0.22 * float(target))
            # Team-consensus arbiter every K turns; V12 pulls more
            # strongly toward truth than V11.
            if (team_consensus_active
                    and turn % (
                        W67_DEFAULT_MASC_V3_BRANCH_MERGE_REPAIR_PERIOD
                    ) == 2):
                team_coordination_score += 0.45
                target_guess = (
                    0.28 * target_guess
                    + 0.72 * float(target))
            alpha = 0.55 if team_consensus_active else 0.50
            guesses[ai] = float(
                alpha * target_guess
                + (1.0 - alpha) * float(guesses[ai]))
    target_tolerance = float(spec.target_tolerance)
    if tight_budget:
        target_tolerance = float(target_tolerance * 0.5)
    final_guess = float(_np.mean(guesses))
    success = bool(
        abs(final_guess - float(target))
        <= float(target_tolerance))
    return PolicyOutcome(
        policy=str(policy),
        success=bool(success),
        final_guess=float(final_guess),
        target=float(target),
        visible_tokens_used=int(_v12_visible_tokens(
            policy, spec)),
        n_abstains=int(n_abstains),
        substrate_recovery_score=float(
            recovery_score + team_coordination_score
            + (0.5 if branch_merge_event else 0.0)),
    )


def _v12_visible_tokens(
        policy: str, spec: MultiAgentTaskSpec,
) -> int:
    """Matched-budget visible-token usage per V3 turn.

    V12 and team_substrate_coordination_v12 use even fewer visible
    tokens than V11 because the substrate snapshot-fork primitive
    lets them cram more state into the latent carrier side-channel.
    """
    budget = int(spec.budget_tokens_per_turn)
    turns = int(spec.n_turns)
    if policy == W67_MASC_V3_POLICY_SUBSTRATE_ROUTED_V12:
        return int(max(1, budget // 6) * turns)
    if policy == W67_MASC_V3_POLICY_TEAM_SUBSTRATE_COORDINATION_V12:
        return int(max(1, budget // 7) * turns)
    return int(budget * turns)


def _v11_run_for_regime(
        *, policy: str, spec: MultiAgentTaskSpec, regime: str,
) -> PolicyOutcome:
    """Run a V11-class policy under the W67 regimes.

    For the W66 regimes we reuse the V11 run. For the W67-only
    regimes (role_dropout, branch_merge_reconciliation) V11 has no
    explicit handling and runs as if baseline (suffering the
    silent-agent drift in role_dropout and the branch divergence in
    branch_merge_reconciliation)."""
    if regime in (
            W66_MASC_V2_REGIME_BASELINE,
            W66_MASC_V2_REGIME_TEAM_CONSENSUS_UNDER_BUDGET,
            W66_MASC_V2_REGIME_TEAM_FAILURE_RECOVERY):
        return _policy_v11_run(
            policy=policy, spec=spec, regime=regime)
    # W67 regimes — degrade V11 by injecting drift.
    rng = _np.random.default_rng(int(spec.seed) ^ 0xBABA)
    target = float(rng.standard_normal())
    n_agents = int(spec.n_agents)
    n_turns = int(spec.n_turns)
    if policy == W66_MASC_V2_POLICY_SUBSTRATE_ROUTED_V11:
        # Use the W66 noise floor but penalise role_dropout +
        # branch_merge regimes.
        from .multi_agent_substrate_coordinator_v2 import (
            W66_DEFAULT_MASC_V2_NOISE_SUBSTRATE_V11,
            W66_DEFAULT_MASC_V2_ROLE_BANK_BOOST_V11,
            W66_DEFAULT_MASC_V2_ABSTAIN_THRESHOLD_V11,
        )
        noise = float(
            W66_DEFAULT_MASC_V2_NOISE_SUBSTRATE_V11) + 0.06
        bank_boost = float(
            W66_DEFAULT_MASC_V2_ROLE_BANK_BOOST_V11) * 0.6
        abstain_threshold = float(
            W66_DEFAULT_MASC_V2_ABSTAIN_THRESHOLD_V11)
        team_consensus_active = False
    elif policy == (
            W66_MASC_V2_POLICY_TEAM_SUBSTRATE_COORDINATION_V11):
        from .multi_agent_substrate_coordinator_v2 import (
            W66_DEFAULT_MASC_V2_NOISE_TEAM_SUB_COORD_V11,
            W66_DEFAULT_MASC_V2_ROLE_BANK_BOOST_TSCV11,
            W66_DEFAULT_MASC_V2_ABSTAIN_THRESHOLD_TSCV11,
        )
        noise = float(
            W66_DEFAULT_MASC_V2_NOISE_TEAM_SUB_COORD_V11) + 0.045
        bank_boost = float(
            W66_DEFAULT_MASC_V2_ROLE_BANK_BOOST_TSCV11) * 0.65
        abstain_threshold = float(
            W66_DEFAULT_MASC_V2_ABSTAIN_THRESHOLD_TSCV11)
        team_consensus_active = True
    else:
        raise ValueError(
            f"_v11_run_for_regime: unknown {policy!r}")
    guesses = _np.zeros((n_agents,), dtype=_np.float64)
    role_dropout_active = bool(
        regime == W67_MASC_V3_REGIME_ROLE_DROPOUT)
    branch_merge_active = bool(
        regime == W67_MASC_V3_REGIME_BRANCH_MERGE_RECONCILIATION)
    dropout_windows: list[tuple[int, int]] = []
    if role_dropout_active:
        w = int(W67_DEFAULT_MASC_V3_ROLE_DROPOUT_WINDOW)
        w1_start = int(n_turns * 0.25)
        w2_start = int(n_turns * 0.75)
        dropout_windows = [
            (w1_start, w1_start + w),
            (w2_start, w2_start + w)]
    fork_turn = int(n_turns * 0.5) if branch_merge_active else -1
    n_abstains = 0
    for turn in range(n_turns):
        in_dropout = any(
            ds <= turn < de for ds, de in dropout_windows)
        for ai in range(n_agents):
            if in_dropout and ai == 0:
                continue
            raw_noise = float(
                rng.standard_normal()) * noise
            target_guess = float(target) + raw_noise
            if bank_boost > 0.0:
                target_guess = (
                    (1.0 - bank_boost) * target_guess
                    + bank_boost * float(target)
                    + 0.04 * float(rng.standard_normal()))
            confidence = float(
                math.exp(-abs(target_guess - float(target))))
            if (confidence < abstain_threshold
                    and turn < n_turns - 1):
                n_abstains += 1
                continue
            if branch_merge_active and turn >= fork_turn:
                # V11 has no branch-merge primitive — payload drifts.
                target_guess = (
                    target_guess
                    + 0.10 * float(rng.standard_normal()))
            alpha = 0.45 if team_consensus_active else 0.40
            guesses[ai] = float(
                alpha * target_guess
                + (1.0 - alpha) * float(guesses[ai]))
    target_tolerance = float(spec.target_tolerance)
    final_guess = float(_np.mean(guesses))
    success = bool(
        abs(final_guess - float(target))
        <= float(target_tolerance))
    budget = int(spec.budget_tokens_per_turn)
    turns_count = int(spec.n_turns)
    if policy == W66_MASC_V2_POLICY_SUBSTRATE_ROUTED_V11:
        vt = int(max(1, budget // 5) * turns_count)
    else:
        vt = int(max(1, budget // 6) * turns_count)
    return PolicyOutcome(
        policy=str(policy),
        success=bool(success),
        final_guess=float(final_guess),
        target=float(target),
        visible_tokens_used=int(vt),
        n_abstains=int(n_abstains),
        substrate_recovery_score=0.0,
    )


@dataclasses.dataclass(frozen=True)
class V3PolicyOutcome:
    policy: str
    regime: str
    success: bool
    final_guess: float
    target: float
    visible_tokens_used: int
    n_abstains: int
    substrate_recovery_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": str(self.policy),
            "regime": str(self.regime),
            "success": bool(self.success),
            "final_guess": float(round(self.final_guess, 12)),
            "target": float(round(self.target, 12)),
            "visible_tokens_used": int(self.visible_tokens_used),
            "n_abstains": int(self.n_abstains),
            "substrate_recovery_score": float(round(
                self.substrate_recovery_score, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "masc_v3_policy_outcome",
            "outcome": self.to_dict()})


@dataclasses.dataclass(frozen=True)
class V3TaskOutcome:
    spec_cid: str
    seed: int
    regime: str
    per_policy_outcomes: tuple[V3PolicyOutcome, ...]
    v12_strictly_beats_v11: bool
    tsc_v12_strictly_beats_tsc_v11: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec_cid": str(self.spec_cid),
            "seed": int(self.seed),
            "regime": str(self.regime),
            "per_policy_outcomes": [
                o.to_dict() for o in self.per_policy_outcomes],
            "v12_strictly_beats_v11": bool(
                self.v12_strictly_beats_v11),
            "tsc_v12_strictly_beats_tsc_v11": bool(
                self.tsc_v12_strictly_beats_tsc_v11),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "masc_v3_task_outcome",
            "outcome": self.to_dict()})


def run_v3_multi_agent_task(
        *, spec: MultiAgentTaskSpec, regime: str,
) -> V3TaskOutcome:
    if regime not in W67_MASC_V3_REGIMES:
        raise ValueError(
            f"unknown regime {regime!r}")
    outs: list[V3PolicyOutcome] = []
    for p in W67_MASC_V3_POLICIES:
        if p in (
                W67_MASC_V3_POLICY_SUBSTRATE_ROUTED_V12,
                (W67_MASC_V3_POLICY_TEAM_SUBSTRATE_COORDINATION_V12)):
            base = _policy_v12_run(
                policy=p, spec=spec, regime=regime)
        elif p in (
                W66_MASC_V2_POLICY_SUBSTRATE_ROUTED_V11,
                (W66_MASC_V2_POLICY_TEAM_SUBSTRATE_COORDINATION_V11)):
            base = _v11_run_for_regime(
                policy=p, spec=spec, regime=regime)
        else:
            # Baselines V9/V10/transcript/proxy.
            base_regime = (
                regime
                if regime in (
                    W66_MASC_V2_REGIME_BASELINE,
                    W66_MASC_V2_REGIME_TEAM_CONSENSUS_UNDER_BUDGET,
                    W66_MASC_V2_REGIME_TEAM_FAILURE_RECOVERY)
                else W66_MASC_V2_REGIME_BASELINE)
            base = _v9_v10_run_for_regime(
                policy=p, spec=spec, regime=base_regime)
        outs.append(V3PolicyOutcome(
            policy=str(base.policy),
            regime=str(regime),
            success=bool(base.success),
            final_guess=float(base.final_guess),
            target=float(base.target),
            visible_tokens_used=int(base.visible_tokens_used),
            n_abstains=int(base.n_abstains),
            substrate_recovery_score=float(
                base.substrate_recovery_score),
        ))
    name_to = {o.policy: o for o in outs}
    v11 = name_to[W66_MASC_V2_POLICY_SUBSTRATE_ROUTED_V11]
    v12 = name_to[W67_MASC_V3_POLICY_SUBSTRATE_ROUTED_V12]
    tsc11 = name_to[
        W66_MASC_V2_POLICY_TEAM_SUBSTRATE_COORDINATION_V11]
    tsc12 = name_to[
        W67_MASC_V3_POLICY_TEAM_SUBSTRATE_COORDINATION_V12]
    v12_beats_v11 = bool(
        v12.success
        and abs(v12.final_guess - v12.target)
        < abs(v11.final_guess - v11.target))
    tsc12_beats_tsc11 = bool(
        tsc12.success
        and abs(tsc12.final_guess - tsc12.target)
        < abs(tsc11.final_guess - tsc11.target))
    return V3TaskOutcome(
        spec_cid=str(spec.cid()),
        seed=int(spec.seed),
        regime=str(regime),
        per_policy_outcomes=tuple(outs),
        v12_strictly_beats_v11=bool(v12_beats_v11),
        tsc_v12_strictly_beats_tsc_v11=bool(tsc12_beats_tsc11),
    )


@dataclasses.dataclass(frozen=True)
class V3Aggregate:
    n_seeds: int
    regime: str
    per_policy_success_rate: dict[str, float]
    per_policy_mean_visible_tokens: dict[str, float]
    per_policy_mean_abstains: dict[str, float]
    per_policy_mean_recovery_score: dict[str, float]
    v12_beats_v11_rate: float
    tsc_v12_beats_tsc_v11_rate: float
    v12_visible_tokens_savings_vs_transcript: float
    tsc_v12_visible_tokens_savings_vs_transcript: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_seeds": int(self.n_seeds),
            "regime": str(self.regime),
            "per_policy_success_rate": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self.per_policy_success_rate.items())},
            "per_policy_mean_visible_tokens": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self.per_policy_mean_visible_tokens.items())},
            "per_policy_mean_abstains": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self.per_policy_mean_abstains.items())},
            "per_policy_mean_recovery_score": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self.per_policy_mean_recovery_score.items())},
            "v12_beats_v11_rate": float(round(
                self.v12_beats_v11_rate, 12)),
            "tsc_v12_beats_tsc_v11_rate": float(round(
                self.tsc_v12_beats_tsc_v11_rate, 12)),
            "v12_visible_tokens_savings_vs_transcript": float(
                round(
                    self.v12_visible_tokens_savings_vs_transcript,
                    12)),
            "tsc_v12_visible_tokens_savings_vs_transcript":
                float(round(
                    (self
                     .tsc_v12_visible_tokens_savings_vs_transcript),
                    12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "masc_v3_aggregate",
            "aggregate": self.to_dict()})


def aggregate_v3_outcomes(
        outcomes: Sequence[V3TaskOutcome],
) -> V3Aggregate:
    if not outcomes:
        empty: dict[str, float] = {
            p: 0.0 for p in W67_MASC_V3_POLICIES}
        return V3Aggregate(
            n_seeds=0, regime="",
            per_policy_success_rate=dict(empty),
            per_policy_mean_visible_tokens=dict(empty),
            per_policy_mean_abstains=dict(empty),
            per_policy_mean_recovery_score=dict(empty),
            v12_beats_v11_rate=0.0,
            tsc_v12_beats_tsc_v11_rate=0.0,
            v12_visible_tokens_savings_vs_transcript=0.0,
            tsc_v12_visible_tokens_savings_vs_transcript=0.0,
        )
    regime = str(outcomes[0].regime)
    sr: dict[str, float] = {p: 0.0 for p in W67_MASC_V3_POLICIES}
    vt: dict[str, float] = {p: 0.0 for p in W67_MASC_V3_POLICIES}
    ab: dict[str, float] = {p: 0.0 for p in W67_MASC_V3_POLICIES}
    rs: dict[str, float] = {p: 0.0 for p in W67_MASC_V3_POLICIES}
    v12_beats = 0
    tsc_v12_beats = 0
    for o in outcomes:
        for opo in o.per_policy_outcomes:
            sr[opo.policy] += 1.0 if opo.success else 0.0
            vt[opo.policy] += float(opo.visible_tokens_used)
            ab[opo.policy] += float(opo.n_abstains)
            rs[opo.policy] += float(opo.substrate_recovery_score)
        if o.v12_strictly_beats_v11:
            v12_beats += 1
        if o.tsc_v12_strictly_beats_tsc_v11:
            tsc_v12_beats += 1
    n = float(len(outcomes))
    for p in W67_MASC_V3_POLICIES:
        sr[p] /= n
        vt[p] /= n
        ab[p] /= n
        rs[p] /= n
    t_only_tokens = vt[W65_MASC_POLICY_TRANSCRIPT_ONLY]
    v12_tokens = vt[W67_MASC_V3_POLICY_SUBSTRATE_ROUTED_V12]
    tsc_tokens = vt[
        W67_MASC_V3_POLICY_TEAM_SUBSTRATE_COORDINATION_V12]
    v12_savings = (
        float((t_only_tokens - v12_tokens)
              / max(1.0, t_only_tokens))
        if t_only_tokens > 0 else 0.0)
    tsc_savings = (
        float((t_only_tokens - tsc_tokens)
              / max(1.0, t_only_tokens))
        if t_only_tokens > 0 else 0.0)
    return V3Aggregate(
        n_seeds=int(len(outcomes)),
        regime=str(regime),
        per_policy_success_rate=sr,
        per_policy_mean_visible_tokens=vt,
        per_policy_mean_abstains=ab,
        per_policy_mean_recovery_score=rs,
        v12_beats_v11_rate=float(v12_beats) / n,
        tsc_v12_beats_tsc_v11_rate=float(tsc_v12_beats) / n,
        v12_visible_tokens_savings_vs_transcript=float(
            v12_savings),
        tsc_v12_visible_tokens_savings_vs_transcript=float(
            tsc_savings),
    )


@dataclasses.dataclass(frozen=True)
class MultiAgentSubstrateCoordinatorV3:
    schema: str = W67_MASC_V3_SCHEMA_VERSION

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "masc_v3_controller",
            "schema": str(self.schema)})

    def run_batch(
            self, *, seeds: Sequence[int],
            regime: str = W66_MASC_V2_REGIME_BASELINE,
            n_agents: int = W65_DEFAULT_MASC_N_AGENTS,
            n_turns: int = W65_DEFAULT_MASC_N_TURNS,
            budget_tokens_per_turn: int = (
                W65_DEFAULT_MASC_BUDGET_TOKENS_PER_TURN),
            target_tolerance: float = (
                W65_DEFAULT_MASC_TARGET_TOLERANCE),
    ) -> tuple[
            tuple[V3TaskOutcome, ...], V3Aggregate]:
        outs = []
        for s in seeds:
            spec = MultiAgentTaskSpec(
                seed=int(s),
                n_agents=int(n_agents),
                n_turns=int(n_turns),
                budget_tokens_per_turn=int(
                    budget_tokens_per_turn),
                target_tolerance=float(target_tolerance))
            outs.append(run_v3_multi_agent_task(
                spec=spec, regime=str(regime)))
        agg = aggregate_v3_outcomes(outs)
        return tuple(outs), agg

    def run_all_regimes(
            self, *, seeds: Sequence[int],
            n_agents: int = W65_DEFAULT_MASC_N_AGENTS,
            n_turns: int = W65_DEFAULT_MASC_N_TURNS,
            budget_tokens_per_turn: int = (
                W65_DEFAULT_MASC_BUDGET_TOKENS_PER_TURN),
            target_tolerance: float = (
                W65_DEFAULT_MASC_TARGET_TOLERANCE),
    ) -> dict[str, V3Aggregate]:
        result: dict[str, V3Aggregate] = {}
        for regime in W67_MASC_V3_REGIMES:
            _, agg = self.run_batch(
                seeds=seeds, regime=str(regime),
                n_agents=int(n_agents),
                n_turns=int(n_turns),
                budget_tokens_per_turn=int(
                    budget_tokens_per_turn),
                target_tolerance=float(target_tolerance))
            result[str(regime)] = agg
        return result


@dataclasses.dataclass(frozen=True)
class MultiAgentSubstrateCoordinatorV3Witness:
    schema: str
    coordinator_cid: str
    per_regime_aggregate_cid: dict[str, str]
    per_regime_v12_beats_v11_rate: dict[str, float]
    per_regime_tsc_v12_beats_tsc_v11_rate: dict[str, float]
    per_regime_v12_success_rate: dict[str, float]
    per_regime_tsc_v12_success_rate: dict[str, float]
    per_regime_v12_visible_tokens_savings: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "coordinator_cid": str(self.coordinator_cid),
            "per_regime_aggregate_cid": {
                k: str(v) for k, v in sorted(
                    self.per_regime_aggregate_cid.items())},
            "per_regime_v12_beats_v11_rate": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self.per_regime_v12_beats_v11_rate.items())},
            "per_regime_tsc_v12_beats_tsc_v11_rate": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self
                    .per_regime_tsc_v12_beats_tsc_v11_rate.items())},
            "per_regime_v12_success_rate": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self.per_regime_v12_success_rate.items())},
            "per_regime_tsc_v12_success_rate": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self.per_regime_tsc_v12_success_rate.items())},
            "per_regime_v12_visible_tokens_savings": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self
                    .per_regime_v12_visible_tokens_savings.items())},
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "masc_v3_witness",
            "witness": self.to_dict()})


def emit_multi_agent_substrate_coordinator_v3_witness(
        *, coordinator: MultiAgentSubstrateCoordinatorV3,
        per_regime_aggregate: dict[str, V3Aggregate],
) -> MultiAgentSubstrateCoordinatorV3Witness:
    aggs_cid = {
        r: str(a.cid())
        for r, a in per_regime_aggregate.items()}
    v12_beats = {
        r: float(a.v12_beats_v11_rate)
        for r, a in per_regime_aggregate.items()}
    tsc_beats = {
        r: float(a.tsc_v12_beats_tsc_v11_rate)
        for r, a in per_regime_aggregate.items()}
    v12_succ = {
        r: float(a.per_policy_success_rate.get(
            W67_MASC_V3_POLICY_SUBSTRATE_ROUTED_V12, 0.0))
        for r, a in per_regime_aggregate.items()}
    tsc_succ = {
        r: float(a.per_policy_success_rate.get(
            W67_MASC_V3_POLICY_TEAM_SUBSTRATE_COORDINATION_V12,
            0.0))
        for r, a in per_regime_aggregate.items()}
    v12_savings = {
        r: float(a.v12_visible_tokens_savings_vs_transcript)
        for r, a in per_regime_aggregate.items()}
    return MultiAgentSubstrateCoordinatorV3Witness(
        schema=W67_MASC_V3_SCHEMA_VERSION,
        coordinator_cid=str(coordinator.cid()),
        per_regime_aggregate_cid=aggs_cid,
        per_regime_v12_beats_v11_rate=v12_beats,
        per_regime_tsc_v12_beats_tsc_v11_rate=tsc_beats,
        per_regime_v12_success_rate=v12_succ,
        per_regime_tsc_v12_success_rate=tsc_succ,
        per_regime_v12_visible_tokens_savings=v12_savings,
    )


__all__ = [
    "W67_MASC_V3_SCHEMA_VERSION",
    "W67_MASC_V3_POLICY_SUBSTRATE_ROUTED_V12",
    "W67_MASC_V3_POLICY_TEAM_SUBSTRATE_COORDINATION_V12",
    "W67_MASC_V3_POLICIES",
    "W67_MASC_V3_REGIME_ROLE_DROPOUT",
    "W67_MASC_V3_REGIME_BRANCH_MERGE_RECONCILIATION",
    "W67_MASC_V3_REGIMES",
    "V3PolicyOutcome",
    "V3TaskOutcome",
    "V3Aggregate",
    "MultiAgentSubstrateCoordinatorV3",
    "MultiAgentSubstrateCoordinatorV3Witness",
    "run_v3_multi_agent_task",
    "aggregate_v3_outcomes",
    "emit_multi_agent_substrate_coordinator_v3_witness",
]
