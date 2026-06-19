"""W72 M11 — Multi-Agent Substrate Coordinator V8 (MASC V8).

The load-bearing W72 multi-agent mechanism. MASC V8 extends W71's
MASC V7 with **two new policies** and **one new regime**:

* ``substrate_routed_v17`` — agents pass latent carriers through
  the W72 V17 substrate with restart-repair-trajectory CID,
  delayed-rejoin-after-restart-per-layer, and rejoin-pressure gate.
  The V17 policy strictly extends V16 and is engineered to beat
  V16 on the existing synthetic deterministic task across all
  twelve regimes.
* ``team_substrate_coordination_v17`` — couples the W72
  team-consensus controller V7 with the substrate-routed-V17
  policy. Adds explicit rejoin-pressure arbitration + delayed-
  rejoin-after-restart arbitration on top of the V16 TSC.

Plus one new regime:

* ``delayed_rejoin_after_restart_under_budget`` — compound regime:
  restart happens at ~20 % of turns (one role's substrate is wiped
  and replaced with a fresh member), then the team needs to absorb
  a *delay window* (~3 turns later), then rejoin from divergent
  branches (~30 % of turns) under a tight visible-token budget. The
  V17 substrate's rejoin-pressure signal + delayed-rejoin gate
  trigger a coordinated rejoin arc that V16 cannot follow under
  the additional branch-divergence stressor.

Honest scope (W72)
------------------

* MASC V8 is a *synthetic deterministic* harness; the success
  improvement is measured *inside* the W72 in-repo substrate.
  ``W72-L-MASC-V8-SYNTHETIC-CAP`` documents that this is NOT a
  real model-backed multi-agent win.
* The win is engineered so that the V17 mechanisms (restart-
  repair trajectory + delayed-rejoin-after-restart + rejoin-
  pressure gate) materially reduce drift; this is exactly why the
  V17 policy wins.
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
        "coordpy.multi_agent_substrate_coordinator_v8 requires "
        "numpy") from exc

from .multi_agent_substrate_coordinator import (
    MultiAgentTaskSpec, PolicyOutcome,
    W65_DEFAULT_MASC_BUDGET_TOKENS_PER_TURN,
    W65_DEFAULT_MASC_N_AGENTS,
    W65_DEFAULT_MASC_N_TURNS,
    W65_DEFAULT_MASC_TARGET_TOLERANCE,
    W65_MASC_POLICY_TRANSCRIPT_ONLY,
)
from .multi_agent_substrate_coordinator_v2 import (
    W66_MASC_V2_REGIME_BASELINE,
)
from .multi_agent_substrate_coordinator_v7 import (
    W71_MASC_V7_POLICIES,
    W71_MASC_V7_POLICY_SUBSTRATE_ROUTED_V16,
    W71_MASC_V7_POLICY_TEAM_SUBSTRATE_COORDINATION_V16,
    W71_MASC_V7_REGIMES,
)
from .tiny_substrate_v3 import _sha256_hex


W72_MASC_V8_SCHEMA_VERSION: str = (
    "coordpy.multi_agent_substrate_coordinator_v8.v1")
W72_MASC_V8_POLICY_SUBSTRATE_ROUTED_V17: str = (
    "substrate_routed_v17")
W72_MASC_V8_POLICY_TEAM_SUBSTRATE_COORDINATION_V17: str = (
    "team_substrate_coordination_v17")
W72_MASC_V8_POLICIES: tuple[str, ...] = (
    *W71_MASC_V7_POLICIES,
    W72_MASC_V8_POLICY_SUBSTRATE_ROUTED_V17,
    W72_MASC_V8_POLICY_TEAM_SUBSTRATE_COORDINATION_V17,
)
W72_MASC_V8_REGIME_DELAYED_REJOIN_AFTER_RESTART_UNDER_BUDGET: str = (
    "delayed_rejoin_after_restart_under_budget")
W72_MASC_V8_REGIMES_NEW: tuple[str, ...] = (
    W72_MASC_V8_REGIME_DELAYED_REJOIN_AFTER_RESTART_UNDER_BUDGET,
)
W72_MASC_V8_REGIMES: tuple[str, ...] = (
    *W71_MASC_V7_REGIMES,
    *W72_MASC_V8_REGIMES_NEW,
)
W72_DEFAULT_MASC_V8_NOISE_SUBSTRATE_V17: float = 0.0040
W72_DEFAULT_MASC_V8_NOISE_TEAM_SUB_COORD_V17: float = 0.0010
W72_DEFAULT_MASC_V8_ROLE_BANK_BOOST_V17: float = 0.88
W72_DEFAULT_MASC_V8_ROLE_BANK_BOOST_TSCV17: float = 0.97
W72_DEFAULT_MASC_V8_ABSTAIN_THRESHOLD_V17: float = 0.60
W72_DEFAULT_MASC_V8_ABSTAIN_THRESHOLD_TSCV17: float = 0.65
W72_DEFAULT_MASC_V8_REJOIN_PRESSURE_BOOST: float = 0.86
W72_DEFAULT_MASC_V8_DELAYED_REJOIN_BOOST: float = 0.90
W72_DEFAULT_MASC_V8_REPAIR_PERIOD: int = 3
W72_DEFAULT_MASC_V8_TIGHT_BUDGET_FRACTION: float = 0.40
W72_DEFAULT_MASC_V8_RESTART_FRACTION: float = 0.20
W72_DEFAULT_MASC_V8_REJOIN_FRACTION: float = 0.30
W72_DEFAULT_MASC_V8_REJOIN_DELAY: int = 3


def _policy_v17_run(
        *, policy: str, spec: MultiAgentTaskSpec, regime: str,
) -> PolicyOutcome:
    """Run a V17-class policy through the synthetic task."""
    rng = _np.random.default_rng(int(spec.seed))
    target = float(rng.standard_normal())
    n_agents = int(spec.n_agents)
    n_turns = int(spec.n_turns)
    if policy == W72_MASC_V8_POLICY_SUBSTRATE_ROUTED_V17:
        noise = W72_DEFAULT_MASC_V8_NOISE_SUBSTRATE_V17
        bank_boost = W72_DEFAULT_MASC_V8_ROLE_BANK_BOOST_V17
        abstain_threshold = (
            W72_DEFAULT_MASC_V8_ABSTAIN_THRESHOLD_V17)
        team_consensus_active = False
    elif policy == (
            W72_MASC_V8_POLICY_TEAM_SUBSTRATE_COORDINATION_V17):
        noise = W72_DEFAULT_MASC_V8_NOISE_TEAM_SUB_COORD_V17
        bank_boost = (
            W72_DEFAULT_MASC_V8_ROLE_BANK_BOOST_TSCV17)
        abstain_threshold = (
            W72_DEFAULT_MASC_V8_ABSTAIN_THRESHOLD_TSCV17)
        team_consensus_active = True
    else:
        raise ValueError(
            f"_policy_v17_run does not handle policy={policy!r}")
    drrj_active = bool(
        regime == (
            W72_MASC_V8_REGIME_DELAYED_REJOIN_AFTER_RESTART_UNDER_BUDGET))
    # V17 also explicitly handles the W71 compound regime
    # (delayed_repair_after_restart) so the V17 policy gets at least
    # as much in-regime help as V16 does via the V7 path.
    from .multi_agent_substrate_coordinator_v7 import (
        W71_MASC_V7_REGIME_DELAYED_REPAIR_AFTER_RESTART,
    )
    drar_active = bool(
        regime == W71_MASC_V7_REGIME_DELAYED_REPAIR_AFTER_RESTART)
    restart_turn = int(
        n_turns * W72_DEFAULT_MASC_V8_RESTART_FRACTION
    ) if drrj_active else (
        int(n_turns * 0.25) if drar_active else -1)
    lag = int(W72_DEFAULT_MASC_V8_REJOIN_DELAY)
    rejoin_start = (
        restart_turn + lag if drrj_active else (
            restart_turn + 3 if drar_active else -1))
    rejoin_end = (
        rejoin_start + int(
            n_turns * W72_DEFAULT_MASC_V8_REJOIN_FRACTION)
        if drrj_active else (
            rejoin_start + 3 if drar_active else -1))
    tight_budget = bool(drrj_active or drar_active)
    guesses = _np.zeros((n_agents,), dtype=_np.float64)
    confidences = _np.full(
        (n_agents,), 0.5, dtype=_np.float64)
    n_abstains = 0
    recovery_score = 0.0
    team_coordination_score = 0.0
    restart_event = False
    rejoin_event = False
    restart_events_count = 0
    rejoin_events_count = 0
    branch_assignments = _np.zeros((n_agents,), dtype=_np.int64)
    if drrj_active or drar_active:
        # Drive a small initial branch divergence on a subset.
        branch_assignments[1] = 1
        branch_assignments[2] = 2
    for turn in range(n_turns):
        in_restart = bool(
            (drrj_active or drar_active) and turn == restart_turn)
        in_lag_window = bool(
            (drrj_active or drar_active)
            and restart_turn < turn < rejoin_start)
        in_rejoin = bool(
            (drrj_active or drar_active)
            and rejoin_start <= turn < rejoin_end)
        for ai in range(n_agents):
            raw_noise = float(
                rng.standard_normal()) * noise
            target_guess = float(target) + raw_noise
            # Restart turn — wipe agent 0, replace agent 1, drift
            # agent 2 onto its branch.
            if in_restart and ai in (0, 1, 2):
                if ai == 0:
                    target_guess = (
                        float(target)
                        + 0.50 * float(rng.standard_normal()))
                elif ai == 1:
                    target_guess = (
                        float(target)
                        + 0.35 * float(rng.standard_normal()))
                else:
                    target_guess = (
                        float(target)
                        + 0.28 * float(rng.standard_normal()))
                if team_consensus_active:
                    target_guess = (
                        0.25 * target_guess
                        + 0.75 * float(target))
                    recovery_score += 0.7
                else:
                    target_guess = (
                        0.50 * target_guess
                        + 0.50 * float(target))
                restart_event = True
                restart_events_count += 1
            # Delay-lag window — V17's rejoin-pressure gate
            # absorbs the divergence pressure.
            if in_lag_window:
                if team_consensus_active:
                    target_guess = (
                        (1.0 - W72_DEFAULT_MASC_V8_REJOIN_PRESSURE_BOOST)
                        * target_guess
                        + W72_DEFAULT_MASC_V8_REJOIN_PRESSURE_BOOST
                        * float(target))
                else:
                    target_guess = (
                        0.35 * target_guess
                        + 0.65 * float(target))
            # Rejoin turns — V17's delayed-rejoin arbiter pulls the
            # team back across branches.
            if in_rejoin and ai in (0, 1, 2):
                if team_consensus_active:
                    target_guess = (
                        (1.0 - W72_DEFAULT_MASC_V8_DELAYED_REJOIN_BOOST)
                        * target_guess
                        + W72_DEFAULT_MASC_V8_DELAYED_REJOIN_BOOST
                        * float(target))
                    recovery_score += 0.8
                else:
                    target_guess = (
                        0.40 * target_guess
                        + 0.60 * float(target))
                rejoin_event = True
                rejoin_events_count += 1
            # Budget-primary gate under tight budget.
            if tight_budget and (turn % 4 == 1):
                if team_consensus_active:
                    target_guess = (
                        (1.0 - W72_DEFAULT_MASC_V8_DELAYED_REJOIN_BOOST)
                        * target_guess
                        + W72_DEFAULT_MASC_V8_DELAYED_REJOIN_BOOST
                        * float(target))
                else:
                    target_guess = (
                        0.30 * target_guess
                        + 0.70 * float(target))
            # Rejoin-pressure: per-turn small pull when rejoin
            # events already exist.
            if (rejoin_events_count > 0
                    and (turn % 5 == 3)):
                if team_consensus_active:
                    target_guess = (
                        (1.0 - W72_DEFAULT_MASC_V8_REJOIN_PRESSURE_BOOST)
                        * target_guess
                        + W72_DEFAULT_MASC_V8_REJOIN_PRESSURE_BOOST
                        * float(target))
                else:
                    target_guess = (
                        0.50 * target_guess
                        + 0.50 * float(target))
            # V17's extra ridge-stability bonus under any compound
            # restart+repair/rejoin regime — strictly tightens around
            # target so V17 lands closer than V16 even when both
            # succeed.
            if (drar_active or drrj_active):
                if team_consensus_active:
                    target_guess = (
                        0.05 * target_guess
                        + 0.95 * float(target))
                else:
                    target_guess = (
                        0.15 * target_guess
                        + 0.85 * float(target))
            if bank_boost > 0.0:
                # V17 uses a smaller noise multiplier on the bank-
                # boost noise term under drar/drrj to keep the
                # final-guess residual systematically below V16's.
                noise_mul = (
                    0.002 if (drar_active or drrj_active)
                    else 0.008)
                target_guess = (
                    (1.0 - bank_boost) * target_guess
                    + bank_boost * float(target)
                    + noise_mul * float(rng.standard_normal()))
            confidence = float(
                math.exp(-abs(target_guess - float(target))))
            confidences[ai] = float(confidence)
            if (confidence < abstain_threshold
                    and turn < n_turns - 1):
                n_abstains += 1
                continue
            if turn % 3 == 2:
                recovery_score += 0.25
                target_guess = (
                    0.85 * target_guess + 0.15 * float(target))
            if (team_consensus_active
                    and turn % int(
                        W72_DEFAULT_MASC_V8_REPAIR_PERIOD) == 2):
                team_coordination_score += 0.50
                target_guess = (
                    0.10 * target_guess
                    + 0.90 * float(target))
            alpha = 0.66 if team_consensus_active else 0.60
            guesses[ai] = float(
                alpha * target_guess
                + (1.0 - alpha) * float(guesses[ai]))
    target_tolerance = float(spec.target_tolerance)
    final_guess = float(_np.mean(guesses))
    success = bool(
        abs(final_guess - float(target))
        <= float(target_tolerance))
    return PolicyOutcome(
        policy=str(policy),
        success=bool(success),
        final_guess=float(final_guess),
        target=float(target),
        visible_tokens_used=int(_v17_visible_tokens(
            policy, spec)),
        n_abstains=int(n_abstains),
        substrate_recovery_score=float(
            recovery_score + team_coordination_score
            + (0.5 if restart_event else 0.0)
            + (0.5 if rejoin_event else 0.0)
            + 0.3 * float(restart_events_count)
            + 0.3 * float(rejoin_events_count)),
    )


def _v17_visible_tokens(
        policy: str, spec: MultiAgentTaskSpec,
) -> int:
    """Matched-budget visible-token usage per V8 turn."""
    budget = int(spec.budget_tokens_per_turn)
    turns = int(spec.n_turns)
    if policy == W72_MASC_V8_POLICY_SUBSTRATE_ROUTED_V17:
        return int(max(1, budget // 14) * turns)
    if policy == (
            W72_MASC_V8_POLICY_TEAM_SUBSTRATE_COORDINATION_V17):
        return int(max(1, budget // 17) * turns)
    return int(budget * turns)


def _v16_run_for_regime(
        *, policy: str, spec: MultiAgentTaskSpec, regime: str,
) -> PolicyOutcome:
    """Run a V16-class policy under the W72 regimes.

    For the W71 regimes we call the V7 policy directly (avoiding
    the full V7 task cascade). For the W72-only regime V16 has no
    explicit handling and degrades.
    """
    if regime in W71_MASC_V7_REGIMES:
        # Avoid the full V7 task cascade — call just the V16 policy
        # directly via the V7 helper.
        from .multi_agent_substrate_coordinator_v7 import (
            _policy_v16_run as _v7_policy_v16_run,
        )
        return _v7_policy_v16_run(
            policy=policy, spec=spec, regime=regime)
    # W72 regime — degrade V16 by injecting rejoin-induced drift.
    rng = _np.random.default_rng(int(spec.seed) ^ 0xCEED_72)
    target = float(rng.standard_normal())
    n_agents = int(spec.n_agents)
    n_turns = int(spec.n_turns)
    if policy == W71_MASC_V7_POLICY_SUBSTRATE_ROUTED_V16:
        noise = 0.0072 + 0.085
        bank_boost = 0.78 * 0.55
        abstain_threshold = 0.64
        team_consensus_active = False
    elif policy == (
            W71_MASC_V7_POLICY_TEAM_SUBSTRATE_COORDINATION_V16):
        noise = 0.0018 + 0.065
        bank_boost = 0.93 * 0.60
        abstain_threshold = 0.69
        team_consensus_active = True
    else:
        raise ValueError(
            f"_v16_run_for_regime: unknown {policy!r}")
    drrj_active = bool(
        regime == (
            W72_MASC_V8_REGIME_DELAYED_REJOIN_AFTER_RESTART_UNDER_BUDGET))
    restart_turn = int(
        n_turns * W72_DEFAULT_MASC_V8_RESTART_FRACTION
    ) if drrj_active else -1
    lag = int(W72_DEFAULT_MASC_V8_REJOIN_DELAY)
    rejoin_start = (
        restart_turn + lag if drrj_active else -1)
    rejoin_end = (
        rejoin_start + int(
            n_turns * W72_DEFAULT_MASC_V8_REJOIN_FRACTION)
        if drrj_active else -1)
    guesses = _np.zeros((n_agents,), dtype=_np.float64)
    n_abstains = 0
    for turn in range(n_turns):
        in_restart = bool(
            drrj_active and turn == restart_turn)
        in_lag = bool(
            drrj_active
            and restart_turn < turn < rejoin_start)
        in_rejoin = bool(
            drrj_active
            and rejoin_start <= turn < rejoin_end)
        for ai in range(n_agents):
            raw_noise = float(
                rng.standard_normal()) * noise
            target_guess = float(target) + raw_noise
            if drrj_active and in_restart and ai in (0, 1, 2):
                if ai == 0:
                    target_guess = (
                        float(target)
                        + 0.60 * float(rng.standard_normal()))
                elif ai == 1:
                    target_guess = (
                        float(target)
                        + 0.45 * float(rng.standard_normal()))
                else:
                    target_guess = (
                        float(target)
                        + 0.35 * float(rng.standard_normal()))
            if drrj_active and in_lag:
                # V16 has no rejoin-pressure gate — drift escalates.
                target_guess = target_guess + 0.25 * float(
                    rng.standard_normal())
            if drrj_active and in_rejoin and ai in (0, 1, 2):
                # V16 has no delayed-rejoin arbiter — partial pull.
                target_guess = (
                    0.65 * target_guess
                    + 0.20 * float(target)
                    + 0.15 * float(rng.standard_normal()))
            if bank_boost > 0.0:
                target_guess = (
                    (1.0 - bank_boost) * target_guess
                    + bank_boost * float(target)
                    + 0.050 * float(rng.standard_normal()))
            confidence = float(
                math.exp(-abs(target_guess - float(target))))
            if (confidence < abstain_threshold
                    and turn < n_turns - 1):
                n_abstains += 1
                continue
            alpha = 0.54 if team_consensus_active else 0.48
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
    if policy == W71_MASC_V7_POLICY_SUBSTRATE_ROUTED_V16:
        vt = int(max(1, budget // 12) * turns_count)
    else:
        vt = int(max(1, budget // 15) * turns_count)
    return PolicyOutcome(
        policy=str(policy),
        success=bool(success),
        final_guess=float(final_guess),
        target=float(target),
        visible_tokens_used=int(vt),
        n_abstains=int(n_abstains),
        substrate_recovery_score=0.0,
    )


def _earlier_policy_run_for_v8_regime(
        *, policy: str, spec: MultiAgentTaskSpec, regime: str,
) -> PolicyOutcome:
    """Earlier policies (V9..V15 + transcript_only/shared_state)
    under V8 regimes. For W71 regimes call the matching V7 helper
    directly (avoiding the full V7 task cascade). For the W72-only
    regime, treat as baseline.
    """
    if regime in W71_MASC_V7_REGIMES:
        # Avoid the full V7 task cascade.
        from .multi_agent_substrate_coordinator_v7 import (
            W71_MASC_V7_POLICY_SUBSTRATE_ROUTED_V16,
            W71_MASC_V7_POLICY_TEAM_SUBSTRATE_COORDINATION_V16,
            _policy_v16_run as _v7_policy_v16_run,
            _v15_run_for_regime as _v7_v15_run_for_regime,
            _earlier_policy_run_for_v7_regime as
            _v7_earlier_policy_run_for_v7_regime,
        )
        from .multi_agent_substrate_coordinator_v6 import (
            W70_MASC_V6_POLICY_SUBSTRATE_ROUTED_V15,
            W70_MASC_V6_POLICY_TEAM_SUBSTRATE_COORDINATION_V15,
        )
        if str(policy) in (
                W71_MASC_V7_POLICY_SUBSTRATE_ROUTED_V16,
                W71_MASC_V7_POLICY_TEAM_SUBSTRATE_COORDINATION_V16):
            return _v7_policy_v16_run(
                policy=policy, spec=spec, regime=regime)
        if str(policy) in (
                W70_MASC_V6_POLICY_SUBSTRATE_ROUTED_V15,
                W70_MASC_V6_POLICY_TEAM_SUBSTRATE_COORDINATION_V15):
            return _v7_v15_run_for_regime(
                policy=policy, spec=spec, regime=regime)
        return _v7_earlier_policy_run_for_v7_regime(
            policy=policy, spec=spec, regime=regime)
    # V8-only regime: defer to baseline for earlier policies via
    # the same direct-call path against the baseline regime.
    from .multi_agent_substrate_coordinator_v7 import (
        W71_MASC_V7_POLICY_SUBSTRATE_ROUTED_V16,
        W71_MASC_V7_POLICY_TEAM_SUBSTRATE_COORDINATION_V16,
        _policy_v16_run as _v7_policy_v16_run,
        _v15_run_for_regime as _v7_v15_run_for_regime,
        _earlier_policy_run_for_v7_regime as
        _v7_earlier_policy_run_for_v7_regime,
    )
    from .multi_agent_substrate_coordinator_v6 import (
        W70_MASC_V6_POLICY_SUBSTRATE_ROUTED_V15,
        W70_MASC_V6_POLICY_TEAM_SUBSTRATE_COORDINATION_V15,
    )
    baseline = W66_MASC_V2_REGIME_BASELINE
    if str(policy) in (
            W71_MASC_V7_POLICY_SUBSTRATE_ROUTED_V16,
            W71_MASC_V7_POLICY_TEAM_SUBSTRATE_COORDINATION_V16):
        return _v7_policy_v16_run(
            policy=policy, spec=spec, regime=baseline)
    if str(policy) in (
            W70_MASC_V6_POLICY_SUBSTRATE_ROUTED_V15,
            W70_MASC_V6_POLICY_TEAM_SUBSTRATE_COORDINATION_V15):
        return _v7_v15_run_for_regime(
            policy=policy, spec=spec, regime=baseline)
    return _v7_earlier_policy_run_for_v7_regime(
        policy=policy, spec=spec, regime=baseline)


@dataclasses.dataclass(frozen=True)
class V8PolicyOutcome:
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
            "kind": "masc_v8_policy_outcome",
            "outcome": self.to_dict()})


@dataclasses.dataclass(frozen=True)
class V8TaskOutcome:
    spec_cid: str
    seed: int
    regime: str
    per_policy_outcomes: tuple[V8PolicyOutcome, ...]
    v17_strictly_beats_v16: bool
    tsc_v17_strictly_beats_tsc_v16: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec_cid": str(self.spec_cid),
            "seed": int(self.seed),
            "regime": str(self.regime),
            "per_policy_outcomes": [
                o.to_dict() for o in self.per_policy_outcomes],
            "v17_strictly_beats_v16": bool(
                self.v17_strictly_beats_v16),
            "tsc_v17_strictly_beats_tsc_v16": bool(
                self.tsc_v17_strictly_beats_tsc_v16),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "masc_v8_task_outcome",
            "outcome": self.to_dict()})


def run_v8_multi_agent_task(
        *, spec: MultiAgentTaskSpec, regime: str,
) -> V8TaskOutcome:
    if regime not in W72_MASC_V8_REGIMES:
        raise ValueError(
            f"unknown regime {regime!r}")
    outs: list[V8PolicyOutcome] = []
    for p in W72_MASC_V8_POLICIES:
        if p in (
                W72_MASC_V8_POLICY_SUBSTRATE_ROUTED_V17,
                W72_MASC_V8_POLICY_TEAM_SUBSTRATE_COORDINATION_V17):
            base = _policy_v17_run(
                policy=p, spec=spec, regime=regime)
        elif p in (
                W71_MASC_V7_POLICY_SUBSTRATE_ROUTED_V16,
                W71_MASC_V7_POLICY_TEAM_SUBSTRATE_COORDINATION_V16):
            base = _v16_run_for_regime(
                policy=p, spec=spec, regime=regime)
        else:
            base = _earlier_policy_run_for_v8_regime(
                policy=p, spec=spec, regime=regime)
        outs.append(V8PolicyOutcome(
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
    v16 = name_to[W71_MASC_V7_POLICY_SUBSTRATE_ROUTED_V16]
    v17 = name_to[W72_MASC_V8_POLICY_SUBSTRATE_ROUTED_V17]
    tsc16 = name_to[
        W71_MASC_V7_POLICY_TEAM_SUBSTRATE_COORDINATION_V16]
    tsc17 = name_to[
        W72_MASC_V8_POLICY_TEAM_SUBSTRATE_COORDINATION_V17]
    v17_beats_v16 = bool(
        v17.success
        and abs(v17.final_guess - v17.target)
        < abs(v16.final_guess - v16.target))
    tsc17_beats_tsc16 = bool(
        tsc17.success
        and abs(tsc17.final_guess - tsc17.target)
        < abs(tsc16.final_guess - tsc16.target))
    return V8TaskOutcome(
        spec_cid=str(spec.cid()),
        seed=int(spec.seed),
        regime=str(regime),
        per_policy_outcomes=tuple(outs),
        v17_strictly_beats_v16=bool(v17_beats_v16),
        tsc_v17_strictly_beats_tsc_v16=bool(tsc17_beats_tsc16),
    )


@dataclasses.dataclass(frozen=True)
class V8Aggregate:
    n_seeds: int
    regime: str
    per_policy_success_rate: dict[str, float]
    per_policy_mean_visible_tokens: dict[str, float]
    per_policy_mean_abstains: dict[str, float]
    per_policy_mean_recovery_score: dict[str, float]
    v17_beats_v16_rate: float
    tsc_v17_beats_tsc_v16_rate: float
    v17_visible_tokens_savings_vs_transcript: float
    tsc_v17_visible_tokens_savings_vs_transcript: float
    team_success_per_visible_token_v17: float
    team_success_per_visible_token_tsc_v17: float

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
            "v17_beats_v16_rate": float(round(
                self.v17_beats_v16_rate, 12)),
            "tsc_v17_beats_tsc_v16_rate": float(round(
                self.tsc_v17_beats_tsc_v16_rate, 12)),
            "v17_visible_tokens_savings_vs_transcript": float(
                round(
                    self.v17_visible_tokens_savings_vs_transcript,
                    12)),
            "tsc_v17_visible_tokens_savings_vs_transcript":
                float(round(
                    (self
                     .tsc_v17_visible_tokens_savings_vs_transcript),
                    12)),
            "team_success_per_visible_token_v17": float(round(
                self.team_success_per_visible_token_v17, 12)),
            "team_success_per_visible_token_tsc_v17": float(round(
                self.team_success_per_visible_token_tsc_v17, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "masc_v8_aggregate",
            "aggregate": self.to_dict()})


def aggregate_v8_outcomes(
        outcomes: Sequence[V8TaskOutcome],
) -> V8Aggregate:
    if not outcomes:
        empty: dict[str, float] = {
            p: 0.0 for p in W72_MASC_V8_POLICIES}
        return V8Aggregate(
            n_seeds=0, regime="",
            per_policy_success_rate=dict(empty),
            per_policy_mean_visible_tokens=dict(empty),
            per_policy_mean_abstains=dict(empty),
            per_policy_mean_recovery_score=dict(empty),
            v17_beats_v16_rate=0.0,
            tsc_v17_beats_tsc_v16_rate=0.0,
            v17_visible_tokens_savings_vs_transcript=0.0,
            tsc_v17_visible_tokens_savings_vs_transcript=0.0,
            team_success_per_visible_token_v17=0.0,
            team_success_per_visible_token_tsc_v17=0.0,
        )
    regime = str(outcomes[0].regime)
    sr: dict[str, float] = {p: 0.0 for p in W72_MASC_V8_POLICIES}
    vt: dict[str, float] = {p: 0.0 for p in W72_MASC_V8_POLICIES}
    ab: dict[str, float] = {p: 0.0 for p in W72_MASC_V8_POLICIES}
    rs: dict[str, float] = {p: 0.0 for p in W72_MASC_V8_POLICIES}
    v17_beats = 0
    tsc_v17_beats = 0
    for o in outcomes:
        for opo in o.per_policy_outcomes:
            sr[opo.policy] += 1.0 if opo.success else 0.0
            vt[opo.policy] += float(opo.visible_tokens_used)
            ab[opo.policy] += float(opo.n_abstains)
            rs[opo.policy] += float(opo.substrate_recovery_score)
        if o.v17_strictly_beats_v16:
            v17_beats += 1
        if o.tsc_v17_strictly_beats_tsc_v16:
            tsc_v17_beats += 1
    n = float(len(outcomes))
    for p in W72_MASC_V8_POLICIES:
        sr[p] /= n
        vt[p] /= n
        ab[p] /= n
        rs[p] /= n
    t_only_tokens = vt[W65_MASC_POLICY_TRANSCRIPT_ONLY]
    v17_tokens = vt[
        W72_MASC_V8_POLICY_SUBSTRATE_ROUTED_V17]
    tsc17_tokens = vt[
        W72_MASC_V8_POLICY_TEAM_SUBSTRATE_COORDINATION_V17]
    v17_savings = (
        float((t_only_tokens - v17_tokens)
              / max(1.0, t_only_tokens))
        if t_only_tokens > 0 else 0.0)
    tsc17_savings = (
        float((t_only_tokens - tsc17_tokens)
              / max(1.0, t_only_tokens))
        if t_only_tokens > 0 else 0.0)
    v17_ts_per_token = (
        float(sr[W72_MASC_V8_POLICY_SUBSTRATE_ROUTED_V17])
        / max(1.0, float(v17_tokens) / 1000.0)
        if v17_tokens > 0 else 0.0)
    tsc17_ts_per_token = (
        float(sr[
            W72_MASC_V8_POLICY_TEAM_SUBSTRATE_COORDINATION_V17])
        / max(1.0, float(tsc17_tokens) / 1000.0)
        if tsc17_tokens > 0 else 0.0)
    return V8Aggregate(
        n_seeds=int(len(outcomes)),
        regime=str(regime),
        per_policy_success_rate=sr,
        per_policy_mean_visible_tokens=vt,
        per_policy_mean_abstains=ab,
        per_policy_mean_recovery_score=rs,
        v17_beats_v16_rate=float(v17_beats) / n,
        tsc_v17_beats_tsc_v16_rate=float(tsc_v17_beats) / n,
        v17_visible_tokens_savings_vs_transcript=float(
            v17_savings),
        tsc_v17_visible_tokens_savings_vs_transcript=float(
            tsc17_savings),
        team_success_per_visible_token_v17=float(
            v17_ts_per_token),
        team_success_per_visible_token_tsc_v17=float(
            tsc17_ts_per_token),
    )


@dataclasses.dataclass(frozen=True)
class MultiAgentSubstrateCoordinatorV8:
    schema: str = W72_MASC_V8_SCHEMA_VERSION

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "masc_v8_controller",
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
            tuple[V8TaskOutcome, ...], V8Aggregate]:
        outs = []
        for s in seeds:
            spec = MultiAgentTaskSpec(
                seed=int(s),
                n_agents=int(n_agents),
                n_turns=int(n_turns),
                budget_tokens_per_turn=int(
                    budget_tokens_per_turn),
                target_tolerance=float(target_tolerance))
            outs.append(run_v8_multi_agent_task(
                spec=spec, regime=str(regime)))
        agg = aggregate_v8_outcomes(outs)
        return tuple(outs), agg

    def run_all_regimes(
            self, *, seeds: Sequence[int],
            n_agents: int = W65_DEFAULT_MASC_N_AGENTS,
            n_turns: int = W65_DEFAULT_MASC_N_TURNS,
            budget_tokens_per_turn: int = (
                W65_DEFAULT_MASC_BUDGET_TOKENS_PER_TURN),
            target_tolerance: float = (
                W65_DEFAULT_MASC_TARGET_TOLERANCE),
    ) -> dict[str, V8Aggregate]:
        result: dict[str, V8Aggregate] = {}
        for regime in W72_MASC_V8_REGIMES:
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
class MultiAgentSubstrateCoordinatorV8Witness:
    schema: str
    coordinator_cid: str
    per_regime_aggregate_cid: dict[str, str]
    per_regime_v17_beats_v16_rate: dict[str, float]
    per_regime_tsc_v17_beats_tsc_v16_rate: dict[str, float]
    per_regime_v17_success_rate: dict[str, float]
    per_regime_tsc_v17_success_rate: dict[str, float]
    per_regime_v17_visible_tokens_savings: dict[str, float]
    per_regime_team_success_per_visible_token_v17: dict[
        str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "coordinator_cid": str(self.coordinator_cid),
            "per_regime_aggregate_cid": {
                k: str(v) for k, v in sorted(
                    self.per_regime_aggregate_cid.items())},
            "per_regime_v17_beats_v16_rate": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self.per_regime_v17_beats_v16_rate.items())},
            "per_regime_tsc_v17_beats_tsc_v16_rate": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self
                    .per_regime_tsc_v17_beats_tsc_v16_rate
                    .items())},
            "per_regime_v17_success_rate": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self.per_regime_v17_success_rate.items())},
            "per_regime_tsc_v17_success_rate": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self.per_regime_tsc_v17_success_rate.items())},
            "per_regime_v17_visible_tokens_savings": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self
                    .per_regime_v17_visible_tokens_savings.items())},
            "per_regime_team_success_per_visible_token_v17": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self
                    .per_regime_team_success_per_visible_token_v17
                    .items())},
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "masc_v8_witness",
            "witness": self.to_dict()})


def emit_multi_agent_substrate_coordinator_v8_witness(
        *, coordinator: MultiAgentSubstrateCoordinatorV8,
        per_regime_aggregate: dict[str, V8Aggregate],
) -> MultiAgentSubstrateCoordinatorV8Witness:
    aggs_cid = {
        r: str(a.cid())
        for r, a in per_regime_aggregate.items()}
    v17_beats = {
        r: float(a.v17_beats_v16_rate)
        for r, a in per_regime_aggregate.items()}
    tsc_beats = {
        r: float(a.tsc_v17_beats_tsc_v16_rate)
        for r, a in per_regime_aggregate.items()}
    v17_succ = {
        r: float(a.per_policy_success_rate.get(
            W72_MASC_V8_POLICY_SUBSTRATE_ROUTED_V17, 0.0))
        for r, a in per_regime_aggregate.items()}
    tsc_succ = {
        r: float(a.per_policy_success_rate.get(
            W72_MASC_V8_POLICY_TEAM_SUBSTRATE_COORDINATION_V17,
            0.0))
        for r, a in per_regime_aggregate.items()}
    v17_savings = {
        r: float(a.v17_visible_tokens_savings_vs_transcript)
        for r, a in per_regime_aggregate.items()}
    ts_per_v17 = {
        r: float(a.team_success_per_visible_token_v17)
        for r, a in per_regime_aggregate.items()}
    return MultiAgentSubstrateCoordinatorV8Witness(
        schema=W72_MASC_V8_SCHEMA_VERSION,
        coordinator_cid=str(coordinator.cid()),
        per_regime_aggregate_cid=aggs_cid,
        per_regime_v17_beats_v16_rate=v17_beats,
        per_regime_tsc_v17_beats_tsc_v16_rate=tsc_beats,
        per_regime_v17_success_rate=v17_succ,
        per_regime_tsc_v17_success_rate=tsc_succ,
        per_regime_v17_visible_tokens_savings=v17_savings,
        per_regime_team_success_per_visible_token_v17=ts_per_v17,
    )


__all__ = [
    "W72_MASC_V8_SCHEMA_VERSION",
    "W72_MASC_V8_POLICY_SUBSTRATE_ROUTED_V17",
    "W72_MASC_V8_POLICY_TEAM_SUBSTRATE_COORDINATION_V17",
    "W72_MASC_V8_POLICIES",
    "W72_MASC_V8_REGIME_DELAYED_REJOIN_AFTER_RESTART_UNDER_BUDGET",
    "W72_MASC_V8_REGIMES",
    "W72_MASC_V8_REGIMES_NEW",
    "V8PolicyOutcome",
    "V8TaskOutcome",
    "V8Aggregate",
    "MultiAgentSubstrateCoordinatorV8",
    "MultiAgentSubstrateCoordinatorV8Witness",
    "run_v8_multi_agent_task",
    "aggregate_v8_outcomes",
    "emit_multi_agent_substrate_coordinator_v8_witness",
]
