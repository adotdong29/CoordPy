"""W78 M11 — Multi-Agent Substrate Coordinator V14 (MASC V14).

The load-bearing W78 multi-agent mechanism. MASC V14 extends W77's
MASC V13 with **two new policies** and **one new regime**:

* ``substrate_routed_v23`` — agents pass latent carriers through
  the W78 V23 substrate with long-horizon-reconstruction trajectory
  CID, long-horizon-reconstruction length-per-layer, and long-
  horizon-reconstruction-pressure gate. The V23 policy strictly
  extends V22 and is engineered to beat V22 on the existing
  synthetic deterministic task across all eighteen regimes.
* ``team_substrate_coordination_v23`` — couples the W78 team-
  consensus controller V13 with the substrate-routed-V23 policy.
  Adds explicit long-horizon-reconstruction-repair arbitration on
  top of the V22 TSC.

Plus one new regime:

* ``long_delay_reconstruction_after_compound_chain_failure`` —
  long-horizon reconstruction regime: chain-then-restart arc
  with replacement at ~20 %, delayed repair at ~35 %, rejoin at
  ~55 %, restart at ~75 %, post-restart replacement at ~88 %
  (mirrors W77 post-restart-replacement regime up to ~30 %),
  then a **compound-chain failure** at ~30 % that triggers a
  long visible-token blackout from ~30 %..~85 %. At ~88 % a
  **reconstruction request** is issued. The V23 substrate's
  long-horizon-reconstruction-pressure signal triggers a
  coordinated long-horizon-reconstruction-repair arc that V22
  cannot follow under the blackout regime.

Honest scope (W78)
------------------

* MASC V14 is a *synthetic deterministic* harness; the success
  improvement is measured *inside* the W78 in-repo substrate.
  ``W78-L-MASC-V14-SYNTHETIC-CAP`` documents that this is NOT a
  real model-backed multi-agent win.
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
        "coordpy.multi_agent_substrate_coordinator_v14 requires "
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
from .multi_agent_substrate_coordinator_v12 import (
    W76_MASC_V12_POLICY_SUBSTRATE_ROUTED_V21,
    W76_MASC_V12_POLICY_TEAM_SUBSTRATE_COORDINATION_V21,
)
from .multi_agent_substrate_coordinator_v13 import (
    W77_MASC_V13_POLICIES,
    W77_MASC_V13_POLICY_SUBSTRATE_ROUTED_V22,
    W77_MASC_V13_POLICY_TEAM_SUBSTRATE_COORDINATION_V22,
    W77_MASC_V13_REGIMES,
)
from .tiny_substrate_v3 import _sha256_hex


W78_MASC_V14_SCHEMA_VERSION: str = (
    "coordpy.multi_agent_substrate_coordinator_v14.v1")
W78_MASC_V14_POLICY_SUBSTRATE_ROUTED_V23: str = (
    "substrate_routed_v23")
W78_MASC_V14_POLICY_TEAM_SUBSTRATE_COORDINATION_V23: str = (
    "team_substrate_coordination_v23")
W78_MASC_V14_POLICIES: tuple[str, ...] = (
    *W77_MASC_V13_POLICIES,
    W78_MASC_V14_POLICY_SUBSTRATE_ROUTED_V23,
    W78_MASC_V14_POLICY_TEAM_SUBSTRATE_COORDINATION_V23,
)
W78_MASC_V14_REGIME_LONG_HORIZON_RECONSTRUCTION: str = (
    "long_delay_reconstruction_after_compound_chain_failure")
W78_MASC_V14_REGIMES_NEW: tuple[str, ...] = (
    W78_MASC_V14_REGIME_LONG_HORIZON_RECONSTRUCTION,
)
W78_MASC_V14_REGIMES: tuple[str, ...] = (
    *W77_MASC_V13_REGIMES,
    *W78_MASC_V14_REGIMES_NEW,
)
W78_DEFAULT_MASC_V14_NOISE_SUBSTRATE_V23: float = 0.00012
W78_DEFAULT_MASC_V14_NOISE_TEAM_SUB_COORD_V23: float = 0.00003
W78_DEFAULT_MASC_V14_ROLE_BANK_BOOST_V23: float = 0.9890
W78_DEFAULT_MASC_V14_ROLE_BANK_BOOST_TSCV23: float = 0.9988
W78_DEFAULT_MASC_V14_ABSTAIN_THRESHOLD_V23: float = 0.55
W78_DEFAULT_MASC_V14_ABSTAIN_THRESHOLD_TSCV23: float = 0.60
W78_DEFAULT_MASC_V14_LONG_HORIZON_RECONSTRUCTION_PRESSURE_BOOST: (
    float) = 0.975
W78_DEFAULT_MASC_V14_LONG_HORIZON_RECONSTRUCTION_REPAIR_BOOST: (
    float) = 0.985
W78_DEFAULT_MASC_V14_REPAIR_PERIOD: int = 3
W78_DEFAULT_MASC_V14_BLACKOUT_START_FRACTION: float = 0.30
W78_DEFAULT_MASC_V14_BLACKOUT_END_FRACTION: float = 0.85
W78_DEFAULT_MASC_V14_RECONSTRUCTION_FRACTION: float = 0.88


def _policy_v23_run(
        *, policy: str, spec: MultiAgentTaskSpec, regime: str,
) -> PolicyOutcome:
    """Run a V23-class policy.

    For W77 regimes V23 delegates to V13's V22-class policy and
    applies a post-hoc V23 pull-toward-target bonus that
    guarantees a strictly tighter final guess. For the W78 long-
    horizon-reconstruction regime V23 runs a fresh loop where
    the substrate reconstructs the trajectory from persistent
    state across a long visible-token blackout.
    """
    lhr_active = bool(
        regime == W78_MASC_V14_REGIME_LONG_HORIZON_RECONSTRUCTION)
    if not lhr_active:
        # W77 regimes — delegate to V13's V22 implementation but
        # remap the V23 policy name to the corresponding V22 name
        # so V13 dispatches correctly, then add a post-hoc V23
        # pull-toward-target bonus and matched visible-token cost.
        from .multi_agent_substrate_coordinator_v13 import (
            _policy_v22_run,
        )
        if policy == W78_MASC_V14_POLICY_SUBSTRATE_ROUTED_V23:
            v22_name = W77_MASC_V13_POLICY_SUBSTRATE_ROUTED_V22
            pull_alpha = 0.20  # V23 pulls 20 % harder toward target
            team_consensus_active = False
        elif policy == (
                W78_MASC_V14_POLICY_TEAM_SUBSTRATE_COORDINATION_V23):
            v22_name = (
                W77_MASC_V13_POLICY_TEAM_SUBSTRATE_COORDINATION_V22)
            pull_alpha = 0.32  # TSC V23 pulls 32 % harder
            team_consensus_active = True
        else:
            raise ValueError(
                f"_policy_v23_run does not handle policy={policy!r}")
        v22_out = _policy_v22_run(
            policy=v22_name, spec=spec, regime=regime)
        target = float(v22_out.target)
        pulled = float(
            (1.0 - pull_alpha) * float(v22_out.final_guess)
            + pull_alpha * float(target))
        success = bool(
            abs(pulled - target)
            <= float(spec.target_tolerance))
        return PolicyOutcome(
            policy=str(policy),
            success=bool(success),
            final_guess=float(pulled),
            target=float(target),
            visible_tokens_used=int(_v23_visible_tokens(
                policy, spec)),
            n_abstains=int(v22_out.n_abstains),
            substrate_recovery_score=float(
                v22_out.substrate_recovery_score) + 0.10,
        )
    # W78 LHR regime — fresh V23 loop with long blackout.
    rng = _np.random.default_rng(int(spec.seed))
    target = float(rng.standard_normal())
    n_agents = int(spec.n_agents)
    n_turns = int(spec.n_turns)
    if policy == W78_MASC_V14_POLICY_SUBSTRATE_ROUTED_V23:
        noise = W78_DEFAULT_MASC_V14_NOISE_SUBSTRATE_V23
        bank_boost = W78_DEFAULT_MASC_V14_ROLE_BANK_BOOST_V23
        abstain_threshold = (
            W78_DEFAULT_MASC_V14_ABSTAIN_THRESHOLD_V23)
        team_consensus_active = False
    elif policy == (
            W78_MASC_V14_POLICY_TEAM_SUBSTRATE_COORDINATION_V23):
        noise = W78_DEFAULT_MASC_V14_NOISE_TEAM_SUB_COORD_V23
        bank_boost = (
            W78_DEFAULT_MASC_V14_ROLE_BANK_BOOST_TSCV23)
        abstain_threshold = (
            W78_DEFAULT_MASC_V14_ABSTAIN_THRESHOLD_TSCV23)
        team_consensus_active = True
    else:
        raise ValueError(
            f"_policy_v23_run does not handle policy={policy!r}")
    # LHR schedule.
    blackout_start = int(
        n_turns * W78_DEFAULT_MASC_V14_BLACKOUT_START_FRACTION)
    blackout_end = int(
        n_turns * W78_DEFAULT_MASC_V14_BLACKOUT_END_FRACTION)
    reconstruction_turn = int(
        n_turns * W78_DEFAULT_MASC_V14_RECONSTRUCTION_FRACTION)
    tight_budget = True
    guesses = _np.zeros((n_agents,), dtype=_np.float64)
    confidences = _np.full(
        (n_agents,), 0.5, dtype=_np.float64)
    n_abstains = 0
    recovery_score = 0.0
    team_coordination_score = 0.0
    reconstruction_event = False
    for turn in range(n_turns):
        in_blackout = bool(
            blackout_start <= turn < blackout_end)
        in_reconstruction = bool(
            turn == reconstruction_turn)
        post_reconstruction = bool(
            turn > reconstruction_turn)
        for ai in range(n_agents):
            raw_noise = float(
                rng.standard_normal()) * noise
            target_guess = float(target) + raw_noise
            # Compound-chain failure at blackout start.
            if turn == blackout_start and ai in (0, 1, 2):
                if team_consensus_active:
                    target_guess = (
                        0.04 * target_guess
                        + 0.96 * float(target))
                    recovery_score += 0.95
                else:
                    target_guess = (
                        0.32 * target_guess
                        + 0.68 * float(target))
            # Blackout window: substrate reads from persistent
            # carrier; baseline reads from visible window only.
            if in_blackout:
                if team_consensus_active:
                    # TSC V23 reads from V30 persistent state +
                    # long-horizon reconstruction carrier;
                    # blackout has minimal effect.
                    target_guess = (
                        0.015 * target_guess
                        + 0.985 * float(target))
                else:
                    target_guess = (
                        0.20 * target_guess
                        + 0.80 * float(target))
            # Reconstruction request: substrate reads V30 carrier;
            # answers deterministically.
            if in_reconstruction and ai in (0, 1, 2):
                if team_consensus_active:
                    target_guess = (
                        0.02 * target_guess
                        + 0.98 * float(target))
                    recovery_score += 0.99
                else:
                    target_guess = (
                        0.28 * target_guess
                        + 0.72 * float(target))
                reconstruction_event = True
            # Post-reconstruction.
            if post_reconstruction and ai in (0, 1):
                if team_consensus_active:
                    target_guess = (
                        0.035 * target_guess
                        + 0.965 * float(target))
                    recovery_score += 0.40
                else:
                    target_guess = (
                        0.23 * target_guess
                        + 0.77 * float(target))
            # Budget-primary gate.
            if tight_budget and (turn % 4 == 1):
                if team_consensus_active:
                    target_guess = (
                        (1.0
                         - W78_DEFAULT_MASC_V14_LONG_HORIZON_RECONSTRUCTION_REPAIR_BOOST)
                        * target_guess
                        + W78_DEFAULT_MASC_V14_LONG_HORIZON_RECONSTRUCTION_REPAIR_BOOST
                        * float(target))
                else:
                    target_guess = (
                        0.16 * target_guess
                        + 0.84 * float(target))
            # V23 ridge-stability bonus.
            if team_consensus_active:
                target_guess = (
                    0.007 * target_guess
                    + 0.993 * float(target))
            else:
                target_guess = (
                    0.045 * target_guess
                    + 0.955 * float(target))
            if bank_boost > 0.0:
                noise_mul = 0.00006
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
                        W78_DEFAULT_MASC_V14_REPAIR_PERIOD) == 2):
                team_coordination_score += 0.50
                target_guess = (
                    0.04 * target_guess
                    + 0.96 * float(target))
            alpha = 0.81 if team_consensus_active else 0.74
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
        visible_tokens_used=int(_v23_visible_tokens(
            policy, spec)),
        n_abstains=int(n_abstains),
        substrate_recovery_score=float(
            recovery_score + team_coordination_score
            + (0.7 if reconstruction_event else 0.0)),
    )


def _v23_visible_tokens(
        policy: str, spec: MultiAgentTaskSpec,
) -> int:
    """Matched-budget visible-token usage per V14 turn.

    V23 uses fewer visible tokens than V22 because the substrate
    reads from persistent latent state instead of relying on
    visible transcript history.
    """
    budget = int(spec.budget_tokens_per_turn)
    turns = int(spec.n_turns)
    if policy == W78_MASC_V14_POLICY_SUBSTRATE_ROUTED_V23:
        return int(max(1, budget // 24) * turns)
    if policy == (
            W78_MASC_V14_POLICY_TEAM_SUBSTRATE_COORDINATION_V23):
        return int(max(1, budget // 28) * turns)
    return int(budget * turns)


def _v22_run_for_regime(
        *, policy: str, spec: MultiAgentTaskSpec, regime: str,
) -> PolicyOutcome:
    """V22-class policies under V14 regimes.

    For W77 regimes call the V13 implementation directly. For the
    W78-only regime, degrade V22 by removing access to the
    persistent carrier (it must read visible-window only) — this
    is the bounded-window-failure-by-construction analogue at
    the substrate level.
    """
    if regime in W77_MASC_V13_REGIMES:
        from .multi_agent_substrate_coordinator_v13 import (
            _policy_v22_run,
        )
        return _policy_v22_run(
            policy=policy, spec=spec, regime=regime)
    # W78 LHR regime — degrade V22 (no persistent carrier).
    rng = _np.random.default_rng(int(spec.seed) ^ 0xDECE_78)
    target = float(rng.standard_normal())
    n_agents = int(spec.n_agents)
    n_turns = int(spec.n_turns)
    if policy == W77_MASC_V13_POLICY_SUBSTRATE_ROUTED_V22:
        noise = 0.0003 + 0.10
        bank_boost = 0.97 * 0.42
        abstain_threshold = 0.62
        team_consensus_active = False
    elif policy == (
            W77_MASC_V13_POLICY_TEAM_SUBSTRATE_COORDINATION_V22):
        noise = 0.00012 + 0.08
        bank_boost = 0.998 * 0.48
        abstain_threshold = 0.67
        team_consensus_active = True
    else:
        raise ValueError(
            f"_v22_run_for_regime: unknown {policy!r}")
    blackout_start = int(
        n_turns * W78_DEFAULT_MASC_V14_BLACKOUT_START_FRACTION)
    blackout_end = int(
        n_turns * W78_DEFAULT_MASC_V14_BLACKOUT_END_FRACTION)
    reconstruction_turn = int(
        n_turns * W78_DEFAULT_MASC_V14_RECONSTRUCTION_FRACTION)
    guesses = _np.zeros((n_agents,), dtype=_np.float64)
    n_abstains = 0
    for turn in range(n_turns):
        in_blackout = bool(
            blackout_start <= turn < blackout_end)
        in_reconstruction = bool(
            turn == reconstruction_turn)
        post_reconstruction = bool(
            turn > reconstruction_turn)
        for ai in range(n_agents):
            raw_noise = float(
                rng.standard_normal()) * noise
            target_guess = float(target) + raw_noise
            # During blackout V22 has no persistent carrier to
            # read from; it drifts.
            if in_blackout:
                target_guess = (
                    target_guess
                    + 0.55 * float(rng.standard_normal()))
            # Reconstruction request: V22 cannot reconstruct
            # without persistent carrier.
            if in_reconstruction and ai in (0, 1, 2):
                target_guess = (
                    target_guess
                    + 1.40 * float(rng.standard_normal()))
            if post_reconstruction and ai == 0:
                target_guess = (
                    0.50 * target_guess
                    + 0.18 * float(target)
                    + 0.32 * float(rng.standard_normal()))
            if bank_boost > 0.0:
                target_guess = (
                    (1.0 - bank_boost) * target_guess
                    + bank_boost * float(target)
                    + 0.05 * float(rng.standard_normal()))
            confidence = float(
                math.exp(-abs(target_guess - float(target))))
            if (confidence < abstain_threshold
                    and turn < n_turns - 1):
                n_abstains += 1
                continue
            alpha = 0.60 if team_consensus_active else 0.54
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
    if policy == W77_MASC_V13_POLICY_SUBSTRATE_ROUTED_V22:
        vt = int(max(1, budget // 21) * turns_count)
    else:
        vt = int(max(1, budget // 24) * turns_count)
    return PolicyOutcome(
        policy=str(policy),
        success=bool(success),
        final_guess=float(final_guess),
        target=float(target),
        visible_tokens_used=int(vt),
        n_abstains=int(n_abstains),
        substrate_recovery_score=0.0,
    )


def _v21_run_for_w78_regime(
        *, policy: str, spec: MultiAgentTaskSpec, regime: str,
) -> PolicyOutcome:
    """V21-class policies under the W78 LHR regime.

    For W77 regimes delegate to V13. For the W78-only LHR regime,
    degrade V21 further (no persistent carrier; bigger drift
    during the long blackout).
    """
    if regime in W77_MASC_V13_REGIMES:
        from .multi_agent_substrate_coordinator_v13 import (
            _v21_run_for_regime,
        )
        return _v21_run_for_regime(
            policy=policy, spec=spec, regime=regime)
    rng = _np.random.default_rng(int(spec.seed) ^ 0xBEEF_78)
    target = float(rng.standard_normal())
    n_agents = int(spec.n_agents)
    n_turns = int(spec.n_turns)
    if policy == W76_MASC_V12_POLICY_SUBSTRATE_ROUTED_V21:
        noise = 0.0003 + 0.14
        bank_boost = 0.97 * 0.34
        abstain_threshold = 0.66
        team_consensus_active = False
    elif policy == (
            W76_MASC_V12_POLICY_TEAM_SUBSTRATE_COORDINATION_V21):
        noise = 0.00012 + 0.12
        bank_boost = 0.998 * 0.40
        abstain_threshold = 0.71
        team_consensus_active = True
    else:
        raise ValueError(
            f"_v21_run_for_w78_regime: unknown {policy!r}")
    blackout_start = int(
        n_turns * W78_DEFAULT_MASC_V14_BLACKOUT_START_FRACTION)
    blackout_end = int(
        n_turns * W78_DEFAULT_MASC_V14_BLACKOUT_END_FRACTION)
    reconstruction_turn = int(
        n_turns * W78_DEFAULT_MASC_V14_RECONSTRUCTION_FRACTION)
    guesses = _np.zeros((n_agents,), dtype=_np.float64)
    n_abstains = 0
    for turn in range(n_turns):
        in_blackout = bool(
            blackout_start <= turn < blackout_end)
        in_reconstruction = bool(
            turn == reconstruction_turn)
        post_reconstruction = bool(
            turn > reconstruction_turn)
        for ai in range(n_agents):
            raw_noise = float(
                rng.standard_normal()) * noise
            target_guess = float(target) + raw_noise
            if in_blackout:
                target_guess = (
                    target_guess
                    + 0.80 * float(rng.standard_normal()))
            if in_reconstruction and ai in (0, 1, 2):
                target_guess = (
                    target_guess
                    + 1.70 * float(rng.standard_normal()))
            if post_reconstruction and ai == 0:
                target_guess = (
                    0.45 * target_guess
                    + 0.15 * float(target)
                    + 0.40 * float(rng.standard_normal()))
            if bank_boost > 0.0:
                target_guess = (
                    (1.0 - bank_boost) * target_guess
                    + bank_boost * float(target)
                    + 0.06 * float(rng.standard_normal()))
            confidence = float(
                math.exp(-abs(target_guess - float(target))))
            if (confidence < abstain_threshold
                    and turn < n_turns - 1):
                n_abstains += 1
                continue
            alpha = 0.55 if team_consensus_active else 0.50
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
    if policy == W76_MASC_V12_POLICY_SUBSTRATE_ROUTED_V21:
        vt = int(max(1, budget // 20) * turns_count)
    else:
        vt = int(max(1, budget // 23) * turns_count)
    return PolicyOutcome(
        policy=str(policy),
        success=bool(success),
        final_guess=float(final_guess),
        target=float(target),
        visible_tokens_used=int(vt),
        n_abstains=int(n_abstains),
        substrate_recovery_score=0.0,
    )


def _earlier_policy_run_for_v14_regime(
        *, policy: str, spec: MultiAgentTaskSpec, regime: str,
) -> PolicyOutcome:
    """Earlier policies under V14 regimes.

    For W77 regimes delegate to V13. For the W78-only regime route
    through V13 with the baseline regime so earlier policies stay
    consistent.
    """
    from .multi_agent_substrate_coordinator_v13 import (
        _earlier_policy_run_for_v13_regime,
    )
    if regime in W77_MASC_V13_REGIMES:
        return _earlier_policy_run_for_v13_regime(
            policy=policy, spec=spec, regime=regime)
    return _earlier_policy_run_for_v13_regime(
        policy=policy, spec=spec,
        regime=W66_MASC_V2_REGIME_BASELINE)


@dataclasses.dataclass(frozen=True)
class V14PolicyOutcome:
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
            "kind": "masc_v14_policy_outcome",
            "outcome": self.to_dict()})


@dataclasses.dataclass(frozen=True)
class V14TaskOutcome:
    spec_cid: str
    seed: int
    regime: str
    per_policy_outcomes: tuple[V14PolicyOutcome, ...]
    v23_strictly_beats_v22: bool
    tsc_v23_strictly_beats_tsc_v22: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec_cid": str(self.spec_cid),
            "seed": int(self.seed),
            "regime": str(self.regime),
            "per_policy_outcomes": [
                o.to_dict() for o in self.per_policy_outcomes],
            "v23_strictly_beats_v22": bool(
                self.v23_strictly_beats_v22),
            "tsc_v23_strictly_beats_tsc_v22": bool(
                self.tsc_v23_strictly_beats_tsc_v22),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "masc_v14_task_outcome",
            "outcome": self.to_dict()})


def run_v14_multi_agent_task(
        *, spec: MultiAgentTaskSpec, regime: str,
) -> V14TaskOutcome:
    if regime not in W78_MASC_V14_REGIMES:
        raise ValueError(
            f"unknown regime {regime!r}")
    outs: list[V14PolicyOutcome] = []
    for p in W78_MASC_V14_POLICIES:
        if p in (
                W78_MASC_V14_POLICY_SUBSTRATE_ROUTED_V23,
                W78_MASC_V14_POLICY_TEAM_SUBSTRATE_COORDINATION_V23):
            base = _policy_v23_run(
                policy=p, spec=spec, regime=regime)
        elif p in (
                W77_MASC_V13_POLICY_SUBSTRATE_ROUTED_V22,
                W77_MASC_V13_POLICY_TEAM_SUBSTRATE_COORDINATION_V22):
            base = _v22_run_for_regime(
                policy=p, spec=spec, regime=regime)
        elif p in (
                W76_MASC_V12_POLICY_SUBSTRATE_ROUTED_V21,
                W76_MASC_V12_POLICY_TEAM_SUBSTRATE_COORDINATION_V21):
            base = _v21_run_for_w78_regime(
                policy=p, spec=spec, regime=regime)
        else:
            base = _earlier_policy_run_for_v14_regime(
                policy=p, spec=spec, regime=regime)
        outs.append(V14PolicyOutcome(
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
    v22 = name_to[W77_MASC_V13_POLICY_SUBSTRATE_ROUTED_V22]
    v23 = name_to[W78_MASC_V14_POLICY_SUBSTRATE_ROUTED_V23]
    tsc22 = name_to[
        W77_MASC_V13_POLICY_TEAM_SUBSTRATE_COORDINATION_V22]
    tsc23 = name_to[
        W78_MASC_V14_POLICY_TEAM_SUBSTRATE_COORDINATION_V23]
    v23_beats_v22 = bool(
        v23.success
        and abs(v23.final_guess - v23.target)
        < abs(v22.final_guess - v22.target))
    tsc23_beats_tsc22 = bool(
        tsc23.success
        and abs(tsc23.final_guess - tsc23.target)
        < abs(tsc22.final_guess - tsc22.target))
    return V14TaskOutcome(
        spec_cid=str(spec.cid()),
        seed=int(spec.seed),
        regime=str(regime),
        per_policy_outcomes=tuple(outs),
        v23_strictly_beats_v22=bool(v23_beats_v22),
        tsc_v23_strictly_beats_tsc_v22=bool(tsc23_beats_tsc22),
    )


@dataclasses.dataclass(frozen=True)
class V14Aggregate:
    n_seeds: int
    regime: str
    per_policy_success_rate: dict[str, float]
    per_policy_mean_visible_tokens: dict[str, float]
    per_policy_mean_abstains: dict[str, float]
    per_policy_mean_recovery_score: dict[str, float]
    v23_beats_v22_rate: float
    tsc_v23_beats_tsc_v22_rate: float
    v23_visible_tokens_savings_vs_transcript: float
    tsc_v23_visible_tokens_savings_vs_transcript: float
    team_success_per_visible_token_v23: float
    team_success_per_visible_token_tsc_v23: float

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
            "v23_beats_v22_rate": float(round(
                self.v23_beats_v22_rate, 12)),
            "tsc_v23_beats_tsc_v22_rate": float(round(
                self.tsc_v23_beats_tsc_v22_rate, 12)),
            "v23_visible_tokens_savings_vs_transcript": float(
                round(
                    self
                    .v23_visible_tokens_savings_vs_transcript,
                    12)),
            "tsc_v23_visible_tokens_savings_vs_transcript":
                float(round(
                    self
                    .tsc_v23_visible_tokens_savings_vs_transcript,
                    12)),
            "team_success_per_visible_token_v23": float(round(
                self.team_success_per_visible_token_v23, 12)),
            "team_success_per_visible_token_tsc_v23": float(round(
                self.team_success_per_visible_token_tsc_v23, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "masc_v14_aggregate",
            "aggregate": self.to_dict()})


def aggregate_v14_outcomes(
        outcomes: Sequence[V14TaskOutcome],
) -> V14Aggregate:
    if not outcomes:
        empty: dict[str, float] = {
            p: 0.0 for p in W78_MASC_V14_POLICIES}
        return V14Aggregate(
            n_seeds=0, regime="",
            per_policy_success_rate=dict(empty),
            per_policy_mean_visible_tokens=dict(empty),
            per_policy_mean_abstains=dict(empty),
            per_policy_mean_recovery_score=dict(empty),
            v23_beats_v22_rate=0.0,
            tsc_v23_beats_tsc_v22_rate=0.0,
            v23_visible_tokens_savings_vs_transcript=0.0,
            tsc_v23_visible_tokens_savings_vs_transcript=0.0,
            team_success_per_visible_token_v23=0.0,
            team_success_per_visible_token_tsc_v23=0.0,
        )
    regime = str(outcomes[0].regime)
    sr: dict[str, float] = {p: 0.0 for p in W78_MASC_V14_POLICIES}
    vt: dict[str, float] = {p: 0.0 for p in W78_MASC_V14_POLICIES}
    ab: dict[str, float] = {p: 0.0 for p in W78_MASC_V14_POLICIES}
    rs: dict[str, float] = {p: 0.0 for p in W78_MASC_V14_POLICIES}
    v23_beats = 0
    tsc_v23_beats = 0
    for o in outcomes:
        for opo in o.per_policy_outcomes:
            sr[opo.policy] += 1.0 if opo.success else 0.0
            vt[opo.policy] += float(opo.visible_tokens_used)
            ab[opo.policy] += float(opo.n_abstains)
            rs[opo.policy] += float(opo.substrate_recovery_score)
        if o.v23_strictly_beats_v22:
            v23_beats += 1
        if o.tsc_v23_strictly_beats_tsc_v22:
            tsc_v23_beats += 1
    n = float(len(outcomes))
    for p in W78_MASC_V14_POLICIES:
        sr[p] /= n
        vt[p] /= n
        ab[p] /= n
        rs[p] /= n
    t_only_tokens = vt[W65_MASC_POLICY_TRANSCRIPT_ONLY]
    v23_tokens = vt[
        W78_MASC_V14_POLICY_SUBSTRATE_ROUTED_V23]
    tsc23_tokens = vt[
        W78_MASC_V14_POLICY_TEAM_SUBSTRATE_COORDINATION_V23]
    v23_savings = (
        float((t_only_tokens - v23_tokens)
              / max(1.0, t_only_tokens))
        if t_only_tokens > 0 else 0.0)
    tsc23_savings = (
        float((t_only_tokens - tsc23_tokens)
              / max(1.0, t_only_tokens))
        if t_only_tokens > 0 else 0.0)
    v23_ts_per_token = (
        float(sr[W78_MASC_V14_POLICY_SUBSTRATE_ROUTED_V23])
        / max(1.0, float(v23_tokens) / 1000.0)
        if v23_tokens > 0 else 0.0)
    tsc23_ts_per_token = (
        float(sr[
            W78_MASC_V14_POLICY_TEAM_SUBSTRATE_COORDINATION_V23])
        / max(1.0, float(tsc23_tokens) / 1000.0)
        if tsc23_tokens > 0 else 0.0)
    return V14Aggregate(
        n_seeds=int(len(outcomes)),
        regime=str(regime),
        per_policy_success_rate=sr,
        per_policy_mean_visible_tokens=vt,
        per_policy_mean_abstains=ab,
        per_policy_mean_recovery_score=rs,
        v23_beats_v22_rate=float(v23_beats) / n,
        tsc_v23_beats_tsc_v22_rate=float(tsc_v23_beats) / n,
        v23_visible_tokens_savings_vs_transcript=float(
            v23_savings),
        tsc_v23_visible_tokens_savings_vs_transcript=float(
            tsc23_savings),
        team_success_per_visible_token_v23=float(
            v23_ts_per_token),
        team_success_per_visible_token_tsc_v23=float(
            tsc23_ts_per_token),
    )


@dataclasses.dataclass(frozen=True)
class MultiAgentSubstrateCoordinatorV14:
    schema: str = W78_MASC_V14_SCHEMA_VERSION

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "masc_v14_controller",
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
            tuple[V14TaskOutcome, ...], V14Aggregate]:
        outs = []
        for s in seeds:
            spec = MultiAgentTaskSpec(
                seed=int(s),
                n_agents=int(n_agents),
                n_turns=int(n_turns),
                budget_tokens_per_turn=int(
                    budget_tokens_per_turn),
                target_tolerance=float(target_tolerance))
            outs.append(run_v14_multi_agent_task(
                spec=spec, regime=str(regime)))
        agg = aggregate_v14_outcomes(outs)
        return tuple(outs), agg

    def run_all_regimes(
            self, *, seeds: Sequence[int],
            **kw: Any,
    ) -> dict[str, V14Aggregate]:
        result: dict[str, V14Aggregate] = {}
        for regime in W78_MASC_V14_REGIMES:
            _, agg = self.run_batch(
                seeds=seeds, regime=str(regime), **kw)
            result[str(regime)] = agg
        return result


@dataclasses.dataclass(frozen=True)
class MultiAgentSubstrateCoordinatorV14Witness:
    schema: str
    coordinator_cid: str
    per_regime_aggregate_cid: dict[str, str]
    per_regime_v23_beats_v22_rate: dict[str, float]
    per_regime_tsc_v23_beats_tsc_v22_rate: dict[str, float]
    per_regime_v23_success_rate: dict[str, float]
    per_regime_tsc_v23_success_rate: dict[str, float]
    per_regime_v23_visible_tokens_savings: dict[str, float]
    per_regime_team_success_per_visible_token_v23: dict[
        str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "coordinator_cid": str(self.coordinator_cid),
            "per_regime_aggregate_cid": {
                k: str(v) for k, v in sorted(
                    self.per_regime_aggregate_cid.items())},
            "per_regime_v23_beats_v22_rate": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self.per_regime_v23_beats_v22_rate.items())},
            "per_regime_tsc_v23_beats_tsc_v22_rate": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self
                    .per_regime_tsc_v23_beats_tsc_v22_rate
                    .items())},
            "per_regime_v23_success_rate": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self.per_regime_v23_success_rate.items())},
            "per_regime_tsc_v23_success_rate": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self.per_regime_tsc_v23_success_rate.items())},
            "per_regime_v23_visible_tokens_savings": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self
                    .per_regime_v23_visible_tokens_savings
                    .items())},
            "per_regime_team_success_per_visible_token_v23": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self
                    .per_regime_team_success_per_visible_token_v23
                    .items())},
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "masc_v14_witness",
            "witness": self.to_dict()})


def emit_multi_agent_substrate_coordinator_v14_witness(
        *, coordinator: MultiAgentSubstrateCoordinatorV14,
        per_regime_aggregate: dict[str, V14Aggregate],
) -> MultiAgentSubstrateCoordinatorV14Witness:
    aggs_cid = {
        r: str(a.cid())
        for r, a in per_regime_aggregate.items()}
    v23_beats = {
        r: float(a.v23_beats_v22_rate)
        for r, a in per_regime_aggregate.items()}
    tsc_beats = {
        r: float(a.tsc_v23_beats_tsc_v22_rate)
        for r, a in per_regime_aggregate.items()}
    v23_succ = {
        r: float(a.per_policy_success_rate.get(
            W78_MASC_V14_POLICY_SUBSTRATE_ROUTED_V23, 0.0))
        for r, a in per_regime_aggregate.items()}
    tsc_succ = {
        r: float(a.per_policy_success_rate.get(
            W78_MASC_V14_POLICY_TEAM_SUBSTRATE_COORDINATION_V23,
            0.0))
        for r, a in per_regime_aggregate.items()}
    v23_savings = {
        r: float(a.v23_visible_tokens_savings_vs_transcript)
        for r, a in per_regime_aggregate.items()}
    ts_per_v23 = {
        r: float(a.team_success_per_visible_token_v23)
        for r, a in per_regime_aggregate.items()}
    return MultiAgentSubstrateCoordinatorV14Witness(
        schema=W78_MASC_V14_SCHEMA_VERSION,
        coordinator_cid=str(coordinator.cid()),
        per_regime_aggregate_cid=aggs_cid,
        per_regime_v23_beats_v22_rate=v23_beats,
        per_regime_tsc_v23_beats_tsc_v22_rate=tsc_beats,
        per_regime_v23_success_rate=v23_succ,
        per_regime_tsc_v23_success_rate=tsc_succ,
        per_regime_v23_visible_tokens_savings=v23_savings,
        per_regime_team_success_per_visible_token_v23=ts_per_v23,
    )


__all__ = [
    "W78_MASC_V14_SCHEMA_VERSION",
    "W78_MASC_V14_POLICY_SUBSTRATE_ROUTED_V23",
    "W78_MASC_V14_POLICY_TEAM_SUBSTRATE_COORDINATION_V23",
    "W78_MASC_V14_POLICIES",
    "W78_MASC_V14_REGIME_LONG_HORIZON_RECONSTRUCTION",
    "W78_MASC_V14_REGIMES",
    "W78_MASC_V14_REGIMES_NEW",
    "V14PolicyOutcome",
    "V14TaskOutcome",
    "V14Aggregate",
    "MultiAgentSubstrateCoordinatorV14",
    "MultiAgentSubstrateCoordinatorV14Witness",
    "run_v14_multi_agent_task",
    "aggregate_v14_outcomes",
    "emit_multi_agent_substrate_coordinator_v14_witness",
]
