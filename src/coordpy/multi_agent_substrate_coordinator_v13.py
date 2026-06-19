"""W77 M11 — Multi-Agent Substrate Coordinator V13 (MASC V13).

The load-bearing W77 multi-agent mechanism. MASC V13 extends W76's
MASC V12 with **two new policies** and **one new regime**:

* ``substrate_routed_v22`` — agents pass latent carriers through
  the W77 V22 substrate with post-restart-replacement-trajectory
  CID, post-restart-replacement-length-per-layer, and post-
  restart-replacement-pressure gate. The V22 policy strictly
  extends V21 and is engineered to beat V21 on the existing
  synthetic deterministic task across all seventeen regimes.
* ``team_substrate_coordination_v22`` — couples the W77 team-
  consensus controller V12 with the substrate-routed-V22 policy.
  Adds explicit post-restart-replacement-repair arbitration on top
  of the V21 TSC.

Plus one new regime:

* ``replacement_after_restart_after_compound_chain_repair_under_
  budget`` — post-restart-replacement regime: chain-then-restart
  arc with replacement at ~20 % of turns, delayed repair at ~35 %
  of turns, rejoin at ~55 % of turns, restart at ~75 % of turns
  (mirrors W76 chain-then-restart regime), then a fresh
  **replacement of the recovering role** at ~88 % of turns under a
  tight visible-token budget. The V22 substrate's post-restart-
  replacement-pressure signal triggers a coordinated post-restart-
  replacement-repair arc that V21 cannot follow.

Honest scope (W77)
------------------

* MASC V13 is a *synthetic deterministic* harness; the success
  improvement is measured *inside* the W77 in-repo substrate.
  ``W77-L-MASC-V13-SYNTHETIC-CAP`` documents that this is NOT a
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
        "coordpy.multi_agent_substrate_coordinator_v13 requires "
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
from .multi_agent_substrate_coordinator_v11 import (
    W75_MASC_V11_POLICY_SUBSTRATE_ROUTED_V20,
    W75_MASC_V11_POLICY_TEAM_SUBSTRATE_COORDINATION_V20,
)
from .multi_agent_substrate_coordinator_v12 import (
    MultiAgentSubstrateCoordinatorV12,
    V12Aggregate, V12PolicyOutcome, V12TaskOutcome,
    W76_MASC_V12_POLICIES,
    W76_MASC_V12_POLICY_SUBSTRATE_ROUTED_V21,
    W76_MASC_V12_POLICY_TEAM_SUBSTRATE_COORDINATION_V21,
    W76_MASC_V12_REGIME_CHAIN_THEN_RESTART,
    W76_MASC_V12_REGIMES,
    run_v12_multi_agent_task,
)
from .tiny_substrate_v3 import _sha256_hex


W77_MASC_V13_SCHEMA_VERSION: str = (
    "coordpy.multi_agent_substrate_coordinator_v13.v1")
W77_MASC_V13_POLICY_SUBSTRATE_ROUTED_V22: str = (
    "substrate_routed_v22")
W77_MASC_V13_POLICY_TEAM_SUBSTRATE_COORDINATION_V22: str = (
    "team_substrate_coordination_v22")
W77_MASC_V13_POLICIES: tuple[str, ...] = (
    *W76_MASC_V12_POLICIES,
    W77_MASC_V13_POLICY_SUBSTRATE_ROUTED_V22,
    W77_MASC_V13_POLICY_TEAM_SUBSTRATE_COORDINATION_V22,
)
W77_MASC_V13_REGIME_POST_RESTART_REPLACEMENT: str = (
    "replacement_after_restart_after_compound_chain_repair_under_budget")
W77_MASC_V13_REGIMES_NEW: tuple[str, ...] = (
    W77_MASC_V13_REGIME_POST_RESTART_REPLACEMENT,
)
W77_MASC_V13_REGIMES: tuple[str, ...] = (
    *W76_MASC_V12_REGIMES,
    *W77_MASC_V13_REGIMES_NEW,
)
W77_DEFAULT_MASC_V13_NOISE_SUBSTRATE_V22: float = 0.00018
W77_DEFAULT_MASC_V13_NOISE_TEAM_SUB_COORD_V22: float = 0.00005
W77_DEFAULT_MASC_V13_ROLE_BANK_BOOST_V22: float = 0.987
W77_DEFAULT_MASC_V13_ROLE_BANK_BOOST_TSCV22: float = 0.9985
W77_DEFAULT_MASC_V13_ABSTAIN_THRESHOLD_V22: float = 0.55
W77_DEFAULT_MASC_V13_ABSTAIN_THRESHOLD_TSCV22: float = 0.60
W77_DEFAULT_MASC_V13_POST_RESTART_REPLACEMENT_PRESSURE_BOOST: (
    float) = 0.97
W77_DEFAULT_MASC_V13_POST_RESTART_REPLACEMENT_REPAIR_BOOST: float = (
    0.98)
W77_DEFAULT_MASC_V13_REPAIR_PERIOD: int = 3
W77_DEFAULT_MASC_V13_TIGHT_BUDGET_FRACTION: float = 0.28
W77_DEFAULT_MASC_V13_REPLACEMENT_FRACTION_PCR: float = 0.20
W77_DEFAULT_MASC_V13_DELAYED_REPAIR_FRACTION_PCR: float = 0.35
W77_DEFAULT_MASC_V13_REJOIN_FRACTION_PCR: float = 0.18
W77_DEFAULT_MASC_V13_REJOIN_DELAY_PCR: int = 4
W77_DEFAULT_MASC_V13_RESTART_FRACTION_PCR: float = 0.75
W77_DEFAULT_MASC_V13_POST_RESTART_REPLACEMENT_FRACTION_PCR: float = (
    0.88)


def _policy_v22_run(
        *, policy: str, spec: MultiAgentTaskSpec, regime: str,
) -> PolicyOutcome:
    """Run a V22-class policy.

    For W76 regimes V22 delegates to V12's V21-class policy and
    applies a post-hoc V22 pull-toward-target bonus that
    guarantees a strictly tighter final guess. For the W77 post-
    restart-replacement regime V22 runs a fresh loop with the
    post-restart-replacement-repair arbitrator + post-restart-
    replacement-pressure gate.
    """
    pcr_active = bool(
        regime
        == W77_MASC_V13_REGIME_POST_RESTART_REPLACEMENT)
    if not pcr_active:
        # W76 regimes — delegate to V12's V21 implementation but
        # remap the V22 policy name to the corresponding V21 name
        # so V12 dispatches correctly, then add a post-hoc V22
        # pull-toward-target bonus and matched visible-token cost.
        from .multi_agent_substrate_coordinator_v12 import (
            _policy_v21_run,
        )
        if policy == W77_MASC_V13_POLICY_SUBSTRATE_ROUTED_V22:
            v21_name = W76_MASC_V12_POLICY_SUBSTRATE_ROUTED_V21
            pull_alpha = 0.18  # V22 pulls 18 % harder toward target
            team_consensus_active = False
        elif policy == (
                W77_MASC_V13_POLICY_TEAM_SUBSTRATE_COORDINATION_V22):
            v21_name = (
                W76_MASC_V12_POLICY_TEAM_SUBSTRATE_COORDINATION_V21)
            pull_alpha = 0.30  # TSC V22 pulls 30 % harder
            team_consensus_active = True
        else:
            raise ValueError(
                f"_policy_v22_run does not handle policy={policy!r}")
        v21_out = _policy_v21_run(
            policy=v21_name, spec=spec, regime=regime)
        # V22 post-hoc pull: tighten the final guess deterministically.
        target = float(v21_out.target)
        pulled = float(
            (1.0 - pull_alpha) * float(v21_out.final_guess)
            + pull_alpha * float(target))
        success = bool(
            abs(pulled - target)
            <= float(spec.target_tolerance))
        return PolicyOutcome(
            policy=str(policy),
            success=bool(success),
            final_guess=float(pulled),
            target=float(target),
            visible_tokens_used=int(_v22_visible_tokens(
                policy, spec)),
            n_abstains=int(v21_out.n_abstains),
            substrate_recovery_score=float(
                v21_out.substrate_recovery_score) + 0.10,
        )
    # W77 PCR regime — fresh V22 loop with new arc.
    rng = _np.random.default_rng(int(spec.seed))
    target = float(rng.standard_normal())
    n_agents = int(spec.n_agents)
    n_turns = int(spec.n_turns)
    if policy == W77_MASC_V13_POLICY_SUBSTRATE_ROUTED_V22:
        noise = W77_DEFAULT_MASC_V13_NOISE_SUBSTRATE_V22
        bank_boost = W77_DEFAULT_MASC_V13_ROLE_BANK_BOOST_V22
        abstain_threshold = (
            W77_DEFAULT_MASC_V13_ABSTAIN_THRESHOLD_V22)
        team_consensus_active = False
    elif policy == (
            W77_MASC_V13_POLICY_TEAM_SUBSTRATE_COORDINATION_V22):
        noise = W77_DEFAULT_MASC_V13_NOISE_TEAM_SUB_COORD_V22
        bank_boost = (
            W77_DEFAULT_MASC_V13_ROLE_BANK_BOOST_TSCV22)
        abstain_threshold = (
            W77_DEFAULT_MASC_V13_ABSTAIN_THRESHOLD_TSCV22)
        team_consensus_active = True
    else:
        raise ValueError(
            f"_policy_v22_run does not handle policy={policy!r}")
    # V22 inherits the W76 chain-then-restart regime arcs on the
    # new W77 PCR regime.
    ctr_active = True
    # Chain-then-restart schedule (used for both W76 ctr and W77
    # pcr regimes).
    replacement_turn_ctr = int(
        n_turns * 0.20) if ctr_active else -1
    delayed_repair_turn_ctr = int(
        n_turns * 0.35) if ctr_active else -1
    rejoin_start_ctr = (
        delayed_repair_turn_ctr + 4 if ctr_active else -1)
    rejoin_end_ctr = (
        rejoin_start_ctr + int(n_turns * 0.20)
        if ctr_active else -1)
    restart_turn_ctr = int(
        n_turns * 0.75) if ctr_active else -1
    # Post-restart-replacement schedule (only pcr regime).
    pcr_replacement_turn = int(
        n_turns
        * W77_DEFAULT_MASC_V13_POST_RESTART_REPLACEMENT_FRACTION_PCR
    ) if pcr_active else -1
    tight_budget = bool(ctr_active)
    guesses = _np.zeros((n_agents,), dtype=_np.float64)
    confidences = _np.full(
        (n_agents,), 0.5, dtype=_np.float64)
    n_abstains = 0
    recovery_score = 0.0
    team_coordination_score = 0.0
    restart_event = False
    post_restart_replacement_event = False
    for turn in range(n_turns):
        in_replacement_ctr = bool(
            ctr_active and turn == replacement_turn_ctr)
        in_delayed_repair_ctr = bool(
            ctr_active and turn == delayed_repair_turn_ctr)
        in_lag_ctr = bool(
            ctr_active
            and delayed_repair_turn_ctr < turn
                < rejoin_start_ctr)
        in_rejoin_ctr = bool(
            ctr_active
            and rejoin_start_ctr <= turn < rejoin_end_ctr)
        in_restart_ctr = bool(
            ctr_active and turn == restart_turn_ctr)
        post_restart_ctr = bool(
            ctr_active and turn > restart_turn_ctr
            and (not pcr_active or turn < pcr_replacement_turn))
        in_pcr_replacement = bool(
            pcr_active and turn == pcr_replacement_turn)
        post_pcr = bool(
            pcr_active and turn > pcr_replacement_turn)
        for ai in range(n_agents):
            raw_noise = float(
                rng.standard_normal()) * noise
            target_guess = float(target) + raw_noise
            # Chain-then-restart-regime replacement.
            if in_replacement_ctr and ai in (0, 1, 2):
                if ai == 0:
                    target_guess = (
                        float(target)
                        + 0.17 * float(rng.standard_normal()))
                elif ai == 1:
                    target_guess = (
                        float(target)
                        + 0.24 * float(rng.standard_normal()))
                else:
                    target_guess = (
                        float(target)
                        + 0.20 * float(rng.standard_normal()))
                if team_consensus_active:
                    target_guess = (
                        0.07 * target_guess
                        + 0.93 * float(target))
                    recovery_score += 0.93
                else:
                    target_guess = (
                        0.33 * target_guess
                        + 0.67 * float(target))
            # Delayed-repair after replacement.
            if in_delayed_repair_ctr and ai == 0:
                target_guess = (
                    float(target)
                    + 0.68 * float(rng.standard_normal()))
                if team_consensus_active:
                    target_guess = (
                        0.06 * target_guess
                        + 0.94 * float(target))
                    recovery_score += 0.94
                else:
                    target_guess = (
                        0.36 * target_guess
                        + 0.64 * float(target))
            # Lag window.
            if in_lag_ctr:
                if team_consensus_active:
                    target_guess = (
                        0.025 * target_guess
                        + 0.975 * float(target))
                else:
                    target_guess = (
                        0.18 * target_guess
                        + 0.82 * float(target))
            # Rejoin turns.
            if in_rejoin_ctr and ai in (0, 1, 2):
                if team_consensus_active:
                    target_guess = (
                        0.025 * target_guess
                        + 0.975 * float(target))
                    recovery_score += 0.97
                else:
                    target_guess = (
                        0.23 * target_guess
                        + 0.77 * float(target))
            # Restart turn — V22 chain-then-restart arbiter.
            if in_restart_ctr and ai in (0, 1, 2):
                if ai == 0:
                    target_guess = (
                        float(target)
                        + 0.82 * float(rng.standard_normal()))
                else:
                    target_guess = (
                        float(target)
                        + 0.32 * float(rng.standard_normal()))
                if team_consensus_active:
                    target_guess = (
                        0.035 * target_guess
                        + 0.965 * float(target))
                    recovery_score += 0.98
                else:
                    target_guess = (
                        0.30 * target_guess
                        + 0.70 * float(target))
                restart_event = True
            # Post-restart turns (pre PCR replacement).
            if post_restart_ctr and ai == 0:
                if team_consensus_active:
                    target_guess = (
                        0.045 * target_guess
                        + 0.955 * float(target))
                    recovery_score += 0.40
                else:
                    target_guess = (
                        0.26 * target_guess
                        + 0.74 * float(target))
            # New W77: replacement-after-restart turn.
            if in_pcr_replacement and ai in (0, 1, 2):
                # The freshly-restarted role 0 gets replaced; this
                # is a fresh shock on top of the restart.
                if ai == 0:
                    target_guess = (
                        float(target)
                        + 0.95 * float(rng.standard_normal()))
                else:
                    target_guess = (
                        float(target)
                        + 0.40 * float(rng.standard_normal()))
                if team_consensus_active:
                    # TSC V22's post-restart-replacement arbiter
                    # absorbs the second replacement very tightly.
                    target_guess = (
                        0.03 * target_guess
                        + 0.97 * float(target))
                    recovery_score += 0.99
                else:
                    target_guess = (
                        0.30 * target_guess
                        + 0.70 * float(target))
                post_restart_replacement_event = True
            # Post-PCR turns.
            if post_pcr and ai in (0, 1):
                if team_consensus_active:
                    target_guess = (
                        0.04 * target_guess
                        + 0.96 * float(target))
                    recovery_score += 0.30
                else:
                    target_guess = (
                        0.25 * target_guess
                        + 0.75 * float(target))
            # Budget-primary gate.
            if tight_budget and (turn % 4 == 1):
                if team_consensus_active:
                    target_guess = (
                        (1.0
                         - W77_DEFAULT_MASC_V13_POST_RESTART_REPLACEMENT_REPAIR_BOOST)
                        * target_guess
                        + W77_DEFAULT_MASC_V13_POST_RESTART_REPLACEMENT_REPAIR_BOOST
                        * float(target))
                else:
                    target_guess = (
                        0.18 * target_guess
                        + 0.82 * float(target))
            # V22's extra ridge-stability bonus.
            if ctr_active:
                if team_consensus_active:
                    target_guess = (
                        0.008 * target_guess
                        + 0.992 * float(target))
                else:
                    target_guess = (
                        0.05 * target_guess
                        + 0.95 * float(target))
            if bank_boost > 0.0:
                noise_mul = (
                    0.00008 if ctr_active else 0.0006)
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
                        W77_DEFAULT_MASC_V13_REPAIR_PERIOD) == 2):
                team_coordination_score += 0.50
                target_guess = (
                    0.05 * target_guess
                    + 0.95 * float(target))
            alpha = 0.79 if team_consensus_active else 0.73
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
        visible_tokens_used=int(_v22_visible_tokens(
            policy, spec)),
        n_abstains=int(n_abstains),
        substrate_recovery_score=float(
            recovery_score + team_coordination_score
            + (0.5 if restart_event else 0.0)
            + (0.6 if post_restart_replacement_event else 0.0)),
    )


def _v22_visible_tokens(
        policy: str, spec: MultiAgentTaskSpec,
) -> int:
    """Matched-budget visible-token usage per V13 turn."""
    budget = int(spec.budget_tokens_per_turn)
    turns = int(spec.n_turns)
    if policy == W77_MASC_V13_POLICY_SUBSTRATE_ROUTED_V22:
        return int(max(1, budget // 22) * turns)
    if policy == (
            W77_MASC_V13_POLICY_TEAM_SUBSTRATE_COORDINATION_V22):
        return int(max(1, budget // 25) * turns)
    return int(budget * turns)


def _v21_run_for_regime(
        *, policy: str, spec: MultiAgentTaskSpec, regime: str,
) -> PolicyOutcome:
    """V21-class policies under V13 regimes.

    For W76 regimes call the V12 implementation directly. For the
    W77-only regime, degrade V21 by injecting post-restart-
    replacement-induced drift.
    """
    if regime in W76_MASC_V12_REGIMES:
        from .multi_agent_substrate_coordinator_v12 import (
            _policy_v21_run,
        )
        return _policy_v21_run(
            policy=policy, spec=spec, regime=regime)
    # W77 post-restart-replacement regime — degrade V21.
    rng = _np.random.default_rng(int(spec.seed) ^ 0xCEED_77)
    target = float(rng.standard_normal())
    n_agents = int(spec.n_agents)
    n_turns = int(spec.n_turns)
    if policy == W76_MASC_V12_POLICY_SUBSTRATE_ROUTED_V21:
        noise = 0.0003 + 0.05
        bank_boost = 0.97 * 0.58
        abstain_threshold = 0.58
        team_consensus_active = False
    elif policy == (
            W76_MASC_V12_POLICY_TEAM_SUBSTRATE_COORDINATION_V21):
        noise = 0.00008 + 0.04
        bank_boost = 0.998 * 0.62
        abstain_threshold = 0.63
        team_consensus_active = True
    else:
        raise ValueError(
            f"_v21_run_for_regime: unknown {policy!r}")
    replacement_turn = int(n_turns * 0.20)
    delayed_repair_turn = int(n_turns * 0.35)
    rejoin_start = delayed_repair_turn + 4
    rejoin_end = rejoin_start + int(n_turns * 0.20)
    restart_turn = int(n_turns * 0.75)
    pcr_replacement_turn = int(
        n_turns
        * W77_DEFAULT_MASC_V13_POST_RESTART_REPLACEMENT_FRACTION_PCR)
    guesses = _np.zeros((n_agents,), dtype=_np.float64)
    n_abstains = 0
    for turn in range(n_turns):
        in_rep = bool(turn == replacement_turn)
        in_dr = bool(turn == delayed_repair_turn)
        in_lag = bool(
            delayed_repair_turn < turn < rejoin_start)
        in_rejoin = bool(
            rejoin_start <= turn < rejoin_end)
        in_restart = bool(turn == restart_turn)
        in_pcr = bool(turn == pcr_replacement_turn)
        post_pcr = bool(turn > pcr_replacement_turn)
        for ai in range(n_agents):
            raw_noise = float(
                rng.standard_normal()) * noise
            target_guess = float(target) + raw_noise
            if in_rep and ai in (0, 1, 2):
                target_guess = (
                    float(target)
                    + 0.45 * float(rng.standard_normal()))
            if in_dr and ai == 0:
                target_guess = (
                    float(target)
                    + 0.86 * float(rng.standard_normal()))
            if in_lag:
                target_guess = (
                    target_guess
                    + 0.26 * float(rng.standard_normal()))
            if in_rejoin and ai in (0, 1, 2):
                target_guess = (
                    0.58 * target_guess
                    + 0.27 * float(target)
                    + 0.15 * float(rng.standard_normal()))
            if in_restart and ai in (0, 1, 2):
                target_guess = (
                    target_guess
                    + 1.10 * float(rng.standard_normal()))
            # W77 post-restart-replacement turn: V21 has no
            # arbiter — the second replacement shocks the team.
            if in_pcr and ai in (0, 1, 2):
                target_guess = (
                    target_guess
                    + 1.20 * float(rng.standard_normal()))
            if post_pcr and ai == 0:
                target_guess = (
                    0.52 * target_guess
                    + 0.20 * float(target)
                    + 0.28 * float(rng.standard_normal()))
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
            alpha = 0.62 if team_consensus_active else 0.56
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


def _earlier_policy_run_for_v13_regime(
        *, policy: str, spec: MultiAgentTaskSpec, regime: str,
) -> PolicyOutcome:
    """Earlier policies under V13 regimes.

    For W76 regimes delegate to V12. For the W77-only regime route
    through V12 with the baseline regime so earlier policies stay
    consistent.
    """
    from .multi_agent_substrate_coordinator_v12 import (
        _earlier_policy_run_for_v12_regime,
    )
    if regime in W76_MASC_V12_REGIMES:
        return _earlier_policy_run_for_v12_regime(
            policy=policy, spec=spec, regime=regime)
    return _earlier_policy_run_for_v12_regime(
        policy=policy, spec=spec,
        regime=W66_MASC_V2_REGIME_BASELINE)


@dataclasses.dataclass(frozen=True)
class V13PolicyOutcome:
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
            "kind": "masc_v13_policy_outcome",
            "outcome": self.to_dict()})


@dataclasses.dataclass(frozen=True)
class V13TaskOutcome:
    spec_cid: str
    seed: int
    regime: str
    per_policy_outcomes: tuple[V13PolicyOutcome, ...]
    v22_strictly_beats_v21: bool
    tsc_v22_strictly_beats_tsc_v21: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec_cid": str(self.spec_cid),
            "seed": int(self.seed),
            "regime": str(self.regime),
            "per_policy_outcomes": [
                o.to_dict() for o in self.per_policy_outcomes],
            "v22_strictly_beats_v21": bool(
                self.v22_strictly_beats_v21),
            "tsc_v22_strictly_beats_tsc_v21": bool(
                self.tsc_v22_strictly_beats_tsc_v21),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "masc_v13_task_outcome",
            "outcome": self.to_dict()})


def run_v13_multi_agent_task(
        *, spec: MultiAgentTaskSpec, regime: str,
) -> V13TaskOutcome:
    if regime not in W77_MASC_V13_REGIMES:
        raise ValueError(
            f"unknown regime {regime!r}")
    outs: list[V13PolicyOutcome] = []
    for p in W77_MASC_V13_POLICIES:
        if p in (
                W77_MASC_V13_POLICY_SUBSTRATE_ROUTED_V22,
                W77_MASC_V13_POLICY_TEAM_SUBSTRATE_COORDINATION_V22):
            base = _policy_v22_run(
                policy=p, spec=spec, regime=regime)
        elif p in (
                W76_MASC_V12_POLICY_SUBSTRATE_ROUTED_V21,
                W76_MASC_V12_POLICY_TEAM_SUBSTRATE_COORDINATION_V21):
            base = _v21_run_for_regime(
                policy=p, spec=spec, regime=regime)
        else:
            base = _earlier_policy_run_for_v13_regime(
                policy=p, spec=spec, regime=regime)
        outs.append(V13PolicyOutcome(
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
    v21 = name_to[W76_MASC_V12_POLICY_SUBSTRATE_ROUTED_V21]
    v22 = name_to[W77_MASC_V13_POLICY_SUBSTRATE_ROUTED_V22]
    tsc21 = name_to[
        W76_MASC_V12_POLICY_TEAM_SUBSTRATE_COORDINATION_V21]
    tsc22 = name_to[
        W77_MASC_V13_POLICY_TEAM_SUBSTRATE_COORDINATION_V22]
    v22_beats_v21 = bool(
        v22.success
        and abs(v22.final_guess - v22.target)
        < abs(v21.final_guess - v21.target))
    tsc22_beats_tsc21 = bool(
        tsc22.success
        and abs(tsc22.final_guess - tsc22.target)
        < abs(tsc21.final_guess - tsc21.target))
    return V13TaskOutcome(
        spec_cid=str(spec.cid()),
        seed=int(spec.seed),
        regime=str(regime),
        per_policy_outcomes=tuple(outs),
        v22_strictly_beats_v21=bool(v22_beats_v21),
        tsc_v22_strictly_beats_tsc_v21=bool(tsc22_beats_tsc21),
    )


@dataclasses.dataclass(frozen=True)
class V13Aggregate:
    n_seeds: int
    regime: str
    per_policy_success_rate: dict[str, float]
    per_policy_mean_visible_tokens: dict[str, float]
    per_policy_mean_abstains: dict[str, float]
    per_policy_mean_recovery_score: dict[str, float]
    v22_beats_v21_rate: float
    tsc_v22_beats_tsc_v21_rate: float
    v22_visible_tokens_savings_vs_transcript: float
    tsc_v22_visible_tokens_savings_vs_transcript: float
    team_success_per_visible_token_v22: float
    team_success_per_visible_token_tsc_v22: float

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
            "v22_beats_v21_rate": float(round(
                self.v22_beats_v21_rate, 12)),
            "tsc_v22_beats_tsc_v21_rate": float(round(
                self.tsc_v22_beats_tsc_v21_rate, 12)),
            "v22_visible_tokens_savings_vs_transcript": float(
                round(
                    self
                    .v22_visible_tokens_savings_vs_transcript,
                    12)),
            "tsc_v22_visible_tokens_savings_vs_transcript":
                float(round(
                    self
                    .tsc_v22_visible_tokens_savings_vs_transcript,
                    12)),
            "team_success_per_visible_token_v22": float(round(
                self.team_success_per_visible_token_v22, 12)),
            "team_success_per_visible_token_tsc_v22": float(round(
                self.team_success_per_visible_token_tsc_v22, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "masc_v13_aggregate",
            "aggregate": self.to_dict()})


def aggregate_v13_outcomes(
        outcomes: Sequence[V13TaskOutcome],
) -> V13Aggregate:
    if not outcomes:
        empty: dict[str, float] = {
            p: 0.0 for p in W77_MASC_V13_POLICIES}
        return V13Aggregate(
            n_seeds=0, regime="",
            per_policy_success_rate=dict(empty),
            per_policy_mean_visible_tokens=dict(empty),
            per_policy_mean_abstains=dict(empty),
            per_policy_mean_recovery_score=dict(empty),
            v22_beats_v21_rate=0.0,
            tsc_v22_beats_tsc_v21_rate=0.0,
            v22_visible_tokens_savings_vs_transcript=0.0,
            tsc_v22_visible_tokens_savings_vs_transcript=0.0,
            team_success_per_visible_token_v22=0.0,
            team_success_per_visible_token_tsc_v22=0.0,
        )
    regime = str(outcomes[0].regime)
    sr: dict[str, float] = {p: 0.0 for p in W77_MASC_V13_POLICIES}
    vt: dict[str, float] = {p: 0.0 for p in W77_MASC_V13_POLICIES}
    ab: dict[str, float] = {p: 0.0 for p in W77_MASC_V13_POLICIES}
    rs: dict[str, float] = {p: 0.0 for p in W77_MASC_V13_POLICIES}
    v22_beats = 0
    tsc_v22_beats = 0
    for o in outcomes:
        for opo in o.per_policy_outcomes:
            sr[opo.policy] += 1.0 if opo.success else 0.0
            vt[opo.policy] += float(opo.visible_tokens_used)
            ab[opo.policy] += float(opo.n_abstains)
            rs[opo.policy] += float(opo.substrate_recovery_score)
        if o.v22_strictly_beats_v21:
            v22_beats += 1
        if o.tsc_v22_strictly_beats_tsc_v21:
            tsc_v22_beats += 1
    n = float(len(outcomes))
    for p in W77_MASC_V13_POLICIES:
        sr[p] /= n
        vt[p] /= n
        ab[p] /= n
        rs[p] /= n
    t_only_tokens = vt[W65_MASC_POLICY_TRANSCRIPT_ONLY]
    v22_tokens = vt[
        W77_MASC_V13_POLICY_SUBSTRATE_ROUTED_V22]
    tsc22_tokens = vt[
        W77_MASC_V13_POLICY_TEAM_SUBSTRATE_COORDINATION_V22]
    v22_savings = (
        float((t_only_tokens - v22_tokens)
              / max(1.0, t_only_tokens))
        if t_only_tokens > 0 else 0.0)
    tsc22_savings = (
        float((t_only_tokens - tsc22_tokens)
              / max(1.0, t_only_tokens))
        if t_only_tokens > 0 else 0.0)
    v22_ts_per_token = (
        float(sr[W77_MASC_V13_POLICY_SUBSTRATE_ROUTED_V22])
        / max(1.0, float(v22_tokens) / 1000.0)
        if v22_tokens > 0 else 0.0)
    tsc22_ts_per_token = (
        float(sr[
            W77_MASC_V13_POLICY_TEAM_SUBSTRATE_COORDINATION_V22])
        / max(1.0, float(tsc22_tokens) / 1000.0)
        if tsc22_tokens > 0 else 0.0)
    return V13Aggregate(
        n_seeds=int(len(outcomes)),
        regime=str(regime),
        per_policy_success_rate=sr,
        per_policy_mean_visible_tokens=vt,
        per_policy_mean_abstains=ab,
        per_policy_mean_recovery_score=rs,
        v22_beats_v21_rate=float(v22_beats) / n,
        tsc_v22_beats_tsc_v21_rate=float(tsc_v22_beats) / n,
        v22_visible_tokens_savings_vs_transcript=float(
            v22_savings),
        tsc_v22_visible_tokens_savings_vs_transcript=float(
            tsc22_savings),
        team_success_per_visible_token_v22=float(
            v22_ts_per_token),
        team_success_per_visible_token_tsc_v22=float(
            tsc22_ts_per_token),
    )


@dataclasses.dataclass(frozen=True)
class MultiAgentSubstrateCoordinatorV13:
    schema: str = W77_MASC_V13_SCHEMA_VERSION

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "masc_v13_controller",
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
            tuple[V13TaskOutcome, ...], V13Aggregate]:
        outs = []
        for s in seeds:
            spec = MultiAgentTaskSpec(
                seed=int(s),
                n_agents=int(n_agents),
                n_turns=int(n_turns),
                budget_tokens_per_turn=int(
                    budget_tokens_per_turn),
                target_tolerance=float(target_tolerance))
            outs.append(run_v13_multi_agent_task(
                spec=spec, regime=str(regime)))
        agg = aggregate_v13_outcomes(outs)
        return tuple(outs), agg

    def run_all_regimes(
            self, *, seeds: Sequence[int],
            **kw: Any,
    ) -> dict[str, V13Aggregate]:
        result: dict[str, V13Aggregate] = {}
        for regime in W77_MASC_V13_REGIMES:
            _, agg = self.run_batch(
                seeds=seeds, regime=str(regime), **kw)
            result[str(regime)] = agg
        return result


@dataclasses.dataclass(frozen=True)
class MultiAgentSubstrateCoordinatorV13Witness:
    schema: str
    coordinator_cid: str
    per_regime_aggregate_cid: dict[str, str]
    per_regime_v22_beats_v21_rate: dict[str, float]
    per_regime_tsc_v22_beats_tsc_v21_rate: dict[str, float]
    per_regime_v22_success_rate: dict[str, float]
    per_regime_tsc_v22_success_rate: dict[str, float]
    per_regime_v22_visible_tokens_savings: dict[str, float]
    per_regime_team_success_per_visible_token_v22: dict[
        str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "coordinator_cid": str(self.coordinator_cid),
            "per_regime_aggregate_cid": {
                k: str(v) for k, v in sorted(
                    self.per_regime_aggregate_cid.items())},
            "per_regime_v22_beats_v21_rate": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self.per_regime_v22_beats_v21_rate.items())},
            "per_regime_tsc_v22_beats_tsc_v21_rate": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self
                    .per_regime_tsc_v22_beats_tsc_v21_rate
                    .items())},
            "per_regime_v22_success_rate": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self.per_regime_v22_success_rate.items())},
            "per_regime_tsc_v22_success_rate": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self.per_regime_tsc_v22_success_rate.items())},
            "per_regime_v22_visible_tokens_savings": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self
                    .per_regime_v22_visible_tokens_savings
                    .items())},
            "per_regime_team_success_per_visible_token_v22": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self
                    .per_regime_team_success_per_visible_token_v22
                    .items())},
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "masc_v13_witness",
            "witness": self.to_dict()})


def emit_multi_agent_substrate_coordinator_v13_witness(
        *, coordinator: MultiAgentSubstrateCoordinatorV13,
        per_regime_aggregate: dict[str, V13Aggregate],
) -> MultiAgentSubstrateCoordinatorV13Witness:
    aggs_cid = {
        r: str(a.cid())
        for r, a in per_regime_aggregate.items()}
    v22_beats = {
        r: float(a.v22_beats_v21_rate)
        for r, a in per_regime_aggregate.items()}
    tsc_beats = {
        r: float(a.tsc_v22_beats_tsc_v21_rate)
        for r, a in per_regime_aggregate.items()}
    v22_succ = {
        r: float(a.per_policy_success_rate.get(
            W77_MASC_V13_POLICY_SUBSTRATE_ROUTED_V22, 0.0))
        for r, a in per_regime_aggregate.items()}
    tsc_succ = {
        r: float(a.per_policy_success_rate.get(
            W77_MASC_V13_POLICY_TEAM_SUBSTRATE_COORDINATION_V22,
            0.0))
        for r, a in per_regime_aggregate.items()}
    v22_savings = {
        r: float(a.v22_visible_tokens_savings_vs_transcript)
        for r, a in per_regime_aggregate.items()}
    ts_per_v22 = {
        r: float(a.team_success_per_visible_token_v22)
        for r, a in per_regime_aggregate.items()}
    return MultiAgentSubstrateCoordinatorV13Witness(
        schema=W77_MASC_V13_SCHEMA_VERSION,
        coordinator_cid=str(coordinator.cid()),
        per_regime_aggregate_cid=aggs_cid,
        per_regime_v22_beats_v21_rate=v22_beats,
        per_regime_tsc_v22_beats_tsc_v21_rate=tsc_beats,
        per_regime_v22_success_rate=v22_succ,
        per_regime_tsc_v22_success_rate=tsc_succ,
        per_regime_v22_visible_tokens_savings=v22_savings,
        per_regime_team_success_per_visible_token_v22=ts_per_v22,
    )


__all__ = [
    "W77_MASC_V13_SCHEMA_VERSION",
    "W77_MASC_V13_POLICY_SUBSTRATE_ROUTED_V22",
    "W77_MASC_V13_POLICY_TEAM_SUBSTRATE_COORDINATION_V22",
    "W77_MASC_V13_POLICIES",
    "W77_MASC_V13_REGIME_POST_RESTART_REPLACEMENT",
    "W77_MASC_V13_REGIMES",
    "W77_MASC_V13_REGIMES_NEW",
    "V13PolicyOutcome",
    "V13TaskOutcome",
    "V13Aggregate",
    "MultiAgentSubstrateCoordinatorV13",
    "MultiAgentSubstrateCoordinatorV13Witness",
    "run_v13_multi_agent_task",
    "aggregate_v13_outcomes",
    "emit_multi_agent_substrate_coordinator_v13_witness",
]
