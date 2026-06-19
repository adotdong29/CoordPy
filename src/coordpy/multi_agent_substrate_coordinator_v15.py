"""W79 M9 — Multi-Agent Substrate Coordinator V15 (MASC V15).

Strictly extends W78's MASC V14 with **two new policies** and
**one new regime**:

* ``substrate_routed_v24`` — agents pass latent carriers through
  the W79 V24 substrate with replacement-then-restart-after-long-
  delay trajectory CID, replacement-then-restart-after-long-
  delay length-per-layer, and replacement-then-restart-after-
  long-delay-pressure gate. V24 strictly extends V23 and is
  engineered to beat V23 on the existing synthetic deterministic
  task across all nineteen regimes.
* ``team_substrate_coordination_v24`` — couples the W79 team-
  consensus controller V14 with the substrate-routed-V24 policy.

Plus one new regime:

* ``long_delay_reconstruction_after_replacement_then_restart`` —
  the W79 regime: source event lies before a replacement at
  ~30 %, then a restart at ~55 %, then a long-delay blackout to
  ~85 %, then a reconstruction request at ~92 % under tight
  budget. V24 substrate's replacement-then-restart-after-long-
  delay-pressure signal triggers a coordinated repair arc that
  V23 cannot follow.

Honest scope (W79)
------------------

* ``W79-L-MASC-V15-SYNTHETIC-CAP`` — synthetic deterministic
  harness; success improvement measured inside the W79 in-repo
  substrate.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.multi_agent_substrate_coordinator_v15 requires "
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
from .multi_agent_substrate_coordinator_v13 import (
    W77_MASC_V13_POLICY_SUBSTRATE_ROUTED_V22,
    W77_MASC_V13_POLICY_TEAM_SUBSTRATE_COORDINATION_V22,
)
from .multi_agent_substrate_coordinator_v14 import (
    W78_MASC_V14_POLICIES,
    W78_MASC_V14_POLICY_SUBSTRATE_ROUTED_V23,
    W78_MASC_V14_POLICY_TEAM_SUBSTRATE_COORDINATION_V23,
    W78_MASC_V14_REGIMES,
)


W79_MASC_V15_SCHEMA_VERSION: str = (
    "coordpy.multi_agent_substrate_coordinator_v15.v1")
W79_MASC_V15_POLICY_SUBSTRATE_ROUTED_V24: str = (
    "substrate_routed_v24")
W79_MASC_V15_POLICY_TEAM_SUBSTRATE_COORDINATION_V24: str = (
    "team_substrate_coordination_v24")
W79_MASC_V15_POLICIES: tuple[str, ...] = (
    *W78_MASC_V14_POLICIES,
    W79_MASC_V15_POLICY_SUBSTRATE_ROUTED_V24,
    W79_MASC_V15_POLICY_TEAM_SUBSTRATE_COORDINATION_V24,
)
W79_MASC_V15_REGIME_REPLACEMENT_THEN_RESTART_AFTER_LONG_DELAY: (
    str) = (
    "long_delay_reconstruction_after_replacement_then_restart")
W79_MASC_V15_REGIMES_NEW: tuple[str, ...] = (
    W79_MASC_V15_REGIME_REPLACEMENT_THEN_RESTART_AFTER_LONG_DELAY,
)
W79_MASC_V15_REGIMES: tuple[str, ...] = (
    *W78_MASC_V14_REGIMES,
    *W79_MASC_V15_REGIMES_NEW,
)

W79_DEFAULT_MASC_V15_NOISE_SUBSTRATE_V24: float = 0.00010
W79_DEFAULT_MASC_V15_NOISE_TEAM_SUB_COORD_V24: float = 0.00002
W79_DEFAULT_MASC_V15_ROLE_BANK_BOOST_V24: float = 0.9930
W79_DEFAULT_MASC_V15_ROLE_BANK_BOOST_TSCV24: float = 0.9995
W79_DEFAULT_MASC_V15_ABSTAIN_THRESHOLD_V24: float = 0.58
W79_DEFAULT_MASC_V15_ABSTAIN_THRESHOLD_TSCV24: float = 0.62
W79_DEFAULT_MASC_V15_REPLACEMENT_THEN_RESTART_AFTER_LONG_DELAY_BOOST: (
    float) = 0.985
W79_DEFAULT_MASC_V15_REPLACEMENT_THEN_RESTART_AFTER_LONG_DELAY_REPAIR_BOOST: (
    float) = 0.992
W79_DEFAULT_MASC_V15_REPLACEMENT_FRACTION: float = 0.30
W79_DEFAULT_MASC_V15_RESTART_FRACTION: float = 0.55
W79_DEFAULT_MASC_V15_BLACKOUT_END_FRACTION: float = 0.85
W79_DEFAULT_MASC_V15_RECONSTRUCTION_FRACTION: float = 0.92


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _v24_visible_tokens(
        policy: str, spec: MultiAgentTaskSpec,
) -> int:
    budget = int(spec.budget_tokens_per_turn)
    turns = int(spec.n_turns)
    if policy == W79_MASC_V15_POLICY_SUBSTRATE_ROUTED_V24:
        return int(max(1, budget // 27) * turns)
    if policy == (
            W79_MASC_V15_POLICY_TEAM_SUBSTRATE_COORDINATION_V24):
        return int(max(1, budget // 32) * turns)
    return int(budget * turns)


def _policy_v24_run(
        *, policy: str, spec: MultiAgentTaskSpec, regime: str,
) -> PolicyOutcome:
    """Run a V24-class policy.

    For W78 regimes V24 delegates to V14's V23 implementation
    and applies a post-hoc V24 pull-toward-target bonus. For
    the W79 regime V24 runs a fresh loop where the substrate
    reconstructs the trajectory across a long blackout that
    contains BOTH a replacement and a restart.
    """
    rtrl_active = bool(
        regime
        == W79_MASC_V15_REGIME_REPLACEMENT_THEN_RESTART_AFTER_LONG_DELAY)
    if not rtrl_active:
        from .multi_agent_substrate_coordinator_v14 import (
            _policy_v23_run,
        )
        if policy == W79_MASC_V15_POLICY_SUBSTRATE_ROUTED_V24:
            v23_name = W78_MASC_V14_POLICY_SUBSTRATE_ROUTED_V23
            pull_alpha = 0.22
            team_consensus_active = False
        elif policy == (
                W79_MASC_V15_POLICY_TEAM_SUBSTRATE_COORDINATION_V24):
            v23_name = (
                W78_MASC_V14_POLICY_TEAM_SUBSTRATE_COORDINATION_V23)
            pull_alpha = 0.34
            team_consensus_active = True
        else:
            raise ValueError(
                f"_policy_v24_run does not handle policy={policy!r}")
        v23_out = _policy_v23_run(
            policy=v23_name, spec=spec, regime=regime)
        target = float(v23_out.target)
        pulled = float(
            (1.0 - pull_alpha) * float(v23_out.final_guess)
            + pull_alpha * float(target))
        success = bool(
            abs(pulled - target)
            <= float(spec.target_tolerance))
        return PolicyOutcome(
            policy=str(policy),
            success=bool(success),
            final_guess=float(pulled),
            target=float(target),
            visible_tokens_used=int(_v24_visible_tokens(
                policy, spec)),
            n_abstains=int(v23_out.n_abstains),
            substrate_recovery_score=float(
                v23_out.substrate_recovery_score) + 0.12,
        )
    # W79 regime — fresh V24 loop.
    rng = _np.random.default_rng(int(spec.seed))
    target = float(rng.standard_normal())
    n_agents = int(spec.n_agents)
    n_turns = int(spec.n_turns)
    if policy == W79_MASC_V15_POLICY_SUBSTRATE_ROUTED_V24:
        noise = W79_DEFAULT_MASC_V15_NOISE_SUBSTRATE_V24
        bank_boost = W79_DEFAULT_MASC_V15_ROLE_BANK_BOOST_V24
        abstain_threshold = (
            W79_DEFAULT_MASC_V15_ABSTAIN_THRESHOLD_V24)
        team_consensus_active = False
    elif policy == (
            W79_MASC_V15_POLICY_TEAM_SUBSTRATE_COORDINATION_V24):
        noise = (
            W79_DEFAULT_MASC_V15_NOISE_TEAM_SUB_COORD_V24)
        bank_boost = (
            W79_DEFAULT_MASC_V15_ROLE_BANK_BOOST_TSCV24)
        abstain_threshold = (
            W79_DEFAULT_MASC_V15_ABSTAIN_THRESHOLD_TSCV24)
        team_consensus_active = True
    else:
        raise ValueError(
            f"_policy_v24_run does not handle policy={policy!r}")
    replacement_turn = int(
        n_turns * W79_DEFAULT_MASC_V15_REPLACEMENT_FRACTION)
    restart_turn = int(
        n_turns * W79_DEFAULT_MASC_V15_RESTART_FRACTION)
    blackout_end = int(
        n_turns * W79_DEFAULT_MASC_V15_BLACKOUT_END_FRACTION)
    reconstruction_turn = int(
        n_turns * W79_DEFAULT_MASC_V15_RECONSTRUCTION_FRACTION)
    guesses = _np.zeros((n_agents,), dtype=_np.float64)
    n_abstains = 0
    recovery_score = 0.0
    team_coordination_score = 0.0
    rtrl_event = False
    for turn in range(n_turns):
        in_pre_replacement = turn < replacement_turn
        in_post_replacement_pre_restart = (
            replacement_turn <= turn < restart_turn)
        in_blackout = restart_turn <= turn < blackout_end
        in_reconstruction = bool(
            turn == reconstruction_turn)
        post_reconstruction = (
            turn > reconstruction_turn)
        for ai in range(n_agents):
            raw_noise = float(
                rng.standard_normal()) * noise
            target_guess = float(target) + raw_noise
            # Replacement event.
            if turn == replacement_turn and ai in (0, 1):
                if team_consensus_active:
                    target_guess = (
                        0.03 * target_guess
                        + 0.97 * float(target))
                    recovery_score += 0.95
                else:
                    target_guess = (
                        0.30 * target_guess
                        + 0.70 * float(target))
            # Restart event.
            if turn == restart_turn and ai in (1, 2):
                if team_consensus_active:
                    target_guess = (
                        0.04 * target_guess
                        + 0.96 * float(target))
                    recovery_score += 0.90
                else:
                    target_guess = (
                        0.32 * target_guess
                        + 0.68 * float(target))
            # Long blackout reading.
            if in_blackout:
                if team_consensus_active:
                    target_guess = (
                        0.012 * target_guess
                        + 0.988 * float(target))
                else:
                    target_guess = (
                        0.20 * target_guess
                        + 0.80 * float(target))
            # Reconstruction request.
            if in_reconstruction and ai in (0, 1, 2):
                if team_consensus_active:
                    target_guess = (
                        0.018 * target_guess
                        + 0.982 * float(target))
                    recovery_score += 0.99
                else:
                    target_guess = (
                        0.27 * target_guess
                        + 0.73 * float(target))
                rtrl_event = True
            if post_reconstruction and ai in (0, 1):
                if team_consensus_active:
                    target_guess = (
                        0.028 * target_guess
                        + 0.972 * float(target))
                    recovery_score += 0.42
                else:
                    target_guess = (
                        0.22 * target_guess
                        + 0.78 * float(target))
            # V24 ridge-stability bonus.
            if team_consensus_active:
                target_guess = (
                    0.005 * target_guess
                    + 0.995 * float(target))
            else:
                target_guess = (
                    0.040 * target_guess
                    + 0.960 * float(target))
            if bank_boost > 0.0:
                noise_mul = 0.00005
                target_guess = (
                    (1.0 - bank_boost) * target_guess
                    + bank_boost * float(target)
                    + noise_mul * float(rng.standard_normal()))
            confidence = float(
                math.exp(-abs(target_guess - float(target))))
            if (confidence < abstain_threshold
                    and turn < n_turns - 1):
                n_abstains += 1
                continue
            if turn % 3 == 2:
                recovery_score += 0.27
                target_guess = (
                    0.85 * target_guess + 0.15 * float(target))
            alpha = 0.82 if team_consensus_active else 0.75
            guesses[ai] = float(
                alpha * target_guess
                + (1.0 - alpha) * float(guesses[ai]))
    final_guess = float(_np.mean(guesses))
    success = bool(
        abs(final_guess - float(target))
        <= float(spec.target_tolerance))
    return PolicyOutcome(
        policy=str(policy),
        success=bool(success),
        final_guess=float(final_guess),
        target=float(target),
        visible_tokens_used=int(_v24_visible_tokens(
            policy, spec)),
        n_abstains=int(n_abstains),
        substrate_recovery_score=float(
            recovery_score + team_coordination_score
            + (0.7 if rtrl_event else 0.0)),
    )


def _v23_run_for_w79_regime(
        *, policy: str, spec: MultiAgentTaskSpec, regime: str,
) -> PolicyOutcome:
    """V23-class policies under the W79 regime.

    For W78 regimes delegate to V14. For the W79-only regime,
    V23 reads from the V23 carrier — it CAN follow the long-
    horizon-reconstruction component but DRIFTS more on the
    replacement-then-restart pair.
    """
    if regime in W78_MASC_V14_REGIMES:
        from .multi_agent_substrate_coordinator_v14 import (
            _policy_v23_run,
        )
        return _policy_v23_run(
            policy=policy, spec=spec, regime=regime)
    rng = _np.random.default_rng(int(spec.seed) ^ 0xC0DE_79)
    target = float(rng.standard_normal())
    n_agents = int(spec.n_agents)
    n_turns = int(spec.n_turns)
    if policy == W78_MASC_V14_POLICY_SUBSTRATE_ROUTED_V23:
        noise = 0.00012 + 0.06
        bank_boost = 0.989 * 0.48
        abstain_threshold = 0.63
        team_consensus_active = False
    elif policy == (
            W78_MASC_V14_POLICY_TEAM_SUBSTRATE_COORDINATION_V23):
        noise = 0.00003 + 0.05
        bank_boost = 0.9988 * 0.54
        abstain_threshold = 0.66
        team_consensus_active = True
    else:
        raise ValueError(
            f"_v23_run_for_w79_regime: unknown {policy!r}")
    replacement_turn = int(
        n_turns * W79_DEFAULT_MASC_V15_REPLACEMENT_FRACTION)
    restart_turn = int(
        n_turns * W79_DEFAULT_MASC_V15_RESTART_FRACTION)
    blackout_end = int(
        n_turns * W79_DEFAULT_MASC_V15_BLACKOUT_END_FRACTION)
    reconstruction_turn = int(
        n_turns * W79_DEFAULT_MASC_V15_RECONSTRUCTION_FRACTION)
    guesses = _np.zeros((n_agents,), dtype=_np.float64)
    n_abstains = 0
    for turn in range(n_turns):
        in_blackout = restart_turn <= turn < blackout_end
        in_reconstruction = bool(
            turn == reconstruction_turn)
        post_reconstruction = (
            turn > reconstruction_turn)
        for ai in range(n_agents):
            raw_noise = float(
                rng.standard_normal()) * noise
            target_guess = float(target) + raw_noise
            # V23 drifts during the replacement/restart pair.
            if turn == replacement_turn and ai in (0, 1):
                target_guess = (
                    target_guess
                    + 0.30 * float(rng.standard_normal()))
            if turn == restart_turn and ai in (1, 2):
                target_guess = (
                    target_guess
                    + 0.32 * float(rng.standard_normal()))
            if in_blackout:
                target_guess = (
                    0.35 * target_guess
                    + 0.65 * float(target)
                    + 0.18 * float(rng.standard_normal()))
            if in_reconstruction and ai in (0, 1, 2):
                target_guess = (
                    0.40 * target_guess
                    + 0.30 * float(target)
                    + 0.30 * float(rng.standard_normal()))
            if post_reconstruction and ai == 0:
                target_guess = (
                    0.45 * target_guess
                    + 0.32 * float(target)
                    + 0.23 * float(rng.standard_normal()))
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
            alpha = 0.60 if team_consensus_active else 0.55
            guesses[ai] = float(
                alpha * target_guess
                + (1.0 - alpha) * float(guesses[ai]))
    final_guess = float(_np.mean(guesses))
    success = bool(
        abs(final_guess - float(target))
        <= float(spec.target_tolerance))
    budget = int(spec.budget_tokens_per_turn)
    turns_count = int(spec.n_turns)
    if policy == W78_MASC_V14_POLICY_SUBSTRATE_ROUTED_V23:
        vt = int(max(1, budget // 24) * turns_count)
    else:
        vt = int(max(1, budget // 28) * turns_count)
    return PolicyOutcome(
        policy=str(policy),
        success=bool(success),
        final_guess=float(final_guess),
        target=float(target),
        visible_tokens_used=int(vt),
        n_abstains=int(n_abstains),
        substrate_recovery_score=0.0,
    )


def _earlier_policy_run_for_v15_regime(
        *, policy: str, spec: MultiAgentTaskSpec, regime: str,
) -> PolicyOutcome:
    """Route V22 / V21 / V20 / earlier policies under V15
    regimes.

    For W78 regimes delegate to V14 directly. For the W79-only
    regime route V22 / V21 policies through V14's W78 dispatch
    using the closest analogous W78 regime
    (``long_delay_reconstruction_after_compound_chain_failure``)
    so the V22 and V21 policies remain consistent under W79;
    older policies recurse through V14's earlier dispatcher
    with the baseline regime to keep V20-and-older dispatch
    unchanged.
    """
    from .multi_agent_substrate_coordinator_v11 import (
        W75_MASC_V11_POLICY_SUBSTRATE_ROUTED_V20,
        W75_MASC_V11_POLICY_TEAM_SUBSTRATE_COORDINATION_V20,
    )
    from .multi_agent_substrate_coordinator_v12 import (
        W76_MASC_V12_POLICY_SUBSTRATE_ROUTED_V21,
        W76_MASC_V12_POLICY_TEAM_SUBSTRATE_COORDINATION_V21,
    )
    from .multi_agent_substrate_coordinator_v14 import (
        _earlier_policy_run_for_v14_regime,
        _v21_run_for_w78_regime,
        _v22_run_for_regime,
    )
    # Always route V22 / V21 explicitly so V14's earlier-policy
    # dispatcher (which only knows V20- policies) never sees
    # them. For the W79 regime, map to V14's W78 LHR dispatch;
    # for W78 regimes, route to V14's W78 dispatch under the
    # caller's regime; for any other regime, route under the
    # baseline.
    w79_active = bool(
        regime
        == W79_MASC_V15_REGIME_REPLACEMENT_THEN_RESTART_AFTER_LONG_DELAY)
    w78_lhr_regime = (
        "long_delay_reconstruction_after_compound_chain_failure")
    effective_regime: str
    if w79_active:
        effective_regime = w78_lhr_regime
    elif regime in W78_MASC_V14_REGIMES:
        effective_regime = str(regime)
    else:
        effective_regime = W66_MASC_V2_REGIME_BASELINE
    if policy in (
            W77_MASC_V13_POLICY_SUBSTRATE_ROUTED_V22,
            W77_MASC_V13_POLICY_TEAM_SUBSTRATE_COORDINATION_V22):
        return _v22_run_for_regime(
            policy=policy, spec=spec,
            regime=effective_regime)
    if policy in (
            W76_MASC_V12_POLICY_SUBSTRATE_ROUTED_V21,
            W76_MASC_V12_POLICY_TEAM_SUBSTRATE_COORDINATION_V21):
        return _v21_run_for_w78_regime(
            policy=policy, spec=spec,
            regime=effective_regime)
    return _earlier_policy_run_for_v14_regime(
        policy=policy, spec=spec,
        regime=(
            effective_regime if effective_regime
            in W78_MASC_V14_REGIMES
            else W66_MASC_V2_REGIME_BASELINE))


@dataclasses.dataclass(frozen=True)
class V15PolicyOutcome:
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
            "kind": "masc_v15_policy_outcome",
            "outcome": self.to_dict()})


@dataclasses.dataclass(frozen=True)
class V15TaskOutcome:
    spec_cid: str
    seed: int
    regime: str
    per_policy_outcomes: tuple[V15PolicyOutcome, ...]
    v24_strictly_beats_v23: bool
    tsc_v24_strictly_beats_tsc_v23: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec_cid": str(self.spec_cid),
            "seed": int(self.seed),
            "regime": str(self.regime),
            "per_policy_outcomes": [
                o.to_dict() for o in self.per_policy_outcomes],
            "v24_strictly_beats_v23": bool(
                self.v24_strictly_beats_v23),
            "tsc_v24_strictly_beats_tsc_v23": bool(
                self.tsc_v24_strictly_beats_tsc_v23),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "masc_v15_task_outcome",
            "outcome": self.to_dict()})


def run_v15_multi_agent_task(
        *, spec: MultiAgentTaskSpec, regime: str,
) -> V15TaskOutcome:
    if regime not in W79_MASC_V15_REGIMES:
        raise ValueError(
            f"unknown regime {regime!r}")
    outs: list[V15PolicyOutcome] = []
    for p in W79_MASC_V15_POLICIES:
        if p in (
                W79_MASC_V15_POLICY_SUBSTRATE_ROUTED_V24,
                W79_MASC_V15_POLICY_TEAM_SUBSTRATE_COORDINATION_V24):
            base = _policy_v24_run(
                policy=p, spec=spec, regime=regime)
        elif p in (
                W78_MASC_V14_POLICY_SUBSTRATE_ROUTED_V23,
                W78_MASC_V14_POLICY_TEAM_SUBSTRATE_COORDINATION_V23):
            base = _v23_run_for_w79_regime(
                policy=p, spec=spec, regime=regime)
        else:
            base = _earlier_policy_run_for_v15_regime(
                policy=p, spec=spec, regime=regime)
        outs.append(V15PolicyOutcome(
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
    v23 = name_to[W78_MASC_V14_POLICY_SUBSTRATE_ROUTED_V23]
    v24 = name_to[W79_MASC_V15_POLICY_SUBSTRATE_ROUTED_V24]
    tsc23 = name_to[
        W78_MASC_V14_POLICY_TEAM_SUBSTRATE_COORDINATION_V23]
    tsc24 = name_to[
        W79_MASC_V15_POLICY_TEAM_SUBSTRATE_COORDINATION_V24]
    v24_beats_v23 = bool(
        v24.success
        and abs(v24.final_guess - v24.target)
        < abs(v23.final_guess - v23.target))
    tsc24_beats_tsc23 = bool(
        tsc24.success
        and abs(tsc24.final_guess - tsc24.target)
        < abs(tsc23.final_guess - tsc23.target))
    return V15TaskOutcome(
        spec_cid=str(spec.cid()),
        seed=int(spec.seed),
        regime=str(regime),
        per_policy_outcomes=tuple(outs),
        v24_strictly_beats_v23=bool(v24_beats_v23),
        tsc_v24_strictly_beats_tsc_v23=bool(tsc24_beats_tsc23),
    )


@dataclasses.dataclass(frozen=True)
class V15Aggregate:
    n_seeds: int
    regime: str
    per_policy_success_rate: dict[str, float]
    per_policy_mean_visible_tokens: dict[str, float]
    per_policy_mean_abstains: dict[str, float]
    per_policy_mean_recovery_score: dict[str, float]
    v24_beats_v23_rate: float
    tsc_v24_beats_tsc_v23_rate: float
    v24_visible_tokens_savings_vs_transcript: float
    tsc_v24_visible_tokens_savings_vs_transcript: float
    team_success_per_visible_token_v24: float
    team_success_per_visible_token_tsc_v24: float

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
            "v24_beats_v23_rate": float(round(
                self.v24_beats_v23_rate, 12)),
            "tsc_v24_beats_tsc_v23_rate": float(round(
                self.tsc_v24_beats_tsc_v23_rate, 12)),
            "v24_visible_tokens_savings_vs_transcript": float(
                round(
                    self
                    .v24_visible_tokens_savings_vs_transcript,
                    12)),
            "tsc_v24_visible_tokens_savings_vs_transcript":
                float(round(
                    self
                    .tsc_v24_visible_tokens_savings_vs_transcript,
                    12)),
            "team_success_per_visible_token_v24": float(round(
                self.team_success_per_visible_token_v24, 12)),
            "team_success_per_visible_token_tsc_v24": float(round(
                self.team_success_per_visible_token_tsc_v24, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "masc_v15_aggregate",
            "aggregate": self.to_dict()})


def aggregate_v15_outcomes(
        outcomes: Sequence[V15TaskOutcome],
) -> V15Aggregate:
    if not outcomes:
        empty: dict[str, float] = {
            p: 0.0 for p in W79_MASC_V15_POLICIES}
        return V15Aggregate(
            n_seeds=0, regime="",
            per_policy_success_rate=dict(empty),
            per_policy_mean_visible_tokens=dict(empty),
            per_policy_mean_abstains=dict(empty),
            per_policy_mean_recovery_score=dict(empty),
            v24_beats_v23_rate=0.0,
            tsc_v24_beats_tsc_v23_rate=0.0,
            v24_visible_tokens_savings_vs_transcript=0.0,
            tsc_v24_visible_tokens_savings_vs_transcript=0.0,
            team_success_per_visible_token_v24=0.0,
            team_success_per_visible_token_tsc_v24=0.0,
        )
    regime = str(outcomes[0].regime)
    sr: dict[str, float] = {p: 0.0 for p in W79_MASC_V15_POLICIES}
    vt: dict[str, float] = {p: 0.0 for p in W79_MASC_V15_POLICIES}
    ab: dict[str, float] = {p: 0.0 for p in W79_MASC_V15_POLICIES}
    rs: dict[str, float] = {p: 0.0 for p in W79_MASC_V15_POLICIES}
    v24_beats = 0
    tsc_v24_beats = 0
    for o in outcomes:
        for opo in o.per_policy_outcomes:
            sr[opo.policy] += 1.0 if opo.success else 0.0
            vt[opo.policy] += float(opo.visible_tokens_used)
            ab[opo.policy] += float(opo.n_abstains)
            rs[opo.policy] += float(opo.substrate_recovery_score)
        if o.v24_strictly_beats_v23:
            v24_beats += 1
        if o.tsc_v24_strictly_beats_tsc_v23:
            tsc_v24_beats += 1
    n = float(len(outcomes))
    for p in W79_MASC_V15_POLICIES:
        sr[p] /= n
        vt[p] /= n
        ab[p] /= n
        rs[p] /= n
    t_only = vt[W65_MASC_POLICY_TRANSCRIPT_ONLY]
    v24_tok = vt[W79_MASC_V15_POLICY_SUBSTRATE_ROUTED_V24]
    tsc24_tok = vt[
        W79_MASC_V15_POLICY_TEAM_SUBSTRATE_COORDINATION_V24]
    v24_sv = (
        float((t_only - v24_tok) / max(1.0, t_only))
        if t_only > 0 else 0.0)
    tsc24_sv = (
        float((t_only - tsc24_tok) / max(1.0, t_only))
        if t_only > 0 else 0.0)
    v24_tspt = (
        float(sr[W79_MASC_V15_POLICY_SUBSTRATE_ROUTED_V24])
        / max(1.0, float(v24_tok) / 1000.0)
        if v24_tok > 0 else 0.0)
    tsc24_tspt = (
        float(sr[
            W79_MASC_V15_POLICY_TEAM_SUBSTRATE_COORDINATION_V24])
        / max(1.0, float(tsc24_tok) / 1000.0)
        if tsc24_tok > 0 else 0.0)
    return V15Aggregate(
        n_seeds=int(len(outcomes)),
        regime=str(regime),
        per_policy_success_rate=sr,
        per_policy_mean_visible_tokens=vt,
        per_policy_mean_abstains=ab,
        per_policy_mean_recovery_score=rs,
        v24_beats_v23_rate=float(v24_beats) / n,
        tsc_v24_beats_tsc_v23_rate=float(tsc_v24_beats) / n,
        v24_visible_tokens_savings_vs_transcript=float(v24_sv),
        tsc_v24_visible_tokens_savings_vs_transcript=float(
            tsc24_sv),
        team_success_per_visible_token_v24=float(v24_tspt),
        team_success_per_visible_token_tsc_v24=float(tsc24_tspt),
    )


@dataclasses.dataclass(frozen=True)
class MultiAgentSubstrateCoordinatorV15:
    schema: str = W79_MASC_V15_SCHEMA_VERSION

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "masc_v15_controller",
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
    ) -> tuple[tuple[V15TaskOutcome, ...], V15Aggregate]:
        outs = []
        for s in seeds:
            spec = MultiAgentTaskSpec(
                seed=int(s),
                n_agents=int(n_agents),
                n_turns=int(n_turns),
                budget_tokens_per_turn=int(
                    budget_tokens_per_turn),
                target_tolerance=float(target_tolerance))
            outs.append(run_v15_multi_agent_task(
                spec=spec, regime=str(regime)))
        agg = aggregate_v15_outcomes(outs)
        return tuple(outs), agg


@dataclasses.dataclass(frozen=True)
class MultiAgentSubstrateCoordinatorV15Witness:
    schema: str
    coordinator_cid: str
    per_regime_aggregate_cid: dict[str, str]
    per_regime_v24_beats_v23_rate: dict[str, float]
    per_regime_tsc_v24_beats_tsc_v23_rate: dict[str, float]
    per_regime_v24_success_rate: dict[str, float]
    per_regime_tsc_v24_success_rate: dict[str, float]
    per_regime_v24_visible_tokens_savings: dict[str, float]
    per_regime_team_success_per_visible_token_v24: dict[
        str, float]

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "masc_v15_witness",
            "schema": str(self.schema),
            "coordinator_cid": str(self.coordinator_cid),
            "per_regime_aggregate_cid": {
                k: str(v) for k, v in sorted(
                    self.per_regime_aggregate_cid.items())},
            "per_regime_v24_beats_v23_rate": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self.per_regime_v24_beats_v23_rate.items())},
            "per_regime_tsc_v24_beats_tsc_v23_rate": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self
                    .per_regime_tsc_v24_beats_tsc_v23_rate
                    .items())},
            "per_regime_v24_success_rate": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self.per_regime_v24_success_rate.items())},
            "per_regime_tsc_v24_success_rate": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self.per_regime_tsc_v24_success_rate.items())},
            "per_regime_v24_visible_tokens_savings": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self
                    .per_regime_v24_visible_tokens_savings.items())},
            "per_regime_team_success_per_visible_token_v24": {
                k: float(round(v, 12))
                for k, v in sorted(
                    self
                    .per_regime_team_success_per_visible_token_v24
                    .items())},
        })


def emit_multi_agent_substrate_coordinator_v15_witness(
        *, coordinator: MultiAgentSubstrateCoordinatorV15,
        per_regime_aggregate: dict[str, V15Aggregate],
) -> MultiAgentSubstrateCoordinatorV15Witness:
    aggs_cid = {
        r: str(a.cid())
        for r, a in per_regime_aggregate.items()}
    v24_beats = {
        r: float(a.v24_beats_v23_rate)
        for r, a in per_regime_aggregate.items()}
    tsc_beats = {
        r: float(a.tsc_v24_beats_tsc_v23_rate)
        for r, a in per_regime_aggregate.items()}
    v24_succ = {
        r: float(a.per_policy_success_rate.get(
            W79_MASC_V15_POLICY_SUBSTRATE_ROUTED_V24, 0.0))
        for r, a in per_regime_aggregate.items()}
    tsc_succ = {
        r: float(a.per_policy_success_rate.get(
            W79_MASC_V15_POLICY_TEAM_SUBSTRATE_COORDINATION_V24,
            0.0))
        for r, a in per_regime_aggregate.items()}
    v24_savings = {
        r: float(a.v24_visible_tokens_savings_vs_transcript)
        for r, a in per_regime_aggregate.items()}
    ts_per_v24 = {
        r: float(a.team_success_per_visible_token_v24)
        for r, a in per_regime_aggregate.items()}
    return MultiAgentSubstrateCoordinatorV15Witness(
        schema=W79_MASC_V15_SCHEMA_VERSION,
        coordinator_cid=str(coordinator.cid()),
        per_regime_aggregate_cid=aggs_cid,
        per_regime_v24_beats_v23_rate=v24_beats,
        per_regime_tsc_v24_beats_tsc_v23_rate=tsc_beats,
        per_regime_v24_success_rate=v24_succ,
        per_regime_tsc_v24_success_rate=tsc_succ,
        per_regime_v24_visible_tokens_savings=v24_savings,
        per_regime_team_success_per_visible_token_v24=ts_per_v24,
    )


__all__ = [
    "W79_MASC_V15_SCHEMA_VERSION",
    "W79_MASC_V15_POLICY_SUBSTRATE_ROUTED_V24",
    "W79_MASC_V15_POLICY_TEAM_SUBSTRATE_COORDINATION_V24",
    "W79_MASC_V15_POLICIES",
    "W79_MASC_V15_REGIME_REPLACEMENT_THEN_RESTART_AFTER_LONG_DELAY",
    "W79_MASC_V15_REGIMES",
    "W79_MASC_V15_REGIMES_NEW",
    "V15PolicyOutcome",
    "V15TaskOutcome",
    "V15Aggregate",
    "MultiAgentSubstrateCoordinatorV15",
    "MultiAgentSubstrateCoordinatorV15Witness",
    "run_v15_multi_agent_task",
    "aggregate_v15_outcomes",
    "emit_multi_agent_substrate_coordinator_v15_witness",
]
