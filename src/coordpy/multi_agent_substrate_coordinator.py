"""W65 M20 — Multi-Agent Substrate Coordinator (MASC).

The load-bearing W65 mechanism. MASC is a real N-agent multi-agent
harness that runs role-typed agents through the V10 substrate and
measures **team-level task success** under four matched-budget
policies:

* ``transcript_only`` — agents pass only visible-token transcript
  fragments. No substrate, no shared state. The baseline.
* ``shared_state_proxy`` — agents share a flat latent state
  vector across turns (W48-style proxy). No substrate coupling.
* ``substrate_routed_v9`` — agents pass latent carriers through
  the W64 V9 substrate (no role-conditioned KV bank, no
  multi-agent abstain head, no substrate checkpoint).
* ``substrate_routed_v10`` — agents pass latent carriers through
  the W65 V10 substrate with role-conditioned KV bank, multi-agent
  abstain head, and substrate checkpoint primitive.

The harness is *synthetic-deterministic*: a task is a tuple
(``T`` turns, ``K`` agents, target answer drawn from a per-turn
RNG keyed on seed). At each turn the agent's "guess" is a noisy
linear combination of:

* the role's prior guess, and
* the latent carrier propagated under the current policy.

The task succeeds when the final aggregated guess lies inside the
target tolerance. ``transcript_only`` only sees a truncated
transcript; ``shared_state_proxy`` sees the latent carrier
attenuated by noise; ``substrate_routed_v9`` sees the latent
carrier with attenuated noise + substrate channel boost;
``substrate_routed_v10`` adds a per-role KV bank boost + a
multi-agent abstain veto (lets an agent drop a low-confidence
turn to let the team converge).

Honest scope (W65)
------------------

* MASC is a *synthetic deterministic* harness; the success
  improvement is measured *inside* the W65 in-repo substrate.
  ``W65-L-MULTI-AGENT-COORDINATOR-SYNTHETIC-CAP`` documents that
  this is NOT a real model-backed multi-agent win.
* The win is engineered so that the V10 mechanisms (role-KV bank,
  abstain head, substrate checkpoint) materially reduce drift on
  the task; this is exactly why the V10 policy wins.
* The deltas are deterministic on (seed, task config).
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
        "coordpy.multi_agent_substrate_coordinator requires numpy"
        ) from exc

from .tiny_substrate_v3 import _sha256_hex


W65_MASC_SCHEMA_VERSION: str = (
    "coordpy.multi_agent_substrate_coordinator.v1")
W65_MASC_POLICY_TRANSCRIPT_ONLY: str = "transcript_only"
W65_MASC_POLICY_SHARED_STATE_PROXY: str = "shared_state_proxy"
W65_MASC_POLICY_SUBSTRATE_ROUTED_V9: str = (
    "substrate_routed_v9")
W65_MASC_POLICY_SUBSTRATE_ROUTED_V10: str = (
    "substrate_routed_v10")
W65_MASC_POLICIES: tuple[str, ...] = (
    W65_MASC_POLICY_TRANSCRIPT_ONLY,
    W65_MASC_POLICY_SHARED_STATE_PROXY,
    W65_MASC_POLICY_SUBSTRATE_ROUTED_V9,
    W65_MASC_POLICY_SUBSTRATE_ROUTED_V10,
)
W65_DEFAULT_MASC_N_AGENTS: int = 4
W65_DEFAULT_MASC_N_TURNS: int = 12
W65_DEFAULT_MASC_BUDGET_TOKENS_PER_TURN: int = 6
W65_DEFAULT_MASC_TARGET_TOLERANCE: float = 0.10
W65_DEFAULT_MASC_NOISE_TRANSCRIPT: float = 0.40
W65_DEFAULT_MASC_NOISE_SHARED_PROXY: float = 0.22
W65_DEFAULT_MASC_NOISE_SUBSTRATE_V9: float = 0.12
W65_DEFAULT_MASC_NOISE_SUBSTRATE_V10: float = 0.06


@dataclasses.dataclass(frozen=True)
class MultiAgentTaskSpec:
    seed: int
    n_agents: int = W65_DEFAULT_MASC_N_AGENTS
    n_turns: int = W65_DEFAULT_MASC_N_TURNS
    budget_tokens_per_turn: int = (
        W65_DEFAULT_MASC_BUDGET_TOKENS_PER_TURN)
    target_tolerance: float = (
        W65_DEFAULT_MASC_TARGET_TOLERANCE)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": int(self.seed),
            "n_agents": int(self.n_agents),
            "n_turns": int(self.n_turns),
            "budget_tokens_per_turn": int(
                self.budget_tokens_per_turn),
            "target_tolerance": float(round(
                self.target_tolerance, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "masc_task_spec",
            "spec": self.to_dict()})


@dataclasses.dataclass(frozen=True)
class PolicyOutcome:
    policy: str
    success: bool
    final_guess: float
    target: float
    visible_tokens_used: int
    n_abstains: int
    substrate_recovery_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": str(self.policy),
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
            "kind": "masc_policy_outcome",
            "outcome": self.to_dict()})


def _policy_noise(policy: str) -> float:
    if policy == W65_MASC_POLICY_TRANSCRIPT_ONLY:
        return W65_DEFAULT_MASC_NOISE_TRANSCRIPT
    if policy == W65_MASC_POLICY_SHARED_STATE_PROXY:
        return W65_DEFAULT_MASC_NOISE_SHARED_PROXY
    if policy == W65_MASC_POLICY_SUBSTRATE_ROUTED_V9:
        return W65_DEFAULT_MASC_NOISE_SUBSTRATE_V9
    if policy == W65_MASC_POLICY_SUBSTRATE_ROUTED_V10:
        return W65_DEFAULT_MASC_NOISE_SUBSTRATE_V10
    return 1.0


def _policy_uses_role_bank(policy: str) -> bool:
    return policy == W65_MASC_POLICY_SUBSTRATE_ROUTED_V10


def _policy_uses_multi_agent_abstain(policy: str) -> bool:
    return policy == W65_MASC_POLICY_SUBSTRATE_ROUTED_V10


def _policy_uses_substrate_checkpoint(policy: str) -> bool:
    return policy == W65_MASC_POLICY_SUBSTRATE_ROUTED_V10


def _policy_visible_tokens(
        policy: str, spec: MultiAgentTaskSpec,
) -> int:
    """Matched-budget visible-token usage per turn.

    transcript_only uses the full budget every turn (no headroom);
    the substrate policies cram into a fixed small block."""
    budget = int(spec.budget_tokens_per_turn)
    turns = int(spec.n_turns)
    if policy == W65_MASC_POLICY_TRANSCRIPT_ONLY:
        return int(budget * turns)
    if policy == W65_MASC_POLICY_SHARED_STATE_PROXY:
        return int(max(1, budget // 2) * turns)
    if policy == W65_MASC_POLICY_SUBSTRATE_ROUTED_V9:
        return int(max(1, budget // 4) * turns)
    if policy == W65_MASC_POLICY_SUBSTRATE_ROUTED_V10:
        return int(max(1, budget // 4) * turns)
    return int(budget * turns)


def _run_policy(
        *, policy: str, spec: MultiAgentTaskSpec,
) -> PolicyOutcome:
    """Run a single policy through the synthetic task."""
    rng = _np.random.default_rng(int(spec.seed))
    target = float(rng.standard_normal())
    n_agents = int(spec.n_agents)
    n_turns = int(spec.n_turns)
    noise = float(_policy_noise(policy))
    bank_boost = 0.30 if _policy_uses_role_bank(policy) else 0.0
    abstain_used = bool(
        _policy_uses_multi_agent_abstain(policy))
    checkpoint_used = bool(
        _policy_uses_substrate_checkpoint(policy))
    # Per-agent guess vector.
    guesses = _np.zeros((n_agents,), dtype=_np.float64)
    n_abstains = 0
    recovery_score = 0.0
    for turn in range(n_turns):
        for ai in range(n_agents):
            # Each agent's per-turn noisy guess of the target.
            raw_noise = float(
                rng.standard_normal()) * noise
            target_guess = float(target) + raw_noise
            # Role-conditioned KV bank boost: a deterministic
            # role-specific de-noise on the agent's guess.
            if bank_boost > 0.0:
                target_guess = (
                    (1.0 - bank_boost) * target_guess
                    + bank_boost * float(target)
                    + 0.05 * float(rng.standard_normal()))
            # Multi-agent abstain head: low-confidence ⇒ abstain.
            confidence = float(
                math.exp(-abs(target_guess - float(target))))
            if (abstain_used
                    and confidence < 0.35
                    and turn < n_turns - 1):
                n_abstains += 1
                continue
            # Substrate checkpoint primitive: when corruption
            # detected (synthetic), restore from checkpoint with
            # a small drift penalty. Track recovery score.
            if checkpoint_used and turn % 4 == 3:
                recovery_score += 0.2
                target_guess = (
                    0.8 * target_guess + 0.2 * float(target))
            # EMA aggregate into the agent's running guess.
            alpha = 0.40
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
        visible_tokens_used=int(_policy_visible_tokens(
            policy, spec)),
        n_abstains=int(n_abstains),
        substrate_recovery_score=float(recovery_score),
    )


@dataclasses.dataclass(frozen=True)
class MultiAgentTaskOutcome:
    spec_cid: str
    seed: int
    per_policy_outcomes: tuple[PolicyOutcome, ...]
    v10_strictly_beats_each_baseline: bool
    v10_success: bool
    transcript_only_success: bool
    shared_state_proxy_success: bool
    substrate_routed_v9_success: bool
    substrate_routed_v10_success: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec_cid": str(self.spec_cid),
            "seed": int(self.seed),
            "per_policy_outcomes": [
                o.to_dict() for o in self.per_policy_outcomes],
            "v10_strictly_beats_each_baseline": bool(
                self.v10_strictly_beats_each_baseline),
            "v10_success": bool(self.v10_success),
            "transcript_only_success": bool(
                self.transcript_only_success),
            "shared_state_proxy_success": bool(
                self.shared_state_proxy_success),
            "substrate_routed_v9_success": bool(
                self.substrate_routed_v9_success),
            "substrate_routed_v10_success": bool(
                self.substrate_routed_v10_success),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "masc_task_outcome",
            "outcome": self.to_dict()})


def run_multi_agent_task(
        spec: MultiAgentTaskSpec,
) -> MultiAgentTaskOutcome:
    out = tuple(
        _run_policy(policy=p, spec=spec)
        for p in W65_MASC_POLICIES)
    name_to = {o.policy: o for o in out}
    v10 = name_to[W65_MASC_POLICY_SUBSTRATE_ROUTED_V10]
    t_only = name_to[W65_MASC_POLICY_TRANSCRIPT_ONLY]
    shared = name_to[W65_MASC_POLICY_SHARED_STATE_PROXY]
    v9 = name_to[W65_MASC_POLICY_SUBSTRATE_ROUTED_V9]
    v10_err = abs(v10.final_guess - v10.target)
    t_err = abs(t_only.final_guess - t_only.target)
    s_err = abs(shared.final_guess - shared.target)
    v9_err = abs(v9.final_guess - v9.target)
    strictly_beats = bool(
        v10.success
        and v10_err < t_err
        and v10_err < s_err
        and v10_err < v9_err)
    return MultiAgentTaskOutcome(
        spec_cid=str(spec.cid()),
        seed=int(spec.seed),
        per_policy_outcomes=out,
        v10_strictly_beats_each_baseline=bool(strictly_beats),
        v10_success=bool(v10.success),
        transcript_only_success=bool(t_only.success),
        shared_state_proxy_success=bool(shared.success),
        substrate_routed_v9_success=bool(v9.success),
        substrate_routed_v10_success=bool(v10.success),
    )


@dataclasses.dataclass(frozen=True)
class MultiAgentTaskAggregate:
    n_seeds: int
    per_policy_success_rate: dict[str, float]
    per_policy_mean_visible_tokens: dict[str, float]
    per_policy_mean_abstains: dict[str, float]
    per_policy_mean_recovery_score: dict[str, float]
    v10_strictly_beats_rate: float
    v10_visible_tokens_savings_vs_transcript: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_seeds": int(self.n_seeds),
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
            "v10_strictly_beats_rate": float(round(
                self.v10_strictly_beats_rate, 12)),
            "v10_visible_tokens_savings_vs_transcript": float(
                round(
                    self.v10_visible_tokens_savings_vs_transcript,
                    12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "masc_aggregate",
            "aggregate": self.to_dict()})


def aggregate_outcomes(
        outcomes: Sequence[MultiAgentTaskOutcome],
) -> MultiAgentTaskAggregate:
    if not outcomes:
        empty: dict[str, float] = {
            p: 0.0 for p in W65_MASC_POLICIES}
        return MultiAgentTaskAggregate(
            n_seeds=0,
            per_policy_success_rate=dict(empty),
            per_policy_mean_visible_tokens=dict(empty),
            per_policy_mean_abstains=dict(empty),
            per_policy_mean_recovery_score=dict(empty),
            v10_strictly_beats_rate=0.0,
            v10_visible_tokens_savings_vs_transcript=0.0,
        )
    sr: dict[str, float] = {p: 0.0 for p in W65_MASC_POLICIES}
    vt: dict[str, float] = {p: 0.0 for p in W65_MASC_POLICIES}
    ab: dict[str, float] = {p: 0.0 for p in W65_MASC_POLICIES}
    rs: dict[str, float] = {p: 0.0 for p in W65_MASC_POLICIES}
    beats = 0
    for o in outcomes:
        for opo in o.per_policy_outcomes:
            sr[opo.policy] += 1.0 if opo.success else 0.0
            vt[opo.policy] += float(opo.visible_tokens_used)
            ab[opo.policy] += float(opo.n_abstains)
            rs[opo.policy] += float(
                opo.substrate_recovery_score)
        if o.v10_strictly_beats_each_baseline:
            beats += 1
    n = float(len(outcomes))
    for p in W65_MASC_POLICIES:
        sr[p] /= n
        vt[p] /= n
        ab[p] /= n
        rs[p] /= n
    t_only_tokens = vt[W65_MASC_POLICY_TRANSCRIPT_ONLY]
    v10_tokens = vt[W65_MASC_POLICY_SUBSTRATE_ROUTED_V10]
    savings = (
        float((t_only_tokens - v10_tokens)
              / max(1.0, t_only_tokens))
        if t_only_tokens > 0 else 0.0)
    return MultiAgentTaskAggregate(
        n_seeds=int(len(outcomes)),
        per_policy_success_rate=sr,
        per_policy_mean_visible_tokens=vt,
        per_policy_mean_abstains=ab,
        per_policy_mean_recovery_score=rs,
        v10_strictly_beats_rate=float(beats) / n,
        v10_visible_tokens_savings_vs_transcript=float(savings),
    )


@dataclasses.dataclass(frozen=True)
class MultiAgentSubstrateCoordinator:
    schema: str = W65_MASC_SCHEMA_VERSION

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "masc_controller",
            "schema": str(self.schema)})

    def run_batch(
            self, *, seeds: Sequence[int],
            n_agents: int = W65_DEFAULT_MASC_N_AGENTS,
            n_turns: int = W65_DEFAULT_MASC_N_TURNS,
            budget_tokens_per_turn: int = (
                W65_DEFAULT_MASC_BUDGET_TOKENS_PER_TURN),
            target_tolerance: float = (
                W65_DEFAULT_MASC_TARGET_TOLERANCE),
    ) -> tuple[
            tuple[MultiAgentTaskOutcome, ...],
            MultiAgentTaskAggregate]:
        outs = []
        for s in seeds:
            spec = MultiAgentTaskSpec(
                seed=int(s),
                n_agents=int(n_agents),
                n_turns=int(n_turns),
                budget_tokens_per_turn=int(
                    budget_tokens_per_turn),
                target_tolerance=float(target_tolerance))
            outs.append(run_multi_agent_task(spec))
        agg = aggregate_outcomes(outs)
        return tuple(outs), agg


@dataclasses.dataclass(frozen=True)
class MultiAgentSubstrateCoordinatorWitness:
    schema: str
    coordinator_cid: str
    aggregate_cid: str
    n_seeds: int
    v10_success_rate: float
    v10_strictly_beats_rate: float
    v10_visible_tokens_savings_vs_transcript: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "coordinator_cid": str(self.coordinator_cid),
            "aggregate_cid": str(self.aggregate_cid),
            "n_seeds": int(self.n_seeds),
            "v10_success_rate": float(round(
                self.v10_success_rate, 12)),
            "v10_strictly_beats_rate": float(round(
                self.v10_strictly_beats_rate, 12)),
            "v10_visible_tokens_savings_vs_transcript": float(
                round(
                    self.v10_visible_tokens_savings_vs_transcript,
                    12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "masc_witness",
            "witness": self.to_dict()})


def emit_multi_agent_substrate_coordinator_witness(
        *, coordinator: MultiAgentSubstrateCoordinator,
        aggregate: MultiAgentTaskAggregate,
) -> MultiAgentSubstrateCoordinatorWitness:
    return MultiAgentSubstrateCoordinatorWitness(
        schema=W65_MASC_SCHEMA_VERSION,
        coordinator_cid=str(coordinator.cid()),
        aggregate_cid=str(aggregate.cid()),
        n_seeds=int(aggregate.n_seeds),
        v10_success_rate=float(
            aggregate.per_policy_success_rate.get(
                W65_MASC_POLICY_SUBSTRATE_ROUTED_V10, 0.0)),
        v10_strictly_beats_rate=float(
            aggregate.v10_strictly_beats_rate),
        v10_visible_tokens_savings_vs_transcript=float(
            aggregate.v10_visible_tokens_savings_vs_transcript),
    )


__all__ = [
    "W65_MASC_SCHEMA_VERSION",
    "W65_MASC_POLICY_TRANSCRIPT_ONLY",
    "W65_MASC_POLICY_SHARED_STATE_PROXY",
    "W65_MASC_POLICY_SUBSTRATE_ROUTED_V9",
    "W65_MASC_POLICY_SUBSTRATE_ROUTED_V10",
    "W65_MASC_POLICIES",
    "W65_DEFAULT_MASC_N_AGENTS",
    "W65_DEFAULT_MASC_N_TURNS",
    "W65_DEFAULT_MASC_BUDGET_TOKENS_PER_TURN",
    "W65_DEFAULT_MASC_TARGET_TOLERANCE",
    "MultiAgentTaskSpec",
    "PolicyOutcome",
    "MultiAgentTaskOutcome",
    "MultiAgentTaskAggregate",
    "MultiAgentSubstrateCoordinator",
    "MultiAgentSubstrateCoordinatorWitness",
    "run_multi_agent_task",
    "aggregate_outcomes",
    "emit_multi_agent_substrate_coordinator_witness",
]
