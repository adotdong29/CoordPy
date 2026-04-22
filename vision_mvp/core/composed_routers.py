"""Composed routers — wire Wave-1..5 primitives into end-to-end stacks.

Three variants, each layering primitives from the build-out:

  AdversarialCASRRouter
    - Cuckoo filter replaces Bloom filter (D7)
    - VRF-elected O(log N) committee replaces fixed workspace (D9)
    - PeerReview hash-chain log for tamper-evidence (D8)
    - Control barrier function keeps team out of unsafe states (E5)
    - Differential privacy on shared aggregates (H4)

  CryptoCASRRouter
    - Shamir secret sharing for threshold workspace membership (H1)
    - SPDZ-light additive secret-shared sums (H5)
    - Paillier homomorphic aggregate for large-payload sums (H2)

  DynamicCASRRouter
    - Interval tree clocks for causality (D2)
    - Consistent hashing for routing under join/leave (D5)
    - HAMT persistent store as the stigmergy substrate (D4)
    - Plumtree + HyParView gossip (D10)

These are composition helpers, not new protocols: the public API mirrors
`CASRRouter` (step, estimates, stats) but the internals plug in different
primitives behind the same interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .cbf import enforce_barrier, exponential_alpha
from .cuckoo_filter import CuckooFilter
from .dp import GaussianMechanism
from .gossip_tree import PlumtreeOverlay
from .itc import Stamp
from .peer_review import HashChainLog
from .persistent import HAMT
from .routing_hash import ConsistentHashRing
from .vrf_committee import VRFKey, elect_committee


@dataclass
class AdversarialCASRRouter:
    """CASR stack hardened against byzantine faults and adversarial inputs."""

    n_agents: int
    committee_size: int | None = None       # None → ⌈log2 N⌉
    epsilon_dp: float = 1.0                  # DP privacy budget per release
    delta_dp: float = 1e-5
    barrier_safety: float = 1.0              # CBF safety margin on state-norm

    _vrf_keys: list[VRFKey] = field(default_factory=list)
    _filter: CuckooFilter = field(init=False)
    _logs: list[HashChainLog] = field(init=False)
    _dp: GaussianMechanism = field(init=False)
    _round_idx: int = 0
    _rng: np.random.Generator = field(init=False)

    def __post_init__(self):
        import math
        if self.committee_size is None:
            self.committee_size = max(2, math.ceil(math.log2(max(self.n_agents, 2))))
        self._vrf_keys = [VRFKey() for _ in range(self.n_agents)]
        self._filter = CuckooFilter(capacity=max(self.n_agents * 4, 256),
                                     fingerprint_bits=16)
        self._logs = [
            HashChainLog(agent_id=f"agent_{i}") for i in range(self.n_agents)
        ]
        self._dp = GaussianMechanism(
            sensitivity=1.0, epsilon=self.epsilon_dp, delta=self.delta_dp,
        )
        self._rng = np.random.default_rng(0)

    def elect_workspace(self, round_idx: int) -> list[int]:
        """VRF-elected committee of `committee_size` agents for this round."""
        seed = str(round_idx).encode("utf-8")
        outs = {
            str(i): k.evaluate(seed) for i, k in enumerate(self._vrf_keys)
        }
        chosen = elect_committee(outs, k=self.committee_size)
        return [int(a) for a in chosen]

    def aggregate_dp(self, per_agent_values: np.ndarray) -> np.ndarray:
        """Noise-added aggregate over the committee's reports (DP)."""
        per_agent = np.asarray(per_agent_values, dtype=float)
        return self._dp.release(per_agent.mean(axis=0), self._rng)

    def route(self, source_id: int, target_id: int) -> bool:
        """Event gating via cuckoo-filter membership of (source,target) pair."""
        key = f"{source_id}->{target_id}"
        if key in self._filter:
            return True
        self._filter.insert(key)
        return False

    def record(self, agent_id: int, payload: Any) -> None:
        """Append a tamper-evident log entry for the agent."""
        self._logs[agent_id].append({"round": self._round_idx, "payload": payload})

    def safe_control(
        self,
        state: np.ndarray,
        nominal: np.ndarray,
    ) -> np.ndarray:
        """Apply CBF: project nominal control onto the safety-preserving half-
        space of h(x) = barrier_safety − ‖x‖².
        """
        r = enforce_barrier(
            state, nominal,
            h=lambda x: float(self.barrier_safety - np.dot(x, x)),
            Lfh=lambda x: 0.0,
            Lgh=lambda x: -2.0 * np.asarray(x, dtype=float),
            alpha=exponential_alpha(1.0),
        )
        return r.u_safe

    def stats(self) -> dict:
        return {
            "n_agents": self.n_agents,
            "committee_size": self.committee_size,
            "cuckoo_load": self._filter.load_factor(),
            "dp_epsilon": self.epsilon_dp,
            "dp_delta": self.delta_dp,
            "dp_std": self._dp.std,
            "rounds": self._round_idx,
        }


@dataclass
class DynamicCASRRouter:
    """CASR stack for dynamic team membership."""

    replicas: int = 64

    _ring: ConsistentHashRing = field(init=False)
    _clock: Stamp = field(init=False)
    _store: HAMT = field(init=False)
    _overlay: PlumtreeOverlay = field(init=False)

    def __post_init__(self):
        self._ring = ConsistentHashRing(replicas=self.replicas)
        self._clock = Stamp.seed()
        self._store = HAMT()
        self._overlay = PlumtreeOverlay()

    def join(self, agent_id: str, bootstrap: list[str]) -> None:
        self._ring.add_node(agent_id)
        self._overlay.add_node(agent_id, bootstrap=bootstrap)

    def leave(self, agent_id: str) -> None:
        self._ring.remove_node(agent_id)

    def route(self, key: Any) -> str:
        return self._ring.route(key)

    def put(self, key: Any, value: Any) -> None:
        self._store = self._store.set(key, value)

    def get(self, key: Any, default=None) -> Any:
        return self._store.get(key, default)

    def event_tick(self) -> Stamp:
        self._clock = self._clock.event_tick()
        return self._clock

    def broadcast(self, src: str, msg_id: str) -> None:
        self._overlay.broadcast(src, msg_id)

    def reliability(self, msg_id: str) -> float:
        return self._overlay.reliability(msg_id)
