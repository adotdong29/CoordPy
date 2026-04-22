"""Pre-shared randomness — VISION_MILLIONS Idea 9.

All agents share a deterministic pseudo-random stream. Communication then
sends *deltas from the expected stream* rather than absolute values. For
highly-predictable events, the delta is near zero and can be heavily
compressed (or omitted entirely).

This is the Slepian-Wolf / Wyner-Ziv rate with maximum side-information:
the shared random stream is free side-information present at every agent.
The communication rate needed drops from H(X) to H(X | side_info) bits.

Concrete primitive:
    SharedRNG(seed): deterministic PRNG shared by all agents.
    encode(value): delta = value - PRNG.next_expected()
    decode(delta): value = delta + PRNG.next_expected()

For N agents to agree on a common state, each broadcasts delta; receivers
add the PRNG prediction. If the prediction is near the value, delta is
tiny. This compounds with the surprise filter — a message is skipped
entirely if the delta is below the threshold.
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass, field


@dataclass
class SharedRNG:
    """A single-stream pseudo-random generator shared across all agents.

    Agents keep a synchronized counter; calling .expected_next() returns
    the next "expected" vector in the shared stream. All agents agree on
    this value without communication.
    """
    seed: int
    dim: int
    _step: int = 0
    _rng: np.random.Generator = field(default=None)  # type: ignore

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)

    def expected_next(self) -> np.ndarray:
        """Return the next predicted vector. Advances the counter."""
        x = self._rng.standard_normal(self.dim)
        self._step += 1
        return x

    def encode(self, value: np.ndarray) -> np.ndarray:
        """Return the delta (value − expected) to broadcast."""
        expected = self.expected_next()
        return value - expected

    def decode(self, delta: np.ndarray) -> np.ndarray:
        """Reconstruct value from delta (uses the SAME step, so this must
        be called on a separate RNG replica that is synchronized)."""
        expected = self.expected_next()
        return delta + expected

    @staticmethod
    def encoded_bits(delta: np.ndarray, precision: float = 1e-3) -> int:
        """Approximate # bits needed to transmit this delta at given
        precision. For a normally-distributed delta of std σ, this is
        dim * log2(σ / precision) bits.
        """
        std = float(np.std(delta) + 1e-10)
        return int(delta.size * max(0, math.ceil(math.log2(std / precision + 1.0))))


@dataclass
class DeltaChannel:
    """A simulated shared-randomness communication channel.

    Two endpoints each carry a SharedRNG initialized with the same seed.
    Sending a value means: encode via local RNG (producing a small delta),
    broadcast the delta, receiver decodes via its own synced RNG.
    """
    seed: int
    dim: int
    precision: float = 1e-3
    _sender: SharedRNG = field(default=None)  # type: ignore
    _receiver: SharedRNG = field(default=None)  # type: ignore
    _bits_sent: int = 0

    def __post_init__(self):
        self._sender = SharedRNG(seed=self.seed, dim=self.dim)
        self._receiver = SharedRNG(seed=self.seed, dim=self.dim)

    def send(self, value: np.ndarray) -> np.ndarray:
        delta = self._sender.encode(value)
        self._bits_sent += SharedRNG.encoded_bits(delta, self.precision)
        recovered = self._receiver.decode(delta)
        return recovered

    def bits_used(self) -> int:
        return self._bits_sent


def naive_bits(value: np.ndarray, precision: float = 1e-3) -> int:
    """# bits to transmit a value WITHOUT shared randomness — the full
    vector at the given precision."""
    std = float(np.std(value) + 1e-10)
    return int(value.size * math.ceil(math.log2(std / precision + 1.0)))
