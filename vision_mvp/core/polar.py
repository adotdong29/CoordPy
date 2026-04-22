"""Polar codes — capacity-achieving, explicit, O(N log N).

Arıkan (2009). Via the 2×2 kernel F = [[1, 0], [1, 1]], the Kronecker-power
F^⊗n polarises the n = 2ⁿ subchannels into ones with capacity approaching 1
and ones approaching 0. For a target code rate R, place information bits in
the K = ⌊R·N⌋ most-reliable subchannels and fix the rest (frozen bits) to 0.

Successive-Cancellation (SC) decoding recursively splits the received vector
and makes decisions bit-by-bit based on LLRs. Complexity O(N log N) in both
encoder and decoder.

This is an explicit, deterministic coding scheme — not a Monte-Carlo design
like LDPC. Used as a Wave-4 alternative for the holographic-boundary layer
and as a concrete capacity-achieving baseline for benchmarks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _polar_encode_bits(u: np.ndarray) -> np.ndarray:
    """Encode u via F^⊗n · u (mod 2), recursively. Input length = 2^n."""
    n = u.size
    if n == 1:
        return u.copy()
    half = n // 2
    u_even = u[:half]
    u_odd = u[half:]
    # u_top + u_bot for the low half; u_bot for the high half
    low = _polar_encode_bits(np.bitwise_xor(u_even, u_odd))
    high = _polar_encode_bits(u_odd)
    return np.concatenate([low, high])


@dataclass
class PolarCode:
    """Polar code for rate K/N on a binary symmetric channel at design p."""

    N: int                   # code length — must be power of 2
    K: int                   # # information bits
    frozen_mask: np.ndarray  # (N,) bool: True where frozen
    info_positions: np.ndarray  # (K,) indices of non-frozen

    def __post_init__(self):
        n = self.N
        if n & (n - 1):
            raise ValueError("N must be a power of 2")
        if not (0 < self.K <= self.N):
            raise ValueError("0 < K ≤ N required")

    @classmethod
    def design_bec(cls, N: int, K: int, epsilon: float = 0.5) -> "PolarCode":
        """Design code by Bhattacharyya-parameter recursion for BEC.

        Channel-reliability proxy: higher Z → less reliable. Freeze the K
        most-reliable positions, keep the rest frozen.
        """
        n = int(np.log2(N))
        # Z_0 = ε; recursion: Z_{+}(a,a) = 2a - a², Z_{-}(a,a) = a²
        z = np.array([epsilon])
        for _ in range(n):
            a = z
            z_minus = a * a
            z_plus = 2 * a - a * a
            z = np.concatenate([z_minus, z_plus])
        # Frozen positions: worst K indices by Z (highest = least reliable)
        order = np.argsort(z)          # ascending reliability
        info = order[:K]               # K best reliabilities
        frozen = np.ones(N, dtype=bool)
        frozen[info] = False
        return cls(N=N, K=K, frozen_mask=frozen, info_positions=np.sort(info))

    def encode(self, message: np.ndarray) -> np.ndarray:
        """Encode a length-K binary message into length-N codeword."""
        m = np.asarray(message, dtype=np.int8)
        if m.size != self.K:
            raise ValueError(f"message must have K={self.K} bits")
        u = np.zeros(self.N, dtype=np.int8)
        u[self.info_positions] = m
        # Polar transform
        return _polar_encode_bits(u.astype(np.int8))

    # ------ Successive-cancellation decoder ------

    def decode(self, received: np.ndarray, llr_scale: float = 4.0) -> np.ndarray:
        """SC decoder. `received` is a length-N binary vector (hard decisions)."""
        y = np.asarray(received, dtype=float)
        if y.size != self.N:
            raise ValueError(f"received must have N={self.N} entries")
        # Convert binary to LLRs: +llr_scale if bit 0, −llr_scale if bit 1
        llr = llr_scale * (1 - 2 * y)

        u_hat = np.zeros(self.N, dtype=np.int8)
        self._sc_decode(llr, 0, u_hat)
        return u_hat[self.info_positions]

    def _sc_decode(self, llr: np.ndarray, offset: int, u_hat: np.ndarray) -> np.ndarray:
        """Recursive SC decoder; mutates u_hat. Returns the length-N/2 bits."""
        N = llr.size
        if N == 1:
            pos = offset
            if self.frozen_mask[pos]:
                u_hat[pos] = 0
            else:
                u_hat[pos] = 0 if llr[0] > 0 else 1
            return np.array([u_hat[pos]], dtype=np.int8)
        half = N // 2
        # f node: combine top/bottom LLRs
        llr_left = np.sign(llr[:half]) * np.sign(llr[half:]) * np.minimum(np.abs(llr[:half]), np.abs(llr[half:]))
        left_bits = self._sc_decode(llr_left, offset, u_hat)
        # g node: given left bits, compute right LLRs
        llr_right = llr[half:] + (1 - 2.0 * left_bits) * llr[:half]
        right_bits = self._sc_decode(llr_right, offset + half, u_hat)
        return np.concatenate([np.bitwise_xor(left_bits, right_bits), right_bits])
