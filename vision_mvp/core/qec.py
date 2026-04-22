"""Classical simulation of a CSS/repetition quantum-error-correcting code.

Kitaev (2003); Raussendorf-Harrington (2007). The surface code in the
noise-free limit reduces to two independent classical repetition codes — one
for X errors, one for Z errors. For simulation purposes (and for the
boundary-bulk-reconstruction story in Idea 5), we implement the 3-qubit
repetition code plus a small surface-like layout.

Encoder:
  |0> → |000>, |1> → |111>   (3-qubit rep code)
Error model:
  each physical qubit flips independently with probability p.
Decoder:
  majority vote of the three qubits.
Logical error rate:
  p_L = 3p²(1 − p) + p³

We expose:
  - `encode(bit)` → 3-bit codeword
  - `flip_channel(codeword, p)` → noisy codeword
  - `majority_decode(codeword)` → estimated logical bit
  - `threshold_p()` → p at which logical error = physical error (= 0.5)

Stabilizer representation over GF(2) is added for the surface code: we build
the X-stabilizer parity-check matrix for a d×d lattice and expose syndrome
extraction — enough to demonstrate bulk reconstruction from boundary
syndromes (Idea 5).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ------------------- 3-qubit repetition ------------------------------------

def encode_rep3(bit: int) -> np.ndarray:
    if bit not in (0, 1):
        raise ValueError("bit must be 0 or 1")
    return np.full(3, bit, dtype=np.int8)


def flip_channel(codeword: np.ndarray, p: float, rng: np.random.Generator) -> np.ndarray:
    flips = rng.random(codeword.size) < p
    return np.bitwise_xor(codeword, flips.astype(np.int8))


def majority_decode(codeword: np.ndarray) -> int:
    return 1 if int(np.sum(codeword)) >= 2 else 0


def logical_error_rate_rep3(p: float) -> float:
    """Logical error of 3-qubit repetition under bit-flip noise."""
    return 3 * p ** 2 * (1 - p) + p ** 3


# ------------------- surface-code skeleton ---------------------------------

@dataclass
class SurfaceLayout:
    """d×d toric-code lattice with X-stabilizer plaquettes.

    Each face of the lattice has an X-check over its 4 incident edges.
    Returns the (d², 2d²) parity-check matrix H over GF(2), where columns
    index qubits (edges) and rows index plaquette stabilizers.
    """

    d: int
    parity_check: np.ndarray = None  # type: ignore

    def __post_init__(self):
        d = self.d
        if d < 2:
            raise ValueError("d must be ≥ 2")
        n_qubits = 2 * d * d     # horizontal + vertical edges
        n_stab = d * d
        H = np.zeros((n_stab, n_qubits), dtype=np.int8)
        # row index = r*d + c identifies plaquette at (row r, col c)
        for r in range(d):
            for c in range(d):
                stab_idx = r * d + c
                # Four edges around plaquette (indices with wrap-around)
                # Horizontal edges (indexed 0 .. d²-1): top = r*d+c; bottom = ((r+1) mod d) * d + c
                top = r * d + c
                bottom = ((r + 1) % d) * d + c
                # Vertical edges (indexed d² .. 2d²-1): left = r*d+c; right = r*d + (c+1) mod d
                left = d * d + r * d + c
                right = d * d + r * d + ((c + 1) % d)
                for q in (top, bottom, left, right):
                    H[stab_idx, q] = 1
        self.parity_check = H

    def syndrome(self, error: np.ndarray) -> np.ndarray:
        """Compute the stabilizer syndrome for a given error pattern."""
        if error.size != self.parity_check.shape[1]:
            raise ValueError("error length mismatch")
        return (self.parity_check @ error) % 2
