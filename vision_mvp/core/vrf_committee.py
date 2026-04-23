"""Hash-based Verifiable Random Function for committee sampling.

Micali, Rabin, Vadhan (1999). A VRF provides a function
    (sk, input) ↦ (output, proof)
such that anyone with the *public key* can verify (proof, input) → output,
but only the secret-key holder can compute it. The output is pseudorandom
for anyone without sk.

This module implements a *hash-based* VRF using Ed25519 signatures as the
pseudorandom bits: given input x and secret key sk,

    sig = Sign_sk(x)
    output = SHA-256(sig)
    proof = sig

Verification: check that sig is a valid signature of x under pk, then
recompute SHA-256(sig). Deterministic (unlike plain randomness).

Used to elect unbiased committees for the O(log N) workspace: each agent
computes VRF(sk_i, round_index); the k smallest outputs form the committee.
Adversary-resistant because the output is determined by sk the adversary
doesn't hold.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

# Guarded — see vision_mvp/core/peer_review.py for rationale.
try:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey, Ed25519PublicKey,
    )
    from cryptography.hazmat.primitives import serialization
    _CRYPTOGRAPHY_AVAILABLE = True
except ImportError:  # pragma: no cover
    Ed25519PrivateKey = None  # type: ignore[assignment,misc]
    Ed25519PublicKey = None  # type: ignore[assignment,misc]
    serialization = None  # type: ignore[assignment]
    _CRYPTOGRAPHY_AVAILABLE = False


def _require_cryptography() -> None:
    if not _CRYPTOGRAPHY_AVAILABLE:
        raise RuntimeError(
            "vrf_committee requires the 'cryptography' extra. "
            "Install with: pip install 'wevra[crypto]'")


@dataclass
class VRFOutput:
    output: int      # uniform-ish integer in [0, 2^256)
    proof: bytes     # Ed25519 signature

    def as_fraction(self) -> float:
        return self.output / (1 << 256)


class VRFKey:
    """VRF wrapper around Ed25519."""

    def __init__(self, seed: bytes | None = None):
        _require_cryptography()
        if seed is not None:
            if len(seed) != 32:
                raise ValueError("seed must be 32 bytes")
            self._sk = Ed25519PrivateKey.from_private_bytes(seed)
        else:
            self._sk = Ed25519PrivateKey.generate()
        self.public_key: Ed25519PublicKey = self._sk.public_key()

    def evaluate(self, input_bytes: bytes) -> VRFOutput:
        sig = self._sk.sign(input_bytes)
        h = hashlib.sha256(sig).digest()
        return VRFOutput(
            output=int.from_bytes(h, "big"),
            proof=sig,
        )

    def public_key_bytes(self) -> bytes:
        return self.public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )


def verify_vrf(
    output: VRFOutput, input_bytes: bytes, public_key: Ed25519PublicKey,
) -> bool:
    """Verify that `output` was produced by holder of sk for `input_bytes`."""
    try:
        public_key.verify(output.proof, input_bytes)
    except Exception:
        return False
    recomputed = int.from_bytes(hashlib.sha256(output.proof).digest(), "big")
    return recomputed == output.output


def elect_committee(
    outputs: dict[str, VRFOutput], k: int,
) -> list[str]:
    """The k agents with smallest VRF outputs form the committee."""
    sorted_agents = sorted(outputs.items(), key=lambda kv: kv[1].output)
    return [a for a, _ in sorted_agents[:k]]
