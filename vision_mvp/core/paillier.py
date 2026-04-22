"""Paillier — additively-homomorphic encryption (teaching implementation).

Paillier (1999). The public operation Enc is additively homomorphic:
    Enc(m1) · Enc(m2) mod n² = Enc(m1 + m2 mod n)

Scalar multiplication: Enc(m)^k mod n² = Enc(k·m mod n).

This gives CASR an aggregate-sum primitive for federated settings: each
agent encrypts its contribution under a shared public key; anyone can
aggregate without decrypting; only the key-holder can decrypt the sum.

Not side-channel-hardened — suitable for simulation. Production deployments
should use a reviewed library.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from math import gcd


def _miller_rabin(n: int, rounds: int = 32) -> bool:
    """Probabilistic primality test."""
    if n < 4:
        return n in (2, 3)
    if n % 2 == 0:
        return False
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for _ in range(rounds):
        a = secrets.randbelow(n - 3) + 2
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False
    return True


def _gen_prime(bits: int) -> int:
    while True:
        p = secrets.randbits(bits) | 1 | (1 << (bits - 1))
        if _miller_rabin(p):
            return p


@dataclass
class PaillierPublicKey:
    n: int             # modulus
    g: int             # generator

    def n_squared(self) -> int:
        return self.n * self.n

    def encrypt(self, m: int, r: int | None = None) -> int:
        nsq = self.n_squared()
        m = m % self.n
        if r is None:
            while True:
                r = secrets.randbelow(self.n - 1) + 1
                if gcd(r, self.n) == 1:
                    break
        return (pow(self.g, m, nsq) * pow(r, self.n, nsq)) % nsq


@dataclass
class PaillierSecretKey:
    lam: int           # λ = lcm(p-1, q-1)
    mu: int            # µ = (L(g^λ mod n²))^{-1} mod n
    public: PaillierPublicKey

    def decrypt(self, c: int) -> int:
        nsq = self.public.n_squared()
        x = pow(c, self.lam, nsq)
        l = (x - 1) // self.public.n
        return (l * self.mu) % self.public.n


def keygen(bits: int = 512) -> tuple[PaillierPublicKey, PaillierSecretKey]:
    """Generate a Paillier keypair with RSA-like security parameter `bits`.

    For simulation / tests we use 256 bits (fast, insecure). Real systems need
    ≥ 2048. The function enforces ≥ 128 for speed during unit tests.
    """
    if bits < 128:
        raise ValueError("bits must be ≥ 128")
    # Generate distinct primes of roughly equal size
    p = _gen_prime(bits // 2)
    while True:
        q = _gen_prime(bits // 2)
        if q != p:
            break
    n = p * q
    lam = ((p - 1) * (q - 1)) // gcd(p - 1, q - 1)
    g = n + 1                  # standard choice
    nsq = n * n
    # L function: L(x) = (x - 1) / n
    gl = pow(g, lam, nsq)
    l_val = (gl - 1) // n
    mu = pow(l_val, -1, n)
    pk = PaillierPublicKey(n=n, g=g)
    sk = PaillierSecretKey(lam=lam, mu=mu, public=pk)
    return pk, sk


def homomorphic_add(c1: int, c2: int, pk: PaillierPublicKey) -> int:
    """Enc(m1 + m2) = Enc(m1) · Enc(m2) mod n²."""
    return (c1 * c2) % pk.n_squared()


def homomorphic_scale(c: int, k: int, pk: PaillierPublicKey) -> int:
    """Enc(k · m) = Enc(m)^k mod n²."""
    return pow(c, k, pk.n_squared())
