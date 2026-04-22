"""PeerReview — tamper-evident per-agent logs with spot-check auditing.

Haeberlen, Kouznetsov, Druschel (SOSP 2007). Every agent maintains a hash
chain of its messages; peers periodically request a slice of the chain and
replay the agent's state machine to check consistency. Any divergence is a
detectable fault.

Security property: any deterministic fault is detectable with probability
≥ 1 − (1 − s)^w, where s is the per-message spot-check rate and w is the
number of witnesses per agent.

Signatures use Ed25519 from the `cryptography` library.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey, Ed25519PublicKey,
)
from cryptography.hazmat.primitives import serialization


def _canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


@dataclass
class LogEntry:
    seq: int
    prev_hash: str
    payload: Any
    hash: str
    signature: bytes

    def to_dict(self) -> dict:
        return {
            "seq": self.seq,
            "prev_hash": self.prev_hash,
            "payload": self.payload,
            "hash": self.hash,
            "signature": self.signature.hex(),
        }


class HashChainLog:
    """Per-agent append-only log, signed with Ed25519."""

    def __init__(self, agent_id: str, seed: bytes | None = None):
        self.agent_id = agent_id
        if seed is not None:
            if len(seed) != 32:
                raise ValueError("seed must be 32 bytes")
            self._sk = Ed25519PrivateKey.from_private_bytes(seed)
        else:
            self._sk = Ed25519PrivateKey.generate()
        self.public_key = self._sk.public_key()
        self._entries: list[LogEntry] = []

    def append(self, payload: Any) -> LogEntry:
        seq = len(self._entries)
        prev = self._entries[-1].hash if self._entries else "0" * 64
        body = _canonical({"agent": self.agent_id, "seq": seq,
                           "prev": prev, "payload": payload})
        h = hashlib.sha256(body).hexdigest()
        sig = self._sk.sign(body)
        entry = LogEntry(seq=seq, prev_hash=prev, payload=payload, hash=h, signature=sig)
        self._entries.append(entry)
        return entry

    def entries(self) -> list[LogEntry]:
        return list(self._entries)

    def public_key_bytes(self) -> bytes:
        return self.public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )


def verify_log(
    entries: list[LogEntry], public_key: Ed25519PublicKey, agent_id: str,
) -> tuple[bool, str]:
    """Check signatures and the hash chain. Returns (ok, reason)."""
    prev = "0" * 64
    for i, e in enumerate(entries):
        if e.seq != i:
            return False, f"seq gap at {i}: got {e.seq}"
        if e.prev_hash != prev:
            return False, f"prev-hash mismatch at seq {e.seq}"
        body = _canonical({"agent": agent_id, "seq": e.seq,
                            "prev": e.prev_hash, "payload": e.payload})
        if hashlib.sha256(body).hexdigest() != e.hash:
            return False, f"content-hash mismatch at seq {e.seq}"
        try:
            public_key.verify(e.signature, body)
        except Exception as exc:
            return False, f"signature invalid at seq {e.seq}: {exc}"
        prev = e.hash
    return True, "ok"


def spot_check(
    entries: list[LogEntry],
    sample_rate: float,
    public_key: Ed25519PublicKey,
    agent_id: str,
    seed: int = 0,
) -> tuple[bool, str]:
    """Randomly pick entries and verify each one's signature + self-hash.

    Does NOT re-verify the whole chain (for that, call `verify_log`). Catches
    per-entry forgeries with probability ≥ 1 − (1 − sample_rate)^n, where n
    is the number of malicious entries in the log.
    """
    import random
    rng = random.Random(seed)
    sampled = [e for e in entries if rng.random() < sample_rate]
    for e in sampled:
        body = _canonical({"agent": agent_id, "seq": e.seq,
                            "prev": e.prev_hash, "payload": e.payload})
        if hashlib.sha256(body).hexdigest() != e.hash:
            return False, f"content-hash mismatch at seq {e.seq}"
        try:
            public_key.verify(e.signature, body)
        except Exception as exc:
            return False, f"signature invalid at seq {e.seq}: {exc}"
    return True, f"ok: {len(sampled)} entries verified"
