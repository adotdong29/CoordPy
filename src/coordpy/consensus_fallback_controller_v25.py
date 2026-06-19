"""W79 M5 — Consensus / Fallback Controller V25.

Strictly extends W78's ``coordpy.consensus_fallback_controller_v24``
with two new W79 stages (replacement-then-restart-after-long-delay
arbiter + replacement-then-restart-after-long-delay-best-parent
arbiter), bringing the chain to **forty-four** stages.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any

from .consensus_fallback_controller_v24 import (
    ConsensusFallbackControllerV24,
)


W79_CONSENSUS_V25_SCHEMA_VERSION: str = (
    "coordpy.consensus_fallback_controller_v25.v1")
W79_CONSENSUS_V25_N_STAGES: int = 44


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass
class ConsensusFallbackControllerV25:
    inner_v24: ConsensusFallbackControllerV24
    replacement_then_restart_after_long_delay_threshold: float = 0.5
    replacement_then_restart_after_long_delay_best_parent_threshold: (
        float) = 0.5

    @classmethod
    def init(
            cls, *,
            replacement_then_restart_after_long_delay_threshold: float = 0.5,
            replacement_then_restart_after_long_delay_best_parent_threshold: float = 0.5,
            **v24_kwargs: Any,
    ) -> "ConsensusFallbackControllerV25":
        inner = ConsensusFallbackControllerV24.init(
            **v24_kwargs)
        return cls(
            inner_v24=inner,
            replacement_then_restart_after_long_delay_threshold=float(
                replacement_then_restart_after_long_delay_threshold),
            replacement_then_restart_after_long_delay_best_parent_threshold=float(
                replacement_then_restart_after_long_delay_best_parent_threshold))

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w79_consensus_v25",
            "schema": W79_CONSENSUS_V25_SCHEMA_VERSION,
            "inner_v24_cid": str(self.inner_v24.cid()),
            "replacement_then_restart_after_long_delay_threshold": float(
                round(
                    self
                    .replacement_then_restart_after_long_delay_threshold,
                    12)),
            "replacement_then_restart_after_long_delay_best_parent_threshold": float(
                round(
                    self
                    .replacement_then_restart_after_long_delay_best_parent_threshold,
                    12)),
            "n_stages": int(W79_CONSENSUS_V25_N_STAGES),
        })


@dataclasses.dataclass(frozen=True)
class ConsensusFallbackControllerV25Witness:
    schema: str
    controller_cid: str
    n_stages: int

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w79_consensus_v25_witness",
            "schema": str(self.schema),
            "controller_cid": str(self.controller_cid),
            "n_stages": int(self.n_stages),
        })


def emit_consensus_v25_witness(
        controller: ConsensusFallbackControllerV25,
) -> ConsensusFallbackControllerV25Witness:
    return ConsensusFallbackControllerV25Witness(
        schema=W79_CONSENSUS_V25_SCHEMA_VERSION,
        controller_cid=str(controller.cid()),
        n_stages=int(W79_CONSENSUS_V25_N_STAGES))


__all__ = [
    "W79_CONSENSUS_V25_SCHEMA_VERSION",
    "W79_CONSENSUS_V25_N_STAGES",
    "ConsensusFallbackControllerV25",
    "ConsensusFallbackControllerV25Witness",
    "emit_consensus_v25_witness",
]
