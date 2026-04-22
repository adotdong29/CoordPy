"""CASRRouter — mode-switchable bulletin router for the 3-leg CASR benchmark.

Modes:
  - "full":     baseline, every message is delivered to every non-self agent
  - "casr":     deliver only if source is in recipient's causal footprint
  - "ablation": deliver only if source is in recipient's RANDOM footprint
                (same cardinality as CASR, random membership — controls for
                 the confound that "fewer messages = fewer tokens" regardless
                 of whether the selection was causal)
"""

from __future__ import annotations

from dataclasses import dataclass

from .causal_footprint import CausalFootprint


VALID_MODES = ("full", "casr", "ablation")


@dataclass
class RouterMessage:
    source_id: str
    payload: str
    tokens: int


@dataclass
class RoutingStats:
    delivered: int = 0
    dropped: int = 0
    delivered_tokens: int = 0
    dropped_tokens: int = 0


class CASRRouter:
    """Filters a batch of RouterMessages for a single recipient."""

    def __init__(self, mode: str, footprints: dict[str, CausalFootprint]):
        if mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")
        self.mode = mode
        self.footprints = footprints

    def route(
        self, messages: list[RouterMessage], recipient_id: str
    ) -> tuple[list[RouterMessage], RoutingStats]:
        """Return (delivered, stats) — the subset of `messages` that should be
        handed to `recipient_id`, plus per-call accounting."""
        delivered: list[RouterMessage] = []
        stats = RoutingStats()

        if self.mode == "full":
            recipient_fp = None
        else:
            recipient_fp = self.footprints.get(recipient_id)

        for msg in messages:
            # Never echo a message back to its own source.
            if msg.source_id == recipient_id:
                stats.dropped += 1
                stats.dropped_tokens += msg.tokens
                continue

            if self.mode == "full":
                keep = True
            else:
                # For "casr" or "ablation", check the recipient's footprint.
                if recipient_fp is None:
                    keep = False
                else:
                    keep = msg.source_id in recipient_fp

            if keep:
                delivered.append(msg)
                stats.delivered += 1
                stats.delivered_tokens += msg.tokens
            else:
                stats.dropped += 1
                stats.dropped_tokens += msg.tokens

        return delivered, stats
