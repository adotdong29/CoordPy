"""W52 M8 — Transcript-vs-Shared-State Matched-Budget Comparator.

Under a fixed visible-token budget ``B``, compares two arms:

* **transcript-only** — the team's transcript is truncated to
  ``B`` tokens (the bounded-context default).
* **shared-latent** — the team uses a ``B``-token visible
  header from the W52 quantised codebook instead of the
  transcript.

Measures per-arm retention cosine at a target turn. The
comparator reports the strict gap at matched budget plus the
bit-density gap.

Pure-Python only — reuses the W52 quantised compression.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Any, Sequence

from .autograd_manifold import (
    W47_DEFAULT_TRAIN_SEED,
    _DeterministicLCG,
)
from .quantised_compression import (
    QuantisedBudgetGate,
    QuantisedCodebookV4,
    compress_carrier_quantised,
)


# =============================================================================
# Schema, defaults
# =============================================================================

W52_TRANSCRIPT_VS_SHARED_SCHEMA_VERSION: str = (
    "coordpy.transcript_vs_shared_state.v1")


# =============================================================================
# Helpers
# =============================================================================

def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for i in range(n):
        ai = float(a[i])
        bi = float(b[i])
        dot += ai * bi
        na += ai * ai
        nb += bi * bi
    if na <= 1e-30 or nb <= 1e-30:
        return 0.0
    return float(dot / (math.sqrt(na) * math.sqrt(nb)))


# =============================================================================
# Comparator
# =============================================================================


@dataclasses.dataclass(frozen=True)
class TranscriptVsSharedComparisonResult:
    """Result of the matched-budget comparison."""

    budget_tokens: int
    transcript_retention_cosine: float
    shared_retention_cosine: float
    transcript_arm_visible_tokens: int
    shared_arm_visible_tokens: int
    transcript_arm_structured_bits: int
    shared_arm_structured_bits: int
    bit_density_gap: float
    retention_gap: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "budget_tokens": int(self.budget_tokens),
            "transcript_retention_cosine": float(round(
                self.transcript_retention_cosine, 12)),
            "shared_retention_cosine": float(round(
                self.shared_retention_cosine, 12)),
            "transcript_arm_visible_tokens": int(
                self.transcript_arm_visible_tokens),
            "shared_arm_visible_tokens": int(
                self.shared_arm_visible_tokens),
            "transcript_arm_structured_bits": int(
                self.transcript_arm_structured_bits),
            "shared_arm_structured_bits": int(
                self.shared_arm_structured_bits),
            "bit_density_gap": float(round(
                self.bit_density_gap, 12)),
            "retention_gap": float(round(
                self.retention_gap, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w52_transcript_vs_shared_comparison",
            "result": self.to_dict()})


def _transcript_truncate_value(
        carrier: Sequence[float],
        *,
        budget_tokens: int,
        bits_per_natural_token: float = 6.0,
) -> tuple[list[float], int, int]:
    """Simulate truncating a transcript to ``budget_tokens``.

    We approximate the transcript by treating each ``carrier``
    dimension as one natural-language "token" carrying
    ``bits_per_natural_token`` of structured bits. The
    truncation keeps the first ``budget_tokens`` dimensions
    and zeros the rest — a faithful upper-bound on what the
    bounded-context default loses when the transcript exceeds
    the budget.
    """
    out = [0.0] * len(carrier)
    keep = max(0, int(budget_tokens))
    for i in range(min(keep, len(carrier))):
        out[i] = float(carrier[i])
    structured_bits = int(
        float(bits_per_natural_token) * float(keep))
    visible_tokens = int(keep)
    return out, visible_tokens, structured_bits


def compare_transcript_vs_shared_state(
        carrier: Sequence[float],
        *,
        codebook: QuantisedCodebookV4,
        gate: QuantisedBudgetGate,
        budget_tokens: int = 3,
        bits_per_natural_token: float = 6.0,
) -> TranscriptVsSharedComparisonResult:
    """One matched-budget comparison call.

    Returns a result detailing both arms' retention cosine
    against the original carrier + bit-density.
    """
    # Transcript arm: truncate the carrier to budget_tokens.
    trans_decoded, trans_tokens, trans_bits = (
        _transcript_truncate_value(
            carrier,
            budget_tokens=int(budget_tokens),
            bits_per_natural_token=float(
                bits_per_natural_token)))
    trans_retention = _cosine(carrier, trans_decoded)
    # Shared-latent arm: quantise into the W52 codebook with
    # budget_tokens visible-token cap, decode, measure cosine.
    res = compress_carrier_quantised(
        carrier, codebook=codebook, gate=gate,
        max_visible_tokens=int(budget_tokens))
    decoded = codebook.decode(
        coarse=res.coarse_code,
        fine=res.fine_code,
        ultra=res.ultra_code,
        include_ultra=(len(res.level_mask) >= 3
                        and int(res.level_mask[2]) == 1))
    shared_retention = _cosine(carrier, decoded)
    return TranscriptVsSharedComparisonResult(
        budget_tokens=int(budget_tokens),
        transcript_retention_cosine=float(trans_retention),
        shared_retention_cosine=float(shared_retention),
        transcript_arm_visible_tokens=int(trans_tokens),
        shared_arm_visible_tokens=int(res.visible_tokens),
        transcript_arm_structured_bits=int(trans_bits),
        shared_arm_structured_bits=int(res.structured_bits),
        bit_density_gap=float(
            (res.structured_bits / float(max(1, res.visible_tokens)))
            - (trans_bits / float(max(1, trans_tokens)))),
        retention_gap=float(shared_retention - trans_retention),
    )


# =============================================================================
# Witness
# =============================================================================


@dataclasses.dataclass(frozen=True)
class TranscriptVsSharedWitness:
    comparison_cid: str
    budget_tokens: int
    transcript_retention_cosine: float
    shared_retention_cosine: float
    retention_gap: float
    bit_density_gap: float
    n_probes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "comparison_cid": str(self.comparison_cid),
            "budget_tokens": int(self.budget_tokens),
            "transcript_retention_cosine": float(round(
                self.transcript_retention_cosine, 12)),
            "shared_retention_cosine": float(round(
                self.shared_retention_cosine, 12)),
            "retention_gap": float(round(
                self.retention_gap, 12)),
            "bit_density_gap": float(round(
                self.bit_density_gap, 12)),
            "n_probes": int(self.n_probes),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w52_transcript_vs_shared_witness",
            "witness": self.to_dict()})


def emit_transcript_vs_shared_witness(
        *,
        carriers: Sequence[Sequence[float]],
        codebook: QuantisedCodebookV4,
        gate: QuantisedBudgetGate,
        budget_tokens: int = 3,
        bits_per_natural_token: float = 6.0,
) -> TranscriptVsSharedWitness:
    """Run the comparison across ``carriers`` and seal a witness."""
    if not carriers:
        return TranscriptVsSharedWitness(
            comparison_cid="",
            budget_tokens=int(budget_tokens),
            transcript_retention_cosine=0.0,
            shared_retention_cosine=0.0,
            retention_gap=0.0,
            bit_density_gap=0.0,
            n_probes=0,
        )
    rs: list[TranscriptVsSharedComparisonResult] = []
    for c in carriers:
        rs.append(compare_transcript_vs_shared_state(
            c, codebook=codebook, gate=gate,
            budget_tokens=int(budget_tokens),
            bits_per_natural_token=float(
                bits_per_natural_token)))
    n = float(len(rs))
    tr_mean = float(
        sum(r.transcript_retention_cosine for r in rs)) / n
    sh_mean = float(
        sum(r.shared_retention_cosine for r in rs)) / n
    bd_mean = float(sum(r.bit_density_gap for r in rs)) / n
    gap_mean = float(sum(r.retention_gap for r in rs)) / n
    comparison_cid = _sha256_hex({
        "kind": "w52_transcript_vs_shared_comparison_bundle",
        "results": [r.to_dict() for r in rs],
    })
    return TranscriptVsSharedWitness(
        comparison_cid=str(comparison_cid),
        budget_tokens=int(budget_tokens),
        transcript_retention_cosine=float(tr_mean),
        shared_retention_cosine=float(sh_mean),
        retention_gap=float(gap_mean),
        bit_density_gap=float(bd_mean),
        n_probes=int(len(rs)),
    )


# =============================================================================
# Verifier
# =============================================================================


W52_TRANSCRIPT_VS_SHARED_VERIFIER_FAILURE_MODES: tuple[str, ...] = (
    "w52_transcript_vs_shared_schema_mismatch",
    "w52_transcript_vs_shared_comparison_cid_mismatch",
    "w52_transcript_vs_shared_budget_mismatch",
    "w52_transcript_vs_shared_probe_count_mismatch",
)


def verify_transcript_vs_shared_witness(
        witness: TranscriptVsSharedWitness,
        *,
        expected_comparison_cid: str | None = None,
        expected_budget_tokens: int | None = None,
        expected_n_probes: int | None = None,
) -> dict[str, Any]:
    failures: list[str] = []
    if (expected_comparison_cid is not None
            and witness.comparison_cid
            != expected_comparison_cid):
        failures.append(
            "w52_transcript_vs_shared_comparison_cid_mismatch")
    if (expected_budget_tokens is not None
            and witness.budget_tokens
            != int(expected_budget_tokens)):
        failures.append(
            "w52_transcript_vs_shared_budget_mismatch")
    if (expected_n_probes is not None
            and witness.n_probes != int(expected_n_probes)):
        failures.append(
            "w52_transcript_vs_shared_probe_count_mismatch")
    return {
        "ok": (len(failures) == 0),
        "failures": failures,
        "witness_cid": witness.cid(),
    }


__all__ = [
    "W52_TRANSCRIPT_VS_SHARED_SCHEMA_VERSION",
    "W52_TRANSCRIPT_VS_SHARED_VERIFIER_FAILURE_MODES",
    "TranscriptVsSharedComparisonResult",
    "TranscriptVsSharedWitness",
    "compare_transcript_vs_shared_state",
    "emit_transcript_vs_shared_witness",
    "verify_transcript_vs_shared_witness",
]
