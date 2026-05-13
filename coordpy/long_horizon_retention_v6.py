"""W54 M7 — Long-Horizon Reconstruction V6.

Adds a fifth head (cross-role) to W53 V5's 4-head reconstruction
and stretches ``max_k`` to 24 (vs V5's 16) with degradation
curve probe to ``k=48``.

The 5th head is a per-role projection from the merged V5 output
to a role-specific feature vector — it reconstructs *whose*
state the reconstruction is for (the producer role).

The "degradation-aware" reconstruction returns per-dim
degradation scores alongside the reconstruction: ``degradation_i
= |causal_i - merged_branch_i|`` (treated as a per-dim
disagreement between the causal-only head and the merged-branch
head).

Honest scope: pure-Python only, capsule-layer only.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Any, Sequence

from .autograd_manifold import (
    ParamTensor,
    W47_DEFAULT_INIT_SCALE,
    W47_DEFAULT_TRAIN_SEED,
    _DeterministicLCG,
)
from .long_horizon_retention_v4 import (
    LongHorizonV4Example,
    evaluate_long_horizon_v4_mse_at_k,
)
from .long_horizon_retention_v5 import (
    LongHorizonReconstructionV5Head,
    LongHorizonReconstructionV5Witness,
    V5DegradationPoint,
    W53_DEFAULT_LHR_V5_FLAT_FEATURE_DIM,
    W53_DEFAULT_LHR_V5_HIDDEN_DIM,
    W53_DEFAULT_LHR_V5_N_BRANCHES,
    W53_DEFAULT_LHR_V5_N_CYCLES,
    W53_DEFAULT_LHR_V5_N_MERGE_PAIRS,
)


# =============================================================================
# Schema, defaults
# =============================================================================

W54_LHR_V6_SCHEMA_VERSION: str = (
    "coordpy.long_horizon_retention_v6.v1")

W54_DEFAULT_LHR_V6_FLAT_FEATURE_DIM: int = (
    W53_DEFAULT_LHR_V5_FLAT_FEATURE_DIM)
W54_DEFAULT_LHR_V6_HIDDEN_DIM: int = (
    W53_DEFAULT_LHR_V5_HIDDEN_DIM)
W54_DEFAULT_LHR_V6_MAX_K: int = 24
W54_DEFAULT_LHR_V6_DEGRADATION_K_MAX: int = 48
W54_DEFAULT_LHR_V6_N_BRANCHES: int = (
    W53_DEFAULT_LHR_V5_N_BRANCHES)
W54_DEFAULT_LHR_V6_N_CYCLES: int = (
    W53_DEFAULT_LHR_V5_N_CYCLES)
W54_DEFAULT_LHR_V6_N_MERGE_PAIRS: int = (
    W53_DEFAULT_LHR_V5_N_MERGE_PAIRS)
W54_DEFAULT_LHR_V6_N_ROLES: int = 4


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


def _round_floats(
        values: Sequence[float], precision: int = 12,
) -> list[float]:
    return [float(round(float(v), precision)) for v in values]


# =============================================================================
# LongHorizonReconstructionV6Head
# =============================================================================


@dataclasses.dataclass
class LongHorizonReconstructionV6Head:
    """V6 head — V5 inner + cross-role head + max_k=24."""

    inner_v5: LongHorizonReconstructionV5Head
    out_dim: int
    n_roles: int
    max_k_v6: int
    w_role: ParamTensor

    @classmethod
    def init(
            cls, *,
            carrier_dim: int,
            hidden_dim: int = (
                W54_DEFAULT_LHR_V6_HIDDEN_DIM),
            out_dim: int = (
                W54_DEFAULT_LHR_V6_FLAT_FEATURE_DIM),
            max_k: int = W54_DEFAULT_LHR_V6_MAX_K,
            n_branches: int = (
                W54_DEFAULT_LHR_V6_N_BRANCHES),
            n_cycles: int = W54_DEFAULT_LHR_V6_N_CYCLES,
            n_merge_pairs: int = (
                W54_DEFAULT_LHR_V6_N_MERGE_PAIRS),
            n_roles: int = W54_DEFAULT_LHR_V6_N_ROLES,
            seed: int = W47_DEFAULT_TRAIN_SEED,
            init_scale: float = W47_DEFAULT_INIT_SCALE,
    ) -> "LongHorizonReconstructionV6Head":
        rng = _DeterministicLCG(seed=int(seed))
        inner_max_k = max(16, int(max_k))
        inner = LongHorizonReconstructionV5Head.init(
            carrier_dim=int(carrier_dim),
            hidden_dim=int(hidden_dim),
            out_dim=int(out_dim),
            max_k=int(inner_max_k),
            n_branches=int(n_branches),
            n_cycles=int(n_cycles),
            n_merge_pairs=int(n_merge_pairs),
            seed=int(rng.next_uniform() * (1 << 30)),
            init_scale=float(init_scale))
        role_in = int(out_dim) + int(n_roles)
        w_role = ParamTensor(
            shape=(int(out_dim), int(role_in)),
            values=[])
        w_role.init_seed(
            seed=int(rng.next_uniform() * (1 << 30)),
            scale=float(init_scale) * 0.5)
        return cls(
            inner_v5=inner,
            out_dim=int(out_dim),
            n_roles=int(n_roles),
            max_k_v6=int(max_k),
            w_role=w_role,
        )

    def params(self) -> list[ParamTensor]:
        return list(self.inner_v5.params()) + [self.w_role]

    @property
    def max_k(self) -> int:
        return int(self.max_k_v6)

    @property
    def n_branches(self) -> int:
        return int(self.inner_v5.n_branches)

    @property
    def n_cycles(self) -> int:
        return int(self.inner_v5.n_cycles)

    @property
    def carrier_dim(self) -> int:
        return int(self.inner_v5.carrier_dim)

    def forward_value(
            self, *,
            carrier: Sequence[float],
            k: int,
            branch_index: int,
            cycle_index: int,
            merge_pair_index: int = 0,
            role_index: int = 0,
    ) -> tuple[list[float], list[float], list[float], list[float]]:
        """Return (causal+branch+cycle, merged_branch, cross_role, degradation)."""
        k_clipped_inner = (
            int(k) if int(k) <= self.inner_v5.max_k_v5
            else self.inner_v5.max_k_v5)
        v4_out, merge_out = self.inner_v5.forward_value(
            carrier=carrier,
            k=int(k_clipped_inner),
            branch_index=int(branch_index),
            cycle_index=int(cycle_index),
            merge_pair_index=int(merge_pair_index))
        # Cross-role head: projects (v4_out concat role_onehot) → out.
        nr = max(1, int(self.n_roles))
        oh = [
            1.0 if (i == int(role_index) % nr) else 0.0
            for i in range(nr)
        ]
        role_in = list(v4_out) + oh
        wr = self.w_role.values
        n_in = int(self.out_dim) + int(self.n_roles)
        role_out = [0.0] * int(self.out_dim)
        for r in range(int(self.out_dim)):
            base = r * n_in
            s = 0.0
            for j in range(int(n_in)):
                cj = float(
                    role_in[j] if j < len(role_in) else 0.0)
                s += float(wr[base + j]) * cj
            role_out[r] = float(s)
        # Degradation score: |v4_out - merge_out|.
        degradation = [
            float(abs(float(v4_out[i] if i < len(v4_out) else 0.0)
                       - float(merge_out[i]
                               if i < len(merge_out) else 0.0)))
            for i in range(int(self.out_dim))
        ]
        if int(k) > int(self.max_k_v6):
            role_out = [0.0] * int(self.out_dim)
            degradation = [1.0] * int(self.out_dim)
        return v4_out, merge_out, role_out, degradation

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": str(W54_LHR_V6_SCHEMA_VERSION),
            "inner_v5_cid": str(self.inner_v5.cid()),
            "out_dim": int(self.out_dim),
            "n_roles": int(self.n_roles),
            "max_k_v6": int(self.max_k_v6),
            "w_role": self.w_role.to_dict(),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w54_lhr_v6_head",
            "head": self.to_dict()})


# =============================================================================
# Degradation curve
# =============================================================================


def evaluate_v6_degradation_curve(
        head: LongHorizonReconstructionV6Head,
        examples: Sequence[LongHorizonV4Example],
        *,
        k_max: int = W54_DEFAULT_LHR_V6_DEGRADATION_K_MAX,
) -> list[V5DegradationPoint]:
    out: list[V5DegradationPoint] = []
    for k in range(1, int(k_max) + 1):
        if k <= head.max_k_v6:
            mse = float(evaluate_long_horizon_v4_mse_at_k(
                head.inner_v5.inner_v4, examples, int(k)))
            out.append(V5DegradationPoint(
                k=int(k), mse=float(mse),
                is_degraded=False))
        else:
            tot = 0.0
            n = 0
            for ex in examples:
                tgt = ex.target_features
                for v in tgt:
                    tot += float(v) * float(v)
                    n += 1
            mse = float(tot) / float(max(1, n))
            out.append(V5DegradationPoint(
                k=int(k), mse=float(mse),
                is_degraded=True))
    return out


# =============================================================================
# Witness
# =============================================================================


@dataclasses.dataclass(frozen=True)
class LongHorizonReconstructionV6Witness:
    head_cid: str
    inner_v5_cid: str
    max_k: int
    n_roles: int
    mse_at_k8: float
    mse_at_k12: float
    mse_at_k16: float
    mse_at_k18: float
    mse_at_k24: float
    degradation_curve: tuple[V5DegradationPoint, ...]
    n_examples: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "head_cid": str(self.head_cid),
            "inner_v5_cid": str(self.inner_v5_cid),
            "max_k": int(self.max_k),
            "n_roles": int(self.n_roles),
            "mse_at_k8": float(round(self.mse_at_k8, 12)),
            "mse_at_k12": float(round(self.mse_at_k12, 12)),
            "mse_at_k16": float(round(self.mse_at_k16, 12)),
            "mse_at_k18": float(round(self.mse_at_k18, 12)),
            "mse_at_k24": float(round(self.mse_at_k24, 12)),
            "degradation_curve": [
                p.to_dict() for p in self.degradation_curve],
            "n_examples": int(self.n_examples),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w54_lhr_v6_witness",
            "witness": self.to_dict()})


def emit_lhr_v6_witness(
        *,
        head: LongHorizonReconstructionV6Head,
        examples: Sequence[LongHorizonV4Example],
        k_max_for_degradation: int = (
            W54_DEFAULT_LHR_V6_DEGRADATION_K_MAX),
) -> LongHorizonReconstructionV6Witness:
    if not examples:
        return LongHorizonReconstructionV6Witness(
            head_cid=str(head.cid()),
            inner_v5_cid=str(head.inner_v5.cid()),
            max_k=int(head.max_k_v6),
            n_roles=int(head.n_roles),
            mse_at_k8=0.0,
            mse_at_k12=0.0,
            mse_at_k16=0.0,
            mse_at_k18=0.0,
            mse_at_k24=0.0,
            degradation_curve=(),
            n_examples=0,
        )
    inner_v4 = head.inner_v5.inner_v4
    mse8 = float(evaluate_long_horizon_v4_mse_at_k(
        inner_v4, examples, 8))
    mse12 = float(evaluate_long_horizon_v4_mse_at_k(
        inner_v4, examples, 12))
    mse16 = float(evaluate_long_horizon_v4_mse_at_k(
        inner_v4, examples, 16))
    mse18 = (
        float(evaluate_long_horizon_v4_mse_at_k(
            inner_v4, examples, 18))
        if int(head.max_k_v6) >= 18 else 1.0)
    mse24 = (
        float(evaluate_long_horizon_v4_mse_at_k(
            inner_v4, examples, 24))
        if int(head.max_k_v6) >= 24 else 1.0)
    curve = evaluate_v6_degradation_curve(
        head, examples,
        k_max=int(k_max_for_degradation))
    return LongHorizonReconstructionV6Witness(
        head_cid=str(head.cid()),
        inner_v5_cid=str(head.inner_v5.cid()),
        max_k=int(head.max_k_v6),
        n_roles=int(head.n_roles),
        mse_at_k8=float(mse8),
        mse_at_k12=float(mse12),
        mse_at_k16=float(mse16),
        mse_at_k18=float(mse18),
        mse_at_k24=float(mse24),
        degradation_curve=tuple(curve),
        n_examples=int(len(examples)),
    )


# =============================================================================
# Verifier
# =============================================================================

W54_LHR_V6_VERIFIER_FAILURE_MODES: tuple[str, ...] = (
    "w54_lhr_v6_head_cid_mismatch",
    "w54_lhr_v6_inner_v5_cid_mismatch",
    "w54_lhr_v6_max_k_below_floor",
    "w54_lhr_v6_mse_above_ceiling",
    "w54_lhr_v6_degradation_curve_empty",
    "w54_lhr_v6_n_roles_below_floor",
)


def verify_lhr_v6_witness(
        witness: LongHorizonReconstructionV6Witness,
        *,
        expected_head_cid: str | None = None,
        expected_inner_v5_cid: str | None = None,
        min_max_k: int | None = None,
        max_mse_at_k18: float | None = None,
        min_n_roles: int | None = None,
) -> dict[str, Any]:
    failures: list[str] = []
    if (expected_head_cid is not None
            and witness.head_cid != str(expected_head_cid)):
        failures.append("w54_lhr_v6_head_cid_mismatch")
    if (expected_inner_v5_cid is not None
            and witness.inner_v5_cid
            != str(expected_inner_v5_cid)):
        failures.append("w54_lhr_v6_inner_v5_cid_mismatch")
    if (min_max_k is not None
            and witness.max_k < int(min_max_k)):
        failures.append("w54_lhr_v6_max_k_below_floor")
    if (max_mse_at_k18 is not None
            and witness.mse_at_k18 > float(max_mse_at_k18)):
        failures.append("w54_lhr_v6_mse_above_ceiling")
    if not witness.degradation_curve:
        failures.append("w54_lhr_v6_degradation_curve_empty")
    if (min_n_roles is not None
            and witness.n_roles < int(min_n_roles)):
        failures.append("w54_lhr_v6_n_roles_below_floor")
    return {
        "ok": (len(failures) == 0),
        "failures": failures,
        "witness_cid": witness.cid(),
    }


__all__ = [
    "W54_LHR_V6_SCHEMA_VERSION",
    "W54_DEFAULT_LHR_V6_FLAT_FEATURE_DIM",
    "W54_DEFAULT_LHR_V6_HIDDEN_DIM",
    "W54_DEFAULT_LHR_V6_MAX_K",
    "W54_DEFAULT_LHR_V6_DEGRADATION_K_MAX",
    "W54_DEFAULT_LHR_V6_N_BRANCHES",
    "W54_DEFAULT_LHR_V6_N_CYCLES",
    "W54_DEFAULT_LHR_V6_N_MERGE_PAIRS",
    "W54_DEFAULT_LHR_V6_N_ROLES",
    "W54_LHR_V6_VERIFIER_FAILURE_MODES",
    "LongHorizonReconstructionV6Head",
    "LongHorizonReconstructionV6Witness",
    "evaluate_v6_degradation_curve",
    "emit_lhr_v6_witness",
    "verify_lhr_v6_witness",
]
