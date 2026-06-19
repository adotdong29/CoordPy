"""W55 M7 — Long-Horizon Reconstruction V7.

Adds a sixth head (cross-cycle) to W54 V6's 5-head reconstruction
and stretches ``max_k`` to 36 (vs V6's 24) with degradation curve
probe to ``k=72``.

The 6th head reconstructs ``t-k`` for a *different* cycle index
than the producer cycle — a stronger "reach across cycles" test
than V6's cross-role.

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
from .long_horizon_retention_v6 import (
    LongHorizonReconstructionV6Head,
    LongHorizonReconstructionV6Witness,
    W54_DEFAULT_LHR_V6_FLAT_FEATURE_DIM,
    W54_DEFAULT_LHR_V6_HIDDEN_DIM,
    W54_DEFAULT_LHR_V6_MAX_K,
    W54_DEFAULT_LHR_V6_N_BRANCHES,
    W54_DEFAULT_LHR_V6_N_CYCLES,
    W54_DEFAULT_LHR_V6_N_MERGE_PAIRS,
    W54_DEFAULT_LHR_V6_N_ROLES,
)


# =============================================================================
# Schema, defaults
# =============================================================================

W55_LHR_V7_SCHEMA_VERSION: str = (
    "coordpy.long_horizon_retention_v7.v1")

W55_DEFAULT_LHR_V7_FLAT_FEATURE_DIM: int = (
    W54_DEFAULT_LHR_V6_FLAT_FEATURE_DIM)
W55_DEFAULT_LHR_V7_HIDDEN_DIM: int = (
    W54_DEFAULT_LHR_V6_HIDDEN_DIM)
W55_DEFAULT_LHR_V7_MAX_K: int = 36
W55_DEFAULT_LHR_V7_DEGRADATION_K_MAX: int = 72
W55_DEFAULT_LHR_V7_N_BRANCHES: int = (
    W54_DEFAULT_LHR_V6_N_BRANCHES)
W55_DEFAULT_LHR_V7_N_CYCLES: int = (
    W54_DEFAULT_LHR_V6_N_CYCLES)
W55_DEFAULT_LHR_V7_N_MERGE_PAIRS: int = (
    W54_DEFAULT_LHR_V6_N_MERGE_PAIRS)
W55_DEFAULT_LHR_V7_N_ROLES: int = (
    W54_DEFAULT_LHR_V6_N_ROLES)


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
# LongHorizonReconstructionV7Head
# =============================================================================


@dataclasses.dataclass
class LongHorizonReconstructionV7Head:
    """V7 head — V6 inner + cross-cycle head + max_k=36."""

    inner_v6: LongHorizonReconstructionV6Head
    out_dim: int
    n_cycles: int
    max_k_v7: int
    w_cross_cycle: ParamTensor

    @classmethod
    def init(
            cls, *,
            carrier_dim: int,
            hidden_dim: int = (
                W55_DEFAULT_LHR_V7_HIDDEN_DIM),
            out_dim: int = (
                W55_DEFAULT_LHR_V7_FLAT_FEATURE_DIM),
            max_k: int = W55_DEFAULT_LHR_V7_MAX_K,
            n_branches: int = (
                W55_DEFAULT_LHR_V7_N_BRANCHES),
            n_cycles: int = W55_DEFAULT_LHR_V7_N_CYCLES,
            n_merge_pairs: int = (
                W55_DEFAULT_LHR_V7_N_MERGE_PAIRS),
            n_roles: int = W55_DEFAULT_LHR_V7_N_ROLES,
            seed: int = W47_DEFAULT_TRAIN_SEED,
            init_scale: float = W47_DEFAULT_INIT_SCALE,
    ) -> "LongHorizonReconstructionV7Head":
        rng = _DeterministicLCG(seed=int(seed) + 131)
        inner_max_k = max(24, int(max_k))
        inner = LongHorizonReconstructionV6Head.init(
            carrier_dim=int(carrier_dim),
            hidden_dim=int(hidden_dim),
            out_dim=int(out_dim),
            max_k=int(inner_max_k),
            n_branches=int(n_branches),
            n_cycles=int(n_cycles),
            n_merge_pairs=int(n_merge_pairs),
            n_roles=int(n_roles),
            seed=int(rng.next_uniform() * (1 << 30)),
            init_scale=float(init_scale))
        cc_in = int(out_dim) + int(n_cycles)
        w_cc = ParamTensor(
            shape=(int(out_dim), int(cc_in)),
            values=[])
        w_cc.init_seed(
            seed=int(rng.next_uniform() * (1 << 30)),
            scale=float(init_scale) * 0.5)
        return cls(
            inner_v6=inner,
            out_dim=int(out_dim),
            n_cycles=int(n_cycles),
            max_k_v7=int(max_k),
            w_cross_cycle=w_cc,
        )

    def params(self) -> list[ParamTensor]:
        return list(self.inner_v6.params()) + [
            self.w_cross_cycle]

    @property
    def max_k(self) -> int:
        return int(self.max_k_v7)

    @property
    def n_branches(self) -> int:
        return int(self.inner_v6.n_branches)

    @property
    def carrier_dim(self) -> int:
        return int(self.inner_v6.carrier_dim)

    def forward_value(
            self, *,
            carrier: Sequence[float],
            k: int,
            branch_index: int,
            cycle_index: int,
            merge_pair_index: int = 0,
            role_index: int = 0,
            target_cycle_index: int = 0,
    ) -> tuple[list[float], list[float], list[float],
                list[float], list[float]]:
        """Return (v6_main, merged_branch, cross_role, cross_cycle,
        degradation)."""
        k_clipped_inner = (
            int(k) if int(k) <= self.inner_v6.max_k
            else self.inner_v6.max_k)
        v6_main, merge_out, cross_role, degradation = (
            self.inner_v6.forward_value(
                carrier=carrier,
                k=int(k_clipped_inner),
                branch_index=int(branch_index),
                cycle_index=int(cycle_index),
                merge_pair_index=int(merge_pair_index),
                role_index=int(role_index)))
        # Cross-cycle head: projects (cross_role concat cycle_one_hot)
        # to out.
        nc = max(1, int(self.n_cycles))
        oh = [
            1.0 if (i == int(target_cycle_index) % nc)
            else 0.0
            for i in range(nc)
        ]
        cc_in = list(cross_role) + oh
        wc = self.w_cross_cycle.values
        n_in = int(self.out_dim) + int(self.n_cycles)
        cross_cycle = [0.0] * int(self.out_dim)
        for r in range(int(self.out_dim)):
            s = 0.0
            for j in range(n_in):
                s += float(
                    wc[r * n_in + j]) * float(
                        cc_in[j] if j < len(cc_in) else 0.0)
            cross_cycle[r] = math.tanh(s)
        return (
            v6_main, merge_out, cross_role, cross_cycle,
            degradation)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": str(
                W55_LHR_V7_SCHEMA_VERSION),
            "inner_v6_cid": str(self.inner_v6.cid()),
            "out_dim": int(self.out_dim),
            "n_cycles": int(self.n_cycles),
            "max_k_v7": int(self.max_k_v7),
            "w_cross_cycle": self.w_cross_cycle.to_dict(),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w55_lhr_v7_head",
            "head": self.to_dict()})


# =============================================================================
# Degradation probe
# =============================================================================


@dataclasses.dataclass(frozen=True)
class V7DegradationPoint:
    k: int
    mse_main: float
    mse_merged: float
    mse_cross_role: float
    mse_cross_cycle: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "k": int(self.k),
            "mse_main": float(round(self.mse_main, 12)),
            "mse_merged": float(round(self.mse_merged, 12)),
            "mse_cross_role": float(round(
                self.mse_cross_role, 12)),
            "mse_cross_cycle": float(round(
                self.mse_cross_cycle, 12)),
        }


def probe_v7_degradation_curve(
        head: LongHorizonReconstructionV7Head,
        examples: Sequence[LongHorizonV4Example],
        *,
        k_max: int = W55_DEFAULT_LHR_V7_DEGRADATION_K_MAX,
) -> list[V7DegradationPoint]:
    """Probe MSE@k across k ∈ {1, ..., k_max}."""
    out: list[V7DegradationPoint] = []
    if not examples:
        return out
    for k in range(1, int(k_max) + 1):
        mses_main: list[float] = []
        mses_merge: list[float] = []
        mses_cross_role: list[float] = []
        mses_cross_cycle: list[float] = []
        for ex in examples:
            main, merge, cross_role, cross_cycle, _ = (
                head.forward_value(
                    carrier=ex.carrier,
                    k=int(k),
                    branch_index=int(ex.branch_index),
                    cycle_index=int(ex.cycle_index),
                    merge_pair_index=0,
                    role_index=0,
                    target_cycle_index=0))
            tgt = ex.target_features
            mses_main.append(
                sum(
                    (float(main[i] if i < len(main) else 0.0)
                     - float(tgt[i] if i < len(tgt) else 0.0))
                    ** 2
                    for i in range(max(len(main), len(tgt))))
                / float(max(1, len(tgt))))
            mses_merge.append(
                sum(
                    (float(merge[i] if i < len(merge) else 0.0)
                     - float(tgt[i] if i < len(tgt) else 0.0))
                    ** 2
                    for i in range(max(len(merge), len(tgt))))
                / float(max(1, len(tgt))))
            mses_cross_role.append(
                sum(
                    (float(cross_role[i]
                            if i < len(cross_role) else 0.0)
                     - float(tgt[i] if i < len(tgt) else 0.0))
                    ** 2
                    for i in range(
                        max(len(cross_role), len(tgt))))
                / float(max(1, len(tgt))))
            mses_cross_cycle.append(
                sum(
                    (float(cross_cycle[i]
                            if i < len(cross_cycle) else 0.0)
                     - float(tgt[i] if i < len(tgt) else 0.0))
                    ** 2
                    for i in range(
                        max(len(cross_cycle), len(tgt))))
                / float(max(1, len(tgt))))
        out.append(V7DegradationPoint(
            k=int(k),
            mse_main=float(
                sum(mses_main) / max(1, len(mses_main))),
            mse_merged=float(
                sum(mses_merge) / max(1, len(mses_merge))),
            mse_cross_role=float(
                sum(mses_cross_role)
                / max(1, len(mses_cross_role))),
            mse_cross_cycle=float(
                sum(mses_cross_cycle)
                / max(1, len(mses_cross_cycle))),
        ))
    return out


# =============================================================================
# Witness
# =============================================================================


@dataclasses.dataclass(frozen=True)
class LongHorizonReconstructionV7Witness:
    head_cid: str
    n_examples: int
    max_k: int
    main_mse_at_k_max: float
    merged_mse_at_k_max: float
    cross_role_mse_at_k_max: float
    cross_cycle_mse_at_k_max: float
    degradation_curve: tuple[V7DegradationPoint, ...]
    min_mse_in_range: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "head_cid": str(self.head_cid),
            "n_examples": int(self.n_examples),
            "max_k": int(self.max_k),
            "main_mse_at_k_max": float(round(
                self.main_mse_at_k_max, 12)),
            "merged_mse_at_k_max": float(round(
                self.merged_mse_at_k_max, 12)),
            "cross_role_mse_at_k_max": float(round(
                self.cross_role_mse_at_k_max, 12)),
            "cross_cycle_mse_at_k_max": float(round(
                self.cross_cycle_mse_at_k_max, 12)),
            "degradation_curve": [
                p.to_dict()
                for p in self.degradation_curve],
            "min_mse_in_range": float(round(
                self.min_mse_in_range, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w55_lhr_v7_witness",
            "witness": self.to_dict()})


def emit_lhr_v7_witness(
        *,
        head: LongHorizonReconstructionV7Head,
        examples: Sequence[LongHorizonV4Example],
        k_max_for_degradation: int = (
            W55_DEFAULT_LHR_V7_DEGRADATION_K_MAX),
) -> LongHorizonReconstructionV7Witness:
    curve = probe_v7_degradation_curve(
        head, examples,
        k_max=int(k_max_for_degradation))
    max_k = head.max_k
    # Compute MSE at k_max.
    if curve:
        # Find the curve point at k closest to max_k.
        target = curve[-1]
        for p in curve:
            if p.k == max_k:
                target = p
                break
        m_main = target.mse_main
        m_merge = target.mse_merged
        m_role = target.mse_cross_role
        m_cycle = target.mse_cross_cycle
        min_mse = min(
            min(p.mse_main, p.mse_merged,
                 p.mse_cross_role, p.mse_cross_cycle)
            for p in curve
            if p.k <= max_k
        )
    else:
        m_main = m_merge = m_role = m_cycle = min_mse = 0.0
    return LongHorizonReconstructionV7Witness(
        head_cid=str(head.cid()),
        n_examples=int(len(examples)),
        max_k=int(max_k),
        main_mse_at_k_max=float(m_main),
        merged_mse_at_k_max=float(m_merge),
        cross_role_mse_at_k_max=float(m_role),
        cross_cycle_mse_at_k_max=float(m_cycle),
        degradation_curve=tuple(curve),
        min_mse_in_range=float(min_mse),
    )


# =============================================================================
# Verifier
# =============================================================================

W55_LHR_V7_VERIFIER_FAILURE_MODES: tuple[str, ...] = (
    "w55_lhr_v7_head_cid_mismatch",
    "w55_lhr_v7_mse_negative",
    "w55_lhr_v7_max_k_below_floor",
    "w55_lhr_v7_min_mse_above_ceiling",
)


def verify_lhr_v7_witness(
        witness: LongHorizonReconstructionV7Witness,
        *,
        expected_head_cid: str | None = None,
        max_min_mse: float | None = None,
        min_max_k: int | None = None,
) -> dict[str, Any]:
    failures: list[str] = []
    if (expected_head_cid is not None
            and witness.head_cid != str(expected_head_cid)):
        failures.append("w55_lhr_v7_head_cid_mismatch")
    for v in (
            witness.main_mse_at_k_max,
            witness.merged_mse_at_k_max,
            witness.cross_role_mse_at_k_max,
            witness.cross_cycle_mse_at_k_max,
            witness.min_mse_in_range):
        if float(v) < 0.0:
            failures.append("w55_lhr_v7_mse_negative")
            break
    if (min_max_k is not None
            and witness.max_k < int(min_max_k)):
        failures.append("w55_lhr_v7_max_k_below_floor")
    if (max_min_mse is not None
            and witness.min_mse_in_range
            > float(max_min_mse)):
        failures.append(
            "w55_lhr_v7_min_mse_above_ceiling")
    return {
        "ok": (len(failures) == 0),
        "failures": failures,
        "witness_cid": witness.cid(),
    }


__all__ = [
    "W55_LHR_V7_SCHEMA_VERSION",
    "W55_DEFAULT_LHR_V7_FLAT_FEATURE_DIM",
    "W55_DEFAULT_LHR_V7_HIDDEN_DIM",
    "W55_DEFAULT_LHR_V7_MAX_K",
    "W55_DEFAULT_LHR_V7_DEGRADATION_K_MAX",
    "W55_DEFAULT_LHR_V7_N_BRANCHES",
    "W55_DEFAULT_LHR_V7_N_CYCLES",
    "W55_DEFAULT_LHR_V7_N_MERGE_PAIRS",
    "W55_DEFAULT_LHR_V7_N_ROLES",
    "W55_LHR_V7_VERIFIER_FAILURE_MODES",
    "LongHorizonReconstructionV7Head",
    "LongHorizonReconstructionV7Witness",
    "V7DegradationPoint",
    "probe_v7_degradation_curve",
    "emit_lhr_v7_witness",
    "verify_lhr_v7_witness",
]
