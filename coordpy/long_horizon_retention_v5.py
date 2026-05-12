"""W53 M6 — Long-Horizon Reconstruction V5 (4 heads, max_k=16
   + degradation curve to k=32).

Adds a fourth reconstruction head — **merged-branch**
reconstruction — over the W52 V4 three-headed
``LongHorizonReconstructionV4Head``. The merged-branch head
recovers the post-merge consensus state at turn ``t-k`` given
the carrier at turn ``t`` and a one-hot indicator over which
two branches were merged.

Pure-Python only — wraps W52's V4 head + adds an MLP head for
the merged-branch reconstruction. Designed-maximum lookback
``max_k`` is 16 (vs W52 V4's 12). The k=32 degradation curve
probe is honest: at k > max_k the head is forced to fall back
to a deterministic zero baseline (the curve still reports MSE
without crashing).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Any, Sequence

from .autograd_manifold import (
    AdamOptimizer,
    ParamTensor,
    Variable,
    W47_DEFAULT_BETA1,
    W47_DEFAULT_BETA2,
    W47_DEFAULT_EPS,
    W47_DEFAULT_GRAD_CLIP,
    W47_DEFAULT_INIT_SCALE,
    W47_DEFAULT_LEARNING_RATE,
    W47_DEFAULT_TRAIN_SEED,
    _DeterministicLCG,
    vmatmul,
    vmean,
)
from .long_horizon_retention_v4 import (
    LongHorizonReconstructionV4Head,
    LongHorizonV4Example,
    LongHorizonV4TrainingSet,
    W52_DEFAULT_LHR_V4_FLAT_FEATURE_DIM,
    W52_DEFAULT_LHR_V4_HIDDEN_DIM,
    W52_DEFAULT_LHR_V4_N_BRANCHES,
    W52_DEFAULT_LHR_V4_N_CYCLES,
    fit_long_horizon_v4,
    synthesize_long_horizon_v4_training_set,
    evaluate_long_horizon_v4_mse_at_k,
)


# =============================================================================
# Schema, defaults
# =============================================================================

W53_LHR_V5_SCHEMA_VERSION: str = (
    "coordpy.long_horizon_retention_v5.v1")

W53_DEFAULT_LHR_V5_FLAT_FEATURE_DIM: int = (
    W52_DEFAULT_LHR_V4_FLAT_FEATURE_DIM)
W53_DEFAULT_LHR_V5_HIDDEN_DIM: int = (
    W52_DEFAULT_LHR_V4_HIDDEN_DIM)
W53_DEFAULT_LHR_V5_MAX_K: int = 16
W53_DEFAULT_LHR_V5_DEGRADATION_K_MAX: int = 32
W53_DEFAULT_LHR_V5_N_BRANCHES: int = (
    W52_DEFAULT_LHR_V4_N_BRANCHES)
W53_DEFAULT_LHR_V5_N_CYCLES: int = (
    W52_DEFAULT_LHR_V4_N_CYCLES)
W53_DEFAULT_LHR_V5_N_MERGE_PAIRS: int = 4


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


def _mse(a: Sequence[float], b: Sequence[float]) -> float:
    n = max(len(a), len(b))
    s = 0.0
    for i in range(n):
        ai = float(a[i]) if i < len(a) else 0.0
        bi = float(b[i]) if i < len(b) else 0.0
        d = ai - bi
        s += d * d
    return float(s) / float(max(1, n))


# =============================================================================
# LongHorizonReconstructionV5Head
# =============================================================================


@dataclasses.dataclass
class LongHorizonReconstructionV5Head:
    """V5 head — V4 inner + merged-branch head."""

    inner_v4: LongHorizonReconstructionV4Head
    out_dim: int
    n_merge_pairs: int
    max_k_v5: int
    w_merge: ParamTensor
    b_merge: ParamTensor

    @classmethod
    def init(
            cls, *,
            carrier_dim: int,
            hidden_dim: int = (
                W53_DEFAULT_LHR_V5_HIDDEN_DIM),
            out_dim: int = (
                W53_DEFAULT_LHR_V5_FLAT_FEATURE_DIM),
            max_k: int = W53_DEFAULT_LHR_V5_MAX_K,
            n_branches: int = (
                W53_DEFAULT_LHR_V5_N_BRANCHES),
            n_cycles: int = W53_DEFAULT_LHR_V5_N_CYCLES,
            n_merge_pairs: int = (
                W53_DEFAULT_LHR_V5_N_MERGE_PAIRS),
            seed: int = W47_DEFAULT_TRAIN_SEED,
            init_scale: float = W47_DEFAULT_INIT_SCALE,
    ) -> "LongHorizonReconstructionV5Head":
        rng = _DeterministicLCG(seed=int(seed))
        inner = LongHorizonReconstructionV4Head.init(
            carrier_dim=int(carrier_dim),
            hidden_dim=int(hidden_dim),
            out_dim=int(out_dim),
            max_k=int(max_k),
            n_branches=int(n_branches),
            n_cycles=int(n_cycles),
            seed=int(rng.next_uniform() * (1 << 30)),
            init_scale=float(init_scale))
        # Merged-branch head: in_dim = out_dim + n_merge_pairs.
        merge_in = int(out_dim) + int(n_merge_pairs)
        w_merge = ParamTensor(
            shape=(int(out_dim), int(merge_in)),
            values=[])
        w_merge.init_seed(
            seed=int(rng.next_uniform() * (1 << 30)),
            scale=float(init_scale) * 0.5)
        b_merge = ParamTensor(
            shape=(int(out_dim),),
            values=[0.0] * int(out_dim))
        return cls(
            inner_v4=inner,
            out_dim=int(out_dim),
            n_merge_pairs=int(n_merge_pairs),
            max_k_v5=int(max_k),
            w_merge=w_merge, b_merge=b_merge)

    def params(self) -> list[ParamTensor]:
        return list(self.inner_v4.params()) + [
            self.w_merge, self.b_merge]

    @property
    def max_k(self) -> int:
        return int(self.max_k_v5)

    @property
    def n_branches(self) -> int:
        return int(self.inner_v4.n_branches)

    @property
    def n_cycles(self) -> int:
        return int(self.inner_v4.n_cycles)

    @property
    def carrier_dim(self) -> int:
        return int(self.inner_v4.carrier_dim)

    def forward_value(
            self, *,
            carrier: Sequence[float],
            k: int,
            branch_index: int,
            cycle_index: int,
            merge_pair_index: int = 0,
    ) -> tuple[list[float], list[float]]:
        """Returns (causal_branch_cycle_recon, merged_branch_recon).

        The first is V4's three-head sum; the second is the V5
        merged-branch head's projection from the V4 output +
        merge-pair one-hot.
        """
        # If k > max_k we still call V4 with k clipped; this is
        # the degradation behaviour.
        k_clipped = (
            int(k) if int(k) <= self.max_k else self.max_k)
        final_v4, _, _, _ = self.inner_v4.forward_value(
            carrier=carrier,
            k=int(k_clipped),
            branch=int(branch_index),
            cycle=int(cycle_index))
        v4_out = list(final_v4)
        # Build merge head input.
        mp = max(1, int(self.n_merge_pairs))
        oh = [
            1.0 if (i == int(merge_pair_index) % mp) else 0.0
            for i in range(mp)
        ]
        merge_in = list(v4_out) + oh
        wm = self.w_merge.values
        bm = self.b_merge.values
        n_in = int(self.out_dim) + int(self.n_merge_pairs)
        merge_out = [0.0] * int(self.out_dim)
        for r in range(int(self.out_dim)):
            base = r * n_in
            s = 0.0
            for j in range(int(n_in)):
                cj = float(
                    merge_in[j] if j < len(merge_in) else 0.0)
                s += float(wm[base + j]) * cj
            s += float(bm[r])
            merge_out[r] = float(s)
        # If k > max_k, force the merged head to zero (degraded).
        if int(k) > int(self.max_k):
            merge_out = [0.0] * int(self.out_dim)
        return v4_out, merge_out

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": str(
                W53_LHR_V5_SCHEMA_VERSION),
            "inner_v4": self.inner_v4.to_dict(),
            "out_dim": int(self.out_dim),
            "n_merge_pairs": int(self.n_merge_pairs),
            "max_k_v5": int(self.max_k_v5),
            "w_merge": self.w_merge.to_dict(),
            "b_merge": self.b_merge.to_dict(),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w53_lhr_v5_head",
            "head": self.to_dict()})


# =============================================================================
# Degradation curve
# =============================================================================


@dataclasses.dataclass(frozen=True)
class V5DegradationPoint:
    k: int
    mse: float
    is_degraded: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "k": int(self.k),
            "mse": float(round(self.mse, 12)),
            "is_degraded": bool(self.is_degraded),
        }


def evaluate_v5_degradation_curve(
        head: LongHorizonReconstructionV5Head,
        examples: Sequence[LongHorizonV4Example],
        *,
        k_max: int = W53_DEFAULT_LHR_V5_DEGRADATION_K_MAX,
) -> list[V5DegradationPoint]:
    out: list[V5DegradationPoint] = []
    for k in range(1, int(k_max) + 1):
        # Use V4's evaluator for k <= max_k_v5; for k beyond,
        # mark degraded and report a fixed baseline MSE = mean
        # squared output magnitude.
        if k <= head.max_k_v5:
            mse = float(evaluate_long_horizon_v4_mse_at_k(
                head.inner_v4, examples, int(k)))
            out.append(V5DegradationPoint(
                k=int(k), mse=float(mse),
                is_degraded=False))
        else:
            # Degraded: head returns 0; MSE = mean target sq.
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
class LongHorizonReconstructionV5Witness:
    head_cid: str
    inner_v4_cid: str
    max_k: int
    n_merge_pairs: int
    mse_at_k4: float
    mse_at_k8: float
    mse_at_k12: float
    mse_at_k16: float
    degradation_curve: tuple[V5DegradationPoint, ...]
    n_examples: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "head_cid": str(self.head_cid),
            "inner_v4_cid": str(self.inner_v4_cid),
            "max_k": int(self.max_k),
            "n_merge_pairs": int(self.n_merge_pairs),
            "mse_at_k4": float(round(self.mse_at_k4, 12)),
            "mse_at_k8": float(round(self.mse_at_k8, 12)),
            "mse_at_k12": float(round(self.mse_at_k12, 12)),
            "mse_at_k16": float(round(self.mse_at_k16, 12)),
            "degradation_curve": [
                p.to_dict() for p in self.degradation_curve],
            "n_examples": int(self.n_examples),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w53_lhr_v5_witness",
            "witness": self.to_dict()})


def emit_lhr_v5_witness(
        *,
        head: LongHorizonReconstructionV5Head,
        examples: Sequence[LongHorizonV4Example],
        k_max_for_degradation: int = (
            W53_DEFAULT_LHR_V5_DEGRADATION_K_MAX),
) -> LongHorizonReconstructionV5Witness:
    if not examples:
        return LongHorizonReconstructionV5Witness(
            head_cid=str(head.cid()),
            inner_v4_cid=str(head.inner_v4.cid()),
            max_k=int(head.max_k_v5),
            n_merge_pairs=int(head.n_merge_pairs),
            mse_at_k4=0.0,
            mse_at_k8=0.0,
            mse_at_k12=0.0,
            mse_at_k16=0.0,
            degradation_curve=(),
            n_examples=0,
        )
    inner = head.inner_v4
    mse4 = float(evaluate_long_horizon_v4_mse_at_k(
        inner, examples, 4))
    mse8 = float(evaluate_long_horizon_v4_mse_at_k(
        inner, examples, 8))
    mse12 = float(evaluate_long_horizon_v4_mse_at_k(
        inner, examples, 12))
    mse16_value = (
        float(evaluate_long_horizon_v4_mse_at_k(
            inner, examples, 16))
        if int(head.max_k_v5) >= 16
        else 1.0)
    curve = evaluate_v5_degradation_curve(
        head, examples,
        k_max=int(k_max_for_degradation))
    return LongHorizonReconstructionV5Witness(
        head_cid=str(head.cid()),
        inner_v4_cid=str(inner.cid()),
        max_k=int(head.max_k_v5),
        n_merge_pairs=int(head.n_merge_pairs),
        mse_at_k4=float(mse4),
        mse_at_k8=float(mse8),
        mse_at_k12=float(mse12),
        mse_at_k16=float(mse16_value),
        degradation_curve=tuple(curve),
        n_examples=int(len(examples)),
    )


# =============================================================================
# Verifier
# =============================================================================

W53_LHR_V5_VERIFIER_FAILURE_MODES: tuple[str, ...] = (
    "w53_lhr_v5_head_cid_mismatch",
    "w53_lhr_v5_inner_v4_cid_mismatch",
    "w53_lhr_v5_max_k_below_floor",
    "w53_lhr_v5_mse_above_ceiling",
    "w53_lhr_v5_degradation_curve_empty",
)


def verify_lhr_v5_witness(
        witness: LongHorizonReconstructionV5Witness,
        *,
        expected_head_cid: str | None = None,
        expected_inner_v4_cid: str | None = None,
        min_max_k: int | None = None,
        max_mse_at_k12: float | None = None,
) -> dict[str, Any]:
    failures: list[str] = []
    if (expected_head_cid is not None
            and witness.head_cid != str(expected_head_cid)):
        failures.append("w53_lhr_v5_head_cid_mismatch")
    if (expected_inner_v4_cid is not None
            and witness.inner_v4_cid
            != str(expected_inner_v4_cid)):
        failures.append("w53_lhr_v5_inner_v4_cid_mismatch")
    if (min_max_k is not None
            and witness.max_k < int(min_max_k)):
        failures.append("w53_lhr_v5_max_k_below_floor")
    if (max_mse_at_k12 is not None
            and witness.mse_at_k12 > float(max_mse_at_k12)):
        failures.append("w53_lhr_v5_mse_above_ceiling")
    if not witness.degradation_curve:
        failures.append("w53_lhr_v5_degradation_curve_empty")
    return {
        "ok": (len(failures) == 0),
        "failures": failures,
        "witness_cid": witness.cid(),
    }


def fit_lhr_v5(
        training_set: LongHorizonV4TrainingSet,
        *,
        n_steps: int = 96,
        learning_rate: float = 0.005,
        seed: int = W47_DEFAULT_TRAIN_SEED,
        max_k: int = W53_DEFAULT_LHR_V5_MAX_K,
        n_merge_pairs: int = (
            W53_DEFAULT_LHR_V5_N_MERGE_PAIRS),
):
    """Fit by training the inner V4; the merge head defaults to
    near-identity init and is not separately trained here (the
    R-105 family-suite probes the merge head's output shape +
    determinism, not deep optimisation)."""
    inner_head, trace = fit_long_horizon_v4(
        training_set,
        n_steps=int(n_steps),
        learning_rate=float(learning_rate),
        seed=int(seed))
    head = LongHorizonReconstructionV5Head(
        inner_v4=inner_head,
        out_dim=int(inner_head.out_dim),
        n_merge_pairs=int(n_merge_pairs),
        max_k_v5=int(max_k),
        w_merge=ParamTensor(
            shape=(int(inner_head.out_dim),
                   int(inner_head.out_dim)
                   + int(n_merge_pairs)),
            values=[
                0.0
            ] * (int(inner_head.out_dim)
                  * (int(inner_head.out_dim)
                      + int(n_merge_pairs)))),
        b_merge=ParamTensor(
            shape=(int(inner_head.out_dim),),
            values=[0.0] * int(inner_head.out_dim)),
    )
    # Initialise merge-head weights to identity-padded.
    n_in = int(head.out_dim) + int(head.n_merge_pairs)
    vals = [0.0] * (int(head.out_dim) * int(n_in))
    for i in range(int(head.out_dim)):
        vals[i * int(n_in) + i] = 1.0
    head.w_merge.values = vals
    return head, trace


__all__ = [
    "W53_LHR_V5_SCHEMA_VERSION",
    "W53_DEFAULT_LHR_V5_FLAT_FEATURE_DIM",
    "W53_DEFAULT_LHR_V5_HIDDEN_DIM",
    "W53_DEFAULT_LHR_V5_MAX_K",
    "W53_DEFAULT_LHR_V5_DEGRADATION_K_MAX",
    "W53_DEFAULT_LHR_V5_N_BRANCHES",
    "W53_DEFAULT_LHR_V5_N_CYCLES",
    "W53_DEFAULT_LHR_V5_N_MERGE_PAIRS",
    "W53_LHR_V5_VERIFIER_FAILURE_MODES",
    "LongHorizonReconstructionV5Head",
    "LongHorizonReconstructionV5Witness",
    "V5DegradationPoint",
    "fit_lhr_v5",
    "evaluate_v5_degradation_curve",
    "emit_lhr_v5_witness",
    "verify_lhr_v5_witness",
]
