"""W55 M11 — Disagreement Algebra.

A new first-class capsule-native module exposing three operators
over latent state capsule payloads:

* **⊕ (merge)** — content-addressed convex blend of two payloads;
  idempotent on equal inputs (`a ⊕ a = a`).
* **⊖ (difference)** — per-dim absolute disagreement vector;
  self-cancellation (`a ⊖ a = 0`).
* **⊗ (intersection-of-agreement)** — masks dimensions where
  ``|a_i - b_i| ≤ agreement_floor``; emits the mean of the
  agreeing dims and zeros elsewhere. Distributive on the agreement
  subspace: ``(a ⊕ b) ⊗ c = (a ⊗ c) ⊕ (b ⊗ c)`` exactly when
  the agreement subspace of ``a`` and ``b`` contains the relevant
  components.

Honest scope: pure-Python only, capsule-layer only, no
transformer-internal state. Operates on fixed-dim payloads
(typed as tuples of floats) and is content-addressed at every
op so audit walks reconstruct the algebra trace.

The module also exposes a small calculus of named identities
that can be checked at runtime:

* ``check_merge_idempotent(a)`` — verifies ``a ⊕ a == a``.
* ``check_difference_self_cancellation(a)`` — verifies
  ``a ⊖ a`` is the zero vector (within ``epsilon``).
* ``check_intersection_distributivity_on_agreement(a, b, c)`` —
  verifies ``(a ⊕ b) ⊗ c == (a ⊗ c) ⊕ (b ⊗ c)`` componentwise
  on the agreement subspace of ``a`` and ``b``.

W55-L-ALGEBRA-IDENTITIES-ARE-EXACT-ONLY-ON-AGREEMENT: the
distributivity identity is exact only on the agreement subspace;
off it, the two sides agree on supported components but disagree
on disagreement components. This is documented honestly and
falsified by the third check returning the per-dim residual.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Any, Sequence


# =============================================================================
# Schema, defaults
# =============================================================================

W55_DISAGREEMENT_ALGEBRA_SCHEMA_VERSION: str = (
    "coordpy.disagreement_algebra.v1")

W55_DEFAULT_AGREEMENT_FLOOR: float = 0.1
W55_DEFAULT_ALGEBRA_EPSILON: float = 1e-9


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


def _pad_to(
        values: Sequence[float], n: int,
) -> list[float]:
    out = [
        float(values[i] if i < len(values) else 0.0)
        for i in range(int(n))
    ]
    return out


# =============================================================================
# Algebra ops (value-level, no autograd needed)
# =============================================================================


def merge_op(
        a: Sequence[float],
        b: Sequence[float],
        *,
        weight_a: float = 0.5,
        weight_b: float = 0.5,
) -> list[float]:
    """⊕ — convex combine; idempotent on a == b."""
    n = max(len(a), len(b))
    wa = float(weight_a)
    wb = float(weight_b)
    s = wa + wb
    if s <= 1e-30:
        wa = 0.5
        wb = 0.5
    else:
        wa /= s
        wb /= s
    ai = _pad_to(a, n)
    bi = _pad_to(b, n)
    return [wa * ai[i] + wb * bi[i] for i in range(n)]


def difference_op(
        a: Sequence[float],
        b: Sequence[float],
) -> list[float]:
    """⊖ — per-dim absolute disagreement; self-cancels (a ⊖ a = 0)."""
    n = max(len(a), len(b))
    ai = _pad_to(a, n)
    bi = _pad_to(b, n)
    return [abs(ai[i] - bi[i]) for i in range(n)]


def intersection_op(
        a: Sequence[float],
        b: Sequence[float],
        *,
        agreement_floor: float = W55_DEFAULT_AGREEMENT_FLOOR,
) -> tuple[list[float], list[int]]:
    """⊗ — masks to agreement subspace; emits (vec, mask) pair.

    ``vec[i] = mean(a[i], b[i])`` iff ``|a[i] - b[i]| <= floor``,
    else ``vec[i] = 0`` and ``mask[i] = 0``.
    """
    n = max(len(a), len(b))
    ai = _pad_to(a, n)
    bi = _pad_to(b, n)
    out = [0.0] * n
    mask = [0] * n
    floor = float(agreement_floor)
    for i in range(n):
        if abs(ai[i] - bi[i]) <= floor:
            out[i] = (ai[i] + bi[i]) * 0.5
            mask[i] = 1
    return out, mask


def disagreement_norm(values: Sequence[float]) -> float:
    """L2 norm of a disagreement vector."""
    s = 0.0
    for v in values:
        s += float(v) * float(v)
    return float(math.sqrt(s))


def disagreement_max(values: Sequence[float]) -> float:
    """Linf norm of a disagreement vector."""
    if not values:
        return 0.0
    return float(max(abs(float(v)) for v in values))


# =============================================================================
# AlgebraTrace — content-addressed audit of algebra ops
# =============================================================================


@dataclasses.dataclass(frozen=True)
class AlgebraStep:
    """One algebra step recorded for audit."""

    op_kind: str
    inputs_payload: tuple[tuple[float, ...], ...]
    output_payload: tuple[float, ...]
    extras: tuple[tuple[str, str], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "op_kind": str(self.op_kind),
            "inputs_payload": [
                list(_round_floats(p)) for p in self.inputs_payload],
            "output_payload": list(_round_floats(
                self.output_payload)),
            "extras": [
                [str(k), str(v)] for (k, v) in self.extras],
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w55_algebra_step",
            "step": self.to_dict()})


@dataclasses.dataclass
class AlgebraTrace:
    """Audit trail of algebra operations."""

    steps: list[AlgebraStep]

    @classmethod
    def empty(cls) -> "AlgebraTrace":
        return cls(steps=[])

    def add(self, step: AlgebraStep) -> None:
        self.steps.append(step)

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w55_algebra_trace",
            "steps": [s.to_dict() for s in self.steps],
        })

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": str(
                W55_DISAGREEMENT_ALGEBRA_SCHEMA_VERSION),
            "n_steps": int(len(self.steps)),
            "trace_cid": str(self.cid()),
        }


def merge_op_traced(
        a: Sequence[float],
        b: Sequence[float],
        *,
        weight_a: float = 0.5,
        weight_b: float = 0.5,
        trace: AlgebraTrace,
) -> list[float]:
    out = merge_op(a, b, weight_a=weight_a, weight_b=weight_b)
    trace.add(AlgebraStep(
        op_kind="merge",
        inputs_payload=(
            tuple(_round_floats(a)),
            tuple(_round_floats(b))),
        output_payload=tuple(_round_floats(out)),
        extras=(
            ("weight_a", str(round(float(weight_a), 12))),
            ("weight_b", str(round(float(weight_b), 12))),
        ),
    ))
    return out


def difference_op_traced(
        a: Sequence[float],
        b: Sequence[float],
        *,
        trace: AlgebraTrace,
) -> list[float]:
    out = difference_op(a, b)
    trace.add(AlgebraStep(
        op_kind="difference",
        inputs_payload=(
            tuple(_round_floats(a)),
            tuple(_round_floats(b))),
        output_payload=tuple(_round_floats(out)),
        extras=(
            ("l2", str(round(disagreement_norm(out), 12))),
        ),
    ))
    return out


def intersection_op_traced(
        a: Sequence[float],
        b: Sequence[float],
        *,
        agreement_floor: float = W55_DEFAULT_AGREEMENT_FLOOR,
        trace: AlgebraTrace,
) -> tuple[list[float], list[int]]:
    out, mask = intersection_op(
        a, b, agreement_floor=agreement_floor)
    trace.add(AlgebraStep(
        op_kind="intersection",
        inputs_payload=(
            tuple(_round_floats(a)),
            tuple(_round_floats(b))),
        output_payload=tuple(_round_floats(out)),
        extras=(
            ("agreement_floor", str(round(
                float(agreement_floor), 12))),
            ("mask_sum", str(int(sum(mask)))),
        ),
    ))
    return out, mask


# =============================================================================
# Identity checks (per-claim audit primitives)
# =============================================================================


@dataclasses.dataclass(frozen=True)
class IdentityCheckResult:
    name: str
    ok: bool
    residual_l2: float
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": str(self.name),
            "ok": bool(self.ok),
            "residual_l2": float(round(self.residual_l2, 12)),
            "notes": str(self.notes),
        }


def check_merge_idempotent(
        a: Sequence[float],
        *,
        epsilon: float = W55_DEFAULT_ALGEBRA_EPSILON,
) -> IdentityCheckResult:
    """a ⊕ a should equal a (within epsilon)."""
    merged = merge_op(a, a)
    n = max(len(a), len(merged))
    ai = _pad_to(a, n)
    res = sum(
        (float(merged[i]) - float(ai[i])) ** 2
        for i in range(n))
    res = math.sqrt(res)
    return IdentityCheckResult(
        name="merge_idempotent",
        ok=bool(float(res) <= float(epsilon)),
        residual_l2=float(res),
        notes="a ⊕ a == a within epsilon",
    )


def check_difference_self_cancellation(
        a: Sequence[float],
        *,
        epsilon: float = W55_DEFAULT_ALGEBRA_EPSILON,
) -> IdentityCheckResult:
    """a ⊖ a should be zero (within epsilon)."""
    diff = difference_op(a, a)
    res = math.sqrt(sum(float(v) ** 2 for v in diff))
    return IdentityCheckResult(
        name="difference_self_cancellation",
        ok=bool(float(res) <= float(epsilon)),
        residual_l2=float(res),
        notes="a ⊖ a == 0 within epsilon",
    )


def check_intersection_distributivity_on_agreement(
        a: Sequence[float],
        b: Sequence[float],
        c: Sequence[float],
        *,
        agreement_floor: float = W55_DEFAULT_AGREEMENT_FLOOR,
        epsilon: float = 1e-6,
) -> IdentityCheckResult:
    """(a ⊕ b) ⊗ c should equal (a ⊗ c) ⊕ (b ⊗ c) on agreement subspace.

    The identity is exact on dimensions where a and b agree
    (|a_i - b_i| <= floor). Off that subspace, the two sides
    differ but the difference is bounded by the disagreement
    of a and b on that dim.

    Reports the residual on the agreement subspace only.
    """
    lhs_merged = merge_op(a, b)
    lhs, lhs_mask = intersection_op(
        lhs_merged, c, agreement_floor=agreement_floor)
    a_inter, a_mask = intersection_op(
        a, c, agreement_floor=agreement_floor)
    b_inter, b_mask = intersection_op(
        b, c, agreement_floor=agreement_floor)
    rhs = merge_op(a_inter, b_inter)
    # Compare on agreement subspace of a and b.
    n = max(len(a), len(b), len(c))
    ai = _pad_to(a, n)
    bi = _pad_to(b, n)
    res_sq = 0.0
    counted = 0
    for i in range(n):
        if abs(ai[i] - bi[i]) <= float(agreement_floor):
            # On agreement subspace: lhs and rhs should agree.
            l = float(lhs[i] if i < len(lhs) else 0.0)
            r = float(rhs[i] if i < len(rhs) else 0.0)
            res_sq += (l - r) ** 2
            counted += 1
    res = math.sqrt(res_sq)
    return IdentityCheckResult(
        name="intersection_distributivity_on_agreement",
        ok=bool(float(res) <= float(epsilon)),
        residual_l2=float(res),
        notes=(
            f"distributive on {counted}/{n} dims; "
            f"residual_l2 = {res:.4g}"),
    )


# =============================================================================
# Witness
# =============================================================================


@dataclasses.dataclass(frozen=True)
class DisagreementAlgebraWitness:
    """Audit witness for one algebra evaluation."""

    schema_version: str
    n_ops: int
    n_merges: int
    n_differences: int
    n_intersections: int
    trace_cid: str
    identity_check_results: tuple[IdentityCheckResult, ...]
    identity_check_pass_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": str(self.schema_version),
            "n_ops": int(self.n_ops),
            "n_merges": int(self.n_merges),
            "n_differences": int(self.n_differences),
            "n_intersections": int(self.n_intersections),
            "trace_cid": str(self.trace_cid),
            "identity_check_results": [
                r.to_dict()
                for r in self.identity_check_results],
            "identity_check_pass_count": int(
                self.identity_check_pass_count),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w55_algebra_witness",
            "witness": self.to_dict()})


def emit_disagreement_algebra_witness(
        *,
        trace: AlgebraTrace,
        identity_results: Sequence[IdentityCheckResult] = (),
) -> DisagreementAlgebraWitness:
    n_merges = sum(
        1 for s in trace.steps if s.op_kind == "merge")
    n_diff = sum(
        1 for s in trace.steps if s.op_kind == "difference")
    n_inter = sum(
        1 for s in trace.steps if s.op_kind == "intersection")
    return DisagreementAlgebraWitness(
        schema_version=W55_DISAGREEMENT_ALGEBRA_SCHEMA_VERSION,
        n_ops=int(len(trace.steps)),
        n_merges=int(n_merges),
        n_differences=int(n_diff),
        n_intersections=int(n_inter),
        trace_cid=str(trace.cid()),
        identity_check_results=tuple(identity_results),
        identity_check_pass_count=int(sum(
            1 for r in identity_results if r.ok)),
    )


# =============================================================================
# Verifier
# =============================================================================

W55_DISAGREEMENT_ALGEBRA_VERIFIER_FAILURE_MODES: tuple[
        str, ...] = (
    "w55_algebra_schema_mismatch",
    "w55_algebra_trace_cid_mismatch",
    "w55_algebra_op_count_mismatch",
    "w55_algebra_idempotent_fails",
    "w55_algebra_self_cancellation_fails",
    "w55_algebra_distributivity_residual_above_ceiling",
)


def verify_disagreement_algebra_witness(
        witness: DisagreementAlgebraWitness,
        *,
        expected_trace_cid: str | None = None,
        min_pass_count: int | None = None,
        max_distributivity_residual: float | None = None,
) -> dict[str, Any]:
    failures: list[str] = []
    if (witness.schema_version
            != W55_DISAGREEMENT_ALGEBRA_SCHEMA_VERSION):
        failures.append("w55_algebra_schema_mismatch")
    if (expected_trace_cid is not None
            and witness.trace_cid != expected_trace_cid):
        failures.append("w55_algebra_trace_cid_mismatch")
    if (min_pass_count is not None
            and witness.identity_check_pass_count
            < int(min_pass_count)):
        # Synthesise an op-count mismatch only if also short.
        for r in witness.identity_check_results:
            if r.name == "merge_idempotent" and not r.ok:
                failures.append(
                    "w55_algebra_idempotent_fails")
            if (r.name == "difference_self_cancellation"
                    and not r.ok):
                failures.append(
                    "w55_algebra_self_cancellation_fails")
    if max_distributivity_residual is not None:
        for r in witness.identity_check_results:
            if (r.name
                    == "intersection_distributivity_on_agreement"
                    and r.residual_l2
                    > float(max_distributivity_residual)):
                failures.append(
                    "w55_algebra_distributivity_residual"
                    "_above_ceiling")
    return {
        "ok": (len(failures) == 0),
        "failures": failures,
        "witness_cid": witness.cid(),
    }


__all__ = [
    "W55_DISAGREEMENT_ALGEBRA_SCHEMA_VERSION",
    "W55_DEFAULT_AGREEMENT_FLOOR",
    "W55_DEFAULT_ALGEBRA_EPSILON",
    "W55_DISAGREEMENT_ALGEBRA_VERIFIER_FAILURE_MODES",
    "AlgebraStep",
    "AlgebraTrace",
    "IdentityCheckResult",
    "DisagreementAlgebraWitness",
    "merge_op",
    "difference_op",
    "intersection_op",
    "merge_op_traced",
    "difference_op_traced",
    "intersection_op_traced",
    "disagreement_norm",
    "disagreement_max",
    "check_merge_idempotent",
    "check_difference_self_cancellation",
    "check_intersection_distributivity_on_agreement",
    "emit_disagreement_algebra_witness",
    "verify_disagreement_algebra_witness",
]
