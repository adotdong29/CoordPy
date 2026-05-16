"""W67 M18 — Disagreement Algebra V13.

Strictly extends W66's ``coordpy.disagreement_algebra_v12``. V12
added the Jensen-Shannon-equivalence identity. V13 adds:

* **Bregman-equivalence identity** — two probes are
  Bregman-equivalent iff argmax preserved AND the Bregman
  divergence (under a strictly convex generator) ≤ ``bregman_floor``.
* **Bregman-equivalence falsifier** — triggers when Bregman > floor.

V13 reduces to V12 byte-for-byte when neither oracle is provided.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Callable, Sequence

from .disagreement_algebra import AlgebraTrace
from .disagreement_algebra_v12 import (
    emit_disagreement_algebra_v12_witness,
)
from .tiny_substrate_v3 import _sha256_hex


W67_DA_V13_SCHEMA_VERSION: str = (
    "coordpy.disagreement_algebra_v13.v1")


def bregman_divergence_squared_l2(
        a: Sequence[float], b: Sequence[float],
) -> float:
    """Bregman divergence with the squared-L2 generator: equals
    0.5 * ||a - b||^2. The squared-L2 generator is strictly convex
    so this is a proper Bregman divergence."""
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    s = 0.0
    for i in range(n):
        d = float(a[i]) - float(b[i])
        s += float(d * d)
    return float(0.5 * s)


def bregman_divergence_kl(
        a: Sequence[float], b: Sequence[float],
) -> float:
    """Bregman divergence with the negative-entropy generator
    (= KL divergence after normalisation)."""
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    pa = [max(1e-12, float(v)) for v in a[:n]]
    pb = [max(1e-12, float(v)) for v in b[:n]]
    sa = float(sum(pa))
    sb = float(sum(pb))
    if sa <= 0.0 or sb <= 0.0:
        return 0.0
    pa = [v / sa for v in pa]
    pb = [v / sb for v in pb]
    out = 0.0
    for p, q in zip(pa, pb):
        if p > 0.0 and q > 0.0:
            out += float(p * math.log(p / q))
    return float(out)


def check_bregman_equivalence_identity(
        *, bregman_oracle: Callable[[], tuple[bool, float]],
        bregman_floor: float = 0.20,
) -> bool:
    try:
        argmax_ok, br = bregman_oracle()
    except Exception:
        return False
    return bool(argmax_ok) and bool(
        float(br) <= float(bregman_floor))


def bregman_equivalence_falsifier(
        *, bregman_oracle: Callable[[], tuple[bool, float]],
        bregman_floor: float = 0.20,
) -> bool:
    try:
        argmax_ok, br = bregman_oracle()
    except Exception:
        return True
    return (not bool(argmax_ok)) or (
        float(br) > float(bregman_floor))


@dataclasses.dataclass(frozen=True)
class DisagreementAlgebraV13Witness:
    schema: str
    inner_v12_witness_cid: str
    bregman_equiv_ok: bool
    bregman_falsifier_ok: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "inner_v12_witness_cid": str(
                self.inner_v12_witness_cid),
            "bregman_equiv_ok": bool(self.bregman_equiv_ok),
            "bregman_falsifier_ok": bool(
                self.bregman_falsifier_ok),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "disagreement_algebra_v13_witness",
            "witness": self.to_dict()})


def emit_disagreement_algebra_v13_witness(
        *, trace: AlgebraTrace,
        probe_a: Sequence[float],
        probe_b: Sequence[float],
        probe_c: Sequence[float],
        bregman_oracle: (
            Callable[[], tuple[bool, float]] | None) = None,
        bregman_falsifier_oracle: (
            Callable[[], tuple[bool, float]] | None) = None,
        bregman_floor: float = 0.20,
        **v12_kwargs: Any,
) -> DisagreementAlgebraV13Witness:
    v12 = emit_disagreement_algebra_v12_witness(
        trace=trace, probe_a=probe_a, probe_b=probe_b,
        probe_c=probe_c, **v12_kwargs)
    b_ok = True
    if bregman_oracle is not None:
        b_ok = bool(check_bregman_equivalence_identity(
            bregman_oracle=bregman_oracle,
            bregman_floor=float(bregman_floor)))
    b_fals_ok = True
    if bregman_falsifier_oracle is not None:
        b_fals_ok = bool(bregman_equivalence_falsifier(
            bregman_oracle=bregman_falsifier_oracle,
            bregman_floor=float(bregman_floor)))
    return DisagreementAlgebraV13Witness(
        schema=W67_DA_V13_SCHEMA_VERSION,
        inner_v12_witness_cid=str(v12.cid()),
        bregman_equiv_ok=bool(b_ok),
        bregman_falsifier_ok=bool(b_fals_ok),
    )


__all__ = [
    "W67_DA_V13_SCHEMA_VERSION",
    "bregman_divergence_squared_l2",
    "bregman_divergence_kl",
    "check_bregman_equivalence_identity",
    "bregman_equivalence_falsifier",
    "DisagreementAlgebraV13Witness",
    "emit_disagreement_algebra_v13_witness",
]
