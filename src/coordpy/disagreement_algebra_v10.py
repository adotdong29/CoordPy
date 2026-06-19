"""W64 — Disagreement Algebra V10.

Strictly extends W63's ``coordpy.disagreement_algebra_v9``. V9
added the Jensen-Shannon-equivalence identity. V10 adds:

* **Total-variation-equivalence identity** — two probes are
  total-variation-equivalent iff argmax preserved AND TV distance
  ≤ ``tv_floor``.
* **Total-variation-equivalence falsifier** — triggers when
  TV > floor.

V10 reduces to V9 byte-for-byte when neither ``tv_oracle`` nor
``tv_falsifier_oracle`` is provided.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Any, Callable, Sequence

from .disagreement_algebra import AlgebraTrace
from .disagreement_algebra_v9 import (
    DisagreementAlgebraV9Witness,
    emit_disagreement_algebra_v9_witness,
)
from .tiny_substrate_v3 import _sha256_hex


W64_DA_V10_SCHEMA_VERSION: str = (
    "coordpy.disagreement_algebra_v10.v1")


def _softmax(xs: Sequence[float]) -> list[float]:
    if not xs:
        return []
    m = max(float(x) for x in xs)
    e = [math.exp(float(x) - m) for x in xs]
    s = sum(e)
    if s <= 0.0:
        return [1.0 / len(xs)] * len(xs)
    return [v / s for v in e]


def total_variation_distance(
        a: Sequence[float], b: Sequence[float],
) -> float:
    """Total-variation distance over softmax-normalised payloads.
    Returns 0.5 * sum |pa - pb| ∈ [0, 1]."""
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    pa = _softmax(list(a)[:n])
    pb = _softmax(list(b)[:n])
    return float(0.5 * sum(abs(p - q) for p, q in zip(pa, pb)))


def check_tv_equivalence_identity(
        *, tv_oracle: Callable[[], tuple[bool, float]],
        tv_floor: float = 0.30,
) -> bool:
    try:
        argmax_ok, tv = tv_oracle()
    except Exception:
        return False
    return bool(argmax_ok) and bool(
        float(tv) <= float(tv_floor))


def tv_equivalence_falsifier(
        *, tv_oracle: Callable[[], tuple[bool, float]],
        tv_floor: float = 0.30,
) -> bool:
    try:
        argmax_ok, tv = tv_oracle()
    except Exception:
        return True
    return (not bool(argmax_ok)) or (
        float(tv) > float(tv_floor))


@dataclasses.dataclass(frozen=True)
class DisagreementAlgebraV10Witness:
    schema: str
    inner_v9_witness_cid: str
    tv_equiv_ok: bool
    tv_falsifier_ok: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "inner_v9_witness_cid": str(
                self.inner_v9_witness_cid),
            "tv_equiv_ok": bool(self.tv_equiv_ok),
            "tv_falsifier_ok": bool(self.tv_falsifier_ok),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "disagreement_algebra_v10_witness",
            "witness": self.to_dict()})


def emit_disagreement_algebra_v10_witness(
        *, trace: AlgebraTrace,
        probe_a: Sequence[float], probe_b: Sequence[float],
        probe_c: Sequence[float],
        tv_oracle: Callable[[], tuple[bool, float]] | None = None,
        tv_falsifier_oracle: (
            Callable[[], tuple[bool, float]] | None) = None,
        tv_floor: float = 0.30,
        **v9_kwargs: Any,
) -> DisagreementAlgebraV10Witness:
    v9 = emit_disagreement_algebra_v9_witness(
        trace=trace, probe_a=probe_a, probe_b=probe_b,
        probe_c=probe_c, **v9_kwargs)
    tv_ok = True
    if tv_oracle is not None:
        tv_ok = bool(check_tv_equivalence_identity(
            tv_oracle=tv_oracle,
            tv_floor=float(tv_floor)))
    tv_fals_ok = True
    if tv_falsifier_oracle is not None:
        tv_fals_ok = bool(tv_equivalence_falsifier(
            tv_oracle=tv_falsifier_oracle,
            tv_floor=float(tv_floor)))
    return DisagreementAlgebraV10Witness(
        schema=W64_DA_V10_SCHEMA_VERSION,
        inner_v9_witness_cid=str(v9.cid()),
        tv_equiv_ok=bool(tv_ok),
        tv_falsifier_ok=bool(tv_fals_ok),
    )


__all__ = [
    "W64_DA_V10_SCHEMA_VERSION",
    "total_variation_distance",
    "check_tv_equivalence_identity",
    "tv_equivalence_falsifier",
    "DisagreementAlgebraV10Witness",
    "emit_disagreement_algebra_v10_witness",
]
