"""W65 M18 — Disagreement Algebra V11.

Strictly extends W64's ``coordpy.disagreement_algebra_v10``. V10
added the total-variation-equivalence identity. V11 adds:

* **Wasserstein-equivalence identity** — two probes are
  Wasserstein-equivalent iff argmax preserved AND the 1-D
  Wasserstein-1 distance (sorted absolute difference) ≤
  ``wasserstein_floor``.
* **Wasserstein-equivalence falsifier** — triggers when
  Wasserstein > floor.

V11 reduces to V10 byte-for-byte when neither oracle is provided.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Sequence

from .disagreement_algebra import AlgebraTrace
from .disagreement_algebra_v10 import (
    DisagreementAlgebraV10Witness,
    emit_disagreement_algebra_v10_witness,
)
from .tiny_substrate_v3 import _sha256_hex


W65_DA_V11_SCHEMA_VERSION: str = (
    "coordpy.disagreement_algebra_v11.v1")


def wasserstein_1d_distance(
        a: Sequence[float], b: Sequence[float],
) -> float:
    """1-D Wasserstein-1 distance between two sequences (treated
    as point masses): mean of sorted absolute differences."""
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    sa = sorted(float(x) for x in list(a)[:n])
    sb = sorted(float(x) for x in list(b)[:n])
    return float(sum(abs(p - q) for p, q in zip(sa, sb)) / n)


def check_wasserstein_equivalence_identity(
        *, wasserstein_oracle: Callable[[],
            tuple[bool, float]],
        wasserstein_floor: float = 0.20,
) -> bool:
    try:
        argmax_ok, w = wasserstein_oracle()
    except Exception:
        return False
    return bool(argmax_ok) and bool(
        float(w) <= float(wasserstein_floor))


def wasserstein_equivalence_falsifier(
        *, wasserstein_oracle: Callable[[],
            tuple[bool, float]],
        wasserstein_floor: float = 0.20,
) -> bool:
    try:
        argmax_ok, w = wasserstein_oracle()
    except Exception:
        return True
    return (not bool(argmax_ok)) or (
        float(w) > float(wasserstein_floor))


@dataclasses.dataclass(frozen=True)
class DisagreementAlgebraV11Witness:
    schema: str
    inner_v10_witness_cid: str
    wasserstein_equiv_ok: bool
    wasserstein_falsifier_ok: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "inner_v10_witness_cid": str(
                self.inner_v10_witness_cid),
            "wasserstein_equiv_ok": bool(
                self.wasserstein_equiv_ok),
            "wasserstein_falsifier_ok": bool(
                self.wasserstein_falsifier_ok),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "disagreement_algebra_v11_witness",
            "witness": self.to_dict()})


def emit_disagreement_algebra_v11_witness(
        *, trace: AlgebraTrace,
        probe_a: Sequence[float],
        probe_b: Sequence[float],
        probe_c: Sequence[float],
        wasserstein_oracle: (
            Callable[[], tuple[bool, float]] | None) = None,
        wasserstein_falsifier_oracle: (
            Callable[[], tuple[bool, float]] | None) = None,
        wasserstein_floor: float = 0.20,
        **v10_kwargs: Any,
) -> DisagreementAlgebraV11Witness:
    v10 = emit_disagreement_algebra_v10_witness(
        trace=trace, probe_a=probe_a, probe_b=probe_b,
        probe_c=probe_c, **v10_kwargs)
    w_ok = True
    if wasserstein_oracle is not None:
        w_ok = bool(check_wasserstein_equivalence_identity(
            wasserstein_oracle=wasserstein_oracle,
            wasserstein_floor=float(wasserstein_floor)))
    w_fals_ok = True
    if wasserstein_falsifier_oracle is not None:
        w_fals_ok = bool(wasserstein_equivalence_falsifier(
            wasserstein_oracle=wasserstein_falsifier_oracle,
            wasserstein_floor=float(wasserstein_floor)))
    return DisagreementAlgebraV11Witness(
        schema=W65_DA_V11_SCHEMA_VERSION,
        inner_v10_witness_cid=str(v10.cid()),
        wasserstein_equiv_ok=bool(w_ok),
        wasserstein_falsifier_ok=bool(w_fals_ok),
    )


__all__ = [
    "W65_DA_V11_SCHEMA_VERSION",
    "wasserstein_1d_distance",
    "check_wasserstein_equivalence_identity",
    "wasserstein_equivalence_falsifier",
    "DisagreementAlgebraV11Witness",
    "emit_disagreement_algebra_v11_witness",
]
