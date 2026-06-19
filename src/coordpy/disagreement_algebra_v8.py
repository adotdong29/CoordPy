"""W62 — Disagreement Algebra V8.

Strictly extends W61's ``coordpy.disagreement_algebra_v7``. V7
added the attention-pattern-equivalence identity. V8 adds:

* **Wasserstein-1-equivalence identity** — two probes are
  Wasserstein-1-equivalent iff argmax preserved AND Wasserstein-1
  distance ≤ ``wasserstein_floor``.
* **Wasserstein-1-equivalence falsifier** — triggers when
  Wasserstein-1 > floor.

V8 reduces to V7 byte-for-byte when neither
``wasserstein_oracle`` nor ``wasserstein_falsifier_oracle`` is
provided.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Callable, Sequence

from .disagreement_algebra import AlgebraTrace
from .disagreement_algebra_v7 import (
    DisagreementAlgebraV7Witness,
    emit_disagreement_algebra_v7_witness,
)
from .tiny_substrate_v3 import _sha256_hex


W62_DA_V8_SCHEMA_VERSION: str = (
    "coordpy.disagreement_algebra_v8.v1")


def wasserstein_1_distance(
        a: Sequence[float], b: Sequence[float],
) -> float:
    """L1 distance between sorted samples (per-dim normalised)."""
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    sa = sorted(float(x) for x in a)[:n]
    sb = sorted(float(x) for x in b)[:n]
    if n == 0:
        return 0.0
    return float(sum(abs(x - y)
                     for x, y in zip(sa, sb)) / float(n))


def check_wasserstein_equivalence_identity(
        *, wasserstein_oracle: Callable[[], tuple[bool, float]],
        wasserstein_floor: float = 0.5,
) -> bool:
    try:
        argmax_ok, wd = wasserstein_oracle()
    except Exception:
        return False
    return bool(argmax_ok) and bool(
        float(wd) <= float(wasserstein_floor))


def wasserstein_equivalence_falsifier(
        *, wasserstein_oracle: Callable[[], tuple[bool, float]],
        wasserstein_floor: float = 0.5,
) -> bool:
    try:
        argmax_ok, wd = wasserstein_oracle()
    except Exception:
        return True
    return (not bool(argmax_ok)) or (
        float(wd) > float(wasserstein_floor))


@dataclasses.dataclass(frozen=True)
class DisagreementAlgebraV8Witness:
    schema: str
    inner_v7_witness_cid: str
    wasserstein_equiv_ok: bool
    wasserstein_falsifier_ok: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "inner_v7_witness_cid": str(
                self.inner_v7_witness_cid),
            "wasserstein_equiv_ok": bool(
                self.wasserstein_equiv_ok),
            "wasserstein_falsifier_ok": bool(
                self.wasserstein_falsifier_ok),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "disagreement_algebra_v8_witness",
            "witness": self.to_dict()})


def emit_disagreement_algebra_v8_witness(
        *, trace: AlgebraTrace,
        probe_a: Sequence[float], probe_b: Sequence[float],
        probe_c: Sequence[float],
        wasserstein_oracle: Callable[[], tuple[bool, float]]
            | None = None,
        wasserstein_falsifier_oracle: (
            Callable[[], tuple[bool, float]] | None) = None,
        wasserstein_floor: float = 0.5,
        **v7_kwargs: Any,
) -> DisagreementAlgebraV8Witness:
    v7 = emit_disagreement_algebra_v7_witness(
        trace=trace, probe_a=probe_a, probe_b=probe_b,
        probe_c=probe_c, **v7_kwargs)
    ws_ok = True
    if wasserstein_oracle is not None:
        ws_ok = bool(check_wasserstein_equivalence_identity(
            wasserstein_oracle=wasserstein_oracle,
            wasserstein_floor=float(wasserstein_floor)))
    ws_fals_ok = True
    if wasserstein_falsifier_oracle is not None:
        ws_fals_ok = bool(wasserstein_equivalence_falsifier(
            wasserstein_oracle=wasserstein_falsifier_oracle,
            wasserstein_floor=float(wasserstein_floor)))
    return DisagreementAlgebraV8Witness(
        schema=W62_DA_V8_SCHEMA_VERSION,
        inner_v7_witness_cid=str(v7.cid()),
        wasserstein_equiv_ok=bool(ws_ok),
        wasserstein_falsifier_ok=bool(ws_fals_ok),
    )


__all__ = [
    "W62_DA_V8_SCHEMA_VERSION",
    "wasserstein_1_distance",
    "check_wasserstein_equivalence_identity",
    "wasserstein_equivalence_falsifier",
    "DisagreementAlgebraV8Witness",
    "emit_disagreement_algebra_v8_witness",
]
