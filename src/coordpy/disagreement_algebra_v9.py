"""W63 — Disagreement Algebra V9.

Strictly extends W62's ``coordpy.disagreement_algebra_v8``. V8
added the Wasserstein-1-equivalence identity. V9 adds:

* **Jensen-Shannon-equivalence identity** — two probes are
  Jensen-Shannon-equivalent iff argmax preserved AND JS distance
  ≤ ``js_floor``.
* **Jensen-Shannon-equivalence falsifier** — triggers when
  JS > floor.

V9 reduces to V8 byte-for-byte when neither ``js_oracle`` nor
``js_falsifier_oracle`` is provided.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Any, Callable, Sequence

from .disagreement_algebra import AlgebraTrace
from .disagreement_algebra_v8 import (
    DisagreementAlgebraV8Witness,
    emit_disagreement_algebra_v8_witness,
)
from .tiny_substrate_v3 import _sha256_hex


W63_DA_V9_SCHEMA_VERSION: str = (
    "coordpy.disagreement_algebra_v9.v1")


def _softmax(xs: Sequence[float]) -> list[float]:
    if not xs:
        return []
    m = max(float(x) for x in xs)
    e = [math.exp(float(x) - m) for x in xs]
    s = sum(e)
    if s <= 0.0:
        return [1.0 / len(xs)] * len(xs)
    return [v / s for v in e]


def jensen_shannon_1_distance(
        a: Sequence[float], b: Sequence[float],
) -> float:
    """Jensen-Shannon divergence (base e). Returns sqrt(JS)
    ∈ [0, sqrt(ln(2))]."""
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    pa = _softmax(list(a)[:n])
    pb = _softmax(list(b)[:n])
    m = [(p + q) / 2.0 for p, q in zip(pa, pb)]
    def kl(p: list[float], q: list[float]) -> float:
        s = 0.0
        for pi, qi in zip(p, q):
            if pi > 0.0 and qi > 0.0:
                s += pi * math.log(pi / qi)
        return s
    js = 0.5 * kl(pa, m) + 0.5 * kl(pb, m)
    return float(math.sqrt(max(0.0, js)))


def check_js_equivalence_identity(
        *, js_oracle: Callable[[], tuple[bool, float]],
        js_floor: float = 0.30,
) -> bool:
    try:
        argmax_ok, js = js_oracle()
    except Exception:
        return False
    return bool(argmax_ok) and bool(
        float(js) <= float(js_floor))


def js_equivalence_falsifier(
        *, js_oracle: Callable[[], tuple[bool, float]],
        js_floor: float = 0.30,
) -> bool:
    try:
        argmax_ok, js = js_oracle()
    except Exception:
        return True
    return (not bool(argmax_ok)) or (
        float(js) > float(js_floor))


@dataclasses.dataclass(frozen=True)
class DisagreementAlgebraV9Witness:
    schema: str
    inner_v8_witness_cid: str
    js_equiv_ok: bool
    js_falsifier_ok: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "inner_v8_witness_cid": str(
                self.inner_v8_witness_cid),
            "js_equiv_ok": bool(self.js_equiv_ok),
            "js_falsifier_ok": bool(self.js_falsifier_ok),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "disagreement_algebra_v9_witness",
            "witness": self.to_dict()})


def emit_disagreement_algebra_v9_witness(
        *, trace: AlgebraTrace,
        probe_a: Sequence[float], probe_b: Sequence[float],
        probe_c: Sequence[float],
        js_oracle: Callable[[], tuple[bool, float]] | None = None,
        js_falsifier_oracle: (
            Callable[[], tuple[bool, float]] | None) = None,
        js_floor: float = 0.30,
        **v8_kwargs: Any,
) -> DisagreementAlgebraV9Witness:
    v8 = emit_disagreement_algebra_v8_witness(
        trace=trace, probe_a=probe_a, probe_b=probe_b,
        probe_c=probe_c, **v8_kwargs)
    js_ok = True
    if js_oracle is not None:
        js_ok = bool(check_js_equivalence_identity(
            js_oracle=js_oracle,
            js_floor=float(js_floor)))
    js_fals_ok = True
    if js_falsifier_oracle is not None:
        js_fals_ok = bool(js_equivalence_falsifier(
            js_oracle=js_falsifier_oracle,
            js_floor=float(js_floor)))
    return DisagreementAlgebraV9Witness(
        schema=W63_DA_V9_SCHEMA_VERSION,
        inner_v8_witness_cid=str(v8.cid()),
        js_equiv_ok=bool(js_ok),
        js_falsifier_ok=bool(js_fals_ok),
    )


__all__ = [
    "W63_DA_V9_SCHEMA_VERSION",
    "jensen_shannon_1_distance",
    "check_js_equivalence_identity",
    "js_equivalence_falsifier",
    "DisagreementAlgebraV9Witness",
    "emit_disagreement_algebra_v9_witness",
]
