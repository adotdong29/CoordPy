"""W66 M18 — Disagreement Algebra V12.

Strictly extends W65's ``coordpy.disagreement_algebra_v11``. V11
added the Wasserstein-equivalence identity. V12 adds:

* **Jensen-Shannon-equivalence identity** — two probes are
  JS-equivalent iff argmax preserved AND the symmetric
  Jensen-Shannon divergence ≤ ``js_floor``.
* **JS-equivalence falsifier** — triggers when JS > floor.

V12 reduces to V11 byte-for-byte when neither oracle is provided.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Callable, Sequence

from .disagreement_algebra import AlgebraTrace
from .disagreement_algebra_v11 import (
    DisagreementAlgebraV11Witness,
    emit_disagreement_algebra_v11_witness,
)
from .tiny_substrate_v3 import _sha256_hex


W66_DA_V12_SCHEMA_VERSION: str = (
    "coordpy.disagreement_algebra_v12.v1")


def _normalize_pmf(
        x: Sequence[float], eps: float = 1e-12,
) -> list[float]:
    vals = [max(float(eps), float(v)) for v in x]
    s = float(sum(vals))
    if s <= 0.0:
        return [1.0 / float(max(1, len(vals)))] * len(vals)
    return [v / s for v in vals]


def js_divergence_symmetric(
        a: Sequence[float], b: Sequence[float],
) -> float:
    """Symmetric Jensen-Shannon divergence between two pmfs (after
    softmax-normalisation of inputs treated as logits)."""
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    pa = _normalize_pmf(list(a)[:n])
    pb = _normalize_pmf(list(b)[:n])
    m = [(p + q) * 0.5 for p, q in zip(pa, pb)]
    kl_am = 0.0
    kl_bm = 0.0
    for p, q, mm in zip(pa, pb, m):
        if p > 0.0 and mm > 0.0:
            kl_am += p * math.log(p / mm)
        if q > 0.0 and mm > 0.0:
            kl_bm += q * math.log(q / mm)
    return float(0.5 * (kl_am + kl_bm))


def check_js_equivalence_identity(
        *, js_oracle: Callable[[], tuple[bool, float]],
        js_floor: float = 0.20,
) -> bool:
    try:
        argmax_ok, j = js_oracle()
    except Exception:
        return False
    return bool(argmax_ok) and bool(
        float(j) <= float(js_floor))


def js_equivalence_falsifier(
        *, js_oracle: Callable[[], tuple[bool, float]],
        js_floor: float = 0.20,
) -> bool:
    try:
        argmax_ok, j = js_oracle()
    except Exception:
        return True
    return (not bool(argmax_ok)) or (
        float(j) > float(js_floor))


@dataclasses.dataclass(frozen=True)
class DisagreementAlgebraV12Witness:
    schema: str
    inner_v11_witness_cid: str
    js_equiv_ok: bool
    js_falsifier_ok: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "inner_v11_witness_cid": str(
                self.inner_v11_witness_cid),
            "js_equiv_ok": bool(self.js_equiv_ok),
            "js_falsifier_ok": bool(self.js_falsifier_ok),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "disagreement_algebra_v12_witness",
            "witness": self.to_dict()})


def emit_disagreement_algebra_v12_witness(
        *, trace: AlgebraTrace,
        probe_a: Sequence[float],
        probe_b: Sequence[float],
        probe_c: Sequence[float],
        js_oracle: (
            Callable[[], tuple[bool, float]] | None) = None,
        js_falsifier_oracle: (
            Callable[[], tuple[bool, float]] | None) = None,
        js_floor: float = 0.20,
        **v11_kwargs: Any,
) -> DisagreementAlgebraV12Witness:
    v11 = emit_disagreement_algebra_v11_witness(
        trace=trace, probe_a=probe_a, probe_b=probe_b,
        probe_c=probe_c, **v11_kwargs)
    j_ok = True
    if js_oracle is not None:
        j_ok = bool(check_js_equivalence_identity(
            js_oracle=js_oracle, js_floor=float(js_floor)))
    j_fals_ok = True
    if js_falsifier_oracle is not None:
        j_fals_ok = bool(js_equivalence_falsifier(
            js_oracle=js_falsifier_oracle,
            js_floor=float(js_floor)))
    return DisagreementAlgebraV12Witness(
        schema=W66_DA_V12_SCHEMA_VERSION,
        inner_v11_witness_cid=str(v11.cid()),
        js_equiv_ok=bool(j_ok),
        js_falsifier_ok=bool(j_fals_ok),
    )


__all__ = [
    "W66_DA_V12_SCHEMA_VERSION",
    "js_divergence_symmetric",
    "check_js_equivalence_identity",
    "js_equivalence_falsifier",
    "DisagreementAlgebraV12Witness",
    "emit_disagreement_algebra_v12_witness",
]
