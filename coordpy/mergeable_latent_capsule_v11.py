"""W63 M13 — Mergeable Latent State Capsule V11.

Strictly extends W62's ``coordpy.mergeable_latent_capsule_v10``.
V10 added ``replay_dominance_witness_chain``,
``disagreement_wasserstein_distance``, and two algebra signatures.
V11 adds:

* ``hidden_wins_witness_chain`` — union-inherited at merge time.
  Each entry is a CID of a witness from the W63 hidden-state
  bridge V7 *with positive hidden-wins margin*.
* ``prefix_reuse_witness_chain`` — union-inherited at merge time.
  Each entry is a CID of a successful prefix-reuse decision.
* ``disagreement_jensen_shannon_distance`` — Jensen-Shannon
  divergence scalar computed at merge time between parents'
  payload distributions.
* Two new algebra signatures:
  ``hidden_wins_propagation`` and
  ``jensen_shannon_disagreement``.

Honest scope
------------

* Hidden-wins / prefix-reuse chains inherit as union; no
  probabilistic weighting.
* JS divergence is computed on softmax-normalised payloads with
  base e; not a calibrated information-theoretic divergence on
  the underlying probability distributions.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

from .mergeable_latent_capsule_v4 import (
    W56_MLSC_V4_ALGEBRA_MERGE,
)
from .mergeable_latent_capsule_v10 import (
    MergeableLatentCapsuleV10,
    MergeOperatorV10,
    W62_MLSC_V10_KNOWN_ALGEBRA_SIGNATURES,
    wrap_v9_as_v10,
)
from .tiny_substrate_v3 import _sha256_hex


W63_MLSC_V11_SCHEMA_VERSION: str = (
    "coordpy.mergeable_latent_capsule_v11.v1")

W63_MLSC_V11_ALGEBRA_HIDDEN_WINS_PROPAGATION: str = (
    "hidden_wins_propagation")
W63_MLSC_V11_ALGEBRA_JENSEN_SHANNON_DISAGREEMENT: str = (
    "jensen_shannon_disagreement")

W63_MLSC_V11_KNOWN_ALGEBRA_SIGNATURES: tuple[str, ...] = (
    *W62_MLSC_V10_KNOWN_ALGEBRA_SIGNATURES,
    W63_MLSC_V11_ALGEBRA_HIDDEN_WINS_PROPAGATION,
    W63_MLSC_V11_ALGEBRA_JENSEN_SHANNON_DISAGREEMENT,
)


def _softmax_arr(xs: Sequence[float]) -> list[float]:
    if not xs:
        return []
    m = max(float(x) for x in xs)
    e = [math.exp(float(x) - m) for x in xs]
    s = sum(e)
    if s <= 0.0:
        return [1.0 / len(xs)] * len(xs)
    return [v / s for v in e]


def _jensen_shannon_distance(
        a: Sequence[float], b: Sequence[float],
) -> float:
    """JS divergence between softmax distributions over a, b.
    Returns sqrt(JS) ∈ [0, sqrt(ln(2))]."""
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    pa = _softmax_arr(list(a)[:n])
    pb = _softmax_arr(list(b)[:n])
    m = [(p + q) / 2.0 for p, q in zip(pa, pb)]
    def kl(p: list[float], q: list[float]) -> float:
        s = 0.0
        for pi, qi in zip(p, q):
            if pi > 0.0 and qi > 0.0:
                s += pi * math.log(pi / qi)
        return s
    js = 0.5 * kl(pa, m) + 0.5 * kl(pb, m)
    return float(math.sqrt(max(0.0, js)))


@dataclasses.dataclass(frozen=True)
class MergeableLatentCapsuleV11:
    schema: str
    inner_v10: MergeableLatentCapsuleV10
    hidden_wins_witness_chain: tuple[str, ...]
    prefix_reuse_witness_chain: tuple[str, ...]
    disagreement_jensen_shannon_distance: float
    algebra_signature_v11: str

    @property
    def payload(self) -> tuple[float, ...]:
        return self.inner_v10.payload

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "inner_v10_cid": str(self.inner_v10.cid()),
            "hidden_wins_witness_chain": list(
                self.hidden_wins_witness_chain),
            "prefix_reuse_witness_chain": list(
                self.prefix_reuse_witness_chain),
            "disagreement_jensen_shannon_distance": float(
                round(
                    self.disagreement_jensen_shannon_distance,
                    12)),
            "algebra_signature_v11": str(
                self.algebra_signature_v11),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w63_mlsc_v11",
            "capsule": self.to_dict()})


def wrap_v10_as_v11(
        v10_capsule: MergeableLatentCapsuleV10, *,
        hidden_wins_witness_chain: Sequence[str] = (),
        prefix_reuse_witness_chain: Sequence[str] = (),
        disagreement_jensen_shannon_distance: float = 0.0,
        algebra_signature_v11: str = W56_MLSC_V4_ALGEBRA_MERGE,
) -> MergeableLatentCapsuleV11:
    if (algebra_signature_v11 not in
            W63_MLSC_V11_KNOWN_ALGEBRA_SIGNATURES):
        algebra_signature_v11 = W56_MLSC_V4_ALGEBRA_MERGE
    return MergeableLatentCapsuleV11(
        schema=W63_MLSC_V11_SCHEMA_VERSION,
        inner_v10=v10_capsule,
        hidden_wins_witness_chain=tuple(
            str(s) for s in hidden_wins_witness_chain),
        prefix_reuse_witness_chain=tuple(
            str(s) for s in prefix_reuse_witness_chain),
        disagreement_jensen_shannon_distance=float(
            disagreement_jensen_shannon_distance),
        algebra_signature_v11=str(algebra_signature_v11),
    )


@dataclasses.dataclass
class MergeOperatorV11:
    factor_dim: int = 6

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w63_mlsc_v11_merge_operator",
            "factor_dim": int(self.factor_dim),
        })

    def merge(
            self,
            capsules: Sequence[MergeableLatentCapsuleV11],
            *,
            hidden_wins_witness_chain: Sequence[str] = (),
            prefix_reuse_witness_chain: Sequence[str] = (),
            algebra_signature_v11: str = (
                W56_MLSC_V4_ALGEBRA_MERGE),
            **v10_kwargs: Any,
    ) -> MergeableLatentCapsuleV11:
        if not capsules:
            raise ValueError(
                "merge requires at least 1 capsule")
        v10_op = MergeOperatorV10(
            factor_dim=int(self.factor_dim))
        merged_v10 = v10_op.merge(
            [c.inner_v10 for c in capsules], **v10_kwargs)
        hw_set: list[str] = []
        for c in capsules:
            for a in c.hidden_wins_witness_chain:
                if a not in hw_set:
                    hw_set.append(a)
        for a in hidden_wins_witness_chain:
            if a not in hw_set:
                hw_set.append(a)
        pr_set: list[str] = []
        for c in capsules:
            for a in c.prefix_reuse_witness_chain:
                if a not in pr_set:
                    pr_set.append(a)
        for a in prefix_reuse_witness_chain:
            if a not in pr_set:
                pr_set.append(a)
        js = 0.0
        if len(capsules) >= 2:
            js = _jensen_shannon_distance(
                list(capsules[0].payload),
                list(capsules[1].payload))
        return wrap_v10_as_v11(
            merged_v10,
            hidden_wins_witness_chain=tuple(hw_set),
            prefix_reuse_witness_chain=tuple(pr_set),
            disagreement_jensen_shannon_distance=float(js),
            algebra_signature_v11=str(algebra_signature_v11),
        )


@dataclasses.dataclass(frozen=True)
class MergeableLatentCapsuleV11Witness:
    schema: str
    capsule_cid: str
    inner_v10_cid: str
    hidden_wins_witness_chain_depth: int
    prefix_reuse_witness_chain_depth: int
    disagreement_jensen_shannon_distance: float
    algebra_signature_v11: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "capsule_cid": str(self.capsule_cid),
            "inner_v10_cid": str(self.inner_v10_cid),
            "hidden_wins_witness_chain_depth": int(
                self.hidden_wins_witness_chain_depth),
            "prefix_reuse_witness_chain_depth": int(
                self.prefix_reuse_witness_chain_depth),
            "disagreement_jensen_shannon_distance": float(
                round(
                    self.disagreement_jensen_shannon_distance,
                    12)),
            "algebra_signature_v11": str(
                self.algebra_signature_v11),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w63_mlsc_v11_witness",
            "witness": self.to_dict()})


def emit_mlsc_v11_witness(
        capsule: MergeableLatentCapsuleV11,
) -> MergeableLatentCapsuleV11Witness:
    return MergeableLatentCapsuleV11Witness(
        schema=W63_MLSC_V11_SCHEMA_VERSION,
        capsule_cid=str(capsule.cid()),
        inner_v10_cid=str(capsule.inner_v10.cid()),
        hidden_wins_witness_chain_depth=int(len(
            capsule.hidden_wins_witness_chain)),
        prefix_reuse_witness_chain_depth=int(len(
            capsule.prefix_reuse_witness_chain)),
        disagreement_jensen_shannon_distance=float(
            capsule.disagreement_jensen_shannon_distance),
        algebra_signature_v11=str(
            capsule.algebra_signature_v11),
    )


__all__ = [
    "W63_MLSC_V11_SCHEMA_VERSION",
    "W63_MLSC_V11_ALGEBRA_HIDDEN_WINS_PROPAGATION",
    "W63_MLSC_V11_ALGEBRA_JENSEN_SHANNON_DISAGREEMENT",
    "W63_MLSC_V11_KNOWN_ALGEBRA_SIGNATURES",
    "MergeableLatentCapsuleV11",
    "wrap_v10_as_v11",
    "MergeOperatorV11",
    "MergeableLatentCapsuleV11Witness",
    "emit_mlsc_v11_witness",
]
