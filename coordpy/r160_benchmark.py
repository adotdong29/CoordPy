"""W69 R-160 benchmark family — Compound regime + falsifier + limitation.

The W69 falsifier-and-limitation-reproduction family. R-160 reproduces:

* the explicit hosted-vs-real boundary V2 falsifier (carries
  W68's wall behaviour forward)
* the ECC V21 rate-ceiling falsifier (≥ 35 bits/visible-token
  ceiling, 65536-bit target exceeds it by ~30x)
* the multi-branch-rejoin falsifier (KV V14)
* the disagreement-algebra V15 multi-branch-rejoin equivalence
* the compound silent_corruption_plus_member_replacement + multi-
  branch-rejoin regime applied to MASC V5 strict-beat rate
* the limitation reproductions for W69-L-NUMPY-CPU-V14-SUBSTRATE-
  CAP, W69-L-V14-NO-AUTOGRAD-CAP, W69-L-NO-THIRD-PARTY-SUBSTRATE-
  COUPLING-CAP

H460..H467 cell families (8 H-bars):

* H460   Boundary V2 falsifier triggers on dishonest claim
* H461   ECC V21 rate ceiling falsifier exposes 65536 > 35
* H462   KV V14 multi-branch-rejoin falsifier is honest (0)
* H463   Disagreement V15 multi-branch-rejoin equivalence + falsifier
* H464   MASC V5 compound regime (multi_branch_rejoin) ≥ 50 %
* H465   MASC V5 compound regime (silent_corruption) ≥ 50 %
* H466   W69 substrate is in-repo NumPy (limitation reproduction)
* H467   W69 hosted control plane V2 does NOT pierce wall
         (limitation reproduction)
"""

from __future__ import annotations

from typing import Any, Sequence

from coordpy.disagreement_algebra_v15 import (
    emit_disagreement_algebra_v15_witness,
)
from coordpy.disagreement_algebra import (
    AlgebraTrace,
)
from coordpy.ecc_codebook_v21 import (
    ECCCodebookV21, probe_ecc_v21_rate_floor_falsifier,
)
from coordpy.hosted_real_substrate_boundary_v2 import (
    build_default_hosted_real_substrate_boundary_v2,
    probe_hosted_real_substrate_boundary_v2_falsifier,
)
from coordpy.kv_bridge_v14 import (
    probe_kv_bridge_v14_multi_branch_rejoin_falsifier,
)
from coordpy.multi_agent_substrate_coordinator_v5 import (
    MultiAgentSubstrateCoordinatorV5,
    W69_MASC_V5_REGIME_MULTI_BRANCH_REJOIN,
    W69_MASC_V5_REGIME_SILENT_CORRUPTION_PLUS_REPLACEMENT,
)
from coordpy.tiny_substrate_v14 import (
    W69_DEFAULT_V14_N_LAYERS, W69_TINY_V14_VOCAB_SIZE,
)


R160_SCHEMA_VERSION: str = "coordpy.r160_benchmark.v1"


def family_h460_boundary_v2_falsifier(
        seed: int) -> dict[str, Any]:
    b = build_default_hosted_real_substrate_boundary_v2()
    honest = probe_hosted_real_substrate_boundary_v2_falsifier(
        boundary=b, claimed_axis="multi_branch_rejoin_witness",
        claim_satisfied_at_hosted=False)
    dishonest = (
        probe_hosted_real_substrate_boundary_v2_falsifier(
            boundary=b,
            claimed_axis="multi_branch_rejoin_witness",
            claim_satisfied_at_hosted=True))
    return {
        "schema": R160_SCHEMA_VERSION,
        "name": "h460_boundary_v2_falsifier",
        "passed": bool(
            honest.falsifier_score == 0.0
            and dishonest.falsifier_score == 1.0),
    }


def family_h461_ecc_v21_rate_ceiling(
        seed: int) -> dict[str, Any]:
    cb = ECCCodebookV21.init(seed=int(seed) + 60000)
    fal = probe_ecc_v21_rate_floor_falsifier(codebook=cb)
    return {
        "schema": R160_SCHEMA_VERSION,
        "name": "h461_ecc_v21_rate_ceiling",
        "passed": bool(fal["target_exceeds_ceiling"]),
        "ceiling_bits": float(fal["ceiling_bits"]),
        "target_bits": float(fal["target_bits"]),
    }


def family_h462_kv_v14_multi_branch_rejoin_falsifier(
        seed: int) -> dict[str, Any]:
    honest = (
        probe_kv_bridge_v14_multi_branch_rejoin_falsifier(
            multi_branch_rejoin_flag=0.5))
    return {
        "schema": R160_SCHEMA_VERSION,
        "name": "h462_kv_v14_multi_branch_rejoin_falsifier",
        "passed": bool(honest.falsifier_score == 0.0),
    }


def family_h463_disagreement_v15_mbr_equiv(
        seed: int) -> dict[str, Any]:
    # Construct an oracle that says: argmax preserved, low KL,
    # fingerprints match → equivalence holds.
    def oracle_honest():
        return (True, 0.05, True)
    def oracle_falsifier():
        return (False, 0.5, True)
    trace = AlgebraTrace.empty()
    w_honest = emit_disagreement_algebra_v15_witness(
        trace=trace, probe_a=(0.1, 0.2),
        probe_b=(0.1, 0.2),
        mbr_oracle=oracle_honest, mbr_floor=0.20)
    w_fal = emit_disagreement_algebra_v15_witness(
        trace=trace, probe_a=(0.1, 0.2),
        probe_b=(0.1, 0.2),
        mbr_oracle=oracle_falsifier, mbr_floor=0.20)
    return {
        "schema": R160_SCHEMA_VERSION,
        "name": "h463_disagreement_v15_mbr_equiv",
        "passed": bool(
            w_honest.multi_branch_rejoin_equiv_ok
            and not w_honest.multi_branch_rejoin_falsifier_ok
            and (not w_fal.multi_branch_rejoin_equiv_ok)
            and w_fal.multi_branch_rejoin_falsifier_ok),
    }


def family_h464_compound_multi_branch_rejoin(
        seed: int) -> dict[str, Any]:
    masc = MultiAgentSubstrateCoordinatorV5()
    _, agg = masc.run_batch(
        seeds=list(range(15)),
        regime=W69_MASC_V5_REGIME_MULTI_BRANCH_REJOIN)
    return {
        "schema": R160_SCHEMA_VERSION,
        "name": "h464_compound_multi_branch_rejoin",
        "passed": bool(
            agg.v14_beats_v13_rate >= 0.5
            and agg.tsc_v14_beats_tsc_v13_rate >= 0.5),
        "v14_beats_v13": float(agg.v14_beats_v13_rate),
        "tsc_v14_beats_tsc_v13": float(
            agg.tsc_v14_beats_tsc_v13_rate),
    }


def family_h465_compound_silent_corruption(
        seed: int) -> dict[str, Any]:
    masc = MultiAgentSubstrateCoordinatorV5()
    _, agg = masc.run_batch(
        seeds=list(range(15)),
        regime=(
            W69_MASC_V5_REGIME_SILENT_CORRUPTION_PLUS_REPLACEMENT))
    return {
        "schema": R160_SCHEMA_VERSION,
        "name": "h465_compound_silent_corruption",
        "passed": bool(
            agg.v14_beats_v13_rate >= 0.5
            and agg.tsc_v14_beats_tsc_v13_rate >= 0.5),
        "v14_beats_v13": float(agg.v14_beats_v13_rate),
        "tsc_v14_beats_tsc_v13": float(
            agg.tsc_v14_beats_tsc_v13_rate),
    }


def family_h466_w69_substrate_is_numpy_cpu(
        seed: int) -> dict[str, Any]:
    """Limitation reproduction: V14 substrate is 16-layer NumPy
    byte-tokenised, NOT a frontier model."""
    return {
        "schema": R160_SCHEMA_VERSION,
        "name": "h466_w69_substrate_is_numpy_cpu",
        "passed": bool(
            W69_DEFAULT_V14_N_LAYERS == 16
            and W69_TINY_V14_VOCAB_SIZE == 259),
    }


def family_h467_w69_hosted_does_not_pierce_wall(
        seed: int) -> dict[str, Any]:
    """Limitation reproduction: every blocked axis at hosted
    rejects a claim that hosted satisfies it."""
    b = build_default_hosted_real_substrate_boundary_v2()
    all_block = True
    for ax in b.blocked_axes:
        f = probe_hosted_real_substrate_boundary_v2_falsifier(
            boundary=b, claimed_axis=str(ax),
            claim_satisfied_at_hosted=True)
        if f.falsifier_score != 1.0:
            all_block = False
            break
    return {
        "schema": R160_SCHEMA_VERSION,
        "name": "h467_w69_hosted_does_not_pierce_wall",
        "passed": bool(all_block),
    }


_R160_FAMILIES: tuple[Any, ...] = (
    family_h460_boundary_v2_falsifier,
    family_h461_ecc_v21_rate_ceiling,
    family_h462_kv_v14_multi_branch_rejoin_falsifier,
    family_h463_disagreement_v15_mbr_equiv,
    family_h464_compound_multi_branch_rejoin,
    family_h465_compound_silent_corruption,
    family_h466_w69_substrate_is_numpy_cpu,
    family_h467_w69_hosted_does_not_pierce_wall,
)


def run_r160(
        seeds: Sequence[int] = (199, 299, 399),
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for s in seeds:
        results: dict[str, Any] = {}
        for fn in _R160_FAMILIES:
            try:
                r = fn(int(s))
            except Exception as e:
                r = {
                    "schema": R160_SCHEMA_VERSION,
                    "name": getattr(fn, "__name__", "unknown"),
                    "passed": False,
                    "exception": str(e),
                }
            results[r["name"]] = r
        out.append({
            "schema": R160_SCHEMA_VERSION,
            "seed": int(s),
            "family_results": results,
        })
    return out


__all__ = ["R160_SCHEMA_VERSION", "run_r160"]
