"""W74 R-180 benchmark family — Handoff V6 + Falsifier + Limitation.

The W74 falsifier-and-limitation-reproduction family + Plane A↔B
compound-aware handoff family.

H950..H963 cell families (14 H-bars):

* H950   Handoff V6 envelope is content-addressed
* H951   Handoff V6 text-only routes to plane A
* H952   Handoff V6 substrate-only routes to plane B
* H953   Handoff V6 compound_pressure ≥ floor + substrate_trust
         ≥ floor promotes to real_substrate_only with
         compound_alignment = 1.0
* H954   Handoff V6 compound_repair_trajectory + compound_window
         > 0 falls back to
         compound_repair_after_delayed_repair_then_replacement_fallback
* H955   Handoff V6 cross-plane savings ≥ 82 %
* H956   Handoff V6 compound falsifier honest=0, dishonest=1
* H957   Boundary V7 falsifier honest=0 / dishonest=1
* H958   KV V19 compound-pressure falsifier honest = 0
* H959   MASC V10 compound regime
         (replacement_after_delayed_repair_under_budget) ≥ 50 %
         strict-beat
* H960   W74 substrate is in-repo NumPy (limitation reproduction)
* H961   W74 hosted control plane V7 does NOT pierce wall
         (limitation reproduction; 34 blocked-axes)
* H962   No-version-bump invariant: ``coordpy.__version__ ==
         "0.5.20"`` and ``SDK_VERSION == "coordpy.sdk.v3.43"``
* H963   Frontier substrate access still blocked (V7 frontier-
         blocked axes set non-empty and unchanged from W70)
"""

from __future__ import annotations

from typing import Any, Sequence

from coordpy import SDK_VERSION, __version__
from coordpy.hosted_real_handoff_coordinator import HandoffRequest
from coordpy.hosted_real_handoff_coordinator_v2 import (
    HandoffRequestV2,
)
from coordpy.hosted_real_handoff_coordinator_v3 import (
    HandoffRequestV3,
)
from coordpy.hosted_real_handoff_coordinator_v4 import (
    HandoffRequestV4,
)
from coordpy.hosted_real_handoff_coordinator import (
    W69_HANDOFF_DECISION_HOSTED_ONLY,
    W69_HANDOFF_DECISION_REAL_SUBSTRATE_ONLY,
)
from coordpy.hosted_real_handoff_coordinator_v5 import (
    HandoffRequestV5,
)
from coordpy.hosted_real_handoff_coordinator_v6 import (
    HandoffRequestV6, HostedRealHandoffCoordinatorV6,
    W74_HANDOFF_DECISION_COMPOUND_REPAIR_FALLBACK,
    hosted_real_handoff_v6_compound_aware_savings,
    probe_hosted_real_handoff_v6_compound_falsifier,
)
from coordpy.hosted_real_substrate_boundary_v7 import (
    W74_FRONTIER_BLOCKED_AXES,
    build_default_hosted_real_substrate_boundary_v7,
    probe_hosted_real_substrate_boundary_v7_falsifier,
)
from coordpy.hosted_real_substrate_boundary_v3 import (
    W70_FRONTIER_BLOCKED_AXES,
)
from coordpy.kv_bridge_v19 import (
    probe_kv_bridge_v19_compound_pressure_falsifier,
)
from coordpy.multi_agent_substrate_coordinator_v10 import (
    MultiAgentSubstrateCoordinatorV10,
    W74_MASC_V10_REGIME_REPLACEMENT_AFTER_DELAYED_REPAIR,
)
from coordpy.tiny_substrate_v19 import (
    W74_DEFAULT_V19_N_LAYERS, W74_TINY_V19_VOCAB_SIZE,
)


R180_SCHEMA_VERSION: str = "coordpy.r180_benchmark.v1"


def _req_v6(*, rc: str, compound_pressure: float = 0.0,
            compound_repair_trajectory_cid: str = "",
            compound_window_turns: int = 0,
            needs_text_only: bool = True,
            needs_substrate_state_access: bool = False,
            restart_pressure: float = 0.0,
            rejoin_pressure: float = 0.0,
            replacement_pressure: float = 0.0,
            dominant_repair_label: int = 0,
            visible_token_budget: int = 256,
            baseline_token_cost: int = 512,
            ) -> HandoffRequestV6:
    return HandoffRequestV6(
        inner_v5=HandoffRequestV5(
            inner_v4=HandoffRequestV4(
                inner_v3=HandoffRequestV3(
                    inner_v2=HandoffRequestV2(
                        inner_v1=HandoffRequest(
                            request_cid=str(rc),
                            needs_text_only=bool(
                                needs_text_only),
                            needs_substrate_state_access=bool(
                                needs_substrate_state_access)),
                        visible_token_budget=int(
                            visible_token_budget),
                        baseline_token_cost=int(
                            baseline_token_cost),
                        dominant_repair_label=int(
                            dominant_repair_label)),
                    restart_pressure=float(restart_pressure),
                    delayed_repair_trajectory_cid="",
                    delay_turns=0,
                    expected_substrate_trust=0.7),
                rejoin_pressure=float(rejoin_pressure),
                restart_repair_trajectory_cid="",
                rejoin_lag_turns=0,
                expected_substrate_trust_v4=0.7),
            replacement_pressure=float(replacement_pressure),
            replacement_repair_trajectory_cid="",
            replacement_lag_turns=0,
            expected_substrate_trust_v5=0.7),
        compound_pressure=float(compound_pressure),
        compound_repair_trajectory_cid=str(
            compound_repair_trajectory_cid),
        compound_window_turns=int(compound_window_turns),
        expected_substrate_trust_v6=0.7)


def family_h950_handoff_v6_envelope_content_addressed(
        seed: int) -> dict[str, Any]:
    c1 = HostedRealHandoffCoordinatorV6()
    c2 = HostedRealHandoffCoordinatorV6()
    req = _req_v6(rc=f"h950_{int(seed)}")
    e1 = c1.decide_v6(
        req_v6=req, substrate_self_checksum_cid="x")
    e2 = c2.decide_v6(
        req_v6=req, substrate_self_checksum_cid="x")
    return {
        "schema": R180_SCHEMA_VERSION,
        "name":
            "h950_handoff_v6_envelope_content_addressed",
        "passed": bool(e1.cid() == e2.cid()),
    }


def family_h951_handoff_v6_text_only_route(
        seed: int) -> dict[str, Any]:
    c = HostedRealHandoffCoordinatorV6()
    e = c.decide_v6(
        req_v6=_req_v6(rc=f"h951_{int(seed)}"))
    return {
        "schema": R180_SCHEMA_VERSION,
        "name": "h951_handoff_v6_text_only_route",
        "passed": bool(
            str(e.decision_v6)
            == W69_HANDOFF_DECISION_HOSTED_ONLY),
    }


def family_h952_handoff_v6_substrate_only_route(
        seed: int) -> dict[str, Any]:
    c = HostedRealHandoffCoordinatorV6()
    e = c.decide_v6(
        req_v6=_req_v6(
            rc=f"h952_{int(seed)}",
            needs_text_only=False,
            needs_substrate_state_access=True,
            restart_pressure=0.3,
            dominant_repair_label=1,
            visible_token_budget=128))
    return {
        "schema": R180_SCHEMA_VERSION,
        "name": "h952_handoff_v6_substrate_only_route",
        "passed": bool(
            str(e.decision_v6)
            == W69_HANDOFF_DECISION_REAL_SUBSTRATE_ONLY),
    }


def family_h953_handoff_v6_compound_promotion(
        seed: int) -> dict[str, Any]:
    c = HostedRealHandoffCoordinatorV6()
    e = c.decide_v6(
        req_v6=_req_v6(
            rc=f"h953_{int(seed)}",
            compound_pressure=0.8))
    return {
        "schema": R180_SCHEMA_VERSION,
        "name": "h953_handoff_v6_compound_promotion",
        "passed": bool(
            str(e.decision_v6)
            == W69_HANDOFF_DECISION_REAL_SUBSTRATE_ONLY
            and e.compound_alignment == 1.0),
    }


def family_h954_handoff_v6_compound_fallback(
        seed: int) -> dict[str, Any]:
    c = HostedRealHandoffCoordinatorV6()
    e = c.decide_v6(
        req_v6=_req_v6(
            rc=f"h954_{int(seed)}",
            compound_repair_trajectory_cid="cmp_xyz",
            compound_window_turns=8))
    return {
        "schema": R180_SCHEMA_VERSION,
        "name": "h954_handoff_v6_compound_fallback",
        "passed": bool(
            str(e.decision_v6)
            == W74_HANDOFF_DECISION_COMPOUND_REPAIR_FALLBACK
            and e.compound_repair_drtr_fallback_active
            and e.compound_repair_drtr_alignment == 1.0),
    }


def family_h955_handoff_v6_cross_plane_savings(
        seed: int) -> dict[str, Any]:
    sav = hosted_real_handoff_v6_compound_aware_savings(
        n_turns=14)
    return {
        "schema": R180_SCHEMA_VERSION,
        "name": "h955_handoff_v6_cross_plane_savings",
        "passed": bool(sav["saving_ratio"] >= 0.82),
        "ratio": float(sav["saving_ratio"]),
    }


def family_h956_handoff_v6_compound_falsifier(
        seed: int) -> dict[str, Any]:
    c = HostedRealHandoffCoordinatorV6()
    e = c.decide_v6(
        req_v6=_req_v6(rc=f"h956_{int(seed)}"))
    honest = (
        probe_hosted_real_handoff_v6_compound_falsifier(
            envelope_v6=e, claim_satisfied=False))
    dishonest = (
        probe_hosted_real_handoff_v6_compound_falsifier(
            envelope_v6=e, claim_satisfied=True))
    return {
        "schema": R180_SCHEMA_VERSION,
        "name":
            "h956_handoff_v6_compound_falsifier",
        "passed": bool(
            honest.falsifier_score == 0.0
            and dishonest.falsifier_score == 1.0),
    }


def family_h957_boundary_v7_falsifier(
        seed: int) -> dict[str, Any]:
    b = build_default_hosted_real_substrate_boundary_v7()
    honest = probe_hosted_real_substrate_boundary_v7_falsifier(
        boundary=b,
        claimed_axis="compound_repair_trajectory_cid",
        claim_satisfied_at_hosted=False)
    dishonest = (
        probe_hosted_real_substrate_boundary_v7_falsifier(
            boundary=b,
            claimed_axis="compound_repair_trajectory_cid",
            claim_satisfied_at_hosted=True))
    return {
        "schema": R180_SCHEMA_VERSION,
        "name": "h957_boundary_v7_falsifier",
        "passed": bool(
            honest.falsifier_score == 0.0
            and dishonest.falsifier_score == 1.0),
    }


def family_h958_kv_v19_compound_pressure_falsifier(
        seed: int) -> dict[str, Any]:
    honest = probe_kv_bridge_v19_compound_pressure_falsifier(
        compound_pressure_flag=1)
    return {
        "schema": R180_SCHEMA_VERSION,
        "name":
            "h958_kv_v19_compound_pressure_falsifier",
        "passed": bool(honest.falsifier_score == 0.0),
    }


def family_h959_compound_replacement_after_dr(
        seed: int) -> dict[str, Any]:
    masc = MultiAgentSubstrateCoordinatorV10()
    _, agg = masc.run_batch(
        seeds=list(range(15)),
        regime=W74_MASC_V10_REGIME_REPLACEMENT_AFTER_DELAYED_REPAIR)
    return {
        "schema": R180_SCHEMA_VERSION,
        "name":
            "h959_compound_replacement_after_dr",
        "passed": bool(
            agg.v19_beats_v18_rate >= 0.5
            and agg.tsc_v19_beats_tsc_v18_rate >= 0.5),
        "v19_beats_v18": float(agg.v19_beats_v18_rate),
        "tsc_v19_beats_tsc_v18": float(
            agg.tsc_v19_beats_tsc_v18_rate),
    }


def family_h960_w74_substrate_is_numpy_cpu(
        seed: int) -> dict[str, Any]:
    """Limitation reproduction: V19 substrate is 21-layer NumPy
    byte-tokenised, NOT a frontier model."""
    return {
        "schema": R180_SCHEMA_VERSION,
        "name": "h960_w74_substrate_is_numpy_cpu",
        "passed": bool(
            W74_DEFAULT_V19_N_LAYERS == 21
            and W74_TINY_V19_VOCAB_SIZE == 259),
    }


def family_h961_w74_hosted_does_not_pierce_wall(
        seed: int) -> dict[str, Any]:
    """Limitation reproduction: every blocked axis at hosted V7
    rejects a claim that hosted satisfies it."""
    b = build_default_hosted_real_substrate_boundary_v7()
    all_block = True
    for ax in b.blocked_axes:
        f = probe_hosted_real_substrate_boundary_v7_falsifier(
            boundary=b, claimed_axis=str(ax),
            claim_satisfied_at_hosted=True)
        if f.falsifier_score != 1.0:
            all_block = False
            break
    return {
        "schema": R180_SCHEMA_VERSION,
        "name": "h961_w74_hosted_does_not_pierce_wall",
        "passed": bool(all_block),
    }


def family_h962_no_version_bump_invariant(
        seed: int) -> dict[str, Any]:
    return {
        "schema": R180_SCHEMA_VERSION,
        "name": "h962_no_version_bump_invariant",
        "passed": bool(
            str(__version__) == "0.5.20"
            and str(SDK_VERSION) == "coordpy.sdk.v3.43"),
    }


def family_h963_frontier_still_blocked(
        seed: int) -> dict[str, Any]:
    """Frontier-blocked axes set is non-empty and equal to W70's
    set; W74 does not unblock anything at the frontier."""
    return {
        "schema": R180_SCHEMA_VERSION,
        "name": "h963_frontier_still_blocked",
        "passed": bool(
            len(W74_FRONTIER_BLOCKED_AXES) >= 1
            and (
                tuple(W74_FRONTIER_BLOCKED_AXES)
                == tuple(W70_FRONTIER_BLOCKED_AXES))),
    }


_R180_FAMILIES: tuple[Any, ...] = (
    family_h950_handoff_v6_envelope_content_addressed,
    family_h951_handoff_v6_text_only_route,
    family_h952_handoff_v6_substrate_only_route,
    family_h953_handoff_v6_compound_promotion,
    family_h954_handoff_v6_compound_fallback,
    family_h955_handoff_v6_cross_plane_savings,
    family_h956_handoff_v6_compound_falsifier,
    family_h957_boundary_v7_falsifier,
    family_h958_kv_v19_compound_pressure_falsifier,
    family_h959_compound_replacement_after_dr,
    family_h960_w74_substrate_is_numpy_cpu,
    family_h961_w74_hosted_does_not_pierce_wall,
    family_h962_no_version_bump_invariant,
    family_h963_frontier_still_blocked,
)


def run_r180(
        seeds: Sequence[int] = (199, 299, 399),
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for s in seeds:
        results: dict[str, Any] = {}
        for fn in _R180_FAMILIES:
            try:
                r = fn(int(s))
            except Exception as e:
                r = {
                    "schema": R180_SCHEMA_VERSION,
                    "name": getattr(fn, "__name__", "unknown"),
                    "passed": False,
                    "exception": str(e),
                }
            results[r["name"]] = r
        out.append({
            "schema": R180_SCHEMA_VERSION,
            "seed": int(s),
            "family_results": results,
        })
    return out


__all__ = ["R180_SCHEMA_VERSION", "run_r180"]
