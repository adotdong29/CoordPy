#!/usr/bin/env python3
"""W113-β — tier-2 stronger-model readiness artifact (NIM-free).

Emits the LOCKED tier-2 ranking + same-filtered-slice applicability + spend rule
(``coordpy.tier2_readiness_v1``) for the W113 resistant slice, under BOTH
main-lane outcome branches, so the no-spend / spend decision is auditable on
disk regardless of how the Maverick pilot lands.  $0 NIM.

Usage::

    python scripts/run_w113_tier2_readiness_v1.py
"""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import sys  # noqa: E402
sys.path.insert(0, str(ROOT))

from coordpy.tier2_readiness_v1 import (  # noqa: E402
    TIER2_RANKING,
    assess_tier2_applicability_v1,
    decide_tier2_spend_v1,
)

# The W113 resistant slice's minimum contest_date (from the preflight P6).
W113_RESISTANT_SLICE_DATE_MIN = "2025-01-11"

OUT = ROOT / "results" / "w113" / "tier2_readiness"


def main() -> int:
    applic = assess_tier2_applicability_v1(
        slice_date_min=W113_RESISTANT_SLICE_DATE_MIN)
    branches = {
        outcome: decide_tier2_spend_v1(
            main_lane_outcome=outcome,
            slice_date_min=W113_RESISTANT_SLICE_DATE_MIN).to_dict()
        for outcome in ("RESISTANT_SUPERIORITY_REOPENS",
                        "RESISTANT_MARGIN_NON_MECHANISM",
                        "EXPOSURE_CONFIRMED")
    }
    artifact = {
        "schema": "coordpy.w113_tier2_readiness.v1",
        "milestone": "W113-beta",
        "slice_date_min": W113_RESISTANT_SLICE_DATE_MIN,
        "ranking": [
            {"model_id": c.model_id, "family": c.family,
             "rank_tier": c.rank_tier, "rank_within_tier": c.rank_within_tier,
             "note": c.note}
            for c in TIER2_RANKING],
        "applicability": [a.to_dict() for a in applic],
        "spend_decision_by_outcome": branches,
        "ts_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tier2_readiness.json").write_text(
        json.dumps(artifact, indent=2, default=str))

    print("=== W113-β tier-2 stronger-model readiness ===")
    print(f"  resistant slice min date: {W113_RESISTANT_SLICE_DATE_MIN}")
    print("  ranking + applicability:")
    for a in applic:
        print(f"    [{a.rank_within_tier}] {a.model_id:42s} "
              f"cutoff={a.cutoff_boundary}[{a.cutoff_confidence}] "
              f"certifiably_resistant={a.slice_certifiably_resistant}")
    print("  spend decision (all branches):")
    for outcome, d in branches.items():
        print(f"    {outcome:34s} earns={d['main_lane_earns_escalation']} "
              f"n_certifiable={d['n_certifiable_targets']} "
              f"spend_eligible={d['spend_eligible']}")
    any_spend = any(d["spend_eligible"] for d in branches.values())
    print(f"  ANY tier-2 spend eligible in W113: {any_spend}")
    print(f"  next instrument if blocked: "
          f"{branches['EXPOSURE_CONFIRMED']['next_instrument_if_blocked'][:80]}")
    print(f"  artifact: {OUT / 'tier2_readiness.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
