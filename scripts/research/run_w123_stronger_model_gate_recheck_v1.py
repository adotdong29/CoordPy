#!/usr/bin/env python3
"""W123 Lane-γ — stronger-model gate re-confirmation.

Re-confirms the per-model disclosure gate from the AUTHORITATIVE recorded W120 +
W121 certification verdicts (primary sources last directly re-fetched 2026-05-30
in W120), asserts the gate is unchanged, and records its W123-specific
mootness: even a primary-KNOWN stronger model could not run a large-n matched
pilot because Lane α proved the ≥100 battlefield is supply-unreachable.

NIM-free, read-only. Emits results/w123/stronger_model_gate/gate_recheck_v1.json.
"""

from __future__ import annotations

import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W120 = os.path.join(REPO, "results", "w120", "icpc_battlefield", "battlefield_verdict.json")
W121 = os.path.join(REPO, "results", "w121", "exposed_control", "exposed_control_verdict.json")
W123_SUPPLY = os.path.join(REPO, "results", "w123", "largen_supply", "supply_census_verdict_v1.json")
OUT_DIR = os.path.join(REPO, "results", "w123", "stronger_model_gate")
OUT = os.path.join(OUT_DIR, "gate_recheck_v1.json")

EXPECTED_DECISION_CID = "258b6ed794b45a18a94829e7f86000dbea8cfd662692425f0188a2e04d7fd1bc"


def _load(p):
    with open(p) as fh:
        return json.load(fh)


def main() -> int:
    w120 = _load(W120)["result"]
    w121 = _load(W121)["result"]
    supply = _load(W123_SUPPLY) if os.path.isfile(W123_SUPPLY) else {}

    d120 = w120["disclosure_summary"]["counts"]
    d121 = w121["disclosure_summary"]["counts"]
    matrix = w120["disclosure_matrix"]

    gate_unchanged = (
        d120 == d121 == {"KNOWN": 1, "UNKNOWN": 4}
        and w120["lcb_inherited_decision_cid"] == EXPECTED_DECISION_CID
        and w121["lcb_inherited_decision_cid"] == EXPECTED_DECISION_CID
        and not w120["disclosure_summary"]["any_usable_new_known_cutoff_target"]
    )

    # W123 mootness: the gate is not just structurally closed, it is moot —
    # the large-n battlefield Lane β would need cannot be built (Lane α).
    largen_unreachable = supply.get("verdict", "").startswith(
        "LARGEN_MATCHED_BATTLEFIELD_UNREACHABLE")

    verdict = {
        "schema": "coordpy.w123_stronger_model_gate_recheck.v1",
        "milestone": "W123",
        "lane": "gamma",
        "primary_sources_last_direct_refetch": "2026-05-30 (W120)",
        "disclosure_counts_w120": d120,
        "disclosure_counts_w121": d121,
        "decision_cid": EXPECTED_DECISION_CID[:8],
        "decision_cid_invariant": w120["lcb_inherited_decision_cid"] == EXPECTED_DECISION_CID,
        "per_model": [
            {"model_id": m["model_id"], "primary_status": m["primary_status"],
             "blocker": m["certifiable_blocker"]}
            for m in matrix
        ],
        "any_usable_new_known_cutoff_target": w120["disclosure_summary"][
            "any_usable_new_known_cutoff_target"],
        "gate_unchanged_vs_w120_w121": gate_unchanged,
        "gate_structurally_closed": True,
        "gate_moot_because_largen_unreachable": largen_unreachable,
        "verdict": (
            "STRONGER_MODEL_GATE_CLOSED_AND_MOOT" if (gate_unchanged and largen_unreachable)
            else "STRONGER_MODEL_GATE_CLOSED" if gate_unchanged
            else "GATE_CHANGED_INVESTIGATE"
        ),
        "note": (
            "Only meta/llama-4-maverick (Aug-2024) is primary-KNOWN. "
            "Qwen3-Coder-480B / DeepSeek-V4-pro / Mistral-Small-4-119B-2603 / GLM-5 "
            "remain UNKNOWN from primary sources. Even a primary-KNOWN stronger model "
            "could not run a large-n matched pilot: Lane α proved the ≥100 battlefield "
            "is supply-unreachable from the official ICPC family."
        ),
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(verdict, fh, indent=2, sort_keys=True)
        fh.write("\n")

    print("W123 Lane-γ stronger-model gate re-check")
    print(f"  disclosure counts : W120={d120} W121={d121}")
    print(f"  decision CID inv. : {verdict['decision_cid_invariant']} ({verdict['decision_cid']})")
    print(f"  gate unchanged    : {gate_unchanged}")
    print(f"  gate moot (Lane α): {largen_unreachable}")
    print(f"  verdict           : {verdict['verdict']}")
    print(f"wrote {OUT}")
    return 0 if verdict["verdict"] != "GATE_CHANGED_INVESTIGATE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
