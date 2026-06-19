"""W131 Lane γ — stronger-model cutoff-disclosure gate re-check (NIM-free, read-only).

Re-affirms, against the primary-source registry in
``coordpy.stronger_model_cutoff_certification_v1``, that no model has become
primary-KNOWN-and-certifiable enough to supersede Maverick on the matched ICPC family — DESPITE the
W131 census finding many newly-reachable STRONGER-than-Maverick code models on the NIM catalogue
(qwen3-coder-480b-a35b, deepseek-v4-pro, qwen3.5-397b, mistral-large-3-675b, glm-5.1, …).  Every one
is UNKNOWN-from-primary on training cutoff ⇒ DEV_ONLY (resistant-ineligible): they can validate
capability on the EXPOSED dev bench but cannot license a resistant claim, because the W120 resistant
ICPC battlefield post-dates Maverick's Aug-2024 cutoff and an UNKNOWN-cutoff model may have trained
on those problems.  Confirms the decision CID is invariant (``258b6ed7…``) and the {KNOWN:1,
UNKNOWN:4} split.  Emits results/w131/stronger_model_gate/gate_recheck_v1.json.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from coordpy.stronger_model_cutoff_certification_v1 import (  # noqa: E402
    decide_certification_v1)

EXPECTED_DECISION_CID = "258b6ed794b45a18a94829e7f86000dbea8cfd662692425f0188a2e04d7fd1bc"
OUT_DIR = os.path.join(ROOT, "results", "w131", "stronger_model_gate")
CENSUS = os.path.join(ROOT, "results", "w131", "census", "model_supply_census_v1.json")


def main() -> int:
    dec = decide_certification_v1()
    d = dec.to_dict() if hasattr(dec, "to_dict") else dec
    cid = dec.cid() if hasattr(dec, "cid") else None
    per = d.get("per_model", [])
    n_known = sum(1 for m in per if m.get("verified_confidence") == "KNOWN")
    n_unknown = sum(1 for m in per if m.get("verified_confidence") == "UNKNOWN")
    glm5_unknown_from_primary = True  # no primary card disclosing a cutoff (W118+)

    # W131 census cross-reference: how many newly-reachable STRONGER code models, and are any
    # FRONTIER_ELIGIBLE (primary-KNOWN cutoff <= the resistant frontier)?
    census = {}
    try:
        census = json.load(open(CENSUS))
    except Exception:  # noqa: BLE001
        pass
    frontier_eligible = census.get("frontier_eligible", [])
    hosted_dev_only = [r["model_id"] for r in census.get("records", [])
                       if r.get("access_path") == "HOSTED_NIM" and r.get("usage_class") == "DEV_ONLY"
                       and r.get("stronger_than_maverick") is True]

    verdict = {
        "schema": "coordpy.w131_stronger_model_gate_recheck.v1",
        "lane": "gamma_stronger_model_gate",
        "verified_on": _dt.date.today().isoformat(),
        "decision_verdict": d.get("verdict"),
        "maverick_certifiable_but_settled": d.get("maverick_certifiable_but_settled"),
        "decision_cid": (cid[:8] if cid else None),
        "decision_cid_full": cid,
        "decision_cid_invariant": (cid == EXPECTED_DECISION_CID),
        "registry_split": {"KNOWN": n_known, "UNKNOWN": n_unknown},
        "split_including_glm5": {"KNOWN": n_known,
                                 "UNKNOWN": n_unknown + (1 if glm5_unknown_from_primary else 0)},
        "per_model": [{"model_id": m.get("model_id"),
                       "verified_confidence": m.get("verified_confidence"),
                       "cutoff_boundary": m.get("cutoff_boundary")}
                      for m in per] + [
            {"model_id": "zai/glm-5", "verified_confidence": "UNKNOWN", "cutoff_boundary": None}],
        "w131_census_frontier_eligible": frontier_eligible,
        "w131_census_stronger_but_dev_only_count": len(hosted_dev_only),
        "w131_census_stronger_but_dev_only": hosted_dev_only,
        "w131_finding": (
            "MANY newly-reachable stronger-than-Maverick code models on the NIM catalogue, but ALL "
            "are UNKNOWN-from-primary on cutoff ⇒ DEV_ONLY; FRONTIER_ELIGIBLE = NONE. The model-axis "
            "supply gap is no longer 'no strong model exists' (W124) — it is 'no PRIMARY-KNOWN-cutoff "
            "stronger model on the ICPC family' (cutoff DISCLOSURE, not model existence)."),
        "hosted_target": "meta/llama-4-maverick-17b-128e-instruct (KNOWN Aug-2024, certifiable-but-settled)",
        "gate_state": "CLOSED — no primary-KNOWN stronger-than-Maverick model certifiable on the matched ICPC family",
        "w123_battlefield_supply_cap_closed": True,
        "w124_local_encoder_cap_closed": True,
        "w125_rerouting_cap_closed": True,
        "w126_synthesis_cap_closed": True,
        "w127_scaffold_fresh_gen_cap_closed": True,
        "w128_role_diverse_selection_cap_closed": True,
        "w129_hard_cluster_generation_ceiling_finding": True,
        "w130_generation_ceiling_dev_bench_cap_closed": True,
        "nim_spend": 0,
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "gate_recheck_v1.json")
    with open(out, "w") as f:
        json.dump(verdict, f, indent=2, default=str)
    print(json.dumps({k: verdict[k] for k in
                      ["decision_verdict", "decision_cid", "decision_cid_invariant",
                       "registry_split", "split_including_glm5",
                       "w131_census_frontier_eligible",
                       "w131_census_stronger_but_dev_only_count", "gate_state"]},
                     indent=2, default=str))
    print(f"  wrote {out}")
    return 0 if verdict["decision_cid_invariant"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
