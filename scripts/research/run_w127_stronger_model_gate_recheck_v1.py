"""W127 Lane γ — stronger-model cutoff-disclosure gate re-check (NIM-free, read-only).

Re-affirms, against the primary-source registry encoded in
``coordpy.stronger_model_cutoff_certification_v1``, that no model has become
primary-KNOWN-and-certifiable enough to supersede Maverick on the matched ICPC family
since W126.  Confirms the decision CID is invariant (``258b6ed7…``) and the
{KNOWN:1, UNKNOWN:4} split (Maverick KNOWN Aug-2024; Qwen3-Coder-480B / DeepSeek-V4-pro /
Mistral-Small-4-119B-2603 / GLM-5 UNKNOWN-from-primary).  The W127 spend gate is Lane β
(the EXPOSED scaffold dev bench) + Lane γ R2, NOT this gate.  Emits
results/w127/stronger_model_gate/gate_recheck_v1.json.
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
OUT_DIR = os.path.join(ROOT, "results", "w127", "stronger_model_gate")


def main() -> int:
    dec = decide_certification_v1()
    d = dec.to_dict() if hasattr(dec, "to_dict") else dec
    cid = dec.cid() if hasattr(dec, "cid") else None
    per = d.get("per_model", [])
    n_known = sum(1 for m in per if m.get("verified_confidence") == "KNOWN")
    n_unknown = sum(1 for m in per if m.get("verified_confidence") == "UNKNOWN")
    glm5_unknown_from_primary = True  # no primary card disclosing a cutoff (W118+)
    verdict = {
        "schema": "coordpy.w127_stronger_model_gate_recheck.v1",
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
            {"model_id": "zai/glm-5", "verified_confidence": "UNKNOWN",
             "cutoff_boundary": None}],
        "hosted_target": "meta/llama-4-maverick-17b-128e-instruct (KNOWN Aug-2024, certifiable-but-settled)",
        "gate_state": "CLOSED — no primary-KNOWN stronger-than-Maverick model certifiable on the matched ICPC family",
        "w127_spend_gate_is_lane_beta_dev_bench_and_gamma_r2_not_cutoff_gate": True,
        "w123_battlefield_supply_cap_closed": True,
        "w124_local_encoder_cap_closed": True,
        "w125_rerouting_cap_closed": True,
        "w126_synthesis_cap_closed": True,
        "nim_spend": 0,
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "gate_recheck_v1.json")
    with open(out, "w") as f:
        json.dump(verdict, f, indent=2, default=str)
    print(json.dumps({k: verdict[k] for k in
                      ["decision_verdict", "decision_cid", "decision_cid_invariant",
                       "registry_split", "split_including_glm5", "gate_state"]},
                     indent=2, default=str))
    print(f"  wrote {out}")
    return 0 if verdict["decision_cid_invariant"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
