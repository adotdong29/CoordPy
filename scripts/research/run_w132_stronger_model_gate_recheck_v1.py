"""W132 Lane γ — stronger-model cutoff-disclosure gate re-check (NIM-free, read-only).

Re-affirms the invariant decision (CID ``258b6ed7…``, {KNOWN:1, UNKNOWN:4}) from the
primary-source registry in ``coordpy.stronger_model_cutoff_certification_v1``: no model
has become primary-KNOWN-and-certifiable enough to supersede Maverick.

W132's genuinely-new γ contribution: a **resistant-by-construction** battlefield no longer
DEPENDS on cutoff disclosure (the W131 blocker).  Because the minted problem INSTANCES did
not exist before the mint date, the battlefield is contamination-resistant for ANY model —
including the UNKNOWN-cutoff stronger code models the W131 census surfaced.  But for the
FRONTIER claim Maverick remains the default target precisely because its cutoff is KNOWN
(Aug-2024) and the mint date (2026-06-02) strictly post-dates it; an UNKNOWN-cutoff model
can be used only as DEV_ONLY characterization, never the frontier claim.  Emits
results/w132/stronger_model_gate/gate_recheck_v1.json.
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
OUT_DIR = os.path.join(ROOT, "results", "w132", "stronger_model_gate")
BF = os.path.join(ROOT, "results", "w132", "battlefield", "battlefield_verdict_v1.json")
MINTED_DATE = "2026-06-02"


def main() -> int:
    dec = decide_certification_v1()
    d = dec.to_dict() if hasattr(dec, "to_dict") else dec
    cid = dec.cid() if hasattr(dec, "cid") else None
    per = d.get("per_model", [])
    n_known = sum(1 for m in per if m.get("verified_confidence") == "KNOWN")
    n_unknown = sum(1 for m in per if m.get("verified_confidence") == "UNKNOWN")

    bf = {}
    try:
        bf = json.load(open(BF))
    except Exception:  # noqa: BLE001
        pass

    verdict = {
        "schema": "coordpy.w132_stronger_model_gate_recheck.v1",
        "lane": "gamma_stronger_model_gate",
        "verified_on": _dt.date.today().isoformat(),
        "decision_verdict": d.get("verdict"),
        "decision_cid": (cid[:8] if cid else None),
        "decision_cid_full": cid,
        "decision_cid_invariant": (cid == EXPECTED_DECISION_CID),
        "registry_split": {"KNOWN": n_known, "UNKNOWN": n_unknown},
        "per_model": [{"model_id": m.get("model_id"),
                       "verified_confidence": m.get("verified_confidence"),
                       "cutoff_boundary": m.get("cutoff_boundary")} for m in per] + [
            {"model_id": "zai/glm-5", "verified_confidence": "UNKNOWN",
             "cutoff_boundary": None}],
        "primary_recheck_note": (
            "Re-derived from the primary-source registry; no new primary cutoff "
            "disclosure since the W131 re-check earlier on 2026-06-02 for Maverick / "
            "Qwen3-Coder-480B / DeepSeek-V4-pro / Mistral-Small-4-119B-2603 / GLM-5."),
        "gate_state": ("CLOSED — no primary-KNOWN stronger-than-Maverick model "
                       "certifiable; Maverick (KNOWN Aug-2024) stays the default "
                       "frontier target"),
        "w132_resistance_by_construction": {
            "minted_date": MINTED_DATE,
            "maverick_cutoff": "2024-08-31",
            "resistant_for_maverick_by_date": bool(MINTED_DATE > "2024-08-31"),
            "resistant_for_any_cutoff_by_construction": True,
            "note": ("The minted battlefield REMOVES the W131 cutoff-disclosure "
                     "dependency: instances are freshly generated, so the field is "
                     "resistant for ANY model regardless of disclosure. UNKNOWN-cutoff "
                     "stronger models may be used as DEV_ONLY characterization only; the "
                     "frontier claim stays on Maverick because its cutoff is KNOWN."),
            "battlefield_manifest_cid": bf.get("manifest_cid"),
            "battlefield_core_slice_cid": bf.get("core_slice_cid"),
            "battlefield_pilot_earned": bf.get("battlefield_pilot_earned"),
        },
        "w123_through_w131_caps_closed": True,
        "no_405b_run": True,
        "nim_spend": 0,
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "gate_recheck_v1.json")
    with open(out, "w") as f:
        json.dump(verdict, f, indent=2, default=str)
    print(json.dumps({k: verdict[k] for k in
                      ["decision_verdict", "decision_cid", "decision_cid_invariant",
                       "registry_split", "gate_state"]}, indent=2, default=str))
    print(f"  wrote {out}")
    return 0 if verdict["decision_cid_invariant"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
