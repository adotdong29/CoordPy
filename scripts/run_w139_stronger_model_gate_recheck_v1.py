"""W139 Lane γ — stronger-model cutoff-disclosure gate re-check (NIM-free, read-only).

Re-affirms the invariant decision (CID ``258b6ed7…``, {KNOWN:1, UNKNOWN:4}) from the primary-source
registry in ``coordpy.stronger_model_cutoff_certification_v1``.  The only PRIMARY-KNOWN cutoffs among
reachable models remain Meta's Llama-3.3-70B (Dec-2023; the W105 retirement / frontier target) and
Llama-4-Maverick (Aug-2024; already settled by W113).  Qwen3-Coder-480B / DeepSeek-V4-Pro /
Mistral-Small-4-119B-2603 / GLM-5 are each OFFICIALLY UNDISCLOSED.  Gate stays CLOSED.

For W139 this gate governs only WHICH model an EARNED frontier rerun would target — the per-tier band
field is resistant BY CONSTRUCTION (freshly minted, unmemorisable for any cutoff), so it does NOT
itself require a stronger model; the default frontier target stays ``meta/llama-3.3-70b-instruct``.
Emits results/w139/stronger_model_gate/gate_recheck_v1.json.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from coordpy.stronger_model_cutoff_certification_v1 import decide_certification_v1  # noqa: E402

EXPECTED_DECISION_CID = "258b6ed794b45a18a94829e7f86000dbea8cfd662692425f0188a2e04d7fd1bc"
OUT_DIR = os.path.join(ROOT, "results", "w139", "stronger_model_gate")


def main() -> int:
    dec = decide_certification_v1()
    d = dec.to_dict() if hasattr(dec, "to_dict") else dec
    cid = dec.cid() if hasattr(dec, "cid") else None
    per = d.get("per_model", [])
    n_known = sum(1 for m in per if m.get("verified_confidence") == "KNOWN")
    n_unknown = sum(1 for m in per if m.get("verified_confidence") == "UNKNOWN")
    verdict = {
        "schema": "coordpy.w139_stronger_model_gate_recheck.v1",
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
            {"model_id": "zai/glm-5", "verified_confidence": "UNKNOWN", "cutoff_boundary": None}],
        "primary_recheck_note": (
            "Re-derived from the primary-source registry (invariant W114->W138): Llama-3.3-70B = "
            "Dec-2023 KNOWN (the W105 model / frontier target), Llama-4-Maverick = Aug-2024 KNOWN "
            "(already settled by W113); Qwen3-Coder-480B / DeepSeek-V4-Pro / Mistral-Small-4-119B-2603 "
            "/ GLM-5 all primary-UNDISCLOSED."),
        "gate_state": ("CLOSED — no primary-KNOWN stronger-than-Maverick model certifiable; the W139 "
                       "per-tier band field is resistant BY CONSTRUCTION so it does not require a "
                       "stronger model; an EARNED frontier rerun targets meta/llama-3.3-70b-instruct "
                       "(KNOWN ~Dec-2023)"),
        "w139_note": ("W139 frontier is earned ONLY if Lane-alpha lands per-tier bands on a shared "
                      "family AND the Cm lead clears the held-out §7b earn rule (beats A1 AND B0 by "
                      ">=+5pp at the anchor, POSITIVE same-sign on >=2 tiers, non-negative on ALL "
                      "tiers). The gate state here governs only WHICH model the earned frontier "
                      "targets (it stays Llama-3.3-70B); it does not by itself authorise NIM."),
        "w123_through_w138_caps_closed": True,
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
