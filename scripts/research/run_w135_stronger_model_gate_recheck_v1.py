"""W135 Lane γ — stronger-model cutoff-disclosure gate re-check (NIM-free, read-only).

Re-affirms the invariant decision (CID ``258b6ed7…``, {KNOWN:1, UNKNOWN:4}) from the
primary-source registry in ``coordpy.stronger_model_cutoff_certification_v1``, corroborated by the
W135 primary-source research pass (2026-06-03): the only PRIMARY-KNOWN cutoffs among reachable
models are Meta's Llama-3.3-70B (Dec-2023; the W105 retirement / frontier model) and Llama-4-Maverick
(Aug-2024; already settled). Qwen3-Coder-480B / DeepSeek-V4-Pro / Mistral-Small-4-119B-2603 / GLM-5
are each OFFICIALLY UNDISCLOSED in their primary sources; the new entrant MiniMax-M3 (released
2026-06-01) has no published model card / cutoff yet. Gate stays CLOSED; frontier target stays
`meta/llama-3.3-70b-instruct`. Emits results/w135/stronger_model_gate/gate_recheck_v1.json.
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
OUT_DIR = os.path.join(ROOT, "results", "w135", "stronger_model_gate")


def main() -> int:
    dec = decide_certification_v1()
    d = dec.to_dict() if hasattr(dec, "to_dict") else dec
    cid = dec.cid() if hasattr(dec, "cid") else None
    per = d.get("per_model", [])
    n_known = sum(1 for m in per if m.get("verified_confidence") == "KNOWN")
    n_unknown = sum(1 for m in per if m.get("verified_confidence") == "UNKNOWN")
    verdict = {
        "schema": "coordpy.w135_stronger_model_gate_recheck.v1",
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
            "Re-derived from the primary-source registry AND the W135 primary-source research pass "
            "(2026-06-03): Llama-3.3-70B = Dec-2023 KNOWN (Meta llama3_3 MODEL_CARD), "
            "Llama-4-Maverick = Aug-2024 KNOWN (Meta llama4 MODEL_CARD, already settled); "
            "Qwen3-Coder-480B / DeepSeek-V4-Pro (NIM card: 'Training Data Collection: Undisclosed') / "
            "Mistral-Small-4-119B-2603 / GLM-5 (arXiv:2602.15763, no cutoff stated) all "
            "primary-UNDISCLOSED; new entrant MiniMax-M3 (2026-06-01) has no published card/cutoff."),
        "gate_state": ("CLOSED — no primary-KNOWN stronger-than-Maverick model certifiable; frontier "
                       "target stays meta/llama-3.3-70b-instruct (KNOWN ~Dec-2023, the W105 model)"),
        "w135_note": ("W135 frontier was NOT earned (the §7a held-out dev gate FAILED: the "
                      "solution-structure witness S4 tied the EW1 counterexample C1 and blind "
                      "reflexion B0 at +0.00pp), so $0 frontier NIM was spent regardless of the gate."),
        "w123_through_w134_caps_closed": True,
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
