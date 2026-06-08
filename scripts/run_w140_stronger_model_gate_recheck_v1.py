"""W140 Lane γ — stronger-model cutoff-disclosure gate re-check (NIM-free, read-only).

Re-affirms the invariant decision (CID ``258b6ed7…``, {KNOWN:1, UNKNOWN:4}) from the primary-source
registry in ``coordpy.stronger_model_cutoff_certification_v1``, re-verified from PRIMARY sources in
the W140 research lane (``docs/RESULTS_W140_RESEARCH_V1.md`` §D): Llama-4-Maverick = Aug-2024 KNOWN
(already settled by W113); Qwen3-Coder-480B / DeepSeek-V4-Pro / Mistral-Small-4-119B-2603 / GLM-5 each
OFFICIALLY UNDISCLOSED.  NEW W140 observation (recorded, NOT gate-opening): Gemma 4 now carries a
primary-disclosed Jan-2025 cutoff — but AT (not before) the ~Jan-2025 boundary and with
code-competence-vs-the-bench unestablished, so it is NOT a qualifying certifiable-stronger model.

For W140 the gate governs only WHICH model an EARNED frontier rerun would target — the tutor field is
resistant BY CONSTRUCTION (freshly minted, unmemorisable for any cutoff), so it does NOT itself
require a stronger model; the default frontier target stays ``meta/llama-3.3-70b-instruct``.
Emits results/w140/stronger_model_gate/gate_recheck_v1.json.
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
OUT_DIR = os.path.join(ROOT, "results", "w140", "stronger_model_gate")


def main() -> int:
    dec = decide_certification_v1()
    d = dec.to_dict() if hasattr(dec, "to_dict") else dec
    cid = dec.cid() if hasattr(dec, "cid") else None
    per = d.get("per_model", [])
    n_known = sum(1 for m in per if m.get("verified_confidence") == "KNOWN")
    n_unknown = sum(1 for m in per if m.get("verified_confidence") == "UNKNOWN")
    verdict = {
        "schema": "coordpy.w140_stronger_model_gate_recheck.v1",
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
        "w140_primary_recheck_note": (
            "Re-verified from PRIMARY official sources in the W140 research lane (invariant "
            "W114->W139): Llama-4-Maverick = Aug-2024 KNOWN (official Llama-4 card; settled W113); "
            "Qwen3-Coder-480B (Qwen HF/blog/arXiv 2505.09388) / DeepSeek-V4-Pro (official V4 card PDF) "
            "/ Mistral-Small-4-119B-2603 (Mistral docs/HF/news) / GLM-5 (zai-org HF + arXiv 2602.15763) "
            "all primary-UNDISCLOSED. NEW: Gemma 4 (ai.google.dev) discloses a Jan-2025 cutoff — AT, "
            "not before, the ~Jan-2025 resistant-instrument boundary, code-competence unestablished => "
            "NOT a qualifying certifiable-stronger model; the gate stays {KNOWN:1, UNKNOWN:4}."),
        "gate_state": ("CLOSED — no primary-KNOWN stronger-than-Maverick model certifiable; the W140 "
                       "tutor field is resistant BY CONSTRUCTION so it does not require a stronger "
                       "model; an EARNED frontier rerun targets meta/llama-3.3-70b-instruct (KNOWN "
                       "~Dec-2023)"),
        "w140_note": ("W140's weak-model lift is on freshly-minted needed-and-unknown-technique cells; "
                      "the gate governs only the frontier TARGET if a strict §7b anchor∧weak earn is "
                      "met — it does not by itself authorise NIM."),
        "w123_through_w139_caps_closed": True,
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
