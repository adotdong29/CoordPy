"""W125 Lane α — controller-native mechanism: structural slate + NIM-free contract checks.

Self-contained ($0, no packages, no NIM). Evaluates the C0/C1/C2/C3 structural
fake-different test (RUNBOOK § 4) and the four NIM-free contract checks (RUNBOOK § 5)
on the lead controller (C3), using a deterministic synthetic stdin/stdout problem so the
mechanism's structural properties are exercised without the resistant package load.

Emits results/w125/lane_alpha/{slate_verdict.json, contract_verdict.json}.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import coordpy.controller_native_code_mechanism_v1 as M  # noqa: E402

OUT_DIR = os.path.join(ROOT, "results", "w125", "lane_alpha")


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    now = _dt.datetime.now(_dt.timezone.utc).isoformat()

    slate = M.evaluate_mechanism_slate()
    slate_out = {
        "schema": "coordpy.w125_lane_alpha_slate.v1", "lane": "alpha_structural",
        "verified_on": "2026-05-31", "generated_at": now,
        "fake_different_test_bites": ("reflexion_B" in slate.fake_candidates
                                      and "C0_reflexion_relabeled" in slate.fake_candidates),
        **slate.to_dict(),
        "c1_cid": M.C1RolePlanControllerV1().cid(),
        "c2_cid": M.C2RouterSelectControllerV1().cid(),
        "c3_cid": M.C3DigestRoutedRepairControllerV1().cid(),
    }
    with open(os.path.join(OUT_DIR, "slate_verdict.json"), "w") as f:
        json.dump(slate_out, f, indent=2, default=str)

    prob, pool = M.synthetic_contract_problem()
    cc = M.run_contract_checks(pool, prob, K=5)
    contract_out = {
        "schema": "coordpy.w125_lane_alpha_contract.v1", "lane": "alpha_contract",
        "verified_on": "2026-05-31", "generated_at": now,
        "synthetic_problem_id": prob.problem_id, "K": 5, "nim_spend": 0,
        **cc.to_dict(),
    }
    with open(os.path.join(OUT_DIR, "contract_verdict.json"), "w") as f:
        json.dump(contract_out, f, indent=2, default=str)

    print(json.dumps({
        "lead": slate.lead, "real": list(slate.real_candidates),
        "fake": list(slate.fake_candidates),
        "fake_different_test_bites": slate_out["fake_different_test_bites"],
        "contract_all_pass": cc.all_pass, "contract": cc.to_dict(),
    }, indent=2, default=str))
    print(f"  wrote {OUT_DIR}/slate_verdict.json + contract_verdict.json")
    return 0 if (cc.all_pass and slate.lead) else 1


if __name__ == "__main__":
    raise SystemExit(main())
