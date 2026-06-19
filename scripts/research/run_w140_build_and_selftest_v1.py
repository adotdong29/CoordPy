#!/usr/bin/env python3
"""W140 Lane α — build + $0 self-test + leak audit + shared-family report (RUNBOOK_W140 §4/§7).

Reuses the W139 per-tier band calibration ($0 re-spend) to SELECT the shared families (in-band on the
anchor AND the weak 8B tier), compiles every TC-kind tutor for each shared family, runs the
deterministic ``tutor_leak_gate_v1`` (asserting no legit leak + planted leaks BITE), proves each holed
skeleton is completable to a correct program, and records the tutor artifacts + CIDs.  NO NIM.

Run:  .venv/bin/python scripts/run_w140_build_and_selftest_v1.py
        [--calibration results/w139/w139_per_tier_calibration_v1.json]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from coordpy.headroom_band_slate_v3 import (  # noqa: E402
    CX_FACTORIES, EXTRA_CX_FACTORIES, FUNC_FACTORIES)
from coordpy.resistant_by_construction_battlefield_v1 import mint_problem_v1  # noqa: E402
from coordpy import family_tutor_compiler_v1 as T  # noqa: E402
from coordpy import cross_tier_tutor_bench_v1 as B  # noqa: E402

BAND_LO, BAND_HI, A0_CEIL = 0.15, 0.85, 0.80
TRAIN_SEED_BASE = 140_100_000


def _factory_for(fam):
    return CX_FACTORIES.get(fam) or FUNC_FACTORIES.get(fam) or EXTRA_CX_FACTORIES.get(fam)


def _shared_families(cal: dict, anchor_tier: str, weak_tier: str) -> list[dict]:
    """A family is SHARED iff it has an in-band per-tier cell on BOTH the anchor and the weak tier
    (RUNBOOK_W140 §4); knobs may differ per tier."""
    bft = cal.get("band_for_tier", {})
    anchor = bft.get(anchor_tier, {})
    weak = bft.get(weak_tier, {})
    out = []
    for fam in sorted(set(anchor) & set(weak)):
        out.append({"family": fam, "mode": anchor[fam].get("mode", ""),
                    "anchor_knob": anchor[fam]["knob_value"], "anchor_a1": anchor[fam].get("a1_rate"),
                    "weak_knob": weak[fam]["knob_value"], "weak_a1": weak[fam].get("a1_rate")})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibration", default="results/w139/w139_per_tier_calibration_v1.json")
    ap.add_argument("--out", default="results/w140/w140_build_selftest_v1.json")
    args = ap.parse_args()

    with open(os.path.join(ROOT, args.calibration)) as f:
        cal = json.load(f)
    anchor_tier = cal.get("anchor_tier", "strong")
    weak_tier = "small"
    shared = _shared_families(cal, anchor_tier, weak_tier)
    print(f"reused per_tier_calibration_cid: {cal.get('per_tier_calibration_cid')}")
    print(f"anchor_tier={anchor_tier} weak_tier={weak_tier}")
    print(f"SHARED families (in-band on anchor AND weak): {[s['family'] for s in shared]}")
    for s in shared:
        print(f"  {s['family']:28s} mode={s['mode']:18s} "
              f"anchor@{s['anchor_knob']}(a1={s['anchor_a1']}) weak@{s['weak_knob']}(a1={s['weak_a1']})")

    checks = {"all_pass": True}
    fam_records = []
    for s in shared:
        fam = s["family"]
        fac = _factory_for(fam)
        if fac is None:
            checks["all_pass"] = False
            fam_records.append({"family": fam, "error": "no factory"})
            continue
        # compile at the ANCHOR knob (the tutor is family-level; knob only affects the instance)
        tmpl = fac(int(s["anchor_knob"]))
        prob = mint_problem_v1(tmpl.minted, global_seed=TRAIN_SEED_BASE, timeout_s=8.0)
        rec = {"family": fam, "algo_sig": tmpl.minted.algo_sig, "admitted": bool(prob.gates.admitted),
               "headroom_note": tmpl.headroom_note, "has_technique_spec": tmpl.minted.algo_sig in
               T.TECHNIQUE_LIBRARY, "tutors": {}, "planted_leaks_bite": {}, "skeleton_completable": None}
        # all TC kinds compile + pass the leak gate
        for kind in (T.TC1_CARD, T.TC2_REWRITE, T.TC3_COMPRESSED, T.TC5_STAGED, T.T6_NEG):
            tut = T.COMPILERS_BY_KIND[kind](tmpl)
            lr = T.tutor_leak_gate_v1(tut, tmpl, prob, timeout_s=8.0)
            rec["tutors"][kind] = {"tutor_cid": tut.cid(), "tokens": tut.token_count(),
                                   "leaked": lr.leaked, "leak_report": lr.to_dict(),
                                   "genuine": T.tutor_is_genuinely_new_v1(tut)["genuinely_new"]}
            if kind != T.T6_NEG and lr.leaked:
                checks["all_pass"] = False
        # skeleton completable (correct-fill passes hidden secret cases)
        comp = T.skeleton_is_completable_v1(tmpl, prob, timeout_s=8.0)
        rec["skeleton_completable"] = comp
        if comp.get("completable") is not True:
            checks["all_pass"] = False
        # planted leaks MUST bite (falsifiability of the gate)
        leak_a = dataclasses.replace(T.compile_witness_rewrite_tutor_v1(tmpl),
                                     skeleton=tmpl.minted.ref_source)
        spec = T.TECHNIQUE_LIBRARY.get(tmpl.minted.algo_sig)
        disc = max(spec.correct_fill.values(), key=len) if spec else ""
        leak_b = dataclasses.replace(T.compile_family_card_v1(tmpl),
                                     key_move=f"the exact rule is: {disc}") if disc else None
        bite_a = T.tutor_leak_gate_v1(leak_a, tmpl, prob).leaked
        bite_b = T.tutor_leak_gate_v1(leak_b, tmpl, prob).leaked if leak_b else None
        rec["planted_leaks_bite"] = {"skeleton_eq_ref": bite_a, "discriminator_in_card": bite_b}
        if not bite_a or (leak_b is not None and not bite_b):
            checks["all_pass"] = False
        fam_records.append(rec)
        print(f"  [{fam}] tutors_clean={all(not rec['tutors'][k]['leaked'] for k in rec['tutors'] if k!=T.T6_NEG)} "
              f"completable={comp.get('completable')} planted_bite={bite_a}/{bite_b}")

    checks["n_shared_families"] = len(shared)
    checks["shared_family_threshold_met"] = bool(len(shared) >= 2)
    if len(shared) < 2:
        checks["all_pass"] = False

    report = {"schema": "w140_build_selftest_v1",
              "reused_per_tier_calibration_cid": cal.get("per_tier_calibration_cid"),
              "reused_slate_fingerprint_cid": cal.get("slate_fingerprint_cid"),
              "anchor_tier": anchor_tier, "weak_tier": weak_tier,
              "shared_families": shared, "family_records": fam_records, "checks": checks}
    out_path = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n=== W140 BUILD+SELFTEST: all_pass={checks['all_pass']} "
          f"n_shared={len(shared)} (>=2: {checks['shared_family_threshold_met']}) ===")
    print(f"-> {out_path}")
    return 0 if checks["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
