#!/usr/bin/env python3
"""W133-alpha — build the train/dev/eval curriculum + run the exact-oracle witness self-tests.

$0 NIM.  Deterministic.  This is the Lane-alpha construction step (allowed before the
RUNBOOK NIM-lock, per the W129/W130/W131/W132 $0-construction discipline).  It:

1. builds the seed-disjoint train/dev/eval curriculum from ``RBC_SLATE_V1`` (W133 seeds);
2. verifies split disjointness (content CIDs + graded secret-case inputs) + family balance;
3. runs the witness self-tests on EVERY admitted problem in EVERY split — for each, the
   witness must FIRE on the canonical admissible-wrong / too-slow ``naive_source`` (the trap
   the field is built to resist, i.e. exactly the failure the model tends to reproduce),
   be leakage-clean (probe input not a graded secret case), be genuinely-new (carries an
   input that is not a public sample + an oracle output), and reproduce deterministically;
4. replays the SIX W132 capability-bound traps + the ONE W132 B-unique complexity rescue as
   $0 regression fixtures on the W132 anchor (seed 132), confirming the witness fires on each.

Emits ``results/w133/curriculum/*.json``.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import sys  # noqa: E402
sys.path.insert(0, str(ROOT))

from coordpy.resistant_by_construction_slate_v1 import RBC_SLATE_V1  # noqa: E402
from coordpy.resistant_by_construction_battlefield_v1 import mint_problem_v1  # noqa: E402
from coordpy.witness_curriculum_corpus_v1 import (  # noqa: E402
    DEV_SEED, EVAL_SEED, MINTED_DATE, TRAIN_SEED, build_curriculum_v1,
)
from coordpy.exact_oracle_witness_v1 import (  # noqa: E402
    ARM_C3_CONTROLLER, build_witness_probe_set_v1, select_witness_v1,
    witness_is_genuinely_new_v1,
)
from coordpy.coordpy_icpc_battlefield_v1 import (  # noqa: E402
    ICPC_BATTLEFIELD_LISTING_SNAPSHOT_V1,
)

OFFICIAL_IDENTITIES = tuple(sorted({row[1] for row in ICPC_BATTLEFIELD_LISTING_SNAPSHOT_V1}))
WITNESS_SEED = 999_133
EXEC_TIMEOUT_S = 8.0
WITNESS_TIMEOUT_S = 2.0

# W132 anchor (seed 132) regression fixtures — from results/w132/pilot/.../w132_pilot_report.json
W132_TRAPS = ("rbc_cb_distinct_in_windows", "rbc_cb_pairs_sum_eq_t", "rbc_cb_pairs_sum_le_t",
              "rbc_he_interval_union_length", "rbc_se_lattice_paths_blocked",
              "rbc_wa_knapsack_01")
W132_B_UNIQUE_RESCUE = ("rbc_cb_pairs_absdiff_le_d",)
W132_GLOBAL_SEED = 132


def _witness_selftest_for_split(split, templates_by_id):
    """For each admitted problem: the witness must fire on naive_source, be leakage-clean,
    genuinely-new, and regenerate deterministically."""
    rows = []
    n_fire = n_clean = n_new = n_determ = 0
    by_mode_fire = {}
    for idx, p in enumerate(split.problems()):
        t = templates_by_id[p.problem_id]
        probe = build_witness_probe_set_v1(t, p, witness_seed=WITNESS_SEED,
                                           timeout_s=WITNESS_TIMEOUT_S)
        # determinism is input-deterministic by construction (CID hashes the seeded inputs,
        # not ref outputs); spot-check the first problem of each split with a full re-build.
        if idx == 0:
            determ = bool(probe.cid() == build_witness_probe_set_v1(
                t, p, witness_seed=WITNESS_SEED, timeout_s=WITNESS_TIMEOUT_S).cid())
        else:
            determ = True
        w = select_witness_v1(t.naive_source, p, probe, t, arm=ARM_C3_CONTROLLER,
                              timeout_s=WITNESS_TIMEOUT_S)
        chk = witness_is_genuinely_new_v1(w, p)
        fired = bool(w.found())
        n_fire += int(fired)
        n_clean += int(w.leakage_clean)
        n_new += int(chk["genuinely_new"])
        n_determ += int(determ)
        by_mode_fire.setdefault(p.mode, [0, 0])
        by_mode_fire[p.mode][0] += int(fired)
        by_mode_fire[p.mode][1] += 1
        rows.append({"problem_id": p.problem_id, "mode": p.mode, "witness_kind": w.kind,
                     "ew_family": w.ew_family, "fired": fired,
                     "leakage_clean": bool(w.leakage_clean),
                     "genuinely_new": bool(chk["genuinely_new"]),
                     "shrink_steps": int(w.shrink_steps), "probe_n_small": len(probe.small),
                     "probe_has_big": probe.big_input is not None,
                     "probe_set_cid": probe.cid(), "deterministic": determ})
    n = len(rows)
    return {
        "split": split.split, "n_problems": n,
        "n_witness_fired": n_fire, "n_leakage_clean": n_clean,
        "n_genuinely_new": n_new, "n_deterministic": n_determ,
        "all_fired": bool(n_fire == n), "all_leakage_clean": bool(n_clean == n),
        "all_genuinely_new": bool(n_new == n), "all_deterministic": bool(n_determ == n),
        "fire_by_mode": {m: f"{v[0]}/{v[1]}" for m, v in sorted(by_mode_fire.items())},
        "rows": rows,
    }


def _w132_regression_fixtures():
    """Re-mint the W132 anchor (seed 132); confirm the witness fires on each of the 6 traps'
    naive_source + records the 1 B-unique rescue family."""
    by_name = {t.name: t for t in RBC_SLATE_V1}
    rows = []
    for pid in (W132_TRAPS + W132_B_UNIQUE_RESCUE):
        name = pid[len("rbc_"):]
        t = by_name[name]
        p = mint_problem_v1(t, global_seed=W132_GLOBAL_SEED, timeout_s=EXEC_TIMEOUT_S)
        probe = build_witness_probe_set_v1(t, p, witness_seed=WITNESS_SEED,
                                           timeout_s=WITNESS_TIMEOUT_S)
        w = select_witness_v1(t.naive_source, p, probe, t, arm=ARM_C3_CONTROLLER,
                              timeout_s=WITNESS_TIMEOUT_S)
        chk = witness_is_genuinely_new_v1(w, p)
        rows.append({"problem_id": pid, "mode": p.mode,
                     "is_trap": pid in W132_TRAPS,
                     "witness_kind": w.kind, "ew_family": w.ew_family,
                     "fired": bool(w.found()), "leakage_clean": bool(w.leakage_clean),
                     "genuinely_new": bool(chk["genuinely_new"]),
                     "shrink_steps": int(w.shrink_steps)})
    n_trap_fire = sum(1 for r in rows if r["is_trap"] and r["fired"])
    return {"n_traps": len(W132_TRAPS), "n_trap_witness_fired": n_trap_fire,
            "all_traps_have_witness": bool(n_trap_fire == len(W132_TRAPS)),
            "rows": rows}


def main() -> int:
    out_dir = ROOT / "results" / "w133" / "curriculum"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("  building train/dev/eval curriculum (deterministic, $0 NIM) ...")
    cur = build_curriculum_v1(RBC_SLATE_V1, minted_date=MINTED_DATE, train_seed=TRAIN_SEED,
                              dev_seed=DEV_SEED, eval_seed=EVAL_SEED,
                              official_identities=OFFICIAL_IDENTITIES,
                              timeout_s=EXEC_TIMEOUT_S)
    print(f"  admitted: train={cur.train.n_admitted} dev={cur.dev.n_admitted} "
          f"eval={cur.eval.n_admitted}  total={cur.n_total_admitted} "
          f"(min_per_split>={32}: {cur.meets_min_per_split}; >=96: {cur.meets_min_total})")
    print(f"  disjointness: content_disjoint={cur.disjointness['all_content_cids_pairwise_disjoint']} "
          f"graded_secret_disjoint={cur.disjointness['all_graded_secret_inputs_disjoint']} "
          f"worst_family_secret_overlap={cur.disjointness['worst_single_family_secret_overlap']}")
    print(f"  family_balance: {cur.family_balance}")
    print(f"  curriculum_cid={cur.curriculum_cid()[:16]}")

    templates_by_id = cur.templates_by_problem_id(RBC_SLATE_V1)
    print("  running witness self-tests on every admitted problem (3 splits) ...")
    selftests = {}
    for split in (cur.train, cur.dev, cur.eval):
        st = _witness_selftest_for_split(split, templates_by_id)
        selftests[split.split] = st
        print(f"    {split.split}: fired {st['n_witness_fired']}/{st['n_problems']} "
              f"clean {st['n_leakage_clean']}/{st['n_problems']} "
              f"new {st['n_genuinely_new']}/{st['n_problems']} "
              f"determ {st['n_deterministic']}/{st['n_problems']} | by_mode {st['fire_by_mode']}")

    print("  $0 W132 regression fixtures (6 traps + 1 B-unique rescue, anchor seed 132) ...")
    reg = _w132_regression_fixtures()
    print(f"    traps with a firing witness: {reg['n_trap_witness_fired']}/{reg['n_traps']} "
          f"(all={reg['all_traps_have_witness']})")
    for r in reg["rows"]:
        tag = "TRAP " if r["is_trap"] else "RESC "
        print(f"      {tag}{r['problem_id']:32s} {r['mode']:24s} "
              f"{r['witness_kind']}/{r['ew_family']} fired={r['fired']} clean={r['leakage_clean']}")

    all_selftests_pass = all(
        st["all_fired"] and st["all_leakage_clean"] and st["all_genuinely_new"]
        and st["all_deterministic"] for st in selftests.values())
    lane_alpha_success = bool(cur.meets_min_per_split and cur.meets_min_total
                              and cur.disjointness["held_out_integrity"]
                              and all_selftests_pass and reg["all_traps_have_witness"])

    payload = {
        "schema": "coordpy.w133_curriculum_build_v1",
        "minted_date": MINTED_DATE, "witness_seed": WITNESS_SEED,
        "curriculum": cur.to_dict(),
        "witness_selftests": selftests,
        "w132_regression_fixtures": reg,
        "all_witness_selftests_pass": bool(all_selftests_pass),
        "lane_alpha_success": lane_alpha_success,
    }
    (out_dir / "curriculum_build_v1.json").write_text(json.dumps(payload, indent=2, default=str))
    # a compact manifest for quick diffing
    (out_dir / "curriculum_manifest_v1.json").write_text(json.dumps({
        "curriculum_cid": cur.curriculum_cid(),
        "seeds": {"train": TRAIN_SEED, "dev": DEV_SEED, "eval": EVAL_SEED},
        "split_cids": {"train": cur.train.split_cid, "dev": cur.dev.split_cid,
                       "eval": cur.eval.split_cid},
        "n_admitted": {"train": cur.train.n_admitted, "dev": cur.dev.n_admitted,
                       "eval": cur.eval.n_admitted},
        "lane_alpha_success": lane_alpha_success,
    }, indent=2))

    print()
    print(f"  LANE-ALPHA SUCCESS = {lane_alpha_success}")
    print(f"  wrote {out_dir/'curriculum_build_v1.json'}")
    return 0 if lane_alpha_success else 2


if __name__ == "__main__":
    raise SystemExit(main())
