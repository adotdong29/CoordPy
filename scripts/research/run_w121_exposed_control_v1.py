#!/usr/bin/env python3
"""W121 — matched EXPOSED official-ICPC control construction + dual-field comparison
(NIM-FREE; the exposed-pilot-readiness + same-family-contrast artifact).

Runs the deterministic ``run_exposed_control_construction_v1`` over the SHA-pinned
exposed listing (the immediately-preceding PRE-cutoff year-drops of the SAME official
ICPC org surface families W120 used), re-verifies the live snapshot SHA, prints the
exposed count + per-model EXPOSED certification + the matched-family comparison vs the
W120 resistant battlefield, and writes the verdict JSON.  Machine-checkable proof that:
the exposed control reaches >= 30 tier-1 pure pass-fail (44), Maverick is
EXPOSED-certifiable, and the exposed + resistant battlefields are the same official
family differing only in cutoff side ⇒ the exposed-control pilot is EARNED.  No model
inference; safe to run anywhere.

Usage::

    python scripts/run_w121_exposed_control_v1.py
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import sys  # noqa: E402
sys.path.insert(0, str(ROOT))

from coordpy.coordpy_icpc_exposed_control_v1 import (  # noqa: E402
    W121_EXPOSED_RAW_CLASSIFICATION_SHA256,
    classify_exposed_listing_v1,
    run_exposed_control_construction_v1,
    select_exposed_core_slice_v1,
)
from coordpy.coordpy_icpc_battlefield_v1 import core_slice_cid_v1  # noqa: E402


def _verify_snapshot_sha(snapshot_path: Path) -> tuple[bool, str]:
    if not snapshot_path.exists():
        return False, "(snapshot file absent)"
    snap = json.loads(snapshot_path.read_text())
    canon = json.dumps(snap["listing"], sort_keys=True,
                       separators=(",", ":")).encode()
    sha = hashlib.sha256(canon).hexdigest()
    return (sha == W121_EXPOSED_RAW_CLASSIFICATION_SHA256), sha


def main() -> int:
    ap = argparse.ArgumentParser(description="W121 exposed-control construction")
    ap.add_argument(
        "--out-dir",
        default=str(ROOT / "results" / "w121" / "exposed_control"))
    ap.add_argument("--n-slice", type=int, default=30)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ok_sha, live_sha = _verify_snapshot_sha(out_dir / "exposed_listing_live.json")
    print(f"  snapshot SHA pinned-match: {ok_sha} (live={live_sha[:16]}…)")

    verified_on = _dt.date.today().isoformat()
    res = run_exposed_control_construction_v1(verified_on=verified_on)
    m = res.manifest
    a = res.admissibility
    mf = res.matched_family

    print(f"  EXPOSED surfaces    : {'+'.join(m.surfaces)}  (pre-cutoff year-drops)")
    print(f"  n_seen / n_admitted : {m.n_seen} / {m.n_admitted}")
    print(f"  core pure pass-fail : {m.n_core_passfail}  (>= 30: "
          f"{m.n_core_passfail >= 30})  [+float {m.n_float} +custom "
          f"{m.n_custom_validator}]")
    print(f"  dates (EXPOSED)     : {m.date_min} .. {m.date_max}  "
          f"(all <= {m.boundary})")
    print(f"  month histogram     : {m.month_histogram}")
    print(f"  manifest_cid        : {m.manifest_cid()}")
    st = res.grader_selftest
    print(f"  grader self-test    : {st['n_problems_self_tested']} all-pass problems / "
          f"{st['n_cases_passed']}/{st['n_cases_run']} cases; each-surface="
          f"{st['grader_proven_executable_each_surface']}")
    for sk, sv in st["per_surface"].items():
        print(f"      - {sk:18s}: {sv['n_problems']} problems {sv['n_cases_passed']}"
              f"/{sv['n_cases_run']} all_pass={sv['all_pass']}")
    print(f"  admissibility       : identity={a.identity_admissible} "
          f"grader={a.grader_admissible} EXPOSED_PILOT={a.pilot_admissible}")
    print(f"  exclusion audit     : {res.exclusion_audit.by_exclusion_reason}")
    print(f"  verdict             : {res.verdict}  exposed_pilot_earned="
          f"{res.exposed_pilot_earned}  n_certifiable={res.n_exposed_certifiable_models}")
    print(f"  lcb_inherited cid   : "
          f"{res.to_dict()['lcb_inherited_decision_cid'][:16]}… (258b6ed7 invariant)")
    for pm in res.per_model:
        print(f"    - {pm.model_id:42s} exposed_cert={pm.exposed_certifiable} "
              f"pilot={pm.pilot_admissible} (n_exposed={pm.n_exposed_before})")

    print("  ---- matched-family comparison (EXPOSED vs W120 RESISTANT) ----")
    print(f"    shared families   : {'+'.join(mf.shared_surface_families)}  "
          f"same_family={mf.differs_only_in_cutoff_side}")
    print(f"    EXPOSED   core={mf.exposed_n_core}  {mf.exposed_date_min}.."
          f"{mf.exposed_date_max}  cid={mf.exposed_core_slice_cid[:16]}…")
    print(f"    RESISTANT core={mf.resistant_n_core}  {mf.resistant_date_min}.."
          f"{mf.resistant_date_max}  cid={mf.resistant_core_slice_cid[:16]}…")
    print(f"    same org/format/grader/tiers/difficulty/model = "
          f"{mf.same_org and mf.same_package_format_family and mf.same_grader_and_oracle and mf.same_tier_discipline and mf.same_difficulty_class and mf.same_model_and_evaluator_line}")

    probs = classify_exposed_listing_v1()
    sl = select_exposed_core_slice_v1(probs, n_problems=int(args.n_slice))
    print(f"  exposed {args.n_slice}-slice : {len(sl)} problems  "
          f"cid={core_slice_cid_v1(sl)[:16]}…")

    payload = {
        "schema": "coordpy.w121_exposed_control.v1",
        "milestone": "W121-alpha-beta-gamma",
        "verified_on": verified_on,
        "snapshot_sha_pinned_match": ok_sha,
        "result": res.to_dict(),
        "result_cid": res.cid(),
        "pilot_slice_n": int(args.n_slice),
        "pilot_slice_cid": core_slice_cid_v1(sl),
        "pilot_slice_problem_ids": [p.problem_id for p in sl],
        "ts_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    out_path = out_dir / "exposed_control_verdict.json"
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"  wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
