#!/usr/bin/env python3
"""W120 — official-ICPC multi-surface battlefield construction + admission + certification
(NIM-FREE; the count-gap-closure + pilot-readiness artifact).

Runs the deterministic ``run_battlefield_construction_v1`` over the SHA-pinned combined
RMRC + ECNA listing, re-verifies the live snapshot SHA, prints the count-gap closure +
per-model certification, and writes the verdict JSON.  This is the machine-checkable
proof of whether the official ICPC resistant battlefield reaches >= 30 pure pass-fail
tasks (it does: 45) and whether a model is certifiable (Maverick is) ⇒ the pilot is
EARNED.  No model inference; safe to run anywhere.

Usage::

    python scripts/run_w120_icpc_battlefield_v1.py
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

from coordpy.coordpy_icpc_battlefield_v1 import (  # noqa: E402
    W120_RAW_CLASSIFICATION_SHA256,
    classify_battlefield_listing_v1,
    core_slice_cid_v1,
    run_battlefield_construction_v1,
    select_battlefield_core_slice_v1,
)


def _verify_snapshot_sha(snapshot_path: Path) -> tuple[bool, str]:
    """Recompute the raw-classification SHA over the live snapshot's problems and
    compare to the pinned constant (R6 reproducibility)."""
    if not snapshot_path.exists():
        return False, "(snapshot file absent)"
    snap = json.loads(snapshot_path.read_text())
    canon = json.dumps(snap["problems"], sort_keys=True,
                       separators=(",", ":")).encode()
    sha = hashlib.sha256(canon).hexdigest()
    return (sha == W120_RAW_CLASSIFICATION_SHA256), sha


def main() -> int:
    ap = argparse.ArgumentParser(description="W120 ICPC battlefield construction")
    ap.add_argument(
        "--out-dir",
        default=str(ROOT / "results" / "w120" / "icpc_battlefield"))
    ap.add_argument("--n-slice", type=int, default=30)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ok_sha, live_sha = _verify_snapshot_sha(out_dir / "battlefield_snapshot.json")
    print(f"  snapshot SHA pinned-match: {ok_sha} (live={live_sha[:16]}…)")

    verified_on = _dt.date.today().isoformat()
    res = run_battlefield_construction_v1(verified_on=verified_on)
    m = res.manifest
    a = res.admissibility

    print(f"  surfaces            : {'+'.join(m.surfaces)}")
    print(f"  n_seen / n_admitted : {m.n_seen} / {m.n_admitted}")
    print(f"  core pure pass-fail : {m.n_core_passfail}  "
          f"(>= 30: {m.n_core_passfail >= 30})  [+float {m.n_float} +custom "
          f"{m.n_custom_validator}]")
    print(f"  dates               : {m.date_min} .. {m.date_max}")
    print(f"  month histogram     : {m.month_histogram}")
    print(f"  manifest_cid        : {m.manifest_cid()}")
    print(f"  core_slice_cid      : {m.core_slice_cid()}")
    print(f"  grader self-test    : RMRC {res.grader_selftest['rmrc']['n_cases_passed']}"
          f"/{res.grader_selftest['rmrc']['n_cases_run']} + ECNA "
          f"{res.grader_selftest['ecna']['n_cases_passed']}"
          f"/{res.grader_selftest['ecna']['n_cases_run']} "
          f"=> each-surface={res.grader_selftest['grader_proven_executable_each_surface']}")
    print(f"  admissibility       : identity={a.identity_admissible} "
          f"grader={a.grader_admissible} PILOT={a.pilot_admissible}")
    print(f"  exclusion audit     : {res.exclusion_audit.by_exclusion_reason}")
    print(f"  verdict             : {res.verdict}  pilot_earned={res.pilot_earned}  "
          f"n_certifiable={res.n_identity_certifiable_models}")
    print(f"  lcb_inherited cid   : "
          f"{res.to_dict()['lcb_inherited_decision_cid'][:16]}… (258b6ed7 invariant)")
    for pm in res.per_model:
        print(f"    - {pm.model_id:42s} certifiable={pm.identity_certifiable} "
              f"pilot={pm.pilot_admissible}")

    # the deterministic core pilot slice (what the NIM pilot will run on)
    probs = classify_battlefield_listing_v1()
    sl = select_battlefield_core_slice_v1(probs, n_problems=int(args.n_slice))
    print(f"  pilot {args.n_slice}-slice    : {len(sl)} problems  "
          f"cid={core_slice_cid_v1(sl)[:16]}…")

    payload = {
        "schema": "coordpy.w120_icpc_battlefield.v1",
        "milestone": "W120-alpha-beta-gamma",
        "verified_on": verified_on,
        "snapshot_sha_pinned_match": ok_sha,
        "result": res.to_dict(),
        "result_cid": res.cid(),
        "pilot_slice_n": int(args.n_slice),
        "pilot_slice_cid": core_slice_cid_v1(sl),
        "pilot_slice_problem_ids": [p.problem_id for p in sl],
        "ts_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    out_path = out_dir / "battlefield_verdict.json"
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"  wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
