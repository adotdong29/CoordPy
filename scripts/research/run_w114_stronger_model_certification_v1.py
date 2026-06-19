#!/usr/bin/env python3
"""W114-β/γ — per-model post-cutoff certification + instrument-frontier gate.

NIM-free.  Emits the LOCKED W114 certification decision
(``coordpy.stronger_model_cutoff_certification_v1``): for the LATEST REAL
LiveCodeBench release and the OFFICIAL (primary-source-verified) model cutoffs,
is ANY reachable stronger-than-Maverick model CERTIFIABLY contamination-resistant
on the available instrument?  If yes, names the target (the earned pilot); if no,
records the dated blocker + the exact next-instrument requirement (W115).

It ALSO re-verifies the pinned functional month histogram against the live
SHA-pinned ``release_v6`` corpus when present (a preflight-style discharge that
the instrument frontier the decision rests on matches real bytes), and asserts
the W113 registry confidence == the W114-verified confidence for every model
(the consistency guard).  $0 NIM.

Usage::

    python scripts/run_w114_stronger_model_certification_v1.py
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from coordpy.stronger_model_cutoff_certification_v1 import (  # noqa: E402
    LATEST_RESISTANT_INSTRUMENT,
    VERDICT_NONE,
    decide_certification_v1,
)
from coordpy.livecodebench_resistant_slice_v1 import (  # noqa: E402
    normalize_contest_date_v1,
)

OUT = ROOT / "results" / "w114" / "certification"


def _live_corpus_path() -> str:
    return os.environ.get(
        "COORDPY_LIVECODEBENCH_CACHE",
        os.path.expanduser("~/.cache/coordpy/livecodebench-test6.jsonl"))


def _reverify_histogram_against_corpus() -> dict:
    """Re-derive the functional month histogram from the live SHA-pinned corpus.

    Returns a dict with ``available`` / ``sha_ok`` / ``histogram_match`` /
    details.  Does NOT raise on a missing corpus (the pinned histogram is the
    authority; this is a best-effort discharge that real bytes agree)."""
    path = _live_corpus_path()
    inst = LATEST_RESISTANT_INSTRUMENT
    if not (os.path.exists(path) and os.path.getsize(path) > 0):
        return {"available": False, "path": path,
                "note": "corpus absent; pinned histogram used as authority"}
    raw = open(path, "rb").read()
    sha = hashlib.sha256(raw).hexdigest()
    sha_ok = (sha.lower() == inst.jsonl_sha256.lower())
    hist: dict[str, int] = {}
    n_func = 0
    for line in raw.decode("utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        if not str(row.get("starter_code") or "").strip():
            continue  # stdin/stdout = non-functional
        n_func += 1
        day = normalize_contest_date_v1(row.get("contest_date"))
        if day:
            hist[day[:7]] = hist.get(day[:7], 0) + 1
    match = (hist == dict(inst.functional_month_histogram))
    return {"available": True, "path": path, "sha_ok": sha_ok,
            "n_functional": n_func, "histogram": dict(sorted(hist.items())),
            "histogram_match": match}


def main() -> int:
    decision = decide_certification_v1()
    corpus_check = _reverify_histogram_against_corpus()

    # Consistency guard: W113 registry confidence == W114-verified confidence.
    inconsistent = [m.model_id for m in decision.per_model
                    if not m.confidence_consistent]

    artifact = {
        "schema": "coordpy.w114_certification.v1",
        "milestone": "W114-beta-gamma",
        "decision": decision.to_dict(),
        "decision_cid": decision.cid(),
        "corpus_reverification": corpus_check,
        "confidence_consistency_ok": (len(inconsistent) == 0),
        "confidence_inconsistent_models": inconsistent,
        "ts_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "certification_verdict.json").write_text(
        json.dumps(artifact, indent=2, default=str))

    inst = decision.instrument
    print("=== W114 stronger-model post-cutoff certification ===")
    print(f"  latest resistant FUNCTIONAL instrument: {inst['release']} "
          f"(functional {inst['functional_date_min']}.."
          f"{inst['functional_date_max']}; n={inst['n_functional']})")
    print(f"  MIN_RESISTANT_SLICE = {decision.min_resistant_slice}")
    print("  per-model certification (C1 known / C2 >=30 / C3 reach-stronger / "
          "C4 not-settled):")
    for m in decision.per_model:
        print(f"    [{m.rank_tier}.{m.rank_within_tier}] {m.model_id:42s} "
              f"cutoff={m.cutoff_boundary}[{m.cutoff_confidence}] "
              f"n_res={m.n_functional_resistant:2d} "
              f"C1={int(m.c1_cutoff_known)} C2={int(m.c2_enough_resistant)} "
              f"C3={int(m.c3_reachable_stronger_comparable)} "
              f"C4={int(m.c4_not_already_settled)} "
              f"=> certifiable={m.certifiable_for_new_pilot}")
    print(f"  VERDICT: {decision.verdict}"
          + (f" (target={decision.target_model})" if decision.target_model
             else ""))
    print(f"  maverick_certifiable_but_settled: "
          f"{decision.maverick_certifiable_but_settled}")
    print(f"  decision CID: {decision.cid()}")
    print(f"  corpus re-verification: available={corpus_check.get('available')}"
          f" sha_ok={corpus_check.get('sha_ok')} "
          f"histogram_match={corpus_check.get('histogram_match')}")
    print(f"  W113<->W114 confidence consistency OK: {len(inconsistent) == 0}")
    if decision.verdict == VERDICT_NONE:
        print("  W115 BLOCKER:")
        print(f"    {decision.w115_blocker}")
        print("  NEXT-INSTRUMENT REQUIREMENT:")
        print(f"    {decision.next_instrument_requirement}")
    print(f"  artifact: {OUT / 'certification_verdict.json'}")

    # Exit non-zero ONLY on an integrity failure (corpus present but SHA/hist
    # mismatch, or a W113<->W114 confidence divergence) — NOT on a no-go verdict
    # (a no-go is a valid, expected outcome).
    integrity_ok = (len(inconsistent) == 0) and (
        (not corpus_check.get("available"))
        or (corpus_check.get("sha_ok") and corpus_check.get("histogram_match")))
    return 0 if integrity_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
