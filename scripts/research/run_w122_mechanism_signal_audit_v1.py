#!/usr/bin/env python3
"""W122 Lane beta (NIM-FREE) — ICPC M3 mechanism-signal audit.

Reads the W120 resistant + W121 exposed reflexion-call sidecars (the full prompt/response
record of every NIM call), classifies each FAILING reflexion turn by the public signal the
model actually saw, and applies the pre-committed earn rule
(``audit_icpc_mechanism_signal_v1``, RUNBOOK_W122 section 4) to decide whether the
executor-grounded patcher (M3) is honestly LIVE on official ICPC.

On official ICPC the hidden oracle is SECRET token-diff (it returns only "wrong answer on a
hidden case", never the expected value), so ``grader_reveals_hidden_expected=False`` and
M3's exclusive expected/actual signal is structurally ~0 => KILL the lane NIM-free.  $0 NIM.

Usage::

    python scripts/run_w122_mechanism_signal_audit_v1.py
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import sys  # noqa: E402
sys.path.insert(0, str(ROOT))

from coordpy.coordpy_icpc_paired_seed_closure_v1 import (  # noqa: E402
    audit_icpc_mechanism_signal_v1,
    classify_sidecar_turns_v1,
)

# The full (non-canary) W120 resistant + W121 exposed pilot sidecars.
RESISTANT_SIDECAR_GLOB = (
    "results/w120/icpc_pilot/w120_icpc_pilot_*_2026*/icpc_reflexion_calls.jsonl")
EXPOSED_SIDECAR_GLOB = (
    "results/w121/exposed_pilot/w121_exposed_pilot_*_2026*/exposed_reflexion_calls.jsonl")


def _load_full_sidecar(glob_pat: str) -> tuple[str, list[dict]]:
    """Pick the largest matching sidecar (the full 330-call run, not a 2-problem canary)."""
    paths = [p for p in glob.glob(str(ROOT / glob_pat)) if "canary" not in p]
    if not paths:
        raise SystemExit(f"no sidecar matched {glob_pat}")
    path = max(paths, key=lambda p: Path(p).stat().st_size)
    recs = [json.loads(ln) for ln in open(path)]
    return path, recs


def main() -> int:
    ap = argparse.ArgumentParser(description="W122 ICPC M3 mechanism-signal audit (NIM-free)")
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "w122" / "mechanism_audit"))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rpath, rrecs = _load_full_sidecar(RESISTANT_SIDECAR_GLOB)
    epath, erecs = _load_full_sidecar(EXPOSED_SIDECAR_GLOB)
    turns = classify_sidecar_turns_v1(rrecs) + classify_sidecar_turns_v1(erecs)

    # Official ICPC grading regime: SECRET token-diff oracle => does NOT reveal the hidden
    # expected (anti-cheat). This is the load-bearing regime fact.
    audit = audit_icpc_mechanism_signal_v1(
        turn_classes=turns, grader_reveals_hidden_expected=False)

    print(f"  resistant sidecar : {Path(rpath).name}  ({len(rrecs)} calls)")
    print(f"  exposed   sidecar : {Path(epath).name}  ({len(erecs)} calls)")
    print(f"  reflexion turns   : {audit.n_reflexion_turns}")
    print(f"  counts            : {audit.counts}")
    print(f"  public-sample-wrong frac : {audit.public_sample_wrong_fraction:.2%} "
          "(reflexion ALREADY feeds the expected here)")
    print(f"  hidden-only frac         : {audit.hidden_only_fraction:.2%} "
          "(hidden expected is SECRET => M3 gets nothing reflexion lacks)")
    print(f"  runtime-traceback frac   : {audit.runtime_traceback_fraction:.2%} "
          "(reflexion ALREADY shows the stderr tail)")
    print(f"  grader_reveals_hidden_expected : {audit.grader_reveals_hidden_expected}")
    print(f"  m3_exclusive_signal_fraction   : {audit.m3_exclusive_signal_fraction:.3f} "
          f"(floor {audit.m3_signal_floor:.2f})")
    print(f"  VERDICT : {audit.verdict}")
    print(f"  {audit.rationale}")

    payload = {
        "schema": "coordpy.w122_mechanism_audit.v1",
        "milestone": "W122-beta",
        "resistant_sidecar": str(Path(rpath).relative_to(ROOT)),
        "exposed_sidecar": str(Path(epath).relative_to(ROOT)),
        "audit": audit.to_dict(),
        "ts_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    out_path = out_dir / "mechanism_audit_verdict.json"
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"  wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
