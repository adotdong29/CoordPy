#!/usr/bin/env python3
"""W106 — margin-cap dispatch driver (NO NIM).

Reads the W105 consolidated Phase 3 retirement verdict
(``phase3_retirement_verdict.json``) and computes the W106
margin-cap dispatch GO / NO-GO decision via
``coordpy.margin_cap_dispatch_v1`` (explicit import).

Default models the realized W105 Verdict-C / sub-case C1 case:
the only confirmation form on offer is rescue-concentrated, and a
fair broad-slice multi-seed Phase 3 verdict already exists (this
verdict).  Both push GATE 2 to FAIL ⇒ NO-GO.

Usage::

    python scripts/run_w106_margin_cap_dispatch.py \
        [--verdict <phase3_retirement_verdict.json>] \
        [--slice-type rescue_concentrated|fair_broad] \
        [--fair-result-exists/--no-fair-result-exists] \
        [--out-dir results/w106/margin_cap_dispatch]

No NIM key required; no network; deterministic.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Explicit import only — NOT via coordpy/__init__.py.
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from coordpy.margin_cap_dispatch_v1 import (  # noqa: E402
    SLICE_FAIR_BROAD,
    SLICE_RESCUE_CONCENTRATED,
    build_margin_cap_dispatch_decision_v1,
    format_margin_cap_dispatch_markdown_v1,
)

_DEFAULT_VERDICT = (
    "results/w105/humaneval_plus_phase3_retirement_bench/"
    "w105_phase3_FINAL_consolidated/phase3_retirement_verdict.json")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verdict", default=_DEFAULT_VERDICT)
    ap.add_argument(
        "--slice-type",
        choices=[SLICE_RESCUE_CONCENTRATED, SLICE_FAIR_BROAD],
        default=SLICE_RESCUE_CONCENTRATED)
    ap.add_argument(
        "--fair-result-exists", dest="fair_result_exists",
        action="store_true", default=True)
    ap.add_argument(
        "--no-fair-result-exists", dest="fair_result_exists",
        action="store_false")
    ap.add_argument(
        "--out-dir", default="results/w106/margin_cap_dispatch")
    args = ap.parse_args()

    with open(args.verdict, encoding="utf-8") as fh:
        verdict = json.load(fh)

    decision = build_margin_cap_dispatch_decision_v1(
        phase3_verdict=verdict,
        proposed_confirmation_slice_type=args.slice_type,
        fair_broad_phase3_result_exists=args.fair_result_exists)

    md = format_margin_cap_dispatch_markdown_v1(decision=decision)
    print(md)
    print(f"decision CID: {decision.cid()}")

    os.makedirs(args.out_dir, exist_ok=True)
    out_json = os.path.join(
        args.out_dir, "margin_cap_dispatch_decision.json")
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(
            {"decision": decision.to_dict(),
             "decision_cid": decision.cid()},
            fh, indent=2, sort_keys=True)
    out_md = os.path.join(
        args.out_dir, "margin_cap_dispatch_decision.md")
    with open(out_md, "w", encoding="utf-8") as fh:
        fh.write(md)
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")

    # NO-GO is the expected/honest outcome under the W106 RUNBOOK;
    # exit 0 either way (the decision is the deliverable, not a
    # pass/fail gate).
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
