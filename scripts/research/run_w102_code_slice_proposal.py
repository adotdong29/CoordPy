#!/usr/bin/env python3
"""W102 — code-side slice-selection + candidate-ranking helper
driver (COO-14 deliverable).

Reads the W101 arsenal-mining report and emits:

* A Markdown candidate-direction ranking table (`ranking.md`).
* A Markdown cheap-pilot slice proposal per bench (`slice_<bench>.md`).
* A combined JSON artifact (`proposals.json`) the runbook can
  reference by CID.

NIM-free; no model loading; runs in seconds.

Usage::

    python scripts/run_w102_code_slice_proposal.py
    python scripts/run_w102_code_slice_proposal.py --bench humaneval --n-problems 30
    python scripts/run_w102_code_slice_proposal.py --mining-report results/w101/arsenal_mining/<RUN>/mining_report.json
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from coordpy.code_slice_selector_v1 import (  # noqa: E402
    format_ranking_markdown,
    format_slice_proposal_markdown,
    load_mining_report,
    propose_cheap_pilot_slice,
    rank_candidate_benches,
)


def _latest_mining_report() -> Path:
    for sub in ("w102", "w101"):
        latest_ptr = (
            ROOT / "results" / sub / "arsenal_mining"
            / "latest_run.txt")
        if latest_ptr.exists():
            pointer = latest_ptr.read_text().strip()
            cand = Path(pointer)
            if not cand.is_absolute():
                cand = latest_ptr.parent / cand
            target = cand / "mining_report.json"
            if target.exists():
                return target
    return (
        ROOT / "results" / "w101" / "arsenal_mining"
        / "MISSING" / "mining_report.json")


def main() -> int:
    ap = argparse.ArgumentParser(description=(
        "W102 code-side slice-selection + candidate-ranking "
        "helper driver"))
    ap.add_argument(
        "--mining-report",
        default=str(_latest_mining_report()),
        help="Path to W101/W102 arsenal-mining report JSON")
    ap.add_argument(
        "--benches", nargs="+",
        default=["humaneval", "mbpp"],
        help=(
            "Benches to rank + propose slices for "
            "(default: humaneval mbpp)"))
    ap.add_argument(
        "--n-problems", type=int, default=30,
        help="Cheap-pilot slice size per bench (default 30)")
    ap.add_argument(
        "--out-dir",
        default=str(
            ROOT / "results" / "w102"
            / "code_slice_proposals"),
        help="Output root")
    args = ap.parse_args()

    mining_path = Path(args.mining_report)
    print(f"  mining report = {mining_path}")
    if not mining_path.exists():
        print(f"  ERROR: mining report not found at {mining_path}")
        return 2
    mining = load_mining_report(mining_path)

    run_id = _dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_dir = (
        Path(args.out_dir) / f"w102_slice_proposals_{run_id}")
    out_dir.mkdir(parents=True, exist_ok=True)

    rankings = rank_candidate_benches(
        mining=mining, benches=args.benches)
    ranking_md = format_ranking_markdown(rankings)
    (out_dir / "ranking.md").write_text(ranking_md)
    print()
    print(ranking_md)

    proposals_json: dict[str, dict] = {}
    for bench in args.benches:
        try:
            proposal = propose_cheap_pilot_slice(
                mining=mining, bench=bench,
                n_problems=int(args.n_problems))
        except ValueError as e:
            print(f"  SKIP {bench}: {e}")
            continue
        slice_md = format_slice_proposal_markdown(proposal)
        (out_dir / f"slice_{bench}.md").write_text(slice_md)
        print(slice_md)
        proposals_json[bench] = proposal.to_dict()

    payload = {
        "schema": "coordpy.w102_code_slice_proposal_v1",
        "mining_report_path": str(mining_path),
        "benches": list(args.benches),
        "n_problems": int(args.n_problems),
        "rankings": [r.to_dict() for r in rankings],
        "proposals": proposals_json,
    }
    (out_dir / "proposals.json").write_text(
        json.dumps(payload, indent=2, default=str))
    latest = out_dir.parent / "latest_run.txt"
    latest.write_text(out_dir.name + "\n")
    print(f"  out_dir: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
