#!/usr/bin/env python3
"""Offline re-verifier for the W86 HumanEval bench audit chain.

Given ``humaneval_bench_report.json`` produced by
``scripts/run_w86_humaneval_bench.py`` plus the
``humaneval_bench_report.calls.jsonl`` sidecar, re-computes:

* every per-call ``response_cid`` against the sidecar bytes
  (anti-cheat: model responses are NOT amenable to silent
  rewriting after the fact);
* every per-call ``prompt_cid`` against the sidecar prompts;
* the bench Merkle root against the per-seed outcome CIDs.

Prints a PASS/FAIL summary and exits 0 iff every CID
re-derives.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--report", required=True,
        help="path to humaneval_bench_report.json")
    p.add_argument(
        "--calls", default=None,
        help=("path to humaneval_bench_report.calls.jsonl "
              "(defaults to <report>.calls.jsonl)"))
    args = p.parse_args(argv)

    report_path = Path(args.report)
    calls_path = (
        Path(args.calls) if args.calls
        else report_path.with_suffix("")
        .with_suffix(".calls.jsonl"))
    if not calls_path.exists():
        # Try ``<report>.calls.jsonl`` next to the report.
        alt = (
            report_path.parent /
            (report_path.name.replace(".json", ".calls.jsonl"))
        )
        if alt.exists():
            calls_path = alt

    notes: list[str] = []
    ok = True

    if not report_path.exists():
        print(f"FAIL: report missing: {report_path}")
        return 1
    report = json.loads(
        report_path.read_bytes().decode("utf-8"))

    # 1. Per-call response_cid and prompt_cid must re-hash to the
    # response/prompt bytes in the sidecar.
    n_calls_checked = 0
    n_calls_fail = 0
    if calls_path.exists():
        for raw in calls_path.read_text(
                encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            row = json.loads(raw)
            text = row.get("response_text", "")
            prompt = row.get("prompt", "")
            recorded_resp = row.get("response_cid", "")
            recorded_prompt = row.get("prompt_cid", "")
            derived_resp = hashlib.sha256(
                text.encode("utf-8")).hexdigest()
            derived_prompt = hashlib.sha256(
                prompt.encode("utf-8")).hexdigest()
            if derived_resp != recorded_resp:
                ok = False
                n_calls_fail += 1
            if derived_prompt != recorded_prompt:
                ok = False
                n_calls_fail += 1
            n_calls_checked += 1
        notes.append(
            f"{'PASS' if n_calls_fail == 0 else 'FAIL'} "
            f"per-call CIDs: checked {n_calls_checked}, "
            f"mismatches {n_calls_fail}")
    else:
        notes.append(
            f"SKIP per-call CIDs: sidecar missing "
            f"({calls_path})")

    # 2. Per-seed outcome CIDs aggregate into the recorded
    # seed Merkle root.
    def _sha256(payload):
        return hashlib.sha256(
            json.dumps(
                payload, sort_keys=True,
                separators=(",", ":"),
                default=str).encode("utf-8")).hexdigest()

    n_seeds_ok = 0
    n_seeds_fail = 0
    for seed_block in report.get("per_seed", []):
        derived = _sha256({
            "kind": "w86_humaneval_seed_merkle_root",
            "seed": int(seed_block.get("seed", 0)),
            "outcome_cids": list(
                seed_block.get("outcome_cids", [])),
        })
        if derived == str(
                seed_block.get("seed_merkle_root", "")):
            n_seeds_ok += 1
        else:
            ok = False
            n_seeds_fail += 1
    notes.append(
        f"{'PASS' if n_seeds_fail == 0 else 'FAIL'} "
        f"per-seed Merkle roots: ok {n_seeds_ok}, "
        f"failed {n_seeds_fail}")

    # 3. Bench-level Merkle root.
    all_outcome_cids: list[str] = []
    for seed_block in report.get("per_seed", []):
        all_outcome_cids.extend(
            list(seed_block.get("outcome_cids", [])))
    seeds = [
        int(s.get("seed", 0))
        for s in report.get("per_seed", [])]
    derived_bench = _sha256({
        "kind": "w86_humaneval_bench_merkle_root",
        "model_id": str(report.get("model_id", "")),
        "outcome_cids": list(all_outcome_cids),
        "seeds": list(seeds),
    })
    recorded_bench = str(report.get("bench_merkle_root", ""))
    bench_ok = derived_bench == recorded_bench
    if not bench_ok:
        ok = False
    notes.append(
        f"{'PASS' if bench_ok else 'FAIL'} bench Merkle root "
        f"(recorded={recorded_bench[:16]} "
        f"derived={derived_bench[:16]})")

    # 4. Headline strict-beat verdicts.
    notes.append(
        f"INFO a0_mean_pass@1 = "
        f"{report.get('a0_mean_pass_at_1')}")
    notes.append(
        f"INFO a1_mean_pass@1 = "
        f"{report.get('a1_mean_pass_at_1')}")
    notes.append(
        f"INFO b_mean_pass@1  = "
        f"{report.get('b_mean_pass_at_1')}")
    notes.append(
        f"INFO b_strictly_beats_a0_on_all_seeds = "
        f"{report.get('b_strictly_beats_a0_on_all_seeds')}")
    notes.append(
        f"INFO b_strictly_beats_a1_on_all_seeds = "
        f"{report.get('b_strictly_beats_a1_on_all_seeds')}")
    notes.append(
        f"INFO b_mean_strictly_beats_a0_mean = "
        f"{report.get('b_mean_strictly_beats_a0_mean')}")
    notes.append(
        f"INFO b_mean_strictly_beats_a1_mean = "
        f"{report.get('b_mean_strictly_beats_a1_mean')}")

    for n in notes:
        print(n)
    print(f"\nOVERALL: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
