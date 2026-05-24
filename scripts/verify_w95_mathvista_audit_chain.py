#!/usr/bin/env python3
"""W95 MathVista bench — offline audit verifier.

Re-derives the per-call SHAs, per-seed Merkle roots, and bench
Merkle root from the on-disk sidecars + bench report.  Refuses
to declare PASS if any element drifts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _canonical_bytes(payload) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _verify_sidecar(path: Path) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    calls = []
    with open(path) as f:
        for line in f:
            if line.strip():
                calls.append(json.loads(line))
    n_bad = 0
    for rec in calls:
        p = str(rec.get("prompt", ""))
        r = str(rec.get("response_text", ""))
        p_exp = str(rec.get("prompt_sha256", ""))
        r_exp = str(rec.get("response_sha256", ""))
        p_act = hashlib.sha256(p.encode("utf-8")).hexdigest()
        r_act = hashlib.sha256(r.encode("utf-8")).hexdigest()
        if (p_act.lower() != p_exp.lower()
                or r_act.lower() != r_exp.lower()):
            n_bad += 1
    return len(calls), n_bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report")
    ap.add_argument("--run-dir")
    args = ap.parse_args()

    if args.report:
        report_path = Path(args.report)
        run_dir = report_path.parent
    elif args.run_dir:
        run_dir = Path(args.run_dir)
        report_path = run_dir / "bench_report.json"
    else:
        ptr = (
            ROOT / "results" / "w95"
            / "mathvista_pilot" / "latest_run.txt")
        if not ptr.exists():
            raise SystemExit(
                "no --report/--run-dir given and no "
                "latest_run.txt under results/w95/mathvista_pilot/")
        v = ptr.read_text().strip()
        c = Path(v)
        if not c.is_absolute():
            c = ptr.parent / c
        run_dir = c
        report_path = run_dir / "bench_report.json"

    text_sidecar = run_dir / "text_calls.jsonl"
    vlm_sidecar = run_dir / "vlm_calls.jsonl"
    pp_sidecar = run_dir / "per_problem.jsonl"
    print(f"report = {report_path}")
    if not report_path.exists():
        raise SystemExit(f"report not found: {report_path}")
    with open(report_path) as f:
        report = json.load(f)

    n_pass = 0
    n_fail = 0

    def check(name, ok, detail=""):
        nonlocal n_pass, n_fail
        if ok:
            n_pass += 1
            print(f"  PASS  {name}")
        else:
            n_fail += 1
            print(
                f"  FAIL  {name}"
                f"{(': ' + detail) if detail else ''}")

    nt, ntbad = _verify_sidecar(text_sidecar)
    check(
        f"text-sidecar SHA (N={nt})",
        ntbad == 0,
        f"{ntbad} mismatches" if ntbad else "")
    nv, nvbad = _verify_sidecar(vlm_sidecar)
    check(
        f"vlm-sidecar SHA (N={nv})",
        nvbad == 0,
        f"{nvbad} mismatches" if nvbad else "")
    if pp_sidecar.exists():
        with open(pp_sidecar) as f:
            n_pp = sum(1 for ln in f if ln.strip())
        check(f"per-problem sidecar lines (N={n_pp})", n_pp > 0)

    per_seed = report.get("per_seed", [])
    per_seed_ok = True
    for ps in per_seed:
        d = _sha256_hex({
            "kind": "w95_mathvista_seed_merkle_root_v1",
            "seed": int(ps["seed"]),
            "outcome_cids": list(ps["outcome_cids"]),
        })
        if d != str(ps["seed_merkle_root"]):
            per_seed_ok = False
    check(
        f"per-seed Merkle root (N={len(per_seed)})",
        per_seed_ok)

    all_cids: list[str] = []
    for ps in per_seed:
        all_cids.extend(ps["outcome_cids"])
    derived = _sha256_hex({
        "kind": "w95_mathvista_bench_merkle_root_v1",
        "vlm_model_id": str(report.get("vlm_model_id", "")),
        "text_model_id": str(report.get("text_model_id", "")),
        "corpus_parquet_sha256": str(
            report.get("corpus_parquet_sha256", "")),
        "corpus_merkle_root": str(
            report.get("corpus_merkle_root", "")),
        "outcome_cids": all_cids,
        "seeds": [int(ps["seed"]) for ps in per_seed],
        "n_problems": int(report.get("n_problems", 0)),
        "K": int(report.get("K_multi_sample", 0)),
    })
    check(
        "bench Merkle root",
        derived == str(report.get("bench_merkle_root", "")))

    a0 = float(report["a0_text_mean_pass_at_1"])
    a1 = float(report["a1_vlm_mean_pass_at_1"])
    b = float(report["b_vlm_team_mean_pass_at_1"])
    d_a0 = float(report.get(
        "b_mean_minus_a0_text_mean_pp", 0.0))
    d_a1 = float(report.get(
        "b_mean_minus_a1_vlm_mean_pp", 0.0))
    n_seeds = int(report.get("n_seeds", 0))
    ba0 = list(report.get("b_beats_a0_text_per_seed", []))
    ba1 = list(report.get("b_beats_a1_vlm_per_seed", []))
    n_b_ge_a1_per_seed = list(report.get(
        "n_b_ge_a1_problems_per_seed", []))

    print()
    print(f"  Parquet SHA: {report.get('corpus_parquet_sha256')}")
    print(
        f"  Corpus Merkle: "
        f"{report.get('corpus_merkle_root')}")
    print(f"  A0_text    mean pass@1: {a0:.4f}")
    print(f"  A1_vlm K=5 mean pass@1: {a1:.4f}")
    print(f"  B_vlm_team mean pass@1: {b:.4f}")
    print(f"  B − A0_text: {d_a0:+.2f} pp")
    print(f"  B − A1_vlm:  {d_a1:+.2f} pp")
    print(
        f"  B > A0_text per seed: {ba0} "
        f"({sum(1 for x in ba0 if x)}/{n_seeds})")
    print(
        f"  B > A1_vlm  per seed: {ba1} "
        f"({sum(1 for x in ba1 if x)}/{n_seeds})")
    print(f"  B ≥ A1 problems per seed: {n_b_ge_a1_per_seed}")

    # Phase 2 pilot gates (single-seed) and/or Phase 3
    # retirement bars (multi-seed) — re-derive whichever
    # the bench produced.
    phase2_path = run_dir / "phase2_gates.json"
    phase3_path = run_dir / "phase3_retirement_bars.json"
    if phase2_path.exists():
        with open(phase2_path) as f:
            p2 = json.load(f)
        print()
        print(
            "  W95 Phase 2 pilot gates "
            f"(overall_passes={p2['overall_passes']}):")
        for g in p2["gates"]:
            check(
                f"  {g['gate']}",
                bool(g["pass"]),
                g["summary"] if not g["pass"] else "")
    if phase3_path.exists():
        with open(phase3_path) as f:
            p3 = json.load(f)
        print()
        print(
            "  W95 Phase 3 retirement bars "
            f"(overall_passes={p3['overall_passes']}):")
        for g in p3["gates"]:
            check(
                f"  {g['gate']}",
                bool(g["pass"]),
                g["summary"] if not g["pass"] else "")
    print()
    print(f"  PASS: {n_pass}")
    print(f"  FAIL: {n_fail}")
    if n_fail == 0:
        print("OVERALL: PASS (audit chain re-derives + gates)")
    else:
        print("OVERALL: FAIL")
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
