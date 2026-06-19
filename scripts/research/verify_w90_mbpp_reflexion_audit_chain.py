"""W90 MBPP sequential-reflexion bench — offline audit verifier."""
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report")
    parser.add_argument("--run-dir")
    args = parser.parse_args()

    if args.report:
        report_path = Path(args.report)
        run_dir = report_path.parent
    elif args.run_dir:
        run_dir = Path(args.run_dir)
        report_path = run_dir / "mbpp_reflexion_bench_report.json"
    else:
        ptr = ROOT / "results" / "w90" / "mbpp_reflexion" / "latest_run.txt"
        if not ptr.exists():
            raise SystemExit(
                "no --report/--run-dir given and no latest_run.txt")
        v = ptr.read_text().strip()
        c = Path(v)
        if not c.is_absolute():
            c = ptr.parent / c
        run_dir = c
        report_path = run_dir / "mbpp_reflexion_bench_report.json"

    sidecar = run_dir / "mbpp_reflexion_calls.jsonl"
    print(f"report  = {report_path}")
    print(f"sidecar = {sidecar}")
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
            print(f"  FAIL  {name}{(': ' + detail) if detail else ''}")

    if sidecar.exists():
        calls = []
        with open(sidecar) as f:
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
        check(
            f"sidecar per-call SHA-256 (N={len(calls)})",
            n_bad == 0,
            f"{n_bad} mismatches" if n_bad else "")
    else:
        check("sidecar present", False)

    per_seed = report.get("per_seed", [])
    per_seed_ok = True
    for ps in per_seed:
        d = _sha256_hex({
            "kind": "w90_mbpp_seed_merkle_root",
            "seed": int(ps["seed"]),
            "outcome_cids": list(ps["outcome_cids"]),
        })
        if d != str(ps["seed_merkle_root"]):
            per_seed_ok = False
    check(
        f"per-seed Merkle root (N={len(per_seed)})", per_seed_ok)

    all_cids: list[str] = []
    for ps in per_seed:
        all_cids.extend(ps["outcome_cids"])
    derived = _sha256_hex({
        "kind": "w90_mbpp_bench_merkle_root",
        "model_id": str(report.get("model_id", "")),
        "outcome_cids": all_cids,
        "seeds": [int(ps["seed"]) for ps in per_seed],
    })
    check(
        "bench Merkle root",
        derived == str(report.get("bench_merkle_root", "")))

    a0 = float(report["a0_mean_pass_at_1"])
    a1 = float(report["a1_mean_pass_at_1"])
    b = float(report["b_mean_pass_at_1"])
    delta = float(report.get(
        "b_mean_minus_a1_mean_pp", (b - a1) * 100.0))
    n_seeds = int(report.get("n_seeds", 0))
    b_a0 = list(report.get("b_beats_a0_per_seed", []))
    b_a1 = list(report.get("b_beats_a1_per_seed", []))
    print()
    print(f"  A0 mean pass@1: {a0:.4f}")
    print(f"  A1 mean pass@1: {a1:.4f}")
    print(f"  B  mean pass@1: {b:.4f}")
    print(f"  B − A1: {delta:+.2f} pp")
    print(
        f"  B > A0 per seed: {b_a0} "
        f"({sum(1 for x in b_a0 if x)}/{n_seeds})")
    print(
        f"  B > A1 per seed: {b_a1} "
        f"({sum(1 for x in b_a1 if x)}/{n_seeds})")
    bar1 = bool(
        report.get("b_mean_strictly_beats_a1_mean", False))
    bar2 = bool(delta >= 1.0)
    bar3 = bool(
        report.get("b_mean_strictly_beats_a0_mean", False))
    bar4 = bool(
        sum(1 for x in b_a1 if x) * 2 >= n_seeds + 1)
    print()
    print("  W90 retirement bars vs "
          "W89-L-HUMANEVAL-REFLEXION-V2-HUMANEVAL-K5-SCALE-CAP "
          "(MBPP generalisation):")
    check("(1) b_mean_strictly_beats_a1_mean", bar1)
    check(f"(2) B − A1 ≥ +1.0 pp ({delta:+.2f})", bar2)
    check("(3) b_mean_strictly_beats_a0_mean", bar3)
    check(
        f"(4) B beats A1 on > half seeds "
        f"({sum(1 for x in b_a1 if x)}/{n_seeds})",
        bar4)
    all_met = bar1 and bar2 and bar3 and bar4
    print()
    print(f"  PASS: {n_pass}")
    print(f"  FAIL: {n_fail}")
    if n_fail == 0:
        print("OVERALL: PASS (audit chain re-derives)")
    else:
        print("OVERALL: FAIL")
    print(
        f"W90 MBPP strong-retirement bar (all 4): "
        f"{'MET' if all_met else 'NOT MET'}")
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
