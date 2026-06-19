"""W88 HumanEval sequential-reflexion bench — offline audit chain verifier.

Reads the bench report JSON + per-call sidecar JSONL and re-derives:

* every per-call ``prompt_cid`` / ``response_cid`` byte-for-byte
  from the persisted prompt/response strings in the sidecar
* the per-seed Merkle root from the outcome CIDs
* the bench Merkle root from the per-seed roots + model_id +
  seeds

If any of these mismatch, the verifier prints FAIL and exits
non-zero.

It also prints the load-bearing strict-improvement bools that
define whether W88 retires
``W86-L-HUMANEVAL-V1-A1-SAME-BUDGET-NOT-BEATEN``.
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="W88 HumanEval-reflexion bench audit "
                    "chain verifier")
    parser.add_argument(
        "--report",
        help="Path to humaneval_reflexion_bench_report.json")
    parser.add_argument(
        "--run-dir",
        help="If --report is not given, --run-dir is used to "
             "find the bench report + calls sidecar (otherwise "
             "the 'latest_run.txt' pointer is consulted)")
    args = parser.parse_args()

    if args.report:
        report_path = Path(args.report)
        run_dir = report_path.parent
    elif args.run_dir:
        run_dir = Path(args.run_dir)
        report_path = (
            run_dir / "humaneval_reflexion_bench_report.json")
    else:
        ptr = (
            ROOT / "results" / "w88" / "humaneval_reflexion"
            / "latest_run.txt")
        if not ptr.exists():
            raise SystemExit(
                "no --report / --run-dir given and no "
                "latest_run.txt found")
        pointer_value = ptr.read_text().strip()
        candidate = Path(pointer_value)
        # Pointers are relative to the pointer's directory in V2;
        # absolute pointers from V1 still work.
        if not candidate.is_absolute():
            candidate = ptr.parent / candidate
        run_dir = candidate
        report_path = (
            run_dir / "humaneval_reflexion_bench_report.json")

    sidecar_path = run_dir / "humaneval_reflexion_calls.jsonl"

    print(f"report  = {report_path}")
    print(f"sidecar = {sidecar_path}")
    if not report_path.exists():
        raise SystemExit(f"report not found: {report_path}")

    with open(report_path) as f:
        report = json.load(f)

    n_pass = 0
    n_fail = 0

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal n_pass, n_fail
        if ok:
            n_pass += 1
            print(f"  PASS  {name}")
        else:
            n_fail += 1
            print(f"  FAIL  {name}{(': ' + detail) if detail else ''}")

    # Verify per-call CIDs from sidecar
    if sidecar_path.exists():
        sidecar_calls = []
        with open(sidecar_path) as f:
            for line in f:
                if not line.strip():
                    continue
                sidecar_calls.append(json.loads(line))
        n_calls = len(sidecar_calls)
        n_bad = 0
        for rec in sidecar_calls:
            p = str(rec.get("prompt", ""))
            r = str(rec.get("response_text", ""))
            p_sha_expected = str(rec.get("prompt_sha256", ""))
            r_sha_expected = str(rec.get("response_sha256", ""))
            p_sha_actual = hashlib.sha256(
                p.encode("utf-8")).hexdigest()
            r_sha_actual = hashlib.sha256(
                r.encode("utf-8")).hexdigest()
            if (p_sha_actual.lower() != p_sha_expected.lower()
                    or r_sha_actual.lower() !=
                    r_sha_expected.lower()):
                n_bad += 1
        check(
            f"sidecar per-call SHA-256 re-derive (N={n_calls})",
            n_bad == 0,
            f"{n_bad} mismatches" if n_bad else "")
    else:
        check("sidecar present", False,
              f"sidecar not found at {sidecar_path}")

    # Re-derive per-seed Merkle root from outcome_cids
    n_seeds = int(report.get("n_seeds", 0))
    per_seed_ok = True
    for ps in report.get("per_seed", []):
        seed = int(ps["seed"])
        outcome_cids = list(ps["outcome_cids"])
        recorded_root = str(ps["seed_merkle_root"])
        derived = _sha256_hex({
            "kind": "w88_humaneval_reflexion_seed_merkle_root",
            "seed": int(seed),
            "outcome_cids": outcome_cids,
        })
        if derived != recorded_root:
            per_seed_ok = False
    check(
        f"per-seed Merkle root re-derive (N={n_seeds})",
        per_seed_ok)

    # Re-derive bench Merkle root from all outcome CIDs
    all_cids: list[str] = []
    for ps in report.get("per_seed", []):
        all_cids.extend(ps["outcome_cids"])
    derived_bench = _sha256_hex({
        "kind": "w88_humaneval_reflexion_bench_merkle_root",
        "model_id": str(report.get("model_id", "")),
        "outcome_cids": all_cids,
        "seeds": [
            int(ps["seed"])
            for ps in report.get("per_seed", [])],
    })
    check(
        "bench Merkle root re-derive",
        derived_bench == str(report.get("bench_merkle_root", "")))

    # ----- Load-bearing strict-improvement bools -----
    a0 = float(report["a0_mean_pass_at_1"])
    a1 = float(report["a1_mean_pass_at_1"])
    b = float(report["b_mean_pass_at_1"])
    b_minus_a1 = float(report.get(
        "b_mean_minus_a1_mean_pp", 0.0))
    b_beats_a0_seeds = list(
        report.get("b_beats_a0_per_seed", []))
    b_beats_a1_seeds = list(
        report.get("b_beats_a1_per_seed", []))
    b_beats_a0_count = sum(1 for x in b_beats_a0_seeds if x)
    b_beats_a1_count = sum(1 for x in b_beats_a1_seeds if x)
    print()
    print(f"  A0 mean pass@1: {a0:.4f}")
    print(f"  A1 mean pass@1: {a1:.4f}")
    print(f"  B  mean pass@1: {b:.4f}")
    print(f"  B − A1: {b_minus_a1:+.2f} pp")
    print(f"  B > A0 per seed: {b_beats_a0_seeds} "
          f"({b_beats_a0_count}/{n_seeds})")
    print(f"  B > A1 per seed: {b_beats_a1_seeds} "
          f"({b_beats_a1_count}/{n_seeds})")

    # The W88 retirement bars (mirror docs/RUNBOOK_W88.md)
    strong_beats_a1_mean = bool(
        report.get("b_mean_strictly_beats_a1_mean", False))
    strong_margin_ok = bool(b_minus_a1 >= 1.0)
    per_seed_majority = bool(
        b_beats_a1_count * 2 >= n_seeds + 1)  # > half
    strong_beats_a0_mean = bool(
        report.get("b_mean_strictly_beats_a0_mean", False))

    print()
    print("  W88 retirement bars vs "
          "W86-L-HUMANEVAL-V1-A1-SAME-BUDGET-NOT-BEATEN:")
    check("(1) b_mean_strictly_beats_a1_mean",
          strong_beats_a1_mean)
    check(f"(2) B − A1 margin ≥ +1.0 pp ({b_minus_a1:+.2f})",
          strong_margin_ok)
    check("(3) b_mean_strictly_beats_a0_mean (W86 reproduces)",
          strong_beats_a0_mean)
    check(
        f"(4) B beats A1 on > half of seeds "
        f"({b_beats_a1_count}/{n_seeds})",
        per_seed_majority)
    all_bars_met = (
        strong_beats_a1_mean and strong_margin_ok
        and strong_beats_a0_mean and per_seed_majority)

    print()
    print(f"  PASS: {n_pass}")
    print(f"  FAIL: {n_fail}")
    if n_fail == 0:
        print("OVERALL: PASS (audit chain re-derives)")
    else:
        print("OVERALL: FAIL")
    print(
        f"W88 strong-retirement bar (all 4): "
        f"{'MET' if all_bars_met else 'NOT MET'}")

    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
