"""W88 cross-modal code bench — offline audit chain verifier.

Re-derives every per-call SHA-256 from the persisted text + VLM
sidecars, every per-seed Merkle root from outcome CIDs, and the
bench Merkle root.  Also prints the load-bearing
strict-improvement bools that drive whether W88 retires
``W87-L-MULTI-MODAL-V1-NO-CROSS-MODAL-INJECT-CAP``.
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


def _verify_sidecar(path: Path, label: str,
                    include_image: bool = False) -> tuple[int, int]:
    """Return (n_calls, n_bad)."""
    if not path.exists():
        return 0, 0
    calls = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            calls.append(json.loads(line))
    n_bad = 0
    for rec in calls:
        p = str(rec.get("prompt", ""))
        r = str(rec.get("response_text", ""))
        p_sha_expected = str(rec.get("prompt_sha256", ""))
        r_sha_expected = str(rec.get("response_sha256", ""))
        p_sha_actual = hashlib.sha256(
            p.encode("utf-8")).hexdigest()
        r_sha_actual = hashlib.sha256(
            r.encode("utf-8")).hexdigest()
        if (p_sha_actual.lower() != p_sha_expected.lower()
                or r_sha_actual.lower()
                != r_sha_expected.lower()):
            n_bad += 1
    return len(calls), n_bad


def main() -> int:
    parser = argparse.ArgumentParser(
        description="W88 cross-modal code bench audit verifier")
    parser.add_argument(
        "--report",
        help="Path to cross_modal_code_bench_report.json")
    parser.add_argument(
        "--run-dir",
        help="Run directory (otherwise latest_run.txt is used)")
    args = parser.parse_args()

    if args.report:
        report_path = Path(args.report)
        run_dir = report_path.parent
    elif args.run_dir:
        run_dir = Path(args.run_dir)
        report_path = run_dir / (
            "cross_modal_code_bench_report.json")
    else:
        ptr = (
            ROOT / "results" / "w88" / "cross_modal_code"
            / "latest_run.txt")
        if not ptr.exists():
            raise SystemExit(
                "no --report / --run-dir given and no "
                "latest_run.txt found")
        pointer_value = ptr.read_text().strip()
        candidate = Path(pointer_value)
        if not candidate.is_absolute():
            candidate = ptr.parent / candidate
        run_dir = candidate
        report_path = run_dir / (
            "cross_modal_code_bench_report.json")

    text_sidecar = run_dir / "text_calls.jsonl"
    vlm_sidecar = run_dir / "vlm_calls.jsonl"

    print(f"report = {report_path}")
    print(f"text   = {text_sidecar}")
    print(f"vlm    = {vlm_sidecar}")
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
            print(f"  FAIL  {name}"
                  f"{(': ' + detail) if detail else ''}")

    n_text, n_text_bad = _verify_sidecar(
        text_sidecar, "text", include_image=False)
    check(
        f"text-sidecar SHA-256 re-derive (N={n_text})",
        n_text_bad == 0,
        f"{n_text_bad} mismatches" if n_text_bad else "")
    n_vlm, n_vlm_bad = _verify_sidecar(
        vlm_sidecar, "vlm", include_image=True)
    check(
        f"vlm-sidecar SHA-256 re-derive (N={n_vlm})",
        n_vlm_bad == 0,
        f"{n_vlm_bad} mismatches" if n_vlm_bad else "")

    # Per-seed Merkle roots
    per_seed = report.get("per_seed", [])
    per_seed_ok = True
    for ps in per_seed:
        derived = _sha256_hex({
            "kind": "w88_cross_modal_code_seed_merkle_root",
            "seed": int(ps["seed"]),
            "outcome_cids": list(ps["outcome_cids"]),
        })
        if derived != str(ps["seed_merkle_root"]):
            per_seed_ok = False
    check(
        f"per-seed Merkle root re-derive (N={len(per_seed)})",
        per_seed_ok)

    # Bench Merkle root
    all_cids: list[str] = []
    for ps in per_seed:
        all_cids.extend(ps["outcome_cids"])
    derived_bench = _sha256_hex({
        "kind": "w88_cross_modal_code_bench_merkle_root",
        "vlm_model_id": str(report.get("vlm_model_id", "")),
        "code_model_id": str(report.get("code_model_id", "")),
        "outcome_cids": all_cids,
        "seeds": [int(ps["seed"]) for ps in per_seed],
    })
    check(
        "bench Merkle root re-derive",
        derived_bench == str(report.get("bench_merkle_root", "")))

    # Load-bearing bools
    a0 = float(report["a0_text_mean_pass_at_1"])
    a1 = float(report["a1_vlm_mean_pass_at_1"])
    b = float(report["b_cross_mean_pass_at_1"])
    b_minus_a0 = float(report.get(
        "b_cross_mean_minus_a0_text_mean_pp", 0.0))
    b_minus_a1 = float(report.get(
        "b_cross_mean_minus_a1_vlm_mean_pp", 0.0))
    n_seeds = int(report.get("n_seeds", 0))
    b_beats_a0_seeds = list(
        report.get("b_cross_beats_a0_text_per_seed", []))
    b_beats_a1_seeds = list(
        report.get("b_cross_beats_a1_vlm_per_seed", []))
    b_beats_a0_count = sum(1 for x in b_beats_a0_seeds if x)
    b_beats_a1_count = sum(1 for x in b_beats_a1_seeds if x)

    print()
    print(f"  A0_text mean pass@1: {a0:.4f}")
    print(f"  A1_vlm  mean pass@1: {a1:.4f}")
    print(f"  B_cross mean pass@1: {b:.4f}")
    print(f"  B − A0_text: {b_minus_a0:+.2f} pp")
    print(f"  B − A1_vlm:  {b_minus_a1:+.2f} pp")
    print(
        f"  B > A0_text per seed: {b_beats_a0_seeds} "
        f"({b_beats_a0_count}/{n_seeds})")
    print(
        f"  B > A1_vlm  per seed: {b_beats_a1_seeds} "
        f"({b_beats_a1_count}/{n_seeds})")

    strong_beats_a0_mean = bool(
        report.get(
            "b_cross_mean_strictly_beats_a0_text_mean", False))
    strong_beats_a1_mean = bool(
        report.get(
            "b_cross_mean_strictly_beats_a1_vlm_mean", False))
    strong_a0_margin = bool(b_minus_a0 >= 5.0)
    strong_a1_margin = bool(b_minus_a1 >= 5.0)
    per_seed_a0_majority = bool(
        b_beats_a0_count * 2 >= n_seeds + 1)
    per_seed_a1_majority = bool(
        b_beats_a1_count * 2 >= n_seeds + 1)

    print()
    print("  W88 retirement bars vs "
          "W87-L-MULTI-MODAL-V1-NO-CROSS-MODAL-INJECT-CAP:")
    check(
        "(1) b_cross_mean_strictly_beats_a0_text_mean "
        "(image is load-bearing)",
        strong_beats_a0_mean)
    check(
        "(2) b_cross_mean_strictly_beats_a1_vlm_mean "
        "(team organisation is load-bearing)",
        strong_beats_a1_mean)
    check(
        f"(3) B − A0_text margin ≥ +5.0 pp "
        f"({b_minus_a0:+.2f})",
        strong_a0_margin)
    check(
        f"(4) B − A1_vlm margin ≥ +5.0 pp "
        f"({b_minus_a1:+.2f})",
        strong_a1_margin)
    check(
        f"(5) B beats A0_text on > half of seeds "
        f"({b_beats_a0_count}/{n_seeds})",
        per_seed_a0_majority)
    check(
        f"(6) B beats A1_vlm on > half of seeds "
        f"({b_beats_a1_count}/{n_seeds})",
        per_seed_a1_majority)
    all_bars_met = (
        strong_beats_a0_mean and strong_beats_a1_mean
        and strong_a0_margin and strong_a1_margin
        and per_seed_a0_majority and per_seed_a1_majority)

    print()
    print(f"  PASS: {n_pass}")
    print(f"  FAIL: {n_fail}")
    if n_fail == 0:
        print("OVERALL: PASS (audit chain re-derives)")
    else:
        print("OVERALL: FAIL")
    print(
        f"W88 cross-modal strong-retirement bar (all 6): "
        f"{'MET' if all_bars_met else 'NOT MET'}")

    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
