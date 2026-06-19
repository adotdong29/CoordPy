"""W90 cross-modal VLM-in-loop bench — offline audit verifier."""
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--report")
    parser.add_argument("--run-dir")
    args = parser.parse_args()

    if args.report:
        report_path = Path(args.report)
        run_dir = report_path.parent
    elif args.run_dir:
        run_dir = Path(args.run_dir)
        report_path = run_dir / "cross_modal_vlm_loop_bench_report.json"
    else:
        ptr = (
            ROOT / "results" / "w90" / "cross_modal_vlm_loop"
            / "latest_run.txt")
        if not ptr.exists():
            raise SystemExit(
                "no --report/--run-dir given and no latest_run.txt")
        v = ptr.read_text().strip()
        c = Path(v)
        if not c.is_absolute():
            c = ptr.parent / c
        run_dir = c
        report_path = run_dir / "cross_modal_vlm_loop_bench_report.json"

    text_sidecar = run_dir / "text_calls.jsonl"
    vlm_sidecar = run_dir / "vlm_calls.jsonl"
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
            print(f"  FAIL  {name}{(': ' + detail) if detail else ''}")

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

    per_seed = report.get("per_seed", [])
    per_seed_ok = True
    for ps in per_seed:
        d = _sha256_hex({
            "kind": "w90_cross_modal_vlm_loop_seed_merkle_root",
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
        "kind": "w90_cross_modal_vlm_loop_bench_merkle_root",
        "vlm_model_id": str(report.get("vlm_model_id", "")),
        "text_model_id": str(report.get("text_model_id", "")),
        "outcome_cids": all_cids,
        "seeds": [int(ps["seed"]) for ps in per_seed],
    })
    check(
        "bench Merkle root",
        derived == str(report.get("bench_merkle_root", "")))

    a0 = float(report["a0_text_mean_pass_at_1"])
    a1 = float(report["a1_vlm_mean_pass_at_1"])
    b = float(report["b_vlm_loop_mean_pass_at_1"])
    d_a0 = float(report.get(
        "b_vlm_loop_mean_minus_a0_text_mean_pp", 0.0))
    d_a1 = float(report.get(
        "b_vlm_loop_mean_minus_a1_vlm_mean_pp", 0.0))
    n_seeds = int(report.get("n_seeds", 0))
    ba0 = list(report.get(
        "b_vlm_loop_beats_a0_text_per_seed", []))
    ba1 = list(report.get(
        "b_vlm_loop_beats_a1_vlm_per_seed", []))
    print()
    print(f"  A0_text     mean pass@1: {a0:.4f}")
    print(f"  A1_vlm      mean pass@1: {a1:.4f}")
    print(f"  B_vlm_loop  mean pass@1: {b:.4f}")
    print(f"  B − A0_text: {d_a0:+.2f} pp")
    print(f"  B − A1_vlm:  {d_a1:+.2f} pp")
    print(
        f"  B > A0_text per seed: {ba0} "
        f"({sum(1 for x in ba0 if x)}/{n_seeds})")
    print(
        f"  B > A1_vlm  per seed: {ba1} "
        f"({sum(1 for x in ba1 if x)}/{n_seeds})")

    s_a0 = bool(report.get(
        "b_vlm_loop_mean_strictly_beats_a0_text_mean", False))
    s_a1 = bool(report.get(
        "b_vlm_loop_mean_strictly_beats_a1_vlm_mean", False))
    m_a0 = bool(d_a0 >= 5.0)
    m_a1 = bool(d_a1 >= 5.0)
    p_a0 = bool(sum(1 for x in ba0 if x) * 2 >= n_seeds + 1)
    p_a1 = bool(sum(1 for x in ba1 if x) * 2 >= n_seeds + 1)
    print()
    print("  W90 retirement bars vs "
          "W88-L-CROSS-MODAL-CODE-V1-SPLIT-NOT-LOAD-BEARING-CAP "
          "(VLM-in-loop replacement):")
    check("(1) b > a0_text mean (image load-bearing)", s_a0)
    check("(2) b > a1_vlm mean (team-organisation load-bearing)", s_a1)
    check(f"(3) B − A0_text ≥ +5.0 pp ({d_a0:+.2f})", m_a0)
    check(f"(4) B − A1_vlm ≥ +5.0 pp ({d_a1:+.2f})", m_a1)
    check(f"(5) B > A0_text > half seeds "
          f"({sum(1 for x in ba0 if x)}/{n_seeds})", p_a0)
    check(f"(6) B > A1_vlm > half seeds "
          f"({sum(1 for x in ba1 if x)}/{n_seeds})", p_a1)
    all_met = s_a0 and s_a1 and m_a0 and m_a1 and p_a0 and p_a1
    print()
    print(f"  PASS: {n_pass}")
    print(f"  FAIL: {n_fail}")
    if n_fail == 0:
        print("OVERALL: PASS (audit chain re-derives)")
    else:
        print("OVERALL: FAIL")
    print(
        f"W90 cross-modal VLM-loop strong-retirement bar "
        f"(all 6): {'MET' if all_met else 'NOT MET'}")
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
