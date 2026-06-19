#!/usr/bin/env python3
"""W132-α — build + gate the CoordPy-owned resistant-by-construction battlefield ($0 NIM).

Mints the W132 slate, runs every Lane-α quality gate (exact-oracle self-test, small-vs-
large brute agreement, discriminating-hidden-case, public/hidden split, deterministic
regeneration, novelty/near-duplicate guard, pass-fail-only), certifies Maverick
RESISTANCE on the minted (post-cutoff, freshly-generated) date boundary, selects the
deterministic mode-stratified core 30-slice, and writes the manifest + verdict JSON.

No model inference; the only code execution is the answer-key / oracle subprocess.

Usage::

    python scripts/run_w132_build_battlefield_v1.py
    python scripts/run_w132_build_battlefield_v1.py --minted-date 2026-06-02 --seed 132
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import sys  # noqa: E402
sys.path.insert(0, str(ROOT))

from coordpy.resistant_by_construction_battlefield_v1 import (  # noqa: E402
    DEFAULT_EXEC_TIMEOUT_S,
    certify_resistance_v1,
    core_slice_cid_v1,
    mint_battlefield_v1,
    select_core_slice_v1,
)
from coordpy.resistant_by_construction_slate_v1 import RBC_SLATE_V1  # noqa: E402
from coordpy.coordpy_icpc_battlefield_v1 import (  # noqa: E402
    ICPC_BATTLEFIELD_LISTING_SNAPSHOT_V1,
)

W132_TARGET_MODEL = "meta/llama-4-maverick-17b-128e-instruct"
# official ICPC short-names (W120 listing) — the paraphrase guard refuses any minted
# statement that embeds an official problem identity token.
OFFICIAL_IDENTITIES = tuple(sorted({row[1] for row in ICPC_BATTLEFIELD_LISTING_SNAPSHOT_V1}))


def main() -> int:
    ap = argparse.ArgumentParser(description="W132 resistant-by-construction battlefield")
    ap.add_argument("--minted-date", default="2026-06-02")
    ap.add_argument("--seed", type=int, default=132)
    ap.add_argument("--timeout-s", type=float, default=DEFAULT_EXEC_TIMEOUT_S)
    ap.add_argument("--n-core", type=int, default=30)
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "w132" / "battlefield"))
    args = ap.parse_args()

    print(f"  minting {len(RBC_SLATE_V1)} templates "
          f"(seed={args.seed}, minted_date={args.minted_date}, "
          f"timeout={args.timeout_s}s) ...")
    bf = mint_battlefield_v1(
        RBC_SLATE_V1, global_seed=args.seed, minted_date=args.minted_date,
        timeout_s=float(args.timeout_s), official_identities=OFFICIAL_IDENTITIES)

    print("  --- per-problem gates ---")
    for p in bf.all_minted:
        g = p.gates
        tag = "ADMIT" if g.admitted else "DROP "
        print(f"   {tag} {p.problem_id:38s} {p.mode:26s} "
              f"brute={g.n_brute_checked:2d} naive_fail={g.n_naive_secret_fail} "
              f"{'' if g.admitted else g.reason}")
    if bf.novelty_rejected:
        print("  --- novelty rejections ---")
        for r in bf.novelty_rejected:
            print(f"   NOVELTY-DROP {r.problem_id} ~ {r.nearest_id} (J={r.jaccard:.3f})")

    print(f"\n  n_minted={bf.manifest.n_minted} n_gate_pass={bf.manifest.n_gate_pass} "
          f"n_admitted={bf.manifest.n_admitted} meets_min_slice={bf.meets_min_slice}")
    print(f"  mode_histogram   = {bf.manifest.mode_histogram}")
    print(f"  family_histogram = {bf.manifest.family_histogram}")
    print(f"  manifest_cid     = {bf.manifest.manifest_cid()[:16]}…  "
          f"raw_cid={bf.manifest.raw_cid[:16]}…")

    cert = certify_resistance_v1(
        model_id=W132_TARGET_MODEL, minted_date=args.minted_date,
        n_core=bf.manifest.n_admitted, raw_cid=bf.manifest.raw_cid)
    print(f"  resistance: {cert.note}  (reused_gate_certifiable={cert.reused_gate_certifiable})")

    core = select_core_slice_v1(bf, n_problems=int(args.n_core))
    core_cid = core_slice_cid_v1(core)
    core_modes: dict[str, int] = {}
    for p in core:
        core_modes[p.mode] = core_modes.get(p.mode, 0) + 1
    print(f"  core slice n={len(core)} cid={core_cid[:16]}…  modes={core_modes}")

    # ---- determinism: re-mint and assert byte-identical manifest CID ----
    bf2 = mint_battlefield_v1(
        RBC_SLATE_V1, global_seed=args.seed, minted_date=args.minted_date,
        timeout_s=float(args.timeout_s), official_identities=OFFICIAL_IDENTITIES)
    deterministic = bool(bf2.manifest.manifest_cid() == bf.manifest.manifest_cid())
    print(f"  deterministic_regeneration = {deterministic}")

    pilot_earned = bool(bf.meets_min_slice and cert.resistant
                        and len(core) >= int(args.n_core) and deterministic)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    verdict = {
        "schema": "coordpy.w132_battlefield_build.v1",
        "minted_date": args.minted_date, "global_seed": int(args.seed),
        "exec_timeout_s": float(args.timeout_s),
        "n_templates": len(RBC_SLATE_V1),
        "n_minted": bf.manifest.n_minted, "n_gate_pass": bf.manifest.n_gate_pass,
        "n_admitted": bf.manifest.n_admitted,
        "meets_min_slice_30": bf.meets_min_slice,
        "mode_histogram": bf.manifest.mode_histogram,
        "family_histogram": bf.manifest.family_histogram,
        "manifest_cid": bf.manifest.manifest_cid(),
        "raw_cid": bf.manifest.raw_cid,
        "core_slice_cid": core_cid, "core_slice_n": len(core),
        "core_slice_modes": core_modes,
        "core_slice_problem_ids": [p.problem_id for p in core],
        "resistance_cert": cert.to_dict(),
        "deterministic_regeneration": deterministic,
        "novelty_rejected": [r.to_dict() for r in bf.novelty_rejected],
        "official_identities_guarded": len(OFFICIAL_IDENTITIES),
        "battlefield_pilot_earned": pilot_earned,
        "target_model": W132_TARGET_MODEL,
    }
    (out_dir / "battlefield_verdict_v1.json").write_text(
        json.dumps(verdict, indent=2, default=str))
    (out_dir / "battlefield_manifest_v1.json").write_text(
        json.dumps(bf.to_dict(), indent=2, default=str))
    print(f"\n  battlefield_pilot_earned = {pilot_earned}")
    print(f"  wrote {out_dir}/battlefield_verdict_v1.json (+ manifest)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
