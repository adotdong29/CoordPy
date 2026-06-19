"""W142 Lane α — moderate-`p` family screen driver.

Runs the locked RUNBOOK_W142 screen: $0 gates (G1 parser-neutral / G2 exact-oracle discriminating /
G3 gated-accumulator extractable / G4 novelty) on every candidate FIRST, then — only for $0-passing
families and only with --gen — measures the FAIR raw efficient-rate `p` over the neutral-prompt bank on
the frontier anchor, applies the moderate-`p` band rule (p∈[0.10,0.50] ∧ Wilson-95% excludes 0,1), and
reports whether Lane α succeeds (≥3 admitted families OR ≥2 modes, AND ≥2 distinct technique veins).

The $0 result (no --gen) already machine-checks the decisive de-risk finding: G3 ADMITS the counting /
two-deque veins and REJECTS the prefix-hash + binary-search-on-answer controls.

Usage:
  python scripts/run_w142_family_screen_v1.py --dollar0-only          # $0 gates, no NIM
  python scripts/run_w142_family_screen_v1.py --K 12 --prompts 0,1    # full screen (NIM)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from coordpy.moderate_p_family_slate_v1 import (  # noqa: E402
    build_screen_slate_v1, screen_slate_fingerprint_cid_v1, MODERATE_P_LO, MODERATE_P_HI)
from coordpy.moderate_p_family_screen_v1 import (  # noqa: E402
    dollar0_gates_v1, screen_family_v1, summarize_screen_v1)

MODEL = "meta/llama-3.3-70b-instruct"
MINTED_DATE = "2026-06-08"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dollar0-only", action="store_true", help="run only the $0 gates (no NIM)")
    ap.add_argument("--K", type=int, default=12, help="K_screen per neutral prompt")
    ap.add_argument("--prompts", default="0,1", help="comma FNB prompt indices")
    ap.add_argument("--knob", type=int, default=50000)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--timeout", type=float, default=4.0)
    ap.add_argument("--gate-timeout", type=float, default=6.0)
    ap.add_argument("--families", default="", help="optional comma subset of family ids to screen")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    prompt_indices = [int(x) for x in args.prompts.split(",") if x.strip() != ""]
    slate = build_screen_slate_v1(knob=args.knob)
    if args.families:
        keep = {f.strip() for f in args.families.split(",") if f.strip()}
        slate = [c for c in slate if c.family in keep]
    slate_cid = screen_slate_fingerprint_cid_v1(knob=args.knob)
    print(f"=== W142 screen slate CID {slate_cid[:16]} ; {len(slate)} candidates ; "
          f"band [{MODERATE_P_LO},{MODERATE_P_HI}] ===", flush=True)

    out_dir = os.path.dirname(args.out) if args.out else os.path.join(ROOT, "results", "w142", "screen")
    os.makedirs(out_dir, exist_ok=True)
    sidecar_path = (args.out.replace(".json", "") if args.out
                    else os.path.join(out_dir, "screen")) + "_calls.jsonl"

    gen = None
    if not args.dollar0_only:
        sidecar = open(sidecar_path, "w")  # noqa: SIM115

        def _writer(rec: dict) -> None:
            sidecar.write(json.dumps(rec) + "\n")
            sidecar.flush()
        from scripts.run_w108_livecodebench_pilot import _build_nim_gen  # noqa: E402
        gen = _build_nim_gen(model=MODEL, sidecar_writer=_writer, inter_call_sleep_s=0.0)

    results = []
    seen_sigs: list[str] = []
    seen_stmts: list[str] = []
    for c in slate:
        # $0 gates first (cumulative novelty check vs already-seen families)
        if gen is None:
            g = dollar0_gates_v1(c, timeout_s=args.gate_timeout, known_algo_sigs=seen_sigs,
                                 known_statements=seen_stmts)
            mt = c.factory(c.knob).minted
            seen_sigs.append(mt.algo_sig)
            seen_stmts.append(mt.statement)
            match = "OK" if g.g3_extractable == c.expect_extractable else "**G3-MISMATCH**"
            print(f"[$0] {c.family:34s} {c.vein:24s} G1={int(g.g1_parser_neutral)} "
                  f"G2={int(g.g2_discriminating)} G3={int(g.g3_extractable)} G4={int(g.g4_novel)} "
                  f"(npred={g.n_pred_holes},nadd={g.n_add_holes}) expectExtr={c.expect_extractable} {match}",
                  flush=True)
            results.append({"family": c.family, "vein": c.vein, "mode": c.mode,
                            "gates": g.to_dict(), "expect_extractable": c.expect_extractable})
            continue
        # full screen (NIM) — only $0-passing families get measured
        res = screen_family_v1(c, gen=gen, K_screen=args.K, prompt_indices=prompt_indices,
                               temperature=args.temperature, max_tokens=args.max_tokens,
                               timeout_s=args.timeout, known_algo_sigs=seen_sigs,
                               known_statements=seen_stmts)
        mt = c.factory(c.knob).minted
        seen_sigs.append(mt.algo_sig)
        seen_stmts.append(mt.statement)
        print(f"[screen] {c.family:34s} {c.vein:22s} $0={int(res.gates.all_pass)} "
              f"p_med={res.p_median:.3f} (min {res.p_min:.2f}/max {res.p_max:.2f}) "
              f"Wilson=[{res.wilson_lo:.3f},{res.wilson_hi:.3f}] band={int(res.in_band)} "
              f"ADMITTED={int(res.admitted)}", flush=True)
        results.append(res.to_dict())

    out_path = args.out or os.path.join(out_dir,
                                        "w142_dollar0_gates.json" if gen is None else "w142_screen.json")
    summary = {"schema": "coordpy.w142_family_screen.v1", "model": MODEL, "slate_cid": slate_cid,
               "band": [MODERATE_P_LO, MODERATE_P_HI], "dollar0_only": bool(gen is None),
               "K_screen": args.K, "prompt_indices": prompt_indices, "results": results}
    if gen is not None:
        # rebuild FamilyScreenResultV1s for the verdict (we kept dicts; re-screen verdict from objects)
        fres = []
        for c in slate:
            r = next((x for x in results if x.get("family") == c.family), None)
            if r is None:
                continue
        # recompute verdict from the live result objects we already have
        # (simpler: re-run summarize on freshly-built objects is unnecessary; store the raw flags)
        admitted = [r for r in results if r.get("admitted")]
        veins = sorted({r["vein"] for r in admitted})
        modes = sorted({r["mode"] for r in admitted})
        span_ok = (len(admitted) >= 3) or (len(modes) >= 2)
        summary["verdict"] = {
            "n_admitted": len(admitted),
            "admitted_families": [r["family"] for r in admitted],
            "admitted_veins": veins, "admitted_modes": modes,
            "span_ok_>=3fam_OR_>=2mode": bool(span_ok),
            "lane_alpha_success": bool(span_ok and len(veins) >= 2),
        }
        print(f"\n=== W142 Lane α: {len(admitted)} admitted "
              f"({[r['family'] for r in admitted]}); veins={veins}; modes={modes}; "
              f"success={summary['verdict']['lane_alpha_success']} ===", flush=True)
    else:
        g3_ok = all((r["gates"]["g3_extractable"] == r["expect_extractable"]) for r in results)
        summary["g3_predictions_all_correct"] = bool(g3_ok)
        print(f"\n=== W142 $0 gates: G3 predictions all correct = {g3_ok} "
              f"(extractable veins admitted, prefix-hash + BSoA rejected) ===", flush=True)

    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"-> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
