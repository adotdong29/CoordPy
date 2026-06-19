"""W141 dev probe — emergent self-tutoring vs verified-selection (no-oracle), at equal budget.

For each family: DISCOVER a leak-audited holed-skeleton scaffold from the team's OWN no-oracle-
verified winning sample on a teacher instance, then AMORTIZE it across held-out members and compare
A1 (self-consistency pool@K), Baseline B (no-oracle verified-selection@K), and ST (self-tutoring:
K scaffolded → no-oracle verified-selection).  Supply pre-screen: a family whose discovery ABSTAINs
yields no scaffold (KEEP ≡ A1, correct $0 abstain).  The no-oracle guard holds by construction (the
model never sees ref/naive/brute/secret; grading is SCORING-only).

Usage:
  python scripts/run_w141_self_tutoring_dev_probe_v1.py \
      --families sum_nearest_smaller_left,count_pairs_sum_le_t --knob 20000 \
      --K 5 --members 3 --teacher-seed 1 --out results/w141/dev/<name>.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from coordpy.headroom_band_slate_v3 import CX_FACTORIES, FUNC_FACTORIES  # noqa: E402
from coordpy.resistant_by_construction_battlefield_v1 import mint_problem_v1  # noqa: E402
from coordpy.self_tutoring_controller_v1 import (  # noqa: E402
    discover_self_scaffold_v1, run_member_arms_v1, amortization_verdict_v1)
from scripts.run_w108_livecodebench_pilot import _build_nim_gen  # noqa: E402

MODEL = "meta/llama-3.3-70b-instruct"   # the FRONTIER target (resistant by construction)
MINTED_DATE = "2026-06-08"


def _factory(fam: str):
    return CX_FACTORIES.get(fam) or FUNC_FACTORIES.get(fam)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default="sum_nearest_smaller_left")
    ap.add_argument("--knob", type=int, default=20000)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--K-re", type=int, default=5)
    ap.add_argument("--K-discover", type=int, default=0, help="discover K (0=use --K); set higher for "
                    "reliable supply while amortizing at a low --K to expose the win on high-p families")
    ap.add_argument("--members", type=int, default=3)
    ap.add_argument("--teacher-seed", type=int, default=1)
    ap.add_argument("--member-seed-base", type=int, default=100)
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--timeout", type=float, default=4.0)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--out", default="")
    ap.add_argument("--mock", action="store_true", help="$0 plumbing: gen returns ref/brute by prompt")
    args = ap.parse_args()

    fams = [f.strip() for f in args.families.split(",") if f.strip()]
    out_dir = os.path.dirname(args.out) if args.out else os.path.join(ROOT, "results", "w141", "dev")
    os.makedirs(out_dir, exist_ok=True)
    sidecar_path = (args.out.replace(".json", "") if args.out else os.path.join(out_dir, "probe")) + "_calls.jsonl"
    sidecar = open(sidecar_path, "w")  # noqa: SIM115

    def _writer(rec: dict) -> None:
        sidecar.write(json.dumps(rec) + "\n")
        sidecar.flush()

    if args.mock:
        gen = None  # mock built per-family below
    else:
        gen = _build_nim_gen(model=MODEL, sidecar_writer=_writer, inter_call_sleep_s=0.0)

    results = []
    for fam in fams:
        fac = _factory(fam)
        if fac is None:
            print(f"[skip] unknown family {fam}", flush=True)
            continue
        template = fac(args.knob)
        teacher = mint_problem_v1(fac(args.knob).minted, global_seed=args.teacher_seed)
        g = _mock_gen(template) if args.mock else gen
        print(f"\n=== {fam}@{args.knob} : DISCOVER (teacher seed {args.teacher_seed}) ===", flush=True)
        disc = discover_self_scaffold_v1(template, teacher, gen=g, K=args.K,
                                         temperature=args.temperature, max_tokens=args.max_tokens,
                                         timeout_s=args.timeout, minted_date=MINTED_DATE)
        print(f"  discovered={disc.discovered} reason={disc.selection.reason} "
              f"n_winners={sum(1 for v in disc.selection.verdicts if v.is_winner)} "
              f"winner_passes_secret={disc.winner_passes_secret} "
              f"compile={disc.compile_report.reason if disc.compile_report else None}", flush=True)

        members = []
        for s in range(args.members):
            m = mint_problem_v1(fac(args.knob).minted, global_seed=args.member_seed_base + s)
            res = run_member_arms_v1(template, m, disc.scaffold, gen=g, K=args.K, K_re=args.K_re,
                                     temperature=args.temperature, max_tokens=args.max_tokens,
                                     timeout_s=args.timeout, minted_date=MINTED_DATE, member_key=f"m{s}")
            members.append(res)
            print(f"  member m{s}: A1_pool={res.a1_pool_pass} B_sel={res.b_selected_pass} "
                  f"ST_sel={res.st_selected_pass}", flush=True)
        verdict = amortization_verdict_v1(fam, disc.discovered, members)
        print(f"  VERDICT {fam}: A1={verdict.a1_solved} B={verdict.b_solved} ST={verdict.st_solved} "
              f"/{verdict.n_members}  ST-B={verdict.st_minus_b:+d} earned={verdict.earned} "
              f"non_neg={verdict.non_negative}", flush=True)
        results.append({"family": fam, "knob": args.knob, "discover": disc.to_dict(),
                        "members": [m.to_dict() for m in members], "verdict": verdict.to_dict()})

    n_earned = sum(1 for r in results if r["verdict"]["earned"])
    fams_earned = [r["family"] for r in results if r["verdict"]["earned"]]
    span_modes = _modes(fams_earned)
    summary = {
        "schema": "coordpy.w141_self_tutoring_dev_probe.v1", "model": MODEL, "knob": args.knob,
        "K": args.K, "K_re": args.K_re, "members": args.members, "mock": args.mock,
        "families": fams, "n_families_earned": n_earned, "families_earned": fams_earned,
        "earned_modes": sorted(span_modes),
        "span_ok_>=3_families_OR_>=2_modes": bool(n_earned >= 3 or len(span_modes) >= 2),
        "results": results,
    }
    sidecar.close()
    out_path = args.out or os.path.join(out_dir, "w141_dev_probe.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n=== W141 DEV: {n_earned}/{len(results)} families earned (ST>B); "
          f"earned={fams_earned}; span_ok={summary['span_ok_>=3_families_OR_>=2_modes']} ===")
    print(f"-> {out_path}\n-> {sidecar_path}")
    return 0


_FUNC = set(FUNC_FACTORIES)


def _modes(fams_earned) -> set:
    return {("FUNC" if f in _FUNC else "COMPLEXITY") for f in fams_earned}


def _mock_gen(template):
    ref = template.minted.ref_source
    brute = template.minted.brute_source
    naive = template.minted.naive_source
    st = {"n": 0}

    def gen(prompt: str, max_tokens: int, temperature: float):
        st["n"] += 1
        pl = prompt.lower()
        if "obviously-correct" in pl:
            return (f"```python\n{brute}\n```", 1)
        if "blank" in pl or "skeleton" in pl:
            return (f"```python\n{ref}\n```", 1)
        if "print" in pl and "=====" in prompt:
            return ("```python\nprint('1')\nprint('=====')\nprint('1')\n```", 1)
        return (f"```python\n{ref if st['n'] % 3 == 0 else brute}\n```", 1)

    return gen


if __name__ == "__main__":
    raise SystemExit(main())
