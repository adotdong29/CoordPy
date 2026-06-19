"""W129 Lane α — $0-NIM evaluation of the public-signal selection oracle on the STORED W128
hard-cluster pools (the regression-pair precursor test).

Reconstructs the 11 W128 dev-bench pools by replaying stored generations, then runs the
NIM-free SO1/SO2/SO4 (and SOLEAD without a verifier) selectors over each, grading the
committed candidate on the official SECRET cases.  Reports, per variant:
  committed X/11, abstains, and MIS-COMMITS (committed a hidden-WRONG candidate — the W128
  pawnshop failure; the key SAFETY metric a principled selector must drive to 0), plus the
  stored regression pair {blueberrywaffle (keep), pawnshop (B1 / abstain / A0), sunandmoon}.
$0 NIM.  SO3 (verifier-final) is NOT run here (needs fresh NIM; see the dev-bench script).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import coordpy.family_scaffold_generation_v1 as G  # noqa: E402
import coordpy.role_diverse_algorithm_search_v1 as R  # noqa: E402
import coordpy.public_signal_selection_oracle_v1 as S  # noqa: E402
from scripts.run_w129_stored_pool_recon_v1 import (  # noqa: E402
    make_replay_gen, reconstruct_target, _secret_pass)

OUT_DIR = os.path.join(ROOT, "results", "w129", "selector_stored")
NIMFREE_VARIANTS = ("SO1", "SO2", "SO4", "SOLEAD")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exposed-root", default="/tmp/w121_icpc")
    ap.add_argument("--calls", default="results/w128/dev_bench/"
                    "w128_dev_bench_20260601T185815Z_fixed/dev_bench_calls.jsonl")
    ap.add_argument("--w128-verdict", default="results/w128/dev_bench/"
                    "hard_cluster_dev_bench_verdict.json")
    ap.add_argument("--families", default=",".join(R.NON_SCAFFOLDABLE_FAMILIES))
    ap.add_argument("--timeout-s", type=float, default=10.0)
    args = ap.parse_args()

    w128 = json.load(open(os.path.join(ROOT, args.w128_verdict)))
    base_pass = {r["short_name"]: r["baseline_pass"] for r in w128["per_target"]}
    rda4_pass = {r["short_name"]: r["rda_committed_pass"] for r in w128["per_target"]}
    # seed SECRET grades from the $0 recon (every candidate already graded on secret there) —
    # the committed code is ALWAYS one of those candidates, so this is an exact O(1) lookup that
    # avoids re-running the slow O(N²) committed candidates on the official secret cases.
    recon_path = os.path.join(ROOT, "results/w129/recon/stored_pool_recon_v1.json")
    recon_secret: dict[str, bool] = {}
    if os.path.exists(recon_path):
        rj = json.load(open(recon_path))
        for t in rj.get("per_target", []):
            for c in t.get("candidates", []):
                recon_secret[c["code_sha"]] = bool(c["secret_pass"])

    hard_families = tuple(f.strip() for f in args.families.split(",") if f.strip())
    probs = G.load_exposed_problems_v1(args.exposed_root)
    fam_of = {p.short_name: G.target_family_ranking_v1(p.statement, p.samples).family
              for p in probs}
    dev = sorted([p for p in probs if fam_of[p.short_name] in hard_families],
                 key=lambda p: p.short_name)
    gen = make_replay_gen([os.path.join(ROOT, args.calls)])

    import hashlib as _hl
    per_target = []
    for ep in dev:
        prob, arts, impls = reconstruct_target(gen, ep)
        secret_cache: dict[str, bool] = {}  # per-problem: grade each unique committed code ONCE

        def _sec(code):
            if not code or not code.strip():
                return False
            sha16 = R._sha(code)[:16]
            if sha16 in recon_secret:        # exact recon lookup (no subprocess)
                return recon_secret[sha16]
            k = _hl.sha256(code.encode()).hexdigest()
            if k not in secret_cache:        # fallback (committed code not a recon candidate)
                secret_cache[k] = _secret_pass(prob, code, timeout_s=args.timeout_s)
            return secret_cache[k]

        sels = {}
        for v in NIMFREE_VARIANTS:
            sel = S.select_so_v1(prob, impls, arts, variant=v, timeout_s=args.timeout_s,
                                 seed_tag=ep.short_name)
            committed_pass = bool(sel.committed_code) and _sec(sel.committed_code)
            sels[v] = {"committed_label": sel.committed_label, "abstained": sel.abstained,
                       "branch": sel.branch, "evidence_used": sel.evidence_used,
                       "committed_pass": committed_pass,
                       "mis_commit": bool(sel.committed_code) and not committed_pass,
                       "n_public_survivors": sel.n_public_survivors,
                       "n_post_falsifier": sel.n_post_falsifier_survivors}
        per_target.append({"short_name": ep.short_name, "family": fam_of[ep.short_name],
                           "baseline_pass": base_pass.get(ep.short_name),
                           "rda4_pass_w128": rda4_pass.get(ep.short_name), "sels": sels})
        bp = "B" if base_pass.get(ep.short_name) else "."
        line = f"    {ep.short_name:22s} base={bp} "
        for v in NIMFREE_VARIANTS:
            s = sels[v]
            mark = ("P" if s["committed_pass"] else ("X" if s["mis_commit"] else "~"))  # pass/miscommit/abstain
            line += f"{v}={mark}({s['committed_label'] or 'abst'}) "
        print(line)

    # aggregate
    agg = {}
    for v in NIMFREE_VARIANTS:
        committed = sum(1 for r in per_target if r["sels"][v]["committed_pass"])
        mis = sum(1 for r in per_target if r["sels"][v]["mis_commit"])
        abst = sum(1 for r in per_target if r["sels"][v]["abstained"])
        uniq = sum(1 for r in per_target if r["sels"][v]["committed_pass"]
                   and not r["baseline_pass"])
        regr = sum(1 for r in per_target if r["baseline_pass"]
                   and not r["sels"][v]["committed_pass"])
        agg[v] = {"committed_pass": committed, "mis_commits": mis, "abstains": abst,
                  "unique_vs_baseline": uniq, "regressions_vs_baseline": regr,
                  "net_vs_baseline": uniq - regr}

    def _pair(name):
        r = next(r for r in per_target if r["short_name"] == name)
        return {v: {"label": r["sels"][v]["committed_label"],
                    "pass": r["sels"][v]["committed_pass"],
                    "abstain": r["sels"][v]["abstained"],
                    "mis_commit": r["sels"][v]["mis_commit"]} for v in NIMFREE_VARIANTS}

    regression_pair = {"pawnshop": _pair("pawnshop"), "blueberrywaffle": _pair("blueberrywaffle"),
                       "sunandmoon": _pair("sunandmoon")}
    control = S.fake_selection_control_v1()
    trust_exam = S.examine_trust_machinery_applicability_v1()

    out = {
        "schema": "coordpy.w129_selector_stored_eval.v1", "lane": "alpha_selector_stored",
        "nim_spend": 0, "verified_on": _dt.date.today().isoformat(),
        "n_targets": len(per_target),
        "w128_baseline_pass": sum(1 for r in per_target if r["baseline_pass"]),
        "w128_rda4_pass": sum(1 for r in per_target if r["rda4_pass_w128"]),
        "w128_pool_pass": w128["earn_gate"]["rda_pool_total_pass"],
        "aggregate": agg, "regression_pair": regression_pair,
        "fake_selection_control": control,
        "trust_machinery_examination": trust_exam,
        "per_target": per_target,
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "stored_selector_eval_v1.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)

    print()
    print(f"  W128: baseline {out['w128_baseline_pass']}/11  RDA4 {out['w128_rda4_pass']}/11  "
          f"pool {out['w128_pool_pass']}/11")
    for v in NIMFREE_VARIANTS:
        a = agg[v]
        print(f"  {v:7s} committed={a['committed_pass']}/11  mis_commits={a['mis_commits']}  "
              f"abstains={a['abstains']}  net_vs_base={a['net_vs_baseline']:+d}")
    print(f"  control_passes={control['control_passes']}  "
          f"substrate_bridge_killed={trust_exam['substrate_controller_literal_bridge_killed']}")
    print("  regression pair:")
    for k, v in regression_pair.items():
        print(f"    {k:16s} {v}")
    print(f"  -> {os.path.join(OUT_DIR, 'stored_selector_eval_v1.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
