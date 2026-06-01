"""W129 Lane β0 — CHEAP SO3 verifier-final probe on the STORED W128 pools (RUNBOOK_W129 § 6).

The NIM-free recon proved `pawnshop` is public-signal UNDER-DETERMINED (hidden-correct B1 and
hidden-wrong A0 are byte-identical on all public signal).  The ONLY lever that can break a
2-way behavioural tie is an external correctness JUDGE.  β0 tests the strongest available one:
the SO3 verifier-final (a single model call that SEES every candidate + its public/derived
verdict + the invariants and CHOOSES one or ABSTAINs, public-signal only — NEVER the secret).

This reconstructs the stored pools by REPLAY (candidates already paid) and issues ONE fresh
verifier call per problem with >=2 public survivors (≤ ~12 NIM, EXPOSED dev — operator-greenlit).

β0 PASSES iff the verifier COMMITS pawnshop's B1 (cashes out the under-determined tie) AND keeps
blueberrywaffle + sunandmoon AND has 0 mis-commits.  β0 fail ⇒ the in-loop signal is
non-discriminating even for a model judge ⇒ register the under-determination cap; do NOT fire β1.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
import threading

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import coordpy.family_scaffold_generation_v1 as G  # noqa: E402
import coordpy.role_diverse_algorithm_search_v1 as R  # noqa: E402
import coordpy.public_signal_selection_oracle_v1 as S  # noqa: E402
from coordpy.icpc_reflexion_bench_v1 import grade_on_secret_v1  # noqa: E402
from scripts.run_w129_stored_pool_recon_v1 import (  # noqa: E402
    make_replay_gen, reconstruct_target, _public_pass)
from scripts.run_w127_exposed_dev_bench_v1 import _build_local_nim_gen  # noqa: E402

OUT_DIR = os.path.join(ROOT, "results", "w129", "dev_bench")
MODEL = "meta/llama-4-maverick-17b-128e-instruct"


def _secret_ok(prob, code, timeout_s=10.0):
    if not code or not code.strip():
        return False
    try:
        ok, _t, _n = grade_on_secret_v1(prob, code, timeout_s=timeout_s)
        return bool(ok)
    except Exception:  # noqa: BLE001
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exposed-root", default="/tmp/w121_icpc")
    ap.add_argument("--calls", default="results/w128/dev_bench/"
                    "w128_dev_bench_20260601T185815Z_fixed/dev_bench_calls.jsonl")
    ap.add_argument("--families", default=",".join(R.NON_SCAFFOLDABLE_FAMILIES))
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--verifier-temp", type=float, default=0.0)
    ap.add_argument("--timeout-s", type=float, default=10.0)
    ap.add_argument("--read-timeout-s", type=float, default=120.0)
    args = ap.parse_args()

    hard_families = tuple(f.strip() for f in args.families.split(",") if f.strip())
    probs = G.load_exposed_problems_v1(args.exposed_root)
    fam_of = {p.short_name: G.target_family_ranking_v1(p.statement, p.samples).family
              for p in probs}
    dev = sorted([p for p in probs if fam_of[p.short_name] in hard_families],
                 key=lambda p: p.short_name)
    replay = make_replay_gen([os.path.join(ROOT, args.calls)])

    run_id = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_run = os.path.join(OUT_DIR, f"w129_so3_stored_probe_{run_id}")
    os.makedirs(out_run, exist_ok=True)
    sidecar = open(os.path.join(out_run, "verifier_calls.jsonl"), "w")
    lock = threading.Lock()

    def sidecar_writer(rec):
        with lock:
            sidecar.write(json.dumps(rec, separators=(",", ":")) + "\n")
            sidecar.flush()

    nim_gen = _build_local_nim_gen(model=MODEL, sidecar_writer=sidecar_writer,
                                   read_timeout_s=args.read_timeout_s)
    nim_calls = [0]

    def counting_gen(prompt, mt, temp):
        nim_calls[0] += 1
        return nim_gen(prompt, mt, temp)

    per_target = []
    for ep in dev:
        prob, arts, impls = reconstruct_target(replay, ep)
        survivors = R._public_survivors(prob, impls, timeout_s=args.timeout_s)
        # SO3 verifier-final (fires a real call only when >=2 survivors)
        so3 = S.select_so_v1(prob, impls, arts, variant="SO3", gen=counting_gen,
                             max_tokens=args.max_tokens, verifier_temp=args.verifier_temp,
                             timeout_s=args.timeout_s, seed_tag=ep.short_name)
        committed_pass = bool(so3.committed_code) and _secret_ok(prob, so3.committed_code,
                                                                 timeout_s=args.timeout_s)
        rec = {"short_name": ep.short_name, "family": fam_of[ep.short_name],
               "n_public_survivors": len(survivors),
               "committed_label": so3.committed_label, "abstained": so3.abstained,
               "branch": so3.branch, "committed_pass": committed_pass,
               "mis_commit": bool(so3.committed_code) and not committed_pass,
               "verifier_detail": so3.detail}
        per_target.append(rec)
        mk = "P" if committed_pass else ("X" if rec["mis_commit"] else "~")
        print(f"    {ep.short_name:22s} surv={len(survivors)} SO3={mk}"
              f"({so3.committed_label or 'abstain'}) branch={so3.branch}", flush=True)
    sidecar.close()

    def _g(name):
        return next(r for r in per_target if r["short_name"] == name)
    pawn, blue, sun = _g("pawnshop"), _g("blueberrywaffle"), _g("sunandmoon")
    pawn_cashed = (pawn["committed_label"] == "B1" and pawn["committed_pass"])
    pawn_no_miscommit = not pawn["mis_commit"]
    keep_blue = blue["committed_pass"]
    keep_sun = sun["committed_pass"]
    total_miscommit = sum(1 for r in per_target if r["mis_commit"])
    committed = sum(1 for r in per_target if r["committed_pass"])
    b0_pass = pawn_cashed and keep_blue and keep_sun and total_miscommit == 0

    verdict = {
        "schema": "coordpy.w129_so3_stored_probe.v1", "lane": "beta0_so3_stored_probe",
        "verified_on": _dt.date.today().isoformat(), "model_id": MODEL,
        "nim_calls": nim_calls[0], "n_targets": len(per_target),
        "committed_pass": committed, "mis_commits": total_miscommit,
        "pawnshop_cashed_out_B1": pawn_cashed, "pawnshop_no_miscommit": pawn_no_miscommit,
        "keep_blueberrywaffle": keep_blue, "keep_sunandmoon": keep_sun,
        "b0_pass": b0_pass,
        "verdict_label": ("SO3_VERIFIER_BREAKS_UNDER_DETERMINED_TIE" if b0_pass
                          else "SO3_VERIFIER_CANNOT_BREAK_UNDER_DETERMINED_TIE"),
        "regression_pair": {"pawnshop": pawn, "blueberrywaffle": blue, "sunandmoon": sun},
        "per_target": per_target,
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    with open(os.path.join(out_run, "so3_stored_probe_verdict.json"), "w") as f:
        json.dump(verdict, f, indent=2, default=str)
    with open(os.path.join(OUT_DIR, "so3_stored_probe_verdict.json"), "w") as f:
        json.dump(verdict, f, indent=2, default=str)

    print()
    print(f"  SO3 stored probe: committed {committed}/{len(per_target)}  "
          f"mis_commits={total_miscommit}  nim_calls={nim_calls[0]}")
    print(f"  pawnshop: label={pawn['committed_label']} pass={pawn['committed_pass']} "
          f"(cashed_out_B1={pawn_cashed})  blue_keep={keep_blue}  sun_keep={keep_sun}")
    print(f"  β0 VERDICT: {verdict['verdict_label']} (b0_pass={b0_pass})")
    print(f"  -> {out_run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
