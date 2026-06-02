"""W131 Lane α — code-competent model SUPPLY census (NIM-free; local smoke is $0).

Builds the three-surface reachability/capability matrix (local-HF / local-Ollama / hosted-NIM)
cross-cut with the stronger-model cutoff gate, runs the tiny same-family code smoke gate on the
reachable local code models, and identifies the best honest dev-bench candidate(s).  Emits
results/w131/census/model_supply_census_v1.json.  Hosted code smoke is deferred to the dev-bench
canary (the operator-greenlit NIM lane), so this script spends $0.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import coordpy.code_model_supply_census_v1 as C  # noqa: E402

OUT_DIR = os.path.join(ROOT, "results", "w131", "census")
DEFAULT_SMOKE = "qwen2.5-coder:32b,qwen2.5-coder:7b,lexi-coder:latest,deepseek-r1:7b"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ollama-base-url", default="http://localhost:11434")
    ap.add_argument("--no-smoke-local", action="store_true")
    ap.add_argument("--local-smoke-models", default=DEFAULT_SMOKE,
                    help="comma-separated local model names to smoke ('' = all reachable)")
    ap.add_argument("--local-smoke-timeout-s", type=float, default=180.0)
    args = ap.parse_args()

    smoke_models = ([m.strip() for m in args.local_smoke_models.split(",") if m.strip()]
                    if args.local_smoke_models else None)

    print("  building W131 code-model supply census (local smoke = $0; hosted = $0 GET) …",
          flush=True)
    census = C.build_census_v1(
        smoke_local=not args.no_smoke_local, local_smoke_models=smoke_models,
        ollama_base_url=args.ollama_base_url,
        local_smoke_timeout_s=args.local_smoke_timeout_s)

    d = census.to_dict()
    d["lane"] = "alpha_code_model_supply_census"
    d["verified_on"] = _dt.date.today().isoformat()
    d["census_cid"] = census.cid()
    d["resistant_instrument_frontier"] = C.RESISTANT_INSTRUMENT_FRONTIER
    d["nim_spend"] = 0
    d["generated_at"] = _dt.datetime.now(_dt.timezone.utc).isoformat()

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "model_supply_census_v1.json")
    with open(out, "w") as f:
        json.dump(d, f, indent=2, default=str)

    print()
    print(f"  === W131 MODEL SUPPLY CENSUS (cid {census.cid()[:16]}…) ===")
    print(f"  local-HF loadable: {census.local_hf_loadable}  ({census.local_hf_reason})")
    print(f"  ollama reachable:  {census.ollama_reachable}")
    print(f"  hosted-NIM reachable: {census.hosted_nim_reachable}  {census.hosted_blocked_reason}")
    print()
    print(f"  {'model_id':46s} {'path':12s} {'prior':11s} {'param':11s} "
          f"{'smoke':8s} {'cutoff':13s} {'usage':16s}")
    for r in census.records:
        sm = ("pass" if r.smoke_pass else "FAIL" if r.smoke_ran else "defer")
        print(f"  {r.model_id[:46]:46s} {r.access_path:12s} {r.code_prior:11s} "
              f"{r.param_hint[:11]:11s} {sm:8s} {r.cutoff_disclosure:13s} {r.usage_class:16s}")
    print()
    print(f"  best LOCAL dev candidate ($0):  {census.best_local_dev_candidate()}")
    print(f"  best HOSTED dev candidate (NIM): {census.best_hosted_dev_candidate()}")
    print(f"  FRONTIER_ELIGIBLE (resistant-honest): {census.frontier_eligible() or 'NONE'}")
    print(f"  DEV_ONLY count: {len(census.dev_only())}")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
