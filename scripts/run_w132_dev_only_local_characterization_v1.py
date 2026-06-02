#!/usr/bin/env python3
"""W132-β DEV_ONLY — local-Ollama instrument characterization on the minted battlefield.

STRICTLY DEV_ONLY / NOT-A-FRONTIER-CLAIM.  The Maverick frontier pilot
(``run_w132_calibration_and_pilot_v1.py``) is the primary β; when the NVIDIA NIM endpoint
is unreachable (as it was this session — even a 16-token Maverick call timed out at 90s),
this driver runs the SAME validated A0/A1/B bench + the SAME exact-oracle grader on the
minted core slice using a LOCAL Ollama code model, to (1) prove the β pipeline executes
end-to-end on the minted field and (2) characterize the minted instrument's difficulty.

It is legitimate as a RESISTANT test even with an UNKNOWN-cutoff local model BECAUSE the
battlefield is resistant *by construction* (fresh instances) — the W132 γ payoff that the
official-ICPC inherited battlefields could not claim.  But ``qwen2.5-coder:7b`` is far
weaker than Maverick, so the result characterizes the INSTRUMENT + the mechanism on a weak
local model — it is NOT the frontier claim and CANNOT retire anything.  Writes
results/w132/dev_only_local/ with every field tagged ``frontier_claim=false``.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import sys  # noqa: E402
sys.path.insert(0, str(ROOT))

from coordpy.resistant_by_construction_battlefield_v1 import (  # noqa: E402
    certify_resistance_v1, core_slice_cid_v1, mint_battlefield_v1, select_core_slice_v1)
from coordpy.resistant_by_construction_slate_v1 import RBC_SLATE_V1  # noqa: E402
from coordpy.icpc_reflexion_bench_v1 import (  # noqa: E402
    IcpcBenchConfigV1, run_icpc_reflexion_bench_v1)
from coordpy.coordpy_icpc_battlefield_v1 import ICPC_BATTLEFIELD_LISTING_SNAPSHOT_V1  # noqa: E402
from scripts.run_w108_livecodebench_pilot import _evaluate_phase2_gates, _mlb_rates  # noqa: E402

MINTED_DATE = "2026-06-02"
GLOBAL_SEED = 132
EXEC_TIMEOUT_S = 8.0
OFFICIAL_IDENTITIES = tuple(sorted({row[1] for row in ICPC_BATTLEFIELD_LISTING_SNAPSHOT_V1}))
MODE_ORDER = ("COMPLEXITY_BLIND", "HIDDEN_EDGE_STATE_MISS", "SEARCH_ENUM",
              "WRONG_ALGORITHM_ADMISSIBLE")
OLLAMA_URL = "http://localhost:11434/api/chat"


def _build_ollama_gen(model, sidecar_writer=None, max_retries=3):
    def _gen(prompt, max_tokens, temperature):
        body = {"model": str(model),
                "messages": [{"role": "user", "content": str(prompt)}],
                "stream": False,
                "options": {"temperature": float(temperature),
                            "num_predict": int(max_tokens)}}
        data = json.dumps(body).encode()
        last = None
        for _ in range(max_retries):
            t0 = time.time()
            try:
                req = urllib.request.Request(
                    OLLAMA_URL, data=data,
                    headers={"Content-Type": "application/json"}, method="POST")
                with urllib.request.urlopen(req, timeout=300) as r:
                    payload = json.loads(r.read().decode("utf-8", "replace"))
                text = str((payload.get("message") or {}).get("content") or "")
                wall = int((time.time() - t0) * 1000)
                if sidecar_writer is not None:
                    sidecar_writer({"model_id": str(model), "backend": "ollama",
                                    "prompt_len": len(prompt), "response_len": len(text),
                                    "temperature": float(temperature),
                                    "max_tokens": int(max_tokens), "wall_ms": wall,
                                    "response_text": text})
                return text, wall
            except Exception as e:  # noqa: BLE001
                last = e
        raise RuntimeError(f"ollama gen failed: {last}")
    return _gen


def _mode_span_slice(core, n):
    by = {m: [p for p in core if p.mode == m] for m in MODE_ORDER}
    picks, i = [], 0
    while len(picks) < n and any(by.values()):
        m = MODE_ORDER[i % len(MODE_ORDER)]
        if by[m]:
            picks.append(by[m].pop(0))
        i += 1
    return picks


def main() -> int:
    ap = argparse.ArgumentParser(description="W132 DEV_ONLY local characterization")
    ap.add_argument("--model", default="qwen2.5-coder:7b")
    ap.add_argument("--n", type=int, default=6, help="mode-spanning #problems")
    ap.add_argument("--seed", type=int, default=132_001)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "w132" / "dev_only_local"))
    args = ap.parse_args()

    bf = mint_battlefield_v1(RBC_SLATE_V1, global_seed=GLOBAL_SEED,
                             minted_date=MINTED_DATE, timeout_s=EXEC_TIMEOUT_S,
                             official_identities=OFFICIAL_IDENTITIES)
    cert = certify_resistance_v1(model_id=str(args.model), minted_date=MINTED_DATE,
                                 n_core=bf.manifest.n_admitted, raw_cid=bf.manifest.raw_cid)
    core = select_core_slice_v1(bf, n_problems=30)
    cid = core_slice_cid_v1(core)
    run = _mode_span_slice(core, int(args.n))
    subset = [p.to_pilot_problem(minted_date=MINTED_DATE) for p in run]
    modes = {}
    for p in run:
        modes[p.mode] = modes.get(p.mode, 0) + 1
    print(f"  DEV_ONLY local: model={args.model} n={len(subset)} modes={modes} "
          f"core_cid={cid[:16]}…")
    print(f"  resistance(by construction, model-agnostic)={cert.resistant}")

    run_id = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_dir) / f"w132_devonly_{args.model.replace(':','_')}_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    sc = open(out_dir / "reflexion_calls.jsonl", "w")
    gen = _build_ollama_gen(args.model, sidecar_writer=lambda r: (
        sc.write(json.dumps(r, separators=(",", ":")) + "\n"), sc.flush()))
    cfg = IcpcBenchConfigV1(K_multi_sample=5, seeds=(int(args.seed),),
                            sampling_temperature=0.7,
                            max_tokens_per_call=int(args.max_tokens),
                            executor_timeout_s=EXEC_TIMEOUT_S)
    t0 = time.time()
    report = run_icpc_reflexion_bench_v1(
        gen=gen, model_id=str(args.model), subset=subset, config=cfg,
        on_problem_start=lambda s, i, t: print(
            f"  p_idx={i+1}/{len(subset)} qid={t}", flush=True))
    sc.close()
    wall = time.time() - t0
    mlb = _mlb_rates(report)
    gates = _evaluate_phase2_gates(report=report, mlb=mlb)
    a1 = float(report.a1_mean_pass_at_1)
    non_degen = bool(0.0 < a1 < 0.90 and mlb["n_b_invoked_reflexion"] >= 1)

    rep = report.to_dict()
    rep.update({
        "DEV_ONLY": True, "frontier_claim": False, "can_retire": False,
        "backend": "ollama", "wall_s": round(wall, 2),
        "minted_date": MINTED_DATE, "core_slice_cid": cid,
        "resistant_by_construction": cert.resistant,
        "slice_problem_ids": [p.problem_id for p in run], "slice_modes": modes,
        "mlb": mlb, "phase2_evaluation": gates,
        "non_degenerate": non_degen,
        "note": ("DEV_ONLY local-model instrument characterization; qwen2.5-coder:7b is "
                 "far weaker than the Maverick frontier target and this CANNOT retire "
                 "anything. The Maverick frontier pilot is NIM-infra-blocked this session "
                 "(push-button re-run). Resistance holds by construction for this "
                 "UNKNOWN-cutoff local model too (the W132 gamma payoff)."),
    })
    (out_dir / "dev_only_report.json").write_text(json.dumps(rep, indent=2, default=str))
    (Path(args.out_dir) / "latest.txt").write_text(out_dir.name + "\n")
    print()
    print(f"  [DEV_ONLY] WALL {wall:.0f}s; A0={report.a0_mean_pass_at_1*100:.1f}% "
          f"A1={a1*100:.1f}% B={report.b_mean_pass_at_1*100:.1f}% "
          f"B-A1={report.b_mean_minus_a1_mean_pp:+.2f}pp")
    print(f"  [DEV_ONLY] MLB-1 {mlb['mlb1_invocation_rate']*100:.0f}% "
          f"({mlb['n_b_invoked_reflexion']}/{mlb['n_problems_total']}); "
          f"MLB-2 {mlb['mlb2_rescue_rate']*100:.0f}% "
          f"({mlb['n_b_rescued_via_reflexion']}/{mlb['n_b_invoked_reflexion']})")
    print(f"  [DEV_ONLY] non_degenerate={non_degen}; verdict={gates['verdict_label']} "
          f"(DEV_ONLY — NOT a frontier/retirement claim)")
    print(f"  out_dir: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
