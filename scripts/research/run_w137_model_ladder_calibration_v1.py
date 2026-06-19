#!/usr/bin/env python3
"""W137 Lane α — model-ladder hardness calibration (HC3 + HC4).

Runs single-shot A0 across the locked ladder (small `meta/llama-3.1-8b-instruct` + strong anchor
`meta/llama-3.3-70b-instruct`) + a small strong-anchor A1, then admits the headroom band per
RUNBOOK_W137 §6.  Writes ``results/w137/w137_calibration_v1.json`` with the surviving template
names + calibration_cid (locked before any Lane-β / corpus admission spend).

Run:  .venv/bin/python scripts/run_w137_model_ladder_calibration_v1.py [--n-a0 3] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coordpy.hard_battlefield_slate_v2 import build_hard_slate_v2, slate_fingerprint_cid_v1  # noqa: E402
from coordpy.model_ladder_calibration_v1 import LADDER_V1, run_calibration_v1  # noqa: E402
from scripts.run_w108_livecodebench_pilot import _build_nim_gen  # noqa: E402

LOCKED_SLATE_CID = "2ce207c567324e4322f308e58a1fc2c88a8d4bdd0e340d2ec8a1b867d82b3f70"
OUT = "results/w137/w137_calibration_v1.json"
CALLS = "results/w137/w137_calibration_calls.jsonl"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-a0", type=int, default=3)
    ap.add_argument("--n-a1", type=int, default=1)
    ap.add_argument("--k-a1", type=int, default=3)
    ap.add_argument("--hc3-ceiling", type=float, default=0.80)
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--mint-timeout", type=float, default=3.0, help="mint timeout (grading stays 8.0)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N templates (probe)")
    args = ap.parse_args()

    assert slate_fingerprint_cid_v1() == LOCKED_SLATE_CID, "SLATE DRIFT — refusing to spend NIM"

    os.makedirs("results/w137", exist_ok=True)
    calls_fp = open(CALLS, "a")

    def sidecar(rec: dict) -> None:
        calls_fp.write(json.dumps(rec) + "\n")
        calls_fp.flush()

    _gen_cache: dict = {}

    def gen_for_model(model_id: str):
        if model_id not in _gen_cache:
            _gen_cache[model_id] = _build_nim_gen(model=model_id, sidecar_writer=sidecar)
        return _gen_cache[model_id]

    templates = build_hard_slate_v2()
    if args.limit > 0:
        templates = templates[:args.limit]

    n_calls = {"n": 0}

    def on_call(tpl: str, model: str, idx: int) -> None:
        n_calls["n"] += 1
        print(f"  [{n_calls['n']}] {tpl[:30]:30s} {model.split('/')[-1]:26s} inst{idx}", flush=True)

    t0 = time.time()
    print(f"calibrating {len(templates)} templates on ladder {[m.model_id for m in LADDER_V1]}")
    rep = run_calibration_v1(
        gen_for_model=gen_for_model, templates=templates, ladder=LADDER_V1,
        n_a0=args.n_a0, n_a1=args.n_a1, K_a1=args.k_a1, hc3_ceiling=args.hc3_ceiling,
        max_tokens=args.max_tokens, mint_timeout_s=args.mint_timeout, on_call=on_call)
    wall = time.time() - t0
    calls_fp.close()

    out = rep.to_dict()
    out["slate_fingerprint_cid"] = slate_fingerprint_cid_v1()
    out["wall_s"] = round(wall, 1)
    out["n_a0"] = args.n_a0
    out["n_a1"] = args.n_a1
    out["k_a1"] = args.k_a1
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)

    print("\n=== CALIBRATION RESULT ===")
    print(f"{'template':34s} {'mode':16s} {'small_a0':>9s} {'strong_a0':>10s} {'strong_a1':>10s} {'verdict':>10s}")
    for t in rep.per_template:
        sm = next((m for m in t.per_model if m.tier == "small"), None)
        st = next((m for m in t.per_model if m.tier == "strong"), None)
        print(f"{t.template_name:34s} {t.mode[:16]:16s} {sm.a0_rate if sm else 0:9.2f} "
              f"{st.a0_rate if st else 0:10.2f} {st.a1_rate if st else 0:10.2f} "
              f"{'ADMIT' if t.admitted else 'cull':>10s}  {t.reason}")
    surv = rep.surviving_template_names()
    print(f"\nsurviving (headroom band): {len(surv)}/{len(rep.per_template)}")
    print(f"  {list(surv)}")
    print(f"calibration_cid: {rep.calibration_cid()}")
    print(f"wall: {wall:.0f}s  calls: {n_calls['n']} (+A1)  -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
