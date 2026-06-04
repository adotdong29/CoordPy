#!/usr/bin/env python3
"""W136 — STANDARD-I/O confirmation (the bulletproof headline: the 'ceiling' vanishes with standard I/O).

If the 3 capability-bound traps fail ONLY because of the whitespace-flattened input format, then
presenting the SAME problems with the STANDARD one-structure-per-line format and running A0 (single-shot,
NO feedback, NO mechanism) should make the model one-shot them.  The expected outputs are unchanged (the
reference reads ``sys.stdin.read().split()`` and is format-agnostic), so reformatting only the INPUT
presentation is a faithful, leakage-free transform.

    python scripts/run_w136_standard_io_confirm_v1.py        # ~3 NIM (3 traps x A0)
"""
from __future__ import annotations

import datetime as _dt
import dataclasses
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from coordpy.noncomplexity_structure_corpus_v1 import (  # noqa: E402
    MINTED_DATE as W135_MINTED_DATE, load_corpus_v1, select_dev_bench_slice_v1,
)
from coordpy.icpc_reflexion_bench_v1 import (  # noqa: E402
    _initial_prompt, extract_candidate_code_v1, grade_on_secret_v1,
)
from scripts.run_w108_livecodebench_pilot import _build_nim_gen  # noqa: E402

MODEL = "meta/llama-3.3-70b-instruct"
OUT = ROOT / "results" / "w136" / "standard_io"


def _reformat(base: str, inp: str) -> str:
    """Reformat a whitespace-flattened trap input to the STANDARD one-structure-per-line layout.
    (Header on line 1; then one pair/triple/grid-row per line.)  Output is the same token multiset
    re-line-wrapped, so the format-agnostic reference yields the identical answer."""
    toks = inp.split()
    if base == "rbc_wa_knapsack_01":               # "N W" then N (w v) pairs
        n, w = toks[0], toks[1]; rest = toks[2:]
        rows = [f"{rest[2 * i]} {rest[2 * i + 1]}" for i in range(n_int(n))]
        return f"{n} {w}\n" + "\n".join(rows)
    if base == "rbc_wa_weighted_interval_scheduling":   # "N" then N (s e w) triples
        n = toks[0]; rest = toks[1:]
        rows = [f"{rest[3 * i]} {rest[3 * i + 1]} {rest[3 * i + 2]}" for i in range(n_int(n))]
        return f"{n}\n" + "\n".join(rows)
    if base == "rbc_se_lattice_paths_blocked":      # "R C" then R grid-row strings
        r, c = toks[0], toks[1]; rows = toks[2:2 + n_int(r)]
        return f"{r} {c}\n" + "\n".join(rows)
    return inp


def n_int(s):
    return int(s)


def main() -> int:
    w135 = load_corpus_v1(ROOT / "results" / "w135" / "corpus" / "corpus_cache.pkl")
    by = {p.problem_id.split("__")[0]: p for p in select_dev_bench_slice_v1(w135.dev, per_family=1)}
    traps = ["rbc_se_lattice_paths_blocked", "rbc_wa_knapsack_01", "rbc_wa_weighted_interval_scheduling"]
    OUT.mkdir(parents=True, exist_ok=True)
    run_id = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = OUT / f"stdio_{run_id}"; out_dir.mkdir(parents=True, exist_ok=True)
    sc = open(out_dir / "calls.jsonl", "w")
    gen = _build_nim_gen(model=MODEL, sidecar_writer=lambda r: (sc.write(json.dumps(r) + "\n"), sc.flush()))
    print(f"  STANDARD-I/O confirmation: A0 single-shot on {len(traps)} reformatted traps, model={MODEL}")
    t0 = time.time()
    results = {}
    for base in traps:
        p = by[base]
        # build a STANDARD-I/O variant: reformat sample + secret INPUTS; expected outputs unchanged
        new_samples = tuple((_reformat(base, i), o) for i, o in p.samples)
        new_secret = tuple((_reformat(base, i), o) for i, o in p.secret_cases)
        p2 = dataclasses.replace(p, samples=new_samples, secret_cases=new_secret)
        pilot = p2.to_pilot_problem(minted_date=W135_MINTED_DATE)
        # sanity: the stored ref still matches the reformatted cases (format-agnostic)
        ref_ok = grade_on_secret_v1(pilot, p.ref_source, timeout_s=8.0)[0]
        text, _ = gen(_initial_prompt(pilot), 1536, 0.7)     # A0: single shot, NO feedback
        code = extract_candidate_code_v1(response_text=text)
        passed, _, _ = grade_on_secret_v1(pilot, code, timeout_s=8.0)
        results[base] = {"mode": p.mode, "ref_still_passes_reformatted": bool(ref_ok),
                         "A0_single_shot_passed": bool(passed)}
        print(f"    [{base.replace('rbc_','')}] ref_ok={ref_ok}  A0_single_shot_passed={passed}", flush=True)
    sc.close()
    one_shot = [b for b in traps if results[b]["A0_single_shot_passed"]]
    payload = {"schema": "coordpy.w136_standard_io_confirm_v1", "model_id": MODEL,
               "wall_s": round(time.time() - t0, 2), "results": results,
               "traps_one_shot_with_standard_io": one_shot, "n_one_shot": len(one_shot),
               "note": ("A0 = single shot, NO feedback, NO mechanism. If the traps one-shot with standard "
                        "one-per-line I/O, the apparent 'wrong-algorithm ceiling' was PURELY the "
                        "whitespace-flattened input format — the model's algorithm was always correct.")}
    (out_dir / "standard_io_report.json").write_text(json.dumps(payload, indent=2, default=str))
    (OUT / "latest.txt").write_text(out_dir.name + "\n")
    print(f"\n  traps one-shot by A0 with STANDARD I/O (no feedback): {len(one_shot)}/3 -> {one_shot}")
    print(f"  out: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
