"""W129 Lane α diagnostic — dump the exact pawnshop A0/B1 under-determination ($0 NIM).

Prints the statement, public samples, model-derived counterexamples, and the A0 (wrong) vs
B1 (correct) candidate code + their outputs on samples/derived cases, so we can judge whether
ANY public-signal oracle (executable invariants, harder auto-derived cases) could separate them.
"""
from __future__ import annotations
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import coordpy.family_scaffold_generation_v1 as G  # noqa: E402
import coordpy.role_diverse_algorithm_search_v1 as R  # noqa: E402
from scripts.run_w129_stored_pool_recon_v1 import (  # noqa: E402
    make_replay_gen, reconstruct_target, _public_pass, _secret_pass)

CALLS = os.path.join(ROOT, "results/w128/dev_bench/"
                     "w128_dev_bench_20260601T185815Z_fixed/dev_bench_calls.jsonl")


def main() -> int:
    probs = G.load_exposed_problems_v1("/tmp/w121_icpc")
    by_name = {p.short_name: p for p in probs}
    gen = make_replay_gen([CALLS])
    for name in ("pawnshop", "blueberrywaffle"):
        ep = by_name[name]
        prob, arts, impls = reconstruct_target(gen, ep)
        print("=" * 78)
        print(f"### {name}  (family-derived)")
        print("--- STATEMENT (first 1400 chars) ---")
        print(ep.statement[:1400])
        print("--- PUBLIC SAMPLES ---")
        for i, (inp, exp) in enumerate(prob.samples):
            print(f"  sample{i+1} IN={inp!r} EXP={exp!r}")
        print(f"--- INVARIANTS ({len(arts.invariants)}) ---")
        for iv in arts.invariants:
            print("  -", iv)
        print(f"--- DERIVED COUNTEREXAMPLES ({len(arts.counterexamples)}) ---")
        for j, (cin, cexp) in enumerate(arts.counterexamples):
            print(f"  cx{j+1} IN={cin!r} PRED_EXP={cexp!r}")
        for im in impls:
            pub = _public_pass(prob, im.code) if im.parses else False
            sec = _secret_pass(prob, im.code) if im.parses else False
            tag = "CORRECT" if sec else ("PUB_OK_SECRET_FAIL" if pub else "PUB_FAIL")
            print(f"--- CANDIDATE {im.label}  [{tag}] parses={im.parses} ---")
            if im.label in (("A0", "B1") if name == "pawnshop" else ("A0", "B1", "C2")):
                print(im.code[:1100])
                # outputs on derived
                outs = []
                for cin, _e in arts.counterexamples:
                    out, _d = R._run_capture_stdout_v1(im.code, cin, timeout_s=5.0)
                    outs.append(out)
                print(f"   derived-outputs: {outs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
