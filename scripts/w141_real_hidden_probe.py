#!/usr/bin/env python3
"""W141: grade the REAL 70B Calm samples against the REAL minted secret cases, to learn the
true ground-truth behavior (pass / TLE / wrong-answer) at the actual hidden N. $0 local."""
import sys, json, re, hashlib, time
from coordpy.headroom_band_slate_v3 import FUNC_FACTORIES, CX_FACTORIES, EXTRA_CX_FACTORIES
from coordpy.resistant_by_construction_battlefield_v1 import mint_problem_v1, _exec_capture_v1

JSONL = "results/w140/hard_lift/w140_hardlift_20260608T000336Z/calls_strong.jsonl"
PROBE_SEED_BASE = 140_950_000
def _sha(o): return hashlib.sha256(json.dumps(o, sort_keys=True).encode()).hexdigest()
def seed_for(fam, knob): return PROBE_SEED_BASE + int(_sha({"hf": fam, "k": knob})[:6], 16) % 100000

def extract(t):
    m = re.search(r"```(?:python)?\s*\n(.*?)```", t, re.DOTALL)
    return m.group(1) if m else t

def grade_program(src, prob, exec_t):
    """Return (verdict, per_case). verdict in {'pass','wrong','tle','error'}."""
    per = []
    final = 'pass'
    for inp, exp in prob.secret_cases:
        r = _exec_capture_v1(src, inp, timeout_s=exec_t)
        n = int(inp.split()[0])
        if r.timed_out:
            per.append((n, 'tle'));
            if final == 'pass': final = 'tle'
        elif r.returncode != 0:
            per.append((n, 'err'))
            if final in ('pass',): final = 'error'
        elif r.stdout.strip() == exp.strip():
            per.append((n, 'ok'))
        else:
            per.append((n, 'WRONG'))
            final = 'wrong'   # wrong dominates
    return final, per

def load_family_samples(title_sub):
    out = []
    with open(JSONL) as f:
        for gi, line in enumerate(f):
            d = json.loads(line); p = d['prompt']
            if title_sub not in p or 'reflective' in p:
                continue
            out.append((gi, extract(d['response_text']), d['temperature']))
    return out

CELLS = {
    'subarrays_sum_and_range': (30000, 'Calm blocks.', FUNC_FACTORIES),
    'sum_nearest_smaller_left': (50000, 'Nearest smaller to the left.', CX_FACTORIES),
    'kth_smallest_pair_distance': (20000, 'K-th smallest pair gap.', EXTRA_CX_FACTORIES),
    'max_j_minus_i_le': (50000, 'Widest non-decreasing index pair.', CX_FACTORIES),
}

if __name__ == '__main__':
    fam = sys.argv[1] if len(sys.argv) > 1 else 'subarrays_sum_and_range'
    exec_t = float(sys.argv[2]) if len(sys.argv) > 2 else 8.0
    knob, title, facs = CELLS[fam]
    tmpl = facs[fam](knob)
    base = seed_for(fam, knob)
    probs = [mint_problem_v1(tmpl.minted, global_seed=base + r, timeout_s=1.0) for r in range(4)]
    print(f"=== {fam}@{knob}  exec_timeout={exec_t}s  mode={tmpl.minted.mode} disc={tmpl.minted.discriminator}", flush=True)

    # reference programs: correct brute (O(N^2)) and the WRONG naive
    brute_src = tmpl.minted.brute_source
    naive_src = tmpl.minted.naive_source
    ref_src = tmpl.minted.ref_source
    for name, src in [('REF(efficient)', ref_src), ('BRUTE(correct O(N^2))', brute_src), ('NAIVE(trap)', naive_src)]:
        t0 = time.time()
        v, per = grade_program(src, probs[0], exec_t)
        print(f"  [{name:24s}] prob0 verdict={v:6s} per={per}  ({time.time()-t0:.1f}s)", flush=True)

    samples = load_family_samples(title)
    print(f"  loaded {len(samples)} baseline samples", flush=True)
    # grade each sample against ALL 4 problems; a1 = mean pass@1 across the 4 (or across instances).
    # For pass@1 with K samples: here samples come pre-generated; we report per-sample verdict on prob0
    # AND aggregate "fraction of (sample,problem) pairs that PASS".
    npass = ntle = nwrong = nerr = 0
    per_sample = []
    for gi, src, temp in samples:
        # grade on the matched problem instance set: each sample's prompt was one instance; but we don't
        # track which. Grade on prob0 as the canonical instance (all 4 are same family/knob, ~same N).
        v, per = grade_program(src, probs[0], exec_t)
        per_sample.append((gi, temp, v))
        if v == 'pass': npass += 1
        elif v == 'tle': ntle += 1
        elif v == 'wrong': nwrong += 1
        else: nerr += 1
        print(f"    idx={gi:3d} temp={temp} verdict={v:6s} {per}", flush=True)
    n = len(samples)
    print(f"\n  SUMMARY {fam}: n={n}  pass={npass} ({100*npass/n:.0f}%)  TLE={ntle}  WRONG={nwrong}  err={nerr}", flush=True)
    print(f"  => true a1(pass@1 on prob0) = {npass/n:.3f}", flush=True)
