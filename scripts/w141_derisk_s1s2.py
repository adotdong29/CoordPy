#!/usr/bin/env python3
"""W141 de-risk: test whether NO-ORACLE signals S1 (brute-agreement on small) + S2
(timeout on large) can separate correct-efficient samples from wrong/slow ones.

$0 / local-only. Uses the spec ref ONLY to assign ground-truth labels for SCORING the
signals; the mechanism (S1/S2) never sees ref.
"""
import json, re, subprocess, sys, tempfile, os, time, random, signal
from collections import defaultdict

JSONL = "results/w140/hard_lift/w140_hardlift_20260608T000336Z/calls_strong.jsonl"

# ----------------------------------------------------------------- spec sources (verbatim from coordpy/headroom_band_slate_v3.py)
CALM_REF = ("import sys\n"
       "from collections import deque\n"
       "d=sys.stdin.buffer.read().split()\n"
       "n=int(d[0]);S=int(d[1]);R=int(d[2]);a=[int(x) for x in d[3:3+n]]\n"
       "cur=0;l=0;cnt=0;mx=deque();mn=deque()\n"
       "for r in range(n):\n"
       "    cur+=a[r]\n"
       "    while mx and a[mx[-1]]<=a[r]:mx.pop()\n"
       "    mx.append(r)\n"
       "    while mn and a[mn[-1]]>=a[r]:mn.pop()\n"
       "    mn.append(r)\n"
       "    while l<=r and (cur>S or a[mx[0]]-a[mn[0]]>R):\n"
       "        cur-=a[l]\n"
       "        if mx[0]==l:mx.popleft()\n"
       "        if mn[0]==l:mn.popleft()\n"
       "        l+=1\n"
       "    cnt+=r-l+1\n"
       "print(cnt)\n")

NSL_REF = ("import sys\n"
       "d=sys.stdin.buffer.read().split()\n"
       "n=int(d[0]);a=[int(x) for x in d[1:1+n]]\n"
       "st=[];total=0\n"
       "for v in a:\n"
       "    while st and st[-1]>=v:st.pop()\n"
       "    total+=st[-1] if st else -1\n"
       "    st.append(v)\n"
       "print(total)\n")

KTH_REF = ("import sys\n"
       "d=sys.stdin.buffer.read().split()\n"
       "n=int(d[0]);K=int(d[1]);a=[int(x) for x in d[2:2+n]]\n"
       "a.sort()\n"
       "def cnt_le(dd):\n"
       "    c=0;l=0\n"
       "    for r in range(n):\n"
       "        while a[r]-a[l]>dd:l+=1\n"
       "        c+=r-l\n"
       "    return c\n"
       "lo=0;hi=a[-1]-a[0]\n"
       "while lo<hi:\n"
       "    mid=(lo+hi)//2\n"
       "    if cnt_le(mid)>=K:hi=mid\n"
       "    else:lo=mid+1\n"
       "print(lo)\n")

# the TEAM's brute for Calm = the spec brute (correct O(N^2) double loop). S1 compares against THIS.
CALM_BRUTE = ("import sys\n"
       "d=sys.stdin.buffer.read().split()\n"
       "n=int(d[0]);S=int(d[1]);R=int(d[2]);a=[int(x) for x in d[3:3+n]]\n"
       "cnt=0\n"
       "for i in range(n):\n"
       "    s=0;mx=-1;mn=10**18\n"
       "    for j in range(i,n):\n"
       "        s+=a[j]\n"
       "        if a[j]>mx:mx=a[j]\n"
       "        if a[j]<mn:mn=a[j]\n"
       "        if s<=S and mx-mn<=R:cnt+=1\n"
       "print(cnt)\n")

WIDE_REF = ("import sys\n"
       "d=sys.stdin.buffer.read().split()\n"
       "n=int(d[0]);a=[int(x) for x in d[1:1+n]]\n"
       "lmin=[0]*n;rmax=[0]*n\n"
       "lmin[0]=a[0]\n"
       "for i in range(1,n):lmin[i]=lmin[i-1] if lmin[i-1]<a[i] else a[i]\n"
       "rmax[n-1]=a[n-1]\n"
       "for i in range(n-2,-1,-1):rmax[i]=rmax[i+1] if rmax[i+1]>a[i] else a[i]\n"
       "i=0;j=0;best=0\n"
       "while i<n and j<n:\n"
       "    if lmin[i]<=rmax[j]:\n"
       "        if j-i>best:best=j-i\n"
       "        j+=1\n"
       "    else:\n"
       "        i+=1\n"
       "print(best)\n")

# ----------------------------------------------------------------- input generators
# S1 small-input bank: must be DISCRIMINATING (i.e. force the dropped constraint / edge to bind).
# These are what the TEAM would generate from the STATEMENT alone (no hidden tests).

def gen_calm_small(rng, n):
    # mirror the hidden-style: values 0..200, R small (gap cap binds), S large (sum rarely binds)
    R = rng.randint(10, 40)
    a = [rng.randint(0, 200) for _ in range(n)]
    S = rng.randint(n*40, n*80)
    return f"{n} {S} {R}\n" + " ".join(map(str, a)) + "\n"

def gen_calm_small_PUBLICONLY(rng, n):
    # the WEAK bank: like the public gp() — entries within R of each other -> gap cap NEVER binds
    base = rng.randint(0, 50); R = rng.randint(5, 15)
    a = [base + rng.randint(0, R) for _ in range(n)]
    S = sum(a) + rng.randint(0, 20)
    return f"{n} {S} {R}\n" + " ".join(map(str, a)) + "\n"

def gen_nsl_small(rng, n):
    a = [rng.randint(1, 20) for _ in range(n)]
    return f"{n}\n" + " ".join(map(str, a)) + "\n"

def gen_kth_small(rng, n):
    a = [rng.randint(1, 30) for _ in range(n)]
    K = rng.randint(1, n*(n-1)//2)
    return f"{n} {K}\n" + " ".join(map(str, a)) + "\n"

def gen_wide_small(rng, n):
    a = [rng.randint(1, 20) for _ in range(n)]
    return f"{n}\n" + " ".join(map(str, a)) + "\n"

# S2 large-input: force the O(N^2) worst case
def gen_calm_large(rng, n):
    R = rng.randint(10, 40)
    a = [rng.randint(0, 200) for _ in range(n)]
    S = rng.randint(n*40, n*80)
    return f"{n} {S} {R}\n" + " ".join(map(str, a)) + "\n"

def gen_nsl_large(rng, n):
    # strictly decreasing -> worst case for naive inner scan
    return f"{n}\n" + " ".join(map(str, range(n, 0, -1))) + "\n"

def gen_kth_large(rng, n):
    a = [rng.randint(1, 10**9) for _ in range(n)]
    K = rng.randint(1, n*(n-1)//2)
    return f"{n} {K}\n" + " ".join(map(str, a)) + "\n"

def gen_wide_large(rng, n):
    return f"{n}\n" + " ".join(map(str, range(n, 0, -1))) + "\n"

# TIMEOUT-family brutes = the spec naive (O(N^2), CORRECT). S1 compares candidate vs this.
NSL_BRUTE = ("import sys\n"
       "d=sys.stdin.buffer.read().split()\n"
       "n=int(d[0]);a=[int(x) for x in d[1:1+n]]\n"
       "total=0\n"
       "for i in range(n):\n"
       "    v=a[i];found=-1\n"
       "    for j in range(i-1,-1,-1):\n"
       "        if a[j]<v:found=a[j];break\n"
       "    total+=found\n"
       "print(total)\n")
KTH_BRUTE = ("import sys\n"
       "d=sys.stdin.buffer.read().split()\n"
       "n=int(d[0]);K=int(d[1]);a=[int(x) for x in d[2:2+n]]\n"
       "ds=[]\n"
       "for i in range(n):\n"
       "    ai=a[i]\n"
       "    for j in range(i+1,n):\n"
       "        ds.append(ai-a[j] if ai>a[j] else a[j]-ai)\n"
       "ds.sort()\n"
       "print(ds[K-1])\n")
WIDE_BRUTE = ("import sys\n"
       "d=sys.stdin.buffer.read().split()\n"
       "n=int(d[0]);a=[int(x) for x in d[1:1+n]]\n"
       "best=0\n"
       "for i in range(n):\n"
       "    ai=a[i]\n"
       "    for j in range(n-1,i,-1):\n"
       "        if ai<=a[j]:\n"
       "            if j-i>best:best=j-i\n"
       "            break\n"
       "print(best)\n")

FAMILIES = {
    "Calm blocks.": dict(ref=CALM_REF, brute=CALM_BRUTE, small=gen_calm_small, small_weak=gen_calm_small_PUBLICONLY,
                          large=gen_calm_large, n_large=30000, n_corr=400),
    "Nearest smaller to the left.": dict(ref=NSL_REF, brute=NSL_BRUTE, small=gen_nsl_small, large=gen_nsl_large, n_large=50000, n_corr=600),
    "K-th smallest pair gap.": dict(ref=KTH_REF, brute=KTH_BRUTE, small=gen_kth_small, large=gen_kth_large, n_large=20000, n_corr=400),
    "Widest non-decreasing index pair.": dict(ref=WIDE_REF, brute=WIDE_BRUTE, small=gen_wide_small, large=gen_wide_large, n_large=50000, n_corr=600),
}

# ----------------------------------------------------------------- code extraction + runner
def extract_code(text):
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if m: return m.group(1)
    # fallback: maybe no closing fence
    m = re.search(r"```(?:python)?\s*\n(.*)", text, re.DOTALL)
    if m: return m.group(1)
    return text

def run_prog(src, stdin_data, timeout):
    """Run src as a python script with stdin; return (status, stdout). status in
    {'ok','timeout','error'}."""
    try:
        with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as f:
            f.write(src); path = f.name
        try:
            p = subprocess.run([sys.executable, path], input=stdin_data.encode(),
                               capture_output=True, timeout=timeout)
            if p.returncode != 0:
                return ('error', p.stderr.decode()[-200:])
            return ('ok', p.stdout.decode().strip())
        finally:
            os.unlink(path)
    except subprocess.TimeoutExpired:
        return ('timeout', '')
    except Exception as e:
        return ('error', str(e))

# ----------------------------------------------------------------- main per-family analysis
def analyze_family(title, samples, n_small=20, small_n_range=(4,9), s2_timeout=3.0, small_timeout=2.0,
                   seed=12345, use_weak_bank=False):
    fam = FAMILIES[title]
    ref = fam['ref']
    rng = random.Random(seed)
    gen_small = fam.get('small_weak') if (use_weak_bank and 'small_weak' in fam) else fam['small']
    gen_large = fam['large']; n_large = fam['n_large']

    # build the S1 small-input bank ONCE (shared across candidates), with ref answers
    bank = []
    for _ in range(n_small):
        n = rng.randint(*small_n_range)
        inp = gen_small(rng, n)
        st, out = run_prog(ref, inp, 5.0)
        assert st == 'ok', f"ref failed on small input: {st} {out}"
        bank.append((inp, out))
    # one large input + ref answer (ref must finish fast)
    large_inp = gen_large(rng, n_large)
    st, ref_large = run_prog(ref, large_inp, 10.0)
    assert st == 'ok', f"ref failed/slow on large: {st}"
    # one MEDIUM input where O(N^2) still finishes (~few sec) -> use to label correct-vs-wrong
    # for the SLOW samples (which would TLE on the real large input). Same distribution as large.
    # CORRECTNESS-ONLY oracle input: N small enough that EVEN an O(N^3) slicing brute finishes fast,
    # but large enough that a dropped-constraint / wrong-algo sample diverges from ref. Big-N value dist.
    n_corr = fam.get('n_corr', 400)
    corr_inp = gen_large(rng, n_corr)         # same (adversarial) distribution as the large input
    st, ref_corr = run_prog(ref, corr_inp, 15.0)
    assert st == 'ok', f"ref failed on corr input: {st}"

    # the TEAM's own brute (what S1 actually compares against — NOT ref). On valid small inputs the
    # team-brute and ref agree, so the S1 reference answers are computed from the team-brute here.
    team_brute = fam.get('brute', ref)
    bank_brute = []
    for inp, _refout in bank:
        st, bout = run_prog(team_brute, inp, 5.0)
        assert st == 'ok', f"team-brute failed on small input: {st}"
        bank_brute.append((inp, bout))

    results = []
    for idx, src in samples:
        # ---- GROUND TRUTH (ref-based; SCORING ONLY) ----
        # correctness on the small bank (vs ref) AND on the bigger correctness input (vs ref)
        gt_small = True
        for inp, exp in bank:
            stt, oo = run_prog(src, inp, small_timeout)
            if stt != 'ok' or oo != exp:
                gt_small = False; break
        stt_c, oo_c = run_prog(src, corr_inp, 15.0)   # generous: O(N^3) at N=400 is ~ms-to-sub-sec
        gt_corr = (stt_c == 'ok' and oo_c == ref_corr)
        is_correct = gt_small and gt_corr

        # TRUE efficiency: run large at the REAL S2 timeout
        stt_s2, oo_s2 = run_prog(src, large_inp, s2_timeout)
        truly_fast_under_s2 = (stt_s2 == 'ok')
        truly_correct_large = (oo_s2 == ref_large) if stt_s2 == 'ok' else gt_corr  # proxy if TLE

        if is_correct and truly_fast_under_s2:
            label = 'correct_efficient'
        elif is_correct and not truly_fast_under_s2:
            label = 'correct_slow'      # right answer, but TLEs (the O(N^2)/O(N^3) brute)
        else:
            label = 'wrong'             # diverges from ref on small bank or on the correctness input

        # ---- NO-ORACLE SIGNALS (mechanism: never sees ref) ----
        # S1: candidate AGREES with the TEAM's self-written brute on every small bank input.
        s1_pass = True
        for inp, bexp in bank_brute:
            stt, oo = run_prog(src, inp, small_timeout)
            if stt != 'ok' or oo != bexp:
                s1_pass = False; break
        # S2: candidate FINISHES on the large input within the wall-clock timeout.
        s2_pass = truly_fast_under_s2

        results.append(dict(idx=idx, label=label, s1=s1_pass, s2=s2_pass,
                            gt_small=gt_small, gt_corr=gt_corr, gt_large=truly_correct_large,
                            fast=truly_fast_under_s2, large_status=stt_s2))
        print(f"    [prog {title[:12]:12s}] idx={idx:3d} label={label:18s} S1={int(s1_pass)} S2={int(s2_pass)} largestat={stt_s2}", flush=True)
    return results, bank, ref_large

# ----------------------------------------------------------------- driver
def load_samples():
    by_title = defaultdict(list)  # title -> list of (global_idx, src) baseline-only
    by_title_all = defaultdict(list)
    with open(JSONL) as f:
        for i, line in enumerate(f):
            d = json.loads(line)
            p = d['prompt']
            m = re.search(r'Problem:\s*\n(.+?)\n', p)
            title = m.group(1).strip() if m else '??'
            src = extract_code(d['response_text'])
            is_baseline = 'reflective debugging loop' not in p
            by_title_all[title].append((i, src, is_baseline))
            if is_baseline:
                by_title[title].append((i, src))
    return by_title, by_title_all

if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'all'
    only = None
    if mode.startswith('only:'):
        only = mode.split(':',1)[1]
    by_title, by_all = load_samples()
    all_targets = ["Calm blocks.", "Nearest smaller to the left.", "K-th smallest pair gap.",
               "Widest non-decreasing index pair."]
    targets = [t for t in all_targets if (only is None or only.lower() in t.lower())]
    for title in targets:
        samples = by_title[title]
        if not samples:
            print(f"\n### {title}: NO baseline samples"); continue
        weak = (mode == 'weak' and title == "Calm blocks.")
        res, bank, refl = analyze_family(title, samples, use_weak_bank=weak)
        # tally
        from collections import Counter
        labc = Counter(r['label'] for r in res)
        n = len(res)
        ce = [r for r in res if r['label']=='correct_efficient']
        # S1^S2 identification
        ident = [r for r in res if r['s1'] and r['s2']]
        # false positives: passes S1^S2 but NOT correct_efficient
        fp = [r for r in ident if r['label'] != 'correct_efficient']
        # false negatives: is correct_efficient but does NOT pass S1^S2
        fn = [r for r in ce if not (r['s1'] and r['s2'])]
        tp = [r for r in ident if r['label']=='correct_efficient']
        print(f"\n### {title}  (n={n} baseline samples){'  [WEAK public-only S1 bank]' if weak else ''}")
        print(f"  label dist: {dict(labc)}")
        print(f"  S1-pass={sum(r['s1'] for r in res)}  S2-pass={sum(r['s2'] for r in res)}  S1^S2-pass={len(ident)}")
        print(f"  TP(correct_eff & S1^S2)={len(tp)}  FP(passes S1^S2 but not ce)={len(fp)}  FN(ce but fails S1^S2)={len(fn)}")
        print(f"  correct_efficient EXISTS to extract: {'YES n='+str(len(ce)) if ce else 'NO'}")
        if fp:
            print(f"  !! FALSE POSITIVES: idx={[r['idx'] for r in fp]} labels={[r['label'] for r in fp]}")
        if fn:
            print(f"  FN detail: idx={[r['idx'] for r in fn]} (s1={[r['s1'] for r in fn]} s2={[r['s2'] for r in fn]})")
        # per-sample table
        for r in sorted(res, key=lambda x:x['idx']):
            mark = ''
            if r['s1'] and r['s2'] and r['label']!='correct_efficient': mark=' <-FALSE_POS'
            if r['label']=='correct_efficient' and not(r['s1'] and r['s2']): mark=' <-FALSE_NEG'
            print(f"    idx={r['idx']:3d} label={r['label']:18s} S1={int(r['s1'])} S2={int(r['s2'])} gtSmall={int(r['gt_small'])} gtCorr={int(r['gt_corr'])} fast={int(r['fast'])} ({r['large_status']}){mark}")
