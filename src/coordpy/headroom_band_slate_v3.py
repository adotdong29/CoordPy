"""W138 / COO-9 — headroom-band battlefield slate v3 (continuous-knob families).

W137 proved the parser-neutral repaired field is BIMODAL at 70B — every template's A1 landed at
{0,1} and only 1/17 survived band admission.  Two confounds narrowed it: (a) A1 was measured with
``n_a1=1`` (so a1_rate ∈ {0,1} BY CONSTRUCTION — ``count_pairs_absdiff_le_d`` was culled HC4_DEAD on a
single coin-flip yet the W137 mechanism bench rescued it at +33.33pp), and (b) the slate spanned too
few mechanism-fixable modes/families for the §7a/§7b SPAN gate (the *real* W137 $0-frontier cause).

This slate fixes (b) on the construction side: every family is a **knob-parameterized FACTORY**, so the
calibration can sweep a CONTINUOUS hardness knob (hidden input size N — the best-attested smooth knob,
CLRS-Text arXiv:2406.04229; FuncBenchGen arXiv:2509.26553) and admit the band cells where the strong
anchor's pass@K rate is intermediate (IRT peak-information at p≈0.5; metabench arXiv:2407.12844 drops
mean-acc>95%/zero-variance items).  The fix for (a) lives in ``headroom_band_calibration_v2`` (A1 as a
population RATE over n_cal≥8 + Wilson interval).

Two earn paths exist under the locked §7a/§7b SPAN rule (``≥2 modes OR ≥3 families``):
* **Path B (high-confidence):** the PROVEN complexity witness (W133 +6.06 vs B0 / +12.12 vs A1; W137
  +33.33pp) rescuing across **≥3 distinct COMPLEXITY families** (the "complexity-only" exclusion bites
  only on a *single* family).  This slate supplies SEVEN complexity families of graded intrinsic
  difficulty so ≥3 land in the rescue band.
* **Path A (2nd-mode test):** two NON-complexity families (compositional + multi-constraint) probe
  whether a counterexample witness can lift a SECOND mode — the open question W133's EW1 +0.00 left
  (it was tested on a bimodal field with no almost-right-fixable instances).  If these saturate or stay
  dead, that CONFIRMS the architecture-requirement R7/R9 (the 2nd mode is capability-bound at 70B).

The seven complexity ref/naive/brute sources are REUSED VERBATIM from the W137-validated
``hard_battlefield_slate_v2`` (already HC1+HC2-clean); only the hidden generators are re-parameterized
by the knob.  Every family is parser-neutral by construction (``IoShapeV1`` + ``render_normal_form_v1``)
and ships the exact-oracle ref/independent-brute/admissible-wrong-naive discipline.  Pure /
deterministic / explicit-import only; ``coordpy/__init__.py`` untouched.
"""
from __future__ import annotations

import dataclasses
import random
from typing import Any, Callable

from .resistant_by_construction_battlefield_v1 import (
    DISC_OUTPUT_MISMATCH, DISC_TIMEOUT,
    MODE_COMPLEXITY_BLIND, MODE_HIDDEN_EDGE, MODE_WRONG_ALGORITHM)
from .hard_battlefield_slate_v2 import ParserNeutralTemplateV2, make_pn_template
from .parser_neutral_io_v1 import array_line, io_shape, scalar_line
from .resistant_by_construction_battlefield_v1 import _sha256_hex

BAND_SLATE_V3_SCHEMA_VERSION: str = "coordpy.headroom_band_slate_v3.v1"

# Knob grids — LOCKED before calibration.  Hidden N is the continuous complexity knob.  At N >= ~2e4 a
# Python O(N^2) naive clearly TLEs the 8s budget while the O(N log N) / O(N) reference finishes with
# margin, so a pass is structural (the model wrote the efficient algorithm), not a timing-wall flake.
CX_KNOB_GRID: tuple[int, ...] = (20_000, 50_000)
# Functional families: hidden N large enough to force an efficient solution; the discriminator is a
# WRONG-ANSWER trap (a dropped stage / dropped constraint), not a timeout.
FUNC_KNOB_GRID: tuple[int, ...] = (4_000, 30_000)


def _jit(rng: random.Random, n: int) -> int:
    """+/-10% jitter around a knob so replicas are not identical-size (per-instance novelty)."""
    lo = max(2, int(n * 0.9))
    hi = max(lo + 1, int(n * 1.1))
    return rng.randint(lo, hi)


# ============================================================ COMPLEXITY families (TIMEOUT)
# ref/naive/brute REUSED VERBATIM from hard_battlefield_slate_v2 (W137-validated, HC1+HC2-clean); only
# gen_hidden is re-parameterized by the knob ``n_hidden``.

def _cx_count_inversions(n_hidden: int) -> ParserNeutralTemplateV2:
    shape = io_shape(scalar_line("N"), array_line("A", "N"))
    ref = ("import sys\n"
           "d=sys.stdin.buffer.read().split()\n"
           "n=int(d[0]);a=[int(x) for x in d[1:1+n]]\n"
           "import bisect\n"
           "srt=sorted(set(a));bit=[0]*(len(srt)+1);inv=0\n"
           "def upd(i):\n"
           "    i+=1\n"
           "    while i<len(bit):bit[i]+=1;i+=i&-i\n"
           "def qry(i):\n"
           "    i+=1;s=0\n"
           "    while i>0:s+=bit[i];i-=i&-i\n"
           "    return s\n"
           "for v in reversed(a):\n"
           "    r=bisect.bisect_left(srt,v)\n"
           "    inv+=qry(r-1) if r>0 else 0\n"
           "    upd(r)\n"
           "print(inv)\n")
    naive = ("import sys\n"
             "d=sys.stdin.buffer.read().split()\n"
             "n=int(d[0]);a=[int(x) for x in d[1:1+n]]\n"
             "inv=0\n"
             "for i in range(n):\n"
             "    ai=a[i]\n"
             "    for j in range(i+1,n):\n"
             "        if ai>a[j]:inv+=1\n"
             "print(inv)\n")
    stmt = ("Count inversions.\n\n"
            "Given an array A of N integers, count the number of pairs (i, j) with i < j and "
            "A[i] > A[j].\n\n"
            "Input: line 1 contains N; line 2 contains N integers A[1..N].\n"
            "Output: the number of inversions.\n"
            "Constraints: 1 <= N <= 200000, 1 <= A[i] <= 10^9.")

    def gp(rng: random.Random) -> list[dict]:
        return [{"N": (n := rng.randint(4, 9)), "A": [rng.randint(1, 20) for _ in range(n)]}
                for _ in range(3)]

    def gh(rng: random.Random) -> list[dict]:
        return [{"N": (n := _jit(rng, n_hidden)), "A": [rng.randint(1, 10 ** 9) for _ in range(n)]}
                for _ in range(3)]

    return make_pn_template(
        name=f"b_count_inversions_n{n_hidden}", family="count_inversions",
        mode=MODE_COMPLEXITY_BLIND, statement=stmt, ref_source=ref, naive_source=naive,
        brute_source=naive, algo_sig="bit_inversions", discriminator=DISC_TIMEOUT,
        brute_cap_tokens=40, shape=shape, gen_public_data=gp, gen_hidden_data=gh,
        headroom_note=f"naive O(N^2) TLEs at N~{n_hidden}; needs BIT/merge-sort O(N log N)")


def _cx_longest_subarray_sum_le_s(n_hidden: int) -> ParserNeutralTemplateV2:
    shape = io_shape(scalar_line("N", "S"), array_line("A", "N"))
    ref = ("import sys\n"
           "d=sys.stdin.buffer.read().split()\n"
           "n=int(d[0]);S=int(d[1]);a=[int(x) for x in d[2:2+n]]\n"
           "best=0;cur=0;l=0\n"
           "for r in range(n):\n"
           "    cur+=a[r]\n"
           "    while cur>S and l<=r:cur-=a[l];l+=1\n"
           "    if r-l+1>best:best=r-l+1\n"
           "print(best)\n")
    naive = ("import sys\n"
             "d=sys.stdin.buffer.read().split()\n"
             "n=int(d[0]);S=int(d[1]);a=[int(x) for x in d[2:2+n]]\n"
             "best=0\n"
             "for i in range(n):\n"
             "    s=0\n"
             "    for j in range(i,n):\n"
             "        s+=a[j]\n"
             "        if s<=S:\n"
             "            if j-i+1>best:best=j-i+1\n"
             "        else:break\n"
             "print(best)\n")
    stmt = ("Longest bounded-sum subarray.\n\n"
            "Given N non-negative integers A and a bound S, output the length of the longest contiguous "
            "subarray whose sum is at most S.\n\n"
            "Input: line 1 contains N and S; line 2 contains N integers A[1..N].\n"
            "Output: the maximum length (0 if no element is <= S).\n"
            "Constraints: 1 <= N <= 200000, 0 <= A[i] <= 10^9, 0 <= S <= 10^18.")

    def gp(rng: random.Random) -> list[dict]:
        out = []
        for _ in range(3):
            n = rng.randint(4, 9)
            a = [rng.randint(0, 9) for _ in range(n)]
            out.append({"N": n, "S": rng.randint(0, sum(a) + 5), "A": a})
        return out

    def gh(rng: random.Random) -> list[dict]:
        out = []
        for _ in range(3):
            n = _jit(rng, n_hidden)
            out.append({"N": n, "S": rng.randint(n // 2, n * 50),
                        "A": [rng.randint(0, 100) for _ in range(n)]})
        return out

    return make_pn_template(
        name=f"b_longest_subarray_sum_le_s_n{n_hidden}", family="longest_bounded_subarray",
        mode=MODE_COMPLEXITY_BLIND, statement=stmt, ref_source=ref, naive_source=naive,
        brute_source=naive, algo_sig="sliding_window", discriminator=DISC_TIMEOUT,
        brute_cap_tokens=40, shape=shape, gen_public_data=gp, gen_hidden_data=gh,
        headroom_note=f"naive O(N^2) expanding-window TLEs at N~{n_hidden}; needs two-pointer O(N)")


def _cx_count_pairs_sum_le_t(n_hidden: int) -> ParserNeutralTemplateV2:
    shape = io_shape(scalar_line("N", "T"), array_line("A", "N"))
    ref = ("import sys\n"
           "d=sys.stdin.buffer.read().split()\n"
           "n=int(d[0]);T=int(d[1]);a=[int(x) for x in d[2:2+n]]\n"
           "a.sort()\n"
           "i=0;j=n-1;cnt=0\n"
           "while i<j:\n"
           "    if a[i]+a[j]<=T:\n"
           "        cnt+=j-i;i+=1\n"
           "    else:\n"
           "        j-=1\n"
           "print(cnt)\n")
    naive = ("import sys\n"
             "d=sys.stdin.buffer.read().split()\n"
             "n=int(d[0]);T=int(d[1]);a=[int(x) for x in d[2:2+n]]\n"
             "cnt=0\n"
             "for i in range(n):\n"
             "    ai=a[i]\n"
             "    for j in range(i+1,n):\n"
             "        if ai+a[j]<=T:cnt+=1\n"
             "print(cnt)\n")
    stmt = ("Cheap pair tally.\n\n"
            "Call an unordered pair of distinct positions CHEAP when the two values they hold add up to "
            "no more than the budget T. You are given N values and the budget T; report how many cheap "
            "pairs exist.\n\n"
            "Input: line 1 contains N and T; line 2 contains N integers A[1..N].\n"
            "Output: how many cheap pairs there are.\n"
            "Constraints: 1 <= N <= 200000, 1 <= A[i] <= 10^9, 2 <= T <= 2*10^9.")

    def gp(rng: random.Random) -> list[dict]:
        return [{"N": (n := rng.randint(4, 9)), "T": rng.randint(2, 40),
                 "A": [rng.randint(1, 20) for _ in range(n)]} for _ in range(3)]

    def gh(rng: random.Random) -> list[dict]:
        return [{"N": (n := _jit(rng, n_hidden)), "T": rng.randint(1, 2 * 10 ** 9),
                 "A": [rng.randint(1, 10 ** 9) for _ in range(n)]} for _ in range(3)]

    return make_pn_template(
        name=f"b_count_pairs_sum_le_t_n{n_hidden}", family="count_pairs_sum_le_t",
        mode=MODE_COMPLEXITY_BLIND, statement=stmt, ref_source=ref, naive_source=naive,
        brute_source=naive, algo_sig="sort_two_pointer_pairsum", discriminator=DISC_TIMEOUT,
        brute_cap_tokens=40, shape=shape, gen_public_data=gp, gen_hidden_data=gh,
        headroom_note=f"naive O(N^2) pair scan TLEs at N~{n_hidden}; needs sort + two-pointer O(N log N)")


def _cx_count_pairs_absdiff_le_d(n_hidden: int) -> ParserNeutralTemplateV2:
    shape = io_shape(scalar_line("N", "D"), array_line("A", "N"))
    ref = ("import sys\n"
           "d=sys.stdin.buffer.read().split()\n"
           "n=int(d[0]);D=int(d[1]);a=[int(x) for x in d[2:2+n]]\n"
           "a.sort()\n"
           "l=0;cnt=0\n"
           "for r in range(n):\n"
           "    while a[r]-a[l]>D:l+=1\n"
           "    cnt+=r-l\n"
           "print(cnt)\n")
    naive = ("import sys\n"
             "d=sys.stdin.buffer.read().split()\n"
             "n=int(d[0]);D=int(d[1]);a=[int(x) for x in d[2:2+n]]\n"
             "cnt=0\n"
             "for i in range(n):\n"
             "    ai=a[i]\n"
             "    for j in range(i+1,n):\n"
             "        dv=ai-a[j]\n"
             "        if -D<=dv<=D:cnt+=1\n"
             "print(cnt)\n")
    stmt = ("Similar-value position pairs.\n\n"
            "Two positions in the array are SIMILAR when the absolute difference of the values stored "
            "there is no greater than the tolerance D. You are given N stored values and the tolerance "
            "D; report how many unordered position pairs {i, j} with i != j are similar.\n\n"
            "Input: line 1 contains N and D; line 2 contains N integers A[1..N].\n"
            "Output: how many similar position pairs exist.\n"
            "Constraints: 1 <= N <= 200000, 1 <= A[i] <= 10^9, 0 <= D <= 10^9.")

    def gp(rng: random.Random) -> list[dict]:
        return [{"N": (n := rng.randint(4, 9)), "D": rng.randint(0, 10),
                 "A": [rng.randint(1, 20) for _ in range(n)]} for _ in range(3)]

    def gh(rng: random.Random) -> list[dict]:
        return [{"N": (n := _jit(rng, n_hidden)), "D": rng.randint(0, 10 ** 9),
                 "A": [rng.randint(1, 10 ** 9) for _ in range(n)]} for _ in range(3)]

    return make_pn_template(
        name=f"b_count_pairs_absdiff_le_d_n{n_hidden}", family="count_pairs_absdiff_le_d",
        mode=MODE_COMPLEXITY_BLIND, statement=stmt, ref_source=ref, naive_source=naive,
        brute_source=naive, algo_sig="sort_sliding_window_absdiff", discriminator=DISC_TIMEOUT,
        brute_cap_tokens=40, shape=shape, gen_public_data=gp, gen_hidden_data=gh,
        headroom_note=f"naive O(N^2) pair scan TLEs at N~{n_hidden}; needs sort + sliding-window")


def _cx_sum_nearest_smaller_left(n_hidden: int) -> ParserNeutralTemplateV2:
    shape = io_shape(scalar_line("N"), array_line("A", "N"))
    ref = ("import sys\n"
           "d=sys.stdin.buffer.read().split()\n"
           "n=int(d[0]);a=[int(x) for x in d[1:1+n]]\n"
           "st=[];total=0\n"
           "for v in a:\n"
           "    while st and st[-1]>=v:st.pop()\n"
           "    total+=st[-1] if st else -1\n"
           "    st.append(v)\n"
           "print(total)\n")
    naive = ("import sys\n"
             "d=sys.stdin.buffer.read().split()\n"
             "n=int(d[0]);a=[int(x) for x in d[1:1+n]]\n"
             "total=0\n"
             "for i in range(n):\n"
             "    v=a[i];found=-1\n"
             "    for j in range(i-1,-1,-1):\n"
             "        if a[j]<v:found=a[j];break\n"
             "    total+=found\n"
             "print(total)\n")
    stmt = ("Nearest smaller to the left.\n\n"
            "Given an array A of N integers, for each position i let L(i) be the value of the nearest "
            "element to the LEFT of i that is strictly smaller than A[i], or -1 if no such element "
            "exists. Output the sum of L(i) over all i.\n\n"
            "Input: line 1 contains N; line 2 contains N integers A[1..N].\n"
            "Output: the sum of nearest-smaller-to-the-left values.\n"
            "Constraints: 1 <= N <= 200000, 1 <= A[i] <= 10^9.")

    def gp(rng: random.Random) -> list[dict]:
        return [{"N": (n := rng.randint(4, 9)), "A": [rng.randint(1, 20) for _ in range(n)]}
                for _ in range(3)]

    def gh(rng: random.Random) -> list[dict]:
        # worst case for the naive inner scan: strictly decreasing array forces O(N^2)
        return [{"N": (n := _jit(rng, n_hidden)), "A": list(range(n, 0, -1))} for _ in range(3)]

    return make_pn_template(
        name=f"b_sum_nearest_smaller_left_n{n_hidden}", family="sum_nearest_smaller_left",
        mode=MODE_COMPLEXITY_BLIND, statement=stmt, ref_source=ref, naive_source=naive,
        brute_source=naive, algo_sig="monotonic_stack_prev_smaller", discriminator=DISC_TIMEOUT,
        brute_cap_tokens=40, shape=shape, gen_public_data=gp, gen_hidden_data=gh,
        headroom_note=f"naive O(N^2) inner left-scan TLEs at N~{n_hidden}; needs a monotonic stack O(N)")


def _cx_count_subarrays_sum_le_s(n_hidden: int) -> ParserNeutralTemplateV2:
    shape = io_shape(scalar_line("N", "S"), array_line("A", "N"))
    ref = ("import sys\n"
           "d=sys.stdin.buffer.read().split()\n"
           "n=int(d[0]);S=int(d[1]);a=[int(x) for x in d[2:2+n]]\n"
           "l=0;cur=0;cnt=0\n"
           "for r in range(n):\n"
           "    cur+=a[r]\n"
           "    while cur>S:cur-=a[l];l+=1\n"
           "    cnt+=r-l+1\n"
           "print(cnt)\n")
    naive = ("import sys\n"
             "d=sys.stdin.buffer.read().split()\n"
             "n=int(d[0]);S=int(d[1]);a=[int(x) for x in d[2:2+n]]\n"
             "cnt=0\n"
             "for i in range(n):\n"
             "    s=0\n"
             "    for j in range(i,n):\n"
             "        s+=a[j]\n"
             "        if s<=S:cnt+=1\n"
             "        else:break\n"
             "print(cnt)\n")
    stmt = ("Tally of light blocks.\n\n"
            "A contiguous non-empty block of the sequence is called LIGHT when the total of its entries "
            "does not exceed the capacity S. You are given N non-negative entries and the capacity S; "
            "report how many light blocks the sequence contains.\n\n"
            "Input: line 1 contains N and S; line 2 contains N integers A[1..N].\n"
            "Output: how many light blocks there are.\n"
            "Constraints: 1 <= N <= 200000, 0 <= A[i] <= 10^9, 0 <= S <= 10^18.")

    def gp(rng: random.Random) -> list[dict]:
        out = []
        for _ in range(3):
            n = rng.randint(4, 9)
            a = [rng.randint(0, 9) for _ in range(n)]
            out.append({"N": n, "S": rng.randint(0, sum(a) + 5), "A": a})
        return out

    def gh(rng: random.Random) -> list[dict]:
        return [{"N": (n := _jit(rng, n_hidden)), "S": rng.randint(n * 10, n * 80),
                 "A": [rng.randint(0, 100) for _ in range(n)]} for _ in range(3)]

    return make_pn_template(
        name=f"b_count_subarrays_sum_le_s_n{n_hidden}", family="count_subarrays_sum_le_s",
        mode=MODE_COMPLEXITY_BLIND, statement=stmt, ref_source=ref, naive_source=naive,
        brute_source=naive, algo_sig="two_pointer_count_subarrays", discriminator=DISC_TIMEOUT,
        brute_cap_tokens=40, shape=shape, gen_public_data=gp, gen_hidden_data=gh,
        headroom_note=f"naive O(N^2) expanding count TLEs at N~{n_hidden}; needs two-pointer O(N)")


def _cx_max_j_minus_i_le(n_hidden: int) -> ParserNeutralTemplateV2:
    shape = io_shape(scalar_line("N"), array_line("A", "N"))
    ref = ("import sys\n"
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
    naive = ("import sys\n"
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
    stmt = ("Widest non-decreasing index pair.\n\n"
            "Given an array A of N integers, find the maximum value of j - i over all index pairs with "
            "i <= j and A[i] <= A[j]. Output that maximum (0 if no pair other than i = j qualifies).\n\n"
            "Input: line 1 contains N; line 2 contains N integers A[1..N].\n"
            "Output: the maximum j - i.\n"
            "Constraints: 1 <= N <= 200000, 1 <= A[i] <= 10^9.")

    def gp(rng: random.Random) -> list[dict]:
        return [{"N": (n := rng.randint(4, 9)), "A": [rng.randint(1, 20) for _ in range(n)]}
                for _ in range(3)]

    def gh(rng: random.Random) -> list[dict]:
        return [{"N": (n := _jit(rng, n_hidden)), "A": list(range(n, 0, -1))} for _ in range(3)]

    return make_pn_template(
        name=f"b_max_j_minus_i_le_n{n_hidden}", family="max_j_minus_i_le",
        mode=MODE_COMPLEXITY_BLIND, statement=stmt, ref_source=ref, naive_source=naive,
        brute_source=naive, algo_sig="prefix_min_suffix_max_two_pointer", discriminator=DISC_TIMEOUT,
        brute_cap_tokens=40, shape=shape, gen_public_data=gp, gen_hidden_data=gh,
        headroom_note=f"naive O(N^2) all-pairs scan TLEs at N~{n_hidden}; needs prefix-min/suffix-max O(N)")


# ============================================================ FUNCTIONAL families (WRONG_ANSWER)
# The 2nd-mode test (Path A): the difficulty is a DROPPED STAGE / DROPPED CONSTRAINT, not a timeout.
# The naive passes every public sample (where the dropped piece is a no-op) and fails >=1 hidden case
# (where it binds) — a clean OUTPUT_MISMATCH discriminator.  A counterexample witness should reveal the
# binding input; whether the 70B can ACT on it is exactly the open W133-EW1 question.

def _ms_mod_then_maxsub(n_hidden: int) -> ParserNeutralTemplateV2:
    """Compositional (depth-2): B[i] = A[i] mod M, then max-subarray-sum of (B[i] - K)."""
    shape = io_shape(scalar_line("N", "M", "K"), array_line("A", "N"))
    ref = ("import sys\n"
           "d=sys.stdin.buffer.read().split()\n"
           "n=int(d[0]);M=int(d[1]);K=int(d[2]);a=[int(x) for x in d[3:3+n]]\n"
           "cur=(a[0]%M)-K;best=cur\n"
           "for i in range(1,n):\n"
           "    c=(a[i]%M)-K\n"
           "    cur=c if cur+c<c else cur+c\n"
           "    if cur>best:best=cur\n"
           "print(best)\n")
    naive = ("import sys\n"
             "d=sys.stdin.buffer.read().split()\n"
             "n=int(d[0]);M=int(d[1]);K=int(d[2]);a=[int(x) for x in d[3:3+n]]\n"
             "cur=a[0]-K;best=cur\n"
             "for i in range(1,n):\n"
             "    c=a[i]-K\n"
             "    cur=c if cur+c<c else cur+c\n"
             "    if cur>best:best=cur\n"
             "print(best)\n")
    brute = ("import sys\n"
             "d=sys.stdin.buffer.read().split()\n"
             "n=int(d[0]);M=int(d[1]);K=int(d[2]);a=[int(x) for x in d[3:3+n]]\n"
             "best=None\n"
             "for i in range(n):\n"
             "    s=0\n"
             "    for j in range(i,n):\n"
             "        s+=(a[j]%M)-K\n"
             "        best=s if best is None or s>best else best\n"
             "print(best)\n")
    stmt = ("Reduced-stream best segment.\n\n"
            "Given N raw values A, a modulus M and an offset K, first reduce each value to "
            "R[i] = (A[i] mod M) - K. Then output the maximum possible sum of a non-empty contiguous "
            "segment of the reduced array R.\n\n"
            "Input: line 1 contains N, M and K; line 2 contains N integers A[1..N].\n"
            "Output: the maximum contiguous segment sum of R.\n"
            "Constraints: 1 <= N <= 200000, 1 <= M <= 10^9, 0 <= K <= 10^9, 0 <= A[i] <= 10^9.")

    def gp(rng: random.Random) -> list[dict]:
        # public: A[i] < M so the mod is a no-op -> the (wrong) no-mod naive coincides with ref
        out = []
        for _ in range(3):
            n = rng.randint(4, 8)
            M = rng.randint(20, 50)
            out.append({"N": n, "M": M, "K": rng.randint(0, 10),
                        "A": [rng.randint(0, M - 1) for _ in range(n)]})
        return out

    def gh(rng: random.Random) -> list[dict]:
        # hidden: A[i] spans >> M so the mod genuinely changes the reduced stream and the answer
        out = []
        for _ in range(3):
            n = _jit(rng, n_hidden)
            M = rng.randint(50, 500)
            out.append({"N": n, "M": M, "K": rng.randint(1, M),
                        "A": [rng.randint(0, 10 ** 9) for _ in range(n)]})
        return out

    return make_pn_template(
        name=f"b_mod_then_maxsub_n{n_hidden}", family="mod_then_maxsub",
        mode=MODE_WRONG_ALGORITHM, statement=stmt, ref_source=ref, naive_source=naive,
        brute_source=brute, algo_sig="compose_mod_kadane", discriminator=DISC_OUTPUT_MISMATCH,
        brute_cap_tokens=40, shape=shape, gen_public_data=gp, gen_hidden_data=gh,
        headroom_note="2-stage compose; naive drops the mod stage (no-op on public, binds on hidden)")


def _ce_subarrays_sum_and_range(n_hidden: int) -> ParserNeutralTemplateV2:
    """Multi-constraint: count subarrays with sum <= S AND (max - min) <= R."""
    shape = io_shape(scalar_line("N", "S", "R"), array_line("A", "N"))
    ref = ("import sys\n"
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
    naive = ("import sys\n"
             "d=sys.stdin.buffer.read().split()\n"
             "n=int(d[0]);S=int(d[1]);R=int(d[2]);a=[int(x) for x in d[3:3+n]]\n"
             "cur=0;l=0;cnt=0\n"
             "for r in range(n):\n"
             "    cur+=a[r]\n"
             "    while cur>S:cur-=a[l];l+=1\n"
             "    cnt+=r-l+1\n"
             "print(cnt)\n")
    brute = ("import sys\n"
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
    stmt = ("Calm blocks.\n\n"
            "A contiguous non-empty block of the sequence is CALM when BOTH of these hold: its entries "
            "total at most S, and the gap between its largest and smallest entry is at most R. You are "
            "given N non-negative entries, the total cap S and the gap cap R; report how many calm "
            "blocks the sequence contains.\n\n"
            "Input: line 1 contains N, S and R; line 2 contains N integers A[1..N].\n"
            "Output: how many calm blocks there are.\n"
            "Constraints: 1 <= N <= 200000, 0 <= A[i] <= 10^9, 0 <= S <= 10^18, 0 <= R <= 10^9.")

    def gp(rng: random.Random) -> list[dict]:
        # public: all entries within R of each other -> the gap cap never binds -> sum-only naive == ref
        out = []
        for _ in range(3):
            n = rng.randint(4, 8)
            base = rng.randint(0, 50)
            R = rng.randint(5, 15)
            a = [base + rng.randint(0, R) for _ in range(n)]
            out.append({"N": n, "S": sum(a) + rng.randint(0, 20), "R": R, "A": a})
        return out

    def gh(rng: random.Random) -> list[dict]:
        # hidden: values modest (so the SUM cap passes for many windows) but the spread (up to ~200)
        # far exceeds R (10-40), so the GAP cap genuinely binds on wide windows the sum-only naive
        # wrongly counts.  Both constraints active => the sum-only naive reliably OVERCOUNTS.
        out = []
        for _ in range(3):
            n = _jit(rng, n_hidden)
            R = rng.randint(10, 40)
            out.append({"N": n, "S": rng.randint(n * 40, n * 80), "R": R,
                        "A": [rng.randint(0, 200) for _ in range(n)]})
        return out

    return make_pn_template(
        name=f"b_subarrays_sum_and_range_n{n_hidden}", family="subarrays_sum_and_range",
        mode=MODE_HIDDEN_EDGE, statement=stmt, ref_source=ref, naive_source=naive,
        brute_source=brute, algo_sig="two_pointer_two_constraint_deques",
        discriminator=DISC_OUTPUT_MISMATCH, brute_cap_tokens=40, shape=shape,
        gen_public_data=gp, gen_hidden_data=gh,
        headroom_note="2 constraints; naive drops the range cap (no-op on public, binds on hidden)")


# ============================================================ EXTRA complexity families (hedge)
# Harder complexity families where a 70B is BORDERLINE on writing the efficient algorithm (the band
# sweet spot between the saturated textbook families and the capability-dead ones).  Gated behind
# ``include_extra`` so the DEFAULT slate (and its CID + any running calibration) is unaffected; wired
# in only if the first calibration's complexity band is thin (<3 families).  TIMEOUT discriminator.

def _cx_subarrays_at_most_k_distinct(n_hidden: int) -> ParserNeutralTemplateV2:
    shape = io_shape(scalar_line("N", "K"), array_line("A", "N"))
    ref = ("import sys\n"
           "from collections import defaultdict\n"
           "d=sys.stdin.buffer.read().split()\n"
           "n=int(d[0]);K=int(d[1]);a=[int(x) for x in d[2:2+n]]\n"
           "cnt=0;l=0;freq=defaultdict(int);dist=0\n"
           "for r in range(n):\n"
           "    if freq[a[r]]==0:dist+=1\n"
           "    freq[a[r]]+=1\n"
           "    while dist>K:\n"
           "        freq[a[l]]-=1\n"
           "        if freq[a[l]]==0:dist-=1\n"
           "        l+=1\n"
           "    cnt+=r-l+1\n"
           "print(cnt)\n")
    naive = ("import sys\n"
             "d=sys.stdin.buffer.read().split()\n"
             "n=int(d[0]);K=int(d[1]);a=[int(x) for x in d[2:2+n]]\n"
             "cnt=0\n"
             "for i in range(n):\n"
             "    s=set()\n"
             "    for j in range(i,n):\n"
             "        s.add(a[j])\n"
             "        if len(s)<=K:cnt+=1\n"
             "        else:break\n"
             "print(cnt)\n")
    stmt = ("Tally of varied blocks.\n\n"
            "A contiguous non-empty block of the sequence is VARIED-OK when it contains at most K "
            "distinct values. You are given N values and the bound K; report how many varied-OK blocks "
            "the sequence contains.\n\n"
            "Input: line 1 contains N and K; line 2 contains N integers A[1..N].\n"
            "Output: how many varied-OK blocks there are.\n"
            "Constraints: 1 <= N <= 200000, 1 <= K <= N, 1 <= A[i] <= 10^9.")

    def gp(rng: random.Random) -> list[dict]:
        return [{"N": (n := rng.randint(4, 9)), "K": rng.randint(1, 4),
                 "A": [rng.randint(1, 8) for _ in range(n)]} for _ in range(3)]

    def gh(rng: random.Random) -> list[dict]:
        # K large + many distinct values => the naive inner loop rarely breaks => O(N^2)
        return [{"N": (n := _jit(rng, n_hidden)), "K": n,
                 "A": [rng.randint(1, 10 ** 9) for _ in range(n)]} for _ in range(3)]

    return make_pn_template(
        name=f"b_subarrays_at_most_k_distinct_n{n_hidden}", family="subarrays_at_most_k_distinct",
        mode=MODE_COMPLEXITY_BLIND, statement=stmt, ref_source=ref, naive_source=naive,
        brute_source=naive, algo_sig="sliding_window_at_most_k_distinct", discriminator=DISC_TIMEOUT,
        brute_cap_tokens=40, shape=shape, gen_public_data=gp, gen_hidden_data=gh,
        headroom_note=f"naive O(N^2) distinct-set scan TLEs at N~{n_hidden}; needs at-most-K sliding window O(N)")


def _cx_kth_smallest_pair_distance(n_hidden: int) -> ParserNeutralTemplateV2:
    shape = io_shape(scalar_line("N", "K"), array_line("A", "N"))
    ref = ("import sys\n"
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
    naive = ("import sys\n"
             "d=sys.stdin.buffer.read().split()\n"
             "n=int(d[0]);K=int(d[1]);a=[int(x) for x in d[2:2+n]]\n"
             "ds=[]\n"
             "for i in range(n):\n"
             "    ai=a[i]\n"
             "    for j in range(i+1,n):\n"
             "        ds.append(ai-a[j] if ai>a[j] else a[j]-ai)\n"
             "ds.sort()\n"
             "print(ds[K-1])\n")
    stmt = ("K-th smallest pair gap.\n\n"
            "Consider every unordered pair of distinct positions; the GAP of a pair is the absolute "
            "difference of the two values it holds. You are given N values and an index K; output the "
            "K-th smallest gap (1-indexed) over all N*(N-1)/2 pairs.\n\n"
            "Input: line 1 contains N and K; line 2 contains N integers A[1..N].\n"
            "Output: the K-th smallest pair gap.\n"
            "Constraints: 2 <= N <= 200000, 1 <= K <= N*(N-1)/2, 1 <= A[i] <= 10^9.")

    def gp(rng: random.Random) -> list[dict]:
        out = []
        for _ in range(3):
            n = rng.randint(4, 8)
            out.append({"N": n, "K": rng.randint(1, n * (n - 1) // 2),
                        "A": [rng.randint(1, 30) for _ in range(n)]})
        return out

    def gh(rng: random.Random) -> list[dict]:
        out = []
        for _ in range(3):
            n = _jit(rng, n_hidden)
            out.append({"N": n, "K": rng.randint(1, n * (n - 1) // 2),
                        "A": [rng.randint(1, 10 ** 9) for _ in range(n)]})
        return out

    return make_pn_template(
        name=f"b_kth_smallest_pair_distance_n{n_hidden}", family="kth_smallest_pair_distance",
        mode=MODE_COMPLEXITY_BLIND, statement=stmt, ref_source=ref, naive_source=naive,
        brute_source=naive, algo_sig="binary_search_answer_two_pointer", discriminator=DISC_TIMEOUT,
        brute_cap_tokens=40, shape=shape, gen_public_data=gp, gen_hidden_data=gh,
        headroom_note=f"naive O(N^2) all-pairs sort TLEs at N~{n_hidden}; needs binary-search-on-answer + two-pointer O(N log maxA)")


# ============================================================ slate registry

# family_id -> factory(knob_value) -> ParserNeutralTemplateV2
CX_FACTORIES: dict[str, Callable[[int], ParserNeutralTemplateV2]] = {
    "count_inversions": _cx_count_inversions,
    "longest_bounded_subarray": _cx_longest_subarray_sum_le_s,
    "count_pairs_sum_le_t": _cx_count_pairs_sum_le_t,
    "count_pairs_absdiff_le_d": _cx_count_pairs_absdiff_le_d,
    "sum_nearest_smaller_left": _cx_sum_nearest_smaller_left,
    "count_subarrays_sum_le_s": _cx_count_subarrays_sum_le_s,
    "max_j_minus_i_le": _cx_max_j_minus_i_le,
}
FUNC_FACTORIES: dict[str, Callable[[int], ParserNeutralTemplateV2]] = {
    "mod_then_maxsub": _ms_mod_then_maxsub,
    "subarrays_sum_and_range": _ce_subarrays_sum_and_range,
}
# hedge: harder 70B-borderline complexity families, wired in ONLY when include_extra=True (so the
# default slate CID is unaffected) — used if the first calibration's complexity band is thin.
EXTRA_CX_FACTORIES: dict[str, Callable[[int], ParserNeutralTemplateV2]] = {
    "subarrays_at_most_k_distinct": _cx_subarrays_at_most_k_distinct,
    "kth_smallest_pair_distance": _cx_kth_smallest_pair_distance,
}


@dataclasses.dataclass(frozen=True)
class BandCandidateV1:
    cell_id: str          # f"{family}@{knob}"
    family: str
    mode: str
    knob_name: str        # "n_hidden"
    knob_value: int
    template: ParserNeutralTemplateV2


def build_band_candidates_v3(*, cx_knobs: tuple[int, ...] = CX_KNOB_GRID,
                             func_knobs: tuple[int, ...] = FUNC_KNOB_GRID,
                             include_extra: bool = False,
                             ) -> list[BandCandidateV1]:
    """The full (family x knob) candidate grid — the search space the calibration sweeps.
    ``include_extra`` adds the harder hedge complexity families (changes the slate CID)."""
    out: list[BandCandidateV1] = []
    cx = dict(CX_FACTORIES)
    if include_extra:
        cx.update(EXTRA_CX_FACTORIES)
    for fam, fac in cx.items():
        for k in cx_knobs:
            t = fac(k)
            out.append(BandCandidateV1(cell_id=f"{fam}@{k}", family=fam, mode=t.minted.mode,
                                       knob_name="n_hidden", knob_value=k, template=t))
    for fam, fac in FUNC_FACTORIES.items():
        for k in func_knobs:
            t = fac(k)
            out.append(BandCandidateV1(cell_id=f"{fam}@{k}", family=fam, mode=t.minted.mode,
                                       knob_name="n_hidden", knob_value=k, template=t))
    return out


def band_slate_fingerprint_cid_v1(*, cx_knobs: tuple[int, ...] = CX_KNOB_GRID,
                                  func_knobs: tuple[int, ...] = FUNC_KNOB_GRID,
                                  include_extra: bool = False) -> str:
    cands = build_band_candidates_v3(cx_knobs=cx_knobs, func_knobs=func_knobs,
                                     include_extra=include_extra)
    return _sha256_hex({"k": "w138_band_slate_v3",
                        "cells": [{"cell": c.cell_id, "family": c.family, "mode": c.mode,
                                   "algo_sig": c.template.minted.algo_sig,
                                   "disc": c.template.minted.discriminator,
                                   "stmt": c.template.minted.statement,
                                   "ref": c.template.minted.ref_source,
                                   "shape": c.template.io_shape.shape_cid()} for c in cands]})


__all__ = [
    "BAND_SLATE_V3_SCHEMA_VERSION", "CX_KNOB_GRID", "FUNC_KNOB_GRID",
    "CX_FACTORIES", "FUNC_FACTORIES", "EXTRA_CX_FACTORIES", "BandCandidateV1",
    "build_band_candidates_v3", "band_slate_fingerprint_cid_v1",
]
