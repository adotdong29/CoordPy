"""W132 / COO-9 — the resistant-by-construction problem SLATE (the minted families).

Thirty-four hand-authored algorithmic problem GENERATORS, four targeted failure
families (the W130/W131 atlas modes).  Each template ships a scalable correct
``ref_source`` (the answer-key oracle), an independent obviously-correct ``brute_source``
(the small-case cross-check), and an admissible-wrong ``naive_source`` (the trap):

* ``COMPLEXITY_BLIND`` — the naive IS the correct algorithm, but O(N^2); it TLEs on the
  large hidden stress case while the O(N log N)/O(N) reference finishes.
* ``HIDDEN_EDGE_STATE_MISS`` — the naive is a plausible solution that is right on the
  typical public cases and wrong on a constructed corner (wrap / overlap / type / tie /
  inclusivity / sign).
* ``WRONG_ALGORITHM_ADMISSIBLE`` — the naive is a NAMED-but-wrong technique (a greedy
  where DP is required); the hidden case is a constructed greedy-defeating instance.
* ``SEARCH_ENUM`` — a small-n exhaustive oracle is exact; the naive is a plausible
  miscount (ordered-vs-unordered / wrong recurrence / blocks ignored).

Generators are seeded ``random.Random``; the PUBLIC cases are constructed naive-SAFE and
the HIDDEN cases constructed naive-TRAP (with randomized magnitudes for freshness) so the
discriminating gate passes by construction, not by luck.  No statement, solver, or case
is copied from any official benchmark; the families are textbook algorithm-families
(family-level inspiration is allowed, same-problem reuse is not).
"""
from __future__ import annotations

import random
from typing import Callable

from .resistant_by_construction_battlefield_v1 import (
    DISC_OUTPUT_MISMATCH,
    DISC_TIMEOUT,
    MODE_COMPLEXITY_BLIND,
    MODE_HIDDEN_EDGE,
    MODE_SEARCH_ENUM,
    MODE_WRONG_ALGORITHM,
    MintedTemplateV1,
)
from .coordpy_icpc_battlefield_v1 import KIND_PASSFAIL

# Large stress N for the COMPLEXITY family: O(N^2) ~ 3.2e9 inner ops (>>8s TLE on CPython);
# O(N log N) reference finishes in well under a second.
CB_BIG: int = 80_000
CB_MED: int = 1_500            # brute-checkable medium (cross-checks ref at non-trivial N)


def _case(header, arr) -> str:
    h = " ".join(str(x) for x in header) if isinstance(header, (list, tuple)) else str(header)
    return h + "\n" + " ".join(str(x) for x in arr)


# ============================================================ generic array generators

def _arr_gen(specs) -> Callable[[random.Random], list[str]]:
    """specs: list of (n, lo, hi, header_fn|None).  header_fn(rng,n,arr)->list tokens;
    None => header is just [n]."""
    def f(rng: random.Random) -> list[str]:
        out = []
        for (n, lo, hi, hf) in specs:
            arr = [rng.randint(lo, hi) for _ in range(n)]
            head = hf(rng, n, arr) if hf else [n]
            out.append(_case(head, arr))
        return out
    return f


def _T(rng, lo, hi):
    return rng.randint(lo, hi)


def _hid_nearest_smaller(rng: random.Random) -> list[str]:
    """Hidden cases for the nearest-smaller-left problem.  The naive scans left until it
    finds a smaller element, so on RANDOM data it short-circuits (≈O(N)); the large
    stress case must be the worst case (strictly DECREASING, no smaller-on-left ever) to
    force the genuine O(N^2) TLE.  Small/medium cases stay random to cross-check ref."""
    out = []
    for n in (9, 12):
        out.append(_case([n], [rng.randint(1, 20) for _ in range(n)]))
    out.append(_case([CB_MED], [rng.randint(1, 10**9) for _ in range(CB_MED)]))
    out.append(_case([CB_BIG], list(range(CB_BIG, 0, -1))))
    return out


# ============================================================================ slate

def build_slate_v1() -> tuple[MintedTemplateV1, ...]:
    S: list[MintedTemplateV1] = []

    # ============================================================ COMPLEXITY_BLIND (9)
    # For every complexity problem naive_source == brute_source (the O(N^2) algorithm):
    # correct-but-slow.  discriminator = TIMEOUT (TLE on CB_BIG); brute cross-checks ref
    # on the small + CB_MED cases.

    # CB1 — count pairs i<j with a_i + a_j <= T
    ref = ("import sys\n"
           "d=sys.stdin.buffer.read().split()\n"
           "n=int(d[0]);T=int(d[1]);a=sorted(int(x) for x in d[2:2+n])\n"
           "i=0;j=n-1;c=0\n"
           "while i<j:\n"
           "    if a[i]+a[j]<=T:\n"
           "        c+=j-i;i+=1\n"
           "    else:\n"
           "        j-=1\n"
           "print(c)\n")
    naive = ("import sys\n"
             "d=sys.stdin.buffer.read().split()\n"
             "n=int(d[0]);T=int(d[1]);a=[int(x) for x in d[2:2+n]]\n"
             "c=0\n"
             "for i in range(n):\n"
             "    ai=a[i]\n"
             "    for j in range(i+1,n):\n"
             "        if ai+a[j]<=T:c+=1\n"
             "print(c)\n")
    hT = lambda r, n, a: [n, _T(r, 5, 60)]
    hTbig = lambda r, n, a: [n, _T(r, 10**9, 2 * 10**9)]
    S.append(MintedTemplateV1(
        name="cb_pairs_sum_le_t", family="pair_threshold_count",
        mode=MODE_COMPLEXITY_BLIND, kind=KIND_PASSFAIL, float_tol=0.0,
        statement=(
            "Count pairs with bounded sum.\n\n"
            "You are given N integers a_1..a_N and a threshold T. Count the number of "
            "pairs of indices (i, j) with i < j and a_i + a_j <= T.\n\n"
            "Input: the first line contains two integers N and T. The second line "
            "contains N integers.\nOutput: a single integer, the count.\n"
            "Constraints: 1 <= N <= 100000, 1 <= a_i <= 10^9, 1 <= T <= 2*10^9."),
        ref_source=ref, naive_source=naive, brute_source=naive,
        algo_sig="sort_two_pointer", discriminator=DISC_TIMEOUT, brute_cap_tokens=4000,
        gen_public=_arr_gen([(4, 1, 20, hT), (6, 1, 25, hT), (7, 1, 30, hT)]),
        gen_hidden=_arr_gen([(8, 1, 30, hT), (11, 1, 40, hT),
                             (CB_MED, 1, 10**9, hTbig), (CB_BIG, 1, 10**9, hTbig)])))

    # CB2 — count pairs i<j with |a_i - a_j| <= D
    ref = ("import sys\n"
           "d=sys.stdin.buffer.read().split()\n"
           "n=int(d[0]);D=int(d[1]);a=sorted(int(x) for x in d[2:2+n])\n"
           "i=0;c=0\n"
           "for j in range(n):\n"
           "    while a[j]-a[i]>D:i+=1\n"
           "    c+=j-i\n"
           "print(c)\n")
    naive = ("import sys\n"
             "d=sys.stdin.buffer.read().split()\n"
             "n=int(d[0]);D=int(d[1]);a=[int(x) for x in d[2:2+n]]\n"
             "c=0\n"
             "for i in range(n):\n"
             "    ai=a[i]\n"
             "    for j in range(i+1,n):\n"
             "        if abs(ai-a[j])<=D:c+=1\n"
             "print(c)\n")
    hD = lambda r, n, a: [n, _T(r, 1, 15)]
    hDbig = lambda r, n, a: [n, _T(r, 10**8, 5 * 10**8)]
    S.append(MintedTemplateV1(
        name="cb_pairs_absdiff_le_d", family="pair_distance_count",
        mode=MODE_COMPLEXITY_BLIND, kind=KIND_PASSFAIL, float_tol=0.0,
        statement=(
            "Count close pairs.\n\n"
            "Given N integers and a value D, count the pairs (i, j), i < j, with "
            "|a_i - a_j| <= D.\n\n"
            "Input: first line N and D; second line N integers.\nOutput: the count.\n"
            "Constraints: 1 <= N <= 100000, 1 <= a_i <= 10^9, 0 <= D <= 10^9."),
        ref_source=ref, naive_source=naive, brute_source=naive,
        algo_sig="sort_sliding_window", discriminator=DISC_TIMEOUT, brute_cap_tokens=4000,
        gen_public=_arr_gen([(4, 1, 15, hD), (6, 1, 20, hD), (7, 1, 25, hD)]),
        gen_hidden=_arr_gen([(9, 1, 25, hD), (12, 1, 30, hD),
                             (CB_MED, 1, 10**9, hDbig), (CB_BIG, 1, 10**9, hDbig)])))

    # CB3 — count inversions (pairs i<j with a_i > a_j)
    ref = ("import sys\n"
           "d=sys.stdin.buffer.read().split()\n"
           "n=int(d[0]);a=[int(x) for x in d[1:1+n]]\n"
           "import bisect\n"
           "srt=sorted(set(a));comp={v:i+1 for i,v in enumerate(srt)}\n"
           "bit=[0]*(len(srt)+2);inv=0\n"
           "def upd(i):\n"
           "    while i<len(bit):bit[i]+=1;i+=i&-i\n"
           "def qry(i):\n"
           "    s=0\n"
           "    while i>0:s+=bit[i];i-=i&-i\n"
           "    return s\n"
           "seen=0\n"
           "for x in a:\n"
           "    r=comp[x];inv+=seen-qry(r);upd(r);seen+=1\n"
           "print(inv)\n")
    naive = ("import sys\n"
             "d=sys.stdin.buffer.read().split()\n"
             "n=int(d[0]);a=[int(x) for x in d[1:1+n]]\n"
             "c=0\n"
             "for i in range(n):\n"
             "    ai=a[i]\n"
             "    for j in range(i+1,n):\n"
             "        if ai>a[j]:c+=1\n"
             "print(c)\n")
    S.append(MintedTemplateV1(
        name="cb_count_inversions", family="inversion_count",
        mode=MODE_COMPLEXITY_BLIND, kind=KIND_PASSFAIL, float_tol=0.0,
        statement=(
            "Count inversions.\n\n"
            "Given a sequence of N integers, count the number of inversions: pairs "
            "(i, j) with i < j and a_i > a_j.\n\n"
            "Input: first line N; second line N integers.\nOutput: the number of "
            "inversions.\nConstraints: 1 <= N <= 100000, 1 <= a_i <= 10^9."),
        ref_source=ref, naive_source=naive, brute_source=naive,
        algo_sig="fenwick_bit", discriminator=DISC_TIMEOUT, brute_cap_tokens=4000,
        gen_public=_arr_gen([(4, 1, 9, None), (6, 1, 12, None), (7, 1, 20, None)]),
        gen_hidden=_arr_gen([(9, 1, 20, None), (12, 1, 30, None),
                             (CB_MED, 1, 10**9, None), (CB_BIG, 1, 10**9, None)])))

    # CB4 — maximum subarray sum (values may be negative)
    ref = ("import sys\n"
           "d=sys.stdin.buffer.read().split()\n"
           "n=int(d[0]);a=[int(x) for x in d[1:1+n]]\n"
           "best=cur=a[0]\n"
           "for x in a[1:]:\n"
           "    cur=x if cur<0 else cur+x\n"
           "    if cur>best:best=cur\n"
           "print(best)\n")
    naive = ("import sys\n"
             "d=sys.stdin.buffer.read().split()\n"
             "n=int(d[0]);a=[int(x) for x in d[1:1+n]]\n"
             "best=a[0]\n"
             "for i in range(n):\n"
             "    s=0\n"
             "    for j in range(i,n):\n"
             "        s+=a[j]\n"
             "        if s>best:best=s\n"
             "print(best)\n")
    neg = lambda r, n, a: [n]
    S.append(MintedTemplateV1(
        name="cb_max_subarray_sum", family="max_subarray",
        mode=MODE_COMPLEXITY_BLIND, kind=KIND_PASSFAIL, float_tol=0.0,
        statement=(
            "Maximum subarray sum.\n\n"
            "Given N integers (which may be negative), find the maximum possible sum of "
            "a non-empty contiguous subarray.\n\n"
            "Input: first line N; second line N integers.\nOutput: the maximum subarray "
            "sum.\nConstraints: 1 <= N <= 100000, -10^9 <= a_i <= 10^9."),
        ref_source=ref, naive_source=naive, brute_source=naive,
        algo_sig="kadane", discriminator=DISC_TIMEOUT, brute_cap_tokens=4000,
        gen_public=_arr_gen([(4, -9, 9, None), (6, -12, 12, None), (7, -15, 15, None)]),
        gen_hidden=_arr_gen([(9, -20, 20, None), (12, -30, 30, None),
                             (CB_MED, -10**9, 10**9, None),
                             (CB_BIG, -10**9, 10**9, None)])))

    # CB5 — number of subarrays whose sum equals K (values may be negative)
    ref = ("import sys\n"
           "from collections import defaultdict\n"
           "d=sys.stdin.buffer.read().split()\n"
           "n=int(d[0]);K=int(d[1]);a=[int(x) for x in d[2:2+n]]\n"
           "seen=defaultdict(int);seen[0]=1;s=0;c=0\n"
           "for x in a:\n"
           "    s+=x;c+=seen[s-K];seen[s]+=1\n"
           "print(c)\n")
    naive = ("import sys\n"
             "d=sys.stdin.buffer.read().split()\n"
             "n=int(d[0]);K=int(d[1]);a=[int(x) for x in d[2:2+n]]\n"
             "c=0\n"
             "for i in range(n):\n"
             "    s=0\n"
             "    for j in range(i,n):\n"
             "        s+=a[j]\n"
             "        if s==K:c+=1\n"
             "print(c)\n")
    hK = lambda r, n, a: [n, _T(r, -5, 5)]
    hKbig = lambda r, n, a: [n, _T(r, -3, 3)]
    S.append(MintedTemplateV1(
        name="cb_subarrays_sum_eq_k", family="subarray_sum_count",
        mode=MODE_COMPLEXITY_BLIND, kind=KIND_PASSFAIL, float_tol=0.0,
        statement=(
            "Subarrays with a given sum.\n\n"
            "Given N integers (possibly negative) and a target K, count the contiguous "
            "subarrays whose elements sum to exactly K.\n\n"
            "Input: first line N and K; second line N integers.\nOutput: the count.\n"
            "Constraints: 1 <= N <= 100000, -10^4 <= a_i <= 10^4, |K| <= 10^9."),
        ref_source=ref, naive_source=naive, brute_source=naive,
        algo_sig="prefix_hashmap", discriminator=DISC_TIMEOUT, brute_cap_tokens=4000,
        gen_public=_arr_gen([(4, -4, 4, hK), (6, -4, 4, hK), (7, -5, 5, hK)]),
        gen_hidden=_arr_gen([(9, -5, 5, hK), (12, -6, 6, hK),
                             (CB_MED, -3, 3, hKbig), (CB_BIG, -3, 3, hKbig)])))

    # CB6 — longest contiguous subarray with sum <= S (non-negative array)
    ref = ("import sys\n"
           "d=sys.stdin.buffer.read().split()\n"
           "n=int(d[0]);S=int(d[1]);a=[int(x) for x in d[2:2+n]]\n"
           "i=0;s=0;best=0\n"
           "for j in range(n):\n"
           "    s+=a[j]\n"
           "    while s>S:s-=a[i];i+=1\n"
           "    if j-i+1>best:best=j-i+1\n"
           "print(best)\n")
    naive = ("import sys\n"
             "d=sys.stdin.buffer.read().split()\n"
             "n=int(d[0]);S=int(d[1]);a=[int(x) for x in d[2:2+n]]\n"
             "best=0\n"
             "for i in range(n):\n"
             "    s=0\n"
             "    for j in range(i,n):\n"
             "        s+=a[j]\n"
             "        if s<=S and j-i+1>best:best=j-i+1\n"
             "print(best)\n")
    hS = lambda r, n, a: [n, _T(r, 5, 40)]
    hSbig = lambda r, n, a: [n, _T(r, 10**6, 5 * 10**6)]
    S.append(MintedTemplateV1(
        name="cb_longest_subarray_sum_le_s", family="bounded_window_length",
        mode=MODE_COMPLEXITY_BLIND, kind=KIND_PASSFAIL, float_tol=0.0,
        statement=(
            "Longest bounded-sum window.\n\n"
            "Given N non-negative integers and a bound S, find the maximum length of a "
            "contiguous subarray whose sum is at most S (0 if none).\n\n"
            "Input: first line N and S; second line N integers.\nOutput: the maximum "
            "length.\nConstraints: 1 <= N <= 100000, 0 <= a_i <= 10^4, 0 <= S <= 10^9."),
        ref_source=ref, naive_source=naive, brute_source=naive,
        algo_sig="two_pointer_window", discriminator=DISC_TIMEOUT, brute_cap_tokens=4000,
        gen_public=_arr_gen([(4, 1, 10, hS), (6, 1, 12, hS), (7, 1, 15, hS)]),
        gen_hidden=_arr_gen([(9, 1, 15, hS), (12, 1, 20, hS),
                             (CB_MED, 0, 10**4, hSbig), (CB_BIG, 0, 10**4, hSbig)])))

    # CB7 — sum over all windows of size W of (number of distinct values)
    ref = ("import sys\n"
           "from collections import defaultdict\n"
           "d=sys.stdin.buffer.read().split()\n"
           "n=int(d[0]);W=int(d[1]);a=[int(x) for x in d[2:2+n]]\n"
           "cnt=defaultdict(int);distinct=0;total=0\n"
           "for j in range(n):\n"
           "    if cnt[a[j]]==0:distinct+=1\n"
           "    cnt[a[j]]+=1\n"
           "    if j>=W:\n"
           "        o=a[j-W];cnt[o]-=1\n"
           "        if cnt[o]==0:distinct-=1\n"
           "    if j>=W-1:total+=distinct\n"
           "print(total)\n")
    naive = ("import sys\n"
             "d=sys.stdin.buffer.read().split()\n"
             "n=int(d[0]);W=int(d[1]);a=[int(x) for x in d[2:2+n]]\n"
             "total=0\n"
             "for i in range(n-W+1):\n"
             "    total+=len(set(a[i:i+W]))\n"
             "print(total)\n")
    hW = lambda r, n, a: [n, max(1, n // 2)]
    S.append(MintedTemplateV1(
        name="cb_distinct_in_windows", family="window_distinct_sum",
        mode=MODE_COMPLEXITY_BLIND, kind=KIND_PASSFAIL, float_tol=0.0,
        statement=(
            "Distinct values over sliding windows.\n\n"
            "Given N integers and a window width W, for every contiguous window of "
            "length W compute the number of distinct values it contains, and output the "
            "sum of these counts over all windows.\n\n"
            "Input: first line N and W; second line N integers.\nOutput: the total.\n"
            "Constraints: 1 <= W <= N <= 100000, 1 <= a_i <= 10^9."),
        ref_source=ref, naive_source=naive, brute_source=naive,
        algo_sig="sliding_window_counts", discriminator=DISC_TIMEOUT, brute_cap_tokens=4000,
        gen_public=_arr_gen([(4, 1, 4, hW), (6, 1, 5, hW), (8, 1, 6, hW)]),
        gen_hidden=_arr_gen([(10, 1, 6, hW), (16, 1, 8, hW),
                             (CB_MED, 1, 50, hW), (CB_BIG, 1, 1000, hW)])))

    # CB8 — for each i, distance to nearest j<i with a_j < a_i; output the sum of distances
    ref = ("import sys\n"
           "d=sys.stdin.buffer.read().split()\n"
           "n=int(d[0]);a=[int(x) for x in d[1:1+n]]\n"
           "st=[];tot=0\n"
           "for i in range(n):\n"
           "    while st and a[st[-1]]>=a[i]:st.pop()\n"
           "    if st:tot+=i-st[-1]\n"
           "    st.append(i)\n"
           "print(tot)\n")
    naive = ("import sys\n"
             "d=sys.stdin.buffer.read().split()\n"
             "n=int(d[0]);a=[int(x) for x in d[1:1+n]]\n"
             "tot=0\n"
             "for i in range(n):\n"
             "    for j in range(i-1,-1,-1):\n"
             "        if a[j]<a[i]:tot+=i-j;break\n"
             "print(tot)\n")
    S.append(MintedTemplateV1(
        name="cb_nearest_smaller_left", family="monotonic_stack",
        mode=MODE_COMPLEXITY_BLIND, kind=KIND_PASSFAIL, float_tol=0.0,
        statement=(
            "Nearest smaller element on the left.\n\n"
            "For each position i (1-indexed), let p(i) be the largest j < i with "
            "a_j < a_i, or absent if none. Output the sum of (i - p(i)) over all i that "
            "have such a j.\n\n"
            "Input: first line N; second line N integers.\nOutput: the sum of "
            "distances.\nConstraints: 1 <= N <= 100000, 1 <= a_i <= 10^9."),
        ref_source=ref, naive_source=naive, brute_source=naive,
        algo_sig="monotonic_stack", discriminator=DISC_TIMEOUT, brute_cap_tokens=4000,
        gen_public=_arr_gen([(4, 1, 9, None), (6, 1, 12, None), (7, 1, 15, None)]),
        gen_hidden=_hid_nearest_smaller))

    # CB9 — count pairs i<j with a_i + a_j EXACTLY equal to T
    ref = ("import sys\n"
           "from collections import defaultdict\n"
           "d=sys.stdin.buffer.read().split()\n"
           "n=int(d[0]);T=int(d[1]);a=[int(x) for x in d[2:2+n]]\n"
           "seen=defaultdict(int);c=0\n"
           "for x in a:\n"
           "    c+=seen[T-x];seen[x]+=1\n"
           "print(c)\n")
    naive = ("import sys\n"
             "d=sys.stdin.buffer.read().split()\n"
             "n=int(d[0]);T=int(d[1]);a=[int(x) for x in d[2:2+n]]\n"
             "c=0\n"
             "for i in range(n):\n"
             "    for j in range(i+1,n):\n"
             "        if a[i]+a[j]==T:c+=1\n"
             "print(c)\n")
    hTe = lambda r, n, a: [n, _T(r, 4, 30)]
    hTeBig = lambda r, n, a: [n, _T(r, 2, 40)]
    S.append(MintedTemplateV1(
        name="cb_pairs_sum_eq_t", family="pair_exact_sum_count",
        mode=MODE_COMPLEXITY_BLIND, kind=KIND_PASSFAIL, float_tol=0.0,
        statement=(
            "Count exact-sum pairs.\n\n"
            "Given N integers and a target T, count the pairs (i, j), i < j, with "
            "a_i + a_j exactly equal to T.\n\n"
            "Input: first line N and T; second line N integers.\nOutput: the count.\n"
            "Constraints: 1 <= N <= 100000, 1 <= a_i <= 20, 2 <= T <= 40."),
        ref_source=ref, naive_source=naive, brute_source=naive,
        algo_sig="hashmap_complement", discriminator=DISC_TIMEOUT, brute_cap_tokens=4000,
        gen_public=_arr_gen([(4, 1, 15, hTe), (6, 1, 18, hTe), (7, 1, 20, hTe)]),
        gen_hidden=_arr_gen([(9, 1, 20, hTe), (12, 1, 20, hTe),
                             (CB_MED, 1, 20, hTeBig), (CB_BIG, 1, 20, hTeBig)])))

    # ============================================================ HIDDEN_EDGE (8)

    # HE1 — circular maximum subarray sum (wraparound allowed)
    ref = ("import sys\n"
           "d=sys.stdin.buffer.read().split()\n"
           "n=int(d[0]);a=[int(x) for x in d[1:1+n]]\n"
           "tot=0;mx=cur_mx=a[0];mn=cur_mn=a[0]\n"
           "for x in a:tot+=x\n"
           "cur_mx=a[0];cur_mn=a[0];mx=a[0];mn=a[0]\n"
           "for x in a[1:]:\n"
           "    cur_mx=x if cur_mx<0 else cur_mx+x\n"
           "    if cur_mx>mx:mx=cur_mx\n"
           "    cur_mn=x if cur_mn>0 else cur_mn+x\n"
           "    if cur_mn<mn:mn=cur_mn\n"
           "print(mx if mx<0 else max(mx,tot-mn))\n")
    naive = ("import sys\n"
             "d=sys.stdin.buffer.read().split()\n"
             "n=int(d[0]);a=[int(x) for x in d[1:1+n]]\n"
             "best=cur=a[0]\n"
             "for x in a[1:]:\n"
             "    cur=x if cur<0 else cur+x\n"
             "    if cur>best:best=cur\n"
             "print(best)\n")
    brute = ("import sys\n"
             "d=sys.stdin.buffer.read().split()\n"
             "n=int(d[0]);a=[int(x) for x in d[1:1+n]]\n"
             "best=a[0]\n"
             "for i in range(n):\n"
             "    s=0\n"
             "    for L in range(1,n+1):\n"
             "        s+=a[(i+L-1)%n]\n"
             "        if s>best:best=s\n"
             "print(best)\n")

    def _pub_he1(rng):
        out = []
        for n in (4, 5, 6):
            # naive-safe: best window does NOT wrap -> make a strong positive interior run
            a = [-(rng.randint(1, 9)) for _ in range(n)]
            i = rng.randint(0, n - 2)
            a[i] = rng.randint(20, 40); a[i + 1] = rng.randint(20, 40)
            out.append(_case([n], a))
        return out

    def _hid_he1(rng):
        out = []
        # trap: wrap-around optimal -> big positives at the two ends, dip in the middle
        for n in (5, 6, 7):
            a = [rng.randint(8, 15) for _ in range(n)]
            mid = n // 2
            a[mid] = -(rng.randint(30, 50))
            out.append(_case([n], a))
        # trap: all-negative (answer is the single max element, never wrap)
        for n in (4, 5):
            a = [-(rng.randint(1, 20)) for _ in range(n)]
            out.append(_case([n], a))
        return out
    S.append(MintedTemplateV1(
        name="he_circular_max_subarray", family="circular_max_subarray",
        mode=MODE_HIDDEN_EDGE, kind=KIND_PASSFAIL, float_tol=0.0,
        statement=(
            "Maximum circular subarray sum.\n\n"
            "Given N integers arranged in a CIRCLE, find the maximum sum of a non-empty "
            "contiguous block. A block may wrap around from the end of the array to the "
            "beginning.\n\n"
            "Input: first line N; second line N integers.\nOutput: the maximum circular "
            "subarray sum.\nConstraints: 1 <= N <= 100000, -10^4 <= a_i <= 10^4."),
        ref_source=ref, naive_source=naive, brute_source=brute,
        algo_sig="circular_kadane", discriminator=DISC_OUTPUT_MISMATCH,
        brute_cap_tokens=400, gen_public=_pub_he1, gen_hidden=_hid_he1))

    # HE2 — total length of the union of intervals
    ref = ("import sys\n"
           "d=sys.stdin.buffer.read().split()\n"
           "n=int(d[0]);iv=[]\n"
           "k=1\n"
           "for _ in range(n):\n"
           "    l=int(d[k]);r=int(d[k+1]);k+=2;iv.append((l,r))\n"
           "iv.sort();tot=0;cl=cr=None\n"
           "for l,r in iv:\n"
           "    if cr is None:cl,cr=l,r\n"
           "    elif l>cr:tot+=cr-cl;cl,cr=l,r\n"
           "    else:cr=max(cr,r)\n"
           "if cr is not None:tot+=cr-cl\n"
           "print(tot)\n")
    naive = ("import sys\n"
             "d=sys.stdin.buffer.read().split()\n"
             "n=int(d[0]);tot=0;k=1\n"
             "for _ in range(n):\n"
             "    l=int(d[k]);r=int(d[k+1]);k+=2;tot+=r-l\n"
             "print(tot)\n")
    brute = ("import sys\n"
             "d=sys.stdin.buffer.read().split()\n"
             "n=int(d[0]);pts=set();k=1\n"
             "for _ in range(n):\n"
             "    l=int(d[k]);r=int(d[k+1]);k+=2\n"
             "    for x in range(l,r):pts.add(x)\n"
             "print(len(pts))\n")

    def _pub_he2(rng):
        out = []
        for n in (3, 4):
            # disjoint intervals -> naive (sum of lengths) is correct
            iv = []; cur = rng.randint(0, 3)
            for _ in range(n):
                l = cur + rng.randint(1, 3); r = l + rng.randint(1, 4)
                iv.append((l, r)); cur = r + rng.randint(1, 3)
            flat = [x for p in iv for x in p]
            out.append(_case([n], flat))
        return out

    def _hid_he2(rng):
        out = []
        for n in (3, 4, 5):
            # overlapping/nested -> naive double counts
            iv = []; base = rng.randint(0, 5)
            for _ in range(n):
                l = base + rng.randint(0, 4); r = l + rng.randint(3, 8)
                iv.append((l, r))
            flat = [x for p in iv for x in p]
            out.append(_case([n], flat))
        return out
    S.append(MintedTemplateV1(
        name="he_interval_union_length", family="interval_union",
        mode=MODE_HIDDEN_EDGE, kind=KIND_PASSFAIL, float_tol=0.0,
        statement=(
            "Total covered length.\n\n"
            "You are given N half-open intervals [l, r). Output the total length of "
            "their union (overlapping regions are counted once).\n\n"
            "Input: first line N; then N lines each with two integers l and r (l < r).\n"
            "Output: the length of the union.\nConstraints: 1 <= N <= 100000, "
            "0 <= l < r <= 10^9."),
        ref_source=ref, naive_source=naive, brute_source=brute,
        algo_sig="sort_sweep", discriminator=DISC_OUTPUT_MISMATCH,
        brute_cap_tokens=400, gen_public=_pub_he2, gen_hidden=_hid_he2))

    # HE3 — balanced brackets with multiple types
    ref = ("import sys\n"
           "s=sys.stdin.buffer.read().decode().strip()\n"
           "m={')':'(',']':'[','}':'{'};st=[];ok=True\n"
           "for ch in s:\n"
           "    if ch in '([{':st.append(ch)\n"
           "    elif ch in ')]}':\n"
           "        if not st or st.pop()!=m[ch]:ok=False;break\n"
           "print('YES' if ok and not st else 'NO')\n")
    naive = ("import sys\n"
             "s=sys.stdin.buffer.read().decode().strip()\n"
             "op=sum(1 for c in s if c in '([{')\n"
             "cl=sum(1 for c in s if c in ')]}')\n"
             "print('YES' if op==cl else 'NO')\n")
    brute = ref  # the stack solution is the obviously-correct oracle for strings

    _PAIRS = [("(", ")"), ("[", "]"), ("{", "}")]

    def _dyck(rng, npairs):
        s = ""; bal = 0; ro = npairs; rc = npairs; stack = []
        while ro or rc:
            if ro and (bal == 0 or rng.random() < 0.5):
                t = rng.choice(_PAIRS); s += t[0]; stack.append(t[1]); bal += 1; ro -= 1
            else:
                s += stack.pop(); bal -= 1; rc -= 1
        return s

    def _pub_he3(rng):
        out = []
        for npairs in (1, 2, 3, 4):
            out.append(_dyck(rng, npairs))            # truly balanced -> YES, naive YES
        for _ in range(2):                            # extra opens -> NO, naive NO
            s = _dyck(rng, rng.randint(1, 3)) + rng.choice("([{")
            out.append(s)
        return out

    def _hid_he3(rng):
        out = []
        for _ in range(4):
            t1, t2 = rng.sample(_PAIRS, 2)
            out.append(t1[0] + t2[0] + t1[1] + t2[1])   # "([)]" interleave -> NO, naive YES
        for _ in range(2):
            t = rng.choice(_PAIRS)
            out.append(t[1] + t[0])                      # ")(" -> NO, naive YES
        out.append(_dyck(rng, 3))                        # a true YES for balance
        return out
    S.append(MintedTemplateV1(
        name="he_balanced_brackets", family="bracket_matching",
        mode=MODE_HIDDEN_EDGE, kind=KIND_PASSFAIL, float_tol=0.0,
        statement=(
            "Balanced bracket sequence.\n\n"
            "A string consists only of the bracket characters ( ) [ ] { }. Decide "
            "whether it is balanced: every opening bracket is closed by the matching "
            "type in the correct order.\n\n"
            "Input: a single line containing the bracket string (it may be empty).\n"
            "Output: YES if the string is balanced, otherwise NO.\nConstraints: the "
            "string length is at most 100000."),
        ref_source=ref, naive_source=naive, brute_source=brute,
        algo_sig="stack_matching", discriminator=DISC_OUTPUT_MISMATCH,
        brute_cap_tokens=400, gen_public=_pub_he3, gen_hidden=_hid_he3))

    # HE4 — count values within an INCLUSIVE range [L, R]
    ref = ("import sys\n"
           "d=sys.stdin.buffer.read().split()\n"
           "n=int(d[0]);L=int(d[1]);R=int(d[2]);a=[int(x) for x in d[3:3+n]]\n"
           "print(sum(1 for x in a if L<=x<=R))\n")
    naive = ("import sys\n"
             "d=sys.stdin.buffer.read().split()\n"
             "n=int(d[0]);L=int(d[1]);R=int(d[2]);a=[int(x) for x in d[3:3+n]]\n"
             "print(sum(1 for x in a if L<=x<R))\n")
    brute = ref

    def _pub_he4(rng):
        out = []
        for n in (5, 6, 7):
            L = rng.randint(1, 5); R = L + rng.randint(3, 8)
            # naive-safe: no element equals R (so inclusive vs exclusive upper agree)
            a = [rng.choice([rng.randint(L, R - 1), rng.randint(R + 1, R + 10),
                             rng.randint(0, L - 1) if L > 0 else 0]) for _ in range(n)]
            out.append(_case([n, L, R], a))
        return out

    def _hid_he4(rng):
        out = []
        for n in (5, 6, 7):
            L = rng.randint(1, 5); R = L + rng.randint(3, 8)
            a = [rng.choice([R, R, rng.randint(L, R), rng.randint(R + 1, R + 5)])
                 for _ in range(n)]           # several equal exactly R -> off-by-one bites
            if R not in a:
                a[0] = R
            out.append(_case([n, L, R], a))
        return out
    S.append(MintedTemplateV1(
        name="he_count_in_range_inclusive", family="inclusive_range_count",
        mode=MODE_HIDDEN_EDGE, kind=KIND_PASSFAIL, float_tol=0.0,
        statement=(
            "Count values in an inclusive range.\n\n"
            "Given N integers and bounds L and R, count how many values x satisfy "
            "L <= x <= R (both endpoints included).\n\n"
            "Input: first line N, L, R; second line N integers.\nOutput: the count.\n"
            "Constraints: 1 <= N <= 100000, 0 <= L <= R <= 10^9."),
        ref_source=ref, naive_source=naive, brute_source=brute,
        algo_sig="inclusive_filter", discriminator=DISC_OUTPUT_MISMATCH,
        brute_cap_tokens=400, gen_public=_pub_he4, gen_hidden=_hid_he4))

    # HE5 — most frequent value, ties broken by SMALLEST value
    ref = ("import sys\n"
           "from collections import Counter\n"
           "d=sys.stdin.buffer.read().split()\n"
           "n=int(d[0]);a=[int(x) for x in d[1:1+n]]\n"
           "c=Counter(a);best=max(c.values())\n"
           "print(min(v for v,k in c.items() if k==best))\n")
    naive = ("import sys\n"
             "from collections import Counter\n"
             "d=sys.stdin.buffer.read().split()\n"
             "n=int(d[0]);a=[int(x) for x in d[1:1+n]]\n"
             "c=Counter(a)\n"
             "print(c.most_common(1)[0][0])\n")
    brute = ("import sys\n"
             "d=sys.stdin.buffer.read().split()\n"
             "n=int(d[0]);a=[int(x) for x in d[1:1+n]]\n"
             "vals=sorted(set(a));best=-1;ans=None\n"
             "for v in vals:\n"
             "    k=a.count(v)\n"
             "    if k>best:best=k;ans=v\n"
             "print(ans)\n")

    def _pub_he5(rng):
        out = []
        for n in (5, 6, 7):
            # unique mode: one value repeated strictly more than any other
            base = rng.randint(1, 9); a = [base] * 3 + [rng.randint(10, 20) for _ in range(n - 3)]
            rng.shuffle(a)
            out.append(_case([n], a))
        return out

    def _hid_he5(rng):
        out = []
        for _ in range(4):
            # tie at the top: a LARGER value appears first; smallest must win
            small = rng.randint(1, 5); big = small + rng.randint(5, 15)
            a = [big, big, small, small] + [rng.randint(20, 30)]
            rng.shuffle(a)
            out.append(_case([len(a)], a))
        return out
    S.append(MintedTemplateV1(
        name="he_mode_smallest_tiebreak", family="mode_tiebreak",
        mode=MODE_HIDDEN_EDGE, kind=KIND_PASSFAIL, float_tol=0.0,
        statement=(
            "Most frequent value, smallest on ties.\n\n"
            "Given N integers, output the value that occurs most often. If several "
            "values share the highest frequency, output the SMALLEST of them.\n\n"
            "Input: first line N; second line N integers.\nOutput: the answer value.\n"
            "Constraints: 1 <= N <= 100000, 1 <= a_i <= 10^9."),
        ref_source=ref, naive_source=naive, brute_source=brute,
        algo_sig="count_min_tiebreak", discriminator=DISC_OUTPUT_MISMATCH,
        brute_cap_tokens=400, gen_public=_pub_he5, gen_hidden=_hid_he5))

    # HE6 — second-largest DISTINCT value
    ref = ("import sys\n"
           "d=sys.stdin.buffer.read().split()\n"
           "n=int(d[0]);a=sorted(set(int(x) for x in d[1:1+n]),reverse=True)\n"
           "print(a[1] if len(a)>=2 else 'NONE')\n")
    naive = ("import sys\n"
             "d=sys.stdin.buffer.read().split()\n"
             "n=int(d[0]);a=sorted((int(x) for x in d[1:1+n]),reverse=True)\n"
             "print(a[1] if n>=2 else 'NONE')\n")
    brute = ref

    def _pub_he6(rng):
        out = []
        for n in (4, 5, 6):
            a = rng.sample(range(1, 50), n)        # all distinct -> naive==ref
            out.append(_case([n], a))
        return out

    def _hid_he6(rng):
        out = []
        for _ in range(4):
            mx = rng.randint(20, 40); sec = rng.randint(1, mx - 1)
            a = [mx, mx] + [sec] + [rng.randint(1, sec) for _ in range(2)]  # duplicated max
            rng.shuffle(a)
            out.append(_case([len(a)], a))
        return out
    S.append(MintedTemplateV1(
        name="he_second_largest_distinct", family="second_distinct",
        mode=MODE_HIDDEN_EDGE, kind=KIND_PASSFAIL, float_tol=0.0,
        statement=(
            "Second largest distinct value.\n\n"
            "Given N integers, output the second largest DISTINCT value. If there are "
            "fewer than two distinct values, output NONE.\n\n"
            "Input: first line N; second line N integers.\nOutput: the value, or NONE.\n"
            "Constraints: 1 <= N <= 100000, 1 <= a_i <= 10^9."),
        ref_source=ref, naive_source=naive, brute_source=brute,
        algo_sig="distinct_sort", discriminator=DISC_OUTPUT_MISMATCH,
        brute_cap_tokens=400, gen_public=_pub_he6, gen_hidden=_hid_he6))

    # HE7 — maximum gap between consecutive elements after sorting
    ref = ("import sys\n"
           "d=sys.stdin.buffer.read().split()\n"
           "n=int(d[0]);a=sorted(int(x) for x in d[1:1+n])\n"
           "print(max((a[i+1]-a[i] for i in range(n-1)),default=0))\n")
    naive = ("import sys\n"
             "d=sys.stdin.buffer.read().split()\n"
             "n=int(d[0]);a=[int(x) for x in d[1:1+n]]\n"
             "print(max((a[i+1]-a[i] for i in range(n-1)),default=0))\n")
    brute = ref

    def _pub_he7(rng):
        out = []
        for n in (4, 5, 6):
            a = sorted(rng.sample(range(0, 80), n))   # already sorted -> naive==ref
            out.append(_case([n], a))
        return out

    def _hid_he7(rng):
        out = []
        for n in (5, 6, 7):
            a = rng.sample(range(0, 200), n)          # unsorted -> naive (no sort) wrong
            # ensure not accidentally sorted
            if a == sorted(a):
                a[0], a[-1] = a[-1], a[0]
            out.append(_case([n], a))
        return out
    S.append(MintedTemplateV1(
        name="he_max_gap_sorted", family="max_adjacent_gap",
        mode=MODE_HIDDEN_EDGE, kind=KIND_PASSFAIL, float_tol=0.0,
        statement=(
            "Maximum gap after sorting.\n\n"
            "Given N integers, sort them in non-decreasing order and output the maximum "
            "difference between two consecutive elements (0 if N < 2).\n\n"
            "Input: first line N; second line N integers.\nOutput: the maximum gap.\n"
            "Constraints: 1 <= N <= 100000, 0 <= a_i <= 10^9."),
        ref_source=ref, naive_source=naive, brute_source=brute,
        algo_sig="sort_adjacent_diff", discriminator=DISC_OUTPUT_MISMATCH,
        brute_cap_tokens=400, gen_public=_pub_he7, gen_hidden=_hid_he7))

    # HE8 — number of distinct ABSOLUTE values
    ref = ("import sys\n"
           "d=sys.stdin.buffer.read().split()\n"
           "n=int(d[0]);print(len({abs(int(x)) for x in d[1:1+n]}))\n")
    naive = ("import sys\n"
             "d=sys.stdin.buffer.read().split()\n"
             "n=int(d[0]);print(len({int(x) for x in d[1:1+n]}))\n")
    brute = ref

    def _pub_he8(rng):
        out = []
        for n in (4, 5, 6):
            a = [rng.randint(1, 40) for _ in range(n)]   # all positive -> abs is identity
            out.append(_case([n], a))
        return out

    def _hid_he8(rng):
        out = []
        for n in (5, 6, 7):
            base = [rng.randint(1, 20) for _ in range(n // 2 + 1)]
            a = base + [-x for x in base[:n - len(base)]]   # +/- pairs collapse under abs
            rng.shuffle(a)
            out.append(_case([len(a)], a))
        return out
    S.append(MintedTemplateV1(
        name="he_distinct_abs_values", family="distinct_abs",
        mode=MODE_HIDDEN_EDGE, kind=KIND_PASSFAIL, float_tol=0.0,
        statement=(
            "Distinct magnitudes.\n\n"
            "Given N integers (which may be negative), output the number of distinct "
            "ABSOLUTE values among them.\n\n"
            "Input: first line N; second line N integers.\nOutput: the count of distinct "
            "absolute values.\nConstraints: 1 <= N <= 100000, -10^9 <= a_i <= 10^9."),
        ref_source=ref, naive_source=naive, brute_source=brute,
        algo_sig="abs_set", discriminator=DISC_OUTPUT_MISMATCH,
        brute_cap_tokens=400, gen_public=_pub_he8, gen_hidden=_hid_he8))

    # ============================================================ WRONG_ALGORITHM (8)

    # WA1 — minimum coins (greedy fails; DP required)
    ref = ("import sys\n"
           "d=sys.stdin.buffer.read().split()\n"
           "c=int(d[0]);N=int(d[1]);coins=[int(x) for x in d[2:2+c]]\n"
           "INF=float('inf');dp=[0]+[INF]*N\n"
           "for v in range(1,N+1):\n"
           "    for co in coins:\n"
           "        if co<=v and dp[v-co]+1<dp[v]:dp[v]=dp[v-co]+1\n"
           "print(dp[N] if dp[N]!=INF else -1)\n")
    naive = ("import sys\n"
             "d=sys.stdin.buffer.read().split()\n"
             "c=int(d[0]);N=int(d[1]);coins=sorted((int(x) for x in d[2:2+c]),reverse=True)\n"
             "cnt=0;rem=N\n"
             "for co in coins:\n"
             "    if co<=rem:cnt+=rem//co;rem%=co\n"
             "print(cnt if rem==0 else -1)\n")
    brute = ("import sys\n"
             "from collections import deque\n"
             "d=sys.stdin.buffer.read().split()\n"
             "c=int(d[0]);N=int(d[1]);coins=[int(x) for x in d[2:2+c]]\n"
             "seen=[False]*(N+1);seen[0]=True;q=deque([0]);dist=[0]*(N+1);ans=-1\n"
             "while q:\n"
             "    amt=q.popleft()\n"
             "    if amt==N:ans=dist[amt];break\n"
             "    for co in coins:\n"
             "        na=amt+co\n"
             "        if na<=N and not seen[na]:seen[na]=True;dist[na]=dist[amt]+1;q.append(na)\n"
             "print(ans)\n")

    def _pub_wa1(rng):
        out = []
        for _ in range(3):
            coins = [1, 5, 10, 25]              # canonical -> greedy optimal
            N = rng.randint(1, 80)
            out.append(_case([len(coins), N], coins))
        return out

    def _hid_wa1(rng):
        out = []
        for _ in range(4):
            a = rng.randint(3, 25)
            coins = [1, a, a + 1]               # non-canonical: target 2a breaks greedy
            N = 2 * a
            out.append(_case([len(coins), N], coins))
        return out
    S.append(MintedTemplateV1(
        name="wa_min_coins", family="coin_change_min",
        mode=MODE_WRONG_ALGORITHM, kind=KIND_PASSFAIL, float_tol=0.0,
        statement=(
            "Fewest coins.\n\n"
            "You have C coin denominations (unlimited supply of each) and must make an "
            "amount N. Output the minimum number of coins, or -1 if it is impossible.\n\n"
            "Input: first line C and N; second line the C denominations.\nOutput: the "
            "minimum coin count, or -1.\nConstraints: 1 <= C <= 20, 1 <= N <= 20000, "
            "1 <= denomination <= N."),
        ref_source=ref, naive_source=naive, brute_source=brute,
        algo_sig="dp_unbounded_min", discriminator=DISC_OUTPUT_MISMATCH,
        brute_cap_tokens=400, gen_public=_pub_wa1, gen_hidden=_hid_wa1))

    # WA2 — maximum non-adjacent sum (greedy by parity fails; DP required)
    ref = ("import sys\n"
           "d=sys.stdin.buffer.read().split()\n"
           "n=int(d[0]);a=[int(x) for x in d[1:1+n]]\n"
           "inc=0;exc=0\n"
           "for x in a:inc,exc=exc+x,max(inc,exc)\n"
           "print(max(inc,exc))\n")
    naive = ("import sys\n"
             "d=sys.stdin.buffer.read().split()\n"
             "n=int(d[0]);a=[int(x) for x in d[1:1+n]]\n"
             "print(max(sum(a[0::2]),sum(a[1::2])))\n")
    brute = ("import sys\n"
             "d=sys.stdin.buffer.read().split()\n"
             "n=int(d[0]);a=[int(x) for x in d[1:1+n]]\n"
             "best=0\n"
             "for m in range(1<<n):\n"
             "    if m&(m>>1):continue\n"
             "    s=sum(a[i] for i in range(n) if m>>i&1)\n"
             "    if s>best:best=s\n"
             "print(best)\n")

    def _pub_wa2(rng):
        out = []
        for n in (5, 6, 7):
            # big on even indices, tiny on odd -> taking all evens (a parity) is optimal
            a = [rng.randint(20, 40) if i % 2 == 0 else rng.randint(1, 3)
                 for i in range(n)]
            out.append(_case([n], a))
        return out

    def _hid_wa2(rng):
        out = []
        for _ in range(4):
            x = rng.randint(20, 40); y = rng.randint(1, 5)
            a = [x, y, y, x]                    # optimal = idx0+idx3 = 2x; parity gives x+y
            out.append(_case([len(a)], a))
        for _ in range(2):
            x = rng.randint(15, 30); y = rng.randint(1, 4)
            a = [x, y, y, y, x]
            out.append(_case([len(a)], a))
        return out
    S.append(MintedTemplateV1(
        name="wa_max_nonadjacent_sum", family="max_independent_subset",
        mode=MODE_WRONG_ALGORITHM, kind=KIND_PASSFAIL, float_tol=0.0,
        statement=(
            "Maximum non-adjacent sum.\n\n"
            "Given N non-negative integers in a row, choose a subset with no two chosen "
            "elements adjacent, maximizing the sum. Output that maximum sum.\n\n"
            "Input: first line N; second line N integers.\nOutput: the maximum sum.\n"
            "Constraints: 1 <= N <= 100000, 0 <= a_i <= 10^4."),
        ref_source=ref, naive_source=naive, brute_source=brute,
        algo_sig="dp_house_robber", discriminator=DISC_OUTPUT_MISMATCH,
        brute_cap_tokens=400, gen_public=_pub_wa2, gen_hidden=_hid_wa2))

    # WA3 — weighted interval scheduling (greedy-by-earliest-finish fails; DP required)
    ref = ("import sys,bisect\n"
           "d=sys.stdin.buffer.read().split()\n"
           "n=int(d[0]);iv=[];k=1\n"
           "for _ in range(n):\n"
           "    s=int(d[k]);e=int(d[k+1]);w=int(d[k+2]);k+=3;iv.append((e,s,w))\n"
           "iv.sort();E=[x[0] for x in iv];dp=[0]*(n+1)\n"
           "for i in range(1,n+1):\n"
           "    e,s,w=iv[i-1];j=bisect.bisect_right(E,s,0,i-1)\n"
           "    dp[i]=max(dp[i-1],w+dp[j])\n"
           "print(dp[n])\n")
    naive = ("import sys\n"
             "d=sys.stdin.buffer.read().split()\n"
             "n=int(d[0]);iv=[];k=1\n"
             "for _ in range(n):\n"
             "    s=int(d[k]);e=int(d[k+1]);w=int(d[k+2]);k+=3;iv.append((e,s,w))\n"
             "iv.sort();last=-1;tot=0\n"
             "for e,s,w in iv:\n"
             "    if s>=last:tot+=w;last=e\n"
             "print(tot)\n")
    brute = ("import sys\n"
             "d=sys.stdin.buffer.read().split()\n"
             "n=int(d[0]);iv=[];k=1\n"
             "for _ in range(n):\n"
             "    s=int(d[k]);e=int(d[k+1]);w=int(d[k+2]);k+=3;iv.append((s,e,w))\n"
             "best=0\n"
             "for m in range(1<<n):\n"
             "    sel=sorted(iv[i] for i in range(n) if m>>i&1)\n"
             "    ok=all(sel[t][0]>=sel[t-1][1] for t in range(1,len(sel)))\n"
             "    if ok:\n"
             "        s=sum(x[2] for x in sel)\n"
             "        if s>best:best=s\n"
             "print(best)\n")

    def _pub_wa3(rng):
        out = []
        for _ in range(3):
            # equal weights -> greedy (max count) is also max weight
            n = rng.randint(3, 5); rows = []; t = 0
            for _ in range(n):
                s = t + rng.randint(0, 2); e = s + rng.randint(1, 3)
                rows.append((s, e, 1)); t = e
            flat = [x for r in rows for x in r]
            out.append(_case([n], flat))
        return out

    def _hid_wa3(rng):
        out = []
        for _ in range(4):
            L = rng.randint(5, 8); W = L + rng.randint(3, 8)
            rows = [(0, L, W)] + [(i, i + 1, 1) for i in range(L)]   # one fat vs many unit
            n = len(rows)
            flat = [x for r in rows for x in r]
            out.append(_case([n], flat))
        return out
    S.append(MintedTemplateV1(
        name="wa_weighted_interval_scheduling", family="weighted_scheduling",
        mode=MODE_WRONG_ALGORITHM, kind=KIND_PASSFAIL, float_tol=0.0,
        statement=(
            "Weighted interval scheduling.\n\n"
            "You are given N jobs; job k occupies the half-open time interval [s_k, e_k) "
            "and pays w_k. Choose a set of pairwise non-overlapping jobs maximizing total "
            "pay. Output that maximum total pay.\n\n"
            "Input: first line N; then N lines each with s, e, w.\nOutput: the maximum "
            "total weight.\nConstraints: 1 <= N <= 2000, 0 <= s < e <= 10^9, "
            "1 <= w <= 10^9."),
        ref_source=ref, naive_source=naive, brute_source=brute,
        algo_sig="dp_binary_search", discriminator=DISC_OUTPUT_MISMATCH,
        brute_cap_tokens=400, gen_public=_pub_wa3, gen_hidden=_hid_wa3))

    # WA4 — minimum partition difference (greedy fails; subset-sum DP required)
    ref = ("import sys\n"
           "d=sys.stdin.buffer.read().split()\n"
           "n=int(d[0]);a=[int(x) for x in d[1:1+n]];tot=sum(a)\n"
           "poss=1\n"
           "for x in a:poss|=poss<<x\n"
           "best=tot\n"
           "for s in range(tot+1):\n"
           "    if poss>>s&1:\n"
           "        diff=abs(tot-2*s)\n"
           "        if diff<best:best=diff\n"
           "print(best)\n")
    naive = ("import sys\n"
             "d=sys.stdin.buffer.read().split()\n"
             "n=int(d[0]);a=sorted((int(x) for x in d[1:1+n]),reverse=True)\n"
             "a1=0;b1=0\n"
             "for x in a:\n"
             "    if a1<=b1:a1+=x\n"
             "    else:b1+=x\n"
             "print(abs(a1-b1))\n")
    brute = ("import sys\n"
             "d=sys.stdin.buffer.read().split()\n"
             "n=int(d[0]);a=[int(x) for x in d[1:1+n]];tot=sum(a);best=tot\n"
             "for m in range(1<<n):\n"
             "    s=sum(a[i] for i in range(n) if m>>i&1)\n"
             "    diff=abs(tot-2*s)\n"
             "    if diff<best:best=diff\n"
             "print(best)\n")

    def _pub_wa4(rng):
        out = []
        for npairs in (2, 3, 3):
            a = []
            for _ in range(npairs):
                v = rng.randint(1, 200); a += [v, v]   # equal pairs -> greedy balances to 0
            rng.shuffle(a)
            out.append(_case([len(a)], a))
        return out

    # multisets where longest-processing-time greedy is verified strictly suboptimal
    _WA4_BASES = ([5, 4, 3, 2, 2], [8, 7, 6, 5, 4], [6, 5, 5, 4, 3])

    def _hid_wa4(rng):
        out = []
        picks = [_WA4_BASES[0], _WA4_BASES[1], _WA4_BASES[2], rng.choice(_WA4_BASES)]
        for b in picks:
            k = rng.randint(1, 40)
            a = [x * k for x in b]
            out.append(_case([len(a)], a))
        return out
    S.append(MintedTemplateV1(
        name="wa_min_partition_diff", family="balanced_partition",
        mode=MODE_WRONG_ALGORITHM, kind=KIND_PASSFAIL, float_tol=0.0,
        statement=(
            "Minimum partition difference.\n\n"
            "Split N non-negative integers into two groups to minimize the absolute "
            "difference of the two group sums. Output that minimum difference.\n\n"
            "Input: first line N; second line N integers.\nOutput: the minimum "
            "achievable |sumA - sumB|.\nConstraints: 1 <= N <= 100, 1 <= a_i <= 1000."),
        ref_source=ref, naive_source=naive, brute_source=brute,
        algo_sig="subset_sum_dp", discriminator=DISC_OUTPUT_MISMATCH,
        brute_cap_tokens=400, gen_public=_pub_wa4, gen_hidden=_hid_wa4))

    # WA5 — longest strictly increasing subsequence (length); naive = longest run
    ref = ("import sys,bisect\n"
           "d=sys.stdin.buffer.read().split()\n"
           "n=int(d[0]);a=[int(x) for x in d[1:1+n]];tails=[]\n"
           "for x in a:\n"
           "    i=bisect.bisect_left(tails,x)\n"
           "    if i==len(tails):tails.append(x)\n"
           "    else:tails[i]=x\n"
           "print(len(tails))\n")
    naive = ("import sys\n"
             "d=sys.stdin.buffer.read().split()\n"
             "n=int(d[0]);a=[int(x) for x in d[1:1+n]];best=1 if n else 0;cur=1\n"
             "for i in range(1,n):\n"
             "    if a[i]>a[i-1]:cur+=1;best=max(best,cur)\n"
             "    else:cur=1\n"
             "print(best)\n")
    brute = ("import sys\n"
             "d=sys.stdin.buffer.read().split()\n"
             "n=int(d[0]);a=[int(x) for x in d[1:1+n]]\n"
             "dp=[1]*n\n"
             "for i in range(n):\n"
             "    for j in range(i):\n"
             "        if a[j]<a[i] and dp[j]+1>dp[i]:dp[i]=dp[j]+1\n"
             "print(max(dp) if n else 0)\n")

    def _pub_wa5(rng):
        out = []
        for n in (4, 5, 6):
            a = sorted(rng.sample(range(1, 50), n))   # increasing -> run == LIS
            out.append(_case([n], a))
        return out

    def _hid_wa5(rng):
        out = []
        for _ in range(4):
            # interleave so LIS is non-contiguous and longer than the longest run
            k = rng.randint(3, 5)
            a = []
            for t in range(k):
                a.append(t + 1); a.append(rng.randint(40, 60))
            out.append(_case([len(a)], a))
        return out
    S.append(MintedTemplateV1(
        name="wa_longest_increasing_subseq", family="lis",
        mode=MODE_WRONG_ALGORITHM, kind=KIND_PASSFAIL, float_tol=0.0,
        statement=(
            "Longest increasing subsequence.\n\n"
            "Given N integers, output the length of the longest STRICTLY increasing "
            "subsequence (the chosen elements need not be contiguous).\n\n"
            "Input: first line N; second line N integers.\nOutput: the length.\n"
            "Constraints: 1 <= N <= 100000, 1 <= a_i <= 10^9."),
        ref_source=ref, naive_source=naive, brute_source=brute,
        algo_sig="patience_lis", discriminator=DISC_OUTPUT_MISMATCH,
        brute_cap_tokens=400, gen_public=_pub_wa5, gen_hidden=_hid_wa5))

    # WA6 — 0/1 knapsack (ratio-greedy fails; DP required)
    ref = ("import sys\n"
           "d=sys.stdin.buffer.read().split()\n"
           "n=int(d[0]);W=int(d[1]);it=[];k=2\n"
           "for _ in range(n):\n"
           "    wt=int(d[k]);val=int(d[k+1]);k+=2;it.append((wt,val))\n"
           "dp=[0]*(W+1)\n"
           "for wt,val in it:\n"
           "    for c in range(W,wt-1,-1):\n"
           "        if dp[c-wt]+val>dp[c]:dp[c]=dp[c-wt]+val\n"
           "print(dp[W])\n")
    naive = ("import sys\n"
             "d=sys.stdin.buffer.read().split()\n"
             "n=int(d[0]);W=int(d[1]);it=[];k=2\n"
             "for _ in range(n):\n"
             "    wt=int(d[k]);val=int(d[k+1]);k+=2;it.append((wt,val))\n"
             "it.sort(key=lambda x:-x[1]/x[0]);cap=W;tot=0\n"
             "for wt,val in it:\n"
             "    if wt<=cap:cap-=wt;tot+=val\n"
             "print(tot)\n")
    brute = ("import sys\n"
             "d=sys.stdin.buffer.read().split()\n"
             "n=int(d[0]);W=int(d[1]);it=[];k=2\n"
             "for _ in range(n):\n"
             "    wt=int(d[k]);val=int(d[k+1]);k+=2;it.append((wt,val))\n"
             "best=0\n"
             "for m in range(1<<n):\n"
             "    wt=sum(it[i][0] for i in range(n) if m>>i&1)\n"
             "    if wt<=W:\n"
             "        v=sum(it[i][1] for i in range(n) if m>>i&1)\n"
             "        if v>best:best=v\n"
             "print(best)\n")

    def _pub_wa6(rng):
        out = []
        for _ in range(3):
            n = rng.randint(3, 5)
            it = [(rng.randint(1, 5), rng.randint(1, 9)) for _ in range(n)]
            W = sum(w for w, _ in it) + rng.randint(0, 3)   # all fit -> greedy=optimal
            flat = [x for p in it for x in p]
            out.append(_case([n, W], flat))
        return out

    def _hid_wa6(rng):
        out = []
        # the textbook ratio-greedy failure: ratios 6,5,4 but the two heavier items win
        base = [(10, 60), (20, 100), (30, 120)]; baseW = 50
        for _ in range(4):
            k = rng.randint(1, 20)
            it = [(w * k, v * k) for w, v in base]; W = baseW * k
            flat = [x for p in it for x in p]
            out.append(_case([len(it), W], flat))
        return out
    S.append(MintedTemplateV1(
        name="wa_knapsack_01", family="knapsack",
        mode=MODE_WRONG_ALGORITHM, kind=KIND_PASSFAIL, float_tol=0.0,
        statement=(
            "0/1 knapsack.\n\n"
            "There are N items; item k has weight w_k and value v_k. Choose a subset with "
            "total weight at most W maximizing total value. Output the maximum value.\n\n"
            "Input: first line N and W; then N lines each with w and v.\nOutput: the "
            "maximum value.\nConstraints: 1 <= N <= 100, 1 <= W <= 2000, 1 <= w,v <= "
            "10^4."),
        ref_source=ref, naive_source=naive, brute_source=brute,
        algo_sig="knapsack_dp", discriminator=DISC_OUTPUT_MISMATCH,
        brute_cap_tokens=400, gen_public=_pub_wa6, gen_hidden=_hid_wa6))

    # WA7 — maximum product of a contiguous subarray (sign flips; Kadane-max fails)
    ref = ("import sys\n"
           "d=sys.stdin.buffer.read().split()\n"
           "n=int(d[0]);a=[int(x) for x in d[1:1+n]]\n"
           "best=hi=lo=a[0]\n"
           "for x in a[1:]:\n"
           "    if x<0:hi,lo=lo,hi\n"
           "    hi=max(x,hi*x);lo=min(x,lo*x)\n"
           "    if hi>best:best=hi\n"
           "print(best)\n")
    naive = ("import sys\n"
             "d=sys.stdin.buffer.read().split()\n"
             "n=int(d[0]);a=[int(x) for x in d[1:1+n]]\n"
             "best=cur=a[0]\n"
             "for x in a[1:]:\n"
             "    cur=max(x,cur*x)\n"
             "    if cur>best:best=cur\n"
             "print(best)\n")
    brute = ("import sys\n"
             "d=sys.stdin.buffer.read().split()\n"
             "n=int(d[0]);a=[int(x) for x in d[1:1+n]];best=a[0]\n"
             "for i in range(n):\n"
             "    p=1\n"
             "    for j in range(i,n):\n"
             "        p*=a[j]\n"
             "        if p>best:best=p\n"
             "print(best)\n")

    def _pub_wa7(rng):
        out = []
        for n in (4, 5, 6):
            a = [rng.randint(1, 6) for _ in range(n)]   # all positive -> max-only ok
            out.append(_case([n], a))
        return out

    def _hid_wa7(rng):
        out = []
        for _ in range(4):
            # two negatives flanking -> their product is the max; max-only Kadane misses it
            x = rng.randint(2, 6); y = rng.randint(2, 6)
            a = [-x, rng.randint(1, 4), -y]
            out.append(_case([len(a)], a))
        for _ in range(2):
            a = [-(rng.randint(2, 5)), -(rng.randint(2, 5)), -(rng.randint(2, 5))]
            out.append(_case([len(a)], a))
        return out
    S.append(MintedTemplateV1(
        name="wa_max_product_subarray", family="max_product_subarray",
        mode=MODE_WRONG_ALGORITHM, kind=KIND_PASSFAIL, float_tol=0.0,
        statement=(
            "Best contiguous product.\n\n"
            "A sensor logged N integer readings r_1..r_N (some may be negative or zero). "
            "Among every non-empty contiguous segment of readings, report the largest "
            "achievable product of the values in a segment.\n\n"
            "Input: first line N; second line the N readings.\nOutput: the largest "
            "segment product.\nConstraints: 1 <= N <= 100000, -50 <= r_i <= 50."),
        ref_source=ref, naive_source=naive, brute_source=brute,
        algo_sig="dp_min_max_product", discriminator=DISC_OUTPUT_MISMATCH,
        brute_cap_tokens=400, gen_public=_pub_wa7, gen_hidden=_hid_wa7))

    # WA8 — longest common subsequence length; naive = longest common substring
    ref = ("import sys\n"
           "d=sys.stdin.buffer.read().split()\n"
           "A=d[0];B=d[1];n=len(A);m=len(B)\n"
           "dp=[[0]*(m+1) for _ in range(n+1)]\n"
           "for i in range(1,n+1):\n"
           "    for j in range(1,m+1):\n"
           "        dp[i][j]=dp[i-1][j-1]+1 if A[i-1]==B[j-1] else max(dp[i-1][j],dp[i][j-1])\n"
           "print(dp[n][m])\n")
    naive = ("import sys\n"
             "d=sys.stdin.buffer.read().split()\n"
             "A=d[0];B=d[1];n=len(A);m=len(B);best=0\n"
             "dp=[[0]*(m+1) for _ in range(n+1)]\n"
             "for i in range(1,n+1):\n"
             "    for j in range(1,m+1):\n"
             "        if A[i-1]==B[j-1]:dp[i][j]=dp[i-1][j-1]+1;best=max(best,dp[i][j])\n"
             "print(best)\n")
    brute = ("import sys\n"
             "from functools import lru_cache\n"
             "d=sys.stdin.buffer.read().split()\n"
             "A=d[0];B=d[1]\n"
             "@lru_cache(None)\n"
             "def f(i,j):\n"
             "    if i==len(A) or j==len(B):return 0\n"
             "    if A[i]==B[j]:return 1+f(i+1,j+1)\n"
             "    return max(f(i+1,j),f(i,j+1))\n"
             "print(f(0,0))\n")

    def _alpha(rng, n):
        return "".join(rng.choice("abcde") for _ in range(n))

    def _pub_wa8(rng):
        out = []
        for _ in range(3):
            base = _alpha(rng, rng.randint(3, 5))
            out.append(base + " " + base)     # identical -> LCS == LCSubstring == len
        return out

    def _hid_wa8(rng):
        out = []
        for _ in range(4):
            # interleave a common subsequence that is NOT contiguous in either string
            A = "a" + _alpha(rng, 2) + "b" + _alpha(rng, 2) + "c"
            B = "a" + _alpha(rng, 1) + "b" + _alpha(rng, 1) + "c"
            out.append(A + " " + B)
        return out
    S.append(MintedTemplateV1(
        name="wa_longest_common_subseq", family="lcs",
        mode=MODE_WRONG_ALGORITHM, kind=KIND_PASSFAIL, float_tol=0.0,
        statement=(
            "Longest common subsequence.\n\n"
            "Given two lowercase strings A and B, output the length of their longest "
            "common SUBSEQUENCE (characters in order, not necessarily contiguous).\n\n"
            "Input: a single line with A and B separated by a space.\nOutput: the LCS "
            "length.\nConstraints: 1 <= |A|, |B| <= 2000."),
        ref_source=ref, naive_source=naive, brute_source=brute,
        algo_sig="lcs_dp", discriminator=DISC_OUTPUT_MISMATCH,
        brute_cap_tokens=400, gen_public=_pub_wa8, gen_hidden=_hid_wa8))

    # ============================================================ SEARCH_ENUM (9)
    MOD = "10**9+7"

    # SE1 — count subsets summing to T (0/1); naive = unbounded (allows reuse)
    ref = ("import sys\n"
           "d=sys.stdin.buffer.read().split()\n"
           "c=int(d[0]);T=int(d[1]);a=[int(x) for x in d[2:2+c]];MOD=%s\n"
           "dp=[0]*(T+1);dp[0]=1\n"
           "for x in a:\n"
           "    for v in range(T,x-1,-1):dp[v]=(dp[v]+dp[v-x])%%MOD\n"
           "print(dp[T]%%MOD)\n" % MOD)
    naive = ("import sys\n"
             "d=sys.stdin.buffer.read().split()\n"
             "c=int(d[0]);T=int(d[1]);a=[int(x) for x in d[2:2+c]];MOD=%s\n"
             "dp=[0]*(T+1);dp[0]=1\n"
             "for x in a:\n"
             "    for v in range(x,T+1):dp[v]=(dp[v]+dp[v-x])%%MOD\n"
             "print(dp[T]%%MOD)\n" % MOD)
    brute = ("import sys\n"
             "d=sys.stdin.buffer.read().split()\n"
             "c=int(d[0]);T=int(d[1]);a=[int(x) for x in d[2:2+c]];MOD=%s;cnt=0\n"
             "for m in range(1<<c):\n"
             "    if sum(a[i] for i in range(c) if m>>i&1)==T:cnt+=1\n"
             "print(cnt%%MOD)\n" % MOD)

    def _pub_se1(rng):
        out = []
        for _ in range(3):
            n = rng.randint(3, 5); T = rng.randint(15, 25)
            a = [rng.randint(T // 2 + 1, T) for _ in range(n)]   # each > T/2 -> no reuse
            out.append(_case([n, T], a))
        return out

    def _hid_se1(rng):
        out = []
        for _ in range(4):
            n = rng.randint(4, 6); T = rng.randint(6, 12)
            a = [rng.randint(1, 4) for _ in range(n)]   # small -> reuse changes the count
            out.append(_case([n, T], a))
        return out
    S.append(MintedTemplateV1(
        name="se_subset_sum_count", family="subset_sum_count",
        mode=MODE_SEARCH_ENUM, kind=KIND_PASSFAIL, float_tol=0.0,
        statement=(
            "Count subsets with a given sum.\n\n"
            "Given C values and a target T, count the subsets (each element used at most "
            "once) whose sum is exactly T. Output the count modulo 1000000007.\n\n"
            "Input: first line C and T; second line the C values.\nOutput: the count mod "
            "1000000007.\nConstraints: 1 <= C <= 100, 1 <= T <= 2000, 1 <= value <= T."),
        ref_source=ref, naive_source=naive, brute_source=brute,
        algo_sig="01_subset_count_dp", discriminator=DISC_OUTPUT_MISMATCH,
        brute_cap_tokens=400, gen_public=_pub_se1, gen_hidden=_hid_se1))

    # SE2 — count binary strings of length N with no two adjacent 1s (Fibonacci)
    ref = ("import sys\n"
           "n=int(sys.stdin.read().split()[0]);MOD=%s\n"
           "a,b=1,1\n"
           "for _ in range(n):a,b=b,(a+b)%%MOD\n"
           "print(b%%MOD)\n" % MOD)
    naive = ("import sys\n"
             "n=int(sys.stdin.read().split()[0]);MOD=%s\n"
             "print((pow(2,n,MOD)-(n-1)*pow(2,n-2,MOD))%%MOD if n>=2 else pow(2,n,MOD))\n"
             % MOD)
    brute = ("import sys\n"
             "n=int(sys.stdin.read().split()[0]);MOD=%s;cnt=0\n"
             "for m in range(1<<n):\n"
             "    if not(m&(m>>1)):cnt+=1\n"
             "print(cnt%%MOD)\n" % MOD)
    S.append(MintedTemplateV1(
        name="se_binary_no_adjacent_ones", family="fib_no_adjacent",
        mode=MODE_SEARCH_ENUM, kind=KIND_PASSFAIL, float_tol=0.0,
        statement=(
            "Binary strings without adjacent ones.\n\n"
            "Count the binary strings of length N that contain no two adjacent 1s. "
            "Output the count modulo 1000000007.\n\n"
            "Input: a single integer N.\nOutput: the count mod 1000000007.\n"
            "Constraints: 1 <= N <= 100000."),
        ref_source=ref, naive_source=naive, brute_source=brute,
        algo_sig="fibonacci_dp", discriminator=DISC_OUTPUT_MISMATCH,
        brute_cap_tokens=8,
        gen_public=lambda r: [str(1), str(2)],
        gen_hidden=lambda r: [str(r.randint(3, 7)), str(r.randint(8, 14)),
                              str(r.randint(15, 18))]))

    # SE3 — number of ways to climb N stairs in steps of 1 or 2 (Fibonacci, ordered)
    ref = ("import sys\n"
           "n=int(sys.stdin.read().split()[0]);MOD=%s\n"
           "a,b=1,1\n"
           "for _ in range(n):a,b=b,(a+b)%%MOD\n"
           "print(a%%MOD)\n" % MOD)
    naive = ("import sys\n"
             "n=int(sys.stdin.read().split()[0]);MOD=%s\n"
             "print((n//2+1)%%MOD)\n" % MOD)
    brute = ("import sys\n"
             "from functools import lru_cache\n"
             "n=int(sys.stdin.read().split()[0]);MOD=%s\n"
             "@lru_cache(None)\n"
             "def f(k):\n"
             "    if k<0:return 0\n"
             "    if k==0:return 1\n"
             "    return (f(k-1)+f(k-2))%%MOD\n"
             "print(f(n)%%MOD)\n" % MOD)
    S.append(MintedTemplateV1(
        name="se_count_stair_climbings", family="stair_ways",
        mode=MODE_SEARCH_ENUM, kind=KIND_PASSFAIL, float_tol=0.0,
        statement=(
            "Counting stair climbings.\n\n"
            "You climb a staircase of N steps, taking 1 or 2 steps at a time. Count the "
            "number of distinct ORDERED sequences of moves that reach the top. Output "
            "the count modulo 1000000007.\n\n"
            "Input: a single integer N.\nOutput: the count mod 1000000007.\n"
            "Constraints: 1 <= N <= 100000."),
        ref_source=ref, naive_source=naive, brute_source=brute,
        algo_sig="fibonacci_ordered", discriminator=DISC_OUTPUT_MISMATCH,
        brute_cap_tokens=8,
        gen_public=lambda r: [str(1), str(2)],
        gen_hidden=lambda r: [str(r.randint(3, 7)), str(r.randint(8, 14)),
                              str(r.randint(15, 20))]))

    # SE4 — number of ordered ways to score N points with plays of 1, 2 or 3 (tribonacci)
    ref = ("import sys\n"
           "n=int(sys.stdin.read().split()[0]);MOD=%s\n"
           "f=[0]*(n+3);f[0]=1\n"
           "for i in range(1,n+1):f[i]=(f[i-1]+(f[i-2] if i>=2 else 0)+(f[i-3] if i>=3 else 0))%%MOD\n"
           "print(f[n]%%MOD)\n" % MOD)
    naive = ("import sys\n"
             "from functools import lru_cache\n"
             "n=int(sys.stdin.read().split()[0]);MOD=%s\n"
             "@lru_cache(None)\n"
             "def f(rem,mx):\n"
             "    if rem==0:return 1\n"
             "    t=0\n"
             "    for s in range(1,min(mx,rem)+1):t+=f(rem-s,s)\n"
             "    return t%%MOD\n"
             "print(f(n,3)%%MOD)\n" % MOD)
    brute = ("import sys\n"
             "from functools import lru_cache\n"
             "n=int(sys.stdin.read().split()[0]);MOD=%s\n"
             "@lru_cache(None)\n"
             "def f(rem):\n"
             "    if rem==0:return 1\n"
             "    t=0\n"
             "    for s in (1,2,3):\n"
             "        if s<=rem:t+=f(rem-s)\n"
             "    return t%%MOD\n"
             "print(f(n)%%MOD)\n" % MOD)
    S.append(MintedTemplateV1(
        name="se_score_combinations_ordered", family="tribonacci_ordered",
        mode=MODE_SEARCH_ENUM, kind=KIND_PASSFAIL, float_tol=0.0,
        statement=(
            "Ordered ways to reach a score.\n\n"
            "A play scores 1, 2, or 3 points. Count the number of distinct ORDERED "
            "sequences of plays whose points total exactly N. Output the count modulo "
            "1000000007.\n\n"
            "Input: a single integer N.\nOutput: the count mod 1000000007.\n"
            "Constraints: 1 <= N <= 100000."),
        ref_source=ref, naive_source=naive, brute_source=brute,
        algo_sig="tribonacci_dp", discriminator=DISC_OUTPUT_MISMATCH,
        brute_cap_tokens=8,
        gen_public=lambda r: [str(1), str(2)],
        gen_hidden=lambda r: [str(r.randint(3, 7)), str(r.randint(8, 13)),
                              str(r.randint(14, 18))]))

    # SE5 — number of distinct BSTs with N keys (Catalan); naive = factorial
    ref = ("import sys\n"
           "n=int(sys.stdin.read().split()[0]);MOD=%s\n"
           "C=[0]*(n+1);C[0]=1\n"
           "for i in range(1,n+1):\n"
           "    for j in range(i):C[i]=(C[i]+C[j]*C[i-1-j])%%MOD\n"
           "print(C[n]%%MOD)\n" % MOD)
    naive = ("import sys,math\n"
             "n=int(sys.stdin.read().split()[0]);MOD=%s\n"
             "print(math.factorial(n)%%MOD)\n" % MOD)
    brute = ("import sys,math\n"
             "n=int(sys.stdin.read().split()[0]);MOD=%s\n"
             "print((math.comb(2*n,n)//(n+1))%%MOD)\n" % MOD)
    S.append(MintedTemplateV1(
        name="se_count_bsts_catalan", family="catalan_bst",
        mode=MODE_SEARCH_ENUM, kind=KIND_PASSFAIL, float_tol=0.0,
        statement=(
            "Counting binary search trees.\n\n"
            "Count the number of structurally distinct binary search trees that store N "
            "distinct keys. Output the count modulo 1000000007.\n\n"
            "Input: a single integer N.\nOutput: the count mod 1000000007.\n"
            "Constraints: 0 <= N <= 1000."),
        ref_source=ref, naive_source=naive, brute_source=brute,
        algo_sig="catalan_dp", discriminator=DISC_OUTPUT_MISMATCH,
        brute_cap_tokens=8,
        gen_public=lambda r: [str(1), str(2)],
        gen_hidden=lambda r: [str(r.randint(3, 6)), str(r.randint(7, 10)),
                              str(r.randint(11, 14))]))

    # SE6 — number of integer partitions of N; naive = number of compositions 2^(N-1)
    ref = ("import sys\n"
           "n=int(sys.stdin.read().split()[0])\n"
           "dp=[0]*(n+1);dp[0]=1\n"
           "for k in range(1,n+1):\n"
           "    for v in range(k,n+1):dp[v]+=dp[v-k]\n"
           "print(dp[n])\n")
    naive = ("import sys\n"
             "n=int(sys.stdin.read().split()[0])\n"
             "print(2**(n-1) if n>=1 else 1)\n")
    brute = ("import sys\n"
             "from functools import lru_cache\n"
             "n=int(sys.stdin.read().split()[0])\n"
             "@lru_cache(None)\n"
             "def f(rem,mx):\n"
             "    if rem==0:return 1\n"
             "    t=0\n"
             "    for k in range(min(rem,mx),0,-1):t+=f(rem-k,k)\n"
             "    return t\n"
             "print(f(n,n))\n")
    S.append(MintedTemplateV1(
        name="se_integer_partition_count", family="partition_count",
        mode=MODE_SEARCH_ENUM, kind=KIND_PASSFAIL, float_tol=0.0,
        statement=(
            "Integer partitions.\n\n"
            "Count the number of ways to write N as a sum of positive integers where "
            "order does NOT matter (so 3 = 2+1 = 1+1+1 gives 3 partitions). Output the "
            "exact count.\n\n"
            "Input: a single integer N.\nOutput: the number of partitions of N.\n"
            "Constraints: 1 <= N <= 60."),
        ref_source=ref, naive_source=naive, brute_source=brute,
        algo_sig="partition_dp", discriminator=DISC_OUTPUT_MISMATCH,
        brute_cap_tokens=8,
        gen_public=lambda r: [str(1), str(2)],
        gen_hidden=lambda r: [str(r.randint(3, 8)), str(r.randint(9, 16)),
                              str(r.randint(17, 25))]))

    # SE7 — count monotone lattice paths avoiding blocked cells; naive ignores blocks
    ref = ("import sys\n"
           "d=sys.stdin.read().split()\n"
           "R=int(d[0]);C=int(d[1]);g=d[2:2+R];MOD=%s\n"
           "dp=[[0]*C for _ in range(R)]\n"
           "for i in range(R):\n"
           "    for j in range(C):\n"
           "        if g[i][j]=='#':dp[i][j]=0\n"
           "        elif i==0 and j==0:dp[i][j]=1\n"
           "        else:\n"
           "            v=0\n"
           "            if i>0:v+=dp[i-1][j]\n"
           "            if j>0:v+=dp[i][j-1]\n"
           "            dp[i][j]=v%%MOD\n"
           "print(dp[R-1][C-1]%%MOD)\n" % MOD)
    naive = ("import sys,math\n"
             "d=sys.stdin.read().split()\n"
             "R=int(d[0]);C=int(d[1]);MOD=%s\n"
             "print(math.comb(R+C-2,R-1)%%MOD)\n" % MOD)
    brute = ("import sys\n"
             "from functools import lru_cache\n"
             "d=sys.stdin.read().split()\n"
             "R=int(d[0]);C=int(d[1]);g=d[2:2+R];MOD=%s\n"
             "@lru_cache(None)\n"
             "def f(i,j):\n"
             "    if i>=R or j>=C or g[i][j]=='#':return 0\n"
             "    if i==R-1 and j==C-1:return 1\n"
             "    return (f(i+1,j)+f(i,j+1))%%MOD\n"
             "print(f(0,0)%%MOD)\n" % MOD)

    def _pub_se7(rng):
        out = []
        for _ in range(3):
            R = rng.randint(2, 4); C = rng.randint(2, 4)
            rows = ["." * C for _ in range(R)]          # no blocks -> binomial is correct
            out.append(_case([R, C], rows))
        return out

    def _hid_se7(rng):
        out = []
        for _ in range(4):
            R = rng.randint(3, 5); C = rng.randint(3, 5)
            grid = [["." for _ in range(C)] for _ in range(R)]
            # add a couple of blocks, never the start or end
            for _ in range(rng.randint(1, 2)):
                i = rng.randint(0, R - 1); j = rng.randint(0, C - 1)
                if (i, j) != (0, 0) and (i, j) != (R - 1, C - 1):
                    grid[i][j] = "#"
            rows = ["".join(r) for r in grid]
            out.append(_case([R, C], rows))
        return out
    S.append(MintedTemplateV1(
        name="se_lattice_paths_blocked", family="grid_path_count",
        mode=MODE_SEARCH_ENUM, kind=KIND_PASSFAIL, float_tol=0.0,
        statement=(
            "Counting blocked lattice paths.\n\n"
            "On an R x C grid you move only right or down from the top-left cell to the "
            "bottom-right cell. Some cells marked '#' are blocked and may not be "
            "entered; '.' cells are free (the start and end are always free). Count the "
            "number of paths, modulo 1000000007.\n\n"
            "Input: first line R and C; then R lines each a string of '.' and '#'.\n"
            "Output: the number of paths mod 1000000007.\nConstraints: 1 <= R, C <= 50."),
        ref_source=ref, naive_source=naive, brute_source=brute,
        algo_sig="grid_dp", discriminator=DISC_OUTPUT_MISMATCH,
        brute_cap_tokens=400, gen_public=_pub_se7, gen_hidden=_hid_se7))

    # SE8 — number of ways to make change (UNORDERED); naive counts ordered sequences
    ref = ("import sys\n"
           "d=sys.stdin.buffer.read().split()\n"
           "c=int(d[0]);N=int(d[1]);coins=[int(x) for x in d[2:2+c]];MOD=%s\n"
           "dp=[0]*(N+1);dp[0]=1\n"
           "for co in coins:\n"
           "    for v in range(co,N+1):dp[v]=(dp[v]+dp[v-co])%%MOD\n"
           "print(dp[N]%%MOD)\n" % MOD)
    naive = ("import sys\n"
             "d=sys.stdin.buffer.read().split()\n"
             "c=int(d[0]);N=int(d[1]);coins=[int(x) for x in d[2:2+c]];MOD=%s\n"
             "dp=[0]*(N+1);dp[0]=1\n"
             "for v in range(1,N+1):\n"
             "    for co in coins:\n"
             "        if co<=v:dp[v]=(dp[v]+dp[v-co])%%MOD\n"
             "print(dp[N]%%MOD)\n" % MOD)
    brute = ("import sys\n"
             "from functools import lru_cache\n"
             "d=sys.stdin.buffer.read().split()\n"
             "c=int(d[0]);N=int(d[1]);coins=sorted(set(int(x) for x in d[2:2+c]));MOD=%s\n"
             "@lru_cache(None)\n"
             "def f(i,rem):\n"
             "    if rem==0:return 1\n"
             "    if i>=len(coins) or rem<0:return 0\n"
             "    return (f(i+1,rem)+f(i,rem-coins[i]))%%MOD\n"
             "print(f(0,N)%%MOD)\n" % MOD)

    def _pub_se8(rng):
        out = []
        for _ in range(3):
            co = [rng.choice([1, 2, 5])]      # single coin -> ordered == unordered == 1 way
            N = rng.randint(1, 20)
            if N % co[0] != 0:
                N = co[0] * rng.randint(1, 6)
            out.append(_case([1, N], co))
        return out

    def _hid_se8(rng):
        out = []
        for _ in range(4):
            co = sorted(rng.sample([1, 2, 3, 4, 5], rng.randint(2, 3)))
            N = rng.randint(5, 12)
            out.append(_case([len(co), N], co))
        return out
    S.append(MintedTemplateV1(
        name="se_change_making_count", family="coin_change_ways",
        mode=MODE_SEARCH_ENUM, kind=KIND_PASSFAIL, float_tol=0.0,
        statement=(
            "Counting ways to make change.\n\n"
            "Given C coin denominations (unlimited supply) and an amount N, count the "
            "number of distinct ways to make N where order does NOT matter (so for coins "
            "{1,2} and N=3 the ways {1+1+1, 1+2} give 2). Output the count modulo "
            "1000000007.\n\n"
            "Input: first line C and N; second line the C denominations.\nOutput: the "
            "count mod 1000000007.\nConstraints: 1 <= C <= 50, 1 <= N <= 5000."),
        ref_source=ref, naive_source=naive, brute_source=brute,
        algo_sig="coin_change_unordered_dp", discriminator=DISC_OUTPUT_MISMATCH,
        brute_cap_tokens=400, gen_public=_pub_se8, gen_hidden=_hid_se8))

    return tuple(S)


RBC_SLATE_V1: tuple[MintedTemplateV1, ...] = build_slate_v1()

__all__ = ["build_slate_v1", "RBC_SLATE_V1", "CB_BIG", "CB_MED"]
