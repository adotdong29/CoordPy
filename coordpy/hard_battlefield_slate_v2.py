"""W137 / COO-9 — parser-neutral HARD battlefield slate v2.

W136 proved the W132 generated WRONG_ALGORITHM / SEARCH_ENUM field was (a) I/O-format-confounded and
(b) once I/O-normalised, one-shot / low-algorithm-headroom for a 70B model.  Lane-γ primary-source
research (LiveCodeBench-Pro arXiv:2506.11928 — LLMs solve implementation-heavy problems but fail
nuanced algorithmic reasoning; AlgBench arXiv:2601.04996 — a 92%→~49% cliff at global optimization;
competitive-error-analysis arXiv:2506.22954 — "algorithmically-correct-but-TLE" and edge/boundary are
the dominant failure modes) pins the lesson: **real 70B headroom lives where the NAIVE solution is
implementation-trivial but wrong by COMPLEXITY (passes small samples, TLEs the large hidden case) or
by an unhandled CASE — not "an obviously different algorithm."**

This slate mints such traps, every one PARSER-NEUTRAL by construction: each template declares an
:class:`~coordpy.parser_neutral_io_v1.IoShapeV1` and renders inputs through
:func:`~coordpy.parser_neutral_io_v1.render_normal_form_v1`, so a strict per-line reader and a
read-all-tokens reader recover byte-identical data (the W137 build self-test runs the HC1 gate on
every minted case).  The exact-oracle ref/brute/naive discipline (W132) is preserved verbatim: each
template ships a scalable correct ``ref_source`` (the answer key), an INDEPENDENT obviously-correct
``brute_source`` (small-case cross-check), and an admissible-wrong ``naive_source`` that passes every
public sample yet fails ≥1 hidden case (TIMEOUT for COMPLEXITY_BLIND, WRONG_ANSWER otherwise).

The slate is intentionally a RANGE of difficulty — some textbook traps (expected to be culled by the
model-ladder hardness calibration as zero-discrimination one-shots, demonstrating the calibration
works) and some genuinely hard ones (the headroom band).  Admission is decided EMPIRICALLY by the
calibration (HC3/HC4/HC5), never asserted here.

Reuses ``MintedTemplateV1`` (the W132 framework) verbatim.  Pure / deterministic / explicit-import
only; ``coordpy/__init__.py`` untouched.
"""
from __future__ import annotations

import dataclasses
import random
from typing import Any, Callable

from .resistant_by_construction_battlefield_v1 import (
    DISC_OUTPUT_MISMATCH, DISC_TIMEOUT, MintedTemplateV1,
    MODE_COMPLEXITY_BLIND, MODE_HIDDEN_EDGE, MODE_SEARCH_ENUM, MODE_WRONG_ALGORITHM)
from .coordpy_icpc_battlefield_v1 import KIND_PASSFAIL
from .parser_neutral_io_v1 import (
    IoShapeV1, array_line, grid, io_shape, render_normal_form_v1, rows, scalar_line)

HARD_SLATE_V2_SCHEMA_VERSION: str = "coordpy.hard_battlefield_slate_v2.v1"


@dataclasses.dataclass(frozen=True)
class ParserNeutralTemplateV2:
    """A W132 ``MintedTemplateV1`` whose inputs are rendered in canonical normal form, plus the
    :class:`IoShapeV1` HC1 checks against and a note on WHY it should be hard for a 70B model."""

    minted: MintedTemplateV1
    io_shape: IoShapeV1
    headroom_note: str

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.minted.name, "family": self.minted.family,
                "mode": self.minted.mode, "algo_sig": self.minted.algo_sig,
                "discriminator": self.minted.discriminator,
                "io_shape": self.io_shape.to_dict(), "headroom_note": self.headroom_note}


def make_pn_template(*, name: str, family: str, mode: str, statement: str,
                     ref_source: str, naive_source: str, brute_source: str,
                     algo_sig: str, discriminator: str, brute_cap_tokens: int,
                     shape: IoShapeV1,
                     gen_public_data: Callable[[random.Random], list[dict]],
                     gen_hidden_data: Callable[[random.Random], list[dict]],
                     headroom_note: str, kind: str = KIND_PASSFAIL,
                     float_tol: float = 0.0) -> ParserNeutralTemplateV2:
    """Wrap structured generators into a ``MintedTemplateV1`` whose ``gen_public``/``gen_hidden``
    emit canonical normal form (one logical item per line) via ``render_normal_form_v1``."""

    def _gp(rng: random.Random) -> list[str]:
        return [render_normal_form_v1(shape, d) for d in gen_public_data(rng)]

    def _gh(rng: random.Random) -> list[str]:
        return [render_normal_form_v1(shape, d) for d in gen_hidden_data(rng)]

    mt = MintedTemplateV1(
        name=name, family=family, mode=mode, kind=kind, float_tol=float(float_tol),
        statement=statement, ref_source=ref_source, naive_source=naive_source,
        brute_source=brute_source, algo_sig=algo_sig, discriminator=discriminator,
        brute_cap_tokens=int(brute_cap_tokens), gen_public=_gp, gen_hidden=_gh)
    return ParserNeutralTemplateV2(minted=mt, io_shape=shape, headroom_note=headroom_note)


# ============================================================ COMPLEXITY_BLIND family
# Naive is implementation-trivial O(N^2) and CORRECT; the hidden case is large enough that it TLEs
# the same 8s budget the pilot grades under, while the O(N log N) reference finishes with margin.

def _t_count_inversions() -> ParserNeutralTemplateV2:
    shape = io_shape(scalar_line("N"), array_line("A", "N"))
    ref = (
        "import sys\n"
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
    naive = (
        "import sys\n"
        "d=sys.stdin.buffer.read().split()\n"
        "n=int(d[0]);a=[int(x) for x in d[1:1+n]]\n"
        "inv=0\n"
        "for i in range(n):\n"
        "    ai=a[i]\n"
        "    for j in range(i+1,n):\n"
        "        if ai>a[j]:inv+=1\n"
        "print(inv)\n")
    brute = naive  # the O(N^2) double loop IS the obviously-correct independent oracle (small cases)
    stmt = (
        "Count inversions.\n\n"
        "Given an array A of N integers, count the number of pairs (i, j) with i < j and "
        "A[i] > A[j].\n\n"
        "Input: line 1 contains N; line 2 contains N integers A[1..N].\n"
        "Output: the number of inversions.\n"
        "Constraints: 1 <= N <= 200000, 1 <= A[i] <= 10^9.")

    def gp(rng: random.Random) -> list[dict]:
        out = []
        for _ in range(3):
            n = rng.randint(4, 9)
            out.append({"N": n, "A": [rng.randint(1, 20) for _ in range(n)]})
        return out

    def gh(rng: random.Random) -> list[dict]:
        out = []
        for _ in range(3):
            n = rng.randint(70000, 90000)
            out.append({"N": n, "A": [rng.randint(1, 10 ** 9) for _ in range(n)]})
        return out

    return make_pn_template(
        name="cb_count_inversions_v2", family="count_inversions", mode=MODE_COMPLEXITY_BLIND,
        statement=stmt, ref_source=ref, naive_source=naive, brute_source=brute,
        algo_sig="bit_inversions", discriminator=DISC_TIMEOUT, brute_cap_tokens=40, shape=shape,
        gen_public_data=gp, gen_hidden_data=gh,
        headroom_note="naive O(N^2) double loop TLEs at N~8e4; needs BIT/merge-sort O(N log N)")


def _t_longest_subarray_sum_le_s() -> ParserNeutralTemplateV2:
    shape = io_shape(scalar_line("N", "S"), array_line("A", "N"))
    ref = (
        "import sys\n"
        "d=sys.stdin.buffer.read().split()\n"
        "n=int(d[0]);S=int(d[1]);a=[int(x) for x in d[2:2+n]]\n"
        "best=0;cur=0;l=0\n"
        "for r in range(n):\n"
        "    cur+=a[r]\n"
        "    while cur>S and l<=r:cur-=a[l];l+=1\n"
        "    if r-l+1>best:best=r-l+1\n"
        "print(best)\n")
    naive = (
        "import sys\n"
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
    brute = naive
    stmt = (
        "Longest bounded-sum subarray.\n\n"
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
            n = rng.randint(70000, 90000)
            a = [rng.randint(0, 100) for _ in range(n)]
            out.append({"N": n, "S": rng.randint(n // 2, n * 50), "A": a})
        return out

    return make_pn_template(
        name="cb_longest_subarray_sum_le_s_v2", family="longest_bounded_subarray",
        mode=MODE_COMPLEXITY_BLIND, statement=stmt, ref_source=ref, naive_source=naive,
        brute_source=brute, algo_sig="sliding_window", discriminator=DISC_TIMEOUT,
        brute_cap_tokens=40, shape=shape, gen_public_data=gp, gen_hidden_data=gh,
        headroom_note="naive O(N^2) expanding-window TLEs at N~8e4; needs two-pointer O(N)")


# ============================================================ WRONG_ALGORITHM_ADMISSIBLE family
# The naive is a tempting, named, plausible technique that is the WRONG algorithm; it agrees with the
# reference on the (benign) public samples and disagrees on an adversarial hidden case.

def _t_house_robber_circular() -> ParserNeutralTemplateV2:
    shape = io_shape(scalar_line("N"), array_line("A", "N"))
    ref = (
        "import sys\n"
        "d=sys.stdin.buffer.read().split()\n"
        "n=int(d[0]);a=[int(x) for x in d[1:1+n]]\n"
        "def rob(b):\n"
        "    inc=0;exc=0\n"
        "    for x in b:\n"
        "        inc,exc=exc+x,max(inc,exc)\n"
        "    return max(inc,exc)\n"
        "if n==1:print(a[0])\n"
        "else:print(max(rob(a[:-1]),rob(a[1:])))\n")
    naive = (
        "import sys\n"
        "d=sys.stdin.buffer.read().split()\n"
        "n=int(d[0]);a=[int(x) for x in d[1:1+n]]\n"
        "inc=0;exc=0\n"
        "for x in a:\n"
        "    inc,exc=exc+x,max(inc,exc)\n"
        "print(max(inc,exc))\n")  # plain LINEAR house-robber: ignores the circular wrap
    brute = (
        "import sys\n"
        "d=sys.stdin.buffer.read().split()\n"
        "n=int(d[0]);a=[int(x) for x in d[1:1+n]]\n"
        "best=0\n"
        "for m in range(1<<n):\n"
        "    ok=True\n"
        "    for i in range(n):\n"
        "        if (m>>i&1) and (m>>((i+1)%n)&1):ok=False;break\n"
        "    if ok:\n"
        "        s=sum(a[i] for i in range(n) if m>>i&1)\n"
        "        if s>best:best=s\n"
        "print(best)\n")
    stmt = (
        "Circular house robber.\n\n"
        "N houses are arranged in a CIRCLE (house N is adjacent to house 1). House i holds A[i]. "
        "You may not rob two adjacent houses. Output the maximum total you can rob.\n\n"
        "Input: line 1 contains N; line 2 contains N integers A[1..N].\n"
        "Output: the maximum total.\n"
        "Constraints: 1 <= N <= 100000, 0 <= A[i] <= 10^4.")

    def gp(rng: random.Random) -> list[dict]:
        # benign: a clear interior maximum, so linear and circular agree
        out = []
        for _ in range(3):
            n = rng.randint(4, 6)
            a = [rng.randint(0, 3) for _ in range(n)]
            a[n // 2] = rng.randint(20, 30)   # dominant interior pick -> wrap never binds
            out.append({"N": n, "A": a})
        return out

    def gh(rng: random.Random) -> list[dict]:
        # adversarial: large equal ends so linear robber grabs BOTH ends (illegal on a circle)
        out = []
        for _ in range(4):
            n = rng.choice([4, 6, 8])
            a = [1] * n
            v = rng.randint(5, 9)
            a[0] = v
            a[-1] = v
            out.append({"N": n, "A": a})
        return out

    return make_pn_template(
        name="wa_house_robber_circular_v2", family="house_robber_circular",
        mode=MODE_WRONG_ALGORITHM, statement=stmt, ref_source=ref, naive_source=naive,
        brute_source=brute, algo_sig="circular_dp_two_pass", discriminator=DISC_OUTPUT_MISMATCH,
        brute_cap_tokens=40, shape=shape, gen_public_data=gp, gen_hidden_data=gh,
        headroom_note="tempting plain linear house-robber grabs both circle ends; needs the "
                      "two-pass max(rob(a[:-1]), rob(a[1:]))")


def _t_min_coins_arbitrary() -> ParserNeutralTemplateV2:
    shape = io_shape(scalar_line("M", "T"), array_line("C", "M"))
    ref = (
        "import sys\n"
        "d=sys.stdin.buffer.read().split()\n"
        "m=int(d[0]);T=int(d[1]);c=[int(x) for x in d[2:2+m]]\n"
        "INF=float('inf');dp=[0]+[INF]*T\n"
        "for v in range(1,T+1):\n"
        "    for coin in c:\n"
        "        if coin<=v and dp[v-coin]+1<dp[v]:dp[v]=dp[v-coin]+1\n"
        "print(dp[T] if dp[T]<INF else -1)\n")
    naive = (
        "import sys\n"
        "d=sys.stdin.buffer.read().split()\n"
        "m=int(d[0]);T=int(d[1]);c=sorted((int(x) for x in d[2:2+m]),reverse=True)\n"
        "rem=T;cnt=0\n"
        "for coin in c:\n"
        "    if rem<=0:break\n"
        "    k=rem//coin;cnt+=k;rem-=k*coin\n"
        "print(cnt if rem==0 else -1)\n")  # greedy largest-first: wrong for non-canonical systems
    brute = (
        "import sys\n"
        "from collections import deque\n"
        "d=sys.stdin.buffer.read().split()\n"
        "m=int(d[0]);T=int(d[1]);c=[int(x) for x in d[2:2+m]]\n"
        "seen={0};q=deque([(0,0)]);ans=-1\n"
        "while q:\n"
        "    v,steps=q.popleft()\n"
        "    if v==T:ans=steps;break\n"
        "    for coin in c:\n"
        "        nv=v+coin\n"
        "        if nv<=T and nv not in seen:seen.add(nv);q.append((nv,steps+1))\n"
        "print(ans)\n")
    stmt = (
        "Minimum coins.\n\n"
        "You have M coin denominations C[1..M] (each may be used any number of times) and a target "
        "amount T. Output the minimum number of coins that sum to exactly T, or -1 if impossible.\n\n"
        "Input: line 1 contains M and T; line 2 contains M integers C[1..M].\n"
        "Output: the minimum coin count, or -1.\n"
        "Constraints: 1 <= M <= 50, 1 <= T <= 20000, 1 <= C[i] <= 20000.")

    def gp(rng: random.Random) -> list[dict]:
        # canonical-ish systems where greedy is optimal
        out = []
        for _ in range(3):
            c = [1, 5, 10, 25]
            T = rng.randint(6, 40)
            out.append({"M": len(c), "T": T, "C": c})
        return out

    def gh(rng: random.Random) -> list[dict]:
        # non-canonical systems where greedy over-counts (e.g. {1,3,4}, T=6 -> greedy 3, opt 2)
        out = []
        specs = [([1, 3, 4], 6), ([1, 5, 6, 9], 11), ([1, 4, 5], 8), ([1, 6, 7], 12)]
        for c, T in specs:
            out.append({"M": len(c), "T": T, "C": c})
        return out

    return make_pn_template(
        name="wa_min_coins_arbitrary_v2", family="min_coins_arbitrary", mode=MODE_WRONG_ALGORITHM,
        statement=stmt, ref_source=ref, naive_source=naive, brute_source=brute,
        algo_sig="unbounded_min_coin_dp", discriminator=DISC_OUTPUT_MISMATCH, brute_cap_tokens=60,
        shape=shape, gen_public_data=gp, gen_hidden_data=gh,
        headroom_note="largest-first greedy is wrong for non-canonical coin systems; needs DP")


# ============================================================ SEARCH_ENUM family
# A counting problem whose obvious recurrence is subtly wrong (right for the benign public sizes,
# wrong on the adversarial hidden sizes).

def _t_climb_stairs_123() -> ParserNeutralTemplateV2:
    shape = io_shape(scalar_line("N"))
    ref = (
        "import sys\n"
        "n=int(sys.stdin.buffer.read().split()[0]);MOD=10**9+7\n"
        "# ordered ways to climb N stairs taking steps of 1, 2 or 3 (tribonacci)\n"
        "dp=[0]*(max(n,2)+1);dp[0]=1;dp[1]=1;dp[2]=2\n"
        "for i in range(3,n+1):dp[i]=(dp[i-1]+dp[i-2]+dp[i-3])%MOD\n"
        "print(dp[n]%MOD)\n")
    naive = (
        "import sys\n"
        "n=int(sys.stdin.buffer.read().split()[0]);MOD=10**9+7\n"
        "# tempting but wrong: counts steps of 1 or 2 only (forgets the 3-step) -> Fibonacci\n"
        "a,b=1,1\n"
        "for _ in range(n):a,b=b,(a+b)%MOD\n"
        "print(a%MOD)\n")
    brute = (
        "import sys\n"
        "sys.setrecursionlimit(10000)\n"
        "n=int(sys.stdin.buffer.read().split()[0]);MOD=10**9+7\n"
        "from functools import lru_cache\n"
        "@lru_cache(maxsize=None)\n"
        "def f(r):\n"
        "    if r==0:return 1\n"
        "    if r<0:return 0\n"
        "    return (f(r-1)+f(r-2)+f(r-3))%MOD\n"
        "print(f(n)%MOD)\n")
    stmt = (
        "Climbing stairs (1, 2 or 3 steps).\n\n"
        "Count the number of distinct ORDERED ways to climb a staircase of N steps, where on each "
        "move you may go up 1, 2, or 3 steps. Output the count modulo 1000000007.\n\n"
        "Input: one line containing N.\n"
        "Output: the number of ways mod 1000000007.\n"
        "Constraints: 1 <= N <= 1000000.")

    def gp(rng: random.Random) -> list[dict]:
        # benign: N in {1,2} where the {1,2,3} count equals the {1,2} count (1 and 2)
        return [{"N": rng.choice([1, 2])} for _ in range(3)]

    def gh(rng: random.Random) -> list[dict]:
        # N >= 3 where the 3-step contribution makes tribonacci diverge from Fibonacci
        return [{"N": rng.randint(3, 18)} for _ in range(4)]

    return make_pn_template(
        name="se_climb_stairs_123_v2", family="climb_stairs_123", mode=MODE_SEARCH_ENUM,
        statement=stmt, ref_source=ref, naive_source=naive, brute_source=brute,
        algo_sig="tribonacci_dp", discriminator=DISC_OUTPUT_MISMATCH, brute_cap_tokens=2,
        shape=shape, gen_public_data=gp, gen_hidden_data=gh,
        headroom_note="the {1,2}-only Fibonacci count matches at N<=2; the 3-step term diverges at N>=3")


def _t_lattice_paths_blocked_nf() -> ParserNeutralTemplateV2:
    # the W132 lattice trap reborn PARSER-NEUTRAL: grid rows are one-per-line (GRID io_shape).
    shape = io_shape(scalar_line("R", "C"), grid("G", "R", "C"))
    ref = (
        "import sys\n"
        "d=sys.stdin.read().split()\n"
        "R=int(d[0]);C=int(d[1]);g=d[2:2+R];MOD=10**9+7\n"
        "dp=[[0]*C for _ in range(R)]\n"
        "for i in range(R):\n"
        "    for j in range(C):\n"
        "        if g[i][j]=='#':dp[i][j]=0;continue\n"
        "        if i==0 and j==0:dp[i][j]=1;continue\n"
        "        up=dp[i-1][j] if i>0 else 0\n"
        "        lf=dp[i][j-1] if j>0 else 0\n"
        "        dp[i][j]=(up+lf)%MOD\n"
        "print(dp[R-1][C-1]%MOD)\n")
    naive = (
        "import sys\n"
        "import math\n"
        "d=sys.stdin.buffer.read().split()\n"
        "R=int(d[0]);C=int(d[1]);MOD=10**9+7\n"
        "# tempting but wrong: closed-form binomial C(R+C-2, R-1) IGNORING the blocked cells\n"
        "print(math.comb(R+C-2,R-1)%MOD)\n")
    brute = (
        "import sys\n"
        "sys.setrecursionlimit(100000)\n"
        "d=sys.stdin.read().split()\n"
        "R=int(d[0]);C=int(d[1]);g=d[2:2+R]\n"
        "from functools import lru_cache\n"
        "@lru_cache(maxsize=None)\n"
        "def f(i,j):\n"
        "    if i>=R or j>=C or g[i][j]=='#':return 0\n"
        "    if i==R-1 and j==C-1:return 1\n"
        "    return f(i+1,j)+f(i,j+1)\n"
        "print(f(0,0))\n")
    stmt = (
        "Counting monotone lattice paths with obstacles.\n\n"
        "On an R-by-C grid you start at the top-left cell and want to reach the bottom-right cell, "
        "moving only RIGHT or DOWN one cell at a time. Some cells are blocked ('#') and cannot be "
        "entered; open cells are '.'. The start and end cells are open. Output the number of "
        "distinct paths modulo 1000000007.\n\n"
        "Input: line 1 contains R and C; then R lines follow, each a string of C characters "
        "('.' for open, '#' for blocked).\n"
        "Output: the number of paths mod 1000000007.\n"
        "Constraints: 1 <= R, C <= 1000.")

    def _mk(rng: random.Random, blocked: bool) -> dict:
        R = rng.randint(3, 5)
        C = rng.randint(3, 5)
        g = [["." for _ in range(C)] for _ in range(R)]
        if blocked:
            # interior cells (never the start/end) — every one lies on some monotone path, so a
            # block here strictly reduces the true count below the obstacle-blind binomial.
            cells = [(i, j) for i in range(R) for j in range(C)
                     if (i, j) not in ((0, 0), (R - 1, C - 1))]
            rng.shuffle(cells)
            n_blocks = min(rng.randint(1, 2), len(cells))
            for i, j in cells[:n_blocks]:
                g[i][j] = "#"
        return {"R": R, "C": C, "G": ["".join(r) for r in g]}

    def gp(rng: random.Random) -> list[dict]:
        # benign: NO blocked cells -> binomial closed form equals the DP count
        return [_mk(rng, blocked=False) for _ in range(3)]

    def gh(rng: random.Random) -> list[dict]:
        # adversarial: >=1 blocked interior cell -> binomial over-counts
        return [_mk(rng, blocked=True) for _ in range(4)]

    return make_pn_template(
        name="se_lattice_paths_blocked_nf_v2", family="grid_path_count", mode=MODE_SEARCH_ENUM,
        statement=stmt, ref_source=ref, naive_source=naive, brute_source=brute,
        algo_sig="grid_dp_obstacles", discriminator=DISC_OUTPUT_MISMATCH, brute_cap_tokens=40,
        shape=shape, gen_public_data=gp, gen_hidden_data=gh,
        headroom_note="closed-form binomial ignores obstacles; needs grid DP (W132 trap, parser-neutral)")


# ============================================================ extended COMPLEXITY_BLIND family

def _t_count_pairs_sum_le_t() -> ParserNeutralTemplateV2:
    shape = io_shape(scalar_line("N", "T"), array_line("A", "N"))
    ref = (
        "import sys\n"
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
    naive = (
        "import sys\n"
        "d=sys.stdin.buffer.read().split()\n"
        "n=int(d[0]);T=int(d[1]);a=[int(x) for x in d[2:2+n]]\n"
        "cnt=0\n"
        "for i in range(n):\n"
        "    ai=a[i]\n"
        "    for j in range(i+1,n):\n"
        "        if ai+a[j]<=T:cnt+=1\n"
        "print(cnt)\n")
    brute = naive
    stmt = (
        "Cheap pair tally.\n\n"
        "Call an unordered pair of distinct positions CHEAP when the two values they hold add up to "
        "no more than the budget T. You are given N values and the budget T; report how many cheap "
        "pairs exist.\n\n"
        "Input: line 1 contains N and T; line 2 contains N integers A[1..N].\n"
        "Output: how many cheap pairs there are.\n"
        "Constraints: 1 <= N <= 200000, 1 <= A[i] <= 10^9, 2 <= T <= 2*10^9.")

    def gp(rng: random.Random) -> list[dict]:
        out = []
        for _ in range(3):
            n = rng.randint(4, 9)
            out.append({"N": n, "T": rng.randint(2, 40), "A": [rng.randint(1, 20) for _ in range(n)]})
        return out

    def gh(rng: random.Random) -> list[dict]:
        out = []
        for _ in range(3):
            n = rng.randint(70000, 90000)
            out.append({"N": n, "T": rng.randint(1, 2 * 10 ** 9),
                        "A": [rng.randint(1, 10 ** 9) for _ in range(n)]})
        return out

    return make_pn_template(
        name="cb_count_pairs_sum_le_t_v2", family="count_pairs_sum_le_t",
        mode=MODE_COMPLEXITY_BLIND, statement=stmt, ref_source=ref, naive_source=naive,
        brute_source=brute, algo_sig="sort_two_pointer_pairsum", discriminator=DISC_TIMEOUT,
        brute_cap_tokens=40, shape=shape, gen_public_data=gp, gen_hidden_data=gh,
        headroom_note="naive O(N^2) pair scan TLEs at N~8e4; needs sort + two-pointer O(N log N)")


def _t_count_pairs_absdiff_le_d() -> ParserNeutralTemplateV2:
    shape = io_shape(scalar_line("N", "D"), array_line("A", "N"))
    ref = (
        "import sys\n"
        "d=sys.stdin.buffer.read().split()\n"
        "n=int(d[0]);D=int(d[1]);a=[int(x) for x in d[2:2+n]]\n"
        "a.sort()\n"
        "l=0;cnt=0\n"
        "for r in range(n):\n"
        "    while a[r]-a[l]>D:l+=1\n"
        "    cnt+=r-l\n"
        "print(cnt)\n")
    naive = (
        "import sys\n"
        "d=sys.stdin.buffer.read().split()\n"
        "n=int(d[0]);D=int(d[1]);a=[int(x) for x in d[2:2+n]]\n"
        "cnt=0\n"
        "for i in range(n):\n"
        "    ai=a[i]\n"
        "    for j in range(i+1,n):\n"
        "        dv=ai-a[j]\n"
        "        if -D<=dv<=D:cnt+=1\n"
        "print(cnt)\n")
    brute = naive
    stmt = (
        "Similar-value position pairs.\n\n"
        "Two positions in the array are SIMILAR when the absolute difference of the values stored "
        "there is no greater than the tolerance D. You are given N stored values and the tolerance "
        "D; report how many unordered position pairs {i, j} with i != j are similar.\n\n"
        "Input: line 1 contains N and D; line 2 contains N integers A[1..N].\n"
        "Output: how many similar position pairs exist.\n"
        "Constraints: 1 <= N <= 200000, 1 <= A[i] <= 10^9, 0 <= D <= 10^9.")

    def gp(rng: random.Random) -> list[dict]:
        out = []
        for _ in range(3):
            n = rng.randint(4, 9)
            out.append({"N": n, "D": rng.randint(0, 10), "A": [rng.randint(1, 20) for _ in range(n)]})
        return out

    def gh(rng: random.Random) -> list[dict]:
        out = []
        for _ in range(3):
            n = rng.randint(70000, 90000)
            out.append({"N": n, "D": rng.randint(0, 10 ** 9),
                        "A": [rng.randint(1, 10 ** 9) for _ in range(n)]})
        return out

    return make_pn_template(
        name="cb_count_pairs_absdiff_le_d_v2", family="count_pairs_absdiff_le_d",
        mode=MODE_COMPLEXITY_BLIND, statement=stmt, ref_source=ref, naive_source=naive,
        brute_source=brute, algo_sig="sort_sliding_window_absdiff", discriminator=DISC_TIMEOUT,
        brute_cap_tokens=40, shape=shape, gen_public_data=gp, gen_hidden_data=gh,
        headroom_note="naive O(N^2) pair scan TLEs at N~8e4; needs sort + sliding-window O(N log N)")


def _t_sum_nearest_smaller_left() -> ParserNeutralTemplateV2:
    shape = io_shape(scalar_line("N"), array_line("A", "N"))
    ref = (
        "import sys\n"
        "d=sys.stdin.buffer.read().split()\n"
        "n=int(d[0]);a=[int(x) for x in d[1:1+n]]\n"
        "st=[];total=0\n"
        "for v in a:\n"
        "    while st and st[-1]>=v:st.pop()\n"
        "    total+=st[-1] if st else -1\n"
        "    st.append(v)\n"
        "print(total)\n")
    naive = (
        "import sys\n"
        "d=sys.stdin.buffer.read().split()\n"
        "n=int(d[0]);a=[int(x) for x in d[1:1+n]]\n"
        "total=0\n"
        "for i in range(n):\n"
        "    v=a[i];found=-1\n"
        "    for j in range(i-1,-1,-1):\n"
        "        if a[j]<v:found=a[j];break\n"
        "    total+=found\n"
        "print(total)\n")
    brute = naive
    stmt = (
        "Nearest smaller to the left.\n\n"
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
        return [{"N": (n := rng.randint(70000, 90000)), "A": list(range(n, 0, -1))}
                for _ in range(3)]

    return make_pn_template(
        name="cb_sum_nearest_smaller_left_v2", family="sum_nearest_smaller_left",
        mode=MODE_COMPLEXITY_BLIND, statement=stmt, ref_source=ref, naive_source=naive,
        brute_source=brute, algo_sig="monotonic_stack_prev_smaller", discriminator=DISC_TIMEOUT,
        brute_cap_tokens=40, shape=shape, gen_public_data=gp, gen_hidden_data=gh,
        headroom_note="naive O(N^2) inner left-scan TLEs at N~8e4; needs a monotonic stack O(N)")


def _t_count_subarrays_sum_le_s() -> ParserNeutralTemplateV2:
    shape = io_shape(scalar_line("N", "S"), array_line("A", "N"))
    ref = (
        "import sys\n"
        "d=sys.stdin.buffer.read().split()\n"
        "n=int(d[0]);S=int(d[1]);a=[int(x) for x in d[2:2+n]]\n"
        "l=0;cur=0;cnt=0\n"
        "for r in range(n):\n"
        "    cur+=a[r]\n"
        "    while cur>S:cur-=a[l];l+=1\n"
        "    cnt+=r-l+1\n"
        "print(cnt)\n")
    naive = (
        "import sys\n"
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
    brute = naive
    stmt = (
        "Tally of light blocks.\n\n"
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
        out = []
        for _ in range(3):
            n = rng.randint(70000, 90000)
            out.append({"N": n, "S": rng.randint(n * 10, n * 80),
                        "A": [rng.randint(0, 100) for _ in range(n)]})
        return out

    return make_pn_template(
        name="cb_count_subarrays_sum_le_s_v2", family="count_subarrays_sum_le_s",
        mode=MODE_COMPLEXITY_BLIND, statement=stmt, ref_source=ref, naive_source=naive,
        brute_source=brute, algo_sig="two_pointer_count_subarrays", discriminator=DISC_TIMEOUT,
        brute_cap_tokens=40, shape=shape, gen_public_data=gp, gen_hidden_data=gh,
        headroom_note="naive O(N^2) expanding-window count TLEs at N~8e4; needs two-pointer O(N)")


def _t_max_j_minus_i_le() -> ParserNeutralTemplateV2:
    shape = io_shape(scalar_line("N"), array_line("A", "N"))
    ref = (
        "import sys\n"
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
    naive = (
        "import sys\n"
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
    brute = naive
    stmt = (
        "Widest non-decreasing index pair.\n\n"
        "Given an array A of N integers, find the maximum value of j - i over all index pairs with "
        "i <= j and A[i] <= A[j]. Output that maximum (0 if no pair other than i = j qualifies).\n\n"
        "Input: line 1 contains N; line 2 contains N integers A[1..N].\n"
        "Output: the maximum j - i.\n"
        "Constraints: 1 <= N <= 200000, 1 <= A[i] <= 10^9.")

    def gp(rng: random.Random) -> list[dict]:
        return [{"N": (n := rng.randint(4, 9)), "A": [rng.randint(1, 20) for _ in range(n)]}
                for _ in range(3)]

    def gh(rng: random.Random) -> list[dict]:
        return [{"N": (n := rng.randint(70000, 90000)), "A": list(range(n, 0, -1))}
                for _ in range(3)]

    return make_pn_template(
        name="cb_max_j_minus_i_le_v2", family="max_j_minus_i_le", mode=MODE_COMPLEXITY_BLIND,
        statement=stmt, ref_source=ref, naive_source=naive, brute_source=brute,
        algo_sig="prefix_min_suffix_max_two_pointer", discriminator=DISC_TIMEOUT, brute_cap_tokens=40,
        shape=shape, gen_public_data=gp, gen_hidden_data=gh,
        headroom_note="naive O(N^2) all-pairs scan TLEs at N~8e4; needs prefix-min/suffix-max two-pointer O(N)")


# ============================================================ extended WRONG_ALGORITHM family
# (incl. parser-neutral reincarnations of the W132 knapsack + weighted-interval traps)

def _t_weighted_interval_scheduling() -> ParserNeutralTemplateV2:
    shape = io_shape(scalar_line("N"), rows("IV", "N", "s", "e", "w"))
    ref = (
        "import sys, bisect\n"
        "d=sys.stdin.buffer.read().split()\n"
        "n=int(d[0]);iv=[]\n"
        "k=1\n"
        "for _ in range(n):\n"
        "    s=int(d[k]);e=int(d[k+1]);w=int(d[k+2]);k+=3\n"
        "    iv.append((e,s,w))\n"
        "iv.sort()\n"
        "ends=[x[0] for x in iv]\n"
        "dp=[0]*(n+1)\n"
        "for i in range(1,n+1):\n"
        "    e,s,w=iv[i-1]\n"
        "    j=bisect.bisect_right(ends,s,0,i-1)\n"
        "    dp[i]=max(dp[i-1],dp[j]+w)\n"
        "print(dp[n])\n")
    naive = (
        "import sys\n"
        "d=sys.stdin.buffer.read().split()\n"
        "n=int(d[0]);iv=[]\n"
        "k=1\n"
        "for _ in range(n):\n"
        "    s=int(d[k]);e=int(d[k+1]);w=int(d[k+2]);k+=3\n"
        "    iv.append((e,s,w))\n"
        "iv.sort()\n"
        "last=-1;tot=0\n"
        "for e,s,w in iv:\n"
        "    if s>=last:tot+=w;last=e\n"
        "print(tot)\n")
    brute = (
        "import sys\n"
        "d=sys.stdin.buffer.read().split()\n"
        "n=int(d[0]);iv=[]\n"
        "k=1\n"
        "for _ in range(n):\n"
        "    s=int(d[k]);e=int(d[k+1]);w=int(d[k+2]);k+=3\n"
        "    iv.append((s,e,w))\n"
        "best=0\n"
        "for m in range(1<<n):\n"
        "    sel=[iv[i] for i in range(n) if m>>i&1]\n"
        "    sel.sort()\n"
        "    ok=True\n"
        "    for a in range(1,len(sel)):\n"
        "        if sel[a][0]<sel[a-1][1]:ok=False;break\n"
        "    if ok:\n"
        "        tot=sum(x[2] for x in sel)\n"
        "        if tot>best:best=tot\n"
        "print(best)\n")
    stmt = (
        "Weighted interval scheduling.\n\n"
        "There are N jobs. Job i occupies the half-open time interval [s, e) and pays weight w. "
        "Two jobs CONFLICT if their intervals overlap (a job may start exactly when another ends). "
        "Choose a subset of mutually non-conflicting jobs maximizing the total weight. Output that "
        "maximum total weight.\n\n"
        "Input: line 1 contains N; then N lines follow, each containing s, e and w for one job.\n"
        "Output: the maximum total weight.\n"
        "Constraints: 1 <= N <= 100000, 0 <= s < e <= 10^9, 1 <= w <= 10^9.")

    def gp(rng: random.Random) -> list[dict]:
        out = []
        for _ in range(3):
            n = rng.randint(3, 5)
            w = rng.randint(2, 7)
            iv = []
            for _ in range(n):
                s = rng.randint(0, 8)
                e = s + rng.randint(1, 4)
                iv.append((s, e, w))
            out.append({"N": n, "IV": iv})
        return out

    def gh(rng: random.Random) -> list[dict]:
        specs = [
            [(0, 10, 100), (0, 2, 5), (2, 4, 5), (4, 6, 5), (6, 8, 5)],
            [(0, 6, 50), (0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 4, 4), (4, 5, 4)],
            [(0, 8, 30), (0, 2, 3), (2, 4, 3), (4, 6, 3), (6, 8, 3)],
            [(1, 9, 40), (1, 3, 2), (3, 5, 2), (5, 7, 2), (7, 9, 2)],
        ]
        return [{"N": len(sp), "IV": sp} for sp in specs]

    return make_pn_template(
        name="wa_weighted_interval_scheduling_v2", family="weighted_interval_scheduling",
        mode=MODE_WRONG_ALGORITHM, statement=stmt, ref_source=ref, naive_source=naive,
        brute_source=brute, algo_sig="weighted_interval_dp_bsearch",
        discriminator=DISC_OUTPUT_MISMATCH, brute_cap_tokens=40, shape=shape,
        gen_public_data=gp, gen_hidden_data=gh,
        headroom_note="earliest-finish greedy maximizes job COUNT not WEIGHT; needs end-sorted DP "
                      "with binary search (W132 trap, parser-neutral)")


def _t_knapsack_01() -> ParserNeutralTemplateV2:
    shape = io_shape(scalar_line("N", "W"), rows("IT", "N", "w", "v"))
    ref = (
        "import sys\n"
        "d=sys.stdin.buffer.read().split()\n"
        "n=int(d[0]);W=int(d[1]);items=[]\n"
        "k=2\n"
        "for _ in range(n):\n"
        "    wt=int(d[k]);vl=int(d[k+1]);k+=2;items.append((wt,vl))\n"
        "dp=[0]*(W+1)\n"
        "for wt,vl in items:\n"
        "    for c in range(W,wt-1,-1):\n"
        "        if dp[c-wt]+vl>dp[c]:dp[c]=dp[c-wt]+vl\n"
        "print(dp[W])\n")
    naive = (
        "import sys\n"
        "d=sys.stdin.buffer.read().split()\n"
        "n=int(d[0]);W=int(d[1]);items=[]\n"
        "k=2\n"
        "for _ in range(n):\n"
        "    wt=int(d[k]);vl=int(d[k+1]);k+=2;items.append((wt,vl))\n"
        "items.sort(key=lambda it:(-(it[1]/it[0]) if it[0]>0 else -1e18))\n"
        "rem=W;tot=0\n"
        "for wt,vl in items:\n"
        "    if wt<=rem:rem-=wt;tot+=vl\n"
        "print(tot)\n")
    brute = (
        "import sys\n"
        "d=sys.stdin.buffer.read().split()\n"
        "n=int(d[0]);W=int(d[1]);items=[]\n"
        "k=2\n"
        "for _ in range(n):\n"
        "    wt=int(d[k]);vl=int(d[k+1]);k+=2;items.append((wt,vl))\n"
        "best=0\n"
        "for m in range(1<<n):\n"
        "    sw=0;sv=0\n"
        "    for i in range(n):\n"
        "        if m>>i&1:sw+=items[i][0];sv+=items[i][1]\n"
        "    if sw<=W and sv>best:best=sv\n"
        "print(best)\n")
    stmt = (
        "0/1 knapsack.\n\n"
        "There are N items; item i has weight w and value v, and may be taken at most once. Given a "
        "knapsack capacity W, choose a subset of items whose total weight is at most W maximizing the "
        "total value. Output that maximum total value.\n\n"
        "Input: line 1 contains N and W; then N lines follow, each containing w and v for one item.\n"
        "Output: the maximum total value.\n"
        "Constraints: 1 <= N <= 2000, 1 <= W <= 100000, 1 <= w <= W, 1 <= v <= 10^9.")

    def gp(rng: random.Random) -> list[dict]:
        out = []
        for _ in range(3):
            n = rng.randint(3, 5)
            items = [(rng.randint(1, 4), rng.randint(1, 9)) for _ in range(n)]
            W = sum(w for w, _ in items) + rng.randint(0, 3)
            out.append({"N": n, "W": W, "IT": items})
        return out

    def gh(rng: random.Random) -> list[dict]:
        specs = [
            (50, [(10, 60), (20, 100), (30, 120)]),
            (10, [(5, 7), (5, 7), (6, 13)]),
            (8, [(3, 6), (3, 6), (5, 11)]),
            (6, [(4, 7), (3, 5), (3, 5)]),
        ]
        return [{"N": len(items), "W": W, "IT": items} for W, items in specs]

    return make_pn_template(
        name="wa_knapsack_01_v2", family="knapsack_01", mode=MODE_WRONG_ALGORITHM,
        statement=stmt, ref_source=ref, naive_source=naive, brute_source=brute,
        algo_sig="zero_one_knapsack_dp", discriminator=DISC_OUTPUT_MISMATCH, brute_cap_tokens=40,
        shape=shape, gen_public_data=gp, gen_hidden_data=gh,
        headroom_note="value/weight ratio greedy is the fractional heuristic; 0/1 knapsack needs "
                      "the capacity DP (W132 trap, parser-neutral)")


def _t_min_subset_diff() -> ParserNeutralTemplateV2:
    shape = io_shape(scalar_line("N"), array_line("A", "N"))
    ref = (
        "import sys\n"
        "d=sys.stdin.buffer.read().split()\n"
        "n=int(d[0]);a=[int(x) for x in d[1:1+n]]\n"
        "tot=sum(a)\n"
        "reach=1\n"
        "for x in a:reach|=reach<<x\n"
        "best=tot\n"
        "for s in range(tot+1):\n"
        "    if reach>>s&1:\n"
        "        diff=abs(tot-2*s)\n"
        "        if diff<best:best=diff\n"
        "print(best)\n")
    naive = (
        "import sys\n"
        "d=sys.stdin.buffer.read().split()\n"
        "n=int(d[0]);a=[int(x) for x in d[1:1+n]]\n"
        "a.sort(reverse=True)\n"
        "p=0;q=0\n"
        "for x in a:\n"
        "    if p<=q:p+=x\n"
        "    else:q+=x\n"
        "print(abs(p-q))\n")
    brute = (
        "import sys\n"
        "d=sys.stdin.buffer.read().split()\n"
        "n=int(d[0]);a=[int(x) for x in d[1:1+n]]\n"
        "tot=sum(a);best=tot\n"
        "for m in range(1<<n):\n"
        "    s=0\n"
        "    for i in range(n):\n"
        "        if m>>i&1:s+=a[i]\n"
        "    diff=abs(tot-2*s)\n"
        "    if diff<best:best=diff\n"
        "print(best)\n")
    stmt = (
        "Minimum partition difference.\n\n"
        "Partition the multiset of N non-negative integers A into two groups (either group may be "
        "empty). Output the minimum possible absolute difference between the two group sums.\n\n"
        "Input: line 1 contains N; line 2 contains N integers A[1..N].\n"
        "Output: the minimum absolute difference.\n"
        "Constraints: 1 <= N <= 100, 0 <= A[i] <= 1000, sum(A) <= 100000.")

    def gp(rng: random.Random) -> list[dict]:
        out = []
        for _ in range(3):
            half = [rng.randint(1, 5) for _ in range(rng.randint(2, 3))]
            a = half + half[:]
            rng.shuffle(a)
            out.append({"N": len(a), "A": a})
        return out

    def gh(rng: random.Random) -> list[dict]:
        specs = [[8, 7, 6, 5, 4], [3, 1, 1, 2, 2, 1], [7, 7, 6, 1, 1], [10, 9, 8, 7, 6, 5]]
        return [{"N": len(a), "A": a} for a in specs]

    return make_pn_template(
        name="wa_min_subset_diff_v2", family="min_partition_diff", mode=MODE_WRONG_ALGORITHM,
        statement=stmt, ref_source=ref, naive_source=naive, brute_source=brute,
        algo_sig="subset_sum_bitset_dp", discriminator=DISC_OUTPUT_MISMATCH, brute_cap_tokens=40,
        shape=shape, gen_public_data=gp, gen_hidden_data=gh,
        headroom_note="largest-first balancing greedy is not optimal; needs subset-sum DP")


# ============================================================ HIDDEN_EDGE_STATE_MISS family
# Naive handles the common case but misses one corner; the counterexample witness (C1) should help.

def _t_max_subarray_kadane() -> ParserNeutralTemplateV2:
    shape = io_shape(scalar_line("N"), array_line("A", "N"))
    ref = (
        "import sys\n"
        "d=sys.stdin.buffer.read().split()\n"
        "n=int(d[0]);a=[int(x) for x in d[1:1+n]]\n"
        "best=a[0];cur=a[0]\n"
        "for x in a[1:]:\n"
        "    cur=max(x,cur+x)\n"
        "    if cur>best:best=cur\n"
        "print(best)\n")
    naive = (
        "import sys\n"
        "d=sys.stdin.buffer.read().split()\n"
        "n=int(d[0]);a=[int(x) for x in d[1:1+n]]\n"
        "best=0;cur=0\n"
        "for x in a:\n"
        "    cur+=x\n"
        "    if cur>best:best=cur\n"
        "    if cur<0:cur=0\n"
        "print(best)\n")
    brute = (
        "import sys\n"
        "d=sys.stdin.buffer.read().split()\n"
        "n=int(d[0]);a=[int(x) for x in d[1:1+n]]\n"
        "best=a[0]\n"
        "for i in range(n):\n"
        "    s=0\n"
        "    for j in range(i,n):\n"
        "        s+=a[j]\n"
        "        if s>best:best=s\n"
        "print(best)\n")
    stmt = (
        "Maximum subarray sum.\n\n"
        "Given an array A of N integers (which may be negative), output the maximum possible sum of a "
        "non-empty contiguous subarray.\n\n"
        "Input: line 1 contains N; line 2 contains N integers A[1..N].\n"
        "Output: the maximum non-empty subarray sum.\n"
        "Constraints: 1 <= N <= 200000, -10^9 <= A[i] <= 10^9.")

    def gp(rng: random.Random) -> list[dict]:
        out = []
        for _ in range(3):
            n = rng.randint(4, 7)
            a = [rng.randint(-3, -1) for _ in range(n)]
            a[rng.randrange(n)] = rng.randint(5, 12)
            out.append({"N": n, "A": a})
        return out

    def gh(rng: random.Random) -> list[dict]:
        return [{"N": (n := rng.randint(3, 6)), "A": [rng.randint(-9, -1) for _ in range(n)]}
                for _ in range(4)]

    return make_pn_template(
        name="he_max_subarray_kadane_v2", family="max_subarray", mode=MODE_HIDDEN_EDGE,
        statement=stmt, ref_source=ref, naive_source=naive, brute_source=brute,
        algo_sig="kadane_no_zero_floor", discriminator=DISC_OUTPUT_MISMATCH, brute_cap_tokens=40,
        shape=shape, gen_public_data=gp, gen_hidden_data=gh,
        headroom_note="reset-to-0 Kadane returns 0 on all-negative input; true answer is max single element")


def _t_longest_subarray_sum_k() -> ParserNeutralTemplateV2:
    shape = io_shape(scalar_line("N", "K"), array_line("A", "N"))
    ref = (
        "import sys\n"
        "d=sys.stdin.buffer.read().split()\n"
        "n=int(d[0]);K=int(d[1]);a=[int(x) for x in d[2:2+n]]\n"
        "first={0:0}\n"
        "pre=0;best=0\n"
        "for i in range(1,n+1):\n"
        "    pre+=a[i-1]\n"
        "    if pre-K in first:\n"
        "        L=i-first[pre-K]\n"
        "        if L>best:best=L\n"
        "    if pre not in first:first[pre]=i\n"
        "print(best)\n")
    naive = (
        "import sys\n"
        "d=sys.stdin.buffer.read().split()\n"
        "n=int(d[0]);K=int(d[1]);a=[int(x) for x in d[2:2+n]]\n"
        "last={0:0}\n"
        "pre=0;best=0\n"
        "for i in range(1,n+1):\n"
        "    pre+=a[i-1]\n"
        "    if pre-K in last:\n"
        "        L=i-last[pre-K]\n"
        "        if L>best:best=L\n"
        "    last[pre]=i\n"
        "print(best)\n")
    brute = (
        "import sys\n"
        "d=sys.stdin.buffer.read().split()\n"
        "n=int(d[0]);K=int(d[1]);a=[int(x) for x in d[2:2+n]]\n"
        "best=0\n"
        "for i in range(n):\n"
        "    s=0\n"
        "    for j in range(i,n):\n"
        "        s+=a[j]\n"
        "        if s==K and j-i+1>best:best=j-i+1\n"
        "print(best)\n")
    stmt = (
        "Longest subarray with a given sum.\n\n"
        "Given an array A of N integers (which may be negative) and an integer K, output the length "
        "of the longest contiguous subarray whose elements sum to exactly K, or 0 if none exists.\n\n"
        "Input: line 1 contains N and K; line 2 contains N integers A[1..N].\n"
        "Output: the maximum such length (0 if no subarray sums to K).\n"
        "Constraints: 1 <= N <= 200000, -10^9 <= K <= 10^9, -10^4 <= A[i] <= 10^4.")

    def gp(rng: random.Random) -> list[dict]:
        out = []
        for _ in range(3):
            n = rng.randint(4, 7)
            a = [rng.randint(1, 6) for _ in range(n)]
            i = rng.randrange(n)
            j = rng.randint(i, n - 1)
            out.append({"N": n, "K": sum(a[i:j + 1]), "A": a})
        return out

    def gh(rng: random.Random) -> list[dict]:
        specs = [(0, [0, 0, 0, 0]), (3, [3, 0, 0, 0, 0]), (0, [2, -2, 2, -2, 0]), (5, [5, 0, 0, 1, -1])]
        return [{"N": len(a), "K": K, "A": a} for K, a in specs]

    return make_pn_template(
        name="he_longest_subarray_sum_k_v2", family="longest_subarray_sum_k", mode=MODE_HIDDEN_EDGE,
        statement=stmt, ref_source=ref, naive_source=naive, brute_source=brute,
        algo_sig="prefix_first_occurrence_map", discriminator=DISC_OUTPUT_MISMATCH,
        brute_cap_tokens=40, shape=shape, gen_public_data=gp, gen_hidden_data=gh,
        headroom_note="storing the LAST prefix-sum index shrinks the window; needs FIRST occurrence")


# ============================================================ extended SEARCH_ENUM family

def _t_compositions_1234() -> ParserNeutralTemplateV2:
    shape = io_shape(scalar_line("N"))
    ref = (
        "import sys\n"
        "n=int(sys.stdin.buffer.read().split()[0]);MOD=10**9+7\n"
        "dp=[0]*(max(n,3)+1);dp[0]=1\n"
        "for i in range(1,n+1):\n"
        "    for p in (1,2,3,4):\n"
        "        if i-p>=0:dp[i]=(dp[i]+dp[i-p])%MOD\n"
        "print(dp[n]%MOD)\n")
    naive = (
        "import sys\n"
        "n=int(sys.stdin.buffer.read().split()[0]);MOD=10**9+7\n"
        "dp=[0]*(max(n,2)+1);dp[0]=1\n"
        "for i in range(1,n+1):\n"
        "    for p in (1,2,3):\n"
        "        if i-p>=0:dp[i]=(dp[i]+dp[i-p])%MOD\n"
        "print(dp[n]%MOD)\n")
    brute = (
        "import sys\n"
        "sys.setrecursionlimit(10000)\n"
        "n=int(sys.stdin.buffer.read().split()[0]);MOD=10**9+7\n"
        "from functools import lru_cache\n"
        "@lru_cache(maxsize=None)\n"
        "def f(r):\n"
        "    if r==0:return 1\n"
        "    if r<0:return 0\n"
        "    return (f(r-1)+f(r-2)+f(r-3)+f(r-4))%MOD\n"
        "print(f(n)%MOD)\n")
    stmt = (
        "Counting compositions with parts up to four.\n\n"
        "Count the number of distinct ORDERED ways to write the integer N as a sum of positive parts, "
        "where every part is 1, 2, 3, or 4 (order matters: 1+3 and 3+1 are different). Output the "
        "count modulo 1000000007.\n\n"
        "Input: one line containing N.\n"
        "Output: the number of compositions mod 1000000007.\n"
        "Constraints: 1 <= N <= 1000000.")

    def gp(rng: random.Random) -> list[dict]:
        return [{"N": rng.choice([1, 2, 3])} for _ in range(3)]

    def gh(rng: random.Random) -> list[dict]:
        return [{"N": rng.randint(4, 20)} for _ in range(4)]

    return make_pn_template(
        name="se_compositions_1234_v2", family="compositions_1234", mode=MODE_SEARCH_ENUM,
        statement=stmt, ref_source=ref, naive_source=naive, brute_source=brute,
        algo_sig="tetranacci_dp", discriminator=DISC_OUTPUT_MISMATCH, brute_cap_tokens=2,
        shape=shape, gen_public_data=gp, gen_hidden_data=gh,
        headroom_note="parts-{1,2,3} tribonacci matches at N<=3; the 4-part term diverges at N>=4")


# ============================================================ slate assembly

_BUILDERS: tuple[Callable[[], ParserNeutralTemplateV2], ...] = (
    # COMPLEXITY_BLIND
    _t_count_inversions, _t_longest_subarray_sum_le_s,
    _t_count_pairs_sum_le_t, _t_count_pairs_absdiff_le_d, _t_sum_nearest_smaller_left,
    _t_count_subarrays_sum_le_s, _t_max_j_minus_i_le,
    # WRONG_ALGORITHM_ADMISSIBLE
    _t_house_robber_circular, _t_min_coins_arbitrary,
    _t_weighted_interval_scheduling, _t_knapsack_01, _t_min_subset_diff,
    # HIDDEN_EDGE_STATE_MISS
    _t_max_subarray_kadane, _t_longest_subarray_sum_k,
    # SEARCH_ENUM
    _t_climb_stairs_123, _t_lattice_paths_blocked_nf, _t_compositions_1234,
)


def build_hard_slate_v2() -> list[ParserNeutralTemplateV2]:
    """The full parser-neutral hard slate (deterministic order)."""
    return [b() for b in _BUILDERS]


def minted_slate_v2() -> list[MintedTemplateV1]:
    """Just the ``MintedTemplateV1`` objects (what ``mint_battlefield_v1`` consumes)."""
    return [t.minted for t in build_hard_slate_v2()]


def io_shape_registry_v2() -> dict[str, IoShapeV1]:
    """name -> IoShapeV1 (what the HC1 parser-neutrality gate checks each minted problem against)."""
    return {t.minted.name: t.io_shape for t in build_hard_slate_v2()}


def slate_fingerprint_cid_v1() -> str:
    """Deterministic content hash of the FULL candidate slate (names, modes, statements, and
    ref/naive/brute sources + io_shapes).  Locked in ``docs/RUNBOOK_W137.md`` before any NIM so the
    calibration / bench assert the slate has not drifted."""
    import hashlib
    import json
    fps = sorted(
        ({"name": t.minted.name, "mode": t.minted.mode, "family": t.minted.family,
          "algo_sig": t.minted.algo_sig, "disc": t.minted.discriminator,
          "stmt": t.minted.statement, "ref": t.minted.ref_source,
          "naive": t.minted.naive_source, "brute": t.minted.brute_source,
          "io_shape": t.io_shape.to_dict()} for t in build_hard_slate_v2()),
        key=lambda d: d["name"])
    return hashlib.sha256(
        json.dumps(fps, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


__all__ = [
    "HARD_SLATE_V2_SCHEMA_VERSION", "ParserNeutralTemplateV2", "make_pn_template",
    "build_hard_slate_v2", "minted_slate_v2", "io_shape_registry_v2", "slate_fingerprint_cid_v1",
]
