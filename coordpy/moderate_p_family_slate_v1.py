"""W142 / COO-9 — moderate-`p` family-screen slate (Lane α construction).

W141 earned the first clean no-oracle resistant SAME-BUDGET superiority on ONE family
(``count_pairs_sum_le_t``, fair ``p=0.08``) but the §7b ≥3-family / ≥2-mode span was unmet: the
frontier 70B is BIMODAL per technique (KNOWS it ⇒ ``p≳0.7`` ⇒ verified-selection saturates / CLUELESS
⇒ ``p≈0`` ⇒ nothing to extract). The amortization win lives in the MODERATE-`p` band ``p∈[0.10,0.50]``
(IRT peak Fisher information at ``p≈0.5``; metabench arXiv:2407.12844 / Fluid arXiv:2509.11106), and it
is inherently MULTI-FAMILY — it fires on every family with discoverable-but-rare supply.

This slate supplies the W142 screen candidates, in the two veins the W141 AST extractor can turn into a
clean leak-audited holed skeleton (a SINGLE printed accumulator updated by ``acc += <expr>`` GATED by an
``if``/shrink-``while`` predicate — see ``self_tutoring_technique_extractor_v1``):

  * sort + two-pointer / sliding-window COUNTING (COMPLEXITY_BLIND): ``count_pairs_sum`` (the W141 win),
    ``count_pairs_absdiff``, NEW ``count_pairs_product`` (product monotonicity), NEW ``count_triples_sum``;
  * two-deque sliding-window: ``subarrays_sum_and_range`` (HIDDEN_EDGE; W140 measured the 70B at A1=25%
    on this exact two-deque family — the best prior that an extractable moderate-`p` family exists outside
    counting-pairs), NEW ``longest_subarray_absdiff_le_limit`` (COMPLEXITY two-deque);
  * monotonic-stack ``sum_nearest_smaller_left`` (W141 ``p=0.67`` high-anchor, included as a known-high ref).

It ALSO ships two NON-extractable NEGATIVE CONTROLS the W142 $0 extractability gate (G3) must REJECT
(machine-checking the W142 de-risk finding): a prefix-mod-hash family (the technique lives in dict
maintenance ⇒ no gating predicate ⇒ a contribution-only hole that LEAKS) and the existing
binary-search-on-answer family (the printed answer is a reassignment, not an accumulator).

Every recipe reuses the W138 ``make_pn_template`` discipline (parser-neutral ``IoShapeV1`` + exact-oracle
``ref`` / independent ``brute`` / admissible-wrong ``naive`` + TIMEOUT or OUTPUT_MISMATCH discriminator);
surfaces are DISGUISED (no technique/efficiency cue). Pure / deterministic / explicit-import only;
``coordpy/__init__.py`` untouched.
"""
from __future__ import annotations

import dataclasses
import random
from typing import Callable

from .resistant_by_construction_battlefield_v1 import (
    DISC_TIMEOUT, MODE_COMPLEXITY_BLIND, _sha256_hex)
from .hard_battlefield_slate_v2 import ParserNeutralTemplateV2, make_pn_template
from .parser_neutral_io_v1 import array_line, io_shape, scalar_line
from .headroom_band_slate_v3 import (
    CX_FACTORIES, FUNC_FACTORIES, EXTRA_CX_FACTORIES)

MODERATE_P_FAMILY_SLATE_V1_SCHEMA_VERSION: str = "coordpy.moderate_p_family_slate_v1.v1"

# The W142 moderate-`p` win band (distinct from the W138 headroom band [0.15,0.85]); the lower edge is
# above 0 so discovery is feasible, the upper edge is where verified-selection@K=4 catches up
# ((1-0.5)^4=6.25pp, the floor of a retirement-grade per-member edge).
MODERATE_P_LO: float = 0.10
MODERATE_P_HI: float = 0.50

# vein tags (technique CLASS — the screen reports class-count alongside family-NAME count so 3 surfaces
# of one meta-technique are not sold as 3 independent families)
VEIN_SORT_TWO_POINTER: str = "sort_two_pointer_counting"
VEIN_TWO_DEQUE: str = "two_deque_sliding_window"
VEIN_MONOTONIC_STACK: str = "monotonic_stack"
VEIN_PREFIX_HASH: str = "prefix_hash"            # NON-extractable control
VEIN_BINARY_SEARCH_ANSWER: str = "binary_search_on_answer"   # NON-extractable control


def _jit(rng: random.Random, n: int) -> int:
    lo = max(2, int(n * 0.9))
    hi = max(lo + 1, int(n * 1.1))
    return rng.randint(lo, hi)


# ============================================================ NEW counting families (extractable)

def _cx_count_pairs_product_le_t(n_hidden: int) -> ParserNeutralTemplateV2:
    """Count unordered pairs whose VALUES multiply to at most a budget.  sort + both-ends two-pointer
    (product is monotone in the partner because values are positive) — the SAME clean accumulator-gated
    shape as the W141 winner (``if a[i]*a[j]<=T: cnt += j-i``), but a less-obvious monotonicity argument
    than sums ⇒ a genuinely different surface, not a rename."""
    shape = io_shape(scalar_line("N", "T"), array_line("A", "N"))
    ref = ("import sys\n"
           "d=sys.stdin.buffer.read().split()\n"
           "n=int(d[0]);T=int(d[1]);a=[int(x) for x in d[2:2+n]]\n"
           "a.sort()\n"
           "i=0;j=n-1;cnt=0\n"
           "while i<j:\n"
           "    if a[i]*a[j]<=T:\n"
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
             "        if ai*a[j]<=T:cnt+=1\n"
             "print(cnt)\n")
    stmt = ("Affordable pair tally.\n\n"
            "Call an unordered pair of distinct positions AFFORDABLE when the two values they hold "
            "multiply to no more than the budget T. You are given N values and the budget T; report how "
            "many affordable pairs exist.\n\n"
            "Input: line 1 contains N and T; line 2 contains N integers A[1..N].\n"
            "Output: how many affordable pairs there are.\n"
            "Constraints: 1 <= N <= 200000, 1 <= A[i] <= 10^9, 1 <= T <= 10^18.")

    def gp(rng: random.Random) -> list[dict]:
        return [{"N": (n := rng.randint(4, 9)), "T": rng.randint(2, 400),
                 "A": [rng.randint(1, 20) for _ in range(n)]} for _ in range(3)]

    def gh(rng: random.Random) -> list[dict]:
        return [{"N": (n := _jit(rng, n_hidden)), "T": rng.randint(1, 10 ** 18),
                 "A": [rng.randint(1, 10 ** 9) for _ in range(n)]} for _ in range(3)]

    return make_pn_template(
        name=f"b_count_pairs_product_le_t_n{n_hidden}", family="count_pairs_product_le_t",
        mode=MODE_COMPLEXITY_BLIND, statement=stmt, ref_source=ref, naive_source=naive,
        brute_source=naive, algo_sig="sort_two_pointer_pairprod", discriminator=DISC_TIMEOUT,
        brute_cap_tokens=40, shape=shape, gen_public_data=gp, gen_hidden_data=gh,
        headroom_note=f"naive O(N^2) pair scan TLEs at N~{n_hidden}; needs sort + two-pointer O(N log N)")


def _cx_count_triples_sum_lt_t(n_hidden: int) -> ParserNeutralTemplateV2:
    """Count index triples i<j<k whose three values sum below a budget.  sort + fix-one + two-pointer
    (O(N^2)) vs the O(N^3) triple loop.  The ``cnt += r-l`` contribution under the accept predicate is a
    clean gated accumulator; the O(N^3) naive TLEs at a SMALL N (~3000), de-risking the timeout check."""
    shape = io_shape(scalar_line("N", "T"), array_line("A", "N"))
    ref = ("import sys\n"
           "d=sys.stdin.buffer.read().split()\n"
           "n=int(d[0]);T=int(d[1]);a=[int(x) for x in d[2:2+n]]\n"
           "a.sort()\n"
           "cnt=0\n"
           "for i in range(n):\n"
           "    l=i+1;r=n-1\n"
           "    while l<r:\n"
           "        if a[i]+a[l]+a[r]<T:\n"
           "            cnt+=r-l;l+=1\n"
           "        else:\n"
           "            r-=1\n"
           "print(cnt)\n")
    naive = ("import sys\n"
             "d=sys.stdin.buffer.read().split()\n"
             "n=int(d[0]);T=int(d[1]);a=[int(x) for x in d[2:2+n]]\n"
             "cnt=0\n"
             "for i in range(n):\n"
             "    ai=a[i]\n"
             "    for j in range(i+1,n):\n"
             "        s2=ai+a[j]\n"
             "        for k in range(j+1,n):\n"
             "            if s2+a[k]<T:cnt+=1\n"
             "print(cnt)\n")
    stmt = ("Light triple tally.\n\n"
            "Call a triple of three distinct positions LIGHT when the three values they hold add up to "
            "strictly less than the budget T. You are given N values and the budget T; report how many "
            "light triples (counting each set of three positions once) exist.\n\n"
            "Input: line 1 contains N and T; line 2 contains N integers A[1..N].\n"
            "Output: how many light triples there are.\n"
            "Constraints: 1 <= N <= 2000, 1 <= A[i] <= 10^9, 1 <= T <= 3*10^9.")

    def gp(rng: random.Random) -> list[dict]:
        return [{"N": (n := rng.randint(4, 8)), "T": rng.randint(3, 60),
                 "A": [rng.randint(1, 20) for _ in range(n)]} for _ in range(3)]

    def gh(rng: random.Random) -> list[dict]:
        # O(N^3) naive TLEs at N~1500 (3.4e9); the O(N^2) ref (2.25e6) finishes in ~0.7s — comfortably
        # under both the 4s screen grader AND the verifier's 2s efficiency probe at the stated n_max
        # (2000 -> O(N^2)=4e6 ~1.2s).  Cap N regardless of the passed knob.
        out = []
        for _ in range(3):
            n = min(_jit(rng, n_hidden), 1500)
            out.append({"N": n, "T": rng.randint(3, 3 * 10 ** 9),
                        "A": [rng.randint(1, 10 ** 9) for _ in range(n)]})
        return out

    return make_pn_template(
        name=f"b_count_triples_sum_lt_t_n{n_hidden}", family="count_triples_sum_lt_t",
        mode=MODE_COMPLEXITY_BLIND, statement=stmt, ref_source=ref, naive_source=naive,
        brute_source=naive, algo_sig="sort_fix_two_pointer_triplesum", discriminator=DISC_TIMEOUT,
        brute_cap_tokens=60, shape=shape, gen_public_data=gp, gen_hidden_data=gh,
        headroom_note="naive O(N^3) triple scan TLEs at N~3000; needs sort + fix-one + two-pointer O(N^2)")


def _cx_count_subarrays_range_le_l(n_hidden: int) -> ParserNeutralTemplateV2:
    """Count contiguous blocks whose (max - min) <= limit.  Two monotonic deques + sliding window O(N)
    vs the O(N^2) expand-window naive.  The W140-corroborated two-deque vein (frontier A1=25%), a DISTINCT
    technique class from sort+two-pointer counting.  COUNTING form (not longest) so the accumulator
    ``cnt += r-l+1`` sits IMMEDIATELY AFTER the window-shrink ``while a[mx[0]]-a[mn[0]]>L:`` ⇒ pattern (b)
    blanks the DISCRIMINATING max-min shrink predicate INTO the hole (like ``subarrays_sum_and_range``),
    so the holed skeleton carries the technique and clears the leak gate (the longest-form's generic
    ``if r-l+1>best`` gating left the max-min logic visible ⇒ leak)."""
    shape = io_shape(scalar_line("N", "L"), array_line("A", "N"))
    ref = ("import sys\n"
           "from collections import deque\n"
           "d=sys.stdin.buffer.read().split()\n"
           "n=int(d[0]);L=int(d[1]);a=[int(x) for x in d[2:2+n]]\n"
           "mx=deque();mn=deque();l=0;cnt=0\n"
           "for r in range(n):\n"
           "    while mx and a[mx[-1]]<=a[r]:mx.pop()\n"
           "    mx.append(r)\n"
           "    while mn and a[mn[-1]]>=a[r]:mn.pop()\n"
           "    mn.append(r)\n"
           "    while a[mx[0]]-a[mn[0]]>L:\n"
           "        if mx[0]==l:mx.popleft()\n"
           "        if mn[0]==l:mn.popleft()\n"
           "        l+=1\n"
           "    cnt+=r-l+1\n"
           "print(cnt)\n")
    naive = ("import sys\n"
             "d=sys.stdin.buffer.read().split()\n"
             "n=int(d[0]);L=int(d[1]);a=[int(x) for x in d[2:2+n]]\n"
             "cnt=0\n"
             "for i in range(n):\n"
             "    mx=mn=a[i]\n"
             "    for j in range(i,n):\n"
             "        if a[j]>mx:mx=a[j]\n"
             "        if a[j]<mn:mn=a[j]\n"
             "        if mx-mn<=L:cnt+=1\n"
             "        else:break\n"
             "print(cnt)\n")
    stmt = ("Tally of steady blocks.\n\n"
            "A contiguous non-empty block of the sequence is STEADY when the gap between its largest and "
            "smallest entry is at most the tolerance L. You are given N entries and the tolerance L; "
            "report how many steady blocks the sequence contains.\n\n"
            "Input: line 1 contains N and L; line 2 contains N integers A[1..N].\n"
            "Output: how many steady blocks there are.\n"
            "Constraints: 1 <= N <= 200000, 1 <= A[i] <= 10^9, 0 <= L <= 10^9.")

    def gp(rng: random.Random) -> list[dict]:
        return [{"N": (n := rng.randint(4, 9)), "L": rng.randint(0, 8),
                 "A": [rng.randint(1, 20) for _ in range(n)]} for _ in range(3)]

    def gh(rng: random.Random) -> list[dict]:
        # values in a tight range with a large L => the window never shrinks => the naive expand-window
        # is O(N^2) and TLEs at N~30000; the two-deque ref is O(N).
        return [{"N": (n := _jit(rng, n_hidden)), "L": 10 ** 9,
                 "A": [rng.randint(1, 5) for _ in range(n)]} for _ in range(3)]

    return make_pn_template(
        name=f"b_count_subarrays_range_le_l_n{n_hidden}",
        family="count_subarrays_range_le_l",
        mode=MODE_COMPLEXITY_BLIND, statement=stmt, ref_source=ref, naive_source=naive,
        brute_source=naive, algo_sig="two_deque_count_bounded_range", discriminator=DISC_TIMEOUT,
        brute_cap_tokens=60, shape=shape, gen_public_data=gp, gen_hidden_data=gh,
        headroom_note=f"naive O(N^2) expand-window TLEs at N~{n_hidden}; needs two monotonic deques O(N)")


# ============================================================ NON-extractable negative control

def _cx_count_subarrays_sum_divisible_k(n_hidden: int) -> ParserNeutralTemplateV2:
    """NEGATIVE CONTROL for the G3 extractability gate.  Count contiguous blocks whose sum is divisible
    by K — the efficient technique is the prefix-mod FREQUENCY map.  The accumulator ``cnt += freq[r]``
    has NO gating predicate (it sits at the bare loop-body level); the discriminating logic lives in the
    DICT MAINTENANCE (``freq[r]+=1``), which the extractor cannot hole.  ⇒ G3 must REJECT this (no
    gated-accumulator / would leak the technique), machine-checking the W142 de-risk finding."""
    shape = io_shape(scalar_line("N", "K"), array_line("A", "N"))
    ref = ("import sys\n"
           "from collections import defaultdict\n"
           "d=sys.stdin.buffer.read().split()\n"
           "n=int(d[0]);K=int(d[1]);a=[int(x) for x in d[2:2+n]]\n"
           "freq=defaultdict(int);freq[0]=1;pre=0;cnt=0\n"
           "for x in a:\n"
           "    pre=(pre+x)%K\n"
           "    cnt+=freq[pre]\n"
           "    freq[pre]+=1\n"
           "print(cnt)\n")
    naive = ("import sys\n"
             "d=sys.stdin.buffer.read().split()\n"
             "n=int(d[0]);K=int(d[1]);a=[int(x) for x in d[2:2+n]]\n"
             "cnt=0\n"
             "for i in range(n):\n"
             "    s=0\n"
             "    for j in range(i,n):\n"
             "        s+=a[j]\n"
             "        if s%K==0:cnt+=1\n"
             "print(cnt)\n")
    stmt = ("Tally of clearing blocks.\n\n"
            "A contiguous non-empty block of the sequence CLEARS K when the total of its entries is "
            "divisible by K. You are given N entries and the divisor K; report how many clearing blocks "
            "the sequence contains.\n\n"
            "Input: line 1 contains N and K; line 2 contains N integers A[1..N].\n"
            "Output: how many clearing blocks there are.\n"
            "Constraints: 1 <= N <= 200000, 1 <= A[i] <= 10^9, 1 <= K <= 10^9.")

    def gp(rng: random.Random) -> list[dict]:
        return [{"N": (n := rng.randint(4, 9)), "K": rng.randint(2, 7),
                 "A": [rng.randint(1, 20) for _ in range(n)]} for _ in range(3)]

    def gh(rng: random.Random) -> list[dict]:
        return [{"N": (n := _jit(rng, n_hidden)), "K": rng.randint(2, 10 ** 9),
                 "A": [rng.randint(1, 10 ** 9) for _ in range(n)]} for _ in range(3)]

    return make_pn_template(
        name=f"b_count_subarrays_sum_divisible_k_n{n_hidden}", family="count_subarrays_sum_divisible_k",
        mode=MODE_COMPLEXITY_BLIND, statement=stmt, ref_source=ref, naive_source=naive,
        brute_source=naive, algo_sig="prefix_mod_frequency", discriminator=DISC_TIMEOUT,
        brute_cap_tokens=40, shape=shape, gen_public_data=gp, gen_hidden_data=gh,
        headroom_note=f"naive O(N^2) running-sum scan TLEs at N~{n_hidden}; needs prefix-mod frequency map O(N)")


# ============================================================ slate registry

NEW_CX_FACTORIES: dict[str, Callable[[int], ParserNeutralTemplateV2]] = {
    "count_pairs_product_le_t": _cx_count_pairs_product_le_t,
    "count_triples_sum_lt_t": _cx_count_triples_sum_lt_t,
    "count_subarrays_range_le_l": _cx_count_subarrays_range_le_l,
}
# G3 negative controls (extractability gate MUST reject these):
NEG_CONTROL_FACTORIES: dict[str, Callable[[int], ParserNeutralTemplateV2]] = {
    "count_subarrays_sum_divisible_k": _cx_count_subarrays_sum_divisible_k,   # prefix-hash, no gate
    "kth_smallest_pair_distance": EXTRA_CX_FACTORIES["kth_smallest_pair_distance"],  # BSoA, no accumulator
}


@dataclasses.dataclass(frozen=True)
class ScreenCandidateV1:
    family: str
    vein: str
    mode: str
    knob: int
    factory: Callable[[int], ParserNeutralTemplateV2]
    expect_extractable: bool       # the LOCKED pre-screen prediction (G3 must match this)
    note: str = ""


# the W142 screen slate (LOCKED order).  Knobs: counting/two-deque at the COMPLEXITY grid; triples
# self-caps internally.  expect_extractable is the pre-registered prediction the $0 G3 gate verifies.
def build_screen_slate_v1(*, knob: int = 50_000) -> list[ScreenCandidateV1]:
    out: list[ScreenCandidateV1] = []
    # extractable counting-pair vein
    out.append(ScreenCandidateV1("count_pairs_sum_le_t", VEIN_SORT_TWO_POINTER, MODE_COMPLEXITY_BLIND,
                                 knob, CX_FACTORIES["count_pairs_sum_le_t"], True, "W141 win, re-confirm p"))
    out.append(ScreenCandidateV1("count_pairs_absdiff_le_d", VEIN_SORT_TWO_POINTER, MODE_COMPLEXITY_BLIND,
                                 knob, CX_FACTORIES["count_pairs_absdiff_le_d"], True, "sibling counting-pair"))
    out.append(ScreenCandidateV1("count_pairs_product_le_t", VEIN_SORT_TWO_POINTER, MODE_COMPLEXITY_BLIND,
                                 knob, NEW_CX_FACTORIES["count_pairs_product_le_t"], True, "NEW product monotonicity"))
    out.append(ScreenCandidateV1("count_triples_sum_lt_t", VEIN_SORT_TWO_POINTER, MODE_COMPLEXITY_BLIND,
                                 knob, NEW_CX_FACTORIES["count_triples_sum_lt_t"], True, "NEW O(N^3)->O(N^2)"))
    # two-deque vein (distinct technique class; W140-corroborated)
    out.append(ScreenCandidateV1("count_subarrays_range_le_l", VEIN_TWO_DEQUE, MODE_COMPLEXITY_BLIND,
                                 knob, NEW_CX_FACTORIES["count_subarrays_range_le_l"], True,
                                 "NEW two-deque COMPLEXITY (counting form, gated shrink)"))
    out.append(ScreenCandidateV1("subarrays_sum_and_range", VEIN_TWO_DEQUE,
                                 FUNC_FACTORIES["subarrays_sum_and_range"](knob).minted.mode,
                                 30_000, FUNC_FACTORIES["subarrays_sum_and_range"], True,
                                 "existing two-deque HIDDEN_EDGE; W140 A1=25%"))
    # monotonic-stack known-high anchor
    out.append(ScreenCandidateV1("sum_nearest_smaller_left", VEIN_MONOTONIC_STACK, MODE_COMPLEXITY_BLIND,
                                 knob, CX_FACTORIES["sum_nearest_smaller_left"], True, "W141 p=0.67 high anchor"))
    # NON-extractable negative controls (G3 must REJECT)
    out.append(ScreenCandidateV1("count_subarrays_sum_divisible_k", VEIN_PREFIX_HASH, MODE_COMPLEXITY_BLIND,
                                 knob, NEG_CONTROL_FACTORIES["count_subarrays_sum_divisible_k"], False,
                                 "G3 control: prefix-hash, no gated accumulator"))
    out.append(ScreenCandidateV1("kth_smallest_pair_distance", VEIN_BINARY_SEARCH_ANSWER, MODE_COMPLEXITY_BLIND,
                                 knob, NEG_CONTROL_FACTORIES["kth_smallest_pair_distance"], False,
                                 "G3 control: BSoA, printed answer is not an accumulator"))
    return out


def screen_slate_fingerprint_cid_v1(*, knob: int = 50_000) -> str:
    cands = build_screen_slate_v1(knob=knob)
    return _sha256_hex({"k": "w142_moderate_p_screen_slate_v1",
                        "band": [MODERATE_P_LO, MODERATE_P_HI],
                        "cells": [{"family": c.family, "vein": c.vein, "mode": c.mode, "knob": c.knob,
                                   "algo_sig": c.factory(c.knob).minted.algo_sig,
                                   "stmt": c.factory(c.knob).minted.statement,
                                   "ref": c.factory(c.knob).minted.ref_source,
                                   "expect_extractable": c.expect_extractable} for c in cands]})


__all__ = [
    "MODERATE_P_FAMILY_SLATE_V1_SCHEMA_VERSION", "MODERATE_P_LO", "MODERATE_P_HI",
    "VEIN_SORT_TWO_POINTER", "VEIN_TWO_DEQUE", "VEIN_MONOTONIC_STACK", "VEIN_PREFIX_HASH",
    "VEIN_BINARY_SEARCH_ANSWER", "NEW_CX_FACTORIES", "NEG_CONTROL_FACTORIES",
    "ScreenCandidateV1", "build_screen_slate_v1", "screen_slate_fingerprint_cid_v1",
]
