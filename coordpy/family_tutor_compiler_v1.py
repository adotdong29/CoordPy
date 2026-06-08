"""W140 / COO-9 — family-level TUTOR compiler + deterministic no-leakage gate (Lane α).

W139 ACHIEVED non-negativity on every tier but the witness gain stayed anchor-concentrated:
the weak 8B has ``witness_usability_rate = 0.00`` — it is told "your O(N^2) is too slow" but cannot
ACT on the bare per-problem diagnostic (it writes broken efficient code, 0/5 repair).  The W139 fix
was to KEEP (do no harm); W140 attacks USABILITY: compile the anchor's winning witness into a
FAMILY-LEVEL TUTOR a weak model can actually consume.

The compiler turns the OWNED-ORACLE structure of a family (its ``headroom_note`` + ``algo_sig`` +
a generic, leak-audited technique scaffold) into a teaching object.  Primary-source grounding (each
wired to a design choice, see ``docs/RESULTS_W140_RESEARCH_V1.md``):
  * small models REFINE well but FAIL at verification (arXiv:2404.17140) ⇒ the tutor CARRIES the
    execution-oracle routing (TC2 witness→rewrite), it does not merely name a technique;
  * principle-only hints MISLEAD weak learners; the winning object is two-layer name + worked
    skeleton (arXiv:2404.02213 CHI'24; arXiv:2410.09008 ICLR'25 lifted a 7B with exactly this) ⇒
    TC1 carries a key-move + TC2 a holed skeleton;
  * minimal property-oriented feedback beats verbose for weak learners (arXiv:2506.18315;
    arXiv:2502.12143) ⇒ TC3 compressed;
  * the winning mechanism DIFFERS by capability and a strong-tier curriculum can HARM the weak tier
    (arXiv:2503.08681; arXiv:2510.06101) ⇒ the bench routes the tutor by a measured per-tier
    tutor-usability prior and KEEPs where it does not lift (the W139 controller, fed a tutor);
  * the decisive leak guarantee is BEHAVIORAL (hidden tests disjoint), because text leak-detectors
    are defeatable (arXiv:2402.02823) and exact-match recall is ~0.075 (arXiv:2510.21087) ⇒ the
    grader stays ``grade_on_secret_v1`` on a DISJOINT hidden bank; the text gate is corroboration.

NO-LEAKAGE CONTRACT (``tutor_leak_gate_v1``, deterministic, $0): a tutor is FAMILY-LEVEL or
TRAIN-derived, never the target answer.  The DISCRIMINATING logic (the predicate/aggregation that
separates the correct algorithm from the naive) is ALWAYS a load-bearing HOLE the model must fill;
the gate asserts (1) no reference paste (bounded verbatim run vs ``ref_source``), (2) public-only
literals (no secret-case literal), (3) a PROCEDURE not a closed-form answer, (4) hole-substance (the
skeleton with holes trivially stubbed FAILS public ⇒ the holes carry the answer-logic), (5) the
one-liner-family guard (cards disabled where naming == coding), (6) discriminator-shape-only
descriptions, (7) template/instance invariance (no instance constants).  Reuses the W127-corrected
CONTIGUOUS-block tripwire (never the per-line check that false-positived on boilerplate).

Pure / deterministic except the (already-audited) program-execution subprocess used by the
hole-substance check; NO model inference lives here (that is the W140 driver/bench).  Explicit-import
only; ``coordpy/__init__.py`` untouched.
"""
from __future__ import annotations

import dataclasses
import re
from typing import Any, Optional, Sequence

from .resistant_by_construction_battlefield_v1 import (
    MintedProblemV1, _exec_capture_v1, _sha256_hex, _tok_count)
from .hard_battlefield_slate_v2 import ParserNeutralTemplateV2

FAMILY_TUTOR_COMPILER_V1_SCHEMA_VERSION: str = "coordpy.family_tutor_compiler_v1.v1"

# tutor kinds (the §4b TC slate)
TC1_CARD: str = "TC1_FAMILY_ALGORITHM_CARD"
TC2_REWRITE: str = "TC2_WITNESS_TO_REWRITE"
TC3_COMPRESSED: str = "TC3_COMPRESSED_STRESS"
TC5_STAGED: str = "TC5_WEAK_TIER_STAGED"
T6_NEG: str = "T6_NEGATIVE_CONTROL"

# observed-failure kinds the witness→rewrite routing keys on (mirrors exact_oracle_witness_v1)
OBS_TIMEOUT: str = "TIMEOUT"
OBS_WRONG_ANSWER: str = "WRONG_ANSWER"

HOLE_RE = re.compile(r"__HOLE_[A-Z0-9_]+__")
# leak-gate thresholds (asserted falsifiable in the self-test: legit tutors pass, planted leaks bite).
# The DECISIVE checks are no_discriminator_leak (the answer-distinguishing expression is absent /
# always a hole) + holes_are_substantive (the tutor's own skeleton stubbed FAILS public).  The
# contiguous-run cap is a SECONDARY gross-paste tripwire: a legit holed skeleton may share the I/O
# boilerplate + the standard monotonic-deque/two-pointer idiom with ref (a teachable PRIMITIVE), so
# the cap is generous and only a near-complete paste (which additionally carries the discriminator)
# exceeds it.
MAX_VERBATIM_REF_RUN_TOKENS: int = 50   # secondary gross-paste tripwire (idiom/boilerplate is OK)
MIN_DISCRIMINATOR_RUN_TOKENS: int = 5   # a discriminating expr of >=5 norm-tokens present => leak
TC3_TOKEN_BUDGET: int = 130             # the compressed tutor's hard model-facing token cap


# ===================================================== per-technique generic content (leak-audited)

@dataclasses.dataclass(frozen=True)
class TechniqueSpecV1:
    """A GENERIC, family-LEVEL technique scaffold keyed by ``algo_sig`` — textbook-pattern content,
    never an instance solution.  ``skeleton`` is a holed Python template whose DISCRIMINATING
    decisions are ``__HOLE_*__`` markers; ``correct_fill`` (used ONLY at $0 self-test to prove the
    scaffold is completable to a correct program — NEVER shown to a model) and ``trivial_fill``
    (used to prove the holes are load-bearing) are NOT part of the model-facing tutor."""
    algo_sig: str
    technique_name: str             # e.g. "sort + two-pointer"
    key_move: str                   # the single structural insight (one line, family-level)
    primitive_hint: str
    bug_warnings: tuple[str, ...]
    invariants: tuple[str, ...]
    skeleton: str                   # holed Python template (model-facing in TC2)
    correct_fill: dict[str, str]    # hole -> correct expression (self-test only; never emitted)
    trivial_fill: dict[str, str]    # hole -> trivial stub (leak gate hole-substance only)

    def holes(self) -> tuple[str, ...]:
        return tuple(sorted(set(HOLE_RE.findall(self.skeleton))))

    def fill(self, mapping: dict[str, str]) -> str:
        out = self.skeleton
        for h, v in mapping.items():
            out = out.replace(h, v)
        return out


# The technique library — one entry per algo_sig used by the W140 shared families (+ hedges for the
# other COMPLEXITY families the field can expand to).  Every skeleton blanks the DISCRIMINATOR.
TECHNIQUE_LIBRARY: dict[str, TechniqueSpecV1] = {
    # --- shared family A: count_pairs_sum_le_t (COMPLEXITY) ---
    "sort_two_pointer_pairsum": TechniqueSpecV1(
        algo_sig="sort_two_pointer_pairsum",
        technique_name="sort + two-pointer",
        key_move=("After sorting, sweep one pointer from the left and one from the right; when the "
                  "two ends satisfy the threshold, EVERY pair between them does too, so you can add "
                  "a whole block of pairs at once instead of looping over them."),
        primitive_hint="list.sort() is O(N log N); the two-pointer sweep is one O(N) pass after it.",
        bug_warnings=(
            "Do NOT keep the O(N^2) double loop — it is the thing that times out.",
            "When the ends pass, add the BLOCK count, not 1 (you accept many pairs at once).",
            "Pairs are unordered — advance exactly one pointer per step so you never double count.",
        ),
        invariants=(
            "PRE: the array is sorted ascending.",
            "INV: every pair with left index < i has already been fully counted.",
        ),
        skeleton=(
            "import sys\n"
            "data = sys.stdin.buffer.read().split()\n"
            "n = int(data[0]); T = int(data[1])\n"
            "A = [int(x) for x in data[2:2+n]]\n"
            "A.sort()\n"
            "i = 0; j = n - 1; cnt = 0\n"
            "while i < j:\n"
            "    if __HOLE_PRED__:        # do the two ends satisfy the budget?\n"
            "        cnt += __HOLE_BLOCK__  # how many pairs does accepting (i, j) add at once?\n"
            "        i += 1\n"
            "    else:\n"
            "        j -= 1\n"
            "print(cnt)\n"),
        correct_fill={"__HOLE_PRED__": "A[i] + A[j] <= T", "__HOLE_BLOCK__": "j - i"},
        trivial_fill={"__HOLE_PRED__": "True", "__HOLE_BLOCK__": "0"},
    ),
    # --- shared family B: subarrays_sum_and_range (HIDDEN_EDGE) ---
    "two_pointer_two_constraint_deques": TechniqueSpecV1(
        algo_sig="two_pointer_two_constraint_deques",
        technique_name="sliding window with a max-deque and a min-deque",
        key_move=("Slide a window and shrink it while EITHER constraint is violated — the sum cap OR "
                  "the spread cap. The naive that checks only the sum silently drops the spread "
                  "constraint, which is exactly the hidden bug."),
        primitive_hint=("collections.deque gives O(1) ends; keep one decreasing (window max) and one "
                        "increasing (window min) so max-min is O(1)."),
        bug_warnings=(
            "Do NOT drop the (max - min) <= R check — it is the hidden discriminator the naive misses.",
            "Shrink while EITHER cap is exceeded (an OR condition), not only the sum cap.",
            "Maintain max and min incrementally with the deques; do not recompute them O(N) each step.",
        ),
        invariants=(
            "INV: after the shrink step the window [l, r] satisfies BOTH caps.",
            "INV: the deques hold window indices in monotone order (max decreasing, min increasing).",
        ),
        skeleton=(
            "import sys\n"
            "from collections import deque\n"
            "data = sys.stdin.buffer.read().split()\n"
            "n = int(data[0]); S = int(data[1]); R = int(data[2])\n"
            "A = [int(x) for x in data[3:3+n]]\n"
            "l = 0; cur = 0; cnt = 0\n"
            "mx = deque(); mn = deque()\n"
            "for r in range(n):\n"
            "    cur += A[r]\n"
            "    while mx and A[mx[-1]] <= A[r]: mx.pop()\n"
            "    mx.append(r)\n"
            "    while mn and A[mn[-1]] >= A[r]: mn.pop()\n"
            "    mn.append(r)\n"
            "    while l <= r and (__HOLE_SHRINK__):  # shrink while EITHER cap is violated\n"
            "        cur -= A[l]\n"
            "        if mx[0] == l: mx.popleft()\n"
            "        if mn[0] == l: mn.popleft()\n"
            "        l += 1\n"
            "    cnt += __HOLE_ADD__  # how many valid windows END at r?\n"
            "print(cnt)\n"),
        correct_fill={"__HOLE_SHRINK__": "cur > S or A[mx[0]] - A[mn[0]] > R",
                      "__HOLE_ADD__": "r - l + 1"},
        trivial_fill={"__HOLE_SHRINK__": "False", "__HOLE_ADD__": "0"},
    ),
    # --- hedges for the other COMPLEXITY families (technique-level, not fielded unless expanded) ---
    "monotonic_stack_prev_smaller": TechniqueSpecV1(
        algo_sig="monotonic_stack_prev_smaller",
        technique_name="monotonic stack",
        key_move=("Keep an increasing stack; before pushing A[i], pop everything >= A[i]; the new top "
                  "is the nearest strictly-smaller element to the left — O(N) total, no inner scan."),
        primitive_hint="A plain list used as a stack (append / pop / [-1]) is enough.",
        bug_warnings=(
            "Do NOT rescan leftwards for each i — that is the O(N^2) that times out.",
            "Pop with the correct strictness (>=) before reading the new top.",
            "Use -1 when the stack is empty after popping.",
        ),
        invariants=("INV: the stack is strictly increasing bottom-to-top over a subsequence of seen values.",),
        skeleton=(
            "import sys\n"
            "data = sys.stdin.buffer.read().split()\n"
            "n = int(data[0]); A = [int(x) for x in data[1:1+n]]\n"
            "st = []; total = 0\n"
            "for v in A:\n"
            "    while __HOLE_POP__:   # pop until the top can be the nearest-smaller-left\n"
            "        st.pop()\n"
            "    total += __HOLE_TAKE__  # the nearest smaller to the left, or -1\n"
            "    st.append(v)\n"
            "print(total)\n"),
        correct_fill={"__HOLE_POP__": "st and st[-1] >= v", "__HOLE_TAKE__": "st[-1] if st else -1"},
        trivial_fill={"__HOLE_POP__": "False", "__HOLE_TAKE__": "0"},
    ),
    # --- hard KNOWLEDGE-gap families (8B A1≈0 unprompted; W140 lift-regime, W140-iter2) ---
    "prefix_min_suffix_max_two_pointer": TechniqueSpecV1(
        algo_sig="prefix_min_suffix_max_two_pointer",
        technique_name="prefix-min + suffix-max with a two-pointer walk",
        key_move=("Precompute lmin[i] = the minimum of A[0..i] and rmax[j] = the maximum of "
                  "A[j..n-1]; then walk two pointers i, j: when a valid widest pair can span [i, j], "
                  "record j-i and advance j; otherwise advance i. O(N), no O(N^2) all-pairs scan."),
        primitive_hint="Two prefix arrays + two integer pointers; no extra structure.",
        bug_warnings=(
            "Do NOT scan all O(N^2) index pairs — that times out.",
            "lmin is a running MIN from the left; rmax is a running MAX from the right.",
            "Advance j when the pair is feasible (to widen), else advance i.",
        ),
        invariants=("INV: lmin is non-increasing, rmax is non-decreasing along the walk.",),
        skeleton=(
            "import sys\n"
            "data = sys.stdin.buffer.read().split()\n"
            "n = int(data[0]); A = [int(x) for x in data[1:1+n]]\n"
            "lmin = [0]*n; rmax = [0]*n\n"
            "lmin[0] = A[0]\n"
            "for i in range(1, n): lmin[i] = lmin[i-1] if lmin[i-1] < A[i] else A[i]\n"
            "rmax[n-1] = A[n-1]\n"
            "for i in range(n-2, -1, -1): rmax[i] = rmax[i+1] if rmax[i+1] > A[i] else A[i]\n"
            "i = 0; j = 0; best = 0\n"
            "while i < n and j < n:\n"
            "    if __HOLE_FEASIBLE__:   # can a qualifying pair span the window [i, j]?\n"
            "        if j - i > best: best = j - i\n"
            "        j += 1\n"
            "    else:\n"
            "        i += 1\n"
            "print(best)\n"),
        correct_fill={"__HOLE_FEASIBLE__": "lmin[i] <= rmax[j]"},
        trivial_fill={"__HOLE_FEASIBLE__": "False"},
    ),
    "binary_search_answer_two_pointer": TechniqueSpecV1(
        algo_sig="binary_search_answer_two_pointer",
        technique_name="binary search on the answer + two-pointer count",
        key_move=("Sort, then BINARY-SEARCH the answer value d: for a candidate d, count how many "
                  "pairs have gap <= d with an O(N) two-pointer sweep; move the search toward the "
                  "smallest d whose count reaches the target. O(N log maxA), no O(N^2) all-pairs."),
        primitive_hint="Sort + a count helper + an integer binary search on the value range.",
        bug_warnings=(
            "Do NOT materialise all O(N^2) pair gaps and sort them — that times out.",
            "The count helper is monotone in d; that is what makes binary search valid.",
            "Search the VALUE range [0, max-min], not indices.",
        ),
        invariants=("INV: cnt_le(d) is non-decreasing in d; the search keeps the feasible half.",),
        skeleton=(
            "import sys\n"
            "data = sys.stdin.buffer.read().split()\n"
            "n = int(data[0]); K = int(data[1]); A = [int(x) for x in data[2:2+n]]\n"
            "A.sort()\n"
            "def cnt_le(dd):\n"
            "    c = 0; l = 0\n"
            "    for r in range(n):\n"
            "        while A[r] - A[l] > dd: l += 1\n"
            "        c += __HOLE_COUNT__   # pairs ending at r with gap <= dd\n"
            "    return c\n"
            "lo = 0; hi = A[-1] - A[0]\n"
            "while lo < hi:\n"
            "    mid = (lo + hi) // 2\n"
            "    if __HOLE_DIRECTION__:   # is mid already large enough?\n"
            "        hi = mid\n"
            "    else:\n"
            "        lo = mid + 1\n"
            "print(lo)\n"),
        correct_fill={"__HOLE_COUNT__": "r - l", "__HOLE_DIRECTION__": "cnt_le(mid) >= K"},
        trivial_fill={"__HOLE_COUNT__": "0", "__HOLE_DIRECTION__": "False"},
    ),
    "two_pointer_count_subarrays": TechniqueSpecV1(
        algo_sig="two_pointer_count_subarrays",
        technique_name="two-pointer sliding window",
        key_move=("Grow r, and while the window sum exceeds S shrink from l; every window ending at r "
                  "contributes (r - l + 1) valid subarrays — one O(N) pass, not O(N^2)."),
        primitive_hint="Two integer indices and a running sum; no extra structure needed.",
        bug_warnings=(
            "Do NOT re-sum every subarray from scratch — that is the O(N^2) that times out.",
            "Add (r - l + 1) per r, not 1.",
            "Shrink with a while, not an if (the sum can exceed S by a lot).",
        ),
        invariants=("INV: after shrinking, [l, r] is the longest window ending at r with sum <= S.",),
        skeleton=(
            "import sys\n"
            "data = sys.stdin.buffer.read().split()\n"
            "n = int(data[0]); S = int(data[1])\n"
            "A = [int(x) for x in data[2:2+n]]\n"
            "l = 0; cur = 0; cnt = 0\n"
            "for r in range(n):\n"
            "    cur += A[r]\n"
            "    while __HOLE_SHRINK__:   # shrink while the window breaks the cap\n"
            "        cur -= A[l]; l += 1\n"
            "    cnt += __HOLE_ADD__       # windows ending at r\n"
            "print(cnt)\n"),
        correct_fill={"__HOLE_SHRINK__": "cur > S", "__HOLE_ADD__": "r - l + 1"},
        trivial_fill={"__HOLE_SHRINK__": "False", "__HOLE_ADD__": "0"},
    ),
}


# ===================================================== the compiled tutor object

@dataclasses.dataclass(frozen=True)
class FamilyTutorV1:
    """A leak-audited, family-LEVEL teaching object the bench injects as STATIC prompt text
    (compiled $0 outside the K budget; the same object is reused for every instance of the family)."""
    family: str
    algo_sig: str
    tc_kind: str
    technique_name: str
    budget_fact: str                       # when/why the naive fails (from the owned headroom_note)
    key_move: str
    primitive_hint: str
    bug_warnings: tuple[str, ...]
    invariants: tuple[str, ...]
    skeleton: str                          # "" for TC1/TC3 card-only variants
    rewrite_routes: tuple[tuple[str, str], ...]  # (observed_kind, instruction) for TC2
    stages: tuple[str, ...]                # for TC5 (progressive)
    is_negative_control: bool = False

    def to_prompt_block(self, *, observed_kind: Optional[str] = None) -> str:
        """Render the model-facing teaching block.  For TC2, ``observed_kind`` selects the
        witness-routed rewrite instruction (family-level routing, not instance leakage)."""
        if self.is_negative_control:
            return ("=== Coaching note ===\n"
                    "Think carefully and write a better, more efficient complete program. "
                    "Avoid your previous mistake.")
        lines: list[str] = [f"=== Technique tutor for this problem family ==="]
        lines.append(f"Technique: {self.technique_name}.")
        if self.budget_fact:
            lines.append(f"Why a naive attempt fails here: {self.budget_fact}.")
        if self.key_move:
            lines.append(f"Key idea: {self.key_move}")
        if self.primitive_hint:
            lines.append(f"Tools: {self.primitive_hint}")
        if self.tc_kind == TC2_REWRITE and self.rewrite_routes:
            route = dict(self.rewrite_routes)
            instr = route.get(observed_kind or "", route.get("DEFAULT", ""))
            if instr:
                lines.append(f"What your last attempt got wrong: {instr}")
        if self.bug_warnings and self.tc_kind != TC3_COMPRESSED:
            lines.append("Common mistakes to avoid:")
            lines.extend(f"  - {w}" for w in self.bug_warnings)
        if self.invariants and self.tc_kind not in (TC3_COMPRESSED,):
            lines.append("Keep these invariants true:")
            lines.extend(f"  - {v}" for v in self.invariants)
        if self.skeleton:
            lines.append("Fill in the blanks of this technique skeleton (the blanks are the "
                         "decisions that make it correct — work them out for THIS problem; do not "
                         "just copy):")
            lines.append(f"```python\n{self.skeleton}```")
        return "\n".join(lines)

    def model_facing_text(self) -> str:
        """The full text a model could ever see from this tutor (used by the leak gate)."""
        if self.tc_kind == TC2_REWRITE and self.rewrite_routes:
            return "\n".join(self.to_prompt_block(observed_kind=k)
                             for k, _ in self.rewrite_routes) + "\n" + self.to_prompt_block()
        return self.to_prompt_block()

    def token_count(self) -> int:
        return _tok_count(self.model_facing_text())

    def cid(self) -> str:
        return _sha256_hex({"k": "w140_family_tutor_v1", "family": self.family,
                            "algo_sig": self.algo_sig, "tc": self.tc_kind,
                            "text": self.model_facing_text()})

    def to_dict(self) -> dict[str, Any]:
        return {"family": self.family, "algo_sig": self.algo_sig, "tc_kind": self.tc_kind,
                "technique_name": self.technique_name, "has_skeleton": bool(self.skeleton),
                "n_bug_warnings": len(self.bug_warnings), "n_invariants": len(self.invariants),
                "is_negative_control": bool(self.is_negative_control),
                "token_count": self.token_count(), "tutor_cid": self.cid()}


def _sanitize_scale(text: str) -> str:
    """Strip >=4-digit numbers (the family knob N) from a card field — the knob is a family-level
    SCALE statement, but some families' arrays literally contain the knob value (e.g. range(n,0,-1)),
    which would false-positive the secret-literal guard.  The exact N carries no pedagogical value."""
    return re.sub(r"\d{4,}", "large N", text or "")


def _parse_headroom_note(note: str) -> tuple[str, str]:
    """Deterministically split the owned-oracle ``headroom_note`` into (why_naive_fails, needs), with
    the knob scale sanitized out of the why-clause.
    e.g. 'naive O(N^2) pair scan TLEs at N~50000; needs sort + two-pointer O(N log N)' ->
    ('naive O(N^2) pair scan TLEs at N~large N', 'sort + two-pointer O(N log N)')."""
    note = (note or "").strip()
    if "; needs " in note:
        why, needs = note.split("; needs ", 1)
    elif " needs " in note:
        why, needs = note.split(" needs ", 1)
    else:
        why, needs = note, ""
    return _sanitize_scale(why.strip().rstrip(".")), needs.strip().rstrip(".")


def _spec_for(template: ParserNeutralTemplateV2) -> Optional[TechniqueSpecV1]:
    return TECHNIQUE_LIBRARY.get(template.minted.algo_sig)


# ===================================================== the TC compilers

def compile_family_card_v1(template: ParserNeutralTemplateV2) -> FamilyTutorV1:
    """TC1 — deterministic family algorithm card from the owned ``headroom_note`` + ``algo_sig`` +
    the generic technique spec.  No code skeleton (lowest leak; tests whether NAMING the technique
    is the signal a weak model can act on)."""
    spec = _spec_for(template)
    why, _needs = _parse_headroom_note(template.headroom_note)
    if spec is None:  # unknown technique — degrade to the owned-oracle note only (still family-level)
        return FamilyTutorV1(
            family=template.minted.family, algo_sig=template.minted.algo_sig, tc_kind=TC1_CARD,
            technique_name=_parse_headroom_note(template.headroom_note)[1] or "an efficient method",
            budget_fact=why, key_move="", primitive_hint="", bug_warnings=(), invariants=(),
            skeleton="", rewrite_routes=(), stages=())
    return FamilyTutorV1(
        family=template.minted.family, algo_sig=spec.algo_sig, tc_kind=TC1_CARD,
        technique_name=spec.technique_name, budget_fact=why, key_move=spec.key_move,
        primitive_hint=spec.primitive_hint, bug_warnings=spec.bug_warnings,
        invariants=spec.invariants, skeleton="", rewrite_routes=(), stages=())


def compile_witness_rewrite_tutor_v1(template: ParserNeutralTemplateV2) -> FamilyTutorV1:
    """TC2 — the family card PLUS a witness→rewrite routing (observed TLE / WRONG_ANSWER -> the
    named root cause + a specific rewrite) PLUS the holed technique skeleton.  Carries the oracle
    signal (arXiv:2404.17140) and a worked scaffold (arXiv:2404.02213) — the highest-lift arm."""
    spec = _spec_for(template)
    why, _ = _parse_headroom_note(template.headroom_note)
    if spec is None:
        return compile_family_card_v1(template)
    routes = (
        (OBS_TIMEOUT, f"your program is too slow (it times out on large inputs) — replace the "
                      f"brute approach with {spec.technique_name}."),
        (OBS_WRONG_ANSWER, f"your program returns the wrong count on a hidden input — it is missing "
                           f"part of the logic ({spec.key_move.split('.')[0].lower()})."),
        ("DEFAULT", f"fix your approach using {spec.technique_name}."),
    )
    return FamilyTutorV1(
        family=template.minted.family, algo_sig=spec.algo_sig, tc_kind=TC2_REWRITE,
        technique_name=spec.technique_name, budget_fact=why, key_move=spec.key_move,
        primitive_hint=spec.primitive_hint, bug_warnings=spec.bug_warnings,
        invariants=spec.invariants, skeleton=spec.skeleton, rewrite_routes=routes, stages=())


def compile_compressed_tutor_v1(template: ParserNeutralTemplateV2) -> FamilyTutorV1:
    """TC3 — the smallest sufficient object (one technique + one key move + one bug warning), under a
    hard token budget (arXiv:2506.18315 / arXiv:2502.12143: minimal beats verbose for weak learners)."""
    spec = _spec_for(template)
    why, _ = _parse_headroom_note(template.headroom_note)
    if spec is None:
        return compile_family_card_v1(template)
    one_warn = (spec.bug_warnings[0],) if spec.bug_warnings else ()
    return FamilyTutorV1(
        family=template.minted.family, algo_sig=spec.algo_sig, tc_kind=TC3_COMPRESSED,
        technique_name=spec.technique_name, budget_fact="", key_move=spec.key_move,
        primitive_hint="", bug_warnings=one_warn, invariants=(), skeleton="",
        rewrite_routes=(), stages=())


def compile_staged_tutor_v1(template: ParserNeutralTemplateV2) -> FamilyTutorV1:
    """TC5 — progressive disclosure (stage strings, least->most); the controller emits the minimum
    stage and escalates on a failed execution.  Stage budget = highest stage reached."""
    spec = _spec_for(template)
    if spec is None:
        return compile_family_card_v1(template)
    base = compile_compressed_tutor_v1(template)
    s1 = compile_family_card_v1(template)
    s2 = compile_witness_rewrite_tutor_v1(template)
    stages = (base.to_prompt_block(), s1.to_prompt_block(),
              s2.to_prompt_block(observed_kind="DEFAULT"))
    return dataclasses.replace(s2, tc_kind=TC5_STAGED, stages=stages)


def make_negative_control_tutor_v1(template: ParserNeutralTemplateV2) -> FamilyTutorV1:
    """T6 — a deliberately content-free 'tutor' (a generic do-better instruction, NO technique).
    The fake-different test MUST classify it FAKE_DIFFERENT."""
    return FamilyTutorV1(
        family=template.minted.family, algo_sig=template.minted.algo_sig, tc_kind=T6_NEG,
        technique_name="", budget_fact="", key_move="", primitive_hint="", bug_warnings=(),
        invariants=(), skeleton="", rewrite_routes=(), stages=(), is_negative_control=True)


# ===================================================== the deterministic no-leakage gate

def _norm_tokens(text: str) -> list[str]:
    """Lowercase alphanumeric token stream (whitespace/punct collapsed) — defeats cosmetic restyle."""
    return re.findall(r"[a-z0-9_]+", (text or "").lower())


def _longest_common_run(a: Sequence[str], b: Sequence[str]) -> int:
    """Longest contiguous shared token run (the W127-corrected CONTIGUOUS-block tripwire)."""
    if not a or not b:
        return 0
    bset_index: dict[str, list[int]] = {}
    for j, t in enumerate(b):
        bset_index.setdefault(t, []).append(j)
    best = 0
    prev: dict[int, int] = {}
    for i, t in enumerate(a):
        cur: dict[int, int] = {}
        for j in bset_index.get(t, ()):
            run = prev.get(j - 1, 0) + 1
            cur[j] = run
            if run > best:
                best = run
        prev = cur
    return best


def _numeric_literals(text: str) -> set[str]:
    return set(re.findall(r"\d+", text or ""))


@dataclasses.dataclass(frozen=True)
class TutorLeakReportV1:
    tutor_cid: str
    family: str
    tc_kind: str
    no_discriminator_leak: bool     # (DECISIVE) the answer-distinguishing expr is absent / a hole
    leaked_discriminators: tuple[str, ...]
    holes_are_substantive: bool     # (DECISIVE) the tutor's OWN skeleton stubbed FAILS public
    no_reference_paste: bool        # (secondary) bounded gross verbatim run vs ref_source
    longest_ref_run: int
    public_only_literals: bool      # no secret-only numeric literal in the tutor
    leaked_literals: tuple[str, ...]
    is_procedure_not_answer: bool   # control-flow present, no closed-form == public output
    one_liner_family_ok: bool       # ref non-trivial -> card permitted
    discriminator_shape_only: bool  # no concrete array literal in any 'fails on' text
    template_invariant: bool        # no instance constant (statement/secret) in the tutor

    @property
    def leaked(self) -> bool:
        return not (self.no_discriminator_leak and self.holes_are_substantive
                    and self.no_reference_paste and self.public_only_literals
                    and self.is_procedure_not_answer and self.one_liner_family_ok
                    and self.discriminator_shape_only and self.template_invariant)

    def to_dict(self) -> dict[str, Any]:
        return {"tutor_cid": self.tutor_cid, "family": self.family, "tc_kind": self.tc_kind,
                "leaked": self.leaked, "no_discriminator_leak": self.no_discriminator_leak,
                "leaked_discriminators": list(self.leaked_discriminators),
                "holes_are_substantive": self.holes_are_substantive,
                "no_reference_paste": self.no_reference_paste, "longest_ref_run": self.longest_ref_run,
                "public_only_literals": self.public_only_literals,
                "leaked_literals": list(self.leaked_literals),
                "is_procedure_not_answer": self.is_procedure_not_answer,
                "one_liner_family_ok": self.one_liner_family_ok,
                "discriminator_shape_only": self.discriminator_shape_only,
                "template_invariant": self.template_invariant}


def _ref_statement_count(ref_source: str) -> int:
    return sum(1 for ln in (ref_source or "").splitlines()
              if ln.strip() and not ln.strip().startswith(("import", "from", "#")))


def tutor_leak_gate_v1(tutor: FamilyTutorV1, template: ParserNeutralTemplateV2,
                       problem: MintedProblemV1, *, timeout_s: float = 8.0,
                       max_ref_run: int = MAX_VERBATIM_REF_RUN_TOKENS) -> TutorLeakReportV1:
    """Deterministic $0 no-leakage gate.  The DECISIVE guarantee is behavioral (grading on the
    disjoint hidden bank); this gate is the corroborating text+structure audit (§3 RUNBOOK_W140)."""
    text = tutor.model_facing_text()
    ttoks = _norm_tokens(text)
    ref = template.minted.ref_source
    spec = _spec_for(template)

    # (DECISIVE 1) no discriminator leak — the tutor must NOT contain any DISCRIMINATING expression
    # (the spec's correct hole-fills: the exact predicate/aggregation that separates the correct
    # algorithm from the naive) as a contiguous normalized token run.  In a legit tutor these are
    # ALWAYS holes; a reference paste reproduces them.  This is the principled "is the answer present?"
    # check (independent of how much shareable boilerplate/idiom the scaffold shows).
    leaked_disc: list[str] = []
    if spec is not None and not tutor.is_negative_control:
        for expr in spec.correct_fill.values():
            etoks = _norm_tokens(expr)
            if len(etoks) >= MIN_DISCRIMINATOR_RUN_TOKENS and \
                    _longest_common_run(ttoks, etoks) >= len(etoks):
                leaked_disc.append(expr)
    no_disc_leak = (len(leaked_disc) == 0)

    # (DECISIVE 2) hole-substance — the TUTOR'S OWN skeleton must have load-bearing holes: a nonempty
    # skeleton with NO holes is a full solution; a holed skeleton trivially stubbed must FAIL public.
    holes_ok = True
    sk = tutor.skeleton
    if sk:
        sk_holes = HOLE_RE.findall(sk)
        if not sk_holes:
            holes_ok = False  # a "skeleton" with no holes IS the answer
        elif spec is not None:
            stub = sk
            for h, v in spec.trivial_fill.items():
                stub = stub.replace(h, v)
            stub = HOLE_RE.sub("0", stub)  # any remaining holes -> trivial constant
            # test against the SECRET (hidden) bank, not just public — public answers can be
            # degenerate (e.g. all 0 when the threshold is small) and coincide with the trivial stub,
            # which would falsely pass; the secret bank is diverse and non-degenerate by construction.
            cases = list(problem.secret_cases) or list(problem.samples)
            stub_passes_all = bool(cases)
            for inp, exp in cases:
                r = _exec_capture_v1(stub, inp, timeout_s=float(timeout_s))
                if r.timed_out or r.returncode != 0 or r.stdout.strip() != exp.strip():
                    stub_passes_all = False
                    break
            holes_ok = not stub_passes_all  # stub passes the whole hidden bank -> holes carry no logic

    # (secondary) gross reference-paste tripwire — a FULL paste reproduces ~all of ref contiguously;
    # a legit holed scaffold may share the I/O boilerplate + the technique CONSTRUCTION block (a
    # teachable primitive — e.g. the prefix-min/suffix-max or monotonic-deque idiom), so the bar is a
    # FRACTION of ref, not a fixed token count (the decisive defenses are discriminator-absence +
    # hole-substance, which catch any paste that carries the answer-logic).
    ref_toks = _norm_tokens(ref)
    run = _longest_common_run(ttoks, ref_toks)
    no_ref_paste = bool(tutor.is_negative_control or run <= int(max_ref_run)
                        or (ref_toks and run < 0.75 * len(ref_toks)))

    # public-only literals — no numeric literal that appears ONLY in a secret case
    pub_lits: set[str] = set()
    for inp, exp in problem.samples:
        pub_lits |= _numeric_literals(inp) | _numeric_literals(exp)
    sec_lits: set[str] = set()
    for inp, exp in problem.secret_cases:
        sec_lits |= _numeric_literals(inp) | _numeric_literals(exp)
    structural = {"0", "1", "2", "3"}  # array offsets / tiny constants are not instance data
    leaked = sorted(t for t in _numeric_literals(text)
                    if t in sec_lits and t not in pub_lits and t not in structural)
    public_only = (len(leaked) == 0)

    # a procedure, not a closed-form answer: if a skeleton is present it must carry control flow (a
    # bare formula yielding the answer is rejected).  A MULTI-digit public OUTPUT appearing verbatim
    # as a standalone token is suspicious (an instance value in a family-level object); single-digit
    # outputs are NOT flagged (they collide with technique prose like "O(N^2)" / "step 1").  Per-
    # instance answer leakage is otherwise covered by public_only_literals + template_invariant.
    has_control = (not tutor.skeleton) or bool(re.search(r"\b(for|while|if)\b", tutor.skeleton))
    text_tokens = set(re.findall(r"\d+", text))
    pub_outputs = {exp.strip() for _, exp in problem.samples
                   if exp.strip().isdigit() and len(exp.strip()) >= 2}
    states_answer = any(o in text_tokens for o in pub_outputs)
    is_procedure = bool(has_control and not states_answer)

    # one-liner-family guard: a card is only permitted if the reference is non-trivial
    one_liner_ok = bool(tutor.is_negative_control or _ref_statement_count(ref) >= 3)

    # discriminator-shape-only: no concrete array literal (>=3 numbers in brackets) in the text
    disc_shape_only = not bool(re.search(r"\[\s*-?\d+\s*(,\s*-?\d+\s*){2,}\]", text))

    # template/instance invariance: the tutor shares no long verbatim run with THIS statement
    # and carries no secret-only literal (already checked); family-level by construction
    stmt_run = _longest_common_run(ttoks, _norm_tokens(problem.statement))
    template_inv = bool(tutor.is_negative_control or stmt_run <= int(max_ref_run))

    return TutorLeakReportV1(
        tutor_cid=tutor.cid(), family=tutor.family, tc_kind=tutor.tc_kind,
        no_discriminator_leak=no_disc_leak, leaked_discriminators=tuple(leaked_disc),
        holes_are_substantive=holes_ok, no_reference_paste=no_ref_paste, longest_ref_run=int(run),
        public_only_literals=public_only, leaked_literals=tuple(leaked),
        is_procedure_not_answer=is_procedure, one_liner_family_ok=one_liner_ok,
        discriminator_shape_only=disc_shape_only, template_invariant=template_inv)


def skeleton_is_completable_v1(template: ParserNeutralTemplateV2, problem: MintedProblemV1, *,
                               timeout_s: float = 8.0) -> dict[str, Any]:
    """$0 self-test ONLY: prove the holed scaffold is a VALID teaching object — the correct_fill
    completes it to a program that PASSES the hidden secret cases (so the holes are the whole gap
    between scaffold and solution, and the scaffold teaches a real technique)."""
    spec = _spec_for(template)
    if spec is None:
        return {"completable": None, "reason": "no technique spec"}
    filled = spec.fill(spec.correct_fill)
    ok = True
    for inp, exp in problem.secret_cases:
        r = _exec_capture_v1(filled, inp, timeout_s=float(timeout_s))
        if r.timed_out or r.returncode != 0 or r.stdout.strip() != exp.strip():
            ok = False
            break
    return {"completable": bool(ok), "n_holes": len(spec.holes()), "algo_sig": spec.algo_sig}


def tutor_is_genuinely_new_v1(tutor: FamilyTutorV1) -> dict[str, Any]:
    """Structural 'not prompt decoration' check: a REAL tutor carries a named technique AND at least
    one of {key_move, skeleton, bug_warnings}; the T6 negative control (content-free) is FAKE."""
    if tutor.is_negative_control:
        return {"genuinely_new": False, "is_negative_control": True,
                "carries_technique": False, "reason": "content-free do-better instruction"}
    carries = bool(tutor.technique_name) and bool(
        tutor.key_move or tutor.skeleton or tutor.bug_warnings)
    return {"genuinely_new": bool(carries), "is_negative_control": False,
            "carries_technique": carries,
            "reason": "named technique + actionable content" if carries else "empty"}


COMPILERS_BY_KIND = {
    TC1_CARD: compile_family_card_v1,
    TC2_REWRITE: compile_witness_rewrite_tutor_v1,
    TC3_COMPRESSED: compile_compressed_tutor_v1,
    TC5_STAGED: compile_staged_tutor_v1,
    T6_NEG: make_negative_control_tutor_v1,
}


__all__ = [
    "FAMILY_TUTOR_COMPILER_V1_SCHEMA_VERSION",
    "TC1_CARD", "TC2_REWRITE", "TC3_COMPRESSED", "TC5_STAGED", "T6_NEG",
    "OBS_TIMEOUT", "OBS_WRONG_ANSWER", "TC3_TOKEN_BUDGET",
    "TechniqueSpecV1", "TECHNIQUE_LIBRARY", "FamilyTutorV1",
    "compile_family_card_v1", "compile_witness_rewrite_tutor_v1", "compile_compressed_tutor_v1",
    "compile_staged_tutor_v1", "make_negative_control_tutor_v1", "COMPILERS_BY_KIND",
    "TutorLeakReportV1", "tutor_leak_gate_v1", "skeleton_is_completable_v1",
    "tutor_is_genuinely_new_v1",
]
