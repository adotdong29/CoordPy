"""W130 Lane α — generator-failure atlas (NIM-free diagnosis; explicit-import only).

W128 lifted the hard-cluster generation ceiling (role-diverse pool 3/11 > plain 2/11) but
RDA4 committed only 2/11; W129 attacked the SELECTOR directly and proved the binding cap is
GENERATION, not selection (committed <= pool ceiling = baseline+1 < the +2 earn bar, regardless
of selector quality).  W129 left the sharp question OPEN:

    WHY is the hard-cluster pool ceiling still too low, and WHICH generator-side failures are
    actually dominant?  How many pool-dead problems are GENERATOR-fixable under the same budget
    vs capability failures, and which are merely SELECTOR-fixable (already W129's domain)?

This module is the diagnosis.  It reconstructs the FULL old W128/W129 candidate pool
(``plain`` U ``scaffold`` U ``rda``) per hard-cluster dev target from the stored generations,
grades every candidate with a MECHANICAL failure signature (the repo's only execution path —
``_run_capture_stdout_v1`` mirrors ``grade_icpc_candidate_case_v1`` exactly), cross-checks an
OFFLINE-ONLY accepted-algorithm reference (NEVER model-facing — the W127/W128 atlas discipline),
and classifies each problem's dominant generator-failure mode under a taxonomy LOCKED in code
BEFORE any result is interpreted (the W129 $0-recon discipline,
``feedback_never_prewrite_results_before_data``).

NO model is called.  $0 NIM.  The accepted solution is used ONLY as an offline analyst signal
for the admissibility heuristic; it never enters any prompt and never grades anything (grading
is on the official public + secret cases).
"""
from __future__ import annotations

import dataclasses
import re
from typing import Optional, Sequence

from coordpy.icpc_reflexion_bench_v1 import IcpcPilotProblemV1, grade_on_secret_v1
from coordpy.role_diverse_algorithm_search_v1 import (
    SketchV1, _norm_out, _parses, _run_capture_stdout_v1, _sha)

# ---------------------------------------------------------------------------
# LOCKED taxonomy (pre-committed before any result is interpreted)
# ---------------------------------------------------------------------------
#: dominant generator-failure mode per problem.  Ordered most-actionable-first within the
#: pool-dead block.  A problem is GENERATION-bound iff it is pool-dead (no candidate in the
#: entire plain U scaffold U rda pool passes the official secret cases).
GENERATOR_FAILURE_MODES: tuple[str, ...] = (
    # pool-BEARING (>=1 candidate passes secret) -> NOT a generation failure
    "SOLVED",                      # baseline/RDA committed a correct candidate; no gap
    "SELECTION_FIXABLE",           # >=1 hidden-correct AND >=1 hidden-wrong public survivor:
                                   # the W129 selector layer, NOT generation
    # pool-DEAD (generation-bound), ordered by how "close" the generator got
    "HIDDEN_EDGE_STATE_MISS",      # a candidate passes ALL public but fails secret: a
                                   # near-correct algorithm with a state/invariant/edge bug
    "COMPLEXITY_BLIND",            # dominant non-pass failure is TLE: the algorithm is
                                   # asymptotically too slow for the stated bound
    "WRONG_ALGORITHM_ADMISSIBLE",  # 0 public survivors, but >=1 ANALYZE sketch matches the
                                   # accepted approach: the idea was proposed, the derivation
                                   # or implementation is wrong
    "WRONG_ALGORITHM_NO_SKETCH",   # 0 public survivors AND no sketch matches the accepted
                                   # approach: a capability failure (the hardest mode)
    "PARSE_IO_FAILURE",            # dominant failure is parse/crash: a trivial IO/format bug
)

#: modes a stronger SAME-BUDGET generator can plausibly attack (vs a pure capability failure).
GENERATOR_FIXABLE_MODES: frozenset[str] = frozenset({
    "HIDDEN_EDGE_STATE_MISS", "COMPLEXITY_BLIND", "WRONG_ALGORITHM_ADMISSIBLE",
    "PARSE_IO_FAILURE"})

#: per-candidate failure typing (mechanical; from the official execution path).
FAIL_TYPES: tuple[str, ...] = (
    "PASS", "HIDDEN_FAIL", "WRONG_ANSWER", "TLE", "CRASH", "PARSE_ERR")


# ---------------------------------------------------------------------------
# offline accepted-algorithm reference signature (NEVER model-facing)
# ---------------------------------------------------------------------------
# A curated competitive-programming idiom lexicon.  ``specific`` tags are structural enough to
# be discriminating; ``weak`` tags (sort / simulate) appear almost everywhere and are NOT used
# alone to call a sketch admissible.  This is a TRANSPARENT HEURISTIC (the W127 47%-concordant
# theme-classifier lesson), reported as such — never as ground truth.
_IDIOM_LEXICON: tuple[tuple[str, bool, tuple[str, ...]], ...] = (
    # (tag, specific?, regex-ish keyword patterns matched case-insensitively)
    ("graph_search", True, (r"\bbfs\b", r"\bdfs\b", r"\bdeque\b", r"adjac", r"neighbou?r",
                            r"\bvisited\b", r"\bgraph\b", r"popleft", r"\bedges?\b",
                            r"\bnodes?\b", r"union[- ]?find", r"disjoint set", r"\bdsu\b")),
    ("dynamic_programming", True, (r"\bdp\b", r"\bdp\[", r"\bmemo", r"lru_cache",
                                   r"dynamic program", r"subproblem", r"bottom[- ]?up",
                                   r"top[- ]?down")),
    ("binary_search", True, (r"binary search", r"\bbisect", r"\blo\b\s*[=,]", r"\bhi\b\s*[=,]",
                             r"\bmid\b\s*=", r"while\s+lo")),
    ("greedy_heap", True, (r"\bgreedy\b", r"heapq", r"heappush", r"heappop", r"priority queue",
                           r"\bpq\b")),
    ("two_pointer", True, (r"two[- ]?pointer", r"sliding window", r"\bleft\b.*\bright\b")),
    ("number_theory", True, (r"\bgcd\b", r"\bsieve\b", r"\bprime", r"factorial", r"\bcomb\b",
                             r"binomial", r"modular", r"\bmod\b", r"pow\(", r"\bncr\b",
                             r"inverse mod")),
    ("recursion_backtrack", True, (r"backtrack", r"recursi", r"\brecurse\b")),
    ("interval_sort", False, (r"\bsort", r"sorted\(", r"\bintervals?\b")),
    ("simulation", False, (r"\bsimulat", r"\bstep\b", r"\bgrid\b", r"\bmove\b", r"\bdx\b",
                           r"\bdy\b", r"direction")),
)


def algorithm_signature_v1(texts: Sequence[str]) -> dict:
    """Detect which idiom tags appear in a bag of texts (accepted code or sketch outlines).

    Returns ``{"specific": [...], "weak": [...], "all": [...]}``.  Offline-only.
    """
    blob = "\n".join(str(t) for t in texts).lower()
    specific, weak = [], []
    for tag, is_specific, pats in _IDIOM_LEXICON:
        if any(re.search(p, blob) for p in pats):
            (specific if is_specific else weak).append(tag)
    return {"specific": specific, "weak": weak, "all": specific + weak}


def sketch_admissible_v1(sketches: Sequence[SketchV1],
                         accepted_codes: Sequence[str]) -> dict:
    """Heuristic: did the ANALYZE stage propose an algorithm that MATCHES the accepted
    approach on >=1 SPECIFIC (discriminating) idiom tag?  Offline-only; never model-facing.

    A match on a SPECIFIC tag (graph_search / dp / binary_search / greedy_heap / two_pointer /
    number_theory / recursion_backtrack) is admissible.  A match only on WEAK tags
    (sort / simulate) is NOT enough (those appear in almost every program).
    """
    acc_sig = algorithm_signature_v1(list(accepted_codes))
    sk_text = [f"{s.approach_name} {s.outline}" for s in sketches]
    sk_sig = algorithm_signature_v1(sk_text)
    matched_specific = sorted(set(acc_sig["specific"]) & set(sk_sig["specific"]))
    return {
        "accepted_specific": acc_sig["specific"], "accepted_weak": acc_sig["weak"],
        "sketch_specific": sk_sig["specific"], "matched_specific": matched_specific,
        "admissible": bool(matched_specific),
    }


# ---------------------------------------------------------------------------
# mechanical per-candidate failure signature
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class CandidateSignatureV1:
    label: str
    arm: str                 # plain / scaffold / rda
    parses: bool
    public_pass: bool        # passes ALL public samples
    secret_pass: bool
    fail_type: str           # one of FAIL_TYPES
    code_sha: str

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _first_public_fail_type(problem: IcpcPilotProblemV1, code: str, *,
                            timeout_s: float) -> str:
    """Type the FIRST failing public sample mechanically with a SINGLE capture per sample.

    All hard-cluster dev targets are KIND_PASSFAIL (exact token compare), so a whitespace-
    collapsed ``_norm_out`` equality matches the official grader's intent while spending only ONE
    subprocess per sample (vs grade-then-recapture).  TLE/crash/parse are typed from the capture
    sentinel + digest.
    """
    for inp, exp in problem.samples:
        out, _dig = _run_capture_stdout_v1(code, inp, timeout_s=timeout_s)
        if out == "<TIMEOUT>":
            return "TLE"
        if out.startswith("<ERR"):
            return "CRASH"
        if out == "<PARSE_ERR>":
            return "PARSE_ERR"
        if _norm_out(out) != _norm_out(exp):
            return "WRONG_ANSWER"
    return "PASS"  # all public samples passed


def classify_candidate_v1(problem: IcpcPilotProblemV1, code: str, *, label: str, arm: str,
                          timeout_s: float = 3.0, secret_timeout_s: float = 6.0
                          ) -> CandidateSignatureV1:
    """Grade one candidate and assign a mechanical failure type."""
    parses = _parses(code)
    if not parses:
        return CandidateSignatureV1(label, arm, False, False, False, "PARSE_ERR", _sha(code)[:16])
    pub = _first_public_fail_type(problem, code, timeout_s=timeout_s)
    public_pass = (pub == "PASS")
    secret_pass = False
    if public_pass:
        try:
            ok, _tail, _n = grade_on_secret_v1(problem, code, timeout_s=secret_timeout_s)
            secret_pass = bool(ok)
        except Exception:  # noqa: BLE001
            secret_pass = False
    if public_pass:
        fail_type = "PASS" if secret_pass else "HIDDEN_FAIL"
    else:
        fail_type = pub
    return CandidateSignatureV1(label, arm, True, public_pass, secret_pass, fail_type,
                                _sha(code)[:16])


# ---------------------------------------------------------------------------
# per-problem record + dominant-mode classifier
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class ProblemFailureRecordV1:
    short_name: str
    family: str
    surface: str
    date: str
    n_candidates: int
    n_parse: int
    n_public_survivors: int
    n_pub_correct: int          # public-survivor AND secret-pass
    n_pub_wrong: int            # public-survivor AND secret-FAIL (hidden fail)
    counts: dict                # {fail_type: n} over all candidates
    pool_bearing: bool
    accepted_specific: list
    sketch_specific: list
    matched_specific: list
    sketch_admissible: bool
    dominant_mode: str
    selector_fixable: bool
    generator_fixable: bool
    committed_w128: Optional[bool]
    note: str

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def classify_problem_failure_v1(*, pool_bearing: bool, n_pub_correct: int, n_pub_wrong: int,
                                counts: dict, sketch_admissible: bool) -> tuple[str, str]:
    """Deterministic dominant-mode classifier.  Returns (mode, note).  LOCKED logic."""
    n_hidden = int(counts.get("HIDDEN_FAIL", 0))
    n_tle = int(counts.get("TLE", 0))
    n_crash = int(counts.get("CRASH", 0))
    n_parse_err = int(counts.get("PARSE_ERR", 0))
    n_wrong = int(counts.get("WRONG_ANSWER", 0))

    if pool_bearing:
        if n_pub_correct >= 1 and n_pub_wrong >= 1:
            return "SELECTION_FIXABLE", (
                f"pool-bearing; {n_pub_correct} hidden-correct + {n_pub_wrong} hidden-wrong "
                "public survivor(s) -> a selection tie (W129 domain), not generation")
        return "SOLVED", "pool-bearing; the committed candidate is correct (no generation gap)"

    # pool-DEAD: generation-bound.  Order by how close the generator got.
    if n_hidden >= 1:
        return "HIDDEN_EDGE_STATE_MISS", (
            f"{n_hidden} candidate(s) pass ALL public but fail secret -> near-correct "
            "algorithm with a state/invariant/edge bug (generator-fixable)")
    # dominant non-pass failure type among parseable wrong candidates
    if n_tle >= 1 and n_tle >= max(n_crash, n_parse_err, n_wrong):
        return "COMPLEXITY_BLIND", (
            f"TLE dominates ({n_tle} of pool) -> algorithm asymptotically too slow "
            "(generator-fixable via complexity-gated sketch)")
    if (n_parse_err + n_crash) > n_wrong and (n_parse_err + n_crash) >= 1:
        return "PARSE_IO_FAILURE", (
            f"parse/crash dominates ({n_parse_err + n_crash}) -> trivial IO/format bug")
    # wrong-answer dominant: split on whether the idea was even proposed
    if sketch_admissible:
        return "WRONG_ALGORITHM_ADMISSIBLE", (
            "0 public survivors; >=1 sketch matches the accepted approach -> idea proposed, "
            "derivation/impl wrong (generator-fixable)")
    return "WRONG_ALGORITHM_NO_SKETCH", (
        "0 public survivors; no sketch matches the accepted approach -> capability failure "
        "(the hardest mode; a same-budget generator is unlikely to crack it)")


def build_problem_record_v1(*, short_name: str, family: str, surface: str, date: str,
                            candidates: Sequence[CandidateSignatureV1],
                            sketches: Sequence[SketchV1], accepted_codes: Sequence[str],
                            committed_w128: Optional[bool] = None) -> ProblemFailureRecordV1:
    """Aggregate per-candidate signatures + the offline admissibility check into one record."""
    counts: dict = {t: 0 for t in FAIL_TYPES}
    for c in candidates:
        counts[c.fail_type] = counts.get(c.fail_type, 0) + 1
    n_parse = sum(1 for c in candidates if c.parses)
    pub_surv = [c for c in candidates if c.public_pass]
    n_pub_correct = sum(1 for c in pub_surv if c.secret_pass)
    n_pub_wrong = sum(1 for c in pub_surv if not c.secret_pass)
    pool_bearing = any(c.secret_pass for c in candidates)
    adm = sketch_admissible_v1(sketches, accepted_codes)
    mode, note = classify_problem_failure_v1(
        pool_bearing=pool_bearing, n_pub_correct=n_pub_correct, n_pub_wrong=n_pub_wrong,
        counts=counts, sketch_admissible=adm["admissible"])
    return ProblemFailureRecordV1(
        short_name=short_name, family=family, surface=surface, date=date,
        n_candidates=len(candidates), n_parse=n_parse, n_public_survivors=len(pub_surv),
        n_pub_correct=n_pub_correct, n_pub_wrong=n_pub_wrong, counts=counts,
        pool_bearing=pool_bearing, accepted_specific=adm["accepted_specific"],
        sketch_specific=adm["sketch_specific"], matched_specific=adm["matched_specific"],
        sketch_admissible=adm["admissible"], dominant_mode=mode,
        selector_fixable=(mode == "SELECTION_FIXABLE"),
        generator_fixable=(mode in GENERATOR_FIXABLE_MODES),
        committed_w128=committed_w128, note=note)


# ---------------------------------------------------------------------------
# atlas-level aggregation + the W130 generator-attackability summary
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class GeneratorFailureAtlasV1:
    n_problems: int
    pool_bearing: list          # short_names
    pool_dead: list             # short_names
    mode_histogram: dict        # {mode: count}
    selector_fixable: list      # short_names (W129 domain)
    generator_fixable: list     # pool-dead AND generator-fixable
    capability_failures: list   # WRONG_ALGORITHM_NO_SKETCH
    top_generator_modes: list   # [(mode, count)] over pool-dead, most-common first
    records: list               # [ProblemFailureRecordV1.to_dict()]

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        return d


def build_atlas_v1(records: Sequence[ProblemFailureRecordV1]) -> GeneratorFailureAtlasV1:
    pool_bearing = [r.short_name for r in records if r.pool_bearing]
    pool_dead = [r.short_name for r in records if not r.pool_bearing]
    hist: dict = {}
    for r in records:
        hist[r.dominant_mode] = hist.get(r.dominant_mode, 0) + 1
    sel_fix = [r.short_name for r in records if r.selector_fixable]
    gen_fix = [r.short_name for r in records
               if (not r.pool_bearing) and r.generator_fixable]
    cap = [r.short_name for r in records if r.dominant_mode == "WRONG_ALGORITHM_NO_SKETCH"]
    dead_modes: dict = {}
    for r in records:
        if not r.pool_bearing:
            dead_modes[r.dominant_mode] = dead_modes.get(r.dominant_mode, 0) + 1
    top = sorted(dead_modes.items(), key=lambda kv: (-kv[1], kv[0]))
    return GeneratorFailureAtlasV1(
        n_problems=len(records), pool_bearing=pool_bearing, pool_dead=pool_dead,
        mode_histogram=hist, selector_fixable=sel_fix, generator_fixable=gen_fix,
        capability_failures=cap, top_generator_modes=top,
        records=[r.to_dict() for r in records])
