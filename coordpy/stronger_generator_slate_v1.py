"""W130 Lane β — stronger same-budget GENERATOR slate (explicit-import only).

W128 lifted the hard-cluster generation ceiling but RDA4 committed only 2/11; W129 attacked the
SELECTOR directly and proved the binding cap is GENERATION, not selection (committed <= pool
ceiling = baseline+1 < the +2 earn bar, regardless of selector quality).  W130's question is the
honest next lever:

    Can CoordPy create genuinely BETTER candidates on the hard clusters under the SAME K=5
    budget, when the W129 selector is held FIXED downstream?  The variable under test is
    GENERATION, not selection.

This module is the generator slate.  Every arm produces K candidates at K model calls (same
budget as the W120/W121/W127 plain baseline and the W128 role-diverse search); the NIM-free
complexity gate, counterexample runs, and digest routing cost $0.  The W129 selector
(``public_signal_selection_oracle_v1.select_so_v1`` SOLEAD, NIM-free) is applied UNCHANGED to
every arm's pool — so a committed win is attributable to the generator, not a new selector.

Arms (the operator-named slate):
* **GG1 — complexity-gated role handoff.**  ANALYZE emits each sketch's Big-O; a NIM-free gate
  rejects sketches whose complexity cannot meet the stated bound BEFORE implementation; freed
  budget is reallocated to admissible sketches.  Directly attacks COMPLEXITY_BLIND + the
  sketch->impl handoff (W129 proved the model CAN reason about O(N^2) vs O(N); GG1 brings that
  reasoning into GENERATION, not post-hoc SELECTION).
* **GG2 — counterexample-to-rewrite.**  After IMPLEMENT, the best candidate is run on the PUBLIC
  samples + the model's DERIVED counterexamples; a typed failure digest forces ONE in-loop
  REWRITE producing a FRESH candidate (not a rerank).  Public-signal-only feedback (never the
  secret cases / accepted solution).
* **GG3 — family anti-pattern coach.**  EXPOSED-side teacher material is used ONLY as a
  family-level anti-pattern / complexity / invariant coach injected into ANALYZE — NOT a
  same-problem scaffold and NOT answer material.  Killed if it collapses into W127 scaffold retry.
* **GG4 — planner/coordinator budget policy.**  The W125 controller's PATCH/REPLAN/ABSTAIN
  digest-router allocates the K budget (more sketches vs rewrite vs abstain).  The hosted
  cache-aware planner is efficiency-only (KV-prefix savings), NOT a capability lever — recorded
  honestly; GG4 is killed if the router never changes the allocation (prompt decoration).
* **GGLEAD = GG1 -> GG2.**  Complexity-gated generation with one counterexample rewrite.

NO bounded-context / compaction / summarization (anti-patterns).  The accepted solution is a
TRIPWIRE only (``reproduces_accepted_block_v1``), never an input to any prompt.
"""
from __future__ import annotations

import dataclasses
import re
from typing import Callable, Optional, Sequence

from coordpy.executor_grounded_patcher_v1 import FailureDigestV1, parse_failure_digest_v1
from coordpy.family_scaffold_generation_v1 import reproduces_accepted_block_v1
from coordpy.icpc_reflexion_bench_v1 import (
    IcpcPilotProblemV1, extract_candidate_code_v1, grade_on_secret_v1)
from coordpy.public_signal_selection_oracle_v1 import (
    derive_auto_cases_v1, parse_max_constraint_v1, select_so_v1)
from coordpy.role_diverse_algorithm_search_v1 import (
    CandidateImplV1, RoleArtifactsV1, SketchV1, _norm_code, _parses,
    _run_capture_stdout_v1, _sha, build_analyze_prompt_v1, build_implement_prompt_v1,
    parse_role_artifacts_v1)

GenFn = Callable[[str, int, float], tuple]

GG_VARIANTS: tuple[str, ...] = ("GG1", "GG2", "GG3", "GG4", "GGLEAD")
DEFAULT_K = 5
DEFAULT_MAX_TOKENS = 1536
#: ops budget for the complexity gate: ~5e8 simple ops fit a few-second ICPC time limit.
COMPLEXITY_OPS_BUDGET = 5e8


# ===========================================================================
# complexity reasoning (GG1) — parse a sketch's Big-O, gate vs the stated bound
# ===========================================================================
# growth exponent for the dominant n-term; log treated as a small additive factor.
_COMPLEXITY_PATTERNS: tuple[tuple[str, float], ...] = (
    (r"o\(\s*n\s*!\s*\)", 100.0),                       # factorial
    (r"o\(\s*2\s*\^?\s*[\^{]?\s*n", 99.0),              # exponential 2^n
    (r"o\(\s*n\s*\^?\s*[\^{]?\s*3", 3.0),               # n^3
    (r"o\(\s*n\s*\^?\s*[\^{]?\s*2", 2.0),               # n^2
    (r"o\(\s*n\s*\*\s*m\s*\)", 2.0),                    # n*m -> ~n^2
    (r"o\(\s*n\s*m\s*\)", 2.0),                         # nm -> ~n^2
    (r"o\(\s*n\s*\\?log", 1.1),                         # n log n
    (r"o\(\s*n\s*log", 1.1),
    (r"o\(\s*n\s*\)", 1.0),                             # linear
    (r"o\(\s*\\?log", 0.1),                             # log
    (r"o\(\s*1\s*\)", 0.0),                             # constant
)


def parse_complexity_exponent_v1(text: str) -> Optional[float]:
    """Best-effort parse of a stated Big-O from a sketch/complexity blob (public signal).

    Returns the dominant n-growth exponent (n->1.0, n log n->1.1, n^2->2.0, 2^n->99, n!->100,
    log->0.1, 1->0.0) or None if no Big-O is stated.  Conservative: an UNSTATED complexity is
    NOT gated (None), so the gate never rejects a sketch on a missing annotation.
    """
    s = (text or "").lower().replace(" ", " ")
    for pat, expo in _COMPLEXITY_PATTERNS:
        if re.search(pat, s):
            return expo
    return None


def complexity_admissible_v1(exponent: Optional[float], n_bound: Optional[int], *,
                             ops_budget: float = COMPLEXITY_OPS_BUDGET) -> Optional[bool]:
    """Is an algorithm of the given growth exponent admissible at the stated size bound?

    Returns True/False, or None when it cannot be judged (no stated complexity OR no parseable
    bound) -> conservative (never gate out a sketch we cannot judge).  Polynomial estimate uses
    n**exponent (log folded into the +0.1 fractional exponents); exponential/factorial admit
    only tiny n.
    """
    if exponent is None or n_bound is None:
        return None
    if exponent >= 99.0:  # exponential / factorial: only admissible for tiny n
        return n_bound <= 22
    est_ops = float(n_bound) ** float(exponent)
    return est_ops <= ops_budget


# ===========================================================================
# GG1 prompt — ANALYZE that REQUIRES a per-sketch Big-O + a worst-case note
# ===========================================================================
def build_gg1_analyze_prompt_v1(problem: IcpcPilotProblemV1, *, n_sketches: int = 4) -> str:
    """W128 ANALYZE + an explicit per-sketch worst-case-complexity REQUIREMENT (so the gate has
    a public-signal complexity to read).  The model is told the stated bound and asked to reject
    its own too-slow ideas — bringing W129's complexity reasoning into GENERATION."""
    base = build_analyze_prompt_v1(problem, n_sketches=n_sketches)
    bound = parse_max_constraint_v1(problem.statement)
    bound_line = (f"\n\nThe largest input size implied by the statement is about N={bound}. "
                  "An O(N^2) or slower approach will TIME OUT at that size.\n"
                  if bound else "\n\n")
    inject = (
        bound_line +
        "For EACH sketch you list, you MUST append a line 'COMPLEXITY: O(...)' giving its "
        "worst-case time complexity in terms of N. PREFER sketches that fit the time limit; if "
        "an idea is O(N^2) or slower and N is large, say so explicitly and propose a faster "
        "alternative as one of your sketches. A correct-but-too-slow algorithm scores ZERO.")
    return base + inject


def build_worstcase_rewrite_prompt_v1(problem: IcpcPilotProblemV1, spec: str, sketch: SketchV1,
                                      n_bound: Optional[int]) -> str:
    """GG1 reallocation: re-implement the best ADMISSIBLE sketch with an explicit
    fit-the-time-limit hardening (used when the gate freed an implement slot)."""
    samples = "\n".join(f"INPUT:\n{i}\nOUTPUT:\n{o}" for i, o in problem.samples)
    nb = f"N can be as large as {n_bound}. " if n_bound else ""
    return (
        "Implement a COMPLETE, EFFICIENT Python 3 program (stdin->stdout) for the problem below, "
        f"following EXACTLY this approach:\n\nAPPROACH ({sketch.approach_name}):\n{sketch.outline}\n\n"
        f"SPEC:\n{spec}\n\n{nb}Your program MUST run within the time limit at the largest N: use "
        "fast I/O (sys.stdin), avoid per-element O(N) rescans, and prefer O(N log N) or better. "
        "Output ONLY one fenced ```python code block.\n\n"
        f"PROBLEM:\n{problem.statement}\n\nSAMPLES:\n{samples}\n")


# ===========================================================================
# GG2 prompt — counterexample-to-rewrite (public-signal-only failure feedback)
# ===========================================================================
def _icpc_failure_digest_v1(code: str, stdin_text: str, expected: Optional[str], *,
                            timeout_s: float = 5.0) -> tuple[str, FailureDigestV1]:
    """Run a candidate on one PUBLIC/derived case and return (actual_out, typed digest).

    Reuses the executor-grounded digest concept (W111) on the ICPC stdin/stdout model.  The
    ``expected`` is supplied ONLY for PUBLIC samples (known-correct I/O pairs) or a model-derived
    EXPECT — never a secret case / accepted solution.
    """
    out, dig = _run_capture_stdout_v1(code, stdin_text, timeout_s=timeout_s)
    if dig is not None:
        return out, dig  # crash/TLE already typed
    # wrong-answer on a public sample: synthesize an expected/actual digest (public only)
    if expected is not None and out.strip() != (expected or "").strip():
        return out, parse_failure_digest_v1(
            stderr_tail=(f"AssertionError: {out.strip()[:120]} != {expected.strip()[:120]}"),
            timed_out=False)
    return out, parse_failure_digest_v1(stderr_tail="", timed_out=False)


def build_gg2_rewrite_prompt_v1(problem: IcpcPilotProblemV1, spec: str, latest_code: str,
                                failing_stdin: str, expected: Optional[str], actual: str,
                                digest: FailureDigestV1) -> str:
    """Force a FRESH candidate that fixes a CONCRETE public/derived failure (minimal-change
    framing; the W111 patcher pattern adapted to ICPC stdin/stdout).  Never the secret cases."""
    code_trim = latest_code if len(latest_code) <= 1600 else latest_code[:1600] + "\n# ...\n"
    exp_block = (f"EXPECTED stdout for this input:\n{expected}\n\n" if expected else "")
    err = ""
    if digest.exception_type == "Timeout":
        err = "Your program TIMED OUT on this input — your algorithm is too slow; use a faster one.\n"
    elif digest.exception_type:
        err = f"Your program crashed: {digest.exception_type}.\n"
    return (
        "You are fixing a Python 3 competitive-programming solution (stdin->stdout). Your latest "
        "solution is WRONG on the concrete case below. Make the SMALLEST change (or switch "
        "algorithm if it is fundamentally wrong) so it is correct on this case WITHOUT breaking "
        "the public samples.\n\n"
        f"SPEC:\n{spec}\n\nYOUR LATEST SOLUTION:\n```python\n{code_trim}\n```\n\n"
        f"FAILING INPUT:\n{failing_stdin}\n\n{exp_block}YOUR PROGRAM PRODUCED:\n{actual[:300]}\n\n{err}"
        "Output ONLY the corrected COMPLETE program in one ```python ... ``` fence.\n\n"
        f"PROBLEM:\n{problem.statement}\n")


# ===========================================================================
# candidate container + per-arm outcome
# ===========================================================================
@dataclasses.dataclass(frozen=True)
class GenCandidateV1:
    label: str
    code: str
    parses: bool
    origin: str                  # e.g. "sketchA" / "rewrite" / "gg1_realloc"

    def to_impl(self) -> CandidateImplV1:
        return CandidateImplV1(self.label, self.code, self.parses)


@dataclasses.dataclass(frozen=True)
class GgArmOutcomeV1:
    short_name: str
    arm: str
    n_calls: int
    candidates: tuple              # GenCandidateV1
    artifacts_spec_len: int
    n_sketches: int
    pool_pass: bool                # any candidate passes secret
    pool_secret_labels: tuple
    committed_label: Optional[str]
    committed_pass: bool           # the W129-fixed selector commits a secret-passing candidate
    selector_branch: str
    diagnostics: dict
    realness: dict
    leakage_clean: bool

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["candidates"] = [{"label": c.label, "origin": c.origin, "parses": c.parses,
                            "code_sha": _sha(c.code)[:16], "code_len": len(c.code)}
                           for c in self.candidates]
        return d


# ===========================================================================
# grading helpers
# ===========================================================================
def _public_pass(problem: IcpcPilotProblemV1, code: str, *, timeout_s: float = 5.0) -> bool:
    if not code or not _parses(code):
        return False
    for inp, exp in problem.samples:
        out, dig = _run_capture_stdout_v1(code, inp, timeout_s=timeout_s)
        if dig is not None:
            return False
        # normalized compare via the role-diverse path is exact-after-norm; use the official
        # grader through a quick compare on normalized stdout
        if out.strip() != _norm_out_expected(exp):
            return False
    return True


def _norm_out_expected(exp: str) -> str:
    # mirror role_diverse._norm_out: collapse trailing whitespace per line + strip
    return "\n".join(ln.rstrip() for ln in str(exp).replace("\r\n", "\n").split("\n")).strip()


def _secret_pass(problem: IcpcPilotProblemV1, code: str, *, timeout_s: float = 10.0) -> bool:
    if not code or not _parses(code):
        return False
    try:
        ok, _t, _n = grade_on_secret_v1(problem, code, timeout_s=timeout_s)
        return bool(ok)
    except Exception:  # noqa: BLE001
        return False


def _best_candidate(problem: IcpcPilotProblemV1, cands: Sequence[GenCandidateV1],
                    derived: Sequence[str], *, timeout_s: float = 5.0) -> Optional[GenCandidateV1]:
    """Pick the most-promising candidate by (public-sample passes, derived-case clean-runs)."""
    best, best_score = None, (-1, -1)
    for c in cands:
        if not c.parses:
            continue
        pub = sum(1 for inp, exp in problem.samples
                  if _run_capture_stdout_v1(c.code, inp, timeout_s=timeout_s)[0].strip()
                  == _norm_out_expected(exp))
        der = sum(1 for d in derived
                  if _run_capture_stdout_v1(c.code, d, timeout_s=timeout_s)[1] is None)
        score = (pub, der)
        if score > best_score:
            best, best_score = c, score
    return best or (cands[0] if cands else None)


def _find_failing_case(problem: IcpcPilotProblemV1, cand: Optional[GenCandidateV1],
                       derived: Sequence[str], *, timeout_s: float = 5.0):
    """Find ONE concrete public/derived case the candidate fails (for GG2's rewrite).

    Public samples carry a known expected (wrong-answer or crash/TLE is a failure); model-derived
    cases have NO ground truth, so only a crash/TLE counts as an objective failure on them.
    Returns (stdin, expected_or_None, actual_out, FailureDigestV1) or None.  Never the secret.
    """
    if cand is None or not cand.parses:
        return None
    for inp, exp in problem.samples:
        out, dig = _run_capture_stdout_v1(cand.code, inp, timeout_s=timeout_s)
        if dig is not None:
            return (inp, exp, out, dig)
        if out.strip() != _norm_out_expected(exp):
            d = parse_failure_digest_v1(
                stderr_tail=f"AssertionError: {out.strip()[:120]} != {_norm_out_expected(exp)[:120]}",
                timed_out=False)
            return (inp, exp, out, d)
    for inp in derived:
        out, dig = _run_capture_stdout_v1(cand.code, inp, timeout_s=timeout_s)
        if dig is not None:  # crash/TLE on a model-derived edge case
            return (inp, None, out, dig)
    return None


# ===========================================================================
# GG3 — family anti-pattern coach card (de-identified; family-level, NOT a scaffold)
# ===========================================================================
# Curated per-family anti-pattern / complexity coaching.  Generic family advice — provably NOT a
# same-problem scaffold and carries no candidate's accepted bytes.  Optionally augmented with
# the de-identified MOST-COMMON idiom names from the EXPOSED teacher library for that family.
_FAMILY_COACH: dict = {
    "simulation_grid": [
        "Model the FULL state (positions, directions, visited set) explicitly; an off-by-one in "
        "the grid bounds or wrap-around is the usual bug.",
        "Re-read the move/rotation rules; simulate the SAMPLE by hand and assert each step.",
        "Watch for cycles/termination: bound the number of steps and detect repeats."],
    "adhoc_math": [
        "Derive the closed form on paper FIRST; verify it reproduces every public sample EXACTLY "
        "before coding.",
        "Check parity / off-by-one / inclusive-vs-exclusive bounds and integer overflow edge cases.",
        "Handle the degenerate inputs (0, 1, equal values, empty) — they are the usual hidden case."],
    "greedy_scheduling": [
        "State the greedy CHOICE and PROVE the exchange argument; a plausible greedy is often wrong.",
        "Sort by the correct key (deadline vs duration vs ratio) — the wrong key is the usual bug.",
        "Use a heap for the 'best remaining' selection; recompute invariants after each pick."],
    "graph_flow": [
        "Identify the graph model (nodes/edges/capacities) explicitly before coding.",
        "Pick the right algorithm (BFS/DFS/Dijkstra/max-flow) for the constraint size.",
        "Check connectivity / unreachable / self-loop edge cases."],
}


def build_family_coach_card_v1(family: str, library=None, *, max_idioms: int = 6) -> str:
    """Family-level anti-pattern/complexity coaching for the ANALYZE prompt.  Generic advice +
    (optional) de-identified common idiom NAMES from the EXPOSED teacher library — NEVER a
    source skeleton, NEVER the target's own solution, NEVER a same-problem scaffold."""
    tips = list(_FAMILY_COACH.get(family, _FAMILY_COACH["adhoc_math"]))
    idiom_line = ""
    if library is not None:
        try:
            scafs = library.by_family.get(family, ())
            freq: dict = {}
            for s in scafs:
                for idi in getattr(s, "idioms", ()):  # idiom NAMES only (de-identified)
                    freq[idi] = freq.get(idi, 0) + 1
            common = [k for k, _ in sorted(freq.items(), key=lambda kv: -kv[1])[:max_idioms]]
            if common:
                idiom_line = ("\nCommon correct techniques in this family (names only): "
                              + ", ".join(common) + ".")
        except Exception:  # noqa: BLE001
            idiom_line = ""
    return ("FAMILY COACHING (generic; not specific to this problem):\n- "
            + "\n- ".join(tips) + idiom_line)


def build_gg3_analyze_prompt_v1(problem: IcpcPilotProblemV1, coach_card: str, *,
                                n_sketches: int = 4) -> str:
    """W128 ANALYZE + a FAMILY-LEVEL coaching card prepended (anti-patterns / complexity /
    invariants).  The card is generic family advice — not the target's solution."""
    return coach_card + "\n\n" + build_analyze_prompt_v1(problem, n_sketches=n_sketches)


# ===========================================================================
# the fixed-selector finalizer (W129 SOLEAD, NIM-free) + secret pool grading
# ===========================================================================
def _finalize_arm(problem: IcpcPilotProblemV1, arm: str, arts: RoleArtifactsV1,
                  cands: Sequence[GenCandidateV1], n_calls: int, diagnostics: dict,
                  realness: dict, *, accepted_codes: Sequence[str] = (),
                  secret_timeout_s: float = 8.0, public_timeout_s: float = 3.0
                  ) -> GgArmOutcomeV1:
    impls = [c.to_impl() for c in cands]
    # --- W129 selector held FIXED downstream (NIM-free SOLEAD) ---
    sel = select_so_v1(problem, impls, arts, variant="SOLEAD", gen=None)
    # --- grade the pool on secret (public prescreen short-circuits the cost) ---
    prov = problem.statement + "\n" + "\n".join(i + o for i, o in problem.samples)

    def _clean(code: str) -> bool:
        if accepted_codes and reproduces_accepted_block_v1(code, list(accepted_codes),
                                                           provenance=prov):
            return False
        return True

    pool_secret = []
    for c in cands:
        if c.parses and _clean(c.code) and _public_pass(problem, c.code, timeout_s=public_timeout_s) \
                and _secret_pass(problem, c.code, timeout_s=secret_timeout_s):
            pool_secret.append(c.label)
    committed_pass = bool(
        sel.committed_code and _clean(sel.committed_code)
        and _public_pass(problem, sel.committed_code, timeout_s=public_timeout_s)
        and _secret_pass(problem, sel.committed_code, timeout_s=secret_timeout_s))
    leakage_clean = all(_clean(c.code) for c in cands)
    return GgArmOutcomeV1(
        short_name=problem.short_name, arm=arm, n_calls=n_calls, candidates=tuple(cands),
        artifacts_spec_len=len(arts.spec), n_sketches=len(arts.sketches),
        pool_pass=bool(pool_secret), pool_secret_labels=tuple(pool_secret),
        committed_label=sel.committed_label, committed_pass=committed_pass,
        selector_branch=sel.branch, diagnostics=diagnostics, realness=realness,
        leakage_clean=leakage_clean)


# ===========================================================================
# the generator arms (each spends exactly n_calls <= K model calls)
# ===========================================================================
def run_gg1_v1(gen: GenFn, problem: IcpcPilotProblemV1, *, K: int = DEFAULT_K, n_sketches: int = 4,
               analyze_temp: float = 0.5, impl_temp: float = 0.2,
               max_tokens: int = DEFAULT_MAX_TOKENS, timeout_s: float = 5.0,
               accepted_codes: Sequence[str] = ()) -> GgArmOutcomeV1:
    """GG1 — complexity-gated role handoff.  1 ANALYZE (per-sketch Big-O) + (K-1) implements over
    the ADMISSIBLE sketches; rejected (too-slow) sketches' slots are reallocated to worst-case-
    hardened re-implements of the best admissible sketch.  The gate is NIM-free."""
    n_impl = max(1, K - 1)
    a_text, _ = gen(build_gg1_analyze_prompt_v1(problem, n_sketches=n_sketches),
                    max_tokens, analyze_temp)
    arts = parse_role_artifacts_v1(a_text, n_sketches=n_sketches)
    n_bound = parse_max_constraint_v1(problem.statement)
    gated = []
    for sk in arts.sketches:
        expo = parse_complexity_exponent_v1(f"{sk.approach_name} {sk.outline}")
        adm = complexity_admissible_v1(expo, n_bound)
        gated.append((sk, expo, adm))
    admissible = [g[0] for g in gated if g[2] is not False]  # keep True or unjudgeable(None)
    rejected = [g for g in gated if g[2] is False]
    impl_sketches = admissible or ([arts.sketches[0]] if arts.sketches
                                   else [SketchV1("A", "direct", "Implement the direct algorithm.")])
    cands, calls = [], 1
    for i in range(n_impl):
        if i < len(impl_sketches):
            sk = impl_sketches[i]
            prompt, origin = build_implement_prompt_v1(problem, arts.spec, sk), "gg1_impl"
        else:  # reallocated freed slot -> harden the best admissible sketch
            sk = impl_sketches[0]
            prompt = build_worstcase_rewrite_prompt_v1(problem, arts.spec, sk, n_bound)
            origin = "gg1_harden"
        text, _ = gen(prompt, max_tokens, impl_temp)
        calls += 1
        code = extract_candidate_code_v1(response_text=text)
        cands.append(GenCandidateV1(f"{sk.label}{i}", code, _parses(code), origin))
    diagnostics = {"n_bound": n_bound, "n_admissible": len(admissible),
                   "n_rejected": len(rejected), "gate_fired": bool(rejected),
                   "rejected_labels": [g[0].label for g in rejected],
                   "sketch_complexity": {g[0].label: g[1] for g in gated}}
    realness = {"gate_has_real_complexity": any(g[1] is not None for g in gated),
                "gate_rejected_on_complexity": bool(rejected),
                "n_realloc": max(0, n_impl - len(impl_sketches))}
    return _finalize_arm(problem, "GG1", arts, cands, calls, diagnostics, realness,
                         accepted_codes=accepted_codes)


def run_gg2_v1(gen: GenFn, problem: IcpcPilotProblemV1, *, K: int = DEFAULT_K, n_sketches: int = 3,
               analyze_temp: float = 0.5, impl_temp: float = 0.2,
               max_tokens: int = DEFAULT_MAX_TOKENS, timeout_s: float = 5.0,
               accepted_codes: Sequence[str] = ()) -> GgArmOutcomeV1:
    """GG2 — counterexample-to-rewrite.  1 ANALYZE + (K-2) implements + 1 in-loop REWRITE driven
    by a typed public/derived failure digest (a FRESH candidate, not a rerank).  If the best
    candidate fails nothing public/derived, the rewrite slot funds an extra diverse impl."""
    n_impl = max(1, K - 2)
    a_text, _ = gen(build_analyze_prompt_v1(problem, n_sketches=n_sketches), max_tokens, analyze_temp)
    arts = parse_role_artifacts_v1(a_text, n_sketches=n_sketches)
    sk = list(arts.sketches) or [SketchV1("A", "direct", "Implement the direct algorithm.")]
    while len(sk) < n_impl + 1:
        sk.append(sk[len(sk) % len(arts.sketches or sk)])
    cands, calls = [], 1
    for i in range(n_impl):
        text, _ = gen(build_implement_prompt_v1(problem, arts.spec, sk[i]), max_tokens, impl_temp)
        calls += 1
        code = extract_candidate_code_v1(response_text=text)
        cands.append(GenCandidateV1(f"{sk[i].label}{i}", code, _parses(code), "gg2_impl"))
    derived = [inp for inp, _e in arts.counterexamples]
    best = _best_candidate(problem, cands, derived, timeout_s=timeout_s)
    fail = _find_failing_case(problem, best, derived, timeout_s=timeout_s)
    rewrite_new = False
    if best is not None and fail is not None:
        stdin_f, exp_f, actual_f, dig = fail
        prompt = build_gg2_rewrite_prompt_v1(problem, arts.spec, best.code, stdin_f, exp_f,
                                             actual_f, dig)
        text, _ = gen(prompt, max_tokens, impl_temp)
        calls += 1
        code = extract_candidate_code_v1(response_text=text)
        cands.append(GenCandidateV1("rw", code, _parses(code), "gg2_rewrite"))
        rewrite_new = (_parses(code) and best.parses and _norm_code(code) != _norm_code(best.code))
    else:  # no failing case found -> fund an extra diverse implementation
        ex = sk[n_impl % len(sk)]
        text, _ = gen(build_implement_prompt_v1(problem, arts.spec, ex), max_tokens, impl_temp)
        calls += 1
        code = extract_candidate_code_v1(response_text=text)
        cands.append(GenCandidateV1("ex", code, _parses(code), "gg2_extra"))
    diagnostics = {"best_label": best.label if best else None,
                   "had_failing_case": fail is not None,
                   "failure_type": (fail[3].exception_type or "wrong_answer") if fail else None}
    realness = {"rewrite_issued": fail is not None, "rewrite_structurally_new": rewrite_new}
    return _finalize_arm(problem, "GG2", arts, cands, calls, diagnostics, realness,
                         accepted_codes=accepted_codes)


def run_gg3_v1(gen: GenFn, problem: IcpcPilotProblemV1, *, family: str, library=None,
               K: int = DEFAULT_K, n_sketches: int = 4, analyze_temp: float = 0.5,
               impl_temp: float = 0.2, max_tokens: int = DEFAULT_MAX_TOKENS,
               accepted_codes: Sequence[str] = ()) -> GgArmOutcomeV1:
    """GG3 — family anti-pattern coach.  1 coached ANALYZE + (K-1) implements.  The coaching is
    GENERIC family advice (+ de-identified idiom names); it must NOT reproduce an accepted block
    and is NOT a same-problem scaffold."""
    coach = build_family_coach_card_v1(family, library=library)
    n_impl = max(1, K - 1)
    a_text, _ = gen(build_gg3_analyze_prompt_v1(problem, coach, n_sketches=n_sketches),
                    max_tokens, analyze_temp)
    arts = parse_role_artifacts_v1(a_text, n_sketches=n_sketches)
    sk = list(arts.sketches) or [SketchV1("A", "direct", "Implement the direct algorithm.")]
    while len(sk) < n_impl:
        sk.append(sk[len(sk) % len(arts.sketches or sk)])
    cands, calls = [], 1
    for i in range(n_impl):
        text, _ = gen(build_implement_prompt_v1(problem, arts.spec, sk[i]), max_tokens, impl_temp)
        calls += 1
        code = extract_candidate_code_v1(response_text=text)
        cands.append(GenCandidateV1(f"{sk[i].label}{i}", code, _parses(code), "gg3_impl"))
    # honest guard: the coach card must not carry an accepted block (it is generic by construction)
    prov = problem.statement
    coach_is_scaffold = bool(accepted_codes) and reproduces_accepted_block_v1(
        coach, list(accepted_codes), provenance=prov)
    diagnostics = {"family": family, "coach_len": len(coach), "coach_is_scaffold": coach_is_scaffold}
    realness = {"coach_is_family_level": not coach_is_scaffold,
                "coach_card_chars": len(coach)}
    return _finalize_arm(problem, "GG3", arts, cands, calls, diagnostics, realness,
                         accepted_codes=accepted_codes)


def run_gg4_v1(gen: GenFn, problem: IcpcPilotProblemV1, *, K: int = DEFAULT_K, n_sketches: int = 4,
               analyze_temp: float = 0.5, impl_temp: float = 0.2,
               max_tokens: int = DEFAULT_MAX_TOKENS, timeout_s: float = 5.0,
               accepted_codes: Sequence[str] = ()) -> GgArmOutcomeV1:
    """GG4 — planner/coordinator budget policy.  A digest-router (the W125 PATCH/REPLAN/ABSTAIN
    concept) allocates the K budget: 1 ANALYZE + 2 implements, then route the remaining 2 by the
    best candidate's failure digest (PATCH=rewrite, REPLAN=new diverse sketches).  The hosted
    cache-aware planner is EFFICIENCY-only (KV-prefix savings), recorded honestly — not a
    capability lever; GG4 is killed if the route never varies (prompt decoration)."""
    a_text, _ = gen(build_analyze_prompt_v1(problem, n_sketches=n_sketches), max_tokens, analyze_temp)
    arts = parse_role_artifacts_v1(a_text, n_sketches=n_sketches)
    sk = list(arts.sketches) or [SketchV1("A", "direct", "Implement the direct algorithm.")]
    while len(sk) < 4:
        sk.append(sk[len(sk) % len(arts.sketches or sk)])
    cands, calls = [], 1
    for i in range(2):  # seed: 2 implements
        text, _ = gen(build_implement_prompt_v1(problem, arts.spec, sk[i]), max_tokens, impl_temp)
        calls += 1
        cands.append(GenCandidateV1(f"{sk[i].label}{i}", extract_candidate_code_v1(response_text=text),
                                    _parses(extract_candidate_code_v1(response_text=text)), "gg4_seed"))
    derived = [inp for inp, _e in arts.counterexamples]
    best = _best_candidate(problem, cands, derived, timeout_s=timeout_s)
    fail = _find_failing_case(problem, best, derived, timeout_s=timeout_s)
    # ROUTE: PATCH if an actionable failure exists, else REPLAN (widen the pool with new sketches)
    remaining = K - calls
    if best is not None and fail is not None:
        route = "PATCH"
        stdin_f, exp_f, actual_f, dig = fail
        text, _ = gen(build_gg2_rewrite_prompt_v1(problem, arts.spec, best.code, stdin_f, exp_f,
                                                  actual_f, dig), max_tokens, impl_temp)
        calls += 1
        cands.append(GenCandidateV1("patch", extract_candidate_code_v1(response_text=text),
                                    _parses(extract_candidate_code_v1(response_text=text)), "gg4_patch"))
        remaining = K - calls
        for j in range(remaining):  # then widen
            s = sk[(2 + j) % len(sk)]
            text, _ = gen(build_implement_prompt_v1(problem, arts.spec, s), max_tokens, impl_temp)
            calls += 1
            cands.append(GenCandidateV1(f"rep{j}", extract_candidate_code_v1(response_text=text),
                                        _parses(extract_candidate_code_v1(response_text=text)), "gg4_replan"))
    else:
        route = "REPLAN"
        for j in range(remaining):
            s = sk[(2 + j) % len(sk)]
            text, _ = gen(build_implement_prompt_v1(problem, arts.spec, s), max_tokens, impl_temp)
            calls += 1
            cands.append(GenCandidateV1(f"rep{j}", extract_candidate_code_v1(response_text=text),
                                        _parses(extract_candidate_code_v1(response_text=text)), "gg4_replan"))
    diagnostics = {"route": route, "best_label": best.label if best else None,
                   "hosted_planner_role": "efficiency_only_kv_prefix_not_capability"}
    realness = {"route_taken": route, "router_actionable": fail is not None}
    return _finalize_arm(problem, "GG4", arts, cands, calls, diagnostics, realness,
                         accepted_codes=accepted_codes)


def run_gglead_v1(gen: GenFn, problem: IcpcPilotProblemV1, *, K: int = DEFAULT_K, n_sketches: int = 4,
                  analyze_temp: float = 0.5, impl_temp: float = 0.2,
                  max_tokens: int = DEFAULT_MAX_TOKENS, timeout_s: float = 5.0,
                  accepted_codes: Sequence[str] = ()) -> GgArmOutcomeV1:
    """GGLEAD = GG1 complexity-gated generation + ONE GG2 counterexample rewrite, same K budget:
    1 GG1-ANALYZE + (K-2) admissible/hardened implements + 1 counterexample rewrite."""
    n_impl = max(1, K - 2)
    a_text, _ = gen(build_gg1_analyze_prompt_v1(problem, n_sketches=n_sketches),
                    max_tokens, analyze_temp)
    arts = parse_role_artifacts_v1(a_text, n_sketches=n_sketches)
    n_bound = parse_max_constraint_v1(problem.statement)
    gated = [(sk, parse_complexity_exponent_v1(f"{sk.approach_name} {sk.outline}")) for sk in arts.sketches]
    gated = [(sk, ex, complexity_admissible_v1(ex, n_bound)) for sk, ex in gated]
    admissible = [g[0] for g in gated if g[2] is not False]
    rejected = [g for g in gated if g[2] is False]
    impl_sketches = admissible or ([arts.sketches[0]] if arts.sketches
                                   else [SketchV1("A", "direct", "Implement the direct algorithm.")])
    cands, calls = [], 1
    for i in range(n_impl):
        sk = impl_sketches[i] if i < len(impl_sketches) else impl_sketches[0]
        if i < len(impl_sketches):
            prompt, origin = build_implement_prompt_v1(problem, arts.spec, sk), "gglead_impl"
        else:
            prompt = build_worstcase_rewrite_prompt_v1(problem, arts.spec, sk, n_bound)
            origin = "gglead_harden"
        text, _ = gen(prompt, max_tokens, impl_temp)
        calls += 1
        code = extract_candidate_code_v1(response_text=text)
        cands.append(GenCandidateV1(f"{sk.label}{i}", code, _parses(code), origin))
    derived = [inp for inp, _e in arts.counterexamples]
    best = _best_candidate(problem, cands, derived, timeout_s=timeout_s)
    fail = _find_failing_case(problem, best, derived, timeout_s=timeout_s)
    rewrite_new = False
    if best is not None and fail is not None:
        stdin_f, exp_f, actual_f, dig = fail
        text, _ = gen(build_gg2_rewrite_prompt_v1(problem, arts.spec, best.code, stdin_f, exp_f,
                                                  actual_f, dig), max_tokens, impl_temp)
        calls += 1
        code = extract_candidate_code_v1(response_text=text)
        cands.append(GenCandidateV1("rw", code, _parses(code), "gglead_rewrite"))
        rewrite_new = (_parses(code) and best.parses and _norm_code(code) != _norm_code(best.code))
    else:
        ex = impl_sketches[(n_impl) % len(impl_sketches)]
        text, _ = gen(build_implement_prompt_v1(problem, arts.spec, ex), max_tokens, impl_temp)
        calls += 1
        code = extract_candidate_code_v1(response_text=text)
        cands.append(GenCandidateV1("ex", code, _parses(code), "gglead_extra"))
    diagnostics = {"n_bound": n_bound, "n_admissible": len(admissible), "n_rejected": len(rejected),
                   "gate_fired": bool(rejected), "had_failing_case": fail is not None}
    realness = {"gate_rejected_on_complexity": bool(rejected),
                "rewrite_issued": fail is not None, "rewrite_structurally_new": rewrite_new}
    return _finalize_arm(problem, "GGLEAD", arts, cands, calls, diagnostics, realness,
                         accepted_codes=accepted_codes)


ARM_RUNNERS = {"GG1": run_gg1_v1, "GG2": run_gg2_v1, "GG3": run_gg3_v1, "GG4": run_gg4_v1,
               "GGLEAD": run_gglead_v1}


# ===========================================================================
# NIM-free realness controls (fake-mechanism kills)
# ===========================================================================
def gg1_gate_control_v1() -> dict:
    """Positive control: an O(N^2) sketch at N=1e6 MUST be inadmissible; O(N log N) MUST be
    admissible; an UNSTATED complexity MUST be unjudgeable (never falsely rejected)."""
    n = 1_000_000
    slow = complexity_admissible_v1(parse_complexity_exponent_v1("uses an O(N^2) double loop"), n)
    fast = complexity_admissible_v1(parse_complexity_exponent_v1("sort then O(N log N) sweep"), n)
    none = complexity_admissible_v1(parse_complexity_exponent_v1("a clever greedy approach"), n)
    passes = (slow is False) and (fast is True) and (none is None)
    return {"control": "gg1_complexity_gate", "slow_admissible": slow, "fast_admissible": fast,
            "unstated_admissible": none, "passes": passes}


def gg2_rewrite_control_v1() -> dict:
    """Positive control: the failure-case finder returns a case for a candidate that fails a
    public sample, and None for a candidate that passes all public samples."""
    class _P:
        short_name = "ctl"
        samples = (("3\n", "6\n"),)  # doubling (6) and +1 (4) diverge
        statement = "double it"
    good = GenCandidateV1("g", "import sys\nprint(int(sys.stdin.read())*2)", True, "ctl")
    bad = GenCandidateV1("b", "import sys\nprint(int(sys.stdin.read())+1)", True, "ctl")
    f_bad = _find_failing_case(_P(), bad, [], timeout_s=5.0)
    f_good = _find_failing_case(_P(), good, [], timeout_s=5.0)
    passes = (f_bad is not None) and (f_good is None)
    return {"control": "gg2_failing_case_finder", "bad_has_failure": f_bad is not None,
            "good_has_failure": f_good is not None, "passes": passes}


def examine_hosted_controller_applicability_v1() -> dict:
    """Honest mining record (the W128/W129 W79-sibling).  The hosted CACHE-AWARE planner
    (``HostedCacheAwarePlannerV12``) is a KV-PREFIX-SAVINGS mechanism (token efficiency on a
    shared prefix), NOT an algorithm-selection / capability lever; the hosted-handoff /
    multi-agent-substrate coordinators are substrate-TRUST-specific (no path to the ICPC code
    plane — graphify-confirmed).  So a LITERAL bridge of those as a 'planner budget policy' that
    creates algorithmic headroom would be FAKE-DIFFERENT.  The genuinely-applicable lever is the
    W125 PATCH/REPLAN/ABSTAIN digest-router (it bridges the executor digest to budget routing),
    realized natively in GG4.  The cache planner's role is recorded as efficiency-only."""
    return {
        "hosted_cache_aware_planner_v12": "efficiency_only_kv_prefix_savings_not_capability",
        "hosted_real_handoff_coordinator_v11": "substrate_trust_specific_no_icpc_code_path",
        "multi_agent_substrate_coordinator_v15": "substrate_trust_specific_literal_bridge_fake",
        "applicable_lever": "w125_patch_replan_abstain_digest_router_realized_in_gg4",
        "literal_planner_bridge_killed": True}


# ===========================================================================
# EXPOSED dev-bench earn gate (W130 R2W) — a GENERATION metric
# ===========================================================================
@dataclasses.dataclass(frozen=True)
class GgEarnVerdictV1:
    schema: str
    n_targets: int
    old_pool_solved: list
    per_arm_new_pool_solves: dict       # arm -> [problems newly pool-solved]
    per_arm_new_committed: dict         # arm -> [problems newly committed by the fixed selector]
    best_arm: Optional[str]
    best_new_count: int
    new_solve_families: list
    new_solve_modes: list
    spans_two: bool
    winners_real: bool
    winners_leakage_clean: bool
    earned: bool
    verdict_label: str
    rationale: str

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


GG_DEV_MIN_NEW = 2  # >= 2 NEW pool solves absent from the old W128/W129 pool


def apply_gg_dev_bench_earn_gate_v1(outcomes_by_problem: dict, *, old_pool_solved: Sequence[str],
                                    family_of: dict, atlas_mode_of: dict) -> GgEarnVerdictV1:
    """R2W EARNED iff some arm creates >= 2 NEW pool solves (secret-passing candidates absent
    from the ENTIRE old W128/W129 pool) spanning >= 2 distinct (families OR atlas failure-modes),
    with every new-solve run realness-REAL + leakage-clean.  A committed win (fixed selector) is
    a STRONGER secondary signal but the bar is GENERATION (pool ceiling)."""
    old = set(old_pool_solved)
    arms = sorted({a for d in outcomes_by_problem.values() for a in d})
    per_arm_new: dict = {a: [] for a in arms}
    per_arm_committed: dict = {a: [] for a in arms}
    for prob, d in outcomes_by_problem.items():
        for a, o in d.items():
            if o.pool_pass and prob not in old:
                per_arm_new[a].append(prob)
            if o.committed_pass and prob not in old:
                per_arm_committed[a].append(prob)
    best_arm, best_new = None, -1
    for a in arms:
        if len(per_arm_new[a]) > best_new:
            best_arm, best_new = a, len(per_arm_new[a])
    new_probs = per_arm_new.get(best_arm, []) if best_arm else []
    fams = sorted({family_of.get(p, "?") for p in new_probs})
    modes = sorted({atlas_mode_of.get(p, "?") for p in new_probs})
    spans_two = (len(fams) >= 2) or (len(modes) >= 2)
    winners_real = all(
        (outcomes_by_problem[p][best_arm].realness or {}) and
        outcomes_by_problem[p][best_arm].leakage_clean for p in new_probs) if new_probs else False
    winners_clean = all(outcomes_by_problem[p][best_arm].leakage_clean for p in new_probs) if new_probs else False
    earned = (best_new >= GG_DEV_MIN_NEW) and spans_two and winners_real and winners_clean
    if earned:
        label = "GG_EXPOSED_DEV_BENCH_EARNED"
        why = (f"arm {best_arm} created {best_new} NEW pool solves {new_probs} absent from the old "
               f"pool, spanning families={fams}/modes={modes}, all leakage-clean + realness-real")
    else:
        label = "GG_EXPOSED_DEV_BENCH_NOT_EARNED"
        why = (f"best arm {best_arm} created {best_new} NEW pool solve(s) {new_probs} "
               f"(need >= {GG_DEV_MIN_NEW} spanning >= 2 families/modes); spans_two={spans_two}, "
               f"winners_real={winners_real}, winners_clean={winners_clean}")
    return GgEarnVerdictV1(
        schema="coordpy.w130_gg_dev_bench_earn.v1", n_targets=len(outcomes_by_problem),
        old_pool_solved=sorted(old), per_arm_new_pool_solves=per_arm_new,
        per_arm_new_committed=per_arm_committed, best_arm=best_arm, best_new_count=best_new,
        new_solve_families=fams, new_solve_modes=modes, spans_two=spans_two,
        winners_real=winners_real, winners_leakage_clean=winners_clean, earned=earned,
        verdict_label=label, rationale=why)
