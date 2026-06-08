"""W141 / COO-9 — emergent self-tutoring controller (Lane β orchestration).

Ties the W141 loop into one same-budget run, with the NO-ORACLE guard enforced by construction (the
model never sees ref/naive/brute/secret; the verifier uses only public + self-generated inputs + a
self-written brute):

  DISCOVER (on a teacher instance):  write a self-brute  +  generate K diverse efficient candidates
    →  no-oracle verify (self-brute agreement on self-gen small inputs + W134 complexity witness)
    →  pick the correct+efficient WINNER  →  AST-extract a leak-audited holed-skeleton scaffold.
  AMORTIZE (across held-out family members of the SAME technique): apply the ONE discovered scaffold
    to each member (a cheap scaffolded re-attempt), verify-select the answer per member.

The earn is family-level AMORTIZATION: verified-selection (Baseline B) must re-discover the technique
per member (K samples each); self-tutoring discovers ONCE and applies the scaffold to all — so at
EQUAL total generation budget over M members, discover-once-apply-many solves strictly more (the
discovery cost is amortized).  ``grade_on_secret_v1`` is used ONLY to SCORE the arms (never inside the
mechanism).  NON-NEGATIVITY: if discovery ABSTAINs or the scaffold is discarded, the member falls
back to plain self-consistency (KEEP ≡ A1).  Explicit-import only; ``coordpy/__init__.py`` untouched.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Callable, Optional, Sequence

from .resistant_by_construction_battlefield_v1 import MintedProblemV1, _exec_capture_v1
from .hard_battlefield_slate_v2 import ParserNeutralTemplateV2
from .icpc_reflexion_bench_v1 import (
    extract_candidate_code_v1, grade_on_secret_v1, IcpcPilotProblemV1)
from .public_signal_selection_oracle_v1 import derive_auto_cases_v1
from .no_oracle_verifier_v1 import select_winner_v1, WinnerSelectionV1
from .self_tutoring_technique_extractor_v1 import compile_tutor_from_winner_v1, SelfTutorCompileReportV1
from .family_tutor_compiler_v1 import FamilyTutorV1

SELF_TUTORING_CONTROLLER_V1_SCHEMA_VERSION: str = "coordpy.self_tutoring_controller_v1.v1"

GenFn = Callable[[str, int, float], Any]   # (prompt, max_tokens, temperature) -> (text, meta) | text


def _gen_text(gen: GenFn, prompt: str, max_tokens: int, temperature: float) -> str:
    out = gen(prompt, max_tokens, temperature)
    if isinstance(out, tuple):
        return out[0] or ""
    return out or ""


def _brute_prompt(pilot: IcpcPilotProblemV1) -> str:
    return (f"{pilot.statement}\n\n"
            "Write a SIMPLE, OBVIOUSLY-CORRECT brute-force Python 3 solution. It may be slow — "
            "prioritize correctness and clarity over speed (use the most direct method, even O(N^2)). "
            "Read all input from stdin, write only the answer to stdout, matching the exact format. "
            "Return ONLY one ```python code block.")


def _efficient_prompt(pilot: IcpcPilotProblemV1) -> str:
    # TRULY NEUTRAL baseline = the standard self-consistency prompt (the fair retirement baseline).
    # It names NO technique AND gives NO efficiency/time-limit/large-input hint (those words pushed the
    # 70B to write the efficient algorithm ~92% of the time on NSL — v3 confound; the genuine
    # self-consistency efficient-rate is ~32%, per the de-risk).  The model writes whatever it would
    # naturally; only the SCAFFOLD (the once-discovered technique) is supplied to the ST arm.  This is
    # the apples-to-apples "self-tutoring vs same-budget self-consistency" comparison the retirements use.
    return (f"{pilot.statement}\n\n"
            "Write a Python 3 program that solves this problem. Read all input from stdin and write the "
            "answer to stdout in the exact format shown. Return ONLY one ```python code block.")


def _scaffold_prompt(pilot: IcpcPilotProblemV1, scaffold: FamilyTutorV1) -> str:
    return (f"{pilot.statement}\n\n{scaffold.to_prompt_block()}\n\n"
            "Now write the COMPLETE, efficient Python 3 program for THIS problem, filling in every "
            "blank with the correct decision for this problem. Read from stdin, write to stdout, exact "
            "format. Return ONLY one ```python code block.")


def _self_small_inputs(pilot: IcpcPilotProblemV1, *, max_cases: int = 8) -> list[str]:
    """NO-ORACLE small inputs: format-preserving mutations of the PUBLIC samples (given)."""
    try:
        return [c for c in derive_auto_cases_v1(pilot, max_cases=max_cases)]
    except Exception:  # noqa: BLE001
        return [inp for inp, _ in pilot.samples][:max_cases]


def _adversarial_bank_prompt(pilot: IcpcPilotProblemV1) -> str:
    return (f"{pilot.statement}\n\n"
            "Write a Python 3 program that PRINTS several small but TRICKY test INPUTS for THIS "
            "problem (to stress-test candidate solutions). Print at least 12 inputs; separate "
            "consecutive inputs with a line containing exactly '====='. Each printed input must be a "
            "COMPLETE, VALID stdin in the EXACT input format above, with small size (N <= 60). Cover "
            "the adversarial cases that make each stated constraint BIND: all-equal values, strictly "
            "increasing, strictly decreasing, values at the stated minimum and maximum bounds, and the "
            "smallest valid N. The program reads NO input and prints ONLY those inputs. Return ONLY "
            "one ```python code block.")


def _gen_adversarial_bank_v1(pilot: IcpcPilotProblemV1, gen: GenFn, *, max_tokens: int = 1024,
                             timeout_s: float = 4.0, cap: int = 24) -> list[str]:
    """The CONSTRAINT-ADVERSARIAL S1 bank (Agent A): the team writes a generator that prints tricky
    constraint-binding inputs; we exec it in an isolated subprocess (the audited ``_exec_capture_v1``)
    and split on the delimiter.  This is the documented escape from the W125 'looks_right_fails_hidden'
    cap (28/30 discrimination vs 0/30 for a public-only bank).  Returns [] on any failure (the caller
    still has the random + public-mutation cases)."""
    code = extract_candidate_code_v1(response_text=_gen_text(gen, _adversarial_bank_prompt(pilot),
                                                             max_tokens, 0.3))
    if not code.strip():
        return []
    r = _exec_capture_v1(code, "", timeout_s=float(timeout_s))
    if r.timed_out or r.returncode != 0 or not r.stdout:
        return []
    parts = [p.strip() for p in r.stdout.split("=====")]
    return [p for p in parts if p.strip()][:cap]


def _build_small_bank_v1(pilot: IcpcPilotProblemV1, template: ParserNeutralTemplateV2,
                         gen: Optional[GenFn], *, n_random: int = 10, max_tokens: int = 1024,
                         timeout_s: float = 4.0) -> list[str]:
    """The full no-oracle S1 bank: random small inputs (template generator) + public-sample mutations
    + the model-adversarial constraint-binding bank.  All sources are public/self-generated (no
    oracle)."""
    import random as _random
    bank: list[str] = []
    try:
        for s in range(n_random):
            bank.append("\n".join(template.minted.gen_public(_random.Random(7000 + s))))
    except Exception:  # noqa: BLE001
        pass
    bank += _self_small_inputs(pilot)
    if gen is not None:
        bank += _gen_adversarial_bank_v1(pilot, gen, max_tokens=max_tokens, timeout_s=timeout_s)
    seen: set[str] = set()
    out: list[str] = []
    for b in bank:
        k = b.strip()
        if k and k not in seen:
            seen.add(k)
            out.append(k)
    return out[:32]


@dataclasses.dataclass(frozen=True)
class DiscoverResultV1:
    scaffold: Optional[FamilyTutorV1]
    selection: WinnerSelectionV1
    compile_report: Optional[SelfTutorCompileReportV1]
    n_gen_calls: int            # budget spent on discovery (1 brute + K candidates)
    winner_passes_secret: Optional[bool]   # SCORING ONLY (never used by the mechanism)

    @property
    def discovered(self) -> bool:
        return self.scaffold is not None

    def to_dict(self) -> dict[str, Any]:
        return {"discovered": self.discovered, "n_gen_calls": self.n_gen_calls,
                "selection": self.selection.to_dict(),
                "compile": (self.compile_report.to_dict() if self.compile_report else None),
                "winner_passes_secret": self.winner_passes_secret}


def discover_self_scaffold_v1(
        template: ParserNeutralTemplateV2, problem: MintedProblemV1, *, gen: GenFn,
        K: int = 5, temperature: float = 0.7, max_tokens: int = 1536, timeout_s: float = 4.0,
        minted_date: str = "2026-06-08") -> DiscoverResultV1:
    """One teacher instance: self-brute + K diverse efficient candidates → no-oracle verify → extract.
    Spends K+1 gen calls.  Returns a (possibly None) scaffold; None ⇒ caller KEEPs."""
    pilot = problem.to_pilot_problem(minted_date=minted_date)
    brute_code = extract_candidate_code_v1(response_text=_gen_text(gen, _brute_prompt(pilot), max_tokens, 0.2))
    cands = [extract_candidate_code_v1(response_text=_gen_text(gen, _efficient_prompt(pilot), max_tokens, temperature))
             for _ in range(K)]
    small = _build_small_bank_v1(pilot, template, gen, timeout_s=timeout_s)
    sel = select_winner_v1(cands, statement=pilot.statement, samples=list(problem.samples),
                           small_inputs=small, brute_code=brute_code,
                           consensus_probe=(small[0] if small else ""), timeout_s=timeout_s)
    scaffold: Optional[FamilyTutorV1] = None
    rep: Optional[SelfTutorCompileReportV1] = None
    win_secret: Optional[bool] = None
    if not sel.abstained and sel.winner_code:
        win_secret = _passes_secret(problem, sel.winner_code, timeout_s)  # SCORING ONLY
        scaffold, rep = compile_tutor_from_winner_v1(sel.winner_code, template, problem, timeout_s=timeout_s)
    return DiscoverResultV1(scaffold, sel, rep, n_gen_calls=K + 1, winner_passes_secret=win_secret)


@dataclasses.dataclass(frozen=True)
class MemberArmResultV1:
    member_key: str
    a1_pool_pass: bool          # self-consistency: ANY of K plain passes secret (pool ceiling)
    b_selected_pass: bool       # verified-selection: the no-oracle pick passes secret
    st_selected_pass: bool      # self-tutoring: the scaffolded no-oracle pick passes secret
    n_gen_a1: int
    n_gen_st: int
    n_plain: int = 0            # PER-SAMPLE rate evidence (the metric the integer solve hides):
    n_plain_pass: int = 0      #   raw rate p = n_plain_pass / n_plain
    n_scaffold: int = 0
    n_scaffold_pass: int = 0   #   scaffolded rate q = n_scaffold_pass / n_scaffold

    def to_dict(self) -> dict[str, Any]:
        return {"member": self.member_key, "a1_pool_pass": self.a1_pool_pass,
                "b_selected_pass": self.b_selected_pass, "st_selected_pass": self.st_selected_pass,
                "n_gen_a1": self.n_gen_a1, "n_gen_st": self.n_gen_st,
                "n_plain": self.n_plain, "n_plain_pass": self.n_plain_pass,
                "n_scaffold": self.n_scaffold, "n_scaffold_pass": self.n_scaffold_pass}


def run_member_arms_v1(
        template: ParserNeutralTemplateV2, problem: MintedProblemV1, scaffold: Optional[FamilyTutorV1],
        *, gen: GenFn, K: int = 5, K_re: int = 4, temperature: float = 0.7, max_tokens: int = 1536,
        timeout_s: float = 4.0, minted_date: str = "2026-06-08", member_key: str = "") -> MemberArmResultV1:
    """One held-out member: A1 (K plain), Baseline B (K plain → no-oracle select), Self-Tutoring
    (K_re scaffolded → no-oracle select).  If no scaffold, ST falls back to B (KEEP ≡ verified
    selection, never worse than A1)."""
    pilot = problem.to_pilot_problem(minted_date=minted_date)
    small = _build_small_bank_v1(pilot, template, gen, timeout_s=timeout_s)
    brute_code = extract_candidate_code_v1(response_text=_gen_text(gen, _brute_prompt(pilot), max_tokens, 0.2))
    # A1 / B share the same K plain efficient candidates (same budget)
    plain = [extract_candidate_code_v1(response_text=_gen_text(gen, _efficient_prompt(pilot), max_tokens, temperature))
             for _ in range(K)]
    plain_pass = [_passes_secret(problem, c, timeout_s) for c in plain]   # SCORING ONLY (raw rate p)
    a1_pool = any(plain_pass)
    sel_b = select_winner_v1(plain, statement=pilot.statement, samples=list(problem.samples),
                             small_inputs=small, brute_code=brute_code,
                             consensus_probe=(small[0] if small else ""), timeout_s=timeout_s)
    b_pass = bool((not sel_b.abstained) and sel_b.winner_code
                  and _passes_secret(problem, sel_b.winner_code, timeout_s))
    # Self-tutoring: scaffolded re-attempts (or KEEP ≡ B if no scaffold)
    if scaffold is not None:
        st_cands = [extract_candidate_code_v1(response_text=_gen_text(gen, _scaffold_prompt(pilot, scaffold), max_tokens, temperature))
                    for _ in range(K_re)]
        st_pass_each = [_passes_secret(problem, c, timeout_s) for c in st_cands]   # SCORING ONLY (q)
        sel_st = select_winner_v1(st_cands, statement=pilot.statement, samples=list(problem.samples),
                                  small_inputs=small, brute_code=brute_code,
                                  consensus_probe=(small[0] if small else ""), timeout_s=timeout_s)
        st_pass = bool((not sel_st.abstained) and sel_st.winner_code
                       and _passes_secret(problem, sel_st.winner_code, timeout_s))
        n_gen_st, n_scaffold, n_scaffold_pass = K_re, len(st_cands), sum(st_pass_each)
    else:
        st_pass = b_pass
        n_gen_st, n_scaffold, n_scaffold_pass = 0, 0, 0
    return MemberArmResultV1(member_key=member_key, a1_pool_pass=a1_pool, b_selected_pass=b_pass,
                             st_selected_pass=st_pass, n_gen_a1=K, n_gen_st=n_gen_st,
                             n_plain=len(plain), n_plain_pass=sum(plain_pass),
                             n_scaffold=n_scaffold, n_scaffold_pass=n_scaffold_pass)


@dataclasses.dataclass(frozen=True)
class AmortizationVerdictV1:
    family: str
    n_members: int
    a1_solved: int
    b_solved: int
    st_solved: int
    discovered: bool
    st_minus_b: int
    st_minus_a1: int
    earned: bool                 # ST solves strictly more held-out members than B
    non_negative: bool           # ST >= B on every member (KEEP guarantee)
    raw_rate_p: float = 0.0      # PER-SAMPLE evidence: plain efficient-pass rate
    scaffold_rate_q: float = 0.0 #   scaffolded efficient-pass rate
    scaffold_lift_pp: float = 0.0  # 100*(q - p): the self-derived scaffold's generation lift

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def amortization_verdict_v1(family: str, discovered: bool, members: Sequence[MemberArmResultV1]
                            ) -> AmortizationVerdictV1:
    a1 = sum(m.a1_pool_pass for m in members)
    b = sum(m.b_selected_pass for m in members)
    st = sum(m.st_selected_pass for m in members)
    non_neg = all(m.st_selected_pass >= m.b_selected_pass for m in members)
    n_pl = sum(m.n_plain for m in members)
    n_sc = sum(m.n_scaffold for m in members)
    p = (sum(m.n_plain_pass for m in members) / n_pl) if n_pl else 0.0
    q = (sum(m.n_scaffold_pass for m in members) / n_sc) if n_sc else 0.0
    return AmortizationVerdictV1(
        family=family, n_members=len(members), a1_solved=a1, b_solved=b, st_solved=st,
        discovered=discovered, st_minus_b=st - b, st_minus_a1=st - a1,
        earned=bool(st > b), non_negative=bool(non_neg),
        raw_rate_p=round(p, 4), scaffold_rate_q=round(q, 4), scaffold_lift_pp=round(100 * (q - p), 2))


def _passes_secret(problem: MintedProblemV1, code: str, timeout_s: float) -> bool:
    """SCORING ONLY — grade a candidate on the hidden bank. NEVER called by the mechanism."""
    pilot = problem.to_pilot_problem(minted_date="2026-06-08")
    passed, _stderr, _n = grade_on_secret_v1(pilot, code, timeout_s=timeout_s)
    return bool(passed)
