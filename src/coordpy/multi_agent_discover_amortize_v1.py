"""W143 — multi-agent DISCOVER-THEN-AMORTIZE team composition (the class-gap bridge).

Fuses three previously-DISCONNECTED lines (graphify: NO PATH self_tutoring↔team/consensus; separate
communities) into a single genuine multi-agent discover-then-amortize TEAM:

  * W128 role-diverse discovery   (`role_diverse_algorithm_search_v1`: STRATEGIST sketches + IMPLEMENTERS)
  * W142b no-oracle verifier/extr (`no_oracle_verifier_v2.select_winner_v2` + `compile_tutor_from_winner_v1`)
  * W48/W52 shared-state transfer (structured holed-skeleton object vs raw transcript vs structure-empty)

It is the FIRST 1-hop import bridge between the self-tutoring cluster (community 4109) and the
role-diverse-search cluster.  The team mechanism here is NIM-free testable (pass a mock ``gen``).

TEAM-REALITY (RUNBOOK §2): >=3 explicit roles with distinct responsibilities — STRATEGIST (diverse
algorithmic sketches, statement-only, no oracle), IMPLEMENTERS (one per sketch, forced to follow its
assigned algorithm), BRUTE-AUTHORS / VERIFIER-QUORUM (>=2 independent self-brute roles under different
convention prompts; their consensus anchors the correctness cluster), EXTRACTOR/TEACHER (compile the
holed-skeleton scaffold from the verified winner), AMORTIZERS (per member, read the scaffold from a
structured shared-state object and solve).  The team COMMIT step (verifier-quorum cluster-with-a-brute
consensus) is not reducible to one prompt chain.

ABLATION KNOBS (RUNBOOK §5 + §14): ``role_diverse`` (candidate diversity on/off), ``brute_diverse``
(verifier-quorum diversity on/off), ``transfer`` (shared_state / transcript / empty / none), and
``rationale_alien`` (the Q1 noise control: sketches from a DIFFERENT problem).  Every config spends the
SAME discovery budget K_d (candidate-side) + K_b (brute-side) and the SAME amortize budget K_a — the
budget-parity invariant (RUNBOOK §7), enforced by ``team_budget_parity_v1``.

STRICTLY NO-ORACLE: STRATEGIST sketches + brute-author brutes are model-self-generated; the hidden
grader is never read by the mechanism path (``passes_secret_fn`` is SCORING-ONLY audit, optional).
NON-NEGATIVE: failed discovery (no clean extractable verified winner) => scaffold=None => KEEP (== B0).
"""
from __future__ import annotations

import dataclasses
from typing import Any, Callable, Optional, Sequence

from .icpc_reflexion_bench_v1 import extract_candidate_code_v1
from .self_tutoring_controller_v1 import _efficient_prompt, _scaffold_prompt, _gen_text
from .no_oracle_verifier_v2 import select_winner_v2
from .self_tutoring_technique_extractor_v1 import compile_tutor_from_winner_v1
from .family_tutor_compiler_v1 import FamilyTutorV1, make_negative_control_tutor_v1
from .role_diverse_algorithm_search_v1 import (
    SketchV1, RoleArtifactsV1, CandidateImplV1, DiversityReportV1,
    build_analyze_prompt_v1, build_implement_prompt_v1, parse_role_artifacts_v1,
    compute_diversity_v1, fake_diversity_control_v1, DEFAULT_N_SKETCHES, _parses)

GenFn = Callable[[str, int, float], Any]
MULTI_AGENT_DISCOVER_AMORTIZE_V1_SCHEMA_VERSION: str = "coordpy.multi_agent_discover_amortize_v1.v1"

# ---- roles (team-reality #1) ---------------------------------------------------------------------
ROLE_STRATEGIST = "strategist"
ROLE_IMPLEMENTER = "implementer"
ROLE_BRUTE_AUTHOR = "brute_author"
ROLE_EXTRACTOR = "extractor"
ROLE_AMORTIZER = "amortizer"
TEAM_ROLES: tuple[str, ...] = (ROLE_STRATEGIST, ROLE_IMPLEMENTER, ROLE_BRUTE_AUTHOR,
                               ROLE_EXTRACTOR, ROLE_AMORTIZER)

# ---- transfer modes (the shared-state ablation, RUNBOOK §14 Q2 3-arm) --------------------------
TRANSFER_SHARED_STATE = "shared_state"   # compiled holed FamilyTutorV1 (structured abstraction) == W142b ST
TRANSFER_TRANSCRIPT = "transcript"       # raw verified-winner code text (the W52 transcript-only null)
TRANSFER_EMPTY = "empty"                 # content-free tutor (structure-but-empty control)
TRANSFER_NONE = "none"                   # no transfer (== B0 plain path)
TRANSFER_MODES: tuple[str, ...] = (TRANSFER_SHARED_STATE, TRANSFER_TRANSCRIPT, TRANSFER_EMPTY, TRANSFER_NONE)


# ==================================================================================================
# verifier-quorum: >=2 independent brute-author roles under DISTINCT convention prompts
# ==================================================================================================
def brute_prompt_default_v1(pilot) -> str:
    """Brute convention A (the W142b convention-explicit brute) — careful unordered-pair / all-conditions."""
    return (f"{pilot.statement}\n\n"
            "Write a SIMPLE, OBVIOUSLY-CORRECT brute-force Python 3 solution. Prioritize CORRECTNESS "
            "over speed (use the most direct method, even O(N^2) or O(N^3)). Implement the EXACT "
            "definition in the statement: pay careful attention to WHICH items are counted (e.g. count "
            "each unordered pair {i,j} with i<j exactly once, not ordered pairs) and to EVERY stated "
            "condition (if several conditions must ALL hold for an item to count, check them all). "
            "Make sure your output matches the public example(s) EXACTLY. Read all input from stdin, "
            "write only the answer to stdout. Return ONLY one ```python code block.")


def brute_prompt_alt_v1(pilot) -> str:
    """Brute convention B (a DISTINCT independent verifier role): nested-loop enumeration, explicit
    counter, no clever early-exit — a different code path that should AGREE with convention A on
    correct outputs but is written independently (the quorum's second voter)."""
    return (f"{pilot.statement}\n\n"
            "Act as an INDEPENDENT reference checker. Write a reference Python 3 solution by DIRECT "
            "EXHAUSTIVE ENUMERATION: explicitly loop over every candidate (every index, pair, triple, "
            "or subarray as the statement requires), test the full condition with a plain boolean "
            "check, and increment an integer counter. Do not optimize and do not use any library "
            "shortcut that hides the definition. Re-read the statement and handle EVERY clause and the "
            "exact counting rule (unordered vs ordered, strict vs non-strict). Match the public "
            "sample(s) exactly. Read stdin, print only the answer. Return ONLY one ```python code block.")


def _gen_brutes(gen: GenFn, pilot, *, K_b: int, brute_diverse: bool,
                max_tokens: int, temperature: float) -> list[str]:
    """K_b self-brutes.  If ``brute_diverse`` the quorum alternates >=2 distinct convention prompts
    (diverse independent verifier roles); else all K_b share convention A (== W142b single-controller).
    EITHER WAY exactly K_b model calls (budget parity)."""
    out: list[str] = []
    for i in range(K_b):
        if brute_diverse and (i % 2 == 1):
            prompt = brute_prompt_alt_v1(pilot)
        else:
            prompt = brute_prompt_default_v1(pilot)
        out.append(extract_candidate_code_v1(response_text=_gen_text(gen, prompt, max_tokens, temperature)))
    return out


# ==================================================================================================
# config + result
# ==================================================================================================
@dataclasses.dataclass(frozen=True)
class TeamConfigV1:
    """Ablation knobs for one discovery arm.  Presets via ``arm_config``."""
    arm: str
    role_diverse: bool          # True: STRATEGIST + IMPLEMENTERS (W128); False: i.i.d. _efficient_prompt
    brute_diverse: bool         # True: verifier-quorum across >=2 brute conventions; False: single convention
    transfer: str               # TRANSFER_*  (how the AMORTIZER reads the discovered technique)
    rationale_alien: bool = False   # Q1 noise control: STRATEGIST sketches from a DIFFERENT-family problem
    n_sketches: int = DEFAULT_N_SKETCHES

    def is_team(self) -> bool:
        """Team-reality #1/#3: a genuine multi-agent arm has role-diverse discovery OR a brute-author
        quorum (>=3 distinct roles firing with a cross-role commit step).  ST (both off) is the
        single-controller baseline, NOT a team."""
        return bool(self.role_diverse or self.brute_diverse)

    def n_active_roles(self) -> int:
        n = 2  # IMPLEMENTER/candidate-author + EXTRACTOR always present
        if self.role_diverse:
            n += 1  # STRATEGIST
        if self.brute_diverse:
            n += 1  # >=2 distinct BRUTE-AUTHOR roles (quorum)
        else:
            n += 0  # a single brute author folds into one verifier role
        if self.transfer in (TRANSFER_SHARED_STATE, TRANSFER_TRANSCRIPT, TRANSFER_EMPTY):
            n += 1  # AMORTIZER reads a transferred artifact
        return n

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["is_team"] = self.is_team()
        d["n_active_roles"] = self.n_active_roles()
        return d


def arm_config(arm: str) -> TeamConfigV1:
    """The LOCKED arm slate (RUNBOOK §5 + §14)."""
    presets = {
        # single-controller baseline (== W142b ST): i.i.d. candidates, single-convention brutes, scaffold transfer
        "ST":      TeamConfigV1("ST",      role_diverse=False, brute_diverse=False, transfer=TRANSFER_SHARED_STATE),
        # the LEAD team: role-diverse candidates + verifier-quorum + structured shared-state
        "MA_FULL": TeamConfigV1("MA_FULL", role_diverse=True,  brute_diverse=True,  transfer=TRANSFER_SHARED_STATE),
        # ablations (each isolates one team lever, same budget)
        "MA_RD":   TeamConfigV1("MA_RD",   role_diverse=False, brute_diverse=True,  transfer=TRANSFER_SHARED_STATE),  # role-diversity OFF
        "MA_Q":    TeamConfigV1("MA_Q",    role_diverse=True,  brute_diverse=False, transfer=TRANSFER_SHARED_STATE),  # quorum OFF
        "MA_SS":   TeamConfigV1("MA_SS",   role_diverse=True,  brute_diverse=True,  transfer=TRANSFER_TRANSCRIPT),    # shared-state OFF -> transcript
        "MA_SE":   TeamConfigV1("MA_SE",   role_diverse=True,  brute_diverse=True,  transfer=TRANSFER_EMPTY),         # structure-but-empty (Q2)
        # negative controls
        "NEG_RAT": TeamConfigV1("NEG_RAT", role_diverse=True,  brute_diverse=True,  transfer=TRANSFER_SHARED_STATE, rationale_alien=True),  # Q1 alien-rationale
    }
    if arm not in presets:
        raise ValueError(f"unknown arm {arm!r}; known: {sorted(presets)}")
    return presets[arm]


ARM_SLATE: tuple[str, ...] = ("ST", "MA_FULL", "MA_RD", "MA_Q", "MA_SS", "MA_SE", "NEG_RAT")


@dataclasses.dataclass(frozen=True)
class TeamDiscoverResultV1:
    arm: str
    discovered: bool
    scaffold: Optional[FamilyTutorV1]          # the holed-skeleton tutor (None => KEEP == B0)
    winner_code: Optional[str]                 # raw verified winner (for the transcript transfer)
    diversity_classify: str                    # "REAL" | "FAKE_DIVERSE" | "NA" (i.i.d. arm)
    diversity: Optional[dict]                  # DiversityReportV1.to_dict() | None
    n_analyze: int                             # ANALYZE model calls (0 or 1)
    n_candidates: int                          # candidate-side model calls
    n_brutes: int                              # brute-side model calls
    n_model_calls: int                         # total discovery model calls == G_d
    select_reason: str
    winner_passes_secret: Optional[bool]       # SCORING-ONLY audit (NOT used by the mechanism)
    n_correct_brutes: Optional[int]            # SCORING-ONLY audit
    n_disc_tries: int

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["scaffold"] = (self.scaffold.cid() if self.scaffold is not None else None)
        d["winner_code"] = bool(self.winner_code)
        return d


# ==================================================================================================
# DISCOVERY (role-diverse + verifier-quorum + extraction)
# ==================================================================================================
def _gen_candidates(gen: GenFn, teacher_pilot, *, config: TeamConfigV1, K_d: int,
                    max_tokens: int, temperature: float,
                    rationale_pilot=None) -> tuple[list[str], Optional[RoleArtifactsV1],
                                                   list[CandidateImplV1], int, int]:
    """Return (candidate_codes, artifacts|None, impls, n_analyze, n_candidate_calls).

    role_diverse: 1 ANALYZE (on the teacher, or on ``rationale_pilot`` if rationale_alien) -> sketches;
    then (K_d-1) IMPLEMENT calls round-robin over the sketches (on the TEACHER).  Total candidate-side
    calls == K_d (the ANALYZE replaces one i.i.d. candidate -> budget parity with ST).

    else (i.i.d.): K_d candidates from the neutral _efficient_prompt(teacher).  Total == K_d.
    """
    if not config.role_diverse:
        cands = [extract_candidate_code_v1(response_text=_gen_text(gen, _efficient_prompt(teacher_pilot), max_tokens, temperature))
                 for _ in range(K_d)]
        return cands, None, [], 0, K_d

    # STRATEGIST: diverse algorithmic sketches (statement-only, no oracle).
    analyze_src = rationale_pilot if (config.rationale_alien and rationale_pilot is not None) else teacher_pilot
    analyze_text = _gen_text(gen, build_analyze_prompt_v1(analyze_src, n_sketches=config.n_sketches),
                             max_tokens, temperature)
    artifacts = parse_role_artifacts_v1(analyze_text, n_sketches=config.n_sketches)
    sketches = list(artifacts.sketches)
    if not sketches:  # degenerate parse: fall back to a single neutral sketch so IMPLEMENT still fires
        sketches = [SketchV1("A", "direct", "implement a correct, efficient solution")]

    n_impl = max(0, K_d - 1)  # the ANALYZE call consumes 1 of the K_d candidate-side calls
    impls: list[CandidateImplV1] = []
    cands: list[str] = []
    for j in range(n_impl):
        sk = sketches[j % len(sketches)]
        code = extract_candidate_code_v1(
            response_text=_gen_text(gen, build_implement_prompt_v1(teacher_pilot, artifacts.spec, sk),
                                    max_tokens, temperature))
        impls.append(CandidateImplV1(label=f"{sk.label}{j}", code=code, parses=_parses(code)))
        cands.append(code)
    return cands, artifacts, impls, 1, K_d


def team_discover_v1(gen: GenFn, teacher_minted, template, *, config: TeamConfigV1,
                     K_d: int, K_b: int, max_tokens: int = 1536, temperature: float = 0.7,
                     brute_temp: float = 0.4, timeout_s: float = 4.0, minted_date: str = "2026-06-08",
                     max_disc_tries: int = 1, passes_secret_fn=None,
                     rationale_minted=None) -> TeamDiscoverResultV1:
    """One discovery attempt for ``config`` at the (fragile) budget (K_d candidate calls + K_b brute
    calls).  ``max_disc_tries=1`` is the LOAD-BEARING setting (single fragile-budget shot — the
    difference between arms is purely allocation+diversity, not retry budget).  Raising it reproduces
    the W142b deterministic-discovery retry loop (amortized one-time cost).

    No-oracle: brutes + sketches are model-generated; ``passes_secret_fn`` (if given) is a SCORING-ONLY
    audit recorded but never used to pick the winner.
    """
    teacher_pilot = teacher_minted.to_pilot_problem(minted_date=minted_date)
    rationale_pilot = (rationale_minted.to_pilot_problem(minted_date=minted_date)
                       if rationale_minted is not None else None)

    scaffold: Optional[FamilyTutorV1] = None
    winner_code: Optional[str] = None
    diversity: Optional[dict] = None
    diversity_classify = "NA"
    win_secret: Optional[bool] = None
    n_correct: Optional[int] = None
    n_analyze_tot = 0
    n_cand_tot = 0
    n_brute_tot = 0
    sel_reason = "no_attempt"
    n_tries = 0
    cands_acc: list[str] = []

    while scaffold is None and n_tries < max_disc_tries:
        n_tries += 1
        brutes = _gen_brutes(gen, teacher_pilot, K_b=K_b, brute_diverse=config.brute_diverse,
                             max_tokens=max_tokens, temperature=brute_temp)
        n_brute_tot += K_b
        cands, artifacts, impls, n_analyze, n_cand = _gen_candidates(
            gen, teacher_pilot, config=config, K_d=K_d, max_tokens=max_tokens,
            temperature=temperature, rationale_pilot=rationale_pilot)
        n_analyze_tot += n_analyze
        n_cand_tot += n_cand
        cands_acc = cands_acc + cands

        if artifacts is not None:
            rep = compute_diversity_v1(artifacts, impls, sample_inputs=[s for s, _ in teacher_pilot.samples])
            diversity = rep.to_dict()
            diversity_classify = rep.classify()

        sel = select_winner_v2(cands_acc, statement=teacher_pilot.statement,
                               samples=list(teacher_minted.samples), small_inputs=[],
                               brute_codes=brutes, io_shape=template.io_shape, timeout_s=timeout_s)
        sel_reason = sel.reason
        if sel.abstained:
            continue
        # multi-winner extraction: try EVERY verified-correct winner until one extracts a clean scaffold
        winners = [cands_acc[v.idx] for v in sel.verdicts if v.is_winner and 0 <= v.idx < len(cands_acc)]
        if sel.winner_code:
            winners = [sel.winner_code] + [w for w in winners if w != sel.winner_code]
        for w in winners:
            if passes_secret_fn is not None and win_secret is None:
                win_secret = bool(passes_secret_fn(teacher_minted, w, timeout_s))
            scf, _cr = compile_tutor_from_winner_v1(w, template, teacher_minted, timeout_s=timeout_s)
            if scf is not None:
                scaffold, winner_code = scf, w
                if passes_secret_fn is not None:
                    win_secret = bool(passes_secret_fn(teacher_minted, w, timeout_s))
                break
        if winner_code is None and sel.winner_code:
            winner_code = sel.winner_code  # keep the verified winner for the transcript transfer even if it didn't extract

    # n_correct_brutes is a SCORING-ONLY audit the driver fills (needs the grader); left None here.
    return TeamDiscoverResultV1(
        arm=config.arm, discovered=scaffold is not None, scaffold=scaffold, winner_code=winner_code,
        diversity_classify=diversity_classify, diversity=diversity,
        n_analyze=n_analyze_tot, n_candidates=n_cand_tot, n_brutes=n_brute_tot,
        n_model_calls=n_cand_tot + n_brute_tot,  # candidate-side (incl. the ANALYZE) + brute-side == per-try (K_d+K_b)
        select_reason=sel_reason, winner_passes_secret=win_secret, n_correct_brutes=n_correct,
        n_disc_tries=n_tries)


# ==================================================================================================
# AMORTIZE (the transfer ablation: shared-state vs transcript vs structure-empty vs none)
# ==================================================================================================
def _transcript_prompt(pilot, winner_code: str) -> str:
    """TRANSFER_TRANSCRIPT: the AMORTIZER receives the RAW verified-winner code (a worked solution to a
    related same-family problem) instead of the abstracted holed skeleton.  Carries MORE literal tokens
    than the holed skeleton — so a shared-state win over this arm is the STRUCTURE, not the token count
    (RUNBOOK §14 Q2)."""
    return (f"{pilot.statement}\n\n"
            "Here is a worked Python 3 solution to a CLOSELY RELATED problem from the same family. "
            "Study its approach and adapt it to solve THIS problem (sizes/constraints may differ):\n\n"
            f"```python\n{winner_code}\n```\n\n"
            "Read all input from stdin and write the answer to stdout in the exact format shown. "
            "Return ONLY one ```python code block.")


def amortize_prompt_v1(pilot, *, transfer: str, scaffold: Optional[FamilyTutorV1],
                       winner_code: Optional[str], empty_tutor: Optional[FamilyTutorV1]) -> Optional[str]:
    """The AMORTIZER's prompt for the given transfer mode.  Returns None when the transfer cannot be
    supplied (=> caller KEEPs == B0)."""
    if transfer == TRANSFER_SHARED_STATE:
        return _scaffold_prompt(pilot, scaffold) if scaffold is not None else None
    if transfer == TRANSFER_TRANSCRIPT:
        return _transcript_prompt(pilot, winner_code) if winner_code else None
    if transfer == TRANSFER_EMPTY:
        return _scaffold_prompt(pilot, empty_tutor) if empty_tutor is not None else None
    if transfer == TRANSFER_NONE:
        return _efficient_prompt(pilot)
    raise ValueError(f"unknown transfer {transfer!r}")


def amortize_member_v1(gen: GenFn, member_minted, template, *, config: TeamConfigV1,
                       discover: TeamDiscoverResultV1, brutes: Sequence[str], K_a: int,
                       max_tokens: int = 1536, temperature: float = 0.7, timeout_s: float = 4.0,
                       minted_date: str = "2026-06-08", b0_pass: Optional[bool] = None,
                       empty_tutor: Optional[FamilyTutorV1] = None,
                       passes_secret_fn=None) -> bool:
    """Amortize one family member under the arm's transfer mode.  K_a scaffolded draws -> v2 select ->
    pass.  NON-NEGATIVE: if the transfer can't be supplied (e.g. discovery failed) -> KEEP == B0.
    SCORING-ONLY ``passes_secret_fn`` grades the committed winner."""
    mp = member_minted.to_pilot_problem(minted_date=minted_date)
    prompt = amortize_prompt_v1(mp, transfer=config.transfer, scaffold=discover.scaffold,
                                winner_code=discover.winner_code, empty_tutor=empty_tutor)
    if prompt is None:
        return bool(b0_pass) if b0_pass is not None else False  # KEEP == B0
    cands = [extract_candidate_code_v1(response_text=_gen_text(gen, prompt, max_tokens, temperature))
             for _ in range(K_a)]
    sel = select_winner_v2(cands, statement=mp.statement, samples=list(member_minted.samples),
                           small_inputs=[], brute_codes=list(brutes), io_shape=template.io_shape,
                           timeout_s=timeout_s)
    if sel.abstained or not sel.winner_code:
        return bool(b0_pass) if b0_pass is not None else False  # KEEP
    if passes_secret_fn is not None:
        return bool(passes_secret_fn(member_minted, sel.winner_code, timeout_s))
    return True


# ==================================================================================================
# budget parity (RUNBOOK §7) — every arm spends the SAME G_d (K_d + K_b) + M*K_a
# ==================================================================================================
@dataclasses.dataclass(frozen=True)
class TeamArmBudgetV1:
    arm: str
    declared_discovery: int       # K_d + K_b  (candidate-side + brute-side, one-time)
    actual_discovery: int
    declared_amortize: int        # M * K_a
    actual_amortize: int
    ok: bool

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class TeamBudgetReportV1:
    M: int
    K_a: int
    K_d: int
    K_b: int
    G_d: int                      # the shared one-time discovery budget = K_d + K_b
    arms: tuple[TeamArmBudgetV1, ...]
    same_budget_identity_holds: bool
    note: str

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["arms"] = [a.to_dict() for a in self.arms]
        return d


def team_budget_parity_v1(*, M: int, K_a: int, K_d: int, K_b: int,
                          observed: dict[str, dict[str, int]] | None = None) -> TeamBudgetReportV1:
    """Pre-registered team budget identity.  Each arm's one-time DISCOVERY budget == K_d + K_b and its
    AMORTIZE budget == M*K_a; ``observed[arm] = {"discovery": n, "amortize": n}`` flips ``ok`` (and the
    run's ``same_budget_identity_holds``) if any arm exceeds its declared budget.  The retirement claim
    is the EQUAL-G claim: ST and every MA arm spend the same total (G_d one-time + M*K_a); G_d is
    reported separately with per-member share K_d/M -> 0."""
    G_d = K_d + K_b
    observed = observed or {}
    arms: list[TeamArmBudgetV1] = []
    all_ok = True
    for arm in ARM_SLATE:
        obs = observed.get(arm, {})
        act_d = int(obs.get("discovery", G_d))
        act_a = int(obs.get("amortize", M * K_a))
        ok = (act_d <= G_d) and (act_a <= M * K_a)
        all_ok = all_ok and ok
        arms.append(TeamArmBudgetV1(arm=arm, declared_discovery=G_d, actual_discovery=act_d,
                                    declared_amortize=M * K_a, actual_amortize=act_a, ok=ok))
    return TeamBudgetReportV1(
        M=M, K_a=K_a, K_d=K_d, K_b=K_b, G_d=G_d, arms=tuple(arms),
        same_budget_identity_holds=bool(all_ok),
        note=(f"equal-G: every arm (ST + all MA) spends one-time discovery G_d={G_d} (={K_d} candidate "
              f"+ {K_b} brute) + amortize M*K_a={M * K_a}; the MA ANALYZE call REPLACES one i.i.d. "
              f"candidate (no extra budget); K_d reported separately, per-member share K_d/M={round(K_d / M, 3) if M else 0}"))


# ==================================================================================================
# earn / load-bearing decision (RUNBOOK §8 + §14 DPI-band)
# ==================================================================================================
@dataclasses.dataclass(frozen=True)
class TeamEarnVerdictV1:
    earned: bool
    ma_minus_a1_pp: float
    ma_minus_b0_pp: float
    ma_minus_st_pp: float
    n_modes: int
    neg_le_b0: bool
    ma_gt_neg: bool
    diversity_real: bool
    dpi_band_ok: bool                 # ST disc-rate < 1 at the fragile budget (the §14 pre-condition)
    load_bearing: bool                # MA-ST>=+3.33 OR broader span OR ablation-collapse
    load_bearing_reason: str
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def apply_team_earn_gate_v1(*, ma_full_pp_over_a1: float, ma_full_pp_over_b0: float,
                            ma_minus_st_pp: float, n_modes: int, neg_le_b0: bool, ma_gt_neg: bool,
                            diversity_real: bool, st_disc_rate: float,
                            ablation_collapse: bool = False,
                            broader_span_than_st: bool = False,
                            margin_pp: float = 5.0, load_bearing_pp: float = 3.33,
                            min_modes: int = 2) -> TeamEarnVerdictV1:
    """The strict multi-agent earn gate (RUNBOOK §8).  A retirement-grade team result requires ALL of:
    MA-FULL beats A1 and B0 by >=margin_pp; spans >=min_modes; NEG<=B0 and MA>NEG; diversity REAL;
    the §14 DPI-band pre-condition (ST disc-rate < 1 at the fragile budget); AND team load-bearing via
    MA-ST>=load_bearing_pp OR broader span than ST OR an ablation collapse."""
    reasons: list[str] = []
    dpi_band_ok = st_disc_rate < 1.0
    if not dpi_band_ok:
        reasons.append(f"DPI-band FAIL: ST disc-rate {st_disc_rate:.2f} not < 1 (baseline does not fail to discover; "
                       "outside the fragile band a tie is the predicted null)")
    if ma_full_pp_over_a1 < margin_pp:
        reasons.append(f"MA-A1 {ma_full_pp_over_a1:+.1f}pp < {margin_pp}")
    if ma_full_pp_over_b0 < margin_pp:
        reasons.append(f"MA-B0 {ma_full_pp_over_b0:+.1f}pp < {margin_pp}")
    if n_modes < min_modes:
        reasons.append(f"span {n_modes} modes < {min_modes}")
    if not neg_le_b0:
        reasons.append("NEG > B0 (fake-different team lifts)")
    if not ma_gt_neg:
        reasons.append("MA not > NEG")
    if not diversity_real:
        reasons.append("diversity not REAL (fake-different)")
    lb_reason = []
    if ma_minus_st_pp >= load_bearing_pp:
        lb_reason.append(f"MA-ST {ma_minus_st_pp:+.1f}pp >= {load_bearing_pp}")
    if broader_span_than_st:
        lb_reason.append("MA spans broader than ST at same budget")
    if ablation_collapse:
        lb_reason.append("ablation collapse (removing RD/Q/SS destroys the earn)")
    load_bearing = bool(lb_reason)
    if not load_bearing:
        reasons.append("team NOT load-bearing (MA ties ST; no ablation collapse)")
    earned = (dpi_band_ok and ma_full_pp_over_a1 >= margin_pp and ma_full_pp_over_b0 >= margin_pp
              and n_modes >= min_modes and neg_le_b0 and ma_gt_neg and diversity_real and load_bearing)
    return TeamEarnVerdictV1(
        earned=earned, ma_minus_a1_pp=round(ma_full_pp_over_a1, 2), ma_minus_b0_pp=round(ma_full_pp_over_b0, 2),
        ma_minus_st_pp=round(ma_minus_st_pp, 2), n_modes=n_modes, neg_le_b0=neg_le_b0, ma_gt_neg=ma_gt_neg,
        diversity_real=diversity_real, dpi_band_ok=dpi_band_ok, load_bearing=load_bearing,
        load_bearing_reason=("; ".join(lb_reason) if lb_reason else "none"),
        reasons=tuple(reasons) if reasons else ("ALL_PASS",))


__all__ = [
    "MULTI_AGENT_DISCOVER_AMORTIZE_V1_SCHEMA_VERSION",
    "ROLE_STRATEGIST", "ROLE_IMPLEMENTER", "ROLE_BRUTE_AUTHOR", "ROLE_EXTRACTOR", "ROLE_AMORTIZER", "TEAM_ROLES",
    "TRANSFER_SHARED_STATE", "TRANSFER_TRANSCRIPT", "TRANSFER_EMPTY", "TRANSFER_NONE", "TRANSFER_MODES",
    "brute_prompt_default_v1", "brute_prompt_alt_v1",
    "TeamConfigV1", "arm_config", "ARM_SLATE", "TeamDiscoverResultV1", "team_discover_v1",
    "amortize_prompt_v1", "amortize_member_v1",
    "TeamArmBudgetV1", "TeamBudgetReportV1", "team_budget_parity_v1",
    "TeamEarnVerdictV1", "apply_team_earn_gate_v1",
]
