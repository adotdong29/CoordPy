"""W135 / COO-9 — oracle-derived solution-STRUCTURE witness on the resistant-by-construction
battlefield.

W133 turned the owned battlefield into a TEACHER via an exact-oracle witness, and found the sharp
split: the EW2 COMPLEXITY witness is real + load-bearing, but the EW1 COUNTEREXAMPLE witness adds
**+0.00 pp over blind reflexion** on the WRONG_ALGORITHM / SEARCH_ENUM modes (0 rescues vs B0) — the
**wrong-algorithm capability ceiling**. W134 then localised the complexity lever as single-family +
sub-oracle. So the live blocker is no longer complexity detection; it is wrong-algorithm /
solution-structure capability.

This module attacks that ceiling with **structure**, not just a counterexample. Because every minted
problem ships an exact ``ref_source`` answer-key AND an independent exhaustive ``brute_source``
oracle, we can compute — on a FRESH, disjoint, token-minimal counterexample ``X`` — a far richer
SANCTIONED feedback object than EW1's bare ``(X, expected, observed)`` triple:

* **SW1 — greedy-failure certificate** (primary; leakage-safest): the optimum ``V* = ref(X)``, the
  obvious/greedy value ``V_greedy = naive(X)`` (the template's canonical admissible-wrong rule), the
  objective gap, and an ATTRIBUTION line ("a local/greedy choice is dominated; a global method is
  required"). No solver source, no recurrence.
* **SW2 — optimal-substructure / recurrence witness** (exploratory; leak-constrained): a compact
  LADDER of optimal sub-values over OBVIOUS sub-instances of ``X`` (array → prefixes; single-int N →
  the exact-value sequence over small ``1..N``), each disjoint from the graded cases, each value from
  the owned oracle, plus a property hint ("optimal substructure; recover the recurrence"). Never the
  recurrence/state itself.
* **SW3 — search-frontier witness** (SEARCH_ENUM): the exact count vs the naive's wrong count, an
  ATTRIBUTION line (over/under-counts by Δ — wrong ordering/recurrence), and the exact-count SEQUENCE
  over the small disjoint ladder (the search-frontier structure exposing the wrong recurrence).
* **SW4 — structure-to-rewrite controller** (LEAD): the richest applicable union, with a shift-left
  edge-case enumeration step prepended; falls back to the EW1 counterexample if no structure is
  extractable, so it is never worse than C1.

Anti-cheat / no-leakage (LOCKED — see ``docs/RUNBOOK_W135.md`` §4):

* the structure witness is the ONLY sanctioned feedback object; ``ref_source`` / ``brute_source`` /
  ``naive_source`` are NEVER placed in any model-facing path — they are EXECUTED (subprocess) to
  derive OUTPUTS, never rendered as source (a structural test asserts ``to_prompt_block`` carries no
  ``def ``/``import ``/``class `` solver source);
* the counterexample input ``X`` AND every sub-instance in a ladder are asserted byte-disjoint from
  the graded ``secret_cases`` (``leakage_clean``); the recurrence/algorithm and the augmented DP
  state are NEVER stated (Pu, OOPSLA 2011: the state design is the human insight that would leak);
* grading is the audited ``grade_on_secret_v1`` on a DISJOINT hidden bank (pass iff ALL secret cases
  pass), so memorising the shown counterexample / small-N ladder CANNOT pass — the witness tests
  GENERALISATION, not memorisation.

Research-grounded FORM (primary sources, applied here): minimal-by-input-token + property phrasing
(PGS, arXiv:2506.18315), explicit ATTRIBUTION (loop-invariant repair TOPLAS 2025; Dolcetti
arXiv:2412.14841), shift-left edge-case enumeration (SolidCoder, arXiv:2604.19825), and SW2 treated
as exploratory + leak-constrained (Pu OOPSLA 2011; KNARsack arXiv:2509.15239 — recurrence induction
from a table needs a solver/training we lack).

The structure-witness arm (``run_structure_witness_arm_v1``) is a strict **same-budget** swap of the
W120 reflexion arm: identical K / model / temperature / attempt-0 prompt; the ONLY change is the
between-attempt feedback object. It is scored by the SAME audited grader + the SAME W108 evaluator
that scored W89 / W105 / W120 / W132 / W133 / W134, so "S - A1" is computed byte-identically to
"B - A1".

Reuses (explicit-import only, NO duplication): the exact-oracle witness probe + counterexample
search + shrink from ``exact_oracle_witness_v1``; the minted-problem record + subprocess from
``resistant_by_construction_battlefield_v1``; the pilot record, candidate extractor, secret grader,
and prompt scaffolds from ``icpc_reflexion_bench_v1``. Pure / deterministic except the
(already-audited) program-execution subprocess; NO model inference lives here (that is the W135 driver
script).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from typing import Any, Optional

from .resistant_by_construction_battlefield_v1 import (
    MintedProblemV1,
    MintedTemplateV1,
    _exec_capture_v1,
    _tok_count,
)
from .exact_oracle_witness_v1 import (
    WitnessV1,
    WitnessProbeSetV1,
    build_witness_probe_set_v1,
    find_counterexample_witness_v1,
)
from .icpc_reflexion_bench_v1 import (
    IcpcArmOutcomeV1,
    W120_ICPC_REFLEXION_BENCH_V1_SCHEMA_VERSION,
    _initial_prompt,
    _samples_block,
    extract_candidate_code_v1,
    grade_on_secret_v1,
    sample_feedback_v1,
)

W135_SOLUTION_STRUCTURE_WITNESS_V1_SCHEMA_VERSION: str = (
    "coordpy.solution_structure_witness_v1.v1")

# ---- structure-witness kinds -------------------------------------------------------
STRUCT_GREEDY_FAILURE: str = "GREEDY_FAILURE"           # SW1
STRUCT_OPTIMAL_SUBSTRUCTURE: str = "OPTIMAL_SUBSTRUCTURE"  # SW2
STRUCT_SEARCH_FRONTIER: str = "SEARCH_FRONTIER"        # SW3
STRUCT_NONE: str = "NONE"

# ---- structure-witness arms (the same-budget feedback policies) --------------------
ARM_S1_GREEDY: str = "S1"          # greedy-failure certificate
ARM_S2_SUBSTRUCTURE: str = "S2"    # optimal-substructure ladder (exploratory)
ARM_S3_SEARCH: str = "S3"          # search-frontier exact-count
ARM_S4_CONTROLLER: str = "S4"      # structure-to-rewrite controller (LEAD)
STRUCTURE_ARMS: tuple[str, ...] = (ARM_S1_GREEDY, ARM_S2_SUBSTRUCTURE, ARM_S3_SEARCH,
                                   ARM_S4_CONTROLLER)

# ---- probe / ladder sizing (LOCKED; the witness compute budget) --------------------
STRUCTURE_PROBE_TIMEOUT_S: float = 2.0   # the candidate finishes a small probe within this
ORACLE_SUBINSTANCE_TIMEOUT_S: float = 4.0  # ref/naive on a sub-instance
LADDER_MAX_RUNGS: int = 6                # at most this many sub-value rungs (bounded prompt)
LADDER_INT_CAP: int = 8                  # single-integer-N: reveal exact values over 1..min(N,cap)
MAX_VALUE_CHARS: int = 80                # truncate a rendered oracle value
SW5_MIN_DECISIONS: int = 200             # SW5 data floor (documented; not expected from one dev seed)


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


# ===================================================== the structure ladder + witness records

@dataclasses.dataclass(frozen=True)
class StructureLadderRungV1:
    """One sub-instance of the counterexample X with its oracle-optimal value.  ``summary`` is a
    short human label; ``optimal_value`` is the oracle OUTPUT (never the oracle program)."""

    size_tokens: int
    summary: str
    optimal_value: str
    leakage_clean: bool

    def to_dict(self) -> dict[str, Any]:
        return {"size_tokens": int(self.size_tokens), "summary": self.summary,
                "optimal_value_len": len(self.optimal_value),
                "leakage_clean": bool(self.leakage_clean)}


@dataclasses.dataclass(frozen=True)
class StructureWitnessV1:
    """An oracle-derived solution-structure witness, anchored on a token-minimal disjoint
    counterexample.  ``to_prompt_block(arm)`` is the ONLY model-facing text; it carries oracle
    OUTPUTS + derived structural summaries, never the oracle PROGRAM and never the recurrence."""

    kind: str                       # STRUCT_*
    sw_family: str                  # SW1 | SW2 | SW3 | NONE
    counterexample: WitnessV1       # the underlying EW1 counterexample (the anchor)
    optimal_value: str              # V* = ref(X)
    greedy_value: str               # V_greedy = naive(X) (the obvious/greedy rule's value); "" if NA
    objective_gap: str              # str(|V* - V_greedy|) when both are integer scalars; "" otherwise
    naive_overcounts: Optional[bool]  # SE: True if naive>exact, False if naive<exact, None if NA
    ladder: tuple[StructureLadderRungV1, ...]
    leakage_clean: bool

    def found(self) -> bool:
        return self.kind != STRUCT_NONE and self.counterexample.found()

    def has_attribution_contrast(self) -> bool:
        """True iff the witness carries a naive-vs-optimal scalar contrast beyond EW1's triple."""
        return bool(self.greedy_value != "" and self.greedy_value != self.optimal_value)

    def _shift_left_block(self) -> str:
        return (
            "First, BEFORE writing code, list 2-3 worst-case / adversarial inputs your previous "
            "approach would get wrong (boundary sizes, ties, all-equal, decreasing, the structure "
            "the counterexample below exposes). Then write a solution that handles them.\n")

    def to_prompt_block(self, arm: str) -> str:
        """The sanctioned feedback object shown to the model, rendered per arm.  Always anchored on
        the EW1 counterexample; the structure SUBSET differs by arm.  Never any solver source / no
        recurrence formula / no hidden-case answer."""
        if not self.found():
            return ("No counterexample was found and the program looked correct on small inputs; "
                    "re-examine the problem statement for an overlooked structural case.")
        ce = self.counterexample
        x = ce.probe_input.rstrip()
        vstar = self.optimal_value.rstrip()
        obs = (ce.observed_output.rstrip() if ce.observed_kind == "WRONG_ANSWER"
               else "(your program timed out)" if ce.observed_kind == "TIMEOUT"
               else "(your program crashed)")
        head = (
            "Exact structural diagnostic (the hidden tests are NOT shown, but this small input "
            "exposes the same structural bug; do NOT special-case it):\n"
            f"Input:\n{x}\n\nCorrect answer (from the reference oracle): {vstar}\n"
            f"Your program produced: {obs}\n")

        def _ladder_text(label: str) -> str:
            if len(self.ladder) < 2:
                return ""
            rows = "\n".join(f"  {r.summary} -> optimal {r.optimal_value}" for r in self.ladder)
            return (f"\n{label}\n{rows}\n")

        if arm == ARM_S1_GREEDY:
            gap = f" (a gap of {self.objective_gap})" if self.objective_gap else ""
            gline = (f"A natural locally-greedy / obvious approach gives {self.greedy_value} on this "
                     f"input{gap}, but the optimum is {vstar}. " if self.greedy_value else "")
            return (head + "\n" + gline +
                    "The locally-optimal choice is DOMINATED here: a correct solution must weigh "
                    "non-local combinations (a global optimisation), not commit to the greedy/local "
                    "choice. Rewrite with a method that considers the global structure.\n\n"
                    + self._shift_left_block() +
                    "Provide ONLY the corrected complete Python 3 program in a ```python ... ``` fence:")
        if arm == ARM_S2_SUBSTRUCTURE:
            ladder = _ladder_text("Optimal values on smaller sub-instances (the optimal "
                                  "substructure of THIS input):")
            return (head + ladder +
                    "\nNotice the optimum on each sub-instance reuses the optima of SMALLER "
                    "sub-instances (optimal substructure) — it is NOT obtained by extending a "
                    "greedy/local choice. Recover the recurrence that produces this table and "
                    "implement it (do not hard-code these values).\n\n"
                    + self._shift_left_block() +
                    "Provide ONLY the corrected complete Python 3 program in a ```python ... ``` fence:")
        if arm == ARM_S3_SEARCH:
            if self.naive_overcounts is True:
                dirn = f"OVER-counts by {self.objective_gap}" if self.objective_gap else "OVER-counts"
            elif self.naive_overcounts is False:
                dirn = f"UNDER-counts by {self.objective_gap}" if self.objective_gap else "UNDER-counts"
            else:
                dirn = "computes the wrong count"
            cline = (f"Your program's count ({self.greedy_value}) {dirn} versus the exact count "
                     f"({vstar}). " if self.greedy_value else "")
            ladder = _ladder_text("Exact counts on small instances (the true search frontier):")
            return (head + ladder + "\n" + cline +
                    "Your recurrence / counting assumption is wrong (e.g. ordered vs unordered, "
                    "reuse allowed vs not, or the wrong state). Re-derive the exact count from the "
                    "search structure above and implement that recurrence.\n\n"
                    + self._shift_left_block() +
                    "Provide ONLY the corrected complete Python 3 program in a ```python ... ``` fence:")
        # ARM_S4_CONTROLLER — the richest applicable union
        parts = [head]
        if self.has_attribution_contrast():
            if self.naive_overcounts is not None:
                dirn = ("OVER-counts" if self.naive_overcounts else "UNDER-counts")
                gp = f" by {self.objective_gap}" if self.objective_gap else ""
                parts.append(f"\nYour approach {dirn}{gp} here ({self.greedy_value}) versus the exact "
                             f"answer ({vstar}) — the counting recurrence/state is wrong.")
            else:
                gp = f" (gap {self.objective_gap})" if self.objective_gap else ""
                parts.append(f"\nA locally-greedy / obvious approach gives {self.greedy_value}{gp}; "
                             f"the optimum is {vstar} — the local choice is dominated.")
        parts.append(_ladder_text("Optimal values on smaller sub-instances (the structure to exploit):"))
        parts.append("\nUse this structure: the optimum reuses optima of smaller sub-instances "
                     "(optimal substructure) — find and implement the recurrence (do not hard-code "
                     "these values).\n\n" + self._shift_left_block() +
                     "Provide ONLY the corrected complete Python 3 program in a ```python ... ``` fence:")
        return "".join(p for p in parts if p)

    def cid(self) -> str:
        return _sha256_hex({"kind": "w135_structure_witness_v1", "sw_family": self.sw_family,
                            "struct_kind": self.kind,
                            "probe_input_sha256": _sha256_hex(self.counterexample.probe_input),
                            "optimal_value": self.optimal_value, "greedy_value": self.greedy_value,
                            "objective_gap": self.objective_gap,
                            "ladder": [r.summary + "=" + r.optimal_value for r in self.ladder]})

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "sw_family": self.sw_family,
                "found": self.found(), "counterexample": self.counterexample.to_dict(),
                "optimal_value_len": len(self.optimal_value),
                "greedy_value_len": len(self.greedy_value),
                "objective_gap": self.objective_gap,
                "naive_overcounts": self.naive_overcounts,
                "n_ladder_rungs": len(self.ladder), "ladder": [r.to_dict() for r in self.ladder],
                "has_attribution_contrast": self.has_attribution_contrast(),
                "leakage_clean": bool(self.leakage_clean),
                "structure_witness_cid": self.cid()}


def _none_structure(ce: Optional[WitnessV1] = None) -> StructureWitnessV1:
    from .exact_oracle_witness_v1 import _none_witness
    return StructureWitnessV1(
        kind=STRUCT_NONE, sw_family="NONE", counterexample=ce or _none_witness(),
        optimal_value="", greedy_value="", objective_gap="", naive_overcounts=None,
        ladder=(), leakage_clean=True)


# ===================================================== sub-instance ladder (the structure)

def _subinstances_v1(x: str) -> list[str]:
    """Generic OBVIOUS sub-instances of X (no clever/augmented state).

    * "<header...>\\n<space-separated array>" → prefixes of the array (with the count header fixed);
    * a single bare integer N on one line → the small ladder 1..min(N, LADDER_INT_CAP).

    Returns [] for shapes with no obvious parameterisation (grids / strings / multi-line) — the
    ladder is then empty and the witness is honestly not genuinely-new-by-ladder for that problem."""
    lines = x.split("\n")
    out: list[str] = []
    if len(lines) >= 2 and len(lines[-1].split()) >= 2:
        toks = lines[-1].split()
        head = lines[:-1]
        htoks = head[0].split() if head else []
        n = len(toks)
        sizes = sorted({k for k in (1, 2, n // 2, n - 1, n) if 1 <= k <= n})
        for k in sizes:
            sub = toks[:k]
            h = list(htoks)
            if h and h[0].isdigit() and int(h[0]) == n:
                h[0] = str(k)
            nh = ([" ".join(h)] + head[1:]) if head else []
            out.append("\n".join(nh + [" ".join(sub)]))
    elif len(lines) == 1 and lines[0].strip().isdigit():
        big_n = int(lines[0].strip())
        for k in range(1, min(big_n, LADDER_INT_CAP) + 1):
            out.append(str(k))
    return out


def _summarize_subinstance_v1(s: str) -> str:
    lines = s.split("\n")
    if len(lines) >= 2 and lines[-1].split():
        toks = lines[-1].split()
        preview = " ".join(toks[:6]) + (" ..." if len(toks) > 6 else "")
        return f"n={len(toks)}: [{preview}]"
    return f"N={lines[0].strip()}"


def _int_or_none(s: str) -> Optional[int]:
    t = s.strip()
    try:
        return int(t)
    except (TypeError, ValueError):
        return None


def _scalar_gap_v1(vstar: str, vgreedy: str) -> tuple[str, Optional[bool]]:
    """Return (gap_str, naive_overcounts) when both outputs are single integer scalars; else
    ("", None).  ``naive_overcounts`` is True iff vgreedy > vstar (the SEARCH_ENUM over-count signal)."""
    a, b = _int_or_none(vstar), _int_or_none(vgreedy)
    if a is None or b is None:
        return "", None
    return str(abs(a - b)), bool(b > a)


def _build_ladder_v1(x: str, template: MintedTemplateV1, secret_inputs: set,
                     *, timeout_s: float) -> tuple[StructureLadderRungV1, ...]:
    """Optimal sub-values over the OBVIOUS sub-instances of X (each disjoint from the graded bank,
    each value from the owned oracle)."""
    rungs: list[StructureLadderRungV1] = []
    seen: set = set()
    for s in _subinstances_v1(x):
        if s in seen:
            continue
        seen.add(s)
        if s in secret_inputs:          # leakage: never reveal an optimal value of a graded case
            continue
        r = _exec_capture_v1(template.ref_source, s, timeout_s=timeout_s)
        if r.timed_out or r.returncode != 0:
            continue
        rungs.append(StructureLadderRungV1(
            size_tokens=_tok_count(s), summary=_summarize_subinstance_v1(s),
            optimal_value=r.stdout.strip()[:MAX_VALUE_CHARS], leakage_clean=True))
        if len(rungs) >= LADDER_MAX_RUNGS:
            break
    return tuple(rungs)


# ===================================================== the structure-witness builder

def build_structure_witness_v1(code: str, problem: MintedProblemV1, probe: WitnessProbeSetV1,
                               template: MintedTemplateV1, *,
                               timeout_s: float = STRUCTURE_PROBE_TIMEOUT_S,
                               oracle_timeout_s: float = ORACLE_SUBINSTANCE_TIMEOUT_S,
                               ) -> StructureWitnessV1:
    """Build the oracle-derived structure witness for one candidate.

    Anchors on a token-minimal fresh disjoint counterexample (reusing EW1); if the candidate is
    value-correct on every small probe (no counterexample) the witness is NONE (structure has nothing
    to teach — correctly silent on a value-correct-but-slow program, the complexity negative control).
    Then computes V* = ref(X), V_greedy = naive(X), the objective gap / over-count signal, and the
    optimal-substructure ladder over disjoint sub-instances of X."""
    ce = find_counterexample_witness_v1(code, problem, probe, template, timeout_s=timeout_s)
    if not ce.found():
        return _none_structure(ce)

    secret_inputs = {inp for inp, _ in problem.secret_cases}
    x = ce.probe_input
    vstar = ce.expected_output            # = ref(X), already computed by EW1
    nr = _exec_capture_v1(template.naive_source, x, timeout_s=oracle_timeout_s)
    vgreedy = nr.stdout.strip()[:MAX_VALUE_CHARS] if (not nr.timed_out and nr.returncode == 0) else ""
    gap, overcounts = _scalar_gap_v1(vstar, vgreedy)
    ladder = _build_ladder_v1(x, template, secret_inputs, timeout_s=oracle_timeout_s)
    leak_clean = bool(ce.leakage_clean and all(r.leakage_clean for r in ladder))

    # kind classification (oracle-side route label; NOT shown to the model — the model sees only the
    # rendered block).  A genuine count over/under-count contrast => SEARCH_FRONTIER; a >=2-rung
    # ladder => OPTIMAL_SUBSTRUCTURE; a greedy scalar gap => GREEDY_FAILURE.
    has_contrast = bool(vgreedy != "" and vgreedy != vstar)
    if overcounts is not None and has_contrast:
        kind, fam = STRUCT_SEARCH_FRONTIER, "SW3"
    elif len(ladder) >= 2:
        kind, fam = STRUCT_OPTIMAL_SUBSTRUCTURE, "SW2"
    elif has_contrast:
        kind, fam = STRUCT_GREEDY_FAILURE, "SW1"
    else:
        # only the bare counterexample is available (no ladder, no scalar contrast) -> not
        # genuinely-new beyond EW1; report as GREEDY_FAILURE shell (genuinely_new test will fail it).
        kind, fam = STRUCT_GREEDY_FAILURE, "SW1"
    return StructureWitnessV1(
        kind=kind, sw_family=fam, counterexample=ce, optimal_value=vstar, greedy_value=vgreedy,
        objective_gap=gap, naive_overcounts=overcounts, ladder=ladder, leakage_clean=leak_clean)


def structure_witness_is_genuinely_new_v1(witness: StructureWitnessV1,
                                          problem: MintedProblemV1) -> dict[str, Any]:
    """Machine-checkable 'not just a counterexample / not judge-bit-plus-words' test.  A structure
    witness is genuinely NEW iff it carries information EW1's bare ``(X, expected, observed)`` triple
    structurally lacks: a >=2-rung optimal sub-value LADDER, OR a naive-vs-optimal scalar ATTRIBUTION
    contrast — AND it is leakage-clean (X + every sub-instance disjoint from the graded bank)."""
    sample_inputs = {inp for inp, _ in problem.samples}
    found = witness.found()
    ce = witness.counterexample
    is_new_input = bool(found and ce.probe_input not in sample_inputs)
    has_ladder = len(witness.ladder) >= 2
    obs = ce.observed_output.strip()[:MAX_VALUE_CHARS]
    # The canonical-greedy value is genuinely-new info ONLY iff it differs from BOTH the optimum AND
    # what the candidate produced — otherwise it is merely EW1's ``observed`` restated (which is the
    # case in the naive/ref self-test, where the candidate IS the canonical naive, so genuinely-new
    # there rests on the LADDER; at bench the candidate is the model, usually != the canonical greedy).
    greedy_is_new = bool(witness.greedy_value != "" and witness.greedy_value != witness.optimal_value
                         and witness.greedy_value != obs)
    carries_structure = bool(has_ladder or greedy_is_new)
    genuinely_new = bool(found and is_new_input and carries_structure and witness.leakage_clean)
    return {"genuinely_new": genuinely_new, "found": found,
            "input_not_a_public_sample": is_new_input,
            "has_substructure_ladder": has_ladder, "greedy_value_is_new_datapoint": greedy_is_new,
            "has_attribution_contrast": witness.has_attribution_contrast(),
            "n_ladder_rungs": len(witness.ladder),
            "leakage_clean": bool(witness.leakage_clean),
            "kind": witness.kind, "sw_family": witness.sw_family}


# ===================================================== structure-reflexion prompt + arm

def _structure_reflexion_prompt(problem, history, witness: StructureWitnessV1, arm: str,
                                attempt_idx: int) -> str:
    """Between-attempt prompt: the SAME scaffold as the W120 reflexion prompt (judge bit + stderr +
    public-sample results) PLUS the oracle-derived STRUCTURE block — a strict superset of the blind
    feedback AND of EW1's counterexample, so any gain is attributable to the structure."""
    chunks: list[str] = []
    for i, (cand, passed, stderr_tail, sample_fb) in enumerate(history):
        cand_trim = cand if len(cand) <= 1500 else (cand[:1500] + "\n# ...(truncated)\n")
        verdict = ("ACCEPTED by the judge (all hidden tests passed)" if passed
                   else "REJECTED by the judge (failed at least one hidden test)")
        se = f"\nExecutor stderr (tail):\n{stderr_tail.strip()}" if stderr_tail.strip() else ""
        sf = f"\nPublic sample results:\n{sample_fb}" if sample_fb.strip() else ""
        chunks.append(f"--- Attempt {i+1} ({verdict}) ---\n"
                      f"```python\n{cand_trim}\n```{se}{sf}")
    wblock = witness.to_prompt_block(arm)
    return (
        "You are an expert ICPC competitor on a reflective debugging loop. You are on "
        f"attempt {attempt_idx + 1} out of 5. Below are your previous attempts with the judge "
        "verdict and the PUBLIC sample-case results, followed by an EXACT structural diagnostic of "
        "the last attempt's failure derived from a reference oracle. Use the structure to produce a "
        "NEW corrected COMPLETE Python 3 stdin/stdout program. Do not repeat a previous attempt "
        "verbatim and do not hard-code the shown values.\n\n"
        f"Problem:\n{problem.statement}\n\n"
        f"{_samples_block(problem)}\n\n"
        f"{chr(10).join(chunks)}\n\n"
        f"=== Exact structural diagnostic of attempt {len(history)} ===\n{wblock}")


@dataclasses.dataclass(frozen=True)
class StructureArmTraceV1:
    """Per-problem audit of one structure-witness arm: which structure fired on each invoked attempt."""
    problem_id: str
    arm_id: str
    struct_kinds: tuple[str, ...]
    sw_families: tuple[str, ...]
    any_structure_found: bool
    any_genuinely_new: bool
    all_leakage_clean: bool
    max_ladder_rungs: int

    def rescue_is_structural(self) -> bool:
        """True iff any invoked attempt fired a genuine structure witness (a ladder or a scalar
        attribution contrast) — i.e. the arm's effect is NOT trivial parse/format repair."""
        return bool(self.any_genuinely_new)

    def to_dict(self) -> dict[str, Any]:
        return {"problem_id": self.problem_id, "arm_id": self.arm_id,
                "struct_kinds": list(self.struct_kinds), "sw_families": list(self.sw_families),
                "any_structure_found": bool(self.any_structure_found),
                "any_genuinely_new": bool(self.any_genuinely_new),
                "all_leakage_clean": bool(self.all_leakage_clean),
                "max_ladder_rungs": int(self.max_ladder_rungs)}


def run_structure_witness_arm_v1(*, seed: int, template: MintedTemplateV1, problem: MintedProblemV1,
                                 probe: WitnessProbeSetV1, gen, K: int, temperature: float,
                                 max_tokens: int, timeout_s: float, arm: str, minted_date: str,
                                 witness_timeout_s: float = STRUCTURE_PROBE_TIMEOUT_S,
                                 oracle_timeout_s: float = ORACLE_SUBINSTANCE_TIMEOUT_S,
                                 ) -> tuple[IcpcArmOutcomeV1, StructureArmTraceV1]:
    """Same-budget structure-witness reflexion arm.  Byte-identical structure to the W120 ``_run_b``
    (attempt-0 = the standard initial prompt; K attempts; one model call per attempt; no early stop),
    except the between-attempt feedback object is the oracle-derived structure witness.

    The model is graded ONLY on ``problem.secret_cases`` (the audited grader); the witness is computed
    from a FRESH disjoint probe set — never the graded cases."""
    pilot = problem.to_pilot_problem(minted_date=str(minted_date))
    history: list[tuple[str, bool, str, str]] = []
    per_call: list[bool] = []
    first_pass_idx = -1
    skinds: list[str] = []
    sfams: list[str] = []
    any_new = False
    leak_clean = True
    max_rungs = 0
    for k in range(int(K)):
        if k == 0:
            prompt = _initial_prompt(pilot)
        else:
            last_code = history[-1][0]
            witness = build_structure_witness_v1(last_code, problem, probe, template,
                                                 timeout_s=witness_timeout_s,
                                                 oracle_timeout_s=oracle_timeout_s)
            skinds.append(witness.kind)
            sfams.append(witness.sw_family)
            gn = structure_witness_is_genuinely_new_v1(witness, problem)
            any_new = any_new or bool(gn["genuinely_new"])
            max_rungs = max(max_rungs, len(witness.ladder))
            if witness.found() and not witness.leakage_clean:
                leak_clean = False
            prompt = _structure_reflexion_prompt(pilot, tuple(history), witness, arm, attempt_idx=k)
        text, _ = gen(prompt, max_tokens, float(temperature))
        code = extract_candidate_code_v1(response_text=text)
        passed, stderr_tail, _ = grade_on_secret_v1(pilot, code, timeout_s=timeout_s)
        per_call.append(bool(passed))
        sfb = sample_feedback_v1(pilot, code, timeout_s=timeout_s)
        history.append((code, bool(passed), stderr_tail, sfb))
        if passed and first_pass_idx == -1:
            first_pass_idx = int(k)
    outcome = IcpcArmOutcomeV1(
        schema=W120_ICPC_REFLEXION_BENCH_V1_SCHEMA_VERSION, seed=int(seed),
        question_id=problem.problem_id, arm_id=str(arm),
        final_passed=bool(first_pass_idx >= 0), n_model_calls=int(K),
        per_call_passed=tuple(per_call), first_pass_attempt_idx=int(first_pass_idx))
    trace = StructureArmTraceV1(
        problem_id=problem.problem_id, arm_id=str(arm),
        struct_kinds=tuple(skinds), sw_families=tuple(sfams),
        any_structure_found=any(kk != STRUCT_NONE for kk in skinds),
        any_genuinely_new=bool(any_new), all_leakage_clean=bool(leak_clean),
        max_ladder_rungs=int(max_rungs))
    return outcome, trace


__all__ = [
    "W135_SOLUTION_STRUCTURE_WITNESS_V1_SCHEMA_VERSION",
    "STRUCT_GREEDY_FAILURE", "STRUCT_OPTIMAL_SUBSTRUCTURE", "STRUCT_SEARCH_FRONTIER", "STRUCT_NONE",
    "ARM_S1_GREEDY", "ARM_S2_SUBSTRUCTURE", "ARM_S3_SEARCH", "ARM_S4_CONTROLLER", "STRUCTURE_ARMS",
    "LADDER_MAX_RUNGS", "LADDER_INT_CAP", "SW5_MIN_DECISIONS",
    "StructureLadderRungV1", "StructureWitnessV1",
    "build_structure_witness_v1", "structure_witness_is_genuinely_new_v1",
    "StructureArmTraceV1", "run_structure_witness_arm_v1",
]
