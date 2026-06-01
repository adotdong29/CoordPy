"""W128 Lane α — Role-Diverse Algorithm Search V1 (COO-9 sibling).

The honest different-mechanism answer to the W127 ``RESISTANT_SCAFFOLD_FRESH_GEN_CAP``.
W127 proved that a family-SCAFFOLD line (retrieve a structural skeleton -> generate K i.i.d.
candidates) earns weakly on the EXPOSED dev bench yet creates **0** new solves on the
non-memorizable resistant field.  W126 proved $0 deterministic synthesis is dead (oracle
ceiling 0/22).  W125 proved $0 re-routing of the existing pool is dead
(``blind_selection_headroom = 0``).  The remaining honest lever for the atlas's
**non-scaffoldable** clusters (``graph_flow`` / ``simulation_grid`` — diverse wrong-algorithm
failures that no reusable skeleton covers) is NOT a better template: it is a **role-diverse
algorithm SEARCH** that turns single-shot generation into generate -> verify -> select ->
abstain, at the SAME K=5 model-call budget as the plain baseline.

This module is held OUTSIDE the stable SDK contract: explicit-import-only,
``coordpy/__init__.py`` untouched, no version bump, no PyPI.

Mechanism (one target, K model calls, matched to baseline A1's K=5):

* **Call 1 — ANALYZE (RDA1 fused roles).**  One structured call returns *materially different
  intermediate artifacts*: a SPEC (restated I/O contract + constraints), 2-4 checkable
  INVARIANTS, a COMPLEXITY budget, ``n_sketches`` **distinct algorithm sketches** (named
  approach + outline), and 3-6 DERIVED counterexamples (new edge-case inputs the public
  samples do NOT cover, optionally with the model's predicted expected output).
* **Calls 2..K — IMPLEMENT (implementer role).**  One implementation per sketch, conditioned
  on the SPEC + that ONE sketch (low temperature: the diversity comes from the *sketch*, not
  from sampling).

Then a fully **NIM-free** generate->verify->select->abstain over the implementations:

* **RDA1** naive: commit the first implementation that passes all PUBLIC samples.
* **RDA2** counterexample-guided (executor-grounded via ``parse_failure_digest_v1``): run the
  public-survivors on the DERIVED counterexamples, group by behaviour signature, commit a
  representative of the majority (agreement) class — eliminating outliers.
* **RDA3** role-invariant ABSTAIN (REAL bridge to ``role_invariant_synthesis``): feed the
  survivor agreement to ``select_role_invariance_decision``; if the survivors irreconcilably
  diverge with no strict-majority quorum -> ``INVARIANCE_DIVERGED_ABSTAINED`` -> ABSTAIN
  rather than commit a coin-flip.
* **RDA4** two-axis consensus + fallback (REAL bridge to ``integrated_synthesis``): combine
  the producer axis (impl-consensus) with a trust axis (the candidate matching the model's
  predicted-expected on the derived counterexamples) via
  ``select_integrated_synthesis_decision``; commit on both-axes agreement, abstain on
  divergence, producer-only fallback when no predicted-expected is available.

Honest mining note (RDA4 — which candidate died and why): the operator-named W79 substrate
controllers (``team_consensus_controller_v14`` / ``consensus_fallback_controller_v25`` /
``hosted_cost_planner_v12`` / ``hosted_real_handoff_coordinator_v11``) were examined; their
decision logic is parameterised over substrate-trust quantities
(``replacement_then_restart_after_long_delay`` pressure / trust floors), NOT over
code-candidate consensus.  A *literal* bridge would require fabricating substrate inputs =
fake-different.  So the consensus/abstain role is filled by the W41/W42 synthesis decisions,
which ARE the closed-form set-consensus / abstain-on-divergence primitives and ARE
semantically aligned — see ``examine_substrate_controller_applicability_v1``.

A structural **fake-diversity** detector (``DiversityReportV1.classify``) and a synthetic
``fake_diversity_control_v1`` positive control (identical sketches MUST classify
``FAKE_DIVERSE``) give the mechanism the same NIM-free realness surface W125's
``MechanismFingerprintV1`` gave the controller line.

No leakage: every prompt contains ONLY the target's PUBLIC statement + PUBLIC samples; the
target's accepted solution / secret cases are NEVER shown.  Candidates are graded on the
official secret cases (the EVAL), and checked against the accepted solution ONLY by the W126/
W127 leakage guards (the accepted text is a tripwire, never an input).
"""
from __future__ import annotations

import ast as _ast
import dataclasses
import hashlib
import json
import re
import subprocess
import sys
from typing import Any, Callable, Optional, Sequence

from .coordpy_icpc_battlefield_v1 import KIND_PASSFAIL, grade_icpc_candidate_case_v1
from .icpc_reflexion_bench_v1 import (
    IcpcPilotProblemV1, extract_candidate_code_v1, grade_on_secret_v1)
from .executor_grounded_patcher_v1 import FailureDigestV1, parse_failure_digest_v1
from .resistant_capability_atlas_v1 import classify_family_v1
# --- REAL synthesis bridge (RDA3 / RDA4 abstain-on-divergence) ---
from .role_invariant_synthesis import (  # noqa: F401  (re-export for the bridge edge)
    W42_BRANCH_INVARIANCE_DIVERGED_ABSTAINED, W42_BRANCH_INVARIANCE_RATIFIED,
    select_role_invariance_decision)
from .integrated_synthesis import (  # noqa: F401
    W41_BRANCH_INTEGRATED_AXES_DIVERGED_ABSTAINED, W41_BRANCH_INTEGRATED_BOTH_AXES,
    classify_producer_axis_branch, classify_trust_axis_branch,
    select_integrated_synthesis_decision)

# (prompt, max_tokens, temperature) -> (text, wall_ms)
GenFn = Callable[[str, int, float], Any]

RDA_VARIANTS: tuple[str, ...] = ("RDA1", "RDA2", "RDA3", "RDA4")
DEFAULT_N_SKETCHES = 4
SKETCH_JACCARD_FAKE_THRESHOLD = 0.80  # any sketch pair more similar than this => not diverse
MAX_DERIVED_CASES = 6
DERIVED_CASE_TIMEOUT_S = 5.0
# tolerant of markdown headers (### / ####), bold (**), and leading list markers — the model
# routinely emits "#### SKETCH A:" / "**SKETCH A**" rather than a bare "SKETCH A:".
_SECTION_RE = re.compile(r"(?im)^\s*#{0,6}\s*\*{0,3}\s*(SPEC|INVARIANTS?|COMPLEXITY|"
                         r"SKETCHES|SKETCH[_\s]?([A-D1-4])|COUNTEREXAMPLES?)\b[:\-\s*]*")


# =============================================================================
# helpers
# =============================================================================
def _canon(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      default=str).encode("utf-8")


def _sha(payload: Any) -> str:
    return hashlib.sha256(_canon(payload)).hexdigest()


def _parses(code: str) -> bool:
    if not code or not code.strip():
        return False
    try:
        _ast.parse(code)
        return True
    except SyntaxError:
        return False


def _norm_code(code: str) -> str:
    """Normalize for distinctness comparison: strip comments / docstrings / blank lines,
    collapse whitespace.  AST-unparse when it parses (so cosmetic reformatting collapses)."""
    if _parses(code):
        try:
            tree = _ast.parse(code)
            for node in _ast.walk(tree):
                if (isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef,
                                      _ast.ClassDef, _ast.Module))
                        and node.body and isinstance(node.body[0], _ast.Expr)
                        and isinstance(getattr(node.body[0], "value", None), _ast.Constant)
                        and isinstance(node.body[0].value.value, str)):
                    node.body = node.body[1:] or [_ast.Pass()]
            return _ast.unparse(tree)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass
    lines = [ln.split("#", 1)[0].rstrip() for ln in code.splitlines()]
    return "\n".join(ln for ln in lines if ln.strip())


def _token_set(text: str) -> set:
    return set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]+", (text or "").lower()))


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    u = a | b
    return (len(a & b) / len(u)) if u else 0.0


def _norm_out(s: str) -> str:
    return " ".join((s or "").split())


def _run_capture_stdout_v1(code: str, stdin_text: str, *,
                           timeout_s: float = DERIVED_CASE_TIMEOUT_S,
                           python_exe: Optional[str] = None
                           ) -> tuple[str, Optional[FailureDigestV1]]:
    """Run a candidate on one stdin and capture a normalized stdout signature.

    Mirrors ``grade_icpc_candidate_case_v1``'s execution model EXACTLY (isolated
    ``python -I -c`` subprocess, captured output, hard timeout) — the repo's only code path —
    but returns the stdout SIGNATURE (for candidate-vs-candidate agreement on the model's
    DERIVED counterexamples, which have no ground truth).  Errors/timeouts become a typed
    sentinel + an executor-grounded ``FailureDigestV1`` (RDA2)."""
    if not _parses(code):
        return "<PARSE_ERR>", None
    py = python_exe or sys.executable
    try:
        proc = subprocess.run([py, "-I", "-c", code],
                              input=stdin_text.encode("utf-8"),
                              capture_output=True, timeout=float(timeout_s), check=False)
    except subprocess.TimeoutExpired:
        return "<TIMEOUT>", parse_failure_digest_v1(stderr_tail="", timed_out=True)
    rc = int(proc.returncode)
    if rc != 0:
        tail = proc.stderr.decode("utf-8", "replace")[-300:]
        return f"<ERR:{rc}>", parse_failure_digest_v1(stderr_tail=tail, timed_out=False)
    return _norm_out(proc.stdout.decode("utf-8", "replace")), None


# =============================================================================
# role artifacts
# =============================================================================
@dataclasses.dataclass(frozen=True)
class SketchV1:
    label: str
    approach_name: str
    outline: str

    def to_dict(self) -> dict:
        return {"label": self.label, "approach_name": self.approach_name,
                "outline": self.outline}


@dataclasses.dataclass(frozen=True)
class RoleArtifactsV1:
    spec: str
    invariants: tuple[str, ...]
    complexity: str
    sketches: tuple[SketchV1, ...]
    counterexamples: tuple[tuple[str, Optional[str]], ...]  # (stdin, predicted_expected|None)
    raw: str

    def to_dict(self) -> dict:
        return {"spec_len": len(self.spec), "n_invariants": len(self.invariants),
                "complexity": self.complexity[:120],
                "sketches": [s.to_dict() for s in self.sketches],
                "n_counterexamples": len(self.counterexamples),
                "n_counterexamples_with_expected": sum(
                    1 for _i, e in self.counterexamples if e)}


@dataclasses.dataclass(frozen=True)
class CandidateImplV1:
    label: str
    code: str
    parses: bool


# =============================================================================
# prompts + parsing
# =============================================================================
def build_analyze_prompt_v1(problem: IcpcPilotProblemV1, *,
                            n_sketches: int = DEFAULT_N_SKETCHES) -> str:
    samples = "\n".join(
        f"--- sample {i + 1} ---\nINPUT:\n{inp}\nOUTPUT:\n{exp}"
        for i, (inp, exp) in enumerate(problem.samples))
    labels = ", ".join(chr(ord("A") + i) for i in range(n_sketches))
    return (
        "You are a competitive-programming ANALYSIS team. Do NOT write the final solution "
        "yet. For the problem below, produce these labelled sections EXACTLY:\n\n"
        "SPEC:\n<restate the input/output contract and the numeric constraints/bounds>\n\n"
        "INVARIANTS:\n<2-4 short checkable properties any CORRECT output must satisfy, one "
        "per line, prefixed '- '>\n\n"
        "COMPLEXITY:\n<the worst-case input size and the target time complexity; state "
        "whether brute force fits>\n\n"
        f"SKETCHES:\nGive {n_sketches} MATERIALLY DIFFERENT algorithmic approaches labelled "
        f"{labels}. Each MUST use a genuinely different algorithm/idea (NOT the same approach "
        "reworded). Format each as 'SKETCH X: <approach name>' then 3-6 outline steps. If you "
        "cannot find that many distinct ideas, still give your best distinct alternatives.\n\n"
        "COUNTEREXAMPLES:\nGive 3-6 SMALL edge-case INPUTS (within the stated constraints) "
        "that the provided samples do NOT cover and that would distinguish a correct solution "
        "from a plausible-but-wrong one. Format each as 'CASE:' then the raw stdin on the "
        "following lines, then optionally 'EXPECT:' and the correct stdout if you are certain. "
        "Separate cases with a line containing only '==='.\n\n"
        f"PROBLEM (statement):\n{problem.statement}\n\nPUBLIC SAMPLES:\n{samples}\n")


def build_implement_prompt_v1(problem: IcpcPilotProblemV1, spec: str,
                              sketch: SketchV1) -> str:
    samples = "\n".join(
        f"INPUT:\n{inp}\nOUTPUT:\n{exp}" for inp, exp in problem.samples)
    return (
        "Implement a COMPLETE Python 3 program (read stdin, print to stdout) for the problem "
        "below, following EXACTLY this approach (do not switch to a different algorithm):\n\n"
        f"APPROACH ({sketch.approach_name}):\n{sketch.outline}\n\n"
        f"SPEC:\n{spec}\n\n"
        "Output ONLY one fenced ```python code block. Read all of stdin, print the answer.\n\n"
        f"PROBLEM:\n{problem.statement}\n\nSAMPLES:\n{samples}\n")


def _slice_sections(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    matches = list(_SECTION_RE.finditer(text or ""))
    for i, m in enumerate(matches):
        key = m.group(1).upper().replace(" ", "_")
        if key.startswith("INVARIANT"):
            key = "INVARIANTS"
        elif key.startswith("COUNTEREXAMPLE"):
            key = "COUNTEREXAMPLES"
        elif key.startswith("SKETCH"):
            lab = (m.group(2) or "").upper()
            key = ("SKETCH_" + lab) if lab else "SKETCHES"
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        out[key] = text[m.end():end].strip()
    return out


def parse_role_artifacts_v1(text: str, *,
                            n_sketches: int = DEFAULT_N_SKETCHES) -> RoleArtifactsV1:
    """Tolerant parser for the ANALYZE call output."""
    sec = _slice_sections(text or "")
    spec = sec.get("SPEC", "")
    complexity = sec.get("COMPLEXITY", "")
    invariants = tuple(
        ln.strip(" -*\t").strip()
        for ln in sec.get("INVARIANTS", "").splitlines()
        if ln.strip(" -*\t").strip())[:6]

    # sketches: prefer per-label sections, else split the SKETCHES blob on 'SKETCH X:'
    sketches: list[SketchV1] = []
    for i in range(n_sketches):
        lab = chr(ord("A") + i)
        body = sec.get(f"SKETCH_{lab}", "")
        if body:
            first, _, rest = body.partition("\n")
            sketches.append(SketchV1(lab, first.strip()[:120] or f"sketch {lab}",
                                     (first + "\n" + rest).strip()))
    if not sketches:
        blob = sec.get("SKETCHES", text or "")
        parts = re.split(r"(?im)^[\s#*>_]*SKETCH\s*([A-D1-4])\b\s*[:\-)\.]*", blob)
        # parts = [pre, label, body, label, body, ...]
        for j in range(1, len(parts) - 1, 2):
            lab = parts[j].upper()
            body = parts[j + 1].strip()
            name = body.split("\n", 1)[0].strip()[:120]
            sketches.append(SketchV1(lab, name or f"sketch {lab}", body))
    sketches = sketches[:n_sketches]

    # counterexamples: split on '===' inside COUNTEREXAMPLES; each has CASE/EXPECT
    cxs: list[tuple[str, Optional[str]]] = []
    cx_blob = sec.get("COUNTEREXAMPLES", "")
    for chunk in re.split(r"(?m)^\s*===\s*$", cx_blob):
        if not chunk.strip():
            continue
        cm = re.search(r"(?is)CASE\s*:?\s*(.*?)(?:EXPECT\s*:?\s*(.*))?$", chunk)
        if not cm:
            continue
        stdin_text = (cm.group(1) or "").strip("\n")
        expect = (cm.group(2) or "").strip() or None
        if stdin_text.strip():
            cxs.append((stdin_text, expect))
    return RoleArtifactsV1(spec=spec, invariants=invariants, complexity=complexity,
                           sketches=tuple(sketches), counterexamples=tuple(cxs[:MAX_DERIVED_CASES]),
                           raw=text or "")


# =============================================================================
# diversity / fake-diversity detector  (NIM-free realness surface)
# =============================================================================
@dataclasses.dataclass(frozen=True)
class DiversityReportV1:
    n_sketches: int
    max_sketch_jaccard: float
    n_distinct_impls: int
    n_impls: int
    counterexamples_new: bool
    invariants_nonempty: bool

    @property
    def diversity_real(self) -> bool:
        return (self.n_sketches >= 2
                and self.max_sketch_jaccard < SKETCH_JACCARD_FAKE_THRESHOLD
                and self.n_distinct_impls >= 2
                and self.counterexamples_new
                and self.invariants_nonempty)

    def classify(self) -> str:
        return "REAL" if self.diversity_real else "FAKE_DIVERSE"

    def to_dict(self) -> dict:
        return {"n_sketches": self.n_sketches,
                "max_sketch_jaccard": round(self.max_sketch_jaccard, 4),
                "n_distinct_impls": self.n_distinct_impls, "n_impls": self.n_impls,
                "counterexamples_new": self.counterexamples_new,
                "invariants_nonempty": self.invariants_nonempty,
                "diversity_real": self.diversity_real, "classify": self.classify()}


def compute_diversity_v1(artifacts: RoleArtifactsV1, impls: Sequence[CandidateImplV1],
                         sample_inputs: Sequence[str]) -> DiversityReportV1:
    sk = artifacts.sketches
    max_j = 0.0
    for i in range(len(sk)):
        for j in range(i + 1, len(sk)):
            max_j = max(max_j, _jaccard(_token_set(sk[i].outline),
                                        _token_set(sk[j].outline)))
    distinct = {_norm_code(im.code) for im in impls if im.parses}
    sample_norm = {_norm_out(s) for s in sample_inputs}
    cx_new = any(_norm_out(inp) not in sample_norm
                 for inp, _e in artifacts.counterexamples)
    return DiversityReportV1(
        n_sketches=len(sk), max_sketch_jaccard=max_j,
        n_distinct_impls=len(distinct), n_impls=sum(1 for im in impls if im.parses),
        counterexamples_new=bool(cx_new and artifacts.counterexamples),
        invariants_nonempty=bool(artifacts.invariants))


def fake_diversity_control_v1() -> DiversityReportV1:
    """Synthetic POSITIVE CONTROL: identical sketches + samples-as-counterexamples MUST
    classify ``FAKE_DIVERSE`` (proves the detector bites — the W125 C0 analogue)."""
    same = "do a brute force loop over all n items and print the count"
    arts = RoleArtifactsV1(
        spec="s", invariants=("x>=0",), complexity="O(n)",
        sketches=(SketchV1("A", "brute force", same), SketchV1("B", "brute force", same)),
        counterexamples=(("1\n", None),), raw="")
    impls = (CandidateImplV1("A", "print(1)", True), CandidateImplV1("B", "print(1)", True))
    return compute_diversity_v1(arts, impls, sample_inputs=["1\n"])


# =============================================================================
# selection  (RDA1..RDA4)  — all NIM-free over the SAME generations
# =============================================================================
@dataclasses.dataclass(frozen=True)
class RdaSelectionV1:
    variant: str
    committed_label: Optional[str]
    committed_code: Optional[str]
    abstained: bool
    branch: str
    n_public_survivors: int
    consensus_fraction: float

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["committed_code"] = bool(self.committed_code)  # don't serialize code blob
        d["committed_code_sha"] = (_sha(self.committed_code)[:16]
                                   if self.committed_code else None)
        return d


def _public_survivors(problem: IcpcPilotProblemV1, impls: Sequence[CandidateImplV1], *,
                      timeout_s: float) -> list[CandidateImplV1]:
    out = []
    for im in impls:
        if not im.parses:
            continue
        ok = True
        for inp, exp in problem.samples:
            r = grade_icpc_candidate_case_v1(
                candidate_code=im.code, stdin_text=inp, expected_stdout=exp,
                kind=problem.kind, float_tol=problem.float_tol, timeout_s=timeout_s)
            if not r.passed:
                ok = False
                break
        if ok:
            out.append(im)
    return out


def _behavior_signatures(survivors: Sequence[CandidateImplV1],
                         derived_inputs: Sequence[str], *,
                         timeout_s: float) -> dict[str, tuple[str, ...]]:
    sigs: dict[str, tuple[str, ...]] = {}
    for im in survivors:
        vec = []
        for inp in derived_inputs:
            out, _dig = _run_capture_stdout_v1(im.code, inp, timeout_s=timeout_s)
            vec.append(out)
        sigs[im.label] = tuple(vec)
    return sigs


def select_committed_v1(problem: IcpcPilotProblemV1, impls: Sequence[CandidateImplV1],
                        artifacts: RoleArtifactsV1, *, variant: str,
                        timeout_s: float = DERIVED_CASE_TIMEOUT_S) -> RdaSelectionV1:
    survivors = _public_survivors(problem, impls, timeout_s=timeout_s)
    ns = len(survivors)
    if ns == 0:
        # RDA1 falls back to first-parsing impl; RDA2+ abstain (no public-validated candidate)
        if variant == "RDA1":
            first = next((im for im in impls if im.parses), None)
            return RdaSelectionV1(variant, first.label if first else None,
                                  first.code if first else None, first is None,
                                  "NO_PUBLIC_SURVIVOR_FALLBACK", 0, 0.0)
        return RdaSelectionV1(variant, None, None, True, "NO_PUBLIC_SURVIVOR_ABSTAIN", 0, 0.0)
    if variant == "RDA1" or ns == 1:
        return RdaSelectionV1(variant, survivors[0].label, survivors[0].code, False,
                              "SINGLE_OR_NAIVE", ns, 1.0 / ns)

    derived = [inp for inp, _e in artifacts.counterexamples]
    if not derived:  # no discriminating cases -> RDA2+ reduce to first survivor
        return RdaSelectionV1(variant, survivors[0].label, survivors[0].code, False,
                              "NO_DERIVED_CASES", ns, 1.0 / ns)
    sigs = _behavior_signatures(survivors, derived, timeout_s=timeout_s)
    # group by behaviour signature (RDA2 agreement classes)
    classes: dict[tuple[str, ...], list[str]] = {}
    for lab, sig in sigs.items():
        classes.setdefault(sig, []).append(lab)
    modal_sig, modal_labels = max(classes.items(), key=lambda kv: (len(kv[1]), -ord(kv[1][0][0]) if kv[1] else 0))
    consensus_fraction = len(modal_labels) / ns
    rep = next(im for im in survivors if im.label == modal_labels[0])

    if variant == "RDA2":
        return RdaSelectionV1(variant, rep.label, rep.code, False,
                              f"MAJORITY_{len(modal_labels)}_OF_{ns}", ns, consensus_fraction)

    # RDA3 — role-invariant abstain-on-divergence (REAL bridge)
    branch_ri, _out, score = select_role_invariance_decision(
        integrated_services=[f"sig::{lab}" for lab in modal_labels],
        expected_services=[f"sig::{lab}" for lab in sigs],
        policy_match_found=True)
    quorum = consensus_fraction > 0.5  # strict majority
    if branch_ri == W42_BRANCH_INVARIANCE_RATIFIED or quorum:
        if variant == "RDA3":
            return RdaSelectionV1(variant, rep.label, rep.code, False,
                                  "RI_RATIFIED" if branch_ri == W42_BRANCH_INVARIANCE_RATIFIED
                                  else "RI_QUORUM", ns, consensus_fraction)
    elif variant == "RDA3":
        return RdaSelectionV1(variant, None, None, True,
                              "RI_DIVERGED_ABSTAINED", ns, consensus_fraction)

    # RDA4 — two-axis producer/trust consensus + fallback (REAL bridge)
    # trust axis = candidates matching the model's predicted-expected on derived cases
    expected_map = {inp: e for inp, e in artifacts.counterexamples if e}
    if expected_map:
        trust_labels = []
        for lab in sigs:
            sig = sigs[lab]
            match = all(_norm_out(expected_map[derived[k]]) == sig[k]
                        for k in range(len(derived)) if derived[k] in expected_map)
            if match:
                trust_labels.append(lab)
        prod_branch = classify_producer_axis_branch(services=[f"sig::{lab}" for lab in modal_labels])
        trust_branch = (W42_BRANCH_INVARIANCE_RATIFIED if trust_labels
                        else "trust_no_trigger")
        # reuse the role-invariance ratify constant string as the trust-ratified token
        from .integrated_synthesis import W41_TRUST_AXIS_RATIFIED, W41_TRUST_AXIS_NO_TRIGGER
        t_branch = W41_TRUST_AXIS_RATIFIED if trust_labels else W41_TRUST_AXIS_NO_TRIGGER
        integ_branch, integ_services = select_integrated_synthesis_decision(
            producer_axis_branch=prod_branch, trust_axis_branch=t_branch,
            producer_services=[f"sig::{lab}" for lab in modal_labels],
            trust_services=[f"sig::{lab}" for lab in trust_labels])
        if integ_branch == W41_BRANCH_INTEGRATED_AXES_DIVERGED_ABSTAINED:
            return RdaSelectionV1(variant, None, None, True, "INTEGRATED_DIVERGED_ABSTAINED",
                                  ns, consensus_fraction)
        # both-axes or producer-only -> commit the agreed rep (prefer a trust-ratified rep)
        commit_lab = next((l for l in modal_labels if l in trust_labels), rep.label)
        commit = next(im for im in survivors if im.label == commit_lab)
        return RdaSelectionV1(variant, commit.label, commit.code, False,
                              f"INTEGRATED_{integ_branch}", ns, consensus_fraction)
    # no predicted-expected -> trust no-trigger -> producer-only (commit majority rep)
    if not quorum:
        return RdaSelectionV1(variant, None, None, True, "PRODUCER_NO_QUORUM_ABSTAIN",
                              ns, consensus_fraction)
    return RdaSelectionV1(variant, rep.label, rep.code, False, "PRODUCER_ONLY_FALLBACK",
                          ns, consensus_fraction)


# =============================================================================
# orchestrator
# =============================================================================
@dataclasses.dataclass(frozen=True)
class RdaTargetOutcomeV1:
    short_name: str
    family: str
    n_calls: int
    diversity: dict
    pool_pass: bool                       # any implementation passes secret (gen ceiling)
    pool_first_label: Optional[str]
    selections: dict                      # variant -> RdaSelectionV1.to_dict()
    committed_pass: dict                  # variant -> bool
    abstained: dict                       # variant -> bool
    n_public_survivors: int
    n_impls_parse: int
    leakage_clean: bool = True            # all impl + committed candidates pass the guard
    notes: str = ""

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def cid(self) -> str:
        return _sha(self.to_dict())


def run_role_diverse_search_v1(gen: GenFn, problem: IcpcPilotProblemV1, *,
                               K: int = 5, n_sketches: int = DEFAULT_N_SKETCHES,
                               analyze_temp: float = 0.5, impl_temp: float = 0.2,
                               max_tokens: int = 1536, timeout_s: float = 10.0,
                               family: str = "", grade_secret: bool = True,
                               leakage_check: Optional[Callable[[str], bool]] = None
                               ) -> RdaTargetOutcomeV1:
    """One target, K model calls = 1 ANALYZE + (K-1) IMPLEMENT (matched to baseline A1 K)."""
    n_impl = max(1, K - 1)
    n_sketches = min(n_sketches, n_impl)
    a_text, _w = gen(build_analyze_prompt_v1(problem, n_sketches=n_sketches),
                     max_tokens, analyze_temp)
    arts = parse_role_artifacts_v1(a_text, n_sketches=n_sketches)
    n_calls = 1

    # one implement call per sketch (pad with the strongest sketch if model gave fewer)
    sketches = list(arts.sketches)
    if not sketches:
        sketches = [SketchV1("A", "direct", "Implement the most direct correct algorithm.")]
    while len(sketches) < n_impl:
        sketches.append(sketches[len(sketches) % len(arts.sketches or sketches)])
    impls: list[CandidateImplV1] = []
    for i in range(n_impl):
        sk = sketches[i]
        text, _w = gen(build_implement_prompt_v1(problem, arts.spec, sk), max_tokens, impl_temp)
        code = extract_candidate_code_v1(response_text=text)
        impls.append(CandidateImplV1(f"{sk.label}{i}", code, _parses(code)))
        n_calls += 1

    div = compute_diversity_v1(arts, impls, [inp for inp, _e in problem.samples])

    # pool ceiling (diagnostic): does ANY parsing impl pass secret?
    pool_pass, pool_label = False, None
    secret_cache: dict[str, bool] = {}

    def _secret_ok(code: str) -> bool:
        key = _sha(code)
        if key in secret_cache:
            return secret_cache[key]
        if not grade_secret or not _parses(code):
            secret_cache[key] = False
            return False
        ok, _tail, _n = grade_on_secret_v1(problem, code, timeout_s=timeout_s)
        secret_cache[key] = bool(ok)
        return bool(ok)

    for im in impls:
        if im.parses and _secret_ok(im.code):
            pool_pass, pool_label = True, im.label
            break

    selections, committed_pass, abstained = {}, {}, {}
    nps = 0
    committed_codes: list[str] = []
    for variant in RDA_VARIANTS:
        sel = select_committed_v1(problem, impls, arts, variant=variant, timeout_s=timeout_s)
        nps = sel.n_public_survivors
        selections[variant] = sel.to_dict()
        cp = bool(sel.committed_code) and _secret_ok(sel.committed_code)
        committed_pass[variant] = cp
        abstained[variant] = sel.abstained
        if sel.committed_code:
            committed_codes.append(sel.committed_code)

    # § 3 leakage: every impl + committed candidate must pass the guard (accepted = tripwire)
    leak_clean = True
    if leakage_check is not None:
        for code in [im.code for im in impls if im.parses] + committed_codes:
            try:
                if not leakage_check(code):
                    leak_clean = False
                    break
            except Exception:  # noqa: BLE001 — a guard error is treated as NOT-clean (safe)
                leak_clean = False
                break

    return RdaTargetOutcomeV1(
        short_name=problem.short_name, family=family or "", n_calls=n_calls,
        diversity=div.to_dict(), pool_pass=pool_pass, pool_first_label=pool_label,
        selections=selections, committed_pass=committed_pass, abstained=abstained,
        n_public_survivors=nps, n_impls_parse=sum(1 for im in impls if im.parses),
        leakage_clean=leak_clean,
        notes=f"n_sketches_parsed={len(arts.sketches)} n_cx={len(arts.counterexamples)}")


# =============================================================================
# honest mining: W79 substrate-controller applicability examination (RDA4 kill record)
# =============================================================================
def examine_substrate_controller_applicability_v1() -> dict:
    """NIM-free: record WHY the operator-named W79 substrate controllers are NOT a clean
    code-candidate consensus bridge (a literal call would be fake-different), so RDA4's
    consensus/abstain role is filled by the W41/W42 synthesis decisions instead.

    Returns a machine-checkable applicability report (used in tests + the verdict)."""
    import inspect
    report = {"schema": "coordpy.w128_substrate_controller_applicability.v1", "controllers": {}}
    checks = {
        "team_consensus_controller_v14": (
            "coordpy.team_consensus_controller_v14", "TeamConsensusControllerV14"),
        "consensus_fallback_controller_v25": (
            "coordpy.consensus_fallback_controller_v25", "ConsensusFallbackControllerV25"),
        "hosted_cost_planner_v12": (
            "coordpy.hosted_cost_planner_v12", "HostedCostPlannerV12"),
        "hosted_real_handoff_coordinator_v11": (
            "coordpy.hosted_real_handoff_coordinator_v11", "HostedRealHandoffCoordinatorV11"),
    }
    import importlib
    for name, (mod, cls) in checks.items():
        try:
            klass = getattr(importlib.import_module(mod), cls)
            fields = [f.name for f in dataclasses.fields(klass)] if dataclasses.is_dataclass(klass) else []
            src = inspect.getsource(klass)
            substrate_specific = bool(
                re.search(r"substrate|replacement_then_restart|trust_floor|"
                          r"long_delay|hidden|kv|prefix|attention", src, re.I))
            code_candidate_applicable = bool(
                re.search(r"candidate|stdout|secret|pass@|public.?sample|sketch", src, re.I))
            report["controllers"][name] = {
                "fields_sample": fields[:6], "substrate_specific": substrate_specific,
                "code_candidate_applicable": code_candidate_applicable,
                "literal_bridge_would_be_fake": substrate_specific and not code_candidate_applicable}
        except Exception as e:  # noqa: BLE001
            report["controllers"][name] = {"error": f"{type(e).__name__}: {e}"}
    report["all_substrate_specific"] = all(
        c.get("literal_bridge_would_be_fake", False)
        for c in report["controllers"].values() if "error" not in c)
    report["consensus_provided_by"] = [
        "role_invariant_synthesis.select_role_invariance_decision",
        "integrated_synthesis.select_integrated_synthesis_decision"]
    return report


# =============================================================================
# W128 hard-cluster dev-bench earn gate  (R1' — strict)
# =============================================================================
DEV_MIN_NET_GAIN = 2

# the atlas non-scaffoldable families (the W128 hard clusters); graph_flow has 0 EXPOSED
# supply (census) so the named-present hard cluster on the EXPOSED side is simulation_grid.
NON_SCAFFOLDABLE_FAMILIES = ("graph_flow", "simulation_grid", "adhoc_math",
                             "search_enumeration", "greedy_scheduling")
NAMED_HARD_CLUSTER = "simulation_grid"


@dataclasses.dataclass(frozen=True)
class RdaDevBenchTargetResultV1:
    short_name: str
    family: str
    baseline_pass: bool
    scaffold_pass: Optional[bool]
    rda_committed_pass: bool      # the FULL mechanism (RDA4)
    rda_pool_pass: bool           # generation ceiling
    rda_abstained: bool
    diversity_real: bool
    leakage_clean: bool
    failure_was_trivial: bool

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class RdaDevBenchEarnVerdictV1:
    schema: str
    n_targets: int
    baseline_total_pass: int
    scaffold_total_pass: int
    rda_total_pass: int
    rda_pool_total_pass: int
    rda_unique_solves: int        # rda_committed ∧ ¬baseline
    rda_regressions: int          # baseline ∧ ¬rda_committed
    net_rda_gain: int
    net_vs_scaffold: int
    gain_families: tuple[str, ...]
    gain_distinct_families: int
    gain_includes_named_cluster: bool
    gain_is_nontrivial: bool
    winners_diversity_real: bool
    all_leakage_clean: bool
    r1a_net_gain: bool
    r1b_cluster_spread: bool
    r1c_clean_and_real: bool
    r1d_nontrivial: bool
    r1e_beats_scaffold: bool
    earned: bool
    verdict_label: str
    rationale: str

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["gain_families"] = list(self.gain_families)
        return d


def apply_rda_dev_bench_earn_gate_v1(
        results: Sequence[RdaDevBenchTargetResultV1]) -> RdaDevBenchEarnVerdictV1:
    n = len(results)
    base = sum(1 for r in results if r.baseline_pass)
    scaf = sum(1 for r in results if r.scaffold_pass)
    rda = sum(1 for r in results if r.rda_committed_pass)
    pool = sum(1 for r in results if r.rda_pool_pass)
    uniq = [r for r in results if r.rda_committed_pass and not r.baseline_pass]
    regr = [r for r in results if r.baseline_pass and not r.rda_committed_pass]
    net = len(uniq) - len(regr)
    scaf_uniq = sum(1 for r in results if r.scaffold_pass and not r.baseline_pass)
    scaf_regr = sum(1 for r in results if r.baseline_pass and not r.scaffold_pass)
    net_scaf = scaf_uniq - scaf_regr
    gain_fams = tuple(sorted({r.family for r in uniq}))
    includes_named = any(r.family == NAMED_HARD_CLUSTER for r in uniq)
    nontrivial = any(not r.failure_was_trivial for r in uniq) if uniq else False
    winners_real = all(r.diversity_real for r in uniq) if uniq else False
    clean = all(r.leakage_clean for r in results)

    r1a = net >= DEV_MIN_NET_GAIN
    r1b = (len(gain_fams) >= 2) or includes_named
    r1c = winners_real and clean
    r1d = nontrivial
    r1e = net >= net_scaf
    earned = r1a and r1b and r1c and r1d and r1e
    if earned:
        label = "ROLE_DIVERSE_HARD_CLUSTER_DEV_BENCH_EARNED"
        why = (f"net role-diverse gain {net:+d} (>= +{DEV_MIN_NET_GAIN}) over baseline on "
               f"{n} hard-cluster targets, families={list(gain_fams)} "
               f"(named-cluster solve={includes_named}), winners diversity-REAL + "
               f"leakage-clean + nontrivial, and net >= scaffold net ({net_scaf:+d}).")
    else:
        fails = [k for k, v in (("R1a_net", r1a), ("R1b_spread", r1b),
                                ("R1c_clean_real", r1c), ("R1d_nontrivial", r1d),
                                ("R1e_beats_scaffold", r1e)) if not v]
        label = "ROLE_DIVERSE_HARD_CLUSTER_DEV_BENCH_NOT_EARNED"
        why = (f"net role-diverse gain {net:+d} (need >= +{DEV_MIN_NET_GAIN}); "
               f"failed gates: {fails}; baseline {base}/{n} -> rda {rda}/{n} "
               f"(pool ceiling {pool}/{n}); scaffold net {net_scaf:+d}.")
    return RdaDevBenchEarnVerdictV1(
        schema="coordpy.w128_rda_dev_bench_earn.v1", n_targets=n,
        baseline_total_pass=base, scaffold_total_pass=scaf, rda_total_pass=rda,
        rda_pool_total_pass=pool, rda_unique_solves=len(uniq), rda_regressions=len(regr),
        net_rda_gain=net, net_vs_scaffold=net - net_scaf, gain_families=gain_fams,
        gain_distinct_families=len(gain_fams), gain_includes_named_cluster=includes_named,
        gain_is_nontrivial=nontrivial, winners_diversity_real=winners_real,
        all_leakage_clean=clean, r1a_net_gain=r1a, r1b_cluster_spread=r1b,
        r1c_clean_and_real=r1c, r1d_nontrivial=r1d, r1e_beats_scaffold=r1e,
        earned=earned, verdict_label=label, rationale=why)
