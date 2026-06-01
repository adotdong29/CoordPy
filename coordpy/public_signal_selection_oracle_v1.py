"""W129 Lane α — Public-Signal Selection Oracle V1 (COO-9 sibling).

W128 proved a genuinely-different role-diverse algorithm SEARCH is REAL and **lifts the
generation ceiling** (pool 3/11 > plain baseline 2/11) but is **NOT EARNED**: RDA4 committed
only 2/11 (net +0 = +1 ``blueberrywaffle`` − 1 ``pawnshop``).  The load-bearing W128 finding:
the bottleneck is the **verification-based SELECTION layer, not generation** — without a
hidden-test oracle the public/derived signal could not convert the lifted ceiling into
committed wins.  W129 attacks the SELECTOR directly.

The W129 $0 recon (``scripts/run_w129_stored_pool_recon_v1.py``) localizes the miss precisely:

* ``blueberrywaffle`` (the W128 unique win): the hidden-WRONG public-survivor DIFFERS from the
  two hidden-CORRECT survivors on the model-derived cases ⇒ **separable** ⇒ a falsifier /
  differential-consensus selector keeps it.  W129 must NOT regress it.
* ``pawnshop`` (the W128 regression): the hidden-CORRECT ``B1`` and the hidden-WRONG ``A0`` are
  two near-identical greedy programs that produce **byte-identical output on every public
  sample AND every model-derived case** (``separable_on_derived = False``), with **no
  brute-force/reference sketch**.  The choice is **public-signal UNDER-DETERMINED**: no
  NIM-free agreement/consensus/falsifier oracle can break a 2-way behavioral tie with no
  reference.  W128 committed ``A0`` (alphabetical-first) ⇒ a MIS-COMMIT.

So the honest selector slate is:

* **SO1 — public-derived falsifier stack** (NIM-free).  Run public survivors on the model's
  DERIVED cases PLUS auto-generated FORMAT-PRESERVING mutations of the samples; falsify a
  candidate that CRASHES / TLEs / format-violates / contradicts the model's predicted-expected
  on a case where ≥1 other survivor stays clean (DIFFERENTIAL — a mutation that breaks every
  survivor is a bad mutation, ignored).  Typed via ``parse_failure_digest_v1``.
* **SO2 — differential disagreement selector** (NIM-free, REAL bridge to
  ``integrated_synthesis``).  Group survivors by behaviour signature over the case union;
  producer axis = signature-majority class; trust axis = the falsifier-survivor set; combine
  via ``select_integrated_synthesis_decision`` — commit the agreed rep, ABSTAIN on divergence.
* **SO3 — verifier-final chooser** (needs ``gen``; mines the ``mathvista_bench_v2``
  verifier-final pattern).  A final model call SEES the candidates + each one's public/derived
  verdict and makes a REAL final CHOICE (``CHOOSE <label>``) or ABSTAINs, on public-signal
  evidence only.  This is the only lever that can break a public-signal-under-determined tie
  (an external correctness JUDGE), and it carries a skeptical prior
  (``W96-L-MATHVISTA-BENCH-V2-VLM-VERIFIER-FINAL-K5-CAP``).
* **SO4 — trust-weighted abstain ensemble** (NIM-free).  Each survivor → a trust scalar =
  falsifier-survival fraction; integrity = format/crash-clean.  Realizes the
  ``integrity_trust_coupled_consensus_v1`` integrity-penalty + trust-weighted-quorum + ABSTAIN
  CONCEPT natively over HONEST code-correctness trust signals.  The substrate
  ``TrustWeightedConsensusController`` (latent ``MergeableLatentCapsuleV3`` + cosine + merge)
  literal bridge is KILLED as latent-specific fake-different
  (``examine_trust_machinery_applicability_v1``) — the W128 W79 lesson.

A principled selector ABSTAINS on a public-signal-under-determined tie rather than mis-commit
(SO2/SO4); SO1-naive falls back to first survivor (the W128 RDA1 behaviour, kept for contrast).

No leakage: every model-facing prompt + every oracle case uses ONLY the PUBLIC statement,
PUBLIC samples, model role artifacts, candidate code, candidate outputs, and typed failure
digests.  The accepted solution / secret cases are NEVER shown to any selector path (tripwire
only — the caller's ``leakage_check``).  Held outside the stable SDK contract: explicit-import
only, ``coordpy/__init__.py`` untouched, no version bump, no PyPI.
"""
from __future__ import annotations

import dataclasses
import hashlib
import importlib
import inspect
import json
import re
from typing import Any, Callable, Optional, Sequence

from .role_diverse_algorithm_search_v1 import (
    CandidateImplV1, RoleArtifactsV1, _norm_code, _norm_out, _parses, _public_survivors,
    _run_capture_stdout_v1, _sha)
from .icpc_reflexion_bench_v1 import IcpcPilotProblemV1, extract_candidate_code_v1
from .integrated_synthesis import (
    W41_BRANCH_INTEGRATED_AXES_DIVERGED_ABSTAINED, W41_PRODUCER_AXIS_FIRED,
    W41_PRODUCER_AXIS_NO_TRIGGER, W41_TRUST_AXIS_NO_TRIGGER, W41_TRUST_AXIS_RATIFIED,
    select_integrated_synthesis_decision)

# (prompt, max_tokens, temperature) -> (text, wall_ms)
GenFn = Callable[[str, int, float], Any]

SO_VARIANTS: tuple[str, ...] = ("SO1", "SO2", "SO3", "SO4", "SOLEAD")
MAX_AUTO_CASES = 6
AUTO_CASE_TIMEOUT_S = 5.0
TRUST_QUORUM = 0.5  # SO4: max-trust survivor must exceed this share to commit (else abstain)


# =============================================================================
# format signature (public-signal output-shape oracle, conservative)
# =============================================================================
def _out_shape(s: str) -> tuple[int, int, bool]:
    """(n_lines, n_tokens, all_numeric) of a normalized stdout — a loose FORMAT descriptor."""
    s = (s or "").strip()
    if not s:
        return (0, 0, True)
    lines = [ln for ln in s.splitlines() if ln.strip()]
    toks = s.split()
    numeric = all(re.fullmatch(r"[-+]?\d+(?:\.\d+)?", t or "") for t in toks) if toks else True
    return (len(lines), len(toks), numeric)


def sample_output_shape_v1(problem: IcpcPilotProblemV1) -> dict:
    """Derive a loose expected-output FORMAT from the PUBLIC sample outputs (no secret)."""
    shapes = [_out_shape(exp) for _inp, exp in problem.samples]
    numeric = all(s[2] for s in shapes)
    return {"numeric": numeric, "min_tokens": min((s[1] for s in shapes), default=0)}


def _format_violation(out: str, shape: dict) -> bool:
    """Flag a CLEAR format violation only (conservative: avoid false-falsifying)."""
    if out in ("<PARSE_ERR>",) or out.startswith("<ERR:") or out == "<TIMEOUT>":
        return True
    ls, tks, num = _out_shape(out)
    if shape.get("numeric") and tks > 0 and not num:
        return True  # samples are all-numeric but this candidate emitted non-numeric tokens
    if tks == 0:  # empty output where samples produce content
        return shape.get("min_tokens", 0) > 0
    return False


# =============================================================================
# auto-generated FORMAT-PRESERVING public-signal cases (no secret)
# =============================================================================
def derive_auto_cases_v1(problem: IcpcPilotProblemV1, *, max_cases: int = MAX_AUTO_CASES,
                         seed_tag: str = "") -> list[str]:
    """Deterministic FORMAT-PRESERVING mutations of the PUBLIC sample inputs.

    Goal: produce extra inputs that may EXPOSE a behavioural divergence between near-identical
    candidates (the ``pawnshop`` failure mode), WITHOUT inventing out-of-format instances.
    Strategy (per sample input): (a) the sample as-is; (b) for each line that is a list of >=2
    integers, a deterministic rotation of that line's tokens (preserves multiset + line count +
    token count + numeric format).  A rotation is the safest mutation that still changes the
    instance.  Determinism: rotation amount keyed by a stable hash (no Math.random)."""
    out: list[str] = []
    seen: set[str] = set()
    for si, (inp, _exp) in enumerate(problem.samples):
        cand_inputs = [inp]
        lines = inp.split("\n")
        for li, ln in enumerate(lines):
            toks = ln.split()
            if len(toks) >= 2 and all(re.fullmatch(r"[-+]?\d+", t) for t in toks):
                amt = 1 + (int(hashlib.sha256(f"{seed_tag}|{si}|{li}".encode()).hexdigest(), 16)
                           % max(1, len(toks) - 1))
                rot = toks[amt:] + toks[:amt]
                mutated = list(lines)
                mutated[li] = " ".join(rot)
                cand_inputs.append("\n".join(mutated))
        for ci in cand_inputs:
            key = _norm_out(ci)
            if key not in seen:
                seen.add(key)
                out.append(ci)
            if len(out) >= max_cases:
                return out
    return out[:max_cases]


# =============================================================================
# complexity STRESS cases (public-signal: the stated constraint + a large valid instance)
# =============================================================================
STRESS_TIMEOUT_S = 2.0   # tight: an asymptotically-slower candidate TLEs fast (detected differentially)
STRESS_CAP = 30000  # runtime cap on the scaled size


def parse_max_constraint_v1(statement: str) -> Optional[int]:
    """Best-effort parse of the largest size BOUND stated in the problem (public signal).

    Catches ``10^5`` / ``3 \\cdot 10^5`` / ``2 \\times 10^6`` / bare ``100000`` / ``100,000``.
    Returns the max plausible size bound in [1000, 2e6], else None (then NO stress case is made —
    conservative: never scale beyond a bound we cannot read, to avoid false-falsifying a
    correct candidate that is fine within its real constraint)."""
    s = statement or ""
    # strip LaTeX digit separators/spacing so "300\,000" / "300 000" / "300,000" -> "300000"
    s = re.sub(r"(?<=\d)(?:\\,|\\;|\\ |~|,|\s)(?=\d\d\d(?:\D|$))", "", s)
    cands: list[int] = []
    for m in re.finditer(r"(\d+)\s*(?:\\cdot|\\times|\*|x)?\s*10\s*[\^{]\s*(\d+)", s):
        try:
            cands.append(int(m.group(1)) * (10 ** int(m.group(2))))
        except ValueError:
            pass
    for m in re.finditer(r"10\s*[\^{]\s*(\d+)", s):
        cands.append(10 ** int(m.group(1)))
    for m in re.finditer(r"\b(\d{4,7})\b", s):
        cands.append(int(m.group(1)))
    cands = [c for c in cands if 1000 <= c <= 2_000_000]
    return max(cands) if cands else None


def derive_stress_cases_v1(problem: IcpcPilotProblemV1, *, cap: int = STRESS_CAP) -> list[str]:
    """One LARGE format-preserving instance scaled to the STATED constraint (public signal).

    Scales the sample whose first token is a small int N (a plausible size) up to
    ``min(parsed_constraint, cap)`` by CYCLING the sample's N-length integer lists (preserves
    format + value range).  Returns [] when no constraint is parseable (conservative).  Used
    DIFFERENTIALLY: a candidate that TLEs here while a peer stays clean is asymptotically slower
    ⇒ it would TLE on the hidden large cases the constraint implies (the pawnshop A0 pattern).

    ``W129_STRESS_OFF`` (env) skips this component — used to time the eval cheaply; it changes NO
    pool-bearing outcome here (the generic format-preserving scale-up does not construct the
    adversarial worst-case that would expose an O(N²) candidate, so it never falsifies)."""
    import os as _os
    if _os.environ.get("W129_STRESS_OFF"):
        return []
    bound = parse_max_constraint_v1(problem.statement)
    if not bound:
        return []
    target = min(int(bound), cap)
    if target < 2000:  # too small to expose an O(N^2) vs O(N) gap
        return []
    for inp, _exp in problem.samples:
        lines = inp.rstrip("\n").split("\n")
        if not lines:
            continue
        head = lines[0].split()
        if not (head and re.fullmatch(r"\d+", head[0])):
            continue
        n = int(head[0])
        if not (1 <= n <= 200):
            continue
        new_lines = []
        for li, ln in enumerate(lines):
            toks = ln.split()
            if li == 0:
                nt = list(toks)
                nt[0] = str(target)
                new_lines.append(" ".join(nt))
            elif len(toks) == n and all(re.fullmatch(r"-?\d+", t) for t in toks):
                vals = (toks * (target // n + 1))[:target]
                new_lines.append(" ".join(vals))
            else:
                new_lines.append(ln)
        return ["\n".join(new_lines) + "\n"]
    return []


# =============================================================================
# per-candidate grading against the public-signal oracle
# =============================================================================
@dataclasses.dataclass(frozen=True)
class CandidateGradeV1:
    label: str
    parses: bool
    public_pass: bool
    n_cases: int
    n_clean: int               # cases where this candidate produced a clean, format-OK output
    n_crash: int
    n_format_violation: int
    n_pred_mismatch: int
    differentially_falsified: bool   # falsified on a case where >=1 other survivor stayed clean
    survival_frac: float
    sig: tuple[str, ...]
    reasons: tuple[str, ...]

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def grade_candidates_v1(problem: IcpcPilotProblemV1, artifacts: RoleArtifactsV1,
                        survivors: Sequence[CandidateImplV1], *,
                        timeout_s: float = AUTO_CASE_TIMEOUT_S,
                        seed_tag: str = "") -> tuple[list[CandidateGradeV1], list[str]]:
    """Grade public survivors against the public-signal falsifier oracle (NIM-free).

    Cases = model DERIVED counterexamples + auto FORMAT-PRESERVING mutations.  A candidate is
    DIFFERENTIALLY falsified on a case iff it crashes/TLEs/format-violates (or contradicts the
    model's predicted-expected) AND >=1 other survivor produced a clean, format-OK output on
    the SAME case (so a globally-bad case cannot falsify anyone)."""
    shape = sample_output_shape_v1(problem)
    # (stdin, predicted_expected | None, per-case timeout, in_signature).  Only the model's
    # DERIVED counterexamples (meant-valid edge cases) feed the behaviour SIGNATURE used by SO2
    # majority — AUTO mutations + STRESS scale-ups are FALSIFICATION-ONLY (differential
    # crash/TLE/format), because two CORRECT candidates may legitimately differ on a malformed
    # mutation or a huge scaled input and that must not split the agreement majority.
    cases: list[tuple[str, Optional[str], float, bool]] = []
    cases += [(inp, exp, timeout_s, True) for inp, exp in artifacts.counterexamples]
    cases += [(inp, None, timeout_s, False)
              for inp in derive_auto_cases_v1(problem, seed_tag=seed_tag)]
    cases += [(inp, None, STRESS_TIMEOUT_S, False) for inp in derive_stress_cases_v1(problem)]
    in_sig = [c[3] for c in cases]
    case_inputs = [c[0] for c in cases]
    # per-candidate per-case output + clean flag
    outs: dict[str, list[str]] = {}
    clean: dict[str, list[bool]] = {}
    for im in survivors:
        o_row, c_row = [], []
        for inp, _exp, ct, _insig in cases:
            out, _dig = _run_capture_stdout_v1(im.code, inp, timeout_s=ct)
            crashed = out.startswith("<ERR:") or out in ("<TIMEOUT>", "<PARSE_ERR>")
            fmt_bad = _format_violation(out, shape)
            o_row.append(out)
            c_row.append(not crashed and not fmt_bad)
        outs[im.label] = o_row
        clean[im.label] = c_row
    grades: list[CandidateGradeV1] = []
    for im in survivors:
        n_crash = sum(1 for o in outs[im.label]
                      if o.startswith("<ERR:") or o in ("<TIMEOUT>", "<PARSE_ERR>"))
        n_fmt = sum(1 for o in outs[im.label] if _format_violation(o, shape)) - n_crash
        n_fmt = max(0, n_fmt)
        # predicted-expected mismatch (model-supplied expected on derived cases only)
        n_pred = 0
        for k, (inp, exp, _ct, _insig) in enumerate(cases):
            if exp:
                if _norm_out(exp) != outs[im.label][k]:
                    n_pred += 1
        # differential falsification
        diff_fals = False
        reasons: list[str] = []
        for k in range(len(case_inputs)):
            if not clean[im.label][k] and any(clean[o][k] for o in clean if o != im.label):
                diff_fals = True
                tag = ("crash" if outs[im.label][k].startswith("<ERR:")
                       or outs[im.label][k] in ("<TIMEOUT>", "<PARSE_ERR>") else "fmt")
                reasons.append(f"diff_{tag}@case{k}")
        n_clean = sum(clean[im.label])
        grades.append(CandidateGradeV1(
            label=im.label, parses=im.parses, public_pass=True, n_cases=len(case_inputs),
            n_clean=n_clean, n_crash=n_crash, n_format_violation=n_fmt, n_pred_mismatch=n_pred,
            differentially_falsified=diff_fals,
            survival_frac=(n_clean / len(case_inputs)) if case_inputs else 1.0,
            sig=tuple(o for k, o in enumerate(outs[im.label]) if in_sig[k]),
            reasons=tuple(reasons)))
    return grades, case_inputs


# =============================================================================
# the selection variants  (SO1 / SO2 / SO3 / SO4 / SOLEAD)
# =============================================================================
@dataclasses.dataclass(frozen=True)
class SoSelectionV1:
    variant: str
    committed_label: Optional[str]
    committed_code: Optional[str]
    abstained: bool
    branch: str
    n_public_survivors: int
    n_post_falsifier_survivors: int
    evidence_used: bool        # did a DISCRIMINATING public signal drive the choice?
    detail: dict

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["committed_code"] = bool(self.committed_code)
        d["committed_code_sha"] = _sha(self.committed_code)[:16] if self.committed_code else None
        return d


def _commit(variant, im, branch, ns, npf, evidence, detail) -> SoSelectionV1:
    return SoSelectionV1(variant, im.label, im.code, False, branch, ns, npf, evidence, detail)


def _abstain(variant, branch, ns, npf, evidence, detail) -> SoSelectionV1:
    return SoSelectionV1(variant, None, None, True, branch, ns, npf, evidence, detail)


def _sig_classes(grades: Sequence[CandidateGradeV1]) -> dict[tuple, list[str]]:
    classes: dict[tuple, list[str]] = {}
    for g in grades:
        classes.setdefault(g.sig, []).append(g.label)
    return classes


def select_so_v1(problem: IcpcPilotProblemV1, impls: Sequence[CandidateImplV1],
                 artifacts: RoleArtifactsV1, *, variant: str,
                 gen: Optional[GenFn] = None, max_tokens: int = 1536,
                 verifier_temp: float = 0.0, timeout_s: float = AUTO_CASE_TIMEOUT_S,
                 seed_tag: str = "") -> SoSelectionV1:
    """One public-signal selection over the SAME generations (SO1/SO2/SO4 NIM-free; SO3 = 1 gen call)."""
    survivors = _public_survivors(problem, impls, timeout_s=timeout_s)
    ns = len(survivors)
    if ns == 0:
        return _abstain(variant, "NO_PUBLIC_SURVIVOR_ABSTAIN", 0, 0, False, {})
    if ns == 1:
        return _commit(variant, survivors[0], "SINGLE_SURVIVOR", 1, 1, False,
                       {"note": "only one public survivor"})

    grades, case_inputs = grade_candidates_v1(problem, artifacts, survivors,
                                              timeout_s=timeout_s, seed_tag=seed_tag)
    gmap = {g.label: g for g in grades}
    post = [im for im in survivors if not gmap[im.label].differentially_falsified]
    npf = len(post)
    falsifier_bit = npf < ns  # the falsifier stack eliminated >=1 survivor (discriminating)

    # ----- SO1: public-derived falsifier stack, naive on the surviving set -----
    if variant == "SO1":
        if npf == 1:
            return _commit(variant, post[0], "FALSIFIER_UNIQUE", ns, npf, True,
                           {"eliminated": [g.label for g in grades if g.differentially_falsified]})
        pool = post or survivors
        # surviving set still ambiguous -> first survivor (W128 RDA1 contrast), evidence iff falsified some
        return _commit(variant, pool[0], "FALSIFIER_FIRST", ns, npf, falsifier_bit,
                       {"survivors": [im.label for im in pool]})

    # ----- SO2: differential disagreement via integrated_synthesis -----
    if variant == "SO2":
        pool = post or survivors
        classes = _sig_classes([gmap[im.label] for im in pool])
        n_classes = len(classes)
        modal_sig, modal = max(classes.items(), key=lambda kv: (len(kv[1]), kv[1]))
        strict_majority = len(modal) > len(pool) / 2
        struct: dict[str, list[str]] = {}
        for im in pool:
            struct.setdefault(_norm_code(im.code), []).append(im.label)
        # producer fires ONLY on a REAL adjudicable disagreement (>=2 behaviour classes + a
        # strict majority); a UNANIMOUS class is not a "majority" — there is nothing to adjudicate.
        producer = (W41_PRODUCER_AXIS_FIRED if (n_classes >= 2 and strict_majority)
                    else W41_PRODUCER_AXIS_NO_TRIGGER)
        # trust axis ratified iff the falsifier stack actually eliminated a survivor.
        trust = W41_TRUST_AXIS_RATIFIED if falsifier_bit else W41_TRUST_AXIS_NO_TRIGGER
        integ_branch, integ_services = select_integrated_synthesis_decision(
            producer_axis_branch=producer, trust_axis_branch=trust,
            producer_services=[f"sig::{l}" for l in modal],
            trust_services=[f"sig::{im.label}" for im in post])
        if integ_branch == W41_BRANCH_INTEGRATED_AXES_DIVERGED_ABSTAINED:
            return _abstain(variant, "INTEGRATED_DIVERGED_ABSTAINED", ns, npf, True,
                            {"classes": {str(k): v for k, v in classes.items()}})
        if integ_services:  # a real producer/trust signal gave a committed set
            svc = [s.split("::", 1)[1] for s in integ_services if "::" in s]
            commit_lab = next((im.label for im in pool if im.label in svc),
                              (modal[0] if modal else pool[0].label))
            commit = next(im for im in pool if im.label == commit_lab)
            return _commit(variant, commit, f"INTEGRATED_{integ_branch}", ns, npf, True,
                           {"modal": modal, "branch": integ_branch, "n_classes": n_classes})
        # NEITHER_AXIS: unanimous / no-majority with no discriminator.
        if len(struct) == 1:
            # structurally identical survivors == the SAME program -> safe (not evidence-driven).
            return _commit(variant, pool[0], "SAME_PROGRAM_COMMIT", ns, npf, False,
                           {"note": "structurally identical survivors"})
        # structurally-distinct survivors agreeing on all public signal -> UNDER-DETERMINED tie
        # (the pawnshop pattern) -> ABSTAIN rather than mis-commit a coin-flip.
        return _abstain(variant, "UNDER_DETERMINED_TIE_ABSTAIN", ns, npf, False,
                        {"classes": {str(k): v for k, v in classes.items()},
                         "n_struct": len(struct)})

    # ----- SO4: trust-weighted abstain ensemble (native realization of the W83 concept) -----
    if variant == "SO4":
        pool = post or survivors
        # honest code trust = falsifier-survival fraction; integrity = format/crash clean
        ranked = sorted(pool, key=lambda im: (-gmap[im.label].survival_frac, im.label))
        top = ranked[0]
        top_trust = gmap[top.label].survival_frac
        total = sum(gmap[im.label].survival_frac for im in pool) or 1.0
        top_share = (gmap[top.label].survival_frac) / total
        # commit only if the top survivor's trust SHARE clears quorum AND it is integrity-clean
        integrity_clean = gmap[top.label].n_crash == 0 and gmap[top.label].n_format_violation == 0
        discriminating = top_share > TRUST_QUORUM and any(
            gmap[im.label].survival_frac < top_trust for im in pool)
        if discriminating and integrity_clean:
            return _commit(variant, top, "TRUST_WEIGHTED_COMMIT", ns, npf, True,
                           {"top_share": round(top_share, 3), "top_trust": round(top_trust, 3)})
        return _abstain(variant, "TRUST_TIE_ABSTAIN", ns, npf, False,
                        {"top_share": round(top_share, 3),
                         "trusts": {im.label: round(gmap[im.label].survival_frac, 3) for im in pool}})

    # ----- SO3: verifier-final chooser (1 model call; the only tie-breaker for under-determined) -----
    if variant == "SO3":
        if gen is None:
            raise ValueError("SO3 verifier-final requires a gen function")
        pool = post or survivors
        if len(pool) == 1:
            return _commit(variant, pool[0], "VERIFIER_SINGLE_SURVIVOR", ns, npf, True, {})
        prompt = build_verifier_final_prompt_v1(problem, artifacts,
                                                [gmap[im.label] for im in pool], pool)
        vtext, _w = gen(prompt, max_tokens, verifier_temp)
        choice = parse_verifier_choice_v1(vtext, [im.label for im in pool])
        if choice is None or choice == "ABSTAIN":
            return _abstain(variant, "VERIFIER_ABSTAIN", ns, npf, True,
                            {"verifier_raw": vtext[:200]})
        commit = next((im for im in pool if im.label == choice), None)
        if commit is None:  # verifier named a non-survivor -> safety abstain
            return _abstain(variant, "VERIFIER_NONSURVIVOR_ABSTAIN", ns, npf, True,
                            {"choice": choice, "survivors": [im.label for im in pool]})
        return _commit(variant, commit, "VERIFIER_CHOSE", ns, npf, True,
                       {"choice": choice, "verifier_raw": vtext[:200]})

    # ----- SOLEAD: SO1 falsifier -> SO2 differential -> SO3 verifier on residual tie -----
    if variant == "SOLEAD":
        s1 = select_so_v1(problem, impls, artifacts, variant="SO1", timeout_s=timeout_s,
                          seed_tag=seed_tag)
        if s1.n_post_falsifier_survivors == 1:
            return dataclasses.replace(s1, variant="SOLEAD", branch="LEAD_" + s1.branch)
        s2 = select_so_v1(problem, impls, artifacts, variant="SO2", timeout_s=timeout_s,
                          seed_tag=seed_tag)
        if not s2.abstained:
            return dataclasses.replace(s2, variant="SOLEAD", branch="LEAD_" + s2.branch)
        if gen is not None:
            s3 = select_so_v1(problem, impls, artifacts, variant="SO3", gen=gen,
                              max_tokens=max_tokens, verifier_temp=verifier_temp,
                              timeout_s=timeout_s, seed_tag=seed_tag)
            return dataclasses.replace(s3, variant="SOLEAD", branch="LEAD_" + s3.branch)
        return dataclasses.replace(s2, variant="SOLEAD", branch="LEAD_" + s2.branch)

    raise ValueError(f"unknown selector variant {variant!r}")


# =============================================================================
# SO3 verifier-final prompt + parse  (mined from mathvista_bench_v2)
# =============================================================================
_VERIFIER_FINAL_SYSTEM = (
    "You are an expert ICPC judge. Several candidate Python programs were written for the "
    "problem below. Each ALREADY passes every provided public sample, so the samples do not "
    "distinguish them. Your job is to pick the ONE candidate most likely to be CORRECT on the "
    "hidden judge tests, reasoning ONLY from the problem statement, the samples, the checkable "
    "invariants, and the candidate SOURCE CODE (look for off-by-one, wrong tie-breaking, "
    "incorrect equality/containment checks, edge-case mishandling, wrong complexity). You do "
    "NOT have the hidden tests. If after careful reading you genuinely cannot tell which is "
    "more likely correct, answer ABSTAIN rather than guess.")


def build_verifier_final_prompt_v1(problem: IcpcPilotProblemV1, artifacts: RoleArtifactsV1,
                                   grades: Sequence[CandidateGradeV1],
                                   pool: Sequence[CandidateImplV1]) -> str:
    samples = "\n".join(f"INPUT:\n{inp}\nOUTPUT:\n{exp}" for inp, exp in problem.samples)
    inv = "\n".join(f"- {iv}" for iv in artifacts.invariants) or "(none stated)"
    blocks = []
    gmap = {g.label: g for g in grades}
    for im in pool:
        g = gmap[im.label]
        blocks.append(
            f"=== CANDIDATE {im.label} (public-samples: PASS; derived-clean "
            f"{g.n_clean}/{g.n_cases}) ===\n```python\n{im.code}\n```")
    body = "\n\n".join(blocks)
    labels = ", ".join(im.label for im in pool)
    return (
        f"{_VERIFIER_FINAL_SYSTEM}\n\n"
        f"PROBLEM:\n{problem.statement}\n\n"
        f"PUBLIC SAMPLES:\n{samples}\n\n"
        f"CHECKABLE INVARIANTS (a correct output must satisfy these):\n{inv}\n\n"
        f"CANDIDATES (all pass the public samples):\n{body}\n\n"
        f"Reason briefly about the correctness of each candidate, then on the LAST line output "
        f"EXACTLY one of:\n  CHOOSE <label>   (one of: {labels})\n  ABSTAIN\n")


_CHOOSE_RE = re.compile(r"(?im)^\s*CHOOSE\s*[:#]?\s*([A-Za-z]?\d{0,2}[A-Za-z]?\d{0,2})\b")
_ABSTAIN_RE = re.compile(r"(?im)\bABSTAIN\b")


def parse_verifier_choice_v1(text: str, labels: Sequence[str]) -> Optional[str]:
    """Parse the verifier's final line. Returns a label, 'ABSTAIN', or None (unparseable)."""
    t = text or ""
    # prefer the LAST explicit CHOOSE line
    choose = list(_CHOOSE_RE.finditer(t))
    if choose:
        tok = choose[-1].group(1).strip()
        for lab in labels:
            if tok.lower() == lab.lower():
                return lab
        # fallback: bare label token appearing after the last CHOOSE
        for lab in labels:
            if re.search(rf"(?i)\b{re.escape(lab)}\b", t[choose[-1].start():]):
                return lab
    if _ABSTAIN_RE.search(t):
        return "ABSTAIN"
    # last-ditch: a unique label mention anywhere
    hits = [lab for lab in labels if re.search(rf"(?i)\b{re.escape(lab)}\b", t)]
    return hits[-1] if len(set(hits)) == 1 and hits else None


# =============================================================================
# honest mining: trust-machinery applicability examination  (the W128 W79 analogue)
# =============================================================================
def examine_trust_machinery_applicability_v1() -> dict:
    """NIM-free, machine-checkable: record WHY the substrate
    ``TrustWeightedConsensusController`` literal bridge to code-candidate selection is
    fake-different (latent-capsule / cosine / merge-specific), so SO4 realizes the
    integrity-trust-coupled ABSTAIN concept NATIVELY over honest code trust signals."""
    report = {"schema": "coordpy.w129_trust_machinery_applicability.v1", "modules": {}}
    targets = {
        "trust_weighted_consensus_controller": (
            "coordpy.trust_weighted_consensus_controller", "TrustWeightedConsensusController"),
        "integrity_trust_coupled_consensus_v1": (
            "coordpy.integrity_trust_coupled_consensus_v1", "IntegrityTrustCoupledDecisionV1"),
    }
    for name, (mod, sym) in targets.items():
        try:
            obj = getattr(importlib.import_module(mod), sym)
            src = inspect.getsource(obj)
            latent_specific = bool(re.search(
                r"MergeableLatentCapsule|cosine|MergeOperator|latent|capsule|prefix|kv\b",
                src, re.I))
            code_candidate_applicable = bool(re.search(
                r"candidate|stdout|secret|pass@|public.?sample|sketch|falsif", src, re.I))
            report["modules"][name] = {
                "latent_specific": latent_specific,
                "code_candidate_applicable": code_candidate_applicable,
                "literal_bridge_would_be_fake": latent_specific and not code_candidate_applicable}
        except Exception as e:  # noqa: BLE001
            report["modules"][name] = {"error": f"{type(e).__name__}: {e}"}
    ctrl = report["modules"].get("trust_weighted_consensus_controller", {})
    report["substrate_controller_literal_bridge_killed"] = bool(
        ctrl.get("literal_bridge_would_be_fake", False))
    report["so4_trust_signal"] = ("falsifier_survival_fraction + format/crash integrity "
                                  "(honest code-correctness proxy); integrity-penalty + "
                                  "trust-weighted-quorum + ABSTAIN concept realized natively")
    report["concept_source"] = "integrity_trust_coupled_consensus_v1 (integrity-adjusted trust)"
    return report


# =============================================================================
# fake-selection positive control  (the realness surface — W125/W128 analogue)
# =============================================================================
def fake_selection_control_v1() -> dict:
    """A degenerate selector that commits by ALPHABETICAL-first while IGNORING all public
    signal MUST be flagged ``evidence_used = False`` (no discriminating signal drove it) — the
    exact W128 ``pawnshop`` mis-commit pattern.  A real SO commit cites a discriminating signal
    (a falsifier eliminated a survivor, a strict majority, or a verifier choice)."""
    # two STRUCTURALLY-DISTINCT but behaviourally-identical survivors == the pawnshop danger
    # (they agree on every public-derivable signal; only the hidden tests would separate them).
    prob = IcpcPilotProblemV1(problem_id="ctl", short_name="ctl", source_repo="ctl",
                              contest_date="2020-01-01", statement="s", kind="passfail",
                              float_tol=0.0, samples=(("1\n", "1\n"),), secret_cases=())
    impls = [CandidateImplV1("A0", "print(1)", True),
             CandidateImplV1("B1", "x = 1\nprint(x)", True)]  # different AST, same output
    arts = RoleArtifactsV1(spec="s", invariants=("x>=0",), complexity="O(1)",
                           sketches=(), counterexamples=(("1\n", None),), raw="")
    # SO2/SO4 must ABSTAIN (no discriminator); a naive alphabetical commit has evidence_used=False
    s2 = select_so_v1(prob, impls, arts, variant="SO2")
    s4 = select_so_v1(prob, impls, arts, variant="SO4")
    return {"schema": "coordpy.w129_fake_selection_control.v1",
            "so2_abstained_on_tie": s2.abstained, "so2_evidence_used": s2.evidence_used,
            "so4_abstained_on_tie": s4.abstained, "so4_evidence_used": s4.evidence_used,
            "control_passes": (s2.abstained and not s2.evidence_used
                               and s4.abstained and not s4.evidence_used)}
