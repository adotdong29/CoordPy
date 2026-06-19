"""W132 / COO-9 — CoordPy-OWNED resistant-by-construction algorithmic battlefield.

The W123→W131 chain established that the third-retirement search is blocked on the
RESISTANT side by *supply* (W123 official-package supply cap), *encoder supply* (W124),
*re-routing* (W125), *deterministic synthesis* (W126), *scaffold transfer* (W127),
*selection* (W128/W129), *generation* (W130), and *model disclosure* (W131).  Every one
of those caps shares a single upstream dependency: a contamination-RESISTANT battlefield
we either inherit (official ICPC, supply-capped) or certify against a model whose cutoff
is disclosed (cutoff-disclosure-capped).  W132 removes that dependency by *minting* the
battlefield itself.

This module is the Lane-α core.  It mints a battlefield of **resistant-by-construction**
stdin/stdout algorithmic problems, each:

* freshly generated (a problem instance that did not exist before the mint date — so no
  model of any cutoff can have memorised the *answer*; contamination-resistant by
  construction, not merely by inherited date), and
* **targeted at the failure families that actually beat the mechanism stack** in the
  W130/W131 generator-failure atlas: ``WRONG_ALGORITHM_ADMISSIBLE`` (a plausible named
  technique that is the wrong algorithm), ``HIDDEN_EDGE_STATE_MISS`` (public samples
  underspecify the corner), ``COMPLEXITY_BLIND`` (the obvious approach is asymptotically
  too slow), and ``SEARCH_ENUM`` (a small-n exhaustive oracle is exact and a scalable
  reference is cross-checked against it).

Each minted problem ships THREE independent executable programs so correctness is
machine-checked, never asserted:

* ``ref_source``   — the scalable CORRECT program; its stdout IS the official answer key.
* ``brute_source`` — an INDEPENDENT, obviously-correct exhaustive oracle, run on the
  small cases to cross-check ``ref_source`` (the small-vs-large agreement gate).
* ``naive_source`` — the ADMISSIBLE-WRONG trap (a named-but-wrong algorithm, or the
  obvious-but-too-slow algorithm); it must PASS every public sample (so it "looks right")
  and FAIL at least one hidden case (so the battlefield genuinely *resists* it).

The model under test (Lane β) is shown ONLY the statement + public samples — never
``ref_source``/``naive_source``/the hidden cases — and is graded by the SAME audited
``grade_icpc_candidate_case_v1`` token-diff / float oracle the official-ICPC bench uses,
exit-0-iff-EVERY-hidden-case-passes, NO LLM judge.  The minted problems are emitted as
``IcpcPilotProblemV1`` so the *already-validated* W120 reflexion bench (the W88/W89
mechanism that earned the only two retirements) consumes them verbatim.

Anti-cheat / honesty (LOCKED, enforced in code — see ``docs/RUNBOOK_W132.md``):

* the answer key is an executable program's output, not a hand-written constant;
* every admitted problem passes the brute-vs-ref agreement gate on its small cases;
* every admitted problem passes the discriminating-hidden-case gate (naive looks-right /
  fails-hidden) — a problem the naive trap *also* solves is NOT admitted (it does not
  resist);
* a mechanical near-duplicate / novelty guard drops any problem whose statement is too
  close to another minted problem or to an official ICPC identity;
* deterministic regeneration from the mint seed (a content-addressed manifest CID);
* pass-fail-only grading (tier-1 token-diff / tier-2 deterministic float), no scoring.

Reuses (explicit-import-only, NO duplication): ``IcpcPilotProblemV1`` +
``grade_icpc_candidate_case_v1`` + ``judge_icpc_output_v1`` + ``MIN_RESISTANT_SLICE`` +
the ``KIND_*`` tags + the W114 ``certify_model_v1`` C1..C4 gate (as RESISTANCE
corroboration).  Pure / deterministic / read-only except the answer-key subprocess (the
only code execution; no model inference — that lives in the W132 pilot SCRIPT).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import random
import subprocess
import sys
from typing import Any, Callable, Optional, Sequence

from .coordpy_icpc_battlefield_v1 import (
    KIND_PASSFAIL,
    KIND_PASSFAIL_FLOAT,
    judge_icpc_output_v1,
)
from .icpc_reflexion_bench_v1 import IcpcPilotProblemV1
from .livecodebench_resistant_slice_v1 import MIN_RESISTANT_SLICE
from .stronger_model_cutoff_certification_v1 import (
    LatestResistantInstrumentV1,
    STRONGER_MODEL_CANDIDATES,
    certify_model_v1,
)
from .coordpy_icpc_public_functional_v1 import MAVERICK_CUTOFF_BOUNDARY

W132_RBC_BATTLEFIELD_V1_SCHEMA_VERSION: str = (
    "coordpy.resistant_by_construction_battlefield_v1.v1")

RBC_BATTLEFIELD_INSTRUMENT_ID: str = "coordpy_rbc_battlefield_v1"

# ---- targeted failure families (named to match the W130/W131 atlas modes) --------
MODE_WRONG_ALGORITHM: str = "WRONG_ALGORITHM_ADMISSIBLE"
MODE_HIDDEN_EDGE: str = "HIDDEN_EDGE_STATE_MISS"
MODE_COMPLEXITY_BLIND: str = "COMPLEXITY_BLIND"
MODE_SEARCH_ENUM: str = "SEARCH_ENUM"
TARGET_MODES: tuple[str, ...] = (
    MODE_WRONG_ALGORITHM, MODE_HIDDEN_EDGE, MODE_COMPLEXITY_BLIND, MODE_SEARCH_ENUM)

# ---- how the naive trap is expected to be separated by the hidden cases -----------
DISC_OUTPUT_MISMATCH: str = "OUTPUT_MISMATCH"   # naive returns a WRONG answer
DISC_TIMEOUT: str = "TIMEOUT"                   # naive is correct-but-too-slow (TLE)

# Mint-date cutoff boundary the answer-key subprocess is bounded by.  Sized so a correct
# O(N log N)/O(N) reference finishes with a wide margin while an O(N^2) naive TLEs on the
# large complexity stress case at the SAME budget the pilot grades under.
DEFAULT_EXEC_TIMEOUT_S: float = 8.0
# Near-duplicate ceiling: statement char-5-gram Jaccard >= this rejects the later problem.
NOVELTY_JACCARD_MAX: float = 0.55


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


# ===================================================== the only code-execution path

@dataclasses.dataclass(frozen=True)
class ExecCaptureV1:
    stdout: str
    returncode: int
    timed_out: bool


def _exec_capture_v1(source: str, stdin_text: str, *,
                     timeout_s: float = DEFAULT_EXEC_TIMEOUT_S,
                     python_exe: Optional[str] = None) -> ExecCaptureV1:
    """Run a complete stdin/stdout program in a fresh isolated subprocess and CAPTURE its
    stdout (used to mint the answer key + run the brute oracle).  Mirrors the audited
    ``run_icpc_stdin_executor_v1`` execution shape (``python -I -c``) but returns stdout."""
    py = python_exe or sys.executable
    try:
        proc = subprocess.run([py, "-I", "-c", source],
                              input=stdin_text.encode("utf-8"),
                              capture_output=True, timeout=float(timeout_s),
                              check=False)
    except subprocess.TimeoutExpired:
        return ExecCaptureV1("", -9, True)
    return ExecCaptureV1(
        proc.stdout.decode("utf-8", "replace"), int(proc.returncode), False)


# ===================================================== the minted-problem template

@dataclasses.dataclass(frozen=True)
class MintedTemplateV1:
    """A resistant-by-construction problem GENERATOR (one algorithmic identity).

    ``gen_public``/``gen_hidden`` map a seeded ``random.Random`` to the list of stdin
    strings; the answer key for each is the captured stdout of ``ref_source``.  ``brute_
    source`` is the independent oracle (run only on inputs with <= ``brute_cap_tokens``
    whitespace tokens, so an exponential brute is never run on a large case).
    """

    name: str
    family: str            # short algorithm-family label (NOT an official problem name)
    mode: str              # one of TARGET_MODES
    kind: str              # KIND_PASSFAIL | KIND_PASSFAIL_FLOAT
    float_tol: float
    statement: str
    ref_source: str
    naive_source: str
    brute_source: str
    algo_sig: str          # the correct technique tag (diversity accounting)
    discriminator: str     # DISC_OUTPUT_MISMATCH | DISC_TIMEOUT
    brute_cap_tokens: int
    gen_public: Callable[[random.Random], list[str]]
    gen_hidden: Callable[[random.Random], list[str]]


# ===================================================== minted problem + gate record

@dataclasses.dataclass(frozen=True)
class ProblemGateRecordV1:
    g_passfail_only: bool
    g_reference_solvable: bool
    g_oracle_small_agreement: bool
    n_brute_checked: int
    g_discriminating: bool
    naive_passes_all_public: bool
    n_naive_secret_fail: int
    naive_fail_kinds: tuple[str, ...]
    g_split_integrity: bool
    admitted: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"g_passfail_only": self.g_passfail_only,
                "g_reference_solvable": self.g_reference_solvable,
                "g_oracle_small_agreement": self.g_oracle_small_agreement,
                "n_brute_checked": int(self.n_brute_checked),
                "g_discriminating": self.g_discriminating,
                "naive_passes_all_public": self.naive_passes_all_public,
                "n_naive_secret_fail": int(self.n_naive_secret_fail),
                "naive_fail_kinds": list(self.naive_fail_kinds),
                "g_split_integrity": self.g_split_integrity,
                "admitted": bool(self.admitted), "reason": self.reason}


@dataclasses.dataclass(frozen=True)
class MintedProblemV1:
    problem_id: str
    name: str
    family: str
    mode: str
    kind: str
    float_tol: float
    statement: str
    samples: tuple[tuple[str, str], ...]
    secret_cases: tuple[tuple[str, str], ...]
    ref_source: str
    naive_source: str
    brute_source: str
    algo_sig: str
    discriminator: str
    ref_max_runtime_s: float
    gates: ProblemGateRecordV1

    def to_pilot_problem(self, *, minted_date: str) -> IcpcPilotProblemV1:
        """The model-facing object: statement + PUBLIC samples + the hidden GRADER.  The
        reference / naive / brute sources are NOT included (never leaked to the model)."""
        return IcpcPilotProblemV1(
            problem_id=self.problem_id, short_name=self.name,
            source_repo=RBC_BATTLEFIELD_INSTRUMENT_ID, contest_date=str(minted_date),
            statement=self.statement, kind=self.kind, float_tol=float(self.float_tol),
            samples=self.samples, secret_cases=self.secret_cases)

    def content_cid(self) -> str:
        return _sha256_hex({"kind": "rbc_minted_problem_v1", "name": self.name,
                            "mode": self.mode, "statement": self.statement,
                            "samples": [list(s) for s in self.samples],
                            "secret_cases": [list(s) for s in self.secret_cases],
                            "ref_source": self.ref_source})

    def to_dict(self) -> dict[str, Any]:
        return {"problem_id": self.problem_id, "name": self.name, "family": self.family,
                "mode": self.mode, "kind": self.kind, "float_tol": float(self.float_tol),
                "algo_sig": self.algo_sig, "discriminator": self.discriminator,
                "n_samples": len(self.samples), "n_secret": len(self.secret_cases),
                "statement_len": len(self.statement),
                "ref_max_runtime_s": round(float(self.ref_max_runtime_s), 3),
                "content_cid": self.content_cid(), "gates": self.gates.to_dict()}


def _problem_id(name: str) -> str:
    return f"rbc_{name}"


def _tok_count(s: str) -> int:
    return len(s.split())


def mint_problem_v1(template: MintedTemplateV1, *, global_seed: int,
                    timeout_s: float = DEFAULT_EXEC_TIMEOUT_S) -> MintedProblemV1:
    """Deterministically mint one problem from a template: generate the seeded inputs,
    compute the answer key as ``ref_source`` stdout, and run the per-problem gates."""
    rng = random.Random(_sha256_hex({"seed": int(global_seed), "name": template.name}))
    pub_inputs = list(template.gen_public(rng))
    hid_inputs = list(template.gen_hidden(rng))

    # ---- answer key = ref_source stdout (the only oracle); also the solvability gate ----
    def _key(inp: str) -> tuple[str, float, bool, int]:
        r = _exec_capture_v1(template.ref_source, inp, timeout_s=timeout_s)
        return (r.stdout, 0.0, r.timed_out, r.returncode)

    ref_ok = True
    ref_max_rt = 0.0
    samples: list[tuple[str, str]] = []
    secret: list[tuple[str, str]] = []
    for inp in pub_inputs:
        out, _, to, rc = _key(inp)
        if to or rc != 0:
            ref_ok = False
        samples.append((inp, out))
    for inp in hid_inputs:
        out, _, to, rc = _key(inp)
        if to or rc != 0:
            ref_ok = False
        secret.append((inp, out))
    # secret = PUBLIC ∪ HIDDEN (a real judge runs samples too); samples ⊂ secret.
    full_secret = samples + secret

    gates = _evaluate_problem_gates_v1(
        template, samples=samples, secret=tuple(full_secret),
        ref_solvable=ref_ok, timeout_s=timeout_s)

    return MintedProblemV1(
        problem_id=_problem_id(template.name), name=template.name,
        family=template.family, mode=template.mode, kind=template.kind,
        float_tol=float(template.float_tol), statement=template.statement,
        samples=tuple(samples), secret_cases=tuple(full_secret),
        ref_source=template.ref_source, naive_source=template.naive_source,
        brute_source=template.brute_source, algo_sig=template.algo_sig,
        discriminator=template.discriminator, ref_max_runtime_s=float(ref_max_rt),
        gates=gates)


def _evaluate_problem_gates_v1(template: MintedTemplateV1, *,
                               samples: Sequence[tuple[str, str]],
                               secret: Sequence[tuple[str, str]],
                               ref_solvable: bool,
                               timeout_s: float) -> ProblemGateRecordV1:
    kind = template.kind
    g_passfail = kind in (KIND_PASSFAIL, KIND_PASSFAIL_FLOAT)

    # ---- small-vs-large agreement: independent brute oracle == ref answer key ----------
    n_checked = 0
    g_oracle = True
    for inp, out in secret:
        if _tok_count(inp) > int(template.brute_cap_tokens):
            continue
        b = _exec_capture_v1(template.brute_source, inp, timeout_s=timeout_s)
        n_checked += 1
        if b.timed_out or b.returncode != 0 or not judge_icpc_output_v1(
                got_stdout=b.stdout, expected=out, kind=kind,
                float_tol=float(template.float_tol)):
            g_oracle = False
    if n_checked == 0:           # a non-vacuous cross-check is required
        g_oracle = False

    # ---- discriminating gate: naive looks-right on public, fails >=1 hidden ------------
    naive_pub_ok = True
    for inp, out in samples:
        r = _grade(template.naive_source, inp, out, kind, template.float_tol, timeout_s)
        if not r[0]:
            naive_pub_ok = False
    n_naive_fail = 0
    fail_kinds: list[str] = []
    sample_set = {s[0] for s in samples}
    for inp, out in secret:
        if inp in sample_set:
            continue
        ok, to = _grade(template.naive_source, inp, out, kind, template.float_tol,
                        timeout_s)
        if not ok:
            n_naive_fail += 1
            fail_kinds.append("TIMEOUT" if to else "WRONG_ANSWER")
    want = "TIMEOUT" if template.discriminator == DISC_TIMEOUT else "WRONG_ANSWER"
    g_disc = bool(naive_pub_ok and n_naive_fail >= 1 and want in fail_kinds)

    # ---- public/hidden split integrity ----
    g_split = bool(sample_set and sample_set.issubset({s[0] for s in secret})
                   and len(secret) > len(samples))

    admitted = bool(g_passfail and ref_solvable and g_oracle and g_disc and g_split)
    if admitted:
        reason = "ADMITTED"
    elif not ref_solvable:
        reason = "REF_NOT_SOLVABLE_IN_BUDGET"
    elif not g_oracle:
        reason = f"ORACLE_DISAGREEMENT_OR_NO_BRUTE_CHECK(n={n_checked})"
    elif not g_disc:
        reason = (f"NOT_DISCRIMINATING(naive_pub_ok={naive_pub_ok},"
                  f"fail={n_naive_fail},want={want},got={sorted(set(fail_kinds))})")
    elif not g_split:
        reason = "SPLIT_INTEGRITY"
    else:
        reason = "PASSFAIL_ONLY"
    return ProblemGateRecordV1(
        g_passfail_only=g_passfail, g_reference_solvable=bool(ref_solvable),
        g_oracle_small_agreement=g_oracle, n_brute_checked=int(n_checked),
        g_discriminating=g_disc, naive_passes_all_public=bool(naive_pub_ok),
        n_naive_secret_fail=int(n_naive_fail), naive_fail_kinds=tuple(fail_kinds),
        g_split_integrity=g_split, admitted=admitted, reason=reason)


def _grade(source: str, stdin_text: str, expected: str, kind: str,
           float_tol: float, timeout_s: float) -> tuple[bool, bool]:
    """Return (passed, timed_out) for a candidate program on one case (token-diff/float)."""
    r = _exec_capture_v1(source, stdin_text, timeout_s=timeout_s)
    if r.timed_out:
        return (False, True)
    if r.returncode != 0:
        return (False, False)
    return (bool(judge_icpc_output_v1(got_stdout=r.stdout, expected=expected,
                                      kind=kind, float_tol=float(float_tol))), False)


# ===================================================== novelty / near-duplicate guard

def _char_ngrams(s: str, n: int = 5) -> set[str]:
    t = " ".join(str(s).lower().split())
    return {t[i:i + n] for i in range(max(0, len(t) - n + 1))} or {t}


def statement_jaccard_v1(a: str, b: str) -> float:
    ga, gb = _char_ngrams(a), _char_ngrams(b)
    inter = len(ga & gb)
    union = len(ga | gb)
    return float(inter / union) if union else 0.0


@dataclasses.dataclass(frozen=True)
class NoveltyRejectionV1:
    problem_id: str
    nearest_id: str
    jaccard: float

    def to_dict(self) -> dict[str, Any]:
        return {"problem_id": self.problem_id, "nearest_id": self.nearest_id,
                "jaccard": round(float(self.jaccard), 4)}


def novelty_filter_v1(problems: Sequence[MintedProblemV1], *,
                      official_identities: Sequence[str] = (),
                      jaccard_max: float = NOVELTY_JACCARD_MAX,
                      ) -> tuple[tuple[MintedProblemV1, ...], tuple[NoveltyRejectionV1, ...]]:
    """Mechanical near-duplicate guard.  A problem is rejected iff (a) its statement
    char-5-gram Jaccard with an already-accepted problem >= ``jaccard_max``, or (b) its
    statement contains an official ICPC identity token (paraphrase guard)."""
    accepted: list[MintedProblemV1] = []
    rejected: list[NoveltyRejectionV1] = []
    off = {str(x).lower() for x in official_identities}
    for p in problems:
        low = p.statement.lower()
        off_hit = next((o for o in off if o and len(o) >= 6 and o in low), "")
        if off_hit:
            rejected.append(NoveltyRejectionV1(p.problem_id, f"official:{off_hit}", 1.0))
            continue
        worst_id, worst = "", 0.0
        for q in accepted:
            j = statement_jaccard_v1(p.statement, q.statement)
            if j > worst:
                worst_id, worst = q.problem_id, j
        if worst >= float(jaccard_max):
            rejected.append(NoveltyRejectionV1(p.problem_id, worst_id, worst))
        else:
            accepted.append(p)
    return tuple(accepted), tuple(rejected)


# ===================================================== Maverick resistance certification

@dataclasses.dataclass(frozen=True)
class RbcResistanceCertV1:
    model_id: str
    minted_date: str
    model_cutoff_boundary: str
    minted_after_cutoff: bool         # date-resistant (could not have trained on it)
    novel_by_construction: bool       # instance did not exist => resistant for ANY cutoff
    reused_gate_certifiable: bool     # corroboration via the W114 C1..C4 gate
    resistant: bool
    note: str

    def to_dict(self) -> dict[str, Any]:
        return {"model_id": self.model_id, "minted_date": self.minted_date,
                "model_cutoff_boundary": self.model_cutoff_boundary,
                "minted_after_cutoff": self.minted_after_cutoff,
                "novel_by_construction": self.novel_by_construction,
                "reused_gate_certifiable": self.reused_gate_certifiable,
                "resistant": self.resistant, "note": self.note}


def certify_resistance_v1(*, model_id: str, minted_date: str, n_core: int,
                          raw_cid: str,
                          model_cutoff_boundary: str = MAVERICK_CUTOFF_BOUNDARY,
                          ) -> RbcResistanceCertV1:
    """Resistance is established TWO ways and corroborated by the audited W114 gate:

    (1) date: ``minted_date`` strictly post-dates the model's KNOWN cutoff, so the model
        could not have trained on these problems; and
    (2) construction: the *instances* are freshly generated and did not exist at any
        training time — resistant for ANY cutoff (incl. UNKNOWN-cutoff models), which the
        official-ICPC inherited battlefields could not claim.
    """
    after = bool(str(minted_date) > str(model_cutoff_boundary))
    novel = True  # freshly-minted instances by construction (the mint seed is post-hoc)
    reused = False
    try:
        instrument = LatestResistantInstrumentV1(
            release=RBC_BATTLEFIELD_INSTRUMENT_ID, jsonl_sha256=str(raw_cid),
            n_functional=int(n_core), functional_date_min=str(minted_date),
            functional_date_max=str(minted_date),
            functional_month_histogram={str(minted_date)[:7]: int(n_core)},
            note="CoordPy-minted resistant-by-construction battlefield")
        cand = next((c for c in STRONGER_MODEL_CANDIDATES
                     if model_id in (c.model_id, getattr(c, "model_id", ""))), None)
        if cand is not None:
            cand = dataclasses.replace(cand, already_settled_on_instrument=False)
            cert = certify_model_v1(cand, instrument=instrument)
            reused = bool(cert.certifiable_for_new_pilot)
    except Exception:
        reused = False
    resistant = bool(after and novel)
    note = (f"RESISTANT_BY_CONSTRUCTION: minted {minted_date} > cutoff "
            f"{model_cutoff_boundary} (date) AND freshly-minted instances (construction)"
            if resistant else "NOT_RESISTANT")
    return RbcResistanceCertV1(
        model_id=str(model_id), minted_date=str(minted_date),
        model_cutoff_boundary=str(model_cutoff_boundary), minted_after_cutoff=after,
        novel_by_construction=novel, reused_gate_certifiable=reused,
        resistant=resistant, note=note)


# ===================================================== battlefield manifest + build

@dataclasses.dataclass(frozen=True)
class RbcManifestV1:
    schema: str
    instrument_id: str
    minted_date: str
    global_seed: int
    n_templates: int
    n_minted: int
    n_gate_pass: int
    n_admitted: int
    admitted_problem_ids: tuple[str, ...]
    mode_histogram: dict[str, int]
    family_histogram: dict[str, int]
    raw_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.schema, "instrument_id": self.instrument_id,
                "minted_date": self.minted_date, "global_seed": int(self.global_seed),
                "n_templates": int(self.n_templates), "n_minted": int(self.n_minted),
                "n_gate_pass": int(self.n_gate_pass), "n_admitted": int(self.n_admitted),
                "admitted_problem_ids": list(self.admitted_problem_ids),
                "mode_histogram": dict(self.mode_histogram),
                "family_histogram": dict(self.family_histogram),
                "raw_cid": self.raw_cid}

    def manifest_cid(self) -> str:
        return _sha256_hex({"kind": "rbc_battlefield_manifest_v1",
                            "instrument_id": self.instrument_id,
                            "minted_date": self.minted_date,
                            "admitted_problem_ids": list(self.admitted_problem_ids),
                            "raw_cid": self.raw_cid})


@dataclasses.dataclass(frozen=True)
class RbcBattlefieldV1:
    schema: str
    minted_date: str
    manifest: RbcManifestV1
    problems: tuple[MintedProblemV1, ...]            # admitted, novelty-clean, ordered
    all_minted: tuple[MintedProblemV1, ...]          # every minted (incl. rejected)
    novelty_rejected: tuple[NoveltyRejectionV1, ...]
    meets_min_slice: bool

    def admitted(self) -> tuple[MintedProblemV1, ...]:
        return self.problems

    def to_pilot_problems(self) -> list[IcpcPilotProblemV1]:
        return [p.to_pilot_problem(minted_date=self.minted_date) for p in self.problems]

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.schema, "minted_date": self.minted_date,
                "manifest": self.manifest.to_dict(),
                "manifest_cid": self.manifest.manifest_cid(),
                "meets_min_slice": bool(self.meets_min_slice),
                "n_novelty_rejected": len(self.novelty_rejected),
                "novelty_rejected": [r.to_dict() for r in self.novelty_rejected],
                "problems": [p.to_dict() for p in self.problems],
                "rejected_problems": [p.to_dict() for p in self.all_minted
                                      if not p.gates.admitted]}


def _raw_cid_for(problems: Sequence[MintedProblemV1]) -> str:
    return _sha256_hex({"kind": "rbc_raw_cid_v1",
                        "content_cids": sorted(p.content_cid() for p in problems)})


def mint_battlefield_v1(slate: Sequence[MintedTemplateV1], *, global_seed: int,
                        minted_date: str,
                        timeout_s: float = DEFAULT_EXEC_TIMEOUT_S,
                        official_identities: Sequence[str] = (),
                        min_slice: int = MIN_RESISTANT_SLICE) -> RbcBattlefieldV1:
    """Mint every template, run the per-problem gates, apply the novelty filter to the
    gate-passing problems, and assemble the deterministic manifest."""
    minted = [mint_problem_v1(t, global_seed=global_seed, timeout_s=timeout_s)
              for t in slate]
    gate_pass = [p for p in minted if p.gates.admitted]
    # deterministic order so the manifest CID is stable (mode, then name)
    gate_pass.sort(key=lambda p: (p.mode, p.name))
    admitted, rejected = novelty_filter_v1(
        gate_pass, official_identities=official_identities)

    mode_hist: dict[str, int] = {}
    fam_hist: dict[str, int] = {}
    for p in admitted:
        mode_hist[p.mode] = mode_hist.get(p.mode, 0) + 1
        fam_hist[p.family] = fam_hist.get(p.family, 0) + 1
    raw = _raw_cid_for(admitted)
    manifest = RbcManifestV1(
        schema=W132_RBC_BATTLEFIELD_V1_SCHEMA_VERSION,
        instrument_id=RBC_BATTLEFIELD_INSTRUMENT_ID, minted_date=str(minted_date),
        global_seed=int(global_seed), n_templates=len(list(slate)),
        n_minted=len(minted), n_gate_pass=len(gate_pass), n_admitted=len(admitted),
        admitted_problem_ids=tuple(p.problem_id for p in admitted),
        mode_histogram=dict(sorted(mode_hist.items())),
        family_histogram=dict(sorted(fam_hist.items())), raw_cid=raw)
    return RbcBattlefieldV1(
        schema=W132_RBC_BATTLEFIELD_V1_SCHEMA_VERSION, minted_date=str(minted_date),
        manifest=manifest, problems=tuple(admitted), all_minted=tuple(minted),
        novelty_rejected=tuple(rejected),
        meets_min_slice=bool(len(admitted) >= int(min_slice)))


def select_core_slice_v1(battlefield: RbcBattlefieldV1, *, n_problems: int = 30,
                         ) -> tuple[MintedProblemV1, ...]:
    """Outcome-blind, deterministic, mode-stratified core slice (spans all 4 families;
    largest-remainder apportionment, key tie-break) so the pilot is never all-one-family."""
    probs = list(battlefield.problems)
    strata: dict[str, list[MintedProblemV1]] = {}
    for p in probs:
        strata.setdefault(p.mode, []).append(p)
    for k in strata:
        strata[k].sort(key=lambda p: p.name)
    counts = {k: len(v) for k, v in strata.items()}
    n = min(int(n_problems), len(probs))
    s = sum(counts.values()) or 1
    raw = {k: (n * v / s) for k, v in counts.items()}
    base = {k: int(x) for k, x in raw.items()}
    rem = int(n) - sum(base.values())
    order = sorted(counts, key=lambda k: (-(raw[k] - base[k]), k))
    for k in order[:max(0, rem)]:
        base[k] += 1
    chosen: list[MintedProblemV1] = []
    for k in sorted(strata):
        chosen.extend(strata[k][:base.get(k, 0)])
    if len(chosen) < n:
        ids = {p.problem_id for p in chosen}
        rest = sorted((p for p in probs if p.problem_id not in ids),
                      key=lambda p: (p.mode, p.name))
        chosen.extend(rest[:n - len(chosen)])
    chosen = chosen[:n]
    chosen.sort(key=lambda p: (p.mode, p.name))
    return tuple(chosen)


def core_slice_cid_v1(slice_problems: Sequence[MintedProblemV1]) -> str:
    return _sha256_hex({"kind": "rbc_core_slice_v1",
                        "problem_ids": [p.problem_id for p in slice_problems]})


__all__ = [
    "W132_RBC_BATTLEFIELD_V1_SCHEMA_VERSION", "RBC_BATTLEFIELD_INSTRUMENT_ID",
    "MODE_WRONG_ALGORITHM", "MODE_HIDDEN_EDGE", "MODE_COMPLEXITY_BLIND",
    "MODE_SEARCH_ENUM", "TARGET_MODES",
    "DISC_OUTPUT_MISMATCH", "DISC_TIMEOUT",
    "DEFAULT_EXEC_TIMEOUT_S", "NOVELTY_JACCARD_MAX",
    "ExecCaptureV1", "_exec_capture_v1",
    "MintedTemplateV1", "ProblemGateRecordV1", "MintedProblemV1",
    "mint_problem_v1", "statement_jaccard_v1",
    "NoveltyRejectionV1", "novelty_filter_v1",
    "RbcResistanceCertV1", "certify_resistance_v1",
    "RbcManifestV1", "RbcBattlefieldV1", "mint_battlefield_v1",
    "select_core_slice_v1", "core_slice_cid_v1",
]
