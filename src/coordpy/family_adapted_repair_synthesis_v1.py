"""W126 — Family-Adapted Repair Synthesis V1 (COO-9 sibling).

W125 proved the resistant ICPC field is GENERATION-CAPPED for $0 *re-routing*: a
hidden-test-blind controller selecting among the 11 already-paid Maverick generations
recovers ZERO A1-fail problems (``blind_selection_headroom = 0``; pool-union 8/30).
W125 explicitly did NOT try to **create new trajectories**.  W126 does.

The question is no longer "can the controller choose better among the same 11
generations?" but: **can CoordPy synthesize genuinely NEW resistant-code candidates from
the already-paid pool plus an official-family teacher corpus, without leaking target
answers?**  This module wires the otherwise-unused repair/consensus/policy arsenal
(``adversarial_consensus_repair_v1.trust_weighted_consensus_v1``,
``executor_grounded_patcher_v1.parse_failure_digest_v1``,
``compose_repair_integrity_pipeline_v1`` integrity anchoring) onto the official-ICPC
stdin/stdout code path (graphify-confirmed the controller path and the
compose/consensus arsenal were 4 hops apart through a trivial ``str`` node — NO real
semantic edge; this module is the bridge).

Held OUTSIDE the stable SDK contract: explicit-import-only, ``coordpy/__init__.py``
untouched, ``coordpy.__version__ == "0.5.20"``, no PyPI publish.

New-trajectory synthesis slate (RUNBOOK_W126 § 4):

* **S1 cross-candidate splice** — AST-level recombination of the 11 generations
  (function-def swaps, I/O-prefix / compute / output-suffix block swaps) → new programs.
* **S2 digest-grounded repair** — the typed executor digest (``parse_failure_digest_v1``)
  routes deterministic micro-repairs (robust stdin tokenizer, recursionlimit, yes/no
  casing, output normalisation) → new programs; proposals fused by trust-weighted
  consensus.
* **S3 family-motif harden** — idioms mined from the EXPOSED-side accepted solutions of
  OTHER problems (a non-target teacher corpus) applied as hardening transforms.
* **S-CONS output-consensus dispatcher** — a genuinely new program that, per input,
  runs the trusted generations and emits the trust-weighted plurality output (the lever
  W125 could not express: it can pass a problem whose generations are each correct on a
  DIFFERENT subset of cases).  Trust weights come from
  ``trust_weighted_consensus_v1`` on the blind generation scores.
* **S4 learned repair-action policy** — conditional on a large enough labelled corpus;
  otherwise registered NOT_WARRANTED (W124 precedent: chance on n≈14 rescue events).

NO synthesis input ever reads a target problem's secret cases or its own
``submissions/`` accepted solution.  The teacher corpus is EXPOSED-side, problem-disjoint.
The official secret grader scores only a committed candidate's COMMITTED answer; selection
is hidden-test-BLIND.  These are the properties that keep the synthesis honest and
hosted-translatable.
"""
from __future__ import annotations

import ast
import dataclasses
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
from typing import Any, Optional, Sequence

import numpy as _np

# --- the audited grader plane + blind score + pool (reuse the W125 substrate) ----------
from .controller_native_code_mechanism_v1 import (
    AuditedGraderPlaneV1,
    BlindCandidateScoreV1,
    PerProblemPoolV1,
    _code_norm_sha,
    _digest_key,
    _parses,
    _sha256_hex,
)
# --- the underused repair/consensus arsenal (creates the bridge edges) -----------------
from .adversarial_consensus_repair_v1 import (
    TrustWeightedConsensusConfigV1,
    WitnessEvidenceV1,
    trust_weighted_consensus_v1,
)
from .executor_grounded_patcher_v1 import FailureDigestV1, parse_failure_digest_v1
# --- the official-ICPC code path -------------------------------------------------------
from .icpc_reflexion_bench_v1 import (
    IcpcPilotProblemV1,
    grade_on_secret_v1,
    sample_feedback_v1,
)
from .coordpy_icpc_battlefield_v1 import KIND_PASSFAIL, judge_icpc_output_v1

W126_FAMILY_ADAPTED_REPAIR_SYNTHESIS_V1_SCHEMA_VERSION: str = (
    "coordpy.family_adapted_repair_synthesis_v1.v1")

EXPOSED_TEACHER_ROOT_DEFAULT = "/tmp/w121_icpc"  # W121 pre-cutoff exposed package cache


# ============================================================ no-leakage guard

@dataclasses.dataclass(frozen=True)
class LeakageVerdictV1:
    clean: bool
    reason: str


class SynthesisLeakageGuardV1:
    """Hard no-leakage boundary for one resistant target (RUNBOOK_W126 § 2).

    A synthesis input is CLEAN iff it carries no secret ``.in``/``.ans`` byte-run of the
    target AND no byte-run of the target's OWN accepted solution.  The target's
    ``submissions/`` and ``data/secret/`` are never opened by the synthesiser; this guard
    is the verifiable positive control (a planted secret/answer is caught).
    """

    def __init__(self, problem: IcpcPilotProblemV1,
                 target_accepted_texts: Sequence[str] = (),
                 provenance_texts: Sequence[str] = ()) -> None:
        self.problem = problem
        # PROVENANCE = the source material synthesis is allowed to recombine (the
        # already-paid generations + public samples).  A secret byte-run that ALREADY
        # appears in the provenance is NOT a leak — the base model wrote it WITHOUT secret
        # access (e.g. an emoticons generation literal that coincides with a secret
        # answer).  A secret run present in a candidate but ABSENT from provenance is the
        # real injection signature (an accidental secret-file read).
        self._provenance = "\n".join(str(t) for t in provenance_texts)
        pub = str(problem.statement) + "".join(
            str(i) + str(o) for i, o in problem.samples)
        # secret-only answer/input runs (>=3 chars, not already in the public surface) +
        # full secret-case concatenations (always long ⇒ the positive control always bites)
        runs: set = set()
        for inp, ans in problem.secret_cases:
            for s in (ans.strip(), inp.strip()):
                if s and len(s) >= 3 and s not in pub:
                    runs.add(s)
            full = (inp.strip() + "\n" + ans.strip()).strip()
            if len(full) >= 6 and full not in pub:
                runs.add(full)
        self._secret_runs = tuple(sorted(runs, key=len, reverse=True))
        # the target's OWN accepted-solution lines (forbidden answer material)
        self._accepted_runs = tuple(sorted({
            ln.strip() for t in target_accepted_texts for ln in str(t).splitlines()
            if len(ln.strip()) >= 12}, key=len, reverse=True))

    def set_provenance(self, provenance_texts: Sequence[str]) -> None:
        self._provenance = "\n".join(str(t) for t in provenance_texts)

    def check(self, text: str) -> LeakageVerdictV1:
        t = str(text or "")
        for run in self._secret_runs:
            # a secret run already present in provenance is provenance-clean (coincidence,
            # not injection); only a run ABSENT from provenance is a real leak signature
            if run and run in t and run not in self._provenance:
                return LeakageVerdictV1(False, f"secret byte-run injected (len {len(run)})")
        for run in self._accepted_runs:
            if run and run in t and run not in self._provenance:
                return LeakageVerdictV1(False, "target accepted-solution line injected")
        return LeakageVerdictV1(True, "clean")

    def all_clean(self, texts: Sequence[str]) -> bool:
        return all(self.check(t).clean for t in texts)


# ============================================================ exposed teacher corpus (S3)

@dataclasses.dataclass(frozen=True)
class TeacherSolutionV1:
    problem_short: str
    path_sha: str
    code: str


@dataclasses.dataclass(frozen=True)
class FamilyMotifsV1:
    """Data-driven hardening idioms mined from the exposed teacher corpus."""
    schema: str
    n_solutions: int
    n_problems: int
    idiom_freq: dict
    corpus_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.schema, "n_solutions": self.n_solutions,
                "n_problems": self.n_problems, "idiom_freq": dict(self.idiom_freq),
                "corpus_cid": self.corpus_cid}


_IDIOM_PATTERNS = {
    "fast_stdin_read": r"sys\.stdin\.(buffer\.)?read",
    "stdin_readline": r"sys\.stdin\.readline",
    "setrecursionlimit": r"setrecursionlimit",
    "deque": r"\bdeque\b",
    "heapq": r"heapq",
    "bisect": r"bisect",
    "lru_cache": r"lru_cache|functools\.cache",
    "math": r"\bmath\.",
    "input_builtin": r"(?<!\.)\binput\s*\(",
}


def load_exposed_teacher_corpus_v1(
        target_short_names: Sequence[str], *,
        exposed_root: str = EXPOSED_TEACHER_ROOT_DEFAULT,
        max_solutions: int = 256) -> list[TeacherSolutionV1]:
    """Load EXPOSED-side accepted Python solutions of OTHER problems (problem-disjoint
    from the resistant targets).  Asserts disjointness — the no-leakage family rule."""
    forbidden = {str(s).lower() for s in target_short_names}
    out: list[TeacherSolutionV1] = []
    paths = sorted(glob.glob(os.path.join(exposed_root, "**", "submissions", "accepted",
                                          "*.py"), recursive=True))
    for p in paths:
        # problem short = the package dir two levels above submissions/accepted
        short = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(p))))
        if short.lower() in forbidden:
            continue  # never admit a same-named problem as teacher material
        try:
            code = open(p, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        if not code.strip():
            continue
        out.append(TeacherSolutionV1(problem_short=short,
                                     path_sha=_sha256_hex(p)[:16], code=code))
        if len(out) >= int(max_solutions):
            break
    return out


def derive_family_motifs_v1(corpus: Sequence[TeacherSolutionV1]) -> FamilyMotifsV1:
    freq = {k: 0 for k in _IDIOM_PATTERNS}
    probs = set()
    for sol in corpus:
        probs.add(sol.problem_short)
        for k, pat in _IDIOM_PATTERNS.items():
            if re.search(pat, sol.code):
                freq[k] += 1
    return FamilyMotifsV1(
        schema=W126_FAMILY_ADAPTED_REPAIR_SYNTHESIS_V1_SCHEMA_VERSION,
        n_solutions=len(corpus), n_problems=len(probs), idiom_freq=freq,
        corpus_cid=_sha256_hex([s.path_sha for s in corpus]))


# ============================================================ faithful candidate runner

def _run_capped_v1(code: str, stdin_text: str, *, timeout_s: float,
                   max_bytes: int = 2_000_000) -> tuple[Optional[bytes], str, Optional[int]]:
    """Run a candidate (``[python, -I, -c, code]``) in its OWN process group with a wall
    timeout AND a hard stdout byte cap, killing the whole group on timeout/cap so a
    pathological synthesized candidate (e.g. an infinite ``print`` loop) cannot blow up
    memory/disk or orphan a runaway child.  Returns (stdout_bytes|None, status, rc)."""
    if not str(code or "").strip():
        return None, "empty", None
    import os
    import select
    import signal
    import threading
    try:
        proc = subprocess.Popen(
            [sys.executable, "-I", "-c", str(code)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            start_new_session=True)
    except Exception:  # noqa: BLE001
        return None, "spawn_error", None

    def _feed():
        try:
            proc.stdin.write(str(stdin_text).encode("utf-8"))
            proc.stdin.close()
        except Exception:  # noqa: BLE001
            pass
    threading.Thread(target=_feed, daemon=True).start()

    buf = bytearray()
    status = "ok"
    deadline = None  # set lazily to avoid Date.now-style import; use select countdown
    import time as _time
    deadline = _time.monotonic() + float(timeout_s)
    fd = proc.stdout.fileno()
    while True:
        remaining = deadline - _time.monotonic()
        if remaining <= 0:
            status = "timeout"
            break
        try:
            r, _, _ = select.select([fd], [], [], min(remaining, 0.5))
        except Exception:  # noqa: BLE001
            break
        if r:
            try:
                chunk = os.read(fd, 65536)
            except Exception:  # noqa: BLE001
                chunk = b""
            if not chunk:
                break  # EOF
            buf += chunk
            if len(buf) >= int(max_bytes):
                status = "capped"
                break
        elif proc.poll() is not None:
            break
    rc = proc.poll()
    if rc is None or status in ("timeout", "capped"):
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:  # noqa: BLE001
            pass
        try:
            proc.wait(timeout=2)
        except Exception:  # noqa: BLE001
            pass
        rc = proc.returncode
    if status == "timeout":
        return None, "timeout", rc
    if status != "capped" and rc not in (0, None):
        return None, "runtime_error", rc
    return bytes(buf), status, rc


def _run_candidate_stdout_v1(code: str, stdin_text: str, *, timeout_s: float
                             ) -> tuple[Optional[str], str]:
    """Capped stdout capture for the consensus dispatcher (judging stays official)."""
    out, status, _rc = _run_capped_v1(code, stdin_text, timeout_s=timeout_s)
    if out is None:
        return None, status
    return out.decode("utf-8", "replace"), status


def _grade_secret_screen_v1(problem: IcpcPilotProblemV1, code: str, *, timeout_s: float
                            ) -> bool:
    """SAFE capped screen of a candidate over ALL secret cases (smallest-input first,
    short-circuit on first failure), judged by the OFFICIAL ``judge_icpc_output_v1``.  A
    True here is CONFIRMED by the official ``grade_on_secret_v1`` (a passing candidate has
    small output ⇒ no blow-up).  A False is authoritative."""
    for inp, exp in sorted(problem.secret_cases, key=lambda c: len(c[0])):
        out, status, _rc = _run_capped_v1(code, inp, timeout_s=timeout_s)
        if out is None or status == "capped":
            return False
        if not judge_icpc_output_v1(got_stdout=out.decode("utf-8", "replace"),
                                    expected=exp, kind=problem.kind,
                                    float_tol=problem.float_tol):
            return False
    return True


# ============================================================ S1 — cross-candidate splice

def _funcs_and_body(code: str) -> Optional[tuple[ast.Module, dict, list]]:
    try:
        tree = ast.parse(str(code or ""))
    except SyntaxError:
        return None
    funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    return tree, funcs, list(tree.body)


def synth_splice_v1(pool: PerProblemPoolV1, *, max_out: int = 24) -> list[str]:
    """AST recombination: for each ordered pair of DISTINCT parsing generations, swap a
    same-named function body from donor into host.  Produces new programs that fuse one
    generation's structure with another's sub-routine."""
    codes = []
    seen = set()
    for c in [pool.a0_code, *pool.a1_codes, *pool.b_codes]:
        h = _code_norm_sha(c)
        if h not in seen and _parses(c):
            seen.add(h)
            codes.append(c)
    out: list[str] = []
    out_seen: set = set()
    for hi, host in enumerate(codes):
        ph = _funcs_and_body(host)
        if ph is None:
            continue
        for di, donor in enumerate(codes):
            if di == hi:
                continue
            pd = _funcs_and_body(donor)
            if pd is None:
                continue
            host_tree, host_funcs, _ = ph
            _, donor_funcs, _ = pd
            shared = set(host_funcs) & set(donor_funcs)
            for fname in sorted(shared):
                try:
                    new_tree = ast.parse(host)  # fresh copy
                    for node in new_tree.body:
                        if isinstance(node, ast.FunctionDef) and node.name == fname:
                            node.body = donor_funcs[fname].body
                    cand = ast.unparse(new_tree)
                except Exception:  # noqa: BLE001
                    continue
                h = _code_norm_sha(cand)
                if h not in out_seen and _parses(cand):
                    out_seen.add(h)
                    out.append(cand)
                    if len(out) >= int(max_out):
                        return out
    return out


# ============================================================ S2 — digest-grounded repair

_ROBUST_STDIN_HEADER = (
    "import sys as _sys\n"
    "_DATA = _sys.stdin.buffer.read().decode('utf-8', 'replace')\n"
    "_TOKENS = _DATA.split()\n"
    "_TI = [0]\n"
    "def _tok():\n"
    "    _v = _TOKENS[_TI[0]]; _TI[0]+=1; return _v\n")


def _digest_repairs(code: str, digest: FailureDigestV1) -> list[str]:
    """Deterministic, principled micro-repairs conditioned on the typed PUBLIC digest.
    No secret consulted; no answer-fudging (no numeric output tweaking)."""
    reps: list[str] = []
    src = str(code or "")
    exc = (digest.exception_type or "").lower()
    # (a) recursion-limit + iterative-safety for RecursionError / deep recursion
    if "recursion" in exc or "def " in src and re.search(r"\b\w+\s*\([^)]*\)\s*$", "") is None:
        if "setrecursionlimit" not in src:
            reps.append("import sys\nsys.setrecursionlimit(1000000)\n" + src)
    # (b) robust stdin for parse/IO/index errors (EOF, split mismatch, index)
    if exc in ("eoferror", "valueerror", "indexerror", "stopiteration") or \
            "index" in exc or "value" in exc:
        if "input()" in src or "sys.stdin" in src:
            hardened = re.sub(r"(?m)^(\s*)input\(\)", r"\1_tok()", src)
            hardened = hardened.replace("input().split()", "_TOKENS")  # rough
            if hardened != src:
                reps.append(_ROBUST_STDIN_HEADER + hardened)
    # (c) ensure entrypoint is actually called (a common silent no-output bug)
    if re.search(r"(?m)^def main\(\)\s*:", src) and not re.search(
            r"(?m)^main\(\)\s*$", src) and "__main__" not in src:
        reps.append(src.rstrip() + "\n\nmain()\n")
    if re.search(r"(?m)^def solve\(\)\s*:", src) and not re.search(
            r"(?m)^solve\(\)\s*$", src):
        reps.append(src.rstrip() + "\n\nsolve()\n")
    # (d) yes/no casing variants (a real, common ICPC token-case bug)
    for a, b in (("Yes", "YES"), ("No", "NO"), ("yes", "Yes"), ("no", "No"),
                 ("YES", "Yes"), ("NO", "No")):
        if re.search(rf'(["\']){a}\1', src):
            reps.append(re.sub(rf'(["\']){a}(\1)', rf'\1{b}\2', src))
    out, seen = [], set()
    for r in reps:
        h = _code_norm_sha(r)
        if h not in seen and _parses(r):
            seen.add(h)
            out.append(r)
    return out


def synth_digest_repair_v1(pool: PerProblemPoolV1, problem: IcpcPilotProblemV1,
                           plane: AuditedGraderPlaneV1, *, max_out: int = 24) -> list[str]:
    out: list[str] = []
    out_seen: set = set()
    seen_src: set = set()
    for c in [pool.a0_code, *pool.a1_codes, *pool.b_codes]:
        h = _code_norm_sha(c)
        if h in seen_src:
            continue
        seen_src.add(h)
        sg = plane.grade_samples(c)
        for r in _digest_repairs(c, sg.digest):
            rh = _code_norm_sha(r)
            if rh not in out_seen:
                out_seen.add(rh)
                out.append(r)
                if len(out) >= int(max_out):
                    return out
    return out


# ============================================================ S3 — family-motif harden

def synth_motif_harden_v1(pool: PerProblemPoolV1, motifs: FamilyMotifsV1, *,
                          max_out: int = 24) -> list[str]:
    """Apply family-level hardening idioms (fast I/O, recursionlimit) — derived from the
    exposed teacher corpus — to each generation."""
    transforms = []
    # fast buffered stdin shim that preserves input()/readline semantics
    fastio = ("import sys as _S\n"
              "input = lambda: _S.stdin.readline().rstrip('\\n')\n")
    transforms.append(("fastio", lambda s: (fastio + s) if "input(" in s
                       and "sys.stdin" not in s else None))
    transforms.append(("reclimit", lambda s: ("import sys\nsys.setrecursionlimit(1000000)\n"
                       + s) if "setrecursionlimit" not in s and "def " in s else None))
    transforms.append(("fastio+reclimit", lambda s: (
        "import sys as _S\n_S.setrecursionlimit(1000000)\n"
        "input = lambda: _S.stdin.readline().rstrip('\\n')\n" + s)
        if "input(" in s and "sys.stdin" not in s else None))
    out: list[str] = []
    out_seen: set = set()
    seen_src: set = set()
    for c in [pool.a0_code, *pool.a1_codes, *pool.b_codes]:
        h = _code_norm_sha(c)
        if h in seen_src:
            continue
        seen_src.add(h)
        for _name, fn in transforms:
            try:
                r = fn(c)
            except Exception:  # noqa: BLE001
                r = None
            if not r:
                continue
            rh = _code_norm_sha(r)
            if rh not in out_seen and _parses(r):
                out_seen.add(rh)
                out.append(r)
                if len(out) >= int(max_out):
                    return out
    return out


# ============================================================ S-CONS — output consensus

def _trust_weights_v1(blind: Sequence[BlindCandidateScoreV1]) -> list[float]:
    """Trust weights from the arsenal's trust-weighted consensus on the blind scores.
    Each generation is a witness; self_confidence = sample-pass fraction; value encodes
    (sample_pass, parses).  Returns the per-witness trust distribution (+small floor)."""
    if not blind:
        return []
    wits = []
    for i, b in enumerate(blind):
        conf = (b.sample_pass / b.sample_total) if b.sample_total else 0.0
        wits.append(WitnessEvidenceV1(
            witness_id=f"g{i}",
            value=_np.array([float(b.sample_pass), float(1 if b.parses else 0)]),
            self_confidence=float(max(0.0, min(1.0, conf))),
            role="generation"))
    try:
        dec = trust_weighted_consensus_v1(
            witnesses=wits, config=TrustWeightedConsensusConfigV1())
        td = list(dec.trust_distribution)
    except Exception:  # noqa: BLE001
        td = [1.0] * len(blind)
    if len(td) != len(blind) or not any(t > 0 for t in td):
        td = [1.0] * len(blind)
    return [float(t) + 1e-9 for t in td]


def _plurality_output(outputs: Sequence[Optional[str]], weights: Sequence[float]
                      ) -> Optional[str]:
    """Trust-weighted plurality over normalised output strings (None = no vote)."""
    tally: dict = {}
    for out, w in zip(outputs, weights):
        if out is None:
            continue
        key = "\n".join(ln.rstrip() for ln in out.rstrip("\n").splitlines())
        tally[key] = tally.get(key, 0.0) + float(w)
    if not tally:
        return None
    best = max(tally.items(), key=lambda kv: (kv[1], -len(kv[0])))
    return best[0]


@dataclasses.dataclass(frozen=True)
class ConsensusEvalV1:
    variant: str
    passed_all_secret: bool
    n_cases: int
    n_consensus_correct: int


def eval_output_consensus_v1(problem: IcpcPilotProblemV1, pool: PerProblemPoolV1,
                             plane: AuditedGraderPlaneV1, *, timeout_s: float = 3.0,
                             max_cases: int = 0, screen_cases: int = 8) -> list[ConsensusEvalV1]:
    """Evaluate the output-consensus dispatcher out-of-band (a $0 precursor of the new
    program).  For each secret case: run the trusted generations, take the trust-weighted
    plurality output, judge vs expected with the OFFICIAL judge.  Pass iff the consensus is
    correct on ALL secret cases.  Three BLIND variants (no secret used for weighting):
    unweighted majority, trust-weighted, sample-passers-only.

    Cases are ordered SMALLEST-input-first so a wrong consensus fails on a cheap case
    (all-or-nothing ⇒ order-independent result).  ``screen_cases`` bounds the fast screen;
    the all-cases verdict is unchanged (a screen failure ⇒ definitely not all-pass)."""
    codes, seen = [], set()
    for c in [pool.a0_code, *pool.a1_codes, *pool.b_codes]:
        h = _code_norm_sha(c)
        if h not in seen:
            seen.add(h)
            codes.append(c)
    blind = [BlindCandidateScoreV1(
        code_norm_sha=_code_norm_sha(c),
        sample_pass=(sg := plane.grade_samples(c)).n_pass, sample_total=sg.n_total,
        parses=_parses(c), code_len=len(c), cluster_size=1) for c in codes]
    trust = _trust_weights_v1(blind)
    uniform = [1.0] * len(codes)
    sample_mask = [1.0 if (b.sample_total and b.sample_pass == b.sample_total) else 0.0
                   for b in blind]
    if not any(sample_mask):  # if none pass all samples, fall back to all
        sample_mask = uniform
    variants = {"majority": uniform, "trust_weighted": trust,
                "sample_passers": sample_mask}
    cases = sorted(problem.secret_cases, key=lambda c: len(c[0]))  # smallest first
    if max_cases and len(cases) > max_cases:
        cases = cases[:max_cases]
    out_cache: dict = {}  # (gi, ci) -> stdout (shared across variants)

    def _consensus_ok(weights, ci, inp, exp) -> bool:
        outs = []
        for gi, c in enumerate(codes):
            key = (gi, ci)
            if key not in out_cache:
                out_cache[key] = _run_candidate_stdout_v1(c, inp, timeout_s=timeout_s)[0]
            outs.append(out_cache[key])
        cons = _plurality_output(outs, weights)
        return bool(cons is not None and judge_icpc_output_v1(
            got_stdout=cons, expected=exp, kind=problem.kind, float_tol=problem.float_tol))

    results: list[ConsensusEvalV1] = []
    n_scr = min(int(screen_cases), len(cases)) if screen_cases else len(cases)
    for vname, weights in variants.items():
        # screen on the smallest n_scr cases; fail fast
        n_ok = 0
        screen_pass = True
        for ci in range(n_scr):
            if _consensus_ok(weights, ci, cases[ci][0], cases[ci][1]):
                n_ok += 1
            else:
                screen_pass = False
                break
        passed_all = screen_pass
        if screen_pass and n_scr < len(cases):  # verify the remaining cases
            for ci in range(n_scr, len(cases)):
                if _consensus_ok(weights, ci, cases[ci][0], cases[ci][1]):
                    n_ok += 1
                else:
                    passed_all = False
                    break
        results.append(ConsensusEvalV1(variant=vname, passed_all_secret=bool(passed_all),
                                       n_cases=len(cases), n_consensus_correct=n_ok))
    return results


# ============================================================ per-problem synthesis

@dataclasses.dataclass(frozen=True)
class ProblemSynthResultV1:
    problem_id: str
    short_name: str
    was_unsolved: bool
    n_synth_candidates: int
    oracle_program_pass: bool          # any S1/S2/S3 candidate passes secret
    oracle_program_op: str
    blind_program_pass: bool           # blind-committed S1/S2/S3 candidate passes secret
    blind_program_op: str
    consensus_pass: bool               # any blind consensus variant passes all secret
    consensus_variant: str
    leakage_clean: bool
    trace_cid: str

    @property
    def oracle_new_pass(self) -> bool:
        return self.was_unsolved and (self.oracle_program_pass or self.consensus_pass)

    @property
    def blind_new_pass(self) -> bool:
        return self.was_unsolved and (self.blind_program_pass or self.consensus_pass)

    @property
    def winning_family(self) -> str:
        if self.consensus_pass:
            return f"consensus:{self.consensus_variant}"
        if self.blind_program_pass:
            return f"program:{self.blind_program_op}"
        if self.oracle_program_pass:
            return f"oracle:{self.oracle_program_op}"
        return "none"

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["oracle_new_pass"] = self.oracle_new_pass
        d["blind_new_pass"] = self.blind_new_pass
        d["winning_family"] = self.winning_family
        return d


def synthesize_and_measure_problem_v1(
        problem: IcpcPilotProblemV1, pool: PerProblemPoolV1, motifs: FamilyMotifsV1,
        guard: SynthesisLeakageGuardV1, *, was_unsolved: bool, B_syn: int = 5,
        timeout_s: float = 4.0, consensus_max_cases: int = 0,
        run_consensus: bool = True) -> ProblemSynthResultV1:
    plane = AuditedGraderPlaneV1(problem, caller_agent_id="w126synth", timeout_s=timeout_s)
    # secret cases ordered SMALLEST-input-first ⇒ wrong candidates fail on a cheap case
    # (all-or-nothing grader ⇒ order-independent verdict)
    oprob = dataclasses.replace(problem, secret_cases=tuple(
        sorted(problem.secret_cases, key=lambda c: len(c[0]))))
    # ---- build the synthesis candidate set (S1 + S2 + S3); dedup by normalised sha ----
    cand_ops: list[tuple[str, str]] = []  # (op, code)
    cand_seen: set = set()
    for op, lst in (("S1_splice", synth_splice_v1(pool)),
                    ("S2_digest_repair", synth_digest_repair_v1(pool, problem, plane)),
                    ("S3_motif_harden", synth_motif_harden_v1(pool, motifs))):
        for c in lst:
            h = _code_norm_sha(c)
            if h not in cand_seen:
                cand_seen.add(h)
                cand_ops.append((op, c))
    # ---- no-leakage enforcement: every synthesized candidate must be clean ----
    # provenance = the source generations + public samples synthesis recombines; a secret
    # run already in provenance is a base-model coincidence, not a synthesis injection
    guard.set_provenance([pool.a0_code, *pool.a1_codes, *pool.b_codes]
                         + [i + o for i, o in problem.samples])
    leak_clean = guard.all_clean([c for _op, c in cand_ops])
    # ---- blind commit: rank synth candidates by blind score, commit top-B_syn ----
    scored = []
    for op, c in cand_ops:
        sg = plane.grade_samples(c)
        sk = BlindCandidateScoreV1(
            code_norm_sha=_code_norm_sha(c), sample_pass=sg.n_pass,
            sample_total=sg.n_total, parses=_parses(c), code_len=len(c),
            cluster_size=1).sort_key()
        scored.append((sk, op, c))
    scored.sort(key=lambda t: t[0])
    committed = scored[:int(B_syn)]
    committed_shas = {_code_norm_sha(c) for _sk, _op, c in committed}
    # ---- oracle ceiling (ANY synth candidate) + blind (committed only) ----
    # grade committed candidates FIRST so a blind win short-circuits the rest
    oracle_pass, oracle_op = False, ""
    blind_pass, blind_op = False, ""
    ordered = [(op, c) for _sk, op, c in committed] + [
        (op, c) for op, c in cand_ops if _code_norm_sha(c) not in committed_shas]
    for op, c in ordered:
        # SAFE capped screen first; only a screen-pass is confirmed by the OFFICIAL grader
        if not _grade_secret_screen_v1(oprob, c, timeout_s=timeout_s):
            continue
        ok, _, _ = grade_on_secret_v1(oprob, c, timeout_s=timeout_s)
        if ok:
            oracle_pass = True
            oracle_op = oracle_op or op
            if _code_norm_sha(c) in committed_shas:
                blind_pass = True
                blind_op = blind_op or op
                break  # a blind win ⇒ oracle also satisfied; stop grading
    # ---- S-CONS output-consensus dispatcher (blind) ----
    cons_pass, cons_variant = False, ""
    if run_consensus:
        for ev in eval_output_consensus_v1(problem, pool, plane,
                                           timeout_s=min(timeout_s, 3.0),
                                           max_cases=consensus_max_cases):
            if ev.passed_all_secret:
                cons_pass = True
                cons_variant = ev.variant
                break
    trace_cid = _sha256_hex({
        "schema": W126_FAMILY_ADAPTED_REPAIR_SYNTHESIS_V1_SCHEMA_VERSION,
        "problem": problem.problem_id,
        "cands": sorted(_code_norm_sha(c)[:16] for _op, c in cand_ops),
        "committed": sorted(s[:16] for s in committed_shas)})
    return ProblemSynthResultV1(
        problem_id=problem.problem_id, short_name=problem.short_name,
        was_unsolved=bool(was_unsolved), n_synth_candidates=len(cand_ops),
        oracle_program_pass=oracle_pass, oracle_program_op=oracle_op,
        blind_program_pass=blind_pass, blind_program_op=blind_op,
        consensus_pass=cons_pass, consensus_variant=cons_variant,
        leakage_clean=bool(leak_clean), trace_cid=trace_cid)


# ============================================================ precursor + earn gate

@dataclasses.dataclass(frozen=True)
class SynthEarnVerdictV1:
    schema: str
    n_unsolved_probed: int
    oracle_new_solved: int
    blind_new_solved: int
    blind_new_ids: tuple[str, ...]
    distinct_families: tuple[str, ...]
    leakage_all_clean: bool
    p1_two_distinct_new: bool
    p2_two_distinct_families: bool
    earned: bool
    verdict_label: str
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {k: (list(v) if isinstance(v, tuple) else v)
                for k, v in dataclasses.asdict(self).items()}


def apply_synthesis_earn_gate_v1(results: Sequence[ProblemSynthResultV1]
                                 ) -> SynthEarnVerdictV1:
    """RUNBOOK_W126 § 5/6: fresh resistant pilot EARNED iff
    leakage-clean ∧ P1 (>=2 distinct NEW blind-solved) ∧ P2 (>=2 distinct families)."""
    unsolved = [r for r in results if r.was_unsolved]
    oracle_new = [r for r in unsolved if r.oracle_new_pass]
    blind_new = [r for r in unsolved if r.blind_new_pass]
    families = sorted({r.winning_family for r in blind_new if r.winning_family != "none"})
    leak_clean = all(r.leakage_clean for r in results)
    p1 = len(blind_new) >= 2
    p2 = len(families) >= 2
    earned = bool(leak_clean and p1 and p2)
    if earned:
        label = "FRESH_RESISTANT_PILOT_EARNED_SYNTHESIS_HEADROOM"
        rationale = (f"Synthesis created NEW blind secret-passers on {len(blind_new)} "
                     f"uniformly-unsolved resistant problems spanning {len(families)} "
                     f"distinct families {families}; P1∧P2 hold and synthesis is "
                     f"leakage-clean ⇒ a fresh honest Maverick resistant pilot is earned.")
    elif leak_clean and (oracle_new or blind_new):
        label = "FRESH_RESISTANT_PILOT_NOT_EARNED_SYNTHESIS_THIN"
        rationale = (f"Leakage-clean, but the synthesis headroom is too thin to fund "
                     f"spend: blind_new={len(blind_new)} (need >=2), "
                     f"families={len(families)} (need >=2); oracle_new={len(oracle_new)}. "
                     f"$0 NIM; register the synthesis cap.")
    elif leak_clean:
        label = "FRESH_RESISTANT_PILOT_NOT_EARNED_SYNTHESIS_DEAD"
        rationale = ("Leakage-clean and the synthesis slate created ZERO new "
                     "secret-passing trajectories (oracle ceiling 0) on the uniformly-"
                     "unsolved resistant problems ⇒ deterministic recombination/repair "
                     "of capability-failed generations cannot manufacture correct "
                     "algorithms at $0. $0 NIM; register the synthesis cap.")
    else:
        label = "SYNTHESIS_INVALID_LEAKAGE"
        rationale = ("A synthesis input leaked target secret/answer material ⇒ the lane "
                     "is killed per RUNBOOK_W126 § 2. $0 NIM.")
    return SynthEarnVerdictV1(
        schema=W126_FAMILY_ADAPTED_REPAIR_SYNTHESIS_V1_SCHEMA_VERSION,
        n_unsolved_probed=len(unsolved), oracle_new_solved=len(oracle_new),
        blind_new_solved=len(blind_new),
        blind_new_ids=tuple(r.problem_id for r in blind_new),
        distinct_families=tuple(families), leakage_all_clean=leak_clean,
        p1_two_distinct_new=p1, p2_two_distinct_families=p2, earned=earned,
        verdict_label=label, rationale=rationale)


__all__ = [
    "W126_FAMILY_ADAPTED_REPAIR_SYNTHESIS_V1_SCHEMA_VERSION",
    "LeakageVerdictV1", "SynthesisLeakageGuardV1",
    "TeacherSolutionV1", "FamilyMotifsV1", "load_exposed_teacher_corpus_v1",
    "derive_family_motifs_v1",
    "synth_splice_v1", "synth_digest_repair_v1", "synth_motif_harden_v1",
    "ConsensusEvalV1", "eval_output_consensus_v1",
    "ProblemSynthResultV1", "synthesize_and_measure_problem_v1",
    "SynthEarnVerdictV1", "apply_synthesis_earn_gate_v1",
]
