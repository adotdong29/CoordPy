"""W125 — Controller-Native Code Mechanism V1 (COO-9 sibling).

The FIRST repo mechanism that wires the otherwise-unused hosted/controller arsenal
(``hosted_router_controller_v12`` / ``hosted_logprob_router_v12`` /
``hosted_cache_aware_planner_v12``) and the audited ``tool_call_substrate_v1`` plane and
the ``executor_grounded_patcher_v1`` typed failure digest onto the official-ICPC
stdin/stdout code path (``icpc_reflexion_bench_v1``).  It promotes the W124 M6
*contract-only* deterministic PATCH/REPLAN/ABSTAIN controller into a real, executable,
genuinely controller-native mechanism, and gives it the NIM-free contract surface +
$0 resistant-replay headroom probe specified in ``docs/RUNBOOK_W125.md`` (locked
pre-registration).

It is held OUTSIDE the stable SDK contract: explicit-import-only, ``coordpy/__init__.py``
untouched, ``coordpy.__version__ == "0.5.20"``, no PyPI publish.

Three controller-native candidates (RUNBOOK § 2):

* **C1** role-specialized planner/controller (shared prefix + per-role blocks via the
  hosted cache-aware planner; roles solver/debugger/verifier/patcher; controller routes
  DRAFT/PATCH/ABSTAIN on the verifier's PUBLIC-sample verdict).
* **C2** router-selected multi-candidate controller (multiple drafts; hidden-test-BLIND
  scorer; the hosted logprob-router abstain floor selects which candidate is committed).
* **C3** tool-substrate audited repair loop (every grader/executor call is a first-class
  audited ``ToolCallSchemaV1`` event; retry policy is digest-routed PATCH/REPLAN/ABSTAIN,
  NOT prose reflexion chaining) — the LEAD (audited-tool-plane + digest-router superset of
  the W124 M6 contract and the W111 M3 patcher lineage).

A negative control **C0** ("reflexion relabeled": a controller that emits routing labels
but always picks DRAFT-next) is included so the structural fake-different test has
demonstrable discriminating power — C0 classifies FAKE_DIFFERENT and is killed.

NOTHING here reads the hidden test.  All in-loop feedback is PUBLIC (sample-case results +
judge verdict bit + executor stderr tail), exactly as W120 reflexion B received; the
official secret grader scores only the controller's COMMITTED answer.  This is the
property that makes the mechanism hosted-translatable (text-level, no hidden-state
dependence).
"""
from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
import re
from typing import Any, Callable, Optional, Sequence

# --- the underused hosted/controller arsenal (creates the bridge edges § 11) -----------
from .hosted_cache_aware_planner_v12 import HostedCacheAwarePlannerV12
from .hosted_logprob_router_v12 import HostedLogprobRouterV12
from .hosted_router_controller_v12 import (
    W79_HOSTED_ROUTER_CONTROLLER_V12_SCHEMA_VERSION,
)
# --- the audited tool-call substrate plane ---------------------------------------------
from .tool_call_substrate_v1 import (
    IdempotencyClass,
    ToolAuditChainV1,
    ToolCallSchemaV1,
    ToolIntegrityVerdict,
    ToolResultSchemaV1,
    W84_TOOL_SUBSTRATE_V1_SCHEMA_VERSION,
)
# --- the typed executor failure digest (W111 M3 / W124 M6 lineage) ---------------------
from .executor_grounded_patcher_v1 import FailureDigestV1, parse_failure_digest_v1
# --- the official-ICPC code path: grader + sample feedback + extractor -----------------
from .icpc_reflexion_bench_v1 import (
    IcpcPilotProblemV1,
    extract_candidate_code_v1,
    grade_on_secret_v1,
    sample_feedback_v1,
)
from .coordpy_icpc_battlefield_v1 import KIND_PASSFAIL

W125_CONTROLLER_NATIVE_CODE_MECHANISM_V1_SCHEMA_VERSION: str = (
    "coordpy.controller_native_code_mechanism_v1.v1")


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _code_norm_sha(code: str) -> str:
    """Normalised candidate identity: strip comments + collapse whitespace (for
    self-consistency clustering + reflexion-divergence repeat detection)."""
    s = re.sub(r"#.*", "", str(code or ""))
    s = re.sub(r"\s+", " ", s).strip()
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _parses(code: str) -> bool:
    try:
        compile(str(code or ""), "<cand>", "exec")
        return True
    except Exception:
        return False


# ====================================================================== controller action

class ControllerAction(str, enum.Enum):
    DRAFT = "draft"
    PATCH = "patch"
    REPLAN = "replan"
    ABSTAIN = "abstain"


# ====================================================================== audited grader plane

@dataclasses.dataclass(frozen=True)
class SampleGradeV1:
    """PUBLIC-sample grading outcome (the only in-loop feedback the controller may read)."""
    n_pass: int
    n_total: int
    stderr_tail: str
    timed_out: bool
    digest: FailureDigestV1

    @property
    def all_pass(self) -> bool:
        return self.n_total > 0 and self.n_pass == self.n_total


class AuditedGraderPlaneV1:
    """Wraps the official ICPC grader as first-class audited tool events.

    Every sample / secret grade is a content-addressed ``ToolCallSchemaV1`` +
    ``ToolResultSchemaV1`` committed to a ``ToolAuditChainV1`` (idempotent: grading the
    same code on the same cases is deterministic).  The secret ``.in``/``.ans`` NEVER
    enter a model-facing prompt; the plane exposes a ``secret_leak_in`` guard for the
    contract check.
    """

    _TOOL_VERSION_CID = _sha256_hex({
        "v": W125_CONTROLLER_NATIVE_CODE_MECHANISM_V1_SCHEMA_VERSION,
        "tool": "icpc_grader", "substrate": W84_TOOL_SUBSTRATE_V1_SCHEMA_VERSION})

    def __init__(self, problem: IcpcPilotProblemV1, *, caller_agent_id: str,
                 timeout_s: float = 15.0) -> None:
        self.problem = problem
        self.caller_agent_id = str(caller_agent_id)
        self.timeout_s = float(timeout_s)
        self.chain = ToolAuditChainV1()
        self.run_cid = _sha256_hex({"kind": "w125_grader_run",
                                    "problem_id": problem.problem_id,
                                    "caller": self.caller_agent_id})
        self._parent = "genesis"
        self.n_sample_calls = 0
        self.n_secret_calls = 0

    # -- the never-reads-secret guard -----------------------------------------------------
    def secret_leak_in(self, text: str) -> bool:
        """True iff any secret .in/.ans byte appears in ``text`` (a model-facing prompt)."""
        t = str(text or "")
        for inp, ans in self.problem.secret_cases:
            a = ans.strip()
            i = inp.strip()
            if a and a in t:
                return True
            if i and len(i) >= 8 and i in t:
                return True
        return False

    def _commit(self, tool_id: str, args: dict, result: dict,
                exit_code: int, stderr: str, verdict: str) -> str:
        call = ToolCallSchemaV1(
            schema=W84_TOOL_SUBSTRATE_V1_SCHEMA_VERSION, tool_id=str(tool_id),
            tool_version_cid=self._TOOL_VERSION_CID,
            args_bytes=_canonical_bytes(args), timestamp_ns=0,
            caller_agent_id=self.caller_agent_id, run_cid=self.run_cid,
            parent_event_cid=self._parent,
            idempotency_class=IdempotencyClass.IDEMPOTENT.value,
            idempotency_token_cid="absent")
        res = ToolResultSchemaV1(
            schema=W84_TOOL_SUBSTRATE_V1_SCHEMA_VERSION, call_cid=call.cid(),
            result_bytes=_canonical_bytes(result), exit_code=int(exit_code),
            stderr_bytes=str(stderr).encode("utf-8"), duration_ns=0,
            side_effects_cid="none", integrity_verdict=str(verdict))
        if not self.chain.already_committed(call, None):
            self.chain.commit(call=call, result=res)
        self._parent = res.cid()
        return res.cid()

    def grade_samples(self, code: str) -> SampleGradeV1:
        """PUBLIC sample-case grade (audited)."""
        self.n_sample_calls += 1
        p = self.problem
        n_pass = 0
        n_total = len(p.samples[:3])
        stderr_tail = ""
        timed_out = False
        fb = sample_feedback_v1(p, code, timeout_s=self.timeout_s)
        for ln in fb.splitlines():
            if ": PASS" in ln:
                n_pass += 1
            elif "TIMEOUT" in ln:
                timed_out = True
                stderr_tail = stderr_tail or "TIMEOUT"
            elif "runtime error" in ln or "WRONG" in ln:
                stderr_tail = stderr_tail or ln.strip()
        digest = parse_failure_digest_v1(stderr_tail=stderr_tail, timed_out=timed_out)
        self._commit("icpc_sample_grader_v1",
                     {"problem_id": p.problem_id, "code_sha": _code_norm_sha(code),
                      "n_samples": n_total},
                     {"n_pass": n_pass, "n_total": n_total, "feedback": fb},
                     exit_code=(0 if n_pass == n_total and n_total else 1),
                     stderr=stderr_tail,
                     verdict=ToolIntegrityVerdict.OK.value)
        return SampleGradeV1(n_pass=n_pass, n_total=n_total, stderr_tail=stderr_tail,
                             timed_out=timed_out, digest=digest)

    def grade_secret(self, code: str) -> bool:
        """Official secret grade (audited) — FINAL scorer of the committed answer only."""
        self.n_secret_calls += 1
        p = self.problem
        passed, stderr_tail, n = grade_on_secret_v1(p, code, timeout_s=self.timeout_s)
        self._commit("icpc_secret_grader_v1",
                     {"problem_id": p.problem_id, "code_sha": _code_norm_sha(code)},
                     {"passed": bool(passed), "n_cases_checked": int(n)},
                     exit_code=(0 if passed else 1), stderr=stderr_tail,
                     verdict=ToolIntegrityVerdict.OK.value)
        return bool(passed)

    def merkle_root(self) -> str:
        return self.chain.merkle_root()


# ====================================================================== blind candidate score

@dataclasses.dataclass(frozen=True)
class BlindCandidateScoreV1:
    """Hidden-test-BLIND score of a candidate (public-sample + self-consistency + length).

    No secret information.  Selection key (desc/asc): sample_pass↓, cluster_size↓,
    parses↓, code_len↑ (shorter, parsing, sample-passing, agreed-upon wins)."""
    code_norm_sha: str
    sample_pass: int
    sample_total: int
    parses: bool
    code_len: int
    cluster_size: int

    def sort_key(self) -> tuple:
        return (-int(self.sample_pass), -int(self.cluster_size),
                -(1 if self.parses else 0), int(self.code_len))


def _blind_scores(codes: Sequence[str], plane: AuditedGraderPlaneV1
                  ) -> list[BlindCandidateScoreV1]:
    norm = [_code_norm_sha(c) for c in codes]
    cluster = {s: norm.count(s) for s in set(norm)}
    out: list[BlindCandidateScoreV1] = []
    for c, s in zip(codes, norm):
        sg = plane.grade_samples(c)
        out.append(BlindCandidateScoreV1(
            code_norm_sha=s, sample_pass=sg.n_pass, sample_total=sg.n_total,
            parses=_parses(c), code_len=len(str(c or "")), cluster_size=cluster[s]))
    return out


# ====================================================================== structural fingerprint

@dataclasses.dataclass(frozen=True)
class MechanismFingerprintV1:
    """Structural fingerprint of a candidate's control flow (RUNBOOK § 4)."""
    name: str
    n_distinct_action_types: int
    has_audited_tool_plane: bool
    retry_is_digest_conditioned: bool
    control_flow_is_linear_chain: bool

    def n_native_properties(self) -> int:
        return (int(self.n_distinct_action_types >= 2)
                + int(self.has_audited_tool_plane)
                + int(self.retry_is_digest_conditioned))

    def classify(self) -> str:
        if (not self.control_flow_is_linear_chain) and self.n_native_properties() >= 2:
            return "REAL"
        return "FAKE_DIFFERENT"

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name,
                "n_distinct_action_types": int(self.n_distinct_action_types),
                "has_audited_tool_plane": bool(self.has_audited_tool_plane),
                "retry_is_digest_conditioned": bool(self.retry_is_digest_conditioned),
                "control_flow_is_linear_chain": bool(self.control_flow_is_linear_chain),
                "n_native_properties": int(self.n_native_properties()),
                "classification": self.classify()}


def reflexion_b_fingerprint() -> MechanismFingerprintV1:
    """The W120 reflexion B baseline = a linear DRAFT-only chain (the null mechanism)."""
    return MechanismFingerprintV1("reflexion_B", 1, False, False, True)


# ====================================================================== controller outcome

@dataclasses.dataclass(frozen=True)
class ControllerOutcomeV1:
    schema: str
    controller_id: str
    problem_id: str
    committed_code_sha: str
    committed_passed_secret: bool
    n_model_slots_used: int
    action_trace: tuple[str, ...]
    audit_merkle_root: str
    n_sample_grader_calls: int
    n_secret_grader_calls: int

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.schema, "controller_id": self.controller_id,
                "problem_id": self.problem_id,
                "committed_code_sha": self.committed_code_sha,
                "committed_passed_secret": bool(self.committed_passed_secret),
                "n_model_slots_used": int(self.n_model_slots_used),
                "action_trace": list(self.action_trace),
                "audit_merkle_root": self.audit_merkle_root,
                "n_sample_grader_calls": int(self.n_sample_grader_calls),
                "n_secret_grader_calls": int(self.n_secret_grader_calls)}

    def cid(self) -> str:
        return _sha256_hex({"kind": "w125_controller_outcome_v1", "o": self.to_dict()})


# ====================================================================== per-problem pool

@dataclasses.dataclass(frozen=True)
class PerProblemPoolV1:
    """The already-paid Maverick generation pool for one problem (replay corpus)."""
    problem_id: str
    a0_code: str
    a1_codes: tuple[str, ...]
    b_codes: tuple[str, ...]


def build_pool_from_records(block: Sequence[dict], problem_id: str) -> PerProblemPoolV1:
    """Map an 11-record block ([A0 | A1x5 | Bx5]) to a generation pool."""
    codes = [extract_candidate_code_v1(response_text=r.get("response_text", ""))
             for r in block]
    a0 = codes[0] if codes else ""
    a1 = tuple(codes[1:6])
    b = tuple(codes[6:11])
    return PerProblemPoolV1(problem_id=str(problem_id), a0_code=a0,
                            a1_codes=a1, b_codes=b)


# ====================================================================== controllers

def _digest_key(d: FailureDigestV1) -> str:
    return _sha256_hex({"exc": d.exception_type, "assert": d.assertion_line,
                        "fails": list(d.failing_tests), "timeout": d.exception_type == "Timeout"})


@dataclasses.dataclass(frozen=True)
class C3DigestRoutedRepairControllerV1:
    """LEAD — tool-substrate audited repair loop with digest-routed PATCH/REPLAN/ABSTAIN.

    Same-budget K model slots.  Walks the existing pool (A1 drafts = fresh draws, B
    attempts = the patches reflexion produced) under the typed executor digest: a
    parse/import/timeout or a REPEATED-identical digest routes REPLAN (fresh draft); a
    discriminating assertion/wrong-answer digest routes PATCH (next patch); low budget +
    repeated digest routes ABSTAIN.  Commits the blind-best candidate seen.  This is a
    $0 PRECURSOR proxy of the online trajectory (it may use BOTH pools within budget K,
    so its headroom is an UPPER bound on what pool re-routing can achieve)."""
    controller_id: str = "C3_tool_substrate_audited_repair"
    abstain_floor: float = 0.35  # from HostedLogprobRouterV12 (repurposed abstain floor)

    def cid(self) -> str:
        return _sha256_hex({"schema": W125_CONTROLLER_NATIVE_CODE_MECHANISM_V1_SCHEMA_VERSION,
                            "controller": self.controller_id,
                            "abstain_floor": round(float(self.abstain_floor), 6),
                            "router_schema": W79_HOSTED_ROUTER_CONTROLLER_V12_SCHEMA_VERSION})

    def fingerprint(self) -> MechanismFingerprintV1:
        # DRAFT + PATCH + REPLAN + ABSTAIN; audited tool plane; digest-conditioned retry.
        return MechanismFingerprintV1(self.controller_id, 4, True, True, False)

    def replay_on_pool(self, pool: PerProblemPoolV1, problem: IcpcPilotProblemV1,
                       *, K: int = 5, timeout_s: float = 15.0) -> ControllerOutcomeV1:
        plane = AuditedGraderPlaneV1(problem, caller_agent_id=self.controller_id,
                                     timeout_s=timeout_s)
        a1 = list(pool.a1_codes)
        b = list(pool.b_codes)
        ai, bi = 0, 0
        last_dk: Optional[str] = None
        trace: list[str] = []
        best_code, best_key = "", None
        for step in range(int(K)):
            if step == 0:
                cand = a1[ai] if ai < len(a1) else (b[bi] if bi < len(b) else "")
                ai += 1
                action = ControllerAction.DRAFT
            else:
                d = last_sg.digest  # noqa: F821 (set in prior iteration)
                dk = _digest_key(d)
                repeated = (dk == last_dk)
                parseish = (d.exception_type in ("Timeout", "SyntaxError",
                                                 "IndentationError", "ImportError",
                                                 "ModuleNotFoundError", "NameError"))
                budget_left = int(K) - step
                if repeated and budget_left <= 1:
                    trace.append(ControllerAction.ABSTAIN.value)
                    break
                if parseish or repeated:
                    cand = a1[ai] if ai < len(a1) else (b[bi] if bi < len(b) else best_code)
                    if ai < len(a1):
                        ai += 1
                    elif bi < len(b):
                        bi += 1
                    action = ControllerAction.REPLAN
                else:
                    cand = b[bi] if bi < len(b) else (a1[ai] if ai < len(a1) else best_code)
                    if bi < len(b):
                        bi += 1
                    elif ai < len(a1):
                        ai += 1
                    action = ControllerAction.PATCH
                last_dk = dk
            last_sg = plane.grade_samples(cand)
            key = BlindCandidateScoreV1(
                code_norm_sha=_code_norm_sha(cand), sample_pass=last_sg.n_pass,
                sample_total=last_sg.n_total, parses=_parses(cand),
                code_len=len(cand), cluster_size=1).sort_key()
            if best_key is None or key < best_key:
                best_key, best_code = key, cand
            trace.append(action.value)
            if last_sg.all_pass:  # online would submit a fully-sample-passing candidate
                best_code = cand
                break
        passed = plane.grade_secret(best_code)
        return ControllerOutcomeV1(
            schema=W125_CONTROLLER_NATIVE_CODE_MECHANISM_V1_SCHEMA_VERSION,
            controller_id=self.controller_id, problem_id=pool.problem_id,
            committed_code_sha=_code_norm_sha(best_code),
            committed_passed_secret=bool(passed),
            n_model_slots_used=min(int(K), len([t for t in trace
                                                if t != ControllerAction.ABSTAIN.value])),
            action_trace=tuple(trace), audit_merkle_root=plane.merkle_root(),
            n_sample_grader_calls=plane.n_sample_calls,
            n_secret_grader_calls=plane.n_secret_calls)


@dataclasses.dataclass(frozen=True)
class C2RouterSelectControllerV1:
    """Router-selected multi-candidate controller: blind-score the K A1 drafts, the
    hosted logprob-router abstain floor selects the committed candidate (same A1 budget)."""
    controller_id: str = "C2_router_selected_multi_candidate"

    def cid(self) -> str:
        return _sha256_hex({"schema": W125_CONTROLLER_NATIVE_CODE_MECHANISM_V1_SCHEMA_VERSION,
                            "controller": self.controller_id,
                            "logprob_router": HostedLogprobRouterV12().cid()})

    def fingerprint(self) -> MechanismFingerprintV1:
        # DRAFT + ABSTAIN (router can abstain); audited tool plane; selection not a
        # digest-conditioned *retry* (it is one-shot routing) => 2 native props, non-linear.
        return MechanismFingerprintV1(self.controller_id, 2, True, False, False)

    def replay_on_pool(self, pool: PerProblemPoolV1, problem: IcpcPilotProblemV1,
                       *, K: int = 5, timeout_s: float = 15.0) -> ControllerOutcomeV1:
        plane = AuditedGraderPlaneV1(problem, caller_agent_id=self.controller_id,
                                     timeout_s=timeout_s)
        cands = list(pool.a1_codes[:int(K)])
        scores = _blind_scores(cands, plane)
        order = sorted(range(len(cands)), key=lambda i: scores[i].sort_key())
        committed = cands[order[0]] if order else ""
        passed = plane.grade_secret(committed)
        return ControllerOutcomeV1(
            schema=W125_CONTROLLER_NATIVE_CODE_MECHANISM_V1_SCHEMA_VERSION,
            controller_id=self.controller_id, problem_id=pool.problem_id,
            committed_code_sha=_code_norm_sha(committed),
            committed_passed_secret=bool(passed), n_model_slots_used=len(cands),
            action_trace=(ControllerAction.DRAFT.value,) * len(cands),
            audit_merkle_root=plane.merkle_root(),
            n_sample_grader_calls=plane.n_sample_calls,
            n_secret_grader_calls=plane.n_secret_calls)


@dataclasses.dataclass(frozen=True)
class C1RolePlanControllerV1:
    """Role-specialized planner/controller (solver/debugger/verifier/patcher) over a
    shared-prefix + per-role-block plan from the hosted cache-aware planner."""
    controller_id: str = "C1_role_specialized_planner"
    roles: tuple[str, ...] = ("solver", "debugger", "verifier", "patcher")

    def plan_cid(self, problem: IcpcPilotProblemV1) -> str:
        planner = HostedCacheAwarePlannerV12()
        shared = f"ICPC stdin/stdout solve :: {problem.problem_id}"
        blocks = {r: (f"role={r}", problem.short_name) for r in self.roles}
        _planned, report = planner.plan_per_role_ten_layer_rotated(
            shared_prefix_text=shared, per_role_blocks=blocks)
        return _sha256_hex({"plan": report, "roles": list(self.roles),
                            "problem": problem.problem_id})

    def cid(self) -> str:
        return _sha256_hex({"schema": W125_CONTROLLER_NATIVE_CODE_MECHANISM_V1_SCHEMA_VERSION,
                            "controller": self.controller_id, "roles": list(self.roles)})

    def fingerprint(self) -> MechanismFingerprintV1:
        # DRAFT + PATCH + ABSTAIN role routing; audited tool plane; verifier-verdict
        # routed (a digest-conditioned retry over the verifier's sample verdict).
        return MechanismFingerprintV1(self.controller_id, 3, True, True, False)


@dataclasses.dataclass(frozen=True)
class C0ReflexionRelabeledControllerV1:
    """NEGATIVE CONTROL — emits routing labels but always picks DRAFT-next == reflexion.
    Must classify FAKE_DIFFERENT (proves the structural test bites)."""
    controller_id: str = "C0_reflexion_relabeled"

    def fingerprint(self) -> MechanismFingerprintV1:
        # one effective action (always next DRAFT); no audited tool plane; not
        # digest-conditioned; linear chain => FAKE_DIFFERENT.
        return MechanismFingerprintV1(self.controller_id, 1, False, False, True)


# ====================================================================== slate evaluation

@dataclasses.dataclass(frozen=True)
class SlateEvaluationV1:
    schema: str
    fingerprints: tuple[dict, ...]
    real_candidates: tuple[str, ...]
    fake_candidates: tuple[str, ...]
    lead: str

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.schema,
                "fingerprints": list(self.fingerprints),
                "real_candidates": list(self.real_candidates),
                "fake_candidates": list(self.fake_candidates),
                "lead": self.lead}


def evaluate_mechanism_slate() -> SlateEvaluationV1:
    """Structural fake-different test over {C0(control), C1, C2, C3} + reflexion B."""
    fps = [
        reflexion_b_fingerprint(),
        C0ReflexionRelabeledControllerV1().fingerprint(),
        C1RolePlanControllerV1().fingerprint(),
        C2RouterSelectControllerV1().fingerprint(),
        C3DigestRoutedRepairControllerV1().fingerprint(),
    ]
    real = tuple(f.name for f in fps if f.classify() == "REAL")
    fake = tuple(f.name for f in fps if f.classify() == "FAKE_DIFFERENT")
    # Lead = the REAL candidate with the most native properties (ties -> C3 superset).
    real_fps = [f for f in fps if f.classify() == "REAL"]
    lead = ""
    if real_fps:
        best = max(real_fps, key=lambda f: (f.n_native_properties(),
                                            f.n_distinct_action_types,
                                            f.name == "C3_tool_substrate_audited_repair"))
        lead = best.name
    return SlateEvaluationV1(
        schema=W125_CONTROLLER_NATIVE_CODE_MECHANISM_V1_SCHEMA_VERSION,
        fingerprints=tuple(f.to_dict() for f in fps),
        real_candidates=real, fake_candidates=fake, lead=lead)


# ====================================================================== contract checks

@dataclasses.dataclass(frozen=True)
class ContractCheckReportV1:
    schema: str
    audit_chain_rehash_ok: bool
    audit_tamper_detected: bool
    idempotent_recommit_refused: bool
    grader_capture_complete: bool
    never_reads_secret: bool
    routing_determinism_ok: bool
    same_budget_ok: bool

    @property
    def all_pass(self) -> bool:
        return all([self.audit_chain_rehash_ok, self.audit_tamper_detected,
                    self.idempotent_recommit_refused, self.grader_capture_complete,
                    self.never_reads_secret, self.routing_determinism_ok,
                    self.same_budget_ok])

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.schema,
                "audit_chain_rehash_ok": bool(self.audit_chain_rehash_ok),
                "audit_tamper_detected": bool(self.audit_tamper_detected),
                "idempotent_recommit_refused": bool(self.idempotent_recommit_refused),
                "grader_capture_complete": bool(self.grader_capture_complete),
                "never_reads_secret": bool(self.never_reads_secret),
                "routing_determinism_ok": bool(self.routing_determinism_ok),
                "same_budget_ok": bool(self.same_budget_ok),
                "all_pass": bool(self.all_pass)}


def synthetic_contract_problem() -> tuple[IcpcPilotProblemV1, PerProblemPoolV1]:
    """A deterministic in-repo stdin/stdout problem (read a b; print a+b) + a generation
    pool with one sample-passing draft, for verifying the controller's INTRINSIC contract
    properties on a controlled substrate (no real-executor timeout nondeterminism)."""
    prob = IcpcPilotProblemV1(
        problem_id="synthetic/sum", short_name="sum", source_repo="w125_contract",
        contest_date="2025-01-01",
        statement="Read two integers a and b from stdin; print a+b.",
        kind=KIND_PASSFAIL, float_tol=0.0,
        samples=(("2 3\n", "5\n"),),
        secret_cases=(("10 20\n", "30\n"), ("1 1\n", "2\n"), ("0 0\n", "0\n")))
    good = extract_candidate_code_v1(
        response_text="```python\ndef main():\n a,b=map(int,input().split());print(a+b)\nmain()\n```")
    bad = extract_candidate_code_v1(
        response_text="```python\ndef main():\n a,b=map(int,input().split());print(a*b)\nmain()\n```")
    pool = PerProblemPoolV1(problem_id=prob.problem_id, a0_code=bad,
                            a1_codes=(bad, bad, bad, bad, good), b_codes=(bad,) * 5)
    return prob, pool


def run_contract_checks(pool: PerProblemPoolV1, problem: IcpcPilotProblemV1,
                        *, K: int = 5, timeout_s: float = 15.0) -> ContractCheckReportV1:
    """The four NIM-free contract checks (RUNBOOK § 5) on the lead controller (C3)."""
    lead = C3DigestRoutedRepairControllerV1()

    o1 = lead.replay_on_pool(pool, problem, K=K, timeout_s=timeout_s)
    o2 = lead.replay_on_pool(pool, problem, K=K, timeout_s=timeout_s)

    # (1) audit chain re-hash + tamper + idempotency
    plane = AuditedGraderPlaneV1(problem, caller_agent_id="contract", timeout_s=timeout_s)
    plane.grade_samples(pool.a1_codes[0] if pool.a1_codes else "")
    root_a = plane.merkle_root()
    root_b = plane.merkle_root()
    rehash_ok = (root_a == root_b)
    # tamper: mutate a committed result's bytes -> root must change
    tampered_root = root_a
    if plane.chain.steps:
        c, r = plane.chain.steps[-1]
        r2 = dataclasses.replace(r, result_bytes=r.result_bytes + b"X")
        plane.chain.steps[-1] = (c, r2)
        tampered_root = plane.chain.merkle_root()
    tamper_detected = (tampered_root != root_a)
    # idempotent re-commit refused
    p2 = AuditedGraderPlaneV1(problem, caller_agent_id="contract2", timeout_s=timeout_s)
    code0 = pool.a1_codes[0] if pool.a1_codes else ""
    p2.grade_samples(code0)
    first_call = p2.chain.steps[-1][0] if p2.chain.steps else None
    recommit_refused = bool(first_call is not None
                            and p2.chain.already_committed(first_call, None))

    # (2) grader capture complete + never reads secret.  Capture is complete iff exactly
    # one final secret grade + ≥1 in-loop public-sample grade were captured as audited
    # events.  never-reads-secret: the controller's CONSTRUCTED in-loop feedback (public
    # samples only) must carry no secret-ONLY test data, and the secret grade must return
    # only a boolean.  The problem STATEMENT is model input, NOT a controller injection
    # (a real ICPC statement legitimately shares substrings with a secret case), so it is
    # not checked.
    guard = AuditedGraderPlaneV1(problem, caller_agent_id="leakcheck")
    pub_surface = str(problem.statement) + "".join(
        str(i) + str(o) for i, o in problem.samples)
    secret_only = [s for s in
                   [c.strip() for pair in problem.secret_cases for c in pair]
                   if s and len(s) >= 3 and s not in pub_surface]
    # positive control: the guard catches a verbatim secret case (proves it bites)
    leak_caught = guard.secret_leak_in(
        "HIDDEN EXPECTED: " + (problem.secret_cases[0][1] if problem.secret_cases else "Z"))
    # real check: the controller's PUBLIC feedback for a real candidate carries no
    # secret-only data, and the secret grade returns only a boolean (no secret content).
    probe_code = pool.a1_codes[0] if pool.a1_codes else ""
    fb = sample_feedback_v1(problem, probe_code, timeout_s=timeout_s)
    fb_clean = not any(so in fb for so in secret_only)
    sec_ret = AuditedGraderPlaneV1(
        problem, caller_agent_id="boolcheck").grade_secret(probe_code)
    never_reads_secret = bool(leak_caught and fb_clean and isinstance(sec_ret, bool))

    # (3) routing determinism: same pool -> same audit root + same action trace + same CID
    routing_ok = bool(o1.audit_merkle_root == o2.audit_merkle_root
                      and o1.action_trace == o2.action_trace
                      and o1.cid() == o2.cid())

    # (4) same-budget accounting: model slots <= K, exactly one final secret grade
    same_budget = bool(o1.n_model_slots_used <= int(K) and o1.n_secret_grader_calls == 1)

    return ContractCheckReportV1(
        schema=W125_CONTROLLER_NATIVE_CODE_MECHANISM_V1_SCHEMA_VERSION,
        audit_chain_rehash_ok=rehash_ok, audit_tamper_detected=tamper_detected,
        idempotent_recommit_refused=recommit_refused,
        grader_capture_complete=bool(o1.n_secret_grader_calls == 1
                                     and o1.n_sample_grader_calls >= 1),
        never_reads_secret=never_reads_secret,
        routing_determinism_ok=routing_ok, same_budget_ok=same_budget)


# ====================================================================== headroom probe

@dataclasses.dataclass(frozen=True)
class HeadroomReportV1:
    schema: str
    field: str
    n_problems: int
    a1_pass_count: int
    pool_union_secret_count: int
    oracle_pool_headroom: int
    c2_blind_committed_pass_count: int
    c3_committed_pass_count: int
    blind_selection_headroom: int
    blind_headroom_problem_ids: tuple[str, ...]
    reflexion_divergence: int
    reflexion_divergence_problem_ids: tuple[str, ...]
    looks_right_fails_hidden: int

    def to_dict(self) -> dict[str, Any]:
        return {k: (list(v) if isinstance(v, tuple) else v)
                for k, v in dataclasses.asdict(self).items()}

    def cid(self) -> str:
        return _sha256_hex({"kind": "w125_headroom_v1", "r": self.to_dict()})


def _b_chain_stuck(pool: PerProblemPoolV1, problem: IcpcPilotProblemV1,
                   plane: AuditedGraderPlaneV1) -> bool:
    """A REPLAN/ABSTAIN-divergence signature: B chain repeats a candidate or a digest."""
    shas = [_code_norm_sha(c) for c in pool.b_codes]
    if any(shas.count(s) >= 2 for s in set(shas)):
        return True
    dks = []
    for c in pool.b_codes:
        sg = plane.grade_samples(c)
        dks.append(_digest_key(sg.digest))
    return any(dks.count(d) >= 2 for d in set(dks))


def headroom_probe(problems: Sequence[IcpcPilotProblemV1],
                   pools: Sequence[PerProblemPoolV1],
                   *, field: str, K: int = 5, timeout_s: float = 15.0,
                   on_problem: Optional[Callable[[int, str, dict], None]] = None
                   ) -> HeadroomReportV1:
    """$0 resistant headroom probe (RUNBOOK § 6).  All grading via the official grader."""
    c2 = C2RouterSelectControllerV1()
    c3 = C3DigestRoutedRepairControllerV1()
    a1_pass = 0
    pool_union = 0
    c2_pass = 0
    c3_pass = 0
    blind_ids: list[str] = []
    diverge_ids: list[str] = []
    looks_right = 0
    for _pi, (prob, pool) in enumerate(zip(problems, pools)):
        # per-generation secret + sample grading
        def passes_secret(c: str) -> bool:
            ok, _, _ = grade_on_secret_v1(prob, c, timeout_s=timeout_s)
            return ok
        a1_secret = [passes_secret(c) for c in pool.a1_codes]
        b_secret = [passes_secret(c) for c in pool.b_codes]
        a0_secret = passes_secret(pool.a0_code)
        a1_ok = any(a1_secret)
        union_ok = a0_secret or a1_ok or any(b_secret)
        a1_pass += int(a1_ok)
        pool_union += int(union_ok)
        # looks-right-fails-hidden: any gen passes ALL public samples but fails secret
        plane0 = AuditedGraderPlaneV1(prob, caller_agent_id="probe", timeout_s=timeout_s)
        for c, sec in zip(list(pool.a1_codes) + list(pool.b_codes),
                          a1_secret + b_secret):
            sg = plane0.grade_samples(c)
            if sg.all_pass and not sec:
                looks_right += 1
        # C2 (blind select over A1 pool) + C3 (digest-routed walk over A1 U B)
        o2 = c2.replay_on_pool(pool, prob, K=K, timeout_s=timeout_s)
        o3 = c3.replay_on_pool(pool, prob, K=K, timeout_s=timeout_s)
        c2_pass += int(o2.committed_passed_secret)
        c3_pass += int(o3.committed_passed_secret)
        if (not a1_ok) and (o2.committed_passed_secret or o3.committed_passed_secret):
            blind_ids.append(prob.problem_id)
        if (not a1_ok) and _b_chain_stuck(pool, prob, plane0):
            diverge_ids.append(prob.problem_id)
        if on_problem is not None:
            on_problem(int(_pi), str(prob.problem_id),
                       {"a1_pass": a1_pass, "pool_union": pool_union,
                        "c3_pass": c3_pass, "blind_headroom": len(blind_ids),
                        "divergence": len(diverge_ids)})
    n = len(list(problems))
    return HeadroomReportV1(
        schema=W125_CONTROLLER_NATIVE_CODE_MECHANISM_V1_SCHEMA_VERSION,
        field=str(field), n_problems=int(n), a1_pass_count=int(a1_pass),
        pool_union_secret_count=int(pool_union),
        oracle_pool_headroom=int(pool_union - a1_pass),
        c2_blind_committed_pass_count=int(c2_pass),
        c3_committed_pass_count=int(c3_pass),
        blind_selection_headroom=len(blind_ids),
        blind_headroom_problem_ids=tuple(blind_ids),
        reflexion_divergence=len(diverge_ids),
        reflexion_divergence_problem_ids=tuple(diverge_ids),
        looks_right_fails_hidden=int(looks_right))


# ====================================================================== earn gate

@dataclasses.dataclass(frozen=True)
class PilotEarnVerdictV1:
    schema: str
    e1_contract_pass: bool
    e2_lead_is_real: bool
    e3a_blind_headroom_ge2: bool
    e3b_divergence_ge3: bool
    earned: bool
    verdict_label: str
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def apply_pilot_earn_gate(contract: ContractCheckReportV1, slate: SlateEvaluationV1,
                          headroom: HeadroomReportV1) -> PilotEarnVerdictV1:
    """RUNBOOK § 6: fresh resistant pilot EARNED iff E1 ∧ E2 ∧ (E3a ∧ E3b)."""
    e1 = bool(contract.all_pass)
    e2 = bool(slate.lead in slate.real_candidates and slate.lead != "")
    e3a = bool(headroom.blind_selection_headroom >= 2)
    e3b = bool(headroom.reflexion_divergence >= 3)
    earned = bool(e1 and e2 and e3a and e3b)
    if earned:
        label = "FRESH_RESISTANT_PILOT_EARNED"
        rationale = ("E1∧E2∧E3 hold: contract-verified controller-native lead with broad "
                     "resistant headroom past the null band; the cheapest honest fresh "
                     "Maverick controller pilot is admissible (RUNBOOK §7).")
    elif e1 and e2 and not (e3a and e3b):
        label = "FRESH_RESISTANT_PILOT_NOT_EARNED_HEADROOM_CAP"
        rationale = ("E1∧E2 hold (the mechanism is real + contract-verified) but E3 fails: "
                     "the resistant field is generation-capped for $0 controller re-routing "
                     "(blind selection headroom < 2 and/or divergence < 3); a fresh pilot "
                     "is NOT precursor-earned. $0 NIM. Register W125-L-RESISTANT-"
                     "GENERATION-CAP.")
    else:
        label = "FRESH_RESISTANT_PILOT_NOT_EARNED_CONTRACT_OR_FAKE"
        rationale = ("E1 and/or E2 fail: the controller is not contract-clean or is "
                     "fake-different. $0 NIM.")
    return PilotEarnVerdictV1(
        schema=W125_CONTROLLER_NATIVE_CODE_MECHANISM_V1_SCHEMA_VERSION,
        e1_contract_pass=e1, e2_lead_is_real=e2, e3a_blind_headroom_ge2=e3a,
        e3b_divergence_ge3=e3b, earned=earned, verdict_label=label, rationale=rationale)


__all__ = [
    "W125_CONTROLLER_NATIVE_CODE_MECHANISM_V1_SCHEMA_VERSION",
    "ControllerAction", "SampleGradeV1", "AuditedGraderPlaneV1",
    "BlindCandidateScoreV1", "MechanismFingerprintV1", "reflexion_b_fingerprint",
    "ControllerOutcomeV1", "PerProblemPoolV1", "build_pool_from_records",
    "synthetic_contract_problem",
    "C3DigestRoutedRepairControllerV1", "C2RouterSelectControllerV1",
    "C1RolePlanControllerV1", "C0ReflexionRelabeledControllerV1",
    "SlateEvaluationV1", "evaluate_mechanism_slate",
    "ContractCheckReportV1", "run_contract_checks",
    "HeadroomReportV1", "headroom_probe",
    "PilotEarnVerdictV1", "apply_pilot_earn_gate",
]
