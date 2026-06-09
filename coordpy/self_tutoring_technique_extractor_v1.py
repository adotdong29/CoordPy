"""W141 / COO-9 — self-tutoring technique extractor (Lane α core novelty).

Turns a TEAM-DISCOVERED, no-oracle-VERIFIED correct+efficient program into the SAME leak-audited
family-level holed-skeleton ``FamilyTutorV1`` that W140 compiled from the exact-oracle witness — but
here derived from the model's OWN winning sample, with NO oracle.  The discriminating logic (the
ACCEPT/ aggregation decision + its controlling predicate) is blanked into ``__HOLE_*__`` markers by an
AST pass; ``correct_fill`` = the winner's ORIGINAL expressions (so refilling reconstructs the
verified-correct program EXACTLY ⇒ completable BY CONSTRUCTION); ``trivial_fill`` = constant stubs
(so the gate's ``holes_are_substantive`` check bites).  Every extracted tutor is cleared through the
EXISTING ``tutor_leak_gate_v1`` (``spec_override``) + a completability re-run; on ANY failure the
caller DISCARDS the scaffold and KEEPs ≡ A1 (non-negativity preserved, exactly the W139/W140 floor).

This removes the last oracle dependency in the W140 lift: W141 = (W140 holed-skeleton lift) −
(owned-oracle) + (no-oracle self-verified winner as the technique source).  Pure/deterministic except
the already-audited program-execution subprocess used by the completability + substance checks; NO
model inference here (that is the W141 driver/bench).  Explicit-import only; ``__init__.py`` untouched.
"""
from __future__ import annotations

import ast
import dataclasses
from typing import Any, Optional

from .resistant_by_construction_battlefield_v1 import MintedProblemV1, _exec_capture_v1
from .hard_battlefield_slate_v2 import ParserNeutralTemplateV2
from .family_tutor_compiler_v1 import (
    TechniqueSpecV1, FamilyTutorV1, TutorLeakReportV1, tutor_leak_gate_v1,
    HOLE_RE, _parse_headroom_note, TC2_REWRITE, OBS_TIMEOUT, OBS_WRONG_ANSWER)

SELF_TUTORING_TECHNIQUE_EXTRACTOR_V1_SCHEMA_VERSION: str = \
    "coordpy.self_tutoring_technique_extractor_v1.v1"

_MAX_HOLES: int = 4          # over-blanking guard: a teachable scaffold has a few key decisions
_MAX_HOLE_EXPR_TOKENS: int = 40  # do not blank a giant expression (it is not a single decision)


@dataclasses.dataclass(frozen=True)
class ExtractedSkeletonV1:
    """The result of AST hole-derivation from a concrete winning program."""
    skeleton: str
    correct_fill: dict[str, str]      # hole -> the winner's original expression (self-test only)
    trivial_fill: dict[str, str]      # hole -> constant stub (gate hole-substance only)
    n_pred_holes: int
    n_add_holes: int
    ok: bool                          # parsed + found >=1 substantive hole site, holes <= cap
    reason: str

    @property
    def holes(self) -> tuple[str, ...]:
        return tuple(sorted(self.correct_fill))


# ----------------------------------------------------------------- AST hole derivation

def _parents(tree: ast.AST) -> dict[int, ast.AST]:
    par: dict[int, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            par[id(child)] = node
    return par


def _accumulator_names(tree: ast.AST) -> list[str]:
    """The variable(s) printed at the end — the answer accumulator(s). Robust across phrasings:
    print(X), print(X % M), sys.stdout.write(str(X))."""
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and (
                (isinstance(node.func, ast.Name) and node.func.id == "print")
                or (isinstance(node.func, ast.Attribute) and node.func.attr == "write")):
            for arg in node.args:
                for sub in ast.walk(arg):
                    if isinstance(sub, ast.Name) and sub.id not in names:
                        names.append(sub.id)
    return names


def _trace_through_functions(tree: ast.AST, accs: list[str]) -> list[str]:
    """Real models compute the answer in a HELPER function (``result = solve(arr); print(result)`` or
    ``print(solve(arr))``).  Add the returned-variable names of any local function whose call-result is
    printed or assigned to a printed name — so the accumulator search reaches the running ``total +=``
    inside the helper.  Iterates to a fixed point over chained helpers."""
    funcs = {f.name: f for f in ast.walk(tree) if isinstance(f, ast.FunctionDef)}
    if not funcs:
        return accs
    accs = list(accs)

    def _returned_names(fname: str) -> list[str]:
        out: list[str] = []
        for ret in ast.walk(funcs[fname]):
            if isinstance(ret, ast.Return) and ret.value is not None:
                for sub in ast.walk(ret.value):
                    if isinstance(sub, ast.Name):
                        out.append(sub.id)
        return out

    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            # result = solve(...)  where result is (transitively) printed
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                fn = node.value.func
                fname = fn.id if isinstance(fn, ast.Name) else None
                tgts = [t.id for t in node.targets if isinstance(t, ast.Name)]
                if fname in funcs and any(t in accs for t in tgts):
                    for nm in _returned_names(fname):
                        if nm not in accs:
                            accs.append(nm); changed = True
            # print(solve(...))  directly
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print":
                for arg in node.args:
                    for sub in ast.walk(arg):
                        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name) and sub.func.id in funcs:
                            for nm in _returned_names(sub.func.id):
                                if nm not in accs:
                                    accs.append(nm); changed = True
    return accs


def _enclosing_loop_body(node: ast.AST, par: dict[int, ast.AST]) -> Optional[list]:
    """The statement list of the innermost For/While whose body (or orelse) contains ``node``."""
    cur = node
    while id(cur) in par:
        p = par[id(cur)]
        if isinstance(p, (ast.For, ast.While)):
            for blk in (p.body, getattr(p, "orelse", [])):
                if any(cur is s or _contains(s, cur) for s in blk):
                    return blk
        cur = p
    return None


def _contains(root: ast.AST, target: ast.AST) -> bool:
    return any(child is target for child in ast.walk(root))


def _stmt_index_in(block: list, node: ast.AST) -> Optional[int]:
    for i, s in enumerate(block):
        if s is node or _contains(s, node):
            return i
    return None


def derive_holes_from_ast_v1(code: str) -> ExtractedSkeletonV1:
    """Blank the ACCEPT/aggregation decision(s) of a concrete winning program into ``__HOLE_*__``
    markers, recording the original expressions as ``correct_fill`` (⇒ completable by construction).

    v1 handles the two dominant resistant-COMPLEXITY shapes:
      (a) ``acc += <expr>`` INSIDE ``if <pred>:``  (sort+two-pointer, count-pairs) — blank both;
      (b) ``while <shrink>: ...`` then ``acc += <expr>`` at the same block level (sliding-window,
          count-subarrays / bounded-subarray) — blank the aggregation + the immediately-preceding
          shrink predicate.
    Auxiliary maintenance loops (monotonic-deque ``while stk and ...: stk.pop()``) are NOT blanked
    (they are not the controlling predicate of the accumulator); under-blanked extractions are caught
    downstream by the gate's reference-paste tripwire and discarded.  Returns ``ok=False`` (caller
    KEEPs) on any parse failure / no-accumulator / no-substantive-site / over-cap."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ExtractedSkeletonV1("", {}, {}, 0, 0, False, "winner_does_not_parse")
    par = _parents(tree)
    accs = _accumulator_names(tree)
    accs = _trace_through_functions(tree, accs)   # real models put the accumulator inside a helper fn
    if not accs:
        return ExtractedSkeletonV1("", {}, {}, 0, 0, False, "no_printed_accumulator")

    add_nodes: list[ast.AST] = []      # the .value of acc += <value>
    pred_nodes: list[ast.AST] = []     # the controlling predicate (If.test / shrink While.test)
    for node in ast.walk(tree):
        is_acc_aug = (isinstance(node, ast.AugAssign)
                      and isinstance(node.op, (ast.Add, ast.Sub))
                      and isinstance(node.target, ast.Name) and node.target.id in accs)
        is_acc_set = (isinstance(node, ast.Assign) and len(node.targets) == 1
                      and isinstance(node.targets[0], ast.Name) and node.targets[0].id in accs
                      and isinstance(node.value, ast.BinOp))  # acc = acc + <expr> style
        if not (is_acc_aug or is_acc_set):
            continue
        val = node.value
        # W142b — counting-by-accept-predicate shape (`count += 1` per item that SATISFIES the
        # condition): the contribution is a bare constant, so the discriminating DECISION is the
        # controlling ACCEPT predicate (which items count), not the +1.  Blank the predicate, NOT the
        # constant add (additive: count_pairs/NSL have non-constant adds and are unaffected; this only
        # rescues the constant-add case v1 dropped as `no_accumulator_update`).
        if isinstance(val, ast.Constant):
            pred = _controlling_predicate(node, par)
            if pred is not None and not _is_trivial_test(pred):
                pred_nodes.append(pred)
            continue
        if len(ast.dump(val)) > 0 and len(_unparse(val).split()) > _MAX_HOLE_EXPR_TOKENS:
            continue
        add_nodes.append(val)
        # find the controlling predicate
        pred = _controlling_predicate(node, par)
        if pred is not None and not _is_trivial_test(pred):
            pred_nodes.append(pred)

    if not add_nodes and not pred_nodes:
        return ExtractedSkeletonV1("", {}, {}, 0, 0, False, "no_accumulator_update")

    # dedupe predicate nodes by identity
    seen: set[int] = set()
    pred_nodes = [p for p in pred_nodes if not (id(p) in seen or seen.add(id(p)))]

    correct_fill: dict[str, str] = {}
    trivial_fill: dict[str, str] = {}
    repl: dict[int, str] = {}
    for k, n in enumerate(add_nodes):
        m = f"__HOLE_ADD_{k}__"
        correct_fill[m] = _unparse(n)
        trivial_fill[m] = "0"
        repl[id(n)] = m
    for k, n in enumerate(pred_nodes):
        m = f"__HOLE_PRED_{k}__"
        correct_fill[m] = _unparse(n)
        trivial_fill[m] = "False"
        repl[id(n)] = m

    if len(correct_fill) > _MAX_HOLES:
        return ExtractedSkeletonV1("", {}, {}, 0, 0, False, f"over_cap_{len(correct_fill)}_holes")

    holed = _HoleReplacer(repl).visit(tree)  # transform the SAME tree the node ids came from
    ast.fix_missing_locations(holed)
    try:
        skeleton = _unparse(holed)
    except Exception as exc:  # noqa: BLE001 — unparse can choke on the placeholder graft
        return ExtractedSkeletonV1("", {}, {}, 0, 0, False, f"unparse_failed:{type(exc).__name__}")
    # the placeholder Name nodes must survive as HOLE markers in the source
    if not all(h in skeleton for h in correct_fill):
        return ExtractedSkeletonV1("", {}, {}, 0, 0, False, "holes_lost_in_unparse")
    return ExtractedSkeletonV1(
        skeleton=skeleton, correct_fill=correct_fill, trivial_fill=trivial_fill,
        n_pred_holes=len(pred_nodes), n_add_holes=len(add_nodes), ok=True, reason="ok")


def _controlling_predicate(acc_stmt: ast.AST, par: dict[int, ast.AST]) -> Optional[ast.AST]:
    """The predicate that decides whether/how the accumulator updates: the test of the nearest If
    ancestor whose body holds the update (pattern a), else the test of the While immediately
    preceding the update in the same block (pattern b, the window-shrink)."""
    # pattern (a): nearest enclosing If inside a loop
    cur = acc_stmt
    while id(cur) in par:
        p = par[id(cur)]
        if isinstance(p, ast.If) and any(cur is s or _contains(s, cur) for s in p.body):
            # only if this If is itself inside a loop (a per-iteration accept test)
            if _has_loop_ancestor(p, par):
                return p.test
        if isinstance(p, (ast.For, ast.While)):
            break  # left the immediate loop body without finding a guarding If
        cur = p
    # pattern (b): the While immediately preceding the update at the same block level
    block = _enclosing_loop_body(acc_stmt, par)
    if block is not None:
        idx = _stmt_index_in(block, acc_stmt)
        if idx is not None:
            for j in range(idx - 1, -1, -1):
                if isinstance(block[j], ast.While):
                    return block[j].test
    return None


def _has_loop_ancestor(node: ast.AST, par: dict[int, ast.AST]) -> bool:
    cur = node
    while id(cur) in par:
        cur = par[id(cur)]
        if isinstance(cur, (ast.For, ast.While)):
            return True
    return False


def _is_trivial_test(test: ast.AST) -> bool:
    """A predicate hole must be a real decision — not ``while stack`` (a truthiness maintenance loop)
    or a bare constant."""
    if isinstance(test, (ast.Constant, ast.Name)):
        return True
    return False


class _HoleReplacer(ast.NodeTransformer):
    def __init__(self, repl: dict[int, str]) -> None:
        self._repl = repl

    def visit(self, node: ast.AST) -> ast.AST:
        if id(node) in self._repl:
            return ast.copy_location(ast.Name(id=self._repl[id(node)], ctx=ast.Load()), node)
        return super().visit(node)

    def generic_visit(self, node: ast.AST) -> ast.AST:
        return super().generic_visit(node)


def _unparse(node: ast.AST) -> str:
    return ast.unparse(node)


# ----------------------------------------------------------------- compile to a FamilyTutorV1

@dataclasses.dataclass(frozen=True)
class SelfTutorCompileReportV1:
    compiled: bool
    reason: str
    extracted: Optional[ExtractedSkeletonV1]
    leak: Optional[TutorLeakReportV1]
    completable: Optional[bool]
    stub_fails_secret: Optional[bool]

    def to_dict(self) -> dict[str, Any]:
        return {"compiled": self.compiled, "reason": self.reason,
                "n_holes": len(self.extracted.correct_fill) if self.extracted else 0,
                "leaked": (self.leak.leaked if self.leak else None),
                "completable": self.completable, "stub_fails_secret": self.stub_fails_secret}


def _build_spec(template: ParserNeutralTemplateV2, ex: ExtractedSkeletonV1) -> TechniqueSpecV1:
    why, needs = _parse_headroom_note(template.headroom_note)
    return TechniqueSpecV1(
        algo_sig=template.minted.algo_sig,
        technique_name=needs or "an efficient method",
        key_move=("Work out the blanked DECISIONS for this problem — the accept/shrink condition and "
                  "what each accepted step contributes — instead of the naive scan."),
        primitive_hint="", bug_warnings=(f"The naive way {why}." if why else "",),
        invariants=(), skeleton=ex.skeleton, correct_fill=dict(ex.correct_fill),
        trivial_fill=dict(ex.trivial_fill))


def compile_tutor_from_winner_v1(
        winner_code: str, template: ParserNeutralTemplateV2, problem: MintedProblemV1, *,
        timeout_s: float = 8.0) -> tuple[Optional[FamilyTutorV1], SelfTutorCompileReportV1]:
    """Self-derive a leak-audited holed-skeleton ``FamilyTutorV1`` from a no-oracle-VERIFIED winning
    program.  Returns ``(tutor, report)`` or ``(None, report)`` when the extraction must be DISCARDED
    (caller KEEPs ≡ A1).  Validation order: AST extract → completability (refill→secret pass) →
    hole-substance (trivial stub → secret FAIL) → the W140 leak gate (spec_override)."""
    ex = derive_holes_from_ast_v1(winner_code)
    if not ex.ok:
        return None, SelfTutorCompileReportV1(False, f"extract:{ex.reason}", ex, None, None, None)

    spec = _build_spec(template, ex)
    secret = list(problem.secret_cases) or list(problem.samples)

    # completability: refilling correct_fill must reconstruct a program that PASSES the hidden bank.
    filled = spec.fill(spec.correct_fill)
    completable = _passes_all(filled, secret, timeout_s)
    if not completable:
        return None, SelfTutorCompileReportV1(False, "not_completable", ex, None, False, None)

    # hole-substance: the trivially-stubbed skeleton must FAIL the hidden bank (holes carry the logic).
    stub = spec.fill(spec.trivial_fill)
    stub = HOLE_RE.sub("0", stub)
    stub_fails = not _passes_all(stub, secret, timeout_s)
    if not stub_fails:
        return None, SelfTutorCompileReportV1(False, "holes_not_substantive", ex, None, True, False)

    why, needs = _parse_headroom_note(template.headroom_note)
    routes = (
        (OBS_TIMEOUT, f"your program is too slow (times out on large inputs) — use {needs or 'the '
                      'efficient technique'} and fill the blanked decisions."),
        (OBS_WRONG_ANSWER, "your program returns the wrong result on a hidden input — the blanked "
                           "accept/aggregation decision is where the logic is missing."),
        ("DEFAULT", f"fix your approach using {needs or 'the efficient technique'}; the skeleton's "
                    "blanks are the decisions that make it correct."),
    )
    tutor = FamilyTutorV1(
        family=template.minted.family, algo_sig=template.minted.algo_sig, tc_kind=TC2_REWRITE,
        technique_name=spec.technique_name, budget_fact=why, key_move=spec.key_move,
        primitive_hint="", bug_warnings=tuple(w for w in spec.bug_warnings if w),
        invariants=(), skeleton=ex.skeleton, rewrite_routes=routes, stages=())

    leak = tutor_leak_gate_v1(tutor, template, problem, timeout_s=timeout_s, spec_override=spec)
    if leak.leaked:
        return None, SelfTutorCompileReportV1(False, "leak_gate", ex, leak, True, True)
    return tutor, SelfTutorCompileReportV1(True, "ok", ex, leak, True, True)


def _passes_all(code: str, cases: list, timeout_s: float) -> bool:
    if not cases:
        return False
    for inp, exp in cases:
        r = _exec_capture_v1(code, inp, timeout_s=float(timeout_s))
        if r.timed_out or r.returncode != 0 or r.stdout.strip() != exp.strip():
            return False
    return True
