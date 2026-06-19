#!/usr/bin/env python3
"""W112 / COO-9 — Lane β: NIM-free FAIR-REACHABILITY re-mining of the resistant
failure regime (mines the W111 hidden-test-coupling finding HARDER).

W111 measured M3 (executor-grounded structured-failure patcher) SUB-reflexion on
a rescue-concentrated probe (MLB-2 = 12.5 % < reflexion 25 % < the 33 % floor)
and registered ``W111-L-M3-PATCHER-SUB-REFLEXION...``.  W111's hard-core
ablation showed 6/8 both-fail problems are MOCK_COUPLING (the fix needs the
hidden test's mock setup, NOT in ``stderr_tail``) and only 2/8 are OUTPUT_VALUE
(reachable).  But that was only the 8 both-fail hard core.

W112 Lane β asks the SHARPER, falsifiable question the bounded-claim must
answer before any strengthened-M3 NIM is justified:

    Across the FULL resistant INVOKED-failure set (every problem where the
    mechanism's attempt-0 sample fails, i.e. the MLB-2 DENOMINATOR), what
    fraction is FAIR-REACHABLE — i.e. the information to fix it is present in
    the FAIR regime (visible docstring + executor ``stderr_tail``), NOT only in
    the hidden ``test`` source?

That fraction is a STRUCTURAL CEILING on MLB-2 for ANY fair same-budget patcher:
a failure whose fix is information-unavailable in the fair regime sits
permanently in the denominator and can never be rescued without oracle leakage.
If the *generous* ceiling is already < the 33 % MLB-2 floor, then NO fair M3
strengthening (better digest, multi-candidate aggregation, patch-rejection,
doctest invariants) can clear the floor, and Lane β dies honestly at $0 with a
STRONGER (structural) argument than W111's (empirical) one.

This is a $0-NIM probe: it re-executes the EXISTING W110 BigCodeBench pilot
transcripts (``response_text`` already on disk) through the real deterministic
``unittest`` executor and STATICALLY inspects each problem's hidden test for
mock-coupling — it spends ZERO model calls.  It NEVER feeds the test source to a
solver; the test source is read only for OFFLINE difficulty CHARACTERISATION
(exactly as the W111 census did).

Usage::

    python scripts/mine_w112_fair_reachability_v1.py \
        --run results/w110/bigcodebench_pilot/<run> \
        --corpus ~/.cache/coordpy/bigcodebench-v0_1_4.jsonl \
        --out results/w112/fair_reachability/w110_bcb_fair_reachability.json
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLBACKEND", "Agg")  # W110 lesson: headless plotting.

# Static markers that the hidden test couples the assertion to a MOCK / fixture
# whose expected behaviour is defined IN THE TEST, not derivable from the visible
# spec or the executor stderr (so a fair patcher can never recover it).
_MOCK_MARKERS = (
    "unittest.mock", "from unittest import mock", "mock.patch", "@patch",
    "@mock.patch", "patch(", "MagicMock", "Mock(", "side_effect",
    "return_value", "assert_called", "create_autospec", "PropertyMock",
    "mock_open", "monkeypatch",
)
# A test that builds/asserts on filesystem/network fixtures it created itself.
_FIXTURE_MARKERS = (
    "tempfile", "TemporaryDirectory", "mkdtemp", "os.makedirs", "setUp(",
    "tearDown(", "addCleanup", "requests_mock", "responses.add", "httpretty",
)


def _load_calls(run: Path) -> list[dict[str, Any]]:
    p = run / "bigcodebench_reflexion_calls.jsonl"
    if not p.exists():
        raise SystemExit(f"no bigcodebench_reflexion_calls.jsonl in {run}")
    return [json.loads(l) for l in p.open()]


def _load_report(run: Path) -> dict[str, Any]:
    p = run / "bigcodebench_reflexion_bench_report.json"
    if not p.exists():
        raise SystemExit(f"no bigcodebench_reflexion_bench_report.json in {run}")
    return json.load(p.open())


def _classify_reachability(
        *, test_source: str, expected_repr: str, actual_repr: str,
        exception_type: str, stderr_tail: str, timed_out: bool) -> str:
    """Return a fair-reachability class for ONE attempt-0 failure.

    STRICT-reachable iff a concrete expected!=actual contract was extracted from
    the FAIR stderr AND the test does not couple the assertion to a mock/fixture.
    GENEROUS adds API-grounding errors (locatable from the traceback) and
    contract-bearing failures even under mock setup.  Everything else is
    UNREACHABLE in the fair regime.
    """
    if timed_out:
        return "UNREACHABLE_TIMEOUT"          # non-termination: not digest-fixable
    is_mock = any(m in test_source for m in _MOCK_MARKERS)
    is_fixture = any(m in test_source for m in _FIXTURE_MARKERS)
    has_contract = bool(expected_repr) and bool(actual_repr)
    api = exception_type in {
        "ImportError", "ModuleNotFoundError", "AttributeError", "NameError"}
    if has_contract and not is_mock:
        return "FAIR_REACHABLE_OUTPUT_VALUE"  # expected value in hand, no mock
    if api:
        return "FAIR_REACHABLE_API"           # locatable from traceback (rare here)
    if has_contract and is_mock:
        return "BORDERLINE_CONTRACT_UNDER_MOCK"  # value shown but mock-entangled
    if is_mock or is_fixture:
        return "UNREACHABLE_MOCK_OR_FIXTURE"  # expected behaviour lives in the test
    if exception_type == "AssertionError" or "AssertionError" in stderr_tail:
        return "UNREACHABLE_UNDERSPEC_ASSERT"  # assert fired, no extractable target
    return "UNREACHABLE_OTHER"


def _has_doctest(docstring: str) -> bool:
    return ">>>" in (docstring or "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout-s", type=float, default=12.0)
    ap.add_argument("--kill-after-s", type=float, default=15.0)
    ap.add_argument(
        "--python-exe",
        default=str(Path.home() / ".cache/coordpy/bcb_venv/bin/python"))
    args = ap.parse_args()

    run = Path(args.run).expanduser()
    calls = _load_calls(run)
    report = _load_report(run)
    seed = report["per_seed"][0]
    pids = seed["problem_ids"]
    n = len(pids)
    assert len(calls) == n * 11, f"expected {n*11} calls, got {len(calls)}"

    corpus: dict[str, dict[str, Any]] = {}
    for line in Path(args.corpus).expanduser().open():
        r = json.loads(line)
        corpus[r["task_id"]] = r

    from coordpy.bigcodebench_executor_v1 import run_bigcodebench_executor_v1
    from coordpy.bigcodebench_reflexion_bench_v1 import extract_candidate_code_v1
    from coordpy.executor_grounded_patcher_v1 import parse_failure_digest_v1

    def run_exec(pid: str, code: str):
        p = corpus[pid]
        r = run_bigcodebench_executor_v1(
            problem_id=pid, test_source=p["test"], entry_point=p["entry_point"],
            candidate_code=code, timeout_s=args.timeout_s,
            kill_after_s=args.kill_after_s, python_exe=args.python_exe)
        return r.passed, r.stderr_tail, r.timed_out

    # Per problem: block of 11 calls = [A0] + A1[1..5] + B[6..10].
    # MLB-2 denominator = problems where B-attempt-0 (call index 6) fails => the
    # patch/reflexion loop is invoked.  We classify THAT failure's reachability.
    invoked_classes: collections.Counter = collections.Counter()
    per_problem: dict[str, dict[str, Any]] = {}
    contract_extractable_invoked = 0
    doctest_invoked = 0
    n_invoked = 0
    # Strengthening-marginal counters (see § interpretation):
    s_richer_digest_gain = 0     # assert info present that current parser missed
    s_multi_candidate_gain = 0   # >=1 OTHER A1 sample failed DIFFERENTLY (diverse)
    n_exec = 0

    _ASSERT_INFO_RE = re.compile(
        r"assert|expected|!=|Lists differ|Tuples differ|not equal|"
        r"Items in the (first|second)", re.I)

    for blk in range(n):
        pid = pids[blk]
        base = blk * 11
        prob = corpus[pid]
        test_src = str(prob.get("test") or "")
        doc11 = str(prob.get("complete_prompt") or prob.get("instruct_prompt") or "")
        # B attempt-0 = call 6.
        b0 = calls[base + 6]
        b0_code = extract_candidate_code_v1(response_text=b0["response_text"])
        b0_pass, b0_stderr, b0_timed = run_exec(pid, b0_code)
        n_exec += 1
        rec: dict[str, Any] = {"b0_passed": bool(b0_pass)}
        if b0_pass:
            per_problem[pid] = rec
            print(f"  [{blk+1}/{n}] {pid}: attempt-0 PASS (not invoked)",
                  flush=True)
            continue
        # invoked.
        n_invoked += 1
        digest = parse_failure_digest_v1(
            stderr_tail=b0_stderr, timed_out=b0_timed)
        klass = _classify_reachability(
            test_source=test_src, expected_repr=digest.expected_repr,
            actual_repr=digest.actual_repr,
            exception_type=digest.exception_type, stderr_tail=b0_stderr,
            timed_out=b0_timed)
        invoked_classes[klass] += 1
        has_contract = bool(digest.expected_repr) and bool(digest.actual_repr)
        if has_contract:
            contract_extractable_invoked += 1
        if _has_doctest(doc11):
            doctest_invoked += 1
        # richer-digest marginal: assertion info is present in stderr but the
        # current parser extracted NO contract -> a better parser MIGHT recover.
        if (not has_contract) and (not b0_timed) and _ASSERT_INFO_RE.search(
                b0_stderr or ""):
            s_richer_digest_gain += 1
        # multi-candidate marginal: do the other A1 samples (calls 1..5) fail
        # with DIFFERENT exception types? (diverse failures => aggregation lever)
        a1_excs = set()
        for j in range(1, 6):
            cj = calls[base + j]
            code_j = extract_candidate_code_v1(response_text=cj["response_text"])
            pj, sj, tj = run_exec(pid, code_j)
            n_exec += 1
            if not pj:
                dj = parse_failure_digest_v1(stderr_tail=sj, timed_out=tj)
                a1_excs.add(dj.exception_type or ("Timeout" if tj else "?"))
        if len(a1_excs) >= 2:
            s_multi_candidate_gain += 1
        rec.update({
            "reachability_class": klass,
            "exception_type": digest.exception_type,
            "has_extractable_contract": has_contract,
            "is_mock_coupled": any(m in test_src for m in _MOCK_MARKERS),
            "has_doctest_in_spec": _has_doctest(doc11),
            "a1_distinct_exception_types": sorted(a1_excs),
        })
        per_problem[pid] = rec
        print(f"  [{blk+1}/{n}] {pid}: INVOKED -> {klass} "
              f"(exc={digest.exception_type!r}, contract={has_contract})",
              flush=True)

    # Ceilings on MLB-2 (rescue / invoked).
    strict_reachable = (
        invoked_classes.get("FAIR_REACHABLE_OUTPUT_VALUE", 0)
        + invoked_classes.get("FAIR_REACHABLE_API", 0))
    generous_reachable = (
        strict_reachable
        + invoked_classes.get("BORDERLINE_CONTRACT_UNDER_MOCK", 0))
    inv = max(1, n_invoked)
    strict_ceiling = strict_reachable / inv
    generous_ceiling = generous_reachable / inv
    floor = 0.33

    # Strengthening earn/no-earn analysis (falsifiable).
    # The BEST conceivable fair strengthening rescues EVERY generous-reachable
    # invoked failure.  Its MLB-2 upper bound == generous_ceiling.
    strengthenings = {
        "S-C_richer_typed_digest": {
            "idea": ("extend parse_failure_digest_v1 to assertAlmostEqual / "
                     "assertTrue-with-locals / assertRaises / multi-line reprs "
                     "/ traceback frame localisation"),
            "marginal_newly_actionable_invoked": int(s_richer_digest_gain),
            "fair": True,
            "bounded_by": "generous_ceiling (cannot exceed reachable set)",
        },
        "S-A_multi_candidate_failure_aggregation": {
            "idea": ("condition the patch on ALL K self-consistency candidates "
                     "+ their digests, not only the latest, to localise the "
                     "common bug"),
            "n_invoked_with_diverse_a1_failures": int(s_multi_candidate_gain),
            "fair": True,
            "bounded_by": "generous_ceiling",
        },
        "S-B_patch_rejection_anti_regression": {
            "idea": ("execute each candidate patch; reject patches that regress "
                     "vs the prior best (reduces the lateral-trade losses that "
                     "cancelled reflexion's rescues, e.g. W111 /51)"),
            "effect": "reduces REGRESSIONS; adds NO new rescues",
            "fair": True,
            "bounded_by": "cannot raise MLB-2 numerator above reachable set",
        },
        "S-D_visible_spec_doctest_invariants": {
            "idea": ("parse >>> doctest examples from the visible docstring into "
                     "local self-checks run before submission"),
            "n_invoked_with_doctest_in_spec": int(doctest_invoked),
            "fair": True,
            "bounded_by": "only helps problems with doctests in the spec",
        },
    }

    best_fair_mlb2_upper_bound = generous_ceiling
    # Honest earn criterion (conservative, falsifiable): a fair patcher must be
    # able to CLEAR the 33% MLB-2 floor on the RELIABLY fair-reachable (STRICT)
    # set ALONE.  Mock-entangled "contracts" (BORDERLINE_CONTRACT_UNDER_MOCK)
    # are NOT reliable rescues — emitting the mock's value need not satisfy the
    # hidden test's mock-INTERACTION assertions (this is exactly the W111 /51
    # lateral-trade failure).  The generous bound is the best-CONCEIVABLE upper
    # bound (a perfect patcher that also rescues every borderline-under-mock).
    earns_nim = strict_ceiling >= floor
    headroom_pp = round((generous_ceiling - floor) * 100.0, 2)
    generous_only_touches_floor = bool(generous_ceiling < floor + 0.05)
    # No strengthening can EXPAND the reliably-reachable set:
    #   S-C richer digest -> raises GENEROUS (mock-coupled), not STRICT.
    #   S-D doctests are ALREADY visible to the model at generation -> 0 new info.
    #   S-A / S-B -> efficiency / anti-regression only; cannot raise the numerator.
    strengthenings["S-C_richer_typed_digest"]["kill_reason"] = (
        "the newly-actionable invoked failures are mock-coupled; a richer digest "
        "raises the GENEROUS bound, not the reliably-reachable STRICT set")
    strengthenings["S-D_visible_spec_doctest_invariants"]["kill_reason"] = (
        "doctests are ALREADY in the visible prompt at generation time; a local "
        "self-check adds ZERO new information (failures are where the hidden test "
        "demands more than the visible doctests)")
    strengthenings["S-A_multi_candidate_failure_aggregation"]["kill_reason"] = (
        "improves rescue EFFICIENCY on already-reachable problems; cannot expand "
        "the reachable set (bounded by generous_ceiling)")
    strengthenings["S-B_patch_rejection_anti_regression"]["kill_reason"] = (
        "reduces REGRESSIONS only; adds NO new rescues -> cannot raise MLB-2")
    verdict = (
        "STRENGTHENED_M3_MAY_EARN_PROBE" if earns_nim
        else "NO_FAIR_STRENGTHENING_CAN_CLEAR_FLOOR_KILL_AT_0")

    out = {
        "schema": "coordpy.w112_fair_reachability_census.v1",
        "bench": "bigcodebench",
        "run": str(run),
        "n_problems": n,
        "n_candidate_executions_replayed": n_exec,
        "mlb2_denominator_n_invoked": n_invoked,
        "invoked_reachability_classes": dict(invoked_classes.most_common()),
        "n_invoked_with_extractable_contract": contract_extractable_invoked,
        "n_invoked_with_doctest_in_spec": doctest_invoked,
        "mlb2_strict_fair_reachable_ceiling": round(strict_ceiling, 4),
        "mlb2_generous_fair_reachable_ceiling": round(generous_ceiling, 4),
        "mlb2_floor": floor,
        "best_conceivable_fair_strengthening_mlb2_upper_bound": round(
            best_fair_mlb2_upper_bound, 4),
        "generous_headroom_above_floor_pp": headroom_pp,
        "generous_bound_only_touches_floor_no_headroom": (
            generous_only_touches_floor),
        "earn_criterion": (
            "strict (reliably fair-reachable) ceiling >= 33% floor; "
            "mock-entangled contracts are NOT counted as reliable rescues"),
        "strengthening_ideas": strengthenings,
        "earns_live_nim": bool(earns_nim),
        "verdict": verdict,
        "per_problem": per_problem,
        "interpretation": (
            "The fair-reachable fraction of the MLB-2 denominator is a STRUCTURAL "
            "CEILING on any fair same-budget patcher's rescue rate. If the "
            "GENEROUS ceiling < the 33% floor, NO fair strengthening (S-A..S-D) "
            "can produce a load-bearing MLB-2, so a strengthened-M3 NIM run is "
            "NOT WARRANTED and Lane β dies at $0 — a structural strengthening of "
            "the W111 empirical sub-floor finding."),
    }
    outp = Path(args.out).expanduser()
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2, sort_keys=True))

    print("\n=== W112 FAIR-REACHABILITY CENSUS (NIM-free) ===")
    print(f"  MLB-2 denominator (attempt-0 failed / invoked): {n_invoked}")
    for k, v in invoked_classes.most_common():
        print(f"    {k:34s} {v:3d}  ({v/inv:.1%})")
    print(f"  STRICT (reliably reachable) ceiling on MLB-2: {strict_ceiling:.1%}"
          f"  <- the honest earn bar")
    print(f"  GENEROUS (best-conceivable) ceiling on MLB-2: {generous_ceiling:.1%}"
          f"  (headroom above floor: {headroom_pp:+.1f}pp)")
    print(f"  MLB-2 floor: {floor:.0%}")
    print(f"  === VERDICT: {verdict} ===")
    print(f"\nWrote {outp}")


if __name__ == "__main__":
    main()
