"""W129 — public-signal selection oracle tests (direct-exec; no pytest dependency).

Run: python3 tests/test_w129_public_signal_selection_oracle.py
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import coordpy  # noqa: E402
import coordpy.public_signal_selection_oracle_v1 as S  # noqa: E402
from coordpy.icpc_reflexion_bench_v1 import IcpcPilotProblemV1  # noqa: E402
from coordpy.role_diverse_algorithm_search_v1 import CandidateImplV1, RoleArtifactsV1  # noqa: E402

_PASS = [0]
_FAIL = [0]


def check(name, cond):
    if cond:
        _PASS[0] += 1
        print(f"  ok   {name}")
    else:
        _FAIL[0] += 1
        print(f"  FAIL {name}")


def _prob(samples, secret=()):
    return IcpcPilotProblemV1(problem_id="t", short_name="t", source_repo="t",
                              contest_date="2020-01-01", statement="echo the number",
                              kind="passfail", float_tol=0.0, samples=tuple(samples),
                              secret_cases=tuple(secret))


def _arts(cx, inv=("x>=0",)):
    return RoleArtifactsV1(spec="s", invariants=tuple(inv), complexity="O(1)",
                           sketches=(), counterexamples=tuple(cx), raw="")


def test_stable_boundary():
    check("version 1.2.0", coordpy.__version__ == "1.2.1")
    check("sdk v3.43", coordpy.SDK_VERSION == "coordpy.sdk.v3.43")
    check("module not in public surface", "public_signal_selection_oracle_v1"
          not in getattr(coordpy, "__all__", []))


def test_fake_selection_control():
    ctl = S.fake_selection_control_v1()
    check("control passes (SO2+SO4 abstain on no-evidence tie)", ctl["control_passes"])
    check("so2 abstained on tie", ctl["so2_abstained_on_tie"])
    check("so2 evidence_used False on tie", ctl["so2_evidence_used"] is False)
    check("so4 abstained on tie", ctl["so4_abstained_on_tie"])


def test_trust_machinery_kill():
    rep = S.examine_trust_machinery_applicability_v1()
    check("substrate controller literal bridge KILLED",
          rep["substrate_controller_literal_bridge_killed"] is True)
    ctrl = rep["modules"]["trust_weighted_consensus_controller"]
    check("controller is latent-specific", ctrl.get("latent_specific") is True)
    check("controller not code-candidate applicable",
          ctrl.get("code_candidate_applicable") is False)
    check("so4 trust signal is honest code proxy", "falsifier_survival" in rep["so4_trust_signal"])


def test_so_separable_falsifier_commits():
    # 3 candidates: A0,C2 correct (echo n); B1 wrong (crashes on a derived case)
    prob = _prob([("3\n", "3\n")])
    arts = _arts([("0\n", None)])  # derived case 0 -> B1 divides-by-zero crash, A0/C2 print 0
    good = "n=int(input())\nprint(n)"
    bad = "n=int(input())\nprint(n//(n-0) if False else (1//n if n==0 else n))"  # crashes on 0
    impls = [CandidateImplV1("A0", good, True), CandidateImplV1("B1", bad, True),
             CandidateImplV1("C2", good, True)]
    s1 = S.select_so_v1(prob, impls, arts, variant="SO1")
    check("SO1 commits a survivor on separable case", not s1.abstained)
    check("SO1 does NOT commit the crashing B1", s1.committed_label != "B1")
    check("SO1 evidence_used (falsifier eliminated B1)", s1.evidence_used)


def test_so_under_determined_abstains():
    # two STRUCTURALLY-DISTINCT but behaviourally-identical survivors (the pawnshop pattern)
    prob = _prob([("5\n", "5\n")])
    arts = _arts([("7\n", None)])
    code_a = "n=int(input())\nprint(n)"
    code_b = "n=int(input())\nresult=n\nprint(result)"  # different AST, identical behaviour
    impls = [CandidateImplV1("A0", code_a, True), CandidateImplV1("B1", code_b, True)]
    s2 = S.select_so_v1(prob, impls, arts, variant="SO2")
    s4 = S.select_so_v1(prob, impls, arts, variant="SO4")
    check("SO2 ABSTAINS on under-determined structural tie (no mis-commit)", s2.abstained)
    check("SO4 ABSTAINS on under-determined structural tie (no mis-commit)", s4.abstained)
    check("SO2 evidence_used False on no-discriminator tie", s2.evidence_used is False)
    # structurally IDENTICAL survivors are the SAME program -> safe to commit (not a mis-commit)
    same = [CandidateImplV1("A0", code_a, True), CandidateImplV1("B1", code_a, True)]
    s2s = S.select_so_v1(prob, same, arts, variant="SO2")
    check("SO2 commits structurally-identical survivors (same program)", not s2s.abstained)
    check("SO2 same-program commit is not evidence-driven", s2s.evidence_used is False)


def test_verifier_choice_parser():
    labels = ["A0", "B1", "C2"]
    check("parses CHOOSE B1", S.parse_verifier_choice_v1("reasoning...\nCHOOSE B1", labels) == "B1")
    check("parses trailing ABSTAIN",
          S.parse_verifier_choice_v1("cannot tell\nABSTAIN", labels) == "ABSTAIN")
    check("last CHOOSE wins",
          S.parse_verifier_choice_v1("CHOOSE A0\n...\nCHOOSE C2", labels) == "C2")
    check("unparseable -> None", S.parse_verifier_choice_v1("hmm no answer", labels) is None)


def test_verifier_prompt_no_leakage():
    # the verifier prompt must contain ONLY public statement/samples/invariants/candidate code
    prob = _prob([("4\n", "4\n")], secret=[("SECRET_INPUT_999\n", "SECRET_ANS_999\n")])
    arts = _arts([("2\n", None)])
    code = "n=int(input())\nprint(n)"
    impls = [CandidateImplV1("A0", code, True), CandidateImplV1("B1", code, True)]
    grades, _ci = S.grade_candidates_v1(prob, arts, impls)
    prompt = S.build_verifier_final_prompt_v1(prob, arts, grades, impls)
    check("verifier prompt excludes secret input", "SECRET_INPUT_999" not in prompt)
    check("verifier prompt excludes secret answer", "SECRET_ANS_999" not in prompt)
    check("verifier prompt includes candidate code", "print(n)" in prompt)
    check("verifier prompt asks for CHOOSE/ABSTAIN", "CHOOSE" in prompt and "ABSTAIN" in prompt)


def test_auto_cases_format_preserving():
    prob = _prob([("3\n1 2 3\n", "ok\n")])
    cases = S.derive_auto_cases_v1(prob, seed_tag="t")
    check("auto cases generated", len(cases) >= 1)
    # a rotation of "1 2 3" preserves the multiset + token count
    has_rot = any(sorted(c.split("\n")[1].split()) == ["1", "2", "3"]
                  for c in cases if len(c.split("\n")) > 1 and c.split("\n")[1].split())
    check("auto case preserves integer-line multiset", has_rot)


def main():
    for fn in (test_stable_boundary, test_fake_selection_control, test_trust_machinery_kill,
               test_so_separable_falsifier_commits, test_so_under_determined_abstains,
               test_verifier_choice_parser, test_verifier_prompt_no_leakage,
               test_auto_cases_format_preserving):
        print(f"== {fn.__name__} ==")
        fn()
    print(f"\n{_PASS[0]} passed, {_FAIL[0]} failed")
    return 1 if _FAIL[0] else 0


if __name__ == "__main__":
    raise SystemExit(main())
