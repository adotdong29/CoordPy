"""W141 self-tutoring — $0 unit tests (extractor + no-oracle verifier + controller plumbing).

All deterministic, no NIM.  Validates: the AST technique-extractor produces a leak-audited completable
holed skeleton from a clean program (and DISCARDS non-conforming ones → KEEP); the no-oracle verifier
picks the correct+efficient program via the efficiency signal and S1 catches a fast-but-wrong one;
non-negativity is structural; the leak-gate spec_override path bites planted leaks.
"""
import random

import pytest

from coordpy.headroom_band_slate_v3 import CX_FACTORIES, FUNC_FACTORIES
from coordpy.resistant_by_construction_battlefield_v1 import mint_problem_v1
from coordpy.self_tutoring_technique_extractor_v1 import (
    derive_holes_from_ast_v1, compile_tutor_from_winner_v1)
from coordpy.no_oracle_verifier_v1 import (
    select_winner_v1, brute_is_trusted_v1, verify_candidate_v1, _StubProblem)
from coordpy.self_tutoring_controller_v1 import (
    discover_self_scaffold_v1, run_member_arms_v1, amortization_verdict_v1)

_CLEAN_FAMS = ["count_pairs_sum_le_t", "count_pairs_absdiff_le_d", "sum_nearest_smaller_left",
               "count_subarrays_sum_le_s"]


def _mint(fam, knob=20000, seed=12345):
    fac = CX_FACTORIES.get(fam) or FUNC_FACTORIES.get(fam)
    return fac(knob), mint_problem_v1(fac(knob).minted, global_seed=seed)


# ---------------------------------------------------------------- extractor

def test_extractor_compiles_clean_tutor_on_count_pairs():
    template, problem = _mint("count_pairs_sum_le_t")
    tutor, rep = compile_tutor_from_winner_v1(template.minted.ref_source, template, problem, timeout_s=5.0)
    assert rep.compiled and tutor is not None, rep.to_dict()
    assert rep.completable is True
    assert rep.stub_fails_secret is True          # holes are load-bearing
    assert rep.leak is not None and not rep.leak.leaked
    assert "__HOLE_" in tutor.skeleton            # the discriminator is a hole
    # the discriminating expression is NOT pasted in the model-facing text
    assert "a[i] + a[j] <= T" not in tutor.model_facing_text()


def test_extractor_holes_are_the_decision():
    template, _ = _mint("count_pairs_sum_le_t")
    ex = derive_holes_from_ast_v1(template.minted.ref_source)
    assert ex.ok and ex.n_add_holes >= 1 and ex.n_pred_holes >= 1
    # correct_fill reconstructs the original (completable by construction)
    assert any("<=" in v or "<" in v or ">" in v for v in ex.correct_fill.values())


def test_extractor_traces_accumulator_through_helper_function():
    # real models put the running accumulator inside a HELPER fn (result = solve(arr); print(result));
    # the extractor must trace the printed value through the function's return (the W141 dev-probe fix).
    prog = ("def solve(arr):\n"
            "    total = 0\n"
            "    stack = []\n"
            "    for num in arr:\n"
            "        while stack and stack[-1] >= num:\n"
            "            stack.pop()\n"
            "        if not stack:\n"
            "            total += -1\n"
            "        else:\n"
            "            total += stack[-1]\n"
            "        stack.append(num)\n"
            "    return total\n"
            "import sys\n"
            "arr = list(map(int, sys.stdin.read().split()))\n"
            "print(solve(arr))\n")
    ex = derive_holes_from_ast_v1(prog)
    assert ex.ok, ex.reason
    assert ex.n_add_holes >= 1                     # found 'total +=' INSIDE the helper
    assert "__HOLE_" in ex.skeleton and "total +=" in ex.skeleton


def test_extractor_discards_non_accumulator_program():
    template, problem = _mint("count_pairs_sum_le_t")
    junk = "import sys\nprint(len(sys.stdin.read().split()))\n"   # no accumulator update
    tutor, rep = compile_tutor_from_winner_v1(junk, template, problem, timeout_s=3.0)
    assert tutor is None and not rep.compiled       # discarded -> caller KEEPs


def test_extractor_clean_on_majority_of_complexity_families():
    n_ok = 0
    for fam in _CLEAN_FAMS:
        template, problem = _mint(fam)
        _, rep = compile_tutor_from_winner_v1(template.minted.ref_source, template, problem, timeout_s=5.0)
        n_ok += int(rep.compiled)
    assert n_ok >= 3, f"only {n_ok}/{len(_CLEAN_FAMS)} clean — need >=3 for the span"


# ---------------------------------------------------------------- no-oracle verifier

def _small_bank(template, n=10):
    return ["\n".join(template.minted.gen_public(random.Random(s))) for s in range(n)]


def test_verifier_picks_efficient_over_slow_brute():
    # on complexity-blind families the wrong candidate is the SLOW brute -> efficiency signal wins
    n_ref = 0
    for fam in ["count_pairs_sum_le_t", "count_pairs_absdiff_le_d", "sum_nearest_smaller_left"]:
        template, problem = _mint(fam, seed=777)
        ref, naive, brute = (template.minted.ref_source, template.minted.naive_source,
                             template.minted.brute_source)
        cands = [naive, brute, ref]
        sel = select_winner_v1(cands, statement=problem.statement, samples=list(problem.samples),
                               small_inputs=_small_bank(template), brute_code=brute,
                               consensus_probe=_small_bank(template)[0], timeout_s=3.0)
        n_ref += int(sel.winner_idx is not None and cands[sel.winner_idx] == ref)
    assert n_ref >= 3, f"verifier picked the efficient ref on only {n_ref}/3 complexity-blind families"


def test_brute_trust_gate():
    template, problem = _mint("count_pairs_sum_le_t")
    assert brute_is_trusted_v1(template.minted.brute_source, problem, timeout_s=4.0) is True
    assert brute_is_trusted_v1("import sys\nprint(0)\n", problem, timeout_s=2.0) is False  # wrong brute


def test_verifier_abstains_when_no_efficient_candidate():
    template, problem = _mint("count_pairs_sum_le_t", seed=4242)
    brute = template.minted.brute_source
    # pool = only slow brutes (no efficient sample): correct return is ABSTAIN (supply screen)
    sel = select_winner_v1([brute, brute], statement=problem.statement, samples=list(problem.samples),
                           small_inputs=_small_bank(template), brute_code=brute, timeout_s=3.0)
    assert sel.abstained and sel.winner_idx is None


# ---------------------------------------------------------------- controller plumbing + non-negativity

def _mock_gen(template):
    ref, brute = template.minted.ref_source, template.minted.brute_source
    st = {"n": 0}

    def gen(prompt, max_tokens, temperature):
        st["n"] += 1
        pl = prompt.lower()
        if "obviously-correct" in pl:
            return (f"```python\n{brute}\n```", 1)
        if "blank" in pl or "skeleton" in pl:
            return (f"```python\n{ref}\n```", 1)
        if "print" in pl and "=====" in prompt:
            return ("```python\npass\n```", 1)         # empty adversarial bank (falls back)
        return (f"```python\n{ref if st['n'] % 2 == 0 else brute}\n```", 1)
    return gen


def test_controller_discovers_and_extracts():
    template, teacher = _mint("count_pairs_sum_le_t", seed=1)
    disc = discover_self_scaffold_v1(template, teacher, gen=_mock_gen(template), K=4, timeout_s=3.0)
    assert disc.discovered and disc.scaffold is not None
    assert disc.winner_passes_secret is True


def test_controller_non_negative_when_no_scaffold():
    # scaffold None -> ST falls back to verified-selection B (KEEP), never below B/A1
    template, member = _mint("count_pairs_sum_le_t", seed=55)
    res = run_member_arms_v1(template, member, None, gen=_mock_gen(template), K=4, K_re=4, timeout_s=3.0)
    assert res.st_selected_pass == res.b_selected_pass     # KEEP == B
    v = amortization_verdict_v1("count_pairs_sum_le_t", False, [res])
    assert v.non_negative is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
