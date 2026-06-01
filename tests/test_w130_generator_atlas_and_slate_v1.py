"""W130 tests — generator-failure atlas + stronger generator slate (NIM-free, falsifiability-first).

Covers the LOCKED atlas taxonomy + classifier, the complexity gate, the GG realness controls,
the honest hosted-controller examination, the leakage guard, and the EXPOSED dev-bench earn gate.
All $0 (no model call).  Run directly (``python tests/test_w130_...py``) or via pytest.
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import coordpy.generator_failure_atlas_v1 as A  # noqa: E402
import coordpy.stronger_generator_slate_v1 as S  # noqa: E402
from coordpy.role_diverse_algorithm_search_v1 import SketchV1  # noqa: E402


# ---------------------------------------------------------------------------
# Lane α — generator-failure atlas
# ---------------------------------------------------------------------------
def test_taxonomy_locked():
    assert A.GENERATOR_FAILURE_MODES[0] == "SOLVED"
    # every generator-fixable mode is a real taxonomy member
    assert A.GENERATOR_FIXABLE_MODES <= set(A.GENERATOR_FAILURE_MODES)
    assert "WRONG_ALGORITHM_NO_SKETCH" not in A.GENERATOR_FIXABLE_MODES  # capability failure
    assert "SELECTION_FIXABLE" not in A.GENERATOR_FIXABLE_MODES  # W129 domain, not generation


def test_classifier_pool_bearing_branches():
    # both correct + wrong public survivors -> SELECTION_FIXABLE (a tie, W129 domain)
    m, _ = A.classify_problem_failure_v1(pool_bearing=True, n_pub_correct=2, n_pub_wrong=1,
                                         counts={}, sketch_admissible=False)
    assert m == "SELECTION_FIXABLE"
    # pool-bearing, only-correct survivors -> SOLVED
    m, _ = A.classify_problem_failure_v1(pool_bearing=True, n_pub_correct=2, n_pub_wrong=0,
                                         counts={}, sketch_admissible=False)
    assert m == "SOLVED"


def test_classifier_pool_dead_branches():
    # HIDDEN_EDGE: a candidate passes all public but fails secret
    m, _ = A.classify_problem_failure_v1(pool_bearing=False, n_pub_correct=0, n_pub_wrong=0,
                                         counts={"HIDDEN_FAIL": 1, "WRONG_ANSWER": 5},
                                         sketch_admissible=False)
    assert m == "HIDDEN_EDGE_STATE_MISS"
    # COMPLEXITY_BLIND: TLE dominates
    m, _ = A.classify_problem_failure_v1(pool_bearing=False, n_pub_correct=0, n_pub_wrong=0,
                                         counts={"TLE": 4, "WRONG_ANSWER": 1},
                                         sketch_admissible=False)
    assert m == "COMPLEXITY_BLIND"
    # WRONG_ALGORITHM_ADMISSIBLE vs NO_SKETCH split on admissibility
    m_adm, _ = A.classify_problem_failure_v1(pool_bearing=False, n_pub_correct=0, n_pub_wrong=0,
                                             counts={"WRONG_ANSWER": 10}, sketch_admissible=True)
    m_no, _ = A.classify_problem_failure_v1(pool_bearing=False, n_pub_correct=0, n_pub_wrong=0,
                                            counts={"WRONG_ANSWER": 10}, sketch_admissible=False)
    assert m_adm == "WRONG_ALGORITHM_ADMISSIBLE"
    assert m_no == "WRONG_ALGORITHM_NO_SKETCH"
    # PARSE_IO: parse/crash dominates wrong-answer
    m, _ = A.classify_problem_failure_v1(pool_bearing=False, n_pub_correct=0, n_pub_wrong=0,
                                         counts={"CRASH": 6, "WRONG_ANSWER": 2}, sketch_admissible=True)
    assert m == "PARSE_IO_FAILURE"


def test_algorithm_signature_detects_idioms():
    sig = A.algorithm_signature_v1(["from collections import deque\n# bfs over the grid"])
    assert "graph_search" in sig["specific"]
    sig2 = A.algorithm_signature_v1(["just sort the list and print"])
    assert "interval_sort" in sig2["weak"] and not sig2["specific"]


def test_sketch_admissible_requires_specific_match():
    acc = ["import heapq\n# greedy with a priority queue"]
    # a sketch naming greedy/heap matches on a SPECIFIC idiom -> admissible
    good = A.sketch_admissible_v1([SketchV1("A", "greedy heap", "use a heap, pop the best")], acc)
    assert good["admissible"] and "greedy_heap" in good["matched_specific"]
    # a sketch matching only a WEAK idiom (sort) is NOT admissible
    weak = A.sketch_admissible_v1([SketchV1("B", "sort", "sort then print")], acc)
    assert not weak["admissible"]


def test_atlas_aggregation_separates_generator_from_capability():
    def rec(name, mode, pool):
        return A.ProblemFailureRecordV1(
            short_name=name, family="adhoc_math", surface="s", date="d", n_candidates=5,
            n_parse=5, n_public_survivors=0, n_pub_correct=0, n_pub_wrong=0, counts={},
            pool_bearing=pool, accepted_specific=[], sketch_specific=[], matched_specific=[],
            sketch_admissible=False, dominant_mode=mode,
            selector_fixable=(mode == "SELECTION_FIXABLE"),
            generator_fixable=(mode in A.GENERATOR_FIXABLE_MODES), committed_w128=None, note="")
    atlas = A.build_atlas_v1([
        rec("a", "WRONG_ALGORITHM_ADMISSIBLE", False),
        rec("b", "HIDDEN_EDGE_STATE_MISS", False),
        rec("c", "WRONG_ALGORITHM_NO_SKETCH", False),
        rec("d", "SOLVED", True)])
    assert set(atlas.generator_fixable) == {"a", "b"}
    assert atlas.capability_failures == ["c"]
    assert atlas.pool_dead == ["a", "b", "c"] and atlas.pool_bearing == ["d"]


# ---------------------------------------------------------------------------
# Lane β — stronger generator slate
# ---------------------------------------------------------------------------
def test_complexity_parsing():
    assert S.parse_complexity_exponent_v1("O(N^2) double loop") == 2.0
    assert S.parse_complexity_exponent_v1("O(N log N) sort") == 1.1
    assert S.parse_complexity_exponent_v1("O(N)") == 1.0
    assert S.parse_complexity_exponent_v1("O(2^N)") == 99.0
    assert S.parse_complexity_exponent_v1("a clever greedy") is None  # unstated -> None


def test_complexity_gate_falsifiable():
    n = 1_000_000
    assert S.complexity_admissible_v1(2.0, n) is False     # O(N^2) at 1e6 -> inadmissible
    assert S.complexity_admissible_v1(1.1, n) is True      # O(N log N) -> admissible
    assert S.complexity_admissible_v1(None, n) is None     # unstated -> unjudgeable (never rejected)
    assert S.complexity_admissible_v1(2.0, None) is None   # no bound -> unjudgeable
    assert S.complexity_admissible_v1(99.0, 10) is True    # exponential ok for tiny n
    assert S.complexity_admissible_v1(99.0, 1000) is False  # exponential rejected for big n


def test_gg1_gate_control_passes():
    c = S.gg1_gate_control_v1()
    assert c["passes"] and c["slow_admissible"] is False and c["fast_admissible"] is True


def test_gg2_rewrite_control_passes():
    c = S.gg2_rewrite_control_v1()
    # falsifiable: a public-failing candidate yields a failing case; a passing one does not
    assert c["passes"] and c["bad_has_failure"] and not c["good_has_failure"]


def test_hosted_controller_literal_bridge_killed():
    rec = S.examine_hosted_controller_applicability_v1()
    assert rec["literal_planner_bridge_killed"] is True
    assert "efficiency_only" in rec["hosted_cache_aware_planner_v12"]
    assert "substrate_trust_specific" in rec["multi_agent_substrate_coordinator_v15"]
    assert "w125" in rec["applicable_lever"]


def test_family_coach_is_not_a_scaffold():
    card = S.build_family_coach_card_v1("adhoc_math")
    assert "FAMILY COACHING" in card and "not specific to this problem" in card
    # generic family advice carries no candidate's accepted bytes (no long contiguous code block)
    assert "def " not in card and "import " not in card


def _mk_outcome(name, arm, pool_pass, committed_pass=False, real=True, clean=True):
    return S.GgArmOutcomeV1(
        short_name=name, arm=arm, n_calls=5, candidates=(), artifacts_spec_len=10, n_sketches=4,
        pool_pass=pool_pass, pool_secret_labels=(("x",) if pool_pass else ()),
        committed_label=("x" if committed_pass else None), committed_pass=committed_pass,
        selector_branch="b", diagnostics={}, realness={"ok": real}, leakage_clean=clean)


def test_earn_gate_requires_two_new_solves_spanning_two():
    fam = {"p1": "adhoc_math", "p2": "simulation_grid", "p3": "adhoc_math"}
    mode = {"p1": "WRONG_ALGORITHM_ADMISSIBLE", "p2": "HIDDEN_EDGE_STATE_MISS",
            "p3": "WRONG_ALGORITHM_ADMISSIBLE"}
    old = ("pawnshop", "sunandmoon", "blueberrywaffle")
    # 1 new solve -> NOT earned
    one = {"p1": {"GG1": _mk_outcome("p1", "GG1", True)},
           "p2": {"GG1": _mk_outcome("p2", "GG1", False)}}
    v1 = S.apply_gg_dev_bench_earn_gate_v1(one, old_pool_solved=old, family_of=fam, atlas_mode_of=mode)
    assert not v1.earned and v1.best_new_count == 1
    # 2 new solves, SAME family AND same mode -> spans_two False -> NOT earned
    same = {"p1": {"GG1": _mk_outcome("p1", "GG1", True)},
            "p3": {"GG1": _mk_outcome("p3", "GG1", True)}}
    v2 = S.apply_gg_dev_bench_earn_gate_v1(same, old_pool_solved=old, family_of=fam, atlas_mode_of=mode)
    assert not v2.earned and v2.best_new_count == 2 and not v2.spans_two
    # 2 new solves spanning 2 families AND modes, clean+real -> EARNED
    span = {"p1": {"GG1": _mk_outcome("p1", "GG1", True)},
            "p2": {"GG1": _mk_outcome("p2", "GG1", True)}}
    v3 = S.apply_gg_dev_bench_earn_gate_v1(span, old_pool_solved=old, family_of=fam, atlas_mode_of=mode)
    assert v3.earned and v3.best_new_count == 2 and v3.spans_two


def test_earn_gate_ignores_old_pool_solves():
    # solving an OLD pool problem does NOT count as a new solve
    fam = {"pawnshop": "adhoc_math"}
    mode = {"pawnshop": "SELECTION_FIXABLE"}
    d = {"pawnshop": {"GG1": _mk_outcome("pawnshop", "GG1", True)}}
    v = S.apply_gg_dev_bench_earn_gate_v1(d, old_pool_solved=("pawnshop",), family_of=fam,
                                          atlas_mode_of=mode)
    assert v.best_new_count == 0 and not v.earned


def test_earn_gate_leakage_blocks_earn():
    fam = {"p1": "adhoc_math", "p2": "simulation_grid"}
    mode = {"p1": "WRONG_ALGORITHM_ADMISSIBLE", "p2": "HIDDEN_EDGE_STATE_MISS"}
    dirty = {"p1": {"GG1": _mk_outcome("p1", "GG1", True, clean=False)},
             "p2": {"GG1": _mk_outcome("p2", "GG1", True, clean=True)}}
    v = S.apply_gg_dev_bench_earn_gate_v1(dirty, old_pool_solved=(), family_of=fam,
                                          atlas_mode_of=mode)
    assert not v.winners_leakage_clean and not v.earned


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
        passed += 1
    print(f"\n{passed}/{len(fns)} W130 tests passed")
