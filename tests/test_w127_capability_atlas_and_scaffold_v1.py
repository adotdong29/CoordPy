"""W127 — tests for resistant_capability_atlas_v1 + family_scaffold_generation_v1
(falsifiability-first).

Validated by DIRECT EXECUTION
(``python tests/test_w127_capability_atlas_and_scaffold_v1.py``) because the local
pytest/attrs env is broken (see W124/W125/W126 notes).  Each ``test_*`` raises on failure;
``main`` runs all and prints a PASS/FAIL tally.  No NIM.
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import coordpy.resistant_capability_atlas_v1 as A  # noqa: E402
import coordpy.family_scaffold_generation_v1 as G  # noqa: E402

EXPOSED_ROOT = "/tmp/w121_icpc"


# ============================================================ Lane α — atlas

def test_taxonomy_is_locked_ten_families():
    assert len(A.LOCKED_FAMILY_TAXONOMY) == 10
    assert "graph_flow" in A.LOCKED_FAMILY_TAXONOMY
    assert A.SCAFFOLDABLE_FAMILIES.issubset(set(A.LOCKED_FAMILY_TAXONOMY))


def test_family_classifier_is_deterministic():
    s = "Given a graph with N vertices and M edges, find the shortest path."
    a = A.classify_family_v1(statement=s, sample_text="", generation_codes=[])
    b = A.classify_family_v1(statement=s, sample_text="", generation_codes=[])
    assert a.family == b.family and a.scores == b.scores


def test_family_classifier_picks_graph_on_graph_statement():
    s = ("You are given an undirected graph with vertices and edges. Determine if the "
         "graph is connected and find the shortest path between two nodes.")
    fc = A.classify_family_v1(statement=s, sample_text="", generation_codes=[])
    assert fc.family == "graph_flow", f"expected graph_flow, got {fc.family}"


def test_family_classifier_picks_geometry():
    s = ("Given N points in the plane, compute the area of the convex polygon and the "
         "Euclidean distance between the two farthest points.")
    fc = A.classify_family_v1(statement=s, sample_text="", generation_codes=[])
    assert fc.family == "geometry", f"expected geometry, got {fc.family}"


def test_family_classifier_falls_back_to_adhoc_when_silent():
    fc = A.classify_family_v1(statement="zzz qqq", sample_text="", generation_codes=[])
    assert fc.family == "adhoc_math"


def test_failure_category_mapping():
    assert A._gen_failure_category({"parses": False}) == "parse_error"
    assert A._gen_failure_category({"parses": True, "digest_exc": "Timeout"}) == "timeout"
    assert A._gen_failure_category(
        {"parses": True, "digest_exc": "ValueError"}) == "runtime_error"
    assert A._gen_failure_category(
        {"parses": True, "digest_exc": "", "secret_pass": False}) == "wrong_answer"
    assert A._gen_failure_category(
        {"parses": True, "digest_exc": "", "secret_pass": True}) == "ok"


def test_reference_signal_detects_flow():
    fam, hits = A.classify_reference_family_v1(
        ["def bfs(): pass\n# ford_fulkerson max flow with adjacency"])
    assert fam == "graph_flow"


def test_reference_signal_unknown_on_empty():
    fam, hits = A.classify_reference_family_v1([""])
    assert fam == "unknown" and hits == []


def test_constraint_extraction():
    assert A._extract_max_constraint("N can be up to 10^9 in this problem") >= 10 ** 9
    assert A._extract_max_constraint("small") == 0


def _toy_entry(fam_stmt, *, ref_texts=(), cache=None):
    class P:
        problem_id = "icpc_na-ecna-archive_toy"
        short_name = "toy"
        source_repo = "ecna"
        contest_date = "2024-09-01"
        statement = fam_stmt
        samples = (("1\n", "1\n"),)
        secret_cases = (("1\n", "1\n"),)
    cache = cache or {"any_all_sample_pass": False, "distinct_codes": 3,
                      "distinct_digests": 2,
                      "gens": [{"parses": True, "digest_exc": "", "secret_pass": False,
                                "sample_pass": 0, "sample_total": 1}]}
    return A.build_atlas_entry_v1(problem=P(), generation_codes=["print(1)"],
                                  cache_problem=cache,
                                  teacher_family_count={"geometry": 5},
                                  reference_texts=ref_texts)


def test_atlas_entry_confidence_ref_conflict():
    # statement screams geometry; reference says graph_flow -> ref_conflict
    e = _toy_entry("points polygon area convex distance coordinate",
                   ref_texts=["bfs ford_fulkerson adjacency"])
    assert e.dominant_algorithm_family == "geometry"
    assert e.reference_family_signal == "graph_flow"
    assert e.family_confidence == "ref_conflict"
    assert e.atlas_label_agrees is False


def test_atlas_entry_confidence_confirmed_and_scaffoldable():
    e = _toy_entry("points polygon area convex distance coordinate intersection",
                   ref_texts=["math.hypot convex_hull cross("])
    assert e.dominant_algorithm_family == "geometry"
    assert e.family_confidence == "ref_confirmed"
    assert e.scaffoldable_flag is True  # geometry in SCAFFOLDABLE + coverage 5 >= 2


def test_atlas_entry_unconfirmed_when_no_reference():
    e = _toy_entry("points polygon area convex distance", ref_texts=[])
    assert e.family_confidence == "unconfirmed"


def test_atlas_clustering_and_concentration():
    entries = [_toy_entry("points polygon area convex coordinate distance") for _ in range(3)]
    atlas = A.build_capability_atlas_v1(entries)
    assert atlas.n_problems == 3
    assert atlas.dominant_cluster == "geometry"
    assert atlas.cluster_counts.get("geometry") == 3
    assert atlas.concentration_top2_frac == 1.0
    assert atlas.failure_mode_summary["by_category"]["wrong_answer"] == 3


def test_atlas_cid_deterministic():
    e1 = [_toy_entry("points polygon area")]
    e2 = [_toy_entry("points polygon area")]
    assert A.build_capability_atlas_v1(e1).atlas_cid == A.build_capability_atlas_v1(e2).atlas_cid


# ============================================================ Lane β — scaffold gen

def test_exposed_loader_returns_disjoint_gradeable():
    probs = G.load_exposed_problems_v1(EXPOSED_ROOT)
    assert len(probs) >= 20
    assert all(p.secret_cases and p.samples and p.accepted_codes for p in probs)
    assert len({p.short_name for p in probs}) == len(probs)  # unique


def test_skeleton_deidentifies_and_preserves_structure():
    code = ("def solve(myname):\n    total_sum = 0\n    for itemvalue in myname:\n"
            "        total_sum += itemvalue\n    return total_sum\n")
    skel, idioms, outline = G._extract_skeleton_v1(code)
    assert "myname" not in skel and "total_sum" not in skel  # locals renamed
    assert "for" in skel and "return" in skel                # structure preserved
    assert "func1" in skel or "v1" in skel


def test_skeleton_masks_long_string_literal():
    code = "x = '" + ("A" * 80) + "'\nprint(x)\n"
    skel, _i, _o = G._extract_skeleton_v1(code)
    assert "A" * 80 not in skel  # long literal truncated/masked


def test_skeleton_falls_back_on_unparseable():
    skel, idioms, outline = G._extract_skeleton_v1("def (: this is not python !!!")
    assert "skeleton unavailable" in outline or skel.startswith("#")


def test_idiom_detection():
    assert "bfs_queue" in G._detect_idioms("from collections import deque\nq=deque()")
    assert "modular_arith" in G._detect_idioms("print(pow(2,10,1000000007))")


def test_library_keyed_by_family_and_cid_deterministic():
    probs = G.load_exposed_problems_v1(EXPOSED_ROOT)[:12]
    l1 = G.build_scaffold_library_v1(probs)
    l2 = G.build_scaffold_library_v1(probs)
    assert l1.library_cid == l2.library_cid
    assert l1.n_scaffolds >= 1
    assert set(l1.by_family).issubset(set(A.LOCKED_FAMILY_TAXONOMY))


def test_retrieval_drops_same_problem():
    probs = G.load_exposed_problems_v1(EXPOSED_ROOT)
    lib = G.build_scaffold_library_v1(probs)            # ALL problems are teachers
    # pick a target that is in the library; retrieval must drop its own scaffold
    t = probs[0]
    fam = G.target_public_family_v1(t.statement, t.samples)
    rr = G.retrieve_scaffolds_v1(target_short=t.short_name, target_statement=t.statement,
                                 target_family=fam, library=lib, R=3)
    assert all(sc.source_problem.lower() != t.short_name.lower() for sc in rr.scaffolds)
    assert rr.leakage_clean


def test_retrieval_near_duplicate_guard():
    # a scaffold whose skeleton == the target statement must be dropped as near-duplicate
    sc = G.AlgorithmScaffoldV1(family="geometry", source_problem="other", source_sha="x",
                               idioms=(), outline="o",
                               skeleton="alpha beta gamma delta epsilon zeta eta theta")
    lib = G.ScaffoldLibraryV1(schema="s", by_family={"geometry": (sc,)}, n_scaffolds=1,
                              n_source_problems=1, library_cid="c")
    rr = G.retrieve_scaffolds_v1(
        target_short="tgt",
        target_statement="alpha beta gamma delta epsilon zeta eta theta",
        target_family="geometry", library=lib, R=2, max_overlap=0.2)
    assert len(rr.scaffolds) == 0 and rr.dropped_near_duplicate == 1


def test_prioritized_families_includes_group():
    cls = A.classify_family_v1(statement="maze grid robot moves cells board direction",
                               sample_text="", generation_codes=[])
    prio = G.prioritized_families_v1(cls)
    # simulation_grid group includes graph_flow
    assert "graph_flow" in prio or "simulation_grid" in prio


def test_scaffolded_prompt_marks_template_and_baseline_has_none():
    probs = G.load_exposed_problems_v1(EXPOSED_ROOT)
    t = probs[0].as_pilot_problem()
    sc = G.AlgorithmScaffoldV1(family="geometry", source_problem="other", source_sha="x",
                               idioms=("geometry_prim",), outline="o", skeleton="print(1)")
    p_scaf = G.build_scaffolded_prompt_v1(t, [sc])
    p_base = G.build_plain_prompt_v1(t)
    assert "TEMPLATE" in p_scaf and "NOT a solution" in p_scaf
    assert "TEMPLATE" not in p_base
    assert "geometry" in p_scaf


def test_g4_policy_deterministic_and_learned_not_warranted():
    sc = G.AlgorithmScaffoldV1(family="geometry", source_problem="o", source_sha="x",
                               idioms=(), outline="o", skeleton="print(1)")
    lib = G.ScaffoldLibraryV1(schema="s", by_family={"geometry": (sc,)}, n_scaffolds=1,
                              n_source_problems=1, library_cid="c")
    d = G.scaffold_action_policy_v1(target_family="geometry", library=lib,
                                    n_labelled_events=5)
    assert d.action == "scaffold:geometry" and d.learned_warranted is False
    d2 = G.scaffold_action_policy_v1(target_family="adhoc_math", library=lib)
    assert d2.action == "plain"   # adhoc_math not in SCAFFOLDABLE_FAMILIES


def _res(short, fam, base, scaf):
    return G.DevBenchTargetResultV1(
        short_name=short, family=fam, families_pulled=(fam,), n_scaffolds=1,
        baseline_pass=base, scaffold_pass=scaf, baseline_first_pass_k=0 if base else -1,
        scaffold_first_pass_k=0 if scaf else -1, failure_family_was_trivial=False,
        leakage_clean=True)


def test_earn_gate_earned_two_families():
    res = [_res("a", "geometry", False, True), _res("b", "dp_optimization", False, True),
           _res("c", "string_processing", True, True)]
    v = G.apply_dev_bench_earn_gate_v1(res)
    assert v.earned and v.verdict_label == "EXPOSED_SCAFFOLD_DEV_BENCH_EARNED"
    assert v.net_scaffold_gain == 2 and v.gain_distinct_families == 2


def test_earn_gate_thin_single_family():
    res = [_res("a", "geometry", False, True), _res("b", "geometry", False, True),
           _res("c", "string_processing", True, False)]
    v = G.apply_dev_bench_earn_gate_v1(res)
    # net = 2 - 1... scaffold pass {a,b}=2, baseline pass {c}=1 => net +1 => THIN
    assert not v.earned and v.verdict_label == "EXPOSED_SCAFFOLD_DEV_BENCH_THIN"


def test_earn_gate_dead_on_no_gain():
    res = [_res("a", "geometry", True, False), _res("b", "dp_optimization", True, True)]
    v = G.apply_dev_bench_earn_gate_v1(res)
    assert v.net_scaffold_gain <= 0
    assert v.verdict_label == "EXPOSED_SCAFFOLD_DEV_BENCH_DEAD"


def test_earn_gate_requires_two_families_not_one():
    res = [_res("a", "geometry", False, True), _res("b", "geometry", False, True),
           _res("c", "geometry", False, True)]
    v = G.apply_dev_bench_earn_gate_v1(res)
    assert v.net_scaffold_gain == 3 and v.gain_distinct_families == 1
    assert not v.earned  # R1b fails


def test_earn_gate_invalid_on_leakage():
    bad = G.DevBenchTargetResultV1(
        short_name="a", family="geometry", families_pulled=("geometry",), n_scaffolds=1,
        baseline_pass=False, scaffold_pass=True, baseline_first_pass_k=-1,
        scaffold_first_pass_k=0, failure_family_was_trivial=False, leakage_clean=False)
    v = G.apply_dev_bench_earn_gate_v1([bad])
    assert v.verdict_label == "DEV_BENCH_INVALID_LEAKAGE" and not v.earned


def test_pipeline_leakage_positive_control():
    from coordpy.family_adapted_repair_synthesis_v1 import SynthesisLeakageGuardV1
    probs = G.load_exposed_problems_v1(EXPOSED_ROOT)
    ep = probs[0]
    prob = ep.as_pilot_problem()
    guard = SynthesisLeakageGuardV1(prob, target_accepted_texts=list(ep.accepted_codes))
    secret_ans = prob.secret_cases[0][1].strip()
    clean, reason = G.assert_scaffold_pipeline_clean_v1(
        target_short=ep.short_name, scaffolds=(),
        candidate_texts=[f"print({secret_ans!r})"], guard=guard)
    assert not clean, "leakage positive control (secret) did not bite"


def test_accepted_block_leak_caught_but_boilerplate_clean():
    """The corrected accepted-line tripwire: a reproduced CONTIGUOUS block of the accepted
    solution is caught; a single shared common idiom is NOT (boilerplate-robust)."""
    accepted = ("n, k = map(int, input().split())\n"
                "total = 0\n"
                "for a in range(1, n + 1):\n"
                "    total += special_transform(a, k)\n"
                "print(total)\n")
    # planting the whole accepted solution => contiguous block => CAUGHT
    assert G.reproduces_accepted_block_v1(accepted, [accepted])
    # a genuinely different solution sharing only the boilerplate input line => CLEAN
    different = ("n, k = map(int, input().split())\n"
                 "acc = 0\n"
                 "idx = 1\n"
                 "while idx <= n:\n"
                 "    acc = (acc * 10 + idx) % k\n"
                 "    idx += 1\n"
                 "print(acc)\n")
    assert not G.reproduces_accepted_block_v1(different, [accepted]), \
        "boilerplate-only overlap wrongly flagged as accepted-block leak"


def test_accepted_block_provenance_excludes():
    """A block already in provenance (legitimately shown, e.g. a teacher skeleton) is not
    counted as a target-accepted leak."""
    accepted = "a = 1\nb = 2\nc = a + b\nprint(c)\n"
    assert not G.reproduces_accepted_block_v1(accepted, [accepted], provenance=accepted)


def main() -> int:
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    npass = 0
    for t in tests:
        try:
            t()
            print(f"  PASS {t.__name__}")
            npass += 1
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL {t.__name__}: {e}")
    print(f"\n  {npass}/{len(tests)} passed")
    return 0 if npass == len(tests) else 1


if __name__ == "__main__":
    raise SystemExit(main())
