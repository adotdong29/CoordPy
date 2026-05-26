"""W102 — code_slice_selector_v1 unit tests."""
from __future__ import annotations

import json

import pytest

from coordpy import code_slice_selector_v1


def _synthetic_mining_report() -> dict:
    """Build a synthetic mining report shaped like W101's
    output."""
    return {
        "schema": "coordpy.w101_arsenal_mining_v1",
        "humaneval": {
            "bench_kind": "humaneval",
            "n_seeds": 3,
            "n_problems_per_seed": 30,
            "per_seed": {
                "1": {"a1_pass_rate": 0.8, "b_pass_rate": 0.93,
                      "b_minus_a1_pp": 13.0},
                "2": {"a1_pass_rate": 0.86, "b_pass_rate": 0.83,
                      "b_minus_a1_pp": -3.0},
                "3": {"a1_pass_rate": 0.9, "b_pass_rate": 0.97,
                      "b_minus_a1_pp": 7.0},
            },
            "aggregate": {
                "n_a1_only_wins": 3,
                "n_b_only_wins": 8,
                "n_shared_wins": 74,
                "n_shared_fails": 5,
                "a1_only_wins": [
                    "1:HumanEval/17",
                    "2:HumanEval/122",
                    "2:HumanEval/137",
                ],
                "b_only_wins": [
                    "1:HumanEval/118", "1:HumanEval/16",
                    "1:HumanEval/160", "1:HumanEval/76",
                    "1:HumanEval/91",
                    "2:HumanEval/121",
                    "3:HumanEval/83", "3:HumanEval/84",
                ],
                "shared_wins": [
                    f"1:HumanEval/{i}" for i in range(74)],
                "shared_fails": [
                    "1:HumanEval/84",
                    "2:HumanEval/132", "2:HumanEval/140",
                    "2:HumanEval/91",
                    "3:HumanEval/32",
                ],
            },
            "mechanism_load_bearing_estimate": {
                "fraction_b_wins_from_reflexion_rescue": 0.0976,
                "n_b_wins_total": 82,
                "n_b_only_rescues": 8,
            },
        },
        "mbpp": {
            "bench_kind": "mbpp",
            "n_seeds": 5,
            "n_problems_per_seed": 30,
            "per_seed": {
                "1": {"a1_pass_rate": 0.90, "b_pass_rate": 0.90,
                      "b_minus_a1_pp": 0.0},
                "2": {"a1_pass_rate": 0.70, "b_pass_rate": 0.77,
                      "b_minus_a1_pp": 6.67},
                "3": {"a1_pass_rate": 0.83, "b_pass_rate": 0.87,
                      "b_minus_a1_pp": 3.33},
                "4": {"a1_pass_rate": 0.90, "b_pass_rate": 0.90,
                      "b_minus_a1_pp": 0.0},
                "5": {"a1_pass_rate": 0.80, "b_pass_rate": 0.77,
                      "b_minus_a1_pp": -3.33},
            },
            "aggregate": {
                "n_a1_only_wins": 3,
                "n_b_only_wins": 5,
                "n_shared_wins": 121,
                "n_shared_fails": 21,
                "a1_only_wins": [
                    "3:448", "5:765", "5:780"],
                "b_only_wins": [
                    "2:83", "2:87", "3:439", "3:777", "5:101"],
                "shared_wins": [
                    f"1:{i+1}" for i in range(121)],
                "shared_fails": [
                    "1:229", "1:407", "1:802",
                    "2:255", "2:431", "2:442", "2:595",
                    "2:638", "2:776", "2:780",
                    "3:462", "3:579", "3:638",
                    "4:396", "4:461", "4:638",
                    "5:124", "5:228", "5:255", "5:430",
                    "5:640"],
            },
            "mechanism_load_bearing_estimate": {
                "fraction_b_wins_from_reflexion_rescue": 0.0397,
                "n_b_wins_total": 126,
                "n_b_only_rescues": 5,
            },
        },
    }


# ---------------------------------------------------------------
# Composite score + ranking
# ---------------------------------------------------------------


def test_composite_score_monotonic_in_rescue():
    s_low = code_slice_selector_v1.compute_composite_score(
        rescue_fraction=0.01,
        hard_cluster_size=5,
        mean_b_minus_a1_pp=1.0,
        per_seed_margin_std_pp=2.0,
        n_problems_per_seed=30)
    s_high = code_slice_selector_v1.compute_composite_score(
        rescue_fraction=0.20,
        hard_cluster_size=5,
        mean_b_minus_a1_pp=1.0,
        per_seed_margin_std_pp=2.0,
        n_problems_per_seed=30)
    assert s_high > s_low


def test_composite_score_monotonic_in_margin():
    s_low = code_slice_selector_v1.compute_composite_score(
        rescue_fraction=0.10,
        hard_cluster_size=5,
        mean_b_minus_a1_pp=-3.0,
        per_seed_margin_std_pp=2.0,
        n_problems_per_seed=30)
    s_high = code_slice_selector_v1.compute_composite_score(
        rescue_fraction=0.10,
        hard_cluster_size=5,
        mean_b_minus_a1_pp=10.0,
        per_seed_margin_std_pp=2.0,
        n_problems_per_seed=30)
    assert s_high > s_low


def test_rank_candidate_benches_humaneval_first():
    mining = _synthetic_mining_report()
    rankings = (
        code_slice_selector_v1.rank_candidate_benches(
            mining=mining, benches=("humaneval", "mbpp")))
    assert len(rankings) == 2
    assert rankings[0].bench == "humaneval"
    assert rankings[1].bench == "mbpp"
    assert rankings[0].composite_score > rankings[1].composite_score
    assert rankings[0].rescue_fraction == pytest.approx(
        0.0976, abs=0.001)


# ---------------------------------------------------------------
# Slice proposal
# ---------------------------------------------------------------


def test_propose_cheap_pilot_slice_priority_order():
    mining = _synthetic_mining_report()
    proposal = (
        code_slice_selector_v1.propose_cheap_pilot_slice(
            mining=mining, bench="humaneval",
            n_problems=20))
    assert proposal.n_problems == 20
    assert (
        proposal.cheap_pilot_budget_nim_calls
        == 20 * 11)
    # First 8 entries should be from b_only_wins (8 available)
    assert all(
        e.cluster == "b_only_wins"
        for e in proposal.proposal[:8])
    # Next 5 should be from shared_fails
    assert all(
        e.cluster == "shared_fails"
        for e in proposal.proposal[8:13])
    # Next 3 should be from a1_only_wins
    assert all(
        e.cluster == "a1_only_wins"
        for e in proposal.proposal[13:16])
    # Remaining top-up from shared_wins
    assert all(
        e.cluster == "shared_wins"
        for e in proposal.proposal[16:])


def test_propose_cheap_pilot_slice_deterministic():
    mining = _synthetic_mining_report()
    p1 = code_slice_selector_v1.propose_cheap_pilot_slice(
        mining=mining, bench="humaneval", n_problems=15)
    p2 = code_slice_selector_v1.propose_cheap_pilot_slice(
        mining=mining, bench="humaneval", n_problems=15)
    assert p1.proposal_cid == p2.proposal_cid


def test_propose_cheap_pilot_slice_refuses_anti_pattern():
    mining = _synthetic_mining_report()
    with pytest.raises(ValueError):
        code_slice_selector_v1.propose_cheap_pilot_slice(
            mining=mining, bench="humaneval",
            n_problems=5,
            bench_module_name=(
                "coordpy.bounded_window_baseline_v1"))


def test_propose_cheap_pilot_slice_missing_bench_raises():
    mining = _synthetic_mining_report()
    with pytest.raises(ValueError):
        code_slice_selector_v1.propose_cheap_pilot_slice(
            mining=mining, bench="nonexistent",
            n_problems=10)


def test_format_slice_proposal_markdown_includes_table():
    mining = _synthetic_mining_report()
    proposal = (
        code_slice_selector_v1.propose_cheap_pilot_slice(
            mining=mining, bench="mbpp", n_problems=10))
    md = (
        code_slice_selector_v1
        .format_slice_proposal_markdown(proposal))
    assert "| Seed |" in md
    assert "| task_id |" in md
    assert "proposal CID" in md
    assert proposal.proposal_cid in md
    # All 10 entries appear
    for e in proposal.proposal:
        assert e.task_id in md


def test_format_ranking_markdown_table():
    mining = _synthetic_mining_report()
    rankings = (
        code_slice_selector_v1.rank_candidate_benches(
            mining=mining))
    md = (
        code_slice_selector_v1
        .format_ranking_markdown(rankings))
    assert "Rank" in md
    assert "humaneval" in md
    assert "mbpp" in md


# ---------------------------------------------------------------
# Load mining report from disk
# ---------------------------------------------------------------


def test_load_mining_report_roundtrip(tmp_path):
    mining = _synthetic_mining_report()
    path = tmp_path / "mining.json"
    path.write_text(json.dumps(mining))
    loaded = code_slice_selector_v1.load_mining_report(path)
    assert loaded == mining


def test_load_mining_report_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        code_slice_selector_v1.load_mining_report(
            tmp_path / "nope.json")
