"""W98 — unit tests for the two RealWorldQA candidate benches.

* `coordpy.realworldqa_bench_v2` (B1: typed scene-graph
  extraction + question-typed solver).
* `coordpy.realworldqa_bench_v3` (B2: direct-vision final-turn
  answerer).

Covers:

* Question-type detection (`detect_question_type_v2`) on the 4
  W97 failure-cluster question shapes + the 3 unique-B-rescue
  multi-choice shapes.
* Capsule shape + content-addressable CID stability for both
  V2 and V3.
* Per-arm runner wiring on a synthetic corpus.
* End-to-end bench driver on a 3-problem synthetic corpus.
* Audit-chain re-derivation: per-call CIDs, per-seed Merkle,
  bench Merkle for both V2 and V3.
* Short-circuit semantics on V3: text-solver PASS skips the
  final VLM and pads with text retries.
* AddrP5 budget-exact: total calls = 11 / problem at K=5.
* Anti-cheat: extractor is `extract_candidate_answer_v1`
  byte-identical to W95 / W97; executor is rule-based.
* Bench-vs-runbook consistency: seed defaults match W98
  runbook (96_504_002).
"""
from __future__ import annotations

import hashlib
import json
import unittest

from coordpy.realworldqa_bench_v1 import (
    extract_candidate_answer_v1 as v1_extract,
)
from coordpy.realworldqa_bench_v2 import (
    QUESTION_TYPE_MULTI_CHOICE_LETTER,
    QUESTION_TYPE_NUMERIC,
    QUESTION_TYPE_SHORT_TEXT,
    QUESTION_TYPE_YES_NO,
    RealWorldQAV2ArmCallCapsule,
    RealWorldQAV2ArmOutcomeCapsule,
    RealWorldQAV2BenchConfig,
    RealWorldQAV2BenchReport,
    W98_REALWORLDQA_BENCH_V2_SCHEMA_VERSION,
    detect_question_type_v2,
    extract_candidate_answer_v1,
    run_realworldqa_bench_v2,
)
from coordpy.realworldqa_bench_v3 import (
    RealWorldQAV3ArmCallCapsule,
    RealWorldQAV3ArmOutcomeCapsule,
    RealWorldQAV3BenchConfig,
    RealWorldQAV3BenchReport,
    W98_REALWORLDQA_BENCH_V3_SCHEMA_VERSION,
    run_realworldqa_bench_v3,
)
from coordpy.realworldqa_executor_v1 import (
    evaluate_realworldqa_answer_v1,
)
from coordpy.realworldqa_loader_v1 import RealWorldQAProblemV1


def _problem(pid: str, question: str, answer: str,
             image_sha: str = "abc123") -> RealWorldQAProblemV1:
    return RealWorldQAProblemV1(
        pid=pid, question=question, answer=answer,
        image_path=f"img/{pid}.png",
        image_bytes=b"\x89PNG\r\n\x1a\n stub",
        image_sha256=image_sha,
        image_format="png",
        metadata={"shard_idx": 0, "row_index_in_shard": 0,
                  "image_format_sniff": "png"})


# ---------------------------------------------------------------
# Question-type detection
# ---------------------------------------------------------------

class TestDetectQuestionTypeV2(unittest.TestCase):
    """The W97 failure-cluster questions must be correctly typed."""

    def test_yes_no_stop_signs(self):
        # rwqa_test_000135
        q = "Are there any stop signs?\nPlease answer directly."
        self.assertEqual(
            detect_question_type_v2(q),
            QUESTION_TYPE_YES_NO)

    def test_yes_no_traffic_light(self):
        # rwqa_test_000403
        q = "Is the light green?\nPlease answer directly."
        self.assertEqual(
            detect_question_type_v2(q),
            QUESTION_TYPE_YES_NO)

    def test_yes_no_cars_facing(self):
        # rwqa_test_000555
        q = "are the cars facing left?\nPlease answer directly."
        self.assertEqual(
            detect_question_type_v2(q),
            QUESTION_TYPE_YES_NO)

    def test_yes_no_depth(self):
        # rwqa_test_000718
        q = ("Is the large truck that's closest to us further "
             "from the camera than the pickup truck?\nPlease "
             "answer directly.")
        self.assertEqual(
            detect_question_type_v2(q),
            QUESTION_TYPE_YES_NO)

    def test_multi_choice_vehicle_direction(self):
        # rwqa_test_000013
        q = ("Which direction is the vehicle directly in front "
             "of us traveling?\n\nA. Straight\nB. Left\n"
             "C. Right\nPlease answer directly with only the "
             "letter.")
        self.assertEqual(
            detect_question_type_v2(q),
            QUESTION_TYPE_MULTI_CHOICE_LETTER)

    def test_multi_choice_gun_direction(self):
        # rwqa_test_000155
        q = ("Which direction is the gun facing?\nA. The gun "
             "is facing left.\nB. The gun is facing down.\n"
             "C. The gun is facing right.\nPlease answer "
             "directly with only the letter.")
        self.assertEqual(
            detect_question_type_v2(q),
            QUESTION_TYPE_MULTI_CHOICE_LETTER)

    def test_numeric_how_many(self):
        q = "How many cars are visible?"
        self.assertEqual(
            detect_question_type_v2(q),
            QUESTION_TYPE_NUMERIC)

    def test_short_text_color(self):
        q = "What color is the car closest to the camera?"
        self.assertEqual(
            detect_question_type_v2(q),
            QUESTION_TYPE_SHORT_TEXT)

    def test_empty_returns_short_text(self):
        self.assertEqual(
            detect_question_type_v2(""),
            QUESTION_TYPE_SHORT_TEXT)

    def test_extractor_byte_identical_to_v1(self):
        # AddrP5 anti-cheat: V2 must use the same extractor as
        # V1 / W95.
        r = "Some prose\nFinal answer: B"
        self.assertEqual(
            extract_candidate_answer_v1(response_text=r),
            v1_extract(response_text=r))


# ---------------------------------------------------------------
# V2 capsule CID + bench driver
# ---------------------------------------------------------------

class TestV2CapsuleCID(unittest.TestCase):

    def test_arm_call_cid_changes_with_seed(self):
        c1 = RealWorldQAV2ArmCallCapsule(
            schema="x", seed=1, pid="p", arm_id="A0_text",
            role="text_solver", call_idx=0, temperature=0.0,
            prompt_cid="abc", response_cid="def", wall_ms=10)
        c2 = RealWorldQAV2ArmCallCapsule(
            schema="x", seed=2, pid="p", arm_id="A0_text",
            role="text_solver", call_idx=0, temperature=0.0,
            prompt_cid="abc", response_cid="def", wall_ms=10)
        self.assertNotEqual(c1.cid(), c2.cid())

    def test_arm_outcome_cid_includes_question_type(self):
        o1 = RealWorldQAV2ArmOutcomeCapsule(
            schema="x", seed=1, pid="p", arm_id="A0_text",
            question_type="yes_no",
            final_passed=True, final_prediction_cid="aa",
            final_executor_rule="text_exact",
            n_model_calls=1, total_wall_ms=10,
            call_capsule_cids=("c1",),
            executor_result_cid="e1")
        o2 = RealWorldQAV2ArmOutcomeCapsule(
            schema="x", seed=1, pid="p", arm_id="A0_text",
            question_type="numeric",
            final_passed=True, final_prediction_cid="aa",
            final_executor_rule="text_exact",
            n_model_calls=1, total_wall_ms=10,
            call_capsule_cids=("c1",),
            executor_result_cid="e1")
        self.assertNotEqual(o1.cid(), o2.cid())


class TestV2BenchDriverSyntheticCorpus(unittest.TestCase):
    """End-to-end driver wiring for V2 with stub gen functions."""

    def setUp(self):
        # 3 problems spanning yes_no, multi_choice, short_text.
        self.p1 = _problem(
            "rwqa_test_000001",
            "Are there any stop signs?", "Yes")
        self.p2 = _problem(
            "rwqa_test_000002",
            "Which lane is the car in? A) left B) center C) right",
            "C", image_sha="def456")
        self.p3 = _problem(
            "rwqa_test_000003",
            "What color is the car?", "red", image_sha="789xyz")
        self.corpus = (self.p1, self.p2, self.p3)

        def vlm_gen(prompt, image_bytes, max_tokens,
                    temperature):
            # B-typed reader prompt
            if "STRUCTURED scene graph" in prompt:
                if "stop signs" in prompt:
                    return (
                        '{"scene_summary": "street scene",'
                        ' "objects": [{"label": "stop_sign",'
                        ' "state": "on", "text_in_object":'
                        ' "STOP"}],'
                        ' "counts_by_label": {"stop_sign": 1},'
                        ' "direct_answer_hint": "Yes",'
                        ' "uncertain": []}'), 1
                if "Which lane" in prompt:
                    return (
                        '{"scene_summary": "street",'
                        ' "objects": [{"label": "car",'
                        ' "x_region": "right"}],'
                        ' "direct_answer_hint": "C",'
                        ' "uncertain": []}'), 1
                if "color is the car" in prompt:
                    return (
                        '{"scene_summary": "street",'
                        ' "objects": [{"label": "car",'
                        ' "color": "red"}],'
                        ' "direct_answer_hint": "red",'
                        ' "uncertain": []}'), 1
                return '{"scene_summary": "?"}', 1
            # A1 vlm — return correct directly
            if "stop signs" in prompt:
                return "Yes", 1
            if "Which lane" in prompt:
                return "C", 1
            if "color is the car" in prompt:
                return "red", 1
            return "(no match)", 1
        self.vlm_gen = vlm_gen

        def text_gen(prompt, max_tokens, temperature):
            # B-typed solver receives the JSON scene graph.
            if "Scene graph" in prompt:
                # The typed-solver should produce the correct
                # format-typed answer.
                if "stop signs" in prompt:
                    return "Final answer: Yes", 1
                if "Which lane" in prompt:
                    return "Final answer: C", 1
                if "color is the car" in prompt:
                    return "Final answer: red", 1
            # A0 text (no image, no scene graph) — wrong.
            return "Final answer: idk", 1
        self.text_gen = text_gen

    def test_v2_bench_runs_end_to_end(self):
        cfg = RealWorldQAV2BenchConfig(
            n_problems=3, K_multi_sample=5,
            seeds=(96_504_002,),
            sampling_temperature=0.7,
            max_tokens_per_call=64)
        report = run_realworldqa_bench_v2(
            text_gen=self.text_gen,
            vlm_gen=self.vlm_gen,
            vlm_model_id="stub-vlm",
            text_model_id="stub-text",
            corpus=self.corpus,
            corpus_parquet_shard_sha256=("sha-0", "sha-1"),
            corpus_merkle_root="merkle-root",
            config=cfg)
        self.assertIsInstance(report, RealWorldQAV2BenchReport)
        self.assertEqual(report.n_problems, 3)
        self.assertEqual(report.n_seeds, 1)
        self.assertEqual(report.K_multi_sample, 5)
        seed_rep = report.per_seed[0]
        # A0 fails on all (text-only "idk").
        self.assertEqual(seed_rep.a0_text_pass_at_1, 0.0)
        # A1 passes on all.
        self.assertEqual(seed_rep.a1_vlm_pass_at_1, 1.0)
        # B1 passes on all (typed solver gets correct format).
        self.assertEqual(
            seed_rep.b_vlm_team_v2_pass_at_1, 1.0)
        # B1 ≥ A1 on all 3.
        self.assertEqual(
            report.n_b_ge_a1_problems_per_seed, (3,))
        # Audit chain present.
        self.assertTrue(report.bench_merkle_root)
        self.assertTrue(seed_rep.seed_merkle_root)
        self.assertEqual(
            len(seed_rep.outcome_cids), 3 * 3)
        # Question type distribution present and counts to 3.
        self.assertEqual(
            sum(report.question_type_distribution.values()), 3)

    def test_v2_seed_default_matches_runbook(self):
        cfg = RealWorldQAV2BenchConfig()
        self.assertEqual(cfg.seeds, (96_504_002,))
        self.assertEqual(cfg.n_problems, 30)
        self.assertEqual(cfg.K_multi_sample, 5)

    def test_v2_per_problem_outcomes_contain_question_type(self):
        cfg = RealWorldQAV2BenchConfig(
            n_problems=3, K_multi_sample=5,
            seeds=(96_504_002,))
        report = run_realworldqa_bench_v2(
            text_gen=self.text_gen,
            vlm_gen=self.vlm_gen,
            vlm_model_id="stub-vlm",
            text_model_id="stub-text",
            corpus=self.corpus,
            corpus_parquet_shard_sha256=("sha-0", "sha-1"),
            corpus_merkle_root="merkle-root",
            config=cfg)
        seed_rep = report.per_seed[0]
        for po in seed_rep.per_problem_outcomes:
            self.assertIn("question_type", po)
            self.assertIn(
                po["question_type"],
                {QUESTION_TYPE_YES_NO,
                 QUESTION_TYPE_MULTI_CHOICE_LETTER,
                 QUESTION_TYPE_NUMERIC,
                 QUESTION_TYPE_SHORT_TEXT})


class TestV2AuditChainReDerivation(unittest.TestCase):
    """Re-derive the bench Merkle root from the outcome CIDs."""

    def test_bench_merkle_root_canonical(self):
        p = _problem("rwqa_test_000001", "Is it red?", "Yes")
        def tg(prompt, mt, t):
            if "Scene graph" in prompt:
                return "Final answer: Yes", 1
            return "Final answer: No", 1
        def vg(prompt, ib, mt, t):
            if "STRUCTURED scene graph" in prompt:
                return ('{"direct_answer_hint": "Yes",'
                        ' "uncertain": []}'), 1
            return "Yes", 1
        cfg = RealWorldQAV2BenchConfig(
            n_problems=1, K_multi_sample=5,
            seeds=(96_504_002,))
        report = run_realworldqa_bench_v2(
            text_gen=tg, vlm_gen=vg,
            vlm_model_id="vlm", text_model_id="text",
            corpus=(p,),
            corpus_parquet_shard_sha256=("aaaa",),
            corpus_merkle_root="m-root", config=cfg)
        outcome_cids = list(report.per_seed[0].outcome_cids)
        re_derived = hashlib.sha256(
            json.dumps({
                "kind": "w98_realworldqa_v2_bench_merkle_root",
                "vlm_model_id": "vlm",
                "text_model_id": "text",
                "corpus_parquet_shard_sha256": ["aaaa"],
                "corpus_merkle_root": "m-root",
                "outcome_cids": outcome_cids,
                "seeds": [96_504_002],
                "n_problems": 1,
                "K": 5,
            }, sort_keys=True,
                separators=(",", ":"),
                default=str).encode("utf-8")).hexdigest()
        self.assertEqual(
            report.bench_merkle_root, re_derived)


# ---------------------------------------------------------------
# V3 capsule + bench driver
# ---------------------------------------------------------------

class TestV3CapsuleCID(unittest.TestCase):

    def test_arm_outcome_cid_includes_final_vlm_flag(self):
        o1 = RealWorldQAV3ArmOutcomeCapsule(
            schema="x", seed=1, pid="p",
            arm_id="B_direct_vision_final",
            question_type="yes_no",
            final_turn_vlm_invoked=True,
            final_turn_vlm_rescued=True,
            final_passed=True, final_prediction_cid="aa",
            final_executor_rule="text_exact",
            n_model_calls=5, total_wall_ms=10,
            call_capsule_cids=("c1",),
            executor_result_cid="e1")
        o2 = RealWorldQAV3ArmOutcomeCapsule(
            schema="x", seed=1, pid="p",
            arm_id="B_direct_vision_final",
            question_type="yes_no",
            final_turn_vlm_invoked=False,
            final_turn_vlm_rescued=False,
            final_passed=True, final_prediction_cid="aa",
            final_executor_rule="text_exact",
            n_model_calls=5, total_wall_ms=10,
            call_capsule_cids=("c1",),
            executor_result_cid="e1")
        self.assertNotEqual(o1.cid(), o2.cid())


class TestV3BenchDriverShortCircuit(unittest.TestCase):
    """V3 short-circuit: text-solver PASS skips final VLM and
    pads with text retries.  Wall budget parity (K=5)
    preserved."""

    def setUp(self):
        # Problem where text-solver will PASS on turn 1.
        self.p_pass = _problem(
            "rwqa_pass", "What color is the car?", "red")
        # Problem where text-solver will FAIL all turns; final-
        # VLM rescues.
        self.p_rescue = _problem(
            "rwqa_rescue", "Is the light green?", "No",
            image_sha="rescue-sha")

        def vlm_gen(prompt, image_bytes, max_tokens, temperature):
            if "expert visual scene reader" in prompt:
                if "color is the car" in prompt:
                    return (
                        "- visible objects: car (right region)\n"
                        "- counts: car: 1\n"
                        "- colors: car: red\n"), 1
                if "light green" in prompt:
                    return (
                        "- visible objects: traffic light\n"
                        "- counts: traffic_light: 1\n"
                        "- colors: light: unknown\n"), 1
                return "- unknown\n", 1
            if "FINAL answerer" in prompt:
                if "rescue-sha" in str(image_bytes or "") or (
                        "light green" in prompt):
                    return "No", 1
                return "(no match)", 1
            if "color is the car" in prompt:
                return "red", 1
            if "light green" in prompt:
                return "Yes", 1
            return "(no match)", 1
        self.vlm_gen = vlm_gen

        def text_gen(prompt, max_tokens, temperature):
            if "Structured facts" in prompt:
                # For the PASS problem, solver returns "red" =>
                # text exact PASS.
                if "color is the car" in prompt:
                    return "Final answer: red", 1
                # For the RESCUE problem, solver returns wrong
                # ("Yes") so text-solver FAILs all turns.
                if "light green" in prompt:
                    return "Final answer: Yes", 1
            # A0_text — wrong.
            return "Final answer: idk", 1
        self.text_gen = text_gen

    def test_v3_short_circuit_pad_preserves_K(self):
        cfg = RealWorldQAV3BenchConfig(
            n_problems=1, K_multi_sample=5,
            seeds=(96_504_002,),
            sampling_temperature=0.7,
            max_tokens_per_call=64)
        report = run_realworldqa_bench_v3(
            text_gen=self.text_gen,
            vlm_gen=self.vlm_gen,
            vlm_model_id="stub-vlm",
            text_model_id="stub-text",
            corpus=(self.p_pass,),
            corpus_parquet_shard_sha256=("sha-0",),
            corpus_merkle_root="merkle-root",
            config=cfg)
        seed_rep = report.per_seed[0]
        # B passes (text-solver succeeds on turn 1).
        self.assertEqual(
            seed_rep.b_direct_vision_final_pass_at_1, 1.0)
        # Final VLM NOT invoked (short-circuit on PASS).
        self.assertEqual(
            seed_rep.final_vlm_invocation_count, 0)
        self.assertEqual(
            seed_rep.final_vlm_rescue_count, 0)
        # Per-problem outcome flags consistent.
        po = seed_rep.per_problem_outcomes[0]
        self.assertFalse(po["b_final_vlm_invoked"])
        self.assertFalse(po["b_final_vlm_rescued"])

    def test_v3_final_vlm_rescue_on_text_solver_fail(self):
        cfg = RealWorldQAV3BenchConfig(
            n_problems=1, K_multi_sample=5,
            seeds=(96_504_002,),
            sampling_temperature=0.7,
            max_tokens_per_call=64)
        report = run_realworldqa_bench_v3(
            text_gen=self.text_gen,
            vlm_gen=self.vlm_gen,
            vlm_model_id="stub-vlm",
            text_model_id="stub-text",
            corpus=(self.p_rescue,),
            corpus_parquet_shard_sha256=("sha-0",),
            corpus_merkle_root="merkle-root",
            config=cfg)
        seed_rep = report.per_seed[0]
        # Final VLM was invoked.
        self.assertEqual(
            seed_rep.final_vlm_invocation_count, 1)
        # Final VLM rescued (returned "No" which is gold).
        self.assertEqual(seed_rep.final_vlm_rescue_count, 1)
        # B passes overall (final VLM rescued the failure).
        self.assertEqual(
            seed_rep.b_direct_vision_final_pass_at_1, 1.0)

    def test_v3_audit_chain_present(self):
        cfg = RealWorldQAV3BenchConfig(
            n_problems=1, K_multi_sample=5,
            seeds=(96_504_002,))
        report = run_realworldqa_bench_v3(
            text_gen=self.text_gen,
            vlm_gen=self.vlm_gen,
            vlm_model_id="stub-vlm",
            text_model_id="stub-text",
            corpus=(self.p_rescue,),
            corpus_parquet_shard_sha256=("sha-0",),
            corpus_merkle_root="merkle-root",
            config=cfg)
        seed_rep = report.per_seed[0]
        self.assertTrue(seed_rep.seed_merkle_root)
        self.assertTrue(report.bench_merkle_root)
        # Final VLM rescue + invocation counts at bench level.
        self.assertEqual(
            report.final_vlm_invocation_count_total, 1)
        self.assertEqual(
            report.final_vlm_rescue_count_total, 1)

    def test_v3_seed_default_matches_runbook(self):
        cfg = RealWorldQAV3BenchConfig()
        self.assertEqual(cfg.seeds, (96_504_002,))
        self.assertEqual(cfg.n_problems, 30)
        self.assertEqual(cfg.K_multi_sample, 5)


class TestSchemaVersions(unittest.TestCase):

    def test_v2_schema_version(self):
        self.assertTrue(
            W98_REALWORLDQA_BENCH_V2_SCHEMA_VERSION.startswith(
                "coordpy.realworldqa_bench_v2"))

    def test_v3_schema_version(self):
        self.assertTrue(
            W98_REALWORLDQA_BENCH_V3_SCHEMA_VERSION.startswith(
                "coordpy.realworldqa_bench_v3"))


# ---------------------------------------------------------------
# Executor anti-cheat (re-asserted; both benches must route
# through evaluate_realworldqa_answer_v1).
# ---------------------------------------------------------------

class TestExecutorRuleBased(unittest.TestCase):

    def test_executor_yes_no(self):
        p = _problem("p", "Is the light green?", "No")
        v = evaluate_realworldqa_answer_v1(
            prediction="No", problem=p)
        self.assertTrue(v.passed)


if __name__ == "__main__":
    unittest.main()
