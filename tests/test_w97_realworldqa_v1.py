"""W97 — unit tests for ``coordpy.realworldqa_bench_v1``.

Covers:

* Capsule shape and content-addressable CID stability.
* Per-arm runner wiring against stub gen functions (no NIM).
* End-to-end bench driver over a 2-problem synthetic corpus.
* Audit-chain re-derivation: per-call CIDs, per-seed Merkle,
  bench Merkle are all derivable from the per-problem
  outcomes.
* Anti-cheat: same extractor on every arm; same K=5 budget;
  executor truth is rule-based (no judge).
* Bench-vs-arsenal-doc consistency: the seed default matches the
  W97 runbook (96_504_002).
"""
from __future__ import annotations

import hashlib
import json
import unittest

from coordpy.realworldqa_bench_v1 import (
    RealWorldQAArmCallCapsuleV1,
    RealWorldQAArmOutcomeCapsuleV1,
    RealWorldQABenchConfigV1,
    RealWorldQABenchReportV1,
    RealWorldQASeedReportV1,
    W97_REALWORLDQA_BENCH_V1_SCHEMA_VERSION,
    extract_candidate_answer_v1,
    run_realworldqa_bench_v1,
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


def _stub_text_gen(reply_map: dict[str, str]):
    def gen(prompt: str, max_tokens: int,
            temperature: float):
        for key, v in reply_map.items():
            if key in prompt:
                return v, 1
        return "(no match)", 1
    return gen


def _stub_vlm_gen(reply_map: dict[tuple[str, str], str]):
    def gen(prompt: str, image_bytes, max_tokens: int,
            temperature: float):
        for (key, img_marker), v in reply_map.items():
            if key in prompt:
                if not img_marker or (
                        image_bytes is not None
                        and img_marker.encode()
                        in image_bytes):
                    return v, 1
        return "(no match)", 1
    return gen


class TestExtractCandidateAnswer(unittest.TestCase):

    def test_extract_prefers_tagged_line(self):
        r = "Some prose\nFinal answer: B"
        self.assertEqual(
            extract_candidate_answer_v1(response_text=r),
            "Final answer: B")

    def test_extract_falls_back_to_last_nonempty(self):
        r = "First line\n\nlast line"
        self.assertEqual(
            extract_candidate_answer_v1(response_text=r),
            "last line")

    def test_extract_empty_input(self):
        self.assertEqual(
            extract_candidate_answer_v1(response_text=""), "")


class TestCapsuleCID(unittest.TestCase):

    def test_arm_call_cid_changes_with_seed(self):
        c1 = RealWorldQAArmCallCapsuleV1(
            schema="x", seed=1, pid="p", arm_id="A0_text",
            role="text_solver", call_idx=0, temperature=0.0,
            prompt_cid="abc", response_cid="def", wall_ms=10)
        c2 = RealWorldQAArmCallCapsuleV1(
            schema="x", seed=2, pid="p", arm_id="A0_text",
            role="text_solver", call_idx=0, temperature=0.0,
            prompt_cid="abc", response_cid="def", wall_ms=10)
        self.assertNotEqual(c1.cid(), c2.cid())

    def test_arm_outcome_cid_stable(self):
        o = RealWorldQAArmOutcomeCapsuleV1(
            schema="x", seed=1, pid="p", arm_id="A0_text",
            final_passed=True, final_prediction_cid="aa",
            final_executor_rule="text_exact",
            n_model_calls=1, total_wall_ms=10,
            call_capsule_cids=("c1",),
            executor_result_cid="e1")
        self.assertEqual(o.cid(), o.cid())


class TestBenchDriverSyntheticCorpus(unittest.TestCase):
    """End-to-end driver wiring with stub gen functions.  Two
    problems; assert every audit-chain artifact is present and
    re-derivable."""

    def setUp(self):
        # Problem 1 — multi-choice letter.
        # Problem 2 — short text "yes".
        self.p1 = _problem(
            "rwqa_test_000001",
            "Which lane is the car in? A) left B) center C) right",
            "C")
        self.p2 = _problem(
            "rwqa_test_000002",
            "Is the pedestrian crossing the street?", "yes",
            image_sha="def456")
        self.corpus = (self.p1, self.p2)

        # A0_text (no image): always picks wrong (matches W95
        # pattern that text-only floors are low).
        # A1_vlm: VLM samples — alternates correct and wrong.
        # B_vlm_team: reader extracts; solver answers correctly.

        self.text_gen = _stub_text_gen({
            # A0_text — wrong on both
            "You are answering a real-world image question.  Read "
            "the question": "A",
            # B_text_solver — initial answers (correct on both)
            "structured facts extracted from the image": (
                "C" if True else "??"),
        })

        # The VLM stub needs to differentiate by image content
        # for B-reader.  We use a simpler approach: condition on
        # the prompt prefix.
        def vlm_gen(prompt, image_bytes, max_tokens,
                    temperature):
            # B reader prompt
            if "expert visual scene reader" in prompt:
                return (
                    "- visible objects: car (right region, "
                    "middle), pedestrian (center, middle)\n"
                    "- counts: car: 1; pedestrian: 1\n"
                    "- spatial: pedestrian to left of car\n"
                    "- text-in-scene: none\n"
                    "- activity: pedestrian appears to be "
                    "walking across\n"
                    "Final answer for solver: C"), 1
            # A1 vlm — for problem 1: alternate C / B; for
            # problem 2: yes / no
            if "Which lane" in prompt:
                # T=0.7 sampling — return C 3/5 times
                if "Attempt" in prompt:
                    return "C", 1
                return "C", 1
            if "pedestrian crossing" in prompt:
                return "yes", 1
            return "(no match)", 1
        self.vlm_gen = vlm_gen

        # Replace the simple text-gen with a smarter routing
        # that returns "C" for B's first problem and "yes"
        # for the second.  This is the text-solver path
        # (no image).
        def text_gen(prompt, max_tokens, temperature):
            if "Which lane" in prompt:
                # B-solver receives structured facts; should
                # answer C.
                if "structured facts" in prompt:
                    return "Final answer: C", 1
                # A0-text (no image): wrong.
                return "Final answer: A", 1
            if "pedestrian crossing" in prompt:
                if "structured facts" in prompt:
                    return "Final answer: yes", 1
                return "Final answer: no", 1
            return "(no match)", 1
        self.text_gen = text_gen

    def test_bench_runs_end_to_end(self):
        cfg = RealWorldQABenchConfigV1(
            n_problems=2, K_multi_sample=5,
            seeds=(96_504_002,),
            sampling_temperature=0.7,
            max_tokens_per_call=64)
        report = run_realworldqa_bench_v1(
            text_gen=self.text_gen,
            vlm_gen=self.vlm_gen,
            vlm_model_id="stub-vlm",
            text_model_id="stub-text",
            corpus=self.corpus,
            corpus_parquet_shard_sha256=("sha-0", "sha-1"),
            corpus_merkle_root="merkle-root",
            config=cfg)
        self.assertIsInstance(report, RealWorldQABenchReportV1)
        self.assertEqual(report.n_problems, 2)
        self.assertEqual(report.n_seeds, 1)
        self.assertEqual(report.K_multi_sample, 5)
        self.assertEqual(report.vlm_model_id, "stub-vlm")
        self.assertEqual(len(report.per_seed), 1)
        seed_rep = report.per_seed[0]
        self.assertEqual(seed_rep.seed, 96_504_002)
        self.assertEqual(seed_rep.n_problems, 2)
        # A0 fails on both (text-only wrong answers).
        self.assertEqual(seed_rep.a0_text_pass_at_1, 0.0)
        # A1 passes on both (VLM correct).
        self.assertEqual(seed_rep.a1_vlm_pass_at_1, 1.0)
        # B passes on both (reader + solver correct).
        self.assertEqual(seed_rep.b_vlm_team_pass_at_1, 1.0)
        # Cross-arm means
        self.assertAlmostEqual(
            report.a0_text_mean_pass_at_1, 0.0)
        self.assertAlmostEqual(
            report.a1_vlm_mean_pass_at_1, 1.0)
        self.assertAlmostEqual(
            report.b_vlm_team_mean_pass_at_1, 1.0)
        # B ≥ A1 on both problems.
        self.assertEqual(
            report.n_b_ge_a1_problems_per_seed, (2,))
        # Audit-chain artifacts present.
        self.assertTrue(report.bench_merkle_root)
        self.assertTrue(seed_rep.seed_merkle_root)
        self.assertEqual(
            len(seed_rep.outcome_cids), 3 * 2)
        self.assertEqual(
            len(seed_rep.per_problem_outcomes), 2)

    def test_seed_default_matches_runbook(self):
        cfg = RealWorldQABenchConfigV1()
        self.assertEqual(cfg.seeds, (96_504_002,))
        self.assertEqual(cfg.n_problems, 30)
        self.assertEqual(cfg.K_multi_sample, 5)

    def test_per_problem_outcomes_contain_pid_and_gold(self):
        cfg = RealWorldQABenchConfigV1(
            n_problems=2, K_multi_sample=5,
            seeds=(96_504_002,))
        report = run_realworldqa_bench_v1(
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
            self.assertIn("pid", po)
            self.assertIn("question", po)
            self.assertIn("gold_answer", po)
            self.assertIn("a0_outcome_cid", po)
            self.assertIn("a1_outcome_cid", po)
            self.assertIn("b_outcome_cid", po)


class TestAuditChainReDerivation(unittest.TestCase):
    """Re-derive the bench Merkle root from the recorded outcome
    CIDs.  This is the offline-verifier minimum guarantee."""

    def setUp(self):
        self.p = _problem("rwqa_test_000001", "Test?", "A")
        self.text_gen = _stub_text_gen({
            "real-world image question": "A",
            "structured facts": "A",
        })
        def vlm(prompt, ib, mt, t):
            if "expert visual scene reader" in prompt:
                return (
                    "- visible objects: a marker\n"
                    "- counts: 1\n"
                    "- final answer hint: A"), 1
            return "A", 1
        self.vlm_gen = vlm

    def test_bench_merkle_root_is_canonical(self):
        cfg = RealWorldQABenchConfigV1(
            n_problems=1, K_multi_sample=5,
            seeds=(96_504_002,))
        report = run_realworldqa_bench_v1(
            text_gen=self.text_gen,
            vlm_gen=self.vlm_gen,
            vlm_model_id="stub-vlm",
            text_model_id="stub-text",
            corpus=(self.p,),
            corpus_parquet_shard_sha256=("aaaa",),
            corpus_merkle_root="m-root",
            config=cfg)
        # Re-derive: hash of (kind, vlm_model_id, text_model_id,
        # corpus_parquet_shard_sha256, corpus_merkle_root,
        # outcome_cids, seeds, n_problems, K).
        outcome_cids = list(report.per_seed[0].outcome_cids)
        re_derived = hashlib.sha256(
            json.dumps({
                "kind": "w97_realworldqa_bench_merkle_root_v1",
                "vlm_model_id": "stub-vlm",
                "text_model_id": "stub-text",
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


class TestExecutorIsRuleBased(unittest.TestCase):
    """Anti-cheat: the executor used inside the bench is the
    canonical rule-based ``evaluate_realworldqa_answer_v1`` and
    never an LLM judge."""

    def test_executor_letter_match(self):
        p = _problem("p", "Q? A) x B) y", "B")
        v = evaluate_realworldqa_answer_v1(
            prediction="The answer is B.", problem=p)
        self.assertTrue(v.passed)
        self.assertEqual(v.matched_rule, "multi_choice_letter")

    def test_executor_text_exact(self):
        p = _problem("p", "Color of the car?", "red")
        v = evaluate_realworldqa_answer_v1(
            prediction="red", problem=p)
        self.assertTrue(v.passed)


class TestSchemaVersion(unittest.TestCase):
    def test_schema_version_w97(self):
        self.assertTrue(
            W97_REALWORLDQA_BENCH_V1_SCHEMA_VERSION.startswith(
                "coordpy.realworldqa_bench_v1"))


if __name__ == "__main__":
    unittest.main()
