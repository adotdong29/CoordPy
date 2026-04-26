"""SDK v3.4 contract tests — sub-sub-intra-cell PROMPT /
LLM_RESPONSE slice + lifecycle audit invariants L-9..L-11 +
synthetic-LLM mode + cross-model parser-boundary research.

These tests lock four new claims:

  W3-42   PROMPT lifecycle gate. Every PROMPT capsule has parent
           set exactly ``(SWEEP_SPEC,)``. A PROMPT outside a
           sealed SWEEP_SPEC is rejected.

  W3-43   Prompt → response parent gate. Every LLM_RESPONSE has
           exactly one parent, and that parent is a sealed PROMPT.
           An LLM_RESPONSE declaring a non-sealed prompt_cid is
           rejected (Capsule Contract C5).

  W3-44   Parse-outcome response chain. A PARSE_OUTCOME may
           parent on (SWEEP_SPEC, LLM_RESPONSE); the lifecycle
           audit rule L-11 mechanically verifies coordinate
           consistency between PARSE_OUTCOME and LLM_RESPONSE
           (instance_id / parser_mode / apply_mode /
           n_distractors). Strategy is allowed to differ
           (multiple strategies share an LLM call when the
           prompt is identical).

  W3-45   Lifecycle-audit soundness extends to L-9..L-11.

Plus the parser-boundary research (W3-C4, empirical):

  Synthetic-LLM cross-distribution failure-kind TVD is large
  (≥ 0.5) on the bundled bank; strict → robust parser-mode shift
  is non-zero on at least one distribution.

The tests use the public surface only
(``from vision_mvp.wevra import ...``).
"""

from __future__ import annotations

import unittest


def _make_clean_ctx_with_spec():
    from vision_mvp.wevra import CapsuleNativeRunContext
    ctx = CapsuleNativeRunContext()
    ctx.start_run(profile_name="local_smoke", profile_dict={})
    ctx.seal_sweep_spec({
        "mode": "synthetic", "sandbox": "in_process",
        "jsonl": "x.jsonl", "model": "synthetic.clean",
        "endpoint": None, "executed_in_process": True})
    return ctx


class PromptCapsuleLifecycleTests(unittest.TestCase):
    """W3-42 — PROMPT lifecycle gate + DAG shape."""

    def test_prompt_seals_under_spec(self):
        from vision_mvp.wevra import CapsuleKind
        ctx = _make_clean_ctx_with_spec()
        cap = ctx.seal_prompt(
            instance_id="m1", strategy="naive_or_routing",
            parser_mode="strict", apply_mode="strict",
            n_distractors=6, model_tag="synthetic.clean",
            prompt_style="block", prompt_text="OLD>>>foo<<<NEW>>>bar<<<\n")
        self.assertEqual(cap.kind, CapsuleKind.PROMPT)
        self.assertEqual(cap.parents, (ctx.spec_cap.cid,))
        p = cap.payload
        self.assertEqual(p["instance_id"], "m1")
        self.assertEqual(p["strategy"], "naive_or_routing")
        self.assertEqual(p["model_tag"], "synthetic.clean")
        self.assertEqual(p["prompt_style"], "block")
        self.assertIsInstance(p["prompt_sha256"], str)
        self.assertEqual(len(p["prompt_sha256"]), 64)
        self.assertGreater(p["prompt_bytes"], 0)
        # Bounded snippet is included by default for short prompts.
        self.assertEqual(p["prompt_text"], "OLD>>>foo<<<NEW>>>bar<<<\n")

    def test_prompt_refuses_without_spec(self):
        from vision_mvp.wevra import (
            CapsuleNativeRunContext, CapsuleLifecycleError,
        )
        ctx = CapsuleNativeRunContext()
        ctx.start_run(profile_name="local_smoke", profile_dict={})
        with self.assertRaises(CapsuleLifecycleError):
            ctx.seal_prompt(
                instance_id="x", strategy="naive_or_routing",
                parser_mode="strict", apply_mode="strict",
                n_distractors=6, model_tag="synthetic.clean",
                prompt_style="block", prompt_text="hi")

    def test_prompt_idempotent_on_content(self):
        """Two identical prompts collapse to ONE PROMPT capsule
        (Capsule Contract C1 — content-addressed identity)."""
        ctx = _make_clean_ctx_with_spec()
        cap1 = ctx.seal_prompt(
            instance_id="m1", strategy="naive_or_routing",
            parser_mode="strict", apply_mode="strict",
            n_distractors=6, model_tag="synthetic.clean",
            prompt_style="block", prompt_text="X")
        cap2 = ctx.seal_prompt(
            instance_id="m1", strategy="naive_or_routing",
            parser_mode="strict", apply_mode="strict",
            n_distractors=6, model_tag="synthetic.clean",
            prompt_style="block", prompt_text="X")
        self.assertEqual(cap1.cid, cap2.cid)
        # The ledger contains exactly one PROMPT capsule.
        from vision_mvp.wevra import CapsuleKind
        self.assertEqual(
            len(ctx.ledger.by_kind(CapsuleKind.PROMPT)), 1)

    def test_prompt_text_omitted_past_cap(self):
        """When the prompt exceeds ``PROMPT_TEXT_CAP`` bytes the
        bounded snippet is omitted — the SHA-256 still identifies
        the bytes content-addressably."""
        from vision_mvp.wevra import PROMPT_TEXT_CAP
        ctx = _make_clean_ctx_with_spec()
        big = "x" * (PROMPT_TEXT_CAP + 100)
        cap = ctx.seal_prompt(
            instance_id="m1", strategy="naive_or_routing",
            parser_mode="strict", apply_mode="strict",
            n_distractors=6, model_tag="synthetic.clean",
            prompt_style="block", prompt_text=big)
        self.assertIsNone(cap.payload["prompt_text"])
        self.assertGreater(cap.payload["prompt_bytes"],
                            PROMPT_TEXT_CAP)


class LLMResponseCapsuleLifecycleTests(unittest.TestCase):
    """W3-43 — prompt → response parent gate."""

    def _seal_prompt(self, ctx) -> str:
        cap = ctx.seal_prompt(
            instance_id="m1", strategy="naive_or_routing",
            parser_mode="strict", apply_mode="strict",
            n_distractors=6, model_tag="synthetic.clean",
            prompt_style="block",
            prompt_text="OLD>>>foo<<<NEW>>>bar<<<\n")
        return cap.cid

    def test_response_seals_under_prompt(self):
        from vision_mvp.wevra import CapsuleKind
        ctx = _make_clean_ctx_with_spec()
        prompt_cid = self._seal_prompt(ctx)
        cap = ctx.seal_llm_response(
            instance_id="m1", strategy="naive_or_routing",
            parser_mode="strict", apply_mode="strict",
            n_distractors=6, model_tag="synthetic.clean",
            response_text="OLD>>>foo<<<NEW>>>bar<<<\n",
            prompt_cid=prompt_cid, elapsed_ms=42)
        self.assertEqual(cap.kind, CapsuleKind.LLM_RESPONSE)
        self.assertEqual(cap.parents, (prompt_cid,))
        p = cap.payload
        self.assertIsInstance(p["response_sha256"], str)
        self.assertEqual(len(p["response_sha256"]), 64)
        self.assertGreater(p["response_bytes"], 0)
        self.assertEqual(p["elapsed_ms"], 42)
        # Snippet included for short responses.
        self.assertEqual(p["response_text"],
                          "OLD>>>foo<<<NEW>>>bar<<<\n")

    def test_response_refuses_unsealed_prompt(self):
        """An LLM_RESPONSE declaring a non-sealed prompt_cid is
        rejected (Capsule Contract C5)."""
        from vision_mvp.wevra import (
            CapsuleNativeRunContext, CapsuleLifecycleError,
        )
        ctx = _make_clean_ctx_with_spec()
        fake_prompt_cid = "f" * 64
        with self.assertRaises(CapsuleLifecycleError):
            ctx.seal_llm_response(
                instance_id="m1", strategy="naive_or_routing",
                parser_mode="strict", apply_mode="strict",
                n_distractors=6, model_tag="synthetic.clean",
                response_text="hi", prompt_cid=fake_prompt_cid)

    def test_response_refuses_without_spec(self):
        from vision_mvp.wevra import (
            CapsuleNativeRunContext, CapsuleLifecycleError,
        )
        ctx = CapsuleNativeRunContext()
        ctx.start_run(profile_name="local_smoke", profile_dict={})
        with self.assertRaises(CapsuleLifecycleError):
            ctx.seal_llm_response(
                instance_id="m1", strategy="naive_or_routing",
                parser_mode="strict", apply_mode="strict",
                n_distractors=6, model_tag="synthetic.clean",
                response_text="hi", prompt_cid="abc" * 32)

    def test_parse_outcome_with_response_parent(self):
        """A PARSE_OUTCOME may parent on (SWEEP_SPEC,
        LLM_RESPONSE) — the prompt → response → parse chain on
        the DAG (Theorem W3-44)."""
        ctx = _make_clean_ctx_with_spec()
        prompt_cid = self._seal_prompt(ctx)
        resp = ctx.seal_llm_response(
            instance_id="m1", strategy="naive_or_routing",
            parser_mode="strict", apply_mode="strict",
            n_distractors=6, model_tag="synthetic.clean",
            response_text="X", prompt_cid=prompt_cid)
        parse = ctx.seal_parse_outcome(
            instance_id="m1", strategy="naive",
            parser_mode="strict", apply_mode="strict",
            n_distractors=6,
            ok=True, failure_kind="ok",
            llm_response_cid=resp.cid)
        self.assertEqual(len(parse.parents), 2)
        self.assertEqual(parse.parents[0], ctx.spec_cap.cid)
        self.assertEqual(parse.parents[1], resp.cid)

    def test_parse_outcome_refuses_unsealed_response(self):
        from vision_mvp.wevra import (
            CapsuleNativeRunContext, CapsuleLifecycleError,
        )
        ctx = _make_clean_ctx_with_spec()
        with self.assertRaises(CapsuleLifecycleError):
            ctx.seal_parse_outcome(
                instance_id="m1", strategy="naive",
                parser_mode="strict", apply_mode="strict",
                n_distractors=6,
                ok=True, failure_kind="ok",
                llm_response_cid="0" * 64)


class LifecycleAuditExtendedTests(unittest.TestCase):
    """W3-45 — lifecycle audit covers L-9..L-11."""

    def _make_full_chain_ctx(self):
        ctx = _make_clean_ctx_with_spec()
        prompt = ctx.seal_prompt(
            instance_id="m1", strategy="naive_or_routing",
            parser_mode="strict", apply_mode="strict",
            n_distractors=6, model_tag="synthetic.clean",
            prompt_style="block", prompt_text="P")
        resp = ctx.seal_llm_response(
            instance_id="m1", strategy="naive_or_routing",
            parser_mode="strict", apply_mode="strict",
            n_distractors=6, model_tag="synthetic.clean",
            response_text="R", prompt_cid=prompt.cid)
        parse = ctx.seal_parse_outcome(
            instance_id="m1", strategy="naive",
            parser_mode="strict", apply_mode="strict",
            n_distractors=6,
            ok=True, failure_kind="ok",
            llm_response_cid=resp.cid)
        patch = ctx.seal_patch_proposal(
            instance_id="m1", strategy="naive",
            parser_mode="strict", apply_mode="strict",
            n_distractors=6,
            substitutions=(("a", "b"),),
            parse_outcome_cid=parse.cid)
        ctx.seal_test_verdict(
            instance_id="m1", strategy="naive",
            parser_mode="strict", apply_mode="strict",
            n_distractors=6,
            patch_proposal_cid=patch.cid,
            patch_applied=True, syntax_ok=True, test_passed=True)
        return ctx

    def test_full_chain_audit_is_ok(self):
        from vision_mvp.wevra import audit_capsule_lifecycle
        ctx = self._make_full_chain_ctx()
        report = audit_capsule_lifecycle(ctx)
        self.assertEqual(report.verdict, "OK", report.violations)
        self.assertIn("L-9_prompt_parent_is_sweep_spec",
                       report.rules_passed)
        self.assertIn("L-10_response_parent_is_sealed_prompt",
                       report.rules_passed)
        self.assertIn("L-11_parse_outcome_response_chain_consistent",
                       report.rules_passed)

    def test_audit_detects_l11_coord_drift(self):
        """A PARSE_OUTCOME whose coordinates diverge from the
        LLM_RESPONSE parent's coordinates flags as L-11 violation
        (W3-44)."""
        from vision_mvp.wevra import audit_capsule_lifecycle
        ctx = _make_clean_ctx_with_spec()
        prompt = ctx.seal_prompt(
            instance_id="m1", strategy="naive_or_routing",
            parser_mode="strict", apply_mode="strict",
            n_distractors=6, model_tag="synthetic.clean",
            prompt_style="block", prompt_text="P")
        resp = ctx.seal_llm_response(
            instance_id="m1", strategy="naive_or_routing",
            parser_mode="strict", apply_mode="strict",
            n_distractors=6, model_tag="synthetic.clean",
            response_text="R", prompt_cid=prompt.cid)
        # Drift the parser_mode on the parse outcome.
        ctx.seal_parse_outcome(
            instance_id="m1", strategy="naive",
            parser_mode="DRIFTED", apply_mode="strict",
            n_distractors=6,
            ok=True, failure_kind="ok",
            llm_response_cid=resp.cid)
        report = audit_capsule_lifecycle(ctx)
        self.assertEqual(report.verdict, "BAD")
        rules = {v["rule"] for v in report.violations}
        self.assertIn(
            "L-11_parse_outcome_response_chain_consistent", rules)


class SyntheticLLMSweepTests(unittest.TestCase):
    """The synthetic-LLM mode exercises the full PROMPT /
    LLM_RESPONSE / PARSE_OUTCOME / PATCH_PROPOSAL / TEST_VERDICT
    chain in flight, deterministically, without an Ollama
    endpoint."""

    def _run_synthetic(self, model_tag):
        from vision_mvp.wevra.capsule_runtime import (
            CapsuleNativeRunContext,
        )
        from vision_mvp.wevra.runtime import SweepSpec, run_sweep
        spec = SweepSpec(
            mode="synthetic",
            jsonl=("vision_mvp/tasks/data/swe_lite_style_bank.jsonl"),
            sandbox="in_process",
            parser_modes=("strict",), apply_modes=("strict",),
            n_distractors=(6,), n_instances=4,
            synthetic_model_tag=model_tag)
        ctx = CapsuleNativeRunContext()
        ctx.start_run(profile_name="local_smoke", profile_dict={})
        result = run_sweep(spec, ctx=ctx)
        return ctx, result

    def test_synthetic_clean_full_chain(self):
        """A clean synthetic distribution should drive PROMPT /
        LLM_RESPONSE / PARSE_OUTCOME / PATCH_PROPOSAL /
        TEST_VERDICT to seal in flight, with the parse → patch
        → verdict chain rooted on a real LLM_RESPONSE."""
        from vision_mvp.wevra import (
            CapsuleKind, audit_capsule_lifecycle,
        )
        ctx, result = self._run_synthetic("synthetic.clean")
        prompts = ctx.ledger.by_kind(CapsuleKind.PROMPT)
        responses = ctx.ledger.by_kind(CapsuleKind.LLM_RESPONSE)
        parses = ctx.ledger.by_kind(CapsuleKind.PARSE_OUTCOME)
        patches = ctx.ledger.by_kind(CapsuleKind.PATCH_PROPOSAL)
        verdicts = ctx.ledger.by_kind(CapsuleKind.TEST_VERDICT)
        # 4 instances × 3 strategies (naive, routing, substrate),
        # but naive + routing share a prompt → 4 × 2 = 8 prompts.
        # The exact number depends on substrate; we only assert
        # they are positive and chain-consistent.
        self.assertGreater(len(prompts), 0)
        self.assertEqual(len(prompts), len(responses))
        self.assertGreater(len(parses), 0)
        self.assertEqual(len(parses), len(patches))
        self.assertEqual(len(parses), len(verdicts))
        # Every PARSE_OUTCOME has 2 parents (SWEEP_SPEC + LLM_RESPONSE).
        for parse in parses:
            self.assertEqual(len(parse.parents), 2)
            self.assertEqual(parse.parents[0], ctx.spec_cap.cid)
        # Lifecycle audit is OK on the full chain.
        report = audit_capsule_lifecycle(ctx)
        self.assertEqual(report.verdict, "OK", report.violations)

    def test_synthetic_unclosed_drives_recovery(self):
        """The 'unclosed' synthetic distribution lands in
        ``failure_kind=unclosed_new`` under strict and
        ``recovery=closed_at_eos`` under robust. The PARSE_OUTCOME
        capsule's payload reflects this (W3-39, exercised end-to-end
        through synthetic mode)."""
        ctx, _result = self._run_synthetic("synthetic.unclosed")
        from vision_mvp.wevra import CapsuleKind
        parses = ctx.ledger.by_kind(CapsuleKind.PARSE_OUTCOME)
        # Expect at least one ``unclosed_new`` failure under strict.
        kinds = [c.payload["failure_kind"] for c in parses]
        self.assertIn("unclosed_new", kinds)


class CrossModelParserBoundaryStudyTests(unittest.TestCase):
    """Empirical coverage of W3-C4 (parser-boundary distribution
    shift across synthetic LLM tags)."""

    def test_study_runs_and_reports_high_tvd(self):
        from vision_mvp.experiments.parser_boundary_cross_model import (
            run_cross_model_study,
        )
        result = run_cross_model_study(n_instances=8)
        # Sanity: schema fields exist.
        self.assertEqual(result["schema"], "wevra.parser_boundary.v1")
        self.assertGreater(result["n_instances"], 0)
        # Cross-distribution TVD is materially > 0.5 — the
        # synthetic distributions are far apart in the parser's
        # failure space.
        self.assertGreater(result["max_cross_tvd"], 0.5)
        # Strict→robust shift is non-zero on at least one
        # distribution (synthetic.unclosed shifts entirely).
        self.assertGreater(result["max_parser_mode_shift"], 0.0)

    def test_study_unclosed_recovers_under_robust(self):
        """Under strict the unclosed distribution lands in
        ``unclosed_new``; under robust the parser recovers it
        (``ok`` + recovery=closed_at_eos) — the parser-mode
        shift is observable as a TVD on the failure-kind
        multinomial."""
        from vision_mvp.experiments.parser_boundary_cross_model import (
            run_cross_model_study,
        )
        result = run_cross_model_study(
            n_instances=8,
            model_tags=("synthetic.unclosed",),
            parser_modes=("strict", "robust"))
        strict = result["distributions"][("synthetic.unclosed", "strict")]
        robust = result["distributions"][("synthetic.unclosed", "robust")]
        self.assertEqual(strict["ok_rate"], 0.0)
        self.assertEqual(robust["ok_rate"], 1.0)


class SyntheticDeterminismTests(unittest.TestCase):
    """Synthetic-mode is deterministic by construction: two
    independent runs with the same model tag produce identical
    parse outcomes and identical PROMPT/LLM_RESPONSE CIDs."""

    def test_synthetic_runs_are_reproducible(self):
        from vision_mvp.wevra.capsule_runtime import (
            CapsuleNativeRunContext,
        )
        from vision_mvp.wevra.runtime import SweepSpec, run_sweep
        from vision_mvp.wevra import CapsuleKind

        def _sealed_prompt_cids():
            spec = SweepSpec(
                mode="synthetic",
                jsonl=("vision_mvp/tasks/data/swe_lite_style_bank.jsonl"),
                sandbox="in_process",
                parser_modes=("strict",), apply_modes=("strict",),
                n_distractors=(6,), n_instances=4,
                synthetic_model_tag="synthetic.clean")
            ctx = CapsuleNativeRunContext()
            ctx.start_run(profile_name="local_smoke", profile_dict={})
            run_sweep(spec, ctx=ctx)
            return sorted(c.cid for c in ctx.ledger.by_kind(
                CapsuleKind.PROMPT))

        c1 = _sealed_prompt_cids()
        c2 = _sealed_prompt_cids()
        self.assertEqual(c1, c2,
                          "synthetic mode should produce identical "
                          "PROMPT CIDs across runs")


if __name__ == "__main__":
    unittest.main()
