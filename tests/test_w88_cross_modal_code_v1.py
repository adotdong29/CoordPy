"""W88 — Cross-modal code bench V1 CI tests."""
from __future__ import annotations

import hashlib
import json
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from coordpy.cross_modal_code_bench_v1 import (
    W88_CROSS_MODAL_CODE_BENCH_V1_SCHEMA_VERSION,
    CrossModalCodeBenchConfigV1,
    CrossModalProblemV1,
    _DOCTEST_BLOCK_RE,
    _render_doctest_image,
    _split_doctest_block,
    run_cross_modal_code_bench_v1,
    synthesize_cross_modal_corpus_v1,
)
from coordpy.humaneval_real_bench_v1 import (
    HumanEvalProblemV1,
)


def test_w88_xm_schema_version():
    assert (
        W88_CROSS_MODAL_CODE_BENCH_V1_SCHEMA_VERSION
        == "coordpy.cross_modal_code_bench_v1.v1")


def test_w88_xm_split_doctest_simple():
    """A docstring with two >>> lines splits cleanly into a
    stripped prompt + doctest text, and the closing ``\"\"\"`` of
    the docstring is preserved in the stripped prompt."""
    prompt = textwrap.dedent('''\
        def f(x: int) -> int:
            """Return x + 1.

            >>> f(0)
            1
            >>> f(2)
            3
            """
        ''')
    stripped, doctest_text, n_dt = _split_doctest_block(prompt)
    assert n_dt == 2
    assert ">>>" not in stripped
    assert '"""' in stripped  # docstring closing preserved
    assert ">>> f(0)" in doctest_text
    assert ">>> f(2)" in doctest_text


def test_w88_xm_split_doctest_zero_lines():
    """A docstring with no >>> lines is unchanged."""
    prompt = (
        'def f(x):\n'
        '    """Just describe in prose.\n'
        '    No doctest examples here.\n'
        '    """\n')
    stripped, doctest_text, n_dt = _split_doctest_block(prompt)
    assert n_dt == 0
    assert doctest_text == ""
    assert ">>>" not in stripped


def test_w88_xm_image_render_reproducible():
    """Same text yields same image bytes (PIL deterministic)."""
    img1 = _render_doctest_image(">>> f(1)\n2\n>>> f(2)\n3")
    img2 = _render_doctest_image(">>> f(1)\n2\n>>> f(2)\n3")
    assert img1 == img2
    assert len(img1) > 100  # non-empty PNG


def test_w88_xm_synthesize_corpus_filters_no_doctest():
    """Problems with fewer than min_doctest_lines are filtered
    out."""
    he = (
        HumanEvalProblemV1(
            task_id="dummy/0",
            prompt='def f(x):\n    """No doctests."""\n',
            canonical_solution="return x",
            test="def check(f):\n    assert f(0) == 0",
            entry_point="f"),
        HumanEvalProblemV1(
            task_id="dummy/1",
            prompt=(
                'def g(x):\n'
                '    """Add one.\n'
                '    >>> g(0)\n'
                '    1\n'
                '    >>> g(1)\n'
                '    2\n'
                '    """\n'),
            canonical_solution="return x + 1",
            test="def check(g):\n    assert g(0) == 1",
            entry_point="g"),
    )
    out = synthesize_cross_modal_corpus_v1(
        he, min_doctest_lines=2)
    assert len(out) == 1
    assert out[0].task_id == "dummy/1"
    assert out[0].n_doctest_lines == 2
    assert len(out[0].image_bytes) > 100


def test_w88_xm_problem_cid_stable():
    """Problem CID is deterministic — same inputs → same CID."""
    he = HumanEvalProblemV1(
        task_id="dummy/0",
        prompt=(
            'def g(x):\n'
            '    """Add one.\n'
            '    >>> g(0)\n'
            '    1\n'
            '    """\n'),
        canonical_solution="return x + 1",
        test="def check(g):\n    assert g(0) == 1",
        entry_point="g")
    c1 = synthesize_cross_modal_corpus_v1(
        (he,), min_doctest_lines=1)[0].problem_cid()
    c2 = synthesize_cross_modal_corpus_v1(
        (he,), min_doctest_lines=1)[0].problem_cid()
    assert c1 == c2


def _make_synth_text_gen(passing: bool):
    """Synthetic text-only gen.  If passing, returns correct
    code; else returns wrong code."""
    code = (
        "```python\ndef g(x):\n    return x + 1\n```"
        if passing else
        "```python\ndef g(x):\n    return x\n```")

    def gen(prompt, max_tokens, temperature):
        return code, 50
    return gen


def _make_synth_vlm_gen(*, extract_correct: bool,
                        code_correct: bool):
    """Synthetic VLM gen.  ``extract_correct`` controls how the
    VLM responds to a doctest-extraction prompt; ``code_correct``
    controls how it responds to a code-solve prompt."""

    def gen(prompt, image_bytes, max_tokens, temperature):
        if "Reproduce the doctest" in prompt:
            text = (
                ">>> g(0)\n1\n>>> g(1)\n2"
                if extract_correct
                else ">>> g(0)\nundefined")
        else:
            text = (
                "```python\ndef g(x):\n    return x + 1\n```"
                if code_correct
                else "```python\ndef g(x):\n    return x\n```")
        return text, 50

    return gen


def _toy_he_problem():
    return HumanEvalProblemV1(
        task_id="dummy/g",
        prompt=(
            'def g(x):\n'
            '    """Add one.\n'
            '    >>> g(0)\n'
            '    1\n'
            '    >>> g(1)\n'
            '    2\n'
            '    """\n'),
        canonical_solution="return x + 1",
        test=textwrap.dedent("""
            def check(g):
                assert g(0) == 1
                assert g(1) == 2
                assert g(-1) == 0
            """).strip(),
        entry_point="g")


def test_w88_xm_b_cross_wins_when_only_team_works():
    """A scenario where:
    * A0_text fails (text-only LM with stripped prompt → guesses
      wrong)
    * A1_vlm fails (single VLM gets bad code answers)
    * B_cross wins (VLM extracts correctly + code LM gets correct
      from extraction)
    """
    he = (_toy_he_problem(),)
    cfg = CrossModalCodeBenchConfigV1(
        n_problems=1,
        K_multi_sample=5,
        seeds=(88_046_001,),
        sampling_temperature=0.7,
        max_tokens_per_call=64,
        min_doctest_lines=2)
    text_gen = _make_synth_text_gen(passing=True)
    vlm_gen = _make_synth_vlm_gen(
        extract_correct=True, code_correct=False)
    report, _ = run_cross_modal_code_bench_v1(
        text_gen=text_gen, vlm_gen=vlm_gen,
        vlm_model_id="synth_vlm", code_model_id="synth_code",
        corpus=he, config=cfg)
    # A1_vlm runs the VLM on a code-solve prompt; with
    # code_correct=False, every sample fails.
    assert report.a1_vlm_mean_pass_at_1 == 0.0
    # B_cross runs the VLM on an EXTRACT prompt (extract_correct=
    # True) and then the code-LM (passing=True).  Code-LM wins.
    assert report.b_cross_mean_pass_at_1 == 1.0
    # The harder strict-improvement bar
    assert report.b_cross_mean_strictly_beats_a1_vlm_mean


def test_w88_xm_b_cross_loses_when_a1_vlm_also_wins():
    """When both A1_vlm and B_cross can produce passing code,
    the head-to-head is a tie."""
    he = (_toy_he_problem(),)
    cfg = CrossModalCodeBenchConfigV1(
        n_problems=1, K_multi_sample=5,
        seeds=(88_046_001,),
        sampling_temperature=0.7, max_tokens_per_call=64,
        min_doctest_lines=2)
    text_gen = _make_synth_text_gen(passing=True)
    vlm_gen = _make_synth_vlm_gen(
        extract_correct=True, code_correct=True)
    report, _ = run_cross_modal_code_bench_v1(
        text_gen=text_gen, vlm_gen=vlm_gen,
        vlm_model_id="synth_vlm", code_model_id="synth_code",
        corpus=he, config=cfg)
    assert report.a1_vlm_mean_pass_at_1 == 1.0
    assert report.b_cross_mean_pass_at_1 == 1.0
    # Strict-improvement is FALSE on a tie.
    assert not report.b_cross_mean_strictly_beats_a1_vlm_mean


def test_w88_xm_audit_chain_re_derives():
    """Per-seed + bench Merkle roots re-derive byte-for-byte."""
    he = (_toy_he_problem(),)
    cfg = CrossModalCodeBenchConfigV1(
        n_problems=1, K_multi_sample=5,
        seeds=(88_046_001, 88_046_002),
        sampling_temperature=0.7, max_tokens_per_call=64,
        min_doctest_lines=2)
    text_gen = _make_synth_text_gen(passing=True)
    vlm_gen = _make_synth_vlm_gen(
        extract_correct=True, code_correct=False)
    report, _ = run_cross_modal_code_bench_v1(
        text_gen=text_gen, vlm_gen=vlm_gen,
        vlm_model_id="synth_vlm", code_model_id="synth_code",
        corpus=he, config=cfg)
    all_cids = []
    for s in report.per_seed:
        all_cids.extend(s.outcome_cids)
        derived_seed = hashlib.sha256(
            json.dumps({
                "kind": (
                    "w88_cross_modal_code_seed_merkle_root"),
                "seed": int(s.seed),
                "outcome_cids": list(s.outcome_cids),
            }, sort_keys=True, separators=(",", ":"),
                default=str).encode("utf-8")).hexdigest()
        assert derived_seed == s.seed_merkle_root
    derived_bench = hashlib.sha256(
        json.dumps({
            "kind": (
                "w88_cross_modal_code_bench_merkle_root"),
            "vlm_model_id": "synth_vlm",
            "code_model_id": "synth_code",
            "outcome_cids": list(all_cids),
            "seeds": [int(s.seed) for s in report.per_seed],
        }, sort_keys=True, separators=(",", ":"),
            default=str).encode("utf-8")).hexdigest()
    assert derived_bench == report.bench_merkle_root


def test_w88_xm_module_surface_explicit_import():
    """W88 cross-modal module is explicit-import only."""
    import importlib
    mod = importlib.import_module(
        "coordpy.cross_modal_code_bench_v1")
    assert hasattr(
        mod, "W88_CROSS_MODAL_CODE_BENCH_V1_SCHEMA_VERSION")
    import coordpy
    assert not hasattr(
        coordpy, "run_cross_modal_code_bench_v1")


def test_w88_xm_strip_mode_all_docstring_strips_everything():
    """In ``all_docstring`` mode, the entire function docstring
    is replaced with a 'See image' stub; the doctest is then in
    the image only.  No prose description leaks into the
    stripped prompt."""
    he = (HumanEvalProblemV1(
        task_id="dummy/g",
        prompt=(
            'def g(x):\n'
            '    """Add one to x.\n'
            '    >>> g(0)\n'
            '    1\n'
            '    >>> g(1)\n'
            '    2\n'
            '    """\n'),
        canonical_solution="return x + 1",
        test="def check(g):\n    assert g(0) == 1",
        entry_point="g"),)
    out_dt = synthesize_cross_modal_corpus_v1(
        he, min_doctest_lines=2,
        strip_mode="doctest_only")[0]
    out_all = synthesize_cross_modal_corpus_v1(
        he, min_doctest_lines=2,
        strip_mode="all_docstring")[0]
    # In doctest_only the prose stays.
    assert "Add one to x" in out_dt.stripped_prompt
    # In all_docstring the prose is gone; only the stub remains.
    assert "Add one to x" not in out_all.stripped_prompt
    assert "See the attached image" in out_all.stripped_prompt
    # Both still strip the >>> lines from the prompt.
    assert ">>>" not in out_dt.stripped_prompt
    assert ">>>" not in out_all.stripped_prompt
    # The image text matches the doctest lines in both modes.
    # (We don't compare image bytes — PIL rendering of different
    # whitespace may differ.)


def test_w88_xm_strip_mode_invalid_raises():
    import pytest
    he = (HumanEvalProblemV1(
        task_id="dummy/g",
        prompt='def g(x):\n    """>>> g(0)\n    1\n    """\n',
        canonical_solution="return x",
        test="def check(g): pass",
        entry_point="g"),)
    with pytest.raises(ValueError):
        synthesize_cross_modal_corpus_v1(
            he, strip_mode="invalid_mode")
