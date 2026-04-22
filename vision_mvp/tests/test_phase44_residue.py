"""Phase 44 tests — raw-text capture, refined semantic taxonomy,
and public SWE-bench-Lite readiness.

Five blocks:

  1. ``RawCaptureStore`` schema + round-trip (read/write).
  2. The capturing generator captures raw text, the parse outcome,
     and proposed substitutions.
  3. Refined classifier (``classify_semantic_outcome_v2``) partitions
     coarse Phase-43 labels cleanly and is monotone on sentinel
     inputs (= Phase-43 behaviour).
  4. Phase-44 sweep driver produces paired parent + capture
     artifacts on mock.
  5. Public SWE-bench-Lite readiness verdict on the bundled bank
     (``ready: true`` on all five checks, N = 57).
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")))

from vision_mvp.tasks.swe_raw_capture import (
    SCHEMA_VERSION, RawCaptureRecord, RawCaptureStore,
    make_capturing_generator,
)
from vision_mvp.tasks.swe_semantic_taxonomy import (
    ALL_REFINED_LABELS, ALL_SEMANTIC_LABELS,
    ALL_SEMANTIC_LABELS_V2, REFINEMENT_MAP,
    SEM_INCOMPLETE_MULTI_HUNK, SEM_NARROW_FIX_TEST_OVERFIT, SEM_OK,
    SEM_PARTIAL_MULTI_HUNK_SUCCESS, SEM_RIGHT_FILE_WRONG_SPAN,
    SEM_RIGHT_SITE_WRONG_LOGIC, SEM_RIGHT_SPAN_WRONG_LOGIC,
    SEM_STRUCTURAL_SEMANTIC_INERT, SEM_STRUCTURAL_VALID_INERT,
    SEM_TEST_OVERFIT, SEM_WRONG_EDIT_SITE,
    classify_semantic_outcome, classify_semantic_outcome_v2,
    refine_semantic_outcome,
)


# ---------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------


_BUGGY_SRC = (
    "def add(a, b):\n"
    "    return a + b\n"
    "\n"
    "def factorial(n):\n"
    "    result = 0\n"
    "    for i in range(1, n + 1):\n"
    "        result *= i\n"
    "    return result\n"
    "\n"
    "def is_prime(n):\n"
    "    if n < 2:\n"
    "        return False\n"
    "    return True\n"
)


_GOLD_FACT = (("    result = 0\n", "    result = 1\n"),)


# ---------------------------------------------------------------
# Block 1 — RawCaptureStore round-trip
# ---------------------------------------------------------------


def test_phase44_capture_store_schema_version():
    s = RawCaptureStore(meta={"x": 1})
    d = s.as_dict()
    assert d["schema_version"] == SCHEMA_VERSION
    assert d["meta"] == {"x": 1}
    assert d["records"] == []


def test_phase44_capture_record_roundtrip():
    r = RawCaptureRecord(
        instance_id="ext-calc-001", strategy="substrate",
        parser_mode="robust", apply_mode="strict", n_distractors=6,
        raw_text="OLD>>>\nx = 0\n<<<NEW>>>\nx = 1\n<<<",
        raw_text_sha256="deadbeef",
        parse_outcome={"ok": True, "failure_kind": "ok",
                        "recovery": "", "detail": "",
                        "substitutions_count": 1},
        proposed_patch=(("x = 0\n", "x = 1\n"),),
        applied_patch=(("x = 0\n", "x = 1\n"),),
        patched_source_sha256="cafef00d",
        error_kind="", test_passed=True, captured_at_s=1.0,
    )
    d = r.as_dict()
    r2 = RawCaptureRecord.from_dict(d)
    assert r2 == r


def test_phase44_capture_store_read_write(tmp_path=None):
    if tmp_path is None:
        tmp_path = tempfile.mkdtemp()
    path = os.path.join(str(tmp_path), "cap.json")
    s = RawCaptureStore(meta={"m": "x"})
    s.records.append(RawCaptureRecord(
        instance_id="i", strategy="substrate",
        parser_mode="robust", apply_mode="strict", n_distractors=6,
        raw_text="", raw_text_sha256="", parse_outcome={},
        proposed_patch=(), applied_patch=(),
        patched_source_sha256="", error_kind="", test_passed=False,
        captured_at_s=0.0,
    ))
    s.write(path)
    s2 = RawCaptureStore.read(path)
    assert s2.meta == {"m": "x"}
    assert len(s2.records) == 1
    assert s2.records[0].instance_id == "i"


def test_phase44_capture_store_version_mismatch_raises(tmp_path=None):
    if tmp_path is None:
        tmp_path = tempfile.mkdtemp()
    path = os.path.join(str(tmp_path), "bad.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"schema_version": "wrong", "records": []}, fh)
    try:
        RawCaptureStore.read(path)
    except ValueError as ex:
        assert "schema mismatch" in str(ex)
        return
    raise AssertionError("expected ValueError")


# ---------------------------------------------------------------
# Block 2 — Capturing generator captures what it sees
# ---------------------------------------------------------------


def test_phase44_capturing_generator_records_raw_text_and_outcome():
    from vision_mvp.tasks.swe_bench_bridge import (
        SWEBenchStyleTask, parse_unified_diff,
    )
    store = RawCaptureStore(meta={"model": "test"})
    fake_text = (
        "OLD>>>\n"
        "    result = 0\n"
        "<<<NEW>>>\n"
        "    result = 1\n"
        "<<<"
    )
    gen = make_capturing_generator(
        base_gen=None,
        store=store,
        parser_mode="robust",
        apply_mode="strict",
        n_distractors=6,
        llm_call=lambda prompt: fake_text,
        unified_diff_parser=parse_unified_diff,
    )
    task = SWEBenchStyleTask(
        instance_id="ext-calc-001", repo="x/y",
        base_commit="v0", problem_statement="p",
        buggy_file_relpath="calc.py",
        buggy_function="factorial",
        gold_patch=_GOLD_FACT,
        test_source="def test(m): pass",
    )
    ctx = {"hunk": "def factorial(n): ..."}
    out = gen(task, ctx, _BUGGY_SRC, "issue")
    assert out.patch == (("    result = 0", "    result = 1"),)
    assert store._open_rows, "generator did not open a row"
    (key, slot), = store._open_rows.items()
    assert key == ("ext-calc-001", "substrate", "robust", "strict", 6)
    assert slot["raw_text"] == fake_text
    assert slot["parse_outcome"]["ok"] is True
    assert slot["proposed_patch"] == (("    result = 0",
                                        "    result = 1"),)


def test_phase44_capturing_generator_handles_parse_failure():
    from vision_mvp.tasks.swe_bench_bridge import (
        SWEBenchStyleTask, parse_unified_diff,
    )
    store = RawCaptureStore()
    # Prose-only → parse failure under PARSER_ROBUST.
    prose = "The bug is on line 5. Change 0 to 1."
    gen = make_capturing_generator(
        base_gen=None, store=store,
        parser_mode="robust", apply_mode="strict", n_distractors=0,
        llm_call=lambda p: prose,
        unified_diff_parser=parse_unified_diff,
    )
    task = SWEBenchStyleTask(
        instance_id="i", repo="r", base_commit="v0",
        problem_statement="p", buggy_file_relpath="x.py",
        buggy_function="f",
        gold_patch=_GOLD_FACT,
        test_source="def test(m): pass",
    )
    out = gen(task, ctx={"hunk": "f"}, buggy_source=_BUGGY_SRC,
                 issue_summary="")
    assert out.patch == ()
    (_key, slot), = store._open_rows.items()
    assert slot["proposed_patch"] == ()
    assert slot["parse_outcome"]["ok"] is False


# ---------------------------------------------------------------
# Block 3 — Refined classifier
# ---------------------------------------------------------------


def test_phase44_refined_sentinel_path_returns_coarse():
    """When the sentinel proposed_patch is passed (Phase-43 path),
    the refined classifier returns the coarse label unchanged
    (Theorem P44-2 safety)."""
    sentinel = (("__sentinel__", "__sentinel__"),)
    out = refine_semantic_outcome(
        coarse_label=SEM_WRONG_EDIT_SITE,
        buggy_source=_BUGGY_SRC, gold_patch=_GOLD_FACT,
        proposed_patch=sentinel, applied_patch=None,
        patched_source=None,
    )
    assert out == SEM_WRONG_EDIT_SITE


def test_phase44_refined_right_file_wrong_span():
    """Proposed OLD matches inside the buggy source but on a span
    disjoint from gold's OLD — classifier returns
    SEM_RIGHT_FILE_WRONG_SPAN."""
    proposed = (("    if n < 2:\n        return False\n",
                 "    if n < 1:\n        return False\n"),)
    out = refine_semantic_outcome(
        coarse_label=SEM_WRONG_EDIT_SITE,
        buggy_source=_BUGGY_SRC, gold_patch=_GOLD_FACT,
        proposed_patch=proposed,
    )
    assert out == SEM_RIGHT_FILE_WRONG_SPAN


def test_phase44_refined_wrong_site_when_no_match():
    """Proposed OLD doesn't anchor anywhere → remains
    SEM_WRONG_EDIT_SITE."""
    proposed = (("nothing like the buggy source\n",
                 "replacement\n"),)
    out = refine_semantic_outcome(
        coarse_label=SEM_WRONG_EDIT_SITE,
        buggy_source=_BUGGY_SRC, gold_patch=_GOLD_FACT,
        proposed_patch=proposed,
    )
    assert out == SEM_WRONG_EDIT_SITE


def test_phase44_refined_partial_multi_hunk_success():
    """Gold has two hunks; proposed covers one and its NEW agrees
    with gold NEW → SEM_PARTIAL_MULTI_HUNK_SUCCESS."""
    gold = (
        ("    result = 0\n", "    result = 1\n"),
        ("    return result\n", "    return int(result)\n"),
    )
    proposed = (("    result = 0\n", "    result = 1\n"),)
    out = refine_semantic_outcome(
        coarse_label=SEM_INCOMPLETE_MULTI_HUNK,
        buggy_source=_BUGGY_SRC, gold_patch=gold,
        proposed_patch=proposed,
    )
    assert out == SEM_PARTIAL_MULTI_HUNK_SUCCESS


def test_phase44_refined_incomplete_multi_hunk_no_success():
    """Gold has two hunks; proposed covers one but its NEW
    *disagrees* with gold NEW → stays SEM_INCOMPLETE_MULTI_HUNK."""
    gold = (
        ("    result = 0\n", "    result = 1\n"),
        ("    return result\n", "    return int(result)\n"),
    )
    # Proposed fixes the first site but WRONG logic.
    proposed = (("    result = 0\n", "    result = -1\n"),)
    out = refine_semantic_outcome(
        coarse_label=SEM_INCOMPLETE_MULTI_HUNK,
        buggy_source=_BUGGY_SRC, gold_patch=gold,
        proposed_patch=proposed,
    )
    assert out == SEM_INCOMPLETE_MULTI_HUNK


def test_phase44_refined_right_span_wrong_logic_renaming():
    """coarse SEM_RIGHT_SITE_WRONG_LOGIC + real OLD overlap →
    SEM_RIGHT_SPAN_WRONG_LOGIC (Phase-44 spelling)."""
    gold = (("    result = 0\n"
              "    for i in range(1, n + 1):\n",
              "    result = 1\n"
              "    for i in range(1, n + 1):\n"),)
    proposed = (("    result = 0\n"
                  "    for i in range(1, n + 1):\n",
                  "    result = -1\n"
                  "    for i in range(1, n + 1):\n"),)
    out = refine_semantic_outcome(
        coarse_label=SEM_RIGHT_SITE_WRONG_LOGIC,
        buggy_source=_BUGGY_SRC, gold_patch=gold,
        proposed_patch=proposed,
    )
    assert out == SEM_RIGHT_SPAN_WRONG_LOGIC


def test_phase44_refined_narrow_fix_test_overfit():
    """Shared token overlap between gold NEW and proposed NEW with
    a narrowing guard → SEM_NARROW_FIX_TEST_OVERFIT."""
    gold = (("    result = 0\n"
              "    for i in range(1, n + 1):\n",
              "    result = 1\n"
              "    for i in range(1, n + 1):\n"),)
    proposed = (("    result = 0\n"
                  "    for i in range(1, n + 1):\n",
                  "    result = 1 if n > 0 else 0\n"
                  "    for i in range(1, n + 1):\n"),)
    out = refine_semantic_outcome(
        coarse_label=SEM_TEST_OVERFIT,
        buggy_source=_BUGGY_SRC, gold_patch=gold,
        proposed_patch=proposed,
        error_detail="assert",
    )
    assert out == SEM_NARROW_FIX_TEST_OVERFIT


def test_phase44_refined_structural_valid_inert_wspc_equality():
    """Patched source byte-equal to buggy source under whitespace
    normalisation → SEM_STRUCTURAL_VALID_INERT."""
    # Patched source is literally the buggy source (maximally inert).
    out = refine_semantic_outcome(
        coarse_label=SEM_STRUCTURAL_SEMANTIC_INERT,
        buggy_source=_BUGGY_SRC, gold_patch=_GOLD_FACT,
        proposed_patch=_GOLD_FACT,
        patched_source=_BUGGY_SRC,
    )
    assert out == SEM_STRUCTURAL_VALID_INERT


def test_phase44_refined_structural_inert_when_source_differs():
    """Patched source differs from buggy source → stays
    SEM_STRUCTURAL_SEMANTIC_INERT (cannot prove byte-inertness)."""
    out = refine_semantic_outcome(
        coarse_label=SEM_STRUCTURAL_SEMANTIC_INERT,
        buggy_source=_BUGGY_SRC, gold_patch=_GOLD_FACT,
        proposed_patch=_GOLD_FACT,
        patched_source=_BUGGY_SRC.replace("result = 0",
                                             "result = 'one'"),
    )
    assert out == SEM_STRUCTURAL_SEMANTIC_INERT


def test_phase44_refined_pass_ok_unchanged():
    """SEM_OK never refines."""
    out = refine_semantic_outcome(
        coarse_label=SEM_OK,
        buggy_source=_BUGGY_SRC, gold_patch=_GOLD_FACT,
        proposed_patch=_GOLD_FACT,
    )
    assert out == SEM_OK


def test_phase44_refined_classify_v2_monotone_on_sentinel():
    """v2 classifier on sentinel proposed_patch ≡ v1 classifier
    (safety guarantee).
    """
    sentinel = (("__sentinel__", "__sentinel__"),)
    v1 = classify_semantic_outcome(
        buggy_source=_BUGGY_SRC, gold_patch=_GOLD_FACT,
        proposed_patch=sentinel,
        error_kind="test_assert", test_passed=False)
    v2 = classify_semantic_outcome_v2(
        buggy_source=_BUGGY_SRC, gold_patch=_GOLD_FACT,
        proposed_patch=sentinel,
        error_kind="test_assert", test_passed=False)
    assert v1 == v2


def test_phase44_refinement_map_is_a_legal_partition():
    """Every Phase-43 coarse label mapped in REFINEMENT_MAP partitions
    into a superset that *contains* the coarse label — a strict
    refinement.
    """
    for (coarse, refined_set) in REFINEMENT_MAP.items():
        assert coarse in ALL_SEMANTIC_LABELS
        assert coarse in refined_set, (
            f"coarse label {coarse!r} must remain a legal refined "
            f"label (Phase-44 refinement is a partition, not a replacement)")
        for r in refined_set:
            assert r in ALL_SEMANTIC_LABELS_V2, (
                f"refined label {r!r} not in ALL_SEMANTIC_LABELS_V2")


def test_phase44_refined_labels_are_new():
    """Every new Phase-44 label is in the v2 union and disjoint from
    the Phase-43 v1 set (except for direct synonym SEM_RIGHT_SPAN_WRONG_LOGIC
    which is a strict superset)."""
    v1 = set(ALL_SEMANTIC_LABELS)
    v2 = set(ALL_SEMANTIC_LABELS_V2)
    refined = set(ALL_REFINED_LABELS)
    assert refined.isdisjoint(v1)
    assert refined.issubset(v2)
    assert v1.issubset(v2)


# ---------------------------------------------------------------
# Block 4 — Phase-44 mock sweep produces paired artifacts
# ---------------------------------------------------------------


def test_phase44_mock_sweep_writes_parent_and_capture(tmp_path=None):
    """The Phase-44 driver runs the Phase-42-shape sweep in mock
    mode and produces both the parent artifact and a capture JSON
    whose records cover every (instance, strategy, parser_mode,
    apply_mode, nd) cell.
    """
    if tmp_path is None:
        tmp_path = tempfile.mkdtemp()
    tmp = str(tmp_path)
    import subprocess
    parent = os.path.join(tmp, "parent.json")
    capture = os.path.join(tmp, "cap.json")
    env = {**os.environ, "PYTHONPATH": os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", ".."))}
    r = subprocess.run(
        [sys.executable, "-m",
         "vision_mvp.experiments.phase44_semantic_residue",
         "--mode", "mock",
         "--sandbox", "subprocess",
         "--parser-modes", "strict", "robust",
         "--apply-modes", "strict",
         "--n-distractors", "0",
         "--n-instances", "4",
         "--out-parent", parent,
         "--out-capture", capture],
        capture_output=True, text=True, env=env, timeout=120)
    assert r.returncode == 0, f"driver failed: stderr={r.stderr[-400:]}"
    assert os.path.exists(parent), "parent artifact not written"
    assert os.path.exists(capture), "capture artifact not written"
    with open(parent) as fh:
        p = json.load(fh)
    with open(capture) as fh:
        c = json.load(fh)
    assert c["schema_version"] == SCHEMA_VERSION
    # 4 instances × 2 parser modes × 1 apply mode × 1 nd × 3 strategies
    # = 24 capture rows.
    assert c["n_records"] == 4 * 2 * 1 * 1 * 3, (
        f"expected 24 rows, got {c['n_records']}")
    # Every measurement in the parent must have a corresponding
    # capture record.
    parent_keys = set()
    for cell in p["cells"]:
        for m in cell["report"]["measurements"]:
            parent_keys.add((
                m["instance_id"], m["strategy"],
                cell["parser_mode"], cell["apply_mode"],
                cell["n_distractors"]))
    capture_keys = set(
        (r["instance_id"], r["strategy"], r["parser_mode"],
         r["apply_mode"], r["n_distractors"])
        for r in c["records"])
    assert parent_keys == capture_keys, (
        "capture keys do not match parent measurements")


# ---------------------------------------------------------------
# Block 5 — public-SWE-bench-Lite readiness verdict
# ---------------------------------------------------------------


def test_phase44_public_readiness_verdict_on_bundled_bank():
    from vision_mvp.experiments.phase44_public_readiness import (
        run_readiness,
    )
    bank_path = os.path.join(
        os.path.dirname(__file__), "..", "tasks", "data",
        "swe_lite_style_bank.jsonl")
    verdict = run_readiness(bank_path, limit=10,
                              sandbox_name="subprocess")
    assert verdict["n"] == 10
    assert verdict["n_passed_all"] == 10, (
        f"readiness blocked on 10-instance subset: "
        f"{verdict['blockers']}")
    assert verdict["ready"] is True


def test_phase44_public_readiness_full_bundled_bank():
    from vision_mvp.experiments.phase44_public_readiness import (
        run_readiness,
    )
    bank_path = os.path.join(
        os.path.dirname(__file__), "..", "tasks", "data",
        "swe_lite_style_bank.jsonl")
    verdict = run_readiness(bank_path, limit=None,
                              sandbox_name="subprocess")
    assert verdict["n"] == 57
    # Every check must pass on every instance.
    for (check, r) in verdict["checks"].items():
        assert r["failed"] == 0, (
            f"readiness check {check!r} failed on "
            f"{r['failed']} / {verdict['n']} instances: "
            f"{r['failures'][:3]}")
    assert verdict["ready"] is True


def test_phase44_public_readiness_detects_broken_jsonl(tmp_path=None):
    from vision_mvp.experiments.phase44_public_readiness import (
        run_readiness,
    )
    if tmp_path is None:
        tmp_path = tempfile.mkdtemp()
    tmp = str(tmp_path)
    broken = os.path.join(tmp, "broken.jsonl")
    with open(broken, "w", encoding="utf-8") as fh:
        # Missing instance_id + missing patch / gold_patch.
        fh.write(json.dumps({"repo": "x"}) + "\n")
    verdict = run_readiness(broken, limit=None,
                              sandbox_name="in_process")
    assert verdict["ready"] is False
    assert verdict["n"] == 1
    assert verdict["checks"]["schema"]["failed"] == 1
    # Blockers list is non-empty.
    assert verdict["blockers"]
