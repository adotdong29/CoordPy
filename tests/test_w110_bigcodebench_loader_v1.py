"""W110 — BigCodeBench loader V1 tests (NIM-free, stdlib-only)."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile

import pytest

from coordpy.bigcodebench_loader_v1 import (
    BigCodeBenchCorpusError,
    assert_bigcodebench_pinned,
    canonical_program_v1,
    load_bigcodebench_v1,
    parse_bigcodebench,
    validate_row_schema,
)


def _row(task_id="BigCodeBench/0", entry="task_func",
         libs="['random', 'itertools']"):
    return {
        "task_id": task_id,
        "complete_prompt": "import random\n\ndef task_func(a, b):\n    \"\"\"add.\"\"\"\n",
        "code_prompt": "import random\n\ndef task_func(a, b):\n",
        "canonical_solution": "    return a + b\n",
        "test": "import unittest\nclass TestCases(unittest.TestCase):\n    def test(self):\n        self.assertEqual(task_func(1,2),3)\n",
        "entry_point": entry,
        "libs": libs,
    }


def test_pin_required():
    with pytest.raises(BigCodeBenchCorpusError):
        assert_bigcodebench_pinned(expected_sha256=None)
    # does not raise when pinned
    assert_bigcodebench_pinned(expected_sha256="deadbeef")


def test_schema_validation():
    ok, _ = validate_row_schema(_row())
    assert ok
    bad = _row()
    del bad["test"]
    ok, reason = validate_row_schema(bad)
    assert not ok and "test" in reason
    empty_entry = _row(entry="")
    ok, reason = validate_row_schema(empty_entry)
    assert not ok and "entry_point" in reason


def test_parse_and_libs_coercion():
    raw = ("\n".join(json.dumps(_row(task_id=f"BigCodeBench/{i}"))
                      for i in range(3))).encode("utf-8")
    probs = parse_bigcodebench(raw)
    assert len(probs) == 3
    # libs is a python-repr string -> coerced to a real tuple
    assert probs[0].libs == ("random", "itertools")
    assert probs[0].n_libs() == 2
    assert probs[0].entry_point == "task_func"


def test_parse_refuses_bad_schema():
    bad = _row()
    del bad["canonical_solution"]
    raw = (json.dumps(bad)).encode("utf-8")
    with pytest.raises(BigCodeBenchCorpusError):
        parse_bigcodebench(raw)


def test_canonical_program_assembly():
    probs = parse_bigcodebench(json.dumps(_row()).encode("utf-8"))
    prog = canonical_program_v1(probs[0])
    assert "def task_func(a, b):" in prog
    assert "return a + b" in prog
    # prompt comes before the body
    assert prog.index("def task_func") < prog.index("return a + b")


def test_sha_pin_enforced_on_load():
    rows = "\n".join(json.dumps(_row(task_id=f"BigCodeBench/{i}"))
                     for i in range(2)) + "\n"
    raw = rows.encode("utf-8")
    sha = hashlib.sha256(raw).hexdigest()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "bcb.jsonl")
        with open(path, "wb") as f:
            f.write(raw)
        # correct SHA loads
        probs = load_bigcodebench_v1(cache_path=path, expected_sha256=sha)
        assert len(probs) == 2
        # wrong SHA refuses
        with pytest.raises(BigCodeBenchCorpusError):
            load_bigcodebench_v1(cache_path=path, expected_sha256="0" * 64)
