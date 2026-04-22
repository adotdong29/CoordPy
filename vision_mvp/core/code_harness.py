"""Code extraction, AST validation, and sandboxed execution.

LLMs produce prose. We need real Python code that (a) parses, (b) defines
the target function with the target signature, (c) actually runs against
a test harness without escaping the sandbox.

Pipeline:
  1. Extract code from LLM output: prefer fenced ```python blocks,
     fall back to whole text. Parse each candidate with `ast.parse`;
     keep the longest that parses.
  2. Validate: ensure the required `FunctionDef` name is present, with
     a compatible argument list.
  3. Compose: concatenate all agents' accepted code into one module.
  4. Run tests in a SUBPROCESS with CPU + memory + wall-clock limits;
     parse JSON stdout.

Subprocess sandboxing is safer than the `exec` with restricted-builtins
trick, which is known to be bypassable. Not bulletproof (the subprocess
can still touch the filesystem) but sufficient when the model is trusted
but fallible.
"""

from __future__ import annotations
import ast
import re
import json
import resource
import subprocess
import tempfile
import textwrap
import os
from dataclasses import dataclass, field
from typing import Optional


FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)
FENCE_OPEN_RE = re.compile(r"```(?:python|py)?\s*\n(.*)$", re.DOTALL)


def extract_code(text: str) -> str | None:
    """Return the longest code block in `text` that successfully parses as
    Python. Returns None if nothing parses."""
    if not text:
        return None
    # First pass: fenced blocks
    blocks = FENCE_RE.findall(text)
    # Unclosed fence (LLM truncation) — take what's after the last ```
    if not blocks:
        m = FENCE_OPEN_RE.search(text)
        if m:
            blocks = [m.group(1)]
    # Last resort: the whole output
    candidates = list(blocks) if blocks else [text]
    # Sort by length descending; pick first that parses
    candidates.sort(key=len, reverse=True)
    for block in candidates:
        try:
            ast.parse(textwrap.dedent(block))
            return textwrap.dedent(block)
        except SyntaxError:
            continue
    return None


def function_is_defined(code: str, name: str) -> bool:
    """True iff `code` contains a top-level `def name(...)`."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return True
    return False


def function_signature(code: str, name: str) -> list[str] | None:
    """Return positional arg names of the first top-level `def name(...)`."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return [a.arg for a in node.args.args]
    return None


@dataclass
class RunResult:
    passed: bool
    n_passed: int = 0
    n_total: int = 0
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    syntax_error: bool = False
    composite_score: float = 0.0
    per_test: dict = field(default_factory=dict)


def run_sandboxed(module_src: str, test_src: str,
                  timeout_s: int = 15, mem_mb: int = 512) -> RunResult:
    """Run `test_src` against `module_src` in a subprocess with limits.

    `test_src` should import from `mod` (we name the file mod.py) and emit
    a JSON object to stdout of the form {"test_id": bool, ...}.
    """
    # Quick sanity: does the module parse?
    try:
        ast.parse(module_src)
    except SyntaxError as e:
        return RunResult(passed=False, syntax_error=True,
                          stderr=f"SyntaxError: {e}")

    def _set_limits():
        # Applied to the subprocess on POSIX
        try:
            resource.setrlimit(resource.RLIMIT_CPU,
                               (timeout_s, timeout_s))
            resource.setrlimit(resource.RLIMIT_AS,
                               (mem_mb * 1024 * 1024,
                                mem_mb * 1024 * 1024))
        except (ValueError, OSError):
            # Some systems don't support; continue anyway.
            pass

    with tempfile.TemporaryDirectory() as tmp:
        mod_path = os.path.join(tmp, "mod.py")
        test_path = os.path.join(tmp, "runtest.py")
        with open(mod_path, "w") as f:
            f.write(module_src)
        with open(test_path, "w") as f:
            f.write(test_src)
        try:
            proc = subprocess.run(
                ["python3", "runtest.py"],
                cwd=tmp,
                capture_output=True,
                text=True,
                timeout=timeout_s + 2,
                preexec_fn=_set_limits,
            )
        except subprocess.TimeoutExpired as e:
            return RunResult(passed=False, timed_out=True,
                              stderr=f"timed out after {timeout_s}s",
                              stdout=(e.stdout or "") if e.stdout else "")

    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()
    # Parse the last JSON object on stdout
    per_test = {}
    try:
        # Look for a line that starts with { and parses
        for line in stdout.splitlines()[::-1]:
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                per_test = json.loads(line)
                break
    except json.JSONDecodeError:
        pass
    n_total = len(per_test)
    n_passed = sum(1 for v in per_test.values() if v is True)
    return RunResult(
        passed=(n_passed == n_total and n_total > 0),
        n_passed=n_passed,
        n_total=n_total,
        stdout=stdout,
        stderr=stderr,
        per_test=per_test,
    )
