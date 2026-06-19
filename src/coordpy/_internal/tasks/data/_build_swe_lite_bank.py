"""Build the Phase-41 larger SWE-bench-Lite-style bank.

The Phase-40 ``swe_real_shape_mini.jsonl`` ships six self-authored
SWE-bench-shape instances. Phase 41 grows that to
**~24 instances** covering a broader, disciplined shape spectrum:

  * single-hunk substitutions (Phase-40 style) — baseline class;
  * **multi-hunk, single-file** — exercises ``apply_patch``'s
    left-to-right semantics on two independent edits;
  * **multi-function, single-file** — the patch touches one
    function but the file also contains other functions whose
    text the substrate-vs-naive attribution study uses as
    distractor bytes;
  * **return-value semantics** bugs, **off-by-one** bugs,
    **operator-typo** bugs, **wrong-branch** bugs, **missing-edge-
    case** bugs, **mutation-vs-copy** bugs, **ordering** bugs,
    **unicode / whitespace** edge cases;
  * deliberately short ``old`` anchors on some instances to
    stress byte-strict matching (the permissiveness-study
    surface);
  * long ``old`` anchors on others (to give naive a sizable raw-
    text handle the substrate deliberately withholds).

Reproducibility precondition: every instance's ``patch`` parses
cleanly with ``parse_unified_diff``, every ``old`` block in the
parsed substitutions appears *exactly once* in the inline
``repo_files`` source, and every ``test_source`` asserts the
behaviour the buggy file currently fails. We run each instance
through the oracle round-trip before writing the JSONL so the
artifact is a known-good corpus.

The bank is intentionally *self-authored*: it ships in-repo so
the full Phase-41 evaluation runs offline in seconds. Pointing
the Phase-41 driver at a real SWE-bench Lite JSONL is a
``--jsonl <path>`` parameter change — the loader, sandbox, and
substrate are unchanged.
"""

from __future__ import annotations

import io
import json
import os
import sys
from contextlib import redirect_stdout, redirect_stderr

# Self-contained so this script can be run directly.
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)

from coordpy._internal.tasks.swe_bench_bridge import parse_unified_diff, apply_patch


def _mk_diff(relpath: str, hunks: list[tuple[int, list[str]]]) -> str:
    """Compose a unified diff for ``relpath`` from a list of
    ``(start_line, hunk_lines)`` hunks.

    Each ``hunk_lines`` is a list of lines already prefixed with
    ``" "`` / ``"-"`` / ``"+"``. The function emits the correct
    ``@@`` headers so ``parse_unified_diff`` can parse the diff.
    """
    out = [f"--- a/{relpath}", f"+++ b/{relpath}"]
    for start, lines in hunks:
        old_len = sum(1 for ln in lines if ln.startswith((" ", "-")))
        new_len = sum(1 for ln in lines if ln.startswith((" ", "+")))
        out.append(f"@@ -{start},{old_len} +{start},{new_len} @@")
        out.extend(lines)
    return "\n".join(out) + "\n"


def _run_and_assert(instance_id: str, buggy_src: str, patch: str,
                     relpath: str, test_source: str) -> None:
    """Round-trip a single instance: parse diff, apply, run test.
    Raise if the patched source fails the hidden test.
    """
    parsed = parse_unified_diff(patch)
    if relpath not in parsed:
        raise ValueError(
            f"[{instance_id}] diff does not touch {relpath}; "
            f"covers {sorted(parsed)}")
    subs = parsed[relpath]
    # Require unique-match on every hunk.
    for i, (old, _new) in enumerate(subs):
        count = buggy_src.count(old)
        if count != 1:
            raise ValueError(
                f"[{instance_id}] hunk {i} old block does not "
                f"appear exactly once (count={count}) in source:\n"
                f"---OLD---\n{old}\n---SRC---\n{buggy_src}\n---")
    patched, ok, reason = apply_patch(buggy_src, subs)
    if not ok:
        raise ValueError(f"[{instance_id}] apply_patch failed: {reason}")
    mod_globals = {"__name__": "patched", "__builtins__": __builtins__}
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        exec(compile(patched, relpath, "exec"), mod_globals)
    tg = {"__name__": "test", "__builtins__": __builtins__}
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        exec(compile(test_source, "test", "exec"), tg)

    class _Mod:
        pass
    m = _Mod()
    for k, v in mod_globals.items():
        if not k.startswith("__"):
            setattr(m, k, v)
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        tg["test"](m)


INSTANCES: list[dict] = []


def _register(instance_id: str, repo: str, problem: str,
              relpath: str, buggy_function: str, buggy_src: str,
              patch: str, test_source: str) -> None:
    _run_and_assert(instance_id, buggy_src, patch, relpath, test_source)
    INSTANCES.append({
        "instance_id": instance_id,
        "repo": repo,
        "base_commit": "v0",
        "problem_statement": problem,
        "buggy_file_relpath": relpath,
        "buggy_function": buggy_function,
        "patch": patch,
        "test_source": test_source,
        "repo_files": {relpath: buggy_src},
    })


# ============================================================
# Instance set. Each tuple is a self-contained synthesized
# SWE-bench-Lite-style instance.
# ============================================================


# ---------- Phase-40 carry-over class (kept as-is) ----------

_register(
    "ext-calc-001", "external/calc",
    "`factorial(n)` returns 0 for every n -- the seed value is wrong. "
    "Expected: factorial(0)=1, factorial(5)=120.",
    "calc.py", "factorial",
    '"""Mini calculator module."""\n'
    '\n'
    'def add(a, b):\n'
    '    return a + b\n'
    '\n'
    'def sub(a, b):\n'
    '    return a - b\n'
    '\n'
    'def factorial(n):\n'
    '    """Return n!. BUG: returns 1 for n=0 incorrectly via\n'
    '    range(1, n+1) starting from 1 -- that part is fine --\n'
    '    but seeds with 0 instead of 1 so every result is 0."""\n'
    '    result = 0\n'
    '    for i in range(1, n + 1):\n'
    '        result *= i\n'
    '    return result\n'
    '\n'
    'def is_prime(n):\n'
    '    if n < 2:\n'
    '        return False\n'
    '    for i in range(2, int(n ** 0.5) + 1):\n'
    '        if n % i == 0:\n'
    '            return False\n'
    '    return True\n',
    "--- a/calc.py\n"
    "+++ b/calc.py\n"
    "@@ -11,3 +11,3 @@ def factorial(n):\n"
    "     but seeds with 0 instead of 1 so every result is 0.\"\"\"\n"
    "-    result = 0\n"
    "+    result = 1\n"
    "     for i in range(1, n + 1):\n",
    "def test(module):\n"
    "    assert module.add(2, 3) == 5\n"
    "    assert module.factorial(0) == 1\n"
    "    assert module.factorial(5) == 120\n"
    "    assert module.is_prime(2) is True\n"
    "    assert module.is_prime(9) is False\n",
)


_register(
    "ext-strings-001", "external/strings",
    "`title_case(\"hello world\")` returns `'hello world'` instead of "
    "`'Hello World'`. Each word's first letter must be upper-cased.",
    "strings.py", "title_case",
    '"""Mini string helper module."""\n'
    '\n'
    'def reverse(s):\n'
    '    return s[::-1]\n'
    '\n'
    'def title_case(s):\n'
    '    """Capitalise every word. BUG: lower-cases the first\n'
    '    letter of each word instead of upper-casing it.\n'
    '    """\n'
    '    return " ".join(w[0].lower() + w[1:].lower()\n'
    '                    for w in s.split() if w)\n'
    '\n'
    'def count_vowels(s):\n'
    '    return sum(1 for c in s.lower() if c in "aeiou")\n',
    "--- a/strings.py\n"
    "+++ b/strings.py\n"
    "@@ -9,2 +9,2 @@ def title_case(s):\n"
    "     \"\"\"\n"
    "-    return \" \".join(w[0].lower() + w[1:].lower()\n"
    "+    return \" \".join(w[0].upper() + w[1:].lower()\n"
    "                     for w in s.split() if w)\n",
    "def test(module):\n"
    "    assert module.reverse('abc') == 'cba'\n"
    "    assert module.title_case('hello world') == 'Hello World'\n"
    "    assert module.title_case('the quick brown fox') == 'The Quick Brown Fox'\n"
    "    assert module.count_vowels('Hello') == 2\n",
)


_register(
    "ext-list-001", "external/list",
    "`last([10, 20, 30])` returns 20 but should return 30. The function "
    "returns the wrong index when the list has at least 2 items.",
    "listops.py", "last",
    '"""Mini list operations module."""\n'
    '\n'
    'def first(items):\n'
    '    return items[0]\n'
    '\n'
    'def last(items):\n'
    '    """Return the last item. BUG: off-by-one -- returns the\n'
    '    second-to-last when the list has >= 2 items."""\n'
    '    if not items:\n'
    '        raise IndexError("empty list")\n'
    '    if len(items) == 1:\n'
    '        return items[0]\n'
    '    return items[-2]\n'
    '\n'
    'def unique(items):\n'
    '    seen = set()\n'
    '    out = []\n'
    '    for x in items:\n'
    '        if x not in seen:\n'
    '            seen.add(x)\n'
    '            out.append(x)\n'
    '    return out\n',
    "--- a/listops.py\n"
    "+++ b/listops.py\n"
    "@@ -12,1 +12,1 @@ def last(items):\n"
    "-    return items[-2]\n"
    "+    return items[-1]\n",
    "def test(module):\n"
    "    assert module.first([10, 20, 30]) == 10\n"
    "    assert module.last([10, 20, 30]) == 30\n"
    "    assert module.last([42]) == 42\n"
    "    assert module.unique([1, 2, 2, 3, 1]) == [1, 2, 3]\n",
)


_register(
    "ext-dict-001", "external/dict",
    "`merge(a, b)` mutates `a` in place; callers expect `a` unchanged "
    "after the call. Return a new dict.",
    "dictops.py", "merge",
    '"""Mini dict helpers."""\n'
    '\n'
    'def merge(a, b):\n'
    '    """Return a new dict with a\'s keys overridden by b\'s.\n'
    '    BUG: returns ``a`` directly mutated instead of a copy.\n'
    '    """\n'
    '    a.update(b)\n'
    '    return a\n'
    '\n'
    'def keys_sorted(d):\n'
    '    return sorted(d.keys())\n'
    '\n'
    'def value_count(d, value):\n'
    '    return sum(1 for v in d.values() if v == value)\n',
    "--- a/dictops.py\n"
    "+++ b/dictops.py\n"
    "@@ -6,2 +6,3 @@ def merge(a, b):\n"
    "     \"\"\"\n"
    "-    a.update(b)\n"
    "-    return a\n"
    "+    out = dict(a)\n"
    "+    out.update(b)\n"
    "+    return out\n",
    "def test(module):\n"
    "    a = {'x': 1, 'y': 2}\n"
    "    b = {'y': 20, 'z': 30}\n"
    "    merged = module.merge(a, b)\n"
    "    assert merged == {'x': 1, 'y': 20, 'z': 30}\n"
    "    assert a == {'x': 1, 'y': 2}\n"
    "    assert module.keys_sorted({'b': 1, 'a': 2}) == ['a', 'b']\n",
)


_register(
    "ext-text-001", "external/text",
    "`word_count(s)` returns one fewer than the number of whitespace-"
    "separated tokens. Expected: word_count('a b c') == 3.",
    "text.py", "word_count",
    '"""Mini text helpers."""\n'
    '\n'
    'def word_count(s):\n'
    '    """Count words. BUG: subtracts 1 unnecessarily."""\n'
    '    return len(s.split()) - 1\n'
    '\n'
    'def strip_punct(s):\n'
    '    return "".join(c for c in s if c.isalnum() or c.isspace())\n',
    "--- a/text.py\n"
    "+++ b/text.py\n"
    "@@ -4,2 +4,2 @@ def word_count(s):\n"
    "     \"\"\"Count words. BUG: subtracts 1 unnecessarily.\"\"\"\n"
    "-    return len(s.split()) - 1\n"
    "+    return len(s.split())\n",
    "def test(module):\n"
    "    assert module.word_count('a b c') == 3\n"
    "    assert module.word_count('hello world') == 2\n"
    "    assert module.word_count('') == 0\n"
    "    assert module.word_count('one') == 1\n",
)


_register(
    "ext-math-001", "external/math",
    "`average(xs)` returns the median, not the arithmetic mean. "
    "average([1, 2, 3, 4]) should be 2.5.",
    "stats.py", "average",
    '"""Mini stats helpers."""\n'
    '\n'
    'def average(xs):\n'
    '    """BUG: returns median instead of mean."""\n'
    '    return sorted(xs)[len(xs) // 2]\n'
    '\n'
    'def variance(xs):\n'
    '    m = sum(xs) / len(xs)\n'
    '    return sum((x - m) ** 2 for x in xs) / len(xs)\n',
    "--- a/stats.py\n"
    "+++ b/stats.py\n"
    "@@ -4,2 +4,2 @@ def average(xs):\n"
    "     \"\"\"BUG: returns median instead of mean.\"\"\"\n"
    "-    return sorted(xs)[len(xs) // 2]\n"
    "+    return sum(xs) / len(xs)\n",
    "def test(module):\n"
    "    assert module.average([1, 2, 3, 4]) == 2.5\n"
    "    assert module.average([10]) == 10\n"
    "    assert module.average([2, 4, 6]) == 4\n",
)


# ---------- Phase-41 expansion: operator-typo / comparator bugs ----------

_register(
    "ext-cmp-001", "external/cmp",
    "`max_of(a, b)` returns the smaller argument instead of the larger. "
    "max_of(3, 7) should be 7.",
    "cmp.py", "max_of",
    '"""Min/max helpers."""\n'
    '\n'
    'def min_of(a, b):\n'
    '    if a < b:\n'
    '        return a\n'
    '    return b\n'
    '\n'
    'def max_of(a, b):\n'
    '    """Return the larger of a and b. BUG: comparator flipped."""\n'
    '    if a < b:\n'
    '        return a\n'
    '    return b\n'
    '\n'
    'def clamp(x, lo, hi):\n'
    '    if x < lo:\n'
    '        return lo\n'
    '    if x > hi:\n'
    '        return hi\n'
    '    return x\n',
    "--- a/cmp.py\n"
    "+++ b/cmp.py\n"
    "@@ -10,4 +10,4 @@ def max_of(a, b):\n"
    "     \"\"\"Return the larger of a and b. BUG: comparator flipped.\"\"\"\n"
    "-    if a < b:\n"
    "-        return a\n"
    "-    return b\n"
    "+    if a > b:\n"
    "+        return a\n"
    "+    return b\n",
    "def test(module):\n"
    "    assert module.max_of(3, 7) == 7\n"
    "    assert module.max_of(10, 5) == 10\n"
    "    assert module.min_of(3, 7) == 3\n"
    "    assert module.clamp(5, 0, 10) == 5\n"
    "    assert module.clamp(-1, 0, 10) == 0\n",
)


_register(
    "ext-range-001", "external/range",
    "`inclusive_range(lo, hi)` stops one short of `hi`. "
    "inclusive_range(1, 3) should return [1, 2, 3].",
    "range_utils.py", "inclusive_range",
    '"""Inclusive range helpers."""\n'
    '\n'
    'def inclusive_range(lo, hi):\n'
    '    """Return list of ints from lo through hi.\n'
    '    BUG: off-by-one, stops before hi.\n'
    '    """\n'
    '    return list(range(lo, hi))\n'
    '\n'
    'def step_range(lo, hi, step):\n'
    '    return list(range(lo, hi, step))\n',
    "--- a/range_utils.py\n"
    "+++ b/range_utils.py\n"
    "@@ -7,1 +7,1 @@ def inclusive_range(lo, hi):\n"
    "-    return list(range(lo, hi))\n"
    "+    return list(range(lo, hi + 1))\n",
    "def test(module):\n"
    "    assert module.inclusive_range(1, 3) == [1, 2, 3]\n"
    "    assert module.inclusive_range(0, 0) == [0]\n"
    "    assert module.step_range(0, 10, 2) == [0, 2, 4, 6, 8]\n",
)


_register(
    "ext-guard-001", "external/guard",
    "`safe_div(a, b)` raises ZeroDivisionError when b == 0 instead of "
    "returning 0. safe_div(5, 0) should be 0.",
    "guard.py", "safe_div",
    '"""Safe arithmetic."""\n'
    '\n'
    'def safe_div(a, b):\n'
    '    """Return a / b, 0 when b == 0.\n'
    '    BUG: missing zero-guard.\n'
    '    """\n'
    '    return a / b\n'
    '\n'
    'def safe_mod(a, b):\n'
    '    if b == 0:\n'
    '        return 0\n'
    '    return a % b\n',
    "--- a/guard.py\n"
    "+++ b/guard.py\n"
    "@@ -7,1 +7,3 @@ def safe_div(a, b):\n"
    "-    return a / b\n"
    "+    if b == 0:\n"
    "+        return 0\n"
    "+    return a / b\n",
    "def test(module):\n"
    "    assert module.safe_div(10, 2) == 5\n"
    "    assert module.safe_div(5, 0) == 0\n"
    "    assert module.safe_mod(10, 3) == 1\n"
    "    assert module.safe_mod(10, 0) == 0\n",
)


_register(
    "ext-neg-001", "external/negate",
    "`negate(x)` returns the argument unchanged when it should return "
    "its negation.",
    "negate.py", "negate",
    '"""Negation helpers."""\n'
    '\n'
    'def is_even(n):\n'
    '    return n % 2 == 0\n'
    '\n'
    'def negate(x):\n'
    '    """Return -x. BUG: missing unary minus."""\n'
    '    return x\n',
    "--- a/negate.py\n"
    "+++ b/negate.py\n"
    "@@ -7,1 +7,1 @@ def negate(x):\n"
    "-    return x\n"
    "+    return -x\n",
    "def test(module):\n"
    "    assert module.negate(3) == -3\n"
    "    assert module.negate(-7) == 7\n"
    "    assert module.is_even(4) is True\n"
    "    assert module.is_even(5) is False\n",
)


# ---------- Phase-41 expansion: missing-edge-case bugs ----------

_register(
    "ext-empty-001", "external/empty",
    "`first_or_none(seq)` raises IndexError on empty input. Should "
    "return None when the sequence is empty.",
    "first.py", "first_or_none",
    '"""First-element helpers."""\n'
    '\n'
    'def first_or_none(seq):\n'
    '    """Return seq[0] or None if empty.\n'
    '    BUG: missing empty-check.\n'
    '    """\n'
    '    return seq[0]\n'
    '\n'
    'def second_or_none(seq):\n'
    '    if len(seq) < 2:\n'
    '        return None\n'
    '    return seq[1]\n',
    "--- a/first.py\n"
    "+++ b/first.py\n"
    "@@ -7,1 +7,3 @@ def first_or_none(seq):\n"
    "-    return seq[0]\n"
    "+    if not seq:\n"
    "+        return None\n"
    "+    return seq[0]\n",
    "def test(module):\n"
    "    assert module.first_or_none([1, 2, 3]) == 1\n"
    "    assert module.first_or_none([]) is None\n"
    "    assert module.second_or_none([1, 2]) == 2\n"
    "    assert module.second_or_none([1]) is None\n",
)


_register(
    "ext-strip-001", "external/strip",
    "`clean(s)` fails on None input. Should return empty string when "
    "given None.",
    "clean.py", "clean",
    '"""String cleaning."""\n'
    '\n'
    'def clean(s):\n'
    '    """Strip and lowercase. BUG: crashes on None."""\n'
    '    return s.strip().lower()\n'
    '\n'
    'def shout(s):\n'
    '    return s.upper() + "!"\n',
    "--- a/clean.py\n"
    "+++ b/clean.py\n"
    "@@ -4,1 +4,3 @@ def clean(s):\n"
    "-    return s.strip().lower()\n"
    "+    if s is None:\n"
    "+        return \"\"\n"
    "+    return s.strip().lower()\n",
    "def test(module):\n"
    "    assert module.clean('  HELLO ') == 'hello'\n"
    "    assert module.clean(None) == ''\n"
    "    assert module.shout('hi') == 'HI!'\n",
)


# ---------- Phase-41 expansion: wrong-branch bugs ----------

_register(
    "ext-branch-001", "external/branch",
    "`absolute(x)` returns -x for positive inputs. Should return x for "
    "non-negative inputs, -x for negative.",
    "absval.py", "absolute",
    '"""Absolute value helpers."""\n'
    '\n'
    'def absolute(x):\n'
    '    """Return |x|. BUG: branches are swapped."""\n'
    '    if x < 0:\n'
    '        return x\n'
    '    return -x\n'
    '\n'
    'def sign(x):\n'
    '    if x > 0:\n'
    '        return 1\n'
    '    if x < 0:\n'
    '        return -1\n'
    '    return 0\n',
    "--- a/absval.py\n"
    "+++ b/absval.py\n"
    "@@ -5,3 +5,3 @@ def absolute(x):\n"
    "     \"\"\"Return |x|. BUG: branches are swapped.\"\"\"\n"
    "-    if x < 0:\n"
    "-        return x\n"
    "-    return -x\n"
    "+    if x < 0:\n"
    "+        return -x\n"
    "+    return x\n",
    "def test(module):\n"
    "    assert module.absolute(5) == 5\n"
    "    assert module.absolute(-3) == 3\n"
    "    assert module.absolute(0) == 0\n"
    "    assert module.sign(-4) == -1\n"
    "    assert module.sign(7) == 1\n"
    "    assert module.sign(0) == 0\n",
)


# ---------- Phase-41 expansion: mutation-vs-copy ----------

_register(
    "ext-mut-001", "external/mut",
    "`append_one(xs)` mutates the caller's list. Should return a new "
    "list with 1 appended.",
    "mutops.py", "append_one",
    '"""List mutation helpers."""\n'
    '\n'
    'def append_one(xs):\n'
    '    """Return a new list with 1 appended.\n'
    '    BUG: mutates the input instead of copying.\n'
    '    """\n'
    '    xs.append(1)\n'
    '    return xs\n'
    '\n'
    'def head(xs):\n'
    '    return xs[0] if xs else None\n',
    "--- a/mutops.py\n"
    "+++ b/mutops.py\n"
    "@@ -7,2 +7,3 @@ def append_one(xs):\n"
    "-    xs.append(1)\n"
    "-    return xs\n"
    "+    out = list(xs)\n"
    "+    out.append(1)\n"
    "+    return out\n",
    "def test(module):\n"
    "    xs = [10, 20]\n"
    "    out = module.append_one(xs)\n"
    "    assert out == [10, 20, 1]\n"
    "    assert xs == [10, 20]\n"
    "    assert module.head([5]) == 5\n",
)


# ---------- Phase-41 expansion: multi-hunk patch ----------

_register(
    "ext-multi-001", "external/multi",
    "The counter module's increment mis-increments by 2 and the reset "
    "sets the counter to 1 instead of 0. Both bugs must be fixed.",
    "counter.py", "Counter",
    '"""Counter with bugs in two separate methods."""\n'
    '\n'
    'class Counter:\n'
    '    def __init__(self):\n'
    '        self.value = 0\n'
    '\n'
    '    def increment(self):\n'
    '        """BUG: increments by 2 instead of 1."""\n'
    '        self.value += 2\n'
    '\n'
    '    def decrement(self):\n'
    '        self.value -= 1\n'
    '\n'
    '    def reset(self):\n'
    '        """BUG: resets to 1 instead of 0."""\n'
    '        self.value = 1\n'
    '\n'
    '    def get(self):\n'
    '        return self.value\n',
    "--- a/counter.py\n"
    "+++ b/counter.py\n"
    "@@ -7,2 +7,2 @@ class Counter:\n"
    "     def increment(self):\n"
    "         \"\"\"BUG: increments by 2 instead of 1.\"\"\"\n"
    "-        self.value += 2\n"
    "+        self.value += 1\n"
    "@@ -14,2 +14,2 @@ class Counter:\n"
    "     def reset(self):\n"
    "         \"\"\"BUG: resets to 1 instead of 0.\"\"\"\n"
    "-        self.value = 1\n"
    "+        self.value = 0\n",
    "def test(module):\n"
    "    c = module.Counter()\n"
    "    c.increment()\n"
    "    assert c.get() == 1\n"
    "    c.increment(); c.increment()\n"
    "    assert c.get() == 3\n"
    "    c.decrement()\n"
    "    assert c.get() == 2\n"
    "    c.reset()\n"
    "    assert c.get() == 0\n",
)


# ---------- Phase-41 expansion: ordering bugs ----------

_register(
    "ext-order-001", "external/order",
    "`rank_desc(xs)` returns the ascending order. Should sort descending "
    "so the first element is the largest.",
    "ranking.py", "rank_desc",
    '"""Ranking helpers."""\n'
    '\n'
    'def rank_asc(xs):\n'
    '    return sorted(xs)\n'
    '\n'
    'def rank_desc(xs):\n'
    '    """BUG: sorts ascending instead of descending."""\n'
    '    return sorted(xs)\n'
    '\n'
    'def top_k(xs, k):\n'
    '    return sorted(xs, reverse=True)[:k]\n',
    "--- a/ranking.py\n"
    "+++ b/ranking.py\n"
    "@@ -7,2 +7,2 @@ def rank_desc(xs):\n"
    "     \"\"\"BUG: sorts ascending instead of descending.\"\"\"\n"
    "-    return sorted(xs)\n"
    "+    return sorted(xs, reverse=True)\n",
    "def test(module):\n"
    "    assert module.rank_asc([3, 1, 2]) == [1, 2, 3]\n"
    "    assert module.rank_desc([3, 1, 2]) == [3, 2, 1]\n"
    "    assert module.top_k([1, 5, 2, 4, 3], 2) == [5, 4]\n",
)


# ---------- Phase-41 expansion: unicode / whitespace ----------

_register(
    "ext-uni-001", "external/unicode",
    "`is_ascii(s)` returns True on non-ASCII input. Should return False.",
    "asciiops.py", "is_ascii",
    '"""ASCII helpers."""\n'
    '\n'
    'def is_ascii(s):\n'
    '    """Return whether every char is ASCII.\n'
    '    BUG: returns True always.\n'
    '    """\n'
    '    return True\n'
    '\n'
    'def strip_non_ascii(s):\n'
    '    return "".join(c for c in s if ord(c) < 128)\n',
    "--- a/asciiops.py\n"
    "+++ b/asciiops.py\n"
    "@@ -7,1 +7,1 @@ def is_ascii(s):\n"
    "-    return True\n"
    "+    return all(ord(c) < 128 for c in s)\n",
    "def test(module):\n"
    "    assert module.is_ascii('hello') is True\n"
    "    assert module.is_ascii('h\\u00e9llo') is False\n"
    "    assert module.strip_non_ascii('h\\u00e9llo') == 'hllo'\n",
)


# ---------- Phase-41 expansion: conditional polarity ----------

_register(
    "ext-bool-001", "external/bool",
    "`is_nonempty(seq)` returns False for non-empty sequences. Should "
    "return True for anything with at least one element.",
    "boolops.py", "is_nonempty",
    '"""Boolean helpers."""\n'
    '\n'
    'def is_empty(seq):\n'
    '    return len(seq) == 0\n'
    '\n'
    'def is_nonempty(seq):\n'
    '    """BUG: negation swapped; returns False for non-empty."""\n'
    '    return len(seq) == 0\n',
    "--- a/boolops.py\n"
    "+++ b/boolops.py\n"
    "@@ -6,2 +6,2 @@ def is_nonempty(seq):\n"
    "     \"\"\"BUG: negation swapped; returns False for non-empty.\"\"\"\n"
    "-    return len(seq) == 0\n"
    "+    return len(seq) > 0\n",
    "def test(module):\n"
    "    assert module.is_empty([]) is True\n"
    "    assert module.is_empty([1]) is False\n"
    "    assert module.is_nonempty([1]) is True\n"
    "    assert module.is_nonempty([]) is False\n",
)


# ---------- Phase-41 expansion: aggregation bugs ----------

_register(
    "ext-agg-001", "external/agg",
    "`product(xs)` returns 0 for every input because it seeds with 0 "
    "instead of 1. product([2, 3, 4]) should be 24.",
    "agg.py", "product",
    '"""Aggregation helpers."""\n'
    '\n'
    'def summ(xs):\n'
    '    return sum(xs)\n'
    '\n'
    'def product(xs):\n'
    '    """BUG: accumulator seeded with 0."""\n'
    '    p = 0\n'
    '    for x in xs:\n'
    '        p *= x\n'
    '    return p\n'
    '\n'
    'def count_truthy(xs):\n'
    '    return sum(1 for x in xs if x)\n',
    "--- a/agg.py\n"
    "+++ b/agg.py\n"
    "@@ -7,1 +7,1 @@ def product(xs):\n"
    "-    p = 0\n"
    "+    p = 1\n",
    "def test(module):\n"
    "    assert module.summ([1, 2, 3]) == 6\n"
    "    assert module.product([2, 3, 4]) == 24\n"
    "    assert module.product([5]) == 5\n"
    "    assert module.count_truthy([0, 1, '', 'x', None]) == 2\n",
)


# ---------- Phase-41 expansion: string slice bugs ----------

_register(
    "ext-slice-001", "external/slice",
    "`first_n(s, n)` returns the last n characters instead of the first "
    "n. Expected: first_n('hello', 2) == 'he'.",
    "slicing.py", "first_n",
    '"""Slicing helpers."""\n'
    '\n'
    'def first_n(s, n):\n'
    '    """BUG: slices the tail instead of the head."""\n'
    '    return s[-n:]\n'
    '\n'
    'def last_n(s, n):\n'
    '    return s[-n:] if n else ""\n',
    "--- a/slicing.py\n"
    "+++ b/slicing.py\n"
    "@@ -5,1 +5,1 @@ def first_n(s, n):\n"
    "-    return s[-n:]\n"
    "+    return s[:n]\n",
    "def test(module):\n"
    "    assert module.first_n('hello', 2) == 'he'\n"
    "    assert module.first_n('abc', 0) == ''\n"
    "    assert module.last_n('hello', 2) == 'lo'\n",
)


# ---------- Phase-41 expansion: regex escape ----------

_register(
    "ext-path-001", "external/path",
    "`basename(p)` returns the directory portion instead of the file "
    "basename. basename('a/b/c.txt') should be 'c.txt'.",
    "paths.py", "basename",
    '"""Minimal path helpers."""\n'
    '\n'
    'def basename(p):\n'
    '    """Return the filename portion of a path.\n'
    '    BUG: returns the directory instead of the basename.\n'
    '    """\n'
    '    return p.rsplit("/", 1)[0]\n'
    '\n'
    'def dirname(p):\n'
    '    if "/" not in p:\n'
    '        return ""\n'
    '    return p.rsplit("/", 1)[0]\n',
    "--- a/paths.py\n"
    "+++ b/paths.py\n"
    "@@ -6,2 +6,2 @@ def basename(p):\n"
    "     \"\"\"\n"
    "-    return p.rsplit(\"/\", 1)[0]\n"
    "+    return p.rsplit(\"/\", 1)[-1]\n",
    "def test(module):\n"
    "    assert module.basename('a/b/c.txt') == 'c.txt'\n"
    "    assert module.basename('x.txt') == 'x.txt'\n"
    "    assert module.dirname('a/b/c.txt') == 'a/b'\n"
    "    assert module.dirname('x.txt') == ''\n",
)


# ---------- Phase-41 expansion: return-type bug ----------

_register(
    "ext-type-001", "external/type",
    "`to_bool(s)` returns the string unchanged. Should return Python "
    "True/False based on 'true'/'false' text (case-insensitive).",
    "tobool.py", "to_bool",
    '"""Type conversion helpers."""\n'
    '\n'
    'def to_bool(s):\n'
    '    """Return True/False from a string.\n'
    '    BUG: returns the input unchanged.\n'
    '    """\n'
    '    return s\n'
    '\n'
    'def to_int(s, default=0):\n'
    '    try:\n'
    '        return int(s)\n'
    '    except ValueError:\n'
    '        return default\n',
    "--- a/tobool.py\n"
    "+++ b/tobool.py\n"
    "@@ -7,1 +7,1 @@ def to_bool(s):\n"
    "-    return s\n"
    "+    return s.strip().lower() == \"true\"\n",
    "def test(module):\n"
    "    assert module.to_bool('true') is True\n"
    "    assert module.to_bool('False') is False\n"
    "    assert module.to_bool(' TRUE ') is True\n"
    "    assert module.to_int('42') == 42\n"
    "    assert module.to_int('x', 9) == 9\n",
)


# ---------- Phase-41 expansion: equality operator ----------

_register(
    "ext-eq-001", "external/eq",
    "`same_set(a, b)` uses order-sensitive equality. Should return True "
    "if two iterables contain the same elements regardless of order.",
    "setops.py", "same_set",
    '"""Set helpers."""\n'
    '\n'
    'def same_set(a, b):\n'
    '    """Return True iff iterables have same elements (any order).\n'
    '    BUG: compares order-sensitive lists.\n'
    '    """\n'
    '    return list(a) == list(b)\n'
    '\n'
    'def disjoint(a, b):\n'
    '    return not (set(a) & set(b))\n',
    "--- a/setops.py\n"
    "+++ b/setops.py\n"
    "@@ -7,1 +7,1 @@ def same_set(a, b):\n"
    "-    return list(a) == list(b)\n"
    "+    return set(a) == set(b)\n",
    "def test(module):\n"
    "    assert module.same_set([1, 2, 3], [3, 2, 1]) is True\n"
    "    assert module.same_set([1, 2], [1, 2, 3]) is False\n"
    "    assert module.disjoint([1, 2], [3, 4]) is True\n"
    "    assert module.disjoint([1, 2], [2, 3]) is False\n",
)


# ---------- Phase-41 expansion: counter / histogram ----------

_register(
    "ext-hist-001", "external/hist",
    "`histogram(xs)` returns empty dict because it never writes to the "
    "accumulator. Should return a {value: count} dict.",
    "hist.py", "histogram",
    '"""Histogram helpers."""\n'
    '\n'
    'def histogram(xs):\n'
    '    """Return {value: count}.\n'
    '    BUG: loop body does not update the accumulator.\n'
    '    """\n'
    '    out = {}\n'
    '    for x in xs:\n'
    '        pass\n'
    '    return out\n',
    "--- a/hist.py\n"
    "+++ b/hist.py\n"
    "@@ -7,2 +7,2 @@ def histogram(xs):\n"
    "     for x in xs:\n"
    "-        pass\n"
    "+        out[x] = out.get(x, 0) + 1\n",
    "def test(module):\n"
    "    assert module.histogram([1, 2, 1]) == {1: 2, 2: 1}\n"
    "    assert module.histogram([]) == {}\n"
    "    assert module.histogram(['a']) == {'a': 1}\n",
)


# ---------- Phase-41 expansion: boolean composition ----------

_register(
    "ext-allany-001", "external/allany",
    "`all_positive(xs)` returns True on a list containing a negative "
    "number. Should return True iff every element is > 0.",
    "allany.py", "all_positive",
    '"""All/any helpers."""\n'
    '\n'
    'def all_positive(xs):\n'
    '    """BUG: uses `any` instead of `all`."""\n'
    '    return any(x > 0 for x in xs)\n'
    '\n'
    'def any_negative(xs):\n'
    '    return any(x < 0 for x in xs)\n',
    "--- a/allany.py\n"
    "+++ b/allany.py\n"
    "@@ -5,1 +5,1 @@ def all_positive(xs):\n"
    "-    return any(x > 0 for x in xs)\n"
    "+    return all(x > 0 for x in xs)\n",
    "def test(module):\n"
    "    assert module.all_positive([1, 2, 3]) is True\n"
    "    assert module.all_positive([1, -2, 3]) is False\n"
    "    assert module.all_positive([]) is True\n"
    "    assert module.any_negative([1, 2, -3]) is True\n",
)


# ---------- Phase-41 expansion: bounded accumulator ----------

_register(
    "ext-cumsum-001", "external/cumsum",
    "`cumulative(xs)` overwrites the running total instead of "
    "appending successive partial sums.",
    "cumops.py", "cumulative",
    '"""Cumulative helpers."""\n'
    '\n'
    'def cumulative(xs):\n'
    '    """Return running sum prefix. BUG: stores the final scalar\n'
    '    instead of the list of prefix sums.\n'
    '    """\n'
    '    total = 0\n'
    '    out = []\n'
    '    for x in xs:\n'
    '        total += x\n'
    '        out = total\n'
    '    return out\n',
    "--- a/cumops.py\n"
    "+++ b/cumops.py\n"
    "@@ -10,1 +10,1 @@ def cumulative(xs):\n"
    "-        out = total\n"
    "+        out.append(total)\n",
    "def test(module):\n"
    "    assert module.cumulative([1, 2, 3]) == [1, 3, 6]\n"
    "    assert module.cumulative([]) == []\n"
    "    assert module.cumulative([5]) == [5]\n",
)


# ---------- Phase-41 expansion: key extraction ----------

_register(
    "ext-key-001", "external/key",
    "`group_by_parity(xs)` puts every element in the 'odd' bucket. "
    "Should route even values to 'even' and odd values to 'odd'.",
    "grouping.py", "group_by_parity",
    '"""Grouping helpers."""\n'
    '\n'
    'def group_by_parity(xs):\n'
    '    """BUG: parity test inverted / wrong bucket."""\n'
    '    out = {"even": [], "odd": []}\n'
    '    for x in xs:\n'
    '        out["odd"].append(x)\n'
    '    return out\n',
    "--- a/grouping.py\n"
    "+++ b/grouping.py\n"
    "@@ -6,1 +6,4 @@ def group_by_parity(xs):\n"
    "-        out[\"odd\"].append(x)\n"
    "+        if x % 2 == 0:\n"
    "+            out[\"even\"].append(x)\n"
    "+        else:\n"
    "+            out[\"odd\"].append(x)\n",
    "def test(module):\n"
    "    g = module.group_by_parity([1, 2, 3, 4])\n"
    "    assert g == {'even': [2, 4], 'odd': [1, 3]}\n"
    "    empty = module.group_by_parity([])\n"
    "    assert empty == {'even': [], 'odd': []}\n",
)


# ---------- Phase-41 expansion: bounded search ----------

_register(
    "ext-idx-001", "external/idx",
    "`find_index(xs, target)` returns 0 when the target is missing. "
    "Should return -1 when missing.",
    "finding.py", "find_index",
    '"""Find helpers."""\n'
    '\n'
    'def find_index(xs, target):\n'
    '    """Return index of target, -1 if missing.\n'
    '    BUG: returns 0 on miss instead of -1.\n'
    '    """\n'
    '    for i, x in enumerate(xs):\n'
    '        if x == target:\n'
    '            return i\n'
    '    return 0\n',
    "--- a/finding.py\n"
    "+++ b/finding.py\n"
    "@@ -9,1 +9,1 @@ def find_index(xs, target):\n"
    "-    return 0\n"
    "+    return -1\n",
    "def test(module):\n"
    "    assert module.find_index([10, 20, 30], 20) == 1\n"
    "    assert module.find_index([10, 20, 30], 99) == -1\n"
    "    assert module.find_index([], 5) == -1\n",
)


# =============================================================================
# Phase 42 expansion — push the bank past 50 instances so the
# communication-bounded conjecture (C41-1) can be empirically
# falsified with a cleanly above-threshold N.
# =============================================================================

# ---------- P42: string manipulation ----------

_register(
    "ext-pad-001", "external/pad",
    "`zpad(n, width)` pads with spaces when it should pad with zeros.",
    "pad.py", "zpad",
    '"""Zero-pad helpers."""\n'
    '\n'
    'def zpad(n, width):\n'
    '    """BUG: uses str.rjust with space fill, not zero fill."""\n'
    '    return str(n).rjust(width)\n'
    '\n'
    'def truncate(s, n):\n'
    '    return s[:n]\n',
    "--- a/pad.py\n"
    "+++ b/pad.py\n"
    "@@ -4,1 +4,1 @@ def zpad(n, width):\n"
    "-    return str(n).rjust(width)\n"
    "+    return str(n).rjust(width, '0')\n",
    "def test(module):\n"
    "    assert module.zpad(5, 3) == '005'\n"
    "    assert module.zpad(123, 3) == '123'\n"
    "    assert module.truncate('hello', 3) == 'hel'\n",
)


_register(
    "ext-join-001", "external/join",
    "`sep_join(parts, sep)` ignores the separator and concatenates directly. "
    "Should join using the given separator.",
    "joining.py", "sep_join",
    '"""Join helpers."""\n'
    '\n'
    'def sep_join(parts, sep):\n'
    '    """BUG: concatenates without sep."""\n'
    '    return "".join(parts)\n'
    '\n'
    'def csv_line(parts):\n'
    '    return ",".join(str(p) for p in parts)\n',
    "--- a/joining.py\n"
    "+++ b/joining.py\n"
    "@@ -4,1 +4,1 @@ def sep_join(parts, sep):\n"
    "-    return \"\".join(parts)\n"
    "+    return sep.join(parts)\n",
    "def test(module):\n"
    "    assert module.sep_join(['a', 'b', 'c'], '-') == 'a-b-c'\n"
    "    assert module.sep_join([], '|') == ''\n"
    "    assert module.csv_line([1, 2, 3]) == '1,2,3'\n",
)


_register(
    "ext-repl-001", "external/repl",
    "`double(s)` returns the input unchanged. Should return the input "
    "concatenated with itself.",
    "repl.py", "double",
    '"""Repl helpers."""\n'
    '\n'
    'def double(s):\n'
    '    """BUG: returns the original."""\n'
    '    return s\n'
    '\n'
    'def triple(s):\n'
    '    return s + s + s\n',
    "--- a/repl.py\n"
    "+++ b/repl.py\n"
    "@@ -4,1 +4,1 @@ def double(s):\n"
    "-    return s\n"
    "+    return s + s\n",
    "def test(module):\n"
    "    assert module.double('ab') == 'abab'\n"
    "    assert module.double('') == ''\n"
    "    assert module.triple('x') == 'xxx'\n",
)


# ---------- P42: numeric guards ----------

_register(
    "ext-abs-001", "external/abs",
    "`nonneg(x)` returns negative values unchanged. Should clamp negatives "
    "to zero.",
    "nonneg.py", "nonneg",
    '"""Non-negative clamp."""\n'
    '\n'
    'def nonneg(x):\n'
    '    """BUG: no guard — negatives pass through."""\n'
    '    return x\n'
    '\n'
    'def nonpos(x):\n'
    '    return x if x < 0 else 0\n',
    "--- a/nonneg.py\n"
    "+++ b/nonneg.py\n"
    "@@ -4,1 +4,3 @@ def nonneg(x):\n"
    "-    return x\n"
    "+    if x < 0:\n"
    "+        return 0\n"
    "+    return x\n",
    "def test(module):\n"
    "    assert module.nonneg(5) == 5\n"
    "    assert module.nonneg(-3) == 0\n"
    "    assert module.nonneg(0) == 0\n"
    "    assert module.nonpos(-4) == -4\n"
    "    assert module.nonpos(7) == 0\n",
)


_register(
    "ext-pow-001", "external/pow",
    "`square(x)` returns 2*x instead of x*x.",
    "powops.py", "square",
    '"""Power helpers."""\n'
    '\n'
    'def square(x):\n'
    '    """BUG: multiplies by 2 instead of squaring."""\n'
    '    return x * 2\n'
    '\n'
    'def cube(x):\n'
    '    return x * x * x\n',
    "--- a/powops.py\n"
    "+++ b/powops.py\n"
    "@@ -4,1 +4,1 @@ def square(x):\n"
    "-    return x * 2\n"
    "+    return x * x\n",
    "def test(module):\n"
    "    assert module.square(3) == 9\n"
    "    assert module.square(0) == 0\n"
    "    assert module.square(-4) == 16\n"
    "    assert module.cube(3) == 27\n",
)


_register(
    "ext-mod-001", "external/mod",
    "`wrap_index(i, n)` uses `i - n` when it should use `i % n`.",
    "wrap.py", "wrap_index",
    '"""Modular index helpers."""\n'
    '\n'
    'def wrap_index(i, n):\n'
    '    """BUG: subtracts n instead of mod n."""\n'
    '    return i - n\n'
    '\n'
    'def next_index(i, n):\n'
    '    return (i + 1) % n\n',
    "--- a/wrap.py\n"
    "+++ b/wrap.py\n"
    "@@ -4,1 +4,1 @@ def wrap_index(i, n):\n"
    "-    return i - n\n"
    "+    return i % n\n",
    "def test(module):\n"
    "    assert module.wrap_index(7, 3) == 1\n"
    "    assert module.wrap_index(0, 5) == 0\n"
    "    assert module.next_index(4, 5) == 0\n",
)


# ---------- P42: sequence construction ----------

_register(
    "ext-reverse-001", "external/reverse",
    "`reverse_list(xs)` returns a shallow copy without reversing.",
    "revlist.py", "reverse_list",
    '"""Reverse helpers."""\n'
    '\n'
    'def reverse_list(xs):\n'
    '    """BUG: does not actually reverse."""\n'
    '    return list(xs)\n'
    '\n'
    'def reverse_str(s):\n'
    '    return s[::-1]\n',
    "--- a/revlist.py\n"
    "+++ b/revlist.py\n"
    "@@ -4,1 +4,1 @@ def reverse_list(xs):\n"
    "-    return list(xs)\n"
    "+    return list(reversed(xs))\n",
    "def test(module):\n"
    "    assert module.reverse_list([1, 2, 3]) == [3, 2, 1]\n"
    "    assert module.reverse_list([]) == []\n"
    "    assert module.reverse_str('abc') == 'cba'\n",
)


_register(
    "ext-zip-001", "external/zip",
    "`pair_with(xs, ys)` takes only the first list. Should zip element-wise.",
    "zipops.py", "pair_with",
    '"""Zip helpers."""\n'
    '\n'
    'def pair_with(xs, ys):\n'
    '    """BUG: ignores ys."""\n'
    '    return [(x, x) for x in xs]\n'
    '\n'
    'def enum_list(xs):\n'
    '    return list(enumerate(xs))\n',
    "--- a/zipops.py\n"
    "+++ b/zipops.py\n"
    "@@ -4,1 +4,1 @@ def pair_with(xs, ys):\n"
    "-    return [(x, x) for x in xs]\n"
    "+    return list(zip(xs, ys))\n",
    "def test(module):\n"
    "    assert module.pair_with([1, 2], ['a', 'b']) == [(1, 'a'), (2, 'b')]\n"
    "    assert module.pair_with([], []) == []\n"
    "    assert module.enum_list(['a', 'b']) == [(0, 'a'), (1, 'b')]\n",
)


# ---------- P42: dict helpers ----------

_register(
    "ext-invert-001", "external/invert",
    "`invert_dict(d)` returns an empty dict. Should swap keys and values.",
    "invdict.py", "invert_dict",
    '"""Dict invert."""\n'
    '\n'
    'def invert_dict(d):\n'
    '    """BUG: returns empty dict instead of inverting."""\n'
    '    return {}\n'
    '\n'
    'def has_key(d, k):\n'
    '    return k in d\n',
    "--- a/invdict.py\n"
    "+++ b/invdict.py\n"
    "@@ -4,1 +4,1 @@ def invert_dict(d):\n"
    "-    return {}\n"
    "+    return {v: k for k, v in d.items()}\n",
    "def test(module):\n"
    "    assert module.invert_dict({'a': 1, 'b': 2}) == {1: 'a', 2: 'b'}\n"
    "    assert module.invert_dict({}) == {}\n"
    "    assert module.has_key({'x': 1}, 'x') is True\n",
)


_register(
    "ext-getdef-001", "external/getdef",
    "`get_or_default(d, k, default)` returns None when the key is missing "
    "instead of returning the default.",
    "getdef.py", "get_or_default",
    '"""Dict get helpers."""\n'
    '\n'
    'def get_or_default(d, k, default):\n'
    '    """BUG: falls back to None, not default."""\n'
    '    if k in d:\n'
    '        return d[k]\n'
    '    return None\n'
    '\n'
    'def safe_pop(d, k):\n'
    '    return d.pop(k, None)\n',
    "--- a/getdef.py\n"
    "+++ b/getdef.py\n"
    "@@ -6,1 +6,1 @@ def get_or_default(d, k, default):\n"
    "-    return None\n"
    "+    return default\n",
    "def test(module):\n"
    "    assert module.get_or_default({'a': 1}, 'a', 0) == 1\n"
    "    assert module.get_or_default({'a': 1}, 'b', 0) == 0\n"
    "    assert module.safe_pop({'x': 2}, 'x') == 2\n",
)


# ---------- P42: recursion / iteration ----------

_register(
    "ext-fib-001", "external/fib",
    "`fib(n)` always returns 0. Should return the nth Fibonacci number "
    "with fib(0)=0, fib(1)=1.",
    "fibops.py", "fib",
    '"""Fibonacci helpers."""\n'
    '\n'
    'def fib(n):\n'
    '    """BUG: returns 0 for all n."""\n'
    '    a, b = 0, 1\n'
    '    for _ in range(n):\n'
    '        a = a\n'
    '    return a\n'
    '\n'
    'def fact(n):\n'
    '    r = 1\n'
    '    for i in range(2, n + 1):\n'
    '        r *= i\n'
    '    return r\n',
    "--- a/fibops.py\n"
    "+++ b/fibops.py\n"
    "@@ -6,1 +6,1 @@ def fib(n):\n"
    "-        a = a\n"
    "+        a, b = b, a + b\n",
    "def test(module):\n"
    "    assert module.fib(0) == 0\n"
    "    assert module.fib(1) == 1\n"
    "    assert module.fib(6) == 8\n"
    "    assert module.fact(5) == 120\n",
)


_register(
    "ext-gcd-001", "external/gcd",
    "`gcd(a, b)` recurses with the wrong argument order and never terminates "
    "— capture the bug as always returning the smaller argument.",
    "gcdops.py", "gcd",
    '"""GCD helpers."""\n'
    '\n'
    'def gcd(a, b):\n'
    '    """BUG: returns min instead of the real gcd."""\n'
    '    return a if a < b else b\n'
    '\n'
    'def lcm(a, b):\n'
    '    # Use gcd internally; intentionally correct.\n'
    '    def _gcd(x, y):\n'
    '        while y:\n'
    '            x, y = y, x % y\n'
    '        return x\n'
    '    return (a * b) // _gcd(a, b) if a and b else 0\n',
    "--- a/gcdops.py\n"
    "+++ b/gcdops.py\n"
    "@@ -4,1 +4,4 @@ def gcd(a, b):\n"
    "-    return a if a < b else b\n"
    "+    while b:\n"
    "+        a, b = b, a % b\n"
    "+    return a\n",
    "def test(module):\n"
    "    assert module.gcd(12, 18) == 6\n"
    "    assert module.gcd(7, 13) == 1\n"
    "    assert module.gcd(10, 5) == 5\n"
    "    assert module.lcm(4, 6) == 12\n",
)


# ---------- P42: conditional chain (multi-hunk) ----------

_register(
    "ext-multi-002", "external/multi2",
    "A `StopLight` class has two methods with bugs — `next_color` skips "
    "the yellow state, and `reset` sets the color to 'red' but capital "
    "instead of lowercase. Both must be fixed.",
    "stoplight.py", "StopLight",
    '"""Stop light state machine."""\n'
    '\n'
    'class StopLight:\n'
    '    def __init__(self):\n'
    '        self.color = "red"\n'
    '\n'
    '    def next_color(self):\n'
    '        """BUG: jumps red -> green, skipping yellow."""\n'
    '        if self.color == "red":\n'
    '            self.color = "green"\n'
    '        elif self.color == "green":\n'
    '            self.color = "yellow"\n'
    '        else:\n'
    '            self.color = "red"\n'
    '\n'
    '    def reset(self):\n'
    '        """BUG: sets capital RED, tests expect lowercase."""\n'
    '        self.color = "RED"\n'
    '\n'
    '    def get(self):\n'
    '        return self.color\n',
    "--- a/stoplight.py\n"
    "+++ b/stoplight.py\n"
    "@@ -7,6 +7,8 @@ class StopLight:\n"
    "     def next_color(self):\n"
    "         \"\"\"BUG: jumps red -> green, skipping yellow.\"\"\"\n"
    "         if self.color == \"red\":\n"
    "-            self.color = \"green\"\n"
    "-        elif self.color == \"green\":\n"
    "             self.color = \"yellow\"\n"
    "+        elif self.color == \"yellow\":\n"
    "+            self.color = \"green\"\n"
    "+        elif self.color == \"green\":\n"
    "+            self.color = \"red\"\n"
    "         else:\n"
    "             self.color = \"red\"\n"
    "@@ -16,1 +18,1 @@ class StopLight:\n"
    "-        self.color = \"RED\"\n"
    "+        self.color = \"red\"\n",
    "def test(module):\n"
    "    s = module.StopLight()\n"
    "    assert s.get() == 'red'\n"
    "    s.next_color(); assert s.get() == 'yellow'\n"
    "    s.next_color(); assert s.get() == 'green'\n"
    "    s.next_color(); assert s.get() == 'red'\n"
    "    s.reset(); assert s.get() == 'red'\n",
)


# ---------- P42: boolean short-circuit ----------

_register(
    "ext-boolop-001", "external/boolop",
    "`safe_lookup(d, k)` raises when k is missing even though it checks "
    "membership — the `or` expression is mis-ordered.",
    "lookup.py", "safe_lookup",
    '"""Lookup helper."""\n'
    '\n'
    'def safe_lookup(d, k):\n'
    '    """BUG: raises KeyError on missing k.\n'
    '    (Evaluates d[k] before the guard.)\n'
    '    """\n'
    '    return d[k] or (None if k not in d else d[k])\n'
    '\n'
    'def has_either(d, ks):\n'
    '    return any(k in d for k in ks)\n',
    "--- a/lookup.py\n"
    "+++ b/lookup.py\n"
    "@@ -3,5 +3,5 @@ def safe_lookup(d, k):\n"
    "     \"\"\"BUG: raises KeyError on missing k.\n"
    "     (Evaluates d[k] before the guard.)\n"
    "     \"\"\"\n"
    "-    return d[k] or (None if k not in d else d[k])\n"
    "+    return d[k] if k in d else None\n",
    "def test(module):\n"
    "    assert module.safe_lookup({'x': 1}, 'x') == 1\n"
    "    assert module.safe_lookup({'x': 1}, 'y') is None\n"
    "    assert module.has_either({'a': 1}, ['b', 'a']) is True\n",
)


# ---------- P42: exception handling ----------

_register(
    "ext-try-001", "external/try",
    "`parse_int_or(s, default)` swallows all exceptions including "
    "KeyboardInterrupt. Should only catch ValueError.",
    "tryops.py", "parse_int_or",
    '"""Safe int parsing."""\n'
    '\n'
    'def parse_int_or(s, default):\n'
    '    """BUG: catches everything. Must catch only ValueError."""\n'
    '    try:\n'
    '        return int(s)\n'
    '    except:\n'
    '        return default\n'
    '\n'
    'def parse_float_or(s, default):\n'
    '    try:\n'
    '        return float(s)\n'
    '    except (ValueError, TypeError):\n'
    '        return default\n',
    "--- a/tryops.py\n"
    "+++ b/tryops.py\n"
    "@@ -5,1 +5,1 @@ def parse_int_or(s, default):\n"
    "-    except:\n"
    "+    except ValueError:\n",
    "def test(module):\n"
    "    assert module.parse_int_or('5', 0) == 5\n"
    "    assert module.parse_int_or('x', 9) == 9\n"
    "    assert module.parse_float_or('1.5', 0) == 1.5\n",
)


# ---------- P42: nested data ----------

_register(
    "ext-flat-001", "external/flat",
    "`flatten(xss)` returns the nested list unchanged. Should produce a "
    "flat list of elements.",
    "flatten.py", "flatten",
    '"""Flatten helpers."""\n'
    '\n'
    'def flatten(xss):\n'
    '    """BUG: returns the list-of-lists unchanged."""\n'
    '    return xss\n'
    '\n'
    'def chunk(xs, n):\n'
    '    return [xs[i:i + n] for i in range(0, len(xs), n)]\n',
    "--- a/flatten.py\n"
    "+++ b/flatten.py\n"
    "@@ -4,1 +4,1 @@ def flatten(xss):\n"
    "-    return xss\n"
    "+    return [x for xs in xss for x in xs]\n",
    "def test(module):\n"
    "    assert module.flatten([[1, 2], [3]]) == [1, 2, 3]\n"
    "    assert module.flatten([]) == []\n"
    "    assert module.chunk([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]\n",
)


# ---------- P42: format / representation ----------

_register(
    "ext-repr-001", "external/repr",
    "`to_percent(x)` returns the raw value without a percent sign and "
    "without multiplying by 100.",
    "percent.py", "to_percent",
    '"""Percent formatting."""\n'
    '\n'
    'def to_percent(x):\n'
    '    """BUG: returns raw float."""\n'
    '    return x\n'
    '\n'
    'def to_hex(x):\n'
    '    return hex(x)\n',
    "--- a/percent.py\n"
    "+++ b/percent.py\n"
    "@@ -4,1 +4,1 @@ def to_percent(x):\n"
    "-    return x\n"
    "+    return f\"{x * 100:.1f}%\"\n",
    "def test(module):\n"
    "    assert module.to_percent(0.25) == '25.0%'\n"
    "    assert module.to_percent(1.0) == '100.0%'\n"
    "    assert module.to_hex(255) == '0xff'\n",
)


# ---------- P42: sentinel values ----------

_register(
    "ext-sent-001", "external/sent",
    "`argmax(xs)` returns 0 on an empty list instead of -1 (the "
    "missing-sentinel).",
    "argmax.py", "argmax",
    '"""Argmax helpers."""\n'
    '\n'
    'def argmax(xs):\n'
    '    """Return index of max, -1 when empty.\n'
    '    BUG: returns 0 for empty xs.\n'
    '    """\n'
    '    if not xs:\n'
    '        return 0\n'
    '    best_i = 0\n'
    '    for i, x in enumerate(xs):\n'
    '        if x > xs[best_i]:\n'
    '            best_i = i\n'
    '    return best_i\n'
    '\n'
    'def argmin(xs):\n'
    '    if not xs:\n'
    '        return -1\n'
    '    best_i = 0\n'
    '    for i, x in enumerate(xs):\n'
    '        if x < xs[best_i]:\n'
    '            best_i = i\n'
    '    return best_i\n',
    "--- a/argmax.py\n"
    "+++ b/argmax.py\n"
    "@@ -6,1 +6,1 @@ def argmax(xs):\n"
    "-        return 0\n"
    "+        return -1\n",
    "def test(module):\n"
    "    assert module.argmax([3, 1, 4, 1, 5]) == 4\n"
    "    assert module.argmax([]) == -1\n"
    "    assert module.argmin([3, 1, 4]) == 1\n",
)


# ---------- P42: recursion base case ----------

_register(
    "ext-base-001", "external/base",
    "`power(base, exp)` returns 0 when exp == 0 instead of 1.",
    "powerops.py", "power",
    '"""Integer power."""\n'
    '\n'
    'def power(base, exp):\n'
    '    """BUG: base case returns 0 for exp=0."""\n'
    '    if exp == 0:\n'
    '        return 0\n'
    '    r = 1\n'
    '    for _ in range(exp):\n'
    '        r *= base\n'
    '    return r\n'
    '\n'
    'def sign_of(x):\n'
    '    return (x > 0) - (x < 0)\n',
    "--- a/powerops.py\n"
    "+++ b/powerops.py\n"
    "@@ -5,1 +5,1 @@ def power(base, exp):\n"
    "-        return 0\n"
    "+        return 1\n",
    "def test(module):\n"
    "    assert module.power(2, 10) == 1024\n"
    "    assert module.power(3, 0) == 1\n"
    "    assert module.power(5, 1) == 5\n"
    "    assert module.sign_of(-2) == -1\n",
)


# ---------- P42: inventory iteration ----------

_register(
    "ext-count-001", "external/count",
    "`count_occurrences(xs, target)` always returns 0. The loop body is "
    "missing the increment on a match.",
    "occur.py", "count_occurrences",
    '"""Occurrence counting."""\n'
    '\n'
    'def count_occurrences(xs, target):\n'
    '    """BUG: never increments the counter."""\n'
    '    n = 0\n'
    '    for x in xs:\n'
    '        if x == target:\n'
    '            pass\n'
    '    return n\n'
    '\n'
    'def first_index(xs, target):\n'
    '    for i, x in enumerate(xs):\n'
    '        if x == target:\n'
    '            return i\n'
    '    return -1\n',
    "--- a/occur.py\n"
    "+++ b/occur.py\n"
    "@@ -6,1 +6,1 @@ def count_occurrences(xs, target):\n"
    "-            pass\n"
    "+            n += 1\n",
    "def test(module):\n"
    "    assert module.count_occurrences([1, 2, 1, 3, 1], 1) == 3\n"
    "    assert module.count_occurrences([], 5) == 0\n"
    "    assert module.first_index([10, 20, 30], 20) == 1\n",
)


# ---------- P42: operator precedence ----------

_register(
    "ext-prec-001", "external/prec",
    "`combine_flags(a, b, c)` uses incorrect operator precedence. "
    "`a & b | c` is parsed as `(a & b) | c` — we want `a & (b | c)`.",
    "flagops.py", "combine_flags",
    '"""Bitwise flag helpers."""\n'
    '\n'
    'def combine_flags(a, b, c):\n'
    '    """BUG: needs (b | c), got (a & b) | c."""\n'
    '    return a & b | c\n'
    '\n'
    'def toggle(a, mask):\n'
    '    return a ^ mask\n',
    "--- a/flagops.py\n"
    "+++ b/flagops.py\n"
    "@@ -4,1 +4,1 @@ def combine_flags(a, b, c):\n"
    "-    return a & b | c\n"
    "+    return a & (b | c)\n",
    "def test(module):\n"
    "    assert module.combine_flags(0b110, 0b011, 0b100) == 0b110\n"
    "    assert module.combine_flags(0, 0xFF, 0xFF) == 0\n"
    "    assert module.toggle(0b110, 0b011) == 0b101\n",
)


# ---------- P42: state transitions (class) ----------

_register(
    "ext-class-001", "external/class",
    "A `Stack` class's `pop` returns None even when non-empty. Should "
    "return the top element.",
    "stack.py", "Stack",
    '"""Stack implementation."""\n'
    '\n'
    'class Stack:\n'
    '    def __init__(self):\n'
    '        self._items = []\n'
    '\n'
    '    def push(self, x):\n'
    '        self._items.append(x)\n'
    '\n'
    '    def pop(self):\n'
    '        """BUG: returns None regardless of stack state."""\n'
    '        return None\n'
    '\n'
    '    def peek(self):\n'
    '        return self._items[-1] if self._items else None\n'
    '\n'
    '    def size(self):\n'
    '        return len(self._items)\n',
    "--- a/stack.py\n"
    "+++ b/stack.py\n"
    "@@ -10,1 +10,3 @@ class Stack:\n"
    "-        return None\n"
    "+        if not self._items:\n"
    "+            return None\n"
    "+        return self._items.pop()\n",
    "def test(module):\n"
    "    s = module.Stack()\n"
    "    s.push(1); s.push(2); s.push(3)\n"
    "    assert s.size() == 3\n"
    "    assert s.pop() == 3\n"
    "    assert s.peek() == 2\n"
    "    s.pop(); s.pop()\n"
    "    assert s.pop() is None\n",
)


# ---------- P42: set algebra ----------

_register(
    "ext-setx-001", "external/setx",
    "`symmetric_difference(a, b)` returns the union instead.",
    "setx.py", "symmetric_difference",
    '"""Set helpers."""\n'
    '\n'
    'def symmetric_difference(a, b):\n'
    '    """BUG: returns union instead of symmetric difference."""\n'
    '    return set(a) | set(b)\n'
    '\n'
    'def intersection(a, b):\n'
    '    return set(a) & set(b)\n',
    "--- a/setx.py\n"
    "+++ b/setx.py\n"
    "@@ -4,1 +4,1 @@ def symmetric_difference(a, b):\n"
    "-    return set(a) | set(b)\n"
    "+    return set(a) ^ set(b)\n",
    "def test(module):\n"
    "    assert module.symmetric_difference([1, 2, 3], [2, 3, 4]) == {1, 4}\n"
    "    assert module.intersection([1, 2], [2, 3]) == {2}\n",
)


# ---------- P42: running aggregate ----------

_register(
    "ext-maxx-001", "external/maxx",
    "`running_max(xs)` returns the list unchanged. Should emit the max "
    "seen so far at each index.",
    "rmax.py", "running_max",
    '"""Running maximum."""\n'
    '\n'
    'def running_max(xs):\n'
    '    """BUG: does not update the rolling max."""\n'
    '    out = []\n'
    '    for x in xs:\n'
    '        out.append(x)\n'
    '    return out\n'
    '\n'
    'def running_sum(xs):\n'
    '    total = 0\n'
    '    out = []\n'
    '    for x in xs:\n'
    '        total += x\n'
    '        out.append(total)\n'
    '    return out\n',
    "--- a/rmax.py\n"
    "+++ b/rmax.py\n"
    "@@ -5,1 +5,2 @@ def running_max(xs):\n"
    "-        out.append(x)\n"
    "+        m = x if not out else max(out[-1], x)\n"
    "+        out.append(m)\n",
    "def test(module):\n"
    "    assert module.running_max([1, 3, 2, 5, 4]) == [1, 3, 3, 5, 5]\n"
    "    assert module.running_max([]) == []\n"
    "    assert module.running_sum([1, 2, 3]) == [1, 3, 6]\n",
)


# ---------- P42: default argument correction ----------

_register(
    "ext-defarg-001", "external/defarg",
    "`greet(name)` always greets 'stranger'. Should greet the given name.",
    "greet.py", "greet",
    '"""Greet helpers."""\n'
    '\n'
    'def greet(name):\n'
    '    """BUG: ignores name."""\n'
    '    return "hello stranger"\n'
    '\n'
    'def wave(name):\n'
    '    return f"wave {name}"\n',
    "--- a/greet.py\n"
    "+++ b/greet.py\n"
    "@@ -4,1 +4,1 @@ def greet(name):\n"
    "-    return \"hello stranger\"\n"
    "+    return f\"hello {name}\"\n",
    "def test(module):\n"
    "    assert module.greet('Alice') == 'hello Alice'\n"
    "    assert module.greet('') == 'hello '\n"
    "    assert module.wave('Bob') == 'wave Bob'\n",
)


# ---------- P42: inverted comparator ----------

_register(
    "ext-sortkey-001", "external/sortkey",
    "`sort_by_length(xs)` sorts by the first element instead of the "
    "string length.",
    "sklen.py", "sort_by_length",
    '"""Sort by length."""\n'
    '\n'
    'def sort_by_length(xs):\n'
    '    """BUG: sorts lexicographically, not by length."""\n'
    '    return sorted(xs)\n'
    '\n'
    'def sort_desc(xs):\n'
    '    return sorted(xs, reverse=True)\n',
    "--- a/sklen.py\n"
    "+++ b/sklen.py\n"
    "@@ -4,1 +4,1 @@ def sort_by_length(xs):\n"
    "-    return sorted(xs)\n"
    "+    return sorted(xs, key=len)\n",
    "def test(module):\n"
    "    assert module.sort_by_length(['ccc', 'a', 'bb']) == ['a', 'bb', 'ccc']\n"
    "    assert module.sort_desc([1, 3, 2]) == [3, 2, 1]\n",
)


# ---------- P42: string contains ----------

_register(
    "ext-contain-001", "external/contain",
    "`contains_any(s, needles)` returns True for every input. Should "
    "check whether any of the needles appears in s.",
    "contain.py", "contains_any",
    '"""Substring helpers."""\n'
    '\n'
    'def contains_any(s, needles):\n'
    '    """BUG: returns True regardless of input."""\n'
    '    return True\n'
    '\n'
    'def startswith_any(s, prefixes):\n'
    '    return any(s.startswith(p) for p in prefixes)\n',
    "--- a/contain.py\n"
    "+++ b/contain.py\n"
    "@@ -4,1 +4,1 @@ def contains_any(s, needles):\n"
    "-    return True\n"
    "+    return any(n in s for n in needles)\n",
    "def test(module):\n"
    "    assert module.contains_any('hello', ['ell', 'xyz']) is True\n"
    "    assert module.contains_any('abc', ['xyz']) is False\n"
    "    assert module.startswith_any('hello world', ['hi', 'he']) is True\n",
)


# ---------- P42: graph walk ----------

_register(
    "ext-graph-001", "external/graph",
    "`reachable(graph, start)` fails to visit neighbours. Should return "
    "the set of all nodes reachable from start.",
    "graph.py", "reachable",
    '"""Graph reachability."""\n'
    '\n'
    'def reachable(graph, start):\n'
    '    """BUG: does not traverse neighbours."""\n'
    '    seen = set()\n'
    '    stack = [start]\n'
    '    while stack:\n'
    '        n = stack.pop()\n'
    '        if n in seen:\n'
    '            continue\n'
    '        seen.add(n)\n'
    '    return seen\n'
    '\n'
    'def has_node(graph, n):\n'
    '    return n in graph\n',
    "--- a/graph.py\n"
    "+++ b/graph.py\n"
    "@@ -9,1 +9,2 @@ def reachable(graph, start):\n"
    "         seen.add(n)\n"
    "+        stack.extend(graph.get(n, ()))\n",
    "def test(module):\n"
    "    g = {'a': ['b'], 'b': ['c'], 'c': []}\n"
    "    assert module.reachable(g, 'a') == {'a', 'b', 'c'}\n"
    "    assert module.reachable(g, 'c') == {'c'}\n"
    "    assert module.has_node(g, 'a') is True\n",
)


# ---------- P42: binary search (off-by-one) ----------

_register(
    "ext-bsearch-001", "external/bsearch",
    "`bsearch(xs, target)` uses the wrong bound — never matches the "
    "final element of the list.",
    "bsearch.py", "bsearch",
    '"""Binary search."""\n'
    '\n'
    'def bsearch(xs, target):\n'
    '    """BUG: hi init is len(xs) - 1; should be len(xs).\n'
    '    The half-open-interval convention is used in the body.\n'
    '    """\n'
    '    lo = 0\n'
    '    hi = len(xs) - 1\n'
    '    while lo < hi:\n'
    '        mid = (lo + hi) // 2\n'
    '        if xs[mid] == target:\n'
    '            return mid\n'
    '        if xs[mid] < target:\n'
    '            lo = mid + 1\n'
    '        else:\n'
    '            hi = mid\n'
    '    return -1\n'
    '\n'
    'def contains(xs, target):\n'
    '    return target in xs\n',
    "--- a/bsearch.py\n"
    "+++ b/bsearch.py\n"
    "@@ -6,1 +6,1 @@ def bsearch(xs, target):\n"
    "-    hi = len(xs) - 1\n"
    "+    hi = len(xs)\n",
    "def test(module):\n"
    "    assert module.bsearch([1, 3, 5, 7, 9], 9) == 4\n"
    "    assert module.bsearch([1, 3, 5, 7, 9], 5) == 2\n"
    "    assert module.bsearch([1, 3, 5, 7, 9], 11) == -1\n"
    "    assert module.contains([1, 2, 3], 2) is True\n",
)


def main() -> None:
    out_path = os.path.join(HERE, "swe_lite_style_bank.jsonl")
    with open(out_path, "w", encoding="utf-8") as fh:
        for row in INSTANCES:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(INSTANCES)} instances to {out_path}")
    print("Instance IDs:")
    for row in INSTANCES:
        print(f"  {row['instance_id']:>20}  {row['repo']}")


if __name__ == "__main__":
    main()
