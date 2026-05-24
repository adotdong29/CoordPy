"""W95 — MathVista answer-match executor V1.

Deterministic, no-judge executor that compares a free-form
prediction against a ``MathVistaProblemV1``'s gold answer using
the answer-type / question-type schema fields.  This is the W95
analogue of the W86 ``run_humaneval_executor_v1`` subprocess
oracle: a single function with the SAME semantics for every arm,
so the bench's per-arm pass rate is computed under one consistent
truth function.

The MathVista canonical evaluation distinguishes three top-level
shapes:

  * ``question_type == "multi_choice"`` — the gold ``answer`` is
    one of the entries in ``choices``.  A prediction passes if
    it either:
      - emits a letter (``A``..``Z``) that indexes ``choices``
        into the gold entry, OR
      - emits the gold choice text (case-insensitive, trimmed),
        possibly enclosed in trivial wrappers.
  * ``question_type == "free_form"`` AND
    ``answer_type in {"integer", "float"}`` — numeric.  Parse the
    last number in the prediction, optionally divided by 100 if
    the prediction emits a percent.  Match within tolerance
    derived from the per-problem ``precision`` field (number of
    decimals).
  * Everything else — canonical text match (lowercase, strip
    whitespace, strip trivial wrappers).

The executor never calls a model; it is the W95 anti-cheat /
oracle boundary.  See ``docs/RUNBOOK_W95.md`` for the discipline.

Honest scope (W95)
------------------

* ``W95-L-MATHVISTA-EXECUTOR-V1-NUMERIC-PARSER-CAP`` — V1 parses
  the LAST number-like token in the prediction.  Predictions that
  emit a chain of numbers will be matched against the last one;
  this matches the MathVista canonical evaluator's behaviour and
  is documented anti-cheat: every arm gets the same parser.
* ``W95-L-MATHVISTA-EXECUTOR-V1-MULTI-CHOICE-LETTER-WINDOW-CAP``
  — V1 reads a single-letter answer if and only if it appears as
  a standalone token (e.g., ``A``, ``(A)``, ``A.``); embedded
  ``A`` inside other words is ignored.
* ``W95-L-MATHVISTA-EXECUTOR-V1-NO-LLM-JUDGE-CAP`` — V1 never
  uses an LLM as a judge.  This is a deliberate W88 anti-cheat
  inheritance: a judge-based eval would let providers' judges
  bias the per-arm numbers.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import string
from typing import Any

from .mathvista_loader_v1 import MathVistaProblemV1


W95_MATHVISTA_EXECUTOR_V1_SCHEMA_VERSION: str = (
    "coordpy.mathvista_executor_v1.v1")


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass(frozen=True)
class MathVistaExecutorResultV1:
    """The W95 executor's verdict for a single (prediction,
    problem) pair.  ``passed`` is the canonical pass/fail bit
    used by every bench arm; the diagnostics fields aid debugging
    + audit-chain re-derivation but do not affect ``passed``."""

    passed: bool
    matched_rule: str
    normalized_prediction: str
    normalized_gold: str
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": bool(self.passed),
            "matched_rule": str(self.matched_rule),
            "normalized_prediction": str(self.normalized_prediction),
            "normalized_gold": str(self.normalized_gold),
            "diagnostics": dict(self.diagnostics),
        }


# ---------------------------------------------------------------
# Canonical normalizers
# ---------------------------------------------------------------

_PUNCT_STRIP = "".join(
    c for c in string.punctuation if c not in "-./%")


def _strip_trivial_wrappers(text: str) -> str:
    """Remove leading/trailing wrappers that models commonly emit
    around final answers (e.g. ``**42**``, ``"42"``, ``$42$``)."""
    s = text.strip()
    for _ in range(4):
        if not s:
            break
        if (len(s) >= 2 and s[0] == s[-1]
                and s[0] in {'"', "'", "`", "*", "_", "$"}):
            s = s[1:-1].strip()
            continue
        # Strip leading "Answer:" / "answer is" boilerplate.
        for prefix in (
                "the answer is", "answer is",
                "final answer:", "answer:", "ans:",
                "the answer:", "result:",):
            if s.lower().startswith(prefix):
                s = s[len(prefix):].strip()
                break
        else:
            break
    return s


def _canonical_text(text: str) -> str:
    """Lowercase, strip whitespace, strip trivial wrappers."""
    return _strip_trivial_wrappers(text).lower().strip()


_NUMBER_RE = re.compile(
    r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?"   # 1,234,567(.89)
    r"|-?\d+\.\d+"                        # 3.14
    r"|-?\.\d+"                           # .5
    r"|-?\d+/\d+"                         # 3/4
    r"|-?\d+"                              # 42
)


def _strip_units_and_punct(s: str) -> str:
    """Strip ``$``, ``%``, currency words, and common trailing
    punctuation."""
    out = s.replace(",", "")
    out = re.sub(r"\$|€|£|¥", "", out)
    out = re.sub(r"\s+", " ", out).strip()
    out = out.strip(_PUNCT_STRIP)
    return out


def _parse_last_number(s: str) -> tuple[float | None, bool]:
    """Extract the last number-like token from ``s``.  Returns
    ``(value_or_None, is_percent)``.  Percent values are returned
    DIVIDED BY 100 (so 25% → 0.25)."""
    cleaned = s
    # Find all number tokens; choose the last.
    matches = list(_NUMBER_RE.finditer(cleaned))
    if not matches:
        return None, False
    m = matches[-1]
    raw = m.group(0)
    # Check whether the next non-space char is a '%'.
    tail_idx = m.end()
    while tail_idx < len(cleaned) and cleaned[tail_idx].isspace():
        tail_idx += 1
    is_percent = (
        tail_idx < len(cleaned) and cleaned[tail_idx] == "%")
    raw_no_commas = raw.replace(",", "")
    try:
        if "/" in raw_no_commas:
            num_str, den_str = raw_no_commas.split("/", 1)
            num = float(num_str)
            den = float(den_str)
            if den == 0:
                return None, False
            value = num / den
        else:
            value = float(raw_no_commas)
    except ValueError:
        return None, False
    if is_percent:
        value = value / 100.0
    return value, bool(is_percent)


_LETTER_TOKEN_RE = re.compile(
    r"(?:^|[^A-Za-z0-9])\(?([A-Z])\)?(?=$|[^A-Za-z0-9])")


def _extract_letter(text: str) -> str:
    """Pull out an A-Z letter answer if present as a standalone
    token.  Returns ``""`` if not found.  Only the FIRST such
    token is read — models almost always emit the answer letter
    early in a structured response and adding scan-from-end can
    pick up later capitalised words by accident."""
    if not text:
        return ""
    m = _LETTER_TOKEN_RE.search(text)
    if not m:
        return ""
    return m.group(1)


def _precision_to_decimals(precision: float) -> int:
    """The MathVista ``precision`` field gives the number of
    decimal places the gold answer is reported to.  The canonical
    upstream evaluator uses ``int(precision)`` then rounds both
    pred and gold to that many decimals before strict-comparing.
    We mirror that exactly.  Values like ``0.01`` (sometimes seen
    in dataset rows where the author meant "tolerance 0.01" but
    stored it under the precision field) collapse to 0 decimals
    just like the upstream evaluator does, so we agree by
    construction."""
    if precision is None:
        return 0
    try:
        return int(round(float(precision)))
    except (TypeError, ValueError):
        return 0


def _round_to_precision(value: float, precision: float) -> float:
    p = _precision_to_decimals(precision)
    if p <= 0:
        return float(round(value))
    return float(round(value, p))


# ---------------------------------------------------------------
# Top-level evaluator
# ---------------------------------------------------------------


def _evaluate_multi_choice(
        prediction: str, problem: MathVistaProblemV1,
) -> MathVistaExecutorResultV1:
    norm_pred = _canonical_text(prediction)
    norm_gold = _canonical_text(problem.answer)
    choices = problem.choices or ()
    # Path A — letter match.
    letter = _extract_letter(prediction)
    diagnostics: dict[str, Any] = {
        "letter": str(letter),
        "n_choices": int(len(choices)),
    }
    if letter and choices:
        idx = ord(letter) - ord("A")
        if 0 <= idx < len(choices):
            picked = _canonical_text(choices[idx])
            if picked == norm_gold:
                return MathVistaExecutorResultV1(
                    passed=True,
                    matched_rule="multi_choice_letter",
                    normalized_prediction=str(picked),
                    normalized_gold=str(norm_gold),
                    diagnostics=diagnostics)
    # Path B — choice-text exact match.
    if choices:
        for c in choices:
            if _canonical_text(c) == norm_pred:
                if _canonical_text(c) == norm_gold:
                    return MathVistaExecutorResultV1(
                        passed=True,
                        matched_rule="multi_choice_text",
                        normalized_prediction=str(norm_pred),
                        normalized_gold=str(norm_gold),
                        diagnostics=diagnostics)
                return MathVistaExecutorResultV1(
                    passed=False,
                    matched_rule="multi_choice_text_wrong",
                    normalized_prediction=str(norm_pred),
                    normalized_gold=str(norm_gold),
                    diagnostics=diagnostics)
    # Path C — choice-text contained-in-prediction match (the
    # gold choice text appears in the prediction with no other
    # choice text appearing).
    if choices:
        present = []
        for c in choices:
            if _canonical_text(c) in norm_pred:
                present.append(_canonical_text(c))
        if len(present) == 1 and present[0] == norm_gold:
            return MathVistaExecutorResultV1(
                passed=True,
                matched_rule="multi_choice_unique_contained",
                normalized_prediction=str(norm_pred),
                normalized_gold=str(norm_gold),
                diagnostics=diagnostics)
    return MathVistaExecutorResultV1(
        passed=False,
        matched_rule="multi_choice_no_match",
        normalized_prediction=str(norm_pred),
        normalized_gold=str(norm_gold),
        diagnostics=diagnostics)


def _evaluate_numeric(
        prediction: str, problem: MathVistaProblemV1,
) -> MathVistaExecutorResultV1:
    pred_cleaned = _strip_trivial_wrappers(prediction)
    gold_cleaned = _strip_trivial_wrappers(problem.answer)
    pred_val, pred_is_pct = _parse_last_number(pred_cleaned)
    gold_val, gold_is_pct = _parse_last_number(gold_cleaned)
    decimals = _precision_to_decimals(problem.precision)
    diagnostics: dict[str, Any] = {
        "pred_value": pred_val,
        "gold_value": gold_val,
        "pred_is_percent": bool(pred_is_pct),
        "gold_is_percent": bool(gold_is_pct),
        "decimals": int(decimals),
        "precision": float(problem.precision),
    }
    if pred_val is None or gold_val is None:
        return MathVistaExecutorResultV1(
            passed=False,
            matched_rule="numeric_unparseable",
            normalized_prediction=str(pred_cleaned),
            normalized_gold=str(gold_cleaned),
            diagnostics=diagnostics)
    # MathVista canonical eval rounds both pred and gold to
    # ``int(precision)`` decimals (or to integer for
    # ``answer_type == "integer"``) and strict-compares.
    answer_type = (problem.answer_type or "").lower()
    if answer_type == "integer":
        pred_rounded = float(round(pred_val))
        gold_rounded = float(round(gold_val))
    else:
        pred_rounded = _round_to_precision(
            pred_val, problem.precision)
        gold_rounded = _round_to_precision(
            gold_val, problem.precision)
    diagnostics["pred_rounded"] = pred_rounded
    diagnostics["gold_rounded"] = gold_rounded
    passed = abs(pred_rounded - gold_rounded) < 1e-9
    return MathVistaExecutorResultV1(
        passed=bool(passed),
        matched_rule="numeric_tolerance",
        normalized_prediction=str(pred_rounded),
        normalized_gold=str(gold_rounded),
        diagnostics=diagnostics)


def _evaluate_text(
        prediction: str, problem: MathVistaProblemV1,
) -> MathVistaExecutorResultV1:
    norm_pred = _canonical_text(prediction)
    norm_gold = _canonical_text(problem.answer)
    if not norm_gold:
        return MathVistaExecutorResultV1(
            passed=False,
            matched_rule="text_no_gold",
            normalized_prediction=str(norm_pred),
            normalized_gold=str(norm_gold),
            diagnostics={})
    if norm_pred == norm_gold:
        return MathVistaExecutorResultV1(
            passed=True,
            matched_rule="text_exact",
            normalized_prediction=str(norm_pred),
            normalized_gold=str(norm_gold),
            diagnostics={})
    # Unit-stripped match.
    if (_strip_units_and_punct(norm_pred)
            == _strip_units_and_punct(norm_gold)):
        return MathVistaExecutorResultV1(
            passed=True,
            matched_rule="text_unit_stripped",
            normalized_prediction=str(norm_pred),
            normalized_gold=str(norm_gold),
            diagnostics={})
    return MathVistaExecutorResultV1(
        passed=False,
        matched_rule="text_no_match",
        normalized_prediction=str(norm_pred),
        normalized_gold=str(norm_gold),
        diagnostics={})


def evaluate_answer_v1(
        *,
        prediction: str,
        problem: MathVistaProblemV1,
) -> MathVistaExecutorResultV1:
    """Top-level dispatcher.  Routes to one of the three rule
    families based on the problem's schema fields, then returns a
    content-addressable verdict.

    The dispatcher is the W95 anti-cheat boundary: every arm in
    the bench will route through this exact function.
    """
    qt = (problem.question_type or "").lower()
    at = (problem.answer_type or "").lower()
    if not isinstance(prediction, str):
        prediction = str(prediction)
    if qt == "multi_choice":
        return _evaluate_multi_choice(prediction, problem)
    if qt == "free_form":
        if at in {"integer", "float"}:
            return _evaluate_numeric(prediction, problem)
        # Lists are evaluated as canonical text.
        return _evaluate_text(prediction, problem)
    # Unknown shape: route on answer_type alone.
    if at in {"integer", "float"}:
        return _evaluate_numeric(prediction, problem)
    return _evaluate_text(prediction, problem)


def executor_self_test_on_gold_v1(
        problems: tuple[MathVistaProblemV1, ...],
) -> dict[str, Any]:
    """Feed each problem's gold answer back through
    ``evaluate_answer_v1`` and report the pass rate.  A
    well-formed executor must achieve 100 % pass on gold; any
    pass rate < 100 % indicates a class of problems the executor
    cannot recognise as correct under its own normalisation rules
    — which would silently penalise EVERY arm.

    The result includes per-rule pass counts and the list of
    pids that failed self-test so the harness can refuse to
    proceed.
    """
    n = len(problems)
    n_pass = 0
    by_rule: dict[str, int] = {}
    failed_pids: list[dict[str, Any]] = []
    for p in problems:
        v = evaluate_answer_v1(
            prediction=p.answer, problem=p)
        by_rule[v.matched_rule] = (
            by_rule.get(v.matched_rule, 0) + 1)
        if v.passed:
            n_pass += 1
        else:
            failed_pids.append({
                "pid": str(p.pid),
                "question_type": str(p.question_type),
                "answer_type": str(p.answer_type),
                "answer": str(p.answer),
                "rule": str(v.matched_rule),
                "normalized_pred": str(v.normalized_prediction),
                "normalized_gold": str(v.normalized_gold),
            })
    pass_rate = float(n_pass) / float(n) if n > 0 else 0.0
    return {
        "schema": W95_MATHVISTA_EXECUTOR_V1_SCHEMA_VERSION,
        "n_problems": int(n),
        "n_pass": int(n_pass),
        "pass_rate": float(pass_rate),
        "by_rule": dict(by_rule),
        "failed_pids": list(failed_pids),
    }


__all__ = [
    "W95_MATHVISTA_EXECUTOR_V1_SCHEMA_VERSION",
    "MathVistaExecutorResultV1",
    "evaluate_answer_v1",
    "executor_self_test_on_gold_v1",
]
