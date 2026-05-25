"""W96-D — RealWorldQA short-answer executor V1.

Deterministic, no-judge executor that compares a free-form
prediction against a ``RealWorldQAProblemV1``'s single canonical
``answer`` string under three rule classes:

  * **multi-choice letter**: if the gold answer is a single
    capital letter (``A``..``Z``) and the prediction contains
    that letter as a standalone token, PASS.  This catches the
    multi-choice subset of RealWorldQA where the answer is
    "A" / "B" / etc.
  * **numeric**: if both prediction and gold parse as numbers,
    match within ±5 % relative tolerance (absolute floor of 5e-2
    for near-zero gold), mirroring ChartQA's relaxed-accuracy
    pattern.  RealWorldQA has a small numeric subset (counts,
    measurements).
  * **canonical text**: lowercase, strip whitespace, strip
    trivial wrappers; unit-stripped text exact match; gold-text
    contained-in-prediction match (when no other plausible
    answer is also present).

The executor never calls a model; it is the W96-D D2 anti-cheat
boundary.  See ``docs/RUNBOOK_W96D.md`` for the discipline.

Honest scope (W96-D D2)
-----------------------

* ``W96-L-REALWORLDQA-EXECUTOR-V1-NUMERIC-PARSER-CAP`` — V1 parses
  the LAST number-like token in the prediction (same parser as
  the W95 / W96-D ChartQA executors).
* ``W96-L-REALWORLDQA-EXECUTOR-V1-NO-LLM-JUDGE-CAP`` — V1 never
  uses an LLM as a judge.  Documented anti-cheat from W88.
* ``W96-L-REALWORLDQA-EXECUTOR-V1-SINGLE-ANSWER-CAP`` —
  RealWorldQA's HF schema stores a single ``answer`` string per
  problem; V1 treats the answer as a 1-element label tuple and
  routes through the same matchers as ChartQA.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import string
from typing import Any

from .realworldqa_loader_v1 import RealWorldQAProblemV1


W96_REALWORLDQA_EXECUTOR_V1_SCHEMA_VERSION: str = (
    "coordpy.realworldqa_executor_v1.v1")


REALWORLDQA_RELAXED_RELATIVE_TOLERANCE: float = 0.05
REALWORLDQA_RELAXED_ABSOLUTE_FLOOR: float = 0.05


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass(frozen=True)
class RealWorldQAExecutorResultV1:
    passed: bool
    matched_rule: str
    normalized_prediction: str
    normalized_gold: str
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": bool(self.passed),
            "matched_rule": str(self.matched_rule),
            "normalized_prediction": str(
                self.normalized_prediction),
            "normalized_gold": str(self.normalized_gold),
            "diagnostics": dict(self.diagnostics),
        }


_PUNCT_STRIP = "".join(
    c for c in string.punctuation if c not in "-./%")


def _strip_trivial_wrappers(text: str) -> str:
    s = text.strip()
    for _ in range(4):
        if not s:
            break
        if (len(s) >= 2 and s[0] == s[-1]
                and s[0] in {'"', "'", "`", "*", "_", "$"}):
            s = s[1:-1].strip()
            continue
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
    return _strip_trivial_wrappers(text).lower().strip()


_NUMBER_RE = re.compile(
    r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?"
    r"|-?\d+\.\d+"
    r"|-?\.\d+"
    r"|-?\d+/\d+"
    r"|-?\d+"
)


def _strip_units_and_punct(s: str) -> str:
    out = s.replace(",", "")
    out = re.sub(r"\$|€|£|¥", "", out)
    out = re.sub(r"\s+", " ", out).strip()
    out = out.strip(_PUNCT_STRIP)
    return out


def _parse_last_number(s: str) -> tuple[float | None, bool]:
    cleaned = s
    matches = list(_NUMBER_RE.finditer(cleaned))
    if not matches:
        return None, False
    m = matches[-1]
    raw = m.group(0)
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
    if not text:
        return ""
    m = _LETTER_TOKEN_RE.search(text)
    if not m:
        return ""
    return m.group(1)


def _relaxed_numeric_match(
        pred_val: float, gold_val: float,
        *,
        rel_tol: float = REALWORLDQA_RELAXED_RELATIVE_TOLERANCE,
        abs_floor: float = REALWORLDQA_RELAXED_ABSOLUTE_FLOOR,
) -> bool:
    if abs(gold_val) < 1e-9:
        return bool(abs(pred_val - gold_val) <= float(abs_floor))
    rel = abs(pred_val - gold_val) / abs(gold_val)
    return bool(rel <= float(rel_tol))


def _is_letter_answer(gold: str) -> bool:
    g = str(gold).strip()
    return bool(len(g) == 1 and g.isalpha() and g == g.upper())


def evaluate_realworldqa_answer_v1(
        *,
        prediction: str,
        problem: RealWorldQAProblemV1,
) -> RealWorldQAExecutorResultV1:
    """Top-level dispatcher for RealWorldQA short-answer
    evaluation."""
    if not isinstance(prediction, str):
        prediction = str(prediction)
    gold = problem.answer or ""
    norm_pred = _canonical_text(prediction)
    norm_gold = _canonical_text(gold)
    diagnostics: dict[str, Any] = {}

    # Rule 1 — multi-choice letter answer.
    if _is_letter_answer(gold):
        pred_letter = _extract_letter(prediction)
        diagnostics["pred_letter"] = str(pred_letter)
        diagnostics["gold_letter"] = str(gold.strip().upper())
        if pred_letter == gold.strip().upper():
            return RealWorldQAExecutorResultV1(
                passed=True,
                matched_rule="multi_choice_letter",
                normalized_prediction=str(pred_letter),
                normalized_gold=str(gold.strip().upper()),
                diagnostics=diagnostics)

    pred_val, pred_is_pct = _parse_last_number(
        _strip_trivial_wrappers(prediction))
    gold_val, gold_is_pct = _parse_last_number(
        _strip_trivial_wrappers(gold))
    diagnostics.update({
        "pred_value": pred_val,
        "gold_value": gold_val,
        "pred_is_percent": bool(pred_is_pct),
        "gold_is_percent": bool(gold_is_pct),
    })

    # Rule 2 — relaxed numeric.
    if pred_val is not None and gold_val is not None:
        if _relaxed_numeric_match(pred_val, gold_val):
            return RealWorldQAExecutorResultV1(
                passed=True,
                matched_rule="numeric_relaxed",
                normalized_prediction=str(pred_val),
                normalized_gold=str(gold_val),
                diagnostics={
                    **diagnostics,
                    "relative_tolerance": (
                        REALWORLDQA_RELAXED_RELATIVE_TOLERANCE),
                })

    # Rule 3 — canonical text exact.
    if norm_pred == norm_gold and norm_gold:
        return RealWorldQAExecutorResultV1(
            passed=True,
            matched_rule="text_exact",
            normalized_prediction=str(norm_pred),
            normalized_gold=str(norm_gold),
            diagnostics=diagnostics)

    # Rule 4 — unit-stripped text exact.
    stripped_pred = _strip_units_and_punct(norm_pred)
    stripped_gold = _strip_units_and_punct(norm_gold)
    if stripped_pred == stripped_gold and stripped_gold:
        return RealWorldQAExecutorResultV1(
            passed=True,
            matched_rule="text_unit_stripped",
            normalized_prediction=str(stripped_pred),
            normalized_gold=str(stripped_gold),
            diagnostics=diagnostics)

    # Rule 5 — gold-text contained-in-prediction.  Skipped when
    # both pred and gold parse as numbers (false-positive guard
    # mirroring the ChartQA executor: "0" in "0.10" is not a
    # numeric match).
    both_numeric = (
        pred_val is not None and gold_val is not None)
    if (not both_numeric
            and norm_gold
            and norm_gold in norm_pred
            and len(norm_gold) >= 2):
        return RealWorldQAExecutorResultV1(
            passed=True,
            matched_rule="text_contained",
            normalized_prediction=str(norm_pred),
            normalized_gold=str(norm_gold),
            diagnostics=diagnostics)

    return RealWorldQAExecutorResultV1(
        passed=False,
        matched_rule="no_match",
        normalized_prediction=str(norm_pred),
        normalized_gold=str(norm_gold),
        diagnostics=diagnostics)


def executor_self_test_on_gold_v1(
        problems: tuple[RealWorldQAProblemV1, ...],
) -> dict[str, Any]:
    n = len(problems)
    n_pass = 0
    by_rule: dict[str, int] = {}
    failed_pids: list[dict[str, Any]] = []
    for p in problems:
        v = evaluate_realworldqa_answer_v1(
            prediction=p.answer, problem=p)
        by_rule[v.matched_rule] = (
            by_rule.get(v.matched_rule, 0) + 1)
        if v.passed:
            n_pass += 1
        else:
            failed_pids.append({
                "pid": str(p.pid),
                "answer": str(p.answer),
                "rule": str(v.matched_rule),
                "normalized_pred": str(v.normalized_prediction),
                "normalized_gold": str(v.normalized_gold),
            })
    pass_rate = float(n_pass) / float(n) if n > 0 else 0.0
    return {
        "schema": W96_REALWORLDQA_EXECUTOR_V1_SCHEMA_VERSION,
        "n_problems": int(n),
        "n_pass": int(n_pass),
        "pass_rate": float(pass_rate),
        "by_rule": dict(by_rule),
        "failed_pids": list(failed_pids),
    }


__all__ = [
    "W96_REALWORLDQA_EXECUTOR_V1_SCHEMA_VERSION",
    "REALWORLDQA_RELAXED_RELATIVE_TOLERANCE",
    "REALWORLDQA_RELAXED_ABSOLUTE_FLOOR",
    "RealWorldQAExecutorResultV1",
    "evaluate_realworldqa_answer_v1",
    "executor_self_test_on_gold_v1",
]
