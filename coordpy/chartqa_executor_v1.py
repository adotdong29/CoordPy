"""W96-D — ChartQA relaxed-accuracy executor V1.

Deterministic, no-judge executor that compares a free-form
prediction against a ``ChartQAProblemV1``'s gold ``labels`` tuple
using the canonical ChartQA "relaxed accuracy" rules:

  * **numeric answers**: a prediction passes if its parsed numeric
    value is within ±5 % relative tolerance of any gold label's
    parsed numeric value (with an absolute fallback of 5e-2 for
    near-zero gold values).
  * **text answers**: a prediction passes if its canonicalised
    text matches any gold label's canonicalised text exactly (case-
    insensitive, trimmed, with trivial wrappers stripped).
  * **contained-in match**: if a single gold label's canonicalised
    text appears as a standalone token in the prediction and no
    other label's text appears, that is a PASS — handles models
    that emit "The answer is 42 dollars." style.

This is the W96-D analogue of the W95 ``evaluate_answer_v1``
function: a single function with the SAME semantics for every
arm, so the bench's per-arm pass rate is computed under one
consistent truth function.  The 5 % relative tolerance follows
the original ChartQA paper (Masry et al., 2022) which defines
relaxed accuracy as "predicted numerical answer within 5 % of
gold for numeric questions, exact match for text".

The executor never calls a model; it is the W96-D anti-cheat /
oracle boundary.  See ``docs/RUNBOOK_W96D.md`` for the discipline.

Honest scope (W96-D)
--------------------

* ``W96-L-CHARTQA-EXECUTOR-V1-NUMERIC-PARSER-CAP`` — V1 parses
  the LAST number-like token in the prediction (same as the W95
  MathVista executor).  Every arm gets the same parser.
* ``W96-L-CHARTQA-EXECUTOR-V1-NO-LLM-JUDGE-CAP`` — V1 never uses
  an LLM as a judge.  Documented anti-cheat inheritance from W88.
* ``W96-L-CHARTQA-EXECUTOR-V1-RELAXED-TOLERANCE-FIXED-CAP`` — V1
  uses 5 % relative tolerance as the canonical relaxed-accuracy
  rule.  The tolerance is not configurable per-problem (the HF
  ChartQA schema does not carry per-problem precision).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import string
from typing import Any

from .chartqa_loader_v1 import ChartQAProblemV1


W96_CHARTQA_EXECUTOR_V1_SCHEMA_VERSION: str = (
    "coordpy.chartqa_executor_v1.v1")


# Canonical ChartQA relaxed-accuracy tolerances (Masry et al.,
# 2022).  Both are documented anti-cheat constants — every arm
# gets the same numbers.
CHARTQA_RELAXED_RELATIVE_TOLERANCE: float = 0.05  # 5 %
CHARTQA_RELAXED_ABSOLUTE_FLOOR: float = 0.05      # near-zero fallback


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass(frozen=True)
class ChartQAExecutorResultV1:
    """The W96-D executor's verdict for a single (prediction,
    problem) pair.  ``passed`` is the canonical pass/fail bit
    used by every bench arm."""

    passed: bool
    matched_rule: str
    matched_label_idx: int
    normalized_prediction: str
    normalized_gold: str
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": bool(self.passed),
            "matched_rule": str(self.matched_rule),
            "matched_label_idx": int(self.matched_label_idx),
            "normalized_prediction": str(self.normalized_prediction),
            "normalized_gold": str(self.normalized_gold),
            "diagnostics": dict(self.diagnostics),
        }


# ---------------------------------------------------------------
# Canonical normalisers (mirroring W95 executor patterns)
# ---------------------------------------------------------------

_PUNCT_STRIP = "".join(
    c for c in string.punctuation if c not in "-./%")


def _strip_trivial_wrappers(text: str) -> str:
    """Remove leading/trailing wrappers that models commonly emit
    around final answers (``**42**``, ``"42"``, ``$42$``,
    ``Answer: 42``)."""
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
    DIVIDED BY 100 (so ``25%`` → ``0.25``)."""
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


# ---------------------------------------------------------------
# Top-level evaluator
# ---------------------------------------------------------------


def _relaxed_numeric_match(
        pred_val: float, gold_val: float,
        *,
        rel_tol: float = CHARTQA_RELAXED_RELATIVE_TOLERANCE,
        abs_floor: float = CHARTQA_RELAXED_ABSOLUTE_FLOOR,
) -> bool:
    """ChartQA canonical relaxed-accuracy: within 5 % relative
    tolerance, with an absolute floor for near-zero gold values."""
    if abs(gold_val) < 1e-9:
        return bool(abs(pred_val - gold_val) <= float(abs_floor))
    rel = abs(pred_val - gold_val) / abs(gold_val)
    return bool(rel <= float(rel_tol))


def evaluate_chartqa_answer_v1(
        *,
        prediction: str,
        problem: ChartQAProblemV1,
) -> ChartQAExecutorResultV1:
    """Top-level dispatcher.  Tries each gold label in turn under
    the canonical ChartQA relaxed-accuracy rules; returns PASS on
    the first match, else FAIL with diagnostics from the closest
    attempt.

    Trial order for each label:

      1. Relaxed numeric match (if both pred and gold parse as
         numbers).
      2. Canonical text exact match.
      3. Unit-stripped text exact match.
      4. Contained-in match (gold label canonical text appears
         as substring of canonical prediction).

    The dispatcher is the W96-D anti-cheat boundary: every arm in
    the bench will route through this exact function.
    """
    if not isinstance(prediction, str):
        prediction = str(prediction)

    norm_pred = _canonical_text(prediction)
    pred_val, pred_is_pct = _parse_last_number(
        _strip_trivial_wrappers(prediction))

    labels = problem.labels or ()
    diagnostics: dict[str, Any] = {
        "n_labels": int(len(labels)),
        "pred_value": pred_val,
        "pred_is_percent": bool(pred_is_pct),
    }

    if not labels:
        return ChartQAExecutorResultV1(
            passed=False,
            matched_rule="no_gold_label",
            matched_label_idx=-1,
            normalized_prediction=str(norm_pred),
            normalized_gold="",
            diagnostics=diagnostics)

    closest_attempt: dict[str, Any] = {}
    for idx, gold in enumerate(labels):
        norm_gold = _canonical_text(gold)
        gold_val, gold_is_pct = _parse_last_number(
            _strip_trivial_wrappers(gold))

        # Rule 1 — relaxed numeric.
        if pred_val is not None and gold_val is not None:
            if _relaxed_numeric_match(pred_val, gold_val):
                return ChartQAExecutorResultV1(
                    passed=True,
                    matched_rule="numeric_relaxed",
                    matched_label_idx=int(idx),
                    normalized_prediction=str(pred_val),
                    normalized_gold=str(gold_val),
                    diagnostics={
                        **diagnostics,
                        "gold_value": gold_val,
                        "gold_is_percent": bool(gold_is_pct),
                        "relative_tolerance": (
                            CHARTQA_RELAXED_RELATIVE_TOLERANCE),
                    })

        # Rule 2 — canonical text exact.
        if norm_pred == norm_gold and norm_gold:
            return ChartQAExecutorResultV1(
                passed=True,
                matched_rule="text_exact",
                matched_label_idx=int(idx),
                normalized_prediction=str(norm_pred),
                normalized_gold=str(norm_gold),
                diagnostics={**diagnostics})

        # Rule 3 — unit-stripped text exact.
        stripped_pred = _strip_units_and_punct(norm_pred)
        stripped_gold = _strip_units_and_punct(norm_gold)
        if (stripped_pred == stripped_gold
                and stripped_gold):
            return ChartQAExecutorResultV1(
                passed=True,
                matched_rule="text_unit_stripped",
                matched_label_idx=int(idx),
                normalized_prediction=str(stripped_pred),
                normalized_gold=str(stripped_gold),
                diagnostics={**diagnostics})

        # Rule 4 — contained-in (gold is a token in prediction).
        # Skipped when both pred and gold parse as numbers: the
        # numeric-relaxed check already covers that case, and a
        # substring match on numeric tokens (e.g. "0" in "0.10")
        # is a false positive that would silently break the
        # executor.
        both_numeric = (
            pred_val is not None and gold_val is not None)
        if (not both_numeric
                and norm_gold
                and norm_gold in norm_pred
                and len(norm_gold) >= 1):
            # Only accept if no OTHER label's text is also present.
            others_present = False
            for j, other in enumerate(labels):
                if j == idx:
                    continue
                other_norm = _canonical_text(other)
                if (other_norm
                        and other_norm != norm_gold
                        and other_norm in norm_pred):
                    others_present = True
                    break
            if not others_present:
                return ChartQAExecutorResultV1(
                    passed=True,
                    matched_rule="text_contained",
                    matched_label_idx=int(idx),
                    normalized_prediction=str(norm_pred),
                    normalized_gold=str(norm_gold),
                    diagnostics={**diagnostics})

        if not closest_attempt:
            closest_attempt = {
                "first_gold_label_idx": int(idx),
                "first_gold_value": gold_val,
                "first_gold_normalized": str(norm_gold),
            }

    # No rule matched any label → FAIL.
    return ChartQAExecutorResultV1(
        passed=False,
        matched_rule="no_match",
        matched_label_idx=-1,
        normalized_prediction=str(norm_pred),
        normalized_gold=str(
            closest_attempt.get("first_gold_normalized", "")),
        diagnostics={**diagnostics, **closest_attempt})


def executor_self_test_on_gold_v1(
        problems: tuple[ChartQAProblemV1, ...],
) -> dict[str, Any]:
    """Feed each problem's first gold label back through
    ``evaluate_chartqa_answer_v1`` and report the pass rate.  A
    well-formed executor must achieve ~100 % on gold; failures
    indicate a class of problems the executor cannot recognise as
    correct under its own normalisation rules — which would
    silently penalise EVERY arm.
    """
    n = len(problems)
    n_pass = 0
    by_rule: dict[str, int] = {}
    failed_pids: list[dict[str, Any]] = []
    for p in problems:
        gold_pred = (
            p.labels[0] if p.labels else "")
        v = evaluate_chartqa_answer_v1(
            prediction=gold_pred, problem=p)
        by_rule[v.matched_rule] = (
            by_rule.get(v.matched_rule, 0) + 1)
        if v.passed:
            n_pass += 1
        else:
            failed_pids.append({
                "pid": str(p.pid),
                "labels": list(p.labels),
                "rule": str(v.matched_rule),
                "normalized_pred": str(v.normalized_prediction),
                "normalized_gold": str(v.normalized_gold),
            })
    pass_rate = float(n_pass) / float(n) if n > 0 else 0.0
    return {
        "schema": W96_CHARTQA_EXECUTOR_V1_SCHEMA_VERSION,
        "n_problems": int(n),
        "n_pass": int(n_pass),
        "pass_rate": float(pass_rate),
        "by_rule": dict(by_rule),
        "failed_pids": list(failed_pids),
    }


__all__ = [
    "W96_CHARTQA_EXECUTOR_V1_SCHEMA_VERSION",
    "CHARTQA_RELAXED_RELATIVE_TOLERANCE",
    "CHARTQA_RELAXED_ABSOLUTE_FLOOR",
    "ChartQAExecutorResultV1",
    "evaluate_chartqa_answer_v1",
    "executor_self_test_on_gold_v1",
]
