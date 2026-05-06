"""Collaborative-module task — 5 LLM agents write Python that must pass tests.

Target: a small Transaction-Ledger Analyzer library with 5 interconnected
functions. Each of 5 agents (one per function) writes a single function.
All outputs are concatenated into one Python module and run against a
test suite.

Agent specialties and the function each writes:
  - parser      → parse_transaction(line: str) -> dict
  - validator   → validate_transaction(txn: dict) -> bool
  - aggregator  → aggregate_by_category(txns: list[dict]) -> dict[str, Decimal]
  - detector    → detect_anomalies(txns: list[dict], z_threshold: float=2.0) -> list[dict]
  - integrator  → analyze_ledger(raw_lines: list[str]) -> dict  (calls all 4)

The integrator function FORCES real dependency: it must call the other
four in the right order with matching signatures. If any agent gets a
signature wrong, the test suite catches it.

Partial-credit scoring (weighted by composition depth):
    parser_tests      * 0.15
  + validator_tests   * 0.15
  + aggregator_tests  * 0.20
  + detector_tests    * 0.20
  + integrator_tests  * 0.30
  (+ 0.10 composition bonus if analyze_ledger is callable + returns dict)

So a broken integrator can't mask working upstream functions, and a
working integrator is worth 40% of the total by itself.
"""

from __future__ import annotations
from dataclasses import dataclass


# --------------------------- Specification ------------------------------

MODULE_HEADER = """\
\"\"\"Transaction-Ledger Analyzer.
Each of 5 functions is written by a different LLM agent. They compose into
a single analyze_ledger pipeline.
\"\"\"
from decimal import Decimal
from datetime import date, datetime
import statistics
"""


FUNCTION_SPECS = {
    "parser": {
        "name": "parse_transaction",
        "signature": "(line: str) -> dict",
        "spec": (
            "Parse a CSV line 'YYYY-MM-DD,TYPE,AMOUNT,CATEGORY' into a dict "
            "{date: datetime.date, type: str ('DEBIT' or 'CREDIT'), amount: "
            "Decimal, category: str}. Raise ValueError on any malformed input "
            "(wrong field count, unknown type, negative amount, bad date). "
            "Categories are lowercased on output. Imports already provided: "
            "Decimal, date, datetime, statistics."
        ),
        "example": (
            "parse_transaction('2026-01-15,DEBIT,45.20,Food') returns "
            "{'date': date(2026,1,15), 'type': 'DEBIT', 'amount': Decimal('45.20'), 'category': 'food'}"
        ),
    },
    "validator": {
        "name": "validate_transaction",
        "signature": "(txn: dict) -> bool",
        "spec": (
            "Return True iff the transaction dict has all four keys (date, "
            "type, amount, category), amount is a Decimal > 0, type is in "
            "{'DEBIT', 'CREDIT'}, date is a datetime.date not in the future "
            "(past or today only), and category is a non-empty string. Return "
            "False on any violation — do NOT raise."
        ),
        "example": "validate_transaction({'date': date(2026,1,15), 'type': 'DEBIT', 'amount': Decimal('10'), 'category': 'food'}) returns True",
    },
    "aggregator": {
        "name": "aggregate_by_category",
        "signature": "(txns: list[dict]) -> dict",
        "spec": (
            "Given a list of validated transaction dicts, return a dict "
            "{category: Decimal} where each value is the SIGNED sum of "
            "amounts for that category: CREDIT contributes +amount, DEBIT "
            "contributes -amount. Omit categories with no transactions. "
            "All output values are Decimal."
        ),
        "example": (
            "aggregate_by_category([{'date':date(2026,1,15),'type':'DEBIT','amount':Decimal('10'),'category':'food'}, "
            "{'date':date(2026,1,15),'type':'CREDIT','amount':Decimal('50'),'category':'food'}]) returns {'food': Decimal('40')}"
        ),
    },
    "detector": {
        "name": "detect_anomalies",
        "signature": "(txns: list[dict], z_threshold: float = 2.0) -> list[dict]",
        "spec": (
            "Return transactions whose amount (as float) exceeds z_threshold "
            "standard deviations from the mean of their own category. Use "
            "statistics.stdev (require ≥ 2 samples in a category; skip "
            "categories with fewer). Return in the same order as input."
        ),
        "example": "detect_anomalies returns items with unusually high/low amount within their category",
    },
    "integrator": {
        "name": "analyze_ledger",
        "signature": "(raw_lines: list[str]) -> dict",
        "spec": (
            "Pipeline: call parse_transaction on each line (catching "
            "ValueError), then validate_transaction on each parsed dict, "
            "then aggregate_by_category + detect_anomalies on the validated "
            "list. Return {'totals': <aggregate>, 'anomalies': <list>, "
            "'rejected': <int of lines that parsed OK but failed "
            "validation>, 'parse_errors': <int of lines that raised "
            "ValueError during parsing>}. Must call the other 4 functions "
            "by name (parse_transaction, validate_transaction, "
            "aggregate_by_category, detect_anomalies)."
        ),
        "example": "analyze_ledger(['2026-01-15,DEBIT,10,food']) returns a dict with totals, anomalies, rejected, parse_errors keys",
    },
}


SPEC_ORDER = ["parser", "validator", "aggregator", "detector", "integrator"]


def agent_prompt(specialty: str, dependency_outputs: dict[str, str] = None) -> str:
    """Build the LLM prompt for one specialty agent."""
    s = FUNCTION_SPECS[specialty]
    parts = [
        f"You are the {specialty} specialist on a 5-person Python team.",
        f"Your job: write ONE Python function named `{s['name']}` with signature {s['signature']}.",
        f"Specification:\n  {s['spec']}",
        f"Example:\n  {s['example']}",
    ]
    if dependency_outputs:
        parts.append(
            "You can reference these functions written by teammates. "
            "Do NOT re-define them — they will be concatenated with yours:"
        )
        for dep, src in dependency_outputs.items():
            dep_name = FUNCTION_SPECS[dep]["name"]
            parts.append(f"  - `{dep_name}` is available. Preview:\n```python\n{src[:500]}\n```")
    parts.append(
        "Output ONLY the Python function in a ```python code fence. "
        "No other commentary. The function will be concatenated with 4 "
        "other functions and executed against an automated test suite."
    )
    return "\n\n".join(parts)


# --------------------------- Test suite ---------------------------------

# The test runner script written to disk in the sandbox. Imports from `mod`
# (the concatenated agent outputs) and emits {test_id: pass_bool} JSON.
TEST_RUNNER_SRC = r"""
import json, sys, traceback
from decimal import Decimal
from datetime import date, datetime, timedelta

results = {}

def _check(tid, fn):
    try:
        ok = bool(fn())
    except Exception as e:
        ok = False
    results[tid] = ok

try:
    import mod
except Exception as e:
    print(json.dumps({"_import_error": False}))
    sys.exit(0)

# --- Parser (0.15 weight) ---
_check("parse_basic", lambda: (
    isinstance(mod.parse_transaction("2026-01-15,DEBIT,45.20,Food"), dict)
    and mod.parse_transaction("2026-01-15,DEBIT,45.20,Food")["amount"] == Decimal("45.20")
    and mod.parse_transaction("2026-01-15,DEBIT,45.20,Food")["type"] == "DEBIT"
    and mod.parse_transaction("2026-01-15,DEBIT,45.20,Food")["category"] == "food"
    and mod.parse_transaction("2026-01-15,DEBIT,45.20,Food")["date"] == date(2026,1,15)
))
_check("parse_credit", lambda: mod.parse_transaction("2024-11-30,CREDIT,100,salary")["type"] == "CREDIT")
def _parse_rejects_malformed():
    try:
        mod.parse_transaction("not a csv line")
        return False
    except ValueError:
        return True
    except Exception:
        return False
_check("parse_rejects_malformed", _parse_rejects_malformed)

# --- Validator (0.15 weight) ---
_check("validate_ok", lambda: mod.validate_transaction(
    {"date": date(2024,1,1), "type": "DEBIT", "amount": Decimal("10"), "category": "food"}
) is True)
_check("validate_future", lambda: mod.validate_transaction(
    {"date": date(2100,1,1), "type": "DEBIT", "amount": Decimal("10"), "category": "food"}
) is False)
_check("validate_bad_type", lambda: mod.validate_transaction(
    {"date": date(2024,1,1), "type": "UNKNOWN", "amount": Decimal("10"), "category": "food"}
) is False)

# --- Aggregator (0.20 weight) ---
_check("aggregate_signs", lambda: (
    mod.aggregate_by_category([
        {"date": date(2024,1,1), "type": "DEBIT", "amount": Decimal("10"), "category": "food"},
        {"date": date(2024,1,2), "type": "CREDIT", "amount": Decimal("50"), "category": "food"},
    ]) == {"food": Decimal("40")}
))
_check("aggregate_multi_cat", lambda: (
    mod.aggregate_by_category([
        {"date": date(2024,1,1), "type": "DEBIT", "amount": Decimal("20"), "category": "rent"},
        {"date": date(2024,1,2), "type": "DEBIT", "amount": Decimal("5"), "category": "food"},
    ]) == {"rent": Decimal("-20"), "food": Decimal("-5")}
))
_check("aggregate_empty", lambda: mod.aggregate_by_category([]) == {})

# --- Detector (0.20 weight) ---
_check("detect_spike", lambda: (
    any(a["amount"] == Decimal("500") for a in mod.detect_anomalies([
        {"date": date(2024,1,1), "type": "DEBIT", "amount": Decimal("10"), "category": "food"},
        {"date": date(2024,1,2), "type": "DEBIT", "amount": Decimal("12"), "category": "food"},
        {"date": date(2024,1,3), "type": "DEBIT", "amount": Decimal("11"), "category": "food"},
        {"date": date(2024,1,4), "type": "DEBIT", "amount": Decimal("9"), "category": "food"},
        {"date": date(2024,1,5), "type": "DEBIT", "amount": Decimal("500"), "category": "food"},
    ], 1.5))
))
_check("detect_no_false_positive", lambda: (
    len(mod.detect_anomalies([
        {"date": date(2024,1,1), "type": "DEBIT", "amount": Decimal("10"), "category": "food"},
        {"date": date(2024,1,2), "type": "DEBIT", "amount": Decimal("12"), "category": "food"},
        {"date": date(2024,1,3), "type": "DEBIT", "amount": Decimal("11"), "category": "food"},
    ], 2.0)) == 0
))
_check("detect_handles_small_category", lambda: (
    isinstance(mod.detect_anomalies([
        {"date": date(2024,1,1), "type": "DEBIT", "amount": Decimal("10"), "category": "solo"},
    ], 2.0), list)
))

# --- Integrator (0.30 weight) ---
_check("integrate_basic", lambda: (
    isinstance(mod.analyze_ledger([
        "2024-01-15,DEBIT,10,food",
        "2024-01-15,CREDIT,500,salary",
    ]), dict)
))
_check("integrate_rejects_bad_line", lambda: (
    mod.analyze_ledger(["not a csv"])["parse_errors"] == 1
))
_check("integrate_composes_all", lambda: (
    # One valid line, one parseable-but-invalid (future), one parse error
    lambda r: (r["parse_errors"] == 1
               and r["rejected"] == 1
               and r["totals"].get("food", Decimal("0")) == Decimal("-10"))
)(mod.analyze_ledger([
    "2024-01-15,DEBIT,10,food",
    "2100-12-31,DEBIT,999,food",   # future date — invalid
    "bad line",                     # unparseable
])))

print(json.dumps(results))
"""


# --------------------------- Scoring ------------------------------------

TEST_WEIGHTS = {
    "parse_basic": 0.05,
    "parse_credit": 0.05,
    "parse_rejects_malformed": 0.05,
    "validate_ok": 0.05,
    "validate_future": 0.05,
    "validate_bad_type": 0.05,
    "aggregate_signs": 0.07,
    "aggregate_multi_cat": 0.07,
    "aggregate_empty": 0.06,
    "detect_spike": 0.07,
    "detect_no_false_positive": 0.07,
    "detect_handles_small_category": 0.06,
    "integrate_basic": 0.10,
    "integrate_rejects_bad_line": 0.10,
    "integrate_composes_all": 0.10,
}
# Total weight = 1.00


def score_tests(per_test: dict[str, bool]) -> dict:
    passed = [k for k, v in per_test.items() if v is True]
    failed = [k for k, v in per_test.items() if v is False]
    weighted = sum(TEST_WEIGHTS.get(k, 0) for k in passed)
    return {
        "weighted_score": round(weighted, 3),
        "n_passed": len(passed),
        "n_total": len(TEST_WEIGHTS),
        "per_test": per_test,
        "passed_tests": passed,
        "failed_tests": failed,
    }


def compose_module(agent_code: dict[str, str]) -> str:
    """Concatenate per-specialty function bodies into one module."""
    parts = [MODULE_HEADER]
    for specialty in SPEC_ORDER:
        src = agent_code.get(specialty, "")
        if src:
            parts.append(f"# === {specialty}: {FUNCTION_SPECS[specialty]['name']} ===")
            parts.append(src)
    return "\n\n".join(parts)
