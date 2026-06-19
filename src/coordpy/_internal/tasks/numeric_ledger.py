"""NumericLedger — 12-agent numerical-convention codesign task.

Same structure as Phase 14's ProtocolKit:
  - 5 producer/consumer pairs (10 agents)
  - 2 multi-producer integrators (2 agents)
  - 25 tests: 3 per pair + 5 per integrator

Difference: the coordination surface is *numerical conventions*, not
dict-key naming. Each pair hides a convention — rounding mode, scale unit,
NaN policy, overflow policy, signed-integer encoding — and pair tests only
pass when producer and consumer chose the same convention. Integrator tests
round-trip through 3 producer conventions at once.

This is the phase-17 generality probe. If CASR's advantage persists here
with the same harness (bar the trigger, which has to sense convention
drift rather than dict-key drift), the mechanism is not ProtocolKit-
specific.
"""

from __future__ import annotations


# --------------------------- Module header --------------------------------

MODULE_HEADER = """\
\"\"\"NumericLedger — a 12-function collaborative module.
Each pair of agents must agree on a hidden numerical convention
(rounding, scale, NaN, overflow, signed encoding). Pair and integrator
tests only pass when conventions match.
\"\"\"
from __future__ import annotations
import math
"""


# --------------------------- Catalog --------------------------------------

FUNCTION_SPECS: dict[str, dict] = {

    # ---- Pair 1: rounding convention -------------------------------------
    "round_amount": {
        "name": "round_amount",
        "signature": "(value: float, decimals: int) -> float",
        "spec": (
            "Round `value` to `decimals` places. THE EXACT ROUNDING RULE "
            "IS YOUR CHOICE — half-up, banker's rounding (half-to-even), "
            "floor-half, ceil-half, or any other plausible convention. "
            "The pair consumer `check_rounded` will test whether `r == "
            "round_amount(v, d)` holds under whatever rule you pick; stay "
            "internally consistent. Also used by the integrator "
            "`daily_total`."
        ),
        "example": (
            "round_amount(0.5, 0)   # may return 0.0 (banker's) or 1.0 "
            "(half-up) — your pick"
        ),
    },
    "check_rounded": {
        "name": "check_rounded",
        "signature": "(value: float, rounded: float, decimals: int) -> bool",
        "spec": (
            "Return True iff `rounded` is EQUAL to the value that "
            "`round_amount(value, decimals)` would return under the "
            "shared convention. THE CONVENTION IS DEFINED BY THE "
            "PRODUCER `round_amount`. If drafting independently with no "
            "teammate code visible, pick a natural default rule. IF YOU "
            "CAN SEE `round_amount`'s CODE in this prompt, mirror its "
            "exact rule. Equality tolerance 1e-9."
        ),
        "example": (
            "check_rounded(0.5, 0.0, 0) -> True under banker's rounding"
        ),
    },

    # ---- Pair 2: scale convention ---------------------------------------
    "to_ledger": {
        "name": "to_ledger",
        "signature": "(dollars: float) -> int",
        "spec": (
            "Convert a dollar amount to an integer ledger unit. THE UNIT "
            "IS YOUR CHOICE — cents (×100), mils (×1000), tenth-cents "
            "(×1000), whatever. Must be reversible by the pair consumer "
            "`from_ledger` so that `from_ledger(to_ledger(x)) == x` "
            "within 1e-6 tolerance. Also used by integrator "
            "`daily_total`."
        ),
        "example": (
            "to_ledger(1.50) -> 150 (cents) or 1500 (mils) — your pick"
        ),
    },
    "from_ledger": {
        "name": "from_ledger",
        "signature": "(units: int) -> float",
        "spec": (
            "Convert ledger units back to dollars. THE UNIT IS DEFINED "
            "BY THE PRODUCER `to_ledger`. If drafting independently, "
            "pick a natural default (cents is most common). IF YOU CAN "
            "SEE `to_ledger`'s CODE, mirror its exact multiplier."
        ),
        "example": "from_ledger(150) -> 1.50  (if producer used cents)",
    },

    # ---- Pair 3: NaN policy ---------------------------------------------
    "reduce_amounts": {
        "name": "reduce_amounts",
        "signature": "(values: list) -> float",
        "spec": (
            "Sum a list of floats that may contain NaN. THE NaN POLICY "
            "IS YOUR CHOICE: propagate (any NaN -> NaN), skip (filter "
            "NaN before summing), or zero (treat NaN as 0). Your pair "
            "consumer `is_valid_reduction` must be able to verify the "
            "result under the same policy. An empty list must return 0.0. "
            "Also used by integrator `daily_total`."
        ),
        "example": (
            "reduce_amounts([1.0, float('nan'), 2.0])   # NaN, 3.0, or 3.0"
        ),
    },
    "is_valid_reduction": {
        "name": "is_valid_reduction",
        "signature": "(values: list, result: float) -> bool",
        "spec": (
            "Return True iff `result` is a valid reduction of `values` "
            "under the NaN policy fixed by `reduce_amounts`. Treat NaN "
            "equality carefully: if your policy is propagate, then "
            "result is valid iff `math.isnan(result)` holds when any "
            "input is NaN. If skip, result must equal the sum of non-NaN "
            "inputs within 1e-6. If zero, result is the sum with NaNs "
            "treated as 0. IF YOU CAN SEE `reduce_amounts`'s CODE, "
            "mirror its policy."
        ),
        "example": (
            "is_valid_reduction([1.0, float('nan'), 2.0], 3.0) -> True "
            "under skip policy"
        ),
    },

    # ---- Pair 4: overflow convention ------------------------------------
    "add_capped": {
        "name": "add_capped",
        "signature": "(a: int, b: int, cap: int) -> int",
        "spec": (
            "Add two ints, handling overflow at ±`cap`. THE OVERFLOW "
            "POLICY IS YOUR CHOICE: saturate (clamp result to [-cap, "
            "cap]) or wrap (use modular arithmetic mod 2*cap around 0, "
            "so 101 with cap=100 becomes -99). `cap > 0`. Pair consumer "
            "`predict_overflow` must predict the same result. Also "
            "used by integrator `settle_position`."
        ),
        "example": (
            "add_capped(99, 50, 100) -> 100 (saturate) or -51 (wrap)"
        ),
    },
    "predict_overflow": {
        "name": "predict_overflow",
        "signature": "(a: int, b: int, cap: int) -> int",
        "spec": (
            "Predict exactly what `add_capped(a, b, cap)` returns. THE "
            "OVERFLOW POLICY IS DEFINED BY THE PRODUCER `add_capped`. "
            "If drafting independently, pick a natural default. IF YOU "
            "CAN SEE `add_capped`'s CODE, mirror it."
        ),
        "example": "predict_overflow(99, 50, 100) -> 100 (saturate)",
    },

    # ---- Pair 5: signed encoding ----------------------------------------
    "encode_signed": {
        "name": "encode_signed",
        "signature": "(value: int, bits: int) -> int",
        "spec": (
            "Encode a signed integer `value` into a `bits`-bit "
            "representation, returned as a NON-NEGATIVE int in "
            "[0, 2**bits). THE REPRESENTATION IS YOUR CHOICE: two's "
            "complement, sign-magnitude, excess-(2**(bits-1)), etc. "
            "`bits >= 2`. Pair consumer `decode_signed` must invert. "
            "Also used by integrator `settle_position`."
        ),
        "example": (
            "encode_signed(-1, 8)   # 255 (two's complement) or "
            "129 (sign-magnitude)"
        ),
    },
    "decode_signed": {
        "name": "decode_signed",
        "signature": "(raw: int, bits: int) -> int",
        "spec": (
            "Decode a `bits`-bit signed-integer representation produced "
            "by `encode_signed`. THE REPRESENTATION IS DEFINED BY THE "
            "PRODUCER. If drafting independently, pick a natural "
            "default (two's complement is most common). IF YOU CAN SEE "
            "`encode_signed`'s CODE, mirror it."
        ),
        "example": "decode_signed(255, 8) -> -1 (under two's complement)",
    },

    # ---- Integrator 1: daily_total --------------------------------------
    "daily_total": {
        "name": "daily_total",
        "signature": "(entries: list) -> int",
        "spec": (
            "Each element of `entries` is a list of floats (possibly "
            "containing NaN). Process them as: for each entry, call "
            "reduce_amounts to collapse to one float, then round_amount "
            "with decimals=2, then to_ledger to convert to integer "
            "ledger units; sum the integer results. YOU MUST CALL the "
            "three named functions — do not reimplement them. Empty "
            "entries ([] is a valid entry) must contribute whatever "
            "reduce_amounts([]) returns, rounded and scaled. If any "
            "entry's reduction is NaN under the shared policy, skip "
            "that entry (contribute 0)."
        ),
        "example": (
            "daily_total([[1.0, 2.0], [3.0]]) -> 600 (cents) "
            "under skip+half-up+cents conventions"
        ),
    },

    # ---- Integrator 2: settle_position ----------------------------------
    "settle_position": {
        "name": "settle_position",
        "signature": "(base: int, delta: int, cap: int, bits: int) -> int",
        "spec": (
            "Apply `delta` to `base` via `add_capped(base, delta, cap)`, "
            "then encode the result via `encode_signed(result, bits)`. "
            "Return the encoded non-negative int. YOU MUST CALL the two "
            "named functions — do not reimplement them."
        ),
        "example": (
            "settle_position(10, 5, 100, 8) -> encode_signed(15, 8)"
        ),
    },
}


SPEC_ORDER: list[str] = [
    # producers (tier 0)
    "round_amount", "to_ledger", "reduce_amounts",
    "add_capped", "encode_signed",
    # consumers (tier 1)
    "check_rounded", "from_ledger", "is_valid_reduction",
    "predict_overflow", "decode_signed",
    # integrators (tier 1)
    "daily_total", "settle_position",
]


CALL_GRAPH: dict[str, list[str]] = {
    "round_amount":       [],
    "to_ledger":          [],
    "reduce_amounts":     [],
    "add_capped":         [],
    "encode_signed":      [],
    "check_rounded":      ["round_amount"],
    "from_ledger":        ["to_ledger"],
    "is_valid_reduction": ["reduce_amounts"],
    "predict_overflow":   ["add_capped"],
    "decode_signed":      ["encode_signed"],
    "daily_total":        ["reduce_amounts", "round_amount", "to_ledger"],
    "settle_position":    ["add_capped", "encode_signed"],
}


# --------------------------- Agent prompt --------------------------------

def agent_prompt(specialty: str, dependency_outputs: dict[str, str] | None = None) -> str:
    s = FUNCTION_SPECS[specialty]
    parts = [
        f"You are the {specialty} specialist on a 12-person Python team "
        f"building NumericLedger. The task DELIBERATELY leaves each pair's "
        f"NUMERICAL CONVENTION (rounding mode, scale unit, NaN policy, "
        f"overflow policy, signed encoding) underspecified — the producer "
        f"picks a convention, the consumer must match it.",
        f"Write ONE Python function named `{s['name']}` with signature "
        f"{s['signature']}.",
        f"Specification:\n  {s['spec']}",
        f"Example:\n  {s['example']}",
        "The module header already imports `math`; you may use it.",
    ]
    if dependency_outputs:
        parts.append(
            "You can reference these teammate functions. Do NOT re-define "
            "them — they will be concatenated with yours. MIRROR THEIR "
            "EXACT NUMERICAL CONVENTIONS:"
        )
        for dep, src in dependency_outputs.items():
            dep_name = FUNCTION_SPECS[dep]["name"]
            parts.append(
                f"  - `{dep_name}` is available. Preview:\n"
                f"```python\n{src[:500]}\n```"
            )
    parts.append(
        "Output ONLY the Python function in a ```python code fence. No "
        "other commentary. Your function will be concatenated with 11 "
        "other functions and executed against an automated test suite "
        "whose pair and integrator tests only pass if your convention "
        "matches your teammate's."
    )
    return "\n\n".join(parts)


# --------------------------- Module composition --------------------------

def compose_module(agent_code: dict[str, str]) -> str:
    parts = [MODULE_HEADER]
    for specialty in SPEC_ORDER:
        src = agent_code.get(specialty, "")
        if src:
            parts.append(f"# === {specialty}: {FUNCTION_SPECS[specialty]['name']} ===")
            parts.append(src)
    return "\n\n".join(parts)


# --------------------------- Test suite ---------------------------------

TEST_RUNNER_SRC = r"""
import json, sys, math

results = {}

def _check(tid, fn):
    try:
        ok = bool(fn())
    except Exception:
        ok = False
    results[tid] = ok

try:
    import mod
except Exception:
    print(json.dumps({"_import_error": False}))
    sys.exit(0)

# ================ Pair 1: rounding (3) ================
_check("round_half", lambda:
    mod.check_rounded(0.5, mod.round_amount(0.5, 0), 0) is True)
_check("round_one_half", lambda:
    mod.check_rounded(1.5, mod.round_amount(1.5, 0), 0) is True)
_check("round_neg_half", lambda:
    mod.check_rounded(-0.5, mod.round_amount(-0.5, 0), 0) is True)

# ================ Pair 2: scale (3) ================
_check("scale_dollar", lambda:
    abs(mod.from_ledger(mod.to_ledger(1.50)) - 1.50) < 1e-6)
_check("scale_hundred", lambda:
    abs(mod.from_ledger(mod.to_ledger(100.00)) - 100.00) < 1e-6)
_check("scale_small", lambda:
    abs(mod.from_ledger(mod.to_ledger(0.01)) - 0.01) < 1e-6)

# ================ Pair 3: NaN policy (3) ================
_check("nan_none", lambda:
    mod.is_valid_reduction([1.0, 2.0, 3.0], mod.reduce_amounts([1.0, 2.0, 3.0])) is True)
_check("nan_mixed", lambda:
    mod.is_valid_reduction([1.0, float('nan'), 2.0], mod.reduce_amounts([1.0, float('nan'), 2.0])) is True)
_check("nan_empty", lambda:
    mod.is_valid_reduction([], mod.reduce_amounts([])) is True)

# ================ Pair 4: overflow (3) ================
_check("over_none", lambda:
    mod.add_capped(5, 3, 100) == mod.predict_overflow(5, 3, 100))
_check("over_positive", lambda:
    mod.add_capped(99, 50, 100) == mod.predict_overflow(99, 50, 100))
_check("over_negative", lambda:
    mod.add_capped(-99, -50, 100) == mod.predict_overflow(-99, -50, 100))

# ================ Pair 5: signed (3) ================
_check("sign_pos", lambda:
    mod.decode_signed(mod.encode_signed(5, 8), 8) == 5)
_check("sign_neg", lambda:
    mod.decode_signed(mod.encode_signed(-5, 8), 8) == -5)
_check("sign_edge", lambda:
    mod.decode_signed(mod.encode_signed(-127, 8), 8) == -127)

# ================ Integrator: daily_total (5) ================
def _dt_basic():
    r = mod.daily_total([[1.0, 2.0], [3.0]])
    # Should be 6 dollars converted to ledger units; convention-dependent
    # but consumer `from_ledger` inverts it cleanly for the shared policy.
    return abs(mod.from_ledger(r) - 6.0) < 1e-6
_check("dt_basic", _dt_basic)

def _dt_empty_entry():
    r = mod.daily_total([[], [1.0]])
    return abs(mod.from_ledger(r) - 1.0) < 1e-6
_check("dt_empty_entry", _dt_empty_entry)

def _dt_nan_skip_or_zero():
    # Under skip or zero policy, [[1,NaN,2]] contributes 3.0 (skip) or 3.0
    # (zero). Under propagate the entry is NaN; integrator spec says
    # "skip that entry (contribute 0)". Both legal outcomes → the
    # integrator's own post-processing masks the convention choice.
    r = mod.daily_total([[1.0, float('nan'), 2.0]])
    # `from_ledger(r)` should be either 3.0 (skip) or 0.0 (propagate-skip)
    v = mod.from_ledger(r)
    return abs(v - 3.0) < 1e-6 or abs(v - 0.0) < 1e-6
_check("dt_nan_ok", _dt_nan_skip_or_zero)

def _dt_sum_of_entries():
    r = mod.daily_total([[1.0], [2.0], [3.0]])
    return abs(mod.from_ledger(r) - 6.0) < 1e-6
_check("dt_sum_entries", _dt_sum_of_entries)

def _dt_rounds():
    # [0.5, 0.5] sums to 1.0 exactly (convention-agnostic). integer ledger
    # of 1.0 is convention-dependent, but from_ledger inverts cleanly.
    r = mod.daily_total([[0.5, 0.5]])
    return abs(mod.from_ledger(r) - 1.0) < 1e-6
_check("dt_rounds", _dt_rounds)

# ================ Integrator: settle_position (5) ================
def _sp_within():
    # base+delta well inside cap -> no overflow; decode should give base+delta
    r = mod.settle_position(10, 5, 100, 8)
    return mod.decode_signed(r, 8) == 15
_check("sp_within", _sp_within)

def _sp_overflow_positive():
    # base=80, delta=50, cap=100 → result should be predict_overflow(80, 50, 100)
    r = mod.settle_position(80, 50, 100, 8)
    return mod.decode_signed(r, 8) == mod.predict_overflow(80, 50, 100)
_check("sp_overflow_positive", _sp_overflow_positive)

def _sp_overflow_negative():
    r = mod.settle_position(-80, -50, 100, 8)
    return mod.decode_signed(r, 8) == mod.predict_overflow(-80, -50, 100)
_check("sp_overflow_negative", _sp_overflow_negative)

def _sp_negative():
    r = mod.settle_position(-10, 5, 100, 8)
    return mod.decode_signed(r, 8) == -5
_check("sp_negative", _sp_negative)

def _sp_zero():
    r = mod.settle_position(0, 0, 100, 8)
    return mod.decode_signed(r, 8) == 0
_check("sp_zero", _sp_zero)

print(json.dumps(results))
"""


# --------------------------- Scoring -----------------------------------

_TEST_NAMES = [
    "round_half", "round_one_half", "round_neg_half",
    "scale_dollar", "scale_hundred", "scale_small",
    "nan_none", "nan_mixed", "nan_empty",
    "over_none", "over_positive", "over_negative",
    "sign_pos", "sign_neg", "sign_edge",
    "dt_basic", "dt_empty_entry", "dt_nan_ok",
    "dt_sum_entries", "dt_rounds",
    "sp_within", "sp_overflow_positive", "sp_overflow_negative",
    "sp_negative", "sp_zero",
]
assert len(_TEST_NAMES) == 25, len(_TEST_NAMES)

TEST_WEIGHTS = {t: 1.0 / len(_TEST_NAMES) for t in _TEST_NAMES}
assert abs(sum(TEST_WEIGHTS.values()) - 1.0) < 1e-9


# Per-pair test groupings — used by C7 (convention-propagation analysis).
PAIR_TESTS: dict[str, list[str]] = {
    "rounding":  ["round_half", "round_one_half", "round_neg_half"],
    "scale":     ["scale_dollar", "scale_hundred", "scale_small"],
    "nan":       ["nan_none", "nan_mixed", "nan_empty"],
    "overflow":  ["over_none", "over_positive", "over_negative"],
    "signed":    ["sign_pos", "sign_neg", "sign_edge"],
}
INTEGRATOR_TESTS: dict[str, list[str]] = {
    "daily_total": ["dt_basic", "dt_empty_entry", "dt_nan_ok",
                    "dt_sum_entries", "dt_rounds"],
    "settle_position": ["sp_within", "sp_overflow_positive",
                         "sp_overflow_negative", "sp_negative", "sp_zero"],
}


def score_tests(per_test: dict[str, bool]) -> dict:
    passed = [k for k, v in per_test.items() if v is True]
    failed = [k for k, v in per_test.items() if v is False]
    weighted = sum(TEST_WEIGHTS.get(k, 0) for k in passed)
    return {
        "weighted_score": round(weighted, 4),
        "n_passed": len(passed),
        "n_total": len(TEST_WEIGHTS),
        "per_test": per_test,
        "passed_tests": passed,
        "failed_tests": failed,
    }


# --------------------------- Reference solution ------------------------

# All agents share one convention:
#   rounding = half-up (math.floor(value + 0.5))
#   scale    = cents (×100)
#   nan      = skip
#   overflow = saturate
#   signed   = two's complement

_REF: dict[str, str] = {
    "round_amount": '''\
def round_amount(value: float, decimals: int) -> float:
    mult = 10 ** decimals
    shifted = value * mult
    if shifted >= 0:
        rounded = math.floor(shifted + 0.5)
    else:
        rounded = -math.floor(-shifted + 0.5)
    return rounded / mult
''',
    "check_rounded": '''\
def check_rounded(value: float, rounded: float, decimals: int) -> bool:
    mult = 10 ** decimals
    shifted = value * mult
    if shifted >= 0:
        expected = math.floor(shifted + 0.5)
    else:
        expected = -math.floor(-shifted + 0.5)
    expected = expected / mult
    return abs(rounded - expected) < 1e-9
''',
    "to_ledger": '''\
def to_ledger(dollars: float) -> int:
    # cents, half-up
    x = dollars * 100.0
    if x >= 0:
        return int(math.floor(x + 0.5))
    return -int(math.floor(-x + 0.5))
''',
    "from_ledger": '''\
def from_ledger(units: int) -> float:
    return units / 100.0
''',
    "reduce_amounts": '''\
def reduce_amounts(values: list) -> float:
    total = 0.0
    for v in values:
        if isinstance(v, float) and math.isnan(v):
            continue        # skip policy
        total += float(v)
    return total
''',
    "is_valid_reduction": '''\
def is_valid_reduction(values: list, result: float) -> bool:
    expected = 0.0
    for v in values:
        if isinstance(v, float) and math.isnan(v):
            continue
        expected += float(v)
    return abs(result - expected) < 1e-6
''',
    "add_capped": '''\
def add_capped(a: int, b: int, cap: int) -> int:
    s = a + b
    if s > cap:
        return cap
    if s < -cap:
        return -cap
    return s
''',
    "predict_overflow": '''\
def predict_overflow(a: int, b: int, cap: int) -> int:
    s = a + b
    if s > cap:
        return cap
    if s < -cap:
        return -cap
    return s
''',
    "encode_signed": '''\
def encode_signed(value: int, bits: int) -> int:
    mod = 1 << bits
    return value % mod
''',
    "decode_signed": '''\
def decode_signed(raw: int, bits: int) -> int:
    limit = 1 << (bits - 1)
    if raw >= limit:
        return raw - (1 << bits)
    return raw
''',
    "daily_total": '''\
def daily_total(entries: list) -> int:
    total = 0
    for entry in entries:
        r = reduce_amounts(entry)
        if isinstance(r, float) and math.isnan(r):
            continue
        rounded = round_amount(r, 2)
        total += to_ledger(rounded)
    return total
''',
    "settle_position": '''\
def settle_position(base: int, delta: int, cap: int, bits: int) -> int:
    result = add_capped(base, delta, cap)
    return encode_signed(result, bits)
''',
}


# --------------------------- Main: verify reference -------------------

if __name__ == "__main__":
    import sys
    from coordpy._internal.core.code_harness import run_sandboxed

    module = compose_module(_REF)
    result = run_sandboxed(module, TEST_RUNNER_SRC, timeout_s=20)
    score = score_tests(result.per_test)
    print(f"{score['n_passed']}/{score['n_total']} reference passes "
          f"(weighted {score['weighted_score']})")
    if score["n_passed"] != len(TEST_WEIGHTS):
        print("FAILED TESTS:")
        for tid, ok in result.per_test.items():
            if not ok:
                print(f"  - {tid}")
        print("\nSTDERR:")
        print(result.stderr)
        print("\nSTDOUT:")
        print(result.stdout)
        sys.exit(1)
    else:
        print("All reference tests pass.")
