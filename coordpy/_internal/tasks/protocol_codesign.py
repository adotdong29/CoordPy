"""ProtocolKit — a 12-agent codesign task with underspecified schemas.

Unlike `library_v2.py` (TinyStore), where every function could be
implemented correctly in isolation given its spec, ProtocolKit
DELIBERATELY underspecifies each producer-consumer pair: the exact
shape of the shared dict (field names, encoding) is left for the
producer to decide. A consumer can only work correctly if it sees the
producer's actual draft (round 2 with CASR) — otherwise it guesses a
different convention and the pair tests fail.

Why this task exists: the Phase-12 benchmark (RESULTS_PHASE12.md)
showed CASR matched random routing on token savings because the
underlying task was independently solvable. Here, round 2 *must*
matter: causal routing picks the right producer preview for each
consumer, random routing almost never does.

Sub-groups (5 pairs + 2 integrators):
  Pair A  — event header:   make_event_header + read_event_kind
  Pair B  — success frame:  wrap_ok + is_ok
  Pair C  — error frame:    make_error + get_error_code
  Pair D  — range spec:     make_range + range_contains
  Pair E  — page token:     make_page_token + parse_page_token
  Integrators: process_event (calls 3 producers),
               query_page    (calls 2 producers).

API mirrors library_v2.py exactly — same exported names, same
`score_tests` return shape, same `compose_module` / `agent_prompt`
signatures.
"""

from __future__ import annotations


# --------------------------- Specification ------------------------------

MODULE_HEADER = """\
\"\"\"ProtocolKit — a 12-function collaborative module.
Each function is written by a different LLM agent; producer/consumer
pairs must agree on a private dict schema for end-to-end tests to pass.
\"\"\"
from __future__ import annotations
import base64
"""


FUNCTION_SPECS = {
    # --- producers ---
    "make_event_header": {
        "name": "make_event_header",
        "signature": "(kind: str, ts: int) -> dict",
        "spec": (
            "Build and return a dict representing an event header that "
            "carries the event `kind` (a string like 'login') and the "
            "unix-second timestamp `ts`. THE EXACT FIELD NAMES ARE YOUR "
            "CHOICE — pick any plausible convention. Your teammate who "
            "writes `read_event_kind` will read from whatever dict you "
            "return, so make it internally consistent with what a "
            "reader would expect. Two integrators (process_event) will "
            "also embed your dict under the key 'header' of a larger "
            "record."
        ),
        "example": (
            "make_event_header('login', 1000) -> "
            "{'kind': 'login', 'ts': 1000}   # one valid convention"
        ),
    },
    "wrap_ok": {
        "name": "wrap_ok",
        "signature": "(value) -> dict",
        "spec": (
            "Wrap any Python value into a SUCCESS envelope dict that can "
            "later be distinguished from an error frame. INTERNAL KEYS "
            "ARE UP TO YOU — pick a convention a reader would naturally "
            "expect. Your teammate `is_ok` will read from whatever you "
            "return. The envelope must be a dict (not a tuple/list) so "
            "that integrators can store it under the key 'body' of a "
            "larger record."
        ),
        "example": (
            "wrap_ok(42) -> {'ok': True, 'value': 42}   # one valid "
            "convention"
        ),
    },
    "make_error": {
        "name": "make_error",
        "signature": "(code: int, msg: str) -> dict",
        "spec": (
            "Build an error frame dict from an integer `code` and string "
            "`msg`. The dict must be distinguishable from a success "
            "envelope (produced by `wrap_ok`). INTERNAL FIELD NAMES ARE "
            "YOUR CHOICE — your teammate `get_error_code` will read the "
            "integer code from whatever you return. Also used by the "
            "integrator `process_event` when `ok` is False."
        ),
        "example": (
            "make_error(404, 'not found') -> "
            "{'err': True, 'code': 404, 'msg': 'not found'}   # one "
            "valid convention"
        ),
    },
    "make_range": {
        "name": "make_range",
        "signature": "(start: int, end: int) -> dict",
        "spec": (
            "Build a dict representing an INCLUSIVE-START, EXCLUSIVE-END "
            "integer range [start, end). FIELD NAMES ARE YOUR CHOICE. "
            "Your teammate `range_contains` will read back `start` and "
            "`end` from whatever dict you return to test membership. "
            "Also used by the integrator `query_page`."
        ),
        "example": (
            "make_range(0, 10) -> {'lo': 0, 'hi': 10}   # one valid "
            "convention"
        ),
    },
    "make_page_token": {
        "name": "make_page_token",
        "signature": "(offset: int, limit: int) -> str",
        "spec": (
            "Encode the pair (offset, limit) into a URL-safe STRING "
            "token of your design (base64, colon-separated, JSON, "
            "whatever). Must round-trip exactly through your teammate "
            "`parse_page_token`. The `base64` module is pre-imported in "
            "the module header if you want to use it. Also used by the "
            "integrator `query_page`."
        ),
        "example": (
            "make_page_token(100, 20) -> 'MTAwOjIw'   # one valid "
            "base64 convention"
        ),
    },
    # --- consumers ---
    "read_event_kind": {
        "name": "read_event_kind",
        "signature": "(header: dict) -> str",
        "spec": (
            "Given a header dict produced by `make_event_header`, "
            "return the event kind as a string. THE EXACT INPUT SHAPE "
            "IS DEFINED BY THE PRODUCER `make_event_header`. If you "
            "are drafting independently with NO teammate code visible, "
            "make your best guess at a common convention (e.g. a 'kind' "
            "key, or 'type', or 'event'). IF YOU CAN SEE THE PRODUCER'S "
            "CODE IN THIS PROMPT, use ITS exact field name — that is "
            "the canonical truth."
        ),
        "example": (
            "read_event_kind({'kind': 'login', 'ts': 1000}) -> 'login'"
        ),
    },
    "is_ok": {
        "name": "is_ok",
        "signature": "(envelope: dict) -> bool",
        "spec": (
            "Return True iff `envelope` is a success envelope produced "
            "by `wrap_ok` (as opposed to an error frame produced by "
            "`make_error`). THE EXACT INPUT SHAPE IS DEFINED BY THE "
            "PRODUCER `wrap_ok`. If drafting independently with no "
            "teammate code visible, guess a common convention (e.g. an "
            "'ok' boolean key, or the absence of an 'err'/'error' key). "
            "IF YOU CAN SEE `wrap_ok`'s CODE, mirror its exact "
            "convention."
        ),
        "example": (
            "is_ok({'ok': True, 'value': 42}) -> True"
        ),
    },
    "get_error_code": {
        "name": "get_error_code",
        "signature": "(frame: dict) -> int",
        "spec": (
            "Extract the integer error code from a frame built by "
            "`make_error`. THE EXACT INPUT SHAPE IS DEFINED BY THE "
            "PRODUCER `make_error`. If drafting independently with no "
            "teammate code visible, guess a common convention (a 'code' "
            "key is the usual one). IF YOU CAN SEE `make_error`'s "
            "CODE, mirror its exact key."
        ),
        "example": (
            "get_error_code({'err': True, 'code': 404, 'msg': 'nf'}) "
            "-> 404"
        ),
    },
    "range_contains": {
        "name": "range_contains",
        "signature": "(rng: dict, x: int) -> bool",
        "spec": (
            "Return True iff the integer `x` is in the range "
            "[rng.start, rng.end) using whatever dict shape "
            "`make_range` returned. THE EXACT INPUT SHAPE IS DEFINED "
            "BY THE PRODUCER `make_range`. If drafting independently "
            "with no teammate code visible, guess a common convention "
            "(e.g. keys 'start'/'end', 'lo'/'hi', or 'from'/'to'). IF "
            "YOU CAN SEE `make_range`'s CODE, mirror its exact keys."
        ),
        "example": (
            "range_contains({'lo': 0, 'hi': 10}, 5) -> True"
        ),
    },
    "parse_page_token": {
        "name": "parse_page_token",
        "signature": "(token: str) -> tuple",
        "spec": (
            "Decode a string token produced by `make_page_token` into "
            "a tuple (offset, limit). THE EXACT ENCODING IS DEFINED "
            "BY THE PRODUCER `make_page_token`. If drafting "
            "independently with no teammate code visible, guess a "
            "common convention (base64 of 'offset:limit', JSON, a "
            "hyphen-separated pair). IF YOU CAN SEE `make_page_token`'s "
            "CODE, use ITS exact inverse."
        ),
        "example": (
            "parse_page_token('MTAwOjIw') -> (100, 20)"
        ),
    },
    # --- integrators ---
    "process_event": {
        "name": "process_event",
        "signature": (
            "(kind: str, ts: int, ok: bool, err_code: int = 0, "
            "err_msg: str = \"\") -> dict"
        ),
        "spec": (
            "Build a composite event record. If `ok` is True, return "
            "{'header': make_event_header(kind, ts), "
            "'body': wrap_ok(None)}. If `ok` is False, return "
            "{'header': make_event_header(kind, ts), "
            "'body': make_error(err_code, err_msg)}. YOU MUST CALL the "
            "three named functions `make_event_header`, `wrap_ok`, and "
            "`make_error` — do not reimplement their payload shapes. "
            "If teammate code for those producers is visible, use "
            "their exact signatures."
        ),
        "example": (
            "process_event('login', 1000, True) -> "
            "{'header': <header-dict>, 'body': <ok-envelope>}"
        ),
    },
    "query_page": {
        "name": "query_page",
        "signature": (
            "(start: int, end: int, offset: int, limit: int) -> dict"
        ),
        "spec": (
            "Return {'range': make_range(start, end), "
            "'token': make_page_token(offset, limit)}. YOU MUST CALL "
            "`make_range` and `make_page_token` by name — do not "
            "reimplement either. If teammate code for those producers "
            "is visible, use their exact signatures."
        ),
        "example": (
            "query_page(0, 100, 25, 10) -> "
            "{'range': <range-dict>, 'token': <page-token-string>}"
        ),
    },
}


# Topologically ordered: producers first, then consumers, then integrators.
SPEC_ORDER = [
    "make_event_header",
    "wrap_ok",
    "make_error",
    "make_range",
    "make_page_token",
    "read_event_kind",
    "is_ok",
    "get_error_code",
    "range_contains",
    "parse_page_token",
    "process_event",
    "query_page",
]


CALL_GRAPH = {
    "make_event_header": [],
    "read_event_kind":   ["make_event_header"],
    "wrap_ok":           [],
    "is_ok":             ["wrap_ok"],
    "make_error":        [],
    "get_error_code":    ["make_error"],
    "make_range":        [],
    "range_contains":    ["make_range"],
    "make_page_token":   [],
    "parse_page_token":  ["make_page_token"],
    "process_event":     ["make_event_header", "wrap_ok", "make_error"],
    "query_page":        ["make_range", "make_page_token"],
}


def agent_prompt(specialty: str, dependency_outputs: dict[str, str] = None) -> str:
    """Build the LLM prompt for one specialty agent."""
    s = FUNCTION_SPECS[specialty]
    parts = [
        f"You are the {specialty} specialist on a 12-person Python team "
        f"building a collaborative protocol kit called ProtocolKit. "
        f"The task deliberately leaves the exact dict field names for "
        f"each producer/consumer pair UNDERSPECIFIED — the producer "
        f"picks a convention, the consumer must match it.",
        f"Your job: write ONE Python function named `{s['name']}` with "
        f"signature {s['signature']}.",
        f"Specification:\n  {s['spec']}",
        f"Example:\n  {s['example']}",
        "The module header already imports `base64`; you may use it.",
    ]
    if dependency_outputs:
        parts.append(
            "You can reference these functions written by teammates. "
            "Do NOT re-define them — they will be concatenated with "
            "yours. USE THEIR EXACT FIELD NAMES / ENCODINGS:"
        )
        for dep, src in dependency_outputs.items():
            dep_name = FUNCTION_SPECS[dep]["name"]
            parts.append(
                f"  - `{dep_name}` is available. Preview:\n"
                f"```python\n{src[:500]}\n```"
            )
    parts.append(
        "Output ONLY the Python function in a ```python code fence. "
        "No other commentary. The function will be concatenated with "
        "11 other functions and executed against an automated test "
        "suite whose pair tests only pass if your convention matches "
        "your teammate's."
    )
    return "\n\n".join(parts)


# --------------------------- Test suite ---------------------------------

TEST_RUNNER_SRC = r"""
import json, sys, traceback

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

# --- Pair A: event header (5) ---
_check("ev_roundtrip_login", lambda: (
    mod.read_event_kind(mod.make_event_header("login", 1000)) == "login"
))
_check("ev_roundtrip_click", lambda: (
    mod.read_event_kind(mod.make_event_header("click", 2000)) == "click"
))
_check("ev_header_is_dict", lambda: (
    isinstance(mod.make_event_header("x", 0), dict)
))
_check("ev_header_has_ts_somewhere", lambda: (
    12345 in mod.make_event_header("x", 12345).values()
))
def _ev_both_kinds():
    h1 = mod.make_event_header("a", 1)
    h2 = mod.make_event_header("b", 2)
    return (mod.read_event_kind(h1) == "a"
            and mod.read_event_kind(h2) == "b")
_check("ev_consumer_handles_both_kinds", _ev_both_kinds)

# --- Pair B: success envelope (3) ---
_check("ok_roundtrip", lambda: mod.is_ok(mod.wrap_ok(42)) is True)
_check("ok_string_value", lambda: mod.is_ok(mod.wrap_ok("hello")) is True)
_check("ok_none_value", lambda: mod.is_ok(mod.wrap_ok(None)) is True)

# --- Pair C: error frame (3) ---
_check("err_roundtrip_code", lambda: (
    mod.get_error_code(mod.make_error(404, "not found")) == 404
))
_check("err_roundtrip_code_500", lambda: (
    mod.get_error_code(mod.make_error(500, "boom")) == 500
))
_check("err_frame_is_dict", lambda: (
    isinstance(mod.make_error(1, "x"), dict)
))

# --- Pair D: range spec (3) ---
_check("range_contains_inside", lambda: (
    mod.range_contains(mod.make_range(0, 10), 5) is True
))
_check("range_contains_exclusive_end", lambda: (
    mod.range_contains(mod.make_range(0, 10), 10) is False
))
_check("range_contains_before_start", lambda: (
    mod.range_contains(mod.make_range(5, 10), 3) is False
))

# --- Pair E: page token (3) ---
_check("page_token_roundtrip", lambda: (
    tuple(mod.parse_page_token(mod.make_page_token(100, 20))) == (100, 20)
))
_check("page_token_zero_offset", lambda: (
    tuple(mod.parse_page_token(mod.make_page_token(0, 50))) == (0, 50)
))
_check("page_token_is_string", lambda: (
    isinstance(mod.make_page_token(10, 10), str)
))

# --- Integrator: process_event (4) ---
def _pe_ok():
    r = mod.process_event("login", 1000, True)
    return (isinstance(r, dict)
            and "header" in r and "body" in r
            and mod.read_event_kind(r["header"]) == "login"
            and mod.is_ok(r["body"]) is True)
_check("process_event_ok", _pe_ok)

def _pe_err():
    r = mod.process_event("fail", 999, False, 500, "boom")
    return mod.get_error_code(r["body"]) == 500
_check("process_event_err", _pe_err)

def _pe_kinds():
    r1 = mod.process_event("a", 1, True)
    r2 = mod.process_event("b", 2, True)
    return (mod.read_event_kind(r1["header"]) == "a"
            and mod.read_event_kind(r2["header"]) == "b")
_check("process_event_preserves_kind", _pe_kinds)

def _pe_structure():
    r = mod.process_event("x", 0, True)
    return isinstance(r, dict) and set(r.keys()) == {"header", "body"}
_check("process_event_structure", _pe_structure)

# --- Integrator: query_page (4) ---
def _qp_structure():
    r = mod.query_page(0, 10, 0, 5)
    return isinstance(r, dict) and set(r.keys()) == {"range", "token"}
_check("query_page_structure", _qp_structure)

def _qp_range_contains():
    r = mod.query_page(0, 10, 0, 5)
    return mod.range_contains(r["range"], 0) is True
_check("query_page_range_contains", _qp_range_contains)

def _qp_token_roundtrip():
    r = mod.query_page(0, 10, 7, 3)
    return tuple(mod.parse_page_token(r["token"])) == (7, 3)
_check("query_page_token_roundtrip", _qp_token_roundtrip)

def _qp_composed():
    r = mod.query_page(0, 100, 25, 10)
    return (mod.range_contains(r["range"], 50) is True
            and tuple(mod.parse_page_token(r["token"])) == (25, 10))
_check("query_page_composed", _qp_composed)

print(json.dumps(results))
"""


# --------------------------- Scoring ------------------------------------

# Weights sum to 1.0. Integrator tests are weighted higher because they
# exercise multi-agent interfaces end-to-end.
TEST_WEIGHTS = {
    # Pair A (5) — 0.15
    "ev_roundtrip_login":          0.03,
    "ev_roundtrip_click":          0.03,
    "ev_header_is_dict":           0.03,
    "ev_header_has_ts_somewhere":  0.03,
    "ev_consumer_handles_both_kinds": 0.03,
    # Pair B (3) — 0.09
    "ok_roundtrip":     0.03,
    "ok_string_value":  0.03,
    "ok_none_value":    0.03,
    # Pair C (3) — 0.09
    "err_roundtrip_code":     0.03,
    "err_roundtrip_code_500": 0.03,
    "err_frame_is_dict":      0.03,
    # Pair D (3) — 0.09
    "range_contains_inside":        0.03,
    "range_contains_exclusive_end": 0.03,
    "range_contains_before_start":  0.03,
    # Pair E (3) — 0.09
    "page_token_roundtrip":    0.03,
    "page_token_zero_offset":  0.03,
    "page_token_is_string":    0.03,
    # process_event (4) — 0.24  (integrator)
    "process_event_ok":             0.06,
    "process_event_err":            0.06,
    "process_event_preserves_kind": 0.06,
    "process_event_structure":      0.06,
    # query_page (4) — 0.25  (integrator)
    "query_page_structure":       0.06,
    "query_page_range_contains":  0.06,
    "query_page_token_roundtrip": 0.06,
    "query_page_composed":        0.07,
}
# Sanity: weights total to 1.0 (25 tests).
assert abs(sum(TEST_WEIGHTS.values()) - 1.0) < 1e-9, sum(TEST_WEIGHTS.values())


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
            parts.append(
                f"# === {specialty}: {FUNCTION_SPECS[specialty]['name']} ==="
            )
            parts.append(src)
    return "\n\n".join(parts)


# --------------------------- Reference solution -------------------------
#
# ALL agents in the reference use ONE consistent convention:
#   header   : {"kind": <str>, "ts": <int>}
#   ok env   : {"ok": True,  "value": <any>}
#   error    : {"err": True, "code": <int>, "msg": <str>}
#   range    : {"lo": <int>, "hi": <int>}  (inclusive lo, exclusive hi)
#   page tok : base64(f"{offset}:{limit}") url-safe
#
_REFERENCE_SOLUTIONS = {
    "make_event_header": '''\
def make_event_header(kind: str, ts: int) -> dict:
    return {"kind": kind, "ts": ts}
''',
    "wrap_ok": '''\
def wrap_ok(value) -> dict:
    return {"ok": True, "value": value}
''',
    "make_error": '''\
def make_error(code: int, msg: str) -> dict:
    return {"err": True, "code": code, "msg": msg}
''',
    "make_range": '''\
def make_range(start: int, end: int) -> dict:
    return {"lo": start, "hi": end}
''',
    "make_page_token": '''\
def make_page_token(offset: int, limit: int) -> str:
    raw = f"{offset}:{limit}".encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii")
''',
    "read_event_kind": '''\
def read_event_kind(header: dict) -> str:
    return header["kind"]
''',
    "is_ok": '''\
def is_ok(envelope: dict) -> bool:
    return isinstance(envelope, dict) and envelope.get("ok") is True
''',
    "get_error_code": '''\
def get_error_code(frame: dict) -> int:
    return int(frame["code"])
''',
    "range_contains": '''\
def range_contains(rng: dict, x: int) -> bool:
    return rng["lo"] <= x < rng["hi"]
''',
    "parse_page_token": '''\
def parse_page_token(token: str) -> tuple:
    raw = base64.urlsafe_b64decode(token.encode("ascii")).decode("utf-8")
    a, _, b = raw.partition(":")
    return (int(a), int(b))
''',
    "process_event": '''\
def process_event(kind: str, ts: int, ok: bool,
                  err_code: int = 0, err_msg: str = "") -> dict:
    header = make_event_header(kind, ts)
    if ok:
        body = wrap_ok(None)
    else:
        body = make_error(err_code, err_msg)
    return {"header": header, "body": body}
''',
    "query_page": '''\
def query_page(start: int, end: int, offset: int, limit: int) -> dict:
    return {
        "range": make_range(start, end),
        "token": make_page_token(offset, limit),
    }
''',
}


# --------------------------- Main: verify reference --------------------

if __name__ == "__main__":
    import sys
    from coordpy._internal.core.code_harness import run_sandboxed

    module = compose_module(_REFERENCE_SOLUTIONS)
    result = run_sandboxed(module, TEST_RUNNER_SRC, timeout_s=20)
    print(f"{result.n_passed}/{result.n_total} reference passes")
    if result.n_passed != 25:
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
