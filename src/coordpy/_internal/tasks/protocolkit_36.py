"""ProtocolKit-36 — 36-agent scale-up of the Phase-14 codesign task.

Structure (mirrors `protocol_codesign.py` at 3× the agent count):
  - 15 producer/consumer pairs (30 agents)
  - 6  multi-producer integrators (6 agents)
  - 48 tests: 2 per pair + 3 per integrator

Same conceit: each pair deliberately underspecifies its shared dict schema.
The producer picks a convention; the consumer must match it. Without round-2
routing, a random consumer guesses the wrong key and fails all pair tests.

Reference solution uses a single consistent convention across all 36
functions — when every agent's prompt contains the reference drafts, the
composed module passes 48/48.
"""

from __future__ import annotations

from dataclasses import dataclass


# --------------------------- Module header --------------------------------

MODULE_HEADER = """\
\"\"\"ProtocolKit-36 — a 36-function collaborative module.\"\"\"
from __future__ import annotations
import base64
"""


# --------------------------- Function catalog -----------------------------

@dataclass(frozen=True)
class Spec:
    specialty: str
    signature: str
    short_spec: str
    example: str


# 15 producer specs (tier 0)
_PRODUCERS: list[Spec] = [
    Spec("make_event_header", "(kind: str, ts: int) -> dict",
         "event header carrying `kind` (str) and `ts` (int unix-second)",
         "make_event_header('login', 1000) -> {'kind': 'login', 'ts': 1000}"),
    Spec("wrap_ok", "(value) -> dict",
         "success envelope wrapping any value, distinguishable from error",
         "wrap_ok(42) -> {'ok': True, 'value': 42}"),
    Spec("make_error", "(code: int, msg: str) -> dict",
         "error frame with int `code` and str `msg`; distinct from success",
         "make_error(404, 'nf') -> {'err': True, 'code': 404, 'msg': 'nf'}"),
    Spec("make_range", "(start: int, end: int) -> dict",
         "inclusive-start exclusive-end integer range [start, end)",
         "make_range(0, 10) -> {'lo': 0, 'hi': 10}"),
    Spec("make_page_token", "(offset: int, limit: int) -> str",
         "URL-safe STRING token encoding (offset, limit); must round-trip",
         "make_page_token(100, 20) -> 'MTAwOjIw'"),
    Spec("make_auth_token", "(user_id: int, expires_at: int) -> dict",
         "auth token with a user id and a unix-second expiry; dict",
         "make_auth_token(5, 9999) -> {'user_id': 5, 'expires_at': 9999}"),
    Spec("make_coord", "(x: int, y: int) -> dict",
         "2-D integer point dict",
         "make_coord(3, 4) -> {'x': 3, 'y': 4}"),
    Spec("make_money", "(amount: int, currency: str) -> dict",
         "monetary amount in minor units + 3-letter ISO currency code",
         "make_money(2500, 'USD') -> {'amount': 2500, 'currency': 'USD'}"),
    Spec("make_span", "(begin: int, end: int) -> dict",
         "time span from begin to end (unix-second timestamps)",
         "make_span(100, 400) -> {'begin': 100, 'end': 400}"),
    Spec("make_user", "(uid: int, first: str, last: str) -> dict",
         "user record with id and first/last name",
         "make_user(1, 'Ada', 'Lovelace') -> {'uid': 1, 'first': 'Ada', 'last': 'Lovelace'}"),
    Spec("make_file_meta", "(size: int, mime: str) -> dict",
         "file metadata: size in bytes + mime type",
         "make_file_meta(1024, 'text/plain') -> {'size': 1024, 'mime': 'text/plain'}"),
    Spec("make_status", "(code: int, label: str) -> dict",
         "HTTP-style status with integer code and short label",
         "make_status(200, 'OK') -> {'code': 200, 'label': 'OK'}"),
    Spec("make_priority", "(level: str, num: int) -> dict",
         "priority record with a string level and an integer rank",
         "make_priority('high', 9) -> {'level': 'high', 'num': 9}"),
    Spec("make_blob_ref", "(digest: str, length: int) -> dict",
         "reference to a stored blob: content hash digest + length",
         "make_blob_ref('abc123', 500) -> {'digest': 'abc123', 'length': 500}"),
    Spec("make_rate_limit", "(budget: int, window_s: int) -> dict",
         "rate limit: integer budget and int window-in-seconds",
         "make_rate_limit(100, 60) -> {'budget': 100, 'window_s': 60}"),
]


# 15 consumer specs (tier 1) — one per producer, reading its dict back
_CONSUMERS: list[Spec] = [
    Spec("read_event_kind", "(header: dict) -> str",
         "return the event `kind` stored in a header built by make_event_header",
         "read_event_kind({'kind': 'login', 'ts': 1000}) -> 'login'"),
    Spec("is_ok", "(envelope: dict) -> bool",
         "True iff envelope is a success envelope built by wrap_ok, False if error",
         "is_ok({'ok': True, 'value': 42}) -> True"),
    Spec("get_error_code", "(frame: dict) -> int",
         "extract the integer code from a frame built by make_error",
         "get_error_code({'err': True, 'code': 404, 'msg': 'nf'}) -> 404"),
    Spec("range_contains", "(rng: dict, x: int) -> bool",
         "True iff integer x lies in [start, end) of a dict built by make_range",
         "range_contains({'lo': 0, 'hi': 10}, 5) -> True"),
    Spec("parse_page_token", "(token: str) -> tuple",
         "decode a string token built by make_page_token into (offset, limit)",
         "parse_page_token('MTAwOjIw') -> (100, 20)"),
    Spec("verify_auth_token", "(token: dict, now: int) -> bool",
         "True iff `now` is before the expiry field in the token",
         "verify_auth_token({'user_id': 5, 'expires_at': 9999}, 100) -> True"),
    Spec("coord_distance", "(a: dict, b: dict) -> int",
         "Manhattan distance |a.x - b.x| + |a.y - b.y| between two coord dicts",
         "coord_distance({'x': 0, 'y': 0}, {'x': 3, 'y': 4}) -> 7"),
    Spec("format_money", "(money: dict) -> str",
         "format a money dict as 'AMOUNT CCY' (e.g. '2500 USD')",
         "format_money({'amount': 2500, 'currency': 'USD'}) -> '2500 USD'"),
    Spec("span_duration", "(span: dict) -> int",
         "return end - begin for a span dict",
         "span_duration({'begin': 100, 'end': 400}) -> 300"),
    Spec("user_display_name", "(user: dict) -> str",
         "return 'First Last' string from a user record",
         "user_display_name({'uid': 1, 'first': 'Ada', 'last': 'Lovelace'}) -> 'Ada Lovelace'"),
    Spec("file_is_text", "(meta: dict) -> bool",
         "True iff the mime type starts with 'text/'",
         "file_is_text({'size': 10, 'mime': 'text/plain'}) -> True"),
    Spec("status_is_success", "(status: dict) -> bool",
         "True iff the numeric code is in the 2xx range (200-299 inclusive)",
         "status_is_success({'code': 200, 'label': 'OK'}) -> True"),
    Spec("priority_rank", "(priority: dict) -> int",
         "return the numeric rank from a priority dict",
         "priority_rank({'level': 'high', 'num': 9}) -> 9"),
    Spec("blob_id", "(ref: dict) -> str",
         "return the digest (content hash) string from a blob reference",
         "blob_id({'digest': 'abc123', 'length': 500}) -> 'abc123'"),
    Spec("rate_allows", "(rl: dict, n: int) -> bool",
         "True iff n is less than or equal to the budget",
         "rate_allows({'budget': 100, 'window_s': 60}, 50) -> True"),
]


# 6 integrators (tier 1) — each composes 3 producers
@dataclass(frozen=True)
class IntegratorSpec:
    specialty: str
    signature: str
    short_spec: str
    example: str
    calls: tuple[str, ...]
    keys: tuple[str, ...]     # output dict keys, paired with calls


_INTEGRATORS: list[IntegratorSpec] = [
    IntegratorSpec(
        specialty="audit_log_entry",
        signature="(kind: str, ts: int, uid: int, expires_at: int, "
                  "first: str, last: str) -> dict",
        short_spec=(
            "Return {'header': make_event_header(kind, ts), "
            "'auth': make_auth_token(uid, expires_at), "
            "'user': make_user(uid, first, last)}. You MUST call the three "
            "named producer functions — do not reimplement their shapes."
        ),
        example="audit_log_entry('login', 1, 5, 9999, 'Ada', 'Lovelace') "
                "-> {'header': ..., 'auth': ..., 'user': ...}",
        calls=("make_event_header", "make_auth_token", "make_user"),
        keys=("header", "auth", "user"),
    ),
    IntegratorSpec(
        specialty="transfer_request",
        signature="(amount: int, currency: str, uid: int, expires_at: int, "
                  "code: int, label: str) -> dict",
        short_spec=(
            "Return {'money': make_money(amount, currency), "
            "'auth': make_auth_token(uid, expires_at), "
            "'status': make_status(code, label)}. Call all three producers."
        ),
        example="transfer_request(2500, 'USD', 5, 9999, 200, 'OK') "
                "-> {'money': ..., 'auth': ..., 'status': ...}",
        calls=("make_money", "make_auth_token", "make_status"),
        keys=("money", "auth", "status"),
    ),
    IntegratorSpec(
        specialty="position_update",
        signature="(x: int, y: int, begin: int, end: int, "
                  "uid: int, first: str, last: str) -> dict",
        short_spec=(
            "Return {'coord': make_coord(x, y), 'span': make_span(begin, end), "
            "'user': make_user(uid, first, last)}. Call all three producers."
        ),
        example="position_update(3, 4, 100, 400, 1, 'Ada', 'Lovelace') "
                "-> {'coord': ..., 'span': ..., 'user': ...}",
        calls=("make_coord", "make_span", "make_user"),
        keys=("coord", "span", "user"),
    ),
    IntegratorSpec(
        specialty="file_upload_result",
        signature="(size: int, mime: str, digest: str, length: int, "
                  "value) -> dict",
        short_spec=(
            "Return {'file': make_file_meta(size, mime), "
            "'blob': make_blob_ref(digest, length), "
            "'body': wrap_ok(value)}. Call all three producers."
        ),
        example="file_upload_result(1024, 'text/plain', 'abc', 500, None) "
                "-> {'file': ..., 'blob': ..., 'body': ...}",
        calls=("make_file_meta", "make_blob_ref", "wrap_ok"),
        keys=("file", "blob", "body"),
    ),
    IntegratorSpec(
        specialty="throttled_page",
        signature="(budget: int, window_s: int, offset: int, limit: int, "
                  "start: int, end: int) -> dict",
        short_spec=(
            "Return {'rate': make_rate_limit(budget, window_s), "
            "'token': make_page_token(offset, limit), "
            "'range': make_range(start, end)}. Call all three producers."
        ),
        example="throttled_page(100, 60, 0, 5, 0, 10) "
                "-> {'rate': ..., 'token': ..., 'range': ...}",
        calls=("make_rate_limit", "make_page_token", "make_range"),
        keys=("rate", "token", "range"),
    ),
    IntegratorSpec(
        specialty="priority_error",
        signature="(level: str, num: int, code: int, msg: str, "
                  "kind: str, ts: int) -> dict",
        short_spec=(
            "Return {'priority': make_priority(level, num), "
            "'error': make_error(code, msg), "
            "'header': make_event_header(kind, ts)}. Call all three producers."
        ),
        example="priority_error('high', 9, 500, 'boom', 'fail', 42) "
                "-> {'priority': ..., 'error': ..., 'header': ...}",
        calls=("make_priority", "make_error", "make_event_header"),
        keys=("priority", "error", "header"),
    ),
]


# --------------------------- Catalog assembly ----------------------------

_PAIR_LINKS: dict[str, str] = {
    p.specialty: c.specialty for p, c in zip(_PRODUCERS, _CONSUMERS)
}  # producer -> consumer
_CONSUMER_OF_PRODUCER = dict(_PAIR_LINKS)
_PRODUCER_OF_CONSUMER = {v: k for k, v in _PAIR_LINKS.items()}


def _producer_spec(p: Spec) -> dict:
    linked_consumer = _CONSUMER_OF_PRODUCER[p.specialty]
    return {
        "name": p.specialty,
        "signature": p.signature,
        "spec": (
            f"Return a value per the signature. Purpose: {p.short_spec}. "
            f"THE EXACT FIELD NAMES / ENCODING ARE YOUR CHOICE — pick any "
            f"plausible convention, but your teammate `{linked_consumer}` "
            f"must read from whatever dict you return, and integrators may "
            f"also embed your output in larger records. Stay internally "
            f"consistent."
        ),
        "example": p.example,
    }


def _consumer_spec(c: Spec) -> dict:
    linked_producer = _PRODUCER_OF_CONSUMER[c.specialty]
    return {
        "name": c.specialty,
        "signature": c.signature,
        "spec": (
            f"Purpose: {c.short_spec}. THE EXACT INPUT SHAPE IS DEFINED BY "
            f"THE PRODUCER `{linked_producer}`. If drafting independently "
            f"with NO teammate code visible, make a best guess at a common "
            f"convention. IF YOU CAN SEE the producer's code in this prompt, "
            f"use ITS exact field names / encoding — that is the canonical "
            f"truth."
        ),
        "example": c.example,
    }


def _integrator_spec(it: IntegratorSpec) -> dict:
    return {
        "name": it.specialty,
        "signature": it.signature,
        "spec": (
            f"{it.short_spec} If teammate code for the producers is visible "
            f"in this prompt, use their EXACT signatures."
        ),
        "example": it.example,
    }


FUNCTION_SPECS: dict[str, dict] = {}
for _p in _PRODUCERS:
    FUNCTION_SPECS[_p.specialty] = _producer_spec(_p)
for _c in _CONSUMERS:
    FUNCTION_SPECS[_c.specialty] = _consumer_spec(_c)
for _it in _INTEGRATORS:
    FUNCTION_SPECS[_it.specialty] = _integrator_spec(_it)


SPEC_ORDER: list[str] = (
    [p.specialty for p in _PRODUCERS]
    + [c.specialty for c in _CONSUMERS]
    + [it.specialty for it in _INTEGRATORS]
)
assert len(SPEC_ORDER) == 36, len(SPEC_ORDER)


CALL_GRAPH: dict[str, list[str]] = {}
for p in _PRODUCERS:
    CALL_GRAPH[p.specialty] = []
for c, p in zip(_CONSUMERS, _PRODUCERS):
    CALL_GRAPH[c.specialty] = [p.specialty]
for it in _INTEGRATORS:
    CALL_GRAPH[it.specialty] = list(it.calls)


# --------------------------- Agent prompt --------------------------------

def agent_prompt(specialty: str, dependency_outputs: dict[str, str] | None = None) -> str:
    s = FUNCTION_SPECS[specialty]
    parts = [
        f"You are the {specialty} specialist on a 36-person Python team "
        f"building ProtocolKit-36 — small typed primitives where pairs of "
        f"agents must agree on private dict schemas.",
        f"Write ONE Python function named `{s['name']}` with signature "
        f"{s['signature']}.",
        f"Specification:\n  {s['spec']}",
        f"Example:\n  {s['example']}",
        "The module header already imports `base64`; you may use it.",
    ]
    if dependency_outputs:
        parts.append(
            "You can reference these functions written by teammates. Do NOT "
            "re-define them — they will be concatenated with yours. USE THEIR "
            "EXACT FIELD NAMES / ENCODINGS:"
        )
        for dep, src in dependency_outputs.items():
            dep_name = FUNCTION_SPECS[dep]["name"]
            parts.append(
                f"  - `{dep_name}` is available. Preview:\n"
                f"```python\n{src[:500]}\n```"
            )
    parts.append(
        "Output ONLY the Python function in a ```python code fence. No other "
        "commentary. The function will be concatenated with 35 other "
        "functions and executed against an automated test suite whose pair "
        "and integrator tests only pass if your convention matches your "
        "teammates'."
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
import json, sys

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

# === Pair tests (2 per pair, 15 pairs = 30 tests) ===

# Pair 1 — event header
_check("ev_kind_login", lambda: mod.read_event_kind(mod.make_event_header("login", 1)) == "login")
_check("ev_kind_click", lambda: mod.read_event_kind(mod.make_event_header("click", 2)) == "click")

# Pair 2 — success envelope
_check("ok_int",  lambda: mod.is_ok(mod.wrap_ok(42)) is True)
_check("ok_none", lambda: mod.is_ok(mod.wrap_ok(None)) is True)

# Pair 3 — error frame
_check("err_404", lambda: mod.get_error_code(mod.make_error(404, "nf")) == 404)
_check("err_500", lambda: mod.get_error_code(mod.make_error(500, "boom")) == 500)

# Pair 4 — range
_check("range_in",  lambda: mod.range_contains(mod.make_range(0, 10), 5) is True)
_check("range_out", lambda: mod.range_contains(mod.make_range(0, 10), 10) is False)

# Pair 5 — page token
_check("page_rt",   lambda: tuple(mod.parse_page_token(mod.make_page_token(100, 20))) == (100, 20))
_check("page_zero", lambda: tuple(mod.parse_page_token(mod.make_page_token(0, 50))) == (0, 50))

# Pair 6 — auth token
_check("auth_valid",   lambda: mod.verify_auth_token(mod.make_auth_token(5, 9999), 100) is True)
_check("auth_expired", lambda: mod.verify_auth_token(mod.make_auth_token(5, 10), 9999) is False)

# Pair 7 — coord
_check("coord_zero",  lambda: mod.coord_distance(mod.make_coord(0, 0), mod.make_coord(0, 0)) == 0)
_check("coord_manhattan", lambda: mod.coord_distance(mod.make_coord(0, 0), mod.make_coord(3, 4)) == 7)

# Pair 8 — money
_check("money_format_usd", lambda: mod.format_money(mod.make_money(2500, "USD")) == "2500 USD")
_check("money_format_eur", lambda: mod.format_money(mod.make_money(75, "EUR")) == "75 EUR")

# Pair 9 — span
_check("span_basic", lambda: mod.span_duration(mod.make_span(100, 400)) == 300)
_check("span_zero",  lambda: mod.span_duration(mod.make_span(500, 500)) == 0)

# Pair 10 — user
_check("user_ada",     lambda: mod.user_display_name(mod.make_user(1, "Ada", "Lovelace")) == "Ada Lovelace")
_check("user_alan",    lambda: mod.user_display_name(mod.make_user(2, "Alan", "Turing")) == "Alan Turing")

# Pair 11 — file meta
_check("file_txt",  lambda: mod.file_is_text(mod.make_file_meta(10, "text/plain")) is True)
_check("file_bin",  lambda: mod.file_is_text(mod.make_file_meta(10, "application/octet-stream")) is False)

# Pair 12 — status
_check("status_200",  lambda: mod.status_is_success(mod.make_status(200, "OK")) is True)
_check("status_500",  lambda: mod.status_is_success(mod.make_status(500, "err")) is False)

# Pair 13 — priority
_check("prio_high", lambda: mod.priority_rank(mod.make_priority("high", 9)) == 9)
_check("prio_low",  lambda: mod.priority_rank(mod.make_priority("low", 1)) == 1)

# Pair 14 — blob ref
_check("blob_abc", lambda: mod.blob_id(mod.make_blob_ref("abc123", 500)) == "abc123")
_check("blob_def", lambda: mod.blob_id(mod.make_blob_ref("def456", 999)) == "def456")

# Pair 15 — rate limit
_check("rate_allows",  lambda: mod.rate_allows(mod.make_rate_limit(100, 60), 50) is True)
_check("rate_denies",  lambda: mod.rate_allows(mod.make_rate_limit(100, 60), 200) is False)

# === Integrator tests (3 per integrator × 6 = 18 tests) ===

# audit_log_entry
def _audit_struct():
    r = mod.audit_log_entry("login", 1, 5, 9999, "Ada", "Lovelace")
    return isinstance(r, dict) and set(r.keys()) == {"header", "auth", "user"}
_check("audit_structure", _audit_struct)
def _audit_kind():
    r = mod.audit_log_entry("login", 1, 5, 9999, "Ada", "Lovelace")
    return mod.read_event_kind(r["header"]) == "login"
_check("audit_kind_roundtrip", _audit_kind)
def _audit_user():
    r = mod.audit_log_entry("x", 0, 7, 100, "Alan", "Turing")
    return mod.user_display_name(r["user"]) == "Alan Turing"
_check("audit_user_roundtrip", _audit_user)

# transfer_request
def _tx_struct():
    r = mod.transfer_request(2500, "USD", 5, 9999, 200, "OK")
    return isinstance(r, dict) and set(r.keys()) == {"money", "auth", "status"}
_check("transfer_structure", _tx_struct)
def _tx_money():
    r = mod.transfer_request(100, "EUR", 1, 2, 200, "OK")
    return mod.format_money(r["money"]) == "100 EUR"
_check("transfer_money_roundtrip", _tx_money)
def _tx_status():
    r = mod.transfer_request(1, "USD", 1, 2, 404, "nf")
    return mod.status_is_success(r["status"]) is False
_check("transfer_status_roundtrip", _tx_status)

# position_update
def _pos_struct():
    r = mod.position_update(1, 2, 100, 200, 3, "Ada", "L")
    return isinstance(r, dict) and set(r.keys()) == {"coord", "span", "user"}
_check("position_structure", _pos_struct)
def _pos_span():
    r = mod.position_update(0, 0, 100, 400, 1, "A", "B")
    return mod.span_duration(r["span"]) == 300
_check("position_span_roundtrip", _pos_span)
def _pos_user():
    r = mod.position_update(1, 1, 0, 1, 5, "Ada", "Lovelace")
    return mod.user_display_name(r["user"]) == "Ada Lovelace"
_check("position_user_roundtrip", _pos_user)

# file_upload_result
def _fu_struct():
    r = mod.file_upload_result(1024, "text/plain", "abc", 100, None)
    return isinstance(r, dict) and set(r.keys()) == {"file", "blob", "body"}
_check("file_upload_structure", _fu_struct)
def _fu_text():
    r = mod.file_upload_result(10, "text/html", "x", 1, None)
    return mod.file_is_text(r["file"]) is True
_check("file_upload_mime", _fu_text)
def _fu_body():
    r = mod.file_upload_result(10, "image/png", "x", 1, "payload")
    return mod.is_ok(r["body"]) is True
_check("file_upload_body_ok", _fu_body)

# throttled_page
def _tp_struct():
    r = mod.throttled_page(100, 60, 0, 5, 0, 10)
    return isinstance(r, dict) and set(r.keys()) == {"rate", "token", "range"}
_check("throttled_structure", _tp_struct)
def _tp_range():
    r = mod.throttled_page(100, 60, 0, 5, 0, 10)
    return mod.range_contains(r["range"], 5) is True
_check("throttled_range", _tp_range)
def _tp_token():
    r = mod.throttled_page(100, 60, 25, 10, 0, 1000)
    return tuple(mod.parse_page_token(r["token"])) == (25, 10)
_check("throttled_token", _tp_token)

# priority_error
def _pe_struct():
    r = mod.priority_error("high", 9, 500, "boom", "fail", 42)
    return isinstance(r, dict) and set(r.keys()) == {"priority", "error", "header"}
_check("prioerr_structure", _pe_struct)
def _pe_code():
    r = mod.priority_error("low", 1, 404, "nf", "x", 0)
    return mod.get_error_code(r["error"]) == 404
_check("prioerr_code", _pe_code)
def _pe_kind():
    r = mod.priority_error("med", 5, 500, "e", "event", 7)
    return mod.read_event_kind(r["header"]) == "event"
_check("prioerr_kind", _pe_kind)

print(json.dumps(results))
"""


# --------------------------- Scoring ------------------------------------

# 48 tests total: each weighted 1/48 for simplicity. Integrator-heavy
# weighting would match Phase 14 but makes cross-phase comparison noisier;
# flat weights produce a score directly equivalent to fraction-passing.
_TEST_NAMES = [
    # 30 pair tests
    "ev_kind_login", "ev_kind_click",
    "ok_int", "ok_none",
    "err_404", "err_500",
    "range_in", "range_out",
    "page_rt", "page_zero",
    "auth_valid", "auth_expired",
    "coord_zero", "coord_manhattan",
    "money_format_usd", "money_format_eur",
    "span_basic", "span_zero",
    "user_ada", "user_alan",
    "file_txt", "file_bin",
    "status_200", "status_500",
    "prio_high", "prio_low",
    "blob_abc", "blob_def",
    "rate_allows", "rate_denies",
    # 18 integrator tests
    "audit_structure", "audit_kind_roundtrip", "audit_user_roundtrip",
    "transfer_structure", "transfer_money_roundtrip", "transfer_status_roundtrip",
    "position_structure", "position_span_roundtrip", "position_user_roundtrip",
    "file_upload_structure", "file_upload_mime", "file_upload_body_ok",
    "throttled_structure", "throttled_range", "throttled_token",
    "prioerr_structure", "prioerr_code", "prioerr_kind",
]
TEST_WEIGHTS = {t: 1.0 / len(_TEST_NAMES) for t in _TEST_NAMES}
assert abs(sum(TEST_WEIGHTS.values()) - 1.0) < 1e-9


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


# --------------------------- Reference solution -------------------------

_REF: dict[str, str] = {
    # --- producers (consistent convention) ---
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
    "make_auth_token": '''\
def make_auth_token(user_id: int, expires_at: int) -> dict:
    return {"user_id": user_id, "expires_at": expires_at}
''',
    "make_coord": '''\
def make_coord(x: int, y: int) -> dict:
    return {"x": x, "y": y}
''',
    "make_money": '''\
def make_money(amount: int, currency: str) -> dict:
    return {"amount": amount, "currency": currency}
''',
    "make_span": '''\
def make_span(begin: int, end: int) -> dict:
    return {"begin": begin, "end": end}
''',
    "make_user": '''\
def make_user(uid: int, first: str, last: str) -> dict:
    return {"uid": uid, "first": first, "last": last}
''',
    "make_file_meta": '''\
def make_file_meta(size: int, mime: str) -> dict:
    return {"size": size, "mime": mime}
''',
    "make_status": '''\
def make_status(code: int, label: str) -> dict:
    return {"code": code, "label": label}
''',
    "make_priority": '''\
def make_priority(level: str, num: int) -> dict:
    return {"level": level, "num": num}
''',
    "make_blob_ref": '''\
def make_blob_ref(digest: str, length: int) -> dict:
    return {"digest": digest, "length": length}
''',
    "make_rate_limit": '''\
def make_rate_limit(budget: int, window_s: int) -> dict:
    return {"budget": budget, "window_s": window_s}
''',
    # --- consumers ---
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
    "verify_auth_token": '''\
def verify_auth_token(token: dict, now: int) -> bool:
    return int(now) < int(token["expires_at"])
''',
    "coord_distance": '''\
def coord_distance(a: dict, b: dict) -> int:
    return abs(a["x"] - b["x"]) + abs(a["y"] - b["y"])
''',
    "format_money": '''\
def format_money(money: dict) -> str:
    return f"{money['amount']} {money['currency']}"
''',
    "span_duration": '''\
def span_duration(span: dict) -> int:
    return int(span["end"]) - int(span["begin"])
''',
    "user_display_name": '''\
def user_display_name(user: dict) -> str:
    return f"{user['first']} {user['last']}"
''',
    "file_is_text": '''\
def file_is_text(meta: dict) -> bool:
    return str(meta["mime"]).startswith("text/")
''',
    "status_is_success": '''\
def status_is_success(status: dict) -> bool:
    c = int(status["code"])
    return 200 <= c <= 299
''',
    "priority_rank": '''\
def priority_rank(priority: dict) -> int:
    return int(priority["num"])
''',
    "blob_id": '''\
def blob_id(ref: dict) -> str:
    return ref["digest"]
''',
    "rate_allows": '''\
def rate_allows(rl: dict, n: int) -> bool:
    return int(n) <= int(rl["budget"])
''',
    # --- integrators ---
    "audit_log_entry": '''\
def audit_log_entry(kind, ts, uid, expires_at, first, last):
    return {
        "header": make_event_header(kind, ts),
        "auth":   make_auth_token(uid, expires_at),
        "user":   make_user(uid, first, last),
    }
''',
    "transfer_request": '''\
def transfer_request(amount, currency, uid, expires_at, code, label):
    return {
        "money":  make_money(amount, currency),
        "auth":   make_auth_token(uid, expires_at),
        "status": make_status(code, label),
    }
''',
    "position_update": '''\
def position_update(x, y, begin, end, uid, first, last):
    return {
        "coord": make_coord(x, y),
        "span":  make_span(begin, end),
        "user":  make_user(uid, first, last),
    }
''',
    "file_upload_result": '''\
def file_upload_result(size, mime, digest, length, value):
    return {
        "file": make_file_meta(size, mime),
        "blob": make_blob_ref(digest, length),
        "body": wrap_ok(value),
    }
''',
    "throttled_page": '''\
def throttled_page(budget, window_s, offset, limit, start, end):
    return {
        "rate":  make_rate_limit(budget, window_s),
        "token": make_page_token(offset, limit),
        "range": make_range(start, end),
    }
''',
    "priority_error": '''\
def priority_error(level, num, code, msg, kind, ts):
    return {
        "priority": make_priority(level, num),
        "error":    make_error(code, msg),
        "header":   make_event_header(kind, ts),
    }
''',
}


# --------------------------- Main: verify reference --------------------

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
