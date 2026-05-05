"""TinyStore — a 12-agent collaborative-module task.

Each of 12 LLM agents writes ONE function of a mini in-memory key-value
store. The 12 functions are concatenated into a single module and run
against a 31-test suite.

Sub-groups:
  parse      — parse_command, validate_command, normalize_key
  store      — new_store, apply_set, apply_del
  query      — get_value, list_prefix, incr_value
  orchestrate— execute, run_script, summarize

The CALL_GRAPH defines each agent's causal footprint: which teammates'
outputs its function actually invokes. The CASR router uses this to pick
which dependency previews to include in each agent's prompt.

API mirrors collaborative_module.py exactly — same exported names, same
`score_tests` return shape, same `compose_module` / `agent_prompt`
signatures.
"""

from __future__ import annotations


# --------------------------- Specification ------------------------------

MODULE_HEADER = """\
\"\"\"TinyStore — a 12-function collaborative module.
Each function is written by a different LLM agent; they compose into a
command-dispatch pipeline over a shared dict-backed store.
\"\"\"
from __future__ import annotations
"""


FUNCTION_SPECS = {
    # --- parse group ---
    "parse_command": {
        "name": "parse_command",
        "signature": "(line: str) -> dict",
        "spec": (
            "Parse a single command line into a dict. Supported forms: "
            "'SET key=value', 'GET key', 'DEL key', 'INCR key', "
            "'LIST prefix'. Return {'op': <op>, 'key': <str>, "
            "'value': <str or None>}. For SET the value is the substring "
            "after the first '='. For non-SET ops, 'value' is None. "
            "Leading/trailing whitespace on the whole line is stripped. "
            "Raise ValueError for unknown op, missing key, or malformed "
            "SET (no '=')."
        ),
        "example": (
            "parse_command('SET foo=bar') -> "
            "{'op': 'SET', 'key': 'foo', 'value': 'bar'}"
        ),
    },
    "validate_command": {
        "name": "validate_command",
        "signature": "(cmd: dict) -> bool",
        "spec": (
            "Return True iff cmd is a dict with 'op' in "
            "{'SET','GET','DEL','INCR','LIST'}, a non-empty string 'key', "
            "and (if op == 'SET') a string 'value'. Never raise — return "
            "False for any violation including non-dict inputs."
        ),
        "example": (
            "validate_command({'op':'SET','key':'a','value':'b'}) -> True"
        ),
    },
    "normalize_key": {
        "name": "normalize_key",
        "signature": "(key: str) -> str",
        "spec": (
            "Strip surrounding whitespace and lowercase the key. Raise "
            "ValueError if the result is empty or if the input is not a "
            "string. Idempotent: normalize_key(normalize_key(k)) == "
            "normalize_key(k)."
        ),
        "example": "normalize_key('  Foo ') -> 'foo'",
    },
    # --- store group ---
    "new_store": {
        "name": "new_store",
        "signature": "() -> dict",
        "spec": (
            "Return a fresh empty store dict: "
            "{'data': {}, 'log': []}. No arguments."
        ),
        "example": "new_store() -> {'data': {}, 'log': []}",
    },
    "apply_set": {
        "name": "apply_set",
        "signature": "(store: dict, key: str, value: str) -> dict",
        "spec": (
            "Set store['data'][normalize_key(key)] = value and append "
            "('SET', key, value) to store['log']. Return the (mutated) "
            "store. Must call normalize_key."
        ),
        "example": (
            "apply_set({'data':{},'log':[]}, 'Foo', 'bar') mutates and "
            "returns a store with data={'foo':'bar'} and one log entry."
        ),
    },
    "apply_del": {
        "name": "apply_del",
        "signature": "(store: dict, key: str) -> dict",
        "spec": (
            "If normalize_key(key) is in store['data'], delete it. "
            "Otherwise no-op. In both cases, append ('DEL', key, None) to "
            "store['log']. Return the store. Must call normalize_key."
        ),
        "example": "apply_del(store, 'foo') removes 'foo' if present.",
    },
    # --- query group ---
    "get_value": {
        "name": "get_value",
        "signature": "(store: dict, key: str) -> str | None",
        "spec": (
            "Return store['data'].get(normalize_key(key)). Return None if "
            "the key is absent. Never raise on a missing key (but "
            "ValueError from normalize_key on empty-key input may "
            "propagate). Must call normalize_key."
        ),
        "example": "get_value(store, 'foo') -> 'bar' or None",
    },
    "list_prefix": {
        "name": "list_prefix",
        "signature": "(store: dict, prefix: str) -> list[str]",
        "spec": (
            "Return a sorted list of keys in store['data'] whose keys "
            "startswith the normalized prefix. Use normalize_key on the "
            "prefix BUT tolerate empty prefix (return all keys) by "
            "checking for empty-after-strip BEFORE calling normalize_key. "
            "Keys in the store are already normalized, so compare via "
            "startswith."
        ),
        "example": (
            "list_prefix(store_with_keys_[foo,foobar,baz], 'foo') -> "
            "['foo', 'foobar']"
        ),
    },
    "incr_value": {
        "name": "incr_value",
        "signature": "(store: dict, key: str) -> dict",
        "spec": (
            "If get_value(store, key) is None, apply_set the key to '1'. "
            "Else parse the existing value as int; on success, apply_set "
            "to str(int(value)+1). On non-int existing value, raise "
            "ValueError. Return the store. Must call apply_set and "
            "get_value by name."
        ),
        "example": "incr_value(store, 'counter') advances the counter.",
    },
    # --- orchestrate group ---
    "execute": {
        "name": "execute",
        "signature": "(store: dict, cmd: dict) -> tuple",
        "spec": (
            "Dispatch a validated cmd dict to the appropriate function. "
            "Return (store, result). result is: the value for GET (str or "
            "None), sorted list for LIST, None for SET/DEL, the store "
            "dict for INCR. Raise ValueError if cmd is invalid per "
            "validate_command. Must call apply_set, apply_del, get_value, "
            "list_prefix, and incr_value by name."
        ),
        "example": (
            "execute(store, {'op':'GET','key':'foo','value':None}) -> "
            "(store, 'bar')"
        ),
    },
    "run_script": {
        "name": "run_script",
        "signature": "(lines: list[str]) -> dict",
        "spec": (
            "For each line, call parse_command (catching ValueError -> "
            "increment errors and skip), then validate_command "
            "(False -> increment errors and skip), then execute on a "
            "single shared store created via new_store(). Collect results "
            "in order (only for successfully-executed commands). Return "
            "{'final_store': store, 'results': [...], 'errors': <int>}. "
            "Must call parse_command, validate_command, new_store, and "
            "execute by name."
        ),
        "example": (
            "run_script(['SET a=1','GET a']) -> "
            "{'final_store': ..., 'results': [None, '1'], 'errors': 0}"
        ),
    },
    "summarize": {
        "name": "summarize",
        "signature": "(store: dict) -> dict",
        "spec": (
            "Return {'n_keys': len(store['data']), "
            "'n_ops': len(store['log']), 'top_prefix': <str>} where "
            "top_prefix is the most common 3-character prefix across the "
            "keys in store['data'] (empty string '' if the store is "
            "empty). Ties broken by lexicographic order (smallest wins)."
        ),
        "example": (
            "summarize({'data':{'foobar':'x','fooqux':'y'},'log':[]}) -> "
            "{'n_keys':2, 'n_ops':0, 'top_prefix':'foo'}"
        ),
    },
}


# Topologically ordered: parse/normalize first, store primitives, queries,
# orchestrators last.
SPEC_ORDER = [
    "parse_command",
    "validate_command",
    "normalize_key",
    "new_store",
    "apply_set",
    "apply_del",
    "get_value",
    "list_prefix",
    "incr_value",
    "execute",
    "run_script",
    "summarize",
]


CALL_GRAPH = {
    "parse_command":    [],
    "validate_command": [],
    "normalize_key":    [],
    "new_store":        [],
    "apply_set":        ["normalize_key"],
    "apply_del":        ["normalize_key"],
    "get_value":        ["normalize_key"],
    "list_prefix":      ["normalize_key"],
    "incr_value":       ["apply_set", "get_value"],
    "execute":          ["apply_set", "apply_del", "get_value",
                         "list_prefix", "incr_value"],
    "run_script":       ["parse_command", "validate_command",
                         "new_store", "execute"],
    "summarize":        [],
}


def agent_prompt(specialty: str, dependency_outputs: dict[str, str] = None) -> str:
    """Build the LLM prompt for one specialty agent."""
    s = FUNCTION_SPECS[specialty]
    parts = [
        f"You are the {specialty} specialist on a 12-person Python team "
        f"building a small in-memory key-value store called TinyStore.",
        f"Your job: write ONE Python function named `{s['name']}` with "
        f"signature {s['signature']}.",
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
            parts.append(
                f"  - `{dep_name}` is available. Preview:\n"
                f"```python\n{src[:500]}\n```"
            )
    parts.append(
        "Output ONLY the Python function in a ```python code fence. "
        "No other commentary. The function will be concatenated with 11 "
        "other functions and executed against an automated test suite."
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

# --- parse_command (4) ---
_check("parse_set", lambda: (
    mod.parse_command("SET foo=bar") == {"op": "SET", "key": "foo", "value": "bar"}
))
_check("parse_get", lambda: (
    mod.parse_command("GET foo") == {"op": "GET", "key": "foo", "value": None}
))
_check("parse_list", lambda: (
    mod.parse_command("LIST pre") == {"op": "LIST", "key": "pre", "value": None}
))
def _parse_bad():
    try:
        mod.parse_command("FROB x")
        return False
    except ValueError:
        return True
    except Exception:
        return False
_check("parse_rejects_unknown_op", _parse_bad)

# --- validate_command (3) ---
_check("validate_ok", lambda: mod.validate_command(
    {"op": "SET", "key": "a", "value": "b"}) is True)
_check("validate_set_missing_value", lambda: mod.validate_command(
    {"op": "SET", "key": "a", "value": None}) is False)
_check("validate_empty_key", lambda: mod.validate_command(
    {"op": "GET", "key": "", "value": None}) is False)

# --- normalize_key (3) ---
_check("normalize_trim_lower", lambda: mod.normalize_key("  Foo ") == "foo")
def _norm_empty():
    try:
        mod.normalize_key("   ")
        return False
    except ValueError:
        return True
    except Exception:
        return False
_check("normalize_reject_empty", _norm_empty)
_check("normalize_idempotent", lambda: (
    mod.normalize_key(mod.normalize_key("Bar")) == mod.normalize_key("Bar")
))

# --- new_store (1) ---
_check("new_store_empty", lambda: (
    mod.new_store() == {"data": {}, "log": []}
))

# --- apply_set (2) ---
_check("apply_set_basic", lambda: (
    (lambda s: (mod.apply_set(s, "Foo", "bar"),
                s["data"] == {"foo": "bar"} and len(s["log"]) == 1)[1])
    ({"data": {}, "log": []})
))
_check("apply_set_overwrite", lambda: (
    (lambda s: (mod.apply_set(s, "k", "v1"), mod.apply_set(s, "k", "v2"),
                s["data"] == {"k": "v2"} and len(s["log"]) == 2)[-1])
    ({"data": {}, "log": []})
))

# --- apply_del (2) ---
_check("apply_del_existing", lambda: (
    (lambda s: (mod.apply_set(s, "k", "v"), mod.apply_del(s, "K"),
                s["data"] == {} and len(s["log"]) == 2)[-1])
    ({"data": {}, "log": []})
))
_check("apply_del_missing_noop", lambda: (
    (lambda s: (mod.apply_del(s, "nope"),
                s["data"] == {} and len(s["log"]) == 1)[-1])
    ({"data": {}, "log": []})
))

# --- get_value (2) ---
_check("get_existing", lambda: (
    (lambda s: (mod.apply_set(s, "k", "v"),
                mod.get_value(s, "K") == "v")[-1])
    ({"data": {}, "log": []})
))
_check("get_missing_none", lambda: (
    mod.get_value({"data": {}, "log": []}, "nope") is None
))

# --- list_prefix (3) ---
def _list_setup():
    s = {"data": {}, "log": []}
    mod.apply_set(s, "foo", "1")
    mod.apply_set(s, "foobar", "2")
    mod.apply_set(s, "baz", "3")
    return s
_check("list_finds_prefix", lambda: (
    mod.list_prefix(_list_setup(), "foo") == ["foo", "foobar"]
))
_check("list_empty_prefix_all", lambda: (
    sorted(mod.list_prefix(_list_setup(), "")) == ["baz", "foo", "foobar"]
))
_check("list_case_insensitive", lambda: (
    mod.list_prefix(_list_setup(), "FOO") == ["foo", "foobar"]
))

# --- incr_value (3) ---
_check("incr_existing_int", lambda: (
    (lambda s: (mod.apply_set(s, "c", "4"), mod.incr_value(s, "c"),
                s["data"]["c"] == "5")[-1])
    ({"data": {}, "log": []})
))
_check("incr_missing_creates_one", lambda: (
    (lambda s: (mod.incr_value(s, "c"),
                s["data"]["c"] == "1")[-1])
    ({"data": {}, "log": []})
))
def _incr_non_int():
    s = {"data": {}, "log": []}
    mod.apply_set(s, "c", "hello")
    try:
        mod.incr_value(s, "c")
        return False
    except ValueError:
        return True
    except Exception:
        return False
_check("incr_non_int_raises", _incr_non_int)

# --- execute (3) ---
_check("exec_set", lambda: (
    (lambda s: (mod.execute(s, {"op":"SET","key":"a","value":"1"}),
                s["data"] == {"a": "1"})[-1])
    ({"data": {}, "log": []})
))
_check("exec_get", lambda: (
    (lambda s: (mod.apply_set(s, "a", "1"),
                mod.execute(s, {"op":"GET","key":"a","value":None})[1] == "1")[-1])
    ({"data": {}, "log": []})
))
def _exec_invalid():
    try:
        mod.execute({"data": {}, "log": []}, {"op": "NOPE", "key": "x", "value": None})
        return False
    except ValueError:
        return True
    except Exception:
        return False
_check("exec_invalid_raises", _exec_invalid)

# --- run_script (3) ---
_check("script_basic", lambda: (
    isinstance(mod.run_script(["SET a=1", "GET a"]), dict)
    and mod.run_script(["SET a=1", "GET a"])["errors"] == 0
    and mod.run_script(["SET a=1", "GET a"])["results"][-1] == "1"
))
_check("script_skips_bad", lambda: (
    mod.run_script(["SET a=1", "FROB nope", "GET a"])["errors"] == 1
))
_check("script_final_state", lambda: (
    mod.run_script(["SET a=1", "SET b=2", "DEL a"])
    ["final_store"]["data"] == {"b": "2"}
))

# --- summarize (2) ---
_check("summarize_empty", lambda: (
    mod.summarize({"data": {}, "log": []})
    == {"n_keys": 0, "n_ops": 0, "top_prefix": ""}
))
_check("summarize_populated", lambda: (
    (lambda r: r["n_keys"] == 2 and r["n_ops"] == 0 and r["top_prefix"] == "foo")(
        mod.summarize({"data": {"foobar": "x", "fooqux": "y"}, "log": []})
    )
))

print(json.dumps(results))
"""


# --------------------------- Scoring ------------------------------------

# Weights sum to 1.0. Orchestration tests (execute/run_script) get the
# highest weight because they exercise cross-agent interfaces.
TEST_WEIGHTS = {
    # parse (4) — 0.12 total
    "parse_set": 0.03,
    "parse_get": 0.03,
    "parse_list": 0.03,
    "parse_rejects_unknown_op": 0.03,
    # validate (3) — 0.09 total
    "validate_ok": 0.03,
    "validate_set_missing_value": 0.03,
    "validate_empty_key": 0.03,
    # normalize (3) — 0.09 total
    "normalize_trim_lower": 0.03,
    "normalize_reject_empty": 0.03,
    "normalize_idempotent": 0.03,
    # new_store (1) — 0.02
    "new_store_empty": 0.02,
    # apply_set (2) — 0.06
    "apply_set_basic": 0.03,
    "apply_set_overwrite": 0.03,
    # apply_del (2) — 0.06
    "apply_del_existing": 0.03,
    "apply_del_missing_noop": 0.03,
    # get_value (2) — 0.05
    "get_existing": 0.025,
    "get_missing_none": 0.025,
    # list_prefix (3) — 0.08
    "list_finds_prefix": 0.03,
    "list_empty_prefix_all": 0.025,
    "list_case_insensitive": 0.025,
    # incr_value (3) — 0.09
    "incr_existing_int": 0.03,
    "incr_missing_creates_one": 0.03,
    "incr_non_int_raises": 0.03,
    # execute (3) — 0.12  (orchestration)
    "exec_set": 0.04,
    "exec_get": 0.04,
    "exec_invalid_raises": 0.04,
    # run_script (3) — 0.18  (top-level orchestration)
    "script_basic": 0.06,
    "script_skips_bad": 0.06,
    "script_final_state": 0.06,
    # summarize (2) — 0.04
    "summarize_empty": 0.02,
    "summarize_populated": 0.02,
}
# Sanity: weights total to 1.0 (31 tests).
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

_REFERENCE_SOLUTIONS = {
    "parse_command": '''\
def parse_command(line: str) -> dict:
    if not isinstance(line, str):
        raise ValueError("line must be a string")
    line = line.strip()
    if not line:
        raise ValueError("empty line")
    parts = line.split(None, 1)
    if len(parts) < 2:
        raise ValueError("missing key")
    op, rest = parts[0], parts[1].strip()
    if op not in {"SET", "GET", "DEL", "INCR", "LIST"}:
        raise ValueError("unknown op: " + op)
    if op == "SET":
        if "=" not in rest:
            raise ValueError("SET requires key=value")
        k, _, v = rest.partition("=")
        k = k.strip()
        if not k:
            raise ValueError("empty key")
        return {"op": "SET", "key": k, "value": v}
    else:
        if not rest:
            raise ValueError("missing key")
        return {"op": op, "key": rest, "value": None}
''',
    "validate_command": '''\
def validate_command(cmd: dict) -> bool:
    if not isinstance(cmd, dict):
        return False
    op = cmd.get("op")
    key = cmd.get("key")
    value = cmd.get("value")
    if op not in {"SET", "GET", "DEL", "INCR", "LIST"}:
        return False
    if not isinstance(key, str) or not key:
        return False
    if op == "SET":
        if not isinstance(value, str):
            return False
    return True
''',
    "normalize_key": '''\
def normalize_key(key: str) -> str:
    if not isinstance(key, str):
        raise ValueError("key must be a string")
    k = key.strip().lower()
    if not k:
        raise ValueError("empty key")
    return k
''',
    "new_store": '''\
def new_store() -> dict:
    return {"data": {}, "log": []}
''',
    "apply_set": '''\
def apply_set(store: dict, key: str, value: str) -> dict:
    nk = normalize_key(key)
    store["data"][nk] = value
    store["log"].append(("SET", key, value))
    return store
''',
    "apply_del": '''\
def apply_del(store: dict, key: str) -> dict:
    nk = normalize_key(key)
    if nk in store["data"]:
        del store["data"][nk]
    store["log"].append(("DEL", key, None))
    return store
''',
    "get_value": '''\
def get_value(store: dict, key: str):
    nk = normalize_key(key)
    return store["data"].get(nk)
''',
    "list_prefix": '''\
def list_prefix(store: dict, prefix: str) -> list:
    if not isinstance(prefix, str):
        raise ValueError("prefix must be a string")
    stripped = prefix.strip()
    if not stripped:
        return sorted(store["data"].keys())
    np = normalize_key(prefix)
    return sorted(k for k in store["data"].keys() if k.startswith(np))
''',
    "incr_value": '''\
def incr_value(store: dict, key: str) -> dict:
    current = get_value(store, key)
    if current is None:
        apply_set(store, key, "1")
        return store
    try:
        n = int(current)
    except (ValueError, TypeError):
        raise ValueError("value is not an integer: " + repr(current))
    apply_set(store, key, str(n + 1))
    return store
''',
    "execute": '''\
def execute(store: dict, cmd: dict):
    if not validate_command(cmd):
        raise ValueError("invalid command: " + repr(cmd))
    op = cmd["op"]
    key = cmd["key"]
    value = cmd.get("value")
    if op == "SET":
        apply_set(store, key, value)
        return (store, None)
    if op == "GET":
        return (store, get_value(store, key))
    if op == "DEL":
        apply_del(store, key)
        return (store, None)
    if op == "LIST":
        return (store, list_prefix(store, key))
    if op == "INCR":
        incr_value(store, key)
        return (store, store)
    raise ValueError("unknown op: " + op)
''',
    "run_script": '''\
def run_script(lines):
    store = new_store()
    results = []
    errors = 0
    for line in lines:
        try:
            cmd = parse_command(line)
        except ValueError:
            errors += 1
            continue
        if not validate_command(cmd):
            errors += 1
            continue
        try:
            _, result = execute(store, cmd)
        except ValueError:
            errors += 1
            continue
        results.append(result)
    return {"final_store": store, "results": results, "errors": errors}
''',
    "summarize": '''\
def summarize(store: dict) -> dict:
    data = store.get("data", {})
    log = store.get("log", [])
    if not data:
        return {"n_keys": 0, "n_ops": len(log), "top_prefix": ""}
    from collections import Counter
    counts = Counter(k[:3] for k in data.keys())
    # Most common by count, break ties by lexicographic order
    best = min(counts.keys(), key=lambda p: (-counts[p], p))
    return {"n_keys": len(data), "n_ops": len(log), "top_prefix": best}
''',
}


# --------------------------- Main: verify reference --------------------

if __name__ == "__main__":
    import sys
    from coordpy._internal.core.code_harness import run_sandboxed

    module = compose_module(_REFERENCE_SOLUTIONS)
    result = run_sandboxed(module, TEST_RUNNER_SRC, timeout_s=20)
    print(f"{result.n_passed}/{result.n_total} reference passes")
    if result.n_passed != 31:
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
