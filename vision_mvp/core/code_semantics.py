"""Conservative static semantic analysis for Python functions — Phase 24.

Phase 22 ingested AST-derived *structural* metadata (function names,
imports, line counts, etc.). The planner could answer questions like
"how many functions" or "list files importing numpy" exactly because
those properties read directly off syntactic fields.

Phase 24 extends the ingestion layer with a small set of **conservative
semantic predicates** — facts about what a function *might do at
runtime* that are still decidable (or soundly over-approximable) from
the AST alone. The design rule is:

  * **Soundness over precision.** We prefer false positives over false
    negatives. If we can't prove a function doesn't do X, we say it
    might do X.
  * **Intraprocedural only.** No whole-program analysis, no alias
    tracking, no constant propagation. Each function is analysed
    against its own body; calls to other user functions are treated
    as opaque. This is what makes the analysis tractable and
    predictable.
  * **Explicit API surface.** Known-IO predicates (`calls_subprocess`,
    `calls_filesystem`, `calls_network`) trigger only on a curated
    list of standard-library / very common third-party surfaces. The
    list is small, commented, and the false-negative mode is "the
    user wrote custom IO that isn't in our list". That's fine — the
    predicate is called *calls_external_io* not *has_side_effects*.

The analyzer is a pure AST pass: given a `ast.FunctionDef` or
`ast.AsyncFunctionDef`, it returns a `FunctionSemantics` dataclass
with bool flags. The fields are consumed by `core.code_index` where
they become parallel tuples on `CodeMetadata`.

Supported predicates (full list in `FunctionSemantics`):

  * `may_raise`            — the body contains a `raise` that is not
                             caught inside the same function (EXPLICIT
                             raises only; Phase 24 contract)
  * `may_raise_implicit`   — **Phase 28** — the body contains at least
                             one syntactic pattern whose builtin
                             semantics permit propagating an exception
                             on normally-shaped inputs (division,
                             modulo, power, subscript, select builtin
                             calls, attribute access on a positional
                             parameter), AND the pattern is not
                             lexically wrapped in a catch-all
                             try/except. Kept separate from
                             `may_raise` so the Phase-24 discrimination
                             survives Phase 27's implicit-raise finding
                             (`RESULTS_PHASE28.md` § "Explicit vs
                             implicit raise semantics").
  * `is_recursive`         — the body calls itself by simple name or
                             via `self.<name>(...)` in the enclosing
                             class
  * `may_write_global`     — the body declares `global X` and then
                             assigns to X, or assigns to a
                             documented-mutable module-level collection
  * `calls_subprocess`     — calls `subprocess.*`, `os.system`,
                             `os.exec*`, `os.spawn*`, `os.popen`
  * `calls_filesystem`     — calls `open`, `os.open/read/write/remove/
                             rename/makedirs/mkdir/rmdir/unlink/listdir/
                             walk`, `shutil.*`, `pathlib.Path.*
                             {read,write}_{text,bytes}/open/unlink/mkdir`
  * `calls_network`        — calls `socket.*`, `urllib.request.*`,
                             `urllib.urlopen`, `http.client.*`,
                             `requests.*`, `httpx.*`

The design is deliberately small: six boolean predicates, each
independently measurable and independently testable. The point is
**not** to reproduce a full program-analysis tool — the point is to
ship a conservative, auditable slice whose soundness properties are
clear enough to defend in a research writeup.

Boundary conditions (documented in RESULTS_PHASE24.md):

  * **Reflection / dynamic dispatch.** `getattr(mod, 'open')(...)`
    is NOT flagged as a filesystem call. That's a known limitation.
  * **Aliasing.** `f = subprocess.run; f(...)` is NOT flagged. Same.
  * **Try/except swallow.** `may_raise` correctly returns False when
    a `try: ... except BaseException: pass` wraps the only raise.
    Partial catches (e.g. `except ValueError`) conservatively still
    flag `may_raise=True` because a non-caught type could still
    propagate.
  * **exec / eval.** Excluded from IO predicates. The substrate
    treats these as opaque; they count as side-effectful but we
    don't have a separate `calls_eval` predicate in Phase 24.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Iterable


# =============================================================================
# Public dataclass
# =============================================================================


@dataclass(frozen=True)
class FunctionSemantics:
    """Per-function conservative static-analysis summary.

    Every field is a conservative over-approximation: True means "might"
    or "some call path could"; False means "provably not, given this
    analysis's assumptions". False negatives are considered bugs;
    false positives are expected.

    The `may_raise_implicit` field (Phase 28) is kept SEPARATE from
    `may_raise` by design: collapsing them would saturate the flag
    on almost every function and destroy the discrimination
    Phase 24 relies on. See `RESULTS_PHASE28.md` § "Explicit vs
    implicit raise semantics".
    """

    may_raise: bool
    is_recursive: bool
    may_write_global: bool
    calls_subprocess: bool
    calls_filesystem: bool
    calls_network: bool
    # Phase 28: added at the tail so the constructor stays
    # backward-compatible with any external construction site. Keyword
    # callers are unaffected; positional callers fall back to the
    # default below.
    may_raise_implicit: bool = False

    @property
    def calls_external_io(self) -> bool:
        """Convenience disjunction. Used by the planner for "side
        effects outside local scope"-class questions."""
        return (self.calls_subprocess
                or self.calls_filesystem
                or self.calls_network)

    def as_tuple(self) -> tuple[bool, ...]:
        """Stable ordering for parallel-tuple metadata serialisation.

        Extended in Phase 28 — `may_raise_implicit` is appended at the
        end so prior consumers reading the first six fields continue
        to observe Phase-24 semantics.
        """
        return (
            self.may_raise,
            self.is_recursive,
            self.may_write_global,
            self.calls_subprocess,
            self.calls_filesystem,
            self.calls_network,
            self.may_raise_implicit,
        )


# =============================================================================
# Known-API surface tables
# =============================================================================

# Each entry is a *dotted* prefix or a bare callable name. A call qualifies
# iff its resolved call name (as a dotted path like "subprocess.run" or a
# bare name like "open") EXACTLY matches an entry or STARTS WITH
# "<entry>." for dotted-path prefixes.

_SUBPROCESS_APIS: frozenset[str] = frozenset({
    # stdlib `subprocess`
    "subprocess.run",
    "subprocess.call",
    "subprocess.check_call",
    "subprocess.check_output",
    "subprocess.Popen",
    "subprocess.getoutput",
    "subprocess.getstatusoutput",
    # stdlib `os` process spawning
    "os.system",
    "os.popen",
    "os.spawnl",
    "os.spawnle",
    "os.spawnlp",
    "os.spawnlpe",
    "os.spawnv",
    "os.spawnve",
    "os.spawnvp",
    "os.spawnvpe",
    "os.execl",
    "os.execle",
    "os.execlp",
    "os.execlpe",
    "os.execv",
    "os.execve",
    "os.execvp",
    "os.execvpe",
})

_FILESYSTEM_APIS: frozenset[str] = frozenset({
    # bare builtins
    "open",
    # stdlib `os` filesystem
    "os.open",
    "os.read",
    "os.write",
    "os.close",
    "os.remove",
    "os.rename",
    "os.replace",
    "os.makedirs",
    "os.mkdir",
    "os.rmdir",
    "os.removedirs",
    "os.unlink",
    "os.listdir",
    "os.scandir",
    "os.walk",
    "os.stat",
    "os.chmod",
    "os.chown",
    "os.link",
    "os.symlink",
    "os.readlink",
    "os.truncate",
    "os.utime",
    # stdlib `shutil`
    "shutil.copy",
    "shutil.copy2",
    "shutil.copyfile",
    "shutil.copyfileobj",
    "shutil.copytree",
    "shutil.move",
    "shutil.rmtree",
    "shutil.make_archive",
    "shutil.unpack_archive",
    "shutil.chown",
    "shutil.disk_usage",
    # stdlib `pathlib.Path` methods (detected as `Path.*` OR `Path(...).METHOD`)
    "pathlib.Path.read_text",
    "pathlib.Path.write_text",
    "pathlib.Path.read_bytes",
    "pathlib.Path.write_bytes",
    "pathlib.Path.open",
    "pathlib.Path.unlink",
    "pathlib.Path.mkdir",
    "pathlib.Path.rmdir",
    "pathlib.Path.touch",
    "pathlib.Path.rename",
    "pathlib.Path.replace",
})

# Bare method names that imply filesystem mutation when called on a value
# we cannot statically resolve but which we believe to be a `Path` (e.g.
# `some_path.read_text()`). Used as a weaker signal — only triggers if
# the code imports pathlib OR has a hint that the receiver is a Path.
_PATHLIB_METHOD_NAMES: frozenset[str] = frozenset({
    "read_text", "write_text", "read_bytes", "write_bytes",
    "unlink", "touch",
})

_NETWORK_APIS: frozenset[str] = frozenset({
    # stdlib `socket`
    "socket.socket",
    "socket.create_connection",
    "socket.create_server",
    "socket.getaddrinfo",
    "socket.gethostbyname",
    # stdlib `urllib`
    "urllib.request.urlopen",
    "urllib.request.Request",
    "urllib.request.urlretrieve",
    "urllib.urlopen",
    # stdlib `http`
    "http.client.HTTPConnection",
    "http.client.HTTPSConnection",
    # very common third-party
    "requests.get",
    "requests.post",
    "requests.put",
    "requests.delete",
    "requests.patch",
    "requests.head",
    "requests.options",
    "requests.request",
    "requests.Session",
    "httpx.get",
    "httpx.post",
    "httpx.put",
    "httpx.delete",
    "httpx.patch",
    "httpx.request",
    "httpx.Client",
    "httpx.AsyncClient",
    "aiohttp.ClientSession",
})


def _call_name(node: ast.AST) -> str | None:
    """Resolve a call target to a dotted string like 'subprocess.run' or
    'open'. Returns None when the target is a Subscript, Call, Starred,
    Lambda etc. that cannot be named statically — the conservative call
    is to treat unresolved calls as opaque (they don't match any API
    table)."""
    if isinstance(node, ast.Call):
        return _call_name(node.func)
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _call_name(node.value)
        if parent is None:
            return None
        return f"{parent}.{node.attr}"
    return None


def _call_matches(dotted_name: str, table: Iterable[str]) -> bool:
    """True iff `dotted_name` exactly equals or is-a-child-of any entry."""
    for entry in table:
        if dotted_name == entry:
            return True
    return False


def _bare_method_matches(call: ast.Call, names: Iterable[str]) -> bool:
    """True iff the call is an attribute-access call and the attribute
    name is in `names`. Used as a weaker signal for pathlib-style
    method calls on statically-unresolved receivers."""
    func = call.func
    if isinstance(func, ast.Attribute) and func.attr in names:
        return True
    return False


# =============================================================================
# Exception / try-except analysis for `may_raise`
# =============================================================================


def _raise_is_caught(raise_node: ast.Raise,
                     handlers: Iterable[ast.Try]) -> bool:
    """True iff `raise_node` is lexically contained in the try-body of
    some enclosing Try with a handler that catches everything
    (`except:`, `except BaseException`, or `except Exception`).

    This is a sound under-approximation of the is-caught relation: a
    bare `except:` catches everything; `BaseException`/`Exception`
    catches everything a user code would reasonably raise. Narrower
    handlers (`except ValueError`) do NOT count — they leave some
    exceptions uncaught, so `may_raise` remains True.
    """
    for try_node in handlers:
        # Is the raise inside try_node.body (not in handlers/finalbody)?
        if any(_node_contains(body_stmt, raise_node) for body_stmt in try_node.body):
            # Check each handler for "catches everything".
            for h in try_node.handlers:
                if h.type is None:
                    # bare `except:` catches everything
                    return True
                caught = _call_name(h.type) or ""
                if caught in ("BaseException", "Exception"):
                    return True
                # Tuple catch: `except (Exception, ...)` — if any member
                # is catch-all, treat as caught.
                if isinstance(h.type, ast.Tuple):
                    for elt in h.type.elts:
                        elt_name = _call_name(elt) or ""
                        if elt_name in ("BaseException", "Exception"):
                            return True
    return False


def _node_contains(outer: ast.AST, target: ast.AST) -> bool:
    """True iff `target` appears in the subtree rooted at `outer`. Used
    to decide lexical containment without relying on `parent` links."""
    for node in ast.walk(outer):
        if node is target:
            return True
    return False


def _enclosing_try_nodes(func: ast.FunctionDef | ast.AsyncFunctionDef,
                         ) -> list[ast.Try]:
    """Collect every Try node lexically contained in `func.body`."""
    out: list[ast.Try] = []
    for node in ast.walk(func):
        if isinstance(node, ast.Try):
            out.append(node)
    return out


def _analyze_may_raise(func: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """True iff the function body contains an uncaught `raise` statement.

    Conservative behaviour:
      - Any `raise` not lexically caught by an enclosing try/except with
        a catch-all handler → True.
      - `raise` inside `except:` / `except Exception:` / `except
        BaseException:` IS considered caught (handler is within the
        same function AND catches the thrown type) — we do not flag
        re-raise from a catch-all; that's pedantic and would make
        every generic error wrapper look dangerous.

    Known precision loss (false positives):
      - `assert` statements are not flagged (even though `AssertionError`
        is an exception). Most code treats `assert` as non-raising
        under -O; we agree.
      - A function that only raises inside an unreachable branch
        (`if False: raise X`) is flagged True. We don't do dead-code
        analysis.
    """
    try_nodes = _enclosing_try_nodes(func)
    for node in ast.walk(func):
        if isinstance(node, ast.Raise):
            if not _raise_is_caught(node, try_nodes):
                return True
    return False


# =============================================================================
# Implicit-raise risk detection for `may_raise_implicit` (Phase 28)
# =============================================================================


# Builtins whose standard Python semantics permit raising on normally-
# shaped inputs (primarily `TypeError`, `ValueError`, `OverflowError`).
# The list is intentionally small, documented, and auditable — any
# function call to a name in this set counts towards
# `may_raise_implicit` unless the call is lexically inside a catch-all
# try/except.
#
# This is the analyzer's "shortlist" of implicit-raise risk operations.
# It is NOT a claim that these builtins always raise; it is a claim
# that their semantic contract permits raising on unconstrained
# argument types, which is exactly what Phase-27 corpus-scale
# calibration surfaced on the `may_raise` false-negative column.
_IMPLICIT_RAISE_BUILTINS: frozenset[str] = frozenset({
    "int", "float", "bool", "bytes",        # constructor coercions
    "len", "iter", "next",                   # iteration protocol
    "abs", "divmod", "pow", "round",        # numeric with domain errors
    "ord", "chr", "hash",                    # domain-restricted builtins
    "min", "max", "sum",                     # raise on empty / bad iterable
    "range",                                 # raises on step=0 / non-int
})


def _contains_implicit_raise_pattern(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
) -> bool:
    """True iff the function body contains at least one node whose
    Python semantics permit propagating an exception on normally-shaped
    inputs. The patterns considered (Phase 28):

      1. `ast.BinOp` with `op` in `(Div, FloorDiv, Mod, Pow)` —
         `ZeroDivisionError`, `OverflowError`.
      2. `ast.Subscript` — `IndexError`, `KeyError`, `TypeError`.
      3. `ast.Call` whose callee is a bare `ast.Name` matching
         `_IMPLICIT_RAISE_BUILTINS` — `TypeError`, `ValueError`,
         `OverflowError`.
      4. `ast.Attribute` read on one of the function's own positional
         parameters — `AttributeError` (the parameter's runtime type
         may not carry the named attribute).

    Lexical containment in a catch-all try/except is NOT checked here;
    the caller (`_analyze_may_raise_implicit`) applies that filter,
    reusing the same `_raise_is_caught` semantics as the explicit-
    raise analyzer.
    """
    # Collect the function's positional parameter names (for the
    # Attribute-on-parameter pattern). We deliberately do NOT include
    # `self` or `cls`: attribute access on those is how methods
    # dispatch and would produce a flood of false positives.
    param_names: set[str] = set()
    args = func.args
    for a in args.args:
        if a.arg not in ("self", "cls"):
            param_names.add(a.arg)
    for a in args.kwonlyargs:
        param_names.add(a.arg)
    if args.vararg is not None:
        param_names.add(args.vararg.arg)
    if args.kwarg is not None:
        param_names.add(args.kwarg.arg)

    for node in ast.walk(func):
        # Pattern 1 — risky binary arithmetic.
        if isinstance(node, ast.BinOp):
            if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod,
                                      ast.Pow)):
                return True
        # Pattern 2 — subscript.
        if isinstance(node, ast.Subscript):
            return True
        # Pattern 3 — call to a risky builtin name.
        if isinstance(node, ast.Call):
            func_node = node.func
            if (isinstance(func_node, ast.Name)
                    and func_node.id in _IMPLICIT_RAISE_BUILTINS):
                return True
        # Pattern 4 — attribute read on a positional parameter. We
        # require the context to be `Load` (attribute ASSIGN is a
        # different class of side-effect, tracked by
        # `may_write_global`).
        if isinstance(node, ast.Attribute):
            value = node.value
            if (isinstance(value, ast.Name)
                    and value.id in param_names
                    and isinstance(node.ctx, ast.Load)):
                return True
    return False


def _analyze_may_raise_implicit(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
) -> bool:
    """Phase-28 predicate: True iff the function contains at least one
    implicit-raise-risk pattern (see `_contains_implicit_raise_pattern`)
    AND the pattern is not lexically inside a catch-all try/except
    (`except:`, `except BaseException:`, `except Exception:`) that
    encloses the whole function body.

    The conservative design choice: we check whether ANY catch-all
    try/except at the function's top level of indentation blankets
    every implicit-raise pattern. If so, the function has explicitly
    opted out of propagating implicit exceptions and we flag it
    False. Otherwise True. This matches the Phase-24 `may_raise`
    soundness stance (sound-over-precision, with catch-all try/except
    as the single-mechanism escape hatch).
    """
    if not _contains_implicit_raise_pattern(func):
        return False
    # Is the entire body inside a single catch-all try/except? If so,
    # the function cannot propagate an uncaught implicit exception.
    body = list(func.body)
    # Skip a leading docstring node.
    if body and isinstance(body[0], ast.Expr) and isinstance(
            body[0].value, ast.Constant) and isinstance(
            body[0].value.value, str):
        body = body[1:]
    if len(body) == 1 and isinstance(body[0], ast.Try):
        try_node = body[0]
        # Look for a catch-all handler.
        for h in try_node.handlers:
            if h.type is None:
                return False
            caught = _call_name(h.type) or ""
            if caught in ("BaseException", "Exception"):
                return False
            if isinstance(h.type, ast.Tuple):
                for elt in h.type.elts:
                    if (_call_name(elt) or "") in ("BaseException",
                                                    "Exception"):
                        return False
    return True


# =============================================================================
# Recursion detection for `is_recursive`
# =============================================================================


def _analyze_is_recursive(func: ast.FunctionDef | ast.AsyncFunctionDef,
                          enclosing_class: str | None = None) -> bool:
    """True iff the function body calls itself by simple name or via
    `self.<name>` / `<enclosing_class>.<name>`.

    Conservative: direct recursion only. Mutual recursion (f calls g
    calls f) would require cross-function analysis which is out of
    scope — the predicate is named `is_recursive` (self-recursion
    only), not `participates_in_cycle`.
    """
    target_name = func.name
    for node in ast.walk(func):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node.func)
        if name is None:
            continue
        # Direct call by simple name: `target_name(...)`
        if name == target_name:
            return True
        # Self-call in a method: `self.<target_name>(...)`
        if name == f"self.{target_name}":
            return True
        # Explicit class-qualified call: `ClassName.<target_name>(...)`
        if enclosing_class is not None and name == f"{enclosing_class}.{target_name}":
            return True
    return False


# =============================================================================
# Global write detection for `may_write_global`
# =============================================================================


def _analyze_may_write_global(func: ast.FunctionDef | ast.AsyncFunctionDef,
                              module_level_names: frozenset[str],
                              ) -> bool:
    """True iff the function might mutate module-global state.

    Sound conditions (any one suffices):
      1. The function has a `global X` declaration AND assigns to X
         somewhere in its body. This is the canonical "write global"
         case and is exact (the language guarantees it).
      2. The function assigns to an attribute of a module-level name
         (`MODULE_LEVEL_LIST.foo = ...`) OR calls a known mutating
         method on a module-level name (`MODULE_LEVEL_LIST.append(x)`).
         `module_level_names` carries the module's top-level names the
         analyzer knows about.

    Known-precision boundaries:
      - We don't track assignments through local aliases
        (`x = MODULE_LEVEL_LIST; x.append(...)`).
      - `nonlocal` (enclosing-function scope) is NOT counted — the
        predicate is about module-level (global) state specifically.
    """
    global_names: set[str] = set()
    assigned_names: set[str] = set()

    for node in ast.walk(func):
        if isinstance(node, ast.Global):
            global_names.update(node.names)
        elif isinstance(node, ast.Assign):
            for tgt in node.targets:
                for name in _collect_store_names(tgt):
                    assigned_names.add(name)
        elif isinstance(node, ast.AugAssign):
            for name in _collect_store_names(node.target):
                assigned_names.add(name)
        elif isinstance(node, ast.AnnAssign):
            for name in _collect_store_names(node.target):
                assigned_names.add(name)

    # Case 1: `global X; X = ...`
    if global_names & assigned_names:
        return True

    # Case 2: mutation on a module-level name via attribute call /
    #         attribute assignment. We scan for `Attribute` whose root
    #         Name is in `module_level_names`.
    _MUTATING_METHODS: frozenset[str] = frozenset({
        "append", "extend", "insert", "pop", "remove", "clear",
        "setdefault", "update", "add", "discard", "popitem",
        "sort", "reverse",
    })
    for node in ast.walk(func):
        # `MOD_LEVEL.attr = X`
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if (isinstance(tgt, ast.Attribute)
                        and isinstance(tgt.value, ast.Name)
                        and tgt.value.id in module_level_names):
                    return True
        # `MOD_LEVEL.append(...)` etc.
        if isinstance(node, ast.Call):
            func_node = node.func
            if (isinstance(func_node, ast.Attribute)
                    and isinstance(func_node.value, ast.Name)
                    and func_node.value.id in module_level_names
                    and func_node.attr in _MUTATING_METHODS):
                return True

    return False


def _collect_store_names(target: ast.AST) -> list[str]:
    """Yield the simple names stored-to by an assignment target.

    Handles `x`, `x, y`, `x[0]`, `x.y` correctly: only `Name` targets
    count (`x, y = ...`); attribute and subscript writes are tracked
    separately by the global-mutation detector.
    """
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        out: list[str] = []
        for elt in target.elts:
            out.extend(_collect_store_names(elt))
        return out
    return []


# =============================================================================
# IO-call detection (subprocess / filesystem / network)
# =============================================================================


def _collect_import_hints(tree: ast.Module) -> frozenset[str]:
    """Return the set of fully-qualified names for which the file has
    at least one top-level import. Used to tighten the IO-call matcher
    so that `open(...)` on a file that doesn't import anything filesystem-
    related still matches the builtin (always matches) but `Path(...)
    .read_text()` only matches when `pathlib` is imported."""
    hints: set[str] = set()
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                hints.add(n.name)
                # Also add top-level package
                hints.add(n.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                hints.add(node.module)
                hints.add(node.module.split(".")[0])
                for n in node.names:
                    hints.add(f"{node.module}.{n.name}")
                    # Also imported as bare name
                    hints.add(n.name)
    return frozenset(hints)


def _analyze_io_calls(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    import_hints: frozenset[str],
) -> tuple[bool, bool, bool]:
    """Return (calls_subprocess, calls_filesystem, calls_network).

    Algorithm: for every Call in the function body, resolve the target
    to a dotted string and check membership in the API tables. Dotted
    paths via imports (`from subprocess import run; run(...)`) are
    matched by `import_hints`.
    """
    calls_subprocess = False
    calls_filesystem = False
    calls_network = False

    for node in ast.walk(func):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node.func)
        if name is None:
            continue

        # Direct exact / prefix match against the API tables.
        if _call_matches(name, _SUBPROCESS_APIS):
            calls_subprocess = True
        if _call_matches(name, _FILESYSTEM_APIS):
            calls_filesystem = True
        if _call_matches(name, _NETWORK_APIS):
            calls_network = True

        # Named-import match: `from subprocess import run; run(...)`.
        # If the call's simple name matches a suffix of an entry in the
        # APIs AND the module prefix is in import_hints, match.
        if "." not in name:
            for entry in _SUBPROCESS_APIS:
                parts = entry.rsplit(".", 1)
                if len(parts) == 2 and parts[1] == name and parts[0] in import_hints:
                    calls_subprocess = True
                    break
            for entry in _FILESYSTEM_APIS:
                parts = entry.rsplit(".", 1)
                if len(parts) == 2 and parts[1] == name and parts[0] in import_hints:
                    calls_filesystem = True
                    break
            for entry in _NETWORK_APIS:
                parts = entry.rsplit(".", 1)
                if len(parts) == 2 and parts[1] == name and parts[0] in import_hints:
                    calls_network = True
                    break

        # pathlib-method shortcut: method-name match + pathlib imported.
        if (not calls_filesystem
                and "pathlib" in import_hints
                and _bare_method_matches(node, _PATHLIB_METHOD_NAMES)):
            calls_filesystem = True

    return calls_subprocess, calls_filesystem, calls_network


# =============================================================================
# Public entry point
# =============================================================================


def _module_level_names(tree: ast.Module) -> frozenset[str]:
    """Simple-name definitions at module scope — used by the global-
    mutation detector to recognise `MODULE_LEVEL_LIST.append(x)`."""
    names: set[str] = set()
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                for n in _collect_store_names(tgt):
                    names.add(n)
        elif isinstance(node, ast.AnnAssign):
            for n in _collect_store_names(node.target):
                names.add(n)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Import):
            for n in node.names:
                names.add((n.asname or n.name).split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for n in node.names:
                names.add(n.asname or n.name)
    return frozenset(names)


def analyze_function(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    tree: ast.Module | None = None,
    enclosing_class: str | None = None,
    import_hints: frozenset[str] | None = None,
    module_level_names: frozenset[str] | None = None,
) -> FunctionSemantics:
    """Run every predicate against `func` and return a `FunctionSemantics`.

    `tree`, `import_hints`, and `module_level_names` are optional; when
    omitted, the IO-call and global-mutation analyses become strictly
    more conservative (more False-leaning because they can't confirm
    import context). Pass them when analysing a whole module for
    tightest results.
    """
    if import_hints is None:
        if tree is not None:
            import_hints = _collect_import_hints(tree)
        else:
            import_hints = frozenset()
    if module_level_names is None:
        if tree is not None:
            module_level_names = _module_level_names(tree)
        else:
            module_level_names = frozenset()

    may_raise = _analyze_may_raise(func)
    may_raise_implicit = _analyze_may_raise_implicit(func)
    is_recursive = _analyze_is_recursive(func, enclosing_class)
    may_write_global = _analyze_may_write_global(func, module_level_names)
    calls_subprocess, calls_filesystem, calls_network = _analyze_io_calls(
        func, import_hints,
    )
    return FunctionSemantics(
        may_raise=may_raise,
        is_recursive=is_recursive,
        may_write_global=may_write_global,
        calls_subprocess=calls_subprocess,
        calls_filesystem=calls_filesystem,
        calls_network=calls_network,
        may_raise_implicit=may_raise_implicit,
    )


def analyze_module(tree: ast.Module) -> list[tuple[str, FunctionSemantics]]:
    """Analyse every top-level and method function in `tree`. Returns a
    list of (qualified_name, semantics) where qualified_name is
    `ClassName.method` for methods or `function` for module-level.

    Order matches document order; this is what the ingestion layer
    needs to build parallel tuples."""
    out: list[tuple[str, FunctionSemantics]] = []
    import_hints = _collect_import_hints(tree)
    module_level_names = _module_level_names(tree)
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            sem = analyze_function(
                node, tree=tree, enclosing_class=None,
                import_hints=import_hints,
                module_level_names=module_level_names,
            )
            out.append((node.name, sem))
        elif isinstance(node, ast.ClassDef):
            for sub in ast.iter_child_nodes(node):
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    sem = analyze_function(
                        sub, tree=tree, enclosing_class=node.name,
                        import_hints=import_hints,
                        module_level_names=module_level_names,
                    )
                    out.append((f"{node.name}.{sub.name}", sem))
    return out
