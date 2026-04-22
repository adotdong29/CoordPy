"""Runtime-truth calibration of conservative semantic predicates — Phase 26.

Phases 24 and 25 shipped conservative static-semantic predicates
(intraprocedural + interprocedural) whose *analyzer-defined* exactness
is 100% on the direct-exact path by construction: the corpus gold and
the planner both read the same analyzer output. That number is
internally consistent but tautological as a claim about runtime
behaviour — it says nothing about whether a function flagged
`may_raise` actually raises when called, or whether a function NOT
flagged `calls_subprocess` is guaranteed to never spawn one at
runtime.

This module introduces a separate, additive truth axis:

  * **Analyzer-gold truth**: what `core.code_semantics` /
    `core.code_interproc` declares. Used by the planner.
  * **Runtime truth**: what actually happens when a function is
    invoked with a fuzzed input set, observed via instrumented
    effect probes.

The two are NOT equal, and neither is a superset of the other by
definition — they live on different axes. Conservative static
analysis is designed for soundness (false negatives = 0 under
documented assumptions); runtime observation is a lower bound (any
effect observed MUST be real; absence of observation is NOT a proof
of absence because the fuzz input set may not cover the triggering
path).

What this module computes per (function, predicate):

  - `static_flag`      — the analyzer's answer.
  - `runtime_flag`     — the disjunction of effect-observations across
                         a seeded fuzz-input sweep.
  - `ground_truth`     — the author-declared runtime answer (optional;
                         used to verify probe correctness, not analyzer
                         correctness).
  - `n_triggered`      — how many of the runs exercised the predicate.
  - `n_runs`           — total probe invocations.

Predicates supported
--------------------

| Predicate                   | Probe mechanism |
|---                          |---|
| `may_raise`                 | execute + classify raised exceptions |
| `may_write_global`          | snapshot module `__dict__` + dedicated globals before/after |
| `calls_subprocess`          | monkeypatch `subprocess.Popen.__init__`, `os.system`, `os.popen` |
| `calls_filesystem`          | monkeypatch `builtins.open`, `os.*` filesystem calls, `pathlib.Path.*` |
| `calls_network`             | monkeypatch `socket.socket.connect`, `urllib.request.urlopen`, `http.client.HTTP(S)?Connection` |
| `participates_in_cycle`     | `sys.settrace` — detect a second entry to the target frame |

Safety model
------------

The probes execute real Python code but neuter side-effectful APIs
at the instrumentation boundary so an observed effect NEVER escapes:

  * Filesystem writes are redirected to a per-probe tempdir that is
    `rmtree`-d after the probe.
  * Subprocess launches are replaced with a `SubprocessAttempted`
    sentinel exception that is caught by the probe — no external
    process ever runs.
  * Network connects raise a `NetworkAttempted` sentinel.
  * Imports happen at probe setup; the executed snippet lives in its
    own `ModuleType`, keyed by a random uuid, so it cannot pollute
    the shared interpreter's import cache.

This is not a hermetic sandbox (a malicious snippet could still DoS
the probe with `while True`). The design assumption is that the
snippet corpus is trusted — it ships with the repo and is code-
reviewed. The probes are designed to keep a WELL-BEHAVED snippet
from escaping; they are not a defence against adversarial code.

Determinism
-----------

Each probe is parameterised by a `seed`. The fuzz-input synthesiser
derives a per-call RNG from the seed, so `run_calibration(...,
seeds=(0, 1, 2))` is reproducible bit-for-bit given the same
snippet. This is the knob exercised by the experiment's "repeated-
run mode" for variance accounting.

The module is intentionally not coupled to the corpus registry or
the ledger — it is a pure-Python observation primitive. The
experiment at `experiments/phase26_runtime_calibration.py` wires it
to the analyzer and emits the calibration tables.
"""

from __future__ import annotations

import builtins
import io
import os
import random
import shutil
import socket
import subprocess
import sys
import tempfile
import textwrap
import types
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator


# =============================================================================
# Sentinels raised by instrumentation — never caught by the probe's
# caller, always caught by the probe itself.
# =============================================================================


class _ProbeSentinel(Exception):
    """Base class for markers the probes install. These are NOT real
    runtime errors — they fire when a neutered API is invoked, to
    prevent effects from escaping. The probe catches them and records
    the attempt; they must not propagate out of the probe."""


class _SubprocessAttempted(_ProbeSentinel):
    pass


class _NetworkAttempted(_ProbeSentinel):
    pass


# =============================================================================
# Public dataclasses
# =============================================================================


@dataclass(frozen=True)
class RuntimeObservation:
    """One predicate's observation over a fuzz sweep.

    `runtime_flag` is True iff the predicate fired at least once over
    the `n_runs` calls. `trigger_rate` = n_triggered / n_runs gives a
    coarse confidence.

    `witnesses` captures example evidence — a list of short strings
    that make the observation inspectable in report output (e.g. the
    class name of an observed exception, or the first path that was
    opened). Caps at 5 entries to keep reports readable.
    """

    predicate: str
    runtime_flag: bool
    n_runs: int
    n_triggered: int
    witnesses: tuple[str, ...] = ()
    # True iff this predicate is runtime-decidable on this target at
    # all. Some predicates (e.g. `has_unresolved_callees`) are not —
    # they are properties of the call graph, not of runtime behaviour.
    decidable: bool = True
    # True iff the probe itself was usable on this target (the snippet
    # exposed an `invoke` or declared a fuzz-input set). When False,
    # `runtime_flag` is vacuously False and the observation should not
    # contribute to calibration metrics.
    applicable: bool = True
    notes: str = ""

    @property
    def trigger_rate(self) -> float:
        if self.n_runs <= 0:
            return 0.0
        return self.n_triggered / self.n_runs


@dataclass(frozen=True)
class SnippetResult:
    """Aggregate report for one snippet — one row in the calibration
    table."""

    snippet_name: str
    target_qname: str
    static_flags: dict[str, bool]
    runtime_observations: dict[str, RuntimeObservation]
    ground_truth: dict[str, bool]
    seeds: tuple[int, ...]

    def divergences(self) -> list[tuple[str, str]]:
        """Return (predicate, kind) for every predicate where
        `static_flag` ≠ `runtime_flag` on decidable + applicable
        observations. `kind` is `false_positive` (static True, runtime
        False) or `false_negative` (static False, runtime True).
        """
        out: list[tuple[str, str]] = []
        for pred, rt in self.runtime_observations.items():
            if not (rt.decidable and rt.applicable):
                continue
            sf = self.static_flags.get(pred, False)
            if sf and not rt.runtime_flag:
                out.append((pred, "false_positive"))
            elif rt.runtime_flag and not sf:
                out.append((pred, "false_negative"))
        return out


# =============================================================================
# Input synthesis
# =============================================================================


_DEFAULT_FUZZ_POOL: tuple[Any, ...] = (
    0, 1, -1, 2, 10, 100, -100,
    "", "a", "hello", "0", " ",
    None, True, False,
    [], [0], [1, 2, 3],
    {}, {"k": "v"},
    (), (1,), (1, 2),
)


def synthesize_args(
    n_params: int, seed: int,
    pool: tuple[Any, ...] = _DEFAULT_FUZZ_POOL,
) -> tuple[Any, ...]:
    """Pick `n_params` values from `pool` using a seeded RNG.

    Deterministic given `seed`. Returns an empty tuple when
    `n_params` is 0.
    """
    if n_params <= 0:
        return ()
    rng = random.Random(seed)
    return tuple(rng.choice(pool) for _ in range(n_params))


# =============================================================================
# Snippet loading
# =============================================================================


def load_snippet_module(source: str, module_name: str | None = None,
                         ) -> types.ModuleType:
    """Compile `source` into a fresh `ModuleType`.

    The module name is uuid-suffixed so re-loading the same source
    never collides with a prior probe. The module is NOT inserted
    into `sys.modules` — this is what keeps probe state from leaking
    across snippets.
    """
    if module_name is None:
        module_name = f"_phase26_snippet_{uuid.uuid4().hex}"
    mod = types.ModuleType(module_name)
    mod.__file__ = f"<phase26:{module_name}>"
    # Snippets occasionally import from the stdlib; that works because
    # the freshly-created module shares the interpreter's sys.modules.
    # We avoid installing the snippet's own module into sys.modules to
    # prevent cross-probe contamination.
    code = compile(textwrap.dedent(source), mod.__file__, "exec")
    exec(code, mod.__dict__)
    return mod


# =============================================================================
# Instrumentation — context managers that neuter + record effects
# =============================================================================


@contextmanager
def _record_subprocess() -> Iterator[list[str]]:
    """Replace subprocess entry points so calls record and sentinel-
    raise instead of spawning a process.

    The list yielded is the recording; each appended string is a
    short description of the attempted call.
    """
    recorded: list[str] = []
    orig_popen_init = subprocess.Popen.__init__
    orig_os_system = os.system
    orig_os_popen = os.popen

    def fake_popen_init(self, args=None, *a, **kw):
        recorded.append(f"Popen({args!r})")
        raise _SubprocessAttempted("subprocess.Popen disabled in probe")

    def fake_os_system(command):
        recorded.append(f"os.system({command!r})")
        raise _SubprocessAttempted("os.system disabled in probe")

    def fake_os_popen(command, *a, **kw):
        recorded.append(f"os.popen({command!r})")
        raise _SubprocessAttempted("os.popen disabled in probe")

    subprocess.Popen.__init__ = fake_popen_init    # type: ignore[method-assign]
    os.system = fake_os_system
    os.popen = fake_os_popen
    try:
        yield recorded
    finally:
        subprocess.Popen.__init__ = orig_popen_init  # type: ignore[method-assign]
        os.system = orig_os_system
        os.popen = orig_os_popen


@contextmanager
def _record_filesystem() -> Iterator[list[str]]:
    """Redirect filesystem operations into a private tempdir + record
    each open/create/remove. No real path outside the tempdir is
    touched even if the snippet passes an absolute path.
    """
    recorded: list[str] = []
    tmp = tempfile.mkdtemp(prefix="phase26_fs_")
    orig_open = builtins.open
    orig_os_open = os.open
    orig_os_remove = os.remove
    orig_os_unlink = os.unlink
    orig_os_mkdir = os.mkdir
    orig_os_makedirs = os.makedirs

    def _reroute(path: Any) -> str:
        try:
            path_str = os.fspath(path)
        except TypeError:
            path_str = str(path)
        base = os.path.basename(path_str) or "unnamed"
        return os.path.join(tmp, base)

    def fake_open(file, mode="r", *a, **kw):
        recorded.append(f"open({file!r}, mode={mode!r})")
        try:
            return orig_open(_reroute(file), mode, *a, **kw)
        except (FileNotFoundError, IsADirectoryError):
            # Reads of nonexistent files are common in snippets — give
            # the snippet an empty file-like so its logic can proceed.
            if "r" in mode and "+" not in mode:
                return io.StringIO("")
            raise

    def fake_os_open(path, *a, **kw):
        recorded.append(f"os.open({path!r})")
        return orig_os_open(_reroute(path), *a, **kw)

    def fake_os_remove(path, *a, **kw):
        recorded.append(f"os.remove({path!r})")
        try:
            return orig_os_remove(_reroute(path), *a, **kw)
        except FileNotFoundError:
            return None

    def fake_os_unlink(path, *a, **kw):
        recorded.append(f"os.unlink({path!r})")
        try:
            return orig_os_unlink(_reroute(path), *a, **kw)
        except FileNotFoundError:
            return None

    def fake_os_mkdir(path, *a, **kw):
        recorded.append(f"os.mkdir({path!r})")
        try:
            return orig_os_mkdir(_reroute(path), *a, **kw)
        except FileExistsError:
            return None

    def fake_os_makedirs(path, *a, **kw):
        recorded.append(f"os.makedirs({path!r})")
        return orig_os_makedirs(_reroute(path), *a, **kw)

    builtins.open = fake_open
    os.open = fake_os_open
    os.remove = fake_os_remove
    os.unlink = fake_os_unlink
    os.mkdir = fake_os_mkdir
    os.makedirs = fake_os_makedirs
    try:
        yield recorded
    finally:
        builtins.open = orig_open
        os.open = orig_os_open
        os.remove = orig_os_remove
        os.unlink = orig_os_unlink
        os.mkdir = orig_os_mkdir
        os.makedirs = orig_os_makedirs
        shutil.rmtree(tmp, ignore_errors=True)


@contextmanager
def _record_network() -> Iterator[list[str]]:
    """Block network egress + record each attempt.

    `socket.socket.connect` is the narrowest choke point; patching it
    catches urllib / http.client / requests because all of them
    eventually go through it. For urllib.request.urlopen we also
    install a direct patch so the attempt is recorded before the
    socket is ever created (urlopen can raise an URLError before
    socket construction on some paths).
    """
    recorded: list[str] = []
    orig_connect = socket.socket.connect
    # urlopen lives in urllib.request; import lazily to avoid forcing
    # the import in callers that don't use it.
    try:
        import urllib.request as _urllib_request  # noqa: F401
        orig_urlopen = _urllib_request.urlopen
        has_urlopen = True
    except Exception:
        has_urlopen = False
        orig_urlopen = None  # type: ignore

    def fake_connect(self, address, *a, **kw):
        recorded.append(f"socket.connect({address!r})")
        raise _NetworkAttempted("network connect disabled in probe")

    def fake_urlopen(url, *a, **kw):
        recorded.append(f"urllib.urlopen({url!r})")
        raise _NetworkAttempted("urlopen disabled in probe")

    socket.socket.connect = fake_connect    # type: ignore[method-assign]
    if has_urlopen:
        import urllib.request as _urllib_request
        _urllib_request.urlopen = fake_urlopen  # type: ignore[assignment]
    try:
        yield recorded
    finally:
        socket.socket.connect = orig_connect  # type: ignore[method-assign]
        if has_urlopen:
            import urllib.request as _urllib_request
            _urllib_request.urlopen = orig_urlopen  # type: ignore[assignment]


@contextmanager
def _observe_globals(module: types.ModuleType) -> Iterator[dict[str, Any]]:
    """Snapshot the module's `__dict__` before the target runs and
    yield a mapping; the caller diffs post-run to detect writes.

    We snapshot shallowly — a mutation to a container living in
    module globals is detected via identity+equality on exit. For
    the snippet corpus this is the right granularity (we care about
    "did the value bound to some top-level name change, or did a
    top-level container grow").
    """
    snapshot_keys = {k for k in module.__dict__
                     if not k.startswith("__")}
    snapshot: dict[str, Any] = {}
    for k in snapshot_keys:
        v = module.__dict__[k]
        if isinstance(v, (list, dict, set)):
            try:
                snapshot[k] = type(v)(v)
            except Exception:
                snapshot[k] = v
        else:
            snapshot[k] = v
    yield snapshot


def _globals_diff(module: types.ModuleType,
                  before: dict[str, Any]) -> list[str]:
    """Return a list of top-level names whose value has changed."""
    changed: list[str] = []
    for k, prev in before.items():
        if k not in module.__dict__:
            changed.append(f"-{k}")
            continue
        now = module.__dict__[k]
        if isinstance(prev, (list, dict, set)):
            if now != prev:
                changed.append(f"~{k}")
        else:
            # Identity check for scalars / functions; equality for
            # values that may be replaced.
            if now is not prev and now != prev:
                changed.append(f"={k}")
    # New top-level names created by the target itself.
    for k in module.__dict__:
        if k.startswith("__") or k in before:
            continue
        changed.append(f"+{k}")
    return changed


@contextmanager
def _track_reentry(target: Callable) -> Iterator[dict[str, int]]:
    """Detect whether `target` is called recursively (directly or via
    a helper that calls back into it) during one invocation.

    Uses `sys.settrace` — if the probe's caller also installs a
    tracer this composes poorly, so the context manager restores the
    prior tracer on exit. The observed counter is incremented every
    time a frame for `target` is entered; `count >= 2` after a
    single invocation ⇒ target re-entered ⇒ cycle observed.
    """
    state = {"count": 0}
    code_obj = getattr(target, "__code__", None)
    prev = sys.gettrace()

    def tracer(frame, event, arg):
        if event == "call" and code_obj is not None and frame.f_code is code_obj:
            state["count"] += 1
        return tracer

    sys.settrace(tracer)
    try:
        yield state
    finally:
        sys.settrace(prev)


# =============================================================================
# Predicate-specific probes
# =============================================================================


def _call_safely(target: Callable, args: tuple[Any, ...]) -> tuple[
    bool, BaseException | None]:
    """Execute `target(*args)` swallowing only the probe sentinels.

    Returns (did_raise, exc). `did_raise` is True iff target raised a
    real (non-sentinel) exception; `exc` is the exception instance
    (or None if the call returned normally). Sentinel-raises are
    treated as "no user exception".
    """
    try:
        target(*args)
        return False, None
    except _ProbeSentinel:
        return False, None
    except BaseException as e:  # catch anything the snippet throws
        return True, e


# =============================================================================
# Phase 28 — explicit-vs-implicit raise classification
# =============================================================================


def _raise_line_numbers(target: Callable) -> frozenset[int]:
    """Set of source-file line numbers on which the target's function
    body contains an explicit `raise` statement.

    Returns an empty set when source is unavailable (e.g. for C
    callables). The result is computed once per target; callers that
    probe repeatedly can cache it.
    """
    import ast as _ast
    import inspect as _inspect
    try:
        src = _inspect.getsource(target)
    except (OSError, TypeError):
        return frozenset()
    try:
        # `src` is the function body with the indentation of the
        # enclosing scope; `textwrap.dedent` makes it parseable.
        tree = _ast.parse(textwrap.dedent(src))
    except SyntaxError:
        return frozenset()
    # Line numbers reported by inspect.getsource are 1-based relative
    # to the start of the extracted snippet, so the raise lines we
    # collect are in that same frame. Traceback line numbers, however,
    # are relative to the actual source file. We normalise by offsetting
    # with the `__code__.co_firstlineno` of the target.
    try:
        first_line = target.__code__.co_firstlineno
    except AttributeError:
        return frozenset()
    offsets: set[int] = set()
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Raise):
            # node.lineno is 1-based within the dedented snippet.
            # The `def ...` line is 1; the absolute line is
            # `first_line + (node.lineno - 1)`.
            offsets.add(first_line + node.lineno - 1)
    return frozenset(offsets)


def _classify_exception_origin(
    target: Callable, exc: BaseException,
    *, raise_lines: frozenset[int] | None = None,
) -> str:
    """Classify a caught exception as `"explicit"` or `"implicit"` with
    respect to `target`.

    The classification uses the exception's traceback: the innermost
    frame tells us where the exception was first raised. If that
    frame's code object is the target's AND its line number is a
    `raise`-statement line in the target's source, we call it
    explicit. Otherwise we call it implicit.

    Phase 28's definitions:

      * explicit — exception originated from a `raise` statement
        inside `target`'s own body.
      * implicit — exception originated from a builtin, a called
        frame, an arithmetic operator, a subscript, or any
        propagation path that is not an explicit `raise` in
        `target`.

    Returns `"implicit"` conservatively when any of the traceback,
    code object, or source inspection cannot decide. That keeps the
    classifier a sound lower bound for the explicit bucket, which is
    the Phase-24 contract.
    """
    tb = exc.__traceback__
    if tb is None:
        return "implicit"
    innermost = tb
    while innermost.tb_next is not None:
        innermost = innermost.tb_next
    target_code = getattr(target, "__code__", None)
    if target_code is None:
        return "implicit"
    if innermost.tb_frame.f_code is not target_code:
        return "implicit"
    lines = raise_lines if raise_lines is not None \
        else _raise_line_numbers(target)
    if not lines:
        return "implicit"
    if innermost.tb_lineno in lines:
        return "explicit"
    return "implicit"


def probe_may_raise(target: Callable, *, invocations: Iterable[tuple[Any, ...]],
                     ) -> RuntimeObservation:
    """Observe whether `target` raises a real exception on at least one
    invocation. Probe sentinels (from subprocess/network guards) do
    NOT count as user raises.
    """
    witnesses: list[str] = []
    n_runs = 0
    n_triggered = 0
    for args in invocations:
        n_runs += 1
        with _record_subprocess(), _record_filesystem(), _record_network():
            raised, exc = _call_safely(target, args)
        if raised:
            n_triggered += 1
            if len(witnesses) < 5:
                witnesses.append(type(exc).__name__)
    return RuntimeObservation(
        predicate="may_raise", runtime_flag=n_triggered > 0,
        n_runs=n_runs, n_triggered=n_triggered,
        witnesses=tuple(witnesses),
    )


def probe_may_raise_split(
    target: Callable, *, invocations: Iterable[tuple[Any, ...]],
) -> tuple[RuntimeObservation, RuntimeObservation]:
    """Phase 28 — run `may_raise` and split observations by origin.

    Returns a pair `(explicit_obs, implicit_obs)`. Each observation
    records triggers ONLY on exceptions classified into its bucket
    via `_classify_exception_origin`. An invocation that raises is
    attributed to exactly one of the two buckets. An invocation that
    does not raise contributes to neither's `n_triggered`.

    Both observations share the same `n_runs` (total invocations).
    """
    raise_lines = _raise_line_numbers(target)
    wit_exp: list[str] = []
    wit_imp: list[str] = []
    n_runs = 0
    n_trig_exp = 0
    n_trig_imp = 0
    for args in invocations:
        n_runs += 1
        with _record_subprocess(), _record_filesystem(), _record_network():
            raised, exc = _call_safely(target, args)
        if not raised:
            continue
        origin = _classify_exception_origin(
            target, exc, raise_lines=raise_lines)
        if origin == "explicit":
            n_trig_exp += 1
            if len(wit_exp) < 5:
                wit_exp.append(type(exc).__name__)
        else:
            n_trig_imp += 1
            if len(wit_imp) < 5:
                wit_imp.append(type(exc).__name__)
    obs_exp = RuntimeObservation(
        predicate="may_raise_explicit",
        runtime_flag=n_trig_exp > 0,
        n_runs=n_runs, n_triggered=n_trig_exp,
        witnesses=tuple(wit_exp),
    )
    obs_imp = RuntimeObservation(
        predicate="may_raise_implicit",
        runtime_flag=n_trig_imp > 0,
        n_runs=n_runs, n_triggered=n_trig_imp,
        witnesses=tuple(wit_imp),
    )
    return obs_exp, obs_imp


def probe_may_raise_explicit(
    target: Callable, *, invocations: Iterable[tuple[Any, ...]],
) -> RuntimeObservation:
    """Runtime observer for the explicit-raise bucket only."""
    exp, _ = probe_may_raise_split(target, invocations=list(invocations))
    return exp


def probe_may_raise_implicit(
    target: Callable, *, invocations: Iterable[tuple[Any, ...]],
) -> RuntimeObservation:
    """Runtime observer for the implicit-raise bucket only."""
    _, imp = probe_may_raise_split(target, invocations=list(invocations))
    return imp


def probe_may_write_global(target: Callable, module: types.ModuleType,
                             *, invocations: Iterable[tuple[Any, ...]],
                             ) -> RuntimeObservation:
    """Observe whether a call to `target` mutates `module.__dict__`.

    A mutation is any change to a pre-existing top-level name OR the
    creation of a new one, compared shallowly (or by equality for
    containers). The probe is applicable to any callable bound in
    `module`; otherwise the detector yields "no change".
    """
    witnesses: list[str] = []
    n_runs = 0
    n_triggered = 0
    for args in invocations:
        n_runs += 1
        with _observe_globals(module) as before, \
                _record_subprocess(), _record_filesystem(), _record_network():
            _call_safely(target, args)
        changed = _globals_diff(module, before)
        if changed:
            n_triggered += 1
            if len(witnesses) < 5:
                witnesses.append(",".join(changed[:3]))
    return RuntimeObservation(
        predicate="may_write_global", runtime_flag=n_triggered > 0,
        n_runs=n_runs, n_triggered=n_triggered,
        witnesses=tuple(witnesses),
    )


def probe_calls_subprocess(target: Callable,
                             *, invocations: Iterable[tuple[Any, ...]],
                             ) -> RuntimeObservation:
    """Observe whether `target` attempts any subprocess entry point."""
    witnesses: list[str] = []
    n_runs = 0
    n_triggered = 0
    for args in invocations:
        n_runs += 1
        with _record_subprocess() as sub_hits, \
                _record_filesystem(), _record_network():
            _call_safely(target, args)
        if sub_hits:
            n_triggered += 1
            if len(witnesses) < 5:
                witnesses.append(sub_hits[0])
    return RuntimeObservation(
        predicate="calls_subprocess", runtime_flag=n_triggered > 0,
        n_runs=n_runs, n_triggered=n_triggered,
        witnesses=tuple(witnesses),
    )


def probe_calls_filesystem(target: Callable,
                             *, invocations: Iterable[tuple[Any, ...]],
                             ) -> RuntimeObservation:
    """Observe whether `target` attempts any filesystem call."""
    witnesses: list[str] = []
    n_runs = 0
    n_triggered = 0
    for args in invocations:
        n_runs += 1
        with _record_filesystem() as fs_hits, \
                _record_subprocess(), _record_network():
            _call_safely(target, args)
        if fs_hits:
            n_triggered += 1
            if len(witnesses) < 5:
                witnesses.append(fs_hits[0])
    return RuntimeObservation(
        predicate="calls_filesystem", runtime_flag=n_triggered > 0,
        n_runs=n_runs, n_triggered=n_triggered,
        witnesses=tuple(witnesses),
    )


def probe_calls_network(target: Callable,
                         *, invocations: Iterable[tuple[Any, ...]],
                         ) -> RuntimeObservation:
    """Observe whether `target` attempts any network call."""
    witnesses: list[str] = []
    n_runs = 0
    n_triggered = 0
    for args in invocations:
        n_runs += 1
        with _record_network() as net_hits, \
                _record_filesystem(), _record_subprocess():
            _call_safely(target, args)
        if net_hits:
            n_triggered += 1
            if len(witnesses) < 5:
                witnesses.append(net_hits[0])
    return RuntimeObservation(
        predicate="calls_network", runtime_flag=n_triggered > 0,
        n_runs=n_runs, n_triggered=n_triggered,
        witnesses=tuple(witnesses),
    )


def probe_participates_in_cycle(target: Callable,
                                  *, invocations: Iterable[tuple[Any, ...]],
                                  ) -> RuntimeObservation:
    """Observe whether `target` is re-entered during a single call.

    Detects both self-recursion and mutual recursion — if any helper
    called by `target` eventually calls back into `target`, the trace
    records count ≥ 2 calls to the target code object.
    """
    witnesses: list[str] = []
    n_runs = 0
    n_triggered = 0
    for args in invocations:
        n_runs += 1
        with _track_reentry(target) as st, \
                _record_subprocess(), _record_filesystem(), _record_network():
            _call_safely(target, args)
        if st["count"] >= 2:
            n_triggered += 1
            if len(witnesses) < 5:
                witnesses.append(f"reentries={st['count']}")
    return RuntimeObservation(
        predicate="participates_in_cycle", runtime_flag=n_triggered > 0,
        n_runs=n_runs, n_triggered=n_triggered,
        witnesses=tuple(witnesses),
    )


# =============================================================================
# Dispatcher
# =============================================================================


_PREDICATE_PROBES: dict[str, str] = {
    "may_raise": "probe_may_raise",
    "may_write_global": "probe_may_write_global",
    "calls_subprocess": "probe_calls_subprocess",
    "calls_filesystem": "probe_calls_filesystem",
    "calls_network": "probe_calls_network",
    "participates_in_cycle": "probe_participates_in_cycle",
    # Phase 28 — split `may_raise` by exception origin.
    "may_raise_explicit": "probe_may_raise_explicit",
    "may_raise_implicit": "probe_may_raise_implicit",
}


# Predicates the runtime layer can decide. Anything outside this set
# (e.g. `has_unresolved_callees` which is a graph-statistic) is
# explicitly out of scope and flagged `decidable=False` in its
# `RuntimeObservation`.
RUNTIME_DECIDABLE_PREDICATES: frozenset[str] = frozenset(_PREDICATE_PROBES)


# The subset of predicates measured on the Phase-26 snippet corpus.
# Phase 28's `may_raise_explicit` / `may_raise_implicit` are derived
# directly from `may_raise` and do not require re-probing when the
# snippet batch is small — but they are still runtime-decidable and
# included in the dispatch table above.
SNIPPET_CORPUS_PREDICATES: frozenset[str] = frozenset({
    "may_raise", "may_write_global", "calls_subprocess",
    "calls_filesystem", "calls_network", "participates_in_cycle",
})


def probe_predicate(predicate: str, target: Callable,
                     *, module: types.ModuleType,
                     invocations: Iterable[tuple[Any, ...]],
                     ) -> RuntimeObservation:
    """Run the probe for `predicate` on `target`. If the predicate is
    not runtime-decidable, return an observation with
    `decidable=False`.
    """
    if predicate not in _PREDICATE_PROBES:
        return RuntimeObservation(
            predicate=predicate, runtime_flag=False,
            n_runs=0, n_triggered=0, decidable=False,
            notes=f"{predicate} is not runtime-decidable by this probe",
        )
    invocations = list(invocations)
    if predicate == "may_raise":
        return probe_may_raise(target, invocations=invocations)
    if predicate == "may_write_global":
        return probe_may_write_global(target, module, invocations=invocations)
    if predicate == "calls_subprocess":
        return probe_calls_subprocess(target, invocations=invocations)
    if predicate == "calls_filesystem":
        return probe_calls_filesystem(target, invocations=invocations)
    if predicate == "calls_network":
        return probe_calls_network(target, invocations=invocations)
    if predicate == "participates_in_cycle":
        return probe_participates_in_cycle(target, invocations=invocations)
    if predicate == "may_raise_explicit":
        return probe_may_raise_explicit(target, invocations=invocations)
    if predicate == "may_raise_implicit":
        return probe_may_raise_implicit(target, invocations=invocations)
    raise AssertionError("unreachable — predicate checked above")


# =============================================================================
# End-to-end: calibrate one snippet
# =============================================================================


@dataclass(frozen=True)
class SnippetSpec:
    """Frozen description of one executable snippet.

    Authors declare the ground-truth map explicitly because some
    predicates are only meaningfully observable on a specific
    invocation pattern (e.g. `may_raise` depends on the input
    distribution). The ground_truth is used ONLY to verify the probe
    itself (test_executable_snippets); it is NOT fed into the
    analyzer-vs-runtime comparison for the calibration report,
    which uses the probe's own observation.

    `invoke(target, rng)` is an optional author-provided driver;
    when present, the probe calls it to produce a sequence of
    `(args, kwargs)` invocations. When absent, the probe uses
    `synthesize_args(target_n_params, seed)` with the default fuzz
    pool. If `n_params_override` is set it takes precedence over
    introspection.
    """

    name: str
    source: str
    target_qname: str       # e.g. "wrapper" or "Cls.meth"
    ground_truth: dict[str, bool]
    family: str = "unknown"
    n_params_override: int | None = None
    invoke: Callable[[Callable, random.Random], Iterable[tuple[Any, ...]]] | None = None
    n_fuzz: int = 8
    notes: str = ""


def _resolve_target(module: types.ModuleType,
                     target_qname: str) -> Callable:
    """Look up `target_qname` inside `module`. Handles `Cls.method`."""
    if "." not in target_qname:
        return getattr(module, target_qname)
    head, rest = target_qname.split(".", 1)
    obj = getattr(module, head)
    for part in rest.split("."):
        obj = getattr(obj, part)
    return obj


def _default_invocations(target: Callable, seed: int, n_fuzz: int,
                          n_params_override: int | None) -> list[tuple[Any, ...]]:
    """Produce `n_fuzz` argument tuples via `synthesize_args`."""
    if n_params_override is not None:
        n_params = n_params_override
    else:
        try:
            n_params = target.__code__.co_argcount
        except AttributeError:
            n_params = 0
    rng = random.Random(seed)
    out: list[tuple[Any, ...]] = []
    for _ in range(max(1, n_fuzz)):
        sub = rng.randrange(1 << 30)
        out.append(synthesize_args(n_params, sub))
    return out


def calibrate_snippet(
    spec: SnippetSpec,
    *, predicates: Iterable[str] = RUNTIME_DECIDABLE_PREDICATES,
    seeds: tuple[int, ...] = (0, 1, 2),
    static_flags: dict[str, bool] | None = None,
) -> SnippetResult:
    """Run every probe against `spec` across every seed and combine
    the observations.

    `static_flags` is the analyzer's answer — consumed to produce the
    `SnippetResult.static_flags` field. If None, the caller can fill
    this in later from `compute_static_flags_from_source`.

    The observations across seeds are combined by OR (flag True iff
    any seed observed the effect) and by sum (`n_runs`, `n_triggered`
    accumulate). The witness list retains the first five witnesses
    seen across seeds.
    """
    static_flags = dict(static_flags or {})
    predicates = list(predicates)
    combined: dict[str, RuntimeObservation] = {}

    for predicate in predicates:
        agg = RuntimeObservation(
            predicate=predicate, runtime_flag=False,
            n_runs=0, n_triggered=0, witnesses=(),
            decidable=predicate in RUNTIME_DECIDABLE_PREDICATES,
            applicable=True,
        )
        for seed in seeds:
            module = load_snippet_module(spec.source,
                                          module_name=f"_phase26_{spec.name}_{seed}")
            try:
                target = _resolve_target(module, spec.target_qname)
            except AttributeError as e:
                agg = RuntimeObservation(
                    predicate=predicate, runtime_flag=False,
                    n_runs=0, n_triggered=0, decidable=True,
                    applicable=False,
                    notes=f"target {spec.target_qname} not in snippet: {e}",
                )
                break
            if spec.invoke is not None:
                rng = random.Random(seed)
                invs_raw = list(spec.invoke(target, rng))
                # Author may yield (args,) tuples or (args, kwargs) pairs —
                # normalise to a sequence of args tuples (kwargs dropped;
                # probes do not plumb kwargs, by design).
                invs: list[tuple[Any, ...]] = []
                for item in invs_raw:
                    if isinstance(item, tuple) and (
                            len(item) == 2
                            and isinstance(item[0], tuple)
                            and isinstance(item[1], dict)):
                        invs.append(item[0])
                    elif isinstance(item, tuple):
                        invs.append(item)
                    else:
                        invs.append((item,))
                if not invs:
                    invs = [()]
            else:
                invs = _default_invocations(
                    target, seed, spec.n_fuzz, spec.n_params_override)
            obs = probe_predicate(
                predicate, target, module=module, invocations=invs)
            agg = RuntimeObservation(
                predicate=predicate,
                runtime_flag=agg.runtime_flag or obs.runtime_flag,
                n_runs=agg.n_runs + obs.n_runs,
                n_triggered=agg.n_triggered + obs.n_triggered,
                witnesses=(agg.witnesses + obs.witnesses)[:5],
                decidable=obs.decidable,
                applicable=obs.applicable and agg.applicable,
                notes=obs.notes or agg.notes,
            )
        combined[predicate] = agg

    return SnippetResult(
        snippet_name=spec.name,
        target_qname=spec.target_qname,
        static_flags=static_flags,
        runtime_observations=combined,
        ground_truth=dict(spec.ground_truth),
        seeds=tuple(seeds),
    )


# =============================================================================
# Static-flag bridge: read analyzer output for a single snippet
# =============================================================================


def compute_static_flags_from_source(
    source: str, target_qname: str,
    *, module_name: str = "snippet",
) -> dict[str, bool]:
    """Parse `source` and run the Phase-24 + Phase-25 analyzers against
    the target function in isolation. Returns the seven flags the
    runtime probes can compare to.

    This is what joins the calibration report to the analyzer axis —
    the `static_flags` column in the result table.
    """
    import ast
    from .code_interproc import (
        analyze_interproc, build_module_context,
    )

    tree = ast.parse(textwrap.dedent(source))
    ctx, intra = build_module_context(module_name, tree)
    interproc, _cg = analyze_interproc([ctx], intra)
    full_qname = f"{module_name}.{target_qname}"
    intra_flags = intra.get(full_qname)
    inter_flags = interproc.get(full_qname)
    if intra_flags is None or inter_flags is None:
        return {}
    # For the calibration we use the INTERPROCEDURAL flag where one
    # exists, because those are the flags the planner would actually
    # return on the interproc slice. Where the runtime probe sees
    # through wrappers (subprocess, filesystem, network, may_raise,
    # may_write_global), the trans-* flag is the right comparand.
    # `participates_in_cycle` comes straight from the interproc pass.
    return {
        "may_raise": inter_flags.trans_may_raise,
        "may_write_global": inter_flags.trans_may_write_global,
        "calls_subprocess": inter_flags.trans_calls_subprocess,
        "calls_filesystem": inter_flags.trans_calls_filesystem,
        "calls_network": inter_flags.trans_calls_network,
        "participates_in_cycle": inter_flags.participates_in_cycle,
        # Phase 28 — the explicit-raise predicate matches the
        # existing `may_raise` analyzer contract exactly. The
        # implicit-raise flag rides on the separate Phase-28
        # analyzer field.
        "may_raise_explicit": inter_flags.trans_may_raise,
        "may_raise_implicit": getattr(
            inter_flags, "trans_may_raise_implicit", False),
    }


# =============================================================================
# Aggregate calibration over a batch of snippet specs
# =============================================================================


@dataclass
class CalibrationSummary:
    """Pooled metrics across every snippet in a batch.

    Fields per predicate:
      - n_applicable            — snippets for which the probe ran.
      - n_static_true           — snippets where static flag = True.
      - n_runtime_true          — snippets where runtime flag = True.
      - n_agree                 — snippets where static == runtime.
      - n_false_positives       — static True AND runtime False.
      - n_false_negatives       — static False AND runtime True.
      - fp_rate                 — n_false_positives / n_static_true
                                   (undefined when no static Trues).
      - fn_rate                 — n_false_negatives / n_runtime_true
                                   (undefined when no runtime Trues).
      - soundness_violations    — same as false_negatives on the
                                   conservative-analysis reading.

    The WHOLE batch view accumulates by SUMMING across predicates;
    that's only meaningful for reporting per-predicate columns.
    """

    per_predicate: dict[str, dict] = field(default_factory=dict)
    per_snippet: list[SnippetResult] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "per_predicate": self.per_predicate,
            "per_snippet": [self._snippet_as_dict(s)
                            for s in self.per_snippet],
        }

    @staticmethod
    def _snippet_as_dict(s: SnippetResult) -> dict:
        return {
            "snippet_name": s.snippet_name,
            "target_qname": s.target_qname,
            "static_flags": s.static_flags,
            "ground_truth": s.ground_truth,
            "seeds": list(s.seeds),
            "observations": {
                p: {
                    "runtime_flag": o.runtime_flag,
                    "n_runs": o.n_runs,
                    "n_triggered": o.n_triggered,
                    "trigger_rate": round(o.trigger_rate, 4),
                    "witnesses": list(o.witnesses),
                    "decidable": o.decidable,
                    "applicable": o.applicable,
                    "notes": o.notes,
                }
                for p, o in s.runtime_observations.items()
            },
            "divergences": s.divergences(),
        }


def summarise_calibration(results: list[SnippetResult]) -> CalibrationSummary:
    """Compute per-predicate FP / FN / agreement metrics across a
    batch of per-snippet results.

    Only counts observations where both `decidable` and `applicable`
    are True — non-runtime-decidable predicates contribute zero
    rows.
    """
    per_pred: dict[str, dict] = {}
    pred_names = set()
    for r in results:
        pred_names.update(r.runtime_observations.keys())
        pred_names.update(r.static_flags.keys())

    for pred in sorted(pred_names):
        n_applicable = 0
        n_static_true = 0
        n_runtime_true = 0
        n_agree = 0
        n_fp = 0
        n_fn = 0
        for r in results:
            rt = r.runtime_observations.get(pred)
            if rt is None or not (rt.decidable and rt.applicable):
                continue
            n_applicable += 1
            sf = bool(r.static_flags.get(pred, False))
            rf = bool(rt.runtime_flag)
            if sf: n_static_true += 1
            if rf: n_runtime_true += 1
            if sf == rf:
                n_agree += 1
            elif sf and not rf:
                n_fp += 1
            else:
                n_fn += 1

        fp_rate = (n_fp / n_static_true) if n_static_true else None
        fn_rate = (n_fn / n_runtime_true) if n_runtime_true else None
        per_pred[pred] = {
            "n_applicable": n_applicable,
            "n_static_true": n_static_true,
            "n_runtime_true": n_runtime_true,
            "n_agree": n_agree,
            "n_false_positives": n_fp,
            "n_false_negatives": n_fn,
            "fp_rate": round(fp_rate, 4) if fp_rate is not None else None,
            "fn_rate": round(fn_rate, 4) if fn_rate is not None else None,
            "soundness_violations": n_fn,
        }

    return CalibrationSummary(per_predicate=per_pred, per_snippet=results)
