"""Executable snippet corpus for runtime-truth calibration — Phase 26.

Each snippet is a self-contained Python module with exactly one target
function. The author declares, per predicate, what the runtime
behaviour of that function should be when exercised with the default
fuzz inputs. The Phase-26 harness then:

  1. Runs the Phase-24 / Phase-25 analyzer against the snippet and
     records the seven static flags.
  2. Runs the runtime probes from `core.code_runtime_calibration`
     against the target in a sandboxed context (neutered
     subprocess / filesystem / network, sys.settrace for cycles).
  3. Compares analyzer vs runtime per predicate, per snippet, across
     seeds, producing the calibration tables.

The snippets are grouped into families that exercise specific
analyzer vs runtime regimes:

  * `direct`      — the effect happens directly in the target body.
                    Analyzer and runtime should both observe it.
  * `wrapper`     — the target calls a helper that does the effect.
                    Intraprocedural analyzer should MISS the effect;
                    interprocedural (Phase 25) should catch it;
                    runtime should observe it.
  * `chain`       — longer call chains (a → b → c → effect).
  * `cycle`       — mutual recursion; SCC membership = runtime re-entry.
  * `guarded`     — analyzer flags a `raise` that runtime never triggers
                    (false-positive case for conservative analysis).
  * `dead`        — `if False: raise ...` — static True, runtime False.
  * `hidden`      — runtime effect via a surface the analyzer doesn't
                    know about (e.g. `eval`, `getattr`). Might be an
                    analyzer soundness hole.
  * `negative`    — target does nothing effectful; both axes should
                    agree on False.

Each snippet's ground_truth dict carries the author-declared answer
per predicate; see `core.code_runtime_calibration.SnippetSpec` for
the schema. Ground_truth is used only to VALIDATE the probes
(`test_executable_snippets.py` checks probe output matches
ground_truth), not to validate the analyzer — that's what the
calibration delta measures.

IMPORTANT: these snippets run real Python. They rely on the probe
harness to prevent effects from escaping. Keep additions small and
obviously safe; the test suite will execute every snippet at import
time.
"""

from __future__ import annotations

from ..core.code_runtime_calibration import SnippetSpec


# =============================================================================
# Helper: build a snippet with defaults
# =============================================================================


def _spec(name: str, source: str, target_qname: str,
          ground_truth: dict[str, bool], *,
          family: str = "direct",
          n_params_override: int | None = None,
          n_fuzz: int = 6,
          notes: str = "") -> SnippetSpec:
    full_gt = {
        "may_raise": False, "may_write_global": False,
        "calls_subprocess": False, "calls_filesystem": False,
        "calls_network": False, "participates_in_cycle": False,
    }
    full_gt.update(ground_truth)
    return SnippetSpec(
        name=name, source=source, target_qname=target_qname,
        ground_truth=full_gt, family=family,
        n_params_override=n_params_override, n_fuzz=n_fuzz,
        notes=notes,
    )


# =============================================================================
# Family: direct (effect in target body)
# =============================================================================


S_NOP = SnippetSpec(
    name="nop_arith",
    source=(
        "def nop_arith(a, b):\n"
        "    x = a + b\n"
        "    y = x * 2\n"
        "    return y\n"
    ),
    target_qname="nop_arith",
    family="negative",
    n_params_override=2,
    # Drive with ints so arithmetic succeeds — the point is to exercise
    # the "negative control" behaviour, not to fuzz arithmetic-on-junk.
    invoke=lambda target, rng: [
        (rng.randrange(-10, 10), rng.randrange(-10, 10)) for _ in range(6)
    ],
    ground_truth={
        "may_raise": False, "calls_subprocess": False,
        "calls_filesystem": False, "calls_network": False,
        "may_write_global": False, "participates_in_cycle": False,
    },
    notes="Pure arithmetic. Every predicate should be False on both axes.",
)


S_DIRECT_RAISE = _spec(
    "direct_raise",
    """
    def direct_raise():
        raise ValueError("boom")
    """,
    target_qname="direct_raise",
    family="direct",
    n_params_override=0,
    ground_truth={"may_raise": True},
    notes="Unconditional raise. Both axes True.",
)


S_DIRECT_SUBPROCESS = _spec(
    "direct_subprocess",
    """
    import subprocess
    def direct_subprocess():
        subprocess.run(["/bin/echo", "hi"], check=False)
    """,
    target_qname="direct_subprocess",
    family="direct",
    n_params_override=0,
    ground_truth={"calls_subprocess": True},
    notes="Direct subprocess.run. Both axes True.",
)


S_DIRECT_FILESYSTEM = _spec(
    "direct_filesystem",
    """
    def direct_filesystem():
        f = open("phase26_probe.tmp", "w")
        f.write("x")
        f.close()
    """,
    target_qname="direct_filesystem",
    family="direct",
    n_params_override=0,
    ground_truth={"calls_filesystem": True},
    notes="Direct open(...,'w'). Both axes True.",
)


S_DIRECT_NETWORK = _spec(
    "direct_network",
    """
    import socket
    def direct_network():
        s = socket.socket()
        s.connect(("127.0.0.1", 1))
    """,
    target_qname="direct_network",
    family="direct",
    n_params_override=0,
    ground_truth={"calls_network": True},
    notes="Direct socket.connect. Both axes True.",
)


S_DIRECT_GLOBAL_WRITE = _spec(
    "direct_global_write",
    """
    COUNTER = 0
    def direct_global_write():
        global COUNTER
        COUNTER += 1
    """,
    target_qname="direct_global_write",
    family="direct",
    n_params_override=0,
    ground_truth={"may_write_global": True},
    notes="Canonical `global X; X = ...` pattern. Both axes True.",
)


# =============================================================================
# Family: wrapper — intraprocedural misses, interprocedural catches
# =============================================================================


S_WRAPPER_RAISE = _spec(
    "wrapper_raise",
    """
    def _helper():
        raise RuntimeError("inner")
    def wrapper_raise():
        _helper()
    """,
    target_qname="wrapper_raise",
    family="wrapper",
    n_params_override=0,
    ground_truth={"may_raise": True},
    notes="Wrapper calls helper that raises. Intra miss, trans catches.",
)


S_WRAPPER_SUBPROCESS = _spec(
    "wrapper_subprocess",
    """
    import subprocess
    def _spawn():
        subprocess.run(["/bin/true"], check=False)
    def wrapper_subprocess():
        _spawn()
    """,
    target_qname="wrapper_subprocess",
    family="wrapper",
    n_params_override=0,
    ground_truth={"calls_subprocess": True},
    notes="Wrapper → subprocess helper. Trans catches; intra misses.",
)


S_WRAPPER_FILESYSTEM = _spec(
    "wrapper_filesystem",
    """
    def _write_file():
        f = open("wrapper_out.tmp", "w")
        f.write("y")
        f.close()
    def wrapper_filesystem():
        _write_file()
    """,
    target_qname="wrapper_filesystem",
    family="wrapper",
    n_params_override=0,
    ground_truth={"calls_filesystem": True},
)


S_WRAPPER_NETWORK = _spec(
    "wrapper_network",
    """
    import socket
    def _connect():
        s = socket.socket()
        s.connect(("127.0.0.1", 1))
    def wrapper_network():
        _connect()
    """,
    target_qname="wrapper_network",
    family="wrapper",
    n_params_override=0,
    ground_truth={"calls_network": True},
)


S_WRAPPER_GLOBAL_WRITE = _spec(
    "wrapper_global_write",
    """
    STATE = []
    def _mutate():
        STATE.append(1)
    def wrapper_global_write():
        _mutate()
    """,
    target_qname="wrapper_global_write",
    family="wrapper",
    n_params_override=0,
    ground_truth={"may_write_global": True},
    notes="Helper mutates module-level list. Intra miss on wrapper.",
)


# =============================================================================
# Family: chain — longer indirection
# =============================================================================


S_CHAIN_RAISE = _spec(
    "chain_raise",
    """
    def _c():
        raise KeyError("c")
    def _b():
        _c()
    def _a():
        _b()
    def chain_raise():
        _a()
    """,
    target_qname="chain_raise",
    family="chain",
    n_params_override=0,
    ground_truth={"may_raise": True},
    notes="3-hop chain. Trans catches; intra misses.",
)


S_CHAIN_SUBPROCESS = _spec(
    "chain_subprocess",
    """
    import subprocess
    def _c():
        subprocess.run(["/bin/true"])
    def _b():
        _c()
    def chain_subprocess():
        _b()
    """,
    target_qname="chain_subprocess",
    family="chain",
    n_params_override=0,
    ground_truth={"calls_subprocess": True},
)


# =============================================================================
# Family: cycle — mutual recursion
# =============================================================================


S_MUTUAL_RECURSION = _spec(
    "mutual_recursion",
    """
    def ping(n):
        if n <= 0:
            return n
        return pong(n - 1)
    def pong(n):
        if n <= 0:
            return n
        return ping(n - 1)
    def cycle_entry():
        return ping(3)
    """,
    target_qname="cycle_entry",
    family="cycle",
    n_params_override=0,
    ground_truth={"participates_in_cycle": False},
    notes="`cycle_entry` itself is NOT in the cycle; ping and pong are. "
          "We probe cycle_entry to confirm the analyzer agrees "
          "(participates_in_cycle=False on cycle_entry). A separate "
          "target probes ping directly.",
)


S_SELF_RECURSION = SnippetSpec(
    name="self_recursion",
    source=(
        "def fib(n):\n"
        "    if n < 2:\n"
        "        return n\n"
        "    return fib(n - 1) + fib(n - 2)\n"
    ),
    target_qname="fib",
    family="cycle",
    n_params_override=1,
    # Drive with integer inputs only so fib doesn't raise on fuzz junk.
    # Ground_truth intentionally says may_raise=False: with these
    # inputs the function never raises (even though with fuzz strings
    # it would TypeError on `n < 2`).
    invoke=lambda target, rng: [(rng.randrange(2, 5),) for _ in range(5)],
    ground_truth={
        "may_raise": False, "may_write_global": False,
        "calls_subprocess": False, "calls_filesystem": False,
        "calls_network": False, "participates_in_cycle": True,
    },
    notes="Self-recursion → SCC of size 1 with self-loop → cycle. "
          "Runtime observes re-entry via settrace. Driven with ints in "
          "[2,5) so the function is exercised exactly where it recurses.",
)


S_PING_IN_CYCLE = SnippetSpec(
    name="ping_in_cycle",
    source=(
        "def ping(n):\n"
        "    if n <= 0:\n"
        "        return n\n"
        "    return pong(n - 1)\n"
        "def pong(n):\n"
        "    if n <= 0:\n"
        "        return n\n"
        "    return ping(n - 1)\n"
    ),
    target_qname="ping",
    family="cycle",
    n_params_override=1,
    invoke=lambda target, rng: [(rng.randrange(2, 5),) for _ in range(5)],
    ground_truth={
        "may_raise": False, "may_write_global": False,
        "calls_subprocess": False, "calls_filesystem": False,
        "calls_network": False, "participates_in_cycle": True,
    },
    notes="Direct probe of ping. Trans+SCC identify both as in-cycle; "
          "runtime settrace observes re-entry when n >= 2.",
)


# =============================================================================
# Family: guarded / dead — precision cases
# =============================================================================


S_CAUGHT_RAISE = _spec(
    "caught_raise",
    """
    def caught_raise():
        try:
            raise ValueError("x")
        except Exception:
            return 0
    """,
    target_qname="caught_raise",
    family="guarded",
    n_params_override=0,
    ground_truth={"may_raise": False},
    notes="Analyzer's `_raise_is_caught` rule — catch-all swallows. "
          "Both axes agree on False.",
)


S_DEAD_RAISE = _spec(
    "dead_raise",
    """
    def dead_raise():
        if False:
            raise RuntimeError("unreachable")
        return 0
    """,
    target_qname="dead_raise",
    family="dead",
    n_params_override=0,
    ground_truth={"may_raise": False},
    notes="Analyzer IS control-flow-insensitive: flags may_raise=True. "
          "Runtime never triggers. This is the canonical FALSE-POSITIVE "
          "case that calibration should measure.",
)


S_CONDITIONAL_RAISE = SnippetSpec(
    name="conditional_raise",
    source=(
        "def conditional_raise(x):\n"
        "    if x is None:\n"
        "        raise TypeError('x is None')\n"
        "    return x\n"
    ),
    target_qname="conditional_raise",
    family="direct",
    n_params_override=1,
    # Guaranteed to hit the raising branch; fuzz-driven sweep is
    # stochastic and not worth relying on.
    invoke=lambda target, rng: [(None,), (1,), ("x",)],
    ground_truth={
        "may_raise": True, "may_write_global": False,
        "calls_subprocess": False, "calls_filesystem": False,
        "calls_network": False, "participates_in_cycle": False,
    },
    notes="Conditional raise, driven with a curated invoke list that "
          "includes None so the raising branch is always exercised.",
)


# =============================================================================
# Family: hidden — effect via surfaces the analyzer doesn't know
# =============================================================================


S_HIDDEN_SUBPROCESS_VIA_EVAL = _spec(
    "hidden_subprocess_via_eval",
    """
    def hidden_subprocess_via_eval():
        # eval is opaque to the conservative analyzer.
        expr = "__import__('subprocess').run(['/bin/true'])"
        eval(expr)
    """,
    target_qname="hidden_subprocess_via_eval",
    family="hidden",
    n_params_override=0,
    ground_truth={"calls_subprocess": True},
    notes="Runtime OBSERVES subprocess attempt (via eval). Analyzer's "
          "intraprocedural and interprocedural slices do NOT flag "
          "calls_subprocess — `eval` is explicitly documented as out "
          "of scope. This is the canonical analyzer FALSE-NEGATIVE "
          "case the calibration harness should surface.",
)


S_HIDDEN_FILESYSTEM_VIA_GETATTR = _spec(
    "hidden_filesystem_via_getattr",
    """
    import builtins
    def hidden_filesystem_via_getattr():
        # getattr + builtins — opaque to the static analyzer.
        fn = getattr(builtins, "open")
        f = fn("hidden_out.tmp", "w")
        f.write("z")
        f.close()
    """,
    target_qname="hidden_filesystem_via_getattr",
    family="hidden",
    n_params_override=0,
    ground_truth={"calls_filesystem": True},
    notes="Reflection over builtins. Runtime observes; analyzer "
          "probably misses (false negative).",
)


# =============================================================================
# Full registry
# =============================================================================


_ALL_SPECS: tuple[SnippetSpec, ...] = (
    S_NOP,
    S_DIRECT_RAISE, S_DIRECT_SUBPROCESS, S_DIRECT_FILESYSTEM,
    S_DIRECT_NETWORK, S_DIRECT_GLOBAL_WRITE,
    S_WRAPPER_RAISE, S_WRAPPER_SUBPROCESS, S_WRAPPER_FILESYSTEM,
    S_WRAPPER_NETWORK, S_WRAPPER_GLOBAL_WRITE,
    S_CHAIN_RAISE, S_CHAIN_SUBPROCESS,
    S_MUTUAL_RECURSION, S_SELF_RECURSION, S_PING_IN_CYCLE,
    S_CAUGHT_RAISE, S_DEAD_RAISE, S_CONDITIONAL_RAISE,
    S_HIDDEN_SUBPROCESS_VIA_EVAL, S_HIDDEN_FILESYSTEM_VIA_GETATTR,
)


def default_snippet_registry() -> list[SnippetSpec]:
    """Return the canonical Phase-26 snippet list. Deterministic order."""
    return list(_ALL_SPECS)


def snippets_by_family(family: str) -> list[SnippetSpec]:
    return [s for s in _ALL_SPECS if s.family == family]


def snippet_by_name(name: str) -> SnippetSpec:
    for s in _ALL_SPECS:
        if s.name == name:
            return s
    raise KeyError(f"no snippet named {name!r}")
