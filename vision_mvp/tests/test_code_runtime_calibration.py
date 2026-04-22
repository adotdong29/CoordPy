"""Tests for the runtime-truth calibration layer — Phase 26.

These tests pin three things:

  1. **Probe correctness.** Each per-predicate probe correctly
     reports True / False on a handcrafted minimal snippet whose
     runtime behaviour is unambiguous.
  2. **Sandbox integrity.** The probes neuter dangerous APIs — a
     snippet that tries to spawn a real subprocess, open a file at a
     real absolute path, or connect a real socket does NOT escape
     the probe. Whatever monkeypatching installs MUST be restored
     after the probe exits.
  3. **Determinism.** The same seed on the same snippet yields the
     same observation.
  4. **Calibration semantics.** `summarise_calibration` correctly
     classifies static/runtime divergences as FP / FN and computes
     soundness_violations.

The probes are designed to be composable: all three of
`_record_subprocess` / `_record_filesystem` / `_record_network` can
nest in one `with` block. These tests compose them aggressively.
"""

from __future__ import annotations

import builtins
import os
import socket
import subprocess
import sys
import unittest

from vision_mvp.core.code_runtime_calibration import (
    CalibrationSummary, RuntimeObservation, SnippetResult, SnippetSpec,
    _record_filesystem, _record_network, _record_subprocess,
    _track_reentry, calibrate_snippet,
    compute_static_flags_from_source, load_snippet_module,
    probe_calls_filesystem, probe_calls_network,
    probe_calls_subprocess, probe_may_raise, probe_may_write_global,
    probe_participates_in_cycle, probe_predicate,
    summarise_calibration, synthesize_args,
)


# =============================================================================
# Synthesize args
# =============================================================================


class TestSynthesizeArgs(unittest.TestCase):

    def test_empty_params_returns_empty_tuple(self):
        self.assertEqual(synthesize_args(0, seed=0), ())

    def test_deterministic_given_seed(self):
        a = synthesize_args(3, seed=42)
        b = synthesize_args(3, seed=42)
        self.assertEqual(a, b)

    def test_different_seeds_may_differ(self):
        # Not guaranteed to differ for every pair, but over a batch at
        # least some must differ (else the RNG is broken). Compare
        # via repr so unhashable pool elements (dicts, lists) are
        # still distinguishable.
        vs = [repr(synthesize_args(3, seed=s)) for s in range(20)]
        self.assertGreater(len(set(vs)), 1)


# =============================================================================
# Snippet loading
# =============================================================================


class TestLoadSnippetModule(unittest.TestCase):

    def test_module_has_target(self):
        src = "def foo(): return 42\n"
        mod = load_snippet_module(src)
        self.assertTrue(hasattr(mod, "foo"))
        self.assertEqual(mod.foo(), 42)

    def test_each_load_is_isolated(self):
        src = "X = []\ndef f(): X.append(1)\n"
        a = load_snippet_module(src)
        b = load_snippet_module(src)
        a.f()
        # Module b is a SEPARATE namespace — a.f() must not have
        # mutated b.X.
        self.assertEqual(a.X, [1])
        self.assertEqual(b.X, [])


# =============================================================================
# Sandbox context managers — isolation + restoration
# =============================================================================


class TestSandboxRestoration(unittest.TestCase):

    def test_record_subprocess_restores_original(self):
        orig_popen_init = subprocess.Popen.__init__
        orig_system = os.system
        with _record_subprocess():
            self.assertIsNot(subprocess.Popen.__init__, orig_popen_init)
        self.assertIs(subprocess.Popen.__init__, orig_popen_init)
        self.assertIs(os.system, orig_system)

    def test_record_filesystem_restores_builtin_open(self):
        orig_open = builtins.open
        with _record_filesystem():
            self.assertIsNot(builtins.open, orig_open)
        self.assertIs(builtins.open, orig_open)

    def test_record_network_restores_socket_connect(self):
        orig = socket.socket.connect
        with _record_network():
            self.assertIsNot(socket.socket.connect, orig)
        self.assertIs(socket.socket.connect, orig)

    def test_filesystem_reroutes_writes_into_tempdir(self):
        """A `write` to what looks like an absolute path must land
        inside the probe's temp directory — never at the real path.
        """
        # A sentinel path that MUST NOT be created on the host.
        sentinel = "/tmp/phase26_sentinel_never_created.txt"
        # Belt-and-braces: ensure it didn't already exist.
        try:
            os.remove(sentinel)
        except OSError:
            pass
        with _record_filesystem() as fs_hits:
            f = builtins.open(sentinel, "w")
            f.write("if you see this at the real path, the probe leaked")
            f.close()
        self.assertTrue(fs_hits, "open() attempt should have been recorded")
        self.assertFalse(os.path.exists(sentinel),
                          "probe must NOT create the real sentinel path")

    def test_network_connect_raises_sentinel(self):
        """The sentinel must be caught by the probe layer — the
        context manager is the INSTRUMENT, not the catcher. Verify
        that calling connect DOES raise (so the probe's _call_safely
        observes the attempt), and that the attempt is recorded.
        """
        from vision_mvp.core.code_runtime_calibration import _NetworkAttempted
        s = socket.socket()
        with _record_network() as hits:
            with self.assertRaises(_NetworkAttempted):
                s.connect(("127.0.0.1", 1))
        self.assertTrue(hits)

    def test_subprocess_popen_raises_sentinel(self):
        from vision_mvp.core.code_runtime_calibration import (
            _SubprocessAttempted)
        with _record_subprocess() as hits:
            with self.assertRaises(_SubprocessAttempted):
                subprocess.Popen(["/bin/true"])
        self.assertTrue(hits)


# =============================================================================
# Per-predicate probes on handcrafted minimal snippets
# =============================================================================


class TestProbeMayRaise(unittest.TestCase):

    def test_unconditional_raise_observed(self):
        src = "def f(): raise ValueError('x')\n"
        mod = load_snippet_module(src)
        obs = probe_may_raise(mod.f, invocations=[()])
        self.assertTrue(obs.runtime_flag)
        self.assertEqual(obs.n_runs, 1)
        self.assertEqual(obs.n_triggered, 1)
        self.assertIn("ValueError", obs.witnesses)

    def test_no_raise(self):
        src = "def f(a, b): return a + b\n"
        mod = load_snippet_module(src)
        obs = probe_may_raise(mod.f, invocations=[(1, 2), (3, 4)])
        self.assertFalse(obs.runtime_flag)
        self.assertEqual(obs.n_triggered, 0)

    def test_conditional_raise_triggers_on_bad_input(self):
        src = "def f(x):\n    if x is None: raise TypeError('n')\n    return x\n"
        mod = load_snippet_module(src)
        obs = probe_may_raise(
            mod.f, invocations=[(1,), (None,), (2,)])
        self.assertTrue(obs.runtime_flag)
        self.assertEqual(obs.n_triggered, 1)

    def test_sentinel_raises_do_not_count_as_may_raise(self):
        """A subprocess attempt inside target must NOT be reported
        as may_raise=True — the probe sentinel is explicitly
        excluded from user exceptions."""
        src = (
            "import subprocess\n"
            "def f(): subprocess.run(['/bin/true'])\n"
        )
        mod = load_snippet_module(src)
        obs = probe_may_raise(mod.f, invocations=[()])
        self.assertFalse(obs.runtime_flag,
                          "sentinel raise should not count as may_raise")


class TestProbeMayWriteGlobal(unittest.TestCase):

    def test_global_assignment_detected(self):
        src = (
            "X = 0\n"
            "def f():\n"
            "    global X\n"
            "    X = X + 1\n"
        )
        mod = load_snippet_module(src)
        obs = probe_may_write_global(mod.f, mod, invocations=[()])
        self.assertTrue(obs.runtime_flag)

    def test_container_mutation_detected(self):
        src = (
            "S = []\n"
            "def f(): S.append(1)\n"
        )
        mod = load_snippet_module(src)
        obs = probe_may_write_global(mod.f, mod, invocations=[()])
        self.assertTrue(obs.runtime_flag)

    def test_pure_function_not_flagged(self):
        src = "def f(a, b): return a + b\n"
        mod = load_snippet_module(src)
        obs = probe_may_write_global(mod.f, mod,
                                      invocations=[(1, 2), (3, 4)])
        self.assertFalse(obs.runtime_flag)


class TestProbeCallsSubprocess(unittest.TestCase):

    def test_direct_subprocess_detected(self):
        src = (
            "import subprocess\n"
            "def f(): subprocess.run(['/bin/true'])\n"
        )
        mod = load_snippet_module(src)
        obs = probe_calls_subprocess(mod.f, invocations=[()])
        self.assertTrue(obs.runtime_flag)

    def test_pure_function_not_flagged(self):
        src = "def f(): return 1\n"
        mod = load_snippet_module(src)
        obs = probe_calls_subprocess(mod.f, invocations=[()])
        self.assertFalse(obs.runtime_flag)

    def test_os_system_detected(self):
        src = (
            "import os\n"
            "def f(): os.system('echo x')\n"
        )
        mod = load_snippet_module(src)
        obs = probe_calls_subprocess(mod.f, invocations=[()])
        self.assertTrue(obs.runtime_flag)


class TestProbeCallsFilesystem(unittest.TestCase):

    def test_open_write_detected(self):
        src = "def f(): open('x.txt', 'w').write('x')\n"
        mod = load_snippet_module(src)
        obs = probe_calls_filesystem(mod.f, invocations=[()])
        self.assertTrue(obs.runtime_flag)

    def test_pure_function_not_flagged(self):
        src = "def f(a): return a + 1\n"
        mod = load_snippet_module(src)
        obs = probe_calls_filesystem(mod.f, invocations=[(1,), (2,)])
        self.assertFalse(obs.runtime_flag)


class TestProbeCallsNetwork(unittest.TestCase):

    def test_socket_connect_detected(self):
        src = (
            "import socket\n"
            "def f():\n"
            "    s = socket.socket()\n"
            "    s.connect(('127.0.0.1', 1))\n"
        )
        mod = load_snippet_module(src)
        obs = probe_calls_network(mod.f, invocations=[()])
        self.assertTrue(obs.runtime_flag)

    def test_pure_function_not_flagged(self):
        src = "def f(): return 1\n"
        mod = load_snippet_module(src)
        obs = probe_calls_network(mod.f, invocations=[()])
        self.assertFalse(obs.runtime_flag)


class TestProbeParticipatesInCycle(unittest.TestCase):

    def test_self_recursion_detected(self):
        src = (
            "def fib(n):\n"
            "    if n < 2: return n\n"
            "    return fib(n - 1) + fib(n - 2)\n"
        )
        mod = load_snippet_module(src)
        obs = probe_participates_in_cycle(mod.fib, invocations=[(3,)])
        self.assertTrue(obs.runtime_flag)

    def test_non_recursive_not_flagged(self):
        src = "def f(x): return x + 1\n"
        mod = load_snippet_module(src)
        obs = probe_participates_in_cycle(mod.f, invocations=[(1,)])
        self.assertFalse(obs.runtime_flag)

    def test_mutual_recursion_detected_via_trace(self):
        src = (
            "def ping(n):\n"
            "    if n <= 0: return n\n"
            "    return pong(n - 1)\n"
            "def pong(n):\n"
            "    if n <= 0: return n\n"
            "    return ping(n - 1)\n"
        )
        mod = load_snippet_module(src)
        obs = probe_participates_in_cycle(mod.ping, invocations=[(3,)])
        self.assertTrue(obs.runtime_flag)

    def test_track_reentry_restores_prior_tracer(self):
        """The tracer installed by _track_reentry must be removed on
        exit and must not leak to the outer interpreter."""
        def trivial(): pass
        prev = sys.gettrace()
        with _track_reentry(trivial) as _:
            pass
        self.assertIs(sys.gettrace(), prev)


# =============================================================================
# Dispatcher
# =============================================================================


class TestProbeDispatcher(unittest.TestCase):

    def test_unknown_predicate_marked_undecidable(self):
        src = "def f(): pass\n"
        mod = load_snippet_module(src)
        obs = probe_predicate(
            "has_unresolved_callees", mod.f, module=mod,
            invocations=[()])
        self.assertFalse(obs.decidable)
        self.assertEqual(obs.n_runs, 0)

    def test_every_known_predicate_roundtrips(self):
        src = "def f(): return 1\n"
        mod = load_snippet_module(src)
        for pred in ("may_raise", "may_write_global",
                     "calls_subprocess", "calls_filesystem",
                     "calls_network", "participates_in_cycle"):
            obs = probe_predicate(pred, mod.f, module=mod,
                                   invocations=[()])
            self.assertTrue(obs.decidable, msg=pred)
            self.assertTrue(obs.applicable, msg=pred)


# =============================================================================
# End-to-end: calibrate_snippet
# =============================================================================


class TestCalibrateSnippet(unittest.TestCase):

    def test_direct_raise_runtime_agrees_with_static(self):
        spec = SnippetSpec(
            name="t", source="def t(): raise ValueError('x')\n",
            target_qname="t", ground_truth={"may_raise": True},
            n_params_override=0, n_fuzz=2,
        )
        static = compute_static_flags_from_source(
            spec.source, spec.target_qname)
        res = calibrate_snippet(
            spec, predicates=["may_raise"], seeds=(0, 1),
            static_flags=static)
        self.assertTrue(res.static_flags["may_raise"])
        self.assertTrue(res.runtime_observations["may_raise"].runtime_flag)
        self.assertEqual(res.divergences(), [])

    def test_dead_raise_is_false_positive(self):
        """Dead-code raise: analyzer flags True (no CF analysis),
        runtime never triggers → false-positive surfaced by
        divergences()."""
        spec = SnippetSpec(
            name="d",
            source=(
                "def d():\n"
                "    if False: raise RuntimeError('x')\n"
                "    return 0\n"
            ),
            target_qname="d",
            ground_truth={"may_raise": False},
            n_params_override=0, n_fuzz=3,
        )
        static = compute_static_flags_from_source(
            spec.source, spec.target_qname)
        self.assertTrue(static["may_raise"],
                         "analyzer should over-approximate dead-code raise")
        res = calibrate_snippet(
            spec, predicates=["may_raise"], seeds=(0, 1),
            static_flags=static)
        self.assertFalse(res.runtime_observations["may_raise"].runtime_flag)
        self.assertIn(("may_raise", "false_positive"), res.divergences())

    def test_hidden_subprocess_via_eval_is_false_negative(self):
        """eval hides subprocess from the analyzer — runtime observes."""
        spec = SnippetSpec(
            name="h",
            source=(
                "def h():\n"
                "    eval(\"__import__('subprocess').run(['/bin/true'])\")\n"
            ),
            target_qname="h",
            ground_truth={"calls_subprocess": True},
            n_params_override=0, n_fuzz=2,
        )
        static = compute_static_flags_from_source(
            spec.source, spec.target_qname)
        self.assertFalse(static.get("calls_subprocess", False),
                          "analyzer should MISS subprocess through eval")
        res = calibrate_snippet(
            spec, predicates=["calls_subprocess"], seeds=(0, 1),
            static_flags=static)
        obs = res.runtime_observations["calls_subprocess"]
        self.assertTrue(obs.runtime_flag,
                         "runtime probe should observe subprocess attempt")
        self.assertIn(("calls_subprocess", "false_negative"),
                       res.divergences())


# =============================================================================
# Determinism
# =============================================================================


class TestDeterminism(unittest.TestCase):

    def test_same_seed_same_observation(self):
        spec = SnippetSpec(
            name="f",
            source="def f(x):\n    if x is None: raise TypeError('n')\n    return x\n",
            target_qname="f",
            ground_truth={"may_raise": True},
            n_params_override=1, n_fuzz=8,
        )
        a = calibrate_snippet(spec, predicates=["may_raise"], seeds=(7,))
        b = calibrate_snippet(spec, predicates=["may_raise"], seeds=(7,))
        ob_a = a.runtime_observations["may_raise"]
        ob_b = b.runtime_observations["may_raise"]
        self.assertEqual(ob_a.n_runs, ob_b.n_runs)
        self.assertEqual(ob_a.n_triggered, ob_b.n_triggered)
        self.assertEqual(ob_a.runtime_flag, ob_b.runtime_flag)


# =============================================================================
# Summarise
# =============================================================================


class TestSummariseCalibration(unittest.TestCase):

    def _fake_result(self, name: str, static: bool, runtime: bool,
                      pred: str = "may_raise") -> SnippetResult:
        return SnippetResult(
            snippet_name=name, target_qname="t",
            static_flags={pred: static},
            runtime_observations={pred: RuntimeObservation(
                predicate=pred, runtime_flag=runtime,
                n_runs=1, n_triggered=int(runtime),
            )},
            ground_truth={pred: runtime},
            seeds=(0,),
        )

    def test_agree_counts_correctly(self):
        rs = [self._fake_result("a", True, True),
              self._fake_result("b", False, False)]
        s = summarise_calibration(rs)
        pp = s.per_predicate["may_raise"]
        self.assertEqual(pp["n_applicable"], 2)
        self.assertEqual(pp["n_agree"], 2)
        self.assertEqual(pp["n_false_positives"], 0)
        self.assertEqual(pp["n_false_negatives"], 0)

    def test_false_positive_surfaces(self):
        rs = [self._fake_result("fp", True, False),
              self._fake_result("ok", True, True)]
        s = summarise_calibration(rs)
        pp = s.per_predicate["may_raise"]
        self.assertEqual(pp["n_false_positives"], 1)
        # FP rate = fp / static_true
        self.assertAlmostEqual(pp["fp_rate"], 0.5, places=4)

    def test_false_negative_is_soundness_violation(self):
        rs = [self._fake_result("fn", False, True)]
        s = summarise_calibration(rs)
        pp = s.per_predicate["may_raise"]
        self.assertEqual(pp["n_false_negatives"], 1)
        self.assertEqual(pp["soundness_violations"], 1)


if __name__ == "__main__":
    unittest.main()
