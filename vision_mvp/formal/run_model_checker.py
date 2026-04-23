"""Run the TLC model checker against ``CapsuleContract.tla``.

The TLA+ specification of the Capsule Contract lives next to this file.
``TLCModelChecker`` is a thin, robust wrapper around the ``tlc`` binary
that (a) detects whether TLA+ is installed, (b) writes a deterministic
``.cfg`` for the run, (c) invokes TLC with a user-specified state-space
bound, and (d) parses the human-readable TLC output into a structured
dict so callers — tests, CI pipelines, release checks — can assert on
the outcome programmatically.

This module does *not* attempt to install TLA+ itself; the jar is a ~30MB
binary distribution and pulling it down silently on import would be a
CI-hostile side effect.  Instead, ``setup_tla_plus`` prints a single
clear instruction when the binary is absent, and ``run_model_check``
reports ``installed=False`` so dependent callers can skip-with-reason.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


TLA_DIR = Path(__file__).resolve().parent
TLA_MODULE = TLA_DIR / "CapsuleContract.tla"
TLA_CFG = TLA_DIR / "CapsuleContract.cfg"


INVARIANTS = (
    "Inv_C1_Identity",
    "Inv_C2_TypedClaim",
    "Inv_C3_Lifecycle",
    "Inv_C4_Budget",
    "Inv_C5_Provenance",
    "Inv_C6_Frozen",
)


@dataclass
class ModelCheckResult:
    success: bool
    installed: bool
    states_explored: int
    distinct_states: int
    invariants_verified: tuple[str, ...]
    violations: tuple[str, ...]
    stdout: str = ""
    stderr: str = ""

    def as_dict(self) -> dict:
        return {
            "success": self.success,
            "installed": self.installed,
            "states_explored": self.states_explored,
            "distinct_states": self.distinct_states,
            "invariants_verified": list(self.invariants_verified),
            "violations": list(self.violations),
        }


class TLCModelChecker:
    """Driver for ``tlc CapsuleContract.tla`` with result parsing."""

    def __init__(self, tla_module: Path = TLA_MODULE,
                 cfg_path: Path = TLA_CFG) -> None:
        self.tla_module = tla_module
        self.cfg_path = cfg_path

    # ---- setup --------------------------------------------------------------

    def setup_tla_plus(self) -> bool:
        """Return True iff a ``tlc`` binary is on ``$PATH``.

        Does not download the jar — TLA+ is distributed as a Java jar
        (tla2tools.jar); the typical install is ``brew install tla-plus``
        on macOS or the GitHub release tarball elsewhere.  If ``tlc`` is
        absent, print a one-liner install hint and return False.
        """

        if shutil.which("tlc") is not None:
            return True
        print("[tlc] not found on $PATH.")
        print("      install: brew install tla-plus  (macOS)")
        print("               or fetch tla2tools.jar from")
        print("               https://github.com/tlaplus/tlaplus/releases")
        return False

    # ---- config -------------------------------------------------------------

    def write_cfg(self, max_ledger_size: int = 4,
                  n_kinds: int = 3) -> None:
        """Write a TLC ``.cfg`` file bounding the state space.

        The TLA+ spec keeps SHA256 and canonical-JSON opaque, so the
        finite model we hand to TLC just has to bound ``MaxLedgerSize``
        and give a small finite ``Kinds`` set.  4 x 3 = 12 symbolic
        slots is plenty to blow through 10^6 states when the transitions
        are fully interleaved.
        """

        kinds = ",".join(f'"K{i}"' for i in range(n_kinds))
        cfg = (
            "SPECIFICATION Spec\n"
            f"CONSTANTS\n"
            f"    Kinds = {{{kinds}}}\n"
            f"    MaxLedgerSize = {max_ledger_size}\n"
            f'    GenesisHash = "GENESIS"\n'
            "INVARIANTS\n"
            + "".join(f"    {name}\n" for name in INVARIANTS)
        )
        self.cfg_path.write_text(cfg)

    # ---- run ----------------------------------------------------------------

    def run_model_check(self, max_depth: int = 1000,
                        workers: int = 8,
                        write_cfg: bool = True
                        ) -> ModelCheckResult:
        """Execute TLC and return a structured result.

        Parameters mirror TLC's CLI: ``-depth`` caps DFS depth,
        ``-workers`` fans out BFS-level exploration.  If TLC is absent,
        returns ``installed=False`` and ``success=False`` without raising.
        """

        if not self.setup_tla_plus():
            return ModelCheckResult(
                success=False, installed=False,
                states_explored=0, distinct_states=0,
                invariants_verified=(), violations=(),
                stderr="tlc not installed",
            )

        if write_cfg:
            self.write_cfg()

        cmd = [
            "tlc",
            str(self.tla_module),
            "-config", str(self.cfg_path),
            "-depth", str(max_depth),
            "-workers", str(workers),
            "-deadlock",
        ]

        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True,
                cwd=str(TLA_DIR),
                timeout=600,
                check=False,
            )
        except FileNotFoundError:
            return ModelCheckResult(
                success=False, installed=False,
                states_explored=0, distinct_states=0,
                invariants_verified=(), violations=(),
                stderr="tlc exec failed",
            )

        return self._parse(proc.stdout, proc.stderr, proc.returncode)

    # ---- parsing ------------------------------------------------------------

    def _parse(self, stdout: str, stderr: str,
               returncode: int) -> ModelCheckResult:
        """Parse TLC's stdout — line-based, best-effort, robust to
        minor format drift between TLC releases."""

        text = stdout + "\n" + stderr

        m_states = re.search(r"([\d,]+) states? generated", text)
        m_distinct = re.search(r"([\d,]+) distinct states? found", text)
        states = int(m_states.group(1).replace(",", "")) if m_states else 0
        distinct = (int(m_distinct.group(1).replace(",", ""))
                    if m_distinct else 0)

        violations: list[str] = []
        for inv in INVARIANTS:
            pat = re.compile(
                rf"Invariant\s+{re.escape(inv)}\s+is\s+violated")
            if pat.search(text):
                violations.append(inv)

        verified = tuple(i for i in INVARIANTS if i not in violations)
        success = (returncode == 0 and not violations
                   and "Model checking completed" in text)

        return ModelCheckResult(
            success=success,
            installed=True,
            states_explored=states,
            distinct_states=distinct,
            invariants_verified=verified,
            violations=tuple(violations),
            stdout=stdout,
            stderr=stderr,
        )


def main() -> None:
    checker = TLCModelChecker()
    result = checker.run_model_check(max_depth=1000, workers=4)
    print(result.as_dict())


if __name__ == "__main__":
    main()
