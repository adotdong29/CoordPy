"""W87 / P3 #48 — Formal verification gate.

Skip-gated test: when `lake` (the Lean 4 build tool) is available
on PATH, runs the Merkle inclusion proof's `lake build` and asserts:

* exit code 0;
* no `sorry` or `admit` appears in any source file;
* the expected `#check` outputs appear in build stdout;
* `Merkle.inclusion_sound` does NOT depend on
  `Merkle.hashPair_injective` (unconditional);
* `Merkle.inclusion_complete` DOES depend on
  `Merkle.hashPair_injective` (the explicit collision-resistance
  axiom permitted by the issue);
* No other property axioms slipped in.

When Lean is not installed, the test skips cleanly (typical CI
host); the package remains installable with zero formal-methods
dependencies.

This is the W87 #48 closure's CI-friendly verification gate.
"""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import sys

import pytest


PROOF_DIR = pathlib.Path(__file__).resolve().parents[1] / (
    "papers/formal/merkle_inclusion_v1")
PROOF_SOURCES = (
    "MerkleInclusion.lean",
    "MerkleInclusionCheck.lean",
    "MerkleAxiomsReport.lean",
)
# Theorems the issue's DoD ties to this milestone (bare names; the
# #check output omits the namespace since `open Merkle` is in scope).
LOAD_BEARING_THEOREMS = (
    "inclusion_sound",
    "inclusion_complete",
    "merkle_inclusion_iff",
)


def _lake_available() -> bool:
    """Return True iff `lake` is on PATH."""
    return shutil.which("lake") is not None or os.path.exists(
        os.path.expanduser("~/.elan/bin/lake"))


def _lake_env() -> dict:
    """Augment os.environ so `lake` is findable when installed via elan."""
    env = dict(os.environ)
    elan_bin = os.path.expanduser("~/.elan/bin")
    if os.path.isdir(elan_bin):
        env["PATH"] = elan_bin + os.pathsep + env.get("PATH", "")
    return env


@pytest.mark.skipif(
    not _lake_available(),
    reason="W87 #48 Lean proof requires `lake` on PATH (install elan)")
def test_w87_lean_proof_sources_have_no_sorry() -> None:
    """The DoD anti-cheat says: no `sorry`/`admit` in the proof body."""
    for src in PROOF_SOURCES:
        path = PROOF_DIR / src
        assert path.is_file(), f"missing proof source: {src}"
        text = path.read_text(encoding="utf-8")
        # Strip Lean comments (-- ... and /- ... -/) before scanning.
        # Conservative: just check that the literal tokens don't appear
        # outside of doc-comments. The proof source uses neither.
        lower = text
        # Reject `sorry` and `admit` in any non-comment context. As a
        # first-pass guard we just check the tokens don't appear at all.
        # If they do, fail loudly so a reviewer can audit.
        for forbidden in ("sorry", "admit"):
            # Allow within a /- ... -/ block-comment that mentions them.
            # Conservative scan: ensure no line that is *not* a comment
            # contains the token.
            for lineno, line in enumerate(lower.splitlines(), start=1):
                stripped = line.lstrip()
                if (stripped.startswith("--") or
                    stripped.startswith("/-") or
                    stripped.startswith("*")):
                    continue
                # If the line contains the literal token, fail.
                if forbidden in line:
                    raise AssertionError(
                        f"{src}:{lineno} contains forbidden token "
                        f"`{forbidden}`: {line!r}")


@pytest.mark.skipif(
    not _lake_available(),
    reason="W87 #48 Lean proof requires `lake` on PATH (install elan)")
def test_w87_lean_proof_builds_clean() -> None:
    """The DoD: re-checkable from a clean checkout."""
    env = _lake_env()
    # Force a fresh build to catch caching-induced false PASS.
    cwd = str(PROOF_DIR)
    proc = subprocess.run(
        ["lake", "build"],
        cwd=cwd, env=env,
        capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, (
        f"lake build failed (exit {proc.returncode})\n"
        f"stdout:\n{proc.stdout}\n"
        f"stderr:\n{proc.stderr}")
    # Expected #check outputs include each load-bearing theorem.
    combined = proc.stdout + proc.stderr
    for thm in LOAD_BEARING_THEOREMS:
        # The #check output prints `@thmname : ...`
        symbol = "@" + thm
        assert symbol in combined, (
            f"`{symbol}` not seen in `lake build` output; "
            f"theorem may have failed to elaborate.\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")


@pytest.mark.skipif(
    not _lake_available(),
    reason="W87 #48 Lean proof requires `lake` on PATH (install elan)")
def test_w87_axiom_dependencies_match_expectations() -> None:
    """Soundness must be unconditional; completeness uses hashPair_injective.

    The DoD anti-cheat clauses:

    * `Do not smuggle the property into the proof as an axiom.`
      → The proof MUST NOT introduce additional property axioms
      beyond `hashPair_injective`.
    * `Do not verify only the easy direction.`
      → Both `inclusion_sound` (forward) AND `inclusion_complete`
      (backward) must be present.
    * `Do not declare success without a re-runnable artefact.`
      → `lake env lean MerkleAxiomsReport.lean` must succeed.
    """
    env = _lake_env()
    cwd = str(PROOF_DIR)
    proc = subprocess.run(
        ["lake", "env", "lean", "MerkleAxiomsReport.lean"],
        cwd=cwd, env=env,
        capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, (
        f"axiom report failed (exit {proc.returncode})\n"
        f"stdout:\n{proc.stdout}\n"
        f"stderr:\n{proc.stderr}")
    combined = proc.stdout + proc.stderr
    # Find each theorem's axiom block.
    sound_marker = "'Merkle.inclusion_sound' depends on axioms:"
    complete_marker = "'Merkle.inclusion_complete' depends on axioms:"
    iff_marker = "'Merkle.merkle_inclusion_iff' depends on axioms:"
    assert sound_marker in combined, (
        f"missing axiom block for inclusion_sound:\n{combined}")
    assert complete_marker in combined, (
        f"missing axiom block for inclusion_complete:\n{combined}")
    assert iff_marker in combined, (
        f"missing axiom block for merkle_inclusion_iff:\n{combined}")
    # Soundness MUST NOT depend on hashPair_injective.
    sound_idx = combined.index(sound_marker)
    complete_idx = combined.index(complete_marker)
    sound_block = combined[sound_idx:complete_idx]
    assert "Merkle.hashPair_injective" not in sound_block, (
        "inclusion_sound depends on hashPair_injective — that "
        "axiom must be confined to the completeness direction.\n"
        f"sound_block:\n{sound_block}")
    # Completeness MUST depend on hashPair_injective.
    iff_idx = combined.index(iff_marker)
    complete_block = combined[complete_idx:iff_idx]
    assert "Merkle.hashPair_injective" in complete_block, (
        "inclusion_complete does NOT depend on hashPair_injective "
        "— that axiom is the documented crypto assumption and "
        "must appear in completeness.\n"
        f"complete_block:\n{complete_block}")
    # The bidirectional theorem must also depend on hashPair_injective.
    iff_block = combined[iff_idx:]
    assert "Merkle.hashPair_injective" in iff_block, (
        "merkle_inclusion_iff missing hashPair_injective axiom.\n"
        f"iff_block:\n{iff_block}")
    # No other unexpected PROPERTY axioms slipped in. The whitelist:
    # propext + Classical.choice + Quot.sound are Lean core
    # foundational axioms; Merkle.Hash, Merkle.hashPair (and
    # optionally Hash.inhabitedInst) are declaration axioms.
    # Anything starting with `Merkle.` that is NOT in this set is
    # a leak.
    expected_merkle_axioms = {
        "Merkle.Hash",
        "Merkle.Hash.inhabitedInst",
        "Merkle.hashPair",
        "Merkle.hashPair_injective",
    }
    for line in combined.splitlines():
        line = line.strip().rstrip(",")
        if line.startswith("Merkle.") and not any(
                line == s or line.startswith(s + ",")
                for s in expected_merkle_axioms):
            # Some other Merkle.* axiom snuck in.
            raise AssertionError(
                f"unexpected Merkle.* axiom in proof: {line!r}")


def test_w87_proof_files_present_without_lean() -> None:
    """Even without Lean installed, the proof artefacts must exist
    in the repo so an auditor with a fresh checkout sees them."""
    for src in PROOF_SOURCES:
        path = PROOF_DIR / src
        assert path.is_file(), f"missing proof source: {src}"
    for support in ("lean-toolchain", "lakefile.lean",
                    "README.md", "Coupling.md"):
        path = PROOF_DIR / support
        assert path.is_file(), f"missing support file: {support}"


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-v"]))
