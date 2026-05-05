# Security policy

## Supported versions

Only the latest minor release of CoordPy receives security fixes. The
v3.43 line is the final release of the SDK v3.4x research programme;
post-3.43 fixes will ship on the next minor.

| Version | Supported          |
| ------- | ------------------ |
| 0.5.x   | :white_check_mark: |
| < 0.5   | :x:                |

## Reporting a vulnerability

Please **do not** open a public GitHub issue for security problems.

Email the maintainers with details and, if possible, a reproducer:

- **Contact:** open a private security advisory at
  <https://github.com/adotdong29/context-zero/security/advisories/new>

We aim to acknowledge reports within 5 business days and to publish a
fix or mitigation within 30 days for confirmed issues.

## Scope

In scope:

- The CoordPy SDK / runtime (``coordpy``).
- The CLIs (``coordpy``, ``coordpy-import``, ``coordpy-ci``,
  ``coordpy-capsule``).
- The capsule-graph schemas and provenance manifests they emit.

Out of scope:

- The research-grade ``coordpy.__experimental__`` surface.
- Issues that require an attacker to already control the local Python
  environment (e.g. malicious imports, write access to ``$HOME``).
- Misuse of optional sandbox extras (``coordpy[docker]``,
  ``RestrictedPython``); the sandbox is opt-in and not a hardened
  isolation boundary.
