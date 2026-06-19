# Security

## Supported versions

Only the latest minor release receives security fixes.

| Version | Supported |
| ------- | --------- |
| 1.2.x   | yes       |
| < 1.2   | no        |

## Reporting

Please do not open a public GitHub issue for security problems.
Open a private security advisory at
<https://github.com/adotdong29/CoordPy/security/advisories/new>.

We aim to acknowledge reports within five business days and to
ship a fix or mitigation within thirty days for confirmed issues.

## Scope

In scope:

- The `coordpy.adk` library surface and runtime.
- The `coordpy` SDK and runtime.
- The CLIs (`coordpy`, `coordpy-team`, `coordpy-capsule`,
  `coordpy-subject`, `coordpy-import`, `coordpy-ci`).
- The on-disk schemas (capsule view, team result, provenance
  manifest, product report, CI verdict, import audit).

Out of scope:

- The research surface under `coordpy.__experimental__`.
- Issues that require an attacker to already control the local
  Python environment (malicious imports, write access to `$HOME`).
- The optional sandbox extras (`coordpy-ai[docker]`,
  `RestrictedPython`). They are convenience wrappers, not a
  hardened isolation boundary.
