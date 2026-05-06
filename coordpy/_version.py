"""Single source of truth for the CoordPy version.

Kept in its own module so it can be imported without triggering the
heavy ``coordpy/__init__.py`` re-exports.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.5.18"
