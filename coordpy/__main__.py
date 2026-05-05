"""Allow ``python -m coordpy`` as an alias for the ``coordpy`` CLI."""
from __future__ import annotations

from ._cli import main_run


if __name__ == "__main__":
    raise SystemExit(main_run())
