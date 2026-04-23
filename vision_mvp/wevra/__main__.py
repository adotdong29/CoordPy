"""Allow ``python -m vision_mvp.wevra`` as an alias for ``wevra``."""
from __future__ import annotations

from ._cli import main_run


if __name__ == "__main__":
    raise SystemExit(main_run())
