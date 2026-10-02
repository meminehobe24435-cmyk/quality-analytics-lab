"""允许 `py -3.12 -m qalab ...` 直接调用 CLI。"""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
