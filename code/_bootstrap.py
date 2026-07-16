"""Allow run_scripts entry points to work both directly and with ``python -m``."""

from __future__ import annotations

import sys
from pathlib import Path

_CODE = Path(__file__).resolve().parent
_REPO = _CODE.parent
for path in (_REPO, _CODE):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)
