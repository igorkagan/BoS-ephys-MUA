"""Ensure repo root and code/ are on sys.path for tests."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_CODE = _REPO / "code"
for _p in (_REPO, _CODE):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)
