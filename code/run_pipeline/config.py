"""Shared pipeline configuration."""

from __future__ import annotations

RUN_BOTH_PROCESSING = False
ZSCORE_MUA = True  # used when RUN_BOTH_PROCESSING is False

_override: tuple[bool, ...] | None = None


def set_processing_modes(*, also_original: bool = False) -> None:
    """Override processing modes for the current process (CLI --also-original)."""
    global _override
    _override = (False, True) if also_original else None


def reset_processing_modes() -> None:
    global _override
    _override = None


def zscore_modes() -> tuple[bool, ...]:
    """Processing modes to run for steps that support raw + z-scored output."""
    if _override is not None:
        return _override
    return (False, True) if RUN_BOTH_PROCESSING else (ZSCORE_MUA,)


def dual_processing_modes() -> bool:
    """True when both raw and z-scored outputs are emitted."""
    return len(zscore_modes()) > 1
