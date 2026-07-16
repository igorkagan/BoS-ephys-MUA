"""Pipeline orchestration for curated and session-list runs."""

from __future__ import annotations

__all__ = [
    "ALL_STEPS",
    "build_flat_session_list_context",
    "parse_steps",
    "run_confederate_pipeline",
    "run_condition_pipeline",
    "run_curated_pipeline",
    "run_dual_nhp_pipeline",
    "run_pipeline_steps",
    "verify_sessions",
]

def __getattr__(name: str):
    if name == "verify_sessions":
        from run_pipeline.context import verify_sessions
        return verify_sessions
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from run_pipeline import runner

    return getattr(runner, name)
