"""Pipeline orchestration: curated conditions, flat session lists, DUAL_NHP runs."""

from __future__ import annotations

import warnings
from pathlib import Path

from bos_mua.steps import best_worst as pbw
from bos_mua.steps import combine as cs
from bos_mua.steps import consistency as acc
from bos_mua.steps import session_lr as psl
from bos_mua.confederate import iter_confederate_runs
from bos_mua.dual_nhp import iter_dual_nhp_runs
from bos_mua.preprocess import (
    choice_config_for_monkey,
    recording_monkey_from_condition_label,
    resolve_condition_output_dir,
    trial_filters_for_condition,
)
from bos_mua.run_context import PipelineContext, apply_pipeline_context, session_ids_for_run
from bos_mua.session_lists import SessionListConfig, load_session_list

ALL_STEPS = (
    "session_lr",
    "consistency",
    "combine",
    "array_combined",
    "best_worst",
    "timing_compare",
)


def parse_steps(steps_arg: str) -> list[str]:
    steps = [s.strip() for s in steps_arg.split(",") if s.strip()]
    unknown = set(steps) - set(ALL_STEPS)
    if unknown:
        raise ValueError(
            f"Unknown step(s): {', '.join(sorted(unknown))}. "
            f"Valid: {', '.join(ALL_STEPS)}"
        )
    return steps


def verify_sessions(ctx: PipelineContext) -> list[str]:
    """Return session IDs with existing directories; warn on missing."""
    present: list[str] = []
    missing: list[str] = []
    for session_id in ctx.session_ids:
        if ctx.session_dir(session_id).exists():
            present.append(session_id)
        else:
            missing.append(session_id)
    for session_id in missing:
        warnings.warn(f"Missing session directory: {ctx.session_dir(session_id)}")
    if not present:
        raise FileNotFoundError(
            f"No session directories found under {ctx.data_root} for "
            f"{ctx.condition_label!r}"
        )
    return present


def _print_run_header(ctx: PipelineContext) -> None:
    choice = f"{ctx.choice_field} L={ctx.left_choice} R={ctx.right_choice}"
    print(f"DATA_ROOT: {ctx.data_root}")
    print(f"Layout: {ctx.layout} | label: {ctx.condition_label}")
    print(f"Recording monkey: {ctx.recording_monkey or '(infer per session)'}")
    print(f"Choice: {choice}")
    print(f"Output base: {ctx.output_base}")
    print(f"Sessions: {len(ctx.session_ids)}")
    for session_id in ctx.session_ids:
        print(f"  {session_id}")
    print(f"TRIAL_FILTERS: {ctx.trial_filters}")


def run_pipeline_steps(ctx: PipelineContext, steps: list[str]) -> None:
    """Run selected pipeline steps for one RunSpec / PipelineContext."""
    apply_pipeline_context(ctx)
    session_ids = verify_sessions(ctx)
    _print_run_header(ctx)

    if "session_lr" in steps:
        print("\n" + "=" * 72)
        print("STEP: plot_session_lr_mua")
        print("=" * 72)
        psl.main()

    if "consistency" in steps:
        print("\n" + "=" * 72)
        print("STEP: assess_cross_session_consistency")
        print("=" * 72)
        acc.main()

    if "combine" in steps:
        print("\n" + "=" * 72)
        print("STEP: combine_sessions (z-scored combined)")
        print("=" * 72)
        if len(session_ids) < cs.MIN_SESSIONS:
            warnings.warn(
                f"Skipping combine for {ctx.condition_label}: need >={cs.MIN_SESSIONS} "
                f"sessions, found {len(session_ids)}"
            )
        else:
            output_dir = resolve_condition_output_dir(
                cs.OUTPUT_DIR, True, ctx.condition_label, "combined",
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            cs.plot_condition_combined(ctx.condition_label, output_dir, session_ids=session_ids)

    if "array_combined" in steps:
        print("\n" + "=" * 72)
        print("STEP: plot_export_condition_arrays (array-mean combined L/R)")
        print("=" * 72)
        if len(session_ids) < cs.MIN_SESSIONS:
            warnings.warn(
                f"Skipping array_combined for {ctx.condition_label}: "
                f"need >={cs.MIN_SESSIONS} sessions, found {len(session_ids)}"
            )
        else:
            from bos_mua.steps.array_combined import plot_condition_array_combined

            plot_condition_array_combined(ctx)

    if "best_worst" in steps:
        print("\n" + "=" * 72)
        print("STEP: plot_best_worst_channels")
        print("=" * 72)
        modes = (False, True) if acc.RUN_BOTH_PROCESSING else (acc.ZSCORE_MUA,)
        for zscore_mua in modes:
            pbw.run_channel_rank_plots(ctx.condition_label, zscore_mua)

    print(f"\nPipeline finished for {ctx.condition_label}")


def build_curated_context(condition: str, trial_filters: dict) -> PipelineContext:
    """Build context for nested MUA_curated_sessions layout."""
    monkey = recording_monkey_from_condition_label(condition)
    choice = choice_config_for_monkey(monkey)
    data_root = Path(psl.DATA_ROOT)
    session_ids = session_ids_for_run(data_root, condition)
    figures_path = Path(psl.OUTPUT_DIR)
    output_base = figures_path.parent if figures_path.name == "figures" else Path(".")
    return PipelineContext(
        data_root=data_root,
        session_ids=session_ids,
        output_base=output_base,
        condition_label=condition,
        condition_key=condition,
        layout="curated",
        trial_filters=trial_filters,
        choice_field=choice.field,
        left_choice=list(choice.left),
        right_choice=list(choice.right),
        recording_monkey=monkey,
    )


def build_flat_session_list_context(cfg: SessionListConfig) -> PipelineContext:
    """Build context for a flat-export session list (curated-style filters, one monkey)."""
    monkey = recording_monkey_from_condition_label(cfg.condition_key)
    choice = choice_config_for_monkey(monkey)
    return PipelineContext(
        data_root=cfg.root_folder,
        session_ids=cfg.session_ids,
        output_base=cfg.output_folder,
        condition_label=cfg.list_name,
        condition_key=cfg.condition_key,
        layout="flat",
        trial_filters=trial_filters_for_condition(cfg.condition_key),
        choice_field=choice.field,
        left_choice=list(choice.left),
        right_choice=list(choice.right),
        recording_monkey=monkey,
    )


def run_dual_nhp_pipeline(
    session_lists_path: str | Path,
    steps: list[str],
    *,
    monkey: str | None = None,
    go_seq: str = "all",
) -> None:
    """Run all requested DUAL_NHP monkey × timing combinations."""
    session_lists_path = Path(session_lists_path)
    ran = False
    monkeys_run: set[str] = set()
    per_run_steps = [s for s in steps if s != "timing_compare"]
    for ctx in iter_dual_nhp_runs(
        session_lists_path, monkey=monkey, go_seq=go_seq,
    ):
        ran = True
        monkeys_run.add(ctx.recording_monkey)
        print("\n" + "#" * 72)
        print(f"# DUAL_NHP / {ctx.condition_label} ({len(ctx.session_ids)} sessions)")
        print("#" * 72)
        run_pipeline_steps(ctx, per_run_steps)
    if not ran:
        raise FileNotFoundError(
            "No DUAL_NHP runs matched (check session list and --monkey / --go-seq filters)"
        )
    if "timing_compare" in steps:
        from bos_mua.steps import timing_compare as timing

        for name in sorted(monkeys_run):
            print("\n" + "#" * 72)
            print(f"# DUAL_NHP / {name}_first_second_comparison")
            print("#" * 72)
            timing.main(["--monkey", name, "--session-lists", str(session_lists_path)])


def run_confederate_pipeline(
    session_lists_path: str | Path,
    list_name: str,
    steps: list[str],
    *,
    go_seq: str = "all",
) -> None:
    """Run confederate list × AgoB/BgoA combinations plus optional timing comparison."""
    session_lists_path = Path(session_lists_path)
    cfg = load_session_list(list_name, session_lists_path)
    monkey = recording_monkey_from_condition_label(cfg.condition_key)
    per_run_steps = [s for s in steps if s != "timing_compare"]
    ran = False
    if per_run_steps:
        for ctx in iter_confederate_runs(cfg, go_seq=go_seq):
            ran = True
            print("\n" + "#" * 72)
            print(f"# {list_name} / {ctx.condition_label} ({len(ctx.session_ids)} sessions)")
            print("#" * 72)
            run_pipeline_steps(ctx, per_run_steps)
    elif "timing_compare" in steps:
        ran = True
    if not ran:
        raise FileNotFoundError(
            f"No confederate runs matched for {list_name!r} (check --go-seq filter)"
        )
    if "timing_compare" in steps:
        from bos_mua.steps import timing_compare as timing

        print("\n" + "#" * 72)
        print(f"# {list_name} / {monkey}_first_second_comparison")
        print("#" * 72)
        timing.main([
            "--monkey", monkey,
            "--list-name", list_name,
            "--session-lists", str(session_lists_path),
        ])
