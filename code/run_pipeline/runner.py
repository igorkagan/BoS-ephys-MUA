"""Pipeline orchestration: curated, session-list, DUAL_NHP runs."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterator
from pathlib import Path

from analyze_stability import arrays as array_step
from analyze_stability import combine as cs
from analyze_stability import consistency as acc
from analyze_stability import deep_dives as pbw
from analyze_stability import pref_unpref as pu
from analyze_stability import session_lr as psl
from analyze_stability import stability_across_sessions as sas
from load_data.cache import (
    BranchCacheView,
    RunDataCache,
    branch_has_summaries,
    require_pooled_disk_cache,
    load_summaries_disk_cache,
    trim_allocator,
)
from load_data.io import session_sort_key
from load_data.sessions import SessionListConfig, load_dual_nhp_configs, load_session_list
from process_channels.preprocess import (
    DUAL_NHP_GO_SEQS,
    TrialBranchSpec,
    branch_specs_for_run,
    choice_config_for_actor_side,
    recording_actor_side,
    recording_monkey_from_condition_label,
    resolve_condition_output_dir,
    solo_output_subdir_for_actor_side,
    trial_filters_for_condition,
    trial_filters_for_go_seq,
)
from run_pipeline.confederate import iter_confederate_runs
from run_pipeline.config import zscore_modes
from run_pipeline.context import PipelineContext, apply_pipeline_context, verify_sessions
from run_pipeline.contracts import comparison_requests
from run_pipeline.curated import curated_condition_output, iter_curated_runs, verify_curated_condition
from run_pipeline.run_log import pipeline_run_log
from run_pipeline.dual_nhp import iter_dual_nhp_runs
from run_pipeline.output_contracts import validate_pipeline_outputs

ALL_STEPS = (
    "session_lr",
    "pref_unpref",
    "consistency",
    "stability_across_sessions",
    "combine",
    "array_combined",
    "best_worst",
    "comparisons",
)

_NON_DATA_STEPS = frozenset({"comparisons", "stability_across_sessions", "pref_unpref"})
_SUMMARY_STEPS = frozenset({"session_lr", "consistency", "best_worst"})
_POOLED_STEPS = frozenset({"combine", "array_combined"})


def parse_steps(steps_arg: str) -> list[str]:
    steps = [s.strip() for s in steps_arg.split(",") if s.strip()]
    if "timing_compare" in steps:
        warnings.warn(
            "'timing_compare' is deprecated; use 'comparisons'",
            DeprecationWarning,
            stacklevel=2,
        )
        steps = ["comparisons" if step == "timing_compare" else step for step in steps]
    unknown = set(steps) - set(ALL_STEPS)
    if unknown:
        raise ValueError(
            f"Unknown step(s): {', '.join(sorted(unknown))}. "
            f"Valid: {', '.join(ALL_STEPS)}"
        )
    return steps


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


def _consistency_reference(session_ids: list[str]) -> tuple[str, int]:
    if acc.REFERENCE_SESSION and acc.REFERENCE_SESSION in session_ids:
        ref_session = acc.REFERENCE_SESSION
    else:
        ref_session = min(session_ids, key=session_sort_key)
    return ref_session, session_ids.index(ref_session)


def branch_context(base_ctx: PipelineContext, spec: TrialBranchSpec) -> PipelineContext:
    """Child context for one branch (go-seq × social folder under base output)."""
    output_base = base_ctx.output_base
    if spec.output_subdir:
        output_base = base_ctx.output_base / spec.output_subdir
    condition_label = spec.condition_label or base_ctx.condition_label
    return PipelineContext(
        data_root=base_ctx.data_root,
        session_ids=list(base_ctx.session_ids),
        output_base=output_base,
        condition_label=condition_label,
        condition_key=base_ctx.condition_key,
        layout=base_ctx.layout,
        trial_filters=spec.trial_filters,
        source_kind=base_ctx.source_kind,
        session_parent=base_ctx.session_parent,
        choice_field=base_ctx.choice_field,
        left_choice=list(base_ctx.left_choice),
        right_choice=list(base_ctx.right_choice),
        recording_monkey=base_ctx.recording_monkey,
    )


def _specs_for_go_seq_contexts(
    contexts: list[PipelineContext],
    *,
    include_solo: bool,
    shared_parent: Path,
) -> list[TrialBranchSpec]:
    """Merge AgoB/BgoA contexts into one load with unique branch names/paths."""
    specs: list[TrialBranchSpec] = []
    for ctx in contexts:
        rel_root = ctx.output_base.relative_to(shared_parent)
        for spec in branch_specs_for_run(ctx, include_solo=include_solo):
            subdir = str(rel_root / spec.output_subdir) if spec.output_subdir else str(rel_root)
            specs.append(
                TrialBranchSpec(
                    name=f"{ctx.condition_label}__{spec.name}",
                    output_subdir=subdir.replace("\\", "/"),
                    trial_filters=spec.trial_filters,
                    condition_label=ctx.condition_label,
                ),
            )
    return specs


def _load_key(ctx: PipelineContext) -> tuple:
    return (
        str(ctx.data_root),
        tuple(ctx.session_ids),
        ctx.layout,
        ctx.source_kind,
        ctx.session_parent,
        ctx.recording_monkey,
        ctx.choice_field,
        tuple(ctx.left_choice),
        tuple(ctx.right_choice),
    )


def _disk_view_for_branch(
    child_ctx: PipelineContext,
    session_ids: list[str],
    *,
    need_pooled: bool,
    need_summaries: bool,
) -> BranchCacheView:
    summaries: dict[bool, list] = {}
    if need_summaries:
        for zscore_mua in zscore_modes():
            cached = load_summaries_disk_cache(
                child_ctx.output_base, child_ctx.condition_label, zscore_mua,
            )
            if cached:
                summaries[zscore_mua] = cached
        if not summaries:
            raise FileNotFoundError(
                f"Missing averages+SI cache under {child_ctx.output_base / '.cache'}\n"
                "Run the full pipeline once first."
            )
    pooled = None
    if need_pooled:
        pooled = require_pooled_disk_cache(
            child_ctx.output_base, child_ctx.condition_label, True,
        )
    return BranchCacheView(
        ctx=child_ctx,
        session_ids=session_ids,
        summaries=summaries,
        trials={},
        pooled=pooled,
    )


def _cache_has_trials(cache) -> bool:
    trials = getattr(cache, "trials", None) or {}
    return any(bool(sessions) for sessions in trials.values())


def run_pipeline_steps(
    ctx: PipelineContext,
    steps: list[str],
    *,
    cache_view: BranchCacheView | None = None,
) -> None:
    """Run selected pipeline steps for one RunSpec / PipelineContext."""
    apply_pipeline_context(ctx)
    session_ids = verify_sessions(ctx)
    _print_run_header(ctx)

    data_steps = [s for s in steps if s not in _NON_DATA_STEPS]
    if cache_view is None:
        cache = RunDataCache(ctx, session_ids)
        if data_steps:
            print("\n" + "=" * 72)
            print("STEP: load session/channel data (single read per channel)")
            print("=" * 72)

            def _on_session(session_id: str) -> None:
                if "session_lr" in steps:
                    psl.plot_one_session_from_cache(cache, session_id)

            cache.populate(data_steps, on_session=_on_session)
            if "combine" in steps or "array_combined" in steps:
                for name in cache.branches:
                    cache.ensure_pooled(name)
        active = cache
    else:
        active = cache_view

    if "session_lr" in steps:
        print("\n" + "=" * 72)
        if _cache_has_trials(active):
            print("STEP: plot_session_lr_mua")
            print("=" * 72)
            psl.plot_from_cache(active, session_ids)
        else:
            print("STEP: plot_session_lr_mua (already written during load)")
            print("=" * 72)

    if "pref_unpref" in steps:
        print("\n" + "=" * 72)
        print("STEP: pref_unpref (MWU-sig channels, per-session preference)")
        print("=" * 72)
        summaries = sas.summaries_for_run(ctx, active.summaries)
        if not summaries:
            warnings.warn(
                f"Skipping pref_unpref for {ctx.condition_label}: "
                "no z-scored summaries (run the pipeline once or use disk cache)"
            )
        else:
            pu.run_from_summaries(ctx, summaries)

    ref_session, ref_idx = _consistency_reference(session_ids)

    if "consistency" in steps:
        print("\n" + "=" * 72)
        print("STEP: assess_cross_session_consistency")
        print("=" * 72)
        if len(session_ids) < acc.MIN_SESSIONS:
            raise ValueError(
                f"Need at least {acc.MIN_SESSIONS} sessions, found {len(session_ids)}"
            )
        acc.run_from_cache(active, session_ids, ref_session, ref_idx)

    if "stability_across_sessions" in steps:
        print("\n" + "=" * 72)
        print("STEP: stability_across_sessions")
        print("=" * 72)
        summaries = sas.summaries_for_run(ctx, active.summaries)
        sessions_with_summaries = len({s.session_id for s in summaries})
        if len(session_ids) < sas.MIN_SESSIONS:
            warnings.warn(
                f"Skipping stability_across_sessions for {ctx.condition_label}: "
                f"need >={sas.MIN_SESSIONS} sessions, found {len(session_ids)}"
            )
        elif sessions_with_summaries < sas.MIN_SESSIONS:
            warnings.warn(
                f"Skipping stability_across_sessions for {ctx.condition_label}: "
                f"need >={sas.MIN_SESSIONS} sessions with summaries, "
                f"found {sessions_with_summaries}"
            )
        elif not summaries:
            warnings.warn(
                f"Skipping stability_across_sessions for {ctx.condition_label}: "
                "no z-scored summaries (run consistency first or use disk cache)"
            )
        else:
            sas.run_stability_across_sessions(ctx, session_ids, summaries)

    if "combine" in steps:
        print("\n" + "=" * 72)
        print("STEP: combine_sessions (z-scored combined)")
        print("=" * 72)
        if len(session_ids) < cs.MIN_SESSIONS:
            warnings.warn(
                f"Skipping combine for {ctx.condition_label}: need >={cs.MIN_SESSIONS} "
                f"sessions, found {len(session_ids)}"
            )
        elif active.pooled is None:
            warnings.warn(f"Skipping combine for {ctx.condition_label}: no pooled data")
        else:
            output_dir = resolve_condition_output_dir(
                cs.OUTPUT_DIR, True, ctx.condition_label, "combined",
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            cs.plot_from_pooled(
                active.pooled,
                ctx.condition_label,
                output_dir,
                session_ids,
                trial_filters=ctx.trial_filters,
            )

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
            array_step.plot_condition_array_combined(ctx, pooled=active.pooled)

    if "best_worst" in steps:
        if "consistency" in steps:
            print("\n" + "=" * 72)
            print("STEP: plot_best_worst_channels (skipped — already in consistency)")
            print("=" * 72)
        else:
            print("\n" + "=" * 72)
            print("STEP: plot_best_worst_channels")
            print("=" * 72)
            for zscore_mua in zscore_modes():
                pbw.run_from_cache(active, ctx.condition_label, zscore_mua)

    if data_steps:
        if cache_view is not None:
            cache_view.write_disk_caches()
        elif isinstance(active, RunDataCache) and active.branches:
            active.write_disk_caches(ctx)

    if cache_view is not None or not isinstance(active, RunDataCache) or active.branches:
        validate_pipeline_outputs(ctx, steps, active)
    print(f"\nPipeline finished for {ctx.condition_label}")


def run_branched_pipeline(
    base_ctx: PipelineContext,
    steps: list[str],
    *,
    include_solo: bool = True,
    branch_specs: list[TrialBranchSpec] | None = None,
) -> None:
    """Load once per channel, then run pipeline steps for each social/go-seq branch."""
    apply_pipeline_context(base_ctx)
    session_ids = verify_sessions(base_ctx)
    specs = branch_specs if branch_specs is not None else branch_specs_for_run(
        base_ctx, include_solo=include_solo,
    )
    data_steps = [s for s in steps if s not in _NON_DATA_STEPS]
    need_summaries = bool(set(data_steps) & _SUMMARY_STEPS)
    need_pooled = bool(set(data_steps) & _POOLED_STEPS)

    if not data_steps:
        for spec in specs:
            child_ctx = branch_context(base_ctx, spec)
            cached = load_summaries_disk_cache(
                child_ctx.output_base, child_ctx.condition_label, True,
            )
            if not cached:
                warnings.warn(
                    f"Skipping {spec.output_subdir or spec.name} branch for "
                    f"{base_ctx.condition_label}: no z-scored disk summary cache"
                )
                continue
            view = BranchCacheView(
                ctx=child_ctx,
                session_ids=session_ids,
                summaries={True: cached},
                trials={},
                pooled=None,
            )
            label_suffix = f" / {spec.output_subdir}" if spec.output_subdir else ""
            print("\n" + "=" * 72)
            print(f"BRANCH: {child_ctx.condition_label}{label_suffix}")
            print("=" * 72)
            run_pipeline_steps(child_ctx, steps, cache_view=view)
        return

    # Replot combine/array_combined from saved stacked trials (no .mat re-read).
    if need_pooled and not need_summaries:
        for spec in specs:
            child_ctx = branch_context(base_ctx, spec)
            label_suffix = f" / {spec.output_subdir}" if spec.output_subdir else ""
            print("\n" + "=" * 72)
            print(f"BRANCH: {child_ctx.condition_label}{label_suffix} (disk stacked trials)")
            print("=" * 72)
            view = _disk_view_for_branch(
                child_ctx,
                session_ids,
                need_pooled=True,
                need_summaries=(
                    "stability_across_sessions" in steps or "pref_unpref" in steps
                ),
            )
            run_pipeline_steps(child_ctx, steps, cache_view=view)
            del view
            trim_allocator()
        return

    shared = RunDataCache(base_ctx, session_ids)
    print("\n" + "=" * 72)
    print("STEP: load session/channel data (single read per channel, multi-branch)")
    print("=" * 72)
    _print_run_header(base_ctx)
    print(f"Branches: {', '.join(spec.output_subdir or spec.name for spec in specs)}")
    if "session_lr" in steps:
        print("session_lr PDFs are written per session during load, then trial arrays are dropped")

    def _on_session(session_id: str) -> None:
        if "session_lr" not in steps:
            return
        for spec in specs:
            child_ctx = branch_context(base_ctx, spec)
            apply_pipeline_context(child_ctx)
            view = shared.branch_view(child_ctx, spec.name)
            psl.plot_one_session_from_cache(view, session_id)

    shared.populate(data_steps, specs, on_session=_on_session)

    for spec in specs:
        child_ctx = branch_context(base_ctx, spec)
        branch_state = shared.branch(spec.name)
        is_primary = spec.name == "dyadic" or spec.name.endswith("__dyadic")
        if not is_primary and not branch_has_summaries(branch_state):
            warnings.warn(
                f"Skipping {spec.output_subdir} branch for {child_ctx.condition_label}: "
                "no channel summaries"
            )
            shared.release_branch(spec.name)
            continue
        if need_pooled:
            shared.ensure_pooled(spec.name)
        label_suffix = f" / {spec.output_subdir}" if spec.output_subdir else ""
        print("\n" + "=" * 72)
        print(f"BRANCH: {child_ctx.condition_label}{label_suffix}")
        print("=" * 72)
        view = shared.branch_view(child_ctx, spec.name)
        run_pipeline_steps(child_ctx, steps, cache_view=view)
        shared.release_branch(spec.name)


def _run_comparisons(
    *,
    monkey: str,
    social_context: str = "dyadic",
    comparison_axis: str = "go_sequence",
    go_seq: str | None = None,
    zscore: bool = True,
    session_lists_path: Path | None = None,
    list_name: str | None = None,
    curated_condition: str | None = None,
    only_combined: bool = False,
) -> None:
    from compare_conditions import compare as timing

    args = [
        "--monkey",
        monkey,
        "--social-context",
        social_context,
        "--comparison-axis",
        comparison_axis,
        "--zscore",
        str(zscore),
    ]
    if go_seq is not None:
        args.extend(["--go-seq", go_seq])
    if curated_condition:
        args.extend(["--curated-condition", curated_condition])
    else:
        args.extend(["--session-lists", str(session_lists_path)])
        if list_name and list_name != "DUAL_NHP":
            args.extend(["--list-name", list_name])
    if only_combined:
        args.append("--only-combined")
    timing.main(args)


def _run_comparison_suite(
    *,
    monkey: str,
    suite_label: str,
    include_solo: bool,
    go_seq: str,
    session_lists_path: Path | None = None,
    list_name: str | None = None,
    curated_condition: str | None = None,
    session_ids: list[str] | None = None,
    only_combined: bool = False,
) -> None:
    """Run exactly the comparison matrix supported by the selected branches."""
    if session_ids:
        actor_side = recording_actor_side(session_ids[0], monkey)
    else:
        actor_side = "A" if monkey == "Elmo" else "B"
    solo_subdir = solo_output_subdir_for_actor_side(actor_side)

    requests = comparison_requests(include_solo=include_solo, go_seq=go_seq)
    if go_seq != "all":
        warnings.warn(
            "Skipping AgoB-vs-BgoA comparisons because --go-seq selected only "
            f"{go_seq}; both branches are required."
        )

    for request in requests:
        for zscore_mua in zscore_modes():
            if request.comparison_axis == "go_sequence":
                social = solo_subdir if request.social_context == "solo" else "Dyadic"
                label = f"{monkey}_{social}_first_second_comparison"
            else:
                label = f"{monkey}_{request.go_seq}/Dyadic_vs_{solo_subdir}_comparison"
            mode = "zscored" if zscore_mua else "original"
            print("\n" + "#" * 72)
            print(f"# {suite_label} / {label} / {mode}")
            print("#" * 72)
            _run_comparisons(
                monkey=monkey,
                comparison_axis=request.comparison_axis,
                social_context=request.social_context,
                go_seq=request.go_seq,
                zscore=zscore_mua,
                session_lists_path=session_lists_path,
                list_name=list_name,
                curated_condition=curated_condition,
                only_combined=only_combined,
            )


def _go_seq_from_condition_label(label: str) -> str | None:
    for seq in DUAL_NHP_GO_SEQS:
        if label.endswith(f"_{seq}"):
            return seq
    return None


def _write_pref_comparison_suites(
    group: list[PipelineContext],
    *,
    include_solo: bool,
) -> None:
    """Fill comparison combined/ folders with pref overlays from disk summaries."""
    from compare_conditions.compare import (
        go_sequence_spec,
        processing_scoped_output,
        social_context_spec,
    )

    if not group:
        return
    monkey = group[0].recording_monkey
    session_ids = list(group[0].session_ids)
    if not session_ids or not monkey:
        return
    actor_side = recording_actor_side(session_ids[0], monkey)
    solo_subdir = solo_output_subdir_for_actor_side(actor_side)
    alignment = group[0].alignment_event_label(session_ids)
    by_seq: dict[str, PipelineContext] = {}
    for ctx in group:
        seq = _go_seq_from_condition_label(ctx.condition_label)
        if seq is not None:
            by_seq[seq] = ctx

    def _load_branch(ctx: PipelineContext, subdir: str) -> list:
        cached = load_summaries_disk_cache(
            ctx.output_base / subdir, ctx.condition_label, True,
        )
        return cached or []

    def _write(summaries_a, summaries_b, out_root, spec, title: str) -> None:
        if not summaries_a and not summaries_b:
            warnings.warn(f"Skipping pref comparison {spec.file_tag}: no summaries")
            return
        combined_dir = processing_scoped_output(out_root, True) / pu.COMBINED_SUBDIR
        pu.write_comparison_pref_combined(
            summaries_a,
            summaries_b,
            combined_dir,
            file_tag=spec.file_tag,
            label_a=spec.label_a,
            label_b=spec.label_b,
            suptitle_grids=(
                f"{title} | combined sessions | {alignment} | pref vs unpref"
            ),
            suptitle_arrays=(
                f"{title} | array mean ± SE | {alignment} | pref vs unpref"
            ),
        )

    if include_solo:
        for seq, ctx in by_seq.items():
            spec = social_context_spec(solo_subdir, seq)
            _write(
                _load_branch(ctx, "Dyadic"),
                _load_branch(ctx, solo_subdir),
                ctx.output_base / f"Dyadic_vs_{solo_subdir}_comparison",
                spec,
                f"{ctx.condition_label} {spec.file_tag}",
            )

    shared_parent = group[0].output_base.parent
    if (
        all(ctx.output_base.parent == shared_parent for ctx in group)
        and "AgoB" in by_seq
        and "BgoA" in by_seq
    ):
        socials = ["Dyadic"]
        if include_solo:
            socials.append(solo_subdir)
        for social in socials:
            spec = go_sequence_spec(social)
            _write(
                _load_branch(by_seq["AgoB"], social),
                _load_branch(by_seq["BgoA"], social),
                shared_parent / f"{monkey}_{social}_first_second_comparison",
                spec,
                f"{monkey} {social} {spec.file_tag}",
            )


def run_condition_pipeline(
    contexts: Iterator[PipelineContext],
    steps: list[str],
    *,
    include_solo: bool = True,
    run_header: Callable[[PipelineContext], str] | None = None,
    comparisons: Callable[[], None] | None = None,
) -> None:
    """Shared engine: one load for all go-seqs that share sessions, then comparisons."""
    per_run_steps = [s for s in steps if s != "comparisons"]
    context_list = list(contexts)
    ran = False
    if per_run_steps:
        groups: dict[tuple, list[PipelineContext]] = {}
        for ctx in context_list:
            groups.setdefault(_load_key(ctx), []).append(ctx)

        for group in groups.values():
            ran = True
            if len(group) == 1:
                ctx = group[0]
                if run_header:
                    print("\n" + "#" * 72)
                    print(run_header(ctx))
                    print("#" * 72)
                run_branched_pipeline(ctx, per_run_steps, include_solo=include_solo)
            elif any(ctx.output_base.parent != group[0].output_base.parent for ctx in group):
                for ctx in group:
                    if run_header:
                        print("\n" + "#" * 72)
                        print(run_header(ctx))
                        print("#" * 72)
                    run_branched_pipeline(ctx, per_run_steps, include_solo=include_solo)
            else:
                shared_parent = group[0].output_base.parent
                labels = ", ".join(ctx.condition_label for ctx in group)
                print("\n" + "#" * 72)
                print(f"# one-load group: {labels} ({len(group[0].session_ids)} sessions)")
                print("#" * 72)
                shared_ctx = PipelineContext(
                    data_root=group[0].data_root,
                    session_ids=list(group[0].session_ids),
                    output_base=shared_parent,
                    condition_label=group[0].condition_key or shared_parent.name,
                    condition_key=group[0].condition_key,
                    layout=group[0].layout,
                    trial_filters={},
                    source_kind=group[0].source_kind,
                    session_parent=group[0].session_parent,
                    choice_field=group[0].choice_field,
                    left_choice=list(group[0].left_choice),
                    right_choice=list(group[0].right_choice),
                    recording_monkey=group[0].recording_monkey,
                )
                specs = _specs_for_go_seq_contexts(
                    group, include_solo=include_solo, shared_parent=shared_parent,
                )
                run_branched_pipeline(
                    shared_ctx,
                    per_run_steps,
                    include_solo=include_solo,
                    branch_specs=specs,
                )
            if "pref_unpref" in per_run_steps:
                _write_pref_comparison_suites(group, include_solo=include_solo)
    elif "comparisons" in steps:
        ran = True
    if not ran:
        raise FileNotFoundError("No pipeline runs matched the requested filters")
    if "comparisons" in steps and comparisons is not None:
        comparisons()


def run_curated_pipeline(
    condition: str,
    steps: list[str],
    *,
    go_seq: str = "all",
    include_solo: bool = True,
    data_root: Path | None = None,
    output_base: Path | None = None,
    only_combined: bool = False,
) -> None:
    """Curated: discover sessions from disk; same engine as confederate lists."""
    verify_curated_condition(condition, data_root=data_root)
    monkey = recording_monkey_from_condition_label(condition)
    contexts = list(
        iter_curated_runs(
            condition, go_seq=go_seq, data_root=data_root, output_base=output_base,
        )
    )
    if not contexts:
        raise FileNotFoundError(f"No curated runs for {condition!r}")

    log_root = curated_condition_output(condition, figures_root=output_base)

    def _comparisons() -> None:
        first_ctx = contexts[0]
        _run_comparison_suite(
            monkey=monkey,
            suite_label=condition,
            include_solo=include_solo,
            go_seq=go_seq,
            curated_condition=condition,
            session_ids=first_ctx.session_ids,
            only_combined=only_combined,
        )

    with pipeline_run_log(log_root):
        run_condition_pipeline(
            iter(contexts),
            steps,
            include_solo=include_solo,
            run_header=lambda ctx: (
                f"# curated / {condition} / {ctx.condition_label} ({len(ctx.session_ids)} sessions)"
            ),
            comparisons=_comparisons if "comparisons" in steps else None,
        )


def build_flat_session_list_context(cfg: SessionListConfig) -> PipelineContext:
    """Build context for a flat-export session list (single AgoB run)."""
    monkey = recording_monkey_from_condition_label(cfg.condition_key)
    if not cfg.session_ids:
        raise ValueError(f"No sessions in list {cfg.list_name!r}")
    actor_side = recording_actor_side(cfg.session_ids[0], monkey)
    choice = choice_config_for_actor_side(actor_side)
    return PipelineContext(
        data_root=cfg.root_folder,
        session_ids=cfg.session_ids,
        output_base=cfg.output_folder,
        condition_label=cfg.list_name,
        condition_key=cfg.condition_key,
        layout="flat",
        trial_filters=trial_filters_for_condition(cfg.condition_key, actor_side),
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
    include_solo: bool = True,
    only_combined: bool = False,
) -> None:
    """Run all requested DUAL_NHP monkey x timing combinations."""
    session_lists_path = Path(session_lists_path)
    monkeys_run: set[str] = set()

    cfg, dual_split = load_dual_nhp_configs(session_lists_path)

    def _track(ctx: PipelineContext) -> str:
        monkeys_run.add(ctx.recording_monkey)
        return f"# DUAL_NHP / {ctx.condition_label} ({len(ctx.session_ids)} sessions)"

    def _comparisons() -> None:
        selected = [monkey] if monkey else [name for name, ids in dual_split.items() if ids]
        for name in sorted(monkeys_run or set(selected)):
            _run_comparison_suite(
                monkey=name,
                suite_label="DUAL_NHP",
                include_solo=include_solo,
                go_seq=go_seq,
                session_lists_path=session_lists_path,
                list_name="DUAL_NHP",
                session_ids=dual_split.get(name),
                only_combined=only_combined,
            )

    with pipeline_run_log(cfg.output_folder):
        run_condition_pipeline(
            iter_dual_nhp_runs(session_lists_path, monkey=monkey, go_seq=go_seq),
            steps,
            include_solo=include_solo,
            run_header=_track,
            comparisons=_comparisons if "comparisons" in steps else None,
        )


def run_confederate_pipeline(
    session_lists_path: str | Path,
    list_name: str,
    steps: list[str],
    *,
    go_seq: str = "all",
    include_solo: bool = True,
    only_combined: bool = False,
) -> None:
    """Run confederate list x AgoB/BgoA — same engine as curated."""
    session_lists_path = Path(session_lists_path)
    cfg = load_session_list(list_name, session_lists_path)
    monkey = recording_monkey_from_condition_label(cfg.condition_key)

    def _comparisons() -> None:
        _run_comparison_suite(
            monkey=monkey,
            suite_label=list_name,
            include_solo=include_solo,
            go_seq=go_seq,
            session_lists_path=session_lists_path,
            list_name=list_name,
            session_ids=cfg.session_ids,
            only_combined=only_combined,
        )

    with pipeline_run_log(cfg.output_folder):
        run_condition_pipeline(
            iter_confederate_runs(cfg, go_seq=go_seq),
            steps,
            include_solo=include_solo,
            run_header=lambda ctx: (
                f"# {list_name} / {ctx.condition_label} ({len(ctx.session_ids)} sessions)"
            ),
            comparisons=_comparisons if "comparisons" in steps else None,
        )
