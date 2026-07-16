"""In-memory and on-disk cache for one pipeline run."""

from __future__ import annotations

import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from process_channels.features import ChannelSummary, process_channel_mua
from load_data.io import (
    build_base_mask,
    channel_number,
    choice_mask,
    discover_channel_files,
    load_time_vector,
    load_trial_labels,
    trial_window_means,
    window_indices,
)
from run_pipeline.config import zscore_modes
from process_channels.preprocess import TrialBranchSpec, zscore_channel_trials, zscore_reference_mask
from run_pipeline.context import PipelineContext
from analyze_stability import combine as cs
from analyze_stability import consistency as acc

INVALID_LABELS = frozenset({"NONE", "None", "none", ""})


@dataclass
class PooledTrialData:
    t_ms: np.ndarray
    win_idx: np.ndarray
    left_by_ch: dict[int, list[np.ndarray]]
    right_by_ch: dict[int, list[np.ndarray]]
    sess_left_by_ch: dict[int, list[float]]
    sess_right_by_ch: dict[int, list[float]]


@dataclass
class BranchState:
    """Summaries and trials for one trial-type branch (dyadic or solo)."""

    summaries: dict[bool, list[ChannelSummary]] = field(default_factory=dict)
    trials: dict[bool, dict[str, dict[int, tuple[np.ndarray, np.ndarray]]]] = field(
        default_factory=dict,
    )
    pooled: PooledTrialData | None = None


@dataclass
class BranchCacheView:
    """Facade passed to pipeline steps for one branch's cached data."""

    ctx: PipelineContext
    session_ids: list[str]
    summaries: dict[bool, list[ChannelSummary]]
    trials: dict[bool, dict[str, dict[int, tuple[np.ndarray, np.ndarray]]]]
    pooled: PooledTrialData | None
    t_ms: np.ndarray | None = None
    win_idx: np.ndarray | None = None

    def write_disk_caches(self) -> None:
        for zscore_mua, summaries in self.summaries.items():
            if summaries:
                write_summaries_disk_cache(
                    self.ctx.output_base,
                    self.ctx.condition_label,
                    zscore_mua,
                    summaries,
                )
        if self.pooled is not None:
            write_pooled_disk_cache(
                self.ctx.output_base,
                self.ctx.condition_label,
                True,
                self.pooled,
            )


@dataclass
class RunDataCache:
    """Cache for one timing run; may hold dyadic and solo branches."""

    ctx: PipelineContext
    session_ids: list[str]
    loadmat_calls: int = 0
    branches: dict[str, BranchState] = field(default_factory=dict)
    branch_specs: list[TrialBranchSpec] = field(default_factory=list)
    t_ms: np.ndarray | None = None
    win_idx: np.ndarray | None = None

    @property
    def summaries(self) -> dict[bool, list[ChannelSummary]]:
        return self._default_branch().summaries

    @property
    def trials(self) -> dict[bool, dict[str, dict[int, tuple[np.ndarray, np.ndarray]]]]:
        return self._default_branch().trials

    @property
    def pooled(self) -> PooledTrialData | None:
        return self._default_branch().pooled

    def branch(self, name: str) -> BranchState:
        if name not in self.branches:
            raise KeyError(f"Unknown branch {name!r}; have {sorted(self.branches)}")
        return self.branches[name]

    def branch_view(self, branch_ctx: PipelineContext, branch_name: str) -> BranchCacheView:
        state = self.branch(branch_name)
        return BranchCacheView(
            ctx=branch_ctx,
            session_ids=self.session_ids,
            summaries=state.summaries,
            trials=state.trials,
            pooled=state.pooled,
            t_ms=self.t_ms,
            win_idx=self.win_idx,
        )

    def _default_branch(self) -> BranchState:
        if "dyadic" in self.branches:
            return self.branches["dyadic"]
        if len(self.branches) == 1:
            return next(iter(self.branches.values()))
        raise ValueError("No default branch in cache")

    def populate(
        self,
        steps: list[str],
        branch_specs: list[TrialBranchSpec] | None = None,
    ) -> None:
        modes = self._modes_for_steps(steps)
        if not modes:
            return

        if branch_specs is None:
            branch_specs = [
                TrialBranchSpec("dyadic", "", dict(self.ctx.trial_filters)),
            ]
        self.branch_specs = list(branch_specs)

        for spec in branch_specs:
            state = BranchState()
            for zscore_mua in modes:
                state.summaries[zscore_mua] = []
                state.trials[zscore_mua] = {}
            self.branches[spec.name] = state

        for session_id in self.session_ids:
            self._load_session(session_id, modes, branch_specs)

        if "combine" in steps or "array_combined" in steps:
            for spec in branch_specs:
                self.branches[spec.name].pooled = self._build_pooled(
                    spec.name, True,
                )

    def _modes_for_steps(self, steps: list[str]) -> tuple[bool, ...]:
        modes: set[bool] = set()
        if any(s in steps for s in ("session_lr", "consistency", "best_worst")):
            modes.update(zscore_modes())
        if "combine" in steps or "array_combined" in steps:
            modes.add(True)
        return tuple(sorted(modes))

    def _load_session(
        self,
        session_id: str,
        modes: tuple[bool, ...],
        branch_specs: list[TrialBranchSpec],
    ) -> None:
        session_dir = self.ctx.session_dir(session_id)
        event = self.ctx.alignment_event(session_id)
        event_dir = session_dir / event
        if not event_dir.exists():
            warnings.warn(f"Skipping {session_id}: missing event dir {event_dir}")
            return

        need_zscore = True in modes
        try:
            labels = load_trial_labels(session_dir, session_id)
            zscore_ref = None
            if need_zscore:
                monkey = self.ctx.resolved_recording_monkey(session_id)
                zscore_ref = zscore_reference_mask(labels, monkey, session_id=session_id)

            t_ms = load_time_vector(
                event_dir, session_id, event, acc.PRE_POST_TAG,
            )
            win_idx = window_indices(t_ms, acc.ANALYSIS_WINDOW_MS)
            if win_idx.size == 0:
                warnings.warn(f"Skipping {session_id}: empty analysis window")
                return

            if self.t_ms is None:
                self.t_ms = t_ms
                self.win_idx = win_idx
            elif not np.allclose(self.t_ms, t_ms):
                warnings.warn(f"Skipping {session_id}: time vector mismatch")
                return
        except Exception as exc:
            warnings.warn(f"Skipping {session_id}: {exc}")
            return

        branch_masks: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for spec in branch_specs:
            base_mask = build_base_mask(
                labels, spec.trial_filters, invalid_labels=INVALID_LABELS,
            )
            left_mask = choice_mask(
                labels, base_mask, self.ctx.left_choice, field=self.ctx.choice_field,
            )
            right_mask = choice_mask(
                labels, base_mask, self.ctx.right_choice, field=self.ctx.choice_field,
            )
            branch_masks[spec.name] = (left_mask, right_mask)

        session_counts: dict[str, dict[bool, int]] = {
            spec.name: {False: 0, True: 0} for spec in branch_specs
        }
        for ch_path in discover_channel_files(event_dir):
            ch_num = channel_number(ch_path)
            try:
                mua_raw = self._load_channel_mat(ch_path)
            except Exception as exc:
                warnings.warn(f"Skipping ch{ch_num:03d} in {session_id}: {exc}")
                continue

            mua_z = None
            if need_zscore:
                mua_z = zscore_channel_trials(mua_raw, reference_mask=zscore_ref)

            for spec in branch_specs:
                left_mask, right_mask = branch_masks[spec.name]
                state = self.branches[spec.name]
                process_kwargs = dict(
                    ch_num=ch_num,
                    session_id=session_id,
                    t_ms=t_ms,
                    win_idx=win_idx,
                    left_mask=left_mask,
                    right_mask=right_mask,
                    gaussian_smooth_ms=acc.GAUSSIAN_SMOOTH_MS,
                    min_trials=acc.MIN_TRIALS_PER_GROUP,
                )

                if False in modes:
                    result = process_channel_mua(mua_raw, **process_kwargs)
                    if result is not None:
                        state.summaries[False].append(result.summary)
                        state.trials[False].setdefault(session_id, {})[ch_num] = (
                            result.left_trials, result.right_trials,
                        )
                        session_counts[spec.name][False] += 1

                if need_zscore and mua_z is not None:
                    result = process_channel_mua(mua_z, **process_kwargs)
                    if result is not None:
                        state.summaries[True].append(result.summary)
                        state.trials[True].setdefault(session_id, {})[ch_num] = (
                            result.left_trials, result.right_trials,
                        )
                        session_counts[spec.name][True] += 1

        loaded_any = False
        for spec in branch_specs:
            counts = session_counts[spec.name]
            n_loaded = max(counts.values()) if counts else 0
            if not n_loaded:
                continue
            loaded_any = True
            parts = []
            if False in modes and counts[False]:
                parts.append(f"raw={counts[False]}")
            if need_zscore and counts[True]:
                parts.append(f"z={counts[True]}")
            detail = ", ".join(parts) if parts else str(n_loaded)
            branch_tag = spec.output_subdir or "Dyadic"
            print(f"  {session_id} [{branch_tag}]: {detail} channels (1× loadmat each)")

        if not loaded_any:
            print(f"  {session_id}: no channels loaded")

    def _load_channel_mat(self, ch_path: Path) -> np.ndarray:
        from scipy.io import loadmat

        self.loadmat_calls += 1
        return loadmat(ch_path)["cur_output_data"]

    def _build_pooled(self, branch_name: str, zscore_mua: bool) -> PooledTrialData:
        if self.t_ms is None or self.win_idx is None:
            raise ValueError(f"No usable sessions for {self.ctx.condition_label}")

        state = self.branch(branch_name)
        left_by_ch: dict[int, list[np.ndarray]] = defaultdict(list)
        right_by_ch: dict[int, list[np.ndarray]] = defaultdict(list)
        sess_left_by_ch: dict[int, list[float]] = defaultdict(list)
        sess_right_by_ch: dict[int, list[float]] = defaultdict(list)

        session_trials = state.trials.get(zscore_mua, {})
        for session_id in self.session_ids:
            ch_trials = session_trials.get(session_id, {})
            for ch_num, (left_trials, right_trials) in ch_trials.items():
                if left_trials.size:
                    left_by_ch[ch_num].append(left_trials)
                if right_trials.size:
                    right_by_ch[ch_num].append(right_trials)

                left_rates = trial_window_means(left_trials, self.win_idx)
                right_rates = trial_window_means(right_trials, self.win_idx)
                n_left = int(np.sum(~np.isnan(left_rates))) if left_rates.size else 0
                n_right = int(np.sum(~np.isnan(right_rates))) if right_rates.size else 0
                if n_left >= cs.MIN_TRIALS_PER_GROUP and n_right >= cs.MIN_TRIALS_PER_GROUP:
                    sess_left_by_ch[ch_num].append(float(np.nanmean(left_rates)))
                    sess_right_by_ch[ch_num].append(float(np.nanmean(right_rates)))

        return PooledTrialData(
            t_ms=self.t_ms,
            win_idx=self.win_idx,
            left_by_ch=dict(left_by_ch),
            right_by_ch=dict(right_by_ch),
            sess_left_by_ch=dict(sess_left_by_ch),
            sess_right_by_ch=dict(sess_right_by_ch),
        )

    def write_disk_caches(self, branch_ctx: PipelineContext | None = None) -> None:
        ctx = branch_ctx or self.ctx
        state = self._branch_state_for_ctx(ctx)
        for zscore_mua, summaries in state.summaries.items():
            if summaries:
                write_summaries_disk_cache(
                    ctx.output_base,
                    ctx.condition_label,
                    zscore_mua,
                    summaries,
                )
        if state.pooled is not None:
            write_pooled_disk_cache(
                ctx.output_base,
                ctx.condition_label,
                True,
                state.pooled,
            )

    def _branch_state_for_ctx(self, ctx: PipelineContext) -> BranchState:
        if ctx.output_base == self.ctx.output_base:
            return self._default_branch()
        for spec in self.branch_specs:
            if spec.output_subdir and ctx.output_base == self.ctx.output_base / spec.output_subdir:
                return self.branches[spec.name]
        return self._default_branch()


def branch_has_summaries(state: BranchState) -> bool:
    """True if the branch produced at least one channel summary."""
    return any(state.summaries.get(zscore_mua, []) for zscore_mua in (False, True))


def disk_cache_dir(output_base: Path, condition_label: str) -> Path:
    """Per-run cache dir. ``output_base`` is already scoped (e.g. ``.../Curius_AgoB``)."""
    del condition_label  # kept for API compatibility with callers
    return output_base / ".cache"


def disk_cache_path(output_base: Path, condition_label: str, zscore_mua: bool) -> Path:
    tag = "zscore" if zscore_mua else "raw"
    return disk_cache_dir(output_base, condition_label) / f"summaries_{tag}.npz"


def pooled_disk_cache_path(output_base: Path, condition_label: str, zscore_mua: bool) -> Path:
    tag = "zscore" if zscore_mua else "raw"
    return disk_cache_dir(output_base, condition_label) / f"pooled_{tag}.npz"


def _legacy_disk_cache_path(output_base: Path, condition_label: str, zscore_mua: bool) -> Path:
    """Pre-fix layout: ``{output_base}/.cache/{condition_label}/summaries_*.npz``."""
    tag = "zscore" if zscore_mua else "raw"
    return output_base / ".cache" / condition_label / f"summaries_{tag}.npz"


def _split_stacked_parts(stacked: np.ndarray, part_rows: np.ndarray) -> list[np.ndarray]:
    if stacked.size == 0 or part_rows.size == 0:
        return []
    parts: list[np.ndarray] = []
    offset = 0
    for n_rows in part_rows.tolist():
        n = int(n_rows)
        parts.append(np.asarray(stacked[offset : offset + n], dtype=float))
        offset += n
    return parts


def write_pooled_disk_cache(
    output_base: Path,
    condition_label: str,
    zscore_mua: bool,
    pooled: PooledTrialData,
) -> Path:
    """Save stacked L/R trial traces for one figure folder."""
    path = pooled_disk_cache_path(output_base, condition_label, zscore_mua)
    path.parent.mkdir(parents=True, exist_ok=True)
    channels = sorted(
        set(pooled.left_by_ch) | set(pooled.right_by_ch)
        | set(pooled.sess_left_by_ch) | set(pooled.sess_right_by_ch)
    )
    payload: dict[str, np.ndarray] = {
        "t_ms": np.asarray(pooled.t_ms, dtype=float),
        "win_idx": np.asarray(pooled.win_idx, dtype=np.int32),
        "channels": np.asarray(channels, dtype=np.int32),
    }
    for ch in channels:
        left_parts = pooled.left_by_ch.get(ch, [])
        right_parts = pooled.right_by_ch.get(ch, [])
        if left_parts:
            payload[f"L_{ch}"] = np.vstack(left_parts)
            payload[f"L_{ch}_rows"] = np.asarray(
                [part.shape[0] for part in left_parts], dtype=np.int32,
            )
        if right_parts:
            payload[f"R_{ch}"] = np.vstack(right_parts)
            payload[f"R_{ch}_rows"] = np.asarray(
                [part.shape[0] for part in right_parts], dtype=np.int32,
            )
        sess_left = pooled.sess_left_by_ch.get(ch, [])
        sess_right = pooled.sess_right_by_ch.get(ch, [])
        if sess_left:
            payload[f"sessL_{ch}"] = np.asarray(sess_left, dtype=float)
        if sess_right:
            payload[f"sessR_{ch}"] = np.asarray(sess_right, dtype=float)
    np.savez_compressed(path, **payload)
    return path


def load_pooled_disk_cache(
    output_base: Path,
    condition_label: str,
    zscore_mua: bool,
) -> PooledTrialData | None:
    path = pooled_disk_cache_path(output_base, condition_label, zscore_mua)
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=False)
    channels = [int(ch) for ch in np.asarray(data["channels"]).tolist()]
    left_by_ch: dict[int, list[np.ndarray]] = {}
    right_by_ch: dict[int, list[np.ndarray]] = {}
    sess_left_by_ch: dict[int, list[float]] = {}
    sess_right_by_ch: dict[int, list[float]] = {}
    for ch in channels:
        left_key, left_rows_key = f"L_{ch}", f"L_{ch}_rows"
        right_key, right_rows_key = f"R_{ch}", f"R_{ch}_rows"
        if left_key in data.files:
            left_by_ch[ch] = _split_stacked_parts(data[left_key], data[left_rows_key])
        if right_key in data.files:
            right_by_ch[ch] = _split_stacked_parts(data[right_key], data[right_rows_key])
        sess_left_key, sess_right_key = f"sessL_{ch}", f"sessR_{ch}"
        if sess_left_key in data.files:
            sess_left_by_ch[ch] = [float(v) for v in np.asarray(data[sess_left_key]).tolist()]
        if sess_right_key in data.files:
            sess_right_by_ch[ch] = [float(v) for v in np.asarray(data[sess_right_key]).tolist()]
    return PooledTrialData(
        t_ms=np.asarray(data["t_ms"], dtype=float),
        win_idx=np.asarray(data["win_idx"], dtype=np.int32),
        left_by_ch=left_by_ch,
        right_by_ch=right_by_ch,
        sess_left_by_ch=sess_left_by_ch,
        sess_right_by_ch=sess_right_by_ch,
    )


def require_pooled_disk_cache(
    output_base: Path,
    condition_label: str,
    zscore_mua: bool = True,
) -> PooledTrialData:
    """Load stacked trials from disk, or raise with the expected path."""
    pooled = load_pooled_disk_cache(output_base, condition_label, zscore_mua)
    if pooled is not None:
        return pooled
    path = pooled_disk_cache_path(output_base, condition_label, zscore_mua)
    raise FileNotFoundError(
        f"Missing stacked-trial cache: {path}\n"
        "Run the full pipeline once (include combine or array_combined) first."
    )


def _summary_to_arrays(summary: ChannelSummary) -> dict[str, np.ndarray | float | int | str | bool]:
    return {
        "session_id": np.array(summary.session_id),
        "channel": np.int32(summary.channel),
        "array_name": np.array(summary.array_name),
        "index_in_array": np.int32(summary.index_in_array),
        "n_left": np.int32(summary.n_left),
        "n_right": np.int32(summary.n_right),
        "t_ms": summary.t_ms,
        "mean_left": summary.mean_left,
        "mean_right": summary.mean_right,
        "diff": summary.diff,
        "si": np.float64(summary.si),
        "mwu_p": np.float64(summary.mwu_p) if summary.mwu_p is not None else np.nan,
        "pref_side": np.array(summary.pref_side),
        "evoked_p_left": np.float64(summary.evoked_p_left) if summary.evoked_p_left is not None else np.nan,
        "evoked_p_right": np.float64(summary.evoked_p_right) if summary.evoked_p_right is not None else np.nan,
        "task_evoked": np.bool_(summary.task_evoked),
    }


def _summary_from_arrays(row: dict[str, np.ndarray]) -> ChannelSummary:
    def _optional_float(val: float) -> float | None:
        return None if not np.isfinite(val) else float(val)

    session_id = str(row["session_id"].item()) if row["session_id"].shape else str(row["session_id"])
    pref_side = str(row["pref_side"].item()) if row["pref_side"].shape else str(row["pref_side"])
    array_name = str(row["array_name"].item()) if row["array_name"].shape else str(row["array_name"])
    return ChannelSummary(
        session_id=session_id,
        channel=int(row["channel"]),
        array_name=array_name,
        index_in_array=int(row["index_in_array"]),
        n_left=int(row["n_left"]),
        n_right=int(row["n_right"]),
        t_ms=np.asarray(row["t_ms"], dtype=float),
        mean_left=np.asarray(row["mean_left"], dtype=float),
        mean_right=np.asarray(row["mean_right"], dtype=float),
        diff=np.asarray(row["diff"], dtype=float),
        si=float(row["si"]),
        mwu_p=_optional_float(float(row["mwu_p"])),
        pref_side=pref_side,
        evoked_p_left=_optional_float(float(row["evoked_p_left"])),
        evoked_p_right=_optional_float(float(row["evoked_p_right"])),
        task_evoked=bool(row["task_evoked"]),
    )


def write_summaries_disk_cache(
    output_base: Path,
    condition_label: str,
    zscore_mua: bool,
    summaries: list[ChannelSummary],
) -> Path:
    path = disk_cache_path(output_base, condition_label, zscore_mua)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {"n_summaries": np.int32(len(summaries))}
    for i, summary in enumerate(summaries):
        for key, val in _summary_to_arrays(summary).items():
            payload[f"{i}_{key}"] = val
    np.savez_compressed(path, **payload)
    return path


def load_summaries_disk_cache(
    output_base: Path,
    condition_label: str,
    zscore_mua: bool,
) -> list[ChannelSummary] | None:
    path = disk_cache_path(output_base, condition_label, zscore_mua)
    if not path.exists():
        path = _legacy_disk_cache_path(output_base, condition_label, zscore_mua)
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    n = int(data["n_summaries"])
    summaries: list[ChannelSummary] = []
    for i in range(n):
        row = {key.split("_", 1)[1]: data[key] for key in data.files if key.startswith(f"{i}_")}
        summaries.append(_summary_from_arrays(row))
    return summaries
