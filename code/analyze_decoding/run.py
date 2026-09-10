"""Orchestrate curated session decoding + combined average."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from decodanda import Decodanda

from analyze_decoding.combine_decode import combine_session_results
from analyze_decoding.config import (
    BIN_STEP_MS,
    BIN_WIDTH_MS,
    CROSS_VALIDATIONS,
    DECODE_WINDOW_MS,
    FIGURES_ROOT,
    GAUSSIAN_SMOOTH_MODE,
    GAUSSIAN_SMOOTH_MS,
    MIN_TRIALS_PER_CONDITION,
    NSHUFFLES,
    TRAINING_FRACTION,
    ZSCORE_MUA,
    default_curated_data_root,
)
from analyze_decoding.data import SessionDecodeData, build_session_decode_data
from analyze_decoding.io_cache import (
    cache_matches_params,
    load_session_result,
    save_combined_result,
    save_session_result,
)
from analyze_decoding.paths import combined_dir, decoding_dir, session_decode_stem
from analyze_decoding.plots import plot_combined_decode, plot_session_decode
from analyze_decoding.session_decode import decode_session_intime, enough_trials_per_side
from analyze_decoding.targets import get_target
from process_channels.preprocess import DUAL_NHP_GO_SEQS
from run_pipeline.curated import discover_curated_sessions


def _remove_session_decode_outputs(*paths: Path) -> None:
    for path in paths:
        if path.exists():
            path.unlink()


def _restrict_bins(sess: SessionDecodeData, max_bins: int) -> SessionDecodeData:
    centers = sess.bin_centers_ms[:max_bins]
    keep = set(float(t) for t in centers)
    t_attr = np.asarray(sess.data["time_from_onset"], dtype=float)
    sel = np.zeros(t_attr.shape, dtype=bool)
    for t in keep:
        sel |= t_attr == t
    data = {k: np.asarray(v)[sel] for k, v in sess.data.items()}
    return SessionDecodeData(
        session_id=sess.session_id,
        go_seq=sess.go_seq,
        trial_type=sess.trial_type,
        actor_side=sess.actor_side,
        alignment_event=sess.alignment_event,
        data=data,
        conditions=sess.conditions,
        bin_centers_ms=centers,
        n_trials=sess.n_trials,
        n_channels=sess.n_channels,
        n_left=sess.n_left,
        n_right=sess.n_right,
        channel_numbers=sess.channel_numbers,
    )


def _validate_first_bin(sess: SessionDecodeData, target_key: str) -> None:
    t0 = float(sess.bin_centers_ms[0])
    sel = np.asarray(sess.data["time_from_onset"], dtype=float) == t0
    bin_data = {k: np.asarray(v)[sel] for k, v in sess.data.items()}
    min_tr = min(MIN_TRIALS_PER_CONDITION, sess.n_left, sess.n_right)
    dec = Decodanda(
        data=bin_data,
        conditions=sess.conditions,
        verbose=False,
        min_trials_per_condition=max(2, min_tr),
        min_data_per_condition=max(2, min_tr),
    )
    perfs, nulls = dec.decode(
        training_fraction=TRAINING_FRACTION,
        cross_validations=3,
        nshuffles=0,
        return_CV=True,
        plot=False,
    )
    print(
        f"  validate bin0={t0:.0f}ms "
        f"perf={float(np.nanmean(perfs[target_key])):.3f}"
    )


def _replot_branch(
    condition: str,
    go_seq: str,
    trial_type: str,
    *,
    target_name: str,
    figures_root: Path,
    session_ids: list[str],
    n_shuf: int,
    n_cv: int,
    smooth_ms: float = GAUSSIAN_SMOOTH_MS,
    recording_monkey: str | None = None,
) -> Path:
    """Reload session npz (any layout with perf_mean), rewrite PDFs + combined clusters."""
    # target_name here is the output folder label (may differ from decode target key)
    out_dir = decoding_dir(
        condition,
        go_seq,
        trial_type,
        target_name,
        figures_root=figures_root,
        recording_monkey=recording_monkey,
    )
    results = []
    alignment_event = ""
    for sid in session_ids:
        npz_path = out_dir / f"{session_decode_stem(sid)}.npz"
        pdf_path = out_dir / f"{session_decode_stem(sid)}.pdf"
        if not npz_path.exists():
            print(f"[replot skip] missing {npz_path.name}")
            continue
        result = load_session_result(npz_path)
        if not enough_trials_per_side(result.n_left, result.n_right):
            _remove_session_decode_outputs(npz_path, pdf_path)
            print(
                f"[replot skip] {sid}: L={result.n_left} R={result.n_right} "
                f"(need ≥{MIN_TRIALS_PER_CONDITION}/side)"
            )
            continue
        # legacy null (n_bins, n_shuf) → leave cluster_mask empty unless time-locked
        if result.null.ndim == 2 and result.null.shape[0] == result.bin_centers_ms.size:
            # old layout: no valid session clusters
            result.cluster_mask = np.zeros(result.bin_centers_ms.shape, dtype=bool)
        plot_session_decode(
            result,
            pdf_path,
            condition=condition,
            smooth_ms=smooth_ms,
            cv=n_cv,
            train=TRAINING_FRACTION,
            nshuffles=n_shuf if result.null.ndim != 2 else (
                result.null.shape[0]
                if result.null.shape[0] != result.bin_centers_ms.size
                else result.null.shape[1]
            ),
        )
        results.append(result)
        alignment_event = result.alignment_event
        print(f"[replot] {pdf_path.name}")

    if not results:
        print(f"[warn] nothing to replot for {trial_type}/{go_seq}")
        return out_dir

    combined = combine_session_results(results)
    cdir = combined_dir(out_dir)
    save_combined_result(cdir / "mean_ci_decode.npz", combined)
    plot_combined_decode(
        combined,
        cdir / "mean_ci_decode.pdf",
        condition=condition,
        go_seq=go_seq,
        trial_type=trial_type,
        target=target_name,
        alignment_event=alignment_event or results[0].alignment_event,
        smooth_ms=smooth_ms,
        cv=n_cv,
        train=TRAINING_FRACTION,
        nshuffles=n_shuf,
    )
    print(f"[combined] {cdir / 'mean_ci_decode.pdf'} (n={combined.n_sessions_used})")
    return out_dir


def run_decode_branch(
    condition: str,
    go_seq: str,
    trial_type: str,
    *,
    target_name: str = "actor_choice",
    decode_name: str | None = None,
    data_root: Path | None = None,
    figures_root: Path | None = None,
    session_ids: list[str] | None = None,
    force: bool = False,
    validate_only: bool = False,
    replot_only: bool = False,
    nshuffles: int | None = None,
    cross_validations: int | None = None,
    max_sessions: int | None = None,
    max_bins: int | None = None,
    smooth_ms: float | None = None,
    session_parent: str | None = None,
    recording_monkey: str | None = None,
) -> Path:
    """Run one timing × trial-type branch; return decoding output directory."""
    root = data_root if data_root is not None else default_curated_data_root()
    figs = figures_root if figures_root is not None else FIGURES_ROOT
    target = get_target(target_name)
    target_key = next(iter(target.conditions))
    folder = decode_name if decode_name is not None else target.name
    smooth = GAUSSIAN_SMOOTH_MS if smooth_ms is None else float(smooth_ms)

    ids = list(session_ids or discover_curated_sessions(condition, data_root=root))
    if max_sessions is not None:
        ids = ids[:max_sessions]

    n_shuf = NSHUFFLES if nshuffles is None else nshuffles
    n_cv = CROSS_VALIDATIONS if cross_validations is None else cross_validations

    if replot_only:
        return _replot_branch(
            condition,
            go_seq,
            trial_type,
            target_name=folder,
            figures_root=figs,
            session_ids=ids,
            n_shuf=n_shuf,
            n_cv=n_cv,
            smooth_ms=smooth,
            recording_monkey=recording_monkey,
        )

    out_dir = decoding_dir(
        condition,
        go_seq,
        trial_type,
        folder,
        figures_root=figs,
        recording_monkey=recording_monkey,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[out] {out_dir} (smooth_ms={smooth:g})")

    results = []
    alignment_event = ""
    for sid in ids:
        stem = session_decode_stem(sid)
        npz_path = out_dir / f"{stem}.npz"
        pdf_path = out_dir / f"{stem}.pdf"

        if (
            npz_path.exists()
            and not force
            and not validate_only
            and cache_matches_params(
                npz_path,
                bin_width_ms=BIN_WIDTH_MS,
                bin_step_ms=BIN_STEP_MS,
                window_ms=DECODE_WINDOW_MS,
                nshuffles=n_shuf,
                smooth_ms=smooth,
                smooth_mode=GAUSSIAN_SMOOTH_MODE,
            )
        ):
            result = load_session_result(npz_path)
            if not enough_trials_per_side(result.n_left, result.n_right):
                _remove_session_decode_outputs(npz_path, pdf_path)
                print(
                    f"  [skip] {sid}: L={result.n_left} R={result.n_right} "
                    f"(need ≥{MIN_TRIALS_PER_CONDITION}/side)"
                )
                continue
            plot_session_decode(
                result,
                pdf_path,
                condition=condition,
                smooth_ms=smooth,
                cv=n_cv,
                train=TRAINING_FRACTION,
                nshuffles=n_shuf,
            )
            results.append(result)
            alignment_event = result.alignment_event
            print(f"[cache] {sid} ({trial_type}/{go_seq})")
            continue

        print(f"[build] {sid} ({trial_type}/{go_seq})")
        try:
            sess = build_session_decode_data(
                sid,
                condition=condition,
                go_seq=go_seq,
                trial_type=trial_type,
                target=target,
                data_root=root,
                window_ms=DECODE_WINDOW_MS,
                bin_width_ms=BIN_WIDTH_MS,
                bin_step_ms=BIN_STEP_MS,
                smooth_ms=smooth,
                zscore_mua=ZSCORE_MUA,
                session_parent=session_parent,
                recording_monkey=recording_monkey,
            )
        except (ValueError, FileNotFoundError) as exc:
            _remove_session_decode_outputs(npz_path, pdf_path)
            print(f"  [skip] {sid}: {exc}")
            continue
        if not enough_trials_per_side(sess.n_left, sess.n_right):
            _remove_session_decode_outputs(npz_path, pdf_path)
            print(
                f"  [skip] {sid}: L={sess.n_left} R={sess.n_right} "
                f"(need ≥{MIN_TRIALS_PER_CONDITION}/side)"
            )
            continue
        alignment_event = sess.alignment_event
        print(
            f"  trials={sess.n_trials} L={sess.n_left} R={sess.n_right} "
            f"ch={sess.n_channels} bins={sess.bin_centers_ms.size}"
        )

        if validate_only:
            _validate_first_bin(sess, target_key)
            continue

        if max_bins is not None and max_bins < sess.bin_centers_ms.size:
            sess = _restrict_bins(sess, max_bins)

        result = decode_session_intime(
            sess,
            target_key=target_key,
            training_fraction=TRAINING_FRACTION,
            cross_validations=n_cv,
            nshuffles=n_shuf,
            min_trials_per_condition=MIN_TRIALS_PER_CONDITION,
        )
        from analyze_decoding.session_decode import SessionDecodeResult

        if not isinstance(result, SessionDecodeResult):
            raise RuntimeError("expected single-target SessionDecodeResult")
        if not enough_trials_per_side(result.n_left, result.n_right) or np.all(
            ~np.isfinite(result.perf_mean)
        ):
            _remove_session_decode_outputs(npz_path, pdf_path)
            print(f"  [skip] {sid}: empty decode (L={result.n_left} R={result.n_right})")
            continue
        save_session_result(
            npz_path,
            result,
            bin_width_ms=BIN_WIDTH_MS,
            bin_step_ms=BIN_STEP_MS,
            window_ms=DECODE_WINDOW_MS,
            nshuffles=n_shuf,
            smooth_ms=smooth,
            smooth_mode=GAUSSIAN_SMOOTH_MODE,
        )
        plot_session_decode(
            result,
            pdf_path,
            condition=condition,
            smooth_ms=smooth,
            cv=n_cv,
            train=TRAINING_FRACTION,
            nshuffles=n_shuf,
        )
        results.append(result)
        print(f"  wrote {pdf_path.name}")

    if validate_only:
        return out_dir

    if not results:
        print(f"[warn] no results for {trial_type}/{go_seq}")
        return out_dir

    combined = combine_session_results(results)
    cdir = combined_dir(out_dir)
    save_combined_result(cdir / "mean_ci_decode.npz", combined)
    plot_combined_decode(
        combined,
        cdir / "mean_ci_decode.pdf",
        condition=condition,
        go_seq=go_seq,
        trial_type=trial_type,
        target=folder,
        alignment_event=alignment_event or results[0].alignment_event,
        smooth_ms=smooth,
        cv=n_cv,
        train=TRAINING_FRACTION,
        nshuffles=n_shuf,
    )
    print(f"[combined] {cdir / 'mean_ci_decode.pdf'} (n={combined.n_sessions_used})")
    return out_dir


def run_decode_curated(
    condition: str,
    *,
    go_seqs: tuple[str, ...] | None = None,
    trial_types: tuple[str, ...] = ("Dyadic", "SoloA"),
    target_name: str = "actor_choice",
    decode_name: str | None = None,
    data_root: Path | None = None,
    figures_root: Path | None = None,
    force: bool = False,
    validate_only: bool = False,
    replot_only: bool = False,
    nshuffles: int | None = None,
    cross_validations: int | None = None,
    max_sessions: int | None = None,
    max_bins: int | None = None,
    smooth_ms: float | None = None,
    session_ids: list[str] | None = None,
    session_parent: str | None = None,
    recording_monkey: str | None = None,
) -> None:
    seqs = go_seqs if go_seqs is not None else DUAL_NHP_GO_SEQS
    for go_seq in seqs:
        for trial_type in trial_types:
            run_decode_branch(
                condition,
                go_seq,
                trial_type,
                target_name=target_name,
                decode_name=decode_name,
                data_root=data_root,
                figures_root=figures_root,
                session_ids=session_ids,
                session_parent=session_parent,
                recording_monkey=recording_monkey,
                force=force,
                validate_only=validate_only,
                replot_only=replot_only,
                nshuffles=nshuffles,
                cross_validations=cross_validations,
                max_sessions=max_sessions,
                max_bins=max_bins,
                smooth_ms=smooth_ms,
            )
