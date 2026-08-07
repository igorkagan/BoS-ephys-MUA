"""Dyadic decoding grids: choice_ab (unbalanced/balanced) and same_diff."""

from __future__ import annotations

from pathlib import Path

import numpy as np

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
from analyze_decoding.data import (
    build_aligned_multilabel_bundle,
    subset_for_conditions,
)
from analyze_decoding.paths import decoding_dir, session_decode_stem
from analyze_decoding.plots_grid import (
    plot_choice_ab_grid_combined,
    plot_choice_ab_grid_session,
    plot_same_diff_combined,
    plot_same_diff_session,
)
from analyze_decoding.session_decode import SessionDecodeResult, decode_session_intime
from analyze_decoding.targets import (
    BALANCED_AB_CONDITIONS,
    align_events_in_order,
)
from run_pipeline.curated import discover_curated_sessions

ROW_KEYS = ("A_choice", "B_choice")


def _save_result_arrays(result: SessionDecodeResult) -> dict:
    return {
        "perf_mean": result.perf_mean,
        "perf_sem_cv": result.perf_sem_cv,
        "null": result.null,
        "pvalues": result.pvalues,
        "cluster_mask": result.cluster_mask,
        "cluster_p": result.cluster_p,
        "cluster_threshold": result.cluster_threshold,
        "n_trials": result.n_trials,
        "n_left": result.n_left,
        "n_right": result.n_right,
        "n_channels": result.n_channels,
        "balanced": result.balanced,
        "target_key": result.target_key,
        "alignment_event": result.alignment_event,
    }


def _load_result_arrays(
    z: np.lib.npyio.NpzFile,
    prefix: str,
    *,
    session_id: str,
    go_seq: str,
    trial_type: str,
    bin_centers: np.ndarray,
) -> SessionDecodeResult:
    return SessionDecodeResult(
        session_id=session_id,
        go_seq=go_seq,
        trial_type=trial_type,
        target_key=str(z[f"{prefix}_target_key"]),
        bin_centers_ms=bin_centers,
        perf_mean=z[f"{prefix}_perf_mean"],
        perf_sem_cv=z[f"{prefix}_perf_sem_cv"],
        null=z[f"{prefix}_null"],
        pvalues=z[f"{prefix}_pvalues"],
        n_trials=int(z[f"{prefix}_n_trials"]),
        n_left=int(z[f"{prefix}_n_left"]),
        n_right=int(z[f"{prefix}_n_right"]),
        n_channels=int(z[f"{prefix}_n_channels"]),
        alignment_event=str(z[f"{prefix}_alignment_event"]),
        cluster_mask=np.asarray(z[f"{prefix}_cluster_mask"], dtype=bool),
        cluster_p=np.asarray(z[f"{prefix}_cluster_p"], dtype=float),
        cluster_threshold=float(z[f"{prefix}_cluster_threshold"]),
        balanced=bool(z[f"{prefix}_balanced"]),
    )


def _prefix(event: str, key: str) -> str:
    side = "A" if event.startswith("A_") else "B"
    return f"{side}__{key}"


def save_choice_ab_npz(
    path: Path,
    *,
    session_id: str,
    go_seq: str,
    align_events: tuple[str, str],
    panels: dict[tuple[str, str], SessionDecodeResult],
    balanced: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sample = next(iter(panels.values()))
    n_shuf = int(sample.null.shape[0]) if sample.null.ndim == 2 else NSHUFFLES
    payload: dict = {
        "session_id": session_id,
        "go_seq": go_seq,
        "trial_type": "Dyadic",
        "balanced": balanced,
        "align_0": align_events[0],
        "align_1": align_events[1],
        "bin_centers_ms": sample.bin_centers_ms,
        "null_layout": "time_locked",
        "nshuffles": n_shuf,
        "bin_width_ms": BIN_WIDTH_MS,
        "bin_step_ms": BIN_STEP_MS,
        "bin_ms": BIN_STEP_MS,
        "smooth_ms": GAUSSIAN_SMOOTH_MS,
        "smooth_mode": GAUSSIAN_SMOOTH_MODE,
        "window_lo": DECODE_WINDOW_MS[0],
        "window_hi": DECODE_WINDOW_MS[1],
    }
    for (event, key), res in panels.items():
        pfx = _prefix(event, key)
        arrs = _save_result_arrays(res)
        for ak, av in arrs.items():
            payload[f"{pfx}_{ak}"] = av
    np.savez_compressed(path, **payload)


def load_choice_ab_npz(path: Path) -> tuple[tuple[str, str], dict[tuple[str, str], SessionDecodeResult]]:
    z = np.load(path, allow_pickle=True)
    align_events = (str(z["align_0"]), str(z["align_1"]))
    centers = z["bin_centers_ms"]
    sid, go = str(z["session_id"]), str(z["go_seq"])
    panels = {}
    for event in align_events:
        for key in ROW_KEYS:
            pfx = _prefix(event, key)
            panels[(event, key)] = _load_result_arrays(
                z, pfx, session_id=sid, go_seq=go, trial_type="Dyadic", bin_centers=centers,
            )
    return align_events, panels


def save_same_diff_npz(
    path: Path,
    *,
    session_id: str,
    go_seq: str,
    align_events: tuple[str, str],
    panels: dict[str, SessionDecodeResult],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sample = next(iter(panels.values()))
    n_shuf = int(sample.null.shape[0]) if sample.null.ndim == 2 else NSHUFFLES
    payload: dict = {
        "session_id": session_id,
        "go_seq": go_seq,
        "trial_type": "Dyadic",
        "align_0": align_events[0],
        "align_1": align_events[1],
        "bin_centers_ms": sample.bin_centers_ms,
        "null_layout": "time_locked",
        "nshuffles": n_shuf,
        "bin_width_ms": BIN_WIDTH_MS,
        "bin_step_ms": BIN_STEP_MS,
        "bin_ms": BIN_STEP_MS,
        "smooth_ms": GAUSSIAN_SMOOTH_MS,
        "smooth_mode": GAUSSIAN_SMOOTH_MODE,
        "window_lo": DECODE_WINDOW_MS[0],
        "window_hi": DECODE_WINDOW_MS[1],
    }
    for event, res in panels.items():
        pfx = "A" if event.startswith("A_") else "B"
        for ak, av in _save_result_arrays(res).items():
            payload[f"{pfx}_{ak}"] = av
    np.savez_compressed(path, **payload)


def load_same_diff_npz(path: Path) -> tuple[tuple[str, str], dict[str, SessionDecodeResult]]:
    z = np.load(path, allow_pickle=True)
    align_events = (str(z["align_0"]), str(z["align_1"]))
    centers = z["bin_centers_ms"]
    sid, go = str(z["session_id"]), str(z["go_seq"])
    panels = {}
    for event in align_events:
        pfx = "A" if event.startswith("A_") else "B"
        panels[event] = _load_result_arrays(
            z, pfx, session_id=sid, go_seq=go, trial_type="Dyadic", bin_centers=centers,
        )
    return align_events, panels


def _cache_ok(path: Path, *, balanced_flag: bool | None = None) -> bool:
    if not path.exists():
        return False
    try:
        z = np.load(path, allow_pickle=True)
    except Exception:
        return False
    if "null_layout" not in z.files or str(z["null_layout"]) != "time_locked":
        return False
    if int(z["nshuffles"]) != NSHUFFLES:
        return False
    if "bin_width_ms" not in z.files or abs(float(z["bin_width_ms"]) - BIN_WIDTH_MS) > 1e-6:
        return False
    if "bin_step_ms" not in z.files or abs(float(z["bin_step_ms"]) - BIN_STEP_MS) > 1e-6:
        return False
    if "smooth_ms" not in z.files or abs(float(z["smooth_ms"]) - GAUSSIAN_SMOOTH_MS) > 1e-6:
        return False
    if "smooth_mode" not in z.files or str(z["smooth_mode"]) != str(GAUSSIAN_SMOOTH_MODE):
        return False
    centers = z["bin_centers_ms"]
    if centers.size < 30:
        return False
    # verify actual null depth (metadata can lie if older saves used config default)
    null_keys = [k for k in z.files if k.endswith("_null")]
    if not null_keys:
        return False
    null0 = z[null_keys[0]]
    if null0.ndim != 2 or null0.shape[0] != NSHUFFLES or null0.shape[1] != centers.size:
        return False
    if balanced_flag is not None and "balanced" in z.files:
        if bool(z["balanced"]) != balanced_flag:
            return False
    return True


def run_choice_ab_grid(
    condition: str,
    go_seq: str,
    *,
    balanced: bool,
    data_root: Path | None = None,
    figures_root: Path | None = None,
    session_ids: list[str] | None = None,
    force: bool = False,
    max_sessions: int | None = None,
    nshuffles: int | None = None,
    cross_validations: int | None = None,
) -> Path:
    root = data_root or default_curated_data_root()
    figs = figures_root or FIGURES_ROOT
    ids = list(session_ids or discover_curated_sessions(condition, data_root=root))
    if max_sessions is not None:
        ids = ids[:max_sessions]

    out_dir = decoding_dir(condition, go_seq, "Dyadic", "choice_ab_grid", figures_root=figs)
    out_dir.mkdir(parents=True, exist_ok=True)
    align_events = align_events_in_order(go_seq)
    n_shuf = NSHUFFLES if nshuffles is None else nshuffles
    n_cv = CROSS_VALIDATIONS if cross_validations is None else cross_validations
    suffix = "_balanced" if balanced else ""

    all_session_panels: list[dict[tuple[str, str], SessionDecodeResult]] = []

    for sid in ids:
        stem = session_decode_stem(sid)
        npz_path = out_dir / f"{stem}{suffix}.npz"
        pdf_path = out_dir / f"{stem}{suffix}.pdf"

        if not force and _cache_ok(npz_path, balanced_flag=balanced):
            _, panels = load_choice_ab_npz(npz_path)
            plot_choice_ab_grid_session(
                panels,
                align_events=align_events,
                out_pdf=pdf_path,
                condition=condition,
                go_seq=go_seq,
                session_id=sid,
                balanced=balanced,
            )
            all_session_panels.append(panels)
            print(f"[cache] choice_ab{suffix} {sid}")
            continue

        print(f"[build] choice_ab{suffix} {sid} ({go_seq})")
        panels: dict[tuple[str, str], SessionDecodeResult] = {}
        try:
            for event in align_events:
                bundle = build_aligned_multilabel_bundle(
                    sid,
                    condition=condition,
                    go_seq=go_seq,
                    trial_type="Dyadic",
                    data_root=root,
                    alignment_event=event,
                    window_ms=DECODE_WINDOW_MS,
                    bin_width_ms=BIN_WIDTH_MS,
                    bin_step_ms=BIN_STEP_MS,
                    smooth_ms=GAUSSIAN_SMOOTH_MS,
                    zscore_mua=ZSCORE_MUA,
                )
                _fill_choice_ab_panels(
                    panels,
                    bundle=bundle,
                    event=event,
                    balanced=balanced,
                    n_cv=n_cv,
                    n_shuf=n_shuf,
                )
        except (ValueError, FileNotFoundError) as exc:
            print(f"  [skip] {sid}: {exc}")
            continue

        if len(panels) != len(align_events) * len(ROW_KEYS):
            print(f"  [skip] {sid}: incomplete panels ({len(panels)})")
            continue

        save_choice_ab_npz(
            npz_path,
            session_id=sid,
            go_seq=go_seq,
            align_events=align_events,
            panels=panels,
            balanced=balanced,
        )
        plot_choice_ab_grid_session(
            panels,
            align_events=align_events,
            out_pdf=pdf_path,
            condition=condition,
            go_seq=go_seq,
            session_id=sid,
            balanced=balanced,
        )
        all_session_panels.append(panels)
        print(f"  wrote {pdf_path.name}")

    if not all_session_panels:
        print(f"[warn] no choice_ab{suffix} results for {go_seq}")
        return out_dir

    combined_panels: dict[tuple[str, str], object] = {}
    for event in align_events:
        for key in ROW_KEYS:
            k = (event, key)
            combined_panels[k] = combine_session_results(
                [p[k] for p in all_session_panels],
            )

    cdir = out_dir / "combined"
    cdir.mkdir(parents=True, exist_ok=True)
    # persist combined lightly
    comb_path = cdir / f"mean_ci_decode{suffix}.npz"
    comb_payload = {
        "go_seq": go_seq,
        "balanced": balanced,
        "align_0": align_events[0],
        "align_1": align_events[1],
    }
    for (event, key), comb in combined_panels.items():
        pfx = _prefix(event, key)
        comb_payload[f"{pfx}_mean"] = comb.mean
        comb_payload[f"{pfx}_ci_low"] = comb.ci_low
        comb_payload[f"{pfx}_ci_high"] = comb.ci_high
        comb_payload[f"{pfx}_cluster_mask"] = comb.cluster_mask
        comb_payload[f"{pfx}_session_curves"] = comb.session_curves
        comb_payload[f"{pfx}_bin_centers_ms"] = comb.bin_centers_ms
        comb_payload[f"{pfx}_n_sessions"] = comb.n_sessions_used
    np.savez_compressed(comb_path, **comb_payload)

    plot_choice_ab_grid_combined(
        combined_panels,  # type: ignore[arg-type]
        align_events=align_events,
        out_pdf=cdir / f"mean_ci_decode{suffix}.pdf",
        condition=condition,
        go_seq=go_seq,
        balanced=balanced,
    )
    print(f"[combined] choice_ab{suffix} {go_seq}")
    return out_dir


def _fill_choice_ab_panels(
    panels: dict[tuple[str, str], SessionDecodeResult],
    *,
    bundle,
    event: str,
    balanced: bool,
    n_cv: int,
    n_shuf: int,
) -> None:
    if balanced:
        sess = subset_for_conditions(
            bundle, BALANCED_AB_CONDITIONS, count_key="A_choice",
        )
        print(
            f"  align={event.split('_')[0]} trials={sess.n_trials} "
            f"A L/R={sess.n_left}/{sess.n_right} ch={sess.n_channels}"
        )
        multi = decode_session_intime(
            sess,
            training_fraction=TRAINING_FRACTION,
            cross_validations=n_cv,
            nshuffles=n_shuf,
            min_trials_per_condition=MIN_TRIALS_PER_CONDITION,
        )
        assert isinstance(multi, dict)
        for key in ROW_KEYS:
            panels[(event, key)] = multi[key]
        return

    for key in ROW_KEYS:
        sess = subset_for_conditions(
            bundle, {key: ["left", "right"]}, count_key=key,
        )
        print(
            f"  align={event.split('_')[0]} {key} "
            f"trials={sess.n_trials} L/R={sess.n_left}/{sess.n_right}"
        )
        res = decode_session_intime(
            sess,
            target_key=key,
            training_fraction=TRAINING_FRACTION,
            cross_validations=n_cv,
            nshuffles=n_shuf,
            min_trials_per_condition=MIN_TRIALS_PER_CONDITION,
        )
        assert isinstance(res, SessionDecodeResult)
        panels[(event, key)] = res


def run_same_diff_grid(
    condition: str,
    go_seq: str,
    *,
    data_root: Path | None = None,
    figures_root: Path | None = None,
    session_ids: list[str] | None = None,
    force: bool = False,
    max_sessions: int | None = None,
    nshuffles: int | None = None,
    cross_validations: int | None = None,
) -> Path:
    root = data_root or default_curated_data_root()
    figs = figures_root or FIGURES_ROOT
    ids = list(session_ids or discover_curated_sessions(condition, data_root=root))
    if max_sessions is not None:
        ids = ids[:max_sessions]

    out_dir = decoding_dir(condition, go_seq, "Dyadic", "same_diff", figures_root=figs)
    out_dir.mkdir(parents=True, exist_ok=True)
    align_events = align_events_in_order(go_seq)
    n_shuf = NSHUFFLES if nshuffles is None else nshuffles
    n_cv = CROSS_VALIDATIONS if cross_validations is None else cross_validations

    all_panels: list[dict[str, SessionDecodeResult]] = []
    for sid in ids:
        stem = session_decode_stem(sid)
        npz_path = out_dir / f"{stem}.npz"
        pdf_path = out_dir / f"{stem}.pdf"

        if not force and _cache_ok(npz_path):
            _, panels = load_same_diff_npz(npz_path)
            plot_same_diff_session(
                panels,
                align_events=align_events,
                out_pdf=pdf_path,
                condition=condition,
                go_seq=go_seq,
                session_id=sid,
            )
            all_panels.append(panels)
            print(f"[cache] same_diff {sid}")
            continue

        print(f"[build] same_diff {sid} ({go_seq})")
        panels = {}
        try:
            for event in align_events:
                bundle = build_aligned_multilabel_bundle(
                    sid,
                    condition=condition,
                    go_seq=go_seq,
                    trial_type="Dyadic",
                    data_root=root,
                    alignment_event=event,
                )
                sess = subset_for_conditions(
                    bundle, {"same_diff": ["same", "diff"]}, count_key="same_diff",
                )
                print(
                    f"  align={event.split('_')[0]} trials={sess.n_trials} "
                    f"same/diff={sess.n_left}/{sess.n_right}"
                )
                res = decode_session_intime(
                    sess,
                    target_key="same_diff",
                    training_fraction=TRAINING_FRACTION,
                    cross_validations=n_cv,
                    nshuffles=n_shuf,
                    min_trials_per_condition=MIN_TRIALS_PER_CONDITION,
                )
                assert isinstance(res, SessionDecodeResult)
                panels[event] = res
        except (ValueError, FileNotFoundError) as exc:
            print(f"  [skip] {sid}: {exc}")
            continue

        save_same_diff_npz(
            npz_path,
            session_id=sid,
            go_seq=go_seq,
            align_events=align_events,
            panels=panels,
        )
        plot_same_diff_session(
            panels,
            align_events=align_events,
            out_pdf=pdf_path,
            condition=condition,
            go_seq=go_seq,
            session_id=sid,
        )
        all_panels.append(panels)
        print(f"  wrote {pdf_path.name}")

    if not all_panels:
        return out_dir

    combined = {
        event: combine_session_results([p[event] for p in all_panels])
        for event in align_events
    }
    cdir = out_dir / "combined"
    cdir.mkdir(parents=True, exist_ok=True)
    comb_path = cdir / "mean_ci_decode.npz"
    comb_payload: dict = {
        "go_seq": go_seq,
        "align_0": align_events[0],
        "align_1": align_events[1],
    }
    for event, comb in combined.items():
        pfx = "A" if event.startswith("A_") else "B"
        comb_payload[f"{pfx}_mean"] = comb.mean
        comb_payload[f"{pfx}_ci_low"] = comb.ci_low
        comb_payload[f"{pfx}_ci_high"] = comb.ci_high
        comb_payload[f"{pfx}_cluster_mask"] = comb.cluster_mask
        comb_payload[f"{pfx}_session_curves"] = comb.session_curves
        comb_payload[f"{pfx}_bin_centers_ms"] = comb.bin_centers_ms
        comb_payload[f"{pfx}_n_sessions"] = comb.n_sessions_used
    np.savez_compressed(comb_path, **comb_payload)
    plot_same_diff_combined(
        combined,
        align_events=align_events,
        out_pdf=cdir / "mean_ci_decode.pdf",
        condition=condition,
        go_seq=go_seq,
    )
    print(f"[combined] same_diff {go_seq}")
    return out_dir


def run_dyadic_all_grids(
    condition: str,
    *,
    go_seqs: tuple[str, ...],
    data_root: Path | None = None,
    figures_root: Path | None = None,
    force: bool = False,
    max_sessions: int | None = None,
    nshuffles: int | None = None,
    cross_validations: int | None = None,
) -> None:
    for go_seq in go_seqs:
        run_choice_ab_grid(
            condition, go_seq, balanced=False,
            data_root=data_root, figures_root=figures_root,
            force=force, max_sessions=max_sessions,
            nshuffles=nshuffles, cross_validations=cross_validations,
        )
        run_choice_ab_grid(
            condition, go_seq, balanced=True,
            data_root=data_root, figures_root=figures_root,
            force=force, max_sessions=max_sessions,
            nshuffles=nshuffles, cross_validations=cross_validations,
        )
        run_same_diff_grid(
            condition, go_seq,
            data_root=data_root, figures_root=figures_root,
            force=force, max_sessions=max_sessions,
            nshuffles=nshuffles, cross_validations=cross_validations,
        )
