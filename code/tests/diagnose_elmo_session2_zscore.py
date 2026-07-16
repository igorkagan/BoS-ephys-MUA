#!/usr/bin/env python3
"""Diagnose unusually high z-scored MUA in Elmo DUAL_NHP session 2."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from load_data.io import (
    build_base_mask,
    channel_files_by_number,
    load_trial_labels,
)
from process_channels.preprocess import (
    choice_config_for_recording,
    trial_filters_for_dual_nhp_monkey,
    zscore_channel_trials,
)
from load_data.sessions import load_dual_nhp_configs

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(r"S:/taskcontroller/SCP_DATA/SCP-CTRL-01/MUA_export_per_session")
ALIGN = "A_InitialFixationReleaseTime_ms"
EXAMPLE_CHANNELS = (69, 1, 80, 120)


def channel_stats(raw: np.ndarray, z: np.ndarray) -> dict:
    finite_raw = np.isfinite(raw)
    finite_z = np.isfinite(z)
    row_ok = ~np.all(np.isnan(raw), axis=1)
    raw_f = raw[finite_raw]
    z_f = z[finite_z]
    pop_sd = float(np.std(raw_f, ddof=0)) if raw_f.size else np.nan
    return {
        "n_trials": int(raw.shape[0]),
        "n_valid_rows": int(row_ok.sum()),
        "raw_mean": float(np.mean(raw_f)) if raw_f.size else np.nan,
        "raw_sd_pop": pop_sd,
        "raw_min": float(np.min(raw_f)) if raw_f.size else np.nan,
        "raw_max": float(np.max(raw_f)) if raw_f.size else np.nan,
        "raw_p99": float(np.percentile(raw_f, 99)) if raw_f.size else np.nan,
        "z_mean": float(np.mean(z_f)) if z_f.size else np.nan,
        "z_sd_pop": float(np.std(z_f, ddof=0)) if z_f.size else np.nan,
        "z_max_abs": float(np.max(np.abs(z_f))) if z_f.size else np.nan,
        "z_p99_abs": float(np.percentile(np.abs(z_f), 99)) if z_f.size else np.nan,
        "zero_sd": pop_sd == 0 or not np.isfinite(pop_sd),
        "nan_frac": float(1.0 - finite_raw.mean()) if raw.size else np.nan,
    }


def load_channel(event_dir: Path, ch: int) -> np.ndarray | None:
    by_ch = channel_files_by_number(event_dir)
    path = by_ch.get(ch)
    if path is None:
        return None
    return loadmat(path)["cur_output_data"].astype(float)


def summarize_session(session_id: str, filters: dict) -> dict:
    sdir = DATA_ROOT / session_id
    ev = sdir / ALIGN
    labels = load_trial_labels(sdir, session_id)
    base_mask = build_base_mask(labels, filters)
    choice_cfg = choice_config_for_recording(session_id, "Elmo")
    left_mask = base_mask & np.isin(labels[choice_cfg.field], choice_cfg.left)
    right_mask = base_mask & np.isin(labels[choice_cfg.field], choice_cfg.right)

    per_ch: list[dict] = []
    example: dict[int, dict] = {}

    by_ch = channel_files_by_number(ev)
    for ch in sorted(by_ch):
        raw = load_channel(ev, ch)
        if raw is None:
            continue
        z = zscore_channel_trials(raw)
        st = channel_stats(raw, z)
        st["channel"] = ch
        per_ch.append(st)
        if ch in EXAMPLE_CHANNELS:
            row_ok = ~np.all(np.isnan(raw), axis=1)
            filt = base_mask & row_ok
            example[ch] = {
                **st,
                "filtered_trials": int(filt.sum()),
                "left_trials": int((left_mask & row_ok).sum()),
                "right_trials": int((right_mask & row_ok).sum()),
                "raw_filtered_sd": float(np.std(raw[filt], ddof=0)) if filt.any() else np.nan,
                "z_filtered_max_abs": float(np.max(np.abs(z[filt]))) if filt.any() else np.nan,
            }

    if not per_ch:
        return {"session_id": session_id, "n_channels": 0}

    zero_sd_frac = float(np.mean([s["zero_sd"] for s in per_ch]))
    z_max_abs = [s["z_max_abs"] for s in per_ch]
    raw_sd = [s["raw_sd_pop"] for s in per_ch if np.isfinite(s["raw_sd_pop"])]

    return {
        "session_id": session_id,
        "datetime": session_id.split(".")[0],
        "n_channels": len(per_ch),
        "dyadic_trials": int(base_mask.sum()),
        "left_trials": int(left_mask.sum()),
        "right_trials": int(right_mask.sum()),
        "zero_sd_frac": zero_sd_frac,
        "raw_sd_median": float(np.median(raw_sd)) if raw_sd else np.nan,
        "raw_sd_min": float(np.min(raw_sd)) if raw_sd else np.nan,
        "raw_sd_p10": float(np.percentile(raw_sd, 10)) if raw_sd else np.nan,
        "z_max_abs_median": float(np.median(z_max_abs)),
        "z_max_abs_p95": float(np.percentile(z_max_abs, 95)),
        "z_max_abs_max": float(np.max(z_max_abs)),
        "example_channels": example,
    }


def print_row(label: str, rows: list[dict], key: str, fmt: str = ".4g") -> None:
    vals = [r.get(key, np.nan) for r in rows]
    parts = [f"{label:22s}"] + [f"{v:{fmt}}" if isinstance(v, (int, float)) else str(v) for v in vals]
    print("  ".join(parts))


def main() -> None:
    _, split = load_dual_nhp_configs(REPO_ROOT / "session_lists.m")
    sessions = split["Elmo"]
    filters = trial_filters_for_dual_nhp_monkey("Elmo", session_id=sessions[1])

    print("Elmo DUAL_NHP sessions (B suffix):")
    for i, sid in enumerate(sessions, 1):
        print(f"  {i}. {sid}")

    print(f"\nTrial filters: {filters}\n")

    summaries = [summarize_session(sid, filters) for sid in sessions]
    headers = [s["datetime"] for s in summaries]
    print("Comparison (columns = sessions):")
    print("  " + "  ".join(f"{h:>18s}" for h in headers))
    print_row("dyadic_trials", summaries, "dyadic_trials", "d")
    print_row("left_trials", summaries, "left_trials", "d")
    print_row("right_trials", summaries, "right_trials", "d")
    print_row("zero_sd_frac", summaries, "zero_sd_frac", ".3f")
    print_row("raw_sd_median", summaries, "raw_sd_median", ".4g")
    print_row("raw_sd_min", summaries, "raw_sd_min", ".4g")
    print_row("raw_sd_p10", summaries, "raw_sd_p10", ".4g")
    print_row("z_max_abs_median", summaries, "z_max_abs_median", ".3g")
    print_row("z_max_abs_p95", summaries, "z_max_abs_p95", ".3g")
    print_row("z_max_abs_max", summaries, "z_max_abs_max", ".3g")

    print("\nExample channels (session 2 vs others):")
    for ch in EXAMPLE_CHANNELS:
        print(f"\n  ch{ch:03d}:")
        for s in summaries:
            ex = s.get("example_channels", {}).get(ch)
            if not ex:
                print(f"    {s['datetime']}: missing")
                continue
            print(
                f"    {s['datetime']}: raw_sd={ex['raw_sd_pop']:.4g} "
                f"z_max={ex['z_max_abs']:.3g} filt_trials={ex['filtered_trials']} "
                f"L/R={ex['left_trials']}/{ex['right_trials']}"
            )

    # Outlier trial scan on session 2 ch069
    sid2 = sessions[1]
    raw69 = load_channel(DATA_ROOT / sid2 / ALIGN, 69)
    if raw69 is not None:
        row_ok = ~np.all(np.isnan(raw69), axis=1)
        trial_sd = np.nanstd(raw69[row_ok], axis=1, ddof=0)
        trial_mean = np.nanmean(raw69[row_ok], axis=1)
        top = np.argsort(trial_sd)[-5:][::-1]
        print(f"\nSession 2 ch069 top-5 trials by per-trial SD:")
        for ti in top:
            print(
                f"  trial {ti}: mean={trial_mean[ti]:.4g} sd={trial_sd[ti]:.4g} "
                f"max={np.nanmax(raw69[ti]):.4g}"
            )

    # Also check AgoB filter for comparison
    filters_agob = dict(filters)
    filters_agob["go_seq_500_list"] = ["AgoB"]
    print("\n--- Same stats with AgoB filter (not default for Elmo) ---")
    summaries_agob = [summarize_session(sid, filters_agob) for sid in sessions]
    print_row("dyadic_trials", summaries_agob, "dyadic_trials", "d")
    print_row("z_max_abs_median", summaries_agob, "z_max_abs_median", ".3g")


if __name__ == "__main__":
    main()
