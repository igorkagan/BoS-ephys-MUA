"""Per-session time-resolved decoding with CV SEM + time-locked nulls."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from decodanda import Decodanda
from decodanda.utilities import z_pval
from tqdm import tqdm

from analyze_decoding.config import (
    CROSS_VALIDATIONS,
    MIN_TRIALS_PER_CONDITION,
    NSHUFFLES,
    TRAINING_FRACTION,
)
from analyze_decoding.data import SessionDecodeData
from analyze_decoding.stats_cluster import ClusterTestResult, cluster_p_from_null_curves


@dataclass
class SessionDecodeResult:
    session_id: str
    go_seq: str
    trial_type: str
    target_key: str
    bin_centers_ms: np.ndarray
    perf_mean: np.ndarray
    perf_sem_cv: np.ndarray
    null: np.ndarray  # (nshuffles, n_bins) time-locked
    pvalues: np.ndarray
    n_trials: int
    n_left: int
    n_right: int
    n_channels: int
    alignment_event: str
    cluster_mask: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    cluster_p: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    cluster_threshold: float = np.nan
    balanced: bool = False


def _sem(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size <= 1:
        return 0.0
    return float(np.std(x, ddof=1) / np.sqrt(x.size))


def _bin_masks(t_attr: np.ndarray, centers: np.ndarray) -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    for t0 in centers:
        sel = t_attr == float(t0)
        if not np.any(sel):
            sel = np.isclose(t_attr, float(t0))
        masks.append(sel)
    return masks


def _decode_bin(
    bin_data: dict[str, np.ndarray],
    conditions: dict[str, list[str]],
    *,
    training_fraction: float,
    cross_validations: int,
    min_trials_per_condition: int,
    return_cv: bool,
    verbose: bool,
) -> dict[str, tuple[float, float]]:
    """Return {key: (mean_acc, sem_cv)} for each semantic key in conditions."""
    dec = Decodanda(
        data=bin_data,
        conditions=conditions,
        verbose=verbose,
        min_trials_per_condition=min_trials_per_condition,
        min_data_per_condition=min_trials_per_condition,
    )
    perfs, _null = dec.decode(
        training_fraction=training_fraction,
        cross_validations=cross_validations,
        nshuffles=0,
        return_CV=return_cv,
        plot=False,
        non_semantic=False,
    )
    out: dict[str, tuple[float, float]] = {}
    for key in conditions:
        if key not in perfs:
            continue
        if return_cv:
            cv = np.asarray(perfs[key], dtype=float)
            out[key] = (float(np.nanmean(cv)), _sem(cv))
        else:
            out[key] = (float(perfs[key]), 0.0)
    return out


def _empty_result(
    sess: SessionDecodeData,
    target_key: str,
    n_bins: int,
    nshuffles: int,
    *,
    balanced: bool,
) -> SessionDecodeResult:
    return SessionDecodeResult(
        session_id=sess.session_id,
        go_seq=sess.go_seq,
        trial_type=sess.trial_type,
        target_key=target_key,
        bin_centers_ms=sess.bin_centers_ms,
        perf_mean=np.full(n_bins, np.nan),
        perf_sem_cv=np.full(n_bins, np.nan),
        null=np.full((nshuffles, n_bins), np.nan),
        pvalues=np.full(n_bins, np.nan),
        n_trials=sess.n_trials,
        n_left=sess.n_left,
        n_right=sess.n_right,
        n_channels=sess.n_channels,
        alignment_event=sess.alignment_event,
        cluster_mask=np.zeros(n_bins, dtype=bool),
        cluster_p=np.array([], dtype=float),
        cluster_threshold=np.nan,
        balanced=balanced,
    )


def decode_session_intime(
    sess: SessionDecodeData,
    *,
    target_key: str | None = None,
    training_fraction: float = TRAINING_FRACTION,
    cross_validations: int = CROSS_VALIDATIONS,
    nshuffles: int = NSHUFFLES,
    min_trials_per_condition: int = MIN_TRIALS_PER_CONDITION,
    verbose: bool = False,
    show_progress: bool = True,
    rng_seed: int = 0,
) -> SessionDecodeResult | dict[str, SessionDecodeResult]:
    """Decode one session bin-by-bin with time-locked null label shuffles.

    If ``sess.conditions`` has one key (or ``target_key`` set), returns a single
    ``SessionDecodeResult``. If multiple keys (balanced), returns ``{key: result}``.
    """
    conditions = dict(sess.conditions)
    keys = list(conditions.keys())
    if not keys:
        raise ValueError("sess.conditions is empty")
    if target_key is not None:
        if target_key not in conditions:
            raise ValueError(f"{target_key!r} not in conditions {keys}")
        conditions = {target_key: conditions[target_key]}
        keys = [target_key]

    multi = len(keys) > 1
    balanced = multi
    centers = sess.bin_centers_ms
    n_bins = centers.size

    # min trials: for multi-var Decodanda needs enough per condition cell;
    # gate on n_left/n_right of primary when provided
    if sess.n_left < min_trials_per_condition or sess.n_right < min_trials_per_condition:
        if multi:
            return {k: _empty_result(sess, k, n_bins, nshuffles, balanced=balanced) for k in keys}
        return _empty_result(sess, keys[0], n_bins, nshuffles, balanced=False)

    t_attr = np.asarray(sess.data["time_from_onset"], dtype=float)
    masks = _bin_masks(t_attr, centers)
    trial_ids = np.asarray(sess.data["trial"], dtype=int)
    first = next((m for m in masks if np.any(m)), None)
    if first is None:
        if multi:
            return {k: _empty_result(sess, k, n_bins, nshuffles, balanced=balanced) for k in keys}
        return _empty_result(sess, keys[0], n_bins, nshuffles, balanced=False)

    trial_order = trial_ids[first]
    n_trials = trial_order.size
    # per-key trial labels (length n_trials)
    trial_labels = {k: np.asarray(sess.data[k])[first] for k in keys}

    perf_mean = {k: np.full(n_bins, np.nan) for k in keys}
    perf_sem = {k: np.full(n_bins, np.nan) for k in keys}
    null = {k: np.full((nshuffles, n_bins), np.nan) for k in keys}
    pvalues = {k: np.full(n_bins, np.nan) for k in keys}

    tag = sess.session_id.split(".")[0]
    obs_iter = range(n_bins)
    if show_progress:
        obs_iter = tqdm(obs_iter, total=n_bins, desc=f"obs {tag}", leave=False)
    for i in obs_iter:
        sel = masks[i]
        if not np.any(sel):
            continue
        bin_data = {k: np.asarray(v)[sel] for k, v in sess.data.items()}
        try:
            scores = _decode_bin(
                bin_data,
                conditions,
                training_fraction=training_fraction,
                cross_validations=cross_validations,
                min_trials_per_condition=min_trials_per_condition,
                return_cv=True,
                verbose=verbose,
            )
        except Exception as exc:
            if verbose:
                print(f"  bin {centers[i]:.0f} ms skipped: {exc}")
            continue
        for k, (m, s) in scores.items():
            perf_mean[k][i] = m
            perf_sem[k][i] = s

    rng = np.random.default_rng(rng_seed)
    shuf_iter = range(nshuffles)
    if show_progress:
        shuf_iter = tqdm(shuf_iter, total=nshuffles, desc=f"null {tag}", leave=False)
    for k_shuf in shuf_iter:
        # joint permutation of trials → all label keys (preserves A–B correlation)
        perm = rng.permutation(n_trials)
        shuf_trial_labels = {k: trial_labels[k][perm] for k in keys}
        for i in range(n_bins):
            sel = masks[i]
            if not np.any(sel):
                continue
            bin_data = {key: np.asarray(val)[sel] for key, val in sess.data.items()}
            tid = trial_ids[sel]
            for key in keys:
                if tid.size != n_trials or not np.array_equal(tid, trial_order):
                    label_map = {
                        int(t): shuf_trial_labels[key][j]
                        for j, t in enumerate(trial_order)
                    }
                    bin_data[key] = np.array([label_map[int(t)] for t in tid], dtype=object)
                else:
                    bin_data[key] = shuf_trial_labels[key]
            try:
                scores = _decode_bin(
                    bin_data,
                    conditions,
                    training_fraction=training_fraction,
                    cross_validations=cross_validations,
                    min_trials_per_condition=min_trials_per_condition,
                    return_cv=False,
                    verbose=False,
                )
            except Exception:
                continue
            for key, (m, _) in scores.items():
                null[key][k_shuf, i] = m

    results: dict[str, SessionDecodeResult] = {}
    for key in keys:
        for i in range(n_bins):
            if np.isfinite(perf_mean[key][i]) and np.any(np.isfinite(null[key][:, i])):
                _, p = z_pval(perf_mean[key][i], null[key][:, i])
                pvalues[key][i] = float(p)
        cluster: ClusterTestResult = cluster_p_from_null_curves(
            perf_mean[key], null[key], chance=0.5,
        )
        # class counts for this key
        levels = conditions[key]
        lab0 = trial_labels[key]
        n0 = int(np.sum(lab0 == levels[0]))
        n1 = int(np.sum(lab0 == levels[1]))
        results[key] = SessionDecodeResult(
            session_id=sess.session_id,
            go_seq=sess.go_seq,
            trial_type=sess.trial_type,
            target_key=key,
            bin_centers_ms=centers,
            perf_mean=perf_mean[key],
            perf_sem_cv=perf_sem[key],
            null=null[key],
            pvalues=pvalues[key],
            n_trials=sess.n_trials,
            n_left=n0,
            n_right=n1,
            n_channels=sess.n_channels,
            alignment_event=sess.alignment_event,
            cluster_mask=cluster.mask,
            cluster_p=np.array([c.p_value for c in cluster.clusters], dtype=float),
            cluster_threshold=cluster.threshold,
            balanced=balanced,
        )

    if multi:
        return results
    return results[keys[0]]
