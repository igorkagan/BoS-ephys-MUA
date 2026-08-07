"""Audit same/diff imbalance and compare Decodanda vs naive unbalanced decode."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

from analyze_decoding.config import (
    BIN_STEP_MS,
    BIN_WIDTH_MS,
    DECODE_WINDOW_MS,
    GAUSSIAN_SMOOTH_MS,
    TRAINING_FRACTION,
    default_curated_data_root,
)
from analyze_decoding.data import (
    build_aligned_multilabel_bundle,
    subset_for_conditions,
)
from analyze_decoding.session_decode import decode_session_intime
from analyze_decoding.targets import B_EVENT, align_events_in_order
from run_pipeline.curated import discover_curated_sessions


def audit_session_counts(
    condition: str = "Elmo_BLOCKED",
    go_seq: str = "AgoB",
    *,
    data_root: Path | None = None,
) -> None:
    root = data_root or default_curated_data_root()
    ids = discover_curated_sessions(condition, data_root=root)
    print(f"=== same/diff counts {condition} {go_seq} Dyadic ===")
    print(f"{'session':<22} {'same':>5} {'diff':>5} {'ratio':>7} {'N_eff':>6}")
    for sid in ids:
        short = sid.split(".")[0]
        try:
            bundle = build_aligned_multilabel_bundle(
                sid,
                condition=condition,
                go_seq=go_seq,
                trial_type="Dyadic",
                data_root=root,
                alignment_event=B_EVENT,
                window_ms=DECODE_WINDOW_MS,
                bin_width_ms=BIN_WIDTH_MS,
                bin_step_ms=BIN_STEP_MS,
                smooth_ms=GAUSSIAN_SMOOTH_MS,
            )
            sess = subset_for_conditions(
                bundle, {"same_diff": ["same", "diff"]}, count_key="same_diff",
            )
        except Exception as exc:
            print(f"{short:<22} SKIP {exc}")
            continue
        n_same, n_diff = sess.n_left, sess.n_right
        ratio = n_same / max(n_diff, 1)
        n_eff = 2 * min(n_same, n_diff)
        print(f"{short:<22} {n_same:5d} {n_diff:5d} {ratio:7.2f} {n_eff:6d}")


def _bin_matrix(sess, t0: float) -> tuple[np.ndarray, np.ndarray]:
    """Return X (n_trials, n_ch), y (0/1) for one time bin center."""
    t_attr = np.asarray(sess.data["time_from_onset"], dtype=float)
    sel = np.isclose(t_attr, float(t0)) | (t_attr == float(t0))
    if not np.any(sel):
        # nearest center
        centers = sess.bin_centers_ms
        t0 = float(centers[np.argmin(np.abs(centers - t0))])
        sel = np.isclose(t_attr, t0)
    X = np.asarray(sess.data["raster"])[sel]
    lab = np.asarray(sess.data["same_diff"])[sel]
    y = (lab == "diff").astype(int)
    return X, y


def naive_vs_decodanda(
    session_id: str,
    *,
    condition: str = "Elmo_BLOCKED",
    go_seq: str = "AgoB",
    data_root: Path | None = None,
    t_query_ms: float = 150.0,
    cv: int = 20,
) -> None:
    root = data_root or default_curated_data_root()
    event = align_events_in_order(go_seq)[1]  # 2nd action = B in AgoB
    print(f"\n=== naive vs Decodanda {session_id.split('.')[0]} align={event} ===")
    bundle = build_aligned_multilabel_bundle(
        session_id,
        condition=condition,
        go_seq=go_seq,
        trial_type="Dyadic",
        data_root=root,
        alignment_event=event,
        bin_width_ms=BIN_WIDTH_MS,
        bin_step_ms=BIN_STEP_MS,
        smooth_ms=GAUSSIAN_SMOOTH_MS,
    )
    sess = subset_for_conditions(
        bundle, {"same_diff": ["same", "diff"]}, count_key="same_diff",
    )
    print(f"trials={sess.n_trials} same/diff={sess.n_left}/{sess.n_right}")
    majority = max(sess.n_left, sess.n_right) / sess.n_trials
    print(f"majority-class floor ≈ {majority:.3f}")

    # pick nearest bin to t_query
    t0 = float(sess.bin_centers_ms[np.argmin(np.abs(sess.bin_centers_ms - t_query_ms))])
    X, y = _bin_matrix(sess, t0)
    print(f"bin center={t0:.1f} ms  X={X.shape}")

    skf = StratifiedKFold(n_splits=min(cv, int(y.sum()), int((y == 0).sum())), shuffle=True, random_state=0)
    accs, bal_accs = [], []
    for train_idx, test_idx in skf.split(X, y):
        clf = LogisticRegression(max_iter=1000, solver="lbfgs")
        clf.fit(X[train_idx], y[train_idx])
        pred = clf.predict(X[test_idx])
        accs.append(accuracy_score(y[test_idx], pred))
        bal_accs.append(balanced_accuracy_score(y[test_idx], pred))
    print(f"naive accuracy (unbalanced)  mean={np.mean(accs):.3f} ± {np.std(accs):.3f}")
    print(f"naive balanced_accuracy      mean={np.mean(bal_accs):.3f} ± {np.std(bal_accs):.3f}")

    # Decodanda single-bin via full curve (reuse pipeline; report peak near t0)
    res = decode_session_intime(
        sess,
        target_key="same_diff",
        training_fraction=TRAINING_FRACTION,
        cross_validations=cv,
        nshuffles=5,
        show_progress=False,
    )
    i = int(np.argmin(np.abs(res.bin_centers_ms - t0)))
    print(
        f"Decodanda @ {res.bin_centers_ms[i]:.1f} ms  "
        f"acc={res.perf_mean[i]:.3f}  peak={np.nanmax(res.perf_mean):.3f}"
    )


def main() -> None:
    root = default_curated_data_root()
    audit_session_counts(data_root=root)
    ids = discover_curated_sessions("Elmo_BLOCKED", data_root=root)
    for sid in ids[:2]:
        naive_vs_decodanda(sid, data_root=root)


if __name__ == "__main__":
    main()
