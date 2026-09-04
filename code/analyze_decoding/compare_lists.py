"""Compare decoding combined caches across confederate lists (no re-decode)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from analyze_decoding.combine_decode import CombinedDecodeResult
from analyze_decoding.io_cache import load_combined_result
from analyze_decoding.stats_cluster import cluster_p_two_sample, welch_t_curves
from load_data.sessions import load_session_list
from process_channels.preprocess import DUAL_NHP_GO_SEQS
from run_pipeline.dual_nhp import dual_nhp_run_label

ANALYSES = ("same_diff", "actor_own")
LIST_TAGS = ("BLOCKED", "SHUFFLED")
# Confederate *_CONF lists: recorded monkey is side A.
OWN_ACTOR_SIDE = "A"


def list_name(monkey: str, blocked: bool) -> str:
    tag = "BLOCKED" if blocked else "SHUFFLED"
    return f"{monkey}_{tag}_CONF"


def same_diff_combined_npz(data_root: Path, monkey: str, go_seq: str, *, blocked: bool) -> Path:
    lst = list_name(monkey, blocked)
    branch = dual_nhp_run_label(monkey, go_seq)
    return (
        data_root
        / lst
        / branch
        / "Dyadic"
        / "decoding"
        / "same_diff"
        / "combined"
        / "mean_ci_decode.npz"
    )


def compare_output_dir(
    data_root: Path,
    monkey: str,
    go_seq: str,
    analysis: str,
) -> Path:
    return (
        data_root
        / "decode_compare"
        / f"{monkey}_BLOCKED_vs_SHUFFLED"
        / dual_nhp_run_label(monkey, go_seq)
        / "Dyadic"
        / "decoding"
        / analysis
    )


def load_same_diff_combined(path: Path) -> dict[str, CombinedDecodeResult]:
    """Rebuild per-alignment CombinedDecodeResult from same_diff combined npz."""
    z = np.load(path, allow_pickle=True)
    panels: dict[str, CombinedDecodeResult] = {}
    for event in (str(z["align_0"]), str(z["align_1"])):
        pfx = "A" if event.startswith("A_") else "B"
        panels[event] = combined_from_prefix(z, pfx)
    return panels


def _curves_2d(z, key: str, n_bins: int) -> np.ndarray:
    curves = np.asarray(z[key], dtype=float)
    if curves.size == 0:
        return np.empty((0, n_bins))
    if curves.ndim == 1:
        return curves.reshape(1, -1)
    return curves


def combined_from_prefix(z, pfx: str) -> CombinedDecodeResult:
    t = np.asarray(z[f"{pfx}_bin_centers_ms"], dtype=float)
    curves = _curves_2d(z, f"{pfx}_session_curves", t.size)
    mask = (
        np.asarray(z[f"{pfx}_cluster_mask"], dtype=bool)
        if f"{pfx}_cluster_mask" in z.files
        else np.zeros(t.shape, dtype=bool)
    )
    return CombinedDecodeResult(
        bin_centers_ms=t,
        mean=np.asarray(z[f"{pfx}_mean"], dtype=float),
        ci_low=np.asarray(z[f"{pfx}_ci_low"], dtype=float),
        ci_high=np.asarray(z[f"{pfx}_ci_high"], dtype=float),
        session_curves=curves,
        session_ids=[],
        n_sessions_used=int(z[f"{pfx}_n_sessions"]),
        cluster_mask=mask,
    )


def _require_matching_centers(
    a: CombinedDecodeResult,
    b: CombinedDecodeResult,
    *,
    what: str = "combined caches",
) -> np.ndarray:
    t_a = a.bin_centers_ms
    t_b = b.bin_centers_ms
    if t_a.shape != t_b.shape or not np.allclose(t_a, t_b, equal_nan=True):
        raise ValueError(f"bin centers mismatch between {what}")
    return t_a


def compare_two_combined(
    group_a: CombinedDecodeResult,
    group_b: CombinedDecodeResult,
    *,
    n_perm: int = 5000,
    cluster_forming_p: float = 0.05,
    seed: int = 0,
    what: str = "groups",
) -> dict:
    t = _require_matching_centers(group_a, group_b, what=what)
    t_stat = welch_t_curves(group_a.session_curves, group_b.session_curves)
    cluster = cluster_p_two_sample(
        group_a.session_curves,
        group_b.session_curves,
        n_perm=n_perm,
        cluster_forming_p=cluster_forming_p,
        seed=seed,
    )
    return {
        "bin_centers_ms": t,
        "t_welch": t_stat,
        "cluster": cluster,
    }


def compare_same_diff_panels(
    blocked: dict[str, CombinedDecodeResult],
    shuffled: dict[str, CombinedDecodeResult],
    *,
    n_perm: int = 5000,
    cluster_forming_p: float = 0.05,
    seed: int = 0,
) -> dict[str, dict]:
    """Per-alignment two-sample cluster test. Keys = align events."""
    events = list(blocked)
    missing = set(events) ^ set(shuffled)
    if missing:
        raise ValueError(f"alignment mismatch: {sorted(missing)}")
    out: dict[str, dict] = {}
    for event in events:
        blk = blocked[event]
        shf = shuffled[event]
        cmp = compare_two_combined(
            blk,
            shf,
            n_perm=n_perm,
            cluster_forming_p=cluster_forming_p,
            seed=seed,
            what="BLOCKED and SHUFFLED combined caches",
        )
        out[event] = {**cmp, "blocked": blk, "shuffled": shf}
    return out


def save_same_diff_compare(path: Path, results: dict[str, dict], *, go_seq: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    events = list(results)
    payload: dict = {
        "go_seq": go_seq,
        "align_0": events[0],
        "align_1": events[1] if len(events) > 1 else events[0],
    }
    for event, res in results.items():
        pfx = "A" if event.startswith("A_") else "B"
        cluster = res["cluster"]
        blk = res["blocked"]
        shf = res["shuffled"]
        payload[f"{pfx}_bin_centers_ms"] = res["bin_centers_ms"]
        payload[f"{pfx}_t_welch"] = res["t_welch"]
        payload[f"{pfx}_cluster_mask"] = cluster.mask
        payload[f"{pfx}_cluster_threshold"] = (
            float(cluster.threshold) if np.isfinite(cluster.threshold) else np.nan
        )
        payload[f"{pfx}_cluster_p"] = np.array(
            [c.p_value for c in cluster.clusters], dtype=float
        )
        payload[f"{pfx}_cluster_start"] = np.array(
            [c.start for c in cluster.clusters], dtype=int
        )
        payload[f"{pfx}_cluster_stop"] = np.array(
            [c.stop for c in cluster.clusters], dtype=int
        )
        payload[f"{pfx}_blocked_mean"] = blk.mean
        payload[f"{pfx}_blocked_ci_low"] = blk.ci_low
        payload[f"{pfx}_blocked_ci_high"] = blk.ci_high
        payload[f"{pfx}_blocked_session_curves"] = blk.session_curves
        payload[f"{pfx}_blocked_n_sessions"] = blk.n_sessions_used
        payload[f"{pfx}_shuffled_mean"] = shf.mean
        payload[f"{pfx}_shuffled_ci_low"] = shf.ci_low
        payload[f"{pfx}_shuffled_ci_high"] = shf.ci_high
        payload[f"{pfx}_shuffled_session_curves"] = shf.session_curves
        payload[f"{pfx}_shuffled_n_sessions"] = shf.n_sessions_used
    np.savez_compressed(path, **payload)


def run_same_diff_compare(
    monkey: str,
    go_seq: str,
    *,
    data_root: Path,
    n_perm: int = 5000,
    cluster_forming_p: float = 0.05,
    seed: int = 0,
) -> Path:
    from analyze_decoding.plots_grid import plot_same_diff_compare

    blk_path = same_diff_combined_npz(data_root, monkey, go_seq, blocked=True)
    shf_path = same_diff_combined_npz(data_root, monkey, go_seq, blocked=False)
    if not blk_path.exists():
        raise FileNotFoundError(f"missing BLOCKED combined: {blk_path}")
    if not shf_path.exists():
        raise FileNotFoundError(f"missing SHUFFLED combined: {shf_path}")

    blocked = load_same_diff_combined(blk_path)
    shuffled = load_same_diff_combined(shf_path)
    results = compare_same_diff_panels(
        blocked,
        shuffled,
        n_perm=n_perm,
        cluster_forming_p=cluster_forming_p,
        seed=seed,
    )
    out_dir = compare_output_dir(data_root, monkey, go_seq, "same_diff")
    out_dir.mkdir(parents=True, exist_ok=True)
    save_same_diff_compare(out_dir / "mean_ci_compare.npz", results, go_seq=go_seq)
    plot_same_diff_compare(
        results,
        out_pdf=out_dir / "mean_ci_compare.pdf",
        monkey=monkey,
        go_seq=go_seq,
    )
    for event, res in results.items():
        n_sig = int(np.sum(res["cluster"].mask)) if res["cluster"].mask.size else 0
        n_cl = len(res["cluster"].clusters)
        print(
            f"  {event}: n_blocked={res['blocked'].n_sessions_used} "
            f"n_shuffled={res['shuffled'].n_sessions_used} "
            f"clusters={n_cl} sig_bins={n_sig}"
        )
    return out_dir


def own_action_prefix(actor_side: str = OWN_ACTOR_SIDE) -> str:
    key = "A_choice" if actor_side == "A" else "B_choice"
    return f"{actor_side}__{key}"


def actor_own_dyadic_npz(
    data_root: Path,
    monkey: str,
    go_seq: str,
    *,
    blocked: bool,
) -> Path:
    lst = list_name(monkey, blocked)
    return (
        data_root
        / lst
        / dual_nhp_run_label(monkey, go_seq)
        / "Dyadic"
        / "decoding"
        / "choice_ab_grid"
        / "combined"
        / "mean_ci_decode.npz"
    )


def actor_own_solo_npz(
    data_root: Path,
    monkey: str,
    go_seq: str,
    *,
    blocked: bool,
) -> Path:
    lst = list_name(monkey, blocked)
    return (
        data_root
        / lst
        / dual_nhp_run_label(monkey, go_seq)
        / "SoloA"
        / "decoding"
        / "actor_choice"
        / "combined"
        / "mean_ci_decode.npz"
    )


def actor_own_output_dir(
    data_root: Path,
    monkey: str,
    go_seq: str,
    *,
    blocked: bool,
) -> Path:
    tag = "BLOCKED" if blocked else "SHUFFLED"
    return (
        data_root
        / "decode_compare"
        / f"{monkey}_{tag}_Dyadic_vs_SoloA"
        / dual_nhp_run_label(monkey, go_seq)
        / "decoding"
        / "actor_choice"
    )


def load_choice_ab_own_action(
    path: Path,
    *,
    actor_side: str = OWN_ACTOR_SIDE,
) -> CombinedDecodeResult:
    """Own-action panel: recorded monkey's L/R at own release (unbalanced choice_ab)."""
    z = np.load(path, allow_pickle=True)
    return combined_from_prefix(z, own_action_prefix(actor_side))


def save_actor_own_compare(path: Path, result: dict, *, go_seq: str, list_tag: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cluster = result["cluster"]
    dy = result["dyadic"]
    so = result["solo"]
    payload = {
        "go_seq": go_seq,
        "list_tag": list_tag,
        "target": "actor_choice",
        "bin_centers_ms": result["bin_centers_ms"],
        "t_welch": result["t_welch"],
        "cluster_mask": cluster.mask,
        "cluster_threshold": (
            float(cluster.threshold) if np.isfinite(cluster.threshold) else np.nan
        ),
        "cluster_p": np.array([c.p_value for c in cluster.clusters], dtype=float),
        "cluster_start": np.array([c.start for c in cluster.clusters], dtype=int),
        "cluster_stop": np.array([c.stop for c in cluster.clusters], dtype=int),
        "dyadic_mean": dy.mean,
        "dyadic_ci_low": dy.ci_low,
        "dyadic_ci_high": dy.ci_high,
        "dyadic_session_curves": dy.session_curves,
        "dyadic_n_sessions": dy.n_sessions_used,
        "solo_mean": so.mean,
        "solo_ci_low": so.ci_low,
        "solo_ci_high": so.ci_high,
        "solo_session_curves": so.session_curves,
        "solo_n_sessions": so.n_sessions_used,
        "solo_session_ids": np.array(so.session_ids, dtype=object),
    }
    np.savez_compressed(path, **payload)


def run_actor_own_compare(
    monkey: str,
    go_seq: str,
    *,
    blocked: bool,
    data_root: Path,
    n_perm: int = 5000,
    cluster_forming_p: float = 0.05,
    seed: int = 0,
) -> Path:
    from analyze_decoding.plots_grid import plot_actor_own_compare

    list_tag = "BLOCKED" if blocked else "SHUFFLED"
    dy_path = actor_own_dyadic_npz(data_root, monkey, go_seq, blocked=blocked)
    so_path = actor_own_solo_npz(data_root, monkey, go_seq, blocked=blocked)
    if not dy_path.exists():
        raise FileNotFoundError(f"missing Dyadic own-action combined: {dy_path}")
    if not so_path.exists():
        raise FileNotFoundError(f"missing SoloA actor_choice combined: {so_path}")

    dyadic = load_choice_ab_own_action(dy_path)
    solo = load_combined_result(so_path)
    cmp = compare_two_combined(
        dyadic,
        solo,
        n_perm=n_perm,
        cluster_forming_p=cluster_forming_p,
        seed=seed,
        what="Dyadic and SoloA combined caches",
    )
    result = {**cmp, "dyadic": dyadic, "solo": solo}
    out_dir = actor_own_output_dir(data_root, monkey, go_seq, blocked=blocked)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_actor_own_compare(
        out_dir / "mean_ci_compare.npz", result, go_seq=go_seq, list_tag=list_tag,
    )
    plot_actor_own_compare(
        result,
        out_pdf=out_dir / "mean_ci_compare.pdf",
        monkey=monkey,
        go_seq=go_seq,
        list_tag=list_tag,
    )
    n_sig = int(np.sum(result["cluster"].mask)) if result["cluster"].mask.size else 0
    print(
        f"  n_dyadic={dyadic.n_sessions_used} n_solo={solo.n_sessions_used} "
        f"clusters={len(result['cluster'].clusters)} sig_bins={n_sig}"
    )
    return out_dir


def run_decode_compare(
    *,
    monkeys: tuple[str, ...] = ("Elmo", "Curius"),
    go_seq: str = "AgoB",
    analysis: str = "same_diff",
    list_tags: tuple[str, ...] = LIST_TAGS,
    data_root: Path | None = None,
    session_lists: Path | str = "session_lists.m",
    n_perm: int = 5000,
    cluster_forming_p: float = 0.05,
    seed: int = 0,
) -> list[Path]:
    if analysis not in ANALYSES:
        raise ValueError(f"unknown analysis {analysis!r}; valid: {ANALYSES}")
    seqs = DUAL_NHP_GO_SEQS if go_seq == "all" else (go_seq,)
    for seq in seqs:
        if seq not in DUAL_NHP_GO_SEQS:
            raise ValueError(f"unknown go_seq {seq!r}")
    bad_lists = [t for t in list_tags if t not in LIST_TAGS]
    if bad_lists:
        raise ValueError(f"unknown list tags {bad_lists}; valid: {LIST_TAGS}")
    if data_root is None:
        cfg = load_session_list(list_name(monkeys[0], True), Path(session_lists))
        data_root = Path(cfg.root_folder)
    written: list[Path] = []
    for monkey in monkeys:
        for seq in seqs:
            if analysis == "same_diff":
                print(f"[compare] {monkey} {seq} {analysis} BLOCKED vs SHUFFLED")
                written.append(
                    run_same_diff_compare(
                        monkey,
                        seq,
                        data_root=data_root,
                        n_perm=n_perm,
                        cluster_forming_p=cluster_forming_p,
                        seed=seed,
                    )
                )
            elif analysis == "actor_own":
                for tag in list_tags:
                    blocked = tag == "BLOCKED"
                    print(
                        f"[compare] {monkey} {seq} actor_own "
                        f"{tag} Dyadic vs SoloA"
                    )
                    written.append(
                        run_actor_own_compare(
                            monkey,
                            seq,
                            blocked=blocked,
                            data_root=data_root,
                            n_perm=n_perm,
                            cluster_forming_p=cluster_forming_p,
                            seed=seed,
                        )
                    )
    return written
