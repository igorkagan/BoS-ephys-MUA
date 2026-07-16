"""Cross-session waveform stability vs calendar gap (post-consistency analysis)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from process_channels.features import ChannelSummary
from load_data.cache import load_summaries_disk_cache
from load_data.io import session_gap_days
from analyze_stability.metrics import session_similarity_matrix
from analyze_stability.temporal_core import (
    compute_stabilities_from_tensors,
    greedy_session_clusters,
    invert_tradeoff,
    pairwise_channel_rows,
    per_channel_max_gap_at_r,
    pick_sweet_spot,
    rank_channels_for_tradeoff,
    recommend_combinable_gap,
    sweep_channel_session_tradeoff,
    write_best_subset_csv,
    write_channel_max_gap_csv,
    write_gap_bin_csv,
    write_pair_rows_csv,
    write_session_clusters_csv,
    write_tradeoff_csv,
    pair_r_lookup,
)
from analyze_stability.plots_tradeoff import (
    build_explorer_figure,
    plot_retention_heatmap,
    plot_tradeoff_pdf,
    write_explorer_html,
)
from analyze_stability.tensors import build_tensors
from analyze_stability.plots_temporal import (
    plot_gap_bin_summary,
    plot_r_vs_gap_scatter,
    plot_session_gap_heatmap,
)
from process_channels.preprocess import resolve_condition_output_dir
from run_pipeline.context import PipelineContext

MIN_SESSIONS = 3
DEFAULT_MEDIAN_R_THRESHOLD = 0.5
DEFAULT_P25_R_THRESHOLD = 0.3
DEFAULT_PAIR_R_FOR_CHANNEL = 0.5


def default_out_dir(output_base: Path) -> Path:
    return resolve_condition_output_dir(
        output_base, True, "", "stability_across_sessions",
    )


def run_stability_across_sessions(
    ctx: PipelineContext,
    session_ids: list[str],
    summaries: list[ChannelSummary],
    *,
    out_dir: Path | None = None,
    median_r_threshold: float = DEFAULT_MEDIAN_R_THRESHOLD,
    p25_r_threshold: float = DEFAULT_P25_R_THRESHOLD,
    pair_r_for_channel: float = DEFAULT_PAIR_R_FOR_CHANNEL,
) -> Path:
    """Compute gap-vs-stability metrics and write CSVs/PDFs/HTML."""
    present = {s.session_id for s in summaries}
    ordered = [sid for sid in session_ids if sid in present]
    if len(ordered) < MIN_SESSIONS:
        raise ValueError(
            f"Need >={MIN_SESSIONS} sessions with summaries, found {len(ordered)}"
        )

    channels, si_matrix, _p_matrix, diff_tensor, _t_ms = build_tensors(summaries, ordered)
    stabilities = compute_stabilities_from_tensors(
        ordered, channels, diff_tensor, si_matrix, summaries,
    )
    stable_channels = {s.channel for s in stabilities if s.stable}

    pair_rows = pairwise_channel_rows(ordered, channels, diff_tensor, si_matrix)
    rec = recommend_combinable_gap(
        pair_rows,
        stable_channels,
        median_r_threshold=median_r_threshold,
        p25_r_threshold=p25_r_threshold,
    )
    ch_max_gap = per_channel_max_gap_at_r(
        pair_rows, stable_channels, r_threshold=pair_r_for_channel,
    )
    conservative_gap = min(rec.max_gap_days_median_r, rec.max_gap_days_p25_r)
    if conservative_gap <= 0:
        conservative_gap = 7.0
    clusters = greedy_session_clusters(ordered, max_gap_days=conservative_gap)

    dest = out_dir if out_dir is not None else default_out_dir(ctx.output_base)
    dest.mkdir(parents=True, exist_ok=True)

    write_pair_rows_csv(pair_rows, dest / "pairwise_channel_gap.csv")
    write_gap_bin_csv(list(rec.gap_bin_summaries), dest / "gap_bin_summary.csv")
    write_channel_max_gap_csv(ch_max_gap, dest / "stable_channel_max_gap.csv")
    write_session_clusters_csv(clusters, dest / "recommended_session_clusters.csv")

    sim = session_similarity_matrix(diff_tensor)
    n = len(ordered)
    gap_mat = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            gap_mat[i, j] = session_gap_days(ordered[i], ordered[j])

    title = f"{ctx.condition_label} | stability across sessions (stable n={len(stable_channels)})"
    plot_r_vs_gap_scatter(pair_rows, stable_channels, title, dest / "r_vs_gap_scatter.pdf")
    plot_gap_bin_summary(
        list(rec.gap_bin_summaries), title, dest / "gap_bin_median_r.pdf",
    )
    plot_session_gap_heatmap(
        ordered, sim, gap_mat, title, dest / "session_similarity_and_gap.pdf",
    )

    ranked = rank_channels_for_tradeoff(stabilities)
    tradeoff_rows = sweep_channel_session_tradeoff(
        ordered, ranked, pair_rows, r_min=pair_r_for_channel,
    )
    inverse = invert_tradeoff(tradeoff_rows)
    sweet = pick_sweet_spot(tradeoff_rows)

    write_tradeoff_csv(tradeoff_rows, dest / "channel_session_tradeoff.csv")
    if sweet is not None:
        write_best_subset_csv(sweet, dest / "best_combine_subset.csv")
        plot_retention_heatmap(
            sweet,
            pair_r_lookup(pair_rows),
            title,
            dest / "subset_retention_heatmap.pdf",
        )

    tradeoff_title = f"{ctx.condition_label} | channel-session tradeoff"
    explorer_fig = build_explorer_figure(tradeoff_rows, sweet, inverse, title=tradeoff_title)
    write_explorer_html(explorer_fig, dest / "channel_session_explorer.html")
    plot_tradeoff_pdf(tradeoff_rows, sweet, tradeoff_title, dest / "channel_session_tradeoff.pdf")

    print(f"Sessions: {len(ordered)} | stable channels: {len(stable_channels)}")
    print(f"Pair rows: {len(pair_rows)} | stable-channel pair rows: {rec.n_pair_rows_used}")
    print(rec.notes)
    print(f"Greedy clusters (gap <= {conservative_gap:.0f} d): {len(clusters)}")
    for cid, cluster in enumerate(clusters, start=1):
        ids = ", ".join(s.split(".")[0] for s in cluster)
        print(f"  cluster {cid} ({len(cluster)} sessions): {ids}")
    print(f"Outputs: {dest}")
    if sweet is not None:
        print(
            f"Sweet spot: {sweet.n_channels} ch x {sweet.max_n_sessions} sess "
            f"(coverage={sweet.coverage}, min_r={sweet.min_pair_r:.3f})"
        )
    print(f"Interactive explorer: {dest / 'channel_session_explorer.html'}")
    return dest


def summaries_for_run(
    ctx: PipelineContext,
    active_summaries: dict[bool, list[ChannelSummary]] | None,
) -> list[ChannelSummary]:
    """Z-scored summaries from in-memory cache or disk."""
    if active_summaries:
        cached = active_summaries.get(True) or []
        if cached:
            return cached
    return load_summaries_disk_cache(ctx.output_base, ctx.condition_label, zscore_mua=True)
