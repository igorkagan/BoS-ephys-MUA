"""Cross-session stability as a function of calendar time between sessions."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

from load_data.io import parse_session_datetime, session_gap_days
from analyze_stability.metrics import (
    ChannelStability,
    assess_channel_stability,
    channel_task_evoked_all_sessions,
)


def _summaries_lookup(all_summaries: list) -> dict[str, dict[int, object]]:
    out: dict[str, dict[int, object]] = {}
    for summary in all_summaries:
        out.setdefault(summary.session_id, {})[summary.channel] = summary
    return out

# Default gates (match consistency.py)
DEFAULT_R_STABLE = 0.5
DEFAULT_ICC_STABLE = 0.4
DEFAULT_SIGN_STABLE = 0.7
DEFAULT_PAIR_R_COMBINE = 0.5
DEFAULT_PAIR_R_COMBINE_P25 = 0.3

GAP_BIN_EDGES_DAYS = (0.0, 1.0, 3.0, 7.0, 14.0, 30.0, 60.0, np.inf)
GAP_BIN_LABELS = (
    "0",
    "0-1d",
    "1-3d",
    "3-7d",
    "7-14d",
    "14-30d",
    "30-60d",
    "60d+",
)


@dataclass(frozen=True)
class ChannelPairRow:
    session_i: str
    session_j: str
    gap_days: float
    channel: int
    array_name: str
    waveform_r: float
    abs_delta_si: float
    sign_match: bool


@dataclass(frozen=True)
class GapBinSummary:
    bin_label: str
    gap_min_days: float
    gap_max_days: float
    n_pairs: int
    n_channels: int
    median_waveform_r: float
    p25_waveform_r: float
    median_abs_delta_si: float
    fraction_sign_match: float


@dataclass(frozen=True)
class TemporalRecommendation:
    max_gap_days_median_r: float
    max_gap_days_p25_r: float
    median_r_threshold: float
    p25_r_threshold: float
    n_stable_channels: int
    n_pair_rows_used: int
    gap_bin_summaries: tuple[GapBinSummary, ...]
    notes: str


def _array_name_for_channel(channel: int) -> str:
    from load_data.io import channel_to_array

    name, _ = channel_to_array(channel)
    return name


def pairwise_channel_rows(
    session_ids: list[str],
    channels: list[int],
    diff_tensor: np.ndarray,
    si_matrix: np.ndarray,
) -> list[ChannelPairRow]:
    """One row per (session pair, channel) with waveform r and |ΔSI|."""
    rows: list[ChannelPairRow] = []
    n_sess = len(session_ids)
    for i in range(n_sess):
        for j in range(i + 1, n_sess):
            gap = session_gap_days(session_ids[i], session_ids[j])
            for ci, ch in enumerate(channels):
                a = diff_tensor[i, ci]
                b = diff_tensor[j, ci]
                si_a = si_matrix[i, ci]
                si_b = si_matrix[j, ci]
                if not np.any(np.isfinite(a)) or not np.any(np.isfinite(b)):
                    continue
                mask = np.isfinite(a) & np.isfinite(b)
                if mask.sum() < 3:
                    continue
                r, _ = pearsonr(a[mask], b[mask])
                if not np.isfinite(r):
                    continue
                abs_d_si = np.nan
                sign_match = False
                if np.isfinite(si_a) and np.isfinite(si_b):
                    abs_d_si = float(abs(si_a - si_b))
                    sign_match = (si_a > 0) == (si_b > 0) if si_a != 0 and si_b != 0 else si_a == si_b
                rows.append(
                    ChannelPairRow(
                        session_i=session_ids[i],
                        session_j=session_ids[j],
                        gap_days=gap,
                        channel=ch,
                        array_name=_array_name_for_channel(ch),
                        waveform_r=float(r),
                        abs_delta_si=abs_d_si,
                        sign_match=sign_match,
                    ),
                )
    return rows


def compute_stabilities_from_tensors(
    session_ids: list[str],
    channels: list[int],
    diff_tensor: np.ndarray,
    si_matrix: np.ndarray,
    all_summaries: list,
    *,
    r_thresh: float = DEFAULT_R_STABLE,
    icc_thresh: float = DEFAULT_ICC_STABLE,
    sign_thresh: float = DEFAULT_SIGN_STABLE,
) -> list[ChannelStability]:
    """Per-channel stability (same gates as consistency step)."""
    lookup = _summaries_lookup(all_summaries)
    stabilities: list[ChannelStability] = []
    for ci, ch in enumerate(channels):
        traces = diff_tensor[:, ci, :]
        valid = np.any(np.isfinite(traces), axis=1)
        if int(np.sum(valid)) < 3:
            continue
        task_evoked, n_evoked, _ = channel_task_evoked_all_sessions(lookup, session_ids, ch)
        stabilities.append(
            assess_channel_stability(
                ch,
                _array_name_for_channel(ch),
                traces[valid],
                si_matrix[valid, ci],
                r_thresh,
                icc_thresh,
                sign_thresh,
                task_evoked=task_evoked,
                n_sessions_task_evoked=n_evoked,
            ),
        )
    return stabilities


def _gap_bin_index(gap_days: float) -> int:
    for idx, edge in enumerate(GAP_BIN_EDGES_DAYS[1:]):
        if gap_days <= edge:
            return idx
    return len(GAP_BIN_LABELS) - 1


def summarize_pairs_by_gap_bin(
    rows: list[ChannelPairRow],
) -> list[GapBinSummary]:
    """Aggregate pair metrics into calendar-gap bins."""
    if not rows:
        return []

    by_bin: dict[int, list[ChannelPairRow]] = {}
    for row in rows:
        by_bin.setdefault(_gap_bin_index(row.gap_days), []).append(row)

    summaries: list[GapBinSummary] = []
    for bi in sorted(by_bin):
        group = by_bin[bi]
        rs = np.array([r.waveform_r for r in group], dtype=float)
        deltas = np.array([r.abs_delta_si for r in group if np.isfinite(r.abs_delta_si)], dtype=float)
        signs = [r.sign_match for r in group if np.isfinite(r.abs_delta_si)]
        lo = GAP_BIN_EDGES_DAYS[bi]
        hi = GAP_BIN_EDGES_DAYS[bi + 1] if bi + 1 < len(GAP_BIN_EDGES_DAYS) else np.inf
        summaries.append(
            GapBinSummary(
                bin_label=GAP_BIN_LABELS[bi],
                gap_min_days=float(lo),
                gap_max_days=float(hi) if np.isfinite(hi) else float("inf"),
                n_pairs=len(group),
                n_channels=len({r.channel for r in group}),
                median_waveform_r=float(np.median(rs)),
                p25_waveform_r=float(np.percentile(rs, 25)),
                median_abs_delta_si=float(np.median(deltas)) if deltas.size else np.nan,
                fraction_sign_match=float(np.mean(signs)) if signs else np.nan,
            ),
        )
    return summaries


def _max_gap_below_threshold(
    bin_summaries: list[GapBinSummary],
    *,
    value_attr: str,
    threshold: float,
) -> float:
    """Largest bin upper edge where aggregate metric stays >= threshold."""
    max_gap = 0.0
    for summary in bin_summaries:
        val = getattr(summary, value_attr)
        if not np.isfinite(val) or val < threshold:
            break
        if np.isfinite(summary.gap_max_days):
            max_gap = summary.gap_max_days
    return max_gap


def recommend_combinable_gap(
    pair_rows: list[ChannelPairRow],
    stable_channels: set[int],
    *,
    median_r_threshold: float = DEFAULT_PAIR_R_COMBINE,
    p25_r_threshold: float = DEFAULT_PAIR_R_COMBINE_P25,
) -> TemporalRecommendation:
    """Recommend max inter-session gap for pooling, using stable-channel pairs only."""
    stable_rows = [r for r in pair_rows if r.channel in stable_channels]
    bin_summaries = summarize_pairs_by_gap_bin(stable_rows)
    max_med = _max_gap_below_threshold(
        bin_summaries, value_attr="median_waveform_r", threshold=median_r_threshold,
    )
    max_p25 = _max_gap_below_threshold(
        bin_summaries, value_attr="p25_waveform_r", threshold=p25_r_threshold,
    )
    conservative = min(max_med, max_p25) if max_med and max_p25 else min(max_med, max_p25)
    notes = (
        f"Use gap <= {conservative:.0f} d (conservative) for across-session combine on stable channels; "
        f"median-r limit {max_med:.0f} d @ r>={median_r_threshold}, "
        f"p25-r limit {max_p25:.0f} d @ r>={p25_r_threshold}."
    )
    return TemporalRecommendation(
        max_gap_days_median_r=max_med,
        max_gap_days_p25_r=max_p25,
        median_r_threshold=median_r_threshold,
        p25_r_threshold=p25_r_threshold,
        n_stable_channels=len(stable_channels),
        n_pair_rows_used=len(stable_rows),
        gap_bin_summaries=tuple(bin_summaries),
        notes=notes,
    )


def per_channel_max_gap_at_r(
    pair_rows: list[ChannelPairRow],
    stable_channels: set[int],
    *,
    r_threshold: float = DEFAULT_PAIR_R_COMBINE,
) -> dict[int, float]:
    """Per stable channel: max calendar gap among pairs with waveform r >= threshold."""
    out: dict[int, float] = {}
    for ch in stable_channels:
        gaps = [r.gap_days for r in pair_rows if r.channel == ch and r.waveform_r >= r_threshold]
        out[ch] = max(gaps) if gaps else 0.0
    return out


def greedy_session_clusters(
    session_ids: list[str],
    *,
    max_gap_days: float,
) -> list[list[str]]:
    """Greedy clusters: sessions within max_gap_days of cluster anchor (calendar order)."""
    ordered = sorted(session_ids, key=lambda s: parse_session_datetime(s))
    clusters: list[list[str]] = []
    current: list[str] = []
    anchor: str | None = None
    for sid in ordered:
        if not current:
            current = [sid]
            anchor = sid
            continue
        assert anchor is not None
        if session_gap_days(anchor, sid) <= max_gap_days:
            current.append(sid)
        else:
            clusters.append(current)
            current = [sid]
            anchor = sid
    if current:
        clusters.append(current)
    return clusters


def write_pair_rows_csv(rows: list[ChannelPairRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "session_i", "session_j", "gap_days", "channel", "array_name",
                "waveform_r", "abs_delta_si", "sign_match",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "session_i": row.session_i.split(".")[0],
                    "session_j": row.session_j.split(".")[0],
                    "gap_days": f"{row.gap_days:.4f}",
                    "channel": row.channel,
                    "array_name": row.array_name,
                    "waveform_r": f"{row.waveform_r:.6f}",
                    "abs_delta_si": (
                        f"{row.abs_delta_si:.6f}" if np.isfinite(row.abs_delta_si) else ""
                    ),
                    "sign_match": int(row.sign_match),
                },
            )


def write_gap_bin_csv(summaries: list[GapBinSummary], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "bin_label", "gap_min_days", "gap_max_days", "n_pairs", "n_channels",
                "median_waveform_r", "p25_waveform_r", "median_abs_delta_si",
                "fraction_sign_match",
            ],
        )
        writer.writeheader()
        for s in summaries:
            writer.writerow(
                {
                    "bin_label": s.bin_label,
                    "gap_min_days": s.gap_min_days,
                    "gap_max_days": "" if not np.isfinite(s.gap_max_days) else s.gap_max_days,
                    "n_pairs": s.n_pairs,
                    "n_channels": s.n_channels,
                    "median_waveform_r": f"{s.median_waveform_r:.6f}",
                    "p25_waveform_r": f"{s.p25_waveform_r:.6f}",
                    "median_abs_delta_si": (
                        f"{s.median_abs_delta_si:.6f}" if np.isfinite(s.median_abs_delta_si) else ""
                    ),
                    "fraction_sign_match": (
                        f"{s.fraction_sign_match:.6f}" if np.isfinite(s.fraction_sign_match) else ""
                    ),
                },
            )


def write_channel_max_gap_csv(channel_gaps: dict[int, float], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["channel", "array_name", "max_gap_days_at_r"])
        writer.writeheader()
        for ch in sorted(channel_gaps):
            writer.writerow(
                {
                    "channel": ch,
                    "array_name": _array_name_for_channel(ch),
                    "max_gap_days_at_r": f"{channel_gaps[ch]:.4f}",
                },
            )


@dataclass(frozen=True)
class TradeoffRow:
    n_channels: int
    channel_ids: tuple[int, ...]
    max_n_sessions: int
    session_ids: tuple[str, ...]
    coverage: int
    min_pair_r: float


def rank_channels_for_tradeoff(stabilities: list[ChannelStability]) -> list[int]:
    """Rank channels for top-k sweep: median r, then ICC, then |SI|."""
    ranked = sorted(
        stabilities,
        key=lambda s: (s.median_pairwise_r, s.icc, s.si_median_abs),
        reverse=True,
    )
    return [s.channel for s in ranked]


def pair_r_lookup(rows: list[ChannelPairRow]) -> dict[tuple[str, str, int], float]:
    out: dict[tuple[str, str, int], float] = {}
    for row in rows:
        a, b = row.session_i, row.session_j
        if a > b:
            a, b = b, a
        out[(a, b, row.channel)] = row.waveform_r
    return out


def pair_waveform_r(
    lookup: dict[tuple[str, str, int], float],
    session_a: str,
    session_b: str,
    channel: int,
) -> float | None:
    a, b = session_a, session_b
    if a > b:
        a, b = b, a
    r = lookup.get((a, b, channel))
    return float(r) if r is not None else None


def build_session_compatibility_graph(
    session_ids: list[str],
    channels: list[int],
    pair_lookup: dict[tuple[str, str, int], float],
    *,
    r_min: float = DEFAULT_PAIR_R_COMBINE,
) -> dict[str, set[str]]:
    """Undirected graph: edge iff every channel has waveform r >= r_min."""
    n = len(session_ids)
    adj: dict[str, set[str]] = {sid: set() for sid in session_ids}
    for i in range(n):
        for j in range(i + 1, n):
            si, sj = session_ids[i], session_ids[j]
            ok = True
            for ch in channels:
                r = pair_waveform_r(pair_lookup, si, sj, ch)
                if r is None or r < r_min:
                    ok = False
                    break
            if ok:
                adj[si].add(sj)
                adj[sj].add(si)
    return adj


def max_clique(adjacency: dict[str, set[str]]) -> list[str]:
    """Largest clique (Bron–Kerbosch with pivot). Ties broken by sorted session id."""
    nodes = sorted(adjacency)
    if not nodes:
        return []

    best: list[str] = []

    def choose_pivot(candidates: set[str], excluded: set[str]) -> str | None:
        union = candidates | excluded
        pivot = None
        best_degree = -1
        for node in union:
            degree = len(adjacency[node] & candidates)
            if degree > best_degree:
                best_degree = degree
                pivot = node
        return pivot

    def expand(clique: list[str], candidates: set[str], excluded: set[str]) -> None:
        nonlocal best
        if not candidates and not excluded:
            if len(clique) > len(best):
                best = list(clique)
            elif len(clique) == len(best) and sorted(clique) < sorted(best):
                best = list(clique)
            return
        pivot = choose_pivot(candidates, excluded)
        if pivot is None:
            return
        for node in sorted(candidates - adjacency[pivot]):
            neighbors = adjacency[node]
            expand(
                clique + [node],
                candidates & neighbors,
                excluded & neighbors,
            )
            candidates.remove(node)
            excluded.add(node)

    expand([], set(nodes), set())
    return sorted(best)


def _min_r_in_clique(
    clique: list[str],
    channels: list[int],
    pair_lookup: dict[tuple[str, str, int], float],
) -> float:
    if len(clique) < 2:
        return 1.0
    mins: list[float] = []
    for i in range(len(clique)):
        for j in range(i + 1, len(clique)):
            for ch in channels:
                r = pair_waveform_r(pair_lookup, clique[i], clique[j], ch)
                if r is not None:
                    mins.append(r)
    return float(min(mins)) if mins else float("nan")


def sweep_channel_session_tradeoff(
    session_ids: list[str],
    ranked_channels: list[int],
    pair_rows: list[ChannelPairRow],
    *,
    r_min: float = DEFAULT_PAIR_R_COMBINE,
) -> list[TradeoffRow]:
    """Top-k channel sweep: max combinable sessions = max clique size."""
    if not ranked_channels or not session_ids:
        return []

    pair_lookup = pair_r_lookup(pair_rows)
    rows: list[TradeoffRow] = []
    for k in range(1, len(ranked_channels) + 1):
        ch_set = ranked_channels[:k]
        adj = build_session_compatibility_graph(session_ids, ch_set, pair_lookup, r_min=r_min)
        clique = max_clique(adj)
        min_r = _min_r_in_clique(clique, ch_set, pair_lookup)
        n_sess = len(clique)
        rows.append(
            TradeoffRow(
                n_channels=k,
                channel_ids=tuple(ch_set),
                max_n_sessions=n_sess,
                session_ids=tuple(clique),
                coverage=k * n_sess,
                min_pair_r=min_r,
            ),
        )
    return rows


def invert_tradeoff(rows: list[TradeoffRow]) -> dict[int, int]:
    """For each m sessions, max k with max_n_sessions(k) >= m."""
    if not rows:
        return {}
    max_m = max(r.max_n_sessions for r in rows)
    out: dict[int, int] = {}
    for m in range(1, max_m + 1):
        eligible = [r.n_channels for r in rows if r.max_n_sessions >= m]
        out[m] = max(eligible) if eligible else 0
    return out


def pick_sweet_spot(rows: list[TradeoffRow]) -> TradeoffRow | None:
    """Argmax coverage; tie-break toward more sessions then more channels."""
    if not rows:
        return None
    return max(rows, key=lambda r: (r.coverage, r.max_n_sessions, r.n_channels))


def write_tradeoff_csv(rows: list[TradeoffRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "n_channels", "channel_ids", "max_n_sessions", "session_ids",
                "coverage", "min_pair_r",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "n_channels": row.n_channels,
                    "channel_ids": ";".join(str(c) for c in row.channel_ids),
                    "max_n_sessions": row.max_n_sessions,
                    "session_ids": ";".join(s.split(".")[0] for s in row.session_ids),
                    "coverage": row.coverage,
                    "min_pair_r": f"{row.min_pair_r:.6f}" if np.isfinite(row.min_pair_r) else "",
                },
            )


def write_best_subset_csv(row: TradeoffRow, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "n_channels", "channel_ids", "max_n_sessions", "session_ids",
                "coverage", "min_pair_r",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "n_channels": row.n_channels,
                "channel_ids": ";".join(str(c) for c in row.channel_ids),
                "max_n_sessions": row.max_n_sessions,
                "session_ids": ";".join(s.split(".")[0] for s in row.session_ids),
                "coverage": row.coverage,
                "min_pair_r": f"{row.min_pair_r:.6f}" if np.isfinite(row.min_pair_r) else "",
            },
        )


def write_session_clusters_csv(clusters: list[list[str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["cluster_id", "n_sessions", "session_ids", "span_days"],
        )
        writer.writeheader()
        for cid, cluster in enumerate(clusters, start=1):
            dts = [parse_session_datetime(s) for s in cluster]
            span = (max(dts) - min(dts)).total_seconds() / 86400.0 if len(dts) > 1 else 0.0
            writer.writerow(
                {
                    "cluster_id": cid,
                    "n_sessions": len(cluster),
                    "session_ids": ";".join(s.split(".")[0] for s in cluster),
                    "span_days": f"{span:.4f}",
                },
            )
