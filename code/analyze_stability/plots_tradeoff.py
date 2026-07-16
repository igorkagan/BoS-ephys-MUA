"""Interactive channel–session tradeoff explorer (Plotly HTML + static PDFs)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from analyze_stability.temporal_core import TradeoffRow, pair_r_lookup, pair_waveform_r

HEATMAP_DPI = 300


def _short_session(sid: str) -> str:
    return sid.split(".")[0]


def _channel_label(ch: int) -> str:
    return f"ch{ch:03d}"


def _hover_text(row: TradeoffRow) -> str:
    ch_str = ", ".join(_channel_label(c) for c in row.channel_ids[:12])
    if len(row.channel_ids) > 12:
        ch_str += f", … (+{len(row.channel_ids) - 12})"
    sess_str = ", ".join(_short_session(s) for s in row.session_ids[:8])
    if len(row.session_ids) > 8:
        sess_str += f", … (+{len(row.session_ids) - 8})"
    min_r = f"{row.min_pair_r:.3f}" if np.isfinite(row.min_pair_r) else "n/a"
    return (
        f"<b>{row.n_channels} channels → {row.max_n_sessions} sessions</b><br>"
        f"Channels: {ch_str}<br>"
        f"Sessions: {sess_str}<br>"
        f"Coverage: {row.coverage}<br>"
        f"Min pairwise r: {min_r}"
    )


def build_explorer_figure(
    rows: list[TradeoffRow],
    sweet: TradeoffRow | None,
    inverse_lookup: dict[int, int],
    *,
    title: str,
) -> go.Figure:
    if not rows:
        fig = go.Figure()
        fig.update_layout(title=title)
        return fig

    xs = [r.n_channels for r in rows]
    ys = [r.max_n_sessions for r in rows]
    coverages = [r.coverage for r in rows]
    hovers = [_hover_text(r) for r in rows]

    inv_m = sorted(inverse_lookup)
    inv_k = [inverse_lookup[m] for m in inv_m]

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.62, 0.38],
        subplot_titles=("Channels ↔ sessions (max clique)", "Coverage (k × sessions)"),
        horizontal_spacing=0.08,
    )

    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines+markers",
            name="top-k sweep",
            line=dict(color="#1f77b4", width=2),
            marker=dict(size=7),
            text=hovers,
            hovertemplate="%{text}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    if inv_m:
        fig.add_trace(
            go.Scatter(
                x=inv_k,
                y=inv_m,
                mode="lines",
                name="≥m sessions → max k",
                line=dict(color="#ff7f0e", width=1.5, dash="dot"),
                hovertemplate=(
                    "Need ≥%{y} sessions<br>Max channels: %{x}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

    if sweet is not None:
        fig.add_trace(
            go.Scatter(
                x=[sweet.n_channels],
                y=[sweet.max_n_sessions],
                mode="markers",
                name="sweet spot",
                marker=dict(symbol="star", size=14, color="#d62728", line=dict(width=1, color="black")),
                text=[_hover_text(sweet)],
                hovertemplate="%{text}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Scatter(
            x=xs,
            y=coverages,
            mode="lines+markers",
            name="coverage",
            line=dict(color="#2ca02c", width=2),
            marker=dict(size=6),
            text=hovers,
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    if sweet is not None:
        fig.add_trace(
            go.Scatter(
                x=[sweet.n_channels],
                y=[sweet.coverage],
                mode="markers",
                marker=dict(symbol="star", size=14, color="#d62728", line=dict(width=1, color="black")),
                hovertemplate=(
                    f"Sweet spot<br>Coverage: {sweet.coverage}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    spike_kw = dict(showspikes=True, spikemode="across", spikesnap="cursor", spikecolor="#888", spikethickness=1)
    fig.update_xaxes(title_text="Channels (top-k)", row=1, col=1, **spike_kw)
    fig.update_yaxes(title_text="Max combinable sessions", row=1, col=1, **spike_kw)
    fig.update_xaxes(title_text="Channels (top-k)", row=1, col=2, **spike_kw)
    fig.update_yaxes(title_text="Coverage", row=1, col=2, **spike_kw)

    fig.update_layout(
        title=title,
        hovermode="closest",
        template="plotly_white",
        height=480,
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.8)"),
        margin=dict(l=60, r=30, t=60, b=50),
    )
    return fig


def write_explorer_html(fig: go.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path), include_plotlyjs="cdn", full_html=True)


def plot_tradeoff_pdf(
    rows: list[TradeoffRow],
    sweet: TradeoffRow | None,
    title: str,
    out_path: Path,
) -> None:
    if not rows:
        return
    xs = [r.n_channels for r in rows]
    ys = [r.max_n_sessions for r in rows]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(xs, ys, "o-", color="#1f77b4", label="max sessions")
    if sweet is not None:
        ax.plot(
            sweet.n_channels,
            sweet.max_n_sessions,
            "*",
            markersize=14,
            color="#d62728",
            label=f"sweet spot (cov={sweet.coverage})",
        )
    ax.set_xlabel("Top-k channels")
    ax.set_ylabel("Max combinable sessions (clique)")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", dpi=HEATMAP_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_retention_heatmap(
    row: TradeoffRow,
    pair_lookup: dict[tuple[str, str, int], float],
    title: str,
    out_path: Path,
) -> None:
    """Channel × session min-r to other sessions in clique (diagnostic)."""
    sessions = list(row.session_ids)
    channels = list(row.channel_ids)
    if len(sessions) < 2 or not channels:
        return

    n_ch = len(channels)
    mat = np.full((n_ch, len(sessions)), np.nan, dtype=float)
    for ci, ch in enumerate(channels):
        for si, sid in enumerate(sessions):
            rs = []
            for sj in sessions:
                if sj == sid:
                    continue
                r = pair_waveform_r(pair_lookup, sid, sj, ch)
                if r is not None:
                    rs.append(r)
            if rs:
                mat[ci, si] = min(rs)

    fig, ax = plt.subplots(figsize=(max(6, len(sessions) * 0.35), max(4, n_ch * 0.25)))
    im = ax.imshow(mat, aspect="auto", vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(sessions)))
    ax.set_xticklabels([_short_session(s) for s in sessions], rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(n_ch))
    ax.set_yticklabels([_channel_label(c) for c in channels], fontsize=7)
    ax.set_xlabel("Session (min r vs other clique sessions)")
    ax.set_ylabel("Channel")
    ax.set_title(f"{title}\nmin r per channel×session", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8, label="min r")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", dpi=HEATMAP_DPI, bbox_inches="tight")
    plt.close(fig)
