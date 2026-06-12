# Best10 / Worst10 channel selection

This document describes how `best10_chXXX.pdf` and `worst10_chXXX.pdf` are chosen in the cross-session consistency pipeline.

**Scripts:** `assess_cross_session_consistency.py` (automatic on full runs), `plot_best_worst_channels.py` (standalone for both monkeys).

**Output locations:**

```
figures/{original,zscored}/consistency/{Elmo_BLOCKED,Curius_BLOCKED}/best10_chXXX.pdf
figures/{original,zscored}/consistency/{Elmo_BLOCKED,Curius_BLOCKED}/worst10_chXXX.pdf
```

---

## 1. Eligible channels

Only **nominal** channels (ch001–ch160) that have usable data in **≥3 sessions** (`MIN_SESSIONS = 3`).

A session counts if that channel has a non-NaN difference wave **Δ(t) = μ_L(t) − μ_R(t)** after trial filtering, optional z-scoring, Gaussian smoothing, and L/R split. Channels missing in most sessions never enter the ranking.

---

## 2. Ranking metric: `median_pairwise_r`

For each eligible channel, independently of the binary **stable** flag:

1. Take **Δ(t)** from every session where the channel exists.
2. For each **pair of sessions** (i, j), compute Pearson **r** between their Δ waveforms (only at time points finite in both; requires ≥3 overlapping samples).
3. Collect all pairwise r values for that channel.
4. **`median_pairwise_r`** = median of those r values.

**Interpretation:** shape consistency of the L−R difference wave across days. High r → similar tuning timecourse; low or negative r → drift or remapping.

Implementation: `pairwise_correlations()` and `assess_channel_stability()` in `bos_mua/stability.py`.

---

## 3. Best vs worst

All eligible channels in a condition are sorted by `median_pairwise_r` **descending** (`rank_best_worst_channels()`):

| Set | Selection |
|-----|-----------|
| **best10** | Top 10 highest median r |
| **worst10** | Bottom 10 lowest median r (listed lowest-first in output order) |

Default count: `BEST_WORST_N = 10` in `assess_cross_session_consistency.py`.

Ties follow Python’s stable sort order.

---

## 4. What is *not* used for ranking

These metrics appear in the plot title but **do not** determine best/worst:

- **Stable flag** — requires median r ≥ 0.5 **and** ICC ≥ 0.4 **and** sign concordance ≥ 0.7
- **SI** — scalar L/R bias in the analysis window: `(L−R)/(abs(L)+abs(R))`
- **ICC(2,1)** — absolute agreement of Δ(t) across sessions
- **Sign concordance** — fraction of sessions sharing the majority SI sign

A channel can appear in **best10** but still be **unstable** under the full joint criteria if ICC or sign concordance fail.

---

## 5. Scope

Ranking is done **separately** for each:

- Condition: `Elmo_BLOCKED`, `Curius_BLOCKED`
- Processing mode: `original` (raw MUA) vs `z-scored` (per-channel trial z-score within session)

The same nominal channel can rank differently across monkeys or processing modes.

---

## 6. Plot contents

Each PDF is a **deep dive**: 2×5 grid (one panel per session), **red** = mean left-choice trials, **blue** = mean right-choice trials, analysis window shaded, missing sessions labelled `[missing]`.

The suptitle includes: condition, event, trial filters, processing label, channel ID, array, **median r**, **ICC**, **sign concordance** (`n_same/n_sessions`), and **stable/unstable** flag.

---

## Summary

**Best** = highest cross-session Δ-waveform correlation; **worst** = lowest — among channels present in at least three sessions.
