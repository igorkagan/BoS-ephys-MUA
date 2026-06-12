# Best10 tuned + SI-stable channels

This document describes how `best10_tuned_stable_chXXX.pdf` files are chosen.

**Scripts:** `assess_cross_session_consistency.py` (automatic on full runs), `plot_best_worst_channels.py` (standalone for both monkeys).

**Output:**

```
figures/{original,zscored}/consistency/{Elmo_BLOCKED,Curius_BLOCKED}/best10_tuned_stable_chXXX.pdf
```

See also: [best-and-worst-channels.md](best-and-worst-channels.md) (ranking by Δ-waveform correlation only).

---

## Goal

Channels with **strong L/R tuning** (high |SI|) whose **SI stays similar across sessions** — without requiring the full joint `stable` flag (median Δ r + ICC + sign).

---

## 1. Eligibility pool

Same as other consistency metrics: nominal ch001–ch160, usable data in **≥ `MIN_SESSIONS` (3)** sessions.

---

## 2. SI stability gate (must pass all)

| Criterion | Metric | Default threshold |
|-----------|--------|-------------------|
| Direction stability | `sign_concordance` | ≥ **0.7** (`SIGN_CONCORDANCE_THRESH`) |
| Magnitude stability | `si_std` — std of SI across sessions | ≤ **0.30** (`TUNED_STABLE_SI_STD_MAX`) |
| Tuning floor | `si_median_abs` = median(\|SI_s\|) | ≥ **0.10** (`TUNED_STABLE_SI_ABS_MIN`) |

Config in `assess_cross_session_consistency.py`.

---

## 3. Ranking

Among channels passing the gate, sort by:

1. **`si_median_abs`** descending (strongest tuning)
2. Lower **`si_std`** (tie-break: more SI-stable)
3. Higher **`sign_concordance`**
4. Higher **`median_pairwise_r`** (Δ-waveform shape bonus)

Take top **`TUNED_STABLE_N` (10)**.

If fewer than 10 pass the gate, all qualifiers are plotted and a warning is logged.

---

## 4. Additional CSV columns

`channel_stability.csv` includes:

- `si_median_abs`
- `si_pairwise_median_delta` — median \|SI_i − SI_j\| across session pairs
- `tuned_stable_rank` — 1–10 if selected, empty otherwise

---

## 5. Plot contents

Same layout as other deep dives: 2×5 session grid, red = left, blue = right, missing sessions labelled.

Title includes: `median |SI|`, `si_std`, sign concordance, `median pairwise r`.

---

## Summary

**Tuned-stable best10** = highest median |SI| among channels with consistent SI sign, low cross-session SI scatter, and non-trivial tuning strength.
