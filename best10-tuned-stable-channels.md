# Best10 tuned + SI-stable channels

How `best10_tuned_stable_chXXX.pdf` files are chosen.

**Scripts:** `assess_cross_session_consistency.py` (automatic on full runs), `plot_best_worst_channels.py` (standalone).

See also: [best-and-worst-channels.md](best-and-worst-channels.md) (ranking by Δ-waveform correlation only).

---

## Output locations

**Curated:**

```
figures/{original,zscored}/consistency/{Elmo_BLOCKED,Elmo_SHUFFLED,Curius_BLOCKED,Curius_SHUFFLED}/
  best10_tuned_stable_chXXX.pdf
```

**DUAL_NHP / flat session lists:**

```
{root_folder}/DUAL_NHP/{Monkey}_{AgoB|BgoA}/figures/{original,zscored}/consistency/
  best10_tuned_stable_chXXX.pdf
```

In **original** mode, channels often fail the tuned-stable gate (stricter SI criteria on raw MUA); **z-scored** mode typically produces these PDFs. A warning is logged when fewer than 10 channels pass.

---

## Goal

Channels with **strong L/R tuning** (high |SI|) and **stable SI across sessions** — without requiring the full joint `stable` flag (median Δ r + ICC + sign).

---

## 1. Eligibility pool

Nominal ch001–ch160, usable in **≥ `MIN_SESSIONS` (3)** sessions.

---

## 2. SI stability gate (must pass all)

| Criterion | Metric | Default threshold |
|-----------|--------|-------------------|
| Direction stability | `sign_concordance` | ≥ **0.7** |
| Magnitude stability | `si_std` | ≤ **0.30** |
| Tuning floor | `si_median_abs` | ≥ **0.10** |
| Task engagement | task-evoked (Friedman) | p < α on L or R trials |

Thresholds: `SIGN_CONCORDANCE_THRESH`, `TUNED_STABLE_SI_STD_MAX`, `TUNED_STABLE_SI_ABS_MIN` in `assess_cross_session_consistency.py`.

---

## 3. Ranking

Among gate passers, sort by:

1. **`si_median_abs`** descending
2. Lower **`si_std`**
3. Higher **`sign_concordance`**
4. Higher **`median_pairwise_r`**

Top **`TUNED_STABLE_N` (10)**. If fewer pass, all qualifiers are plotted + warning logged.

---

## 4. CSV columns

`channel_stability.csv` adds: `si_median_abs`, `si_pairwise_median_delta`, `tuned_stable_rank`.

---

## Summary

**Tuned-stable best10** = strongest median |SI| among SI-stable, task-evoked channels — computed independently per run label and processing mode.
