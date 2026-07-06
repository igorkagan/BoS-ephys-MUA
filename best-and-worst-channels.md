# Best10 / Worst10 channel selection

How `best10_chXXX.pdf` and `worst10_chXXX.pdf` are chosen in the cross-session consistency pipeline.

**Modules:** `bos_mua/steps/consistency.py` (automatic on full runs), `bos_mua/steps/best_worst.py` (standalone).

---

## Output locations

**Curated** (`scripts/run_curated.py`):

```
figures/{original,zscored}/consistency/{Elmo_BLOCKED,Elmo_SHUFFLED,Curius_BLOCKED,Curius_SHUFFLED}/
  best10_chXXX.pdf
  worst10_chXXX.pdf
```

**DUAL_NHP / flat session lists** (`scripts/run_session_list.py`):

```
{root_folder}/DUAL_NHP/{Monkey}_{AgoB|BgoA}/figures/{original,zscored}/consistency/
  best10_chXXX.pdf
  worst10_chXXX.pdf
```

Same filenames and ranking logic in both layouts; only the parent path differs.

---

## 1. Eligible channels

Nominal channels ch001–ch160 with usable data in **≥3 sessions** (`MIN_SESSIONS = 3`).

A session counts if that channel has a non-NaN difference wave **Δ(t) = μ_L(t) − μ_R(t)** after trial filtering, optional z-scoring (actor-trial reference), Gaussian smoothing, and L/R split.

---

## 2. Ranking metric: `median_pairwise_r`

For each eligible channel:

1. Take **Δ(t)** from every session where the channel exists.
2. For each session pair (i, j), Pearson **r** between Δ waveforms (≥3 overlapping finite time points).
3. **`median_pairwise_r`** = median of all pairwise r values.

Implementation: `pairwise_correlations()` and `assess_channel_stability()` in `bos_mua/stability.py`.

---

## 3. Best vs worst

Sort eligible channels by `median_pairwise_r` **descending** (`rank_best_worst_channels()`):

| Set | Selection |
|-----|-----------|
| **best10** | Top 10 highest median r |
| **worst10** | Bottom 10 lowest median r |

Default: `BEST_WORST_N = 10`.

---

## 4. What is *not* used for ranking

Not used for best/worst selection (but shown in plot titles):

- **Stable flag** (median r + ICC + sign concordance joint gate)
- **SI**, **ICC**, **sign concordance**

---

## 5. Scope

Ranking is **separate** for each:

- **Run label** — e.g. `Elmo_BLOCKED`, `Curius_AgoB`, `Elmo_BgoA`
- **Processing mode** — `original` vs `zscored`

L/R colors: **red** = left choice, **blue** = right choice (`Al`/`Ar` or `Bl`/`Br` depending on run).

---

## Summary

**Best** = highest cross-session Δ-waveform correlation; **worst** = lowest — among channels present in ≥3 sessions.
