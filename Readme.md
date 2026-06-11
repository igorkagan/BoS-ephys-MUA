# Curated MUA data layout

Base folder for the curated MUA data:

```
S:\taskcontroller\SCP_DATA\SCP-CTRL-01\MUA_curated_sessions
```

## Top-level structure

Four subfolders contain collected sessions for Curius and Elmo, in BLOCKED and SHUFFLED conditions:

- `Curius_BLOCKED`
- `Curius_SHUFFLED`
- `Elmo_BLOCKED`
- `Elmo_SHUFFLED`

Each of these contains one subfolder per included session. The folder name is the session ID.

Example session path:

```
Elmo_BLOCKED/20210401T124246.A_Elmo.B_KN.SCP_01/
```

## Session folder contents

### Trial metadata (`.mat`)

Each session folder contains two trial-info files:

- `${session_ID}.trialinfo.mat` — full trial metadata
- `${session_ID}.trialinfo.4python.mat` — subset for Python; contains only `cur_raster_labels` from the full file

Example:

```
20210401T124246.A_Elmo.B_KN.SCP_01.trialinfo.mat
20210401T124246.A_Elmo.B_KN.SCP_01.trialinfo.4python.mat
```

Both files store lists (numeric arrays or cell arrays of strings) with one entry per trial, describing trial properties across several dimensions.

**For parsing, prefer `*.trialinfo.4python.mat`** — it is smaller and exposes the fields needed for trial selection.

**Missing values:** strings `NONE`, `None`, or `none` mean a value could not be assigned for that trial. Exclude these trials from analysis.

### Summary statistics (`.mat`)

Per-channel averaged MUA timecourses across different trial subsets, e.g.:

```
MUA.20210401T124246.A_Elmo.B_KN.SCP_01.Elmo.A.statistic_summary_table.mat
```

These are secondary to the event-aligned data below; use them for summary plots, not primary trial-by-trial parsing.

### Event-aligned MUA data

Each session folder contains one subfolder per exported alignment event. **`A_InitialFixationReleaseTime_ms`** is the primary event for analysis.

Inside each alignment-event subfolder:

1. **One `.mat` file per recorded non-bad channel**, e.g.:

   ```
   20210401T124246.A_Elmo.B_KN.SCP_01.A_InitialFixationReleaseTime_ms.ch001.MUA.pre1000ms.post1000ms.event_aligned_data.mat
   ```

   Variable: `cur_output_data` — processed MUA as an event-aligned 2D array.

   - **Dimension 1:** trials (rows). If the event did not occur in a trial, the entire row is `NaN`.
   - **Dimension 2:** time points around the event (currently always 2001 samples).

   **Exclude trials whose row is all `NaN`.**

2. **One shared time axis file** for that alignment event, e.g.:

   ```
   20210401T124246.A_Elmo.B_KN.SCP_01.A_InitialFixationReleaseTime_ms.MUA.pre1000ms.post1000ms.x_vector_ms.mat
   ```

   Variable: `x_vector_ms` — time in milliseconds relative to the alignment event.

## Trial label fields (`cur_raster_labels`)

Load from `*.trialinfo.4python.mat`. All lists have length = number of trials and align row-wise with `cur_output_data`.

| Field | Values / meaning |
|---|---|
| `A_LR_pos_list` | Monkey's final target choice: `Al` (left), `Ar` (right) |
| `TrialSubType_list` | Trial type. Use `Dyadic` and `SoloA`; ignore `None`, `SemiSolo`, and others |
| `go_seq_500_list` | Action sequence: `ABgo` (simultaneous go), `AgoB` (A first), `BgoA` (A last). Ignore `None` |
| `conf_predictability_list` | Partner predictability. Use `Blocked` in BLOCKED sessions, `Shuffled` in SHUFFLED sessions; ignore `Free` and `None` |
| `A_Reward_list` | Reward to agent A (monkey in these sessions): `RA0`–`RA4`. Ignore `RA0` and `None` |

## Trial selection

Combine boolean masks over the label lists to select trial subsets. Row indices from the mask index into `cur_output_data` for each channel.

### MATLAB

```matlab
blocked_dyadic_rewarded_monkeyfirst_trial_logical_index = ...
    ismember(cur_raster_labels.conf_predictability_list, {'Blocked'}) ...
    & ismember(cur_raster_labels.TrialSubType_list, {'Dyadic'}) ...
    & ismember(cur_raster_labels.A_Reward_list, {'RA1', 'RA2', 'RA3', 'RA4'}) ...
    & ismember(cur_raster_labels.go_seq_500_list, {'AgoB'});

monkey_selected_right_logical_index = ismember(cur_raster_labels.A_LR_pos_list, {'Ar'});
monkey_selected_left_logical_index  = ismember(cur_raster_labels.A_LR_pos_list, {'Al'});
```

### Python (parsing workflow)

```python
import numpy as np
from scipy.io import loadmat

session_id = "20210401T124246.A_Elmo.B_KN.SCP_01"
session_dir = f".../Elmo_BLOCKED/{session_id}"

# 1. Trial labels
trialinfo = loadmat(f"{session_dir}/{session_id}.trialinfo.4python.mat")
labels = trialinfo["cur_raster_labels"].item()  # structured array -> dict-like

def as_str_list(field):
    """Convert MATLAB cell array of strings to Python list of str."""
    return [s.strip() if isinstance(s, str) else str(s).strip() for s in np.atleast_1d(field).ravel()]

predictability = as_str_list(labels["conf_predictability_list"])
subtype      = as_str_list(labels["TrialSubType_list"])
reward       = as_str_list(labels["A_Reward_list"])
go_seq       = as_str_list(labels["go_seq_500_list"])
lr_pos       = as_str_list(labels["A_LR_pos_list"])

invalid = {"NONE", "None", "none", ""}

def valid_mask(*fields):
    return np.logical_and.reduce([~np.isin(f, list(invalid)) for f in fields])

blocked_dyadic_rewarded_monkeyfirst = (
    np.isin(predictability, ["Blocked"])
    & np.isin(subtype, ["Dyadic"])
    & np.isin(reward, ["RA1", "RA2", "RA3", "RA4"])
    & np.isin(go_seq, ["AgoB"])
    & valid_mask(predictability, subtype, reward, go_seq)
)

monkey_right = np.isin(lr_pos, ["Ar"]) & valid_mask(lr_pos)
monkey_left  = np.isin(lr_pos, ["Al"]) & valid_mask(lr_pos)

# 2. Event-aligned MUA for one channel
event = "A_InitialFixationReleaseTime_ms"
ch_file = (
    f"{session_dir}/{event}/"
    f"{session_id}.{event}.ch001.MUA.pre1000ms.post1000ms.event_aligned_data.mat"
)
t_file = (
    f"{session_dir}/{event}/"
    f"{session_id}.{event}.MUA.pre1000ms.post1000ms.x_vector_ms.mat"
)

mua = loadmat(ch_file)["cur_output_data"]          # shape: (n_trials, 2001)
t_ms = loadmat(t_file)["x_vector_ms"].ravel()

# Drop trials with missing labels or all-NaN rows
row_ok = valid_mask(predictability, subtype, reward, go_seq, lr_pos) & ~np.all(np.isnan(mua), axis=1)
mua_sel = mua[blocked_dyadic_rewarded_monkeyfirst & row_ok]
```

## Parsing checklist

1. Walk `Curius_*` / `Elmo_*` session folders.
2. Load `*.trialinfo.4python.mat` → `cur_raster_labels`.
3. Build trial masks; exclude `NONE`/`None`/`none` and all-`NaN` rows.
4. Load `cur_output_data` per channel from the relevant alignment-event subfolder.
5. Load `x_vector_ms` once per alignment event (shared across channels).
6. Apply the same trial mask to every channel's `cur_output_data` so trial indices stay aligned across channels.

## Cross-session consistency

Nominal channel IDs (`ch001`–`ch160`) are physical electrode indices and may not reflect the same neural tissue across days (electrode drift). Before pooling or comparing channels across sessions, run consistency checks within a condition folder.

**Script:** `assess_cross_session_consistency.py` — compares all sessions in one condition (e.g. `Elmo_BLOCKED`).

**Quick replot (heatmaps only):** `replot_consistency_figures.py`

```bash
python assess_cross_session_consistency.py
python replot_consistency_figures.py   # si_heatmap + signed_sig_heatmap only
```

**Output roots** (set `RUN_BOTH_PROCESSING = True` to generate both):

| Mode | Session L/R PDFs | Consistency figures |
|---|---|---|
| Raw MUA | `./figures/original/` | `./figures/original/consistency/{CONDITION}/` |
| Z-scored MUA | `./figures/zscored/` | `./figures/zscored/consistency/{CONDITION}/` |

Z-scoring: per channel, concatenate all trials in a session, subtract mean, divide by SD, then proceed with trial masks and smoothing. Use z-scored outputs when session-to-session amplitude differences (e.g. motivation, signal strength) dominate raw MUA; use original when absolute units matter.

---

### Pipeline (what the script computes)

For each **session × channel**:

1. Load event-aligned MUA; optionally z-score trials per channel.
2. Apply trial filters (configurable; default: Blocked dyadic, AgoB, rewarded RA1–RA4).
3. Split trials by monkey choice (`Al` = left, `Ar` = right); Gaussian-smooth (default 50 ms).
4. Compute mean left/right timecourses and **difference wave** `Δ(t) = μ_L(t) − μ_R(t)`.
5. In the analysis window (default −500 to +500 ms relative to event):
   - Window-mean firing rates per trial (left vs right).
   - **Selectivity index** `SI = (L̄ − R̄) / (abs(L̄) + abs(R̄))` where L̄, R̄ are mean window rates.
   - **Mann–Whitney U** test on per-trial window means (two-sided).

Channels need ≥ `MIN_TRIALS_PER_GROUP` trials per side to contribute statistics.

**Layout convention:** heatmaps are **portrait full page** (8.5×11″); **Y = channel** (ch001–ch160, horizontal lines separate arrays A1–A5), **X = session** (date prefix in label). Session trace colors in Tier 2/3 use the **`cool`** colormap (early → late sessions).

---

### Core measures

| Measure | Definition | Interpretation |
|---|---|---|
| **SI** | `(L̄ − R̄) / (abs(L̄) + abs(R̄))` in analysis window | Signed L/R bias. **+** → left-preferring, **−** → right-preferring, **0** → balanced. Bounded in [−1, 1] when L̄, R̄ ≥ 0. |
| **Δ(t)** | `μ_L(t) − μ_R(t)` at each time point | Full timecourse of L vs R difference; used for shape-based comparisons. |
| **Mann–Whitney p** | On per-trial window means (L vs R) | Non-parametric test of L/R difference within session. Not corrected for multiple channels. |
| **Median pairwise r** | Median Pearson r of **Δ(t)** across all session pairs for one channel | **Shape** consistency: do difference-waveforms correlate across days? High r → similar temporal profile; low/negative r → drift or remapping. |
| **ICC(2,1)** | Two-way random ICC; sessions = raters, time points = targets | **Absolute agreement** of Δ(t) across sessions (includes amplitude). ICC ≥ 0.4 is a common “fair” threshold in config. |
| **Sign concordance** | Fraction of sessions sharing the majority SI sign | **Direction** stability: does the channel stay left- or right-preferring? Reported as `n_same_sign / n_sessions` in CSV. |
| **Stable flag** | `True` iff median r ≥ 0.5 **and** ICC ≥ 0.4 **and** sign concordance ≥ 0.7 | Conservative joint criterion (all three must pass). Thresholds are configurable at top of script. |
| **Session similarity** | For each session pair: median across channels of r(Δ_session_i, Δ_session_j) | Global view: which recording days look most alike in L/R difference structure? |

**Reading SI vs Δ:** SI is a single scalar summary of bias in the analysis window; Δ(t) captures *when* during the epoch L and R diverge. A channel can have consistent SI sign but low pairwise r if waveform *shape* changes across sessions.

---

### Visualization tiers

Figures are grouped into three tiers: **overview → array-level waveforms → channel diagnostics**.

#### Tier 1 — Session × channel overview

Use these first to see which channels and sessions are worth inspecting.

| File | What it shows | How to read it |
|---|---|---|
| **`si_heatmap.pdf`** | Heatmap of SI for every session (columns) × channel (rows). Diverging **RdBu_r**; symmetric color limits ± max abs(SI). | Scan for vertical stripes (session effects) vs horizontal bands (array/channel structure). Stable channels show similar color across sessions. |
| **`signed_sig_heatmap.pdf`** | `sign(SI) × −log10(p)`; entries with **p ≥ α** (default 0.05) are scaled to 25% intensity. | Combines direction and significance. Bright saturated cells = significant L/R difference in that session; faint = non-significant. |
| **`session_similarity.pdf`** | Square matrix: pairwise session similarity (median r of Δ across channels). **Viridis**, range [−1, 1]. | Diagonal = 1. Off-diagonal blocks reveal clusters of “similar” recording days; low similarity suggests poor cross-session comparability for that pair. |
| **`si_stability_{A1–A5}.pdf`** | Two panels per array. **Top:** scatter of SI in each session vs SI in **reference session** (earliest by default); points colored by session (`cool`); dashed unity line. **Bottom:** violin plot of SI across sessions, one violin per channel in the array. | Top: channels on the unity line are stable in SI magnitude/sign vs reference. Bottom: wide violins = high cross-session SI variability; narrow = stable tuning. |
| **`channel_stability.csv`** | One row per channel: `median_pairwise_r`, `icc`, `sign_concordance`, `n_same_sign`, `si_mean`, `si_std`, `stable`. | Sort by `median_pairwise_r` or filter `stable == True` for channels safe to aggregate. Primary quantitative summary. |

#### Tier 2 — Difference-wave consensus (per array)

| File | What it shows | How to read it |
|---|---|---|
| **`delta_consensus_{A1–A5}.pdf`** | 4×8 grid (32 channels per array). Each panel: **median Δ(t)** (black) ± **IQR** (gray band) across sessions; optional thin **session traces** colored by date (`cool`); shaded box = analysis window; gray dashed = event time. Annotation box: median r, sign concordance (`n_same/n`), ICC, stable/unstable flag. | Consensus curve = “typical” L−R difference for that nominal channel. Tight IQR + overlapping session traces → shape agreement. Diverging session traces or wide IQR → unstable. Compare stable vs unstable annotations. |

#### Tier 3 — Deep dive (unstable channels)

| File | What it shows | How to read it |
|---|---|---|
| **`deep_dive_ch{NNN}.pdf`** | 2×5 grid: one mini-panel per session for a single channel. **Red** = mean left-choice trials, **blue** = mean right-choice trials; shared y-axis; analysis window shaded. Title includes trial counts (L, R). | Generated for the **least stable** channels (default: top 20 by lowest median pairwise r among unstable). Inspect whether instability comes from amplitude scaling, timing shifts, sign flips, or missing data. Compare to Tier 2 consensus for the same channel. |

---

### Practical workflow

1. Run `assess_cross_session_consistency.py` for the condition of interest (both original and z-scored if amplitude drift is a concern).
2. **Tier 1:** Scan `si_heatmap.pdf` and `channel_stability.csv`; note channels with high sign concordance but low r (direction stable, shape not).
3. **Tier 2:** Open `delta_consensus_*.pdf` for arrays with candidate channels; confirm waveform agreement in the analysis window.
4. **Tier 3:** Use `deep_dive_ch*.pdf` to diagnose failures (gain change vs remapping vs noise).
5. **Decision:** Pool or average across sessions only for **`stable == True`** channels (or relax criteria explicitly in analysis). Exclude or treat separately all others when using nominal channel IDs.

Default stability thresholds in script: **median pairwise r ≥ 0.5**, **ICC ≥ 0.4**, **sign concordance ≥ 0.7**. Adjust `R_STABLE_THRESH`, `ICC_STABLE_THRESH`, and `SIGN_CONCORDANCE_THRESH` if your use case needs stricter or looser pooling.

---

### Configuration pointers

Key settings at the top of `assess_cross_session_consistency.py`:

- `CONDITION_FOLDER`, `SESSION_IDS`, `TRIAL_FILTERS`, `LEFT_CHOICE` / `RIGHT_CHOICE`
- `ANALYSIS_WINDOW_MS`, `GAUSSIAN_SMOOTH_MS`, `ZSCORE_MUA`, `RUN_BOTH_PROCESSING`
- `REFERENCE_SESSION` — reference for Tier 1 SI scatter (default: earliest session)
- `DEEP_DIVE_UNSTABLE_ONLY`, `MAX_DEEP_DIVE_CHANNELS` — Tier 3 selection
- `SHOW_SESSION_TRACES`, `SESSION_COLORMAP` — Tier 2 session overlay appearance
