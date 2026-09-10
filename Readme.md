# BoS-ephys-MUA

Event-aligned multiunit activity from Elmo and Curius in a dyadic choice task.

**Docs:** [wiki](https://github.com/igorkagan/BoS-ephys-MUA/wiki) — commands, output trees, z-score, pref, decode, comparisons.

| Dataset | Raw sessions | Figures |
|---|---|---|
| Curated | `MUA_curated_sessions/{Monkey}_{BLOCKED\|SHUFFLED}/{session_id}/` | `figures/{CONDITION}/` |
| Export lists | `{root_folder}/{session_id}/` (`session_lists.m`) | `{root_folder}/{list_name}/` |

```bash
conda activate bos-ephys
PYTHONPATH=code python -u code/run_scripts/run_curated.py Elmo_BLOCKED
PYTHONPATH=code python -u code/run_scripts/run_session_list.py Elmo_BLOCKED_CONF
```

---

# Session files

Curated data root:

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

Averaged timecourses for quick look-plots. Trial-by-trial work uses the event-aligned files below.

### Event-aligned MUA data

Each session folder contains one subfolder per exported alignment event. The pipeline loads MUA from the **recorded monkey's** fixation-release folder:

| Recorded actor side | Alignment event folder |
|---|---|
| A (e.g. `A_Elmo`, Curius DUAL_NHP) | `A_InitialFixationReleaseTime_ms/` |
| B (e.g. `...B` suffix / `B_Elmo`, Elmo DUAL_NHP) | `B_InitialFixationReleaseTime_ms/` |

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
| `B_LR_pos_list` | Partner B's target choice: `Bl` (left), `Br` (right) |
| `TrialSubType_list` | Trial type. Use `Dyadic` and `SoloA`; ignore `None`, `SemiSolo`, and others |
| `go_seq_500_list` | Action sequence: `ABgo` (simultaneous go), `AgoB` (A first), `BgoA` (A last). Ignore `None` |
| `conf_predictability_list` | Partner predictability. Use `Blocked` in BLOCKED sessions, `Shuffled` in SHUFFLED sessions; ignore `Free` and `None` |
| `A_Reward_list` | Reward to agent A: `RA0`–`RA4`. Ignore `RA0` and `None` |
| `B_Reward_list` | Reward to agent B: `RB0`–`RB4`. Ignore `RB0` and `None` |

## Trial selection

Combine boolean masks over the label lists to select trial subsets. Row indices from the mask index into `cur_output_data` for each channel.

### Pipeline rule (all runners)

L/R choice and dyadic reward tiers follow **actor side from the session ID** ([`code/process_channels/preprocess.py`](code/process_channels/preprocess.py)):

| Actor side | Session ID pattern | L/R field | Left / right labels | Dyadic reward field |
|---|---|---|---|---|
| **A** | `.A_{Monkey}.` matches recorded monkey | `A_LR_pos_list` | `Al` / `Ar` | `A_Reward_list` → RA1–RA4 |
| **B** | `.B_{Monkey}.` matches recorded monkey | `B_LR_pos_list` | `Bl` / `Br` | `B_Reward_list` → RB1–RB4 |

`recording_actor_side(session_id, recording_monkey)` parses the ID; `choice_config_for_recording(...)` and `trial_filters_for_go_seq(..., actor_side)` build masks. Solo branches use the same actor side for reward tiers and output subdirs (`SoloA` / `SoloB`).

Example: Elmo in `20210401T124246.A_Elmo.B_KN.SCP_01` is actor **A** → `A_LR_pos_list` and `A_Reward_list`.

Audit per-session counts (correct vs legacy monkey-name choice):

```bash
python -u code/run_scripts/audit_trial_selection.py curated --condition Elmo_BLOCKED --go-seq AgoB
python -u code/run_scripts/audit_trial_selection.py dual_nhp --go-seq AgoB
```

CSV defaults: `figures/{CONDITION}/trial_selection_{AgoB|BgoA}.csv` (curated) or `{DUAL_NHP output}/trial_selection_{AgoB|BgoA}.csv`.

### MATLAB (single-session example, actor A)

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

# 2. Event-aligned MUA for one channel (use actor-side folder for the recorded monkey)
event = "A_InitialFixationReleaseTime_ms"  # or B_InitialFixationReleaseTime_ms when actor is B
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

## Ubuntu 20.04 (Python environment)

Ubuntu 20.04’s `/usr/bin/python3` is 3.8; this repo needs **3.12+**. Leave the system Python in place (apt depends on it) and use conda:

Recommended setup on lab machines: install **Miniforge** in your home directory and use a dedicated conda env. No `sudo` required; system Python stays untouched.

### One-time install

```bash
cd /tmp
curl -fsSL -o Miniforge3-Linux-x86_64.sh \
  https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p ~/miniforge3
~/miniforge3/bin/conda init bash
# open a new shell, then:
conda activate base
mamba create -y -n bos-ephys python=3.12 numpy scipy matplotlib plotly
```

Alternatively, after creating the env:

```bash
conda activate bos-ephys
pip install -r requirements.txt
```

### Run the pipeline

From the repo root, always use the conda env (not `/usr/bin/python3`):

```bash
cd ~/Documents/GitHub/BoS-ephys-MUA
conda activate bos-ephys
PYTHONPATH=code python -u code/run_scripts/run_curated.py Elmo_BLOCKED
PYTHONPATH=code python -u code/run_scripts/run_session_list.py Elmo_BLOCKED_CONF
```

Verify:

```bash
python --version          # Python 3.12.x
python -c "import numpy, scipy, matplotlib, plotly"
```

### SMB/CIFS note

If `root_folder` in [`session_lists.m`](session_lists.m) points at a mounted share (e.g. `~/snd/...`), matplotlib can fail with `OSError: [Errno 16] Device or resource busy` when overwriting an existing PDF. Close any PDF viewer on those files, or delete the partial output subfolder before rerunning a failed step.

## Repository layout

All Python code lives under **`code/`**. Runnable entry points are in **`code/run_scripts/`**. Run from the repo root:

```bash
python -u code/run_scripts/run_curated.py Elmo_BLOCKED
python -u code/run_scripts/run_session_list.py DUAL_NHP
python -u code/run_scripts/audit_conf_sessions.py --all-conf
```

```
BoS-ephys-MUA/
  session_lists.m
  readme.md
  requirements.txt
  code/
    load_data/             # I/O, session lists, disk cache, audit
    process_channels/      # preprocess, features, evoked
    analyze_decoding/      # Decodanda (run_decode_*.py)
    analyze_stability/     # consistency, combine, arrays, pref_unpref, temporal
    compare_conditions/    # condition-vs-condition comparison PDFs
    run_pipeline/          # unified runner (curated + session-list)
    run_scripts/           # CLIs (_bootstrap.py adds repo + code/ to sys.path)
      run_curated.py
      run_session_list.py
      run_decode_curated.py
      run_decode_session_list.py
      run_decode_compare.py
      run_session_lr.py    # partial reruns (one step each)
      run_consistency.py
      run_combine.py
      run_array_combined.py
      run_best_worst.py
      run_comparisons.py
      run_stability_across_sessions.py
      audit_conf_sessions.py
      audit_trial_selection.py
      replot_consistency.py
    tests/
  audit/                   # session label QC CSVs
  figures/                 # pipeline PDF/CSV outputs (curated runs)
  lists/                   # curated session selection lists
```

| Role | Location |
|---|---|
| Data I/O, session lists, cache | `code/load_data/` |
| Channel preprocessing & features | `code/process_channels/` |
| Cross-session stability & combine | `code/analyze_stability/` |
| Decoding | `code/analyze_decoding/` |
| Condition comparisons | `code/compare_conditions/` |
| Pipeline orchestration | `code/run_pipeline/runner.py` |
| What you execute | `code/run_scripts/*.py` |
| Extended docs | [wiki](https://github.com/igorkagan/BoS-ephys-MUA/wiki) |

---

## Analysis pipeline

Same engine for both datasets: [`code/run_pipeline/runner.py`](code/run_pipeline/runner.py). Folder trees, z-score, pref, decode: [wiki](https://github.com/igorkagan/BoS-ephys-MUA/wiki).

| | Curated | Export lists (`session_lists.m`) |
|---|---|---|
| Command | `run_curated.py Elmo_BLOCKED` | `run_session_list.py Elmo_BLOCKED_CONF` |
| Sessions | Every subdir under the condition folder | Named cell array |
| Writes | `figures/{CONDITION}/` | `{root_folder}/{list_name}/` |

Inner layout is the same: `{Monkey}_{AgoB|BgoA}/{Dyadic|SoloA}/` plus `{Monkey}_Dyadic_first_second_comparison/` and `{Monkey}_{go}/Dyadic_vs_SoloA_comparison/`. Plot colors: **left = red**, **right = blue**. Log: `{output_root}/pipeline.log`.

Default processing is z-scored (actor-trial μ/σ per channel). `--also-original` also writes raw under `original/` (z-scored then under `zscored/`). Caches: `{branch}/.cache/summaries_zscore.npz`.

```bash
PYTHONPATH=code python -u code/run_scripts/run_curated.py Elmo_BLOCKED
PYTHONPATH=code python -u code/run_scripts/run_curated.py Elmo_BLOCKED --go-seq AgoB --dyadic-only
PYTHONPATH=code python -u code/run_scripts/run_curated.py Elmo_BLOCKED --steps session_lr,consistency

PYTHONPATH=code python -u code/run_scripts/run_session_list.py --list
PYTHONPATH=code python -u code/run_scripts/run_session_list.py Elmo_BLOCKED_CONF
PYTHONPATH=code python -u code/run_scripts/run_session_list.py DUAL_NHP --monkey Curius --go-seq BgoA
```

Decode (separate CLIs, same output tree, `decoding/`):

```bash
PYTHONPATH=code python -u code/run_scripts/run_decode_curated.py Elmo_BLOCKED --mode dyadic_all
PYTHONPATH=code python -u code/run_scripts/run_decode_session_list.py Elmo_BLOCKED_CONF --mode dyadic_all
PYTHONPATH=code python -u code/run_scripts/run_decode_compare.py
```

### Steps

| Step | Writes |
|---|---|
| `session_lr` | Per-session L/R PSTHs |
| `pref_unpref` | Combined preferred vs unpreferred |
| `consistency` | SI heatmaps, `channel_stability.csv`, consensus Δ, best/worst dives |
| `stability_across_sessions` | HTML explorer |
| `combine` | Per-channel combined L/R |
| `array_combined` | Array-mean ± SE |
| `best_worst` | Deep dives (also written by `consistency`) |
| `comparisons` | AgoB vs BgoA and Dyadic vs Solo overlays |

Partial reruns: `run_session_lr.py`, `run_consistency.py`, `run_combine.py`, `run_array_combined.py`, `run_comparisons.py`, `run_best_worst.py`, `run_stability_across_sessions.py`. Restrict sessions on those scripts with `SESSION_IDS` in the module.

### Curated conditions

Input: `MUA_curated_sessions/{Monkey}_{BLOCKED|SHUFFLED}/{session_id}/`.

| Condition | Timing folders | Trial mask |
|---|---|---|
| `Elmo_BLOCKED` / `Curius_BLOCKED` | `{Monkey}_AgoB`, `{Monkey}_BgoA` | Dyadic or actor solos + go-seq + actor reward + `Blocked` |
| `Elmo_SHUFFLED` / `Curius_SHUFFLED` | same | same with `Shuffled` |

L/R from the session-ID actor side (`choice_config_for_recording`). Confederate `*_CONF` lists use the same splits; DUAL_NHP adds Curius and Elmo (four folders). DUAL_NHP trial mask is Dyadic + go-seq + actor reward.

```
figures/Elmo_BLOCKED/
  pipeline.log
  Elmo_AgoB/Dyadic/   Elmo_AgoB/SoloA/
  Elmo_BgoA/…
  Elmo_AgoB/Dyadic_vs_SoloA_comparison/
  Elmo_Dyadic_first_second_comparison/
```

```
{root}/Elmo_BLOCKED_CONF/          # same inner tree
{root}/DUAL_NHP/Curius_AgoB/…      # plus Elmo_* folders
  decode_compare/                  # from run_decode_compare.py
```

DUAL_NHP recorded-monkey split: datetime suffix **`U`** or no `B` → Curius; suffix **`B`** → Elmo.

### Other commands

| Script | Use |
|---|---|
| `run_decode_curated.py` / `run_decode_session_list.py` | Choice decode |
| `run_decode_compare.py` | Overlay combined decode caches |
| `replot_consistency.py` | Rebuild heatmaps |
| `audit_conf_sessions.py` | Label QC → `audit/` |
| `audit_trial_selection.py` | Per-session L/R counts |
| `plan_pipeline.py` / `audit_outputs.py` | Expected files vs disk |

```bash
python -u code/run_scripts/audit_conf_sessions.py --all-conf
python -u code/run_scripts/audit_conf_sessions.py --dual-nhp
cd code && python -m run_scripts.check
python -m run_scripts.plan_pipeline Elmo_BLOCKED
python -m run_scripts.audit_outputs ../figures/Elmo_BLOCKED
```

## Cross-session consistency

How to read the heatmaps and CSVs: [wiki: Channel stability](https://github.com/igorkagan/BoS-ephys-MUA/wiki/Consistency).

Nominal channel IDs (`ch001`–`ch160`) are physical electrode indices. Missing channels are NaN / empty axes. Stability metrics use sessions where that channel had usable data.

### Pipeline (what the script computes)

For each **session × channel**:

1. Load event-aligned MUA; optionally z-score per channel (μ/σ from actor trials).
2. Apply trial filters (curated: Blocked/Shuffled dyadic AgoB/BgoA + actor-side reward; DUAL_NHP: Dyadic + AgoB or BgoA + actor-side RA/RB).
3. Split trials by actor-side choice field (`Al`/`Ar` or `Bl`/`Br`); Gaussian-smooth (default 50 ms).
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
| **Mann–Whitney p** | On per-trial window means (L vs R) | Per session × channel, uncorrected across electrodes. |
| **Median pairwise r** | Median Pearson r of **Δ(t)** across all session pairs for one channel | **Shape** consistency: do difference-waveforms correlate across days? High r → similar temporal profile; low/negative r → drift or remapping. |
| **ICC(2,1)** | Two-way random ICC; sessions = raters, time points = targets | **Absolute agreement** of Δ(t) across sessions (includes amplitude). ICC ≥ 0.4 is a common “fair” threshold in config. |
| **Sign concordance** | Fraction of sessions sharing the majority SI sign | **Direction** stability: does the channel stay left- or right-preferring? Reported as `n_same_sign / n_sessions` in CSV. |
| **Stable flag** | `True` iff median r ≥ 0.5 **and** ICC ≥ 0.4 **and** sign concordance ≥ 0.7 | Conservative joint criterion (all three must pass). Thresholds are configurable at top of script. |
| **Session similarity** | For each session pair: median across channels of r(Δ_session_i, Δ_session_j) | Global view: which recording days look most alike in L/R difference structure? |

**Reading SI vs Δ:** SI is a single scalar summary of bias in the analysis window; Δ(t) captures *when* during the epoch L and R diverge. A channel can have consistent SI sign but low pairwise r if waveform *shape* changes across sessions.

---

### Visualization tiers

Figures below live under:

- **Curated:** `figures/{CONDITION}/{Monkey}_{AgoB|BgoA}/Dyadic/consistency/` (or `SoloA/`)
- **Session-list:** `{root_folder}/{list_name}/{Monkey}_{AgoB|BgoA}/Dyadic/consistency/`

With `--also-original`, insert `original/` or `zscored/` after the trial-type folder.

Figures are grouped into three tiers: **overview → array-level waveforms → channel diagnostics**.

#### Tier 1 — Session × channel overview

Use these first to see which channels and sessions are worth inspecting.

| File | What it shows | How to read it |
|---|---|---|
| **`si_heatmap.pdf`** | Heatmap of SI for every session (columns) × channel (rows). Diverging **RdBu_r**; symmetric color limits ± max abs(SI). | Scan for vertical stripes (session effects) vs horizontal bands (array/channel structure). Stable channels show similar color across sessions. |
| **`signed_sig_heatmap.pdf`** | `sign(SI) × −log10(p)`; entries with **p ≥ α** (default 0.05) are scaled to 25% intensity. | Combines direction and significance. Bright saturated cells = significant L/R difference in that session; faint = non-significant. |
| **`session_similarity.pdf`** | Square matrix: pairwise session similarity (median r of Δ across channels). **Viridis**, range [−1, 1]. | Diagonal = 1. Off-diagonal blocks reveal clusters of “similar” recording days; low similarity suggests poor cross-session comparability for that pair. |
| **`si_stability_{A1–A5}.pdf`** | Two panels per array. **Top:** scatter of SI in each session vs SI in **reference session** (earliest by default); points colored by session (`cool`); dashed unity line. **Bottom:** violin plot of SI across sessions, one violin per channel in the array. | Top: channels on the unity line are stable in SI magnitude/sign vs reference. Bottom: wide violins = high cross-session SI variability; narrow = stable tuning. |
| **`channel_stability.csv`** | One row per channel with ≥ `MIN_SESSIONS` present: `median_pairwise_r`, `icc`, `sign_concordance`, `n_same_sign`, `si_mean`, `si_std`, `stable`. | Sort by `median_pairwise_r` or filter `stable == True` for channels safe to aggregate. Primary quantitative summary. |
| **`channel_presence.csv`** | Per session × array: count and list of **missing** nominal channels. | Use to see which electrodes were absent or failed QC in each recording. |

#### Tier 2 — Difference-wave consensus (per array)

| File | What it shows | How to read it |
|---|---|---|
| **`delta_consensus_{A1–A5}.pdf`** | 4×8 grid (32 channels per array). Each panel: **median Δ(t)** (black) ± **IQR** (gray band) across sessions; optional thin **session traces** colored by date (`cool`); shaded box = analysis window; gray dashed = event time. Annotation box: median r, sign concordance (`n_same/n`), ICC, stable/unstable flag. | Consensus curve = “typical” L−R difference for that nominal channel. Tight IQR + overlapping session traces → shape agreement. Diverging session traces or wide IQR → unstable. Compare stable vs unstable annotations. |

#### Tier 3 — Deep dive (unstable channels)

| File | What it shows | How to read it |
|---|---|---|
| **`best10_ch*.pdf`**, **`worst10_ch*.pdf`** | L/R deep dive for top/bottom 10 by **median pairwise r** of Δ waveforms. | See [best-and-worst-channels.md](best-and-worst-channels.md). |
| **`best10_tuned_stable_ch*.pdf`** | L/R deep dive for top 10 **strongly tuned** channels with **stable SI** across sessions. | See [best10-tuned-stable-channels.md](best10-tuned-stable-channels.md). |
| **`deep_dive_ch{NNN}.pdf`** | 2×5 grid: one mini-panel per session for a single channel. **Red** = mean left-choice trials, **blue** = mean right-choice trials; shared y-axis; analysis window shaded. Title includes trial counts (L, R). | Generated for the **least stable** channels (default: top 20 by lowest median pairwise r among unstable). Inspect whether instability comes from amplitude scaling, timing shifts, sign flips, or missing data. Compare to Tier 2 consensus for the same channel. |

---

### Reading order

1. `si_heatmap.pdf` + `channel_stability.csv` — direction and which electrodes stay put.
2. `delta_consensus_*.pdf` — waveform agreement in the analysis window.
3. `deep_dive_ch*.pdf` — per-day L/R when shape disagrees (gain vs remapping vs noise).

`stable == True` means median r ≥ 0.5, ICC ≥ 0.4, and sign concordance ≥ 0.7 (`R_STABLE_THRESH`, `ICC_STABLE_THRESH`, `SIGN_CONCORDANCE_THRESH` in the script). Those channels are the straightforward ones to pool across days.

---

### Configuration pointers

Key settings at the top of [`code/analyze_stability/consistency.py`](code/analyze_stability/consistency.py), [`code/run_scripts/run_curated.py`](code/run_scripts/run_curated.py), and [`code/run_scripts/run_session_list.py`](code/run_scripts/run_session_list.py):

- `CONDITION_FOLDER` / CLI list name — which run to analyze (curated condition or `session_lists.m` entry)
- `SESSION_IDS` — `None` = all sessions in folder/list; or an explicit list (individual scripts only)
- `TRIAL_FILTERS` — `None` = auto from condition name / list prefix via `trial_filters_for_go_seq` + actor side
- `LEFT_CHOICE` / `RIGHT_CHOICE` — overridden per run from `choice_config_for_recording` (actor side)
- `ANALYSIS_WINDOW_MS`, `GAUSSIAN_SMOOTH_MS`, `ZSCORE_MUA`, `RUN_BOTH_PROCESSING`
- `REFERENCE_SESSION` — reference for Tier 1 SI scatter (default: earliest session)
- `DEEP_DIVE_UNSTABLE_ONLY`, `MAX_DEEP_DIVE_CHANNELS` — Tier 3 selection
- `SHOW_SESSION_TRACES`, `SESSION_COLORMAP` — Tier 2 session overlay appearance

`comparisons` runs once per monkey after the selected go-seq branches. AgoB vs BgoA needs `--go-seq all`. `--dyadic-only` writes Dyadic branches only.

Every branch writes `pipeline_manifest.json`; every comparison writes `comparison_manifest.json` (filters, paths, `B − A` delta).

Parity tests that need the network data:

```bash
cd code
BOS_RUN_DATA_TESTS=1 python -m unittest tests.test_curated_trialinfo_vs_export -v
```
