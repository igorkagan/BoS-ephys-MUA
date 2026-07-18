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

L/R choice and dyadic reward tiers follow **actor side from the session ID**, not the recorded monkey's name. Implemented in [`code/process_channels/preprocess.py`](code/process_channels/preprocess.py):

| Actor side | Session ID pattern | L/R field | Left / right labels | Dyadic reward field |
|---|---|---|---|---|
| **A** | `.A_{Monkey}.` matches recorded monkey | `A_LR_pos_list` | `Al` / `Ar` | `A_Reward_list` → RA1–RA4 |
| **B** | `.B_{Monkey}.` matches recorded monkey | `B_LR_pos_list` | `Bl` / `Br` | `B_Reward_list` → RB1–RB4 |

`recording_actor_side(session_id, recording_monkey)` parses the ID; `choice_config_for_recording(...)` and `trial_filters_for_go_seq(..., actor_side)` build masks. Solo branches use the same actor side for reward tiers and output subdirs (`SoloA` / `SoloB`).

Example: Elmo recorded in `20210401T124246.A_Elmo.B_KN.SCP_01` is actor **A** → `A_LR_pos_list` and `A_Reward_list`, not B-side fields.

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

Ubuntu 20.04 ships **Python 3.8** as `/usr/bin/python3`. That is too old for this repo (CI tests **3.12+**; some code uses 3.10+ syntax such as `zip(..., strict=True)`). **Do not replace or remove the system Python** — apt and OS tools depend on it.

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
    analyze_stability/     # consistency, combine, arrays, temporal
    compare_conditions/    # condition-vs-condition comparison PDFs
    run_pipeline/          # unified runner (curated + session-list)
    run_scripts/           # CLIs (_bootstrap.py adds repo + code/ to sys.path)
      run_curated.py
      run_session_list.py
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
| Condition comparisons | `code/compare_conditions/` |
| Pipeline orchestration | `code/run_pipeline/runner.py` |
| What you execute | `code/run_scripts/*.py` |

---

## Analysis pipeline

Curated and session-list runs share **one engine** ([`code/run_pipeline/runner.py`](code/run_pipeline/runner.py)): branched dyadic/solo loads, disk cache, and explicit comparisons. They differ only in **how sessions are collected**:

| | **Curated** | **Session-list (`session_lists.m`)** |
|---|---|---|
| **Runner** | [`code/run_scripts/run_curated.py`](code/run_scripts/run_curated.py) | [`code/run_scripts/run_session_list.py`](code/run_scripts/run_session_list.py) |
| **Data root** | `MUA_curated_sessions/{Monkey}_{BLOCKED\|SHUFFLED}/` | `root_folder` in `session_lists.m` (flat `{session_id}/` folders) |
| **Sessions** | All subdirs under condition folder | Explicit list in named cell array |
| **Typical lists** | `Curius_BLOCKED`, `Elmo_SHUFFLED`, … | `DUAL_NHP`, `Elmo_BLOCKED_CONF`, … |

### Pipeline steps (default: all)

| Step | Module | Curated output | DUAL_NHP / confederate flat-list output |
|---|---|---|---|
| `session_lr` | `code/analyze_stability/session_lr.py` | `figures/{CONDITION}/{Monkey}_{AgoB\|BgoA}/Dyadic/` (or `SoloA`/`SoloB`) | `{output}/{Monkey}_{AgoB\|BgoA}/Dyadic/` |
| `consistency` | `code/analyze_stability/consistency.py` | `…/Dyadic/consistency/` | `{output}/…/Dyadic/consistency/` |
| `stability_across_sessions` | `code/analyze_stability/stability_across_sessions.py` | `…/Dyadic/stability_across_sessions/` | same pattern |
| `combine` | `code/analyze_stability/combine.py` | `…/Dyadic/combined/` | `{output}/…/Dyadic/combined/` |
| `array_combined` | `code/analyze_stability/arrays.py` | same `…/combined/` (array-mean ± SE) | `{output}/…/combined/*_arrays_combined_LR.pdf` |
| `best_worst` | `code/analyze_stability/deep_dives.py` | deep dives under `…/consistency/` | deep dives under `…/consistency/` |
| `comparisons` | `code/compare_conditions/compare.py` | `figures/{CONDITION}/{Monkey}_Dyadic_first_second_comparison/` plus fixed-go-sequence social comparisons | Same relative comparison tree under `{root}/{list}/` |

Steps always run in table order. Pass a subset with `--steps`, e.g. `--steps session_lr,consistency`, or use a partial runner: `python -u code/run_scripts/run_consistency.py Curius_BLOCKED`.

**Z-scoring:** per channel, μ/σ from **actor trials** only (Dyadic + that monkey's active solos; see [`code/process_channels/preprocess.py`](code/process_channels/preprocess.py) `zscore_reference_mask`), then apply analysis trial masks and smoothing. Transform is applied to all finite samples in the matrix.

**Run log:** curated and session-list runners tee stdout/stderr to `{output_root}/pipeline.log` via [`code/run_pipeline/run_log.py`](code/run_pipeline/run_log.py) (e.g. `figures/Elmo_BLOCKED/pipeline.log`, `{root}/DUAL_NHP/pipeline.log`).

**Pipeline data cache:** [`code/run_pipeline/runner.py`](code/run_pipeline/runner.py) builds a `RunDataCache` once per go-seq run. Each channel `.mat` is loaded **once** per `(session, channel)`; z-scored summaries/trials are derived in memory by default (`RUN_BOTH_PROCESSING=False` in [`code/run_pipeline/config.py`](code/run_pipeline/config.py)). Pass `--also-original` to also emit raw outputs under `original/` (with z-scored under `zscored/`). Disk snapshots land in `{branch}/.cache/summaries_{raw|zscore}.npz` for comparisons. `best_worst` is skipped when `consistency` ran.

Plot colors: **left = red**, **right = blue**.

---

### Curated pipeline — `code/run_scripts/run_curated.py`

**Sessions are not listed in the runner.** Every subdirectory under the condition folder is processed (`discover_sessions` in [`code/load_data/io.py`](code/load_data/io.py)), sorted by date prefix in the session ID.

Each condition runs **two go-seq splits** (`Elmo_AgoB`, `Elmo_BgoA` for `Elmo_BLOCKED`), each with **dyadic + solo** trial branches (unless `--dyadic-only`). Same step engine as confederate session lists.

**Input data** (network path):

```
S:\taskcontroller\SCP_DATA\SCP-CTRL-01\MUA_curated_sessions\
  Elmo_BLOCKED\
    20210401T124246.A_Elmo.B_KN.SCP_01\
    ...
```

**Conditions** (`MONKEY_CONDITIONS` in [`code/process_channels/preprocess.py`](code/process_channels/preprocess.py)):

| Condition | go_seq runs | Trial filters (per run) |
|---|---|---|
| `Elmo_BLOCKED` / `Curius_BLOCKED` | `{Monkey}_AgoB`, `{Monkey}_BgoA` | Dyadic + `{AgoB\|BgoA}` + actor-side RA/RB tiers + `conf_predictability=Blocked` |
| `Elmo_SHUFFLED` / `Curius_SHUFFLED` | same | same with `Shuffled` |

L/R split: actor-side choice from session ID (`choice_config_for_recording`); e.g. `A_Elmo` sessions → `A_LR_pos_list` (`Al`/`Ar`), not monkey-name defaults.

```bash
python -u code/run_scripts/run_curated.py Elmo_BLOCKED
```

**Output layout** — everything under `figures/{CONDITION}/`:

```
figures/Elmo_BLOCKED/
  pipeline.log
  Elmo_AgoB/
    Dyadic/
      .cache/summaries_zscore.npz
      {session}_..._LR.pdf
      consistency/
      combined/
      stability_across_sessions/
    SoloA/
      (same structure)
  Elmo_BgoA/
    Dyadic/
    SoloA/
  Elmo_Dyadic_first_second_comparison/
```

Add `--also-original` for `Dyadic/original/` and `Dyadic/zscored/` subfolders.

```bash
python -u code/run_scripts/run_curated.py Elmo_BLOCKED --go-seq AgoB --dyadic-only
python -u code/run_scripts/run_curated.py Elmo_BLOCKED --steps session_lr,consistency
python -u code/run_scripts/run_stability_across_sessions.py Elmo_BLOCKED
```

To restrict sessions when running individual steps, set `SESSION_IDS` in [`code/analyze_stability/consistency.py`](code/analyze_stability/consistency.py), [`code/analyze_stability/deep_dives.py`](code/analyze_stability/deep_dives.py), or [`code/run_scripts/replot_consistency.py`](code/run_scripts/replot_consistency.py).

---

### Session-list pipeline — `code/run_scripts/run_session_list.py`

Reads session IDs from [`session_lists.m`](session_lists.m). Each named cell array (e.g. `DUAL_NHP`, `Elmo_BLOCKED_CONF`) defines `root_folder`, sessions, and output path `{root_folder}/{list_name}/`.

**Flat export layout:**

```
S:\...\MUA_export_per_session\
  20210127T130717.A_Elmo.B_FS.SCP_01\    ← session folders directly under root_folder
  ...
```

Confederate lists (`*_CONF`) use the go-seq split pipeline (see table below). Other flat lists without the confederate naming pattern fall back to a single AgoB-only run via `build_flat_session_list_context`.

**Named lists in [`session_lists.m`](session_lists.m):**

| List | Sessions | Output |
|---|---|---|
| `DUAL_NHP` | 8 paired exports | see below |
| `Elmo_BLOCKED_CONF` | 40 | `{root}/Elmo_BLOCKED_CONF/{Monkey}_{AgoB\|BgoA}/figures/…` |
| `Elmo_SHUFFLED_CONF` | 13 | `{root}/Elmo_SHUFFLED_CONF/{Monkey}_{AgoB\|BgoA}/figures/…` |
| `Curius_BLOCKED_CONF` | 22 | `{root}/Curius_BLOCKED_CONF/{Monkey}_{AgoB\|BgoA}/figures/…` |
| `Curius_SHUFFLED_CONF` | 15 | `{root}/Curius_SHUFFLED_CONF/{Monkey}_{AgoB\|BgoA}/figures/…` |
| `ElmoBLOCKED_SpikeSortedSessions` | 6 | *(not runnable — name lacks `Elmo_BLOCKED_` prefix)* |

Confederate lists run **two timing splits** (AgoB + BgoA) plus **`comparisons`** (go-sequence and social-context axes), mirroring DUAL_NHP per-monkey layout. Trial filters: Dyadic + `{AgoB|BgoA}` + Blocked/Shuffled + actor-side reward tiers (RA or RB). L/R: actor-side choice field per session (`Al`/`Ar` or `Bl`/`Br`).

**Output layout** (example `Curius_SHUFFLED_CONF`):

```
{root_folder}/Curius_SHUFFLED_CONF/
  Curius_AgoB/figures/{original|zscored}/...
  Curius_BgoA/figures/...
  Curius_AgoB/figures/zscored/combined/*_arrays_combined_LR.pdf
  Curius_Dyadic_first_second_comparison/    # comparisons
```

```bash
python -u code/run_scripts/run_session_list.py --list
python -u code/run_scripts/run_session_list.py Curius_SHUFFLED_CONF
python -u code/run_scripts/run_session_list.py Elmo_BLOCKED_CONF --go-seq BgoA
python -u code/run_scripts/run_session_list.py Curius_SHUFFLED_CONF --steps session_lr,consistency
python -u code/run_scripts/audit_conf_sessions.py --all-conf
python -u code/run_scripts/run_comparisons.py Curius_SHUFFLED_CONF
```

Validate parsing:

```bash
cd code && python -m run_scripts.check
```

**Note:** An older monolithic run may exist at `{root}/{list_name}/figures/` (AgoB-only, pre go-seq split); re-run overwrites nothing there — new outputs go under `{Monkey}_{AgoB|BgoA}/`.

#### DUAL_NHP — paired dual-monkey export

Eight paired sessions (4 dates × 2 spike-sorted exports). Split by **recorded monkey**:

- Datetime suffix **`U`** or no `B` → **Curius** export  
- Datetime suffix **`B`** → **Elmo** export  

(Future **confederate** lists: only one monkey's neural data — sessions assigned via `.A_Curius.` / `.A_Elmo.` in the ID; empty monkey buckets are skipped.)

**Four analysis runs** (recorded monkey × timing), each with isolated output. L/R and reward follow **actor side per session** (not a fixed field per monkey name):

| Run | go_seq filter | L/R (typical) | Reward (typical) |
|---|---|---|---|
| `Curius_AgoB` | AgoB | `A_LR` when Curius is A-side in that session | `A_Reward` RA1–RA4 |
| `Curius_BgoA` | BgoA | same rule per session | same |
| `Elmo_AgoB` | AgoB | `B_LR` when Elmo is B-side; `A_LR` when A-side | RB or RA per session |
| `Elmo_BgoA` | BgoA | same rule per session | same |

Trial filters per run: Dyadic + `{AgoB|BgoA}` + actor-side RA1–RA4 or RB1–RB4 (no `conf_predictability`).

**Output layout:**

```
{root_folder}/DUAL_NHP/
  Curius_AgoB/figures/{original|zscored}/...
  Curius_BgoA/figures/...
  Elmo_AgoB/figures/...
  Elmo_BgoA/figures/...
  {Monkey}_{AgoB|BgoA}/figures/zscored/combined/
    {Condition}_{event}_A1_combined_LR.pdf      # per-channel panels
    {Condition}_{event}_arrays_combined_LR.pdf   # array-mean ± SE (array_combined step)
  {Monkey}_Dyadic_first_second_comparison/             # go-sequence comparison
    si_scatter.pdf, delta_si_heatmap.pdf, similar10/worst10 deep dives, CSVs, ...
```

```bash
# Full pipeline: 4 monkey×timing runs + array plots + first/second comparison
python -u code/run_scripts/run_session_list.py DUAL_NHP

# Subset
python -u code/run_scripts/run_session_list.py DUAL_NHP --go-seq AgoB
python -u code/run_scripts/run_session_list.py DUAL_NHP --monkey Curius --go-seq BgoA
python -u code/run_scripts/run_session_list.py DUAL_NHP --steps session_lr,consistency

# Partial reruns (one step)
python -u code/run_scripts/run_array_combined.py DUAL_NHP
python -u code/run_scripts/run_comparisons.py DUAL_NHP --monkey Curius
```

**DUAL_NHP CLI flags:** `--monkey {Curius|Elmo}`, `--go-seq {AgoB|BgoA|all}` (default `all`).

---

### Individual steps

Re-run a single pipeline step without the full runner:

| Script | Step |
|---|---|
| [`code/run_scripts/run_session_lr.py`](code/run_scripts/run_session_lr.py) | `session_lr` |
| [`code/run_scripts/run_consistency.py`](code/run_scripts/run_consistency.py) | `consistency` |
| [`code/run_scripts/run_stability_across_sessions.py`](code/run_scripts/run_stability_across_sessions.py) | `stability_across_sessions` |
| [`code/run_scripts/run_combine.py`](code/run_scripts/run_combine.py) | `combine` |
| [`code/run_scripts/run_array_combined.py`](code/run_scripts/run_array_combined.py) | `array_combined` |
| [`code/run_scripts/run_comparisons.py`](code/run_scripts/run_comparisons.py) | `comparisons` |
| [`code/run_scripts/run_best_worst.py`](code/run_scripts/run_best_worst.py) | `best_worst` |
| [`code/run_scripts/replot_consistency.py`](code/run_scripts/replot_consistency.py) | Replot heatmaps only |
| [`code/run_scripts/audit_conf_sessions.py`](code/run_scripts/audit_conf_sessions.py) | Session label audit → `audit/all_conf_audit.csv`, `audit/DUAL_NHP_audit.csv` |
| [`code/run_scripts/audit_trial_selection.py`](code/run_scripts/audit_trial_selection.py) | Actor-side vs legacy L/R trial counts → `{output}/trial_selection_{AgoB\|BgoA}.csv` |

The runners patch module globals via [`code/run_pipeline/context.py`](code/run_pipeline/context.py) (`PipelineContext`: filters, choice field from `choice_config_for_recording`, `recording_monkey`, paths). DUAL_NHP logic lives in [`code/run_pipeline/dual_nhp.py`](code/run_pipeline/dual_nhp.py).

Default is z-scored only. Pass `--also-original` on any pipeline runner, or set `RUN_BOTH_PROCESSING = True` in [`code/run_pipeline/config.py`](code/run_pipeline/config.py).

### Session audit

Label QC for confederate and DUAL_NHP lists — rewarded trial counts per `TrialSubType`, problems (missing fields, suspicious zero-rewarded counts):

```bash
python -u code/run_scripts/audit_conf_sessions.py --all-conf    # 4 _CONF lists → audit/all_conf_audit.csv
python -u code/run_scripts/audit_conf_sessions.py --dual-nhp    # DUAL_NHP (8 sessions) → audit/DUAL_NHP_audit.csv
python -u code/run_scripts/audit_conf_sessions.py --list Curius_SHUFFLED_CONF
```

CSV columns: `session_id`, `session_pair` (e.g. `A_Curius.B_MK` from the full session ID), `dataset`, `problem`, `n_SoloA`, … `_CONF` lists use list-level recording monkey; **DUAL_NHP** infers Curius vs Elmo export per session (`U`/no suffix → A+RA; `B` suffix → B+RB).

## Cross-session consistency

Nominal channel IDs (`ch001`–`ch160`) are physical electrode indices and may not reflect the same neural tissue across days (electrode drift). **Channels can be missing** in individual sessions (no file or insufficient trials); the pipeline always uses the fixed A1–A5 layout (ch001–032 … ch129–160), assigns **NaN** where data are absent, and shows **empty labelled axes** in overview plots. Stability metrics use only sessions where that channel had usable data.

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
| **Mann–Whitney p** | On per-trial window means (L vs R) | Non-parametric test of L/R difference within session. Not corrected for multiple channels. |
| **Median pairwise r** | Median Pearson r of **Δ(t)** across all session pairs for one channel | **Shape** consistency: do difference-waveforms correlate across days? High r → similar temporal profile; low/negative r → drift or remapping. |
| **ICC(2,1)** | Two-way random ICC; sessions = raters, time points = targets | **Absolute agreement** of Δ(t) across sessions (includes amplitude). ICC ≥ 0.4 is a common “fair” threshold in config. |
| **Sign concordance** | Fraction of sessions sharing the majority SI sign | **Direction** stability: does the channel stay left- or right-preferring? Reported as `n_same_sign / n_sessions` in CSV. |
| **Stable flag** | `True` iff median r ≥ 0.5 **and** ICC ≥ 0.4 **and** sign concordance ≥ 0.7 | Conservative joint criterion (all three must pass). Thresholds are configurable at top of script. |
| **Session similarity** | For each session pair: median across channels of r(Δ_session_i, Δ_session_j) | Global view: which recording days look most alike in L/R difference structure? |

**Reading SI vs Δ:** SI is a single scalar summary of bias in the analysis window; Δ(t) captures *when* during the epoch L and R diverge. A channel can have consistent SI sign but low pairwise r if waveform *shape* changes across sessions.

---

### Visualization tiers

Figures below live under:

- **Curated:** `figures/{original|zscored}/consistency/{CONDITION}/`
- **Session-list (flat):** `{root_folder}/{list_name}/figures/{original|zscored}/consistency/`  
  DUAL_NHP: `{root_folder}/DUAL_NHP/{Monkey}_{AgoB|BgoA}/figures/{original|zscored}/consistency/`

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

### Practical workflow

1. Run the appropriate runner (`code/run_scripts/run_curated.py {CONDITION}` or `code/run_scripts/run_session_list.py DUAL_NHP`).
2. **Tier 1:** Scan `si_heatmap.pdf` and `channel_stability.csv`; note channels with high sign concordance but low r (direction stable, shape not).
3. **Tier 2:** Open `delta_consensus_*.pdf` for arrays with candidate channels; confirm waveform agreement in the analysis window.
4. **Tier 3:** Use `deep_dive_ch*.pdf` to diagnose failures (gain change vs remapping vs noise).
5. **Decision:** Pool or average across sessions only for **`stable == True`** channels (or relax criteria explicitly in analysis). Exclude or treat separately all others when using nominal channel IDs.

Default stability thresholds in script: **median pairwise r ≥ 0.5**, **ICC ≥ 0.4**, **sign concordance ≥ 0.7**. Adjust `R_STABLE_THRESH`, `ICC_STABLE_THRESH`, and `SIGN_CONCORDANCE_THRESH` if your use case needs stricter or looser pooling.

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

`comparisons` runs once per monkey after the selected go-seq branches. AgoB-vs-BgoA requires `--go-seq all`; Dyadic-vs-Solo runs only for selected go-sequences and is omitted by `--dyadic-only`.

Every branch writes `pipeline_manifest.json`; every comparison writes
`comparison_manifest.json` with exact filters, dimensions, source paths, sessions,
alignment events, and the `B - A` delta definition. Re-check an existing output tree:

```bash
cd code
python -m run_scripts.plan_pipeline Elmo_BLOCKED
python -m run_scripts.audit_outputs ../figures/Elmo_BLOCKED
```

Data-dependent parity tests are opt-in:

```bash
cd code
$env:BOS_RUN_DATA_TESTS = "1"
python -m unittest tests.test_curated_trialinfo_vs_export -v
```
