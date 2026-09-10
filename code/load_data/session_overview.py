"""Per-session inclusion overview for confederate lists (Excel/CSV)."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

from analyze_decoding.config import MIN_TRIALS_PER_CONDITION
from analyze_decoding.paths import session_decode_stem
from analyze_stability.consistency import MIN_TRIALS_PER_GROUP
from analyze_stability.pref_unpref import MWU_ALPHA
from load_data.audit import session_pair_from_id
from load_data.cache import disk_cache_path, _legacy_disk_cache_path
from load_data.io import (
    INVALID_LABELS,
    discover_channel_files,
    load_trial_labels,
    session_sort_key,
)
from load_data.sessions import (
    is_confederate_list,
    list_available_session_lists,
    load_session_list,
)
from load_data.trial_selection_counts import filter_choice_counts
from openpyxl.styles import Font
from process_channels.preprocess import (
    DUAL_NHP_GO_SEQS,
    alignment_event_for_recording,
    choice_config_for_actor_side,
    recording_actor_side,
    recording_monkey_from_condition_label,
    trial_filters_for_go_seq,
    trial_filters_for_solo_from_dyadic,
)

CONF_LISTS = (
    "Elmo_BLOCKED_CONF",
    "Elmo_SHUFFLED_CONF",
    "Curius_BLOCKED_CONF",
    "Curius_SHUFFLED_CONF",
)
MUA_MIN = MIN_TRIALS_PER_GROUP
DECODE_MIN = MIN_TRIALS_PER_CONDITION

COLUMN_KEY: tuple[tuple[str, str], ...] = (
    ("dataset", "session_lists.m list name"),
    ("monkey", "Recorded monkey"),
    ("condition", "BLOCKED or SHUFFLED (trial-filter key)"),
    ("session_id", "Full session folder name"),
    ("datetime", "Date prefix"),
    ("partner", "A_*.B_* tokens from the session ID"),
    ("actor_side", "Recorded monkey's side in the session ID"),
    ("session_dir_exists", "Export folder under root_folder"),
    ("trialinfo_exists", "*.trialinfo.4python.mat"),
    ("alignment_event", "A_ or B_ InitialFixationReleaseTime_ms"),
    ("alignment_dir_exists", "Alignment folder present"),
    ("n_channel_files", "Event-aligned channel .mat files"),
    ("n_trials", "Rows in trialinfo"),
    ("n_Dyadic / n_SoloA / …", "Raw TrialSubType counts (no other filters)"),
    ("n_AgoB / n_BgoA / n_ABgo", "Raw go_seq_500_list counts"),
    ("n_Blocked / n_Shuffled / n_Free", "Raw conf_predictability_list counts"),
    ("dyadic_{go}_n / _L / _R", "Pipeline Dyadic mask, then actor L/R"),
    ("solo_{go}_n / _L / _R", "Pipeline SoloA/SoloB mask (no Blocked/Shuffled tag)"),
    ("*_mua_ok", f"L≥{MUA_MIN} and R≥{MUA_MIN} (summary/PSTH gate)"),
    ("*_decode_ok", f"L≥{DECODE_MIN} and R≥{DECODE_MIN} (Decodanda gate)"),
    ("*_exclusion", "Why that branch is empty or has too few left/right trials"),
    ("*_n_ch_cache", "Channels extracted for that branch"),
    ("*_n_ch_tuned", "Extracted channels with a left/right preference (p<0.05)"),
    ("locked_{go}_n_ch", "Channels in both Dyadic and Solo; preference taken from solo"),
    ("locked_{go}_included", "That count > 0 (used in Dyadic-vs-Solo plots)"),
    ("decode_*", "Session decode PDF present on disk"),
    ("notes", "Plain-language problems: missing files, too few trials, extract mismatch"),
)

_FIELD_PLAIN = {
    "TrialSubType_list": "trial type",
    "go_seq_500_list": "go order",
    "conf_predictability_list": "Blocked/Shuffled",
    "A_Reward_list": "A reward",
    "B_Reward_list": "B reward",
    "A_LR_pos_list": "A left/right",
    "B_LR_pos_list": "B left/right",
}


def _allowed_plain(field: str, allowed: list[str]) -> str:
    values = set(allowed)
    if values == {"Dyadic"}:
        return "Dyadic"
    if values <= {"SoloA", "SoloARewardAB", "SoloB", "SoloBRewardAB"}:
        return "solo"
    if values == {"Blocked"}:
        return "Blocked"
    if values == {"Shuffled"}:
        return "Shuffled"
    if values == {"AgoB"}:
        return "AgoB"
    if values == {"BgoA"}:
        return "BgoA"
    if field in ("A_Reward_list", "B_Reward_list"):
        return "rewarded"
    return "/".join(allowed)


def _predictability_word(condition_key: str) -> str | None:
    if condition_key.endswith("_SHUFFLED"):
        return "Shuffled"
    if condition_key.endswith("_BLOCKED"):
        return "Blocked"
    return None


def _branch_note_label(trial_type: str, condition_key: str) -> str:
    if trial_type == "Dyadic":
        pred = _predictability_word(condition_key)
        return f"Dyadic {pred}" if pred else "Dyadic"
    return "Solo"


def _count_isin(labels: dict[str, np.ndarray], field: str, values: list[str]) -> int:
    if field not in labels:
        return 0
    return int(np.isin(labels[field], values).sum())


def _peel_empty_reason(labels: dict[str, np.ndarray], filters: dict[str, list[str]]) -> str:
    """Why the trial mask is empty, in plain language."""
    n_trials = len(next(iter(labels.values())))
    mask = np.ones(n_trials, dtype=bool)
    for field, allowed in filters.items():
        if field not in labels:
            col = _FIELD_PLAIN.get(field, field)
            return f"trialinfo has no {col} column"
        before = int(mask.sum())
        mask &= np.isin(labels[field], allowed)
        mask &= ~np.isin(labels[field], list(INVALID_LABELS))
        after = int(mask.sum())
        if after == 0:
            extra = (
                f" ({before} left before this cut)"
                if before and before < n_trials
                else ""
            )
            if field in ("TrialSubType_list", "conf_predictability_list"):
                return f"no trials{extra}"
            what = _allowed_plain(field, allowed)
            if field in ("A_Reward_list", "B_Reward_list"):
                side = "A" if field.startswith("A_") else "B"
                what = f"rewarded {side}"
            return f"no {what} trials{extra}"
    for field in ("A_LR_pos_list", "B_LR_pos_list"):
        if field not in labels:
            continue
        before = int(mask.sum())
        mask &= ~np.isin(labels[field], list(INVALID_LABELS))
        if int(mask.sum()) == 0:
            side = "A" if field.startswith("A_") else "B"
            extra = f" ({before} left before this cut)" if before else ""
            return f"remaining trials have no valid {side} left/right{extra}"
    return "no trials left after filters"


def _branch_status(
    labels: dict[str, np.ndarray],
    filters: dict[str, list[str]],
    choice,
    *,
    tag: str,
    condition_key: str = "",
) -> dict[str, object]:
    counts = filter_choice_counts(
        labels, filters, choice_field=choice.field, left=choice.left, right=choice.right,
    )
    n, left_n, right_n = counts["total"], counts["left"], counts["right"]
    mua_ok = left_n >= MUA_MIN and right_n >= MUA_MIN
    decode_ok = left_n >= DECODE_MIN and right_n >= DECODE_MIN
    go_seq = tag.split("_", 1)[1]
    trial_type = "Dyadic" if tag.startswith("dyadic_") else "SoloA"
    prefix = f"{go_seq} {_branch_note_label(trial_type, condition_key)}"
    if n == 0:
        exclusion = f"{prefix}: {_peel_empty_reason(labels, filters)}"
    elif not mua_ok:
        exclusion = f"{prefix}: left={left_n}, right={right_n} (need ≥{MUA_MIN} each)"
    else:
        exclusion = ""
    if mua_ok and not decode_ok:
        decode_note = (
            f"{prefix}: left={left_n}, right={right_n} "
            f"(need ≥{DECODE_MIN} each for decode)"
        )
    else:
        decode_note = ""
    return {
        f"{tag}_n": n,
        f"{tag}_L": left_n,
        f"{tag}_R": right_n,
        f"{tag}_mua_ok": mua_ok,
        f"{tag}_decode_ok": decode_ok,
        f"{tag}_exclusion": exclusion,
        f"{tag}_decode_note": decode_note,
    }


def _is_tuned_meta(mwu_p: float, pref_side: str, *, alpha: float = MWU_ALPHA) -> bool:
    if not np.isfinite(mwu_p) or float(mwu_p) >= alpha:
        return False
    return pref_side in ("L", "R")


def _summaries_npz_path(output_base: Path, condition_label: str) -> Path | None:
    path = disk_cache_path(output_base, condition_label, True)
    if path.exists():
        return path
    legacy = _legacy_disk_cache_path(output_base, condition_label, True)
    return legacy if legacy.exists() else None


def _load_summary_index(path: Path) -> dict[str, list[tuple[int, float, str]]]:
    """session_id → list of (channel, mwu_p, pref_side). Skips PSTH arrays.

    Copy off CIFS first: npz is a zip of thousands of arrays; seeking it on
    a network mount is much slower than a sequential copy + local reads.
    """
    nested: dict[str, list[tuple[int, float, str]]] = defaultdict(list)
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as handle:
        tmp = Path(handle.name)
    try:
        shutil.copyfile(path, tmp)
        with np.load(tmp, allow_pickle=True) as data:
            n = int(data["n_summaries"])
            for i in range(n):
                sid = data[f"{i}_session_id"]
                session_id = str(sid.item()) if getattr(sid, "shape", ()) else str(sid)
                pref = data[f"{i}_pref_side"]
                pref_side = str(pref.item()) if getattr(pref, "shape", ()) else str(pref)
                nested[session_id].append(
                    (int(data[f"{i}_channel"]), float(data[f"{i}_mwu_p"]), pref_side)
                )
    finally:
        tmp.unlink(missing_ok=True)
    return nested


def _cache_counts(rows: list[tuple[int, float, str]]) -> tuple[int, int]:
    n_ch = len(rows)
    n_tuned = sum(1 for _, mwu_p, pref_side in rows if _is_tuned_meta(mwu_p, pref_side))
    return n_ch, n_tuned


def _locked_nch(
    dyadic: dict[str, list[tuple[int, float, str]]],
    solo: dict[str, list[tuple[int, float, str]]],
) -> dict[str, int]:
    out: dict[str, int] = {}
    for session_id in set(dyadic) & set(solo):
        dy_ch = {ch for ch, _, _ in dyadic[session_id]}
        n = 0
        for channel, mwu_p, pref_side in solo[session_id]:
            if channel in dy_ch and _is_tuned_meta(mwu_p, pref_side):
                n += 1
        if n:
            out[session_id] = n
    return out


def _decode_pdf(branch: Path, trial_type: str, target: str, session_id: str) -> bool:
    path = branch / trial_type / "decoding" / target / f"{session_decode_stem(session_id)}.pdf"
    return path.is_file()


def _n_channel_files(event_dir: Path) -> int:
    if not event_dir.is_dir():
        return 0
    try:
        return len(discover_channel_files(event_dir))
    except FileNotFoundError:
        return 0


def overview_row(
    *,
    list_name: str,
    session_id: str,
    session_dir: Path,
    output_root: Path,
    monkey: str,
    condition_key: str,
    caches: dict[tuple[str, str], dict[str, list[tuple[int, float, str]]]],
    locked: dict[str, dict[str, int]],
) -> dict[str, object]:
    actor_side = recording_actor_side(session_id, monkey)
    choice = choice_config_for_actor_side(actor_side)
    alignment = alignment_event_for_recording(session_id, monkey)
    event_dir = session_dir / alignment
    trialinfo = session_dir / f"{session_id}.trialinfo.4python.mat"
    notes: list[str] = []

    row: dict[str, object] = {
        "dataset": list_name,
        "monkey": monkey,
        "condition": condition_key.split("_", 1)[1],
        "session_id": session_id,
        "datetime": session_sort_key(session_id),
        "partner": session_pair_from_id(session_id),
        "actor_side": actor_side,
        "session_dir_exists": session_dir.is_dir(),
        "trialinfo_exists": trialinfo.is_file(),
        "alignment_event": alignment,
        "alignment_dir_exists": event_dir.is_dir(),
        "n_channel_files": _n_channel_files(event_dir),
    }
    if not session_dir.is_dir():
        notes.append("session folder missing")
    if not trialinfo.is_file():
        notes.append("trialinfo missing")
    if session_dir.is_dir() and not event_dir.is_dir():
        notes.append("no alignment folder")
    if event_dir.is_dir() and int(row["n_channel_files"]) == 0:
        notes.append("no channel files")

    if not trialinfo.is_file():
        row["notes"] = "; ".join(notes) if notes else ""
        return row

    labels = load_trial_labels(session_dir, session_id)
    row["n_trials"] = len(next(iter(labels.values())))
    row["n_Dyadic"] = _count_isin(labels, "TrialSubType_list", ["Dyadic"])
    row["n_SoloA"] = _count_isin(labels, "TrialSubType_list", ["SoloA"])
    row["n_SoloARewardAB"] = _count_isin(labels, "TrialSubType_list", ["SoloARewardAB"])
    row["n_SoloB"] = _count_isin(labels, "TrialSubType_list", ["SoloB"])
    row["n_SoloBRewardAB"] = _count_isin(labels, "TrialSubType_list", ["SoloBRewardAB"])
    row["n_SemiSolo"] = _count_isin(labels, "TrialSubType_list", ["SemiSolo"])
    row["n_AgoB"] = _count_isin(labels, "go_seq_500_list", ["AgoB"])
    row["n_BgoA"] = _count_isin(labels, "go_seq_500_list", ["BgoA"])
    row["n_ABgo"] = _count_isin(labels, "go_seq_500_list", ["ABgo"])
    row["n_Blocked"] = _count_isin(labels, "conf_predictability_list", ["Blocked"])
    row["n_Shuffled"] = _count_isin(labels, "conf_predictability_list", ["Shuffled"])
    row["n_Free"] = _count_isin(labels, "conf_predictability_list", ["Free"])

    for go_seq in DUAL_NHP_GO_SEQS:
        dyadic_filters = trial_filters_for_go_seq(condition_key, go_seq, actor_side)
        solo_filters = trial_filters_for_solo_from_dyadic(dyadic_filters, actor_side)
        row.update(
            _branch_status(
                labels, dyadic_filters, choice,
                tag=f"dyadic_{go_seq}", condition_key=condition_key,
            )
        )
        row.update(
            _branch_status(
                labels, solo_filters, choice,
                tag=f"solo_{go_seq}", condition_key=condition_key,
            )
        )

        go_dir = output_root / f"{monkey}_{go_seq}"
        for branch_name, trial_type in (("dyadic", "Dyadic"), ("solo", "SoloA")):
            cache_rows = caches.get((go_seq, trial_type), {}).get(session_id, [])
            n_ch, n_tuned = _cache_counts(cache_rows)
            row[f"{branch_name}_{go_seq}_n_ch_cache"] = n_ch
            row[f"{branch_name}_{go_seq}_n_ch_tuned"] = n_tuned
            in_cache = n_ch > 0
            row[f"{branch_name}_{go_seq}_in_cache"] = in_cache
            mua_ok = bool(row[f"{branch_name}_{go_seq}_mua_ok"])
            label = _branch_note_label(trial_type, condition_key)
            if mua_ok and not in_cache:
                notes.append(
                    f"{go_seq} {label}: enough left/right trials but no channels were extracted"
                )
            if in_cache and not mua_ok:
                notes.append(
                    f"{go_seq} {label}: channels extracted despite too few left/right trials"
                )

        locked_n = int(locked.get(go_seq, {}).get(session_id, 0))
        row[f"locked_{go_seq}_n_ch"] = locked_n
        row[f"locked_{go_seq}_included"] = locked_n > 0
        both_extracted = bool(row[f"dyadic_{go_seq}_in_cache"] and row[f"solo_{go_seq}_in_cache"])
        if both_extracted and locked_n == 0:
            dy_label = _branch_note_label("Dyadic", condition_key)
            notes.append(
                f"{go_seq}: no channels with a Solo left/right preference "
                f"that also appear in {dy_label}"
            )

        row[f"decode_dyadic_{go_seq}_same_diff"] = _decode_pdf(
            go_dir, "Dyadic", "same_diff", session_id,
        )
        row[f"decode_dyadic_{go_seq}_choice_ab"] = _decode_pdf(
            go_dir, "Dyadic", "choice_ab_grid", session_id,
        )
        row[f"decode_solo_{go_seq}_actor"] = _decode_pdf(
            go_dir, "SoloA", "actor_choice", session_id,
        )

        for key in (f"dyadic_{go_seq}_exclusion", f"solo_{go_seq}_exclusion"):
            if row[key]:
                notes.append(str(row[key]))
        for key in (f"dyadic_{go_seq}_decode_note", f"solo_{go_seq}_decode_note"):
            if row[key]:
                notes.append(str(row[key]))

    seen: set[str] = set()
    unique_notes: list[str] = []
    for item in notes:
        if item not in seen:
            seen.add(item)
            unique_notes.append(item)
    row["notes"] = "; ".join(unique_notes)
    return row


def _load_list_caches(output_root: Path, monkey: str, condition_label_by_go: dict[str, str]):
    caches: dict[tuple[str, str], dict[str, list[tuple[int, float, str]]]] = {}
    locked: dict[str, dict[str, int]] = {}
    for go_seq in DUAL_NHP_GO_SEQS:
        label = condition_label_by_go[go_seq]
        go_dir = output_root / f"{monkey}_{go_seq}"
        dy_path = _summaries_npz_path(go_dir / "Dyadic", label)
        so_path = _summaries_npz_path(go_dir / "SoloA", label)
        print(f"  index {monkey}_{go_seq} Dyadic…", flush=True)
        dyadic = _load_summary_index(dy_path) if dy_path else {}
        print(f"  index {monkey}_{go_seq} SoloA…", flush=True)
        solo = _load_summary_index(so_path) if so_path else {}
        caches[(go_seq, "Dyadic")] = dyadic
        caches[(go_seq, "SoloA")] = solo
        locked[go_seq] = _locked_nch(dyadic, solo)
    return caches, locked


def build_overview_rows(
    *,
    session_lists_path: str | Path = "session_lists.m",
    list_names: tuple[str, ...] = CONF_LISTS,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for list_name in list_names:
        if not is_confederate_list(list_name):
            raise ValueError(f"Not a confederate list: {list_name!r}")
        cfg = load_session_list(list_name, session_lists_path)
        monkey = recording_monkey_from_condition_label(cfg.condition_key)
        print(f"Loading caches {list_name}…", flush=True)
        caches, locked = _load_list_caches(
            cfg.output_folder,
            monkey,
            {go: f"{monkey}_{go}" for go in DUAL_NHP_GO_SEQS},
        )
        for session_id in cfg.session_ids:
            rows.append(
                overview_row(
                    list_name=list_name,
                    session_id=session_id,
                    session_dir=cfg.root_folder / session_id,
                    output_root=cfg.output_folder,
                    monkey=monkey,
                    condition_key=cfg.condition_key,
                    caches=caches,
                    locked=locked,
                )
            )
        print(f"  {list_name}: {len(cfg.session_ids)} sessions", flush=True)
    return sessions_frame(rows)


def sessions_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    table = pd.DataFrame.from_records(rows)
    bool_cols = [
        col for col in table.columns
        if col.endswith(("_ok", "_exists", "_included", "_in_cache"))
        or col.startswith("decode_")
    ]
    int_cols = [
        col for col in table.columns
        if col.startswith("n_")
        or col.endswith(("_n", "_L", "_R", "_n_ch", "_n_ch_cache", "_n_ch_tuned", "_n_ch_files"))
        or col == "n_channel_files"
        or col.endswith("_n_ch")
    ]
    for col in bool_cols:
        table[col] = table[col].fillna(False).astype(bool)
    for col in int_cols:
        if col in table.columns:
            table[col] = pd.to_numeric(table[col], errors="coerce").fillna(0).astype(int)
    if "notes" in table.columns:
        table["notes"] = table["notes"].fillna("").astype(str)
    return table


def dataset_summary(sessions: pd.DataFrame) -> pd.DataFrame:
    records = []
    for dataset, sub in sessions.groupby("dataset", sort=False):
        rec = {
            "dataset": dataset,
            "n_listed": len(sub),
            "n_session_dir": int(sub["session_dir_exists"].sum()),
            "n_trialinfo": int(sub["trialinfo_exists"].sum()),
        }
        for go_seq in DUAL_NHP_GO_SEQS:
            rec[f"dyadic_{go_seq}_mua_ok"] = int(sub[f"dyadic_{go_seq}_mua_ok"].sum())
            rec[f"solo_{go_seq}_mua_ok"] = int(sub[f"solo_{go_seq}_mua_ok"].sum())
            rec[f"dyadic_{go_seq}_in_cache"] = int(sub[f"dyadic_{go_seq}_in_cache"].sum())
            rec[f"solo_{go_seq}_in_cache"] = int(sub[f"solo_{go_seq}_in_cache"].sum())
            rec[f"locked_{go_seq}_included"] = int(sub[f"locked_{go_seq}_included"].sum())
            rec[f"dyadic_{go_seq}_decode_ok"] = int(sub[f"dyadic_{go_seq}_decode_ok"].sum())
            rec[f"decode_same_diff_{go_seq}"] = int(sub[f"decode_dyadic_{go_seq}_same_diff"].sum())
        rec["n_with_notes"] = int((sub["notes"].astype(str).str.len() > 0).sum())
        records.append(rec)
    return pd.DataFrame.from_records(records)


def column_key_frame() -> pd.DataFrame:
    return pd.DataFrame(COLUMN_KEY, columns=["column", "meaning"])


def write_overview_excel(sessions: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = dataset_summary(sessions)
    key = column_key_frame()
    drop_cols = [c for c in sessions.columns if c.endswith("_decode_note")]
    table = sessions.drop(columns=drop_cols, errors="ignore")
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        table.to_excel(writer, sheet_name="sessions", index=False)
        summary.to_excel(writer, sheet_name="dataset_summary", index=False)
        key.to_excel(writer, sheet_name="column_key", index=False)
        for sheet_name in ("sessions", "dataset_summary", "column_key"):
            ws = writer.sheets[sheet_name]
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            ws.auto_filter.ref = ws.calculate_dimension()
            for cell in ws[1]:
                cell.font = Font(bold=True)
            for column in ws.columns:
                letter = column[0].column_letter
                header = str(column[0].value or "")
                width = min(max(len(header) + 2, 12), 42)
                if header == "notes":
                    width = 60
                ws.column_dimensions[letter].width = width
    return path


def available_conf_lists(session_lists_path: str | Path = "session_lists.m") -> list[str]:
    names = [
        name
        for name in list_available_session_lists(session_lists_path)
        if is_confederate_list(name) and name in CONF_LISTS
    ]
    return names or list(CONF_LISTS)
