"""Audit session lists: trial-type counts and label problems."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bos_mua.io import load_trial_labels
from bos_mua.preprocess import (
    recording_monkey_from_condition_label,
    recording_monkey_from_session_id,
)
from bos_mua.session_lists import (
    DUAL_NHP_LIST_NAME,
    condition_key_from_list_name,
    is_confederate_list,
    load_session_list,
)

REWARDED_A = frozenset({"RA1", "RA2", "RA3", "RA4"})
REWARDED_B = frozenset({"RB1", "RB2", "RB3", "RB4"})
TRIALINFO_NAME = "trialinfo.4python.mat"

TRIAL_TYPES: tuple[str, ...] = (
    "Dyadic",
    "SemiSolo",
    "SoloA",
    "SoloARewardAB",
    "SoloB",
    "SoloBRewardAB",
)

COUNT_COLUMNS: tuple[str, ...] = tuple(f"n_{t}" for t in TRIAL_TYPES)

_SOLO_A_TYPES = frozenset({"SoloA", "SoloARewardAB"})
_SOLO_B_TYPES = frozenset({"SoloB", "SoloBRewardAB"})
_DYADIC_TYPES = frozenset({"Dyadic", "SemiSolo"})


def session_id_prefix(session_id: str) -> str:
    """Return datetime prefix only (e.g. 20230607T115959)."""
    return session_id.split(".", 1)[0]


def session_pair_from_id(session_id: str) -> str:
    """Return A-side and B-side monkey tokens (e.g. A_Curius.B_VC)."""
    parts = session_id.split(".")
    if len(parts) >= 3:
        return f"{parts[1]}.{parts[2]}"
    return ""


def reward_field_for_trial_type(trial_type: str, list_monkey: str) -> str:
    """Return A_Reward_list or B_Reward_list for a trial subtype."""
    if trial_type in _SOLO_A_TYPES:
        return "A_Reward_list"
    if trial_type in _SOLO_B_TYPES:
        return "B_Reward_list"
    if trial_type in _DYADIC_TYPES:
        return "A_Reward_list" if list_monkey == "Curius" else "B_Reward_list"
    raise ValueError(f"Unknown trial type: {trial_type!r}")


def list_monkey_from_list_name(list_name: str) -> str:
    condition_key = condition_key_from_list_name(list_name)
    return recording_monkey_from_condition_label(condition_key)


def recording_monkey_for_audit(session_id: str, list_monkey: str | None) -> str:
    if list_monkey is not None:
        return list_monkey
    return recording_monkey_from_session_id(session_id)


@dataclass(frozen=True)
class SessionAuditRow:
    session_id: str
    session_pair: str
    dataset: str
    problem: str
    n_Dyadic: str
    n_SemiSolo: str
    n_SoloA: str
    n_SoloARewardAB: str
    n_SoloB: str
    n_SoloBRewardAB: str

    def count_values(self) -> tuple[str, ...]:
        return (
            self.n_Dyadic,
            self.n_SemiSolo,
            self.n_SoloA,
            self.n_SoloARewardAB,
            self.n_SoloB,
            self.n_SoloBRewardAB,
        )


def rewarded_labels_for_field(reward_field: str) -> frozenset[str]:
    if reward_field == "A_Reward_list":
        return REWARDED_A
    if reward_field == "B_Reward_list":
        return REWARDED_B
    raise ValueError(f"Unknown reward field: {reward_field!r}")


def _missing_field_message(field: str) -> str:
    return f"{field} absent from {TRIALINFO_NAME}"


def _suspicious_zero_rewarded(trial_type: str, n_trials: int) -> str:
    return (
        f"suspicious: {n_trials} {trial_type} trials but 0 rewarded "
        f"(check RA/RB labels)"
    )


def _empty_row(session_id: str, dataset: str, problem: str) -> SessionAuditRow:
    return SessionAuditRow(
        session_id=session_id_prefix(session_id),
        session_pair=session_pair_from_id(session_id),
        dataset=dataset,
        problem=problem,
        n_Dyadic="-",
        n_SemiSolo="-",
        n_SoloA="-",
        n_SoloARewardAB="-",
        n_SoloB="-",
        n_SoloBRewardAB="-",
    )


def _count_rewarded_for_type(
    labels: dict[str, np.ndarray],
    trial_type: str,
    list_monkey: str,
) -> tuple[str, list[str]]:
    """Return display value and problem strings for one trial subtype."""
    if "TrialSubType_list" not in labels:
        return "-", [_missing_field_message("TrialSubType_list")]

    subtypes = labels["TrialSubType_list"].astype(str)
    type_mask = subtypes == trial_type
    n_trials = int(np.sum(type_mask))
    if n_trials == 0:
        return "-", []

    reward_field = reward_field_for_trial_type(trial_type, list_monkey)
    if reward_field not in labels:
        return "-", [_missing_field_message(reward_field)]

    rewards = labels[reward_field].astype(str)
    rewarded = rewarded_labels_for_field(reward_field)
    count = int(np.sum(type_mask & np.isin(rewards, list(rewarded))))
    problems: list[str] = []
    if count == 0:
        problems.append(_suspicious_zero_rewarded(trial_type, n_trials))
    return str(count), problems


def _merge_problems(problems: list[str]) -> str:
    seen: set[str] = set()
    unique: list[str] = []
    for problem in problems:
        if problem not in seen:
            seen.add(problem)
            unique.append(problem)
    return "-" if not unique else "; ".join(unique)


def audit_labels(
    labels: dict[str, np.ndarray],
    *,
    session_id: str,
    dataset: str,
    list_monkey: str | None,
) -> SessionAuditRow:
    """Audit one session from an already-loaded label dict."""
    monkey = recording_monkey_for_audit(session_id, list_monkey)
    problems: list[str] = []
    counts: dict[str, str] = {}

    for trial_type in TRIAL_TYPES:
        value, type_problems = _count_rewarded_for_type(labels, trial_type, monkey)
        counts[f"n_{trial_type}"] = value
        problems.extend(type_problems)

    return SessionAuditRow(
        session_id=session_id_prefix(session_id),
        session_pair=session_pair_from_id(session_id),
        dataset=dataset,
        problem=_merge_problems(problems),
        n_Dyadic=counts["n_Dyadic"],
        n_SemiSolo=counts["n_SemiSolo"],
        n_SoloA=counts["n_SoloA"],
        n_SoloARewardAB=counts["n_SoloARewardAB"],
        n_SoloB=counts["n_SoloB"],
        n_SoloBRewardAB=counts["n_SoloBRewardAB"],
    )


def audit_session(
    session_id: str,
    *,
    dataset: str,
    session_dir: Path,
    list_monkey: str | None = None,
) -> SessionAuditRow:
    """Audit one session on disk."""
    if not session_dir.is_dir():
        return _empty_row(session_id, dataset, "session dir missing")

    trialinfo_path = session_dir / f"{session_id}.{TRIALINFO_NAME}"
    if not trialinfo_path.is_file():
        return _empty_row(session_id, dataset, f"{TRIALINFO_NAME} missing")

    labels = load_trial_labels(session_dir, session_id)
    try:
        return audit_labels(
            labels,
            session_id=session_id,
            dataset=dataset,
            list_monkey=list_monkey,
        )
    except Exception as exc:
        return _empty_row(session_id, dataset, f"label load error: {exc}")


def audit_session_list(
    list_name: str,
    *,
    session_lists_path: str | Path = "session_lists.m",
) -> list[SessionAuditRow]:
    """Audit all sessions in one confederate list."""
    if not is_confederate_list(list_name):
        raise ValueError(f"Not a confederate list: {list_name!r}")

    cfg = load_session_list(list_name, session_lists_path)
    list_monkey = list_monkey_from_list_name(list_name)
    return [
        audit_session(
            session_id,
            dataset=list_name,
            session_dir=cfg.root_folder / session_id,
            list_monkey=list_monkey,
        )
        for session_id in cfg.session_ids
    ]


def audit_dual_nhp_list(
    *,
    session_lists_path: str | Path = "session_lists.m",
) -> list[SessionAuditRow]:
    """Audit DUAL_NHP sessions with per-export recording monkey."""
    cfg = load_session_list(DUAL_NHP_LIST_NAME, session_lists_path)
    return [
        audit_session(
            session_id,
            dataset=DUAL_NHP_LIST_NAME,
            session_dir=cfg.root_folder / session_id,
            list_monkey=None,
        )
        for session_id in cfg.session_ids
    ]


def audit_all_conf_lists(
    *,
    session_lists_path: str | Path = "session_lists.m",
) -> list[SessionAuditRow]:
    """Audit every confederate list in session_lists.m."""
    from bos_mua.session_lists import list_available_session_lists

    rows: list[SessionAuditRow] = []
    for list_name in list_available_session_lists(session_lists_path):
        if is_confederate_list(list_name):
            rows.extend(audit_session_list(list_name, session_lists_path=session_lists_path))
    return rows


def _row_fieldnames() -> list[str]:
    return ["session_id", "session_pair", "dataset", "problem", *COUNT_COLUMNS]


def _row_to_dict(row: SessionAuditRow) -> dict[str, str]:
    return {
        "session_id": row.session_id,
        "session_pair": row.session_pair,
        "dataset": row.dataset,
        "problem": row.problem,
        "n_Dyadic": row.n_Dyadic,
        "n_SemiSolo": row.n_SemiSolo,
        "n_SoloA": row.n_SoloA,
        "n_SoloARewardAB": row.n_SoloARewardAB,
        "n_SoloB": row.n_SoloB,
        "n_SoloBRewardAB": row.n_SoloBRewardAB,
    }


def write_csv(rows: list[SessionAuditRow], path: str | Path) -> Path:
    """Write audit rows to CSV."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_row_fieldnames())
        writer.writeheader()
        for row in rows:
            writer.writerow(_row_to_dict(row))
    return out_path


def format_markdown(rows: list[SessionAuditRow], *, title: str | None = None) -> str:
    """Format audit rows as a markdown table."""
    headers = _row_fieldnames()
    lines: list[str] = []
    if title:
        lines.append(f"## {title}")
        lines.append("")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        values = [_row_to_dict(row)[h] for h in headers]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)
