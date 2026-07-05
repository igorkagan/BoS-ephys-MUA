"""Parse MATLAB session list definitions from session_lists.m."""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path

ARRAY_START_RE = re.compile(r"^\s*(\w+)\s*=\s*\{")
ROOT_FOLDER_RE = re.compile(r"^\s*root_folder\s*=\s*'([^']*)'\s*;?\s*$")
SESSION_QUOTED_RE = re.compile(r"'([^']+)'")
CONDITION_PREFIX_RE = re.compile(r"^(Elmo|Curius)_(BLOCKED|SHUFFLED)")
CONFEDERATE_LIST_RE = re.compile(r"^(Elmo|Curius)_(BLOCKED|SHUFFLED)_")

DUAL_NHP_LIST_NAME = "DUAL_NHP"
DUAL_NHP_MONKEYS = ("Curius", "Elmo")


@dataclass(frozen=True)
class SessionListConfig:
    list_name: str
    root_folder: Path
    output_folder: Path
    session_ids: list[str]
    condition_key: str


def condition_key_from_list_name(list_name: str) -> str:
    """Derive trial-filter condition key (e.g. Elmo_BLOCKED) from a list name."""
    match = CONDITION_PREFIX_RE.match(list_name)
    if not match:
        raise ValueError(
            f"Cannot derive condition key from list name {list_name!r}; "
            "expected prefix Elmo_BLOCKED, Elmo_SHUFFLED, Curius_BLOCKED, or Curius_SHUFFLED"
        )
    return f"{match.group(1)}_{match.group(2)}"


def is_dual_nhp_list(list_name: str) -> bool:
    return list_name == DUAL_NHP_LIST_NAME


def is_confederate_list(list_name: str) -> bool:
    """True for Elmo/Curius BLOCKED/SHUFFLED confederate lists (not DUAL_NHP)."""
    return bool(CONFEDERATE_LIST_RE.match(list_name))


def monkey_for_dual_session(session_id: str) -> str:
    """Return owning monkey for a session export (paired U/B or confederate)."""
    from bos_mua.preprocess import recording_monkey_from_session_id

    return recording_monkey_from_session_id(session_id)


def split_sessions_by_recording_monkey(session_ids: list[str]) -> dict[str, list[str]]:
    """Split sessions by recorded monkey, preserving order."""
    split: dict[str, list[str]] = {monkey: [] for monkey in DUAL_NHP_MONKEYS}
    for session_id in session_ids:
        split[monkey_for_dual_session(session_id)].append(session_id)
    return split


def split_dual_nhp_sessions(session_ids: list[str]) -> dict[str, list[str]]:
    """Split DUAL_NHP sessions into per-monkey lists, preserving order."""
    return split_sessions_by_recording_monkey(session_ids)


def _validate_dual_nhp_split(split: dict[str, list[str]]) -> None:
    for monkey in DUAL_NHP_MONKEYS:
        if not split[monkey]:
            warnings.warn(f"DUAL_NHP: no sessions assigned to {monkey}")
    if len(split["Curius"]) != len(split["Elmo"]):
        warnings.warn(
            f"DUAL_NHP: uneven split (Curius={len(split['Curius'])}, "
            f"Elmo={len(split['Elmo'])})"
        )


def _parse_root_folder(text: str) -> Path:
    for line in text.splitlines():
        match = ROOT_FOLDER_RE.match(line)
        if match:
            return Path(match.group(1))
    raise ValueError("session_lists.m: missing root_folder = '...'")


def _sessions_from_array_block(lines: list[str]) -> list[str]:
    sessions: list[str] = []
    for line in lines:
        code_part = line.split("%", 1)[0].replace("...", "")
        for session_id in SESSION_QUOTED_RE.findall(code_part):
            session_id = session_id.strip()
            if session_id:
                sessions.append(session_id)
    return sessions


def _parse_named_arrays(text: str) -> dict[str, list[str]]:
    arrays: dict[str, list[str]] = {}
    current_name: str | None = None
    block_lines: list[str] = []

    for line in text.splitlines():
        start_match = ARRAY_START_RE.match(line)
        if start_match:
            if current_name is not None:
                arrays[current_name] = _sessions_from_array_block(block_lines)
            current_name = start_match.group(1)
            block_lines = []
            if "};" in line:
                arrays[current_name] = _sessions_from_array_block([line])
                current_name = None
                block_lines = []
            continue

        if current_name is not None:
            block_lines.append(line)
            if "};" in line:
                arrays[current_name] = _sessions_from_array_block(block_lines)
                current_name = None
                block_lines = []

    return arrays


def summarize_session_lists(
    path: str | Path = "session_lists.m",
) -> list[tuple[str, int, Path, str]]:
    """Return (list_name, n_sessions, output_folder, detail) for each named array."""
    list_path = Path(path)
    text = list_path.read_text(encoding="utf-8")
    root_folder = _parse_root_folder(text)
    arrays = _parse_named_arrays(text)
    rows: list[tuple[str, int, Path, str]] = []
    for name, session_ids in sorted(arrays.items()):
        output_folder = root_folder / name
        detail = ""
        if is_dual_nhp_list(name):
            split = split_dual_nhp_sessions(session_ids)
            detail = f"Curius: {len(split['Curius'])}, Elmo: {len(split['Elmo'])}"
        elif is_confederate_list(name):
            from bos_mua.preprocess import recording_monkey_from_condition_label

            monkey = recording_monkey_from_condition_label(condition_key_from_list_name(name))
            detail = f"{monkey}: AgoB + BgoA runs"
        rows.append((name, len(session_ids), output_folder, detail))
    return rows


def list_available_session_lists(path: str | Path = "session_lists.m") -> list[str]:
    """Return all named cell-array list names in session_lists.m."""
    text = Path(path).read_text(encoding="utf-8")
    return sorted(_parse_named_arrays(text).keys())


def load_session_list(
    list_name: str,
    path: str | Path = "session_lists.m",
) -> SessionListConfig:
    """Load one session list definition from session_lists.m."""
    list_path = Path(path)
    text = list_path.read_text(encoding="utf-8")
    arrays = _parse_named_arrays(text)
    if list_name not in arrays:
        available = ", ".join(sorted(arrays))
        raise KeyError(f"Unknown session list {list_name!r}. Available: {available}")

    root_folder = _parse_root_folder(text)
    if is_dual_nhp_list(list_name):
        condition_key = DUAL_NHP_LIST_NAME
    else:
        condition_key = condition_key_from_list_name(list_name)

    return SessionListConfig(
        list_name=list_name,
        root_folder=root_folder,
        output_folder=root_folder / list_name,
        session_ids=arrays[list_name],
        condition_key=condition_key,
    )


def load_dual_nhp_configs(
    path: str | Path = "session_lists.m",
) -> tuple[SessionListConfig, dict[str, list[str]]]:
    """Load DUAL_NHP list and split sessions by owning monkey."""
    cfg = load_session_list(DUAL_NHP_LIST_NAME, path)
    split = split_dual_nhp_sessions(cfg.session_ids)
    _validate_dual_nhp_split(split)
    return cfg, split
