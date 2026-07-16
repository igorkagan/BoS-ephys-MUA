#!/usr/bin/env python3
"""Report per-session trial counts under actor-side pipeline selection rules."""



from __future__ import annotations



import _bootstrap  # noqa: F401

import argparse

import csv

from pathlib import Path



from load_data.io import load_trial_labels

from load_data.trial_selection_counts import filter_choice_counts

from process_channels.preprocess import (

    ChoiceConfig,

    choice_config_for_actor_side,

    recording_actor_side,

    recording_monkey_from_condition_label,

    trial_filters_for_go_seq,

    trial_filters_for_solo_from_dyadic,

)

from run_pipeline.curated import curated_condition_output, curated_data_root, discover_curated_sessions

from load_data.sessions import DUAL_NHP_LIST_NAME, load_dual_nhp_configs



REPO = Path(__file__).resolve().parents[2]





def _pre_fix_monkey_name_choice(recording_monkey: str) -> ChoiceConfig:

    """Old bug (monkey name instead of actor side) — for audit comparison only."""

    if recording_monkey == "Elmo":

        return ChoiceConfig(field="B_LR_pos_list", left=["Bl"], right=["Br"])

    if recording_monkey == "Curius":

        return choice_config_for_actor_side("A")

    raise ValueError(f"Unknown monkey: {recording_monkey!r}")





def audit_session(

    session_dir: Path,

    session_id: str,

    recording_monkey: str,

    condition_key: str,

    go_seq: str,

) -> dict[str, str | int]:

    labels = load_trial_labels(session_dir, session_id)

    actor = recording_actor_side(session_id, recording_monkey)

    correct = choice_config_for_actor_side(actor)

    legacy = _pre_fix_monkey_name_choice(recording_monkey)

    dyadic = trial_filters_for_go_seq(condition_key, go_seq, actor)

    solo = trial_filters_for_solo_from_dyadic(dyadic, actor)



    corr_dy = filter_choice_counts(

        labels, dyadic, choice_field=correct.field, left=correct.left, right=correct.right,

    )

    leg_dy = filter_choice_counts(

        labels, dyadic, choice_field=legacy.field, left=legacy.left, right=legacy.right,

    )

    corr_so = filter_choice_counts(

        labels, solo, choice_field=correct.field, left=correct.left, right=correct.right,

    )

    leg_so = filter_choice_counts(

        labels, solo, choice_field=legacy.field, left=legacy.left, right=legacy.right,

    )



    return {

        "session_id": session_id.split(".")[0],

        "actor_side": actor,

        "dyadic_base": corr_dy["total"],

        "dyadic_L_correct": corr_dy["left"],

        "dyadic_R_correct": corr_dy["right"],

        "dyadic_L_legacy": leg_dy["left"],

        "dyadic_R_legacy": leg_dy["right"],

        "solo_base": corr_so["total"],

        "solo_L_correct": corr_so["left"],

        "solo_R_correct": corr_so["right"],

        "solo_L_legacy": leg_so["left"],

        "solo_R_legacy": leg_so["right"],

        "legacy_mismatch": int(

            (corr_dy["left"], corr_dy["right"]) != (leg_dy["left"], leg_dy["right"])

            or (corr_so["left"], corr_so["right"]) != (leg_so["left"], leg_so["right"])

        ),

    }





def audit_curated(condition: str, go_seq: str, data_root: Path | None) -> list[dict]:

    monkey = recording_monkey_from_condition_label(condition)

    root = curated_data_root(data_root)

    rows = []

    for sid in discover_curated_sessions(condition, data_root=root):

        session_dir = root / condition / sid

        rows.append(audit_session(session_dir, sid, monkey, condition, go_seq))

    return rows





def audit_dual_nhp(session_lists: Path, go_seq: str) -> list[dict]:

    cfg, split = load_dual_nhp_configs(session_lists)

    rows = []

    for monkey, sids in split.items():

        for sid in sids:

            rows.append(

                audit_session(cfg.root_folder / sid, sid, monkey, DUAL_NHP_LIST_NAME, go_seq)

            )

    return rows





def write_csv(rows: list[dict], path: Path) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:

        return

    with path.open("w", newline="", encoding="utf-8") as fh:

        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))

        writer.writeheader()

        writer.writerows(rows)





def main() -> None:

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("target", choices=["curated", "dual_nhp"])

    parser.add_argument("--condition", default="Elmo_BLOCKED")

    parser.add_argument("--go-seq", default="AgoB", choices=["AgoB", "BgoA"])

    parser.add_argument("--session-lists", type=Path, default=REPO / "session_lists.m")

    parser.add_argument("--output", type=Path, default=None)

    args = parser.parse_args()



    if args.target == "curated":

        rows = audit_curated(args.condition, args.go_seq, None)

        default_out = curated_condition_output(args.condition) / f"trial_selection_{args.go_seq}.csv"

    else:

        cfg, _ = load_dual_nhp_configs(args.session_lists)

        rows = audit_dual_nhp(args.session_lists, args.go_seq)

        default_out = cfg.output_folder / f"trial_selection_{args.go_seq}.csv"



    out = args.output or default_out

    write_csv(rows, out)

    mism = sum(int(r["legacy_mismatch"]) for r in rows)

    print(f"Wrote {len(rows)} sessions to {out}")

    print(f"Legacy monkey-name choice mismatch on {mism}/{len(rows)} sessions")





if __name__ == "__main__":

    main()

