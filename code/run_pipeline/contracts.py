"""Pure run-matrix contracts shared by orchestration, dry runs, and tests."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from process_channels.preprocess import DUAL_NHP_GO_SEQS


@dataclass(frozen=True)
class ComparisonRequest:
    comparison_axis: str
    social_context: str
    go_seq: str | None

    def as_dict(self) -> dict:
        return asdict(self)


def comparison_requests(
    *,
    include_solo: bool,
    go_seq: str,
) -> tuple[ComparisonRequest, ...]:
    """Return the exact supported comparison matrix for selected pipeline branches."""
    if go_seq != "all" and go_seq not in DUAL_NHP_GO_SEQS:
        raise ValueError(f"Unknown go sequence {go_seq!r}")

    requests: list[ComparisonRequest] = []
    if go_seq == "all":
        requests.append(
            ComparisonRequest("go_sequence", "dyadic", None)
        )
        if include_solo:
            requests.append(
                ComparisonRequest("go_sequence", "solo", None)
            )

    if include_solo:
        selected = DUAL_NHP_GO_SEQS if go_seq == "all" else (go_seq,)
        requests.extend(
            ComparisonRequest("social_context", "dyadic", selected_go)
            for selected_go in selected
        )
    return tuple(requests)
