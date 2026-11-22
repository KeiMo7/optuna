"""Persisted, queryable search-space snapshots per round.

Snapshots live in the study's system attributes under a single key, so they
behave identically on the in-memory, RDB and journal backends and require no
storage-schema change. Old studies that predate this module simply have no
snapshot key and open read-only without error.

A snapshot records the union of distributions suggested in one round, the
number of the first trial run under that round, and the out-of-range markers
produced against the current round's space. Historical trials are never
rewritten: markers reference trial numbers only.
"""

from __future__ import annotations

import json
from typing import cast
from typing import TYPE_CHECKING

from optuna.distributions import distribution_to_json
from optuna.distributions import json_to_distribution


if TYPE_CHECKING:
    from optuna.distributions import BaseDistribution
    from optuna.study import Study

_SNAPSHOT_KEY = "evolving_space:snapshots"
_BASE_SPACE_KEY = "evolving_space:base_space"
_FIRST_ROUND_OPEN_KEY = "evolving_space:first_round_open"
_NEXT_ROUND_FORCED_KEY = "evolving_space:next_round_forced"


def _serialize_space(space: dict[str, BaseDistribution]) -> dict[str, str]:
    return {name: distribution_to_json(dist) for name, dist in sorted(space.items())}


def _deserialize_space(raw: dict[str, str]) -> dict[str, BaseDistribution]:
    return {name: json_to_distribution(payload) for name, payload in raw.items()}


def load_base_space(study: Study) -> dict[str, BaseDistribution] | None:
    """Return the union of the first round's distributions.

    Conditional parameters appear already in the first round's union, so any
    later trial whose space is a subset of this base space is taking a
    conditional branch rather than deleting a parameter.
    """

    raw = study._storage.get_study_system_attrs(study._study_id).get(_BASE_SPACE_KEY)
    if raw is None:
        return None
    return _deserialize_space(json.loads(raw))


def write_base_space(study: Study, space: dict[str, BaseDistribution]) -> None:
    study._storage.set_study_system_attr(
        study._study_id, _BASE_SPACE_KEY, json.dumps(_serialize_space(space))
    )


def is_first_round_open(study: Study) -> bool:
    """Whether the first round is still accumulating conditional parameters."""

    return bool(
        study._storage.get_study_system_attrs(study._study_id).get(_FIRST_ROUND_OPEN_KEY, False)
    )


def set_first_round_open(study: Study, open_flag: bool) -> None:
    study._storage.set_study_system_attr(study._study_id, _FIRST_ROUND_OPEN_KEY, open_flag)


def mark_new_round(study: Study) -> None:
    """Declare that the next finished trial starts a new search-space round.

    The sampler normally detects round transitions automatically from the
    parameter sets it observes. Deleting a parameter that every earlier trial
    used is indistinguishable from the first sighting of a conditional branch
    from the data alone, so this explicit marker provides a deterministic
    escape hatch. The flag is consumed by the next finished trial.
    """

    study._storage.set_study_system_attr(study._study_id, _NEXT_ROUND_FORCED_KEY, True)


def consume_new_round_marker(study: Study) -> bool:
    attrs = study._storage.get_study_system_attrs(study._study_id)
    forced = bool(attrs.get(_NEXT_ROUND_FORCED_KEY, False))
    if forced:
        study._storage.set_study_system_attr(study._study_id, _NEXT_ROUND_FORCED_KEY, False)
    return forced


def load_snapshots(study: Study) -> list[dict[str, object]]:
    """Return all snapshots recorded for ``study`` (oldest first).

    Returns an empty list for studies written before this feature, so reading
    an old storage never raises.
    """

    raw = study._storage.get_study_system_attrs(study._study_id).get(_SNAPSHOT_KEY)
    if raw is None:
        return []
    snapshots = json.loads(raw)
    assert isinstance(snapshots, list)
    return snapshots


def get_space_at_round(study: Study, round_index: int) -> dict[str, BaseDistribution]:
    """Answer "which dimensions did round ``round_index`` search?"."""

    snapshots = load_snapshots(study)
    if not snapshots:
        raise KeyError("No search-space snapshots are recorded for this study.")
    if not 0 <= round_index < len(snapshots):
        raise IndexError(
            f"Round {round_index} does not exist; the study has {len(snapshots)} round(s)."
        )
    return _deserialize_space(cast(dict[str, str], snapshots[round_index]["space"]))


def latest_snapshot(study: Study) -> dict[str, object] | None:
    snapshots = load_snapshots(study)
    return snapshots[-1] if snapshots else None


def latest_space(study: Study) -> dict[str, BaseDistribution] | None:
    snapshot = latest_snapshot(study)
    if snapshot is None:
        return None
    return _deserialize_space(cast(dict[str, str], snapshot["space"]))


def write_snapshots(study: Study, snapshots: list[dict[str, object]]) -> None:
    study._storage.set_study_system_attr(study._study_id, _SNAPSHOT_KEY, json.dumps(snapshots))


def append_round(
    study: Study,
    space: dict[str, BaseDistribution],
    first_trial_number: int,
    out_of_range_markers: dict[int, dict[str, str]] | None = None,
    note: str = "",
) -> int:
    """Append a new round snapshot and return its round index."""

    snapshots = load_snapshots(study)
    snapshots.append(
        {
            "round": len(snapshots),
            "space": _serialize_space(space),
            "first_trial_number": first_trial_number,
            "out_of_range": {
                str(trial_number): markers
                for trial_number, markers in sorted((out_of_range_markers or {}).items())
            },
            "note": note,
        }
    )
    write_snapshots(study, snapshots)
    return len(snapshots) - 1


def update_latest_round(
    study: Study,
    space: dict[str, BaseDistribution],
    out_of_range_markers: dict[int, dict[str, str]] | None = None,
    note: str = "",
) -> None:
    """Grow the current round's space in place (e.g. a conditional param appears).

    The first-trial number of the round is preserved.
    """

    snapshots = load_snapshots(study)
    current = snapshots[-1]
    current["space"] = _serialize_space(space)
    current["out_of_range"] = {
        str(trial_number): markers
        for trial_number, markers in sorted((out_of_range_markers or {}).items())
    }
    current["note"] = note
    write_snapshots(study, snapshots)
