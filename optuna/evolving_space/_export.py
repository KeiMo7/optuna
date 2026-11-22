"""Tabular export that surfaces missing/out-of-range history points.

The standard :func:`optuna.Study.trials_dataframe` export only knows the raw
params. This export keeps every historical trial untouched and adds two
evolution-aware columns so narrowed ranges never silently clip history:

* ``evolving_space_markers``: JSON string mapping parameter name to
  ``"missing"`` or ``"out_of_range"`` for that trial;
* ``evolving_space_out_of_range``: comma-separated parameter names whose
  historical value falls outside the current space (empty when none).
"""

from __future__ import annotations

import json
from typing import cast
from typing import TYPE_CHECKING

from optuna.evolving_space._evolution import categorize_history
from optuna.evolving_space._evolution import validate_evolution
from optuna.evolving_space._snapshot import _deserialize_space
from optuna.evolving_space._snapshot import get_space_at_round
from optuna.evolving_space._snapshot import latest_snapshot
from optuna.evolving_space._snapshot import load_snapshots


if TYPE_CHECKING:
    import pandas as pd

    from optuna.study import Study


_MARKERS_COLUMN = "evolving_space_markers"
_OUT_OF_RANGE_COLUMN = "evolving_space_out_of_range"


def evolving_trials_dataframe(study: Study) -> pd.DataFrame:
    """Return the study's trials with evolution-marker columns appended.

    Every row corresponds to one stored trial; no trial, parameter value or
    state is modified. Requires :mod:`pandas`.
    """

    frame = study.trials_dataframe(
        attrs=(
            "number",
            "state",
            "values",
            "params",
            "value",
            "datetime_start",
            "datetime_complete",
            "duration",
        )
    )

    snapshot = latest_snapshot(study)
    if snapshot is None:
        frame[_MARKERS_COLUMN] = "{}"
        frame[_OUT_OF_RANGE_COLUMN] = ""
        return frame

    current_space = _deserialize_space(cast(dict[str, str], snapshot["space"]))
    snapshots = load_snapshots(study)
    previous_space = get_space_at_round(study, len(snapshots) - 2) if len(snapshots) >= 2 else None
    report = validate_evolution(previous_space, current_space)
    markers = categorize_history(study._get_trials(deepcopy=False), current_space, report)

    frame[_MARKERS_COLUMN] = frame["number"].map(
        lambda n: json.dumps(markers.get(n, {}), sort_keys=True)
    )
    frame[_OUT_OF_RANGE_COLUMN] = frame["number"].map(
        lambda n: ",".join(
            sorted(name for name, kind in markers.get(n, {}).items() if kind == "out_of_range")
        )
    )
    return frame
