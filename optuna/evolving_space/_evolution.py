"""Validation rules for search-space evolution across rounds.

A *round* is a version of the search space. Between rounds parameters can be
added, removed, re-ranged or (in a few lossless cases) re-typed. This module
contains the pure rules; persistence of the snapshots lives in
:mod:`optuna.evolving_space._snapshot` and the sampler consumes the verdicts.

Nothing here mutates historical trials. Out-of-range history points are reported
as data, never clipped.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING

from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.evolving_space._errors import SearchSpaceConflictError
from optuna.evolving_space._errors import SearchSpaceNoOverlapError


if TYPE_CHECKING:
    from optuna.distributions import BaseDistribution
    from optuna.trial import FrozenTrial


@dataclass(frozen=True)
class ParamEvolution:
    """Verdict for one parameter shared between the old and new space."""

    name: str
    kind: str
    # ``kind`` is one of: "unchanged", "range_widened", "range_narrowed",
    # "int_to_float", "categorical_choices_added", "categorical_choices_removed".
    old_distribution: BaseDistribution | None
    new_distribution: BaseDistribution | None


@dataclass(frozen=True)
class EvolutionReport:
    """Result of comparing two consecutive search spaces."""

    added: tuple[str, ...]
    removed: tuple[str, ...]
    evolved: tuple[ParamEvolution, ...]
    narrowed: tuple[str, ...] = field(default_factory=tuple)

    def out_of_range_param_names(self) -> tuple[str, ...]:
        return self.narrowed

    def summary(self) -> str:
        parts: list[str] = []
        if self.added:
            parts.append(f"added={list(self.added)}")
        if self.removed:
            parts.append(f"removed={list(self.removed)}")
        for e in self.evolved:
            if e.kind != "unchanged":
                parts.append(f"{e.name}: {e.kind}")
        return "; ".join(parts) if parts else "identical space"


def _describe(dist: BaseDistribution | None) -> str:
    if dist is None:
        return "absent"
    if isinstance(dist, CategoricalDistribution):
        return f"{type(dist).__name__}(choices={list(dist.choices)!r})"
    return f"{type(dist).__name__}({dist._asdict()})"


def _check_numeric_evolution(
    name: str,
    old: FloatDistribution | IntDistribution,
    new: FloatDistribution | IntDistribution,
) -> str:
    if old.__class__ is not new.__class__:
        # The only lossless numeric type change is int -> float (every int is a
        # valid float). float -> int cannot be lossless and is refused.
        if isinstance(old, IntDistribution) and isinstance(new, FloatDistribution):
            if old.log != new.log:
                raise SearchSpaceConflictError(
                    f"Parameter '{name}' changes IntDistribution(log={old.log}) to "
                    f"FloatDistribution(log={new.log}): the log configuration differs, "
                    "so the history cannot be mapped losslessly."
                )
            return "int_to_float"
        raise SearchSpaceConflictError(
            f"Parameter '{name}' changes distribution from {_describe(old)} to "
            f"{_describe(new)}: this type change cannot be mapped losslessly, "
            f"so the stored history for '{name}' cannot be reused."
        )

    if old.log != new.log:
        raise SearchSpaceConflictError(
            f"Parameter '{name}' changes the log configuration from log={old.log} "
            f"to log={new.log}: the internal representations are not comparable."
        )
    if old.step != new.step:
        raise SearchSpaceConflictError(
            f"Parameter '{name}' changes the discretization step from step={old.step} "
            f"to step={new.step}: the grids are incompatible and cannot be mapped "
            "without fabricating observations."
        )

    narrowed = (new.low > old.low) or (new.high < old.high)
    widened = (new.low < old.low) or (new.high > old.high)
    if narrowed and not widened:
        return "range_narrowed"
    if widened and not narrowed:
        return "range_widened"
    if narrowed and widened:
        # One bound moves in, the other out: the overlap is still a true
        # sub-range of the old space, treat as narrowed for out-of-range marking.
        return "range_narrowed"
    return "unchanged"


def _check_categorical_evolution(
    name: str,
    old: CategoricalDistribution,
    new: CategoricalDistribution,
) -> str:
    old_choices = list(old.choices)
    new_choices = list(new.choices)
    removed = [c for c in old_choices if c not in new_choices]
    added = [c for c in new_choices if c not in old_choices]
    if removed and added:
        return "categorical_choices_removed"  # history may reference dropped labels
    if removed:
        return "categorical_choices_removed"
    if added:
        return "categorical_choices_added"
    return "unchanged"


def validate_evolution(
    old_space: dict[str, BaseDistribution] | None,
    new_space: dict[str, BaseDistribution],
) -> EvolutionReport:
    """Compare two spaces and raise a readable error on hard conflicts.

    Additions and deletions of parameters are always allowed. Range moves and
    lossless type changes are allowed. Type changes that cannot preserve the
    meaning of historical values raise :class:`SearchSpaceConflictError` naming
    the parameter and the two distributions.
    """

    old_space = old_space or {}
    added = tuple(n for n in new_space if n not in old_space)
    removed = tuple(n for n in old_space if n not in new_space)
    evolved: list[ParamEvolution] = []
    narrowed: list[str] = []

    for name in sorted(set(old_space) & set(new_space)):
        old = old_space[name]
        new = new_space[name]
        if old == new:
            kind = "unchanged"
        elif type(old) is not type(new) and not (
            isinstance(old, IntDistribution) and isinstance(new, FloatDistribution)
        ):
            # Any other kind mismatch, including numeric <-> categorical, is
            # lossless-incompatible.
            raise SearchSpaceConflictError(
                f"Parameter '{name}' changes distribution from {_describe(old)} to "
                f"{_describe(new)}: the distribution kinds are incompatible, so "
                f"historical values of '{name}' cannot be mapped to the new space."
            )
        elif isinstance(old, CategoricalDistribution):
            assert isinstance(new, CategoricalDistribution)
            kind = _check_categorical_evolution(name, old, new)
        else:
            assert isinstance(old, (FloatDistribution, IntDistribution))
            assert isinstance(new, (FloatDistribution, IntDistribution))
            kind = _check_numeric_evolution(name, old, new)

        evolved.append(
            ParamEvolution(name=name, kind=kind, old_distribution=old, new_distribution=new)
        )
        if kind in ("range_narrowed", "categorical_choices_removed"):
            narrowed.append(name)

    report = EvolutionReport(
        added=added,
        removed=removed,
        evolved=tuple(evolved),
        narrowed=tuple(narrowed),
    )

    if old_space and not (set(new_space) & set(old_space)):
        raise SearchSpaceNoOverlapError(
            "The new search space shares no parameter with the previous round "
            f"(old={sorted(old_space)}, new={sorted(new_space)}). Every historical "
            "trial would be uninformative for the new dimensions; refusing to "
            "continue rather than producing trials that teach the sampler nothing."
        )
    return report


def is_out_of_range(value: object, distribution: BaseDistribution) -> bool:
    """Whether a historical value lies outside the given (newer) distribution.

    Only numeric bounds are checked here; categorical membership is checked via
    :func:`categorical_choice_dropped`.
    """

    if isinstance(distribution, CategoricalDistribution):
        return value not in distribution.choices
    if isinstance(distribution, FloatDistribution):
        if not isinstance(value, (int, float)):
            return True
        return not distribution._contains(distribution.to_internal_repr(float(value)))
    if isinstance(distribution, IntDistribution):
        if not isinstance(value, int):
            return True
        return not distribution._contains(distribution.to_internal_repr(value))
    return False


def categorize_history(
    trials: list[FrozenTrial],
    new_space: dict[str, BaseDistribution],
    report: EvolutionReport,
) -> dict[int, dict[str, str]]:
    """Return per-trial markers for parameters missing or out of range.

    The key is the trial number; the value maps parameter name to one of
    ``"missing"`` (the trial predates the parameter) or ``"out_of_range"`` (the
    trial has a real value that falls outside the current bounds/choices).
    Trials themselves are never modified.
    """

    narrowed = set(report.out_of_range_param_names())
    markers: dict[int, dict[str, str]] = {}
    for trial in trials:
        marks: dict[str, str] = {}
        for name, dist in new_space.items():
            if name not in trial.params:
                marks[name] = "missing"
            elif name in narrowed and is_out_of_range(trial.params[name], dist):
                marks[name] = "out_of_range"
        if marks:
            markers[trial.number] = marks
    return markers
