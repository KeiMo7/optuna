"""Unit tests for the pure evolution rules."""

from __future__ import annotations

import pytest

from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.evolving_space import SearchSpaceConflictError
from optuna.evolving_space import SearchSpaceNoOverlapError
from optuna.evolving_space import validate_evolution
from optuna.evolving_space._evolution import is_out_of_range


def test_add_and_remove_parameters_are_allowed() -> None:
    old = {"x": FloatDistribution(-1.0, 1.0)}
    new = {
        "x": FloatDistribution(-1.0, 1.0),
        "y": IntDistribution(0, 4),
    }
    report = validate_evolution(old, new)
    assert report.added == ("y",)
    assert report.removed == ()

    report_back = validate_evolution(new, old)
    assert report_back.added == ()
    assert report_back.removed == ("y",)


def test_range_narrowed_and_widened_are_classified() -> None:
    old = {"x": FloatDistribution(0.0, 10.0)}
    narrowed = validate_evolution(old, {"x": FloatDistribution(2.0, 8.0)})
    assert narrowed.evolved[0].kind == "range_narrowed"
    assert narrowed.out_of_range_param_names() == ("x",)

    widened = validate_evolution(old, {"x": FloatDistribution(-2.0, 12.0)})
    assert widened.evolved[0].kind == "range_widened"
    assert widened.out_of_range_param_names() == ()


def test_int_to_float_is_accepted_float_to_int_is_rejected() -> None:
    old = {"x": IntDistribution(0, 10)}
    report = validate_evolution(old, {"x": FloatDistribution(0.0, 10.0)})
    assert report.evolved[0].kind == "int_to_float"

    with pytest.raises(SearchSpaceConflictError, match="'x'"):
        validate_evolution({"x": FloatDistribution(0.0, 10.0)}, {"x": IntDistribution(0, 10)})


def test_numeric_to_categorical_conflict_names_parameter() -> None:
    old = {"x": FloatDistribution(0.0, 1.0)}
    new = {"x": CategoricalDistribution(("a", "b"))}
    with pytest.raises(SearchSpaceConflictError) as excinfo:
        validate_evolution(old, new)
    message = str(excinfo.value)
    assert "'x'" in message
    assert "FloatDistribution" in message
    assert "CategoricalDistribution" in message


def test_log_flag_change_is_rejected() -> None:
    old = {"x": FloatDistribution(1.0, 10.0, log=False)}
    new = {"x": FloatDistribution(1.0, 10.0, log=True)}
    with pytest.raises(SearchSpaceConflictError, match="log"):
        validate_evolution(old, new)


def test_categorical_choices_evolution() -> None:
    old = {"c": CategoricalDistribution(("a", "b"))}
    added = validate_evolution(old, {"c": CategoricalDistribution(("a", "b", "c"))})
    assert added.evolved[0].kind == "categorical_choices_added"

    removed = validate_evolution(old, {"c": CategoricalDistribution(("a",))})
    assert removed.evolved[0].kind == "categorical_choices_removed"
    assert removed.out_of_range_param_names() == ("c",)


def test_disjoint_spaces_raise_no_overlap() -> None:
    old = {"x": FloatDistribution(0.0, 1.0)}
    with pytest.raises(SearchSpaceNoOverlapError, match="no parameter"):
        validate_evolution(old, {"y": FloatDistribution(0.0, 1.0)})


def test_first_round_never_raises_no_overlap() -> None:
    report = validate_evolution(None, {"x": FloatDistribution(0.0, 1.0)})
    assert report.added == ("x",)


def test_is_out_of_range() -> None:
    numeric = FloatDistribution(0.0, 1.0)
    assert is_out_of_range(1.5, numeric)
    assert not is_out_of_range(0.5, numeric)

    categorical = CategoricalDistribution(("a", "b"))
    assert is_out_of_range("c", categorical)
    assert not is_out_of_range("a", categorical)

    integer_dist = IntDistribution(0, 5)
    assert is_out_of_range(6, integer_dist)
    assert not is_out_of_range(3, integer_dist)
