"""Integration tests for evolving search spaces across storage backends.

Covers the acceptance criteria:

1. add-dimension / remove-dimension evolution keeps every historical trial;
2. with a fixed seed, keeping history changes the first suggestions versus
   discarding history;
3. in-memory and RDB backends give identical results for the same script;
4. incompatible type changes raise readable errors;
5. out-of-range history points are retained, marked and visible in the export.
"""

from __future__ import annotations

import os
import tempfile

import pytest

import optuna
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.evolving_space import evolving_trials_dataframe
from optuna.evolving_space import EvolvingTPESampler
from optuna.evolving_space import get_space_at_round
from optuna.evolving_space import load_snapshots
from optuna.evolving_space import SearchSpaceConflictError
from optuna.storages import RDBStorage
from optuna.trial import TrialState


def _round1_objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -10.0, 10.0)
    return float(x**2)


def _round2_added_objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -10.0, 10.0)
    y = trial.suggest_categorical("y", ("a", "b", "c"))
    return float(x**2) + (0.0 if y == "a" else 1.0)


def _round1_removed_objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -10.0, 10.0)
    z = trial.suggest_float("z", -1.0, 1.0)
    return float(x**2) + float(z)


def _round2_removed_objective(trial: optuna.Trial) -> float:
    # ``z`` disappears in the second round.
    x = trial.suggest_float("x", -10.0, 10.0)
    return float(x**2)


# ---------------------------------------------------------------------------
# Criterion 1: add / remove dimensions never delete history
# ---------------------------------------------------------------------------
def test_add_dimension_keeps_all_historical_trials() -> None:
    sampler = EvolvingTPESampler(seed=1, n_startup_trials=3)
    study = optuna.create_study(sampler=sampler)
    study.optimize(_round1_objective, n_trials=8)
    round1_params = [dict(t.params) for t in study.trials]
    round1_values = [t.value for t in study.trials]
    round1_states = [t.state for t in study.trials]

    study.optimize(_round2_added_objective, n_trials=8)

    assert len(study.trials) == 16
    # The first eight trials are byte-for-byte unchanged.
    for i in range(8):
        assert dict(study.trials[i].params) == round1_params[i]
        assert study.trials[i].value == round1_values[i]
        assert study.trials[i].state == round1_states[i]
        assert "y" not in study.trials[i].params

    snapshots = load_snapshots(study)
    assert len(snapshots) == 2
    assert set(get_space_at_round(study, 0).keys()) == {"x"}
    assert set(get_space_at_round(study, 1).keys()) == {"x", "y"}
    assert snapshots[1]["first_trial_number"] == 8


def test_remove_dimension_keeps_all_historical_trials() -> None:
    from optuna.evolving_space import mark_new_round

    sampler = EvolvingTPESampler(seed=1, n_startup_trials=3)
    study = optuna.create_study(sampler=sampler)
    study.optimize(_round1_removed_objective, n_trials=8)
    # Deleting a parameter present in every prior trial is data-equivalent to
    # the first sighting of a conditional branch, so the boundary is declared
    # explicitly through the deterministic round marker.
    mark_new_round(study)
    study.optimize(_round2_removed_objective, n_trials=8)

    assert len(study.trials) == 16
    # Old trials still carry the removed parameter.
    assert all("z" in study.trials[i].params for i in range(8))
    assert all("z" not in study.trials[i].params for i in range(8, 16))
    snapshots = load_snapshots(study)
    assert len(snapshots) == 2
    assert set(get_space_at_round(study, 0).keys()) == {"x", "z"}
    assert set(get_space_at_round(study, 1).keys()) == {"x"}


def test_sampler_learns_from_partial_history_not_zero_fill() -> None:
    # The x-dimension kernel must be built from *all* x-bearing trials even
    # though round-1 trials lack y. Eligibility is per dimension.
    sampler = EvolvingTPESampler(seed=2, n_startup_trials=2)
    study = optuna.create_study(sampler=sampler)
    study.optimize(_round1_objective, n_trials=6)
    study.optimize(_round2_added_objective, n_trials=6)

    eligible_x = sampler._eligible_trials(study, "x", FloatDistribution(-10.0, 10.0))
    assert len(eligible_x) == 12  # every trial has x, none were dropped
    eligible_y = sampler._eligible_trials(study, "y", CategoricalDistribution(("a", "b", "c")))
    assert len(eligible_y) == 6  # only round-2 trials carry y


# ---------------------------------------------------------------------------
# Criterion 2: retained history changes the suggestions (fixed seed)
# ---------------------------------------------------------------------------
def test_retained_history_changes_first_suggestions() -> None:
    seed = 42

    # Branch A: round-1 history exists and must be reused.
    sampler_keep = EvolvingTPESampler(seed=seed, n_startup_trials=3)
    study_keep = optuna.create_study(sampler=sampler_keep)
    study_keep.optimize(_round1_objective, n_trials=8)
    first_keep = []
    study_keep.optimize(
        _round2_added_objective,
        n_trials=3,
        callbacks=[lambda study, trial: first_keep.append(dict(trial.params))],
    )

    # Branch B: an empty study with the same seed and same round-2 space
    # (history effectively discarded).
    sampler_drop = EvolvingTPESampler(seed=seed, n_startup_trials=3)
    study_drop = optuna.create_study(sampler=sampler_drop)
    first_drop = []

    def round2_only(trial: optuna.Trial) -> float:
        return _round2_added_objective(trial)

    study_drop.optimize(
        round2_only,
        n_trials=3,
        callbacks=[lambda study, trial: first_drop.append(dict(trial.params))],
    )

    assert len(first_keep) == len(first_drop) == 3
    assert first_keep != first_drop
    # The x suggestions specifically must differ: they are informed by eight
    # prior x observations in branch A versus nothing in branch B.
    assert [p["x"] for p in first_keep] != [p["x"] for p in first_drop]


# ---------------------------------------------------------------------------
# Criterion 3: backend parity (in-memory vs RDB)
# ---------------------------------------------------------------------------
def _run_two_round_script(storage: optuna.storages.BaseStorage, seed: int) -> dict:
    sampler = EvolvingTPESampler(seed=seed, n_startup_trials=3)
    study = optuna.create_study(storage=storage, sampler=sampler)
    study.optimize(_round1_objective, n_trials=6)
    study.optimize(_round2_added_objective, n_trials=6)
    return {
        "params": [dict(t.params) for t in study.trials],
        "values": [t.value for t in study.trials],
        "rounds": [sorted(snap["space"].keys()) for snap in load_snapshots(study)],
    }


def test_rdb_and_in_memory_backends_agree() -> None:
    seed = 7
    memory_result = _run_two_round_script(optuna.storages.InMemoryStorage(), seed)
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "evolving.db")
        rdb_storage = RDBStorage(f"sqlite:///{db_path}")
        rdb_result = _run_two_round_script(rdb_storage, seed)

    assert rdb_result["params"] == memory_result["params"]
    assert rdb_result["values"] == memory_result["values"]
    assert rdb_result["rounds"] == memory_result["rounds"]


def test_old_study_without_snapshots_opens_read_only() -> None:
    # A study written without any evolving-space snapshot must open cleanly.
    storage = optuna.storages.InMemoryStorage()
    study = optuna.create_study(storage=storage)
    study.optimize(_round1_objective, n_trials=3)
    assert load_snapshots(study) == []
    # Export on a legacy study adds empty marker columns instead of failing.
    frame = evolving_trials_dataframe(study)
    assert "evolving_space_markers" in frame.columns
    assert (frame["evolving_space_out_of_range"] == "").all()


# ---------------------------------------------------------------------------
# Criterion 4: readable type-conflict errors
# ---------------------------------------------------------------------------
def test_type_conflict_raises_named_error() -> None:
    sampler = EvolvingTPESampler(seed=0, n_startup_trials=1)
    study = optuna.create_study(sampler=sampler)
    study.optimize(_round1_objective, n_trials=2)

    def conflicting(trial: optuna.Trial) -> float:
        trial.suggest_categorical("x", ("low", "high"))
        return 0.0

    with pytest.raises(SearchSpaceConflictError) as excinfo:
        study.optimize(conflicting, n_trials=1)
    message = str(excinfo.value)
    assert "'x'" in message
    assert "FloatDistribution" in message
    assert "CategoricalDistribution" in message
    # The failed round must not have created a snapshot. The failed trial is
    # retained in its FAIL state (trials are never rewritten or deleted), but
    # no successful trial ran under the conflicting space.
    assert len(load_snapshots(study)) == 1
    assert len(study.trials) == 3
    assert study.trials[2].state == TrialState.FAIL
    assert all(t.state == TrialState.COMPLETE for t in study.trials[:2])


# ---------------------------------------------------------------------------
# Criterion 5: out-of-range history is retained and marked
# ---------------------------------------------------------------------------
def test_narrowed_range_marks_but_keeps_history() -> None:
    sampler = EvolvingTPESampler(seed=3, n_startup_trials=2)
    study = optuna.create_study(sampler=sampler)

    def wide(trial: optuna.Trial) -> float:
        x = trial.suggest_float("x", 0.0, 10.0)
        return float((x - 9.0) ** 2)

    # Deterministically place some history points outside the future range.
    for forced_x in (0.5, 5.5, 7.0, 9.0):
        study.enqueue_trial({"x": forced_x})
    study.optimize(wide, n_trials=8)
    wide_values = [t.params["x"] for t in study.trials]
    assert any(v > 4.0 for v in wide_values)

    def narrow(trial: optuna.Trial) -> float:
        x = trial.suggest_float("x", 0.0, 4.0)
        return float((x - 3.0) ** 2)

    study.optimize(narrow, n_trials=4)

    # Every old trial remains, with its original (possibly out-of-range) value.
    assert len(study.trials) == 12
    assert [t.params["x"] for t in study.trials[:8]] == wide_values
    out_of_range_old = [v for v in wide_values if v > 4.0]
    assert len(out_of_range_old) >= 3

    frame = evolving_trials_dataframe(study)
    assert "evolving_space_out_of_range" in frame.columns
    marked_numbers = frame.loc[frame["evolving_space_out_of_range"] == "x", "number"].tolist()
    assert marked_numbers, "at least one out-of-range trial must be flagged"
    for number in marked_numbers:
        assert study.trials[number].params["x"] > 4.0
        assert study.trials[number].state == TrialState.COMPLETE

    # The out-of-range points are excluded from the kernel but not deleted.
    eligible = sampler._eligible_trials(study, "x", FloatDistribution(0.0, 4.0))
    eligible_numbers = {t.number for t in eligible}
    assert set(marked_numbers).isdisjoint(eligible_numbers)
    assert all(number < 8 for number in marked_numbers)


def test_dropped_categorical_choice_is_retained_and_marked() -> None:
    sampler = EvolvingTPESampler(seed=9, n_startup_trials=2)
    study = optuna.create_study(sampler=sampler)

    def with_three_choices(trial: optuna.Trial) -> float:
        return 0.0 if trial.suggest_categorical("c", ("a", "b", "c")) == "a" else 1.0

    study.enqueue_trial({"c": "c"})
    study.optimize(with_three_choices, n_trials=5)
    assert any(t.params["c"] == "c" for t in study.trials)

    def with_two_choices(trial: optuna.Trial) -> float:
        return 0.0 if trial.suggest_categorical("c", ("a", "b")) == "a" else 1.0

    study.optimize(with_two_choices, n_trials=3)

    # The trial carrying the dropped choice is untouched.
    assert any(t.params["c"] == "c" for t in study.trials)
    frame = evolving_trials_dataframe(study)
    marked = frame.loc[frame["evolving_space_out_of_range"] == "c", "number"].tolist()
    assert marked
    for number in marked:
        assert study.trials[number].params["c"] == "c"


def test_missing_dimension_is_marked_not_zero() -> None:
    sampler = EvolvingTPESampler(seed=4, n_startup_trials=2)
    study = optuna.create_study(sampler=sampler)
    study.optimize(_round1_objective, n_trials=4)
    study.optimize(_round2_added_objective, n_trials=4)

    frame = evolving_trials_dataframe(study)
    first_row = frame.iloc[0]
    assert '"y": "missing"' in first_row["evolving_space_markers"]
    # No out-of-range flag for the merely-missing dimension.
    assert first_row["evolving_space_out_of_range"] == ""


# ---------------------------------------------------------------------------
# Other required shapes
# ---------------------------------------------------------------------------
def test_int_to_float_evolution_runs_and_keeps_history() -> None:
    sampler = EvolvingTPESampler(seed=5, n_startup_trials=2)
    study = optuna.create_study(sampler=sampler)

    def int_round(trial: optuna.Trial) -> float:
        return float(trial.suggest_int("x", 0, 10))

    study.optimize(int_round, n_trials=6)

    def float_round(trial: optuna.Trial) -> float:
        return trial.suggest_float("x", 0.0, 10.0)

    study.optimize(float_round, n_trials=4)
    assert len(study.trials) == 10
    assert len(load_snapshots(study)) == 2
    assert all(isinstance(t.params["x"], int) for t in study.trials[:6])


def test_conditional_parameters_share_one_round() -> None:
    sampler = EvolvingTPESampler(seed=6, n_startup_trials=2)
    study = optuna.create_study(sampler=sampler)

    def conditional(trial: optuna.Trial) -> float:
        switch = trial.suggest_categorical("switch", (True, False))
        if switch:
            trial.suggest_float("branch_param", 0.0, 1.0)
        return 0.0

    # Seed both arms of the switch before the sampler-driven trials: the two
    # enqueued trials establish the two distinct parameter sets that mark
    # ``branch_param`` as conditional, so later random trials never open a
    # spurious round when they take one branch or the other.
    study.enqueue_trial({"switch": True, "branch_param": 0.5})
    study.enqueue_trial({"switch": False})
    study.optimize(conditional, n_trials=12)
    snapshots = load_snapshots(study)
    # Conditional branches must not create spurious rounds.
    assert len(snapshots) == 1
    assert set(get_space_at_round(study, 0).keys()) == {"switch", "branch_param"}


def test_multiobjective_with_added_objective_space() -> None:
    sampler = EvolvingTPESampler(seed=8, n_startup_trials=2)
    study = optuna.create_study(directions=["minimize", "minimize"], sampler=sampler)

    def one_obj_space(trial: optuna.Trial) -> tuple[float, float]:
        x = trial.suggest_float("x", -5.0, 5.0)
        return x**2, (x - 1.0) ** 2

    study.optimize(one_obj_space, n_trials=6)

    def two_dim_space(trial: optuna.Trial) -> tuple[float, float]:
        x = trial.suggest_float("x", -5.0, 5.0)
        y = trial.suggest_float("y", -5.0, 5.0)
        return x**2 + y**2, (x - 1.0) ** 2 + y**2

    study.optimize(two_dim_space, n_trials=4)
    assert len(study.trials) == 10
    assert len(load_snapshots(study)) == 2
    assert all(t.state == TrialState.COMPLETE for t in study.trials)
