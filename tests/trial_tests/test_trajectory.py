from __future__ import annotations

import pytest

import optuna
from optuna.exceptions import DuplicatedStepError
from optuna.exceptions import TrajectoryCorruptedError
from optuna.exceptions import UpdateFinishedTrialError
from optuna.testing.storages import STORAGE_MODES
from optuna.testing.storages import StorageSupplier
from optuna.trial import TrialState


# gRPC/redis backends are unavailable in some CI environments; the storage-parametrized
# coverage still runs against in-memory, sqlite, cached sqlite and journal.
_AVAILABLE_MODES = [m for m in STORAGE_MODES if "grpc" not in m and "redis" not in m]


class _PruneAtStep(optuna.pruners.BasePruner):
    """Deterministically prune once the reported step reaches ``threshold``."""

    def __init__(self, threshold: int = 1) -> None:
        self._threshold = threshold

    def prune(self, study: "optuna.study.Study", trial: "optuna.trial.FrozenTrial") -> bool:
        return trial.last_step is not None and trial.last_step >= self._threshold


def _make_study(storage_specifier: str, **kwargs: object) -> optuna.Study:
    with StorageSupplier(storage_specifier) as storage:
        return optuna.create_study(storage=storage, **kwargs)


# ---------------------------------------------------------------------------
# 1. Normal trajectory: full ordered reconstruction of a completed trial.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("storage_mode", _AVAILABLE_MODES)
def test_normal_trajectory_records_ordered_steps(storage_mode: str) -> None:
    with StorageSupplier(storage_mode) as storage:
        study = optuna.create_study(direction="maximize", storage=storage, enable_trajectory=True)

        def objective(trial: optuna.Trial) -> float:
            trial.suggest_float("x", 0.0, 1.0)
            for step in range(5):
                trial.report(float(step) * 0.1, step)
            return 1.0

        study.optimize(objective, n_trials=1)

        traj = study.get_trial_trajectory(0)
        kinds = [e["kind"] for e in traj.events]
        assert kinds[0] == "running"
        assert kinds[-1] == "state"
        assert traj.final_state == TrialState.COMPLETE.name
        step_events = traj.steps
        # Hard constraint: unique and ordered step numbers.
        steps = [e["step"] for e in step_events]
        assert steps == sorted(steps)
        assert len(steps) == len(set(steps))
        assert steps == [0, 1, 2, 3, 4]
        assert traj.intermediate_values == study.trials[0].intermediate_values


# ---------------------------------------------------------------------------
# 2. Report after the pruning decision is rejected (closes the prune window).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("storage_mode", _AVAILABLE_MODES)
def test_report_after_prune_is_rejected(storage_mode: str) -> None:
    with StorageSupplier(storage_mode) as storage:
        study = optuna.create_study(
            direction="maximize",
            storage=storage,
            pruner=_PruneAtStep(threshold=1),
            enable_trajectory=True,
        )
        rejected: dict[str, str] = {}

        def objective(trial: optuna.Trial) -> float:
            trial.suggest_float("x", 0.0, 1.0)
            trial.report(0.0, 0)
            trial.report(1.0, 1)
            assert trial.should_prune() is True
            # We are now inside the prune-decision -> state-persistence window.
            with pytest.raises(UpdateFinishedTrialError) as excinfo:
                trial.report(99.0, 2)
            rejected["message"] = str(excinfo.value)
            raise optuna.TrialPruned()

        study.optimize(objective, n_trials=1)

        frozen = study.trials[0]
        assert frozen.state == TrialState.PRUNED
        # Acceptance assertion: no new intermediate value is accepted after pruning.
        assert 2 not in frozen.intermediate_values
        assert set(frozen.intermediate_values) == {0, 1}
        traj = study.get_trial_trajectory(0)
        assert traj.is_pruned
        assert traj.pruned_at_step == 1
        assert all(e["step"] != 2 for e in traj.steps)
        # The error message states the current status explicitly.
        assert "pruned" in rejected["message"].lower()


def test_report_on_finished_trial_raises_with_state() -> None:
    study = optuna.create_study(enable_trajectory=True)

    def objective(trial: optuna.Trial) -> float:
        trial.suggest_float("x", 0.0, 1.0)
        trial.report(0.1, 0)
        return 1.0

    study.optimize(objective, n_trials=1)
    trial = study.ask()  # a fresh running trial; finish it via tell
    trial.report(0.0, 0)
    study.tell(trial, 1.0)
    with pytest.raises(UpdateFinishedTrialError) as excinfo:
        trial.report(0.5, 1)
    assert "finished" in str(excinfo.value).lower()


# ---------------------------------------------------------------------------
# 3. Duplicate step reporting is deterministically rejected.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("storage_mode", _AVAILABLE_MODES)
def test_duplicate_step_is_rejected(storage_mode: str) -> None:
    with StorageSupplier(storage_mode) as storage:
        study = optuna.create_study(storage=storage, enable_trajectory=True)

        def objective(trial: optuna.Trial) -> float:
            trial.suggest_float("x", 0.0, 1.0)
            trial.report(0.5, 0)
            trial.report(0.9, 0)  # same step -> reject
            return 1.0

        with pytest.raises(DuplicatedStepError):
            study.optimize(objective, n_trials=1)

        # First value is retained; the duplicate never landed.
        assert study.trials[0].intermediate_values == {0: 0.5}


# ---------------------------------------------------------------------------
# 4. A failed trial keeps its state and its failure reason.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("storage_mode", _AVAILABLE_MODES)
def test_failure_state_and_reason_are_preserved(storage_mode: str) -> None:
    with StorageSupplier(storage_mode) as storage:
        study = optuna.create_study(storage=storage, enable_trajectory=True)

        def objective(trial: optuna.Trial) -> float:
            trial.suggest_float("x", 0.0, 1.0)
            trial.report(0.1, 0)
            raise RuntimeError("intentional boom")

        study.optimize(objective, n_trials=1, catch=(RuntimeError,))

        frozen = study.trials[0]
        assert frozen.state == TrialState.FAIL
        traj = study.get_trial_trajectory(0)
        assert traj.is_failed
        assert traj.final_state == TrialState.FAIL.name
        assert traj.failure_reason is not None
        assert traj.failure_reason[0] == "RuntimeError"
        assert "intentional boom" in traj.failure_reason[1]


# ---------------------------------------------------------------------------
# 5. Switch off: behavior is identical to historical Optuna.
# ---------------------------------------------------------------------------
def test_switch_off_keeps_legacy_behavior() -> None:
    study = optuna.create_study()  # default disabled
    assert study.trajectory_enabled is False

    def objective(trial: optuna.Trial) -> float:
        trial.suggest_float("x", 0.0, 1.0)
        trial.report(0.5, 0)
        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            trial.report(0.9, 0)  # legacy: warn + ignore, no exception
            assert any("already reported" in str(w.message) for w in caught)
        return 1.0

    study.optimize(objective, n_trials=1)
    assert study.trials[0].intermediate_values == {0: 0.5}
    assert study.get_all_trajectories() == []


def test_no_report_trial_writes_no_trajectory() -> None:
    # Even with the switch on, a trial that never reports an intermediate value performs
    # no trajectory storage write and matches the legacy behavior exactly.
    from optuna.trial._trajectory import has_trajectory

    study = optuna.create_study(enable_trajectory=True)

    def objective(trial: optuna.Trial) -> float:
        return trial.suggest_float("x", 0.0, 1.0)

    study.optimize(objective, n_trials=3)
    for frozen in study.trials:
        assert frozen.state == TrialState.COMPLETE
        assert not has_trajectory(frozen)
        assert "optuna:trajectory:v1" not in frozen.system_attrs
    assert study.get_all_trajectories() == []
    assert study.validate_trajectories() == {}


@pytest.mark.parametrize("storage_mode", _AVAILABLE_MODES)
def test_switch_persists_across_reload(storage_mode: str) -> None:
    with StorageSupplier(storage_mode) as storage:
        study = optuna.create_study(storage=storage, study_name="persist", enable_trajectory=True)

        def objective(trial: optuna.Trial) -> float:
            trial.suggest_float("x", 0.0, 1.0)
            trial.report(0.0, 0)
            return 1.0

        study.optimize(objective, n_trials=1)
        reloaded = optuna.load_study(study_name="persist", storage=storage)
        assert reloaded.trajectory_enabled is True
        traj = reloaded.get_trial_trajectory(0)
        assert traj.final_state == TrialState.COMPLETE.name


# ---------------------------------------------------------------------------
# 6. Repeat groups and parameter-combination history lookup.
# ---------------------------------------------------------------------------
def test_repeat_groups_and_param_lookup() -> None:
    study = optuna.create_study(
        sampler=optuna.samplers.RandomSampler(seed=3), enable_trajectory=True
    )

    def objective(trial: optuna.Trial) -> float:
        x = trial.suggest_categorical("x", [1, 2])
        return float(x)

    study.optimize(objective, n_trials=10)
    groups = study.get_repeat_groups()
    repeated = {k: v for k, v in groups.items() if len(v) > 1}
    assert repeated  # some combination was sampled more than once
    # Each repeat remains a distinct, individually counted trial.
    for members in repeated.values():
        assert len({t.number for t in members}) == len(members)

    matched = study.get_trials_with_params({"x": 1})
    assert matched and all(t.params["x"] == 1 for t in matched)


# ---------------------------------------------------------------------------
# 7. Corruption is located to a trial (and step) instead of crashing silently.
# ---------------------------------------------------------------------------
def test_corrupted_trajectory_is_located(tmp_path: object) -> None:
    import json
    import sqlite3

    db_path = f"{tmp_path}/corrupt.sqlite"  # type: ignore[operator]
    study = optuna.create_study(
        storage=f"sqlite:///{db_path}", study_name="c", enable_trajectory=True
    )

    def objective(trial: optuna.Trial) -> float:
        trial.suggest_float("x", 0.0, 1.0)
        for step in range(3):
            trial.report(float(step), step)
        return 1.0

    study.optimize(objective, n_trials=1)

    key = "optuna:trajectory:v1"
    con = sqlite3.connect(db_path)
    trial_id, raw = con.execute(
        "SELECT trial_id, value_json FROM trial_system_attributes WHERE key=?", (key,)
    ).fetchone()
    payload = json.loads(raw)
    payload["events"].append(dict(payload["events"][1]))  # duplicate step
    con.execute(
        "UPDATE trial_system_attributes SET value_json=? WHERE trial_id=? AND key=?",
        (json.dumps(payload), trial_id, key),
    )
    con.commit()
    con.close()

    fresh = optuna.load_study(study_name="c", storage=f"sqlite:///{db_path}")
    problems = fresh.validate_trajectories()
    assert 0 in problems
    assert "step=0" in problems[0]
    with pytest.raises(TrajectoryCorruptedError):
        fresh.get_trial_trajectory(0)
