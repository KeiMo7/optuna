from __future__ import annotations

import threading
import time

import pytest

import optuna
from optuna.resumable import CheckpointCorruptedError
from optuna.resumable import InMemoryLeaseStore
from optuna.resumable import LeaseLostError
from optuna.resumable import ResumableExecutor
from optuna.resumable import ResumableTrial
from optuna.resumable import SearchSpaceChangedError
from optuna.resumable import StorageCheckpointStore
from optuna.resumable._checkpoint import FileCheckpointStore
from optuna.trial import TrialState


N_STEPS = 10


def make_objective(suspend_steps=(), suspend_once=None):
    """Deterministic objective: sum of (x - 2)**2 + step over N_STEPS steps."""

    def objective(trial):
        x = trial.suggest_float("x", -10, 10)
        total = trial.checkpoint_payload.get("total", 0.0) if trial.resumed else 0.0
        for step in range(trial.start_step, N_STEPS):
            total += (x - 2) ** 2 + step
            trial.report(total, step)
            if step in suspend_steps and (suspend_once is None or not suspend_once["used"]):
                if suspend_once is not None:
                    suspend_once["used"] = True
                trial.suspend({"total": total})
        return total

    return objective


def make_blocking_suspend_objective(step_to_block, reached_event):
    """Objective that reports, then waits at `step_to_block` for a suspend request."""

    def objective(trial):
        x = trial.suggest_float("x", -10, 10)
        total = trial.checkpoint_payload.get("total", 0.0) if trial.resumed else 0.0
        for step in range(trial.start_step, N_STEPS):
            total += (x - 2) ** 2 + step
            trial.report(total, step)
            if step == step_to_block and not trial.resumed:
                reached_event.set()
                deadline = time.time() + 30
                while not trial.should_suspend():
                    if time.time() > deadline:
                        raise RuntimeError("suspend was never requested")
                    time.sleep(0.01)
                trial.suspend({"total": total})
        return total

    return objective


def plain_objective(trial):
    x = trial.suggest_float("x", -10, 10)
    total = 0.0
    for step in range(N_STEPS):
        total += (x - 2) ** 2 + step
        trial.report(total, step)
    return total


def make_study(seed=42):
    return optuna.create_study(
        storage=optuna.storages.InMemoryStorage(),
        sampler=optuna.samplers.RandomSampler(seed=seed),
    )


def make_executor(study, executor_id="exec-1", **kwargs):
    kwargs.setdefault("lease_ttl", None)
    return ResumableExecutor(study, executor_id=executor_id, **kwargs)


def test_suspend_and_resume_matches_uninterrupted():
    baseline_study = make_study()
    baseline_study.optimize(plain_objective, n_trials=1)
    baseline = baseline_study.trials[0]

    study = make_study()
    executor = make_executor(study)
    summary = executor.run(make_objective(suspend_steps={4}), n_trials=1)

    assert summary.suspended == 1
    assert summary.completed == 1
    trial = study.trials[0]
    assert trial.state == TrialState.COMPLETE
    assert trial.value == baseline.value
    assert trial.intermediate_values == baseline.intermediate_values
    assert trial.params == baseline.params


def test_multiple_suspend_resume_cycles_single_history():
    study = make_study()
    executor = make_executor(study)
    summary = executor.run(make_objective(suspend_steps={2, 5, 7}), n_trials=1)
    assert summary.suspended == 3
    trial = study.trials[0]
    assert trial.state == TrialState.COMPLETE
    # Exactly one record per step: no duplicates from the three resumes.
    assert sorted(trial.intermediate_values) == list(range(N_STEPS))
    expected_total = sum((trial.params["x"] - 2) ** 2 + s for s in range(N_STEPS))
    assert trial.value == pytest.approx(expected_total)


def test_resume_on_a_different_executor():
    study = make_study()
    reached_step_4 = threading.Event()

    ex1 = make_executor(study, executor_id="machine-1")
    result_holder = {}

    def run_ex1():
        result_holder["summary"] = ex1.run(
            make_blocking_suspend_objective(4, reached_step_4), n_trials=1
        )

    thread = threading.Thread(target=run_ex1)
    thread.start()
    assert reached_step_4.wait(timeout=10)
    ex1.request_suspend()  # simulate the scheduler reclaiming machine-1
    thread.join(timeout=10)
    assert result_holder["summary"].suspended == 1
    assert study.trials[0].state == TrialState.WAITING  # resumable, not running

    # A different executor (another process/machine) picks the trial up.
    ex2 = make_executor(study, executor_id="machine-2")
    summary = ex2.run(make_objective(), n_trials=1)
    assert summary.completed == 1
    trial = study.trials[0]
    assert trial.state == TrialState.COMPLETE
    expected_total = sum((trial.params["x"] - 2) ** 2 + s for s in range(N_STEPS))
    assert trial.value == pytest.approx(expected_total)
    assert sorted(trial.intermediate_values) == list(range(N_STEPS))


def test_slot_released_during_suspend():
    study = make_study()
    lease_store = InMemoryLeaseStore()
    ex1 = make_executor(study, executor_id="exec-1", lease_store=lease_store, max_concurrent=1)
    reached_step_2 = threading.Event()

    result_holder = {}

    def run_ex1():
        result_holder["summary"] = ex1.run(
            make_blocking_suspend_objective(2, reached_step_2), n_trials=1
        )

    thread = threading.Thread(target=run_ex1)
    thread.start()
    assert reached_step_2.wait(timeout=10)
    assert lease_store.count_active(study._study_id) == 1  # slot occupied while running
    ex1.request_suspend()
    thread.join(timeout=10)
    assert result_holder["summary"].suspended == 1
    # The concurrency slot was released: another executor can use it right away.
    assert lease_store.count_active(study._study_id) == 0
    other_trial = study.ask()
    lease_store.acquire(other_trial._trial_id, study._study_id, "exec-2", None, max_concurrent=1)


def test_cancel_reclaims_slot_checkpoint_and_state(tmp_path):
    study = make_study()
    checkpoint_store = FileCheckpointStore(tmp_path)
    executor = make_executor(study, checkpoint_store=checkpoint_store)
    reached_step_2 = threading.Event()

    def run_ex():
        executor.run(make_blocking_suspend_objective(2, reached_step_2), n_trials=1)

    thread = threading.Thread(target=run_ex)
    thread.start()
    assert reached_step_2.wait(timeout=10)
    executor.request_suspend()
    thread.join(timeout=10)
    trial_id = study.trials[0]._trial_id
    assert checkpoint_store.load(study._study_id, trial_id) is not None

    executor.cancel_trial(trial_id)

    assert checkpoint_store.load(study._study_id, trial_id) is None
    assert executor.lease_store.get(trial_id) is None
    assert executor.lease_store.count_active(study._study_id) == 0
    assert study.trials[0].state == TrialState.FAIL


def test_stale_executor_cannot_write_after_takeover():
    study = make_study()
    clock = [1000.0]
    lease_store = InMemoryLeaseStore(now=lambda: clock[0])
    executor = make_executor(study, lease_store=lease_store, lease_ttl=10.0)

    trial = study.ask()
    trial_id = trial._trial_id
    lease = lease_store.acquire(trial_id, study._study_id, "exec-1", ttl=10.0)
    rt = ResumableTrial(
        trial,
        study_id=study._study_id,
        lease=lease,
        lease_store=lease_store,
        checkpoint_store=StorageCheckpointStore(study._storage),
        checkpoint=None,
        should_suspend=lambda: False,
        checkpoint_on_report=False,
    )
    rt.report(1.0, 0)  # works while the lease is valid

    # The executor is partitioned away, its lease expires, another executor
    # takes the trial over.
    clock[0] += 11.0
    lease_store.takeover(trial_id, "exec-2", ttl=10.0)

    # The old executor briefly comes back to life: every write is refused.
    with pytest.raises(LeaseLostError):
        rt.report(2.0, 1)
    with pytest.raises(LeaseLostError):
        rt.suggest_float("y", 0, 1)
    with pytest.raises(LeaseLostError):
        rt.set_user_attr("k", "v")
    assert executor  # keep linters quiet about the unused fixture-style var


def test_crashed_executor_reclaimed_by_lease_expiry():
    study = make_study()
    lease_store = InMemoryLeaseStore()
    ex1 = make_executor(study, executor_id="exec-1", lease_store=lease_store, lease_ttl=0.3)
    ex2 = make_executor(study, executor_id="exec-2", lease_store=lease_store, lease_ttl=0.3)

    # Simulate exec-1 crashing mid-trial: a RUNNING trial, a lease, no cleanup.
    trial = study.ask()
    trial_id = trial._trial_id
    lease_store.acquire(trial_id, study._study_id, "exec-1", ttl=0.3)
    assert study.trials[0].state == TrialState.RUNNING

    time.sleep(0.4)  # lease expires because no heartbeat arrives
    start = time.time()
    reaped = ex2.reap_expired()
    bounded = time.time() - start
    assert reaped == [trial_id]
    assert bounded < 5.0  # reclamation happens in bounded time
    assert study.trials[0].state == TrialState.WAITING
    # The slot is free again.
    assert lease_store.count_active(study._study_id) == 0
    assert ex1 is not None  # the crashed executor is out of the picture


def test_concurrent_reap_only_one_winner():
    study = make_study()
    lease_store = InMemoryLeaseStore()
    trial = study.ask()
    trial_id = trial._trial_id
    lease_store.acquire(trial_id, study._study_id, "dead-exec", ttl=0.2)
    time.sleep(0.3)

    executors = [
        make_executor(study, executor_id=f"exec-{i}", lease_store=lease_store, lease_ttl=0.2)
        for i in range(2)
    ]
    barrier = threading.Barrier(2)
    results: list[list[int]] = []

    def reap(ex):
        barrier.wait(timeout=10)
        results.append(ex.reap_expired())

    threads = [threading.Thread(target=reap, args=(ex,)) for ex in executors]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # Exactly one executor reclaimed the trial; the other got nothing.
    assert sorted(results, key=len) == [[], [trial_id]]
    assert study.trials[0].state == TrialState.WAITING


def test_crash_resume_from_last_reported_step():
    study = make_study()
    lease_store = InMemoryLeaseStore()
    executed_steps: list[int] = []

    def objective(trial):
        x = trial.suggest_float("x", -10, 10)
        total = 0.0
        for step in range(trial.start_step, N_STEPS):
            executed_steps.append(step)
            total = sum((x - 2) ** 2 + s for s in range(step + 1))
            trial.report(total, step)
        return total

    # exec-1 runs the first 5 steps, then "crashes" (no cleanup, lease expires).
    ex1 = make_executor(study, executor_id="exec-1", lease_store=lease_store, lease_ttl=0.3)
    trial = study.ask()
    trial_id = trial._trial_id
    lease = lease_store.acquire(trial_id, study._study_id, "exec-1", ttl=0.3)
    rt = ResumableTrial(
        trial,
        study_id=study._study_id,
        lease=lease,
        lease_store=lease_store,
        checkpoint_store=StorageCheckpointStore(study._storage),
        checkpoint=None,
        should_suspend=lambda: False,
        checkpoint_on_report=True,
    )
    x = rt.suggest_float("x", -10, 10)
    for step in range(5):
        executed_steps.append(step)
        rt.report(sum((x - 2) ** 2 + s for s in range(step + 1)), step)
    # crash: no release, no state change
    assert ex1 is not None

    time.sleep(0.4)
    ex2 = make_executor(study, executor_id="exec-2", lease_store=lease_store, lease_ttl=0.3)
    assert ex2.reap_expired() == [trial_id]
    summary = ex2.run(objective, n_trials=1)
    assert summary.completed == 1

    # Steps 0-4 were executed exactly once; nothing was re-run.
    assert sorted(executed_steps) == list(range(N_STEPS))
    assert len(executed_steps) == N_STEPS
    trial = study.trials[0]
    assert trial.state == TrialState.COMPLETE
    assert sorted(trial.intermediate_values) == list(range(N_STEPS))


def test_corrupted_checkpoint_refused_and_preserved(tmp_path):
    study = make_study()
    checkpoint_store = FileCheckpointStore(tmp_path)
    executor = make_executor(study, checkpoint_store=checkpoint_store)
    reached_step_2 = threading.Event()

    def run_ex():
        executor.run(make_blocking_suspend_objective(2, reached_step_2), n_trials=1)

    thread = threading.Thread(target=run_ex)
    thread.start()
    assert reached_step_2.wait(timeout=10)
    executor.request_suspend()
    thread.join(timeout=10)
    trial_id = study.trials[0]._trial_id

    path = tmp_path / f"study_{study._study_id}_trial_{trial_id}.ckpt.json"
    path.write_text('{"format_version": 1, "trial_id": 999, "checksum": "tampered"}')

    ex2 = make_executor(study, executor_id="exec-2", checkpoint_store=checkpoint_store)
    with pytest.raises(CheckpointCorruptedError):
        ex2.run(make_objective(), n_trials=1)
    # The trial was not resumed from a partial state; the record is preserved.
    assert study.trials[0].state == TrialState.WAITING
    assert "tampered" in path.read_text()


def test_search_space_change_refused_with_diff():
    study = make_study()
    executor = make_executor(study)
    reached_step_2 = threading.Event()

    def run_ex():
        executor.run(make_blocking_suspend_objective(2, reached_step_2), n_trials=1)

    thread = threading.Thread(target=run_ex)
    thread.start()
    assert reached_step_2.wait(timeout=10)
    executor.request_suspend()
    thread.join(timeout=10)

    def changed_objective(trial):
        x = trial.suggest_float("x", -5, 5)  # different range than before suspension
        return x

    ex2 = make_executor(study, executor_id="exec-2")
    with pytest.raises(SearchSpaceChangedError) as excinfo:
        ex2.run(changed_objective, n_trials=1)
    assert "'x'" in str(excinfo.value)
    # The trial keeps its original parameters and stays resumable.
    trial = study.trials[0]
    assert trial.state == TrialState.WAITING
    assert -10 <= trial.params["x"] <= 10


def test_pruned_and_failed_trials_keep_standard_semantics():
    study = make_study()
    executor = make_executor(study)

    def pruning_objective(trial):
        trial.report(1.0, 0)
        raise optuna.TrialPruned()

    def failing_objective(trial):
        raise ValueError("boom")

    summary = executor.run(pruning_objective, n_trials=1)
    assert summary.pruned == 1
    assert study.trials[0].state == TrialState.PRUNED

    summary = executor.run(failing_objective, n_trials=1)
    assert summary.failed == 1
    assert study.trials[1].state == TrialState.FAIL
    # Leases of finished trials are always released.
    assert executor.lease_store.count_active(study._study_id) == 0


def test_lease_ttl_none_disables_heartbeat_and_reaper():
    study = make_study()
    executor = make_executor(study, lease_ttl=None)
    assert executor.reap_expired() == []
    summary = executor.run(make_objective(), n_trials=1)
    assert summary.completed == 1
