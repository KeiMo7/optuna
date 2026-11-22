"""Backend and concurrency tests for evolving search spaces.

These cover requirement 一.7 (multiple workers on one study with different
space versions) and 一.8 (reopening a stored study on a new space), plus the
three-backend semantic parity demanded in 二.3.
"""

from __future__ import annotations

import multiprocessing
import os
import tempfile

import optuna
from optuna.evolving_space import EvolvingTPESampler
from optuna.evolving_space import load_snapshots
from optuna.evolving_space import mark_new_round
from optuna.storages import JournalStorage
from optuna.storages import RDBStorage
from optuna.storages.journal import JournalFileBackend


def _round1(trial: optuna.Trial) -> float:
    return float(trial.suggest_float("x", -5.0, 5.0) ** 2)


def _round2(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -5.0, 5.0)
    y = trial.suggest_int("y", 0, 3)
    return float(x**2 + y)


def _run_script(storage: optuna.storages.BaseStorage, seed: int) -> dict:
    sampler = EvolvingTPESampler(seed=seed, n_startup_trials=2)
    study = optuna.create_study(storage=storage, sampler=sampler)
    study.optimize(_round1, n_trials=5)
    study.optimize(_round2, n_trials=5)
    return {
        "n_trials": len(study.trials),
        "rounds": len(load_snapshots(study)),
        "round_dims": [sorted(snap["space"].keys()) for snap in load_snapshots(study)],
        "params": [dict(t.params) for t in study.trials],
        "values": [t.value for t in study.trials],
    }


def test_journal_matches_in_memory_backend() -> None:
    seed = 11
    memory = _run_script(optuna.storages.InMemoryStorage(), seed)
    with tempfile.TemporaryDirectory() as tmpdir:
        journal_path = os.path.join(tmpdir, "evolving.log")
        journal = JournalStorage(JournalFileBackend(file_path=journal_path))
        on_disk = _run_script(journal, seed)

    assert on_disk == memory
    assert on_disk["rounds"] == 2
    assert on_disk["round_dims"] == [["x"], ["x", "y"]]


def test_rdb_matches_in_memory_backend() -> None:
    seed = 13
    memory = _run_script(optuna.storages.InMemoryStorage(), seed)
    with tempfile.TemporaryDirectory() as tmpdir:
        rdb = RDBStorage(f"sqlite:///{os.path.join(tmpdir, 'e.db')}")
        on_disk = _run_script(rdb, seed)
    assert on_disk == memory


def test_reopen_stored_study_and_continue_with_new_space() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "reopen.db")
        storage = RDBStorage(f"sqlite:///{db_path}")
        sampler = EvolvingTPESampler(seed=5, n_startup_trials=2)
        study = optuna.create_study(storage=storage, sampler=sampler)
        study.optimize(_round1, n_trials=5)
        study_name = study.study_name
        del study

        # Reopen read-only-ish with a fresh sampler and evolve the space.
        reopened = optuna.load_study(
            study_name=study_name,
            storage=f"sqlite:///{db_path}",
            sampler=EvolvingTPESampler(seed=6, n_startup_trials=2),
        )
        assert len(load_snapshots(reopened)) == 1
        reopened.optimize(_round2, n_trials=4)

        assert len(reopened.trials) == 9
        assert len(load_snapshots(reopened)) == 2
        # Original five trials are untouched.
        for trial in reopened.trials[:5]:
            assert set(trial.params) == {"x"}


def _worker(url: str, study_name: str, n_trials: int) -> None:
    study = optuna.load_study(
        study_name=study_name,
        storage=url,
        sampler=EvolvingTPESampler(seed=0, n_startup_trials=1),
    )
    study.optimize(_round2, n_trials=n_trials)


def test_multiple_workers_with_evolved_space() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        url = f"sqlite:///{os.path.join(tmpdir, 'workers.db')}"
        storage = RDBStorage(url)
        study = optuna.create_study(
            storage=storage,
            study_name="worker-study",
            sampler=EvolvingTPESampler(seed=0, n_startup_trials=1),
        )
        study.optimize(_round1, n_trials=4)
        mark_new_round(study)

        ctx = multiprocessing.get_context("spawn")
        processes = [ctx.Process(target=_worker, args=(url, "worker-study", 3)) for _ in range(2)]
        for p in processes:
            p.start()
        for p in processes:
            p.join(timeout=120)
        assert all(p.exitcode == 0 for p in processes)

        final = optuna.load_study(study_name="worker-study", storage=url)
        assert len(final.trials) == 10
        # Every historical trial remains; round 1 trials lack y, round 2 have it.
        assert all("y" not in t.params for t in final.trials[:4])
        assert all("y" in t.params for t in final.trials[4:])
        snapshots = load_snapshots(final)
        assert len(snapshots) >= 2
