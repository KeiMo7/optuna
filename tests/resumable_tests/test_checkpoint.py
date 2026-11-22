from __future__ import annotations

import json

import pytest

import optuna
from optuna.resumable import Checkpoint
from optuna.resumable import CheckpointCorruptedError
from optuna.resumable import CheckpointVersionError
from optuna.resumable import FileCheckpointStore
from optuna.resumable import StorageCheckpointStore
from optuna.resumable._checkpoint import _SYSTEM_ATTR_KEY
from optuna.resumable._checkpoint import CHECKPOINT_FORMAT_VERSION
from optuna.resumable._checkpoint import dumps_checkpoint


def make_checkpoint(**overrides) -> Checkpoint:
    kwargs = dict(
        study_id=1,
        trial_id=2,
        next_step=5,
        intermediate_values={0: 1.0, 1: 0.5},
        distributions={"x": '{"name": "FloatDistribution", "attributes": {}}'},
        params={"x": 0.5},
        payload={"total": 1.5},
        executor_id="exec-a",
        fencing_token=3,
    )
    kwargs.update(overrides)
    return Checkpoint(**kwargs)


def test_file_store_roundtrip(tmp_path):
    store = FileCheckpointStore(tmp_path)
    ckpt = make_checkpoint()
    store.save(ckpt)
    assert store.load(1, 2) == ckpt
    store.delete(1, 2)
    assert store.load(1, 2) is None


def test_file_store_missing_returns_none(tmp_path):
    assert FileCheckpointStore(tmp_path).load(1, 999) is None


def test_file_store_corrupted_content_refused_and_preserved(tmp_path):
    store = FileCheckpointStore(tmp_path)
    ckpt = make_checkpoint()
    store.save(ckpt)
    path = tmp_path / "study_1_trial_2.ckpt.json"
    original = path.read_bytes()
    tampered = original.replace(b'"total": 1.5', b'"total": 9.9')
    assert tampered != original
    path.write_bytes(tampered)  # valid JSON, but the checksum no longer matches
    with pytest.raises(CheckpointCorruptedError) as excinfo:
        store.load(1, 2)
    assert excinfo.value.trial_id == 2
    # The original record is left untouched for inspection.
    assert path.read_bytes() == tampered


def test_file_store_invalid_json_refused(tmp_path):
    store = FileCheckpointStore(tmp_path)
    path = tmp_path / "study_1_trial_2.ckpt.json"
    path.write_text("{not json")
    with pytest.raises(CheckpointCorruptedError):
        store.load(1, 2)


def test_version_mismatch_refused(tmp_path):
    store = FileCheckpointStore(tmp_path)
    ckpt = make_checkpoint()
    store.save(ckpt)
    path = tmp_path / "study_1_trial_2.ckpt.json"
    record = json.loads(path.read_text())
    record["format_version"] = CHECKPOINT_FORMAT_VERSION + 1
    # Re-sign so that only the version check can fire.
    body = json.dumps({k: v for k, v in record.items() if k != "checksum"}, sort_keys=True)
    import hashlib

    record["checksum"] = hashlib.sha256(body.encode()).hexdigest()
    path.write_text(json.dumps(record))
    with pytest.raises(CheckpointVersionError) as excinfo:
        store.load(1, 2)
    assert excinfo.value.found == CHECKPOINT_FORMAT_VERSION + 1
    assert excinfo.value.expected == CHECKPOINT_FORMAT_VERSION


def test_storage_store_roundtrip():
    storage = optuna.storages.InMemoryStorage()
    study = optuna.create_study(storage=storage)
    trial_id = storage.create_new_trial(study._study_id)
    store = StorageCheckpointStore(storage)
    ckpt = make_checkpoint(trial_id=trial_id, study_id=study._study_id)
    store.save(ckpt)
    loaded = store.load(study._study_id, trial_id)
    assert loaded == ckpt
    store.delete(study._study_id, trial_id)
    assert store.load(study._study_id, trial_id) is None


def test_storage_store_corruption_detected():
    storage = optuna.storages.InMemoryStorage()
    study = optuna.create_study(storage=storage)
    trial_id = storage.create_new_trial(study._study_id)
    store = StorageCheckpointStore(storage)
    store.save(make_checkpoint(trial_id=trial_id, study_id=study._study_id))
    storage.set_trial_system_attr(
        trial_id, _SYSTEM_ATTR_KEY, '{"format_version": 1, "checksum": "bogus"}'
    )
    with pytest.raises(CheckpointCorruptedError):
        store.load(study._study_id, trial_id)


def test_checkpoint_serialization_is_stable():
    ckpt = make_checkpoint()
    assert dumps_checkpoint(ckpt) == dumps_checkpoint(ckpt)
