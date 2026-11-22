"""Versioned, integrity-checked checkpoints of suspended trials.

A checkpoint records everything needed to continue a trial from the exact step
at which it was suspended:

- ``next_step``: the first training step that has *not* been reported yet,
- ``intermediate_values``: the values reported so far (verified against the
  storage on resume, so the recorded history can never silently diverge),
- ``distributions`` / ``params``: the search space as it was when the trial
  was suspended (compared on resume; any change aborts the resume),
- ``payload``: free-form user data handed back to the objective on resume.

Every record carries a format version and a SHA-256 checksum. A corrupted or
version-mismatched record is rejected and left untouched, never resumed from.
"""

from __future__ import annotations

import abc
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from optuna.resumable._errors import CheckpointCorruptedError
from optuna.resumable._errors import CheckpointVersionError
from optuna.storages import BaseStorage


CHECKPOINT_FORMAT_VERSION = 1

_SYSTEM_ATTR_KEY = "resumable:checkpoint"


class Checkpoint:
    """The resumable state of one suspended trial."""

    def __init__(
        self,
        *,
        study_id: int,
        trial_id: int,
        next_step: int,
        intermediate_values: dict[int, float],
        distributions: dict[str, str],
        params: dict[str, Any],
        payload: dict[str, Any],
        executor_id: str,
        fencing_token: int,
    ) -> None:
        self.study_id = study_id
        self.trial_id = trial_id
        self.next_step = next_step
        self.intermediate_values = dict(intermediate_values)
        self.distributions = dict(distributions)
        self.params = dict(params)
        self.payload = dict(payload)
        self.executor_id = executor_id
        self.fencing_token = fencing_token

    def to_record(self) -> dict[str, Any]:
        return {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "study_id": self.study_id,
            "trial_id": self.trial_id,
            "next_step": self.next_step,
            "intermediate_values": {str(k): v for k, v in self.intermediate_values.items()},
            "distributions": self.distributions,
            "params": self.params,
            "payload": self.payload,
            "executor_id": self.executor_id,
            "fencing_token": self.fencing_token,
        }

    @classmethod
    def from_record(cls, record: dict[str, Any], *, source: str) -> "Checkpoint":
        version = record.get("format_version")
        if version != CHECKPOINT_FORMAT_VERSION:
            raise CheckpointVersionError(
                int(record.get("trial_id", -1)), source, version, CHECKPOINT_FORMAT_VERSION
            )
        try:
            return cls(
                study_id=int(record["study_id"]),
                trial_id=int(record["trial_id"]),
                next_step=int(record["next_step"]),
                intermediate_values={
                    int(k): float(v) for k, v in record["intermediate_values"].items()
                },
                distributions={str(k): str(v) for k, v in record["distributions"].items()},
                params=dict(record["params"]),
                payload=dict(record["payload"]),
                executor_id=str(record["executor_id"]),
                fencing_token=int(record["fencing_token"]),
            )
        except (KeyError, TypeError, ValueError) as e:
            raise CheckpointCorruptedError(
                int(record.get("trial_id", -1)), source, f"missing or invalid field: {e}"
            ) from e

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Checkpoint) and self.to_record() == other.to_record()

    def __repr__(self) -> str:
        return (
            f"Checkpoint(trial_id={self.trial_id}, next_step={self.next_step}, "
            f"payload={self.payload!r})"
        )


def _encode(record: dict[str, Any]) -> str:
    body = json.dumps(record, sort_keys=True)
    record = dict(record)
    record["checksum"] = hashlib.sha256(body.encode("utf-8")).hexdigest()
    return json.dumps(record, sort_keys=True)


def _decode(text: str, *, source: str) -> dict[str, Any]:
    try:
        record = json.loads(text)
    except (ValueError, TypeError) as e:
        raise CheckpointCorruptedError(-1, source, f"not valid JSON: {e}") from e
    if not isinstance(record, dict):
        raise CheckpointCorruptedError(-1, source, "top-level value is not an object")
    checksum = record.pop("checksum", None)
    if checksum is None:
        raise CheckpointCorruptedError(int(record.get("trial_id", -1)), source, "missing checksum")
    body = json.dumps(record, sort_keys=True)
    actual = hashlib.sha256(body.encode("utf-8")).hexdigest()
    if actual != checksum:
        raise CheckpointCorruptedError(
            int(record.get("trial_id", -1)), source, "checksum mismatch"
        )
    return record


def dumps_checkpoint(checkpoint: Checkpoint) -> str:
    """Serialize a checkpoint to a self-verifying JSON string."""

    return _encode(checkpoint.to_record())


def loads_checkpoint(text: str, *, source: str) -> Checkpoint:
    """Parse a checkpoint, rejecting corruption and version mismatches."""

    return Checkpoint.from_record(_decode(text, source=source), source=source)


class BaseCheckpointStore(metaclass=abc.ABCMeta):
    """Interface of checkpoint persistence used by the executor."""

    @abc.abstractmethod
    def save(self, checkpoint: Checkpoint) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def load(self, study_id: int, trial_id: int) -> Checkpoint | None:
        """Return the checkpoint, or ``None`` if the trial was never suspended.

        Raises:
            CheckpointCorruptedError: If the record fails integrity checks.
            CheckpointVersionError: If the record uses an unsupported format version.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def delete(self, study_id: int, trial_id: int) -> None:
        raise NotImplementedError


class FileCheckpointStore(BaseCheckpointStore):
    """Checkpoints as JSON files in a (possibly shared) directory.

    Writes are atomic (temporary file + rename), so a crash mid-write never
    leaves a half-written checkpoint behind.
    """

    def __init__(self, directory: str | os.PathLike[str]) -> None:
        self._directory = Path(directory)
        self._directory.mkdir(parents=True, exist_ok=True)

    def _path(self, study_id: int, trial_id: int) -> Path:
        return self._directory / f"study_{study_id}_trial_{trial_id}.ckpt.json"

    def save(self, checkpoint: Checkpoint) -> None:
        path = self._path(checkpoint.study_id, checkpoint.trial_id)
        fd, tmp_name = tempfile.mkstemp(
            dir=str(self._directory), prefix=path.name + ".", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(dumps_checkpoint(checkpoint))
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_name, path)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise

    def load(self, study_id: int, trial_id: int) -> Checkpoint | None:
        path = self._path(study_id, trial_id)
        source = str(path)
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except OSError as e:
            raise CheckpointCorruptedError(trial_id, source, f"unreadable: {e}") from e
        return loads_checkpoint(text, source=source)

    def delete(self, study_id: int, trial_id: int) -> None:
        try:
            self._path(study_id, trial_id).unlink()
        except FileNotFoundError:
            pass


class StorageCheckpointStore(BaseCheckpointStore):
    """Checkpoints stored as a trial system attribute of the same storage.

    This keeps the progress record in the exact same ledger as the trial data
    and needs no shared file system. The payload must be JSON-serializable.
    """

    def __init__(self, storage: BaseStorage) -> None:
        self._storage = storage

    def save(self, checkpoint: Checkpoint) -> None:
        self._storage.set_trial_system_attr(
            checkpoint.trial_id, _SYSTEM_ATTR_KEY, dumps_checkpoint(checkpoint)
        )

    def load(self, study_id: int, trial_id: int) -> Checkpoint | None:
        raw = self._storage.get_trial(trial_id).system_attrs.get(_SYSTEM_ATTR_KEY)
        if raw is None:
            return None
        if not isinstance(raw, str):
            raise CheckpointCorruptedError(
                trial_id, f"system attr {_SYSTEM_ATTR_KEY!r}", "value is not a string"
            )
        return loads_checkpoint(raw, source=f"system attr {_SYSTEM_ATTR_KEY!r}")

    def delete(self, study_id: int, trial_id: int) -> None:
        self._storage.set_trial_system_attr(trial_id, _SYSTEM_ATTR_KEY, None)
