"""Resumable trial execution with leases, checkpoints and failover.

.. note::
    This is an experimental, purely additive layer. Nothing here changes the
    behavior of :meth:`optuna.study.Study.optimize`, pruning, or any storage
    backend unless you explicitly run trials through
    :class:`~optuna.resumable.ResumableExecutor`.
"""

from optuna.resumable._checkpoint import BaseCheckpointStore
from optuna.resumable._checkpoint import Checkpoint
from optuna.resumable._checkpoint import FileCheckpointStore
from optuna.resumable._checkpoint import StorageCheckpointStore
from optuna.resumable._errors import CheckpointCorruptedError
from optuna.resumable._errors import CheckpointError
from optuna.resumable._errors import CheckpointVersionError
from optuna.resumable._errors import LeaseConflictError
from optuna.resumable._errors import LeaseLostError
from optuna.resumable._errors import ResumableError
from optuna.resumable._errors import SearchSpaceChangedError
from optuna.resumable._errors import SlotExhaustedError
from optuna.resumable._errors import TakeoverRejectedError
from optuna.resumable._executor import ResumableExecutor
from optuna.resumable._executor import RunSummary
from optuna.resumable._lease import BaseLeaseStore
from optuna.resumable._lease import InMemoryLeaseStore
from optuna.resumable._lease import LeaseRecord
from optuna.resumable._lease import RDBLeaseStore
from optuna.resumable._trial import ResumableTrial


__all__ = [
    "BaseCheckpointStore",
    "BaseLeaseStore",
    "Checkpoint",
    "CheckpointCorruptedError",
    "CheckpointError",
    "CheckpointVersionError",
    "FileCheckpointStore",
    "InMemoryLeaseStore",
    "LeaseConflictError",
    "LeaseLostError",
    "LeaseRecord",
    "RDBLeaseStore",
    "ResumableError",
    "ResumableExecutor",
    "ResumableTrial",
    "RunSummary",
    "SearchSpaceChangedError",
    "SlotExhaustedError",
    "StorageCheckpointStore",
    "TakeoverRejectedError",
]
