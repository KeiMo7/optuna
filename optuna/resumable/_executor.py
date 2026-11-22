"""Lease-based executor that can suspend, resume and take over trials."""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Sequence
import math
import os
import socket
import threading
import time
from typing import NamedTuple
import uuid

import optuna
from optuna import exceptions
from optuna._experimental import experimental_class
from optuna.resumable._checkpoint import BaseCheckpointStore
from optuna.resumable._checkpoint import Checkpoint
from optuna.resumable._checkpoint import FileCheckpointStore
from optuna.resumable._checkpoint import StorageCheckpointStore
from optuna.resumable._errors import _SuspendRequested
from optuna.resumable._errors import CheckpointCorruptedError
from optuna.resumable._errors import LeaseConflictError
from optuna.resumable._errors import LeaseLostError
from optuna.resumable._errors import ResumableError
from optuna.resumable._errors import SearchSpaceChangedError
from optuna.resumable._errors import SlotExhaustedError
from optuna.resumable._errors import TakeoverRejectedError
from optuna.resumable._lease import BaseLeaseStore
from optuna.resumable._lease import InMemoryLeaseStore
from optuna.resumable._lease import LeaseRecord
from optuna.resumable._lease import RDBLeaseStore
from optuna.resumable._trial import ResumableTrial
from optuna.storages import BaseStorage
from optuna.storages import InMemoryStorage
from optuna.storages import RDBStorage
from optuna.storages._cached_storage import _CachedStorage
from optuna.trial import TrialState


_logger = optuna.logging.get_logger(__name__)


class RunSummary(NamedTuple):
    """Outcome counters of one :meth:`ResumableExecutor.run` call."""

    completed: int
    pruned: int
    failed: int
    suspended: int
    reaped: int


def _unwrap_storage(storage: BaseStorage) -> BaseStorage:
    if isinstance(storage, _CachedStorage):
        return storage._backend
    return storage


@experimental_class("5.3.0")
class ResumableExecutor:
    """Run trials of a study with suspend/resume and lease-based failover.

    The executor is a purely additive layer on top of any existing study:

    - **Suspend**: the objective calls ``trial.suspend(payload)`` (or reacts to
      :meth:`request_suspend` via ``trial.should_suspend()``). Progress and the
      reported intermediate values are checkpointed, the trial goes back to a
      resumable (waiting) state, and its lease — i.e. its concurrency slot —
      is released.
    - **Resume**: any executor (another process or machine) pops the trial,
      validates the checkpoint (version, checksum, search space, intermediate
      history) and continues the objective from ``trial.start_step``.
    - **Failover**: a crashed executor stops renewing its lease. Once the
      lease expires, another executor takes it over atomically; the fencing
      token is bumped so the crashed executor can never write again, even if
      it briefly comes back.

    Args:
        study: The study to run. Any storage backend works; leases and
            checkpoints default to matching stores for
            :class:`~optuna.storages.InMemoryStorage` and
            :class:`~optuna.storages.RDBStorage`.
        executor_id: Unique id of this executor. Defaults to
            ``<hostname>-<pid>-<random>``.
        lease_store: Custom lease ledger. Defaults to an
            :class:`~optuna.resumable.InMemoryLeaseStore` for in-memory
            studies and an :class:`~optuna.resumable.RDBLeaseStore` on the
            same database URL for RDB studies.
        checkpoint_store: Custom checkpoint store. Defaults to a
            :class:`~optuna.resumable.FileCheckpointStore` when
            ``checkpoint_dir`` is given, otherwise checkpoints are kept as
            trial system attributes of the study's own storage.
        checkpoint_dir: Directory for file-based checkpoints (use a shared
            file system for multi-machine clusters).
        max_concurrent: Maximum number of trials running concurrently in the
            study across *all* executors. ``None`` means unlimited.
        lease_ttl: Seconds a lease stays valid without renewal. ``None``
            disables expiry and heartbeats (single-process mode).
        heartbeat_interval: Seconds between lease renewals. Must be shorter
            than ``lease_ttl``.
        checkpoint_on_report: If :obj:`True`, every ``trial.report`` also
            refreshes the checkpoint, so a crashed trial resumes from its
            last reported step instead of from scratch.
        poll_interval: Seconds to wait before retrying when no concurrency
            slot is free.
    """

    def __init__(
        self,
        study: optuna.study.Study,
        *,
        executor_id: str | None = None,
        lease_store: BaseLeaseStore | None = None,
        checkpoint_store: BaseCheckpointStore | None = None,
        checkpoint_dir: str | os.PathLike[str] | None = None,
        max_concurrent: int | None = None,
        lease_ttl: float | None = 30.0,
        heartbeat_interval: float | None = 10.0,
        checkpoint_on_report: bool = True,
        poll_interval: float = 0.5,
    ) -> None:
        self._study = study
        self._storage = _unwrap_storage(study._storage)
        self._study_id = study._study_id
        self._executor_id = executor_id or (
            f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        )
        if lease_store is None:
            if isinstance(self._storage, RDBStorage):
                lease_store = RDBLeaseStore(self._storage.url)
            elif isinstance(self._storage, InMemoryStorage):
                lease_store = InMemoryLeaseStore()
            else:
                raise ValueError(
                    "Cannot infer a lease store for storage "
                    f"{type(self._storage).__name__}; pass lease_store explicitly."
                )
        if checkpoint_store is None:
            if checkpoint_dir is not None:
                checkpoint_store = FileCheckpointStore(checkpoint_dir)
            else:
                checkpoint_store = StorageCheckpointStore(self._storage)
        if lease_ttl is None:
            heartbeat_interval = None
        elif heartbeat_interval is None or heartbeat_interval >= lease_ttl:
            heartbeat_interval = lease_ttl / 3
        self._leases = lease_store
        self._checkpoints = checkpoint_store
        self._max_concurrent = max_concurrent
        self._lease_ttl = lease_ttl
        self._heartbeat_interval = heartbeat_interval
        self._checkpoint_on_report = checkpoint_on_report
        self._poll_interval = poll_interval

        self._suspend_event = threading.Event()
        self._lease_lost = threading.Event()

    # ------------------------------------------------------------------
    # Properties.
    # ------------------------------------------------------------------
    @property
    def executor_id(self) -> str:
        return self._executor_id

    @property
    def study(self) -> optuna.study.Study:
        return self._study

    @property
    def lease_store(self) -> BaseLeaseStore:
        return self._leases

    @property
    def checkpoint_store(self) -> BaseCheckpointStore:
        return self._checkpoints

    # ------------------------------------------------------------------
    # External control.
    # ------------------------------------------------------------------
    def request_suspend(self) -> None:
        """Ask the currently running trial to suspend at its next opportunity."""

        self._suspend_event.set()

    def cancel_trial(self, trial_id: int) -> None:
        """Cancel a trial and reclaim its slot, checkpoint and temporary state.

        The lease (concurrency slot) is released — taking it over first if the
        holder's lease already expired — the checkpoint is deleted, and an
        unfinished trial is moved to ``FAIL`` so it can never be resumed from
        reclaimed state.
        """

        lease = self._leases.get(trial_id)
        if lease is not None:
            if lease.executor_id == self._executor_id:
                self._leases.release(lease)
            elif lease.is_expired():
                try:
                    taken = self._leases.takeover(trial_id, self._executor_id, self._lease_ttl)
                    self._leases.release(taken)
                except TakeoverRejectedError:
                    pass
            else:
                raise LeaseConflictError(trial_id, lease.executor_id, self._executor_id)
        self._checkpoints.delete(self._study_id, trial_id)
        frozen = self._storage.get_trial(trial_id)
        if not frozen.state.is_finished():
            self._storage.set_trial_state_values(trial_id, TrialState.FAIL)

    # ------------------------------------------------------------------
    # Reaping.
    # ------------------------------------------------------------------
    def reap_expired(self) -> list[int]:
        """Take over RUNNING trials whose lease expired and make them resumable.

        Returns the ids of the trials this call reclaimed. The takeover is a
        single atomic compare-and-swap on the lease ledger, so concurrent
        reapers cannot both succeed.
        """

        if self._lease_ttl is None:
            return []
        now = time.time()
        reaped: list[int] = []
        running = self._storage.get_all_trials(
            self._study_id, deepcopy=False, states=(TrialState.RUNNING,)
        )
        for frozen in running:
            trial_id = frozen._trial_id
            lease = self._leases.get(trial_id)
            if lease is not None and not lease.is_expired(now):
                continue
            if lease is None:
                # RUNNING without a lease row: only possible in the tiny window
                # between ask() and acquire(), or after a crash in that window.
                # Bound the window by the lease TTL before reclaiming.
                started = frozen.datetime_start
                if started is None or (now - started.timestamp()) <= self._lease_ttl:
                    continue
            else:
                try:
                    taken = self._leases.takeover(trial_id, self._executor_id, self._lease_ttl)
                except TakeoverRejectedError:
                    continue
                # The fencing token was bumped by the takeover; the old holder is
                # fenced out from now on even if it briefly revives.
                self._leases.release(taken)
            try:
                self._storage.set_trial_state_values(trial_id, TrialState.WAITING)
            except exceptions.UpdateFinishedTrialError:
                continue
            reaped.append(trial_id)
            _logger.info(
                f"Executor {self._executor_id!r} reclaimed trial {trial_id} after lease expiry."
            )
        return reaped

    # ------------------------------------------------------------------
    # Main loop.
    # ------------------------------------------------------------------
    def run(
        self,
        objective: Callable[[ResumableTrial], float],
        n_trials: int | None = None,
        timeout: float | None = None,
        catch: Sequence[type[Exception]] = (Exception,),
    ) -> RunSummary:
        """Run trials until ``n_trials`` of them finished (or ``timeout`` hits).

        Suspended trials do not count towards ``n_trials``; they are picked up
        again by this or another executor. Pruned and failed trials follow the
        exact semantics of :meth:`optuna.study.Study.optimize`.
        """

        if n_trials is None and timeout is None:
            raise ValueError("Either n_trials or timeout must be specified.")
        completed = pruned = failed = suspended = reaped = 0
        deadline = None if timeout is None else time.time() + timeout
        catch_tuple = tuple(catch)
        self._suspend_event.clear()
        while n_trials is None or completed + pruned + failed < n_trials:
            if deadline is not None and time.time() >= deadline:
                break
            reaped += len(self.reap_expired())
            trial = self._study.ask()
            trial_id = trial._trial_id
            self._lease_lost.clear()
            try:
                lease = self._leases.acquire(
                    trial_id,
                    self._study_id,
                    self._executor_id,
                    self._lease_ttl,
                    self._max_concurrent,
                )
            except (SlotExhaustedError, LeaseConflictError):
                self._requeue(trial_id)
                time.sleep(self._poll_interval)
                continue
            try:
                checkpoint = self._load_and_verify(trial_id)
            except ResumableError:
                self._requeue(trial_id)
                self._leases.release(lease)
                raise
            stop_heartbeat = self._start_heartbeat(lease)
            rt = ResumableTrial(
                trial,
                study_id=self._study_id,
                lease=lease,
                lease_store=self._leases,
                checkpoint_store=self._checkpoints,
                checkpoint=checkpoint,
                should_suspend=self._suspend_event.is_set,
                checkpoint_on_report=self._checkpoint_on_report,
            )
            try:
                value = objective(rt)
                self._leases.check(lease)
                if self._tell_safely(trial, value):
                    completed += 1
            except _SuspendRequested:
                suspended += 1
                self._set_state_if_owned(lease, trial_id, TrialState.WAITING)
                if self._suspend_event.is_set():
                    # The suspension was requested externally (e.g. the machine
                    # is being reclaimed): wind down instead of picking the
                    # trial up again ourselves.
                    break
            except LeaseLostError:
                # Another executor owns the trial now; touching its state or
                # writing results would violate at-most-once execution.
                _logger.warning(
                    f"Executor {self._executor_id!r} lost the lease of trial "
                    f"{trial_id}; abandoning it to the new owner."
                )
            except SearchSpaceChangedError:
                self._set_state_if_owned(lease, trial_id, TrialState.WAITING)
                raise
            except exceptions.TrialPruned:
                if self._tell_safely(trial, None, state=TrialState.PRUNED):
                    pruned += 1
            except catch_tuple as e:
                _logger.warning(f"Trial {trial_id} failed with: {e!r}")
                if self._tell_safely(trial, None, state=TrialState.FAIL):
                    failed += 1
            finally:
                stop_heartbeat()
                self._leases.release(lease)
        return RunSummary(
            completed=completed,
            pruned=pruned,
            failed=failed,
            suspended=suspended,
            reaped=reaped,
        )

    # ------------------------------------------------------------------
    # Internals.
    # ------------------------------------------------------------------
    def _requeue(self, trial_id: int) -> None:
        try:
            self._storage.set_trial_state_values(trial_id, TrialState.WAITING)
        except exceptions.UpdateFinishedTrialError:
            pass

    def _tell_safely(
        self,
        trial: optuna.trial.Trial,
        value: float | None,
        state: TrialState | None = None,
    ) -> bool:
        """Tell, but only while this executor still plausibly owns the trial.

        A concurrent reap can requeue or finish the trial between the objective
        and the tell; in that case the tell is skipped instead of failing.
        """

        try:
            frozen = self._study.tell(trial, value, state=state)
            return frozen is not None
        except ValueError:
            _logger.warning(
                f"Skipping tell for trial {trial._trial_id}: its state changed "
                "concurrently (likely reclaimed after lease expiry)."
            )
            return False

    def _set_state_if_owned(self, lease: LeaseRecord, trial_id: int, state: TrialState) -> None:
        try:
            self._leases.check(lease)
        except LeaseLostError:
            return
        try:
            self._storage.set_trial_state_values(trial_id, state)
        except exceptions.UpdateFinishedTrialError:
            pass

    def _start_heartbeat(self, lease: LeaseRecord) -> Callable[[], None]:
        if self._heartbeat_interval is None:
            return lambda: None
        stop = threading.Event()
        holder: dict[str, LeaseRecord] = {"lease": lease}

        def _loop() -> None:
            while not stop.wait(self._heartbeat_interval):
                try:
                    holder["lease"] = self._leases.renew(holder["lease"], self._lease_ttl)
                except LeaseLostError:
                    self._lease_lost.set()
                    return
                except Exception as e:  # Transient store errors must not kill the heartbeat.
                    _logger.warning(
                        f"Heartbeat for trial {lease.trial_id} failed with {e!r}; retrying."
                    )

        thread = threading.Thread(target=_loop, name="resumable-heartbeat", daemon=True)
        thread.start()

        def _stop() -> None:
            stop.set()
            thread.join(timeout=self._heartbeat_interval + 1)

        return _stop

    def _load_and_verify(self, trial_id: int) -> Checkpoint | None:
        checkpoint = self._checkpoints.load(self._study_id, trial_id)
        if checkpoint is None:
            return None
        source = f"checkpoint store {type(self._checkpoints).__name__}"
        if checkpoint.trial_id != trial_id or checkpoint.study_id != self._study_id:
            raise CheckpointCorruptedError(
                trial_id,
                source,
                f"record belongs to study {checkpoint.study_id} / trial {checkpoint.trial_id}",
            )
        frozen = self._storage.get_trial(trial_id)
        # The search space must be exactly the one recorded at suspend time.
        import optuna.distributions as _distributions

        current_distributions = {
            name: _distributions.distribution_to_json(dist)
            for name, dist in frozen.distributions.items()
        }
        differences: list[str] = []
        for name in sorted(set(checkpoint.distributions) | set(current_distributions)):
            old = checkpoint.distributions.get(name)
            new = current_distributions.get(name)
            if old is None:
                differences.append(f"parameter {name!r} was added after suspension: {new}")
            elif new is None:
                differences.append(f"parameter {name!r} is missing after suspension: {old}")
            elif old != new:
                differences.append(f"parameter {name!r}: distribution changed from {old} to {new}")
        if differences:
            raise SearchSpaceChangedError(trial_id, differences)
        # The reported history in the storage must cover the checkpoint exactly.
        stored = frozen.intermediate_values
        for step, value in checkpoint.intermediate_values.items():
            if step not in stored:
                raise CheckpointCorruptedError(
                    trial_id,
                    source,
                    f"reported value of step {step} is missing from the storage",
                )
            stored_value = stored[step]
            if stored_value != value and not (math.isnan(stored_value) and math.isnan(value)):
                raise CheckpointCorruptedError(
                    trial_id,
                    source,
                    f"reported value of step {step} diverged: checkpoint has "
                    f"{value}, storage has {stored_value}",
                )
        # A crash between report() and the checkpoint write can leave the
        # storage one step ahead; the storage is authoritative for reported
        # values, so adopt the extra steps instead of re-reporting them.
        if stored:
            adopted_next_step = max(checkpoint.next_step, max(stored) + 1)
            if adopted_next_step != checkpoint.next_step or len(stored) != len(
                checkpoint.intermediate_values
            ):
                checkpoint = Checkpoint(
                    study_id=checkpoint.study_id,
                    trial_id=checkpoint.trial_id,
                    next_step=adopted_next_step,
                    intermediate_values=dict(stored),
                    distributions=checkpoint.distributions,
                    params=checkpoint.params,
                    payload=checkpoint.payload,
                    executor_id=checkpoint.executor_id,
                    fencing_token=checkpoint.fencing_token,
                )
        return checkpoint
