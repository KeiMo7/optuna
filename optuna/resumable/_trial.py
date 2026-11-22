"""The trial handle passed to objectives running under :class:`ResumableExecutor`."""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Sequence
import datetime
from typing import Any

from optuna import distributions
from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalChoiceType
from optuna.resumable._checkpoint import BaseCheckpointStore
from optuna.resumable._checkpoint import Checkpoint
from optuna.resumable._errors import _SuspendRequested
from optuna.resumable._errors import SearchSpaceChangedError
from optuna.resumable._lease import BaseLeaseStore
from optuna.resumable._lease import LeaseRecord
from optuna.trial import Trial


class ResumableTrial:
    """A fenced, checkpointing wrapper around :class:`~optuna.trial.Trial`.

    Every mutating operation first validates the executor's lease (fencing
    token). If the trial was taken over by another executor, the stale
    executor's next operation raises
    :class:`~optuna.resumable.LeaseLostError` instead of writing anything.

    The objective receives an instance of this class (not a plain
    :class:`~optuna.trial.Trial`) when run through
    :class:`~optuna.resumable.ResumableExecutor`.
    """

    def __init__(
        self,
        trial: Trial,
        *,
        study_id: int,
        lease: LeaseRecord,
        lease_store: BaseLeaseStore,
        checkpoint_store: BaseCheckpointStore,
        checkpoint: Checkpoint | None,
        should_suspend: Callable[[], bool],
        checkpoint_on_report: bool,
    ) -> None:
        self._trial = trial
        self._study_id = study_id
        self._lease = lease
        self._lease_store = lease_store
        self._checkpoint_store = checkpoint_store
        self._checkpoint = checkpoint
        self._should_suspend_cb = should_suspend
        self._checkpoint_on_report = checkpoint_on_report

        self._next_step = checkpoint.next_step if checkpoint is not None else 0
        self._intermediate_values: dict[int, float] = (
            dict(checkpoint.intermediate_values) if checkpoint is not None else {}
        )
        self._last_payload: dict[str, Any] = (
            dict(checkpoint.payload) if checkpoint is not None else {}
        )

    # ------------------------------------------------------------------
    # Introspection (read-only, no fencing required).
    # ------------------------------------------------------------------
    @property
    def trial_id(self) -> int:
        """The storage-level id of the wrapped trial."""

        return self._trial._trial_id

    @property
    def number(self) -> int:
        return self._trial.number

    @property
    def params(self) -> dict[str, Any]:
        return self._trial.params

    @property
    def distributions(self) -> dict[str, BaseDistribution]:
        return self._trial.distributions

    @property
    def user_attrs(self) -> dict[str, Any]:
        return self._trial.user_attrs

    @property
    def datetime_start(self) -> datetime.datetime | None:
        return self._trial.datetime_start

    @property
    def resumed(self) -> bool:
        """Whether this execution continues a previously suspended trial."""

        return self._checkpoint is not None

    @property
    def start_step(self) -> int:
        """The first step that still needs to be executed (0 for fresh trials)."""

        return self._next_step

    @property
    def checkpoint_payload(self) -> dict[str, Any] | None:
        """The user payload stored at the last suspend/checkpoint, if any."""

        return dict(self._last_payload) if self._checkpoint is not None else None

    @property
    def restored_intermediate_values(self) -> dict[int, float]:
        """Intermediate values reported before the suspension, in step order."""

        return dict(self._intermediate_values)

    # ------------------------------------------------------------------
    # Fencing.
    # ------------------------------------------------------------------
    def _check_lease(self) -> None:
        self._lease_store.check(self._lease)

    def _check_search_space(self, name: str, distribution: BaseDistribution) -> None:
        if self._checkpoint is None:
            return
        recorded = self._checkpoint.distributions.get(name)
        if recorded is None:
            # A parameter first suggested after the suspend point is legitimate.
            return
        new_json = distributions.distribution_to_json(distribution)
        if new_json != recorded:
            raise SearchSpaceChangedError(
                self.trial_id,
                [f"parameter {name!r}: distribution changed from {recorded} to {new_json}"],
            )

    # ------------------------------------------------------------------
    # Suggestions (fenced + search-space checked).
    # ------------------------------------------------------------------
    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        *,
        step: float | None = None,
        log: bool = False,
    ) -> float:
        self._check_lease()
        dist = distributions.FloatDistribution(low, high, step=step, log=log)
        self._check_search_space(name, dist)
        return self._trial.suggest_float(name, low, high, step=step, log=log)

    def suggest_int(self, name: str, low: int, high: int, step: int = 1, log: bool = False) -> int:
        self._check_lease()
        dist = distributions.IntDistribution(low, high, step=step, log=log)
        self._check_search_space(name, dist)
        return self._trial.suggest_int(name, low, high, step=step, log=log)

    def suggest_categorical(
        self, name: str, choices: Sequence[CategoricalChoiceType]
    ) -> CategoricalChoiceType:
        self._check_lease()
        dist = distributions.CategoricalDistribution(choices)
        self._check_search_space(name, dist)
        return self._trial.suggest_categorical(name, choices)

    def suggest_uniform(self, name: str, low: float, high: float) -> float:
        self._check_lease()
        return self._trial.suggest_uniform(name, low, high)

    def suggest_loguniform(self, name: str, low: float, high: float) -> float:
        self._check_lease()
        return self._trial.suggest_loguniform(name, low, high)

    def suggest_discrete_uniform(self, name: str, low: float, high: float, q: float) -> float:
        self._check_lease()
        return self._trial.suggest_discrete_uniform(name, low, high, q)

    # ------------------------------------------------------------------
    # Reporting (fenced + auto-checkpointing).
    # ------------------------------------------------------------------
    def report(self, value: float, step: int) -> None:
        self._check_lease()
        self._trial.report(value, step)
        self._intermediate_values[step] = float(value)
        self._next_step = max(self._next_step, step + 1)
        if self._checkpoint_on_report:
            self._write_checkpoint()

    def should_prune(self) -> bool:
        return self._trial.should_prune()

    def set_user_attr(self, key: str, value: Any) -> None:
        self._check_lease()
        self._trial.set_user_attr(key, value)

    # ------------------------------------------------------------------
    # Suspension.
    # ------------------------------------------------------------------
    def should_suspend(self) -> bool:
        """Whether the executor was asked to suspend the current trial."""

        return self._should_suspend_cb()

    def save_checkpoint(self, payload: dict[str, Any] | None = None) -> None:
        """Persist the current progress without suspending.

        ``payload`` replaces the user payload handed back on resume. Calling
        this periodically lets a crashed trial resume from meaningful user
        state instead of only from the last reported step.
        """

        self._check_lease()
        if payload is not None:
            self._last_payload = dict(payload)
        self._write_checkpoint()

    def suspend(self, payload: dict[str, Any] | None = None) -> None:
        """Suspend the trial: save progress and release it for another executor.

        This method never returns normally; it unwinds the objective via an
        internal exception which the executor turns into a state transition
        (RUNNING -> resumable) plus a lease/slot release.
        """

        self.save_checkpoint(payload)
        raise _SuspendRequested()

    def _write_checkpoint(self) -> None:
        trial = self._trial
        checkpoint = Checkpoint(
            study_id=self._study_id,
            trial_id=self.trial_id,
            next_step=self._next_step,
            intermediate_values=self._intermediate_values,
            distributions={
                name: distributions.distribution_to_json(dist)
                for name, dist in trial.distributions.items()
            },
            params=dict(trial.params),
            payload=self._last_payload,
            executor_id=self._lease.executor_id,
            fencing_token=self._lease.fencing_token,
        )
        self._checkpoint_store.save(checkpoint)
