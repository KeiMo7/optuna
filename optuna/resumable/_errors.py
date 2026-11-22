"""Error types for the resumable-execution layer.

Every error carries enough context (trial id, executor id, fencing token where
applicable) to locate the conflicting trial and executor from a log line.
"""

from __future__ import annotations

from typing import Any


class ResumableError(RuntimeError):
    """Base class of all errors raised by :mod:`optuna.resumable`."""


class LeaseConflictError(ResumableError):
    """A trial already has an active lease held by another executor."""

    def __init__(self, trial_id: int, holder: str | None, requester: str) -> None:
        self.trial_id = trial_id
        self.holder = holder
        self.requester = requester
        super().__init__(
            f"Trial {trial_id} is already leased by executor {holder!r}; "
            f"executor {requester!r} cannot acquire it. "
            "If the holder is dead, wait for the lease to expire and take it over."
        )


class SlotExhaustedError(ResumableError):
    """No free concurrency slot is available for the study."""

    def __init__(self, study_id: int, max_concurrent: int, requester: str) -> None:
        self.study_id = study_id
        self.max_concurrent = max_concurrent
        self.requester = requester
        super().__init__(
            f"All {max_concurrent} concurrency slot(s) of study {study_id} are in use; "
            f"executor {requester!r} must wait for a lease to be released or to expire."
        )


class LeaseLostError(ResumableError):
    """The executor no longer holds a valid lease for the trial.

    Raised when a write is attempted with a stale fencing token, an expired
    lease, or after another executor took the trial over. The trial must not
    be touched by the raising executor any more.
    """

    def __init__(
        self,
        trial_id: int,
        executor_id: str,
        fencing_token: int,
        reason: str,
        current_holder: str | None = None,
    ) -> None:
        self.trial_id = trial_id
        self.executor_id = executor_id
        self.fencing_token = fencing_token
        self.current_holder = current_holder
        super().__init__(
            f"Executor {executor_id!r} lost the lease of trial {trial_id} "
            f"(fencing token {fencing_token}): {reason}."
            + (f" The trial is now held by executor {current_holder!r}." if current_holder else "")
        )


class TakeoverRejectedError(ResumableError):
    """A takeover attempt lost the race against another executor."""

    def __init__(self, trial_id: int, requester: str, current_holder: str | None) -> None:
        self.trial_id = trial_id
        self.requester = requester
        self.current_holder = current_holder
        super().__init__(
            f"Executor {requester!r} could not take over trial {trial_id}: "
            + (
                f"the lease is now held by executor {current_holder!r}."
                if current_holder
                else "the lease no longer exists or is not expired."
            )
        )


class CheckpointError(ResumableError):
    """Base class of checkpoint-related errors."""


class CheckpointCorruptedError(CheckpointError):
    """The stored progress cannot be decoded or fails integrity checks.

    The original checkpoint record is always left untouched so that the trial
    can be inspected or retried; the trial is never resumed from a partial
    state.
    """

    def __init__(self, trial_id: int, source: str, reason: str) -> None:
        self.trial_id = trial_id
        self.source = source
        super().__init__(
            f"Checkpoint of trial {trial_id} at {source!r} is corrupted: {reason}. "
            "Resume is refused and the original record is preserved."
        )


class CheckpointVersionError(CheckpointError):
    """The checkpoint was written by an incompatible format version."""

    def __init__(self, trial_id: int, source: str, found: Any, expected: int) -> None:
        self.trial_id = trial_id
        self.source = source
        self.found = found
        self.expected = expected
        super().__init__(
            f"Checkpoint of trial {trial_id} at {source!r} has format version "
            f"{found!r}, but this build only supports version {expected}. "
            "Resume is refused and the original record is preserved."
        )


class SearchSpaceChangedError(ResumableError):
    """The search space observed at resume differs from the suspended one."""

    def __init__(self, trial_id: int, differences: list[str]) -> None:
        self.trial_id = trial_id
        self.differences = list(differences)
        diff_text = "; ".join(self.differences) or "unknown difference"
        super().__init__(
            f"Search space of trial {trial_id} changed since it was suspended: "
            f"{diff_text}. Resume is refused so that no parameter is silently dropped."
        )


class _SuspendRequested(Exception):
    """Internal control-flow exception used to unwind a suspending objective."""
