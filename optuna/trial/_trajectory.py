from __future__ import annotations

import datetime
import json
import os
import tempfile
from typing import Any
from typing import TYPE_CHECKING

from optuna.exceptions import DuplicatedStepError
from optuna.exceptions import TrajectoryCorruptedError
from optuna.exceptions import UpdateFinishedTrialError
from optuna.trial._state import TrialState


if TYPE_CHECKING:
    from optuna.storages import BaseStorage
    from optuna.trial import FrozenTrial


# ---------------------------------------------------------------------------
# Storage keys / event kinds
# ---------------------------------------------------------------------------
#
# The trajectory is persisted as a *single* trial system attribute. Trial system
# attributes are supported by every storage backend (in-memory, RDB, journal and the
# gRPC proxy), so the feature behaves identically across backends without any schema
# migration. The whole ordered event log is rewritten atomically on each flush.
_TRAJ_SYSTEM_ATTR_KEY = "optuna:trajectory:v1"
_STUDY_ENABLED_SYSTEM_ATTR_KEY = "optuna:trajectory:enabled"

_EVENT_RUNNING = "running"
_EVENT_STEP = "step"
_EVENT_PRUNE_DECISION = "prune_decision"
_EVENT_STATE = "state"
_EVENT_FAIL = "fail"

_TERMINAL_EVENTS = (_EVENT_STATE, _EVENT_FAIL)
_VALID_EVENTS = frozenset(
    {_EVENT_RUNNING, _EVENT_STEP, _EVENT_PRUNE_DECISION, _EVENT_STATE, _EVENT_FAIL}
)

# Persist the growing snapshot at most once every ``_FLUSH_BATCH`` appended events, and
# always on the critical events (prune decision, failure, terminal state). This bounds
# the number of full-snapshot rewrites per trial to O(n / batch) instead of O(n), while
# keeping the durability window small and requiring no background thread or extra lock.
_FLUSH_BATCH = 16


def _now_iso() -> str:
    # ISO-8601 with microseconds. Used only for human inspection; ordering of the
    # trajectory is defined by list position (and, for step events, the step number),
    # never by the wall-clock timestamp, so clock skew between workers cannot reorder it.
    return datetime.datetime.now().isoformat(timespec="microseconds")


# ---------------------------------------------------------------------------
# Public read model
# ---------------------------------------------------------------------------
class TrialTrajectory:
    """Ordered trajectory of a single trial.

    The trajectory reconstructs everything that happened to a trial: every reported
    intermediate value, the moment the pruner decided to prune it, and the final state
    transition (including a failure reason when the objective raised).

    Instances are read-only views returned by
    :func:`~optuna.study.Study.get_trial_trajectory` and the module-level helpers below.

    Attributes:
        trial_number:
            Zero-based number of the trial within its study.
        trial_id:
            Internal storage id of the trial.
        events:
            Ordered list of event dictionaries. Each event always contains the keys
            ``kind`` and ``timestamp``. ``step`` events additionally carry ``step`` and
            ``value``; ``prune_decision`` events carry ``step`` (the step at which the
            pruner fired, when known); ``state`` events carry ``state`` (the terminal
            :class:`~optuna.trial.TrialState` name); ``fail`` events additionally carry
            ``error_type`` and ``error_message``.
        pruned_at_step:
            Step at which the pruner first returned :obj:`True`, or :obj:`None` if the
            trial was never pruned. Note that a ``prune_decision`` is recorded the moment
            :func:`~optuna.trial.Trial.should_prune` returns :obj:`True`, which is before
            the trial state is persisted as :class:`~optuna.trial.TrialState.PRUNED`.
        final_state:
            Name of the terminal state, or :obj:`None` while the trial is still running.
        failure_reason:
            ``(error type name, message)`` tuple for failed trials, :obj:`None` otherwise.
    """

    def __init__(
        self,
        trial_number: int,
        trial_id: int,
        events: list[dict[str, Any]],
        pruned_at_step: int | None,
        final_state: str | None,
        failure_reason: tuple[str, str] | None,
    ) -> None:
        self.trial_number = trial_number
        self.trial_id = trial_id
        self.events = events
        self.pruned_at_step = pruned_at_step
        self.final_state = final_state
        self.failure_reason = failure_reason

    # -- convenience projections -------------------------------------------
    @property
    def steps(self) -> list[dict[str, Any]]:
        """The ``step`` events in order, each ``{"step": int, "value": float}``."""
        return [
            {"step": e["step"], "value": e["value"]}
            for e in self.events
            if e["kind"] == _EVENT_STEP
        ]

    @property
    def intermediate_values(self) -> dict[int, float]:
        """Map of step -> value, reconstructed from the ordered step events."""
        return {e["step"]: e["value"] for e in self.events if e["kind"] == _EVENT_STEP}

    @property
    def is_pruned(self) -> bool:
        return self.final_state == TrialState.PRUNED.name

    @property
    def is_failed(self) -> bool:
        return self.final_state == TrialState.FAIL.name

    def to_dict(self) -> dict[str, Any]:
        """Return a plain JSON-serialisable representation (used by the exporter)."""
        return {
            "trial_number": self.trial_number,
            "trial_id": self.trial_id,
            "final_state": self.final_state,
            "pruned_at_step": self.pruned_at_step,
            "failure_reason": list(self.failure_reason) if self.failure_reason else None,
            "events": list(self.events),
        }

    def __repr__(self) -> str:
        return (
            f"TrialTrajectory(number={self.trial_number}, final_state={self.final_state!r}, "
            f"n_steps={len(self.steps)}, pruned_at_step={self.pruned_at_step})"
        )


# ---------------------------------------------------------------------------
# Recorder: one live instance per running trial
# ---------------------------------------------------------------------------
class TrajectoryRecorder:
    """Accumulates and persists the trajectory events of one running trial.

    The recorder holds the authoritative in-memory event log (the source of truth for the
    duplicate-step and post-prune guards) and mirrors it into the trial's system
    attribute so that it survives process restarts and is readable from any worker.

    Durability: when the storage is temporarily unavailable, events are buffered in
    memory and, if the buffer is spilled or the process keeps running, retried on the next
    successful flush. If in-memory buffering itself is at risk (a flush error while a
    non-empty backlog exists), the backlog is additionally appended to a local spill file
    under the OS temp directory, guaranteeing no accepted event is silently dropped.
    """

    def __init__(self, storage: BaseStorage, trial_id: int, trial_number: int) -> None:
        self._storage = storage
        self._trial_id = trial_id
        self._trial_number = trial_number
        self._events: list[dict[str, Any]] = [{"kind": _EVENT_RUNNING, "timestamp": _now_iso()}]
        # Latch closed the instant the pruner says "prune" (before any state is
        # persisted). This closes the report-while-pruned window inside the process.
        self._prune_latched = False
        self._pruned_at_step: int | None = None
        self._final_state: str | None = None
        self._failure_reason: tuple[str, str] | None = None
        self._reported_steps: set[int] = set()
        self._closed = False
        self._spill_path = os.path.join(
            tempfile.gettempdir(), f"optuna_trajectory_trial_{trial_id}.ndjson"
        )
        # Persist the initial RUNNING event immediately.
        self._flush()

    # -- guards -------------------------------------------------------------
    def _ensure_open(self) -> None:
        if self._closed:
            state = self._final_state or "FINISHED"
            raise UpdateFinishedTrialError(
                f"Trial#{self._trial_number} has already finished "
                f"(state={state}) and cannot accept new intermediate values."
            )
        if self._prune_latched:
            raise UpdateFinishedTrialError(
                f"Trial#{self._trial_number} was pruned at step {self._pruned_at_step}; "
                "intermediate values reported after the pruning decision are rejected."
            )

    # -- recording ----------------------------------------------------------
    def record_step(self, step: int, value: float) -> None:
        """Record a reported intermediate value.

        Raises:
            optuna.exceptions.UpdateFinishedTrialError:
                If the trial already finished or was pruned.
            optuna.exceptions.DuplicatedStepError:
                If ``step`` was already reported for this trial. Steps must be unique and
                are stored in reporting order; this is the deterministic, rejecting
                behaviour selected for trajectory mode.
        """
        self._ensure_open()
        if step in self._reported_steps:
            raise DuplicatedStepError(
                f"Trial#{self._trial_number} already has a reported value for {step=}. "
                "Trajectory mode rejects duplicate steps instead of overwriting/ignoring "
                "them. Pick a distinct, monotonically increasing step."
            )
        event = {
            "kind": _EVENT_STEP,
            "step": step,
            "value": float(value),
            "timestamp": _now_iso(),
        }
        self._reported_steps.add(step)
        self._append(event)

    def record_prune_decision(self, step: int | None) -> None:
        """Latch the pruning decision and record the prune moment."""
        if self._prune_latched or self._closed:
            return
        self._prune_latched = True
        self._pruned_at_step = step
        # Critical event: flush immediately so the prune moment is durable even if the
        # process dies before the next batch boundary / finalize.
        self._append(
            {
                "kind": _EVENT_PRUNE_DECISION,
                "step": step,
                "timestamp": _now_iso(),
            },
            force_flush=True,
        )

    def record_failure(self, error_type: str, message: str) -> None:
        if self._closed or self._failure_reason is not None:
            # Idempotent: the optimize loop and the tell() path can both observe a failure.
            return
        self._failure_reason = (error_type, message)
        self._append(
            {
                "kind": _EVENT_FAIL,
                "error_type": error_type,
                "error_message": message,
                "timestamp": _now_iso(),
            },
            force_flush=True,
        )

    def finalize(self, state: TrialState) -> None:
        """Record the terminal state transition and close the trajectory."""
        if self._closed:
            return
        self._final_state = state.name
        self._append(
            {
                "kind": _EVENT_STATE,
                "state": state.name,
                "timestamp": _now_iso(),
            },
            force_flush=True,
        )
        self._closed = True

    # -- persistence --------------------------------------------------------
    def _append(self, event: dict[str, Any], force_flush: bool = False) -> None:
        self._events.append(event)
        if force_flush or len(self._events) % _FLUSH_BATCH == 0:
            self._flush()
        # Otherwise the event stays in the authoritative in-memory log and is persisted
        # at the next batch boundary or critical event. On an ungraceful crash, at most
        # the last (< _FLUSH_BATCH) non-critical events rely on the spill file below.

    def _snapshot(self) -> dict[str, Any]:
        return {
            "version": 1,
            "trial_id": self._trial_id,
            "pruned_at_step": self._pruned_at_step,
            "final_state": self._final_state,
            "failure_reason": list(self._failure_reason) if self._failure_reason else None,
            "events": self._events,
        }

    def _flush(self) -> None:
        payload = self._snapshot()
        try:
            self._storage.set_trial_system_attr(self._trial_id, _TRAJ_SYSTEM_ATTR_KEY, payload)
        except Exception:
            # Storage unavailable: never drop. Spill the full snapshot locally so it
            # survives even an abrupt process exit, then keep the event in memory and
            # retry on the next flush.
            self._spill(payload)
            # Re-raise only at finalize so a transient storage outage mid-trial does not
            # kill the objective; in-memory + spill buffering carries the data.
            if self._closed:
                raise
        else:
            # The durable storage now reflects the full snapshot, so any earlier spill
            # file for this trial is redundant and can be removed.
            self._discard_spill()

    def _spill(self, payload: dict[str, Any]) -> None:
        try:
            with open(self._spill_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, sort_keys=True) + "\n")
        except OSError:
            # Last resort: the event remains in the in-memory log. We cannot do more
            # without a writable filesystem; surface this in the message of the next
            # finalize() attempt rather than silently pretending durability.
            pass

    def _discard_spill(self) -> None:
        try:
            os.remove(self._spill_path)
        except FileNotFoundError:
            pass
        except OSError:
            # A stale spill file is harmless (it is a strict prefix of the durable log);
            # leaving it behind is preferable to failing the trial.
            pass

    # -- read model ---------------------------------------------------------
    def to_trajectory(self) -> TrialTrajectory:
        return TrialTrajectory(
            trial_number=self._trial_number,
            trial_id=self._trial_id,
            events=list(self._events),
            pruned_at_step=self._pruned_at_step,
            final_state=self._final_state,
            failure_reason=self._failure_reason,
        )


# ---------------------------------------------------------------------------
# Loading / validation / querying (operate on persisted FrozenTrial data)
# ---------------------------------------------------------------------------
def _raw_payload(frozen_trial: FrozenTrial) -> dict[str, Any] | None:
    return frozen_trial.system_attrs.get(_TRAJ_SYSTEM_ATTR_KEY)


def has_trajectory(frozen_trial: FrozenTrial) -> bool:
    """Return whether a persisted trajectory exists for the given frozen trial."""
    return _raw_payload(frozen_trial) is not None


def load_trajectory(frozen_trial: FrozenTrial, *, validate: bool = True) -> TrialTrajectory:
    """Reconstruct a :class:`TrialTrajectory` from a persisted frozen trial.

    Args:
        frozen_trial:
            A frozen trial returned by the storage / study query APIs.
        validate:
            When :obj:`True` (default), raise
            :class:`~optuna.exceptions.TrajectoryCorruptedError` on any structural problem.

    Raises:
        optuna.exceptions.TrajectoryCorruptedError:
            If the persisted payload is malformed. The error message identifies the trial
            number and, where applicable, the offending step.
        KeyError:
            If the trial has no trajectory. Use :func:`has_trajectory` to check first.
    """
    payload = _raw_payload(frozen_trial)
    if payload is None:
        raise KeyError(
            f"Trial#{frozen_trial.number} has no recorded trajectory "
            "(enable_trajectory was off, or the trial predates the feature)."
        )
    events = payload.get("events")
    if validate:
        validate_trajectory(frozen_trial)
    if events is None:
        events = []
    return TrialTrajectory(
        trial_number=frozen_trial.number,
        trial_id=frozen_trial._trial_id,
        events=list(events),
        pruned_at_step=payload.get("pruned_at_step"),
        final_state=payload.get("final_state"),
        failure_reason=tuple(payload["failure_reason"]) if payload.get("failure_reason") else None,
    )


def validate_trajectory(frozen_trial: FrozenTrial) -> None:
    """Validate the structural integrity of one trial's persisted trajectory.

    Raises:
        optuna.exceptions.TrajectoryCorruptedError:
            With a message pinpointing the trial number and the offending step (when the
            problem is step-related) on the first problem found.
    """
    number = frozen_trial.number
    payload = _raw_payload(frozen_trial)
    if not isinstance(payload, dict):
        raise TrajectoryCorruptedError(
            f"Trial#{number}: trajectory payload is not a JSON object "
            f"(got {type(payload).__name__})."
        )
    events = payload.get("events")
    if not isinstance(events, list):
        raise TrajectoryCorruptedError(f"Trial#{number}: trajectory 'events' is not a list.")

    seen_steps: set[int] = set()
    terminal_seen = False
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            raise TrajectoryCorruptedError(
                f"Trial#{number}: event at index {index} is not an object."
            )
        kind = event.get("kind")
        if kind not in _VALID_EVENTS:
            raise TrajectoryCorruptedError(
                f"Trial#{number}: event at index {index} has unknown kind {kind!r}."
            )
        if kind in _TERMINAL_EVENTS:
            terminal_seen = True
        if kind == _EVENT_STEP:
            step = event.get("step")
            if not isinstance(step, int) or isinstance(step, bool):
                raise TrajectoryCorruptedError(
                    f"Trial#{number}: step event at index {index} has non-int step {step!r}."
                )
            if step in seen_steps:
                raise TrajectoryCorruptedError(
                    f"Trial#{number}: duplicate {step=} detected at event index {index}; "
                    "trajectory steps must be unique."
                )
            if not isinstance(event.get("value"), (int, float)) or isinstance(
                event.get("value"), bool
            ):
                raise TrajectoryCorruptedError(
                    f"Trial#{number}: step {step} has a non-numeric value {event.get('value')!r}."
                )
            seen_steps.add(step)
        if terminal_seen and kind not in _TERMINAL_EVENTS:
            # Any non-terminal event after the terminal marker means the ordering was
            # corrupted (e.g. a value accepted after the trial finished).
            raise TrajectoryCorruptedError(
                f"Trial#{number}: event of kind {kind!r} appears after the terminal "
                f"event at index {index}; no step/prune event may follow completion."
            )


def validate_study_trajectories(frozen_trials: list[FrozenTrial]) -> dict[int, str]:
    """Validate trajectories for every trial that has one.

    Returns:
        A mapping of ``trial_number -> error message`` for the corrupted trajectories.
        An empty mapping means all present trajectories are valid. Trials without a
        trajectory (feature was off) are skipped, not reported as errors.
    """
    problems: dict[int, str] = {}
    for trial in frozen_trials:
        if not has_trajectory(trial):
            continue
        try:
            validate_trajectory(trial)
        except TrajectoryCorruptedError as exc:
            problems[trial.number] = str(exc)
    return problems


def group_repeats_by_params(
    frozen_trials: list[FrozenTrial],
) -> dict[tuple[tuple[str, Any], ...], list[FrozenTrial]]:
    """Group trials that share the exact same parameter combination (repeat runs).

    Two trials belong to the same repeat group iff they suggested identical parameter
    names and values. Each member remains an independent trial and is counted separately
    in any statistics.

    Returns:
        Mapping from a hashable parameter signature (sorted ``(name, value)`` tuples) to
        the list of frozen trials in that group. Groups with a single member represent
        parameter combinations that were only tried once.
    """
    groups: dict[tuple[tuple[str, Any], ...], list[FrozenTrial]] = {}
    for trial in frozen_trials:
        signature = tuple(sorted(trial.params.items(), key=lambda kv: kv[0]))
        groups.setdefault(signature, []).append(trial)
    return groups


def trials_matching_params(
    frozen_trials: list[FrozenTrial], params: dict[str, Any]
) -> list[FrozenTrial]:
    """Return the historical trials whose parameters exactly match ``params``.

    This powers "look up the performance distribution of one parameter combination".
    Only the supplied parameter names are compared, but all of them must match.
    """
    return [
        trial
        for trial in frozen_trials
        if all(trial.params.get(name) == value for name, value in params.items())
        and all(name in trial.params for name in params)
    ]
