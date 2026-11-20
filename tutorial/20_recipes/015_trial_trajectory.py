"""
.. _trial_trajectory:

Trial Trajectories (Ordered Event Logs)
=======================================

This recipe explains the opt-in **trial trajectory** feature: an ordered, append-only
record of everything that happens to a trial -- every reported intermediate value, the
exact moment the pruner decides to prune it, and its terminal state (including a failure
reason).

Why a separate feature?
-----------------------

By default :func:`~optuna.trial.Trial.report` keeps only the latest mapping of
``step -> value`` and silently ignores a value reported at a step that was already
reported. There is also a window between the pruner returning :obj:`True` from
:func:`~optuna.trial.Trial.should_prune` and the trial state actually being persisted as
:class:`~optuna.trial.TrialState.PRUNED`; during that window the trial is still
``RUNNING`` in the storage.

When trajectories are enabled, Optuna closes that window and records a complete,
ordered event log that reconstructs the true history of each trial.
"""

###################################################################################################
# Enabling trajectories
# ----------------------
#
# Pass ``enable_trajectory=True`` when creating the study. The switch is persisted on the
# study, so every process that later :func:`~optuna.study.load_study`-ies the same storage
# observes the same setting, and a reloaded study does not need the flag again.

import optuna


def objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -3.0, 3.0)
    for step in range(5):
        trial.report((x - 0.5) ** 2 + 0.1 * step, step)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return (x - 0.5) ** 2


study = optuna.create_study(enable_trajectory=True)
print("trajectory enabled:", study.trajectory_enabled)
study.optimize(objective, n_trials=3)

###################################################################################################
# Reading a trajectory
# --------------------
#
# :func:`~optuna.study.Study.get_trial_trajectory` reads a single trial on demand (it does
# not load the whole study). Each :class:`~optuna.trial._trajectory.TrialTrajectory` exposes
# the ordered ``events``, convenience projections (``steps`` / ``intermediate_values``),
# the ``pruned_at_step``, the ``final_state`` and the ``failure_reason``.

trajectory = study.get_trial_trajectory(0)
print(trajectory)
print("event kinds:", [event["kind"] for event in trajectory.events])
print("steps:", trajectory.steps)

###################################################################################################
# Strict reporting semantics
# --------------------------
#
# With the feature on, :func:`~optuna.trial.Trial.report` becomes deterministic and strict:
#
# * Reporting at a **duplicate step** raises
#   :class:`~optuna.exceptions.DuplicatedStepError` instead of being silently ignored.
# * Reporting **after the pruning decision** or after the trial has finished raises
#   :class:`~optuna.exceptions.UpdateFinishedTrialError` with the current state in the
#   message. The rejected value never reaches the storage.

strict_study = optuna.create_study(enable_trajectory=True)
trial = strict_study.ask()
trial.report(0.1, 0)
try:
    trial.report(0.2, 0)  # same step -> explicit rejection
except optuna.exceptions.DuplicatedStepError as e:
    print("duplicate rejected:", e)
strict_study.tell(trial, 0.1)

###################################################################################################
# Failures are preserved
# ----------------------
#
# A trial whose objective raises is recorded as :class:`~optuna.trial.TrialState.FAIL` and
# the trajectory keeps the exception type and message rather than treating it as merely
# unfinished.


def failing_objective(trial: optuna.Trial) -> float:
    trial.suggest_float("x", 0.0, 1.0)
    trial.report(0.1, 0)
    raise RuntimeError("something went wrong")


fail_study = optuna.create_study(enable_trajectory=True)
optuna.logging.set_verbosity(optuna.logging.ERROR)  # silence the expected failure log
fail_study.optimize(failing_objective, n_trials=1, catch=(RuntimeError,))
optuna.logging.set_verbosity(optuna.logging.INFO)
failed_traj = fail_study.get_trial_trajectory(0)
print("final state:", failed_traj.final_state)
print("failure reason:", failed_traj.failure_reason)

###################################################################################################
# Repeat runs and parameter-combination history
# ---------------------------------------------
#
# Trials that suggested identical parameters are repeat runs of the same combination.
# :func:`~optuna.study.Study.get_repeat_groups` groups them while keeping each trial
# separate (they are counted individually), and
# :func:`~optuna.study.Study.get_trials_with_params` looks up the historical performance of
# one specific combination.

lookup = study.get_trials_with_params({"x": 0.0})  # exact-match history lookup
groups = study.get_repeat_groups()  # signature -> list of repeat FrozenTrials
print("repeat group count:", len(groups))

###################################################################################################
# Exporting and validating
# ------------------------
#
# :func:`~optuna.study.Study.export_trajectories` writes a self-contained JSON snapshot.
# :func:`~optuna.study.Study.validate_trajectories` checks every persisted trajectory and
# returns a mapping of ``trial_number -> error message`` that pinpoints the trial and, when
# the problem is step-related, the offending step.

path = "trajectory_export.json"
study.export_trajectories(path)
problems = study.validate_trajectories()
assert problems == {}  # empty mapping means all trajectories are structurally valid

###################################################################################################
# Known limitations and compatibility
# -----------------------------------
#
# * The feature is **opt-in and off by default**. With it off, behavior is identical to
#   older Optuna, including the warn-and-ignore handling of duplicate steps. Existing
#   sampler and pruner interfaces are unchanged.
# * Trajectories are stored as a single trial system attribute, which every supported
#   backend (in-memory, RDB/SQLite, Journal and the gRPC proxy) already provides, so no
#   schema migration is needed and the behavior is consistent across backends.
# * Trials produced by an older version, or while the switch was off, simply have no
#   trajectory. Reading one raises :class:`KeyError`; use
#   :func:`~optuna.trial._trajectory.has_trajectory` or
#   :func:`~optuna.study.Study.get_all_trajectories` (which skips them) instead. Old data
#   does not need conversion and continues to work; it only lacks the new event log.
# * If the storage is temporarily unavailable, accepted events are buffered in memory and
#   mirrored to a local spill file under the OS temp directory, then re-written on the next
#   successful flush, so no accepted event is silently dropped.
