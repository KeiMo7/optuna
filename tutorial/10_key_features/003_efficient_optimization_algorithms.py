"""
.. _pruning:

Efficient Optimization Algorithms
=================================

Optuna enables efficient hyperparameter optimization by
adopting state-of-the-art algorithms for sampling hyperparameters and
pruning efficiently unpromising trials.

Sampling Algorithms
-------------------

Samplers basically continually narrow down the search space using the records of suggested parameter values and evaluated objective values,
leading to an optimal search space which giving off parameters leading to better objective values.
More detailed explanation of how samplers suggest parameters is in :class:`~optuna.samplers.BaseSampler`.

Optuna provides the following sampling algorithms:

- Grid Search implemented in :class:`~optuna.samplers.GridSampler`

- Exhaustive Search implemented in :class:`~optuna.samplers.BruteForceSampler`

- Random Search implemented in :class:`~optuna.samplers.RandomSampler`

- Tree-structured Parzen Estimator algorithm implemented in :class:`~optuna.samplers.TPESampler`

- CMA-ES based algorithm implemented in :class:`~optuna.samplers.CmaEsSampler`

- Gaussian process-based algorithm implemented in :class:`~optuna.samplers.GPSampler`

- Algorithm to enable partial fixed parameters implemented in :class:`~optuna.samplers.PartialFixedSampler`

- Nondominated Sorting Genetic Algorithm II implemented in :class:`~optuna.samplers.NSGAIISampler`

- A Quasi Monte Carlo sampling algorithm implemented in :class:`~optuna.samplers.QMCSampler`

The default sampler is :class:`~optuna.samplers.TPESampler`.

Switching Samplers
------------------

"""

import optuna


###################################################################################################
# By default, Optuna uses :class:`~optuna.samplers.TPESampler` as follows.

study = optuna.create_study()
print(f"Sampler is {study.sampler.__class__.__name__}")

###################################################################################################
# If you want to use different samplers for example :class:`~optuna.samplers.RandomSampler`
# and :class:`~optuna.samplers.CmaEsSampler`,

study = optuna.create_study(sampler=optuna.samplers.RandomSampler())
print(f"Sampler is {study.sampler.__class__.__name__}")

study = optuna.create_study(sampler=optuna.samplers.CmaEsSampler())
print(f"Sampler is {study.sampler.__class__.__name__}")


###################################################################################################
# Pruning Algorithms
# ------------------
#
# ``Pruners`` automatically stop unpromising trials at the early stages of the training (a.k.a., automated early-stopping).
# Currently :mod:`~optuna.pruners` module is expected to be used only for single-objective optimization.
#
# Optuna provides the following pruning algorithms:
#
# - Median pruning algorithm implemented in :class:`~optuna.pruners.MedianPruner`
#
# - Non-pruning algorithm implemented in :class:`~optuna.pruners.NopPruner`
#
# - Algorithm to operate pruner with tolerance implemented in :class:`~optuna.pruners.PatientPruner`
#
# - Algorithm to prune specified percentile of trials implemented in :class:`~optuna.pruners.PercentilePruner`
#
# - Asynchronous Successive Halving algorithm implemented in :class:`~optuna.pruners.SuccessiveHalvingPruner`
#
# - Hyperband algorithm implemented in :class:`~optuna.pruners.HyperbandPruner`
#
# - Threshold pruning algorithm implemented in :class:`~optuna.pruners.ThresholdPruner`
#
# - A pruning algorithm based on `Wilcoxon signed-rank test <https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test>`__ implemented in :class:`~optuna.pruners.WilcoxonPruner`
#
# We use :class:`~optuna.pruners.MedianPruner` in most examples,
# though basically it is outperformed by :class:`~optuna.pruners.SuccessiveHalvingPruner` and
# :class:`~optuna.pruners.HyperbandPruner` as in `this benchmark result <https://github.com/optuna/optuna/wiki/Benchmarks-with-Kurobako>`__.
#
#
# Activating Pruners
# ------------------
# To turn on the pruning feature, you need to call :func:`~optuna.trial.Trial.report` and :func:`~optuna.trial.Trial.should_prune` after each step of the iterative training.
# :func:`~optuna.trial.Trial.report` periodically monitors the intermediate objective values.
# :func:`~optuna.trial.Trial.should_prune` decides termination of the trial that does not meet a predefined condition.
#
# We would recommend using integration modules for major machine learning frameworks.
# Exclusive list is :mod:`~optuna.integration` and usecases are available in `optuna-examples <https://github.com/optuna/optuna-examples/>`__.


import logging
import sys

import sklearn.datasets
import sklearn.linear_model
import sklearn.model_selection


def objective(trial):
    iris = sklearn.datasets.load_iris()
    classes = list(set(iris.target))
    train_x, valid_x, train_y, valid_y = sklearn.model_selection.train_test_split(
        iris.data, iris.target, test_size=0.25, random_state=0
    )

    alpha = trial.suggest_float("alpha", 1e-5, 1e-1, log=True)
    clf = sklearn.linear_model.SGDClassifier(alpha=alpha)

    for step in range(100):
        clf.partial_fit(train_x, train_y, classes=classes)

        # Report intermediate objective value.
        intermediate_value = 1.0 - clf.score(valid_x, valid_y)
        trial.report(intermediate_value, step)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.TrialPruned()

    return 1.0 - clf.score(valid_x, valid_y)


###################################################################################################
# Set up the median stopping rule as the pruning condition.

# Add stream handler of stdout to show the messages
optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
study = optuna.create_study(pruner=optuna.pruners.MedianPruner())
study.optimize(objective, n_trials=20)

###################################################################################################
# As you can see, several trials were pruned (stopped) before they finished all of the iterations.
# The format of message is ``"Trial <Trial Number> pruned."``.

###################################################################################################
# Which Sampler and Pruner Should be Used?
# ----------------------------------------
#
# From the benchmark results which are available at `optuna/optuna - wiki "Benchmarks with Kurobako" <https://github.com/optuna/optuna/wiki/Benchmarks-with-Kurobako>`__, at least for not deep learning tasks, we would say that
#
# * For :class:`~optuna.samplers.RandomSampler`, :class:`~optuna.pruners.MedianPruner` is the best.
# * For :class:`~optuna.samplers.TPESampler`, :class:`~optuna.pruners.HyperbandPruner` is the best.
#
# However, note that the benchmark is not deep learning.
# For deep learning tasks,
# consult the below table.
# This table is from the `Ozaki et al., Hyperparameter Optimization Methods: Overview and Characteristics, in IEICE Trans, Vol.J103-D No.9 pp.615-631, 2020 <https://doi.org/10.14923/transinfj.2019JDR0003>`__ paper,
# which is written in Japanese.
#
# +---------------------------+-----------------------------------------+---------------------------------------------------------------+
# | Parallel Compute Resource | Categorical/Conditional Hyperparameters | Recommended Algorithms                                        |
# +===========================+=========================================+===============================================================+
# | Limited                   | No                                      | TPE. GP-EI if search space is low-dimensional and continuous. |
# +                           +-----------------------------------------+---------------------------------------------------------------+
# |                           | Yes                                     | TPE. GP-EI if search space is low-dimensional and continuous  |
# +---------------------------+-----------------------------------------+---------------------------------------------------------------+
# | Sufficient                | No                                      | CMA-ES, Random Search                                         |
# +                           +-----------------------------------------+---------------------------------------------------------------+
# |                           | Yes                                     | Random Search or Genetic Algorithm                            |
# +---------------------------+-----------------------------------------+---------------------------------------------------------------+
#

###################################################################################################
# Integration Modules for Pruning
# -------------------------------
# To implement pruning mechanism in much simpler forms, Optuna provides integration modules for the following libraries.
#
# For the complete list of Optuna's integration modules, see :mod:`~optuna.integration`.
#
# For example, `LightGBMPruningCallback <https://optuna-integration.readthedocs.io/en/stable/reference/generated/optuna_integration.LightGBMPruningCallback.html>`__ introduces pruning without directly changing the logic of training iteration.
# (See also `example <https://github.com/optuna/optuna-examples/blob/main/lightgbm/lightgbm_integration.py>`__ for the entire script.)
#
# .. code-block:: text
#
#         import optuna.integration
#
#         pruning_callback = optuna.integration.LightGBMPruningCallback(trial, 'validation-error')
#         gbm = lgb.train(param, dtrain, valid_sets=[dvalid], callbacks=[pruning_callback])

###################################################################################################
# Recording Pruning Decisions
# ----------------------------
# By default Optuna only stores each trial's final state, so *why* a trial was pruned — at which
# step, against which threshold, and compared to which historical trials — is not retained.
# :class:`~optuna.pruners.PruningRecorder` fills that gap. It is opt-in and does not change when
# pruning happens or which trials are pruned; when it is not attached, pruning behaves exactly as
# before.
#
# Attach it through :func:`~optuna.study.create_study` with the ``pruning_recorder`` argument:

recorder = optuna.pruners.PruningRecorder()
study = optuna.create_study(
    direction="minimize",
    pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1),
    pruning_recorder=recorder,
)


def recorded_objective(trial):
    x = trial.suggest_float("x", -10.0, 10.0)
    for step in range(6):
        trial.report((x - 2.0) ** 2 + 0.1 * step, step)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return (x - 2.0) ** 2


study.optimize(recorded_objective, n_trials=20)

###################################################################################################
# Every :meth:`~optuna.trial.Trial.should_prune` call now leaves a
# :class:`~optuna.pruners.PruningDecisionRecord`. The records for pruned trials name the decision
# step, the intermediate value, the rule and threshold applied, and the historical trials that
# participated in the comparison — referenced by their stable trial ``number``, never a list
# position.

if recorder.pruned_records():
    record = recorder.pruned_records()[0]
    print("pruned trial", record.trial_number, "at step", record.step)
    print("basis:", record.decision_basis)
    print("compared against trial numbers:", [r.number for r in record.referenced_trials])

###################################################################################################
# For each pruned trial the recorder also gives a counterfactual estimate of the final value had
# the trial been allowed to continue. The estimate is always labeled as such, together with its
# method and confidence level. When too few completed trials are available, no number is returned
# at all — an explicit reason is given instead, so unsupported guesses never appear as data.

if recorder.pruned_records():
    estimate = recorder.pruned_records()[0].estimate
    print("estimate method:", estimate.method)
    print("confidence level:", estimate.confidence_level)
    if estimate.is_estimate:
        print("prediction interval:", (round(estimate.lower, 3), round(estimate.upper, 3)))
    else:
        print("no interval:", estimate.insufficient_reason)

###################################################################################################
# Aggregate statistics are maintained incrementally, so they stay cheap even for very large
# studies. :meth:`~optuna.pruners.PruningRecorder.summary` groups decisions by pruner, returns the
# pruning-rate curve over search progress, and flags intervals where one pruner prunes an
# unusually long run of trials (a typical symptom of a misconfigured threshold).

summary = recorder.summary()
print("decisions by pruner:", summary["by_pruner"])
print("anomaly flags:", summary["anomalies"])

###################################################################################################
# To persist records beyond the process lifetime — records otherwise live in memory — pass a
# ``sink`` callable. A failing sink only emits a warning; it never stops the search.
#
# .. code-block:: python
#
#     def jsonl_sink(record):
#         with open("pruning_decisions.jsonl", "a") as f:
#             f.write(json.dumps(record.to_dict()) + "\n")
#
#     recorder = optuna.pruners.PruningRecorder(sink=jsonl_sink)
#
# Known limitations: the recorder captures only decisions made after it is attached; pruning
# decisions from pre-existing stored trials cannot be reconstructed because no evidence was
# persisted. Counterfactual intervals assume an approximately linear intermediate-to-final
# relationship within each step and, like the pruning feature itself, single-objective studies
# only.
