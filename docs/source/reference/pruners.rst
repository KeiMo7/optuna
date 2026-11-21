.. module:: optuna.pruners

optuna.pruners
==============

The :mod:`~optuna.pruners` module defines a :class:`~optuna.pruners.BasePruner` class characterized by an abstract :meth:`~optuna.pruners.BasePruner.prune` method, which, for a given trial and its associated study, returns a boolean value representing whether the trial should be pruned. This determination is made based on stored intermediate values of the objective function, as previously reported for the trial using :meth:`optuna.trial.Trial.report`. The remaining classes in this module represent child classes, inheriting from :class:`~optuna.pruners.BasePruner`, which implement different pruning strategies.

.. warning::
    Currently :mod:`~optuna.pruners` module is expected to be used only for single-objective optimization.

.. seealso::
    :ref:`pruning` tutorial explains the concept of the pruner classes and a minimal example.

.. seealso::
    :ref:`user_defined_pruner` tutorial could be helpful if you want to implement your own pruner classes.

Recording pruning decisions
----------------------------

:class:`~optuna.pruners.PruningRecorder` makes pruning decisions auditable without changing
when pruning happens or what is pruned. It is **opt-in**: attach it through
:func:`~optuna.study.create_study` (or :func:`~optuna.study.load_study`) with
``pruning_recorder=PruningRecorder()``. When no recorder is attached, pruning behaves exactly
as before.

Every call to :meth:`~optuna.trial.Trial.should_prune` produces one
:class:`~optuna.pruners.PruningDecisionRecord` containing:

* the decision step, the intermediate value at that step, and the boolean verdict;
* the pruner class, its configuration, a human-readable decision basis and structured
  evidence (percentile thresholds, warmup/startup gates, rung cutoffs, ...);
* stable references (:class:`~optuna.pruners.TrialRef`, keyed by the trial's stable
  ``number`` — never a list position) to the historical trials in the comparison. Per-trial
  references are attached to pruning decisions; non-pruning decisions retain the quantitative
  basis and the comparison-set size instead;
* for pruned trials, a counterfactual :class:`~optuna.pruners.EstimateInterval` of the final
  value had the trial continued. It is always labeled as an estimate with its method and
  confidence level; when too few completed trials are available, no number is produced and an
  explicit ``insufficient_reason`` is given.

The recorder maintains statistics incrementally (per-pruner counts, a pruning-rate curve over
search progress, and anomaly flags for a pruner that prunes an unusually long run of trials),
so :meth:`~optuna.pruners.PruningRecorder.summary` does not scan the full trial history and
stays cheap for studies with 100k+ trials. A record ``sink`` callback can stream each record
elsewhere (e.g. to disk); sink failures are logged as warnings and never interrupt the search.

.. note::
    Known limitations: (1) decisions are kept in memory — attach a ``sink`` to persist them;
    (2) attaching a recorder to an already existing study only records future decisions,
    because earlier pruning decisions were never persisted and cannot be reconstructed;
    (3) counterfactual estimates assume the relationship between intermediate and final
    values is approximately linear within each step — treat them as labeled estimates, not
    observations; (4) only single-objective studies are supported, matching the existing
    pruning machinery.

.. autosummary::
    :toctree: generated/
    :nosignatures:

    BasePruner
    MedianPruner
    NopPruner
    PatientPruner
    PercentilePruner
    PruningRecorder
    PruningDecisionRecord
    TrialRef
    EstimateInterval
    SuccessiveHalvingPruner
    HyperbandPruner
    ThresholdPruner
    WilcoxonPruner
