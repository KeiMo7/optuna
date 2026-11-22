"""A TPE sampler that keeps learning while the search space evolves.

The stock :class:`~optuna.samplers.TPESampler` builds an *intersection* search
space: a parameter must appear with one identical distribution in every
finished trial, and in the multivariate path a trial missing any current
dimension is dropped wholesale. Adding, removing or re-ranging a parameter
therefore discards the information accumulated in earlier rounds.

``EvolvingTPESampler`` instead works per dimension:

* a trial missing a dimension is simply *unknown* for that dimension - it is
  never treated as zero and never dropped from the other dimensions;
* historical values outside the current bounds are kept (the stored trial is
  untouched) but excluded from the kernel for that dimension and flagged as
  ``out_of_range`` in the round snapshot and the exported table;
* each parameter is modelled independently with the same Parzen estimator and
  acquisition rule as univariate TPE, so stored values keep biasing the
  suggestions instead of restarting from random search.

Round snapshots are written in :meth:`after_trial`, from the union of
distributions the finished trial actually used, so conditional parameters
(which arrive through :meth:`sample_independent`) end up in the same round as
the trial that enabled them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from optuna.distributions import BaseDistribution
from optuna.evolving_space._evolution import categorize_history
from optuna.evolving_space._evolution import EvolutionReport
from optuna.evolving_space._evolution import is_out_of_range
from optuna.evolving_space._evolution import validate_evolution
from optuna.evolving_space._snapshot import append_round
from optuna.evolving_space._snapshot import consume_new_round_marker
from optuna.evolving_space._snapshot import get_space_at_round
from optuna.evolving_space._snapshot import is_first_round_open
from optuna.evolving_space._snapshot import latest_space
from optuna.evolving_space._snapshot import load_snapshots
from optuna.evolving_space._snapshot import set_first_round_open
from optuna.evolving_space._snapshot import update_latest_round
from optuna.evolving_space._snapshot import write_base_space
from optuna.samplers._base import BaseSampler
from optuna.samplers._lazy_random_state import LazyRandomState
from optuna.samplers._random import RandomSampler
from optuna.samplers._tpe.parzen_estimator import _ParzenEstimator
from optuna.samplers._tpe.parzen_estimator import _ParzenEstimatorParameters
from optuna.samplers._tpe.sampler import _split_trials
from optuna.samplers._tpe.sampler import default_gamma
from optuna.samplers._tpe.sampler import default_gamma_multiobjective
from optuna.samplers._tpe.sampler import default_weights
from optuna.samplers._tpe.sampler import TPESampler
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


if TYPE_CHECKING:
    from collections.abc import Sequence

    from optuna.study import Study


class EvolvingTPESampler(BaseSampler):
    """TPE sampler supporting search spaces that change between rounds.

    Args:
        n_startup_trials:
            Number of trials per parameter for which that parameter is sampled
            uniformly. The counter is per parameter, so a freshly added
            parameter gets its own startup period while older parameters keep
            using their history.
        n_ei_candidates:
            Number of candidate samples drawn to evaluate the expected
            improvement, mirroring :class:`~optuna.samplers.TPESampler`.
        prior_weight:
            Prior weight of the Parzen estimator.
        consider_magic_clip:
            Whether to enable magic clip in the numerical kernels.
        consider_endpoints:
            Whether the numerical kernels take the endpoints into account.
        seed:
            Random seed.
    """

    def __init__(
        self,
        n_startup_trials: int = 10,
        n_ei_candidates: int = 24,
        prior_weight: float = 1.0,
        consider_magic_clip: bool = True,
        consider_endpoints: bool = False,
        seed: int | None = None,
    ) -> None:
        self._n_startup_trials = n_startup_trials
        self._n_ei_candidates = n_ei_candidates
        self._parzen_estimator_parameters = _ParzenEstimatorParameters(
            prior_weight=prior_weight,
            consider_magic_clip=consider_magic_clip,
            consider_endpoints=consider_endpoints,
            weights=default_weights,
            multivariate=False,
        )
        self._rng = LazyRandomState(seed)
        self._random_sampler = RandomSampler(seed=seed)

    def reseed_rng(self) -> None:
        self._rng.rng.seed()
        self._random_sampler.reseed_rng()

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def infer_relative_search_space(
        self, study: Study, trial: FrozenTrial
    ) -> dict[str, BaseDistribution]:
        # The full current space is declared by the objective itself; the
        # sampler does not shrink it to the intersection.
        return {}

    def sample_relative(
        self,
        study: Study,
        trial: FrozenTrial,
        search_space: dict[str, BaseDistribution],
    ) -> dict[str, object]:
        if not search_space:
            return {}
        for name, dist in search_space.items():
            self._validate_param(study, name, dist)
        return {
            name: self._sample_one(study, trial, name, dist) for name, dist in search_space.items()
        }

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> object:
        # Conditional parameters get the same history-aware treatment.
        self._validate_param(study, param_name, param_distribution)
        return self._sample_one(study, trial, param_name, param_distribution)

    @staticmethod
    def _validate_param(study: Study, param_name: str, distribution: BaseDistribution) -> None:
        """Raise the readable evolution error before storage sees the param.

        The storage layer only knows the generic distribution-kind rule; the
        sampler owns the round history and can name the offending parameter
        with its old and new distributions.
        """

        previous_space = latest_space(study)
        if previous_space is None or param_name not in previous_space:
            return
        old = previous_space[param_name]
        if old == distribution:
            return
        # Reuse the per-parameter rules without the whole-space "no overlap"
        # guard, which is meaningless for a single suggest call.
        validate_evolution({param_name: old}, {param_name: distribution})

    def _eligible_trials(
        self,
        study: Study,
        param_name: str,
        distribution: BaseDistribution,
    ) -> list[FrozenTrial]:
        """Trials that carry a *usable* value for this dimension.

        Missing-dimension trials are excluded for this dimension only (they
        remain eligible for every dimension they do have). Out-of-range values
        are likewise excluded here but preserved in storage and marked in the
        snapshot - never silently clipped.
        """

        states = (TrialState.COMPLETE, TrialState.PRUNED)
        trials = study._get_trials(deepcopy=False, states=states, use_cache=True)
        eligible: list[FrozenTrial] = []
        for t in trials:
            if param_name not in t.params:
                continue  # unknown for this dimension: not zero, not a dropped trial
            if is_out_of_range(t.params[param_name], distribution):
                continue  # kept in storage, flagged out_of_range, excluded from kernel
            eligible.append(t)
        return eligible

    def _sample_one(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        distribution: BaseDistribution,
    ) -> object:
        trials = self._eligible_trials(study, param_name, distribution)
        if len(trials) < self._n_startup_trials:
            return self._random_sampler.sample_independent(study, trial, param_name, distribution)

        gamma = default_gamma if len(study.directions) <= 1 else default_gamma_multiobjective
        below_trials, above_trials = _split_trials(study, trials, gamma(len(trials)))
        one_dim_space = {param_name: distribution}
        mpe_below = _ParzenEstimator(
            observations=self._internal_repr(below_trials, one_dim_space),
            search_space=one_dim_space,
            parameters=self._parzen_estimator_parameters,
        )
        mpe_above = _ParzenEstimator(
            observations=self._internal_repr(above_trials, one_dim_space),
            search_space=one_dim_space,
            parameters=self._parzen_estimator_parameters,
        )
        samples = mpe_below.sample(self._rng.rng, self._n_ei_candidates)
        acq = mpe_below.log_pdf(samples) - mpe_above.log_pdf(samples)
        internal = TPESampler._compare(samples, acq)
        return distribution.to_external_repr(internal[param_name])

    @staticmethod
    def _internal_repr(
        trials: Sequence[FrozenTrial],
        search_space: dict[str, BaseDistribution],
    ) -> dict[str, np.ndarray]:
        values: dict[str, list[float]] = {name: [] for name in search_space}
        for t in trials:
            for name, dist in search_space.items():
                if name in t.params:
                    values[name].append(dist.to_internal_repr(t.params[name]))
        return {name: np.asarray(v) for name, v in values.items()}

    def before_trial(self, study: Study, trial: FrozenTrial) -> None:
        self._random_sampler.before_trial(study, trial)

    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Sequence[float] | None,
    ) -> None:
        if state == TrialState.FAIL:
            # A failed trial never observed its full space; do not let it
            # create a round or a snapshot.
            pass
        elif state in (TrialState.COMPLETE, TrialState.PRUNED):
            self._record_round(study, trial)
        self._random_sampler.after_trial(study, trial, state, values)

    # ------------------------------------------------------------------
    # Round bookkeeping, driven by finished trials
    # ------------------------------------------------------------------
    def _record_round(self, study: Study, trial: FrozenTrial) -> EvolutionReport:
        """Validate the trial's space and keep the round snapshots current.

        The first round stays *open* until the first genuine evolution, so the
        conditional parameters of a branching objective are all absorbed into
        round 0 regardless of which branch the early trials take. Once a
        parameter present in every prior trial changes or disappears, round 0
        closes and later conditional branches are recognized via historical
        omission patterns.
        """

        trial_space = dict(trial.distributions)
        previous_space = latest_space(study)
        historical = study._get_trials(
            deepcopy=False, states=(TrialState.COMPLETE, TrialState.PRUNED)
        )

        if previous_space is None:
            report = validate_evolution(None, trial_space)
            write_base_space(study, trial_space)
            set_first_round_open(study, True)
            append_round(
                study,
                trial_space,
                first_trial_number=trial.number,
                out_of_range_markers=categorize_history(historical, trial_space, report),
                note=report.summary(),
            )
            return report

        if trial_space == previous_space:
            return validate_evolution(previous_space, trial_space)

        forced_new_round = consume_new_round_marker(study)
        if (
            is_first_round_open(study)
            and not forced_new_round
            and self._extends_first_round(study, previous_space, trial_space)
        ):
            # Still discovering the first round's conditional branches.
            merged_space = {**previous_space, **trial_space}
            write_base_space(study, merged_space)
            report = validate_evolution(previous_space, merged_space)
            update_latest_round(
                study,
                merged_space,
                out_of_range_markers=categorize_history(historical, merged_space, report),
                note=report.summary(),
            )
            return report

        # Either the first round already closed, or this is the first genuine
        # evolution. A trial that merely omits a historically-conditional
        # parameter still belongs to the current round.
        if self._is_conditional_branch(study, previous_space, trial_space):
            report = validate_evolution(previous_space, previous_space)
            update_latest_round(
                study,
                previous_space,
                out_of_range_markers=categorize_history(historical, previous_space, report),
                note=report.summary(),
            )
            return report

        # Validate (raising the readable conflict error first), close round 0
        # and open a new round.
        report = validate_evolution(previous_space, trial_space)
        set_first_round_open(study, False)
        append_round(
            study,
            trial_space,
            first_trial_number=trial.number,
            out_of_range_markers=categorize_history(historical, trial_space, report),
            note=report.summary(),
        )
        return report

    @staticmethod
    def _extends_first_round(
        study: Study,
        previous_space: dict[str, BaseDistribution],
        trial_space: dict[str, BaseDistribution],
    ) -> bool:
        """Whether a differing trial still fits the open first round.

        Only additions of new parameters and omissions of parameters that are
        already known to be conditional qualify. Re-ranging or re-typing an
        existing parameter always ends the first round. A new parameter name
        counts as conditional only when the round already shows branching
        structure (some earlier trial omitted a parameter); otherwise the
        objective simply grew its search space and a new round must open.
        """

        for name, dist in trial_space.items():
            known = previous_space.get(name)
            if known is not None and known != dist:
                return False
        new_names = set(trial_space) - set(previous_space)
        omitted = set(previous_space) - set(trial_space)
        historical = study._get_trials(
            deepcopy=False,
            states=(TrialState.COMPLETE, TrialState.PRUNED),
            use_cache=True,
        )
        if new_names:
            # The signature of a conditional objective is structural variety
            # in history: earlier trials used more than one distinct parameter
            # set. When that variety exists, a new name is another arm of the
            # conditional structure and stays in the open first round. A single
            # uniform parameter set across all history means a new name is
            # genuine search-space growth that must open a new round.
            distinct_param_sets = {frozenset(t.params.keys()) for t in historical}
            return len(distinct_param_sets) >= 2
        if omitted:
            # The first omission of a parameter is the other side of a
            # conditional branch while the first round is open.
            return True
        return False

    @staticmethod
    def _is_conditional_branch(
        study: Study,
        previous_space: dict[str, BaseDistribution],
        trial_space: dict[str, BaseDistribution],
    ) -> bool:
        """Whether the trial takes a branch within an already known round."""

        if not trial_space.keys() <= previous_space.keys():
            return False
        if any(previous_space[name] != trial_space[name] for name in trial_space):
            return False
        omitted = set(previous_space) - set(trial_space)
        if not omitted:
            return True
        historical = study._get_trials(
            deepcopy=False,
            states=(TrialState.COMPLETE, TrialState.PRUNED),
            use_cache=True,
        )
        # Every omitted parameter must have been omitted by an earlier trial;
        # a parameter present in all history is being deleted, not branched on.
        return all(any(name not in t.params for t in historical) for name in omitted)

    # ------------------------------------------------------------------
    # Introspection helpers used by the export utility and tests
    # ------------------------------------------------------------------
    def get_out_of_range_markers(self, study: Study) -> dict[int, dict[str, str]]:
        """Return ``{trial_number: {param: "missing" | "out_of_range"}}``.

        Recomputed from the raw trials against the newest recorded space, so it
        stays correct when a study is reopened from any backend.
        """

        space = latest_space(study)
        if space is None:
            return {}
        snapshots = load_snapshots(study)
        previous = None
        if len(snapshots) >= 2:
            previous = get_space_at_round(study, len(snapshots) - 2)
        report = validate_evolution(previous, space)
        return categorize_history(study._get_trials(deepcopy=False), space, report)
