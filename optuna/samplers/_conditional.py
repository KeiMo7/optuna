"""Sampler that materializes a :class:`ConditionalSearchSpace` declaration."""

from __future__ import annotations

from typing import Any
from typing import TYPE_CHECKING

from optuna import logging
from optuna._warnings import optuna_warn
from optuna.exceptions import ConditionalConstraintViolation
from optuna.exceptions import InvalidConditionalSearchSpaceError
from optuna.samplers import BaseSampler
from optuna.samplers._lazy_random_state import LazyRandomState
from optuna.trial import TrialState


if TYPE_CHECKING:
    from collections.abc import Sequence

    from optuna.distributions import BaseDistribution
    from optuna.search_space import ConditionalSearchSpace
    from optuna.study import Study
    from optuna.trial import FrozenTrial


_logger = logging.get_logger(__name__)

_ATTR_COMBO_ID = "conditional_search_space:combo_id"
_ATTR_VERSION = "conditional_search_space:version"
_ATTR_REJECTION = "conditional_search_space:rejected_condition"


class ConditionalSampler(BaseSampler):
    """Sampler driven by a declarative :class:`~optuna.search_space.ConditionalSearchSpace`.

    The declaration is the single source of truth.  For every parameter requested by the
    objective, the sampler verifies that the parameter is declared, that the distribution
    passed to ``trial.suggest_*`` matches the declaration and that the parameter is active
    under the values already suggested in the trial.

    Two strategies are available:

    * ``strategy="enumerate"`` statically expands the declaration into legal combinations and
      evaluates them one by one.  This is the grid-like mode and requires a finite space.
    * ``strategy="delegate"`` asks an inner sampler for a value and projects it onto the
      declaration.  When the inner sampler does not support dynamic search spaces, the
      projection makes the resulting suggestions legal; a single degradation notice explains
      the substitution.  This mode never silently falls back to purely random suggestions.

    Args:
        search_space:
            The declaration that drives the sampler.
        strategy:
            Either ``"enumerate"`` or ``"delegate"``.  The default ``"enumerate"`` requires
            every parameter to be statically enumerable.
        fallback_sampler:
            The sampler used to propose values in ``"delegate"`` mode.  If :obj:`None`,
            :class:`~optuna.samplers.RandomSampler` is used.
        seed:
            Seed for the order in which enumerated combinations are visited and for repairs.
        warn_on_fallback:
            Whether to emit a degradation notice when ``"delegate"`` mode is used with an
            inner sampler that does not understand conditional spaces.
    """

    def __init__(
        self,
        search_space: ConditionalSearchSpace,
        *,
        strategy: str = "enumerate",
        fallback_sampler: BaseSampler | None = None,
        seed: int | None = None,
        warn_on_fallback: bool = True,
    ) -> None:
        if strategy not in ("enumerate", "delegate"):
            raise ValueError(
                f"`strategy` must be either 'enumerate' or 'delegate', but got {strategy!r}."
            )
        search_space.validate()
        self._search_space = search_space
        self._strategy = strategy
        self._rng = LazyRandomState(seed or 0)
        self._warn_on_fallback = warn_on_fallback
        self._n_rejected = 0
        self._rejections: dict[str, int] = {}

        if fallback_sampler is None:
            from optuna.samplers._random import RandomSampler

            fallback_sampler = RandomSampler(seed=seed)
        self._fallback_sampler = fallback_sampler
        self._fallback_reason: str | None = self._detect_fallback_reason()

        self._combinations: list[dict[str, Any]] | None = None
        if strategy == "enumerate":
            combinations = search_space.enumerate()
            self._combinations = combinations
            self._rng.rng.shuffle(self._combinations)  # type: ignore[arg-type]
            self._n_min_trials = len(combinations)
        else:
            self._n_min_trials = 0

    def _detect_fallback_reason(self) -> str | None:
        """Return why ``delegate`` mode degrades, or :obj:`None` when it aligns with the space.

        Samplers that cannot represent a conditional space internally make their own
        substitutions (for example, TPE with ``multivariate=True`` falls back to independent
        sampling). The declaration projects those suggestions back onto legal combinations, so
        the study keeps producing valid parameters, but the underlying proposal strategy is
        degraded. This method names that degradation explicitly rather than allowing a silent
        random fallback.
        """

        if self._strategy != "delegate":
            return None
        sampler_name = self._fallback_sampler.__class__.__name__
        supports_conditional = self._fallback_sampler._conditional_search_space() is not None
        if supports_conditional:
            return None
        return (
            f"`{sampler_name}` does not understand the declaration, so the declared conditions "
            "are enforced by projection after the proposal instead of during it; illegal "
            "proposals are replaced with the nearest declared-legal candidate."
        )

    @property
    def search_space(self) -> ConditionalSearchSpace:
        return self._search_space

    def _conditional_search_space(self) -> ConditionalSearchSpace:
        return self._search_space

    @property
    def n_rejected(self) -> int:
        """The number of suggestions rejected by the declaration so far."""

        return self._n_rejected

    @property
    def rejections(self) -> dict[str, int]:
        """Rejected suggestions keyed by declared condition expression."""

        return dict(self._rejections)

    def reseed_rng(self) -> None:
        self._rng.rng.seed()
        self._fallback_sampler.reseed_rng()

    def infer_relative_search_space(
        self, study: Study, trial: FrozenTrial
    ) -> dict[str, BaseDistribution]:
        # The declaration is evaluated parameter by parameter. Relative sampling is disabled so
        # that the declaration controls the active set instead of a multivariate search space.
        return {}

    def sample_relative(
        self, study: Study, trial: FrozenTrial, search_space: dict[str, BaseDistribution]
    ) -> dict[str, Any]:
        return {}

    def before_trial(self, study: Study, trial: FrozenTrial) -> None:
        if self._strategy != "enumerate":
            if self._fallback_reason is not None and self._warn_on_fallback:
                self._warn_on_fallback = False
                optuna_warn("Conditional sampling degraded: " + self._fallback_reason)
            return
        # Enqueued trials carry `fixed_params`; they must not consume a combination slot.
        if "fixed_params" in trial.system_attrs or _ATTR_COMBO_ID in trial.system_attrs:
            return
        assert self._combinations is not None

        if 0 <= trial.number < self._n_min_trials:
            combo_id = trial.number
        else:
            target = self._unvisited_combo_ids(study)
            if len(target) == 0:
                _logger.warning(
                    "`ConditionalSampler` is re-evaluating a configuration because every "
                    "declared combination has been exhausted."
                )
                target = list(range(self._n_min_trials))
            combo_id = int(self._rng.rng.choice(target))

        study._storage.set_trial_system_attr(trial._trial_id, _ATTR_COMBO_ID, combo_id)
        study._storage.set_trial_system_attr(
            trial._trial_id, _ATTR_VERSION, self._search_space.version
        )

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        space = self._search_space
        if param_name not in space.param_names:
            raise InvalidConditionalSearchSpaceError(
                f"The parameter `{param_name}` requested by the objective is not declared in "
                "the conditional search space."
            )

        declared_distribution = space.distribution(param_name)
        if param_distribution != declared_distribution:
            raise InvalidConditionalSearchSpaceError(
                f"The distribution passed for the parameter `{param_name}` does not match the "
                "declaration. The declaration is the single source of truth; update either the "
                "objective or the declaration."
            )

        if not space.is_active(param_name, trial.params):
            violation = ConditionalConstraintViolation(
                space.condition(param_name) or "<always>",
                param_name,
                "the parameter is not active under the values already suggested in this trial",
            )
            self._record_rejection(study, trial, violation)
            raise violation

        if self._strategy == "enumerate" and _ATTR_COMBO_ID in trial.system_attrs:
            assert self._combinations is not None
            combo_id = int(trial.system_attrs[_ATTR_COMBO_ID])
            value = self._combinations[combo_id][param_name]
            self._warn_if_out_of_range(param_name, value, param_distribution)
            return value

        return self._sample_and_project(study, trial, param_name, param_distribution)

    def _sample_and_project(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        value = self._fallback_sampler.sample_independent(
            study, trial, param_name, param_distribution
        )
        if self._is_legal(trial, param_name, value, allow_fixed_only=True):
            return value

        finite = self._finite_candidates(param_name)
        if finite is not None:
            self._rng.rng.shuffle(finite)  # type: ignore[arg-type]
            for candidate in finite:
                if self._is_legal(trial, param_name, candidate, allow_fixed_only=True):
                    return candidate
        else:
            for _ in range(self._max_retries):
                candidate = self._fallback_sampler.sample_independent(
                    study, trial, param_name, param_distribution
                )
                if self._is_legal(trial, param_name, candidate, allow_fixed_only=True):
                    return candidate

        env = dict(trial.params)
        env[param_name] = value
        violation = self._search_space.check_values(env, allow_fixed_only=True)
        assert violation is not None
        self._record_rejection(study, trial, violation.as_exception())
        raise violation.as_exception()

    _max_retries = 128

    def _is_legal(
        self,
        trial: FrozenTrial,
        param_name: str,
        value: Any,
        *,
        allow_fixed_only: bool,
    ) -> bool:
        env = dict(trial.params)
        env[param_name] = value
        return self._search_space.check_values(env, allow_fixed_only=allow_fixed_only) is None

    def _finite_candidates(self, param_name: str) -> list[Any] | None:
        try:
            return list(self._search_space.candidates(param_name))
        except InvalidConditionalSearchSpaceError:
            return None

    def _warn_if_out_of_range(
        self, param_name: str, value: Any, distribution: BaseDistribution
    ) -> None:
        contains = distribution._contains(distribution.to_internal_repr(value))
        if not contains:
            optuna_warn(
                f"The value for the parameter `{param_name}` is not contained in the declared "
                "distribution. The declared combination will be used anyway."
            )

    def _record_rejection(self, study: Study, trial: FrozenTrial, violation: Exception) -> None:
        self._n_rejected += 1
        condition = getattr(violation, "condition", str(violation))
        self._rejections[condition] = self._rejections.get(condition, 0) + 1
        # Tag the trial so the rejection is distinguishable from a normal objective failure in
        # the history. Only the condition expression is stored, never the rejected values.
        study._storage.set_trial_system_attr(trial._trial_id, _ATTR_REJECTION, condition)
        recorder = getattr(study, "_record_conditional_rejection", None)
        if recorder is not None:
            recorder(condition)

    def _unvisited_combo_ids(self, study: Study) -> list[int]:
        visited: set[int] = set()
        running: set[int] = set()
        for trial in study._storage.get_all_trials(study._study_id, deepcopy=False):
            combo_id = trial.system_attrs.get(_ATTR_COMBO_ID)
            if combo_id is None:
                continue
            if trial.system_attrs.get(_ATTR_VERSION) != self._search_space.version:
                continue
            if trial.state.is_finished():
                visited.add(int(combo_id))
            elif trial.state == TrialState.RUNNING:
                running.add(int(combo_id))
        unvisited = set(range(self._n_min_trials)) - visited - running
        if len(unvisited) == 0:
            unvisited = set(range(self._n_min_trials)) - visited
        return list(unvisited)

    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Sequence[float] | None,
    ) -> None:
        if self._strategy != "enumerate":
            return
        target = self._unvisited_combo_ids(study)
        if len(target) == 0:
            study.stop()
        elif len(target) == 1:
            combo_id = study._storage.get_trial_system_attrs(trial._trial_id).get(_ATTR_COMBO_ID)
            if combo_id == target[0]:
                study.stop()

    def is_exhausted(self, study: Study) -> bool:
        return self._strategy == "enumerate" and len(self._unvisited_combo_ids(study)) == 0


__all__ = ["ConditionalSampler"]
