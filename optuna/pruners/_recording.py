from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
import math
import threading
from typing import Any
from typing import Callable
from typing import TYPE_CHECKING
from weakref import WeakKeyDictionary

import numpy as np

from optuna import logging
from optuna.pruners._hyperband import HyperbandPruner
from optuna.pruners._nop import NopPruner
from optuna.pruners._patient import PatientPruner
from optuna.pruners._percentile import PercentilePruner
from optuna.pruners._successive_halving import SuccessiveHalvingPruner
from optuna.pruners._threshold import ThresholdPruner
from optuna.pruners._wilcoxon import WilcoxonPruner
from optuna.study._study_direction import StudyDirection
from optuna.trial._state import TrialState


if TYPE_CHECKING:
    from optuna.pruners._base import BasePruner
    from optuna.study import Study
    from optuna.trial import FrozenTrial


_logger = logging.get_logger(__name__)


# ---------------------------------------------------------------------------
# Public, plain data containers (JSON-serializable; no storage objects)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrialRef:
    """Stable reference to a historical trial.

    A trial ``number`` is unique and consecutive within a study and never reused, so it is a
    stable identifier. List indices are deliberately *not* used because the contents of
    ``study.get_trials()`` change as new trials finish.

    Attributes:
        number:
            Stable trial number (``FrozenTrial.number``).
        trial_id:
            Storage-level trial id (``FrozenTrial._trial_id``), recorded for cross-checking.
        state:
            String representation of the trial state at decision time.
        intermediate_value:
            Intermediate value reported at the decision step by the referenced trial, if any.
    """

    number: int
    trial_id: int
    state: str
    intermediate_value: float | None = None


@dataclass(frozen=True)
class EstimateInterval:
    """Estimated final-value interval for a pruned trial, had it been allowed to continue.

    This is an *estimate*, not an observation. The label is mandatory by design: callers must
    never present the bounds as measured values.

    Attributes:
        lower:
            Lower bound of the prediction interval on the objective-value scale.
        upper:
            Upper bound of the prediction interval on the objective-value scale.
        point_estimate:
            Regression prediction at the trial's current intermediate value.
        confidence_level:
            Confidence level used for the prediction interval, e.g. 0.95.
        method:
            Human-readable name of the estimation method.
        method_description:
            Full description of the method and how the interval is derived.
        n_samples:
            Number of completed historical trials used as regression samples.
        insufficient_reason:
            Set when no interval could be produced. When not :obj:`None`, the numeric fields
            are :obj:`None`.
    """

    lower: float | None
    upper: float | None
    point_estimate: float | None
    confidence_level: float
    method: str
    method_description: str
    n_samples: int
    insufficient_reason: str | None = None

    @property
    def is_estimate(self) -> bool:
        """Return :obj:`True` if numeric estimate bounds are available."""

        return self.insufficient_reason is None


@dataclass
class PruningDecisionRecord:
    """One pruning decision made by :meth:`~optuna.trial.Trial.should_prune`.

    The record is created at the single decision point
    (``BasePruner.prune`` called from ``Trial.should_prune``); the recorder wraps that call
    without changing when it happens or what it returns.

    Attributes:
        trial_number / trial_id:
            Stable identifiers of the trial being judged.
        step:
            The step (``FrozenTrial.last_step``) at which the decision was made.
        intermediate_value:
            The trial's intermediate value at ``step``.
        pruner_class:
            Class name of the pruner that made the decision.
        pruner_params:
            Public configuration of the pruner (its ``__init__`` parameters).
        pruned:
            The boolean returned by the pruner. Equivalent to what
            :meth:`~optuna.trial.Trial.should_prune` returned.
        decision_basis:
            Human-readable explanation of the rule that produced ``pruned``.
        evidence:
            Structured evidence used to build ``decision_basis`` (threshold values, statistics,
            warmup/startup gate results, ...).
        referenced_trials:
            Stable references to the historical trials participating in the comparison.
        estimate:
            Counterfactual final-value estimate, populated for pruned decisions; :obj:`None`
            when the decision was not to prune.
        decision_index:
            Monotonic, zero-based index of the decision within the recorder.
        trial_state_at_decision:
            Trial state observed when the decision was recorded (normally ``RUNNING``).
    """

    trial_number: int
    trial_id: int
    step: int
    intermediate_value: float
    pruner_class: str
    pruner_params: dict[str, Any]
    pruned: bool
    decision_basis: str
    evidence: dict[str, Any]
    referenced_trials: list[TrialRef]
    estimate: EstimateInterval | None
    decision_index: int
    trial_state_at_decision: str = "RUNNING"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary representation."""

        d = asdict(self)
        return d


# ---------------------------------------------------------------------------
# Incremental sufficient statistics for per-step linear regression
# ---------------------------------------------------------------------------


@dataclass
class _StepStats:
    """Sufficient statistics for OLS regression ``final_value ~ intermediate_value`` at a step.

    Stored quantities are sums over completed trials that reported the step, so updating with
    a newly completed trial is O(1) and producing an estimate never scans the trial history.
    """

    n: int = 0
    sum_x: float = 0.0
    sum_y: float = 0.0
    sum_xx: float = 0.0
    sum_xy: float = 0.0
    sum_yy: float = 0.0
    # Per-trial points, retained only for the residual-based prediction-interval correction.
    # Bounded by the number of completed trials reporting the step.
    points: list[tuple[float, float]] = field(default_factory=list)

    def update(self, x: float, y: float) -> None:
        if math.isnan(x) or math.isnan(y):
            return
        self.n += 1
        self.sum_x += x
        self.sum_y += y
        self.sum_xx += x * x
        self.sum_xy += x * y
        self.sum_yy += y * y
        self.points.append((x, y))


# Two-sided t critical values for common confidence levels, df 1..30.
# For larger df the standard-normal approximation is used (converges quickly).
_T_TABLE: dict[float, list[float]] = {
    0.90: [
        6.314,
        2.920,
        2.353,
        2.132,
        2.015,
        1.943,
        1.895,
        1.860,
        1.833,
        1.812,
        1.796,
        1.782,
        1.771,
        1.761,
        1.753,
        1.746,
        1.740,
        1.734,
        1.729,
        1.725,
        1.721,
        1.717,
        1.714,
        1.711,
        1.708,
        1.706,
        1.703,
        1.701,
        1.699,
        1.697,
    ],
    0.95: [
        12.706,
        4.303,
        3.182,
        2.776,
        2.571,
        2.447,
        2.365,
        2.306,
        2.262,
        2.228,
        2.201,
        2.179,
        2.160,
        2.145,
        2.131,
        2.120,
        2.110,
        2.101,
        2.093,
        2.086,
        2.080,
        2.074,
        2.069,
        2.064,
        2.060,
        2.056,
        2.052,
        2.048,
        2.045,
        2.042,
    ],
    0.99: [
        63.657,
        9.925,
        5.841,
        4.604,
        4.032,
        3.707,
        3.499,
        3.355,
        3.250,
        3.169,
        3.106,
        3.055,
        3.012,
        2.977,
        2.947,
        2.921,
        2.898,
        2.878,
        2.861,
        2.845,
        2.831,
        2.819,
        2.807,
        2.797,
        2.787,
        2.779,
        2.771,
        2.763,
        2.756,
        2.750,
    ],
}

_METHOD_NAME = "OLS linear regression prediction interval"
_METHOD_DESCRIPTION = (
    "Ordinary least squares regression of completed trials' final objective values on their "
    "intermediate values at this step, fitted incrementally from completed-trial summaries. "
    "The interval is a two-sided prediction interval for a new observation (mean response plus "
    "residual standard error), using the Student t critical value at the stated confidence "
    "level; normal approximation is used when degrees of freedom exceed 30."
)


def _normal_ppf(p: float) -> float:
    """Inverse standard-normal CDF using the Beasley-Springer-Moro approximation (no scipy)."""

    # Acklam's inverse normal CDF coefficients.
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q) / (
            ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1
        )
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
        (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
    )


def _z_critical(confidence_level: float) -> float:
    return _normal_ppf(1 - (1 - confidence_level) / 2)


# ---------------------------------------------------------------------------
# Evidence builders: reconstruct each built-in pruner's rule from storage state.
# They never invoke the pruner itself, so pruning timing/results are untouched.
# ---------------------------------------------------------------------------


def _refs_from_trials(
    trials: list[FrozenTrial], step: int, *, only_with_value: bool = False
) -> list[TrialRef]:
    """Build stable trial references.

    By default every given trial is referenced. When ``only_with_value`` is set, only trials
    that actually report ``step`` are included — those are precisely the trials that enter
    the percentile/median comparison, so this keeps record size linear in the comparison
    set rather than in the whole study history.
    """

    refs: list[TrialRef] = []
    for t in trials:
        has_value = step in t.intermediate_values
        if only_with_value and not has_value:
            continue
        refs.append(
            TrialRef(
                number=t.number,
                trial_id=t._trial_id,
                state=t.state.name,
                intermediate_value=t.intermediate_values[step] if has_value else None,
            )
        )
    return refs


# Configuration snapshots are immutable after pruner construction, so cache one per pruner
# instance in a weak map instead of monkey-patching the pruner object.
_PARAMS_CACHE: WeakKeyDictionary[BasePruner, dict[str, Any]] = WeakKeyDictionary()


def _pruner_params(pruner: BasePruner) -> dict[str, Any]:
    """Best-effort snapshot of public pruner configuration without importing private attrs.

    Cached per pruner instance in a :class:`~weakref.WeakKeyDictionary`: the configuration is
    immutable after construction, so the snapshot is computed once per pruner instead of once
    per decision, and the entry disappears when the pruner is garbage collected.
    """

    cached = _PARAMS_CACHE.get(pruner)
    if cached is not None:
        return cached

    params: dict[str, Any] = {}
    for key, value in vars(pruner).items():
        if key.startswith("__"):
            continue
        if key.startswith("_"):
            public = key[1:]
        else:
            public = key
        # Hyperband internals include nested pruners; summarize recursively but shallowly.
        if hasattr(value, "prune"):
            params[public] = f"{type(value).__name__}({_pruner_params(value)})"
            continue
        try:
            import json

            json.dumps(value)
            params[public] = value
        except (TypeError, ValueError):
            params[public] = repr(value)
    _PARAMS_CACHE[pruner] = params
    return params


def _explain_percentile(
    pruner: PercentilePruner,
    study: Study,
    trial: FrozenTrial,
    step: int,
    include_refs: bool,
) -> tuple[str, dict[str, Any], list[TrialRef]]:
    # Use the study's thread-local cache: within one trial's lifetime the trial set is
    # stable, and the cache is invalidated on `tell`, so repeated should_prune() calls do
    # not re-query the storage. No copies are made (deepcopy=False).
    completed = study._get_trials(deepcopy=False, states=(TrialState.COMPLETE,), use_cache=True)
    n_trials = len(completed)
    direction = study.direction
    evidence: dict[str, Any] = {
        "rule": "percentile",
        "percentile": pruner._percentile,
        "n_startup_trials": pruner._n_startup_trials,
        "n_warmup_steps": pruner._n_warmup_steps,
        "interval_steps": pruner._interval_steps,
        "n_min_trials": pruner._n_min_trials,
        "n_completed_trials": n_trials,
    }

    if n_trials == 0:
        return "No completed trials yet; the pruner does not prune.", evidence, []
    if n_trials < pruner._n_startup_trials:
        evidence["gate"] = "n_startup_trials"
        return (
            f"Only {n_trials} completed trial(s), below n_startup_trials="
            f"{pruner._n_startup_trials}; not pruned.",
            evidence,
            [],
        )
    if step < pruner._n_warmup_steps:
        evidence["gate"] = "n_warmup_steps"
        return (
            f"Step {step} is within warmup (n_warmup_steps={pruner._n_warmup_steps}); not pruned.",
            evidence,
            [],
        )

    # Single pass: collect the comparison-set values. Stable per-trial references are built
    # only on pruning decisions; otherwise only the comparison-set size is retained.
    refs: list[TrialRef] = []
    values: list[float] = []
    for t in completed:
        iv = t.intermediate_values
        if step in iv:
            v = float(iv[step])
            values.append(v)
            if include_refs:
                refs.append(
                    TrialRef(
                        number=t.number,
                        trial_id=t._trial_id,
                        state=t.state.name,
                        intermediate_value=v,
                    )
                )
    arr = np.asarray(values, dtype=float)
    valid = arr[~np.isnan(arr)] if arr.size else arr
    evidence["n_values_at_step"] = int(valid.size)
    if valid.size < pruner._n_min_trials:
        evidence["gate"] = "n_min_trials"
        return (
            f"Only {valid.size} value(s) at step {step}, below n_min_trials="
            f"{pruner._n_min_trials}; not pruned.",
            evidence,
            refs,
        )

    pct = pruner._percentile
    if direction == StudyDirection.MAXIMIZE:
        pct = 100 - pct
    threshold = float(np.nanpercentile(arr, pct))
    evidence["threshold_percentile_value"] = threshold
    current = float(trial.intermediate_values[step])
    if math.isnan(current):
        return (
            "Current intermediate value is NaN; the pruner prunes all-NaN trials.",
            evidence,
            refs,
        )
    if direction == StudyDirection.MAXIMIZE:
        pruned = current < threshold
        relation = "below" if pruned else "at or above"
    else:
        pruned = current > threshold
        relation = "above" if pruned else "at or below"
    return (
        f"Current value {current} is {relation} the {pruner._percentile}-th percentile "
        f"threshold {threshold} over {valid.size} completed trial(s) at step {step}.",
        evidence,
        refs,
    )


def _explain_threshold(
    pruner: ThresholdPruner, trial: FrozenTrial, step: int
) -> tuple[str, dict[str, Any], list[TrialRef]]:
    evidence: dict[str, Any] = {
        "rule": "threshold",
        "lower": pruner._lower,
        "upper": pruner._upper,
        "n_warmup_steps": pruner._n_warmup_steps,
        "interval_steps": pruner._interval_steps,
    }
    if step < pruner._n_warmup_steps:
        evidence["gate"] = "n_warmup_steps"
        return (
            f"Step {step} is within warmup (n_warmup_steps={pruner._n_warmup_steps}); not pruned.",
            evidence,
            [],
        )
    value = float(trial.intermediate_values[step])
    evidence["current_value"] = value
    if math.isnan(value):
        return "Current value is NaN; the threshold pruner prunes NaN values.", evidence, []
    if value < pruner._lower:
        return (
            f"Value {value} is below the lower threshold {pruner._lower}; pruned.",
            evidence,
            [],
        )
    if value > pruner._upper:
        return (
            f"Value {value} exceeds the upper threshold {pruner._upper}; pruned.",
            evidence,
            [],
        )
    return (
        f"Value {value} is within [{pruner._lower}, {pruner._upper}]; not pruned.",
        evidence,
        [],
    )


def _explain_patient(
    pruner: PatientPruner,
    study: Study,
    trial: FrozenTrial,
    step: int,
    include_refs: bool,
) -> tuple[str, dict[str, Any], list[TrialRef]]:
    evidence: dict[str, Any] = {
        "rule": "patient",
        "patience": pruner._patience,
        "min_delta": pruner._min_delta,
        "wrapped_pruner": type(pruner._wrapped_pruner).__name__
        if pruner._wrapped_pruner is not None
        else None,
    }
    iv = trial.intermediate_values
    steps = sorted(iv.keys())
    if len(steps) <= pruner._patience + 1:
        evidence["gate"] = "insufficient_reported_steps"
        return (
            f"Only {len(steps)} reported step(s), need more than patience+1="
            f"{pruner._patience + 1}; not pruned.",
            evidence,
            [],
        )
    before = [iv[s] for s in steps[: -pruner._patience - 1]]
    after = [iv[s] for s in steps[-pruner._patience - 1 :]]
    direction = study.direction
    if direction == StudyDirection.MINIMIZE:
        best_before, best_after = float(np.nanmin(before)), float(np.nanmin(after))
        no_improvement = best_before + pruner._min_delta < best_after
    else:
        best_before, best_after = float(np.nanmax(before)), float(np.nanmax(after))
        no_improvement = best_before - pruner._min_delta > best_after
    evidence.update(
        {
            "best_before_patience": best_before,
            "best_after_patience": best_after,
            "no_improvement_detected": bool(no_improvement),
        }
    )
    if not no_improvement:
        return (
            f"Improvement over the last {pruner._patience} reported step(s) exceeds "
            f"min_delta={pruner._min_delta}; the patience gate keeps the trial alive.",
            evidence,
            [],
        )
    if pruner._wrapped_pruner is None:
        return (
            "No improvement within patience and no wrapped pruner configured; pruned.",
            evidence,
            [],
        )
    # Delegate the explanation to the wrapped pruner's rule.
    basis, sub_evidence, refs = _explain(pruner._wrapped_pruner, study, trial, step, include_refs)
    evidence["wrapped"] = sub_evidence
    return (
        f"Patience gate tripped ({best_before} -> {best_after}); wrapped "
        f"{type(pruner._wrapped_pruner).__name__} then decides. {basis}",
        evidence,
        refs,
    )


def _explain_successive_halving(
    pruner: SuccessiveHalvingPruner,
    study: Study,
    trial: FrozenTrial,
    step: int,
    include_refs: bool,
) -> tuple[str, dict[str, Any], list[TrialRef]]:
    evidence: dict[str, Any] = {
        "rule": "successive_halving",
        "min_resource": pruner._min_resource,
        "reduction_factor": pruner._reduction_factor,
        "min_early_stopping_rate": pruner._min_early_stopping_rate,
        "bootstrap_count": pruner._bootstrap_count,
    }
    trials = study._get_trials(deepcopy=False, use_cache=True)
    rung = 0
    rung_key = f"completed_rung_{rung}"
    while rung_key in trial.system_attrs:
        rung += 1
        rung_key = f"completed_rung_{rung}"
    evidence["rung"] = rung
    value = float(trial.intermediate_values[step])
    evidence["current_value"] = value
    competing = [
        t for t in trials if rung_key in t.system_attrs and t._trial_id != trial._trial_id
    ]
    competing_values = [float(t.system_attrs[rung_key]) for t in competing]
    evidence["n_competing_at_rung"] = len(competing_values)
    refs = _refs_from_trials(competing, step) if include_refs else []
    if math.isnan(value):
        return (
            "Current value is NaN; the successive halving pruner prunes NaN values.",
            evidence,
            refs,
        )
    if len(competing_values) <= pruner._bootstrap_count:
        return (
            f"Only {len(competing_values)} competitor(s) at rung {rung}, at or below "
            f"bootstrap_count={pruner._bootstrap_count}; pruned.",
            evidence,
            refs,
        )
    direction = study.direction
    if direction == StudyDirection.MINIMIZE:
        cutoff = float(np.nanmin(competing_values))
        promotable = value <= cutoff
    else:
        cutoff = float(np.nanmax(competing_values))
        promotable = value >= cutoff
    evidence["rung_promotion_cutoff"] = cutoff
    if promotable:
        return (
            f"Value {value} is promotable to the next rung (cutoff {cutoff}); not pruned.",
            evidence,
            refs,
        )
    return (
        f"Value {value} is not promotable to rung {rung + 1} (cutoff {cutoff} over "
        f"{len(competing_values)} competitor(s)); pruned.",
        evidence,
        refs,
    )


def _explain_hyperband(
    pruner: HyperbandPruner,
    study: Study,
    trial: FrozenTrial,
    step: int,
    include_refs: bool,
) -> tuple[str, dict[str, Any], list[TrialRef]]:
    bracket_id = pruner._get_bracket_id(study, trial)
    bracket_study = pruner._create_bracket_study(study, bracket_id)
    inner = pruner._pruners[bracket_id]
    evidence: dict[str, Any] = {
        "rule": "hyperband",
        "bracket_id": bracket_id,
        "inner_pruner": type(inner).__name__,
    }
    basis, sub_evidence, refs = _explain(inner, bracket_study, trial, step, include_refs)
    evidence["bracket"] = sub_evidence
    if include_refs:
        # Restrict refs to trials in the same bracket (the ones the inner pruner saw).
        bracket_numbers = {
            t.number for t in bracket_study._get_trials(deepcopy=False, use_cache=True)
        }
        refs = [r for r in refs if r.number in bracket_numbers]
    return f"[bracket {bracket_id}] {basis}", evidence, refs


def _explain_wilcoxon(
    pruner: WilcoxonPruner,
    study: Study,
    trial: FrozenTrial,
    step: int,
    include_refs: bool,
) -> tuple[str, dict[str, Any], list[TrialRef]]:
    evidence: dict[str, Any] = {
        "rule": "wilcoxon",
        "p_threshold": pruner._p_threshold,
    }
    # The Wilcoxon pruner compares against the current best trial; surface that reference.
    refs: list[TrialRef] = []
    try:
        best = study.best_trial
        if include_refs:
            refs = _refs_from_trials([best], step)
        evidence["reference_best_trial_number"] = best.number
    except ValueError:
        evidence["reference_best_trial_number"] = None
    evidence["note"] = (
        "The signed-rank statistic is computed inside WilcoxonPruner.prune; this record "
        "captures its configuration and the reference trial without re-running the test."
    )
    return (
        "Wilcoxon signed-rank test against the current best trial at p_threshold="
        f"{pruner._p_threshold}.",
        evidence,
        refs,
    )


def _explain_generic(
    pruner: BasePruner,
    study: Study,
    trial: FrozenTrial,
    step: int,
    include_refs: bool,
) -> tuple[str, dict[str, Any], list[TrialRef]]:
    # Fallback for user-defined pruners: record the candidate history visible at decision time.
    candidates = [
        t
        for t in study._get_trials(deepcopy=False, use_cache=True)
        if t._trial_id != trial._trial_id and step in t.intermediate_values
    ]
    refs = _refs_from_trials(candidates, step) if include_refs else []
    evidence = {
        "rule": "generic",
        "note": (
            "Custom pruner: its internal rule is not introspected. The referenced trials are "
            "the historical trials reporting this step that were visible at decision time."
        ),
        "n_candidates_at_step": len(candidates),
    }
    return (
        f"Decision delegated to custom pruner {type(pruner).__name__}; "
        f"{len(refs)} historical trial(s) reported step {step}.",
        evidence,
        refs,
    )


def _explain(
    pruner: BasePruner,
    study: Study,
    trial: FrozenTrial,
    step: int,
    include_refs: bool,
) -> tuple[str, dict[str, Any], list[TrialRef]]:
    if isinstance(pruner, PercentilePruner):
        return _explain_percentile(pruner, study, trial, step, include_refs)
    if isinstance(pruner, ThresholdPruner):
        return _explain_threshold(pruner, trial, step)
    if isinstance(pruner, PatientPruner):
        return _explain_patient(pruner, study, trial, step, include_refs)
    if isinstance(pruner, HyperbandPruner):
        return _explain_hyperband(pruner, study, trial, step, include_refs)
    if isinstance(pruner, SuccessiveHalvingPruner):
        return _explain_successive_halving(pruner, study, trial, step, include_refs)
    if isinstance(pruner, WilcoxonPruner):
        return _explain_wilcoxon(pruner, study, trial, step, include_refs)
    if isinstance(pruner, NopPruner):
        return "NopPruner never prunes.", {"rule": "nop"}, []
    return _explain_generic(pruner, study, trial, step, include_refs)


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------


class PruningRecorder:
    """Record and analyze pruning decisions without changing pruning behavior.

    The recorder is opt-in: attach one to a study via
    :func:`~optuna.study.create_study` (``pruning_recorder=...``) or
    :meth:`~optuna.pruners.PruningRecorder.attach`. When no recorder is attached,
    :meth:`~optuna.trial.Trial.should_prune` behaves exactly as before.

    Every call to :meth:`~optuna.trial.Trial.should_prune` produces one
    :class:`PruningDecisionRecord`. Statistics (per-pruner counts, pruning-rate curve,
    counterfactual estimates) are maintained incrementally; :meth:`summary` never scans the
    full trial history.

    Args:
        sink:
            Optional callable invoked with each :class:`PruningDecisionRecord` after it is
            stored (e.g. to stream records to disk). Any exception raised by the sink is
            logged as a warning and swallowed: a write failure never stops the search.
        confidence_level:
            Confidence level for counterfactual prediction intervals. Common values are
            0.90, 0.95 and 0.99.
        min_estimation_samples:
            Minimum number of completed historical trials with a value at the decision step
            required to fit the regression. Below it, the estimate is omitted with an
            explicit reason instead of an unsupported number.
        anomaly_window:
            Number of consecutive trials in the sliding window used for anomaly detection.
        anomaly_rate_threshold:
            Pruning rate within ``anomaly_window`` at or above which a streak of pruned
            trials is flagged as a potentially misconfigured pruner interval.
        anomaly_min_streak:
            Minimum number of consecutively pruned trials that triggers an anomaly flag.
    """

    def __init__(
        self,
        sink: Callable[[PruningDecisionRecord], None] | None = None,
        *,
        confidence_level: float = 0.95,
        min_estimation_samples: int = 5,
        anomaly_window: int = 10,
        anomaly_rate_threshold: float = 0.8,
        anomaly_min_streak: int = 8,
    ) -> None:
        if not 0.0 < confidence_level < 1.0:
            raise ValueError(
                f"confidence_level must be strictly between 0 and 1, got {confidence_level}."
            )
        if min_estimation_samples < 2:
            raise ValueError("min_estimation_samples must be at least 2.")
        if anomaly_window < 1 or anomaly_min_streak < 1:
            raise ValueError("anomaly_window and anomaly_min_streak must be positive.")
        if not 0.0 <= anomaly_rate_threshold <= 1.0:
            raise ValueError("anomaly_rate_threshold must be within [0, 1].")

        self._sink = sink
        self._confidence_level = confidence_level
        self._min_estimation_samples = min_estimation_samples
        self._anomaly_window = anomaly_window
        self._anomaly_rate_threshold = anomaly_rate_threshold
        self._anomaly_min_streak = anomaly_min_streak

        self._lock = threading.RLock()
        self._records: list[PruningDecisionRecord] = []
        # Incremental indexes.
        self._step_stats: dict[int, _StepStats] = defaultdict(_StepStats)
        self._completed_final: dict[int, float] = {}  # trial_number -> final value
        self._pruner_stats: dict[str, dict[str, int]] = defaultdict(
            lambda: {"decisions": 0, "pruned": 0}
        )
        self._prune_events: list[tuple[int, int, bool]] = []  # (trial_number, step, pruned)
        self._sink_failures = 0
        self._attached = False

    # -- lifecycle ---------------------------------------------------------

    def attach(self, study: Study) -> Study:
        """Attach this recorder to ``study`` and return the same study.

        Equivalent to passing the recorder through :func:`~optuna.study.create_study`.
        Existing completed trials in the study are indexed once (O(number of trials)); all
        subsequent updates are incremental.
        """

        with self._lock:
            study._pruning_recorder = self
            self._attached = True
            self.bootstrap(study)
        return study

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        # Locks and external sinks are not safely picklable; drop them.
        state["_lock"] = None
        state["_sink"] = None
        # Convert defaultdicts to plain dicts so their factory callables need not be picklable.
        state["_step_stats"] = dict(state["_step_stats"])
        state["_pruner_stats"] = dict(state["_pruner_stats"])
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._lock = threading.RLock()
        self._sink = None
        self._step_stats = defaultdict(_StepStats, self._step_stats)
        self._pruner_stats = defaultdict(lambda: {"decisions": 0, "pruned": 0}, self._pruner_stats)

    # -- incremental ingestion ---------------------------------------------

    def bootstrap(self, study: Study) -> None:
        """Build incremental indexes from trials already stored in ``study``.

        One-time O(n) scan used only when attaching to a study that already has history;
        steady-state updates go through :meth:`on_trial_complete`.
        """

        with self._lock:
            for t in study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)):
                if t.value is None:
                    continue
                self._ingest_completed(t.number, t.intermediate_values, float(t.value))

    def on_trial_complete(self, study: Study, trial: FrozenTrial) -> None:
        """Study-callback entry point: update incremental indexes by one trial."""

        if trial.state != TrialState.COMPLETE or trial.value is None:
            return
        with self._lock:
            self._ingest_completed(trial.number, trial.intermediate_values, float(trial.value))

    def _ingest_completed(
        self, number: int, intermediate_values: dict[int, float], final_value: float
    ) -> None:
        if number in self._completed_final:
            return
        self._completed_final[number] = final_value
        for step, x in intermediate_values.items():
            self._step_stats[step].update(float(x), final_value)

    # -- decision recording (called from Trial.should_prune) ---------------

    def _backfill_from_study(self, study: Study) -> None:
        """Index completed trials not yet seen.

        Reuses the study's thread-local trial cache (already populated by the pruner during
        the same ``should_prune`` call) and only ingests trials missing from the index, so
        the actual regression updates stay incremental even on the ``ask()/tell()`` path.
        """

        trials = study._get_trials(deepcopy=False, states=(TrialState.COMPLETE,), use_cache=True)
        for t in trials:
            if t.number not in self._completed_final and t.value is not None:
                self._ingest_completed(t.number, t.intermediate_values, float(t.value))

    def record_decision(
        self, study: Study, trial: FrozenTrial, pruner: BasePruner, pruned: bool
    ) -> PruningDecisionRecord:
        step = trial.last_step
        assert step is not None  # should_prune is only called after report()
        current_value = float(trial.intermediate_values[step])
        with self._lock:
            self._backfill_from_study(study)
        try:
            # Build per-trial references only when the trial is actually pruned: those are
            # the decisions that need post-mortem reconstruction. Non-pruning decisions keep
            # the quantitative basis and the comparison-set size instead.
            basis, evidence, refs = _explain(pruner, study, trial, step, include_refs=bool(pruned))
        except Exception as e:  # pragma: no cover - defensive, must not break search
            _logger.warning("Pruning evidence reconstruction failed: %s", e)
            basis, evidence, refs = (
                f"Evidence reconstruction failed ({e!r}); pruner returned {pruned}.",
                {"rule": "unavailable", "error": repr(e)},
                [],
            )

        estimate = None
        if pruned:
            estimate = self._estimate(step, current_value)
        else:
            evidence = dict(evidence)
            evidence["referenced_trials_note"] = (
                "Per-trial references are retained only on pruning decisions; the "
                "historical comparison population size is given by n_values_at_step / "
                "n_competing_at_rung / n_candidates_at_step."
            )

        with self._lock:
            record = PruningDecisionRecord(
                trial_number=trial.number,
                trial_id=trial._trial_id,
                step=step,
                intermediate_value=current_value,
                pruner_class=type(pruner).__name__,
                pruner_params=_pruner_params(pruner),
                pruned=bool(pruned),
                decision_basis=basis,
                evidence=evidence,
                referenced_trials=refs,
                estimate=estimate,
                decision_index=len(self._records),
                trial_state_at_decision=trial.state.name,
            )
            self._records.append(record)
            stats = self._pruner_stats[type(pruner).__name__]
            stats["decisions"] += 1
            if pruned:
                stats["pruned"] += 1
            self._prune_events.append((trial.number, step, bool(pruned)))
        self._write_to_sink(record)
        return record

    def _write_to_sink(self, record: PruningDecisionRecord) -> None:
        if self._sink is None:
            return
        try:
            self._sink(record)
        except Exception as e:
            with self._lock:
                self._sink_failures += 1
            _logger.warning(
                "Pruning record sink failed (search continues; %d failure(s) so far): %s",
                self._sink_failures,
                e,
            )

    # -- counterfactual estimation ------------------------------------------

    def _estimate(self, step: int, x0: float) -> EstimateInterval:
        with self._lock:
            stats = self._step_stats.get(step)
            if stats is None or stats.n < self._min_estimation_samples:
                have = 0 if stats is None else stats.n
                return EstimateInterval(
                    lower=None,
                    upper=None,
                    point_estimate=None,
                    confidence_level=self._confidence_level,
                    method=_METHOD_NAME,
                    method_description=_METHOD_DESCRIPTION,
                    n_samples=have,
                    insufficient_reason=(
                        f"insufficient_samples: {have} completed trial(s) with a value at "
                        f"step {step}, need at least {self._min_estimation_samples}"
                    ),
                )

            n = stats.n
            mean_x = stats.sum_x / n
            mean_y = stats.sum_y / n
            sxx = stats.sum_xx - stats.sum_x**2 / n
            if sxx <= 1e-12:
                return EstimateInterval(
                    lower=None,
                    upper=None,
                    point_estimate=None,
                    confidence_level=self._confidence_level,
                    method=_METHOD_NAME,
                    method_description=_METHOD_DESCRIPTION,
                    n_samples=n,
                    insufficient_reason="degenerate_fit: intermediate values at this step "
                    "have (near-)zero variance, so a regression slope is not identifiable",
                )
            slope = (stats.sum_xy - stats.sum_x * stats.sum_y / n) / sxx
            intercept = mean_y - slope * mean_x
            y_hat = intercept + slope * x0

            # Residual standard error.
            sse = stats.sum_yy - intercept * stats.sum_y - slope * stats.sum_xy
            sse = max(sse, 0.0)
            df = n - 2
            sigma2 = sse / df if df > 0 else 0.0
            sigma = math.sqrt(sigma2)
            if self._confidence_level in _T_TABLE and df <= 30 and df >= 1:
                tcrit = _T_TABLE[self._confidence_level][df - 1]
            elif df >= 1:
                tcrit = _z_critical(self._confidence_level)
            else:
                tcrit = float("nan")
            # Prediction interval: Var(y_new - y_hat) = sigma^2 * (1 + 1/n + (x0-mean_x)^2/Sxx)
            h = 1.0 + 1.0 / n + (x0 - mean_x) ** 2 / sxx
            margin = tcrit * sigma * math.sqrt(h)
            return EstimateInterval(
                lower=float(y_hat - margin),
                upper=float(y_hat + margin),
                point_estimate=float(y_hat),
                confidence_level=self._confidence_level,
                method=_METHOD_NAME,
                method_description=_METHOD_DESCRIPTION,
                n_samples=n,
            )

    # -- read APIs ----------------------------------------------------------

    @property
    def records(self) -> list[PruningDecisionRecord]:
        """All decision records in decision order."""

        with self._lock:
            return list(self._records)

    def records_for_trial(self, trial_number: int) -> list[PruningDecisionRecord]:
        """Return every decision recorded for one trial, identified by stable number."""

        with self._lock:
            return [r for r in self._records if r.trial_number == trial_number]

    def pruned_records(self) -> list[PruningDecisionRecord]:
        """Return only the decisions that pruned a trial."""

        with self._lock:
            return [r for r in self._records if r.pruned]

    def pruning_rate_curve(self, n_bins: int = 10) -> dict[str, Any]:
        """Pruning rate over the course of the search.

        Trials are partitioned into ``n_bins`` consecutive buckets by trial number, so the
        curve shows whether pruning concentrated early or late. Computed from the
        incrementally maintained event list, not from a storage scan.

        Returns:
            Dictionary with ``bin_edges``, ``bin_trial_counts``, ``pruned_counts`` and
            ``pruning_rates``.
        """

        with self._lock:
            if not self._prune_events:
                return {
                    "bin_edges": [],
                    "bin_trial_counts": [],
                    "pruned_counts": [],
                    "pruning_rates": [],
                }
            # One row per trial: whether that trial was ever pruned.
            pruned_by_trial: dict[int, bool] = {}
            for number, _, was_pruned in self._prune_events:
                pruned_by_trial[number] = pruned_by_trial.get(number, False) or was_pruned
            numbers = sorted(pruned_by_trial)
            # Split the *observed* trials into at most n_bins contiguous, non-empty groups
            # so the returned curve always has one entry per populated bin (no empty bins).
            actual_bins = min(n_bins, len(numbers))
            edges: list[list[int]] = []
            counts: list[int] = []
            pruned_counts: list[int] = []
            for b in range(actual_bins):
                start = b * len(numbers) // actual_bins
                end = (b + 1) * len(numbers) // actual_bins
                members = numbers[start:end]
                edges.append([members[0], members[-1]])
                counts.append(len(members))
                pruned_counts.append(sum(1 for n in members if pruned_by_trial[n]))
            rates = [p / c if c else 0.0 for p, c in zip(pruned_counts, counts)]
            return {
                "n_bins": actual_bins,
                "bin_edges": edges,
                "bin_trial_counts": counts,
                "pruned_counts": pruned_counts,
                "pruning_rates": rates,
            }

    def detect_anomalies(self) -> list[dict[str, Any]]:
        """Flag intervals where one pruner prunes an unusually large run of trials.

        Two rules, both evaluated on the incremental event stream:

        1. A streak of ``anomaly_min_streak`` consecutively pruned trials.
        2. A sliding window of ``anomaly_window`` trials with pruning rate at or above
           ``anomaly_rate_threshold``.
        """

        with self._lock:
            pruned_by_trial: dict[int, tuple[bool, str]] = {}
            pruner_by_trial: dict[int, str] = {}
            for number, _, was_pruned in self._prune_events:
                prev = pruned_by_trial.get(number, (False, ""))
                pruned_by_trial[number] = (prev[0] or was_pruned, prev[1])
            # Recover pruner class per pruned trial from records.
            for r in self._records:
                if r.pruned:
                    pruner_by_trial[r.trial_number] = r.pruner_class
            ordered = sorted(pruned_by_trial)
            flags: list[dict[str, Any]] = []

            # Rule 1: consecutive pruned streak.
            streak_start: int | None = None
            streak_len = 0
            for n in ordered:
                if pruned_by_trial[n][0]:
                    if streak_start is None:
                        streak_start = n
                    streak_len += 1
                else:
                    if streak_len >= self._anomaly_min_streak and streak_start is not None:
                        flags.append(
                            {
                                "type": "consecutive_pruned_streak",
                                "trial_numbers": [streak_start, n - 1],
                                "length": streak_len,
                                "pruner": pruner_by_trial.get(streak_start),
                            }
                        )
                    streak_start, streak_len = None, 0
            if streak_len >= self._anomaly_min_streak and streak_start is not None:
                flags.append(
                    {
                        "type": "consecutive_pruned_streak",
                        "trial_numbers": [streak_start, ordered[-1]],
                        "length": streak_len,
                        "pruner": pruner_by_trial.get(streak_start),
                    }
                )

            # Rule 2: high-rate sliding window.
            flags_set = {(f["type"], tuple(f["trial_numbers"])) for f in flags}
            window = self._anomaly_window
            for i in range(len(ordered)):
                block = ordered[i : i + window]
                if len(block) < window:
                    break
                pruned_here = sum(1 for n in block if pruned_by_trial[n][0])
                rate = pruned_here / window
                if rate >= self._anomaly_rate_threshold:
                    key = ("high_prune_rate_window", (block[0], block[-1]))
                    if key not in flags_set:
                        flags_set.add(key)
                        # Attribute the window to the pruner of its first pruned member;
                        # block[0] itself may be a non-pruned trial.
                        first_pruned = next((n for n in block if pruned_by_trial[n][0]), None)
                        pruner_name = (
                            pruner_by_trial.get(first_pruned) if first_pruned is not None else None
                        )
                        flags.append(
                            {
                                "type": "high_prune_rate_window",
                                "trial_numbers": [block[0], block[-1]],
                                "window": window,
                                "pruning_rate": rate,
                                "pruner": pruner_name,
                            }
                        )
            return flags

    def summary(self) -> dict[str, Any]:
        """Aggregate statistics maintained incrementally.

        Groups decisions by pruner class and includes the pruning-rate curve and anomaly
        flags. No full trial-history scan is performed.
        """

        with self._lock:
            total_decisions = len(self._records)
            total_pruned = sum(1 for r in self._records if r.pruned)
            per_pruner: dict[str, dict[str, Any]] = {}
            for name, stats in self._pruner_stats.items():
                decisions = stats["decisions"]
                per_pruner[name] = {
                    "decisions": decisions,
                    "pruned": stats["pruned"],
                    "pruning_rate": (stats["pruned"] / decisions if decisions else 0.0),
                }
            return {
                "total_decisions": total_decisions,
                "total_pruned": total_pruned,
                "overall_pruning_rate": (
                    total_pruned / total_decisions if total_decisions else 0.0
                ),
                "by_pruner": per_pruner,
                "pruning_rate_curve": self.pruning_rate_curve(),
                "anomalies": self.detect_anomalies(),
                "sink_failures": self._sink_failures,
                "indexed_completed_trials": len(self._completed_final),
            }

    def verify_consistency(self) -> dict[str, Any]:
        """Recompute aggregate counts from the detailed records and compare.

        Detail records are the source of truth. Any mismatch is reported as a warning entry
        and the recomputed (detail-derived) values are returned.
        """

        with self._lock:
            detail_pruner: dict[str, dict[str, int]] = defaultdict(
                lambda: {"decisions": 0, "pruned": 0}
            )
            total_decisions = 0
            total_pruned = 0
            for r in self._records:
                detail_pruner[r.pruner_class]["decisions"] += 1
                total_decisions += 1
                if r.pruned:
                    detail_pruner[r.pruner_class]["pruned"] += 1
                    total_pruned += 1

            warnings_out: list[str] = []
            for name, stats in self._pruner_stats.items():
                d = detail_pruner.get(name, {"decisions": 0, "pruned": 0})
                if d["decisions"] != stats["decisions"] or d["pruned"] != stats["pruned"]:
                    warnings_out.append(
                        f"Incremental counter for {name} {dict(stats)} disagrees with detail "
                        f"records {dict(d)}; detail records take precedence."
                    )
            if total_decisions != len(self._records):
                warnings_out.append("Total decision count disagrees with the record list.")
            for w in warnings_out:
                _logger.warning(w)
            return {
                "consistent": not warnings_out,
                "warnings": warnings_out,
                "detail_derived": {
                    "total_decisions": total_decisions,
                    "total_pruned": total_pruned,
                    "by_pruner": {k: dict(v) for k, v in detail_pruner.items()},
                },
            }
