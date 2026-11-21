from __future__ import annotations

import json
import logging

import pytest

import optuna
from optuna.pruners import BasePruner
from optuna.pruners import MedianPruner
from optuna.pruners import NopPruner
from optuna.pruners import PercentilePruner
from optuna.pruners import PruningDecisionRecord
from optuna.pruners import PruningRecorder
from optuna.pruners import ThresholdPruner
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


def _make_study(pruner: BasePruner, recorder: PruningRecorder | None = None, **kwargs):
    sampler = optuna.samplers.RandomSampler(seed=0)
    return optuna.create_study(pruner=pruner, sampler=sampler, pruning_recorder=recorder, **kwargs)


def _reporting_objective(n_steps=6):
    def objective(trial: optuna.Trial) -> float:
        x = trial.suggest_float("x", -10, 10)
        for step in range(n_steps):
            # Strong monotone signal between x and objective so regression is identifiable.
            v = (x - 2.0) ** 2 + 0.1 * step
            trial.report(v, step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return (x - 2.0) ** 2

    return objective


# ---------------------------------------------------------------------------
# Case 1: normal pruning records
# ---------------------------------------------------------------------------


def test_normal_pruning_decision_is_recorded_with_full_evidence() -> None:
    recorder = PruningRecorder()
    study = _make_study(MedianPruner(n_startup_trials=2, n_warmup_steps=1), recorder)
    study.optimize(_reporting_objective(), n_trials=15)

    records = recorder.records
    assert len(records) > 0
    assert all(isinstance(r, PruningDecisionRecord) for r in records)

    # Every decision carries the mandatory fields.
    for r in records:
        assert r.step >= 0
        assert isinstance(r.intermediate_value, float)
        assert r.pruner_class == "MedianPruner"
        assert r.pruner_params["n_startup_trials"] == 2
        assert r.decision_basis  # non-empty explanation
        assert isinstance(r.evidence, dict)
        assert r.evidence["rule"] == "percentile"
        # Stable identifiers, never list indices (only populated on pruning decisions).
        for ref in r.referenced_trials:
            assert isinstance(ref.number, int)
            assert isinstance(ref.trial_id, int)
        if not r.pruned:
            # Non-pruning decisions retain the comparison-set size, not per-trial objects.
            assert r.referenced_trials == []
            assert "referenced_trials_note" in r.evidence

    pruned = recorder.pruned_records()
    if pruned:
        r = pruned[0]
        assert r.pruned is True
        assert r.referenced_trials, "a pruned decision must name the trials it was compared to"
        assert "threshold_percentile_value" in r.evidence
        # Referenced numbers must match real trials in the study.
        real_numbers = {t.number for t in study.trials}
        assert all(ref.number in real_numbers for ref in r.referenced_trials)


def test_records_are_json_serializable_and_indexed() -> None:
    recorder = PruningRecorder()
    study = _make_study(MedianPruner(n_startup_trials=1, n_warmup_steps=0), recorder)
    study.optimize(_reporting_objective(), n_trials=8)

    for i, r in enumerate(recorder.records):
        payload = json.dumps(r.to_dict())
        assert json.loads(payload)["decision_index"] == i

    # Per-trial lookup uses the stable trial number.
    some_trial = recorder.records[0].trial_number
    assert all(r.trial_number == some_trial for r in recorder.records_for_trial(some_trial))


def test_grouping_by_pruner_type() -> None:
    recorder = PruningRecorder()
    study = _make_study(ThresholdPruner(upper=1e9), recorder)  # never prunes, but records
    study.optimize(_reporting_objective(n_steps=3), n_trials=4)

    summary = recorder.summary()
    assert "ThresholdPruner" in summary["by_pruner"]
    assert summary["by_pruner"]["ThresholdPruner"]["decisions"] > 0
    assert summary["total_decisions"] == sum(v["decisions"] for v in summary["by_pruner"].values())


def test_pruning_rate_curve_partitions_search_progress() -> None:
    recorder = PruningRecorder()
    study = _make_study(MedianPruner(n_startup_trials=1, n_warmup_steps=0), recorder)
    study.optimize(_reporting_objective(), n_trials=12)

    curve = recorder.pruning_rate_curve(n_bins=4)
    # Exactly n_bins non-empty entries (or fewer when there are fewer trials).
    assert len(curve["pruning_rates"]) == curve["n_bins"] == len(curve["bin_edges"])
    assert all(0.0 <= rate <= 1.0 for rate in curve["pruning_rates"])
    assert sum(curve["bin_trial_counts"]) >= 1
    assert sum(curve["pruned_counts"]) <= sum(curve["bin_trial_counts"])


def test_estimate_interval_carries_method_and_confidence_level() -> None:
    # Acceptance assertion #3: every numeric estimate is labeled with method + confidence.
    recorder = PruningRecorder(confidence_level=0.95, min_estimation_samples=3)
    study = _make_study(MedianPruner(n_startup_trials=1, n_warmup_steps=0), recorder)
    study.optimize(_reporting_objective(), n_trials=20)

    estimates = [r.estimate for r in recorder.pruned_records()]
    assert estimates, "the scenario should produce at least one pruned trial"
    for est in estimates:
        assert est is not None
        assert est.method and est.method_description
        assert est.confidence_level == 0.95
        if est.is_estimate:
            assert est.lower is not None and est.upper is not None
            assert est.lower <= est.point_estimate <= est.upper
            assert est.n_samples >= 3
        else:
            # Degraded case: explicit reason, no unsupported number.
            assert est.insufficient_reason is not None
            assert est.lower is None and est.upper is None


# ---------------------------------------------------------------------------
# Case 2: insufficient samples -> no interval, explicit reason
# ---------------------------------------------------------------------------


def test_insufficient_samples_yields_no_interval_with_reason() -> None:
    recorder = PruningRecorder(min_estimation_samples=10)
    study = _make_study(MedianPruner(n_startup_trials=1, n_warmup_steps=0), recorder)
    study.optimize(_reporting_objective(), n_trials=8)

    for r in recorder.pruned_records():
        est = r.estimate
        assert est is not None and est.is_estimate is False
        assert est.lower is None and est.upper is None and est.point_estimate is None
        assert "insufficient_samples" in est.insufficient_reason
        # Method/confidence metadata is still present even when no number is given.
        assert est.method and 0.0 < est.confidence_level < 1.0


# ---------------------------------------------------------------------------
# Case 3: anomalous pruning rate is flagged
# ---------------------------------------------------------------------------


class _AlwaysPruneAfterWarmup(BasePruner):
    def prune(self, study, trial: FrozenTrial) -> bool:
        if trial.last_step is None or trial.last_step < 1:
            return False
        return True


def test_anomalous_pruning_streak_is_detected() -> None:
    recorder = PruningRecorder(anomaly_window=5, anomaly_rate_threshold=0.8, anomaly_min_streak=4)
    study = _make_study(_AlwaysPruneAfterWarmup(), recorder)
    study.optimize(_reporting_objective(), n_trials=10)

    anomalies = recorder.detect_anomalies()
    assert anomalies, "a pruner that prunes every trial must trigger an anomaly flag"
    kinds = {a["type"] for a in anomalies}
    assert "consecutive_pruned_streak" in kinds or "high_prune_rate_window" in kinds
    for a in anomalies:
        assert a["trial_numbers"][0] <= a["trial_numbers"][1]
        # The responsible pruner must always be attributable, never null.
        assert a["pruner"] == "_AlwaysPruneAfterWarmup"

    # The same flags surface through the aggregate summary.
    assert recorder.summary()["anomalies"] == anomalies


# ---------------------------------------------------------------------------
# Case 4: sink write failure -> warning + search continues
# ---------------------------------------------------------------------------


def test_sink_failure_warns_and_continues(caplog: pytest.LogCaptureFixture) -> None:
    calls = {"n": 0}

    def flaky_sink(record: PruningDecisionRecord) -> None:
        calls["n"] += 1
        raise OSError("disk full")

    recorder = PruningRecorder(sink=flaky_sink)
    study = _make_study(MedianPruner(n_startup_trials=1, n_warmup_steps=0), recorder)
    optuna.logging.enable_propagation()
    try:
        with caplog.at_level(logging.WARNING):
            study.optimize(_reporting_objective(), n_trials=6)
    finally:
        optuna.logging.disable_propagation()

    # Search completed and records were still stored locally despite the sink failing.
    assert len(recorder.records) == calls["n"] > 0
    assert recorder.summary()["sink_failures"] == calls["n"]
    assert any("sink failed" in rec.message for rec in caplog.records)


def test_evidence_reconstruction_failure_does_not_break_search(monkeypatch, caplog) -> None:
    import optuna.pruners._recording as recording_mod

    def boom(*args, **kwargs):
        raise RuntimeError("explain gone wrong")

    monkeypatch.setattr(recording_mod, "_explain", boom)
    recorder = PruningRecorder()
    study = _make_study(MedianPruner(n_startup_trials=1, n_warmup_steps=0), recorder)
    optuna.logging.enable_propagation()
    try:
        with caplog.at_level(logging.WARNING):
            study.optimize(_reporting_objective(), n_trials=4)
    finally:
        optuna.logging.disable_propagation()
    assert len(recorder.records) > 0
    assert any("reconstruction failed" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# Hard guarantee: pruning timing and outcome are unchanged by recording
# ---------------------------------------------------------------------------


def test_pruning_decisions_are_identical_with_and_without_recorder() -> None:
    n_trials = 20

    def run(attach: bool):
        recorder = PruningRecorder() if attach else None
        study = _make_study(
            MedianPruner(n_startup_trials=2, n_warmup_steps=1, interval_steps=2),
            recorder,
        )
        study.optimize(_reporting_objective(), n_trials=n_trials)
        states = [
            (t.number, t.state, dict(sorted(t.intermediate_values.items()))) for t in study.trials
        ]
        return states, recorder

    without, _ = run(False)
    with_rec, recorder = run(True)

    # Same trial numbers, same states, same reported steps/values -> same timing and result.
    assert without == with_rec
    # And the recorded verdicts match the stored trial states.
    pruned_numbers = {n for n, state, _ in without if state == TrialState.PRUNED}
    recorded_pruned = {r.trial_number for r in recorder.pruned_records()}
    assert recorded_pruned == pruned_numbers


def test_disabled_recorder_path_is_untouched() -> None:
    study = _make_study(MedianPruner(), None)
    assert study.pruning_recorder is None
    study.optimize(_reporting_objective(), n_trials=5)
    # No attribute side effects on the trial path when the feature is off.
    assert study.pruning_recorder is None


def test_nop_pruner_records_but_never_prunes() -> None:
    recorder = PruningRecorder()
    study = _make_study(NopPruner(), recorder)
    study.optimize(_reporting_objective(), n_trials=5)
    assert len(recorder.records) > 0
    assert recorder.pruned_records() == []
    assert all(t.state == TrialState.COMPLETE for t in study.trials)


def test_referenced_trials_use_stable_numbers_not_positions() -> None:
    recorder = PruningRecorder(min_estimation_samples=2)
    study = _make_study(PercentilePruner(25.0, n_startup_trials=1), recorder)
    study.optimize(_reporting_objective(), n_trials=10)
    # References exist only on pruning decisions and must resolve by stable number.
    assert recorder.pruned_records(), "scenario should prune at least one trial"
    for r in recorder.pruned_records():
        assert r.referenced_trials
        for ref in r.referenced_trials:
            trial = study.trials[ref.number]
            assert trial._trial_id == ref.trial_id
            assert trial.number == ref.number


def test_verify_consistency_reports_detail_truth() -> None:
    recorder = PruningRecorder()
    study = _make_study(MedianPruner(n_startup_trials=1), recorder)
    study.optimize(_reporting_objective(), n_trials=10)

    report = recorder.verify_consistency()
    assert report["consistent"] is True
    assert report["detail_derived"]["total_decisions"] == len(recorder.records)

    # Simulate incremental-counter corruption: detail records must win and warn.
    name = next(iter(recorder._pruner_stats))
    recorder._pruner_stats[name]["decisions"] += 999
    report = recorder.verify_consistency()
    assert report["consistent"] is False
    assert report["warnings"]
    assert report["detail_derived"]["by_pruner"][name]["decisions"] == len(recorder.records)


def test_bootstrap_indexes_existing_completed_trials() -> None:
    # First study has history but no recorder.
    study = _make_study(MedianPruner(n_startup_trials=1), None)
    study.optimize(_reporting_objective(), n_trials=8)

    recorder = PruningRecorder(min_estimation_samples=2)
    recorder.attach(study)
    indexed = recorder.summary()["indexed_completed_trials"]
    n_completed = len(study.get_trials(states=(TrialState.COMPLETE,)))
    assert indexed == n_completed


def test_pickle_drops_lock_and_sink() -> None:
    import pickle

    recorder = PruningRecorder(sink=lambda r: None)
    study = _make_study(MedianPruner(n_startup_trials=1), recorder)
    study.optimize(_reporting_objective(), n_trials=4)

    restored = pickle.loads(pickle.dumps(recorder))
    assert isinstance(restored, PruningRecorder)
    assert len(restored.records) == len(recorder.records)
    assert restored._sink is None


def test_parallel_jobs_record_without_corruption() -> None:
    recorder = PruningRecorder()
    study = _make_study(MedianPruner(n_startup_trials=2, n_warmup_steps=1), recorder)
    study.optimize(_reporting_objective(), n_trials=12, n_jobs=2)
    assert recorder.verify_consistency()["consistent"] is True
    assert len(recorder.records) == recorder.summary()["total_decisions"]


def test_ask_tell_path_backfills_incremental_index() -> None:
    # Without Study.optimize(), the recorder must still index completed trials lazily so
    # counterfactual estimates have data.
    recorder = PruningRecorder(min_estimation_samples=2)
    study = _make_study(MedianPruner(n_startup_trials=1, n_warmup_steps=0), recorder)

    def run_one(n: int) -> None:
        trial = study.ask()
        x = trial.suggest_float("x", -10, 10)
        target = (x - 2.0) ** 2
        for step in range(5):
            trial.report(target + 0.1 * step, step)
            if trial.should_prune():
                study.tell(trial, state=TrialState.PRUNED)
                return
        study.tell(trial, target)

    for n in range(10):
        run_one(n)

    assert len(recorder.records) > 0
    # Backfill populated the estimation index from completed ask/tell trials.
    assert recorder.summary()["indexed_completed_trials"] >= 1
    assert recorder.verify_consistency()["consistent"] is True


def test_invalid_constructor_arguments_are_rejected() -> None:
    with pytest.raises(ValueError):
        PruningRecorder(confidence_level=1.5)
    with pytest.raises(ValueError):
        PruningRecorder(min_estimation_samples=1)
    with pytest.raises(ValueError):
        PruningRecorder(anomaly_window=0)
    with pytest.raises(ValueError):
        PruningRecorder(anomaly_rate_threshold=2.0)
