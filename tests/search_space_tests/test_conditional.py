from __future__ import annotations

import threading

import pytest

import optuna
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.exceptions import ConditionalConstraintViolation
from optuna.exceptions import ConditionalSearchSpaceVersionError
from optuna.exceptions import EnumerationLimitExceeded
from optuna.exceptions import InvalidConditionalSearchSpaceError
from optuna.exceptions import UnsatisfiableConstraintError
from optuna.samplers import ConditionalSampler
from optuna.samplers import RandomSampler
from optuna.search_space import ConditionalSearchSpace
from optuna.trial import TrialState


def _nested_space() -> ConditionalSearchSpace:
    """Two mutually exclusive groups and three levels of nesting.

    ``kind == "ml"`` splits into ``model == "linear"`` and ``model == "tree"``; the tree branch
    nests further into ``criterion == "logloss"`` and ``criterion == "gini"``.
    """

    space = ConditionalSearchSpace(name="nested")
    space.add_param("kind", CategoricalDistribution(["ml", "stats"]))
    space.add_param("model", CategoricalDistribution(["linear", "tree"]), condition="kind == 'ml'")
    space.add_param(
        "lr",
        FloatDistribution(0.001, 1.0),
        condition="kind == 'ml' and model == 'linear'",
        grid=[0.001, 0.01, 0.1],
    )
    space.add_param("depth", IntDistribution(2, 4), condition="kind == 'ml' and model == 'tree'")
    space.add_param(
        "criterion",
        CategoricalDistribution(["logloss", "gini"]),
        condition="kind == 'ml' and model == 'tree'",
    )
    space.add_param(
        "max_features",
        IntDistribution(1, 2),
        condition="kind == 'ml' and model == 'tree' and criterion == 'logloss'",
    )
    space.add_param(
        "smoothing",
        IntDistribution(1, 2),
        condition="kind == 'ml' and model == 'tree' and criterion == 'gini'",
    )
    space.add_param("window", IntDistribution(1, 3), condition="kind == 'stats'")
    return space


def _manual_nested_count(space: ConditionalSearchSpace) -> int:
    combos = set()
    for kind in space.candidates("kind"):
        if kind == "stats":
            for window in space.candidates("window"):
                combos.add((("kind", kind), ("window", window)))
            continue
        for model in space.candidates("model"):
            if model == "linear":
                for lr in space.candidates("lr"):
                    combos.add((("kind", kind), ("model", model), ("lr", lr)))
                continue
            for depth in space.candidates("depth"):
                for criterion in space.candidates("criterion"):
                    if criterion == "logloss":
                        for value in space.candidates("max_features"):
                            combos.add(
                                (
                                    ("kind", kind),
                                    ("model", model),
                                    ("depth", depth),
                                    ("criterion", criterion),
                                    ("max_features", value),
                                )
                            )
                    else:
                        for value in space.candidates("smoothing"):
                            combos.add(
                                (
                                    ("kind", kind),
                                    ("model", model),
                                    ("depth", depth),
                                    ("criterion", criterion),
                                    ("smoothing", value),
                                )
                            )
    return len(combos)


def test_enumeration_matches_manual_count() -> None:
    space = _nested_space()
    combos = space.enumerate()
    assert len(combos) == _manual_nested_count(space)
    # stats: 3 windows; ml/linear: 3 lr values; ml/tree: 3 depths x 2 criteria x 2 leaf values.
    assert len(combos) == 3 + 3 + 12
    # Each expanded assignment is legal and absence means "not active", not "missing".
    for combo in combos:
        space.validate_values(combo)
    assert len({tuple(sorted(combo.items())) for combo in combos}) == len(combos)


def test_enumeration_deterministic_and_serializable() -> None:
    space = _nested_space()
    payload = space.to_json()
    restored = ConditionalSearchSpace.from_json(payload)
    assert restored.version == space.version
    assert restored.enumerate() == space.enumerate()
    assert restored.diff(space) == "(no declarative difference)"


def test_branch_group_declaration() -> None:
    space = ConditionalSearchSpace()
    space.add_param("route", CategoricalDistribution(["a", "b"]))
    space.add_param("x", IntDistribution(1, 2))
    space.add_param("y", IntDistribution(1, 2))
    space.add_branch_group("route_group", "route", {"a": ["x"], "b": ["y"]})
    space.validate()

    assert space.condition("x") == "route == 'a'"
    assert len(space.enumerate()) == 4
    space.validate_values({"route": "a", "x": 1})
    with pytest.raises(ConditionalConstraintViolation) as info:
        space.validate_values({"route": "b", "x": 1})
    # Providing an inactive branch parameter reports that branch's condition.
    assert "route == 'a'" in str(info.value)


def test_constraints_block_before_suggestion() -> None:
    space = ConditionalSearchSpace()
    space.add_param("low", IntDistribution(1, 5))
    space.add_param("high", IntDistribution(1, 5))
    space.add_param("ratio", IntDistribution(1, 5), condition="high > 2")
    space.add_constraint("low < high")
    space.add_constraint("ratio <= high if high > 2 else True")

    assert space.check_values({"low": 1, "high": 4, "ratio": 4}) is None
    violation = space.check_values({"low": 4, "high": 4, "ratio": 1})
    assert violation is not None
    assert "low < high" in violation.condition
    # An inactive parameter must be absent. This is reported as a violation of its own
    # activation condition, independently of any cross-parameter constraint.
    violation = space.check_values({"low": 1, "high": 2, "ratio": 1})
    assert violation is not None
    assert violation.param_name == "ratio"
    assert "high > 2" in violation.condition


def test_ternary_constraint_is_evaluated_lazily() -> None:
    # Regression: a ternary whose false branch references an inactive parameter must still
    # enforce the selected branch. Previously the whole constraint was skipped because one of
    # its names was absent.
    space = ConditionalSearchSpace()
    space.add_param("model", CategoricalDistribution(["linear", "tree"]))
    space.add_param("lr", IntDistribution(1, 5), condition="model == 'linear'")
    space.add_param("depth", IntDistribution(1, 5), condition="model == 'tree'")
    space.add_constraint("lr < depth if model == 'tree' else lr <= 3")

    assert space.check_values({"model": "linear", "lr": 2}) is None
    violation = space.check_values({"model": "linear", "lr": 4})
    assert violation is not None
    assert "lr <= 3" in violation.condition
    # On the tree branch the `depth` comparison applies once both values are present.
    assert space.check_values({"model": "tree", "depth": 3}) is None
    assert space.check_values({"model": "tree", "depth": 5, "lr": 1}) is not None


def test_constraint_unsatisfiable_names_constraint() -> None:
    space = ConditionalSearchSpace()
    space.add_param("x", IntDistribution(1, 2))
    space.add_param("y", IntDistribution(1, 2))
    space.add_constraint("x + y > 10")
    with pytest.raises(UnsatisfiableConstraintError) as info:
        space.validate()
    assert "x + y > 10" in str(info.value)


def test_declaration_errors() -> None:
    space = ConditionalSearchSpace()
    space.add_param("a", IntDistribution(0, 1), condition="b == 1")
    space.add_param("b", IntDistribution(0, 1), condition="a == 1")
    with pytest.raises(InvalidConditionalSearchSpaceError) as info:
        space.validate()
    assert "cycle" in str(info.value)

    space = ConditionalSearchSpace()
    space.add_param("a", IntDistribution(0, 1), condition="missing == 1")
    with pytest.raises(InvalidConditionalSearchSpaceError) as info:
        space.validate()
    assert "unknown parameter" in str(info.value)
    assert "missing" in str(info.value)

    space = ConditionalSearchSpace()
    space.add_param("route", CategoricalDistribution(["a", "b"]))
    space.add_param("x", IntDistribution(0, 1))
    space.add_branch_group("g1", "route", {"a": ["x"], "b": ["x"]})
    with pytest.raises(InvalidConditionalSearchSpaceError) as info:
        space.validate()
    assert "more than one branch" in str(info.value)

    space = ConditionalSearchSpace()
    with pytest.raises(InvalidConditionalSearchSpaceError):
        space.add_param("x", IntDistribution(0, 1), grid=[5])
    with pytest.raises(InvalidConditionalSearchSpaceError):
        space.add_param("y", IntDistribution(0, 1))
        space.add_param("y", IntDistribution(0, 1))
    with pytest.raises(InvalidConditionalSearchSpaceError):
        space.add_param("z", IntDistribution(0, 1), condition="0 <")
    # A branch group with an unknown parent is reported by `validate`.
    space = ConditionalSearchSpace()
    space.add_param("x", IntDistribution(0, 1))
    space.add_branch_group("g", "unknown", {"a": ["x"]})
    with pytest.raises(InvalidConditionalSearchSpaceError):
        space.validate()


def test_error_messages_do_not_leak_values() -> None:
    space = ConditionalSearchSpace()
    space.add_param("secret_mode", CategoricalDistribution(["tenant-a", "tenant-b"]))
    space.add_param("x", IntDistribution(0, 1), condition="secret_mode == 'tenant-a'")
    with pytest.raises(ConditionalConstraintViolation) as info:
        space.validate_values({"secret_mode": "tenant-b", "x": 1})
    message = str(info.value)
    assert "secret_mode" in message
    # The condition expression is shown, but the value that failed is not.
    assert "tenant-b" not in message
    assert "x" in message


def test_iteration_limit() -> None:
    space = ConditionalSearchSpace()
    space.add_param("x", IntDistribution(1, 100))
    space.add_param("y", IntDistribution(1, 100))
    with pytest.raises(EnumerationLimitExceeded):
        space.enumerate(max_combinations=10)


def test_infinite_domain_requires_grid() -> None:
    space = ConditionalSearchSpace()
    space.add_param("x", FloatDistribution(0.0, 1.0))
    with pytest.raises(InvalidConditionalSearchSpaceError) as info:
        space.enumerate()
    assert "infinite domain" in str(info.value)


def test_disabled_designation_is_noop() -> None:
    space = ConditionalSearchSpace(enabled=False)
    space.add_param("a", CategoricalDistribution(["x", "y"]))
    space.add_param("b", IntDistribution(0, 1), condition="a == 'x'")
    space.validate()
    assert space.check_values({"a": "y", "b": 1}) is None
    assert space.check_values({"a": "x"}) is None


def test_declaration_version_changes_with_content() -> None:
    a = ConditionalSearchSpace()
    a.add_param("x", IntDistribution(0, 1))
    b = ConditionalSearchSpace()
    b.add_param("x", IntDistribution(0, 2))
    assert a.version != b.version
    assert "distribution changed" in b.diff(a)


# ---------------------------------------------------------------------------- sampler tests


def _run(study: optuna.Study, n_trials: int) -> None:
    def objective(trial: optuna.Trial) -> float:
        kind = trial.suggest_categorical("kind", ["ml", "stats"])
        if kind == "stats":
            return float(trial.suggest_int("window", 1, 3))
        model = trial.suggest_categorical("model", ["linear", "tree"])
        if model == "linear":
            return float(trial.suggest_float("lr", 0.001, 1.0))
        depth = trial.suggest_int("depth", 2, 4)
        criterion = trial.suggest_categorical("criterion", ["logloss", "gini"])
        if criterion == "logloss":
            return float(depth + trial.suggest_int("max_features", 1, 2))
        return float(depth + trial.suggest_int("smoothing", 1, 2))

    study.optimize(objective, n_trials=n_trials)


def test_enumerate_sampler_covers_legal_combinations() -> None:
    space = _nested_space()
    study = optuna.create_study(sampler=ConditionalSampler(space, seed=0))
    _run(study, len(space.enumerate()))

    assert len(study.trials) == len(space.enumerate())
    seen = {tuple(sorted(t.params.items())) for t in study.trials}
    expected = {tuple(sorted(combo.items())) for combo in space.enumerate()}
    assert seen == expected
    assert all(t.state == TrialState.COMPLETE for t in study.trials)


def test_delegate_sampler_with_unsupported_sampler_yields_legal_values() -> None:
    space = ConditionalSearchSpace()
    space.add_param("model", CategoricalDistribution(["linear", "tree"]))
    space.add_param("lr", FloatDistribution(0.001, 1.0), condition="model == 'linear'")
    space.add_param("depth", IntDistribution(1, 3), condition="model == 'tree'")
    space.add_constraint("lr < 0.5 if model == 'linear' else True")

    sampler = ConditionalSampler(
        space, strategy="delegate", fallback_sampler=RandomSampler(seed=1), seed=1
    )
    study = optuna.create_study(sampler=sampler)

    def objective(trial: optuna.Trial) -> float:
        model = trial.suggest_categorical("model", ["linear", "tree"])
        if model == "linear":
            return float(trial.suggest_float("lr", 0.001, 1.0))
        return float(trial.suggest_int("depth", 1, 3))

    with pytest.warns(UserWarning, match="does not understand the declaration"):
        study.optimize(objective, n_trials=20)

    for trial in study.trials:
        if trial.params["model"] == "linear":
            assert "depth" not in trial.params
            assert trial.params["lr"] < 0.5
        else:
            assert "lr" not in trial.params
            assert trial.params["depth"] in (1, 2, 3)
    assert space.check_values(study.best_trial.params) is None


def test_invalid_enqueue_does_not_create_trial_or_consume_budget() -> None:
    space = ConditionalSearchSpace()
    space.add_param("model", CategoricalDistribution(["linear", "tree"]))
    space.add_param(
        "lr", FloatDistribution(0.001, 1.0), condition="model == 'linear'", grid=[0.01, 0.5]
    )
    space.add_param("depth", IntDistribution(1, 3), condition="model == 'tree'")

    study = optuna.create_study(sampler=ConditionalSampler(space), conditional_search_space=space)
    before = len(study.trials)
    for _ in range(5):
        with pytest.raises(ConditionalConstraintViolation):
            study.enqueue_trial({"model": "linear", "depth": 2})
    assert len(study.trials) == before == 0
    assert study.conditional_rejections == {}  # enqueue rejection is not a trial suggestion

    study.enqueue_trial({"model": "linear", "lr": 0.5})
    assert len(study.trials) == 1
    assert study.trials[0].state == TrialState.WAITING


def test_rejections_are_counted_separately_from_failures() -> None:
    space = ConditionalSearchSpace()
    space.add_param("x", IntDistribution(1, 3))
    space.add_param("y", FloatDistribution(0.0, 10.0))
    space.add_constraint("x < y")

    sampler = ConditionalSampler(space, strategy="delegate")

    class _IllegalSampler(RandomSampler):
        def sample_independent(self, study, trial, param_name, param_distribution):  # type: ignore[no-untyped-def]
            if param_name == "x":
                return 3
            return 0.0

    sampler._fallback_sampler = _IllegalSampler(seed=0)
    study = optuna.create_study(sampler=sampler)

    def objective(trial: optuna.Trial) -> float:
        x = trial.suggest_int("x", 1, 3)
        y = trial.suggest_float("y", 0.0, 10.0)
        return float(x + y)

    study.optimize(objective, n_trials=3, catch=(ConditionalConstraintViolation,))
    failed = [t for t in study.trials if t.state == TrialState.FAIL]
    assert len(failed) == 3
    assert study.conditional_rejections == {"x < y": 3}
    assert sampler.n_rejected == 3
    # Each rejected trial carries the violated condition, so an invalid combination is
    # distinguishable from a normal objective failure in the history.
    assert all(
        t.system_attrs.get("conditional_search_space:rejected_condition") == "x < y"
        for t in failed
    )
    # Rejected assignments are tracked outside of trial failure states.
    assert not any(t.state == TrialState.PRUNED for t in study.trials)


def test_restore_detects_declaration_change() -> None:
    storage = optuna.storages.InMemoryStorage()
    space = ConditionalSearchSpace()
    space.add_param("a", IntDistribution(0, 1))
    space.add_param("b", IntDistribution(0, 1), condition="a == 1")
    optuna.create_study(storage=storage, study_name="s", conditional_search_space=space)

    loaded = optuna.load_study(study_name="s", storage=storage, conditional_search_space=space)
    assert loaded.conditional_search_space is not None
    assert loaded.conditional_search_space.version == space.version

    changed = ConditionalSearchSpace.from_json(space.to_json())
    changed.add_param("c", IntDistribution(0, 1), condition="a == 1")
    with pytest.raises(ConditionalSearchSpaceVersionError) as info:
        optuna.load_study(study_name="s", storage=storage, conditional_search_space=changed)
    assert "parameter `c`: added" in str(info.value)

    # Loading without an explicit declaration restores the stored one.
    restored = optuna.load_study(study_name="s", storage=storage)
    assert restored.conditional_search_space is not None
    assert restored.conditional_search_space.version == space.version


def test_legacy_objective_without_declaration_still_works() -> None:
    study = optuna.create_study()
    assert study.conditional_search_space is None

    def objective(trial: optuna.Trial) -> float:
        x = trial.suggest_float("x", -1, 1)
        if x > 0:
            return trial.suggest_int("y", 0, 3)
        return 0.0

    study.optimize(objective, n_trials=4)
    assert len(study.trials) == 4
    assert study.conditional_rejections == {}


def test_declaration_mismatch_with_objective_raises_before_sampling() -> None:
    space = ConditionalSearchSpace()
    space.add_param("x", IntDistribution(1, 3))
    study = optuna.create_study(sampler=ConditionalSampler(space))

    def objective(trial: optuna.Trial) -> float:
        return float(trial.suggest_int("x", 1, 5))

    study.optimize(objective, n_trials=1, catch=(InvalidConditionalSearchSpaceError,))
    trial = study.trials[0]
    assert trial.state == TrialState.FAIL
    # The declaration and the objective disagree; no parameter was recorded before failing.
    assert trial.params == {}
    assert len(study.trials) == 1


def test_enumerate_sampler_grid_equivalence() -> None:
    # A declaration with no conditions behaves like a plain grid.
    space = ConditionalSearchSpace()
    space.add_param("a", CategoricalDistribution(["p", "q"]))
    space.add_param("b", IntDistribution(1, 2))
    study = optuna.create_study(sampler=ConditionalSampler(space, seed=3))
    study.optimize(
        lambda t: float(ord(t.suggest_categorical("a", ["p", "q"])) + t.suggest_int("b", 1, 2)),
        n_trials=4,
    )
    seen = {tuple(sorted(t.params.items())) for t in study.trials}
    expected = {
        tuple(sorted(combo.items()))
        for combo in (
            {"a": "p", "b": 1},
            {"a": "p", "b": 2},
            {"a": "q", "b": 1},
            {"a": "q", "b": 2},
        )
    }
    assert seen == expected


def test_declaration_serialization_roundtrip_with_branches_and_constraints() -> None:
    space = ConditionalSearchSpace(name="roundtrip")
    space.add_param("route", CategoricalDistribution(["a", "b"]))
    space.add_param("x", IntDistribution(1, 3))
    space.add_param("y", IntDistribution(1, 3))
    space.add_branch_group("routes", "route", {"a": ["x"], "b": ["y"]})
    space.add_constraint("x != 3", name="avoid_three")
    space.validate()

    payload = space.to_json()
    restored = ConditionalSearchSpace.from_json(payload)
    assert restored.to_dict() == space.to_dict()
    assert restored.version == space.version
    assert restored.enumerate() == space.enumerate()
    # route=a: x in {1, 2} (the constraint removes x=3); route=b: y in {1, 2, 3}.
    assert len(restored.enumerate()) == 5


def test_concurrent_optimize_uses_disjoint_combinations() -> None:
    space = ConditionalSearchSpace()
    space.add_param("a", IntDistribution(0, 1))
    space.add_param("b", IntDistribution(0, 1))
    study = optuna.create_study(
        sampler=ConditionalSampler(space, seed=0),
        storage=optuna.storages.InMemoryStorage(),
    )

    def objective(trial: optuna.Trial) -> float:
        return float(trial.suggest_int("a", 0, 1) + trial.suggest_int("b", 0, 1))

    threads = [
        threading.Thread(target=study.optimize, args=(objective,), kwargs={"n_trials": 2})
        for _ in range(2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    seen = [tuple(sorted(t.params.items())) for t in completed]
    assert len(seen) == len(set(seen)) == 4
    assert set(seen) == {tuple(sorted(c.items())) for c in space.enumerate()}


@pytest.mark.parametrize("strategy", ["enumerate", "delegate"])
def test_strategy_accepts_only_known_values(strategy: str) -> None:
    space = ConditionalSearchSpace()
    space.add_param("x", IntDistribution(0, 1), grid=[0, 1])
    sampler = ConditionalSampler(space, strategy=strategy)
    assert sampler.search_space is space
    with pytest.raises(ValueError):
        ConditionalSampler(space, strategy="bogus")


def test_enumerate_sampler_is_permuted_by_seed() -> None:
    # Enumeration order does not matter for coverage, but the sampler shuffles it so that a
    # truncated run is not biased toward declaration order.
    space = ConditionalSearchSpace()
    space.add_param("x", IntDistribution(0, 9))
    first = ConditionalSampler(space, seed=1)
    second = ConditionalSampler(space, seed=2)
    assert first._combinations is not None and second._combinations is not None
    assert first._combinations != second._combinations
    assert {combo["x"] for combo in first._combinations} == set(range(10))


def test_study_wraps_plain_sampler_when_declaration_is_attached() -> None:
    space = ConditionalSearchSpace()
    space.add_param("route", CategoricalDistribution(["a", "b"]))
    space.add_param("x", IntDistribution(0, 2), condition="route == 'a'")
    space.add_param("y", IntDistribution(0, 2), condition="route == 'b'")

    study = optuna.create_study(sampler=RandomSampler(seed=0), conditional_search_space=space)
    assert isinstance(study.sampler, ConditionalSampler)
    assert study.sampler.search_space is space

    def objective(trial: optuna.Trial) -> float:
        route = trial.suggest_categorical("route", ["a", "b"])
        if route == "a":
            return float(trial.suggest_int("x", 0, 2))
        return float(trial.suggest_int("y", 0, 2))

    with pytest.warns(UserWarning, match="does not understand the declaration"):
        study.optimize(objective, n_trials=10)
    for trial in study.trials:
        if trial.params["route"] == "a":
            assert "y" not in trial.params
        else:
            assert "x" not in trial.params


def test_add_trial_validates_before_creation() -> None:
    space = ConditionalSearchSpace()
    space.add_param("route", CategoricalDistribution(["a", "b"]))
    space.add_param("x", IntDistribution(0, 2), condition="route == 'a'")
    study = optuna.create_study(conditional_search_space=space)

    illegal = optuna.trial.create_trial(
        params={"route": "b", "x": 1},
        distributions={
            "route": CategoricalDistribution(["a", "b"]),
            "x": IntDistribution(0, 2),
        },
        value=1.0,
    )
    with pytest.raises(ConditionalConstraintViolation):
        study.add_trial(illegal)
    assert len(study.trials) == 0

    legal = optuna.trial.create_trial(
        params={"route": "a", "x": 1},
        distributions={
            "route": CategoricalDistribution(["a", "b"]),
            "x": IntDistribution(0, 2),
        },
        value=1.0,
    )
    study.add_trial(legal)
    assert len(study.trials) == 1


def test_copy_study_carries_the_declaration() -> None:
    space = ConditionalSearchSpace()
    space.add_param("route", CategoricalDistribution(["a", "b"]))
    space.add_param("x", IntDistribution(0, 2), condition="route == 'a'")
    source_storage = optuna.storages.InMemoryStorage()
    source = optuna.create_study(
        storage=source_storage, study_name="source", conditional_search_space=space
    )
    source.optimize(
        lambda t: float(
            t.suggest_categorical("route", ["a", "b"]) == "a" and t.suggest_int("x", 0, 2) >= 0
        ),
        n_trials=3,
    )

    target_storage = optuna.storages.InMemoryStorage()
    optuna.copy_study(
        from_study_name="source",
        from_storage=source_storage,
        to_storage=target_storage,
        to_study_name="target",
    )
    target = optuna.load_study(study_name="target", storage=target_storage)
    assert target.conditional_search_space is not None
    assert target.conditional_search_space.version == space.version
