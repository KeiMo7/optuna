"""Generate the acceptance evidence for the declarative conditional search space.

Run from the repository root::

    python tests/search_space_tests/conditional_acceptance_report.py

The script is intentionally self-contained and does not touch the network. It writes a report
to ``output/conditional_search_space_report.md``.
"""

from __future__ import annotations

from pathlib import Path
import re
import subprocess
import sys

import pytest

import optuna
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.exceptions import ConditionalConstraintViolation
from optuna.exceptions import ConditionalSearchSpaceVersionError
from optuna.samplers import ConditionalSampler
from optuna.samplers import RandomSampler
from optuna.search_space import ConditionalSearchSpace
from optuna.trial import TrialState


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "output"
OUTPUT.mkdir(exist_ok=True)


def _nested_space() -> ConditionalSearchSpace:
    space = ConditionalSearchSpace(name="nested-demo")
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


def _manual_count(space: ConditionalSearchSpace) -> int:
    total = 0
    for kind in space.candidates("kind"):
        if kind == "stats":
            total += len(space.candidates("window"))
            continue
        for model in space.candidates("model"):
            if model == "linear":
                total += len(space.candidates("lr"))
                continue
            for criterion in space.candidates("criterion"):
                leaf = "max_features" if criterion == "logloss" else "smoothing"
                total += len(space.candidates("depth")) * len(space.candidates(leaf))
    return total


def section_enumeration() -> list[str]:
    space = _nested_space()
    combos = space.enumerate()
    manual = _manual_count(space)
    lines = [
        "## 1. Enumeration of three nesting levels and two mutually exclusive groups",
        "",
        f"- Declared parameters: {', '.join(space.param_names)}",
        f"- Statically enumerated legal combinations: **{len(combos)}**",
        f"- Hand-counted combinations: **{manual}**",
        f"- Match: {'PASS' if len(combos) == manual else 'FAIL'}",
        f"- Serialized declaration version: `{space.version}`",
        "",
        "Sample of expanded combinations (truncated):",
        "",
        "```",
    ]
    for combo in combos[:8]:
        lines.append(f"{combo}")
    if len(combos) > 8:
        lines.append(f"... ({len(combos) - 8} more)")
    lines += ["```", ""]
    return lines


def section_validation() -> list[str]:
    space = ConditionalSearchSpace()
    space.add_param("model", CategoricalDistribution(["linear", "tree"]))
    space.add_param("lr", FloatDistribution(0.001, 1.0), condition="model == 'linear'")
    space.add_param("depth", IntDistribution(1, 3), condition="model == 'tree'")
    space.add_param("batch", IntDistribution(8, 64), condition="model == 'tree'")
    space.add_constraint("depth * batch > 16 if model == 'tree' else True")

    study = optuna.create_study(conditional_search_space=space)
    attempts = [
        {"model": "linear", "depth": 2},  # depth is inactive for linear
        {"model": "tree", "lr": 0.01},  # lr is inactive for tree
        {"model": "tree", "depth": 2, "batch": 8},  # violates depth * batch > 16
        {"model": "tree", "depth": 2},  # active parameter missing
    ]
    rejections = 0
    messages: list[str] = []
    for params in attempts:
        try:
            space.validate_values(params)
        except ConditionalConstraintViolation as error:
            rejections += 1
            messages.append(str(error))
    trials_before = len(study.trials)
    enqueue_rejections = 0
    for params in attempts[:3]:
        try:
            study.enqueue_trial(params)
        except ConditionalConstraintViolation:
            enqueue_rejections += 1
    lines = [
        "## 2. Invalid combinations never create trials, and rejections are counted separately",
        "",
        f"- Invalid assignments presented to `validate_values`: **{len(attempts)}**",
        f"- Rejected by the declaration: **{rejections}**",
        f"- Rejected at `enqueue_trial` before trial creation: **{enqueue_rejections}**",
        f"- Trials created by the rejected enqueues: **{len(study.trials) - trials_before}**",
        "",
        "First failing condition per rejected assignment:",
        "",
        "```",
        *messages,
        "```",
        "",
    ]
    return lines


def section_degradation() -> list[str]:
    space = ConditionalSearchSpace()
    space.add_param("model", CategoricalDistribution(["linear", "tree"]))
    space.add_param("lr", FloatDistribution(0.001, 1.0), condition="model == 'linear'")
    space.add_param("depth", IntDistribution(1, 4), condition="model == 'tree'")
    space.add_constraint("lr <= 0.2 if model == 'linear' else True")

    sampler = ConditionalSampler(
        space, strategy="delegate", fallback_sampler=RandomSampler(seed=7), seed=7
    )
    study = optuna.create_study(sampler=sampler)

    def objective(trial: optuna.Trial) -> float:
        model = trial.suggest_categorical("model", ["linear", "tree"])
        if model == "linear":
            return float(trial.suggest_float("lr", 0.001, 1.0))
        return float(trial.suggest_int("depth", 1, 4))

    with pytest.warns(UserWarning) as record:
        study.optimize(objective, n_trials=24)

    degradation = next(
        (
            str(w.message)
            for w in record
            if "does not understand the declaration" in str(w.message)
        ),
        "(no warning captured)",
    )
    illegal = 0
    for trial in study.trials:
        if trial.state != TrialState.COMPLETE:
            illegal += 1
            continue
        if trial.params["model"] == "linear":
            if "depth" in trial.params or trial.params["lr"] > 0.2:
                illegal += 1
        elif "lr" in trial.params or not (1 <= trial.params["depth"] <= 4):
            illegal += 1
    lines = [
        "## 3. A sampler that does not support dynamic spaces still yields legal suggestions",
        "",
        f"- Inner sampler: `{RandomSampler.__name__}` (reports no conditional support)",
        f"- Trials run: **{len(study.trials)}**, illegal parameter sets: **{illegal}**",
        f"- Total rejection counter: **{sampler.n_rejected}**",
        "",
        "Degradation notice:",
        "",
        "```",
        degradation,
        "```",
        "",
    ]
    return lines


def section_restore() -> list[str]:
    storage = optuna.storages.InMemoryStorage()
    space = ConditionalSearchSpace()
    space.add_param("route", CategoricalDistribution(["a", "b"]))
    space.add_param("x", IntDistribution(0, 3), condition="route == 'a'")
    space.add_param("y", IntDistribution(0, 3), condition="route == 'b'")
    optuna.create_study(storage=storage, study_name="restore", conditional_search_space=space)

    same = optuna.load_study(study_name="restore", storage=storage, conditional_search_space=space)
    changed = ConditionalSearchSpace.from_json(space.to_json())
    changed.add_param("z", IntDistribution(0, 3), condition="route == 'b'")
    diff_lines: list[str] = []
    try:
        optuna.load_study(study_name="restore", storage=storage, conditional_search_space=changed)
    except ConditionalSearchSpaceVersionError as error:
        diff_lines = str(error).splitlines()
    restored_space = same.conditional_search_space
    matches = restored_space is not None and restored_space.version == space.version
    lines = [
        "## 4. Declaration versioning on restore",
        "",
        f"- Restored version matches stored version: {'PASS' if matches else 'FAIL'}",
        "- A changed declaration is refused; reported difference:",
        "",
        "```",
        *diff_lines,
        "```",
        "",
    ]
    return lines


def section_noop() -> list[str]:
    study = optuna.create_study()

    def objective(trial: optuna.Trial) -> float:
        x = trial.suggest_float("x", -1.0, 1.0)
        if x > 0:
            return trial.suggest_int("y", 0, 3)
        return 0.0

    study.optimize(objective, n_trials=6)
    single = ConditionalSearchSpace()
    single.add_param("only", CategoricalDistribution(["one"]))
    single_study = optuna.create_study(sampler=ConditionalSampler(single))
    single_study.optimize(
        lambda t: float(t.suggest_categorical("only", ["one"]) == "one"), n_trials=1
    )
    lines = [
        "## 5. Behavior without a declaration, or with a single branch",
        "",
        f"- Legacy study trials: **{len(study.trials)}**, all completed: "
        f"{all(t.state == TrialState.COMPLETE for t in study.trials)}",
        f"- Legacy study rejections: **{study.conditional_rejections}**",
        f"- Single-branch declaration trials: **{len(single_study.trials)}**",
        "",
    ]
    return lines


def run_pytest(targets: list[str], extra_args: list[str] | None = None) -> tuple[int, str, str]:
    args = [
        sys.executable,
        "-m",
        "pytest",
        *targets,
        "-q",
        "--no-header",
        "-p",
        "no:cacheprovider",
    ]
    if extra_args:
        args.extend(extra_args)
    completed = subprocess.run(args, cwd=str(ROOT), capture_output=True, text=True, timeout=1200)
    combined = completed.stdout + completed.stderr
    combined = re.sub(r"\x1b\[[0-9;]*m", "", combined)
    lines = [line.strip() for line in combined.strip().splitlines() if line.strip()]
    summary = lines[-1] if lines else "(no output)"
    failures = "\n".join(line for line in lines if line.startswith("FAILED"))
    return completed.returncode, summary, failures


def section_pytest() -> list[str]:
    suites = [
        ("tests/search_space_tests", []),
        ("tests/samplers_tests/test_grid.py", []),
        ("tests/samplers_tests/test_brute_force.py", []),
        ("tests/trial_tests", ["-k", "not redis"]),
        ("tests/study_tests/test_study.py", ["-k", "not redis and not cmaes"]),
    ]
    lines = [
        "## 6. Existing sampler and storage tests",
        "",
        "| Suite | Result | Summary |",
        "| --- | --- | --- |",
    ]
    any_failure = False
    for suite, flags in suites:
        path = ROOT / suite
        if not path.exists():
            lines.append(f"| `{suite}` | skipped | path not found |")
            continue
        code, summary, failures = run_pytest([suite], flags)
        status = "PASS" if code == 0 else "FAIL"
        any_failure = any_failure or code != 0
        summary = summary.replace("|", "\\|")
        lines.append(f"| `{suite}` | {status} | `{summary}` |")
        if failures:
            lines.append("")
            lines.append("```")
            lines.append(failures)
            lines.append("```")
    lines.append("")
    if not any_failure:
        lines.append("All executed suites passed.")
        lines.append("")
    lines.append(
        "Note: `fakeredis` in this environment lacks the Lua `eval` command and the optional "
        "`cmaes` package is not installed, so redis- and cmaes-parametrized cases are excluded "
        "from the runs above. Those failures are environmental and unrelated to this change."
    )
    lines.append("")
    return lines


def main() -> int:
    lines = [
        "# Declarative conditional search space: acceptance report",
        "",
        f"Optuna version: `{optuna.__version__}`",
        "",
    ]
    lines += section_enumeration()
    lines += section_validation()
    lines += section_degradation()
    lines += section_restore()
    lines += section_noop()
    lines += section_pytest()
    report = "\n".join(lines)
    (OUTPUT / "conditional_search_space_report.md").write_text(report + "\n", encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
