class OptunaError(Exception):
    """Base class for Optuna specific errors."""

    pass


class TrialPruned(OptunaError):
    """Exception for pruned trials.

    This error tells a trainer that the current :class:`~optuna.trial.Trial` was pruned. It is
    supposed to be raised after :func:`optuna.trial.Trial.should_prune` as shown in the following
    example.

    See also:
        :class:`optuna.TrialPruned` is an alias of :class:`optuna.exceptions.TrialPruned`.

    Example:

        .. testcode::

            import numpy as np
            from sklearn.datasets import load_iris
            from sklearn.linear_model import SGDClassifier
            from sklearn.model_selection import train_test_split

            import optuna

            X, y = load_iris(return_X_y=True)
            X_train, X_valid, y_train, y_valid = train_test_split(X, y)
            classes = np.unique(y)


            def objective(trial):
                alpha = trial.suggest_float("alpha", 0.0, 1.0)
                clf = SGDClassifier(alpha=alpha)
                n_train_iter = 100

                for step in range(n_train_iter):
                    clf.partial_fit(X_train, y_train, classes=classes)

                    intermediate_value = clf.score(X_valid, y_valid)
                    trial.report(intermediate_value, step)

                    if trial.should_prune():
                        raise optuna.TrialPruned()

                return clf.score(X_valid, y_valid)


            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=20)
    """

    pass


class CLIUsageError(OptunaError):
    """Exception for CLI.

    CLI raises this exception when it receives invalid configuration.
    """

    pass


class StorageInternalError(OptunaError):
    """Exception for storage operation.

    This error is raised when an operation failed in backend DB of storage.
    """

    pass


class DuplicatedStudyError(OptunaError):
    """Exception for a duplicated study name.

    This error is raised when a specified study name already exists in the storage.
    """

    pass


class UpdateFinishedTrialError(OptunaError, RuntimeError):
    """Exception for updating a finished trial.

    This error is raised when attempting to update a finished trial.
    """

    pass


class InvalidConditionalSearchSpaceError(OptunaError, ValueError):
    """Exception for a malformed conditional search space declaration.

    This error is raised while a declaration is being created or validated, e.g. when a
    condition references an unknown parameter, when the dependency graph contains a cycle, or
    when two mutually exclusive branches overlap.
    """

    pass


class ConditionalConstraintViolation(OptunaError, ValueError):
    """Exception for parameter values rejected by a conditional search space declaration.

    The exception references the first condition that the values do not satisfy. It contains
    parameter names and the declared condition expression only.
    """

    def __init__(self, condition: str, param_name: str | None = None, reason: str = "") -> None:
        self.condition = condition
        self.param_name = param_name
        message = f"The declared condition `{condition}` is not satisfied"
        if param_name is not None:
            message += f" (parameter `{param_name}`)"
        if reason:
            message += f": {reason}"
        super().__init__(message + ".")


class UnsatisfiableConstraintError(InvalidConditionalSearchSpaceError):
    """Exception raised when a declared constraint has no feasible values.

    The message names the constraint and the parameters involved so the offending rule can be
    identified without a generic "no feasible solution" report.
    """

    def __init__(self, constraint: str, reason: str = "") -> None:
        self.constraint = constraint
        message = f"The declared constraint `{constraint}` has no feasible assignment"
        if reason:
            message += f": {reason}"
        super().__init__(message + ".")


class EnumerationLimitExceeded(InvalidConditionalSearchSpaceError):
    """Exception raised when a grid enumeration exceeds ``max_combinations``."""

    def __init__(self, limit: int, reached: int) -> None:
        self.limit = limit
        self.reached = reached
        super().__init__(
            f"Enumerating the conditional search space exceeded the limit of {limit} "
            f"combinations (stopped after {reached}). Raise `max_combinations` to enumerate more."
        )


class ConditionalSearchSpaceVersionError(OptunaError, ValueError):
    """Exception raised when a study's declaration version does not match the current one."""

    def __init__(self, expected: str, actual: str, diff: str = "") -> None:
        self.expected = expected
        self.actual = actual
        self.diff = diff
        message = (
            "The conditional search space declaration of this study does not match the current "
            f"one (stored version {expected}, current version {actual})."
        )
        if diff:
            message += f"\nDifference:\n{diff}"
        super().__init__(message)


class ExperimentalWarning(Warning):
    """Experimental Warning class.

    This implementation exists here because the policy of `FutureWarning` has been changed
    since Python 3.7 was released. See the details in
    https://docs.python.org/3/library/warnings.html#warning-categories.
    """

    pass
