"""Declarative conditional search spaces.

This module provides :class:`ConditionalSearchSpace`, a serializable declaration of parameter
dependencies, mutually exclusive branches and cross-parameter numeric constraints.  The
declaration is the single source of truth: it can be validated, statically enumerated and
serialized without executing an objective function.

The implementation is pure computation and depends only on :mod:`optuna.distributions` and
:mod:`optuna.exceptions`.  Declaration and validation errors reference parameter names and
condition expressions only; they never embed parameter values.

Example:

    .. testcode::

        import optuna
        from optuna.distributions import CategoricalDistribution
        from optuna.distributions import FloatDistribution
        from optuna.distributions import IntDistribution

        space = optuna.search_space.ConditionalSearchSpace()
        space.add_param("model", CategoricalDistribution(["linear", "tree"]))
        space.add_param(
            "lr",
            FloatDistribution(1e-4, 1e-1, log=True),
            condition="model == 'linear'",
            grid=[1e-4, 1e-3, 1e-2],
        )
        space.add_param("max_depth", IntDistribution(2, 8), condition="model == 'tree'")

        assert space.validate_values({"model": "linear", "lr": 0.01}) is None
        assert space.validate_values({"model": "tree", "max_depth": 3}) is None

        # Inactive parameters are absent from an enumerated combination, not reported missing.
        combos = space.enumerate()
        assert len(combos) == 3 + 7  # 3 linear leaves + 7 tree depths
"""

from __future__ import annotations

import ast
from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
import decimal
import hashlib
import json
import math
from typing import Any

from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution
from optuna.distributions import distribution_to_json
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.distributions import json_to_distribution
from optuna.exceptions import ConditionalConstraintViolation
from optuna.exceptions import ConditionalSearchSpaceVersionError
from optuna.exceptions import EnumerationLimitExceeded
from optuna.exceptions import InvalidConditionalSearchSpaceError
from optuna.exceptions import UnsatisfiableConstraintError


SCHEMA_VERSION = 1

_ALLOWED_BINOPS: tuple[type[ast.operator], ...] = (
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
)
_ALLOWED_CMPOPS: tuple[type[ast.cmpop], ...] = (
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.Eq,
    ast.NotEq,
    ast.In,
    ast.NotIn,
)
_ALLOWED_UNARYOPS: tuple[type[ast.unaryop], ...] = (ast.Not, ast.USub, ast.UAdd)
_JSON_SCALAR_TYPES = (type(None), bool, int, float, str)


@dataclasses.dataclass(frozen=True)
class _ParamDecl:
    name: str
    distribution: BaseDistribution
    condition: str | None
    grid: tuple[Any, ...] | None


@dataclasses.dataclass(frozen=True)
class _BranchGroup:
    name: str
    parent: str
    # Each branch is ``(value, params)`` where ``params`` are the parameters active for ``value``.
    branches: tuple[tuple[Any, tuple[str, ...]], ...]


@dataclasses.dataclass(frozen=True)
class _Constraint:
    name: str
    expression: str


@dataclasses.dataclass(frozen=True)
class _Violation:
    """The first declared condition that a parameter assignment does not satisfy."""

    condition: str
    param_name: str | None
    reason: str

    def as_exception(self) -> ConditionalConstraintViolation:
        return ConditionalConstraintViolation(
            condition=self.condition, param_name=self.param_name, reason=self.reason
        )


def _is_json_scalar(value: Any) -> bool:
    return any(isinstance(value, t) for t in _JSON_SCALAR_TYPES)


def _parse_expression(expression: str, context: str) -> ast.Expression:
    try:
        parsed = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as e:
        raise InvalidConditionalSearchSpaceError(
            f"The {context} expression `{expression}` is not a valid Python expression."
        ) from e
    return parsed


def _names_in(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _canonical(expression: str, context: str) -> str:
    return ast.unparse(_parse_expression(expression, context))


def _eval_node(node: ast.expr, env: Mapping[str, Any], expression: str) -> Any:
    if isinstance(node, ast.Constant):
        if not _is_json_scalar(node.value):
            raise InvalidConditionalSearchSpaceError(
                f"The condition `{expression}` uses an unsupported constant."
            )
        return node.value

    if isinstance(node, ast.Name):
        if node.id not in env:
            raise KeyError(node.id)
        return env[node.id]

    if isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, _ALLOWED_UNARYOPS):
            raise InvalidConditionalSearchSpaceError(
                f"The condition `{expression}` uses an unsupported unary operator."
            )
        operand = _eval_node(node.operand, env, expression)
        if isinstance(node.op, ast.Not):
            return not operand
        if isinstance(node.op, ast.USub):
            return -operand
        return +operand

    if isinstance(node, ast.BinOp):
        if not isinstance(node.op, _ALLOWED_BINOPS):
            raise InvalidConditionalSearchSpaceError(
                f"The condition `{expression}` uses an unsupported binary operator."
            )
        left = _eval_node(node.left, env, expression)
        right = _eval_node(node.right, env, expression)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.FloorDiv):
            return left // right
        return left % right

    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            result: Any = True
            for value_node in node.values:
                result = _eval_node(value_node, env, expression)
                if not result:
                    return result
            return result
        if isinstance(node.op, ast.Or):
            result = False
            for value_node in node.values:
                result = _eval_node(value_node, env, expression)
                if result:
                    return result
            return result
        raise InvalidConditionalSearchSpaceError(
            f"The condition `{expression}` uses an unsupported boolean operator."
        )

    if isinstance(node, ast.Compare):
        left = _eval_node(node.left, env, expression)
        for op, comparator in zip(node.ops, node.comparators):
            if not isinstance(op, _ALLOWED_CMPOPS):
                raise InvalidConditionalSearchSpaceError(
                    f"The condition `{expression}` uses an unsupported comparison operator."
                )
            right = _eval_node(comparator, env, expression)
            if isinstance(op, ast.Lt):
                ok = left < right
            elif isinstance(op, ast.LtE):
                ok = left <= right
            elif isinstance(op, ast.Gt):
                ok = left > right
            elif isinstance(op, ast.GtE):
                ok = left >= right
            elif isinstance(op, ast.Eq):
                ok = left == right
            elif isinstance(op, ast.NotEq):
                ok = left != right
            elif isinstance(op, ast.In):
                ok = left in right
            else:
                ok = left not in right
            if not ok:
                return False
            left = right
        return True

    if isinstance(node, ast.IfExp):
        return (
            _eval_node(node.body, env, expression)
            if _eval_node(node.test, env, expression)
            else _eval_node(node.orelse, env, expression)
        )

    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return tuple(_eval_node(elt, env, expression) for elt in node.elts)

    raise InvalidConditionalSearchSpaceError(
        f"The condition `{expression}` uses an unsupported expression construct."
    )


def _eval_expression(expression: str, env: Mapping[str, Any]) -> Any:
    parsed = _parse_expression(expression, "condition")
    assert isinstance(parsed.body, ast.expr)
    return _eval_node(parsed.body, env, expression)


def _missing_names(expression: str, env: Mapping[str, Any]) -> bool:
    return not _names_in(_parse_expression(expression, "condition").body) <= set(env)


def _float_candidates(low: float, high: float, step: float) -> tuple[float, ...]:
    d_low = decimal.Decimal(str(low))
    d_high = decimal.Decimal(str(high))
    d_step = decimal.Decimal(str(step))
    values: list[float] = []
    current = d_low
    while current <= d_high:
        values.append(float(current))
        current += d_step
    return tuple(values)


class ConditionalSearchSpace:
    """A declarative, serializable conditional search space.

    A declaration consists of parameters, optional boolean ``condition`` expressions,
    mutually exclusive ``branch groups`` and cross-parameter ``constraints``.

    The parameters in a declaration are *the* source of truth for which parameters exist in a
    trial.  A parameter whose condition does not hold is considered absent, not missing.

    Args:
        name:
            An optional human-readable name for the declaration. It is included in the
            serialized form and in the version hash.
    """

    def __init__(self, name: str | None = None, enabled: bool = True) -> None:
        self.name = name
        self._enabled = enabled
        self._params: dict[str, _ParamDecl] = {}
        self._branch_groups: list[_BranchGroup] = []
        self._constraints: list[_Constraint] = []
        self._validated = False
        self._order: tuple[str, ...] = ()
        self._effective_conditions: dict[str, str | None] = {}

    @property
    def enabled(self) -> bool:
        """Whether the declaration is enforced.

        A disabled declaration keeps its structural checks but stops constraining trials, so
        the behavior matches a study without a declaration.
        """

        return self._enabled

    def disable(self) -> None:
        """Stop enforcing the declaration without discarding it."""

        self._enabled = False

    # ------------------------------------------------------------------ declaration

    def add_param(
        self,
        name: str,
        distribution: BaseDistribution,
        condition: str | None = None,
        grid: Sequence[Any] | None = None,
    ) -> None:
        """Declare a parameter.

        Args:
            name:
                The parameter name.
            distribution:
                The distribution used by ``trial.suggest_*``.
            condition:
                A boolean expression over other declared parameter names. The parameter is
                active only while the expression holds.
            grid:
                Optional explicit candidate values. When given, :meth:`enumerate` uses these
                values instead of deriving them from ``distribution``.
        """

        if name in self._params:
            raise InvalidConditionalSearchSpaceError(
                f"The parameter `{name}` is declared more than once."
            )
        if not name.isidentifier():
            raise InvalidConditionalSearchSpaceError(
                f"The parameter name `{name}` is not a valid identifier."
            )
        condition = _canonical(condition, "condition") if condition is not None else None
        if grid is not None:
            values = tuple(grid)
            if len(values) == 0:
                raise InvalidConditionalSearchSpaceError(
                    f"The grid of the parameter `{name}` is empty."
                )
            for value in values:
                if not self._contains(distribution, value):
                    raise InvalidConditionalSearchSpaceError(
                        f"The grid of the parameter `{name}` contains a value that is not "
                        f"contained in the declared distribution."
                    )
            grid = values
        self._params[name] = _ParamDecl(
            name=name, distribution=distribution, condition=condition, grid=grid
        )
        self._validated = False

    def add_branch_group(
        self,
        name: str,
        parent: str,
        branches: Mapping[Any, Sequence[str]],
    ) -> None:
        """Declare mutually exclusive branches selected by ``parent``.

        Exactly one branch can be active in a trial.  Parameters listed in a branch are active
        only when ``parent`` takes the branch's value.

        Args:
            name:
                The group name. Must be unique.
            parent:
                The selecting parameter. It must be declared before :meth:`validate`.
            branches:
                A mapping from a value of ``parent`` to the parameters active for that value.
        """

        if any(group.name == name for group in self._branch_groups):
            raise InvalidConditionalSearchSpaceError(
                f"The branch group `{name}` is declared more than once."
            )
        if len(branches) == 0:
            raise InvalidConditionalSearchSpaceError(
                f"The branch group `{name}` must contain at least one branch."
            )
        normalized: list[tuple[Any, tuple[str, ...]]] = []
        for value, params in branches.items():
            if not _is_json_scalar(value):
                raise InvalidConditionalSearchSpaceError(
                    f"The branch group `{name}` has a branch value that is not JSON scalar."
                )
            params_tuple = tuple(params)
            if len(params_tuple) == 0:
                raise InvalidConditionalSearchSpaceError(
                    f"A branch of the group `{name}` has no parameter."
                )
            for param in params_tuple:
                if not param.isidentifier():
                    raise InvalidConditionalSearchSpaceError(
                        f"A branch of the group `{name}` references an invalid parameter name."
                    )
            normalized.append((value, params_tuple))
        self._branch_groups.append(
            _BranchGroup(name=name, parent=parent, branches=tuple(normalized))
        )
        self._validated = False

    def add_constraint(self, expression: str, name: str | None = None) -> None:
        """Declare a cross-parameter constraint.

        The expression must evaluate to a truthy value whenever all parameters it references
        are active.  It may be a ternary expression such as ``x < y if cond else True``.
        """

        canonical = _canonical(expression, "constraint")
        if any(c.expression == canonical for c in self._constraints):
            raise InvalidConditionalSearchSpaceError(
                f"The constraint `{canonical}` is declared more than once."
            )
        if name is None:
            name = f"constraint_{len(self._constraints)}"
        if any(c.name == name for c in self._constraints):
            raise InvalidConditionalSearchSpaceError(
                f"The constraint name `{name}` is declared more than once."
            )
        self._constraints.append(_Constraint(name=name, expression=canonical))
        self._validated = False

    # ------------------------------------------------------------------ validation

    def validate(self) -> None:
        """Validate the declaration.

        Raises:
            :class:`~optuna.exceptions.InvalidConditionalSearchSpaceError`:
                If a condition references an unknown parameter, the dependency graph has a
                cycle, a branch overlaps another branch, or a constraint references an
                unknown parameter.
        """

        if self._validated:
            return

        if len(self._params) == 0:
            raise InvalidConditionalSearchSpaceError(
                "The conditional search space does not declare any parameter."
            )

        branch_params: dict[str, str] = {}
        parent_values: dict[str, list[tuple[str, Any]]] = {}
        for group in self._branch_groups:
            if group.parent not in self._params:
                raise InvalidConditionalSearchSpaceError(
                    f"The branch group `{group.name}` references the unknown parent parameter "
                    f"`{group.parent}`."
                )
            parent_decl = self._params[group.parent]
            for value, params in group.branches:
                if not self._contains(parent_decl.distribution, value):
                    raise InvalidConditionalSearchSpaceError(
                        f"The branch group `{group.name}` has a branch value that is not in the "
                        f"domain of the parent parameter `{group.parent}`."
                    )
                for param in params:
                    if param not in self._params:
                        raise InvalidConditionalSearchSpaceError(
                            f"The branch group `{group.name}` references the unknown parameter "
                            f"`{param}`."
                        )
                    if param == group.parent:
                        raise InvalidConditionalSearchSpaceError(
                            f"The parameter `{param}` cannot depend on itself in the branch "
                            f"group `{group.name}`."
                        )
                    if param in branch_params:
                        if branch_params[param] == group.name:
                            raise InvalidConditionalSearchSpaceError(
                                f"The parameter `{param}` appears in more than one branch of "
                                f"the branch group `{group.name}`."
                            )
                        raise InvalidConditionalSearchSpaceError(
                            f"The parameter `{param}` belongs to overlapping branches: "
                            f"`{branch_params[param]}` and `{group.name}`."
                        )
                    branch_params[param] = group.name
                parent_values.setdefault(group.parent, []).append((group.name, value))

        for parent, declared in parent_values.items():
            for i, (group_a, value_a) in enumerate(declared):
                for group_b, value_b in declared[i + 1 :]:
                    if group_a == group_b:
                        continue
                    if self._same_value(value_a, value_b):
                        raise InvalidConditionalSearchSpaceError(
                            f"The branches of `{group_a}` and `{group_b}` overlap on a value of "
                            f"the parent parameter `{parent}`."
                        )

        effective: dict[str, str | None] = {}
        for name, decl in self._params.items():
            conditions = [] if decl.condition is None else [decl.condition]
            for group in self._branch_groups:
                for value, params in group.branches:
                    if name in params:
                        conditions.append(f"{group.parent} == {value!r}")
            effective[name] = (
                _canonical(" and ".join(f"({c})" for c in conditions), "condition")
                if conditions
                else None
            )

        known = set(self._params)
        for name, condition in effective.items():
            if condition is None:
                continue
            unknown = _names_in(_parse_expression(condition, "condition").body) - known
            if unknown:
                raise InvalidConditionalSearchSpaceError(
                    f"The condition of the parameter `{name}` references an unknown parameter: "
                    f"`{sorted(unknown)[0]}`."
                )

        for constraint in self._constraints:
            unknown = (
                _names_in(_parse_expression(constraint.expression, "constraint").body) - known
            )
            if unknown:
                raise InvalidConditionalSearchSpaceError(
                    f"The constraint `{constraint.expression}` references an unknown parameter: "
                    f"`{sorted(unknown)[0]}`."
                )

        self._order = self._topological_order(effective)
        self._effective_conditions = effective
        self._validated = True
        try:
            self._check_constraints_satisfiable()
        except Exception:
            self._validated = False
            raise

    def _topological_order(self, effective: Mapping[str, str | None]) -> tuple[str, ...]:
        dependencies = {
            name: set(self._condition_names_from(cond)) for name, cond in effective.items()
        }
        dependencies = {
            name: {d for d in deps if d in self._params and d != name}
            for name, deps in dependencies.items()
        }
        order: list[str] = []
        permanent: set[str] = set()
        temporary: list[str] = []

        def visit(name: str) -> None:
            if name in permanent:
                return
            if name in temporary:
                cycle = temporary[temporary.index(name) :] + [name]
                raise InvalidConditionalSearchSpaceError(
                    "The parameter conditions form a cycle: "
                    + " -> ".join(f"`{n}`" for n in cycle)
                    + "."
                )
            temporary.append(name)
            for dependency in sorted(dependencies[name]):
                visit(dependency)
            temporary.pop()
            permanent.add(name)
            order.append(name)

        for name in self._params:
            visit(name)
        return tuple(order)

    @staticmethod
    def _condition_names_from(condition: str | None) -> set[str]:
        if condition is None:
            return set()
        return _names_in(_parse_expression(condition, "condition").body)

    def _check_constraints_satisfiable(self) -> None:
        if len(self._constraints) == 0:
            return

        try:
            raw_combos = self._enumerate_impl(max_combinations=10000, skip_constraints=True)
        except (EnumerationLimitExceeded, InvalidConditionalSearchSpaceError):
            # The space is not statically enumerable; structural checks are not available.
            return

        valid = [combo for combo in raw_combos if self._constraints_hold(combo)]
        if len(valid) > 0:
            return

        for constraint in self._constraints:
            if not any(not self._constraint_fails(constraint, combo) for combo in raw_combos):
                raise UnsatisfiableConstraintError(
                    constraint.expression,
                    "no assignment allowed by the other declared conditions satisfies it",
                )
        raise UnsatisfiableConstraintError(
            "; ".join(c.expression for c in self._constraints),
            "the constraints are jointly unsatisfiable over the declared space",
        )

    # ------------------------------------------------------------------ queries

    @property
    def version(self) -> str:
        """A content hash of the declaration, used to detect changes on study restore."""

        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @property
    def param_names(self) -> tuple[str, ...]:
        """Declared parameter names in declaration order."""

        return tuple(self._params)

    def distribution(self, name: str) -> BaseDistribution:
        """Return the declared distribution of ``name``."""

        self.validate()
        if name not in self._params:
            raise InvalidConditionalSearchSpaceError(
                f"The parameter `{name}` is not declared in the conditional search space."
            )
        return self._params[name].distribution

    def condition(self, name: str) -> str | None:
        """Return the effective condition expression of ``name``."""

        self.validate()
        return self._effective_conditions[name]

    def topological_order(self) -> tuple[str, ...]:
        """Return parameter names in an order where parents precede their children."""

        self.validate()
        return self._order

    def is_active(self, name: str, params: Mapping[str, Any]) -> bool:
        """Return whether ``name`` is active under the given partial assignment.

        A parameter with a condition that references an absent parameter is considered
        inactive.
        """

        self.validate()
        if name not in self._params:
            raise InvalidConditionalSearchSpaceError(
                f"The parameter `{name}` is not declared in the conditional search space."
            )
        if not self._enabled:
            return True
        condition = self._effective_conditions[name]
        if condition is None:
            return True
        if _missing_names(condition, params):
            return False
        try:
            return bool(_eval_expression(condition, params))
        except (TypeError, ValueError, ZeroDivisionError) as e:
            raise InvalidConditionalSearchSpaceError(
                f"The condition `{condition}` of the parameter `{name}` could not be evaluated."
            ) from e

    def active_params(self, params: Mapping[str, Any]) -> tuple[str, ...]:
        """Return the names of all active parameters under ``params``."""

        self.validate()
        return tuple(name for name in self._order if self.is_active(name, params))

    # ------------------------------------------------------------------ value validation

    def validate_values(
        self,
        params: Mapping[str, Any],
        *,
        allow_fixed_only: bool = False,
    ) -> None:
        """Validate a complete or partial parameter assignment.

        Args:
            params:
                Parameter values keyed by parameter name.
            allow_fixed_only:
                If :obj:`True`, only parameters present in ``params`` are checked. Parameters
                that are active but absent are allowed. This is used while a trial is being
                suggested incrementally.

        Raises:
            :class:`~optuna.exceptions.ConditionalConstraintViolation`:
                If the assignment violates the first declared condition.
            :class:`~optuna.exceptions.InvalidConditionalSearchSpaceError`:
                If ``params`` contains a parameter that is not declared.
        """

        violation = self.check_values(params, allow_fixed_only=allow_fixed_only)
        if violation is not None:
            raise violation.as_exception()

    def check_values(
        self,
        params: Mapping[str, Any],
        *,
        allow_fixed_only: bool = False,
    ) -> _Violation | None:
        """Return the first violated condition, or :obj:`None` when the assignment is valid."""

        self.validate()

        if not self._enabled:
            return None

        unknown = sorted(set(params) - set(self._params))
        if unknown:
            return _Violation(
                condition="<undeclared parameter>",
                param_name=unknown[0],
                reason="the parameter is not declared in the conditional search space",
            )

        env: dict[str, Any] = {}
        for name in self._order:
            condition = self._effective_conditions[name]
            if condition is not None and not _missing_names(condition, env):
                try:
                    active = bool(_eval_expression(condition, env))
                except (TypeError, ValueError, ZeroDivisionError):
                    raise InvalidConditionalSearchSpaceError(
                        f"The condition `{condition}` of the parameter `{name}` could not be "
                        "evaluated."
                    )
            else:
                active = condition is None

            if active:
                if name not in params:
                    if not allow_fixed_only:
                        return _Violation(
                            condition=condition or "<always>",
                            param_name=name,
                            reason="the parameter is active but missing",
                        )
                else:
                    value = params[name]
                    if not self._contains(self._params[name].distribution, value):
                        return _Violation(
                            condition=condition or "<always>",
                            param_name=name,
                            reason="the value is not contained in the declared distribution",
                        )
                    env[name] = value
            elif name in params:
                return _Violation(
                    condition=condition or "<always>",
                    param_name=name,
                    reason="the parameter is not active but was provided",
                )

            # Constraints are checked incrementally so the offending rule is reported before the
            # trial is created.
            violation = self._first_failed_constraint(env)
            if violation is not None:
                return violation

        return None

    def _first_failed_constraint(self, env: Mapping[str, Any]) -> _Violation | None:
        for constraint in self._constraints:
            if not self._constraint_holds(constraint, env):
                return _Violation(
                    condition=constraint.expression,
                    param_name=None,
                    reason="the declared constraint does not hold",
                )
        return None

    def _constraint_holds(self, constraint: _Constraint, env: Mapping[str, Any]) -> bool:
        """Evaluate a constraint lazily against ``env``.

        Evaluation is lazy so that a ternary such as ``a < b if cond else True`` only reads the
        branch it selects. A parameter that is inactive under ``env`` is absent; if evaluation
        reaches it the name is undefined and the constraint is not applicable yet, so it passes.
        This keeps incremental suggestion and partial assignments consistent: the constraint is
        enforced as soon as every name it actually reads has a value.
        """

        try:
            return bool(_eval_expression(constraint.expression, env))
        except KeyError:
            # A name that the selected branch does not need (or an active parameter that has not
            # been suggested yet). Once it is supplied the constraint is re-evaluated.
            return True
        except (TypeError, ValueError, ZeroDivisionError):
            return False

    def _constraint_fails(self, constraint: _Constraint, env: Mapping[str, Any]) -> bool:
        """Return whether ``constraint`` is applicable and false under ``env``."""

        try:
            return not bool(_eval_expression(constraint.expression, env))
        except KeyError:
            return False
        except (TypeError, ValueError, ZeroDivisionError):
            return True

    def _constraints_hold(self, env: Mapping[str, Any]) -> bool:
        return all(self._constraint_holds(c, env) for c in self._constraints)

    # ------------------------------------------------------------------ enumeration

    def enumerate(self, max_combinations: int = 100000) -> list[dict[str, Any]]:
        """Statically expand the declaration into legal parameter combinations.

        Only finite spaces can be enumerated.  Categorical parameters, integer parameters and
        float parameters with a ``step`` are enumerable; an unconstrained float parameter needs
        an explicit ``grid``.

        Args:
            max_combinations:
                The maximum number of combinations to enumerate.

        Returns:
            A list of assignments. Parameters on inactive branches are absent.

        Raises:
            :class:`~optuna.exceptions.EnumerationLimitExceeded`:
                If more than ``max_combinations`` combinations exist.
            :class:`~optuna.exceptions.InvalidConditionalSearchSpaceError`:
                If a parameter cannot be enumerated statically.
        """

        self.validate()
        return self._enumerate_impl(max_combinations=max_combinations, skip_constraints=False)

    def _enumerate_impl(
        self, max_combinations: int, skip_constraints: bool
    ) -> list[dict[str, Any]]:
        order = self._order
        results: list[dict[str, Any]] = []

        def recurse(index: int, env: dict[str, Any]) -> None:
            if len(results) > max_combinations:
                raise EnumerationLimitExceeded(max_combinations, len(results))
            if index == len(order):
                if skip_constraints or self._constraints_hold(env):
                    results.append(dict(env))
                return
            name = order[index]
            if not self.is_active(name, env):
                recurse(index + 1, env)
                return
            for value in self.candidates(name):
                env[name] = value
                if skip_constraints or self._partial_constraints_ok(env):
                    recurse(index + 1, env)
                del env[name]

        recurse(0, {})
        return results

    def _partial_constraints_ok(self, env: Mapping[str, Any]) -> bool:
        return self._first_failed_constraint(env) is None

    def candidates(self, name: str) -> tuple[Any, ...]:
        """Return the finite candidate values of ``name``.

        Raises:
            :class:`~optuna.exceptions.InvalidConditionalSearchSpaceError`:
                If ``name`` cannot be enumerated statically.
        """

        self.validate()
        if name not in self._params:
            raise InvalidConditionalSearchSpaceError(
                f"The parameter `{name}` is not declared in the conditional search space."
            )
        decl = self._params[name]
        if decl.grid is not None:
            return decl.grid

        distribution = decl.distribution
        if isinstance(distribution, CategoricalDistribution):
            return tuple(distribution.choices)
        if isinstance(distribution, IntDistribution):
            return tuple(
                range(
                    distribution.low,
                    distribution.high + 1,
                    distribution.step,
                )
            )
        if isinstance(distribution, FloatDistribution) and distribution.step is not None:
            return _float_candidates(distribution.low, distribution.high, distribution.step)
        raise InvalidConditionalSearchSpaceError(
            f"The parameter `{name}` cannot be enumerated statically: its distribution has an "
            "infinite domain. Provide `grid` or a `step` to use static enumeration."
        )

    # ------------------------------------------------------------------ constants

    def constants(self, name: str) -> tuple[Any, ...]:
        """Return the candidate values of ``name``.

        This is a convenience alias for :meth:`candidates` used by samplers that filter
        candidate values before suggesting.
        """

        self.validate()
        return self.candidates(name)

    # ------------------------------------------------------------------ serialization

    def to_dict(self) -> dict[str, Any]:
        """Serialize the declaration to a JSON-compatible dictionary."""

        params = []
        for decl in self._params.values():
            entry: dict[str, Any] = {
                "name": decl.name,
                "distribution": json.loads(distribution_to_json(decl.distribution)),
                "condition": decl.condition,
            }
            if decl.grid is not None:
                entry["grid"] = list(decl.grid)
            params.append(entry)

        groups = [
            {
                "name": group.name,
                "parent": group.parent,
                "branches": [
                    {"value": value, "params": list(params)} for value, params in group.branches
                ],
            }
            for group in self._branch_groups
        ]
        return {
            "schema_version": SCHEMA_VERSION,
            "name": self.name,
            "enabled": self._enabled,
            "params": params,
            "branch_groups": groups,
            "constraints": [
                {"name": constraint.name, "expression": constraint.expression}
                for constraint in self._constraints
            ],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ConditionalSearchSpace:
        """Deserialize a declaration produced by :meth:`to_dict`."""

        if data.get("schema_version") != SCHEMA_VERSION:
            raise ConditionalSearchSpaceVersionError(
                expected=str(SCHEMA_VERSION), actual=str(data.get("schema_version"))
            )
        space = cls(name=data.get("name"), enabled=data.get("enabled", True))
        for entry in data["params"]:
            distribution = json_to_distribution(json.dumps(entry["distribution"]))
            space.add_param(
                entry["name"],
                distribution,
                condition=entry.get("condition"),
                grid=entry.get("grid"),
            )
        for group in data.get("branch_groups", []):
            space.add_branch_group(
                group["name"],
                group["parent"],
                {branch["value"]: branch["params"] for branch in group["branches"]},
            )
        for constraint in data.get("constraints", []):
            space.add_constraint(constraint["expression"], name=constraint["name"])
        space.validate()
        return space

    def to_json(self) -> str:
        """Serialize the declaration to a JSON string."""

        return json.dumps(self.to_dict(), sort_keys=True, allow_nan=True)

    @classmethod
    def from_json(cls, payload: str) -> ConditionalSearchSpace:
        """Deserialize a declaration produced by :meth:`to_json`."""

        return cls.from_dict(json.loads(payload))

    def diff(self, other: ConditionalSearchSpace) -> str:
        """Return a description of the differences from ``other``.

        The description references parameter names and condition expressions only; it does not
        include parameter values.
        """

        lines: list[str] = []
        if self.schema_version != other.schema_version:
            lines.append(f"schema version: {other.schema_version} -> {self.schema_version}")
        if self._enabled != other._enabled:
            lines.append(f"enabled: {other._enabled} -> {self._enabled}")
        own = set(self._params)
        theirs = set(other._params)
        for name in sorted(own - theirs):
            lines.append(f"parameter `{name}`: added")
        for name in sorted(theirs - own):
            lines.append(f"parameter `{name}`: removed")
        for name in sorted(own & theirs):
            left = self._params[name]
            right = other._params[name]
            if left.distribution != right.distribution:
                lines.append(f"parameter `{name}`: distribution changed")
            if left.condition != right.condition:
                lines.append(f"parameter `{name}`: condition changed")
            if left.grid != right.grid:
                lines.append(f"parameter `{name}`: grid changed")

        own_groups = {group.name for group in self._branch_groups}
        other_groups = {group.name for group in other._branch_groups}
        for name in sorted(own_groups - other_groups):
            lines.append(f"branch group `{name}`: added")
        for name in sorted(other_groups - own_groups):
            lines.append(f"branch group `{name}`: removed")

        own_constraints = {c.name: c.expression for c in self._constraints}
        other_constraints = {c.name: c.expression for c in other._constraints}
        for name in sorted(set(own_constraints) | set(other_constraints)):
            if own_constraints.get(name) != other_constraints.get(name):
                lines.append(f"constraint `{name}`: changed")
        return "\n".join(lines) if lines else "(no declarative difference)"

    @property
    def schema_version(self) -> int:
        """The serialization schema version."""

        return SCHEMA_VERSION

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _contains(distribution: BaseDistribution, value: Any) -> bool:
        try:
            internal = distribution.to_internal_repr(value)
        except (ValueError, TypeError):
            return False
        return distribution._contains(internal)

    @staticmethod
    def _same_value(left: Any, right: Any) -> bool:
        if isinstance(left, float) and isinstance(right, float):
            if math.isnan(left) and math.isnan(right):
                return True
        try:
            return bool(left == right)
        except Exception:
            return False


__all__ = ["ConditionalSearchSpace", "SCHEMA_VERSION"]
