# Declarative conditional search space: acceptance report

Optuna version: `5.2.0.dev`

## 1. Enumeration of three nesting levels and two mutually exclusive groups

- Declared parameters: kind, model, lr, depth, criterion, max_features, smoothing, window
- Statically enumerated legal combinations: **18**
- Hand-counted combinations: **18**
- Match: PASS
- Serialized declaration version: `6fff4064c5eb944b68e51171112f5a8e1ea720af3d13e2bd302e564f8ce13736`

Sample of expanded combinations (truncated):

```
{'kind': 'ml', 'model': 'linear', 'lr': 0.001}
{'kind': 'ml', 'model': 'linear', 'lr': 0.01}
{'kind': 'ml', 'model': 'linear', 'lr': 0.1}
{'kind': 'ml', 'model': 'tree', 'depth': 2, 'criterion': 'logloss', 'max_features': 1}
{'kind': 'ml', 'model': 'tree', 'depth': 2, 'criterion': 'logloss', 'max_features': 2}
{'kind': 'ml', 'model': 'tree', 'depth': 2, 'criterion': 'gini', 'smoothing': 1}
{'kind': 'ml', 'model': 'tree', 'depth': 2, 'criterion': 'gini', 'smoothing': 2}
{'kind': 'ml', 'model': 'tree', 'depth': 3, 'criterion': 'logloss', 'max_features': 1}
... (10 more)
```

## 2. Invalid combinations never create trials, and rejections are counted separately

- Invalid assignments presented to `validate_values`: **4**
- Rejected by the declaration: **4**
- Rejected at `enqueue_trial` before trial creation: **3**
- Trials created by the rejected enqueues: **0**

First failing condition per rejected assignment:

```
The declared condition `model == 'linear'` is not satisfied (parameter `lr`): the parameter is active but missing.
The declared condition `model == 'linear'` is not satisfied (parameter `lr`): the parameter is not active but was provided.
The declared condition `depth * batch > 16 if model == 'tree' else True` is not satisfied: the declared constraint does not hold.
The declared condition `model == 'tree'` is not satisfied (parameter `batch`): the parameter is active but missing.
```

## 3. A sampler that does not support dynamic spaces still yields legal suggestions

- Inner sampler: `RandomSampler` (reports no conditional support)
- Trials run: **24**, illegal parameter sets: **0**
- Total rejection counter: **0**

Degradation notice:

```
Conditional sampling degraded: `RandomSampler` does not understand the declaration, so the declared conditions are enforced by projection after the proposal instead of during it; illegal proposals are replaced with the nearest declared-legal candidate.
```

## 4. Declaration versioning on restore

- Restored version matches stored version: PASS
- A changed declaration is refused; reported difference:

```
The conditional search space declaration of this study does not match the current one (stored version 630b2df2fd07bbbe2350024b903465740a6ec684bdd13171d737bdfd2717073b, current version 6ff2ddf368c10812f47065beae37cab26e02cccd59b328345cd3c9cd9817e49f).
Difference:
parameter `z`: added
```

## 5. Behavior without a declaration, or with a single branch

- Legacy study trials: **6**, all completed: True
- Legacy study rejections: **{}**
- Single-branch declaration trials: **1**

## 6. Existing sampler and storage tests

| Suite | Result | Summary |
| --- | --- | --- |
| `tests/search_space_tests` | PASS | `33 passed, 2 warnings in 1.07s` |
| `tests/samplers_tests/test_grid.py` | PASS | `16 passed in 1.78s` |
| `tests/samplers_tests/test_brute_force.py` | PASS | `24 passed, 2 warnings in 0.22s` |
| `tests/trial_tests` | PASS | `248 passed, 27 deselected in 26.48s` |
| `tests/study_tests/test_study.py` | PASS | `602 passed, 7 skipped, 106 deselected in 81.31s (0:01:21)` |

All executed suites passed.

Note: `fakeredis` in this environment lacks the Lua `eval` command and the optional `cmaes` package is not installed, so redis- and cmaes-parametrized cases are excluded from the runs above. Those failures are environmental and unrelated to this change.

