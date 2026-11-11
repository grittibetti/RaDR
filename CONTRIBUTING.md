# Contributing an adaptive score

Contributions can add adaptive score functions, improve preprocessing, or extend
the tests and documentation. To set up a development environment, run
`python -m pip install -e ".[dev]"` from the package directory.

## Add one file

Create `radr/scores/my_score.py`:

```python
import numpy as np


def score(F, y, task):
    """Rank candidates by the mean absolute centered bag embedding."""
    centered = F - F.mean(axis=0, keepdims=True)
    return np.mean(np.abs(centered), axis=0)
```

It is immediately discoverable as:

```python
from radr import ARaDRClassifier, available_selectors

assert "my_score" in available_selectors()
model = ARaDRClassifier(selector="my_score", n_features=64, random_state=0)
```

Score discovery automatically connects the new module to the estimators. Filenames must
be valid lowercase Python module names without a leading underscore. Restart the
Python interpreter after adding files. The package build automatically includes
new modules under `radr.scores`. Scorers are imported lazily when selected.

## Score contract

- `F`: finite float array `(n_training_bags, candidate_pool_size)`, containing
  candidate bag means normalized by `sqrt(pool_size)` (cosine map includes sqrt(2)).
- `y`: one original target per training bag, shape `(n_training_bags,)`.
  Classification labels can be strings; encode them inside a supervised score.
- `task`: `"classification"` or `"regression"`.
- Return one finite real scalar per candidate, shape `(candidate_pool_size,)`.
  Larger scores rank higher. Negate a minimization criterion for top-k selection.
- Treat F/y as read-only and compute scores from the supplied training data.
  Scores are recomputed on each training fold during tuning. Use deterministic
  scoring where possible.
- Top-k accepts negative scores and breaks ties in original candidate order.
  `selection_strategy="sample"` requires nonnegative scores with at least D
  positive entries (all-zero scores fall back to uniform); it samples without
  replacement and without importance reweighting. Sampling provides an experimental
  selection strategy. Use top-k to maximize the paper's score objective.

The built-in modules `dv`, `ka`, and `leverage` implement the
[paper's](https://arxiv.org/abs/2610.24892) equations (33), (39),
(41), and (44). The `random` baseline returns equal scores and uses uniform
sampling with `selection_strategy="auto"`. New scores use top-k under auto.
`leverage_ridge` applies only to the built-in leverage selector and its aliases.
For a parameterized custom score, use a top-level function and `functools.partial`.
For persistence, functions must remain importable from the same module path.

## External score modules

Place the same function in an importable module `my_project/scoring.py`:

```python
model = ARaDRClassifier(selector="my_project.scoring:score")
# Or pass the function directly:
from my_project.scoring import score

model = ARaDRClassifier(selector=score)
```

Module references execute Python code; use modules you trust. Declare optional dependencies
in the contribution's documentation or an optional extra, and import them inside
the score module when needed.

## Validate a contribution

Add tests for formula correctness, output shape/finiteness, degenerate inputs,
classification/regression where applicable, deterministic selection, and prediction.
`tests/test_contracts.py::test_file_only_plugin` verifies discovery without a
registry edit. Include a short mathematical explanation and applicable citation
in your module docstring, and document computational/memory complexity.

```bash
python -m pytest
ruff check .
ruff format --check .
python -m build
python -m twine check dist/*
```

Keep contributions focused on source code, tests, and documentation. Keep datasets,
model pickles, environments, caches, and build products in local storage.
Submit a pull request explaining the criterion and its tests.
Please include a reproducible minimal example and dependency versions in bug reports.
