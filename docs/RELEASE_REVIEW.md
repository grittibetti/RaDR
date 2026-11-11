# Compatibility and testing

RaDR supports Python 3.10 and later and runs on the CPU. Core dependencies are
NumPy, pandas, scikit-learn, and joblib. Install the `torch` extra to use the
PyTorch dataset and data-loader helpers.

## Tested environments

The version 0.2.0 test suite passed in these Windows environments:

| Python | NumPy | pandas | scikit-learn | SciPy | Tests |
| --- | --- | --- | --- | --- | --- |
| 3.10.19 | 1.23.5 | 1.5.3 | 1.2.2 | 1.10.1 | 49 passed |
| 3.14.3 | 2.5.4 | 3.1.1 | 1.9.1 | 1.18.1 | 49 passed |

These results were recorded on November 10, 2026. The same 49 tests passed against
an installed wheel in a separate environment. Source and wheel builds, Twine
metadata checks, Ruff checks, and the quickstart example also passed.

The CI configuration targets Python 3.10-3.14 on Windows, Linux, and macOS,
with additional jobs for minimum dependencies and distribution builds. The table
above records completed local checks; hosted CI results provide the status of
the broader platform matrix. Optional PyTorch helpers have separate validation
requirements and are outside the core test suite.

## Test coverage

Tests exercise the following behavior:

- Paper score equations, adaptive feature normalization, and ridge coefficients.
- Classification and regression with linear and kernel downstream models.
- Estimator cloning, cross-validation, reproducibility, and model serialization.
- Adaptive score discovery, external score functions, and score validation.
- Categorical encoding, unseen categories, missing values, and feature schemas.
- Training-data isolation for preprocessing and adaptive selection.
- Bag identifiers, command-line preprocessing, and numeric-array/JSON archives.

## Running checks

Install the development dependencies from the package directory:

```bash
python -m pip install -e ".[dev]"
python -m pytest
ruff check .
ruff format --check .
python examples/quickstart.py
```

Build and validate distributions with:

```bash
python -m build
python -m twine check dist/*
```

To verify installation, install the generated wheel in a fresh environment and
run the example and tests from outside the source directory.

## Computational requirements

Training runs in memory on a single machine. Candidate embeddings require
O(n*M) memory for n bags and M candidate features. Each bag also requires an
instance-by-feature matrix during transformation. Kernel downstream models
construct an n-by-n Gram matrix.

One-hot encoding produces dense arrays, and preprocessing concatenates training
instances to fit scaling statistics. For large datasets, reduce the candidate
pool or bag sizes, consolidate categorical levels, or use linear downstream
models. See the [API reference](API.md#scaling-limits) for details.

## Model and data portability

Save numeric bags with `save_npz` and load them with `load_npz`. The archive stores
numeric arrays and JSON metadata. For fitted estimators and preprocessors, use
joblib and retain the Python and dependency versions used during training.
Load joblib files from trusted sources.

Version 0.2 changes estimator names, adaptive score conventions, and the bag
archive format. Follow the [migration instructions](../CHANGELOG.md#migrating-from-rfdr)
when updating models or data prepared with the earlier RFDR implementation.

## Reporting issues

Include the package version, Python version, operating system, dependency versions,
and a minimal reproducible example. For performance issues, include the number
of bags, bag sizes, input dimensions, feature budget, and candidate-pool size.
