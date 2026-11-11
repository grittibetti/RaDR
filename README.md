# RaDR: Random Feature Distribution Regression

This is the Python code package for the paper
**[Scalable and Adaptive Distribution Regression via Random Features](https://arxiv.org/abs/2610.24892)**
by Yinsong Wang, Xiao Liu, and Yu Ding (2026).

The package implements Random Feature Distribution Regression (RaDR) and its
adaptive variants (A-RaDR). RaDR learns one response per **bag of instances**
using averaged random Fourier features. A-RaDR scores a larger candidate pool
and retains features according to distribution variance, kernel alignment, or
leverage scores.

- `RaDRClassifier` / `RaDRRegressor`: ordinary random-feature bag embeddings.
- `ARaDRClassifier` / `ARaDRRegressor`: adaptive embeddings using distribution
  variance (`dv`), kernel alignment (`ka`), leverage (`leverage`), or a custom score.
- `BagPreprocessor`: row-level tables to numeric bags, with categorical encoding,
  missing-value handling, scaling, and bag-size limits.

Python 3.10+; NumPy, pandas, scikit-learn, and joblib. Runs on the CPU.
See [compatibility and testing](docs/RELEASE_REVIEW.md) for tested environments
and computational requirements.

## Install from source

Run the following command from the package directory containing `pyproject.toml`:

```bash
python -m pip install .
```

In the distribution-regression research repository, the package is located in
`radr_package`; from that repository's root, use `python -m pip install ./radr_package`.

For the optional PyTorch data-loading helpers:

```bash
python -m pip install ".[torch]"
```

Both RaDR and A-RaDR estimators are included in the `radr` distribution. See the
[contribution guide](CONTRIBUTING.md) for development installation and testing.

## Train and predict

Each bag is a nonempty finite numeric array shaped `(n_instances, n_input_features)`.
Bag sizes may differ; feature count and column meaning must agree across bags.
Supply a list of bags (or a rectangular 3D NumPy array) and one target per bag.
Class labels may be strings or numbers; regression targets must be finite numbers.

```python
import numpy as np
from radr import ARaDRClassifier

rng = np.random.default_rng(0)
X = [rng.normal(loc=i % 2, size=(20 + i % 3, 4)) for i in range(100)]
y = np.arange(100) % 2
model = ARaDRClassifier(
    n_features=128,
    pool_multiplier=4,
    selector="dv",
    random_state=0,
)
model.fit(X[:80], y[:80])
print(model.predict(X[80:]))
print(model.test(X[80:], y[80:]))
```

For regression, use `RaDRRegressor` or `ARaDRRegressor` with the same interface.
`downstream="linear"` uses linear SVM classification or ridge regression;
`downstream="kernel"` uses a bag-level RBF kernel and SVM/kernel ridge.
A cloneable sklearn estimator can also be passed as `downstream`.

`get_params`, `set_params`, cloning, and bag-level sklearn cross-validation are
supported. `tune=False` is the fast default. Set `tune=True` for internal CV over
`C_grid` / `alpha_grid` and, for RBF downstream models, `kernel_gamma_grid`.
Every fold refits random features, bandwidths, and adaptive scores on its training
bags. Preprocessing must also be fitted independently within any external CV loop.

## Categorical and row-level data

Split by **bag ID** before fitting preprocessing so every bag stays within one split:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from radr import BagPreprocessor, ARaDRRegressor

df = pd.read_csv("instances.csv")  # bag_id, temperature, material, target
train_ids, test_ids = train_test_split(
    df["bag_id"].unique(),
    test_size=0.2,
    random_state=0,
)
train = df[df.bag_id.isin(train_ids)]
test = df[df.bag_id.isin(test_ids)]
prep = BagPreprocessor(
    bag_id_col="bag_id",
    target_col="target",
    task="regression",
    feature_cols=["temperature", "material"],
    categorical_cols=["material"],
    categorical_encoding="onehot",
    handle_unknown="ignore",
    scaler="standard",
    missing="error",
)
X_train, y_train = prep.fit_transform(train)
X_test, y_test = prep.transform(test)
model = ARaDRRegressor(n_features=64, random_state=0).fit(X_train, y_train)
print(model.test(X_test, y_test))
# Prediction tables can omit the target column:
X_new = prep.transform(test.drop(columns="target"))
print(model.predict(X_new))
```

Encoding options:

| `categorical_encoding` | Behavior | Unseen categories with `handle_unknown="ignore"` |
| --- | --- | --- |
| `onehot` (default) | Dense indicator columns; suitable for nominal categories | All-zero indicator block before scaling |
| `ordinal` | Learned integer codes | `-1` before scaling |
| `numeric` | Strict conversion of numeric strings to numbers | Nonnumeric text raises an error |

Categorical columns are inferred from nonnumeric and boolean dtypes, or supplied
explicitly (including integer-coded categories). Ordinal codes introduce artificial
distances; one-hot is usually a better nominal representation. One-hot output is
dense and can be large for high-cardinality columns. Both encoders learn a fixed
vocabulary from training data; `handle_unknown="error"` rejects unseen values.

`missing="drop"` removes rows with missing features; `"fill"` uses `fill_value`
for numeric values and a distinct categorical missing token; `"error"` rejects
missing values. Infinite features and missing IDs/targets are rejected. Bags below
`min_bag_size` are removed with their targets and IDs. Inspect retained IDs with
`return_metadata=True`. Scaling is instance-weighted across retained training rows
and applies to all encoded columns, including one-hot columns; use `scaler=None`
to retain raw indicators. See [API reference](docs/API.md) for return types and options.

## Command line

```bash
radr-preprocess --input-csv instances.csv --output-dir prepared --task regression --bag-id-col bag_id --target-col target --feature-cols temperature,material --categorical-cols material --categorical-encoding onehot
```

The CLI splits bags first, fits encoders/scaling on training rows, and writes
`train_radr.npz`, `test_radr.npz`, and `preprocessor.joblib`. Original bag IDs are
preserved. `--stratify` supports classification. Omit `--target-col` for unlabeled
preprocessing. Load archives with `radr.load_npz(path)`; these archives store numeric
arrays and JSON. Load the saved transformer with `joblib.load` from trusted sources.
Save a fitted estimator with `joblib.dump` in the same way; record dependency
versions and restore the same environment when loading saved models.

## Paper conventions and extension points

The paper calls the algorithms **RaDR** and **A-RaDR**. Python class names are
`RaDR*` and `ARaDR*`. The default Gaussian cosine map approximates
`exp(-gamma * ||x-y||**2 / 2)`, using frequencies with covariance `gamma I`.
This gamma convention differs by a factor of two from sklearn's RBF kernel.
Adaptive cosine embeddings use final-budget normalization `sqrt(2 / D)`.
All paper scores default to top-D selection. Ridge uses no intercept, with sklearn
`alpha = n_train_bags * lambda` for the paper's regularization parameter.

[Contributing](CONTRIBUTING.md) explains how to add an adaptive variant by adding
one automatically discovered score file. Callables and external
`module:function` scores are supported too. [API](docs/API.md) describes feature
samplers, kernels, scoring, tuning, and fitted attributes.

## Citation

Please cite the following paper when using RaDR or A-RaDR in research,
publications, or other work based on this package:

Yinsong Wang, Xiao Liu, and Yu Ding (2026).
*Scalable and Adaptive Distribution Regression via Random Features*.
arXiv:2610.24892. [doi:10.48550/arXiv.2610.24892](https://doi.org/10.48550/arXiv.2610.24892).

```bibtex
@misc{wang2026radr,
  title         = {Scalable and Adaptive Distribution Regression via Random Features},
  author        = {Wang, Yinsong and Liu, Xiao and Ding, Yu},
  year          = {2026},
  eprint        = {2610.24892},
  archivePrefix = {arXiv},
  primaryClass  = {stat.ML},
  doi           = {10.48550/arXiv.2610.24892},
  url           = {https://arxiv.org/abs/2610.24892}
}
```

The citation is also available in [CITATION.bib](CITATION.bib).

## Documentation and contributions

- [API reference](docs/API.md): estimator parameters, preprocessing, and storage.
- [Contribution guide](CONTRIBUTING.md): add adaptive variants with a score module.
- [Compatibility and testing](docs/RELEASE_REVIEW.md): environments, tests, and resource use.
- [Changelog](CHANGELOG.md): version history and migration instructions.

## License

RaDR is distributed under the [MIT License](LICENSE).
