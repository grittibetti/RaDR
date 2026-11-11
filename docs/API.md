# API reference

## Estimators

`RaDRClassifier`, `RaDRRegressor`, `ARaDRClassifier`, and `ARaDRRegressor`
share the constructor in `radr/estimators.py`.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `n_features` | 256 | Cosine output budget D; positive integer |
| `rf_gamma` | `"median"` | Positive frequency variance; `None` also selects median heuristic |
| `random_feature_distribution` | `"gaussian"` | Gaussian, laplace, cauchy, student_t, or callable |
| `feature_map` | `"cosine"` | Random-bias cosine; `"cos_sin"` returns 2D coordinates, RaDR only |
| `pool_multiplier`, `pool_size` | 8, None | A-RaDR pool M; explicit size overrides multiplier; M >= D |
| `selector` | `"dv"` | Built-in score name, callable, or `module:function` |
| `selection_strategy` | `"auto"` | Top-k except for random baseline; explicit `topk` / `sample` |
| `leverage_ridge` | None | Positive scoring regularizer nu; defaults to 1/n |
| `downstream` | `"linear"` | `linear`, `kernel`, or cloneable sklearn estimator |
| `downstream_kernel` | `"rbf"` | `rbf`, `linear`, or callable `(X, Y)` returning Gram matrix |
| `kernel_gamma` | `"median"` | Positive RBF bandwidth; None also uses median heuristic |
| `kernel_gamma_scale` | 1 | Multiplier on explicit/estimated downstream gamma |
| `C`, `alpha` | 1, 0.001 | Classifier regularization / ridge penalty |
| `tune`, `cv` | False, 3 | Full-estimator CV; integer folds, limited by available bags/classes |
| `C_grid` | (0.1, 1, 10, 100) | Classifier tuning candidates |
| `alpha_grid` | (0.0001, 0.001, 0.01, 0.1, 1) | Regression tuning candidates |
| `kernel_gamma_grid` | (0.25, 0.5, 1, 2, 4) | Multipliers applied to each fold's bandwidth |
| `max_iter` | 5000 | Linear SVM iteration limit |
| `probability` | False | Enable legacy SVC probability calibration for kernel classifier |
| `random_state` | None | Integer seed for reproducibility |
| `dtype` | `"float64"` | Floating-point computation dtype |

`fit(X, y)` returns self; `transform(X)` returns bag embeddings;
`predict(X)` returns one prediction per bag. `score` is accuracy or R-squared.
`test(X, y)` returns accuracy/F1 or MSE/RMSE/MAE/R-squared.
Binary F1 treats the largest sorted class label as positive; multiclass uses macro F1.
`decision_function` and `predict_proba` require support from the downstream learner.
For new probabilistic applications, a custom probabilistic downstream estimator
(e.g. sklearn LogisticRegression) avoids the legacy SVC probability API, which
is deprecated in sklearn 1.9. Supplying a custom learner bypasses built-in
regularization grids; tune its parameters using external GridSearchCV.

Fitted attributes: `random_features_`, `feature_indices_`, `feature_scale_`,
`selector_` (scores, indices, strategy, statistics; None for RaDR),
`train_embeddings_`, `downstream_estimator_`, `rf_gamma_`, `kernel_gamma_`,
`classes_` (classification), `n_features_in_`, `params_`, `fit_summary_`.
Tuned fits also expose `best_params_` and `cv_results_`.

The median heuristic uses inverse median positive squared distance, sampling at
most 2048 points. Kernel gamma uses the same rule on bag embeddings. The Gaussian
kernel is `exp(-gamma * squared_distance / 2)`. Other frequency distributions
change the implied kernel and are extensions beyond the paper's Gaussian setup.
A custom sampler takes `(input_dim, n_features, gamma, rng)` and returns finite
frequencies shaped `(input_dim, n_features)`. `RandomFourierFeatures` also exposes
`fit`, `transform`, `random_weights_`, and `random_offsets_` directly.

`downstream="linear"` ridge fits without an intercept to match the paper.
Use `downstream=Ridge(fit_intercept=True)` if an intercept is desired.
`alpha=n*lambda` matches the ridge equation in the
[paper](https://arxiv.org/abs/2610.24892).

## BagPreprocessor

Use `fit` on training input only and reuse `transform` on evaluation input.
`BagPreprocessor` groups rows into bags, changing the sample count, and returns
tuples for labeled input. Integrate it through an explicit preprocessing step.
For CV with tables, loop over bag-ID folds and fit a fresh preprocessor per fold.
Internal estimator tuning operates on the resulting numeric bags; manage
preprocessor fitting in the bag-ID CV loop.

- Inputs: a pandas DataFrame with `bag_id_col`, or a list/3D array of numeric bags.
  Categorical encoding is available for DataFrames; convert mixed-type arrays to
  a DataFrame first. Keep the same input kind between fit and transform.
- `feature_cols=None` selects all columns except ID, target and sort columns;
  inferred feature names are frozen at fit. Additional columns at transform are
  ignored; missing training feature columns raise an error.
- `target_col` supplies repeated bag targets. `label_mode="bag"` requires one
  unique label per bag. `"aggregate"` uses `aggregation`: first, classification
  mode/majority/any_positive/all_positive/proportion, or regression
  mean/median/min/max. Binary aggregations expect numeric labels; proportion is
  a continuous response and should be used with a regressor.
- `bag_label_map` supplies bag ID -> target labels instead of a target column.
  For array input, pass `y` separately. For DataFrames, supply labels through
  `target_col` or `bag_label_map`.
- `sort_cols` determines instance ordering before subsampling. Bags retain
  first-seen ID order after sorting. `min_bag_size=1`, `max_bag_size=None`;
  `subsample="first"`, `"last"`, or `"random"` with `random_state`.
- `scaler="standard"`, `"minmax"`, or None. Numeric missing `fill_value=0`.
  Categorical options and missing policies are detailed in the README.
- `fit_transform(X, y=None)` and `transform` return `(bags, targets)` when labels
  exist, otherwise just `bags`. `return_metadata=True` always returns
  `(bags, targets_or_None, metadata)`, including retained `bag_ids`.
- `get_feature_names_out()` returns encoded DataFrame feature names.
  `encoder_` stores categories and `scaler_` stores fitted scaling statistics.

`dataframe_to_bags` is a low-level numeric-only grouping helper with the same
ID/feature/target/aggregation arguments, returning `BagDataset`. Use
`BagPreprocessor` to fit encoders and scalers. `train_test_split_bags(bags, targets, bag_ids=None, ...)`
splits already numeric bags and returns two BagDataset objects. For row-level
preprocessing use the ID-first split shown in the README/CLI instead.

## Storage and optional torch

`BagDataset` holds `bags`, optional `targets`, `bag_ids`, and `metadata`, with
`input_dim`, `bag_sizes`, and `len(dataset)` accessors.
`save_npz(path, bags, targets=None, metadata=None, bag_ids=None)` stores concatenated
numeric rows, offsets, and JSON values; metadata/IDs must be JSON-serializable.
`load_npz(path)` reads this numeric-array/JSON format and returns `BagDataset`.
Regenerate legacy object-array archives with `save_npz` before loading them.

`dataset.to_torch()`, `torch_dataloader(dataset, batch_size=32, shuffle=True)`, and
`torch_collate_variable_bags` require the torch extra. Collation pads instances and
returns `x`, boolean `mask`, `length`, and available labels/IDs. Convert string
labels to numeric codes before using torch. Use a single-worker loader on Windows.
See [compatibility and testing](RELEASE_REVIEW.md) for test coverage.

## Scaling limits

Candidate embeddings need O(n*M) memory. Each bag is transformed separately, but
a large single bag still needs its instance-by-feature matrix. Leverage uses a
thin SVD, avoiding a full M-by-M inverse. Kernel downstream models construct an
n-by-n Gram matrix. One-hot preprocessing is dense and currently concatenates
training instances for scaling. For large data, use a smaller pool, bag subsampling,
fewer categorical levels, and linear downstream models. Training runs in memory
on a single machine.
