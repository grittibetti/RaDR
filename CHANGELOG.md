# Changelog

## 0.2.0

- Renamed distribution/import to radr and estimators to RaDR*/ARaDR*.
- Aligned adaptive cosine normalization, KA/DV/leverage scores, top-k defaults,
  and no-intercept ridge with the paper.
- Added discoverable score modules and external module/function plugins.
- Added one-hot, ordinal, and strict numeric conversion with frozen training schema.
- Fixed CLI leakage: split bag IDs before fitting preprocessing; retain IDs and
  save the fitted transformer for new input.
- Refit the complete estimator within tuning folds; make tuning opt-in.
- Fixed sklearn mixin order, finite-input validation, string-label metrics,
  cos/sin truncation, and reproducible classifier seeds.
- Replaced pickle-based NPZ archives with numeric arrays plus JSON.
- Added regression tests, build checks, CI definitions, API and contribution docs.

### Migrating from RFDR

- Change `from rfdr import ...` to `from radr import ...`.
- Replace `RFDRClassifier` / `RFDRRegressor` with `RaDRClassifier` / `RaDRRegressor`,
  and `ARFDRClassifier` / `ARFDRRegressor` with `ARaDRClassifier` / `ARaDRRegressor`.
- Use `radr-preprocess` for command-line preprocessing.
- Regenerate bag archives with `save_npz` and retrain saved models. The archive
  format, adaptive normalization, and score formulas changed in version 0.2.
- Set `tune=True` to enable internal cross-validation. The default is `False`.
- Leverage selection uses top-k by default. Select `selection_strategy="sample"`
  explicitly for sampling. `eerf` remains an alias for `ka`.

Rerun experiments with the updated version when comparing results across releases.
