import numpy as np
import pandas as pd

from radr import ARaDRClassifier, BagPreprocessor, RaDRRegressor


def _toy_classification():
    rng = np.random.default_rng(1)
    X = [rng.normal(loc=i % 2, size=(8 + i % 3, 3)) for i in range(40)]
    y = np.asarray([i % 2 for i in range(40)])
    return X, y


def test_aradr_classifier_smoke():
    X, y = _toy_classification()
    clf = ARaDRClassifier(n_features=16, pool_multiplier=2, tune=False, random_state=0)
    clf.fit(X[:30], y[:30])
    pred = clf.predict(X[30:])
    assert pred.shape == (10,)
    assert "accuracy" in clf.test(X[30:], y[30:])


def test_radr_regressor_smoke():
    rng = np.random.default_rng(2)
    X = [rng.normal(size=(7 + i % 4, 2)) for i in range(35)]
    y = np.asarray([bag[:, 0].mean() for bag in X])
    reg = RaDRRegressor(n_features=24, tune=False, random_state=0)
    reg.fit(X[:25], y[:25])
    assert reg.predict(X[25:]).shape == (10,)


def test_dataframe_preprocessor_smoke():
    df = pd.DataFrame(
        {
            "bag": [0, 0, 1, 1, 2, 2],
            "x1": [0.0, 0.1, 1.0, 1.1, 0.2, 0.3],
            "x2": [1.0, 1.1, 2.0, 2.1, 1.2, 1.3],
            "y": [0, 0, 1, 1, 0, 0],
        }
    )
    prep = BagPreprocessor(bag_id_col="bag", feature_cols=["x1", "x2"], target_col="y")
    X, y = prep.fit_transform(df)
    assert len(X) == 3
    assert y.tolist() == [0, 1, 0]
