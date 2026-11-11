import importlib
import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone, is_classifier, is_regressor
from sklearn.model_selection import cross_val_score
from sklearn.exceptions import NotFittedError

from radr import (
    ARaDRClassifier,
    ARaDRRegressor,
    RaDRClassifier,
    RaDRRegressor,
    BagPreprocessor,
    available_selectors,
    save_npz,
    load_npz,
)
from radr.selectors import (
    resolve_selector,
    distribution_variance_scores,
    label_alignment_scores,
    empirical_leverage_scores,
)
from radr.utils import ensure_bag_list


def toy():
    rng = np.random.default_rng(4)
    bags = [rng.normal(i % 2, 1, (5 + i % 3, 2)) for i in range(18)]
    return bags, np.arange(18) % 2


@pytest.mark.parametrize("cls", [RaDRClassifier, ARaDRClassifier, RaDRRegressor, ARaDRRegressor])
@pytest.mark.parametrize("downstream", ["linear", "kernel"])
def test_estimator_contract(cls, downstream):
    X, y = toy()
    model = cls(n_features=8, pool_multiplier=2, downstream=downstream, random_state=3)
    assert clone(model).get_params() == model.get_params()
    assert is_classifier(model) == ("Classifier" in cls.__name__)
    assert is_regressor(model) == ("Regressor" in cls.__name__)
    with pytest.raises(NotFittedError):
        model.predict(X)
    model.fit(X, y)
    np.testing.assert_allclose(model.transform(X), model.train_embeddings_)
    np.testing.assert_allclose(pickle.loads(pickle.dumps(model)).predict(X), model.predict(X))
    np.testing.assert_allclose(clone(model).fit(X, y).predict(X), model.predict(X))
    assert np.isfinite(cross_val_score(clone(model), X, y, cv=2)).all()
    with pytest.raises(ValueError):
        model.predict([np.ones((4, 3))])


@pytest.mark.parametrize("selector", ["dv", "ka", "leverage", "random", "eerf"])
def test_selectors(selector):
    X, y = toy()
    model = ARaDRClassifier(n_features=5, pool_size=12, selector=selector, random_state=0).fit(X, y)
    assert model.transform(X).shape == (18, 5)
    if selector != "random":
        assert model.selector_["strategy"] == "topk"


def test_paper_equations():
    rng = np.random.default_rng(8)
    F = rng.normal(size=(6, 9))
    y = rng.normal(size=6)
    np.testing.assert_allclose(distribution_variance_scores(F), 9 * ((F - F.mean(0)) ** 2).sum(0))
    np.testing.assert_allclose(label_alignment_scores(F, y, "regression"), 9 * (y @ F) ** 2)
    expected = 9 * np.sum(F * np.linalg.solve(F @ F.T + 0.2 * np.eye(6), F), axis=0)
    np.testing.assert_allclose(empirical_leverage_scores(F, ridge=0.2), expected)
    X, y = toy()
    m = ARaDRRegressor(n_features=4, pool_size=12, random_state=1).fit(X, y)
    raw = np.vstack([m.random_features_.transform(b).mean(0) for b in X])
    np.testing.assert_allclose(m.train_embeddings_, raw[:, m.feature_indices_] * np.sqrt(3))
    expected_coef = np.linalg.solve(
        m.train_embeddings_.T @ m.train_embeddings_ + m.alpha * np.eye(4), m.train_embeddings_.T @ y
    )
    np.testing.assert_allclose(m.downstream_estimator_.coef_, expected_coef, atol=1e-10)


def test_cos_sin_keeps_both_halves():
    X, y = toy()
    m = RaDRRegressor(n_features=7, feature_map="cos_sin").fit(X, y)
    assert m.transform(X).shape == (18, 14)
    with pytest.raises(ValueError, match="cos_sin"):
        ARaDRRegressor(feature_map="cos_sin").fit(X, y)


def test_file_only_plugin(tmp_path, monkeypatch):
    import radr.scores

    (tmp_path / "new_score.py").write_text(
        "import numpy as np\ndef score(F, y, task):\n    return np.arange(F.shape[1], dtype=float)\n"
    )
    monkeypatch.setattr(radr.scores, "__path__", [*radr.scores.__path__, str(tmp_path)])
    importlib.invalidate_caches()
    assert "new_score" in available_selectors()
    X, y = toy()
    m = ARaDRClassifier(n_features=3, pool_size=8, selector="new_score").fit(X, y)
    assert m.feature_indices_.tolist() == [7, 6, 5]
    assert callable(resolve_selector("radr.scores.new_score:score"))


@pytest.mark.parametrize(
    "bad", [lambda F, y, task: np.zeros(1), lambda F, y, task: np.full(F.shape[1], np.nan)]
)
def test_bad_plugin(bad):
    X, y = toy()
    with pytest.raises(ValueError, match="one finite score"):
        ARaDRClassifier(n_features=3, selector=bad).fit(X, y)


def test_cv_refits_selector_per_fold():
    X, y = toy()
    calls = []

    def score(F, y, task):
        calls.append(len(y))
        return np.var(F, axis=0)

    m = ARaDRClassifier(n_features=4, selector=score, tune=True, cv=3, C_grid=(0.1, 1), random_state=0).fit(
        X, y
    )
    assert calls.count(12) == 6
    assert calls.count(18) == 1
    assert "C" in m.best_params_


@pytest.mark.parametrize("cls", [ARaDRClassifier, ARaDRRegressor])
def test_kernel_tuning(cls):
    X, y = toy()
    m = cls(
        n_features=4,
        downstream="kernel",
        tune=True,
        cv=2,
        C_grid=(1,),
        alpha_grid=(0.1,),
        kernel_gamma_grid=(0.5, 2),
        random_state=0,
    ).fit(X, y)
    assert "kernel_gamma_scale" in m.best_params_
    assert np.isfinite(m.predict(X)).all()


@pytest.mark.parametrize(
    "X",
    [
        [],
        np.empty((0, 2, 3)),
        np.empty((2, 0, 3)),
        [np.ones((2, 0))],
        [np.array([[np.nan]])],
        [np.array([[np.inf]])],
        [np.ones((2, 2)), np.ones((2, 3))],
    ],
)
def test_invalid_bags(X):
    with pytest.raises(ValueError):
        ensure_bag_list(X)


@pytest.mark.parametrize(
    "kwargs",
    [{"n_features": 0}, {"n_features": 2.5}, {"rf_gamma": -1}, {"pool_size": 1}, {"pool_multiplier": 0}],
)
def test_invalid_parameters(kwargs):
    X, y = toy()
    with pytest.raises(ValueError):
        ARaDRClassifier(**kwargs).fit(X, y)


def frame():
    return pd.DataFrame(
        {
            "id": ["a", "a", "b", "b"],
            "x": [0.0, 2.0, 4.0, 6.0],
            "cat": ["red", "blue", "red", "blue"],
            "target": ["no", "no", "yes", "yes"],
        }
    )


@pytest.mark.parametrize("encoding, width", [("onehot", 3), ("ordinal", 2)])
def test_categorical_training_vocabulary(encoding, width):
    df = frame()
    prep = BagPreprocessor(bag_id_col="id", target_col="target", categorical_encoding=encoding, scaler=None)
    X, y, meta = prep.fit_transform(df, return_metadata=True)
    assert X[0].shape == (2, width)
    assert y.tolist() == ["no", "yes"]
    assert meta["bag_ids"] == ["a", "b"]
    new = df.drop(columns="target").copy()
    new["cat"] = "unseen"
    result = prep.transform(new)
    assert result[0].shape == (2, width)
    np.testing.assert_allclose(result[0][:, 1:], 0 if encoding == "onehot" else -1)
    assert len(prep.get_feature_names_out()) == width
    strict = clone(prep).set_params(handle_unknown="error").fit(df)
    with pytest.raises(ValueError):
        strict.transform(new)


def test_numeric_conversion_and_training_only_scaler():
    df = frame().assign(cat=["1", "2", "3", "4"])
    prep = BagPreprocessor(bag_id_col="id", target_col="target", categorical_encoding="numeric").fit(df)
    means = prep.scaler_.mean_.copy()
    prep.transform(df.assign(x=10000, cat="50"))
    np.testing.assert_array_equal(prep.scaler_.mean_, means)
    with pytest.raises(ValueError):
        prep.transform(df.assign(cat="not a number"))


@pytest.mark.parametrize("missing", ["fill", "drop", "error"])
def test_missing_values(missing):
    df = frame()
    df.loc[0, "cat"] = None
    prep = BagPreprocessor(bag_id_col="id", target_col="target", missing=missing)
    if missing == "error":
        with pytest.raises(ValueError):
            prep.fit(df)
    else:
        X, y = prep.fit_transform(df)
        assert len(X[0]) == (1 if missing == "drop" else 2)
        assert np.isfinite(np.concatenate(X)).all()


def test_array_missing_and_not_fitted():
    with pytest.raises(NotFittedError):
        BagPreprocessor().transform([np.ones((2, 1))])
    bags = [np.array([[np.nan], [1.0]]), np.array([[2.0]])]
    X, y = BagPreprocessor(missing="fill").fit_transform(bags, [0, 1])
    assert np.isfinite(np.concatenate(X)).all()


def test_safe_archive(tmp_path):
    X, y = toy()
    path = tmp_path / "bags.npz"
    ids = list(range(len(X)))
    save_npz(path, X, y, bag_ids=ids, metadata={"ok": True})
    ds = load_npz(path)
    np.testing.assert_allclose(ds.bags[0], X[0])
    assert ds.bag_ids == ids
    with np.load(path, allow_pickle=False) as archive:
        for key in archive:
            assert archive[key].dtype != object
    save_npz(path, X)
    assert load_npz(path).targets is None


def test_string_label_metrics():
    X, y = toy()
    y = np.where(y, "positive", "negative")
    assert "f1" in RaDRClassifier(n_features=8).fit(X, y).test(X, y)


def test_cli_split_before_fit(tmp_path):
    import joblib
    from radr.cli import main

    df = pd.DataFrame(
        {
            "id": np.repeat(np.arange(10), 2),
            "x": np.repeat(np.arange(10) ** 3, 2),
            "cat": np.repeat([f"category{i}" for i in range(10)], 2),
            "y": np.repeat(np.arange(10) % 2, 2),
        }
    )
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    out = tmp_path / "out"
    main(
        [
            "--input-csv",
            str(path),
            "--output-dir",
            str(out),
            "--task",
            "classification",
            "--bag-id-col",
            "id",
            "--target-col",
            "y",
            "--stratify",
        ]
    )
    train, test = load_npz(out / "train_radr.npz"), load_npz(out / "test_radr.npz")
    assert not set(train.bag_ids) & set(test.bag_ids)
    prep = joblib.load(out / "preprocessor.joblib")
    assert len(prep.encoder_.categories_[0]) == len(train)
    assert prep.scaler_.mean_[0] == df.loc[df.id.isin(train.bag_ids), "x"].mean()


def test_custom_downstream_and_sampler():
    from sklearn.linear_model import LogisticRegression

    X, y = toy()

    def sampler(input_dim, n_features, gamma, rng):
        return rng.normal(size=(input_dim, n_features)) * np.sqrt(gamma)

    m = RaDRClassifier(
        n_features=5, downstream=LogisticRegression(), random_feature_distribution=sampler, random_state=0
    ).fit(X, y)
    assert m.predict_proba(X).shape == (len(X), 2)
    np.testing.assert_allclose(m.predict_proba(X).sum(1), 1)


def test_numeric_categorical_override_and_schema():
    df = frame().assign(cat=[10, 20, 10, 20])
    prep = BagPreprocessor(bag_id_col="id", target_col="target", categorical_cols=["cat"], scaler=None)
    X, _ = prep.fit_transform(df)
    assert X[0].shape[1] == 3
    with pytest.raises(ValueError, match="Missing feature"):
        prep.transform(df.drop(columns="cat"))
    np.testing.assert_allclose(prep.transform(df.assign(extra=123))[0][0], X[0])


def test_sampling_edge_cases():
    from radr.selectors import select_feature_indices

    keep, weights = select_feature_indices(np.zeros(5), 3, strategy="sample", random_state=0)
    assert len(set(keep)) == 3 and weights is None
    with pytest.raises(ValueError, match="Too few positive"):
        select_feature_indices(np.array([1.0, 0.0, 0.0]), 2, strategy="sample")
    with pytest.raises(ValueError, match="nonnegative"):
        select_feature_indices(np.array([-1.0, 1.0]), 1, strategy="sample")
