import numpy as np

from radr import ARaDRClassifier


def main():
    rng = np.random.default_rng(0)
    X = []
    y = []
    for i in range(120):
        label = i % 2
        X.append(rng.normal(loc=0.7 * label, scale=1.0, size=(16 + i % 7, 5)))
        y.append(label)
    y = np.asarray(y)

    clf = ARaDRClassifier(
        n_features=64,
        pool_multiplier=4,
        selector="dv",
        downstream="kernel",
        random_state=0,
    )
    clf.fit(X[:90], y[:90])
    print(clf.test(X[90:], y[90:]))
    print(clf.params_)


if __name__ == "__main__":
    main()
